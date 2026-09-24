"""Native SDK clients: plain SDK code, with token_daddy underneath."""

import base64
import json

import httpx
import httpx2

from tests.conftest import make_registry
from tests.test_adapters import ANTHROPIC_OK, GENERATE_OK, OPENAI_OK
from tests.test_estimate import fake_pdf
from token_daddy import TokenDaddy
from token_daddy.estimate import estimate_body_input_tokens
from token_daddy.limiter import InProcessBackend
from token_daddy.native import anthropic_client, genai_client, openai_client
from token_daddy.settings import EstimationConfig, Family

ENV = {"KEY_A": "a", "KEY_B": "dbc.example.com", "KEY_C": "dapi"}


class Upstream:
    """Mock network: responds per host, records what arrived."""

    def __init__(self, lib, by_host: dict):
        self.lib, self.by_host, self.seen = lib, by_host, []

    def __call__(self, request):
        self.seen.append(request)
        status, body = self.by_host[request.url.host]
        return self.lib.Response(status, json=body, headers={"retry-after": "12"})

    @property
    def transport(self):
        return self.lib.MockTransport(self)


def td_with(accounts, model_cfg):
    registry = make_registry(accounts=accounts, models={"m": model_cfg}, limits={"rps": 100})
    return TokenDaddy(registry.config, backend=InProcessBackend(), environ=ENV, project="p")


async def test_openai_sdk_fails_over_from_a_429_to_bedrock():
    td = td_with(
        {"a": {"provider": "openai", "api_key_env": "KEY_A"},
         "b": {"provider": "bedrock", "region": "us-east-1", "api_key_env": "KEY_A"}},
        {"family": "gpt", "max_output_tokens": 1234, "deployments": [
            {"account": "a", "model_id": "gpt-5.6-sol"},
            {"account": "b", "model_id": "global.openai.gpt-5.6-sol",
             "extra_params": {"store": False}},
        ]},
    )
    up = Upstream(httpx2, {
        "api.openai.com": (429, {"error": {"message": "slow down"}}),
        "bedrock-runtime.us-east-1.amazonaws.com": (200, OPENAI_OK),
    })
    client = openai_client(td, "m", upstream=up.transport)

    response = await client.responses.create(model="m", input="hi")

    assert response.output_text == '{"city": "Paris"}'
    first, second = up.seen
    assert json.loads(first.content)["model"] == "gpt-5.6-sol"
    assert str(second.url) == "https://bedrock-runtime.us-east-1.amazonaws.com/openai/v1/responses"
    sent = json.loads(second.content)
    assert sent["model"] == "global.openai.gpt-5.6-sol"
    assert sent["store"] is False
    assert sent["max_output_tokens"] == 1234  # our cap injected, so the reservation holds
    assert second.headers["authorization"] == "Bearer a"

    rows = {r["deployment"]: r for r in await td.usage()}
    assert rows["m@a"]["errors"] == 1
    assert rows["m@b"]["input_tokens"] == 120
    # The 429'd route is benched for its retry-after.
    hold, result = await td.limiter.acquire([td.registry.deployments["m@a"]],
                                            [td_cost()])
    assert hold is None and result.reasons[0] == "cooldown"


def td_cost():
    from token_daddy.limiter import Cost

    return Cost(1, 1)


async def test_anthropic_sdk_goes_to_databricks_with_bearer_auth():
    td = td_with(
        {"d": {"provider": "databricks", "host_env": "KEY_B", "token_env": "KEY_C"}},
        {"family": "claude", "deployments": [
            {"account": "d", "model_id": "databricks-claude-opus-5-5"}]},
    )
    up = Upstream(httpx2, {"dbc.example.com": (200, ANTHROPIC_OK)})
    client = anthropic_client(td, "m", upstream=up.transport)

    msg = await client.messages.create(
        model="m", max_tokens=500, messages=[{"role": "user", "content": "hi"}]
    )

    assert msg.content[0].text == '{"city": "Paris"}'
    (req,) = up.seen
    assert str(req.url) == "https://dbc.example.com/ai-gateway/anthropic/v1/messages"
    assert req.headers["authorization"] == "Bearer dapi"
    assert "x-api-key" not in req.headers
    assert json.loads(req.content)["model"] == "databricks-claude-opus-5-5"
    assert (await td.usage())[0]["input_tokens"] == 500


async def test_genai_sdk_generate_content_routes_by_model_in_the_url():
    td = td_with(
        {"d": {"provider": "databricks", "host_env": "KEY_B", "token_env": "KEY_C"}},
        {"family": "gemini", "deployments": [
            {"account": "d", "model_id": "system.ai.gemini-3-8-flash"}]},
    )
    up = Upstream(httpx, {"dbc.example.com": (200, GENERATE_OK)})
    client = genai_client(td, "m", upstream=up.transport)

    response = await client.aio.models.generate_content(model="m", contents="hi")

    assert response.text == '{"city": "Paris"}'
    (req,) = up.seen
    assert str(req.url) == (
        "https://dbc.example.com/ai-gateway/gemini/v1beta/models/"
        "system.ai.gemini-3-8-flash:generateContent"
    )
    assert req.headers["authorization"] == "Bearer dapi"
    assert "x-goog-api-key" not in req.headers
    assert (await td.usage())[0]["output_tokens"] == 100


async def test_a_client_error_is_returned_to_the_sdk_not_failed_over():
    td = td_with(
        {"a": {"provider": "openai", "api_key_env": "KEY_A"},
         "b": {"provider": "bedrock", "region": "us-east-1", "api_key_env": "KEY_A"}},
        {"family": "gpt", "deployments": [
            {"account": "a", "model_id": "x"}, {"account": "b", "model_id": "y"}]},
    )
    up = Upstream(httpx2, {"api.openai.com": (400, {"error": {"message": "bad"}})})
    client = openai_client(td, "m", upstream=up.transport)
    import openai
    import pytest

    with pytest.raises(openai.BadRequestError):
        await client.responses.create(model="m", input="hi")
    assert len(up.seen) == 1


def test_inline_base64_files_are_priced_as_files_not_text():
    pdf = fake_pdf(3) + b" " * 500_000  # a big 3-page PDF
    body = {"input": [{"role": "user", "content": [
        {"type": "input_file", "file_data": "data:application/pdf;base64,"
         + base64.b64encode(pdf).decode()},
        {"type": "input_text", "text": "a" * 350},
    ]}]}
    cfg = EstimationConfig(safety_multiplier=1.0)
    assert estimate_body_input_tokens(body, Family.GPT, cfg) == 3 * 1500 + 100
