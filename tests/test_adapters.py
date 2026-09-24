"""Adapters driven through the REAL provider SDKs against mocked HTTP, so the
test covers what actually goes over the wire, not just our dict-building."""

import json

import httpx
import httpx2
import pytest
from pydantic import BaseModel

from tests.conftest import make_registry
from token_daddy.adapters import (
    BadRequest,
    DeploymentError,
    RateLimited,
    Transient,
    UnsupportedInput,
    anthropic_messages,
    genai,
    openai_responses,
)
from token_daddy.types import File, Request, Text

PDF = File(b"%PDF-1.7 << /Type /Page >>", "application/pdf", "a.pdf")
PNG = File(b"\x89PNG....", "image/png", "a.png")
WAV = File(b"RIFF....WAVE", "audio/wav", "a.wav")


class Answer(BaseModel):
    city: str
    confidence: float = 0.5


def registry_for(provider_accounts: dict, family: str, model_ids: dict):
    env_accounts = {}
    for name, cfg in provider_accounts.items():
        env_accounts[name] = cfg
    return make_registry(
        accounts=env_accounts,
        models={
            "m": {
                "family": family,
                "capabilities": ["text", "image", "pdf", "audio", "json_schema", "reasoning"],
                "deployments": [
                    {"account": acct, "model_id": mid, **extra}
                    for acct, (mid, extra) in model_ids.items()
                ],
            }
        },
    )


class Recorder:
    """A mock transport handler that records requests and replays a response."""

    def __init__(self, lib, status=200, body=None, headers=None):
        self.lib, self.status, self.body, self.headers = lib, status, body or {}, headers or {}
        self.requests = []

    def __call__(self, request):
        self.requests.append(request)
        return self.lib.Response(self.status, json=self.body, headers=self.headers)

    @property
    def last_json(self):
        return json.loads(self.requests[-1].content)


# -- OpenAI Responses ----------------------------------------------------------

OPENAI_OK = {
    "id": "resp_1", "object": "response", "created_at": 0, "status": "completed",
    "model": "x", "parallel_tool_calls": True, "tool_choice": "auto", "tools": [],
    "output": [{
        "type": "message", "id": "m1", "status": "completed", "role": "assistant",
        "content": [{"type": "output_text", "text": '{"city": "Paris"}', "annotations": []}],
    }],
    "usage": {
        "input_tokens": 120, "output_tokens": 40, "total_tokens": 160,
        "input_tokens_details": {"cached_tokens": 20},
        "output_tokens_details": {"reasoning_tokens": 30},
    },
}


def openai_setup(recorder):
    registry = registry_for(
        {"a": {"provider": "bedrock", "region": "us-east-1", "api_key_env": "KEY_A"}},
        "gpt", {"a": ("global.openai.gpt-5.6-sol", {"extra_params": {"store": False}})},
    )
    dep = registry.deployments["m@a"]
    adapter = openai_responses.OpenAIResponsesAdapter(timeout_seconds=5)
    adapter._clients[dep.id] = openai_responses.build_client(
        dep, 5, http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(recorder))
    )
    return adapter, dep


async def test_openai_request_goes_to_bedrock_in_responses_shape():
    rec = Recorder(httpx2, body=OPENAI_OK)
    adapter, dep = openai_setup(rec)
    req = Request(model="m", parts=[PDF, PNG, Text("Where?")], system="be brief",
                  response_schema=Answer, thinking="low", max_output_tokens=900)
    result = await adapter.call(dep, req)

    sent = rec.requests[-1]
    assert str(sent.url) == "https://bedrock-runtime.us-east-1.amazonaws.com/openai/v1/responses"
    assert sent.headers["authorization"] == "Bearer a"
    body = rec.last_json
    assert body["model"] == "global.openai.gpt-5.6-sol"
    assert body["store"] is False
    assert body["max_output_tokens"] == 900
    assert body["instructions"] == "be brief"
    assert body["reasoning"] == {"effort": "low"}
    assert [c["type"] for c in body["input"][0]["content"]] == ["input_file", "input_image", "input_text"]
    fmt = body["text"]["format"]
    assert fmt["strict"] is True and fmt["name"] == "Answer"
    assert fmt["schema"]["required"] == ["city", "confidence"]  # strict: all required

    assert result.text == '{"city": "Paris"}'
    assert result.usage.input_tokens == 120
    assert result.usage.output_tokens == 40
    assert result.usage.reasoning_tokens == 30
    assert result.usage.cached_input_tokens == 20


@pytest.mark.parametrize("status,expected", [
    (429, RateLimited), (503, Transient), (400, BadRequest), (404, DeploymentError),
    (401, DeploymentError),
])
async def test_openai_errors_are_classified(status, expected):
    rec = Recorder(httpx2, status=status, body={"error": {"message": "x"}},
                   headers={"retry-after": "7"})
    adapter, dep = openai_setup(rec)
    with pytest.raises(expected) as info:
        await adapter.call(dep, Request(model="m", parts=[Text("hi")]))
    if expected is RateLimited:
        assert info.value.retry_after == 7


async def test_openai_refuses_audio_before_sending():
    rec = Recorder(httpx2, body=OPENAI_OK)
    adapter, dep = openai_setup(rec)
    with pytest.raises(UnsupportedInput) as info:
        await adapter.call(dep, Request(model="m", parts=[WAV]))
    assert info.value.sent is False
    assert rec.requests == []


# -- Anthropic Messages --------------------------------------------------------

ANTHROPIC_OK = {
    "id": "msg_1", "type": "message", "role": "assistant", "model": "x",
    "content": [{"type": "text", "text": '{"city": "Paris"}'}],
    "stop_reason": "end_turn", "stop_sequence": None,
    "usage": {"input_tokens": 100, "output_tokens": 50,
              "cache_read_input_tokens": 400, "cache_creation_input_tokens": 0},
}


def anthropic_setup(recorder, model_id="claude-opus-5-5", provider="databricks"):
    accounts = {
        "databricks": {"a": {"provider": "databricks", "host_env": "KEY_B", "token_env": "KEY_C"}},
        "anthropic": {"a": {"provider": "anthropic", "api_key_env": "KEY_A"}},
    }[provider]
    registry = registry_for(accounts, "claude", {"a": (model_id, {})})
    dep = registry.deployments["m@a"]
    adapter = anthropic_messages.AnthropicMessagesAdapter(timeout_seconds=5)
    adapter._clients[dep.id] = anthropic_messages.build_client(
        dep, 5, http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(recorder))
    )
    return adapter, dep


async def test_anthropic_via_databricks_uses_bearer_and_output_config():
    rec = Recorder(httpx2, body=ANTHROPIC_OK)
    adapter, dep = anthropic_setup(rec)
    req = Request(model="m", parts=[PDF, Text("Where?")], response_schema=Answer,
                  thinking="medium", max_output_tokens=2000)
    result = await adapter.call(dep, req)

    sent = rec.requests[-1]
    assert str(sent.url) == "https://b/ai-gateway/anthropic/v1/messages"
    assert sent.headers["authorization"] == "Bearer c"
    body = rec.last_json
    assert body["model"] == "claude-opus-5-5"
    assert body["max_tokens"] == 2000
    assert body["output_config"]["effort"] == "medium"
    assert body["output_config"]["format"]["type"] == "json_schema"
    assert "thinking" not in body and "tool_choice" not in body
    assert body["messages"][0]["content"][0]["type"] == "document"

    assert result.usage.input_tokens == 500  # uncached + cache reads
    assert result.usage.cached_input_tokens == 400
    assert result.usage.output_tokens == 50


async def test_haiku_gets_a_thinking_budget_not_effort():
    rec = Recorder(httpx2, body=ANTHROPIC_OK)
    adapter, dep = anthropic_setup(rec, model_id="claude-haiku-4-5", provider="anthropic")
    await adapter.call(dep, Request(model="m", parts=[Text("hi")], thinking="low",
                                    max_output_tokens=8000))
    body = rec.last_json
    assert body["thinking"] == {"type": "enabled", "budget_tokens": 2048}
    assert "output_config" not in body
    assert rec.requests[-1].headers["x-api-key"] == "a"


async def test_anthropic_529_overloaded_is_transient():
    rec = Recorder(httpx2, status=529, body={"type": "error", "error": {"type": "overloaded_error"}})
    adapter, dep = anthropic_setup(rec)
    with pytest.raises(Transient):
        await adapter.call(dep, Request(model="m", parts=[Text("hi")]))


# -- Gemini ----------------------------------------------------------------------

GENERATE_OK = {
    "candidates": [{"content": {"role": "model", "parts": [{"text": '{"city": "Paris"}'}]},
                    "finishReason": "STOP"}],
    "usageMetadata": {"promptTokenCount": 300, "candidatesTokenCount": 20,
                      "thoughtsTokenCount": 80, "cachedContentTokenCount": 0},
}


def genai_setup(recorder, provider, model_id):
    accounts = {
        "databricks": {"a": {"provider": "databricks", "host_env": "KEY_B", "token_env": "KEY_C"}},
        "google_ai_studio": {"a": {"provider": "google_ai_studio", "api_key_env": "KEY_A"}},
    }[provider]
    registry = registry_for(accounts, "gemini", {"a": (model_id, {})})
    dep = registry.deployments["m@a"]
    adapter = genai.GenAIAdapter(timeout_seconds=5)
    adapter._clients[dep.id] = genai.build_client(
        dep, 5, {"httpx_async_client": httpx.AsyncClient(transport=httpx.MockTransport(recorder))}
    )
    return adapter, dep


async def test_gemini_generate_content_through_databricks():
    rec = Recorder(httpx, body=GENERATE_OK)
    adapter, dep = genai_setup(rec, "databricks", "system.ai.gemini-3-8-flash")
    req = Request(model="m", parts=[WAV, Text("Transcribe")], response_schema=Answer,
                  thinking="low", max_output_tokens=1000)
    result = await adapter.call(dep, req)

    sent = rec.requests[-1]
    assert str(sent.url).startswith("https://b/ai-gateway/gemini/")
    assert str(sent.url).endswith("system.ai.gemini-3-8-flash:generateContent")
    assert sent.headers["authorization"] == "Bearer c"
    body = rec.last_json
    assert body["generationConfig"]["maxOutputTokens"] == 1000
    assert body["generationConfig"]["responseMimeType"] == "application/json"
    # The SDK itself emits snake_case here; the API accepts both spellings.
    thinking = body["generationConfig"]["thinkingConfig"]
    assert (thinking.get("thinkingLevel") or thinking.get("thinking_level")) == "LOW"
    inline = body["contents"][0]["parts"][0]["inlineData"]
    assert (inline.get("mimeType") or inline.get("mime_type")) == "audio/wav"

    assert result.usage.input_tokens == 300
    assert result.usage.output_tokens == 100  # answer + thinking
    assert result.usage.reasoning_tokens == 80


async def test_gemini_rate_limit_is_classified():
    rec = Recorder(httpx, status=429, body={"error": {"code": 429, "message": "slow",
                                                      "status": "RESOURCE_EXHAUSTED"}})
    adapter, dep = genai_setup(rec, "databricks", "system.ai.gemini-3-8-flash")
    with pytest.raises(RateLimited):
        await adapter.call(dep, Request(model="m", parts=[Text("hi")]))


INTERACTION_OK = {
    "id": "int_1", "status": "completed", "model": "gemini-3.8-flash",
    "steps": [{"type": "model_output", "content": [{"type": "text", "text": '{"city": "Paris"}'}]}],
    "usage": {"total_input_tokens": 50, "total_output_tokens": 10,
              "total_thought_tokens": 5, "total_cached_tokens": 0, "total_tokens": 65},
}


async def test_gemini_interactions_on_ai_studio():
    rec = Recorder(httpx, body=INTERACTION_OK)
    adapter, dep = genai_setup(rec, "google_ai_studio", "gemini-3.8-flash")
    req = Request(model="m", parts=[PDF, Text("Where?")], response_schema=Answer,
                  thinking="high", system="terse")
    result = await adapter.call(dep, req)

    sent = rec.requests[-1]
    assert str(sent.url).endswith("/interactions")
    body = rec.last_json
    assert body["model"] == "gemini-3.8-flash"
    assert body["store"] is False
    assert body["system_instruction"] == "terse"
    assert body["generation_config"]["thinking_level"] == "high"
    # The SDK wraps a content list in a single user_input step.
    assert [i["type"] for i in body["input"][0]["content"]] == ["document", "text"]
    assert body["response_format"]["schema"]["properties"]["city"]["type"] == "string"

    assert result.usage.input_tokens == 50
    assert result.usage.output_tokens == 15
