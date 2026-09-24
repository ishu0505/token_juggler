"""Routing behaviour with fake adapters: which deployment runs, what happens
when one fails, and whether usage and cost land in the tracker."""

import asyncio

import pytest
from pydantic import BaseModel

from tests.conftest import make_registry
from token_daddy import TokenDaddy
from token_daddy.adapters import (
    AdapterResult,
    BadRequest,
    DeploymentError,
    RateLimited,
    Transient,
)
from token_daddy.limiter import InProcessBackend
from token_daddy.router import AllRoutesFailed, NoCapableDeployment, QuotaExceeded
from token_daddy.settings import WireApi
from token_daddy.types import File, Usage


class FakeAdapter:
    """Scripted per model_id: a list of outcomes consumed in order; the last
    one repeats. An outcome is an exception class or 'ok'."""

    def __init__(self, script=None, usage=Usage(100, 20)):
        self.script = script or {}
        self.usage = usage
        self.calls: list[str] = []

    async def call(self, dep, request):
        self.calls.append(dep.model_id)
        steps = self.script.get(dep.model_id, ["ok"])
        step = steps.pop(0) if len(steps) > 1 else steps[0]
        if step != "ok":
            if step is RateLimited:
                raise RateLimited("429", retry_after=30)
            raise step(f"{step.__name__} from {dep.model_id}")
        return AdapterResult(text='{"city": "Paris"}', usage=self.usage)

    async def close(self):
        pass


def make_td(fake, *, limits=None, models=None, backend=None, **kw):
    registry = make_registry(limits=limits or {"rps": 100}, models=models, **kw)
    adapters = {api: fake for api in WireApi}
    return TokenDaddy(registry.config, backend=backend or InProcessBackend(), adapters=adapters,
                      environ={"KEY_A": "a", "KEY_B": "b", "KEY_C": "c"}, project="proj")


async def test_the_first_deployment_serves_when_healthy():
    fake = FakeAdapter()
    td = make_td(fake)
    r = await td.generate("m", "hi")
    assert r.deployment == "m@a"
    assert [a.outcome for a in r.attempts] == ["ok"]


@pytest.mark.parametrize("failure", [RateLimited, Transient, DeploymentError])
async def test_a_failing_deployment_fails_over_and_is_not_retried(failure):
    fake = FakeAdapter({"m-a": [failure]})
    td = make_td(fake)
    r = await td.generate("m", "hi")
    assert r.deployment == "m@b"
    assert fake.calls == ["m-a", "m-b"]  # m-a tried exactly once


async def test_a_bad_request_is_raised_not_failed_over():
    fake = FakeAdapter({"m-a": [BadRequest]})
    td = make_td(fake)
    with pytest.raises(BadRequest):
        await td.generate("m", "hi")
    assert fake.calls == ["m-a"]


async def test_a_429_benches_the_route_for_later_requests_then_it_comes_back():
    now = [1_000_000.0]
    fake = FakeAdapter({"m-a": [RateLimited, "ok"]})
    td = make_td(fake)
    td.limiter.backend = td.tracker._backend = InProcessBackend(clock=lambda: now[0])
    first = await td.generate("m", "hi")
    assert first.deployment == "m@b"
    second = await td.generate("m", "hi")
    assert second.deployment == "m@b"  # m@a is cooling down (retry-after 30s)
    now[0] += 31_000
    third = await td.generate("m", "hi")
    assert third.deployment == "m@a"  # priority routing switched back on its own


async def test_full_primary_spills_to_secondary_then_to_fallback_model():
    models = {
        "m": {"family": "gpt", "deployments": [{"account": "a", "model_id": "m-a"}],
              "routing": {"fallbacks": ["f"]}},
        "f": {"family": "gpt", "deployments": [{"account": "b", "model_id": "f-b"}]},
    }
    fake = FakeAdapter()
    td = make_td(fake, limits={"rps": 2}, models=models)
    served = [(await td.generate("m", "hi", max_wait_seconds=0)).deployment for _ in range(4)]
    assert served == ["m@a", "m@a", "f@b", "f@b"]
    with pytest.raises(QuotaExceeded):
        await td.generate("m", "hi", max_wait_seconds=0)


async def test_when_everything_is_full_the_call_waits_for_capacity():
    fake = FakeAdapter()
    td = make_td(fake, limits={"rps": 10}, deployments=[{"account": "a", "model_id": "m-a"}])
    for _ in range(10):
        await td.generate("m", "hi")
    loop = asyncio.get_running_loop()
    start = loop.time()
    r = await td.generate("m", "hi", max_wait_seconds=5)
    assert r.deployment == "m@a"
    assert 0.05 < loop.time() - start < 1.0  # one unit refills in 100ms at 10 rps


async def test_capability_filter_skips_routes_that_cannot_take_the_payload():
    models = {"m": {
        "family": "gemini",
        "capabilities": ["text", "audio"],
        "deployments": [
            {"account": "a", "model_id": "m-a", "exclude_capabilities": ["audio"],
             "api": "genai_generate_content"},
            {"account": "b", "model_id": "m-b", "api": "genai_generate_content"},
        ],
    }}
    fake = FakeAdapter()
    td = make_td(fake, models=models)
    r = await td.generate("m", [File(b"RIFF", "audio/wav")])
    assert r.deployment == "m@b"


async def test_no_capable_route_is_an_immediate_clear_error():
    fake = FakeAdapter()
    td = make_td(fake)  # gpt-style model with default caps: no audio
    with pytest.raises(NoCapableDeployment):
        await td.generate("m", [File(b"RIFF", "audio/wav")])
    assert fake.calls == []


async def test_all_routes_failing_raises_with_every_attempt_listed():
    fake = FakeAdapter({"m-a": [Transient], "m-b": [DeploymentError]})
    td = make_td(fake)
    with pytest.raises(AllRoutesFailed) as info:
        await td.generate("m", "hi")
    assert [a.outcome for a in info.value.attempts] == ["transient", "deployment_error"]


async def test_opt_in_retry_tries_the_same_route_again():
    registry = make_registry(limits={"rps": 100})
    raw = registry.config.model_dump()
    raw["defaults"]["retry"] = {"enabled": True, "max_attempts": 2, "base_delay_seconds": 0.01}
    from token_daddy.settings import Config

    fake = FakeAdapter({"m-a": [Transient, "ok"]})
    td = TokenDaddy(Config.model_validate(raw), backend=InProcessBackend(),
                    adapters={api: fake for api in WireApi},
                    environ={"KEY_A": "a", "KEY_B": "b"})
    r = await td.generate("m", "hi")
    assert r.deployment == "m@a"
    assert fake.calls == ["m-a", "m-a"]


class City(BaseModel):
    city: str


async def test_usage_and_cost_are_tracked_per_project_and_deployment(backend):
    models = {"m": {
        "family": "claude",
        "price": {"input_per_mtok": 2.0, "output_per_mtok": 10.0},
        "deployments": [
            {"account": "a", "model_id": "m-a", "api": "anthropic_messages"},
            {"account": "b", "model_id": "m-b", "api": "anthropic_messages"},
        ],
    }}
    records = []
    fake = FakeAdapter({"m-a": [RateLimited]}, usage=Usage(1000, 200))
    td = make_td(fake, models=models, backend=backend)
    td.router.on_call = records.append
    r = await td.generate("m", "hi", response_schema=City)
    assert r.parsed == City(city="Paris")
    assert r.cost_usd == pytest.approx((1000 * 2 + 200 * 10) / 1e6)

    rows = {row["deployment"]: row for row in await td.usage()}
    assert rows["m@b"]["requests"] == 1 and rows["m@b"]["errors"] == 0
    assert rows["m@b"]["input_tokens"] == 1000
    assert rows["m@b"]["cost_usd"] == pytest.approx(0.004)
    assert rows["m@a"]["errors"] == 1
    assert [rec.outcome for rec in records] == ["rate_limited", "ok"]
    assert (await td.recent_calls(1))[0]["deployment"] == "m@b"
