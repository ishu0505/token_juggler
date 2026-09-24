"""The quota guarantee. Every test runs against both backends, so the Lua and
the Python implementations of the bucket arithmetic cannot drift apart."""

import asyncio

from tests.conftest import make_registry
from token_daddy.limiter import Cost, InProcessBackend, Limiter
from token_daddy.settings import Strategy

SMALL = Cost(input_tokens=10, output_tokens=10)


def deps(registry, model="m"):
    return list(registry.model(model).deployments)


async def take(limiter, deployments, cost=SMALL, **kw):
    hold, result = await limiter.acquire(deployments, [cost] * len(deployments), **kw)
    return hold, result


async def test_requests_stop_exactly_at_the_limit(backend):
    registry = make_registry(limits={"rps": 5})
    limiter = Limiter(registry, backend)
    only_a = deps(registry)[:1]
    for _ in range(5):
        hold, _ = await take(limiter, only_a)
        assert hold is not None
    hold, result = await take(limiter, only_a)
    assert hold is None
    assert 0 < result.wait_ms <= 1000
    assert "rps" in result.reasons[0]


async def test_a_full_deployment_fails_over_to_the_next(backend):
    registry = make_registry(limits={"rps": 2})
    limiter = Limiter(registry, backend)
    winners = []
    for _ in range(4):
        hold, _ = await take(limiter, deps(registry))
        winners.append(hold.deployment.account)
    assert winners == ["a", "a", "b", "b"]
    hold, _ = await take(limiter, deps(registry))
    assert hold is None


async def test_reservation_is_all_or_nothing_across_scopes(backend):
    """An account-wide bucket that is full must not leave the deployment's own
    bucket charged."""
    registry = make_registry(
        limits={"rps": 10},
        accounts={
            "a": {"provider": "openai", "api_key_env": "KEY_A", "limits": {"rps": 1}},
            "b": {"provider": "openai", "api_key_env": "KEY_B"},
        },
        deployments=[{"account": "a", "model_id": "x"}],
    )
    limiter = Limiter(registry, backend)
    assert (await take(limiter, deps(registry)))[0] is not None
    for _ in range(5):
        hold, result = await take(limiter, deps(registry))
        assert hold is None
        assert "account a rps" in result.reasons[0]
    # Deployment bucket saw exactly one request, so 9 remain once the account frees up.


async def test_a_request_bigger_than_the_whole_quota_never_waits(backend):
    registry = make_registry(limits={"output_tpm": 20_000})
    limiter = Limiter(registry, backend)
    hold, result = await take(limiter, deps(registry), cost=Cost(100, 30_000))
    assert hold is None
    assert result.wait_ms is None
    assert all(r.startswith("too_large") for r in result.reasons)


async def test_strict_reservation_is_refunded_on_settle(backend):
    registry = make_registry(limits={"output_tpm": 20_000})
    limiter = Limiter(registry, backend)
    only_a = deps(registry)[:1]
    first, _ = await take(limiter, only_a, cost=Cost(10, 10_000))
    second, _ = await take(limiter, only_a, cost=Cost(10, 10_000))
    assert first and second
    assert (await take(limiter, only_a, cost=Cost(10, 10_000)))[0] is None
    # The first call only produced 100 tokens: 9,900 come back.
    await limiter.settle(first, actual=Cost(10, 100))
    assert (await take(limiter, only_a, cost=Cost(10, 9_000)))[0] is not None


async def test_a_failed_call_keeps_its_request_but_returns_its_output(backend):
    registry = make_registry(limits={"rps": 1, "output_tpm": 20_000})
    limiter = Limiter(registry, backend)
    only_a = deps(registry)[:1]
    hold, _ = await take(limiter, only_a, cost=Cost(10, 20_000))
    await limiter.settle(hold, actual=None, sent=True)
    hold, result = await take(limiter, only_a, cost=Cost(10, 10))
    assert hold is None and "rps" in result.reasons[0]  # the request still counts


async def test_a_call_that_never_left_is_fully_refunded(backend):
    registry = make_registry(limits={"rps": 1})
    limiter = Limiter(registry, backend)
    only_a = deps(registry)[:1]
    hold, _ = await take(limiter, only_a)
    await limiter.settle(hold, actual=None, sent=False)
    assert (await take(limiter, only_a))[0] is not None


async def test_settle_is_idempotent(backend):
    registry = make_registry(limits={"rps": 1})
    limiter = Limiter(registry, backend)
    hold, _ = await take(limiter, deps(registry)[:1])
    await limiter.settle(hold, actual=None, sent=False)
    await limiter.settle(hold, actual=None, sent=False)  # must not refund twice
    assert (await take(limiter, deps(registry)[:1]))[0] is not None
    assert (await take(limiter, deps(registry)[:1]))[0] is None


async def test_project_share_caps_one_project_but_not_the_other(backend):
    registry = make_registry(limits={"rps": 4}, projects={"p1": {"share": 0.5}})
    p1 = Limiter(registry, backend, project="p1")
    other = Limiter(registry, backend, project="p2")
    only_a = deps(registry)[:1]
    assert (await take(p1, only_a))[0] and (await take(p1, only_a))[0]
    hold, result = await take(p1, only_a)
    assert hold is None and "project p1" in result.reasons[0]
    # p2 has no slice, so it can use what's left of the deployment's quota.
    assert (await take(other, only_a))[0] and (await take(other, only_a))[0]
    assert (await take(other, only_a))[0] is None


async def test_cooldown_benches_a_deployment(backend):
    registry = make_registry(limits={"rps": 100})
    limiter = Limiter(registry, backend)
    a, b = deps(registry)
    await limiter.cooldown(a, seconds=30)
    hold, _ = await take(limiter, [a, b])
    assert hold.deployment is b
    hold, result = await take(limiter, [a])
    assert hold is None and result.reasons[0] == "cooldown"
    assert result.wait_ms > 25_000


async def test_round_robin_alternates(backend):
    registry = make_registry(limits={"rps": 100})
    limiter = Limiter(registry, backend)
    accounts = []
    for _ in range(4):
        hold, _ = await take(limiter, deps(registry), strategy=Strategy.ROUND_ROBIN, rr_scope="m")
        accounts.append(hold.deployment.account)
    assert accounts == ["a", "b", "a", "b"]


async def test_least_used_picks_the_emptier_deployment(backend):
    registry = make_registry(limits={"rps": 10})
    limiter = Limiter(registry, backend)
    a, b = deps(registry)
    for _ in range(3):
        await take(limiter, [a])
    hold, _ = await take(limiter, [a, b], strategy=Strategy.LEAST_USED)
    assert hold.deployment is b


async def test_concurrency_limit_is_released_on_settle(backend):
    registry = make_registry(limits={"max_concurrent": 1})
    limiter = Limiter(registry, backend)
    only_a = deps(registry)[:1]
    hold, _ = await take(limiter, only_a)
    blocked, result = await take(limiter, only_a)
    assert blocked is None and result.reasons[0] == "busy"
    await limiter.settle(hold, actual=SMALL)
    assert (await take(limiter, only_a))[0] is not None


async def test_a_burst_of_200_concurrent_callers_never_exceeds_the_limit(backend):
    registry = make_registry(limits={"rps": 20, "input_tpm": 5_000})
    limiter = Limiter(registry, backend)
    only_a = deps(registry)[:1]
    results = await asyncio.gather(*[take(limiter, only_a, cost=Cost(100, 0)) for _ in range(200)])
    granted = sum(1 for hold, _ in results if hold is not None)
    assert granted == 20  # rps binds before input_tpm (20 x 100 < 5,000)


async def test_headroom_shrinks_the_usable_quota(backend):
    registry = make_registry(limits={"rps": 10}, headroom=0.8)
    limiter = Limiter(registry, backend)
    only_a = deps(registry)[:1]
    granted = 0
    while (await take(limiter, only_a))[0] is not None:
        granted += 1
    assert granted == 8


async def test_buckets_refill_continuously():
    """No window boundary: capacity comes back smoothly, one unit per 1/rate."""
    now = [1_000_000.0]
    backend = InProcessBackend(clock=lambda: now[0])
    registry = make_registry(limits={"rps": 4})
    limiter = Limiter(registry, backend)
    only_a = deps(registry)[:1]
    for _ in range(4):
        assert (await take(limiter, only_a))[0]
    assert (await take(limiter, only_a))[0] is None
    now[0] += 250  # one request's worth of refill at 4 rps
    assert (await take(limiter, only_a))[0]
    assert (await take(limiter, only_a))[0] is None
    now[0] += 10_000  # long idle: back to full, but never above
    granted = 0
    while (await take(limiter, only_a))[0]:
        granted += 1
    assert granted == 4
