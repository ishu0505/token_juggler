"""Load behaviour: what a greedy caller actually gets through, window by window.

The simulated-clock tests hammer the limiter every millisecond for 20 seconds
and then check every rolling window of the granted timestamps - the property
a provider's quota enforcement would see.
"""

import asyncio
import time

import fakeredis

from tests.conftest import make_registry
from tests.test_router import FakeAdapter
from tokenjuggler import TokenJuggler
from tokenjuggler.limiter import Cost, InProcessBackend, Limiter, RedisBackend
from tokenjuggler.settings import WireApi

ONE = Cost(1, 1)


def max_in_rolling_window(timestamps: list[float], window: float) -> int:
    timestamps = sorted(timestamps)
    best, start = 0, 0
    for end, t in enumerate(timestamps):
        while timestamps[start] <= t - window:
            start += 1
        best = max(best, end - start + 1)
    return best


async def greedy(window_mode: str, seconds: int = 20):
    now = [0.0]
    registry = make_registry(limits={"rps": 10})
    raw = registry.config.model_dump()
    raw["defaults"]["window"] = window_mode
    from tokenjuggler.registry import Registry
    from tokenjuggler.settings import Config

    registry = Registry(Config.model_validate(raw), environ={"KEY_A": "a", "KEY_B": "b"})
    limiter = Limiter(registry, InProcessBackend(clock=lambda: now[0]))
    deps = list(registry.model("m").deployments)
    grants: dict[str, list[float]] = {"a": [], "b": []}
    while now[0] < seconds * 1000:
        while True:
            hold, _ = await limiter.acquire(deps, [ONE, ONE])
            if hold is None:
                break
            grants[hold.deployment.account].append(now[0])
        now[0] += 1
    return grants


async def test_token_bucket_bursts_then_holds_the_long_run_rate():
    grants = await greedy("token_bucket")
    for account in "ab":
        ts = grants[account]
        # Long run: never above the limit's rate (+ one initial burst).
        assert len(ts) <= 10 * 20 + 10
        # A rolling second can hold a full burst plus a full second of refill.
        assert max_in_rolling_window(ts, 1000) <= 20
    # Overflow only reached b once a's burst was spent.
    assert min(grants["b"]) >= grants["a"][9]


async def test_sliding_window_mode_never_exceeds_the_limit_in_any_rolling_window():
    grants = await greedy("sliding")
    for account in "ab":
        assert max_in_rolling_window(grants[account], 1000) <= 10


async def test_concurrent_generate_spills_over_and_stays_in_limits_on_redis():
    """End to end through the router, Lua limiter and real time."""
    registry = make_registry(limits={"rps": 20})
    fake = FakeAdapter()
    tj = TokenJuggler(registry.config, adapters={api: fake for api in WireApi},
                    backend=RedisBackend(fakeredis.FakeAsyncRedis(decode_responses=True)),
                    environ={"KEY_A": "a", "KEY_B": "b"})
    served: dict[str, list[float]] = {"m@a": [], "m@b": []}
    tj.router.on_call = lambda rec: served[rec.deployment].append(time.monotonic())

    results = await asyncio.gather(
        *[tj.generate("m", "hi", max_wait_seconds=10) for _ in range(100)]
    )

    assert len(results) == 100
    assert served["m@b"], "overflow should have reached the second deployment"
    for ts in served.values():
        assert max_in_rolling_window(ts, 1.0) <= 40  # token bucket: burst + 1s refill
    # The primary took its whole burst before anything spilled over.
    assert len(served["m@a"]) >= 20
