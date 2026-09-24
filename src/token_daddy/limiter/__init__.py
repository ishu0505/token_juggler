"""Quota enforcement: turns deployments + a call's cost into bucket reservations.

Each deployment's quota is checked at up to three scopes, all-or-nothing:

* the deployment itself (`gpt-5.4@openai`),
* its account, when the account has account-wide limits,
* the calling project's slice, when the project has a `share`.

A call reserves its cost on every bucket of every scope in one atomic step,
and settles afterwards: the difference between reserved and actual usage is
refunded (or charged). Keys all carry a `{namespace}` hash tag so a whole
namespace lives in one Redis Cluster slot.
"""

from __future__ import annotations

import random
import uuid
from dataclasses import dataclass, field

from token_daddy.limiter.backend import (
    AcquireResult,
    Adjustment,
    Backend,
    BucketSpec,
    CandidateSpec,
    InProcessBackend,
    RedisBackend,
    UsageWrite,
    lease_ms_for,
)
from token_daddy.registry import Deployment, Registry
from token_daddy.settings import LIMIT_KINDS, Limits, Strategy, Window

__all__ = [
    "AcquireResult", "Backend", "Cost", "Hold", "InProcessBackend", "Limiter",
    "RedisBackend", "UsageWrite",
]


@dataclass(frozen=True)
class Cost:
    """What one call spends against quotas. Output is whatever was reserved:
    max_output_tokens under strict reservation, an estimate otherwise."""

    input_tokens: int
    output_tokens: int

    def units(self, kind: str) -> int:
        match kind:
            case "requests":
                return 1
            case "input":
                return self.input_tokens
            case "output":
                return self.output_tokens
            case "total":
                return self.input_tokens + self.output_tokens
        raise ValueError(kind)


@dataclass
class Hold:
    """Quota held for one call on one deployment, until `Limiter.settle`."""

    deployment: Deployment
    request_id: str
    cost: Cost
    buckets: tuple[tuple[BucketSpec, str], ...]  # (bucket, kind)
    concurrency_key: str | None
    settled: bool = field(default=False, repr=False)


class Limiter:
    def __init__(self, registry: Registry, backend: Backend, *, project: str | None = None):
        self.registry = registry
        self.backend = backend
        self.project = project
        cfg = registry.config
        self._ns = cfg.namespace
        self._headroom = cfg.defaults.headroom
        self._lease_ms = lease_ms_for(cfg.defaults.request_timeout_seconds)

    # -- keys -----------------------------------------------------------------

    def key(self, *parts: str) -> str:
        return ":".join([f"{{{self._ns}}}", *parts])

    def cooldown_key(self, dep: Deployment) -> str:
        return self.key("cool", dep.id)

    # -- planning -------------------------------------------------------------

    def _scope_buckets(
        self, scope: str, label: str, limits: Limits, share: float, cost: Cost,
        window: Window,
    ) -> list[tuple[BucketSpec, str]]:
        # Sliding: a burst of C/2 plus C/2 of refill per window can never put
        # more than C into any rolling window.
        factor = 0.5 if window is Window.SLIDING else 1.0
        out = []
        for name, value in limits.buckets().items():
            window_s, kind = LIMIT_KINDS[name]
            capacity = value * share * self._headroom * factor
            window_ms = window_s * 1000
            out.append((
                BucketSpec(
                    key=self.key("b", scope, name),
                    per_unit_ms=window_ms / capacity,
                    units=cost.units(kind),
                    window_ms=window_ms,
                    label=f"{label} {name}",
                ),
                kind,
            ))
        return out

    def plan(self, dep: Deployment, cost: Cost) -> tuple[CandidateSpec, tuple]:
        buckets = self._scope_buckets(f"d:{dep.id}", dep.id, dep.limits, 1.0, cost, dep.window)
        if dep.account_limits:
            buckets += self._scope_buckets(
                f"a:{dep.account}", f"account {dep.account}", dep.account_limits, 1.0, cost,
                dep.window,
            )
        share = self.registry.project_share(self.project, dep.id)
        if share:
            buckets += self._scope_buckets(
                f"p:{self.project}:{dep.id}", f"project {self.project} on {dep.id}",
                dep.limits, share, cost, dep.window,
            )
        concurrency = None
        if dep.limits.max_concurrent:
            concurrency = (self.key("conc", dep.id), dep.limits.max_concurrent)
        spec = CandidateSpec(
            deployment_id=dep.id,
            cooldown_key=self.cooldown_key(dep),
            buckets=tuple(b for b, _ in buckets),
            concurrency=concurrency,
        )
        return spec, tuple(buckets)

    # -- the two calls every model call makes ---------------------------------

    async def acquire(
        self,
        deployments: list[Deployment],
        costs: list[Cost],
        *,
        strategy: Strategy = Strategy.PRIORITY,
        rr_scope: str = "",
    ) -> tuple[Hold | None, AcquireResult]:
        """Reserve quota on the best candidate that has room, or on none."""
        order = list(range(len(deployments)))
        if strategy is Strategy.WEIGHTED:
            order = _weighted_order([d.weight for d in deployments])
        planned = [self.plan(deployments[i], costs[i]) for i in order]
        request_id = uuid.uuid4().hex
        result = await self.backend.acquire(
            [spec for spec, _ in planned],
            mode="least" if strategy is Strategy.LEAST_USED else "first",
            rr_key=self.key("rr", rr_scope) if strategy is Strategy.ROUND_ROBIN else None,
            member=request_id,
            lease_ms=self._lease_ms,
        )
        if result.index is None:
            return None, result
        winner = order[result.index]
        spec, buckets = planned[result.index]
        hold = Hold(
            deployment=deployments[winner],
            request_id=request_id,
            cost=costs[winner],
            buckets=buckets,
            concurrency_key=spec.concurrency[0] if spec.concurrency else None,
        )
        return hold, result

    async def settle(
        self,
        hold: Hold,
        *,
        actual: Cost | None,
        sent: bool = True,
        usage: UsageWrite | None = None,
    ) -> None:
        """Correct the reservation to what the call really cost.

        `actual=None` means the call failed. If it never left this process,
        everything is refunded. If it reached the provider, the request and
        its input tokens stay counted - providers count rejected and failed
        requests too - and only the unproduced output is refunded.
        """
        if hold.settled:
            return
        hold.settled = True
        if actual is None:
            actual = (
                Cost(hold.cost.input_tokens, 0) if sent else Cost(0, 0)
            )
            keep_request = sent
        else:
            keep_request = True
        adjustments = []
        for bucket, kind in hold.buckets:
            if kind == "requests":
                delta = 0 if keep_request else -1
            else:
                delta = actual.units(kind) - hold.cost.units(kind)
            if delta:
                adjustments.append(Adjustment(bucket.key, bucket.per_unit_ms, delta))
        await self.backend.settle(
            adjustments,
            concurrency=(hold.concurrency_key, hold.request_id) if hold.concurrency_key else None,
            usage=usage,
        )

    async def cooldown(self, dep: Deployment, seconds: float) -> None:
        await self.backend.cooldown(self.cooldown_key(dep), int(seconds * 1000))


def _weighted_order(weights: list[int]) -> list[int]:
    """A random order where heavier candidates tend to come first."""
    remaining = [i for i, w in enumerate(weights) if w > 0]
    order: list[int] = []
    while remaining:
        pick = random.choices(remaining, weights=[weights[i] for i in remaining])[0]
        order.append(pick)
        remaining.remove(pick)
    # Zero-weight candidates are last resorts, not excluded.
    return order + [i for i, w in enumerate(weights) if w == 0]
