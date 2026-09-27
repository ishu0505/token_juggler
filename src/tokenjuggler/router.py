"""Choosing where a request runs, and what happens when that route fails.

For one request:

1. Keep the requested model's deployments that can carry the payload
   (a PDF needs `pdf`, a schema needs `json_schema`, ...).
2. Ask the limiter to reserve quota on the best of them, per the model's
   strategy. That is one atomic step: a route with no room is never called.
3. Call it. On a 429, 5xx, timeout or route-specific error, give back what
   wasn't used, bench the route if it earned it, and try the next one - never
   the same one again, unless same-route retries were switched on.
4. When every deployment of the model is unavailable, do the same over its
   fallback models, in order.
5. When everything is merely FULL (not broken), wait for the earliest moment
   something has room, up to `max_wait_seconds`.

Switching back needs no state: with the priority strategy every request
starts at the first deployment, so traffic returns there as soon as its
buckets refill or its cooldown ends.
"""

from __future__ import annotations

import asyncio
import inspect
import random
import time
from collections.abc import Callable

from tokenjuggler.adapters import (
    Adapter,
    AdapterError,
    BadRequest,
    DeploymentError,
    RateLimited,
    Transient,
)
from tokenjuggler.estimate import estimate_input_tokens, reservation_cost
from tokenjuggler.limiter import Cost, Limiter
from tokenjuggler.registry import Deployment, Model, Registry
from tokenjuggler.tracking import CallRecord, Tracker, cost_usd
from tokenjuggler.types import Attempt, Request, Response
from tokenjuggler.utils.logger import get_logger

log = get_logger("tokenjuggler.router")


class RoutingError(RuntimeError):
    def __init__(self, message: str, attempts: list[Attempt]):
        detail = "; ".join(f"{a.deployment}: {a.outcome} {a.detail}".strip() for a in attempts)
        super().__init__(f"{message}" + (f" [{detail}]" if detail else ""))
        self.attempts = attempts


class NoCapableDeployment(RoutingError):
    """No configured route can carry this payload (e.g. audio to GPT)."""


class QuotaExceeded(RoutingError):
    """Every route stayed full for longer than max_wait_seconds - or the
    request is larger than any route's whole quota."""


class AllRoutesFailed(RoutingError):
    """Every capable route was tried and failed."""


OnCall = Callable[[CallRecord], object]


class Router:
    def __init__(
        self,
        registry: Registry,
        limiter: Limiter,
        adapters: dict,
        tracker: Tracker,
        *,
        on_call: OnCall | None = None,
    ):
        self.registry = registry
        self.limiter = limiter
        self.adapters = adapters
        self.tracker = tracker
        self.on_call = on_call
        cfg = registry.config
        self._defaults = cfg.defaults
        self._estimation = cfg.estimation

    def _chain(self, model: str) -> list[Model]:
        primary = self.registry.model(model)
        return [primary] + [self.registry.model(m) for m in primary.routing.fallbacks]

    async def generate(self, request: Request, *, max_wait_seconds: float | None = None) -> Response:
        chain = self._chain(request.model)
        required = request.required_capabilities()
        capable = {m.name: [d for d in m.deployments if d.supports(required)] for m in chain}
        if not any(capable.values()):
            raise NoCapableDeployment(
                f"no deployment of {request.model} (or its fallbacks) supports "
                f"{sorted(c.value for c in required)}",
                [],
            )

        t_start = time.perf_counter()
        timing = {"provider": 0.0, "failed_attempts": 0.0, "queue_wait": 0.0,
                  "limiter": 0.0, "settle": 0.0}
        max_wait = self._defaults.max_wait_seconds if max_wait_seconds is None else max_wait_seconds
        deadline = time.monotonic() + max_wait
        attempts: list[Attempt] = []
        failures: dict[str, int] = {}  # deployment id -> failed attempts this request
        retry = self._defaults.retry
        input_estimates: dict = {}

        def usable(dep: Deployment) -> bool:
            allowed = retry.max_attempts if retry.enabled else 1
            return failures.get(dep.id, 0) < allowed

        while True:
            shortest_wait: float | None = None
            blocked: list[str] = []  # why each model's routes had no room
            for model in chain:
                while True:
                    candidates = [d for d in capable[model.name] if usable(d)]
                    if not candidates:
                        break
                    costs = []
                    for d in candidates:
                        if d.family not in input_estimates:
                            input_estimates[d.family] = estimate_input_tokens(
                                request, d.family, self._estimation
                            )
                        costs.append(reservation_cost(
                            request, d, self._estimation,
                            input_tokens=input_estimates[d.family],
                        ))
                    t_acquire = time.perf_counter()
                    hold, result = await self.limiter.acquire(
                        candidates, costs, strategy=model.routing.strategy, rr_scope=model.name
                    )
                    timing["limiter"] += (time.perf_counter() - t_acquire) * 1000
                    if hold is None:
                        blocked.extend(r for r in result.reasons if r)
                        if result.wait_ms is not None:
                            wait = result.wait_ms / 1000
                            shortest_wait = wait if shortest_wait is None else min(shortest_wait, wait)
                        break
                    response = await self._attempt(request, hold, attempts, failures, timing)
                    if response is not None:
                        response.attempts = attempts
                        total = (time.perf_counter() - t_start) * 1000
                        timing["total"] = total
                        timing["overhead"] = total - timing["provider"] - timing[
                            "failed_attempts"] - timing["queue_wait"]
                        response.timings = {k: round(v, 3) for k, v in timing.items()}
                        return response

            if not any(usable(d) for deps in capable.values() for d in deps):
                raise AllRoutesFailed(f"every route for {request.model} failed", attempts)
            if shortest_wait is None:
                raise QuotaExceeded(
                    f"no route for {request.model} can ever take this request: "
                    + "; ".join(dict.fromkeys(blocked)), attempts
                )
            remaining = deadline - time.monotonic()
            if shortest_wait > remaining:
                raise QuotaExceeded(
                    f"every route for {request.model} is full; next capacity in "
                    f"{shortest_wait:.1f}s, over the {max_wait:.0f}s wait limit",
                    attempts,
                )
            # Jitter so a crowd of waiters doesn't wake in lockstep.
            t_wait = time.perf_counter()
            await asyncio.sleep(shortest_wait + random.uniform(0, 0.05))
            timing["queue_wait"] += (time.perf_counter() - t_wait) * 1000

    async def _attempt(self, request, hold, attempts, failures, timing) -> Response | None:
        """One call on a held deployment. Returns the response, or None to
        move on. Raises only for errors no other route would fix."""
        dep: Deployment = hold.deployment
        adapter: Adapter = self.adapters[dep.api]
        started = time.perf_counter()
        try:
            result = await adapter.call(dep, request)
        except AdapterError as error:
            latency = (time.perf_counter() - started) * 1000
            timing["failed_attempts"] += latency
            outcome = _outcome(error)
            record = CallRecord(
                project=self.limiter.project, model=dep.model, deployment=dep.id,
                provider=dep.provider.value, outcome=outcome, latency_ms=latency,
                job_id=request.job_id, error=str(error)[:500],
            )
            await self.limiter.settle(
                hold, actual=None, sent=error.sent, usage=self.tracker.usage_write(record)
            )
            await self._emit(record)
            attempts.append(Attempt(dep.id, outcome, str(error)[:200]))
            if isinstance(error, BadRequest):
                raise
            failures[dep.id] = failures.get(dep.id, 0) + 1
            if isinstance(error, RateLimited):
                bench = error.retry_after or self._defaults.cooldown_seconds
                await self.limiter.cooldown(dep, bench)
                log.warning("%s returned 429; benched for %.0fs", dep.id, bench)
            elif isinstance(error, DeploymentError):
                await self.limiter.cooldown(dep, self._defaults.error_cooldown_seconds)
                log.error("%s failed (%s); benched", dep.id, error)
            elif isinstance(error, Transient):
                log.warning("%s transient failure: %s", dep.id, error)
                if self._defaults.retry.enabled:
                    await asyncio.sleep(self._backoff(failures[dep.id]))
            return None

        latency = (time.perf_counter() - started) * 1000
        timing["provider"] = latency
        cost = cost_usd(dep.price, result.usage)
        record = CallRecord(
            project=self.limiter.project, model=dep.model, deployment=dep.id,
            provider=dep.provider.value, outcome="ok", usage=result.usage, cost_usd=cost,
            latency_ms=latency, job_id=request.job_id,
            # Overhead so far; the settle below is the only part not yet counted.
            overhead_ms=timing["limiter"],
        )
        t_settle = time.perf_counter()
        await self.limiter.settle(
            hold,
            actual=Cost(result.usage.input_tokens, result.usage.output_tokens),
            usage=self.tracker.usage_write(record),
        )
        timing["settle"] += (time.perf_counter() - t_settle) * 1000
        await self._emit(record)
        attempts.append(Attempt(dep.id, "ok"))
        return Response(
            text=result.text,
            model=dep.model,
            deployment=dep.id,
            provider=dep.provider.value,
            usage=result.usage,
            cost_usd=cost,
            latency_ms=latency,
            truncated=result.truncated,
            parsed=_parse(request, result.text, result.truncated),
            raw=result.raw,
        )

    def _backoff(self, attempt: int) -> float:
        policy = self._defaults.retry
        delay = policy.base_delay_seconds * (2 ** (attempt - 1))
        return delay + (random.uniform(0, delay) if policy.jitter else 0)

    async def _emit(self, record: CallRecord) -> None:
        if self.on_call is None:
            return
        try:
            result = self.on_call(record)
            if inspect.isawaitable(result):
                await result
        except Exception as exc:  # a broken sink must not fail the call
            log.error("on_call hook failed: %s", exc)


def _outcome(error: AdapterError) -> str:
    if isinstance(error, RateLimited):
        return "rate_limited"
    if isinstance(error, Transient):
        return "transient"
    if isinstance(error, BadRequest):
        return "bad_request"
    return "deployment_error"


def _parse(request: Request, text: str, truncated: bool):
    schema = request.response_schema
    if not isinstance(schema, type) or truncated or not text:
        return None
    try:
        return schema.model_validate_json(text)
    except ValueError as exc:
        log.warning("response did not match %s: %s", schema.__name__, exc)
        return None
