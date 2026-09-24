"""Redis primitives behind the LLM gate: token buckets, semaphore, job budget.

Split out from `gate.py` so the gate reads as a sequence of clearly-named
steps and this file holds the Redis details. Nothing here decides policy - it
counts, reserves and releases, and the gate decides what to do about it.

Every key is namespaced by provider and model, because limits are per-model.
"""

from __future__ import annotations

import time
from typing import Any

from token_daddy.config import settings
from token_daddy.utils.logger import get_logger

log = get_logger("worker.llm.limits")

# Counters live two minutes so the previous bucket is still readable while the
# current one fills. Anything longer just wastes memory.
BUCKET_TTL_SECONDS = 120

# A semaphore slot older than this is assumed to belong to a crashed worker.
# Long enough for a slow vision call, short enough that a crash does not
# permanently shrink the pool.
SEMAPHORE_STALE_SECONDS = 300

# Connection pool size. Must comfortably exceed the number of tasks that can
# be waiting on the gate at once, because a waiting task holds a connection
# for each round trip. Too small and tasks block acquiring a CONNECTION rather
# than a slot - the limiter then throttles on its own plumbing, which looks
# exactly like the provider being slow and is very hard to diagnose.
MAX_REDIS_CONNECTIONS = 512


def current_bucket() -> int:
    """The minute bucket a call belongs to. Counters are per-minute."""
    return int(time.time() // 60)


def tpm_key(provider: str, model: str) -> str:
    return f"ratelimit:{provider}:{model}:tpm:{current_bucket()}"


def rpm_key(provider: str, model: str) -> str:
    return f"ratelimit:{provider}:{model}:rpm:{current_bucket()}"


def output_tpm_key(workspace: str, model: str = "") -> str:
    """Output tokens per minute — the quota that actually binds.

    Keyed on workspace AND MODEL, because that is how the provider counts it.
    The 429 says so in its own words:

        Exceeded workspace output tokens per minute rate limit for
        databricks-gemini-3-8-flash

    So each model has its own per-minute output allowance at workspace scope.
    An earlier version keyed on the workspace alone; that would have made two
    models share one budget and throttle each other for no reason.

    It is NOT keyed on provider+model the way `tpm_key` is, because the same
    model reached through two routes is still one quota.
    """
    return f"ratelimit:workspace:{workspace}:{model}:output_tpm:{current_bucket()}"


def semaphore_key(provider: str, model: str) -> str:
    return f"semaphore:{provider}:{model}"


def budget_key(job_id: str) -> str:
    return f"budget:job:{job_id}"


def seconds_until_next_bucket() -> float:
    """How long to wait for the counters to reset."""
    return 60 - (time.time() % 60)


class InProcessRedis:
    """A tiny in-memory stand-in for the handful of Redis commands the gate uses.

    Used when REDIS_URL is unset, which happens in tests and in local dev.

    The limits it enforces are REAL but PER-PROCESS. That is the important
    caveat: the whole reason the counters live in Redis is that limits are per
    provider across ALL workers, and twenty workers each staying under the
    ceiling individually will blow it collectively. This fallback cannot see
    other processes, so it protects a single machine and nothing more.

    The alternative - refusing to start without Redis - was worse. It made the
    pipeline impossible to test or run locally, which pushes people towards
    bypassing the gate entirely, and a bypassed gate enforces nothing at all.
    """

    def __init__(self) -> None:
        self._values: dict[str, int] = {}
        self._sorted_sets: dict[str, dict[str, float]] = {}
        self._hashes: dict[str, dict[str, int]] = {}

    async def incrby(self, key: str, amount: int) -> int:
        self._values[key] = self._values.get(key, 0) + amount
        return self._values[key]

    async def decrby(self, key: str, amount: int) -> int:
        return await self.incrby(key, -amount)

    async def incr(self, key: str) -> int:
        return await self.incrby(key, 1)

    async def decr(self, key: str) -> int:
        return await self.incrby(key, -1)

    async def get(self, key: str) -> str | None:
        value = self._values.get(key)
        return str(value) if value is not None else None

    async def hincrby(self, key: str, field: str, amount: int) -> int:
        values = self._hashes.setdefault(key, {})
        values[field] = values.get(field, 0) + amount
        return values[field]

    async def hgetall(self, key: str) -> dict[str, str]:
        return {
            field: str(value)
            for field, value in self._hashes.get(key, {}).items()
        }

    async def expire(self, key: str, seconds: int) -> None:
        """No-op: the process is the lifetime here."""

    async def zadd(self, key: str, mapping: dict, nx: bool = False) -> None:
        members = self._sorted_sets.setdefault(key, {})
        for member, score in mapping.items():
            if nx and member in members:
                continue
            members[member] = score

    async def zrank(self, key: str, member: str) -> int | None:
        members = self._sorted_sets.get(key, {})
        if member not in members:
            return None
        ordered = sorted(members, key=lambda m: members[m])
        return ordered.index(member)

    async def zcard(self, key: str) -> int:
        return len(self._sorted_sets.get(key, {}))

    async def zrem(self, key: str, member: str) -> None:
        self._sorted_sets.get(key, {}).pop(member, None)

    async def zremrangebyscore(self, key: str, low: Any, high: float) -> None:
        members = self._sorted_sets.get(key, {})
        for member in [m for m, score in members.items() if score <= high]:
            del members[member]

    async def keys(self, pattern: str) -> list[str]:
        prefix = pattern.rstrip("*")
        return [
            k
            for k in (*self._values, *self._sorted_sets, *self._hashes)
            if k.startswith(prefix)
        ]

    async def delete(self, *keys: str) -> None:
        for key in keys:
            self._values.pop(key, None)
            self._sorted_sets.pop(key, None)
            self._hashes.pop(key, None)

    async def ping(self) -> bool:
        return True

    async def aclose(self) -> None:
        """Nothing to close."""


_client: Any = None


async def get_redis() -> Any:
    """The shared Redis client, built once per process.

    Cached deliberately. Building a client per call means setting up and
    tearing down a connection pool for every model call - measurably slower
    than the work it protects, which defeats the point of a fast limiter.
    `redis.asyncio` pools internally and is safe to share across tasks.

    Falls back to `InProcessRedis` when REDIS_URL is unset. See that class for
    what that does and does not protect.
    """
    global _client
    if _client is not None:
        return _client

    if not settings.redis_url:
        log.warning(
            "REDIS_URL is not set - rate limits will be enforced PER PROCESS "
            "only. Several workers will not share a quota. Set REDIS_URL "
            "anywhere that matters."
        )
        _client = InProcessRedis()
        return _client

    import redis.asyncio as aioredis

    _client = aioredis.from_url(
        settings.redis_url,
        decode_responses=True,
        max_connections=MAX_REDIS_CONNECTIONS,
    )
    log.info("Redis client ready for rate limiting")
    return _client


async def close_redis() -> None:
    """Close the shared client. For service shutdown and tests."""
    global _client
    if _client is not None:
        await _client.aclose()
        _client = None


async def try_reserve_tokens(redis: Any, provider: str, model: str, tokens: int, ceiling: int) -> bool:
    """Reserve `tokens` against this minute's budget. False if it would breach.

    Reserve-then-reconcile: we cannot know the real token count before the
    call, so we claim an estimate and correct afterwards. On refusal the
    reservation is given straight back, so a rejected attempt costs nothing.
    """
    key = tpm_key(provider, model)
    total = await redis.incrby(key, tokens)
    await redis.expire(key, BUCKET_TTL_SECONDS)

    if total > ceiling:
        await redis.decrby(key, tokens)
        log.debug(
            "TPM refused for %s/%s: %d + %d would exceed %d",
            provider, model, total - tokens, tokens, ceiling,
        )
        return False
    return True


async def try_reserve_output_tokens(
    redis: Any, workspace: str, model: str, tokens: int, ceiling: int
) -> bool:
    """Reserve OUTPUT tokens against the workspace's minute. False if full.

    Same reserve-then-reconcile shape as `try_reserve_tokens`, one level up:
    that one meters total tokens per model, which is not what the provider
    limits. A ceiling of 0 disables the check, for accounts with no such
    quota.
    """
    if ceiling <= 0:
        return True

    key = output_tpm_key(workspace, model)
    total = await redis.incrby(key, tokens)
    await redis.expire(key, BUCKET_TTL_SECONDS)

    if total > ceiling:
        await redis.decrby(key, tokens)
        log.debug(
            "Workspace output TPM refused: %d + %d would exceed %d",
            total - tokens, tokens, ceiling,
        )
        return False
    return True


async def reconcile_output_tokens(
    redis: Any, workspace: str, model: str, estimated: int, actual: int
) -> None:
    """Correct the workspace output bucket once real usage is known."""
    difference = actual - estimated
    if difference == 0:
        return
    key = output_tpm_key(workspace, model)
    await redis.incrby(key, difference)
    await redis.expire(key, BUCKET_TTL_SECONDS)


async def release_output_tokens(
    redis: Any, workspace: str, model: str, tokens: int
) -> None:
    """Give an output reservation straight back.

    Needed on the concurrency-wait path for the same reason `release_request`
    is: that loop reserves, then finds no slot free, and without a release it
    would burn the workspace's whole output minute while waiting.
    """
    if tokens <= 0:
        return
    await redis.decrby(output_tpm_key(workspace, model), tokens)


async def try_reserve_request(redis: Any, provider: str, model: str, ceiling: int) -> bool:
    """Count one request against this minute's request limit."""
    key = rpm_key(provider, model)
    total = await redis.incr(key)
    await redis.expire(key, BUCKET_TTL_SECONDS)

    if total > ceiling:
        await redis.decr(key)
        log.debug("RPM refused for %s/%s: %d exceeds %d", provider, model, total, ceiling)
        return False
    return True


async def release_request(redis: Any, provider: str, model: str) -> None:
    """Give back a request reservation.

    Needed on the concurrency-wait path: that loop reserves tokens AND a
    request, then discovers no slot is free. Without giving the request back,
    every poll iteration permanently burns one request from the minute's
    quota - so under contention the limiter exhausts its own RPM in seconds
    and then makes every caller sleep to the next bucket. It deadlocks itself
    precisely when it is busiest, which is the worst possible time.
    """
    key = rpm_key(provider, model)
    await redis.decr(key)


async def reconcile_tokens(
    redis: Any, provider: str, model: str, estimated: int, actual: int
) -> None:
    """Correct the bucket once the real usage is known.

    The correction can be negative, which is the common case - estimates run
    high on purpose, since briefly under-using a quota costs latency while
    exceeding it costs the whole batch.
    """
    difference = actual - estimated
    if difference == 0:
        return
    key = tpm_key(provider, model)
    await redis.incrby(key, difference)
    await redis.expire(key, BUCKET_TTL_SECONDS)
    log.debug(
        "Reconciled %s/%s by %+d tokens (estimated %d, actual %d)",
        provider, model, difference, estimated, actual,
    )


async def try_acquire_slot(
    redis: Any, provider: str, model: str, request_id: str, max_concurrent: int
) -> bool:
    """Take a concurrency slot. False if all of them are busy.

    A sorted set scored by timestamp, for two reasons.

    **Stale entries can be dropped.** A plain counter that a crashed worker
    never decremented would hold its slot forever, and the pool would silently
    shrink to zero with nothing obviously broken.

    **Winners are decided by RANK, not by a check-then-add.** This matters more
    than it looks. The obvious implementation - count members, add yourself if
    there is room - is not atomic, and under contention every waiter passes the
    count check, every waiter adds itself, every waiter then sees the set is
    over capacity, and every waiter backs off. Nobody makes progress. That is a
    livelock, and it appears exactly when the limiter is busiest.

    Adding first and then checking rank makes the outcome deterministic: the
    `max_concurrent` lowest timestamps hold the slots, everyone else stands
    down. It is also FIFO, so a waiter cannot be starved indefinitely.
    """
    key = semaphore_key(provider, model)
    now = time.time()

    await redis.zremrangebyscore(key, "-inf", now - SEMAPHORE_STALE_SECONDS)

    # Add first, then find out whether we are early enough to count. Re-adding
    # an existing member just updates its score, so a retrying waiter does not
    # accumulate duplicates - but it also means a waiter keeps its original
    # position only if we do not re-score it, which is why NX is used.
    await redis.zadd(key, {request_id: now}, nx=True)

    rank = await redis.zrank(key, request_id)
    if rank is not None and rank < max_concurrent:
        return True

    # Not in the winning set. Leave the entry in place so this waiter keeps its
    # queue position and moves up as slots free - removing it would send it to
    # the back of the queue on every poll and could starve it forever.
    return False


async def release_slot(redis: Any, provider: str, model: str, request_id: str) -> None:
    await redis.zrem(semaphore_key(provider, model), request_id)


async def in_flight(redis: Any, provider: str, model: str, max_concurrent: int = 0) -> int:
    """How many calls HOLD a slot right now.

    Since acquisition is rank-based, the sorted set contains waiters as well as
    holders - so the raw count is the queue, not the in-flight number. Pass
    `max_concurrent` to get holders; omit it to get everyone in the queue.
    """
    key = semaphore_key(provider, model)
    await redis.zremrangebyscore(key, "-inf", time.time() - SEMAPHORE_STALE_SECONDS)
    total = await redis.zcard(key)
    return min(total, max_concurrent) if max_concurrent else total


async def queue_depth(redis: Any, provider: str, model: str, max_concurrent: int) -> int:
    """How many calls are WAITING for a slot. Worth exposing on /health -
    a queue that never drains means the ceilings are wrong."""
    key = semaphore_key(provider, model)
    total = await redis.zcard(key)
    return max(0, total - max_concurrent)


async def add_to_job_budget(redis: Any, job_id: str, tokens: int) -> int:
    """Charge tokens to a job and return its running total."""
    key = budget_key(job_id)
    total = await redis.incrby(key, tokens)
    # Budgets outlive a job only long enough to be inspected afterwards.
    await redis.expire(key, 24 * 60 * 60)
    return total


async def get_job_budget(redis: Any, job_id: str) -> int:
    value = await redis.get(budget_key(job_id))
    return int(value) if value else 0


def usage_key(job_id: str) -> str:
    return f"job:{job_id}:usage"


async def add_usage(
    redis: Any, job_id: str, provider: str, model: str, tokens: int
) -> None:
    """Charge tokens to one provider AND model, beside the job total.

    The total alone answers "did this run cost too much"; this answers "which
    stage and which provider spent it", which is the question that decides
    where to look next. Split by model too, since a run can mix a cheap model
    for indexing with an expensive one for filling.
    """
    key = usage_key(job_id)
    await redis.hincrby(key, f"{provider}:{model}", tokens)
    await redis.expire(key, 24 * 60 * 60)


async def get_usage(redis: Any, job_id: str) -> dict[str, int]:
    """Per provider:model token totals for one job. Empty if nothing ran."""
    raw = await redis.hgetall(usage_key(job_id))
    return {key: int(value) for key, value in (raw or {}).items()}
