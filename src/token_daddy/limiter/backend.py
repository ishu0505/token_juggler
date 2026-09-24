"""Where bucket state lives: Redis (shared) or this process (single worker).

Both backends implement the same GCRA arithmetic - the Redis one in Lua (see
`lua/acquire.lua` for the algorithm), the in-process one in Python - and the
test suite runs every limiter test against both.
"""

from __future__ import annotations

import json
import math
import threading
import time
from collections import defaultdict
from dataclasses import dataclass
from importlib import resources
from typing import Any, Literal, Protocol


@dataclass(frozen=True)
class BucketSpec:
    key: str
    per_unit_ms: float  # window / capacity
    units: float  # what this call spends
    window_ms: int
    label: str  # e.g. "gpt-5.4@openai output_tpm", for error messages


@dataclass(frozen=True)
class CandidateSpec:
    deployment_id: str
    cooldown_key: str
    buckets: tuple[BucketSpec, ...]
    concurrency: tuple[str, int] | None = None  # (key, max in flight)


@dataclass(frozen=True)
class AcquireResult:
    index: int | None  # position in the candidate list that won, 0-based
    # When nothing fit: ms until something might. None means nothing ever
    # will - the request is bigger than every candidate's whole quota.
    wait_ms: float | None = None
    reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class Adjustment:
    key: str
    per_unit_ms: float
    delta_units: float  # actual - reserved; negative is a refund


@dataclass
class UsageWrite:
    rollup_keys: list[tuple[str, int]]  # (hash key, ttl ms)
    fields: dict[str, int]
    index: tuple[str, str] | None = None  # (set key, member)
    stream: tuple[str, int, dict[str, str]] | None = None  # (key, maxlen, entry)


Mode = Literal["first", "least"]


class Backend(Protocol):
    async def acquire(
        self, candidates: list[CandidateSpec], *, mode: Mode, rr_key: str | None,
        member: str, lease_ms: int,
    ) -> AcquireResult: ...

    async def settle(
        self, adjustments: list[Adjustment], *, concurrency: tuple[str, str] | None,
        usage: UsageWrite | None,
    ) -> None: ...

    async def cooldown(self, key: str, ms: int) -> None: ...

    async def read_hashes(self, keys: list[str]) -> list[dict[str, int]]: ...

    async def members(self, key: str) -> set[str]: ...

    async def recent(self, key: str, count: int) -> list[dict[str, str]]: ...

    async def close(self) -> None: ...


def _reason(raw: str, candidate: CandidateSpec) -> str:
    """Turn the script's 'full:2' into 'full: gpt-5.4@openai output_tpm'."""
    kind, _, index = raw.partition(":")
    if index:
        return f"{kind}: {candidate.buckets[int(index) - 1].label}"
    return raw


class _KeyTable:
    """Collects KEYS for a script call, returning 1-based indexes (Lua style)."""

    def __init__(self) -> None:
        self.keys: list[str] = []
        self._index: dict[str, int] = {}

    def __call__(self, key: str) -> int:
        if key not in self._index:
            self.keys.append(key)
            self._index[key] = len(self.keys)
        return self._index[key]


def _lua(name: str) -> str:
    return resources.files("token_daddy.limiter").joinpath(f"lua/{name}").read_text()


class RedisBackend:
    """Shared state in Redis. Two round trips per call: acquire and settle."""

    def __init__(self, client: Any):
        self._redis = client
        self._acquire = client.register_script(_lua("acquire.lua"))
        self._settle = client.register_script(_lua("settle.lua"))

    @classmethod
    def from_url(cls, url: str, *, max_connections: int = 512) -> RedisBackend:
        import redis.asyncio as aioredis

        return cls(aioredis.from_url(url, decode_responses=True, max_connections=max_connections))

    async def acquire(self, candidates, *, mode, rr_key, member, lease_ms):
        keys = _KeyTable()
        payload_cands = []
        for cand in candidates:
            entry: dict[str, Any] = {
                "cool": keys(cand.cooldown_key),
                "b": [[keys(b.key), b.per_unit_ms, b.units, b.window_ms] for b in cand.buckets],
            }
            # Omitted rather than null: cjson decodes null to a truthy sentinel.
            if cand.concurrency:
                entry["conc"] = [keys(cand.concurrency[0]), cand.concurrency[1]]
            payload_cands.append(entry)
        payload = {
            "c": payload_cands, "mode": mode, "rr": keys(rr_key) if rr_key else 0,
            "m": member, "l": lease_ms,
        }
        raw = json.loads(await self._acquire(keys=keys.keys, args=[json.dumps(payload)]))
        if raw["ok"]:
            return AcquireResult(index=raw["ok"] - 1)
        why = raw.get("why") or []
        if isinstance(why, dict):  # cjson encodes an empty table as {}
            why = []
        wait = raw["wait"]
        return AcquireResult(
            index=None,
            wait_ms=None if wait < 0 else float(wait),
            reasons=tuple(_reason(r, c) for r, c in zip(why, candidates)),
        )

    async def settle(self, adjustments, *, concurrency, usage):
        keys = _KeyTable()
        payload: dict[str, Any] = {
            "a": [[keys(a.key), a.per_unit_ms, a.delta_units] for a in adjustments if a.delta_units],
        }
        if concurrency:
            payload["cc"] = [keys(concurrency[0]), concurrency[1]]
        if usage:
            payload["u"] = {
                "k": [keys(k) for k, _ in usage.rollup_keys],
                "ttl": [ttl for _, ttl in usage.rollup_keys],
                "f": usage.fields,
            }
            if usage.index:
                payload["ix"] = [keys(usage.index[0]), usage.index[1]]
            if usage.stream:
                key, maxlen, entry = usage.stream
                flat = [str(v) for pair in entry.items() for v in pair]
                payload["x"] = [keys(key), maxlen, flat]
        if not payload["a"]:
            payload.pop("a")  # an empty list would reach Lua as an empty object
        if not keys.keys:
            return
        await self._settle(keys=keys.keys, args=[json.dumps(payload)])

    async def cooldown(self, key, ms):
        await self._redis.set(key, "1", px=max(1, int(ms)))

    async def read_hashes(self, keys):
        async with self._redis.pipeline(transaction=False) as pipe:
            for key in keys:
                pipe.hgetall(key)
            rows = await pipe.execute()
        return [{k: int(v) for k, v in row.items()} for row in rows]

    async def members(self, key):
        return set(await self._redis.smembers(key))

    async def recent(self, key, count):
        return [entry for _, entry in await self._redis.xrevrange(key, count=count)]

    async def close(self):
        await self._redis.aclose()


class InProcessBackend:
    """The same arithmetic in memory, for tests and single-process use.

    Limits are real but PER PROCESS: several workers each staying under a
    quota will blow it together. Use Redis anywhere more than one process
    shares an account.
    """

    def __init__(self, clock=None):
        self._clock = clock or (lambda: time.time() * 1000)
        self._lock = threading.Lock()
        self._tat: dict[str, float] = {}
        self._expiry: dict[str, float] = {}  # cooldowns
        self._leases: dict[str, dict[str, float]] = defaultdict(dict)
        self._counters: dict[str, int] = defaultdict(int)
        self._hashes: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self._sets: dict[str, set[str]] = defaultdict(set)
        self._streams: dict[str, list[dict[str, str]]] = defaultdict(list)

    def _evaluate(self, cand: CandidateSpec, now: float):
        expires = self._expiry.get(cand.cooldown_key, 0)
        if expires > now:
            return None, "cooldown", expires - now
        tats, score = [], 0.0
        for j, b in enumerate(cand.buckets, start=1):
            cost = b.units * b.per_unit_ms
            if cost > b.window_ms:
                return None, f"too_large:{j}", -1
            new_tat = max(self._tat.get(b.key, now), now) + cost
            if new_tat - now > b.window_ms:
                return None, f"full:{j}", new_tat - now - b.window_ms
            tats.append(new_tat)
            score = max(score, (new_tat - now) / b.window_ms)
        if cand.concurrency:
            key, limit = cand.concurrency
            leases = self._leases[key]
            for m in [m for m, exp in leases.items() if exp <= now]:
                del leases[m]
            if len(leases) >= limit:
                return None, "busy", 50
        return tats, score, 0

    async def acquire(self, candidates, *, mode, rr_key, member, lease_ms):
        with self._lock:
            now = self._clock()
            order = list(range(len(candidates)))
            if rr_key and len(candidates) > 1:
                self._counters[rr_key] += 1
                shift = (self._counters[rr_key] - 1) % len(candidates)
                order = order[shift:] + order[:shift]
            why = [""] * len(candidates)
            min_wait: float | None = None
            best = None
            for ci in order:
                tats, a, b = self._evaluate(candidates[ci], now)
                if tats is not None:
                    if mode != "least":
                        self._commit(candidates[ci], tats, now, member, lease_ms)
                        return AcquireResult(index=ci)
                    if best is None or a < best[1]:
                        best = (ci, a, tats)
                else:
                    why[ci] = a
                    if b >= 0 and (min_wait is None or b < min_wait):
                        min_wait = b
            if best is not None:
                self._commit(candidates[best[0]], best[2], now, member, lease_ms)
                return AcquireResult(index=best[0])
            return AcquireResult(
                index=None,
                wait_ms=min_wait,
                reasons=tuple(_reason(r, c) for r, c in zip(why, candidates)),
            )

    def _commit(self, cand, tats, now, member, lease_ms):
        for b, tat in zip(cand.buckets, tats):
            self._tat[b.key] = tat
        if cand.concurrency:
            self._leases[cand.concurrency[0]][member] = now + lease_ms

    async def settle(self, adjustments, *, concurrency, usage):
        with self._lock:
            now = self._clock()
            for a in adjustments:
                tat = self._tat.get(a.key, now)
                if a.delta_units < 0:
                    tat = max(now, tat + a.delta_units * a.per_unit_ms)
                else:
                    tat = max(tat, now) + a.delta_units * a.per_unit_ms
                self._tat[a.key] = tat
            if concurrency:
                self._leases[concurrency[0]].pop(concurrency[1], None)
            if usage:
                for key, _ttl in usage.rollup_keys:
                    for f, v in usage.fields.items():
                        if v:
                            self._hashes[key][f] += v
                if usage.index:
                    self._sets[usage.index[0]].add(usage.index[1])
                if usage.stream:
                    key, maxlen, entry = usage.stream
                    stream = self._streams[key]
                    stream.append(dict(entry))
                    del stream[:-maxlen]

    async def cooldown(self, key, ms):
        with self._lock:
            self._expiry[key] = self._clock() + ms

    async def read_hashes(self, keys):
        return [dict(self._hashes.get(k, {})) for k in keys]

    async def members(self, key):
        return set(self._sets.get(key, set()))

    async def recent(self, key, count):
        return list(reversed(self._streams.get(key, [])[-count:]))

    async def close(self):
        pass


def lease_ms_for(timeout_seconds: float) -> int:
    """A concurrency lease outlives the request timeout, so a crashed worker's
    slot frees itself instead of shrinking the pool forever."""
    return int(math.ceil(timeout_seconds * 1000)) + 30_000


__all__ = [
    "AcquireResult", "Adjustment", "Backend", "BucketSpec", "CandidateSpec",
    "InProcessBackend", "RedisBackend", "UsageWrite", "lease_ms_for",
]
