"""Usage and cost accounting.

Every attempt - successful or not - becomes a `CallRecord`. Records are
written in the same Redis round trip that settles the quota:

* rollup hashes per project, deployment and minute/hour/day, holding request,
  token and cost counters (cost in micro-dollars, since Redis counts integers);
* an index set of (project, deployment) pairs, so usage queries need no SCAN;
* a capped stream of recent calls, for debugging.

Rollups are what `TokenDaddy.usage()` reads. Anything longer-lived belongs in
an `on_call` sink (Postgres, OTel, ...), which receives every record.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta

from token_daddy.limiter import UsageWrite
from token_daddy.limiter.backend import Backend
from token_daddy.settings import Price
from token_daddy.types import Usage

# granularity -> (bucket seconds, how long rollups are kept in ms)
GRANULARITY = {
    "minute": (60, 2 * 3600 * 1000),
    "hour": (3600, 8 * 86400 * 1000),
    "day": (86400, 90 * 86400 * 1000),
}
STREAM_MAXLEN = 100_000
USAGE_FIELDS = (
    "requests", "errors", "input_tokens", "output_tokens", "reasoning_tokens",
    "cached_input_tokens", "cost_micro_usd", "unpriced",
)


def cost_usd(price: Price | None, usage: Usage) -> float | None:
    """Dollars for one call, or None when no price is configured - an unknown
    price is reported as unknown, never as free."""
    if price is None:
        return None
    cached = usage.cached_input_tokens
    cached_rate = price.cached_input_per_mtok
    if cached_rate is None:
        cached_rate = price.input_per_mtok
    return (
        (usage.input_tokens - cached) * price.input_per_mtok
        + cached * cached_rate
        + usage.output_tokens * price.output_per_mtok
    ) / 1_000_000


@dataclass
class CallRecord:
    project: str | None
    model: str
    deployment: str
    provider: str
    outcome: str  # ok | rate_limited | transient | deployment_error | bad_request
    usage: Usage = field(default_factory=Usage)
    cost_usd: float | None = None
    latency_ms: float = 0.0
    job_id: str | None = None
    error: str = ""
    at: float = field(default_factory=time.time)

    def as_stream_entry(self) -> dict[str, str]:
        flat = asdict(self)
        flat.update(flat.pop("usage"))
        return {k: "" if v is None else str(v) for k, v in flat.items()}


class Tracker:
    def __init__(self, backend: Backend, namespace: str):
        self._backend = backend
        self._ns = namespace

    def _key(self, *parts: str) -> str:
        return ":".join([f"{{{self._ns}}}", *parts])

    def usage_write(self, record: CallRecord) -> UsageWrite:
        project = record.project or "-"
        rollups = []
        for gran, (seconds, ttl_ms) in GRANULARITY.items():
            bucket = int(record.at // seconds)
            rollups.append((self._key("u", project, record.deployment, gran, str(bucket)), ttl_ms))
        u = record.usage
        return UsageWrite(
            rollup_keys=rollups,
            fields={
                "requests": 1,
                "errors": 0 if record.outcome == "ok" else 1,
                "input_tokens": u.input_tokens,
                "output_tokens": u.output_tokens,
                "reasoning_tokens": u.reasoning_tokens,
                "cached_input_tokens": u.cached_input_tokens,
                "cost_micro_usd": round((record.cost_usd or 0) * 1_000_000),
                "unpriced": 1 if record.cost_usd is None and record.outcome == "ok" else 0,
            },
            index=(self._key("u", "idx"), f"{project}|{record.deployment}"),
            stream=(self._key("calls"), STREAM_MAXLEN, record.as_stream_entry()),
        )

    async def usage(
        self,
        *,
        project: str | None = None,
        since: datetime | timedelta = timedelta(hours=24),
        granularity: str = "hour",
    ) -> list[dict]:
        """Totals per (project, deployment) since a point in time, oldest
        bucket included. `project=None` means every project."""
        seconds, _ = GRANULARITY[granularity]
        now = time.time()
        start = (
            now - since.total_seconds() if isinstance(since, timedelta)
            else since.astimezone(UTC).timestamp()
        )
        buckets = [str(b) for b in range(int(start // seconds), int(now // seconds) + 1)]
        pairs = sorted(await self._backend.members(self._key("u", "idx")))
        rows = []
        for pair in pairs:
            proj, _, deployment = pair.partition("|")
            if project is not None and proj != project:
                continue
            hashes = await self._backend.read_hashes(
                [self._key("u", proj, deployment, granularity, b) for b in buckets]
            )
            # Zero counters are never written, so every field starts at 0 here.
            totals: dict[str, int] = dict.fromkeys(USAGE_FIELDS, 0)
            seen = False
            for h in hashes:
                seen = seen or bool(h)
                for k, v in h.items():
                    totals[k] = totals.get(k, 0) + v
            if not seen:
                continue
            micro = totals.pop("cost_micro_usd", 0)
            rows.append({
                "project": None if proj == "-" else proj,
                "deployment": deployment,
                **totals,
                "cost_usd": micro / 1_000_000,
            })
        return rows

    async def recent_calls(self, count: int = 50) -> list[dict[str, str]]:
        return await self._backend.recent(self._key("calls"), count)
