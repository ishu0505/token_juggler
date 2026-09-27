"""Does routing through tokenjuggler add latency?

Part A - live. For each deployment, the SAME request body is sent three ways,
interleaved in random order each round so provider drift hits all modes alike:

  direct  - the raw provider SDK call, nothing else
  tj      - tj.generate() (limiter + adapter + settle + tracking)
  native  - the provider SDK with tokenjuggler's routing transport underneath

Part B - isolated. 300 calls through tj.generate() against a zero-latency fake
provider on the real Redis: everything measured is pure tokenjuggler overhead.

    uv run python experiments/latency_experiment.py [rounds]
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT.parent / ".env")
os.environ.setdefault("REDIS_URL", "redis://localhost:6379/15")

from tokenjuggler import TokenJuggler  # noqa: E402
from tokenjuggler.adapters import AdapterResult, anthropic_messages  # noqa: E402,F401
from tokenjuggler.adapters import genai as genai_adapter  # noqa: E402
from tokenjuggler.adapters import openai_responses  # noqa: E402
from tokenjuggler.settings import WireApi, load_config  # noqa: E402
from tokenjuggler.types import Request, Text, Usage  # noqa: E402

CONFIG = ROOT / "exp3_latency.yaml"
PROMPT = "Name one planet. One word."
MODES = ("direct", "tj", "native")


def stats(samples: list[float]) -> dict:
    s = sorted(samples)
    return {
        "n": len(s),
        "median_ms": round(statistics.median(s), 1),
        "mean_ms": round(statistics.fmean(s), 1),
        "p90_ms": round(s[int(0.9 * (len(s) - 1))], 1),
        "min_ms": round(s[0], 1),
        "max_ms": round(s[-1], 1),
    }


class Deployment:
    """Three ways to make the identical call on one deployment."""

    def __init__(self, tj: TokenJuggler, model: str):
        self.tj, self.model = tj, model
        self.dep = tj.registry.model(model).deployments[0]
        self.request = Request(model=model, parts=[Text(PROMPT)], thinking="low")
        timeout = tj.config.defaults.request_timeout_seconds
        if self.dep.api is WireApi.OPENAI_RESPONSES:
            self.raw = openai_responses.build_client(self.dep, timeout)
            self.native = tj.openai(model)
            self.body = openai_responses.build_request(self.dep, self.request)
        else:
            self.raw = genai_adapter.build_client(self.dep, timeout)
            self.native = tj.genai(model)
            self.body = genai_adapter.build_generate_content(self.dep, self.request)

    async def direct(self) -> dict:
        t = time.perf_counter()
        if self.dep.api is WireApi.OPENAI_RESPONSES:
            await self.raw.responses.create(**self.body)
        else:
            await self.raw.aio.models.generate_content(**self.body)
        return {"total_ms": (time.perf_counter() - t) * 1000}

    async def routed(self) -> dict:
        t = time.perf_counter()
        r = await self.tj.generate(self.model, PROMPT, thinking="low")
        return {"total_ms": (time.perf_counter() - t) * 1000, **{
            f"{k}_ms": v for k, v in r.timings.items() if k != "total"}}

    async def native_call(self) -> dict:
        # Same body; only the model name is the logical one, as a user would write it.
        t = time.perf_counter()
        if self.dep.api is WireApi.OPENAI_RESPONSES:
            await self.native.responses.create(**{**self.body, "model": self.model})
        else:
            await self.native.aio.models.generate_content(**{**self.body, "model": self.model})
        return {"total_ms": (time.perf_counter() - t) * 1000}

    async def run(self, mode: str) -> dict:
        return await {"direct": self.direct, "tj": self.routed, "native": self.native_call}[mode]()


async def part_a(rounds: int) -> dict:
    tj = TokenJuggler.from_config(CONFIG, project="latency-exp")
    stale = await tj.backend._redis.keys("{exp3}*")
    if stale:
        await tj.backend._redis.delete(*stale)
    deployments = [Deployment(tj, m) for m in tj.registry.models]
    samples: dict[str, dict[str, list[dict]]] = {d.model: {m: [] for m in MODES} for d in deployments}

    print("warming up connections (1 call per mode per deployment, not measured)...")
    for d in deployments:
        for mode in MODES:
            await d.run(mode)

    for i in range(1, rounds + 1):
        for d in deployments:
            order = random.sample(MODES, len(MODES))
            for mode in order:
                try:
                    samples[d.model][mode].append(await d.run(mode))
                except Exception as exc:  # keep going; record the failure
                    samples[d.model][mode].append({"error": f"{type(exc).__name__}: {exc}"[:200]})
        print(f"  round {i}/{rounds} done")

    summary = {}
    for model, modes in samples.items():
        ok = {m: [s["total_ms"] for s in v if "total_ms" in s] for m, v in modes.items()}
        entry = {m: stats(v) for m, v in ok.items() if v}
        td_parts = [s for s in modes["tj"] if "overhead_ms" in s]
        entry["td_breakdown_median_ms"] = {
            k: round(statistics.median(s[k] for s in td_parts), 3)
            for k in ("provider_ms", "limiter_ms", "settle_ms", "overhead_ms")
        }
        entry["median_delta_vs_direct_ms"] = {
            m: round(entry[m]["median_ms"] - entry["direct"]["median_ms"], 1)
            for m in ("tj", "native") if m in entry
        }
        entry["errors"] = {m: sum("error" in s for s in v) for m, v in modes.items()}
        summary[model] = entry

    usage = await tj.usage()
    await tj.aclose()
    return {"rounds": rounds, "samples": samples, "summary": summary, "usage_rollup": usage}


class ZeroLatencyProvider:
    async def call(self, dep, request):
        return AdapterResult(text="ok", usage=Usage(10, 2))

    async def close(self):
        pass


async def part_b(n: int = 300) -> dict:
    config = load_config(CONFIG)
    config.namespace = "exp3-overhead"
    fake = ZeroLatencyProvider()
    tj = TokenJuggler(config, adapters={api: fake for api in WireApi}, project="overhead")
    model = next(iter(tj.registry.models))
    for _ in range(20):  # warm the Redis pool and script cache
        await tj.generate(model, PROMPT)
    totals, limiter, settle = [], [], []
    for _ in range(n):
        r = await tj.generate(model, PROMPT)
        totals.append(r.timings["total"])
        limiter.append(r.timings["limiter"])
        settle.append(r.timings["settle"])
    keys = await tj.backend._redis.keys("{exp3-overhead}*")
    if keys:
        await tj.backend._redis.delete(*keys)
    await tj.aclose()
    return {"calls": n, "total": stats(totals), "limiter_acquire": stats(limiter),
            "settle": stats(settle)}


async def main(rounds: int) -> None:
    print("\n== Part A: live, direct vs tj.generate vs native SDK ==")
    a = await part_a(rounds)
    print("\n== Part B: pure tokenjuggler overhead (zero-latency provider, real Redis) ==")
    b = await part_b()
    result = {"run_at": datetime.now().isoformat(timespec="seconds"),
              "redis_url": os.environ["REDIS_URL"], "part_a_live": a, "part_b_overhead": b}
    out = ROOT / "results" / f"exp3_latency_{datetime.now():%Y%m%d_%H%M%S}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(result, indent=2, default=str))

    print(f"\n{'deployment':28} {'mode':7} {'median':>8} {'p90':>8} {'mean':>8}  delta vs direct")
    for model, e in a["summary"].items():
        for mode in MODES:
            if mode in e:
                d = e["median_delta_vs_direct_ms"].get(mode, 0.0)
                print(f"{model:28} {mode:7} {e[mode]['median_ms']:>8} {e[mode]['p90_ms']:>8} "
                      f"{e[mode]['mean_ms']:>8}  {d:+.1f}")
        print(f"{'':28} tj breakdown (median ms): {e['td_breakdown_median_ms']}  errors: {e['errors']}")
    print(f"\nPart B ({b['calls']} calls): total overhead median {b['total']['median_ms']} ms, "
          f"p90 {b['total']['p90_ms']} ms | acquire median {b['limiter_acquire']['median_ms']} ms | "
          f"settle median {b['settle']['median_ms']} ms")
    print(f"saved -> {out.relative_to(ROOT.parent)}")


if __name__ == "__main__":
    asyncio.run(main(int(sys.argv[1]) if len(sys.argv) > 1 else 10))
