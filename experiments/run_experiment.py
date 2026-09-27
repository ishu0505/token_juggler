"""Fire N sequential calls at a model and record where each one landed.

    REDIS_URL=redis://localhost:6379/15 uv run python experiments/run_experiment.py \
        experiments/exp1_gemini38_primary.yaml gemini-3.8-flash 14

Writes experiments/results/<config>_<timestamp>.json: every call (route taken,
every attempt, tokens, cost, latency) plus a summary of when traffic switched
provider, fell back to another model, and was refused.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml
from dotenv import load_dotenv

from tokenjuggler import RoutingError, TokenJuggler

ROOT = Path(__file__).resolve().parent
PROMPTS = [
    "Name one primary colour. One word.",
    "Name one planet. One word.",
    "Name one ocean. One word.",
    "Name one metal. One word.",
]


async def run(config_path: Path, model: str, n_calls: int) -> dict:
    tj = TokenJuggler.from_config(config_path, project="experiments")
    ns = tj.config.namespace
    stale = await tj.backend._redis.keys(f"{{{ns}}}*")  # start from empty buckets
    if stale:
        await tj.backend._redis.delete(*stale)

    primary = tj.registry.model(model)
    first_choice = primary.deployments[0].id
    calls = []
    t0 = time.perf_counter()
    for i in range(1, n_calls + 1):
        prompt = PROMPTS[(i - 1) % len(PROMPTS)]
        started = time.perf_counter()
        entry: dict = {"call": i, "requested_model": model, "prompt": prompt,
                       "at_s": round(started - t0, 2)}
        try:
            r = await tj.generate(model, prompt, thinking="low")
            entry.update({
                "status": "ok",
                "served_model": r.model,
                "deployment": r.deployment,
                "provider": r.provider,
                "switched_provider": r.deployment != first_choice,
                "fell_back_to_other_model": r.model != model,
                "attempts": [a.__dict__ for a in r.attempts],
                "input_tokens": r.usage.input_tokens,
                "output_tokens": r.usage.output_tokens,
                "reasoning_tokens": r.usage.reasoning_tokens,
                "cost_usd": r.cost_usd,
                "latency_ms": round(r.latency_ms),
                "truncated": r.truncated,
                "answer": r.text.strip()[:60],
            })
        except RoutingError as exc:
            entry.update({
                "status": "refused" if type(exc).__name__ == "QuotaExceeded" else "failed",
                "error": f"{type(exc).__name__}: {str(exc)[:300]}",
                "attempts": [a.__dict__ for a in exc.attempts],
                "latency_ms": round((time.perf_counter() - started) * 1000),
            })
        calls.append(entry)
        _print(entry)

    usage = await tj.usage(since=__import__("datetime").timedelta(minutes=30))
    await tj.aclose()
    return {"calls": calls, "summary": _summary(calls, first_choice), "usage_rollup": usage}


def _summary(calls: list[dict], first_choice: str) -> dict:
    ok = [c for c in calls if c["status"] == "ok"]
    per_deployment: dict[str, int] = {}
    for c in ok:
        per_deployment[c["deployment"]] = per_deployment.get(c["deployment"], 0) + 1
    provider_429s = [
        {"call": c["call"], **a} for c in calls for a in c.get("attempts", [])
        if a["outcome"] == "rate_limited"
    ]

    def first(pred):
        return next((c["call"] for c in calls if pred(c)), None)

    return {
        "calls_total": len(calls),
        "calls_ok": len(ok),
        "calls_refused_locally": sum(c["status"] == "refused" for c in calls),
        "calls_failed": sum(c["status"] == "failed" for c in calls),
        "served_per_deployment": per_deployment,
        "calls_on_first_choice_before_switch": sum(1 for c in ok if c["deployment"] == first_choice),
        "first_call_on_another_provider": first(lambda c: c.get("switched_provider")),
        "first_call_on_fallback_model": first(lambda c: c.get("fell_back_to_other_model")),
        "first_refused_call": first(lambda c: c["status"] == "refused"),
        "provider_429s": provider_429s,
        "total_input_tokens": sum(c["input_tokens"] for c in ok),
        "total_output_tokens": sum(c["output_tokens"] for c in ok),
        "total_cost_usd": round(sum(c["cost_usd"] or 0 for c in ok), 6),
        "total_latency_s": round(sum(c["latency_ms"] for c in calls) / 1000, 2),
    }


def _print(c: dict) -> None:
    if c["status"] == "ok":
        tag = "FALLBACK " if c["fell_back_to_other_model"] else ("SWITCHED " if c["switched_provider"] else "")
        print(f"  #{c['call']:>2} {tag:9}{c['deployment']:32} {c['input_tokens']:>4}in "
              f"{c['output_tokens']:>4}out ${c['cost_usd'] or 0:.6f} {c['latency_ms']:>5}ms "
              f"-> {c['answer']!r}")
    else:
        print(f"  #{c['call']:>2} {c['status'].upper():9}{c['error'][:140]}")


def main() -> None:
    config_path, model, n = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3])
    load_dotenv(ROOT.parent / ".env")
    print(f"\n== {config_path.name}: {n} calls to {model} ==")
    result = asyncio.run(run(config_path, model, n))
    result = {
        "experiment": config_path.stem,
        "run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config": yaml.safe_load(config_path.read_text()),
        **result,
    }
    out_dir = ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    out = out_dir / f"{config_path.stem}_{datetime.now():%Y%m%d_%H%M%S}.json"
    out.write_text(json.dumps(result, indent=2, default=str))
    print("\nsummary:", json.dumps(result["summary"], indent=2))
    print(f"saved -> {out.relative_to(ROOT.parent)}")


if __name__ == "__main__":
    os.environ.setdefault("REDIS_URL", "redis://localhost:6379/15")
    main()
