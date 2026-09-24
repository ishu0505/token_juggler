"""`token-daddy` command line.

    token-daddy check            which deployments are routable, and why not
    token-daddy verify [-m M]    one tiny live call per deployment (billed!)
    token-daddy usage [--all]    requests, tokens and cost per deployment

The config comes from --config, else $TOKEN_DADDY_CONFIG, else ./token_daddy.yaml.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
from datetime import timedelta

from token_daddy.adapters import AdapterError
from token_daddy.client import TokenDaddy
from token_daddy.estimate import reservation_cost
from token_daddy.limiter import Cost
from token_daddy.settings import load_config
from token_daddy.types import Request, Text


def _config_path(args) -> str:
    return args.config or os.environ.get("TOKEN_DADDY_CONFIG") or "token_daddy.yaml"


def cmd_check(args) -> int:
    td = TokenDaddy(load_config(_config_path(args)))
    width = max((len(d) for d in [*td.registry.deployments, *td.registry.unavailable]), default=10)
    for dep_id, dep in sorted(td.registry.deployments.items()):
        limits = ", ".join(f"{k}={v:,}" for k, v in dep.limits.buckets().items())
        print(f"  ok   {dep_id:<{width}}  {dep.api.value:<24} {limits}")
    for dep_id, reason in sorted(td.registry.unavailable.items()):
        print(f"  --   {dep_id:<{width}}  {reason}")
    print(f"\n{len(td.registry.deployments)} routable, {len(td.registry.unavailable)} not")
    return 0 if td.registry.deployments else 1


async def _verify(args) -> int:
    td = TokenDaddy(load_config(_config_path(args)), project=args.project)
    deployments = [
        d for d in td.registry.deployments.values()
        if not args.model or d.model in args.model
    ]
    print(f"probing {len(deployments)} deployments with one small billed call each...\n")
    failures = 0
    request = Request(model="", parts=[Text("Reply with the single word: ok")],
                      max_output_tokens=256)
    for dep in sorted(deployments, key=lambda d: d.id):
        request.model = dep.model
        cost = reservation_cost(request, dep, td.config.estimation)
        hold, result = await td.limiter.acquire([dep], [cost])
        if hold is None:
            print(f"  SKIP {dep.id}: no quota right now ({', '.join(result.reasons)})")
            continue
        started = time.perf_counter()
        try:
            out = await td.adapters[dep.api].call(dep, request)
        except AdapterError as error:
            await td.limiter.settle(hold, actual=None, sent=error.sent)
            failures += 1
            print(f"  FAIL {dep.id}: {type(error).__name__}: {str(error)[:160]}")
            continue
        await td.limiter.settle(hold, actual=Cost(out.usage.input_tokens, out.usage.output_tokens))
        ms = (time.perf_counter() - started) * 1000
        print(f"  OK   {dep.id}: {ms:,.0f} ms, {out.usage.input_tokens} in / "
              f"{out.usage.output_tokens} out, said {out.text.strip()[:30]!r}")
    await td.aclose()
    return 1 if failures else 0


async def _usage(args) -> int:
    td = TokenDaddy(load_config(_config_path(args)), project=args.project)
    rows = await td.usage(all_projects=args.all, since=_parse_since(args.since),
                          granularity=args.granularity)
    await td.aclose()
    if not rows:
        print("no usage recorded in that window")
        return 0
    print(f"{'project':<14} {'deployment':<36} {'reqs':>7} {'errs':>5} {'input':>11} "
          f"{'output':>10} {'cost $':>10}")
    for r in sorted(rows, key=lambda r: (r["project"] or "", r["deployment"])):
        cost = f"{r['cost_usd']:.4f}" + ("*" if r.get("unpriced") else "")
        print(f"{r['project'] or '-':<14} {r['deployment']:<36} {r['requests']:>7,} "
              f"{r['errors']:>5,} {r['input_tokens']:>11,} {r['output_tokens']:>10,} {cost:>10}")
    if any(r.get("unpriced") for r in rows):
        print("\n* some calls had no configured price and are not in the cost")
    return 0


def _parse_since(text: str) -> timedelta:
    unit = text[-1]
    value = float(text[:-1])
    return {"m": timedelta(minutes=value), "h": timedelta(hours=value),
            "d": timedelta(days=value)}[unit]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="token-daddy")
    parser.add_argument("--config", help="path to token_daddy.yaml")
    parser.add_argument("--project", help="project name for quota shares and usage")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("check", help="show routable deployments")
    verify = sub.add_parser("verify", help="one tiny live call per deployment")
    verify.add_argument("-m", "--model", action="append", help="only these models")
    usage = sub.add_parser("usage", help="requests, tokens and cost")
    usage.add_argument("--all", action="store_true", help="every project on this Redis")
    usage.add_argument("--since", default="24h", help="e.g. 30m, 24h, 7d")
    usage.add_argument("--granularity", default="hour", choices=["minute", "hour", "day"])
    args = parser.parse_args(argv)
    # Credentials usually live in .env next to the config; real env vars win.
    from dotenv import load_dotenv

    load_dotenv(override=False)

    if args.command == "check":
        sys.exit(cmd_check(args))
    runner = {"verify": _verify, "usage": _usage}[args.command]
    sys.exit(asyncio.run(runner(args)))


if __name__ == "__main__":
    main()
