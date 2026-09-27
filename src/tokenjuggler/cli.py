"""`tokenjuggler` command line.

Using a config (local file by default, or the central one with --central):

    tokenjuggler check                 which deployments are routable, and why not
    tokenjuggler projects              how each deployment's quota is split by project
    tokenjuggler verify [-m MODEL]     one tiny live call per deployment (billed!)
    tokenjuggler usage [--all]         requests, tokens, cost and latency per deployment

Creating and editing a config:

    tokenjuggler init [-t TEMPLATE]   write a starter tokenjuggler.yaml (minimal | full | central)
    tokenjuggler ui                   edit it in a local web page (add --central for Redis)

Managing the central config held in Redis (admins):

    tokenjuggler config push FILE      validate and publish a new version
    tokenjuggler config show           the current version's metadata
    tokenjuggler config pull [-o F]    download the current YAML
    tokenjuggler config history        recent versions
    tokenjuggler config rollback N     re-publish version N

A local config comes from --config, else $TOKENJUGGLER_CONFIG, else
./tokenjuggler.yaml. Redis comes from --redis, else $REDIS_URL.
"""

from __future__ import annotations

import argparse
import asyncio
import getpass
import os
import sys
import time
from datetime import timedelta
from pathlib import Path

from tokenjuggler.adapters import AdapterError
from tokenjuggler.central import CentralConfig, ConfigError
from tokenjuggler.client import TokenJuggler
from tokenjuggler.estimate import reservation_cost
from tokenjuggler.limiter import Cost
from tokenjuggler.settings import load_config, parse_config
from tokenjuggler.types import Request, Text


def _config_path(args) -> str:
    return args.config or os.environ.get("TOKENJUGGLER_CONFIG") or "tokenjuggler.yaml"


def _redis_url(args) -> str:
    url = args.redis or os.environ.get("REDIS_URL")
    if not url:
        raise SystemExit("error: no Redis - pass --redis or set REDIS_URL")
    return url


async def _open(args) -> TokenJuggler:
    """The client every command runs on, from the central or a local config."""
    if args.central:
        return await TokenJuggler.connect(
            _redis_url(args), namespace=args.namespace, project=args.project, refresh_seconds=0
        )
    return TokenJuggler(load_config(_config_path(args)), project=args.project)


# -- config consumers ------------------------------------------------------------


async def cmd_check(args) -> int:
    tj = await _open(args)
    source = f"central config v{tj.config_version}" if args.central else _config_path(args)
    print(f"config: {source} (namespace {tj.config.namespace!r})\n")
    width = max((len(d) for d in [*tj.registry.deployments, *tj.registry.unavailable]), default=10)
    for dep_id, dep in sorted(tj.registry.deployments.items()):
        limits = ", ".join(f"{k}={v:,}" for k, v in dep.limits.buckets().items())
        print(f"  ok   {dep_id:<{width}}  {dep.api.value:<24} {limits}")
    for dep_id, reason in sorted(tj.registry.unavailable.items()):
        print(f"  --   {dep_id:<{width}}  {reason}")
    print(f"\n{len(tj.registry.deployments)} routable, {len(tj.registry.unavailable)} not")
    await tj.aclose()
    return 0 if tj.registry.deployments else 1


async def cmd_projects(args) -> int:
    tj = await _open(args)
    projects = tj.config.projects
    if not projects:
        print("no projects configured: every caller shares each deployment's full quota")
        await tj.aclose()
        return 0
    for dep_id in tj.config.deployment_ids():
        reserves = tj.registry.reservations(dep_id)
        pool = 1 - sum(q.reserve for q in reserves.values())
        print(f"{dep_id}")
        if reserves:
            print(f"    shared pool          {pool:>6.0%}  (anyone; unlisted projects use only this)")
        for name in sorted(projects):
            q = projects[name].quota_for(dep_id)
            parts = []
            if q.reserve:
                parts.append(f"reserve {q.reserve:.0%}" + (" (lends when idle)" if q.lend_idle else ""))
            if q.cap:
                parts.append(f"cap {q.cap:.0%}")
            print(f"    {name:<20} {', '.join(parts) or 'no limits of its own'}")
    await tj.aclose()
    return 0


async def cmd_verify(args) -> int:
    tj = await _open(args)
    deployments = [
        d for d in tj.registry.deployments.values() if not args.model or d.model in args.model
    ]
    print(f"probing {len(deployments)} deployments with one small billed call each...\n")
    failures = 0
    request = Request(model="", parts=[Text("Reply with the single word: ok")],
                      max_output_tokens=256)
    for dep in sorted(deployments, key=lambda d: d.id):
        request.model = dep.model
        cost = reservation_cost(request, dep, tj.config.estimation)
        hold, result = await tj.limiter.acquire([dep], [cost])
        if hold is None:
            print(f"  SKIP {dep.id}: no quota right now ({', '.join(result.reasons)})")
            continue
        started = time.perf_counter()
        try:
            out = await tj.adapters[dep.api].call(dep, request)
        except AdapterError as error:
            await tj.limiter.settle(hold, actual=None, sent=error.sent)
            failures += 1
            print(f"  FAIL {dep.id}: {type(error).__name__}: {str(error)[:160]}")
            continue
        await tj.limiter.settle(hold, actual=Cost(out.usage.input_tokens, out.usage.output_tokens))
        ms = (time.perf_counter() - started) * 1000
        print(f"  OK   {dep.id}: {ms:,.0f} ms, {out.usage.input_tokens} in / "
              f"{out.usage.output_tokens} out, said {out.text.strip()[:30]!r}")
    await tj.aclose()
    return 1 if failures else 0


async def cmd_usage(args) -> int:
    tj = await _open(args)
    rows = await tj.usage(all_projects=args.all, since=_parse_since(args.since),
                          granularity=args.granularity)
    await tj.aclose()
    if not rows:
        print("no usage recorded in that window")
        return 0
    print(f"{'project':<14} {'deployment':<36} {'reqs':>7} {'errs':>5} {'input':>11} "
          f"{'output':>10} {'cost $':>10} {'avg ms':>8}")
    for r in sorted(rows, key=lambda r: (r["project"] or "", r["deployment"])):
        cost = f"{r['cost_usd']:.4f}" + ("*" if r.get("unpriced") else "")
        print(f"{r['project'] or '-':<14} {r['deployment']:<36} {r['requests']:>7,} "
              f"{r['errors']:>5,} {r['input_tokens']:>11,} {r['output_tokens']:>10,} "
              f"{cost:>10} {r['avg_latency_ms']:>8,.0f}")
    if any(r.get("unpriced") for r in rows):
        print("\n* some calls had no configured price and are not in the cost")
    return 0


# -- central config management ------------------------------------------------------


def _central(args, namespace: str) -> CentralConfig:
    from tokenjuggler.limiter import RedisBackend

    return CentralConfig(RedisBackend.from_url(_redis_url(args))._redis, namespace)


async def cmd_config(args) -> int:
    try:
        if args.action == "push":
            text = Path(args.file).read_text()
            namespace = parse_config(text).namespace  # validates before connecting
            central = _central(args, namespace)
            before = await central.version()
            version, config = await central.push(
                text, by=args.by or getpass.getuser(), expected_version=args.expect_version,
            )
            if version == before:
                print(f"unchanged: namespace {namespace!r} is already on this config (v{version})")
            else:
                print(f"published v{version} to namespace {namespace!r} "
                      f"({len(config.models)} models, {len(config.accounts)} accounts, "
                      f"{len(config.projects)} projects). Services pick it up within their "
                      "refresh interval.")
            return 0

        central = _central(args, args.namespace)
        if args.action == "show":
            info = await central.info()
            if not info:
                print(f"no config pushed to namespace {args.namespace!r}")
                return 1
            for k, v in info.items():
                print(f"{k:<11} {v}")
        elif args.action == "pull":
            fetched = await central.fetch()
            if not fetched:
                print(f"no config pushed to namespace {args.namespace!r}", file=sys.stderr)
                return 1
            version, text = fetched
            if args.output:
                Path(args.output).write_text(text)
                print(f"wrote v{version} to {args.output}")
            else:
                print(text)
        elif args.action == "history":
            for h in await central.history():
                print(f"v{h['version']:<4} {h['updated_at']}  by {h['updated_by'] or '-':<14} "
                      f"sha256 {h['sha256'][:12]}")
        elif args.action == "rollback":
            version = await central.rollback(args.version, by=args.by or getpass.getuser())
            print(f"re-published v{args.version} as v{version}")
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


async def cmd_init(args) -> int:
    from tokenjuggler.templates import TEMPLATES, read_template

    if args.list:
        for name, desc in TEMPLATES.items():
            print(f"  {name:<8} {desc}")
        return 0
    target = Path(args.output or _config_path(args))
    if target.exists() and not args.force:
        print(f"error: {target} already exists (use --force to overwrite)", file=sys.stderr)
        return 1
    target.write_text(read_template(args.template))
    print(f"wrote the {args.template!r} template to {target}\n"
          f"next: edit it (or run `tokenjuggler ui`), then `tokenjuggler check`")
    return 0


DOCS_URL = "https://github.com/ishu0505/token_juggler/tree/main/src/tokenjuggler/docs"


def welcome(args) -> int:
    """What a bare `tokenjuggler` prints: where you are and what to run next."""
    from tokenjuggler import __version__
    from tokenjuggler.registry import _SDK, sdk_missing

    path = Path(_config_path(args))
    installed = sorted({extra for api, (_, extra) in _SDK.items() if not sdk_missing(api)})
    try:
        import fastapi  # noqa: F401
        ui = True
    except ImportError:
        ui = False
    print(f"tokenjuggler {__version__} - one interface to GPT, Gemini and Claude across providers\n")
    print(f"  config:     {path} " + ("(found)" if path.exists() else "(not created yet)"))
    print(f"  providers:  {', '.join(installed) or 'none installed'}"
          + ("" if len(installed) == 3 else "   (add more: pip install 'tokenjuggler[openai,gemini,anthropic]')"))
    print(f"  web UI:     {'installed' if ui else 'not installed (pip install tokenjuggler[ui])'}\n")
    steps = []
    if not path.exists():
        steps.append(("tokenjuggler ui", "configure in your browser - templates, validation, docs")
                     if ui else ("tokenjuggler init", "write a starter tokenjuggler.yaml"))
    else:
        steps.append(("tokenjuggler ui", "edit the config in your browser") if ui else
                     ("$EDITOR " + str(path), "edit the config"))
    steps += [("tokenjuggler check", "which providers and models are ready"),
              ("tokenjuggler verify", "one tiny live call per deployment"),
              ("tokenjuggler --help", "every command")]
    print("next:")
    for cmd, what in steps:
        print(f"  {cmd:<26} {what}")
    print(f"\ndocs: {DOCS_URL}" + ("  (also in the web UI's Docs tab)" if ui else ""))
    return 0


async def cmd_ui(args) -> int:
    try:
        import uvicorn

        from tokenjuggler.ui import create_app
    except ImportError:
        print("the web UI isn't installed: pip install 'tokenjuggler[ui]'  "
              "(or: uvx --from 'tokenjuggler[ui]' tokenjuggler ui)", file=sys.stderr)
        return 1

    central = None
    path = None if args.central else _config_path(args)
    if args.central:
        central = _central(args, args.namespace)
    app = create_app(config_path=path, central=central)
    if args.host not in ("127.0.0.1", "localhost", "::1"):
        print(f"warning: serving on {args.host} - anyone who can reach it can edit the config "
              "and publish it. The UI has no login.", file=sys.stderr)
    where = f"central config (namespace {args.namespace!r})" if central else path
    print(f"tokenjuggler UI for {where}\n  -> http://{args.host}:{args.port}  (Ctrl+C to stop)")
    server = uvicorn.Server(uvicorn.Config(app, host=args.host, port=args.port, log_level="warning"))
    if args.open:
        async def open_when_ready():
            while not server.started:
                await asyncio.sleep(0.05)
            import webbrowser

            webbrowser.open(f"http://{args.host}:{args.port}")

        asyncio.get_running_loop().create_task(open_when_ready())
    await server.serve()
    return 0


def _parse_since(text: str) -> timedelta:
    unit = text[-1]
    value = float(text[:-1])
    return {"m": timedelta(minutes=value), "h": timedelta(hours=value),
            "d": timedelta(days=value)}[unit]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="tokenjuggler")
    parser.add_argument("--config", help="path to a local tokenjuggler.yaml")
    parser.add_argument("--central", action="store_true",
                        help="use the central config in Redis instead of a local file")
    parser.add_argument("--namespace", default="tj", help="central config namespace (default tj)")
    parser.add_argument("--redis", help="Redis URL (default $REDIS_URL)")
    parser.add_argument("--project", help="project name for quotas and usage")
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("check", help="show routable deployments")
    sub.add_parser("projects", help="show how quotas are split between projects")
    verify = sub.add_parser("verify", help="one tiny live call per deployment")
    verify.add_argument("-m", "--model", action="append", help="only these models")
    usage = sub.add_parser("usage", help="requests, tokens, cost and latency")
    usage.add_argument("--all", action="store_true", help="every project on this Redis")
    usage.add_argument("--since", default="24h", help="e.g. 30m, 24h, 7d")
    usage.add_argument("--granularity", default="hour", choices=["minute", "hour", "day"])

    init = sub.add_parser("init", help="write a starter config from a template")
    init.add_argument("-t", "--template", default="minimal", help="minimal | full | central")
    init.add_argument("-o", "--output", help="file to write (default: the --config path)")
    init.add_argument("--force", action="store_true", help="overwrite an existing file")
    init.add_argument("--list", action="store_true", help="list the templates")
    ui = sub.add_parser("ui", help="edit the config in a local web page")
    ui.add_argument("--port", type=int, default=8765)
    ui.add_argument("--host", default="127.0.0.1")
    ui.add_argument("--no-open", dest="open", action="store_false",
                    help="don't open the browser automatically")

    config = sub.add_parser("config", help="manage the central config in Redis")
    actions = config.add_subparsers(dest="action", required=True)
    push = actions.add_parser("push", help="validate and publish a YAML file")
    push.add_argument("file")
    push.add_argument("--by", help="who is publishing (default: your username)")
    push.add_argument("--expect-version", type=int,
                      help="refuse if the current version isn't this (guards concurrent edits)")
    actions.add_parser("show", help="current version metadata")
    pull = actions.add_parser("pull", help="print or save the current YAML")
    pull.add_argument("-o", "--output")
    actions.add_parser("history", help="recent versions")
    rollback = actions.add_parser("rollback", help="re-publish an earlier version")
    rollback.add_argument("version", type=int)
    rollback.add_argument("--by")

    args = parser.parse_args(argv)
    # Credentials usually live in .env in the working directory; real env vars win.
    from dotenv import load_dotenv

    load_dotenv(os.path.join(os.getcwd(), ".env"), override=False)

    if args.command is None:
        sys.exit(welcome(args))
    commands = {"check": cmd_check, "projects": cmd_projects, "verify": cmd_verify,
                "usage": cmd_usage, "config": cmd_config, "init": cmd_init, "ui": cmd_ui}
    try:
        sys.exit(asyncio.run(commands[args.command](args)))
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
