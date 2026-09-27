"""A local web UI for building and checking a tokenjuggler config.

    tokenjuggler ui                       # edits ./tokenjuggler.yaml
    tokenjuggler ui --config other.yaml
    tokenjuggler --central ui             # edits the central config in Redis

The page edits the same YAML you could edit by hand. Every change is validated
by the library's own config schema, so the UI can't produce a file the library
would reject. It binds to 127.0.0.1 by default: it can write the config file
and publish to Redis, and has no login of its own.

Secrets are never shown: for each env var an account names, the UI only says
whether it is set in the environment the UI was started in.
"""

from __future__ import annotations

import logging
import os
from importlib import resources
from pathlib import Path
from typing import Any

import yaml
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, ValidationError

from tokenjuggler.central import CentralConfig, ConfigError
from tokenjuggler.registry import Registry
from tokenjuggler.settings import (
    LIMIT_KINDS,
    Config,
    Family,
    Provider,
    Reservation,
    Strategy,
    Window,
    WireApi,
)
from tokenjuggler.templates import TEMPLATES, read_template
from tokenjuggler.types import Capability

# Which account fields each provider uses, and the env var names we suggest.
PROVIDER_FIELDS: dict[str, dict[str, str]] = {
    "openai": {"api_key_env": "OPENAI_API_KEY", "base_url": ""},
    "databricks": {"host_env": "DATABRICKS_HOST", "token_env": "DATABRICKS_TOKEN", "base_url": ""},
    "bedrock": {"api_key_env": "BEDROCK_API_KEY", "region": "us-east-1", "base_url": ""},
    "google_ai_studio": {"api_key_env": "GOOGLE_AI_STUDIO_API_KEY", "base_url": ""},
    "vertex": {"project_env": "VERTEX_PROJECT_ID", "credentials_json_env": "", "location": "global"},
    "anthropic": {"api_key_env": "ANTHROPIC_API_KEY", "base_url": ""},
}
ENV_FIELDS = ("api_key_env", "host_env", "token_env", "project_env", "credentials_json_env")


# The guides, in reading order: (file, title shown in the UI).
DOCS = [
    ("README", "Start here"),
    ("single-project", "One project"),
    ("app-developer", "Developer on a shared setup"),
    ("platform-admin", "Platform admin"),
    ("architecture", "Architecture"),
    ("configuration", "Config reference"),
]


def render_doc(slug: str) -> str:
    import markdown

    text = resources.files("tokenjuggler.docs").joinpath(f"{slug}.md").read_text()
    return markdown.markdown(text, extensions=["tables", "fenced_code", "toc", "sane_lists"])


class RawBody(BaseModel):
    raw: dict[str, Any]


class YamlBody(BaseModel):
    yaml: str


class SaveBody(BaseModel):
    yaml: str


class PushBody(BaseModel):
    yaml: str
    expected_version: int | None = None
    by: str = "ui"


def to_yaml(raw: dict) -> str:
    return yaml.safe_dump(raw, sort_keys=False, allow_unicode=True, width=100)


def analyze(raw: dict, environ: dict[str, str] | None = None) -> dict:
    """Validate a config dict and describe what it would do."""
    environ = os.environ if environ is None else environ
    out: dict[str, Any] = {"yaml": to_yaml(raw)}
    try:
        config = Config.model_validate(raw)
    except ValidationError as exc:
        out["ok"] = False
        out["errors"] = [
            {"loc": ".".join(str(p) for p in err["loc"]) or "(config)",
             "msg": err["msg"].removeprefix("Value error, ")}
            for err in exc.errors()
        ]
        return out

    logging.getLogger("tokenjuggler.registry").disabled = True  # the UI shows it instead
    registry = Registry(config, environ=environ)
    out["ok"] = True
    out["errors"] = []
    out["routable"] = [
        {"id": d.id, "api": d.api.value, "model_id": d.model_id,
         "limits": d.limits.buckets(), "max_concurrent": d.limits.max_concurrent}
        for d in registry.deployments.values()
    ]
    out["unavailable"] = registry.unavailable
    out["env"] = {
        name: {field: {"var": var, "set": bool(environ.get(var))}
               for field in ENV_FIELDS if (var := getattr(acct, field))}
        for name, acct in config.accounts.items()
    }
    split = []
    for dep_id in config.deployment_ids():
        reserves = registry.reservations(dep_id)
        rows = []
        for pname, pcfg in config.projects.items():
            q = pcfg.quota_for(dep_id)
            if q.cap or q.reserve:
                rows.append({"project": pname, "cap": q.cap, "reserve": q.reserve,
                             "lend_idle": bool(q.lend_idle)})
        if rows:
            split.append({"deployment": dep_id,
                          "pool": 1 - sum(q.reserve for q in reserves.values()),
                          "projects": rows})
    out["projects_split"] = split
    out["deployment_ids"] = config.deployment_ids()
    return out


def parse_yaml(text: str) -> dict:
    try:
        raw = yaml.safe_load(text) or {}
    except yaml.YAMLError as exc:
        raise HTTPException(422, f"YAML syntax error: {exc}") from exc
    if not isinstance(raw, dict):
        raise HTTPException(422, "the YAML must be a mapping at the top level")
    return raw


def create_app(
    *,
    config_path: str | None = None,
    central: CentralConfig | None = None,
    environ: dict[str, str] | None = None,
) -> FastAPI:
    """`config_path` is the one file the UI may write. With `central`, the UI
    loads from and publishes to Redis instead (and can still save a copy)."""
    app = FastAPI(title="tokenjuggler config", docs_url=None, redoc_url=None)
    path = Path(config_path).resolve() if config_path else None

    @app.get("/", response_class=HTMLResponse)
    async def index() -> str:
        return resources.files("tokenjuggler.ui").joinpath("static/index.html").read_text()

    @app.get("/api/meta")
    async def meta() -> dict:
        source: dict[str, Any] = {"mode": "central" if central else "file",
                                  "path": str(path) if path else None}
        if central:
            info = await central.info()
            source.update(namespace=central.namespace,
                          version=int(info["version"]) if info else None)
        return {
            "source": source,
            "enums": {
                "provider": [p.value for p in Provider],
                "family": [f.value for f in Family],
                "capability": [c.value for c in Capability],
                "strategy": [s.value for s in Strategy],
                "reservation": [r.value for r in Reservation],
                "window": [w.value for w in Window],
                "api": [a.value for a in WireApi],
                "limit": [*LIMIT_KINDS, "max_concurrent"],
            },
            "limit_help": {k: f"{kind} per {({1: 'second', 60: 'minute', 3600: 'hour', 86400: 'day'})[w]}"
                           for k, (w, kind) in LIMIT_KINDS.items()} | {"max_concurrent": "requests in flight"},
            "provider_fields": PROVIDER_FIELDS,
            "templates": TEMPLATES,
        }

    @app.get("/api/docs")
    async def docs_index() -> list[dict]:
        return [{"slug": slug, "title": title} for slug, title in DOCS]

    @app.get("/api/docs/{slug}")
    async def doc(slug: str) -> dict:
        if slug not in dict(DOCS):
            raise HTTPException(404, f"no doc {slug!r}")
        return {"slug": slug, "html": render_doc(slug)}

    @app.get("/api/config")
    async def load() -> dict:
        if central:
            fetched = await central.fetch()
            if fetched:
                version, text = fetched
                return {"yaml": text, "raw": parse_yaml(text), "version": version, "exists": True}
            return {"yaml": "", "raw": None, "version": None, "exists": False}
        if path and path.exists():
            text = path.read_text()
            return {"yaml": text, "raw": parse_yaml(text), "exists": True}
        return {"yaml": "", "raw": None, "exists": False}

    @app.get("/api/template/{name}")
    async def template(name: str) -> dict:
        try:
            text = read_template(name)
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc
        return {"yaml": text, "raw": parse_yaml(text)}

    @app.post("/api/analyze")
    async def analyze_raw(body: RawBody) -> dict:
        return analyze(body.raw, environ)

    @app.post("/api/parse")
    async def parse(body: YamlBody) -> dict:
        raw = parse_yaml(body.yaml)
        return {"raw": raw, **analyze(raw, environ)}

    @app.post("/api/save")
    async def save(body: SaveBody) -> dict:
        if path is None:
            raise HTTPException(400, "no config file was given to `tokenjuggler ui`; use --config")
        result = analyze(parse_yaml(body.yaml), environ)
        if not result["ok"]:
            raise HTTPException(422, "the config has errors; fix them before saving")
        backup = None
        if path.exists():
            backup = path.with_suffix(path.suffix + ".bak")
            backup.write_text(path.read_text())
        path.write_text(body.yaml)
        return {"path": str(path), "backup": str(backup) if backup else None}

    @app.post("/api/push")
    async def push(body: PushBody) -> dict:
        if central is None:
            raise HTTPException(400, "not connected to a central Redis; start with `tokenjuggler --central ui`")
        try:
            before = await central.version()
            version, _ = await central.push(body.yaml, by=body.by,
                                            expected_version=body.expected_version)
        except ValidationError as exc:
            raise HTTPException(422, f"invalid config: {exc.errors()[0]['msg']}") from exc
        except ConfigError as exc:
            raise HTTPException(409, str(exc)) from exc
        return {"version": version, "changed": version != before}

    return app
