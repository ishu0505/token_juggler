"""The public entry point.

    # Config from a local YAML file:
    tj = TokenJuggler.from_config("tokenjuggler.yaml", project="search-svc")
    # ...or from the central config an admin pushed to the shared Redis:
    tj = await TokenJuggler.connect("redis://quota-redis:6379/0", project="search-svc")

    r = await tj.generate("gpt-5.6-sol", [Text("Summarise"), File.from_path("a.pdf")])
    r.text, r.usage, r.cost_usd, r.deployment

One `TokenJuggler` per process is the intended shape: it owns the SDK clients,
their connection pools and the Redis connection.
"""

from __future__ import annotations

import asyncio
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Self

from pydantic import BaseModel

from tokenjuggler.adapters import default_adapters
from tokenjuggler.central import CentralConfig, ConfigError
from tokenjuggler.limiter import Backend, InProcessBackend, Limiter, RedisBackend
from tokenjuggler.registry import Registry
from tokenjuggler.router import OnCall, Router
from tokenjuggler.settings import Config, load_config, parse_config
from tokenjuggler.tracking import Tracker
from tokenjuggler.types import Part, Request, Response, Text, ThinkingLevel
from tokenjuggler.utils.logger import get_logger

log = get_logger("tokenjuggler")


class TokenJuggler:
    def __init__(
        self,
        config: Config,
        *,
        project: str | None = None,
        backend: Backend | None = None,
        adapters: dict | None = None,
        on_call: OnCall | None = None,
        environ: dict[str, str] | None = None,
    ):
        self.project = project
        self._environ = environ
        self._on_call = on_call
        if backend is None:
            url = config.redis.resolve()
            if url:
                backend = RedisBackend.from_url(url)
            else:
                log.warning(
                    "no Redis configured - quotas are enforced PER PROCESS only. "
                    "Set REDIS_URL wherever more than one process shares an account."
                )
                backend = InProcessBackend()
        self.backend = backend
        self.adapters = adapters or default_adapters(
            timeout_seconds=config.defaults.request_timeout_seconds
        )
        self._native: list[Any] = []
        self._central: CentralConfig | None = None
        self.config_version: int | None = None
        self._refresh_task: asyncio.Task | None = None
        self._apply_config(config)

    def _apply_config(self, config: Config) -> None:
        """(Re)build everything that depends on the config. Swapped in whole,
        so a call already in flight finishes on the config it started with."""
        registry = Registry(config, environ=self._environ)
        limiter = Limiter(registry, self.backend, project=self.project)
        tracker = Tracker(self.backend, config.namespace)
        router = Router(registry, limiter, self.adapters, tracker,
                        on_call=getattr(self, "router", None) and self.router.on_call
                        or self._on_call)
        for adapter in {id(a): a for a in self.adapters.values()}.values():
            # Credentials or endpoints may have changed; SDK clients are rebuilt lazily.
            if hasattr(adapter, "_clients"):
                adapter._clients = {}
            if hasattr(adapter, "_timeout"):
                adapter._timeout = config.defaults.request_timeout_seconds
        self.config, self.registry, self.limiter = config, registry, limiter
        self.tracker, self.router = tracker, router
        if self.project and self.project not in config.projects:
            log.info("project %r has no quota settings; it uses the shared quota", self.project)

    @classmethod
    def from_config(cls, path: str | Path, **kwargs) -> TokenJuggler:
        if not Path(path).exists():
            raise ConfigError(
                f"no config at {path}. Create one with `tokenjuggler init` (a starter file) "
                "or `tokenjuggler ui` (a web editor), or pass the right path."
            )
        return cls(load_config(path), **kwargs)

    @classmethod
    async def connect(
        cls,
        redis_url: str | None = None,
        *,
        namespace: str = "tj",
        project: str | None = None,
        refresh_seconds: float = 30.0,
        **kwargs,
    ) -> TokenJuggler:
        """Load the central config from the shared Redis and keep it current.

        Every `refresh_seconds` the version number is checked (one tiny Redis
        read); when an admin has pushed a new version, it is loaded and swapped
        in without a restart. A new version that fails validation is ignored
        and the running config kept. `refresh_seconds=0` disables reloading.
        """
        import os

        url = redis_url or os.environ.get("REDIS_URL")
        if not url:
            raise ConfigError("connect() needs a Redis URL (argument or REDIS_URL)")
        backend = kwargs.pop("backend", None) or RedisBackend.from_url(url)
        central = CentralConfig(backend._redis, namespace)
        fetched = await central.fetch()
        if fetched is None:
            raise ConfigError(
                f"no config has been pushed to namespace {namespace!r} on this Redis. "
                "An admin publishes one with: tokenjuggler config push tokenjuggler.yaml"
            )
        version, text = fetched
        tj = cls(parse_config(text), project=project, backend=backend, **kwargs)
        tj._central, tj.config_version = central, version
        log.info("loaded central config v%d from namespace %r", version, namespace)
        if refresh_seconds > 0:
            tj._refresh_task = asyncio.create_task(tj._refresh_loop(refresh_seconds))
        return tj

    async def reload_config(self) -> bool:
        """Load the central config now if it changed. True when it did."""
        if self._central is None:
            return False
        if await self._central.version() == self.config_version:
            return False
        fetched = await self._central.fetch()
        if fetched is None:
            return False
        version, text = fetched
        config = parse_config(text)
        if config.namespace != self._central.namespace:
            raise ConfigError(f"central config v{version} is for another namespace")
        self._apply_config(config)
        self.config_version = version
        log.info("switched to central config v%d", version)
        return True

    async def _refresh_loop(self, every: float) -> None:
        while True:
            await asyncio.sleep(every)
            try:
                await self.reload_config()
            except Exception as exc:  # noqa: BLE001 - keep serving on the last good config
                log.error("central config reload failed, keeping v%s: %s",
                          self.config_version, exc)

    # -- unified interface ------------------------------------------------------

    async def generate(
        self,
        model: str,
        parts: str | Part | list[Part],
        *,
        system: str | None = None,
        response_schema: type[BaseModel] | dict | None = None,
        max_output_tokens: int | None = None,
        thinking: ThinkingLevel | None = None,
        temperature: float | None = None,
        job_id: str | None = None,
        max_wait_seconds: float | None = None,
    ) -> Response:
        if isinstance(parts, str):
            parts = [Text(parts)]
        elif not isinstance(parts, list):
            parts = [parts]
        request = Request(
            model=model, parts=parts, system=system, response_schema=response_schema,
            max_output_tokens=max_output_tokens, thinking=thinking,
            temperature=temperature, job_id=job_id,
        )
        return await self.router.generate(request, max_wait_seconds=max_wait_seconds)

    def generate_sync(self, model: str, parts, **kwargs) -> Response:
        """Blocking wrapper for scripts. Runs on a private event loop thread so
        it also works when called from inside another running loop."""
        return _run_sync(self.generate(model, parts, **kwargs))

    # -- native SDK clients -------------------------------------------------------

    def openai(self, model: str | None = None):
        """An `openai.AsyncOpenAI` whose traffic is limited, tracked and
        failed over across every Responses-API deployment."""
        from tokenjuggler.native import openai_client

        return self._keep(openai_client(self, model))

    def genai(self, model: str | None = None):
        """A `google.genai.Client` routed the same way. Use `.aio`."""
        from tokenjuggler.native import genai_client

        return self._keep(genai_client(self, model))

    def anthropic(self, model: str | None = None):
        """An `anthropic.AsyncAnthropic` routed the same way."""
        from tokenjuggler.native import anthropic_client

        return self._keep(anthropic_client(self, model))

    def _keep(self, client):
        self._native.append(client)
        return client

    # -- reporting ------------------------------------------------------------------

    async def usage(
        self,
        *,
        project: str | None = None,
        all_projects: bool = False,
        since: datetime | timedelta = timedelta(hours=24),
        granularity: str = "hour",
    ) -> list[dict]:
        """Requests, tokens and cost per deployment. Defaults to this client's
        project; pass `all_projects=True` for everyone on this Redis."""
        scope = None if all_projects else (project or self.project or "-")
        rows = await self.tracker.usage(project=scope, since=since, granularity=granularity)
        return rows

    async def recent_calls(self, count: int = 50) -> list[dict[str, str]]:
        return await self.tracker.recent_calls(count)

    # -- lifecycle ---------------------------------------------------------------------

    async def aclose(self) -> None:
        if self._refresh_task:
            self._refresh_task.cancel()
        for adapter in {id(a): a for a in self.adapters.values()}.values():
            await adapter.close()
        await self.backend.close()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *exc) -> None:
        await self.aclose()


_loop: asyncio.AbstractEventLoop | None = None
_loop_lock = threading.Lock()


def _run_sync(coro):
    global _loop
    with _loop_lock:
        if _loop is None:
            _loop = asyncio.new_event_loop()
            threading.Thread(target=_loop.run_forever, daemon=True, name="tokenjuggler").start()
    return asyncio.run_coroutine_threadsafe(coro, _loop).result()
