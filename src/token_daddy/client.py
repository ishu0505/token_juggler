"""The public entry point.

    td = TokenDaddy.from_config("token_daddy.yaml", project="search-svc")
    r = await td.generate("gpt-5.6-sol", [Text("Summarise"), File.from_path("a.pdf")])
    r.text, r.usage, r.cost_usd, r.deployment

One `TokenDaddy` per process is the intended shape: it owns the SDK clients,
their connection pools and the Redis connection.
"""

from __future__ import annotations

import asyncio
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from token_daddy.adapters import default_adapters
from token_daddy.limiter import Backend, InProcessBackend, Limiter, RedisBackend
from token_daddy.registry import Registry
from token_daddy.router import OnCall, Router
from token_daddy.settings import Config, load_config
from token_daddy.tracking import Tracker
from token_daddy.types import Part, Request, Response, Text, ThinkingLevel
from token_daddy.utils.logger import get_logger

log = get_logger("token_daddy")


class TokenDaddy:
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
        self.config = config
        self.project = project
        self.registry = Registry(config, environ=environ)
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
        self.limiter = Limiter(self.registry, backend, project=project)
        self.tracker = Tracker(backend, config.namespace)
        self.adapters = adapters or default_adapters(
            timeout_seconds=config.defaults.request_timeout_seconds
        )
        self.router = Router(self.registry, self.limiter, self.adapters, self.tracker,
                             on_call=on_call)
        self._native: list[Any] = []

    @classmethod
    def from_config(cls, path: str | Path, **kwargs) -> TokenDaddy:
        return cls(load_config(path), **kwargs)

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
        from token_daddy.native import openai_client

        return self._keep(openai_client(self, model))

    def genai(self, model: str | None = None):
        """A `google.genai.Client` routed the same way. Use `.aio`."""
        from token_daddy.native import genai_client

        return self._keep(genai_client(self, model))

    def anthropic(self, model: str | None = None):
        """An `anthropic.AsyncAnthropic` routed the same way."""
        from token_daddy.native import anthropic_client

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
        for adapter in {id(a): a for a in self.adapters.values()}.values():
            await adapter.close()
        await self.backend.close()

    async def __aenter__(self) -> TokenDaddy:
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
            threading.Thread(target=_loop.run_forever, daemon=True, name="token-daddy").start()
    return asyncio.run_coroutine_threadsafe(coro, _loop).result()
