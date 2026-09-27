"""An HTTP transport that routes a native SDK's calls through tokenjuggler.

The SDK builds its request exactly as it normally would - pointed at a
placeholder host - and hands it to this transport, which:

1. recognises generation calls (Responses create, Messages create,
   generateContent, Interactions create) and reads the logical model;
2. estimates the cost from the body and reserves quota on the best
   deployment that speaks the same wire API;
3. rewrites URL, auth and model id for that deployment and sends it;
4. reads `usage` from the response and settles; on a 429, 5xx, timeout or
   route error it moves on to the next deployment instead.

Anything else (listing models, retrieving a stored response, ...) goes to the
first deployment untouched. openai and anthropic are built on httpx2 while
google-genai is on httpx, so the logic lives in `RoutingCore` and two thin
transport classes adapt it to each library.
"""

from __future__ import annotations

import asyncio
import json
import random
import re
import time
from typing import Any
from urllib.parse import urlsplit

from tokenjuggler.estimate import estimate_body_input_tokens
from tokenjuggler.limiter import Cost
from tokenjuggler.registry import Deployment, Model
from tokenjuggler.settings import Provider, WireApi
from tokenjuggler.tracking import CallRecord, cost_usd
from tokenjuggler.types import Usage
from tokenjuggler.utils.logger import get_logger

log = get_logger("tokenjuggler.native")

_GENAI_MODEL_PATH = re.compile(r"models/([^/:]+):(generateContent|streamGenerateContent)$")
AI_STUDIO = "https://generativelanguage.googleapis.com"
ANTHROPIC = "https://api.anthropic.com"
_DROP_HEADERS = {"host", "content-length", "authorization", "x-api-key", "x-goog-api-key"}
_DROP_RESPONSE_HEADERS = {"content-encoding", "content-length", "transfer-encoding"}


class RoutingCore:
    def __init__(self, tj, default_model: str | None):
        self.tj = tj
        self.default_model = default_model
        self._vertex_creds: dict[str, Any] = {}  # account -> google credentials

    # -- recognising a generation call -----------------------------------------

    def classify(self, method: str, path: str) -> tuple[WireApi, str | None] | None:
        """(wire api, model from the URL) for a generation call, else None."""
        if method != "POST":
            return None
        if path.endswith("/responses"):
            return WireApi.OPENAI_RESPONSES, None
        if path.endswith("/v1/messages"):
            return WireApi.ANTHROPIC_MESSAGES, None
        if path.endswith("/interactions"):
            return WireApi.GENAI_INTERACTIONS, None
        if match := _GENAI_MODEL_PATH.search(path):
            return WireApi.GENAI_GENERATE_CONTENT, match.group(1)
        return None

    def resolve_model(self, name: str | None) -> Model:
        registry = self.tj.registry
        if name and name in registry.models:
            return registry.models[name]
        if name:  # a provider model id instead of our logical name
            for dep in registry.deployments.values():
                if dep.model_id == name:
                    return registry.models[dep.model]
        if self.default_model:
            return registry.model(self.default_model)
        raise KeyError(f"unknown model {name!r} and no default model for this client")

    # -- the routed path ----------------------------------------------------------

    async def route(self, lib, request, send) -> Any:
        """`lib` is httpx or httpx2; `send` sends a lib.Request upstream."""
        path = urlsplit(str(request.url)).path
        kind = self.classify(request.method, path)
        if kind is None:
            return await self._passthrough(lib, request, send)
        api, url_model = kind
        body = json.loads(await request.aread() or b"{}")
        model = self.resolve_model(url_model or body.get("model"))
        candidates = [d for d in model.deployments if d.api is api]
        if not candidates:
            raise RuntimeError(f"no deployment of {model.name} speaks {api.value}")
        streaming = bool(body.get("stream")) or (url_model is not None and "stream" in path)

        cfg = self.tj.config
        input_tokens = {
            fam: estimate_body_input_tokens(body, fam, cfg.estimation)
            for fam in {d.family for d in candidates}
        }
        deadline = time.monotonic() + cfg.defaults.max_wait_seconds
        tried: set[str] = set()
        last: Any = None
        limiter = self.tj.limiter
        while True:
            pool = [d for d in candidates if d.id not in tried]
            if not pool:
                break
            costs = [
                Cost(input_tokens[d.family], _output_cap(api, body, d)) for d in pool
            ]
            hold, result = await limiter.acquire(
                pool, costs, strategy=model.routing.strategy, rr_scope=model.name
            )
            if hold is None:
                remaining = deadline - time.monotonic()
                if result.wait_ms is None or result.wait_ms / 1000 > remaining:
                    break
                await asyncio.sleep(result.wait_ms / 1000 + random.uniform(0, 0.05))
                continue
            dep = hold.deployment
            tried.add(dep.id)
            outbound = await self._rewrite(lib, request, api, body, dep, url_model, path)
            started = time.perf_counter()
            try:
                response = await send(outbound)
            except Exception as error:  # noqa: BLE001 - any network failure means: next route
                await self._settle_failure(hold, dep, "transient", str(error), started)
                last = error
                continue
            status = response.status_code
            if status == 200 and streaming:
                # Streamed usage arrives in the last event; Mark 1 settles at
                # the reservation instead of parsing the stream.
                await self._settle_success(hold, dep, Usage(hold.cost.input_tokens,
                                                             hold.cost.output_tokens), started)
                return response
            content = await response.aread()
            if status == 200:
                usage = _usage(api, json.loads(content or b"{}"))
                await self._settle_success(hold, dep, usage, started)
                return _replay(lib, response, content, request)
            outcome = _outcome(status)
            await self._settle_failure(hold, dep, outcome, f"HTTP {status}", started)
            if outcome == "bad_request":
                return _replay(lib, response, content, request)  # the SDK raises it
            if outcome == "rate_limited":
                await limiter.cooldown(dep, _retry_after(response) or cfg.defaults.cooldown_seconds)
            elif outcome == "deployment_error":
                await limiter.cooldown(dep, cfg.defaults.error_cooldown_seconds)
            last = _replay(lib, response, content, request)

        if isinstance(last, Exception):
            raise last
        if last is not None:
            return last  # every route failed: the SDK raises the last error
        return lib.Response(
            429, json={"error": {"message": f"tokenjuggler: every route for {model.name} is at quota"}},
            request=request,
        )

    async def _passthrough(self, lib, request, send):
        model = self.resolve_model(self.default_model)
        dep = model.deployments[0]
        outbound = await self._rewrite(lib, request, dep.api, None, dep, None,
                                       urlsplit(str(request.url)).path)
        return await send(outbound)

    # -- request rewriting ----------------------------------------------------------

    async def _rewrite(self, lib, request, api, body, dep: Deployment, url_model, path):
        headers = {k: v for k, v in request.headers.items() if k.lower() not in _DROP_HEADERS}
        query = urlsplit(str(request.url)).query
        creds = dep.credentials

        if api is WireApi.OPENAI_RESPONSES:
            url = f"{dep.base_url}{_suffix_after(path, '/v1')}"
            headers["authorization"] = f"Bearer {creds.api_key or creds.bearer_token}"
        elif api is WireApi.ANTHROPIC_MESSAGES:
            url = f"{dep.base_url or ANTHROPIC}{_suffix_from(path, '/v1/')}"
            if dep.provider is Provider.DATABRICKS:
                headers["authorization"] = f"Bearer {creds.bearer_token}"
            else:
                headers["x-api-key"] = creds.api_key
        elif api is WireApi.GENAI_INTERACTIONS:
            url = f"{dep.base_url or AI_STUDIO}{_suffix_from(path, '/v1')}"
            headers["x-goog-api-key"] = creds.api_key
        else:  # generateContent
            method = path.rsplit(":", 1)[-1] if url_model else ""
            url = await self._generate_content_url(dep, method, headers)

        if body is not None:
            body = dict(body)
            if url_model is None:
                body["model"] = dep.model_id
            _inject_output_cap(api, body, dep)
            if api is WireApi.OPENAI_RESPONSES or api is WireApi.ANTHROPIC_MESSAGES:
                body.update(dep.extra_params)
            content = json.dumps(body).encode()
        else:
            content = await request.aread()
        if query:
            url = f"{url}?{query}"
        return lib.Request(request.method, url, headers=headers, content=content,
                           extensions=request.extensions)

    async def _generate_content_url(self, dep: Deployment, method: str, headers: dict) -> str:
        mid = dep.model_id
        if dep.provider is Provider.DATABRICKS:
            headers["authorization"] = f"Bearer {dep.credentials.bearer_token}"
            return f"{dep.base_url}/v1beta/models/{mid}:{method}"
        if dep.provider is Provider.VERTEX:
            headers["authorization"] = f"Bearer {await self._vertex_token(dep)}"
            loc, project = dep.credentials.location, dep.credentials.project
            host = "aiplatform.googleapis.com" if loc == "global" else f"{loc}-aiplatform.googleapis.com"
            return (f"https://{host}/v1beta1/projects/{project}/locations/{loc}"
                    f"/publishers/google/models/{mid}:{method}")
        headers["x-goog-api-key"] = dep.credentials.api_key
        return f"{dep.base_url or AI_STUDIO}/v1beta/models/{mid}:{method}"

    async def _vertex_token(self, dep: Deployment) -> str:
        """An access token for the account's key (or ADC), refreshed off the
        event loop when stale."""
        import google.auth
        import google.auth.transport.requests

        from tokenjuggler.adapters.genai import VERTEX_SCOPES, vertex_credentials

        if dep.account not in self._vertex_creds:
            creds = vertex_credentials(dep)
            if creds is None:
                creds, _ = await asyncio.to_thread(google.auth.default, scopes=VERTEX_SCOPES)
            self._vertex_creds[dep.account] = creds
        creds = self._vertex_creds[dep.account]
        if not creds.valid:
            await asyncio.to_thread(creds.refresh, google.auth.transport.requests.Request())
        return creds.token

    # -- settling -------------------------------------------------------------------

    async def _settle_success(self, hold, dep, usage: Usage, started: float) -> None:
        cost = cost_usd(dep.price, usage)
        record = CallRecord(
            project=self.tj.project, model=dep.model, deployment=dep.id,
            provider=dep.provider.value, outcome="ok", usage=usage, cost_usd=cost,
            latency_ms=(time.perf_counter() - started) * 1000,
        )
        await self.tj.limiter.settle(
            hold, actual=Cost(usage.input_tokens, usage.output_tokens),
            usage=self.tj.tracker.usage_write(record),
        )
        await self.tj.router._emit(record)

    async def _settle_failure(self, hold, dep, outcome: str, error: str, started: float) -> None:
        record = CallRecord(
            project=self.tj.project, model=dep.model, deployment=dep.id,
            provider=dep.provider.value, outcome=outcome, error=error,
            latency_ms=(time.perf_counter() - started) * 1000,
        )
        await self.tj.limiter.settle(hold, actual=None, sent=True,
                                     usage=self.tj.tracker.usage_write(record))
        await self.tj.router._emit(record)
        log.warning("%s failed (%s: %s); trying the next route", dep.id, outcome, error)


# -- per-API body details ----------------------------------------------------------


def _output_cap(api: WireApi, body: dict, dep: Deployment) -> int:
    match api:
        case WireApi.OPENAI_RESPONSES:
            value = body.get("max_output_tokens")
        case WireApi.ANTHROPIC_MESSAGES:
            value = body.get("max_tokens")
        case WireApi.GENAI_GENERATE_CONTENT:
            gen = body.get("generationConfig") or body.get("generation_config") or {}
            value = gen.get("maxOutputTokens") or gen.get("max_output_tokens")
        case _:
            value = (body.get("generation_config") or {}).get("max_output_tokens")
    return int(value) if value else dep.max_output_tokens


def _inject_output_cap(api: WireApi, body: dict, dep: Deployment) -> None:
    """Send the cap we reserved for. Without it a provider's own default -
    often far larger - would let output exceed the reservation."""
    cap = _output_cap(api, body, dep)
    match api:
        case WireApi.OPENAI_RESPONSES:
            body["max_output_tokens"] = cap
        case WireApi.ANTHROPIC_MESSAGES:
            body["max_tokens"] = cap
        case WireApi.GENAI_GENERATE_CONTENT:
            key = "generationConfig" if "generation_config" not in body else "generation_config"
            gen = dict(body.get(key) or {})
            if not (gen.get("maxOutputTokens") or gen.get("max_output_tokens")):
                gen["maxOutputTokens"] = cap
            body[key] = gen
        case WireApi.GENAI_INTERACTIONS:
            gen = dict(body.get("generation_config") or {})
            gen["max_output_tokens"] = cap
            body["generation_config"] = gen


def _usage(api: WireApi, data: dict) -> Usage:
    match api:
        case WireApi.OPENAI_RESPONSES:
            u = data.get("usage") or {}
            return Usage(
                input_tokens=u.get("input_tokens", 0),
                output_tokens=u.get("output_tokens", 0),
                reasoning_tokens=(u.get("output_tokens_details") or {}).get("reasoning_tokens", 0),
                cached_input_tokens=(u.get("input_tokens_details") or {}).get("cached_tokens", 0),
            )
        case WireApi.ANTHROPIC_MESSAGES:
            u = data.get("usage") or {}
            read = u.get("cache_read_input_tokens") or 0
            write = u.get("cache_creation_input_tokens") or 0
            return Usage(u.get("input_tokens", 0) + read + write, u.get("output_tokens", 0),
                         cached_input_tokens=read)
        case WireApi.GENAI_GENERATE_CONTENT:
            u = data.get("usageMetadata") or {}
            thought = u.get("thoughtsTokenCount", 0)
            return Usage(u.get("promptTokenCount", 0), u.get("candidatesTokenCount", 0) + thought,
                         reasoning_tokens=thought,
                         cached_input_tokens=u.get("cachedContentTokenCount", 0))
        case _:
            u = data.get("usage") or {}
            thought = u.get("total_thought_tokens", 0)
            return Usage(u.get("total_input_tokens", 0), u.get("total_output_tokens", 0) + thought,
                         reasoning_tokens=thought, cached_input_tokens=u.get("total_cached_tokens", 0))


def _outcome(status: int) -> str:
    if status == 429:
        return "rate_limited"
    if status in (401, 403, 404):
        return "deployment_error"
    if status in (408, 409) or status >= 500:
        return "transient"
    return "bad_request"


def _retry_after(response) -> float | None:
    try:
        if ms := response.headers.get("retry-after-ms"):
            return float(ms) / 1000
        if seconds := response.headers.get("retry-after"):
            return float(seconds)
    except ValueError:
        return None
    return None


def _replay(lib, response, content: bytes, request):
    """A fresh response carrying the already-read (and already-decoded) body."""
    headers = {k: v for k, v in response.headers.items()
               if k.lower() not in _DROP_RESPONSE_HEADERS}
    return lib.Response(response.status_code, headers=headers, content=content, request=request)


def _suffix_after(path: str, marker: str) -> str:
    """'/v1/responses' -> '/responses' (base URLs already end in /v1)."""
    index = path.find(marker)
    return path[index + len(marker):] if index >= 0 else path


def _suffix_from(path: str, marker: str) -> str:
    index = path.find(marker)
    return path[index:] if index >= 0 else path


# -- the two library-specific transports ---------------------------------------------


def httpx2_transport(core: RoutingCore):
    import httpx2

    class _Transport(httpx2.AsyncBaseTransport):
        def __init__(self):
            self._upstream = httpx2.AsyncHTTPTransport()

        async def handle_async_request(self, request):
            return await core.route(httpx2, request, self._upstream.handle_async_request)

        async def aclose(self):
            await self._upstream.aclose()

    return _Transport()


def httpx_transport(core: RoutingCore):
    import httpx

    class _Transport(httpx.AsyncBaseTransport):
        def __init__(self):
            self._upstream = httpx.AsyncHTTPTransport()

        async def handle_async_request(self, request):
            return await core.route(httpx, request, self._upstream.handle_async_request)

        async def aclose(self):
            await self._upstream.aclose()

    return _Transport()
