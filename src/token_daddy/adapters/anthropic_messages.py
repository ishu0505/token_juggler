"""Anthropic Messages API - Claude Platform and the Databricks gateway.

Structured output uses `output_config.format`, not a forced tool: Claude
Opus 5.5 rejects forced tool_choice with a 400. Thinking depth is `effort`
on current models; Haiku 4.5 predates effort and still takes a token budget.
"""

from __future__ import annotations

from typing import Any

from token_daddy.adapters.base import AdapterResult, UnsupportedInput, b64, classify
from token_daddy.estimate import output_cap
from token_daddy.registry import Deployment
from token_daddy.settings import Provider
from token_daddy.types import Capability, File, Request, Text, Usage

# Our levels onto Claude's effort scale. Claude has no "minimal".
_EFFORT = {"minimal": "low", "low": "low", "medium": "medium", "high": "high"}
# Haiku 4.5 thinking budgets (tokens); must stay below max_tokens.
_BUDGET = {"minimal": 1024, "low": 2048, "medium": 8192, "high": 16384}
# Model ids that take a thinking budget instead of effort.
_BUDGET_MODELS = ("haiku-4-5",)


def build_client(dep: Deployment, timeout_seconds: float, http_client: Any = None):
    from anthropic import AsyncAnthropic

    kwargs: dict[str, Any] = {
        "base_url": dep.base_url,
        "timeout": timeout_seconds,
        "max_retries": 0,  # the router owns what happens after a failure
    }
    if http_client is not None:
        kwargs["http_client"] = http_client
    if dep.provider is Provider.DATABRICKS:
        # Databricks authenticates with a bearer header; the api_key is unused.
        kwargs["api_key"] = "unused"
        kwargs["default_headers"] = {
            "Authorization": f"Bearer {dep.credentials.bearer_token}"
        }
    else:
        kwargs["api_key"] = dep.credentials.api_key
    return AsyncAnthropic(**kwargs)


class AnthropicMessagesAdapter:
    def __init__(self, *, timeout_seconds: float):
        self._timeout = timeout_seconds
        self._clients: dict[str, Any] = {}

    def client(self, dep: Deployment) -> Any:
        if dep.id not in self._clients:
            self._clients[dep.id] = build_client(dep, self._timeout)
        return self._clients[dep.id]

    async def call(self, dep: Deployment, request: Request) -> AdapterResult:
        kwargs = build_request(dep, request)
        try:
            response = await self.client(dep).messages.create(**kwargs)
        except Exception as error:
            raise classify(error, dep) from error
        return parse_response(response)

    async def close(self) -> None:
        for client in self._clients.values():
            await client.close()
        self._clients.clear()


def build_request(dep: Deployment, request: Request) -> dict:
    max_tokens = output_cap(request, dep)
    kwargs: dict[str, Any] = {
        "model": dep.model_id,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": [_block(p) for p in request.parts]}],
    }
    if request.system:
        kwargs["system"] = request.system
    output_config: dict[str, Any] = {}
    if (schema := request.json_schema()) is not None:
        output_config["format"] = {"type": "json_schema", "schema": schema}
    if request.thinking:
        if any(marker in dep.model_id for marker in _BUDGET_MODELS):
            budget = min(_BUDGET[request.thinking], max(max_tokens - 1024, 1024))
            kwargs["thinking"] = {"type": "enabled", "budget_tokens": budget}
        else:
            output_config["effort"] = _EFFORT[request.thinking]
    if output_config:
        kwargs["output_config"] = output_config
    if request.temperature is not None:
        # Current Opus/Sonnet reject sampling params; sent only when asked for.
        kwargs["temperature"] = request.temperature
    kwargs.update(dep.extra_params)
    return kwargs


def _block(part: Text | File) -> dict:
    if isinstance(part, Text):
        return {"type": "text", "text": part.text}
    source = {"type": "base64", "media_type": part.mime_type, "data": b64(part.data)}
    if part.capability is Capability.IMAGE:
        return {"type": "image", "source": source}
    if part.capability is Capability.PDF:
        return {"type": "document", "source": source}
    if part.mime_type.startswith("text/"):
        return {
            "type": "document",
            "source": {"type": "text", "media_type": "text/plain",
                       "data": part.data.decode("utf-8", errors="replace")},
        }
    raise UnsupportedInput(f"Claude does not take {part.mime_type} input")


def parse_response(response: Any) -> AdapterResult:
    usage = getattr(response, "usage", None)
    cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0
    cache_write = getattr(usage, "cache_creation_input_tokens", 0) or 0
    text = "".join(
        block.text for block in getattr(response, "content", []) or []
        if getattr(block, "type", None) == "text"
    )
    return AdapterResult(
        text=text,
        usage=Usage(
            # Anthropic reports uncached input separately from cache reads and
            # writes; the total is what the request actually carried.
            input_tokens=(getattr(usage, "input_tokens", 0) or 0) + cache_read + cache_write,
            output_tokens=getattr(usage, "output_tokens", 0) or 0,
            cached_input_tokens=cache_read,
        ),
        truncated=getattr(response, "stop_reason", None) == "max_tokens",
        raw=response,
    )
