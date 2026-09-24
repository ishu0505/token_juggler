"""Gemini through the google-genai SDK - two wire APIs, three routes.

* Interactions API (Google's recommended surface since June 2026): AI Studio.
* generateContent: Vertex AI (Interactions isn't on Vertex yet) and the
  Databricks gateway, which proxies the generateContent protocol.

Gemini bills thinking as output but reports it separately from the answer,
so both are added together - otherwise usage never reconciles with the bill.
"""

from __future__ import annotations

from typing import Any

from token_daddy.adapters.base import AdapterResult, b64, classify
from token_daddy.estimate import output_cap
from token_daddy.registry import Deployment
from token_daddy.settings import Provider, WireApi
from token_daddy.types import Capability, Request, Text, Usage

_INTERACTION_CONTENT_TYPE = {
    Capability.IMAGE: "image",
    Capability.PDF: "document",
    Capability.AUDIO: "audio",
    Capability.VIDEO: "video",
}


def build_client(dep: Deployment, timeout_seconds: float, http_options_extra: dict | None = None):
    from google import genai
    from google.genai import types

    options: dict[str, Any] = {"timeout": int(timeout_seconds * 1000)}  # milliseconds
    options.update(http_options_extra or {})
    if dep.provider is Provider.VERTEX:
        return genai.Client(
            vertexai=True,
            project=dep.credentials.project,
            location=dep.credentials.location,
            http_options=types.HttpOptions(**options),
        )
    if dep.provider is Provider.DATABRICKS:
        # The gateway authenticates with a bearer header; the SDK still insists
        # on an api_key, so it gets a placeholder.
        options["base_url"] = dep.base_url
        options["headers"] = {"Authorization": f"Bearer {dep.credentials.bearer_token}"}
        return genai.Client(api_key="databricks", http_options=types.HttpOptions(**options))
    if dep.base_url:
        options["base_url"] = dep.base_url
    return genai.Client(api_key=dep.credentials.api_key, http_options=types.HttpOptions(**options))


class GenAIAdapter:
    def __init__(self, *, timeout_seconds: float):
        self._timeout = timeout_seconds
        self._clients: dict[str, Any] = {}

    def client(self, dep: Deployment) -> Any:
        if dep.id not in self._clients:
            self._clients[dep.id] = build_client(dep, self._timeout)
        return self._clients[dep.id]

    async def call(self, dep: Deployment, request: Request) -> AdapterResult:
        client = self.client(dep)
        try:
            if dep.api is WireApi.GENAI_INTERACTIONS:
                response = await client.aio.interactions.create(
                    **build_interaction(dep, request)
                )
                return parse_interaction(response)
            response = await client.aio.models.generate_content(
                **build_generate_content(dep, request)
            )
            return parse_generate_content(response)
        except Exception as error:
            raise classify(error, dep) from error

    async def close(self) -> None:
        for client in self._clients.values():
            await client.aio.aclose()
        self._clients.clear()


# -- Interactions API ---------------------------------------------------------


def build_interaction(dep: Deployment, request: Request) -> dict:
    items = []
    for part in request.parts:
        if isinstance(part, Text):
            items.append({"type": "text", "text": part.text})
        else:
            kind = _INTERACTION_CONTENT_TYPE.get(part.capability, "document")
            items.append({"type": kind, "data": b64(part.data), "mime_type": part.mime_type})
    generation: dict[str, Any] = {"max_output_tokens": output_cap(request, dep)}
    if request.thinking:
        generation["thinking_level"] = request.thinking
    kwargs: dict[str, Any] = {
        "model": dep.model_id,
        "input": items,
        "generation_config": generation,
        # Nothing depends on server-side history; don't keep a copy.
        "store": False,
    }
    if request.system:
        kwargs["system_instruction"] = request.system
    if (schema := request.json_schema()) is not None:
        kwargs["response_mime_type"] = "application/json"
        kwargs["response_format"] = {
            "type": "text", "mime_type": "application/json", "schema": schema,
        }
    kwargs.update(dep.extra_params)
    return kwargs


def parse_interaction(response: Any) -> AdapterResult:
    usage = getattr(response, "usage", None)
    thought = getattr(usage, "total_thought_tokens", 0) or 0
    return AdapterResult(
        text=getattr(response, "output_text", "") or "",
        usage=Usage(
            input_tokens=getattr(usage, "total_input_tokens", 0) or 0,
            output_tokens=(getattr(usage, "total_output_tokens", 0) or 0) + thought,
            reasoning_tokens=thought,
            cached_input_tokens=getattr(usage, "total_cached_tokens", 0) or 0,
        ),
        truncated=getattr(response, "status", None) in ("incomplete", "budget_exceeded"),
        raw=response,
    )


# -- generateContent ------------------------------------------------------------


def build_generate_content(dep: Deployment, request: Request) -> dict:
    from google.genai import types

    parts = []
    for part in request.parts:
        if isinstance(part, Text):
            parts.append(types.Part.from_text(text=part.text))
        else:
            parts.append(types.Part.from_bytes(data=part.data, mime_type=part.mime_type))
    config: dict[str, Any] = {
        "max_output_tokens": output_cap(request, dep),
        # No tools are ever passed; this also silences the SDK's AFC warning.
        "automatic_function_calling": types.AutomaticFunctionCallingConfig(disable=True),
    }
    if request.system:
        config["system_instruction"] = request.system
    if request.temperature is not None:
        config["temperature"] = request.temperature
    if request.thinking:
        config["thinking_config"] = types.ThinkingConfig(
            thinking_level=getattr(types.ThinkingLevel, request.thinking.upper())
        )
    if (schema := request.json_schema()) is not None:
        config["response_mime_type"] = "application/json"
        config["response_json_schema"] = schema
    config.update(dep.extra_params)
    return {
        "model": dep.model_id,
        "contents": [types.Content(role="user", parts=parts)],
        "config": types.GenerateContentConfig(**config),
    }


def parse_generate_content(response: Any) -> AdapterResult:
    usage = getattr(response, "usage_metadata", None)
    thought = getattr(usage, "thoughts_token_count", 0) or 0
    candidates = getattr(response, "candidates", None) or []
    finish = str(getattr(candidates[0], "finish_reason", "")) if candidates else ""
    return AdapterResult(
        text=getattr(response, "text", "") or "",
        usage=Usage(
            input_tokens=getattr(usage, "prompt_token_count", 0) or 0,
            output_tokens=(getattr(usage, "candidates_token_count", 0) or 0) + thought,
            reasoning_tokens=thought,
            cached_input_tokens=getattr(usage, "cached_content_token_count", 0) or 0,
        ),
        truncated="MAX_TOKENS" in finish.upper(),
        raw=response,
    )
