"""OpenAI Responses API - OpenAI Platform, Databricks gateway, and Bedrock.

All three speak the same protocol, so one adapter covers them; only the base
URL, the credential and the model id differ, and those come from the
deployment. Bedrock additionally gets `store: false` through the
deployment's `extra_params`, so it keeps no copy of the request.
"""

from __future__ import annotations

from typing import Any

from tokenjuggler.adapters.base import (
    AdapterResult,
    UnsupportedInput,
    b64,
    classify,
    strict_json_schema,
)
from tokenjuggler.estimate import output_cap
from tokenjuggler.registry import Deployment
from tokenjuggler.types import Capability, File, Request, Text, Usage


def build_client(dep: Deployment, timeout_seconds: float, http_client: Any = None):
    from openai import AsyncOpenAI

    return AsyncOpenAI(
        # Databricks and Bedrock both accept their token as the bearer key.
        api_key=dep.credentials.api_key or dep.credentials.bearer_token,
        base_url=dep.base_url,
        timeout=timeout_seconds,
        max_retries=0,  # the router owns what happens after a failure
        http_client=http_client,
    )


class OpenAIResponsesAdapter:
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
            response = await self.client(dep).responses.create(**kwargs)
        except Exception as error:
            raise classify(error, dep) from error
        return parse_response(response)

    async def close(self) -> None:
        for client in self._clients.values():
            await client.close()
        self._clients.clear()


def build_request(dep: Deployment, request: Request) -> dict:
    content = [_content_part(part) for part in request.parts]
    kwargs: dict[str, Any] = {
        "model": dep.model_id,
        "input": [{"role": "user", "content": content}],
        "max_output_tokens": output_cap(request, dep),
    }
    if request.system:
        kwargs["instructions"] = request.system
    if request.thinking:
        kwargs["reasoning"] = {"effort": request.thinking}
    if request.temperature is not None:
        kwargs["temperature"] = request.temperature
    if (schema := request.json_schema()) is not None:
        kwargs["text"] = {
            "format": {
                "type": "json_schema",
                "name": request.schema_name(),
                "schema": strict_json_schema(schema),
                "strict": True,
            }
        }
    kwargs.update(dep.extra_params)
    return kwargs


def _content_part(part: Text | File) -> dict:
    if isinstance(part, Text):
        return {"type": "input_text", "text": part.text}
    data_url = f"data:{part.mime_type};base64,{b64(part.data)}"
    if part.capability is Capability.IMAGE:
        return {"type": "input_image", "image_url": data_url}
    if part.capability in (Capability.AUDIO, Capability.VIDEO):
        raise UnsupportedInput(f"the Responses API does not take {part.mime_type} input")
    # input_file covers pdf, docx, xlsx, csv, txt, json and more.
    return {"type": "input_file", "filename": part.filename, "file_data": data_url}


def parse_response(response: Any) -> AdapterResult:
    usage = getattr(response, "usage", None)
    in_details = getattr(usage, "input_tokens_details", None)
    out_details = getattr(usage, "output_tokens_details", None)
    incomplete = getattr(response, "incomplete_details", None)
    return AdapterResult(
        text=getattr(response, "output_text", "") or "",
        usage=Usage(
            input_tokens=getattr(usage, "input_tokens", 0) or 0,
            output_tokens=getattr(usage, "output_tokens", 0) or 0,
            reasoning_tokens=getattr(out_details, "reasoning_tokens", 0) or 0,
            cached_input_tokens=getattr(in_details, "cached_tokens", 0) or 0,
        ),
        truncated=getattr(response, "status", None) == "incomplete"
        and getattr(incomplete, "reason", None) == "max_output_tokens",
        raw=response,
    )
