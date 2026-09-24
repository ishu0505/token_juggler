"""OpenAI structured-output client.

Wrapped rather than used directly so the pipeline depends on
`call_structured` instead of on OpenAI's request shape. When the API changes -
and it does - this file changes and nothing else.

Always reached through `llm_gate`. Constructing this and calling it directly
bypasses rate limiting, per-job budget and usage accounting.

Reaches the provider either directly or through a Databricks AI Gateway - see
`settings.openai_base_url`. The gateway speaks OpenAI's own protocol, so this
file does not care which it is talking to.
"""

from __future__ import annotations

import json
from typing import Any

from token_daddy.llm.base import LLMResponse
from token_daddy.llm.retry import with_retry
from token_daddy.config import settings
from token_daddy.utils.logger import get_logger

log = get_logger("worker.llm.openai")

# Served by the Databricks gateway; a direct-vendor run would use "gpt-5.4".
DEFAULT_MODEL = "system.ai.gpt-5-6-terra"

# Generous, because an indexer response for a 3-page chunk is a long list of
# elements. Overflow is handled by bisecting the chunk, not by raising this.
MAX_OUTPUT_TOKENS = 16_000


class OpenAIClient:
    def __init__(
        self,
        api_key: str | None = None,
        *,
        reasoning_effort: str | None = None,
    ) -> None:
        # A Databricks gateway token, when configured, wins over a direct
        # vendor key: the gateway speaks the same protocol, so only the
        # credential and base URL change, and spend stays on one bill.
        self._base_url = settings.openai_base_url
        self._api_key = (
            api_key
            or (settings.databricks_token if self._base_url else None)
            or settings.openai_api_key
        )
        self._reasoning_effort = reasoning_effort
        self._client: Any = None

    def _ensure_client(self) -> Any:
        """Build the SDK client lazily, so importing this costs nothing."""
        if self._client is None:
            if not self._api_key:
                raise RuntimeError(
                    "No OpenAI credential: set OPENAI_API_KEY, or "
                    "DATABRICKS_HOST and DATABRICKS_TOKEN for the gateway."
                )
            from openai import AsyncOpenAI

            if self._base_url:
                log.info("OpenAI SDK pointed at the gateway: %s", self._base_url)
            # A REQUEST TIMEOUT, because without one a hung connection hangs
            # the job forever. Measured: a fill call sat with no response and
            # no error for over twelve minutes, holding a gate slot the whole
            # time, and nothing anywhere would ever have given up. The retry
            # ladder cannot help with a call that never returns.
            self._client = AsyncOpenAI(
                api_key=self._api_key,
                base_url=self._base_url,
                timeout=settings.llm_request_timeout_seconds,
                max_retries=0,  # `llm/retry.py` owns retries; two ladders fight.
            )
        return self._client

    async def call_structured(
        self,
        *,
        model: str = DEFAULT_MODEL,
        messages: list[dict],
        response_schema: Any,
        files: list[bytes] | None = None,
    ) -> LLMResponse:
        """One structured call. Retries transient failures on top of the gate."""
        client = self._ensure_client()
        request_messages = _with_attachments(messages, files)

        async def _call() -> LLMResponse:
            response = await client.chat.completions.create(
                model=model,
                messages=request_messages,
                max_tokens=MAX_OUTPUT_TOKENS,
                **_sampling(self._reasoning_effort),
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "elements",
                        # Strict mode is what makes the response parse without
                        # repair, and what enforces the block-id enum.
                        "strict": True,
                        "schema": response_schema,
                    },
                },
            )
            return _to_response(response, model)

        log.info("OpenAI call model=%s files=%d", model, len(files or []))
        return await with_retry(_call, what=f"openai/{model}")


def _sampling(reasoning_effort: str | None = None) -> dict:
    """Sampling arguments, omitting anything this model refuses.

    The gpt-5 family rejects any temperature but its default, so temperature is
    sent only when explicitly configured. `seed` is accepted and is the only
    reproducibility lever available there.
    """
    args: dict[str, Any] = {}
    if settings.openai_temperature is not None:
        args["temperature"] = settings.openai_temperature
    effective_reasoning = reasoning_effort or settings.openai_reasoning_effort
    if effective_reasoning is not None:
        args["reasoning_effort"] = effective_reasoning
    if settings.llm_seed is not None:
        args["seed"] = settings.llm_seed
    return args


def _with_attachments(messages: list[dict], files: list[bytes] | None) -> list[dict]:
    """Attach PDF slices to the last user message.

    Native PDF rather than a rendered image: the model gets the text layer
    alongside the rendering, which is strictly more information for a digital
    page.
    """
    if not files:
        return messages

    import base64

    parts: list[dict] = []
    for index, data in enumerate(files):
        encoded = base64.b64encode(data).decode("ascii")
        parts.append(
            {
                "type": "file",
                "file": {
                    "filename": f"chunk_{index}.pdf",
                    "file_data": f"data:application/pdf;base64,{encoded}",
                },
            }
        )

    result = list(messages)
    for position in range(len(result) - 1, -1, -1):
        if result[position]["role"] == "user":
            text = result[position]["content"]
            result[position] = {
                "role": "user",
                "content": [*parts, {"type": "text", "text": text}],
            }
            break
    return result


def _to_response(response: Any, model: str) -> LLMResponse:
    choice = response.choices[0]
    usage = getattr(response, "usage", None)
    completion_details = getattr(usage, "completion_tokens_details", None)

    return LLMResponse(
        text=choice.message.content or "",
        input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
        output_tokens=getattr(usage, "completion_tokens", 0) or 0,
        reasoning_tokens=getattr(completion_details, "reasoning_tokens", 0) or 0,
        model=model,
        # "length" means the output cap was hit, so the JSON is incomplete.
        truncated=choice.finish_reason == "length",
        raw=response,
    )
