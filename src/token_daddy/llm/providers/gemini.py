"""Gemini structured-output client.

Same interface as the OpenAI client, so the pipeline switches providers with a
string. That matters specifically for the escalation ladder in plan 6: retrying
a failed field on the OTHER provider is real evidence, because two providers
failing differently says something one provider failing twice does not.

Always reached through `llm_gate`.

Reaches the provider either directly or through a Databricks AI Gateway - see
`settings.gemini_base_url`.
"""

from __future__ import annotations

from typing import Any

from token_daddy.llm.base import LLMResponse
from token_daddy.llm.retry import with_retry
from token_daddy.config import settings
from token_daddy.utils.logger import get_logger

log = get_logger("worker.llm.gemini")

# Served by the Databricks gateway; a direct-vendor run would use "gemini-3.7-flash".
DEFAULT_MODEL = "system.ai.gemini-3-7-flash"
MAX_OUTPUT_TOKENS = 16_000


class GeminiClient:
    def __init__(
        self,
        api_key: str | None = None,
        *,
        thinking_level: str | None = None,
    ) -> None:
        self._base_url = settings.gemini_base_url
        self._api_key = api_key or settings.google_api_key
        self._thinking_level = thinking_level
        self._client: Any = None

    def _ensure_client(self) -> Any:
        if self._client is None:
            from google import genai

            if self._base_url:
                # The gateway authenticates with a bearer header, so the SDK's
                # own api_key is a placeholder it still insists on having.
                from google.genai import types

                log.info("Gemini SDK pointed at the gateway: %s", self._base_url)
                self._client = genai.Client(
                    api_key="databricks",
                    http_options=types.HttpOptions(
                        base_url=self._base_url,
                        headers={
                            "Authorization": f"Bearer {settings.databricks_token}"
                        },
                        # Milliseconds here, unlike the OpenAI SDK's seconds.
                        # Without it a hung call holds a gate slot forever and
                        # nothing ever gives up - the same defect that was
                        # fixed on the OpenAI side after a call sat for twelve
                        # minutes. `llm/retry.py` turns the timeout into a
                        # retryable failure.
                        timeout=int(settings.llm_request_timeout_seconds * 1000),
                    ),
                )
                return self._client

            if not self._api_key:
                raise RuntimeError(
                    "No Gemini credential: set GOOGLE_API_KEY, or "
                    "DATABRICKS_HOST and DATABRICKS_TOKEN for the gateway."
                )
            self._client = genai.Client(api_key=self._api_key)
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
        from google.genai import types

        system_text, user_text = _split_messages(messages)
        parts: list[Any] = []
        for data in files or []:
            # Native PDF, not a render - the text layer is more information
            # than a picture of it.
            parts.append(types.Part.from_bytes(data=data, mime_type="application/pdf"))
        parts.append(types.Part.from_text(text=user_text))

        async def _call() -> LLMResponse:
            response = await client.aio.models.generate_content(
                model=model,
                contents=[types.Content(role="user", parts=parts)],
                config=types.GenerateContentConfig(
                    system_instruction=system_text or None,
                    response_mime_type="application/json",
                    response_json_schema=response_schema,
                    max_output_tokens=MAX_OUTPUT_TOKENS,
                    seed=settings.llm_seed,
                    **_sampling_config(self._thinking_level),
                ),
            )
            return _to_response(response, model)

        log.info("Gemini call model=%s files=%d", model, len(files or []))
        return await with_retry(_call, what=f"gemini/{model}")


def _sampling_config(thinking_level: str | None = None) -> dict:
    """The settings 3.x takes, omitting any the model would reject.

    Sent as a dict rather than as fixed keyword arguments because an
    unsupported parameter is a 400, not a warning: passing `temperature` to a
    model that does not take it fails the call outright. Omitting a setting is
    always safe; sending a wrong one never is.
    """
    # Imported here, not at module level, matching the rest of this file: the
    # SDK is optional and importing it eagerly would break a process that only
    # ever uses OpenAI.
    from google.genai import types

    config: dict = {}

    if settings.gemini_temperature is not None:
        config["temperature"] = settings.gemini_temperature

    effective_thinking = thinking_level or settings.gemini_thinking_level
    if effective_thinking:
        # Thinking tokens are billed as OUTPUT, which is the quantity the
        # workspace quota limits - so this is a throughput setting as much as
        # a quality one.
        config["thinking_config"] = types.ThinkingConfig(
            thinking_level=effective_thinking
        )

    return config


def _split_messages(messages: list[dict]) -> tuple[str, str]:
    """Gemini takes the system prompt separately, not as a message."""
    system = " ".join(m["content"] for m in messages if m["role"] == "system")
    user = "\n\n".join(m["content"] for m in messages if m["role"] == "user")
    return system, user


def _to_response(response: Any, model: str) -> LLMResponse:
    usage = getattr(response, "usage_metadata", None)

    # Gemini 3.x models think, and the SDK reports thinking tokens separately
    # from the candidate count. They have to be added in, or the usage numbers
    # will not reconcile with the bill.
    reasoning_tokens = getattr(usage, "thoughts_token_count", 0) or 0
    output_tokens = getattr(usage, "candidates_token_count", 0) or 0
    output_tokens += reasoning_tokens

    return LLMResponse(
        text=getattr(response, "text", "") or "",
        input_tokens=getattr(usage, "prompt_token_count", 0) or 0,
        output_tokens=output_tokens,
        reasoning_tokens=reasoning_tokens,
        model=model,
        truncated=_hit_output_cap(response),
        raw=response,
    )


def _hit_output_cap(response: Any) -> bool:
    """Whether generation stopped because it ran out of output budget."""
    candidates = getattr(response, "candidates", None) or []
    if not candidates:
        return False
    reason = str(getattr(candidates[0], "finish_reason", "")).upper()
    return "MAX_TOKENS" in reason
