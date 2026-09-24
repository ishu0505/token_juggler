"""Claude client (Anthropic) - uses ANTHROPIC_API_KEY.

Not part of the Gemini/GPT set the project standardised on, but kept here
conforming to the same BaseLLMClient interface so it stays a drop-in option.
"""

import base64
import json

import anthropic
from pydantic import BaseModel

from token_daddy.config import settings
from token_daddy.llm_clients.base_client import (
    Attachment,
    BaseLLMClient,
    LLMResponse,
    ThinkingLevel,
    TokenUsage,
    UnsupportedAttachmentError,
)
from token_daddy.utils.logger import get_logger

log = get_logger(__name__)

CLAUDE_OPUS_5 = "claude-opus-5"
CLAUDE_SONNET_5 = "claude-sonnet-5"
CLAUDE_HAIKU_4_5 = "claude-haiku-4-5"

DEFAULT_MAX_TOKENS = 16000

# Extended thinking is a token budget rather than a named level, so map our
# four levels onto budgets. Always clamped to leave room for the answer.
_THINKING_BUDGETS: dict[str, int] = {
    "minimal": 1024,
    "low": 2048,
    "medium": 8192,
    "high": 16384,
}


class AnthropicClient(BaseLLMClient):
    provider = "anthropic"
    default_model = CLAUDE_OPUS_5
    supported_models = (CLAUDE_OPUS_5, CLAUDE_SONNET_5, CLAUDE_HAIKU_4_5)

    def __init__(self, model: str | None = None, api_key: str | None = None):
        super().__init__(model)
        self._client = anthropic.Anthropic(api_key=api_key or settings.anthropic_api_key)

    def generate(
        self,
        prompt: str,
        *,
        model: str | None = None,
        system: str | None = None,
        attachments: list[Attachment] | None = None,
        max_output_tokens: int | None = None,
        temperature: float | None = None,
        thinking_level: ThinkingLevel | None = None,
        response_schema: type[BaseModel] | None = None,
    ) -> LLMResponse:
        model = self._validate_model(model or self.model)
        max_tokens = max_output_tokens or DEFAULT_MAX_TOKENS

        content: list[dict] = [self._to_content_block(a) for a in (attachments or [])]
        content.append({"type": "text", "text": prompt})

        kwargs: dict = {}

        # There is no `response_format` on the Messages API. The reliable way
        # to pin the shape is to hand Claude exactly one tool whose input
        # schema is the model, then force it to call that tool - the tool
        # input comes back already conforming.
        if response_schema is not None:
            tool_name = response_schema.__name__
            kwargs["tools"] = [
                {
                    "name": tool_name,
                    "description": (
                        response_schema.__doc__ or f"Return a {tool_name}."
                    ).strip(),
                    "input_schema": response_schema.model_json_schema(),
                }
            ]
            kwargs["tool_choice"] = {"type": "tool", "name": tool_name}

        if thinking_level is not None:
            if response_schema is not None:
                # Extended thinking requires tool_choice=auto, which would
                # let Claude skip the tool and break the schema guarantee.
                log.warning(
                    "Anthropic: ignoring thinking_level=%s because a response_schema "
                    "is set - forced tool use and extended thinking are mutually "
                    "exclusive.",
                    thinking_level,
                )
            else:
                budget = min(_THINKING_BUDGETS[thinking_level], max(max_tokens - 1024, 1024))
                kwargs["thinking"] = {"type": "enabled", "budget_tokens": budget}
                if temperature is not None:
                    log.warning(
                        "Anthropic: dropping temperature=%s - extended thinking "
                        "requires the default temperature.",
                        temperature,
                    )
                    temperature = None

        if temperature is not None:
            kwargs["temperature"] = temperature

        response = self._client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system or "",
            messages=[{"role": "user", "content": content}],
            **kwargs,
        )

        text = self._extract_text(response, structured=response_schema is not None)

        usage = response.usage
        return self._finish(
            text=text,
            model=model,
            usage=TokenUsage.build(
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
            ),
            raw=response,
        )

    @staticmethod
    def _extract_text(response, *, structured: bool) -> str:
        """Pull the answer out, from the forced tool call when there is one."""
        if structured:
            for block in response.content:
                if block.type == "tool_use":
                    return json.dumps(block.input)
            raise ValueError(
                "Claude returned no tool_use block despite a forced tool_choice; "
                "cannot produce the requested schema."
            )
        return "".join(block.text for block in response.content if block.type == "text")

    @staticmethod
    def _to_content_block(attachment: Attachment) -> dict:
        """Claude takes images and PDFs; office formats need converting first."""
        encoded = base64.b64encode(attachment.data).decode("utf-8")

        if attachment.mime_type.startswith("image/"):
            block_type = "image"
        elif attachment.mime_type == "application/pdf":
            block_type = "document"
        else:
            raise UnsupportedAttachmentError(
                f"Claude can't read '{attachment.filename}' ({attachment.mime_type}). "
                "Convert it to PDF first, or send it to the OpenAI client instead."
            )

        return {
            "type": block_type,
            "source": {
                "type": "base64",
                "media_type": attachment.mime_type,
                "data": encoded,
            },
        }
