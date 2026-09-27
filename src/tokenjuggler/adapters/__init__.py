"""One adapter per wire API. The router picks by `Deployment.api`."""

from __future__ import annotations

from tokenjuggler.adapters.anthropic_messages import AnthropicMessagesAdapter
from tokenjuggler.adapters.base import (
    Adapter,
    AdapterError,
    AdapterResult,
    BadRequest,
    DeploymentError,
    RateLimited,
    Transient,
    UnsupportedInput,
)
from tokenjuggler.adapters.genai import GenAIAdapter
from tokenjuggler.adapters.openai_responses import OpenAIResponsesAdapter
from tokenjuggler.settings import WireApi

__all__ = [
    "Adapter", "AdapterError", "AdapterResult", "BadRequest", "DeploymentError",
    "RateLimited", "Transient", "UnsupportedInput", "default_adapters",
]


def default_adapters(*, timeout_seconds: float) -> dict[WireApi, Adapter]:
    genai = GenAIAdapter(timeout_seconds=timeout_seconds)
    return {
        WireApi.OPENAI_RESPONSES: OpenAIResponsesAdapter(timeout_seconds=timeout_seconds),
        WireApi.GENAI_INTERACTIONS: genai,
        WireApi.GENAI_GENERATE_CONTENT: genai,
        WireApi.ANTHROPIC_MESSAGES: AnthropicMessagesAdapter(timeout_seconds=timeout_seconds),
    }
