"""The interface both provider clients implement.

The pipeline depends on exactly one capability: structured output constrained
to a schema. Everything else - streaming, function calling, system prompt
nuances - is deliberately not part of this interface, because the moment it is,
swapping providers stops being possible.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class LLMResponse:
    """What a structured call returns."""

    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    # Included inside output_tokens by both providers; exposed separately so
    # thinking-level experiments can compare the hidden-token component.
    reasoning_tokens: int = 0
    model: str = ""
    # True when the model stopped because it hit the output cap. The caller
    # must check this rather than trying to parse: truncated JSON sometimes
    # parses into something plausible that is missing half its content.
    truncated: bool = False
    raw: Any = field(default=None, repr=False)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class StructuredClient(Protocol):
    """Both providers implement this and nothing more."""

    async def call_structured(
        self,
        *,
        model: str,
        messages: list[dict],
        response_schema: Any,
        files: list[bytes] | None = None,
    ) -> LLMResponse:
        ...
