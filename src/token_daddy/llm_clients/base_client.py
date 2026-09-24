"""The abstraction layer every LLM client implements.

The point of this file: calling code should not care whether it's talking to
Gemini or GPT. You build the same `Attachment` objects, call the same
`generate(...)`, and get back the same `LLMResponse` - including token usage
and an estimated cost - no matter which provider is behind it.

    from token_daddy.llm_clients import get_client, Attachment

    client = get_client("gemini")          # or "openai" / "anthropic"
    reply = client.generate(
        "Summarise this invoice.",
        attachments=[Attachment.from_path("invoice.pdf")],
    )
    print(reply.text, reply.usage.total_tokens)

Ask for a pydantic model back and every provider enforces it natively, so
`reply.text` is always valid JSON for that schema:

    reply = client.generate("Extract the TOC.", response_schema=TOCExtractionResult)
    toc = TOCExtractionResult.model_validate_json(reply.text)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal
import copy
import mimetypes

from pydantic import BaseModel

from token_daddy.pricing import estimate_cost


class UnsupportedAttachmentError(ValueError):
    """Raised when a file type isn't accepted by the chosen provider/model."""


class Provider(StrEnum):
    """The providers `get_client` knows how to build."""

    GEMINI = "gemini"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"


# How hard the model should think before answering. Providers spell this
# differently (OpenAI: reasoning effort, Gemini: thinking level/budget), so
# clients translate it; passing None leaves the provider default alone.
ThinkingLevel = Literal["minimal", "low", "medium", "high"]


@dataclass(frozen=True)
class TokenUsage:
    """Token counts for a single call. Every provider is normalised into this."""

    input_tokens: int
    output_tokens: int
    total_tokens: int
    reasoning_tokens: int = 0

    @classmethod
    def build(
        cls,
        input_tokens: int | None,
        output_tokens: int | None,
        total_tokens: int | None = None,
        reasoning_tokens: int | None = None,
    ) -> "TokenUsage":
        """Normalise provider counts, filling in total if it wasn't reported.

        `output_tokens` must already include `reasoning_tokens` - every
        provider bills thinking as output, so counting it separately would
        under-report what you pay for.
        """
        input_tokens = input_tokens or 0
        output_tokens = output_tokens or 0
        if total_tokens is None:
            total_tokens = input_tokens + output_tokens
        return cls(input_tokens, output_tokens, total_tokens, reasoning_tokens or 0)


@dataclass
class LLMResponse:
    """What every `generate()` call returns, whichever provider ran it."""

    text: str
    model: str
    provider: str
    usage: TokenUsage
    estimated_cost_usd: float | None = None
    raw: Any = field(repr=False, default=None)  # the untouched provider response


@dataclass
class Attachment:
    """A file to send along with the prompt (pdf, image, spreadsheet, ...)."""

    filename: str
    data: bytes
    mime_type: str

    @classmethod
    def from_path(cls, path: str | Path, mime_type: str | None = None) -> "Attachment":
        path = Path(path)
        if mime_type is None:
            guessed, _ = mimetypes.guess_type(path.name)
            mime_type = guessed or "application/octet-stream"
        return cls(filename=path.name, data=path.read_bytes(), mime_type=mime_type)

    @classmethod
    def from_bytes(
        cls, data: bytes, filename: str, mime_type: str | None = None
    ) -> "Attachment":
        """Wrap bytes already in memory - no temp file needed.

        `index_it` slices PDFs into chunks in memory, so this is the path it
        uses; writing each chunk to disk just to read it straight back would
        be pure overhead.
        """
        if mime_type is None:
            guessed, _ = mimetypes.guess_type(filename)
            mime_type = guessed or "application/octet-stream"
        return cls(filename=filename, data=data, mime_type=mime_type)

    @property
    def size_bytes(self) -> int:
        return len(self.data)


def strict_json_schema(model: type[BaseModel]) -> dict:
    """Turn a pydantic model into a schema OpenAI strict mode accepts.

    Strict mode is fussy in three specific ways, and pydantic's own
    `model_json_schema()` violates all three:

    1. Every object needs `additionalProperties: false`.
    2. Every property must be listed in `required` - even ones with defaults.
       That is fine for us: it just means the model always emits the field
       rather than leaving it out for us to fill in.
    3. `default` is rejected outright, so it gets stripped.

    `$defs`/`$ref` are left alone - strict mode handles them natively, and
    inlining them would blow up recursive schemas.
    """
    schema = copy.deepcopy(model.model_json_schema())

    def walk(node: Any) -> None:
        if isinstance(node, list):
            for item in node:
                walk(item)
            return
        if not isinstance(node, dict):
            return

        node.pop("default", None)

        properties = node.get("properties")
        if isinstance(properties, dict):
            node["additionalProperties"] = False
            node["required"] = list(properties)

        for value in node.values():
            walk(value)

    walk(schema)
    return schema


class BaseLLMClient(ABC):
    """Common interface. Subclasses fill in the three class attrs + `generate`."""

    provider: str = "unknown"
    default_model: str = ""
    supported_models: tuple[str, ...] = ()

    def __init__(self, model: str | None = None):
        self.model = self._validate_model(model or self.default_model)

    @classmethod
    def _validate_model(cls, model: str) -> str:
        """Fail loudly on a typo'd model name instead of at API-call time."""
        if model not in cls.supported_models:
            allowed = ", ".join(cls.supported_models)
            raise ValueError(
                f"{cls.provider} does not support model '{model}'. Pick one of: {allowed}"
            )
        return model

    @abstractmethod
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
        """Send a prompt (plus optional files) and return text + token usage.

        `response_schema` switches the provider into structured-output mode:
        `text` comes back as JSON conforming to that model. `temperature` and
        `thinking_level` are passed through only when set - note that
        reasoning models often reject an explicit temperature, so leaving it
        at None is the safe default for those.
        """
        raise NotImplementedError

    def _finish(
        self,
        *,
        text: str,
        model: str,
        usage: TokenUsage,
        raw: Any,
    ) -> LLMResponse:
        """Bolt the cost estimate on so no client has to remember to."""
        return LLMResponse(
            text=text,
            model=model,
            provider=self.provider,
            usage=usage,
            estimated_cost_usd=estimate_cost(
                model, usage.input_tokens, usage.output_tokens
            ),
            raw=raw,
        )
