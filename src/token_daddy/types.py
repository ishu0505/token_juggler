"""The request/response shapes every caller and every adapter share.

Callers build a list of parts - `Text` and `File` - and get back one
`Response`, whichever provider actually served it. Adapters translate these
into each wire API and back; nothing outside `adapters/` sees a provider type.
"""

from __future__ import annotations

import mimetypes
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel


class Capability(StrEnum):
    """What a model (or one route to it) can accept or do."""

    TEXT = "text"
    IMAGE = "image"
    PDF = "pdf"
    AUDIO = "audio"
    VIDEO = "video"
    JSON_SCHEMA = "json_schema"
    REASONING = "reasoning"


ThinkingLevel = Literal["minimal", "low", "medium", "high"]


@dataclass(frozen=True)
class Text:
    text: str


@dataclass(frozen=True)
class File:
    """An attachment held in memory. Never written to Redis - only its size and
    a token estimate leave the process."""

    data: bytes = field(repr=False)
    mime_type: str
    filename: str = "file"

    @classmethod
    def from_path(cls, path: str | Path, mime_type: str | None = None) -> File:
        path = Path(path)
        return cls(path.read_bytes(), mime_type or _guess_mime(path.name), path.name)

    @classmethod
    def from_bytes(cls, data: bytes, filename: str, mime_type: str | None = None) -> File:
        return cls(data, mime_type or _guess_mime(filename), filename)

    @property
    def capability(self) -> Capability:
        """The capability a model needs to read this file."""
        if self.mime_type == "application/pdf":
            return Capability.PDF
        major = self.mime_type.split("/", 1)[0]
        if major == "image":
            return Capability.IMAGE
        if major == "audio":
            return Capability.AUDIO
        if major == "video":
            return Capability.VIDEO
        return Capability.TEXT


Part = Text | File


def _guess_mime(name: str) -> str:
    guessed, _ = mimetypes.guess_type(name)
    return guessed or "application/octet-stream"


@dataclass
class Request:
    """One logical call, before a deployment has been chosen."""

    model: str
    parts: list[Part]
    system: str | None = None
    # A pydantic model class, or a raw JSON schema dict.
    response_schema: type[BaseModel] | dict | None = None
    max_output_tokens: int | None = None
    thinking: ThinkingLevel | None = None
    # Sent only when set: reasoning models reject an explicit temperature.
    temperature: float | None = None
    job_id: str | None = None

    def required_capabilities(self) -> set[Capability]:
        caps = {Capability.TEXT}
        caps.update(p.capability for p in self.parts if isinstance(p, File))
        if self.response_schema is not None:
            caps.add(Capability.JSON_SCHEMA)
        return caps

    def json_schema(self) -> dict | None:
        if self.response_schema is None:
            return None
        if isinstance(self.response_schema, dict):
            return self.response_schema
        return self.response_schema.model_json_schema()

    def schema_name(self) -> str:
        if isinstance(self.response_schema, type):
            return self.response_schema.__name__
        return "response"


@dataclass(frozen=True)
class Usage:
    """Token counts for one call. `output_tokens` already includes reasoning
    tokens - every provider bills thinking as output."""

    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    cached_input_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass(frozen=True)
class Attempt:
    """One try at one deployment, kept so a response explains its route."""

    deployment: str
    outcome: Literal["ok", "rate_limited", "transient", "deployment_error", "skipped"]
    detail: str = ""


@dataclass
class Response:
    text: str
    model: str
    deployment: str
    provider: str
    usage: Usage
    cost_usd: float | None
    latency_ms: float
    truncated: bool = False
    attempts: list[Attempt] = field(default_factory=list)
    parsed: Any = None
    raw: Any = field(default=None, repr=False)
