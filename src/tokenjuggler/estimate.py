"""Pre-call token estimates, from the payload alone.

The estimate only sizes the reservation - the settle step replaces it with the
provider's real count - so it aims to be cheap and slightly high rather than
exact. Nothing here calls a tokenizer or the network, and file bytes never
leave the process: only the resulting numbers reach Redis.
"""

from __future__ import annotations

import json
import math
import re
import struct

from tokenjuggler.limiter import Cost
from tokenjuggler.registry import Deployment
from tokenjuggler.settings import EstimationConfig, Family, Reservation
from tokenjuggler.types import Capability, File, Request, Text

# Page objects in a PDF. Deliberately excludes "/Type /Pages" (the page tree).
_PDF_PAGE = re.compile(rb"/Type\s*/Page(?![a-zA-Z])")
# A PDF whose page objects are all inside compressed object streams hides
# them from the regex above; fall back to a size-based guess.
_PDF_BYTES_PER_PAGE_FALLBACK = 60_000


def pdf_pages(data: bytes) -> int:
    pages = len(_PDF_PAGE.findall(data))
    return pages or max(1, math.ceil(len(data) / _PDF_BYTES_PER_PAGE_FALLBACK))


def audio_seconds(data: bytes, fallback_bytes_per_second: int) -> float:
    """Duration from a WAV header when there is one, else from the size."""
    if len(data) >= 44 and data[:4] == b"RIFF" and data[8:12] == b"WAVE":
        byte_rate = struct.unpack_from("<I", data, 28)[0]
        if byte_rate:
            return (len(data) - 44) / byte_rate
    return len(data) / fallback_bytes_per_second


def estimate_input_tokens(request: Request, family: Family, cfg: EstimationConfig) -> int:
    chars = len(request.system or "")
    tokens = 0.0
    if (schema := request.json_schema()) is not None:
        chars += len(json.dumps(schema))
    for part in request.parts:
        if isinstance(part, Text):
            chars += len(part.text)
            continue
        tokens += _file_tokens(part, family, cfg)
    tokens += chars / cfg.chars_per_token
    return math.ceil(tokens * cfg.safety_multiplier)


def _file_tokens(part: File, family: Family, cfg: EstimationConfig) -> float:
    match part.capability:
        case Capability.PDF:
            return pdf_pages(part.data) * cfg.tokens_per_pdf_page[family]
        case Capability.IMAGE:
            return cfg.tokens_per_image[family]
        case Capability.AUDIO:
            seconds = audio_seconds(part.data, cfg.audio_bytes_per_second)
            return seconds * cfg.tokens_per_audio_second[family]
        case Capability.VIDEO:
            return len(part.data) / 1_000_000 * cfg.tokens_per_video_mb
    if part.mime_type.startswith("text/") or part.mime_type == "application/json":
        return len(part.data) / cfg.chars_per_token
    return len(part.data) / cfg.bytes_per_token_other


_BASE64 = re.compile(r"^[A-Za-z0-9+/=\s]+$")
_MIME_KEYS = ("mime_type", "mimeType", "media_type")
# Strings this long in a payload are file contents, not prompt text.
_MIN_BLOB_CHARS = 1024


def estimate_body_input_tokens(body: dict, family: Family, cfg: EstimationConfig) -> int:
    """Estimate input tokens from a raw provider request body (native SDKs).

    Inline files arrive base64-encoded - as data URLs or beside a mime-type
    key - and are decoded and priced as files; counting their base64 as text
    would inflate a 1 MB PDF into hundreds of thousands of "tokens".
    """
    chars = 0
    tokens = 0.0

    def walk(node, mime_hint: str | None = None) -> None:
        nonlocal chars, tokens
        if isinstance(node, dict):
            hint = next((node[k] for k in _MIME_KEYS if isinstance(node.get(k), str)), mime_hint)
            for key, value in node.items():
                if key in ("model", "type", "role", *_MIME_KEYS):
                    continue
                walk(value, hint)
        elif isinstance(node, list):
            for item in node:
                walk(item, mime_hint)
        elif isinstance(node, str):
            blob = _as_file(node, mime_hint)
            if blob is not None:
                tokens += _file_tokens(blob, family, cfg)
            else:
                chars += len(node)

    walk(body)
    tokens += chars / cfg.chars_per_token
    return math.ceil(tokens * cfg.safety_multiplier)


def _as_file(value: str, mime_hint: str | None) -> File | None:
    import base64
    import binascii

    if value.startswith("data:") and ";base64," in value[:200]:
        header, _, payload = value.partition(",")
        mime = header[5:].split(";", 1)[0]
    elif len(value) >= _MIN_BLOB_CHARS and mime_hint and _BASE64.match(value[:4096]):
        mime, payload = mime_hint, value
    else:
        return None
    try:
        data = base64.b64decode(payload, validate=False)
    except (binascii.Error, ValueError):
        return None
    return File(data, mime)


def output_cap(request: Request, dep: Deployment) -> int:
    """The max_output_tokens actually sent. Includes reasoning tokens on
    every provider, so it is a true upper bound on output."""
    return request.max_output_tokens or dep.max_output_tokens


def reservation_cost(
    request: Request, dep: Deployment, cfg: EstimationConfig, *, input_tokens: int | None = None
) -> Cost:
    """What to reserve for this request on this deployment."""
    if input_tokens is None:
        input_tokens = estimate_input_tokens(request, dep.family, cfg)
    cap = output_cap(request, dep)
    if dep.reservation is Reservation.STRICT:
        output = cap
    else:
        output = min(cap, max(cfg.output_floor, math.ceil(input_tokens * cfg.output_ratio)))
    return Cost(input_tokens=input_tokens, output_tokens=output)
