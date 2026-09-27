"""What every wire adapter shares: the result shape and error classification.

Adapters never retry and never sleep. They make exactly one call and, when it
fails, say what KIND of failure it was - the router decides what to do about
it (usually: try the next deployment).
"""

from __future__ import annotations

import asyncio
import base64
import copy
from dataclasses import dataclass, field
from typing import Any, Protocol

from tokenjuggler.registry import Deployment
from tokenjuggler.types import Request, Usage


@dataclass
class AdapterResult:
    text: str
    usage: Usage
    truncated: bool = False
    raw: Any = field(default=None, repr=False)


class Adapter(Protocol):
    async def call(self, dep: Deployment, request: Request) -> AdapterResult: ...

    async def close(self) -> None: ...


# -- failures ----------------------------------------------------------------


class AdapterError(Exception):
    """Base for classified failures. `sent` says whether the request reached
    the provider - if it did, the provider has counted it against its quota."""

    sent: bool = True

    def __init__(self, message: str, *, cause: BaseException | None = None):
        super().__init__(message)
        self.__cause__ = cause


class RateLimited(AdapterError):
    """A real 429. Our limiter and the provider disagree: bench the route."""

    def __init__(self, message: str, *, retry_after: float | None = None, cause=None):
        super().__init__(message, cause=cause)
        self.retry_after = retry_after


class Transient(AdapterError):
    """5xx, overloaded, timeout, dropped connection. Another route may work."""


class DeploymentError(AdapterError):
    """Specific to this route: bad credentials, no access, unknown model id.
    Fail over, and bench the route for a while - it will not fix itself."""


class BadRequest(AdapterError):
    """The request itself is wrong (400/422). Every route would reject it the
    same way, so it is raised rather than failed over."""


class UnsupportedInput(BadRequest):
    """A part this wire API cannot carry. Raised before sending."""

    sent = False


# Transport errors carry no status code and share no base class across SDKs,
# so they are recognised by name. APITimeoutError matters most: a timeout is
# the one failure that looks permanent if you only check for status codes.
_TRANSPORT_ERROR_NAMES = frozenset({
    "APITimeoutError", "APIConnectionError", "ConnectError", "ConnectTimeout",
    "ReadTimeout", "WriteTimeout", "PoolTimeout", "RemoteProtocolError",
    "ReadError", "ServiceUnavailable", "DeadlineExceeded",
})


def classify(error: BaseException, dep: Deployment) -> AdapterError:
    """Map any SDK exception onto one of the classes above."""
    if isinstance(error, AdapterError):
        return error
    where = f"{dep.id}: {type(error).__name__}: {error}"
    status = _status_of(error)
    if status is not None:
        if status == 429:
            return RateLimited(where, retry_after=_retry_after(error), cause=error)
        if status in (401, 403, 404):
            return DeploymentError(where, cause=error)
        if status in (400, 413, 422):
            return BadRequest(where, cause=error)
        if status in (408, 409) or status >= 500:
            return Transient(where, cause=error)
        return DeploymentError(where, cause=error)
    if type(error).__name__ in _TRANSPORT_ERROR_NAMES or isinstance(
        error, (ConnectionError, TimeoutError, asyncio.TimeoutError)
    ):
        return Transient(where, cause=error)
    raise error  # a bug in our own code - never hide it behind a failover


def _status_of(error: BaseException) -> int | None:
    for attr in ("status_code", "code", "status"):
        value = getattr(error, attr, None)
        if isinstance(value, int) and 100 <= value < 600:
            return value
    return None


def _retry_after(error: BaseException) -> float | None:
    response = getattr(error, "response", None)
    headers = getattr(response, "headers", None)
    if not headers:
        return None
    try:
        if ms := headers.get("retry-after-ms"):
            return float(ms) / 1000
        if seconds := headers.get("retry-after"):
            return float(seconds)
    except (TypeError, ValueError):
        return None
    return None


# -- shared payload helpers ---------------------------------------------------


def b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def strict_json_schema(schema: dict) -> dict:
    """A JSON schema OpenAI's strict mode accepts.

    Strict mode needs `additionalProperties: false` on every object, every
    property listed in `required`, and no `default` anywhere - pydantic's
    `model_json_schema()` violates all three. `$defs`/`$ref` are left alone;
    strict mode resolves them natively.
    """
    schema = copy.deepcopy(schema)

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
