"""Retry with exponential backoff and jitter.

Sits ON TOP of the gate, not instead of it. The gate stops us exceeding a
limit we know about; this handles the provider disagreeing anyway - a 429 from
a limit we mis-measured, or a transient 5xx.

Jitter matters more than it looks: without it, twenty workers that hit a limit
together retry together, and keep colliding in lockstep.
"""

from __future__ import annotations

import asyncio
import random
from collections.abc import Awaitable, Callable
from typing import TypeVar

from token_daddy.utils.logger import get_logger

log = get_logger("worker.llm.retry")

T = TypeVar("T")

MAX_ATTEMPTS = 4
BASE_DELAY_SECONDS = 2.0


# Provider SDKs raise their own transport errors that inherit from nothing we
# can name here without importing them - and this module stays provider
# agnostic on purpose. Matching on class name is the price of that.
#
# `APITimeoutError` is the one that matters: once a request timeout is
# configured, a hung call raises this instead of hanging forever, and it
# carries NO status code and is NOT a builtin TimeoutError. Without this list
# it read as permanent, so a timeout became a hard failure with no retry -
# which is worse than the hang it replaced.
TRANSPORT_ERROR_NAMES = frozenset({
    "APITimeoutError",
    "APIConnectionError",
    "InternalServerError",
    "ServiceUnavailable",
    "DeadlineExceeded",
})


def is_retryable(error: Exception) -> bool:
    """429 and 5xx are worth retrying. A 400 is our bug and never will be."""
    status = getattr(error, "status_code", None) or getattr(error, "status", None)
    if status is not None:
        return status == 429 or 500 <= status < 600

    if type(error).__name__ in TRANSPORT_ERROR_NAMES:
        return True

    # No status - a connection error. Worth one more go.
    return isinstance(error, (ConnectionError, TimeoutError, asyncio.TimeoutError))


async def with_retry(operation: Callable[[], Awaitable[T]], *, what: str = "call") -> T:
    """Run `operation`, retrying transient failures with backoff and jitter."""
    last_error: Exception | None = None

    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            return await operation()
        except Exception as error:
            last_error = error
            if not is_retryable(error) or attempt == MAX_ATTEMPTS:
                raise

            delay = BASE_DELAY_SECONDS * (2 ** (attempt - 1))
            delay += random.uniform(0, delay * 0.25)
            log.warning(
                "%s failed (attempt %d/%d): %s - retrying in %.1fs",
                what, attempt, MAX_ATTEMPTS, error, delay,
            )
            await asyncio.sleep(delay)

    raise last_error  # unreachable; keeps type checkers happy
