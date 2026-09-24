"""The single chokepoint for every model call.

**Nothing calls a provider SDK directly.** Not the indexer, not the filler,
not discovery. Everything goes through `llm_gate`:

    async with llm_gate("openai", "gpt-5.4", job_id, estimated_tokens=2000) as slot:
        response = await client.call_structured(...)
        slot.record_actual(response.total_tokens, response.output_tokens)

It is a context manager rather than a function on purpose: a function is easy
to forget to use, whereas the `async with` is the only way to get a slot at
all. If bypassing were possible, eventually something would bypass it.

What it enforces, in order
--------------------------
1. Token budget for this minute (TPM), reserved up front and reconciled after.
2. Request count for this minute (RPM).
3. Concurrency, via a Redis sorted set that expires stale holders.
4. Per-job token budget - a runaway document fails rather than starving the
   queue behind it.

All four live in Redis rather than in the process, because the limits are per
provider across ALL workers. Twenty concurrent documents can each look
compliant on their own and collectively blow the quota, and then all twenty
fail together.

The waiting is deliberately simple: if a limit is hit, sleep until the next
minute bucket and try again. No priority, no fairness. Add those when there is
evidence they are needed.
"""

from __future__ import annotations

import asyncio
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

from token_daddy.llm import limits
from token_daddy.config import settings
from token_daddy.utils.logger import get_logger

log = get_logger("worker.llm.gate")

# Text token estimate. Crude on purpose - step 5 reconciles it, so precision
# here buys nothing.
CHARS_PER_TOKEN = 4

# A PDF page sent to a vision model, regardless of content.
# TODO: measure per provider against real `usage` responses; both publish
# per-image costs that depend on dimensions, and they change.
ESTIMATED_TOKENS_PER_PAGE = 1_500

# How long to wait for a slot before giving up, so a wedged limiter surfaces
# as an error rather than a job that hangs forever.
MAX_WAIT_SECONDS = 300

# How often to re-check for a free concurrency slot.
POLL_INTERVAL_SECONDS = 0.05


class BudgetExceeded(RuntimeError):
    """A job consumed more tokens than it is allowed.

    Raised rather than throttled: one runaway document should fail on its own
    rather than degrade every job queued behind it.
    """


class GateTimeout(RuntimeError):
    """Waited too long for a slot. Almost always a misconfigured ceiling."""


@dataclass
class ProviderLimits:
    tokens_per_minute: int
    output_tokens_per_minute: int
    requests_per_minute: int
    max_concurrent: int


def limits_for(provider: str) -> ProviderLimits:
    """Ceilings for a provider, from config.

    Config rather than the database for now: `provider_limits` exists and is
    meant to be loaded into Redis at startup, but nothing writes it yet, and
    reading a table that is always empty would just be indirection.
    """
    if provider == "openai":
        return ProviderLimits(
            tokens_per_minute=settings.openai_tpm,
            output_tokens_per_minute=settings.openai_output_tpm,
            requests_per_minute=settings.openai_rpm,
            max_concurrent=settings.openai_max_concurrent,
        )
    return ProviderLimits(
        tokens_per_minute=settings.gemini_tpm,
        output_tokens_per_minute=settings.gemini_output_tpm,
        requests_per_minute=settings.gemini_rpm,
        max_concurrent=settings.gemini_max_concurrent,
    )


# What share of a call's tokens come back as output. Extraction calls send
# a page and receive a small JSON object, so this is low. Only used for the
# pre-call reservation; the real number replaces it on reconcile.
# Measured against the quota page rather than guessed: a benchmark run used
# ~120k total tokens and the account's peak output was 39k/min, so output is
# roughly a THIRD of the total, not a quarter. Rounded up, because
# under-reserving costs a 429 - which fails the call - while over-reserving
# only costs latency, and latency is the cheaper mistake by a wide margin.
OUTPUT_TOKEN_SHARE = 0.4


def estimate_tokens(prompt: str = "", *, pages: int = 0) -> int:
    """Pre-call estimate of TOTAL tokens, used for the per-model reservation."""
    return len(prompt) // CHARS_PER_TOKEN + pages * ESTIMATED_TOKENS_PER_PAGE


def estimate_output_tokens(estimated_total: int) -> int:
    """Pre-call estimate of OUTPUT tokens only.

    A crude fraction on purpose. The reservation is reconciled against the
    real `output_tokens` the moment the call returns, so the estimate only
    has to be the right order of magnitude - and a structured extraction
    call is overwhelmingly input, because the page goes up and a short JSON
    object comes back.
    """
    return max(1, int(estimated_total * OUTPUT_TOKEN_SHARE))


def workspace_id() -> str:
    """What the output quota is scoped to.

    The Databricks workspace host, because that is what the quota belongs to -
    every model behind one gateway shares it. Falls back to a constant so a
    direct-to-provider setup still has one coherent bucket.
    """
    host = settings.databricks_host or ""
    return host.rstrip("/").rsplit("/", 1)[-1] or "default"


@dataclass
class GateSlot:
    """A held slot. The caller reports what the call actually cost."""

    request_id: str
    provider: str
    model: str
    estimated_tokens: int
    job_id: str | None = None
    actual_tokens: int = 0
    # The workspace-scoped OUTPUT reservation. Separate from `estimated_tokens`
    # because the provider limits a different quantity at a different scope -
    # see `limits.output_tpm_key`.
    estimated_output_tokens: int = 0
    actual_output_tokens: int = 0
    workspace: str = ""
    waited_seconds: float = 0.0
    # Set when the call raised. A failed call consumed no tokens.
    failed: bool = False
    _recorded: bool = field(default=False, repr=False)

    def record_actual(self, tokens: int, output_tokens: int = 0) -> None:
        """Report real usage so the reservations can be corrected.

        Not calling this is not fatal - the estimate simply stands - but it
        makes the limiter drift from reality, so the gate warns on exit.

        `output_tokens` corrects the WORKSPACE bucket, which is the one the
        provider actually enforces. Callers that pass only a total still work;
        their output reservation simply stands at the estimate.
        """
        self.actual_tokens = tokens
        if output_tokens:
            self.actual_output_tokens = output_tokens
        self._recorded = True


@asynccontextmanager
async def llm_gate(
    provider: str,
    model: str,
    *,
    job_id: str | None = None,
    estimated_tokens: int = 0,
):
    """Acquire a slot for one model call. Releases everything on exit.

    `job_id` is how a call is charged to a budget. Optional only so one-off
    scripts work; anything inside a pipeline must pass it, or its spend is
    invisible.
    """
    ceilings = limits_for(provider)
    request_id = str(uuid.uuid4())
    redis = await limits.get_redis()
    workspace = workspace_id()
    estimated_output = estimate_output_tokens(estimated_tokens)
    slot = GateSlot(
        request_id=request_id,
        provider=provider,
        model=model,
        estimated_tokens=estimated_tokens,
        job_id=job_id,
        estimated_output_tokens=estimated_output,
        actual_output_tokens=estimated_output,
        workspace=workspace,
    )

    try:
        await _check_job_budget(redis, job_id)
        slot.waited_seconds = await _wait_for_capacity(
            redis, provider, model, estimated_tokens, ceilings, request_id,
            workspace=workspace, estimated_output=estimated_output,
        )

        log.info(
            "Gate acquired | %s/%s job=%s est=%d tokens (%d output) waited=%.1fs",
            provider, model, job_id or "-", estimated_tokens, estimated_output,
            slot.waited_seconds,
        )
        yield slot

    except Exception:
        # The call failed, so it spent nothing. Release the reservation in
        # full rather than letting the estimate stand - otherwise every failed
        # call permanently inflates this minute's token counter, and a
        # document that retries a few times eats a budget it never used.
        slot.failed = True
        raise
    finally:
        await _release(redis, slot)


async def _check_job_budget(redis, job_id: str | None) -> None:
    """Fail the job if it has already spent its allowance."""
    if not job_id:
        return
    spent = await limits.get_job_budget(redis, job_id)
    if spent >= settings.job_token_budget:
        log.error(
            "Job %s exhausted its token budget (%d >= %d)",
            job_id, spent, settings.job_token_budget,
        )
        raise BudgetExceeded(
            f"Job {job_id} has used {spent} tokens, over the "
            f"{settings.job_token_budget} budget."
        )


async def _wait_for_capacity(
    redis,
    provider: str,
    model: str,
    estimated_tokens: int,
    ceilings: ProviderLimits,
    slot_id: str,
    *,
    workspace: str,
    estimated_output: int,
) -> float:
    """Block until a concurrency slot, tokens and a request are all available.

    Returns how long it waited.

    ORDER MATTERS, and it is the opposite of the obvious one. Concurrency is
    checked FIRST because it is the cheap, fast-moving resource: two Redis
    calls, and it frees up on the scale of one model call. Token and request
    budgets are checked after, because they are per-minute and reserving them
    only to hand them straight back is pure churn.

    Doing it the other way round means a waiting caller performs ~10 round
    trips per poll - reserve tokens, reserve request, fail on the slot, give
    both back - and with twenty callers polling at once that traffic alone
    dominates the wait. Checking the slot first makes the waiting path two
    round trips.
    """
    waited = 0.0

    while waited < MAX_WAIT_SECONDS:
        # 1. Concurrency. Cheap, and the one most likely to be the blocker.
        if not await limits.try_acquire_slot(
            redis, provider, model, slot_id, ceilings.max_concurrent
        ):
            await asyncio.sleep(POLL_INTERVAL_SECONDS)
            waited += POLL_INTERVAL_SECONDS
            continue

        # 2. Token budget for this minute.
        if not await limits.try_reserve_tokens(
            redis, provider, model, estimated_tokens, ceilings.tokens_per_minute
        ):
            await limits.release_slot(redis, provider, model, slot_id)
            waited += await _sleep_until_next_bucket("TPM", provider, model)
            continue

        # 3. Request count for this minute.
        if not await limits.try_reserve_request(
            redis, provider, model, ceilings.requests_per_minute
        ):
            await limits.reconcile_tokens(redis, provider, model, estimated_tokens, 0)
            await limits.release_slot(redis, provider, model, slot_id)
            waited += await _sleep_until_next_bucket("RPM", provider, model)
            continue

        # 4. The WORKSPACE output-token minute - the one the provider actually
        #    enforces, shared across every model behind the gateway. Checked
        #    last because it is the most expensive to give back.
        if not await limits.try_reserve_output_tokens(
            redis,
            workspace,
            model,
            estimated_output,
            ceilings.output_tokens_per_minute,
        ):
            await limits.release_request(redis, provider, model)
            await limits.reconcile_tokens(redis, provider, model, estimated_tokens, 0)
            await limits.release_slot(redis, provider, model, slot_id)
            waited += await _sleep_until_next_bucket(
                "workspace output TPM", provider, model
            )
            continue

        return waited

    raise GateTimeout(
        f"Waited {MAX_WAIT_SECONDS}s for a {provider}/{model} slot. "
        "Check the configured ceilings and whether slots are being released."
    )


async def _sleep_until_next_bucket(reason: str, provider: str, model: str) -> float:
    delay = limits.seconds_until_next_bucket()
    log.info(
        "%s limit reached for %s/%s - waiting %.1fs for the next bucket",
        reason, provider, model, delay,
    )
    await asyncio.sleep(delay)
    return delay


async def _release(redis, slot: GateSlot) -> None:
    """Give back the slot and correct the counters. Never raises."""
    try:
        # `request_id` is generated once per acquisition and carried on the
        # slot, rather than looked up from a task-keyed map: id(task) is reused
        # after a task is collected, so a later call could be handed a previous
        # call's slot id and release the wrong entry.
        await limits.release_slot(
            redis, slot.provider, slot.model, slot.request_id
        )

        if slot.failed:
            # Give the whole reservation back: actual usage was zero.
            await limits.reconcile_tokens(
                redis, slot.provider, slot.model, slot.estimated_tokens, 0
            )
            await limits.release_output_tokens(
                redis, slot.workspace, slot.model, slot.estimated_output_tokens
            )
            log.info(
                "Gate released | %s/%s call failed, %d reserved tokens returned",
                slot.provider, slot.model, slot.estimated_tokens,
            )
            return

        if not slot._recorded:
            log.warning(
                "Gate exited without record_actual() for %s/%s - the estimate "
                "of %d tokens will stand and the limiter will drift.",
                slot.provider, slot.model, slot.estimated_tokens,
            )
            return

        await limits.reconcile_tokens(
            redis, slot.provider, slot.model, slot.estimated_tokens, slot.actual_tokens
        )
        await limits.reconcile_output_tokens(
            redis, slot.workspace, slot.model,
            slot.estimated_output_tokens, slot.actual_output_tokens,
        )
        if slot.job_id:
            total = await limits.add_to_job_budget(
                redis, slot.job_id, slot.actual_tokens
            )
            # Beside the total, split by provider and model - that is what
            # `job_complete` reports and what makes cost attributable to a
            # stage rather than to the run as a whole.
            await limits.add_usage(
                redis, slot.job_id, slot.provider, slot.model, slot.actual_tokens
            )
            log.info(
                "Gate released | %s/%s job=%s actual=%d tokens (job total %d)",
                slot.provider, slot.model, slot.job_id, slot.actual_tokens, total,
            )
        else:
            log.info(
                "Gate released | %s/%s actual=%d tokens",
                slot.provider, slot.model, slot.actual_tokens,
            )
    except Exception as exc:
        # A failure here must not mask whatever the caller was doing.
        log.error("Failed to release gate slot cleanly: %s", exc)
