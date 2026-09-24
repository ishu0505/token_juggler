"""Token accounting and cost estimation across LLM calls.

Two pieces:

* ``estimate_cost`` - turn token counts into dollars for a known model.
* ``CostTracker`` - fold many calls into one ``Usage`` summary.

`index_it` fans out one LLM call per PDF chunk, so a single run can be
30+ calls. Each worker returns its own token counts and the pipeline folds
them into one tracker on the main thread once the futures resolve -
``CostTracker`` is deliberately **not** thread-safe, and nothing should
call ``add_raw`` from a worker.

A note on prices
----------------
``MODEL_PRICES`` starts empty on purpose. Rates change, differ per account,
and a wrong number silently produces a confident-looking wrong invoice
estimate. Anything not listed reports ``cost_usd=None`` and shows up in
``Usage.unpriced_models``, so a missing price is visible instead of being
quietly counted as $0.

Fill it in from your provider's current pricing page::

    from token_daddy.pricing import MODEL_PRICES, ModelPrice

    MODEL_PRICES["gpt-5.4-nano"] = ModelPrice(
        input_per_mtok=0.05,
        output_per_mtok=0.40,
    )

Token counts are always exact regardless of whether a price is registered.
"""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel, Field


@dataclass(frozen=True)
class ModelPrice:
    """USD per million tokens for one model."""

    input_per_mtok: float
    output_per_mtok: float

    def cost_for(self, input_tokens: int, output_tokens: int) -> float:
        """Dollars for a call with these token counts."""
        return (
            input_tokens * self.input_per_mtok + output_tokens * self.output_per_mtok
        ) / 1_000_000


# Model name -> price. Empty by default; see the module docstring.
# Reasoning/thinking tokens are billed as output by every provider we support,
# so they must already be folded into ``output_tokens`` before pricing.
MODEL_PRICES: dict[str, ModelPrice] = {}


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float | None:
    """Dollars for one call, or ``None`` when the model has no registered price."""
    price = MODEL_PRICES.get(model)
    if price is None:
        return None
    return price.cost_for(input_tokens, output_tokens)


class Usage(BaseModel):
    """Aggregate token spend across a set of LLM calls."""

    calls: int = Field(default=0, description="Number of LLM calls folded in.")
    input_tokens: int = Field(default=0, description="Total prompt tokens.")
    output_tokens: int = Field(
        default=0,
        description="Total completion tokens, including reasoning tokens.",
    )
    reasoning_tokens: int = Field(
        default=0,
        description="Thinking tokens, already counted inside output_tokens.",
    )
    total_tokens: int = Field(default=0, description="input_tokens + output_tokens.")
    cost_usd: float | None = Field(
        default=None,
        description=(
            "Estimated spend. None when no call had a registered price; "
            "otherwise the sum over priced calls only - check unpriced_models "
            "before treating it as the full bill."
        ),
    )
    models: list[str] = Field(
        default_factory=list,
        description="Distinct models used, in first-seen order.",
    )
    unpriced_models: list[str] = Field(
        default_factory=list,
        description="Models with no entry in MODEL_PRICES, so excluded from cost_usd.",
    )


class CostTracker:
    """Accumulate per-call token counts and costs into one ``Usage``.

    Not thread-safe. Build one per pipeline run on the main thread and fold
    worker results in after they resolve.
    """

    def __init__(self) -> None:
        self._calls = 0
        self._input_tokens = 0
        self._output_tokens = 0
        self._reasoning_tokens = 0
        self._cost = 0.0
        self._priced_calls = 0
        self._models: list[str] = []
        self._unpriced: list[str] = []

    def add_raw(
        self,
        *,
        input_tokens: int,
        output_tokens: int,
        cost: float | None = None,
        model: str = "unknown",
        reasoning_tokens: int = 0,
    ) -> None:
        """Fold one call's counts in.

        ``cost`` is the estimate the client already computed. When it is
        ``None`` the model is recorded as unpriced rather than counted as free.
        """
        self._calls += 1
        self._input_tokens += input_tokens
        self._output_tokens += output_tokens
        self._reasoning_tokens += reasoning_tokens

        if model not in self._models:
            self._models.append(model)

        if cost is None:
            if model not in self._unpriced:
                self._unpriced.append(model)
        else:
            self._cost += cost
            self._priced_calls += 1

    def total(self) -> Usage:
        """Snapshot everything folded in so far."""
        return Usage(
            calls=self._calls,
            input_tokens=self._input_tokens,
            output_tokens=self._output_tokens,
            reasoning_tokens=self._reasoning_tokens,
            total_tokens=self._input_tokens + self._output_tokens,
            cost_usd=self._cost if self._priced_calls else None,
            models=list(self._models),
            unpriced_models=list(self._unpriced),
        )
