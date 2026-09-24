"""Reachability check for the Databricks-routed LLM clients.

Not a correctness test - just: given DATABRICKS_TOKEN/DATABRICKS_HOST, can
every Gemini and GPT model configured in core/llm_clients actually be
reached and answer, through the same unified `get_client(...).generate(...)`
interface every service uses? Skipped entirely without credentials, since it
makes real, billed API calls - not meant to run as part of the default
`pytest` pass, only ad hoc:

    uv run pytest tests/test_llm_clients.py -v -s

Claude/Anthropic is intentionally left out - the project standardised on
Gemini + GPT routed through Databricks; Anthropic stays a direct, unrouted
option (core/llm_clients/anthropic_client.py).
"""

from __future__ import annotations

import pytest

from token_daddy.config import settings
from token_daddy.llm_clients import Provider, get_client
from token_daddy.llm_clients.gemini_client import (
    GEMINI_3_1_PRO_PREVIEW,
    GEMINI_3_6_FLASH,
    GEMINI_3_7_FLASH,
)
from token_daddy.llm_clients.openai_client import (
    GPT_5_4,
    GPT_5_4_MINI,
    GPT_5_4_NANO,
    GPT_5_6_LUNA,
    GPT_5_6_TERRA,
)

PROMPT = "Reply with exactly one word: ok"

MODELS = [
    (Provider.GEMINI, GEMINI_3_1_PRO_PREVIEW),
    (Provider.GEMINI, GEMINI_3_7_FLASH),
    (Provider.GEMINI, GEMINI_3_6_FLASH),
    (Provider.OPENAI, GPT_5_4),
    (Provider.OPENAI, GPT_5_4_MINI),
    (Provider.OPENAI, GPT_5_4_NANO),
    (Provider.OPENAI, GPT_5_6_TERRA),
    (Provider.OPENAI, GPT_5_6_LUNA),
]

pytestmark = pytest.mark.skipif(
    not (settings.databricks_token and settings.databricks_host),
    reason="DATABRICKS_TOKEN/DATABRICKS_HOST not set - this test makes real, billed API calls",
)


@pytest.mark.parametrize("provider,model", MODELS, ids=[model for _, model in MODELS])
def test_model_is_reachable(provider: Provider, model: str) -> None:
    client = get_client(provider, model)
    # Generous headroom on purpose: reasoning models (Gemini 3.x always, GPT
    # depending on effort) bill thinking tokens out of the same budget as the
    # visible answer, so a tight cap can burn the whole thing on invisible
    # reasoning and leave nothing for actual text. Not passing thinking_level
    # at all - accepted effort values differ per model/provider on this
    # gateway (e.g. gpt-5.6-luna rejects "minimal"), so leaving it unset and
    # taking each model's default is the one setting guaranteed to be valid
    # everywhere.
    reply = client.generate(PROMPT, max_output_tokens=512)

    assert reply.text.strip(), f"{provider}/{model} returned an empty response"
    assert reply.model == model
    assert reply.provider == str(provider)

    print(
        f"\n{provider}/{model}: {reply.text.strip()!r} "
        f"({reply.usage.total_tokens} tokens, cost={reply.estimated_cost_usd})"
    )
