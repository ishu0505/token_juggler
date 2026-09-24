"""Construct the structured-output client selected for a pipeline stage."""

from __future__ import annotations

from token_daddy.llm.providers import gemini, openai


def client_for(
    provider: str,
    model: str | None = None,
    *,
    thinking_level: str | None = None,
    reasoning_effort: str | None = None,
):
    """Return a client with optional instance-local reasoning controls."""
    if provider == "gemini":
        return (
            gemini.GeminiClient(thinking_level=thinking_level),
            model or gemini.DEFAULT_MODEL,
        )
    if provider == "openai":
        return (
            openai.OpenAIClient(reasoning_effort=reasoning_effort),
            model or openai.DEFAULT_MODEL,
        )
    raise ValueError(f"Unsupported structured-output provider: {provider}")
