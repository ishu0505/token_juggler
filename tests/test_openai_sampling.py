"""OpenAI sampling and reasoning-effort request controls."""

from types import SimpleNamespace
from unittest.mock import patch

from token_daddy.llm.providers import openai as openai_client


def test_reasoning_effort_is_omitted_by_default() -> None:
    with patch.object(openai_client.settings, "openai_reasoning_effort", None):
        assert "reasoning_effort" not in openai_client._sampling()


def test_reasoning_effort_is_sent_when_explicit() -> None:
    with patch.object(openai_client.settings, "openai_reasoning_effort", "high"):
        assert openai_client._sampling()["reasoning_effort"] == "high"


def test_instance_reasoning_override_wins_over_process_setting() -> None:
    with patch.object(openai_client.settings, "openai_reasoning_effort", "high"):
        assert openai_client._sampling("low")["reasoning_effort"] == "low"


def test_reasoning_tokens_are_exposed_but_not_double_counted() -> None:
    response = SimpleNamespace(
        choices=[SimpleNamespace(
            message=SimpleNamespace(content="{}"),
            finish_reason="stop",
        )],
        usage=SimpleNamespace(
            prompt_tokens=100,
            completion_tokens=80,
            completion_tokens_details=SimpleNamespace(reasoning_tokens=30),
        ),
    )
    parsed = openai_client._to_response(response, "m")
    assert parsed.output_tokens == 80
    assert parsed.reasoning_tokens == 30
