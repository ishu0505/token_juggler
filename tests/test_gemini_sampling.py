"""What gets sent to Gemini, and what deliberately does not.

An unsupported sampling parameter is a 400, not a warning. Passing
`temperature` to a model that does not take it fails the call outright, so the
rule here is: omitting a setting is always safe, sending a wrong one never is.
"""

import unittest
from unittest.mock import patch

from token_daddy.llm.providers import gemini as gemini_client


class SamplingConfigTests(unittest.TestCase):
    def test_low_thinking_by_default(self):
        """Extraction reads a value off a page; it does not reason about it.
        Thinking tokens are billed as OUTPUT, which is the quantity the
        workspace quota limits, so this is a throughput setting too."""
        config = gemini_client._sampling_config()
        self.assertEqual(str(config["thinking_config"].thinking_level).lower(),
                         "thinkinglevel.low")

    def test_temperature_is_low_by_default(self):
        """VERIFIED against the live API - an earlier version of this test
        asserted the opposite, on a guess that 3.8 would reject it. A call
        with 0.0 succeeded, so the guess was wrong."""
        self.assertEqual(gemini_client._sampling_config()["temperature"], 0.0)

    def test_temperature_can_be_omitted_for_a_model_that_rejects_it(self):
        """An unsupported sampling parameter is a 400, not a warning, so the
        escape hatch has to stay."""
        with patch.object(gemini_client.settings, "gemini_temperature", None):
            self.assertNotIn("temperature", gemini_client._sampling_config())

    def test_temperature_is_sent_when_explicitly_configured(self):
        """A model that does take it must still be usable."""
        with patch.object(gemini_client.settings, "gemini_temperature", 0.0):
            self.assertEqual(gemini_client._sampling_config()["temperature"], 0.0)

    def test_thinking_can_be_turned_off_entirely(self):
        with patch.object(gemini_client.settings, "gemini_thinking_level", None):
            self.assertNotIn("thinking_config", gemini_client._sampling_config())

    def test_the_level_is_configurable(self):
        with patch.object(gemini_client.settings, "gemini_thinking_level", "high"):
            config = gemini_client._sampling_config()
            self.assertEqual(str(config["thinking_config"].thinking_level).lower(),
                             "thinkinglevel.high")

    def test_instance_level_wins_over_process_setting(self):
        with patch.object(gemini_client.settings, "gemini_thinking_level", "high"):
            config = gemini_client._sampling_config("low")
            self.assertEqual(str(config["thinking_config"].thinking_level).lower(),
                             "thinkinglevel.low")

    def test_the_default_model_is_the_one_being_benchmarked(self):
        self.assertEqual(gemini_client.DEFAULT_MODEL, "system.ai.gemini-3-7-flash")


class UsageAccountingTests(unittest.TestCase):
    def test_thinking_tokens_count_as_output(self):
        """They are billed as output and they fill the workspace quota. Left
        out, the usage numbers would not reconcile with the bill."""
        class Usage:
            prompt_token_count = 100
            candidates_token_count = 40
            thoughts_token_count = 60

        class Response:
            text = "{}"
            usage_metadata = Usage()

        response = gemini_client._to_response(Response(), "m")
        self.assertEqual(response.output_tokens, 100)
        self.assertEqual(response.input_tokens, 100)
        self.assertEqual(response.reasoning_tokens, 60)


if __name__ == "__main__":
    unittest.main()
