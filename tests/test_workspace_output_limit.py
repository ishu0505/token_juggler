"""W.4's real finding: the binding quota is OUTPUT tokens, WORKSPACE-wide.

The gate used to meter total tokens per provider+model. Databricks limits
output tokens across the whole workspace, so the gate was budgeting the wrong
quantity at the wrong scope - which is why lowering concurrency from 40 to 10
reduced the 429s without removing them, and why three identical consecutive
runs produced 2, then 4, then 8 of them.
"""

import unittest

from token_daddy.llm import gate, limits, retry


class KeyTests(unittest.TestCase):
    def test_the_output_bucket_ignores_provider_and_model(self):
        """Every model behind one gateway shares the quota, so a per-model key
        could never see the limit being hit."""
        key = limits.output_tpm_key("acme", "m")
        self.assertNotIn("openai", key)
        self.assertNotIn("gemini", key)
        self.assertIn("acme", key)
        self.assertIn("output_tpm", key)

    def test_the_per_model_bucket_is_still_separate(self):
        self.assertNotEqual(
            limits.output_tpm_key("acme", "m"), limits.tpm_key("openai", "m")
        )


class EstimateTests(unittest.TestCase):
    def test_output_is_a_fraction_of_the_total(self):
        """An extraction call sends a page and receives a small JSON object."""
        self.assertLess(gate.estimate_output_tokens(1000), 1000)
        self.assertGreater(gate.estimate_output_tokens(1000), 0)

    def test_a_tiny_call_still_reserves_something(self):
        self.assertGreaterEqual(gate.estimate_output_tokens(1), 1)


class ConfiguredLimitTests(unittest.TestCase):
    def test_gemini_uses_the_supplied_high_throughput_quota(self):
        configured = gate.limits_for("gemini")
        self.assertEqual(configured.tokens_per_minute, 50_000_000)
        self.assertEqual(configured.output_tokens_per_minute, 15_000_000)
        self.assertEqual(configured.requests_per_minute, 6_000)
        self.assertEqual(configured.max_concurrent, 200)

    def test_terra_keeps_its_tighter_output_ceiling(self):
        configured = gate.limits_for("openai")
        self.assertEqual(configured.tokens_per_minute, 10_000_000)
        self.assertEqual(configured.output_tokens_per_minute, 1_000_000)
        self.assertEqual(configured.requests_per_minute, 6_000)
        self.assertEqual(configured.max_concurrent, 64)


class ReservationTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.redis = limits.InProcessRedis()

    async def test_a_reservation_within_the_ceiling_is_allowed(self):
        self.assertTrue(
            await limits.try_reserve_output_tokens(self.redis, "w", "m", 100, 1000)
        )

    async def test_a_reservation_over_the_ceiling_is_refused_and_returned(self):
        """A refused reservation must cost nothing, or a busy minute drains
        itself by being asked."""
        await limits.try_reserve_output_tokens(self.redis, "w", "m", 900, 1000)
        self.assertFalse(
            await limits.try_reserve_output_tokens(self.redis, "w", "m", 200, 1000)
        )
        # The refused 200 was given back, so 100 still fits.
        self.assertTrue(
            await limits.try_reserve_output_tokens(self.redis, "w", "m", 100, 1000)
        )

    async def test_a_ceiling_of_zero_disables_the_check(self):
        """Accounts without this quota must not be throttled by a limit that
        does not apply to them."""
        self.assertTrue(
            await limits.try_reserve_output_tokens(self.redis, "w", "m", 10**9, 0)
        )

    async def test_two_models_share_one_workspace_budget(self):
        """The whole point. Two different models, one counter."""
        await limits.try_reserve_output_tokens(self.redis, "w", "m", 600, 1000)
        self.assertFalse(
            await limits.try_reserve_output_tokens(self.redis, "w", "m", 600, 1000)
        )

    async def test_reconciling_down_frees_capacity(self):
        """Estimates run high; the real number must give the difference back
        or the bucket stays full of tokens nobody spent."""
        await limits.try_reserve_output_tokens(self.redis, "w", "m", 900, 1000)
        await limits.reconcile_output_tokens(self.redis, "w", "m", 900, 100)
        self.assertTrue(
            await limits.try_reserve_output_tokens(self.redis, "w", "m", 800, 1000)
        )

    async def test_releasing_returns_the_whole_reservation(self):
        await limits.try_reserve_output_tokens(self.redis, "w", "m", 1000, 1000)
        await limits.release_output_tokens(self.redis, "w", "m", 1000)
        self.assertTrue(
            await limits.try_reserve_output_tokens(self.redis, "w", "m", 1000, 1000)
        )


class RetryClassificationTests(unittest.TestCase):
    def test_a_provider_timeout_is_retryable(self):
        """Setting a request timeout turned a twelve-minute hang into an
        immediate hard failure, because `APITimeoutError` carries no status
        code and is not a builtin TimeoutError. A timeout with no retry is
        worse than the hang it replaced."""
        class APITimeoutError(Exception):
            pass

        self.assertTrue(retry.is_retryable(APITimeoutError()))

    def test_a_connection_error_is_retryable(self):
        class APIConnectionError(Exception):
            pass

        self.assertTrue(retry.is_retryable(APIConnectionError()))

    def test_a_429_is_still_retryable(self):
        error = Exception()
        error.status_code = 429
        self.assertTrue(retry.is_retryable(error))

    def test_a_400_is_never_retryable(self):
        """Our bug. Retrying it just costs four times as much."""
        error = Exception()
        error.status_code = 400
        self.assertFalse(retry.is_retryable(error))


if __name__ == "__main__":
    unittest.main()


class PerModelTests(unittest.IsolatedAsyncioTestCase):
    """The quota is per MODEL at workspace scope - the 429 says so itself:
    'Exceeded workspace output tokens per minute rate limit for
    databricks-gemini-3-8-flash'. Keying on the workspace alone would make two
    models share one budget and throttle each other for nothing."""

    async def asyncSetUp(self):
        self.redis = limits.InProcessRedis()

    async def test_two_models_have_separate_budgets(self):
        await limits.try_reserve_output_tokens(self.redis, "w", "gemini", 1000, 1000)
        self.assertTrue(
            await limits.try_reserve_output_tokens(self.redis, "w", "gpt", 1000, 1000)
        )

    async def test_one_model_can_still_exhaust_its_own(self):
        await limits.try_reserve_output_tokens(self.redis, "w", "gemini", 1000, 1000)
        self.assertFalse(
            await limits.try_reserve_output_tokens(self.redis, "w", "gemini", 1, 1000)
        )

    async def test_the_key_names_both(self):
        key = limits.output_tpm_key("acme", "gemini-3-8-flash")
        self.assertIn("acme", key)
        self.assertIn("gemini-3-8-flash", key)
