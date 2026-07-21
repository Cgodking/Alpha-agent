from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone

from alpha.clients import LocalAIClient, LocalBrainClient, _parse_retry_after


class RetryAfterTests(unittest.TestCase):
    def test_parses_delay_seconds(self):
        self.assertEqual(_parse_retry_after("5"), 5.0)

    def test_missing_or_empty_returns_default(self):
        self.assertEqual(_parse_retry_after(None, default=1.0), 1.0)
        self.assertEqual(_parse_retry_after("", default=2.0), 2.0)

    def test_unparseable_returns_default(self):
        self.assertEqual(_parse_retry_after("soon", default=3.0), 3.0)

    def test_http_date_form_does_not_crash(self):
        future = (datetime.now(timezone.utc) + timedelta(seconds=10)).strftime("%a, %d %b %Y %H:%M:%S GMT")
        self.assertGreaterEqual(_parse_retry_after(future), 0.0)
        # A past date yields a clamped, non-negative wait rather than a crash.
        self.assertEqual(_parse_retry_after("Wed, 21 Oct 2015 07:28:00 GMT"), 0.0)

    def test_value_is_clamped_to_maximum(self):
        self.assertEqual(_parse_retry_after("99999", maximum=30.0), 30.0)


class ClientTests(unittest.TestCase):
    def test_local_ai_client_returns_deterministic_candidates(self):
        client = LocalAIClient(expressions=["rank(close)", "rank(-returns)"])

        candidates = client.generate_candidates(batch_size=2, context={"region": "USA"})

        self.assertEqual([candidate.expression for candidate in candidates], ["rank(close)", "rank(-returns)"])
        self.assertEqual(candidates[0].settings["region"], "USA")

    def test_local_ai_client_does_not_put_research_context_in_settings(self):
        client = LocalAIClient(expressions=["rank(close)"])

        candidates = client.generate_candidates(
            batch_size=1,
            context={
                "region": "USA",
                "cycle_plan": {"mode": "explore", "budget": {"batch_size": 8}},
                "research_context": {"recent_failures": []},
            },
        )

        self.assertNotIn("research_context", candidates[0].settings)
        self.assertNotIn("cycle_plan", candidates[0].settings)

    def test_local_brain_client_simulates_and_does_not_submit_in_dry_run(self):
        client = LocalBrainClient()

        result = client.simulate("rank(close)", {"region": "USA"})
        submitted = client.submit_alpha(result.alpha_id, dry_run=True)

        self.assertTrue(result.alpha_id.startswith("LOCAL"))
        self.assertGreaterEqual(result.metrics["sharpe"], 0)
        self.assertFalse(submitted.submitted)
        self.assertEqual(submitted.stage, "DRY_RUN")


if __name__ == "__main__":
    unittest.main()
