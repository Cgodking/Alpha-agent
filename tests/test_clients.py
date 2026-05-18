from __future__ import annotations

import unittest

from alpha.clients import LocalAIClient, LocalBrainClient


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
            context={"region": "USA", "research_context": {"recent_failures": []}},
        )

        self.assertNotIn("research_context", candidates[0].settings)

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
