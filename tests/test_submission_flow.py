from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from alpha.clients import LocalAIClient, LocalBrainClient
from alpha.db import AlphaStore
from alpha.guards import SubmissionPolicy
from alpha.submission import submit_approved_candidates
from alpha.worker import AlphaWorker


class SubmissionFlowTests(unittest.TestCase):
    def test_submit_approved_candidates_dry_run_records_events(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = AlphaStore(Path(tmp) / "alpha.db")
            store.init()
            worker = AlphaWorker(
                store=store,
                ai_client=LocalAIClient(expressions=["rank(mdl_mock_score)"]),
                brain_client=LocalBrainClient(),
                policy=SubmissionPolicy(auto_submit=False),
                batch_size=1,
            )
            worker.run_once()

            summary = submit_approved_candidates(store, LocalBrainClient(), SubmissionPolicy(auto_submit=False))

            candidate = store.list_candidates()[0]
            events = store.events_for_candidate(candidate["id"])
            self.assertEqual(summary["processed"], 1)
            self.assertEqual(summary["dry_run"], 1)
            self.assertEqual(candidate["status"], "approved")
            self.assertTrue(any(event["event_type"] == "dry_run_submit" for event in events))

    def test_submit_approved_candidates_respects_round_cap(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = AlphaStore(Path(tmp) / "alpha.db")
            store.init()
            worker = AlphaWorker(
                store=store,
                ai_client=LocalAIClient(
                    expressions=[
                        "rank(mdl_mock_score)",
                        "group_rank(ts_rank(divide(mdl_mock_score, cap), 63), industry)",
                        "group_rank(ts_rank(mdl_mock_score, 22), industry)",
                        "rank(ts_mean(mdl_mock_score, 22))",
                        "group_rank(ts_zscore(mdl_mock_score, 20), industry)",
                    ]
                ),
                brain_client=LocalBrainClient(),
                policy=SubmissionPolicy(auto_submit=False),
                batch_size=5,
            )
            worker.run_once()

            summary = submit_approved_candidates(
                store,
                LocalBrainClient(),
                SubmissionPolicy(auto_submit=False, max_final_submits_per_round=4),
            )

            self.assertEqual(summary["processed"], 4)
            self.assertEqual(summary["skipped"], 1)

    def test_submit_approved_candidates_skips_when_platform_count_fails(self):
        class FailingCountBrain(LocalBrainClient):
            def count_submitted_alphas(self, start_date: str, end_date: str) -> int:
                raise RuntimeError("count failed")

        with tempfile.TemporaryDirectory() as tmp:
            store = AlphaStore(Path(tmp) / "alpha.db")
            store.init()
            worker = AlphaWorker(
                store=store,
                ai_client=LocalAIClient(expressions=["rank(mdl_mock_score)"]),
                brain_client=LocalBrainClient(),
                policy=SubmissionPolicy(auto_submit=False),
                batch_size=1,
            )
            worker.run_once()

            summary = submit_approved_candidates(store, FailingCountBrain(), SubmissionPolicy(auto_submit=True))

            candidate = store.list_candidates()[0]
            self.assertEqual(candidate["status"], "approved")
            self.assertEqual(summary["processed"], 0)
            self.assertEqual(summary["skipped"], 1)
            events = store.events_for_candidate(candidate["id"])
            self.assertTrue(any("platform_count_unavailable" in event["metadata_json"] for event in events))


    def test_worker_seeds_round_cap_from_platform_submitted_count(self):
        # Daily/account cap must hold across daemon cycles: if the platform already
        # shows max submissions in the trading-day window, a fresh run_once submits none.
        with tempfile.TemporaryDirectory() as tmp:
            store = AlphaStore(Path(tmp) / "alpha.db")
            store.init()
            brain = LocalBrainClient()
            # Simulate that the cap was already reached on the platform earlier today.
            brain.submitted.extend(["EXISTING1", "EXISTING2", "EXISTING3", "EXISTING4"])
            worker = AlphaWorker(
                store=store,
                ai_client=LocalAIClient(
                    expressions=[
                        "rank(mdl_mock_score)",
                        "group_rank(ts_rank(mdl_mock_score, 22), industry)",
                    ]
                ),
                brain_client=brain,
                policy=SubmissionPolicy(auto_submit=True, max_final_submits_per_round=4),
                batch_size=2,
            )

            summary = worker.run_once()

            self.assertEqual(summary["submitted"], 0)
            self.assertEqual(len(brain.submitted), 4)
            self.assertTrue(
                any("ROUND_SUBMIT_LIMIT_REACHED" in (event["metadata_json"] or "")
                    for candidate in store.list_candidates()
                    for event in store.events_for_candidate(candidate["id"]))
            )

    def test_worker_platform_count_failure_blocks_submission(self):
        class FailingCountBrain(LocalBrainClient):
            def count_submitted_alphas(self, start_date: str, end_date: str) -> int:
                raise RuntimeError("count failed")

        with tempfile.TemporaryDirectory() as tmp:
            store = AlphaStore(Path(tmp) / "alpha.db")
            store.init()
            worker = AlphaWorker(
                store=store,
                ai_client=LocalAIClient(expressions=["rank(mdl_mock_score)"]),
                brain_client=FailingCountBrain(),
                policy=SubmissionPolicy(auto_submit=True, max_final_submits_per_round=4),
                batch_size=1,
            )

            summary = worker.run_once()

            # Unavailable platform count is treated as "cap reached": no new submits.
            self.assertEqual(summary["submitted"], 0)


if __name__ == "__main__":
    unittest.main()
