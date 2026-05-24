from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from alpha.db import AlphaStore
from alpha.queues import build_candidate_queues, classify_candidate


def _store(tmp: str) -> AlphaStore:
    store = AlphaStore(Path(tmp) / "alpha.db")
    store.init()
    return store


def _candidate(store: AlphaStore, expression: str, status: str, metrics=None, checks=None):
    candidate_id = store.insert_candidate(
        expression,
        {"region": "USA", "universe": "TOP3000", "delay": 1, "neutralization": "INDUSTRY"},
        "model:g1",
    )
    if metrics is not None or checks is not None:
        store.update_candidate(
            candidate_id,
            metrics_json=json.dumps(metrics or {}, sort_keys=True),
            checks_json=json.dumps(checks or {}, sort_keys=True),
            alpha_id=f"A{candidate_id}",
        )
    if status != "generated":
        store.transition(candidate_id, status)
    return candidate_id


class CandidateQueueTests(unittest.TestCase):
    def test_build_candidate_queues_prioritizes_pending_and_near_threshold(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = _store(tmp)
            pending_id = _candidate(
                store,
                "rank(pending_signal)",
                "check_pending",
                metrics={"sharpe": 2.8, "fitness": 1.4, "turnover": 0.2},
                checks={"PROD_CORRELATION": {"status": "PENDING"}},
            )
            optimize_id = _candidate(
                store,
                "rank(optimize_signal)",
                "failed",
                metrics={"sharpe": 2.55, "fitness": 1.25, "turnover": 0.8},
                checks={"LOW_TURNOVER": {"status": "FAIL"}},
            )
            trash_id = _candidate(
                store,
                "rank(trash_signal)",
                "failed",
                metrics={"sharpe": 0.1, "fitness": 0.02, "turnover": 0.2},
                checks={"LOW_SHARPE": {"status": "FAIL"}},
            )

            queues = build_candidate_queues(store)

        self.assertEqual(queues["pending"][0]["id"], pending_id)
        self.assertEqual(queues["optimize"][0]["id"], optimize_id)
        self.assertEqual(queues["trash"][0]["id"], trash_id)
        self.assertGreater(queues["pending"][0]["priority"], queues["optimize"][0]["priority"])

    def test_classify_candidate_keeps_hard_correlation_block_out_of_optimize(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = _store(tmp)
            candidate_id = _candidate(
                store,
                "rank(correlated_signal)",
                "failed",
                metrics={"sharpe": 3.0, "fitness": 1.8, "turnover": 0.2},
                checks={"SELF_CORRELATION": {"status": "FAIL", "value": 0.91}},
            )
            row = store.get_candidate(candidate_id)

            queue, reason, priority = classify_candidate(row, store.events_for_candidate(candidate_id))

        self.assertEqual(queue, "trash")
        self.assertEqual(reason, "hard_blocker")
        self.assertLess(priority, 0)

    def test_build_candidate_queues_identifies_submitable_and_explore_seed(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = _store(tmp)
            approved_id = _candidate(store, "rank(good_signal)", "approved", metrics={"sharpe": 2.9, "fitness": 1.6})
            seed_id = _candidate(store, "rank(new_signal)", "generated")

            queues = build_candidate_queues(store)

        self.assertEqual(queues["submitable"][0]["id"], approved_id)
        self.assertEqual(queues["explore_seed"][0]["id"], seed_id)


if __name__ == "__main__":
    unittest.main()
