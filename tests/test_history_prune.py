from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from alpha.db import AlphaStore
from alpha.history_prune import find_low_quality_history_candidates, prune_low_quality_history


class HistoryPruneTests(unittest.TestCase):
    def test_find_low_quality_history_candidates_keeps_promising_failures(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = AlphaStore(Path(tmp) / "alpha.db")
            store.init()
            settings = {"region": "MEA", "universe": "TOP300", "delay": 1, "neutralization": "MARKET"}
            bad_id = store.insert_candidate("rank(bad_signal)", settings, "model:G-1")
            store.update_candidate(
                bad_id,
                metrics_json=json.dumps({"sharpe": 0.05, "fitness": 0.01, "turnover": 0.2}),
                checks_json=json.dumps({"LOW_SHARPE": {"status": "FAIL", "value": 0.05, "limit": 1.58}}),
            )
            store.transition(bad_id, "failed", {"errors": ["LOW_SHARPE:FAIL"]})
            near_id = store.insert_candidate("rank(near_signal)", settings, "model:G-2")
            store.update_candidate(
                near_id,
                metrics_json=json.dumps({"sharpe": 1.45, "fitness": 0.85, "turnover": 0.2}),
                checks_json=json.dumps({"LOW_SHARPE": {"status": "FAIL", "value": 1.45, "limit": 1.58}}),
            )
            store.transition(near_id, "failed", {"errors": ["LOW_SHARPE:FAIL"]})

            selected = find_low_quality_history_candidates(store, settings, quality_max=0.2, limit=10)

        self.assertEqual([item["id"] for item in selected], [bad_id])
        self.assertLessEqual(selected[0]["history_noise_score"], 0.2)

    def test_prune_low_quality_history_archives_selected_candidates(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = AlphaStore(Path(tmp) / "alpha.db")
            store.init()
            settings = {"region": "USA", "universe": "TOP3000", "delay": 0, "neutralization": "INDUSTRY"}
            bad_id = store.insert_candidate("rank(dead_signal)", settings, "model:G-1")
            store.update_candidate(
                bad_id,
                metrics_json=json.dumps({"sharpe": 0.0, "fitness": 0.0}),
                checks_json=json.dumps({"LOW_SHARPE": {"status": "FAIL", "value": 0.0, "limit": 2.69}}),
            )
            store.transition(bad_id, "failed", {"errors": ["LOW_SHARPE:FAIL"]})

            dry_run = prune_low_quality_history(store, settings, quality_max=0.2, limit=10, execute=False)
            executed = prune_low_quality_history(store, settings, quality_max=0.2, limit=10, execute=True)

            self.assertEqual(dry_run["selected"], 1)
            self.assertEqual(dry_run["archived"], 0)
            self.assertEqual(executed["selected"], 1)
            self.assertEqual(executed["archived"], 1)
            self.assertIsNotNone(store.find_duplicate_candidate("rank(dead_signal)", settings))
            with self.assertRaises(KeyError):
                store.get_candidate(bad_id)


if __name__ == "__main__":
    unittest.main()
