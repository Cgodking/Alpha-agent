from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from alpha.db import AlphaStore
from alpha.scheduler import build_cycle_plan


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
            alpha_id=f"A{candidate_id}",
            metrics_json=json.dumps(metrics or {}, sort_keys=True),
            checks_json=json.dumps(checks or {}, sort_keys=True),
        )
    if status != "generated":
        store.transition(candidate_id, status)
    return candidate_id


class SchedulerTests(unittest.TestCase):
    def test_scheduler_recovers_pending_before_new_exploration(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = _store(tmp)
            pending_id = _candidate(
                store,
                "rank(pending_signal)",
                "check_pending",
                metrics={"sharpe": 2.8, "fitness": 1.4, "turnover": 0.2},
                checks={"PROD_CORRELATION": {"status": "PENDING"}},
            )

            plan = build_cycle_plan(
                store,
                {"region": "USA", "universe": "TOP3000", "delay": 1, "neutralization": "INDUSTRY"},
            )

        self.assertEqual(plan["mode"], "recover_pending")
        self.assertEqual(plan["target_candidate_id"], pending_id)
        self.assertIn("pending", plan["reason"])

    def test_scheduler_optimizes_near_threshold_candidate(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = _store(tmp)
            optimize_id = _candidate(
                store,
                "rank(optimize_signal)",
                "failed",
                metrics={"sharpe": 2.55, "fitness": 1.2, "turnover": 0.8},
                checks={"LOW_TURNOVER": {"status": "FAIL"}},
            )

            plan = build_cycle_plan(
                store,
                {"region": "USA", "universe": "TOP3000", "delay": 1, "neutralization": "INDUSTRY"},
            )

        self.assertEqual(plan["mode"], "optimize")
        self.assertEqual(plan["target_candidate_id"], optimize_id)

    def test_scheduler_prefers_setting_sweep_for_approved_dry_run_candidate(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = _store(tmp)
            approved_id = _candidate(
                store,
                "rank(good_signal)",
                "approved",
                metrics={"sharpe": 2.9, "fitness": 1.7, "turnover": 0.2},
                checks={"SELF_CORRELATION": {"status": "PASS"}},
            )

            plan = build_cycle_plan(
                store,
                {"region": "USA", "universe": "TOP3000", "delay": 1, "neutralization": "INDUSTRY"},
            )

        self.assertEqual(plan["mode"], "setting_sweep")
        self.assertEqual(plan["target_candidate_id"], approved_id)

    def test_scheduler_defaults_to_explore_without_higher_value_queue(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = _store(tmp)

            plan = build_cycle_plan(
                store,
                {"region": "USA", "universe": "TOP3000", "delay": 1, "neutralization": "INDUSTRY"},
            )

        self.assertEqual(plan["mode"], "explore")
        self.assertIsNone(plan["target_candidate_id"])

    def test_scheduler_cools_down_repeated_quality_stop_loss(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = _store(tmp)
            store.record_event(None, "quality_stop_loss", {"scope": {"region": "USA"}, "quality_stop_reason": "bad_full_batch"})
            store.record_event(None, "quality_stop_loss", {"scope": {"region": "USA"}, "quality_stop_reason": "bad_full_batch"})

            plan = build_cycle_plan(
                store,
                {"region": "USA", "universe": "TOP3000", "delay": 1, "neutralization": "INDUSTRY"},
            )

        self.assertEqual(plan["mode"], "cooldown")
        self.assertIn("quality_stop_loss", plan["reason"])

    def test_scheduler_does_not_cool_down_for_platform_rate_limit_errors(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = _store(tmp)
            candidate_id = _candidate(store, "rank(rate_limited_signal)", "failed")
            store.record_event(candidate_id, "simulation_error", {"error": "HTTP 429 Retry-After: 5"})

            plan = build_cycle_plan(
                store,
                {"region": "USA", "universe": "TOP3000", "delay": 1, "neutralization": "INDUSTRY"},
            )

        self.assertNotEqual(plan["mode"], "cooldown")


if __name__ == "__main__":
    unittest.main()
