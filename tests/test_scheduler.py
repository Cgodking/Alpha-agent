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


def _quality_cycle(
    store: AlphaStore,
    scope: dict,
    *,
    quality_stop: bool = True,
    probe_simulation_error: bool = False,
) -> None:
    summary = {
        "generated": 1,
        "approved": 0,
        "submitted": 0,
        "failed": 1,
        "pending": 0,
        "skipped": 0,
        "quality_stop_loss": 1 if quality_stop else 0,
    }
    if probe_simulation_error:
        summary["probe_simulation_error"] = 1
    store.record_event(
        None,
        "cycle_outcome",
        {
            "cycle_plan": {"mode": "explore", "scope": scope},
            "summary": summary,
        },
    )


def _optimize_churn_cycle(store: AlphaStore, scope: dict, target_candidate_id: int) -> None:
    store.record_event(
        None,
        "cycle_outcome",
        {
            "cycle_plan": {
                "mode": "optimize",
                "scope": scope,
                "target_candidate_id": target_candidate_id,
                "budget": {"batch_size": 4},
            },
            "summary": {
                "generated": 3,
                "approved": 0,
                "submitted": 0,
                "failed": 3,
                "pending": 0,
                "skipped": 3,
            },
        },
    )


def _explore_cycle(store: AlphaStore, scope: dict) -> None:
    store.record_event(
        None,
        "cycle_outcome",
        {
            "cycle_plan": {"mode": "explore", "scope": scope, "budget": {"batch_size": 8}},
            "summary": {
                "generated": 8,
                "approved": 0,
                "submitted": 0,
                "failed": 8,
                "pending": 0,
                "skipped": 0,
            },
        },
    )


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

    def test_scheduler_lets_fresh_generation_run_while_pending_recheck_is_in_cooldown(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = _store(tmp)
            pending_id = _candidate(
                store,
                "rank(pending_signal)",
                "check_pending",
                metrics={},
                checks={},
            )
            store.record_event(
                pending_id,
                "simulation_pending",
                {"location": "/simulations/slow", "recheck": True},
            )

            plan = build_cycle_plan(
                store,
                {"region": "USA", "universe": "TOP3000", "delay": 1, "neutralization": "INDUSTRY"},
            )

        self.assertEqual(plan["mode"], "explore")
        self.assertIn("pending_recheck_cooldown", plan["reason"])

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

    def test_scheduler_skips_submitted_field_optimize_target(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = _store(tmp)
            scope = {"region": "USA", "universe": "TOP3000", "delay": 1, "neutralization": "INDUSTRY"}
            submitted_target_id = _candidate(
                store,
                "rank(group_rank(ts_rank(vec_avg(ern7_dsu_spe),63),industry))",
                "failed",
                metrics={"sharpe": 2.9, "fitness": 1.49, "turnover": 0.3},
                checks={"LOW_FITNESS": {"status": "FAIL"}},
            )
            store.record_event(
                None,
                "experiment_plan",
                {
                    "target_settings": scope,
                    "submitted_field_avoidance": {"fields": ["ern7_dsu_spe"], "families": ["ern7_dsu"]},
                },
            )

            plan = build_cycle_plan(store, scope, batch_size=8)

        self.assertEqual(plan["mode"], "explore")
        self.assertIsNone(plan["target_candidate_id"])
        self.assertEqual(plan["budget"]["batch_size"], 8)
        self.assertEqual(plan["reason"], "submitted_field_optimize_targets_exhausted")
        self.assertIn(submitted_target_id, plan["constraints"]["blocked_optimize_target_ids"])

    def test_scheduler_uses_next_optimize_target_after_submitted_field_skip(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = _store(tmp)
            scope = {"region": "USA", "universe": "TOP3000", "delay": 1, "neutralization": "INDUSTRY"}
            _candidate(
                store,
                "rank(group_rank(ts_rank(vec_avg(ern7_dsu_spe),63),industry))",
                "failed",
                metrics={"sharpe": 3.2, "fitness": 1.49, "turnover": 0.3},
                checks={"LOW_FITNESS": {"status": "FAIL"}},
            )
            fresh_id = _candidate(
                store,
                "rank(group_rank(ts_rank(fresh_signal,63),industry))",
                "failed",
                metrics={"sharpe": 2.9, "fitness": 1.45, "turnover": 0.3},
                checks={"LOW_FITNESS": {"status": "FAIL"}},
            )
            store.record_event(
                None,
                "experiment_plan",
                {
                    "target_settings": scope,
                    "submitted_field_avoidance": {"fields": ["ern7_dsu_spe"], "families": ["ern7_dsu"]},
                },
            )

            plan = build_cycle_plan(store, scope, batch_size=8)

        self.assertEqual(plan["mode"], "optimize")
        self.assertEqual(plan["target_candidate_id"], fresh_id)

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

    def test_scheduler_enters_production_rescue_after_repeated_quality_stop_loss(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = _store(tmp)
            scope = {"region": "USA", "universe": "TOP3000", "delay": 1, "neutralization": "INDUSTRY"}
            _quality_cycle(store, scope)
            _quality_cycle(store, scope)

            plan = build_cycle_plan(store, scope)

        self.assertEqual(plan["mode"], "production_rescue")
        self.assertIn("quality_stop_loss", plan["reason"])

    def test_scheduler_does_not_enter_production_rescue_after_single_recent_quality_stop(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = _store(tmp)
            scope = {"region": "USA", "universe": "TOP500", "delay": 0, "neutralization": "INDUSTRY"}
            store.record_event(None, "quality_stop_loss", {"scope": scope, "quality_stop_reason": "old_global_event"})
            store.record_event(None, "quality_stop_loss", {"scope": scope, "quality_stop_reason": "old_global_event"})
            _quality_cycle(store, scope)

            plan = build_cycle_plan(store, scope)

        self.assertEqual(plan["mode"], "explore")
        self.assertEqual(plan["reason"], "no_higher_value_queue_available")

    def test_scheduler_ignores_quality_stop_loss_from_probe_simulation_error_cycle(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = _store(tmp)
            scope = {"region": "USA", "universe": "TOP500", "delay": 0, "neutralization": "INDUSTRY"}
            _quality_cycle(store, scope)
            _quality_cycle(store, scope, probe_simulation_error=True)

            plan = build_cycle_plan(store, scope)

        self.assertEqual(plan["mode"], "explore")
        self.assertEqual(plan["reason"], "no_higher_value_queue_available")

    def test_scheduler_does_not_bridge_quality_streak_across_probe_simulation_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = _store(tmp)
            scope = {"region": "USA", "universe": "TOP500", "delay": 0, "neutralization": "INDUSTRY"}
            _quality_cycle(store, scope)
            _quality_cycle(store, scope, probe_simulation_error=True)
            _quality_cycle(store, scope)

            plan = build_cycle_plan(store, scope)

        self.assertEqual(plan["mode"], "explore")
        self.assertEqual(plan["reason"], "no_higher_value_queue_available")

    def test_scheduler_escapes_production_rescue_after_duplicate_only_cycle_without_daemon_stop(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = _store(tmp)
            scope = {"region": "USA", "universe": "TOP500", "delay": 0, "neutralization": "INDUSTRY"}
            _quality_cycle(store, scope)
            _quality_cycle(store, scope)
            store.record_event(
                None,
                "cycle_outcome",
                {
                    "cycle_plan": {"mode": "production_rescue", "scope": scope},
                    "summary": {"generated": 0, "approved": 0, "submitted": 0, "failed": 0, "pending": 0, "skipped": 1},
                },
            )

            plan = build_cycle_plan(store, scope)

        self.assertEqual(plan["mode"], "explore")
        self.assertEqual(plan["reason"], "production_rescue_duplicate_only_recent")
        self.assertEqual(plan["constraints"]["avoid_modes"], ["production_rescue"])

    def test_scheduler_escapes_explore_after_duplicate_only_cycle_without_daemon_stop(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = _store(tmp)
            scope = {"region": "USA", "universe": "TOP500", "delay": 0, "neutralization": "INDUSTRY"}
            store.record_event(
                None,
                "cycle_outcome",
                {
                    "cycle_plan": {"mode": "explore", "scope": scope},
                    "summary": {"generated": 0, "approved": 0, "submitted": 0, "failed": 0, "pending": 0, "skipped": 1},
                },
            )

            plan = build_cycle_plan(store, scope)

        self.assertEqual(plan["mode"], "production_rescue")
        self.assertEqual(plan["reason"], "explore_duplicate_only_recent")
        self.assertEqual(plan["constraints"]["avoid_modes"], ["explore"])

    def test_scheduler_escapes_production_rescue_after_bad_quality_cycle_without_daemon_stop(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = _store(tmp)
            scope = {"region": "USA", "universe": "TOP500", "delay": 0, "neutralization": "INDUSTRY"}
            _quality_cycle(store, scope)
            _quality_cycle(store, scope)
            store.record_event(
                None,
                "cycle_outcome",
                {
                    "cycle_plan": {"mode": "production_rescue", "scope": scope},
                    "summary": {
                        "generated": 1,
                        "failed": 1,
                        "quality_stop_loss": 1,
                        "quality_stop_reason": "bad_full_batch",
                        "probe_reject": 1,
                    },
                },
            )

            plan = build_cycle_plan(store, scope)

        self.assertEqual(plan["mode"], "explore")
        self.assertEqual(plan["reason"], "production_rescue_quality_stop_loss_recent")
        self.assertEqual(plan["constraints"]["avoid_modes"], ["production_rescue"])

    def test_scheduler_escapes_production_rescue_after_probe_simulation_error_without_daemon_stop(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = _store(tmp)
            scope = {"region": "USA", "universe": "TOP500", "delay": 0, "neutralization": "INDUSTRY"}
            _quality_cycle(store, scope)
            _quality_cycle(store, scope)
            store.record_event(
                None,
                "cycle_outcome",
                {
                    "cycle_plan": {"mode": "production_rescue", "scope": scope},
                    "summary": {
                        "generated": 6,
                        "failed": 6,
                        "probe_simulation_error": 6,
                    },
                },
            )

            plan = build_cycle_plan(store, scope)

        self.assertEqual(plan["mode"], "explore")
        self.assertEqual(plan["reason"], "production_rescue_probe_simulation_error_recent")
        self.assertEqual(plan["constraints"]["avoid_modes"], ["production_rescue"])

    def test_scheduler_escapes_optimize_after_quality_stop_loss_without_daemon_stop(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = _store(tmp)
            scope = {"region": "USA", "universe": "TOP500", "delay": 0, "neutralization": "INDUSTRY"}
            store.record_event(
                None,
                "cycle_outcome",
                {
                    "cycle_plan": {"mode": "optimize", "scope": scope, "target_candidate_id": 1},
                    "summary": {
                        "generated": 4,
                        "failed": 4,
                        "quality_stop_loss": 1,
                        "quality_stop_reason": "bad_full_batch",
                    },
                },
            )

            plan = build_cycle_plan(store, scope)

        self.assertEqual(plan["mode"], "explore")
        self.assertEqual(plan["reason"], "optimize_quality_stop_loss_recent")
        self.assertEqual(plan["constraints"]["avoid_modes"], ["optimize"])

    def test_scheduler_escapes_optimize_after_repeated_unproductive_churn(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = _store(tmp)
            scope = {"region": "MEA", "universe": "TOP300", "delay": 1, "neutralization": "INDUSTRY"}
            optimize_id = _candidate(
                store,
                "rank(optimize_signal)",
                "failed",
                metrics={"sharpe": 2.55, "fitness": 1.2, "turnover": 0.8},
                checks={"LOW_TURNOVER": {"status": "FAIL"}},
            )
            _optimize_churn_cycle(store, scope, optimize_id)
            _optimize_churn_cycle(store, scope, optimize_id)

            plan = build_cycle_plan(store, scope)

        self.assertEqual(plan["mode"], "explore")
        self.assertEqual(plan["reason"], "optimize_unproductive_churn_recent")
        self.assertEqual(plan["constraints"]["avoid_modes"], ["optimize"])

    def test_scheduler_keeps_target_cooldown_across_interleaved_explore_cycles(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = _store(tmp)
            scope = {"region": "USA", "universe": "TOP3000", "delay": 1, "neutralization": "INDUSTRY"}
            optimize_id = _candidate(
                store,
                "rank(optimize_signal)",
                "failed",
                metrics={"sharpe": 2.55, "fitness": 1.2, "turnover": 0.8},
                checks={"LOW_FITNESS": {"status": "FAIL"}},
            )
            _optimize_churn_cycle(store, scope, optimize_id)
            _explore_cycle(store, scope)
            _optimize_churn_cycle(store, scope, optimize_id)

            plan = build_cycle_plan(store, scope)

        self.assertEqual(plan["mode"], "explore")
        self.assertEqual(plan["reason"], "optimize_targets_cooldown")
        self.assertIn(optimize_id, plan["constraints"]["blocked_optimize_target_ids"])
        self.assertIn("optimize", plan["constraints"]["avoid_modes"])

    def test_scheduler_cools_overexposed_primary_field_and_blocks_its_optimize_target(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = _store(tmp)
            scope = {"region": "USA", "universe": "TOP3000", "delay": 1, "neutralization": "INDUSTRY"}
            for index in range(7):
                _candidate(
                    store,
                    f"rank(ts_rank(crowded_signal,{20 + index}))",
                    "failed",
                    metrics={"sharpe": 0.2, "fitness": 0.05, "turnover": 0.2},
                    checks={"LOW_SHARPE": {"status": "FAIL"}},
                )
            optimize_id = _candidate(
                store,
                "rank(ts_rank(crowded_signal,63))",
                "failed",
                metrics={"sharpe": 2.55, "fitness": 1.2, "turnover": 0.4},
                checks={"LOW_FITNESS": {"status": "FAIL"}},
            )
            for index in range(12):
                _candidate(
                    store,
                    f"rank(ts_rank(fresh_signal_{index},63))",
                    "failed",
                    metrics={"sharpe": 0.1, "fitness": 0.02, "turnover": 0.2},
                    checks={"LOW_SHARPE": {"status": "FAIL"}},
                )

            plan = build_cycle_plan(store, scope)

        self.assertEqual(plan["mode"], "explore")
        self.assertEqual(plan["reason"], "optimize_targets_cooldown")
        self.assertIn("crowded_signal", plan["constraints"]["cooldown_fields"])
        self.assertIn(optimize_id, plan["constraints"]["blocked_optimize_target_ids"])
        exposure = plan["constraints"]["field_exposure"]["crowded_signal"]
        self.assertEqual(exposure["count"], 8)
        self.assertEqual(exposure["window_candidates"], 20)

    def test_scheduler_escapes_production_rescue_after_duplicate_only_stop(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = _store(tmp)
            scope = {"region": "USA", "universe": "TOP500", "delay": 0, "neutralization": "INDUSTRY"}
            store.record_event(None, "quality_stop_loss", {"scope": scope, "quality_stop_reason": "bad_full_batch"})
            store.record_event(None, "quality_stop_loss", {"scope": scope, "quality_stop_reason": "bad_full_batch"})
            store.set_run_state(
                "daemon",
                {
                    "status": "stopped",
                    "stop_reason": "production_rescue_duplicate_only",
                    "scope": scope,
                },
            )

            plan = build_cycle_plan(store, scope)

        self.assertEqual(plan["mode"], "explore")
        self.assertEqual(plan["reason"], "production_rescue_duplicate_only_recent")
        self.assertEqual(plan["constraints"]["avoid_modes"], ["production_rescue"])

    def test_scheduler_keeps_duplicate_only_escape_after_manual_interrupt(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = _store(tmp)
            scope = {"region": "USA", "universe": "TOP500", "delay": 0, "neutralization": "INDUSTRY"}
            store.record_event(None, "quality_stop_loss", {"scope": scope, "quality_stop_reason": "bad_full_batch"})
            store.record_event(None, "quality_stop_loss", {"scope": scope, "quality_stop_reason": "bad_full_batch"})
            store.record_event(
                None,
                "cycle_outcome",
                {
                    "cycle_plan": {"mode": "production_rescue", "scope": scope},
                    "summary": {"generated": 0, "approved": 0, "submitted": 0, "failed": 0, "pending": 0, "skipped": 1},
                },
            )
            store.record_event(None, "daemon_stopped", {"reason": "production_rescue_duplicate_only"})
            store.record_event(None, "cycle_plan", {"mode": "explore", "scope": scope})
            store.record_event(None, "daemon_stopped", {"reason": "interrupted"})
            store.set_run_state(
                "daemon",
                {
                    "status": "stopped",
                    "stop_reason": "interrupted",
                    "scope": scope,
                },
            )

            plan = build_cycle_plan(store, scope)

        self.assertEqual(plan["mode"], "explore")
        self.assertEqual(plan["reason"], "production_rescue_duplicate_only_recent")
        self.assertEqual(plan["constraints"]["avoid_modes"], ["production_rescue"])

    def test_scheduler_optimizes_probe_ready_candidate_before_production_rescue(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = _store(tmp)
            optimize_id = _candidate(
                store,
                "rank(probe_ready_signal)",
                "failed",
                metrics={"sharpe": 1.5, "fitness": 0.8, "turnover": 0.18},
                checks={
                    "LOW_SHARPE": {"status": "FAIL", "value": 1.5, "limit": 2.69},
                    "LOW_FITNESS": {"status": "FAIL", "value": 0.8, "limit": 1.5},
                },
            )
            store.record_event(optimize_id, "probe_validation", {"stage": "optimize_ready", "reason": "probe_has_signal"})
            store.record_event(None, "quality_stop_loss", {"scope": {"region": "USA"}, "quality_stop_reason": "bad_full_batch"})
            store.record_event(None, "quality_stop_loss", {"scope": {"region": "USA"}, "quality_stop_reason": "bad_full_batch"})

            plan = build_cycle_plan(
                store,
                {"region": "USA", "universe": "TOP3000", "delay": 1, "neutralization": "INDUSTRY"},
            )

        self.assertEqual(plan["mode"], "optimize")
        self.assertEqual(plan["target_candidate_id"], optimize_id)
        self.assertIn("probe_optimize_ready", plan["reason"])

    def test_scheduler_does_not_optimize_probe_ready_candidate_with_too_many_quality_gaps(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = _store(tmp)
            weak_id = _candidate(
                store,
                "rank(ts_decay_linear(ts_backfill(vec_avg(fnd23_significance), 120), 20))",
                "failed",
                metrics={"sharpe": 0.75, "fitness": 0.4, "turnover": 0.0693},
                checks={
                    "LOW_SHARPE": {"status": "FAIL", "value": 0.75, "limit": 2.69},
                    "LOW_FITNESS": {"status": "FAIL", "value": 0.4, "limit": 1.5},
                    "LOW_2Y_SHARPE": {"status": "FAIL", "value": 1.39, "limit": 2.69},
                    "HT_TURNOVER": {"status": "WARNING", "value": 0.0693, "limit": 0.2},
                    "HT_HIGH_TURNOVER_RETURNS_RATIO": {"status": "WARNING", "value": 0.5097, "limit": 0.75},
                    "LOW_TURNOVER": {"status": "PASS", "value": 0.0693, "limit": 0.01},
                    "HIGH_TURNOVER": {"status": "PASS", "value": 0.0693, "limit": 0.7},
                    "CONCENTRATED_WEIGHT": {"status": "PASS"},
                },
            )
            store.record_event(weak_id, "probe_validation", {"stage": "optimize_ready", "reason": "probe_has_signal"})
            store.record_event(None, "quality_stop_loss", {"scope": {"region": "USA"}, "quality_stop_reason": "bad_full_batch"})
            store.record_event(None, "quality_stop_loss", {"scope": {"region": "USA"}, "quality_stop_reason": "bad_full_batch"})

            plan = build_cycle_plan(
                store,
                {"region": "USA", "universe": "TOP3000", "delay": 1, "neutralization": "INDUSTRY"},
            )

        self.assertEqual(plan["mode"], "production_rescue")
        self.assertNotEqual(plan["target_candidate_id"], weak_id)

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
