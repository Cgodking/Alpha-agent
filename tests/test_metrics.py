from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from alpha.db import AlphaStore
from alpha.metrics import compute_efficiency_metrics


def _store(tmp: str) -> AlphaStore:
    store = AlphaStore(Path(tmp) / "alpha.db")
    store.init()
    return store


def _candidate(
    store: AlphaStore,
    expression: str,
    status: str,
    source: str = "model:g1",
    metrics=None,
    checks=None,
    settings=None,
):
    candidate_id = store.insert_candidate(
        expression,
        settings or {"region": "USA", "universe": "TOP3000", "delay": 1, "neutralization": "INDUSTRY"},
        source,
    )
    store.record_event(candidate_id, "generated", {"expression": expression})
    if status in {"preflight_passed", "simulated", "approved", "submitted", "check_pending", "failed"}:
        store.transition(candidate_id, "preflight_passed")
    if metrics is not None or checks is not None:
        store.update_candidate(
            candidate_id,
            metrics_json=json.dumps(metrics or {}, sort_keys=True),
            checks_json=json.dumps(checks or {}, sort_keys=True),
        )
        store.transition(candidate_id, "simulated", {"alpha_id": f"A{candidate_id}"})
    if status != "simulated":
        store.transition(candidate_id, status)
    return candidate_id


class EfficiencyMetricsTests(unittest.TestCase):
    def test_compute_efficiency_metrics_counts_rates_and_waste(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = _store(tmp)
            _candidate(
                store,
                "rank(alpha_signal)",
                "approved",
                metrics={"sharpe": 2.9, "fitness": 1.6, "turnover": 0.2},
                checks={"SELF_CORRELATION": {"status": "PASS"}},
            )
            failed_id = _candidate(
                store,
                "rank(bad_signal)",
                "failed",
                metrics={"sharpe": 0.1, "fitness": 0.02, "turnover": 0.2},
                checks={"LOW_SHARPE": {"status": "FAIL"}},
            )
            store.record_event(failed_id, "submission_guard", {"errors": ["SHARPE_BELOW_MIN:0.100<1.58"]})
            rejected_id = store.insert_candidate(
                "rank(invented_field)",
                {"region": "USA", "universe": "TOP3000", "delay": 1, "neutralization": "INDUSTRY"},
                "model:g2",
            )
            store.record_event(rejected_id, "generated", {"expression": "rank(invented_field)"})
            store.record_event(rejected_id, "preflight_failed", {"errors": ["UNKNOWN_FIELD:invented_field"]})
            store.transition(rejected_id, "failed", {"reason": "preflight"})
            store.record_event(None, "duplicate_candidate_skipped", {"source": "model:g1"})
            store.record_event(None, "structural_duplicate_candidate_skipped", {"source": "model:g1"})

            metrics = compute_efficiency_metrics(store)

        self.assertEqual(metrics["totals"]["generated"], 3)
        self.assertEqual(metrics["totals"]["preflight_passed"], 2)
        self.assertEqual(metrics["totals"]["simulated"], 2)
        self.assertEqual(metrics["totals"]["approved"], 1)
        self.assertEqual(metrics["totals"]["duplicate_skipped"], 2)
        self.assertAlmostEqual(metrics["rates"]["preflight_pass_rate"], 2 / 3)
        self.assertAlmostEqual(metrics["rates"]["approved_per_100_simulations"], 50.0)
        self.assertGreater(metrics["rates"]["simulation_waste_rate"], 0.0)

    def test_compute_efficiency_metrics_groups_by_scope_source_and_field_family(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = _store(tmp)
            _candidate(
                store,
                "rank(anl4_eps_est)",
                "approved",
                source="model:optimizer",
                metrics={"sharpe": 3.0, "fitness": 1.7},
            )
            _candidate(
                store,
                "rank(snt23_score)",
                "failed",
                source="model:generator",
                metrics={"sharpe": 0.2, "fitness": 0.1},
                settings={"region": "CHN", "universe": "TOP2000U", "delay": 0, "neutralization": "INDUSTRY"},
            )

            metrics = compute_efficiency_metrics(store)

        self.assertIn("model:optimizer", metrics["by_source"])
        self.assertEqual(metrics["by_source"]["model:optimizer"]["approved"], 1)
        self.assertIn("USA|TOP3000|D1|INDUSTRY", metrics["by_scope"])
        self.assertIn("CHN|TOP2000U|D0|INDUSTRY", metrics["by_scope"])
        self.assertIn("analyst", metrics["by_field_family"])
        self.assertIn("sentiment", metrics["by_field_family"])

    def test_platform_rate_limit_errors_do_not_count_as_quality_waste(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = _store(tmp)
            candidate_id = _candidate(store, "rank(rate_limited_signal)", "failed")
            store.record_event(candidate_id, "simulation_error", {"error": "HTTP 429 Retry-After: 5"})

            metrics = compute_efficiency_metrics(store)

        self.assertEqual(metrics["totals"]["platform_error_failures"], 1)
        self.assertEqual(metrics["totals"]["quality_waste_failures"], 0)

    def test_platform_settings_errors_do_not_count_as_quality_waste(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = _store(tmp)
            candidate_id = _candidate(store, "rank(settings_error_signal)", "failed")
            store.record_event(
                candidate_id,
                "simulation_error",
                {"error": 'multisimulation creation failed: HTTP 400 [{"settings":{"cyclePlan":["Unexpected property."]}}]'},
            )

            metrics = compute_efficiency_metrics(store)

        self.assertEqual(metrics["totals"]["platform_error_failures"], 1)
        self.assertEqual(metrics["totals"]["quality_waste_failures"], 0)

    def test_platform_poll_timeout_errors_do_not_count_as_quality_waste(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = _store(tmp)
            candidate_id = _candidate(store, "rank(poll_timeout_signal)", "failed")
            store.record_event(
                candidate_id,
                "simulation_error",
                {"error": "simulation polling did not return an alpha id: location=/simulations/slow"},
            )

            metrics = compute_efficiency_metrics(store)

        self.assertEqual(metrics["totals"]["platform_error_failures"], 1)
        self.assertEqual(metrics["totals"]["quality_waste_failures"], 0)


if __name__ == "__main__":
    unittest.main()
