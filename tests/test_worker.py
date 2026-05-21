from __future__ import annotations

import json
import tempfile
import unittest
from datetime import date
from pathlib import Path

from alpha.clients import LocalAIClient, LocalBrainClient
from alpha.db import AlphaStore
from alpha.guards import SubmissionPolicy
from alpha.models import CandidateSpec, DEFAULT_SETTINGS, SimulationFailure
from alpha.worker import AlphaWorker, _current_quarter_date_range


class WorkerTests(unittest.TestCase):
    def test_current_quarter_date_range_uses_full_quarter_end(self):
        self.assertEqual(_current_quarter_date_range(date(2026, 5, 9)), ("2026-04-01", "2026-06-30"))
        self.assertEqual(_current_quarter_date_range(date(2026, 12, 15)), ("2026-10-01", "2026-12-31"))

    def test_worker_passes_research_context_to_ai_client(self):
        class CapturingAI:
            def __init__(self):
                self.context = None

            def generate_candidates(self, batch_size, context):
                self.context = context
                return [CandidateSpec("group_rank(ts_rank(mdl_mock_score, 22), industry)")]

        with tempfile.TemporaryDirectory() as tmp:
            store = AlphaStore(Path(tmp) / "alpha.db")
            store.init()
            ai = CapturingAI()
            worker = AlphaWorker(
                store=store,
                ai_client=ai,
                brain_client=LocalBrainClient(),
                policy=SubmissionPolicy(auto_submit=False),
                batch_size=1,
                context={"region": "USA", "universe": "TOP3000", "delay": 0},
            )

            worker.run_once()

            self.assertEqual(ai.context["region"], "USA")
            self.assertEqual(ai.context["research_context"]["target_settings"]["delay"], 0)
            self.assertEqual(ai.context["research_context"]["generation_policy"]["complexity"], "research_grade")
            self.assertTrue(ai.context["research_context"]["datafields"]["available"])
            self.assertIn("close", ai.context["research_context"]["datafields"]["field_ids"])

    def test_worker_includes_recent_platform_submissions_in_research_context(self):
        class CapturingAI:
            def __init__(self):
                self.context = None

            def generate_candidates(self, batch_size, context):
                self.context = context
                return [CandidateSpec("group_rank(ts_rank(mdl_mock_score, 22), industry)")]

        class BrainWithSubmissions(LocalBrainClient):
            def recent_submitted_alphas(self, settings=None, limit=50):
                return [
                    {
                        "id": "P1",
                        "stage": "OS",
                        "regular": "group_rank(ts_rank(est_q_pre_mean, 63), industry)",
                        "settings": {"region": "USA", "universe": "TOP3000", "delay": 0},
                    }
                ]

        with tempfile.TemporaryDirectory() as tmp:
            store = AlphaStore(Path(tmp) / "alpha.db")
            store.init()
            ai = CapturingAI()
            worker = AlphaWorker(
                store=store,
                ai_client=ai,
                brain_client=BrainWithSubmissions(),
                policy=SubmissionPolicy(auto_submit=False),
                batch_size=1,
                context={"region": "USA", "universe": "TOP3000", "delay": 0},
            )

            worker.run_once()

            avoidance = ai.context["research_context"]["submitted_field_avoidance"]
            self.assertIn("est_q_pre_mean", avoidance["fields"])
            self.assertEqual(avoidance["examples"][0]["source"], "platform_os_alphas")

    def test_worker_includes_platform_pyramid_alphas_in_research_context(self):
        class CapturingAI:
            def __init__(self):
                self.context = None

            def generate_candidates(self, batch_size, context):
                self.context = context
                return [CandidateSpec("group_rank(ts_rank(mdl_mock_score, 22), industry)")]

        class BrainWithPyramids(LocalBrainClient):
            def __init__(self):
                super().__init__()
                self.pyramid_alpha_dates = None

            def get_pyramid_alphas(self, start_date=None, end_date=None):
                self.pyramid_alpha_dates = (start_date, end_date)
                return {
                    "pyramids": [
                        {
                            "category": {"id": "pv", "name": "Price Volume"},
                            "region": "USA",
                            "delay": 0,
                            "alphaCount": 28,
                        }
                    ]
                }

            def get_pyramid_multipliers(self):
                return {
                    "pyramids": [
                        {
                            "category": {"id": "pv", "name": "Price Volume"},
                            "region": "USA",
                            "delay": 0,
                            "multiplier": 1.6,
                        }
                    ]
                }

        with tempfile.TemporaryDirectory() as tmp:
            store = AlphaStore(Path(tmp) / "alpha.db")
            store.init()
            ai = CapturingAI()
            brain = BrainWithPyramids()
            worker = AlphaWorker(
                store=store,
                ai_client=ai,
                brain_client=brain,
                policy=SubmissionPolicy(auto_submit=False),
                batch_size=1,
                context={"region": "USA", "universe": "TOP3000", "delay": 0},
            )

            worker.run_once()

            avoidance = ai.context["research_context"]["lit_tower_avoidance"]
            self.assertEqual(avoidance["tower_names"], ["USA/D0/PV"])
            self.assertEqual(avoidance["lit_towers"][0]["multiplier"], 1.6)
            self.assertRegex(brain.pyramid_alpha_dates[0], r"^\d{4}-\d{2}-01$")
            self.assertRegex(brain.pyramid_alpha_dates[1], r"^\d{4}-(03|06|09|12)-(30|31)$")

    def test_worker_records_experiment_plan_for_each_cycle(self):
        class CapturingAI:
            def __init__(self):
                self.context = None

            def generate_candidates(self, batch_size, context):
                self.context = context
                return [CandidateSpec("group_rank(ts_rank(mdl_mock_score, 22), industry)")]

        with tempfile.TemporaryDirectory() as tmp:
            store = AlphaStore(Path(tmp) / "alpha.db")
            store.init()
            best_id = store.insert_candidate(
                "rank(group_rank(winsorize(ts_mean(analyst_positive_sentiment_logit_presentation,30),std=3),industry))",
                {"region": "USA", "delay": 0},
                "openai_compatible",
            )
            store.update_candidate(best_id, metrics_json='{"sharpe":1.11,"fitness":0.46}')
            store.transition(best_id, "check_pending", {"errors": ["SHARPE_BELOW_MIN:1.110<1.58"]})
            ai = CapturingAI()
            worker = AlphaWorker(
                store=store,
                ai_client=ai,
                brain_client=LocalBrainClient(),
                policy=SubmissionPolicy(auto_submit=False),
                batch_size=1,
                context={"region": "USA", "universe": "TOP3000", "delay": 0},
            )

            worker.run_once()

            plan = ai.context["research_context"]["experiment_plan"]
            events = store.events_for_candidate(None)
            self.assertEqual(plan["mode"], "optimize_best")
            self.assertEqual(plan["target_candidate_id"], best_id)
            self.assertTrue(any(event["event_type"] == "experiment_plan" for event in events))

    def test_worker_records_multi_model_generation_errors_after_partial_success(self):
        class PartialAI:
            def __init__(self):
                self.last_errors = [
                    {"profile": "gemini", "model": "gemini-3-flash-free", "role": "generator", "error": "read timed out"}
                ]
                self.last_plan = {"controller": "flow", "allocation": {"gemini": 1, "glm": 1}}

            def generate_candidates(self, batch_size, context):
                return [CandidateSpec("group_rank(ts_rank(mdl_mock_score, 22), industry)", source="model:glm")]

        with tempfile.TemporaryDirectory() as tmp:
            store = AlphaStore(Path(tmp) / "alpha.db")
            store.init()
            worker = AlphaWorker(
                store=store,
                ai_client=PartialAI(),
                brain_client=LocalBrainClient(),
                policy=SubmissionPolicy(auto_submit=False),
                batch_size=2,
                context={"region": "USA", "universe": "TOP3000", "delay": 0},
            )

            worker.run_once()

            global_events = store.events_for_candidate(None)
            self.assertTrue(any(event["event_type"] == "model_allocation" for event in global_events))
            self.assertTrue(
                any(
                    event["event_type"] == "model_generation_error"
                    and "gemini-3-flash-free" in event["metadata_json"]
                    for event in global_events
                )
            )

    def test_worker_records_intra_round_repair_diagnostics(self):
        class RepairAI:
            def __init__(self):
                self.last_errors = []
                self.last_plan = {
                    "controller": "flow",
                    "allocation": {"gemini": 1},
                    "intra_round_repair": {
                        "action": "refill",
                        "remaining_slots": 1,
                        "refill_allocation": {"glm": 1},
                    },
                }

            def generate_candidates(self, batch_size, context):
                return [CandidateSpec("group_rank(ts_rank(mdl_mock_score, 22), industry)", source="model:glm")]

        with tempfile.TemporaryDirectory() as tmp:
            store = AlphaStore(Path(tmp) / "alpha.db")
            store.init()
            worker = AlphaWorker(
                store=store,
                ai_client=RepairAI(),
                brain_client=LocalBrainClient(),
                policy=SubmissionPolicy(auto_submit=False),
                batch_size=1,
                context={"region": "USA", "universe": "TOP3000", "delay": 0},
            )

            worker.run_once()

            global_events = store.events_for_candidate(None)
            repair_events = [event for event in global_events if event["event_type"] == "intra_round_repair"]
            self.assertEqual(len(repair_events), 1)
            self.assertEqual(json.loads(repair_events[0]["metadata_json"])["refill_allocation"], {"glm": 1})

    def test_worker_records_ai_rationale_and_simulation_feedback_for_next_round(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = AlphaStore(Path(tmp) / "alpha.db")
            store.init()
            worker = AlphaWorker(
                store=store,
                ai_client=LocalAIClient(
                    expressions=["group_rank(ts_rank(mdl_mock_score, 22), industry)"],
                    metadata={"hypothesis": "industry-relative short reversal", "risk_notes": "correlation risk"},
                ),
                brain_client=LocalBrainClient(),
                policy=SubmissionPolicy(auto_submit=False),
                batch_size=1,
            )

            worker.run_once()

            candidate = store.list_candidates()[0]
            events = store.events_for_candidate(candidate["id"])
            generated = [event for event in events if event["event_type"] == "generated"][0]
            self.assertIn("industry-relative short reversal", generated["metadata_json"])

            context = worker._build_ai_context()
            success = context["research_context"]["recent_successes"][0]
            self.assertEqual(success["metrics"]["sharpe"], 2.0)
            self.assertEqual(success["checks"]["SELF_CORRELATION"]["status"], "PASS")
            self.assertEqual(success["generated_metadata"]["hypothesis"], "industry-relative short reversal")

    def test_worker_run_once_approves_clean_candidate_without_real_submit(self):
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

            summary = worker.run_once()

            candidates = store.list_candidates()
            self.assertEqual(summary["generated"], 1)
            self.assertEqual(candidates[0]["status"], "approved")
            self.assertTrue(
                any(event["event_type"] == "dry_run_submit" for event in store.events_for_candidate(candidates[0]["id"]))
            )

    def test_worker_keeps_pending_checks_out_of_approved_queue(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = AlphaStore(Path(tmp) / "alpha.db")
            store.init()
            brain = LocalBrainClient(force_pending_checks=True)
            worker = AlphaWorker(
                store=store,
                ai_client=LocalAIClient(expressions=["rank(mdl_mock_score)"]),
                brain_client=brain,
                policy=SubmissionPolicy(auto_submit=False),
                batch_size=1,
            )

            worker.run_once()

            candidates = store.list_candidates()
            self.assertEqual(candidates[0]["status"], "check_pending")

    def test_worker_fails_check_pending_when_core_checks_failed_even_with_terminal_pending(self):
        class EmptyAI:
            def generate_candidates(self, batch_size, context):
                return []

        with tempfile.TemporaryDirectory() as tmp:
            store = AlphaStore(Path(tmp) / "alpha.db")
            store.init()
            candidate_id = store.insert_candidate(
                "rank(group_rank(ts_rank(mdl_mock_score,22),industry))",
                {"region": "USA", "universe": "TOP3000", "delay": 0, "neutralization": "INDUSTRY"},
                "model:G-1",
            )
            store.update_candidate(
                candidate_id,
                alpha_id="badAlpha",
                metrics_json=json.dumps(
                    {
                        "sharpe": -3.15,
                        "fitness": -0.93,
                        "turnover": 0.9012,
                        "returns": -0.05,
                    }
                ),
                checks_json=json.dumps(
                    {
                        "LOW_SHARPE": {"status": "FAIL", "value": -3.15, "limit": 2.69},
                        "LOW_FITNESS": {"status": "FAIL", "value": -0.93, "limit": 1.5},
                        "HIGH_TURNOVER": {"status": "FAIL", "value": 0.9012, "limit": 0.7},
                        "LOW_SUB_UNIVERSE_SHARPE": {"status": "FAIL", "value": -1.63, "limit": -1.36},
                        "IS_LADDER_SHARPE": {"status": "FAIL", "value": -1.75, "limit": 2.69},
                        "CONCENTRATED_WEIGHT": {"status": "PASS"},
                        "SELF_CORRELATION": {"status": "PENDING"},
                        "PROD_CORRELATION": {"status": "PENDING"},
                        "DATA_DIVERSITY": {"status": "PENDING"},
                        "REGULAR_SUBMISSION": {"status": "PENDING"},
                    }
                ),
            )
            store.transition(candidate_id, "check_pending", {"errors": ["SELF_CORRELATION:PENDING"]})
            worker = AlphaWorker(
                store=store,
                ai_client=EmptyAI(),
                brain_client=LocalBrainClient(),
                policy=SubmissionPolicy(auto_submit=False, min_sharpe=2.69, min_fitness=1.5),
                batch_size=1,
                context={"region": "USA", "universe": "TOP3000", "delay": 0, "neutralization": "INDUSTRY"},
            )

            summary = worker.run_once()

            candidate = store.get_candidate(candidate_id)
            self.assertEqual(candidate["status"], "failed")
            self.assertEqual(summary["failed"], 1)
            events = store.events_for_candidate(candidate_id)
            self.assertTrue(any(event["event_type"] == "pending_recheck_ineligible" for event in events))

    def test_worker_rechecks_terminal_pending_candidate_and_approves_when_checks_pass(self):
        class EmptyAI:
            def generate_candidates(self, batch_size, context):
                return []

        class ResolvingBrain(LocalBrainClient):
            def __init__(self):
                super().__init__()
                self.check_calls = []

            def get_submission_check(self, alpha_id):
                self.check_calls.append(alpha_id)
                return self._checks()

        with tempfile.TemporaryDirectory() as tmp:
            store = AlphaStore(Path(tmp) / "alpha.db")
            store.init()
            candidate_id = store.insert_candidate(
                "rank(group_rank(ts_mean(mdl_mock_score,30),industry))",
                {"region": "USA", "universe": "TOP3000", "delay": 0, "neutralization": "INDUSTRY"},
                "model:G-2",
            )
            store.update_candidate(
                candidate_id,
                alpha_id="readyAlpha",
                metrics_json=json.dumps({"sharpe": 2.9, "fitness": 1.6, "turnover": 0.25, "returns": 0.08}),
                checks_json=json.dumps(
                    {
                        "LOW_SHARPE": {"status": "PASS", "value": 2.9, "limit": 2.69},
                        "LOW_FITNESS": {"status": "PASS", "value": 1.6, "limit": 1.5},
                        "LOW_TURNOVER": {"status": "PASS", "value": 0.25, "limit": 0.01},
                        "HIGH_TURNOVER": {"status": "PASS", "value": 0.25, "limit": 0.7},
                        "CONCENTRATED_WEIGHT": {"status": "PASS"},
                        "LOW_SUB_UNIVERSE_SHARPE": {"status": "PASS", "value": 1.8, "limit": 1.0},
                        "IS_LADDER_SHARPE": {"status": "PASS", "value": 2.8, "limit": 2.69},
                        "SELF_CORRELATION": {"status": "PENDING"},
                        "PROD_CORRELATION": {"status": "PENDING"},
                        "DATA_DIVERSITY": {"status": "PENDING"},
                        "REGULAR_SUBMISSION": {"status": "PENDING"},
                    }
                ),
            )
            store.transition(candidate_id, "check_pending", {"errors": ["SELF_CORRELATION:PENDING"]})
            brain = ResolvingBrain()
            worker = AlphaWorker(
                store=store,
                ai_client=EmptyAI(),
                brain_client=brain,
                policy=SubmissionPolicy(auto_submit=False, min_sharpe=2.69, min_fitness=1.5),
                batch_size=1,
                context={"region": "USA", "universe": "TOP3000", "delay": 0, "neutralization": "INDUSTRY"},
            )

            summary = worker.run_once()

            candidate = store.get_candidate(candidate_id)
            self.assertEqual(brain.check_calls, ["readyAlpha"])
            self.assertEqual(candidate["status"], "approved")
            self.assertEqual(summary["approved"], 1)
            self.assertTrue(
                any(event["event_type"] == "pending_recheck" for event in store.events_for_candidate(candidate_id))
            )

    def test_worker_keeps_submission_quota_fail_pending_after_recheck(self):
        class EmptyAI:
            def generate_candidates(self, batch_size, context):
                return []

        class QuotaFullBrain(LocalBrainClient):
            def get_submission_check(self, alpha_id):
                checks = self._checks()
                checks["SELF_CORRELATION"] = {"status": "PENDING"}
                checks["PROD_CORRELATION"] = {"status": "PENDING"}
                checks["REGULAR_SUBMISSION"] = {"status": "FAIL", "value": 4, "limit": 4}
                return checks

        with tempfile.TemporaryDirectory() as tmp:
            store = AlphaStore(Path(tmp) / "alpha.db")
            store.init()
            candidate_id = store.insert_candidate(
                "rank(group_rank(ts_mean(mdl_mock_score,30),industry))",
                {"region": "USA", "universe": "TOP3000", "delay": 0, "neutralization": "INDUSTRY"},
                "model:G-2",
            )
            store.update_candidate(
                candidate_id,
                alpha_id="quotaFullAlpha",
                metrics_json=json.dumps({"sharpe": 2.9, "fitness": 1.6, "turnover": 0.25, "returns": 0.08}),
                checks_json=json.dumps(
                    {
                        "LOW_SHARPE": {"status": "PASS", "value": 2.9, "limit": 2.69},
                        "LOW_FITNESS": {"status": "PASS", "value": 1.6, "limit": 1.5},
                        "LOW_TURNOVER": {"status": "PASS", "value": 0.25, "limit": 0.01},
                        "HIGH_TURNOVER": {"status": "PASS", "value": 0.25, "limit": 0.7},
                        "CONCENTRATED_WEIGHT": {"status": "PASS"},
                        "LOW_SUB_UNIVERSE_SHARPE": {"status": "PASS", "value": 1.8, "limit": 1.0},
                        "IS_LADDER_SHARPE": {"status": "PASS", "value": 2.8, "limit": 2.69},
                        "SELF_CORRELATION": {"status": "PENDING"},
                        "PROD_CORRELATION": {"status": "PENDING"},
                        "DATA_DIVERSITY": {"status": "PENDING"},
                        "REGULAR_SUBMISSION": {"status": "PENDING"},
                    }
                ),
            )
            store.transition(candidate_id, "check_pending", {"errors": ["SELF_CORRELATION:PENDING"]})
            worker = AlphaWorker(
                store=store,
                ai_client=EmptyAI(),
                brain_client=QuotaFullBrain(),
                policy=SubmissionPolicy(auto_submit=False, min_sharpe=2.69, min_fitness=1.5),
                batch_size=1,
                context={"region": "USA", "universe": "TOP3000", "delay": 0, "neutralization": "INDUSTRY"},
            )

            summary = worker.run_once()

            candidate = store.get_candidate(candidate_id)
            self.assertEqual(candidate["status"], "check_pending")
            self.assertGreaterEqual(summary["pending"], 1)
            events = store.events_for_candidate(candidate_id)
            self.assertTrue(any("REGULAR_SUBMISSION:QUOTA_FULL" in event["metadata_json"] for event in events))

    def test_worker_marks_candidate_failed_after_retry_limit(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = AlphaStore(Path(tmp) / "alpha.db")
            store.init()
            brain = LocalBrainClient(always_fail_simulation=True)
            worker = AlphaWorker(
                store=store,
                ai_client=LocalAIClient(expressions=["rank(mdl_mock_score)"]),
                brain_client=brain,
                policy=SubmissionPolicy(auto_submit=False, max_retries=1),
                batch_size=1,
            )

            worker.run_once()

            candidates = store.list_candidates()
            self.assertEqual(candidates[0]["status"], "failed")
            self.assertEqual(candidates[0]["retry_count"], 1)

    def test_worker_rejects_ai_invented_field_before_simulation(self):
        class CountingBrain(LocalBrainClient):
            def __init__(self):
                super().__init__()
                self.simulation_calls = 0

            def simulate(self, expression, settings):
                self.simulation_calls += 1
                return super().simulate(expression, settings)

        with tempfile.TemporaryDirectory() as tmp:
            store = AlphaStore(Path(tmp) / "alpha.db")
            store.init()
            brain = CountingBrain()
            worker = AlphaWorker(
                store=store,
                ai_client=LocalAIClient(expressions=["rank(ai_invented_field)"]),
                brain_client=brain,
                policy=SubmissionPolicy(auto_submit=False),
                batch_size=1,
            )

            summary = worker.run_once()

            candidate = store.list_candidates()[0]
            events = store.events_for_candidate(candidate["id"])
            self.assertEqual(summary["failed"], 1)
            self.assertEqual(brain.simulation_calls, 0)
            self.assertEqual(candidate["status"], "failed")
            self.assertTrue(any("UNKNOWN_FIELD:ai_invented_field" in event["metadata_json"] for event in events))

    def test_worker_rejects_ai_invented_field_when_datafield_discovery_fails(self):
        class DiscoveryFailingBrain(LocalBrainClient):
            def __init__(self):
                super().__init__()
                self.simulation_calls = 0

            def discover_datafields(self, settings, search_terms=None, max_fields=120):
                raise RuntimeError("datafield discovery unavailable")

            def simulate(self, expression, settings):
                self.simulation_calls += 1
                return super().simulate(expression, settings)

        with tempfile.TemporaryDirectory() as tmp:
            store = AlphaStore(Path(tmp) / "alpha.db")
            store.init()
            brain = DiscoveryFailingBrain()
            worker = AlphaWorker(
                store=store,
                ai_client=LocalAIClient(expressions=["rank(ai_invented_field)"]),
                brain_client=brain,
                policy=SubmissionPolicy(auto_submit=False),
                batch_size=1,
            )

            summary = worker.run_once()

            candidate = store.list_candidates()[0]
            events = store.events_for_candidate(candidate["id"])
            self.assertEqual(summary["failed"], 1)
            self.assertEqual(brain.simulation_calls, 0)
            self.assertEqual(candidate["status"], "failed")
            self.assertTrue(any("UNKNOWN_FIELD:ai_invented_field" in event["metadata_json"] for event in events))

    def test_worker_rejects_vector_field_ts_backfill_before_simulation(self):
        class VectorFieldBrain(LocalBrainClient):
            def __init__(self):
                super().__init__()
                self.simulation_calls = 0

            def discover_datafields(self, settings, search_terms=None, max_fields=120):
                return [
                    {
                        "id": "analyst_sentence_count_presentation",
                        "type": "VECTOR",
                        "dataset": {"id": "analyst83", "name": "Smart Conference call transcript data"},
                        "category": {"id": "analyst", "name": "Analyst"},
                    }
                ]

            def simulate(self, expression, settings):
                self.simulation_calls += 1
                return super().simulate(expression, settings)

        with tempfile.TemporaryDirectory() as tmp:
            store = AlphaStore(Path(tmp) / "alpha.db")
            store.init()
            brain = VectorFieldBrain()
            worker = AlphaWorker(
                store=store,
                ai_client=LocalAIClient(
                    expressions=["rank(ts_backfill(analyst_sentence_count_presentation, 120))"]
                ),
                brain_client=brain,
                policy=SubmissionPolicy(auto_submit=False),
                batch_size=1,
            )

            summary = worker.run_once()

            candidate = store.list_candidates()[0]
            events = store.events_for_candidate(candidate["id"])
            self.assertEqual(summary["failed"], 1)
            self.assertEqual(brain.simulation_calls, 0)
            self.assertEqual(candidate["status"], "failed")
            self.assertTrue(any("INVALID_VECTOR_TS_OPERATOR" in event["metadata_json"] for event in events))

    def test_worker_skips_candidate_when_expression_structure_was_already_tested(self):
        class CountingBrain(LocalBrainClient):
            def __init__(self):
                super().__init__()
                self.simulation_calls = 0

            def simulate_many(self, items):
                self.simulation_calls += len(items)
                return super().simulate_many(items)

            def simulate(self, expression, settings):
                self.simulation_calls += 1
                return super().simulate(expression, settings)

        class StructureDuplicateAI:
            def generate_candidates(self, batch_size, context):
                settings = {"region": "USA", "universe": "TOP3000", "delay": 0}
                return [
                    CandidateSpec(
                        (
                            "group_rank(ts_rank(divide(winsorize(ts_backfill("
                            "mdl_other_score, 66), std=4), cap), 44), industry)"
                        ),
                        settings=settings,
                        source="model:G-1",
                    ),
                    CandidateSpec(
                        "group_rank(ts_delta(mdl_mock_score, 22), industry)",
                        settings=settings,
                        source="model:G-1",
                    ),
                ]

        with tempfile.TemporaryDirectory() as tmp:
            store = AlphaStore(Path(tmp) / "alpha.db")
            store.init()
            store.insert_candidate(
                (
                    "group_rank(ts_rank(divide(winsorize(ts_backfill("
                    "mdl_other_score, 120), std=3), cap), 63), industry)"
                ),
                {"region": "USA", "universe": "TOP3000", "delay": 0},
                "model:G-2",
            )
            store.transition(1, "approved")
            brain = CountingBrain()
            worker = AlphaWorker(
                store=store,
                ai_client=StructureDuplicateAI(),
                brain_client=brain,
                policy=SubmissionPolicy(auto_submit=False),
                batch_size=2,
                context={"region": "USA", "universe": "TOP3000", "delay": 0},
            )

            summary = worker.run_once()

            self.assertEqual(summary["skipped"], 1)
            self.assertEqual(summary["generated"], 1)
            self.assertEqual(brain.simulation_calls, 1)
            self.assertTrue(
                any(
                    event["event_type"] == "structural_duplicate_candidate_skipped"
                    for event in store.events_for_candidate(None)
                )
            )

    def test_worker_ignores_preflight_only_history_for_structural_dedup(self):
        class CountingBrain(LocalBrainClient):
            def __init__(self):
                super().__init__()
                self.simulation_calls = 0

            def simulate_many(self, items):
                self.simulation_calls += len(items)
                return super().simulate_many(items)

            def simulate(self, expression, settings):
                self.simulation_calls += 1
                return super().simulate(expression, settings)

        class PreflightHistoryAI:
            def generate_candidates(self, batch_size, context):
                settings = {"region": "USA", "universe": "TOP3000", "delay": 0}
                return [
                    CandidateSpec(
                        (
                            "group_rank(ts_rank(divide(winsorize(ts_backfill("
                            "mdl_mock_score, 66), std=4), cap), 44), industry)"
                        ),
                        settings=settings,
                        source="model:G-1",
                    )
                ]

        with tempfile.TemporaryDirectory() as tmp:
            store = AlphaStore(Path(tmp) / "alpha.db")
            store.init()
            candidate_id = store.insert_candidate(
                (
                    "group_rank(ts_rank(divide(winsorize(ts_backfill("
                    "mdl_other_score, 120), std=3), cap), 63), industry)"
                ),
                {"region": "USA", "universe": "TOP3000", "delay": 0},
                "model:G-2",
            )
            store.update_candidate(candidate_id, status="preflight_passed")
            brain = CountingBrain()
            worker = AlphaWorker(
                store=store,
                ai_client=PreflightHistoryAI(),
                brain_client=brain,
                policy=SubmissionPolicy(auto_submit=False),
                batch_size=1,
                context={"region": "USA", "universe": "TOP3000", "delay": 0},
            )

            summary = worker.run_once()

            self.assertEqual(summary["skipped"], 0)
            self.assertEqual(summary["generated"], 1)
            self.assertEqual(brain.simulation_calls, 1)

    def test_worker_allows_optimize_best_to_repair_nearby_structural_variants(self):
        class CountingBrain(LocalBrainClient):
            def __init__(self):
                super().__init__()
                self.simulation_calls = 0

            def simulate_many(self, items):
                self.simulation_calls += len(items)
                return super().simulate_many(items)

            def simulate(self, expression, settings):
                self.simulation_calls += 1
                return super().simulate(expression, settings)

        class OptimizeBestWorker(AlphaWorker):
            def _build_ai_context(self):
                return {
                    "region": "USA",
                    "universe": "TOP3000",
                    "delay": 0,
                    "research_context": {
                        "generation_policy": {"avoid_historical_structural_duplicates": True},
                        "datafields": {
                            "available": True,
                            "field_ids": ["mdl_other_score", "mdl_mock_score"],
                            "field_types": {},
                        },
                        "experiment_plan": {"mode": "optimize_best", "optimize_round": 2},
                    },
                }

        class OptimizeBestAI:
            def generate_candidates(self, batch_size, context):
                settings = {"region": "USA", "universe": "TOP3000", "delay": 0}
                return [
                    CandidateSpec(
                        (
                            "group_rank(ts_rank(divide(winsorize(ts_backfill("
                            "mdl_mock_score, 66), std=4), cap), 44), industry)"
                        ),
                        settings=settings,
                        source="model:G-1",
                    )
                ]

        with tempfile.TemporaryDirectory() as tmp:
            store = AlphaStore(Path(tmp) / "alpha.db")
            store.init()
            candidate_id = store.insert_candidate(
                (
                    "group_rank(ts_rank(divide(winsorize(ts_backfill("
                    "mdl_other_score, 120), std=3), cap), 63), industry)"
                ),
                {"region": "USA", "universe": "TOP3000", "delay": 0},
                "model:G-2",
            )
            store.transition(candidate_id, "approved")
            brain = CountingBrain()
            worker = OptimizeBestWorker(
                store=store,
                ai_client=OptimizeBestAI(),
                brain_client=brain,
                policy=SubmissionPolicy(auto_submit=False),
                batch_size=1,
                context={"region": "USA", "universe": "TOP3000", "delay": 0},
            )

            summary = worker.run_once()

            self.assertEqual(summary["skipped"], 0)
            self.assertEqual(summary["generated"], 1)
            self.assertEqual(brain.simulation_calls, 1)

    def test_worker_allows_same_template_for_different_field_families(self):
        class CountingBrain(LocalBrainClient):
            def __init__(self):
                super().__init__()
                self.simulation_calls = 0

            def simulate_many(self, items):
                self.simulation_calls += len(items)
                return super().simulate_many(items)

            def simulate(self, expression, settings):
                self.simulation_calls += 1
                return super().simulate(expression, settings)

        class DifferentFieldAI:
            def generate_candidates(self, batch_size, context):
                settings = {"region": "USA", "universe": "TOP3000", "delay": 0}
                return [
                    CandidateSpec(
                        "group_rank(ts_rank(ts_backfill(mdl_mock_score, 66), 33), industry)",
                        settings=settings,
                        source="model:G-1",
                    )
                ]

        with tempfile.TemporaryDirectory() as tmp:
            store = AlphaStore(Path(tmp) / "alpha.db")
            store.init()
            store.insert_candidate(
                "group_rank(ts_rank(ts_backfill(close, 120), 63), industry)",
                {"region": "USA", "universe": "TOP3000", "delay": 0},
                "model:G-2",
            )
            brain = CountingBrain()
            worker = AlphaWorker(
                store=store,
                ai_client=DifferentFieldAI(),
                brain_client=brain,
                policy=SubmissionPolicy(auto_submit=False),
                batch_size=1,
                context={"region": "USA", "universe": "TOP3000", "delay": 0},
            )

            summary = worker.run_once()

            self.assertEqual(summary["skipped"], 0)
            self.assertEqual(summary["generated"], 1)
            self.assertEqual(brain.simulation_calls, 1)

    def test_worker_retries_simulation_before_success(self):
        class FlakyBrain(LocalBrainClient):
            def __init__(self):
                super().__init__()
                self.calls = 0

            def simulate(self, expression, settings):
                self.calls += 1
                if self.calls == 1:
                    raise RuntimeError("temporary")
                return super().simulate(expression, settings)

        with tempfile.TemporaryDirectory() as tmp:
            store = AlphaStore(Path(tmp) / "alpha.db")
            store.init()
            brain = FlakyBrain()
            worker = AlphaWorker(
                store=store,
                ai_client=LocalAIClient(expressions=["rank(mdl_mock_score)"]),
                brain_client=brain,
                policy=SubmissionPolicy(auto_submit=False, max_retries=2),
                batch_size=1,
            )

            summary = worker.run_once()

            candidate = store.list_candidates()[0]
            self.assertEqual(brain.calls, 2)
            self.assertEqual(summary["approved"], 1)
            self.assertEqual(candidate["status"], "approved")
            self.assertEqual(candidate["retry_count"], 1)

    def test_worker_uses_batch_simulation_for_multiple_preflighted_candidates(self):
        class BatchBrain(LocalBrainClient):
            def __init__(self):
                super().__init__()
                self.batch_calls = []

            def simulate_many(self, items):
                self.batch_calls.append(items)
                return super().simulate_many(items)

        with tempfile.TemporaryDirectory() as tmp:
            store = AlphaStore(Path(tmp) / "alpha.db")
            store.init()
            brain = BatchBrain()
            worker = AlphaWorker(
                store=store,
                ai_client=LocalAIClient(
                    expressions=[
                        "rank(mdl_mock_score)",
                        "group_rank(ts_rank(divide(mdl_mock_score, cap), 63), industry)",
                    ]
                ),
                brain_client=brain,
                policy=SubmissionPolicy(auto_submit=False),
                batch_size=2,
            )

            summary = worker.run_once()

            self.assertEqual(summary["generated"], 2)
            self.assertEqual(len(brain.batch_calls), 1)
            self.assertEqual(len(brain.batch_calls[0]), 2)
            self.assertEqual([row["status"] for row in store.list_candidates()], ["approved", "approved"])

    def test_worker_runs_setting_sweep_without_calling_ai_generation(self):
        class FailingAI:
            def generate_candidates(self, batch_size, context):
                raise AssertionError("AI should not be called for setting_sweep")

        class BatchBrain(LocalBrainClient):
            def __init__(self):
                super().__init__()
                self.batch_calls = []

            def simulate_many(self, items):
                self.batch_calls.append(items)
                return super().simulate_many(items)

        with tempfile.TemporaryDirectory() as tmp:
            store = AlphaStore(Path(tmp) / "alpha.db")
            store.init()
            candidate_id = store.insert_candidate(
                "rank(group_rank(ts_mean(mdl_mock_score,30),industry))",
                {"region": "USA", "delay": 0, "neutralization": "INDUSTRY"},
                "openai_compatible",
            )
            store.update_candidate(
                candidate_id,
                metrics_json='{"sharpe":2.55,"fitness":1.32,"turnover":0.22,"drawdown":0.04}',
                checks_json=json.dumps(
                    {
                        "LOW_SHARPE": {"status": "FAIL", "value": 2.55, "limit": 2.69},
                        "LOW_FITNESS": {"status": "FAIL", "value": 1.32, "limit": 1.5},
                        "LOW_TURNOVER": {"status": "PASS", "value": 0.22, "limit": 0.01},
                        "HIGH_TURNOVER": {"status": "PASS", "value": 0.22, "limit": 0.7},
                        "CONCENTRATED_WEIGHT": {"status": "PASS"},
                        "LOW_SUB_UNIVERSE_SHARPE": {"status": "PASS", "value": 1.2, "limit": 1.0},
                        "IS_LADDER_SHARPE": {"status": "PASS", "value": 2.7, "limit": 2.69},
                        "SELF_CORRELATION": {"status": "PASS", "value": 0.2, "limit": 0.7},
                        "PROD_CORRELATION": {"status": "PASS", "value": 0.1, "limit": 0.7},
                        "DATA_DIVERSITY": {"status": "PASS"},
                        "REGULAR_SUBMISSION": {"status": "PASS"},
                    }
                ),
            )
            store.transition(candidate_id, "check_pending", {"errors": ["SHARPE_BELOW_MIN:2.550<2.69"]})
            brain = BatchBrain()
            worker = AlphaWorker(
                store=store,
                ai_client=FailingAI(),
                brain_client=brain,
                policy=SubmissionPolicy(auto_submit=False),
                batch_size=8,
                context={"region": "USA", "universe": "TOP3000", "delay": 0, "neutralization": "INDUSTRY"},
            )

            summary = worker.run_once()

            rows = store.list_candidates()
            generated = [row for row in rows if row["source"] == "planner_setting_sweep"]
            settings = [row["settings_json"] for row in generated]
            self.assertEqual(summary["generated"], 8)
            self.assertEqual(len(brain.batch_calls), 1)
            self.assertEqual(len(generated), 8)
            self.assertGreater(len(set(settings)), 1)
            self.assertTrue(any('"neutralization":"SUBINDUSTRY"' in value for value in settings))
            simulated_settings = brain.batch_calls[0][0][1]
            self.assertEqual(simulated_settings["instrumentType"], "EQUITY")
            self.assertEqual(simulated_settings["pasteurization"], "ON")
            self.assertEqual(simulated_settings["unitHandling"], "VERIFY")
            self.assertEqual(simulated_settings["language"], "FASTEXPR")

    def test_worker_validates_planned_setting_sweep_candidates(self):
        class PlannerValidatorAI:
            def __init__(self):
                self.calls = []
                self.last_validator_rejections = []

            def generate_candidates(self, batch_size, context):
                raise AssertionError("AI generation should not be called for setting_sweep")

            def validate_candidate_specs(self, candidates, batch_size, context):
                self.calls.append((candidates, batch_size, context))
                return [
                    CandidateSpec(
                        expression=candidate.expression,
                        settings=candidate.settings,
                        source=candidate.source,
                        metadata={**candidate.metadata, "validated_by": "dp-pro-check"},
                    )
                    for candidate in candidates
                ]

        with tempfile.TemporaryDirectory() as tmp:
            store = AlphaStore(Path(tmp) / "alpha.db")
            store.init()
            candidate_id = store.insert_candidate(
                "rank(group_rank(ts_mean(mdl_mock_score,30),industry))",
                {"region": "USA", "delay": 0, "neutralization": "INDUSTRY"},
                "openai_compatible",
            )
            store.update_candidate(
                candidate_id,
                metrics_json='{"sharpe":2.55,"fitness":1.32,"turnover":0.22,"drawdown":0.04}',
                checks_json=json.dumps(
                    {
                        "LOW_SHARPE": {"status": "FAIL", "value": 2.55, "limit": 2.69},
                        "LOW_FITNESS": {"status": "FAIL", "value": 1.32, "limit": 1.5},
                        "LOW_TURNOVER": {"status": "PASS", "value": 0.22, "limit": 0.01},
                        "HIGH_TURNOVER": {"status": "PASS", "value": 0.22, "limit": 0.7},
                        "CONCENTRATED_WEIGHT": {"status": "PASS"},
                        "LOW_SUB_UNIVERSE_SHARPE": {"status": "PASS", "value": 1.2, "limit": 1.0},
                        "IS_LADDER_SHARPE": {"status": "PASS", "value": 2.7, "limit": 2.69},
                        "SELF_CORRELATION": {"status": "PASS", "value": 0.2, "limit": 0.7},
                        "PROD_CORRELATION": {"status": "PASS", "value": 0.1, "limit": 0.7},
                        "DATA_DIVERSITY": {"status": "PASS"},
                        "REGULAR_SUBMISSION": {"status": "PASS"},
                    }
                ),
            )
            store.transition(candidate_id, "check_pending", {"errors": ["SHARPE_BELOW_MIN:2.550<2.69"]})
            ai = PlannerValidatorAI()
            worker = AlphaWorker(
                store=store,
                ai_client=ai,
                brain_client=LocalBrainClient(),
                policy=SubmissionPolicy(auto_submit=False),
                batch_size=8,
                context={"region": "USA", "universe": "TOP3000", "delay": 0, "neutralization": "INDUSTRY"},
            )

            summary = worker.run_once()

            self.assertEqual(len(ai.calls), 1)
            self.assertEqual(ai.calls[0][1], 8)
            generated = [row for row in store.list_candidates() if row["source"] == "planner_setting_sweep"]
            self.assertEqual(summary["generated"], 8)
            self.assertEqual(len(generated), 8)
            first_events = store.events_for_candidate(int(generated[0]["id"]))
            generated_event = next(event for event in first_events if event["event_type"] == "generated")
            metadata = json.loads(generated_event["metadata_json"])
            self.assertEqual(metadata["ai_metadata"]["validated_by"], "dp-pro-check")

    def test_worker_falls_back_to_ai_when_setting_sweep_target_has_unknown_fields(self):
        class CountingAI(LocalAIClient):
            def __init__(self):
                super().__init__()
                self.calls = 0

            def generate_candidates(self, batch_size, context):
                self.calls += 1
                return [CandidateSpec("group_rank(ts_rank(mdl_mock_score,22),industry)")]

        with tempfile.TemporaryDirectory() as tmp:
            store = AlphaStore(Path(tmp) / "alpha.db")
            store.init()
            candidate_id = store.insert_candidate(
                "rank(group_rank(ts_mean(unknown_alpha_field,30),industry))",
                {"region": "USA", "delay": 0, "neutralization": "INDUSTRY"},
                "openai_compatible",
            )
            store.update_candidate(
                candidate_id,
                metrics_json='{"sharpe":2.55,"fitness":1.32,"turnover":0.22,"drawdown":0.04}',
                checks_json=json.dumps(
                    {
                        "LOW_SHARPE": {"status": "FAIL", "value": 2.55, "limit": 2.69},
                        "LOW_FITNESS": {"status": "FAIL", "value": 1.32, "limit": 1.5},
                        "LOW_TURNOVER": {"status": "PASS", "value": 0.22, "limit": 0.01},
                        "HIGH_TURNOVER": {"status": "PASS", "value": 0.22, "limit": 0.7},
                        "CONCENTRATED_WEIGHT": {"status": "PASS"},
                        "LOW_SUB_UNIVERSE_SHARPE": {"status": "PASS", "value": 1.2, "limit": 1.0},
                        "IS_LADDER_SHARPE": {"status": "PASS", "value": 2.7, "limit": 2.69},
                        "SELF_CORRELATION": {"status": "PASS", "value": 0.2, "limit": 0.7},
                        "PROD_CORRELATION": {"status": "PASS", "value": 0.1, "limit": 0.7},
                        "DATA_DIVERSITY": {"status": "PASS"},
                        "REGULAR_SUBMISSION": {"status": "PASS"},
                    }
                ),
            )
            store.transition(candidate_id, "check_pending", {"errors": ["SHARPE_BELOW_MIN:2.550<2.69"]})
            ai = CountingAI()
            worker = AlphaWorker(
                store=store,
                ai_client=ai,
                brain_client=LocalBrainClient(),
                policy=SubmissionPolicy(auto_submit=False),
                batch_size=1,
                context={"region": "USA", "universe": "TOP3000", "delay": 0, "neutralization": "INDUSTRY"},
            )

            summary = worker.run_once()

            self.assertEqual(ai.calls, 1)
            self.assertEqual(summary["generated"], 1)
            self.assertTrue(
                any(event["event_type"] == "setting_sweep_target_invalid" for event in store.events_for_candidate(None))
            )

    def test_worker_falls_back_to_ai_when_setting_sweep_variants_are_exhausted(self):
        class CountingAI(LocalAIClient):
            def __init__(self):
                super().__init__(expressions=["rank(ts_mean(volume, 22))"])
                self.calls = 0
                self.plan_modes = []

            def generate_candidates(self, batch_size, context):
                self.calls += 1
                research_context = context.get("research_context") if isinstance(context, dict) else {}
                plan = research_context.get("experiment_plan") if isinstance(research_context, dict) else {}
                self.plan_modes.append(plan.get("mode") if isinstance(plan, dict) else None)
                return super().generate_candidates(1, context)

        with tempfile.TemporaryDirectory() as tmp:
            store = AlphaStore(Path(tmp) / "alpha.db")
            store.init()
            expression = "rank(group_rank(ts_mean(mdl_mock_score,30),industry))"
            candidate_id = store.insert_candidate(
                expression,
                {"region": "USA", "delay": 0, "neutralization": "INDUSTRY"},
                "openai_compatible",
            )
            store.update_candidate(
                candidate_id,
                metrics_json='{"sharpe":2.55,"fitness":1.32,"turnover":0.22,"drawdown":0.04}',
                checks_json=json.dumps(
                    {
                        "LOW_SHARPE": {"status": "FAIL", "value": 2.55, "limit": 2.69},
                        "LOW_FITNESS": {"status": "FAIL", "value": 1.32, "limit": 1.5},
                        "LOW_TURNOVER": {"status": "PASS", "value": 0.22, "limit": 0.01},
                        "HIGH_TURNOVER": {"status": "PASS", "value": 0.22, "limit": 0.7},
                        "CONCENTRATED_WEIGHT": {"status": "PASS"},
                        "LOW_SUB_UNIVERSE_SHARPE": {"status": "PASS", "value": 1.2, "limit": 1.0},
                        "IS_LADDER_SHARPE": {"status": "PASS", "value": 2.7, "limit": 2.69},
                        "SELF_CORRELATION": {"status": "PASS", "value": 0.2, "limit": 0.7},
                        "PROD_CORRELATION": {"status": "PASS", "value": 0.1, "limit": 0.7},
                        "DATA_DIVERSITY": {"status": "PASS"},
                        "REGULAR_SUBMISSION": {"status": "PASS"},
                    }
                ),
            )
            store.transition(candidate_id, "check_pending", {"errors": ["SHARPE_BELOW_MIN:2.550<2.69"]})
            ai = CountingAI()
            worker = AlphaWorker(
                store=store,
                ai_client=ai,
                brain_client=LocalBrainClient(),
                policy=SubmissionPolicy(auto_submit=False),
                batch_size=8,
                context={"region": "USA", "universe": "TOP3000", "delay": 0, "neutralization": "INDUSTRY"},
            )
            planned = worker._planned_candidates(worker._build_ai_context())
            self.assertIsNotNone(planned)
            for spec in planned:
                store.insert_candidate(spec.expression, spec.settings, spec.source)

            summary = worker.run_once()

            self.assertEqual(ai.calls, 1)
            self.assertEqual(ai.plan_modes, ["explore_new_family"])
            self.assertEqual(summary["generated"], 1)
            self.assertEqual(summary["skipped"], 0)
            sources = [row["source"] for row in store.list_candidates()]
            self.assertIn("local_ai", sources)

    def test_worker_keeps_successful_batch_children_when_one_child_fails(self):
        class PartialBatchBrain(LocalBrainClient):
            def simulate_many(self, items):
                return [
                    self.simulate(items[0][0], items[0][1]),
                    SimulationFailure("Operator ts_backfill does not support event inputs"),
                ]

        with tempfile.TemporaryDirectory() as tmp:
            store = AlphaStore(Path(tmp) / "alpha.db")
            store.init()
            worker = AlphaWorker(
                store=store,
                ai_client=LocalAIClient(
                    expressions=[
                        "rank(mdl_mock_score)",
                        "group_rank(ts_rank(divide(mdl_mock_score, cap), 63), industry)",
                    ]
                ),
                brain_client=PartialBatchBrain(),
                policy=SubmissionPolicy(auto_submit=False),
                batch_size=2,
            )

            summary = worker.run_once()

            candidates = store.list_candidates()
            self.assertEqual(summary["approved"], 1)
            self.assertEqual(summary["failed"], 1)
            self.assertEqual([row["status"] for row in candidates], ["approved", "failed"])
            failed_events = store.events_for_candidate(candidates[1]["id"])
            self.assertTrue(any(event["event_type"] == "simulation_error" for event in failed_events))

    def test_worker_skips_exact_duplicate_without_resimulation(self):
        class CountingBrain(LocalBrainClient):
            def __init__(self):
                super().__init__()
                self.batch_calls = []

            def simulate_many(self, items):
                self.batch_calls.append(items)
                return super().simulate_many(items)

        with tempfile.TemporaryDirectory() as tmp:
            store = AlphaStore(Path(tmp) / "alpha.db")
            store.init()
            settings = dict(DEFAULT_SETTINGS)
            settings.update({"region": "USA", "universe": "TOP3000", "delay": 1, "neutralization": "INDUSTRY"})
            existing_id = store.insert_candidate("rank(mdl_mock_score)", settings, "openai_compatible")
            brain = CountingBrain()
            worker = AlphaWorker(
                store=store,
                ai_client=LocalAIClient(expressions=["rank( mdl_mock_score )"]),
                brain_client=brain,
                policy=SubmissionPolicy(auto_submit=False),
                batch_size=1,
                context={"region": "USA", "universe": "TOP3000", "delay": 1, "neutralization": "INDUSTRY"},
            )

            summary = worker.run_once()

            self.assertEqual(summary["skipped"], 1)
            self.assertEqual([row["id"] for row in store.list_candidates()], [existing_id])
            self.assertEqual(brain.batch_calls, [])
            events = store.events_for_candidate(None)
            self.assertTrue(any(event["event_type"] == "duplicate_candidate_skipped" for event in events))


if __name__ == "__main__":
    unittest.main()
