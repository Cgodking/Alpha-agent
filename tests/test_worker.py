from __future__ import annotations

import json
import os
import tempfile
import time
import unittest
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from alpha.clients import LocalAIClient, LocalBrainClient
from alpha.db import AlphaStore
from alpha.guards import SubmissionPolicy
from alpha.models import CandidateSpec, DEFAULT_SETTINGS, SimulationFailure, SimulationResult, SubmitResult
from alpha.worker import (
    AlphaWorker,
    _ai_generation_stage_timeout_seconds,
    _current_quarter_date_range,
    _deterministic_fallback_candidates,
    _field_scout_fresh_generation_block,
    _simulation_stage_timeout_seconds,
)


class WorkerTests(unittest.TestCase):
    def test_current_quarter_date_range_uses_full_quarter_end(self):
        self.assertEqual(_current_quarter_date_range(date(2026, 5, 9)), ("2026-04-01", "2026-06-30"))
        self.assertEqual(_current_quarter_date_range(date(2026, 12, 15)), ("2026-10-01", "2026-12-31"))

    def test_simulation_stage_default_timeout_is_bounded_for_personal_throughput(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(_simulation_stage_timeout_seconds(), 900.0)

    def test_ai_generation_stage_default_timeout_is_bounded_for_personal_throughput(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(_ai_generation_stage_timeout_seconds(), 90.0)

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

    def test_worker_allows_fresh_ai_generation_when_field_scout_has_no_primary_fields(self):
        class CapturingAI:
            called = False

            def generate_candidates(self, batch_size, context):
                self.called = True
                return [CandidateSpec("group_rank(ts_rank(pv_only_signal, 22), industry)")]

        class OnlyLitPvBrain(LocalBrainClient):
            def discover_datafields(self, settings, search_terms=None, max_fields=120):
                return [
                    {
                        "id": "pv_only_signal",
                        "type": "MATRIX",
                        "dataset": {"id": "pv96", "name": "PV data"},
                        "category": {"id": "pv", "name": "Price Volume"},
                        "coverage": 0.9,
                        "userCount": 0,
                        "alphaCount": 0,
                        "pyramidMultiplier": 1.6,
                    }
                ]

            def get_pyramid_alphas(self, start_date=None, end_date=None):
                return {
                    "pyramids": [
                        {
                            "category": {"id": "pv", "name": "Price Volume"},
                            "region": "USA",
                            "delay": 0,
                            "alphaCount": 3,
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
            with patch.dict("os.environ", {"ALPHA_FIELD_CACHE_DIR": str(Path(tmp) / "cache")}, clear=False):
                worker = AlphaWorker(
                    store=store,
                    ai_client=ai,
                    brain_client=OnlyLitPvBrain(),
                    policy=SubmissionPolicy(auto_submit=False),
                    batch_size=1,
                    context={"region": "USA", "universe": "TOP500", "delay": 0},
                )

                summary = worker.run_once()

            self.assertTrue(ai.called)
            self.assertEqual(summary["generated"], 1)
            self.assertNotIn("field_scout_blocked", summary)
            self.assertEqual(len(store.list_candidates()), 1)
            events = store.events_for_candidate(None)
            self.assertFalse(any(event["event_type"] == "field_scout_generation_blocked" for event in events))

    def test_worker_skips_fresh_ai_generation_when_field_pool_is_empty(self):
        class FailingAI:
            def generate_candidates(self, batch_size, context):
                raise AssertionError("AI should not be called with an empty field allowlist")

        settings = {"region": "USA", "universe": "TOP500", "delay": 0, "neutralization": "INDUSTRY"}
        ai_context = {
            **settings,
            "research_context": {
                "datafields": {"available": False, "field_ids": []},
                "experiment_plan": {"mode": "explore_new_family"},
            },
        }
        with tempfile.TemporaryDirectory() as tmp:
            store = AlphaStore(Path(tmp) / "alpha.db")
            store.init()
            worker = AlphaWorker(
                store=store,
                ai_client=FailingAI(),
                brain_client=LocalBrainClient(),
                policy=SubmissionPolicy(auto_submit=False),
                batch_size=8,
                context=settings,
            )
            worker._build_cycle_ai_context = lambda cycle_plan=None: ai_context

            summary = worker.run_once()

            self.assertEqual(summary["generated"], 0)
            self.assertEqual(summary["skipped"], 8)
            self.assertEqual(summary["empty_field_pool_blocked"], 1)
            self.assertEqual(store.list_candidates(), [])
            events = store.events_for_candidate(None)
            self.assertTrue(
                any(event["event_type"] == "empty_field_pool_generation_blocked" for event in events)
            )

    def test_policy_for_ai_context_does_not_relax_below_configured_floor(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = AlphaStore(Path(tmp) / "alpha.db")
            store.init()
            policy = SubmissionPolicy(auto_submit=False, min_sharpe=1.58, min_fitness=1.0, max_turnover=0.70)
            worker = AlphaWorker(
                store=store,
                ai_client=LocalAIClient(),
                brain_client=LocalBrainClient(),
                policy=policy,
                batch_size=1,
                context={"region": "USA", "universe": "TOP3000", "delay": 0},
            )
            # Trusted thresholds below the configured floor must not weaken the guard.
            relaxed = worker._policy_for_ai_context(
                {
                    "research_context": {
                        "experiment_plan": {
                            "quality_thresholds": {
                                "trusted": True,
                                "required_sharpe": 0.5,
                                "required_fitness": 0.2,
                                "turnover_max": 0.95,
                            }
                        }
                    }
                }
            )
            self.assertEqual(relaxed.min_sharpe, 1.58)
            self.assertEqual(relaxed.min_fitness, 1.0)
            self.assertEqual(relaxed.max_turnover, 0.70)
            # Stricter trusted thresholds are allowed to tighten the guard.
            tighter = worker._policy_for_ai_context(
                {
                    "research_context": {
                        "experiment_plan": {
                            "quality_thresholds": {
                                "trusted": True,
                                "required_sharpe": 2.0,
                                "turnover_max": 0.40,
                            }
                        }
                    }
                }
            )
            self.assertEqual(tighter.min_sharpe, 2.0)
            self.assertEqual(tighter.max_turnover, 0.40)
            self.assertEqual(tighter.min_fitness, 1.0)

        block = _field_scout_fresh_generation_block(
            {
                "research_context": {
                    "experiment_plan": {
                        "mode": "explore_new_family",
                        "production_rescue": {"active": True},
                        "quality_budget": {"slots": {"probe_new_fields": 8}},
                        "probe_recommendations": [],
                    },
                    "field_scout": {
                        "active": True,
                        "status": "ready",
                        "top_fields": [{"field": "anl4_weak_signal"}],
                        "top_primary_fields": [{"field": "anl4_weak_signal"}],
                    },
                }
            }
        )

        self.assertEqual(block["reason"], "PRODUCTION_RESCUE_NO_SAFE_PROBES")

    def test_worker_allows_field_scout_retest_lane_when_no_top_primary_fields(self):
        block = _field_scout_fresh_generation_block(
            {
                "research_context": {
                    "experiment_plan": {
                        "mode": "explore_new_family",
                        "quality_budget": {"slots": {"broad_explore": 8, "probe_new_fields": 0}},
                        "field_scout": {
                            "retest_primary_fields": [
                                {
                                    "field": "fnd6_aqi",
                                    "dataset_id": "fundamental6",
                                    "category": "Fundamental",
                                    "primary_policy": "field_native_retest",
                                }
                            ]
                        },
                    },
                    "field_scout": {
                        "active": False,
                        "status": "no_primary_fields",
                        "top_fields": [{"field": "mdl_overused_signal", "primary_policy": "avoid_primary"}],
                        "top_primary_fields": [],
                    },
                }
            }
        )

        self.assertEqual(block, {})

    def test_worker_runs_production_rescue_probe_templates_without_ai(self):
        class FailingAI:
            def generate_candidates(self, batch_size, context):
                raise AssertionError("AI generation should not be called for production rescue probes")

            def validate_candidate_specs(self, candidates, batch_size, context):
                raise AssertionError("AI validation should not be called for production rescue probes")

        class ProbeBrain(LocalBrainClient):
            def __init__(self):
                super().__init__()
                self.batch_calls = []

            def simulate_many(self, items):
                self.batch_calls.append(items)
                return [
                    SimulationResult(
                        alpha_id="LOW_PROBE",
                        metrics={"sharpe": 0.2, "fitness": 0.05, "turnover": 0.2, "returns": 0.01, "drawdown": 0.2},
                        checks={
                            "LOW_SHARPE": {"status": "FAIL", "value": 0.2, "limit": 2.69},
                            "LOW_FITNESS": {"status": "FAIL", "value": 0.05, "limit": 1.5},
                            "LOW_TURNOVER": {"status": "PASS", "value": 0.2, "limit": 0.01},
                            "HIGH_TURNOVER": {"status": "PASS", "value": 0.2, "limit": 0.7},
                        },
                    ),
                    SimulationResult(
                        alpha_id="OPTIMIZE_PROBE",
                        metrics={"sharpe": 1.4, "fitness": 0.4, "turnover": 0.18, "returns": 0.03, "drawdown": 0.1},
                        checks={
                            "LOW_SHARPE": {"status": "FAIL", "value": 1.4, "limit": 2.69},
                            "LOW_FITNESS": {"status": "FAIL", "value": 0.4, "limit": 1.5},
                            "LOW_TURNOVER": {"status": "PASS", "value": 0.18, "limit": 0.01},
                            "HIGH_TURNOVER": {"status": "PASS", "value": 0.18, "limit": 0.7},
                        },
                    ),
                ]

        ai_context = {
            "region": "USA",
            "universe": "TOP500",
            "delay": 0,
            "research_context": {
                "datafields": {
                    "field_ids": ["mdl_mock_score"],
                    "field_types": {"mdl_mock_score": "MATRIX"},
                    "fields": [
                        {
                            "id": "mdl_mock_score",
                            "type": "MATRIX",
                            "dataset_id": "model16",
                            "category": "Model",
                        }
                    ],
                },
                "experiment_plan": {
                    "mode": "explore_new_family",
                    "production_rescue": {"active": True},
                    "quality_budget": {"slots": {"probe_new_fields": 2}},
                    "quality_thresholds": {
                        "trusted": True,
                        "required_sharpe": 2.69,
                        "required_fitness": 1.5,
                        "optimize_sharpe": 1.345,
                        "optimize_fitness": 0.375,
                        "setting_sweep_sharpe": 2.2865,
                        "setting_sweep_fitness": 1.125,
                        "setting_sweep_readiness": 0.8,
                        "turnover_min": 0.01,
                        "turnover_max": 0.7,
                    },
                    "probe_recommendations": [
                        {
                            "field": "mdl_mock_score",
                            "dataset_id": "model16",
                            "category": "Model",
                            "route": "production_rescue_probe",
                            "templates": [
                                "group_rank(ts_rank(winsorize(ts_backfill(mdl_mock_score,120),std=4),63),industry)",
                                "rank(ts_decay_linear(ts_backfill(mdl_mock_score,120),20))",
                            ],
                        }
                    ],
                },
            },
        }

        with tempfile.TemporaryDirectory() as tmp:
            store = AlphaStore(Path(tmp) / "alpha.db")
            store.init()
            brain = ProbeBrain()
            worker = AlphaWorker(
                store=store,
                ai_client=FailingAI(),
                brain_client=brain,
                policy=SubmissionPolicy(auto_submit=False),
                batch_size=2,
                context={"region": "USA", "universe": "TOP500", "delay": 0, "neutralization": "INDUSTRY"},
            )
            worker._build_cycle_ai_context = lambda cycle_plan=None: ai_context

            summary = worker.run_once()

            rows = store.list_candidates()
            self.assertEqual(summary["generated"], 2)
            self.assertEqual(summary["probe_reject"], 1)
            self.assertEqual(summary["probe_optimize_ready"], 1)
            self.assertEqual(len(brain.batch_calls), 1)
            self.assertEqual([row["source"] for row in rows], ["planner_unverified_probe", "planner_unverified_probe"])
            generated_events = [store.events_for_candidate(int(row["id"]))[0] for row in rows]
            generated_metadata = [json.loads(event["metadata_json"]) for event in generated_events]
            self.assertEqual(
                generated_metadata[0]["ai_metadata"]["validation_stage"],
                "unverified_probe",
            )
            global_events = store.events_for_candidate(None)
            self.assertFalse(any(event["event_type"] == "model_allocation" for event in global_events))
            probe_events = [
                json.loads(event["metadata_json"])
                for row in rows
                for event in store.events_for_candidate(int(row["id"]))
                if event["event_type"] == "probe_validation"
            ]
            self.assertEqual([event["stage"] for event in probe_events], ["reject", "optimize_ready"])

    def test_worker_mixes_standardized_probes_with_ai_when_broad_exploration_has_budget(self):
        class MixedAI:
            def __init__(self):
                self.batch_sizes = []

            def generate_candidates(self, batch_size, context):
                self.batch_sizes.append(batch_size)
                windows = [22, 33, 44, 66, 120, 252]
                return [
                    CandidateSpec(
                        f"rank(ts_rank(mdl_mock_score,{window}))",
                        source=f"model:G-{1 + index % 2}",
                    )
                    for index, window in enumerate(windows[:batch_size])
                ]

        class ProbeBrain(LocalBrainClient):
            def __init__(self):
                super().__init__()
                self.batch_calls = []

            def simulate_many(self, items):
                self.batch_calls.append(items)
                probe_results = [
                    SimulationResult(
                        alpha_id="PROBE_A",
                        metrics={"sharpe": 0.4, "fitness": 0.1, "turnover": 0.2, "returns": 0.01},
                        checks={
                            "LOW_SHARPE": {"status": "FAIL", "value": 0.4, "limit": 2.69},
                            "LOW_FITNESS": {"status": "FAIL", "value": 0.1, "limit": 1.5},
                        },
                    ),
                    SimulationResult(
                        alpha_id="PROBE_B",
                        metrics={"sharpe": 1.4, "fitness": 0.4, "turnover": 0.2, "returns": 0.03},
                        checks={
                            "LOW_SHARPE": {"status": "FAIL", "value": 1.4, "limit": 2.69},
                            "LOW_FITNESS": {"status": "FAIL", "value": 0.4, "limit": 1.5},
                            "LOW_TURNOVER": {"status": "PASS", "value": 0.2, "limit": 0.01},
                            "HIGH_TURNOVER": {"status": "PASS", "value": 0.2, "limit": 0.7},
                        },
                    ),
                ]
                ai_results = [
                    SimulationResult(
                        alpha_id=f"AI_{index}",
                        metrics={"sharpe": 0.9, "fitness": 0.3, "turnover": 0.2, "returns": 0.02},
                        checks={
                            "LOW_SHARPE": {"status": "FAIL", "value": 0.9, "limit": 2.69},
                            "LOW_FITNESS": {"status": "FAIL", "value": 0.3, "limit": 1.5},
                        },
                    )
                    for index in range(max(0, len(items) - len(probe_results)))
                ]
                return probe_results + ai_results

        ai_context = {
            "region": "USA",
            "universe": "TOP500",
            "delay": 0,
            "research_context": {
                "datafields": {
                    "field_ids": ["mdl_mock_score"],
                    "field_types": {"mdl_mock_score": "MATRIX"},
                },
                "experiment_plan": {
                    "mode": "explore_new_family",
                    "quality_budget": {
                        "slots": {"probe_new_fields": 2, "broad_explore": 6},
                        "exploit_fields": [],
                    },
                    "production_rescue": {"active": False},
                    "probe_recommendations": [
                        {
                            "field": "mdl_mock_score",
                            "dataset_id": "model16",
                            "category": "Model",
                            "route": "standardized_probe",
                            "templates": [
                                "rank(ts_mean(mdl_mock_score,20))",
                                "rank(ts_rank(mdl_mock_score,60))",
                            ],
                        }
                    ],
                },
            },
        }

        with tempfile.TemporaryDirectory() as tmp:
            store = AlphaStore(Path(tmp) / "alpha.db")
            store.init()
            brain = ProbeBrain()
            ai = MixedAI()
            worker = AlphaWorker(
                store=store,
                ai_client=ai,
                brain_client=brain,
                policy=SubmissionPolicy(auto_submit=False),
                batch_size=8,
                context={"region": "USA", "universe": "TOP500", "delay": 0, "neutralization": "INDUSTRY"},
            )
            worker._build_cycle_ai_context = lambda cycle_plan=None: ai_context

            summary = worker.run_once()

            rows = store.list_candidates()
            self.assertEqual(summary["generated"], 8)
            self.assertEqual(ai.batch_sizes, [6])
            self.assertEqual(len(brain.batch_calls), 1)
            self.assertEqual(len(brain.batch_calls[0]), 8)
            self.assertEqual(
                [row["source"] for row in rows],
                ["planner_standardized_probe", "planner_standardized_probe"]
                + [f"model:G-{1 + index % 2}" for index in range(6)],
            )
            generated_events = [store.events_for_candidate(int(row["id"]))[0] for row in rows]
            generated_metadata = [json.loads(event["metadata_json"]) for event in generated_events]
            self.assertEqual(generated_metadata[0]["ai_metadata"]["validation_stage"], "standardized_probe")
            global_events = store.events_for_candidate(None)
            self.assertFalse(any(event["event_type"] == "planner_validation_error" for event in global_events))
            probe_events = [
                json.loads(event["metadata_json"])
                for row in rows
                for event in store.events_for_candidate(int(row["id"]))
                if event["event_type"] == "probe_validation"
            ]
            self.assertEqual([event["validation_stage"] for event in probe_events], ["standardized_probe", "standardized_probe"])
            self.assertEqual([event["stage"] for event in probe_events], ["reject", "optimize_ready"])

    def test_worker_standardized_probe_skips_duplicate_templates_and_uses_next_template(self):
        class FailingAI:
            def generate_candidates(self, batch_size, context):
                raise AssertionError("standardized explore probes should not call AI generation")

            def validate_candidate_specs(self, candidates, batch_size, context):
                raise AssertionError("standardized explore probes should not call AI validation")

        class ProbeBrain(LocalBrainClient):
            def __init__(self):
                super().__init__()
                self.simulation_calls = []

            def simulate(self, expression, settings):
                self.simulation_calls.append((expression, settings))
                return SimulationResult(
                    alpha_id="PROBE_UNIQUE",
                    metrics={"sharpe": 0.5, "fitness": 0.12, "turnover": 0.2, "returns": 0.01},
                    checks={
                        "LOW_SHARPE": {"status": "FAIL", "value": 0.5, "limit": 2.69},
                        "LOW_FITNESS": {"status": "FAIL", "value": 0.12, "limit": 1.5},
                    },
                )

        duplicate_expression = "rank(ts_mean(mdl_mock_score,20))"
        unique_expression = "rank(ts_rank(mdl_mock_score,60))"
        settings = {"region": "USA", "universe": "TOP500", "delay": 0, "neutralization": "INDUSTRY"}
        ai_context = {
            "region": "USA",
            "universe": "TOP500",
            "delay": 0,
            "research_context": {
                "datafields": {
                    "field_ids": ["mdl_mock_score"],
                    "field_types": {"mdl_mock_score": "MATRIX"},
                },
                "experiment_plan": {
                    "mode": "explore_new_family",
                    "quality_budget": {
                        "slots": {"probe_new_fields": 1, "broad_explore": 7},
                        "exploit_fields": [],
                    },
                    "production_rescue": {"active": False},
                    "probe_recommendations": [
                        {
                            "field": "mdl_mock_score",
                            "dataset_id": "model16",
                            "category": "Model",
                            "route": "standardized_probe",
                            "templates": [duplicate_expression, unique_expression],
                        }
                    ],
                },
            },
        }

        with tempfile.TemporaryDirectory() as tmp:
            store = AlphaStore(Path(tmp) / "alpha.db")
            store.init()
            store.insert_candidate(duplicate_expression, {**DEFAULT_SETTINGS, **settings}, "planner_standardized_probe")
            brain = ProbeBrain()
            worker = AlphaWorker(
                store=store,
                ai_client=FailingAI(),
                brain_client=brain,
                policy=SubmissionPolicy(auto_submit=False),
                batch_size=8,
                context=settings,
            )
            worker._build_cycle_ai_context = lambda cycle_plan=None: ai_context

            summary = worker.run_once()

            rows = store.list_candidates()
            generated = [row for row in rows if row["expression"] == unique_expression]
            self.assertEqual(summary["generated"], 1)
            self.assertEqual(summary["skipped"], 1)
            self.assertEqual(len(generated), 1)
            self.assertEqual(generated[0]["source"], "planner_standardized_probe")
            self.assertEqual(len(brain.simulation_calls), 1)
            self.assertEqual(brain.simulation_calls[0][0], unique_expression)
            duplicate_events = [
                event
                for event in store.events_for_candidate(None)
                if event["event_type"] == "planned_probe_duplicate_skipped"
            ]
            self.assertEqual(len(duplicate_events), 1)

    def test_worker_standardized_probe_exhaustion_reclaims_slots_for_ai(self):
        class FallbackAI:
            def __init__(self):
                self.batch_sizes = []

            def generate_candidates(self, batch_size, context):
                self.batch_sizes.append(batch_size)
                return [
                    CandidateSpec(f"rank(ts_rank(aggregate_sentiment_score_3,{20 + index}))")
                    for index in range(batch_size)
                ]

            def validate_candidate_specs(self, candidates, batch_size, context):
                raise AssertionError("exhausted standardized probe should not call AI validation")

        class ProbeBrain(LocalBrainClient):
            def __init__(self):
                super().__init__()
                self.simulation_calls = []

            def simulate(self, expression, settings):
                self.simulation_calls.append((expression, settings))
                return super().simulate(expression, settings)

        settings = {"region": "USA", "universe": "TOP500", "delay": 0, "neutralization": "INDUSTRY"}
        expressions = [
            "rank(ts_mean(vec_avg(aggregate_sentiment_score_3), 20))",
            "rank(ts_rank(vec_avg(aggregate_sentiment_score_3), 60))",
            "group_rank(ts_mean(vec_avg(aggregate_sentiment_score_3), 60), industry)",
        ]
        ai_context = {
            **settings,
            "research_context": {
                "field_ids": ["aggregate_sentiment_score_3"],
                "field_types": {"aggregate_sentiment_score_3": "VECTOR"},
                "experiment_plan": {
                    "mode": "explore_new_family",
                    "quality_budget": {
                        "slots": {"probe_new_fields": 1, "broad_explore": 7},
                        "exploit_fields": [],
                    },
                    "production_rescue": {"active": False},
                    "probe_recommendations": [
                        {
                            "field": "aggregate_sentiment_score_3",
                            "dataset_id": "filing_sentiment",
                            "category": "Other",
                            "route": "standardized_probe",
                            "templates": expressions,
                        }
                    ],
                },
            },
        }

        with tempfile.TemporaryDirectory() as tmp:
            store = AlphaStore(Path(tmp) / "alpha.db")
            store.init()
            for expression in expressions:
                store.insert_candidate(expression, {**DEFAULT_SETTINGS, **settings}, "planner_standardized_probe")
            brain = ProbeBrain()
            ai = FallbackAI()
            worker = AlphaWorker(
                store=store,
                ai_client=ai,
                brain_client=brain,
                policy=SubmissionPolicy(auto_submit=False),
                batch_size=8,
                context=settings,
            )
            worker._build_cycle_ai_context = lambda cycle_plan=None: ai_context

            summary = worker.run_once(
                cycle_plan={
                    "mode": "explore",
                    "reason": "pending_recheck_cooldown",
                    "scope": settings,
                }
            )

            self.assertEqual(summary["generated"], 8)
            self.assertEqual(summary["pending"], 0)
            self.assertEqual(summary["skipped"], 3)
            self.assertEqual(ai.batch_sizes, [8])
            self.assertEqual(len(brain.simulation_calls), 8)
            global_events = store.events_for_candidate(None)
            event_types = [event["event_type"] for event in global_events]
            self.assertIn("standardized_probe_exhausted", event_types)
            self.assertIn("cycle_outcome", event_types)

    def test_worker_production_rescue_probe_skips_duplicate_templates_and_uses_next_template(self):
        class FailingAI:
            def generate_candidates(self, batch_size, context):
                raise AssertionError("production rescue probes should not fall back to AI generation")

            def validate_candidate_specs(self, candidates, batch_size, context):
                raise AssertionError("production rescue probes should not call AI validation")

        class ProbeBrain(LocalBrainClient):
            def __init__(self):
                super().__init__()
                self.simulation_calls = []

            def simulate(self, expression, settings):
                self.simulation_calls.append((expression, settings))
                return super().simulate(expression, settings)

        settings = {"region": "USA", "universe": "TOP500", "delay": 0, "neutralization": "INDUSTRY"}
        duplicate_expression = "group_rank(ts_rank(winsorize(ts_backfill(snt21_4neut_conf_low,120),std=4),63),industry)"
        unique_expression = "rank(ts_decay_linear(ts_backfill(snt21_4neut_conf_low,120),20))"
        ai_context = {
            **settings,
            "research_context": {
                "field_ids": ["snt21_4neut_conf_low"],
                "field_types": {"snt21_4neut_conf_low": "MATRIX"},
                "experiment_plan": {
                    "mode": "explore_new_family",
                    "quality_budget": {"slots": {"probe_new_fields": 1}},
                    "production_rescue": {"active": True, "reason": "quality_stop_loss_repeated"},
                    "probe_recommendations": [
                        {
                            "field": "snt21_4neut_conf_low",
                            "dataset_id": "sentiment21",
                            "category": "Sentiment",
                            "route": "production_rescue_probe",
                            "templates": [duplicate_expression, unique_expression],
                        }
                    ],
                },
            },
        }

        with tempfile.TemporaryDirectory() as tmp:
            store = AlphaStore(Path(tmp) / "alpha.db")
            store.init()
            store.insert_candidate(duplicate_expression, {**DEFAULT_SETTINGS, **settings}, "planner_unverified_probe")
            brain = ProbeBrain()
            worker = AlphaWorker(
                store=store,
                ai_client=FailingAI(),
                brain_client=brain,
                policy=SubmissionPolicy(auto_submit=False),
                batch_size=8,
                context=settings,
            )
            worker._build_cycle_ai_context = lambda cycle_plan=None: ai_context

            summary = worker.run_once()

            rows = store.list_candidates()
            generated = [row for row in rows if row["expression"] == unique_expression]
            self.assertEqual(summary["generated"], 1)
            self.assertEqual(summary["skipped"], 1)
            self.assertEqual(len(generated), 1)
            self.assertEqual(generated[0]["source"], "planner_unverified_probe")
            self.assertEqual(len(brain.simulation_calls), 1)
            self.assertEqual(brain.simulation_calls[0][0], unique_expression)
            duplicate_events = [
                event
                for event in store.events_for_candidate(None)
                if event["event_type"] == "planned_probe_duplicate_skipped"
            ]
            self.assertEqual(len(duplicate_events), 1)

    def test_worker_production_rescue_probe_exhaustion_does_not_fall_back_to_ai(self):
        class FailingAI:
            def generate_candidates(self, batch_size, context):
                raise AssertionError("exhausted production rescue probe should not fall back to AI generation")

            def validate_candidate_specs(self, candidates, batch_size, context):
                raise AssertionError("exhausted production rescue probe should not call AI validation")

        class ProbeBrain(LocalBrainClient):
            def __init__(self):
                super().__init__()
                self.simulation_calls = []

            def simulate(self, expression, settings):
                self.simulation_calls.append((expression, settings))
                return super().simulate(expression, settings)

        settings = {"region": "USA", "universe": "TOP500", "delay": 0, "neutralization": "INDUSTRY"}
        expressions = [
            "group_rank(ts_rank(winsorize(ts_backfill(snt21_4neut_conf_low,120),std=4),63),industry)",
            "rank(ts_decay_linear(ts_backfill(snt21_4neut_conf_low,120),20))",
        ]
        ai_context = {
            **settings,
            "research_context": {
                "field_ids": ["snt21_4neut_conf_low"],
                "field_types": {"snt21_4neut_conf_low": "MATRIX"},
                "experiment_plan": {
                    "mode": "explore_new_family",
                    "quality_budget": {"slots": {"probe_new_fields": 1}},
                    "production_rescue": {"active": True, "reason": "quality_stop_loss_repeated"},
                    "probe_recommendations": [
                        {
                            "field": "snt21_4neut_conf_low",
                            "dataset_id": "sentiment21",
                            "category": "Sentiment",
                            "route": "production_rescue_probe",
                            "templates": expressions,
                        }
                    ],
                },
            },
        }

        with tempfile.TemporaryDirectory() as tmp:
            store = AlphaStore(Path(tmp) / "alpha.db")
            store.init()
            for expression in expressions:
                store.insert_candidate(expression, {**DEFAULT_SETTINGS, **settings}, "planner_unverified_probe")
            brain = ProbeBrain()
            worker = AlphaWorker(
                store=store,
                ai_client=FailingAI(),
                brain_client=brain,
                policy=SubmissionPolicy(auto_submit=False),
                batch_size=8,
                context=settings,
            )
            worker._build_cycle_ai_context = lambda cycle_plan=None: ai_context

            summary = worker.run_once(
                cycle_plan={
                    "mode": "explore",
                    "reason": "pending_recheck_cooldown",
                    "scope": settings,
                }
            )

            self.assertEqual(summary["generated"], 0)
            self.assertEqual(summary["pending"], 0)
            self.assertEqual(summary["skipped"], 2)
            self.assertEqual(summary["production_rescue_probe_exhausted"], 1)
            self.assertEqual(brain.simulation_calls, [])
            global_events = store.events_for_candidate(None)
            event_types = [event["event_type"] for event in global_events]
            self.assertIn("production_rescue_probe_exhausted", event_types)
            self.assertIn("cycle_outcome", event_types)

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
            store.update_candidate(
                best_id,
                metrics_json=json.dumps({"sharpe": 2.55, "fitness": 1.32, "turnover": 0.22}),
                checks_json=json.dumps(
                    {
                        "LOW_SHARPE": {"status": "FAIL", "value": 2.55, "limit": 2.69},
                        "LOW_FITNESS": {"status": "FAIL", "value": 1.32, "limit": 1.5},
                        "LOW_TURNOVER": {"status": "PASS", "value": 0.22, "limit": 0.01},
                        "HIGH_TURNOVER": {"status": "PASS", "value": 0.22, "limit": 0.7},
                        "CONCENTRATED_WEIGHT": {"status": "PASS"},
                        "LOW_2Y_SHARPE": {"status": "PASS", "value": 2.8, "limit": 2.69},
                        "LOW_SUB_UNIVERSE_SHARPE": {"status": "PASS", "value": 1.1, "limit": 0.49},
                        "IS_LADDER_SHARPE": {"status": "PASS", "value": 2.8, "limit": 2.69},
                    }
                ),
            )
            store.transition(
                best_id,
                "check_pending",
                {"errors": ["SHARPE_BELOW_MIN:2.550<2.69", "FITNESS_BELOW_MIN:1.320<1.5"]},
            )
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

    def test_worker_honors_cycle_plan_optimize_target_before_rescue_probes(self):
        class CapturingAI:
            def __init__(self):
                self.context = None

            def generate_candidates(self, batch_size, context):
                self.context = context
                return []

        with tempfile.TemporaryDirectory() as tmp:
            store = AlphaStore(Path(tmp) / "alpha.db")
            store.init()
            target_id = store.insert_candidate(
                "rank(ts_mean(mdl_mock_score,30))",
                {"region": "USA", "universe": "TOP500", "delay": 0, "neutralization": "INDUSTRY"},
                "planner_unverified_probe",
            )
            store.update_candidate(
                target_id,
                metrics_json=json.dumps({"sharpe": 0.1, "fitness": 0.05, "turnover": 0.18}),
                checks_json=json.dumps(
                    {
                        "LOW_SHARPE": {"status": "FAIL", "value": 0.1, "limit": 2.69},
                        "LOW_FITNESS": {"status": "FAIL", "value": 0.05, "limit": 1.5},
                        "LOW_2Y_SHARPE": {"status": "FAIL", "value": 0.0, "limit": 2.69},
                        "LOW_SUB_UNIVERSE_SHARPE": {"status": "FAIL", "value": -0.2, "limit": 0.0},
                        "LOW_TURNOVER": {"status": "PASS", "value": 0.18, "limit": 0.01},
                        "HIGH_TURNOVER": {"status": "PASS", "value": 0.18, "limit": 0.7},
                    }
                ),
            )
            store.transition(target_id, "failed", {"reason": "probe_signal"})
            store.record_event(target_id, "probe_validation", {"stage": "optimize_ready"})
            for idx in range(120):
                failed_id = store.insert_candidate(
                    f"rank(weak_signal_{idx})",
                    {"region": "USA", "universe": "TOP500", "delay": 0, "neutralization": "INDUSTRY"},
                    "planner_unverified_probe",
                )
                store.update_candidate(failed_id, metrics_json=json.dumps({"sharpe": 0.0, "fitness": 0.0}))
                store.transition(failed_id, "failed", {"reason": "bad_full_batch"})
            ai = CapturingAI()
            worker = AlphaWorker(
                store=store,
                ai_client=ai,
                brain_client=LocalBrainClient(),
                policy=SubmissionPolicy(auto_submit=False),
                batch_size=4,
                context={"region": "USA", "universe": "TOP500", "delay": 0, "neutralization": "INDUSTRY"},
            )

            worker.run_once(
                cycle_plan={
                    "mode": "optimize",
                    "target_candidate_id": target_id,
                    "budget": {"batch_size": 4},
                    "reason": "probe_optimize_ready_candidate_has_fixable_gap",
                }
            )

            plan = ai.context["research_context"]["experiment_plan"]
            rows = store.list_candidates()

        self.assertEqual(plan["mode"], "optimize_best")
        self.assertEqual(plan["target_candidate_id"], target_id)
        self.assertEqual(len([row for row in rows if row["source"] == "planner_unverified_probe"]), 121)

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

    def test_worker_marks_approved_dry_run_alpha_green_on_platform(self):
        class ColorRecordingBrain(LocalBrainClient):
            def __init__(self):
                super().__init__()
                self.property_updates = []

            def set_alpha_properties(self, alpha_id: str, **properties):
                self.property_updates.append({"alpha_id": alpha_id, **properties})
                return {"id": alpha_id, **properties}

        with tempfile.TemporaryDirectory() as tmp:
            store = AlphaStore(Path(tmp) / "alpha.db")
            store.init()
            brain = ColorRecordingBrain()
            worker = AlphaWorker(
                store=store,
                ai_client=LocalAIClient(expressions=["rank(mdl_mock_score)"]),
                brain_client=brain,
                policy=SubmissionPolicy(auto_submit=False),
                batch_size=1,
            )

            summary = worker.run_once()

            candidate = store.list_candidates()[0]
            events = store.events_for_candidate(candidate["id"])
            alpha_id = candidate["alpha_id"]
            self.assertEqual(brain.property_updates, [{"alpha_id": alpha_id, "color": "GREEN"}])
            self.assertEqual(summary["platform_color_set"], 1)
            self.assertTrue(any(event["event_type"] == "platform_color_set" for event in events))

    def test_worker_records_unverified_submit_attempt_separately_from_dry_run(self):
        class UnverifiedSubmitBrain(LocalBrainClient):
            def __init__(self):
                super().__init__()
                self.submit_calls = []

            def submit_alpha(self, alpha_id: str, dry_run: bool = True) -> SubmitResult:
                self.submit_calls.append({"alpha_id": alpha_id, "dry_run": dry_run})
                return SubmitResult(
                    alpha_id=alpha_id,
                    submitted=False,
                    stage="IS",
                    message="platform did not verify OS",
                )

        with tempfile.TemporaryDirectory() as tmp:
            store = AlphaStore(Path(tmp) / "alpha.db")
            store.init()
            brain = UnverifiedSubmitBrain()
            worker = AlphaWorker(
                store=store,
                ai_client=LocalAIClient(expressions=["rank(mdl_mock_score)"]),
                brain_client=brain,
                policy=SubmissionPolicy(auto_submit=True),
                batch_size=1,
            )

            summary = worker.run_once()

            candidate = store.list_candidates()[0]
            event_types = [event["event_type"] for event in store.events_for_candidate(candidate["id"])]
            self.assertEqual(brain.submit_calls[0]["dry_run"], False)
            self.assertEqual(candidate["status"], "approved")
            self.assertIn("submit_unverified", event_types)
            self.assertNotIn("dry_run_submit", event_types)
            self.assertEqual(summary["submit_unverified"], 1)

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

    def test_worker_keeps_simulation_poll_timeout_pending_for_recovery(self):
        class PollingTimeoutBrain(LocalBrainClient):
            def simulate(self, expression, settings):
                from alpha.models import SimulationPendingError

                raise SimulationPendingError("/simulations/slow")

        with tempfile.TemporaryDirectory() as tmp:
            store = AlphaStore(Path(tmp) / "alpha.db")
            store.init()
            worker = AlphaWorker(
                store=store,
                ai_client=LocalAIClient(expressions=["rank(mdl_mock_score)"]),
                brain_client=PollingTimeoutBrain(),
                policy=SubmissionPolicy(auto_submit=False, max_retries=1),
                batch_size=1,
                context={"region": "USA", "universe": "TOP500", "delay": 0, "neutralization": "INDUSTRY"},
            )

            summary = worker.run_once()

            self.assertEqual(summary["generated"], 1)
            self.assertEqual(summary["pending"], 1)
            self.assertEqual(summary["failed"], 0)
            self.assertNotIn("quality_stop_loss", summary)
            candidate = store.list_candidates()[0]
            self.assertEqual(candidate["status"], "check_pending")
            self.assertFalse(
                any(event["event_type"] == "quality_stop_loss" for event in store.events_for_candidate(None))
            )
            events = store.events_for_candidate(candidate["id"])
            self.assertTrue(
                any(
                    event["event_type"] == "simulation_pending"
                    and "/simulations/slow" in event["metadata_json"]
                    for event in events
                )
            )

    def test_worker_does_not_retry_simulation_stage_timeout(self):
        # A stage timeout must not be retried: the real simulation is likely still running
        # on the platform, so a retry would fire a duplicate and waste quota.
        class SlowBrain(LocalBrainClient):
            def __init__(self):
                super().__init__()
                self.simulate_calls = 0

            def simulate(self, expression, settings):
                self.simulate_calls += 1
                time.sleep(0.5)
                return super().simulate(expression, settings)

        with tempfile.TemporaryDirectory() as tmp:
            store = AlphaStore(Path(tmp) / "alpha.db")
            store.init()
            brain = SlowBrain()
            worker = AlphaWorker(
                store=store,
                ai_client=LocalAIClient(expressions=["rank(mdl_mock_score)"]),
                brain_client=brain,
                policy=SubmissionPolicy(auto_submit=False, max_retries=3),
                batch_size=1,
                context={"region": "USA", "universe": "TOP500", "delay": 0, "neutralization": "INDUSTRY"},
            )
            with patch.dict(os.environ, {"SIMULATION_STAGE_TIMEOUT_SECONDS": "0.05"}, clear=False):
                summary = worker.run_once()

            self.assertEqual(summary["failed"], 1)
            self.assertEqual(brain.simulate_calls, 1)
            candidate = store.list_candidates()[0]
            self.assertEqual(candidate["status"], "failed")
            self.assertTrue(
                any(
                    event["event_type"] == "simulation_error"
                    and "simulation_stage_timeout" in (event["metadata_json"] or "")
                    for event in store.events_for_candidate(candidate["id"])
                )
            )

    def test_worker_recovers_preflight_passed_candidate_before_fresh_generation(self):
        class FailingAI:
            def generate_candidates(self, batch_size, context):
                raise AssertionError("preflight recovery should run before fresh AI generation")

        class RecoveringBrain(LocalBrainClient):
            def __init__(self):
                super().__init__()
                self.simulated = []

            def simulate(self, expression, settings):
                self.simulated.append((expression, settings))
                return SimulationResult(
                    alpha_id="RECOVERED123",
                    metrics={"sharpe": 3.1, "fitness": 1.7, "turnover": 0.2},
                    checks={
                        "LOW_SHARPE": {"status": "PASS", "value": 3.1, "limit": 2.69},
                        "LOW_FITNESS": {"status": "PASS", "value": 1.7, "limit": 1.5},
                        "LOW_TURNOVER": {"status": "PASS", "value": 0.2, "limit": 0.01},
                        "HIGH_TURNOVER": {"status": "PASS", "value": 0.2, "limit": 0.7},
                        "SELF_CORRELATION": {"status": "PASS"},
                        "PROD_CORRELATION": {"status": "PASS"},
                        "DATA_DIVERSITY": {"status": "PASS"},
                        "REGULAR_SUBMISSION": {"status": "PASS", "value": 0, "limit": 4},
                    },
                )

        with tempfile.TemporaryDirectory() as tmp:
            store = AlphaStore(Path(tmp) / "alpha.db")
            store.init()
            settings = {"region": "USA", "universe": "TOP500", "delay": 0, "neutralization": "INDUSTRY"}
            candidate_id = store.insert_candidate("rank(ts_mean(mdl_mock_score,22))", settings, "model:G-1")
            store.transition(candidate_id, "preflight_passed")
            brain = RecoveringBrain()
            worker = AlphaWorker(
                store=store,
                ai_client=FailingAI(),
                brain_client=brain,
                policy=SubmissionPolicy(auto_submit=False, min_sharpe=2.69, min_fitness=1.5),
                batch_size=1,
                context=settings,
                cycle_plan={"mode": "explore", "reason": "preflight_recovery"},
            )

            summary = worker.run_once()

            candidate = store.get_candidate(candidate_id)
            self.assertEqual(summary["generated"], 0)
            self.assertEqual(summary["approved"], 1)
            self.assertEqual(len(brain.simulated), 1)
            self.assertEqual(candidate["status"], "approved")
            self.assertEqual(candidate["alpha_id"], "RECOVERED123")
            stage_events = [
                json.loads(event["metadata_json"])
                for event in store.events_for_candidate(None)
                if event["event_type"] == "cycle_stage"
            ]
            simulation_stages = [event for event in stage_events if event["stage"] == "simulation_candidate_started"]
            self.assertTrue(simulation_stages)
            self.assertEqual(simulation_stages[0]["cycle_mode"], "explore")

    def test_worker_recovers_pending_simulation_location_without_new_generation(self):
        class FailingAI:
            def __init__(self):
                self.calls = 0

            def generate_candidates(self, batch_size, context):
                self.calls += 1
                raise AssertionError("recover_pending should not generate fresh candidates")

        class RecoveringBrain(LocalBrainClient):
            def __init__(self):
                super().__init__()
                self.generated_simulations = 0
                self.resumed_locations = []

            def simulate(self, expression, settings):
                self.generated_simulations += 1
                return super().simulate(expression, settings)

            def resume_simulation(self, location):
                self.resumed_locations.append(location)
                return SimulationResult(
                    alpha_id="RECOVERED1",
                    metrics={"sharpe": 2.9, "fitness": 1.7, "turnover": 0.2, "returns": 0.09},
                    checks={
                        "LOW_SHARPE": {"status": "PASS", "value": 2.9, "limit": 2.69},
                        "LOW_FITNESS": {"status": "PASS", "value": 1.7, "limit": 1.5},
                        "LOW_TURNOVER": {"status": "PASS", "value": 0.2, "limit": 0.01},
                        "HIGH_TURNOVER": {"status": "PASS", "value": 0.2, "limit": 0.7},
                        "SELF_CORRELATION": {"status": "PASS"},
                        "PROD_CORRELATION": {"status": "PASS"},
                        "DATA_DIVERSITY": {"status": "PASS"},
                        "REGULAR_SUBMISSION": {"status": "PASS"},
                    },
                )

        with tempfile.TemporaryDirectory() as tmp:
            store = AlphaStore(Path(tmp) / "alpha.db")
            store.init()
            candidate_id = store.insert_candidate(
                "rank(mdl_mock_score)",
                {"region": "USA", "universe": "TOP3000", "delay": 1, "neutralization": "INDUSTRY"},
                "model:G-1",
            )
            store.transition(candidate_id, "check_pending", {"reason": "simulation_pending"})
            store.record_event(candidate_id, "simulation_pending", {"location": "/simulations/slow"})
            brain = RecoveringBrain()
            ai = FailingAI()
            worker = AlphaWorker(
                store=store,
                ai_client=ai,
                brain_client=brain,
                policy=SubmissionPolicy(auto_submit=False, min_sharpe=2.69, min_fitness=1.5),
                batch_size=1,
                context={"region": "USA", "universe": "TOP3000", "delay": 1, "neutralization": "INDUSTRY"},
            )

            summary = worker.run_once(
                cycle_plan={
                    "mode": "recover_pending",
                    "scope": {"region": "USA", "universe": "TOP3000", "delay": 1, "neutralization": "INDUSTRY"},
                    "target_candidate_id": candidate_id,
                }
            )

            candidate = store.get_candidate(candidate_id)
            self.assertEqual(brain.resumed_locations, ["/simulations/slow"])
            self.assertEqual(brain.generated_simulations, 0)
            self.assertEqual(candidate["alpha_id"], "RECOVERED1")
            self.assertEqual(candidate["status"], "approved")
            self.assertEqual(summary["approved"], 1)
            self.assertEqual(summary["generated"], 0)
            self.assertEqual(ai.calls, 0)

    def test_worker_fails_terminal_pending_simulation_resume_error(self):
        class FailingAI:
            def generate_candidates(self, batch_size, context):
                raise AssertionError("recover_pending should not generate fresh candidates")

        class FailedResumeBrain(LocalBrainClient):
            def __init__(self):
                super().__init__()
                self.resumed_locations = []

            def resume_simulation(self, location):
                self.resumed_locations.append(location)
                raise RuntimeError(
                    "simulation failed on platform: There was an error while running the simulation. "
                    "Please try again or contact BRAIN support if this problem persists."
                )

        with tempfile.TemporaryDirectory() as tmp:
            store = AlphaStore(Path(tmp) / "alpha.db")
            store.init()
            settings = {"region": "USA", "universe": "TOP500", "delay": 0, "neutralization": "INDUSTRY"}
            candidate_id = store.insert_candidate("rank(ts_rank(vec_avg(lending_fee_bid_rate), 60))", settings, "model:G-1")
            store.transition(candidate_id, "check_pending", {"reason": "simulation_pending"})
            store.record_event(candidate_id, "simulation_pending", {"location": "/simulations/failed"})
            brain = FailedResumeBrain()
            worker = AlphaWorker(
                store=store,
                ai_client=FailingAI(),
                brain_client=brain,
                policy=SubmissionPolicy(auto_submit=False, min_sharpe=2.69, min_fitness=1.5),
                batch_size=1,
                context=settings,
            )

            summary = worker.run_once(
                cycle_plan={
                    "mode": "recover_pending",
                    "scope": settings,
                    "target_candidate_id": candidate_id,
                }
            )

            candidate = store.get_candidate(candidate_id)
            events = store.events_for_candidate(candidate_id)
            failure_events = [
                json.loads(event["metadata_json"])
                for event in events
                if event["event_type"] == "status:failed"
            ]
            self.assertEqual(brain.resumed_locations, ["/simulations/failed"])
            self.assertEqual(candidate["status"], "failed")
            self.assertEqual(summary["failed"], 1)
            self.assertEqual(summary["pending"], 0)
            self.assertTrue(any(event["event_type"] == "pending_simulation_resume_error" for event in events))
            self.assertEqual(failure_events[-1]["reason"], "simulation_resume_failed")

    def test_worker_fails_stale_pending_simulation_without_resuming_again(self):
        class FailingAI:
            def generate_candidates(self, batch_size, context):
                raise AssertionError("recover_pending should not generate fresh candidates")

        class BlockingResumeBrain(LocalBrainClient):
            def __init__(self):
                super().__init__()
                self.resume_calls = 0

            def resume_simulation(self, location):
                self.resume_calls += 1
                raise AssertionError("stale pending simulation should be failed before resume")

        with tempfile.TemporaryDirectory() as tmp:
            store = AlphaStore(Path(tmp) / "alpha.db")
            store.init()
            settings = {"region": "USA", "universe": "TOP3000", "delay": 1, "neutralization": "INDUSTRY"}
            candidate_id = store.insert_candidate("rank(mdl_mock_score)", settings, "model:G-1")
            store.transition(candidate_id, "check_pending", {"reason": "simulation_pending"})
            store.record_event(candidate_id, "simulation_pending", {"location": "/simulations/stale"})
            store.record_event(candidate_id, "simulation_pending", {"location": "/simulations/stale"})
            with store.connection() as conn:
                conn.execute(
                    "UPDATE events SET created_at = ? WHERE candidate_id = ? AND event_type = 'simulation_pending'",
                    ("2026-01-01T00:00:00+00:00", candidate_id),
                )
            brain = BlockingResumeBrain()
            worker = AlphaWorker(
                store=store,
                ai_client=FailingAI(),
                brain_client=brain,
                policy=SubmissionPolicy(auto_submit=False, min_sharpe=2.69, min_fitness=1.5),
                batch_size=1,
                context=settings,
            )

            with patch.dict(
                os.environ,
                {"PENDING_RECHECK_STALE_ATTEMPTS": "2", "PENDING_RECHECK_STALE_SECONDS": "60"},
            ):
                summary = worker.run_once(
                    cycle_plan={
                        "mode": "recover_pending",
                        "scope": settings,
                        "target_candidate_id": candidate_id,
                    }
                )

            candidate = store.get_candidate(candidate_id)
            events = store.events_for_candidate(candidate_id)
            self.assertEqual(brain.resume_calls, 0)
            self.assertEqual(candidate["status"], "failed")
            self.assertEqual(summary["failed"], 1)
            self.assertEqual(summary["stale_pending_failed"], 1)
            self.assertTrue(any(event["event_type"] == "pending_recheck_stale_failed" for event in events))

    def test_worker_default_stale_pending_window_is_short_enough_for_throughput(self):
        class FailingAI:
            def generate_candidates(self, batch_size, context):
                raise AssertionError("recover_pending should not generate fresh candidates")

        class BlockingResumeBrain(LocalBrainClient):
            def __init__(self):
                super().__init__()
                self.resume_calls = 0

            def resume_simulation(self, location):
                self.resume_calls += 1
                raise AssertionError("default stale pending simulation should be failed before resume")

        with tempfile.TemporaryDirectory() as tmp:
            store = AlphaStore(Path(tmp) / "alpha.db")
            store.init()
            settings = {"region": "USA", "universe": "TOP500", "delay": 0, "neutralization": "INDUSTRY"}
            candidate_id = store.insert_candidate("rank(ts_decay_linear(ts_backfill(scl12_sentiment, 120), 20))", settings, "model:G-1")
            store.transition(candidate_id, "check_pending", {"reason": "simulation_pending"})
            for _ in range(4):
                store.record_event(candidate_id, "simulation_pending", {"location": "/simulations/stale-default"})
            stale_at = (datetime.now(timezone.utc) - timedelta(minutes=45)).isoformat()
            with store.connection() as conn:
                conn.execute(
                    "UPDATE events SET created_at = ? WHERE candidate_id = ? AND event_type = 'simulation_pending'",
                    (stale_at, candidate_id),
                )
            brain = BlockingResumeBrain()
            worker = AlphaWorker(
                store=store,
                ai_client=FailingAI(),
                brain_client=brain,
                policy=SubmissionPolicy(auto_submit=False, min_sharpe=2.69, min_fitness=1.5),
                batch_size=1,
                context=settings,
            )

            with patch.dict(os.environ, {}, clear=True):
                summary = worker.run_once(
                    cycle_plan={
                        "mode": "recover_pending",
                        "scope": settings,
                        "target_candidate_id": candidate_id,
                    }
                )

            candidate = store.get_candidate(candidate_id)
            self.assertEqual(brain.resume_calls, 0)
            self.assertEqual(candidate["status"], "failed")
            self.assertEqual(summary["failed"], 1)
            self.assertEqual(summary["stale_pending_failed"], 1)

    def test_worker_recover_pending_uses_cycle_plan_target_before_newer_pending(self):
        class FailingAI:
            def generate_candidates(self, batch_size, context):
                raise AssertionError("recover_pending should not generate fresh candidates")

        class RecoveringBrain(LocalBrainClient):
            def __init__(self):
                super().__init__()
                self.resumed_locations = []

            def resume_simulation(self, location):
                self.resumed_locations.append(location)
                return SimulationResult(
                    alpha_id=f"RECOVERED-{location.rsplit('/', 1)[-1]}",
                    metrics={"sharpe": 2.9, "fitness": 1.7, "turnover": 0.2, "returns": 0.09},
                    checks={
                        "LOW_SHARPE": {"status": "PASS", "value": 2.9, "limit": 2.69},
                        "LOW_FITNESS": {"status": "PASS", "value": 1.7, "limit": 1.5},
                        "LOW_TURNOVER": {"status": "PASS", "value": 0.2, "limit": 0.01},
                        "HIGH_TURNOVER": {"status": "PASS", "value": 0.2, "limit": 0.7},
                        "SELF_CORRELATION": {"status": "PASS"},
                        "PROD_CORRELATION": {"status": "PASS"},
                        "DATA_DIVERSITY": {"status": "PASS"},
                        "REGULAR_SUBMISSION": {"status": "PASS"},
                    },
                )

        with tempfile.TemporaryDirectory() as tmp:
            store = AlphaStore(Path(tmp) / "alpha.db")
            store.init()
            settings = {"region": "USA", "universe": "TOP3000", "delay": 1, "neutralization": "INDUSTRY"}
            old_id = store.insert_candidate("rank(old_pending)", settings, "model:G-1")
            store.transition(old_id, "check_pending", {"reason": "simulation_pending"})
            store.record_event(old_id, "simulation_pending", {"location": "/simulations/old"})
            newer_id = store.insert_candidate("rank(new_pending)", settings, "model:G-1")
            store.transition(newer_id, "check_pending", {"reason": "simulation_pending"})
            store.record_event(newer_id, "simulation_pending", {"location": "/simulations/new"})
            brain = RecoveringBrain()
            worker = AlphaWorker(
                store=store,
                ai_client=FailingAI(),
                brain_client=brain,
                policy=SubmissionPolicy(auto_submit=False, min_sharpe=2.69, min_fitness=1.5),
                batch_size=1,
                context=settings,
            )

            summary = worker.run_once(
                cycle_plan={
                    "mode": "recover_pending",
                    "scope": settings,
                    "target_candidate_id": old_id,
                }
            )

            self.assertEqual(brain.resumed_locations, ["/simulations/old"])
            self.assertEqual(store.get_candidate(old_id)["status"], "approved")
            self.assertEqual(store.get_candidate(newer_id)["status"], "check_pending")
            self.assertEqual(summary["approved"], 1)

    def test_worker_does_not_recheck_pending_simulation_during_explore_cycle(self):
        class FreshAI:
            def generate_candidates(self, batch_size, context):
                return [
                    CandidateSpec(
                        "rank(mdl_mock_score)",
                        settings={"region": "USA", "universe": "TOP3000", "delay": 1, "neutralization": "INDUSTRY"},
                        source="model:G-1",
                    )
                ]

        class BlockingResumeBrain(LocalBrainClient):
            def __init__(self):
                super().__init__()
                self.resume_calls = 0

            def resume_simulation(self, location):
                self.resume_calls += 1
                raise AssertionError("explore cycles must not block on pending simulation recovery")

        with tempfile.TemporaryDirectory() as tmp:
            store = AlphaStore(Path(tmp) / "alpha.db")
            store.init()
            pending_id = store.insert_candidate(
                "rank(old_pending)",
                {"region": "USA", "universe": "TOP3000", "delay": 1, "neutralization": "INDUSTRY"},
                "model:G-1",
            )
            store.transition(pending_id, "check_pending", {"reason": "simulation_pending"})
            store.record_event(pending_id, "simulation_pending", {"location": "/simulations/slow", "recheck": True})
            brain = BlockingResumeBrain()
            worker = AlphaWorker(
                store=store,
                ai_client=FreshAI(),
                brain_client=brain,
                policy=SubmissionPolicy(auto_submit=False),
                batch_size=1,
                context={"region": "USA", "universe": "TOP3000", "delay": 1, "neutralization": "INDUSTRY"},
            )

            summary = worker.run_once(
                cycle_plan={
                    "mode": "explore",
                    "scope": {"region": "USA", "universe": "TOP3000", "delay": 1, "neutralization": "INDUSTRY"},
                }
            )

            self.assertEqual(brain.resume_calls, 0)
            self.assertEqual(summary["generated"], 1)
            self.assertEqual(store.get_candidate(pending_id)["status"], "check_pending")

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

    def test_worker_enforces_profile_forbidden_analyst_family_before_simulation(self):
        class CountingBrain(LocalBrainClient):
            def __init__(self):
                super().__init__()
                self.simulation_calls = 0

            def simulate(self, expression, settings):
                self.simulation_calls += 1
                return super().simulate(expression, settings)

            def simulate_many(self, items):
                self.simulation_calls += len(items)
                return super().simulate_many(items)

        class ProfileAI:
            def generate_candidates(self, batch_size, context):
                return [
                    CandidateSpec(
                        "rank(anl10_bad_signal)",
                        settings={"region": "USA", "universe": "TOP500", "delay": 0},
                        source="model:G-2",
                        metadata={
                            "model_profile": "G-2",
                            "profile_guidance": {
                                "field_family": "USA/D0/NEWS only; avoid all analyst fields anywhere in formula",
                                "avoid": ["all analyst fields anywhere in formula", "anl10_*", "actual_update_*"],
                            },
                        },
                    )
                ]

        class ProfileWorker(AlphaWorker):
            def _build_ai_context(self):
                return {
                    "region": "USA",
                    "universe": "TOP500",
                    "delay": 0,
                    "research_context": {
                        "generation_policy": {"auxiliary_fields_must_not_be_primary": True},
                        "datafields": {
                            "available": True,
                            "field_ids": ["anl10_bad_signal"],
                            "field_types": {"anl10_bad_signal": "MATRIX"},
                            "fields": [
                                {
                                    "id": "anl10_bad_signal",
                                    "type": "MATRIX",
                                    "dataset_id": "analyst10",
                                    "category": "Analyst",
                                }
                            ],
                        },
                        "syntax_constraints": {"auxiliary_only_fields": []},
                    },
                }

        with tempfile.TemporaryDirectory() as tmp:
            store = AlphaStore(Path(tmp) / "alpha.db")
            store.init()
            brain = CountingBrain()
            worker = ProfileWorker(
                store=store,
                ai_client=ProfileAI(),
                brain_client=brain,
                policy=SubmissionPolicy(auto_submit=False),
                batch_size=1,
                context={"region": "USA", "universe": "TOP500", "delay": 0},
            )

            summary = worker.run_once()

            candidate = store.list_candidates()[0]
            events = store.events_for_candidate(candidate["id"])
            self.assertEqual(summary["failed"], 1)
            self.assertEqual(brain.simulation_calls, 0)
            self.assertEqual(candidate["status"], "failed")
            self.assertTrue(any("PROFILE_FORBIDDEN_FIELD_FAMILY:ANALYST" in event["metadata_json"] for event in events))

    def test_worker_enforces_profile_required_analyst_family_before_simulation(self):
        class CountingBrain(LocalBrainClient):
            def __init__(self):
                super().__init__()
                self.simulation_calls = 0

            def simulate(self, expression, settings):
                self.simulation_calls += 1
                return super().simulate(expression, settings)

            def simulate_many(self, items):
                self.simulation_calls += len(items)
                return super().simulate_many(items)

        class ProfileAI:
            def generate_candidates(self, batch_size, context):
                return [
                    CandidateSpec(
                        "rank(fnd6_at)",
                        settings={"region": "USA", "universe": "TOP500", "delay": 0},
                        source="model:G-1",
                        metadata={
                            "model_profile": "G-1",
                            "profile_guidance": {
                                "field_family": "ANALYST primary only, using fresh field_scout names/buckets",
                                "avoid": ["non-analyst primaries", "news fields", "fundamental fields"],
                            },
                        },
                    )
                ]

        class ProfileWorker(AlphaWorker):
            def _build_ai_context(self):
                return {
                    "region": "USA",
                    "universe": "TOP500",
                    "delay": 0,
                    "research_context": {
                        "generation_policy": {"auxiliary_fields_must_not_be_primary": True},
                        "datafields": {
                            "available": True,
                            "field_ids": ["fnd6_at", "anl10_bpsff_1924"],
                            "field_types": {"fnd6_at": "MATRIX", "anl10_bpsff_1924": "MATRIX"},
                            "fields": [
                                {
                                    "id": "fnd6_at",
                                    "type": "MATRIX",
                                    "dataset_id": "fundamental6",
                                    "category": "Fundamental",
                                },
                                {
                                    "id": "anl10_bpsff_1924",
                                    "type": "MATRIX",
                                    "dataset_id": "analyst10",
                                    "category": "Analyst",
                                },
                            ],
                        },
                        "syntax_constraints": {"auxiliary_only_fields": []},
                    },
                }

        with tempfile.TemporaryDirectory() as tmp:
            store = AlphaStore(Path(tmp) / "alpha.db")
            store.init()
            brain = CountingBrain()
            worker = ProfileWorker(
                store=store,
                ai_client=ProfileAI(),
                brain_client=brain,
                policy=SubmissionPolicy(auto_submit=False),
                batch_size=1,
                context={"region": "USA", "universe": "TOP500", "delay": 0},
            )

            summary = worker.run_once()

            candidate = store.list_candidates()[0]
            events = store.events_for_candidate(candidate["id"])
            self.assertEqual(summary["failed"], 1)
            self.assertEqual(brain.simulation_calls, 0)
            self.assertEqual(candidate["status"], "failed")
            self.assertTrue(any("PROFILE_REQUIRED_FIELD_FAMILY:ANALYST" in event["metadata_json"] for event in events))

    def test_worker_enforces_profile_required_news_family_before_simulation(self):
        class CountingBrain(LocalBrainClient):
            def __init__(self):
                super().__init__()
                self.simulation_calls = 0

            def simulate(self, expression, settings):
                self.simulation_calls += 1
                return super().simulate(expression, settings)

            def simulate_many(self, items):
                self.simulation_calls += len(items)
                return super().simulate_many(items)

        class ProfileAI:
            def generate_candidates(self, batch_size, context):
                return [
                    CandidateSpec(
                        "rank(vec_avg(insd1_tradesignificance))",
                        settings={"region": "USA", "universe": "TOP500", "delay": 0},
                        source="model:G-2",
                        metadata={
                            "model_profile": "G-2",
                            "profile_guidance": {
                                "field_family": "NEWS-only, routed to surfaced top_primary NEWS fields.",
                                "avoid": ["all non-news primaries"],
                            },
                        },
                    )
                ]

        class ProfileWorker(AlphaWorker):
            def _build_ai_context(self):
                return {
                    "region": "USA",
                    "universe": "TOP500",
                    "delay": 0,
                    "research_context": {
                        "generation_policy": {"auxiliary_fields_must_not_be_primary": True},
                        "datafields": {
                            "available": True,
                            "field_ids": ["insd1_tradesignificance", "nws73_djnsubject"],
                            "field_types": {"insd1_tradesignificance": "VECTOR", "nws73_djnsubject": "VECTOR"},
                            "fields": [
                                {
                                    "id": "insd1_tradesignificance",
                                    "type": "VECTOR",
                                    "dataset_id": "insiders1",
                                    "category": "Insiders",
                                },
                                {
                                    "id": "nws73_djnsubject",
                                    "type": "VECTOR",
                                    "dataset_id": "news73",
                                    "category": "News",
                                },
                            ],
                        },
                        "syntax_constraints": {"auxiliary_only_fields": []},
                    },
                }

        with tempfile.TemporaryDirectory() as tmp:
            store = AlphaStore(Path(tmp) / "alpha.db")
            store.init()
            brain = CountingBrain()
            worker = ProfileWorker(
                store=store,
                ai_client=ProfileAI(),
                brain_client=brain,
                policy=SubmissionPolicy(auto_submit=False),
                batch_size=1,
                context={"region": "USA", "universe": "TOP500", "delay": 0},
            )

            summary = worker.run_once()

            candidate = store.list_candidates()[0]
            events = store.events_for_candidate(candidate["id"])
            self.assertEqual(summary["failed"], 1)
            self.assertEqual(brain.simulation_calls, 0)
            self.assertEqual(candidate["status"], "failed")
            self.assertTrue(any("PROFILE_REQUIRED_FIELD_FAMILY:NEWS" in event["metadata_json"] for event in events))

    def test_worker_does_not_treat_specific_analyst_avoid_pattern_as_all_analyst_ban(self):
        class CountingBrain(LocalBrainClient):
            def __init__(self):
                super().__init__()
                self.simulation_calls = 0

            def simulate(self, expression, settings):
                self.simulation_calls += 1
                return super().simulate(expression, settings)

            def simulate_many(self, items):
                self.simulation_calls += len(items)
                return super().simulate_many(items)

        class ProfileAI:
            def generate_candidates(self, batch_size, context):
                return [
                    CandidateSpec(
                        "rank(vec_avg(anl4_fs_detail_estimate_basic_af_v4_estimate))",
                        settings={"region": "USA", "universe": "TOP500", "delay": 0},
                        source="model:G-1",
                        metadata={
                            "model_profile": "G-1",
                            "profile_guidance": {
                                "field_family": "Analyst4 remains the dominant family and only anchor.",
                                "avoid": ["analyst10 as primary family", "anl10_*"],
                            },
                        },
                    )
                ]

        class ProfileWorker(AlphaWorker):
            def _build_ai_context(self):
                return {
                    "region": "USA",
                    "universe": "TOP500",
                    "delay": 0,
                    "research_context": {
                        "generation_policy": {"auxiliary_fields_must_not_be_primary": True},
                        "datafields": {
                            "available": True,
                            "field_ids": ["anl4_fs_detail_estimate_basic_af_v4_estimate"],
                            "field_types": {"anl4_fs_detail_estimate_basic_af_v4_estimate": "VECTOR"},
                            "fields": [
                                {
                                    "id": "anl4_fs_detail_estimate_basic_af_v4_estimate",
                                    "type": "VECTOR",
                                    "dataset_id": "analyst4",
                                    "category": "Analyst",
                                }
                            ],
                        },
                        "syntax_constraints": {"auxiliary_only_fields": []},
                    },
                }

        with tempfile.TemporaryDirectory() as tmp:
            store = AlphaStore(Path(tmp) / "alpha.db")
            store.init()
            brain = CountingBrain()
            worker = ProfileWorker(
                store=store,
                ai_client=ProfileAI(),
                brain_client=brain,
                policy=SubmissionPolicy(auto_submit=False),
                batch_size=1,
                context={"region": "USA", "universe": "TOP500", "delay": 0},
            )

            summary = worker.run_once()

            candidate = store.list_candidates()[0]
            events = store.events_for_candidate(candidate["id"])
            self.assertEqual(summary["generated"], 1)
            self.assertEqual(brain.simulation_calls, 1)
            self.assertFalse(any("PROFILE_FORBIDDEN_FIELD_FAMILY:ANALYST" in event["metadata_json"] for event in events))

    def test_worker_does_not_treat_non_analyst_primary_avoid_as_analyst_ban(self):
        class CountingBrain(LocalBrainClient):
            def __init__(self):
                super().__init__()
                self.simulation_calls = 0

            def simulate(self, expression, settings):
                self.simulation_calls += 1
                return super().simulate(expression, settings)

            def simulate_many(self, items):
                self.simulation_calls += len(items)
                return super().simulate_many(items)

        class ProfileAI:
            def generate_candidates(self, batch_size, context):
                return [
                    CandidateSpec(
                        "rank(ts_rank(vec_avg(anl4_fsdtlestmtsafv4_item), 63))",
                        settings={"region": "USA", "universe": "TOP500", "delay": 0},
                        source="model:G-1",
                        metadata={
                            "model_profile": "G-1",
                            "profile_guidance": {
                                "field_family": "ANALYST primary only, using fresh field_scout names/buckets",
                                "avoid": ["non-ANALYST primaries", "news fields", "fundamental fields"],
                            },
                        },
                    )
                ]

        class ProfileWorker(AlphaWorker):
            def _build_ai_context(self):
                return {
                    "region": "USA",
                    "universe": "TOP500",
                    "delay": 0,
                    "research_context": {
                        "generation_policy": {"auxiliary_fields_must_not_be_primary": True},
                        "datafields": {
                            "available": True,
                            "field_ids": ["anl4_fsdtlestmtsafv4_item"],
                            "field_types": {"anl4_fsdtlestmtsafv4_item": "VECTOR"},
                            "fields": [
                                {
                                    "id": "anl4_fsdtlestmtsafv4_item",
                                    "type": "VECTOR",
                                    "dataset_id": "analyst4",
                                    "category": "Analyst",
                                }
                            ],
                        },
                        "syntax_constraints": {"auxiliary_only_fields": []},
                    },
                }

        with tempfile.TemporaryDirectory() as tmp:
            store = AlphaStore(Path(tmp) / "alpha.db")
            store.init()
            brain = CountingBrain()
            worker = ProfileWorker(
                store=store,
                ai_client=ProfileAI(),
                brain_client=brain,
                policy=SubmissionPolicy(auto_submit=False),
                batch_size=1,
                context={"region": "USA", "universe": "TOP500", "delay": 0},
            )

            summary = worker.run_once()

            candidate = store.list_candidates()[0]
            events = store.events_for_candidate(candidate["id"])
            self.assertEqual(summary["generated"], 1)
            self.assertEqual(brain.simulation_calls, 1)
            self.assertFalse(any("PROFILE_FORBIDDEN_FIELD_FAMILY:ANALYST" in event["metadata_json"] for event in events))

    def test_worker_enforces_profile_forbidden_sentiment_family_before_simulation(self):
        class CountingBrain(LocalBrainClient):
            def __init__(self):
                super().__init__()
                self.simulation_calls = 0

            def simulate(self, expression, settings):
                self.simulation_calls += 1
                return super().simulate(expression, settings)

            def simulate_many(self, items):
                self.simulation_calls += len(items)
                return super().simulate_many(items)

        class ProfileAI:
            def generate_candidates(self, batch_size, context):
                return [
                    CandidateSpec(
                        "rank(snt23_5pos_mean_296)",
                        settings={"region": "USA", "universe": "TOP500", "delay": 0},
                        source="model:G-2",
                        metadata={
                            "model_profile": "G-2",
                            "profile_guidance": {
                                "field_family": "Non-sentiment only across unlit ANALYST/NEWS/FUNDAMENTAL/RISK/OTHER surfaces.",
                                "avoid": ["all sentiment23 fields as primary or helper legs", "snt23_*"],
                            },
                        },
                    )
                ]

        class ProfileWorker(AlphaWorker):
            def _build_ai_context(self):
                return {
                    "region": "USA",
                    "universe": "TOP500",
                    "delay": 0,
                    "research_context": {
                        "generation_policy": {"auxiliary_fields_must_not_be_primary": True},
                        "datafields": {
                            "available": True,
                            "field_ids": ["snt23_5pos_mean_296"],
                            "field_types": {"snt23_5pos_mean_296": "MATRIX"},
                            "fields": [
                                {
                                    "id": "snt23_5pos_mean_296",
                                    "type": "MATRIX",
                                    "dataset_id": "sentiment23",
                                    "category": "Sentiment",
                                }
                            ],
                        },
                        "syntax_constraints": {"auxiliary_only_fields": []},
                    },
                }

        with tempfile.TemporaryDirectory() as tmp:
            store = AlphaStore(Path(tmp) / "alpha.db")
            store.init()
            brain = CountingBrain()
            worker = ProfileWorker(
                store=store,
                ai_client=ProfileAI(),
                brain_client=brain,
                policy=SubmissionPolicy(auto_submit=False),
                batch_size=1,
                context={"region": "USA", "universe": "TOP500", "delay": 0},
            )

            summary = worker.run_once()

            candidate = store.list_candidates()[0]
            events = store.events_for_candidate(candidate["id"])
            self.assertEqual(summary["failed"], 1)
            self.assertEqual(brain.simulation_calls, 0)
            self.assertEqual(candidate["status"], "failed")
            self.assertTrue(any("PROFILE_FORBIDDEN_FIELD_FAMILY:SENTIMENT" in event["metadata_json"] for event in events))

    def test_worker_skips_candidate_when_expression_structure_was_already_tested(self):
        class CountingBrain(LocalBrainClient):
            def __init__(self):
                super().__init__()
                self.simulation_calls = 0

            def simulate(self, expression, settings):
                self.simulation_calls += 1
                return super().simulate(expression, settings)

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

            def simulate(self, expression, settings):
                self.simulation_calls += 1
                return super().simulate(expression, settings)

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

    def test_worker_marks_full_low_quality_batch_with_quality_stop_loss(self):
        class WeakBatchBrain(LocalBrainClient):
            def simulate_many(self, items):
                return [
                    SimulationResult(
                        alpha_id=f"WEAK{idx}",
                        metrics={
                            "sharpe": 0.24,
                            "fitness": 0.05,
                            "turnover": 0.2,
                            "returns": 0.01,
                            "drawdown": 0.2,
                        },
                        checks={
                            "LOW_SHARPE": {"status": "FAIL", "value": 0.24, "limit": 2.69},
                            "LOW_FITNESS": {"status": "FAIL", "value": 0.05, "limit": 1.5},
                            "LOW_TURNOVER": {"status": "PASS", "value": 0.2, "limit": 0.01},
                            "HIGH_TURNOVER": {"status": "PASS", "value": 0.2, "limit": 0.7},
                            "CONCENTRATED_WEIGHT": {"status": "PASS"},
                            "LOW_SUB_UNIVERSE_SHARPE": {"status": "FAIL", "value": -0.1, "limit": 0.0},
                            "LOW_2Y_SHARPE": {"status": "FAIL", "value": 0.0, "limit": 2.69},
                        },
                    )
                    for idx, _ in enumerate(items)
                ]

        with tempfile.TemporaryDirectory() as tmp:
            store = AlphaStore(Path(tmp) / "alpha.db")
            store.init()
            worker = AlphaWorker(
                store=store,
                ai_client=LocalAIClient(
                    expressions=[
                        "rank(mdl_mock_score)",
                        "rank(mdl_alt_score)",
                        "rank(mdl_other_score)",
                        "rank(add(mdl_mock_score,mdl_alt_score))",
                    ]
                ),
                brain_client=WeakBatchBrain(),
                policy=SubmissionPolicy(auto_submit=False),
                batch_size=4,
                context={"region": "USA", "universe": "TOP3000", "delay": 0},
            )

            summary = worker.run_once()

        self.assertEqual(summary["generated"], 4)
        self.assertEqual(summary["failed"], 4)
        self.assertEqual(summary["quality_stop_loss"], 1)
        self.assertEqual(summary["quality_stop_reason"], "bad_full_batch")
        self.assertEqual(summary["quality_stop_best_sharpe"], 0.24)

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

    def test_worker_refills_after_duplicate_filtering_before_simulation(self):
        class RefillAI:
            def __init__(self):
                self.calls = []

            def generate_candidates(self, batch_size, context):
                self.calls.append(batch_size)
                if len(self.calls) == 1:
                    return [
                        CandidateSpec("rank( mdl_mock_score )"),
                        CandidateSpec("rank(mdl_alt_score)"),
                        CandidateSpec("rank(mdl_other_score)"),
                        CandidateSpec("group_rank(ts_rank(divide(mdl_mock_score, cap), 22), industry)"),
                        CandidateSpec("rank(ts_mean(mdl_alt_score, 20))"),
                        CandidateSpec("rank( mdl_mock_score )"),
                        CandidateSpec("rank(ts_rank(mdl_other_score, 63))"),
                        CandidateSpec("rank( mdl_mock_score )"),
                    ]
                return [
                    CandidateSpec("rank(ts_delta(mdl_mock_score, 5))"),
                    CandidateSpec("rank(ts_delta(mdl_alt_score, 5))"),
                    CandidateSpec("rank(ts_delta(mdl_other_score, 5))"),
                ][:batch_size]

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
            store.insert_candidate("rank(mdl_mock_score)", settings, "seed")
            ai = RefillAI()
            brain = CountingBrain()
            worker = AlphaWorker(
                store=store,
                ai_client=ai,
                brain_client=brain,
                policy=SubmissionPolicy(auto_submit=False),
                batch_size=8,
                context={"region": "USA", "universe": "TOP3000", "delay": 1, "neutralization": "INDUSTRY"},
                cycle_plan={"mode": "explore", "reason": "test_post_dedup_refill"},
            )

            summary = worker.run_once()

            events = store.events_for_candidate(None)
            event_types = [event["event_type"] for event in events]
            stage_events = [
                json.loads(event["metadata_json"])
                for event in events
                if event["event_type"] == "cycle_stage"
            ]
            self.assertEqual(ai.calls, [8, 3])
            self.assertEqual(summary["generated"], 8)
            self.assertEqual(summary["skipped"], 3)
            self.assertEqual(len(brain.batch_calls), 1)
            self.assertEqual(len(brain.batch_calls[0]), 8)
            self.assertIn("post_dedup_refill", [event["stage"] for event in stage_events])
            self.assertTrue(any(event_type == "duplicate_candidate_skipped" for event_type in event_types))

    def test_worker_records_optional_cycle_plan_and_outcome(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = AlphaStore(Path(tmp) / "alpha.db")
            store.init()
            cycle_plan = {
                "mode": "explore",
                "scope": {"region": "USA", "universe": "TOP3000", "delay": 1, "neutralization": "INDUSTRY"},
                "target_candidate_id": None,
                "reason": "test_plan",
            }
            worker = AlphaWorker(
                store=store,
                ai_client=LocalAIClient(),
                brain_client=LocalBrainClient(),
                policy=SubmissionPolicy(auto_submit=False),
                batch_size=1,
                context={"region": "USA", "universe": "TOP3000", "delay": 1, "neutralization": "INDUSTRY"},
                cycle_plan=cycle_plan,
            )

            summary = worker.run_once()

            events = store.events_for_candidate(None)
            event_types = [event["event_type"] for event in events]
            self.assertIn("cycle_plan", event_types)
            self.assertIn("cycle_outcome", event_types)
            self.assertEqual(summary["generated"], 1)

    def test_worker_records_cycle_stage_heartbeats(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = AlphaStore(Path(tmp) / "alpha.db")
            store.init()
            cycle_plan = {
                "mode": "explore",
                "scope": {"region": "USA", "universe": "TOP3000", "delay": 1, "neutralization": "INDUSTRY"},
                "target_candidate_id": None,
                "reason": "test_plan",
            }
            worker = AlphaWorker(
                store=store,
                ai_client=LocalAIClient(expressions=["rank(mdl_mock_score)"]),
                brain_client=LocalBrainClient(),
                policy=SubmissionPolicy(auto_submit=False),
                batch_size=1,
                context={"region": "USA", "universe": "TOP3000", "delay": 1, "neutralization": "INDUSTRY"},
                cycle_plan=cycle_plan,
            )

            worker.run_once()

            stage_events = [
                json.loads(event["metadata_json"])
                for event in store.events_for_candidate(None)
                if event["event_type"] == "cycle_stage"
            ]
            stages = [event["stage"] for event in stage_events]
            self.assertIn("building_context", stages)
            self.assertIn("context_built", stages)
            self.assertIn("selecting_candidates", stages)
            self.assertIn("simulation_started", stages)
            self.assertIn("cycle_finished", stages)
            self.assertTrue(all(event["cycle_mode"] == "explore" for event in stage_events))

    def test_worker_times_out_slow_ai_generation_and_stops_before_fallback(self):
        class SlowAI:
            def generate_candidates(self, batch_size, context):
                time.sleep(0.1)
                return [CandidateSpec("rank(mdl_mock_score)")]

        with tempfile.TemporaryDirectory() as tmp:
            store = AlphaStore(Path(tmp) / "alpha.db")
            store.init()
            worker = AlphaWorker(
                store=store,
                ai_client=SlowAI(),
                brain_client=LocalBrainClient(),
                policy=SubmissionPolicy(auto_submit=False, max_retries=1),
                batch_size=1,
                context={"region": "USA", "universe": "TOP3000", "delay": 1, "neutralization": "INDUSTRY"},
                cycle_plan={"mode": "explore", "reason": "test_timeout"},
            )

            with patch.dict("os.environ", {"AI_GENERATION_STAGE_TIMEOUT_SECONDS": "0.01"}):
                summary = worker.run_once()

            events = store.events_for_candidate(None)
            event_types = [event["event_type"] for event in events]
            stage_events = [
                json.loads(event["metadata_json"])
                for event in events
                if event["event_type"] == "cycle_stage"
            ]
            stages = [event["stage"] for event in stage_events]
            self.assertIn("ai_generation_started", stages)
            self.assertIn("ai_generation_timeout", stages)
            self.assertNotIn("deterministic_generation_fallback", event_types)
            self.assertEqual(summary["ai_generation_timeout"], 1)
            self.assertEqual(summary["generated"], 0)
            self.assertEqual(summary["failed"], 1)

    def test_worker_classifies_ai_dns_failure_as_network_blocked(self):
        class DnsFailingAI:
            def generate_candidates(self, batch_size, context):
                raise RuntimeError(
                    "all multi-model generation paths failed: "
                    "G-1:AI candidate generation failed: AI request failed: "
                    "<urlopen error [Errno -2] Name or service not known>"
                )

        with tempfile.TemporaryDirectory() as tmp:
            store = AlphaStore(Path(tmp) / "alpha.db")
            store.init()
            worker = AlphaWorker(
                store=store,
                ai_client=DnsFailingAI(),
                brain_client=LocalBrainClient(),
                policy=SubmissionPolicy(auto_submit=False, max_retries=1),
                batch_size=8,
                context={"region": "MEA", "universe": "TOP300", "delay": 1, "neutralization": "INDUSTRY"},
                cycle_plan={"mode": "setting_sweep", "reason": "approved_candidate_may_have_setting_upside"},
            )

            summary = worker.run_once()

            self.assertEqual(summary["generated"], 0)
            self.assertEqual(summary["failed"], 1)
            self.assertEqual(summary["ai_network_blocked"], 1)
            events = store.events_for_candidate(None)
            ai_errors = [event for event in events if event["event_type"] == "ai_generation_error"]
            self.assertEqual(len(ai_errors), 1)
            metadata = json.loads(ai_errors[0]["metadata_json"])
            self.assertEqual(metadata["reason"], "ai_network_blocked")
            self.assertEqual(metadata["non_retryable"], True)

    def test_worker_uses_partial_candidates_when_refill_times_out(self):
        class PartialThenSlowAI:
            def __init__(self):
                self.last_partial_candidates = []

            def generate_candidates(self, batch_size, context):
                self.last_partial_candidates = [CandidateSpec("rank(mdl_mock_score)")]
                time.sleep(0.1)
                return []

        with tempfile.TemporaryDirectory() as tmp:
            store = AlphaStore(Path(tmp) / "alpha.db")
            store.init()
            worker = AlphaWorker(
                store=store,
                ai_client=PartialThenSlowAI(),
                brain_client=LocalBrainClient(),
                policy=SubmissionPolicy(auto_submit=False, max_retries=1),
                batch_size=1,
                context={"region": "USA", "universe": "TOP3000", "delay": 1, "neutralization": "INDUSTRY"},
                cycle_plan={"mode": "explore", "reason": "test_partial_timeout"},
            )

            with patch.dict("os.environ", {"AI_GENERATION_STAGE_TIMEOUT_SECONDS": "0.01"}):
                summary = worker.run_once()

            stage_events = [
                json.loads(event["metadata_json"])
                for event in store.events_for_candidate(None)
                if event["event_type"] == "cycle_stage"
            ]
            stages = [event["stage"] for event in stage_events]
            self.assertIn("ai_generation_partial_timeout", stages)
            self.assertNotIn("ai_generation_timeout", stages)
            self.assertEqual(summary["generated"], 1)
            self.assertNotIn("ai_generation_timeout", summary)

    def test_deterministic_fallback_uses_stabilized_field_scout_primary_before_catalog_tail(self):
        ai_context = {
            "region": "USA",
            "universe": "TOP500",
            "delay": 0,
            "neutralization": "INDUSTRY",
            "research_context": {
                "datafields": {
                    "available": True,
                    "field_ids": ["opinion_score_numeric", "anl10_netsmun_1qf_2750"],
                    "field_types": {
                        "opinion_score_numeric": "MATRIX",
                        "anl10_netsmun_1qf_2750": "MATRIX",
                    },
                    "fields": [
                        {
                            "id": "opinion_score_numeric",
                            "type": "MATRIX",
                            "dataset_id": "social_sent_score",
                            "category": "Other",
                            "usage_constraints": ["requires_turnover_stabilizer"],
                        },
                        {
                            "id": "anl10_netsmun_1qf_2750",
                            "type": "MATRIX",
                            "dataset_id": "analyst10",
                            "category": "Analyst",
                        },
                    ],
                },
                "field_scout": {
                    "active": True,
                    "status": "ready",
                    "top_primary_fields": [
                        {
                            "field": "opinion_score_numeric",
                            "type": "MATRIX",
                            "dataset_id": "social_sent_score",
                            "category": "Other",
                            "primary_policy": "prefer_primary",
                            "usage_constraints": ["requires_turnover_stabilizer"],
                        }
                    ],
                    "top_fields": [
                        {
                            "field": "opinion_score_numeric",
                            "type": "MATRIX",
                            "dataset_id": "social_sent_score",
                            "category": "Other",
                            "primary_policy": "prefer_primary",
                            "usage_constraints": ["requires_turnover_stabilizer"],
                        }
                    ],
                },
            },
        }

        candidates = _deterministic_fallback_candidates(1, ai_context, "explicit_local_fallback")

        self.assertEqual(len(candidates), 1)
        self.assertIn("opinion_score_numeric", candidates[0].expression)
        self.assertIn("ts_backfill", candidates[0].expression)
        self.assertEqual(candidates[0].metadata["fallback_field"], "opinion_score_numeric")

    def test_deterministic_fallback_ignores_retest_primary_fields(self):
        ai_context = {
            "region": "USA",
            "universe": "TOP500",
            "delay": 0,
            "neutralization": "INDUSTRY",
            "research_context": {
                "datafields": {
                    "available": True,
                    "field_ids": ["fnd6_aqi", "vega_spread_1m_avg5d_2"],
                    "field_types": {
                        "fnd6_aqi": "MATRIX",
                        "vega_spread_1m_avg5d_2": "VECTOR",
                    },
                    "fields": [
                        {
                            "id": "vega_spread_1m_avg5d_2",
                            "type": "VECTOR",
                            "dataset_id": "options8",
                            "category": "Option",
                        }
                    ],
                },
                "field_scout": {
                    "active": False,
                    "status": "no_primary_fields",
                    "top_primary_fields": [],
                    "retest_primary_fields": [
                        {
                            "field": "fnd6_aqi",
                            "type": "MATRIX",
                            "dataset_id": "fundamental6",
                            "category": "Fundamental",
                            "primary_policy": "avoid_primary",
                            "dataset_reason": "recent_dataset_failure_cluster",
                            "retest_reason": "field_native_retest_after_dataset_cluster",
                        }
                    ],
                    "top_fields": [],
                },
            },
        }

        candidates = _deterministic_fallback_candidates(1, ai_context, "explicit_local_fallback")

        self.assertEqual(len(candidates), 1)
        self.assertNotIn("fnd6_aqi", candidates[0].expression)
        self.assertIn("vega_spread_1m_avg5d_2", candidates[0].expression)
        self.assertEqual(candidates[0].metadata["fallback_field"], "vega_spread_1m_avg5d_2")

    def test_worker_allows_non_retest_field_when_retest_lane_is_present(self):
        class CountingBrain(LocalBrainClient):
            def __init__(self):
                super().__init__()
                self.simulation_calls = 0

            def simulate(self, expression, settings):
                self.simulation_calls += 1
                return super().simulate(expression, settings)

            def simulate_many(self, items):
                self.simulation_calls += len(items)
                return super().simulate_many(items)

        class RetestContextWorker(AlphaWorker):
            def _build_cycle_ai_context(self, cycle_plan=None):
                return {
                    "region": "USA",
                    "universe": "TOP500",
                    "delay": 0,
                    "neutralization": "INDUSTRY",
                    "research_context": {
                        "datafields": {
                            "available": True,
                            "field_ids": ["fnd6_aqi", "snt_social_value"],
                            "field_types": {
                                "fnd6_aqi": "MATRIX",
                                "snt_social_value": "MATRIX",
                            },
                            "fields": [
                                {"id": "fnd6_aqi", "type": "MATRIX", "dataset_id": "fundamental6", "category": "Fundamental"},
                                {
                                    "id": "snt_social_value",
                                    "type": "MATRIX",
                                    "dataset_id": "sentiment",
                                    "category": "Sentiment",
                                },
                            ],
                        },
                        "field_scout": {
                            "active": False,
                            "status": "no_primary_fields",
                            "top_primary_fields": [],
                            "retest_primary_fields": [
                                {
                                    "field": "fnd6_aqi",
                                    "type": "MATRIX",
                                    "dataset_id": "fundamental6",
                                    "category": "Fundamental",
                                    "primary_policy": "avoid_primary",
                                    "dataset_reason": "recent_dataset_failure_cluster",
                                    "retest_reason": "field_native_retest_after_dataset_cluster",
                                }
                            ],
                            "top_fields": [],
                        },
                        "experiment_plan": {"mode": "explore_new_family"},
                    },
                }

        with tempfile.TemporaryDirectory() as tmp:
            store = AlphaStore(Path(tmp) / "alpha.db")
            store.init()
            brain = CountingBrain()
            worker = RetestContextWorker(
                store=store,
                ai_client=LocalAIClient(expressions=["rank(ts_decay_linear(ts_backfill(snt_social_value, 120), 20))"]),
                brain_client=brain,
                policy=SubmissionPolicy(auto_submit=False),
                batch_size=1,
                context={"region": "USA", "universe": "TOP500", "delay": 0, "neutralization": "INDUSTRY"},
                cycle_plan={"mode": "explore", "reason": "test_retest_required"},
            )

            summary = worker.run_once()

            candidate = store.list_candidates()[0]
            events = store.events_for_candidate(candidate["id"])
            self.assertEqual(summary["generated"], 1)
            self.assertGreater(brain.simulation_calls, 0)
            self.assertFalse(any("FIELD_SCOUT_RETEST_FIELD_REQUIRED:fnd6_aqi" in event["metadata_json"] for event in events))

    def test_deterministic_fallback_skips_historical_duplicates_and_uses_next_field(self):
        settings = {"region": "USA", "universe": "TOP500", "delay": 0, "neutralization": "INDUSTRY"}
        used_field = "used_alpha_signal"
        fresh_field = "fresh_alpha_signal"
        ai_context = {
            **settings,
            "research_context": {
                "datafields": {
                    "available": True,
                    "field_ids": [used_field, fresh_field],
                    "field_types": {used_field: "MATRIX", fresh_field: "MATRIX"},
                    "fields": [
                        {"id": used_field, "type": "MATRIX", "dataset_id": "analyst10", "category": "Analyst"},
                        {"id": fresh_field, "type": "MATRIX", "dataset_id": "analyst11", "category": "Analyst"},
                    ],
                },
                "field_scout": {
                    "active": True,
                    "status": "ready",
                    "top_primary_fields": [
                        {"field": used_field, "type": "MATRIX", "dataset_id": "analyst10", "category": "Analyst"},
                        {"field": fresh_field, "type": "MATRIX", "dataset_id": "analyst11", "category": "Analyst"},
                    ],
                    "top_fields": [],
                },
            },
        }

        with tempfile.TemporaryDirectory() as tmp:
            store = AlphaStore(Path(tmp) / "alpha.db")
            store.init()
            for expression in [
                f"rank(ts_mean({used_field},20))",
                f"rank(ts_delta(ts_mean({used_field},66),22))",
                f"group_rank(ts_rank({used_field},63),industry)",
                f"rank(ts_zscore({used_field},63))",
            ]:
                store.insert_candidate(expression, {**DEFAULT_SETTINGS, **settings}, "deterministic_fallback")

            candidates = _deterministic_fallback_candidates(
                2,
                ai_context,
                "explicit_local_fallback",
                is_duplicate=lambda expression, candidate_settings: store.find_duplicate_candidate(
                    expression, candidate_settings
                )
                is not None,
            )

            self.assertEqual(len(candidates), 2)
            self.assertTrue(all(fresh_field in candidate.expression for candidate in candidates))
    def test_worker_times_out_slow_batch_simulation_and_finishes_cycle(self):
        class SlowBatchBrain(LocalBrainClient):
            def simulate_many(self, items):
                time.sleep(0.1)
                return super().simulate_many(items)

        with tempfile.TemporaryDirectory() as tmp:
            store = AlphaStore(Path(tmp) / "alpha.db")
            store.init()
            worker = AlphaWorker(
                store=store,
                ai_client=LocalAIClient(
                    expressions=[
                        "rank(mdl_mock_score)",
                        "rank(mdl_alt_score)",
                    ]
                ),
                brain_client=SlowBatchBrain(),
                policy=SubmissionPolicy(auto_submit=False, max_retries=1),
                batch_size=2,
                context={"region": "USA", "universe": "TOP3000", "delay": 1, "neutralization": "INDUSTRY"},
                cycle_plan={"mode": "explore", "reason": "test_sim_timeout"},
            )

            with patch.dict("os.environ", {"SIMULATION_STAGE_TIMEOUT_SECONDS": "0.01"}):
                summary = worker.run_once()

            stage_events = [
                json.loads(event["metadata_json"])
                for event in store.events_for_candidate(None)
                if event["event_type"] == "cycle_stage"
            ]
            stages = [event["stage"] for event in stage_events]
            self.assertIn("simulation_batch_started", stages)
            self.assertIn("simulation_batch_timeout", stages)
            self.assertIn("cycle_finished", stages)
            self.assertEqual(summary["generated"], 2)
            self.assertEqual(summary["failed"], 2)

    def test_worker_run_once_accepts_call_level_cycle_plan(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = AlphaStore(Path(tmp) / "alpha.db")
            store.init()
            worker = AlphaWorker(
                store=store,
                ai_client=LocalAIClient(),
                brain_client=LocalBrainClient(),
                policy=SubmissionPolicy(auto_submit=False),
                batch_size=1,
                context={"region": "USA", "universe": "TOP3000", "delay": 1, "neutralization": "INDUSTRY"},
            )

            worker.run_once(cycle_plan={"mode": "explore", "reason": "call_level"})

            events = store.events_for_candidate(None)
            cycle_events = [event for event in events if event["event_type"] == "cycle_plan"]
            self.assertEqual(len(cycle_events), 1)
            self.assertEqual(json.loads(cycle_events[0]["metadata_json"])["reason"], "call_level")


if __name__ == "__main__":
    unittest.main()
