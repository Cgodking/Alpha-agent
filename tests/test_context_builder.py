from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from alpha.context_builder import build_ai_research_context, _recent_preflight_rejections
from alpha.db import AlphaStore
from alpha.research_planner import build_experiment_plan, expression_structure_key, _avoid_list, _quality_budget_for_plan


class ContextBuilderTests(unittest.TestCase):
    def test_quality_budget_reserves_all_normal_explore_slots_for_ai_without_positive_evidence(self):
        budget = _quality_budget_for_plan(
            "explore_new_family",
            8,
            {
                "top_primary_fields": [
                    {
                        "field": f"fresh_signal_{index}",
                        "type": "MATRIX",
                        "dataset_id": f"other{index}",
                        "category": "Other",
                        "score": 1.0 - index / 10,
                        "explored_count": 0,
                        "failed_count": 0,
                        "primary_policy": "prefer_primary",
                    }
                    for index in range(4)
                ]
            },
            {},
        )

        self.assertEqual(budget["quality_budget"]["slots"], {"broad_explore": 8})

    def test_build_experiment_plan_allocates_production_budget_and_probe_recommendations(self):
        plan = build_experiment_plan(
            {
                "candidate_count": 8,
                "failure_reasons": {
                    "HIGH_TURNOVER": 8,
                    "CONCENTRATED_WEIGHT": 8,
                    "LOW_SHARPE": 8,
                    "LOW_FITNESS": 8,
                },
                "best_candidate": {
                    "id": 44,
                    "expression": "normalize(snt23_raw_signal)",
                    "fields": ["snt23_raw_signal"],
                    "metrics": {"sharpe": 0.3, "fitness": 0.04, "turnover": 1.54},
                    "checks": {
                        "LOW_SHARPE": {"status": "FAIL"},
                        "LOW_FITNESS": {"status": "FAIL"},
                        "HIGH_TURNOVER": {"status": "FAIL"},
                        "CONCENTRATED_WEIGHT": {"status": "FAIL"},
                        "LOW_2Y_SHARPE": {"status": "FAIL"},
                    },
                },
                "field_scout": {
                    "active": True,
                    "top_primary_fields": [
                        {
                            "field": "anl10_recovery_signal",
                            "type": "MATRIX",
                            "dataset_id": "analyst10",
                            "category": "Analyst",
                            "score": 0.82,
                            "explored_count": 2,
                            "failed_count": 1,
                            "best_sharpe": 1.22,
                            "best_fitness": 0.44,
                            "tower_status": "unlit",
                            "primary_policy": "prefer_primary",
                        },
                        {
                            "field": "mdl262_fresh_signal",
                            "type": "MATRIX",
                            "dataset_id": "model262",
                            "category": "Model",
                            "score": 0.78,
                            "explored_count": 0,
                            "failed_count": 0,
                            "pyramidMultiplier": 1.8,
                            "tower_status": "unlit",
                            "primary_policy": "prefer_primary",
                        },
                        {
                            "field": "snt23_fresh_signal",
                            "type": "MATRIX",
                            "dataset_id": "sentiment23",
                            "category": "Sentiment",
                            "score": 0.76,
                            "explored_count": 0,
                            "failed_count": 0,
                            "pyramidMultiplier": 1.8,
                            "tower_status": "unlit",
                            "primary_policy": "prefer_primary",
                        },
                    ],
                    "buckets": [
                        {"name": "recovery_candidates", "fields": ["anl10_recovery_signal"]},
                        {"name": "high_opportunity_unexplored", "fields": ["mdl262_fresh_signal", "snt23_fresh_signal"]},
                    ],
                },
                "submitted_field_avoidance": {},
                "lit_tower_avoidance": {},
                "route_efficiency": {},
                "structure_diversity_control": {},
                "observed_quality_thresholds": {},
            },
            {"region": "USA", "universe": "TOP500", "delay": 0, "neutralization": "INDUSTRY"},
            batch_size=8,
        )

        self.assertEqual(plan["mode"], "explore_new_family")
        self.assertEqual(
            plan["quality_budget"]["slots"],
            {"exploit_positive_evidence": 5, "probe_new_fields": 2, "broad_explore": 1},
        )
        self.assertEqual(plan["quality_budget"]["priority"], "production_first")
        self.assertEqual(plan["quality_budget"]["exploit_fields"], ["anl10_recovery_signal"])
        self.assertEqual(
            [item["field"] for item in plan["probe_recommendations"][:2]],
            ["mdl262_fresh_signal", "snt23_fresh_signal"],
        )
        sentiment_probe = plan["probe_recommendations"][1]
        self.assertTrue(sentiment_probe["stabilization_required"])
        joined_templates = " ".join(sentiment_probe["templates"])
        self.assertIn("ts_backfill", joined_templates)
        self.assertIn("winsorize", joined_templates)
        self.assertNotIn("rank(ts_mean(snt23_fresh_signal", joined_templates)

    def test_build_experiment_plan_does_not_allocate_probe_slots_without_probe_fields(self):
        plan = build_experiment_plan(
            {
                "candidate_count": 8,
                "failure_reasons": {"LOW_SHARPE": 8, "LOW_FITNESS": 8},
                "best_candidate": {
                    "id": 1,
                    "expression": "rank(dead_signal)",
                    "fields": ["dead_signal"],
                    "metrics": {"sharpe": 0.1, "fitness": 0.0},
                },
                "field_scout": {
                    "active": False,
                    "top_primary_fields": [],
                    "top_fields": [
                        {
                            "field": "dead_signal",
                            "score": 0.2,
                            "explored_count": 4,
                            "failed_count": 4,
                            "field_reason": "recent_field_failure_cluster",
                            "primary_policy": "avoid_primary",
                        }
                    ],
                },
                "submitted_field_avoidance": {},
                "lit_tower_avoidance": {},
                "route_efficiency": {},
                "structure_diversity_control": {},
                "observed_quality_thresholds": {},
            },
            {"region": "USA", "universe": "TOP500", "delay": 0},
            batch_size=8,
        )

        self.assertEqual(plan["probe_recommendations"], [])
        self.assertEqual(plan["quality_budget"]["slots"], {"broad_explore": 8})

    def test_build_experiment_plan_softens_lit_tower_avoidance_during_production_rescue(self):
        plan = build_experiment_plan(
            {
                "candidate_count": 160,
                "failure_reasons": {"LOW_SHARPE": 80, "LOW_FITNESS": 80},
                "best_candidate": {
                    "id": 7,
                    "expression": "rank(weak_unlit_signal)",
                    "fields": ["weak_unlit_signal"],
                    "metrics": {"sharpe": 0.29, "fitness": 0.09, "turnover": 0.2},
                    "checks": {
                        "LOW_SHARPE": {"status": "FAIL", "value": 0.29, "limit": 2.69},
                        "LOW_FITNESS": {"status": "FAIL", "value": 0.09, "limit": 1.5},
                    },
                },
                "field_scout": {
                    "active": True,
                    "top_primary_fields": [
                        {
                            "field": "anl4_weak_signal",
                            "type": "VECTOR",
                            "dataset_id": "analyst4",
                            "category": "Analyst",
                            "score": 0.95,
                            "explored_count": 0,
                            "failed_count": 0,
                            "tower_status": "unlit",
                            "primary_policy": "prefer_primary",
                        }
                    ],
                    "top_fields": [
                        {
                            "field": "mdl262_predictive_signal",
                            "type": "MATRIX",
                            "dataset_id": "model262",
                            "category": "Model",
                            "score": 0.72,
                            "explored_count": 0,
                            "failed_count": 0,
                            "tower_status": "lit",
                            "primary_policy": "avoid_primary",
                        },
                        {
                            "field": "anl4_weak_signal",
                            "type": "VECTOR",
                            "dataset_id": "analyst4",
                            "category": "Analyst",
                            "score": 0.95,
                            "explored_count": 0,
                            "failed_count": 0,
                            "tower_status": "unlit",
                            "primary_policy": "prefer_primary",
                        },
                    ],
                },
                "submitted_field_avoidance": {},
                "lit_tower_avoidance": {
                    "tower_names": ["USA/D0/MODEL", "USA/D0/PV"],
                    "lit_towers": [{"name": "USA/D0/MODEL", "category": "MODEL"}],
                    "unlit_towers": [{"name": "USA/D0/ANALYST", "category": "ANALYST"}],
                },
                "route_efficiency": {
                    "stop_loss_active": True,
                    "failure_streak": 140,
                    "scanned_candidates": 160,
                    "watchlist_count": 0,
                    "submitable_count": 0,
                    "best_sharpe_ratio": 0.11,
                    "best_fitness_ratio": 0.06,
                },
                "structure_diversity_control": {},
                "observed_quality_thresholds": {
                    "required_sharpe": 2.69,
                    "required_fitness": 1.5,
                    "trusted": True,
                },
            },
            {"region": "USA", "universe": "TOP500", "delay": 0, "neutralization": "INDUSTRY"},
            batch_size=8,
        )

        self.assertTrue(plan["production_rescue"]["active"])
        self.assertNotIn("USA/D0/MODEL", plan["avoid"])
        self.assertIn("lit towers are soft", plan["objective"])
        self.assertEqual(plan["quality_budget"]["slots"], {"probe_new_fields": 2})
        self.assertEqual(plan["probe_recommendations"][0]["field"], "anl4_weak_signal")
        self.assertNotIn(
            "mdl262_predictive_signal",
            [item["field"] for item in plan["probe_recommendations"]],
        )

    def test_build_experiment_plan_prefers_unlit_primary_before_unverified_lit_model_rescue_probe(self):
        plan = build_experiment_plan(
            {
                "candidate_count": 160,
                "failure_reasons": {"LOW_SHARPE": 80, "LOW_FITNESS": 80},
                "best_candidate": {
                    "id": 7,
                    "expression": "rank(weak_signal)",
                    "fields": ["weak_signal"],
                    "metrics": {"sharpe": 0.29, "fitness": 0.09, "turnover": 0.2},
                    "checks": {
                        "LOW_SHARPE": {"status": "FAIL", "value": 0.29, "limit": 2.69},
                        "LOW_FITNESS": {"status": "FAIL", "value": 0.09, "limit": 1.5},
                    },
                },
                "field_scout": {
                    "active": True,
                    "top_primary_fields": [
                        {
                            "field": "oth384_presentation_signal",
                            "type": "VECTOR",
                            "dataset_id": "other384",
                            "category": "Other",
                            "score": 0.80,
                            "explored_count": 0,
                            "failed_count": 0,
                            "tower_status": "unlit",
                            "primary_policy": "prefer_primary",
                        }
                    ],
                    "top_fields": [
                        {
                            "field": "distance_to_default_stddev",
                            "type": "MATRIX",
                            "dataset_id": "model28",
                            "category": "Model",
                            "score": 0.90,
                            "explored_count": 0,
                            "failed_count": 0,
                            "tower_status": "lit",
                            "primary_policy": "avoid_primary",
                        },
                        {
                            "field": "oth384_presentation_signal",
                            "type": "VECTOR",
                            "dataset_id": "other384",
                            "category": "Other",
                            "score": 0.80,
                            "explored_count": 0,
                            "failed_count": 0,
                            "tower_status": "unlit",
                            "primary_policy": "prefer_primary",
                        },
                    ],
                },
                "submitted_field_avoidance": {},
                "lit_tower_avoidance": {
                    "tower_names": ["USA/D0/MODEL", "USA/D0/PV"],
                    "lit_towers": [{"name": "USA/D0/MODEL", "category": "MODEL"}],
                    "unlit_towers": [{"name": "USA/D0/OTHER", "category": "OTHER"}],
                },
                "route_efficiency": {
                    "stop_loss_active": True,
                    "failure_streak": 140,
                    "scanned_candidates": 160,
                    "watchlist_count": 0,
                    "submitable_count": 0,
                    "best_sharpe_ratio": 0.11,
                    "best_fitness_ratio": 0.06,
                },
                "structure_diversity_control": {},
                "observed_quality_thresholds": {
                    "required_sharpe": 2.69,
                    "required_fitness": 1.5,
                    "trusted": True,
                },
            },
            {"region": "USA", "universe": "TOP500", "delay": 0, "neutralization": "INDUSTRY"},
            batch_size=8,
        )

        self.assertTrue(plan["production_rescue"]["active"])
        self.assertEqual(plan["probe_recommendations"][0]["field"], "oth384_presentation_signal")
        self.assertNotEqual(plan["probe_recommendations"][0]["field"], "distance_to_default_stddev")

    def test_build_experiment_plan_excludes_event_fields_from_production_rescue_probes(self):
        plan = build_experiment_plan(
            {
                "candidate_count": 160,
                "failure_reasons": {"LOW_SHARPE": 80, "LOW_FITNESS": 80},
                "best_candidate": {
                    "id": 7,
                    "expression": "rank(weak_news_signal)",
                    "fields": ["weak_news_signal"],
                    "metrics": {"sharpe": 0.29, "fitness": 0.09, "turnover": 0.2},
                    "checks": {
                        "LOW_SHARPE": {"status": "FAIL", "value": 0.29, "limit": 2.69},
                        "LOW_FITNESS": {"status": "FAIL", "value": 0.09, "limit": 1.5},
                    },
                },
                "field_scout": {
                    "active": True,
                    "top_primary_fields": [
                        {
                            "field": "news_item_count_generic",
                            "type": "MATRIX",
                            "dataset_id": "nws7",
                            "category": "News",
                            "score": 0.95,
                            "explored_count": 0,
                            "failed_count": 0,
                            "tower_status": "unlit",
                            "primary_policy": "prefer_primary",
                        }
                    ],
                    "top_fields": [
                        {
                            "field": "nws7_news_freq_2_d0_qerf",
                            "type": "MATRIX",
                            "dataset_id": "nws7",
                            "category": "News",
                            "score": 0.95,
                            "explored_count": 0,
                            "failed_count": 0,
                            "tower_status": "unlit",
                            "primary_policy": "prefer_primary",
                        },
                        {
                            "field": "news_item_count_generic",
                            "type": "MATRIX",
                            "dataset_id": "news7",
                            "category": "News",
                            "score": 0.9,
                            "explored_count": 0,
                            "failed_count": 0,
                            "tower_status": "unlit",
                            "primary_policy": "prefer_primary",
                        },
                    ],
                },
                "submitted_field_avoidance": {},
                "lit_tower_avoidance": {},
                "route_efficiency": {
                    "stop_loss_active": True,
                    "failure_streak": 140,
                    "scanned_candidates": 160,
                    "watchlist_count": 0,
                    "submitable_count": 0,
                    "best_sharpe_ratio": 0.11,
                    "best_fitness_ratio": 0.06,
                },
                "structure_diversity_control": {},
                "observed_quality_thresholds": {
                    "required_sharpe": 2.69,
                    "required_fitness": 1.5,
                    "trusted": True,
                },
            },
            {"region": "USA", "universe": "TOP500", "delay": 0, "neutralization": "INDUSTRY"},
            batch_size=8,
        )

        self.assertFalse(plan["production_rescue"]["active"])
        self.assertEqual(plan["production_rescue"]["reason"], "no_safe_probe_recommendations")
        self.assertEqual(plan["probe_recommendations"], [])
        self.assertEqual(plan["quality_budget"]["slots"], {"broad_explore": 8})
        self.assertIn("no safe probe field", plan["quality_budget"]["rationale"])
        self.assertNotIn("Production rescue is active", plan["objective"])
        self.assertNotIn("Prefer production-tested probe motifs", plan["objective"])

    def test_build_experiment_plan_blocks_overused_production_rescue_probe_templates(self):
        overused_expressions = [
            "group_rank(ts_rank(winsorize(ts_backfill(vec_avg(lending_fee_bid_rate), 120), std=4), 63), industry)",
            "group_rank(ts_rank(divide(winsorize(ts_backfill(vec_avg(lending_fee_bid_rate), 120), std=4), cap), 63), industry)",
            "rank(ts_decay_linear(ts_backfill(vec_avg(lending_fee_bid_rate), 120), 20))",
            "rank(multiply(-1, ts_rank(winsorize(ts_backfill(vec_avg(lending_fee_bid_rate), 120), std=4), 33)))",
        ]
        plan = build_experiment_plan(
            {
                "candidate_count": 160,
                "failure_reasons": {"LOW_SHARPE": 80, "LOW_FITNESS": 80},
                "best_candidate": {
                    "id": 7,
                    "expression": "rank(weak_risk_signal)",
                    "fields": ["weak_risk_signal"],
                    "metrics": {"sharpe": 0.29, "fitness": 0.09, "turnover": 0.2},
                },
                "field_scout": {
                    "active": True,
                    "top_fields": [
                        {
                            "field": "lending_fee_bid_rate",
                            "type": "VECTOR",
                            "dataset_id": "risk60",
                            "category": "Risk",
                            "score": 0.95,
                            "explored_count": 0,
                            "failed_count": 0,
                            "tower_status": "unlit",
                            "primary_policy": "prefer_primary",
                        }
                    ],
                    "top_primary_fields": [],
                },
                "lit_tower_avoidance": {},
                "route_efficiency": {
                    "stop_loss_active": True,
                    "failure_streak": 140,
                    "scanned_candidates": 160,
                    "watchlist_count": 0,
                    "submitable_count": 0,
                    "best_sharpe_ratio": 0.11,
                    "best_fitness_ratio": 0.06,
                },
                "structure_diversity_control": {
                    "active": True,
                    "overused_structures": [
                        {
                            "structure_key": expression_structure_key(expression),
                            "failure_rate": 1.0,
                            "best_quality_score": 0.0,
                        }
                        for expression in overused_expressions
                    ],
                },
                "submitted_field_avoidance": {},
                "observed_quality_thresholds": {
                    "required_sharpe": 2.69,
                    "required_fitness": 1.5,
                    "trusted": True,
                },
            },
            {"region": "USA", "universe": "TOP500", "delay": 0, "neutralization": "INDUSTRY"},
            batch_size=8,
        )

        self.assertFalse(plan["production_rescue"]["active"])
        self.assertEqual(plan["probe_recommendations"], [])
        self.assertEqual(plan["quality_budget"]["slots"], {"broad_explore": 8})
        self.assertNotIn("Production rescue is active", plan["objective"])
        self.assertNotIn("Prefer production-tested probe motifs", plan["objective"])

    def test_build_experiment_plan_disables_rescue_when_only_unverified_model_probes_remain(self):
        plan = build_experiment_plan(
            {
                "candidate_count": 160,
                "failure_reasons": {"LOW_SHARPE": 80, "LOW_FITNESS": 80},
                "best_candidate": {
                    "id": 7,
                    "expression": "rank(weak_model_signal)",
                    "fields": ["weak_model_signal"],
                    "metrics": {"sharpe": 0.29, "fitness": 0.09, "turnover": 0.2},
                },
                "field_scout": {
                    "active": True,
                    "top_fields": [
                        {
                            "field": "mdl262_fresh_signal",
                            "type": "MATRIX",
                            "dataset_id": "model262",
                            "category": "Model",
                            "score": 0.95,
                            "explored_count": 0,
                            "failed_count": 0,
                            "tower_status": "unlit",
                            "primary_policy": "prefer_primary",
                        }
                    ],
                    "top_primary_fields": [],
                },
                "lit_tower_avoidance": {},
                "route_efficiency": {
                    "stop_loss_active": True,
                    "failure_streak": 140,
                    "scanned_candidates": 160,
                    "watchlist_count": 0,
                    "submitable_count": 0,
                    "best_sharpe_ratio": 0.11,
                    "best_fitness_ratio": 0.06,
                },
                "structure_diversity_control": {},
                "submitted_field_avoidance": {},
                "observed_quality_thresholds": {
                    "required_sharpe": 2.69,
                    "required_fitness": 1.5,
                    "trusted": True,
                },
            },
            {"region": "USA", "universe": "TOP500", "delay": 0, "neutralization": "INDUSTRY"},
            batch_size=8,
        )

        self.assertFalse(plan["production_rescue"]["active"])
        self.assertEqual(plan["production_rescue"]["reason"], "no_safe_probe_recommendations")
        self.assertEqual(plan["probe_recommendations"], [])
        self.assertEqual(plan["quality_budget"]["slots"], {"broad_explore": 8})

    def test_build_experiment_plan_disables_rescue_when_only_unverified_pv_probes_remain(self):
        plan = build_experiment_plan(
            {
                "candidate_count": 160,
                "failure_reasons": {"LOW_SHARPE": 80, "LOW_FITNESS": 80},
                "best_candidate": {
                    "id": 7,
                    "expression": "rank(weak_pv_signal)",
                    "fields": ["weak_pv_signal"],
                    "metrics": {"sharpe": 0.29, "fitness": 0.09, "turnover": 0.2},
                },
                "field_scout": {
                    "active": True,
                    "top_fields": [
                        {
                            "field": "pv13_high_score_signal",
                            "type": "MATRIX",
                            "dataset_id": "pv13",
                            "category": "Price Volume",
                            "score": 0.95,
                            "explored_count": 0,
                            "failed_count": 0,
                            "tower_status": "unlit",
                            "primary_policy": "prefer_primary",
                        }
                    ],
                    "top_primary_fields": [],
                },
                "lit_tower_avoidance": {},
                "route_efficiency": {
                    "stop_loss_active": True,
                    "failure_streak": 140,
                    "scanned_candidates": 160,
                    "watchlist_count": 0,
                    "submitable_count": 0,
                    "best_sharpe_ratio": 0.11,
                    "best_fitness_ratio": 0.06,
                },
                "structure_diversity_control": {},
                "submitted_field_avoidance": {},
                "observed_quality_thresholds": {
                    "required_sharpe": 2.69,
                    "required_fitness": 1.5,
                    "trusted": True,
                },
            },
            {"region": "USA", "universe": "TOP500", "delay": 0, "neutralization": "INDUSTRY"},
            batch_size=8,
        )

        self.assertFalse(plan["production_rescue"]["active"])
        self.assertEqual(plan["production_rescue"]["reason"], "no_safe_probe_recommendations")
        self.assertEqual(plan["probe_recommendations"], [])

    def test_build_experiment_plan_enters_production_rescue_for_long_scope_trouble(self):
        plan = build_experiment_plan(
            {
                "candidate_count": 66,
                "failure_reasons": {"LOW_SHARPE": 66, "LOW_FITNESS": 66},
                "best_candidate": {
                    "id": 8,
                    "expression": "rank(weak_signal)",
                    "fields": ["weak_signal"],
                    "metrics": {"sharpe": 0.13, "fitness": 0.02},
                },
                "scope_health": {
                    "best_recent_sharpe": 0.74,
                    "best_recent_fitness": 0.3,
                    "trouble_signals": {
                        "failure_streak": 243,
                        "scanned_candidates": 243,
                    },
                },
                "field_scout": {
                    "active": True,
                    "top_primary_fields": [],
                    "top_fields": [
                        {
                            "field": "mdl262_predictive_signal",
                            "type": "MATRIX",
                            "dataset_id": "model262",
                            "category": "Model",
                            "score": 0.72,
                            "explored_count": 0,
                            "failed_count": 0,
                            "tower_status": "lit",
                            "primary_policy": "avoid_primary",
                        }
                    ],
                },
                "lit_tower_avoidance": {"tower_names": ["USA/D0/MODEL"]},
                "route_efficiency": {
                    "stop_loss_active": False,
                    "failure_streak": 66,
                    "scanned_candidates": 66,
                    "best_sharpe_ratio": 0.11,
                    "best_fitness_ratio": 0.06,
                },
                "submitted_field_avoidance": {},
                "observed_quality_thresholds": {
                    "required_sharpe": 2.69,
                    "required_fitness": 1.5,
                    "trusted": True,
                },
            },
            {"region": "USA", "universe": "TOP500", "delay": 0, "neutralization": "INDUSTRY"},
            batch_size=8,
        )

        self.assertFalse(plan["production_rescue"]["active"])
        self.assertEqual(plan["production_rescue"]["reason"], "no_safe_probe_recommendations")
        self.assertEqual(plan["quality_budget"]["slots"], {"broad_explore": 8})

    def test_avoid_list_keeps_actionable_turnover_failures_ahead_of_generic_metric_failures(self):
        avoid = _avoid_list(
            [
                "SHARPE_BELOW_MIN",
                "FITNESS_BELOW_MIN",
                "LOW_SHARPE",
                "LOW_FITNESS",
                "SELF_CORRELATION",
                "PROD_CORRELATION",
                "LOW_2Y_SHARPE",
                "LOW_SUB_UNIVERSE_SHARPE",
                "TURNOVER_ABOVE_MAX",
                "HIGH_TURNOVER",
            ],
            [],
        )

        self.assertIn("TURNOVER_ABOVE_MAX", avoid[:8])
        self.assertIn("HIGH_TURNOVER", avoid[:8])

    def test_recent_preflight_rejections_reads_recent_events_without_candidate_scan(self):
        class CountingStore(AlphaStore):
            def __init__(self, path):
                super().__init__(path)
                self.event_lookup_count = 0

            def events_for_candidate(self, candidate_id):
                self.event_lookup_count += 1
                return super().events_for_candidate(candidate_id)

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            store = CountingStore(base / "alpha.db")
            store.init()
            first_id = store.insert_candidate("rank(close)", {"region": "USA", "delay": 0}, "model:G-1")
            store.record_event(first_id, "preflight_failed", {"errors": ["UNKNOWN_FIELD:bad_field"]})
            for index in range(60):
                store.insert_candidate(f"rank(close_{index})", {"region": "USA", "delay": 0}, "model:G-1")

            result = _recent_preflight_rejections(store, 1)

        self.assertEqual(result["unknown_fields"], ["bad_field"])
        self.assertEqual(store.event_lookup_count, 0)

    def test_build_ai_research_context_loads_knowledge_and_candidate_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            knowledge_dir = base / "knowledge"
            knowledge_dir.mkdir()
            (knowledge_dir / "wqb_rules.md").write_text("All mandatory submission checks must pass.", encoding="utf-8")
            (knowledge_dir / "generation_patterns.md").write_text("Prefer multi-operator hypotheses.", encoding="utf-8")
            (knowledge_dir / "optimization_playbook.md").write_text("Optimize only near official thresholds.", encoding="utf-8")

            store = AlphaStore(base / "alpha.db")
            store.init()
            failed_id = store.insert_candidate("rank(close)", {"region": "USA", "delay": 0}, "openai_compatible")
            store.record_event(failed_id, "simulation_error", {"error": "Attempted to use unknown variable x", "attempt": 1})
            store.transition(failed_id, "failed", {"errors": ["LOW_SHARPE"]})
            approved_id = store.insert_candidate(
                "group_rank(ts_rank(ts_delta(close, 5), 22), industry)",
                {"region": "USA", "delay": 0},
                "openai_compatible",
            )
            store.transition(approved_id, "approved")

            context = build_ai_research_context(
                store,
                {"region": "USA", "universe": "TOP3000", "delay": 0},
                knowledge_dir=knowledge_dir,
                reference_dir=base / "missing-reference",
                field_catalog={"available": True, "field_ids": ["mdl_test_signal"], "fields": []},
            )

        self.assertEqual(context["target_settings"]["delay"], 0)
        self.assertIn("All mandatory submission checks", context["knowledge"]["wqb_rules"])
        self.assertIn("official thresholds", context["knowledge"]["optimization_playbook"])
        self.assertEqual(context["generation_policy"]["complexity"], "research_grade")
        self.assertTrue(context["generation_policy"]["reject_trivial_candidates"])
        self.assertTrue(context["generation_policy"]["auxiliary_fields_must_not_be_primary"])
        self.assertEqual(context["datafields"]["field_ids"], ["mdl_test_signal"])
        self.assertEqual(context["recent_failures"][0]["expression"], "rank(close)")
        self.assertEqual(context["recent_failures"][0]["simulation_errors"][0]["error"], "Attempted to use unknown variable x")
        self.assertEqual(context["recent_successes"][0]["expression"], "group_rank(ts_rank(ts_delta(close, 5), 22), industry)")

    def test_build_ai_research_context_includes_field_scout(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            store = AlphaStore(base / "alpha.db")
            store.init()
            settings = {"region": "USA", "universe": "TOP3000", "delay": 0}
            failed_id = store.insert_candidate("rank(failed_signal)", settings, "model:G-1")
            store.update_candidate(failed_id, metrics_json=json.dumps({"sharpe": -0.2, "fitness": -0.1}))
            store.transition(failed_id, "failed", {"errors": ["LOW_SHARPE:FAIL"]})

            context = build_ai_research_context(
                store,
                settings,
                knowledge_dir=base / "missing-knowledge",
                reference_dir=base / "missing-reference",
                field_catalog={
                    "available": True,
                    "field_ids": ["rare_signal", "failed_signal"],
                    "fields": [
                        {
                            "id": "rare_signal",
                            "type": "MATRIX",
                            "dataset_id": "news",
                            "category": "News",
                            "coverage": 0.9,
                            "userCount": 0,
                            "alphaCount": 0,
                            "pyramidMultiplier": 1.8,
                        },
                        {
                            "id": "failed_signal",
                            "type": "MATRIX",
                            "dataset_id": "model",
                            "category": "Model",
                            "coverage": 0.9,
                            "userCount": 0,
                            "alphaCount": 0,
                            "pyramidMultiplier": 1.8,
                        },
                    ],
                },
            )

        scout = context["field_scout"]
        self.assertTrue(scout["active"])
        self.assertEqual(scout["top_fields"][0]["field"], "rare_signal")
        self.assertEqual(scout["top_primary_fields"][0]["field"], "rare_signal")
        self.assertIn("field_scout", context["experiment_plan"])
        self.assertEqual(context["experiment_plan"]["field_scout"]["top_primary_fields"][0]["field"], "rare_signal")
        self.assertIn("high_opportunity_unexplored", [bucket["name"] for bucket in scout["buckets"]])

    def test_build_ai_research_context_keeps_archived_failures_in_field_scout_memory(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            store = AlphaStore(base / "alpha.db")
            store.init()
            settings = {"region": "USA", "universe": "TOP500", "delay": 0}
            failed_id = store.insert_candidate("rank(archived_bad_signal)", settings, "model:G-2")
            store.update_candidate(
                failed_id,
                metrics_json=json.dumps({"sharpe": -0.4, "fitness": -0.2}),
                checks_json=json.dumps({"LOW_SHARPE": {"status": "FAIL", "value": -0.4, "limit": 2.69}}),
            )
            store.transition(failed_id, "failed", {"errors": ["LOW_SHARPE:FAIL"]})
            store.archive_candidates([failed_id], "low_quality_history", {"quality_max": 0.2})

            context = build_ai_research_context(
                store,
                settings,
                knowledge_dir=base / "missing-knowledge",
                reference_dir=base / "missing-reference",
                field_catalog={
                    "available": True,
                    "field_ids": ["fresh_signal", "archived_bad_signal"],
                    "fields": [
                        {
                            "id": "fresh_signal",
                            "type": "MATRIX",
                            "dataset_id": "news",
                            "category": "News",
                            "coverage": 0.9,
                            "userCount": 0,
                            "alphaCount": 0,
                            "pyramidMultiplier": 1.8,
                        },
                        {
                            "id": "archived_bad_signal",
                            "type": "MATRIX",
                            "dataset_id": "news",
                            "category": "News",
                            "coverage": 0.9,
                            "userCount": 0,
                            "alphaCount": 0,
                            "pyramidMultiplier": 1.8,
                        },
                    ],
                },
            )

        scout_fields = {row["field"]: row for row in context["field_scout"]["top_fields"]}
        archived_row = scout_fields["archived_bad_signal"]
        self.assertEqual(archived_row["explored_count"], 1)
        self.assertEqual(archived_row["failed_count"], 1)
        unexplored_bucket = next(
            bucket for bucket in context["field_scout"]["buckets"] if bucket["name"] == "high_opportunity_unexplored"
        )
        self.assertNotIn("archived_bad_signal", unexplored_bucket["fields"])

    def test_daemon_run_field_scout_ignores_failures_before_run_start(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            store = AlphaStore(base / "alpha.db")
            store.init()
            settings = {"region": "USA", "universe": "TOP500", "delay": 0, "neutralization": "INDUSTRY"}
            failed_id = store.insert_candidate("rank(old_bad_signal)", settings, "model:G-2")
            store.update_candidate(
                failed_id,
                metrics_json=json.dumps({"sharpe": -0.4, "fitness": -0.2}),
                checks_json=json.dumps({"LOW_SHARPE": {"status": "FAIL", "value": -0.4, "limit": 2.69}}),
            )
            store.transition(failed_id, "failed", {"errors": ["LOW_SHARPE:FAIL"]})
            store.set_run_state(
                "daemon",
                {
                    "status": "running",
                    "started_at": "2999-01-01T00:00:00+00:00",
                    "scope": settings,
                },
            )

            context = build_ai_research_context(
                store,
                settings,
                knowledge_dir=base / "missing-knowledge",
                reference_dir=base / "missing-reference",
                field_catalog={
                    "available": True,
                    "field_ids": ["old_bad_signal"],
                    "fields": [
                        {
                            "id": "old_bad_signal",
                            "type": "MATRIX",
                            "dataset_id": "news",
                            "category": "News",
                            "coverage": 0.9,
                            "userCount": 0,
                            "alphaCount": 0,
                            "pyramidMultiplier": 1.8,
                        }
                    ],
                },
            )

        scout_fields = {row["field"]: row for row in context["field_scout"]["top_fields"]}
        old_row = scout_fields["old_bad_signal"]
        self.assertEqual(old_row["explored_count"], 0)
        self.assertEqual(old_row["failed_count"], 0)
        self.assertEqual(context["field_scout"]["top_primary_fields"][0]["field"], "old_bad_signal")

    def test_build_ai_research_context_cools_down_standardized_probe_exhausted_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            store = AlphaStore(base / "alpha.db")
            store.init()
            settings = {"region": "USA", "universe": "TOP500", "delay": 0}
            store.record_event(
                None,
                "standardized_probe_exhausted",
                {
                    "reason": "all_standardized_probe_templates_duplicate",
                    "probe_fields": ["aggregate_sentiment_score_3"],
                    "settings": settings,
                },
            )

            context = build_ai_research_context(
                store,
                settings,
                knowledge_dir=base / "missing-knowledge",
                reference_dir=base / "missing-reference",
                field_catalog={
                    "available": True,
                    "field_ids": ["aggregate_sentiment_score_3", "fresh_other_signal"],
                    "fields": [
                        {
                            "id": "aggregate_sentiment_score_3",
                            "type": "VECTOR",
                            "dataset_id": "filing_sentiment",
                            "category": "Other",
                            "coverage": 0.82,
                            "userCount": 0,
                            "alphaCount": 0,
                            "pyramidMultiplier": 1.8,
                        },
                        {
                            "id": "fresh_other_signal",
                            "type": "VECTOR",
                            "dataset_id": "other_fresh",
                            "category": "Other",
                            "coverage": 0.78,
                            "userCount": 1,
                            "alphaCount": 1,
                            "pyramidMultiplier": 1.7,
                        },
                    ],
                },
            )

        scout_fields = {row["field"]: row for row in context["field_scout"]["top_fields"]}
        self.assertEqual(scout_fields["aggregate_sentiment_score_3"]["field_reason"], "standardized_probe_exhausted")
        self.assertEqual(context["field_scout"]["top_primary_fields"][0]["field"], "fresh_other_signal")

    def test_build_ai_research_context_cools_down_production_rescue_probe_exhausted_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            store = AlphaStore(base / "alpha.db")
            store.init()
            settings = {"region": "USA", "universe": "TOP500", "delay": 0}
            store.record_event(
                None,
                "production_rescue_probe_exhausted",
                {
                    "reason": "all_production_rescue_probe_templates_duplicate",
                    "probe_fields": ["snt21_4neut_conf_low"],
                    "settings": settings,
                },
            )

            context = build_ai_research_context(
                store,
                settings,
                knowledge_dir=base / "missing-knowledge",
                reference_dir=base / "missing-reference",
                field_catalog={
                    "available": True,
                    "field_ids": ["snt21_4neut_conf_low", "fresh_other_signal"],
                    "fields": [
                        {
                            "id": "snt21_4neut_conf_low",
                            "type": "MATRIX",
                            "dataset_id": "sentiment21",
                            "category": "Sentiment",
                            "coverage": 0.82,
                            "userCount": 0,
                            "alphaCount": 0,
                            "pyramidMultiplier": 1.8,
                        },
                        {
                            "id": "fresh_other_signal",
                            "type": "VECTOR",
                            "dataset_id": "other_fresh",
                            "category": "Other",
                            "coverage": 0.78,
                            "userCount": 1,
                            "alphaCount": 1,
                            "pyramidMultiplier": 1.7,
                        },
                    ],
                },
            )

        scout_fields = {row["field"]: row for row in context["field_scout"]["top_fields"]}
        self.assertEqual(scout_fields["snt21_4neut_conf_low"]["field_reason"], "production_rescue_probe_exhausted")
        self.assertEqual(context["field_scout"]["top_primary_fields"][0]["field"], "fresh_other_signal")

    def test_build_ai_research_context_adds_syntax_constraints_and_preflight_rejections(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            store = AlphaStore(base / "alpha.db")
            store.init()
            failed_id = store.insert_candidate(
                "rank(ts_mean(vec_avg(analyst_sentence_count_presentation,30),22))",
                {"region": "USA", "delay": 0},
                "openai_compatible",
            )
            store.record_event(
                failed_id,
                "preflight_failed",
                {
                    "errors": [
                        "INVALID_VECTOR_REDUCER_ARITY:vec_avg",
                        "UNKNOWN_FIELD:other351_sentiment_ml",
                    ]
                },
            )
            store.transition(
                failed_id,
                "failed",
                {
                    "reason": "preflight",
                    "errors": [
                        "INVALID_VECTOR_REDUCER_ARITY:vec_avg",
                        "UNKNOWN_FIELD:other351_sentiment_ml",
                    ],
                },
            )

            context = build_ai_research_context(
                store,
                {"region": "USA", "universe": "TOP3000", "delay": 0},
                knowledge_dir=base / "missing-knowledge",
                reference_dir=base / "missing-reference",
            )

        syntax = context["syntax_constraints"]
        self.assertIn("ts_zscore", syntax["allowed_operators"])
        self.assertIn("close", syntax["auxiliary_only_fields"])
        self.assertIn("vwap", syntax["auxiliary_only_fields"])
        self.assertIn("primary alpha signal", syntax["auxiliary_field_rule"])
        self.assertIn("vec_avg(x) only", syntax["vector_reducer_rule"])
        self.assertEqual(syntax["exact_operator_arity"]["group_mean"], 3)
        self.assertIn("group_mean(x, weight, group)", syntax["operator_arity_rule"])
        self.assertIn("SENTIMENT", syntax["turnover_stabilization_rule"])
        self.assertIn("ts_mean", syntax["turnover_stabilization_rule"])
        self.assertIn("other351_sentiment_ml", syntax["recent_preflight_rejections"]["unknown_fields"])
        self.assertIn("INVALID_VECTOR_REDUCER_ARITY:vec_avg", syntax["recent_preflight_rejections"]["invalid_patterns"])

    def test_build_ai_research_context_ignores_preflight_only_candidates_in_structural_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            store = AlphaStore(base / "alpha.db")
            store.init()
            preflight_id = store.insert_candidate(
                (
                    "group_rank(ts_rank(divide(winsorize(ts_backfill("
                    "mdl_other_score, 120), std=3), cap), 63), industry)"
                ),
                {"region": "USA", "delay": 0},
                "openai_compatible",
            )
            store.update_candidate(preflight_id, status="preflight_passed")
            approved_id = store.insert_candidate(
                "group_rank(ts_delta(mdl_mock_score, 22), industry)",
                {"region": "USA", "delay": 0},
                "openai_compatible",
            )
            store.transition(approved_id, "approved")

            context = build_ai_research_context(
                store,
                {"region": "USA", "universe": "TOP3000", "delay": 0},
                knowledge_dir=base / "missing-knowledge",
                reference_dir=base / "missing-reference",
            )

        structures = context["syntax_constraints"]["recent_expression_structures"]
        expressions = [item["expression"] for item in structures]
        self.assertIn("group_rank(ts_delta(mdl_mock_score, 22), industry)", expressions)
        self.assertNotIn(
            (
                "group_rank(ts_rank(divide(winsorize(ts_backfill("
                "mdl_other_score, 120), std=3), cap), 63), industry)"
            ),
            expressions,
        )

    def test_build_ai_research_context_summarizes_reference_brain_project(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            reference_dir = base / "Brain"
            reference_dir.mkdir()
            (reference_dir / "submitted_alphas.csv").write_text(
                "alpha_id,expression,submission_date,region,universe,delay,sharpe,fitness,status\n"
                'A1,"group_rank(ts_rank(winsorize(ts_backfill(mdl262_field, 120), std=4)/cap, 63), industry)",'
                "2026-01-01T00:00:00Z,USA,TOP3000,0,2.1,1.3,SUBMITTED\n",
                encoding="utf-8",
            )
            (reference_dir / "fail_alphas.csv").write_text(
                "id,regular,fail_reason,sharpe,fitness\n"
                'F1,"rank(close)",LOW_SHARPE,0.3,0.1\n',
                encoding="utf-8",
            )
            (reference_dir / "templates_usa_d0_success_submitted.json").write_text(
                json.dumps({"cap_scaled_model_rank": {"templates": ["group_rank(ts_rank({field}/cap, 63), industry)"]}}),
                encoding="utf-8",
            )
            store = AlphaStore(base / "alpha.db")
            store.init()

            context = build_ai_research_context(
                store,
                {"region": "USA", "universe": "TOP3000", "delay": 0},
                knowledge_dir=base / "missing-knowledge",
                reference_dir=reference_dir,
            )

        reference = context["reference_brain_project"]
        self.assertIn("group_rank(ts_rank(winsorize", reference["submitted_success_examples"][0]["expression"])
        self.assertEqual(reference["recent_failure_examples"][0]["fail_reason"], "LOW_SHARPE")
        self.assertIn("cap_scaled_model_rank", reference["usa_d0_template_families"])

    def test_build_ai_research_context_adds_analysis_and_optimization_plan(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            store = AlphaStore(base / "alpha.db")
            store.init()
            weak_id = store.insert_candidate(
                "rank(group_rank(ts_mean(beta_prediction_uncertainty_2,63),industry))",
                {"region": "USA", "delay": 0},
                "openai_compatible",
            )
            store.update_candidate(weak_id, metrics_json=json.dumps({"sharpe": 0.05, "fitness": 0.01}))
            store.transition(weak_id, "check_pending", {"errors": ["SHARPE_BELOW_MIN:0.050<1.58"]})
            best_id = store.insert_candidate(
                "rank(group_rank(winsorize(ts_mean(analyst_positive_sentiment_logit_presentation,30),std=3) * "
                "winsorize(ts_mean(aggregate_sentiment_total,25),std=3)/cap,industry))",
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

            context = build_ai_research_context(
                store,
                {"region": "USA", "universe": "TOP3000", "delay": 0},
                knowledge_dir=base / "missing-knowledge",
                reference_dir=base / "missing-reference",
            )

        analysis = context["analysis"]
        plan = context["experiment_plan"]
        self.assertEqual(analysis["best_candidate"]["id"], best_id)
        self.assertEqual(analysis["best_candidate"]["sharpe"], 2.55)
        self.assertIn("analyst_positive_sentiment_logit_presentation", analysis["promising_fields"])
        self.assertIn("SHARPE_BELOW_MIN", analysis["failure_reasons"])
        self.assertEqual(plan["mode"], "optimize_best")
        self.assertEqual(plan["target_candidate_id"], best_id)
        self.assertEqual(plan["batch_size"], 8)
        self.assertIn("analyst_positive_sentiment_logit_presentation", plan["keep"])

    def test_build_ai_research_context_honors_scheduler_optimize_target(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            store = AlphaStore(base / "alpha.db")
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

            context = build_ai_research_context(
                store,
                {"region": "USA", "universe": "TOP500", "delay": 0, "neutralization": "INDUSTRY"},
                knowledge_dir=base / "missing-knowledge",
                reference_dir=base / "missing-reference",
                cycle_plan={
                    "mode": "optimize",
                    "target_candidate_id": target_id,
                    "budget": {"batch_size": 4},
                    "reason": "probe_optimize_ready_candidate_has_fixable_gap",
                    "constraints": {
                        "cooldown_fields": ["crowded_secondary_signal"],
                        "field_exposure": {
                            "crowded_secondary_signal": {"count": 12, "window_candidates": 40},
                        },
                    },
                },
            )

        plan = context["experiment_plan"]
        self.assertEqual(plan["mode"], "optimize_best")
        self.assertEqual(plan["target_candidate_id"], target_id)
        self.assertEqual(plan["batch_size"], 4)
        self.assertEqual(plan["quality_budget"]["slots"], {"optimize_anchor": 4})
        self.assertEqual(plan["scheduler_plan"]["reason"], "probe_optimize_ready_candidate_has_fixable_gap")
        self.assertEqual(plan["field_exposure_control"]["cooldown_fields"], ["crowded_secondary_signal"])

    def test_build_ai_research_context_propagates_field_exposure_cooldown(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            store = AlphaStore(base / "alpha.db")
            store.init()
            settings = {"region": "USA", "universe": "TOP3000", "delay": 1, "neutralization": "INDUSTRY"}
            context = build_ai_research_context(
                store,
                settings,
                knowledge_dir=base / "missing-knowledge",
                reference_dir=base / "missing-reference",
                field_catalog={
                    "available": True,
                    "field_ids": ["crowded_signal", "fresh_signal"],
                    "field_types": {"crowded_signal": "MATRIX", "fresh_signal": "MATRIX"},
                    "fields": [
                        {"id": "crowded_signal", "type": "MATRIX", "category": "Other"},
                        {"id": "fresh_signal", "type": "MATRIX", "category": "Other"},
                    ],
                },
                cycle_plan={
                    "mode": "explore",
                    "reason": "optimize_targets_cooldown",
                    "budget": {"batch_size": 8},
                    "constraints": {
                        "cooldown_fields": ["crowded_signal"],
                        "field_exposure": {"crowded_signal": {"count": 8, "window_candidates": 20}},
                    },
                },
            )

        plan = context["experiment_plan"]
        self.assertIn("crowded_signal", plan["avoid"])
        self.assertEqual(plan["field_exposure_control"]["cooldown_fields"], ["crowded_signal"])
        self.assertEqual(plan["scheduler_plan"]["reason"], "optimize_targets_cooldown")

    def test_build_ai_research_context_marks_near_threshold_correlation_target_for_deep_repair(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            store = AlphaStore(base / "alpha.db")
            store.init()
            target_id = store.insert_candidate(
                "group_rank(ts_rank(divide(winsorize(ts_backfill(vec_avg(ern7_dsu_spe),120),std=3),cap),42),country)",
                {"region": "USA", "universe": "TOP3000", "delay": 0, "neutralization": "STATISTICAL"},
                "model:G-1",
            )
            store.update_candidate(
                target_id,
                metrics_json=json.dumps({"sharpe": 3.21, "fitness": 1.46, "turnover": 0.3645}),
                checks_json=json.dumps(
                    {
                        "LOW_SHARPE": {"status": "PASS", "value": 3.21, "limit": 2.69},
                        "LOW_FITNESS": {"status": "FAIL", "value": 1.46, "limit": 1.5},
                        "LOW_TURNOVER": {"status": "PASS", "value": 0.3645, "limit": 0.01},
                        "HIGH_TURNOVER": {"status": "PASS", "value": 0.3645, "limit": 0.7},
                        "PROD_CORRELATION": {"status": "FAIL", "value": 0.714, "limit": 0.7},
                        "DATA_DIVERSITY": {"status": "WARNING"},
                    }
                ),
            )
            store.transition(
                target_id,
                "failed",
                {"errors": ["FITNESS_BELOW_MIN:1.460<1.5", "PROD_CORRELATION:0.714>0.7"]},
            )

            context = build_ai_research_context(
                store,
                {"region": "USA", "universe": "TOP3000", "delay": 0, "neutralization": "STATISTICAL"},
                knowledge_dir=base / "missing-knowledge",
                reference_dir=base / "missing-reference",
                cycle_plan={
                    "mode": "optimize",
                    "target_candidate_id": target_id,
                    "budget": {"batch_size": 4},
                    "reason": "near_threshold_candidate_has_fixable_gap",
                },
            )

        plan = context["experiment_plan"]
        repair = plan["targeted_repair"]
        self.assertEqual(plan["mode"], "optimize_best")
        self.assertTrue(repair["active"])
        self.assertEqual(repair["orchestration"], "deep_repair")
        self.assertIn("fitness_gap", repair["triggers"])
        self.assertIn("prod_correlation", repair["triggers"])
        self.assertIn("ern7_dsu_spe", repair["cooldown_primary_fields"])
        self.assertIn("ern7_dsu_spe", plan["avoid"])
        self.assertNotIn("ern7_dsu_spe", plan["keep"])
        self.assertIn("near-threshold repair", plan["quality_budget"]["policy"])

    def test_cycle_plan_does_not_optimize_submitted_field_target_without_correlation_blocker(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            store = AlphaStore(base / "alpha.db")
            store.init()
            settings = {"region": "USA", "universe": "TOP3000", "delay": 0, "neutralization": "STATISTICAL"}
            platform_submissions = [
                {
                    "id": "OS1",
                    "stage": "OS",
                    "regular": {
                        "code": "rank(group_rank(ts_rank(vec_avg(ern7_dsu_spe), 63), sector))",
                        "operatorCount": 4,
                    },
                    "settings": settings,
                    "sharpe": 3.4,
                    "fitness": 1.61,
                }
            ]
            target_id = store.insert_candidate(
                "rank(multiply(group_rank(ts_rank(divide(winsorize(ts_backfill(vec_avg(ern7_dsu_spe),105),std=4),cap),42),country),group_rank(ts_rank(divide(winsorize(ts_backfill(vec_avg(ern7_dsu_spe),180),std=3),cap),126),country)))",
                settings,
                "model:G-1",
            )
            store.update_candidate(
                target_id,
                metrics_json=json.dumps({"sharpe": 3.09, "fitness": 1.49, "turnover": 0.3059}),
                checks_json=json.dumps(
                    {
                        "LOW_SHARPE": {"status": "PASS", "value": 3.09, "limit": 2.69},
                        "LOW_FITNESS": {"status": "FAIL", "value": 1.49, "limit": 1.5},
                        "DATA_DIVERSITY": {"status": "WARNING"},
                    }
                ),
            )
            store.transition(target_id, "failed", {"errors": ["FITNESS_BELOW_MIN:1.490<1.5"]})

            context = build_ai_research_context(
                store,
                settings,
                knowledge_dir=base / "missing-knowledge",
                reference_dir=base / "missing-reference",
                platform_submissions=platform_submissions,
                cycle_plan={
                    "mode": "optimize",
                    "target_candidate_id": target_id,
                    "budget": {"batch_size": 4},
                    "reason": "near_threshold_candidate_has_fixable_gap",
                },
            )

        plan = context["experiment_plan"]
        self.assertEqual(plan["mode"], "explore_new_family")
        self.assertEqual(plan["abandoned_target_id"], target_id)
        self.assertEqual(plan["abandon_reason"], "SUBMITTED_FIELD_AVOIDANCE")
        self.assertIsNone(plan["target_candidate_id"])
        self.assertIn("ern7_dsu_spe", plan["avoid"])
        self.assertNotIn("ern7_dsu_spe", plan["keep"])
        self.assertIn("ern7_dsu_spe", plan["submitted_field_avoidance"]["fields"])

    def test_build_ai_research_context_disables_rescue_after_duplicate_only_stop(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            store = AlphaStore(base / "alpha.db")
            store.init()
            settings = {"region": "USA", "universe": "TOP500", "delay": 0, "neutralization": "INDUSTRY"}
            for idx in range(80):
                candidate_id = store.insert_candidate(f"rank(weak_signal_{idx})", settings, "planner_unverified_probe")
                store.update_candidate(candidate_id, metrics_json=json.dumps({"sharpe": 0.0, "fitness": 0.0}))
                store.transition(candidate_id, "failed", {"reason": "bad_full_batch"})

            context = build_ai_research_context(
                store,
                settings,
                knowledge_dir=base / "missing-knowledge",
                reference_dir=base / "missing-reference",
                cycle_plan={
                    "mode": "explore",
                    "reason": "production_rescue_duplicate_only_recent",
                    "budget": {"batch_size": 8},
                    "constraints": {"avoid_modes": ["production_rescue"]},
                },
            )

        plan = context["experiment_plan"]
        self.assertEqual(plan["mode"], "explore_new_family")
        self.assertFalse(plan["production_rescue"]["active"])
        self.assertEqual(plan["quality_budget"]["slots"], {"broad_explore": 8})
        self.assertEqual(plan["scheduler_plan"]["reason"], "production_rescue_duplicate_only_recent")

    def test_build_ai_research_context_adds_compressed_history_memory(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            store = AlphaStore(base / "alpha.db")
            store.init()
            settings = {"region": "USA", "universe": "TOP3000", "delay": 0}
            for idx in range(18):
                field = f"hist_signal_{idx % 3}"
                candidate_id = store.insert_candidate(
                    f"group_rank(ts_rank(ts_backfill({field},120),63),industry)",
                    settings,
                    "model:G-1" if idx % 2 == 0 else "model:G-2",
                )
                store.update_candidate(
                    candidate_id,
                    metrics_json=json.dumps(
                        {
                            "sharpe": 0.5 + idx / 10,
                            "fitness": 0.2 + idx / 20,
                            "turnover": 0.1,
                        }
                    ),
                )
                if idx % 5 == 0:
                    store.transition(candidate_id, "approved")
                else:
                    store.transition(candidate_id, "failed", {"errors": ["SHARPE_BELOW_MIN:0.5<1.58"]})

            context = build_ai_research_context(
                store,
                settings,
                knowledge_dir=base / "missing-knowledge",
                reference_dir=base / "missing-reference",
            )

        memory = context["history_memory"]
        field_rows = {row["field"]: row for row in memory["top_fields"]}
        self.assertEqual(memory["scanned_candidates"], 18)
        self.assertEqual(memory["status_counts"]["failed"], 14)
        self.assertEqual(field_rows["hist_signal_0"]["count"], 6)
        self.assertGreaterEqual(field_rows["hist_signal_2"]["best_sharpe"], 2.0)
        self.assertIn("SHARPE_BELOW_MIN", [row["reason"] for row in memory["top_failure_reasons"]])
        self.assertIn("G-1", memory["profile_outcomes"])

    def test_history_memory_counts_probe_validation_dataset_failures(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            store = AlphaStore(base / "alpha.db")
            store.init()
            settings = {"region": "USA", "universe": "TOP500", "delay": 0}
            for idx in range(4):
                candidate_id = store.insert_candidate(
                    f"rank(ts_rank(winsorize(ts_backfill(model37_field_{idx}, 120), std=4), 63))",
                    settings,
                    "planner_unverified_probe",
                )
                store.update_candidate(
                    candidate_id,
                    metrics_json=json.dumps({"sharpe": 0.1, "fitness": 0.02, "turnover": 0.1}),
                )
                store.record_event(
                    candidate_id,
                    "probe_validation",
                    {
                        "stage": "reject",
                        "probe_dataset_id": "model37",
                        "probe_field": f"model37_field_{idx}",
                        "metrics": {"sharpe": 0.1, "fitness": 0.02},
                    },
                )
                store.transition(candidate_id, "failed", {"errors": ["LOW_SHARPE"]})

            context = build_ai_research_context(
                store,
                settings,
                knowledge_dir=base / "missing-knowledge",
                reference_dir=base / "missing-reference",
            )

        datasets = {row["dataset_id"]: row for row in context["history_memory"]["top_field_datasets"]}
        self.assertIn("model37", datasets)
        self.assertEqual(datasets["model37"]["count"], 4)
        self.assertEqual(datasets["model37"]["failed"], 4)

    def test_history_memory_does_not_count_simulation_stage_timeout_as_field_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            store = AlphaStore(base / "alpha.db")
            store.init()
            settings = {"region": "USA", "universe": "TOP500", "delay": 0}
            candidate_id = store.insert_candidate(
                "rank(ts_mean(fnd6_aqi,20))",
                settings,
                "deterministic_fallback",
            )
            store.record_event(
                candidate_id,
                "generated",
                {
                    "ai_metadata": {
                        "fallback_dataset_id": "fundamental6",
                        "fallback_field": "fnd6_aqi",
                    }
                },
            )
            store.record_event(
                candidate_id,
                "simulation_error",
                {
                    "error": "simulation_stage_timeout",
                    "raw": {"timeout_seconds": 300.0, "elapsed_seconds": 300.001},
                },
            )
            store.transition(candidate_id, "failed", {"reason": "simulation_error", "error": "simulation_stage_timeout"})

            context = build_ai_research_context(
                store,
                settings,
                knowledge_dir=base / "missing-knowledge",
                reference_dir=base / "missing-reference",
                field_catalog={
                    "available": True,
                    "field_ids": ["fnd6_aqi"],
                    "fields": [
                        {
                            "id": "fnd6_aqi",
                            "dataset_id": "fundamental6",
                            "category": "Fundamental",
                            "type": "MATRIX",
                        }
                    ],
                },
            )

        memory = context["history_memory"]
        self.assertEqual(memory["status_counts"]["simulation_timeout"], 1)
        self.assertNotIn("failed", memory["status_counts"])
        self.assertEqual(memory["top_fields"], [])
        self.assertEqual(memory["top_field_datasets"], [])

    def test_build_ai_research_context_adds_mechanism_transfer_memory_under_constraints(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            store = AlphaStore(base / "alpha.db")
            store.init()
            settings = {"region": "MEA", "universe": "TOP300", "delay": 1}
            approved_id = store.insert_candidate(
                "group_rank(ts_rank(est_q_pre_mean, 66), industry)",
                settings,
                "model:G-2",
            )
            store.update_candidate(approved_id, metrics_json=json.dumps({"sharpe": 2.4, "fitness": 1.3, "turnover": 0.2}))
            store.transition(approved_id, "approved")
            pv_id = store.insert_candidate(
                "group_rank(ts_mean(divide(vwap, close), 33), industry)",
                settings,
                "model:G-1",
            )
            store.update_candidate(pv_id, metrics_json=json.dumps({"sharpe": 2.1, "fitness": 1.1, "turnover": 0.28}))
            store.transition(pv_id, "failed", {"errors": ["SELF_CORRELATION:FAIL"]})
            for idx in range(120):
                weak_id = store.insert_candidate(
                    f"group_rank(ts_rank(mdl31_weak_signal_{idx}, 63), industry)",
                    settings,
                    "model:G-1",
                )
                store.update_candidate(weak_id, metrics_json=json.dumps({"sharpe": 0.2, "fitness": 0.05, "turnover": 0.4}))
                store.transition(weak_id, "failed", {"errors": ["LOW_SHARPE"]})

            context = build_ai_research_context(
                store,
                settings,
                knowledge_dir=base / "missing-knowledge",
                reference_dir=base / "missing-reference",
                field_catalog={
                    "available": True,
                    "field_ids": ["est_q_pre_mean", "vwap", "close", *[f"mdl31_weak_signal_{idx}" for idx in range(120)]],
                    "fields": [],
                },
            )

        memory = context["history_memory"]
        plan = context["experiment_plan"]
        archetypes = memory["blocked_winner_archetypes"]
        self.assertTrue(memory["scope_health"]["trouble_signals"]["failure_streak"] >= 100)
        self.assertTrue(plan["scope_trouble"]["active"])
        self.assertIn("mechanism_transfer", plan)
        self.assertIn("est_q_pre_mean", plan["mechanism_transfer"]["forbidden_fields"])
        self.assertIn("vwap", plan["mechanism_transfer"]["forbidden_fields"])
        self.assertIn("close", plan["mechanism_transfer"]["forbidden_fields"])
        self.assertTrue(any(row["id"] == approved_id for row in archetypes))
        self.assertTrue(any(row["id"] == pv_id for row in archetypes))
        self.assertTrue(any("time_series_persistence_rank" in row["mechanism_tags"] for row in archetypes))
        self.assertIn("mechanism only", plan["mechanism_transfer"]["policy"])
        # H6: hard-mode mechanism samples must NOT leak the raw blocked expression.
        for row in archetypes:
            self.assertNotIn("expression", row)
        for archetype in plan["mechanism_transfer"].get("archetypes", []):
            self.assertNotIn("expression", archetype)

    def test_history_memory_does_not_treat_known_operators_as_fields_without_catalog(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            store = AlphaStore(base / "alpha.db")
            store.init()
            candidate_id = store.insert_candidate(
                "group_rank(normalize(pasteurize(divide(quantile(mdl31_dy_pct_current), inverse(mdl31_pb_current)))), industry)",
                {"region": "MEA", "universe": "TOP300", "delay": 1},
                "model:G-1",
            )
            store.update_candidate(candidate_id, metrics_json=json.dumps({"sharpe": 1.5, "fitness": 0.72}))
            store.transition(candidate_id, "failed", {"errors": ["LOW_SHARPE"]})

            context = build_ai_research_context(
                store,
                {"region": "MEA", "universe": "TOP300", "delay": 1},
                knowledge_dir=base / "missing-knowledge",
                reference_dir=base / "missing-reference",
            )

        fields = {row["field"] for row in context["history_memory"]["top_fields"]}
        self.assertIn("mdl31_dy_pct_current", fields)
        self.assertIn("mdl31_pb_current", fields)
        self.assertNotIn("pasteurize", fields)
        self.assertNotIn("quantile", fields)
        self.assertNotIn("inverse", fields)

    def test_build_ai_research_context_avoids_recent_approved_submitted_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            store = AlphaStore(base / "alpha.db")
            store.init()
            settings = {"region": "USA", "universe": "TOP3000", "delay": 0}

            approved_id = store.insert_candidate(
                "group_rank(ts_rank(est_q_pre_mean, 63), industry)",
                settings,
                "model:G-1",
            )
            store.update_candidate(approved_id, metrics_json=json.dumps({"sharpe": 2.8, "fitness": 1.6}))
            store.transition(approved_id, "approved")

            repeated_id = store.insert_candidate(
                "group_rank(ts_rank(winsorize(ts_backfill(est_q_pre_mean, 120), std=3), 66), industry)",
                settings,
                "model:G-2",
            )
            store.update_candidate(repeated_id, metrics_json=json.dumps({"sharpe": 3.1, "fitness": 1.8}))
            store.transition(repeated_id, "check_pending", {"errors": ["SELF_CORRELATION:PENDING"]})

            fresh_id = store.insert_candidate(
                "group_rank(ts_rank(fresh_signal_score, 63), industry)",
                settings,
                "model:G-1",
            )
            store.update_candidate(fresh_id, metrics_json=json.dumps({"sharpe": 1.4, "fitness": 0.52}))
            store.transition(
                fresh_id,
                "check_pending",
                {"errors": ["SHARPE_BELOW_MIN:1.400<2.69", "FITNESS_BELOW_MIN:0.520<1.50"]},
            )

            context = build_ai_research_context(
                store,
                settings,
                knowledge_dir=base / "missing-knowledge",
                reference_dir=base / "missing-reference",
                field_catalog={"available": True, "field_ids": ["est_q_pre_mean", "fresh_signal_score"], "fields": []},
            )

        avoidance = context["submitted_field_avoidance"]
        analysis = context["analysis"]
        plan = context["experiment_plan"]
        self.assertIn("est_q_pre_mean", avoidance["fields"])
        self.assertIn("est_q_pre_mean", analysis["submitted_avoid_fields"])
        self.assertEqual(analysis["best_candidate"]["id"], fresh_id)
        self.assertNotEqual(analysis["best_candidate"]["id"], approved_id)
        self.assertNotEqual(analysis["best_candidate"]["id"], repeated_id)
        self.assertIn("est_q_pre_mean", plan["avoid"])
        self.assertNotIn("est_q_pre_mean", plan["keep"])
        self.assertEqual(plan["submitted_field_avoidance"]["fields"], ["est_q_pre_mean"])

    def test_build_ai_research_context_avoids_reference_submitted_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            reference_dir = base / "Brain"
            reference_dir.mkdir()
            reference_dir.joinpath("submitted_alphas.csv").write_text(
                "alpha_id,expression,submission_date,region,universe,delay,sharpe,fitness,returns,margin,status\n"
                "A1,\"{'code': 'group_rank(ts_rank(est_q_pre_mean, 63), industry)', 'operatorCount': 3}\","
                "2026-05-01T00:00:00Z,USA,TOP3000,0,2.7,1.5,0.1,0.01,SUBMITTED\n",
                encoding="utf-8",
            )
            store = AlphaStore(base / "alpha.db")
            store.init()
            settings = {"region": "USA", "universe": "TOP3000", "delay": 0}

            repeated_id = store.insert_candidate(
                "group_rank(ts_rank(winsorize(est_q_pre_mean, std=3), 66), industry)",
                settings,
                "model:G-1",
            )
            store.update_candidate(repeated_id, metrics_json=json.dumps({"sharpe": 3.2, "fitness": 1.7}))
            store.transition(repeated_id, "check_pending", {"errors": ["SELF_CORRELATION:PENDING"]})

            fresh_id = store.insert_candidate(
                "group_rank(ts_rank(fresh_signal_score, 63), industry)",
                settings,
                "model:G-2",
            )
            store.update_candidate(fresh_id, metrics_json=json.dumps({"sharpe": 1.2, "fitness": 0.4}))
            store.transition(fresh_id, "check_pending", {"errors": ["SHARPE_BELOW_MIN:1.200<2.69"]})

            context = build_ai_research_context(
                store,
                settings,
                knowledge_dir=base / "missing-knowledge",
                reference_dir=reference_dir,
            )

        self.assertIn("est_q_pre_mean", context["submitted_field_avoidance"]["fields"])
        self.assertEqual(context["submitted_field_avoidance"]["examples"][0]["source"], "reference_submitted_alphas")
        self.assertEqual(context["analysis"]["best_candidate"]["id"], fresh_id)
        self.assertNotEqual(context["analysis"]["best_candidate"]["id"], repeated_id)
        self.assertIn("est_q_pre_mean", context["experiment_plan"]["avoid"])

    def test_build_ai_research_context_avoids_platform_submitted_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            store = AlphaStore(base / "alpha.db")
            store.init()
            settings = {"region": "USA", "universe": "TOP3000", "delay": 0}
            platform_submissions = [
                {
                    "id": "P1",
                    "stage": "OS",
                    "regular": {"code": "group_rank(ts_rank(est_q_pre_mean, 63), industry)", "operatorCount": 3},
                    "settings": settings,
                    "sharpe": 2.7,
                    "fitness": 1.5,
                }
            ]

            repeated_id = store.insert_candidate(
                "group_rank(ts_rank(winsorize(est_q_pre_mean, std=3), 66), industry)",
                settings,
                "model:G-1",
            )
            store.update_candidate(repeated_id, metrics_json=json.dumps({"sharpe": 3.2, "fitness": 1.7}))
            store.transition(repeated_id, "check_pending", {"errors": ["SELF_CORRELATION:PENDING"]})

            fresh_id = store.insert_candidate(
                "group_rank(ts_rank(fresh_signal_score, 63), industry)",
                settings,
                "model:G-2",
            )
            store.update_candidate(fresh_id, metrics_json=json.dumps({"sharpe": 1.2, "fitness": 0.4}))
            store.transition(fresh_id, "check_pending", {"errors": ["SHARPE_BELOW_MIN:1.200<2.69"]})

            context = build_ai_research_context(
                store,
                settings,
                knowledge_dir=base / "missing-knowledge",
                reference_dir=base / "missing-reference",
                platform_submissions=platform_submissions,
            )

        self.assertIn("est_q_pre_mean", context["submitted_field_avoidance"]["fields"])
        self.assertEqual(context["submitted_field_avoidance"]["examples"][0]["source"], "platform_os_alphas")
        self.assertEqual(context["analysis"]["best_candidate"]["id"], fresh_id)
        self.assertNotEqual(context["analysis"]["best_candidate"]["id"], repeated_id)
        self.assertIn("est_q_pre_mean", context["experiment_plan"]["avoid"])

    def test_build_ai_research_context_adds_lit_tower_avoidance_from_quarter_pyramid_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            store = AlphaStore(base / "alpha.db")
            store.init()
            settings = {"region": "MEA", "universe": "TOP300", "delay": 1}

            context = build_ai_research_context(
                store,
                settings,
                knowledge_dir=base / "missing-knowledge",
                reference_dir=base / "missing-reference",
                platform_pyramid_alphas={
                    "pyramids": [
                        {
                            "category": {"id": "pv", "name": "Price Volume"},
                            "region": "MEA",
                            "delay": 1,
                            "alphaCount": 3,
                        },
                        {
                            "category": {"id": "model", "name": "Model"},
                            "region": "MEA",
                            "delay": 1,
                            "alphaCount": 2,
                        },
                        {
                            "category": {"id": "fundamental", "name": "Fundamental"},
                            "region": "MEA",
                            "delay": 1,
                            "alphaCount": 0,
                        },
                        {
                            "category": {"id": "pv", "name": "Price Volume"},
                            "region": "USA",
                            "delay": 0,
                            "alphaCount": 28,
                        },
                    ]
                },
                platform_pyramid_multipliers={
                    "pyramids": [
                        {
                            "category": {"id": "pv", "name": "Price Volume"},
                            "region": "MEA",
                            "delay": 1,
                            "multiplier": 1.6,
                        },
                        {
                            "category": {"id": "model", "name": "Model"},
                            "region": "MEA",
                            "delay": 1,
                            "multiplier": 1.8,
                        },
                        {
                            "category": {"id": "fundamental", "name": "Fundamental"},
                            "region": "MEA",
                            "delay": 1,
                            "multiplier": 1.5,
                        },
                    ]
                },
            )

        avoidance = context["lit_tower_avoidance"]
        self.assertEqual(avoidance["min_alpha_count"], 3)
        self.assertEqual(avoidance["tower_names"], ["MEA/D1/PV"])
        self.assertEqual(avoidance["categories"], ["PV"])
        self.assertEqual(avoidance["lit_towers"][0]["multiplier"], 1.6)
        self.assertIn("MEA/D1/FUNDAMENTAL", [item["name"] for item in avoidance["unlit_towers"]])
        self.assertIn("MEA/D1/MODEL", [item["name"] for item in avoidance["unlit_towers"]])
        self.assertIn("MEA/D1/PV", context["analysis"]["lit_tower_avoidance"]["tower_names"])
        self.assertIn("MEA/D1/PV", context["experiment_plan"]["avoid"])
        self.assertNotIn("MEA/D1/MODEL", context["experiment_plan"]["avoid"])
        self.assertEqual(context["experiment_plan"]["lit_tower_avoidance"]["source"], "platform_pyramid_alphas")

    def test_build_ai_research_context_does_not_treat_recent_submission_pyramids_as_lit_tower_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            store = AlphaStore(base / "alpha.db")
            store.init()
            settings = {"region": "MEA", "universe": "TOP300", "delay": 1}

            context = build_ai_research_context(
                store,
                settings,
                knowledge_dir=base / "missing-knowledge",
                reference_dir=base / "missing-reference",
                platform_submissions=[
                    {
                        "id": "P1",
                        "stage": "OS",
                        "settings": settings,
                        "pyramidThemes": {
                            "effective": 2,
                            "pyramids": [
                                {"name": "MEA/D1/PV", "multiplier": 1.6},
                                {"name": "MEA/D1/MODEL", "multiplier": 1.8},
                            ],
                        },
                    }
                ],
            )

        avoidance = context["lit_tower_avoidance"]
        self.assertEqual(avoidance["source"], "none")
        self.assertEqual(avoidance["tower_names"], [])

    def test_build_ai_research_context_does_not_infer_towers_from_field_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            store = AlphaStore(base / "alpha.db")
            store.init()
            settings = {"region": "USA", "universe": "TOP3000", "delay": 0}
            approved_id = store.insert_candidate(
                "group_rank(ts_rank(credit_risk_news_component_score, 63), industry)",
                settings,
                "model:G-1",
            )
            store.update_candidate(approved_id, metrics_json=json.dumps({"sharpe": 2.8, "fitness": 1.5}))
            store.transition(approved_id, "approved")

            context = build_ai_research_context(
                store,
                settings,
                knowledge_dir=base / "missing-knowledge",
                reference_dir=base / "missing-reference",
            )

        self.assertEqual(context["lit_tower_avoidance"]["tower_names"], [])
        self.assertEqual(context["lit_tower_avoidance"]["source"], "none")

    def test_build_ai_research_context_keeps_lit_towers_in_plan_avoid_when_weak_fields_are_many(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            store = AlphaStore(base / "alpha.db")
            store.init()
            settings = {"region": "USA", "universe": "TOP3000", "delay": 1}
            for idx in range(20):
                candidate_id = store.insert_candidate(
                    f"group_rank(ts_rank(weak_signal_{idx}, 63), industry)",
                    settings,
                    "model:G-1",
                )
                store.update_candidate(candidate_id, metrics_json=json.dumps({"sharpe": -0.1, "fitness": -0.02}))
                store.transition(candidate_id, "failed", {"errors": ["LOW_SHARPE"]})

            context = build_ai_research_context(
                store,
                settings,
                knowledge_dir=base / "missing-knowledge",
                reference_dir=base / "missing-reference",
                platform_pyramid_alphas={
                    "pyramids": [
                        {
                            "category": {"id": "pv", "name": "Price Volume"},
                            "region": "USA",
                            "delay": 1,
                            "alphaCount": 3,
                        }
                    ]
                },
            )

        self.assertIn("USA/D1/PV", context["experiment_plan"]["avoid"])

    def test_build_ai_research_context_adds_family_diversity_control_for_dominant_family(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            store = AlphaStore(base / "alpha.db")
            store.init()

            for idx, sharpe in enumerate([1.25, 1.18, 1.05, 0.97], start=1):
                candidate_id = store.insert_candidate(
                    (
                        "group_rank(ts_rank(divide(winsorize(ts_backfill("
                        "credit_risk_global_percentile_score, 120), std=4), ts_mean(cap, 22)), 63), industry)"
                    ),
                    {"region": "USA", "delay": 0},
                    "openai_compatible",
                )
                store.update_candidate(candidate_id, metrics_json=json.dumps({"sharpe": sharpe, "fitness": 0.8 + idx * 0.05}))
                store.transition(candidate_id, "check_pending", {"errors": ["SHARPE_BELOW_MIN:1.000<1.58"]})

            for idx, sharpe in enumerate([1.55, 1.48], start=1):
                candidate_id = store.insert_candidate(
                    "group_rank(ts_rank(analyst_positive_sentiment_logit_presentation, 63), industry)",
                    {"region": "USA", "delay": 0},
                    "openai_compatible",
                )
                store.update_candidate(candidate_id, metrics_json=json.dumps({"sharpe": sharpe, "fitness": 0.9 + idx * 0.05}))
                store.transition(candidate_id, "check_pending", {"errors": ["SHARPE_BELOW_MIN:1.000<1.58"]})

            context = build_ai_research_context(
                store,
                {"region": "USA", "universe": "TOP3000", "delay": 0},
                knowledge_dir=base / "missing-knowledge",
                reference_dir=base / "missing-reference",
                field_catalog={
                    "available": True,
                    "field_ids": [
                        "credit_risk_global_percentile_score",
                        "cap",
                        "analyst_positive_sentiment_logit_presentation",
                    ],
                    "fields": [],
                },
            )

        plan = context["experiment_plan"]
        control = plan["family_diversity_control"]
        self.assertEqual(control["dominant_family"], "credit_risk")
        self.assertGreater(control["dominant_share"], 0.5)
        self.assertIn("analyst_positive_sentiment_logit_presentation", plan["keep"])

    def test_build_ai_research_context_keeps_history_scoped_to_target_region(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            store = AlphaStore(base / "alpha.db")
            store.init()
            usa_id = store.insert_candidate(
                "rank(group_rank(winsorize(ts_mean(credit_model_structural_letter_grade_float,30),std=3),industry))",
                {"region": "USA", "universe": "TOP3000", "delay": 0},
                "openai_compatible",
            )
            store.update_candidate(usa_id, metrics_json=json.dumps({"sharpe": 2.4, "fitness": 1.2}))
            store.transition(usa_id, "check_pending", {"errors": ["SHARPE_BELOW_MIN:2.400<1.58"]})
            ind_id = store.insert_candidate(
                "rank(group_rank(winsorize(ts_mean(analyst_positive_sentiment_logit_presentation,30),std=3),industry))",
                {"region": "IND", "universe": "TOP500", "delay": 1},
                "openai_compatible",
            )
            store.update_candidate(
                ind_id,
                metrics_json=json.dumps({"sharpe": 1.52, "fitness": 0.88, "turnover": 0.2}),
                checks_json=json.dumps(
                    {
                        "LOW_SHARPE": {"status": "FAIL", "value": 1.52, "limit": 1.58},
                        "LOW_FITNESS": {"status": "FAIL", "value": 0.88, "limit": 1.0},
                        "LOW_TURNOVER": {"status": "PASS", "value": 0.2, "limit": 0.01},
                        "HIGH_TURNOVER": {"status": "PASS", "value": 0.2, "limit": 0.7},
                        "CONCENTRATED_WEIGHT": {"status": "PASS"},
                        "LOW_SUB_UNIVERSE_SHARPE": {"status": "PASS", "value": 1.0, "limit": 0.8},
                        "IS_LADDER_SHARPE": {"status": "PASS", "value": 1.7, "limit": 1.58},
                    }
                ),
            )
            store.transition(
                ind_id,
                "check_pending",
                {"errors": ["SHARPE_BELOW_MIN:1.520<1.58", "FITNESS_BELOW_MIN:0.880<1.0"]},
            )

            context = build_ai_research_context(
                store,
                {"region": "IND", "universe": "TOP500", "delay": 1},
                knowledge_dir=base / "missing-knowledge",
                reference_dir=base / "missing-reference",
            )

        analysis = context["analysis"]
        plan = context["experiment_plan"]
        self.assertEqual(analysis["best_candidate"]["id"], ind_id)
        self.assertEqual(analysis["best_candidate"]["expression"].count("analyst_positive_sentiment_logit_presentation"), 1)
        self.assertEqual(plan["target_candidate_id"], ind_id)
        self.assertEqual(plan["target_settings"]["region"], "IND")

    def test_build_ai_research_context_does_not_optimize_failed_candidate_when_pending_exists(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            store = AlphaStore(base / "alpha.db")
            store.init()
            failed_id = store.insert_candidate(
                "rank(add(group_rank(ts_rank(ts_backfill(old_signal,120),63),industry),"
                "group_rank(subtract(0,ts_delta(close,5)),industry)))",
                {"region": "USA", "universe": "TOP3000", "delay": 0},
                "openai_compatible",
            )
            store.update_candidate(
                failed_id,
                metrics_json=json.dumps({"sharpe": 2.55, "fitness": 1.2, "turnover": 0.2}),
                checks_json=json.dumps(
                    {
                        "LOW_SHARPE": {"status": "WARNING", "value": 2.55, "limit": 2.69},
                        "LOW_FITNESS": {"status": "WARNING", "value": 1.2, "limit": 1.5},
                    }
                ),
            )
            store.transition(failed_id, "failed", {"errors": ["REVERSION_COMPONENT:WARNING"]})
            pending_id = store.insert_candidate(
                "rank(group_rank(ts_rank(ts_backfill(current_signal,120),63),industry))",
                {"region": "USA", "universe": "TOP3000", "delay": 0},
                "openai_compatible",
            )
            store.update_candidate(
                pending_id,
                metrics_json=json.dumps({"sharpe": 1.9, "fitness": 0.9, "turnover": 0.2}),
                checks_json=json.dumps(
                    {
                        "LOW_SHARPE": {"status": "FAIL", "value": 1.9, "limit": 2.69},
                        "LOW_FITNESS": {"status": "FAIL", "value": 0.9, "limit": 1.5},
                        "LOW_TURNOVER": {"status": "PASS", "value": 0.2, "limit": 0.01},
                        "HIGH_TURNOVER": {"status": "PASS", "value": 0.2, "limit": 0.7},
                    }
                ),
            )
            store.transition(pending_id, "check_pending", {"errors": ["SHARPE_BELOW_MIN:1.900<2.69"]})

            context = build_ai_research_context(
                store,
                {"region": "USA", "universe": "TOP3000", "delay": 0},
                knowledge_dir=base / "missing-knowledge",
                reference_dir=base / "missing-reference",
            )

        self.assertEqual(context["analysis"]["best_candidate"]["id"], pending_id)
        self.assertEqual(context["experiment_plan"]["target_candidate_id"], pending_id)

    def test_build_ai_research_context_uses_scope_thresholds_for_near_threshold_decision(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            store = AlphaStore(base / "alpha.db")
            store.init()
            candidate_id = store.insert_candidate(
                "rank(group_rank(winsorize(ts_mean(analyst_positive_sentiment_logit_presentation,30),std=3),industry))",
                {"region": "USA", "delay": 0, "neutralization": "INDUSTRY"},
                "openai_compatible",
            )
            store.update_candidate(candidate_id, metrics_json=json.dumps({"sharpe": 1.41, "fitness": 0.52}))
            store.transition(candidate_id, "check_pending", {"errors": ["SHARPE_BELOW_MIN:1.410<1.58"]})

            context = build_ai_research_context(
                store,
                {"region": "USA", "universe": "TOP3000", "delay": 0, "neutralization": "INDUSTRY"},
                knowledge_dir=base / "missing-knowledge",
                reference_dir=base / "missing-reference",
            )

        plan = context["experiment_plan"]
        thresholds = plan["quality_thresholds"]
        self.assertEqual(thresholds["required_sharpe"], 2.69)
        self.assertEqual(thresholds["required_fitness"], 1.5)
        self.assertEqual(plan["mode"], "explore_new_family")
        self.assertIsNone(plan["target_candidate_id"])

    def test_build_ai_research_context_does_not_optimize_when_many_quality_checks_are_open(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            store = AlphaStore(base / "alpha.db")
            store.init()
            candidate_id = store.insert_candidate(
                "normalize(add(ts_mean(actual_update_flag_ebi,63),multiply(-0.35,ts_mean(actual_update_flag_prr,126))))",
                {"region": "USA", "universe": "TOP500", "delay": 0, "neutralization": "INDUSTRY"},
                "model:G-1",
            )
            store.update_candidate(
                candidate_id,
                metrics_json=json.dumps({"sharpe": 1.03, "fitness": 0.59, "turnover": 0.14}),
                checks_json=json.dumps(
                    {
                        "LOW_SHARPE": {"status": "FAIL", "value": 1.03, "limit": 2.69},
                        "LOW_FITNESS": {"status": "FAIL", "value": 0.59, "limit": 1.5},
                        "LOW_TURNOVER": {"status": "PASS", "value": 0.14, "limit": 0.01},
                        "HIGH_TURNOVER": {"status": "PASS", "value": 0.14, "limit": 0.7},
                        "CONCENTRATED_WEIGHT": {"status": "PASS"},
                        "LOW_2Y_SHARPE": {"status": "FAIL", "value": 0.22, "limit": 2.69},
                        "LOW_SUB_UNIVERSE_SHARPE": {"status": "FAIL", "value": 0.1, "limit": 0.49},
                        "IS_LADDER_SHARPE": {"status": "FAIL", "value": 0.3, "limit": 2.69},
                    }
                ),
            )
            store.transition(candidate_id, "failed", {"errors": ["LOW_SHARPE:FAIL", "LOW_FITNESS:FAIL"]})

            context = build_ai_research_context(
                store,
                {"region": "USA", "universe": "TOP500", "delay": 0, "neutralization": "INDUSTRY"},
                knowledge_dir=base / "missing-knowledge",
                reference_dir=base / "missing-reference",
            )

        plan = context["experiment_plan"]
        self.assertEqual(context["analysis"]["best_candidate"]["id"], candidate_id)
        self.assertEqual(plan["mode"], "explore_new_family")
        self.assertIsNone(plan["target_candidate_id"])
        self.assertEqual(context["candidate_queues"]["counts"]["optimize"], 0)
        self.assertEqual(context["candidate_queues"]["counts"]["trash"], 1)
        self.assertNotIn("actual_update_flag_ebi", plan["keep"])
        self.assertNotIn("actual_update_flag_prr", plan["keep"])

    def test_build_ai_research_context_does_not_keep_terminal_blocked_anchor_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            store = AlphaStore(base / "alpha.db")
            store.init()
            settings = {"region": "USA", "universe": "TOP500", "delay": 0, "neutralization": "INDUSTRY"}
            candidate_id = store.insert_candidate(
                "group_rank(ts_rank(terminal_signal,63),industry)",
                settings,
                "model:G-1",
            )
            store.update_candidate(
                candidate_id,
                metrics_json=json.dumps({"sharpe": 2.8, "fitness": 1.6, "turnover": 0.22}),
                checks_json=json.dumps(
                    {
                        "LOW_SHARPE": {"status": "PASS", "value": 2.8, "limit": 2.69},
                        "LOW_FITNESS": {"status": "PASS", "value": 1.6, "limit": 1.5},
                        "LOW_TURNOVER": {"status": "PASS", "value": 0.22, "limit": 0.01},
                        "HIGH_TURNOVER": {"status": "PASS", "value": 0.22, "limit": 0.7},
                        "CONCENTRATED_WEIGHT": {"status": "PASS"},
                        "LOW_2Y_SHARPE": {"status": "PASS", "value": 3.0, "limit": 2.69},
                        "LOW_SUB_UNIVERSE_SHARPE": {"status": "PASS", "value": 1.4, "limit": 0.5},
                        "PROD_CORRELATION": {"status": "FAIL", "value": 0.92, "limit": 0.7},
                    }
                ),
            )
            store.transition(candidate_id, "failed", {"errors": ["PROD_CORRELATION:FAIL"]})

            context = build_ai_research_context(
                store,
                settings,
                knowledge_dir=base / "missing-knowledge",
                reference_dir=base / "missing-reference",
                field_catalog={
                    "available": True,
                    "field_ids": ["terminal_signal", "fresh_signal"],
                    "fields": [
                        {
                            "id": "terminal_signal",
                            "type": "MATRIX",
                            "dataset_id": "model",
                            "category": "Model",
                            "coverage": 0.9,
                            "userCount": 0,
                            "alphaCount": 0,
                        },
                        {
                            "id": "fresh_signal",
                            "type": "MATRIX",
                            "dataset_id": "model",
                            "category": "Model",
                            "coverage": 0.9,
                            "userCount": 0,
                            "alphaCount": 0,
                        },
                    ],
                },
            )

        plan = context["experiment_plan"]
        self.assertEqual(plan["mode"], "explore_new_family")
        self.assertEqual(plan["optimization_gap_summary"]["reason"], "terminal_blocker_failed")
        self.assertNotIn("terminal_signal", plan["keep"])
        self.assertIn("terminal_signal", plan["avoid"])

    def test_build_ai_research_context_optimizes_when_only_one_or_two_quality_checks_are_open(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            store = AlphaStore(base / "alpha.db")
            store.init()
            candidate_id = store.insert_candidate(
                "group_rank(ts_rank(ts_backfill(close_signal,120),63),industry)",
                {"region": "USA", "universe": "TOP500", "delay": 0, "neutralization": "INDUSTRY"},
                "model:G-1",
            )
            store.update_candidate(
                candidate_id,
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
            store.transition(candidate_id, "failed", {"errors": ["LOW_SHARPE:FAIL", "LOW_FITNESS:FAIL"]})

            context = build_ai_research_context(
                store,
                {"region": "USA", "universe": "TOP500", "delay": 0, "neutralization": "INDUSTRY"},
                knowledge_dir=base / "missing-knowledge",
                reference_dir=base / "missing-reference",
            )

        plan = context["experiment_plan"]
        self.assertEqual(plan["mode"], "optimize_best")
        self.assertEqual(plan["target_candidate_id"], candidate_id)

    def test_build_ai_research_context_uses_setting_sweep_for_scope_ready_candidate(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            store = AlphaStore(base / "alpha.db")
            store.init()
            candidate_id = store.insert_candidate(
                "rank(group_rank(winsorize(ts_mean(analyst_positive_sentiment_logit_presentation,30),std=3),industry))",
                {"region": "USA", "delay": 0, "neutralization": "INDUSTRY"},
                "openai_compatible",
            )
            store.update_candidate(
                candidate_id,
                metrics_json=json.dumps({"sharpe": 2.55, "fitness": 1.32, "turnover": 0.22, "drawdown": 0.04}),
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

            context = build_ai_research_context(
                store,
                {"region": "USA", "universe": "TOP3000", "delay": 0, "neutralization": "INDUSTRY"},
                knowledge_dir=base / "missing-knowledge",
                reference_dir=base / "missing-reference",
            )

        plan = context["experiment_plan"]
        self.assertEqual(plan["mode"], "setting_sweep")
        self.assertEqual(plan["target_candidate_id"], candidate_id)
        self.assertEqual(len(plan["setting_variants"]), 8)
        self.assertTrue(any(item["neutralization"] == "SUBINDUSTRY" for item in plan["setting_variants"]))
        self.assertTrue(any(item["decay"] == 6 for item in plan["setting_variants"]))

    def test_build_ai_research_context_prefers_submission_quality_over_raw_sharpe(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            store = AlphaStore(base / "alpha.db")
            store.init()
            noisy_id = store.insert_candidate(
                "rank(multiply(group_rank(ts_rank(ts_backfill(noisy_signal,120),63),industry),"
                "group_rank(ts_rank(ts_backfill(risk_gate,120),63),industry)))",
                {"region": "USA", "universe": "TOP3000", "delay": 0},
                "openai_compatible",
            )
            store.update_candidate(
                noisy_id,
                metrics_json=json.dumps({"sharpe": 1.5, "fitness": 0.8, "turnover": 0.62, "drawdown": 0.18}),
                checks_json=json.dumps(
                    {
                        "LOW_SHARPE": {"status": "PASS", "value": 1.5, "limit": 1.58},
                        "LOW_FITNESS": {"status": "PASS", "value": 0.8, "limit": 1.0},
                        "HIGH_TURNOVER": {"status": "FAIL", "value": 0.62, "limit": 0.4},
                        "CONCENTRATED_WEIGHT": {"status": "FAIL"},
                        "IS_LADDER_SHARPE": {"status": "FAIL", "value": 0.8, "limit": 1.58},
                    }
                ),
            )
            store.transition(
                noisy_id,
                "check_pending",
                {"errors": ["HIGH_TURNOVER:FAIL", "CONCENTRATED_WEIGHT:FAIL", "IS_LADDER_SHARPE:FAIL"]},
            )
            cleaner_id = store.insert_candidate(
                "rank(add(group_rank(ts_rank(ts_backfill(clean_signal,120),63),industry),"
                "group_rank(ts_rank(ts_backfill(confirm_signal,120),63),industry)))",
                {"region": "USA", "universe": "TOP3000", "delay": 0},
                "openai_compatible",
            )
            store.update_candidate(
                cleaner_id,
                metrics_json=json.dumps({"sharpe": 1.45, "fitness": 0.72, "turnover": 0.24, "drawdown": 0.05}),
                checks_json=json.dumps(
                    {
                        "LOW_SHARPE": {"status": "PASS", "value": 1.45, "limit": 1.58},
                        "LOW_FITNESS": {"status": "PASS", "value": 0.72, "limit": 1.0},
                        "HIGH_TURNOVER": {"status": "PASS", "value": 0.24, "limit": 0.4},
                        "CONCENTRATED_WEIGHT": {"status": "PASS"},
                        "IS_LADDER_SHARPE": {"status": "PASS", "value": 2.0, "limit": 1.58},
                    }
                ),
            )
            store.transition(cleaner_id, "check_pending", {"errors": ["SELF_CORRELATION:PENDING"]})

            context = build_ai_research_context(
                store,
                {"region": "USA", "universe": "TOP3000", "delay": 0},
                knowledge_dir=base / "missing-knowledge",
                reference_dir=base / "missing-reference",
            )

        best = context["analysis"]["best_candidate"]
        self.assertEqual(best["id"], cleaner_id)
        self.assertGreater(best["quality_components"]["readiness_score"], 0)
        self.assertIn("quality_components", best)
        self.assertNotIn("CONCENTRATED_WEIGHT", best["quality_components"]["failed_checks"])

    def test_build_ai_research_context_scores_region_specific_returns_and_extra_checks(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            store = AlphaStore(base / "alpha.db")
            store.init()
            high_raw_id = store.insert_candidate(
                "rank(group_rank(ts_rank(ts_backfill(high_raw_signal,120),63),industry))",
                {"region": "CHN", "universe": "TOP2000U", "delay": 0},
                "openai_compatible",
            )
            store.update_candidate(
                high_raw_id,
                metrics_json=json.dumps({"sharpe": 3.7, "fitness": 1.6, "returns": 0.03, "turnover": 0.22}),
                checks_json=json.dumps(
                    {
                        "LOW_SHARPE": {"status": "PASS", "value": 3.7, "limit": 3.5},
                        "LOW_FITNESS": {"status": "PASS", "value": 1.6, "limit": 1.5},
                        "LOW_TURNOVER": {"status": "PASS", "value": 0.22, "limit": 0.01},
                        "HIGH_TURNOVER": {"status": "PASS", "value": 0.22, "limit": 0.7},
                        "ROBUST_UNIVERSE_RETENTION": {"status": "FAIL", "value": 0.2, "limit": 0.4},
                    }
                ),
            )
            store.transition(high_raw_id, "check_pending", {"errors": ["ROBUST_UNIVERSE_RETENTION:FAIL"]})
            balanced_id = store.insert_candidate(
                "rank(add(group_rank(ts_rank(ts_backfill(return_signal,120),63),industry),"
                "group_rank(ts_rank(ts_backfill(robust_signal,120),63),industry)))",
                {"region": "CHN", "universe": "TOP2000U", "delay": 0},
                "openai_compatible",
            )
            store.update_candidate(
                balanced_id,
                metrics_json=json.dumps({"sharpe": 3.52, "fitness": 1.51, "returns": 0.13, "turnover": 0.21}),
                checks_json=json.dumps(
                    {
                        "LOW_SHARPE": {"status": "PASS", "value": 3.52, "limit": 3.5},
                        "LOW_FITNESS": {"status": "PASS", "value": 1.51, "limit": 1.5},
                        "LOW_TURNOVER": {"status": "PASS", "value": 0.21, "limit": 0.01},
                        "HIGH_TURNOVER": {"status": "PASS", "value": 0.21, "limit": 0.7},
                        "ROBUST_UNIVERSE_RETENTION": {"status": "PASS", "value": 0.45, "limit": 0.4},
                    }
                ),
            )
            store.transition(balanced_id, "check_pending", {"errors": ["SELF_CORRELATION:PENDING"]})

            context = build_ai_research_context(
                store,
                {"region": "CHN", "universe": "TOP2000U", "delay": 0},
                knowledge_dir=base / "missing-knowledge",
                reference_dir=base / "missing-reference",
            )

        best = context["analysis"]["best_candidate"]
        thresholds = context["experiment_plan"]["quality_thresholds"]
        self.assertEqual(best["id"], balanced_id)
        self.assertEqual(thresholds["required_returns"], 0.12)
        self.assertEqual(thresholds["extra_checks"]["ROBUST_UNIVERSE_RETENTION"], 0.4)
        self.assertIn("extra:ROBUST_UNIVERSE_RETENTION", best["quality_components"]["component_scores"])

    def test_candidate_quality_separates_exploration_and_submission_scores(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            store = AlphaStore(base / "alpha.db")
            store.init()
            partial_id = store.insert_candidate(
                "rank(group_rank(ts_rank(ts_backfill(strong_signal,120),63),industry))",
                {"region": "USA", "universe": "TOP3000", "delay": 0},
                "openai_compatible",
            )
            store.update_candidate(
                partial_id,
                metrics_json=json.dumps({"sharpe": 2.72, "fitness": 1.55, "turnover": 0.22}),
                checks_json=json.dumps(
                    {
                        "LOW_SHARPE": {"status": "PASS", "value": 2.72, "limit": 2.69},
                        "LOW_FITNESS": {"status": "PASS", "value": 1.55, "limit": 1.5},
                        "LOW_TURNOVER": {"status": "PASS", "value": 0.22, "limit": 0.01},
                        "HIGH_TURNOVER": {"status": "PASS", "value": 0.22, "limit": 0.7},
                    }
                ),
            )
            store.transition(partial_id, "check_pending", {"errors": ["SELF_CORRELATION:PENDING"]})

            context = build_ai_research_context(
                store,
                {"region": "USA", "universe": "TOP3000", "delay": 0},
                knowledge_dir=base / "missing-knowledge",
                reference_dir=base / "missing-reference",
            )

        best = context["analysis"]["best_candidate"]
        components = best["quality_components"]
        self.assertEqual(best["id"], partial_id)
        self.assertGreater(components["exploration_score"], components["submission_score"])
        self.assertLess(components["submission_score"], components["readiness_score"])
        self.assertIn("SELF_CORRELATION", components["missing_submission_checks"])
        self.assertIn("DATA_DIVERSITY", components["missing_submission_checks"])
        self.assertNotEqual(context["experiment_plan"]["mode"], "setting_sweep")

    def test_build_ai_research_context_abandons_target_after_three_rounds_without_twenty_percent_gain(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            store = AlphaStore(base / "alpha.db")
            store.init()
            candidate_id = store.insert_candidate(
                "rank(group_rank(winsorize(ts_mean(weak_model_signal,30),std=3),industry))",
                {"region": "USA", "delay": 0},
                "openai_compatible",
            )
            store.update_candidate(candidate_id, metrics_json=json.dumps({"sharpe": 1.0, "fitness": 0.4}))
            store.transition(candidate_id, "check_pending", {"errors": ["SHARPE_BELOW_MIN:1.000<1.58"]})
            for round_number in range(1, 4):
                store.record_event(
                    None,
                    "experiment_plan",
                    {
                        "mode": "optimize_best",
                        "optimization_anchor_id": candidate_id,
                        "target_candidate_id": candidate_id,
                        "baseline_score": 1.2,
                        "optimize_round": round_number,
                    },
                )

            context = build_ai_research_context(
                store,
                {"region": "USA", "universe": "TOP3000", "delay": 0},
                knowledge_dir=base / "missing-knowledge",
                reference_dir=base / "missing-reference",
            )

        plan = context["experiment_plan"]
        self.assertEqual(plan["mode"], "explore_new_family")
        self.assertEqual(plan["abandoned_target_id"], candidate_id)
        self.assertEqual(plan["abandon_reason"], "NO_20_PERCENT_IMPROVEMENT_AFTER_3_ROUNDS")

    def test_build_ai_research_context_does_not_reuse_unrelated_old_optimization_anchor(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            store = AlphaStore(base / "alpha.db")
            store.init()
            old_id = store.insert_candidate(
                "rank(group_rank(ts_mean(old_signal,30),industry))",
                {"region": "USA", "universe": "TOP3000", "delay": 0},
                "model:G-1",
            )
            store.update_candidate(old_id, metrics_json=json.dumps({"sharpe": 1.0, "fitness": 0.4}))
            store.transition(old_id, "failed", {"errors": ["LOW_SHARPE:FAIL"]})
            new_id = store.insert_candidate(
                "rank(group_rank(ts_rank(ts_backfill(new_signal,120),63),industry))",
                {"region": "USA", "universe": "TOP3000", "delay": 0},
                "model:G-2",
            )
            store.update_candidate(new_id, metrics_json=json.dumps({"sharpe": 1.7, "fitness": 0.7, "turnover": 0.2}))
            store.transition(new_id, "check_pending", {"errors": ["SELF_CORRELATION:PENDING"]})
            store.record_event(
                None,
                "experiment_plan",
                {
                    "mode": "optimize_best",
                    "optimization_anchor_id": old_id,
                    "target_candidate_id": old_id,
                    "baseline_score": 1.2,
                    "optimize_round": 1,
                    "target_settings": {"region": "USA", "universe": "TOP3000", "delay": 0},
                },
            )

            context = build_ai_research_context(
                store,
                {"region": "USA", "universe": "TOP3000", "delay": 0},
                knowledge_dir=base / "missing-knowledge",
                reference_dir=base / "missing-reference",
            )

        plan = context["experiment_plan"]
        self.assertEqual(plan["target_candidate_id"], new_id)
        self.assertEqual(plan["optimization_anchor_id"], new_id)
        self.assertEqual(plan["optimize_round"], 1)

    def test_build_ai_research_context_adds_scope_candidate_queues(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            store = AlphaStore(base / "alpha.db")
            store.init()
            core_pass = {
                "LOW_SHARPE": {"status": "PASS", "value": 2.8, "limit": 2.69},
                "LOW_FITNESS": {"status": "PASS", "value": 1.6, "limit": 1.5},
                "LOW_TURNOVER": {"status": "PASS", "value": 0.2, "limit": 0.01},
                "HIGH_TURNOVER": {"status": "PASS", "value": 0.2, "limit": 0.7},
                "CONCENTRATED_WEIGHT": {"status": "PASS"},
                "LOW_SUB_UNIVERSE_SHARPE": {"status": "PASS", "value": 1.5, "limit": 1.0},
                "IS_LADDER_SHARPE": {"status": "PASS", "value": 2.8, "limit": 2.69},
            }
            submitted_id = store.insert_candidate(
                "rank(group_rank(ts_rank(ts_backfill(submit_signal,120),63),industry))",
                {"region": "USA", "universe": "TOP3000", "delay": 0},
                "model:G-1",
            )
            store.update_candidate(
                submitted_id,
                metrics_json=json.dumps({"sharpe": 2.82, "fitness": 1.62, "turnover": 0.2}),
                checks_json=json.dumps(core_pass),
            )
            store.transition(submitted_id, "approved")
            watchlist_id = store.insert_candidate(
                "rank(group_rank(ts_rank(ts_backfill(wait_signal,120),63),industry))",
                {"region": "USA", "universe": "TOP3000", "delay": 0},
                "model:G-2",
            )
            store.update_candidate(
                watchlist_id,
                metrics_json=json.dumps({"sharpe": 2.8, "fitness": 1.6, "turnover": 0.2}),
                checks_json=json.dumps(
                    {
                        **core_pass,
                        "SELF_CORRELATION": {"status": "PENDING"},
                        "PROD_CORRELATION": {"status": "PENDING"},
                    }
                ),
            )
            store.transition(watchlist_id, "check_pending", {"errors": ["SELF_CORRELATION:PENDING"]})
            optimize_id = store.insert_candidate(
                "rank(group_rank(ts_rank(ts_backfill(close_signal,120),63),industry))",
                {"region": "USA", "universe": "TOP3000", "delay": 0},
                "model:G-1",
            )
            store.update_candidate(
                optimize_id,
                metrics_json=json.dumps(
                    {
                        "sharpe": 2.55,
                        "fitness": 1.32,
                        "turnover": 0.22,
                        "drawdown": 0.04,
                        "checks": [{"name": "LOW_SHARPE", "result": "FAIL"}],
                        "investabilityConstrained": {"sharpe": 2.4, "fitness": 1.2},
                    }
                ),
                checks_json=json.dumps(
                    {
                        **core_pass,
                        "LOW_SHARPE": {"status": "FAIL", "value": 2.55, "limit": 2.69},
                        "LOW_FITNESS": {"status": "FAIL", "value": 1.32, "limit": 1.5},
                    }
                ),
            )
            store.transition(optimize_id, "failed", {"errors": ["LOW_SHARPE:FAIL", "LOW_FITNESS:FAIL"]})
            trash_id = store.insert_candidate(
                "rank(close)",
                {"region": "USA", "universe": "TOP3000", "delay": 0},
                "model:G-2",
            )
            store.update_candidate(
                trash_id,
                metrics_json=json.dumps({"sharpe": -3.0, "fitness": -1.0, "turnover": 0.9}),
                checks_json=json.dumps(
                    {
                        "LOW_SHARPE": {"status": "FAIL", "value": -3.0, "limit": 2.69},
                        "LOW_FITNESS": {"status": "FAIL", "value": -1.0, "limit": 1.5},
                        "HIGH_TURNOVER": {"status": "FAIL", "value": 0.9, "limit": 0.7},
                    }
                ),
            )
            store.transition(trash_id, "failed", {"errors": ["LOW_SHARPE:FAIL", "HIGH_TURNOVER:FAIL"]})
            other_scope_id = store.insert_candidate(
                "rank(group_rank(ts_rank(ts_backfill(other_scope_signal,120),63),industry))",
                {"region": "CHN", "universe": "TOP2000U", "delay": 0},
                "model:G-1",
            )
            store.update_candidate(other_scope_id, metrics_json=json.dumps({"sharpe": 4.0, "fitness": 2.0}))
            store.transition(other_scope_id, "approved")

            context = build_ai_research_context(
                store,
                {"region": "USA", "universe": "TOP3000", "delay": 0},
                knowledge_dir=base / "missing-knowledge",
                reference_dir=base / "missing-reference",
                history_limit=12,
            )

        queues = context["candidate_queues"]
        self.assertEqual([item["id"] for item in queues["submitable"]], [submitted_id])
        self.assertEqual([item["id"] for item in queues["watchlist"]], [watchlist_id])
        self.assertEqual([item["id"] for item in queues["optimize"]], [optimize_id])
        self.assertEqual([item["id"] for item in queues["trash"]], [trash_id])
        self.assertEqual(queues["counts"]["submitable"], 1)
        self.assertEqual(context["analysis"]["candidate_queue_counts"]["optimize"], 1)
        self.assertIn("LOW_SHARPE", queues["optimize"][0]["failed_checks"])
        self.assertNotIn("checks", queues["optimize"][0]["metrics"])
        self.assertNotIn("investabilityConstrained", queues["optimize"][0]["metrics"])
        self.assertEqual(queues["watchlist"][0]["queue_reason"], "terminal_checks_waiting")

    def test_build_ai_research_context_adds_route_stop_loss_for_stagnant_scope(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            store = AlphaStore(base / "alpha.db")
            store.init()
            settings = {"region": "MEA", "universe": "TOP300", "delay": 1, "neutralization": "MARKET"}
            for idx in range(140):
                candidate_id = store.insert_candidate(
                    f"group_rank(pasteurize(normalize(quantile(stale_signal_{idx}))),industry)",
                    settings,
                    "model:G-1" if idx % 2 == 0 else "model:G-2",
                )
                store.update_candidate(
                    candidate_id,
                    metrics_json=json.dumps({"sharpe": 0.05 + (idx % 5) * 0.01, "fitness": 0.02, "turnover": 0.2}),
                    checks_json=json.dumps(
                        {
                            "LOW_SHARPE": {"status": "FAIL", "value": 0.08, "limit": 1.58},
                            "LOW_FITNESS": {"status": "FAIL", "value": 0.02, "limit": 1.0},
                            "LOW_2Y_SHARPE": {"status": "FAIL", "value": 0.04, "limit": 1.58},
                        }
                    ),
                )
                store.transition(candidate_id, "failed", {"errors": ["LOW_SHARPE:FAIL", "LOW_FITNESS:FAIL"]})

            context = build_ai_research_context(
                store,
                settings,
                knowledge_dir=base / "missing-knowledge",
                reference_dir=base / "missing-reference",
                field_catalog={"available": True, "field_ids": [f"stale_signal_{idx}" for idx in range(140)], "fields": []},
            )

        route = context["analysis"]["route_efficiency"]
        plan = context["experiment_plan"]
        self.assertTrue(route["stop_loss_active"])
        self.assertGreaterEqual(route["scanned_candidates"], 120)
        self.assertEqual(route["watchlist_count"], 0)
        self.assertEqual(plan["route_stop_loss"]["active"], True)
        self.assertEqual(plan["structure_diversity_control"]["max_batch_candidates_per_structure"], 2)
        self.assertTrue(plan["structure_diversity_control"]["overused_structures"])
        self.assertIn("replace overused formula skeletons", plan["change"])

    def test_build_ai_research_context_does_not_stop_loss_when_watchlist_exists(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            store = AlphaStore(base / "alpha.db")
            store.init()
            settings = {"region": "MEA", "universe": "TOP300", "delay": 1, "neutralization": "MARKET"}
            for idx in range(120):
                candidate_id = store.insert_candidate(
                    f"group_rank(pasteurize(normalize(quantile(weak_signal_{idx}))),industry)",
                    settings,
                    "model:G-1",
                )
                store.update_candidate(candidate_id, metrics_json=json.dumps({"sharpe": 0.05, "fitness": 0.02}))
                store.transition(candidate_id, "failed", {"errors": ["LOW_SHARPE:FAIL"]})
            watch_id = store.insert_candidate(
                "group_rank(ts_rank(ts_backfill(wait_signal,120),63),industry)",
                settings,
                "model:G-2",
            )
            store.update_candidate(
                watch_id,
                metrics_json=json.dumps({"sharpe": 1.7, "fitness": 1.1, "turnover": 0.2}),
                checks_json=json.dumps(
                    {
                        "LOW_SHARPE": {"status": "PASS", "value": 1.7, "limit": 1.58},
                        "LOW_FITNESS": {"status": "PASS", "value": 1.1, "limit": 1.0},
                        "SELF_CORRELATION": {"status": "PENDING"},
                        "PROD_CORRELATION": {"status": "PENDING"},
                    }
                ),
            )
            store.transition(watch_id, "check_pending", {"errors": ["SELF_CORRELATION:PENDING"]})

            context = build_ai_research_context(
                store,
                settings,
                knowledge_dir=base / "missing-knowledge",
                reference_dir=base / "missing-reference",
            )

        route = context["analysis"]["route_efficiency"]
        self.assertEqual(route["watchlist_count"], 1)
        self.assertFalse(route["stop_loss_active"])

    def test_scope_trouble_drops_lit_model_fallback_when_no_safe_rescue_probe_remains(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            store = AlphaStore(base / "alpha.db")
            store.init()
            settings = {"region": "USA", "universe": "TOP500", "delay": 0, "neutralization": "INDUSTRY"}
            for idx in range(125):
                candidate_id = store.insert_candidate(f"rank(dead_route_signal_{idx})", settings, "model:G-1")
                store.update_candidate(
                    candidate_id,
                    metrics_json=json.dumps({"sharpe": 0.01, "fitness": 0.0}),
                    checks_json=json.dumps({"LOW_SHARPE": {"status": "FAIL", "value": 0.01, "limit": 2.69}}),
                )
                store.transition(candidate_id, "failed", {"errors": ["LOW_SHARPE:FAIL"]})

            context = build_ai_research_context(
                store,
                settings,
                knowledge_dir=base / "missing-knowledge",
                reference_dir=base / "missing-reference",
                field_catalog={
                    "available": True,
                    "field_ids": ["model_quality_signal"],
                    "fields": [
                        {
                            "id": "model_quality_signal",
                            "type": "MATRIX",
                            "dataset_id": "model37",
                            "category": "Model",
                            "coverage": 0.9,
                            "userCount": 0,
                            "alphaCount": 0,
                            "pyramidMultiplier": 1.7,
                        }
                    ],
                },
                platform_pyramid_alphas={
                    "pyramids": [
                        {
                            "category": {"id": "model", "name": "Model"},
                            "region": "USA",
                            "delay": 0,
                            "alphaCount": 7,
                        }
                    ]
                },
            )

        scout = context["field_scout"]
        self.assertEqual(scout["status"], "no_primary_fields")
        self.assertEqual(scout["top_primary_fields"], [])
        self.assertFalse(context["experiment_plan"]["production_rescue"]["active"])
        self.assertEqual(context["experiment_plan"]["production_rescue"]["reason"], "no_safe_probe_recommendations")
        self.assertEqual(context["experiment_plan"]["field_scout"]["status"], "no_primary_fields")

    def test_scope_trouble_does_not_keep_lit_fallback_when_rescue_templates_are_blocked(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            store = AlphaStore(base / "alpha.db")
            store.init()
            settings = {"region": "USA", "universe": "TOP500", "delay": 0, "neutralization": "INDUSTRY"}
            blocked_templates = [
                "group_rank(ts_rank(winsorize(ts_backfill(structure_trap_signal, 120), std=4), 63), industry)",
                "group_rank(ts_rank(divide(winsorize(ts_backfill(structure_trap_signal, 120), std=4), cap), 63), industry)",
                "rank(ts_decay_linear(ts_backfill(structure_trap_signal, 120), 20))",
                "rank(multiply(-1, ts_rank(winsorize(ts_backfill(structure_trap_signal, 120), std=4), 33)))",
            ]
            expressions = []
            for template in blocked_templates:
                expressions.extend([template] * 6)
            expressions.extend(f"rank(dead_route_signal_{idx})" for idx in range(101))
            for expression in expressions:
                candidate_id = store.insert_candidate(expression, settings, "model:G-1")
                store.update_candidate(
                    candidate_id,
                    metrics_json=json.dumps({"sharpe": 0.01, "fitness": 0.0}),
                    checks_json=json.dumps({"LOW_SHARPE": {"status": "FAIL", "value": 0.01, "limit": 2.69}}),
                )
                store.transition(candidate_id, "failed", {"errors": ["LOW_SHARPE:FAIL"]})

            context = build_ai_research_context(
                store,
                settings,
                knowledge_dir=base / "missing-knowledge",
                reference_dir=base / "missing-reference",
                field_catalog={
                    "available": True,
                    "field_ids": ["model_quality_signal"],
                    "fields": [
                        {
                            "id": "model_quality_signal",
                            "type": "MATRIX",
                            "dataset_id": "model37",
                            "category": "Model",
                            "coverage": 0.9,
                            "userCount": 0,
                            "alphaCount": 0,
                            "pyramidMultiplier": 1.7,
                        }
                    ],
                },
                platform_pyramid_alphas={
                    "pyramids": [
                        {
                            "category": {"id": "model", "name": "Model"},
                            "region": "USA",
                            "delay": 0,
                            "alphaCount": 7,
                        }
                    ]
                },
            )

        scout = context["field_scout"]
        self.assertEqual(scout["status"], "no_primary_fields")
        self.assertEqual(scout["top_primary_fields"], [])
        self.assertFalse(context["experiment_plan"]["production_rescue"]["active"])
        self.assertEqual(context["experiment_plan"]["field_scout"]["status"], "no_primary_fields")

    def test_build_ai_research_context_suppresses_low_quality_failure_expressions(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            store = AlphaStore(base / "alpha.db")
            store.init()
            settings = {"region": "USA", "universe": "TOP3000", "delay": 0, "neutralization": "INDUSTRY"}
            candidate_id = store.insert_candidate("rank(template_trap_signal)", settings, "model:G-1")
            store.update_candidate(
                candidate_id,
                metrics_json=json.dumps({"sharpe": 0.0, "fitness": 0.0, "turnover": 0.2}),
                checks_json=json.dumps({"LOW_SHARPE": {"status": "FAIL", "value": 0.0, "limit": 2.69}}),
            )
            store.transition(candidate_id, "failed", {"errors": ["LOW_SHARPE:FAIL"]})

            context = build_ai_research_context(
                store,
                settings,
                knowledge_dir=base / "missing-knowledge",
                reference_dir=base / "missing-reference",
                field_catalog={"available": True, "field_ids": ["template_trap_signal"], "fields": []},
            )

        self.assertEqual(context["recent_failures"], [])
        self.assertEqual(context["history_hygiene"]["suppressed_low_quality_failures"], 1)
        self.assertEqual(context["history_hygiene"]["low_quality_score_max"], 0.2)

    def test_route_efficiency_uses_active_run_progress_not_old_submitable_candidates(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            store = AlphaStore(base / "alpha.db")
            store.init()
            settings = {"region": "MEA", "universe": "TOP300", "delay": 1, "neutralization": "MARKET"}
            approved_id = store.insert_candidate("rank(old_good_signal)", settings, "model:G-1")
            store.update_candidate(approved_id, metrics_json=json.dumps({"sharpe": 2.0, "fitness": 1.2}))
            store.transition(approved_id, "approved")
            with store.connection() as conn:
                conn.execute(
                    "UPDATE candidates SET created_at = ?, updated_at = ? WHERE id = ?",
                    ("2026-05-19T00:00:00+00:00", "2026-05-19T00:00:00+00:00", approved_id),
                )
            store.set_run_state(
                "daemon",
                {
                    "status": "stopped",
                    "started_at": "2026-05-19T12:00:00+00:00",
                    "scope": settings,
                },
            )
            for idx in range(125):
                candidate_id = store.insert_candidate(f"rank(dead_signal_{idx})", settings, "model:G-2")
                store.update_candidate(
                    candidate_id,
                    metrics_json=json.dumps({"sharpe": 0.01, "fitness": 0.0}),
                    checks_json=json.dumps({"LOW_SHARPE": {"status": "FAIL", "value": 0.01, "limit": 1.58}}),
                )
                store.transition(candidate_id, "failed", {"errors": ["LOW_SHARPE:FAIL"]})

            context = build_ai_research_context(
                store,
                settings,
                knowledge_dir=base / "missing-knowledge",
                reference_dir=base / "missing-reference",
                field_catalog={"available": True, "field_ids": [f"dead_signal_{idx}" for idx in range(125)], "fields": []},
            )

        self.assertEqual(context["candidate_queues"]["counts"]["submitable"], 1)
        self.assertEqual(context["active_run_candidate_queues"]["counts"]["submitable"], 0)
        self.assertEqual(context["analysis"]["route_efficiency"]["submitable_count"], 0)
        self.assertTrue(context["analysis"]["route_efficiency"]["stop_loss_active"])


if __name__ == "__main__":
    unittest.main()
