from __future__ import annotations

import unittest

from alpha.field_scout import build_field_scout


class FieldScoutTests(unittest.TestCase):
    def test_field_scout_prioritizes_underused_high_coverage_unexplored_fields(self):
        catalog = {
            "available": True,
            "fields": [
                {
                    "id": "crowded_signal",
                    "type": "MATRIX",
                    "dataset_id": "model",
                    "category": "Model",
                    "coverage": 0.95,
                    "userCount": 80,
                    "alphaCount": 40,
                    "pyramidMultiplier": 1.2,
                },
                {
                    "id": "rare_signal",
                    "type": "MATRIX",
                    "dataset_id": "news",
                    "category": "News",
                    "coverage": 0.82,
                    "userCount": 0,
                    "alphaCount": 0,
                    "pyramidMultiplier": 1.8,
                },
                {
                    "id": "failed_signal",
                    "type": "MATRIX",
                    "dataset_id": "sentiment",
                    "category": "Sentiment",
                    "coverage": 0.9,
                    "userCount": 0,
                    "alphaCount": 0,
                    "pyramidMultiplier": 1.8,
                },
            ],
        }
        history_memory = {
            "top_fields": [
                {"field": "failed_signal", "count": 10, "failed": 10, "best_sharpe": 0.1, "best_quality_score": 0.0},
            ]
        }
        lit_tower_avoidance = {
            "lit_towers": [{"category": "MODEL"}],
            "unlit_towers": [{"category": "NEWS"}, {"category": "SENTIMENT"}],
        }

        scout = build_field_scout(
            catalog,
            history_memory=history_memory,
            submitted_avoidance={"fields": []},
            lit_tower_avoidance=lit_tower_avoidance,
        )

        top_fields = [row["field"] for row in scout["top_fields"]]
        self.assertEqual(top_fields[0], "rare_signal")
        self.assertLess(top_fields.index("failed_signal"), len(top_fields))
        self.assertGreater(scout["top_fields"][0]["score"], scout["top_fields"][-1]["score"])
        bucket_names = [bucket["name"] for bucket in scout["buckets"]]
        self.assertIn("high_opportunity_unexplored", bucket_names)
        unexplored = next(bucket for bucket in scout["buckets"] if bucket["name"] == "high_opportunity_unexplored")
        self.assertIn("rare_signal", unexplored["fields"])

    def test_field_scout_uses_full_field_stats_not_only_top_field_summary(self):
        scout = build_field_scout(
            {
                "available": True,
                "fields": [
                    {
                        "id": "recent_failed_signal",
                        "type": "MATRIX",
                        "dataset_id": "analyst10",
                        "category": "Analyst",
                        "coverage": 0.95,
                        "userCount": 0,
                        "alphaCount": 0,
                        "pyramidMultiplier": 1.8,
                    },
                    {
                        "id": "fresh_signal",
                        "type": "MATRIX",
                        "dataset_id": "analyst10",
                        "category": "Analyst",
                        "coverage": 0.95,
                        "userCount": 0,
                        "alphaCount": 0,
                        "pyramidMultiplier": 1.8,
                    },
                ],
            },
            history_memory={
                "top_fields": [],
                "field_stats": [
                    {
                        "field": "recent_failed_signal",
                        "count": 5,
                        "failed": 5,
                        "best_sharpe": 0.1,
                        "best_fitness": 0.02,
                        "best_quality_score": -0.5,
                    }
                ],
            },
        )

        rows = {row["field"]: row for row in scout["top_fields"]}
        self.assertEqual(rows["recent_failed_signal"]["explored_count"], 5)
        self.assertEqual(rows["recent_failed_signal"]["failed_count"], 5)
        bucket = next(bucket for bucket in scout["buckets"] if bucket["name"] == "high_opportunity_unexplored")
        self.assertIn("fresh_signal", bucket["fields"])
        self.assertNotIn("recent_failed_signal", bucket["fields"])

    def test_field_scout_penalizes_new_fields_from_recently_failed_dataset(self):
        scout = build_field_scout(
            {
                "available": True,
                "fields": [
                    {
                        "id": "snt23_new_signal",
                        "type": "MATRIX",
                        "dataset_id": "sentiment23",
                        "category": "Sentiment",
                        "coverage": 1.0,
                        "userCount": 0,
                        "alphaCount": 0,
                        "pyramidMultiplier": 1.8,
                    },
                    {
                        "id": "anl10_new_signal",
                        "type": "MATRIX",
                        "dataset_id": "analyst10",
                        "category": "Analyst",
                        "coverage": 1.0,
                        "userCount": 0,
                        "alphaCount": 0,
                        "pyramidMultiplier": 1.8,
                    },
                ],
            },
            history_memory={
                "top_field_datasets": [
                    {
                        "dataset_id": "sentiment23",
                        "count": 12,
                        "failed": 12,
                        "best_sharpe": 0.66,
                        "best_fitness": 0.11,
                    }
                ]
            },
        )

        rows = {row["field"]: row for row in scout["top_fields"]}
        self.assertEqual(rows["snt23_new_signal"]["dataset_failed_count"], 12)
        self.assertLess(rows["snt23_new_signal"]["score"], rows["anl10_new_signal"]["score"])
        bucket = next(bucket for bucket in scout["buckets"] if bucket["name"] == "high_opportunity_unexplored")
        self.assertIn("anl10_new_signal", bucket["fields"])
        self.assertNotIn("snt23_new_signal", bucket["fields"])

    def test_field_scout_does_not_block_whole_dataset_from_single_failed_field(self):
        scout = build_field_scout(
            {
                "available": True,
                "fields": [
                    {
                        "id": "snt23_new_breadth_signal",
                        "type": "MATRIX",
                        "dataset_id": "sentiment23",
                        "category": "Sentiment",
                        "coverage": 1.0,
                        "userCount": 0,
                        "alphaCount": 0,
                        "pyramidMultiplier": 1.8,
                    },
                    {
                        "id": "anl10_new_signal",
                        "type": "MATRIX",
                        "dataset_id": "analyst10",
                        "category": "Analyst",
                        "coverage": 0.9,
                        "userCount": 0,
                        "alphaCount": 0,
                        "pyramidMultiplier": 1.4,
                    },
                ],
            },
            history_memory={
                "top_field_datasets": [
                    {
                        "dataset_id": "sentiment23",
                        "count": 12,
                        "failed": 12,
                        "distinct_field_count": 1,
                        "best_sharpe": 0.2,
                        "best_fitness": 0.05,
                    }
                ]
            },
        )

        rows = {row["field"]: row for row in scout["top_fields"]}
        self.assertEqual(rows["snt23_new_breadth_signal"]["dataset_failed_count"], 12)
        self.assertEqual(rows["snt23_new_breadth_signal"]["dataset_reason"], "")
        self.assertEqual(rows["snt23_new_breadth_signal"]["primary_policy"], "prefer_primary")

    def test_field_scout_blocks_dataset_after_multiple_distinct_failed_fields(self):
        scout = build_field_scout(
            {
                "available": True,
                "fields": [
                    {
                        "id": "snt23_new_breadth_signal",
                        "type": "MATRIX",
                        "dataset_id": "sentiment23",
                        "category": "Sentiment",
                        "coverage": 1.0,
                        "userCount": 0,
                        "alphaCount": 0,
                        "pyramidMultiplier": 1.8,
                    },
                    {
                        "id": "anl10_new_signal",
                        "type": "MATRIX",
                        "dataset_id": "analyst10",
                        "category": "Analyst",
                        "coverage": 0.9,
                        "userCount": 0,
                        "alphaCount": 0,
                        "pyramidMultiplier": 1.4,
                    },
                ],
            },
            history_memory={
                "top_field_datasets": [
                    {
                        "dataset_id": "sentiment23",
                        "count": 12,
                        "failed": 12,
                        "distinct_field_count": 4,
                        "best_sharpe": 0.2,
                        "best_fitness": 0.05,
                    }
                ]
            },
        )

        rows = {row["field"]: row for row in scout["top_fields"]}
        self.assertEqual(rows["snt23_new_breadth_signal"]["dataset_reason"], "recent_dataset_failure_cluster")
        self.assertEqual(rows["snt23_new_breadth_signal"]["primary_policy"], "avoid_primary")

    def test_field_scout_penalizes_explored_fields_from_failed_dataset_without_positive_evidence(self):
        scout = build_field_scout(
            {
                "available": True,
                "fields": [
                    {
                        "id": "snt23_recent_preflight_only",
                        "type": "MATRIX",
                        "dataset_id": "sentiment23",
                        "category": "Sentiment",
                        "coverage": 1.0,
                        "userCount": 0,
                        "alphaCount": 0,
                        "pyramidMultiplier": 1.8,
                    },
                    {
                        "id": "anl10_fresh_signal",
                        "type": "MATRIX",
                        "dataset_id": "analyst10",
                        "category": "Analyst",
                        "coverage": 1.0,
                        "userCount": 0,
                        "alphaCount": 0,
                        "pyramidMultiplier": 1.8,
                    },
                ],
            },
            history_memory={
                "field_stats": [
                    {
                        "field": "snt23_recent_preflight_only",
                        "count": 1,
                        "failed": 0,
                        "best_sharpe": None,
                        "best_fitness": None,
                    }
                ],
                "top_field_datasets": [
                    {
                        "dataset_id": "sentiment23",
                        "count": 18,
                        "failed": 16,
                        "best_sharpe": 0.31,
                        "best_fitness": 0.05,
                    }
                ],
            },
        )

        rows = {row["field"]: row for row in scout["top_fields"]}
        self.assertEqual(rows["snt23_recent_preflight_only"]["primary_policy"], "avoid_primary")
        self.assertEqual(rows["snt23_recent_preflight_only"]["dataset_reason"], "recent_dataset_failure_cluster")
        bucket_fields = {
            field
            for bucket in scout["buckets"]
            if bucket["name"] in {"high_opportunity_unexplored", "low_user_high_coverage", "high_multiplier_unlit_tower"}
            for field in bucket["fields"]
        }
        self.assertIn("anl10_fresh_signal", bucket_fields)
        self.assertNotIn("snt23_recent_preflight_only", bucket_fields)

    def test_field_scout_marks_repeatedly_failed_field_without_positive_evidence_as_avoid_primary(self):
        scout = build_field_scout(
            {
                "available": True,
                "fields": [
                    {
                        "id": "repeated_dead_signal",
                        "type": "MATRIX",
                        "dataset_id": "news5",
                        "category": "News",
                        "coverage": 1.0,
                        "userCount": 0,
                        "alphaCount": 0,
                        "pyramidMultiplier": 1.7,
                    },
                    {
                        "id": "fresh_news_signal",
                        "type": "MATRIX",
                        "dataset_id": "news12",
                        "category": "News",
                        "coverage": 0.7,
                        "userCount": 0,
                        "alphaCount": 0,
                        "pyramidMultiplier": 1.7,
                    },
                ],
            },
            history_memory={
                "field_stats": [
                    {
                        "field": "repeated_dead_signal",
                        "count": 3,
                        "failed": 3,
                        "best_sharpe": 0.35,
                        "best_fitness": 0.19,
                    }
                ],
            },
        )

        rows = {row["field"]: row for row in scout["top_fields"]}
        self.assertEqual(rows["repeated_dead_signal"]["primary_policy"], "avoid_primary")
        self.assertEqual(rows["repeated_dead_signal"]["field_reason"], "recent_field_failure_cluster")
        self.assertEqual(scout["top_primary_fields"][0]["field"], "fresh_news_signal")
        for bucket in scout["buckets"]:
            self.assertNotIn("repeated_dead_signal", bucket["fields"])

    def test_field_scout_cools_down_single_strong_negative_quality_field(self):
        scout = build_field_scout(
            {
                "available": True,
                "fields": [
                    {
                        "id": "aggregate_open_positions_count",
                        "type": "VECTOR",
                        "dataset_id": "hiring_trends",
                        "category": "Other",
                        "coverage": 0.9,
                        "userCount": 0,
                        "alphaCount": 0,
                        "pyramidMultiplier": 1.8,
                    },
                    {
                        "id": "standardized_opinion_score",
                        "type": "VECTOR",
                        "dataset_id": "social_sent_score",
                        "category": "Other",
                        "coverage": 0.75,
                        "userCount": 1,
                        "alphaCount": 1,
                        "pyramidMultiplier": 1.8,
                    },
                ],
            },
            history_memory={
                "field_stats": [
                    {
                        "field": "aggregate_open_positions_count",
                        "count": 1,
                        "failed": 1,
                        "best_sharpe": -0.09,
                        "best_fitness": -0.03,
                        "avg_sharpe": -0.09,
                        "avg_fitness": -0.03,
                    },
                    {
                        "field": "standardized_opinion_score",
                        "count": 4,
                        "failed": 4,
                        "best_sharpe": 0.67,
                        "best_fitness": 0.43,
                        "avg_sharpe": 0.21,
                        "avg_fitness": 0.08,
                    },
                ],
            },
        )

        rows = {row["field"]: row for row in scout["top_fields"]}
        self.assertEqual(rows["aggregate_open_positions_count"]["primary_policy"], "avoid_primary")
        self.assertEqual(rows["aggregate_open_positions_count"]["field_reason"], "recent_negative_quality")
        self.assertEqual(scout["top_primary_fields"][0]["field"], "standardized_opinion_score")

    def test_field_scout_cools_down_repeated_low_ceiling_failures_despite_modest_fitness(self):
        scout = build_field_scout(
            {
                "available": True,
                "fields": [
                    {
                        "id": "standardized_opinion_score",
                        "type": "VECTOR",
                        "dataset_id": "social_sent_score",
                        "category": "Other",
                        "coverage": 0.75,
                        "userCount": 1,
                        "alphaCount": 1,
                        "pyramidMultiplier": 1.8,
                    },
                    {
                        "id": "fresh_sentiment_route",
                        "type": "MATRIX",
                        "dataset_id": "sentiment23",
                        "category": "Sentiment",
                        "coverage": 0.9,
                        "userCount": 0,
                        "alphaCount": 0,
                        "pyramidMultiplier": 1.8,
                    },
                ],
            },
            history_memory={
                "field_stats": [
                    {
                        "field": "standardized_opinion_score",
                        "count": 7,
                        "failed": 7,
                        "best_sharpe": 0.73,
                        "best_fitness": 0.46,
                        "avg_sharpe": 0.48,
                        "avg_fitness": 0.22,
                    },
                ],
            },
        )

        rows = {row["field"]: row for row in scout["top_fields"]}
        self.assertEqual(rows["standardized_opinion_score"]["primary_policy"], "avoid_primary")
        self.assertEqual(rows["standardized_opinion_score"]["field_reason"], "repeated_low_ceiling_failures")
        self.assertEqual(scout["top_primary_fields"][0]["field"], "fresh_sentiment_route")

    def test_field_scout_cools_down_standardized_probe_exhausted_field(self):
        scout = build_field_scout(
            {
                "available": True,
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
            history_memory={
                "probe_exhausted_fields": [
                    {
                        "field": "aggregate_sentiment_score_3",
                        "count": 1,
                        "reason": "all_standardized_probe_templates_duplicate",
                    }
                ]
            },
        )

        rows = {row["field"]: row for row in scout["top_fields"]}
        self.assertEqual(rows["aggregate_sentiment_score_3"]["primary_policy"], "avoid_primary")
        self.assertEqual(rows["aggregate_sentiment_score_3"]["field_reason"], "standardized_probe_exhausted")
        self.assertEqual(scout["top_primary_fields"][0]["field"], "fresh_other_signal")

    def test_field_scout_cools_down_production_rescue_probe_exhausted_field(self):
        scout = build_field_scout(
            {
                "available": True,
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
            history_memory={
                "probe_exhausted_fields": [
                    {
                        "field": "snt21_4neut_conf_low",
                        "count": 1,
                        "reason": "all_production_rescue_probe_templates_duplicate",
                        "event_type": "production_rescue_probe_exhausted",
                    }
                ]
            },
        )

        rows = {row["field"]: row for row in scout["top_fields"]}
        self.assertEqual(rows["snt21_4neut_conf_low"]["primary_policy"], "avoid_primary")
        self.assertEqual(rows["snt21_4neut_conf_low"]["field_reason"], "production_rescue_probe_exhausted")
        self.assertEqual(scout["top_primary_fields"][0]["field"], "fresh_other_signal")

    def test_field_scout_cools_down_small_failed_dataset_cluster(self):
        scout = build_field_scout(
            {
                "available": True,
                "fields": [
                    {
                        "id": "fnd23_new_route",
                        "type": "VECTOR",
                        "dataset_id": "fundamental23",
                        "category": "Fundamental",
                        "coverage": 0.8,
                        "userCount": 0,
                        "alphaCount": 0,
                        "pyramidMultiplier": 1.6,
                    },
                    {
                        "id": "news12_fresh_route",
                        "type": "MATRIX",
                        "dataset_id": "news12",
                        "category": "News",
                        "coverage": 0.8,
                        "userCount": 0,
                        "alphaCount": 0,
                        "pyramidMultiplier": 1.7,
                    },
                ],
            },
            history_memory={
                "top_field_datasets": [
                    {
                        "dataset_id": "fundamental23",
                        "count": 5,
                        "failed": 5,
                        "best_sharpe": 0.2,
                        "best_fitness": 0.05,
                    }
                ],
            },
        )

        rows = {row["field"]: row for row in scout["top_fields"]}
        self.assertEqual(rows["fnd23_new_route"]["primary_policy"], "avoid_primary")
        self.assertEqual(rows["fnd23_new_route"]["dataset_reason"], "recent_dataset_failure_cluster")
        self.assertEqual(scout["top_primary_fields"][0]["field"], "news12_fresh_route")

    def test_field_scout_does_not_promote_dataset_failure_clusters_when_they_block_everything(self):
        scout = build_field_scout(
            {
                "available": True,
                "fields": [
                    {
                        "id": "anl10_cpssmun_2yf",
                        "type": "MATRIX",
                        "dataset_id": "analyst10",
                        "category": "Analyst",
                        "coverage": 0.9,
                        "userCount": 0,
                        "alphaCount": 0,
                        "pyramidMultiplier": 1.2,
                    },
                    {
                        "id": "anl10_lower_coverage",
                        "type": "MATRIX",
                        "dataset_id": "analyst10",
                        "category": "Analyst",
                        "coverage": 0.2,
                        "userCount": 4,
                        "alphaCount": 4,
                        "pyramidMultiplier": 1.0,
                    },
                ],
            },
            history_memory={
                "top_field_datasets": [
                    {
                        "dataset_id": "analyst10",
                        "count": 16,
                        "failed": 16,
                        "best_sharpe": 0.2,
                        "best_fitness": 0.1,
                    }
                ]
            },
        )

        self.assertFalse(scout["active"])
        self.assertEqual(scout["status"], "no_primary_fields")
        self.assertEqual(scout["top_primary_fields"], [])
        rows = {row["field"]: row for row in scout["top_fields"]}
        self.assertEqual(rows["anl10_cpssmun_2yf"]["primary_policy"], "avoid_primary")
        self.assertEqual(rows["anl10_cpssmun_2yf"]["dataset_reason"], "recent_dataset_failure_cluster")

    def test_field_scout_blocks_four_of_four_dataset_failure_cluster(self):
        scout = build_field_scout(
            {
                "available": True,
                "fields": [
                    {
                        "id": "social_score_sector_percentile",
                        "type": "MATRIX",
                        "dataset_id": "analyst11",
                        "category": "Analyst",
                        "coverage": 1.0,
                        "userCount": 0,
                        "alphaCount": 0,
                        "pyramidMultiplier": 1.8,
                    }
                ],
            },
            history_memory={
                "top_field_datasets": [
                    {
                        "dataset_id": "analyst11",
                        "count": 4,
                        "failed": 4,
                        "best_sharpe": 0.17,
                        "best_fitness": 0.04,
                    }
                ]
            },
        )

        rows = {row["field"]: row for row in scout["top_fields"]}
        self.assertFalse(scout["active"])
        self.assertEqual(rows["social_score_sector_percentile"]["primary_policy"], "avoid_primary")
        self.assertEqual(rows["social_score_sector_percentile"]["dataset_reason"], "recent_dataset_failure_cluster")

    def test_field_scout_blocks_failed_only_dataset_even_with_moderate_best_candidate(self):
        scout = build_field_scout(
            {
                "available": True,
                "fields": [
                    {
                        "id": "credit_risk_country_percentile_score_float_2",
                        "type": "MATRIX",
                        "dataset_id": "model37",
                        "category": "Model",
                        "coverage": 0.88,
                        "userCount": 0,
                        "alphaCount": 0,
                        "pyramidMultiplier": 1.7,
                    },
                    {
                        "id": "fresh_other_signal",
                        "type": "MATRIX",
                        "dataset_id": "other999",
                        "category": "Other",
                        "coverage": 0.75,
                        "userCount": 0,
                        "alphaCount": 0,
                        "pyramidMultiplier": 1.6,
                    },
                ],
            },
            history_memory={
                "top_field_datasets": [
                    {
                        "dataset_id": "model37",
                        "count": 10,
                        "failed": 10,
                        "approved": 0,
                        "submitted": 0,
                        "check_pending": 0,
                        "best_sharpe": 1.22,
                        "best_fitness": 0.54,
                    }
                ]
            },
        )

        rows = {row["field"]: row for row in scout["top_fields"]}
        self.assertEqual(rows["credit_risk_country_percentile_score_float_2"]["primary_policy"], "avoid_primary")
        self.assertEqual(rows["credit_risk_country_percentile_score_float_2"]["dataset_reason"], "failed_only_dataset_cluster")
        self.assertEqual(scout["top_primary_fields"][0]["field"], "fresh_other_signal")

    def test_field_scout_diversifies_top_fields_beyond_one_model_category(self):
        fields = [
            {
                "id": f"mdl262_signal_{idx}",
                "type": "MATRIX",
                "dataset_id": f"model{idx}",
                "category": "Model",
                "coverage": 1.0,
                "userCount": 0,
                "alphaCount": 0,
                "pyramidMultiplier": 1.8,
            }
            for idx in range(8)
        ]
        fields.extend(
            [
                {
                    "id": "fresh_other_signal",
                    "type": "MATRIX",
                    "dataset_id": "other999",
                    "category": "Other",
                    "coverage": 0.8,
                    "userCount": 0,
                    "alphaCount": 0,
                    "pyramidMultiplier": 1.6,
                },
                {
                    "id": "fresh_fundamental_signal",
                    "type": "MATRIX",
                    "dataset_id": "fundamental999",
                    "category": "Fundamental",
                    "coverage": 0.8,
                    "userCount": 0,
                    "alphaCount": 0,
                    "pyramidMultiplier": 1.6,
                },
            ]
        )

        scout = build_field_scout({"available": True, "fields": fields}, top_limit=6)

        top_ids = [row["field"] for row in scout["top_fields"]]
        self.assertIn("fresh_other_signal", top_ids)
        self.assertIn("fresh_fundamental_signal", top_ids)
        self.assertLess(
            sum(1 for row in scout["top_fields"] if row.get("category") == "Model"),
            len(scout["top_fields"]),
        )

    def test_field_scout_surfaces_non_default_retest_lane_when_primary_pool_is_model_pv_only(self):
        scout = build_field_scout(
            {
                "available": True,
                "fields": [
                    {
                        "id": "mdl262_signal",
                        "type": "MATRIX",
                        "dataset_id": "model262",
                        "category": "Model",
                        "coverage": 1.0,
                        "userCount": 0,
                        "alphaCount": 0,
                        "pyramidMultiplier": 1.8,
                    },
                    {
                        "id": "pv13_signal",
                        "type": "MATRIX",
                        "dataset_id": "pv13",
                        "category": "Price Volume",
                        "coverage": 1.0,
                        "userCount": 0,
                        "alphaCount": 0,
                        "pyramidMultiplier": 1.8,
                    },
                    {
                        "id": "anl10_revision_breadth",
                        "type": "MATRIX",
                        "dataset_id": "analyst10",
                        "category": "Analyst",
                        "coverage": 0.95,
                        "userCount": 0,
                        "alphaCount": 0,
                        "pyramidMultiplier": 1.6,
                    },
                    {
                        "id": "fundamental_companyidmap",
                        "type": "VECTOR",
                        "dataset_id": "fundamental14",
                        "category": "Fundamental",
                        "coverage": 1.0,
                        "userCount": 0,
                        "alphaCount": 0,
                        "pyramidMultiplier": 1.8,
                        "description": "Company id mapping helper",
                    },
                ],
            },
            history_memory={
                "top_field_datasets": [
                    {
                        "dataset_id": "analyst10",
                        "count": 8,
                        "failed": 8,
                        "distinct_field_count": 4,
                        "best_sharpe": 0.55,
                        "best_fitness": 0.12,
                    }
                ]
            },
        )

        self.assertTrue(scout["active"])
        self.assertEqual(
            {row["category"] for row in scout["top_primary_fields"]},
            {"Model", "Price Volume"},
        )
        self.assertEqual(scout["retest_primary_fields"][0]["field"], "anl10_revision_breadth")
        self.assertEqual(scout["retest_primary_fields"][0]["retest_reason"], "field_native_retest_after_dataset_cluster")
        self.assertNotIn("fundamental_companyidmap", [row["field"] for row in scout["retest_primary_fields"]])

    def test_field_scout_surfaces_retest_lane_when_dataset_clusters_leave_no_primary_fields(self):
        scout = build_field_scout(
            {
                "available": True,
                "fields": [
                    {
                        "id": "anl10_revision_breadth",
                        "type": "MATRIX",
                        "dataset_id": "analyst10",
                        "category": "Analyst",
                        "coverage": 0.95,
                        "userCount": 0,
                        "alphaCount": 0,
                        "pyramidMultiplier": 1.6,
                    },
                    {
                        "id": "snt23_confidence_width",
                        "type": "MATRIX",
                        "dataset_id": "sentiment23",
                        "category": "Sentiment",
                        "coverage": 0.9,
                        "userCount": 0,
                        "alphaCount": 0,
                        "pyramidMultiplier": 1.6,
                    },
                    {
                        "id": "fnd14_companyidmap",
                        "type": "VECTOR",
                        "dataset_id": "fundamental14",
                        "category": "Fundamental",
                        "coverage": 1.0,
                        "userCount": 0,
                        "alphaCount": 0,
                        "pyramidMultiplier": 1.8,
                        "description": "Company id mapping helper",
                    },
                ],
            },
            history_memory={
                "top_field_datasets": [
                    {
                        "dataset_id": "analyst10",
                        "count": 29,
                        "failed": 29,
                        "distinct_field_count": 29,
                        "best_sharpe": 0.55,
                        "best_fitness": 0.12,
                    },
                    {
                        "dataset_id": "sentiment23",
                        "count": 5,
                        "failed": 5,
                        "distinct_field_count": 5,
                        "best_sharpe": 0.2,
                        "best_fitness": 0.05,
                    },
                ]
            },
        )

        self.assertFalse(scout["active"])
        self.assertEqual(scout["top_primary_fields"], [])
        self.assertEqual(scout["retest_primary_fields"][0]["field"], "snt23_confidence_width")
        self.assertIn("anl10_revision_breadth", [row["field"] for row in scout["retest_primary_fields"]])
        self.assertNotIn("fnd14_companyidmap", [row["field"] for row in scout["retest_primary_fields"]])

    def test_field_scout_keeps_avoid_primary_fields_out_of_primary_buckets(self):
        scout = build_field_scout(
            {
                "available": True,
                "fields": [
                    {
                        "id": "submitted_high_score_signal",
                        "type": "MATRIX",
                        "dataset_id": "fundamental",
                        "category": "Fundamental",
                        "coverage": 1.0,
                        "userCount": 0,
                        "alphaCount": 0,
                        "pyramidMultiplier": 1.8,
                    },
                    {
                        "id": "usable_unsubmitted_signal",
                        "type": "MATRIX",
                        "dataset_id": "news",
                        "category": "News",
                        "coverage": 0.7,
                        "userCount": 0,
                        "alphaCount": 0,
                        "pyramidMultiplier": 1.5,
                    },
                ],
            },
            submitted_avoidance={"fields": ["submitted_high_score_signal"]},
            lit_tower_avoidance={"unlit_towers": [{"category": "FUNDAMENTAL"}, {"category": "NEWS"}]},
        )

        rows = {row["field"]: row for row in scout["top_fields"]}
        self.assertEqual(rows["submitted_high_score_signal"]["primary_policy"], "avoid_primary")
        self.assertEqual(rows["usable_unsubmitted_signal"]["primary_policy"], "prefer_primary")
        for bucket in scout["buckets"]:
            if bucket["name"] in {"high_opportunity_unexplored", "low_user_high_coverage", "high_multiplier_unlit_tower"}:
                self.assertNotIn("submitted_high_score_signal", bucket["fields"])

    def test_field_scout_orders_preferred_primary_fields_before_avoid_primary_fields(self):
        scout = build_field_scout(
            {
                "available": True,
                "fields": [
                    {
                        "id": "submitted_high_score_signal",
                        "type": "MATRIX",
                        "dataset_id": "fundamental",
                        "category": "Fundamental",
                        "coverage": 1.0,
                        "userCount": 0,
                        "alphaCount": 0,
                        "pyramidMultiplier": 1.8,
                    },
                    {
                        "id": "fresh_lower_score_signal",
                        "type": "MATRIX",
                        "dataset_id": "news",
                        "category": "News",
                        "coverage": 0.45,
                        "userCount": 3,
                        "alphaCount": 2,
                        "pyramidMultiplier": 1.2,
                    },
                ],
            },
            submitted_avoidance={"fields": ["submitted_high_score_signal"]},
        )

        self.assertEqual(scout["top_fields"][0]["field"], "fresh_lower_score_signal")
        self.assertEqual(scout["top_fields"][0]["primary_policy"], "prefer_primary")

    def test_field_scout_marks_submitted_fields_as_auxiliary_only(self):
        scout = build_field_scout(
            {
                "available": True,
                "fields": [
                    {
                        "id": "submitted_signal",
                        "type": "MATRIX",
                        "dataset_id": "earnings",
                        "category": "Earnings",
                        "coverage": 0.9,
                        "userCount": 0,
                        "alphaCount": 0,
                        "pyramidMultiplier": 1.8,
                    }
                ],
            },
            submitted_avoidance={"fields": ["submitted_signal"]},
        )

        row = scout["top_fields"][0]
        self.assertEqual(row["field"], "submitted_signal")
        self.assertEqual(row["primary_policy"], "avoid_primary")
        self.assertLess(row["score"], 0.5)

    def test_field_scout_keeps_lit_tower_fields_out_of_fresh_opportunity_bucket(self):
        scout = build_field_scout(
            {
                "available": True,
                "fields": [
                    {
                        "id": "lit_analyst_signal",
                        "type": "MATRIX",
                        "dataset_id": "analyst",
                        "category": "Analyst",
                        "coverage": 0.95,
                        "userCount": 0,
                        "alphaCount": 0,
                        "pyramidMultiplier": 1.8,
                    },
                    {
                        "id": "unlit_fundamental_signal",
                        "type": "MATRIX",
                        "dataset_id": "fundamental",
                        "category": "Fundamental",
                        "coverage": 0.6,
                        "userCount": 0,
                        "alphaCount": 0,
                        "pyramidMultiplier": 1.8,
                    },
                ],
            },
            lit_tower_avoidance={
                "lit_towers": [{"category": "ANALYST"}],
                "unlit_towers": [{"category": "FUNDAMENTAL"}],
            },
        )

        bucket = next(bucket for bucket in scout["buckets"] if bucket["name"] == "high_opportunity_unexplored")
        self.assertIn("unlit_fundamental_signal", bucket["fields"])
        self.assertNotIn("lit_analyst_signal", bucket["fields"])

    def test_field_scout_maps_price_volume_category_to_pv_tower(self):
        scout = build_field_scout(
            {
                "available": True,
                "fields": [
                    {
                        "id": "pv_signal",
                        "type": "MATRIX",
                        "dataset_id": "pv96",
                        "category": "Price Volume",
                        "coverage": 0.9,
                        "userCount": 0,
                        "alphaCount": 0,
                        "pyramidMultiplier": 1.8,
                    }
                ],
            },
            lit_tower_avoidance={"lit_towers": [{"category": "PV"}]},
        )

        self.assertEqual(scout["top_fields"][0]["tower_status"], "lit")
        self.assertEqual(scout["top_fields"][0]["primary_policy"], "avoid_primary")
        bucket_names = [bucket["name"] for bucket in scout["buckets"]]
        self.assertNotIn("high_opportunity_unexplored", bucket_names)

    def test_field_scout_is_not_active_when_no_primary_safe_fields_remain(self):
        scout = build_field_scout(
            {
                "available": True,
                "fields": [
                    {
                        "id": "lit_pv_signal",
                        "type": "MATRIX",
                        "dataset_id": "pv96",
                        "category": "Price Volume",
                        "coverage": 0.9,
                        "userCount": 0,
                        "alphaCount": 0,
                        "pyramidMultiplier": 1.8,
                    },
                    {
                        "id": "failed_sentiment_signal",
                        "type": "MATRIX",
                        "dataset_id": "sentiment23",
                        "category": "Sentiment",
                        "coverage": 1.0,
                        "userCount": 0,
                        "alphaCount": 0,
                        "pyramidMultiplier": 1.8,
                    },
                ],
            },
            history_memory={
                "top_field_datasets": [
                    {
                        "dataset_id": "sentiment23",
                        "count": 8,
                        "failed": 8,
                        "best_sharpe": 0.2,
                        "best_fitness": 0.05,
                    }
                ]
            },
            lit_tower_avoidance={"lit_towers": [{"category": "PV"}]},
        )

        self.assertFalse(scout["active"])
        self.assertEqual(scout["status"], "no_primary_fields")
        self.assertEqual(scout["top_primary_fields"], [])
        self.assertEqual(scout["buckets"], [])

    def test_field_scout_demotes_metadata_fields_as_primary_signals(self):
        scout = build_field_scout(
            {
                "available": True,
                "fields": [
                    {
                        "id": "fnd72_pit_or_bs_q_fundamental_entry_dt",
                        "type": "VECTOR",
                        "dataset_id": "fundamental72",
                        "category": "Fundamental",
                        "coverage": 0.95,
                        "userCount": 0,
                        "alphaCount": 0,
                        "pyramidMultiplier": 1.8,
                        "description": "Fundamental data entry date (timestamp of entry)",
                    },
                    {
                        "id": "fnd72_pit_or_bs_q_bs_inventories",
                        "type": "VECTOR",
                        "dataset_id": "fundamental72",
                        "category": "Fundamental",
                        "coverage": 0.55,
                        "userCount": 0,
                        "alphaCount": 0,
                        "pyramidMultiplier": 1.8,
                        "description": "Inventories",
                    },
                ],
            }
        )

        rows = {row["field"]: row for row in scout["top_fields"]}
        self.assertEqual(rows["fnd72_pit_or_bs_q_fundamental_entry_dt"]["primary_policy"], "avoid_primary")
        self.assertLess(
            rows["fnd72_pit_or_bs_q_fundamental_entry_dt"]["score"],
            rows["fnd72_pit_or_bs_q_bs_inventories"]["score"],
        )

    def test_field_scout_demotes_identifier_and_event_time_fields_as_primary_signals(self):
        scout = build_field_scout(
            {
                "available": True,
                "fields": [
                    {
                        "id": "mcr27_companyidmap",
                        "type": "VECTOR",
                        "dataset_id": "macro27",
                        "category": "Macro",
                        "coverage": 1.0,
                        "userCount": 0,
                        "alphaCount": 0,
                        "pyramidMultiplier": 1.8,
                        "description": "Company id mapping",
                    },
                    {
                        "id": "snt7_universeisin",
                        "type": "VECTOR",
                        "dataset_id": "sentiment7",
                        "category": "Sentiment",
                        "coverage": 1.0,
                        "userCount": 0,
                        "alphaCount": 0,
                        "pyramidMultiplier": 1.8,
                        "description": "Universe ISIN mapping",
                    },
                    {
                        "id": "shrt24_triggertime",
                        "type": "MATRIX",
                        "dataset_id": "shortinterest24",
                        "category": "ShortInterest",
                        "coverage": 1.0,
                        "userCount": 0,
                        "alphaCount": 0,
                        "pyramidMultiplier": 1.8,
                        "description": "Short-sale trigger timestamp",
                    },
                    {
                        "id": "nws94_v2_comp_utc_time",
                        "type": "VECTOR",
                        "dataset_id": "news94",
                        "category": "News",
                        "coverage": 1.0,
                        "userCount": 0,
                        "alphaCount": 0,
                        "pyramidMultiplier": 1.8,
                        "description": "News component UTC time",
                    },
                    {
                        "id": "anl10_eps_revision_score",
                        "type": "MATRIX",
                        "dataset_id": "analyst10",
                        "category": "Analyst",
                        "coverage": 0.7,
                        "userCount": 1,
                        "alphaCount": 1,
                        "pyramidMultiplier": 1.6,
                        "description": "EPS revision score",
                    },
                ],
            },
            lit_tower_avoidance={"unlit_towers": [{"category": "Macro"}, {"category": "Sentiment"}, {"category": "News"}]},
        )

        rows = {row["field"]: row for row in scout["top_fields"]}
        self.assertEqual(rows["mcr27_companyidmap"]["primary_policy"], "avoid_primary")
        self.assertEqual(rows["mcr27_companyidmap"]["metadata_reason"], "identifier_mapping")
        self.assertEqual(rows["snt7_universeisin"]["primary_policy"], "avoid_primary")
        self.assertEqual(rows["snt7_universeisin"]["metadata_reason"], "identifier_mapping")
        self.assertEqual(rows["shrt24_triggertime"]["primary_policy"], "avoid_primary")
        self.assertEqual(rows["shrt24_triggertime"]["metadata_reason"], "event_time_state")
        self.assertEqual(rows["nws94_v2_comp_utc_time"]["primary_policy"], "avoid_primary")
        self.assertEqual(rows["nws94_v2_comp_utc_time"]["metadata_reason"], "event_time_state")
        self.assertEqual(scout["top_primary_fields"][0]["field"], "anl10_eps_revision_score")

    def test_field_scout_demotes_event_state_fields_as_primary_signals(self):
        scout = build_field_scout(
            {
                "available": True,
                "fields": [
                    {
                        "id": "advantageous_position_flag",
                        "type": "MATRIX",
                        "dataset_id": "news12",
                        "category": "News",
                        "coverage": 0.8,
                        "userCount": 0,
                        "alphaCount": 0,
                        "pyramidMultiplier": 1.7,
                    },
                    {
                        "id": "news_category_tag",
                        "type": "VECTOR",
                        "dataset_id": "news12",
                        "category": "News",
                        "coverage": 0.8,
                        "userCount": 0,
                        "alphaCount": 0,
                        "pyramidMultiplier": 1.7,
                    },
                    {
                        "id": "news12_fresh_route",
                        "type": "MATRIX",
                        "dataset_id": "news12",
                        "category": "News",
                        "coverage": 0.7,
                        "userCount": 0,
                        "alphaCount": 0,
                        "pyramidMultiplier": 1.7,
                    },
                ],
            }
        )

        rows = {row["field"]: row for row in scout["top_fields"]}
        self.assertEqual(rows["advantageous_position_flag"]["primary_policy"], "avoid_primary")
        self.assertEqual(rows["advantageous_position_flag"]["metadata_reason"], "event_state_helper")
        self.assertEqual(rows["news_category_tag"]["primary_policy"], "avoid_primary")
        self.assertEqual(rows["news_category_tag"]["metadata_reason"], "event_state_helper")
        self.assertEqual(scout["top_primary_fields"][0]["field"], "news12_fresh_route")

    def test_field_scout_demotes_count_and_period_helper_fields_as_primary_signals(self):
        scout = build_field_scout(
            {
                "available": True,
                "fields": [
                    {
                        "id": "presentation_wordcount",
                        "type": "MATRIX",
                        "dataset_id": "other384",
                        "category": "Other",
                        "coverage": 0.8,
                        "userCount": 0,
                        "alphaCount": 0,
                        "pyramidMultiplier": 1.7,
                    },
                    {
                        "id": "confcall_fiscal_quarter",
                        "type": "MATRIX",
                        "dataset_id": "earnings7",
                        "category": "Earnings",
                        "coverage": 0.8,
                        "userCount": 0,
                        "alphaCount": 0,
                        "pyramidMultiplier": 1.7,
                    },
                    {
                        "id": "analyst_revision_breadth",
                        "type": "MATRIX",
                        "dataset_id": "analyst10",
                        "category": "Analyst",
                        "coverage": 0.8,
                        "userCount": 0,
                        "alphaCount": 0,
                        "pyramidMultiplier": 1.7,
                    },
                ],
            }
        )

        rows = {row["field"]: row for row in scout["top_fields"]}
        self.assertEqual(rows["presentation_wordcount"]["primary_policy"], "avoid_primary")
        self.assertEqual(rows["presentation_wordcount"]["metadata_reason"], "count_period_helper")
        self.assertEqual(rows["confcall_fiscal_quarter"]["primary_policy"], "avoid_primary")
        self.assertEqual(rows["confcall_fiscal_quarter"]["metadata_reason"], "count_period_helper")
        self.assertEqual(scout["top_primary_fields"][0]["field"], "analyst_revision_breadth")

    def test_field_scout_keeps_unproven_insider_raw_fields_out_of_primary_candidates(self):
        scout = build_field_scout(
            {
                "available": True,
                "fields": [
                    {
                        "id": "top_directional_significant_value_2",
                        "type": "MATRIX",
                        "dataset_id": "insider_agg_matrix",
                        "category": "Insiders",
                        "coverage": 0.8,
                        "userCount": 0,
                        "alphaCount": 0,
                        "pyramidMultiplier": 1.5,
                    },
                    {
                        "id": "safe_quality_signal",
                        "type": "MATRIX",
                        "dataset_id": "quality12",
                        "category": "Fundamental",
                        "coverage": 0.75,
                        "userCount": 0,
                        "alphaCount": 0,
                        "pyramidMultiplier": 1.4,
                    }
                ],
            }
        )

        rows = {row["field"]: row for row in scout["top_fields"]}
        self.assertIn("requires_turnover_stabilizer", rows["top_directional_significant_value_2"]["usage_constraints"])
        self.assertEqual(rows["top_directional_significant_value_2"]["primary_policy"], "avoid_primary")
        self.assertEqual(rows["top_directional_significant_value_2"]["primary_block_reason"], "unproven_insider_raw_field")
        self.assertEqual(scout["top_primary_fields"][0]["field"], "safe_quality_signal")

    def test_field_scout_demotes_vector_event_fields_from_primary_candidates(self):
        scout = build_field_scout(
            {
                "available": True,
                "fields": [
                    {
                        "id": "mws38_uniq_action",
                        "type": "VECTOR",
                        "dataset_id": "news38",
                        "category": "News",
                        "coverage": 0.9,
                        "userCount": 0,
                        "alphaCount": 0,
                        "pyramidMultiplier": 1.7,
                    },
                    {
                        "id": "inst6_num_of_institutional_buyers",
                        "type": "MATRIX",
                        "dataset_id": "institutions6",
                        "category": "Institutions",
                        "coverage": 1.0,
                        "userCount": 1,
                        "alphaCount": 1,
                        "pyramidMultiplier": 1.4,
                    },
                ],
            }
        )

        rows = {row["field"]: row for row in scout["top_fields"]}
        self.assertEqual(rows["mws38_uniq_action"]["primary_policy"], "avoid_primary")
        self.assertIn("event_field_not_direct_rank_input", rows["mws38_uniq_action"]["usage_constraints"])
        self.assertEqual(rows["mws38_uniq_action"]["primary_block_reason"], "event_vector_primary_block")
        self.assertEqual(scout["top_primary_fields"][0]["field"], "inst6_num_of_institutional_buyers")

    def test_field_scout_demotes_price_volume_like_field_names_as_primary_signals(self):
        scout = build_field_scout(
            {
                "available": True,
                "fields": [
                    {
                        "id": "after_hours_vwap",
                        "type": "MATRIX",
                        "dataset_id": "news5",
                        "category": "News",
                        "coverage": 1.0,
                        "userCount": 0,
                        "alphaCount": 0,
                        "pyramidMultiplier": 1.8,
                        "description": "After-hours VWAP",
                    },
                    {
                        "id": "credit_model_structural_letter_grade_float",
                        "type": "MATRIX",
                        "dataset_id": "model37",
                        "category": "Model",
                        "coverage": 1.0,
                        "userCount": 0,
                        "alphaCount": 0,
                        "pyramidMultiplier": 1.8,
                        "description": "Credit model structural letter grade",
                    },
                ],
            }
        )

        rows = {row["field"]: row for row in scout["top_fields"]}
        self.assertEqual(rows["after_hours_vwap"]["primary_policy"], "avoid_primary")
        self.assertEqual(rows["after_hours_vwap"]["metadata_reason"], "price_volume_like")
        self.assertLess(rows["after_hours_vwap"]["score"], rows["credit_model_structural_letter_grade_float"]["score"])

    def test_field_scout_demotes_volatility_and_size_fields_as_primary_signals(self):
        scout = build_field_scout(
            {
                "available": True,
                "fields": [
                    {
                        "id": "implied_volatility_mean_90",
                        "type": "MATRIX",
                        "dataset_id": "option8",
                        "category": "Option",
                        "coverage": 1.0,
                        "userCount": 0,
                        "alphaCount": 0,
                        "pyramidMultiplier": 1.8,
                        "description": "Implied volatility mean",
                    },
                    {
                        "id": "market_capitalization_usd_2",
                        "type": "MATRIX",
                        "dataset_id": "fundamental2",
                        "category": "Fundamental",
                        "coverage": 1.0,
                        "userCount": 0,
                        "alphaCount": 0,
                        "pyramidMultiplier": 1.8,
                        "description": "Market capitalization in USD",
                    },
                    {
                        "id": "oth476_mfm_squeeze",
                        "type": "MATRIX",
                        "dataset_id": "other476",
                        "category": "Other",
                        "coverage": 0.8,
                        "userCount": 0,
                        "alphaCount": 0,
                        "pyramidMultiplier": 1.8,
                        "description": "Light squeeze signal",
                    },
                ],
            }
        )

        rows = {row["field"]: row for row in scout["top_fields"]}
        self.assertEqual(rows["implied_volatility_mean_90"]["primary_policy"], "avoid_primary")
        self.assertEqual(rows["implied_volatility_mean_90"]["metadata_reason"], "risk_state_helper")
        self.assertEqual(rows["market_capitalization_usd_2"]["primary_policy"], "avoid_primary")
        self.assertEqual(rows["market_capitalization_usd_2"]["metadata_reason"], "size_helper")
        self.assertEqual(scout["top_primary_fields"][0]["field"], "oth476_mfm_squeeze")

    def test_field_scout_demotes_zero_coverage_fields(self):
        scout = build_field_scout(
            {
                "available": True,
                "fields": [
                    {
                        "id": "zero_coverage_signal",
                        "type": "VECTOR",
                        "dataset_id": "fundamental",
                        "category": "Fundamental",
                        "coverage": 0.0,
                        "userCount": 0,
                        "alphaCount": 0,
                        "pyramidMultiplier": 1.8,
                    },
                    {
                        "id": "usable_coverage_signal",
                        "type": "VECTOR",
                        "dataset_id": "fundamental",
                        "category": "Fundamental",
                        "coverage": 0.35,
                        "userCount": 1,
                        "alphaCount": 1,
                        "pyramidMultiplier": 1.8,
                    },
                ],
            }
        )

        rows = {row["field"]: row for row in scout["top_fields"]}
        self.assertLess(rows["zero_coverage_signal"]["score"], rows["usable_coverage_signal"]["score"])


if __name__ == "__main__":
    unittest.main()
