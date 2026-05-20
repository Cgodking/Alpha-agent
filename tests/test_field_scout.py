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


if __name__ == "__main__":
    unittest.main()
