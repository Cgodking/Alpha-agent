from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from alpha.clients import LocalAIClient, LocalBrainClient
from alpha.db import AlphaStore
from alpha.guards import SubmissionPolicy
from alpha.preflight import validate_expression
from alpha.worker import AlphaWorker


class PreflightTests(unittest.TestCase):
    def test_validate_expression_accepts_known_operator_chain(self):
        errors = validate_expression("rank(ts_mean(close, 22))")

        self.assertEqual(errors, [])

    def test_validate_expression_accepts_wqb_delta_operator(self):
        errors = validate_expression("(-1) * ts_rank(delta(close, 1), 5)")

        self.assertEqual(errors, [])

    def test_validate_expression_accepts_platform_ts_zscore_operator(self):
        errors = validate_expression("rank(ts_zscore(close, 20))")

        self.assertEqual(errors, [])

    def test_validate_expression_rejects_auxiliary_only_fields_as_primary_signal_when_enforced(self):
        errors = validate_expression(
            "group_rank(ts_mean(ts_rank(normalize(divide(vwap, close)), 66), 44), industry)",
            enforce_auxiliary_field_roles=True,
        )

        self.assertTrue(any(error.startswith("AUXILIARY_FIELD_AS_PRIMARY:") for error in errors))

    def test_validate_expression_allows_auxiliary_field_as_normalizer_when_main_field_is_not_auxiliary(self):
        errors = validate_expression(
            "group_rank(ts_rank(divide(earnings_quality_score_raw, close), 63), industry)",
            allowed_fields=["earnings_quality_score_raw"],
            enforce_auxiliary_field_roles=True,
        )

        self.assertEqual(errors, [])

    def test_validate_expression_rejects_auxiliary_only_additive_leg_when_enforced(self):
        errors = validate_expression(
            "rank(add(ts_rank(earnings_quality_score_raw, 63), ts_rank(ts_delta(close, 5), 22)))",
            allowed_fields=["earnings_quality_score_raw"],
            enforce_auxiliary_field_roles=True,
        )

        self.assertTrue(any(error.startswith("AUXILIARY_FIELD_AS_PRIMARY:") for error in errors))

    def test_validate_expression_rejects_unknown_operator(self):
        errors = validate_expression("made_up_operator(close)")

        self.assertIn("UNKNOWN_OPERATOR:made_up_operator", errors)

    def test_validate_expression_rejects_unknown_field_when_allowlist_present(self):
        errors = validate_expression(
            "group_rank(ts_rank(made_up_field, 22), industry)",
            allowed_fields=["mdl_score"],
        )

        self.assertIn("UNKNOWN_FIELD:made_up_field", errors)

    def test_validate_expression_accepts_allowlisted_field_and_named_argument(self):
        errors = validate_expression(
            "group_rank(winsorize(ts_backfill(mdl_score, 120), std=3), industry)",
            allowed_fields=["mdl_score"],
        )

        self.assertEqual(errors, [])

    def test_validate_expression_rejects_vector_field_passed_directly_to_ts_operator(self):
        errors = validate_expression(
            "rank(ts_backfill(analyst_sentence_count_presentation, 120))",
            allowed_fields=["analyst_sentence_count_presentation"],
            field_types={"analyst_sentence_count_presentation": "VECTOR"},
        )

        self.assertIn(
            "INVALID_VECTOR_TS_OPERATOR:ts_backfill:analyst_sentence_count_presentation",
            errors,
        )

    def test_validate_expression_rejects_nested_vector_field_ts_backfill(self):
        errors = validate_expression(
            "rank(group_rank(ts_mean(ts_backfill(analyst_sentence_count_presentation,120),22), industry))",
            allowed_fields=["analyst_sentence_count_presentation"],
            field_types={"analyst_sentence_count_presentation": "VECTOR"},
        )

        self.assertIn(
            "INVALID_VECTOR_TS_OPERATOR:ts_backfill:analyst_sentence_count_presentation",
            errors,
        )

    def test_validate_expression_accepts_vector_field_reduced_before_ts_operator(self):
        errors = validate_expression(
            "rank(ts_mean(vec_avg(analyst_sentence_count_presentation), 22))",
            allowed_fields=["analyst_sentence_count_presentation"],
            field_types={"analyst_sentence_count_presentation": "VECTOR"},
        )

        self.assertEqual(errors, [])

    def test_validate_expression_rejects_vector_reducer_extra_arguments(self):
        errors = validate_expression(
            "rank(ts_mean(vec_avg(analyst_sentence_count_presentation, 30), 22))",
            allowed_fields=["analyst_sentence_count_presentation"],
            field_types={"analyst_sentence_count_presentation": "VECTOR"},
        )

        self.assertIn("INVALID_VECTOR_REDUCER_ARITY:vec_avg", errors)

    def test_validate_expression_rejects_vector_reducer_on_matrix_field(self):
        errors = validate_expression(
            "rank(ts_mean(vec_avg(predicted_beta_change_3m), 22))",
            allowed_fields=["predicted_beta_change_3m"],
            field_types={"predicted_beta_change_3m": "MATRIX"},
        )

        self.assertIn("INVALID_VECTOR_REDUCER_INPUT_TYPE:vec_avg:predicted_beta_change_3m:MATRIX", errors)

    def test_validate_expression_rejects_empty_function_argument(self):
        errors = validate_expression(
            "rank(ts_mean(vec_avg(aggregate_prediction_accuracy_score, ), 30))",
            allowed_fields=["aggregate_prediction_accuracy_score"],
            field_types={"aggregate_prediction_accuracy_score": "VECTOR"},
        )

        self.assertIn("EMPTY_FUNCTION_ARGUMENT", errors)

    def test_worker_fails_candidate_before_simulation_when_preflight_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = AlphaStore(Path(tmp) / "alpha.db")
            store.init()
            brain = LocalBrainClient()
            worker = AlphaWorker(
                store=store,
                ai_client=LocalAIClient(expressions=["made_up_operator(close)"]),
                brain_client=brain,
                policy=SubmissionPolicy(auto_submit=False),
                batch_size=1,
            )

            summary = worker.run_once()

            candidate = store.list_candidates()[0]
            events = store.events_for_candidate(candidate["id"])
            self.assertEqual(summary["failed"], 1)
            self.assertEqual(candidate["status"], "failed")
            self.assertTrue(any(event["event_type"] == "preflight_failed" for event in events))

    def test_worker_applies_auxiliary_primary_field_policy_from_research_context(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = AlphaStore(Path(tmp) / "alpha.db")
            store.init()
            brain = LocalBrainClient()
            worker = AlphaWorker(
                store=store,
                ai_client=LocalAIClient(expressions=["group_rank(ts_rank(divide(vwap, close), 63), industry)"]),
                brain_client=brain,
                policy=SubmissionPolicy(auto_submit=False),
                batch_size=1,
                context={
                    "region": "USA",
                    "universe": "TOP3000",
                    "delay": 1,
                },
            )

            summary = worker.run_once()

            candidate = store.list_candidates()[0]
            events = store.events_for_candidate(candidate["id"])
            self.assertEqual(summary["failed"], 1)
            self.assertEqual(candidate["status"], "failed")
            self.assertTrue(
                any(
                    event["event_type"] == "preflight_failed"
                    and "AUXILIARY_FIELD_AS_PRIMARY:close,vwap" in event["metadata_json"]
                    for event in events
                )
            )


if __name__ == "__main__":
    unittest.main()
