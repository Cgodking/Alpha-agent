from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from alpha.context_builder import build_ai_research_context, _recent_preflight_rejections
from alpha.db import AlphaStore


class ContextBuilderTests(unittest.TestCase):
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
            store.update_candidate(best_id, metrics_json=json.dumps({"sharpe": 1.11, "fitness": 0.46}))
            store.transition(
                best_id,
                "check_pending",
                {"errors": ["SHARPE_BELOW_MIN:1.110<1.58", "FITNESS_BELOW_MIN:0.460<1"]},
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
        self.assertEqual(analysis["best_candidate"]["sharpe"], 1.11)
        self.assertIn("analyst_positive_sentiment_logit_presentation", analysis["promising_fields"])
        self.assertIn("SHARPE_BELOW_MIN", analysis["failure_reasons"])
        self.assertEqual(plan["mode"], "optimize_best")
        self.assertEqual(plan["target_candidate_id"], best_id)
        self.assertEqual(plan["batch_size"], 8)
        self.assertIn("analyst_positive_sentiment_logit_presentation", plan["keep"])

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
            store.update_candidate(ind_id, metrics_json=json.dumps({"sharpe": 1.05, "fitness": 0.41}))
            store.transition(ind_id, "check_pending", {"errors": ["SHARPE_BELOW_MIN:1.050<1.58"]})

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
        self.assertNotEqual(plan["mode"], "setting_sweep")
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


if __name__ == "__main__":
    unittest.main()
