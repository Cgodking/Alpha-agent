from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from alpha.clients import AIClientError, LocalBrainClient, MultiModelAIClient, OpenAICompatibleAIClient
from alpha.db import AlphaStore
from alpha.guards import SubmissionPolicy
from alpha.models import CandidateSpec
from alpha.worker import AlphaWorker


class AiIntegrationTests(unittest.TestCase):
    def test_openai_client_prompt_includes_wqb_constraints(self):
        seen = {}

        def transport(payload):
            seen["payload"] = payload
            return {
                "choices": [
                    {
                        "message": {
                            "content": '{"candidates":[{"expression":"group_rank(ts_rank(ts_delta(close,5),22),industry)"}]}'
                        }
                    }
                ]
            }

        client = OpenAICompatibleAIClient(api_key="test", model="model-x", transport=transport)
        client.generate_candidates(
            1,
            {
                "region": "USA",
                "universe": "TOP3000",
                "delay": 1,
                "research_context": {
                    "datafields": {
                        "available": True,
                        "field_ids": ["close", "analyst_sentence_count_presentation"],
                        "field_types": {"analyst_sentence_count_presentation": "VECTOR"},
                    },
                    "knowledge": {"wqb_rules": "All mandatory checks must pass before submit."},
                    "generation_policy": {"complexity": "research_grade", "reject_trivial_candidates": True},
                    "syntax_constraints": {"auxiliary_only_fields": ["close", "vwap"]},
                },
            },
        )

        prompt = json.dumps(seen["payload"], sort_keys=True)
        self.assertIn("FASTEXPR", prompt)
        self.assertIn("operator", prompt.lower())
        self.assertIn("field", prompt.lower())
        self.assertIn("required_candidate_count", prompt)
        self.assertIn("research_grade", prompt)
        self.assertIn("avoid_trivial_price_volume_only", prompt)
        self.assertIn("All mandatory checks must pass", prompt)
        self.assertIn("VECTOR", prompt)
        self.assertIn("vec_", prompt)
        self.assertIn("ts_backfill", prompt)
        system_prompt = seen["payload"]["messages"][0]["content"]
        self.assertIn("research_context.syntax_constraints", system_prompt)
        self.assertIn("before scalar/value operators such as add, subtract, multiply, divide, normalize, zscore, and group_mean", system_prompt)
        self.assertIn("vec_avg(x) only", system_prompt)
        self.assertIn("Use divide(x, y), never div(x, y)", system_prompt)
        self.assertIn("group_rank(x, group)", system_prompt)
        self.assertIn("auxiliary_only_fields", system_prompt)
        self.assertIn("standalone additive/subtractive leg", system_prompt)
        self.assertNotIn("max_final_submit_per_round", prompt)
        self.assertNotIn("simple_first", prompt)

    def test_openai_client_system_prompt_requires_experiment_plan(self):
        seen = {}

        def transport(payload):
            seen["payload"] = payload
            return {
                "choices": [
                    {
                        "message": {
                            "content": '{"candidates":[{"expression":"group_rank(ts_rank(ts_delta(close,5),22),industry)"}]}'
                        }
                    }
                ]
            }

        client = OpenAICompatibleAIClient(api_key="test", model="model-x", transport=transport)
        client.generate_candidates(
            1,
            {
                "region": "USA",
                "universe": "TOP3000",
                "delay": 0,
                "research_context": {
                    "experiment_plan": {
                        "mode": "optimize_best",
                        "target_candidate_id": 27,
                        "objective": "optimize analyst sentiment plus aggregate sentiment structure",
                        "keep": ["analyst_positive_sentiment_logit_presentation", "aggregate_sentiment_total"],
                        "change": ["window", "normalization"],
                        "avoid": ["beta_prediction_uncertainty_2"],
                    }
                },
            },
        )

        system_prompt = seen["payload"]["messages"][0]["content"]
        user_prompt = seen["payload"]["messages"][1]["content"]
        self.assertIn("research_context.experiment_plan", system_prompt)
        self.assertIn("optimize_best", user_prompt)
        self.assertIn("target_candidate_id", user_prompt)

    def test_openai_client_prompt_mentions_family_diversity_control(self):
        seen = {}

        def transport(payload):
            seen["payload"] = payload
            return {
                "choices": [
                    {
                        "message": {
                            "content": '{"candidates":[{"expression":"group_rank(ts_rank(ts_delta(close,5),22),industry)"}]}'
                        }
                    }
                ]
            }

        client = OpenAICompatibleAIClient(api_key="test", model="model-x", transport=transport)
        client.generate_candidates(
            1,
            {
                "region": "USA",
                "universe": "TOP3000",
                "delay": 0,
                "research_context": {
                    "experiment_plan": {
                        "mode": "optimize_best",
                        "family_diversity_control": {
                            "dominant_family": "credit_risk",
                            "alternate_families": ["analyst", "model211"],
                        },
                    }
                },
            },
        )

        system_prompt = seen["payload"]["messages"][0]["content"]
        self.assertIn("family_diversity_control", system_prompt)
        self.assertIn("dominant family", system_prompt)
        self.assertIn("alternate_families", system_prompt)

    def test_openai_client_prompt_includes_primary_field_scout_view(self):
        seen = {}

        def transport(payload):
            seen["payload"] = payload
            return {
                "choices": [
                    {
                        "message": {
                            "content": '{"candidates":[{"expression":"rank(primary_signal)"}]}'
                        }
                    }
                ]
            }

        client = OpenAICompatibleAIClient(api_key="test", model="model-x", transport=transport)
        client.generate_candidates(
            1,
            {
                "region": "USA",
                "universe": "TOP3000",
                "delay": 0,
                "research_context": {
                    "field_scout": {
                        "active": True,
                        "top_fields": [
                            {
                                "field": "primary_signal",
                                "score": 0.7,
                                "primary_policy": "prefer_primary",
                                "dataset_reason": "",
                            },
                            {
                                "field": "avoid_signal",
                                "score": 0.9,
                                "primary_policy": "avoid_primary",
                                "dataset_reason": "recent_dataset_failure_cluster",
                            },
                        ],
                        "top_primary_fields": [
                            {
                                "field": "primary_signal",
                                "score": 0.7,
                                "primary_policy": "prefer_primary",
                                "usage_constraints": ["requires_turnover_stabilizer"],
                            }
                        ],
                    }
                },
            },
        )

        user_payload = json.loads(seen["payload"]["messages"][1]["content"])
        scout = user_payload["research_context"]["field_scout"]
        self.assertEqual(scout["top_primary_fields"][0]["field"], "primary_signal")
        self.assertEqual(scout["top_primary_fields"][0]["usage_constraints"], ["requires_turnover_stabilizer"])
        self.assertEqual(scout["top_fields"][1]["dataset_reason"], "recent_dataset_failure_cluster")
        self.assertTrue(user_payload["constraints"]["respect_field_scout_usage_constraints"])

    def test_openai_client_prompt_requires_quality_budget_and_probe_recommendations(self):
        seen = {}

        def transport(payload):
            seen["payload"] = payload
            return {
                "choices": [
                    {
                        "message": {
                            "content": '{"candidates":[{"expression":"rank(ts_mean(primary_signal,20))"}]}'
                        }
                    }
                ]
            }

        client = OpenAICompatibleAIClient(api_key="test", model="model-x", transport=transport)
        client.generate_candidates(
            1,
            {
                "region": "USA",
                "universe": "TOP500",
                "delay": 0,
                "research_context": {
                    "experiment_plan": {
                        "mode": "explore_new_family",
                        "quality_budget": {
                            "priority": "production_first",
                            "slots": {"exploit_positive_evidence": 5, "probe_new_fields": 2, "broad_explore": 1},
                        },
                        "probe_recommendations": [
                            {
                                "field": "primary_signal",
                                "templates": ["rank(ts_mean(primary_signal,20))"],
                                "stabilization_required": True,
                            }
                        ],
                    }
                },
            },
        )

        system_prompt = seen["payload"]["messages"][0]["content"]
        user_payload = json.loads(seen["payload"]["messages"][1]["content"])
        self.assertIn("quality_budget", system_prompt)
        self.assertIn("probe_recommendations", system_prompt)
        self.assertTrue(user_payload["constraints"]["respect_quality_budget"])
        self.assertTrue(user_payload["constraints"]["use_probe_recommendations_for_probe_slots"])
        self.assertEqual(
            user_payload["research_context"]["experiment_plan"]["quality_budget"]["slots"]["probe_new_fields"],
            2,
        )

    def test_openai_client_prompt_mentions_submitted_field_avoidance(self):
        seen = {}

        def transport(payload):
            seen["payload"] = payload
            return {
                "choices": [
                    {
                        "message": {
                            "content": '{"candidates":[{"expression":"group_rank(ts_rank(fresh_signal_score,63),industry)"}]}'
                        }
                    }
                ]
            }

        client = OpenAICompatibleAIClient(api_key="test", model="model-x", transport=transport)
        client.generate_candidates(
            1,
            {
                "region": "USA",
                "universe": "TOP3000",
                "delay": 0,
                "research_context": {
                    "experiment_plan": {
                        "mode": "explore_new_family",
                        "submitted_field_avoidance": {
                            "fields": ["est_q_pre_mean"],
                            "policy": "Do not reuse approved/submitted fields.",
                        },
                        "avoid": ["est_q_pre_mean"],
                    }
                },
            },
        )

        prompt = json.dumps(seen["payload"], sort_keys=True)
        system_prompt = seen["payload"]["messages"][0]["content"]
        self.assertIn("submitted_field_avoidance", system_prompt)
        self.assertIn("approved/submitted core fields", system_prompt)
        self.assertIn("avoid_recent_approved_submitted_fields", prompt)
        self.assertIn("est_q_pre_mean", prompt)

    def test_openai_client_prompt_mentions_mechanism_transfer_without_copying_blocked_fields(self):
        seen = {}

        def transport(payload):
            seen["payload"] = payload
            return {
                "choices": [
                    {
                        "message": {
                            "content": '{"candidates":[{"expression":"group_rank(ts_rank(fresh_model_signal,63),industry)"}]}'
                        }
                    }
                ]
            }

        client = OpenAICompatibleAIClient(api_key="test", model="model-x", transport=transport)
        client.generate_candidates(
            1,
            {
                "region": "MEA",
                "universe": "TOP300",
                "delay": 1,
                "research_context": {
                    "experiment_plan": {
                        "mode": "explore_new_family",
                        "scope_trouble": {"active": True},
                        "mechanism_transfer": {
                            "policy": "Use these as mechanism only examples. Do not copy forbidden fields.",
                            "forbidden_fields": ["est_q_pre_mean", "vwap", "close"],
                            "archetypes": [
                                {
                                    "id": 12,
                                    "mechanism_tags": ["time_series_persistence_rank", "relative_price_deviation"],
                                    "forbidden_fields": ["vwap", "close"],
                                }
                            ],
                        },
                    }
                },
            },
        )

        prompt = json.dumps(seen["payload"], sort_keys=True)
        system_prompt = seen["payload"]["messages"][0]["content"]
        self.assertIn("mechanism_transfer", system_prompt)
        self.assertIn("mechanism only", system_prompt)
        self.assertIn("Do not copy", system_prompt)
        self.assertIn("est_q_pre_mean", prompt)

    def test_openai_client_filters_trivial_candidates_when_research_policy_requests_it(self):
        def transport(_payload):
            return {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "candidates": [
                                        {"expression": "rank(close)"},
                                        {"expression": "group_rank(ts_rank(ts_delta(close, 5), 22), industry)"},
                                    ]
                                }
                            )
                        }
                    }
                ]
            }

        client = OpenAICompatibleAIClient(api_key="test", model="model-x", transport=transport)
        candidates = client.generate_candidates(
            2,
            {"research_context": {"generation_policy": {"reject_trivial_candidates": True}}},
        )

        self.assertEqual([item.expression for item in candidates], ["group_rank(ts_rank(ts_delta(close, 5), 22), industry)"])

    def test_openai_client_deduplicates_candidates(self):
        def transport(_payload):
            return {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "candidates": [
                                        {"expression": "rank(close)"},
                                        {"expression": " rank(close) "},
                                        {"expression": "rank(-returns)"},
                                    ]
                                }
                            )
                        }
                    }
                ]
            }

        client = OpenAICompatibleAIClient(api_key="test", model="model-x", transport=transport)
        candidates = client.generate_candidates(
            3,
            {"region": "USA", "research_context": {"generation_policy": {"avoid_structural_duplicates": True}}},
        )

        self.assertEqual([item.expression for item in candidates], ["rank(close)", "rank(-returns)"])

    def test_openai_client_deduplicates_structurally_equivalent_candidates(self):
        def transport(_payload):
            return {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "candidates": [
                                        {
                                            "expression": (
                                                "group_rank(ts_rank(divide(winsorize(ts_backfill("
                                                "credit_risk_news_component_score, 120), std=3), cap), 63), industry)"
                                            )
                                        },
                                        {
                                            "expression": (
                                                "group_rank(ts_rank(divide(winsorize(ts_backfill("
                                                "credit_risk_news_component_score, 66), "
                                                "std=4), cap), 44), industry)"
                                            )
                                        },
                                        {
                                            "expression": (
                                                "group_rank(ts_delta(credit_risk_news_component_score, 22), industry)"
                                            )
                                        },
                                    ]
                                }
                            )
                        }
                    }
                ]
            }

        client = OpenAICompatibleAIClient(api_key="test", model="model-x", transport=transport)
        candidates = client.generate_candidates(
            3,
            {"region": "USA", "research_context": {"generation_policy": {"avoid_structural_duplicates": True}}},
        )

        self.assertEqual(
            [candidate.expression for candidate in candidates],
            [
                (
                    "group_rank(ts_rank(divide(winsorize(ts_backfill("
                    "credit_risk_news_component_score, 120), std=3), cap), 63), industry)"
                ),
                "group_rank(ts_delta(credit_risk_news_component_score, 22), industry)",
            ],
        )

    def test_openai_client_preserves_candidate_research_metadata(self):
        def transport(_payload):
            return {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "candidates": [
                                        {
                                            "expression": "group_rank(ts_rank(ts_delta(close,5),22),industry)",
                                            "hypothesis": "short reversal with industry-relative ranking",
                                            "risk_notes": "watch self correlation",
                                        }
                                    ]
                                }
                            )
                        }
                    }
                ]
            }

        client = OpenAICompatibleAIClient(api_key="test", model="model-x", transport=transport)
        candidates = client.generate_candidates(1, {"region": "USA"})

        self.assertEqual(candidates[0].metadata["hypothesis"], "short reversal with industry-relative ranking")
        self.assertEqual(candidates[0].metadata["risk_notes"], "watch self correlation")

    def test_openai_client_does_not_allow_ai_to_override_run_settings(self):
        seen = {}

        def transport(payload):
            seen["payload"] = payload
            return {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "candidates": [
                                        {
                                            "expression": "group_rank(ts_rank(ts_delta(close,5),22),industry)",
                                            "settings": {
                                                "region": "CHN",
                                                "universe": "TOP2000U",
                                                "delay": 0,
                                                "neutralization": "SUBINDUSTRY",
                                            },
                                        }
                                    ]
                                }
                            )
                        }
                    }
                ]
            }

        client = OpenAICompatibleAIClient(api_key="test", model="model-x", transport=transport)
        candidates = client.generate_candidates(
            1,
            {
                "region": "USA",
                "universe": "TOP3000",
                "delay": 1,
                "neutralization": "INDUSTRY",
                "cycle_plan": {"mode": "explore", "budget": {"batch_size": 8}},
            },
        )

        self.assertEqual(candidates[0].settings["region"], "USA")
        self.assertEqual(candidates[0].settings["universe"], "TOP3000")
        self.assertEqual(candidates[0].settings["delay"], 1)
        self.assertEqual(candidates[0].settings["neutralization"], "INDUSTRY")
        self.assertNotIn("cycle_plan", candidates[0].settings)
        user_payload = json.loads(seen["payload"]["messages"][1]["content"])
        self.assertNotIn("cycle_plan", user_payload["target_settings"])
        self.assertEqual(candidates[0].metadata["proposed_settings"]["region"], "CHN")

    def test_openai_client_reports_invalid_json_as_ai_error(self):
        client = OpenAICompatibleAIClient(
            api_key="test",
            model="model-x",
            transport=lambda _payload: {"choices": [{"message": {"content": "not-json"}}]},
        )

        with self.assertRaises(AIClientError) as ctx:
            client.generate_candidates(1, {"region": "USA"})

        self.assertIn("invalid JSON", str(ctx.exception))

    def test_openai_client_extracts_json_from_wrapped_model_response(self):
        client = OpenAICompatibleAIClient(
            api_key="test",
            model="model-x",
            transport=lambda _payload: {
                "choices": [
                    {
                        "message": {
                            "content": (
                                "Here are the candidates.\n"
                                "```json\n"
                                "{\"candidates\":[{\"expression\":\"rank(ts_mean(close,22))\"}]}\n"
                                "```"
                            )
                        }
                    }
                ]
            },
        )

        candidates = client.generate_candidates(1, {"region": "USA"})

        self.assertEqual(candidates[0].expression, "rank(ts_mean(close,22))")

    def test_openai_client_reports_invalid_json_with_response_preview(self):
        client = OpenAICompatibleAIClient(
            api_key="test",
            model="model-x",
            transport=lambda _payload: {"choices": [{"message": {"content": "not-json response body"}}]},
        )

        with self.assertRaises(AIClientError) as ctx:
            client.generate_candidates(1, {"region": "USA"})

        self.assertIn("invalid JSON", str(ctx.exception))
        self.assertIn("not-json response body", str(ctx.exception))

    def test_openai_client_from_env_reads_ai_settings(self):
        with patch.dict(
            os.environ,
            {"AI_API_KEY": "key-1", "AI_MODEL": "model-y", "AI_BASE_URL": "https://example.test/v1"},
        ):
            client = OpenAICompatibleAIClient.from_env()

        self.assertEqual(client.api_key, "key-1")
        self.assertEqual(client.model, "model-y")
        self.assertEqual(client.base_url, "https://example.test/v1")

    def test_openai_client_accepts_full_chat_completions_url(self):
        client = OpenAICompatibleAIClient(
            api_key="test",
            model="model-x",
            base_url="https://relay.example/v1/chat/completions",
        )

        self.assertEqual(client.chat_completions_url, "https://relay.example/v1/chat/completions")

    def test_openai_client_appends_chat_completions_to_v1_base_url(self):
        client = OpenAICompatibleAIClient(api_key="test", model="model-x", base_url="https://relay.example/v1")

        self.assertEqual(client.chat_completions_url, "https://relay.example/v1/chat/completions")

    def test_openai_client_json_requests_include_max_tokens(self):
        seen = {}

        def transport(payload):
            seen["payload"] = payload
            return {
                "choices": [
                    {
                        "message": {
                            "content": '{"mode":"explore","profile_guidance":{},"rationale":"ok"}'
                        }
                    }
                ]
            }

        client = OpenAICompatibleAIClient(api_key="test", model="model-x", transport=transport)
        client.plan_generation(8, {"region": "USA"}, [{"name": "G-1", "role": "generator"}])

        self.assertIn("max_tokens", seen["payload"])
        self.assertGreaterEqual(seen["payload"]["max_tokens"], 1024)

    def test_multi_model_compacts_controller_context_but_preserves_generator_context(self):
        class Controller:
            profile_name = "flow"
            role = "controller"
            model = "controller"

            def __init__(self):
                self.context = None

            def plan_generation(self, batch_size, context, generator_profiles):
                self.context = context
                return {"mode": "explore", "profile_guidance": {}, "rationale": "ok"}

        class Critic:
            profile_name = "critic"
            role = "critic"
            model = "critic"

            def __init__(self):
                self.context = None

            def critique_generation_plan(self, batch_size, context, generator_profiles, initial_plan):
                self.context = context
                return {"concerns": [], "recommendations": [], "profile_guidance_delta": {}, "rationale": "ok"}

        class Generator:
            profile_name = "G-1"
            role = "generator"
            model = "generator"

            def __init__(self):
                self.context = None

            def generate_candidates(self, batch_size, context):
                self.context = context
                return [CandidateSpec("group_rank(ts_rank(ts_delta(close,5),22),industry)")]

        huge_fields = [
            {"id": f"field_{idx}", "description": "x" * 1000, "extra": list(range(20))}
            for idx in range(400)
        ]
        context = {
            "region": "USA",
            "universe": "TOP3000",
            "delay": 0,
            "research_context": {
                "datafields": {
                    "available": True,
                    "field_ids": [f"field_{idx}" for idx in range(400)],
                    "fields": huge_fields,
                },
                "knowledge": {"wqb_rules": "r" * 6000},
                "experiment_plan": {"mode": "explore_new_family", "avoid": ["est_q_pre_mean"]},
                "recent_failures": [
                    {
                        "id": idx,
                        "expression": "group_rank(ts_rank(failed_%s, 63), industry)" % idx,
                        "metrics": {"sharpe": 0.1},
                    }
                    for idx in range(20)
                ],
                "history_memory": {
                    "scanned_candidates": 400,
                    "top_fields": [{"field": f"field_{idx}", "count": 20 - idx} for idx in range(20)],
                },
                "candidate_queues": {
                    "watchlist": [
                        {
                            "id": idx,
                            "expression": "group_rank(ts_rank(field_%s, 63), industry)" % idx,
                            "metrics": {"sharpe": 1.0},
                        }
                        for idx in range(50)
                    ]
                },
            },
        }
        controller = Controller()
        critic = Critic()
        generator = Generator()
        client = MultiModelAIClient(controllers=[controller, critic], generators=[generator])

        client.generate_candidates(1, context)

        controller_payload_size = len(json.dumps(controller.context, sort_keys=True))
        critic_payload_size = len(json.dumps(critic.context, sort_keys=True))
        self.assertLess(controller_payload_size, 50000)
        self.assertLess(critic_payload_size, 50000)
        self.assertNotIn("fields", controller.context["research_context"]["datafields"])
        self.assertIn("field_399", controller.context["research_context"]["datafields"]["field_ids"])
        self.assertIn("fields", generator.context["research_context"]["datafields"])
        self.assertEqual(len(generator.context["research_context"]["datafields"]["fields"]), 400)
        self.assertLessEqual(len(generator.context["research_context"]["candidate_queues"]["watchlist"]), 5)
        self.assertLessEqual(len(generator.context["research_context"]["recent_failures"]), 6)
        self.assertIn("history_memory", generator.context["research_context"])

    def test_worker_records_ai_generation_failure(self):
        class FailingAI:
            def generate_candidates(self, batch_size, context):
                raise AIClientError("boom")

        with tempfile.TemporaryDirectory() as tmp:
            store = AlphaStore(Path(tmp) / "alpha.db")
            store.init()
            worker = AlphaWorker(
                store=store,
                ai_client=FailingAI(),
                brain_client=LocalBrainClient(),
                policy=SubmissionPolicy(auto_submit=False),
                batch_size=1,
            )

            summary = worker.run_once()

            self.assertEqual(summary["failed"], 1)
            events = store.events_for_candidate(None)
            self.assertTrue(any(event["event_type"] == "ai_generation_error" for event in events))

    def test_worker_stops_before_fallback_after_ai_quota_error(self):
        class QuotaFailingAI:
            def __init__(self):
                self.calls = 0

            def generate_candidates(self, batch_size, context):
                self.calls += 1
                raise AIClientError(
                    'AI request failed: HTTP 403: {"error":{"message":"预扣费额度失败, 用户剩余额度: $0.001, '
                    '需要预扣费额度: $0.029","code":"insufficient_user_quota"}}'
                )

        with tempfile.TemporaryDirectory() as tmp:
            store = AlphaStore(Path(tmp) / "alpha.db")
            store.init()
            ai = QuotaFailingAI()
            worker = AlphaWorker(
                store=store,
                ai_client=ai,
                brain_client=LocalBrainClient(),
                policy=SubmissionPolicy(auto_submit=False, max_retries=3),
                batch_size=1,
            )

            summary = worker.run_once()

            self.assertEqual(ai.calls, 1)
            self.assertEqual(summary["generated"], 0)
            self.assertEqual(summary["failed"], 1)
            self.assertEqual(summary["ai_quota_blocked"], 1)
            self.assertEqual(store.list_candidates(), [])
            events = store.events_for_candidate(None)
            generation_errors = [event for event in events if event["event_type"] == "ai_generation_error"]
            self.assertEqual(len(generation_errors), 1)
            metadata = json.loads(generation_errors[0]["metadata_json"])
            self.assertTrue(metadata["non_retryable"])
            self.assertEqual(metadata["reason"], "ai_quota_blocked")
            self.assertFalse(any(event["event_type"] == "deterministic_generation_fallback" for event in events))

    def test_worker_quota_block_does_not_generate_safe_datafield_fallback(self):
        class QuotaFailingAI:
            def generate_candidates(self, batch_size, context):
                raise AIClientError(
                    'AI request failed: HTTP 403: {"error":{"code":"insufficient_user_quota"}}'
                )

        class MixedFieldBrain(LocalBrainClient):
            def discover_datafields(self, settings, search_terms=None, max_fields=120):
                return [
                    {
                        "id": "raw_insider_signal",
                        "type": "MATRIX",
                        "dataset": {"id": "insider_trx_matrix", "name": "Insider transactions"},
                        "category": {"name": "Insiders"},
                        "coverage": 0.9,
                        "userCount": 0,
                        "alphaCount": 0,
                        "pyramidMultiplier": 1.8,
                        "description": "Raw insider transaction value.",
                    },
                    {
                        "id": "safe_quality_signal",
                        "type": "MATRIX",
                        "dataset": {"id": "quality12", "name": "Quality"},
                        "category": {"name": "Fundamental"},
                        "coverage": 0.8,
                        "userCount": 0,
                        "alphaCount": 0,
                        "pyramidMultiplier": 1.4,
                        "description": "Quality signal.",
                    },
                ]

        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"ALPHA_FIELD_CACHE_DIR": str(Path(tmp) / "field_cache")}):
                store = AlphaStore(Path(tmp) / "alpha.db")
                store.init()
                worker = AlphaWorker(
                    store=store,
                    ai_client=QuotaFailingAI(),
                    brain_client=MixedFieldBrain(),
                    policy=SubmissionPolicy(auto_submit=False, max_retries=1),
                    batch_size=1,
                )

                summary = worker.run_once()

            self.assertEqual(summary["generated"], 0)
            self.assertEqual(summary["failed"], 1)
            self.assertEqual(summary["ai_quota_blocked"], 1)
            self.assertEqual(store.list_candidates(), [])

    def test_worker_retries_ai_generation_before_success(self):
        class FlakyAI:
            def __init__(self):
                self.calls = 0

            def generate_candidates(self, batch_size, context):
                self.calls += 1
                if self.calls == 1:
                    raise AIClientError("temporary")
                return OpenAICompatibleAIClient(
                    api_key="test",
                    model="model-x",
                    transport=lambda _payload: {
                        "choices": [
                            {
                                "message": {
                                    "content": '{"candidates":[{"expression":"group_rank(ts_rank(mdl_mock_score,22),industry)"}]}'
                                }
                            }
                        ]
                    },
                ).generate_candidates(batch_size, context)

        with tempfile.TemporaryDirectory() as tmp:
            store = AlphaStore(Path(tmp) / "alpha.db")
            store.init()
            ai = FlakyAI()
            worker = AlphaWorker(
                store=store,
                ai_client=ai,
                brain_client=LocalBrainClient(),
                policy=SubmissionPolicy(auto_submit=False, max_retries=2),
                batch_size=1,
            )

            summary = worker.run_once()

            self.assertEqual(ai.calls, 2)
            self.assertEqual(summary["approved"], 1)
            event_types = [event["event_type"] for event in store.events_for_candidate(None)]
            self.assertEqual(event_types.count("ai_generation_error"), 1)
            self.assertIn("experiment_plan", event_types)


if __name__ == "__main__":
    unittest.main()
