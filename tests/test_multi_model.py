from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from alpha.clients import AIClientError, MultiModelAIClient, parse_model_profiles_from_env
from alpha.models import CandidateSpec


class FakeController:
    profile_name = "flow"
    model = "grok-4-1-fast-reasoning"
    role = "controller"

    def __init__(self, plan):
        self.plan = plan
        self.calls = []

    def plan_generation(self, batch_size, context, generator_profiles):
        self.calls.append((batch_size, context, generator_profiles))
        return self.plan


class SlowController(FakeController):
    def __init__(self, plan, delay_seconds, request_timeout=0.05):
        super().__init__(plan)
        self.delay_seconds = delay_seconds
        self.request_timeout = request_timeout

    def plan_generation(self, batch_size, context, generator_profiles):
        time.sleep(self.delay_seconds)
        return super().plan_generation(batch_size, context, generator_profiles)


class FinalizingController(FakeController):
    def __init__(self, plan, final_plan):
        super().__init__(plan)
        self.final_plan = final_plan
        self.final_calls = []

    def finalize_generation_plan(self, batch_size, context, generator_profiles, initial_plan, critiques):
        self.final_calls.append((batch_size, context, generator_profiles, initial_plan, critiques))
        return self.final_plan


class RepairingController(FinalizingController):
    def __init__(self, plan, final_plan, repair_decision):
        super().__init__(plan, final_plan)
        self.repair_decision = repair_decision
        self.repair_calls = []

    def repair_candidate_batch(
        self,
        batch_size,
        context,
        generator_profiles,
        final_plan,
        accepted_candidates,
        validator_rejections,
        model_errors,
        critic_feedback,
        remaining_slots,
    ):
        self.repair_calls.append(
            (
                batch_size,
                context,
                generator_profiles,
                final_plan,
                accepted_candidates,
                validator_rejections,
                model_errors,
                critic_feedback,
                remaining_slots,
            )
        )
        return self.repair_decision


class FakeCritic:
    profile_name = "flow-critic"
    model = "deepseek-v4-pro-free"
    role = "critic"

    def __init__(self, critique):
        self.critique = critique
        self.calls = []

    def critique_generation_plan(self, batch_size, context, generator_profiles, initial_plan):
        self.calls.append((batch_size, context, generator_profiles, initial_plan))
        return self.critique


class CandidateCritic(FakeCritic):
    def __init__(self, critique):
        super().__init__(critique)
        self.candidate_calls = []

    def critique_candidate_batch(
        self,
        batch_size,
        context,
        generator_profiles,
        final_plan,
        accepted_candidates,
        validator_rejections,
        model_errors,
        remaining_slots,
    ):
        self.candidate_calls.append(
            (
                batch_size,
                context,
                generator_profiles,
                final_plan,
                accepted_candidates,
                validator_rejections,
                model_errors,
                remaining_slots,
            )
        )
        return self.critique


class FakeGenerator:
    def __init__(self, profile_name, model, role="generator"):
        self.profile_name = profile_name
        self.model = model
        self.role = role
        self.calls = []

    def generate_candidates(self, batch_size, context):
        self.calls.append((batch_size, context))
        profile_offset = sum(ord(char) for char in self.profile_name) % 50
        return [
            CandidateSpec(
                expression=f"group_rank(ts_rank(ts_delta(close,{profile_offset + idx + 2}),22),industry)",
                settings={"region": context.get("region", "USA"), "delay": context.get("delay", 1)},
                source=f"model:{self.profile_name}",
                metadata={"model_profile": self.profile_name, "model": self.model},
            )
            for idx in range(batch_size)
        ]


class FailingGenerator:
    def __init__(self, profile_name, model, error):
        self.profile_name = profile_name
        self.model = model
        self.role = "generator"
        self.error = error
        self.calls = []

    def generate_candidates(self, batch_size, context):
        self.calls.append((batch_size, context))
        raise self.error


class SlowGenerator(FakeGenerator):
    def __init__(self, profile_name, model, delay_seconds, role="generator", request_timeout=60.0):
        super().__init__(profile_name, model, role=role)
        self.delay_seconds = delay_seconds
        self.request_timeout = request_timeout

    def generate_candidates(self, batch_size, context):
        time.sleep(self.delay_seconds)
        return super().generate_candidates(batch_size, context)


class BatchLimitedGenerator(FakeGenerator):
    def __init__(self, profile_name, model, max_batch, role="generator"):
        super().__init__(profile_name, model, role=role)
        self.max_batch = max_batch

    def generate_candidates(self, batch_size, context):
        self.calls.append((batch_size, context))
        if batch_size > self.max_batch:
            raise RuntimeError(f"batch too large: {batch_size}")
        offset = len(self.calls) * 10
        return [
            CandidateSpec(
                expression=f"group_rank(ts_rank(ts_delta(close,{offset + idx + 2 + batch_size}),22),industry)",
                settings={"region": context.get("region", "USA"), "delay": context.get("delay", 1)},
                source=f"model:{self.profile_name}",
                metadata={"model_profile": self.profile_name, "model": self.model},
            )
            for idx in range(batch_size)
        ]


class FakeValidator:
    profile_name = "cheap-check"
    model = "gpt-5.4-nano"
    role = "validator"

    def __init__(self):
        self.calls = []

    def validate_candidate_specs(self, candidates, batch_size, context):
        self.calls.append((candidates, batch_size, context))
        return [
            CandidateSpec(
                expression=candidate.expression,
                settings=candidate.settings,
                source=candidate.source,
                metadata={**candidate.metadata, "validated_by": self.profile_name},
            )
            for candidate in candidates
        ]


class DropFirstPassValidator(FakeValidator):
    def validate_candidate_specs(self, candidates, batch_size, context):
        self.calls.append((candidates, batch_size, context))
        if len(self.calls) == 1:
            self.last_rejections = [
                {"expression": candidate.expression, "settings": candidate.settings, "reason": "too similar"}
                for candidate in candidates[1:]
            ]
            return [candidates[0]]
        self.last_rejections = []
        return list(candidates[:batch_size])


class MultiModelTests(unittest.TestCase):
    def test_parse_model_profiles_from_compact_env_uses_default_relay_credentials(self):
        with patch.dict(
            os.environ,
            {
                "AI_API_KEY": "shared-key",
                "AI_BASE_URL": "https://relay.example/v1",
                "AI_MODEL_PROFILES": (
                    "grok-4-1-fast-reasoning@controller,"
                    "gemini-3-flash-free@generator,"
                    "glm-4.5@optimizer,"
                    "gpt-5.4-nano@validator"
                ),
            },
            clear=True,
        ):
            profiles = parse_model_profiles_from_env()

        self.assertEqual([profile.role for profile in profiles], ["controller", "generator", "optimizer", "validator"])
        self.assertEqual([profile.model for profile in profiles], [
            "grok-4-1-fast-reasoning",
            "gemini-3-flash-free",
            "glm-4.5",
            "gpt-5.4-nano",
        ])
        self.assertTrue(all(profile.api_key == "shared-key" for profile in profiles))
        self.assertTrue(all(profile.base_url == "https://relay.example/v1" for profile in profiles))

    def test_parse_model_profiles_from_file_uses_per_profile_env_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            (base / "grok.env").write_text(
                "AI_API_KEY=grok-key\nAI_BASE_URL=https://grok-relay.example/v1\n",
                encoding="utf-8",
            )
            (base / "gemini.env").write_text(
                "AI_API_KEY=gemini-key\nAI_BASE_URL=https://gemini-relay.example/v1\n",
                encoding="utf-8",
            )
            profiles_path = base / "profiles.json"
            profiles_path.write_text(
                json.dumps(
                    [
                        {
                            "name": "flow",
                            "role": "controller",
                            "model": "grok-4-1-fast-reasoning",
                            "env_file": "grok.env",
                        },
                        {
                            "name": "gemini",
                            "role": "generator",
                            "model": "gemini-3-flash-free",
                            "env_file": "gemini.env",
                            "request_timeout": 12,
                        },
                    ]
                ),
                encoding="utf-8",
            )

            with patch.dict(os.environ, {"AI_MODEL_PROFILES_FILE": str(profiles_path)}, clear=True):
                profiles = parse_model_profiles_from_env()

        self.assertEqual([profile.name for profile in profiles], ["flow", "gemini"])
        self.assertEqual([profile.api_key for profile in profiles], ["grok-key", "gemini-key"])
        self.assertEqual(
            [profile.base_url for profile in profiles],
            ["https://grok-relay.example/v1", "https://gemini-relay.example/v1"],
        )
        self.assertEqual(profiles[0].request_timeout, 60.0)
        self.assertEqual(profiles[1].request_timeout, 12.0)

    def test_parse_model_profiles_from_file_accepts_gbk_encoded_model_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            (base / "gemini.env").write_text(
                "AI_API_KEY=gemini-key\nAI_BASE_URL=https://gemini-relay.example/v1\n",
                encoding="utf-8",
            )
            profiles_path = base / "profiles.json"
            profiles_path.write_text(
                json.dumps(
                    [
                        {
                            "name": "G-1",
                            "role": "generator",
                            "model": "[次]gemini-3-flash-preview",
                            "env_file": "gemini.env",
                        }
                    ],
                    ensure_ascii=False,
                ),
                encoding="gbk",
            )

            with patch.dict(os.environ, {"AI_MODEL_PROFILES_FILE": str(profiles_path)}, clear=True):
                profiles = parse_model_profiles_from_env()

        self.assertEqual(profiles[0].model, "[次]gemini-3-flash-preview")
        self.assertEqual(profiles[0].api_key, "gemini-key")

    def test_parse_model_profiles_accepts_critic_role(self):
        with patch.dict(
            os.environ,
            {
                "AI_API_KEY": "shared-key",
                "AI_BASE_URL": "https://relay.example/v1",
                "AI_MODEL_PROFILES": (
                    "gpt-5.4-pro@controller,"
                    "deepseek-v4-pro-free@critic,"
                    "gemini-3-flash-free@generator"
                ),
            },
            clear=True,
        ):
            profiles = parse_model_profiles_from_env()

        self.assertEqual([profile.role for profile in profiles], ["controller", "critic", "generator"])
        self.assertEqual(profiles[1].name, "critic")

    def test_multi_model_client_uses_controller_allocation_generators_and_validator(self):
        controller = FakeController({"allocation": {"gemini": 3, "glm": 2}})
        gemini = FakeGenerator("gemini", "gemini-3-flash-free")
        glm = FakeGenerator("glm", "glm-4.5")
        validator = FakeValidator()
        client = MultiModelAIClient(
            controllers=[controller],
            generators=[gemini, glm],
            validators=[validator],
        )

        candidates = client.generate_candidates(
            5,
            {
                "region": "USA",
                "delay": 0,
                "research_context": {"experiment_plan": {"mode": "fresh_exploration"}},
            },
        )

        self.assertEqual(gemini.calls[0][0], 3)
        self.assertEqual(glm.calls[0][0], 2)
        self.assertEqual(len(candidates), 5)
        self.assertEqual(len(validator.calls), 1)
        self.assertTrue(all(candidate.metadata["validated_by"] == "cheap-check" for candidate in candidates))
        self.assertTrue(all(candidate.metadata["orchestrated_by"] == "flow" for candidate in candidates))
        self.assertEqual(candidates[0].metadata["model_allocation"], {"gemini": 3, "glm": 2})
        controller_profiles = controller.calls[0][2]
        self.assertEqual([profile["name"] for profile in controller_profiles], ["gemini", "glm"])

    def test_multi_model_client_uses_critic_feedback_for_final_controller_plan(self):
        controller = FinalizingController(
            {
                "mode": "explore",
                "profile_guidance": {"gemini": {"objective": "initial narrow family"}},
            },
            {
                "mode": "explore",
                "profile_guidance": {
                    "gemini": {"objective": "field-native persistence"},
                    "glm": {"objective": "orthogonal normalized variant"},
                },
            },
        )
        critic = FakeCritic({"concerns": ["too narrow"], "recommendations": ["split field families"]})
        gemini = FakeGenerator("gemini", "gemini-3-flash-free")
        glm = FakeGenerator("glm", "glm-5.1-free")
        client = MultiModelAIClient(controllers=[controller, critic], generators=[gemini, glm])

        candidates = client.generate_candidates(
            8,
            {
                "region": "USA",
                "delay": 0,
                "research_context": {"experiment_plan": {"mode": "explore_new_family"}},
            },
        )

        self.assertEqual(len(critic.calls), 1)
        self.assertEqual(critic.calls[0][3]["profile_guidance"]["gemini"]["objective"], "initial narrow family")
        self.assertEqual(len(controller.final_calls), 1)
        self.assertEqual(controller.final_calls[0][4][0]["critic"], "flow-critic")
        self.assertEqual(gemini.calls[0][0], 4)
        self.assertEqual(glm.calls[0][0], 4)
        self.assertEqual(len(candidates), 8)
        self.assertEqual(client.last_plan["controller"], "flow")
        self.assertEqual(client.last_plan["critics"][0]["profile"], "flow-critic")
        self.assertEqual(
            glm.calls[0][1]["research_context"]["profile_guidance"]["objective"],
            "orthogonal normalized variant",
        )

    def test_multi_model_client_uses_controller_for_intra_round_repair_after_critic_feedback(self):
        controller = RepairingController(
            {
                "mode": "explore",
                "profile_guidance": {"gemini": {"objective": "initial family"}},
            },
            {
                "mode": "explore",
                "profile_guidance": {"gemini": {"objective": "final family"}},
            },
            {
                "action": "refill",
                "allocation": {"glm": 3},
                "profile_guidance": {
                    "glm": {
                        "objective": "replace rejected candidates with an orthogonal text family",
                        "avoid": ["initial family local variants"],
                    }
                },
                "rationale": "validator removed near-duplicates",
            },
        )
        critic = CandidateCritic({"concerns": ["accepted batch has one candidate"], "recommendations": ["refill glm"]})
        gemini = FakeGenerator("gemini", "gemini-3-flash-free")
        glm = FakeGenerator("glm", "glm-5.1-free")
        validator = DropFirstPassValidator()
        client = MultiModelAIClient(
            controllers=[controller, critic],
            generators=[gemini, glm],
            validators=[validator],
        )

        candidates = client.generate_candidates(
            4,
            {
                "region": "USA",
                "delay": 0,
                "research_context": {"experiment_plan": {"mode": "explore_new_family"}},
            },
        )

        self.assertEqual(len(candidates), 4)
        self.assertEqual(len(critic.candidate_calls), 1)
        self.assertEqual(critic.candidate_calls[0][7], 3)
        self.assertEqual(len(controller.repair_calls), 1)
        self.assertEqual(controller.repair_calls[0][7][0]["critic"], "flow-critic")
        self.assertEqual(controller.repair_calls[0][8], 3)
        self.assertEqual([call[0] for call in glm.calls], [2, 3])
        repair_context = glm.calls[1][1]["research_context"]
        self.assertEqual(
            repair_context["profile_guidance"]["objective"],
            "replace rejected candidates with an orthogonal text family",
        )
        self.assertEqual(repair_context["intra_round_repair"]["remaining_slots"], 3)
        self.assertEqual(len(validator.calls), 2)
        self.assertEqual(len(client.last_validator_rejections), 3)
        self.assertEqual(client.last_plan["intra_round_repair"]["action"], "refill")
        self.assertEqual(client.last_plan["intra_round_repair"]["refill_allocation"], {"glm": 3})
        self.assertTrue(any(candidate.metadata.get("intra_round_repair") for candidate in candidates))

    def test_multi_model_client_keeps_balanced_allocation_for_optimization(self):
        gemini = FakeGenerator("gemini", "gemini-3-flash-free")
        glm = FakeGenerator("glm", "glm-4.5", role="optimizer")
        client = MultiModelAIClient(generators=[gemini, glm])

        client.generate_candidates(
            8,
            {
                "region": "USA",
                "delay": 0,
                "research_context": {"experiment_plan": {"mode": "optimize_best"}},
            },
        )

        self.assertEqual(gemini.calls[0][0], 4)
        self.assertEqual(glm.calls[0][0], 4)

    def test_multi_model_client_does_not_give_optimizer_extra_allocation(self):
        gemini = FakeGenerator("gemini", "gemini-3-flash-free", role="generator")
        kimi = FakeGenerator("kimi", "kimi-k2.6-free", role="optimizer")
        client = MultiModelAIClient(generators=[gemini, kimi])

        client.generate_candidates(
            8,
            {
                "region": "USA",
                "delay": 0,
                "research_context": {"experiment_plan": {"mode": "optimize_best"}},
            },
        )

        self.assertEqual(gemini.calls[0][0], 4)
        self.assertEqual(kimi.calls[0][0], 4)

    def test_multi_model_client_keeps_partial_candidates_when_one_generator_fails(self):
        gemini = FailingGenerator("gemini", "gemini-3-flash-free", TimeoutError("read timed out"))
        glm = FakeGenerator("glm", "glm-4.5")
        client = MultiModelAIClient(generators=[gemini, glm])

        candidates = client.generate_candidates(
            4,
            {"region": "USA", "delay": 0, "research_context": {"experiment_plan": {"mode": "fresh_exploration"}}},
        )

        self.assertEqual(len(candidates), 2)
        self.assertEqual(candidates[0].source, "model:glm")
        self.assertEqual(client.last_errors[0]["profile"], "gemini")
        self.assertEqual(client.last_errors[0]["model"], "gemini-3-flash-free")
        self.assertIn("read timed out", client.last_errors[0]["error"])

    def test_multi_model_client_keeps_optimizer_allocation_during_exploration(self):
        controller = FakeController({"mode": "explore", "allocation": {"gemini": 4, "glm": 4}})
        gemini = FakeGenerator("gemini", "gemini-3-flash-free", role="generator")
        glm = FakeGenerator("glm", "glm-5.1-free", role="optimizer")
        client = MultiModelAIClient(controllers=[controller], generators=[gemini, glm])

        candidates = client.generate_candidates(
            8,
            {
                "region": "USA",
                "delay": 0,
                "research_context": {"experiment_plan": {"mode": "explore_new_family"}},
            },
        )

        self.assertEqual(gemini.calls[0][0], 4)
        self.assertEqual(glm.calls[0][0], 4)
        self.assertEqual(len(candidates), 8)
        self.assertEqual(client.last_plan["allocation"], {"gemini": 4, "glm": 4})

    def test_multi_model_client_ignores_controller_allocation_during_optimization(self):
        controller = FakeController(
            {
                "mode": "optimize",
                "allocation": {"gemini": 8},
                "profile_guidance": {
                    "gemini": {"objective": "quality family"},
                    "glm": {"objective": "revision family"},
                },
            }
        )
        gemini = FakeGenerator("gemini", "gemini-3-flash-free", role="generator")
        glm = FakeGenerator("glm", "gpt-5.4-pro", role="generator")
        client = MultiModelAIClient(controllers=[controller], generators=[gemini, glm])

        candidates = client.generate_candidates(
            8,
            {
                "region": "USA",
                "delay": 0,
                "research_context": {"experiment_plan": {"mode": "optimize_best"}},
            },
        )

        self.assertEqual(gemini.calls[0][0], 4)
        self.assertEqual(glm.calls[0][0], 4)
        self.assertEqual(len(candidates), 8)
        self.assertEqual(client.last_plan["allocation"], {"gemini": 4, "glm": 4})
        self.assertEqual(client.last_plan["profile_guidance"]["glm"]["objective"], "revision family")

    def test_multi_model_client_fallback_splits_exploration_between_generator_and_optimizer(self):
        gemini = FakeGenerator("gemini", "gemini-3-flash-free", role="generator")
        glm = FakeGenerator("glm", "glm-5.1-free", role="optimizer")
        client = MultiModelAIClient(generators=[gemini, glm])

        candidates = client.generate_candidates(
            8,
            {
                "region": "USA",
                "delay": 0,
                "research_context": {"experiment_plan": {"mode": "explore_new_family"}},
            },
        )

        self.assertEqual(gemini.calls[0][0], 4)
        self.assertEqual(glm.calls[0][0], 4)
        self.assertEqual(len(candidates), 8)
        self.assertEqual(client.last_plan["allocation"], {"gemini": 4, "glm": 4})

    def test_multi_model_client_falls_back_when_controller_hard_times_out(self):
        controller = SlowController({"mode": "explore", "allocation": {"gemini": 8}}, delay_seconds=0.4)
        gemini = FakeGenerator("gemini", "gemini-3-flash-free", role="generator")
        glm = FakeGenerator("glm", "glm-5.1-free", role="optimizer")
        client = MultiModelAIClient(controllers=[controller], generators=[gemini, glm])

        started = time.monotonic()
        candidates = client.generate_candidates(
            8,
            {"region": "USA", "delay": 0, "research_context": {"experiment_plan": {"mode": "explore_new_family"}}},
        )
        elapsed = time.monotonic() - started

        self.assertLess(elapsed, 0.3)
        self.assertEqual(len(candidates), 8)
        self.assertEqual(client.last_plan["allocation"], {"gemini": 4, "glm": 4})
        self.assertEqual(client.last_errors[0]["profile"], "flow")
        self.assertIn("timed out", client.last_errors[0]["error"])

    def test_multi_model_client_runs_allocated_generators_in_parallel(self):
        controller = FakeController({"mode": "optimize", "allocation": {"gemini": 1, "glm": 1}})
        gemini = SlowGenerator("gemini", "gemini-3-flash-free", 0.12, role="generator")
        glm = SlowGenerator("glm", "glm-5.1-free", 0.12, role="optimizer")
        client = MultiModelAIClient(controllers=[controller], generators=[gemini, glm])

        started = time.monotonic()
        candidates = client.generate_candidates(
            2,
            {"region": "USA", "delay": 0, "research_context": {"experiment_plan": {"mode": "optimize_best"}}},
        )
        elapsed = time.monotonic() - started

        self.assertLess(elapsed, 0.20)
        self.assertEqual(len(candidates), 2)

    def test_multi_model_client_keeps_ready_generator_when_peer_hard_times_out(self):
        controller = FakeController({"mode": "explore", "allocation": {"gemini": 4, "glm": 4}})
        gemini = FakeGenerator("gemini", "gemini-3-flash-free", role="generator")
        glm = SlowGenerator("glm", "glm-5.1-free", 0.4, role="optimizer", request_timeout=0.05)
        client = MultiModelAIClient(controllers=[controller], generators=[gemini, glm])

        started = time.monotonic()
        candidates = client.generate_candidates(
            8,
            {"region": "USA", "delay": 0, "research_context": {"experiment_plan": {"mode": "explore_new_family"}}},
        )
        elapsed = time.monotonic() - started

        self.assertLess(elapsed, 0.3)
        self.assertEqual(len(candidates), 4)
        self.assertTrue(all(candidate.source == "model:gemini" for candidate in candidates))
        self.assertEqual(client.last_errors[0]["profile"], "glm")
        self.assertIn("timed out", client.last_errors[0]["error"])

    def test_multi_model_client_splits_large_generator_request_after_failure(self):
        controller = FakeController({"mode": "explore", "allocation": {"gemini": 8}})
        gemini = BatchLimitedGenerator("gemini", "gemini-3-flash-free", max_batch=4)
        client = MultiModelAIClient(controllers=[controller], generators=[gemini])

        candidates = client.generate_candidates(
            8,
            {"region": "USA", "delay": 0, "research_context": {"experiment_plan": {"mode": "explore_new_family"}}},
        )

        self.assertEqual([call[0] for call in gemini.calls], [8, 4, 4])
        self.assertEqual(len(candidates), 8)
        self.assertEqual(client.last_errors[0]["profile"], "gemini")
        self.assertIn("batch too large", client.last_errors[0]["error"])

    def test_multi_model_client_keeps_same_template_for_different_fields(self):
        class FieldFamilyGenerator:
            profile_name = "G-1"
            model = "gemini-3-flash-free"
            role = "generator"

            def generate_candidates(self, batch_size, context):
                settings = {"region": "USA", "delay": 0}
                return [
                    CandidateSpec(
                        "group_rank(ts_rank(ts_backfill(analyst_value_signal, 66), 33), industry)",
                        settings=settings,
                        source="model:G-1",
                    ),
                    CandidateSpec(
                        "group_rank(ts_rank(ts_backfill(news_sentiment_signal, 120), 63), industry)",
                        settings=settings,
                        source="model:G-1",
                    ),
                ][:batch_size]

        client = MultiModelAIClient(generators=[FieldFamilyGenerator()])

        candidates = client.generate_candidates(
            2,
            {
                "region": "USA",
                "delay": 0,
                "research_context": {
                    "generation_policy": {"avoid_structural_duplicates": True},
                    "experiment_plan": {"mode": "explore_new_family"},
                },
            },
        )

        self.assertEqual(len(candidates), 2)

    def test_multi_model_client_does_not_split_retry_nonrecoverable_model_errors(self):
        controller = FakeController({"mode": "explore", "allocation": {"minimax": 4}})
        minimax = FailingGenerator(
            "minimax",
            "MiniMax-M2.7-free",
            RuntimeError("AI request failed: HTTP Error 402: Payment Required"),
        )
        client = MultiModelAIClient(controllers=[controller], generators=[minimax])

        with self.assertRaises(AIClientError):
            client.generate_candidates(
                4,
                {"region": "USA", "delay": 0, "research_context": {"experiment_plan": {"mode": "explore_new_family"}}},
            )

        self.assertEqual([call[0] for call in minimax.calls], [4])
        self.assertEqual(len(client.last_errors), 1)
        self.assertIn("402", client.last_errors[0]["error"])

    def test_multi_model_validator_preserves_setting_sweep_variants_with_same_expression(self):
        class SettingsDroppingValidator(FakeValidator):
            def validate_candidate_specs(self, candidates, batch_size, context):
                self.calls.append((candidates, batch_size, context))
                return [
                    CandidateSpec(
                        expression=candidate.expression,
                        settings={},
                        source=candidate.source,
                        metadata={**candidate.metadata, "validated_by": self.profile_name},
                    )
                    for candidate in candidates
                ]

        validator = SettingsDroppingValidator()
        client = MultiModelAIClient(generators=[FakeGenerator("G-1", "gemini-3-flash-free")], validators=[validator])
        candidates = [
            CandidateSpec(
                expression="rank(group_rank(ts_mean(close,30),industry))",
                settings={"region": "USA", "delay": 0, "decay": 0},
                source="planner_setting_sweep",
            ),
            CandidateSpec(
                expression="rank(group_rank(ts_mean(close,30),industry))",
                settings={"region": "USA", "delay": 0, "decay": 6},
                source="planner_setting_sweep",
            ),
        ]

        validated = client.validate_candidate_specs(
            candidates,
            2,
            {"region": "USA", "delay": 0, "research_context": {"experiment_plan": {"mode": "setting_sweep"}}},
        )

        self.assertEqual(len(validated), 2)
        self.assertEqual([candidate.settings["decay"] for candidate in validated], [0, 6])


if __name__ == "__main__":
    unittest.main()
