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


class DuplicateThenUniqueGenerator(FakeGenerator):
    def generate_candidates(self, batch_size, context):
        self.calls.append((batch_size, context))
        profile_offset = sum(ord(char) for char in self.profile_name) % 50
        if len(self.calls) == 1:
            expression = f"rank(ts_mean(close,{profile_offset + 20}))"
            return [
                CandidateSpec(
                    expression=expression,
                    settings={"region": context.get("region", "USA"), "delay": context.get("delay", 1)},
                    source=f"model:{self.profile_name}",
                    metadata={"model_profile": self.profile_name, "model": self.model},
                )
                for _idx in range(batch_size)
            ]
        refill_offset = profile_offset + 100 * len(self.calls)
        return [
            CandidateSpec(
                expression=f"rank(ts_mean(close,{refill_offset + idx + 20}))",
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


class InvalidThenValidGenerator(FakeGenerator):
    def generate_candidates(self, batch_size, context):
        self.calls.append((batch_size, context))
        if len(self.calls) == 1:
            return [
                CandidateSpec(
                    expression="group_mean(ts_mean(close,22),industry)",
                    settings={"region": context.get("region", "USA"), "delay": context.get("delay", 1)},
                    source=f"model:{self.profile_name}",
                    metadata={"model_profile": self.profile_name, "model": self.model},
                )
            ]
        return [
            CandidateSpec(
                expression="rank(ts_mean(close,22))",
                settings={"region": context.get("region", "USA"), "delay": context.get("delay", 1)},
                source=f"model:{self.profile_name}",
                metadata={"model_profile": self.profile_name, "model": self.model},
            )
        ]


class RawVectorGenerator(FakeGenerator):
    def generate_candidates(self, batch_size, context):
        self.calls.append((batch_size, context))
        return [
            CandidateSpec(
                expression=(
                    "rank(if_else(greater(normalize(anl83_analyst_fkgl_qa),"
                    "normalize(anl83_ceo_forw_sent_pres)),"
                    "normalize(anl83_analyst_pos_logit_qa),"
                    "multiply(-1,normalize(anl83_cfo_sent_score_qa))))"
                ),
                settings={"region": context.get("region", "USA"), "delay": context.get("delay", 1)},
                source=f"model:{self.profile_name}",
                metadata={"model_profile": self.profile_name, "model": self.model},
            )
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
    def test_multi_model_from_env_defaults_to_lean_personal_orchestration(self):
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
            client = MultiModelAIClient.from_env()

        self.assertEqual(client.orchestration_mode, "lean")
        self.assertEqual(client.max_active_generators, 1)

    def test_lean_orchestration_skips_controller_validator_and_uses_one_generator(self):
        controller = FakeController({"allocation": {"gemini": 3, "glm": 2}})
        gemini = FakeGenerator("gemini", "gemini-3-flash-free")
        glm = FakeGenerator("glm", "glm-4.5")
        validator = FakeValidator()
        client = MultiModelAIClient(
            controllers=[controller],
            generators=[gemini, glm],
            validators=[validator],
            orchestration_mode="lean",
            max_active_generators=1,
        )

        candidates = client.generate_candidates(
            5,
            {
                "region": "USA",
                "delay": 0,
                "research_context": {"experiment_plan": {"mode": "fresh_exploration"}},
            },
        )

        self.assertEqual(controller.calls, [])
        self.assertEqual(validator.calls, [])
        self.assertEqual(gemini.calls[0][0], 5)
        self.assertEqual(glm.calls, [])
        self.assertEqual(len(candidates), 5)
        self.assertEqual(client.last_plan["controller"], "lean")
        self.assertEqual(client.last_plan["allocation"], {"gemini": 5})
        self.assertTrue(all(candidate.metadata["orchestrated_by"] == "lean" for candidate in candidates))

    def test_lean_orchestration_rotates_to_less_used_generator(self):
        gemini = FakeGenerator("G-1", "DeepSeek-V4-Pro")
        kimi = FakeGenerator("G-2", "Kimi-K2.6")
        client = MultiModelAIClient(
            generators=[gemini, kimi],
            orchestration_mode="lean",
            max_active_generators=1,
        )

        candidates = client.generate_candidates(
            6,
            {
                "region": "USA",
                "delay": 0,
                "research_context": {
                    "experiment_plan": {"mode": "explore_new_family"},
                    "active_run_history_memory": {
                        "profile_outcomes": {
                            "G-1": {"generated": 20, "failed": 18},
                            "G-2": {"generated": 2, "failed": 1},
                        }
                    },
                },
            },
        )

        self.assertEqual(gemini.calls, [])
        self.assertEqual(kimi.calls[0][0], 6)
        self.assertEqual(client.last_plan["allocation"], {"G-2": 6})
        self.assertTrue(all(candidate.source == "model:G-2" for candidate in candidates))

    def test_targeted_repair_keeps_lean_personal_orchestration_under_lean_default(self):
        controller = FakeController({"allocation": {"gemini": 2, "glm": 2}})
        gemini = FakeGenerator("gemini", "gemini-3-flash-free")
        glm = FakeGenerator("glm", "glm-4.5")
        validator = FakeValidator()
        client = MultiModelAIClient(
            controllers=[controller],
            generators=[gemini, glm],
            validators=[validator],
            orchestration_mode="lean",
            max_active_generators=1,
            targeted_repair_generators=2,
        )

        candidates = client.generate_candidates(
            4,
            {
                "region": "USA",
                "delay": 0,
                "research_context": {
                    "experiment_plan": {
                        "mode": "optimize_best",
                        "targeted_repair": {"active": True, "orchestration": "deep_repair"},
                    }
                },
            },
        )

        self.assertEqual(controller.calls, [])
        self.assertEqual(validator.calls, [])
        self.assertEqual(gemini.calls[0][0], 4)
        self.assertEqual(glm.calls, [])
        self.assertEqual(len(candidates), 4)
        self.assertEqual(client.last_plan["controller"], "lean")
        self.assertEqual(client.last_plan["allocation"], {"gemini": 4})
        self.assertTrue(all(candidate.metadata["orchestrated_by"] == "lean" for candidate in candidates))

    def test_targeted_repair_can_use_deep_orchestration_when_explicitly_enabled(self):
        controller = FakeController({"allocation": {"gemini": 2, "glm": 2}})
        gemini = FakeGenerator("gemini", "gemini-3-flash-free")
        glm = FakeGenerator("glm", "glm-4.5")
        validator = FakeValidator()
        client = MultiModelAIClient(
            controllers=[controller],
            generators=[gemini, glm],
            validators=[validator],
            orchestration_mode="deep",
            max_active_generators=1,
            targeted_repair_generators=2,
        )

        candidates = client.generate_candidates(
            4,
            {
                "region": "USA",
                "delay": 0,
                "research_context": {
                    "experiment_plan": {
                        "mode": "optimize_best",
                        "targeted_repair": {"active": True, "orchestration": "deep_repair"},
                    }
                },
            },
        )

        self.assertEqual(len(controller.calls), 1)
        self.assertEqual(len(validator.calls), 1)
        self.assertEqual(gemini.calls[0][0], 2)
        self.assertEqual(glm.calls[0][0], 2)
        self.assertEqual(len(candidates), 4)
        self.assertEqual(client.last_plan["controller"], "flow")
        self.assertEqual(client.last_plan["allocation"], {"gemini": 2, "glm": 2})
        self.assertTrue(all(candidate.metadata["orchestrated_by"] == "flow" for candidate in candidates))

    def test_default_profile_file_excludes_known_unavailable_models(self):
        profiles_path = Path(__file__).resolve().parents[1] / "config" / "ai_model_profiles.json"
        profiles = json.loads(profiles_path.read_text(encoding="utf-8"))
        models = {str(profile.get("model") or "") for profile in profiles}

        self.assertNotIn("gpt-5.3-codex-A", models)

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

    def test_multi_model_client_refills_when_all_candidates_fail_local_preflight(self):
        controller = RepairingController(
            {"mode": "explore", "profile_guidance": {"gemini": {"objective": "initial family"}}},
            {"mode": "explore", "profile_guidance": {"gemini": {"objective": "final family"}}},
            {"action": "accept", "rationale": "no accepted candidates"},
        )
        generator = InvalidThenValidGenerator("gemini", "gemini-3-flash-free")
        client = MultiModelAIClient(controllers=[controller], generators=[generator])

        candidates = client.generate_candidates(
            1,
            {
                "region": "USA",
                "delay": 0,
                "research_context": {"experiment_plan": {"mode": "explore_new_family"}},
            },
        )

        self.assertEqual([candidate.expression for candidate in candidates], ["rank(ts_mean(close,22))"])
        self.assertEqual([call[0] for call in generator.calls], [1, 1])
        self.assertEqual(client.last_plan["intra_round_repair"]["action"], "refill")
        self.assertEqual(client.last_plan["intra_round_repair"]["reason"], "empty_local_preflight_batch")
        self.assertTrue(candidates[0].metadata.get("intra_round_repair"))

    def test_multi_model_client_locally_repairs_raw_vector_candidate_before_ai_refill(self):
        controller = RepairingController(
            {"mode": "explore", "profile_guidance": {"gemini": {"objective": "analyst83 route"}}},
            {"mode": "explore", "profile_guidance": {"gemini": {"objective": "analyst83 route"}}},
            {"action": "refill", "allocation": {"gemini": 1}, "rationale": "fallback refill"},
        )
        generator = RawVectorGenerator("gemini", "gemini-3-flash-free")
        client = MultiModelAIClient(controllers=[controller], generators=[generator])

        candidates = client.generate_candidates(
            1,
            {
                "region": "USA",
                "delay": 0,
                "research_context": {
                    "experiment_plan": {"mode": "explore_new_family"},
                    "datafields": {
                        "field_ids": [
                            "anl83_analyst_fkgl_qa",
                            "anl83_ceo_forw_sent_pres",
                            "anl83_analyst_pos_logit_qa",
                            "anl83_cfo_sent_score_qa",
                        ],
                        "field_types": {
                            "anl83_analyst_fkgl_qa": "VECTOR",
                            "anl83_ceo_forw_sent_pres": "VECTOR",
                            "anl83_analyst_pos_logit_qa": "VECTOR",
                            "anl83_cfo_sent_score_qa": "VECTOR",
                        },
                        "fields": [
                            {"id": "anl83_analyst_fkgl_qa", "type": "VECTOR", "category": "Analyst"},
                            {"id": "anl83_ceo_forw_sent_pres", "type": "VECTOR", "category": "Analyst"},
                            {"id": "anl83_analyst_pos_logit_qa", "type": "VECTOR", "category": "Analyst"},
                            {"id": "anl83_cfo_sent_score_qa", "type": "VECTOR", "category": "Analyst"},
                        ],
                    },
                },
            },
        )

        self.assertEqual(len(candidates), 1)
        self.assertIn("normalize(vec_avg(anl83_analyst_fkgl_qa))", candidates[0].expression)
        self.assertIn("normalize(vec_avg(anl83_ceo_forw_sent_pres))", candidates[0].expression)
        self.assertEqual([call[0] for call in generator.calls], [1])
        self.assertEqual(controller.repair_calls, [])
        self.assertEqual(client.last_plan["intra_round_repair"]["action"], "local_vector_repair")
        self.assertTrue(candidates[0].metadata.get("local_preflight_repair"))

    def test_multi_model_client_locally_repairs_raw_vector_candidate_without_controller(self):
        generator = RawVectorGenerator("gemini", "gemini-3-flash-free")
        client = MultiModelAIClient(generators=[generator])

        candidates = client.generate_candidates(
            1,
            {
                "region": "USA",
                "delay": 0,
                "research_context": {
                    "experiment_plan": {"mode": "explore_new_family"},
                    "datafields": {
                        "field_ids": [
                            "anl83_analyst_fkgl_qa",
                            "anl83_ceo_forw_sent_pres",
                            "anl83_analyst_pos_logit_qa",
                            "anl83_cfo_sent_score_qa",
                        ],
                        "field_types": {
                            "anl83_analyst_fkgl_qa": "VECTOR",
                            "anl83_ceo_forw_sent_pres": "VECTOR",
                            "anl83_analyst_pos_logit_qa": "VECTOR",
                            "anl83_cfo_sent_score_qa": "VECTOR",
                        },
                        "fields": [
                            {"id": "anl83_analyst_fkgl_qa", "type": "VECTOR", "category": "Analyst"},
                            {"id": "anl83_ceo_forw_sent_pres", "type": "VECTOR", "category": "Analyst"},
                            {"id": "anl83_analyst_pos_logit_qa", "type": "VECTOR", "category": "Analyst"},
                            {"id": "anl83_cfo_sent_score_qa", "type": "VECTOR", "category": "Analyst"},
                        ],
                    },
                },
            },
        )

        self.assertEqual(len(candidates), 1)
        self.assertIn("normalize(vec_avg(anl83_analyst_fkgl_qa))", candidates[0].expression)
        self.assertEqual(client.last_plan["intra_round_repair"]["action"], "local_vector_repair")
        self.assertTrue(candidates[0].metadata.get("local_preflight_repair"))

    def test_multi_model_client_respects_explicit_abandon_after_empty_local_preflight(self):
        controller = RepairingController(
            {"mode": "explore", "profile_guidance": {"gemini": {"objective": "initial family"}}},
            {"mode": "explore", "profile_guidance": {"gemini": {"objective": "final family"}}},
            {"action": "abandon", "rationale": "leave the slot empty"},
        )
        generator = InvalidThenValidGenerator("gemini", "gemini-3-flash-free")
        client = MultiModelAIClient(controllers=[controller], generators=[generator])

        candidates = client.generate_candidates(
            1,
            {
                "region": "USA",
                "delay": 0,
                "research_context": {"experiment_plan": {"mode": "explore_new_family"}},
            },
        )

        self.assertEqual(candidates, [])
        self.assertEqual([call[0] for call in generator.calls], [1])
        self.assertEqual(client.last_plan["intra_round_repair"]["action"], "abandon")
        self.assertNotIn("reason", client.last_plan["intra_round_repair"])

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

    def test_multi_model_client_refills_balanced_lean_batch_after_filtering(self):
        gemini = DuplicateThenUniqueGenerator("gemini", "gemini-3-flash-free")
        glm = DuplicateThenUniqueGenerator("glm", "glm-4.5")
        client = MultiModelAIClient(
            generators=[gemini, glm],
            orchestration_mode="lean",
            max_active_generators=0,
        )

        candidates = client.generate_candidates(
            8,
            {
                "region": "USA",
                "delay": 0,
                "research_context": {"experiment_plan": {"mode": "explore_new_family"}},
            },
        )

        self.assertEqual(len(candidates), 8)
        self.assertEqual([call[0] for call in gemini.calls], [4, 3])
        self.assertEqual([call[0] for call in glm.calls], [4, 3])
        self.assertEqual(client.last_plan["allocation"], {"gemini": 4, "glm": 4})
        self.assertEqual(client.last_plan["intra_round_repair"]["action"], "refill")
        self.assertEqual(client.last_plan["intra_round_repair"]["reason"], "balanced_lean_underfilled")
        self.assertEqual(client.last_plan["intra_round_repair"]["final_count"], 8)

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

    def test_multi_model_client_does_not_inject_profile_guidance_into_optimize_generators(self):
        controller = FakeController(
            {
                "mode": "explore",
                "profile_guidance": {
                    "gemini": {
                        "objective": "fresh NEWS route",
                        "field_family": "NEWS primary only",
                    },
                    "glm": {
                        "objective": "fresh RISK route",
                        "field_family": "RISK primary only",
                    },
                },
            }
        )
        gemini = FakeGenerator("gemini", "gemini-3-flash-free", role="generator")
        glm = FakeGenerator("glm", "gpt-5.4-pro", role="generator")
        client = MultiModelAIClient(controllers=[controller], generators=[gemini, glm])

        candidates = client.generate_candidates(
            4,
            {
                "region": "USA",
                "delay": 0,
                "research_context": {
                    "experiment_plan": {
                        "mode": "optimize_best",
                        "target_candidate_id": 9906,
                        "target_expression": "rank(ts_decay_linear(ts_backfill(vec_avg(fnd23_significance), 120), 20))",
                        "keep": ["fnd23_significance"],
                    }
                },
            },
        )

        self.assertEqual(client.last_plan["profile_guidance"]["gemini"]["field_family"], "NEWS primary only")
        self.assertNotIn("profile_guidance", gemini.calls[0][1]["research_context"])
        self.assertNotIn("profile_guidance", glm.calls[0][1]["research_context"])
        self.assertFalse(any("profile_guidance" in candidate.metadata for candidate in candidates))

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

    def test_multi_model_client_caps_same_structure_when_policy_requests_it(self):
        class SameStructureGenerator:
            profile_name = "G-1"
            model = "gemini-3-flash-free"
            role = "generator"

            def generate_candidates(self, batch_size, context):
                settings = {"region": "MEA", "delay": 1}
                return [
                    CandidateSpec(
                        f"group_rank(pasteurize(normalize(quantile(fresh_signal_{idx}))),industry)",
                        settings=settings,
                        source="model:G-1",
                    )
                    for idx in range(batch_size)
                ]

        client = MultiModelAIClient(generators=[SameStructureGenerator()])

        candidates = client.generate_candidates(
            4,
            {
                "region": "MEA",
                "delay": 1,
                "research_context": {
                    "generation_policy": {
                        "avoid_structural_duplicates": True,
                        "max_batch_candidates_per_structure": 2,
                    },
                    "experiment_plan": {
                        "mode": "explore_new_family",
                        "structure_diversity_control": {"max_batch_candidates_per_structure": 2},
                    },
                },
            },
        )

        self.assertEqual(len(candidates), 2)

    def test_multi_model_client_rejects_overused_structures_in_exploration(self):
        class MixedStructureGenerator:
            profile_name = "G-1"
            model = "gemini-3-flash-free"
            role = "generator"

            def generate_candidates(self, batch_size, context):
                settings = {"region": "MEA", "delay": 1}
                return [
                    CandidateSpec(
                        "group_rank(ts_rank(vec_avg(overused_signal),63),industry)",
                        settings=settings,
                        source="model:G-1",
                    ),
                    CandidateSpec(
                        "rank(vec_avg(fresh_signal))",
                        settings=settings,
                        source="model:G-1",
                    ),
                ]

        client = MultiModelAIClient(generators=[MixedStructureGenerator()])
        context = {
            "region": "MEA",
            "delay": 1,
            "research_context": {
                "experiment_plan": {
                    "mode": "explore_new_family",
                    "structure_diversity_control": {
                        "overused_structures": [
                            {"structure_key": "group_rank(ts_rank(vec_avg(field),#),group:industry)"}
                        ]
                    },
                },
            },
        }

        candidates = client.generate_candidates(2, context)

        self.assertEqual([candidate.expression for candidate in candidates], ["rank(vec_avg(fresh_signal))"])

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

    def test_multi_model_validator_applies_local_preflight_field_type_filter(self):
        validator = FakeValidator()
        client = MultiModelAIClient(validators=[validator])
        settings = {"region": "USA", "universe": "TOP500", "delay": 0}
        invalid = CandidateSpec(
            "rank(ts_mean(vector_signal, 20))",
            settings=settings,
            source="model:G-2",
        )
        valid = CandidateSpec(
            "rank(ts_mean(vec_avg(vector_signal), 20))",
            settings=settings,
            source="model:G-1",
        )
        context = {
            "region": "USA",
            "universe": "TOP500",
            "delay": 0,
            "research_context": {
                "datafields": {
                    "field_ids": ["vector_signal"],
                    "field_types": {"vector_signal": "VECTOR"},
                    "fields": [{"id": "vector_signal", "type": "VECTOR", "category": "Alternative"}],
                },
                "syntax_constraints": {"auxiliary_only_fields": []},
            },
        }

        validated = client.validate_candidate_specs([invalid, valid], 2, context)

        self.assertEqual([candidate.expression for candidate in validated], [valid.expression])
        self.assertEqual(len(client.last_validator_rejections), 1)
        self.assertIn("LOCAL_PREFLIGHT", client.last_validator_rejections[0]["reason"])
        self.assertIn("INVALID_VECTOR_TS_OPERATOR:ts_mean:vector_signal", client.last_validator_rejections[0]["reason"])

    def test_multi_model_validator_rejects_raw_high_turnover_alternative_fields(self):
        validator = FakeValidator()
        client = MultiModelAIClient(validators=[validator])
        settings = {"region": "USA", "universe": "TOP500", "delay": 0}
        invalid = CandidateSpec(
            "normalize(divide(snt23_5dts_gen_305,add(1,abs(snt23_5pos_max_297))))",
            settings=settings,
            source="model:G-2",
        )
        valid = CandidateSpec(
            "rank(ts_mean(snt23_5dts_gen_305, 20))",
            settings=settings,
            source="model:G-1",
        )
        context = {
            "region": "USA",
            "universe": "TOP500",
            "delay": 0,
            "research_context": {
                "experiment_plan": {"mode": "explore_new_family"},
                "datafields": {
                    "field_ids": ["snt23_5dts_gen_305", "snt23_5pos_max_297"],
                    "field_types": {"snt23_5dts_gen_305": "MATRIX", "snt23_5pos_max_297": "MATRIX"},
                    "fields": [
                        {"id": "snt23_5dts_gen_305", "type": "MATRIX", "category": "Sentiment", "dataset_id": "sentiment23"},
                        {"id": "snt23_5pos_max_297", "type": "MATRIX", "category": "Sentiment", "dataset_id": "sentiment23"},
                    ],
                },
                "syntax_constraints": {"auxiliary_only_fields": []},
            },
        }

        validated = client.validate_candidate_specs([invalid, valid], 2, context)

        self.assertEqual([candidate.expression for candidate in validated], [valid.expression])
        self.assertEqual(len(client.last_validator_rejections), 1)
        self.assertIn("HIGH_TURNOVER_RAW_FIELD:snt23_5dts_gen_305", client.last_validator_rejections[0]["reason"])

    def test_multi_model_validator_rejects_profile_family_mismatch_locally(self):
        client = MultiModelAIClient(validators=[])
        settings = {"region": "USA", "universe": "TOP500", "delay": 0}
        invalid = CandidateSpec(
            "rank(add(normalize(vec_avg(conference_analyst_attendees)),multiply(-1,normalize(vec_avg(anl83_numwordoperqa)))))",
            settings=settings,
            source="model:G-1",
            metadata={
                "model_profile": "G-1",
                "profile_guidance": {
                    "field_family": "ANALYST primary only, using fresh field_scout names/buckets",
                    "avoid": ["non-analyst primaries", "news fields"],
                },
            },
        )
        valid = CandidateSpec(
            "rank(ts_mean(anl83_numwordoperqa,20))",
            settings=settings,
            source="model:G-1",
            metadata={
                "model_profile": "G-1",
                "profile_guidance": {
                    "field_family": "ANALYST primary only, using fresh field_scout names/buckets",
                    "avoid": ["non-analyst primaries", "news fields"],
                },
            },
        )
        context = {
            "region": "USA",
            "universe": "TOP500",
            "delay": 0,
            "research_context": {
                "datafields": {
                    "field_ids": ["conference_analyst_attendees", "anl83_numwordoperqa"],
                    "field_types": {
                        "conference_analyst_attendees": "VECTOR",
                        "anl83_numwordoperqa": "MATRIX",
                    },
                    "fields": [
                        {
                            "id": "conference_analyst_attendees",
                            "type": "VECTOR",
                            "category": "Other",
                            "dataset_id": "other384",
                        },
                        {
                            "id": "anl83_numwordoperqa",
                            "type": "MATRIX",
                            "category": "Analyst",
                            "dataset_id": "analyst83",
                        },
                    ],
                },
                "syntax_constraints": {"auxiliary_only_fields": []},
            },
        }

        validated = client.validate_candidate_specs([invalid, valid], 2, context)

        self.assertEqual([candidate.expression for candidate in validated], [valid.expression])
        self.assertEqual(len(client.last_validator_rejections), 1)
        self.assertIn("PROFILE_REQUIRED_FIELD_FAMILY:ANALYST", client.last_validator_rejections[0]["reason"])

    def test_multi_model_generator_context_includes_profile_family_field_ids(self):
        generator = FakeGenerator("G-1", "gpt-5.3-codex-A")
        client = MultiModelAIClient(generators=[generator])
        client.last_plan = {
            "profile_guidance": {
                "G-1": {
                    "field_family": "ANALYST primary only, using fresh field_scout names/buckets",
                    "avoid": ["non-analyst primaries"],
                }
            }
        }
        context = {
            "region": "USA",
            "universe": "TOP500",
            "delay": 0,
            "research_context": {
                "datafields": {
                    "field_ids": ["conference_analyst_attendees", "anl83_numwordoperqa"],
                    "field_types": {
                        "conference_analyst_attendees": "VECTOR",
                        "anl83_numwordoperqa": "MATRIX",
                    },
                    "fields": [
                        {
                            "id": "conference_analyst_attendees",
                            "type": "VECTOR",
                            "category": "Other",
                            "dataset_id": "other384",
                        },
                        {
                            "id": "anl83_numwordoperqa",
                            "type": "MATRIX",
                            "category": "Analyst",
                            "dataset_id": "analyst83",
                        },
                    ],
                }
            },
        }

        generator_context = client._context_for_generator(generator, context)
        research_context = generator_context["research_context"]

        self.assertEqual(research_context["profile_family_field_ids"], ["anl83_numwordoperqa"])
        self.assertEqual(research_context["profile_family_policy"]["required_family"], "ANALYST")

    def test_multi_model_validator_prioritizes_candidates_by_quality_budget_slots(self):
        client = MultiModelAIClient(validators=[])
        settings = {"region": "USA", "universe": "TOP500", "delay": 0}
        broad = CandidateSpec("rank(ts_mean(broad_signal,20))", settings=settings, source="model:G-1")
        exploit = CandidateSpec("rank(ts_mean(anl10_recovery_signal,20))", settings=settings, source="model:G-1")
        probe = CandidateSpec("rank(ts_mean(mdl262_fresh_signal,20))", settings=settings, source="model:G-1")
        broad_two = CandidateSpec("rank(ts_mean(other_broad_signal,20))", settings=settings, source="model:G-1")
        context = {
            "region": "USA",
            "universe": "TOP500",
            "delay": 0,
            "research_context": {
                "experiment_plan": {
                    "quality_budget": {
                        "slots": {"exploit_positive_evidence": 1, "probe_new_fields": 1, "broad_explore": 1},
                        "exploit_fields": ["anl10_recovery_signal"],
                    },
                    "probe_recommendations": [{"field": "mdl262_fresh_signal"}],
                },
                "datafields": {
                    "field_ids": [
                        "broad_signal",
                        "anl10_recovery_signal",
                        "mdl262_fresh_signal",
                        "other_broad_signal",
                    ],
                    "field_types": {
                        "broad_signal": "MATRIX",
                        "anl10_recovery_signal": "MATRIX",
                        "mdl262_fresh_signal": "MATRIX",
                        "other_broad_signal": "MATRIX",
                    },
                    "fields": [
                        {"id": "broad_signal", "type": "MATRIX", "category": "Model"},
                        {"id": "anl10_recovery_signal", "type": "MATRIX", "category": "Analyst"},
                        {"id": "mdl262_fresh_signal", "type": "MATRIX", "category": "Model"},
                        {"id": "other_broad_signal", "type": "MATRIX", "category": "Model"},
                    ],
                },
                "syntax_constraints": {"auxiliary_only_fields": []},
            },
        }

        validated = client.validate_candidate_specs([broad, exploit, probe, broad_two], 3, context)

        self.assertEqual(
            [candidate.expression for candidate in validated],
            [exploit.expression, probe.expression, broad.expression],
        )

    def test_multi_model_validator_does_not_backfill_broad_candidates_during_production_rescue(self):
        client = MultiModelAIClient(validators=[])
        settings = {"region": "USA", "universe": "TOP500", "delay": 0}
        broad = CandidateSpec("rank(ts_mean(anl4_weak_signal,20))", settings=settings, source="model:G-1")
        probe = CandidateSpec("group_rank(ts_rank(mdl262_predictive_signal,63),industry)", settings=settings, source="model:G-1")
        context = {
            "region": "USA",
            "universe": "TOP500",
            "delay": 0,
            "research_context": {
                "experiment_plan": {
                    "production_rescue": {"active": True},
                    "quality_budget": {
                        "slots": {"probe_new_fields": 8},
                        "exploit_fields": [],
                    },
                    "probe_recommendations": [{"field": "mdl262_predictive_signal"}],
                },
                "datafields": {
                    "field_ids": ["anl4_weak_signal", "mdl262_predictive_signal"],
                    "field_types": {
                        "anl4_weak_signal": "MATRIX",
                        "mdl262_predictive_signal": "MATRIX",
                    },
                    "fields": [
                        {"id": "anl4_weak_signal", "type": "MATRIX", "category": "Analyst"},
                        {"id": "mdl262_predictive_signal", "type": "MATRIX", "category": "Model"},
                    ],
                },
                "syntax_constraints": {"auxiliary_only_fields": []},
            },
        }

        validated = client.validate_candidate_specs([broad, probe], 8, context)

        self.assertEqual([candidate.expression for candidate in validated], [probe.expression])


if __name__ == "__main__":
    unittest.main()
