from __future__ import annotations

import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Protocol

from .expression_similarity import expression_signature_metadata, expression_structure_key, expression_variant_key
from .models import CandidateSpec, DEFAULT_SETTINGS, SimulationFailure, SimulationResult, SubmitResult


class AIClient(Protocol):
    def generate_candidates(self, batch_size: int, context: Dict[str, Any]) -> List[CandidateSpec]:
        ...


class AIClientError(RuntimeError):
    pass


@dataclass(frozen=True)
class ModelProfile:
    name: str
    role: str
    model: str
    api_key: str
    base_url: str
    weight: int = 1
    request_timeout: float = 60.0


DEFAULT_MODEL_PROFILE_SPEC = (
    "grok-4-1-fast-reasoning@controller,"
    "gemini-3-flash-free@generator,"
    "glm-4.5@optimizer,"
    "gpt-5.4-nano@validator"
)


def parse_model_profiles_from_env() -> List[ModelProfile]:
    raw, profile_base_dir = _model_profiles_source()
    default_api_key = os.environ.get("AI_API_KEY") or os.environ.get("OPENAI_API_KEY") or ""
    default_base_url = os.environ.get("AI_BASE_URL", "https://api.openai.com/v1")
    if raw.startswith("["):
        try:
            items = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"AI_MODEL_PROFILES invalid JSON: {exc}") from exc
        if not isinstance(items, list):
            raise RuntimeError("AI_MODEL_PROFILES JSON must be a list")
        profiles = [
            _profile_from_mapping(
                item,
                default_api_key=default_api_key,
                default_base_url=default_base_url,
                base_dir=profile_base_dir,
            )
            for item in items
        ]
    else:
        profiles = [
            _profile_from_compact(part, default_api_key=default_api_key, default_base_url=default_base_url)
            for part in raw.split(",")
            if part.strip()
        ]
    if not profiles:
        raise RuntimeError("AI_MODEL_PROFILES did not define any model profiles")
    missing = [profile.name for profile in profiles if not profile.api_key]
    if missing:
        raise RuntimeError(
            "AI_API_KEY or OPENAI_API_KEY is required for AI_CLIENT=multi "
            f"(missing profiles: {', '.join(missing)})"
        )
    return profiles


def _model_profiles_source() -> tuple[str, Path | None]:
    profiles_file = os.environ.get("AI_MODEL_PROFILES_FILE", "").strip()
    if not profiles_file:
        return os.environ.get("AI_MODEL_PROFILES", DEFAULT_MODEL_PROFILE_SPEC).strip(), None
    path = _resolve_profile_path(profiles_file, None)
    if not path.exists():
        raise RuntimeError(f"AI_MODEL_PROFILES_FILE not found: {path}")
    return _read_model_profiles_text(path), path.parent


def _read_model_profiles_text(path: Path) -> str:
    data = path.read_bytes()
    last_error: UnicodeDecodeError | None = None
    for encoding in ("utf-8-sig", "utf-8", "gbk"):
        try:
            return data.decode(encoding).strip()
        except UnicodeDecodeError as exc:
            last_error = exc
    raise RuntimeError(f"AI_MODEL_PROFILES_FILE could not be decoded as UTF-8 or GBK: {path}") from last_error


def _profile_from_mapping(
    item: Any,
    default_api_key: str,
    default_base_url: str,
    base_dir: Path | None = None,
) -> ModelProfile:
    if not isinstance(item, dict):
        raise RuntimeError("AI_MODEL_PROFILES JSON entries must be objects")
    model = str(item.get("model") or "").strip()
    role = _normalize_role(item.get("role") or "generator")
    if not model:
        raise RuntimeError("AI_MODEL_PROFILES entries require model")
    profile_env = _read_profile_env_file(str(item.get("env_file") or ""), base_dir)
    api_key = str(item.get("api_key") or "")
    api_key_env = str(item.get("api_key_env") or "")
    if not api_key and api_key_env:
        api_key = profile_env.get(api_key_env) or os.environ.get(api_key_env, "")
    if not api_key:
        api_key = profile_env.get("AI_API_KEY") or profile_env.get("OPENAI_API_KEY") or ""
    if not api_key:
        api_key = default_api_key
    base_url = str(item.get("base_url") or "")
    base_url_env = str(item.get("base_url_env") or "")
    if not base_url and base_url_env:
        base_url = profile_env.get(base_url_env) or os.environ.get(base_url_env, "")
    if not base_url:
        base_url = profile_env.get("AI_BASE_URL") or ""
    if not base_url:
        base_url = default_base_url
    name = str(item.get("name") or "").strip() or _default_profile_name(model, role)
    request_timeout = _profile_request_timeout(item, profile_env)
    return ModelProfile(
        name=name,
        role=role,
        model=model,
        api_key=api_key,
        base_url=base_url,
        weight=max(1, int(item.get("weight") or 1)),
        request_timeout=request_timeout,
    )


def _read_profile_env_file(env_file: str, base_dir: Path | None) -> Dict[str, str]:
    if not env_file:
        return {}
    path = _resolve_profile_path(env_file, base_dir)
    if not path.exists():
        raise RuntimeError(f"AI model profile env_file not found: {path}")
    values: Dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key:
            continue
        values[key] = _unquote_env_value(value.strip())
    return values


def _resolve_profile_path(path_text: str, base_dir: Path | None) -> Path:
    path = Path(path_text).expanduser()
    if path.is_absolute():
        return path
    cwd_path = Path.cwd() / path
    if cwd_path.exists() or base_dir is None:
        return cwd_path
    base_path = base_dir / path
    if base_path.exists():
        return base_path
    return cwd_path


def _unquote_env_value(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def _profile_request_timeout(item: Dict[str, Any], profile_env: Dict[str, str]) -> float:
    raw = (
        item.get("request_timeout")
        or item.get("timeout")
        or profile_env.get("AI_REQUEST_TIMEOUT_SECONDS")
        or os.environ.get("AI_REQUEST_TIMEOUT_SECONDS")
        or 60.0
    )
    try:
        return max(1.0, float(raw))
    except (TypeError, ValueError):
        return 60.0


def _profile_from_compact(part: str, default_api_key: str, default_base_url: str) -> ModelProfile:
    text = part.strip()
    if "@" in text:
        model, role = text.rsplit("@", 1)
    elif ":" in text:
        role, model = text.split(":", 1)
    else:
        model, role = text, "generator"
    model = model.strip()
    role = _normalize_role(role)
    if not model:
        raise RuntimeError(f"invalid AI_MODEL_PROFILES entry: {part}")
    return ModelProfile(
        name=_default_profile_name(model, role),
        role=role,
        model=model,
        api_key=default_api_key,
        base_url=default_base_url,
        request_timeout=max(1.0, float(os.environ.get("AI_REQUEST_TIMEOUT_SECONDS", "60"))),
    )


def _normalize_role(value: Any) -> str:
    role = str(value or "generator").strip().lower()
    aliases = {
        "flow": "controller",
        "planner": "controller",
        "reviewer": "critic",
        "explorer": "generator",
        "generate": "generator",
        "optimizer": "optimizer",
        "optimiser": "optimizer",
        "check": "validator",
        "checker": "validator",
        "validate": "validator",
        "format": "validator",
    }
    role = aliases.get(role, role)
    if role not in {"controller", "critic", "generator", "optimizer", "validator"}:
        raise RuntimeError(f"unsupported model role: {role}")
    return role


def _default_profile_name(model: str, role: str) -> str:
    lowered = model.lower()
    if role == "critic":
        return "critic"
    if role == "controller" and "grok" in lowered:
        return "flow"
    if "gemini" in lowered:
        return "gemini"
    if "glm" in lowered:
        return "glm"
    if "qwen" in lowered:
        return "qwen"
    if "minimax" in lowered:
        return "minimax"
    if "kimi" in lowered:
        return "kimi"
    if "nano" in lowered:
        return "cheap-check"
    compact = re.sub(r"[^a-z0-9]+", "-", lowered).strip("-")
    return compact or role


class BrainClient(Protocol):
    def simulate(self, expression: str, settings: Dict[str, Any]) -> SimulationResult:
        ...

    def simulate_many(self, items: List[tuple[str, Dict[str, Any]]]) -> List[SimulationResult | SimulationFailure]:
        ...

    def submit_alpha(self, alpha_id: str, dry_run: bool = True) -> SubmitResult:
        ...

    def discover_datafields(
        self,
        settings: Dict[str, Any],
        search_terms: List[str] | None = None,
        max_fields: int = 120,
    ) -> List[Dict[str, Any]]:
        ...

    def get_pyramid_alphas(self, start_date: str | None = None, end_date: str | None = None) -> Dict[str, Any]:
        ...

    def get_pyramid_multipliers(self) -> Dict[str, Any]:
        ...


class LocalAIClient:
    def __init__(self, expressions: List[str] | None = None, metadata: Dict[str, Any] | None = None):
        self.expressions = expressions or [
            "rank(mdl_mock_score)",
            "group_rank(ts_rank(mdl_mock_score, 22), industry)",
            "group_rank(ts_rank(divide(mdl_mock_score, cap), 63), industry)",
            "rank(ts_mean(mdl_mock_score, 22))",
        ]
        self.metadata = metadata or {}

    def generate_candidates(self, batch_size: int, context: Dict[str, Any]) -> List[CandidateSpec]:
        settings = dict(DEFAULT_SETTINGS)
        settings.update({key: value for key, value in dict(context or {}).items() if key != "research_context"})
        candidates: List[CandidateSpec] = []
        for idx in range(batch_size):
            expression = self.expressions[idx % len(self.expressions)]
            candidates.append(
                CandidateSpec(
                    expression=expression,
                    settings=dict(settings),
                    source="local_ai",
                    metadata=dict(self.metadata),
                )
            )
        return candidates


class LocalBrainClient:
    def __init__(self, force_pending_checks: bool = False, always_fail_simulation: bool = False):
        self.force_pending_checks = force_pending_checks
        self.always_fail_simulation = always_fail_simulation
        self.submitted: List[str] = []

    def simulate(self, expression: str, settings: Dict[str, Any]) -> SimulationResult:
        if self.always_fail_simulation:
            raise RuntimeError("local simulation failure")

        digest = hashlib.sha1((expression + repr(sorted(settings.items()))).encode("utf-8")).hexdigest()[:8]
        checks = self._checks()
        metrics = {
            "sharpe": 2.0,
            "fitness": 1.1,
            "turnover": 0.2,
            "returns": 0.08,
            "drawdown": 0.05,
        }
        return SimulationResult(alpha_id=f"LOCAL{digest}", metrics=metrics, checks=checks, raw={"settings": settings})

    def simulate_many(self, items: List[tuple[str, Dict[str, Any]]]) -> List[SimulationResult | SimulationFailure]:
        return [self.simulate(expression, settings) for expression, settings in items]

    def submit_alpha(self, alpha_id: str, dry_run: bool = True) -> SubmitResult:
        if dry_run:
            return SubmitResult(alpha_id=alpha_id, submitted=False, stage="DRY_RUN", message="auto_submit disabled")
        self.submitted.append(alpha_id)
        return SubmitResult(alpha_id=alpha_id, submitted=True, stage="OS", message="local submit accepted")

    def count_submitted_alphas(self, start_date: str, end_date: str) -> int:
        return len(self.submitted)

    def recent_submitted_alphas(self, settings: Dict[str, Any] | None = None, limit: int = 50) -> List[Dict[str, Any]]:
        return []

    def get_pyramid_alphas(self, start_date: str | None = None, end_date: str | None = None) -> Dict[str, Any]:
        return {"pyramids": []}

    def get_pyramid_multipliers(self) -> Dict[str, Any]:
        return {"pyramids": []}

    def discover_datafields(
        self,
        settings: Dict[str, Any],
        search_terms: List[str] | None = None,
        max_fields: int = 120,
    ) -> List[Dict[str, Any]]:
        rows = [
            _datafield_row("close", "Price", "pv1", "Close price", "MATRIX"),
            _datafield_row("returns", "Price", "pv1", "Daily returns", "MATRIX"),
            _datafield_row("volume", "Volume", "pv1", "Daily traded volume", "MATRIX"),
            _datafield_row("cap", "Size", "pv1", "Market capitalization", "MATRIX"),
            _datafield_row("mdl_mock_score", "Model", "model_mock", "Mock model score for offline tests", "MATRIX"),
            _datafield_row("mdl_alt_score", "Model", "model_mock", "Alternate mock model score for offline tests", "MATRIX"),
            _datafield_row("mdl_other_score", "Model", "model_mock", "Other mock model score for offline tests", "MATRIX"),
        ]
        return rows[:max_fields]

    def _checks(self) -> Dict[str, Dict[str, Any]]:
        pending_status = "PENDING" if self.force_pending_checks else "PASS"
        return {
            "LOW_SHARPE": {"status": "PASS", "value": 2.0},
            "LOW_FITNESS": {"status": "PASS", "value": 1.1},
            "LOW_TURNOVER": {"status": "PASS", "value": 0.2},
            "HIGH_TURNOVER": {"status": "PASS", "value": 0.2},
            "CONCENTRATED_WEIGHT": {"status": "PASS"},
            "LOW_SUB_UNIVERSE_SHARPE": {"status": "PASS", "value": 1.2},
            "IS_LADDER_SHARPE": {"status": "PASS", "value": 2.5},
            "SELF_CORRELATION": {"status": "PASS", "value": 0.2},
            "PROD_CORRELATION": {"status": pending_status, "value": 0.3 if not self.force_pending_checks else None},
            "DATA_DIVERSITY": {"status": pending_status},
            "REGULAR_SUBMISSION": {"status": pending_status},
        }


class OpenAICompatibleAIClient:
    """AI generator for OpenAI-compatible chat-completions APIs.

    The transport hook keeps tests offline and also allows non-OpenAI compatible
    gateways to be wrapped without changing the worker.
    """

    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str = "https://api.openai.com/v1",
        transport: Callable[[Dict[str, Any]], Dict[str, Any]] | None = None,
        source: str = "openai_compatible",
        profile_name: str = "default",
        role: str = "generator",
        request_timeout: float = 60.0,
    ):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.transport = transport
        self.source = source
        self.profile_name = profile_name
        self.role = role
        self.request_timeout = max(1.0, float(request_timeout))
        self.last_rejections: List[Dict[str, Any]] = []

    @property
    def chat_completions_url(self) -> str:
        if self.base_url.endswith("/chat/completions"):
            return self.base_url
        return f"{self.base_url}/chat/completions"

    @classmethod
    def from_env(cls) -> "OpenAICompatibleAIClient":
        api_key = os.environ.get("AI_API_KEY") or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("AI_API_KEY or OPENAI_API_KEY is required for AI_CLIENT=openai")
        return cls(
            api_key=api_key,
            model=os.environ.get("AI_MODEL", "gpt-4.1-mini"),
            base_url=os.environ.get("AI_BASE_URL", "https://api.openai.com/v1"),
            request_timeout=float(os.environ.get("AI_REQUEST_TIMEOUT_SECONDS", "60")),
        )

    @classmethod
    def from_profile(cls, profile: ModelProfile) -> "OpenAICompatibleAIClient":
        return cls(
            api_key=profile.api_key,
            model=profile.model,
            base_url=profile.base_url,
            source=f"model:{profile.name}",
            profile_name=profile.name,
            role=profile.role,
            request_timeout=profile.request_timeout,
        )

    def plan_generation(
        self,
        batch_size: int,
        context: Dict[str, Any],
        generator_profiles: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        system = (
            "You control a WorldQuant BRAIN alpha exploration loop. "
            "Read the research_context, current scope, and generator model profiles. "
            "Return strict JSON only: {\"mode\":\"explore|optimize|repair\","
            "\"profile_guidance\":{\"profile_name\":{\"objective\":\"...\",\"field_family\":\"...\","
            "\"mechanism\":\"...\",\"structure\":\"...\",\"avoid\":[]}},\"rationale\":\"...\"}. "
            "Do not spend rationale on how many candidates each model should produce. "
            "The service enforces execution allocation separately, including fixed balanced 4+4 cooperation "
            "between the active generation profiles during exploration. "
            "Your job is to assign distinct research directions to each active profile: field family, signal "
            "mechanism, formula structure, and exhausted anchors or failure modes to avoid. "
            "If research_context.analysis.family_diversity_control is present, keep the dominant family anchored "
            "to at most one profile and route the other profile(s) toward the listed alternate families. "
            "If research_context.submitted_field_avoidance or research_context.analysis.submitted_avoid_fields is "
            "present, treat those approved/submitted core fields as exhausted and do not assign them to any profile. "
            "If research_context.lit_tower_avoidance or experiment_plan.lit_tower_avoidance is present, prefer "
            "unlit pyramid towers for fresh exploration and keep already-lit towers out of new profile routes unless "
            "the plan is explicitly optimizing a near-threshold candidate. "
            "If research_context.experiment_plan.mechanism_transfer is present, use its archetypes as mechanism-only "
            "evidence. Do not copy their expressions or forbidden_fields; assign routes that migrate those mechanisms "
            "to fresh non-submitted primary fields. "
            "If research_context.analysis.route_efficiency.stop_loss_active is true, stop improving the current "
            "route locally and assign different mechanism classes. If structure_diversity_control is present, "
            "avoid overused formula skeletons and require distinct operator geometry between profiles. "
            "If an optimizer profile is active in explore mode, treat it as the second cooperating generator, "
            "not as an allocation decision-maker."
        )
        user = json.dumps(
            {
                "batch_size": batch_size,
                "target_settings": {key: value for key, value in dict(context or {}).items() if key != "research_context"},
                "research_context": dict(context or {}).get("research_context", {}),
                "generator_profiles": generator_profiles,
            },
            sort_keys=True,
        )
        parsed = self._request_json(system, user, temperature=0.2)
        return parsed if isinstance(parsed, dict) else {}

    def critique_generation_plan(
        self,
        batch_size: int,
        context: Dict[str, Any],
        generator_profiles: List[Dict[str, Any]],
        initial_plan: Dict[str, Any],
    ) -> Dict[str, Any]:
        system = (
            "You are flow-critic for a WorldQuant BRAIN alpha exploration loop. "
            "Review the initial controller plan against the research history, active scope, official checks, "
            "and generator profiles. Do not allocate candidate counts and do not replace the plan outright. "
            "Find concrete risks: repeated families, weak anchors, invalid fields/operators, ignored regional "
            "thresholds, over-optimization, correlated G-1/G-2 directions, and missing setting-sweep opportunities. "
            "Flag any plan that reuses fields listed in submitted_field_avoidance or submitted_avoid_fields, because "
            "passing variants are likely to fail self/prod correlation. "
            "Flag fresh-exploration plans that ignore lit_tower_avoidance when platform pyramid evidence is present. "
            "When mechanism_transfer is present, flag plans that copy blocked expressions or forbidden_fields instead "
            "of transferring the mechanism to new primary fields. "
            "When route_stop_loss or structure_diversity_control is present, flag plans that only swap fields inside "
            "the same formula skeleton. "
            "Return strict JSON only: {\"concerns\":[],\"recommendations\":[],"
            "\"profile_guidance_delta\":{\"profile_name\":{\"objective\":\"...\",\"field_family\":\"...\","
            "\"mechanism\":\"...\",\"structure\":\"...\",\"avoid\":[]}},\"rationale\":\"...\"}."
        )
        user = json.dumps(
            {
                "batch_size": batch_size,
                "target_settings": {key: value for key, value in dict(context or {}).items() if key != "research_context"},
                "research_context": dict(context or {}).get("research_context", {}),
                "generator_profiles": generator_profiles,
                "initial_plan": initial_plan,
            },
            sort_keys=True,
        )
        parsed = self._request_json(system, user, temperature=0.1)
        return parsed if isinstance(parsed, dict) else {}

    def finalize_generation_plan(
        self,
        batch_size: int,
        context: Dict[str, Any],
        generator_profiles: List[Dict[str, Any]],
        initial_plan: Dict[str, Any],
        critiques: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        system = (
            "You are flow-main finalizing a WorldQuant BRAIN alpha exploration plan after critic review. "
            "Use the initial plan and critic feedback to produce the final executable plan. "
            "Return strict JSON only with the same schema as plan_generation: "
            "{\"mode\":\"explore|optimize|repair\",\"profile_guidance\":{\"profile_name\":{\"objective\":\"...\","
            "\"field_family\":\"...\",\"mechanism\":\"...\",\"structure\":\"...\",\"avoid\":[]}},"
            "\"rationale\":\"...\"}. Do not allocate candidate counts. The service enforces fixed balanced "
            "4+4 cooperation between active generation profiles when required. "
            "If research_context.analysis.family_diversity_control exists, preserve the split so only one profile "
            "anchors the dominant family and the others move to alternate families. Preserve submitted field "
            "avoidance as a hard constraint across every profile. Preserve lit_tower_avoidance as a soft diversity "
            "constraint: choose unlit pyramid towers for fresh exploration whenever possible. If mechanism_transfer "
            "is active, keep it mechanism only: Do not copy blocked fields or expressions, and route profiles toward "
            "fresh primary fields that preserve the historical signal logic. Preserve route_stop_loss and "
            "structure_diversity_control by forcing different mechanism classes and formula skeletons."
        )
        user = json.dumps(
            {
                "batch_size": batch_size,
                "target_settings": {key: value for key, value in dict(context or {}).items() if key != "research_context"},
                "research_context": dict(context or {}).get("research_context", {}),
                "generator_profiles": generator_profiles,
                "initial_plan": initial_plan,
                "critic_feedback": critiques,
            },
            sort_keys=True,
        )
        parsed = self._request_json(system, user, temperature=0.15)
        return parsed if isinstance(parsed, dict) else {}

    def critique_candidate_batch(
        self,
        batch_size: int,
        context: Dict[str, Any],
        generator_profiles: List[Dict[str, Any]],
        final_plan: Dict[str, Any],
        accepted_candidates: List[Dict[str, Any]],
        validator_rejections: List[Dict[str, Any]],
        model_errors: List[Dict[str, Any]],
        remaining_slots: int,
    ) -> Dict[str, Any]:
        system = (
            "You are flow-critic reviewing a same-round WorldQuant BRAIN alpha candidate batch before simulation. "
            "Do not make the final decision. Identify concrete repair needs: validator removals, duplicate structures, "
            "weak or off-plan mechanisms, repeated exhausted fields, invalid syntax risk, and missing profile coverage. "
            "Also flag fresh candidates that ignore lit_tower_avoidance when the final plan chose unlit pyramid routes. "
            "When final_plan.mechanism_transfer is present, reject candidates that copy mechanism_transfer "
            "forbidden_fields as primary fields instead of transferring the mechanism. "
            "When final_plan.structure_diversity_control is present, reject candidates that refill the batch with "
            "more copies of an already accepted formula skeleton. "
            "Return strict JSON only: {\"concerns\":[],\"recommendations\":[],"
            "\"profile_guidance_delta\":{\"profile_name\":{\"objective\":\"...\",\"field_family\":\"...\","
            "\"mechanism\":\"...\",\"structure\":\"...\",\"avoid\":[]}},\"rationale\":\"...\"}."
        )
        user = json.dumps(
            {
                "batch_size": batch_size,
                "target_settings": {key: value for key, value in dict(context or {}).items() if key != "research_context"},
                "research_context": dict(context or {}).get("research_context", {}),
                "generator_profiles": generator_profiles,
                "final_plan": final_plan,
                "accepted_candidates": accepted_candidates,
                "validator_rejections": validator_rejections,
                "model_errors": model_errors,
                "remaining_slots": remaining_slots,
            },
            sort_keys=True,
        )
        parsed = self._request_json(system, user, temperature=0.1)
        return parsed if isinstance(parsed, dict) else {}

    def repair_candidate_batch(
        self,
        batch_size: int,
        context: Dict[str, Any],
        generator_profiles: List[Dict[str, Any]],
        final_plan: Dict[str, Any],
        accepted_candidates: List[Dict[str, Any]],
        validator_rejections: List[Dict[str, Any]],
        model_errors: List[Dict[str, Any]],
        critic_feedback: List[Dict[str, Any]],
        remaining_slots: int,
    ) -> Dict[str, Any]:
        system = (
            "You are flow-main making the final same-round repair decision for a WorldQuant BRAIN alpha batch. "
            "Use the accepted candidates, validator rejections, model errors, and critic feedback. "
            "You are the decision maker: choose accept, refill, repair, or abandon. "
            "Only choose refill/repair when the batch has useful accepted candidates but needs replacement candidates "
            "before simulation. Do not allocate counts unless refill/repair is needed. "
            "When replacement candidates are needed, preserve submitted-field constraints and lit_tower_avoidance. "
            "If mechanism_transfer is present, use it only for mechanism transfer and never for copying its "
            "forbidden_fields or exact expressions. "
            "If structure_diversity_control is present, replacement candidates must use a different formula skeleton "
            "from the accepted batch where possible. "
            "Return strict JSON only: {\"action\":\"accept|refill|repair|abandon\","
            "\"allocation\":{\"profile_name\":1},\"profile_guidance\":{\"profile_name\":{\"objective\":\"...\","
            "\"field_family\":\"...\",\"mechanism\":\"...\",\"structure\":\"...\",\"avoid\":[]}},"
            "\"rationale\":\"...\"}."
        )
        user = json.dumps(
            {
                "batch_size": batch_size,
                "target_settings": {key: value for key, value in dict(context or {}).items() if key != "research_context"},
                "research_context": dict(context or {}).get("research_context", {}),
                "generator_profiles": generator_profiles,
                "final_plan": final_plan,
                "accepted_candidates": accepted_candidates,
                "validator_rejections": validator_rejections,
                "model_errors": model_errors,
                "critic_feedback": critic_feedback,
                "remaining_slots": remaining_slots,
            },
            sort_keys=True,
        )
        parsed = self._request_json(system, user, temperature=0.15)
        return parsed if isinstance(parsed, dict) else {}

    def validate_candidate_specs(
        self,
        candidates: List[CandidateSpec],
        batch_size: int,
        context: Dict[str, Any],
    ) -> List[CandidateSpec]:
        context = dict(context or {})
        target_context = {key: value for key, value in context.items() if key != "research_context"}
        research_context = context.get("research_context") if isinstance(context.get("research_context"), dict) else {}
        source_by_index = [candidate.source for candidate in candidates]
        metadata_by_index = [dict(candidate.metadata) for candidate in candidates]
        self.last_rejections = []
        system = (
            "You are a validation and formatting gate for WorldQuant BRAIN FASTEXPR candidates. "
            "Do not create unrelated new ideas. Remove invalid, duplicate, invented-field, or clearly trivial entries. "
            "You may make minimal syntax/format fixes only when the intended expression is obvious. "
            "Use only field ids and syntax constraints in research_context. "
            "Preserve candidate settings. If the same expression appears with different settings, treat those as "
            "distinct setting-sweep variants rather than expression duplicates. "
            "Return strict JSON only: {\"candidates\":[{\"expression\":\"...\",\"source\":\"...\","
            "\"settings\":{},\"metadata\":{}}],\"rejected\":[{\"expression\":\"...\",\"settings\":{},\"reason\":\"...\"}]}."
        )
        user = json.dumps(
            {
                "batch_size": batch_size,
                "target_settings": target_context,
                "research_context": research_context,
                "candidates": [
                    {
                        "expression": candidate.expression,
                        "settings": candidate.settings,
                        "source": candidate.source,
                        "metadata": candidate.metadata,
                    }
                    for candidate in candidates
                ],
            },
            sort_keys=True,
        )
        try:
            parsed = self._request_json(system, user, temperature=0.0)
        except Exception:
            self.last_rejections = []
            return [
                CandidateSpec(
                    expression=candidate.expression,
                    settings=candidate.settings,
                    source=candidate.source,
                    metadata={
                        **candidate.metadata,
                        "validator_error": "validation_fallback",
                        "validator_model": self.model,
                    },
                )
                for candidate in candidates[:batch_size]
            ]

        specs = parsed.get("candidates", parsed if isinstance(parsed, list) else [])
        rejected = parsed.get("rejected") if isinstance(parsed, dict) else []
        self.last_rejections = rejected if isinstance(rejected, list) else []
        if not isinstance(specs, list):
            return candidates[:batch_size]
        settings_base = dict(DEFAULT_SETTINGS)
        settings_base.update(target_context)
        validated: List[CandidateSpec] = []
        seen = set()
        for idx, item in enumerate(specs[:batch_size]):
            if not isinstance(item, dict) or not item.get("expression"):
                continue
            expression = str(item["expression"]).strip()
            settings = item.get("settings") if isinstance(item.get("settings"), dict) else None
            if settings is None and idx < len(candidates):
                settings = dict(candidates[idx].settings)
            if settings is None:
                settings = dict(settings_base)
            key = _candidate_spec_key(expression, settings)
            if key in seen:
                continue
            seen.add(key)
            metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
            if not metadata and idx < len(metadata_by_index):
                metadata = dict(metadata_by_index[idx])
            metadata = {
                **metadata,
                "validated_by": self.profile_name,
                "validator_model": self.model,
            }
            source = str(item.get("source") or "")
            if not source and idx < len(source_by_index):
                source = source_by_index[idx]
            validated.append(
                CandidateSpec(
                    expression=expression,
                    settings=dict(settings),
                    source=source or "model:validated",
                    metadata=metadata,
                )
            )
        return validated or candidates[:batch_size]

    def generate_candidates(self, batch_size: int, context: Dict[str, Any]) -> List[CandidateSpec]:
        context = dict(context or {})
        research_context = context.get("research_context") if isinstance(context.get("research_context"), dict) else {}
        research_context = _compact_generator_research_context(research_context)
        target_context = {key: value for key, value in context.items() if key != "research_context"}
        generation_policy = research_context.get("generation_policy", {}) if isinstance(research_context, dict) else {}
        payload = {
            "model": self.model,
            "temperature": 0.65,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You generate WorldQuant BRAIN FASTEXPR alpha candidates for research-grade automated testing. "
                        "Prefer economically motivated multi-operator hypotheses over trivial price/volume ranks. "
                        f"Generate exactly {batch_size} distinct candidate expressions unless the model cannot do so. "
                        "Use only plausible FASTEXPR operators and keep formulas valid for BRAIN simulation. "
                        "When research_context.datafields.available is true, use only field identifiers listed in "
                        "research_context.datafields.field_ids and do not invent datafield ids. "
                        "Treat research_context.experiment_plan as mandatory: follow its mode, objective, keep, "
                        "change, and avoid lists when constructing the batch. For mode optimize_best, generate "
                        "controlled variants around the target_expression instead of unrelated ideas. "
                        "If research_context.experiment_plan.family_diversity_control is present, keep the dominant "
                        "family anchored to a single profile at most and use alternate_families for the other "
                        "active profiles. "
            "If research_context.experiment_plan.submitted_field_avoidance is present, do not use its "
            "approved/submitted core fields or close variants of those successful alphas. "
            "If research_context.experiment_plan.lit_tower_avoidance is present, avoid already-lit pyramid towers "
            "for fresh exploration and prefer the listed unlit_towers; this is a diversity constraint, not a license "
            "to infer tower status from field names. "
            "If research_context.experiment_plan.mechanism_transfer is present, use its archetypes as mechanism only "
            "examples. Do not copy their expressions. Do not copy forbidden_fields into primary alpha legs. Transfer "
            "the mechanism to fresh, allowed, non-submitted fields while keeping auxiliary fields only as helpers. "
            "If research_context.experiment_plan.route_stop_loss is active, do not keep optimizing the same route; "
            "change mechanism class and operator geometry. If structure_diversity_control is present, generate no "
            "more than its max_batch_candidates_per_structure candidates per field-agnostic formula skeleton. "
            "If research_context.profile_guidance is present, treat it as mandatory model-specific "
            "direction and generate candidates only within that assigned route. "
                        "Treat research_context.syntax_constraints as hard syntax policy: use only its "
                        "allowed_operators, avoid its recent_preflight_rejections, and copy field ids exactly. "
                        "Treat research_context.syntax_constraints.auxiliary_only_fields as helper fields only: "
                        "do not make close, vwap, open, high, low, volume, returns, cap, or adv20 the primary "
                        "alpha signal or a standalone additive/subtractive leg. Use them only as denominators, "
                        "scale/liquidity controls, risk filters, or conditions around non-auxiliary datafields. "
                        "Respect datafield types: MATRIX fields may be used directly with time-series operators; "
                        "VECTOR fields must first be reduced with a valid single-argument vec_* reducer before any "
                        "ts_ operator. Never pass a VECTOR field directly into ts_backfill, ts_mean, ts_rank, "
                        "winsorize, rank, or group_rank. vec_avg(x) only: never write vec_avg(x, window); "
                        "put windows outside vector reducers, e.g. ts_mean(vec_avg(vector_field), 30). "
                        "Use divide(x, y), never div(x, y). group_rank(x, group) has exactly two arguments; "
                        "put time windows inside ts_ operators, e.g. group_rank(ts_rank(x, 63), industry), "
                        "never group_rank(x, 63), industry. "
                        "Return strict JSON only: {\"candidates\":[{\"expression\":\"...\",\"settings\":{},"
                        "\"hypothesis\":\"...\",\"risk_notes\":\"...\"}]}. Do not include prose outside JSON."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "batch_size": batch_size,
                            "target_settings": target_context,
                            "research_context": research_context,
                            "constraints": {
                                "complexity": generation_policy.get("complexity", "research_grade"),
                                "avoid_trivial_price_volume_only": True,
                                "prefer_multi_operator_hypotheses": True,
                                "require_hypothesis_notes": True,
                                "required_candidate_count": batch_size,
                                "avoid_unverified_operators": True,
                                "avoid_unverified_fields": True,
                                "use_only_datafields_from_context": True,
                                "respect_datafield_types": True,
                                "vector_fields_require_vec_reduction": True,
                                "use_only_syntax_allowed_operators": True,
                                "vec_reducers_are_single_argument_only": True,
                                "use_divide_not_div": True,
                                "group_rank_accepts_value_and_group_only": True,
                                "vector_fields_must_be_reduced_in_expression_not_just_hypothesis": True,
                                "copy_field_ids_exactly": True,
                                "auxiliary_fields_must_not_be_primary": generation_policy.get(
                                    "auxiliary_fields_must_not_be_primary",
                                    True,
                                ),
                                "avoid_near_duplicates_from_recent_history": True,
                                "avoid_recent_approved_submitted_fields": True,
                                "respect_route_stop_loss": True,
                                "respect_structure_diversity_control": True,
                            },
                        },
                        sort_keys=True,
                    ),
                },
            ],
        }
        try:
            response = self.transport(payload) if self.transport else self._post_chat_completions(payload)
            content = self._extract_content(response)
            parsed = self._parse_json_content(content)
        except AIClientError:
            raise
        except json.JSONDecodeError as exc:
            raise AIClientError(f"AI response invalid JSON: {exc}") from exc
        except Exception as exc:
            raise AIClientError(f"AI candidate generation failed: {exc}") from exc
        specs = parsed.get("candidates", parsed if isinstance(parsed, list) else [])
        settings_base = dict(DEFAULT_SETTINGS)
        settings_base.update(target_context)

        candidates: List[CandidateSpec] = []
        seen = set()
        seen_structures = set()
        reject_trivial = bool(generation_policy.get("reject_trivial_candidates"))
        reject_structural_duplicates = bool(generation_policy.get("avoid_structural_duplicates"))
        for item in specs[:batch_size]:
            if not isinstance(item, dict) or not item.get("expression"):
                continue
            expression = str(item["expression"]).strip()
            if expression in seen:
                continue
            variant_key = expression_variant_key(expression)
            if reject_structural_duplicates and variant_key in seen_structures:
                continue
            if reject_trivial and _is_trivial_expression(expression):
                continue
            seen.add(expression)
            seen_structures.add(variant_key)
            settings = dict(settings_base)
            candidate_settings = item.get("settings") or {}
            metadata = {
                key: value
                for key, value in item.items()
                if key not in {"expression", "settings"} and value not in (None, "")
            }
            if isinstance(candidate_settings, dict) and candidate_settings:
                metadata["proposed_settings"] = candidate_settings
            candidates.append(
                CandidateSpec(
                    expression=expression,
                    settings=settings,
                    source=self.source,
                    metadata={
                        **metadata,
                        **expression_signature_metadata(expression),
                        "model_profile": self.profile_name,
                        "model_role": self.role,
                        "model": self.model,
                    },
                )
            )
        if reject_trivial and specs and not candidates:
            raise AIClientError("AI returned only trivial candidates after research-grade filtering")
        return candidates

    def _request_json(self, system_content: str, user_content: str, temperature: float) -> Any:
        payload = {
            "model": self.model,
            "temperature": temperature,
            "max_tokens": _json_request_max_tokens(),
            "messages": [
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content},
            ],
        }
        try:
            response = self.transport(payload) if self.transport else self._post_chat_completions(payload)
            content = self._extract_content(response)
            return self._parse_json_content(content)
        except AIClientError:
            raise
        except json.JSONDecodeError as exc:
            raise AIClientError(f"AI response invalid JSON: {exc}") from exc
        except Exception as exc:
            raise AIClientError(f"AI JSON request failed: {exc}") from exc

    def _post_chat_completions(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            self.chat_completions_url,
            data=body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.request_timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", "replace")[:800]
            raise RuntimeError(f"AI request failed: HTTP {exc.code}: {body}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"AI request failed: {exc}") from exc

    @staticmethod
    def _extract_content(response: Dict[str, Any]) -> str:
        if "output_text" in response:
            return OpenAICompatibleAIClient._content_to_text(response["output_text"])
        choices = response.get("choices") or []
        if choices:
            message = choices[0].get("message") or {}
            content = OpenAICompatibleAIClient._content_to_text(message.get("content"))
            if content.strip():
                return content
            for fallback_key in ("text", "reasoning_content", "reasoning"):
                content = OpenAICompatibleAIClient._content_to_text(message.get(fallback_key))
                if content.strip():
                    return content
            content = OpenAICompatibleAIClient._content_to_text(choices[0].get("text"))
            if content.strip():
                return content
            finish_reason = str(choices[0].get("finish_reason") or "")
            raise ValueError(f"AI response content empty finish_reason={finish_reason or 'unknown'}")
        raise ValueError("AI response did not contain candidate content")

    @staticmethod
    def _parse_json_content(content: str) -> Any:
        text = str(content or "").strip().lstrip("\ufeff")
        if not text:
            raise AIClientError("AI response content empty")

        candidates = [text]
        for match in re.finditer(r"```(?:json)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL):
            fenced = match.group(1).strip()
            if fenced and fenced not in candidates:
                candidates.append(fenced)
        balanced = OpenAICompatibleAIClient._extract_balanced_json_payload(text)
        if balanced and balanced not in candidates:
            candidates.append(balanced)

        last_error: json.JSONDecodeError | None = None
        for candidate in candidates:
            try:
                return json.loads(candidate)
            except json.JSONDecodeError as exc:
                last_error = exc
        preview = _truncate_text(text.replace("\n", "\\n"), 500)
        if last_error is not None:
            raise AIClientError(f"AI response invalid JSON: {last_error}; preview={preview!r}") from last_error
        raise AIClientError(f"AI response invalid JSON; preview={preview!r}")

    @staticmethod
    def _content_to_text(content: Any) -> str:
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: List[str] = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    for key in ("text", "content", "output_text"):
                        value = item.get(key)
                        if isinstance(value, str):
                            parts.append(value)
                            break
                elif item is not None:
                    parts.append(str(item))
            return "\n".join(part for part in parts if part)
        return str(content)

    @staticmethod
    def _extract_balanced_json_payload(text: str) -> str:
        for index, char in enumerate(text):
            if char not in "{[":
                continue
            end = OpenAICompatibleAIClient._balanced_json_end(text, index)
            if end is not None:
                return text[index:end]
        return ""

    @staticmethod
    def _balanced_json_end(text: str, start: int) -> int | None:
        stack: List[str] = []
        in_string = False
        escaped = False
        for index in range(start, len(text)):
            char = text[index]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == "\"":
                    in_string = False
                continue
            if char == "\"":
                in_string = True
            elif char == "{":
                stack.append("}")
            elif char == "[":
                stack.append("]")
            elif char in "}]":
                if not stack or char != stack[-1]:
                    return None
                stack.pop()
                if not stack:
                    return index + 1
        return None


class MultiModelAIClient:
    def __init__(
        self,
        profiles: List[ModelProfile] | None = None,
        controllers: List[Any] | None = None,
        generators: List[Any] | None = None,
        validators: List[Any] | None = None,
        client_factory: Callable[[ModelProfile], Any] | None = None,
    ):
        self.controllers = list(controllers or [])
        self.generators = list(generators or [])
        self.validators = list(validators or [])
        self.client_factory = client_factory or OpenAICompatibleAIClient.from_profile
        self.last_plan: Dict[str, Any] = {}
        self.last_errors: List[Dict[str, Any]] = []
        self.last_validator_rejections: List[Dict[str, Any]] = []
        for profile in profiles or []:
            client = self.client_factory(profile)
            if profile.role in {"controller", "critic"}:
                self.controllers.append(client)
            elif profile.role in {"generator", "optimizer"}:
                self.generators.append(client)
            elif profile.role == "validator":
                self.validators.append(client)

    @classmethod
    def from_env(cls) -> "MultiModelAIClient":
        return cls(profiles=parse_model_profiles_from_env())

    def generate_candidates(self, batch_size: int, context: Dict[str, Any]) -> List[CandidateSpec]:
        self.last_errors = []
        self.last_validator_rejections = []
        if not self.generators:
            raise AIClientError("AI_CLIENT=multi requires at least one generator or optimizer profile")
        allocation = self._allocation(batch_size, context)
        candidates = self._generate_allocated_candidates(allocation, context)
        candidates = self._tag_orchestration_metadata(candidates, allocation)
        candidates = self._dedupe_candidates(candidates, context)[:batch_size]
        candidates = self.validate_candidate_specs(candidates, batch_size, context, preserve_setting_variants=False)
        initial_rejections = list(self.last_validator_rejections)
        candidates = self._repair_candidates_once(
            candidates,
            batch_size,
            context,
            initial_rejections,
        )
        if not candidates and self.last_errors:
            details = "; ".join(
                f"{item.get('profile')}:{item.get('error')}" for item in self.last_errors
            )
            raise AIClientError(f"all multi-model generation paths failed: {details}")
        return candidates

    def validate_candidate_specs(
        self,
        candidates: List[CandidateSpec],
        batch_size: int,
        context: Dict[str, Any],
        preserve_setting_variants: bool = True,
    ) -> List[CandidateSpec]:
        self.last_validator_rejections = []
        if not self.validators or not candidates:
            return candidates[:batch_size]
        validator = self.validators[0]
        if not hasattr(validator, "validate_candidate_specs"):
            return candidates[:batch_size]
        try:
            before_validation = list(candidates)
            validated = self._call_with_hard_timeout(
                validator,
                validator.validate_candidate_specs,
                candidates,
                batch_size,
                context,
            )
            if not isinstance(validated, list):
                return candidates[:batch_size]
            validated = self._restore_validator_settings(before_validation, validated)
            if preserve_setting_variants:
                validated = self._dedupe_candidates_by_settings(validated)[:batch_size]
            else:
                validated = self._dedupe_candidates(validated, context)[:batch_size]
            self.last_validator_rejections = self._validator_rejections(
                before_validation,
                validated,
                validator,
            )
            return validated
        except Exception as exc:
            self._record_model_error(validator, exc)
            return candidates[:batch_size]

    def _repair_candidates_once(
        self,
        candidates: List[CandidateSpec],
        batch_size: int,
        context: Dict[str, Any],
        initial_rejections: List[Dict[str, Any]],
    ) -> List[CandidateSpec]:
        remaining_slots = max(0, int(batch_size) - len(candidates))
        if remaining_slots <= 0 or not self.controllers:
            self.last_validator_rejections = initial_rejections
            return candidates[:batch_size]

        controller = self.controllers[0]
        if not hasattr(controller, "repair_candidate_batch"):
            self.last_validator_rejections = initial_rejections
            return candidates[:batch_size]

        generator_profiles = [_client_profile_dict(generator) for generator in self.generators]
        final_plan = dict(self.last_plan or {})
        accepted = [_candidate_to_dict(candidate) for candidate in candidates]
        controller_context = self._controller_context(context)
        critiques = self._collect_candidate_critiques(
            batch_size,
            controller_context,
            generator_profiles,
            final_plan,
            accepted,
            initial_rejections,
            list(self.last_errors),
            remaining_slots,
        )
        try:
            decision = self._call_with_hard_timeout(
                controller,
                controller.repair_candidate_batch,
                batch_size,
                controller_context,
                generator_profiles,
                final_plan,
                accepted,
                initial_rejections,
                list(self.last_errors),
                critiques,
                remaining_slots,
            )
        except Exception as exc:
            self._record_model_error(controller, exc)
            self.last_validator_rejections = initial_rejections
            return candidates[:batch_size]
        if not isinstance(decision, dict):
            self.last_validator_rejections = initial_rejections
            return candidates[:batch_size]

        action = str(decision.get("action") or "").strip().lower()
        repair_guidance = self._normalize_repair_guidance(decision)
        repair_record: Dict[str, Any] = {
            "action": action or "accept",
            "remaining_slots": remaining_slots,
            "accepted_count": len(candidates),
            "initial_validator_rejections": len(initial_rejections),
            "critic_feedback": critiques,
            "controller_decision": decision,
        }
        if repair_guidance:
            repair_record["profile_guidance"] = repair_guidance
        self.last_plan["intra_round_repair"] = repair_record

        if action not in {"refill", "repair"}:
            self.last_validator_rejections = initial_rejections
            return candidates[:batch_size]

        refill_allocation = self._normalize_repair_allocation(
            decision.get("allocation") or decision.get("refill_allocation"),
            remaining_slots,
            context,
        )
        if not refill_allocation:
            self.last_validator_rejections = initial_rejections
            return candidates[:batch_size]

        self.last_plan["intra_round_repair"]["refill_allocation"] = refill_allocation
        original_guidance = dict(self.last_plan.get("profile_guidance") or {})
        if repair_guidance:
            self.last_plan["profile_guidance"] = self._merge_profile_guidance(original_guidance, repair_guidance)
        repair_context = self._repair_context(
            context,
            accepted,
            initial_rejections,
            critiques,
            decision,
            remaining_slots,
        )
        refill_candidates = self._generate_allocated_candidates(refill_allocation, repair_context)
        refill_candidates = self._tag_orchestration_metadata(refill_candidates, refill_allocation)
        refill_candidates = self._tag_repair_metadata(refill_candidates, refill_allocation, decision)
        combined = self._dedupe_candidates(candidates + refill_candidates, context)[:batch_size]
        validated = self.validate_candidate_specs(combined, batch_size, context, preserve_setting_variants=False)
        repair_rejections = list(self.last_validator_rejections)
        self.last_validator_rejections = initial_rejections + repair_rejections
        self.last_plan["intra_round_repair"]["generated"] = len(refill_candidates)
        self.last_plan["intra_round_repair"]["final_count"] = len(validated)
        return validated[:batch_size]

    def _allocation(self, batch_size: int, context: Dict[str, Any]) -> Dict[str, int]:
        generator_profiles = [_client_profile_dict(generator) for generator in self.generators]
        if self.controllers:
            controller = self.controllers[0]
            if hasattr(controller, "plan_generation"):
                controller_context = self._controller_context(context)
                try:
                    initial_plan = self._call_with_hard_timeout(
                        controller,
                        controller.plan_generation,
                        batch_size,
                        controller_context,
                        generator_profiles,
                    )
                    plan = initial_plan if isinstance(initial_plan, dict) else {}
                    critiques = self._collect_controller_critiques(
                        batch_size,
                        controller_context,
                        generator_profiles,
                        plan,
                    )
                    final_plan = self._finalize_controller_plan(
                        controller,
                        batch_size,
                        controller_context,
                        generator_profiles,
                        plan,
                        critiques,
                    )
                    final_plan = final_plan if isinstance(final_plan, dict) else plan
                    guidance = self._normalize_profile_guidance(final_plan)
                    normalized = self._fallback_allocation(batch_size, context)
                    if normalized:
                        self.last_plan = {
                            "controller": _client_profile_name(controller),
                            "controller_model": str(getattr(controller, "model", "") or ""),
                            "controller_mode": str(final_plan.get("mode") or plan.get("mode") or ""),
                            "allocation": normalized,
                            "profile_guidance": guidance,
                        }
                        experiment_plan = _experiment_plan_from_context(context)
                        if experiment_plan:
                            self.last_plan["experiment_plan"] = experiment_plan
                        if critiques:
                            self.last_plan["critics"] = critiques
                        if final_plan is not plan:
                            self.last_plan["initial_controller_mode"] = str(plan.get("mode") or "")
                        return normalized
                except Exception as exc:
                    self._record_model_error(controller, exc)
                    pass
        allocation = self._fallback_allocation(batch_size, context)
        self.last_plan = {"controller": "fallback", "allocation": allocation, "profile_guidance": {}}
        experiment_plan = _experiment_plan_from_context(context)
        if experiment_plan:
            self.last_plan["experiment_plan"] = experiment_plan
        return allocation

    def _collect_controller_critiques(
        self,
        batch_size: int,
        context: Dict[str, Any],
        generator_profiles: List[Dict[str, Any]],
        initial_plan: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        critics = [
            critic
            for critic in self.controllers[1:]
            if hasattr(critic, "critique_generation_plan")
        ]
        if not critics:
            return []

        critiques: List[Dict[str, Any]] = []
        if len(critics) == 1:
            critic = critics[0]
            try:
                feedback = self._call_with_hard_timeout(
                    critic,
                    critic.critique_generation_plan,
                    batch_size,
                    context,
                    generator_profiles,
                    initial_plan,
                )
                critiques.append(self._controller_critique_record(critic, feedback))
            except Exception as exc:
                self._record_model_error(critic, exc)
            return critiques

        executor = ThreadPoolExecutor(max_workers=len(critics))
        future_to_critic = {
            executor.submit(
                self._call_with_hard_timeout,
                critic,
                critic.critique_generation_plan,
                batch_size,
                context,
                generator_profiles,
                initial_plan,
            ): critic
            for critic in critics
        }
        try:
            for future, critic in future_to_critic.items():
                try:
                    feedback = future.result()
                except Exception as exc:
                    self._record_model_error(critic, exc)
                    continue
                critiques.append(self._controller_critique_record(critic, feedback))
        finally:
            executor.shutdown(wait=False, cancel_futures=True)
        return critiques

    def _collect_candidate_critiques(
        self,
        batch_size: int,
        context: Dict[str, Any],
        generator_profiles: List[Dict[str, Any]],
        final_plan: Dict[str, Any],
        accepted_candidates: List[Dict[str, Any]],
        validator_rejections: List[Dict[str, Any]],
        model_errors: List[Dict[str, Any]],
        remaining_slots: int,
    ) -> List[Dict[str, Any]]:
        critics = [
            critic
            for critic in self.controllers[1:]
            if hasattr(critic, "critique_candidate_batch")
        ]
        if not critics:
            return []

        critiques: List[Dict[str, Any]] = []
        if len(critics) == 1:
            critic = critics[0]
            try:
                feedback = self._call_with_hard_timeout(
                    critic,
                    critic.critique_candidate_batch,
                    batch_size,
                    context,
                    generator_profiles,
                    final_plan,
                    accepted_candidates,
                    validator_rejections,
                    model_errors,
                    remaining_slots,
                )
                critiques.append(self._controller_critique_record(critic, feedback))
            except Exception as exc:
                self._record_model_error(critic, exc)
            return critiques

        executor = ThreadPoolExecutor(max_workers=len(critics))
        future_to_critic = {
            executor.submit(
                self._call_with_hard_timeout,
                critic,
                critic.critique_candidate_batch,
                batch_size,
                context,
                generator_profiles,
                final_plan,
                accepted_candidates,
                validator_rejections,
                model_errors,
                remaining_slots,
            ): critic
            for critic in critics
        }
        try:
            for future, critic in future_to_critic.items():
                try:
                    feedback = future.result()
                except Exception as exc:
                    self._record_model_error(critic, exc)
                    continue
                critiques.append(self._controller_critique_record(critic, feedback))
        finally:
            executor.shutdown(wait=False, cancel_futures=True)
        return critiques

    def _finalize_controller_plan(
        self,
        controller: Any,
        batch_size: int,
        context: Dict[str, Any],
        generator_profiles: List[Dict[str, Any]],
        initial_plan: Dict[str, Any],
        critiques: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        if not critiques or not hasattr(controller, "finalize_generation_plan"):
            return initial_plan
        try:
            finalized = self._call_with_hard_timeout(
                controller,
                controller.finalize_generation_plan,
                batch_size,
                context,
                generator_profiles,
                initial_plan,
                critiques,
            )
        except Exception as exc:
            self._record_model_error(controller, exc)
            return initial_plan
        return finalized if isinstance(finalized, dict) else initial_plan

    @staticmethod
    def _controller_critique_record(controller: Any, feedback: Any) -> Dict[str, Any]:
        return {
            "critic": _client_profile_name(controller),
            "profile": _client_profile_name(controller),
            "model": str(getattr(controller, "model", "") or ""),
            "feedback": feedback if isinstance(feedback, dict) else {"text": str(feedback or "")},
        }

    @staticmethod
    def _controller_context(context: Dict[str, Any]) -> Dict[str, Any]:
        controller_context = dict(context or {})
        research_context = controller_context.get("research_context")
        if isinstance(research_context, dict):
            controller_context["research_context"] = _compact_controller_research_context(research_context)
        return controller_context

    def _generate_allocated_candidates(
        self,
        allocation: Dict[str, int],
        context: Dict[str, Any],
    ) -> List[CandidateSpec]:
        tasks = [
            (generator, int(allocation.get(_client_profile_name(generator)) or 0))
            for generator in self.generators
        ]
        tasks = [(generator, count) for generator, count in tasks if count > 0]
        if not tasks:
            return []
        if len(tasks) == 1:
            generator, count = tasks[0]
            try:
                generator_context = self._context_for_generator(generator, context)
                return self._call_with_hard_timeout(
                    generator,
                    self._generate_with_split_retry,
                    generator,
                    count,
                    generator_context,
                )
            except Exception as exc:
                self._record_model_error(generator, exc)
                return []

        candidates: List[CandidateSpec] = []
        executor = ThreadPoolExecutor(max_workers=len(tasks))
        started = time.monotonic()
        future_to_task = {
            executor.submit(
                self._generate_with_split_retry,
                generator,
                count,
                self._context_for_generator(generator, context),
            ): (generator, count)
            for generator, count in tasks
        }
        deadlines = {
            future: started + _client_hard_timeout(generator)
            for future, (generator, _count) in future_to_task.items()
        }
        pending = set(future_to_task)
        try:
            while pending:
                now = time.monotonic()
                expired = [future for future in pending if deadlines[future] <= now]
                for future in expired:
                    generator, _count = future_to_task[future]
                    future.cancel()
                    self._record_model_error(
                        generator,
                        TimeoutError(f"model call timed out after {_client_hard_timeout(generator):.1f}s"),
                    )
                    pending.remove(future)
                if not pending:
                    break

                done = {future for future in pending if future.done()}
                if not done:
                    next_deadline = min(deadlines[future] for future in pending)
                    sleep_for = max(0.0, min(0.05, next_deadline - time.monotonic()))
                    if sleep_for:
                        time.sleep(sleep_for)
                    continue

                for future in done:
                    generator, count = future_to_task[future]
                    try:
                        generated = future.result()
                    except Exception as exc:
                        self._record_model_error(generator, exc)
                        pending.remove(future)
                        continue
                    candidates.extend(generated[:count])
                    pending.remove(future)
        finally:
            executor.shutdown(wait=False, cancel_futures=True)
        return candidates

    def _call_with_hard_timeout(self, client: Any, func: Callable[..., Any], *args: Any) -> Any:
        timeout = _client_hard_timeout(client)
        executor = ThreadPoolExecutor(max_workers=1)
        future = executor.submit(func, *args)
        try:
            return future.result(timeout=timeout)
        except FutureTimeoutError as exc:
            future.cancel()
            raise TimeoutError(f"model call timed out after {timeout:.1f}s") from exc
        finally:
            executor.shutdown(wait=False, cancel_futures=True)

    def _generate_with_split_retry(
        self,
        generator: Any,
        count: int,
        context: Dict[str, Any],
    ) -> List[CandidateSpec]:
        if count <= 0:
            return []
        try:
            return generator.generate_candidates(count, context)[:count]
        except Exception as exc:
            self._record_model_error(generator, exc)
            if count <= 1 or not _is_split_retryable_generation_error(exc):
                return []
            left = count // 2
            right = count - left
            recovered = self._generate_with_split_retry(generator, left, context)
            recovered.extend(self._generate_with_split_retry(generator, right, context))
            return recovered[:count]

    def _normalize_allocation(
        self,
        allocation: Any,
        batch_size: int,
        context: Dict[str, Any],
        plan: Dict[str, Any] | None = None,
    ) -> Dict[str, int]:
        if self._balanced_exploration_required(context, plan):
            return self._fallback_allocation(batch_size, context)
        if not isinstance(allocation, dict):
            return {}
        names = [_client_profile_name(generator) for generator in self._eligible_generators(context, plan)]
        if not names:
            names = [_client_profile_name(generator) for generator in self.generators]
        models = {str(getattr(generator, "model", "")): _client_profile_name(generator) for generator in self.generators}
        result = {name: 0 for name in names}
        remaining = int(batch_size)
        for key, raw_count in allocation.items():
            name = str(key)
            if name not in result:
                name = models.get(name, name)
            if name not in result:
                continue
            try:
                count = max(0, int(raw_count))
            except (TypeError, ValueError):
                continue
            count = min(count, remaining)
            result[name] += count
            remaining -= count
            if remaining <= 0:
                break
        if remaining > 0:
            fallback = self._fallback_allocation(remaining, context)
            for name, count in fallback.items():
                result[name] = result.get(name, 0) + count
        return {name: count for name, count in result.items() if count > 0}

    def _normalize_profile_guidance(self, plan: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        raw = plan.get("profile_guidance") or plan.get("guidance") or plan.get("directions")
        if not isinstance(raw, dict):
            return {}
        profile_names = {_client_profile_name(generator) for generator in self.generators}
        model_to_profile = {
            str(getattr(generator, "model", "") or ""): _client_profile_name(generator)
            for generator in self.generators
        }
        normalized: Dict[str, Dict[str, Any]] = {}
        for key, value in raw.items():
            profile_name = str(key)
            if profile_name not in profile_names:
                profile_name = model_to_profile.get(profile_name, profile_name)
            if profile_name not in profile_names or not isinstance(value, dict):
                continue
            cleaned = {
                str(item_key): item_value
                for item_key, item_value in value.items()
                if item_value not in (None, "", [], {})
            }
            if cleaned:
                normalized[profile_name] = cleaned
        return normalized

    def _normalize_repair_guidance(self, decision: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        raw = (
            decision.get("profile_guidance")
            or decision.get("repair_guidance")
            or decision.get("profile_guidance_delta")
            or decision.get("guidance")
        )
        return self._normalize_profile_guidance({"profile_guidance": raw}) if isinstance(raw, dict) else {}

    def _normalize_repair_allocation(
        self,
        raw_allocation: Any,
        remaining_slots: int,
        context: Dict[str, Any],
    ) -> Dict[str, int]:
        remaining = max(0, int(remaining_slots))
        if remaining <= 0:
            return {}
        names = [_client_profile_name(generator) for generator in self.generators]
        model_to_profile = {
            str(getattr(generator, "model", "") or ""): _client_profile_name(generator)
            for generator in self.generators
        }
        result = {name: 0 for name in names}
        if isinstance(raw_allocation, dict):
            for key, raw_count in raw_allocation.items():
                name = str(key)
                if name not in result:
                    name = model_to_profile.get(name, name)
                if name not in result:
                    continue
                try:
                    count = max(0, int(raw_count))
                except (TypeError, ValueError):
                    continue
                count = min(count, remaining)
                result[name] += count
                remaining -= count
                if remaining <= 0:
                    break
        if not any(result.values()):
            return self._fallback_allocation(remaining_slots, context)
        if remaining > 0:
            fallback = self._fallback_allocation(remaining, context)
            for name, count in fallback.items():
                result[name] = result.get(name, 0) + count
        return {name: count for name, count in result.items() if count > 0}

    @staticmethod
    def _merge_profile_guidance(
        base: Dict[str, Dict[str, Any]],
        overlay: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Dict[str, Any]]:
        merged = {profile: dict(guidance) for profile, guidance in base.items() if isinstance(guidance, dict)}
        for profile, guidance in overlay.items():
            if not isinstance(guidance, dict):
                continue
            current = dict(merged.get(profile) or {})
            current.update(guidance)
            merged[profile] = current
        return merged

    @staticmethod
    def _repair_context(
        context: Dict[str, Any],
        accepted_candidates: List[Dict[str, Any]],
        validator_rejections: List[Dict[str, Any]],
        critic_feedback: List[Dict[str, Any]],
        decision: Dict[str, Any],
        remaining_slots: int,
    ) -> Dict[str, Any]:
        repair_context = dict(context or {})
        research_context = repair_context.get("research_context")
        research_context = dict(research_context) if isinstance(research_context, dict) else {}
        research_context["intra_round_repair"] = {
            "remaining_slots": remaining_slots,
            "accepted_candidates": accepted_candidates,
            "validator_rejections": validator_rejections,
            "critic_feedback": critic_feedback,
            "controller_decision": decision,
            "policy": (
                "Generate only replacement candidates for this same-round repair pass. "
                "Avoid local variants of validator-rejected expressions and preserve the controller repair guidance."
            ),
        }
        repair_context["research_context"] = research_context
        return repair_context

    def _context_for_generator(self, generator: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        guidance = dict(self.last_plan.get("profile_guidance") or {})
        profile_name = _client_profile_name(generator)
        profile_guidance = guidance.get(profile_name)
        generator_context = dict(context or {})
        research_context = generator_context.get("research_context")
        research_context = dict(research_context) if isinstance(research_context, dict) else {}
        if isinstance(profile_guidance, dict) and profile_guidance:
            experiment_plan = research_context.get("experiment_plan") if isinstance(research_context.get("experiment_plan"), dict) else {}
            family_control = experiment_plan.get("family_diversity_control") if isinstance(experiment_plan, dict) else {}
            research_context["profile_name"] = profile_name
            research_context["profile_role"] = str(getattr(generator, "role", "") or "")
            research_context["profile_guidance"] = profile_guidance
            if isinstance(family_control, dict) and family_control:
                dominant_family = str(family_control.get("dominant_family") or "").strip()
                alternate_families = [str(item) for item in family_control.get("alternate_families") or [] if str(item).strip()]
                if dominant_family:
                    current_family = str(profile_guidance.get("field_family") or "").strip()
                    if current_family != dominant_family:
                        research_context["family_diversity_control"] = {
                            **family_control,
                            "avoid_families": [dominant_family],
                            "preferred_families": alternate_families,
                        }
                    else:
                        research_context["family_diversity_control"] = {
                            **family_control,
                            "anchor_profile": profile_name,
                        }
            research_context["orchestration_policy"] = (
                "Follow profile_guidance for research direction. Candidate counts are fixed by the service; "
                "do not discuss or change allocation. If family_diversity_control is present, keep the dominant "
                "family anchored to a single profile and avoid it in the other active profiles."
            )
        generator_context["research_context"] = _compact_generator_research_context(research_context)
        return generator_context

    def _fallback_allocation(self, batch_size: int, context: Dict[str, Any]) -> Dict[str, int]:
        eligible_generators = self._eligible_generators(context)
        names = [_client_profile_name(generator) for generator in eligible_generators]
        if not names:
            return {}
        base = batch_size // len(names)
        remainder = batch_size % len(names)
        return {
            name: base + (1 if idx < remainder else 0)
            for idx, name in enumerate(names)
            if base + (1 if idx < remainder else 0) > 0
        }

    def _eligible_generators(
        self,
        context: Dict[str, Any],
        plan: Dict[str, Any] | None = None,
    ) -> List[Any]:
        return list(self.generators)

    @staticmethod
    def _balanced_exploration_required(context: Dict[str, Any], plan: Dict[str, Any] | None = None) -> bool:
        plan_mode = str((plan or {}).get("mode") or "").lower()
        if plan_mode in {"optimize", "repair", "optimize_best"}:
            return False
        research_context = dict(context or {}).get("research_context")
        experiment_plan = research_context.get("experiment_plan") if isinstance(research_context, dict) else {}
        experiment_mode = str(experiment_plan.get("mode") or "").lower() if isinstance(experiment_plan, dict) else ""
        return experiment_mode not in {"optimize_best", "repair"}

    @staticmethod
    def _optimizer_allowed(context: Dict[str, Any], plan: Dict[str, Any] | None = None) -> bool:
        plan_mode = str((plan or {}).get("mode") or "").lower()
        if plan_mode in {"optimize", "repair", "optimize_best"}:
            return True
        research_context = dict(context or {}).get("research_context")
        experiment_plan = research_context.get("experiment_plan") if isinstance(research_context, dict) else {}
        experiment_mode = str(experiment_plan.get("mode") or "").lower() if isinstance(experiment_plan, dict) else ""
        return experiment_mode in {"optimize_best", "repair"}

    @staticmethod
    def _dedupe_candidates(candidates: List[CandidateSpec], context: Dict[str, Any] | None = None) -> List[CandidateSpec]:
        research_context = dict(context or {}).get("research_context")
        generation_policy = research_context.get("generation_policy") if isinstance(research_context, dict) else {}
        reject_structural_duplicates = bool(
            isinstance(generation_policy, dict) and generation_policy.get("avoid_structural_duplicates")
        )
        max_per_structure = _max_batch_candidates_per_structure(context)
        seen = set()
        seen_structures = set()
        structure_counts: Dict[str, int] = {}
        overused_structure_keys = _overused_structure_keys(context)
        deduped: List[CandidateSpec] = []
        for candidate in candidates:
            key = re.sub(r"\s+", "", candidate.expression.lower())
            if key in seen:
                continue
            variant_key = expression_variant_key(candidate.expression)
            if reject_structural_duplicates and variant_key in seen_structures:
                continue
            structure_key = expression_structure_key(candidate.expression)
            if structure_key in overused_structure_keys:
                continue
            if max_per_structure > 0 and int(structure_counts.get(structure_key, 0)) >= max_per_structure:
                continue
            seen.add(key)
            seen_structures.add(variant_key)
            structure_counts[structure_key] = int(structure_counts.get(structure_key, 0)) + 1
            deduped.append(candidate)
        return deduped

    @staticmethod
    def _dedupe_candidates_by_settings(candidates: List[CandidateSpec]) -> List[CandidateSpec]:
        seen = set()
        deduped: List[CandidateSpec] = []
        for candidate in candidates:
            key = _candidate_spec_key(candidate.expression, candidate.settings)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(candidate)
        return deduped

    @staticmethod
    def _restore_validator_settings(
        original: List[CandidateSpec],
        validated: List[CandidateSpec],
    ) -> List[CandidateSpec]:
        by_expression: Dict[str, List[CandidateSpec]] = {}
        for candidate in original:
            by_expression.setdefault(_candidate_key(candidate.expression), []).append(candidate)
        used_by_expression: Dict[str, int] = {}
        restored: List[CandidateSpec] = []
        for idx, candidate in enumerate(validated):
            settings = dict(candidate.settings or {})
            if not settings:
                if idx < len(original) and _candidate_key(original[idx].expression) == _candidate_key(candidate.expression):
                    settings = dict(original[idx].settings)
                else:
                    expression_key = _candidate_key(candidate.expression)
                    used = used_by_expression.get(expression_key, 0)
                    matches = by_expression.get(expression_key) or []
                    if used < len(matches):
                        settings = dict(matches[used].settings)
                        used_by_expression[expression_key] = used + 1
            restored.append(
                CandidateSpec(
                    expression=candidate.expression,
                    settings=settings,
                    source=candidate.source,
                    metadata=candidate.metadata,
                )
            )
        return restored

    def _tag_orchestration_metadata(
        self,
        candidates: List[CandidateSpec],
        allocation: Dict[str, int],
    ) -> List[CandidateSpec]:
        tagged: List[CandidateSpec] = []
        controller = str(self.last_plan.get("controller") or "fallback")
        guidance = dict(self.last_plan.get("profile_guidance") or {})
        for candidate in candidates:
            candidate_guidance = guidance.get(str(candidate.metadata.get("model_profile") or ""))
            metadata = {
                **candidate.metadata,
                "orchestrated_by": controller,
                "model_allocation": allocation,
            }
            experiment_plan = self._active_experiment_plan()
            if experiment_plan:
                metadata.update(experiment_plan)
            if isinstance(candidate_guidance, dict) and candidate_guidance:
                metadata["profile_guidance"] = candidate_guidance
            tagged.append(
                CandidateSpec(
                    expression=candidate.expression,
                    settings=candidate.settings,
                    source=candidate.source,
                    metadata=metadata,
                )
            )
        return tagged

    @staticmethod
    def _tag_repair_metadata(
        candidates: List[CandidateSpec],
        allocation: Dict[str, int],
        decision: Dict[str, Any],
    ) -> List[CandidateSpec]:
        tagged: List[CandidateSpec] = []
        for candidate in candidates:
            tagged.append(
                CandidateSpec(
                    expression=candidate.expression,
                    settings=candidate.settings,
                    source=candidate.source,
                    metadata={
                        **candidate.metadata,
                        "intra_round_repair": True,
                        "repair_round": 1,
                        "repair_action": str(decision.get("action") or ""),
                        "repair_allocation": allocation,
                        "repair_rationale": str(decision.get("rationale") or ""),
                    },
                )
            )
        return tagged

    def _active_experiment_plan(self) -> Dict[str, Any]:
        plan = self.last_plan.get("experiment_plan")
        if not isinstance(plan, dict):
            return {}
        metadata: Dict[str, Any] = {}
        for source_key, target_key in (
            ("mode", "experiment_plan_mode"),
            ("target_candidate_id", "target_candidate_id"),
            ("optimization_anchor_id", "optimization_anchor_id"),
            ("optimize_round", "optimize_round"),
        ):
            value = plan.get(source_key)
            if value not in (None, "", [], {}):
                metadata[target_key] = value
        return metadata

    def _validator_rejections(
        self,
        before_validation: List[CandidateSpec],
        after_validation: List[CandidateSpec],
        validator: Any,
    ) -> List[Dict[str, Any]]:
        accepted = {_candidate_spec_key(candidate.expression, candidate.settings) for candidate in after_validation}
        reported = getattr(validator, "last_rejections", [])
        reported_by_expression: Dict[str, Dict[str, Any]] = {}
        if isinstance(reported, list):
            for item in reported:
                if not isinstance(item, dict):
                    continue
                expression = str(item.get("expression") or "").strip()
                if expression:
                    reported_by_expression[_candidate_key(expression)] = item

        rejections: List[Dict[str, Any]] = []
        for candidate in before_validation:
            key = _candidate_spec_key(candidate.expression, candidate.settings)
            if key in accepted:
                continue
            reported_item = reported_by_expression.get(_candidate_key(candidate.expression), {})
            reason = str(reported_item.get("reason") or reported_item.get("error") or "VALIDATOR_FILTERED")
            rejections.append(
                {
                    "reason": reason,
                    "validator_profile": _client_profile_name(validator),
                    "validator_model": str(getattr(validator, "model", "") or ""),
                    "candidate": _candidate_to_dict(candidate),
                }
            )
        return rejections

    def _record_model_error(self, client: Any, exc: Exception) -> None:
        self.last_errors.append(
            {
                "profile": _client_profile_name(client),
                "model": str(getattr(client, "model", "") or ""),
                "role": str(getattr(client, "role", "") or ""),
                "error": str(exc),
            }
        )


def _client_profile_name(client: Any) -> str:
    name = str(getattr(client, "profile_name", "") or "").strip()
    if name:
        return name
    model = str(getattr(client, "model", "") or "").strip()
    return _default_profile_name(model, str(getattr(client, "role", "generator") or "generator"))


def _json_request_max_tokens() -> int:
    raw = os.environ.get("AI_JSON_MAX_TOKENS", "4096")
    try:
        return max(256, int(float(raw)))
    except (TypeError, ValueError):
        return 4096


def _compact_generator_research_context(research_context: Dict[str, Any]) -> Dict[str, Any]:
    compact = _compact_controller_research_context(research_context)
    if isinstance(research_context.get("datafields"), dict):
        compact["datafields"] = research_context["datafields"]
    if isinstance(research_context.get("history_memory"), dict):
        compact["history_memory"] = _compact_generic(research_context["history_memory"], depth=0, list_limit=40)
    for key in (
        "profile_name",
        "profile_role",
        "profile_guidance",
        "family_diversity_control",
        "orchestration_policy",
        "intra_round_repair",
    ):
        if key in research_context:
            compact[key] = _compact_generic(research_context[key], depth=0, list_limit=20)
    return compact


def _compact_controller_research_context(research_context: Dict[str, Any]) -> Dict[str, Any]:
    compact: Dict[str, Any] = {}
    for key in ("target_settings", "generation_policy", "experiment_plan"):
        if key in research_context:
            compact[key] = research_context[key]
    if isinstance(research_context.get("datafields"), dict):
        compact["datafields"] = _compact_controller_datafields(research_context["datafields"])
    if isinstance(research_context.get("syntax_constraints"), dict):
        compact["syntax_constraints"] = _compact_controller_syntax(research_context["syntax_constraints"])
    if isinstance(research_context.get("analysis"), dict):
        compact["analysis"] = _compact_controller_analysis(research_context["analysis"])
    if isinstance(research_context.get("submitted_field_avoidance"), dict):
        compact["submitted_field_avoidance"] = _compact_submitted_avoidance(research_context["submitted_field_avoidance"])
    if isinstance(research_context.get("lit_tower_avoidance"), dict):
        compact["lit_tower_avoidance"] = _compact_lit_tower_avoidance(research_context["lit_tower_avoidance"])
    if isinstance(research_context.get("candidate_queues"), dict):
        compact["candidate_queues"] = _compact_candidate_queues(research_context["candidate_queues"])
    for key in ("recent_failures", "recent_pending", "recent_successes"):
        rows = research_context.get(key)
        if isinstance(rows, list):
            compact[key] = [_compact_candidate_record(row) for row in rows[:6] if isinstance(row, dict)]
    if isinstance(research_context.get("recent_experiment_plans"), list):
        compact["recent_experiment_plans"] = [
            _compact_generic(item, depth=0, list_limit=6)
            for item in research_context["recent_experiment_plans"][:6]
        ]
    if isinstance(research_context.get("model_feedback"), dict):
        compact["model_feedback"] = _compact_model_feedback(research_context["model_feedback"])
    if isinstance(research_context.get("history_memory"), dict):
        compact["history_memory"] = _compact_generic(research_context["history_memory"], depth=0, list_limit=40)
    if isinstance(research_context.get("reference_brain_project"), dict):
        compact["reference_brain_project"] = _compact_reference_project(research_context["reference_brain_project"])
    if isinstance(research_context.get("knowledge"), dict):
        compact["knowledge"] = {
            str(key): _truncate_text(value, 1600)
            for key, value in research_context["knowledge"].items()
        }
    return compact


def _compact_controller_datafields(datafields: Dict[str, Any]) -> Dict[str, Any]:
    compact: Dict[str, Any] = {}
    for key in ("available", "source", "error"):
        if key in datafields:
            compact[key] = datafields[key]
    field_ids = datafields.get("field_ids")
    if isinstance(field_ids, list):
        compact["field_ids"] = [str(field) for field in field_ids[:500]]
        compact["field_count"] = len(field_ids)
    field_types = datafields.get("field_types")
    if isinstance(field_types, dict):
        compact["field_types"] = dict(list(field_types.items())[:500])
    datasets = datafields.get("datasets")
    if isinstance(datasets, list):
        compact["datasets"] = [_compact_generic(item, depth=0, list_limit=4) for item in datasets[:20]]
    fields = datafields.get("fields")
    if isinstance(fields, list):
        compact["detailed_field_count"] = len(fields)
    return compact


def _compact_controller_syntax(syntax: Dict[str, Any]) -> Dict[str, Any]:
    compact: Dict[str, Any] = {}
    for key in (
        "allowed_operators",
        "operator_rule",
        "field_rule",
        "auxiliary_only_fields",
        "auxiliary_field_rule",
        "vector_reducer_rule",
        "structure_dedup_rule",
        "recent_preflight_rejections",
    ):
        if key in syntax:
            compact[key] = _compact_generic(syntax[key], depth=0, list_limit=40)
    structures = syntax.get("recent_expression_structures")
    if isinstance(structures, list):
        compact["recent_expression_structures"] = [
            _compact_candidate_record(item)
            for item in structures[:10]
            if isinstance(item, dict)
        ]
    return compact


def _compact_controller_analysis(analysis: Dict[str, Any]) -> Dict[str, Any]:
    compact: Dict[str, Any] = {}
    for key in (
        "candidate_count",
        "candidate_queue_counts",
        "failure_reasons",
        "promising_fields",
        "weak_fields",
        "family_diversity_control",
        "submitted_avoid_fields",
        "lit_tower_avoidance",
        "route_efficiency",
        "structure_diversity_control",
        "observed_quality_thresholds",
        "optimization_state",
    ):
        if key in analysis:
            compact[key] = _compact_generic(analysis[key], depth=0, list_limit=40)
    best = analysis.get("best_candidate")
    if isinstance(best, dict):
        compact["best_candidate"] = _compact_candidate_record(best)
    for key, limit in (("field_stats", 30), ("field_family_stats", 20)):
        stats = analysis.get(key)
        if isinstance(stats, dict):
            compact[key] = dict(list(stats.items())[:limit])
    return compact


def _compact_submitted_avoidance(avoidance: Dict[str, Any]) -> Dict[str, Any]:
    compact: Dict[str, Any] = {}
    for key in ("fields", "families", "policy"):
        if key in avoidance:
            compact[key] = _compact_generic(avoidance[key], depth=0, list_limit=80)
    examples = avoidance.get("examples")
    if isinstance(examples, list):
        compact["examples"] = [
            _compact_candidate_record(item)
            for item in examples[:8]
            if isinstance(item, dict)
        ]
    return compact


def _compact_lit_tower_avoidance(avoidance: Dict[str, Any]) -> Dict[str, Any]:
    compact: Dict[str, Any] = {}
    for key in ("source", "source_policy", "tower_names", "categories", "policy"):
        if key in avoidance:
            compact[key] = _compact_generic(avoidance[key], depth=0, list_limit=80)
    for key in ("lit_towers", "unlit_towers", "examples"):
        rows = avoidance.get(key)
        if isinstance(rows, list):
            compact[key] = [_compact_generic(item, depth=0, list_limit=12) for item in rows[:12]]
    return compact


def _compact_candidate_queues(queues: Dict[str, Any]) -> Dict[str, Any]:
    compact: Dict[str, Any] = {}
    if isinstance(queues.get("counts"), dict):
        compact["counts"] = queues["counts"]
    for key in ("submitable", "watchlist", "optimize", "trash", "abandoned"):
        rows = queues.get(key)
        if isinstance(rows, list):
            compact[key] = [_compact_candidate_record(row) for row in rows[:5] if isinstance(row, dict)]
    return compact


def _compact_model_feedback(feedback: Dict[str, Any]) -> Dict[str, Any]:
    compact: Dict[str, Any] = {}
    by_profile = feedback.get("by_profile")
    if isinstance(by_profile, dict):
        compact_profiles: Dict[str, Any] = {}
        for profile, item in list(by_profile.items())[:8]:
            if not isinstance(item, dict):
                continue
            profile_item = {
                key: value
                for key, value in item.items()
                if key not in {"examples", "guidance"}
            }
            if isinstance(item.get("examples"), list):
                profile_item["examples"] = [
                    _compact_candidate_record(row)
                    for row in item["examples"][:2]
                    if isinstance(row, dict)
                ]
            if isinstance(item.get("guidance"), dict):
                profile_item["guidance"] = _compact_generic(item["guidance"], depth=0, list_limit=8)
            compact_profiles[str(profile)] = profile_item
        compact["by_profile"] = compact_profiles
    recent = feedback.get("recent")
    if isinstance(recent, list):
        compact["recent"] = [_compact_generic(item, depth=0, list_limit=8) for item in recent[:10]]
    return compact


def _compact_reference_project(reference: Dict[str, Any]) -> Dict[str, Any]:
    compact: Dict[str, Any] = {}
    for key in ("available", "path"):
        if key in reference:
            compact[key] = reference[key]
    for key in ("submitted_success_examples", "recent_failure_examples"):
        rows = reference.get(key)
        if isinstance(rows, list):
            compact[key] = [_compact_candidate_record(row) for row in rows[:5] if isinstance(row, dict)]
    templates = reference.get("usa_d0_template_families")
    if isinstance(templates, dict):
        compact["usa_d0_template_families"] = list(templates.keys())[:20]
    return compact


def _compact_candidate_record(item: Dict[str, Any]) -> Dict[str, Any]:
    keep = (
        "id",
        "candidate_id",
        "alpha_id",
        "status",
        "source",
        "queue",
        "queue_reason",
        "fields",
        "settings",
        "metrics",
        "checks",
        "quality_score",
        "readiness_score",
        "submission_score",
        "failed_checks",
        "warning_checks",
        "pending_checks",
        "passed_checks",
        "reason",
        "errors",
    )
    compact = {key: _compact_generic(item[key], depth=0, list_limit=12) for key in keep if key in item}
    for key in ("expression", "structure_key", "variant_key", "hypothesis", "risk_notes", "fail_reason"):
        if key in item:
            compact[key] = _truncate_text(item[key], 320)
    event = item.get("last_relevant_event")
    if isinstance(event, dict):
        compact["last_relevant_event"] = _compact_generic(event, depth=0, list_limit=8)
    generated = item.get("generated_metadata")
    if isinstance(generated, dict):
        compact["generated_metadata"] = _compact_generic(generated, depth=0, list_limit=8)
    return compact


def _compact_generic(value: Any, depth: int = 0, list_limit: int = 12) -> Any:
    if depth >= 3:
        return _truncate_text(value, 240)
    if isinstance(value, str):
        return _truncate_text(value, 800)
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    if isinstance(value, list):
        return [_compact_generic(item, depth + 1, list_limit) for item in value[:list_limit]]
    if isinstance(value, dict):
        return {
            str(key): _compact_generic(item, depth + 1, list_limit)
            for key, item in list(value.items())[:list_limit]
        }
    return _truncate_text(value, 240)


def _truncate_text(value: Any, limit: int) -> str:
    text = str(value or "")
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)] + "..."


def _experiment_plan_from_context(context: Dict[str, Any]) -> Dict[str, Any]:
    research_context = context.get("research_context") if isinstance(context, dict) else {}
    experiment_plan = research_context.get("experiment_plan") if isinstance(research_context, dict) else {}
    return dict(experiment_plan) if isinstance(experiment_plan, dict) else {}


def _client_hard_timeout(client: Any) -> float:
    raw = getattr(client, "hard_timeout", None)
    if raw is None:
        raw = getattr(client, "request_timeout", 60.0)
        try:
            return max(0.01, float(raw) * 3.0)
        except (TypeError, ValueError):
            return 180.0
    try:
        return max(0.01, float(raw))
    except (TypeError, ValueError):
        return 180.0


def _is_split_retryable_generation_error(exc: Exception) -> bool:
    message = str(exc or "").lower()
    if not message:
        return False
    retryable_markers = (
        "batch too large",
        "batch size",
        "too many candidates",
        "too many items",
        "maximum batch",
        "max batch",
        "request too large",
    )
    return any(marker in message for marker in retryable_markers)


def _max_batch_candidates_per_structure(context: Dict[str, Any] | None) -> int:
    research_context = dict(context or {}).get("research_context")
    if not isinstance(research_context, dict):
        return 0
    experiment_plan = research_context.get("experiment_plan") if isinstance(research_context.get("experiment_plan"), dict) else {}
    mode = str(experiment_plan.get("mode") or "").lower()
    if mode in {"optimize_best", "setting_sweep", "repair"}:
        return 0
    control = (
        experiment_plan.get("structure_diversity_control")
        if isinstance(experiment_plan.get("structure_diversity_control"), dict)
        else {}
    )
    raw = control.get("max_batch_candidates_per_structure")
    generation_policy = (
        research_context.get("generation_policy") if isinstance(research_context.get("generation_policy"), dict) else {}
    )
    if raw in (None, ""):
        raw = generation_policy.get("max_batch_candidates_per_structure")
    try:
        value = int(float(raw))
    except (TypeError, ValueError):
        return 0
    return max(0, value)


def _overused_structure_keys(context: Dict[str, Any] | None) -> set[str]:
    research_context = dict(context or {}).get("research_context")
    if not isinstance(research_context, dict):
        return set()
    experiment_plan = research_context.get("experiment_plan") if isinstance(research_context.get("experiment_plan"), dict) else {}
    mode = str(experiment_plan.get("mode") or "").lower()
    if mode in {"optimize_best", "setting_sweep", "repair"}:
        return set()
    control = (
        experiment_plan.get("structure_diversity_control")
        if isinstance(experiment_plan.get("structure_diversity_control"), dict)
        else {}
    )
    overused = control.get("overused_structures")
    if not isinstance(overused, list):
        return set()
    return {
        str(item.get("structure_key") or "").strip()
        for item in overused
        if isinstance(item, dict) and str(item.get("structure_key") or "").strip()
    }


def _client_profile_dict(client: Any) -> Dict[str, Any]:
    return {
        "name": _client_profile_name(client),
        "model": str(getattr(client, "model", "") or ""),
        "role": str(getattr(client, "role", "generator") or "generator"),
    }


def _candidate_key(expression: str) -> str:
    return re.sub(r"\s+", "", str(expression or "").lower())


def _candidate_spec_key(expression: str, settings: Dict[str, Any]) -> str:
    try:
        settings_key = json.dumps(settings or {}, sort_keys=True, separators=(",", ":"), default=str)
    except TypeError:
        settings_key = repr(sorted((str(key), str(value)) for key, value in dict(settings or {}).items()))
    return f"{_candidate_key(expression)}|{settings_key}"


def _candidate_to_dict(candidate: CandidateSpec) -> Dict[str, Any]:
    return {
        "expression": candidate.expression,
        "settings": dict(candidate.settings or {}),
        "source": candidate.source,
        "metadata": dict(candidate.metadata or {}),
    }


def _is_trivial_expression(expression: str) -> bool:
    compact = re.sub(r"\s+", "", expression.lower())
    if compact in {
        "rank(close)",
        "rank(open)",
        "rank(high)",
        "rank(low)",
        "rank(volume)",
        "rank(returns)",
        "rank(-returns)",
        "rank(ts_delta(close,5))",
        "rank(ts_delta(close,1))",
    }:
        return True
    if re.fullmatch(r"rank\(ts_delta\((close|open|high|low|volume|returns),\d+\)\)", compact):
        return True
    operators = re.findall(r"\b([a-z_][a-z0-9_]*)\s*\(", compact)
    basic_identifiers = {
        "rank",
        "ts_rank",
        "ts_delta",
        "delta",
        "ts_mean",
        "close",
        "open",
        "high",
        "low",
        "volume",
        "returns",
    }
    identifiers = set(re.findall(r"\b[a-z_][a-z0-9_]*\b", compact))
    return len(operators) <= 2 and identifiers.issubset(basic_identifiers)


class BrainHTTPClient:
    """HTTP adapter for WorldQuant BRAIN.

    The adapter mirrors the old project rule that a submit only succeeds after
    `/alphas/{id}` verifies `stage == OS` and `dateSubmitted` is present.
    """

    def __init__(
        self,
        session: Any | None = None,
        base_url: str = "https://api.worldquantbrain.com",
        max_poll_attempts: int = 5,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.session = session or self._default_session()
        self.base_url = base_url.rstrip("/")
        self.max_poll_attempts = max_poll_attempts
        self.sleep = sleep

    @classmethod
    def from_env(cls) -> "BrainHTTPClient":
        client = cls(base_url=os.environ.get("BRAIN_BASE_URL", "https://api.worldquantbrain.com"))
        email, password = cls.credentials_from_env()
        if email and password:
            client.authenticate(email, password)
        return client

    @staticmethod
    def credentials_from_env() -> tuple[str | None, str | None]:
        email = os.environ.get("BRAIN_EMAIL")
        password = os.environ.get("BRAIN_PASSWORD")
        if email and password:
            return email, password

        credential_file = os.environ.get("BRAIN_CREDENTIALS_FILE")
        if not credential_file:
            return email, password
        path = Path(credential_file)
        if not path.exists():
            raise RuntimeError(f"BRAIN_CREDENTIALS_FILE not found: {path}")
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list) and len(data) >= 2:
            return str(data[0]), str(data[1])
        if isinstance(data, dict):
            return str(data.get("email") or data.get("username") or ""), str(data.get("password") or "")
        raise RuntimeError("BRAIN_CREDENTIALS_FILE must contain [email, password] or {email,password}")

    @staticmethod
    def _default_session() -> Any:
        try:
            import requests  # type: ignore
        except ImportError as exc:
            raise RuntimeError("requests is required for BRAIN_CLIENT=http") from exc
        return requests.Session()

    def authenticate(self, email: str, password: str) -> None:
        try:
            from requests.auth import HTTPBasicAuth  # type: ignore
        except ImportError as exc:
            raise RuntimeError("requests is required for BRAIN authentication") from exc
        self.session.auth = HTTPBasicAuth(email, password)
        response = self.session.post(f"{self.base_url}/authentication")
        if response.status_code not in (200, 201):
            raise RuntimeError(f"BRAIN authentication failed: HTTP {response.status_code}")

    def simulate(self, expression: str, settings: Dict[str, Any]) -> SimulationResult:
        payload = {"type": "REGULAR", "settings": settings, "regular": expression}
        response = self.session.post(f"{self.base_url}/simulations", json=payload)
        if response.status_code != 201 or "Location" not in response.headers:
            raise RuntimeError(f"simulation creation failed: HTTP {response.status_code} {response.text[:200]}")

        location = response.headers["Location"]
        alpha_id = self._poll_simulation_for_alpha(location)
        detail = self.get_alpha_detail(alpha_id)
        is_metrics = detail.get("is", {}) if isinstance(detail, dict) else {}
        checks = self.get_submission_check(alpha_id) or self._checks_from_alpha_detail(detail)
        return SimulationResult(alpha_id=alpha_id, metrics=is_metrics, checks=checks, raw={"detail": detail})

    def simulate_many(self, items: List[tuple[str, Dict[str, Any]]]) -> List[SimulationResult | SimulationFailure]:
        if not items:
            return []
        if len(items) == 1:
            expression, settings = items[0]
            return [self.simulate(expression, settings)]
        if len(items) > 8:
            raise ValueError("BRAIN multisimulation accepts at most 8 regular alphas")

        payload = [
            {"type": "REGULAR", "settings": settings, "regular": expression}
            for expression, settings in items
        ]
        response = self.session.post(f"{self.base_url}/simulations", json=payload)
        if response.status_code != 201 or "Location" not in response.headers:
            raise RuntimeError(f"multisimulation creation failed: HTTP {response.status_code} {response.text[:200]}")

        parent_location = response.headers["Location"]
        child_locations = self._poll_multisimulation_for_children(parent_location)
        if len(child_locations) < len(items):
            raise RuntimeError(
                f"multisimulation returned {len(child_locations)} children for {len(items)} requested alphas"
            )

        results: List[SimulationResult | SimulationFailure] = []
        for child_location in child_locations[: len(items)]:
            try:
                alpha_id = self._poll_simulation_for_alpha(self._simulation_location(child_location))
                detail = self.get_alpha_detail(alpha_id)
                is_metrics = detail.get("is", {}) if isinstance(detail, dict) else {}
                checks = self.get_submission_check(alpha_id) or self._checks_from_alpha_detail(detail)
                results.append(
                    SimulationResult(
                        alpha_id=alpha_id,
                        metrics=is_metrics,
                        checks=checks,
                        raw={
                            "detail": detail,
                            "multisimulation": parent_location,
                            "child": child_location,
                        },
                    )
                )
            except Exception as exc:
                results.append(
                    SimulationFailure(
                        str(exc),
                        raw={"multisimulation": parent_location, "child": child_location},
                    )
                )
        return results

    def _poll_simulation_for_alpha(self, location: str) -> str:
        for _ in range(self.max_poll_attempts):
            response = self.session.get(self._absolute(location))
            retry_after = response.headers.get("Retry-After")
            if retry_after:
                self.sleep(float(retry_after))
                continue
            data = response.json()
            status = str(data.get("status") or "").upper() if isinstance(data, dict) else ""
            if status in {"ERROR", "FAIL", "FAILED"}:
                detail = data.get("detail") or data.get("message") or data.get("error") or data
                raise RuntimeError(f"simulation failed on platform: {detail}")
            alpha = data.get("alpha") or data.get("alphaId") or data.get("id")
            if isinstance(alpha, dict):
                alpha = alpha.get("id")
            if alpha:
                return str(alpha)
            self.sleep(1)
        raise RuntimeError("simulation polling did not return an alpha id")

    def _poll_multisimulation_for_children(self, location: str) -> List[str]:
        attempts = int(os.environ.get("BRAIN_MULTI_POLL_ATTEMPTS", "200"))
        for _ in range(attempts):
            response = self.session.get(self._absolute(location))
            retry_after = response.headers.get("Retry-After")
            if retry_after:
                self.sleep(float(retry_after))
                continue
            data = response.json()
            status = str(data.get("status") or "").upper() if isinstance(data, dict) else ""
            if status in {"ERROR", "FAIL", "FAILED"}:
                children = data.get("children", []) if isinstance(data, dict) else []
                if not children:
                    detail = data.get("detail") or data.get("message") or data.get("error") or data
                    raise RuntimeError(f"multisimulation failed on platform: {detail}")
            children = data.get("children", []) if isinstance(data, dict) else []
            if children:
                return [str(child) for child in children]
            self.sleep(5)
        raise RuntimeError("multisimulation polling did not return children")

    def get_alpha_detail(self, alpha_id: str) -> Dict[str, Any]:
        last_status = 0
        for attempt in range(1, self.max_poll_attempts + 1):
            response = self.session.get(f"{self.base_url}/alphas/{alpha_id}")
            last_status = int(response.status_code)
            if response.status_code == 200:
                return response.json()
            retry_after = response.headers.get("Retry-After")
            if response.status_code in {404, 425, 429, 500, 502, 503, 504} and attempt < self.max_poll_attempts:
                self.sleep(float(retry_after) if retry_after else 1)
                continue
            break
        raise RuntimeError(f"alpha detail failed: HTTP {last_status}")

    def get_submission_check(self, alpha_id: str) -> Dict[str, Dict[str, Any]]:
        path = f"/alphas/{alpha_id}/check"
        try:
            post = self.session.post(f"{self.base_url}{path}")
            retry_after = post.headers.get("Retry-After")
            if retry_after:
                self.sleep(float(retry_after))
        except Exception:
            pass

        for _ in range(self.max_poll_attempts):
            response = self.session.get(f"{self.base_url}{path}")
            retry_after = response.headers.get("Retry-After")
            if retry_after:
                self.sleep(float(retry_after))
                continue
            data = response.json()
            if isinstance(data, list):
                return self._checks_to_dict(data)
            if isinstance(data, dict):
                if "checks" in data:
                    return self._checks_to_dict(data["checks"])
                if "is" in data and isinstance(data["is"], dict):
                    return self._checks_to_dict(data["is"].get("checks", []))
            return {}
        return {}

    def submit_alpha(self, alpha_id: str, dry_run: bool = True) -> SubmitResult:
        if dry_run:
            return SubmitResult(alpha_id=alpha_id, submitted=False, stage="DRY_RUN", message="auto_submit disabled")

        response = self.session.post(f"{self.base_url}/alphas/{alpha_id}/submit")
        if response.status_code not in (200, 201, 204):
            return SubmitResult(alpha_id=alpha_id, submitted=False, stage="REJECTED", message=f"HTTP {response.status_code}")

        self.sleep(3)
        detail = self.get_alpha_detail(alpha_id)
        stage = str(detail.get("stage") or "")
        date_submitted = detail.get("dateSubmitted")
        if stage == "OS" and date_submitted:
            return SubmitResult(alpha_id=alpha_id, submitted=True, stage=stage, message="verified OS")
        return SubmitResult(alpha_id=alpha_id, submitted=False, stage=stage or "UNKNOWN", message="platform did not verify OS")

    def count_submitted_alphas(self, start_date: str, end_date: str) -> int:
        response = self.session.get(
            f"{self.base_url}/users/self/alphas",
            params={
                "stage": "OS",
                "limit": 100,
                "dateSubmitted>": start_date,
                "dateSubmitted<": end_date,
                "order": "-dateSubmitted",
            },
        )
        if response.status_code != 200:
            raise RuntimeError(f"submitted alpha count failed: HTTP {response.status_code}")
        data = response.json()
        if isinstance(data, dict) and data.get("count") not in (None, ""):
            return int(data.get("count") or 0)
        return len(data.get("results", [])) if isinstance(data, dict) else 0

    def recent_submitted_alphas(self, settings: Dict[str, Any] | None = None, limit: int = 50) -> List[Dict[str, Any]]:
        response = self.session.get(
            f"{self.base_url}/users/self/alphas",
            params={
                "stage": "OS",
                "limit": max(1, min(int(limit), 100)),
                "order": "-dateSubmitted",
            },
        )
        if response.status_code != 200:
            raise RuntimeError(f"recent submitted alpha lookup failed: HTTP {response.status_code}")
        data = response.json()
        results = data.get("results", []) if isinstance(data, dict) else []
        return results if isinstance(results, list) else []

    def get_pyramid_alphas(self, start_date: str | None = None, end_date: str | None = None) -> Dict[str, Any]:
        params = {}
        if start_date:
            params["startDate"] = start_date
        if end_date:
            params["endDate"] = end_date
        response = self.session.get(f"{self.base_url}/users/self/activities/pyramid-alphas", params=params)
        if response.status_code != 200:
            raise RuntimeError(f"pyramid alpha lookup failed: HTTP {response.status_code}")
        data = response.json()
        if not isinstance(data, dict):
            data = {"pyramids": []}
        if params:
            data = dict(data)
            data["query"] = dict(params)
        return data

    def get_pyramid_multipliers(self) -> Dict[str, Any]:
        response = self.session.get(f"{self.base_url}/users/self/activities/pyramid-multipliers")
        if response.status_code != 200:
            raise RuntimeError(f"pyramid multiplier lookup failed: HTTP {response.status_code}")
        data = response.json()
        return data if isinstance(data, dict) else {"pyramids": []}

    def discover_datafields(
        self,
        settings: Dict[str, Any],
        search_terms: List[str] | None = None,
        max_fields: int = 120,
    ) -> List[Dict[str, Any]]:
        scope = {
            "instrumentType": settings.get("instrumentType", "EQUITY"),
            "region": settings.get("region", "USA"),
            "delay": settings.get("delay", 1),
            "universe": settings.get("universe", "TOP3000"),
        }
        rows: List[Dict[str, Any]] = []
        seen = set()
        terms = search_terms if search_terms is not None else [""]
        per_request_limit = min(50, max_fields)
        for term in terms:
            if len(rows) >= max_fields:
                break
            params = dict(scope)
            params.update({"limit": per_request_limit, "offset": 0})
            if term:
                params["search"] = term
            response = self._get_with_rate_limit_retry(f"{self.base_url}/data-fields", params=params)
            if response.status_code != 200:
                raise RuntimeError(f"datafield discovery failed: HTTP {response.status_code} {response.text[:200]}")
            payload = response.json()
            candidates = payload.get("results", []) if isinstance(payload, dict) else []
            for row in candidates:
                if not isinstance(row, dict):
                    continue
                field_id = row.get("id")
                if not field_id or field_id in seen:
                    continue
                seen.add(field_id)
                rows.append(row)
                if len(rows) >= max_fields:
                    break
        return rows

    def _get_with_rate_limit_retry(self, url: str, params: Dict[str, Any]) -> Any:
        attempts = int(os.environ.get("BRAIN_DATAFIELD_RETRIES", "2")) + 1
        for attempt in range(attempts):
            response = self.session.get(url, params=params)
            if response.status_code != 429 or attempt == attempts - 1:
                return response
            retry_after = response.headers.get("Retry-After")
            try:
                wait_seconds = min(float(retry_after), 30.0)
            except (TypeError, ValueError):
                wait_seconds = 5.0
            self.sleep(wait_seconds)
        return response

    def _absolute(self, path_or_url: str) -> str:
        if path_or_url.startswith("http"):
            return path_or_url
        return f"{self.base_url}{path_or_url}"

    def _simulation_location(self, child: str) -> str:
        if child.startswith("http") or child.startswith("/simulations/"):
            return child
        return f"/simulations/{child}"

    def _checks_from_alpha_detail(self, detail: Any) -> Dict[str, Dict[str, Any]]:
        if not isinstance(detail, dict):
            return {}
        checks = self._checks_to_dict(detail.get("tests", {}))
        if checks:
            return checks
        is_payload = detail.get("is")
        if isinstance(is_payload, dict):
            return self._checks_to_dict(is_payload.get("checks", []))
        return {}

    @staticmethod
    def _checks_to_dict(checks: Any) -> Dict[str, Dict[str, Any]]:
        result: Dict[str, Dict[str, Any]] = {}
        if isinstance(checks, dict):
            for name, check in checks.items():
                if isinstance(check, dict):
                    status = str(check.get("status") or check.get("result") or "UNKNOWN").upper()
                    result[str(name)] = {"status": status, "value": check.get("value")}
                    for key in ("limit", "message", "year"):
                        if key in check:
                            result[str(name)][key] = check[key]
                else:
                    result[str(name)] = {"status": str(check).upper(), "value": None}
            return result
        if not isinstance(checks, list):
            return result
        for check in checks:
            if not isinstance(check, dict):
                continue
            name = str(check.get("name") or "UNKNOWN")
            status = str(check.get("status") or check.get("result") or "UNKNOWN").upper()
            result[name] = {"status": status, "value": check.get("value")}
            for key in ("limit", "message", "year"):
                if key in check:
                    result[name][key] = check[key]
        return result


def _datafield_row(field_id: str, category: str, dataset_id: str, description: str, field_type: str) -> Dict[str, Any]:
    return {
        "id": field_id,
        "description": description,
        "dataset": {"id": dataset_id, "name": dataset_id},
        "category": {"id": category.lower(), "name": category},
        "type": field_type,
        "coverage": 1.0,
        "dateCoverage": 1.0,
        "userCount": 0,
        "alphaCount": 0,
    }
