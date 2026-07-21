from __future__ import annotations

import json
import os
import re
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List

from .expression_similarity import expression_structure_key
from .preflight import ALLOWED_OPERATORS
from .scopes import PLATFORM_SCOPE_OPTIONS


DEFAULT_PLAN_BATCH_SIZE = 8
MAX_OPTIMIZE_ROUNDS = 3
MIN_OPTIMIZATION_IMPROVEMENT = 1.20
DEFAULT_REQUIRED_SHARPE = 1.58
DEFAULT_REQUIRED_FITNESS = 1.0
OPTIMIZE_SHARPE_RATIO = 0.50
OPTIMIZE_FITNESS_RATIO = 0.50
SETTING_SWEEP_SHARPE_RATIO = 0.85
SETTING_SWEEP_FITNESS_RATIO = 0.75
DEFAULT_OPTIMIZE_READINESS = 0.45
DEFAULT_SETTING_SWEEP_READINESS = 0.80
QUALITY_COMPONENT_CAP = 1.2
OPTIMIZE_MAX_OPEN_QUALITY_GAPS = 2
OPTIMIZE_MIN_KNOWN_QUALITY_CHECKS = 4
OPTIMIZE_FALLBACK_SHARPE_FLOOR = 1.4
OPTIMIZE_FALLBACK_FITNESS_FLOOR = 0.75
SCOPE_TROUBLE_FAILURE_STREAK = 96
SCOPE_TROUBLE_MIN_SCANNED = 120
ROUTE_STOP_LOSS_MIN_SCANNED = 120
ROUTE_STOP_LOSS_FAILURE_STREAK = 80
ROUTE_STOP_LOSS_SHARPE_RATIO = 0.85
MAX_BATCH_CANDIDATES_PER_STRUCTURE = 2
CHECK_PENALTIES = {
    "HIGH_TURNOVER": 0.85,
    "LOW_TURNOVER": 0.45,
    "CONCENTRATED_WEIGHT": 0.95,
    "IS_LADDER_SHARPE": 0.75,
    "LOW_2Y_SHARPE": 0.65,
    "LOW_SUB_UNIVERSE_SHARPE": 0.55,
    "LOW_ROBUST_UNIVERSE_SHARPE": 0.55,
    "SELF_CORRELATION": 0.8,
    "PROD_CORRELATION": 0.8,
    "PRODUCT_CORRELATION": 0.8,
    "DATA_DIVERSITY": 0.45,
    "REGULAR_SUBMISSION": 0.4,
    "LOW_SHARPE": 0.35,
    "LOW_FITNESS": 0.25,
    "LOW_RETURNS": 0.35,
}
CHECK_PASS_BONUSES = {
    "HIGH_TURNOVER": 0.08,
    "LOW_TURNOVER": 0.06,
    "CONCENTRATED_WEIGHT": 0.10,
    "IS_LADDER_SHARPE": 0.10,
    "LOW_SUB_UNIVERSE_SHARPE": 0.07,
    "LOW_ROBUST_UNIVERSE_SHARPE": 0.07,
    "LOW_RETURNS": 0.05,
}
MANDATORY_SUBMISSION_CHECKS = {
    "SELF_CORRELATION",
    "PROD_CORRELATION",
    "PRODUCT_CORRELATION",
    "DATA_DIVERSITY",
    "REGULAR_SUBMISSION",
}
SUBMISSION_CHECK_ALIASES = (
    {"SELF_CORRELATION"},
    {"PROD_CORRELATION", "PRODUCT_CORRELATION"},
    {"DATA_DIVERSITY"},
    {"REGULAR_SUBMISSION"},
)
SETTING_SWEEP_BLOCKING_FAILURES = {
    "HIGH_TURNOVER",
    "LOW_TURNOVER",
    "CONCENTRATED_WEIGHT",
    "SELF_CORRELATION",
    "PROD_CORRELATION",
    "PRODUCT_CORRELATION",
    "DATA_DIVERSITY",
    "REGULAR_SUBMISSION",
    "D0_SUBMISSION",
}
OPTIMIZATION_QUALITY_CHECKS = {
    "LOW_SHARPE",
    "LOW_FITNESS",
    "LOW_RETURNS",
    "LOW_TURNOVER",
    "HIGH_TURNOVER",
    "CONCENTRATED_WEIGHT",
    "IS_LADDER_SHARPE",
    "LOW_2Y_SHARPE",
    "LOW_SUB_UNIVERSE_SHARPE",
    "LOW_ROBUST_UNIVERSE_SHARPE",
    "ROBUST_UNIVERSE_RETENTION",
    "HT_TURNOVER",
    "HT_HIGH_TURNOVER_RETURNS_RATIO",
    "HT_MAX_TRADE_TURNOVER",
    "HT_MAX_POSITION_TURNOVER",
    "INVESTABLE_HIGH_TURNOVER",
}
OPTIMIZATION_TERMINAL_BLOCKERS = MANDATORY_SUBMISSION_CHECKS | {
    "D0_SUBMISSION",
    "POWERPOOL_CORRELATION",
}
QUALITY_METRIC_REQUIREMENTS = {
    "required_sharpe": ("sharpe", "LOW_SHARPE"),
    "required_fitness": ("fitness", "LOW_FITNESS"),
    "required_returns": ("returns", "LOW_RETURNS"),
}
ACTIONABLE_FAILURE_PRIORITY = [
    "INVALID_EVENT_INPUT_OPERATOR",
    "INVALID_VECTOR_TS_OPERATOR",
    "INVALID_OPERATOR_ARITY",
    "TOO_MANY_OPERATORS",
    "PROFILE_FORBIDDEN_FIELD_FAMILY",
    "PROFILE_REQUIRED_FIELD_FAMILY",
    "TURNOVER_ABOVE_MAX",
    "HIGH_TURNOVER",
    "HT_TURNOVER",
    "HT_HIGH_TURNOVER_RETURNS_RATIO",
    "CONCENTRATED_WEIGHT",
    "LOW_TURNOVER",
]
THRESHOLD_COPY_KEYS = (
    "required_returns",
    "turnover_min",
    "turnover_max",
    "preferred_turnover_max",
    "subuniverse_ratio",
    "subuniverse_formula",
    "delay1_sharpe_check",
    "max_trade_required",
    "capacity_sensitive",
    "extra_checks",
)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SCOPE_QUALITY_THRESHOLDS_FILE = PROJECT_ROOT / "config" / "scope_quality_thresholds.json"

_GROUP_TOKENS = {"market", "sector", "industry", "subindustry", "country", "exchange"}
_FAMILY_IGNORE_FIELDS = {
    "cap",
    "close",
    "high",
    "low",
    "open",
    "returns",
    "trade",
    "volume",
    "vwap",
}
_NON_FIELD_TOKENS = {
    "abs",
    "add",
    "and",
    "arc",
    "arg",
    "backfill",
    "bucket",
    "cap",
    "close",
    "densify",
    "divide",
    "filter",
    "group",
    "high",
    "if",
    "is",
    "log",
    "low",
    "max",
    "mean",
    "min",
    "multiply",
    "nan",
    "normalize",
    "not",
    "open",
    "or",
    "rank",
    "returns",
    "scale",
    "sign",
    "std",
    "stddev",
    "subtract",
    "sum",
    "trade",
    "ts",
    "vec",
    "volume",
    "vwap",
    "winsorize",
    "zscore",
    *_GROUP_TOKENS,
}
_OPERATOR_PREFIXES = ("group_", "ts_", "vec_")


def analyze_research_history(research_context: Dict[str, Any]) -> Dict[str, Any]:
    target_settings = research_context.get("target_settings") if isinstance(research_context.get("target_settings"), dict) else {}
    candidates = _candidate_pool(research_context, target_settings)
    best = _best_candidate(candidates)
    failure_reasons = _failure_reasons(candidates)
    field_stats = _field_stats(candidates, research_context)
    field_family_stats = _field_family_stats(candidates, research_context)
    observed_quality_thresholds = _observed_quality_thresholds(candidates)
    optimization_state = _optimization_state(research_context.get("recent_experiment_plans"), best, target_settings)
    history_memory = research_context.get("history_memory") if isinstance(research_context.get("history_memory"), dict) else {}
    mechanism_memory = _mechanism_memory_from_history(history_memory)
    scope_health = history_memory.get("scope_health") if isinstance(history_memory.get("scope_health"), dict) else {}
    active_run_history = (
        research_context.get("active_run_history_memory")
        if isinstance(research_context.get("active_run_history_memory"), dict)
        else {}
    )
    route_scope_health = (
        active_run_history.get("scope_health")
        if isinstance(active_run_history.get("scope_health"), dict)
        else scope_health
    )
    candidate_queue_counts = _candidate_queue_counts(research_context)
    quality_thresholds = _quality_thresholds(
        best or {},
        target_settings,
        {"observed_quality_thresholds": observed_quality_thresholds},
    )
    route_efficiency = _route_efficiency(route_scope_health, candidate_queue_counts, quality_thresholds)
    structure_diversity_control = _structure_diversity_control(candidates, history_memory)

    submitted_avoid_fields = _submitted_avoid_fields(research_context)
    promising_fields = [
        field
        for field in _promising_fields(best, field_stats)
        if field not in submitted_avoid_fields
    ]
    weak_fields = [
        field
        for field, stats in sorted(
            field_stats.items(),
            key=lambda item: (float(item[1].get("avg_sharpe", 0.0)), -int(item[1].get("count", 0)), item[0]),
        )
        if int(stats.get("count", 0)) >= 2 and float(stats.get("avg_sharpe", 0.0)) <= 0.0
    ][:12]
    family_diversity_control = _family_diversity_control(best, field_family_stats)
    lit_tower_avoidance = (
        research_context.get("lit_tower_avoidance") if isinstance(research_context.get("lit_tower_avoidance"), dict) else {}
    )
    field_scout = research_context.get("field_scout") if isinstance(research_context.get("field_scout"), dict) else {}

    return {
        "candidate_count": len(candidates),
        "best_candidate": best or {},
        "candidate_queue_counts": candidate_queue_counts,
        "failure_reasons": dict(failure_reasons.most_common(12)),
        "promising_fields": promising_fields,
        "weak_fields": weak_fields,
        "field_stats": dict(list(field_stats.items())[:40]),
        "field_family_stats": dict(list(field_family_stats.items())[:20]),
        "family_diversity_control": family_diversity_control,
        "submitted_field_avoidance": research_context.get("submitted_field_avoidance") or {},
        "submitted_avoid_fields": submitted_avoid_fields,
        "lit_tower_avoidance": lit_tower_avoidance,
        "field_scout": field_scout,
        "mechanism_memory": mechanism_memory,
        "scope_health": scope_health,
        "route_efficiency": route_efficiency,
        "structure_diversity_control": structure_diversity_control,
        "observed_quality_thresholds": observed_quality_thresholds,
        "optimization_state": optimization_state,
    }


def build_experiment_plan(
    analysis: Dict[str, Any],
    target_settings: Dict[str, Any],
    batch_size: int = DEFAULT_PLAN_BATCH_SIZE,
) -> Dict[str, Any]:
    best = analysis.get("best_candidate") if isinstance(analysis.get("best_candidate"), dict) else {}
    failure_reasons = list((analysis.get("failure_reasons") or {}).keys())
    promising_fields = [str(field) for field in analysis.get("promising_fields") or []]
    weak_fields = [str(field) for field in analysis.get("weak_fields") or []]
    submitted_avoid_fields = [str(field) for field in analysis.get("submitted_avoid_fields") or []]
    submitted_avoidance = (
        analysis.get("submitted_field_avoidance") if isinstance(analysis.get("submitted_field_avoidance"), dict) else {}
    )
    lit_tower_avoidance = (
        analysis.get("lit_tower_avoidance") if isinstance(analysis.get("lit_tower_avoidance"), dict) else {}
    )
    lit_tower_names = _lit_tower_names(lit_tower_avoidance)
    family_diversity_control = (
        analysis.get("family_diversity_control") if isinstance(analysis.get("family_diversity_control"), dict) else {}
    )
    route_stop_loss = analysis.get("route_efficiency") if isinstance(analysis.get("route_efficiency"), dict) else {}
    structure_diversity_control = (
        analysis.get("structure_diversity_control") if isinstance(analysis.get("structure_diversity_control"), dict) else {}
    )
    field_scout = analysis.get("field_scout") if isinstance(analysis.get("field_scout"), dict) else {}
    field_scout_for_plan = _compact_field_scout_for_plan(field_scout)
    optimization_state = analysis.get("optimization_state") if isinstance(analysis.get("optimization_state"), dict) else {}
    sharpe = _float(best.get("sharpe"))
    fitness = _float(best.get("fitness"))
    score = _candidate_score(best)
    quality_thresholds = _quality_thresholds(best, target_settings, analysis)
    quality_components = best.get("quality_components") if isinstance(best.get("quality_components"), dict) else {}
    readiness_score = _float(quality_components.get("readiness_score"))
    submission_score = _float(quality_components.get("submission_score"))
    optimization_gap_summary = _optimization_gap_summary(best, quality_thresholds)
    scope_trouble = _scope_trouble_state(analysis, quality_thresholds)
    mechanism_transfer = _mechanism_transfer_plan(analysis, scope_trouble)
    production_rescue = _production_rescue_policy(route_stop_loss, scope_trouble, target_settings)
    hard_lit_tower_names = [] if production_rescue.get("active") else lit_tower_names
    tower_objective = (
        _production_rescue_objective(production_rescue, lit_tower_avoidance)
        if production_rescue.get("active")
        else _lit_tower_objective(lit_tower_avoidance)
    )

    def with_quality_budget(plan: Dict[str, Any]) -> Dict[str, Any]:
        budget = _quality_budget_for_plan(
            str(plan.get("mode") or ""),
            int(plan.get("batch_size") or batch_size),
            field_scout,
            best,
            production_rescue=production_rescue,
            structure_diversity_control=structure_diversity_control,
        )
        plan["quality_budget"] = budget["quality_budget"]
        plan["probe_recommendations"] = budget["probe_recommendations"]
        plan_production_rescue = dict(production_rescue)
        if (
            plan_production_rescue.get("active")
            and not budget["probe_recommendations"]
            and not budget["quality_budget"].get("slots", {}).get("probe_new_fields")
        ):
            plan_production_rescue.update(
                {
                    "active": False,
                    "reason": "no_safe_probe_recommendations",
                    "previous_reason": production_rescue.get("reason"),
                }
            )
            disabled_note = (
                " Production rescue is disabled for this cycle because no safe probe field or template is "
                "available; continue fresh exploration without production-rescue motifs."
            )
            objective = str(plan.get("objective") or "")
            if tower_objective and tower_objective in objective:
                plan["objective"] = objective.replace(tower_objective, _lit_tower_objective(lit_tower_avoidance) + disabled_note)
            elif objective:
                plan["objective"] = objective + disabled_note
        plan["production_rescue"] = plan_production_rescue
        return plan

    if scope_trouble.get("active") and mechanism_transfer:
        return with_quality_budget({
            "mode": "explore_new_family",
            "target_candidate_id": None,
            "target_expression": "",
            "objective": (
                "Scope trouble mode is active after a long failed streak. Do not keep optimizing the latest weak "
                "or blocked expression. Transfer mechanisms from historical high-signal archetypes onto allowed, "
                "non-submitted, non-auxiliary-primary fields."
                + _mechanism_transfer_objective(mechanism_transfer)
                + _family_diversity_objective(family_diversity_control)
                + _submitted_avoidance_objective(submitted_avoidance)
                + tower_objective
                + _route_stop_loss_objective(route_stop_loss)
                + _structure_diversity_objective(structure_diversity_control)
            ),
            "keep": [],
            "change": [
                "migrate historical winning mechanisms to fresh primary fields",
                "avoid exact expressions and forbidden fields from mechanism_transfer",
                "prefer smoother turnover-aware structures",
                "split profiles across different transferable mechanisms",
                "replace overused formula skeletons",
            ],
            "avoid": _avoid_list(
                failure_reasons,
                hard_lit_tower_names + submitted_avoid_fields + weak_fields + mechanism_transfer.get("forbidden_fields", []),
            ),
            "family_diversity_control": family_diversity_control,
            "submitted_field_avoidance": submitted_avoidance,
            "lit_tower_avoidance": lit_tower_avoidance,
            "field_scout": field_scout_for_plan,
            "scope_trouble": scope_trouble,
            "mechanism_transfer": mechanism_transfer,
            "route_stop_loss": route_stop_loss,
            "structure_diversity_control": structure_diversity_control,
            "batch_size": int(batch_size),
            "target_settings": dict(target_settings),
            "quality_thresholds": quality_thresholds,
            "optimization_gap_summary": optimization_gap_summary,
        })

    if best and optimization_state.get("limit_exhausted"):
        return with_quality_budget({
            "mode": "explore_new_family",
            "target_candidate_id": None,
            "target_expression": "",
            "abandoned_target_id": optimization_state.get("optimization_anchor_id"),
            "abandon_reason": "NO_20_PERCENT_IMPROVEMENT_AFTER_3_ROUNDS",
            "objective": (
                "The previous optimization target used three rounds without at least 20 percent score improvement. "
                "Abandon that family and explore structurally different fields."
                + _submitted_avoidance_objective(submitted_avoidance)
                + tower_objective
                + _route_stop_loss_objective(route_stop_loss)
                + _structure_diversity_objective(structure_diversity_control)
            ),
            "keep": [],
            "change": [
                "switch field family",
                "avoid local variants of the abandoned expression",
                "test simpler field-native mechanisms",
                "replace overused formula skeletons",
            ],
            "avoid": _avoid_list(failure_reasons, hard_lit_tower_names + submitted_avoid_fields + weak_fields + list(best.get("fields") or [])),
            "submitted_field_avoidance": submitted_avoidance,
            "lit_tower_avoidance": lit_tower_avoidance,
            "field_scout": field_scout_for_plan,
            "scope_trouble": scope_trouble,
            "mechanism_transfer": mechanism_transfer,
            "route_stop_loss": route_stop_loss,
            "structure_diversity_control": structure_diversity_control,
            "batch_size": int(batch_size),
            "target_settings": dict(target_settings),
            "quality_thresholds": quality_thresholds,
            "optimization_gap_summary": optimization_gap_summary,
        })

    if (
        best
        and quality_thresholds.get("trusted")
        and submission_score >= _float(quality_thresholds.get("setting_sweep_readiness"), DEFAULT_SETTING_SWEEP_READINESS)
        and not _has_setting_sweep_blockers(best)
    ):
        optimization = _next_optimization_fields(best, optimization_state, score)
        return with_quality_budget({
            "mode": "setting_sweep",
            "target_candidate_id": best.get("id"),
            "optimization_anchor_id": optimization["anchor_id"],
            "optimize_round": optimization["round"],
            "baseline_score": optimization["baseline_score"],
            "baseline_sharpe": optimization["baseline_sharpe"],
            "baseline_fitness": optimization["baseline_fitness"],
            "current_score": round(score, 6),
            "readiness_score": round(readiness_score, 6),
            "submission_score": round(submission_score, 6),
            "target_expression": best.get("expression"),
            "objective": (
                "The expression is near submission thresholds. Keep the expression fixed and test settings "
                "variants before spending more AI tokens on formula changes."
                + _submitted_avoidance_objective(submitted_avoidance)
                + tower_objective
            ),
            "keep": [field for field in (promising_fields[:8] or best.get("fields", [])[:8]) if field not in submitted_avoid_fields],
            "change": ["neutralization", "decay", "truncation"],
            "avoid": _avoid_list(failure_reasons, hard_lit_tower_names + submitted_avoid_fields + weak_fields),
            "setting_variants": _setting_variants(target_settings, batch_size),
            "submitted_field_avoidance": submitted_avoidance,
            "lit_tower_avoidance": lit_tower_avoidance,
            "field_scout": field_scout_for_plan,
            "scope_trouble": scope_trouble,
            "mechanism_transfer": mechanism_transfer,
            "route_stop_loss": route_stop_loss,
            "structure_diversity_control": structure_diversity_control,
            "batch_size": int(batch_size),
            "target_settings": dict(target_settings),
            "quality_thresholds": quality_thresholds,
            "optimization_gap_summary": optimization_gap_summary,
        })

    if (
        best
        and quality_thresholds.get("trusted")
        and optimization_gap_summary.get("eligible")
    ):
        optimization = _next_optimization_fields(best, optimization_state, score)
        return with_quality_budget({
            "mode": "optimize_best",
            "target_candidate_id": best.get("id"),
            "optimization_anchor_id": optimization["anchor_id"],
            "optimize_round": optimization["round"],
            "baseline_score": optimization["baseline_score"],
            "baseline_sharpe": optimization["baseline_sharpe"],
            "baseline_fitness": optimization["baseline_fitness"],
            "current_score": round(score, 6),
            "readiness_score": round(readiness_score, 6),
            "submission_score": round(submission_score, 6),
            "target_expression": best.get("expression"),
            "objective": (
                "Generate eight controlled variants around the best recent candidate. "
                "Improve Sharpe, Fitness, and ladder robustness without drifting into an unrelated field family."
                + _family_diversity_objective(family_diversity_control)
                + _submitted_avoidance_objective(submitted_avoidance)
                + tower_objective
            ),
            "keep": _diversified_keep_fields(
                best,
                analysis.get("field_stats") or {},
                family_diversity_control,
                submitted_avoid_fields,
                8,
            ),
            "change": [
                "time-series windows",
                "signal sign where the prior run was weak",
                "winsorize/std settings",
                "rank versus ts_rank placement",
                "normalization by cap or volatility",
                "one secondary confirmation leg",
            ],
            "avoid": _avoid_list(failure_reasons, hard_lit_tower_names + submitted_avoid_fields + weak_fields),
            "family_diversity_control": family_diversity_control,
            "submitted_field_avoidance": submitted_avoidance,
            "lit_tower_avoidance": lit_tower_avoidance,
            "field_scout": field_scout_for_plan,
            "scope_trouble": scope_trouble,
            "mechanism_transfer": mechanism_transfer,
            "route_stop_loss": route_stop_loss,
            "structure_diversity_control": structure_diversity_control,
            "batch_size": int(batch_size),
            "target_settings": dict(target_settings),
            "quality_thresholds": quality_thresholds,
            "optimization_gap_summary": optimization_gap_summary,
        })

    if analysis.get("candidate_count"):
        threshold_note = (
            " Official quality thresholds for this scope are not observed yet; keep exploring until live "
            "simulation checks provide LOW_SHARPE/LOW_FITNESS limits."
            if not quality_thresholds.get("trusted")
            else ""
        )
        abandon_weak_anchor = _should_abandon_weak_anchor(scope_trouble, route_stop_loss, optimization_gap_summary)
        keep_fields = (
            []
            if abandon_weak_anchor
            else _diversified_keep_fields(
                best,
                analysis.get("field_stats") or {},
                family_diversity_control,
                submitted_avoid_fields,
                5,
            )
        )
        avoid_fields = hard_lit_tower_names + submitted_avoid_fields + weak_fields
        if abandon_weak_anchor:
            avoid_fields += [str(field) for field in best.get("fields") or []]
        return with_quality_budget({
            "mode": "explore_new_family",
            "target_candidate_id": None,
            "target_expression": "",
            "objective": (
                "Recent candidates are far from submission thresholds. Generate a fresh batch from different "
                "field families while avoiding repeated weak structures."
                + threshold_note
                + _family_diversity_objective(family_diversity_control)
                + _submitted_avoidance_objective(submitted_avoidance)
                + tower_objective
                + _route_stop_loss_objective(route_stop_loss)
                + _structure_diversity_objective(structure_diversity_control)
            ),
            "keep": keep_fields,
            "change": [
                "switch field family",
                "test simpler field-native mechanisms",
                "use one clear economic hypothesis per candidate",
                "replace overused formula skeletons",
            ],
            "avoid": _avoid_list(failure_reasons, avoid_fields),
            "family_diversity_control": family_diversity_control,
            "submitted_field_avoidance": submitted_avoidance,
            "lit_tower_avoidance": lit_tower_avoidance,
            "field_scout": field_scout_for_plan,
            "scope_trouble": scope_trouble,
            "mechanism_transfer": mechanism_transfer,
            "route_stop_loss": route_stop_loss,
            "structure_diversity_control": structure_diversity_control,
            "batch_size": int(batch_size),
            "target_settings": dict(target_settings),
            "quality_thresholds": quality_thresholds,
            "optimization_gap_summary": optimization_gap_summary,
        })

    return with_quality_budget({
        "mode": "explore",
        "target_candidate_id": None,
        "target_expression": "",
        "objective": (
            "No usable local evidence yet. Generate diverse research-grade candidates from verified fields."
            + tower_objective
            + _structure_diversity_objective(structure_diversity_control)
        ),
        "keep": [],
        "change": ["cover multiple field families", "use economically grounded windows", "avoid trivial price-volume ranks"],
        "avoid": _avoid_list(
            ["trivial price-volume-only formulas", "unverified fields", "unverified operators"],
            hard_lit_tower_names + submitted_avoid_fields,
        ),
        "family_diversity_control": family_diversity_control,
        "submitted_field_avoidance": submitted_avoidance,
        "lit_tower_avoidance": lit_tower_avoidance,
        "field_scout": field_scout_for_plan,
        "scope_trouble": scope_trouble,
        "mechanism_transfer": mechanism_transfer,
        "route_stop_loss": route_stop_loss,
        "structure_diversity_control": structure_diversity_control,
        "batch_size": int(batch_size),
        "target_settings": dict(target_settings),
        "quality_thresholds": quality_thresholds,
        "optimization_gap_summary": optimization_gap_summary,
    })


HIGH_TURNOVER_PROBE_CATEGORIES = {
    "earnings",
    "insiders",
    "news",
    "sentiment",
    "shortinterest",
    "short interest",
    "socialmedia",
    "social media",
}

HIGH_TURNOVER_PROBE_DATASET_PREFIXES = (
    "earn",
    "ern",
    "insd",
    "insider",
    "news",
    "nws",
    "sentiment",
    "snt",
    "short",
    "shrt",
    "social",
    "scl",
)


def _quality_budget_for_plan(
    mode: str,
    batch_size: int,
    field_scout: Dict[str, Any],
    best: Dict[str, Any],
    production_rescue: Dict[str, Any] | None = None,
    structure_diversity_control: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    batch = max(0, int(batch_size or 0))
    rescue_active = bool(isinstance(production_rescue, dict) and production_rescue.get("active"))
    rows = _quality_budget_field_rows(field_scout, include_lit=rescue_active)
    exploit_fields = [row["field"] for row in rows if _field_has_positive_evidence(row)]
    blocked_structure_keys = _overused_structure_keys(structure_diversity_control) if rescue_active else set()
    probe_rows = [
        row
        for row in rows
        if row.get("field") not in set(exploit_fields)
        and _field_is_probe_worthy(row, include_lit=rescue_active)
        and (not rescue_active or _field_is_production_rescue_probe_candidate(row))
    ]
    probe_recommendations = []
    for row in probe_rows[: max(2, batch)]:
        recommendation = _probe_recommendation(
            row,
            production_rescue=rescue_active,
            blocked_structure_keys=blocked_structure_keys,
        )
        if recommendation.get("templates"):
            probe_recommendations.append(recommendation)
    total_probe_templates = sum(
        len(recommendation.get("templates") or [])
        for recommendation in probe_recommendations
        if isinstance(recommendation, dict)
    )

    if rescue_active:
        if probe_recommendations:
            probe_slots = min(batch, total_probe_templates)
            if not exploit_fields:
                probe_slots = min(probe_slots, 2)
            slots = {"probe_new_fields": probe_slots}
        else:
            slots = {"broad_explore": batch}
        rationale = (
            "Production rescue is active after repeated low-quality scoped batches; use a small probe batch until "
            "a field or structure earns optimize-ready evidence."
            if probe_recommendations
            else "Production rescue has no safe probe field or template; disable the rescue gate and continue fresh exploration."
        )
    elif mode == "setting_sweep":
        slots = {"setting_sweep": batch}
        rationale = "Candidate is close enough for settings-first testing; do not dilute the batch with fresh fields."
    elif mode == "optimize_best":
        probe_slots = min(1, len(probe_recommendations), max(0, batch - 2))
        broad_slots = 1 if batch - probe_slots > 2 else 0
        slots = {
            "optimize_anchor": max(0, batch - probe_slots - broad_slots),
            "probe_new_fields": probe_slots,
            "broad_explore": broad_slots,
        }
        rationale = "Keep most slots on the near-threshold anchor while reserving a small probe lane."
    elif exploit_fields:
        probe_slots = min(2, len(probe_recommendations), max(0, batch - 2))
        broad_slots = 1 if batch - probe_slots > 1 else 0
        slots = {
            "exploit_positive_evidence": max(0, batch - probe_slots - broad_slots),
            "probe_new_fields": probe_slots,
            "broad_explore": broad_slots,
        }
        rationale = "Spend most slots on fields with positive local evidence, then probe a small number of fresh fields."
    elif not probe_recommendations:
        slots = {"broad_explore": batch}
        rationale = "No safe probe fields are currently available; avoid impossible probe allocation."
    else:
        slots = {"broad_explore": batch}
        rationale = (
            "No positive field evidence is available; pass fresh-field probe recommendations to the AI as "
            "research guidance without spending normal explore slots on automatic probe simulations."
        )

    return {
        "quality_budget": {
            "priority": "production_first",
            "batch_size": batch,
            "slots": slots,
            "exploit_fields": exploit_fields[: max(1, batch)],
            "policy": (
                "Field diversity and unlit towers are tie-breakers after quality. Do not spend all slots on one "
                "dataset, one field family, or raw high-turnover alternative signals."
            ),
            "rationale": rationale,
        },
        "probe_recommendations": probe_recommendations,
    }


def _quality_budget_field_rows(field_scout: Dict[str, Any], include_lit: bool = False) -> List[Dict[str, Any]]:
    if not isinstance(field_scout, dict):
        return []
    rows = field_scout.get("top_fields") if include_lit else field_scout.get("top_primary_fields")
    if not isinstance(rows, list):
        rows = field_scout.get("top_primary_fields") if include_lit else field_scout.get("top_fields")
    if not isinstance(rows, list):
        return []
    result = [row for row in rows if isinstance(row, dict) and str(row.get("field") or "").strip()]
    result.sort(
        key=lambda row: (
            0 if _field_has_positive_evidence(row) else 1,
            0 if row.get("primary_policy") == "prefer_primary" else 1,
            1 if include_lit and str(row.get("tower_status") or "").strip().lower() == "lit" else 0,
            _production_rescue_field_priority(row) if include_lit else 0,
            int(row.get("explored_count") or 0),
            -_float(row.get("score")),
            str(row.get("field") or ""),
        )
    )
    return result


def _production_rescue_field_priority(row: Dict[str, Any]) -> int:
    category = str(row.get("category") or "").strip().upper()
    dataset_id = str(row.get("dataset_id") or row.get("datasetId") or "").strip().lower()
    field = str(row.get("field") or "").strip().lower()
    text = f"{category} {dataset_id} {field}"
    if category == "OTHER" or dataset_id.startswith("oth") or "other" in text:
        return 0
    if category in {"FUNDAMENTAL", "FUNDAMENTALS"} or dataset_id.startswith("fnd"):
        return 1
    if category == "OPTION" or "option" in text:
        return 2
    if category == "ANALYST" or dataset_id.startswith("anl"):
        return 3
    if category in {"NEWS", "SENTIMENT"} or dataset_id.startswith(("nws", "snt")):
        return 4
    if category == "MODEL" or dataset_id.startswith("mdl") or "model" in text:
        return 8
    if category == "PV" or "price" in text or "volume" in text:
        return 9
    return 5


def _field_is_production_rescue_probe_candidate(row: Dict[str, Any]) -> bool:
    if _field_has_positive_evidence(row):
        return True
    return not _field_is_default_saturated_probe_category(row)


def _field_is_default_saturated_probe_category(row: Dict[str, Any]) -> bool:
    category = str(row.get("category") or "").strip().upper()
    dataset_id = str(row.get("dataset_id") or row.get("datasetId") or "").strip().lower()
    field = str(row.get("field") or "").strip().lower()
    text = f"{category} {dataset_id} {field}"
    return (
        category == "MODEL"
        or category == "PV"
        or category == "PRICE VOLUME"
        or dataset_id.startswith(("mdl", "pv"))
        or "model" in text
        or "price volume" in text
    )


def _field_has_positive_evidence(row: Dict[str, Any]) -> bool:
    if int(row.get("explored_count") or 0) <= 0:
        return False
    return max(_float(row.get("best_sharpe")), _float(row.get("avg_sharpe"))) >= 1.0 or max(
        _float(row.get("best_fitness")), _float(row.get("avg_fitness"))
    ) >= 0.35


def _field_is_probe_worthy(row: Dict[str, Any], include_lit: bool = False) -> bool:
    if include_lit and _field_is_event_input_restricted(row):
        return False
    lit_soft_allowed = (
        include_lit
        and str(row.get("tower_status") or "").strip().lower() == "lit"
        and not row.get("field_reason")
        and not row.get("dataset_reason")
        and not row.get("metadata_reason")
    )
    if row.get("primary_policy") == "avoid_primary" and not lit_soft_allowed:
        return False
    if row.get("field_reason") or row.get("dataset_reason") or row.get("metadata_reason"):
        return False
    failure_rate = _float(row.get("failed_count")) / max(1.0, _float(row.get("explored_count")))
    dataset_failure_rate = _float(row.get("dataset_failed_count")) / max(1.0, _float(row.get("dataset_count")))
    if int(row.get("explored_count") or 0) >= 2 and failure_rate >= 0.75 and not _field_has_positive_evidence(row):
        return False
    if int(row.get("dataset_count") or 0) >= 5 and dataset_failure_rate >= 0.75 and not _field_has_positive_evidence(row):
        return False
    return True


def _field_is_event_input_restricted(row: Dict[str, Any]) -> bool:
    category = str(row.get("category") or "").strip().lower()
    dataset_id = str(row.get("dataset_id") or row.get("datasetId") or "").strip().lower()
    dataset_name = str(row.get("dataset_name") or row.get("datasetName") or "").strip().lower()
    field = str(row.get("field") or "").strip().lower()
    return (
        "news" in category
        or "event" in category
        or dataset_id.startswith(("news", "nws"))
        or "news" in dataset_name
        or "event" in dataset_name
        or field.startswith(("news", "nws"))
    )


def _overused_structure_keys(structure_diversity_control: Dict[str, Any] | None) -> set[str]:
    if not isinstance(structure_diversity_control, dict):
        return set()
    rows = structure_diversity_control.get("overused_structures")
    if not isinstance(rows, list):
        return set()
    result: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            continue
        key = str(row.get("structure_key") or "").strip()
        if key:
            result.add(key)
    return result


def _probe_recommendation(
    row: Dict[str, Any],
    production_rescue: bool = False,
    blocked_structure_keys: set[str] | None = None,
) -> Dict[str, Any]:
    field = str(row.get("field") or "").strip()
    field_type = str(row.get("type") or "").upper()
    value = f"vec_avg({field})" if field_type == "VECTOR" else field
    stable_required = _field_needs_stabilized_probe(row)
    stable_value = f"winsorize(ts_backfill({value}, 120), std=4)"
    if production_rescue or stable_required:
        stabilizing_templates = [
            f"group_rank(ts_rank({stable_value}, 63), industry)",
            f"group_rank(ts_rank(divide({stable_value}, cap), 63), industry)",
            f"rank(ts_decay_linear(ts_backfill({value}, 120), 20))",
            f"rank(multiply(-1, ts_rank({stable_value}, 33)))",
        ]
    else:
        stabilizing_templates = [
            f"rank(ts_mean({value}, 20))",
            f"rank(ts_rank({value}, 60))",
            f"group_rank(ts_mean({value}, 60), industry)",
            f"rank(multiply(-1, ts_mean({value}, 20)))",
            f"rank(ts_decay_linear({value}, 20))",
        ]
    if production_rescue and blocked_structure_keys:
        stabilizing_templates = [
            template
            for template in stabilizing_templates
            if expression_structure_key(template) not in blocked_structure_keys
        ]
    return {
        "field": field,
        "type": row.get("type"),
        "dataset_id": row.get("dataset_id"),
        "category": row.get("category"),
        "score": row.get("score"),
        "route": "production_rescue_probe" if production_rescue else "standardized_probe",
        "stabilization_required": stable_required,
        "templates": stabilizing_templates[:4] if stable_required else stabilizing_templates[:3],
        "rationale": (
            "Probe direction and smoothing before scale-up; treat unlit tower status as a bonus only after basic "
            "quality and turnover behavior are measured."
        ),
    }


def _field_needs_stabilized_probe(row: Dict[str, Any]) -> bool:
    category = str(row.get("category") or "").strip().lower()
    dataset_id = str(row.get("dataset_id") or row.get("datasetId") or "").strip().lower()
    dataset_name = str(row.get("dataset_name") or row.get("datasetName") or "").strip().lower()
    return (
        category in HIGH_TURNOVER_PROBE_CATEGORIES
        or dataset_id.startswith(HIGH_TURNOVER_PROBE_DATASET_PREFIXES)
        or dataset_name.startswith(HIGH_TURNOVER_PROBE_DATASET_PREFIXES)
    )


def _compact_field_scout_for_plan(field_scout: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(field_scout, dict):
        return {}
    compact: Dict[str, Any] = {}
    for key in (
        "active",
        "status",
        "top_field_count",
        "top_primary_field_count",
        "policy",
        "scoring",
    ):
        if key in field_scout:
            compact[key] = field_scout[key]
    top_fields = field_scout.get("top_fields")
    if isinstance(top_fields, list):
        keep_keys = (
            "field",
            "score",
            "type",
            "dataset_id",
            "category",
            "coverage",
            "userCount",
            "alphaCount",
            "pyramidMultiplier",
            "explored_count",
            "failed_count",
            "dataset_count",
            "dataset_failed_count",
            "best_sharpe",
            "best_fitness",
            "field_reason",
            "dataset_reason",
            "tower_status",
            "primary_policy",
            "usage_constraints",
        )
        compact["top_fields"] = [
            {key: row.get(key) for key in keep_keys if key in row}
            for row in top_fields[:30]
            if isinstance(row, dict)
        ]
    top_primary_fields = field_scout.get("top_primary_fields")
    if isinstance(top_primary_fields, list):
        keep_keys = (
            "field",
            "score",
            "type",
            "dataset_id",
            "category",
            "coverage",
            "userCount",
            "alphaCount",
            "pyramidMultiplier",
            "explored_count",
            "failed_count",
            "dataset_count",
            "dataset_failed_count",
            "best_sharpe",
            "best_fitness",
            "field_reason",
            "dataset_reason",
            "tower_status",
            "primary_policy",
            "usage_constraints",
        )
        compact["top_primary_fields"] = [
            {key: row.get(key) for key in keep_keys if key in row}
            for row in top_primary_fields[:30]
            if isinstance(row, dict)
        ]
    buckets = field_scout.get("buckets")
    if isinstance(buckets, list):
        compact["buckets"] = [
            {
                "name": bucket.get("name"),
                "fields": [str(field) for field in (bucket.get("fields") or [])[:12]],
                "rationale": bucket.get("rationale"),
            }
            for bucket in buckets[:6]
            if isinstance(bucket, dict)
        ]
    return compact


def _candidate_pool(research_context: Dict[str, Any], target_settings: Dict[str, Any]) -> List[Dict[str, Any]]:
    queued = research_context.get("candidate_queues")
    submitted_avoid_fields = set(_submitted_avoid_fields(research_context))
    if isinstance(queued, dict):
        pool: List[Dict[str, Any]] = []
        for key in ("watchlist", "optimize", "trash", "abandoned"):
            rows = queued.get(key)
            if isinstance(rows, list):
                pool.extend(
                    row
                    for row in rows
                    if isinstance(row, dict)
                    and _scope_matches(row.get("settings"), target_settings)
                    and not _candidate_uses_fields(row, submitted_avoid_fields)
                )
        if pool:
            return pool

    pool: List[Dict[str, Any]] = []
    for key in ("recent_pending", "recent_failures"):
        rows = research_context.get(key)
        if isinstance(rows, list):
            pool.extend(
                row
                for row in rows
                if isinstance(row, dict)
                and _scope_matches(row.get("settings"), target_settings)
                and not _candidate_uses_fields(row, submitted_avoid_fields)
            )
    return pool


def _candidate_queue_counts(research_context: Dict[str, Any]) -> Dict[str, int]:
    active_run_queues = research_context.get("active_run_candidate_queues")
    queues = active_run_queues if isinstance(active_run_queues, dict) else research_context.get("candidate_queues")
    if not isinstance(queues, dict):
        return {}
    counts = queues.get("counts")
    if isinstance(counts, dict):
        return {str(key): int(value) for key, value in counts.items() if _is_int_like(value)}
    result: Dict[str, int] = {}
    for key in ("submitable", "watchlist", "optimize", "trash", "abandoned"):
        rows = queues.get(key)
        result[key] = len(rows) if isinstance(rows, list) else 0
    result["total"] = sum(result.values())
    return result


def _is_int_like(value: Any) -> bool:
    try:
        int(value)
    except (TypeError, ValueError):
        return False
    return True


def _optimization_state(plans: Any, best: Dict[str, Any], target_settings: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(plans, list):
        return {}
    lineage_ids = _best_optimization_lineage(best)
    relevant = [
        plan
        for plan in plans
        if isinstance(plan, dict)
        and str(plan.get("mode") or "") in {"optimize_best", "setting_sweep"}
        and (
            _scope_matches(plan.get("target_settings"), target_settings)
            or (
                not isinstance(plan.get("target_settings"), dict)
                and str(plan.get("optimization_anchor_id") or plan.get("target_candidate_id") or "") == str(best.get("id") or "")
            )
        )
        and _plan_matches_lineage(plan, lineage_ids)
    ]
    if not relevant:
        return {}

    latest = relevant[0]
    anchor_id = latest.get("optimization_anchor_id") or latest.get("target_candidate_id")
    if anchor_id in (None, ""):
        return {}

    rounds = 0
    for plan in relevant:
        plan_anchor = plan.get("optimization_anchor_id") or plan.get("target_candidate_id")
        if str(plan_anchor) != str(anchor_id):
            break
        rounds += 1

    baseline_score = _float(latest.get("baseline_score")) or _candidate_score(best)
    baseline_sharpe = _float(latest.get("baseline_sharpe")) or _float(best.get("sharpe"))
    baseline_fitness = _float(latest.get("baseline_fitness")) or _float(best.get("fitness"))
    current_score = _candidate_score(best)
    improvement_ratio = max(
        _positive_ratio(current_score, baseline_score),
        _positive_ratio(_float(best.get("sharpe")), baseline_sharpe),
        _positive_ratio(_float(best.get("fitness")), baseline_fitness),
    )
    return {
        "optimization_anchor_id": anchor_id,
        "rounds": rounds,
        "next_round": rounds + 1,
        "baseline_score": baseline_score,
        "baseline_sharpe": baseline_sharpe,
        "baseline_fitness": baseline_fitness,
        "current_score": current_score,
        "improvement_ratio": improvement_ratio,
        "limit_exhausted": rounds >= MAX_OPTIMIZE_ROUNDS and improvement_ratio < MIN_OPTIMIZATION_IMPROVEMENT,
        "reset_anchor": improvement_ratio >= MIN_OPTIMIZATION_IMPROVEMENT,
    }


def _best_optimization_lineage(best: Dict[str, Any]) -> set[str]:
    ids = {str(best.get("id") or "")}
    metadata = best.get("generated_metadata") if isinstance(best.get("generated_metadata"), dict) else {}
    for key in ("optimization_anchor_id", "target_candidate_id"):
        value = metadata.get(key)
        if value not in (None, ""):
            ids.add(str(value))
    return {value for value in ids if value}


def _plan_matches_lineage(plan: Dict[str, Any], lineage_ids: set[str]) -> bool:
    if not lineage_ids:
        return False
    for key in ("optimization_anchor_id", "target_candidate_id"):
        value = plan.get(key)
        if value not in (None, "") and str(value) in lineage_ids:
            return True
    return False


def _scope_matches(candidate_settings: Any, target_settings: Dict[str, Any]) -> bool:
    if not isinstance(candidate_settings, dict) or not isinstance(target_settings, dict):
        return False
    if _normalized_scope_value(candidate_settings.get("region")) != _normalized_scope_value(target_settings.get("region")):
        return False
    if _normalized_scope_value(candidate_settings.get("delay")) != _normalized_scope_value(target_settings.get("delay")):
        return False
    candidate_universe = _normalized_scope_value(candidate_settings.get("universe"))
    target_universe = _normalized_scope_value(target_settings.get("universe"))
    if candidate_universe and target_universe and candidate_universe != target_universe:
        return False
    return True


def _normalized_scope_value(value: Any) -> str:
    text = str(value if value is not None else "").strip()
    return text.upper() if text and not text.isdigit() else text


def _next_optimization_fields(best: Dict[str, Any], state: Dict[str, Any], score: float) -> Dict[str, Any]:
    if state and not state.get("reset_anchor"):
        return {
            "anchor_id": state.get("optimization_anchor_id") or best.get("id"),
            "round": int(state.get("next_round") or 1),
            "baseline_score": round(_float(state.get("baseline_score")) or score, 6),
            "baseline_sharpe": round(_float(state.get("baseline_sharpe")) or _float(best.get("sharpe")), 6),
            "baseline_fitness": round(_float(state.get("baseline_fitness")) or _float(best.get("fitness")), 6),
        }
    return {
        "anchor_id": best.get("id"),
        "round": 1,
        "baseline_score": round(score, 6),
        "baseline_sharpe": round(_float(best.get("sharpe")), 6),
        "baseline_fitness": round(_float(best.get("fitness")), 6),
    }


def _optimization_gap_summary(best: Dict[str, Any], quality_thresholds: Dict[str, Any]) -> Dict[str, Any]:
    metrics = best.get("metrics") if isinstance(best.get("metrics"), dict) else {}
    if not metrics and ("sharpe" in best or "fitness" in best):
        metrics = best
    checks = _candidate_checks(best, metrics)
    components = best.get("quality_components") if isinstance(best.get("quality_components"), dict) else {}

    known_quality_checks: set[str] = set()
    open_quality_gaps: set[str] = set()
    terminal_blockers: set[str] = set()
    pending_quality_checks: set[str] = set()

    def add_status(raw_name: Any, raw_status: Any) -> None:
        name = _canonical_check_name(raw_name)
        status = str(raw_status or "").strip().upper()
        if not name:
            return
        if name in OPTIMIZATION_TERMINAL_BLOCKERS and status in {"FAIL", "WARNING"}:
            terminal_blockers.add(name)
        if name not in OPTIMIZATION_QUALITY_CHECKS:
            return
        known_quality_checks.add(name)
        if status in {"FAIL", "WARNING"}:
            open_quality_gaps.add(name)
        elif status == "PENDING":
            pending_quality_checks.add(name)

    for raw_name, data in _iter_checks(checks):
        add_status(raw_name, data.get("status", data.get("result")))

    for raw_name in components.get("failed_checks") or []:
        add_status(raw_name, "FAIL")
    for raw_name in components.get("warning_checks") or []:
        add_status(raw_name, "WARNING")
    for raw_name in components.get("pending_checks") or []:
        add_status(raw_name, "PENDING")
    for raw_name in components.get("passed_checks") or []:
        add_status(raw_name, "PASS")

    def add_metric_min_gap(check_name: str, metric_key: str, threshold_key: str) -> None:
        if check_name in known_quality_checks:
            return
        if metric_key not in metrics:
            return
        threshold = _positive_float(quality_thresholds.get(threshold_key))
        if not threshold:
            return
        known_quality_checks.add(check_name)
        if _float(metrics.get(metric_key), default=float("nan")) < threshold:
            open_quality_gaps.add(check_name)

    add_metric_min_gap("LOW_SHARPE", "sharpe", "required_sharpe")
    add_metric_min_gap("LOW_FITNESS", "fitness", "required_fitness")
    add_metric_min_gap("LOW_RETURNS", "returns", "required_returns")

    turnover = _float(metrics.get("turnover"), default=float("nan"))
    if turnover == turnover:
        turnover_min = _positive_float(quality_thresholds.get("turnover_min"))
        turnover_max = _positive_float(quality_thresholds.get("turnover_max"))
        if turnover_min and "LOW_TURNOVER" not in known_quality_checks:
            known_quality_checks.add("LOW_TURNOVER")
            if turnover < turnover_min:
                open_quality_gaps.add("LOW_TURNOVER")
        if turnover_max and "HIGH_TURNOVER" not in known_quality_checks:
            known_quality_checks.add("HIGH_TURNOVER")
            if turnover > turnover_max:
                open_quality_gaps.add("HIGH_TURNOVER")

    required_sharpe = _positive_float(quality_thresholds.get("required_sharpe"))
    required_fitness = _positive_float(quality_thresholds.get("required_fitness"))
    sharpe_floor = max(OPTIMIZE_FALLBACK_SHARPE_FLOOR, required_sharpe * 0.65) if required_sharpe else OPTIMIZE_FALLBACK_SHARPE_FLOOR
    fitness_floor = max(OPTIMIZE_FALLBACK_FITNESS_FLOOR, required_fitness * 0.50) if required_fitness else OPTIMIZE_FALLBACK_FITNESS_FLOOR
    sharpe = _float(metrics.get("sharpe"))
    fitness = _float(metrics.get("fitness"))
    close_metric_fallback = sharpe >= sharpe_floor or fitness >= fitness_floor
    enough_check_coverage = len(known_quality_checks) >= OPTIMIZE_MIN_KNOWN_QUALITY_CHECKS
    open_gap_count = len(open_quality_gaps)
    eligible = (
        bool(open_quality_gaps)
        and open_gap_count <= OPTIMIZE_MAX_OPEN_QUALITY_GAPS
        and not terminal_blockers
        and (enough_check_coverage or close_metric_fallback)
    )

    reason = "eligible_one_or_two_quality_gaps" if eligible else "not_near_enough_for_formula_optimization"
    if terminal_blockers:
        reason = "terminal_blocker_failed"
    elif not open_quality_gaps:
        reason = "no_open_quality_gap"
    elif open_gap_count > OPTIMIZE_MAX_OPEN_QUALITY_GAPS:
        reason = "too_many_quality_gaps"
    elif not enough_check_coverage and not close_metric_fallback:
        reason = "insufficient_check_coverage_and_not_close"

    return {
        "eligible": eligible,
        "reason": reason,
        "open_quality_gap_count": open_gap_count,
        "open_quality_gaps": sorted(open_quality_gaps),
        "known_quality_check_count": len(known_quality_checks),
        "known_quality_checks": sorted(known_quality_checks),
        "pending_quality_checks": sorted(pending_quality_checks),
        "terminal_blockers": sorted(terminal_blockers),
        "close_metric_fallback": close_metric_fallback,
        "sharpe_floor": round(sharpe_floor, 6),
        "fitness_floor": round(fitness_floor, 6),
    }


def _best_candidate(candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
    scored = []
    preferred = [candidate for candidate in candidates if str(candidate.get("status") or "") != "failed"]
    for candidate in preferred or candidates:
        metrics = candidate.get("metrics") if isinstance(candidate.get("metrics"), dict) else {}
        if "sharpe" not in metrics and "fitness" not in metrics:
            continue
        sharpe = _float(metrics.get("sharpe"))
        fitness = _float(metrics.get("fitness"))
        score_details = _candidate_score_details(candidate)
        scored.append((score_details["quality_score"], sharpe, fitness, candidate, score_details))
    if not scored:
        return {}
    _score, sharpe, fitness, candidate, score_details = sorted(scored, key=lambda item: item[0], reverse=True)[0]
    metrics = candidate.get("metrics") if isinstance(candidate.get("metrics"), dict) else {}
    checks = _candidate_checks(candidate, metrics)
    return {
        "id": candidate.get("id"),
        "status": candidate.get("status"),
        "alpha_id": candidate.get("alpha_id"),
        "expression": candidate.get("expression") or "",
        "sharpe": sharpe,
        "fitness": fitness,
        "raw_score": score_details["raw_score"],
        "quality_score": score_details["quality_score"],
        "quality_components": score_details,
        "metrics": metrics,
        "checks": checks,
        "fields": _extract_fields(str(candidate.get("expression") or ""), []),
        "last_relevant_event": candidate.get("last_relevant_event") or {},
    }


def _candidate_score(candidate: Dict[str, Any]) -> float:
    return _candidate_score_details(candidate)["quality_score"]


def candidate_quality_summary(candidate: Dict[str, Any]) -> Dict[str, Any]:
    return _candidate_score_details(candidate)


def candidate_optimization_summary(candidate: Dict[str, Any]) -> Dict[str, Any]:
    metrics = candidate.get("metrics") if isinstance(candidate.get("metrics"), dict) else {}
    checks = _candidate_checks(candidate, metrics)
    thresholds = _candidate_quality_thresholds(candidate, checks)
    return _optimization_gap_summary(candidate, thresholds)


def _candidate_score_details(candidate: Dict[str, Any]) -> Dict[str, Any]:
    metrics = candidate.get("metrics") if isinstance(candidate.get("metrics"), dict) else {}
    if not metrics and ("sharpe" in candidate or "fitness" in candidate):
        metrics = candidate
    checks = _candidate_checks(candidate, metrics)
    thresholds = _candidate_quality_thresholds(candidate, checks)
    sharpe = _float(metrics.get("sharpe"))
    fitness = _float(metrics.get("fitness"))
    raw_score = sharpe + 0.35 * fitness
    penalty = 0.0
    bonus = 0.0
    failed_checks: List[str] = []
    warning_checks: List[str] = []
    pending_checks: List[str] = []
    passed_checks: List[str] = []
    component_scores: Dict[str, float] = {}
    weighted_scores: List[tuple[float, float]] = []

    def add_component(name: str, value: Any, limit: Any, *, mode: str = "min", weight: float = 1.0) -> None:
        number = _float(value, default=float("nan"))
        threshold = _positive_float(limit)
        if number != number or not threshold:
            return
        if mode == "max":
            ratio = QUALITY_COMPONENT_CAP if number <= threshold else threshold / max(number, 1e-12)
        else:
            ratio = number / threshold
        score = max(0.0, min(ratio, QUALITY_COMPONENT_CAP))
        component_scores[name] = round(score, 6)
        weighted_scores.append((score, weight))

    for threshold_key, (metric_key, check_name) in QUALITY_METRIC_REQUIREMENTS.items():
        required = thresholds.get(threshold_key)
        if required in (None, ""):
            continue
        value = metrics.get(metric_key)
        if value in (None, ""):
            value = _check_value(checks, {check_name}, None)
        add_component(threshold_key, value, required, weight=1.0)

    turnover = _float(metrics.get("turnover"), default=-1.0)
    turnover_min = _positive_float(thresholds.get("turnover_min")) or _check_limit(checks, {"LOW_TURNOVER"}, 0.0)
    turnover_max = _positive_float(thresholds.get("turnover_max")) or _check_limit(checks, {"HIGH_TURNOVER"}, 0.0)
    preferred_turnover_max = _positive_float(thresholds.get("preferred_turnover_max"))
    if turnover >= 0:
        turnover_score = 1.0
        if turnover_min and turnover < turnover_min:
            turnover_score = turnover / turnover_min
        elif turnover_max and turnover > turnover_max:
            turnover_score = turnover_max / max(turnover, 1e-12)
        elif preferred_turnover_max and turnover > preferred_turnover_max:
            turnover_score = max(0.0, preferred_turnover_max / max(turnover, 1e-12))
        component_scores["turnover_range"] = round(max(0.0, min(turnover_score, QUALITY_COMPONENT_CAP)), 6)
        weighted_scores.append((max(0.0, min(turnover_score, QUALITY_COMPONENT_CAP)), 0.8))
        if turnover_min and turnover < turnover_min:
            penalty += 0.45
        elif turnover_max and turnover > turnover_max:
            penalty += 0.45
        elif preferred_turnover_max and turnover > preferred_turnover_max:
            penalty += 0.12
        elif turnover_min and turnover_max and turnover_min <= turnover <= turnover_max:
            bonus += 0.08

    extra_checks = thresholds.get("extra_checks") if isinstance(thresholds.get("extra_checks"), dict) else {}
    for raw_name, raw_limit in extra_checks.items():
        name = _canonical_check_name(raw_name)
        value = _check_value(checks, {name}, None)
        if value in (None, ""):
            value = _metric_value_for_check(metrics, name)
        add_component(f"extra:{name}", value, raw_limit, weight=0.8)

    seen_component_checks = {
        "LOW_SHARPE",
        "LOW_FITNESS",
        "LOW_RETURNS",
        "LOW_TURNOVER",
        "HIGH_TURNOVER",
    } | {_canonical_check_name(name) for name in extra_checks.keys()}
    for raw_name, data in _iter_checks(checks):
        name = _canonical_check_name(raw_name)
        if name in seen_component_checks:
            continue
        limit = data.get("limit")
        value = data.get("value")
        status = str(data.get("status") or data.get("result") or "").upper()
        if limit in (None, "") or value in (None, ""):
            continue
        mode = "max" if name.startswith("HIGH_") or name in {"HT_TURNOVER"} else "min"
        weight = 0.45 if status in {"WARNING", "FAIL"} else 0.35
        add_component(f"check:{name}", value, limit, mode=mode, weight=weight)

    drawdown = _float(metrics.get("drawdown"), default=0.0)
    if drawdown > 0.15:
        penalty += 0.25
    elif drawdown > 0.08:
        penalty += 0.12
    elif 0 < drawdown <= 0.06:
        bonus += 0.05

    for raw_name, data in _iter_checks(checks):
        name = _canonical_check_name(raw_name)
        status = str(data.get("status") or data.get("result") or "").upper()
        if status == "FAIL":
            if _is_regular_submission_quota_full(name, data):
                pending_checks.append(name)
                if name in MANDATORY_SUBMISSION_CHECKS:
                    penalty += 0.04
                continue
            failed_checks.append(name)
            if name not in seen_component_checks:
                penalty += CHECK_PENALTIES.get(name, 0.35)
        elif status == "WARNING":
            warning_checks.append(name)
            if name not in seen_component_checks:
                penalty += min(0.25, CHECK_PENALTIES.get(name, 0.25) * 0.45)
        elif status == "PENDING":
            pending_checks.append(name)
            if name in MANDATORY_SUBMISSION_CHECKS:
                penalty += 0.04
        elif status == "PASS":
            passed_checks.append(name)
            bonus += CHECK_PASS_BONUSES.get(name, 0.0)

    missing_submission_checks = _missing_submission_checks(passed_checks, pending_checks, failed_checks, warning_checks)
    pending_submission_checks = _submission_checks_in(pending_checks)
    failed_submission_checks = _submission_checks_in(failed_checks)
    warning_submission_checks = _submission_checks_in(warning_checks)

    readiness_score = 0.0
    if weighted_scores:
        total_weight = sum(weight for _score, weight in weighted_scores)
        readiness_score = sum(score * weight for score, weight in weighted_scores) / max(total_weight, 1e-12)

    capped_bonus = min(bonus, 0.45)
    exploration_score = readiness_score + capped_bonus - penalty
    submission_uncertainty_penalty = (
        0.18 * len(missing_submission_checks)
        + 0.08 * len(pending_submission_checks)
        + 0.05 * len(warning_submission_checks)
        + 0.22 * len(failed_submission_checks)
    )
    submission_score = max(0.0, readiness_score + capped_bonus - penalty - submission_uncertainty_penalty)
    return {
        "raw_score": round(raw_score, 6),
        "quality_score": round(exploration_score, 6),
        "exploration_score": round(exploration_score, 6),
        "submission_score": round(submission_score, 6),
        "readiness_score": round(readiness_score, 6),
        "quality_penalty": round(penalty, 6),
        "quality_bonus": round(capped_bonus, 6),
        "submission_uncertainty_penalty": round(submission_uncertainty_penalty, 6),
        "component_scores": component_scores,
        "failed_checks": failed_checks[:12],
        "warning_checks": warning_checks[:12],
        "pending_checks": pending_checks[:12],
        "passed_checks": passed_checks[:12],
        "missing_submission_checks": missing_submission_checks[:12],
        "pending_submission_checks": pending_submission_checks[:12],
        "failed_submission_checks": failed_submission_checks[:12],
    }


def _has_setting_sweep_blockers(best: Dict[str, Any]) -> bool:
    components = best.get("quality_components") if isinstance(best.get("quality_components"), dict) else {}
    failed = {_canonical_check_name(name) for name in components.get("failed_checks") or []}
    return bool(failed & SETTING_SWEEP_BLOCKING_FAILURES)


def _is_regular_submission_quota_full(name: str, data: Dict[str, Any]) -> bool:
    if name != "REGULAR_SUBMISSION":
        return False
    value = _float(data.get("value"), default=-1.0)
    limit = _float(data.get("limit"), default=-1.0)
    return limit > 0 and value >= limit


def _submission_checks_in(names: List[str]) -> List[str]:
    canonical = {_canonical_check_name(name) for name in names}
    result: List[str] = []
    for aliases in SUBMISSION_CHECK_ALIASES:
        matched = sorted(canonical & aliases)
        if matched:
            result.append(matched[0])
    return result


def _missing_submission_checks(
    passed_checks: List[str],
    pending_checks: List[str],
    failed_checks: List[str],
    warning_checks: List[str],
) -> List[str]:
    known = {
        _canonical_check_name(name)
        for name in [*passed_checks, *pending_checks, *failed_checks, *warning_checks]
    }
    missing: List[str] = []
    for aliases in SUBMISSION_CHECK_ALIASES:
        if not (known & aliases):
            missing.append(sorted(aliases)[0])
    return missing


def _candidate_checks(candidate: Dict[str, Any], metrics: Dict[str, Any]) -> Any:
    checks = candidate.get("checks") if isinstance(candidate.get("checks"), (dict, list)) else {}
    if checks:
        return checks
    embedded = metrics.get("checks") if isinstance(metrics, dict) else {}
    return embedded if isinstance(embedded, (dict, list)) else {}


def _candidate_quality_thresholds(candidate: Dict[str, Any], checks: Any) -> Dict[str, Any]:
    settings = candidate.get("settings") if isinstance(candidate.get("settings"), dict) else {}
    configured = _scope_quality_thresholds(settings)
    fallback = _default_quality_thresholds()
    result: Dict[str, Any] = dict(fallback)
    result.update({key: value for key, value in configured.items() if value not in (None, "")})

    observed = {
        "required_sharpe": _check_limit(checks, {"LOW_SHARPE"}, 0.0),
        "required_fitness": _check_limit(checks, {"LOW_FITNESS"}, 0.0),
        "required_returns": _check_limit(checks, {"LOW_RETURNS"}, 0.0),
        "turnover_min": _check_limit(checks, {"LOW_TURNOVER"}, 0.0),
        "turnover_max": _check_limit(checks, {"HIGH_TURNOVER"}, 0.0),
    }
    for key, value in observed.items():
        if value > 0:
            result[key] = value
    return result


def _canonical_check_name(name: Any) -> str:
    text = str(name or "").strip().upper()
    if text == "PRODCORRELATION":
        return "PROD_CORRELATION"
    if text == "PRODUCTCORRELATION":
        return "PRODUCT_CORRELATION"
    if text == "SELFCORRELATION":
        return "SELF_CORRELATION"
    if text == "DATADIVERSITY":
        return "DATA_DIVERSITY"
    if text == "REGULARSUBMISSION":
        return "REGULAR_SUBMISSION"
    return text


def _check_value(checks: Any, names: set[str], default: Any) -> Any:
    canonical_names = {_canonical_check_name(name) for name in names}
    for name, data in _iter_checks(checks):
        if _canonical_check_name(name) not in canonical_names:
            continue
        value = data.get("value")
        return default if value in (None, "") else value
    return default


def _metric_value_for_check(metrics: Dict[str, Any], check_name: str) -> Any:
    normalized = re.sub(r"[^a-z0-9]+", "", check_name.lower())
    metric_aliases = {
        "lowsharpe": "sharpe",
        "lowfitness": "fitness",
        "lowreturns": "returns",
        "lowturnover": "turnover",
        "highturnover": "turnover",
        "isladdersharpe": "ladder_sharpe",
        "lowsubuniversesharpe": "subuniverse_sharpe",
        "lowrobustuniversesharpe": "robust_universe_sharpe",
        "robustuniversesharpe": "robust_universe_sharpe",
        "robustuniverseretention": "robust_universe_retention",
        "investabilitysharperatio": "investability_sharpe_ratio",
        "amersharpe": "amer_sharpe",
        "apacsharpe": "apac_sharpe",
        "emeasharpe": "emea_sharpe",
        "mostilliquid50aftercostsharpefraction": "most_illiquid_50_after_cost_sharpe_fraction",
    }
    keys = [
        metric_aliases.get(normalized, ""),
        check_name.lower(),
        check_name.lower().replace("_", ""),
        _snake_to_camel(check_name.lower()),
    ]
    for key in keys:
        if key and key in metrics:
            return metrics.get(key)
    return None


def _snake_to_camel(value: str) -> str:
    parts = [part for part in value.split("_") if part]
    if not parts:
        return value
    return parts[0] + "".join(part[:1].upper() + part[1:] for part in parts[1:])


def _quality_thresholds(best: Dict[str, Any], target_settings: Dict[str, Any], analysis: Dict[str, Any]) -> Dict[str, Any]:
    configured = _scope_quality_thresholds(target_settings)
    account_type = configured.get("account_type") or _target_account_type(target_settings, _load_quality_threshold_config())
    observed = analysis.get("observed_quality_thresholds") if isinstance(analysis.get("observed_quality_thresholds"), dict) else {}
    checks = best.get("checks") if isinstance(best.get("checks"), (dict, list)) else {}
    check_sharpe = _check_limit(checks, {"LOW_SHARPE"}, 0.0)
    check_fitness = _check_limit(checks, {"LOW_FITNESS"}, 0.0)
    check_returns = _check_limit(checks, {"LOW_RETURNS"}, 0.0)
    default_thresholds = _default_quality_thresholds()
    configured_specificity = int(configured.get("specificity") or 0)
    exact_config = bool(configured.get("official")) or configured_specificity >= 2
    observed_available = any(
        _positive_float(observed.get(key)) > 0
        for key in ("required_sharpe", "required_fitness", "required_returns")
    )
    best_checks_available = check_sharpe > 0 or check_fitness > 0 or check_returns > 0
    exact_config_available = exact_config and (
        _positive_float(configured.get("required_sharpe")) > 0
        or _positive_float(configured.get("required_fitness")) > 0
    )
    trusted = observed_available or best_checks_available or exact_config_available
    primary = configured if exact_config_available else {}
    secondary = observed if observed_available else {}
    tertiary = (
        {
            "required_sharpe": check_sharpe,
            "required_fitness": check_fitness,
            "required_returns": check_returns,
            "source": "best_candidate_checks",
        }
        if best_checks_available
        else {}
    )
    provisional = configured if configured and not exact_config_available else default_thresholds

    required_sharpe = (
        _positive_float(primary.get("required_sharpe"))
        or _positive_float(secondary.get("required_sharpe"))
        or _positive_float(tertiary.get("required_sharpe"))
        or _positive_float(provisional.get("required_sharpe"))
        or DEFAULT_REQUIRED_SHARPE
    )
    required_fitness = (
        _positive_float(primary.get("required_fitness"))
        or _positive_float(secondary.get("required_fitness"))
        or _positive_float(tertiary.get("required_fitness"))
        or _positive_float(provisional.get("required_fitness"))
        or DEFAULT_REQUIRED_FITNESS
    )
    required_returns = (
        _positive_float(primary.get("required_returns"))
        or _positive_float(secondary.get("required_returns"))
        or _positive_float(tertiary.get("required_returns"))
        or _positive_float(provisional.get("required_returns"))
    )
    source = (
        primary.get("source")
        or secondary.get("source")
        or tertiary.get("source")
        or provisional.get("source")
        or "built_in_default"
    )
    result = {
        "required_sharpe": round(required_sharpe, 6),
        "required_fitness": round(required_fitness, 6),
        "optimize_sharpe": round(required_sharpe * OPTIMIZE_SHARPE_RATIO, 6),
        "optimize_fitness": round(required_fitness * OPTIMIZE_FITNESS_RATIO, 6),
        "setting_sweep_sharpe": round(required_sharpe * SETTING_SWEEP_SHARPE_RATIO, 6),
        "setting_sweep_fitness": round(required_fitness * SETTING_SWEEP_FITNESS_RATIO, 6),
        "optimize_readiness": _planning_threshold("optimize_readiness", DEFAULT_OPTIMIZE_READINESS),
        "setting_sweep_readiness": _planning_threshold("setting_sweep_readiness", DEFAULT_SETTING_SWEEP_READINESS),
        "source": str(source),
        "configured_source": str(configured.get("source") or ""),
        "source_doc": str(configured.get("source_doc") or ""),
        "account_type": str(account_type),
        "official": bool(configured.get("official")),
        "trusted": bool(trusted),
        "check_thresholds": _check_thresholds(checks),
    }
    if required_returns:
        result["required_returns"] = round(required_returns, 6)
    for key in THRESHOLD_COPY_KEYS:
        if key == "required_returns" and required_returns:
            continue
        value = configured.get(key)
        if value not in (None, ""):
            result[key] = value
    return result


def _observed_quality_thresholds(candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
    sharpe_limits: List[float] = []
    fitness_limits: List[float] = []
    returns_limits: List[float] = []
    for candidate in candidates:
        metrics = candidate.get("metrics") if isinstance(candidate.get("metrics"), dict) else {}
        checks = _candidate_checks(candidate, metrics)
        sharpe_limit = _check_limit(checks, {"LOW_SHARPE"}, 0.0)
        fitness_limit = _check_limit(checks, {"LOW_FITNESS"}, 0.0)
        returns_limit = _check_limit(checks, {"LOW_RETURNS"}, 0.0)
        if sharpe_limit > 0:
            sharpe_limits.append(sharpe_limit)
        if fitness_limit > 0:
            fitness_limits.append(fitness_limit)
        if returns_limit > 0:
            returns_limits.append(returns_limit)
    if not sharpe_limits and not fitness_limits and not returns_limits:
        return {}
    result: Dict[str, Any] = {"source": "observed_checks"}
    if sharpe_limits:
        result["required_sharpe"] = max(sharpe_limits)
    if fitness_limits:
        result["required_fitness"] = max(fitness_limits)
    if returns_limits:
        result["required_returns"] = max(returns_limits)
    return result


def _check_limit(checks: Any, names: set[str], default: float) -> float:
    canonical_names = {_canonical_check_name(name) for name in names}
    for name, data in _iter_checks(checks):
        if _canonical_check_name(name) not in canonical_names:
            continue
        limit = _float(data.get("limit"), default=0.0)
        if limit > 0:
            return limit
    return default


def _check_thresholds(checks: Any) -> Dict[str, float]:
    thresholds: Dict[str, float] = {}
    for name, data in _iter_checks(checks):
        limit = _positive_float(data.get("limit"))
        if limit:
            thresholds[_canonical_check_name(name)] = limit
    return thresholds


def _scope_quality_thresholds(target_settings: Dict[str, Any]) -> Dict[str, Any]:
    config = _load_quality_threshold_config()
    account_type = _target_account_type(target_settings, config)
    scopes = config.get("scopes") if isinstance(config, dict) else None
    if not isinstance(scopes, list):
        return {}
    matches: List[tuple[int, Dict[str, Any]]] = []
    target = dict(target_settings)
    target["account_type"] = account_type
    for item in scopes:
        if not isinstance(item, dict) or not _scope_account_matches(item, account_type):
            continue
        if not _scope_threshold_matches(item, target):
            continue
        specificity = _scope_threshold_specificity(item)
        matches.append((specificity, item))
    if not matches:
        return {}
    item = sorted(matches, key=lambda pair: pair[0], reverse=True)[0][1]
    result = {
        "required_sharpe": item.get("required_sharpe"),
        "required_fitness": item.get("required_fitness"),
        "source": item.get("name") or _scope_threshold_source(item),
        "specificity": max(specificity for specificity, _item in matches),
        "account_type": account_type,
        "source_doc": item.get("source_doc"),
        "official": bool(item.get("official")),
    }
    for key in THRESHOLD_COPY_KEYS:
        if key in item:
            result[key] = item.get(key)
    return result


def _default_quality_thresholds() -> Dict[str, Any]:
    config = _load_quality_threshold_config()
    data = config.get("default") if isinstance(config, dict) else None
    if not isinstance(data, dict):
        return {}
    result = {
        "required_sharpe": data.get("required_sharpe"),
        "required_fitness": data.get("required_fitness"),
        "source": data.get("name") or "scope_quality_thresholds.default",
        "account_type": data.get("account_type") or config.get("default_account_type") or "consultant",
        "source_doc": data.get("source_doc"),
        "official": bool(data.get("official")),
    }
    for key in THRESHOLD_COPY_KEYS:
        if key in data:
            result[key] = data.get(key)
    return result


def _planning_threshold(key: str, default: float) -> float:
    config = _load_quality_threshold_config()
    planning = config.get("planning") if isinstance(config.get("planning"), dict) else {}
    return round(_positive_float(planning.get(key)) or default, 6)


def _scope_threshold_matches(item: Dict[str, Any], target_settings: Dict[str, Any]) -> bool:
    target_region = _scope_value(target_settings.get("region"))
    excluded_regions = {_scope_value(value) for value in item.get("exclude_regions") or []}
    if target_region and target_region in excluded_regions:
        return False
    allowed_regions = {_scope_value(value) for value in item.get("regions") or []}
    if allowed_regions and target_region not in allowed_regions:
        return False

    for key in ("account_type", "instrumentType", "region", "delay", "universe", "neutralization"):
        if key not in item:
            continue
        if _scope_value(item.get(key)) != _scope_value(_target_setting_value(target_settings, key)):
            return False
    return True


def _scope_threshold_source(item: Dict[str, Any]) -> str:
    account_type = str(item.get("account_type") or "*").lower()
    region = str(item.get("region") or "*").upper()
    universe = str(item.get("universe") or "*").upper()
    delay = str(item.get("delay") if item.get("delay") is not None else "*")
    return f"scope_quality_thresholds:{account_type}:{region}/{universe}/D{delay}"


def _scope_account_matches(item: Dict[str, Any], account_type: str) -> bool:
    allowed = item.get("account_types")
    if isinstance(allowed, list):
        return _scope_value(account_type) in {_scope_value(value) for value in allowed}
    if "account_type" in item:
        return _scope_value(item.get("account_type")) == _scope_value(account_type)
    return True


def _scope_threshold_specificity(item: Dict[str, Any]) -> int:
    return sum(
        1
        for key in ("account_type", "account_types", "instrumentType", "region", "regions", "delay", "universe", "neutralization")
        if key in item
    )


def _target_setting_value(target_settings: Dict[str, Any], key: str) -> Any:
    if key == "instrumentType":
        return target_settings.get("instrumentType") or target_settings.get("instrument_type") or "EQUITY"
    if key == "account_type":
        return target_settings.get("account_type") or target_settings.get("accountType")
    return target_settings.get(key)


def _target_account_type(target_settings: Dict[str, Any], config: Dict[str, Any]) -> str:
    default = config.get("default") if isinstance(config.get("default"), dict) else {}
    value = (
        target_settings.get("account_type")
        or target_settings.get("accountType")
        or os.getenv("ALPHA_ACCOUNT_TYPE")
        or config.get("default_account_type")
        or default.get("account_type")
        or "consultant"
    )
    text = str(value).strip().lower()
    return text or "consultant"


def _scope_value(value: Any) -> str:
    return str(value if value is not None else "").strip().upper()


def _positive_float(value: Any) -> float:
    number = _float(value, default=0.0)
    return number if number > 0 else 0.0


@lru_cache(maxsize=1)
def _load_quality_threshold_config() -> Dict[str, Any]:
    raw_path = os.getenv("ALPHA_SCOPE_QUALITY_THRESHOLDS_FILE", "").strip()
    path = Path(raw_path) if raw_path else DEFAULT_SCOPE_QUALITY_THRESHOLDS_FILE
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _iter_checks(checks: Any) -> Iterable[tuple[str, Dict[str, Any]]]:
    if isinstance(checks, dict):
        for name, data in checks.items():
            if isinstance(data, dict):
                yield str(name), data
        return
    if isinstance(checks, list):
        for item in checks:
            if isinstance(item, dict) and item.get("name"):
                yield str(item["name"]), item


def _failure_reasons(candidates: List[Dict[str, Any]]) -> Counter[str]:
    reasons: Counter[str] = Counter()
    for candidate in candidates:
        event = candidate.get("last_relevant_event")
        metadata = event.get("metadata") if isinstance(event, dict) else {}
        errors = metadata.get("errors") if isinstance(metadata, dict) else []
        if isinstance(errors, list):
            for error in errors:
                reasons[_normalize_reason(error)] += 1
        for error in candidate.get("simulation_errors") or []:
            if isinstance(error, dict):
                reasons[_normalize_reason(error.get("error"))] += 1
    return reasons


def _field_stats(candidates: List[Dict[str, Any]], research_context: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    known_fields = _known_fields(research_context)
    totals: Dict[str, Dict[str, float]] = defaultdict(lambda: {"count": 0.0, "sharpe": 0.0, "fitness": 0.0})
    for candidate in candidates:
        expression = str(candidate.get("expression") or "")
        fields = _extract_fields(expression, known_fields)
        if not fields:
            continue
        metrics = candidate.get("metrics") if isinstance(candidate.get("metrics"), dict) else {}
        sharpe = _float(metrics.get("sharpe"))
        fitness = _float(metrics.get("fitness"))
        for field in fields:
            totals[field]["count"] += 1
            totals[field]["sharpe"] += sharpe
            totals[field]["fitness"] += fitness

    stats: Dict[str, Dict[str, Any]] = {}
    for field, total in sorted(totals.items(), key=lambda item: (-item[1]["count"], item[0])):
        count = max(1.0, total["count"])
        stats[field] = {
            "count": int(total["count"]),
            "avg_sharpe": round(total["sharpe"] / count, 3),
            "avg_fitness": round(total["fitness"] / count, 3),
        }
    return stats


def _field_family_stats(candidates: List[Dict[str, Any]], research_context: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    known_fields = _known_fields(research_context)
    totals: Dict[str, Dict[str, float]] = defaultdict(lambda: {"count": 0.0, "sharpe": 0.0, "fitness": 0.0})
    for candidate in candidates:
        expression = str(candidate.get("expression") or "")
        fields = _extract_fields(expression, known_fields)
        if not fields:
            continue
        metrics = candidate.get("metrics") if isinstance(candidate.get("metrics"), dict) else {}
        sharpe = _float(metrics.get("sharpe"))
        fitness = _float(metrics.get("fitness"))
        families: List[str] = []
        for field in fields:
            family = _field_family(field)
            if family and family not in families:
                families.append(family)
        for family in families:
            totals[family]["count"] += 1
            totals[family]["sharpe"] += sharpe
            totals[family]["fitness"] += fitness

    stats: Dict[str, Dict[str, Any]] = {}
    for family, total in sorted(totals.items(), key=lambda item: (-item[1]["count"], item[0])):
        count = max(1.0, total["count"])
        stats[family] = {
            "count": int(total["count"]),
            "avg_sharpe": round(total["sharpe"] / count, 3),
            "avg_fitness": round(total["fitness"] / count, 3),
        }
    return stats


def _promising_fields(best: Dict[str, Any], field_stats: Dict[str, Dict[str, Any]]) -> List[str]:
    fields: List[str] = []
    for field in best.get("fields") or []:
        if field not in fields:
            fields.append(field)
    for field, stats in sorted(
        field_stats.items(),
        key=lambda item: (float(item[1].get("avg_sharpe", 0.0)), float(item[1].get("avg_fitness", 0.0))),
        reverse=True,
    ):
        if field not in fields and float(stats.get("avg_sharpe", 0.0)) >= 0.5:
            fields.append(field)
    return fields[:12]


def _diversified_keep_fields(
    best: Dict[str, Any],
    field_stats: Dict[str, Dict[str, Any]],
    family_diversity_control: Dict[str, Any],
    submitted_avoid_fields: List[str],
    limit: int,
) -> List[str]:
    keep: List[str] = []
    dominant_family = str(family_diversity_control.get("dominant_family") or "").strip()
    dominant_limit = 2 if dominant_family else limit
    avoid_fields = set(submitted_avoid_fields)

    def add(field: str) -> None:
        if field and field not in avoid_fields and field not in keep and len(keep) < limit:
            keep.append(field)

    for field in best.get("fields") or []:
        add(str(field))
        if len(keep) >= min(limit, 2):
            break

    ordered_fields = sorted(
        field_stats.items(),
        key=lambda item: (float(item[1].get("avg_sharpe", 0.0)), float(item[1].get("avg_fitness", 0.0))),
        reverse=True,
    )
    for field, _stats in ordered_fields:
        family = _field_family(field)
        if dominant_family and family == dominant_family:
            if sum(1 for item in keep if _field_family(item) == dominant_family) >= dominant_limit:
                continue
        add(field)
        if len(keep) >= limit:
            break
    return keep


def _should_abandon_weak_anchor(
    scope_trouble: Dict[str, Any],
    route_stop_loss: Dict[str, Any],
    optimization_gap_summary: Dict[str, Any],
) -> bool:
    if scope_trouble.get("active") or route_stop_loss.get("stop_loss_active"):
        return True
    if optimization_gap_summary.get("eligible"):
        return False
    return str(optimization_gap_summary.get("reason") or "") in {
        "too_many_quality_gaps",
        "terminal_blocker_failed",
        "insufficient_check_coverage_and_not_close",
    }


def _submitted_avoidance_objective(submitted_avoidance: Dict[str, Any]) -> str:
    fields = [str(field) for field in submitted_avoidance.get("fields") or [] if str(field).strip()]
    if not fields:
        return ""
    return (
        " Avoid recently approved/submitted core fields in this scope because even passing formulas are likely "
        f"to hit self/prod correlation: {', '.join(fields[:8])}."
    )


def _mechanism_memory_from_history(history_memory: Dict[str, Any]) -> Dict[str, Any]:
    archetypes = history_memory.get("blocked_winner_archetypes")
    if not isinstance(archetypes, list) or not archetypes:
        return {}
    compact_archetypes: List[Dict[str, Any]] = []
    forbidden_fields: List[str] = []
    mechanism_tags: List[str] = []
    for item in archetypes[:8]:
        if not isinstance(item, dict):
            continue
        compact = {
            key: item.get(key)
            for key in (
                "id",
                "status",
                "metrics",
                "quality_score",
                "readiness_score",
                "fields",
                "families",
                "blocked_by",
                "forbidden_fields",
                "mechanism_tags",
                "transfer_hint",
            )
            if item.get(key) not in (None, "", [], {})
        }
        compact_archetypes.append(compact)
        for field in item.get("forbidden_fields") or []:
            text = str(field or "").strip()
            if text and text not in forbidden_fields:
                forbidden_fields.append(text)
        for tag in item.get("mechanism_tags") or []:
            text = str(tag or "").strip()
            if text and text not in mechanism_tags:
                mechanism_tags.append(text)
    if not compact_archetypes:
        return {}
    return {
        "policy": (
            "Historical high-signal but blocked candidates are mechanism-only exemplars. "
            "Do not copy their expressions or forbidden_fields; transfer their mechanisms to fresh primary fields."
        ),
        "archetypes": compact_archetypes,
        "forbidden_fields": forbidden_fields[:30],
        "mechanism_tags": mechanism_tags[:20],
    }


def _route_efficiency(
    scope_health: Dict[str, Any],
    queue_counts: Dict[str, Any],
    quality_thresholds: Dict[str, Any],
) -> Dict[str, Any]:
    signals = scope_health.get("trouble_signals") if isinstance(scope_health.get("trouble_signals"), dict) else {}
    scanned = int(_float(signals.get("scanned_candidates")))
    failure_streak = int(_float(signals.get("failure_streak")))
    watchlist_count = int(_float(queue_counts.get("watchlist")))
    submitable_count = int(_float(queue_counts.get("submitable")))
    optimize_count = int(_float(queue_counts.get("optimize")))
    best_recent_sharpe = _float(scope_health.get("best_recent_sharpe"))
    best_recent_fitness = _float(scope_health.get("best_recent_fitness"))
    best_recent_quality_score = _float(scope_health.get("best_recent_quality_score"))
    required_sharpe = _positive_float(quality_thresholds.get("required_sharpe"))
    required_fitness = _positive_float(quality_thresholds.get("required_fitness"))
    sharpe_ratio = _positive_ratio(best_recent_sharpe, required_sharpe) if required_sharpe else 0.0
    fitness_ratio = _positive_ratio(best_recent_fitness, required_fitness) if required_fitness else 0.0
    strong_progress = submitable_count + watchlist_count
    stop_loss_active = (
        scanned >= ROUTE_STOP_LOSS_MIN_SCANNED
        and failure_streak >= ROUTE_STOP_LOSS_FAILURE_STREAK
        and strong_progress == 0
        and sharpe_ratio < ROUTE_STOP_LOSS_SHARPE_RATIO
    )
    return {
        "active": bool(stop_loss_active),
        "stop_loss_active": bool(stop_loss_active),
        "reason": "NO_WATCHLIST_OR_SUBMITABLE_PROGRESS" if stop_loss_active else "",
        "scanned_candidates": scanned,
        "failure_streak": failure_streak,
        "submitable_count": submitable_count,
        "watchlist_count": watchlist_count,
        "optimize_count": optimize_count,
        "best_recent_sharpe": best_recent_sharpe,
        "best_recent_fitness": best_recent_fitness,
        "best_recent_quality_score": best_recent_quality_score,
        "required_sharpe": required_sharpe,
        "required_fitness": required_fitness,
        "best_sharpe_ratio": round(sharpe_ratio, 4),
        "best_fitness_ratio": round(fitness_ratio, 4),
        "policy": (
            "If active, stop spending full rounds on local variants of this route. Switch mechanism and cap repeated "
            "formula skeletons until a candidate reaches watchlist or submitable quality."
        ),
    }


def _structure_diversity_control(candidates: List[Dict[str, Any]], history_memory: Dict[str, Any]) -> Dict[str, Any]:
    top_structures = history_memory.get("top_structures") if isinstance(history_memory.get("top_structures"), list) else []
    overused: List[Dict[str, Any]] = []
    for item in top_structures:
        if not isinstance(item, dict):
            continue
        count = int(_float(item.get("count")))
        failed = int(_float(item.get("failed")))
        failure_rate = _float(item.get("failure_rate"))
        best_quality = _float(item.get("best_quality_score"))
        if count >= 6 and failed >= 4 and failure_rate >= 0.65 and best_quality < 0.45:
            overused.append(
                {
                    "structure_key": item.get("structure_key"),
                    "count": count,
                    "failed": failed,
                    "failure_rate": round(failure_rate, 4),
                    "best_sharpe": item.get("best_sharpe"),
                    "best_fitness": item.get("best_fitness"),
                    "best_quality_score": item.get("best_quality_score"),
                    "example_expression": item.get("example_expression"),
                }
            )
    if not overused:
        counts: Counter[str] = Counter()
        examples: Dict[str, str] = {}
        for candidate in candidates:
            expression = str(candidate.get("expression") or "")
            key = expression_structure_key(expression)
            counts[key] += 1
            examples.setdefault(key, expression)
        for key, count in counts.most_common(6):
            if count >= 3:
                # Same-round repetition cap: these are repeated skeletons in the current
                # batch, not historically failed structures. Do not fabricate failure stats.
                overused.append(
                    {
                        "structure_key": key,
                        "count": count,
                        "same_round_repeats": count,
                        "reason": "same_round_repetition",
                        "example_expression": examples.get(key),
                    }
                )
    return {
        "active": bool(overused),
        "max_batch_candidates_per_structure": MAX_BATCH_CANDIDATES_PER_STRUCTURE,
        "overused_structures": overused[:8],
        "policy": (
            "During exploration, cap same-round candidates with the same field-agnostic formula skeleton. "
            "Use different operator geometry, not only different fields or windows."
        ),
    }


def _scope_trouble_state(analysis: Dict[str, Any], quality_thresholds: Dict[str, Any]) -> Dict[str, Any]:
    health = analysis.get("scope_health") if isinstance(analysis.get("scope_health"), dict) else {}
    signals = health.get("trouble_signals") if isinstance(health.get("trouble_signals"), dict) else {}
    failure_streak = int(_float(signals.get("failure_streak")))
    scanned = int(_float(signals.get("scanned_candidates")))
    best_recent_sharpe = _float(health.get("best_recent_sharpe"))
    best_recent_fitness = _float(health.get("best_recent_fitness"))
    required_sharpe = _positive_float(quality_thresholds.get("required_sharpe"))
    required_fitness = _positive_float(quality_thresholds.get("required_fitness"))
    sharpe_ratio = _positive_ratio(best_recent_sharpe, required_sharpe) if required_sharpe else 0.0
    fitness_ratio = _positive_ratio(best_recent_fitness, required_fitness) if required_fitness else 0.0
    active = failure_streak >= SCOPE_TROUBLE_FAILURE_STREAK and scanned >= SCOPE_TROUBLE_MIN_SCANNED
    return {
        "active": bool(active),
        "reason": "LONG_FAILED_STREAK" if active else "",
        "failure_streak": failure_streak,
        "scanned_candidates": scanned,
        "best_recent_sharpe": best_recent_sharpe,
        "best_recent_fitness": best_recent_fitness,
        "required_sharpe": required_sharpe,
        "required_fitness": required_fitness,
        "best_sharpe_ratio": round(sharpe_ratio, 4),
        "best_fitness_ratio": round(fitness_ratio, 4),
        "policy": (
            "When active, stop local optimization of the latest weak/blocked expression and transfer mechanisms "
            "from historical high-signal archetypes to fresh fields."
        ),
    }


def _mechanism_transfer_plan(analysis: Dict[str, Any], scope_trouble: Dict[str, Any]) -> Dict[str, Any]:
    memory = analysis.get("mechanism_memory") if isinstance(analysis.get("mechanism_memory"), dict) else {}
    archetypes = memory.get("archetypes") if isinstance(memory.get("archetypes"), list) else []
    if not archetypes:
        return {}
    forbidden_fields: List[str] = []
    mechanism_tags: List[str] = []
    for field in memory.get("forbidden_fields") or []:
        text = str(field or "").strip()
        if text and text not in forbidden_fields:
            forbidden_fields.append(text)
    for tag in memory.get("mechanism_tags") or []:
        text = str(tag or "").strip()
        if text and text not in mechanism_tags:
            mechanism_tags.append(text)
    return {
        "active": bool(scope_trouble.get("active")),
        "policy": (
            "Use these as mechanism only examples. Do not copy expressions, do not use forbidden_fields as "
            "primary fields, and do not reopen recently submitted or lit-tower routes. Migrate the mechanism to "
            "allowed non-submitted fields."
        ),
        "mechanism_tags": mechanism_tags[:20],
        "forbidden_fields": forbidden_fields[:30],
        "archetypes": archetypes[:6],
    }


def _mechanism_transfer_objective(mechanism_transfer: Dict[str, Any]) -> str:
    if not mechanism_transfer:
        return ""
    tags = [str(tag) for tag in mechanism_transfer.get("mechanism_tags") or [] if str(tag).strip()]
    forbidden = [str(field) for field in mechanism_transfer.get("forbidden_fields") or [] if str(field).strip()]
    tag_note = f" Transfer mechanisms: {', '.join(tags[:6])}." if tags else ""
    forbidden_note = f" Do not copy forbidden fields: {', '.join(forbidden[:8])}." if forbidden else ""
    return tag_note + forbidden_note


def _route_stop_loss_objective(route_stop_loss: Dict[str, Any]) -> str:
    if not route_stop_loss or not route_stop_loss.get("stop_loss_active"):
        return ""
    return (
        " Route stop-loss is active: recent scoped exploration produced no submitable/watchlist progress. "
        "Do not keep local variants of the same route; switch mechanism class and require visibly different "
        "operator geometry."
    )


def _production_rescue_policy(
    route_stop_loss: Dict[str, Any],
    scope_trouble: Dict[str, Any],
    target_settings: Dict[str, Any],
) -> Dict[str, Any]:
    route_active = isinstance(route_stop_loss, dict) and bool(route_stop_loss.get("stop_loss_active"))
    scope_active = isinstance(scope_trouble, dict) and bool(scope_trouble.get("active"))
    if not route_active and not scope_active:
        return {"active": False}
    region = str(target_settings.get("region") or "").upper()
    delay = str(target_settings.get("delay") if target_settings.get("delay") is not None else "")
    usa_d0 = region == "USA" and delay == "0"
    motifs = [
        "ts_backfill",
        "winsorize",
        "ts_rank",
        "group_rank industry/subindustry",
        "cap normalization",
    ] if usa_d0 else ["ts_backfill", "winsorize", "ts_rank", "group_rank"]
    return {
        "active": True,
        "reason": (route_stop_loss.get("reason") if route_active else "") or (
            "route_stop_loss_active" if route_active else "scope_trouble_active"
        ),
        "lit_tower_policy": "soft_tie_breaker",
        "allow_lit_tower_primary_probes": True,
        "scope_trouble_active": bool(scope_trouble.get("active")) if isinstance(scope_trouble, dict) else False,
        "target_scope": {
            "region": target_settings.get("region"),
            "universe": target_settings.get("universe"),
            "delay": target_settings.get("delay"),
            "neutralization": target_settings.get("neutralization"),
        },
        "preferred_motifs": motifs,
        "policy": (
            "After route stop-loss, production quality outranks pyramid novelty. Previously lit towers may be used "
            "as primary probe fields when they have no local failure or metadata block."
        ),
    }


def _production_rescue_objective(
    production_rescue: Dict[str, Any],
    lit_tower_avoidance: Dict[str, Any],
) -> str:
    if not isinstance(production_rescue, dict) or not production_rescue.get("active"):
        return ""
    names = _lit_tower_names(lit_tower_avoidance)
    lit_note = f" Existing lit towers are soft tie-breakers, not hard exclusions: {', '.join(names[:8])}." if names else ""
    motifs = [str(item) for item in production_rescue.get("preferred_motifs") or [] if str(item).strip()]
    motif_note = f" Prefer production-tested probe motifs: {', '.join(motifs[:6])}." if motifs else ""
    return (
        " Production rescue is active: lit towers are soft, quality is the hard gate, and weak unlit novelty must "
        "not outrank stronger field families."
        + lit_note
        + motif_note
    )


def _structure_diversity_objective(structure_diversity: Dict[str, Any]) -> str:
    if not structure_diversity:
        return ""
    max_per = int(_float(structure_diversity.get("max_batch_candidates_per_structure")))
    overused = structure_diversity.get("overused_structures")
    if not isinstance(overused, list):
        overused = []
    cap_note = (
        f" Same-round exploration is capped at {max_per} candidates per formula skeleton."
        if max_per > 0
        else ""
    )
    overused_note = (
        " Avoid recently overused weak formula skeletons; change the operator geometry, not just fields/windows."
        if overused
        else ""
    )
    return cap_note + overused_note


def _lit_tower_objective(lit_tower_avoidance: Dict[str, Any]) -> str:
    names = _lit_tower_names(lit_tower_avoidance)
    if not names:
        return ""
    unlit = [
        str(item.get("name") or "")
        for item in lit_tower_avoidance.get("unlit_towers") or []
        if isinstance(item, dict) and str(item.get("name") or "").strip()
    ]
    suffix = f" Prefer unlit towers such as {', '.join(unlit[:5])}." if unlit else ""
    return (
        " For fresh exploration, avoid already-lit pyramid towers in this region/delay scope: "
        f"{', '.join(names[:8])}.{suffix}"
    )


def _lit_tower_names(lit_tower_avoidance: Dict[str, Any]) -> List[str]:
    names = lit_tower_avoidance.get("tower_names")
    if not isinstance(names, list):
        return []
    result: List[str] = []
    for name in names:
        text = str(name or "").strip()
        if text and text not in result:
            result.append(text)
    return result


def _submitted_avoid_fields(research_context: Dict[str, Any]) -> List[str]:
    avoidance = research_context.get("submitted_field_avoidance")
    if not isinstance(avoidance, dict):
        return []
    fields = avoidance.get("fields")
    if not isinstance(fields, list):
        return []
    result: List[str] = []
    for field in fields:
        text = str(field or "").strip()
        if text and text not in result:
            result.append(text)
    return result


def _candidate_uses_fields(candidate: Dict[str, Any], fields: set[str]) -> bool:
    if not fields:
        return False
    expression = str(candidate.get("expression") or "")
    candidate_fields = set(_extract_fields(expression, []))
    return bool(candidate_fields & fields)


def _family_diversity_objective(family_diversity_control: Dict[str, Any]) -> str:
    if not family_diversity_control:
        return ""
    dominant_family = str(family_diversity_control.get("dominant_family") or "").strip()
    alternate_families = [str(item) for item in family_diversity_control.get("alternate_families") or [] if str(item).strip()]
    if not dominant_family or not alternate_families:
        return ""
    return (
        f" Enforce family split: keep {dominant_family} anchored to one profile at most, "
        f"and route the other profile(s) toward {', '.join(alternate_families[:3])}."
    )


def _family_diversity_control(best: Dict[str, Any], family_stats: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    if not family_stats:
        return {}
    ordered = sorted(
        family_stats.items(),
        key=lambda item: (int(item[1].get("count", 0)), float(item[1].get("avg_sharpe", 0.0)), item[0]),
        reverse=True,
    )
    dominant_family, dominant_stats = ordered[0]
    dominant_count = int(dominant_stats.get("count", 0))
    total = sum(int(stats.get("count", 0)) for stats in family_stats.values())
    if dominant_count < 4 or total <= 0:
        return {}
    dominant_share = dominant_count / total
    if dominant_share < 0.55 or len(ordered) < 2:
        return {}
    alternate_families = [family for family, _ in ordered[1:5]]
    first_best_field = str((best.get("fields") or [""])[0] if best else "")
    return {
        "dominant_family": dominant_family,
        "dominant_share": round(dominant_share, 3),
        "dominant_count": dominant_count,
        "best_family": _field_family(first_best_field) if first_best_field else "",
        "alternate_families": alternate_families,
        "policy": (
            "Keep the dominant family anchored to at most one active profile. "
            "Other profiles should avoid it and use alternate families."
        ),
    }


def _field_family(field: str) -> str:
    text = str(field or "").strip()
    if not text:
        return ""
    if text.lower() in _FAMILY_IGNORE_FIELDS:
        return ""
    parts = [part for part in text.split("_") if part]
    if len(parts) >= 2:
        return "_".join(parts[:2])
    return parts[0]


def _known_fields(research_context: Dict[str, Any]) -> List[str]:
    datafields = research_context.get("datafields")
    if not isinstance(datafields, dict):
        return []
    field_ids = datafields.get("field_ids")
    return [str(field) for field in field_ids] if isinstance(field_ids, list) else []


def _extract_fields(expression: str, known_fields: Iterable[str]) -> List[str]:
    known = {field for field in known_fields if field}
    if known:
        fields: List[str] = []
        for token in re.findall(r"\b[a-zA-Z_][a-zA-Z0-9_]*\b", expression):
            if token in known and token not in fields:
                fields.append(token)
        return fields

    fields: List[str] = []
    for token in re.findall(r"\b[a-zA-Z_][a-zA-Z0-9_]*\b", expression):
        lower = token.lower()
        if lower in ALLOWED_OPERATORS:
            continue
        if lower in _NON_FIELD_TOKENS:
            continue
        if lower in _GROUP_TOKENS:
            continue
        if any(lower.startswith(prefix) for prefix in _OPERATOR_PREFIXES):
            continue
        if token not in fields:
            fields.append(token)
    return fields


def _normalize_reason(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return "UNKNOWN"
    head = text.split(":", 1)[0].strip()
    if head:
        return re.sub(r"[^A-Za-z0-9_]+", "_", head).strip("_").upper() or "UNKNOWN"
    return "UNKNOWN"


def _avoid_list(failure_reasons: List[str], weak_fields: List[str]) -> List[str]:
    prioritized_reasons = _prioritized_failure_reasons(failure_reasons)
    avoid = list(dict.fromkeys(prioritized_reasons[:8] + weak_fields[:16]))
    return avoid[:24] or ["near-duplicates from recent history", "trivial price-volume-only formulas"]


def _prioritized_failure_reasons(failure_reasons: List[str]) -> List[str]:
    unique = list(dict.fromkeys(str(reason) for reason in failure_reasons if str(reason).strip()))
    priority = [reason for reason in ACTIONABLE_FAILURE_PRIORITY if reason in unique]
    return priority + [reason for reason in unique if reason not in set(priority)]


def _setting_variants(target_settings: Dict[str, Any], batch_size: int) -> List[Dict[str, Any]]:
    base = dict(target_settings)
    current_neutralization = str(base.get("neutralization") or "INDUSTRY").upper()
    current_decay = int(_float(base.get("decay")))
    current_truncation = _float(base.get("truncation")) or 0.05
    neutralizations = _preferred_neutralizations(base, current_neutralization)
    variants: List[Dict[str, Any]] = []
    seen = set()

    def add(**updates: Any) -> None:
        variant = dict(base)
        variant.setdefault("neutralization", current_neutralization)
        variant.setdefault("decay", current_decay)
        variant.setdefault("truncation", current_truncation)
        variant.update(updates)
        key = tuple(sorted((str(k), str(v)) for k, v in variant.items()))
        if key in seen:
            return
        seen.add(key)
        variants.append(variant)

    for neutralization in neutralizations:
        add(neutralization=neutralization)
        if len(variants) >= batch_size:
            return variants[:batch_size]
    for decay in [0, 2, 6, 4, 8, 12]:
        if decay == current_decay:
            continue
        add(decay=decay)
        if len(variants) >= batch_size:
            return variants[:batch_size]
    for truncation in [0.01, 0.03, 0.05, 0.08]:
        if truncation == current_truncation:
            continue
        add(truncation=truncation)
        if len(variants) >= batch_size:
            return variants[:batch_size]
    for neutralization in neutralizations:
        for decay in [2, 4, 6, 8]:
            add(neutralization=neutralization, decay=decay)
            if len(variants) >= batch_size:
                return variants[:batch_size]
    return variants[:batch_size]


def _preferred_neutralizations(target_settings: Dict[str, Any], current: str) -> List[str]:
    valid = _valid_neutralizations(target_settings)
    preferred = [current, "SUBINDUSTRY", "INDUSTRY", "SECTOR", "STATISTICAL", "MARKET", "NONE"]
    return [item for item in dict.fromkeys(preferred) if item in valid and item != current]


def _valid_neutralizations(target_settings: Dict[str, Any]) -> set[str]:
    region = str(target_settings.get("region") or "").upper()
    delay = target_settings.get("delay")
    for row in PLATFORM_SCOPE_OPTIONS:
        if str(row.get("region") or "").upper() == region and str(row.get("delay")) == str(delay):
            return {str(item).upper() for item in row.get("neutralizations") or []}
    return {"NONE", "MARKET", "SECTOR", "INDUSTRY", "SUBINDUSTRY", "STATISTICAL"}


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _positive_ratio(current: float, baseline: float) -> float:
    if baseline <= 0:
        return 0.0
    return current / baseline
