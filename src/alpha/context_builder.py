from __future__ import annotations

import ast
import csv
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from .db import AlphaStore
from .expression_similarity import expression_structure_key, expression_variant_key
from .history_prune import DEFAULT_LOW_QUALITY_SCORE_MAX
from .preflight import ALLOWED_OPERATORS
from .research_planner import analyze_research_history, build_experiment_plan, candidate_quality_summary, _extract_fields


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_KNOWLEDGE_DIR = PROJECT_ROOT / "knowledge"
KNOWLEDGE_FILES = (
    "wqb_rules.md",
    "generation_patterns.md",
    "optimization_playbook.md",
    "usa_d0_patterns.md",
)
MAX_TEXT_CHARS = 6000
CANDIDATE_QUEUE_NAMES = ("submitable", "watchlist", "optimize", "trash", "abandoned")
RECENT_CONTEXT_SCAN_LIMIT = 500
HISTORY_MEMORY_SCAN_LIMIT = 300
MECHANISM_MEMORY_LIMIT = 8
TERMINAL_WAIT_CHECKS = {
    "selfcorrelation",
    "prodcorrelation",
    "productcorrelation",
    "datadiversity",
    "regularsubmission",
    "d0submission",
    "powerpoolcorrelation",
}
LIT_TOWER_MIN_ALPHA_COUNT = 3
SUBMISSION_FIELD_AUXILIARIES = {
    "adv20",
    "cap",
    "close",
    "open",
    "high",
    "low",
    "volume",
    "vwap",
    "returns",
}


def build_ai_research_context(
    store: AlphaStore,
    base_context: Dict[str, Any],
    knowledge_dir: str | Path | None = None,
    reference_dir: str | Path | None = None,
    field_catalog: Dict[str, Any] | None = None,
    platform_submissions: List[Dict[str, Any]] | None = None,
    platform_pyramid_alphas: Dict[str, Any] | List[Dict[str, Any]] | None = None,
    platform_pyramid_multipliers: Dict[str, Any] | List[Dict[str, Any]] | None = None,
    history_limit: int = 8,
) -> Dict[str, Any]:
    target_settings = {key: value for key, value in dict(base_context or {}).items() if key != "research_context"}
    resolved_knowledge_dir = (
        Path(knowledge_dir) if knowledge_dir is not None else Path(os.getenv("ALPHA_KNOWLEDGE_DIR", str(DEFAULT_KNOWLEDGE_DIR)))
    )
    resolved_reference_dir = (
        Path(reference_dir) if reference_dir is not None else Path(os.getenv("REFERENCE_BRAIN_DIR", "/root/brain_alpha/Brain"))
    )

    reference_project = _summarize_reference_project(resolved_reference_dir, target_settings, history_limit)
    history_hygiene: Dict[str, Any] = {
        "low_quality_score_max": DEFAULT_LOW_QUALITY_SCORE_MAX,
        "suppressed_low_quality_failures": 0,
        "policy": (
            "Very weak failed candidates are not exposed as full expressions to the AI. They remain available through "
            "aggregate failure, field, and structure statistics so the model learns what failed without copying bad templates."
        ),
    }

    context = {
        "target_settings": target_settings,
        "generation_policy": {
            "complexity": "research_grade",
            "reject_trivial_candidates": True,
            "avoid_trivial_price_volume_only": True,
            "avoid_structural_duplicates": True,
            "avoid_historical_structural_duplicates": True,
            "max_batch_candidates_per_structure": 2,
            "avoid_recent_submitted_fields": True,
            "avoid_lit_pyramid_towers": True,
            "auxiliary_fields_must_not_be_primary": True,
            "prefer_multi_operator_hypotheses": True,
            "require_hypothesis_notes": True,
            "use_failure_history_as_soft_prior": True,
        },
        "knowledge": _load_knowledge(resolved_knowledge_dir),
        "datafields": field_catalog or {"available": False, "field_ids": []},
        "syntax_constraints": _syntax_constraints(store, history_limit),
        "recent_failures": _recent_candidate_summaries(
            store,
            {"failed"},
            history_limit,
            hygiene=history_hygiene,
            suppress_low_quality_failures=True,
        ),
        "recent_pending": _recent_candidate_summaries(store, {"check_pending"}, history_limit),
        "recent_successes": _recent_candidate_summaries(store, {"approved", "submitted"}, history_limit),
        "history_hygiene": history_hygiene,
        "submitted_field_avoidance": _submitted_field_avoidance(
            store,
            target_settings,
            resolved_reference_dir,
            platform_submissions or [],
            _field_ids_from_catalog(field_catalog),
            max(20, history_limit * 2),
        ),
        "lit_tower_avoidance": _lit_tower_avoidance(
            store,
            target_settings,
            platform_submissions or [],
            platform_pyramid_alphas,
            platform_pyramid_multipliers,
            max(20, history_limit * 2),
        ),
        "recent_experiment_plans": _recent_global_events(store, "experiment_plan", max(12, history_limit)),
        "model_feedback": _model_feedback(store, target_settings, history_limit),
        "reference_brain_project": reference_project,
    }
    context["candidate_queues"] = _candidate_queues(store, target_settings, history_limit)
    active_run_started_at = _active_run_started_at(store, target_settings)
    if active_run_started_at:
        context["active_run_candidate_queues"] = _candidate_queues(
            store,
            target_settings,
            history_limit,
            created_since=active_run_started_at,
        )
        context["active_run_history_memory"] = _history_memory(
            store,
            target_settings,
            _field_ids_from_catalog(field_catalog),
            submitted_avoidance=context.get("submitted_field_avoidance"),
            lit_tower_avoidance=context.get("lit_tower_avoidance"),
            created_since=active_run_started_at,
        )
    context["history_memory"] = _history_memory(
        store,
        target_settings,
        _field_ids_from_catalog(field_catalog),
        submitted_avoidance=context.get("submitted_field_avoidance"),
        lit_tower_avoidance=context.get("lit_tower_avoidance"),
    )
    analysis = analyze_research_history(context)
    context["analysis"] = analysis
    context["experiment_plan"] = build_experiment_plan(analysis, target_settings)
    return context


def _syntax_constraints(store: AlphaStore, history_limit: int) -> Dict[str, Any]:
    return {
        "allowed_operators": sorted(ALLOWED_OPERATORS),
        "operator_rule": "Use only operators in allowed_operators. Do not invent operator names.",
        "field_rule": "Copy field identifiers exactly from datafields.field_ids. Do not rename, prefix, suffix, or guess fields.",
        "auxiliary_only_fields": sorted(SUBMISSION_FIELD_AUXILIARIES),
        "auxiliary_field_rule": (
            "auxiliary_only_fields may be used only as helpers such as denominators, scale/liquidity controls, "
            "risk filters, or conditions around non-auxiliary datafields. They must not be the primary alpha "
            "signal, the only fields in an expression, or standalone additive/subtractive legs."
        ),
        "vector_reducer_rule": (
            "vec_avg(x) only, and all vec_* reducers take exactly one VECTOR field argument and no time window. "
            "Apply time-series windows outside the reducer, for example ts_mean(vec_avg(vector_field), 30)."
        ),
        "structure_dedup_rule": (
            "Do not return local variants that reuse the same operator skeleton and same field ids with only "
            "numeric/window changes. Reusing a robust template across genuinely different field families is allowed."
        ),
        "recent_preflight_rejections": _recent_preflight_rejections(store, history_limit),
        "recent_expression_structures": _recent_expression_structures(store, history_limit),
    }


def _recent_preflight_rejections(store: AlphaStore, limit: int) -> Dict[str, List[str]]:
    unknown_operators: List[str] = []
    unknown_fields: List[str] = []
    invalid_patterns: List[str] = []
    fetch_limit = max(200, limit * 50)
    with store.connection() as conn:
        rows = conn.execute(
            """
            SELECT *
            FROM events
            WHERE event_type IN ('preflight_failed', 'status:failed')
            ORDER BY id DESC
            LIMIT ?
            """,
            (fetch_limit,),
        ).fetchall()
    for event in rows:
        metadata = _loads_json(event["metadata_json"])
        if not isinstance(metadata, dict):
            continue
        errors = metadata.get("errors")
        if not isinstance(errors, list):
            continue
        for error in errors:
            text = str(error)
            if text.startswith("UNKNOWN_OPERATOR:"):
                _append_unique(unknown_operators, text.split(":", 1)[1])
            elif text.startswith("UNKNOWN_FIELD:"):
                _append_unique(unknown_fields, text.split(":", 1)[1])
            elif text.startswith("INVALID_"):
                _append_unique(invalid_patterns, text)
        if len(unknown_operators) + len(unknown_fields) + len(invalid_patterns) >= limit:
            break
    return {
        "unknown_operators": unknown_operators[:limit],
        "unknown_fields": unknown_fields[:limit],
        "invalid_patterns": invalid_patterns[:limit],
    }


def _recent_expression_structures(store: AlphaStore, limit: int) -> List[Dict[str, Any]]:
    structures: List[Dict[str, Any]] = []
    seen = set()
    for candidate in store.list_recent_candidates(max(RECENT_CONTEXT_SCAN_LIMIT, limit * 50)):
        if not _candidate_has_simulation_outcome(candidate):
            continue
        expression = str(candidate.get("expression") or "").strip()
        if not expression:
            continue
        structure_key = expression_structure_key(expression)
        variant_key = expression_variant_key(expression)
        if variant_key in seen:
            continue
        seen.add(variant_key)
        structures.append(
            {
                "candidate_id": candidate.get("id"),
                "source": candidate.get("source"),
                "status": candidate.get("status"),
                "structure_key": structure_key,
                "variant_key": variant_key,
                "expression": expression,
            }
        )
        if len(structures) >= limit:
            break
    return structures


def _candidate_has_simulation_outcome(candidate: Dict[str, Any]) -> bool:
    return str(candidate.get("status") or "") in {"approved", "submitted", "failed", "check_pending"}


def _append_unique(items: List[str], value: str) -> None:
    if value and value not in items:
        items.append(value)


def _load_knowledge(knowledge_dir: Path) -> Dict[str, str]:
    sections: Dict[str, str] = {}
    for filename in KNOWLEDGE_FILES:
        path = knowledge_dir / filename
        if path.exists():
            sections[path.stem] = _read_text(path)
    return sections


def _recent_candidate_summaries(
    store: AlphaStore,
    statuses: set[str],
    limit: int,
    *,
    hygiene: Dict[str, Any] | None = None,
    suppress_low_quality_failures: bool = False,
) -> List[Dict[str, Any]]:
    summaries: List[Dict[str, Any]] = []
    for candidate in store.list_recent_candidates(max(RECENT_CONTEXT_SCAN_LIMIT, limit * 30)):
        if candidate["status"] not in statuses:
            continue
        settings = _loads_json(candidate.get("settings_json"))
        metrics = _loads_json(candidate.get("metrics_json"))
        checks = _loads_json(candidate.get("checks_json"))
        quality = candidate_quality_summary(
            {
                "settings": settings if isinstance(settings, dict) else {},
                "metrics": metrics if isinstance(metrics, dict) else {},
                "checks": checks if isinstance(checks, (dict, list)) else {},
            }
        )
        event_summary = _summarize_candidate_events(store, int(candidate["id"]))
        simulation_errors = _simulation_errors(store, int(candidate["id"]))
        if suppress_low_quality_failures and _is_low_quality_failure_context_noise(
            candidate,
            metrics if isinstance(metrics, dict) else {},
            quality,
            event_summary,
            simulation_errors,
        ):
            if hygiene is not None:
                hygiene["suppressed_low_quality_failures"] = int(hygiene.get("suppressed_low_quality_failures") or 0) + 1
            continue
        item = {
            "id": candidate["id"],
            "expression": candidate["expression"],
            "status": candidate["status"],
            "alpha_id": candidate.get("alpha_id"),
            "retry_count": candidate.get("retry_count", 0),
            "settings": settings,
            "metrics": metrics,
            "checks": checks,
        }
        if event_summary:
            item["last_relevant_event"] = event_summary
        if simulation_errors:
            item["simulation_errors"] = simulation_errors
        generated_metadata = _generated_metadata(store, int(candidate["id"]))
        if generated_metadata:
            item["generated_metadata"] = generated_metadata
        summaries.append(item)
        if len(summaries) >= limit:
            break
    return summaries


def _is_low_quality_failure_context_noise(
    candidate: Dict[str, Any],
    metrics: Dict[str, Any],
    quality: Dict[str, Any],
    event_summary: Optional[Dict[str, Any]],
    simulation_errors: List[Dict[str, Any]],
) -> bool:
    if str(candidate.get("status") or "") != "failed":
        return False
    if simulation_errors:
        return False
    event_type = str((event_summary or {}).get("event_type") or "")
    if event_type.startswith("preflight_failed"):
        return False
    history_noise_score = _history_noise_score(metrics, quality)
    if history_noise_score > DEFAULT_LOW_QUALITY_SCORE_MAX:
        return False
    failed_checks = {str(item).split(":", 1)[0].upper() for item in quality.get("failed_checks") or []}
    if failed_checks & {"UNKNOWN_OPERATOR", "UNKNOWN_FIELD", "INVALID_EXPRESSION"}:
        return False
    return True


def _history_noise_score(metrics: Dict[str, Any], quality: Dict[str, Any]) -> float:
    raw_score = _float_or_none(quality.get("raw_score"))
    if raw_score is None or raw_score == 0:
        sharpe = max(0.0, _float_or_none(metrics.get("sharpe")) or 0.0)
        fitness = max(0.0, _float_or_none(metrics.get("fitness")) or 0.0)
        raw_score = sharpe + 0.35 * fitness
    quality_score = _float_or_none(quality.get("quality_score"))
    if quality_score is None:
        return raw_score
    return min(quality_score, raw_score)


def _submitted_field_avoidance(
    store: AlphaStore,
    target_settings: Dict[str, Any],
    reference_dir: Path,
    platform_submissions: List[Dict[str, Any]],
    known_fields: List[str],
    limit: int,
) -> Dict[str, Any]:
    fields: List[str] = []
    families: List[str] = []
    examples: List[Dict[str, Any]] = []
    inspected = 0
    for candidate in store.list_recent_candidates(max(RECENT_CONTEXT_SCAN_LIMIT, limit * 20)):
        if str(candidate.get("status") or "") not in {"approved", "submitted"}:
            continue
        settings = _loads_json(candidate.get("settings_json"))
        if not _candidate_scope_matches(settings, target_settings):
            continue
        inspected += 1
        expression = str(candidate.get("expression") or "")
        candidate_fields = _submitted_fields_from_expression(expression, known_fields)
        if not candidate_fields:
            continue
        for field in candidate_fields:
            _append_unique(fields, field)
            family = _submission_field_family(field)
            if family:
                _append_unique(families, family)
        if len(examples) < limit:
            metrics = _loads_json(candidate.get("metrics_json"))
            examples.append(
                {
                    "id": candidate.get("id"),
                    "status": candidate.get("status"),
                    "source": candidate.get("source"),
                    "fields": candidate_fields,
                    "settings": settings,
                    "metrics": _compact_metrics(metrics),
                    "expression": expression,
                }
            )
        if inspected >= limit:
            break
    for item in _platform_submitted_field_examples(platform_submissions, target_settings, known_fields, limit):
        for field in item["fields"]:
            _append_unique(fields, field)
            family = _submission_field_family(field)
            if family:
                _append_unique(families, family)
        if len(examples) < limit:
            examples.append(item)
    for item in _reference_submitted_field_examples(reference_dir / "submitted_alphas.csv", target_settings, known_fields, limit):
        for field in item["fields"]:
            _append_unique(fields, field)
            family = _submission_field_family(field)
            if family:
                _append_unique(families, family)
        if len(examples) < limit:
            examples.append(item)
    return {
        "fields": fields[:50],
        "families": families[:30],
        "examples": examples,
        "policy": (
            "Do not generate or optimize candidates that reuse core fields from recently approved/submitted "
            "alphas in the same region/delay scope; production/self correlation is likely to be high."
        ),
    }


_PYRAMID_NAME_RE = re.compile(r"\b([A-Z]{2,4})\s*/\s*D([01])\s*/\s*([A-Z][A-Z0-9_]*)\b")


def _lit_tower_avoidance(
    store: AlphaStore,
    target_settings: Dict[str, Any],
    platform_submissions: List[Dict[str, Any]],
    platform_pyramid_alphas: Dict[str, Any] | List[Dict[str, Any]] | None,
    platform_pyramid_multipliers: Dict[str, Any] | List[Dict[str, Any]] | None,
    limit: int,
) -> Dict[str, Any]:
    multiplier_map = _pyramid_multiplier_map(platform_pyramid_multipliers, target_settings)
    lit_by_name: Dict[str, Dict[str, Any]] = {}
    unlit_by_name: Dict[str, Dict[str, Any]] = {}
    examples: List[Dict[str, Any]] = []
    source = "none"

    for entry in _pyramid_entries(platform_pyramid_alphas):
        tower = _pyramid_tower_from_entry(entry, target_settings, multiplier_map, "platform_pyramid_alphas")
        if not tower:
            continue
        source = "platform_pyramid_alphas"
        alpha_count = _int_or_none(entry.get("alphaCount") if isinstance(entry, dict) else None)
        if alpha_count is not None:
            tower["alpha_count"] = alpha_count
        if int(tower.get("alpha_count") or 0) >= LIT_TOWER_MIN_ALPHA_COUNT:
            _merge_tower(lit_by_name, tower)
        else:
            _merge_tower(unlit_by_name, tower)

    lit_towers = sorted(
        lit_by_name.values(),
        key=lambda item: (-_number(item.get("multiplier")), -int(item.get("alpha_count") or 0), item["name"]),
    )
    unlit_towers = [
        tower for name, tower in unlit_by_name.items() if name not in lit_by_name
    ]
    unlit_towers = sorted(
        unlit_towers,
        key=lambda item: (-_number(item.get("multiplier")), item["name"]),
    )
    tower_names = [tower["name"] for tower in lit_towers]
    categories = sorted({str(tower.get("category") or "").upper() for tower in lit_towers if tower.get("category")})
    return {
        "source": source,
        "date_range": _pyramid_date_range(platform_pyramid_alphas),
        "min_alpha_count": LIT_TOWER_MIN_ALPHA_COUNT,
        "source_policy": (
            "Only platform-provided pyramid alpha counts in the active date window are treated as lit tower "
            "evidence. A tower is lit only when alphaCount reaches 3 for the same region/delay/category. "
            "Recent alpha.pyramids, check payloads, field names, and dataset families are not used to infer "
            "lit tower status."
        ),
        "tower_names": tower_names[:50],
        "categories": categories[:50],
        "lit_towers": lit_towers[:50],
        "unlit_towers": unlit_towers[:50],
        "examples": examples[:limit],
        "policy": (
            "For fresh exploration, prefer towers with fewer than 3 submitted alphas in the active quarter and "
            "avoid anchoring new batches on lit towers in the same region/delay scope. This is a soft diversity "
            "rule; do not discard a near-threshold candidate solely because its eventual pyramid may be lit."
        ),
    }


def _pyramid_date_range(payload: Dict[str, Any] | List[Dict[str, Any]] | None) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    query = payload.get("query") if isinstance(payload.get("query"), dict) else {}
    start_date = payload.get("startDate") or payload.get("start_date") or query.get("startDate") or query.get("start_date")
    end_date = payload.get("endDate") or payload.get("end_date") or query.get("endDate") or query.get("end_date")
    return {key: value for key, value in {"start_date": start_date, "end_date": end_date}.items() if value}


def _pyramid_entries(payload: Dict[str, Any] | List[Dict[str, Any]] | None) -> List[Dict[str, Any]]:
    if isinstance(payload, dict):
        rows = payload.get("pyramids")
        return [row for row in rows if isinstance(row, dict)] if isinstance(rows, list) else []
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    return []


def _pyramid_multiplier_map(
    payload: Dict[str, Any] | List[Dict[str, Any]] | None,
    target_settings: Dict[str, Any],
) -> Dict[str, float]:
    multipliers: Dict[str, float] = {}
    for entry in _pyramid_entries(payload):
        tower = _pyramid_tower_from_entry(entry, target_settings, {}, "platform_pyramid_multipliers")
        if not tower:
            continue
        multiplier = _float_or_none(entry.get("multiplier"))
        if multiplier is not None:
            multipliers[tower["name"]] = multiplier
    return multipliers


def _pyramid_tower_from_entry(
    entry: Dict[str, Any],
    target_settings: Dict[str, Any],
    multiplier_map: Dict[str, float],
    source: str,
) -> Dict[str, Any]:
    name = _pyramid_name_from_entry(entry, target_settings)
    if not name:
        return {}
    region, delay, category = _split_pyramid_name(name)
    category_name = ""
    category_payload = entry.get("category")
    if isinstance(category_payload, dict):
        category_name = str(category_payload.get("name") or "").strip()
    multiplier = _float_or_none(entry.get("multiplier"))
    if multiplier is None:
        multiplier = multiplier_map.get(name)
    tower = {
        "name": name,
        "region": region,
        "delay": delay,
        "category": category,
        "category_name": category_name,
        "source": source,
    }
    if multiplier is not None:
        tower["multiplier"] = multiplier
    alpha_count = _int_or_none(entry.get("alphaCount"))
    if alpha_count is not None:
        tower["alpha_count"] = alpha_count
    return tower


def _pyramid_name_from_entry(entry: Dict[str, Any], target_settings: Dict[str, Any]) -> str:
    raw_name = str(entry.get("name") or "").strip()
    if raw_name:
        name = _normalize_pyramid_name(raw_name, target_settings)
        if name:
            return name

    region = entry.get("region")
    delay = entry.get("delay")
    category = _pyramid_category(entry)
    if region in (None, "") or delay in (None, "") or not category:
        return ""
    name = f"{str(region).strip().upper()}/D{int(float(delay))}/{category}"
    return _normalize_pyramid_name(name, target_settings)


def _pyramid_category(entry: Dict[str, Any]) -> str:
    category = entry.get("category")
    if isinstance(category, dict):
        raw = category.get("id") or category.get("name")
    else:
        raw = category or entry.get("categoryId") or entry.get("theme") or entry.get("pyramid")
    text = str(raw or "").strip()
    if not text:
        return ""
    return re.sub(r"[^A-Z0-9_]+", "_", text.upper()).strip("_")


def _normalize_pyramid_name(value: Any, target_settings: Dict[str, Any]) -> str:
    text = str(value or "").strip().upper()
    match = _PYRAMID_NAME_RE.search(text)
    if not match:
        return ""
    region, delay, category = match.group(1).upper(), int(match.group(2)), match.group(3).upper()
    if not _pyramid_scope_matches(region, delay, target_settings):
        return ""
    return f"{region}/D{delay}/{category}"


def _split_pyramid_name(name: str) -> tuple[str, int, str]:
    match = _PYRAMID_NAME_RE.search(str(name or "").upper())
    if not match:
        return "", 0, ""
    return match.group(1).upper(), int(match.group(2)), match.group(3).upper()


def _pyramid_scope_matches(region: str, delay: int, target_settings: Dict[str, Any]) -> bool:
    target_region = str(target_settings.get("region") or "").strip().upper()
    target_delay = _int_or_none(target_settings.get("delay"))
    if target_region and str(region or "").strip().upper() != target_region:
        return False
    if target_delay is not None and int(delay) != target_delay:
        return False
    return True


def _extract_authoritative_pyramid_towers(
    payload: Any,
    target_settings: Dict[str, Any],
    multiplier_map: Dict[str, float],
    source: str,
) -> List[Dict[str, Any]]:
    towers: Dict[str, Dict[str, Any]] = {}

    def visit(value: Any, key_hint: str = "") -> None:
        if isinstance(value, dict):
            direct = _pyramid_tower_from_entry(value, target_settings, multiplier_map, source)
            if direct:
                _merge_tower(towers, direct)
            for key, child in value.items():
                lowered = str(key or "").lower()
                if lowered in {"pyramids", "pyramidthemes", "checks", "is", "os", "prod", "train", "test"}:
                    visit(child, lowered)
                elif "pyramid" in lowered:
                    visit(child, lowered)
        elif isinstance(value, list):
            for item in value:
                visit(item, key_hint)
        elif isinstance(value, str) and ("pyramid" in key_hint or _PYRAMID_NAME_RE.search(value.upper())):
            for match in _PYRAMID_NAME_RE.finditer(value.upper()):
                name = _normalize_pyramid_name(match.group(0), target_settings)
                if not name:
                    continue
                region, delay, category = _split_pyramid_name(name)
                tower = {
                    "name": name,
                    "region": region,
                    "delay": delay,
                    "category": category,
                    "source": source,
                }
                multiplier = multiplier_map.get(name)
                if multiplier is not None:
                    tower["multiplier"] = multiplier
                _merge_tower(towers, tower)

    visit(payload)
    return sorted(towers.values(), key=lambda item: item["name"])


def _merge_tower(towers: Dict[str, Dict[str, Any]], tower: Dict[str, Any]) -> None:
    name = str(tower.get("name") or "").strip()
    if not name:
        return
    existing = towers.setdefault(name, {"name": name})
    existing.update({key: value for key, value in tower.items() if value not in (None, "")})
    existing_alpha_count = _int_or_none(existing.get("alpha_count"))
    incoming_alpha_count = _int_or_none(tower.get("alpha_count"))
    if incoming_alpha_count is not None and (existing_alpha_count is None or incoming_alpha_count > existing_alpha_count):
        existing["alpha_count"] = incoming_alpha_count


def _platform_alpha_is_submitted(alpha: Dict[str, Any]) -> bool:
    stage = str(alpha.get("stage") or alpha.get("status") or "OS").upper()
    return stage in {"OS", "SUBMITTED", "APPROVED"}


def _platform_alpha_settings(alpha: Dict[str, Any]) -> Dict[str, Any]:
    settings = alpha.get("settings") if isinstance(alpha.get("settings"), dict) else {}
    return {
        "region": settings.get("region") or alpha.get("region"),
        "universe": settings.get("universe") or alpha.get("universe"),
        "delay": settings.get("delay") if settings.get("delay") is not None else alpha.get("delay"),
    }


def _int_or_none(value: Any) -> Optional[int]:
    try:
        if value in (None, ""):
            return None
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _float_or_none(value: Any) -> Optional[float]:
    try:
        if value in (None, ""):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _is_submission_avoid_field(field: str) -> bool:
    text = str(field or "").strip()
    return bool(text) and text.lower() not in SUBMISSION_FIELD_AUXILIARIES


def _submission_field_family(field: str) -> str:
    parts = [part for part in str(field or "").split("_") if part]
    if len(parts) >= 2:
        return "_".join(parts[:2])
    return parts[0] if parts else ""


def _platform_submitted_field_examples(
    platform_submissions: List[Dict[str, Any]],
    target_settings: Dict[str, Any],
    known_fields: List[str],
    limit: int,
) -> List[Dict[str, Any]]:
    examples: List[Dict[str, Any]] = []
    for alpha in platform_submissions:
        if not isinstance(alpha, dict):
            continue
        stage = str(alpha.get("stage") or alpha.get("status") or "OS").upper()
        if stage not in {"OS", "SUBMITTED", "APPROVED"}:
            continue
        settings = alpha.get("settings") if isinstance(alpha.get("settings"), dict) else {}
        scoped_settings = {
            "region": settings.get("region") or alpha.get("region"),
            "universe": settings.get("universe") or alpha.get("universe"),
            "delay": settings.get("delay") if settings.get("delay") is not None else alpha.get("delay"),
        }
        if not _candidate_scope_matches(scoped_settings, target_settings):
            continue
        expression = _reference_expression(alpha.get("regular") or alpha.get("expression") or alpha.get("code") or "")
        candidate_fields = _submitted_fields_from_expression(expression, known_fields)
        if not candidate_fields:
            continue
        is_metrics = alpha.get("is") if isinstance(alpha.get("is"), dict) else {}
        examples.append(
            {
                "id": alpha.get("id") or alpha.get("alpha_id"),
                "status": "submitted",
                "source": "platform_os_alphas",
                "fields": candidate_fields,
                "settings": scoped_settings,
                "metrics": _compact_metrics(
                    {
                        "sharpe": alpha.get("sharpe", is_metrics.get("sharpe")),
                        "fitness": alpha.get("fitness", is_metrics.get("fitness")),
                        "returns": alpha.get("returns", is_metrics.get("returns")),
                        "margin": alpha.get("margin", is_metrics.get("margin")),
                    }
                ),
                "expression": expression,
            }
        )
        if len(examples) >= limit:
            break
    return examples


def _reference_submitted_field_examples(
    path: Path,
    target_settings: Dict[str, Any],
    known_fields: List[str],
    limit: int,
) -> List[Dict[str, Any]]:
    rows = _read_csv(path)
    examples: List[Dict[str, Any]] = []
    for row in reversed(rows):
        if str(row.get("status") or "").upper() not in {"SUBMITTED", "OS", "APPROVED"}:
            continue
        settings = {
            "region": row.get("region"),
            "universe": row.get("universe"),
            "delay": row.get("delay"),
        }
        if not _candidate_scope_matches(settings, target_settings):
            continue
        expression = _reference_expression(row.get("expression") or row.get("regular") or "")
        candidate_fields = _submitted_fields_from_expression(expression, known_fields)
        if not candidate_fields:
            continue
        examples.append(
            {
                "id": row.get("alpha_id"),
                "status": "submitted",
                "source": "reference_submitted_alphas",
                "fields": candidate_fields,
                "settings": settings,
                "metrics": _compact_metrics(
                    {
                        "sharpe": row.get("sharpe"),
                        "fitness": row.get("fitness"),
                        "returns": row.get("returns"),
                        "margin": row.get("margin"),
                    }
                ),
                "expression": expression,
            }
        )
        if len(examples) >= limit:
            break
    return examples


def _reference_expression(value: Any) -> str:
    if isinstance(value, dict):
        code = value.get("code") or value.get("regular") or value.get("expression")
        return str(code or "").strip()
    text = str(value or "").strip()
    if not text:
        return ""
    if text.startswith("{"):
        data: Any = _loads_json(text)
        if not isinstance(data, dict):
            try:
                data = ast.literal_eval(text)
            except (ValueError, SyntaxError):
                data = {}
        if isinstance(data, dict):
            code = data.get("code") or data.get("regular") or data.get("expression")
            if code:
                return str(code).strip()
    return text


def _submitted_fields_from_expression(expression: str, known_fields: List[str]) -> List[str]:
    fields = _extract_fields(expression, known_fields) if known_fields else []
    if not fields:
        fields = _extract_fields(expression, [])
    return [field for field in fields if _is_submission_avoid_field(field)]


def _field_ids_from_catalog(field_catalog: Dict[str, Any] | None) -> List[str]:
    if not isinstance(field_catalog, dict):
        return []
    field_ids = field_catalog.get("field_ids")
    return [str(field) for field in field_ids] if isinstance(field_ids, list) else []


def _simulation_errors(store: AlphaStore, candidate_id: int) -> List[Dict[str, Any]]:
    errors: List[Dict[str, Any]] = []
    for event in store.events_for_candidate(candidate_id):
        if event["event_type"] == "simulation_error":
            metadata = _loads_json(event.get("metadata_json"))
            if isinstance(metadata, dict):
                errors.append(metadata)
    return errors[-3:]


def _generated_metadata(store: AlphaStore, candidate_id: int) -> Dict[str, Any]:
    for event in store.events_for_candidate(candidate_id):
        if event["event_type"] == "generated":
            metadata = _loads_json(event.get("metadata_json"))
            if isinstance(metadata, dict):
                ai_metadata = metadata.get("ai_metadata")
                return ai_metadata if isinstance(ai_metadata, dict) else metadata
    return {}


def _model_feedback(store: AlphaStore, target_settings: Dict[str, Any], limit: int) -> Dict[str, Any]:
    by_profile: Dict[str, Dict[str, Any]] = {}
    recent: List[Dict[str, Any]] = []
    inspected = 0
    for candidate in store.list_recent_candidates(max(RECENT_CONTEXT_SCAN_LIMIT, limit * 30)):
        settings = _loads_json(candidate.get("settings_json"))
        if not _candidate_scope_matches(settings, target_settings):
            continue
        candidate_id = int(candidate["id"])
        generated_metadata = _generated_metadata(store, candidate_id)
        profile = str(generated_metadata.get("model_profile") or _profile_from_source(candidate.get("source")) or "").strip()
        if not profile:
            continue
        inspected += 1
        status = str(candidate.get("status") or "")
        feedback = by_profile.setdefault(
            profile,
            {
                "generated": 0,
                "approved": 0,
                "submitted": 0,
                "failed": 0,
                "check_pending": 0,
                "validator_rejected": 0,
                "preflight_failed": 0,
                "simulation_error": 0,
                "reasons": {},
                "guidance": {},
                "examples": [],
            },
        )
        feedback["generated"] += 1
        if status in {"approved", "submitted", "failed", "check_pending"}:
            feedback[status] += 1
        guidance = generated_metadata.get("profile_guidance")
        if isinstance(guidance, dict) and guidance and not feedback["guidance"]:
            feedback["guidance"] = guidance
        reason = _candidate_feedback_reason(store, candidate_id, feedback)
        if reason:
            reasons = feedback["reasons"]
            reasons[reason] = int(reasons.get(reason, 0)) + 1
        if len(feedback["examples"]) < 3:
            metrics = _loads_json(candidate.get("metrics_json"))
            checks = _loads_json(candidate.get("checks_json"))
            quality = candidate_quality_summary(
                {
                    "settings": settings,
                    "metrics": metrics if isinstance(metrics, dict) else {},
                    "checks": checks if isinstance(checks, (dict, list)) else {},
                }
            )
            feedback["examples"].append(
                {
                    "id": candidate_id,
                    "status": status,
                    "expression": candidate.get("expression"),
                    "reason": reason,
                    "sharpe": metrics.get("sharpe") if isinstance(metrics, dict) else None,
                    "fitness": metrics.get("fitness") if isinstance(metrics, dict) else None,
                    "returns": metrics.get("returns") if isinstance(metrics, dict) else None,
                    "turnover": metrics.get("turnover") if isinstance(metrics, dict) else None,
                    "drawdown": metrics.get("drawdown") if isinstance(metrics, dict) else None,
                    "quality_score": quality.get("quality_score"),
                    "readiness_score": quality.get("readiness_score"),
                    "failed_checks": quality.get("failed_checks"),
                    "warning_checks": quality.get("warning_checks"),
                }
            )
        if len(recent) < limit:
            recent.append(
                {
                    "id": candidate_id,
                    "profile": profile,
                    "status": status,
                    "reason": reason,
                    "guidance": guidance if isinstance(guidance, dict) else {},
                }
            )
        if inspected >= limit * 4:
            break
    return {"by_profile": by_profile, "recent": recent}


def _history_memory(
    store: AlphaStore,
    target_settings: Dict[str, Any],
    known_fields: List[str],
    submitted_avoidance: Dict[str, Any] | None = None,
    lit_tower_avoidance: Dict[str, Any] | None = None,
    created_since: str | None = None,
) -> Dict[str, Any]:
    status_counts: Dict[str, int] = {}
    failure_reasons: Dict[str, int] = {}
    field_stats: Dict[str, Dict[str, Any]] = {}
    family_stats: Dict[str, Dict[str, Any]] = {}
    structure_stats: Dict[str, Dict[str, Any]] = {}
    profile_outcomes: Dict[str, Dict[str, int]] = {}
    blocked_winner_archetypes: List[Dict[str, Any]] = []
    scanned = 0
    failure_streak = 0
    streak_open = True
    best_recent_sharpe: float | None = None
    best_recent_fitness: float | None = None
    best_recent_quality_score: float | None = None
    submitted_fields = {
        str(field)
        for field in (submitted_avoidance or {}).get("fields", [])
        if str(field).strip()
    }
    submitted_families = {
        str(family)
        for family in (submitted_avoidance or {}).get("families", [])
        if str(family).strip()
    }

    candidates = (
        list(reversed(store.list_candidates(created_since=created_since)))
        if created_since
        else store.list_recent_candidates(HISTORY_MEMORY_SCAN_LIMIT)
    )
    for candidate in candidates:
        settings = _loads_json(candidate.get("settings_json"))
        if not _candidate_scope_matches(settings, target_settings):
            continue
        scanned += 1
        status = str(candidate.get("status") or "unknown")
        if streak_open and status == "failed":
            failure_streak += 1
        elif status in {"approved", "submitted", "check_pending", "failed"}:
            streak_open = False
        status_counts[status] = int(status_counts.get(status, 0)) + 1
        metrics = _loads_json(candidate.get("metrics_json"))
        checks = _loads_json(candidate.get("checks_json"))
        quality = candidate_quality_summary(
            {
                "settings": settings,
                "metrics": metrics if isinstance(metrics, dict) else {},
                "checks": checks if isinstance(checks, (dict, list)) else {},
            }
        )
        expression = str(candidate.get("expression") or "")
        structure_key = expression_structure_key(expression)
        fields = _extract_fields(expression, known_fields)
        profile = _profile_from_source(candidate.get("source")) or "unknown"
        best_recent_sharpe = _max_number(best_recent_sharpe, metrics.get("sharpe") if isinstance(metrics, dict) else None)
        best_recent_fitness = _max_number(best_recent_fitness, metrics.get("fitness") if isinstance(metrics, dict) else None)
        best_recent_quality_score = _max_number(best_recent_quality_score, quality.get("quality_score"))
        profile_bucket = profile_outcomes.setdefault(
            profile,
            {"generated": 0, "approved": 0, "submitted": 0, "failed": 0, "check_pending": 0},
        )
        profile_bucket["generated"] += 1
        if status in profile_bucket:
            profile_bucket[status] += 1

        if status == "failed":
            for reason in _failure_reason_names(store, int(candidate["id"]), quality):
                failure_reasons[reason] = int(failure_reasons.get(reason, 0)) + 1

        for field in fields:
            _update_memory_bucket(field_stats.setdefault(field, _new_memory_bucket(field)), status, metrics, quality)
            family = _submission_field_family(field)
            _update_memory_bucket(family_stats.setdefault(family, _new_memory_bucket(family)), status, metrics, quality)
        _update_structure_bucket(
            structure_stats.setdefault(structure_key, _new_structure_bucket(structure_key, expression)),
            status,
            metrics,
            quality,
        )

        archetype = _blocked_winner_archetype(
            store,
            candidate,
            settings,
            status,
            expression,
            fields,
            metrics if isinstance(metrics, dict) else {},
            quality,
            submitted_fields,
            submitted_families,
            lit_tower_avoidance or {},
        )
        if archetype:
            blocked_winner_archetypes.append(archetype)

    return {
        "policy": (
            "Compressed long-run memory. Use this for broad field, family, failure, and model-profile priors; "
            "use recent_* and candidate_queues only for fresh concrete examples."
        ),
        "scan_limit": HISTORY_MEMORY_SCAN_LIMIT,
        "created_since": created_since,
        "scanned_candidates": scanned,
        "status_counts": status_counts,
        "top_fields": _top_memory_buckets(field_stats, 30),
        "top_field_families": _top_memory_buckets(family_stats, 20),
        "top_structures": _top_structure_buckets(structure_stats, 20),
        "top_failure_reasons": [
            {"reason": reason, "count": count}
            for reason, count in sorted(failure_reasons.items(), key=lambda item: (-item[1], item[0]))[:20]
        ],
        "profile_outcomes": profile_outcomes,
        "scope_health": {
            "best_recent_sharpe": best_recent_sharpe,
            "best_recent_fitness": best_recent_fitness,
            "best_recent_quality_score": best_recent_quality_score,
            "trouble_signals": {
                "failure_streak": failure_streak,
                "scanned_candidates": scanned,
                "failed_count": int(status_counts.get("failed") or 0),
                "approved_count": int(status_counts.get("approved") or 0),
                "submitted_count": int(status_counts.get("submitted") or 0),
                "check_pending_count": int(status_counts.get("check_pending") or 0),
            },
        },
        "blocked_winner_archetypes": _top_blocked_winner_archetypes(blocked_winner_archetypes, MECHANISM_MEMORY_LIMIT),
    }


def _blocked_winner_archetype(
    store: AlphaStore,
    candidate: Dict[str, Any],
    settings: Dict[str, Any],
    status: str,
    expression: str,
    fields: List[str],
    metrics: Dict[str, Any],
    quality: Dict[str, Any],
    submitted_fields: set[str],
    submitted_families: set[str],
    lit_tower_avoidance: Dict[str, Any],
) -> Dict[str, Any]:
    if not _is_high_signal_candidate(status, metrics, quality):
        return {}
    field_set = set(fields)
    families = [_submission_field_family(field) for field in fields if _submission_field_family(field)]
    submitted_hits = sorted(field_set & submitted_fields)
    submitted_family_hits = sorted({family for family in families if family in submitted_families})
    non_auxiliary_fields = [field for field in fields if field.lower() not in SUBMISSION_FIELD_AUXILIARIES]
    auxiliary_primary_only = bool(fields) and not non_auxiliary_fields
    reason_names = _failure_reason_names(store, int(candidate["id"]), quality)
    terminal_reasons = sorted(
        reason
        for reason in reason_names
        if _normalize_check_key(reason) in TERMINAL_WAIT_CHECKS
        or reason in {"SELF_CORRELATION", "PROD_CORRELATION", "PRODUCT_CORRELATION", "DATA_DIVERSITY", "REGULAR_SUBMISSION"}
    )
    blocked_by: List[str] = []
    if status in {"approved", "submitted"} or submitted_hits:
        blocked_by.append("submitted_field_avoidance")
    if submitted_family_hits:
        blocked_by.append("submitted_family_avoidance")
    if auxiliary_primary_only:
        blocked_by.append("auxiliary_primary_only")
    if terminal_reasons:
        blocked_by.append("terminal_or_correlation_check")
    if _lit_tower_names_from_avoidance(lit_tower_avoidance) and auxiliary_primary_only:
        blocked_by.append("lit_tower_soft_policy")
    if not blocked_by:
        return {}

    forbidden_fields = list(dict.fromkeys([*submitted_hits, *(fields if auxiliary_primary_only else [])]))
    return {
        "id": candidate.get("id"),
        "status": status,
        "source": candidate.get("source"),
        "settings": {key: settings.get(key) for key in ("region", "universe", "delay", "neutralization") if key in settings},
        "metrics": _compact_metrics(metrics),
        "quality_score": quality.get("quality_score"),
        "readiness_score": quality.get("readiness_score"),
        "fields": fields[:10],
        "families": list(dict.fromkeys(families))[:8],
        "blocked_by": blocked_by,
        "forbidden_fields": forbidden_fields[:12],
        "mechanism_tags": _mechanism_tags(expression, fields),
        "transfer_hint": _mechanism_transfer_hint(expression, fields),
        "policy": "Use as mechanism only. Do not copy this expression or its forbidden_fields.",
        "expression": expression,
    }


def _is_high_signal_candidate(status: str, metrics: Dict[str, Any], quality: Dict[str, Any]) -> bool:
    if status in {"approved", "submitted"}:
        return True
    sharpe = _number(metrics.get("sharpe"))
    fitness = _number(metrics.get("fitness"))
    readiness = _number(quality.get("readiness_score"))
    quality_score = _number(quality.get("quality_score"))
    return sharpe >= 1.4 or fitness >= 0.7 or readiness >= 0.65 or quality_score >= 0.45


def _mechanism_tags(expression: str, fields: List[str]) -> List[str]:
    lowered_expression = expression.lower()
    lowered_fields = {field.lower() for field in fields}
    tags: List[str] = []

    def add(tag: str) -> None:
        if tag not in tags:
            tags.append(tag)

    if {"vwap", "close"} <= lowered_fields or "divide(vwap,close)" in re.sub(r"\s+", "", lowered_expression):
        add("relative_price_deviation")
    if any(field.startswith("analyst") for field in lowered_fields) or any(field.startswith("est_") for field in lowered_fields):
        add("estimate_revision_or_dispersion")
    if "cap" in lowered_fields and "divide" in lowered_expression:
        add("scale_normalized_signal")
    if "ts_mean" in lowered_expression or "ts_decay_linear" in lowered_expression:
        add("medium_horizon_smoothing")
    if "ts_rank" in lowered_expression:
        add("time_series_persistence_rank")
    if "group_rank" in lowered_expression or "group_zscore" in lowered_expression:
        add("cross_sectional_industry_relative")
    if "multiply" in lowered_expression:
        add("confirmation_gate")
    if "subtract" in lowered_expression:
        add("spread_or_differential")
    return tags or ["field_native_signal"]


def _mechanism_transfer_hint(expression: str, fields: List[str]) -> str:
    tags = _mechanism_tags(expression, fields)
    hints = {
        "relative_price_deviation": "migrate relative-deviation logic to non-PV primary fields; keep price/volume only as helpers",
        "estimate_revision_or_dispersion": "search adjacent non-submitted estimate, revision, breadth, or dispersion fields",
        "scale_normalized_signal": "preserve scale-aware normalization with cap or liquidity as helper only",
        "medium_horizon_smoothing": "prefer smoother 22/33/63/120 day persistence instead of short-horizon churn",
        "time_series_persistence_rank": "test persistent ranks rather than one-day deltas",
        "cross_sectional_industry_relative": "keep group-relative ranking or z-scoring for region robustness",
        "confirmation_gate": "use soft confirmation legs rather than sharp multiplicative gates",
        "spread_or_differential": "only use spreads when the two legs have clearly different economic meaning",
    }
    selected = [hints[tag] for tag in tags if tag in hints]
    return "; ".join(selected[:4]) or "reuse the economic mechanism on allowed non-submitted fields"


def _top_blocked_winner_archetypes(items: List[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
    seen = set()
    unique: List[Dict[str, Any]] = []
    for item in sorted(items, key=_blocked_winner_sort_key, reverse=True):
        key = tuple(item.get("mechanism_tags") or []) + tuple(item.get("forbidden_fields") or [])
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
        if len(unique) >= limit:
            break
    return unique


def _blocked_winner_sort_key(item: Dict[str, Any]) -> tuple[float, float, float, int]:
    metrics = item.get("metrics") if isinstance(item.get("metrics"), dict) else {}
    return (
        _number(item.get("quality_score")),
        _number(metrics.get("sharpe")),
        _number(metrics.get("fitness")),
        int(item.get("id") or 0),
    )


def _max_number(current: float | None, value: Any) -> float | None:
    number = _float_or_none(value)
    if number is None:
        return current
    if current is None or number > current:
        return number
    return current


def _lit_tower_names_from_avoidance(avoidance: Dict[str, Any]) -> List[str]:
    names = avoidance.get("tower_names")
    if not isinstance(names, list):
        return []
    return [str(name) for name in names if str(name).strip()]


def _new_memory_bucket(name: str) -> Dict[str, Any]:
    return {
        "name": name,
        "count": 0,
        "approved": 0,
        "submitted": 0,
        "check_pending": 0,
        "failed": 0,
        "best_sharpe": None,
        "best_fitness": None,
        "best_quality_score": None,
    }


def _new_structure_bucket(key: str, expression: str) -> Dict[str, Any]:
    return {
        "structure_key": key,
        "example_expression": _compact_text(expression, 180),
        "count": 0,
        "approved": 0,
        "submitted": 0,
        "check_pending": 0,
        "failed": 0,
        "best_sharpe": None,
        "best_fitness": None,
        "best_quality_score": None,
        "total_sharpe": 0.0,
        "sharpe_count": 0,
    }


def _update_memory_bucket(
    bucket: Dict[str, Any],
    status: str,
    metrics: Any,
    quality: Dict[str, Any],
) -> None:
    bucket["count"] = int(bucket.get("count") or 0) + 1
    if status in {"approved", "submitted", "check_pending", "failed"}:
        bucket[status] = int(bucket.get(status) or 0) + 1
    if isinstance(metrics, dict):
        _update_best_number(bucket, "best_sharpe", metrics.get("sharpe"))
        _update_best_number(bucket, "best_fitness", metrics.get("fitness"))
    _update_best_number(bucket, "best_quality_score", quality.get("quality_score"))


def _update_structure_bucket(
    bucket: Dict[str, Any],
    status: str,
    metrics: Any,
    quality: Dict[str, Any],
) -> None:
    bucket["count"] = int(bucket.get("count") or 0) + 1
    if status in {"approved", "submitted", "check_pending", "failed"}:
        bucket[status] = int(bucket.get(status) or 0) + 1
    if isinstance(metrics, dict):
        sharpe = _float_or_none(metrics.get("sharpe"))
        if sharpe is not None:
            bucket["total_sharpe"] = float(bucket.get("total_sharpe") or 0.0) + sharpe
            bucket["sharpe_count"] = int(bucket.get("sharpe_count") or 0) + 1
        _update_best_number(bucket, "best_sharpe", metrics.get("sharpe"))
        _update_best_number(bucket, "best_fitness", metrics.get("fitness"))
    _update_best_number(bucket, "best_quality_score", quality.get("quality_score"))


def _update_best_number(bucket: Dict[str, Any], key: str, value: Any) -> None:
    number = _float_or_none(value)
    if number is None:
        return
    current = _float_or_none(bucket.get(key))
    if current is None or number > current:
        bucket[key] = number


def _top_memory_buckets(buckets: Dict[str, Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
    rows = sorted(
        buckets.values(),
        key=lambda item: (
            _float_or_none(item.get("best_quality_score")) or -999.0,
            _float_or_none(item.get("best_sharpe")) or -999.0,
            int(item.get("count") or 0),
        ),
        reverse=True,
    )
    compact_rows: List[Dict[str, Any]] = []
    for item in rows[:limit]:
        row = dict(item)
        row["field"] = row.pop("name")
        compact_rows.append({key: value for key, value in row.items() if value is not None})
    return compact_rows


def _top_structure_buckets(buckets: Dict[str, Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for bucket in buckets.values():
        count = int(bucket.get("count") or 0)
        sharpe_count = int(bucket.get("sharpe_count") or 0)
        avg_sharpe = (float(bucket.get("total_sharpe") or 0.0) / sharpe_count) if sharpe_count else None
        rows.append(
            {
                "structure_key": bucket.get("structure_key"),
                "example_expression": bucket.get("example_expression"),
                "count": count,
                "failed": int(bucket.get("failed") or 0),
                "approved": int(bucket.get("approved") or 0),
                "submitted": int(bucket.get("submitted") or 0),
                "check_pending": int(bucket.get("check_pending") or 0),
                "best_sharpe": bucket.get("best_sharpe"),
                "best_fitness": bucket.get("best_fitness"),
                "best_quality_score": bucket.get("best_quality_score"),
                "avg_sharpe": round(avg_sharpe, 6) if avg_sharpe is not None else None,
                "failure_rate": round((int(bucket.get("failed") or 0) / count), 6) if count else None,
            }
        )
    rows.sort(
        key=lambda item: (
            int(item.get("count") or 0),
            int(item.get("failed") or 0),
            _float_or_none(item.get("best_quality_score")) or -999.0,
        ),
        reverse=True,
    )
    return rows[:limit]


def _compact_text(value: Any, limit: int) -> str:
    text = str(value or "")
    return text if len(text) <= limit else text[: max(0, limit - 3)] + "..."


def _failure_reason_names(store: AlphaStore, candidate_id: int, quality: Dict[str, Any]) -> List[str]:
    reasons: List[str] = []
    for event in reversed(store.events_for_candidate(candidate_id)):
        metadata = _loads_json(event.get("metadata_json"))
        if not isinstance(metadata, dict):
            continue
        errors = metadata.get("errors")
        if isinstance(errors, list):
            reasons.extend(str(error).split(":", 1)[0] for error in errors if str(error).strip())
        elif metadata.get("reason"):
            reasons.append(str(metadata["reason"]))
        if reasons:
            return list(dict.fromkeys(reasons))
    reasons = [str(reason).split(":", 1)[0] for reason in quality.get("failed_checks") or [] if str(reason).strip()]
    if reasons:
        return list(dict.fromkeys(reasons))
    return ["UNKNOWN"]


def _candidate_queues(
    store: AlphaStore,
    target_settings: Dict[str, Any],
    limit: int,
    created_since: str | None = None,
) -> Dict[str, Any]:
    queues: Dict[str, Any] = {name: [] for name in CANDIDATE_QUEUE_NAMES}
    abandoned_ids = _abandoned_candidate_ids(store, target_settings)
    inspected = 0
    max_inspected = max(RECENT_CONTEXT_SCAN_LIMIT, limit * 10)
    candidates = (
        list(reversed(store.list_candidates(created_since=created_since)))
        if created_since
        else store.list_recent_candidates(max_inspected)
    )
    for candidate in candidates:
        settings = _loads_json(candidate.get("settings_json"))
        if not _candidate_scope_matches(settings, target_settings):
            continue
        status = str(candidate.get("status") or "")
        if status not in {"approved", "submitted", "check_pending", "failed"}:
            continue
        item = _candidate_queue_item(store, candidate, settings)
        queue, reason = _classify_candidate_queue(item, abandoned_ids)
        item["queue"] = queue
        item["queue_reason"] = reason
        if len(queues[queue]) < limit:
            queues[queue].append(item)
        inspected += 1
        if inspected >= max_inspected:
            break

    for name in CANDIDATE_QUEUE_NAMES:
        queues[name] = sorted(queues[name], key=_candidate_queue_sort_key, reverse=True)
    queues["counts"] = {name: len(queues[name]) for name in CANDIDATE_QUEUE_NAMES}
    queues["counts"]["total"] = sum(int(queues["counts"][name]) for name in CANDIDATE_QUEUE_NAMES)
    if created_since:
        queues["created_since"] = created_since
    return queues


def _active_run_started_at(store: AlphaStore, target_settings: Dict[str, Any]) -> str:
    state = store.get_run_state("daemon")
    started_at = str(state.get("started_at") or "").strip()
    if not started_at:
        return ""
    scope = state.get("scope")
    if isinstance(scope, dict) and not _candidate_scope_matches(scope, target_settings):
        return ""
    return started_at


def _candidate_queue_item(store: AlphaStore, candidate: Dict[str, Any], settings: Dict[str, Any]) -> Dict[str, Any]:
    raw_metrics = _loads_json(candidate.get("metrics_json"))
    raw_checks = _loads_json(candidate.get("checks_json"))
    metrics = _compact_metrics(raw_metrics)
    checks = _compact_checks(raw_checks)
    quality = candidate_quality_summary(
        {
            "settings": settings,
            "metrics": raw_metrics if isinstance(raw_metrics, dict) else {},
            "checks": raw_checks if isinstance(raw_checks, (dict, list)) else {},
        }
    )
    item = {
        "id": candidate["id"],
        "expression": candidate["expression"],
        "status": candidate["status"],
        "source": candidate.get("source"),
        "alpha_id": candidate.get("alpha_id"),
        "retry_count": candidate.get("retry_count", 0),
        "settings": settings,
        "metrics": metrics,
        "checks": checks,
        "sharpe": metrics.get("sharpe"),
        "fitness": metrics.get("fitness"),
        "turnover": metrics.get("turnover"),
        "quality_score": quality.get("quality_score"),
        "readiness_score": quality.get("readiness_score"),
        "failed_checks": quality.get("failed_checks") or [],
        "warning_checks": quality.get("warning_checks") or [],
        "pending_checks": quality.get("pending_checks") or [],
        "passed_checks": quality.get("passed_checks") or [],
    }
    generated_metadata = _generated_metadata(store, int(candidate["id"]))
    if generated_metadata:
        item["generated_metadata"] = generated_metadata
    event_summary = _summarize_candidate_events(store, int(candidate["id"]))
    if event_summary:
        item["last_relevant_event"] = event_summary
    return item


def _compact_metrics(metrics: Any) -> Dict[str, Any]:
    if not isinstance(metrics, dict):
        return {}
    keys = (
        "sharpe",
        "fitness",
        "returns",
        "turnover",
        "drawdown",
        "margin",
        "pnl",
        "longCount",
        "shortCount",
    )
    compact = {key: metrics[key] for key in keys if key in metrics}
    investability = metrics.get("investabilityConstrained")
    if isinstance(investability, dict):
        for key in ("sharpe", "fitness", "returns", "turnover", "drawdown"):
            if key in investability:
                compact[f"investability_{key}"] = investability[key]
    return compact


def _compact_checks(checks: Any) -> Dict[str, Dict[str, Any]]:
    compact: Dict[str, Dict[str, Any]] = {}
    if isinstance(checks, dict):
        source = checks.items()
    elif isinstance(checks, list):
        source = ((item.get("name"), item) for item in checks if isinstance(item, dict) and item.get("name"))
    else:
        return compact
    for raw_name, data in source:
        if not isinstance(data, dict):
            continue
        name = str(raw_name or "").strip()
        if not name:
            continue
        item = {}
        status = data.get("status", data.get("result"))
        if status not in (None, ""):
            item["status"] = status
        for key in ("value", "limit", "year"):
            if key in data:
                item[key] = data[key]
        compact[name] = item
    return compact


def _classify_candidate_queue(item: Dict[str, Any], abandoned_ids: set[int]) -> tuple[str, str]:
    status = str(item.get("status") or "")
    candidate_id = int(item.get("id") or 0)
    if status in {"approved", "submitted"}:
        return "submitable", "approved_or_submitted"
    if candidate_id in abandoned_ids:
        return "abandoned", "optimization_limit_exhausted"
    if _terminal_checks_waiting(item):
        return "watchlist", "terminal_checks_waiting"
    if _candidate_is_optimizable(item):
        return "optimize", "near_threshold_or_promising"
    return "trash", "low_quality_or_blocked"


def _terminal_checks_waiting(item: Dict[str, Any]) -> bool:
    if item.get("failed_checks") or item.get("warning_checks"):
        return False
    pending = [_normalize_check_key(name) for name in item.get("pending_checks") or []]
    return bool(pending) and all(name in TERMINAL_WAIT_CHECKS for name in pending)


def _candidate_is_optimizable(item: Dict[str, Any]) -> bool:
    metrics = item.get("metrics") if isinstance(item.get("metrics"), dict) else {}
    if "sharpe" not in metrics and "fitness" not in metrics:
        return False
    failed = {_normalize_check_key(name) for name in item.get("failed_checks") or []}
    hard_blockers = {
        "highturnover",
        "lowturnover",
        "concentratedweight",
        "prodcorrelation",
        "productcorrelation",
        "selfcorrelation",
        "datadiversity",
        "regularsubmission",
        "d0submission",
    }
    if failed & hard_blockers:
        return False
    readiness = _number(item.get("readiness_score"))
    quality = _number(item.get("quality_score"))
    sharpe = _number(metrics.get("sharpe"))
    fitness = _number(metrics.get("fitness"))
    return readiness >= 0.45 or quality >= 0.0 or sharpe >= 1.0 or fitness >= 0.35


def _candidate_queue_sort_key(item: Dict[str, Any]) -> tuple[float, float, float, int]:
    return (
        _number(item.get("quality_score")),
        _number(item.get("sharpe")),
        _number(item.get("fitness")),
        int(item.get("id") or 0),
    )


def _abandoned_candidate_ids(store: AlphaStore, target_settings: Dict[str, Any]) -> set[int]:
    abandoned: set[int] = set()
    for event in reversed(store.events_for_candidate(None)):
        if event["event_type"] != "experiment_plan":
            continue
        metadata = _loads_json(event.get("metadata_json"))
        if not isinstance(metadata, dict):
            continue
        plan_settings = metadata.get("target_settings")
        if isinstance(plan_settings, dict) and not _candidate_scope_matches(plan_settings, target_settings):
            continue
        target_id = metadata.get("abandoned_target_id")
        if target_id in (None, ""):
            continue
        try:
            abandoned.add(int(target_id))
        except (TypeError, ValueError):
            continue
    return abandoned


def _normalize_check_key(value: Any) -> str:
    return "".join(ch for ch in str(value or "").lower() if ch.isalnum())


def _number(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return -999.0


def _candidate_feedback_reason(store: AlphaStore, candidate_id: int, feedback: Dict[str, Any]) -> str:
    for event in reversed(store.events_for_candidate(candidate_id)):
        event_type = str(event.get("event_type") or "")
        metadata = _loads_json(event.get("metadata_json"))
        if event_type == "validator_rejected":
            feedback["validator_rejected"] += 1
            return str(metadata.get("reason") or "VALIDATOR_FILTERED") if isinstance(metadata, dict) else "VALIDATOR_FILTERED"
        if event_type == "preflight_failed":
            feedback["preflight_failed"] += 1
            errors = metadata.get("errors") if isinstance(metadata, dict) else []
            return str(errors[0]) if isinstance(errors, list) and errors else "PREFLIGHT_FAILED"
        if event_type == "simulation_error":
            feedback["simulation_error"] += 1
            return "SIMULATION_ERROR"
        if event_type.startswith("status:failed") and isinstance(metadata, dict):
            if metadata.get("reason") == "validator_rejected":
                feedback["validator_rejected"] += 1
            errors = metadata.get("errors")
            if isinstance(errors, list) and errors:
                return str(errors[0])
            if metadata.get("reason"):
                return str(metadata["reason"])
    return ""


def _profile_from_source(source: Any) -> str:
    text = str(source or "")
    if text.startswith("model:"):
        return text.split(":", 1)[1]
    return ""


def _candidate_scope_matches(settings: Any, target_settings: Dict[str, Any]) -> bool:
    if not isinstance(settings, dict):
        return False
    for key in ("region", "delay"):
        if str(settings.get(key, "")).upper() != str(target_settings.get(key, "")).upper():
            return False
    target_universe = str(target_settings.get("universe") or "").upper()
    candidate_universe = str(settings.get("universe") or "").upper()
    return not target_universe or not candidate_universe or target_universe == candidate_universe


def _summarize_candidate_events(store: AlphaStore, candidate_id: int) -> Optional[Dict[str, Any]]:
    relevant_prefixes = ("status:failed", "status:check_pending", "submission_guard", "preflight_failed")
    for event in reversed(store.events_for_candidate(candidate_id)):
        if event["event_type"].startswith(relevant_prefixes):
            metadata = _loads_json(event.get("metadata_json"))
            return {
                "event_type": event["event_type"],
                "metadata": metadata,
            }
    return None


def _recent_global_events(store: AlphaStore, event_type: str, limit: int) -> List[Dict[str, Any]]:
    events = []
    for event in reversed(store.events_for_candidate(None)):
        if event["event_type"] != event_type:
            continue
        metadata = _loads_json(event.get("metadata_json"))
        if isinstance(metadata, dict):
            metadata = dict(metadata)
            metadata["created_at"] = event.get("created_at")
            events.append(metadata)
        if len(events) >= limit:
            break
    return events


def _summarize_reference_project(reference_dir: Path, target_settings: Dict[str, Any], limit: int) -> Dict[str, Any]:
    if not reference_dir.exists():
        return {"available": False, "path": str(reference_dir)}

    return {
        "available": True,
        "path": str(reference_dir),
        "submitted_success_examples": _submitted_examples(reference_dir / "submitted_alphas.csv", target_settings, limit),
        "recent_failure_examples": _failure_examples(reference_dir / "fail_alphas.csv", limit),
        "usa_d0_template_families": _usa_d0_template_families(
            reference_dir / "templates_usa_d0_success_submitted.json",
            target_settings,
        ),
    }


def _submitted_examples(path: Path, target_settings: Dict[str, Any], limit: int) -> List[Dict[str, Any]]:
    rows = _read_csv(path)
    if not rows:
        return []
    target_region = str(target_settings.get("region", "")).upper()
    target_delay = str(target_settings.get("delay", ""))
    matching = [
        row
        for row in rows
        if str(row.get("region", "")).upper() == target_region and str(row.get("delay", "")) == target_delay
    ]
    source = matching or rows
    examples: List[Dict[str, Any]] = []
    for row in source[:limit]:
        examples.append(
            {
                "expression": row.get("expression") or row.get("regular") or "",
                "region": row.get("region"),
                "universe": row.get("universe"),
                "delay": row.get("delay"),
                "sharpe": row.get("sharpe"),
                "fitness": row.get("fitness"),
                "status": row.get("status"),
            }
        )
    return [example for example in examples if example["expression"]]


def _failure_examples(path: Path, limit: int) -> List[Dict[str, Any]]:
    examples: List[Dict[str, Any]] = []
    for row in _read_csv(path)[:limit]:
        expression = row.get("regular") or row.get("expression") or ""
        if not expression:
            continue
        examples.append(
            {
                "expression": expression,
                "fail_reason": row.get("fail_reason") or row.get("quality_check") or row.get("tests"),
                "sharpe": row.get("sharpe"),
                "fitness": row.get("fitness"),
            }
        )
    return examples


def _usa_d0_template_families(path: Path, target_settings: Dict[str, Any]) -> Dict[str, Any]:
    if str(target_settings.get("region", "")).upper() != "USA" or str(target_settings.get("delay", "")) != "0":
        return {}
    data = _loads_json(_read_text(path)) if path.exists() else {}
    if not isinstance(data, dict):
        return {}
    return {key: value for key, value in data.items() if key != "_meta"}


def _read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")[:MAX_TEXT_CHARS].strip()


def _loads_json(value: Any) -> Any:
    if not value:
        return {}
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(str(value))
    except json.JSONDecodeError:
        return {}


def _float_or_none(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
