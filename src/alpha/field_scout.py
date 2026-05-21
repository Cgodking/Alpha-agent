from __future__ import annotations

from typing import Any, Dict, List


DEFAULT_TOP_FIELD_LIMIT = 80
DEFAULT_BUCKET_FIELD_LIMIT = 16


def build_field_scout(
    field_catalog: Dict[str, Any] | None,
    *,
    history_memory: Dict[str, Any] | None = None,
    submitted_avoidance: Dict[str, Any] | None = None,
    lit_tower_avoidance: Dict[str, Any] | None = None,
    top_limit: int = DEFAULT_TOP_FIELD_LIMIT,
    bucket_limit: int = DEFAULT_BUCKET_FIELD_LIMIT,
) -> Dict[str, Any]:
    catalog = field_catalog if isinstance(field_catalog, dict) else {}
    fields = catalog.get("fields") if isinstance(catalog.get("fields"), list) else []
    history_by_field = _history_by_field(history_memory)
    submitted_fields = {
        str(field)
        for field in (submitted_avoidance or {}).get("fields", [])
        if str(field).strip()
    }
    lit_categories = _tower_categories(lit_tower_avoidance, "lit_towers")
    unlit_categories = _tower_categories(lit_tower_avoidance, "unlit_towers")

    scored: List[Dict[str, Any]] = []
    for field in fields:
        if not isinstance(field, dict):
            continue
        field_id = str(field.get("id") or "").strip()
        if not field_id:
            continue
        history = history_by_field.get(field_id, {})
        row = _score_field(field, history, submitted_fields, lit_categories, unlit_categories)
        scored.append(row)

    scored.sort(
        key=lambda row: (
            -float(row.get("score", 0.0)),
            int(row.get("explored_count", 0)),
            str(row.get("field") or ""),
        )
    )
    top_fields = scored[: max(0, int(top_limit))]
    buckets = _buckets(top_fields, bucket_limit)
    return {
        "active": bool(top_fields),
        "policy": (
            "Use field_scout as the primary field-selection layer. Prefer high-score fields as primary alpha "
            "signals, diversify across buckets, and treat submitted or crowded fields as helpers unless a plan "
            "explicitly justifies them."
        ),
        "scoring": {
            "coverage": 0.25,
            "scarcity": 0.25,
            "pyramid_multiplier": 0.20,
            "novelty": 0.15,
            "tower_priority": 0.10,
            "history": 0.05,
            "failure_penalty": "up to 0.40",
            "submitted_primary_penalty": 0.45,
        },
        "top_fields": top_fields,
        "buckets": buckets,
    }


def _score_field(
    field: Dict[str, Any],
    history: Dict[str, Any],
    submitted_fields: set[str],
    lit_categories: set[str],
    unlit_categories: set[str],
) -> Dict[str, Any]:
    field_id = str(field.get("id") or "")
    category = _category(field.get("category"))
    explored_count = int(_float(history.get("count")))
    failed_count = int(_float(history.get("failed")))
    coverage_score = _clamp(_float(field.get("coverage")) / 0.8)
    scarcity_score = 0.5 * _inverse_count(field.get("userCount")) + 0.5 * _inverse_count(field.get("alphaCount"))
    multiplier_score = _clamp((_float(field.get("pyramidMultiplier")) - 1.0) / 0.8)
    novelty_score = 1.0 if explored_count <= 0 else 1.0 / (1.0 + explored_count)
    tower_score = _tower_score(category, lit_categories, unlit_categories)
    history_score = _clamp(max(_float(history.get("best_sharpe")), _float(history.get("avg_sharpe"))) / 2.0)
    failure_penalty = 0.0
    if explored_count > 0:
        failure_penalty = min(0.40, 0.40 * (failed_count / max(1, explored_count)))
    submitted_penalty = 0.45 if field_id in submitted_fields else 0.0
    metadata_reason = _metadata_field_reason(field)
    metadata_penalty = 0.55 if metadata_reason else 0.0
    tower_status = "unlit" if category in unlit_categories else "lit" if category in lit_categories else "unknown"
    lit_tower_penalty = 0.25 if tower_status == "lit" else 0.0
    low_coverage_penalty = 0.20 if _float(field.get("coverage")) <= 0.0 else 0.0
    score = (
        coverage_score * 0.25
        + scarcity_score * 0.25
        + multiplier_score * 0.20
        + novelty_score * 0.15
        + tower_score * 0.10
        + history_score * 0.05
        - failure_penalty
        - submitted_penalty
        - metadata_penalty
        - lit_tower_penalty
        - low_coverage_penalty
    )
    primary_policy = "avoid_primary" if field_id in submitted_fields or metadata_reason or tower_status == "lit" else "prefer_primary"
    return {
        "field": field_id,
        "score": round(_clamp(score), 4),
        "type": field.get("type"),
        "dataset_id": field.get("dataset_id"),
        "category": field.get("category"),
        "coverage": field.get("coverage"),
        "userCount": field.get("userCount"),
        "alphaCount": field.get("alphaCount"),
        "pyramidMultiplier": field.get("pyramidMultiplier"),
        "explored_count": explored_count,
        "failed_count": failed_count,
        "best_sharpe": history.get("best_sharpe"),
        "best_fitness": history.get("best_fitness"),
        "tower_status": tower_status,
        "primary_policy": primary_policy,
        "metadata_reason": metadata_reason,
        "score_components": {
            "coverage": round(coverage_score, 4),
            "scarcity": round(scarcity_score, 4),
            "multiplier": round(multiplier_score, 4),
            "novelty": round(novelty_score, 4),
            "tower": round(tower_score, 4),
            "history": round(history_score, 4),
            "failure_penalty": round(failure_penalty, 4),
            "submitted_penalty": round(submitted_penalty, 4),
            "metadata_penalty": round(metadata_penalty, 4),
            "lit_tower_penalty": round(lit_tower_penalty, 4),
            "low_coverage_penalty": round(low_coverage_penalty, 4),
        },
    }


def _buckets(top_fields: List[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
    bucket_specs = [
        (
            "high_opportunity_unexplored",
            lambda row: int(row.get("explored_count", 0)) == 0
            and row.get("primary_policy") != "avoid_primary"
            and row.get("tower_status") != "lit",
            "High opportunity fields not yet tested by this project.",
        ),
        (
            "low_user_high_coverage",
            lambda row: _float(row.get("coverage")) >= 0.6
            and _float(row.get("userCount")) <= 1
            and _float(row.get("alphaCount")) <= 1,
            "Low-crowding fields with enough coverage to be usable.",
        ),
        (
            "high_multiplier_unlit_tower",
            lambda row: _float(row.get("pyramidMultiplier")) >= 1.5 and row.get("tower_status") == "unlit",
            "Fields aligned with unlit high-multiplier towers.",
        ),
        (
            "recovery_candidates",
            lambda row: int(row.get("explored_count", 0)) > 0 and _float(row.get("best_sharpe")) >= 0.5,
            "Previously tested fields with some positive evidence.",
        ),
    ]
    buckets: List[Dict[str, Any]] = []
    for name, predicate, rationale in bucket_specs:
        fields = [str(row["field"]) for row in top_fields if predicate(row)][: max(0, int(limit))]
        if fields:
            buckets.append({"name": name, "fields": fields, "rationale": rationale})
    return buckets


def _history_by_field(history_memory: Dict[str, Any] | None) -> Dict[str, Dict[str, Any]]:
    rows = (history_memory or {}).get("top_fields") if isinstance(history_memory, dict) else []
    result: Dict[str, Dict[str, Any]] = {}
    if not isinstance(rows, list):
        return result
    for row in rows:
        if not isinstance(row, dict):
            continue
        field = str(row.get("field") or row.get("name") or "").strip()
        if field:
            result[field] = row
    return result


def _tower_categories(lit_tower_avoidance: Dict[str, Any] | None, key: str) -> set[str]:
    rows = (lit_tower_avoidance or {}).get(key) if isinstance(lit_tower_avoidance, dict) else []
    categories: set[str] = set()
    if not isinstance(rows, list):
        return categories
    for row in rows:
        if not isinstance(row, dict):
            continue
        category = _category(row.get("category") or row.get("category_name"))
        if category:
            categories.add(category)
    return categories


def _tower_score(category: str, lit_categories: set[str], unlit_categories: set[str]) -> float:
    if category in unlit_categories:
        return 1.0
    if category in lit_categories:
        return 0.15
    return 0.5


def _metadata_field_reason(field: Dict[str, Any]) -> str:
    field_id = str(field.get("id") or "").lower()
    description = str(field.get("description") or "").lower()
    haystack = f"{field_id} {description}"
    patterns = (
        ("currency", ("currency", "_crncy", "fund_crncy")),
        ("entry_date", ("entry date", "timestamp of entry", "data point was entered", "fundamental_entry_dt", "_entry_dt")),
    )
    for reason, needles in patterns:
        if any(needle in haystack for needle in needles):
            return reason
    return ""


def _category(value: Any) -> str:
    normalized = str(value or "").strip().upper().replace("_", "").replace("-", "").replace(" ", "")
    aliases = {
        "PRICEVOLUME": "PV",
        "PRICEVOLUMEDATA": "PV",
        "PRICENVOLUME": "PV",
    }
    return aliases.get(normalized, normalized)


def _inverse_count(value: Any) -> float:
    return 1.0 / (1.0 + max(0.0, _float(value)))


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, float(value)))
