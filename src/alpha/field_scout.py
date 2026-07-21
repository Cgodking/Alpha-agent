from __future__ import annotations

from typing import Any, Dict, List


DEFAULT_TOP_FIELD_LIMIT = 80
DEFAULT_BUCKET_FIELD_LIMIT = 16
FIELD_FAILURE_CLUSTER_MIN_COUNT = 2
LOW_CEILING_FAILURE_MIN_COUNT = 5
DATASET_FAILURE_CLUSTER_MIN_COUNT = 3
HIGH_TURNOVER_RAW_CATEGORIES = {
    "earnings",
    "insiders",
    "news",
    "sentiment",
    "shortinterest",
    "short interest",
    "socialmedia",
    "social media",
}
HIGH_TURNOVER_RAW_DATASET_PREFIXES = (
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


def build_field_scout(
    field_catalog: Dict[str, Any] | None,
    *,
    history_memory: Dict[str, Any] | None = None,
    submitted_avoidance: Dict[str, Any] | None = None,
    lit_tower_avoidance: Dict[str, Any] | None = None,
    allow_lit_tower_fallback: bool = False,
    top_limit: int = DEFAULT_TOP_FIELD_LIMIT,
    bucket_limit: int = DEFAULT_BUCKET_FIELD_LIMIT,
) -> Dict[str, Any]:
    catalog = field_catalog if isinstance(field_catalog, dict) else {}
    fields = catalog.get("fields") if isinstance(catalog.get("fields"), list) else []
    history_by_field = _history_by_field(history_memory)
    history_by_dataset = _history_by_dataset(history_memory)
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
        dataset_history = history_by_dataset.get(_dataset_key(field), {})
        row = _score_field(field, history, dataset_history, submitted_fields, lit_categories, unlit_categories)
        scored.append(row)

    if allow_lit_tower_fallback and not any(row.get("primary_policy") == "prefer_primary" for row in scored):
        scored = [_with_lit_tower_fallback(row) for row in scored]

    scored.sort(
        key=lambda row: (
            0 if row.get("primary_policy") == "prefer_primary" else 1,
            -float(row.get("score", 0.0)),
            int(row.get("explored_count", 0)),
            str(row.get("field") or ""),
        )
    )
    top_fields = _diversified_top_fields(scored, max(0, int(top_limit)))
    top_primary_fields = [row for row in top_fields if row.get("primary_policy") == "prefer_primary"]
    retest_primary_fields = _field_native_retest_fields(scored, top_primary_fields, bucket_limit)
    buckets = _buckets(top_fields, bucket_limit)
    status = "ready" if top_primary_fields else "no_primary_fields" if top_fields else "empty"
    return {
        "active": bool(top_primary_fields),
        "status": status,
        "policy": (
            "Use field_scout as the primary field-selection layer only when top_primary_fields is non-empty. "
            "Prefer high-score fields as primary alpha signals, diversify across buckets, and treat submitted "
            "or crowded fields as helpers unless a plan explicitly justifies them. If status is no_primary_fields, "
            "do not spend a fresh generation round until datafield discovery is broadened or the scope changes."
        ),
        "scoring": {
            "coverage": 0.25,
            "scarcity": 0.25,
            "pyramid_multiplier": 0.20,
            "novelty": 0.15,
            "tower_priority": 0.10,
            "history": 0.05,
            "failure_penalty": "up to 0.40",
            "dataset_failure_penalty": "up to 0.35",
            "submitted_primary_penalty": 0.45,
        },
        "top_fields": top_fields,
        "top_primary_fields": top_primary_fields,
        "retest_primary_fields": retest_primary_fields,
        "top_field_count": len(top_fields),
        "top_primary_field_count": len(top_primary_fields),
        "retest_primary_field_count": len(retest_primary_fields),
        "buckets": buckets,
    }


def _with_lit_tower_fallback(row: Dict[str, Any]) -> Dict[str, Any]:
    if row.get("tower_status") != "lit":
        return row
    if row.get("field_reason") or row.get("dataset_reason") or row.get("metadata_reason") or row.get("primary_block_reason"):
        return row
    components = row.get("score_components") if isinstance(row.get("score_components"), dict) else {}
    if _float(components.get("submitted_penalty")) > 0.0:
        return row
    promoted = dict(row)
    promoted["primary_policy"] = "prefer_primary"
    promoted["lit_tower_fallback"] = True
    promoted["fallback_reason"] = "scope_trouble_no_unlit_primary_fields"
    return promoted


def _with_dataset_risk_fallback(top_fields: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    promoted_fields: set[str] = set()
    result: List[Dict[str, Any]] = []
    for row in top_fields:
        if _dataset_risk_fallback_allowed(row):
            promoted = dict(row)
            promoted["primary_policy"] = "prefer_primary"
            promoted["dataset_risk_fallback"] = True
            promoted["fallback_reason"] = "no_primary_fields_after_dataset_cluster_filter"
            result.append(promoted)
            promoted_fields.add(str(row.get("field") or ""))
            continue
        result.append(row)
    if not promoted_fields:
        return top_fields
    result.sort(
        key=lambda row: (
            0 if row.get("primary_policy") == "prefer_primary" else 1,
            -float(row.get("score", 0.0)),
            int(row.get("explored_count", 0)),
            str(row.get("field") or ""),
        )
    )
    return result


def _dataset_risk_fallback_allowed(row: Dict[str, Any]) -> bool:
    if row.get("primary_policy") != "avoid_primary":
        return False
    if row.get("dataset_reason") != "recent_dataset_failure_cluster":
        return False
    if row.get("field_reason") or row.get("metadata_reason") or row.get("primary_block_reason"):
        return False
    if row.get("tower_status") == "lit":
        return False
    category = str(row.get("category") or "").strip().lower()
    dataset_id = str(row.get("dataset_id") or "").strip().lower()
    if category in HIGH_TURNOVER_RAW_CATEGORIES or dataset_id.startswith(HIGH_TURNOVER_RAW_DATASET_PREFIXES):
        return False
    if int(row.get("explored_count") or 0) > 0:
        return False
    if _float(row.get("coverage")) < 0.5:
        return False
    if _float(row.get("userCount")) > 2 or _float(row.get("alphaCount")) > 2:
        return False
    return True


def _score_field(
    field: Dict[str, Any],
    history: Dict[str, Any],
    dataset_history: Dict[str, Any],
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
    dataset_count = int(_float(dataset_history.get("count")))
    dataset_failed_count = int(_float(dataset_history.get("failed")))
    dataset_failure_rate = dataset_failed_count / max(1, dataset_count)
    dataset_distinct_field_count = _dataset_distinct_field_count(dataset_history, dataset_count)
    dataset_best_sharpe = _float(dataset_history.get("best_sharpe"))
    dataset_best_fitness = _float(dataset_history.get("best_fitness"))
    dataset_success_count = (
        int(_float(dataset_history.get("approved")))
        + int(_float(dataset_history.get("submitted")))
        + int(_float(dataset_history.get("check_pending")))
    )
    weak_dataset_reason = ""
    dataset_failure_penalty = 0.0
    field_has_positive_evidence = (
        max(_float(history.get("best_sharpe")), _float(history.get("avg_sharpe"))) >= 1.0
        or max(_float(history.get("best_fitness")), _float(history.get("avg_fitness"))) >= 0.35
    )
    field_failure_rate = failed_count / max(1, explored_count)
    weak_field_reason = ""
    if (
        explored_count >= FIELD_FAILURE_CLUSTER_MIN_COUNT
        and field_failure_rate >= 0.75
        and not field_has_positive_evidence
    ):
        weak_field_reason = "recent_field_failure_cluster"
    if (
        not weak_field_reason
        and explored_count >= LOW_CEILING_FAILURE_MIN_COUNT
        and field_failure_rate >= 0.75
        and max(_float(history.get("best_sharpe")), _float(history.get("avg_sharpe"))) < 1.0
        and max(_float(history.get("best_fitness")), _float(history.get("avg_fitness"))) < 0.75
    ):
        weak_field_reason = "repeated_low_ceiling_failures"
    if not weak_field_reason and _single_strong_negative_quality(history):
        weak_field_reason = "recent_negative_quality"
    if (
        not weak_field_reason
        and int(_float(history.get("probe_exhausted") or history.get("standardized_probe_exhausted"))) > 0
        and not field_has_positive_evidence
    ):
        weak_field_reason = _probe_exhausted_field_reason(history)
    if (
        dataset_count >= DATASET_FAILURE_CLUSTER_MIN_COUNT
        and dataset_distinct_field_count >= DATASET_FAILURE_CLUSTER_MIN_COUNT
        and dataset_failure_rate >= 0.75
        and (
            (dataset_best_sharpe < 1.0 and dataset_best_fitness < 0.35)
            or dataset_success_count <= 0
        )
        and not field_has_positive_evidence
    ):
        weak_dataset_reason = (
            "failed_only_dataset_cluster"
            if dataset_success_count <= 0 and (dataset_best_sharpe >= 1.0 or dataset_best_fitness >= 0.35)
            else "recent_dataset_failure_cluster"
        )
        dataset_failure_penalty = 0.35
    submitted_penalty = 0.45 if field_id in submitted_fields else 0.0
    metadata_reason = _metadata_field_reason(field)
    metadata_penalty = 0.55 if metadata_reason else 0.0
    usage_constraints = _usage_constraints(field)
    primary_block_reason = _primary_block_reason(field, usage_constraints, field_has_positive_evidence)
    primary_block_penalty = 0.35 if primary_block_reason else 0.0
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
        - dataset_failure_penalty
        - submitted_penalty
        - metadata_penalty
        - primary_block_penalty
        - lit_tower_penalty
        - low_coverage_penalty
    )
    primary_policy = (
        "avoid_primary"
        if field_id in submitted_fields
        or metadata_reason
        or primary_block_reason
        or weak_field_reason
        or weak_dataset_reason
        or tower_status == "lit"
        else "prefer_primary"
    )
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
        "dataset_count": dataset_count,
        "dataset_failed_count": dataset_failed_count,
        "dataset_distinct_field_count": dataset_distinct_field_count,
        "best_sharpe": history.get("best_sharpe"),
        "best_fitness": history.get("best_fitness"),
        "tower_status": tower_status,
        "primary_policy": primary_policy,
        "metadata_reason": metadata_reason,
        "primary_block_reason": primary_block_reason,
        "usage_constraints": usage_constraints,
        "field_reason": weak_field_reason,
        "dataset_reason": weak_dataset_reason,
        "score_components": {
            "coverage": round(coverage_score, 4),
            "scarcity": round(scarcity_score, 4),
            "multiplier": round(multiplier_score, 4),
            "novelty": round(novelty_score, 4),
            "tower": round(tower_score, 4),
            "history": round(history_score, 4),
            "failure_penalty": round(failure_penalty, 4),
            "dataset_failure_penalty": round(dataset_failure_penalty, 4),
            "submitted_penalty": round(submitted_penalty, 4),
            "metadata_penalty": round(metadata_penalty, 4),
            "primary_block_penalty": round(primary_block_penalty, 4),
            "lit_tower_penalty": round(lit_tower_penalty, 4),
            "low_coverage_penalty": round(low_coverage_penalty, 4),
        },
    }


def _dataset_distinct_field_count(dataset_history: Dict[str, Any], fallback_count: int) -> int:
    if "distinct_field_count" in dataset_history:
        return int(_float(dataset_history.get("distinct_field_count")))
    fields = dataset_history.get("distinct_fields")
    if isinstance(fields, list):
        return len({str(field) for field in fields if str(field).strip()})
    return fallback_count


def _field_native_retest_fields(
    rows: List[Dict[str, Any]],
    top_primary_fields: List[Dict[str, Any]],
    limit: int,
) -> List[Dict[str, Any]]:
    if top_primary_fields and not all(_is_default_saturated_category(row) for row in top_primary_fields):
        return []
    candidates: List[Dict[str, Any]] = []
    for row in rows:
        if not _is_field_native_retest_candidate(row):
            continue
        candidate = dict(row)
        candidate["retest_reason"] = "field_native_retest_after_dataset_cluster"
        candidates.append(candidate)
    candidates.sort(
        key=lambda row: (
            int(row.get("dataset_distinct_field_count") or 0),
            int(row.get("dataset_failed_count") or 0),
            -float(row.get("score") or 0.0),
            str(row.get("field") or ""),
        )
    )
    return _diversified_top_fields(candidates, max(0, int(limit or 0)))


def _is_field_native_retest_candidate(row: Dict[str, Any]) -> bool:
    if row.get("primary_policy") != "avoid_primary":
        return False
    if _is_default_saturated_category(row):
        return False
    if row.get("field_reason") or row.get("metadata_reason") or row.get("primary_block_reason"):
        return False
    if row.get("dataset_reason") not in {"recent_dataset_failure_cluster", "failed_only_dataset_cluster"}:
        return False
    if int(row.get("explored_count") or 0) > 0:
        return False
    if _float(row.get("coverage")) < 0.5:
        return False
    return True


def _is_default_saturated_category(row: Dict[str, Any]) -> bool:
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


def _diversified_top_fields(rows: List[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
    if limit <= 0:
        return []
    max_per_dataset = max(2, limit // 8)
    max_per_category = max(2, limit // 4)
    selected: List[Dict[str, Any]] = []
    selected_fields: set[str] = set()
    dataset_counts: Dict[str, int] = {}
    category_counts: Dict[str, int] = {}

    for row in rows:
        field = str(row.get("field") or "")
        if not field or field in selected_fields:
            continue
        dataset = str(row.get("dataset_id") or row.get("datasetId") or "").strip().lower()
        category = str(row.get("category") or "").strip().lower()
        if dataset_counts.get(dataset, 0) >= max_per_dataset:
            continue
        if category_counts.get(category, 0) >= max_per_category:
            continue
        selected.append(row)
        selected_fields.add(field)
        dataset_counts[dataset] = dataset_counts.get(dataset, 0) + 1
        category_counts[category] = category_counts.get(category, 0) + 1
        if len(selected) >= limit:
            return selected

    for row in rows:
        field = str(row.get("field") or "")
        if not field or field in selected_fields:
            continue
        selected.append(row)
        selected_fields.add(field)
        if len(selected) >= limit:
            break
    return selected


def _buckets(top_fields: List[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
    bucket_specs = [
        (
            "high_opportunity_unexplored",
            lambda row: int(row.get("explored_count", 0)) == 0
            and _primary_safe(row),
            "High opportunity fields not yet tested by this project.",
        ),
        (
            "low_user_high_coverage",
            lambda row: _float(row.get("coverage")) >= 0.6
            and _float(row.get("userCount")) <= 1
            and _float(row.get("alphaCount")) <= 1
            and _primary_safe(row),
            "Low-crowding fields with enough coverage to be usable.",
        ),
        (
            "high_multiplier_unlit_tower",
            lambda row: _float(row.get("pyramidMultiplier")) >= 1.5
            and row.get("tower_status") == "unlit"
            and _primary_safe(row),
            "Fields aligned with unlit high-multiplier towers.",
        ),
        (
            "recovery_candidates",
            lambda row: int(row.get("explored_count", 0)) > 0
            and _float(row.get("best_sharpe")) >= 0.5
            and _primary_safe(row),
            "Previously tested fields with some positive evidence.",
        ),
    ]
    buckets: List[Dict[str, Any]] = []
    for name, predicate, rationale in bucket_specs:
        fields = [str(row["field"]) for row in top_fields if predicate(row)][: max(0, int(limit))]
        if fields:
            buckets.append({"name": name, "fields": fields, "rationale": rationale})
    return buckets


def _primary_safe(row: Dict[str, Any]) -> bool:
    return row.get("primary_policy") != "avoid_primary" and row.get("tower_status") != "lit"


def _history_by_field(history_memory: Dict[str, Any] | None) -> Dict[str, Dict[str, Any]]:
    result: Dict[str, Dict[str, Any]] = {}
    if not isinstance(history_memory, dict):
        return result
    for key in ("top_fields", "field_stats"):
        rows = history_memory.get(key)
        if not isinstance(rows, list):
            continue
        for row in rows:
            if not isinstance(row, dict):
                continue
            field = str(row.get("field") or row.get("name") or "").strip()
            if field:
                result[field] = row
    exhausted_rows = history_memory.get("probe_exhausted_fields")
    if isinstance(exhausted_rows, list):
        for row in exhausted_rows:
            if not isinstance(row, dict):
                continue
            field = str(row.get("field") or row.get("name") or "").strip()
            if not field:
                continue
            bucket = dict(result.get(field) or {"field": field})
            exhausted_count = int(_float(row.get("count"))) or 1
            bucket["probe_exhausted"] = exhausted_count
            bucket["probe_exhausted_reason"] = str(row.get("reason") or row.get("event_type") or "standardized_probe_exhausted")
            bucket["probe_exhausted_event_type"] = str(row.get("event_type") or "")
            bucket["standardized_probe_exhausted"] = exhausted_count
            result[field] = bucket
    return result


def _history_by_dataset(history_memory: Dict[str, Any] | None) -> Dict[str, Dict[str, Any]]:
    result: Dict[str, Dict[str, Any]] = {}
    if not isinstance(history_memory, dict):
        return result
    rows = history_memory.get("top_field_datasets") or history_memory.get("field_dataset_stats")
    if not isinstance(rows, list):
        return result
    for row in rows:
        if not isinstance(row, dict):
            continue
        dataset = str(row.get("dataset_id") or row.get("field") or row.get("name") or "").strip().lower()
        if dataset:
            result[dataset] = row
    return result


def _dataset_key(field: Dict[str, Any]) -> str:
    return str(field.get("dataset_id") or field.get("datasetId") or "").strip().lower()


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


def _probe_exhausted_field_reason(history: Dict[str, Any]) -> str:
    reason = str(history.get("probe_exhausted_reason") or "").strip()
    event_type = str(history.get("probe_exhausted_event_type") or "").strip()
    if event_type == "production_rescue_probe_exhausted" or "production_rescue" in reason:
        return "production_rescue_probe_exhausted"
    return "standardized_probe_exhausted"


def _metadata_field_reason(field: Dict[str, Any]) -> str:
    field_id = str(field.get("id") or "").lower()
    description = str(field.get("description") or "").lower()
    haystack = f"{field_id} {description}"
    patterns = (
        ("currency", ("currency", "_crncy", "fund_crncy")),
        (
            "identifier_mapping",
            (
                "companyidmap",
                "company id map",
                "company id mapping",
                "company_name",
                "company name",
                "universeisin",
                "usaisin",
                " isin",
                "_isin",
                "isin ",
                "cusip",
                "sedol",
                "ticker",
                "symbol",
                "gvkey",
                "permno",
            ),
        ),
        (
            "event_time_state",
            (
                "triggertime",
                "trigger time",
                "trigger timestamp",
                "_utc_time",
                " utc time",
                "event time",
                "publish time",
                "publication time",
                "timestamp",
            ),
        ),
        ("entry_date", ("entry date", "timestamp of entry", "data point was entered", "fundamental_entry_dt", "_entry_dt")),
        (
            "event_state_helper",
            (
                "category_tag",
                "_flag",
                " flag",
                "position_flag",
                "advantageous_position",
                "event state",
                "event category",
            ),
        ),
        (
            "count_period_helper",
            (
                "wordcount",
                "word count",
                "paragraphcount",
                "paragraph count",
                "sentcount",
                "sentence count",
                "posnum",
                "fiscal_quarter",
                "fiscal quarter",
                "financial_year",
                "financial year",
                "fiscal_year",
                "fiscal year",
            ),
        ),
        (
            "size_helper",
            (
                "market_capitalization",
                "market capitalization",
                "market cap",
                "mktcap",
            ),
        ),
        (
            "risk_state_helper",
            (
                "volatility",
                "parkinson_volatility",
                "beta_prediction",
                "uncertainty",
                "_srisk",
                " srisk",
            ),
        ),
        (
            "price_volume_like",
            (
                "vwap",
                "volume",
                "price",
                "closing_session",
                "after_hours",
                "pre_market",
                "intraday",
                "minute_interval",
            ),
        ),
    )
    for reason, needles in patterns:
        if any(needle in haystack for needle in needles):
            return reason
    return ""


def _usage_constraints(field: Dict[str, Any]) -> List[str]:
    constraints: List[str] = []
    if _is_high_turnover_raw_field(field):
        constraints.append("requires_turnover_stabilizer")
    if _is_event_field(field):
        constraints.append("event_field_not_direct_rank_input")
    return constraints


def _single_strong_negative_quality(history: Dict[str, Any]) -> bool:
    if int(_float(history.get("count"))) < 1 or int(_float(history.get("failed"))) < 1:
        return False
    best_sharpe = max(_float(history.get("best_sharpe")), _float(history.get("avg_sharpe")))
    best_fitness = max(_float(history.get("best_fitness")), _float(history.get("avg_fitness")))
    return best_sharpe <= 0.0 and best_fitness <= 0.0


def _primary_block_reason(
    field: Dict[str, Any],
    usage_constraints: List[str],
    field_has_positive_evidence: bool = False,
) -> str:
    field_type = str(field.get("type") or "").strip().upper()
    if field_type == "VECTOR" and "event_field_not_direct_rank_input" in usage_constraints:
        return "event_vector_primary_block"
    if "requires_turnover_stabilizer" in usage_constraints and _is_insider_raw_field(field) and not field_has_positive_evidence:
        return "unproven_insider_raw_field"
    return ""


def _is_high_turnover_raw_field(field: Dict[str, Any]) -> bool:
    category = str(field.get("category") or "").strip().lower()
    dataset_id = str(field.get("dataset_id") or field.get("datasetId") or "").strip().lower()
    dataset_name = str(field.get("dataset_name") or field.get("datasetName") or "").strip().lower()
    if category in HIGH_TURNOVER_RAW_CATEGORIES:
        return True
    return dataset_id.startswith(HIGH_TURNOVER_RAW_DATASET_PREFIXES) or dataset_name.startswith(
        HIGH_TURNOVER_RAW_DATASET_PREFIXES
    )


def _is_insider_raw_field(field: Dict[str, Any]) -> bool:
    category = str(field.get("category") or "").strip().lower()
    dataset_id = str(field.get("dataset_id") or field.get("datasetId") or "").strip().lower()
    dataset_name = str(field.get("dataset_name") or field.get("datasetName") or "").strip().lower()
    return (
        category == "insiders"
        or dataset_id.startswith(("insd", "insider"))
        or dataset_name.startswith(("insd", "insider"))
        or "insider" in dataset_name
    )


def _is_event_field(field: Dict[str, Any]) -> bool:
    category = str(field.get("category") or "").lower()
    dataset_id = str(field.get("dataset_id") or field.get("datasetId") or "").lower()
    dataset_name = str(field.get("dataset_name") or field.get("datasetName") or "").lower()
    return (
        "news" in category
        or "event" in category
        or dataset_id.startswith(("news", "nws"))
        or "news" in dataset_name
        or "event" in dataset_name
    )


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
