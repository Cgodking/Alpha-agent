from __future__ import annotations

import json
import re
from typing import Any, Dict, Iterable, List

from .db import AlphaStore
from .models import DEFAULT_SETTINGS


SIMULATED_STATUSES = {"simulated", "metric_passed", "approved", "submitted", "check_pending"}
APPROVED_STATUSES = {"approved", "submitted"}
PLATFORM_ERROR_MARKERS = (
    "429",
    "retry-after",
    "rate limit",
    "too many requests",
    "temporarily unavailable",
    "simulation polling did not return an alpha id",
    "unexpected property",
    "cycleplan",
)


def compute_efficiency_metrics(
    store: AlphaStore,
    target_settings: Dict[str, Any] | None = None,
    *,
    created_since: str | None = None,
) -> Dict[str, Any]:
    rows = [
        row
        for row in store.list_candidates(created_since=created_since)
        if _scope_matches(_loads_dict(row.get("settings_json")), target_settings or {})
    ]
    global_events = store.events_for_candidate(None)
    totals = {
        "generated": len(rows),
        "preflight_passed": 0,
        "simulated": 0,
        "approved": 0,
        "submitted": 0,
        "pending": 0,
        "failed": 0,
        "duplicate_skipped": _duplicate_skip_count(global_events),
        "quality_waste_failures": 0,
        "platform_error_failures": 0,
        "near_threshold": 0,
    }
    by_source: Dict[str, Dict[str, Any]] = {}
    by_scope: Dict[str, Dict[str, Any]] = {}
    by_field_family: Dict[str, Dict[str, Any]] = {}

    for row in rows:
        status = str(row.get("status") or "")
        events = store.events_for_candidate(int(row["id"]))
        metrics = _loads_dict(row.get("metrics_json"))
        settings = _loads_dict(row.get("settings_json"))
        source = str(row.get("source") or "unknown")
        scope_key = _scope_key(settings)
        family = _field_family(str(row.get("expression") or ""))
        platform_error = _has_platform_error(events)
        simulated = _has_event(events, "status:simulated") or bool(row.get("alpha_id")) or status in SIMULATED_STATUSES
        preflight_passed = _has_event(events, "status:preflight_passed") or simulated or status in SIMULATED_STATUSES
        approved = status in APPROVED_STATUSES
        near_threshold = _near_threshold(metrics)

        if preflight_passed:
            totals["preflight_passed"] += 1
        if simulated:
            totals["simulated"] += 1
        if approved:
            totals["approved"] += 1
        if status == "submitted":
            totals["submitted"] += 1
        if status == "check_pending":
            totals["pending"] += 1
        if status == "failed":
            totals["failed"] += 1
            if platform_error:
                totals["platform_error_failures"] += 1
            elif simulated or metrics:
                totals["quality_waste_failures"] += 1
        if near_threshold:
            totals["near_threshold"] += 1

        _accumulate_group(by_source, source, status, preflight_passed, simulated, approved, platform_error)
        _accumulate_group(by_scope, scope_key, status, preflight_passed, simulated, approved, platform_error)
        _accumulate_group(by_field_family, family, status, preflight_passed, simulated, approved, platform_error)

    rates = {
        "preflight_pass_rate": _ratio(totals["preflight_passed"], totals["generated"]),
        "simulation_success_rate": _ratio(totals["simulated"], totals["preflight_passed"]),
        "approved_rate": _ratio(totals["approved"], totals["generated"]),
        "approved_per_100_simulations": 100.0 * _ratio(totals["approved"], totals["simulated"]),
        "duplicate_skip_rate": _ratio(totals["duplicate_skipped"], totals["generated"] + totals["duplicate_skipped"]),
        "simulation_waste_rate": _ratio(totals["quality_waste_failures"], totals["simulated"]),
    }
    return {
        "totals": totals,
        "rates": rates,
        "by_source": _finalize_groups(by_source),
        "by_scope": _finalize_groups(by_scope),
        "by_field_family": _finalize_groups(by_field_family),
    }


def _duplicate_skip_count(events: List[Dict[str, Any]]) -> int:
    return sum(
        1
        for event in events
        if str(event.get("event_type") or "") in {"duplicate_candidate_skipped", "structural_duplicate_candidate_skipped"}
    )


def _accumulate_group(
    groups: Dict[str, Dict[str, Any]],
    key: str,
    status: str,
    preflight_passed: bool,
    simulated: bool,
    approved: bool,
    platform_error: bool,
) -> None:
    item = groups.setdefault(
        key,
        {
            "generated": 0,
            "preflight_passed": 0,
            "simulated": 0,
            "approved": 0,
            "submitted": 0,
            "failed": 0,
            "platform_error_failures": 0,
        },
    )
    item["generated"] += 1
    if preflight_passed:
        item["preflight_passed"] += 1
    if simulated:
        item["simulated"] += 1
    if approved:
        item["approved"] += 1
    if status == "submitted":
        item["submitted"] += 1
    if status == "failed":
        item["failed"] += 1
        if platform_error:
            item["platform_error_failures"] += 1


def _finalize_groups(groups: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    finalized: Dict[str, Dict[str, Any]] = {}
    for key, item in groups.items():
        output = dict(item)
        output["preflight_pass_rate"] = _ratio(output["preflight_passed"], output["generated"])
        output["approved_per_100_simulations"] = 100.0 * _ratio(output["approved"], output["simulated"])
        finalized[key] = output
    return finalized


def _scope_matches(settings: Dict[str, Any], target_settings: Dict[str, Any]) -> bool:
    if not target_settings:
        return True
    merged_candidate = dict(DEFAULT_SETTINGS)
    merged_candidate.update(settings or {})
    merged_target = dict(DEFAULT_SETTINGS)
    merged_target.update(target_settings or {})
    for key in ("region", "universe", "delay", "neutralization"):
        if str(merged_candidate.get(key, "")).upper() != str(merged_target.get(key, "")).upper():
            return False
    return True


def _scope_key(settings: Dict[str, Any]) -> str:
    merged = dict(DEFAULT_SETTINGS)
    merged.update(settings or {})
    return (
        f"{str(merged.get('region') or '').upper()}|"
        f"{str(merged.get('universe') or '').upper()}|"
        f"D{merged.get('delay')}|"
        f"{str(merged.get('neutralization') or '').upper()}"
    )


def _field_family(expression: str) -> str:
    text = expression.lower()
    if re.search(r"\b(?:anl\d+_|analyst_|actual_update_)", text):
        return "analyst"
    if re.search(r"\b(?:snt\d+|sentiment_)", text):
        return "sentiment"
    if re.search(r"\b(?:close|open|high|low|volume|vwap|returns|adv20|cap)\b", text):
        return "price_volume"
    return "other"


def _near_threshold(metrics: Dict[str, Any]) -> bool:
    sharpe = _float(metrics.get("sharpe"))
    fitness = _float(metrics.get("fitness"))
    turnover = _float(metrics.get("turnover"))
    if sharpe >= 1.45:
        return True
    if sharpe < 1.2:
        return False
    return fitness >= 0.75 or 0.01 <= turnover <= 0.9


def _has_event(events: Iterable[Dict[str, Any]], event_type: str) -> bool:
    return any(str(event.get("event_type") or "") == event_type for event in events)


def _has_platform_error(events: Iterable[Dict[str, Any]]) -> bool:
    for event in events:
        if str(event.get("event_type") or "") != "simulation_error":
            continue
        metadata = _loads_dict(event.get("metadata_json"))
        text = json.dumps(metadata, sort_keys=True).lower()
        if any(marker in text for marker in PLATFORM_ERROR_MARKERS):
            return True
    return False


def _loads_dict(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    try:
        data = json.loads(str(value or "{}"))
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _ratio(numerator: int | float, denominator: int | float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0
