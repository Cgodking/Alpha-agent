from __future__ import annotations

import json
from typing import Any, Dict

from .metrics import compute_efficiency_metrics
from .queues import build_candidate_queues


def build_cycle_plan(
    store: Any,
    base_context: Dict[str, Any],
    *,
    batch_size: int = 8,
    created_since: str | None = None,
) -> Dict[str, Any]:
    scope = _scope(base_context)
    queues = build_candidate_queues(store, scope, limit=20)
    metrics = compute_efficiency_metrics(store, scope, created_since=created_since)
    cooldown_reason = _cooldown_reason(store)
    if cooldown_reason:
        return _plan("cooldown", scope, None, batch_size, cooldown_reason, metrics)
    if queues["pending"]:
        target = queues["pending"][0]
        return _plan(
            "recover_pending",
            scope,
            int(target["id"]),
            min(batch_size, 4),
            "pending_candidate_has_existing_simulation",
            metrics,
        )
    if queues["submitable"]:
        target = queues["submitable"][0]
        return _plan(
            "setting_sweep",
            scope,
            int(target["id"]),
            min(batch_size, 8),
            "approved_candidate_may_have_setting_upside",
            metrics,
        )
    if queues["optimize"]:
        target = queues["optimize"][0]
        return _plan(
            "optimize",
            scope,
            int(target["id"]),
            min(batch_size, 4),
            "near_threshold_candidate_has_fixable_gap",
            metrics,
        )
    return _plan("explore", scope, None, batch_size, "no_higher_value_queue_available", metrics)


def _plan(
    mode: str,
    scope: Dict[str, Any],
    target_candidate_id: int | None,
    batch_size: int,
    reason: str,
    metrics: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "mode": mode,
        "scope": scope,
        "target_candidate_id": target_candidate_id,
        "budget": {"batch_size": int(batch_size), "max_rounds": 3 if mode == "optimize" else 1},
        "constraints": {"avoid_structures": [], "cooldown_fields": []},
        "reason": reason,
        "metrics": {
            "totals": metrics.get("totals", {}),
            "rates": metrics.get("rates", {}),
        },
    }


def _scope(context: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "region": context.get("region", "USA"),
        "universe": context.get("universe", "TOP3000"),
        "delay": context.get("delay", 1),
        "neutralization": context.get("neutralization", "INDUSTRY"),
    }


def _cooldown_reason(store: Any) -> str:
    quality_stop_count = 0
    for event in store.events_for_candidate(None):
        event_type = str(event.get("event_type") or "")
        metadata = _loads_dict(event.get("metadata_json"))
        if event_type == "quality_stop_loss":
            quality_stop_count += 1
        if event_type == "simulation_error" and _platform_error(metadata):
            continue
    return "quality_stop_loss_repeated" if quality_stop_count >= 2 else ""


def _platform_error(metadata: Dict[str, Any]) -> bool:
    text = json.dumps(metadata, sort_keys=True).lower()
    return "429" in text or "retry-after" in text or "rate limit" in text


def _loads_dict(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    try:
        data = json.loads(str(value or "{}"))
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}
