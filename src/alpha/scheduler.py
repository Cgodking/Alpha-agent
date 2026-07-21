from __future__ import annotations

import json
import logging
import math
import os
from collections import Counter
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from .metrics import compute_efficiency_metrics
from .queues import build_candidate_queues
from .research_planner import _extract_fields


_log = logging.getLogger("alpha.scheduler")

FIELD_EXPOSURE_WINDOW = 40
FIELD_EXPOSURE_MIN_CANDIDATES = 16
FIELD_EXPOSURE_MIN_COUNT = 6
FIELD_EXPOSURE_MAX_SHARE = 0.25
OPTIMIZE_TARGET_MAX_UNPRODUCTIVE_CYCLES = 2


def _read_events(store: Any, candidate_id: Optional[int]) -> Optional[List[Dict[str, Any]]]:
    """Read events for a candidate, returning None if unavailable.

    None means "could not read" (no reader, or a read error that we log) so callers
    can choose their own fallback. A read failure is logged at debug level rather than
    silently masked, so a real bug surfaces instead of looking like "no signal".
    """
    reader = getattr(store, "events_for_candidate", None)
    if not callable(reader):
        return None
    try:
        return list(reader(candidate_id))
    except Exception:
        _log.debug("scheduler event read failed candidate_id=%s", candidate_id, exc_info=True)
        return None


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
    "market",
    "sector",
    "industry",
    "subindustry",
    "country",
    "exchange",
}


def build_cycle_plan(
    store: Any,
    base_context: Dict[str, Any],
    *,
    batch_size: int = 8,
    created_since: str | None = None,
) -> Dict[str, Any]:
    scope = _scope(base_context)
    effective_created_since = created_since or _active_run_started_at(store)
    field_exposure = _recent_field_exposure(store, scope, effective_created_since)
    cooldown_fields = sorted(field_exposure)
    target_failure_counts = _unproductive_optimize_target_counts(store, scope, effective_created_since)
    blocked_target_ids = sorted(
        target_id
        for target_id, count in target_failure_counts.items()
        if count >= OPTIMIZE_TARGET_MAX_UNPRODUCTIVE_CYCLES
    )
    dynamic_constraints: Dict[str, Any] = {
        "avoid_structures": [],
        "cooldown_fields": cooldown_fields,
    }
    if field_exposure:
        dynamic_constraints["field_exposure"] = field_exposure
    if blocked_target_ids:
        dynamic_constraints["blocked_optimize_target_ids"] = blocked_target_ids
        dynamic_constraints["optimize_target_failure_counts"] = {
            str(target_id): target_failure_counts[target_id] for target_id in blocked_target_ids
        }

    def make_plan(
        mode: str,
        target_candidate_id: int | None,
        plan_batch_size: int,
        reason: str,
        *,
        apply_field_constraints: bool = True,
    ) -> Dict[str, Any]:
        constraints = dynamic_constraints if apply_field_constraints else {}
        return _plan(mode, scope, target_candidate_id, plan_batch_size, reason, metrics, constraints=constraints)

    queues = build_candidate_queues(store, scope, limit=20)
    metrics = compute_efficiency_metrics(store, scope, created_since=created_since)
    optimize_exhausted_reason = _recent_optimize_quality_stop_loss(store, scope) or _recent_optimize_unproductive_churn(
        store,
        scope,
    )
    if optimize_exhausted_reason:
        plan = make_plan("explore", None, batch_size, optimize_exhausted_reason)
        plan["constraints"]["avoid_modes"] = ["optimize"]
        return plan
    exhausted_reason = _production_rescue_exhausted_reason(store, scope)
    if exhausted_reason:
        plan = make_plan("explore", None, batch_size, exhausted_reason)
        plan["constraints"]["avoid_modes"] = ["production_rescue"]
        return plan
    explore_exhausted_reason = _recent_explore_duplicate_only(store, scope)
    if explore_exhausted_reason:
        plan = make_plan("production_rescue", None, batch_size, explore_exhausted_reason)
        plan["constraints"]["avoid_modes"] = ["explore"]
        return plan
    cooldown_reason = _cooldown_reason(store, scope)
    pending_target = _recoverable_pending_target(store, queues["pending"])
    if pending_target:
        target = pending_target
        return make_plan(
            "recover_pending",
            int(target["id"]),
            min(batch_size, 4),
            "pending_candidate_has_existing_simulation",
            apply_field_constraints=False,
        )
    if queues["submitable"]:
        target = queues["submitable"][0]
        return make_plan(
            "setting_sweep",
            int(target["id"]),
            min(batch_size, 8),
            "approved_candidate_may_have_setting_upside",
            apply_field_constraints=False,
        )
    submitted_safe_targets = _filter_submitted_avoidance_targets(store, scope, queues["optimize"])
    blocked_target_set = set(blocked_target_ids)
    cooldown_field_set = set(cooldown_fields)
    optimize_targets = [
        target
        for target in submitted_safe_targets
        if int(target["id"]) not in blocked_target_set
        and not _target_hits_cooldown_fields(target, cooldown_field_set)
    ]
    if optimize_targets:
        target = optimize_targets[0]
        queue_reason = str(target.get("queue_reason") or "near_threshold")
        return make_plan(
            "optimize",
            int(target["id"]),
            min(batch_size, 4),
            f"{queue_reason}_candidate_has_fixable_gap",
        )
    if queues["optimize"]:
        cooldown_blocked_ids = [
            int(target["id"])
            for target in submitted_safe_targets
            if int(target["id"]) in blocked_target_set
            or _target_hits_cooldown_fields(target, cooldown_field_set)
        ]
        reason = "optimize_targets_cooldown" if cooldown_blocked_ids else "submitted_field_optimize_targets_exhausted"
        plan = make_plan("explore", None, batch_size, reason)
        plan["constraints"]["avoid_modes"] = ["optimize"]
        plan["constraints"]["blocked_optimize_target_ids"] = sorted(
            {int(item["id"]) for item in queues["optimize"]}
        )
        return plan
    if cooldown_reason:
        return make_plan("production_rescue", None, batch_size, cooldown_reason)
    if queues["pending"]:
        plan = make_plan("explore", None, batch_size, "pending_recheck_cooldown")
        plan["constraints"]["pending_recheck_cooldown_ids"] = [int(item["id"]) for item in queues["pending"]]
        return plan
    return make_plan("explore", None, batch_size, "no_higher_value_queue_available")


def _plan(
    mode: str,
    scope: Dict[str, Any],
    target_candidate_id: int | None,
    batch_size: int,
    reason: str,
    metrics: Dict[str, Any],
    *,
    constraints: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    plan_constraints = {"avoid_structures": [], "cooldown_fields": []}
    plan_constraints.update(dict(constraints or {}))
    return {
        "mode": mode,
        "scope": scope,
        "target_candidate_id": target_candidate_id,
        "budget": {"batch_size": int(batch_size), "max_rounds": 3 if mode == "optimize" else 1},
        "constraints": plan_constraints,
        "reason": reason,
        "metrics": {
            "totals": metrics.get("totals", {}),
            "rates": metrics.get("rates", {}),
        },
    }


def _active_run_started_at(store: Any) -> str | None:
    getter = getattr(store, "get_run_state", None)
    if not callable(getter):
        return None
    state = getter("daemon", {})
    if not isinstance(state, dict):
        return None
    started_at = str(state.get("started_at") or "").strip()
    return started_at or None


def _recent_field_exposure(
    store: Any,
    scope: Dict[str, Any],
    created_since: str | None,
) -> Dict[str, Dict[str, Any]]:
    lister = getattr(store, "list_candidates", None)
    if not callable(lister):
        return {}
    rows = lister(created_since=created_since) if created_since else lister()
    scoped_rows = [
        row
        for row in rows
        if _cycle_scope_matches(_loads_dict(row.get("settings_json")), scope)
    ][-FIELD_EXPOSURE_WINDOW:]
    if len(scoped_rows) < FIELD_EXPOSURE_MIN_CANDIDATES:
        return {}
    counts: Counter[str] = Counter()
    for row in scoped_rows:
        fields = set(_primary_signal_fields(_extract_fields(str(row.get("expression") or ""), [])))
        counts.update(fields)
    threshold = max(FIELD_EXPOSURE_MIN_COUNT, math.ceil(len(scoped_rows) * FIELD_EXPOSURE_MAX_SHARE))
    return {
        field: {
            "count": count,
            "share": round(count / len(scoped_rows), 6),
            "window_candidates": len(scoped_rows),
            "threshold_count": threshold,
        }
        for field, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        if count >= threshold
    }


def _unproductive_optimize_target_counts(
    store: Any,
    scope: Dict[str, Any],
    created_since: str | None,
) -> Dict[int, int]:
    events = _read_events(store, None)
    if events is None:
        return {}
    counts: Dict[int, int] = {}
    for event in events:
        if created_since and str(event.get("created_at") or "") < created_since:
            continue
        if str(event.get("event_type") or "") != "cycle_outcome":
            continue
        metadata = _loads_dict(event.get("metadata_json"))
        if not _cycle_outcome_scope_matches(metadata, scope):
            continue
        cycle_plan = metadata.get("cycle_plan") if isinstance(metadata.get("cycle_plan"), dict) else {}
        if str(cycle_plan.get("mode") or "") != "optimize":
            continue
        try:
            target_id = int(cycle_plan.get("target_candidate_id") or 0)
        except (TypeError, ValueError):
            continue
        if target_id <= 0:
            continue
        summary = metadata.get("summary") if isinstance(metadata.get("summary"), dict) else {}
        useful = sum(int(summary.get(key) or 0) for key in ("approved", "submitted", "pending"))
        if useful > 0:
            counts.pop(target_id, None)
        elif _optimize_cycle_churned(cycle_plan, summary):
            counts[target_id] = counts.get(target_id, 0) + 1
    return counts


def _target_hits_cooldown_fields(target: Dict[str, Any], cooldown_fields: set[str]) -> bool:
    if not cooldown_fields:
        return False
    fields = _primary_signal_fields(_extract_fields(str(target.get("expression") or ""), []))
    return bool(set(fields) & cooldown_fields)


def _filter_submitted_avoidance_targets(
    store: Any,
    scope: Dict[str, Any],
    optimize_targets: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    avoidance = _latest_submitted_field_avoidance(store, scope)
    if not avoidance:
        return optimize_targets
    return [target for target in optimize_targets if not _target_hits_submitted_avoidance(target, avoidance)]


def _latest_submitted_field_avoidance(store: Any, scope: Dict[str, Any]) -> Dict[str, Any]:
    for event in reversed(store.events_for_candidate(None)):
        if str(event.get("event_type") or "") != "experiment_plan":
            continue
        plan = _loads_dict(event.get("metadata_json"))
        if not plan:
            continue
        plan_scope = plan.get("target_settings") if isinstance(plan.get("target_settings"), dict) else {}
        scheduler_plan = plan.get("scheduler_plan") if isinstance(plan.get("scheduler_plan"), dict) else {}
        if not plan_scope:
            plan_scope = scheduler_plan.get("scope") if isinstance(scheduler_plan.get("scope"), dict) else {}
        if plan_scope and not _cycle_scope_matches(plan_scope, scope):
            continue
        avoidance = plan.get("submitted_field_avoidance") if isinstance(plan.get("submitted_field_avoidance"), dict) else {}
        fields = [str(field) for field in avoidance.get("fields") or [] if str(field).strip()]
        families = [str(family) for family in avoidance.get("families") or [] if str(family).strip()]
        if fields or families:
            return {"fields": fields, "families": families}
    return {}


def _target_hits_submitted_avoidance(target: Dict[str, Any], avoidance: Dict[str, Any]) -> bool:
    fields = _primary_signal_fields(_extract_fields(str(target.get("expression") or ""), []))
    submitted_fields = {str(field) for field in avoidance.get("fields") or [] if str(field).strip()}
    submitted_families = {str(family) for family in avoidance.get("families") or [] if str(family).strip()}
    return any(field in submitted_fields or _submission_field_family(field) in submitted_families for field in fields)


def _primary_signal_fields(fields: List[str]) -> List[str]:
    return [
        field
        for field in fields
        if field not in SUBMISSION_FIELD_AUXILIARIES and not str(field).startswith("pv13_")
    ]


def _submission_field_family(field: str) -> str:
    parts = [part for part in str(field or "").split("_") if part]
    if len(parts) >= 2:
        return "_".join(parts[:2])
    return parts[0] if parts else ""


def _scope(context: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "region": context.get("region", "USA"),
        "universe": context.get("universe", "TOP3000"),
        "delay": context.get("delay", 1),
        "neutralization": context.get("neutralization", "INDUSTRY"),
    }


def _cooldown_reason(store: Any, scope: Dict[str, Any]) -> str:
    quality_stop_count = 0
    saw_cycle_outcome = False
    for event in reversed(store.events_for_candidate(None)):
        event_type = str(event.get("event_type") or "")
        metadata = _loads_dict(event.get("metadata_json"))
        if event_type == "cycle_outcome":
            cycle_plan = metadata.get("cycle_plan") if isinstance(metadata.get("cycle_plan"), dict) else {}
            if not _cycle_scope_matches(cycle_plan.get("scope"), scope):
                continue
            saw_cycle_outcome = True
            summary = metadata.get("summary") if isinstance(metadata.get("summary"), dict) else {}
            if summary.get("quality_stop_loss"):
                if _cycle_summary_has_platform_simulation_error(summary):
                    break
                quality_stop_count += 1
                if quality_stop_count >= 2:
                    return "quality_stop_loss_repeated"
                continue
            break
        if event_type == "simulation_error" and _platform_error(metadata):
            continue

    if saw_cycle_outcome:
        return ""

    for event in reversed(store.events_for_candidate(None)):
        event_type = str(event.get("event_type") or "")
        metadata = _loads_dict(event.get("metadata_json"))
        if event_type != "quality_stop_loss":
            continue
        event_scope = metadata.get("scope") if isinstance(metadata.get("scope"), dict) else {}
        if not event_scope or not _cycle_scope_matches(event_scope, scope):
            continue
        quality_stop_count += 1
        if quality_stop_count >= 2:
            return "quality_stop_loss_repeated"
    return ""


def _production_rescue_exhausted_reason(store: Any, scope: Dict[str, Any]) -> str:
    getter = getattr(store, "get_run_state", None)
    if callable(getter):
        state = getter("daemon", {})
        if (
            isinstance(state, dict)
            and str(state.get("status") or "") == "stopped"
            and str(state.get("stop_reason") or "") == "production_rescue_duplicate_only"
        ):
            previous_scope = state.get("scope")
            if not isinstance(previous_scope, dict) or _scope(previous_scope) == _scope(scope):
                return "production_rescue_duplicate_only_recent"
    if not _recent_duplicate_only_stop_without_new_productive_cycle(store, scope):
        return (
            _recent_production_rescue_probe_simulation_error(store, scope)
            or _recent_production_rescue_quality_stop_without_probe_signal(store, scope)
        )
    return "production_rescue_duplicate_only_recent"


def _recent_optimize_quality_stop_loss(store: Any, scope: Dict[str, Any]) -> str:
    getter = getattr(store, "get_run_state", None)
    if callable(getter):
        state = getter("daemon", {})
        if (
            isinstance(state, dict)
            and str(state.get("status") or "") == "stopped"
            and str(state.get("stop_reason") or "") == "optimize_quality_stop_loss"
        ):
            previous_scope = state.get("scope")
            if not isinstance(previous_scope, dict) or _scope(previous_scope) == _scope(scope):
                return "optimize_quality_stop_loss_recent"
    events = _read_events(store, None)
    if events is None:
        return ""
    for event in reversed(events[-200:]):
        if str(event.get("event_type") or "") != "cycle_outcome":
            continue
        metadata = _loads_dict(event.get("metadata_json"))
        if not _cycle_outcome_scope_matches(metadata, scope):
            continue
        cycle_plan = metadata.get("cycle_plan") if isinstance(metadata.get("cycle_plan"), dict) else {}
        summary = metadata.get("summary") if isinstance(metadata.get("summary"), dict) else {}
        if str(cycle_plan.get("mode") or "") != "optimize":
            return ""
        if not summary.get("quality_stop_loss"):
            return ""
        return "optimize_quality_stop_loss_recent"
    return ""


def _recent_optimize_unproductive_churn(store: Any, scope: Dict[str, Any]) -> str:
    events = _read_events(store, None)
    if events is None:
        return ""
    churn_count = 0
    for event in reversed(events[-200:]):
        if str(event.get("event_type") or "") != "cycle_outcome":
            continue
        metadata = _loads_dict(event.get("metadata_json"))
        if not _cycle_outcome_scope_matches(metadata, scope):
            continue
        cycle_plan = metadata.get("cycle_plan") if isinstance(metadata.get("cycle_plan"), dict) else {}
        summary = metadata.get("summary") if isinstance(metadata.get("summary"), dict) else {}
        if str(cycle_plan.get("mode") or "") != "optimize":
            return ""
        if _cycle_summary_has_platform_simulation_error(summary):
            return ""
        if not _optimize_cycle_churned(cycle_plan, summary):
            return ""
        churn_count += 1
        if churn_count >= 2:
            return "optimize_unproductive_churn_recent"
    return ""


def _optimize_cycle_churned(cycle_plan: Dict[str, Any], summary: Dict[str, Any]) -> bool:
    useful = sum(int(summary.get(key) or 0) for key in ("approved", "submitted", "pending"))
    if useful > 0:
        return False
    generated = int(summary.get("generated") or 0)
    failed = int(summary.get("failed") or 0)
    skipped = int(summary.get("skipped") or 0)
    ai_generation_timeout = int(summary.get("ai_generation_timeout") or 0)
    budget = cycle_plan.get("budget") if isinstance(cycle_plan.get("budget"), dict) else {}
    planned_batch = int(budget.get("batch_size") or 0)
    if ai_generation_timeout > 0:
        return True
    if skipped > 0 and generated == 0:
        return True
    if generated > 0 and failed >= generated:
        return True
    return planned_batch > 0 and generated < planned_batch and (failed > 0 or skipped > 0)


def _recoverable_pending_target(store: Any, pending_queue: List[Dict[str, Any]]) -> Dict[str, Any] | None:
    for item in pending_queue:
        if _pending_recheck_ready(store, int(item.get("id") or 0)):
            return item
    return None


def _pending_recheck_ready(store: Any, candidate_id: int) -> bool:
    if candidate_id <= 0:
        return False
    events = _read_events(store, candidate_id)
    if events is None:
        return True
    latest_pending = None
    for event in reversed(events):
        if str(event.get("event_type") or "") == "simulation_pending":
            latest_pending = event
            break
    if latest_pending is None:
        return True
    age = _event_age_seconds(latest_pending)
    if age is None:
        return True
    return age >= _pending_recheck_cooldown_seconds()


def _pending_recheck_cooldown_seconds() -> float:
    try:
        return max(0.0, float(os.environ.get("PENDING_RECHECK_COOLDOWN_SECONDS", "900")))
    except ValueError:
        return 900.0


def _event_age_seconds(event: Dict[str, Any]) -> float | None:
    text = str(event.get("created_at") or "").strip()
    if not text:
        return None
    try:
        created_at = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if created_at.tzinfo is None:
        created_at = created_at.replace(tzinfo=timezone.utc)
    return max(0.0, (datetime.now(timezone.utc) - created_at.astimezone(timezone.utc)).total_seconds())


def _recent_production_rescue_quality_stop_without_probe_signal(store: Any, scope: Dict[str, Any]) -> str:
    events = _read_events(store, None)
    if events is None:
        return ""
    for event in reversed(events[-200:]):
        if str(event.get("event_type") or "") != "cycle_outcome":
            continue
        metadata = _loads_dict(event.get("metadata_json"))
        if not _cycle_outcome_scope_matches(metadata, scope):
            continue
        cycle_plan = metadata.get("cycle_plan") if isinstance(metadata.get("cycle_plan"), dict) else {}
        summary = metadata.get("summary") if isinstance(metadata.get("summary"), dict) else {}
        if str(cycle_plan.get("mode") or "") != "production_rescue":
            return ""
        if not summary.get("quality_stop_loss"):
            return ""
        if _production_rescue_summary_has_probe_signal(summary):
            return ""
        return "production_rescue_quality_stop_loss_recent"
    return ""


def _recent_production_rescue_probe_simulation_error(store: Any, scope: Dict[str, Any]) -> str:
    events = _read_events(store, None)
    if events is None:
        return ""
    for event in reversed(events[-200:]):
        if str(event.get("event_type") or "") != "cycle_outcome":
            continue
        metadata = _loads_dict(event.get("metadata_json"))
        if not _cycle_outcome_scope_matches(metadata, scope):
            continue
        cycle_plan = metadata.get("cycle_plan") if isinstance(metadata.get("cycle_plan"), dict) else {}
        summary = metadata.get("summary") if isinstance(metadata.get("summary"), dict) else {}
        if str(cycle_plan.get("mode") or "") != "production_rescue":
            return ""
        if int(summary.get("probe_simulation_error") or 0) <= 0:
            return ""
        return "production_rescue_probe_simulation_error_recent"
    return ""


def _recent_duplicate_only_stop_without_new_productive_cycle(store: Any, scope: Dict[str, Any]) -> bool:
    events = _read_events(store, None)
    if events is None:
        return False
    latest_duplicate_only_cycle = _latest_cycle_outcome_was_duplicate_only(events, scope)
    if latest_duplicate_only_cycle:
        return True
    saw_newer_productive_cycle = False
    for event in reversed(events[-200:]):
        event_type = str(event.get("event_type") or "")
        metadata = _loads_dict(event.get("metadata_json"))
        if event_type == "cycle_outcome":
            if _cycle_outcome_scope_matches(metadata, scope) and _cycle_outcome_productive(metadata):
                saw_newer_productive_cycle = True
            continue
        if event_type != "daemon_stopped":
            continue
        if str(metadata.get("reason") or "") != "production_rescue_duplicate_only":
            continue
        if saw_newer_productive_cycle:
            return False
        return _nearest_prior_cycle_outcome_was_duplicate_only(events, event, scope)
    return False


def _recent_explore_duplicate_only(store: Any, scope: Dict[str, Any]) -> str:
    events = _read_events(store, None)
    if events is None:
        return ""
    for event in reversed(events[-200:]):
        if str(event.get("event_type") or "") != "cycle_outcome":
            continue
        metadata = _loads_dict(event.get("metadata_json"))
        if not _cycle_outcome_scope_matches(metadata, scope):
            continue
        if _cycle_outcome_productive(metadata):
            return ""
        cycle_plan = metadata.get("cycle_plan") if isinstance(metadata.get("cycle_plan"), dict) else {}
        summary = metadata.get("summary") if isinstance(metadata.get("summary"), dict) else {}
        if str(cycle_plan.get("mode") or "") == "explore" and int(summary.get("skipped") or 0) > 0:
            return "explore_duplicate_only_recent"
        return ""
    return ""


def _latest_cycle_outcome_was_duplicate_only(events: List[Dict[str, Any]], scope: Dict[str, Any]) -> bool:
    for event in reversed(events[-200:]):
        if str(event.get("event_type") or "") != "cycle_outcome":
            continue
        metadata = _loads_dict(event.get("metadata_json"))
        if not _cycle_outcome_scope_matches(metadata, scope):
            continue
        cycle_plan = metadata.get("cycle_plan") if isinstance(metadata.get("cycle_plan"), dict) else {}
        summary = metadata.get("summary") if isinstance(metadata.get("summary"), dict) else {}
        if _cycle_outcome_productive(metadata):
            return False
        return (
            str(cycle_plan.get("mode") or "") == "production_rescue"
            and int(summary.get("skipped") or 0) > 0
        )
    return False


def _nearest_prior_cycle_outcome_was_duplicate_only(
    events: List[Dict[str, Any]],
    stop_event: Dict[str, Any],
    scope: Dict[str, Any],
) -> bool:
    try:
        stop_id = int(stop_event.get("id") or 0)
    except (TypeError, ValueError):
        stop_id = 0
    for event in reversed(events):
        try:
            event_id = int(event.get("id") or 0)
        except (TypeError, ValueError):
            event_id = 0
        if stop_id and event_id >= stop_id:
            continue
        if str(event.get("event_type") or "") != "cycle_outcome":
            continue
        metadata = _loads_dict(event.get("metadata_json"))
        if not _cycle_outcome_scope_matches(metadata, scope):
            continue
        summary = metadata.get("summary") if isinstance(metadata.get("summary"), dict) else {}
        return (
            int(summary.get("skipped") or 0) > 0
            and sum(int(summary.get(key) or 0) for key in ("generated", "approved", "submitted", "failed", "pending")) == 0
        )
    return False


def _cycle_outcome_scope_matches(metadata: Dict[str, Any], scope: Dict[str, Any]) -> bool:
    cycle_plan = metadata.get("cycle_plan") if isinstance(metadata.get("cycle_plan"), dict) else {}
    cycle_scope = cycle_plan.get("scope") if isinstance(cycle_plan.get("scope"), dict) else {}
    return _cycle_scope_matches(cycle_scope, scope)


def _cycle_scope_matches(left: Any, right: Dict[str, Any]) -> bool:
    return isinstance(left, dict) and _scope(left) == _scope(right)


def _cycle_outcome_productive(metadata: Dict[str, Any]) -> bool:
    summary = metadata.get("summary") if isinstance(metadata.get("summary"), dict) else {}
    return sum(int(summary.get(key) or 0) for key in ("generated", "approved", "submitted", "failed", "pending")) > 0


def _cycle_summary_has_platform_simulation_error(summary: Dict[str, Any]) -> bool:
    if int(summary.get("probe_simulation_error") or 0) > 0:
        return True
    if int(summary.get("platform_error_failures") or 0) > 0:
        return True
    return _platform_error(summary)


def _production_rescue_summary_has_probe_signal(summary: Dict[str, Any]) -> bool:
    return any(int(summary.get(key) or 0) > 0 for key in ("probe_watch", "probe_optimize_ready", "probe_sweep_ready"))


def _platform_error(metadata: Dict[str, Any]) -> bool:
    text = json.dumps(metadata, sort_keys=True).lower()
    return (
        "429" in text
        or "retry-after" in text
        or "rate limit" in text
        or "simulation polling did not return an alpha id" in text
        or "unexpected property" in text
        or "cycleplan" in text
    )


def _loads_dict(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    try:
        data = json.loads(str(value or "{}"))
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}
