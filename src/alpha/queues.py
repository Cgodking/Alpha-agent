from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, Tuple


QUEUE_NAMES = ("submitable", "pending", "optimize", "watchlist", "explore_seed", "trash", "abandoned")
HARD_BLOCKER_NAMES = (
    "selfcorrelation",
    "prodcorrelation",
    "productcorrelation",
    "datadiversity",
    "highturnover",
)
PROBE_OPTIMIZE_MAX_OPEN_QUALITY_GAPS = 2
PROBE_OPTIMIZATION_QUALITY_CHECKS = {
    "concentratedweight",
    "highturnover",
    "htmaxpositionturnover",
    "htmaxtradeturnover",
    "hthighturnoverreturnsratio",
    "htturnover",
    "investablehighturnover",
    "isladdersharpe",
    "low2ysharpe",
    "lowfitness",
    "lowrobustuniversesharpe",
    "lowreturns",
    "lowsharpe",
    "lowsubuniversesharpe",
    "lowturnover",
    "robustuniverseretention",
}


def build_candidate_queues(
    store: Any,
    target_settings: Dict[str, Any] | None = None,
    *,
    limit: int = 50,
) -> Dict[str, List[Dict[str, Any]]]:
    queues: Dict[str, List[Dict[str, Any]]] = {name: [] for name in QUEUE_NAMES}
    for row in store.list_recent_candidates(max(int(limit), 1) * 20):
        settings = _loads_dict(row.get("settings_json"))
        if target_settings and not _scope_matches(settings, target_settings):
            continue
        queue, reason, priority = classify_candidate(row, store.events_for_candidate(int(row["id"])))
        queues[queue].append(_queue_item(row, settings, queue, reason, priority))
    for name in queues:
        queues[name] = sorted(
            queues[name],
            key=lambda item: (float(item["priority"]), int(item["id"])),
            reverse=True,
        )[:limit]
    return queues


def classify_candidate(row: Dict[str, Any], events: Iterable[Dict[str, Any]] | None = None) -> Tuple[str, str, float]:
    status = str(row.get("status") or "")
    metrics = _loads_dict(row.get("metrics_json"))
    checks = _loads_any(row.get("checks_json"))
    retry_count = int(row.get("retry_count") or 0)
    event_rows = list(events or [])
    if status in {"approved", "submitted"}:
        return "submitable", "approved_or_submitted", 100.0 + _quality_score(metrics)
    if status == "check_pending":
        return "pending", "terminal_checks_waiting", 90.0 + _quality_score(metrics)
    if status in {"generated", "preflight_passed"}:
        return "explore_seed", status, 20.0
    if status == "failed":
        if _has_hard_blocker(checks) or _event_reason(event_rows, "hard_blocker"):
            return "trash", "hard_blocker", -10.0
        probe_stage = _latest_probe_stage(event_rows)
        if probe_stage in {"optimize_ready", "sweep_ready"}:
            if not _probe_optimization_worthy(metrics, checks):
                return "trash", "probe_too_many_quality_gaps", -5.0 - retry_count
            return "optimize", f"probe_{probe_stage}", 80.0 + _quality_score(metrics) - retry_count
        if probe_stage == "watch":
            return "watchlist", "probe_watch", 50.0 + _quality_score(metrics) - retry_count
        if _near_threshold(metrics, checks):
            return "optimize", "near_threshold", 70.0 + _quality_score(metrics) - retry_count
        if _watchlist(metrics):
            return "watchlist", "some_signal", 45.0 + _quality_score(metrics) - retry_count
        return "trash", "low_quality", 0.0 - retry_count
    return "abandoned", "unknown_status", -20.0


def _queue_item(row: Dict[str, Any], settings: Dict[str, Any], queue: str, reason: str, priority: float) -> Dict[str, Any]:
    metrics = _loads_dict(row.get("metrics_json"))
    return {
        "id": int(row["id"]),
        "expression": row.get("expression"),
        "status": row.get("status"),
        "source": row.get("source"),
        "alpha_id": row.get("alpha_id"),
        "settings": settings,
        "metrics": metrics,
        "sharpe": metrics.get("sharpe"),
        "fitness": metrics.get("fitness"),
        "turnover": metrics.get("turnover"),
        "queue": queue,
        "queue_reason": reason,
        "priority": round(float(priority), 6),
    }


def _has_hard_blocker(checks: Any) -> bool:
    for name, check in _iter_checks(checks):
        normalized = "".join(ch for ch in str(name).lower() if ch.isalnum())
        status = str(check.get("status") or check.get("result") or "").upper()
        if normalized in HARD_BLOCKER_NAMES and status in {"FAIL", "ERROR"}:
            return True
        if normalized in {"selfcorrelation", "prodcorrelation", "productcorrelation"}:
            try:
                if abs(float(check.get("value"))) > 0.7:
                    return True
            except (TypeError, ValueError):
                pass
    return False


def _probe_optimization_worthy(metrics: Dict[str, Any], checks: Any) -> bool:
    gaps = _open_quality_gaps(checks)
    if len(gaps) > PROBE_OPTIMIZE_MAX_OPEN_QUALITY_GAPS:
        return False
    return _probe_quality_floor(metrics, checks)


def _open_quality_gaps(checks: Any) -> set[str]:
    gaps: set[str] = set()
    for name, check in _iter_checks(checks):
        normalized = "".join(ch for ch in str(name).lower() if ch.isalnum())
        if normalized not in PROBE_OPTIMIZATION_QUALITY_CHECKS:
            continue
        status = str(check.get("status") or check.get("result") or "").upper()
        if status in {"FAIL", "WARNING"}:
            gaps.add(normalized)
    return gaps


def _iter_checks(checks: Any):
    if isinstance(checks, dict):
        for name, check in checks.items():
            if isinstance(check, dict):
                yield name, check
    elif isinstance(checks, list):
        for item in checks:
            if isinstance(item, dict) and item.get("name"):
                yield item["name"], item


def _quality_score(metrics: Dict[str, Any]) -> float:
    return _float(metrics.get("sharpe")) + 0.35 * _float(metrics.get("fitness"))


def _near_threshold(metrics: Dict[str, Any], checks: Any = None) -> bool:
    sharpe = _float(metrics.get("sharpe"))
    fitness = _float(metrics.get("fitness"))
    turnover = _float(metrics.get("turnover"))
    if not 0.01 <= turnover <= 0.9:
        return False
    required_sharpe = _check_limit(checks, "lowsharpe")
    required_fitness = _check_limit(checks, "lowfitness")
    sharpe_floor = max(1.45, required_sharpe * 0.75) if required_sharpe else 1.45
    fitness_floor = max(0.35, required_fitness * 0.60) if required_fitness else 0.35
    return sharpe >= sharpe_floor and fitness >= fitness_floor


def _probe_quality_floor(metrics: Dict[str, Any], checks: Any) -> bool:
    sharpe = _float(metrics.get("sharpe"))
    fitness = _float(metrics.get("fitness"))
    turnover = _float(metrics.get("turnover"))
    if not 0.01 <= turnover <= 0.9:
        return False
    required_sharpe = _check_limit(checks, "lowsharpe")
    required_fitness = _check_limit(checks, "lowfitness")
    sharpe_floor = max(1.2, required_sharpe * 0.50) if required_sharpe else 1.2
    fitness_floor = max(0.35, required_fitness * 0.50) if required_fitness else 0.35
    return sharpe >= sharpe_floor and fitness >= fitness_floor


def _check_limit(checks: Any, target_name: str) -> float:
    for name, check in _iter_checks(checks):
        normalized = "".join(ch for ch in str(name).lower() if ch.isalnum())
        if normalized != target_name:
            continue
        limit = _float(check.get("limit"))
        return limit if limit > 0 else 0.0
    return 0.0


def _watchlist(metrics: Dict[str, Any]) -> bool:
    return _float(metrics.get("sharpe")) >= 0.8 or _float(metrics.get("fitness")) >= 0.35


def _scope_matches(settings: Dict[str, Any], target_settings: Dict[str, Any]) -> bool:
    for key in ("region", "universe", "delay", "neutralization"):
        if key in target_settings and str(settings.get(key, "")).upper() != str(target_settings.get(key, "")).upper():
            return False
    return True


def _event_reason(events: Iterable[Dict[str, Any]], reason: str) -> bool:
    for event in events:
        metadata = _loads_dict(event.get("metadata_json"))
        if str(metadata.get("reason") or "") == reason:
            return True
    return False


def _latest_probe_stage(events: Iterable[Dict[str, Any]]) -> str:
    for event in reversed(list(events)):
        if str(event.get("event_type") or "") != "probe_validation":
            continue
        metadata = _loads_dict(event.get("metadata_json"))
        stage = str(metadata.get("stage") or "").strip()
        if stage:
            return stage
    return ""


def _loads_dict(value: Any) -> Dict[str, Any]:
    loaded = _loads_any(value)
    return loaded if isinstance(loaded, dict) else {}


def _loads_any(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(str(value or "{}"))
    except json.JSONDecodeError:
        return {}


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
