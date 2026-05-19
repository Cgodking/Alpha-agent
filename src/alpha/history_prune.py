from __future__ import annotations

import json
from typing import Any, Dict, List

from .db import AlphaStore
from .models import DEFAULT_SETTINGS
from .research_planner import candidate_quality_summary


DEFAULT_LOW_QUALITY_SCORE_MAX = 0.2


def find_low_quality_history_candidates(
    store: AlphaStore,
    target_settings: Dict[str, Any] | None = None,
    *,
    quality_max: float = DEFAULT_LOW_QUALITY_SCORE_MAX,
    limit: int = 1000,
) -> List[Dict[str, Any]]:
    selected: List[Dict[str, Any]] = []
    target = dict(target_settings or {})
    for row in store.list_recent_candidates(max(0, int(limit)) * 20 or 1000):
        if str(row.get("status") or "") != "failed":
            continue
        settings = _loads_dict(row.get("settings_json"))
        if target and not _scope_matches(settings, target):
            continue
        metrics = _loads_dict(row.get("metrics_json"))
        checks = _loads_checks(row.get("checks_json"))
        quality = candidate_quality_summary({"settings": settings, "metrics": metrics, "checks": checks})
        history_noise_score = _history_noise_score(metrics, quality)
        if history_noise_score > float(quality_max):
            continue
        selected.append(
            {
                "id": int(row["id"]),
                "quality_score": _float(quality.get("quality_score")),
                "history_noise_score": history_noise_score,
                "sharpe": metrics.get("sharpe"),
                "fitness": metrics.get("fitness"),
                "settings": settings,
                "expression": row.get("expression"),
                "failed_checks": quality.get("failed_checks") or [],
            }
        )
        if len(selected) >= max(0, int(limit)):
            break
    return selected


def prune_low_quality_history(
    store: AlphaStore,
    target_settings: Dict[str, Any] | None = None,
    *,
    quality_max: float = DEFAULT_LOW_QUALITY_SCORE_MAX,
    limit: int = 1000,
    execute: bool = False,
) -> Dict[str, Any]:
    selected = find_low_quality_history_candidates(store, target_settings, quality_max=quality_max, limit=limit)
    candidate_ids = [int(item["id"]) for item in selected]
    archived = 0
    if execute and candidate_ids:
        archived = store.archive_candidates(
            candidate_ids,
            "low_quality_history",
            {"quality_max": float(quality_max), "scope": dict(target_settings or {})},
        )
    return {
        "execute": bool(execute),
        "quality_max": float(quality_max),
        "limit": int(limit),
        "selected": len(selected),
        "archived": archived,
        "candidate_ids": candidate_ids[:50],
    }


def _scope_matches(settings: Dict[str, Any], target_settings: Dict[str, Any]) -> bool:
    merged_candidate = dict(DEFAULT_SETTINGS)
    merged_candidate.update(settings or {})
    merged_target = dict(DEFAULT_SETTINGS)
    merged_target.update(target_settings or {})
    for key in ("region", "delay"):
        if str(merged_candidate.get(key, "")).upper() != str(merged_target.get(key, "")).upper():
            return False
    target_universe = str(merged_target.get("universe") or "").upper()
    candidate_universe = str(merged_candidate.get("universe") or "").upper()
    return not target_universe or not candidate_universe or target_universe == candidate_universe


def _loads_dict(value: Any) -> Dict[str, Any]:
    loaded = _loads_json(value)
    return loaded if isinstance(loaded, dict) else {}


def _loads_checks(value: Any) -> Dict[str, Any] | List[Dict[str, Any]]:
    loaded = _loads_json(value)
    return loaded if isinstance(loaded, (dict, list)) else {}


def _loads_json(value: Any) -> Any:
    if not value:
        return {}
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(str(value))
    except json.JSONDecodeError:
        return {}


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _history_noise_score(metrics: Dict[str, Any], quality: Dict[str, Any]) -> float:
    raw_score = _float(quality.get("raw_score"))
    if not raw_score:
        sharpe = max(0.0, _float(metrics.get("sharpe")))
        fitness = max(0.0, _float(metrics.get("fitness")))
        raw_score = sharpe + 0.35 * fitness
    quality_score = _float(quality.get("quality_score"))
    return min(quality_score, raw_score)
