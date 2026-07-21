from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Dict


def daemon_health(store: Any, *, stall_minutes: int = 60) -> Dict[str, Any]:
    state = store.get_run_state("daemon")
    status = str(state.get("status") or "stopped")
    started_at = str(state.get("started_at") or "")
    last_block_reason = "" if status == "running" else str(state.get("stop_reason") or "")
    last_event_at = ""
    for event in reversed(store.events_for_candidate(None)):
        event_type = str(event.get("event_type") or "")
        if not last_event_at:
            last_event_at = str(event.get("created_at") or "")
        if status != "running" and event_type in {"daemon_stopped", "ai_generation_error", "quality_stop_loss"} and not last_block_reason:
            metadata = _loads_dict(event.get("metadata_json"))
            last_block_reason = str(metadata.get("reason") or metadata.get("quality_stop_reason") or "")
    stalled = status == "running" and _minutes_since(last_event_at or started_at) >= float(stall_minutes)
    return {
        "status": status,
        "pid": int(state.get("pid") or 0),
        "started_at": started_at,
        "last_event_at": last_event_at,
        "stalled": bool(stalled),
        "last_block_reason": last_block_reason,
    }


def _minutes_since(value: str) -> float:
    if not value:
        return 999999.0
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return 999999.0
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return (datetime.now(timezone.utc) - parsed).total_seconds() / 60.0


def _loads_dict(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    try:
        data = json.loads(str(value or "{}"))
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}
