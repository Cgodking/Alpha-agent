from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List

from .db import AlphaStore
from .scopes import apply_scope


def parse_scope_json(base: Dict[str, Any], scope_json: str | None) -> List[Dict[str, Any]]:
    if not scope_json:
        return []
    try:
        payload = json.loads(scope_json)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid scope-json: {exc}") from exc
    if not isinstance(payload, list):
        raise ValueError("scope-json must be a list of scope objects")
    scopes: List[Dict[str, Any]] = []
    for item in payload:
        if not isinstance(item, dict):
            raise ValueError("scope-json entries must be objects")
        scopes.append(apply_scope(base, overrides=item))
    return scopes


def next_rotating_scope(store: AlphaStore, scopes: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not scopes:
        raise ValueError("scope rotation needs at least one scope")
    signature = scope_rotation_signature(scopes)
    state = store.get_run_state("scope_rotation")
    if state.get("signature") == signature:
        index = _bounded_index(state.get("next_index"), len(scopes))
    else:
        index = 0
    next_index = (index + 1) % len(scopes)
    selected = dict(scopes[index])
    new_state = {
        "signature": signature,
        "size": len(scopes),
        "last_index": index,
        "next_index": next_index,
        "last_scope": selected,
    }
    store.set_run_state("scope_rotation", new_state)
    store.record_event(None, "scope_rotation_selected", new_state)
    return selected


def scope_rotation_signature(scopes: List[Dict[str, Any]]) -> str:
    normalized = [
        {str(key): _stable_value(value) for key, value in sorted(scope.items(), key=lambda item: str(item[0]))}
        for scope in scopes
    ]
    data = json.dumps(normalized, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(data.encode("utf-8")).hexdigest()[:16]


def _bounded_index(value: Any, size: int) -> int:
    try:
        index = int(value)
    except (TypeError, ValueError):
        return 0
    if index < 0:
        return 0
    return index % max(1, size)


def _stable_value(value: Any) -> Any:
    if isinstance(value, str):
        return value.strip().upper()
    return value
