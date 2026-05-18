from __future__ import annotations

import json
import hashlib
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FIELD_CACHE_DIR = PROJECT_ROOT / "data" / "field_cache"
DEFAULT_FIELD_SEARCH_TERMS = ["", "model", "analyst", "fundamental", "pv", "news", "sentiment"]
DEFAULT_FIELD_LIMIT = 120
DEFAULT_FIELD_CACHE_TTL_SECONDS = 24 * 60 * 60

BUILTIN_FIELDS = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "returns",
    "vwap",
    "cap",
    "adv20",
]


def build_field_catalog(client: Any, target_settings: Dict[str, Any]) -> Dict[str, Any]:
    if not _bool_env("ALPHA_FIELD_DISCOVERY", True):
        return {"available": False, "disabled": True, "field_ids": BUILTIN_FIELDS}

    cache_dir = Path(os.getenv("ALPHA_FIELD_CACHE_DIR", str(DEFAULT_FIELD_CACHE_DIR)))
    ttl_seconds = int(os.getenv("ALPHA_FIELD_CACHE_TTL_SECONDS", str(DEFAULT_FIELD_CACHE_TTL_SECONDS)))
    max_fields = int(os.getenv("ALPHA_FIELD_LIMIT", str(DEFAULT_FIELD_LIMIT)))
    search_terms = _field_search_terms()
    provider_name = client.__class__.__name__
    cache_path = _cache_path(cache_dir, target_settings, search_terms, max_fields, provider_name)

    cached = _read_cache(cache_path, ttl_seconds)
    if cached is not None:
        catalog = _normalize_catalog(cached, target_settings=target_settings)
        catalog["source"] = "cache"
        return catalog

    if not hasattr(client, "discover_datafields"):
        return {
            "available": False,
            "error": f"{client.__class__.__name__} does not support datafield discovery",
            "field_ids": BUILTIN_FIELDS,
        }

    try:
        rows = client.discover_datafields(target_settings, search_terms=search_terms, max_fields=max_fields)
    except Exception as exc:
        return {"available": False, "error": str(exc), "field_ids": BUILTIN_FIELDS}

    catalog = _normalize_catalog(summarize_datafields(rows, target_settings=target_settings, max_fields=max_fields), target_settings=target_settings)
    catalog["source"] = provider_name
    _write_cache(cache_path, catalog)
    return catalog


def summarize_datafields(
    rows: Iterable[Dict[str, Any]],
    target_settings: Dict[str, Any],
    max_fields: int = DEFAULT_FIELD_LIMIT,
) -> Dict[str, Any]:
    fields: List[Dict[str, Any]] = []
    seen = set()
    for row in rows:
        field_id = str(row.get("id") or "").strip()
        if not field_id or field_id in seen:
            continue
        seen.add(field_id)
        dataset = row.get("dataset") if isinstance(row.get("dataset"), dict) else {}
        category = row.get("category") if isinstance(row.get("category"), dict) else {}
        fields.append(
            {
                "id": field_id,
                "type": row.get("type"),
                "dataset_id": dataset.get("id"),
                "dataset_name": dataset.get("name"),
                "category": category.get("name"),
                "coverage": row.get("coverage"),
                "dateCoverage": row.get("dateCoverage"),
                "userCount": row.get("userCount"),
                "alphaCount": row.get("alphaCount"),
                "pyramidMultiplier": row.get("pyramidMultiplier"),
                "description": _short_text(row.get("description"), 140),
            }
        )
        if len(fields) >= max_fields:
            break

    field_ids = list(dict.fromkeys(BUILTIN_FIELDS + [field["id"] for field in fields]))
    field_types = {field_id: "MATRIX" for field_id in BUILTIN_FIELDS}
    field_types.update({field["id"]: str(field.get("type") or "").upper() for field in fields})
    return {
        "available": bool(fields),
        "scope": {
            "instrumentType": target_settings.get("instrumentType", "EQUITY"),
            "region": target_settings.get("region"),
            "universe": target_settings.get("universe"),
            "delay": target_settings.get("delay"),
        },
        "field_ids": field_ids,
        "field_types": field_types,
        "matrix_fields": [field_id for field_id, field_type in field_types.items() if field_type == "MATRIX"],
        "vector_fields": [field_id for field_id, field_type in field_types.items() if field_type == "VECTOR"],
        "fields": fields,
        "datasets": _dataset_summary(fields),
        "field_count": len(fields),
        "rules": [
            "Use only field_ids listed here plus standard price fields.",
            "Do not invent or guess datafield identifiers.",
            "Use MATRIX fields directly with ts_ operators.",
            "Do not pass VECTOR fields directly into ts_ operators such as ts_backfill, ts_mean, or ts_rank.",
            "Reduce VECTOR fields with a valid single-argument vec_* reducer before applying time-series operators.",
            "If the field pool is weak for an idea, propose no candidate for that idea.",
        ],
    }


def _normalize_catalog(catalog: Dict[str, Any], target_settings: Dict[str, Any]) -> Dict[str, Any]:
    normalized = dict(catalog or {})
    fields = normalized.get("fields")
    fields = fields if isinstance(fields, list) else []

    field_ids = normalized.get("field_ids")
    if not isinstance(field_ids, list):
        field_ids = []
    normalized["field_ids"] = list(dict.fromkeys(BUILTIN_FIELDS + [str(field) for field in field_ids if str(field)]))

    raw_types = normalized.get("field_types")
    field_types = {field_id: "MATRIX" for field_id in BUILTIN_FIELDS}
    if isinstance(raw_types, dict):
        field_types.update({str(field): str(field_type).upper() for field, field_type in raw_types.items()})
    for field in fields:
        if isinstance(field, dict) and field.get("id") and field.get("type"):
            field_types[str(field["id"])] = str(field["type"]).upper()
    normalized["field_types"] = field_types
    normalized["matrix_fields"] = [field_id for field_id, field_type in field_types.items() if field_type == "MATRIX"]
    normalized["vector_fields"] = [field_id for field_id, field_type in field_types.items() if field_type == "VECTOR"]

    scope = normalized.get("scope")
    if not isinstance(scope, dict):
        normalized["scope"] = {
            "instrumentType": target_settings.get("instrumentType", "EQUITY"),
            "region": target_settings.get("region"),
            "universe": target_settings.get("universe"),
            "delay": target_settings.get("delay"),
        }

    normalized["rules"] = _catalog_rules(normalized.get("rules"))
    return normalized


def _catalog_rules(existing: Any) -> List[str]:
    rules = [str(rule) for rule in existing if str(rule).strip()] if isinstance(existing, list) else []
    required = [
        "Use only field_ids listed here plus standard price fields.",
        "Do not invent or guess datafield identifiers.",
        "Use MATRIX fields directly with ts_ operators.",
        "Do not pass VECTOR fields directly into ts_ operators such as ts_backfill, ts_mean, or ts_rank.",
        "Reduce VECTOR fields with a valid single-argument vec_* reducer before applying time-series operators.",
        "Never wrap MATRIX fields with vec_* reducers.",
        "If the field pool is weak for an idea, propose no candidate for that idea.",
    ]
    for rule in required:
        if rule not in rules:
            rules.append(rule)
    return rules


def _dataset_summary(fields: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    counts: Dict[str, Dict[str, Any]] = {}
    for field in fields:
        dataset_id = field.get("dataset_id") or "unknown"
        item = counts.setdefault(str(dataset_id), {"id": dataset_id, "name": field.get("dataset_name"), "field_count": 0})
        item["field_count"] += 1
    return sorted(counts.values(), key=lambda item: (-int(item["field_count"]), str(item["id"])))[:20]


def _field_search_terms() -> List[str]:
    raw = os.getenv("ALPHA_FIELD_SEARCHES")
    if raw is None:
        return DEFAULT_FIELD_SEARCH_TERMS
    terms = [item.strip() for item in raw.split(",")]
    return [term for term in terms if term] or [""]


def _cache_path(
    cache_dir: Path,
    target_settings: Dict[str, Any],
    search_terms: List[str],
    max_fields: int,
    provider_name: str,
) -> Path:
    scope = "_".join(
        [
            provider_name,
            str(target_settings.get("instrumentType", "EQUITY")).upper(),
            str(target_settings.get("region", "UNKNOWN")).upper(),
            str(target_settings.get("universe", "UNKNOWN")).upper(),
            f"D{target_settings.get('delay', 'UNKNOWN')}",
            f"N{max_fields}",
            hashlib.sha1("|".join(search_terms).encode("utf-8")).hexdigest()[:12],
        ]
    )
    return cache_dir / f"{scope}.json"


def _read_cache(path: Path, ttl_seconds: int) -> Dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if time.time() - float(data.get("created_at", 0)) > ttl_seconds:
            return None
        catalog = data.get("catalog")
        return catalog if isinstance(catalog, dict) else None
    except Exception:
        return None


def _write_cache(path: Path, catalog: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"created_at": time.time(), "catalog": catalog}
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _short_text(value: Any, limit: int) -> str:
    text = " ".join(str(value or "").split())
    return text[:limit]
