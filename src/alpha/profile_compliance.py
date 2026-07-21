from __future__ import annotations

import re
from typing import Any, Dict, List

from .models import CandidateSpec


PROFILE_FAMILIES = (
    "ANALYST",
    "NEWS",
    "SENTIMENT",
    "SOCIALMEDIA",
    "FUNDAMENTAL",
    "EARNINGS",
    "INSIDERS",
    "INSTITUTIONS",
    "OPTION",
    "RISK",
    "PV",
    "MODEL",
    "OTHER",
)
PROFILE_HELPER_IDENTIFIERS = {
    "open",
    "high",
    "low",
    "close",
    "volume",
    "returns",
    "vwap",
    "cap",
    "adv20",
    "market",
    "sector",
    "industry",
    "subindustry",
    "country",
    "exchange",
}


def profile_compliance_errors(spec: CandidateSpec, context: Dict[str, Any]) -> List[str]:
    expression = str(spec.expression or "")
    used_fields = _used_catalog_fields(expression, _allowed_fields_from_context(context))
    errors = _field_exposure_errors(used_fields, context)
    guidance = spec.metadata.get("profile_guidance") if isinstance(spec.metadata, dict) else None
    if not isinstance(guidance, dict) or not guidance:
        return errors

    metadata = _field_metadata_from_context(context)
    guidance_text = _profile_guidance_text(guidance)
    required_family = profile_required_family(guidance)
    if required_family and (
        not _uses_field_family(used_fields, metadata, required_family)
        or _uses_non_family_catalog_field(used_fields, metadata, required_family)
    ):
        errors.append(f"PROFILE_REQUIRED_FIELD_FAMILY:{required_family}")
    if _profile_requires_analyst_family(guidance) and (
        not _uses_analyst_field(expression, used_fields, metadata)
        or _uses_non_analyst_catalog_field(used_fields, metadata)
    ):
        errors.append("PROFILE_REQUIRED_FIELD_FAMILY:ANALYST")
    if _profile_bans_analyst_family(guidance_text, guidance) and _uses_analyst_field(expression, used_fields, metadata):
        errors.append("PROFILE_FORBIDDEN_FIELD_FAMILY:ANALYST")
    if _profile_requires_sentiment_family(guidance) and (
        not _uses_field_family(used_fields, metadata, "SENTIMENT")
        or _uses_non_family_catalog_field(used_fields, metadata, "SENTIMENT")
    ):
        errors.append("PROFILE_REQUIRED_FIELD_FAMILY:SENTIMENT")
    if _profile_bans_sentiment_family(guidance_text, guidance) and _uses_field_family(used_fields, metadata, "SENTIMENT"):
        errors.append("PROFILE_FORBIDDEN_FIELD_FAMILY:SENTIMENT")

    for pattern in _profile_forbidden_field_patterns(guidance):
        hits = _forbidden_field_pattern_hits(expression, used_fields, pattern)
        for hit in hits:
            errors.append(f"PROFILE_FORBIDDEN_FIELD:{hit}")
    return list(dict.fromkeys(errors))


def _field_exposure_errors(used_fields: List[str], context: Dict[str, Any]) -> List[str]:
    plan = _context_research_context(context).get("experiment_plan")
    if not isinstance(plan, dict):
        return []
    control = plan.get("field_exposure_control")
    if not isinstance(control, dict):
        return []
    cooldown_fields = {str(field) for field in control.get("cooldown_fields") or [] if str(field).strip()}
    return [
        f"PROFILE_COOLDOWN_FIELD:{field}"
        for field in used_fields
        if field in cooldown_fields
        and field.lower() not in PROFILE_HELPER_IDENTIFIERS
        and not field.startswith("pv13_")
    ]


def profile_required_family(guidance: Dict[str, Any]) -> str:
    field_family = str(guidance.get("field_family") or "").strip().lower()
    if not field_family:
        return ""
    for family in PROFILE_FAMILIES:
        token = family.lower()
        if re.search(rf"\bnon[-_\s]?{re.escape(token)}\b", field_family):
            continue
        if (
            re.search(rf"\b{re.escape(token)}[-_\s]?only\b", field_family)
            or f"{token} primary only" in field_family
            or f"{token} primaries only" in field_family
            or field_family.startswith(f"{token} primary")
            or field_family.startswith(f"{token} route")
        ):
            return family
    return ""


def profile_family_fields(context: Dict[str, Any], guidance: Dict[str, Any], limit: int = 80) -> List[str]:
    if not isinstance(guidance, dict) or not guidance:
        return []
    family = profile_required_family(guidance)
    if not family and _profile_requires_analyst_family(guidance):
        family = "ANALYST"
    if not family and _profile_requires_sentiment_family(guidance):
        family = "SENTIMENT"
    if not family:
        return []
    result: List[str] = []
    metadata = _field_metadata_from_context(context)
    for field in _allowed_fields_from_context(context):
        if _field_is_family(field, metadata.get(field, {}), family):
            result.append(field)
            if len(result) >= limit:
                break
    return result


def _context_research_context(context: Dict[str, Any]) -> Dict[str, Any]:
    research_context = context.get("research_context") if isinstance(context, dict) else None
    return research_context if isinstance(research_context, dict) else {}


def _context_datafields(context: Dict[str, Any]) -> Dict[str, Any]:
    datafields = _context_research_context(context).get("datafields")
    return datafields if isinstance(datafields, dict) else {}


def _allowed_fields_from_context(context: Dict[str, Any]) -> List[str]:
    datafields = _context_datafields(context)
    field_ids = datafields.get("field_ids")
    if isinstance(field_ids, list):
        return [str(field) for field in field_ids]
    fields = datafields.get("fields")
    if isinstance(fields, list):
        return [str(field["id"]) for field in fields if isinstance(field, dict) and field.get("id")]
    return []


def _field_metadata_from_context(context: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    fields = _context_datafields(context).get("fields")
    if not isinstance(fields, list):
        return {}
    metadata: Dict[str, Dict[str, Any]] = {}
    for field in fields:
        if isinstance(field, dict) and field.get("id"):
            metadata[str(field["id"])] = field
    return metadata


def _used_catalog_fields(expression: str, allowed_fields: List[str]) -> List[str]:
    used: List[str] = []
    for field in allowed_fields:
        text = str(field or "").strip()
        if text and re.search(rf"\b{re.escape(text)}\b", expression):
            used.append(text)
    return used


def _profile_guidance_text(guidance: Dict[str, Any]) -> str:
    parts: List[str] = []
    for key in ("objective", "field_family", "mechanism", "structure"):
        value = guidance.get(key)
        if value:
            parts.append(str(value))
    avoid = guidance.get("avoid")
    if isinstance(avoid, list):
        parts.extend(str(item) for item in avoid if str(item).strip())
    elif avoid:
        parts.append(str(avoid))
    return " ".join(parts).lower()


def _profile_guidance_direction_text(guidance: Dict[str, Any]) -> str:
    parts: List[str] = []
    for key in ("objective", "field_family", "mechanism", "structure"):
        value = guidance.get(key)
        if value:
            parts.append(str(value))
    return " ".join(parts).lower()


def _profile_bans_analyst_family(guidance_text: str, guidance: Dict[str, Any]) -> bool:
    if re.search(r"\bnon[-_\s]?analyst\b", _profile_guidance_direction_text(guidance)):
        return True
    if "avoid all analyst" in guidance_text or "all analyst fields" in guidance_text:
        return True
    avoid = guidance.get("avoid")
    avoid_items = avoid if isinstance(avoid, list) else [avoid]
    return any(str(item or "").strip().lower() in {"anl*", "anl_*", "analyst_*"} for item in avoid_items)


def _profile_requires_analyst_family(guidance: Dict[str, Any]) -> bool:
    field_family = str(guidance.get("field_family") or "").strip().lower()
    if not field_family or "non-analyst" in field_family or "non analyst" in field_family:
        return False
    return (
        "analyst primary only" in field_family
        or "analyst primaries only" in field_family
        or field_family.startswith("analyst primary")
        or field_family.startswith("analyst route")
        or field_family.startswith("analyst ")
    )


def _profile_requires_sentiment_family(guidance: Dict[str, Any]) -> bool:
    field_family = str(guidance.get("field_family") or "").strip().lower()
    if not field_family or "non-sentiment" in field_family or "non sentiment" in field_family:
        return False
    return (
        "sentiment23 matrix only" in field_family
        or "sentiment only" in field_family
        or "sentiment primaries only" in field_family
        or field_family.startswith("sentiment23 ")
        or field_family.startswith("sentiment ")
    )


def _uses_analyst_field(expression: str, used_fields: List[str], metadata: Dict[str, Dict[str, Any]]) -> bool:
    if re.search(r"\b(?:anl\d+_|analyst_|actual_update_)", expression):
        return True
    for field in used_fields:
        if _field_is_analyst(field, metadata.get(field, {})):
            return True
    return False


def _uses_non_analyst_catalog_field(used_fields: List[str], metadata: Dict[str, Dict[str, Any]]) -> bool:
    for field in used_fields:
        if field.lower() in PROFILE_HELPER_IDENTIFIERS:
            continue
        info = metadata.get(field)
        if not isinstance(info, dict) or not info:
            continue
        if not _field_is_analyst(field, info):
            return True
    return False


def _uses_non_family_catalog_field(
    used_fields: List[str],
    metadata: Dict[str, Dict[str, Any]],
    family: str,
) -> bool:
    for field in used_fields:
        if field.lower() in PROFILE_HELPER_IDENTIFIERS:
            continue
        info = metadata.get(field)
        if not isinstance(info, dict) or not info:
            continue
        if not _field_is_family(field, info, family):
            return True
    return False


def _field_is_analyst(field: str, info: Dict[str, Any]) -> bool:
    if re.match(r"^(?:anl\d+_|analyst_|actual_update_)", str(field or "")):
        return True
    category = str(info.get("category") or "").lower()
    dataset_id = str(info.get("dataset_id") or info.get("datasetId") or "").lower()
    dataset_name = str(info.get("dataset_name") or info.get("datasetName") or "").lower()
    return "analyst" in category or dataset_id.startswith(("anl", "analyst")) or "analyst" in dataset_name


def _uses_field_family(used_fields: List[str], metadata: Dict[str, Dict[str, Any]], family: str) -> bool:
    return any(_field_is_family(field, metadata.get(field, {}), family) for field in used_fields)


def _field_is_family(field: str, info: Dict[str, Any], family: str) -> bool:
    normalized = str(family or "").strip().upper()
    if normalized == "ANALYST":
        return _field_is_analyst(field, info)
    if normalized == "SENTIMENT":
        return _field_is_sentiment(field, info)
    text = str(field or "").lower()
    category = str(info.get("category") or "").lower()
    dataset_id = str(info.get("dataset_id") or info.get("datasetId") or "").lower()
    dataset_name = str(info.get("dataset_name") or info.get("datasetName") or "").lower()
    haystack = " ".join([text, category, dataset_id, dataset_name])
    if normalized == "NEWS":
        return "news" in haystack or "event" in category or dataset_id.startswith(("news", "nws"))
    if normalized == "SOCIALMEDIA":
        return "social" in haystack or dataset_id.startswith(("social", "scl"))
    if normalized == "FUNDAMENTAL":
        return "fundamental" in haystack or dataset_id.startswith(("fnd", "fundamental"))
    if normalized == "EARNINGS":
        return "earnings" in haystack or dataset_id.startswith(("ern", "earnings")) or text.startswith("eps_")
    if normalized == "INSIDERS":
        return "insider" in haystack or dataset_id.startswith(("insd", "insider"))
    if normalized == "INSTITUTIONS":
        return "institution" in haystack or dataset_id.startswith(("inst", "institution"))
    if normalized == "OPTION":
        return "option" in haystack or dataset_id.startswith(("option", "opt"))
    if normalized == "RISK":
        return "risk" in haystack or dataset_id.startswith("risk")
    if normalized == "PV":
        return category in {"price volume", "pv"} or dataset_id.startswith(("pv", "price", "volume"))
    if normalized == "MODEL":
        return "model" in haystack or dataset_id.startswith("model")
    if normalized == "OTHER":
        return text.startswith(("oth", "other_")) or "other" in category or dataset_id.startswith(("oth", "other"))
    return False


def _field_is_sentiment(field: str, info: Dict[str, Any]) -> bool:
    text = str(field or "").lower()
    if re.match(r"^(?:snt\d+|sentiment_)", text):
        return True
    category = str(info.get("category") or "").lower()
    dataset_id = str(info.get("dataset_id") or info.get("datasetId") or "").lower()
    dataset_name = str(info.get("dataset_name") or info.get("datasetName") or "").lower()
    return "sentiment" in category or dataset_id.startswith(("snt", "sentiment")) or "sentiment" in dataset_name


def _profile_bans_sentiment_family(guidance_text: str, guidance: Dict[str, Any]) -> bool:
    if re.search(r"\bnon[-_\s]?sentiment\b", _profile_guidance_direction_text(guidance)):
        return True
    if "all sentiment23 fields" in guidance_text or "all sentiment fields" in guidance_text:
        return True
    avoid = guidance.get("avoid")
    avoid_items = avoid if isinstance(avoid, list) else [avoid]
    return any(str(item or "").strip().lower() in {"snt*", "snt23_*", "sentiment_*"} for item in avoid_items)


def _profile_forbidden_field_patterns(guidance: Dict[str, Any]) -> List[str]:
    avoid = guidance.get("avoid")
    items = avoid if isinstance(avoid, list) else [avoid]
    patterns: List[str] = []
    for item in items:
        text = str(item or "").strip()
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_*]*", text) and "*" in text:
            patterns.append(text)
    return patterns


def _forbidden_field_pattern_hits(expression: str, used_fields: List[str], pattern: str) -> List[str]:
    escaped_pattern = re.escape(pattern).replace(r"\*", ".*")
    regex = re.compile(f"^{escaped_pattern}$")
    hits = [field for field in used_fields if regex.match(field)]
    if hits:
        return hits
    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", pattern) and re.search(rf"\b{re.escape(pattern)}\b", expression):
        return [pattern]
    return []
