from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Tuple

from .models import GuardResult


MANDATORY_CHECKS = {
    "selfcorrelation",
    "prodcorrelation",
    "productcorrelation",
    "datadiversity",
    "regularsubmission",
}

CORRELATION_CHECKS = {
    "selfcorrelation",
    "prodcorrelation",
    "productcorrelation",
}


@dataclass(frozen=True)
class SubmissionPolicy:
    min_sharpe: float = 1.58
    min_fitness: float = 1.0
    min_returns: float = 0.0
    min_turnover: float = 0.01
    max_turnover: float = 0.70
    max_correlation: float = 0.7
    max_final_submits_per_round: int = 4
    max_retries: int = 3
    auto_submit: bool = False


def normalize_check_name(name: str) -> str:
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


def _iter_checks(checks: Any) -> Iterable[Tuple[str, Dict[str, Any]]]:
    if isinstance(checks, dict):
        for name, data in checks.items():
            if isinstance(data, dict):
                yield str(name), data
        return
    if isinstance(checks, list):
        for item in checks:
            if isinstance(item, dict) and item.get("name"):
                yield str(item["name"]), item


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _is_regular_submission_quota_full(normalized_name: str, status: str, data: Dict[str, Any]) -> bool:
    if normalized_name != "regularsubmission" or status != "FAIL":
        return False
    value = _float(data.get("value"), default=-1.0)
    limit = _float(data.get("limit"), default=-1.0)
    return limit > 0 and value >= limit


def evaluate_submission_readiness(
    metrics: Dict[str, Any],
    checks: Any,
    policy: SubmissionPolicy,
    submitted_this_round: int,
) -> GuardResult:
    errors: List[str] = []
    warnings: List[str] = []

    if submitted_this_round >= policy.max_final_submits_per_round:
        errors.append(f"ROUND_SUBMIT_LIMIT_REACHED:{submitted_this_round}/{policy.max_final_submits_per_round}")

    sharpe = _float(metrics.get("sharpe"))
    fitness = _float(metrics.get("fitness"))
    returns = _float(metrics.get("returns"))
    turnover = _float(metrics.get("turnover"))

    if sharpe < policy.min_sharpe:
        errors.append(f"SHARPE_BELOW_MIN:{sharpe:.3f}<{policy.min_sharpe:g}")
    if fitness < policy.min_fitness:
        errors.append(f"FITNESS_BELOW_MIN:{fitness:.3f}<{policy.min_fitness:g}")
    if policy.min_returns > 0 and returns < policy.min_returns:
        errors.append(f"RETURNS_BELOW_MIN:{returns:.3f}<{policy.min_returns:g}")
    if turnover < policy.min_turnover:
        errors.append(f"TURNOVER_BELOW_MIN:{turnover:.3f}<{policy.min_turnover:g}")
    if turnover > policy.max_turnover:
        errors.append(f"TURNOVER_ABOVE_MAX:{turnover:.3f}>{policy.max_turnover:g}")

    seen = set()
    for raw_name, data in _iter_checks(checks):
        normalized = normalize_check_name(raw_name)
        seen.add(normalized)
        status = str(data.get("status") or data.get("result") or "").upper()
        value = data.get("value")

        if normalized in CORRELATION_CHECKS and value not in (None, "", "N/A"):
            corr = abs(_float(value, default=999.0))
            if corr > policy.max_correlation:
                errors.append(f"{raw_name}:{corr:.3f}>{policy.max_correlation:g}")

        if normalized in MANDATORY_CHECKS:
            if status != "PASS":
                if _is_regular_submission_quota_full(normalized, status, data):
                    errors.append(f"{raw_name}:QUOTA_FULL")
                else:
                    errors.append(f"{raw_name}:{status or 'MISSING_STATUS'}")
        elif status == "WARNING":
            warnings.append(raw_name)
        elif status in {"FAIL", "ERROR"}:
            errors.append(f"{raw_name}:{status}")

    missing = MANDATORY_CHECKS - seen
    # prodcorrelation and productcorrelation are aliases; only require one.
    if "prodcorrelation" in missing and "productcorrelation" not in missing:
        missing.discard("prodcorrelation")
    if "productcorrelation" in missing and "prodcorrelation" not in missing:
        missing.discard("productcorrelation")
    for name in sorted(missing):
        errors.append(f"{name}:MISSING")

    return GuardResult(ready=not errors, errors=errors, warnings=warnings, status="ready" if not errors else "blocked")
