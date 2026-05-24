from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

from .guards import SubmissionPolicy


@dataclass(frozen=True)
class AppConfig:
    db_path: Path = Path("alpha.db")
    batch_size: int = 8
    loop_seconds: float = 60.0
    ai_client: str = "local"
    brain_client: str = "local"
    brain_request_timeout_seconds: float = 60.0
    policy: SubmissionPolicy = SubmissionPolicy()
    simulation_context: Dict[str, Any] = field(default_factory=dict)


def _bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _int_env(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or str(value).strip() == "":
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be an integer") from exc


def _float_env(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or str(value).strip() == "":
        return default
    try:
        return float(value)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be a number") from exc


def load_config(db_path: str | None = None, batch_size: int | None = None) -> AppConfig:
    policy = SubmissionPolicy(
        auto_submit=_bool_env("AUTO_SUBMIT", False),
        max_retries=_int_env("MAX_RETRIES", 3),
        max_final_submits_per_round=_int_env("MAX_FINAL_SUBMITS_PER_ROUND", 4),
        min_sharpe=_float_env("MIN_SHARPE", 1.58),
        min_fitness=_float_env("MIN_FITNESS", 1.0),
        max_correlation=_float_env("MAX_CORRELATION", 0.7),
    )
    return AppConfig(
        db_path=Path(db_path or os.getenv("ALPHA_DB", "alpha.db")),
        batch_size=batch_size if batch_size is not None else _int_env("BATCH_SIZE", 8),
        loop_seconds=_float_env("LOOP_SECONDS", 60.0),
        ai_client=os.getenv("AI_CLIENT", "local").strip().lower(),
        brain_client=os.getenv("BRAIN_CLIENT", "local").strip().lower(),
        brain_request_timeout_seconds=_float_env("BRAIN_REQUEST_TIMEOUT_SECONDS", 60.0),
        policy=policy,
        simulation_context={
            "region": os.getenv("ALPHA_REGION", "USA"),
            "universe": os.getenv("ALPHA_UNIVERSE", "TOP3000"),
            "delay": _int_env("ALPHA_DELAY", 1),
            "decay": _int_env("ALPHA_DECAY", 0),
            "neutralization": os.getenv("ALPHA_NEUTRALIZATION", "INDUSTRY"),
            "truncation": _float_env("ALPHA_TRUNCATION", 0.05),
        },
    )
