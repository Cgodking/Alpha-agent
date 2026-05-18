from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


DEFAULT_SETTINGS = {
    "instrumentType": "EQUITY",
    "region": "USA",
    "universe": "TOP3000",
    "delay": 1,
    "decay": 0,
    "neutralization": "INDUSTRY",
    "truncation": 0.05,
    "pasteurization": "ON",
    "unitHandling": "VERIFY",
    "nanHandling": "OFF",
    "language": "FASTEXPR",
    "visualization": False,
}


@dataclass(frozen=True)
class CandidateSpec:
    expression: str
    settings: Dict[str, Any] = field(default_factory=dict)
    source: str = "local_ai"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SimulationResult:
    alpha_id: str
    metrics: Dict[str, Any]
    checks: Dict[str, Dict[str, Any]]
    raw: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SimulationFailure:
    error: str
    raw: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SubmitResult:
    alpha_id: str
    submitted: bool
    stage: str
    message: str = ""


@dataclass(frozen=True)
class GuardResult:
    ready: bool
    errors: List[str]
    warnings: List[str] = field(default_factory=list)
    status: str = "blocked"


@dataclass(frozen=True)
class WorkerSummary:
    generated: int = 0
    approved: int = 0
    submitted: int = 0
    failed: int = 0
    pending: int = 0

    def as_dict(self) -> Dict[str, int]:
        return {
            "generated": self.generated,
            "approved": self.approved,
            "submitted": self.submitted,
            "failed": self.failed,
            "pending": self.pending,
        }
