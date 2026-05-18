from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Tuple

from .clients import BrainClient
from .db import AlphaStore
from .guards import SubmissionPolicy


def trading_day_window(now: datetime | None = None) -> Tuple[str, str]:
    current = now or datetime.now(timezone.utc)
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)
    noon = current.replace(hour=12, minute=0, second=0, microsecond=0)
    start = noon if current >= noon else noon - timedelta(days=1)
    end = start + timedelta(days=1)
    return start.isoformat().replace("+00:00", "Z"), end.isoformat().replace("+00:00", "Z")


def submit_approved_candidates(
    store: AlphaStore,
    brain_client: BrainClient,
    policy: SubmissionPolicy,
) -> Dict[str, int]:
    log = logging.getLogger("alpha.submission")
    approved = store.list_candidates(status="approved")
    platform_count = _platform_submitted_count(brain_client)
    if platform_count is None:
        summary = {"processed": 0, "submitted": 0, "dry_run": 0, "failed": 0, "skipped": len(approved)}
        for candidate in approved:
            store.record_event(candidate["id"], "submit_skipped", {"reason": "platform_count_unavailable"})
        return summary
    remaining = max(0, policy.max_final_submits_per_round - platform_count)
    limit = min(policy.max_final_submits_per_round, remaining)
    summary = {"processed": 0, "submitted": 0, "dry_run": 0, "failed": 0, "skipped": 0}

    for index, candidate in enumerate(approved):
        if index >= limit:
            summary["skipped"] += 1
            store.record_event(candidate["id"], "submit_skipped", {"reason": "round_limit", "platform_count": platform_count})
            continue

        alpha_id = candidate.get("alpha_id")
        if not alpha_id:
            summary["failed"] += 1
            store.record_event(candidate["id"], "submit_failed", {"reason": "missing_alpha_id"})
            continue

        summary["processed"] += 1
        result = brain_client.submit_alpha(str(alpha_id), dry_run=not policy.auto_submit)
        if result.submitted and result.stage == "OS":
            summary["submitted"] += 1
            store.transition(candidate["id"], "submitted", {"alpha_id": alpha_id})
            log.info("approved candidate submitted id=%s alpha_id=%s", candidate["id"], alpha_id)
        elif result.stage == "DRY_RUN":
            summary["dry_run"] += 1
            store.record_event(
                candidate["id"],
                "dry_run_submit",
                {"alpha_id": alpha_id, "stage": result.stage, "message": result.message},
            )
            log.info("approved candidate dry-run id=%s alpha_id=%s", candidate["id"], alpha_id)
        else:
            summary["failed"] += 1
            store.record_event(
                candidate["id"],
                "submit_failed",
                {"alpha_id": alpha_id, "stage": result.stage, "message": result.message},
            )
            log.warning("approved candidate submit failed id=%s alpha_id=%s stage=%s", candidate["id"], alpha_id, result.stage)

    return summary


def _platform_submitted_count(brain_client: BrainClient) -> int | None:
    counter = getattr(brain_client, "count_submitted_alphas", None)
    if not callable(counter):
        return 0
    start, end = trading_day_window()
    try:
        return int(counter(start, end))
    except Exception:
        logging.getLogger("alpha.submission").exception("platform submission count failed; skipping submit")
        return None
