# Throughput Alpha Agent Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a throughput-first personal Alpha research agent that uses stored results to prioritize pending checks, near-threshold optimization, setting sweeps, and bad-route cooldowns.

**Architecture:** Add deterministic read-only metrics, shared queue scoring, and a scheduler cycle plan before the existing worker execution loop. Keep the existing worker and guard behavior compatible by default, then add opt-in throughput mode for daemon runs and compact efficiency visibility in CLI/Web.

**Tech Stack:** Python 3.10, standard-library `unittest`, SQLite via existing `AlphaStore`, existing `CandidateSpec`, `SubmissionPolicy`, `AlphaWorker`, CLI, and standard-library web server.

---

## File Structure

- Create `src/alpha/metrics.py`
  - Read-only efficiency metrics from SQLite candidates and events.
  - No state transitions.
- Create `tests/test_metrics.py`
  - Unit tests for rates, grouping, and platform-error exclusion.
- Create `src/alpha/queues.py`
  - Shared candidate queue classification and compact queue items.
  - Used by scheduler first, then Web/context paths where appropriate.
- Create `tests/test_queues.py`
  - Unit tests for queue priority and hard-blocked candidates.
- Create `src/alpha/scheduler.py`
  - Deterministic `cycle_plan` builder.
  - Uses metrics and queues, but does not execute AI or BRAIN calls.
- Create `tests/test_scheduler.py`
  - Unit tests for pending, optimize, setting sweep, explore, cooldown, and platform-error handling.
- Modify `src/alpha/worker.py`
  - Accept optional `cycle_plan`.
  - Record `cycle_plan` and `cycle_outcome` events.
  - Preserve default behavior when no plan is supplied.
- Modify `tests/test_worker.py`
  - Add compatibility tests around optional `cycle_plan`.
- Modify `src/alpha/cli.py`
  - Add `status --efficiency`, `plan-next`, and `daemon --throughput-mode`.
- Modify `tests/test_cli.py`
  - Cover new CLI outputs and daemon argument flow.
- Modify `src/alpha/web.py`
  - Include efficiency metrics, scheduler preview, and cooldown summary in `/api/status`.
- Modify `tests/test_web.py`
  - Cover compact status additions.
- Modify `src/alpha/config.py`
  - Add explicit startup config parsing errors for numeric environment variables.
- Modify `src/alpha/clients.py`
  - Add central BRAIN HTTP request timeout wrapper.
- Modify `tests/test_config.py` and `tests/test_live_adapters.py`
  - Cover config validation and timeout forwarding/fallback.
- Create `src/alpha/health.py`
  - Personal daemon health/stall summary from run state and recent events.
- Create `tests/test_health.py`
  - Cover healthy, stale, stopped, and blocked states.

Do not stage broad existing uncommitted work. Each commit command below stages only the files for that task.

---

### Task 1: Read-Only Efficiency Metrics

**Files:**
- Create: `src/alpha/metrics.py`
- Create: `tests/test_metrics.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_metrics.py` with:

```python
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from alpha.db import AlphaStore
from alpha.metrics import compute_efficiency_metrics


def _store() -> AlphaStore:
    tmp = tempfile.TemporaryDirectory()
    store = AlphaStore(Path(tmp.name) / "alpha.db")
    store._tmp = tmp  # keep tempdir alive for the test lifetime
    store.init()
    return store


def _candidate(store: AlphaStore, expression: str, status: str, source: str = "model:g1", metrics=None, checks=None, settings=None):
    candidate_id = store.insert_candidate(
        expression,
        settings or {"region": "USA", "universe": "TOP3000", "delay": 1, "neutralization": "INDUSTRY"},
        source,
    )
    store.record_event(candidate_id, "generated", {"expression": expression})
    if status in {"preflight_passed", "simulated", "approved", "submitted", "check_pending", "failed"}:
        store.transition(candidate_id, "preflight_passed")
    if metrics is not None or checks is not None:
        store.update_candidate(
            candidate_id,
            metrics_json=json.dumps(metrics or {}, sort_keys=True),
            checks_json=json.dumps(checks or {}, sort_keys=True),
        )
        store.transition(candidate_id, "simulated", {"alpha_id": f"A{candidate_id}"})
    if status != "simulated":
        store.transition(candidate_id, status)
    return candidate_id


class EfficiencyMetricsTests(unittest.TestCase):
    def test_compute_efficiency_metrics_counts_rates_and_waste(self):
        store = _store()
        _candidate(
            store,
            "rank(alpha_signal)",
            "approved",
            metrics={"sharpe": 2.9, "fitness": 1.6, "turnover": 0.2},
            checks={"SELF_CORRELATION": {"status": "PASS"}},
        )
        failed_id = _candidate(
            store,
            "rank(bad_signal)",
            "failed",
            metrics={"sharpe": 0.1, "fitness": 0.02, "turnover": 0.2},
            checks={"LOW_SHARPE": {"status": "FAIL"}},
        )
        store.record_event(failed_id, "submission_guard", {"errors": ["SHARPE_BELOW_MIN:0.100<1.58"]})
        rejected_id = store.insert_candidate(
            "rank(invented_field)",
            {"region": "USA", "universe": "TOP3000", "delay": 1, "neutralization": "INDUSTRY"},
            "model:g2",
        )
        store.record_event(rejected_id, "generated", {"expression": "rank(invented_field)"})
        store.record_event(rejected_id, "preflight_failed", {"errors": ["UNKNOWN_FIELD:invented_field"]})
        store.transition(rejected_id, "failed", {"reason": "preflight"})
        store.record_event(None, "duplicate_candidate_skipped", {"source": "model:g1"})
        store.record_event(None, "structural_duplicate_candidate_skipped", {"source": "model:g1"})

        metrics = compute_efficiency_metrics(store)

        self.assertEqual(metrics["totals"]["generated"], 3)
        self.assertEqual(metrics["totals"]["preflight_passed"], 2)
        self.assertEqual(metrics["totals"]["simulated"], 2)
        self.assertEqual(metrics["totals"]["approved"], 1)
        self.assertEqual(metrics["totals"]["duplicate_skipped"], 2)
        self.assertAlmostEqual(metrics["rates"]["preflight_pass_rate"], 2 / 3)
        self.assertAlmostEqual(metrics["rates"]["approved_per_100_simulations"], 50.0)
        self.assertGreater(metrics["rates"]["simulation_waste_rate"], 0.0)

    def test_compute_efficiency_metrics_groups_by_scope_source_and_field_family(self):
        store = _store()
        _candidate(store, "rank(anl4_eps_est)", "approved", source="model:optimizer", metrics={"sharpe": 3.0, "fitness": 1.7})
        _candidate(
            store,
            "rank(snt23_score)",
            "failed",
            source="model:generator",
            metrics={"sharpe": 0.2, "fitness": 0.1},
            settings={"region": "CHN", "universe": "TOP2000U", "delay": 0, "neutralization": "INDUSTRY"},
        )

        metrics = compute_efficiency_metrics(store)

        self.assertIn("model:optimizer", metrics["by_source"])
        self.assertEqual(metrics["by_source"]["model:optimizer"]["approved"], 1)
        self.assertIn("USA|TOP3000|D1|INDUSTRY", metrics["by_scope"])
        self.assertIn("CHN|TOP2000U|D0|INDUSTRY", metrics["by_scope"])
        self.assertIn("analyst", metrics["by_field_family"])
        self.assertIn("sentiment", metrics["by_field_family"])

    def test_platform_rate_limit_errors_do_not_count_as_quality_waste(self):
        store = _store()
        candidate_id = _candidate(store, "rank(rate_limited_signal)", "failed")
        store.record_event(candidate_id, "simulation_error", {"error": "HTTP 429 Retry-After: 5"})

        metrics = compute_efficiency_metrics(store)

        self.assertEqual(metrics["totals"]["platform_error_failures"], 1)
        self.assertEqual(metrics["totals"]["quality_waste_failures"], 0)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
PYTHONPATH=src python3 -m unittest tests.test_metrics -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'alpha.metrics'`.

- [ ] **Step 3: Write minimal implementation**

Create `src/alpha/metrics.py`:

```python
from __future__ import annotations

import json
import re
from typing import Any, Dict, Iterable, List

from .db import AlphaStore
from .models import DEFAULT_SETTINGS


SIMULATED_STATUSES = {"simulated", "metric_passed", "approved", "submitted", "check_pending"}
APPROVED_STATUSES = {"approved", "submitted"}
PLATFORM_ERROR_MARKERS = ("429", "retry-after", "rate limit", "too many requests", "temporarily unavailable")


def compute_efficiency_metrics(
    store: AlphaStore,
    target_settings: Dict[str, Any] | None = None,
    *,
    created_since: str | None = None,
) -> Dict[str, Any]:
    rows = [
        row
        for row in store.list_candidates(created_since=created_since)
        if _scope_matches(_loads_dict(row.get("settings_json")), target_settings or {})
    ]
    global_events = store.events_for_candidate(None)
    totals = {
        "generated": len(rows),
        "preflight_passed": 0,
        "simulated": 0,
        "approved": 0,
        "submitted": 0,
        "pending": 0,
        "failed": 0,
        "duplicate_skipped": _duplicate_skip_count(global_events),
        "quality_waste_failures": 0,
        "platform_error_failures": 0,
        "near_threshold": 0,
    }
    by_source: Dict[str, Dict[str, Any]] = {}
    by_scope: Dict[str, Dict[str, Any]] = {}
    by_field_family: Dict[str, Dict[str, Any]] = {}

    for row in rows:
        status = str(row.get("status") or "")
        events = store.events_for_candidate(int(row["id"]))
        metrics = _loads_dict(row.get("metrics_json"))
        settings = _loads_dict(row.get("settings_json"))
        source = str(row.get("source") or "unknown")
        scope_key = _scope_key(settings)
        family = _field_family(str(row.get("expression") or ""))
        platform_error = _has_platform_error(events)
        simulated = _has_event(events, "status:simulated") or bool(row.get("alpha_id")) or status in SIMULATED_STATUSES
        preflight_passed = _has_event(events, "status:preflight_passed") or simulated or status in SIMULATED_STATUSES
        approved = status in APPROVED_STATUSES
        near_threshold = _near_threshold(metrics)

        if preflight_passed:
            totals["preflight_passed"] += 1
        if simulated:
            totals["simulated"] += 1
        if approved:
            totals["approved"] += 1
        if status == "submitted":
            totals["submitted"] += 1
        if status == "check_pending":
            totals["pending"] += 1
        if status == "failed":
            totals["failed"] += 1
            if platform_error:
                totals["platform_error_failures"] += 1
            elif simulated or metrics:
                totals["quality_waste_failures"] += 1
        if near_threshold:
            totals["near_threshold"] += 1

        _accumulate_group(by_source, source, status, preflight_passed, simulated, approved, platform_error)
        _accumulate_group(by_scope, scope_key, status, preflight_passed, simulated, approved, platform_error)
        _accumulate_group(by_field_family, family, status, preflight_passed, simulated, approved, platform_error)

    rates = {
        "preflight_pass_rate": _ratio(totals["preflight_passed"], totals["generated"]),
        "simulation_success_rate": _ratio(totals["simulated"], totals["preflight_passed"]),
        "approved_rate": _ratio(totals["approved"], totals["generated"]),
        "approved_per_100_simulations": 100.0 * _ratio(totals["approved"], totals["simulated"]),
        "duplicate_skip_rate": _ratio(totals["duplicate_skipped"], totals["generated"] + totals["duplicate_skipped"]),
        "simulation_waste_rate": _ratio(totals["quality_waste_failures"], totals["simulated"]),
    }
    return {
        "totals": totals,
        "rates": rates,
        "by_source": _finalize_groups(by_source),
        "by_scope": _finalize_groups(by_scope),
        "by_field_family": _finalize_groups(by_field_family),
    }


def _duplicate_skip_count(events: List[Dict[str, Any]]) -> int:
    return sum(
        1
        for event in events
        if str(event.get("event_type") or "") in {"duplicate_candidate_skipped", "structural_duplicate_candidate_skipped"}
    )


def _accumulate_group(
    groups: Dict[str, Dict[str, Any]],
    key: str,
    status: str,
    preflight_passed: bool,
    simulated: bool,
    approved: bool,
    platform_error: bool,
) -> None:
    item = groups.setdefault(
        key,
        {
            "generated": 0,
            "preflight_passed": 0,
            "simulated": 0,
            "approved": 0,
            "submitted": 0,
            "failed": 0,
            "platform_error_failures": 0,
        },
    )
    item["generated"] += 1
    if preflight_passed:
        item["preflight_passed"] += 1
    if simulated:
        item["simulated"] += 1
    if approved:
        item["approved"] += 1
    if status == "submitted":
        item["submitted"] += 1
    if status == "failed":
        item["failed"] += 1
        if platform_error:
            item["platform_error_failures"] += 1


def _finalize_groups(groups: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    finalized: Dict[str, Dict[str, Any]] = {}
    for key, item in groups.items():
        output = dict(item)
        output["preflight_pass_rate"] = _ratio(output["preflight_passed"], output["generated"])
        output["approved_per_100_simulations"] = 100.0 * _ratio(output["approved"], output["simulated"])
        finalized[key] = output
    return finalized


def _scope_matches(settings: Dict[str, Any], target_settings: Dict[str, Any]) -> bool:
    if not target_settings:
        return True
    merged_candidate = dict(DEFAULT_SETTINGS)
    merged_candidate.update(settings or {})
    merged_target = dict(DEFAULT_SETTINGS)
    merged_target.update(target_settings or {})
    for key in ("region", "universe", "delay", "neutralization"):
        if str(merged_candidate.get(key, "")).upper() != str(merged_target.get(key, "")).upper():
            return False
    return True


def _scope_key(settings: Dict[str, Any]) -> str:
    merged = dict(DEFAULT_SETTINGS)
    merged.update(settings or {})
    return (
        f"{str(merged.get('region') or '').upper()}|"
        f"{str(merged.get('universe') or '').upper()}|"
        f"D{merged.get('delay')}|"
        f"{str(merged.get('neutralization') or '').upper()}"
    )


def _field_family(expression: str) -> str:
    text = expression.lower()
    if re.search(r"\b(?:anl\d+_|analyst_|actual_update_)", text):
        return "analyst"
    if re.search(r"\b(?:snt\d+|sentiment_)", text):
        return "sentiment"
    if re.search(r"\b(?:close|open|high|low|volume|vwap|returns|adv20|cap)\b", text):
        return "price_volume"
    return "other"


def _near_threshold(metrics: Dict[str, Any]) -> bool:
    sharpe = _float(metrics.get("sharpe"))
    fitness = _float(metrics.get("fitness"))
    return sharpe >= 1.2 or fitness >= 0.65


def _has_event(events: Iterable[Dict[str, Any]], event_type: str) -> bool:
    return any(str(event.get("event_type") or "") == event_type for event in events)


def _has_platform_error(events: Iterable[Dict[str, Any]]) -> bool:
    for event in events:
        if str(event.get("event_type") or "") != "simulation_error":
            continue
        metadata = _loads_dict(event.get("metadata_json"))
        text = json.dumps(metadata, sort_keys=True).lower()
        if any(marker in text for marker in PLATFORM_ERROR_MARKERS):
            return True
    return False


def _loads_dict(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    try:
        data = json.loads(str(value or "{}"))
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _ratio(numerator: int | float, denominator: int | float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
PYTHONPATH=src python3 -m unittest tests.test_metrics -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/alpha/metrics.py tests/test_metrics.py
git commit -m "feat: add alpha efficiency metrics"
```

---

### Task 2: Shared Candidate Queues

**Files:**
- Create: `src/alpha/queues.py`
- Create: `tests/test_queues.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_queues.py` with:

```python
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from alpha.db import AlphaStore
from alpha.queues import build_candidate_queues, classify_candidate


def _store() -> AlphaStore:
    tmp = tempfile.TemporaryDirectory()
    store = AlphaStore(Path(tmp.name) / "alpha.db")
    store._tmp = tmp
    store.init()
    return store


def _candidate(store: AlphaStore, expression: str, status: str, metrics=None, checks=None):
    candidate_id = store.insert_candidate(
        expression,
        {"region": "USA", "universe": "TOP3000", "delay": 1, "neutralization": "INDUSTRY"},
        "model:g1",
    )
    if metrics is not None or checks is not None:
        store.update_candidate(
            candidate_id,
            metrics_json=json.dumps(metrics or {}, sort_keys=True),
            checks_json=json.dumps(checks or {}, sort_keys=True),
            alpha_id=f"A{candidate_id}",
        )
    if status != "generated":
        store.transition(candidate_id, status)
    return candidate_id


class CandidateQueueTests(unittest.TestCase):
    def test_build_candidate_queues_prioritizes_pending_and_near_threshold(self):
        store = _store()
        pending_id = _candidate(
            store,
            "rank(pending_signal)",
            "check_pending",
            metrics={"sharpe": 2.8, "fitness": 1.4, "turnover": 0.2},
            checks={"PROD_CORRELATION": {"status": "PENDING"}},
        )
        optimize_id = _candidate(
            store,
            "rank(optimize_signal)",
            "failed",
            metrics={"sharpe": 2.55, "fitness": 1.25, "turnover": 0.8},
            checks={"LOW_TURNOVER": {"status": "FAIL"}},
        )
        trash_id = _candidate(
            store,
            "rank(trash_signal)",
            "failed",
            metrics={"sharpe": 0.1, "fitness": 0.02, "turnover": 0.2},
            checks={"LOW_SHARPE": {"status": "FAIL"}},
        )

        queues = build_candidate_queues(store)

        self.assertEqual(queues["pending"][0]["id"], pending_id)
        self.assertEqual(queues["optimize"][0]["id"], optimize_id)
        self.assertEqual(queues["trash"][0]["id"], trash_id)
        self.assertGreater(queues["pending"][0]["priority"], queues["optimize"][0]["priority"])

    def test_classify_candidate_keeps_hard_correlation_block_out_of_optimize(self):
        store = _store()
        candidate_id = _candidate(
            store,
            "rank(correlated_signal)",
            "failed",
            metrics={"sharpe": 3.0, "fitness": 1.8, "turnover": 0.2},
            checks={"SELF_CORRELATION": {"status": "FAIL", "value": 0.91}},
        )
        row = store.get_candidate(candidate_id)

        queue, reason, priority = classify_candidate(row, store.events_for_candidate(candidate_id))

        self.assertEqual(queue, "trash")
        self.assertEqual(reason, "hard_blocker")
        self.assertLess(priority, 0)

    def test_build_candidate_queues_identifies_submitable_and_explore_seed(self):
        store = _store()
        approved_id = _candidate(store, "rank(good_signal)", "approved", metrics={"sharpe": 2.9, "fitness": 1.6})
        seed_id = _candidate(store, "rank(new_signal)", "generated")

        queues = build_candidate_queues(store)

        self.assertEqual(queues["submitable"][0]["id"], approved_id)
        self.assertEqual(queues["explore_seed"][0]["id"], seed_id)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
PYTHONPATH=src python3 -m unittest tests.test_queues -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'alpha.queues'`.

- [ ] **Step 3: Write minimal implementation**

Create `src/alpha/queues.py`:

```python
from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, Tuple


QUEUE_NAMES = ("submitable", "pending", "optimize", "watchlist", "explore_seed", "trash", "abandoned")
HARD_BLOCKER_NAMES = ("selfcorrelation", "prodcorrelation", "productcorrelation", "datadiversity")


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
        queues[name] = sorted(queues[name], key=lambda item: (float(item["priority"]), int(item["id"])), reverse=True)[:limit]
    return queues


def classify_candidate(row: Dict[str, Any], events: Iterable[Dict[str, Any]] | None = None) -> Tuple[str, str, float]:
    status = str(row.get("status") or "")
    metrics = _loads_dict(row.get("metrics_json"))
    checks = _loads_any(row.get("checks_json"))
    retry_count = int(row.get("retry_count") or 0)
    if status in {"approved", "submitted"}:
        return "submitable", "approved_or_submitted", 100.0 + _quality_score(metrics)
    if status == "check_pending":
        return "pending", "terminal_checks_waiting", 90.0 + _quality_score(metrics)
    if status in {"generated", "preflight_passed"}:
        return "explore_seed", status, 20.0
    if status == "failed":
        if _has_hard_blocker(checks) or _event_reason(events or [], "hard_blocker"):
            return "trash", "hard_blocker", -10.0
        if _near_threshold(metrics):
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


def _near_threshold(metrics: Dict[str, Any]) -> bool:
    sharpe = _float(metrics.get("sharpe"))
    fitness = _float(metrics.get("fitness"))
    turnover = _float(metrics.get("turnover"))
    return sharpe >= 1.45 or fitness >= 0.85 or (sharpe >= 1.2 and 0.01 <= turnover <= 0.9)


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
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
PYTHONPATH=src python3 -m unittest tests.test_queues -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/alpha/queues.py tests/test_queues.py
git commit -m "feat: add shared candidate queues"
```

---

### Task 3: Scheduler Preview Cycle Plan

**Files:**
- Create: `src/alpha/scheduler.py`
- Create: `tests/test_scheduler.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_scheduler.py` with:

```python
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from alpha.db import AlphaStore
from alpha.scheduler import build_cycle_plan


def _store() -> AlphaStore:
    tmp = tempfile.TemporaryDirectory()
    store = AlphaStore(Path(tmp.name) / "alpha.db")
    store._tmp = tmp
    store.init()
    return store


def _candidate(store: AlphaStore, expression: str, status: str, metrics=None, checks=None):
    candidate_id = store.insert_candidate(
        expression,
        {"region": "USA", "universe": "TOP3000", "delay": 1, "neutralization": "INDUSTRY"},
        "model:g1",
    )
    if metrics is not None or checks is not None:
        store.update_candidate(
            candidate_id,
            alpha_id=f"A{candidate_id}",
            metrics_json=json.dumps(metrics or {}, sort_keys=True),
            checks_json=json.dumps(checks or {}, sort_keys=True),
        )
    if status != "generated":
        store.transition(candidate_id, status)
    return candidate_id


class SchedulerTests(unittest.TestCase):
    def test_scheduler_recovers_pending_before_new_exploration(self):
        store = _store()
        pending_id = _candidate(
            store,
            "rank(pending_signal)",
            "check_pending",
            metrics={"sharpe": 2.8, "fitness": 1.4, "turnover": 0.2},
            checks={"PROD_CORRELATION": {"status": "PENDING"}},
        )

        plan = build_cycle_plan(store, {"region": "USA", "universe": "TOP3000", "delay": 1, "neutralization": "INDUSTRY"})

        self.assertEqual(plan["mode"], "recover_pending")
        self.assertEqual(plan["target_candidate_id"], pending_id)
        self.assertIn("pending", plan["reason"])

    def test_scheduler_optimizes_near_threshold_candidate(self):
        store = _store()
        optimize_id = _candidate(
            store,
            "rank(optimize_signal)",
            "failed",
            metrics={"sharpe": 2.55, "fitness": 1.2, "turnover": 0.8},
            checks={"LOW_TURNOVER": {"status": "FAIL"}},
        )

        plan = build_cycle_plan(store, {"region": "USA", "universe": "TOP3000", "delay": 1, "neutralization": "INDUSTRY"})

        self.assertEqual(plan["mode"], "optimize")
        self.assertEqual(plan["target_candidate_id"], optimize_id)

    def test_scheduler_prefers_setting_sweep_for_approved_dry_run_candidate(self):
        store = _store()
        approved_id = _candidate(
            store,
            "rank(good_signal)",
            "approved",
            metrics={"sharpe": 2.9, "fitness": 1.7, "turnover": 0.2},
            checks={"SELF_CORRELATION": {"status": "PASS"}},
        )

        plan = build_cycle_plan(store, {"region": "USA", "universe": "TOP3000", "delay": 1, "neutralization": "INDUSTRY"})

        self.assertEqual(plan["mode"], "setting_sweep")
        self.assertEqual(plan["target_candidate_id"], approved_id)

    def test_scheduler_defaults_to_explore_without_higher_value_queue(self):
        store = _store()

        plan = build_cycle_plan(store, {"region": "USA", "universe": "TOP3000", "delay": 1, "neutralization": "INDUSTRY"})

        self.assertEqual(plan["mode"], "explore")
        self.assertIsNone(plan["target_candidate_id"])

    def test_scheduler_cools_down_repeated_quality_stop_loss(self):
        store = _store()
        store.record_event(None, "quality_stop_loss", {"scope": {"region": "USA"}, "quality_stop_reason": "bad_full_batch"})
        store.record_event(None, "quality_stop_loss", {"scope": {"region": "USA"}, "quality_stop_reason": "bad_full_batch"})

        plan = build_cycle_plan(store, {"region": "USA", "universe": "TOP3000", "delay": 1, "neutralization": "INDUSTRY"})

        self.assertEqual(plan["mode"], "cooldown")
        self.assertIn("quality_stop_loss", plan["reason"])

    def test_scheduler_does_not_cool_down_for_platform_rate_limit_errors(self):
        store = _store()
        candidate_id = _candidate(store, "rank(rate_limited_signal)", "failed")
        store.record_event(candidate_id, "simulation_error", {"error": "HTTP 429 Retry-After: 5"})

        plan = build_cycle_plan(store, {"region": "USA", "universe": "TOP3000", "delay": 1, "neutralization": "INDUSTRY"})

        self.assertNotEqual(plan["mode"], "cooldown")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
PYTHONPATH=src python3 -m unittest tests.test_scheduler -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'alpha.scheduler'`.

- [ ] **Step 3: Write minimal implementation**

Create `src/alpha/scheduler.py`:

```python
from __future__ import annotations

import json
from typing import Any, Dict, List

from .metrics import compute_efficiency_metrics
from .queues import build_candidate_queues


def build_cycle_plan(
    store: Any,
    base_context: Dict[str, Any],
    *,
    batch_size: int = 8,
    created_since: str | None = None,
) -> Dict[str, Any]:
    scope = _scope(base_context)
    queues = build_candidate_queues(store, scope, limit=20)
    metrics = compute_efficiency_metrics(store, scope, created_since=created_since)
    cooldown_reason = _cooldown_reason(store)
    if cooldown_reason:
        return _plan("cooldown", scope, None, batch_size, cooldown_reason, metrics)
    if queues["pending"]:
        target = queues["pending"][0]
        return _plan("recover_pending", scope, int(target["id"]), min(batch_size, 4), "pending_candidate_has_existing_simulation", metrics)
    if queues["submitable"]:
        target = queues["submitable"][0]
        return _plan("setting_sweep", scope, int(target["id"]), min(batch_size, 8), "approved_candidate_may_have_setting_upside", metrics)
    if queues["optimize"]:
        target = queues["optimize"][0]
        return _plan("optimize", scope, int(target["id"]), min(batch_size, 4), "near_threshold_candidate_has_fixable_gap", metrics)
    return _plan("explore", scope, None, batch_size, "no_higher_value_queue_available", metrics)


def _plan(
    mode: str,
    scope: Dict[str, Any],
    target_candidate_id: int | None,
    batch_size: int,
    reason: str,
    metrics: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "mode": mode,
        "scope": scope,
        "target_candidate_id": target_candidate_id,
        "budget": {"batch_size": int(batch_size), "max_rounds": 3 if mode == "optimize" else 1},
        "constraints": {"avoid_structures": [], "cooldown_fields": []},
        "reason": reason,
        "metrics": {
            "totals": metrics.get("totals", {}),
            "rates": metrics.get("rates", {}),
        },
    }


def _scope(context: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "region": context.get("region", "USA"),
        "universe": context.get("universe", "TOP3000"),
        "delay": context.get("delay", 1),
        "neutralization": context.get("neutralization", "INDUSTRY"),
    }


def _cooldown_reason(store: Any) -> str:
    quality_stop_count = 0
    for event in store.events_for_candidate(None):
        event_type = str(event.get("event_type") or "")
        metadata = _loads_dict(event.get("metadata_json"))
        if event_type == "quality_stop_loss":
            quality_stop_count += 1
        if event_type == "simulation_error" and _platform_error(metadata):
            continue
    return "quality_stop_loss_repeated" if quality_stop_count >= 2 else ""


def _platform_error(metadata: Dict[str, Any]) -> bool:
    text = json.dumps(metadata, sort_keys=True).lower()
    return "429" in text or "retry-after" in text or "rate limit" in text


def _loads_dict(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    try:
        data = json.loads(str(value or "{}"))
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
PYTHONPATH=src python3 -m unittest tests.test_scheduler -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/alpha/scheduler.py tests/test_scheduler.py
git commit -m "feat: add throughput scheduler preview"
```

---

### Task 4: Worker Cycle Plan Compatibility

**Files:**
- Modify: `src/alpha/worker.py`
- Modify: `tests/test_worker.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_worker.py`:

```python
    def test_worker_records_optional_cycle_plan_and_outcome(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = AlphaStore(Path(tmp) / "alpha.db")
            store.init()
            cycle_plan = {
                "mode": "explore",
                "scope": {"region": "USA", "universe": "TOP3000", "delay": 1, "neutralization": "INDUSTRY"},
                "target_candidate_id": None,
                "reason": "test_plan",
            }
            worker = AlphaWorker(
                store=store,
                ai_client=LocalAIClient(),
                brain_client=LocalBrainClient(),
                policy=SubmissionPolicy(auto_submit=False),
                batch_size=1,
                context={"region": "USA", "universe": "TOP3000", "delay": 1, "neutralization": "INDUSTRY"},
                cycle_plan=cycle_plan,
            )

            summary = worker.run_once()

            events = store.events_for_candidate(None)
            event_types = [event["event_type"] for event in events]
            self.assertIn("cycle_plan", event_types)
            self.assertIn("cycle_outcome", event_types)
            self.assertEqual(summary["generated"], 1)

    def test_worker_run_once_accepts_call_level_cycle_plan(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = AlphaStore(Path(tmp) / "alpha.db")
            store.init()
            worker = AlphaWorker(
                store=store,
                ai_client=LocalAIClient(),
                brain_client=LocalBrainClient(),
                policy=SubmissionPolicy(auto_submit=False),
                batch_size=1,
                context={"region": "USA", "universe": "TOP3000", "delay": 1, "neutralization": "INDUSTRY"},
            )

            worker.run_once(cycle_plan={"mode": "explore", "reason": "call_level"})

            events = store.events_for_candidate(None)
            cycle_events = [event for event in events if event["event_type"] == "cycle_plan"]
            self.assertEqual(len(cycle_events), 1)
            self.assertEqual(json.loads(cycle_events[0]["metadata_json"])["reason"], "call_level")
```

- [ ] **Step 2: Run test to verify it fails**

Run the narrow tests by class:

```bash
PYTHONPATH=src python3 -m unittest tests.test_worker.WorkerTests.test_worker_records_optional_cycle_plan_and_outcome tests.test_worker.WorkerTests.test_worker_run_once_accepts_call_level_cycle_plan -v
```

Expected: FAIL with `TypeError: AlphaWorker.__init__() got an unexpected keyword argument 'cycle_plan'`.

- [ ] **Step 3: Write minimal implementation**

Modify `src/alpha/worker.py`:

```python
class AlphaWorker:
    def __init__(
        self,
        store: AlphaStore,
        ai_client: AIClient,
        brain_client: BrainClient,
        policy: SubmissionPolicy,
        batch_size: int = 4,
        context: Dict[str, Any] | None = None,
        cycle_plan: Dict[str, Any] | None = None,
    ):
        self.store = store
        self.ai_client = ai_client
        self.brain_client = brain_client
        self.policy = policy
        self.batch_size = batch_size
        self.context = context or {"region": "USA", "universe": "TOP3000", "delay": 1}
        self.cycle_plan = dict(cycle_plan or {})
        self.log = logging.getLogger("alpha.worker")
```

Change `_build_ai_context`:

```python
    def _build_ai_context(self, cycle_plan: Dict[str, Any] | None = None) -> Dict[str, Any]:
        active_cycle_plan = dict(cycle_plan or self.cycle_plan or {})
        ai_context = dict(self.context)
        if active_cycle_plan:
            ai_context["cycle_plan"] = active_cycle_plan
```

After `build_ai_research_context(...)`, add:

```python
        if active_cycle_plan:
            ai_context["research_context"]["cycle_plan"] = active_cycle_plan
            self.store.record_event(None, "cycle_plan", active_cycle_plan)
```

Change `run_once` signature and first context call:

```python
    def run_once(self, cycle_plan: Dict[str, Any] | None = None) -> Dict[str, int]:
        self.log.info("worker cycle started batch_size=%s", self.batch_size)
        summary = {"generated": 0, "approved": 0, "submitted": 0, "failed": 0, "pending": 0, "skipped": 0}
        submitted_this_round = 0
        candidates = None
        ai_context = self._build_ai_context(cycle_plan=cycle_plan)
```

Before the final `return summary`, record the outcome:

```python
        if cycle_plan or self.cycle_plan:
            self.store.record_event(
                None,
                "cycle_outcome",
                {"cycle_plan": cycle_plan or self.cycle_plan, "summary": summary},
            )
```

Also record `cycle_outcome` before early returns for field scout block and AI generation hard blocks:

```python
                if cycle_plan or self.cycle_plan:
                    self.store.record_event(None, "cycle_outcome", {"cycle_plan": cycle_plan or self.cycle_plan, "summary": summary})
                return summary
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
PYTHONPATH=src python3 -m unittest tests.test_worker.WorkerTests.test_worker_records_optional_cycle_plan_and_outcome tests.test_worker.WorkerTests.test_worker_run_once_accepts_call_level_cycle_plan -v
```

Expected: PASS.

- [ ] **Step 5: Run worker regression tests**

Run:

```bash
PYTHONPATH=src python3 -m unittest tests.test_worker -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/alpha/worker.py tests/test_worker.py
git commit -m "feat: let worker record throughput cycle plans"
```

---

### Task 5: CLI Scheduler Preview and Throughput Daemon Mode

**Files:**
- Modify: `src/alpha/cli.py`
- Modify: `tests/test_cli.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_cli.py`:

```python
    def test_cli_plan_next_prints_scheduler_plan(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "alpha.db"
            env_path = Path(tmp) / ".env"
            env_path.write_text("AI_CLIENT=local\nBRAIN_CLIENT=local\nAUTO_SUBMIT=false\n", encoding="utf-8")
            self.assertEqual(main(["--env-file", str(env_path), "--db", str(db_path), "init-db"]), 0)

            with patch("builtins.print") as printed:
                exit_code = main(["--env-file", str(env_path), "--db", str(db_path), "plan-next"])

            self.assertEqual(exit_code, 0)
            payload = json.loads(printed.call_args.args[0])
            self.assertEqual(payload["mode"], "explore")
            self.assertIn("reason", payload)

    def test_cli_status_efficiency_prints_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "alpha.db"
            env_path = Path(tmp) / ".env"
            env_path.write_text("AI_CLIENT=local\nBRAIN_CLIENT=local\nAUTO_SUBMIT=false\n", encoding="utf-8")
            self.assertEqual(main(["--env-file", str(env_path), "--db", str(db_path), "init-db"]), 0)

            with patch("builtins.print") as printed:
                exit_code = main(["--env-file", str(env_path), "--db", str(db_path), "status", "--efficiency"])

            self.assertEqual(exit_code, 0)
            output = "\n".join(str(call.args[0]) for call in printed.call_args_list)
            self.assertIn("generated:", output)
            self.assertIn("preflight_pass_rate:", output)

    def test_cli_daemon_throughput_mode_passes_cycle_plan_to_worker(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "alpha.db"
            env_path = Path(tmp) / ".env"
            env_path.write_text("AI_CLIENT=local\nBRAIN_CLIENT=local\nAUTO_SUBMIT=false\n", encoding="utf-8")
            self.assertEqual(main(["--env-file", str(env_path), "--db", str(db_path), "init-db"]), 0)

            with patch("alpha.cli.time.sleep", side_effect=KeyboardInterrupt):
                exit_code = main(
                    [
                        "--env-file",
                        str(env_path),
                        "--db",
                        str(db_path),
                        "daemon",
                        "--throughput-mode",
                        "--batch-size",
                        "1",
                        "--loop-seconds",
                        "1",
                    ]
                )

            self.assertEqual(exit_code, 0)
            events = AlphaStore(db_path).events_for_candidate(None)
            self.assertTrue(any(event["event_type"] == "cycle_plan" for event in events))
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
PYTHONPATH=src python3 -m unittest tests.test_cli.CliTests.test_cli_plan_next_prints_scheduler_plan tests.test_cli.CliTests.test_cli_status_efficiency_prints_metrics tests.test_cli.CliTests.test_cli_daemon_throughput_mode_passes_cycle_plan_to_worker -v
```

Expected: FAIL because parser has no `plan-next`, `status --efficiency`, or `daemon --throughput-mode`.

- [ ] **Step 3: Write minimal implementation**

Modify imports in `src/alpha/cli.py`:

```python
import json
from .metrics import compute_efficiency_metrics
from .scheduler import build_cycle_plan
```

Change parser setup:

```python
    status = sub.add_parser("status", help="打印候选状态统计")
    status.add_argument("--efficiency", action="store_true", help="打印个人效率指标")
```

Add daemon flag:

```python
    daemon.add_argument("--throughput-mode", action="store_true", help="使用 scheduler cycle_plan 提升个人研究吞吐")
```

Add new subcommand:

```python
    plan_next = sub.add_parser("plan-next", help="预览下一轮 throughput scheduler 计划")
    plan_next.add_argument("--batch-size", type=int, default=None)
    _add_scope_args(plan_next)
```

In daemon loop before worker construction:

```python
                cycle_plan = None
                if getattr(args, "throughput_mode", False):
                    cycle_plan = build_cycle_plan(store, cycle_context, batch_size=cfg.batch_size)
                    cycle_context = dict(cycle_context)
                    if isinstance(cycle_plan.get("scope"), dict):
                        cycle_context.update(cycle_plan["scope"])
                    log.info("daemon_cycle_plan=%s", cycle_plan)
```

Then pass plan:

```python
                summary = _worker(store, cfg.batch_size, cfg.policy, cfg.ai_client, cfg.brain_client, cycle_context).run_once(
                    cycle_plan=cycle_plan
                )
```

For non-throughput mode, `cycle_plan` is `None` and behavior stays compatible.

In `status` branch:

```python
    if args.command == "status":
        counts = store.status_counts()
        log.info("status counts=%s", counts)
        if not counts:
            print("no candidates")
        for status, count in counts.items():
            print(f"{status}: {count}")
        if getattr(args, "efficiency", False):
            metrics = compute_efficiency_metrics(store, simulation_context)
            for key, value in metrics["totals"].items():
                print(f"{key}: {value}")
            for key, value in metrics["rates"].items():
                print(f"{key}: {value:.6f}")
        return 0
```

Add `plan-next` branch before `status` or after `presets`:

```python
    if args.command == "plan-next":
        plan = build_cycle_plan(store, simulation_context, batch_size=cfg.batch_size)
        log.info("plan_next=%s", plan)
        print(json.dumps(plan, sort_keys=True))
        return 0
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
PYTHONPATH=src python3 -m unittest tests.test_cli.CliTests.test_cli_plan_next_prints_scheduler_plan tests.test_cli.CliTests.test_cli_status_efficiency_prints_metrics tests.test_cli.CliTests.test_cli_daemon_throughput_mode_passes_cycle_plan_to_worker -v
```

Expected: PASS.

- [ ] **Step 5: Run CLI regression tests**

Run:

```bash
PYTHONPATH=src python3 -m unittest tests.test_cli -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/alpha/cli.py tests/test_cli.py
git commit -m "feat: add throughput scheduler CLI"
```

---

### Task 6: Web Efficiency Status

**Files:**
- Modify: `src/alpha/web.py`
- Modify: `tests/test_web.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_web.py`:

```python
    def test_control_service_status_includes_efficiency_and_scheduler_plan(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            store = AlphaStore(base / "alpha.db")
            store.init()
            candidate_id = store.insert_candidate(
                "rank(close)",
                {"region": "USA", "universe": "TOP3000", "delay": 1, "neutralization": "INDUSTRY"},
                "local_ai",
            )
            store.transition(candidate_id, "approved")
            service = ControlService(
                store=store,
                db_path=store.path,
                env_file=base / ".env",
                log_file=base / "alpha.log",
                daemon_stdout_log=base / "daemon.log",
                web_log=base / "web.log",
            )

            status = service.status()

            self.assertIn("efficiency", status)
            self.assertIn("scheduler_plan", status)
            self.assertIn("cooldowns", status)
            self.assertEqual(status["scheduler_plan"]["mode"], "setting_sweep")
            self.assertIn("approved_per_100_simulations", status["efficiency"]["rates"])

    def test_control_panel_exposes_efficiency_panel(self):
        self.assertIn('id="efficiency_metrics"', HTML)
        self.assertIn('id="scheduler_plan"', HTML)
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
PYTHONPATH=src python3 -m unittest tests.test_web.WebControlTests.test_control_service_status_includes_efficiency_and_scheduler_plan tests.test_web.WebControlTests.test_control_panel_exposes_efficiency_panel -v
```

Expected: FAIL because status and HTML do not expose these fields yet.

- [ ] **Step 3: Write minimal implementation**

Modify imports in `src/alpha/web.py`:

```python
from .metrics import compute_efficiency_metrics
from .scheduler import build_cycle_plan
```

Inside `ControlService.status`, after `research = self._research_context(state)`:

```python
        active_scope = state.get("scope") if isinstance(state.get("scope"), dict) else {}
        if not active_scope:
            active_scope = research.get("target_settings") if isinstance(research.get("target_settings"), dict) else {}
        efficiency = compute_efficiency_metrics(self.store, active_scope if isinstance(active_scope, dict) else {}, created_since=run_started_at or None)
        scheduler_plan = build_cycle_plan(self.store, active_scope if isinstance(active_scope, dict) else {}, batch_size=int(state.get("batch_size") or MAX_WEB_BACKTEST_BATCH))
```

Add to payload:

```python
            "efficiency": efficiency,
            "scheduler_plan": scheduler_plan,
            "cooldowns": scheduler_plan.get("constraints", {}),
```

Add simple HTML anchors in `HTML` near the existing status panels:

```html
<section>
  <h2>Efficiency</h2>
  <pre id="efficiency_metrics"></pre>
</section>
<section>
  <h2>Scheduler</h2>
  <pre id="scheduler_plan"></pre>
</section>
```

In the existing frontend status render function, add:

```javascript
document.getElementById("efficiency_metrics").textContent = JSON.stringify(data.efficiency || {}, null, 2);
document.getElementById("scheduler_plan").textContent = JSON.stringify(data.scheduler_plan || {}, null, 2);
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
PYTHONPATH=src python3 -m unittest tests.test_web.WebControlTests.test_control_service_status_includes_efficiency_and_scheduler_plan tests.test_web.WebControlTests.test_control_panel_exposes_efficiency_panel -v
```

Expected: PASS.

- [ ] **Step 5: Run Web regression tests**

Run:

```bash
PYTHONPATH=src python3 -m unittest tests.test_web -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/alpha/web.py tests/test_web.py
git commit -m "feat: show throughput efficiency in web status"
```

---

### Task 7: Startup Config Validation and BRAIN HTTP Timeouts

**Files:**
- Modify: `src/alpha/config.py`
- Modify: `src/alpha/clients.py`
- Modify: `tests/test_config.py`
- Modify: `tests/test_live_adapters.py`

- [ ] **Step 1: Write the failing config tests**

Append to `tests/test_config.py`:

```python
    def test_config_reports_invalid_numeric_environment_value(self):
        with patch.dict(os.environ, {"BATCH_SIZE": "not-a-number"}, clear=False):
            with self.assertRaisesRegex(RuntimeError, "BATCH_SIZE must be an integer"):
                load_config()

    def test_config_reads_brain_request_timeout(self):
        with patch.dict(os.environ, {"BRAIN_REQUEST_TIMEOUT_SECONDS": "12.5"}, clear=False):
            cfg = load_config()

        self.assertEqual(cfg.brain_request_timeout_seconds, 12.5)
```

- [ ] **Step 2: Write the failing HTTP timeout test**

Append to `tests/test_live_adapters.py`:

```python
    def test_brain_http_client_passes_request_timeout_to_session(self):
        class TimeoutSession:
            def __init__(self):
                self.calls = []

            def get(self, url, **kwargs):
                self.calls.append(("get", url, kwargs))
                return FakeResponse(200, {"results": []})

        session = TimeoutSession()
        client = BrainHTTPClient(session=session, request_timeout=7.5)

        client.recent_submitted_alphas(limit=1)

        self.assertEqual(session.calls[0][2]["timeout"], 7.5)

    def test_brain_http_client_falls_back_when_fake_session_rejects_timeout(self):
        class LegacySession:
            def get(self, url, **kwargs):
                if "timeout" in kwargs:
                    raise TypeError("unexpected timeout")
                return FakeResponse(200, {"results": []})

        client = BrainHTTPClient(session=LegacySession(), request_timeout=7.5)

        self.assertEqual(client.recent_submitted_alphas(limit=1), [])
```

- [ ] **Step 3: Run tests to verify they fail**

Run:

```bash
PYTHONPATH=src python3 -m unittest tests.test_config tests.test_live_adapters -v
```

Expected: FAIL because config has no explicit numeric error and `BrainHTTPClient` does not accept/pass request timeout.

- [ ] **Step 4: Implement config validation**

Modify `src/alpha/config.py`:

```python
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
```

Add helpers:

```python
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
```

Use them in `load_config`:

```python
        max_retries=_int_env("MAX_RETRIES", 3),
        max_final_submits_per_round=_int_env("MAX_FINAL_SUBMITS_PER_ROUND", 4),
        min_sharpe=_float_env("MIN_SHARPE", 1.58),
        min_fitness=_float_env("MIN_FITNESS", 1.0),
        max_correlation=_float_env("MAX_CORRELATION", 0.7),
```

And:

```python
        batch_size=batch_size if batch_size is not None else _int_env("BATCH_SIZE", 8),
        loop_seconds=_float_env("LOOP_SECONDS", 60.0),
        brain_request_timeout_seconds=_float_env("BRAIN_REQUEST_TIMEOUT_SECONDS", 60.0),
```

For simulation context:

```python
            "delay": _int_env("ALPHA_DELAY", 1),
            "decay": _int_env("ALPHA_DECAY", 0),
            "truncation": _float_env("ALPHA_TRUNCATION", 0.05),
```

- [ ] **Step 5: Implement BRAIN HTTP timeout wrappers**

Modify `BrainHTTPClient.__init__` in `src/alpha/clients.py`:

```python
    def __init__(
        self,
        session: Any | None = None,
        base_url: str = "https://api.worldquantbrain.com",
        max_poll_attempts: int = 5,
        sleep: Callable[[float], None] = time.sleep,
        request_timeout: float = 60.0,
    ):
        self.session = session or self._default_session()
        self.base_url = base_url.rstrip("/")
        self.max_poll_attempts = max_poll_attempts
        self.sleep = sleep
        self.request_timeout = max(1.0, float(request_timeout))
```

Modify `from_env`:

```python
        client = cls(
            base_url=os.environ.get("BRAIN_BASE_URL", "https://api.worldquantbrain.com"),
            request_timeout=float(os.environ.get("BRAIN_REQUEST_TIMEOUT_SECONDS", "60")),
        )
```

Add methods:

```python
    def _get(self, url: str, **kwargs: Any) -> Any:
        return self._request("get", url, **kwargs)

    def _post(self, url: str, **kwargs: Any) -> Any:
        return self._request("post", url, **kwargs)

    def _request(self, method: str, url: str, **kwargs: Any) -> Any:
        call = getattr(self.session, method)
        try:
            return call(url, timeout=self.request_timeout, **kwargs)
        except TypeError as exc:
            if "timeout" not in str(exc).lower():
                raise
            return call(url, **kwargs)
```

Replace direct `self.session.get(...)` with `self._get(...)` and direct `self.session.post(...)` with `self._post(...)` inside `BrainHTTPClient`. Leave OpenAI-compatible AI client transport untouched.

- [ ] **Step 6: Run tests to verify they pass**

Run:

```bash
PYTHONPATH=src python3 -m unittest tests.test_config tests.test_live_adapters -v
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/alpha/config.py src/alpha/clients.py tests/test_config.py tests/test_live_adapters.py
git commit -m "feat: validate config and bound brain http calls"
```

---

### Task 8: Daemon Health and Final Verification

**Files:**
- Create: `src/alpha/health.py`
- Create: `tests/test_health.py`
- Modify: `src/alpha/web.py`
- Modify: `tests/test_web.py`

- [ ] **Step 1: Write the failing health tests**

Create `tests/test_health.py`:

```python
from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from alpha.db import AlphaStore
from alpha.health import daemon_health


def _iso(minutes_ago: int) -> str:
    return (datetime.now(timezone.utc) - timedelta(minutes=minutes_ago)).replace(microsecond=0).isoformat()


class HealthTests(unittest.TestCase):
    def test_daemon_health_reports_stopped_when_no_running_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = AlphaStore(Path(tmp) / "alpha.db")
            store.init()

            health = daemon_health(store)

            self.assertEqual(health["status"], "stopped")
            self.assertEqual(health["stalled"], False)

    def test_daemon_health_reports_stalled_running_daemon_without_recent_events(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = AlphaStore(Path(tmp) / "alpha.db")
            store.init()
            store.set_run_state("daemon", {"status": "running", "pid": 123, "started_at": _iso(120)})

            health = daemon_health(store, stall_minutes=60)

            self.assertEqual(health["status"], "running")
            self.assertEqual(health["stalled"], True)

    def test_daemon_health_reports_recent_block_reason(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = AlphaStore(Path(tmp) / "alpha.db")
            store.init()
            store.set_run_state("daemon", {"status": "stopped", "stop_reason": "ai_quota_blocked"})
            store.record_event(None, "daemon_stopped", {"reason": "ai_quota_blocked"})

            health = daemon_health(store)

            self.assertEqual(health["last_block_reason"], "ai_quota_blocked")
```

- [ ] **Step 2: Write the failing Web status test**

Append to `tests/test_web.py`:

```python
    def test_control_service_status_includes_daemon_health(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            store = AlphaStore(base / "alpha.db")
            store.init()
            service = ControlService(
                store=store,
                db_path=store.path,
                env_file=base / ".env",
                log_file=base / "alpha.log",
                daemon_stdout_log=base / "daemon.log",
                web_log=base / "web.log",
            )

            status = service.status()

            self.assertIn("health", status)
            self.assertEqual(status["health"]["status"], "stopped")
```

- [ ] **Step 3: Run tests to verify they fail**

Run:

```bash
PYTHONPATH=src python3 -m unittest tests.test_health tests.test_web.WebControlTests.test_control_service_status_includes_daemon_health -v
```

Expected: FAIL because `alpha.health` does not exist and Web status has no health field.

- [ ] **Step 4: Implement health summary**

Create `src/alpha/health.py`:

```python
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Dict


def daemon_health(store: Any, *, stall_minutes: int = 60) -> Dict[str, Any]:
    state = store.get_run_state("daemon")
    status = str(state.get("status") or "stopped")
    started_at = str(state.get("started_at") or "")
    last_block_reason = str(state.get("stop_reason") or "")
    last_event_at = ""
    for event in reversed(store.events_for_candidate(None)):
        event_type = str(event.get("event_type") or "")
        if not last_event_at:
            last_event_at = str(event.get("created_at") or "")
        if event_type in {"daemon_stopped", "ai_generation_error", "quality_stop_loss"} and not last_block_reason:
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
```

Modify imports in `src/alpha/web.py`:

```python
from .health import daemon_health
```

Add to `ControlService.status` payload:

```python
            "health": daemon_health(self.store),
```

- [ ] **Step 5: Run tests to verify they pass**

Run:

```bash
PYTHONPATH=src python3 -m unittest tests.test_health tests.test_web.WebControlTests.test_control_service_status_includes_daemon_health -v
```

Expected: PASS.

- [ ] **Step 6: Run full verification**

Run:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/alpha/health.py tests/test_health.py src/alpha/web.py tests/test_web.py
git commit -m "feat: add daemon health summary"
```

---

## Final Verification

After all tasks:

- [ ] Run the full suite:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

Expected: PASS.

- [ ] Run a local scheduler preview:

```bash
PYTHONPATH=src python3 -m alpha.cli --db alpha.db plan-next
```

Expected: JSON with `mode`, `scope`, `target_candidate_id`, `budget`, and `reason`.

- [ ] Run local efficiency status:

```bash
PYTHONPATH=src python3 -m alpha.cli --db alpha.db status --efficiency
```

Expected: status counts plus `generated`, `preflight_pass_rate`, `approved_per_100_simulations`, and related metrics.

- [ ] Run one local dry loop:

```bash
PYTHONPATH=src python3 -m alpha.cli --db /tmp/alpha-throughput-test.db --env-file .env.example daemon --throughput-mode --batch-size 1 --loop-seconds 1 --run-minutes 0.02
```

Expected: daemon exits by time limit, records `cycle_plan`, and does not submit.

## Self-Review

Spec coverage:

- Read-only efficiency metrics: Task 1.
- Shared candidate queues: Task 2.
- Scheduler cycle plan and route cooldown: Task 3.
- Worker compatibility and outcome recording: Task 4.
- CLI efficiency/status and throughput daemon mode: Task 5.
- Web efficiency and scheduler visibility: Task 6.
- Startup config validation and HTTP timeout hardening: Task 7.
- Daemon health/stall summary: Task 8.
- Full verification and local dry loop: Final Verification.

Red-flag scan:

- No unresolved-marker instructions remain.
- Each code-changing task has concrete test snippets, implementation snippets, commands, and expected outcomes.

Type consistency:

- Scheduler returns a plain `dict` cycle plan with `mode`, `scope`, `target_candidate_id`, `budget`, `constraints`, `reason`, and `metrics`.
- Worker accepts `cycle_plan` both in `__init__` and `run_once`.
- CLI/Web use `compute_efficiency_metrics` and `build_cycle_plan`.
- Health summary key is `health` in Web status and `daemon_health()` in `src/alpha/health.py`.
