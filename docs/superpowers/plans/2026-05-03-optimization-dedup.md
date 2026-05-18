# Optimization-Aware Dedup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let near-threshold optimization candidates continue through a limited repair loop while still blocking true local duplicates and stale replayed variants.

**Architecture:** Keep exact duplicate detection unchanged, but relax structural deduplication for `optimize_best` and `setting_sweep` so a candidate can continue to evolve across a small number of rounds. Structural duplicate checks should remain active for ordinary exploration batches and AI response cleanup. Historical deduplication should only consider candidates that have actually completed simulation, not rows that merely passed preflight.

**Tech Stack:** Python 3.10, SQLite, unittest, existing `AlphaStore`, `AlphaWorker`, `OpenAICompatibleAIClient`, `MultiModelAIClient`.

---

### Task 1: Narrow historical deduplication to real simulation history

**Files:**
- Modify: `src/alpha/worker.py`
- Modify: `src/alpha/context_builder.py`
- Modify: `tests/test_worker.py`
- Modify: `tests/test_context_builder.py`

- [ ] **Step 1: Write the failing test**

```python
def test_preflight_only_candidate_does_not_block_future_structural_variant():
    # A row that only reached preflight_passed should not count as a finished
    # historical structural duplicate when the next candidate is still in an
    # optimize/repair cycle.
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src python3 -m unittest tests.test_worker.TestWorker... -v`
Expected: the candidate is skipped too early because the current history scan sees preflight-only rows.

- [ ] **Step 3: Write minimal implementation**

```python
def _candidate_completed_simulation(row: Dict[str, Any]) -> bool:
    status = str(row.get("status") or "")
    return status in {"approved", "submitted", "failed", "check_pending", "simulated"}
```

Use this filter in `_recent_expression_structures()` and `_find_structural_duplicate()` so only completed candidates participate in historical structural deduplication.

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src python3 -m unittest tests.test_worker tests.test_context_builder -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/alpha/worker.py src/alpha/context_builder.py tests/test_worker.py tests/test_context_builder.py
git commit -m "feat: limit historical structural dedup to simulated candidates"
```

### Task 2: Allow limited optimization-phase structural retries

**Files:**
- Modify: `src/alpha/worker.py`
- Modify: `src/alpha/clients.py`
- Modify: `src/alpha/research_planner.py`
- Modify: `tests/test_worker.py`
- Modify: `tests/test_ai_integration.py`

- [ ] **Step 1: Write the failing test**

```python
def test_optimize_best_can_keep_structural_variants_for_three_rounds():
    # The same near-threshold anchor should be allowed to emit a small number
    # of controlled structural variants instead of being blocked immediately.
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src python3 -m unittest tests.test_worker.TestWorker... -v`
Expected: the second-round optimization candidate is skipped by structural dedup.

- [ ] **Step 3: Write minimal implementation**

```python
def _allow_optimization_structural_retry(ai_context: Dict[str, Any], spec: CandidateSpec) -> bool:
    research_context = ai_context.get("research_context") if isinstance(ai_context.get("research_context"), dict) else {}
    experiment_plan = research_context.get("experiment_plan") if isinstance(research_context, dict) else {}
    if not isinstance(experiment_plan, dict):
        return False
    return str(experiment_plan.get("mode") or "") in {"optimize_best", "setting_sweep"}
```

Allow a candidate to bypass structural duplicate rejection when:
1. The current plan is `optimize_best` or `setting_sweep`.
2. The candidate is derived from the current optimization anchor.
3. The round count has not exceeded the configured limit.

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src python3 -m unittest tests.test_worker tests.test_ai_integration -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/alpha/worker.py src/alpha/clients.py src/alpha/research_planner.py tests/test_worker.py tests/test_ai_integration.py
git commit -m "feat: preserve limited optimization retries for near-threshold candidates"
```

### Task 3: Keep the web/control panel responsive after the new retry logic

**Files:**
- Modify: `src/alpha/web.py`
- Modify: `src/alpha/db.py`
- Modify: `tests/test_web.py`
- Modify: `tests/test_db.py`

- [ ] **Step 1: Write the failing test**

```python
def test_status_endpoint_remains_fast_with_large_candidate_history():
    # The status endpoint should not block long enough to break the UI.
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src python3 -m unittest tests.test_web tests.test_db -v`
Expected: status serialization is still slow or lock-prone.

- [ ] **Step 3: Write minimal implementation**

```python
conn.execute("PRAGMA journal_mode=WAL")
conn.execute("PRAGMA busy_timeout = 5000")
```

Add a small cached or bounded status path if necessary so the front-end can refresh without waiting on the entire research context every time.

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src python3 -m unittest tests.test_web tests.test_db -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/alpha/web.py src/alpha/db.py tests/test_web.py tests/test_db.py
git commit -m "feat: keep control panel responsive under heavy history"
```

### Self-Review

Coverage check:
- The plan covers the user requirement to keep near-threshold candidates alive for optimization.
- The plan keeps exact duplicates blocked.
- The plan reduces historical false positives caused by preflight-only rows.
- The plan includes UI/database stability for long-running daemon runs.

Placeholder scan:
- No TBD markers.
- No vague test steps without runnable commands.
- No undefined helper names that are not introduced in the task text.

Type consistency:
- `CandidateSpec`, `AlphaStore`, `AlphaWorker`, and existing research-context shapes are used consistently.
