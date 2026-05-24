# Throughput-First Alpha Agent Design

## Context

This project is a personal WorldQuant BRAIN Alpha agent. The production target is not a multi-user SaaS product. The target is a high-output personal research engine that runs unattended on one server and spends AI tokens and BRAIN simulations on the routes most likely to produce approved or submitted alphas.

The existing system already has the core loop:

```text
context_builder -> research_planner -> AI generation/validation
-> preflight -> BRAIN simulation -> submission guard -> queue/submit
```

The current test baseline is strong: `PYTHONPATH=src python3 -m unittest discover -s tests -v` ran 244 tests successfully in about 23.5 seconds on 2026-05-25.

## Goals

- Increase effective alpha output per hour.
- Reduce wasted AI calls and wasted BRAIN simulations.
- Prioritize candidates that are close to submission thresholds.
- Automatically cool down bad scopes, fields, model routes, and formula structures.
- Make the daemon harder to stall in personal long-running operation.
- Keep the existing guard-first submission safety model.
- Keep implementation incremental and testable.

## Non-Goals

- Multi-user authentication and roles.
- A full SaaS deployment model.
- Large UI redesign.
- Rewriting the alpha loop from scratch.
- Loosening submission guard rules to increase apparent throughput.

## Product Definition

The agent should behave like a personal research operator:

1. Spend first on already-paid work, especially pending submission checks.
2. Spend next on near-threshold candidates that have realistic improvement paths.
3. Run setting sweeps before generating unrelated new ideas when an expression is promising.
4. Explore fresh routes only when no higher-value queue needs attention.
5. Stop or cool down routes that repeatedly produce low-quality results.
6. Record enough metrics to prove whether throughput is improving.

## Architecture

Keep the existing worker as the executor. Add a small decision layer before execution.

```text
AlphaStore history
-> metrics snapshot
-> queue scoring
-> scheduler cycle_plan
-> context_builder / research_planner
-> AI candidates
-> preflight
-> BRAIN simulation
-> guard
-> AlphaStore events
-> next metrics snapshot
```

The new layer should be deterministic and testable. AI remains a generator and validator, not the authority for budget allocation.

## New Modules

### `src/alpha/metrics.py`

Read-only efficiency metrics.

Responsibilities:

- Compute generated count, preflight pass rate, simulation success rate, approved rate, pending rate, duplicate skip rate, and low-quality failure rate.
- Aggregate by scope, model/source, field family, structure key, and time window.
- Identify recent throughput trends for daemon decisions.
- Return compact dictionaries that can be stored as events and shown in CLI/Web.

This module must not change candidate state.

### `src/alpha/queues.py`

Shared candidate queue scoring.

Responsibilities:

- Score candidates into `submitable`, `pending`, `optimize`, `watchlist`, `explore_seed`, `trash`, and `abandoned`.
- Use metrics, checks, status, retry count, events, settings, expression structure, and current scope.
- Provide one queue implementation used by scheduler and Web status.
- Keep queue item summaries compact enough for status endpoints.

### `src/alpha/scheduler.py`

Cycle decision engine.

Responsibilities:

- Read metrics snapshot and queues.
- Decide the next `cycle_plan`.
- Decide scope priority and route cooldowns.
- Decide whether the next round is `recover_pending`, `optimize`, `setting_sweep`, `explore`, or `cooldown`.
- Record why a route was chosen.

The scheduler returns a plain dictionary, for example:

```json
{
  "mode": "optimize",
  "scope": {"region": "USA", "universe": "TOP3000", "delay": 0, "neutralization": "SUBINDUSTRY"},
  "target_candidate_id": 123,
  "budget": {"batch_size": 4, "max_rounds": 3},
  "constraints": {"avoid_structures": [], "cooldown_fields": []},
  "reason": "near_threshold_candidate_has_fixable_turnover_gap"
}
```

## Existing Module Changes

### `src/alpha/worker.py`

Keep worker responsible for execution:

- Generate or accept planned candidates.
- Run duplicate checks.
- Run preflight.
- Run BRAIN simulation.
- Run guard.
- Persist transitions and events.

Add optional `cycle_plan` support:

- Existing `run_once()` remains compatible.
- New paths can pass scheduler intent into AI context and priority decisions.
- Worker records `cycle_plan` and outcome metrics after each cycle.

### `src/alpha/context_builder.py`

Keep context generation but consume shared queue summaries from `queues.py` where possible. Avoid duplicated candidate classification logic between context, scheduler, and Web.

### `src/alpha/research_planner.py`

Keep planner focused on research instructions. It should respond to scheduler mode and selected target, but not own global budget decisions.

### `src/alpha/cli.py`

Add personal efficiency controls:

- `alpha status --efficiency`
- `alpha daemon --throughput-mode`
- Optional dry-run scheduler preview command such as `alpha plan-next`

### `src/alpha/web.py`

Expose compact personal operations data:

- Current scheduler mode.
- Next chosen scope and reason.
- Near-threshold queue.
- Cooldown routes.
- Model/source efficiency.
- Scope efficiency.

Do not make a broad UI redesign in the first implementation pass.

## Scheduling Rules

### Pending First

Candidates in `check_pending` have already consumed simulation cost. The scheduler should prioritize refresh and recheck before new generation when pending candidates exist in the active or high-value scopes.

### Near-Threshold Priority

Candidates close to thresholds enter `optimize`.

Signals:

- Sharpe, fitness, returns, turnover, or sub-universe checks close to required values.
- Only one or two fixable checks open.
- No hard blocker such as high self/prod correlation, invalid syntax, data diversity failure that is unlikely to improve locally, or exhausted submission quota.

Optimization should have a bounded number of rounds. If the candidate fails to improve meaningfully after 2-3 rounds, the route is abandoned or cooled down.

### Setting Sweep Before Fresh Exploration

When expression quality is promising and setting sensitivity is plausible, run `setting_sweep` before unrelated fresh exploration. This includes decay, truncation, neutralization, and delay where valid for the scope.

### Bad Route Cooldown

The scheduler should cool down a route when recent evidence shows repeated waste.

Cooldown dimensions:

- Scope.
- Field.
- Field family.
- Dataset.
- Model/source.
- Expression structure key.

Platform errors and rate limits must not be treated as alpha-quality failures.

### Field Budgeting

Use field scout output as a budget input, not only prompt context.

Raise priority for:

- High coverage fields.
- Underused fields.
- Unlit tower opportunities.
- Fields with positive historical evidence.

Lower priority for:

- Recently submitted fields.
- Lit tower fields when fresh exploration has alternatives.
- Metadata-only or auxiliary-only fields as primary signals.
- Fields or datasets with repeated low-quality failures and no positive evidence.

### Model Budgeting

Model roles should be scored by real outcomes, not by static equal trust.

Examples:

- Exploration slots go to sources with higher preflight and simulation pass rates.
- Optimization slots go to sources with better near-threshold conversion.
- Validator failures are counted as useful only when they prevent bad simulations.
- Non-retryable API quota/config errors stop daemon instead of burning cycles.

### Duplicate Cost Compression

Exact duplicates remain permanently blocked.

Structural duplicates are:

- Blocked during ordinary exploration.
- Allowed only in bounded near-threshold repair or setting sweep cases.
- Counted as waste when generated outside those exceptions.

## Failure Handling

### AI Failures

- Quota and config failures stop daemon.
- Temporary generation failures retry within policy.
- Validator rejections are stored as failed candidates when useful for feedback.

### BRAIN Failures

- Honor `Retry-After`.
- Do not count 429/rate-limit as bad alpha evidence.
- Record platform errors separately from expression-quality failures.
- Preserve partial multisimulation successes.

### Daemon Interruptions

- Mark `preflight_passed` candidates as failed with an interruption reason.
- Record daemon stop reason.
- Avoid leaving stale candidates in queues.

### Quality Stop-Loss

If a full batch fails and the best score is below optimization trigger floors, record quality stop-loss and cool down the active route.

## Efficiency Metrics

The first production metric set:

- `generated_per_hour`
- `preflight_pass_rate`
- `simulation_success_rate`
- `approved_rate`
- `approved_per_100_simulations`
- `near_threshold_followup_rate`
- `near_threshold_conversion_rate`
- `duplicate_skip_rate`
- `simulation_waste_rate`
- `route_cooldown_hits`
- `model_source_pass_rate`
- `scope_approved_rate`
- `field_family_approved_rate`
- `daemon_uptime_without_stall`

Metrics should be computed from SQLite history and events so they work across process restarts.

## Testing

Add focused tests before implementation changes.

### New Test Files

- `tests/test_metrics.py`
  - Computes pass rates and waste rates correctly.
  - Aggregates by scope, model/source, and field family.
  - Ignores platform/rate-limit failures when calculating alpha-quality waste.

- `tests/test_queues.py`
  - Classifies submitable, pending, optimize, watchlist, trash, and abandoned candidates.
  - Gives pending and near-threshold candidates higher priority than fresh exploration seeds.
  - Keeps hard-blocked candidates out of optimize.

- `tests/test_scheduler.py`
  - Chooses `recover_pending` when valuable pending candidates exist.
  - Chooses `optimize` for near-threshold candidates.
  - Chooses `setting_sweep` when expression quality is promising and settings are not exhausted.
  - Chooses `explore` when no higher-value queue is available.
  - Applies cooldown after repeated low-quality route failures.
  - Does not punish routes for platform 429 or temporary API failures.

### Existing Test Extensions

- `tests/test_worker.py`
  - Worker accepts optional `cycle_plan`.
  - Existing `run_once()` behavior remains compatible.
  - Worker records cycle outcome metrics.

- `tests/test_cli.py`
  - `status --efficiency` returns useful metrics.
  - Scheduler preview command returns a deterministic plan.

- `tests/test_web.py`
  - Status endpoint includes scheduler mode, efficiency metrics, and cooldown routes.
  - Status remains compact with large history.

## Implementation Order

1. Add read-only `metrics.py` and tests.
2. Add `queues.py` and tests.
3. Add `scheduler.py` in preview-only mode; record `cycle_plan` events but do not alter worker behavior.
4. Wire scheduler into daemon throughput mode.
5. Let worker consume `cycle_plan` for pending, optimize, setting sweep, and explore priority.
6. Add CLI/Web efficiency views.
7. Add personal-server hardening only where it protects throughput:
   - startup config validation,
   - central HTTP timeouts,
   - daemon stall detection,
   - health summary.

## Acceptance Criteria

The implementation is acceptable when:

- The full unittest suite passes.
- Existing default `run_once` and daemon behavior remains compatible unless `--throughput-mode` is enabled.
- Scheduler preview explains its chosen mode and target.
- Efficiency metrics can be read from CLI and Web.
- Near-threshold candidates are followed up before unrelated exploration.
- Repeated bad routes enter cooldown.
- Platform/rate-limit errors are not misclassified as alpha-quality failures.
- Submission guard behavior remains at least as strict as today.

## Risks

- Overfitting scheduler rules to recent history can reduce exploration diversity. Mitigation: reserve a small exploration budget when no high-value queue dominates.
- More queue logic can duplicate existing context-builder classification. Mitigation: move shared classification into `queues.py` and import it from both callers.
- Metrics can become misleading if events are inconsistent. Mitigation: compute primary rates from candidate table status and use events for secondary explanations.
- Throughput mode may expose latent large-history query costs. Mitigation: use bounded store queries and add performance-focused tests.

## Review Notes

This design deliberately avoids a broad refactor. The first implementation pass should prove throughput improvements with metrics and scheduler behavior before reorganizing large modules like `clients.py`, `context_builder.py`, or `research_planner.py`.

