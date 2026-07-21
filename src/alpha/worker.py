from __future__ import annotations

import json
import inspect
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from datetime import date, datetime, timedelta, timezone
from dataclasses import replace
from typing import Any, Callable, Dict, List, Tuple

from .clients import AIClient, BrainClient
from .context_builder import build_ai_research_context
from .db import AlphaStore
from .expression_similarity import expression_signature_metadata, expression_variant_key
from .field_catalog import build_field_catalog
from .guards import SubmissionPolicy, evaluate_submission_readiness, normalize_check_name
from .models import CandidateSpec, DEFAULT_SETTINGS, SimulationFailure, SimulationPending, SimulationPendingError, SimulationResult
from .preflight import validate_expression
from .profile_compliance import profile_compliance_errors


_TERMINAL_WAIT_CHECKS = {
    "selfcorrelation",
    "prodcorrelation",
    "productcorrelation",
    "datadiversity",
    "regularsubmission",
    "d0submission",
    "powerpoolcorrelation",
}
_TERMINAL_WAIT_STATUSES = {"PENDING", "MISSING", "MISSING_STATUS", "QUOTA_FULL"}
_RECHECK_SCOPE_KEYS = ("instrumentType", "region", "universe", "delay")
QUALITY_STOP_MIN_GENERATED = 4
QUALITY_STOP_DEFAULT_SHARPE = 1.0
QUALITY_STOP_DEFAULT_FITNESS = 0.35


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

    def _active_cycle_plan(self, cycle_plan: Dict[str, Any] | None = None) -> Dict[str, Any]:
        if cycle_plan is not None:
            return dict(cycle_plan)
        return dict(self.cycle_plan)

    def _record_cycle_stage(
        self,
        stage: str,
        cycle_plan: Dict[str, Any] | None = None,
        metadata: Dict[str, Any] | None = None,
    ) -> None:
        active_cycle_plan = self._active_cycle_plan(cycle_plan)
        payload: Dict[str, Any] = {
            "stage": stage,
            "cycle_mode": str(active_cycle_plan.get("mode") or ""),
            "cycle_reason": str(active_cycle_plan.get("reason") or ""),
        }
        scope = active_cycle_plan.get("scope")
        if isinstance(scope, dict):
            payload["scope"] = scope
        if metadata:
            payload.update(metadata)
        self.store.record_event(None, "cycle_stage", payload)

    def _record_cycle_outcome(self, cycle_plan: Dict[str, Any] | None, summary: Dict[str, int]) -> None:
        active_cycle_plan = self._active_cycle_plan(cycle_plan)
        self._record_cycle_stage("cycle_finished", cycle_plan, {"summary": dict(summary)})
        if not active_cycle_plan:
            return
        self.store.record_event(
            None,
            "cycle_outcome",
            {"cycle_plan": active_cycle_plan, "summary": dict(summary)},
        )

    def _build_cycle_ai_context(self, cycle_plan: Dict[str, Any] | None = None) -> Dict[str, Any]:
        active_cycle_plan = self._active_cycle_plan(cycle_plan)
        build_context = self._build_ai_context
        parameters = inspect.signature(build_context).parameters
        accepts_cycle_plan = "cycle_plan" in parameters or any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()
        )
        if active_cycle_plan and accepts_cycle_plan:
            return build_context(cycle_plan=cycle_plan)

        ai_context = build_context()
        if not active_cycle_plan:
            return ai_context

        ai_context = dict(ai_context)
        ai_context["cycle_plan"] = active_cycle_plan
        research_context = ai_context.get("research_context")
        if isinstance(research_context, dict):
            research_context["cycle_plan"] = active_cycle_plan
        self.store.record_event(None, "cycle_plan", active_cycle_plan)
        return ai_context

    def _build_ai_context(self, cycle_plan: Dict[str, Any] | None = None) -> Dict[str, Any]:
        active_cycle_plan = self._active_cycle_plan(cycle_plan)
        ai_context = dict(self.context)
        if active_cycle_plan:
            ai_context["cycle_plan"] = active_cycle_plan
        field_catalog = build_field_catalog(self.brain_client, self.context)
        if not field_catalog.get("available"):
            self.store.record_event(None, "datafield_discovery_warning", field_catalog)
        platform_submissions = self._recent_platform_submissions()
        platform_pyramid_alphas = self._platform_pyramid_alphas()
        platform_pyramid_multipliers = self._platform_pyramid_multipliers()
        ai_context["research_context"] = build_ai_research_context(
            self.store,
            self.context,
            field_catalog=field_catalog,
            platform_submissions=platform_submissions,
            platform_pyramid_alphas=platform_pyramid_alphas,
            platform_pyramid_multipliers=platform_pyramid_multipliers,
            cycle_plan=active_cycle_plan or None,
        )
        if active_cycle_plan:
            ai_context["research_context"]["cycle_plan"] = active_cycle_plan
            self.store.record_event(None, "cycle_plan", active_cycle_plan)
        experiment_plan = ai_context["research_context"].get("experiment_plan")
        if experiment_plan:
            self.store.record_event(None, "experiment_plan", experiment_plan)
        return ai_context

    def _recent_platform_submissions(self) -> List[Dict[str, Any]]:
        reader = getattr(self.brain_client, "recent_submitted_alphas", None)
        if not callable(reader):
            return []
        try:
            submissions = reader(self.context, limit=50)
        except Exception as exc:
            self.store.record_event(None, "platform_submission_sync_warning", {"error": str(exc)})
            self.log.warning("platform submitted alpha lookup failed error=%s", exc)
            return []
        return submissions if isinstance(submissions, list) else []

    def _platform_pyramid_alphas(self) -> Dict[str, Any]:
        reader = getattr(self.brain_client, "get_pyramid_alphas", None)
        if not callable(reader):
            return {}
        start_date, end_date = _current_quarter_date_range()
        try:
            payload = reader(start_date=start_date, end_date=end_date)
        except Exception as exc:
            self.store.record_event(None, "platform_pyramid_sync_warning", {"error": str(exc), "kind": "alphas"})
            self.log.warning("platform pyramid alpha lookup failed error=%s", exc)
            return {}
        if isinstance(payload, dict):
            result = dict(payload)
        else:
            result = {"pyramids": payload if isinstance(payload, list) else []}
        result.setdefault("query", {"startDate": start_date, "endDate": end_date})
        return result

    def _platform_pyramid_multipliers(self) -> Dict[str, Any]:
        reader = getattr(self.brain_client, "get_pyramid_multipliers", None)
        if not callable(reader):
            return {}
        try:
            payload = reader()
        except Exception as exc:
            self.store.record_event(None, "platform_pyramid_sync_warning", {"error": str(exc), "kind": "multipliers"})
            self.log.warning("platform pyramid multiplier lookup failed error=%s", exc)
            return {}
        return payload if isinstance(payload, dict) else {"pyramids": payload if isinstance(payload, list) else []}

    def _platform_submitted_today(self) -> int:
        """Real submissions already made in the current trading-day window.

        Seeds submitted_this_round so the per-round submit cap behaves as a true
        daily/account cap across daemon cycles instead of resetting every run_once.
        Only meaningful under AUTO_SUBMIT (dry-run never reaches the platform).
        """
        if not self.policy.auto_submit:
            return 0
        counter = getattr(self.brain_client, "count_submitted_alphas", None)
        if not callable(counter):
            return 0
        try:
            from .submission import trading_day_window

            start, end = trading_day_window()
            return max(0, int(counter(start, end)))
        except Exception:
            # If the platform count is unavailable, fall back to the configured cap so
            # we do NOT submit more this cycle rather than risk exceeding the daily cap.
            self.log.exception("platform submission count failed; treating daily cap as reached")
            return self.policy.max_final_submits_per_round

    def run_once(self, cycle_plan: Dict[str, Any] | None = None) -> Dict[str, int]:
        self.log.info("worker cycle started batch_size=%s", self.batch_size)
        summary = {"generated": 0, "approved": 0, "submitted": 0, "failed": 0, "pending": 0, "skipped": 0}
        self._planned_duplicate_skip_count = 0
        self._planned_probe_exhausted_count = 0
        self._planned_production_rescue_probe_exhausted_count = 0
        submitted_this_round = self._platform_submitted_today()
        candidates = None
        allow_post_dedup_refill = False
        self._record_cycle_stage("building_context", cycle_plan, {"batch_size": self.batch_size})
        ai_context = self._build_cycle_ai_context(cycle_plan=cycle_plan)
        self._record_cycle_stage("context_built", cycle_plan)
        effective_policy = self._policy_for_ai_context(ai_context)
        cycle_candidate_ids: List[int] = []
        if _should_recheck_pending_candidates(cycle_plan):
            submitted_this_round = self._recheck_pending_candidates(
                ai_context,
                effective_policy,
                summary,
                submitted_this_round,
                cycle_plan=cycle_plan,
            )
            if summary["approved"] or summary["submitted"] or summary["failed"] or summary["pending"]:
                self.log.info("worker pending recheck finished summary=%s", summary)
                self._record_cycle_outcome(cycle_plan, summary)
                return summary
        recovered_preflight = self._recover_preflight_passed_candidates(
            ai_context,
            effective_policy,
            summary,
            submitted_this_round,
            cycle_plan=cycle_plan,
        )
        if recovered_preflight:
            self.log.info("worker preflight recovery finished summary=%s", summary)
            self._record_cycle_outcome(cycle_plan, summary)
            return summary
        active_plan = cycle_plan if cycle_plan is not None else self.cycle_plan
        if str((active_plan or {}).get("mode") or "") == "recover_pending":
            self.log.info("worker recover_pending finished summary=%s", summary)
            self._record_cycle_outcome(cycle_plan, summary)
            return summary
        self._record_cycle_stage("selecting_candidates", cycle_plan)
        candidates = self._planned_candidates(ai_context)
        if (
            candidates is not None
            and all(_is_planner_probe_candidate(candidate) for candidate in candidates)
            and _balanced_ai_generation_enabled(self.ai_client)
        ):
            self.store.record_event(
                None,
                "planned_probe_deferred_to_balanced_ai",
                {
                    "planned_probe_count": len(candidates),
                    "requested_batch_size": int(self.batch_size or 0),
                    "policy": (
                        "Balanced multi-generator mode owns the full batch. Keep the probe plan as AI guidance "
                        "instead of replacing the configured batch with a small deterministic probe batch."
                    ),
                },
            )
            candidates = None
        planned_duplicate_skips = int(getattr(self, "_planned_duplicate_skip_count", 0) or 0)
        if planned_duplicate_skips:
            summary["skipped"] += planned_duplicate_skips
        planned_probe_exhausted = int(getattr(self, "_planned_probe_exhausted_count", 0) or 0)
        if planned_probe_exhausted:
            summary["standardized_probe_exhausted"] = planned_probe_exhausted
        planned_production_rescue_probe_exhausted = int(
            getattr(self, "_planned_production_rescue_probe_exhausted_count", 0) or 0
        )
        if planned_production_rescue_probe_exhausted:
            summary["production_rescue_probe_exhausted"] = planned_production_rescue_probe_exhausted
        if candidates is None:
            field_scout_block = _field_scout_fresh_generation_block(ai_context)
            if field_scout_block:
                self.store.record_event(None, "field_scout_generation_blocked", field_scout_block)
                self.log.warning(
                    "AI candidate generation skipped reason=%s top_fields=%s",
                    field_scout_block.get("reason"),
                    field_scout_block.get("top_field_count"),
                )
                summary["skipped"] += max(0, int(self.batch_size or 0))
                summary["field_scout_blocked"] = 1
                self._record_cycle_outcome(cycle_plan, summary)
                return summary
        if candidates is not None:
            candidates = self._validate_planned_candidates(candidates, ai_context)
            ai_fill_size = _planned_probe_ai_fill_size(candidates, ai_context, self.batch_size)
            if ai_fill_size > 0:
                try:
                    ai_candidates = self._generate_ai_candidates_with_timeout(
                        ai_context,
                        cycle_plan,
                        1,
                        batch_size=ai_fill_size,
                    )
                except Exception as exc:
                    summary["ai_probe_fill_failed"] = 1
                    self.store.record_event(
                        None,
                        "planned_probe_ai_fill_error",
                        {"requested": ai_fill_size, "error": str(exc)},
                    )
                    self.log.warning("planned probe AI fill failed requested=%s error=%s", ai_fill_size, exc)
                else:
                    if isinstance(ai_candidates, list):
                        candidates.extend(ai_candidates)
                        allow_post_dedup_refill = bool(ai_candidates)
            self._record_ai_client_diagnostics()
            self._record_validator_rejections(summary)
        else:
            empty_field_pool = _empty_field_pool_block(ai_context)
            if empty_field_pool:
                self.store.record_event(None, "empty_field_pool_generation_blocked", empty_field_pool)
                self.log.warning(
                    "AI candidate generation skipped reason=empty_field_pool detail=%s",
                    empty_field_pool,
                )
                summary["skipped"] += max(0, int(self.batch_size or 0))
                summary["empty_field_pool_blocked"] = 1
                self._record_cycle_outcome(cycle_plan, summary)
                return summary
            for attempt in range(1, self.policy.max_retries + 1):
                try:
                    candidates = self._generate_ai_candidates_with_timeout(ai_context, cycle_plan, attempt)
                    allow_post_dedup_refill = True
                    self._record_ai_client_diagnostics()
                    self._record_validator_rejections(summary)
                    break
                except Exception as exc:
                    non_retryable_reason = _non_retryable_ai_generation_error(exc)
                    if non_retryable_reason:
                        self._record_ai_client_diagnostics()
                        self.log.error(
                            "AI candidate generation blocked reason=%s attempt=%s error=%s",
                            non_retryable_reason,
                            attempt,
                            exc,
                        )
                        self.store.record_event(
                            None,
                            "ai_generation_error",
                            {
                                "error": str(exc),
                                "attempt": attempt,
                                "non_retryable": True,
                                "reason": non_retryable_reason,
                            },
                        )
                        if non_retryable_reason in {
                            "ai_quota_blocked",
                            "ai_config_blocked",
                            "ai_generation_timeout",
                            "ai_network_blocked",
                        }:
                            summary["failed"] += 1
                            summary[non_retryable_reason] = 1
                            self._record_cycle_outcome(cycle_plan, summary)
                            return summary
                        candidates = _deterministic_fallback_candidates(
                            self.batch_size,
                            ai_context,
                            non_retryable_reason,
                            is_duplicate=lambda expression, settings: self.store.find_duplicate_candidate(
                                expression, settings
                            )
                            is not None,
                        )
                        if candidates:
                            self.store.record_event(
                                None,
                                "deterministic_generation_fallback",
                                {
                                    "reason": non_retryable_reason,
                                    "candidate_count": len(candidates),
                                    "policy": (
                                        "AI generation is unavailable, so the worker generated conservative "
                                        "field_scout/datafield templates locally and will still run normal preflight "
                                        "and simulation gates."
                                    ),
                                },
                            )
                            break
                        summary["failed"] += 1
                        summary[non_retryable_reason] = 1
                        self._record_cycle_outcome(cycle_plan, summary)
                        return summary
                    self.log.exception("AI candidate generation failed attempt=%s", attempt)
                    self.store.record_event(None, "ai_generation_error", {"error": str(exc), "attempt": attempt})
        if candidates is None:
            summary["failed"] += 1
            self._record_cycle_outcome(cycle_plan, summary)
            return summary

        ready_for_simulation: List[Tuple[int, CandidateSpec]] = []
        candidate_queue = list(candidates)
        cursor = 0
        refill_rounds = 0
        duplicate_filtered = 0
        while int(summary.get("generated") or 0) < int(self.batch_size or 0):
            if cursor >= len(candidate_queue):
                if (
                    allow_post_dedup_refill
                    and duplicate_filtered > 0
                    and int(summary.get("generated") or 0) > 0
                    and int(self.batch_size or 0) >= 4
                    and refill_rounds < _post_dedup_refill_rounds()
                ):
                    refill_rounds += 1
                    missing = max(0, int(self.batch_size or 0) - int(summary.get("generated") or 0))
                    self._record_cycle_stage(
                        "post_dedup_refill",
                        cycle_plan,
                        {
                            "round": refill_rounds,
                            "missing": missing,
                            "generated": int(summary.get("generated") or 0),
                            "skipped": int(summary.get("skipped") or 0),
                            "policy": "Request replacement candidates after exact or structural duplicate filtering.",
                        },
                    )
                    try:
                        refill_candidates = self._generate_ai_candidates_with_timeout(
                            ai_context,
                            cycle_plan,
                            self.policy.max_retries + refill_rounds,
                            batch_size=missing,
                        )
                    except Exception as exc:
                        self.store.record_event(
                            None,
                            "post_dedup_refill_error",
                            {"round": refill_rounds, "missing": missing, "error": str(exc)},
                        )
                        self.log.warning("post-dedup refill failed round=%s missing=%s error=%s", refill_rounds, missing, exc)
                        break
                    if not refill_candidates:
                        break
                    candidate_queue.extend(refill_candidates)
                    continue
                break
            spec = candidate_queue[cursor]
            cursor += 1
            duplicate = self.store.find_duplicate_candidate(spec.expression, spec.settings)
            if duplicate is not None:
                summary["skipped"] += 1
                duplicate_filtered += 1
                metadata = {
                    "existing_candidate_id": duplicate.get("id"),
                    "expression": spec.expression,
                    "settings": spec.settings,
                    "source": spec.source,
                }
                self.store.record_event(None, "duplicate_candidate_skipped", metadata)
                self.log.info(
                    "candidate duplicate skipped existing_id=%s source=%s",
                    duplicate.get("id"),
                    spec.source,
                )
                continue
            structural_duplicate = self._find_structural_duplicate(spec, ai_context)
            if structural_duplicate is not None:
                summary["skipped"] += 1
                duplicate_filtered += 1
                metadata = {
                    "existing_candidate_id": structural_duplicate.get("id"),
                    "expression": spec.expression,
                    "settings": spec.settings,
                    "source": spec.source,
                    **expression_signature_metadata(spec.expression),
                }
                self.store.record_event(None, "structural_duplicate_candidate_skipped", metadata)
                self.log.info(
                    "candidate structural duplicate skipped existing_id=%s source=%s",
                    structural_duplicate.get("id"),
                    spec.source,
                )
                continue
            candidate_id = self.store.insert_candidate(spec.expression, spec.settings, spec.source)
            cycle_candidate_ids.append(candidate_id)
            self.log.info("candidate generated id=%s source=%s", candidate_id, spec.source)
            summary["generated"] += 1
            generated_metadata = {"expression": spec.expression}
            if spec.metadata:
                generated_metadata["ai_metadata"] = spec.metadata
            self.store.record_event(candidate_id, "generated", generated_metadata)

            allowed_fields = _allowed_fields_from_context(ai_context)
            field_types = _field_types_from_context(ai_context)
            preflight_errors: List[str] = []
            preflight_errors.extend(profile_compliance_errors(spec, ai_context))
            preflight_errors.extend(validate_expression(
                spec.expression,
                allowed_fields=allowed_fields or None,
                field_types=field_types,
                enforce_auxiliary_field_roles=_enforce_auxiliary_field_roles_from_context(ai_context),
                auxiliary_fields=_auxiliary_fields_from_context(ai_context),
                event_fields=_event_fields_from_context(ai_context),
            ))
            if preflight_errors:
                self.store.record_event(candidate_id, "preflight_failed", {"errors": preflight_errors})
                probe_stage = _production_probe_preflight_stage(spec, preflight_errors)
                self._record_probe_validation(candidate_id, probe_stage, summary)
                self.store.transition(candidate_id, "failed", {"reason": "preflight", "errors": preflight_errors})
                summary["failed"] += 1
                continue
            self.store.transition(candidate_id, "preflight_passed")
            ready_for_simulation.append((candidate_id, spec))

        self._record_cycle_stage("simulation_started", cycle_plan, {"ready_count": len(ready_for_simulation)})
        for candidate_id, spec, result in self._simulate_candidates(ready_for_simulation, cycle_plan=cycle_plan):
            if isinstance(result, SimulationPending):
                self.store.record_event(
                    candidate_id,
                    "simulation_pending",
                    {"location": result.location, "error": result.error, "raw": result.raw},
                )
                self.store.transition(
                    candidate_id,
                    "check_pending",
                    {
                        "reason": "simulation_pending",
                        "errors": ["SIMULATION_PENDING"],
                        "simulation_location": result.location,
                    },
                )
                summary["pending"] += 1
                continue
            if isinstance(result, SimulationFailure):
                self.store.record_event(candidate_id, "simulation_error", {"error": result.error, "raw": result.raw})
                probe_stage = _production_probe_simulation_error_stage(spec, result.error)
                self._record_probe_validation(candidate_id, probe_stage, summary)
                self.store.transition(candidate_id, "failed", {"reason": "simulation_error", "error": result.error})
                summary["failed"] += 1
                continue
            if result is None:
                probe_stage = _production_probe_simulation_error_stage(spec, "retry_limit")
                self._record_probe_validation(candidate_id, probe_stage, summary)
                self.store.transition(candidate_id, "failed", {"reason": "retry_limit"})
                summary["failed"] += 1
                continue

            self.store.update_candidate(
                candidate_id,
                alpha_id=result.alpha_id,
                metrics_json=json.dumps(result.metrics, sort_keys=True),
                checks_json=json.dumps(result.checks, sort_keys=True),
            )
            self.store.transition(candidate_id, "simulated", {"alpha_id": result.alpha_id})
            self.store.transition(candidate_id, "metric_passed", {"alpha_id": result.alpha_id})
            probe_stage = _production_probe_result_stage(spec, result.metrics, result.checks, ai_context)
            self._record_probe_validation(candidate_id, probe_stage, summary)

            submitted_this_round = self._handle_submission_guard(
                candidate_id,
                result.alpha_id,
                result.metrics,
                result.checks,
                effective_policy,
                ai_context,
                submitted_this_round,
                summary,
            )

        quality_stop = _quality_stop_loss_summary(self.store, cycle_candidate_ids, self.batch_size, ai_context)
        if quality_stop:
            summary.update(quality_stop)
            self.store.record_event(None, "quality_stop_loss", quality_stop)

        self.log.info("worker cycle finished summary=%s", summary)
        self._record_cycle_outcome(cycle_plan, summary)
        return summary

    def _generate_ai_candidates_with_timeout(
        self,
        ai_context: Dict[str, Any],
        cycle_plan: Dict[str, Any] | None,
        attempt: int,
        *,
        batch_size: int | None = None,
    ) -> List[CandidateSpec]:
        requested_batch_size = max(1, int(batch_size or self.batch_size or 1))
        timeout = _ai_generation_stage_timeout_seconds()
        self._record_cycle_stage(
            "ai_generation_started",
            cycle_plan,
            {
                "attempt": attempt,
                "batch_size": requested_batch_size,
                "timeout_seconds": timeout,
                "client": type(self.ai_client).__name__,
            },
        )
        started = time.monotonic()
        executor = ThreadPoolExecutor(max_workers=1)
        future = executor.submit(self.ai_client.generate_candidates, requested_batch_size, ai_context)
        try:
            candidates = future.result(timeout=timeout)
        except FutureTimeoutError as exc:
            future.cancel()
            elapsed = time.monotonic() - started
            partial_candidates = getattr(self.ai_client, "last_partial_candidates", None)
            if isinstance(partial_candidates, list) and partial_candidates:
                self._record_cycle_stage(
                    "ai_generation_partial_timeout",
                    cycle_plan,
                    {
                        "attempt": attempt,
                        "batch_size": requested_batch_size,
                        "candidate_count": len(partial_candidates),
                        "timeout_seconds": timeout,
                        "elapsed_seconds": round(elapsed, 3),
                        "client": type(self.ai_client).__name__,
                        "policy": "Use locally accepted candidates already returned by the AI client before a refill timeout.",
                    },
                )
                return partial_candidates[:requested_batch_size]
            self._record_cycle_stage(
                "ai_generation_timeout",
                cycle_plan,
                {
                    "attempt": attempt,
                    "batch_size": requested_batch_size,
                    "timeout_seconds": timeout,
                    "elapsed_seconds": round(elapsed, 3),
                    "client": type(self.ai_client).__name__,
                },
            )
            raise TimeoutError(f"AI candidate generation stage timed out after {timeout:.2f}s") from exc
        finally:
            executor.shutdown(wait=False, cancel_futures=True)
        elapsed = time.monotonic() - started
        self._record_cycle_stage(
            "ai_generation_finished",
            cycle_plan,
            {
                "attempt": attempt,
                "batch_size": requested_batch_size,
                "candidate_count": len(candidates) if isinstance(candidates, list) else 0,
                "elapsed_seconds": round(elapsed, 3),
                "client": type(self.ai_client).__name__,
            },
        )
        return candidates

    def _record_probe_validation(
        self,
        candidate_id: int,
        probe_stage: Dict[str, Any] | None,
        summary: Dict[str, int],
    ) -> None:
        if not probe_stage:
            return
        self.store.record_event(candidate_id, "probe_validation", probe_stage)
        stage = str(probe_stage.get("stage") or "").strip()
        if not stage:
            return
        key = f"probe_{stage}"
        summary[key] = int(summary.get(key, 0)) + 1

    def _find_structural_duplicate(
        self,
        spec: CandidateSpec,
        ai_context: Dict[str, Any],
    ) -> Dict[str, Any] | None:
        if str(spec.source or "") == "planner_setting_sweep":
            return None
        if not _source_uses_structural_dedup(spec.source):
            return None
        research_context = ai_context.get("research_context") if isinstance(ai_context.get("research_context"), dict) else {}
        experiment_plan = research_context.get("experiment_plan") if isinstance(research_context, dict) else {}
        if isinstance(experiment_plan, dict) and str(experiment_plan.get("mode") or "") == "setting_sweep":
            return None
        if isinstance(experiment_plan, dict) and str(experiment_plan.get("mode") or "") == "optimize_best":
            return None
        generation_policy = research_context.get("generation_policy") if isinstance(research_context, dict) else {}
        if not isinstance(generation_policy, dict) or not generation_policy.get("avoid_historical_structural_duplicates"):
            return None

        target_structure = expression_variant_key(spec.expression)
        for candidate in reversed(self.store.list_candidates()):
            if not _candidate_has_simulated_history(candidate):
                continue
            if not _candidate_structural_scope_matches(candidate, spec.settings):
                continue
            if expression_variant_key(str(candidate.get("expression") or "")) == target_structure:
                return candidate
        return None

    def _recheck_pending_candidates(
        self,
        ai_context: Dict[str, Any],
        effective_policy: SubmissionPolicy,
        summary: Dict[str, int],
        submitted_this_round: int,
        cycle_plan: Dict[str, Any] | None = None,
    ) -> int:
        pending = [
            row
            for row in self.store.list_candidates(status="check_pending")
            if _candidate_matches_context(row, self.context)
        ]
        if not pending:
            return submitted_this_round
        pending = _order_pending_for_cycle_plan(pending, cycle_plan)
        limit = max(0, min(int(self.batch_size or 0), 4))
        for row in pending[:limit]:
            candidate_id = int(row["id"])
            if not _candidate_needs_pending_recheck(self.store, candidate_id):
                continue
            stale_pending = _stale_simulation_pending_metadata(self.store, candidate_id)
            if stale_pending:
                self.store.record_event(candidate_id, "pending_recheck_stale_failed", stale_pending)
                self.store.transition(
                    candidate_id,
                    "failed",
                    {
                        "reason": "stale_simulation_pending",
                        "errors": ["STALE_SIMULATION_PENDING"],
                        **stale_pending,
                    },
                )
                summary["failed"] += 1
                summary["stale_pending_failed"] = int(summary.get("stale_pending_failed", 0)) + 1
                continue
            alpha_id = str(row.get("alpha_id") or "").strip()
            metrics = _loads_dict(row.get("metrics_json"))
            checks = _loads_dict(row.get("checks_json"))
            if not alpha_id:
                simulation_location = _latest_simulation_pending_location(self.store, candidate_id)
                if simulation_location and hasattr(self.brain_client, "resume_simulation"):
                    submitted_this_round = self._resume_pending_simulation(
                        candidate_id,
                        simulation_location,
                        effective_policy,
                        ai_context,
                        submitted_this_round,
                        summary,
                    )
                    continue
            guard = evaluate_submission_readiness(metrics, checks, effective_policy, submitted_this_round)
            if guard.ready:
                self.store.record_event(candidate_id, "pending_recheck", {"source": "stored_checks", "ready": True})
                submitted_this_round = self._handle_submission_guard(
                    candidate_id,
                    alpha_id,
                    metrics,
                    checks,
                    effective_policy,
                    ai_context,
                    submitted_this_round,
                    summary,
                    recheck=True,
                )
                continue
            if not _only_terminal_wait_errors(guard.errors):
                self.store.record_event(
                    candidate_id,
                    "pending_recheck_ineligible",
                    {"errors": guard.errors, "warnings": guard.warnings},
                )
                self.store.transition(
                    candidate_id,
                    "failed",
                    {"reason": "core_check_failed_before_terminal_recheck", "errors": guard.errors},
                )
                summary["failed"] += 1
                continue
            if not alpha_id or not hasattr(self.brain_client, "get_submission_check"):
                self.store.record_event(
                    candidate_id,
                    "pending_recheck_waiting",
                    {"errors": guard.errors, "reason": "terminal_checks_not_available"},
                )
                summary["pending"] += 1
                continue
            try:
                latest_checks = self.brain_client.get_submission_check(alpha_id)  # type: ignore[attr-defined]
            except Exception as exc:
                self.log.warning("pending check refresh failed candidate_id=%s alpha_id=%s error=%s", candidate_id, alpha_id, exc)
                self.store.record_event(candidate_id, "pending_recheck_error", {"alpha_id": alpha_id, "error": str(exc)})
                summary["pending"] += 1
                continue
            if not isinstance(latest_checks, dict) or not latest_checks:
                self.store.record_event(candidate_id, "pending_recheck_empty", {"alpha_id": alpha_id})
                summary["pending"] += 1
                continue
            merged_checks = dict(checks)
            merged_checks.update(latest_checks)
            self.store.update_candidate(candidate_id, checks_json=json.dumps(merged_checks, sort_keys=True))
            self.store.record_event(
                candidate_id,
                "pending_recheck",
                {"alpha_id": alpha_id, "refreshed_checks": sorted(latest_checks)},
            )
            submitted_this_round = self._handle_submission_guard(
                candidate_id,
                alpha_id,
                metrics,
                merged_checks,
                effective_policy,
                ai_context,
                submitted_this_round,
                summary,
                recheck=True,
            )
        return submitted_this_round

    def _recover_preflight_passed_candidates(
        self,
        ai_context: Dict[str, Any],
        effective_policy: SubmissionPolicy,
        summary: Dict[str, int],
        submitted_this_round: int,
        cycle_plan: Dict[str, Any] | None = None,
    ) -> int:
        rows = [
            row
            for row in self.store.list_candidates(status="preflight_passed")
            if _candidate_matches_context(row, self.context)
            and _candidate_has_status_transition(self.store, int(row.get("id") or 0), "preflight_passed")
        ]
        rows = sorted(rows, key=lambda row: int(row.get("id") or 0), reverse=True)
        if not rows:
            return 0
        limit = max(1, min(int(self.batch_size or 1), 4))
        recovered = 0
        ready_for_simulation: List[Tuple[int, CandidateSpec]] = []
        for row in rows[:limit]:
            candidate_id = int(row["id"])
            settings = dict(DEFAULT_SETTINGS)
            settings.update(_loads_dict(row.get("settings_json")))
            expression = str(row.get("expression") or "").strip()
            if not expression:
                continue
            self.store.record_event(
                candidate_id,
                "preflight_recovery",
                {"reason": "resume_preflight_passed_candidate_before_fresh_generation"},
            )
            ready_for_simulation.append(
                (
                    candidate_id,
                    CandidateSpec(
                        expression=expression,
                        settings=settings,
                        source=str(row.get("source") or "preflight_recovery"),
                    ),
                )
            )
        for candidate_id, spec, result in self._simulate_candidates(ready_for_simulation, cycle_plan=cycle_plan):
            recovered += 1
            if isinstance(result, SimulationPending):
                self.store.record_event(
                    candidate_id,
                    "simulation_pending",
                    {"location": result.location, "error": result.error, "raw": result.raw, "recovered": True},
                )
                self.store.transition(
                    candidate_id,
                    "check_pending",
                    {
                        "reason": "simulation_pending",
                        "errors": ["SIMULATION_PENDING"],
                        "simulation_location": result.location,
                    },
                )
                summary["pending"] += 1
                continue
            if isinstance(result, SimulationFailure):
                self.store.record_event(
                    candidate_id,
                    "simulation_error",
                    {"error": result.error, "raw": result.raw, "recovered": True},
                )
                self.store.transition(candidate_id, "failed", {"reason": "simulation_error", "error": result.error})
                summary["failed"] += 1
                continue
            if result is None:
                self.store.transition(candidate_id, "failed", {"reason": "retry_limit"})
                summary["failed"] += 1
                continue

            self.store.update_candidate(
                candidate_id,
                alpha_id=result.alpha_id,
                metrics_json=json.dumps(result.metrics, sort_keys=True),
                checks_json=json.dumps(result.checks, sort_keys=True),
            )
            self.store.transition(candidate_id, "simulated", {"alpha_id": result.alpha_id, "recovered": True})
            self.store.transition(candidate_id, "metric_passed", {"alpha_id": result.alpha_id, "recovered": True})
            submitted_this_round = self._handle_submission_guard(
                candidate_id,
                result.alpha_id,
                result.metrics,
                result.checks,
                effective_policy,
                ai_context,
                submitted_this_round,
                summary,
                recheck=True,
            )
        return recovered

    def _resume_pending_simulation(
        self,
        candidate_id: int,
        simulation_location: str,
        effective_policy: SubmissionPolicy,
        ai_context: Dict[str, Any],
        submitted_this_round: int,
        summary: Dict[str, int],
    ) -> int:
        try:
            result = self.brain_client.resume_simulation(simulation_location)  # type: ignore[attr-defined]
        except SimulationPendingError as exc:
            self.store.record_event(
                candidate_id,
                "simulation_pending",
                {"location": exc.location, "error": str(exc), "recheck": True},
            )
            summary["pending"] += 1
            return submitted_this_round
        except Exception as exc:
            self.log.warning(
                "pending simulation resume failed candidate_id=%s location=%s error=%s",
                candidate_id,
                simulation_location,
                exc,
            )
            self.store.record_event(
                candidate_id,
                "pending_simulation_resume_error",
                {"location": simulation_location, "error": str(exc)},
            )
            if _terminal_simulation_resume_error(exc):
                self.store.transition(
                    candidate_id,
                    "failed",
                    {
                        "reason": "simulation_resume_failed",
                        "error": str(exc),
                        "simulation_location": simulation_location,
                    },
                )
                summary["failed"] += 1
            else:
                summary["pending"] += 1
            return submitted_this_round
        self.store.update_candidate(
            candidate_id,
            alpha_id=result.alpha_id,
            metrics_json=json.dumps(result.metrics, sort_keys=True),
            checks_json=json.dumps(result.checks, sort_keys=True),
        )
        self.store.transition(
            candidate_id,
            "simulated",
            {"alpha_id": result.alpha_id, "resumed_location": simulation_location},
        )
        self.store.transition(
            candidate_id,
            "metric_passed",
            {"alpha_id": result.alpha_id, "resumed_location": simulation_location},
        )
        return self._handle_submission_guard(
            candidate_id,
            result.alpha_id,
            result.metrics,
            result.checks,
            effective_policy,
            ai_context,
            submitted_this_round,
            summary,
            recheck=True,
        )

    def _handle_submission_guard(
        self,
        candidate_id: int,
        alpha_id: str,
        metrics: Dict[str, Any],
        checks: Dict[str, Any],
        effective_policy: SubmissionPolicy,
        ai_context: Dict[str, Any],
        submitted_this_round: int,
        summary: Dict[str, int],
        recheck: bool = False,
    ) -> int:
        guard = evaluate_submission_readiness(metrics, checks, effective_policy, submitted_this_round)
        guard_metadata = {"ready": guard.ready, "errors": guard.errors, "warnings": guard.warnings}
        if recheck:
            guard_metadata["recheck"] = True
        quality_thresholds = _quality_thresholds_from_context(ai_context)
        if quality_thresholds:
            guard_metadata["quality_thresholds"] = quality_thresholds
        self.store.record_event(
            candidate_id,
            "submission_guard",
            guard_metadata,
        )

        if not guard.ready:
            self.log.info("candidate blocked id=%s errors=%s", candidate_id, guard.errors)
            if _only_terminal_wait_errors(guard.errors):
                self.store.transition(candidate_id, "check_pending", {"errors": guard.errors})
                summary["pending"] += 1
            else:
                self.store.transition(candidate_id, "failed", {"errors": guard.errors})
                summary["failed"] += 1
            return submitted_this_round

        self.store.transition(candidate_id, "approved")
        self.log.info("candidate approved id=%s alpha_id=%s", candidate_id, alpha_id)
        summary["approved"] += 1
        if not self.policy.auto_submit:
            self._mark_platform_alpha_ready(candidate_id, alpha_id, summary)

        try:
            submit = self.brain_client.submit_alpha(alpha_id, dry_run=not self.policy.auto_submit)
        except Exception as exc:
            # Leave the candidate as "approved" so submit-approved can retry it later;
            # never let a transient submit error crash the daemon loop.
            self.store.record_event(
                candidate_id,
                "submit_error",
                {"alpha_id": alpha_id, "error": str(exc)},
            )
            self.log.warning("candidate submit raised id=%s alpha_id=%s error=%s", candidate_id, alpha_id, exc)
            summary["submit_error"] = summary.get("submit_error", 0) + 1
            return submitted_this_round
        if submit.submitted and submit.stage == "OS":
            self.store.transition(candidate_id, "submitted", {"alpha_id": alpha_id})
            self.log.info("candidate submitted id=%s alpha_id=%s", candidate_id, alpha_id)
            submitted_this_round += 1
            summary["submitted"] += 1
        else:
            is_dry_run = submit.stage == "DRY_RUN"
            event_type = "dry_run_submit" if is_dry_run else "submit_unverified"
            self.store.record_event(
                candidate_id,
                event_type,
                {
                    "alpha_id": alpha_id,
                    "stage": submit.stage,
                    "message": submit.message,
                    "auto_submit": self.policy.auto_submit,
                },
            )
            summary_key = "dry_run" if is_dry_run else "submit_unverified"
            summary[summary_key] = summary.get(summary_key, 0) + 1
            self.log.info("candidate %s id=%s alpha_id=%s stage=%s", event_type, candidate_id, alpha_id, submit.stage)
        return submitted_this_round

    def _mark_platform_alpha_ready(self, candidate_id: int, alpha_id: str, summary: Dict[str, int]) -> None:
        updater = getattr(self.brain_client, "set_alpha_properties", None)
        if not callable(updater):
            return
        try:
            updater(alpha_id, color="GREEN")
        except Exception as exc:
            self.store.record_event(
                candidate_id,
                "platform_color_error",
                {"alpha_id": alpha_id, "color": "GREEN", "error": str(exc)},
            )
            summary["platform_color_error"] = summary.get("platform_color_error", 0) + 1
            self.log.warning("candidate platform color failed id=%s alpha_id=%s error=%s", candidate_id, alpha_id, exc)
            return
        self.store.record_event(candidate_id, "platform_color_set", {"alpha_id": alpha_id, "color": "GREEN"})
        summary["platform_color_set"] = summary.get("platform_color_set", 0) + 1
        self.log.info("candidate platform color set id=%s alpha_id=%s color=GREEN", candidate_id, alpha_id)

    def _policy_for_ai_context(self, ai_context: Dict[str, Any]) -> SubmissionPolicy:
        thresholds = _quality_thresholds_from_context(ai_context)
        if not thresholds or not thresholds.get("trusted"):
            return self.policy

        updates: Dict[str, float] = {}
        threshold_map = {
            "required_sharpe": "min_sharpe",
            "required_fitness": "min_fitness",
            "required_returns": "min_returns",
            "turnover_min": "min_turnover",
            "turnover_max": "max_turnover",
        }
        # Trusted thresholds may only tighten the configured policy, never relax it
        # below the configured floor. min_* keys are lower bounds (clamp upward with
        # max); max_turnover is an upper bound (clamp downward with min).
        for threshold_key, policy_key in threshold_map.items():
            value = _positive_float(thresholds.get(threshold_key))
            if value is None:
                continue
            current = getattr(self.policy, policy_key)
            if policy_key == "max_turnover":
                clamped = min(current, value)
            else:
                clamped = max(current, value)
            if clamped != current:
                updates[policy_key] = clamped
        if not updates:
            return self.policy
        return replace(self.policy, **updates)

    def _record_ai_client_diagnostics(self) -> None:
        plan = getattr(self.ai_client, "last_plan", None)
        if isinstance(plan, dict) and plan:
            self.store.record_event(None, "model_allocation", plan)
            self.log.info("model allocation=%s", plan)
            repair = plan.get("intra_round_repair")
            if isinstance(repair, dict) and repair:
                self.store.record_event(None, "intra_round_repair", repair)
                self.log.info("intra_round_repair=%s", repair)
        errors = getattr(self.ai_client, "last_errors", None)
        if isinstance(errors, list):
            for error in errors:
                if not isinstance(error, dict):
                    continue
                self.store.record_event(None, "model_generation_error", error)
                self.log.warning("model generation error=%s", error)

    def _record_validator_rejections(self, summary: Dict[str, int]) -> None:
        rejections = getattr(self.ai_client, "last_validator_rejections", None)
        if not isinstance(rejections, list):
            return
        for rejection in rejections:
            if not isinstance(rejection, dict):
                continue
            candidate = rejection.get("candidate")
            if not isinstance(candidate, dict):
                continue
            expression = str(candidate.get("expression") or "").strip()
            if not expression:
                continue
            settings = dict(DEFAULT_SETTINGS)
            settings.update(self.context)
            candidate_settings = candidate.get("settings")
            if isinstance(candidate_settings, dict):
                settings.update(candidate_settings)
            source = str(candidate.get("source") or "model:validator_rejected")
            metadata = candidate.get("metadata") if isinstance(candidate.get("metadata"), dict) else {}
            duplicate = self.store.find_duplicate_candidate(expression, settings)
            if duplicate is not None:
                self.store.record_event(
                    None,
                    "validator_rejected_duplicate_skipped",
                    {
                        "existing_candidate_id": duplicate.get("id"),
                        "expression": expression,
                        "source": source,
                        "reason": rejection.get("reason"),
                    },
                )
                continue
            candidate_id = self.store.insert_candidate(expression, settings, source)
            summary["generated"] += 1
            summary["failed"] += 1
            self.store.record_event(candidate_id, "generated", {"expression": expression, "ai_metadata": metadata})
            self.store.record_event(candidate_id, "validator_rejected", rejection)
            self.store.transition(
                candidate_id,
                "failed",
                {
                    "reason": "validator_rejected",
                    "errors": [str(rejection.get("reason") or "VALIDATOR_FILTERED")],
                    "validator_profile": rejection.get("validator_profile"),
                    "validator_model": rejection.get("validator_model"),
                },
            )

    def _planned_candidates(self, ai_context: Dict[str, Any]) -> List[CandidateSpec] | None:
        research_context = ai_context.get("research_context")
        if not isinstance(research_context, dict):
            return None
        plan = research_context.get("experiment_plan")
        if not isinstance(plan, dict):
            return None
        mode = str(plan.get("mode") or "")
        if mode != "setting_sweep":
            standardized_candidates = self._standardized_explore_probe_candidates(plan)
            if standardized_candidates is not None:
                return standardized_candidates
            return self._production_rescue_probe_candidates(plan)
        expression = str(plan.get("target_expression") or "").strip()
        variants = plan.get("setting_variants")
        if not expression or not isinstance(variants, list):
            return None
        validation_errors = _setting_sweep_target_validation_errors(expression, research_context)
        if validation_errors:
            self.store.record_event(
                None,
                "setting_sweep_target_invalid",
                {
                    "target_candidate_id": plan.get("target_candidate_id"),
                    "optimization_anchor_id": plan.get("optimization_anchor_id"),
                    "errors": validation_errors,
                    "expression": expression,
                },
            )
            plan["mode"] = "explore_new_family"
            plan["target_candidate_id"] = None
            plan["abandoned_target_id"] = plan.get("optimization_anchor_id")
            plan["abandon_reason"] = "SETTING_SWEEP_TARGET_INVALID"
            plan["target_expression"] = ""
            plan["objective"] = (
                "The planned setting sweep target contains fields or syntax that are not valid in the current "
                "datafield catalog. Explore fresh valid expressions instead."
            )
            plan["keep"] = []
            plan["change"] = ["switch field family", "use exact current datafield ids", "generate fresh formula structures"]
            plan["avoid_expressions"] = [expression]
            plan.pop("setting_variants", None)
            self.store.record_event(None, "experiment_plan", plan)
            return None
        candidates: List[CandidateSpec] = []
        skipped_duplicates = 0
        for variant in variants[: self.batch_size]:
            if not isinstance(variant, dict):
                continue
            settings = dict(DEFAULT_SETTINGS)
            settings.update(self.context)
            settings.update(variant)
            if self.store.find_duplicate_candidate(expression, settings) is not None:
                skipped_duplicates += 1
                continue
            candidates.append(
                CandidateSpec(
                    expression=expression,
                    settings=settings,
                    source="planner_setting_sweep",
                    metadata={
                        "experiment_plan_mode": "setting_sweep",
                        "target_candidate_id": plan.get("target_candidate_id"),
                        "optimization_anchor_id": plan.get("optimization_anchor_id"),
                        "optimize_round": plan.get("optimize_round"),
                    },
                )
            )
        if skipped_duplicates and not candidates:
            self.store.record_event(
                None,
                "setting_sweep_exhausted",
                {
                    "target_candidate_id": plan.get("target_candidate_id"),
                    "optimization_anchor_id": plan.get("optimization_anchor_id"),
                    "skipped_duplicates": skipped_duplicates,
                },
            )
            plan["mode"] = "explore_new_family"
            plan["target_candidate_id"] = None
            plan["abandoned_target_id"] = plan.get("optimization_anchor_id")
            plan["abandon_reason"] = "SETTING_SWEEP_VARIANTS_EXHAUSTED"
            plan["target_expression"] = ""
            plan["objective"] = (
                "The planned setting sweep variants were already tested. "
                "Explore structurally different expressions instead of repeating the exhausted target."
            )
            plan["keep"] = []
            plan["change"] = [
                "switch field family",
                "avoid local variants of the exhausted expression",
                "generate fresh formula structures",
            ]
            plan["avoid_expressions"] = [expression]
            plan.pop("setting_variants", None)
            self.store.record_event(None, "experiment_plan", plan)
        return candidates or None

    def _production_rescue_probe_candidates(self, plan: Dict[str, Any]) -> List[CandidateSpec] | None:
        mode = str(plan.get("mode") or "")
        if not mode.startswith("explore"):
            return None
        production_rescue = plan.get("production_rescue")
        if not isinstance(production_rescue, dict) or not production_rescue.get("active"):
            return None
        quality_budget = plan.get("quality_budget") if isinstance(plan.get("quality_budget"), dict) else {}
        slots = quality_budget.get("slots") if isinstance(quality_budget, dict) else {}
        try:
            probe_slots = max(0, int(slots.get("probe_new_fields") or 0)) if isinstance(slots, dict) else 0
        except (TypeError, ValueError):
            probe_slots = 0
        if probe_slots <= 0:
            return None
        recommendations = plan.get("probe_recommendations")
        if not isinstance(recommendations, list) or not recommendations:
            return None

        settings = dict(DEFAULT_SETTINGS)
        target_settings = plan.get("target_settings")
        if isinstance(target_settings, dict):
            settings.update(target_settings)
        settings.update(self.context)

        candidates: List[CandidateSpec] = []
        limit = min(max(0, int(self.batch_size or 0)), probe_slots)
        skipped_duplicates = 0
        for recommendation in recommendations:
            if not isinstance(recommendation, dict):
                continue
            templates = recommendation.get("templates")
            if not isinstance(templates, list):
                continue
            for template_index, template in enumerate(templates):
                expression = str(template or "").strip()
                if not expression:
                    continue
                candidate_settings = dict(settings)
                duplicate = self.store.find_duplicate_candidate(expression, candidate_settings)
                if duplicate is not None:
                    skipped_duplicates += 1
                    self._planned_duplicate_skip_count = int(getattr(self, "_planned_duplicate_skip_count", 0) or 0) + 1
                    self.store.record_event(
                        None,
                        "planned_probe_duplicate_skipped",
                        {
                            "existing_candidate_id": duplicate.get("id"),
                            "expression": expression,
                            "settings": candidate_settings,
                            "source": "planner_unverified_probe",
                            "probe_field": recommendation.get("field"),
                            "probe_template_index": template_index,
                        },
                    )
                    continue
                candidates.append(
                    CandidateSpec(
                        expression=expression,
                        settings=candidate_settings,
                        source="planner_unverified_probe",
                        metadata={
                            "experiment_plan_mode": mode,
                            "validation_stage": "unverified_probe",
                            "probe_field": recommendation.get("field"),
                            "probe_dataset_id": recommendation.get("dataset_id"),
                            "probe_category": recommendation.get("category"),
                            "probe_route": recommendation.get("route"),
                            "probe_template_index": template_index,
                            "production_rescue_reason": production_rescue.get("reason"),
                        },
                    )
                )
                if len(candidates) >= limit:
                    return candidates
        if skipped_duplicates and not candidates:
            self._planned_production_rescue_probe_exhausted_count = (
                int(getattr(self, "_planned_production_rescue_probe_exhausted_count", 0) or 0) + 1
            )
            self.store.record_event(
                None,
                "production_rescue_probe_exhausted",
                {
                    "reason": "all_production_rescue_probe_templates_duplicate",
                    "skipped_duplicates": skipped_duplicates,
                    "probe_recommendation_count": len(recommendations),
                    "probe_fields": _probe_recommendation_fields(recommendations),
                    "probe_slots": probe_slots,
                    "settings": settings,
                },
            )
            return []
        if not candidates:
            self.store.record_event(
                None,
                "production_rescue_probe_unavailable",
                {
                    "reason": "NO_PROBE_TEMPLATES",
                    "probe_recommendation_count": len(recommendations),
                    "probe_slots": probe_slots,
                },
            )
            return []
        return candidates

    def _standardized_explore_probe_candidates(self, plan: Dict[str, Any]) -> List[CandidateSpec] | None:
        mode = str(plan.get("mode") or "")
        if mode != "explore_new_family":
            return None
        production_rescue = plan.get("production_rescue")
        if isinstance(production_rescue, dict) and production_rescue.get("active"):
            return None
        quality_budget = plan.get("quality_budget") if isinstance(plan.get("quality_budget"), dict) else {}
        slots = quality_budget.get("slots") if isinstance(quality_budget, dict) else {}
        exploit_fields = quality_budget.get("exploit_fields") if isinstance(quality_budget, dict) else []
        if exploit_fields:
            return None
        try:
            probe_slots = max(0, int(slots.get("probe_new_fields") or 0)) if isinstance(slots, dict) else 0
        except (TypeError, ValueError):
            probe_slots = 0
        if probe_slots <= 0:
            return None
        recommendations = plan.get("probe_recommendations")
        if not isinstance(recommendations, list) or not recommendations:
            return None

        settings = dict(DEFAULT_SETTINGS)
        target_settings = plan.get("target_settings")
        if isinstance(target_settings, dict):
            settings.update(target_settings)
        settings.update(self.context)

        candidates: List[CandidateSpec] = []
        limit = min(max(0, int(self.batch_size or 0)), probe_slots)
        skipped_duplicates = 0
        for recommendation in recommendations:
            if not isinstance(recommendation, dict):
                continue
            templates = recommendation.get("templates")
            if not isinstance(templates, list):
                continue
            for template_index, template in enumerate(templates):
                expression = str(template or "").strip()
                if not expression:
                    continue
                candidate_settings = dict(settings)
                duplicate = self.store.find_duplicate_candidate(expression, candidate_settings)
                if duplicate is not None:
                    skipped_duplicates += 1
                    self._planned_duplicate_skip_count = int(getattr(self, "_planned_duplicate_skip_count", 0) or 0) + 1
                    self.store.record_event(
                        None,
                        "planned_probe_duplicate_skipped",
                        {
                            "existing_candidate_id": duplicate.get("id"),
                            "expression": expression,
                            "settings": candidate_settings,
                            "source": "planner_standardized_probe",
                            "probe_field": recommendation.get("field"),
                            "probe_template_index": template_index,
                        },
                    )
                    continue
                candidates.append(
                    CandidateSpec(
                        expression=expression,
                        settings=candidate_settings,
                        source="planner_standardized_probe",
                        metadata={
                            "experiment_plan_mode": mode,
                            "validation_stage": "standardized_probe",
                            "probe_field": recommendation.get("field"),
                            "probe_dataset_id": recommendation.get("dataset_id"),
                            "probe_category": recommendation.get("category"),
                            "probe_route": recommendation.get("route"),
                            "probe_template_index": template_index,
                        },
                    )
                )
                if len(candidates) >= limit:
                    return candidates
        if skipped_duplicates and not candidates:
            self._planned_probe_exhausted_count = int(getattr(self, "_planned_probe_exhausted_count", 0) or 0) + 1
            self.store.record_event(
                None,
                "standardized_probe_exhausted",
                {
                    "reason": "all_standardized_probe_templates_duplicate",
                    "skipped_duplicates": skipped_duplicates,
                    "probe_recommendation_count": len(recommendations),
                    "probe_fields": _probe_recommendation_fields(recommendations),
                    "probe_slots": probe_slots,
                    "settings": settings,
                },
            )
            return []
        return candidates or None

    def _validate_planned_candidates(
        self,
        candidates: List[CandidateSpec],
        ai_context: Dict[str, Any],
    ) -> List[CandidateSpec]:
        if candidates and all(_is_planner_probe_candidate(candidate) for candidate in candidates):
            return candidates
        if not candidates or not hasattr(self.ai_client, "validate_candidate_specs"):
            return candidates
        try:
            validated = self.ai_client.validate_candidate_specs(candidates, self.batch_size, ai_context)
        except Exception as exc:
            self.log.warning("planner candidate validation failed error=%s", exc)
            self.store.record_event(None, "planner_validation_error", {"error": str(exc)})
            return candidates
        return validated if isinstance(validated, list) else candidates

    def _simulate_candidates(
        self,
        ready_for_simulation: List[Tuple[int, CandidateSpec]],
        cycle_plan: Dict[str, Any] | None = None,
    ) -> List[Tuple[int, CandidateSpec, SimulationResult | SimulationFailure | SimulationPending | None]]:
        if not ready_for_simulation:
            return []
        if len(ready_for_simulation) > 1 and hasattr(self.brain_client, "simulate_many"):
            items = [(spec.expression, spec.settings) for _, spec in ready_for_simulation]
            for attempt in range(1, self.policy.max_retries + 1):
                try:
                    results = self._simulate_many_with_timeout(items, cycle_plan, attempt)
                    if len(results) != len(ready_for_simulation):
                        raise RuntimeError(
                            f"simulate_many returned {len(results)} results for {len(ready_for_simulation)} candidates"
                        )
                    return [
                        (candidate_id, spec, result)
                        for (candidate_id, spec), result in zip(ready_for_simulation, results)
                    ]
                except SimulationPendingError as exc:
                    return [
                        (candidate_id, spec, SimulationPending(exc.location))
                        for candidate_id, spec in ready_for_simulation
                    ]
                except Exception as exc:
                    if attempt >= self.policy.max_retries:
                        self.log.exception("batch simulation failed final_attempt=%s count=%s", attempt, len(items))
                    else:
                        self.log.warning(
                            "batch simulation failed attempt=%s/%s count=%s error=%s",
                            attempt,
                            self.policy.max_retries,
                            len(items),
                            exc,
                        )
                    for candidate_id, _spec in ready_for_simulation:
                        retry_count = self.store.increment_retry(candidate_id)
                        self.store.record_event(
                            candidate_id,
                            "simulation_error",
                            {
                                "error": str(exc),
                                "retry_count": retry_count,
                                "attempt": attempt,
                                "batch": True,
                            },
                        )
            return [(candidate_id, spec, None) for candidate_id, spec in ready_for_simulation]

        simulated: List[Tuple[int, CandidateSpec, SimulationResult | SimulationFailure | SimulationPending | None]] = []
        for candidate_id, spec in ready_for_simulation:
            result = None
            for attempt in range(1, self.policy.max_retries + 1):
                try:
                    result = self._simulate_one_with_timeout(candidate_id, spec, cycle_plan, attempt)
                    break
                except SimulationPendingError as exc:
                    result = SimulationPending(exc.location)
                    break
                except Exception as exc:
                    if attempt >= self.policy.max_retries:
                        self.log.exception("simulation failed candidate_id=%s final_attempt=%s", candidate_id, attempt)
                    else:
                        self.log.warning(
                            "simulation failed candidate_id=%s attempt=%s/%s error=%s",
                            candidate_id,
                            attempt,
                            self.policy.max_retries,
                            exc,
                        )
                    retry_count = self.store.increment_retry(candidate_id)
                    self.store.record_event(
                        candidate_id,
                        "simulation_error",
                        {"error": str(exc), "retry_count": retry_count, "attempt": attempt},
                    )
            simulated.append((candidate_id, spec, result))
        return simulated

    def _simulate_many_with_timeout(
        self,
        items: List[tuple[str, Dict[str, Any]]],
        cycle_plan: Dict[str, Any] | None,
        attempt: int,
    ) -> List[SimulationResult | SimulationFailure | SimulationPending]:
        timeout = _simulation_stage_timeout_seconds()
        self.log.info(
            "simulation batch started count=%s attempt=%s timeout_seconds=%s",
            len(items),
            attempt,
            timeout,
        )
        self._record_cycle_stage(
            "simulation_batch_started",
            cycle_plan,
            {"attempt": attempt, "candidate_count": len(items), "timeout_seconds": timeout},
        )
        started = time.monotonic()
        executor = ThreadPoolExecutor(max_workers=1)
        future = executor.submit(self.brain_client.simulate_many, items)
        try:
            results = future.result(timeout=timeout)
        except FutureTimeoutError:
            future.cancel()
            elapsed = time.monotonic() - started
            self.log.warning(
                "simulation batch timed out count=%s attempt=%s elapsed_seconds=%.3f timeout_seconds=%s",
                len(items),
                attempt,
                elapsed,
                timeout,
            )
            self._record_cycle_stage(
                "simulation_batch_timeout",
                cycle_plan,
                {
                    "attempt": attempt,
                    "candidate_count": len(items),
                    "timeout_seconds": timeout,
                    "elapsed_seconds": round(elapsed, 3),
                },
            )
            return [
                SimulationFailure(
                    "simulation_stage_timeout",
                    {
                        "attempt": attempt,
                        "timeout_seconds": timeout,
                        "elapsed_seconds": round(elapsed, 3),
                    },
                )
                for _ in items
            ]
        finally:
            executor.shutdown(wait=False, cancel_futures=True)
        elapsed = time.monotonic() - started
        self.log.info(
            "simulation batch finished count=%s attempt=%s elapsed_seconds=%.3f",
            len(items),
            attempt,
            elapsed,
        )
        self._record_cycle_stage(
            "simulation_batch_finished",
            cycle_plan,
            {
                "attempt": attempt,
                "candidate_count": len(items),
                "elapsed_seconds": round(elapsed, 3),
            },
        )
        return results

    def _simulate_one_with_timeout(
        self,
        candidate_id: int,
        spec: CandidateSpec,
        cycle_plan: Dict[str, Any] | None,
        attempt: int,
    ) -> SimulationResult | SimulationFailure:
        timeout = _simulation_stage_timeout_seconds()
        self._record_cycle_stage(
            "simulation_candidate_started",
            cycle_plan,
            {"attempt": attempt, "candidate_id": candidate_id, "timeout_seconds": timeout},
        )
        started = time.monotonic()
        executor = ThreadPoolExecutor(max_workers=1)
        future = executor.submit(self.brain_client.simulate, spec.expression, spec.settings)
        try:
            result = future.result(timeout=timeout)
        except FutureTimeoutError:
            future.cancel()
            elapsed = time.monotonic() - started
            self._record_cycle_stage(
                "simulation_candidate_timeout",
                cycle_plan,
                {
                    "attempt": attempt,
                    "candidate_id": candidate_id,
                    "timeout_seconds": timeout,
                    "elapsed_seconds": round(elapsed, 3),
                },
            )
            # Do NOT retry a stage timeout: the simulation is likely still running on the
            # platform, so retrying would fire a duplicate real simulation and waste quota.
            # Return a terminal failure so the caller stops instead of re-submitting.
            return SimulationFailure(
                "simulation_stage_timeout",
                {
                    "attempt": attempt,
                    "timeout_seconds": timeout,
                    "elapsed_seconds": round(elapsed, 3),
                },
            )
        finally:
            executor.shutdown(wait=False, cancel_futures=True)
        elapsed = time.monotonic() - started
        self._record_cycle_stage(
            "simulation_candidate_finished",
            cycle_plan,
            {"attempt": attempt, "candidate_id": candidate_id, "elapsed_seconds": round(elapsed, 3)},
        )
        return result


def _allowed_fields_from_context(ai_context: Dict[str, Any]) -> List[str]:
    research_context = ai_context.get("research_context")
    if not isinstance(research_context, dict):
        return []
    datafields = research_context.get("datafields")
    if not isinstance(datafields, dict):
        return []
    field_ids = datafields.get("field_ids")
    return [str(field) for field in field_ids] if isinstance(field_ids, list) else []


def _empty_field_pool_block(ai_context: Dict[str, Any]) -> Dict[str, Any]:
    """Return a non-empty block descriptor when the field allowlist is empty.

    build_field_catalog always seeds field_ids with BUILTIN_FIELDS, so an empty
    pool only happens when research_context/datafields is missing or malformed.
    Generating against an empty allowlist would fail-closed per-candidate in
    preflight and burn the whole batch; skip the round instead and surface why.
    """
    research_context = ai_context.get("research_context")
    if not isinstance(research_context, dict):
        return {"reason": "missing_research_context"}
    datafields = research_context.get("datafields")
    if not isinstance(datafields, dict):
        return {"reason": "missing_datafields"}
    if _allowed_fields_from_context(ai_context):
        return {}
    return {
        "reason": "empty_field_ids",
        "available": bool(datafields.get("available")),
        "error": datafields.get("error"),
    }


def _field_types_from_context(ai_context: Dict[str, Any]) -> Dict[str, str]:
    research_context = ai_context.get("research_context")
    if not isinstance(research_context, dict):
        return {}
    datafields = research_context.get("datafields")
    if not isinstance(datafields, dict):
        return {}
    field_types = datafields.get("field_types")
    if isinstance(field_types, dict):
        return {str(field): str(field_type) for field, field_type in field_types.items()}
    fields = datafields.get("fields")
    if not isinstance(fields, list):
        return {}
    result: Dict[str, str] = {}
    for field in fields:
        if isinstance(field, dict) and field.get("id") and field.get("type"):
            result[str(field["id"])] = str(field["type"])
    return result


def _field_metadata_from_context(ai_context: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    research_context = ai_context.get("research_context")
    if not isinstance(research_context, dict):
        return {}
    datafields = research_context.get("datafields")
    if not isinstance(datafields, dict):
        return {}
    fields = datafields.get("fields")
    if not isinstance(fields, list):
        return {}
    metadata: Dict[str, Dict[str, Any]] = {}
    for field in fields:
        if isinstance(field, dict) and field.get("id"):
            metadata[str(field["id"])] = field
    return metadata


def _is_unverified_production_probe(spec: CandidateSpec) -> bool:
    return str(spec.source or "") == "planner_unverified_probe"


def _is_planner_probe_candidate(spec: CandidateSpec) -> bool:
    return str(spec.source or "") in {"planner_unverified_probe", "planner_standardized_probe"}


def _balanced_ai_generation_enabled(ai_client: Any) -> bool:
    enabled = getattr(ai_client, "balanced_generation_enabled", False)
    if callable(enabled):
        try:
            enabled = enabled()
        except Exception:
            return False
    return enabled is True


def _planned_probe_ai_fill_size(
    candidates: List[CandidateSpec],
    ai_context: Dict[str, Any],
    batch_size: int,
) -> int:
    if candidates and not all(_is_planner_probe_candidate(candidate) for candidate in candidates):
        return 0
    research_context = ai_context.get("research_context")
    if not isinstance(research_context, dict):
        return 0
    plan = research_context.get("experiment_plan")
    if not isinstance(plan, dict):
        return 0
    quality_budget = plan.get("quality_budget")
    slots = quality_budget.get("slots") if isinstance(quality_budget, dict) else {}
    try:
        broad_explore = int(slots.get("broad_explore") or 0) if isinstance(slots, dict) else 0
    except (TypeError, ValueError):
        broad_explore = 0
    if broad_explore <= 0:
        return 0
    return max(0, int(batch_size or 0) - len(candidates))


def _probe_validation_stage(spec: CandidateSpec, default: str) -> str:
    metadata = spec.metadata if isinstance(spec.metadata, dict) else {}
    return str(metadata.get("validation_stage") or default)


def _production_probe_preflight_stage(spec: CandidateSpec, errors: List[str]) -> Dict[str, Any]:
    if not _is_planner_probe_candidate(spec):
        return {}
    return {
        "stage": "preflight_reject",
        "validation_stage": _probe_validation_stage(spec, "unverified_probe"),
        "next_action": "drop_probe_without_simulation",
        "errors": list(errors),
        **_production_probe_metadata(spec),
    }


def _production_probe_simulation_error_stage(spec: CandidateSpec, error: str) -> Dict[str, Any]:
    if not _is_planner_probe_candidate(spec):
        return {}
    return {
        "stage": "simulation_error",
        "validation_stage": _probe_validation_stage(spec, "unverified_probe"),
        "next_action": "do_not_scale_probe",
        "error": str(error),
        **_production_probe_metadata(spec),
    }


def _production_probe_result_stage(
    spec: CandidateSpec,
    metrics: Dict[str, Any],
    checks: Dict[str, Any],
    ai_context: Dict[str, Any],
) -> Dict[str, Any]:
    if not _is_planner_probe_candidate(spec):
        return {}
    thresholds = _quality_thresholds_from_context(ai_context)
    sharpe = _metric_float(metrics.get("sharpe"))
    fitness = _metric_float(metrics.get("fitness"))
    readiness = _probe_readiness_score(metrics, checks)
    setting_sharpe = _positive_float(thresholds.get("setting_sweep_sharpe"))
    setting_fitness = _positive_float(thresholds.get("setting_sweep_fitness"))
    setting_readiness = _positive_float(thresholds.get("setting_sweep_readiness"))
    optimize_sharpe = _positive_float(thresholds.get("optimize_sharpe")) or QUALITY_STOP_DEFAULT_SHARPE
    optimize_fitness = _positive_float(thresholds.get("optimize_fitness")) or QUALITY_STOP_DEFAULT_FITNESS

    stage = "reject"
    next_action = "downrank_field_or_structure"
    if (
        setting_sharpe
        and setting_fitness
        and sharpe >= setting_sharpe
        and fitness >= setting_fitness
        and (setting_readiness is None or readiness >= setting_readiness)
    ):
        stage = "sweep_ready"
        next_action = "run_settings_sweep_before_any_ai_optimization"
    elif sharpe >= optimize_sharpe and fitness >= optimize_fitness:
        stage = "optimize_ready"
        next_action = "allow_local_optimization_from_verified_probe"
    elif sharpe >= optimize_sharpe * 0.5 or fitness >= optimize_fitness * 0.5:
        stage = "watch"
        next_action = "store_evidence_without_scaling"

    return {
        "stage": stage,
        "validation_stage": _probe_validation_stage(spec, "unverified_probe"),
        "next_action": next_action,
        "metrics": {
            "sharpe": sharpe,
            "fitness": fitness,
            "readiness": readiness,
        },
        "thresholds": {
            "optimize_sharpe": optimize_sharpe,
            "optimize_fitness": optimize_fitness,
            "setting_sweep_sharpe": setting_sharpe,
            "setting_sweep_fitness": setting_fitness,
            "setting_sweep_readiness": setting_readiness,
        },
        **_production_probe_metadata(spec),
    }


def _production_probe_metadata(spec: CandidateSpec) -> Dict[str, Any]:
    metadata = spec.metadata if isinstance(spec.metadata, dict) else {}
    return {
        "probe_field": metadata.get("probe_field"),
        "probe_dataset_id": metadata.get("probe_dataset_id"),
        "probe_category": metadata.get("probe_category"),
        "probe_route": metadata.get("probe_route"),
        "probe_template_index": metadata.get("probe_template_index"),
    }


def _probe_readiness_score(metrics: Dict[str, Any], checks: Dict[str, Any]) -> float:
    quality_components = metrics.get("quality_components")
    if isinstance(quality_components, dict):
        for key in ("readiness_score", "submission_score"):
            value = _positive_float(quality_components.get(key))
            if value is not None:
                return value
    for key in ("readiness_score", "submission_score"):
        value = _positive_float(metrics.get(key))
        if value is not None:
            return value
    if not isinstance(checks, dict) or not checks:
        return 0.0
    terminal_checks = [
        item
        for item in checks.values()
        if isinstance(item, dict) and str(item.get("status") or "").strip().upper() in {"PASS", "FAIL"}
    ]
    if not terminal_checks:
        return 0.0
    passed = sum(1 for item in terminal_checks if str(item.get("status") or "").strip().upper() == "PASS")
    return passed / len(terminal_checks)


def _field_scout_fresh_generation_block(ai_context: Dict[str, Any]) -> Dict[str, Any]:
    research_context = ai_context.get("research_context")
    if not isinstance(research_context, dict):
        return {}
    plan = research_context.get("experiment_plan")
    plan_mode = str(plan.get("mode") or "") if isinstance(plan, dict) else ""
    if not isinstance(plan, dict) or not plan_mode.startswith("explore"):
        return {}
    production_rescue = plan.get("production_rescue")
    quality_budget = plan.get("quality_budget") if isinstance(plan.get("quality_budget"), dict) else {}
    slots = quality_budget.get("slots") if isinstance(quality_budget, dict) else {}
    probe_slots = 0
    if isinstance(slots, dict):
        try:
            probe_slots = max(0, int(slots.get("probe_new_fields") or 0))
        except (TypeError, ValueError):
            probe_slots = 0
    probe_recommendations = plan.get("probe_recommendations")
    if (
        isinstance(production_rescue, dict)
        and production_rescue.get("active")
        and probe_slots > 0
        and not (isinstance(probe_recommendations, list) and probe_recommendations)
    ):
        return {
            "reason": "PRODUCTION_RESCUE_NO_SAFE_PROBES",
            "plan_mode": plan_mode,
            "probe_slots": probe_slots,
            "policy": (
                "Production rescue requires explicit probe recommendations. Skip AI generation rather than "
                "spending tokens on broad weak-field exploration."
            ),
        }
    return {}


def _field_scout_avoid_reason_summary(top_fields: List[Dict[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for row in top_fields:
        if not isinstance(row, dict) or row.get("primary_policy") != "avoid_primary":
            continue
        reasons = [
            str(row.get("field_reason") or "").strip(),
            str(row.get("dataset_reason") or "").strip(),
            str(row.get("metadata_reason") or "").strip(),
            "lit_tower" if str(row.get("tower_status") or "") == "lit" else "",
        ]
        reason = next((item for item in reasons if item), "avoid_primary")
        counts[reason] = counts.get(reason, 0) + 1
    return counts


def _probe_recommendation_fields(recommendations: List[Any]) -> List[str]:
    fields: List[str] = []
    seen: set[str] = set()
    for recommendation in recommendations:
        if not isinstance(recommendation, dict):
            continue
        field = str(recommendation.get("field") or "").strip()
        if not field or field in seen:
            continue
        fields.append(field)
        seen.add(field)
    return fields


def _deterministic_fallback_candidates(
    batch_size: int,
    ai_context: Dict[str, Any],
    reason: str,
    is_duplicate: Callable[[str, Dict[str, Any]], bool] | None = None,
) -> List[CandidateSpec]:
    settings = dict(DEFAULT_SETTINGS)
    for key in DEFAULT_SETTINGS:
        if key in ai_context and ai_context.get(key) not in (None, ""):
            settings[key] = ai_context[key]

    allowed_fields = _allowed_fields_from_context(ai_context)
    field_types = _field_types_from_context(ai_context)
    event_fields = set(_event_fields_from_context(ai_context))
    auxiliary_fields = set(_auxiliary_fields_from_context(ai_context))
    rows = _deterministic_fallback_field_rows(ai_context)
    candidates: List[CandidateSpec] = []
    seen: set[str] = set()
    for row in rows:
        field = str(row.get("field") or row.get("id") or "").strip()
        if not field or field in seen or field in auxiliary_fields:
            continue
        if _deterministic_fallback_field_is_blocked(row):
            continue
        if allowed_fields and field not in set(allowed_fields):
            continue
        if field in event_fields:
            continue
        usage_constraints = row.get("usage_constraints")
        normalized_usage_constraints = _normalized_usage_constraints(usage_constraints)
        if normalized_usage_constraints and "requires_turnover_stabilizer" not in normalized_usage_constraints:
            continue
        field_type = str(row.get("type") or field_types.get(field) or "MATRIX").upper()
        if field_type not in {"MATRIX", "VECTOR"}:
            continue
        seen.add(field)
        for template_index, expression in enumerate(
            _deterministic_fallback_expressions(field, field_type, normalized_usage_constraints),
            start=1,
        ):
            errors = validate_expression(
                expression,
                allowed_fields=allowed_fields,
                field_types=field_types,
                enforce_auxiliary_field_roles=_enforce_auxiliary_field_roles_from_context(ai_context),
                auxiliary_fields=list(auxiliary_fields),
                event_fields=list(event_fields),
            )
            if errors:
                continue
            if is_duplicate is not None and is_duplicate(expression, settings):
                continue
            candidates.append(
                CandidateSpec(
                    expression=expression,
                    settings=dict(settings),
                    source="deterministic_fallback",
                    metadata={
                        "fallback_reason": reason,
                        "fallback_mode": "ai_unavailable_deterministic",
                        "fallback_field": field,
                        "fallback_field_type": field_type,
                        "fallback_dataset_id": row.get("dataset_id") or row.get("datasetId"),
                        "fallback_category": row.get("category"),
                        "fallback_template_index": template_index,
                        "fallback_usage_constraints": normalized_usage_constraints,
                    },
                )
            )
            if len(candidates) >= max(1, int(batch_size or 1)):
                return candidates
    return candidates


def _deterministic_fallback_field_rows(ai_context: Dict[str, Any]) -> List[Dict[str, Any]]:
    research_context = ai_context.get("research_context")
    if not isinstance(research_context, dict):
        return []
    rows: List[Dict[str, Any]] = []
    plan = research_context.get("experiment_plan") if isinstance(research_context.get("experiment_plan"), dict) else {}
    for scout in (plan.get("field_scout") if isinstance(plan, dict) else None, research_context.get("field_scout")):
        if not isinstance(scout, dict):
            continue
        for key in ("top_primary_fields", "top_fields"):
            values = scout.get(key)
            if isinstance(values, list):
                rows.extend(row for row in values if isinstance(row, dict))
    datafields = research_context.get("datafields")
    if not isinstance(datafields, dict):
        return rows
    fields = datafields.get("fields")
    if isinstance(fields, list):
        rows.extend(row for row in fields if isinstance(row, dict))
    return rows


def _deterministic_fallback_field_is_blocked(row: Dict[str, Any]) -> bool:
    if str(row.get("retest_reason") or "").strip():
        if row.get("field_reason") or row.get("metadata_reason") or row.get("primary_block_reason"):
            return True
        return str(row.get("dataset_reason") or "").strip() not in {
            "recent_dataset_failure_cluster",
            "failed_only_dataset_cluster",
        }
    if str(row.get("primary_policy") or "").strip() == "avoid_primary":
        return True
    for key in ("field_reason", "dataset_reason", "metadata_reason"):
        if str(row.get(key) or "").strip():
            return True
    return False


def _normalized_usage_constraints(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []
    constraints: List[str] = []
    for item in value:
        constraint = str(item or "").strip()
        if constraint:
            constraints.append(constraint)
    return constraints


def _deterministic_fallback_expressions(
    field: str,
    field_type: str,
    usage_constraints: List[str] | None = None,
) -> List[str]:
    value = f"vec_avg({field})" if field_type == "VECTOR" else field
    if usage_constraints and "requires_turnover_stabilizer" in usage_constraints:
        stabilized = f"winsorize(ts_backfill({value},120),std=4)"
        return [
            f"group_rank(ts_rank({stabilized},63),industry)",
            f"rank(ts_rank({stabilized},63))",
            f"rank(multiply(-1,ts_rank({stabilized},33)))",
            f"rank(ts_decay_linear(ts_backfill({value},120),20))",
        ]
    return [
        f"rank(ts_mean({value},20))",
        f"rank(ts_delta(ts_mean({value},66),22))",
        f"group_rank(ts_rank({value},63),industry)",
        f"rank(ts_zscore({value},63))",
    ]


def _event_fields_from_context(ai_context: Dict[str, Any]) -> List[str]:
    event_fields: List[str] = []
    for field_id, metadata in _field_metadata_from_context(ai_context).items():
        category = str(metadata.get("category") or "").lower()
        dataset_id = str(metadata.get("dataset_id") or metadata.get("datasetId") or "").lower()
        dataset_name = str(metadata.get("dataset_name") or metadata.get("datasetName") or "").lower()
        if (
            "news" in category
            or "event" in category
            or dataset_id.startswith(("news", "nws"))
            or "news" in dataset_name
            or "event" in dataset_name
        ):
            event_fields.append(field_id)
    return event_fields


def _profile_compliance_errors(spec: CandidateSpec, ai_context: Dict[str, Any]) -> List[str]:
    guidance = spec.metadata.get("profile_guidance") if isinstance(spec.metadata, dict) else None
    if not isinstance(guidance, dict) or not guidance:
        return []

    errors: List[str] = []
    expression = str(spec.expression or "")
    metadata = _field_metadata_from_context(ai_context)
    used_fields = _used_catalog_fields(expression, _allowed_fields_from_context(ai_context))
    guidance_text = _profile_guidance_text(guidance)
    required_family = _profile_required_family(guidance)
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


def _profile_required_family(guidance: Dict[str, Any]) -> str:
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
        if field.lower() in {"open", "high", "low", "close", "volume", "returns", "vwap", "cap", "adv20"}:
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
        if field.lower() in {"open", "high", "low", "close", "volume", "returns", "vwap", "cap", "adv20"}:
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
        return "other" in category or dataset_id.startswith("oth")
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
    token_pattern = re.escape(pattern).replace(r"\*", r"[A-Za-z0-9_]*")
    token_regex = re.compile(r"\b" + token_pattern + r"\b")
    return list(dict.fromkeys(token_regex.findall(expression)))


def _enforce_auxiliary_field_roles_from_context(ai_context: Dict[str, Any]) -> bool:
    research_context = ai_context.get("research_context")
    if not isinstance(research_context, dict):
        return False
    generation_policy = research_context.get("generation_policy")
    if isinstance(generation_policy, dict) and generation_policy.get("auxiliary_fields_must_not_be_primary"):
        return True
    syntax_constraints = research_context.get("syntax_constraints")
    return isinstance(syntax_constraints, dict) and bool(syntax_constraints.get("auxiliary_only_fields"))


def _auxiliary_fields_from_context(ai_context: Dict[str, Any]) -> List[str]:
    research_context = ai_context.get("research_context")
    if not isinstance(research_context, dict):
        return []
    syntax_constraints = research_context.get("syntax_constraints")
    if not isinstance(syntax_constraints, dict):
        return []
    fields = syntax_constraints.get("auxiliary_only_fields")
    return [str(field) for field in fields] if isinstance(fields, list) else []


def _quality_thresholds_from_context(ai_context: Dict[str, Any]) -> Dict[str, Any]:
    research_context = ai_context.get("research_context")
    if not isinstance(research_context, dict):
        return {}
    experiment_plan = research_context.get("experiment_plan")
    if not isinstance(experiment_plan, dict):
        return {}
    thresholds = experiment_plan.get("quality_thresholds")
    return dict(thresholds) if isinstance(thresholds, dict) else {}


def _quality_stop_loss_summary(
    store: AlphaStore,
    candidate_ids: List[int],
    batch_size: int,
    ai_context: Dict[str, Any],
) -> Dict[str, Any]:
    min_generated = min(max(1, int(batch_size or 0)), QUALITY_STOP_MIN_GENERATED)
    if len(candidate_ids) < min_generated:
        return {}

    rows: List[Dict[str, Any]] = []
    for candidate_id in candidate_ids:
        try:
            rows.append(store.get_candidate(candidate_id))
        except KeyError:
            continue
    if len(rows) < min_generated:
        return {}
    if any(str(row.get("status") or "") != "failed" for row in rows):
        return {}
    if any(_failed_due_simulation_unavailable(store, int(row.get("id") or 0)) for row in rows):
        return {}

    thresholds = _quality_thresholds_from_context(ai_context)
    sharpe_floor = _positive_float(thresholds.get("optimize_sharpe")) or QUALITY_STOP_DEFAULT_SHARPE
    fitness_floor = _positive_float(thresholds.get("optimize_fitness")) or QUALITY_STOP_DEFAULT_FITNESS
    best_sharpe = max((_metric_float(_loads_dict(row.get("metrics_json")).get("sharpe")) for row in rows), default=0.0)
    best_fitness = max(
        (_metric_float(_loads_dict(row.get("metrics_json")).get("fitness")) for row in rows),
        default=0.0,
    )
    if best_sharpe >= sharpe_floor or best_fitness >= fitness_floor:
        return {}
    return {
        "quality_stop_loss": 1,
        "quality_stop_reason": "bad_full_batch",
        "scope": {
            "region": ai_context.get("region", "USA"),
            "universe": ai_context.get("universe", "TOP3000"),
            "delay": ai_context.get("delay", 1),
            "neutralization": ai_context.get("neutralization", "INDUSTRY"),
        },
        "quality_stop_candidates": len(rows),
        "quality_stop_best_sharpe": round(best_sharpe, 6),
        "quality_stop_best_fitness": round(best_fitness, 6),
        "quality_stop_sharpe_floor": round(sharpe_floor, 6),
        "quality_stop_fitness_floor": round(fitness_floor, 6),
        "quality_stop_policy": (
            "all generated candidates failed and the best sharpe/fitness stayed below optimization trigger floors"
        ),
    }


def _failed_due_simulation_unavailable(store: AlphaStore, candidate_id: int) -> bool:
    if candidate_id <= 0:
        return False
    try:
        events = store.events_for_candidate(candidate_id)
    except Exception:
        return False
    saw_simulation_error = False
    for event in events:
        event_type = str(event.get("event_type") or "")
        metadata = _loads_dict(event.get("metadata_json"))
        if event_type == "simulation_error":
            saw_simulation_error = True
        if event_type == "status:failed" and str(metadata.get("reason") or "") in {
            "retry_limit",
            "simulation_error",
        }:
            return True
    return saw_simulation_error


def _should_recheck_pending_candidates(cycle_plan: Dict[str, Any] | None) -> bool:
    if cycle_plan is None:
        return True
    return str(cycle_plan.get("mode") or "") == "recover_pending"


def _candidate_has_status_transition(store: AlphaStore, candidate_id: int, status: str) -> bool:
    if candidate_id <= 0:
        return False
    expected = f"status:{status}"
    try:
        events = store.events_for_candidate(candidate_id)
    except Exception:
        return False
    return any(str(event.get("event_type") or "") == expected for event in events)


def _latest_simulation_pending_location(store: AlphaStore, candidate_id: int) -> str:
    if candidate_id <= 0:
        return ""
    try:
        events = store.events_for_candidate(candidate_id)
    except Exception:
        return ""
    for event in reversed(events):
        if str(event.get("event_type") or "") not in {"simulation_pending", "status:check_pending"}:
            continue
        metadata = _loads_dict(event.get("metadata_json"))
        location = str(metadata.get("location") or metadata.get("simulation_location") or "").strip()
        if location:
            return location
    return ""


def _candidate_needs_pending_recheck(store: AlphaStore, candidate_id: int) -> bool:
    if _latest_simulation_pending_location(store, candidate_id):
        return True
    try:
        events = store.events_for_candidate(candidate_id)
    except Exception:
        return False
    for event in reversed(events):
        if str(event.get("event_type") or "") != "status:check_pending":
            continue
        metadata = _loads_dict(event.get("metadata_json"))
        if str(metadata.get("reason") or "") == "simulation_pending":
            return True
        errors = metadata.get("errors")
        if isinstance(errors, list) and _only_terminal_wait_errors([str(error) for error in errors]):
            return True
        return False
    return False


def _stale_simulation_pending_metadata(store: AlphaStore, candidate_id: int) -> Dict[str, Any]:
    location = _latest_simulation_pending_location(store, candidate_id)
    if not location:
        return {}
    try:
        events = store.events_for_candidate(candidate_id)
    except Exception:
        return {}
    matching = []
    for event in events:
        if str(event.get("event_type") or "") != "simulation_pending":
            continue
        metadata = _loads_dict(event.get("metadata_json"))
        event_location = str(metadata.get("location") or metadata.get("simulation_location") or "").strip()
        if event_location == location:
            matching.append(event)
    max_attempts = _pending_recheck_stale_attempts()
    if len(matching) < max_attempts:
        return {}
    first_seen = _event_datetime(matching[0])
    if first_seen is None:
        return {}
    stale_seconds = (datetime.now(timezone.utc) - first_seen.astimezone(timezone.utc)).total_seconds()
    max_age = _pending_recheck_stale_seconds()
    if stale_seconds < max_age:
        return {}
    return {
        "location": location,
        "attempts": len(matching),
        "first_pending_at": first_seen.isoformat(),
        "stale_seconds": round(max(0.0, stale_seconds), 3),
        "stale_attempt_threshold": max_attempts,
        "stale_seconds_threshold": max_age,
    }


def _pending_recheck_stale_attempts() -> int:
    try:
        return max(1, int(os.environ.get("PENDING_RECHECK_STALE_ATTEMPTS", "4")))
    except ValueError:
        return 4


def _pending_recheck_stale_seconds() -> float:
    try:
        return max(0.0, float(os.environ.get("PENDING_RECHECK_STALE_SECONDS", "1800")))
    except ValueError:
        return 1800.0


def _event_datetime(event: Dict[str, Any]) -> datetime | None:
    text = str(event.get("created_at") or "").strip()
    if not text:
        return None
    try:
        result = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if result.tzinfo is None:
        result = result.replace(tzinfo=timezone.utc)
    return result


def _positive_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if result > 0 else None


def _metric_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _loads_dict(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    try:
        data = json.loads(str(value or "{}"))
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def _candidate_structural_scope_matches(row: Dict[str, Any], settings: Dict[str, Any]) -> bool:
    candidate_settings = _loads_dict(row.get("settings_json"))
    merged_candidate = dict(DEFAULT_SETTINGS)
    merged_candidate.update(candidate_settings)
    merged_target = dict(DEFAULT_SETTINGS)
    merged_target.update(settings or {})
    for key in ("instrumentType", "region", "universe", "delay"):
        if _scope_value(merged_candidate.get(key)) != _scope_value(merged_target.get(key)):
            return False
    return True


def _source_uses_structural_dedup(source: Any) -> bool:
    text = str(source or "")
    return text.startswith("model:") or text == "openai_compatible"


def _candidate_has_simulated_history(candidate: Dict[str, Any]) -> bool:
    return str(candidate.get("status") or "") in {"approved", "submitted", "failed", "check_pending"}


def _setting_sweep_target_validation_errors(expression: str, research_context: Dict[str, Any]) -> List[str]:
    datafields = research_context.get("datafields") if isinstance(research_context.get("datafields"), dict) else {}
    field_ids = datafields.get("field_ids")
    if not isinstance(field_ids, list) or not field_ids:
        return []
    field_types = datafields.get("field_types") if isinstance(datafields.get("field_types"), dict) else {}
    return validate_expression(
        expression,
        allowed_fields=field_ids,
        field_types=field_types,
        enforce_auxiliary_field_roles=True,
    )


def _candidate_matches_context(row: Dict[str, Any], context: Dict[str, Any]) -> bool:
    settings = _loads_dict(row.get("settings_json"))
    merged = dict(DEFAULT_SETTINGS)
    merged.update(settings)
    for key in _RECHECK_SCOPE_KEYS:
        if key not in context or context.get(key) in (None, ""):
            continue
        if _scope_value(merged.get(key)) != _scope_value(context.get(key)):
            return False
    return True


def _current_quarter_date_range(today: date | None = None) -> tuple[str, str]:
    current = today or date.today()
    start_month = ((current.month - 1) // 3) * 3 + 1
    start = date(current.year, start_month, 1)
    if start_month == 10:
        next_quarter = date(current.year + 1, 1, 1)
    else:
        next_quarter = date(current.year, start_month + 3, 1)
    end = next_quarter - timedelta(days=1)
    return start.isoformat(), end.isoformat()


def _non_retryable_ai_generation_error(exc: Exception) -> str | None:
    text = str(exc).lower()
    timeout_markers = (
        "generation stage timed out",
        "model call timed out",
        "timed out",
        "timeout",
    )
    if any(marker in text for marker in timeout_markers):
        return "ai_generation_timeout"
    network_markers = (
        "name or service not known",
        "temporary failure in name resolution",
        "nodename nor servname provided",
        "failed to resolve",
        "could not resolve host",
        "connection refused",
        "network is unreachable",
        "no route to host",
        "remote end closed connection without response",
        "connection reset by peer",
        "connection aborted",
    )
    if any(marker in text for marker in network_markers):
        return "ai_network_blocked"
    quota_markers = (
        "insufficient_user_quota",
        "insufficient_quota",
        "prepay",
        "prepaid",
        "预扣费额度失败",
        "用户剩余额度",
        "remaining balance",
    )
    if any(marker in text for marker in quota_markers):
        return "ai_quota_blocked"
    config_markers = ("invalid_api_key", "invalid api key", "unauthorized", "forbidden")
    if any(marker in text for marker in config_markers):
        return "ai_config_blocked"
    return None


def _terminal_simulation_resume_error(exc: Exception) -> bool:
    text = str(exc or "").strip().lower()
    terminal_markers = (
        "simulation failed on platform",
        "there was an error while running the simulation",
    )
    return any(marker in text for marker in terminal_markers)


def _ai_generation_stage_timeout_seconds() -> float:
    raw = os.environ.get("AI_GENERATION_STAGE_TIMEOUT_SECONDS", "90")
    try:
        return max(0.01, float(raw))
    except (TypeError, ValueError):
        return 90.0


def _simulation_stage_timeout_seconds() -> float:
    raw = os.environ.get("SIMULATION_STAGE_TIMEOUT_SECONDS", "900")
    try:
        return max(0.01, float(raw))
    except (TypeError, ValueError):
        return 900.0


def _post_dedup_refill_rounds() -> int:
    raw = os.environ.get("POST_DEDUP_REFILL_ROUNDS", "2")
    try:
        return max(0, int(raw))
    except (TypeError, ValueError):
        return 2


def _scope_value(value: Any) -> str:
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return ""
        try:
            return f"{float(text):.12g}"
        except ValueError:
            return text.upper()
    if isinstance(value, (int, float)):
        return f"{float(value):.12g}"
    return str(value if value is not None else "").strip().upper()


def _pending_recheck_priority(row: Dict[str, Any]) -> tuple[float, int]:
    metrics = _loads_dict(row.get("metrics_json"))
    score = _numeric(metrics.get("sharpe")) + _numeric(metrics.get("fitness"))
    return score, int(row.get("id") or 0)


def _order_pending_for_cycle_plan(
    pending: List[Dict[str, Any]],
    cycle_plan: Dict[str, Any] | None,
) -> List[Dict[str, Any]]:
    ordered = sorted(pending, key=_pending_recheck_priority, reverse=True)
    if not isinstance(cycle_plan, dict) or str(cycle_plan.get("mode") or "") != "recover_pending":
        return ordered
    try:
        target_id = int(cycle_plan.get("target_candidate_id") or 0)
    except (TypeError, ValueError):
        target_id = 0
    if target_id <= 0:
        return ordered
    target = [row for row in ordered if int(row.get("id") or 0) == target_id]
    if not target:
        return ordered
    return [*target, *[row for row in ordered if int(row.get("id") or 0) != target_id]]


def _numeric(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return -999.0


def _only_terminal_wait_errors(errors: List[str]) -> bool:
    return bool(errors) and all(_is_terminal_wait_error(error) for error in errors)


def _is_terminal_wait_error(error: str) -> bool:
    name, separator, status = str(error or "").partition(":")
    if not separator:
        return False
    if normalize_check_name(name) not in _TERMINAL_WAIT_CHECKS:
        return False
    return status.strip().upper() in _TERMINAL_WAIT_STATUSES
