from __future__ import annotations

import json
import inspect
import logging
from datetime import date, timedelta
from dataclasses import replace
from typing import Any, Dict, List, Tuple

from .clients import AIClient, BrainClient
from .context_builder import build_ai_research_context
from .db import AlphaStore
from .expression_similarity import expression_signature_metadata, expression_variant_key
from .field_catalog import build_field_catalog
from .guards import SubmissionPolicy, evaluate_submission_readiness, normalize_check_name
from .models import CandidateSpec, DEFAULT_SETTINGS, SimulationFailure, SimulationResult
from .preflight import validate_expression


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

    def _record_cycle_outcome(self, cycle_plan: Dict[str, Any] | None, summary: Dict[str, int]) -> None:
        active_cycle_plan = self._active_cycle_plan(cycle_plan)
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

    def run_once(self, cycle_plan: Dict[str, Any] | None = None) -> Dict[str, int]:
        self.log.info("worker cycle started batch_size=%s", self.batch_size)
        summary = {"generated": 0, "approved": 0, "submitted": 0, "failed": 0, "pending": 0, "skipped": 0}
        submitted_this_round = 0
        candidates = None
        ai_context = self._build_cycle_ai_context(cycle_plan=cycle_plan)
        effective_policy = self._policy_for_ai_context(ai_context)
        submitted_this_round = self._recheck_pending_candidates(
            ai_context,
            effective_policy,
            summary,
            submitted_this_round,
        )
        candidates = self._planned_candidates(ai_context)
        if candidates is not None:
            candidates = self._validate_planned_candidates(candidates, ai_context)
            self._record_ai_client_diagnostics()
            self._record_validator_rejections(summary)
        else:
            for attempt in range(1, self.policy.max_retries + 1):
                try:
                    candidates = self.ai_client.generate_candidates(self.batch_size, ai_context)
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
        for spec in candidates:
            duplicate = self.store.find_duplicate_candidate(spec.expression, spec.settings)
            if duplicate is not None:
                summary["skipped"] += 1
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
            self.log.info("candidate generated id=%s source=%s", candidate_id, spec.source)
            summary["generated"] += 1
            generated_metadata = {"expression": spec.expression}
            if spec.metadata:
                generated_metadata["ai_metadata"] = spec.metadata
            self.store.record_event(candidate_id, "generated", generated_metadata)

            allowed_fields = _allowed_fields_from_context(ai_context)
            field_types = _field_types_from_context(ai_context)
            preflight_errors = validate_expression(
                spec.expression,
                allowed_fields=allowed_fields,
                field_types=field_types,
                enforce_auxiliary_field_roles=_enforce_auxiliary_field_roles_from_context(ai_context),
                auxiliary_fields=_auxiliary_fields_from_context(ai_context),
            )
            if preflight_errors:
                self.store.record_event(candidate_id, "preflight_failed", {"errors": preflight_errors})
                self.store.transition(candidate_id, "failed", {"reason": "preflight", "errors": preflight_errors})
                summary["failed"] += 1
                continue
            self.store.transition(candidate_id, "preflight_passed")
            ready_for_simulation.append((candidate_id, spec))

        for candidate_id, spec, result in self._simulate_candidates(ready_for_simulation):
            if isinstance(result, SimulationFailure):
                self.store.record_event(candidate_id, "simulation_error", {"error": result.error, "raw": result.raw})
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
            self.store.transition(candidate_id, "simulated", {"alpha_id": result.alpha_id})
            self.store.transition(candidate_id, "metric_passed", {"alpha_id": result.alpha_id})

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

        self.log.info("worker cycle finished summary=%s", summary)
        self._record_cycle_outcome(cycle_plan, summary)
        return summary

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
    ) -> int:
        pending = [
            row
            for row in self.store.list_candidates(status="check_pending")
            if _candidate_matches_context(row, self.context)
        ]
        if not pending:
            return submitted_this_round
        pending = sorted(pending, key=_pending_recheck_priority, reverse=True)
        limit = max(0, min(int(self.batch_size or 0), 4))
        for row in pending[:limit]:
            candidate_id = int(row["id"])
            alpha_id = str(row.get("alpha_id") or "").strip()
            metrics = _loads_dict(row.get("metrics_json"))
            checks = _loads_dict(row.get("checks_json"))
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

        submit = self.brain_client.submit_alpha(alpha_id, dry_run=not self.policy.auto_submit)
        if submit.submitted and submit.stage == "OS":
            self.store.transition(candidate_id, "submitted", {"alpha_id": alpha_id})
            self.log.info("candidate submitted id=%s alpha_id=%s", candidate_id, alpha_id)
            submitted_this_round += 1
            summary["submitted"] += 1
        else:
            self.store.record_event(
                candidate_id,
                "dry_run_submit",
                {"alpha_id": alpha_id, "stage": submit.stage, "message": submit.message},
            )
            self.log.info("candidate dry_run_submit id=%s alpha_id=%s", candidate_id, alpha_id)
        return submitted_this_round

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
        for threshold_key, policy_key in threshold_map.items():
            value = _positive_float(thresholds.get(threshold_key))
            if value is not None:
                updates[policy_key] = value
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
        if not isinstance(plan, dict) or plan.get("mode") != "setting_sweep":
            return None
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

    def _validate_planned_candidates(
        self,
        candidates: List[CandidateSpec],
        ai_context: Dict[str, Any],
    ) -> List[CandidateSpec]:
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
    ) -> List[Tuple[int, CandidateSpec, SimulationResult | SimulationFailure | None]]:
        if not ready_for_simulation:
            return []
        if len(ready_for_simulation) > 1 and hasattr(self.brain_client, "simulate_many"):
            items = [(spec.expression, spec.settings) for _, spec in ready_for_simulation]
            for attempt in range(1, self.policy.max_retries + 1):
                try:
                    results = self.brain_client.simulate_many(items)
                    if len(results) != len(ready_for_simulation):
                        raise RuntimeError(
                            f"simulate_many returned {len(results)} results for {len(ready_for_simulation)} candidates"
                        )
                    return [
                        (candidate_id, spec, result)
                        for (candidate_id, spec), result in zip(ready_for_simulation, results)
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

        simulated: List[Tuple[int, CandidateSpec, SimulationResult | SimulationFailure | None]] = []
        for candidate_id, spec in ready_for_simulation:
            result = None
            for attempt in range(1, self.policy.max_retries + 1):
                try:
                    result = self.brain_client.simulate(spec.expression, spec.settings)
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


def _allowed_fields_from_context(ai_context: Dict[str, Any]) -> List[str]:
    research_context = ai_context.get("research_context")
    if not isinstance(research_context, dict):
        return []
    datafields = research_context.get("datafields")
    if not isinstance(datafields, dict):
        return []
    field_ids = datafields.get("field_ids")
    return [str(field) for field in field_ids] if isinstance(field_ids, list) else []


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


def _positive_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if result > 0 else None


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
