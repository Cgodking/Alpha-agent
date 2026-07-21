from __future__ import annotations

import json
import hmac
import math
import os
import signal
import subprocess
import sys
import time
import threading
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Dict, List
from urllib.parse import parse_qs, urlparse

from .clients import BrainHTTPClient, LocalBrainClient
from .config import load_config
from .context_builder import build_ai_research_context
from .db import AlphaStore, utc_now
from .env_file import load_env_file
from .field_catalog import build_field_catalog
from .field_scout import build_field_scout
from .guards import normalize_check_name
from .health import daemon_health
from .metrics import compute_efficiency_metrics
from .scopes import SCOPE_PRESETS, platform_scope_rows, preset_rows
from .scheduler import build_cycle_plan


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
DEFAULT_DAEMON_STDOUT = PROJECT_ROOT / "logs" / "daemon.stdout.log"
DEFAULT_WEB_LOG = PROJECT_ROOT / "logs" / "web.log"
MAX_WEB_BACKTEST_BATCH = 8
DEFAULT_WEB_RUN_MINUTES = 240.0
STATUS_QUEUE_EXAMPLE_LIMIT = 2
STATUS_QUEUE_ITEM_KEYS = (
    "id",
    "expression",
    "status",
    "source",
    "alpha_id",
    "settings",
    "metrics",
    "sharpe",
    "fitness",
    "turnover",
    "quality_score",
    "readiness_score",
    "failed_checks",
    "warning_checks",
    "pending_checks",
    "queue",
    "queue_reason",
    "last_relevant_event",
)


def build_daemon_argv(
    db_path: Path,
    env_file: Path,
    log_file: Path,
    preset: str = "",
    region: str = "",
    universe: str = "",
    delay: int | None = None,
    neutralization: str = "",
    scope_rotation: List[Dict[str, Any]] | None = None,
    batch_size: int = MAX_WEB_BACKTEST_BATCH,
    loop_seconds: float = 60.0,
    run_minutes: float = DEFAULT_WEB_RUN_MINUTES,
    throughput_mode: bool = True,
    generator_mode: str = "single",
    orchestration_mode: str = "lean",
    auto_submit: bool = False,
) -> List[str]:
    argv = [
        sys.executable,
        "-m",
        "alpha.cli",
        "--db",
        str(db_path),
        "--env-file",
        str(env_file),
        "--log-file",
        str(log_file),
        "daemon",
    ]
    if throughput_mode:
        argv.append("--throughput-mode")
    if generator_mode:
        argv.extend(["--generator-mode", generator_mode])
    if orchestration_mode:
        argv.extend(["--orchestration-mode", orchestration_mode])
    argv.append("--auto-submit" if auto_submit else "--no-auto-submit")
    if scope_rotation:
        argv.extend(["--scope-json", json.dumps(scope_rotation, sort_keys=True, separators=(",", ":"))])
    elif preset:
        argv.extend(["--preset", preset])
    else:
        if region:
            argv.extend(["--region", region])
        if universe:
            argv.extend(["--universe", universe])
        if delay is not None:
            argv.extend(["--delay", str(delay)])
        if neutralization:
            argv.extend(["--neutralization", neutralization])
    argv.extend(
        [
            "--batch-size",
            str(int(batch_size)),
            "--loop-seconds",
            _format_number(loop_seconds),
        ]
    )
    if run_minutes > 0:
        argv.extend(["--run-minutes", _format_number(run_minutes)])
    return argv


def scope_options() -> Dict[str, Any]:
    rows = list(platform_scope_rows())
    regions = sorted({row["region"] for row in rows})
    return {"regions": regions, "scopes": rows}


def tail_file(path: str | Path, line_count: int = 300, max_bytes: int = 2_000_000) -> str:
    path = Path(path)
    if not path.exists():
        return f"{path} does not exist.\n"
    with path.open("rb") as handle:
        handle.seek(0, os.SEEK_END)
        size = handle.tell()
        handle.seek(max(0, size - max_bytes), os.SEEK_SET)
        data = handle.read().decode("utf-8", errors="replace")
    return "".join(data.splitlines(keepends=True)[-line_count:])


def is_pid_running(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    if _pid_state(pid) == "Z":
        return False
    return True


def _pid_state(pid: int) -> str:
    try:
        stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    except OSError:
        return ""
    parts = stat.split()
    return parts[2] if len(parts) > 2 else ""


class ControlService:
    def __init__(
        self,
        store: AlphaStore,
        db_path: Path,
        env_file: Path,
        log_file: Path,
        daemon_stdout_log: Path = DEFAULT_DAEMON_STDOUT,
        web_log: Path = DEFAULT_WEB_LOG,
        popen_factory: Callable[..., Any] = subprocess.Popen,
        pid_running: Callable[[int], bool] = is_pid_running,
        kill_func: Callable[[int, int], None] = os.kill,
        killpg_func: Callable[[int, int], None] = os.killpg,
        sleep_func: Callable[[float], None] = time.sleep,
        status_cache_ttl_seconds: float = 5.0,
        research_context_cache_ttl_seconds: float = 30.0,
    ):
        self.store = store
        self.db_path = Path(db_path)
        self.env_file = Path(env_file)
        self.log_file = Path(log_file)
        self.daemon_stdout_log = Path(daemon_stdout_log)
        self.web_log = Path(web_log)
        self.popen_factory = popen_factory
        self.pid_running = pid_running
        self.kill_func = kill_func
        self.killpg_func = killpg_func
        self.sleep_func = sleep_func
        self.status_cache_ttl_seconds = float(status_cache_ttl_seconds)
        self.research_context_cache_ttl_seconds = float(research_context_cache_ttl_seconds)
        self._status_cache_lock = threading.Lock()
        self._status_cache: Dict[str, Any] = {}
        self._research_context_cache_lock = threading.Lock()
        self._research_context_cache: Dict[str, Any] = {}
        self._manual_submit_lock = threading.Lock()
        # Reference to the daemon Popen we launched, so we can reap it when it exits
        # on its own (run-minutes limit, quota stop) instead of leaving a zombie.
        self._daemon_process: Any = None

    def start(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        state = self.store.get_run_state("daemon")
        pid = int(state.get("pid") or 0)
        if self.pid_running(pid):
            return {"success": False, "message": f"daemon already running pid={pid}", "pid": pid}

        scope_rotation = _scope_rotation_from_payload(payload)
        scope = scope_rotation[0] if scope_rotation else _scope_from_payload(payload)
        preset = scope.get("preset", "")
        batch_size = _bounded_int(payload.get("batch_size"), default=MAX_WEB_BACKTEST_BATCH, minimum=1, maximum=MAX_WEB_BACKTEST_BATCH)
        loop_seconds = _bounded_float(payload.get("loop_seconds"), default=60.0, minimum=1.0, maximum=3600.0)
        run_minutes = _bounded_float(payload.get("run_minutes"), default=DEFAULT_WEB_RUN_MINUTES, minimum=1.0, maximum=1440.0)
        generator_mode = _generator_mode_from_payload(payload)
        orchestration_mode = _orchestration_mode_from_payload(payload)
        auto_submit = _auto_submit_from_payload(payload)
        argv = build_daemon_argv(
            db_path=self.db_path,
            env_file=self.env_file,
            log_file=self.log_file,
            preset=preset,
            region=str(scope.get("region", "")),
            universe=str(scope.get("universe", "")),
            delay=scope.get("delay"),
            neutralization=str(scope.get("neutralization", "")),
            scope_rotation=scope_rotation,
            batch_size=batch_size,
            loop_seconds=loop_seconds,
            run_minutes=run_minutes,
            generator_mode=generator_mode,
            orchestration_mode=orchestration_mode,
            auto_submit=auto_submit,
        )
        env = dict(os.environ)
        env["PYTHONPATH"] = _prepend_path(env.get("PYTHONPATH", ""), SRC_ROOT)
        self.daemon_stdout_log.parent.mkdir(parents=True, exist_ok=True)
        stdout = self.daemon_stdout_log.open("ab")
        try:
            process = self.popen_factory(
                argv,
                cwd=str(PROJECT_ROOT),
                env=env,
                stdout=stdout,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        finally:
            stdout.close()

        started_at = utc_now()
        run_state = {
            "pid": int(process.pid),
            "status": "running",
            "preset": "" if scope_rotation else preset,
            "scope": {key: value for key, value in scope.items() if key != "preset"},
            "scope_rotation": scope_rotation,
            "batch_size": batch_size,
            "loop_seconds": loop_seconds,
            "run_minutes": run_minutes,
            "generator_mode": generator_mode,
            "orchestration_mode": orchestration_mode,
            "auto_submit": auto_submit,
            "throughput_mode": True,
            "stop_after_at": _stop_after_at(run_minutes),
            "argv": argv,
            "started_at": started_at,
            "stopped_at": "",
            "stop_reason": "",
        }
        self.store.set_run_state("daemon", run_state)
        self.store.record_event(None, "web_daemon_started", run_state)
        self._daemon_process = process
        self.clear_status_cache()
        return {"success": True, "message": f"daemon started pid={process.pid}", "pid": int(process.pid), "argv": argv}

    def _reap_daemon_process(self) -> None:
        # Reap our launched daemon if it has exited, so it does not linger as a zombie.
        process = self._daemon_process
        if process is None:
            return
        poll = getattr(process, "poll", None)
        if callable(poll):
            try:
                if poll() is not None:
                    self._daemon_process = None
            except Exception:
                self._daemon_process = None

    def _signal_process(self, pid: int, sig: int) -> None:
        # The daemon is started with start_new_session=True, so its pid is also the
        # process-group id. Signal the whole group first to also reach any children
        # the daemon spawned; fall back to the single pid if the group is gone.
        if pid <= 0:
            return
        try:
            self.killpg_func(pid, sig)
            return
        except (OSError, ProcessLookupError):
            pass
        try:
            self.kill_func(pid, sig)
        except (OSError, ProcessLookupError):
            pass

    def stop(self) -> Dict[str, Any]:
        self._reap_daemon_process()
        state = self.store.get_run_state("daemon")
        pid = int(state.get("pid") or 0)
        if not self.pid_running(pid):
            interrupted_preflight = self._interrupt_recoverable_preflight(state)
            stop_reason = str(state.get("stop_reason") or "").strip() or "exited"
            state.update({"status": "stopped", "stopped_at": utc_now(), "stop_reason": stop_reason})
            self.store.set_run_state("daemon", state)
            self.clear_status_cache()
            return {
                "success": True,
                "message": "daemon is not running",
                "interrupted_preflight": interrupted_preflight,
                "reason": stop_reason,
            }

        self._signal_process(pid, signal.SIGINT)
        self.sleep_func(1.0)
        if self.pid_running(pid):
            self._signal_process(pid, signal.SIGTERM)
            self.sleep_func(1.0)
        if self.pid_running(pid):
            # Last resort: a daemon that ignores SIGINT/SIGTERM must still be killable.
            self._signal_process(pid, signal.SIGKILL)
        latest_state = self.store.get_run_state("daemon")
        if int(latest_state.get("pid") or 0) == pid:
            state = latest_state
        interrupted_preflight = self._interrupt_recoverable_preflight(state)
        stopped_at = str(state.get("stopped_at") or "").strip() or utc_now()
        stop_reason = str(state.get("stop_reason") or "").strip() or "web_stop"
        state.update({"status": "stopped", "stopped_at": stopped_at, "stop_reason": stop_reason})
        self.store.set_run_state("daemon", state)
        self.store.record_event(
            None,
            "web_daemon_stopped",
            {"pid": pid, "interrupted_preflight": interrupted_preflight, "reason": stop_reason},
        )
        self.clear_status_cache()
        return {
            "success": True,
            "message": f"daemon stopped pid={pid}",
            "interrupted_preflight": interrupted_preflight,
            "reason": stop_reason,
        }

    def _count_recoverable_preflight(self, state: Dict[str, Any]) -> int:
        started_at = str(state.get("started_at") or "").strip()
        return self.store.count_preflight_passed_candidates(created_since=started_at or None)

    def _interrupt_recoverable_preflight(self, state: Dict[str, Any]) -> int:
        started_at = str(state.get("started_at") or "").strip()
        return self.store.fail_preflight_passed_candidates(
            created_since=started_at or None,
            reason="interrupted_after_preflight",
        )

    def status(self) -> Dict[str, Any]:
        cached = self._cached_status()
        if cached is not None:
            return cached

        self._reap_daemon_process()
        state = self.store.get_run_state("daemon")
        health = daemon_health(self.store)
        state = _normalize_daemon_state_for_status(state, health=health)
        rotation_state = self.store.get_run_state("scope_rotation")
        pid = int(state.get("pid") or 0)
        running = self.pid_running(pid)
        run_started_at = str(state.get("started_at") or "")
        history_counts = self.store.status_counts()
        counts = self.store.status_counts(created_since=run_started_at) if run_started_at else history_counts
        active_scope = self._active_research_scope(state)
        scheduler_plan = build_cycle_plan(
            self.store,
            active_scope,
            batch_size=int(state.get("batch_size") or MAX_WEB_BACKTEST_BATCH),
        )
        research = (
            self._latest_research_context_from_events(created_since=run_started_at)
            if running
            else self._research_context(state, scheduler_plan)
        )
        efficiency = compute_efficiency_metrics(self.store, active_scope, created_since=run_started_at or None)
        payload = {
            "running": running,
            "daemon": state,
            "scope_rotation_state": rotation_state,
            "pid": pid if running else None,
            "health": health,
            "counts": counts,
            "history_counts": history_counts,
            "model_scores": self._model_scores(created_since=run_started_at),
            "run_started_at": run_started_at,
            "recent_candidates": self._recent_candidates(limit=20, created_since=run_started_at),
            "top_alphas": self._top_alphas(limit=10, created_since=run_started_at),
            "research_plan": research.get("experiment_plan", {}),
            "research_analysis": research.get("analysis", {}),
            "candidate_queues": _compact_candidate_queues_for_status(research.get("candidate_queues", {})),
            "efficiency": efficiency,
            "scheduler_plan": scheduler_plan,
            "cooldowns": scheduler_plan.get("constraints", {}),
            "auto_submit": bool(state.get("auto_submit")),
            "max_final_submits_per_round": _submit_cap_from_env(),
            "logs": {
                "alpha": str(self.log_file),
                "daemon": str(self.daemon_stdout_log),
                "web": str(self.web_log),
            },
            "updated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        }
        self._store_status_cache(payload)
        return deepcopy(payload)

    def clear_logs(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        log_paths = self._log_paths()
        if bool(payload.get("all")):
            names = list(log_paths)
        else:
            name = str(payload.get("file") or "alpha").strip().lower()
            if name not in log_paths:
                raise ValueError(f"unknown log file: {name}")
            names = [name]

        cleared: List[str] = []
        for name in names:
            path = log_paths[name]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.open("w", encoding="utf-8").close()
            cleared.append(name)
        self.store.record_event(None, "web_logs_cleared", {"cleared": cleared})
        return {"success": True, "message": f"cleared logs: {', '.join(cleared)}", "cleared": cleared}

    def check_top_alpha(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        alpha_id, candidate = self._current_top_alpha(payload)
        client = self._brain_client()
        correlation_reader = getattr(client, "get_alpha_correlations", None)
        if callable(correlation_reader):
            correlations = correlation_reader(alpha_id)
            correlations = correlations if isinstance(correlations, dict) else {}
            checks = _loads_json(candidate.get("checks_json"))
            checks = checks if isinstance(checks, dict) else {}
            for key in ("self", "production"):
                item = correlations.get(key)
                if isinstance(item, dict) and item.get("name"):
                    checks[str(item["name"])] = {
                        field: item.get(field)
                        for field in ("status", "value", "limit", "message")
                        if field in item
                    }
            message = "official correlation endpoints loaded"
        else:
            reader = getattr(client, "get_submission_check", None)
            if not callable(reader):
                raise RuntimeError("configured BRAIN client does not support official correlation checks")
            checks = reader(alpha_id)
            checks = checks if isinstance(checks, dict) else {}
            correlations = _correlations_from_checks(checks)
            message = "legacy submission-check correlation loaded"
        correlation_confirmed = all(
            isinstance(correlations.get(key), dict) and correlations[key].get("value") is not None
            for key in ("self", "production")
        )
        blocking_failures = [
            str(name)
            for name, data in checks.items()
            if normalize_check_name(name) not in {"selfcorrelation", "prodcorrelation", "productcorrelation"}
            and isinstance(data, dict)
            and str(data.get("status") or "").upper() == "FAIL"
        ]
        if not correlation_confirmed:
            message = "platform returned no numeric correlation"
            if blocking_failures:
                message += f"; current hard failures: {', '.join(blocking_failures[:6])}"
        if checks:
            self.store.update_candidate(int(candidate["id"]), checks_json=json.dumps(checks, sort_keys=True))
        result = {
            "success": True,
            "alpha_id": alpha_id,
            "message": message,
            "correlation_confirmed": correlation_confirmed,
            "correlations": correlations,
            "checks": checks,
            "blocking_failures": blocking_failures,
            "checked_at": utc_now(),
        }
        self.store.record_event(
            int(candidate["id"]),
            "web_manual_correlation_check",
            {
                "alpha_id": alpha_id,
                "correlation_confirmed": correlation_confirmed,
                "correlations": correlations,
            },
        )
        self.clear_status_cache()
        return result

    def submit_top_alpha(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        alpha_id, candidate = self._current_top_alpha(payload)
        with self._manual_submit_lock:
            client = self._brain_client()
            submit = client.submit_alpha(alpha_id, dry_run=False)
            result = {
                "success": bool(submit.submitted),
                "alpha_id": alpha_id,
                "submitted": bool(submit.submitted),
                "stage": str(submit.stage or "UNKNOWN"),
                "message": str(submit.message or ""),
                "submitted_at": utc_now(),
            }
            self.store.record_event(int(candidate["id"]), "web_manual_submit", result)
            if submit.submitted:
                self.store.transition(
                    int(candidate["id"]),
                    "submitted",
                    {"alpha_id": alpha_id, "source": "web_manual_submit", "stage": result["stage"]},
                )
            self.clear_status_cache()
            return result

    def _current_top_alpha(self, payload: Dict[str, Any]) -> tuple[str, Dict[str, Any]]:
        alpha_id = str(payload.get("alpha_id") or "").strip()
        if not alpha_id:
            raise ValueError("alpha_id is required")
        state = self.store.get_run_state("daemon")
        run_started_at = str(state.get("started_at") or "").strip()
        if not run_started_at:
            raise ValueError("there is no current run")
        top_ids = {row["alpha_id"] for row in self._top_alphas(limit=10, created_since=run_started_at)}
        if alpha_id not in top_ids:
            raise ValueError("alpha_id is not in the current round Top 10")
        matches = [
            row
            for row in self.store.list_candidates(created_since=run_started_at)
            if str(row.get("alpha_id") or "").strip() == alpha_id
        ]
        if not matches:
            raise ValueError("current-round candidate not found")
        return alpha_id, max(matches, key=lambda row: int(row["id"]))

    def _log_paths(self) -> Dict[str, Path]:
        return {
            "alpha": self.log_file,
            "daemon": self.daemon_stdout_log,
            "web": self.web_log,
        }

    def presets(self) -> Dict[str, Any]:
        return {"presets": [{"name": name, **scope} for name, scope in preset_rows()]}

    def scope_options(self) -> Dict[str, Any]:
        return scope_options()

    def fields(self, preset: str, limit: int, payload: Dict[str, Any] | None = None) -> Dict[str, Any]:
        cfg = load_config(db_path=str(self.db_path))
        scope = dict(cfg.simulation_context)
        selected_scope = _scope_from_payload(payload or {"preset": preset})
        if selected_scope.get("preset"):
            scope.update(SCOPE_PRESETS[_preset(selected_scope.get("preset"))])
        else:
            scope.update({key: value for key, value in selected_scope.items() if key != "preset"})
        catalog = build_field_catalog(self._brain_client(), scope)
        catalog["field_scout"] = build_field_scout(catalog)
        catalog["field_ids"] = list(catalog.get("field_ids") or [])[:limit]
        return catalog

    def _recent_candidates(self, limit: int, created_since: str = "") -> List[Dict[str, Any]]:
        rows = list(reversed(self.store.list_candidates(created_since=created_since or None)))[:limit]
        candidates: List[Dict[str, Any]] = []
        for row in rows:
            candidates.append(
                {
                    "id": row["id"],
                    "expression": row["expression"],
                    "status": row["status"],
                    "source": row["source"],
                    "alpha_id": row.get("alpha_id"),
                    "retry_count": row.get("retry_count", 0),
                    "settings": _loads_json(row.get("settings_json")),
                    "metrics": _loads_json(row.get("metrics_json")),
                    "checks": _loads_json(row.get("checks_json")),
                    "created_at": row.get("created_at"),
                    "updated_at": row.get("updated_at"),
                }
            )
        return candidates

    def _top_alphas(self, limit: int, created_since: str = "") -> List[Dict[str, Any]]:
        if not created_since:
            return []
        ranked: List[tuple[float, int, Dict[str, Any]]] = []
        for row in self.store.list_candidates(created_since=created_since):
            alpha_id = str(row.get("alpha_id") or "").strip()
            metrics = _loads_json(row.get("metrics_json"))
            if not alpha_id or not isinstance(metrics, dict):
                continue
            sharpe = _finite_float_or_none(metrics.get("sharpe"))
            fitness = _finite_float_or_none(metrics.get("fitness"))
            if sharpe is None or fitness is None:
                continue
            exact_quality_score = sharpe + 0.35 * fitness
            quality_score = round(exact_quality_score, 6)
            ranked.append(
                (
                    exact_quality_score,
                    int(row["id"]),
                    {
                        "alpha_id": alpha_id,
                        "expression": str(row.get("expression") or ""),
                        "status": str(row.get("status") or ""),
                        "source": str(row.get("source") or ""),
                        "settings": _loads_json(row.get("settings_json")),
                        "sharpe": sharpe,
                        "fitness": fitness,
                        "returns": _finite_float_or_none(metrics.get("returns")),
                        "turnover": _finite_float_or_none(metrics.get("turnover")),
                        "quality_score": quality_score,
                    },
                )
            )
        ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return [{"rank": rank, **item[2]} for rank, item in enumerate(ranked[: max(0, int(limit))], start=1)]

    def _model_scores(self, created_since: str = "") -> Dict[str, Dict[str, Any]]:
        rows = self.store.list_candidates(created_since=created_since or None)
        scores: Dict[str, Dict[str, Any]] = {}
        for row in rows:
            source = str(row.get("source") or "unknown")
            item = scores.setdefault(
                source,
                {
                    "generated": 0,
                    "approved": 0,
                    "failed": 0,
                    "submitted": 0,
                    "pending": 0,
                    "best_sharpe": None,
                    "best_fitness": None,
                },
            )
            item["generated"] += 1
            status = str(row.get("status") or "")
            if status == "approved":
                item["approved"] += 1
            elif status == "failed":
                item["failed"] += 1
            elif status == "submitted":
                item["submitted"] += 1
            elif status == "check_pending":
                item["pending"] += 1
            metrics = _loads_json(row.get("metrics_json"))
            if isinstance(metrics, dict):
                sharpe = _float_or_none(metrics.get("sharpe"))
                fitness = _float_or_none(metrics.get("fitness"))
                if sharpe is not None and (item["best_sharpe"] is None or sharpe > item["best_sharpe"]):
                    item["best_sharpe"] = sharpe
                if fitness is not None and (item["best_fitness"] is None or fitness > item["best_fitness"]):
                    item["best_fitness"] = fitness
        return scores

    def _active_research_scope(self, state: Dict[str, Any]) -> Dict[str, Any]:
        rotation_state = self.store.get_run_state("scope_rotation")
        scope = rotation_state.get("last_scope") if isinstance(rotation_state.get("last_scope"), dict) else {}
        if not scope:
            scope = state.get("scope") if isinstance(state.get("scope"), dict) else {}
        if not scope:
            recent = self._recent_candidates(limit=1)
            if recent:
                scope = recent[0].get("settings") if isinstance(recent[0].get("settings"), dict) else {}
        return scope if isinstance(scope, dict) else {}

    def _research_context(self, state: Dict[str, Any], cycle_plan: Dict[str, Any] | None = None) -> Dict[str, Any]:
        scope = self._active_research_scope(state)
        cache_key = _json_cache_key({"scope": scope or {}, "cycle_plan": cycle_plan or {}})
        cached = self._cached_research_context(cache_key)
        if cached is not None:
            return cached
        research = self._build_research_context(scope or {}, cycle_plan=cycle_plan)
        self._store_research_context_cache(cache_key, research)
        return deepcopy(research)

    def _build_research_context(self, scope: Dict[str, Any], cycle_plan: Dict[str, Any] | None = None) -> Dict[str, Any]:
        try:
            return build_ai_research_context(
                self.store,
                scope or {},
                field_catalog={"available": False, "field_ids": []},
                platform_submissions=self._recent_platform_submissions(scope or {}),
                cycle_plan=cycle_plan,
            )
        except Exception as exc:
            return {"experiment_plan": {"mode": "unavailable", "error": str(exc)}, "analysis": {}}

    def _brain_client(self) -> Any:
        cfg = load_config(db_path=str(self.db_path))
        return BrainHTTPClient.from_env() if cfg.brain_client in {"http", "brain_http", "live"} else LocalBrainClient()

    def _recent_platform_submissions(self, scope: Dict[str, Any]) -> List[Dict[str, Any]]:
        reader = getattr(self._brain_client(), "recent_submitted_alphas", None)
        if not callable(reader):
            return []
        try:
            submissions = reader(scope, limit=50)
        except Exception as exc:
            self.store.record_event(None, "platform_submission_sync_warning", {"error": str(exc), "source": "web_status"})
            return []
        return submissions if isinstance(submissions, list) else []

    def _latest_research_context_from_events(self, created_since: str = "") -> Dict[str, Any]:
        query = """
            SELECT metadata_json
            FROM events
            WHERE candidate_id IS NULL
              AND event_type = 'experiment_plan'
        """
        params: List[Any] = []
        if created_since:
            query += " AND created_at >= ?"
            params.append(created_since)
        query += " ORDER BY id DESC LIMIT 1"
        try:
            with self.store.connection() as conn:
                row = conn.execute(query, params).fetchone()
        except Exception as exc:
            return {"experiment_plan": {"mode": "unavailable", "error": str(exc)}, "analysis": {}}
        if row is None:
            return {"experiment_plan": {"mode": "warming_up"}, "analysis": {}, "candidate_queues": {}}
        plan = _loads_json(row["metadata_json"])
        if not isinstance(plan, dict):
            return {"experiment_plan": {"mode": "unavailable", "error": "latest experiment_plan is not an object"}, "analysis": {}}
        return {
            "experiment_plan": plan,
            "analysis": plan.get("analysis", {}) if isinstance(plan.get("analysis"), dict) else {},
            "candidate_queues": plan.get("candidate_queues", {}) if isinstance(plan.get("candidate_queues"), dict) else {},
        }

    def clear_status_cache(self) -> None:
        with self._status_cache_lock:
            self._status_cache = {}
        with self._research_context_cache_lock:
            self._research_context_cache = {}

    def _cached_status(self) -> Dict[str, Any] | None:
        now = time.monotonic()
        with self._status_cache_lock:
            cached = self._status_cache
            if not cached:
                return None
            if now - float(cached.get("created_at", 0.0)) > self.status_cache_ttl_seconds:
                return None
            return deepcopy(cached.get("payload", {}))

    def _store_status_cache(self, payload: Dict[str, Any]) -> None:
        with self._status_cache_lock:
            self._status_cache = {"created_at": time.monotonic(), "payload": deepcopy(payload)}

    def _cached_research_context(self, cache_key: str) -> Dict[str, Any] | None:
        now = time.monotonic()
        with self._research_context_cache_lock:
            cached = self._research_context_cache
            if not cached:
                return None
            if str(cached.get("cache_key") or "") != cache_key:
                return None
            if now - float(cached.get("created_at", 0.0)) > self.research_context_cache_ttl_seconds:
                return None
            return deepcopy(cached.get("payload", {}))

    def _store_research_context_cache(self, cache_key: str, payload: Dict[str, Any]) -> None:
        with self._research_context_cache_lock:
            self._research_context_cache = {
                "cache_key": cache_key,
                "created_at": time.monotonic(),
                "payload": deepcopy(payload),
            }


def run_web_app(
    db_path: str | Path = "alpha.db",
    env_file: str | Path = ".env",
    log_file: str | Path = "logs/alpha.log",
    host: str = "127.0.0.1",
    port: int = 8080,
) -> int:
    load_env_file(env_file)
    store = AlphaStore(db_path)
    store.init()
    service = ControlService(
        store=store,
        db_path=Path(db_path),
        env_file=Path(env_file),
        log_file=Path(log_file),
    )
    token = (os.environ.get("ALPHA_WEB_TOKEN") or "").strip()
    is_loopback = host in {"127.0.0.1", "localhost", "::1", ""}
    handler = make_handler(service, token=token)
    server = ThreadingHTTPServer((host, int(port)), handler)
    if not is_loopback:
        if token:
            print(
                f"web console bound to {host}:{port} with token auth enabled (X-Alpha-Token / ?token=).",
                flush=True,
            )
        else:
            print(
                f"WARNING: web console bound to {host}:{port} without ALPHA_WEB_TOKEN; "
                "anyone who can reach this port can control the daemon and read logs.",
                flush=True,
            )
    print(f"Alpha control panel listening on http://{host}:{port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("web stopped", flush=True)
    finally:
        server.server_close()
    return 0


def make_handler(service: ControlService, token: str = ""):
    required_token = (token or "").strip()
    MAX_BODY_BYTES = 1 << 20  # 1 MiB cap on request bodies

    class AlphaWebHandler(BaseHTTPRequestHandler):
        def _authorized(self) -> bool:
            if not required_token:
                return True
            supplied = (self.headers.get("X-Alpha-Token") or "").strip()
            if not supplied:
                query = parse_qs(urlparse(self.path).query)
                supplied = _first(query.get("token")) or ""
            # Constant-time compare to avoid leaking the token via timing.
            return hmac.compare_digest(supplied.encode("utf-8"), required_token.encode("utf-8"))

        def _same_origin(self) -> bool:
            # CSRF defense for state-changing requests: reject cross-site Origin/Referer.
            host = (self.headers.get("Host") or "").strip()
            origin = (self.headers.get("Origin") or "").strip()
            if origin:
                origin_host = urlparse(origin).netloc
                return bool(origin_host) and origin_host == host
            referer = (self.headers.get("Referer") or "").strip()
            if referer:
                return urlparse(referer).netloc == host
            # No Origin/Referer (e.g. curl, same-origin fetch): allow.
            return True

        def do_GET(self) -> None:
            if not self._authorized():
                self.send_error(401, "unauthorized")
                return
            parsed = urlparse(self.path)
            query = parse_qs(parsed.query)
            if parsed.path == "/":
                self._send_html(HTML)
            elif parsed.path == "/api/status":
                self._send_json(service.status())
            elif parsed.path == "/api/presets":
                self._send_json(service.presets())
            elif parsed.path == "/api/scope-options":
                self._send_json(service.scope_options())
            elif parsed.path == "/api/logs":
                lines = _bounded_int(_first(query.get("lines")), default=300, minimum=20, maximum=3000)
                log_name = _first(query.get("file")) or "alpha"
                path = service._log_paths().get(log_name, service.log_file)
                self._send_json({"file": log_name, "logs": tail_file(path, line_count=lines)})
            elif parsed.path == "/api/fields":
                preset = _first(query.get("preset")) or "ind"
                payload = {
                    "preset": preset,
                    "region": _first(query.get("region")),
                    "universe": _first(query.get("universe")),
                    "delay": _first(query.get("delay")),
                    "neutralization": _first(query.get("neutralization")),
                }
                limit = _bounded_int(_first(query.get("limit")), default=40, minimum=1, maximum=300)
                self._send_json(service.fields(preset, limit, payload=payload))
            else:
                self.send_error(404)

        def do_POST(self) -> None:
            if not self._authorized():
                self.send_error(401, "unauthorized")
                return
            if not self._same_origin():
                self.send_error(403, "cross-origin request blocked")
                return
            parsed = urlparse(self.path)
            try:
                payload = self._read_json()
                if parsed.path == "/api/start":
                    self._send_json(service.start(payload))
                elif parsed.path == "/api/stop":
                    self._send_json(service.stop())
                elif parsed.path == "/api/top-alpha/check":
                    self._send_json(service.check_top_alpha(payload))
                elif parsed.path == "/api/top-alpha/submit":
                    self._send_json(service.submit_top_alpha(payload))
                elif parsed.path == "/api/clear-logs":
                    self._send_json(service.clear_logs(payload))
                else:
                    self.send_error(404)
            except Exception as exc:
                self._send_json({"success": False, "message": str(exc)}, status=400)

        def log_message(self, format: str, *args: Any) -> None:
            return

        def _read_json(self) -> Dict[str, Any]:
            length = int(self.headers.get("Content-Length") or "0")
            if length <= 0:
                return {}
            if length > MAX_BODY_BYTES:
                # Drain not attempted; refuse oversized bodies to bound memory.
                raise ValueError("request body too large")
            try:
                data = json.loads(self.rfile.read(length).decode("utf-8"))
            except json.JSONDecodeError:
                return {}
            return data if isinstance(data, dict) else {}

        def _send_json(self, data: Dict[str, Any], status: int = 200) -> None:
            body = json.dumps(data, sort_keys=True).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_html(self, html: str) -> None:
            body = html.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return AlphaWebHandler


def _preset(value: Any) -> str:
    preset = str(value or "ind").strip().lower()
    if preset not in SCOPE_PRESETS:
        raise ValueError(f"unknown preset: {preset}")
    return preset


def _scope_from_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    region = str(payload.get("region") or "").strip().upper()
    universe = str(payload.get("universe") or "").strip().upper()
    neutralization = str(payload.get("neutralization") or "").strip().upper()
    delay_value = payload.get("delay")
    delay = None
    if delay_value not in (None, ""):
        delay = int(delay_value)
    if region and universe and delay is not None:
        return _validate_platform_scope(region, universe, delay, neutralization or "INDUSTRY")
    preset = _preset(payload.get("preset"))
    scope = dict(SCOPE_PRESETS[preset])
    scope["preset"] = preset
    return scope


def _scope_rotation_from_payload(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = payload.get("scope_rotation")
    if not isinstance(rows, list) or not rows:
        return []
    scopes: List[Dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("scope_rotation entries must be objects")
        region = str(row.get("region") or "").strip().upper()
        universe = str(row.get("universe") or "").strip().upper()
        neutralization = str(row.get("neutralization") or "").strip().upper() or "INDUSTRY"
        delay = int(row.get("delay"))
        scopes.append(_validate_platform_scope(region, universe, delay, neutralization))
    return [{key: value for key, value in scope.items() if key != "preset"} for scope in scopes]


def _generator_mode_from_payload(payload: Dict[str, Any]) -> str:
    value = str(payload.get("generator_mode") or payload.get("ai_generator_mode") or "single").strip().lower()
    if value in {"balanced", "balanced_4_4", "dual", "4+4"}:
        return "balanced"
    return "single"


def _orchestration_mode_from_payload(payload: Dict[str, Any]) -> str:
    value = str(payload.get("orchestration_mode") or payload.get("ai_orchestration_mode") or "lean").strip().lower()
    if value in {"deep", "quality", "decision", "full"}:
        return "deep"
    return "lean"


def _auto_submit_from_payload(payload: Dict[str, Any]) -> bool:
    value = payload.get("auto_submit")
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {"1", "true", "yes", "on", "auto", "submit"}


def _normalize_daemon_state_for_status(state: Dict[str, Any], *, health: Dict[str, Any] | None = None) -> Dict[str, Any]:
    normalized = dict(state or {})
    generator_mode = str(normalized.get("generator_mode") or "").strip().lower()
    if generator_mode not in {"single", "balanced"}:
        generator_mode = _generator_mode_from_argv(normalized.get("argv"))
    if generator_mode:
        normalized["generator_mode"] = generator_mode
    orchestration_mode = str(normalized.get("orchestration_mode") or "").strip().lower()
    if orchestration_mode not in {"lean", "deep"}:
        orchestration_mode = _orchestration_mode_from_argv(normalized.get("argv"))
    if not orchestration_mode and generator_mode:
        orchestration_mode = "lean"
    if orchestration_mode:
        normalized["orchestration_mode"] = orchestration_mode
    if "auto_submit" not in normalized:
        normalized["auto_submit"] = _auto_submit_from_argv(normalized.get("argv"))
    else:
        normalized["auto_submit"] = bool(normalized.get("auto_submit"))
    if str(normalized.get("status") or "") != "running" and not str(normalized.get("stop_reason") or "").strip():
        health_reason = str(dict(health or {}).get("last_block_reason") or "").strip()
        if health_reason:
            normalized["stop_reason"] = health_reason
    return normalized


def _generator_mode_from_argv(argv: Any) -> str:
    if not isinstance(argv, list):
        return ""
    for index, item in enumerate(argv):
        if str(item) == "--generator-mode" and index + 1 < len(argv):
            value = str(argv[index + 1] or "").strip().lower()
            if value in {"single", "balanced"}:
                return value
    return ""


def _orchestration_mode_from_argv(argv: Any) -> str:
    if not isinstance(argv, list):
        return ""
    for index, item in enumerate(argv):
        if str(item) == "--orchestration-mode" and index + 1 < len(argv):
            value = str(argv[index + 1] or "").strip().lower()
            if value in {"lean", "deep"}:
                return value
    return ""


def _auto_submit_from_argv(argv: Any) -> bool:
    if not isinstance(argv, list):
        return False
    if "--auto-submit" in [str(item) for item in argv]:
        return True
    return False


def _validate_platform_scope(region: str, universe: str, delay: int, neutralization: str) -> Dict[str, Any]:
    for row in platform_scope_rows():
        if row["region"] != region or int(row["delay"]) != int(delay):
            continue
        if universe not in row["universes"]:
            raise ValueError(f"invalid universe for {region} D{delay}: {universe}")
        if neutralization not in row["neutralizations"]:
            raise ValueError(f"invalid neutralization for {region} D{delay}: {neutralization}")
        return {
            "preset": "",
            "region": region,
            "universe": universe,
            "delay": int(delay),
            "neutralization": neutralization,
        }
    raise ValueError(f"invalid region/delay combination: {region} D{delay}")


def _bounded_int(value: Any, default: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return min(max(parsed, minimum), maximum)


def _submit_cap_from_env() -> int:
    return _bounded_int(os.getenv("MAX_FINAL_SUBMITS_PER_ROUND"), default=4, minimum=0, maximum=100)


def _bounded_float(value: Any, default: float, minimum: float, maximum: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = default
    return min(max(parsed, minimum), maximum)


def _format_number(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else str(value)


def _stop_after_at(run_minutes: float) -> str:
    if run_minutes <= 0:
        return ""
    return (datetime.now(timezone.utc).replace(microsecond=0) + timedelta(minutes=run_minutes)).isoformat()


def _prepend_path(existing: str, path: Path) -> str:
    items = [str(path)]
    if existing:
        items.append(existing)
    return os.pathsep.join(items)


def _first(values: List[str] | None) -> str:
    return values[0] if values else ""


def _loads_json(value: Any) -> Any:
    if not value:
        return {}
    try:
        return json.loads(str(value))
    except json.JSONDecodeError:
        return {}


def _json_cache_key(value: Any) -> str:
    try:
        return json.dumps(value if value is not None else {}, sort_keys=True, separators=(",", ":"))
    except TypeError:
        return json.dumps(str(value), sort_keys=True, separators=(",", ":"))


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _finite_float_or_none(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    number = _float_or_none(value)
    return number if number is not None and math.isfinite(number) else None


def _compact_candidate_queues_for_status(queues: Any) -> Dict[str, Any]:
    if not isinstance(queues, dict):
        return {}
    compact: Dict[str, Any] = {}
    for name, value in queues.items():
        if name == "counts":
            compact[name] = value if isinstance(value, dict) else {}
            continue
        if not isinstance(value, list):
            continue
        compact[name] = [_compact_queue_item_for_status(item) for item in value[:STATUS_QUEUE_EXAMPLE_LIMIT] if isinstance(item, dict)]
    return compact


def _compact_queue_item_for_status(item: Dict[str, Any]) -> Dict[str, Any]:
    return {key: item[key] for key in STATUS_QUEUE_ITEM_KEYS if key in item}


def _correlations_from_checks(checks: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    correlations: Dict[str, Dict[str, Any]] = {}
    for name, data in checks.items():
        normalized = normalize_check_name(name)
        if normalized == "selfcorrelation":
            key = "self"
        elif normalized in {"prodcorrelation", "productcorrelation"}:
            key = "production"
        else:
            continue
        item = data if isinstance(data, dict) else {"status": str(data)}
        correlations[key] = {
            field: item.get(field)
            for field in ("status", "value", "limit", "message")
            if field in item
        }
        correlations[key]["name"] = str(name)
    return correlations


HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Alpha 控制台</title>
<style>
:root{
  --bg:#16171a; --bg-2:#1c1d21; --panel:#1f2024; --panel-2:#26272c;
  --line:#2e3036; --line-2:#3a3d45;
  --ink:#e8e9ec; --ink-dim:#a4a7b0; --ink-mute:#6f7480;
  --accent:#e8a849; --accent-dim:#8a6a2e;
  --ok:#5bbf8f; --ok-dim:#2f5d48; --bad:#e06a5e; --bad-dim:#5d322e; --warn:#e0b24a;
  --mono:"JetBrains Mono",ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
  --sans:-apple-system,BlinkMacSystemFont,"Segoe UI",Inter,Arial,sans-serif;
}
*{box-sizing:border-box;}
body{margin:0;background:var(--bg);color:var(--ink);font-family:var(--sans);
  font-size:13px;line-height:1.5;-webkit-font-smoothing:antialiased;}
.mono{font-family:var(--mono);font-variant-numeric:tabular-nums;}

/* ── top status rail ───────────────────────── */
.rail{position:sticky;top:0;z-index:30;display:flex;align-items:center;gap:18px;
  padding:11px 20px;background:linear-gradient(180deg,#1b1c20,#16171a);
  border-bottom:1px solid var(--line);}
.rail .brand{font-weight:700;letter-spacing:.02em;font-size:13px;color:var(--ink);}
.rail .brand b{color:var(--accent);font-weight:800;}
.run-dot{display:inline-flex;align-items:center;gap:8px;font-family:var(--mono);
  font-size:12px;font-weight:600;padding:4px 11px;border-radius:999px;
  border:1px solid var(--line-2);background:var(--bg-2);}
.run-dot .led{width:8px;height:8px;border-radius:50%;background:var(--ink-mute);}
.run-dot.on{color:var(--ok);border-color:var(--ok-dim);} .run-dot.on .led{background:var(--ok);box-shadow:0 0 8px var(--ok);}
.run-dot.off{color:var(--bad);border-color:var(--bad-dim);} .run-dot.off .led{background:var(--bad);}
.rail .stat{font-family:var(--mono);font-size:12px;color:var(--ink-dim);}
.rail .stat b{color:var(--ink);font-weight:600;}
.rail .spacer{margin-left:auto;}
.rail .updated{font-family:var(--mono);font-size:11px;color:var(--ink-mute);}
.menu-btn{display:none;}

/* ── layout shell ──────────────────────────── */
.shell{display:grid;grid-template-columns:300px minmax(0,1fr);min-height:calc(100vh - 45px);}
.side{position:sticky;top:45px;align-self:start;height:calc(100vh - 45px);overflow:auto;
  background:var(--bg-2);border-right:1px solid var(--line);padding:16px 14px;}
.work{min-width:0;padding:18px 20px 40px;}

/* ── side blocks ───────────────────────────── */
.blk{border-top:1px solid var(--line);padding:15px 0 4px;}
.blk:first-child{border-top:0;padding-top:2px;}
.blk-h{display:flex;align-items:center;gap:8px;font-family:var(--mono);font-size:11px;
  letter-spacing:.12em;text-transform:uppercase;color:var(--ink-mute);margin-bottom:11px;}
.blk-h::before{content:"";width:5px;height:5px;border-radius:1px;background:var(--accent);}
label{display:block;margin:9px 0 4px;color:var(--ink-mute);font-size:11px;
  letter-spacing:.04em;text-transform:uppercase;font-family:var(--mono);}
input,select{width:100%;height:34px;padding:0 9px;border:1px solid var(--line-2);
  border-radius:5px;background:var(--panel);color:var(--ink);font-family:var(--mono);
  font-size:12px;outline:none;transition:border-color .12s,box-shadow .12s;}
input:focus,select:focus{border-color:var(--accent-dim);box-shadow:0 0 0 2px rgba(232,168,73,.14);}
select[multiple]{height:auto;padding:6px;}
.inline-check{display:flex;align-items:center;gap:8px;margin:11px 0 4px;
  color:var(--ink-dim);font-size:12px;text-transform:none;letter-spacing:0;font-family:var(--sans);}
.inline-check input{width:auto;height:auto;}
.inline-check.submit-switch{justify-content:space-between;margin-top:13px;padding:8px 10px;
  border:1px solid var(--line-2);border-radius:7px;background:var(--panel-2);color:var(--ink);}
.seg{display:grid;grid-template-columns:1fr 1fr;gap:3px;padding:3px;border:1px solid var(--line-2);
  border-radius:6px;background:var(--panel);}
.seg label{margin:0;text-transform:none;letter-spacing:0;}
.seg input{position:absolute;opacity:0;pointer-events:none;}
.seg span{display:flex;align-items:center;justify-content:center;min-height:28px;border-radius:4px;
  font-family:var(--mono);font-size:12px;color:var(--ink-dim);cursor:pointer;transition:.12s;}
.seg input:checked+span{background:var(--accent);color:#16171a;font-weight:700;}
.acts{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-top:14px;}
button{height:36px;border-radius:6px;border:1px solid var(--line-2);background:var(--panel-2);
  color:var(--ink);font-family:var(--mono);font-size:12px;font-weight:600;cursor:pointer;transition:.12s;}
button:hover{border-color:var(--line-2);background:#2c2e34;}
button.primary{background:var(--accent);border-color:var(--accent);color:#16171a;font-weight:700;}
button.primary:hover{background:#f0b65e;}
button.danger{background:transparent;border-color:var(--bad-dim);color:var(--bad);}
button.danger:hover{background:rgba(224,106,94,.12);}
.msg{min-height:18px;margin-top:10px;font-family:var(--mono);font-size:11px;color:var(--accent);}
.kv{display:flex;justify-content:space-between;gap:10px;padding:4px 0;font-size:12px;}
.kv .k{color:var(--ink-mute);font-family:var(--mono);font-size:11px;}
.kv .v{color:var(--ink-dim);font-family:var(--mono);text-align:right;overflow-wrap:anywhere;}
/* ── workspace panels ──────────────────────── */
.panel{background:var(--panel);border:1px solid var(--line);border-radius:9px;
  padding:15px 16px;margin-bottom:14px;}
.panel-h{display:flex;align-items:center;gap:9px;margin-bottom:13px;}
.panel-h h2{margin:0;font-size:12px;letter-spacing:.1em;text-transform:uppercase;
  color:var(--ink-dim);font-family:var(--mono);font-weight:600;}
.panel-h .tag{margin-left:auto;font-family:var(--mono);font-size:11px;color:var(--ink-mute);}
.panel-h .bar{width:5px;height:13px;border-radius:1px;background:var(--accent);}

/* hero metrics — this round */
.metrics{display:grid;grid-template-columns:repeat(4,1fr) 1.3fr;gap:1px;
  background:var(--line);border-radius:8px;overflow:hidden;}
.metric{background:var(--panel-2);padding:13px 14px;}
.metric .ml{font-family:var(--mono);font-size:10px;letter-spacing:.1em;text-transform:uppercase;color:var(--ink-mute);}
.metric .mv{margin-top:6px;font-family:var(--mono);font-size:26px;font-weight:700;line-height:1;letter-spacing:-.01em;}
.metric.approved .mv{color:var(--ok);} .metric.failed .mv{color:var(--bad);} .metric.pending .mv{color:var(--warn);}
.metric .ms{margin-top:5px;font-family:var(--mono);font-size:11px;color:var(--ink-mute);overflow-wrap:anywhere;}
.metric.span2{display:flex;flex-direction:column;justify-content:center;}

/* progress bar */
.prog{height:7px;border-radius:4px;background:var(--bg);overflow:hidden;margin-top:7px;border:1px solid var(--line);}
.prog>i{display:block;height:100%;background:linear-gradient(90deg,var(--accent-dim),var(--accent));transition:width .4s;}

/* sub note rows */
.note{font-family:var(--mono);font-size:11px;color:var(--ink-mute);margin-top:11px;}
.note b{color:var(--ink-dim);font-weight:600;}

/* queue chips */
.qrow{display:flex;flex-wrap:wrap;gap:8px;}
.chip{display:flex;align-items:baseline;gap:7px;padding:7px 12px;border:1px solid var(--line-2);
  border-radius:7px;background:var(--panel-2);font-family:var(--mono);}
.chip .cl{font-size:10px;letter-spacing:.08em;text-transform:uppercase;color:var(--ink-mute);}
.chip .cv{font-size:17px;font-weight:700;color:var(--ink);}
.chip.hot .cv{color:var(--accent);}
.chip.ready{border-color:var(--ok-dim);background:rgba(93,185,134,.08);}
.chip.ready .cv{color:var(--ok);}

/* readable metric rows (replaces raw JSON) */
.rows{display:grid;gap:9px;}
.mrow{display:grid;grid-template-columns:150px minmax(0,1fr) 64px;align-items:center;gap:12px;}
.mrow.plain{grid-template-columns:150px minmax(0,1fr);}
.mrow .rl{font-family:var(--mono);font-size:11px;color:var(--ink-dim);}
.mrow .rv{min-width:0;font-family:var(--mono);font-size:12px;font-weight:600;color:var(--ink);text-align:right;overflow-wrap:anywhere;}
.mrow.plain .rv{text-align:left;}
.track{min-width:0;height:6px;border-radius:3px;background:var(--bg);overflow:hidden;border:1px solid var(--line);}
.track>i{display:block;height:100%;background:var(--ok);}
.track.amber>i{background:var(--accent);}

/* tables */
.tbl-wrap{overflow-x:auto;}
table{width:100%;border-collapse:collapse;font-size:12px;font-family:var(--mono);}
th,td{padding:8px 9px;text-align:left;vertical-align:top;border-bottom:1px solid var(--line);}
th{font-size:10px;letter-spacing:.08em;text-transform:uppercase;color:var(--ink-mute);font-weight:600;
  position:sticky;top:0;background:var(--panel);}
tbody tr:hover{background:var(--panel-2);}
td code{font-size:11px;color:var(--ink-dim);overflow-wrap:anywhere;}
.top-alpha-table{min-width:980px;font-variant-numeric:tabular-nums;}
.top-alpha-table .top-rank{width:44px;color:var(--ink-mute);font-weight:700;}
.top-alpha-table .platform-alpha-id code{color:var(--ink);font-size:12px;font-weight:700;white-space:nowrap;}
.top-alpha-table .top-score{color:var(--accent);font-weight:700;}
.top-alpha-table tbody tr.top-alpha-row{cursor:pointer;transition:background .15s ease;}
.top-alpha-table tbody tr.top-alpha-row:focus-visible{outline:2px solid var(--accent);outline-offset:-2px;}
.top-alpha-table tbody tr.top-alpha-row.selected{background:var(--panel-2);}
.top-alpha-table tbody tr:first-child .top-rank,
.top-alpha-table tbody tr:first-child .platform-alpha-id code{color:var(--ok);}
.top-alpha-table tbody tr.top-alpha-expression:hover{background:transparent;}
.top-alpha-expression td{padding:14px 9px;border-bottom:1px solid var(--line);}
.top-alpha-expression-h{display:flex;align-items:center;gap:10px;margin-bottom:8px;
  font-family:var(--mono);font-size:10px;letter-spacing:.08em;text-transform:uppercase;color:var(--ink-mute);}
.top-alpha-expression-h code{font-size:11px;letter-spacing:0;text-transform:none;color:var(--accent);overflow-wrap:anywhere;}
.top-alpha-expression pre{margin:0;padding:11px 12px;border:1px solid var(--line);border-radius:6px;
  background:var(--panel-2);white-space:pre-wrap;overflow-wrap:anywhere;word-break:break-word;}
.top-alpha-expression pre code{font:12px/1.6 var(--mono);color:var(--ink);}
.top-actions{white-space:nowrap;}
.top-action-set{display:flex;align-items:center;gap:6px;}
.top-action{height:28px;padding:0 9px;border-radius:5px;font-size:10px;letter-spacing:.03em;}
.top-action.check{border-color:var(--accent-dim);color:var(--accent);background:transparent;}
.top-action.submit{border-color:var(--ok-dim);color:var(--ok);background:transparent;}
.top-action:disabled{cursor:wait;opacity:.45;}
.top-action-result{margin-top:10px;border-top:1px solid var(--line);padding-top:10px;}
.top-action-result-h{display:flex;align-items:center;justify-content:space-between;gap:12px;
  color:var(--ink-dim);font:11px/1.4 var(--mono);}
.top-action-result-h b{color:var(--ink);font-weight:700;}
.correlation-grid{display:grid;grid-template-columns:1fr 1fr;gap:1px;margin-top:8px;
  border:1px solid var(--line);background:var(--line);}
.correlation-item{display:grid;grid-template-columns:minmax(82px,auto) 1fr auto;align-items:center;gap:10px;
  padding:9px 10px;background:var(--panel-2);font:11px/1.4 var(--mono);}
.correlation-item .name{color:var(--ink-mute);text-transform:uppercase;}
.correlation-item .value{color:var(--ink);font-size:14px;font-weight:700;}
.correlation-item .state{color:var(--ink-dim);text-align:right;}
.correlation-item.confirmed .value{color:var(--ok);}
.correlation-item.unconfirmed .value{color:var(--warn);}
.official-checks{display:flex;flex-wrap:wrap;gap:5px;margin-top:8px;}
.official-check{padding:3px 7px;border:1px solid var(--line);border-radius:4px;
  background:var(--panel-2);color:var(--ink-dim);font:10px/1.4 var(--mono);}
.official-check b{color:var(--ink);font-weight:600;}
.pill{display:inline-block;padding:2px 8px;border-radius:999px;font-size:10px;font-weight:700;
  letter-spacing:.04em;text-transform:uppercase;}
.pill.approved,.pill.submitted{background:var(--ok-dim);color:var(--ok);}
.pill.failed{background:var(--bad-dim);color:var(--bad);}
.pill.check_pending,.pill.pending{background:rgba(224,178,74,.16);color:var(--warn);}
.pill.preflight_passed,.pill.generated{background:var(--panel-2);color:var(--ink-dim);}
.qtag{font-size:10px;letter-spacing:.06em;text-transform:uppercase;color:var(--accent);font-weight:700;}
.ready-tag{display:inline-block;margin-left:6px;padding:2px 8px;border-radius:999px;
  background:var(--ok-dim);color:var(--ok);font-family:var(--mono);font-size:10px;font-weight:700;
  letter-spacing:.04em;text-transform:uppercase;white-space:nowrap;}

/* field pool chips */
.fp-meta{font-family:var(--mono);font-size:11px;color:var(--ink-mute);margin-bottom:10px;}
.fp-meta b{color:var(--ink-dim);}
.fp-group{margin-top:10px;}
.fp-group .gl{font-family:var(--mono);font-size:10px;letter-spacing:.08em;text-transform:uppercase;color:var(--accent);margin-bottom:6px;}
.tags{display:flex;flex-wrap:wrap;gap:5px;}
.ftag{font-family:var(--mono);font-size:11px;padding:3px 8px;border-radius:5px;
  background:var(--panel-2);border:1px solid var(--line);color:var(--ink-dim);}

/* logs */
.toolbar{display:flex;align-items:center;gap:8px;flex-wrap:wrap;margin-bottom:11px;}
.toolbar h2{margin-right:auto;}
.toolbar select,.toolbar input{width:auto;}
.toolbar input{width:88px;}
.toolbar button{height:30px;padding:0 12px;}
.logs{height:380px;overflow:auto;white-space:pre-wrap;background:#0f1013;color:#c7ccd6;
  border:1px solid var(--line);border-radius:7px;padding:12px;
  font:12px/1.55 var(--mono);}

.empty{font-family:var(--mono);font-size:11px;color:var(--ink-mute);padding:8px 0;}
.backdrop{display:none;}

/* ── responsive ───────────────────────────── */
@media(max-width:1080px){
  .shell{grid-template-columns:1fr;}
  .side{position:fixed;inset:45px auto 0 0;z-index:25;width:min(86vw,320px);
    transform:translateX(-100%);transition:transform .18s ease;}
  .side.open{transform:translateX(0);}
  .backdrop{display:none;position:fixed;inset:45px 0 0;z-index:20;background:rgba(0,0,0,.5);}
  body.side-open .backdrop{display:block;}
  .menu-btn{display:inline-flex;align-items:center;justify-content:center;height:30px;
    padding:0 12px;border-radius:6px;border:1px solid var(--line-2);background:var(--panel);
    color:var(--ink);font-family:var(--mono);font-size:12px;cursor:pointer;}
  .metrics{grid-template-columns:repeat(2,1fr);}
  .metric.span2{grid-column:1 / -1;}
}
@media(max-width:560px){
  .rail{gap:10px;padding:9px 12px;}
  .rail .stat{display:none;}
  .rail .updated{display:none;}
  .work{padding:14px 12px 36px;}
  .metric .mv{font-size:22px;}
  .mrow{grid-template-columns:110px minmax(0,1fr) 52px;gap:8px;}
  .mrow.plain{grid-template-columns:110px minmax(0,1fr);}
  #top_alphas_panel .tbl-wrap{overflow:visible;}
  .top-alpha-table{min-width:0;}
  .top-alpha-table thead{display:none;}
  .top-alpha-table tbody{display:block;}
  .top-alpha-table tbody tr{display:grid;grid-template-columns:repeat(12,minmax(0,1fr));padding:11px 0;}
  .top-alpha-table tbody tr:hover{background:transparent;}
  .top-alpha-table tbody tr.top-alpha-row.selected{background:var(--panel-2);}
  .top-alpha-table tbody td{display:flex;flex-direction:column;gap:3px;min-width:0;padding:5px 6px;border:0;}
  .top-alpha-table tbody td::before{content:attr(data-label);font-size:9px;letter-spacing:.08em;
    text-transform:uppercase;color:var(--ink-mute);font-weight:600;}
  .top-alpha-table .top-rank{grid-column:span 2;width:auto;}
  .top-alpha-table .platform-alpha-id{grid-column:span 10;}
  .top-alpha-table .platform-alpha-id code{white-space:normal;overflow-wrap:anywhere;}
  .top-alpha-table .top-status,.top-alpha-table .top-score{grid-column:span 6;}
  .top-alpha-table .top-metric{grid-column:span 3;}
  .top-alpha-table .top-scope{grid-column:1 / -1;}
  .top-alpha-table .top-actions{grid-column:1 / -1;}
  .top-alpha-table .top-action-set{display:grid;grid-template-columns:1fr 1fr;gap:8px;}
  .top-alpha-table .top-action{height:34px;width:100%;}
  .top-alpha-table .empty{grid-column:1 / -1;display:block;padding:12px 6px;}
  .top-alpha-table .empty::before{content:none;}
  .top-alpha-table tbody tr.top-alpha-expression{display:block;padding:0 0 12px;}
  .top-alpha-table tbody tr.top-alpha-expression td{display:block;padding:10px 6px 2px;border:0;}
  .top-alpha-table tbody tr.top-alpha-expression td::before{content:none;}
  .correlation-grid{grid-template-columns:1fr;}
  .top-action-result-h{align-items:flex-start;flex-direction:column;gap:3px;}
}
</style>
</head>
<body>
<div class="rail">
  <button class="menu-btn" onclick="openSidebar()">≡ 菜单</button>
  <div class="brand">ALPHA<b>·</b>CONSOLE</div>
  <div id="running" class="run-dot off"><span class="led"></span><span>已停止</span></div>
  <div class="stat">pid <b id="pid">--</b></div>
  <div class="stat keep">auto-stop <b id="stop_after_at">--</b></div>
  <div class="stat">gen <b id="generator_mode_status">--</b></div>
  <div class="stat">mode <b id="orchestration_mode_status">--</b></div>
  <div class="stat">submit <b id="auto_submit_status">手动</b></div>
  <div class="spacer"></div>
  <div class="updated" id="updated">--</div>
</div>
<div class="backdrop" onclick="closeSidebar()"></div>
<div class="shell">
  <aside id="sidebar" class="side">
    <div class="blk">
      <div class="blk-h">控制 · control</div>
      <label>Quick preset</label>
      <select id="preset"></select>
      <label>Region</label>
      <select id="region"></select>
      <label>Delay</label>
      <select id="delay"></select>
      <label>Universe</label>
      <select id="universe"></select>
      <label>Neutralization</label>
      <select id="neutralization"></select>
      <label class="inline-check"><input id="rotate_scopes" type="checkbox"> 轮换多个 scope</label>
      <select id="scope_rotation" multiple size="6"></select>
      <label>Batch size</label>
      <input id="batch_size" type="number" value="8" min="1" max="8">
      <label>生成模式</label>
      <div class="seg" role="radiogroup">
        <label><input type="radio" name="generator_mode" value="single" checked><span>单模型</span></label>
        <label><input type="radio" name="generator_mode" value="balanced"><span>4+4</span></label>
      </div>
      <label>决策模式</label>
      <div class="seg" role="radiogroup">
        <label><input type="radio" name="orchestration_mode" value="lean" checked><span>效率</span></label>
        <label><input type="radio" name="orchestration_mode" value="deep"><span>质量决策</span></label>
      </div>
      <label class="inline-check submit-switch"><span>自动提交达标 alpha</span><input id="auto_submit" type="checkbox"></label>
      <label>Loop seconds</label>
      <input id="loop_seconds" type="number" value="60" min="1" max="3600">
      <label>Run limit</label>
      <select id="run_minutes">
        <option value="120">2 小时</option>
        <option value="240" selected>4 小时</option>
        <option value="480">8 小时</option>
        <option value="custom">自定义</option>
      </select>
      <label>Custom minutes</label>
      <input id="run_minutes_custom" type="number" value="240" min="1" max="1440" disabled>
      <div class="acts">
        <button class="primary" onclick="startDaemon()">▶ 启动</button>
        <button class="danger" onclick="stopDaemon()">■ 停止本轮</button>
      </div>
      <div id="message" class="msg"></div>
    </div>
    <div class="blk">
      <div class="blk-h">字段池 · fields</div>
      <button style="width:100%" onclick="loadFields()">刷新当前 scope 字段</button>
      <div id="fields" class="fp-wrap" style="margin-top:11px"><div class="empty">--</div></div>
    </div>
    <div class="blk">
      <div class="blk-h">进程 · process</div>
      <div class="kv"><span class="k">PID</span><span class="v" id="daemon_pid_side">--</span></div>
      <div class="kv"><span class="k">Scope</span><span class="v" id="daemon_scope">--</span></div>
      <div class="kv"><span class="k">AI Mode</span><span class="v" id="daemon_ai_mode">--</span></div>
      <div class="kv"><span class="k">Submit</span><span class="v" id="daemon_submit_mode">手动</span></div>
      <div class="kv"><span class="k">Started</span><span class="v" id="started_at">--</span></div>
    </div>
    <div class="blk">
      <div class="blk-h">实验计划 · plan</div>
      <div class="kv"><span class="k">Mode</span><span class="v" id="plan_mode">--</span></div>
      <div class="kv"><span class="k">Target</span><span class="v" id="plan_target">--</span></div>
      <div class="kv"><span class="k">Keep</span><span class="v" id="plan_keep">--</span></div>
      <div class="kv"><span class="k">Avoid</span><span class="v" id="plan_avoid">--</span></div>
    </div>
  </aside>
  <main class="work">
    <div class="panel" id="status_panel">
      <div class="panel-h"><span class="bar"></span><h2>本轮状态 · this round</h2>
        <span class="tag" id="run_window">本轮起点：--</span></div>
      <div class="metrics">
        <div class="metric approved"><div class="ml">Approved</div><div class="mv" id="approved">0</div></div>
        <div class="metric pending"><div class="ml">Pending</div><div class="mv" id="pending">0</div></div>
        <div class="metric failed"><div class="ml">Failed</div><div class="mv" id="failed">0</div></div>
        <div class="metric"><div class="ml">PID</div><div class="mv" id="pid_summary" style="font-size:20px">--</div></div>
        <div class="metric span2"><div class="ml">Submitted · 本轮提交</div>
          <div class="mv" id="submitted">0</div>
          <div class="prog"><i id="submit_prog" style="width:0%"></i></div>
          <div class="ms" id="submit_cap">日上限 4</div></div>
      </div>
      <div class="note" id="history_counts">历史累计：--</div>
    </div>

    <div class="panel" id="top_alphas_panel">
      <div class="panel-h"><span class="bar"></span><h2>本轮 Top 10 · top alphas</h2>
        <span class="tag" id="top_alpha_count">0 / 10</span></div>
      <div class="tbl-wrap">
        <table class="top-alpha-table">
          <thead><tr><th>#</th><th>Platform ID</th><th>Status</th><th>Score</th><th>Sharpe</th><th>Fitness</th><th>Returns</th><th>Turnover</th><th>Scope</th><th>Actions</th></tr></thead>
          <tbody id="top_alphas"><tr><td colspan="10" class="empty">本轮暂无可排名的 Alpha 指标</td></tr></tbody>
        </table>
      </div>
    </div>

    <div class="panel" id="queue_panel">
      <div class="panel-h"><span class="bar"></span><h2>候选队列 · queues</h2></div>
      <div class="qrow" id="candidate_queues">
        <div class="chip hot"><span class="cv" id="queue_submitable">0</span><span class="cl">submitable</span></div>
        <div class="chip"><span class="cv" id="queue_watchlist">0</span><span class="cl">watchlist</span></div>
        <div class="chip"><span class="cv" id="queue_optimize">0</span><span class="cl">optimize</span></div>
        <div class="chip"><span class="cv" id="queue_trash">0</span><span class="cl">trash</span></div>
        <div class="chip"><span class="cv" id="queue_abandoned">0</span><span class="cl">abandoned</span></div>
      </div>
      <div class="tbl-wrap" style="margin-top:13px">
        <table>
          <thead><tr><th>Queue</th><th>ID</th><th>Score</th><th>Reason</th><th>Expression</th></tr></thead>
          <tbody id="queue_examples"><tr><td colspan="5" class="empty">暂无数据</td></tr></tbody>
        </table>
      </div>
    </div>

    <div class="panel" id="efficiency_panel">
      <div class="panel-h"><span class="bar"></span><h2>效率 · efficiency</h2></div>
      <div class="rows" id="efficiency_metrics"><div class="empty">--</div></div>
    </div>

    <div class="panel" id="scheduler_panel">
      <div class="panel-h"><span class="bar"></span><h2>调度计划 · scheduler</h2></div>
      <div class="rows" id="scheduler_plan"><div class="empty">--</div></div>
    </div>

    <div class="panel" id="candidates_panel">
      <div class="panel-h"><span class="bar"></span><h2>本轮候选 · candidates</h2></div>
      <div class="tbl-wrap">
        <table>
          <thead><tr><th>ID</th><th>Status</th><th>Model</th><th>Scope</th><th>Metrics</th><th>Expression</th></tr></thead>
          <tbody id="candidates"><tr><td colspan="6" class="empty">暂无数据</td></tr></tbody>
        </table>
      </div>
    </div>

    <div class="panel" id="models_panel">
      <div class="panel-h"><span class="bar"></span><h2>模型表现 · models</h2></div>
      <div class="tbl-wrap">
        <table>
          <thead><tr><th>Model</th><th>Gen</th><th>Appr</th><th>Fail</th><th>Best</th></tr></thead>
          <tbody id="model_scores"><tr><td colspan="5" class="empty">暂无数据</td></tr></tbody>
        </table>
      </div>
    </div>

    <div class="panel" id="logs_panel">
      <div class="panel-h"><span class="bar"></span><h2>日志 · logs</h2>
        <div class="toolbar" style="margin:0 0 0 auto">
          <select id="log_file"><option value="alpha">alpha.log</option><option value="daemon">daemon stdout</option><option value="web">web.log</option></select>
          <input id="lines" type="number" value="300">
          <button onclick="loadLogs()">刷新</button>
          <button class="danger" onclick="clearLogs(false)">清空当前</button>
          <button class="danger" onclick="clearLogs(true)">清空全部</button>
        </div>
      </div>
      <div id="logs" class="logs">loading...</div>
    </div>
  </main>
</div>
<script>
    const $ = (id) => document.getElementById(id);
    let presetRows = [];
    let scopeRows = [];
    let dailyCap = 4;

    function openSidebar() {
      $('sidebar').classList.add('open');
      document.body.classList.add('side-open');
    }
    function closeSidebar() {
      $('sidebar').classList.remove('open');
      document.body.classList.remove('side-open');
    }
    function msg(text) { $('message').textContent = text || ''; }
    function authToken() {
      try {
        const fromUrl = new URLSearchParams(window.location.search).get('token');
        if (fromUrl) { window.localStorage.setItem('alphaWebToken', fromUrl); return fromUrl; }
        return window.localStorage.getItem('alphaWebToken') || '';
      } catch (e) { return ''; }
    }
    async function api(path, options={}) {
      const token = authToken();
      const headers = {'Content-Type':'application/json', ...(options.headers || {})};
      if (token) headers['X-Alpha-Token'] = token;
      const response = await fetch(path, {...options, headers});
      if (response.status === 401) msg('未授权：请用带 ?token=... 的链接打开控制台');
      return await response.json();
    }
    function escapeHtml(value) {
      return String(value || '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
    }
    function formatCounts(counts) {
      const order = ['approved', 'check_pending', 'failed', 'submitted', 'preflight_passed', 'generated'];
      const parts = [];
      for (const key of order) if (counts[key]) parts.push(`${key} ${counts[key]}`);
      for (const key of Object.keys(counts).sort()) if (!order.includes(key) && counts[key]) parts.push(`${key} ${counts[key]}`);
      return parts.join('  ·  ') || '0';
    }
    function generatorModeLabel(mode) {
      if (mode === 'balanced') return '4+4 双模型';
      if (mode === 'single') return '单模型';
      return '未知';
    }
    function orchestrationModeLabel(mode) {
      if (mode === 'deep') return '质量决策';
      if (mode === 'lean') return '效率';
      return '未知';
    }
    function submitModeLabel(autoSubmit) {
      return autoSubmit ? '自动' : '手动';
    }
    function num(value) {
      if (value === null || value === undefined || value === '') return null;
      const parsed = Number(value);
      return Number.isFinite(parsed) ? parsed : null;
    }
    function fmtVal(value) {
      if (value === null || value === undefined || value === '') return '--';
      if (typeof value === 'number') return Number.isInteger(value) ? String(value) : value.toFixed(3);
      if (typeof value === 'boolean') return value ? 'yes' : 'no';
      if (Array.isArray(value)) return value.length ? value.slice(0, 6).join(', ') : '[]';
      if (typeof value === 'object') return JSON.stringify(value);
      return String(value);
    }
    function fmtMetric(value, digits=3) {
      const parsed = num(value);
      return parsed === null ? '--' : parsed.toFixed(digits);
    }
    function fmtPercent(value) {
      const parsed = num(value);
      return parsed === null ? '--' : `${(parsed * 100).toFixed(2)}%`;
    }
    function renderRows(el, obj) {
      el.innerHTML = '';
      const entries = Object.entries(obj || {});
      if (!entries.length) { el.innerHTML = '<div class="empty">暂无数据</div>'; return; }
      for (const [key, value] of entries) {
        const row = document.createElement('div');
        const n = num(value);
        const isProgress = typeof value === 'number' && n !== null && n >= 0 && n <= 1.0001;
        if (isProgress) {
          row.className = 'mrow';
          const pct = Math.min(100, Math.max(0, n * 100));
          const amber = n < 0.5 ? ' amber' : '';
          row.innerHTML = `<div class="rl">${escapeHtml(key)}</div><div class="track${amber}"><i style="width:${pct}%"></i></div><div class="rv">${escapeHtml(fmtVal(value))}</div>`;
        } else {
          row.className = 'mrow plain';
          row.innerHTML = `<div class="rl">${escapeHtml(key)}</div><div class="rv">${escapeHtml(fmtVal(value))}</div>`;
        }
        el.appendChild(row);
      }
    }
    function renderFields(data) {
      const ids = data.field_ids || [];
      const scout = data.field_scout || {};
      const top = (scout.top_fields || []).slice(0, 12);
      const buckets = (scout.buckets || []).slice(0, 5);
      let html = `<div class="fp-meta">available <b>${escapeHtml(data.available)}</b> · source <b>${escapeHtml(data.source || 'unknown')}</b> · <b>${ids.length}</b> fields</div>`;
      if (top.length) {
        html += `<div class="fp-group"><div class="gl">top scout</div><div class="tags">` +
          top.map(f => `<span class="ftag" title="${escapeHtml((f.category || '') + ' ' + (f.primary_policy || ''))}">${escapeHtml(f.field)} · ${Number(f.score || 0).toFixed(2)}</span>`).join('') +
          `</div></div>`;
      }
      for (const bucket of buckets) {
        html += `<div class="fp-group"><div class="gl">${escapeHtml(bucket.name)}</div><div class="tags">` +
          (bucket.fields || []).slice(0, 10).map(field => `<span class="ftag">${escapeHtml(field)}</span>`).join('') +
          `</div></div>`;
      }
      if (ids.length) {
        html += `<div class="fp-group"><div class="gl">field ids</div><div class="tags">` +
          ids.slice(0, 40).map(field => `<span class="ftag">${escapeHtml(field)}</span>`).join('') +
          `</div></div>`;
      }
      $('fields').innerHTML = html || '<div class="empty">(empty)</div>';
    }
    async function loadScopeControls() {
      const [presets, options] = await Promise.all([api('/api/presets'), api('/api/scope-options')]);
      presetRows = presets.presets || [];
      scopeRows = options.scopes || [];
      $('preset').innerHTML = '';
      for (const item of presetRows) {
        const option = document.createElement('option');
        option.value = item.name;
        option.textContent = `${item.name} · ${item.region}/${item.universe}/D${item.delay}`;
        $('preset').appendChild(option);
      }
      $('region').innerHTML = '';
      for (const region of options.regions || []) {
        const option = document.createElement('option');
        option.value = region;
        option.textContent = region;
        $('region').appendChild(option);
      }
      populateScopeRotationOptions();
      $('preset').value = 'ind';
      applyPresetToScope();
      $('preset').addEventListener('change', applyPresetToScope);
      $('region').addEventListener('change', updateDelayOptions);
      $('delay').addEventListener('change', updateUniverseAndNeutralizationOptions);
      $('run_minutes').addEventListener('change', updateRunLimitCustomState);
      updateRunLimitCustomState();
    }
    function updateRunLimitCustomState() { $('run_minutes_custom').disabled = $('run_minutes').value !== 'custom'; }
    function populateScopeRotationOptions() {
      const selectedDefaults = new Set(['USA|TOP3000|0', 'CHN|TOP2000U|0', 'EUR|TOP2500|1', 'GLB|TOP3000|1', 'IND|TOP500|1']);
      $('scope_rotation').innerHTML = '';
      for (const row of scopeRows) {
        const neutralization = (row.neutralizations || []).includes('INDUSTRY') ? 'INDUSTRY' : ((row.neutralizations || [])[0] || 'NONE');
        for (const universe of row.universes || []) {
          const scope = {region: row.region, universe, delay: Number(row.delay), neutralization};
          const option = document.createElement('option');
          option.value = JSON.stringify(scope);
          option.textContent = `${scope.region}/${scope.universe}/D${scope.delay}/${scope.neutralization}`;
          option.selected = selectedDefaults.has(`${scope.region}|${scope.universe}|${scope.delay}`);
          $('scope_rotation').appendChild(option);
        }
      }
    }
    function applyPresetToScope() {
      const row = presetRows.find(item => item.name === $('preset').value);
      if (!row) return;
      $('region').value = row.region;
      updateDelayOptions(row.delay);
      $('universe').value = row.universe;
      $('neutralization').value = row.neutralization;
    }
    function updateDelayOptions(selectedDelay=null) {
      const region = $('region').value;
      const rows = scopeRows.filter(row => row.region === region);
      const delays = [...new Set(rows.map(row => row.delay))].sort((a, b) => Number(a) - Number(b));
      $('delay').innerHTML = '';
      for (const delay of delays) {
        const option = document.createElement('option');
        option.value = delay;
        option.textContent = `D${delay}`;
        $('delay').appendChild(option);
      }
      if (selectedDelay !== null && delays.includes(Number(selectedDelay))) $('delay').value = String(selectedDelay);
      updateUniverseAndNeutralizationOptions();
    }
    function updateUniverseAndNeutralizationOptions() {
      const row = currentScopeRow();
      if (!row) return;
      const previousUniverse = $('universe').value;
      const previousNeutralization = $('neutralization').value;
      $('universe').innerHTML = '';
      for (const universe of row.universes || []) {
        const option = document.createElement('option'); option.value = universe; option.textContent = universe; $('universe').appendChild(option);
      }
      $('neutralization').innerHTML = '';
      for (const neutralization of row.neutralizations || []) {
        const option = document.createElement('option'); option.value = neutralization; option.textContent = neutralization; $('neutralization').appendChild(option);
      }
      if ((row.universes || []).includes(previousUniverse)) $('universe').value = previousUniverse;
      if ((row.neutralizations || []).includes(previousNeutralization)) $('neutralization').value = previousNeutralization;
      if (!$('neutralization').value && (row.neutralizations || []).includes('INDUSTRY')) $('neutralization').value = 'INDUSTRY';
    }
    function currentScopeRow() {
      const region = $('region').value;
      const delay = Number($('delay').value);
      return scopeRows.find(row => row.region === region && Number(row.delay) === delay);
    }
    function scopePayload() {
      return {region: $('region').value, universe: $('universe').value, delay: Number($('delay').value), neutralization: $('neutralization').value};
    }
    function selectedRotationScopes() {
      const scopes = Array.from($('scope_rotation').selectedOptions || []).map(option => JSON.parse(option.value));
      return scopes.length ? scopes : [scopePayload()];
    }
    async function refreshStatus() {
      const data = await api('/api/status');
      const counts = data.counts || {};
      const running = $('running');
      running.className = 'run-dot ' + (data.running ? 'on' : 'off');
      running.innerHTML = `<span class="led"></span><span>${data.running ? '运行中' : '已停止'}</span>`;
      $('approved').textContent = counts.approved || 0;
      $('pending').textContent = counts.check_pending || 0;
      $('failed').textContent = counts.failed || 0;
      const submitted = counts.submitted || 0;
      dailyCap = Number(data.max_final_submits_per_round ?? dailyCap) || dailyCap;
      const submitPct = dailyCap > 0 ? Math.min(100, (submitted / dailyCap) * 100) : 0;
      $('submitted').textContent = submitted;
      $('submit_prog').style.width = `${submitPct}%`;
      $('submit_cap').textContent = `日上限 ${dailyCap} · 已用 ${Math.min(submitted, dailyCap)}/${dailyCap}`;
      $('run_window').textContent = data.run_started_at ? `本轮起点 ${data.run_started_at}` : '本轮起点 --';
      $('history_counts').innerHTML = `历史累计 &nbsp; <b>${escapeHtml(formatCounts(data.history_counts || {}))}</b>`;
      const pid = data.pid || '--';
      $('pid').textContent = pid; $('pid_summary').textContent = pid;
      if ($('daemon_pid_side')) $('daemon_pid_side').textContent = pid;
      const rotationState = data.scope_rotation_state || {};
      const scope = rotationState.last_scope || data.daemon?.scope || {};
      const rotationCount = (data.daemon?.scope_rotation || []).length;
      const scopeText = scope.region ? `${scope.region}/${scope.universe}/D${scope.delay}/${scope.neutralization}` : (data.daemon?.preset || '--');
      $('daemon_scope').textContent = rotationCount ? `轮换${rotationCount}·当前 ${scopeText}` : scopeText;
      $('generator_mode_status').textContent = generatorModeLabel(data.daemon?.generator_mode);
      $('orchestration_mode_status').textContent = orchestrationModeLabel(data.daemon?.orchestration_mode);
      const autoSubmitEnabled = Boolean(data.daemon?.auto_submit || data.auto_submit);
      $('auto_submit_status').textContent = submitModeLabel(autoSubmitEnabled);
      if ($('daemon_ai_mode')) $('daemon_ai_mode').textContent = `${generatorModeLabel(data.daemon?.generator_mode)} · ${orchestrationModeLabel(data.daemon?.orchestration_mode)}`;
      if ($('daemon_submit_mode')) $('daemon_submit_mode').textContent = autoSubmitEnabled ? '自动提交' : '手动确认';
      if (data.running) {
        setRadioValue('generator_mode', data.daemon?.generator_mode);
        setRadioValue('orchestration_mode', data.daemon?.orchestration_mode);
        if ($('auto_submit')) $('auto_submit').checked = autoSubmitEnabled;
      }
      $('started_at').textContent = data.daemon?.started_at || '--';
      $('stop_after_at').textContent = data.daemon?.stop_after_at || '--';
      const plan = data.research_plan || {};
      $('plan_mode').textContent = plan.mode || '--';
      $('plan_target').textContent = plan.target_candidate_id || '--';
      $('plan_keep').textContent = (plan.keep || []).slice(0, 5).join(', ') || '--';
      $('plan_avoid').textContent = (plan.avoid || []).slice(0, 5).join(', ') || '--';
      renderRows($('efficiency_metrics'), data.efficiency || {});
      renderRows($('scheduler_plan'), data.scheduler_plan || {});
      renderTopAlphas(data.top_alphas || []);
      $('updated').textContent = data.updated_at || '--';
      const body = $('candidates'); body.innerHTML = '';
      const recent = data.recent_candidates || [];
      if (!recent.length) body.innerHTML = '<tr><td colspan="6" class="empty">暂无数据</td></tr>';
      for (const row of recent) {
        const settings = row.settings || {}; const metrics = row.metrics || {};
        const tr = document.createElement('tr');
        const readyTag = (!autoSubmitEnabled && row.status === 'approved') ? '<span class="ready-tag">可提交</span>' : '';
        tr.innerHTML = `<td>${row.id}</td><td><span class="pill ${escapeHtml(row.status)}">${escapeHtml(row.status)}</span>${readyTag}</td><td>${escapeHtml(row.source || '--')}</td><td>${escapeHtml(settings.region || '--')}/D${settings.delay ?? '--'}</td><td>S ${metrics.sharpe ?? '--'}<br>F ${metrics.fitness ?? '--'}</td><td><code>${escapeHtml(row.expression)}</code></td>`;
        body.appendChild(tr);
      }
      renderCandidateQueues(data.candidate_queues || {}, autoSubmitEnabled);
      renderModelScores(data.model_scores || {});
    }
    let selectedTopAlphaId = '';
    let latestTopAlphaRows = [];
    const topAlphaActionResults = {};
    const topAlphaActionBusy = {};
    function renderTopAlphas(rows) {
      latestTopAlphaRows = rows;
      const body = $('top_alphas');
      body.innerHTML = '';
      $('top_alpha_count').textContent = `${rows.length} / 10`;
      if (!rows.length) {
        selectedTopAlphaId = '';
        body.innerHTML = '<tr><td colspan="10" class="empty">本轮暂无可排名的 Alpha 指标</td></tr>';
        return;
      }
      const currentIds = new Set(rows.map(row => String(row.alpha_id || '')));
      for (const alphaId of Object.keys(topAlphaActionResults)) {
        if (!currentIds.has(alphaId)) delete topAlphaActionResults[alphaId];
      }
      if (!rows.some(row => String(row.alpha_id) === selectedTopAlphaId)) selectedTopAlphaId = '';
      for (const row of rows) {
        const alphaId = String(row.alpha_id || '');
        const settings = row.settings || {};
        const delay = settings.delay === null || settings.delay === undefined ? '--' : settings.delay;
        const scope = [settings.region, settings.universe, `D${delay}`, settings.neutralization].filter(Boolean).join(' / ') || '--';
        const busyAction = topAlphaActionBusy[alphaId] || '';
        const submitted = row.status === 'submitted';
        const tr = document.createElement('tr');
        tr.className = 'top-alpha-row';
        tr.tabIndex = 0;
        tr.setAttribute('role', 'button');
        tr.setAttribute('aria-expanded', String(alphaId === selectedTopAlphaId));
        tr.classList.toggle('selected', alphaId === selectedTopAlphaId);
        tr.dataset.alphaId = alphaId;
        tr.addEventListener('click', () => toggleTopAlphaExpression(row));
        tr.addEventListener('keydown', event => {
          if (event.target.closest('button')) return;
          if (event.key !== 'Enter' && event.key !== ' ') return;
          event.preventDefault();
          toggleTopAlphaExpression(row, true);
        });
        tr.innerHTML = `<td data-label="Rank" class="top-rank">${escapeHtml(row.rank)}</td>` +
          `<td data-label="Platform ID" class="platform-alpha-id"><code>${escapeHtml(row.alpha_id)}</code></td>` +
          `<td data-label="Status" class="top-status"><span class="pill ${escapeHtml(row.status)}">${escapeHtml(row.status || '--')}</span></td>` +
          `<td data-label="Score" class="top-score">${fmtMetric(row.quality_score)}</td>` +
          `<td data-label="Sharpe" class="top-metric">${fmtMetric(row.sharpe)}</td>` +
          `<td data-label="Fitness" class="top-metric">${fmtMetric(row.fitness)}</td>` +
          `<td data-label="Returns" class="top-metric">${fmtPercent(row.returns)}</td>` +
          `<td data-label="Turnover" class="top-metric">${fmtPercent(row.turnover)}</td>` +
          `<td data-label="Scope" class="top-scope">${escapeHtml(scope)}</td>` +
          `<td data-label="Actions" class="top-actions"><div class="top-action-set">` +
            `<button class="top-action check" type="button" title="查看官方相关性" aria-label="查看 ${escapeHtml(alphaId)} 的官方相关性" ${busyAction ? 'disabled' : ''}>${busyAction === 'check' ? 'CHECKING' : 'CHECK'}</button>` +
            `<button class="top-action submit" type="button" title="直接提交到 WorldQuant BRAIN" aria-label="直接提交 ${escapeHtml(alphaId)}" ${(busyAction || submitted) ? 'disabled' : ''}>${busyAction === 'submit' ? 'SUBMITTING' : (submitted ? 'SUBMITTED' : 'SUBMIT')}</button>` +
          `</div></td>`;
        tr.querySelector('.top-action.check').addEventListener('click', event => {
          event.stopPropagation();
          checkTopAlpha(alphaId);
        });
        tr.querySelector('.top-action.submit').addEventListener('click', event => {
          event.stopPropagation();
          submitTopAlpha(alphaId);
        });
        body.appendChild(tr);
        if (alphaId === selectedTopAlphaId) {
          const detail = document.createElement('tr');
          detail.className = 'top-alpha-expression';
          detail.id = 'top_alpha_expression';
          detail.setAttribute('aria-live', 'polite');
          detail.innerHTML = `<td colspan="10">` +
            `<div class="top-alpha-expression-h"><span>Expression</span><code id="top_alpha_expression_id">${escapeHtml(row.alpha_id || '--')}</code></div>` +
            `<pre><code id="top_alpha_expression_code">${escapeHtml(row.expression || '')}</code></pre>` +
            renderTopAlphaActionResult(alphaId) +
            `</td>`;
          body.appendChild(detail);
        }
      }
    }
    function renderTopAlphaActionResult(alphaId) {
      const action = topAlphaActionResults[alphaId];
      if (!action) return '';
      const data = action.data || {};
      if (action.type === 'submit') {
        const state = data.submitted ? '官方已确认提交' : '平台未确认提交';
        return `<div class="top-action-result"><div class="top-action-result-h"><b>${escapeHtml(state)}</b>` +
          `<span>stage ${escapeHtml(data.stage || 'UNKNOWN')} · ${escapeHtml(data.message || '')}</span></div></div>`;
      }
      const correlations = data.correlations || {};
      const correlationHtml = ['self', 'production'].map(key => {
        const item = correlations[key] || {};
        const available = item.value !== null && item.value !== undefined && item.value !== '';
        const value = available ? fmtMetric(item.value) : '平台未返回';
        const limit = item.limit === null || item.limit === undefined ? '' : ` / limit ${fmtMetric(item.limit)}`;
        return `<div class="correlation-item ${available ? 'confirmed' : 'unconfirmed'}">` +
          `<span class="name">${key === 'self' ? 'Self corr' : 'Prod corr'}</span>` +
          `<span class="value">${escapeHtml(value)}</span>` +
          `<span class="state">${escapeHtml(item.status || 'UNKNOWN')}${escapeHtml(limit)}</span></div>`;
      }).join('');
      const checks = Object.entries(data.checks || {}).map(([name, item]) => {
        const detail = item && typeof item === 'object' ? item : {status: item};
        const value = detail.value === null || detail.value === undefined ? '' : ` ${fmtVal(detail.value)}`;
        return `<span class="official-check">${escapeHtml(name)} <b>${escapeHtml(detail.status || 'UNKNOWN')}${escapeHtml(value)}</b></span>`;
      }).join('');
      return `<div class="top-action-result"><div class="top-action-result-h">` +
        `<b>WorldQuant BRAIN 官方检查</b><span>${escapeHtml(data.message || '')} · ${escapeHtml(data.checked_at || '')}</span></div>` +
        `<div class="correlation-grid">${correlationHtml}</div>` +
        `<div class="official-checks">${checks || '<span class="official-check"><b>NO DATA</b></span>'}</div></div>`;
    }
    async function checkTopAlpha(alphaId) {
      selectedTopAlphaId = alphaId;
      topAlphaActionBusy[alphaId] = 'check';
      renderTopAlphas(latestTopAlphaRows);
      try {
        const data = await api('/api/top-alpha/check', {method:'POST', body:JSON.stringify({alpha_id:alphaId})});
        topAlphaActionResults[alphaId] = {type:'check', data};
      } catch (error) {
        topAlphaActionResults[alphaId] = {type:'check', data:{message:String(error), checks:{}, correlations:{}}};
      } finally {
        delete topAlphaActionBusy[alphaId];
        renderTopAlphas(latestTopAlphaRows);
      }
    }
    async function submitTopAlpha(alphaId) {
      if (!window.confirm(`确定直接提交 ${alphaId} 到 WorldQuant BRAIN？提交后不可撤销。`)) return;
      selectedTopAlphaId = alphaId;
      topAlphaActionBusy[alphaId] = 'submit';
      renderTopAlphas(latestTopAlphaRows);
      try {
        const data = await api('/api/top-alpha/submit', {method:'POST', body:JSON.stringify({alpha_id:alphaId})});
        topAlphaActionResults[alphaId] = {type:'submit', data};
        if (data.submitted) await refreshStatus();
      } catch (error) {
        topAlphaActionResults[alphaId] = {type:'submit', data:{submitted:false, stage:'ERROR', message:String(error)}};
      } finally {
        delete topAlphaActionBusy[alphaId];
        renderTopAlphas(latestTopAlphaRows);
      }
    }
    function toggleTopAlphaExpression(row, restoreFocus=false) {
      const alphaId = String(row.alpha_id || '');
      selectedTopAlphaId = selectedTopAlphaId === alphaId ? '' : alphaId;
      renderTopAlphas(latestTopAlphaRows);
      if (restoreFocus) {
        const target = [...document.querySelectorAll('#top_alphas .top-alpha-row')]
          .find(tr => tr.dataset.alphaId === alphaId);
        target?.focus();
      }
    }
    function renderCandidateQueues(queues, autoSubmitEnabled=false) {
      const counts = queues.counts || {};
      const order = ['submitable', 'watchlist', 'optimize', 'trash', 'abandoned'];
      for (const key of order) {
        $(`queue_${key}`).textContent = counts[key] || 0;
        const chip = $(`queue_${key}`).closest('.chip');
        if (chip) chip.classList.toggle('ready', key === 'submitable' && !autoSubmitEnabled && Number(counts[key] || 0) > 0);
      }
      const body = $('queue_examples'); body.innerHTML = '';
      let any = false;
      for (const key of order) {
        for (const row of (queues[key] || []).slice(0, 2)) {
          any = true;
          const tr = document.createElement('tr');
          const readyTag = (key === 'submitable' && !autoSubmitEnabled) ? '<span class="ready-tag">可提交</span>' : '';
          tr.innerHTML = `<td><span class="qtag">${key}</span>${readyTag}</td><td>${row.id}</td><td>Q ${row.quality_score ?? '--'}<br>R ${row.readiness_score ?? '--'}</td><td>${escapeHtml(row.queue_reason || '--')}</td><td><code>${escapeHtml(row.expression || '')}</code></td>`;
          body.appendChild(tr);
        }
      }
      if (!any) body.innerHTML = '<tr><td colspan="5" class="empty">暂无数据</td></tr>';
    }
    function renderModelScores(scores) {
      const body = $('model_scores'); body.innerHTML = '';
      const rows = Object.entries(scores).sort((a, b) => Number(b[1].best_sharpe ?? -999) - Number(a[1].best_sharpe ?? -999));
      if (!rows.length) { body.innerHTML = '<tr><td colspan="5" class="empty">暂无数据</td></tr>'; return; }
      for (const [model, score] of rows) {
        const tr = document.createElement('tr');
        tr.innerHTML = `<td>${escapeHtml(model)}</td><td>${score.generated || 0}</td><td>${score.approved || 0}</td><td>${score.failed || 0}</td><td>S ${score.best_sharpe ?? '--'}<br>F ${score.best_fitness ?? '--'}</td>`;
        body.appendChild(tr);
      }
    }
    async function startDaemon() {
      const payload = {...scopePayload(), batch_size:Number($('batch_size').value), loop_seconds:Number($('loop_seconds').value), run_minutes:selectedRunMinutes(), generator_mode:selectedGeneratorMode(), orchestration_mode:selectedOrchestrationMode(), auto_submit:selectedAutoSubmit()};
      if ($('rotate_scopes').checked) payload.scope_rotation = selectedRotationScopes();
      const data = await api('/api/start', {method:'POST', body:JSON.stringify(payload)});
      msg(data.message);
      await refreshStatus(); await loadLogs();
    }
    function selectedRunMinutes() {
      if ($('run_minutes').value !== 'custom') return Number($('run_minutes').value);
      const minutes = Number($('run_minutes_custom').value);
      return Number.isFinite(minutes) && minutes > 0 ? minutes : 240;
    }
    function selectedGeneratorMode() {
      const selected = document.querySelector('input[name="generator_mode"]:checked');
      return selected ? selected.value : 'single';
    }
    function selectedOrchestrationMode() {
      const selected = document.querySelector('input[name="orchestration_mode"]:checked');
      return selected ? selected.value : 'lean';
    }
    function selectedAutoSubmit() {
      return Boolean($('auto_submit') && $('auto_submit').checked);
    }
    function setRadioValue(name, value) {
      const selected = document.querySelector(`input[name="${name}"][value="${value}"]`);
      if (selected) selected.checked = true;
    }
    async function stopDaemon() {
      const data = await api('/api/stop', {method:'POST', body:'{}'});
      msg(data.message); await refreshStatus(); await loadLogs();
    }
    async function loadLogs() {
      const data = await api(`/api/logs?file=${$('log_file').value}&lines=${$('lines').value}`);
      $('logs').textContent = data.logs || '';
      $('logs').scrollTop = $('logs').scrollHeight;
    }
    async function clearLogs(all=false) {
      const payload = all ? {all:true} : {file:$('log_file').value};
      const data = await api('/api/clear-logs', {method:'POST', body:JSON.stringify(payload)});
      msg(data.message); await loadLogs();
    }
    async function loadFields() {
      $('fields').innerHTML = '<div class="empty">loading...</div>';
      const params = new URLSearchParams({...scopePayload(), limit: '40'});
      const data = await api(`/api/fields?${params.toString()}`);
      renderFields(data);
    }
    loadScopeControls().then(refreshStatus).then(loadLogs);
    setInterval(refreshStatus, 5000);
    setInterval(loadLogs, 7000);
  </script>
</body>
</html>"""


if __name__ == "__main__":
    raise SystemExit(run_web_app())
