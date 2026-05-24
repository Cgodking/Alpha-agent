from __future__ import annotations

import json
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
from .health import daemon_health
from .metrics import compute_efficiency_metrics
from .scopes import SCOPE_PRESETS, platform_scope_rows, preset_rows
from .scheduler import build_cycle_plan


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
DEFAULT_DAEMON_STDOUT = PROJECT_ROOT / "logs" / "daemon.stdout.log"
DEFAULT_WEB_LOG = PROJECT_ROOT / "logs" / "web.log"
MAX_WEB_BACKTEST_BATCH = 8
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
    run_minutes: float = 240.0,
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
        self.sleep_func = sleep_func
        self.status_cache_ttl_seconds = float(status_cache_ttl_seconds)
        self.research_context_cache_ttl_seconds = float(research_context_cache_ttl_seconds)
        self._status_cache_lock = threading.Lock()
        self._status_cache: Dict[str, Any] = {}
        self._research_context_cache_lock = threading.Lock()
        self._research_context_cache: Dict[str, Any] = {}

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
        run_minutes = _bounded_float(payload.get("run_minutes"), default=240.0, minimum=0.0, maximum=1440.0)
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
            "stop_after_at": _stop_after_at(run_minutes),
            "argv": argv,
            "started_at": started_at,
            "stopped_at": "",
            "stop_reason": "",
        }
        self.store.set_run_state("daemon", run_state)
        self.store.record_event(None, "web_daemon_started", run_state)
        self.clear_status_cache()
        return {"success": True, "message": f"daemon started pid={process.pid}", "pid": int(process.pid), "argv": argv}

    def stop(self) -> Dict[str, Any]:
        state = self.store.get_run_state("daemon")
        pid = int(state.get("pid") or 0)
        if not self.pid_running(pid):
            state.update({"status": "stopped", "stopped_at": utc_now()})
            self.store.set_run_state("daemon", state)
            self.clear_status_cache()
            return {"success": True, "message": "daemon is not running"}

        self.kill_func(pid, signal.SIGINT)
        self.sleep_func(1.0)
        if self.pid_running(pid):
            self.kill_func(pid, signal.SIGTERM)
        state.update({"status": "stopped", "stopped_at": utc_now()})
        self.store.set_run_state("daemon", state)
        self.store.record_event(None, "web_daemon_stopped", {"pid": pid})
        self.clear_status_cache()
        return {"success": True, "message": f"daemon stopped pid={pid}"}

    def status(self) -> Dict[str, Any]:
        cached = self._cached_status()
        if cached is not None:
            return cached

        state = self.store.get_run_state("daemon")
        rotation_state = self.store.get_run_state("scope_rotation")
        pid = int(state.get("pid") or 0)
        running = self.pid_running(pid)
        run_started_at = str(state.get("started_at") or "")
        history_counts = self.store.status_counts()
        counts = self.store.status_counts(created_since=run_started_at) if run_started_at else history_counts
        research = self._research_context(state)
        active_scope = state.get("scope") if isinstance(state.get("scope"), dict) else {}
        if not active_scope:
            active_scope = research.get("target_settings") if isinstance(research.get("target_settings"), dict) else {}
        active_scope = active_scope if isinstance(active_scope, dict) else {}
        efficiency = compute_efficiency_metrics(self.store, active_scope, created_since=run_started_at or None)
        scheduler_plan = build_cycle_plan(
            self.store,
            active_scope,
            batch_size=int(state.get("batch_size") or MAX_WEB_BACKTEST_BATCH),
        )
        payload = {
            "running": running,
            "daemon": state,
            "scope_rotation_state": rotation_state,
            "pid": pid if running else None,
            "health": daemon_health(self.store),
            "counts": counts,
            "history_counts": history_counts,
            "model_scores": self._model_scores(created_since=run_started_at),
            "run_started_at": run_started_at,
            "recent_candidates": self._recent_candidates(limit=20, created_since=run_started_at),
            "research_plan": research.get("experiment_plan", {}),
            "research_analysis": research.get("analysis", {}),
            "candidate_queues": _compact_candidate_queues_for_status(research.get("candidate_queues", {})),
            "efficiency": efficiency,
            "scheduler_plan": scheduler_plan,
            "cooldowns": scheduler_plan.get("constraints", {}),
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
        brain = BrainHTTPClient.from_env() if cfg.brain_client in {"http", "brain_http", "live"} else LocalBrainClient()
        catalog = build_field_catalog(brain, scope)
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

    def _research_context(self, state: Dict[str, Any]) -> Dict[str, Any]:
        rotation_state = self.store.get_run_state("scope_rotation")
        scope = rotation_state.get("last_scope") if isinstance(rotation_state.get("last_scope"), dict) else {}
        if not scope:
            scope = state.get("scope") if isinstance(state.get("scope"), dict) else {}
        if not scope:
            recent = self._recent_candidates(limit=1)
            if recent:
                scope = recent[0].get("settings") if isinstance(recent[0].get("settings"), dict) else {}
        cache_key = _json_cache_key(scope or {})
        cached = self._cached_research_context(cache_key)
        if cached is not None:
            return cached
        research = self._build_research_context(scope or {})
        self._store_research_context_cache(cache_key, research)
        return deepcopy(research)

    def _build_research_context(self, scope: Dict[str, Any]) -> Dict[str, Any]:
        try:
            return build_ai_research_context(
                self.store,
                scope or {},
                field_catalog={"available": False, "field_ids": []},
            )
        except Exception as exc:
            return {"experiment_plan": {"mode": "unavailable", "error": str(exc)}, "analysis": {}}

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
    host: str = "0.0.0.0",
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
    handler = make_handler(service)
    server = ThreadingHTTPServer((host, int(port)), handler)
    print(f"Alpha control panel listening on http://{host}:{port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("web stopped", flush=True)
    finally:
        server.server_close()
    return 0


def make_handler(service: ControlService):
    class AlphaWebHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
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
            parsed = urlparse(self.path)
            payload = self._read_json()
            try:
                if parsed.path == "/api/start":
                    self._send_json(service.start(payload))
                elif parsed.path == "/api/stop":
                    self._send_json(service.stop())
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


HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Alpha 控制台</title>
  <style>
    :root { --bg:#f7f8fa; --surface:#ffffff; --ink:#17202a; --muted:#687386; --line:#dce1e8; --accent:#0f766e; --danger:#b42318; --warn:#b7791f; --log:#10151f; }
    * { box-sizing:border-box; }
    body { margin:0; background:var(--bg); color:var(--ink); font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Arial,sans-serif; }
    header { height:56px; display:flex; align-items:center; justify-content:space-between; padding:0 22px; background:#fff; border-bottom:1px solid var(--line); }
    h1 { font-size:18px; margin:0; letter-spacing:0; }
    main { max-width:1440px; margin:0 auto; padding:18px; display:grid; grid-template-columns:340px 1fr; gap:18px; }
    section { background:var(--surface); border:1px solid var(--line); border-radius:8px; padding:14px; }
    h2 { margin:0 0 12px; font-size:13px; color:#344054; }
    label { display:block; margin:10px 0 5px; color:var(--muted); font-size:12px; }
    input, select { width:100%; height:36px; padding:0 10px; border:1px solid var(--line); border-radius:6px; background:#fff; color:var(--ink); }
    select[multiple] { height:132px; padding:6px 8px; }
    .inline-check { display:flex; align-items:center; gap:8px; color:var(--muted); font-size:12px; }
    .inline-check input { width:auto; height:auto; margin:0; }
    button { height:36px; border-radius:6px; border:1px solid var(--line); background:#fff; font-weight:650; cursor:pointer; }
    button.primary { background:var(--accent); border-color:var(--accent); color:#fff; }
    button.danger { background:var(--danger); border-color:var(--danger); color:#fff; }
    .actions { display:grid; grid-template-columns:1fr 1fr; gap:8px; margin-top:14px; }
    .grid { display:grid; grid-template-columns:repeat(5, minmax(0, 1fr)); gap:10px; }
    .metric { border-top:1px solid var(--line); padding-top:10px; min-height:60px; }
    .label { color:var(--muted); font-size:12px; }
    .value { margin-top:4px; font-size:20px; font-weight:750; overflow-wrap:anywhere; }
    .sub { margin-top:4px; color:var(--muted); font-size:11px; overflow-wrap:anywhere; }
    .ok { color:var(--accent); } .bad { color:var(--danger); } .warn { color:var(--warn); }
    .message { min-height:20px; margin-top:10px; color:var(--muted); font-size:13px; }
    .logs { height:430px; overflow:auto; white-space:pre-wrap; background:var(--log); color:#d6dde8; border-radius:8px; padding:12px; font:12px/1.55 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace; }
    .toolbar { display:flex; gap:8px; align-items:center; margin-bottom:10px; }
    .toolbar h2 { margin-right:auto; margin-bottom:0; }
    .toolbar input, .toolbar select { width:120px; }
    table { width:100%; border-collapse:collapse; font-size:12px; }
    th, td { border-top:1px solid var(--line); padding:8px 6px; text-align:left; vertical-align:top; }
    th { color:var(--muted); font-weight:650; }
    td code { font-size:11px; overflow-wrap:anywhere; }
    .panel-gap { margin-top:18px; }
    .small { font-size:12px; color:var(--muted); line-height:1.45; }
    @media (max-width: 980px) { main { grid-template-columns:1fr; } .grid { grid-template-columns:repeat(2, minmax(0, 1fr)); } }
  </style>
</head>
<body>
  <header>
    <h1>Alpha 控制台</h1>
    <div id="updated" class="label">--</div>
  </header>
  <main>
    <div>
      <section>
        <h2>控制</h2>
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
        <select id="scope_rotation" multiple size="7"></select>
        <label>Batch size</label>
        <input id="batch_size" type="number" value="8" min="1" max="8">
        <label>Loop seconds</label>
        <input id="loop_seconds" type="number" value="60" min="1" max="3600">
        <label>Run limit</label>
        <select id="run_minutes">
          <option value="120">2 小时</option>
          <option value="240" selected>4 小时</option>
          <option value="480">8 小时</option>
          <option value="custom">自定义</option>
          <option value="0">不自动停止</option>
        </select>
        <label>Custom minutes</label>
        <input id="run_minutes_custom" type="number" value="240" min="1" max="1440" disabled>
        <div class="actions">
          <button class="primary" onclick="startDaemon()">启动</button>
          <button class="danger" onclick="stopDaemon()">暂停</button>
        </div>
        <div id="message" class="message"></div>
      </section>
      <section class="panel-gap">
        <h2>字段池</h2>
        <button onclick="loadFields()">刷新当前 scope 字段</button>
        <div id="fields" class="small" style="margin-top:10px">--</div>
      </section>
      <section class="panel-gap">
        <h2>进程</h2>
        <div class="small">PID: <span id="pid">--</span></div>
        <div class="small">Scope: <span id="daemon_scope">--</span></div>
        <div class="small">Started: <span id="started_at">--</span></div>
        <div class="small">Auto stop: <span id="stop_after_at">--</span></div>
      </section>
      <section class="panel-gap">
        <h2>实验计划</h2>
        <div class="small">Mode: <span id="plan_mode">--</span></div>
        <div class="small">Target: <span id="plan_target">--</span></div>
        <div class="small">Keep: <span id="plan_keep">--</span></div>
        <div class="small">Avoid: <span id="plan_avoid">--</span></div>
      </section>
    </div>
    <div>
      <section>
        <h2>本轮状态</h2>
        <div class="grid">
          <div class="metric"><div class="label">Daemon</div><div id="running" class="value">--</div></div>
          <div class="metric"><div class="label">Approved</div><div id="approved" class="value">0</div></div>
          <div class="metric"><div class="label">Pending</div><div id="pending" class="value">0</div></div>
          <div class="metric"><div class="label">Failed</div><div id="failed" class="value">0</div></div>
          <div class="metric"><div class="label">Submitted</div><div id="submitted" class="value">0</div></div>
        </div>
        <div id="run_window" class="small" style="margin-top:10px">--</div>
        <div id="history_counts" class="small">历史累计：--</div>
      </section>
      <section class="panel-gap">
        <h2>候选队列</h2>
        <div class="grid" id="candidate_queues">
          <div class="metric"><div class="label">Submitable</div><div id="queue_submitable" class="value">0</div></div>
          <div class="metric"><div class="label">Watchlist</div><div id="queue_watchlist" class="value">0</div></div>
          <div class="metric"><div class="label">Optimize</div><div id="queue_optimize" class="value">0</div></div>
          <div class="metric"><div class="label">Trash</div><div id="queue_trash" class="value">0</div></div>
          <div class="metric"><div class="label">Abandoned</div><div id="queue_abandoned" class="value">0</div></div>
        </div>
        <table style="margin-top:10px">
          <thead><tr><th>Queue</th><th>ID</th><th>Score</th><th>Reason</th><th>Expression</th></tr></thead>
          <tbody id="queue_examples"></tbody>
        </table>
      </section>
      <section class="panel-gap">
        <h2>Efficiency</h2>
        <pre id="efficiency_metrics" class="small">--</pre>
      </section>
      <section class="panel-gap">
        <h2>Scheduler</h2>
        <pre id="scheduler_plan" class="small">--</pre>
      </section>
      <section class="panel-gap">
        <h2>本轮候选</h2>
        <table>
          <thead><tr><th>ID</th><th>Status</th><th>Model</th><th>Scope</th><th>Metrics</th><th>Expression</th></tr></thead>
          <tbody id="candidates"></tbody>
        </table>
      </section>
      <section class="panel-gap">
        <h2>模型表现</h2>
        <table>
          <thead><tr><th>Model</th><th>Generated</th><th>Approved</th><th>Failed</th><th>Best</th></tr></thead>
          <tbody id="model_scores"></tbody>
        </table>
      </section>
      <section class="panel-gap">
        <div class="toolbar">
          <h2>日志</h2>
          <select id="log_file"><option value="alpha">alpha.log</option><option value="daemon">daemon stdout</option><option value="web">web.log</option></select>
          <input id="lines" type="number" value="300">
          <button onclick="loadLogs()">刷新</button>
          <button class="danger" onclick="clearLogs(false)">清空当前</button>
          <button class="danger" onclick="clearLogs(true)">清空全部</button>
        </div>
        <div id="logs" class="logs">loading...</div>
      </section>
    </div>
  </main>
  <script>
    const $ = (id) => document.getElementById(id);
    let presetRows = [];
    let scopeRows = [];
    function msg(text) { $('message').textContent = text || ''; }
    async function api(path, options={}) {
      const response = await fetch(path, {headers:{'Content-Type':'application/json'}, ...options});
      return await response.json();
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
    function updateRunLimitCustomState() {
      $('run_minutes_custom').disabled = $('run_minutes').value !== 'custom';
    }
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
      if (selectedDelay !== null && delays.includes(Number(selectedDelay))) {
        $('delay').value = String(selectedDelay);
      }
      updateUniverseAndNeutralizationOptions();
    }
    function updateUniverseAndNeutralizationOptions() {
      const row = currentScopeRow();
      if (!row) return;
      const previousUniverse = $('universe').value;
      const previousNeutralization = $('neutralization').value;
      $('universe').innerHTML = '';
      for (const universe of row.universes || []) {
        const option = document.createElement('option');
        option.value = universe;
        option.textContent = universe;
        $('universe').appendChild(option);
      }
      $('neutralization').innerHTML = '';
      for (const neutralization of row.neutralizations || []) {
        const option = document.createElement('option');
        option.value = neutralization;
        option.textContent = neutralization;
        $('neutralization').appendChild(option);
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
      return {
        region: $('region').value,
        universe: $('universe').value,
        delay: Number($('delay').value),
        neutralization: $('neutralization').value,
      };
    }
    function selectedRotationScopes() {
      const scopes = Array.from($('scope_rotation').selectedOptions || []).map(option => JSON.parse(option.value));
      return scopes.length ? scopes : [scopePayload()];
    }
    async function refreshStatus() {
      const data = await api('/api/status');
      const counts = data.counts || {};
      $('running').textContent = data.running ? '运行中' : '已暂停';
      $('running').className = 'value ' + (data.running ? 'ok' : 'bad');
      $('approved').textContent = counts.approved || 0;
      $('pending').textContent = counts.check_pending || 0;
      $('failed').textContent = counts.failed || 0;
      $('submitted').textContent = counts.submitted || 0;
      $('run_window').textContent = data.run_started_at ? `本轮起点：${data.run_started_at}` : '本轮起点：--';
      $('history_counts').textContent = `历史累计：${formatCounts(data.history_counts || {})}`;
      $('pid').textContent = data.pid || '--';
      const rotationState = data.scope_rotation_state || {};
      const scope = rotationState.last_scope || data.daemon?.scope || {};
      const rotationCount = (data.daemon?.scope_rotation || []).length;
      const scopeText = scope.region ? `${scope.region}/${scope.universe}/D${scope.delay}/${scope.neutralization}` : (data.daemon?.preset || '--');
      $('daemon_scope').textContent = rotationCount ? `轮换 ${rotationCount} 个；当前 ${scopeText}；下个 #${(rotationState.next_index ?? 0) + 1}` : scopeText;
      $('started_at').textContent = data.daemon?.started_at || '--';
      $('stop_after_at').textContent = data.daemon?.stop_after_at || '--';
      const plan = data.research_plan || {};
      $('plan_mode').textContent = plan.mode || '--';
      $('plan_target').textContent = plan.target_candidate_id || '--';
      $('plan_keep').textContent = (plan.keep || []).slice(0, 5).join(', ') || '--';
      $('plan_avoid').textContent = (plan.avoid || []).slice(0, 5).join(', ') || '--';
      $('efficiency_metrics').textContent = JSON.stringify(data.efficiency || {}, null, 2);
      $('scheduler_plan').textContent = JSON.stringify(data.scheduler_plan || {}, null, 2);
      $('updated').textContent = data.updated_at || '--';
      const body = $('candidates');
      body.innerHTML = '';
      for (const row of data.recent_candidates || []) {
        const settings = row.settings || {};
        const metrics = row.metrics || {};
        const tr = document.createElement('tr');
        tr.innerHTML = `<td>${row.id}</td><td>${row.status}</td><td>${escapeHtml(row.source || '--')}</td><td>${settings.region || '--'}/D${settings.delay ?? '--'}</td><td>S ${metrics.sharpe ?? '--'}<br>F ${metrics.fitness ?? '--'}</td><td><code>${escapeHtml(row.expression)}</code></td>`;
        body.appendChild(tr);
      }
      renderCandidateQueues(data.candidate_queues || {});
      renderModelScores(data.model_scores || {});
    }
    function renderCandidateQueues(queues) {
      const counts = queues.counts || {};
      const order = ['submitable', 'watchlist', 'optimize', 'trash', 'abandoned'];
      for (const key of order) {
        $(`queue_${key}`).textContent = counts[key] || 0;
      }
      const body = $('queue_examples');
      body.innerHTML = '';
      for (const key of order) {
        for (const row of (queues[key] || []).slice(0, 2)) {
          const tr = document.createElement('tr');
          tr.innerHTML = `<td>${key}</td><td>${row.id}</td><td>Q ${row.quality_score ?? '--'}<br>R ${row.readiness_score ?? '--'}</td><td>${escapeHtml(row.queue_reason || '--')}</td><td><code>${escapeHtml(row.expression || '')}</code></td>`;
          body.appendChild(tr);
        }
      }
    }
    function renderModelScores(scores) {
      const body = $('model_scores');
      body.innerHTML = '';
      const rows = Object.entries(scores).sort((a, b) => {
        const ash = Number(a[1].best_sharpe ?? -999);
        const bsh = Number(b[1].best_sharpe ?? -999);
        return bsh - ash;
      });
      for (const [model, score] of rows) {
        const tr = document.createElement('tr');
        tr.innerHTML = `<td>${escapeHtml(model)}</td><td>${score.generated || 0}</td><td>${score.approved || 0}</td><td>${score.failed || 0}</td><td>S ${score.best_sharpe ?? '--'}<br>F ${score.best_fitness ?? '--'}</td>`;
        body.appendChild(tr);
      }
    }
    async function startDaemon() {
      const payload = {...scopePayload(), batch_size:Number($('batch_size').value), loop_seconds:Number($('loop_seconds').value), run_minutes:selectedRunMinutes()};
      if ($('rotate_scopes').checked) payload.scope_rotation = selectedRotationScopes();
      const data = await api('/api/start', {method:'POST', body:JSON.stringify(payload)});
      msg(data.message);
      await refreshStatus();
      await loadLogs();
    }
    function selectedRunMinutes() {
      if ($('run_minutes').value !== 'custom') return Number($('run_minutes').value);
      const minutes = Number($('run_minutes_custom').value);
      return Number.isFinite(minutes) && minutes > 0 ? minutes : 240;
    }
    async function stopDaemon() {
      const data = await api('/api/stop', {method:'POST', body:'{}'});
      msg(data.message);
      await refreshStatus();
      await loadLogs();
    }
    async function loadLogs() {
      const data = await api(`/api/logs?file=${$('log_file').value}&lines=${$('lines').value}`);
      $('logs').textContent = data.logs || '';
      $('logs').scrollTop = $('logs').scrollHeight;
    }
    async function clearLogs(all=false) {
      const payload = all ? {all:true} : {file:$('log_file').value};
      const data = await api('/api/clear-logs', {method:'POST', body:JSON.stringify(payload)});
      msg(data.message);
      await loadLogs();
    }
    async function loadFields() {
      $('fields').textContent = 'loading...';
      const params = new URLSearchParams({...scopePayload(), limit: '40'});
      const data = await api(`/api/fields?${params.toString()}`);
      const ids = data.field_ids || [];
      const datasets = (data.datasets || []).slice(0, 8).map(d => `${d.id}:${d.field_count}`).join(', ');
      const scout = data.field_scout || {};
      const topScout = (scout.top_fields || []).slice(0, 12)
        .map(f => `${f.field} score=${Number(f.score || 0).toFixed(3)} ${f.category || ''} ${f.primary_policy || ''}`)
        .join('\n');
      const buckets = (scout.buckets || []).slice(0, 5)
        .map(b => `${b.name}: ${(b.fields || []).slice(0, 12).join(', ')}`)
        .join('\n');
      $('fields').textContent =
        `available=${data.available} source=${data.source || 'unknown'}\n${datasets}\n\nfield scout\n${topScout || '(empty)'}\n\nbuckets\n${buckets || '(empty)'}\n\nfield ids\n${ids.join(', ')}`;
    }
    function escapeHtml(value) {
      return String(value || '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
    }
    function formatCounts(counts) {
      const order = ['approved', 'check_pending', 'failed', 'submitted', 'preflight_passed', 'generated'];
      const parts = [];
      for (const key of order) {
        if (counts[key]) parts.push(`${key} ${counts[key]}`);
      }
      for (const key of Object.keys(counts).sort()) {
        if (!order.includes(key) && counts[key]) parts.push(`${key} ${counts[key]}`);
      }
      return parts.join(' / ') || '0';
    }
    loadScopeControls().then(refreshStatus).then(loadLogs);
    setInterval(refreshStatus, 5000);
    setInterval(loadLogs, 7000);
  </script>
</body>
</html>"""


if __name__ == "__main__":
    raise SystemExit(run_web_app())
