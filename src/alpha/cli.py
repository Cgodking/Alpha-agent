from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from .clients import BrainHTTPClient, LocalAIClient, LocalBrainClient, MultiModelAIClient, OpenAICompatibleAIClient
from .config import load_config
from .context_builder import build_ai_research_context
from .db import AlphaStore, utc_now
from .env_file import load_env_file
from .field_catalog import build_field_catalog
from .field_scout import build_field_scout
from .history_prune import DEFAULT_LOW_QUALITY_SCORE_MAX, prune_low_quality_history
from .logging_utils import setup_logging
from .metrics import compute_efficiency_metrics
from .scopes import SCOPE_PRESETS, apply_scope, preset_rows
from .scope_rotation import next_rotating_scope, parse_scope_json
from .scheduler import build_cycle_plan
from .submission import submit_approved_candidates
from .worker import AlphaWorker


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="alpha", description="WQB Alpha 自动探索服务")
    parser.add_argument("--db", default=None, help="SQLite 数据库路径")
    parser.add_argument("--env-file", default=".env", help="dotenv 风格配置文件")
    parser.add_argument("--log-file", default="logs/alpha.log", help="结构化服务日志路径")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("init-db", help="初始化数据库")

    run_once = sub.add_parser("run-once", help="运行一轮生成、回测和评估")
    run_once.add_argument("--batch-size", type=int, default=None)
    _add_scope_args(run_once)

    daemon = sub.add_parser("daemon", help="持续运行直到被停止")
    daemon.add_argument("--batch-size", type=int, default=None)
    daemon.add_argument("--loop-seconds", type=float, default=None)
    daemon.add_argument("--run-minutes", type=float, default=None, help="运行指定分钟数后自动停止")
    daemon.add_argument("--scope-json", default=None, help="daemon 轮换探索的 scope JSON 列表")
    daemon.add_argument("--throughput-mode", action="store_true", help="使用 scheduler cycle_plan 提升个人研究吞吐")
    daemon.add_argument(
        "--generator-mode",
        choices=("single", "balanced"),
        default=None,
        help="AI 生成模型分配：single=单模型省 token；balanced=双生成模型平分 batch",
    )
    daemon.add_argument(
        "--orchestration-mode",
        choices=("lean", "deep"),
        default=None,
        help="AI 编排模式：lean=省 token 直连生成；deep=启用 controller/critic 决策链",
    )
    daemon_auto_submit = daemon.add_mutually_exclusive_group()
    daemon_auto_submit.add_argument(
        "--auto-submit",
        dest="auto_submit",
        action="store_true",
        default=None,
        help="本轮 daemon 对达标 alpha 执行真实提交",
    )
    daemon_auto_submit.add_argument(
        "--no-auto-submit",
        dest="auto_submit",
        action="store_false",
        help="本轮 daemon 只审批达标 alpha，不执行真实提交",
    )
    _add_scope_args(daemon)

    status = sub.add_parser("status", help="打印候选状态统计")
    status.add_argument("--efficiency", action="store_true", help="打印个人效率指标")
    prune = sub.add_parser("prune-history", help="归档并删除低质量失败探索记录")
    prune.add_argument("--quality-max", type=float, default=DEFAULT_LOW_QUALITY_SCORE_MAX)
    prune.add_argument("--limit", type=int, default=1000)
    prune.add_argument("--execute", action="store_true", help="真正执行归档删除；不加则只预览")
    prune.add_argument("--all-scopes", action="store_true", help="清理所有 scope；默认只清理当前 scope")
    _add_scope_args(prune)
    sub.add_parser("presets", help="打印可用探索 scope 预设")
    sub.add_parser("submit-approved", help="提交已通过 guard 的候选")
    check_ai = sub.add_parser("check-ai", help="只生成一个 AI 候选，不做 BRAIN 回测")
    _add_scope_args(check_ai)
    fields = sub.add_parser("fields", help="打印指定 scope 的字段池")
    fields.add_argument("--limit", type=int, default=40)
    _add_scope_args(fields)
    plan_next = sub.add_parser("plan-next", help="预览下一轮 throughput scheduler 计划")
    plan_next.add_argument("--batch-size", type=int, default=None)
    _add_scope_args(plan_next)
    web = sub.add_parser("web", help="启动本地 Web 控制台")
    # Defaults are None so we can distinguish an explicit flag from "fall back to
    # ALPHA_WEB_HOST/ALPHA_WEB_PORT env, then the safe loopback default".
    web.add_argument("--host", default=None)
    web.add_argument("--port", type=int, default=None)
    return parser


def _add_scope_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--preset", choices=sorted(SCOPE_PRESETS), default=None)
    parser.add_argument("--region", default=None)
    parser.add_argument("--universe", default=None)
    parser.add_argument("--delay", type=int, default=None)
    parser.add_argument("--neutralization", default=None)
    parser.add_argument("--decay", type=int, default=None)
    parser.add_argument("--truncation", type=float, default=None)


def _apply_generator_mode_env(args: argparse.Namespace) -> None:
    mode = str(getattr(args, "generator_mode", "") or "").strip().lower()
    orchestration_mode = str(getattr(args, "orchestration_mode", "") or "").strip().lower()
    if orchestration_mode:
        os.environ["AI_ORCHESTRATION_MODE"] = orchestration_mode
    elif mode:
        os.environ["AI_ORCHESTRATION_MODE"] = "lean"
    if not mode:
        return
    if mode == "balanced":
        os.environ["AI_MAX_ACTIVE_GENERATORS"] = "0"
        _ensure_min_float_env("AI_GENERATION_STAGE_TIMEOUT_SECONDS", 180.0)
    elif mode == "single":
        os.environ["AI_MAX_ACTIVE_GENERATORS"] = "1"


def _apply_auto_submit_env(args: argparse.Namespace) -> None:
    auto_submit = getattr(args, "auto_submit", None)
    if auto_submit is None:
        return
    os.environ["AUTO_SUBMIT"] = "true" if bool(auto_submit) else "false"


def _ensure_min_float_env(name: str, minimum: float) -> None:
    raw = os.environ.get(name)
    try:
        value = float(raw) if raw not in {None, ""} else 0.0
    except (TypeError, ValueError):
        value = 0.0
    if value < minimum:
        os.environ[name] = str(int(minimum) if float(minimum).is_integer() else minimum)


def _int_env_cli(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or str(raw).strip() == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be an integer") from exc


def _simulation_context_from_args(base: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    overrides = {
        "region": getattr(args, "region", None),
        "universe": getattr(args, "universe", None),
        "delay": getattr(args, "delay", None),
        "neutralization": getattr(args, "neutralization", None),
        "decay": getattr(args, "decay", None),
        "truncation": getattr(args, "truncation", None),
    }
    return apply_scope(base, preset=getattr(args, "preset", None), overrides=overrides)


def _build_ai_client(name: str):
    if name == "local":
        return LocalAIClient()
    if name in {"openai", "openai_compatible"}:
        return OpenAICompatibleAIClient.from_env()
    if name in {"multi", "model_pool", "orchestrated"}:
        return MultiModelAIClient.from_env()
    raise RuntimeError(f"unsupported AI_CLIENT={name}")


def _build_brain_client(name: str):
    if name == "local":
        return LocalBrainClient()
    if name in {"http", "brain_http", "live"}:
        return BrainHTTPClient.from_env()
    raise RuntimeError(f"unsupported BRAIN_CLIENT={name}")


def _worker(
    store: AlphaStore,
    batch_size: int,
    policy,
    ai_client_name: str,
    brain_client_name: str,
    simulation_context,
) -> AlphaWorker:
    return AlphaWorker(
        store=store,
        ai_client=_build_ai_client(ai_client_name),
        brain_client=_build_brain_client(brain_client_name),
        policy=policy,
        batch_size=batch_size,
        context=simulation_context,
    )


def main(argv: Optional[List[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    load_env_file(args.env_file, override=True)
    _apply_generator_mode_env(args)
    _apply_auto_submit_env(args)
    setup_logging(args.log_file)
    log = logging.getLogger("alpha.cli")
    cfg = load_config(db_path=args.db, batch_size=getattr(args, "batch_size", None))
    simulation_context = _simulation_context_from_args(cfg.simulation_context, args)
    store = AlphaStore(cfg.db_path)

    if args.command == "init-db":
        store.init()
        log.info("database initialized path=%s", cfg.db_path)
        print(f"initialized {cfg.db_path}")
        return 0

    store.init()

    if args.command == "run-once":
        log.info("run_once simulation_context=%s", simulation_context)
        summary = _worker(store, cfg.batch_size, cfg.policy, cfg.ai_client, cfg.brain_client, simulation_context).run_once()
        log.info("run_once summary=%s", summary)
        print(summary)
        return 0

    if args.command == "daemon":
        loop_seconds = args.loop_seconds if args.loop_seconds is not None else cfg.loop_seconds
        run_minutes = float(args.run_minutes or 0.0)
        deadline = time.monotonic() + run_minutes * 60.0 if run_minutes > 0 else None
        scope_rotation = parse_scope_json(cfg.simulation_context, args.scope_json)
        _mark_daemon_started(
            store,
            args,
            simulation_context,
            batch_size=cfg.batch_size,
            loop_seconds=loop_seconds,
            run_minutes=run_minutes,
            scope_rotation=scope_rotation,
            argv=argv,
        )
        try:
            while True:
                if deadline is not None and time.monotonic() >= deadline:
                    _mark_daemon_stopped(store, "time_limit")
                    log.info("daemon time limit reached run_minutes=%s", run_minutes)
                    print("time_limit_reached", flush=True)
                    return 0
                cycle_context = next_rotating_scope(store, scope_rotation) if scope_rotation else simulation_context
                cycle_plan = None
                cycle_batch_size = cfg.batch_size
                try:
                    if getattr(args, "throughput_mode", False):
                        cycle_plan = build_cycle_plan(store, cycle_context, batch_size=cfg.batch_size)
                        cycle_context = dict(cycle_context)
                        if isinstance(cycle_plan.get("scope"), dict):
                            cycle_context.update(cycle_plan["scope"])
                        budget = cycle_plan.get("budget") if isinstance(cycle_plan.get("budget"), dict) else {}
                        try:
                            cycle_batch_size = int(budget.get("batch_size") or cfg.batch_size)
                        except (TypeError, ValueError):
                            cycle_batch_size = cfg.batch_size
                        log.info("daemon_cycle_plan=%s", cycle_plan)
                    log.info("daemon_cycle simulation_context=%s", cycle_context)
                    worker = _worker(store, cycle_batch_size, cfg.policy, cfg.ai_client, cfg.brain_client, cycle_context)
                    summary = worker.run_once(cycle_plan=cycle_plan) if cycle_plan is not None else worker.run_once()
                    log.info("daemon_cycle summary=%s", summary)
                    print(summary, flush=True)
                except Exception as exc:
                    # One failing cycle (transient BRAIN/AI/sqlite error) must not crash
                    # the daemon; log, record, back off, and continue with the next cycle.
                    log.exception("daemon cycle failed; backing off and continuing error=%s", exc)
                    try:
                        store.record_event(None, "daemon_cycle_error", {"error": str(exc)})
                    except Exception:
                        log.exception("failed to record daemon_cycle_error event")
                    print("cycle_error", flush=True)
                    backoff = loop_seconds
                    if deadline is not None:
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            _mark_daemon_stopped(store, "time_limit")
                            log.info("daemon time limit reached run_minutes=%s", run_minutes)
                            print("time_limit_reached", flush=True)
                            return 0
                        backoff = min(backoff, remaining)
                    time.sleep(backoff)
                    continue
                ai_timeout_backoff = bool(summary.get("ai_generation_timeout"))
                ai_network_backoff = bool(summary.get("ai_network_blocked"))
                if summary.get("ai_quota_blocked") or summary.get("ai_config_blocked"):
                    if summary.get("ai_quota_blocked"):
                        reason = "ai_quota_blocked"
                    else:
                        reason = "ai_config_blocked"
                    _mark_daemon_stopped(store, reason)
                    log.warning("daemon stopped reason=%s", reason)
                    print(reason, flush=True)
                    return 0
                if ai_timeout_backoff:
                    log.warning("daemon ai_generation_timeout summary=%s; backing off and continuing", summary)
                    print("ai_generation_timeout_backoff", flush=True)
                if ai_network_backoff:
                    log.warning("daemon ai_network_blocked summary=%s; backing off and continuing", summary)
                    print("ai_network_blocked_backoff", flush=True)
                quality_exhausted_reason = _production_rescue_quality_exhausted_reason(summary, cycle_plan)
                if quality_exhausted_reason:
                    log.warning(
                        "daemon route exhausted reason=%s summary=%s; continuing",
                        quality_exhausted_reason,
                        summary,
                    )
                    print(quality_exhausted_reason, flush=True)
                quality_stop_reason = _production_rescue_quality_stop_reason(summary, cycle_plan)
                if quality_stop_reason:
                    _mark_daemon_stopped(store, quality_stop_reason)
                    log.warning("daemon stopped reason=%s summary=%s", quality_stop_reason, summary)
                    print(quality_stop_reason, flush=True)
                    return 0
                optimize_exhausted_reason = _optimize_quality_exhausted_reason(summary, cycle_plan)
                if optimize_exhausted_reason:
                    log.warning(
                        "daemon route exhausted reason=%s summary=%s; continuing",
                        optimize_exhausted_reason,
                        summary,
                    )
                    print(optimize_exhausted_reason, flush=True)
                probe_error_reason = _production_rescue_probe_error_stop_reason(summary, cycle_plan)
                if probe_error_reason:
                    log.warning(
                        "daemon route exhausted reason=%s summary=%s; continuing",
                        probe_error_reason,
                        summary,
                    )
                    print(probe_error_reason, flush=True)
                duplicate_only_reason = _production_rescue_duplicate_only_stop_reason(summary, cycle_plan)
                if duplicate_only_reason:
                    log.warning("daemon route exhausted reason=%s summary=%s; continuing", duplicate_only_reason, summary)
                    print(duplicate_only_reason, flush=True)
                explore_duplicate_only_reason = _explore_duplicate_only_stop_reason(summary, cycle_plan)
                if explore_duplicate_only_reason:
                    log.warning(
                        "daemon route exhausted reason=%s summary=%s; continuing",
                        explore_duplicate_only_reason,
                        summary,
                    )
                    print(explore_duplicate_only_reason, flush=True)
                if summary.get("quality_stop_loss"):
                    log.warning("daemon quality_stop_loss summary=%s; continuing", summary)
                sleep_seconds = (
                    max(loop_seconds, _ai_generation_timeout_backoff_seconds())
                    if (ai_timeout_backoff or ai_network_backoff)
                    else loop_seconds
                )
                if deadline is not None:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        _mark_daemon_stopped(store, "time_limit")
                        log.info("daemon time limit reached run_minutes=%s", run_minutes)
                        print("time_limit_reached", flush=True)
                        return 0
                    sleep_seconds = min(sleep_seconds, remaining)
                time.sleep(sleep_seconds)
        except KeyboardInterrupt:
            _mark_daemon_stopped(store, "interrupted")
            log.info("daemon stopped")
            print("stopped")
            return 0

    if args.command == "plan-next":
        plan = build_cycle_plan(store, simulation_context, batch_size=cfg.batch_size)
        log.info("plan_next=%s", plan)
        print(json.dumps(plan, sort_keys=True))
        return 0

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

    if args.command == "prune-history":
        target = None if args.all_scopes else simulation_context
        summary = prune_low_quality_history(
            store,
            target,
            quality_max=args.quality_max,
            limit=args.limit,
            execute=bool(args.execute),
        )
        log.info("prune_history summary=%s", summary)
        print(summary)
        return 0

    if args.command == "presets":
        for name, preset in preset_rows():
            print(
                f"{name}: region={preset['region']} universe={preset['universe']} "
                f"delay={preset['delay']} neutralization={preset['neutralization']}"
            )
        return 0

    if args.command == "check-ai":
        ai_client = _build_ai_client(cfg.ai_client)
        brain_client = _build_brain_client(cfg.brain_client)
        target_context = dict(simulation_context)
        field_catalog = build_field_catalog(brain_client, target_context)
        target_context["research_context"] = build_ai_research_context(
            store,
            target_context,
            field_catalog=field_catalog,
        )
        candidates = ai_client.generate_candidates(1, target_context)
        for candidate in candidates:
            print(candidate.expression)
        log.info("check_ai generated=%s", len(candidates))
        return 0

    if args.command == "fields":
        brain_client = _build_brain_client(cfg.brain_client)
        catalog = build_field_catalog(brain_client, simulation_context)
        print(
            "scope: "
            f"region={simulation_context.get('region')} universe={simulation_context.get('universe')} "
            f"delay={simulation_context.get('delay')}"
        )
        print(f"available: {catalog.get('available')} source: {catalog.get('source', 'unknown')}")
        if catalog.get("error"):
            print(f"error: {catalog['error']}")
        datasets = catalog.get("datasets") if isinstance(catalog.get("datasets"), list) else []
        if datasets:
            print("datasets:")
            for dataset in datasets[:10]:
                print(f"  {dataset.get('id')}: {dataset.get('field_count')} fields")
        scout = build_field_scout(catalog)
        top_fields = scout.get("top_fields") if isinstance(scout.get("top_fields"), list) else []
        if top_fields:
            print("field_scout:")
            for row in top_fields[: min(args.limit, 20)]:
                print(
                    f"  {row.get('field')}: score={row.get('score')} "
                    f"category={row.get('category')} policy={row.get('primary_policy')}"
                )
        print("field_ids:")
        for field_id in (catalog.get("field_ids") or [])[: args.limit]:
            print(f"  {field_id}")
        return 0

    if args.command == "web":
        from .web import run_web_app

        # Precedence: explicit --host/--port flag > ALPHA_WEB_HOST/PORT env > safe default.
        host = args.host if args.host is not None else os.getenv("ALPHA_WEB_HOST", "127.0.0.1")
        port = args.port if args.port is not None else _int_env_cli("ALPHA_WEB_PORT", 8080)
        return run_web_app(
            db_path=cfg.db_path,
            env_file=args.env_file,
            log_file=args.log_file,
            host=host,
            port=port,
        )

    if args.command == "submit-approved":
        brain_client = _build_brain_client(cfg.brain_client)
        summary = submit_approved_candidates(store, brain_client, cfg.policy)
        log.info("submit_approved summary=%s", summary)
        print(summary)
        return 0

    parser.error(f"unknown command: {args.command}")
    return 2


def _mark_daemon_started(
    store: AlphaStore,
    args: argparse.Namespace,
    scope: Dict[str, Any],
    *,
    batch_size: int,
    loop_seconds: float,
    run_minutes: float,
    scope_rotation: List[Dict[str, Any]],
    argv: Optional[List[str]],
) -> None:
    started_at = utc_now()
    state = {
        "status": "running",
        "pid": os.getpid(),
        "started_at": started_at,
        "stopped_at": "",
        "stop_reason": "",
        "scope": {
            "region": scope.get("region", "USA"),
            "universe": scope.get("universe", "TOP3000"),
            "delay": scope.get("delay", 1),
            "neutralization": scope.get("neutralization", "INDUSTRY"),
        },
        "scope_rotation": scope_rotation,
        "preset": str(getattr(args, "preset", None) or ""),
        "generator_mode": str(getattr(args, "generator_mode", None) or ""),
        "orchestration_mode": str(
            getattr(args, "orchestration_mode", None) or os.environ.get("AI_ORCHESTRATION_MODE", "")
        ),
        "auto_submit": bool(os.environ.get("AUTO_SUBMIT", "").strip().lower() in {"1", "true", "yes", "on"}),
        "throughput_mode": bool(getattr(args, "throughput_mode", False)),
        "batch_size": int(batch_size),
        "loop_seconds": float(loop_seconds),
        "run_minutes": float(run_minutes),
        "stop_after_at": _daemon_stop_after_at(started_at, run_minutes),
        "argv": [sys.executable, "-m", "alpha.cli", *(list(argv) if argv is not None else sys.argv[1:])],
    }
    store.set_run_state("daemon", state)
    store.record_event(None, "cli_daemon_started", state)


def _daemon_stop_after_at(started_at: str, run_minutes: float) -> str:
    if run_minutes <= 0:
        return ""
    try:
        start = datetime.fromisoformat(started_at)
    except ValueError:
        start = datetime.now(timezone.utc).replace(microsecond=0)
    return (start + timedelta(minutes=float(run_minutes))).replace(microsecond=0).isoformat()


def _mark_daemon_stopped(store: AlphaStore, reason: str) -> None:
    state = store.get_run_state("daemon")
    pid = int(state.get("pid") or 0)
    interrupted_preflight = store.fail_preflight_passed_candidates(
        created_since=str(state.get("started_at") or "").strip() or None,
        reason="interrupted_after_preflight",
    )
    if pid in {0, os.getpid()}:
        state.update({"status": "stopped", "stopped_at": utc_now(), "stop_reason": reason})
        store.set_run_state("daemon", state)
    store.record_event(
        None,
        "daemon_stopped",
        {"pid": os.getpid(), "reason": reason, "interrupted_preflight": interrupted_preflight},
    )


def _production_rescue_quality_stop_reason(summary: Dict[str, Any], cycle_plan: Dict[str, Any] | None) -> str:
    if not summary.get("quality_stop_loss"):
        return ""
    if not isinstance(cycle_plan, dict):
        return ""
    return ""


def _optimize_quality_exhausted_reason(summary: Dict[str, Any], cycle_plan: Dict[str, Any] | None) -> str:
    if not summary.get("quality_stop_loss"):
        return ""
    if not isinstance(cycle_plan, dict):
        return ""
    if str(cycle_plan.get("mode") or "") != "optimize":
        return ""
    has_probe_signal = any(_summary_count(summary, key) > 0 for key in ("probe_watch", "probe_optimize_ready", "probe_sweep_ready"))
    if has_probe_signal:
        return ""
    return "optimize_quality_stop_loss"


def _production_rescue_quality_exhausted_reason(summary: Dict[str, Any], cycle_plan: Dict[str, Any] | None) -> str:
    if not summary.get("quality_stop_loss"):
        return ""
    if not isinstance(cycle_plan, dict):
        return ""
    if str(cycle_plan.get("mode") or "") != "production_rescue":
        return ""
    has_probe_signal = any(_summary_count(summary, key) > 0 for key in ("probe_watch", "probe_optimize_ready", "probe_sweep_ready"))
    if has_probe_signal:
        return ""
    return "production_rescue_quality_stop_loss"


def _production_rescue_probe_error_stop_reason(summary: Dict[str, Any], cycle_plan: Dict[str, Any] | None) -> str:
    if _summary_count(summary, "probe_simulation_error") <= 0:
        return ""
    if not isinstance(cycle_plan, dict):
        return ""
    if str(cycle_plan.get("mode") or "") != "production_rescue":
        return ""
    return "production_rescue_probe_simulation_error"


def _production_rescue_duplicate_only_stop_reason(summary: Dict[str, Any], cycle_plan: Dict[str, Any] | None) -> str:
    if not isinstance(cycle_plan, dict):
        return ""
    if str(cycle_plan.get("mode") or "") != "production_rescue":
        return ""
    if _summary_count(summary, "skipped") <= 0:
        return ""
    productive_keys = ("generated", "approved", "submitted", "failed", "pending")
    if any(_summary_count(summary, key) > 0 for key in productive_keys):
        return ""
    return "production_rescue_duplicate_only"


def _explore_duplicate_only_stop_reason(summary: Dict[str, Any], cycle_plan: Dict[str, Any] | None) -> str:
    if not isinstance(cycle_plan, dict):
        return ""
    if str(cycle_plan.get("mode") or "") != "explore":
        return ""
    if _summary_count(summary, "standardized_probe_exhausted") > 0:
        return ""
    if _summary_count(summary, "production_rescue_probe_exhausted") > 0:
        return ""
    if _summary_count(summary, "field_scout_blocked") > 0:
        return "field_scout_blocked"
    if _summary_count(summary, "skipped") <= 0:
        return ""
    productive_keys = ("generated", "approved", "submitted", "failed", "pending")
    if any(_summary_count(summary, key) > 0 for key in productive_keys):
        return ""
    return "explore_duplicate_only"


def _summary_count(summary: Dict[str, Any], key: str) -> int:
    try:
        return int(summary.get(key) or 0)
    except (TypeError, ValueError):
        return 0


def _ai_generation_timeout_backoff_seconds() -> float:
    try:
        return max(1.0, float(os.environ.get("AI_GENERATION_TIMEOUT_BACKOFF_SECONDS", "600")))
    except (TypeError, ValueError):
        return 600.0


if __name__ == "__main__":
    raise SystemExit(main())
