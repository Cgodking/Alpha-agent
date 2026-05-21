from __future__ import annotations

import argparse
import logging
import os
import time
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
from .scopes import SCOPE_PRESETS, apply_scope, preset_rows
from .scope_rotation import next_rotating_scope, parse_scope_json
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
    _add_scope_args(daemon)

    sub.add_parser("status", help="打印候选状态统计")
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
    web = sub.add_parser("web", help="启动本地 Web 控制台")
    web.add_argument("--host", default="0.0.0.0")
    web.add_argument("--port", type=int, default=8080)
    return parser


def _add_scope_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--preset", choices=sorted(SCOPE_PRESETS), default=None)
    parser.add_argument("--region", default=None)
    parser.add_argument("--universe", default=None)
    parser.add_argument("--delay", type=int, default=None)
    parser.add_argument("--neutralization", default=None)
    parser.add_argument("--decay", type=int, default=None)
    parser.add_argument("--truncation", type=float, default=None)


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
        try:
            while True:
                if deadline is not None and time.monotonic() >= deadline:
                    _mark_daemon_stopped(store, "time_limit")
                    log.info("daemon time limit reached run_minutes=%s", run_minutes)
                    print("time_limit_reached", flush=True)
                    return 0
                cycle_context = next_rotating_scope(store, scope_rotation) if scope_rotation else simulation_context
                log.info("daemon_cycle simulation_context=%s", cycle_context)
                summary = _worker(store, cfg.batch_size, cfg.policy, cfg.ai_client, cfg.brain_client, cycle_context).run_once()
                log.info("daemon_cycle summary=%s", summary)
                print(summary, flush=True)
                if summary.get("ai_quota_blocked") or summary.get("ai_config_blocked"):
                    reason = "ai_quota_blocked" if summary.get("ai_quota_blocked") else "ai_config_blocked"
                    _mark_daemon_stopped(store, reason)
                    log.warning("daemon stopped reason=%s", reason)
                    print(reason, flush=True)
                    return 0
                sleep_seconds = loop_seconds
                if deadline is not None:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        _mark_daemon_stopped(store, "time_limit")
                        log.info("daemon time limit reached run_minutes=%s", run_minutes)
                        print("time_limit_reached", flush=True)
                        return 0
                    sleep_seconds = min(loop_seconds, remaining)
                time.sleep(sleep_seconds)
        except KeyboardInterrupt:
            _mark_daemon_stopped(store, "interrupted")
            log.info("daemon stopped")
            print("stopped")
            return 0

    if args.command == "status":
        counts = store.status_counts()
        log.info("status counts=%s", counts)
        if not counts:
            print("no candidates")
        for status, count in counts.items():
            print(f"{status}: {count}")
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

        return run_web_app(
            db_path=cfg.db_path,
            env_file=args.env_file,
            log_file=args.log_file,
            host=args.host,
            port=args.port,
        )

    if args.command == "submit-approved":
        brain_client = _build_brain_client(cfg.brain_client)
        summary = submit_approved_candidates(store, brain_client, cfg.policy)
        log.info("submit_approved summary=%s", summary)
        print(summary)
        return 0

    parser.error(f"unknown command: {args.command}")
    return 2


def _mark_daemon_stopped(store: AlphaStore, reason: str) -> None:
    state = store.get_run_state("daemon")
    pid = int(state.get("pid") or 0)
    if pid in {0, os.getpid()}:
        state.update({"status": "stopped", "stopped_at": utc_now(), "stop_reason": reason})
        store.set_run_state("daemon", state)
    store.record_event(None, "daemon_stopped", {"pid": os.getpid(), "reason": reason})


if __name__ == "__main__":
    raise SystemExit(main())
