from __future__ import annotations

import json
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

from alpha.db import AlphaStore
from alpha.web import ControlService, HTML, build_daemon_argv, is_pid_running, scope_options, tail_file


class WebControlTests(unittest.TestCase):
    def test_build_daemon_argv_includes_scope_and_runtime_args(self):
        argv = build_daemon_argv(
            db_path=Path("alpha.db"),
            env_file=Path(".env"),
            log_file=Path("logs/alpha.log"),
            preset="chn-d0",
            region="",
            universe="",
            delay=None,
            neutralization="",
            batch_size=3,
            loop_seconds=20,
        )

        self.assertIn("-m", argv)
        self.assertIn("alpha.cli", argv)
        self.assertIn("daemon", argv)
        self.assertIn("--preset", argv)
        self.assertIn("chn-d0", argv)
        self.assertIn("--batch-size", argv)
        self.assertIn("3", argv)
        self.assertIn("--loop-seconds", argv)
        self.assertIn("20", argv)

    def test_build_daemon_argv_includes_run_minutes_limit(self):
        argv = build_daemon_argv(
            db_path=Path("alpha.db"),
            env_file=Path(".env"),
            log_file=Path("logs/alpha.log"),
            preset="ind",
            batch_size=8,
            loop_seconds=60,
            run_minutes=240,
        )

        self.assertIn("--run-minutes", argv)
        self.assertEqual(argv[argv.index("--run-minutes") + 1], "240")

    def test_build_daemon_argv_accepts_custom_scope_without_preset(self):
        argv = build_daemon_argv(
            db_path=Path("alpha.db"),
            env_file=Path(".env"),
            log_file=Path("logs/alpha.log"),
            preset="",
            region="USA",
            universe="TOP2000",
            delay=0,
            neutralization="SUBINDUSTRY",
            batch_size=8,
            loop_seconds=30,
        )

        self.assertNotIn("--preset", argv)
        self.assertIn("--region", argv)
        self.assertIn("USA", argv)
        self.assertIn("--universe", argv)
        self.assertIn("TOP2000", argv)
        self.assertIn("--delay", argv)
        self.assertIn("0", argv)
        self.assertIn("--neutralization", argv)
        self.assertIn("SUBINDUSTRY", argv)

    def test_build_daemon_argv_accepts_scope_rotation(self):
        scope_rotation = [
            {"region": "USA", "universe": "TOP3000", "delay": 0, "neutralization": "INDUSTRY"},
            {"region": "CHN", "universe": "TOP2000U", "delay": 0, "neutralization": "INDUSTRY"},
        ]

        argv = build_daemon_argv(
            db_path=Path("alpha.db"),
            env_file=Path(".env"),
            log_file=Path("logs/alpha.log"),
            scope_rotation=scope_rotation,
            batch_size=8,
            loop_seconds=30,
        )

        self.assertIn("--scope-json", argv)
        encoded = argv[argv.index("--scope-json") + 1]
        self.assertEqual(json.loads(encoded), scope_rotation)
        self.assertNotIn("--region", argv)

    def test_scope_options_include_region_universe_delay_choices(self):
        options = scope_options()
        usa = [item for item in options["scopes"] if item["region"] == "USA" and item["delay"] == 0][0]
        ind = [item for item in options["scopes"] if item["region"] == "IND"][0]

        self.assertIn("TOP2000", usa["universes"])
        self.assertIn("TOP500", ind["universes"])
        self.assertIn("IND", options["regions"])

    def test_is_pid_running_treats_zombie_process_as_stopped(self):
        process = subprocess.Popen([sys.executable, "-c", "pass"])
        try:
            stat_path = Path(f"/proc/{process.pid}/stat")
            for _ in range(50):
                if stat_path.exists() and stat_path.read_text(encoding="utf-8").split()[2] == "Z":
                    break
                time.sleep(0.02)

            self.assertFalse(is_pid_running(process.pid))
        finally:
            process.wait(timeout=2)

    def test_control_panel_exposes_custom_scope_controls_and_batch_limit(self):
        self.assertIn('id="region"', HTML)
        self.assertIn('id="delay"', HTML)
        self.assertIn('id="universe"', HTML)
        self.assertIn('id="neutralization"', HTML)
        self.assertIn('id="rotate_scopes"', HTML)
        self.assertIn('id="scope_rotation"', HTML)
        self.assertIn('id="batch_size" type="number" value="8" min="1" max="8"', HTML)
        self.assertIn('id="run_minutes"', HTML)
        self.assertIn('<option value="120">2 小时</option>', HTML)
        self.assertIn('<option value="240" selected>4 小时</option>', HTML)
        self.assertIn('<option value="custom">自定义</option>', HTML)
        self.assertIn('id="run_minutes_custom" type="number"', HTML)
        self.assertIn("function selectedRunMinutes()", HTML)
        self.assertIn('id="model_scores"', HTML)
        self.assertIn("/api/scope-options", HTML)

    def test_control_panel_exposes_log_clear_controls(self):
        self.assertIn('onclick="clearLogs(false)"', HTML)
        self.assertIn('onclick="clearLogs(true)"', HTML)
        self.assertIn("/api/clear-logs", HTML)
        self.assertIn('<option value="web">web.log</option>', HTML)

    def test_control_panel_exposes_candidate_queue_panel(self):
        self.assertIn('id="candidate_queues"', HTML)
        self.assertIn('id="queue_submitable"', HTML)
        self.assertIn('renderCandidateQueues', HTML)

    def test_tail_file_returns_last_lines(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "alpha.log"
            path.write_text("one\ntwo\nthree\n", encoding="utf-8")

            self.assertEqual(tail_file(path, line_count=2), "two\nthree\n")

    def test_control_service_clear_logs_truncates_selected_log(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            store = AlphaStore(base / "alpha.db")
            store.init()
            alpha_log = base / "alpha.log"
            daemon_log = base / "daemon.log"
            web_log = base / "web.log"
            alpha_log.write_text("alpha\n", encoding="utf-8")
            daemon_log.write_text("daemon\n", encoding="utf-8")
            web_log.write_text("web\n", encoding="utf-8")
            service = ControlService(
                store=store,
                db_path=store.path,
                env_file=base / ".env",
                log_file=alpha_log,
                daemon_stdout_log=daemon_log,
                web_log=web_log,
            )

            result = service.clear_logs({"file": "alpha"})

            self.assertTrue(result["success"])
            self.assertEqual(alpha_log.read_text(encoding="utf-8"), "")
            self.assertEqual(daemon_log.read_text(encoding="utf-8"), "daemon\n")
            self.assertEqual(web_log.read_text(encoding="utf-8"), "web\n")
            self.assertEqual(result["cleared"], ["alpha"])

    def test_control_service_clear_logs_can_clear_all_known_logs(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            store = AlphaStore(base / "alpha.db")
            store.init()
            alpha_log = base / "alpha.log"
            daemon_log = base / "daemon.log"
            web_log = base / "web.log"
            for path in [alpha_log, daemon_log, web_log]:
                path.write_text("old\n", encoding="utf-8")
            service = ControlService(
                store=store,
                db_path=store.path,
                env_file=base / ".env",
                log_file=alpha_log,
                daemon_stdout_log=daemon_log,
                web_log=web_log,
            )

            result = service.clear_logs({"all": True})

            self.assertTrue(result["success"])
            self.assertEqual(set(result["cleared"]), {"alpha", "daemon", "web"})
            self.assertTrue(all(path.read_text(encoding="utf-8") == "" for path in [alpha_log, daemon_log, web_log]))

    def test_control_service_start_records_daemon_state(self):
        class FakeProcess:
            pid = 4321

        calls = []

        def fake_popen(argv, **kwargs):
            calls.append((argv, kwargs))
            return FakeProcess()

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
                popen_factory=fake_popen,
                pid_running=lambda _pid: False,
            )

            result = service.start({"preset": "ind", "batch_size": 2, "loop_seconds": 5, "run_minutes": 120})

            state = store.get_run_state("daemon")
            self.assertTrue(result["success"])
            self.assertEqual(state["pid"], 4321)
            self.assertEqual(state["preset"], "ind")
            self.assertEqual(state["batch_size"], 2)
            self.assertEqual(state["loop_seconds"], 5)
            self.assertEqual(state["run_minutes"], 120)
            self.assertIn("--run-minutes", calls[0][0])
            self.assertIn("120", calls[0][0])
            self.assertIn("ind", calls[0][0])

    def test_control_service_start_caps_batch_at_platform_batch_limit_and_records_custom_scope(self):
        class FakeProcess:
            pid = 9876

        calls = []

        def fake_popen(argv, **kwargs):
            calls.append((argv, kwargs))
            return FakeProcess()

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
                popen_factory=fake_popen,
                pid_running=lambda _pid: False,
            )

            result = service.start(
                {
                    "region": "USA",
                    "universe": "TOP2000",
                    "delay": 0,
                    "neutralization": "SUBINDUSTRY",
                    "batch_size": 99,
                    "loop_seconds": 5,
                }
            )

            state = store.get_run_state("daemon")
            argv = calls[0][0]
            self.assertTrue(result["success"])
            self.assertEqual(state["batch_size"], 8)
            self.assertEqual(state["scope"]["region"], "USA")
            self.assertEqual(state["scope"]["universe"], "TOP2000")
            self.assertEqual(state["scope"]["delay"], 0)
            self.assertIn("--batch-size", argv)
            self.assertIn("8", argv)
            self.assertIn("--region", argv)
            self.assertIn("USA", argv)
            self.assertIn("--universe", argv)
            self.assertIn("TOP2000", argv)

    def test_control_service_start_records_scope_rotation(self):
        class FakeProcess:
            pid = 2468

        calls = []

        def fake_popen(argv, **kwargs):
            calls.append((argv, kwargs))
            return FakeProcess()

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
                popen_factory=fake_popen,
                pid_running=lambda _pid: False,
            )

            result = service.start(
                {
                    "scope_rotation": [
                        {"region": "USA", "universe": "TOP3000", "delay": 0, "neutralization": "INDUSTRY"},
                        {"region": "CHN", "universe": "TOP2000U", "delay": 0, "neutralization": "INDUSTRY"},
                    ],
                    "batch_size": 8,
                    "loop_seconds": 5,
                }
            )

            state = store.get_run_state("daemon")
            argv = calls[0][0]
            self.assertTrue(result["success"])
            self.assertEqual(len(state["scope_rotation"]), 2)
            self.assertEqual(state["scope"]["region"], "USA")
            self.assertIn("--scope-json", argv)

    def test_control_service_start_rejects_invalid_region_universe_delay_combination(self):
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
                pid_running=lambda _pid: False,
            )

            with self.assertRaises(ValueError):
                service.start(
                    {
                        "region": "IND",
                        "universe": "TOP3000",
                        "delay": 0,
                        "neutralization": "INDUSTRY",
                        "batch_size": 8,
                    }
                )

    def test_control_service_stop_sends_interrupt_then_marks_stopped(self):
        killed = []
        running = {"value": True}

        def fake_kill(pid, sig):
            killed.append((pid, sig))
            running["value"] = False

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            store = AlphaStore(base / "alpha.db")
            store.init()
            store.set_run_state("daemon", {"pid": 4321, "preset": "ind"})
            service = ControlService(
                store=store,
                db_path=store.path,
                env_file=base / ".env",
                log_file=base / "alpha.log",
                daemon_stdout_log=base / "daemon.log",
                pid_running=lambda _pid: running["value"],
                kill_func=fake_kill,
                sleep_func=lambda _seconds: None,
            )

            result = service.stop()

            state = store.get_run_state("daemon")
            self.assertTrue(result["success"])
            self.assertEqual(killed, [(4321, signal.SIGINT)])
            self.assertEqual(state["status"], "stopped")

    def test_control_service_status_includes_counts_and_recent_candidates(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            store = AlphaStore(base / "alpha.db")
            store.init()
            candidate_id = store.insert_candidate("rank(close)", {"region": "USA"}, "local_ai")
            store.transition(candidate_id, "approved")
            store.update_candidate(candidate_id, metrics_json=json.dumps({"sharpe": 2.0}))

            service = ControlService(
                store=store,
                db_path=store.path,
                env_file=base / ".env",
                log_file=base / "alpha.log",
                daemon_stdout_log=base / "daemon.log",
                pid_running=lambda _pid: False,
            )

            status = service.status()

            self.assertFalse(status["running"])
            self.assertEqual(status["counts"]["approved"], 1)
            self.assertEqual(status["recent_candidates"][0]["expression"], "rank(close)")
            self.assertEqual(status["recent_candidates"][0]["metrics"]["sharpe"], 2.0)

    def test_control_service_status_includes_model_scores(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            store = AlphaStore(base / "alpha.db")
            store.init()
            gemini_id = store.insert_candidate(
                "group_rank(ts_rank(ts_delta(close,5),22),industry)",
                {"region": "USA"},
                "model:gemini",
            )
            store.update_candidate(gemini_id, metrics_json=json.dumps({"sharpe": 1.3, "fitness": 0.7}))
            store.transition(gemini_id, "failed")
            glm_id = store.insert_candidate(
                "group_rank(ts_rank(ts_delta(close,9),22),industry)",
                {"region": "USA"},
                "model:glm",
            )
            store.update_candidate(glm_id, metrics_json=json.dumps({"sharpe": 1.7, "fitness": 0.9}))
            store.transition(glm_id, "approved")

            service = ControlService(
                store=store,
                db_path=store.path,
                env_file=base / ".env",
                log_file=base / "alpha.log",
                daemon_stdout_log=base / "daemon.log",
                pid_running=lambda _pid: False,
            )

            status = service.status()

            self.assertEqual(status["model_scores"]["model:gemini"]["generated"], 1)
            self.assertEqual(status["model_scores"]["model:gemini"]["failed"], 1)
            self.assertEqual(status["model_scores"]["model:glm"]["approved"], 1)
            self.assertEqual(status["model_scores"]["model:glm"]["best_sharpe"], 1.7)

    def test_control_service_status_uses_cache_within_ttl(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            store = AlphaStore(base / "alpha.db")
            store.init()
            candidate_id = store.insert_candidate("rank(close)", {"region": "USA"}, "local_ai")
            store.transition(candidate_id, "approved")

            service = ControlService(
                store=store,
                db_path=store.path,
                env_file=base / ".env",
                log_file=base / "alpha.log",
                daemon_stdout_log=base / "daemon.log",
                pid_running=lambda _pid: False,
                status_cache_ttl_seconds=60.0,
            )

            research_calls = {"count": 0}
            original_research_context = service._research_context

            def counted_research_context(state):
                research_calls["count"] += 1
                return original_research_context(state)

            service._research_context = counted_research_context

            first = service.status()
            second = service.status()

            self.assertEqual(research_calls["count"], 1)
            self.assertEqual(first["counts"], second["counts"])
            self.assertEqual(first["recent_candidates"], second["recent_candidates"])

    def test_control_service_reuses_research_context_after_status_cache_expires(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            store = AlphaStore(base / "alpha.db")
            store.init()
            candidate_id = store.insert_candidate(
                "group_rank(ts_rank(wait_signal, 63), industry)",
                {"region": "USA", "universe": "TOP3000", "delay": 0},
                "model:G-1",
            )
            store.update_candidate(candidate_id, metrics_json=json.dumps({"sharpe": 1.2, "fitness": 0.5}))
            store.transition(candidate_id, "failed", {"errors": ["LOW_SHARPE:FAIL"]})
            store.set_run_state(
                "daemon",
                {
                    "status": "stopped",
                    "scope": {"region": "USA", "universe": "TOP3000", "delay": 0, "neutralization": "INDUSTRY"},
                },
            )

            service = ControlService(
                store=store,
                db_path=store.path,
                env_file=base / ".env",
                log_file=base / "alpha.log",
                daemon_stdout_log=base / "daemon.log",
                pid_running=lambda _pid: False,
                status_cache_ttl_seconds=0.0,
                research_context_cache_ttl_seconds=60.0,
            )

            research_builds = {"count": 0}
            original_build = service._build_research_context

            def counted_build(scope):
                research_builds["count"] += 1
                return original_build(scope)

            service._build_research_context = counted_build

            first = service.status()
            second = service.status()

            self.assertEqual(research_builds["count"], 1)
            self.assertEqual(first["research_plan"], second["research_plan"])

    def test_control_service_status_scopes_main_counts_to_current_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            store = AlphaStore(base / "alpha.db")
            store.init()
            old_candidate_id = store.insert_candidate("rank(close)", {"region": "USA"}, "local_ai")
            store.transition(old_candidate_id, "failed")
            current_candidate_id = store.insert_candidate("rank(-returns)", {"region": "USA"}, "local_ai")
            store.transition(current_candidate_id, "approved")
            run_started_at = "2026-04-30T00:05:00+00:00"
            with store.connect() as conn:
                conn.execute(
                    "UPDATE candidates SET created_at = ?, updated_at = ? WHERE id = ?",
                    ("2026-04-30T00:00:00+00:00", "2026-04-30T00:00:00+00:00", old_candidate_id),
                )
                conn.execute(
                    "UPDATE candidates SET created_at = ?, updated_at = ? WHERE id = ?",
                    ("2026-04-30T00:10:00+00:00", "2026-04-30T00:10:00+00:00", current_candidate_id),
                )
            store.set_run_state(
                "daemon",
                {
                    "pid": 4321,
                    "status": "running",
                    "started_at": run_started_at,
                    "scope": {"region": "USA", "universe": "TOP3000", "delay": 0, "neutralization": "INDUSTRY"},
                },
            )

            service = ControlService(
                store=store,
                db_path=store.path,
                env_file=base / ".env",
                log_file=base / "alpha.log",
                daemon_stdout_log=base / "daemon.log",
                pid_running=lambda _pid: True,
            )

            status = service.status()

            self.assertEqual(status["counts"], {"approved": 1})
            self.assertEqual(status["history_counts"]["approved"], 1)
            self.assertEqual(status["history_counts"]["failed"], 1)
            self.assertEqual([row["id"] for row in status["recent_candidates"]], [current_candidate_id])
            self.assertEqual(status["run_started_at"], run_started_at)

    def test_control_service_status_includes_current_experiment_plan(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            store = AlphaStore(base / "alpha.db")
            store.init()
            candidate_id = store.insert_candidate(
                "rank(group_rank(winsorize(ts_mean(analyst_positive_sentiment_logit_presentation,30),std=3),industry))",
                {"region": "USA", "delay": 0},
                "openai_compatible",
            )
            store.update_candidate(candidate_id, metrics_json=json.dumps({"sharpe": 1.11, "fitness": 0.46}))
            store.transition(candidate_id, "check_pending", {"errors": ["SHARPE_BELOW_MIN:1.110<1.58"]})

            service = ControlService(
                store=store,
                db_path=store.path,
                env_file=base / ".env",
                log_file=base / "alpha.log",
                daemon_stdout_log=base / "daemon.log",
                pid_running=lambda _pid: False,
            )

            status = service.status()

            self.assertEqual(status["research_plan"]["mode"], "optimize_best")
            self.assertEqual(status["research_plan"]["target_candidate_id"], candidate_id)
            self.assertIn("analyst_positive_sentiment_logit_presentation", status["research_plan"]["keep"])

    def test_control_service_status_includes_candidate_queues(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            store = AlphaStore(base / "alpha.db")
            store.init()
            candidate_id = store.insert_candidate(
                "rank(group_rank(ts_rank(ts_backfill(wait_signal,120),63),industry))",
                {"region": "USA", "universe": "TOP3000", "delay": 0},
                "model:G-1",
            )
            store.record_event(
                candidate_id,
                "generated",
                {
                    "ai_metadata": {
                        "model_profile": "G-1",
                        "profile_guidance": {"long_prompt_context": "x" * 4000},
                    }
                },
            )
            store.update_candidate(
                candidate_id,
                metrics_json=json.dumps({"sharpe": 2.8, "fitness": 1.6, "turnover": 0.2}),
                checks_json=json.dumps(
                    {
                        "LOW_SHARPE": {"status": "PASS", "value": 2.8, "limit": 2.69},
                        "LOW_FITNESS": {"status": "PASS", "value": 1.6, "limit": 1.5},
                        "LOW_TURNOVER": {"status": "PASS", "value": 0.2, "limit": 0.01},
                        "HIGH_TURNOVER": {"status": "PASS", "value": 0.2, "limit": 0.7},
                        "CONCENTRATED_WEIGHT": {"status": "PASS"},
                        "SELF_CORRELATION": {"status": "PENDING"},
                    }
                ),
            )
            store.transition(candidate_id, "check_pending", {"errors": ["SELF_CORRELATION:PENDING"]})
            store.set_run_state(
                "daemon",
                {
                    "status": "stopped",
                    "scope": {"region": "USA", "universe": "TOP3000", "delay": 0, "neutralization": "INDUSTRY"},
                },
            )
            service = ControlService(
                store=store,
                db_path=store.path,
                env_file=base / ".env",
                log_file=base / "alpha.log",
                daemon_stdout_log=base / "daemon.log",
                pid_running=lambda _pid: False,
            )

            status = service.status()

            self.assertIn("candidate_queues", status)
            self.assertEqual(status["candidate_queues"]["watchlist"][0]["id"], candidate_id)
            self.assertNotIn("generated_metadata", status["candidate_queues"]["watchlist"][0])
            self.assertEqual(status["candidate_queues"]["counts"]["watchlist"], 1)

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


if __name__ == "__main__":
    unittest.main()
