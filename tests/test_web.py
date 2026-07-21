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
from alpha.models import SubmitResult
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
        self.assertIn("--throughput-mode", argv)
        self.assertIn("--preset", argv)
        self.assertIn("chn-d0", argv)
        self.assertIn("--batch-size", argv)
        self.assertIn("3", argv)
        self.assertIn("--loop-seconds", argv)
        self.assertIn("20", argv)
        self.assertIn("--no-auto-submit", argv)

    def test_build_daemon_argv_can_enable_auto_submit_explicitly(self):
        argv = build_daemon_argv(
            db_path=Path("alpha.db"),
            env_file=Path(".env"),
            log_file=Path("logs/alpha.log"),
            preset="ind",
            auto_submit=True,
        )

        self.assertIn("--auto-submit", argv)
        self.assertNotIn("--no-auto-submit", argv)

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

    def test_build_daemon_argv_includes_orchestration_mode(self):
        argv = build_daemon_argv(
            db_path=Path("alpha.db"),
            env_file=Path(".env"),
            log_file=Path("logs/alpha.log"),
            preset="ind",
            generator_mode="balanced",
            orchestration_mode="deep",
        )

        self.assertIn("--orchestration-mode", argv)
        self.assertEqual(argv[argv.index("--orchestration-mode") + 1], "deep")

    def test_build_daemon_argv_defaults_to_personal_run_minutes_limit(self):
        argv = build_daemon_argv(
            db_path=Path("alpha.db"),
            env_file=Path(".env"),
            log_file=Path("logs/alpha.log"),
            preset="ind",
            batch_size=8,
            loop_seconds=60,
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
        self.assertIn('class="shell"', HTML)
        self.assertIn('id="sidebar" class="side"', HTML)
        self.assertIn('class="work"', HTML)
        self.assertIn("@media(max-width:1080px)", HTML)
        self.assertIn(".side.open{transform:translateX(0);}", HTML)
        self.assertIn('class="backdrop"', HTML)
        self.assertIn('class="menu-btn"', HTML)
        self.assertIn("function openSidebar()", HTML)
        self.assertIn("function closeSidebar()", HTML)
        self.assertIn("position:sticky;top:0;z-index:30", HTML)
        self.assertIn('id="region"', HTML)
        self.assertIn('id="delay"', HTML)
        self.assertIn('id="universe"', HTML)
        self.assertIn('id="neutralization"', HTML)
        self.assertIn('id="rotate_scopes"', HTML)
        self.assertIn('id="scope_rotation"', HTML)
        self.assertIn('id="batch_size" type="number" value="8" min="1" max="8"', HTML)
        self.assertIn('id="auto_submit" type="checkbox"', HTML)
        self.assertIn("function selectedAutoSubmit()", HTML)
        self.assertIn('id="auto_submit_status"', HTML)
        self.assertIn('id="daemon_submit_mode"', HTML)
        self.assertIn("ready-tag", HTML)
        self.assertIn('name="generator_mode" value="single" checked', HTML)
        self.assertIn('name="generator_mode" value="balanced"', HTML)
        self.assertIn("function selectedGeneratorMode()", HTML)
        self.assertIn('id="generator_mode_status"', HTML)
        self.assertIn('name="orchestration_mode" value="lean" checked', HTML)
        self.assertIn('name="orchestration_mode" value="deep"', HTML)
        self.assertIn("function selectedOrchestrationMode()", HTML)
        self.assertIn("function setRadioValue(name, value)", HTML)
        self.assertIn('id="orchestration_mode_status"', HTML)
        self.assertIn('id="daemon_ai_mode"', HTML)
        self.assertIn('id="run_minutes"', HTML)
        self.assertIn('<option value="120">2 小时</option>', HTML)
        self.assertIn('<option value="240" selected>4 小时</option>', HTML)
        self.assertNotIn('不自动停止', HTML)
        self.assertIn('<option value="custom">自定义</option>', HTML)
        self.assertIn('id="run_minutes_custom" type="number"', HTML)
        self.assertIn("function selectedRunMinutes()", HTML)
        self.assertIn("停止本轮", HTML)
        self.assertNotIn(">暂停</button>", HTML)
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

    def test_control_panel_exposes_responsive_round_top_alphas_panel(self):
        self.assertIn('id="top_alphas_panel"', HTML)
        self.assertIn('本轮 Top 10 · top alphas', HTML)
        self.assertIn('class="top-alpha-table"', HTML)
        self.assertIn('<th>Platform ID</th>', HTML)
        self.assertIn('<th>Score</th>', HTML)
        self.assertIn('<th>Sharpe</th>', HTML)
        self.assertIn('<th>Fitness</th>', HTML)
        self.assertIn('<th>Returns</th>', HTML)
        self.assertIn('<th>Turnover</th>', HTML)
        self.assertIn('<th>Actions</th>', HTML)
        self.assertIn('id="top_alphas"', HTML)
        self.assertIn("detail.id = 'top_alpha_expression'", HTML)
        self.assertIn('id="top_alpha_expression_code"', HTML)
        self.assertIn('function renderTopAlphas(rows)', HTML)
        self.assertIn('function toggleTopAlphaExpression(row, restoreFocus=false)', HTML)
        self.assertIn('function checkTopAlpha(alphaId)', HTML)
        self.assertIn('function submitTopAlpha(alphaId)', HTML)
        self.assertIn("/api/top-alpha/check", HTML)
        self.assertIn("/api/top-alpha/submit", HTML)
        self.assertIn('title="查看官方相关性"', HTML)
        self.assertIn('title="直接提交到 WorldQuant BRAIN"', HTML)
        self.assertIn('window.confirm', HTML)
        self.assertIn("tr.setAttribute('role', 'button')", HTML)
        self.assertIn('本轮暂无可排名的 Alpha 指标', HTML)
        self.assertIn('.top-alpha-table thead{display:none;}', HTML)
        self.assertIn('.top-alpha-table tbody tr{display:grid;', HTML)
        self.assertIn('.top-alpha-table .empty{grid-column:1 / -1;', HTML)
        self.assertIn('.rail .updated{display:none;}', HTML)
        self.assertNotIn('.rail .stat.keep{display:block;}', HTML)
        self.assertIn('renderTopAlphas(data.top_alphas || [])', HTML)

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
            self.assertEqual(state["generator_mode"], "single")
            self.assertEqual(state["auto_submit"], False)
            self.assertEqual(state["throughput_mode"], True)
            self.assertIn("--throughput-mode", calls[0][0])
            self.assertIn("--no-auto-submit", calls[0][0])
            self.assertIn("--generator-mode", calls[0][0])
            self.assertIn("single", calls[0][0])
            self.assertIn("--run-minutes", calls[0][0])
            self.assertIn("120", calls[0][0])
            self.assertIn("ind", calls[0][0])

    def test_control_service_start_accepts_auto_submit_toggle(self):
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

            result = service.start({"preset": "ind", "batch_size": 8, "auto_submit": True})

            state = store.get_run_state("daemon")
            argv = calls[0][0]
            self.assertTrue(result["success"])
            self.assertEqual(state["auto_submit"], True)
            self.assertIn("--auto-submit", argv)
            self.assertNotIn("--no-auto-submit", argv)

    def test_control_service_start_accepts_balanced_generator_mode(self):
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

            result = service.start({"preset": "ind", "batch_size": 8, "generator_mode": "balanced"})

            state = store.get_run_state("daemon")
            argv = calls[0][0]
            self.assertTrue(result["success"])
            self.assertEqual(state["generator_mode"], "balanced")
            self.assertIn("--generator-mode", argv)
            self.assertEqual(argv[argv.index("--generator-mode") + 1], "balanced")

    def test_control_service_start_accepts_deep_orchestration_mode(self):
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

            result = service.start(
                {
                    "preset": "ind",
                    "batch_size": 8,
                    "generator_mode": "balanced",
                    "orchestration_mode": "deep",
                }
            )

            state = store.get_run_state("daemon")
            argv = calls[0][0]
            self.assertTrue(result["success"])
            self.assertEqual(state["generator_mode"], "balanced")
            self.assertEqual(state["orchestration_mode"], "deep")
            self.assertIn("--orchestration-mode", argv)
            self.assertEqual(argv[argv.index("--orchestration-mode") + 1], "deep")

    def test_control_service_start_defaults_to_personal_run_minutes_limit(self):
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

            result = service.start({"preset": "ind", "batch_size": 2, "loop_seconds": 5})

            state = store.get_run_state("daemon")
            self.assertTrue(result["success"])
            self.assertEqual(state["run_minutes"], 240)
            self.assertIn("--run-minutes", calls[0][0])
            self.assertIn("240", calls[0][0])

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

    def test_control_service_stop_preserves_child_stop_reason(self):
        killed = []
        running = {"value": True}

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            store = AlphaStore(base / "alpha.db")
            store.init()
            store.set_run_state("daemon", {"pid": 4321, "preset": "ind", "stop_reason": ""})

            def fake_kill(pid, sig):
                killed.append((pid, sig))
                running["value"] = False
                state = store.get_run_state("daemon")
                state.update({"status": "stopped", "stopped_at": "2026-01-01T00:00:00+00:00", "stop_reason": "interrupted"})
                store.set_run_state("daemon", state)

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
            self.assertEqual(result["reason"], "interrupted")
            self.assertEqual(state["stop_reason"], "interrupted")
            self.assertEqual(state["stopped_at"], "2026-01-01T00:00:00+00:00")

    def test_control_service_stop_sets_reason_when_process_already_exited(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            store = AlphaStore(base / "alpha.db")
            store.init()
            store.set_run_state("daemon", {"pid": 4321, "preset": "ind", "stop_reason": ""})
            service = ControlService(
                store=store,
                db_path=store.path,
                env_file=base / ".env",
                log_file=base / "alpha.log",
                daemon_stdout_log=base / "daemon.log",
                pid_running=lambda _pid: False,
            )

            result = service.stop()

            state = store.get_run_state("daemon")
            self.assertTrue(result["success"])
            self.assertEqual(result["reason"], "exited")
            self.assertEqual(state["status"], "stopped")
            self.assertEqual(state["stop_reason"], "exited")

    def test_control_service_stop_fails_preflight_passed_candidates_from_current_run(self):
        killed = []
        running = {"value": True}

        def fake_kill(pid, sig):
            killed.append((pid, sig))
            running["value"] = False

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            store = AlphaStore(base / "alpha.db")
            store.init()
            store.set_run_state("daemon", {"pid": 4321, "started_at": "2000-01-01T00:00:00+00:00"})
            candidate_id = store.insert_candidate("rank(mdl_signal)", {"region": "USA"}, "model:G-1")
            store.transition(candidate_id, "preflight_passed")
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

            candidate = store.get_candidate(candidate_id)
            events = store.events_for_candidate(candidate_id)
            self.assertTrue(result["success"])
            self.assertEqual(result["interrupted_preflight"], 1)
            self.assertEqual(candidate["status"], "failed")
            self.assertTrue(any("INTERRUPTED_AFTER_PREFLIGHT" in event["metadata_json"] for event in events))

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
            self.assertEqual(status["auto_submit"], False)
            self.assertEqual(status["daemon"]["auto_submit"], False)
            self.assertEqual(status["recent_candidates"][0]["expression"], "rank(close)")
            self.assertEqual(status["recent_candidates"][0]["metrics"]["sharpe"], 2.0)

    def test_control_service_status_infers_generator_mode_from_legacy_argv(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            store = AlphaStore(base / "alpha.db")
            store.init()
            store.set_run_state(
                "daemon",
                {
                    "pid": 4321,
                    "status": "stopped",
                    "argv": ["python", "-m", "alpha.cli", "daemon", "--generator-mode", "balanced"],
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

            self.assertEqual(status["daemon"]["generator_mode"], "balanced")

    def test_control_service_status_fills_legacy_empty_stop_reason_from_health(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            store = AlphaStore(base / "alpha.db")
            store.init()
            store.set_run_state(
                "daemon",
                {
                    "pid": 4321,
                    "status": "stopped",
                    "stop_reason": "",
                    "argv": ["python", "-m", "alpha.cli", "daemon", "--generator-mode", "balanced"],
                },
            )
            store.record_event(None, "daemon_stopped", {"pid": 4321, "reason": "interrupted"})
            service = ControlService(
                store=store,
                db_path=store.path,
                env_file=base / ".env",
                log_file=base / "alpha.log",
                daemon_stdout_log=base / "daemon.log",
                pid_running=lambda _pid: False,
            )

            status = service.status()

            self.assertEqual(status["daemon"]["stop_reason"], "interrupted")
            self.assertEqual(status["health"]["last_block_reason"], "interrupted")

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

            def counted_research_context(state, cycle_plan=None):
                research_calls["count"] += 1
                return original_research_context(state, cycle_plan=cycle_plan)

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

            def counted_build(scope, cycle_plan=None):
                research_builds["count"] += 1
                return original_build(scope, cycle_plan=cycle_plan)

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

    def test_control_service_status_top_alphas_only_uses_current_run_platform_rows_with_complete_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            store = AlphaStore(base / "alpha.db")
            store.init()
            old_id = store.insert_candidate("rank(old_signal)", {"region": "USA"}, "model:old")
            valid_id = store.insert_candidate("rank(valid_signal)", {"region": "USA"}, "model:G-1")
            missing_platform_id = store.insert_candidate("rank(local_only)", {"region": "USA"}, "model:G-2")
            missing_fitness_id = store.insert_candidate("rank(incomplete)", {"region": "USA"}, "model:G-1")
            invalid_metric_id = store.insert_candidate("rank(invalid)", {"region": "USA"}, "model:G-2")
            boolean_metric_id = store.insert_candidate("rank(boolean_metric)", {"region": "USA"}, "model:G-2")
            with store.connect() as conn:
                conn.execute(
                    "UPDATE candidates SET created_at = ?, updated_at = ? WHERE id = ?",
                    ("2026-07-21T19:00:00+00:00", "2026-07-21T19:00:00+00:00", old_id),
                )
            store.update_candidate(
                old_id,
                alpha_id="OLD_PLATFORM_ID",
                metrics_json=json.dumps({"sharpe": 9.0, "fitness": 9.0}),
            )
            store.update_candidate(
                valid_id,
                alpha_id="PLATFORM_A",
                metrics_json=json.dumps({"sharpe": 2.1, "fitness": 1.2, "returns": 0.08, "turnover": 0.24}),
            )
            store.update_candidate(
                missing_platform_id,
                metrics_json=json.dumps({"sharpe": 3.0, "fitness": 2.0}),
            )
            store.update_candidate(
                missing_fitness_id,
                alpha_id="PLATFORM_INCOMPLETE",
                metrics_json=json.dumps({"sharpe": 3.0}),
            )
            store.update_candidate(
                invalid_metric_id,
                alpha_id="PLATFORM_INVALID",
                metrics_json=json.dumps({"sharpe": 3.0, "fitness": "not-a-number"}),
            )
            store.update_candidate(
                boolean_metric_id,
                alpha_id="PLATFORM_BOOLEAN",
                metrics_json=json.dumps({"sharpe": True, "fitness": False}),
            )
            store.set_run_state(
                "daemon",
                {"status": "stopped", "started_at": "2026-07-21T20:00:00+00:00"},
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

            self.assertEqual([row["alpha_id"] for row in status["top_alphas"]], ["PLATFORM_A"])
            self.assertEqual(status["top_alphas"][0]["expression"], "rank(valid_signal)")
            self.assertNotIn("id", status["top_alphas"][0])

    def test_control_service_status_top_alphas_uses_quality_score_and_newer_tie_break(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            store = AlphaStore(base / "alpha.db")
            store.init()
            fixtures = [
                ("PLATFORM_A", 2.0, 1.0),
                ("PLATFORM_B", 1.9, 2.0),
                ("PLATFORM_C", 2.0, 1.0),
            ]
            for index, (alpha_id, sharpe, fitness) in enumerate(fixtures):
                candidate_id = store.insert_candidate(
                    f"rank(signal_{index})",
                    {"region": "USA", "universe": "TOP3000", "delay": 0},
                    "model:G-1",
                )
                store.update_candidate(
                    candidate_id,
                    alpha_id=alpha_id,
                    metrics_json=json.dumps(
                        {"sharpe": sharpe, "fitness": fitness, "returns": 0.07, "turnover": 0.2}
                    ),
                )
                store.transition(candidate_id, "failed" if alpha_id == "PLATFORM_B" else "approved")
            store.set_run_state("daemon", {"status": "stopped", "started_at": "2000-01-01T00:00:00+00:00"})
            service = ControlService(
                store=store,
                db_path=store.path,
                env_file=base / ".env",
                log_file=base / "alpha.log",
                daemon_stdout_log=base / "daemon.log",
                pid_running=lambda _pid: False,
            )

            rows = service.status()["top_alphas"]

            self.assertEqual([row["alpha_id"] for row in rows], ["PLATFORM_B", "PLATFORM_C", "PLATFORM_A"])
            self.assertEqual([row["rank"] for row in rows], [1, 2, 3])
            self.assertEqual([row["quality_score"] for row in rows], [2.6, 2.35, 2.35])
            self.assertEqual(rows[0]["status"], "failed")
            self.assertEqual(rows[0]["settings"]["universe"], "TOP3000")

    def test_control_service_status_top_alphas_sorts_by_unrounded_quality_score(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            store = AlphaStore(base / "alpha.db")
            store.init()
            for alpha_id, sharpe in (("PLATFORM_HIGHER", 2.0000004), ("PLATFORM_LOWER", 2.0000003)):
                candidate_id = store.insert_candidate("rank(signal)", {"region": "USA"}, "model:G-1")
                store.update_candidate(
                    candidate_id,
                    alpha_id=alpha_id,
                    metrics_json=json.dumps({"sharpe": sharpe, "fitness": 0.0}),
                )
            store.set_run_state("daemon", {"status": "stopped", "started_at": "2000-01-01T00:00:00+00:00"})
            service = ControlService(
                store=store,
                db_path=store.path,
                env_file=base / ".env",
                log_file=base / "alpha.log",
                daemon_stdout_log=base / "daemon.log",
                pid_running=lambda _pid: False,
            )

            rows = service.status()["top_alphas"]

            self.assertEqual([row["alpha_id"] for row in rows], ["PLATFORM_HIGHER", "PLATFORM_LOWER"])

    def test_control_service_status_top_alphas_is_limited_to_ten(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            store = AlphaStore(base / "alpha.db")
            store.init()
            for index in range(12):
                candidate_id = store.insert_candidate(f"rank(signal_{index})", {"region": "USA"}, "model:G-1")
                store.update_candidate(
                    candidate_id,
                    alpha_id=f"PLATFORM_{index}",
                    metrics_json=json.dumps(
                        {"sharpe": float(index), "fitness": 1.0, "returns": index / 100, "turnover": 0.2}
                    ),
                )
            store.set_run_state("daemon", {"status": "stopped", "started_at": "2000-01-01T00:00:00+00:00"})
            service = ControlService(
                store=store,
                db_path=store.path,
                env_file=base / ".env",
                log_file=base / "alpha.log",
                daemon_stdout_log=base / "daemon.log",
                pid_running=lambda _pid: False,
            )

            rows = service.status()["top_alphas"]

            self.assertEqual(len(rows), 10)
            self.assertEqual([row["alpha_id"] for row in rows], [f"PLATFORM_{index}" for index in range(11, 1, -1)])
            self.assertEqual(rows[0]["returns"], 0.11)
            self.assertEqual(rows[0]["turnover"], 0.2)

    def test_control_service_status_top_alphas_is_empty_without_a_run_start(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            store = AlphaStore(base / "alpha.db")
            store.init()
            candidate_id = store.insert_candidate("rank(historical_signal)", {"region": "USA"}, "model:G-1")
            store.update_candidate(
                candidate_id,
                alpha_id="HISTORICAL_PLATFORM_ID",
                metrics_json=json.dumps({"sharpe": 4.0, "fitness": 2.0}),
            )
            service = ControlService(
                store=store,
                db_path=store.path,
                env_file=base / ".env",
                log_file=base / "alpha.log",
                daemon_stdout_log=base / "daemon.log",
                pid_running=lambda _pid: False,
            )

            self.assertEqual(service.status()["top_alphas"], [])

    def test_control_service_checks_current_top_alpha_with_official_platform_data(self):
        class FakeBrain:
            def get_alpha_correlations(self, alpha_id):
                self.alpha_id = alpha_id
                return {
                    "self": {"name": "SELF_CORRELATION", "status": "PASS", "value": 0.42, "limit": 0.7},
                    "production": {
                        "name": "PROD_CORRELATION",
                        "status": "PASS",
                        "value": 0.51,
                        "limit": 0.7,
                    },
                }

            def get_submission_check(self, _alpha_id):
                raise AssertionError("direct correlation check must not POST /check")

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            store = AlphaStore(base / "alpha.db")
            store.init()
            candidate_id = store.insert_candidate("rank(signal)", {"region": "USA"}, "model:G-1")
            store.update_candidate(
                candidate_id,
                alpha_id="PLATFORM_A",
                metrics_json=json.dumps({"sharpe": 2.1, "fitness": 1.2}),
            )
            store.set_run_state("daemon", {"status": "stopped", "started_at": "2000-01-01T00:00:00+00:00"})
            service = ControlService(
                store=store,
                db_path=store.path,
                env_file=base / ".env",
                log_file=base / "alpha.log",
                daemon_stdout_log=base / "daemon.log",
            )
            brain = FakeBrain()
            service._brain_client = lambda: brain

            result = service.check_top_alpha({"alpha_id": "PLATFORM_A"})

            self.assertTrue(result["success"])
            self.assertTrue(result["correlation_confirmed"])
            self.assertEqual(result["correlations"]["self"]["value"], 0.42)
            self.assertEqual(result["correlations"]["production"]["value"], 0.51)
            self.assertEqual(brain.alpha_id, "PLATFORM_A")
            saved_checks = json.loads(store.get_candidate(candidate_id)["checks_json"])
            self.assertEqual(saved_checks["PROD_CORRELATION"]["status"], "PASS")

    def test_control_service_keeps_empty_platform_correlation_unconfirmed(self):
        class FakeBrain:
            def get_alpha_correlations(self, _alpha_id):
                return {
                    "self": {"name": "SELF_CORRELATION", "status": "UNCONFIRMED", "value": None},
                    "production": {"name": "PROD_CORRELATION", "status": "UNCONFIRMED", "value": None},
                }

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            store = AlphaStore(base / "alpha.db")
            store.init()
            candidate_id = store.insert_candidate("rank(signal)", {"region": "USA"}, "model:G-1")
            store.update_candidate(
                candidate_id,
                alpha_id="PLATFORM_A",
                metrics_json=json.dumps({"sharpe": 2.1, "fitness": 1.2}),
            )
            store.set_run_state("daemon", {"status": "stopped", "started_at": "2000-01-01T00:00:00+00:00"})
            service = ControlService(
                store=store,
                db_path=store.path,
                env_file=base / ".env",
                log_file=base / "alpha.log",
                daemon_stdout_log=base / "daemon.log",
            )
            service._brain_client = lambda: FakeBrain()

            result = service.check_top_alpha({"alpha_id": "PLATFORM_A"})

            self.assertTrue(result["success"])
            self.assertFalse(result["correlation_confirmed"])
            self.assertIsNone(result["correlations"]["self"]["value"])
            self.assertIn("no numeric correlation", result["message"])

    def test_control_service_manual_submit_requires_current_top_ten_and_syncs_verified_os(self):
        class FakeBrain:
            def submit_alpha(self, alpha_id, dry_run=True):
                self.call = (alpha_id, dry_run)
                return SubmitResult(alpha_id=alpha_id, submitted=True, stage="OS", message="verified OS")

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            store = AlphaStore(base / "alpha.db")
            store.init()
            candidate_id = store.insert_candidate("rank(signal)", {"region": "USA"}, "model:G-1")
            store.update_candidate(
                candidate_id,
                alpha_id="PLATFORM_A",
                metrics_json=json.dumps({"sharpe": 2.1, "fitness": 1.2}),
            )
            store.set_run_state("daemon", {"status": "stopped", "started_at": "2000-01-01T00:00:00+00:00"})
            service = ControlService(
                store=store,
                db_path=store.path,
                env_file=base / ".env",
                log_file=base / "alpha.log",
                daemon_stdout_log=base / "daemon.log",
            )
            brain = FakeBrain()
            service._brain_client = lambda: brain

            with self.assertRaisesRegex(ValueError, "Top 10"):
                service.submit_top_alpha({"alpha_id": "NOT_CURRENT"})
            result = service.submit_top_alpha({"alpha_id": "PLATFORM_A"})

            self.assertTrue(result["submitted"])
            self.assertEqual(result["stage"], "OS")
            self.assertEqual(brain.call, ("PLATFORM_A", False))
            self.assertEqual(store.get_candidate(candidate_id)["status"], "submitted")

    def test_control_service_status_while_running_uses_latest_experiment_plan_without_rebuilding_context(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            store = AlphaStore(base / "alpha.db")
            store.init()
            scope = {"region": "USA", "universe": "TOP500", "delay": 0, "neutralization": "INDUSTRY"}
            store.set_run_state(
                "daemon",
                {
                    "pid": 4321,
                    "status": "running",
                    "started_at": "2026-06-01T00:00:00+00:00",
                    "scope": scope,
                    "batch_size": 8,
                },
            )
            store.record_event(
                None,
                "experiment_plan",
                {
                    "mode": "explore_new_family",
                    "quality_budget": {"priority": "production_first"},
                    "candidate_queues": {"counts": {"watchlist": 0}},
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

            def fail_build(*_args, **_kwargs):
                raise AssertionError("running status should not rebuild research context")

            service._build_research_context = fail_build

            status = service.status()

            self.assertTrue(status["running"])
            self.assertEqual(status["research_plan"]["mode"], "explore_new_family")
            self.assertEqual(status["research_plan"]["quality_budget"]["priority"], "production_first")
            self.assertEqual(status["research_analysis"], {})

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
            store.update_candidate(
                candidate_id,
                metrics_json=json.dumps({"sharpe": 2.55, "fitness": 1.32, "turnover": 0.22}),
                checks_json=json.dumps(
                    {
                        "LOW_SHARPE": {"status": "FAIL", "value": 2.55, "limit": 2.69},
                        "LOW_FITNESS": {"status": "FAIL", "value": 1.32, "limit": 1.5},
                        "LOW_TURNOVER": {"status": "PASS", "value": 0.22, "limit": 0.01},
                        "HIGH_TURNOVER": {"status": "PASS", "value": 0.22, "limit": 0.7},
                        "CONCENTRATED_WEIGHT": {"status": "PASS"},
                        "LOW_2Y_SHARPE": {"status": "PASS", "value": 2.8, "limit": 2.69},
                        "LOW_SUB_UNIVERSE_SHARPE": {"status": "PASS", "value": 1.1, "limit": 0.49},
                        "IS_LADDER_SHARPE": {"status": "PASS", "value": 2.8, "limit": 2.69},
                    }
                ),
            )
            store.transition(
                candidate_id,
                "check_pending",
                {"errors": ["SHARPE_BELOW_MIN:2.550<2.69", "FITNESS_BELOW_MIN:1.320<1.5"]},
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

            self.assertEqual(status["research_plan"]["mode"], "optimize_best")
            self.assertEqual(status["research_plan"]["target_candidate_id"], candidate_id)
            self.assertIn("analyst_positive_sentiment_logit_presentation", status["research_plan"]["keep"])

    def test_control_service_status_avoids_platform_submitted_scheduler_target(self):
        class BrainWithSubmissions:
            def recent_submitted_alphas(self, settings=None, limit=50):
                return [
                    {
                        "id": "OS1",
                        "stage": "OS",
                        "regular": "rank(group_rank(ts_rank(vec_avg(ern7_dsu_spe), 63), sector))",
                        "settings": {"region": "USA", "universe": "TOP3000", "delay": 0, "neutralization": "STATISTICAL"},
                    }
                ]

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            store = AlphaStore(base / "alpha.db")
            store.init()
            settings = {"region": "USA", "universe": "TOP3000", "delay": 0, "neutralization": "STATISTICAL"}
            candidate_id = store.insert_candidate(
                "rank(multiply(group_rank(ts_rank(divide(winsorize(ts_backfill(vec_avg(ern7_dsu_spe),105),std=4),cap),42),country),group_rank(ts_rank(divide(winsorize(ts_backfill(vec_avg(ern7_dsu_spe),180),std=3),cap),126),country)))",
                settings,
                "model:G-1",
            )
            store.update_candidate(
                candidate_id,
                metrics_json=json.dumps({"sharpe": 3.09, "fitness": 1.49, "turnover": 0.3059}),
                checks_json=json.dumps(
                    {
                        "LOW_SHARPE": {"status": "PASS", "value": 3.09, "limit": 2.69},
                        "LOW_FITNESS": {"status": "FAIL", "value": 1.49, "limit": 1.5},
                        "DATA_DIVERSITY": {"status": "WARNING"},
                    }
                ),
            )
            store.transition(candidate_id, "failed", {"errors": ["FITNESS_BELOW_MIN:1.490<1.5"]})
            store.set_run_state("daemon", {"status": "stopped", "scope": settings, "batch_size": 8})
            service = ControlService(
                store=store,
                db_path=store.path,
                env_file=base / ".env",
                log_file=base / "alpha.log",
                daemon_stdout_log=base / "daemon.log",
                pid_running=lambda _pid: False,
            )
            service._brain_client = lambda: BrainWithSubmissions()

            status = service.status()

        plan = status["research_plan"]
        self.assertEqual(status["scheduler_plan"]["mode"], "optimize")
        self.assertEqual(status["scheduler_plan"]["target_candidate_id"], candidate_id)
        self.assertEqual(plan["mode"], "explore_new_family")
        self.assertEqual(plan["abandoned_target_id"], candidate_id)
        self.assertEqual(plan["abandon_reason"], "SUBMITTED_FIELD_AVOIDANCE")
        self.assertNotIn("ern7_dsu_spe", plan["keep"])
        self.assertIn("ern7_dsu_spe", plan["avoid"])
        self.assertIn("ern7_dsu_spe", plan["submitted_field_avoidance"]["fields"])

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

    def test_control_service_status_passes_scheduler_plan_to_research_plan(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            store = AlphaStore(base / "alpha.db")
            store.init()
            scope = {"region": "USA", "universe": "TOP500", "delay": 0, "neutralization": "INDUSTRY"}
            for idx in range(80):
                candidate_id = store.insert_candidate(f"rank(weak_signal_{idx})", scope, "planner_unverified_probe")
                store.update_candidate(candidate_id, metrics_json=json.dumps({"sharpe": 0.0, "fitness": 0.0}))
                store.transition(candidate_id, "failed", {"reason": "bad_full_batch"})
            store.record_event(None, "quality_stop_loss", {"scope": scope, "quality_stop_reason": "bad_full_batch"})
            store.record_event(None, "quality_stop_loss", {"scope": scope, "quality_stop_reason": "bad_full_batch"})
            store.set_run_state(
                "daemon",
                {"status": "stopped", "scope": scope, "stop_reason": "production_rescue_duplicate_only"},
            )
            service = ControlService(
                store=store,
                db_path=store.path,
                env_file=base / ".env",
                log_file=base / "alpha.log",
                daemon_stdout_log=base / "daemon.log",
                web_log=base / "web.log",
                pid_running=lambda _pid: False,
            )

            status = service.status()

        self.assertEqual(status["scheduler_plan"]["mode"], "explore")
        self.assertEqual(status["scheduler_plan"]["reason"], "production_rescue_duplicate_only_recent")
        self.assertEqual(status["research_plan"]["scheduler_plan"]["reason"], status["scheduler_plan"]["reason"])
        self.assertFalse(status["research_plan"]["production_rescue"]["active"])
        self.assertEqual(status["research_plan"]["quality_budget"]["slots"], {"broad_explore": 8})

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


class WebSecurityTests(unittest.TestCase):
    def _make_service(self, tmp: Path) -> ControlService:
        store = AlphaStore(tmp / "alpha.db")
        store.init()
        return ControlService(
            store=store,
            db_path=tmp / "alpha.db",
            env_file=tmp / ".env",
            log_file=tmp / "alpha.log",
        )

    def _serve(self, service, token: str):
        import http.server
        import threading
        from alpha.web import make_handler

        handler = make_handler(service, token=token)
        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        return server, thread

    def _request(self, port: int, method: str, path: str, headers=None, body: bytes | None = None):
        import http.client

        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        conn.request(method, path, body=body, headers=headers or {})
        resp = conn.getresponse()
        resp.read()
        conn.close()
        return resp.status

    def test_token_required_when_configured(self):
        with tempfile.TemporaryDirectory() as tmp:
            service = self._make_service(Path(tmp))
            server, thread = self._serve(service, token="secret-token")
            try:
                port = server.server_address[1]
                self.assertEqual(self._request(port, "GET", "/api/status"), 401)
                self.assertEqual(
                    self._request(port, "GET", "/api/status", headers={"X-Alpha-Token": "wrong"}),
                    401,
                )
                self.assertEqual(
                    self._request(port, "GET", "/api/status", headers={"X-Alpha-Token": "secret-token"}),
                    200,
                )
                self.assertEqual(self._request(port, "GET", "/api/status?token=secret-token"), 200)
            finally:
                server.shutdown()
                server.server_close()

    def test_no_token_allows_access(self):
        with tempfile.TemporaryDirectory() as tmp:
            service = self._make_service(Path(tmp))
            server, thread = self._serve(service, token="")
            try:
                port = server.server_address[1]
                self.assertEqual(self._request(port, "GET", "/api/status"), 200)
            finally:
                server.shutdown()
                server.server_close()

    def test_cross_origin_post_is_blocked(self):
        with tempfile.TemporaryDirectory() as tmp:
            service = self._make_service(Path(tmp))
            server, thread = self._serve(service, token="")
            try:
                port = server.server_address[1]
                host = f"127.0.0.1:{port}"
                # Cross-site Origin must be rejected on state-changing POST.
                self.assertEqual(
                    self._request(
                        port, "POST", "/api/stop",
                        headers={"Content-Type": "application/json", "Origin": "http://evil.example"},
                        body=b"{}",
                    ),
                    403,
                )
                # Same-origin POST passes the CSRF check (status 200 from stop()).
                self.assertEqual(
                    self._request(
                        port, "POST", "/api/stop",
                        headers={"Content-Type": "application/json", "Origin": f"http://{host}"},
                        body=b"{}",
                    ),
                    200,
                )
            finally:
                server.shutdown()
                server.server_close()

    def test_run_web_app_allows_non_loopback_without_token_with_warning(self):
        import os as _os
        from unittest.mock import patch
        from alpha.web import run_web_app

        with tempfile.TemporaryDirectory() as tmp:
            env_path = Path(tmp) / ".env"
            env_path.write_text("AI_CLIENT=local\n", encoding="utf-8")
            with patch.dict(_os.environ, {}, clear=False):
                _os.environ.pop("ALPHA_WEB_TOKEN", None)
                with patch("alpha.web.ThreadingHTTPServer") as server_cls, patch("builtins.print") as print_mock:
                    server = server_cls.return_value
                    server.serve_forever.side_effect = KeyboardInterrupt()
                    run_web_app(
                        db_path=Path(tmp) / "alpha.db",
                        env_file=env_path,
                        log_file=Path(tmp) / "alpha.log",
                        host="0.0.0.0",
                        port=0,
                    )
            printed = "\n".join(str(call.args[0]) for call in print_mock.call_args_list if call.args)
            self.assertIn("without ALPHA_WEB_TOKEN", printed)

    def test_oversized_body_is_rejected(self):
        import http.client
        with tempfile.TemporaryDirectory() as tmp:
            service = self._make_service(Path(tmp))
            server, thread = self._serve(service, token="")
            try:
                port = server.server_address[1]
                big = b'{"x":"' + b"a" * (2 << 20) + b'"}'
                conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
                # Refusing to read a huge body may break the pipe mid-send; either a
                # clean 400 or a connection error counts as "rejected, memory bounded".
                rejected = False
                try:
                    conn.request("POST", "/api/stop", body=big, headers={"Content-Type": "application/json"})
                    resp = conn.getresponse()
                    resp.read()
                    rejected = resp.status == 400
                except (BrokenPipeError, ConnectionError):
                    rejected = True
                finally:
                    conn.close()
                self.assertTrue(rejected)
            finally:
                server.shutdown()
                server.server_close()


if __name__ == "__main__":
    unittest.main()
