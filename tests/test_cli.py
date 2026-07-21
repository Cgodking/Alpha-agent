from __future__ import annotations

import io
import json
import os
import sqlite3
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from alpha.cli import main
from alpha.db import AlphaStore


class CliTests(unittest.TestCase):
    def _local_env(self, tmp: str) -> Path:
        env_path = Path(tmp) / ".env"
        env_path.write_text("AI_CLIENT=local\nBRAIN_CLIENT=local\nAUTO_SUBMIT=false\n", encoding="utf-8")
        return env_path

    def test_cli_init_db_creates_database(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "alpha.db"
            env_path = self._local_env(tmp)

            with patch.dict(os.environ, {}, clear=True):
                exit_code = main(["--env-file", str(env_path), "--db", str(db_path), "init-db"])

            self.assertEqual(exit_code, 0)
            self.assertTrue(db_path.exists())

    def test_cli_run_once_creates_candidates(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "alpha.db"
            env_path = self._local_env(tmp)

            with patch.dict(os.environ, {}, clear=True):
                self.assertEqual(main(["--env-file", str(env_path), "--db", str(db_path), "init-db"]), 0)
                self.assertEqual(main(["--env-file", str(env_path), "--db", str(db_path), "run-once", "--batch-size", "1"]), 0)

            self.assertTrue(db_path.exists())

    def test_cli_prune_history_dry_run_and_execute(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "alpha.db"
            env_path = self._local_env(tmp)
            store = AlphaStore(db_path)
            store.init()
            candidate_id = store.insert_candidate(
                "rank(dead_signal)",
                {"region": "USA", "universe": "TOP3000", "delay": 1, "neutralization": "INDUSTRY"},
                "model:G-1",
            )
            store.update_candidate(
                candidate_id,
                metrics_json=json.dumps({"sharpe": 0.0, "fitness": 0.0}),
                checks_json=json.dumps({"LOW_SHARPE": {"status": "FAIL", "value": 0.0, "limit": 1.58}}),
            )
            store.transition(candidate_id, "failed", {"errors": ["LOW_SHARPE:FAIL"]})

            with patch.dict(os.environ, {}, clear=True):
                dry_stdout = io.StringIO()
                with redirect_stdout(dry_stdout):
                    dry_code = main(["--env-file", str(env_path), "--db", str(db_path), "prune-history", "--all-scopes"])
                execute_stdout = io.StringIO()
                with redirect_stdout(execute_stdout):
                    execute_code = main(
                        [
                            "--env-file",
                            str(env_path),
                            "--db",
                            str(db_path),
                            "prune-history",
                            "--all-scopes",
                            "--execute",
                        ]
                    )

            self.assertEqual(dry_code, 0)
            self.assertEqual(execute_code, 0)
            self.assertIn("'selected': 1", dry_stdout.getvalue())
            self.assertIn("'archived': 1", execute_stdout.getvalue())
            self.assertIsNotNone(store.find_duplicate_candidate("rank(dead_signal)", {"region": "USA", "delay": 1}))
            with self.assertRaises(KeyError):
                store.get_candidate(candidate_id)

    def test_cli_run_once_accepts_us_d0_preset(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "alpha.db"
            env_path = self._local_env(tmp)

            with patch.dict(os.environ, {}, clear=True):
                main(["--env-file", str(env_path), "--db", str(db_path), "init-db"])
                exit_code = main(
                    [
                        "--env-file",
                        str(env_path),
                        "--db",
                        str(db_path),
                        "run-once",
                        "--preset",
                        "us-d0",
                        "--batch-size",
                        "1",
                    ]
                )

            settings = self._latest_settings(db_path)
            self.assertEqual(exit_code, 0)
            self.assertEqual(settings["region"], "USA")
            self.assertEqual(settings["universe"], "TOP3000")
            self.assertEqual(settings["delay"], 0)

    def test_cli_run_once_accepts_global_region_presets(self):
        cases = {
            "chn-d0": {"region": "CHN", "universe": "TOP2000U", "delay": 0},
            "chn": {"region": "CHN", "universe": "TOP2000U", "delay": 1},
            "eur-d0": {"region": "EUR", "universe": "TOP2500", "delay": 0},
            "eur": {"region": "EUR", "universe": "TOP2500", "delay": 1},
            "glb": {"region": "GLB", "universe": "TOP3000", "delay": 1},
            "ind": {"region": "IND", "universe": "TOP500", "delay": 1},
        }
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "alpha.db"
            env_path = self._local_env(tmp)

            with patch.dict(os.environ, {}, clear=True):
                main(["--env-file", str(env_path), "--db", str(db_path), "init-db"])
                for preset, expected in cases.items():
                    with self.subTest(preset=preset):
                        exit_code = main(
                            [
                                "--env-file",
                                str(env_path),
                                "--db",
                                str(db_path),
                                "run-once",
                                "--preset",
                                preset,
                                "--batch-size",
                                "1",
                            ]
                        )
                        settings = self._latest_settings(db_path)
                        self.assertEqual(exit_code, 0)
                        self.assertEqual(settings["region"], expected["region"])
                        self.assertEqual(settings["universe"], expected["universe"])
                        self.assertEqual(settings["delay"], expected["delay"])

    def test_cli_run_once_scope_args_override_env_defaults(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "alpha.db"
            env_path = self._local_env(tmp)

            with patch.dict(os.environ, {}, clear=True):
                main(["--env-file", str(env_path), "--db", str(db_path), "init-db"])
                exit_code = main(
                    [
                        "--env-file",
                        str(env_path),
                        "--db",
                        str(db_path),
                        "run-once",
                        "--region",
                        "USA",
                        "--universe",
                        "TOP3000",
                        "--delay",
                        "0",
                        "--neutralization",
                        "SUBINDUSTRY",
                        "--decay",
                        "6",
                        "--truncation",
                        "0.03",
                        "--batch-size",
                        "1",
                    ]
                )

            settings = self._latest_settings(db_path)
            self.assertEqual(exit_code, 0)
            self.assertEqual(settings["delay"], 0)
            self.assertEqual(settings["neutralization"], "SUBINDUSTRY")
            self.assertEqual(settings["decay"], 6)
            self.assertEqual(settings["truncation"], 0.03)

    def test_cli_daemon_stops_when_run_minutes_limit_is_reached(self):
        class FakeWorker:
            def run_once(self):
                calls.append("run_once")
                return {"generated": 0}

        calls = []
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "alpha.db"
            env_path = self._local_env(tmp)
            stdout = io.StringIO()

            with (
                patch.dict(os.environ, {}, clear=True),
                patch("alpha.cli._worker", return_value=FakeWorker()),
                patch("alpha.cli.time.monotonic", side_effect=[0.0, 0.0, 0.1]),
                patch("alpha.cli.time.sleep") as sleep,
                redirect_stdout(stdout),
            ):
                exit_code = main(
                    [
                        "--env-file",
                        str(env_path),
                        "--db",
                        str(db_path),
                        "daemon",
                        "--batch-size",
                        "1",
                        "--loop-seconds",
                        "60",
                        "--run-minutes",
                        "0.001",
                    ]
                )

        self.assertEqual(exit_code, 0)
        self.assertEqual(calls, ["run_once"])
        sleep.assert_not_called()
        self.assertIn("time_limit_reached", stdout.getvalue())

    def test_cli_daemon_generator_mode_overrides_env_file(self):
        class FakeWorker:
            def run_once(self):
                seen["mode"] = os.environ.get("AI_ORCHESTRATION_MODE")
                seen["max_active"] = os.environ.get("AI_MAX_ACTIVE_GENERATORS")
                seen["timeout"] = os.environ.get("AI_GENERATION_STAGE_TIMEOUT_SECONDS")
                return {"generated": 0}

        seen = {}
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "alpha.db"
            env_path = Path(tmp) / ".env"
            env_path.write_text(
                "AI_CLIENT=local\nBRAIN_CLIENT=local\nAUTO_SUBMIT=false\nAI_MAX_ACTIVE_GENERATORS=1\n",
                encoding="utf-8",
            )
            stdout = io.StringIO()

            with (
                patch.dict(os.environ, {}, clear=True),
                patch("alpha.cli._worker", return_value=FakeWorker()),
                patch("alpha.cli.time.monotonic", side_effect=[0.0, 0.0, 0.1]),
                patch("alpha.cli.time.sleep"),
                redirect_stdout(stdout),
            ):
                exit_code = main(
                    [
                        "--env-file",
                        str(env_path),
                        "--db",
                        str(db_path),
                        "daemon",
                        "--batch-size",
                        "1",
                        "--run-minutes",
                        "0.001",
                        "--generator-mode",
                        "balanced",
                    ]
                )

        self.assertEqual(exit_code, 0)
        self.assertEqual(seen["mode"], "lean")
        self.assertEqual(seen["max_active"], "0")
        self.assertGreaterEqual(float(seen["timeout"]), 180.0)

    def test_cli_daemon_deep_orchestration_mode_keeps_decision_chain_enabled(self):
        class FakeWorker:
            def run_once(self):
                seen["mode"] = os.environ.get("AI_ORCHESTRATION_MODE")
                seen["max_active"] = os.environ.get("AI_MAX_ACTIVE_GENERATORS")
                return {"generated": 0}

        seen = {}
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "alpha.db"
            env_path = Path(tmp) / ".env"
            env_path.write_text(
                "AI_CLIENT=local\nBRAIN_CLIENT=local\nAUTO_SUBMIT=false\nAI_ORCHESTRATION_MODE=lean\n",
                encoding="utf-8",
            )
            stdout = io.StringIO()

            with (
                patch.dict(os.environ, {}, clear=True),
                patch("alpha.cli._worker", return_value=FakeWorker()),
                patch("alpha.cli.time.monotonic", side_effect=[0.0, 0.0, 0.1]),
                patch("alpha.cli.time.sleep"),
                redirect_stdout(stdout),
            ):
                exit_code = main(
                    [
                        "--env-file",
                        str(env_path),
                        "--db",
                        str(db_path),
                        "daemon",
                        "--batch-size",
                        "1",
                        "--run-minutes",
                        "0.001",
                        "--generator-mode",
                        "balanced",
                        "--orchestration-mode",
                        "deep",
                    ]
                )

        self.assertEqual(exit_code, 0)
        self.assertEqual(seen["mode"], "deep")
        self.assertEqual(seen["max_active"], "0")

    def test_cli_daemon_auto_submit_flag_overrides_env_file(self):
        class FakeWorker:
            def run_once(self):
                return {"generated": 0}

        seen = {}

        def fake_worker(_store, _batch_size, policy, *_args):
            seen["auto_submit"] = policy.auto_submit
            return FakeWorker()

        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "alpha.db"
            env_path = Path(tmp) / ".env"
            env_path.write_text("AI_CLIENT=local\nBRAIN_CLIENT=local\nAUTO_SUBMIT=false\n", encoding="utf-8")

            with (
                patch.dict(os.environ, {}, clear=True),
                patch("alpha.cli._worker", side_effect=fake_worker),
                patch("alpha.cli.time.monotonic", side_effect=[0.0, 0.0, 0.1]),
                patch("alpha.cli.time.sleep"),
            ):
                exit_code = main(
                    [
                        "--env-file",
                        str(env_path),
                        "--db",
                        str(db_path),
                        "daemon",
                        "--batch-size",
                        "1",
                        "--run-minutes",
                        "0.001",
                        "--auto-submit",
                    ]
                )

            state = AlphaStore(db_path).get_run_state("daemon")

        self.assertEqual(exit_code, 0)
        self.assertEqual(seen["auto_submit"], True)
        self.assertEqual(state["auto_submit"], True)

    def test_cli_daemon_no_auto_submit_flag_overrides_env_file(self):
        class FakeWorker:
            def run_once(self):
                return {"generated": 0}

        seen = {}

        def fake_worker(_store, _batch_size, policy, *_args):
            seen["auto_submit"] = policy.auto_submit
            return FakeWorker()

        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "alpha.db"
            env_path = Path(tmp) / ".env"
            env_path.write_text("AI_CLIENT=local\nBRAIN_CLIENT=local\nAUTO_SUBMIT=true\n", encoding="utf-8")

            with (
                patch.dict(os.environ, {}, clear=True),
                patch("alpha.cli._worker", side_effect=fake_worker),
                patch("alpha.cli.time.monotonic", side_effect=[0.0, 0.0, 0.1]),
                patch("alpha.cli.time.sleep"),
            ):
                exit_code = main(
                    [
                        "--env-file",
                        str(env_path),
                        "--db",
                        str(db_path),
                        "daemon",
                        "--batch-size",
                        "1",
                        "--run-minutes",
                        "0.001",
                        "--no-auto-submit",
                    ]
                )

            state = AlphaStore(db_path).get_run_state("daemon")

        self.assertEqual(exit_code, 0)
        self.assertEqual(seen["auto_submit"], False)
        self.assertEqual(state["auto_submit"], False)

    def test_cli_daemon_owns_run_state_when_started_directly(self):
        class FakeWorker:
            def run_once(self):
                return {"generated": 0}

        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "alpha.db"
            env_path = self._local_env(tmp)
            store = AlphaStore(db_path)
            store.init()
            store.set_run_state(
                "daemon",
                {
                    "status": "stopped",
                    "pid": 999999,
                    "stop_reason": "production_rescue_duplicate_only",
                    "scope": {"region": "USA", "universe": "TOP500", "delay": 0, "neutralization": "INDUSTRY"},
                },
            )

            with (
                patch.dict(os.environ, {}, clear=True),
                patch("alpha.cli._worker", return_value=FakeWorker()),
                patch("alpha.cli.time.monotonic", side_effect=[0.0, 0.0, 0.1]),
                patch("alpha.cli.time.sleep"),
            ):
                exit_code = main(
                    [
                        "--env-file",
                        str(env_path),
                        "--db",
                        str(db_path),
                        "daemon",
                        "--region",
                        "USA",
                        "--universe",
                        "TOP500",
                        "--delay",
                        "0",
                        "--neutralization",
                        "INDUSTRY",
                        "--batch-size",
                        "1",
                        "--loop-seconds",
                        "60",
                        "--run-minutes",
                        "0.001",
                        "--generator-mode",
                        "balanced",
                    ]
                )

            state = store.get_run_state("daemon")

        self.assertEqual(exit_code, 0)
        self.assertEqual(state["status"], "stopped")
        self.assertEqual(state["stop_reason"], "time_limit")
        self.assertEqual(state["scope"]["universe"], "TOP500")
        self.assertEqual(state["generator_mode"], "balanced")
        self.assertEqual(state["batch_size"], 1)

    def test_cli_daemon_time_limit_fails_preflight_passed_candidates_from_current_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "alpha.db"
            env_path = self._local_env(tmp)
            stdout = io.StringIO()

            class FakeWorker:
                def run_once(self):
                    store = AlphaStore(db_path)
                    candidate_id = store.insert_candidate("rank(mdl_signal)", {"region": "USA"}, "model:G-1")
                    store.transition(candidate_id, "preflight_passed")
                    return {"generated": 1}

            with (
                patch.dict(os.environ, {}, clear=True),
                patch("alpha.cli._worker", return_value=FakeWorker()),
                patch("alpha.cli.time.monotonic", side_effect=[0.0, 0.0, 0.1]),
                patch("alpha.cli.time.sleep"),
                redirect_stdout(stdout),
            ):
                exit_code = main(
                    [
                        "--env-file",
                        str(env_path),
                        "--db",
                        str(db_path),
                        "daemon",
                        "--batch-size",
                        "1",
                        "--loop-seconds",
                        "60",
                        "--run-minutes",
                        "0.001",
                    ]
                )

            store = AlphaStore(db_path)
            candidate = store.list_candidates()[0]
            events = store.events_for_candidate(candidate["id"])
            self.assertEqual(exit_code, 0)
            self.assertEqual(candidate["status"], "failed")
            self.assertTrue(any("INTERRUPTED_AFTER_PREFLIGHT" in event["metadata_json"] for event in events))

    def test_cli_daemon_stops_when_ai_quota_is_blocked(self):
        class FakeWorker:
            def run_once(self):
                calls.append("run_once")
                return {"generated": 0, "failed": 1, "ai_quota_blocked": 1}

        calls = []
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "alpha.db"
            env_path = self._local_env(tmp)
            stdout = io.StringIO()

            with (
                patch.dict(os.environ, {}, clear=True),
                patch("alpha.cli._worker", return_value=FakeWorker()),
                patch("alpha.cli.time.sleep", side_effect=AssertionError("daemon should stop before sleeping")) as sleep,
                redirect_stdout(stdout),
            ):
                exit_code = main(
                    [
                        "--env-file",
                        str(env_path),
                        "--db",
                        str(db_path),
                        "daemon",
                        "--batch-size",
                        "1",
                        "--loop-seconds",
                        "60",
                    ]
                )

        self.assertEqual(exit_code, 0)
        self.assertEqual(calls, ["run_once"])
        sleep.assert_not_called()
        self.assertIn("ai_quota_blocked", stdout.getvalue())

    def test_cli_daemon_backs_off_instead_of_stopping_when_ai_generation_times_out(self):
        class FakeWorker:
            def run_once(self):
                calls.append("run_once")
                return {"generated": 0, "failed": 1, "ai_generation_timeout": 1}

        calls = []
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "alpha.db"
            env_path = self._local_env(tmp)
            stdout = io.StringIO()

            with (
                patch.dict(os.environ, {}, clear=True),
                patch("alpha.cli._worker", return_value=FakeWorker()),
                patch("alpha.cli.time.monotonic", side_effect=[0.0, 0.0, 0.0, 0.7]),
                patch("alpha.cli.time.sleep") as sleep,
                redirect_stdout(stdout),
            ):
                exit_code = main(
                    [
                        "--env-file",
                        str(env_path),
                        "--db",
                        str(db_path),
                        "daemon",
                        "--batch-size",
                        "1",
                        "--loop-seconds",
                        "60",
                        "--run-minutes",
                        "0.01",
                    ]
                )

            store = AlphaStore(db_path)
            state = store.get_run_state("daemon")

        self.assertEqual(exit_code, 0)
        self.assertEqual(calls, ["run_once"])
        sleep.assert_called_once_with(0.6)
        self.assertEqual(state["stop_reason"], "time_limit")
        self.assertIn("ai_generation_timeout", stdout.getvalue())

    def test_cli_daemon_uses_ai_timeout_backoff_when_not_near_deadline(self):
        class FakeWorker:
            def run_once(self):
                calls.append("run_once")
                return {"generated": 0, "failed": 1, "ai_generation_timeout": 1}

        calls = []
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "alpha.db"
            env_path = self._local_env(tmp)
            stdout = io.StringIO()

            with (
                patch.dict(os.environ, {}, clear=True),
                patch("alpha.cli._worker", return_value=FakeWorker()),
                patch("alpha.cli.time.sleep", side_effect=KeyboardInterrupt) as sleep,
                redirect_stdout(stdout),
            ):
                exit_code = main(
                    [
                        "--env-file",
                        str(env_path),
                        "--db",
                        str(db_path),
                        "daemon",
                        "--batch-size",
                        "1",
                        "--loop-seconds",
                        "60",
                    ]
                )

        self.assertEqual(exit_code, 0)
        self.assertEqual(calls, ["run_once"])
        sleep.assert_called_once_with(600.0)
        self.assertIn("ai_generation_timeout_backoff", stdout.getvalue())

    def test_cli_daemon_backs_off_instead_of_stopping_when_ai_network_is_blocked(self):
        class FakeWorker:
            def run_once(self):
                calls.append("run_once")
                return {"generated": 0, "failed": 1, "ai_network_blocked": 1}

        calls = []
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "alpha.db"
            env_path = self._local_env(tmp)
            stdout = io.StringIO()

            with (
                patch.dict(os.environ, {}, clear=True),
                patch("alpha.cli._worker", return_value=FakeWorker()),
                patch("alpha.cli.time.sleep", side_effect=KeyboardInterrupt) as sleep,
                redirect_stdout(stdout),
            ):
                exit_code = main(
                    [
                        "--env-file",
                        str(env_path),
                        "--db",
                        str(db_path),
                        "daemon",
                        "--batch-size",
                        "1",
                        "--loop-seconds",
                        "60",
                    ]
                )

        self.assertEqual(exit_code, 0)
        self.assertEqual(calls, ["run_once"])
        sleep.assert_called_once_with(600.0)
        self.assertIn("ai_network_blocked_backoff", stdout.getvalue())

    def test_cli_daemon_continues_after_production_rescue_bad_probe_batch_without_signal(self):
        class FakeWorker:
            def run_once(self, cycle_plan=None):
                calls.append("run_once")
                if len(calls) > 1:
                    return {"generated": 1, "failed": 1}
                return {
                    "generated": 8,
                    "failed": 8,
                    "quality_stop_loss": 1,
                    "quality_stop_reason": "bad_full_batch",
                    "probe_reject": 8,
                }

        calls = []
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "alpha.db"
            env_path = self._local_env(tmp)
            stdout = io.StringIO()
            plans = [
                {
                    "mode": "production_rescue",
                    "scope": {"region": "USA", "universe": "TOP3000", "delay": 1, "neutralization": "INDUSTRY"},
                    "budget": {"batch_size": 8},
                    "reason": "quality_stop_loss_repeated",
                },
                {
                    "mode": "explore",
                    "scope": {"region": "USA", "universe": "TOP3000", "delay": 1, "neutralization": "INDUSTRY"},
                    "budget": {"batch_size": 8},
                    "reason": "production_rescue_quality_stop_loss_recent",
                    "constraints": {"avoid_modes": ["production_rescue"]},
                },
            ]

            with (
                patch.dict(os.environ, {}, clear=True),
                patch("alpha.cli._worker", return_value=FakeWorker()),
                patch("alpha.cli.build_cycle_plan", side_effect=plans),
                patch("alpha.cli.time.monotonic", side_effect=[0.0, 0.0, 0.0, 0.1, 0.7]),
                patch("alpha.cli.time.sleep") as sleep,
                redirect_stdout(stdout),
            ):
                exit_code = main(
                    [
                        "--env-file",
                        str(env_path),
                        "--db",
                        str(db_path),
                        "daemon",
                        "--batch-size",
                        "8",
                        "--loop-seconds",
                        "60",
                        "--run-minutes",
                        "0.01",
                        "--throughput-mode",
                    ]
                )

            self.assertEqual(exit_code, 0)
            self.assertEqual(calls, ["run_once", "run_once"])
            sleep.assert_called_once_with(0.6)
            self.assertIn("quality_stop_loss", stdout.getvalue())
            self.assertIn("production_rescue_quality_stop_loss", stdout.getvalue())
            self.assertIn("time_limit_reached", stdout.getvalue())
            store = AlphaStore(db_path)
            with store.connection() as conn:
                row = conn.execute("SELECT value FROM run_state WHERE key = 'daemon'").fetchone()
            self.assertIn("time_limit", row["value"])
            self.assertNotIn("production_rescue_quality_stop_loss", row["value"])

    def test_cli_daemon_continues_when_quality_stop_loss_has_probe_signal(self):
        class FakeWorker:
            def run_once(self, cycle_plan=None):
                calls.append("run_once")
                if len(calls) > 1:
                    return {"generated": 0}
                return {
                    "generated": 8,
                    "failed": 8,
                    "quality_stop_loss": 1,
                    "quality_stop_reason": "bad_full_batch",
                    "probe_optimize_ready": 1,
                    "probe_reject": 7,
                }

        calls = []
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "alpha.db"
            env_path = self._local_env(tmp)
            stdout = io.StringIO()

            with (
                patch.dict(os.environ, {}, clear=True),
                patch("alpha.cli._worker", return_value=FakeWorker()),
                patch(
                    "alpha.cli.build_cycle_plan",
                    return_value={
                        "mode": "production_rescue",
                        "scope": {"region": "USA", "universe": "TOP3000", "delay": 1, "neutralization": "INDUSTRY"},
                        "budget": {"batch_size": 8},
                        "reason": "quality_stop_loss_repeated",
                    },
                ),
                patch("alpha.cli.time.monotonic", side_effect=[0.0, 0.0, 0.0, 0.1, 0.7]),
                patch("alpha.cli.time.sleep") as sleep,
                redirect_stdout(stdout),
            ):
                exit_code = main(
                    [
                        "--env-file",
                        str(env_path),
                        "--db",
                        str(db_path),
                        "daemon",
                        "--batch-size",
                        "8",
                        "--loop-seconds",
                        "60",
                        "--run-minutes",
                        "0.01",
                        "--throughput-mode",
                    ]
                )

            self.assertEqual(exit_code, 0)
            self.assertEqual(calls, ["run_once", "run_once"])
            sleep.assert_called_once_with(0.6)
            self.assertIn("quality_stop_loss", stdout.getvalue())
            self.assertIn("time_limit_reached", stdout.getvalue())
            store = AlphaStore(db_path)
            with store.connection() as conn:
                row = conn.execute("SELECT value FROM run_state WHERE key = 'daemon'").fetchone()
            self.assertIn("time_limit", row["value"])
            self.assertNotIn("production_rescue_quality_stop_loss", row["value"])

    def test_cli_daemon_continues_after_production_rescue_probe_simulation_error(self):
        class FakeWorker:
            def run_once(self, cycle_plan=None):
                calls.append("run_once")
                if len(calls) > 1:
                    return {"generated": 1, "failed": 1}
                return {
                    "generated": 1,
                    "failed": 1,
                    "probe_simulation_error": 1,
                }

        calls = []
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "alpha.db"
            env_path = self._local_env(tmp)
            stdout = io.StringIO()
            plans = [
                {
                    "mode": "production_rescue",
                    "scope": {"region": "USA", "universe": "TOP500", "delay": 0, "neutralization": "INDUSTRY"},
                    "budget": {"batch_size": 8},
                    "reason": "quality_stop_loss_repeated",
                },
                {
                    "mode": "explore",
                    "scope": {"region": "USA", "universe": "TOP500", "delay": 0, "neutralization": "INDUSTRY"},
                    "budget": {"batch_size": 8},
                    "reason": "production_rescue_probe_simulation_error_recent",
                    "constraints": {"avoid_modes": ["production_rescue"]},
                },
            ]

            with (
                patch.dict(os.environ, {}, clear=True),
                patch("alpha.cli._worker", return_value=FakeWorker()),
                patch("alpha.cli.build_cycle_plan", side_effect=plans),
                patch("alpha.cli.time.monotonic", side_effect=[0.0, 0.0, 0.0, 0.1, 0.7]),
                patch("alpha.cli.time.sleep") as sleep,
                redirect_stdout(stdout),
            ):
                exit_code = main(
                    [
                        "--env-file",
                        str(env_path),
                        "--db",
                        str(db_path),
                        "daemon",
                        "--batch-size",
                        "8",
                        "--loop-seconds",
                        "60",
                        "--run-minutes",
                        "0.01",
                        "--throughput-mode",
                    ]
                )

            self.assertEqual(exit_code, 0)
            self.assertEqual(calls, ["run_once", "run_once"])
            sleep.assert_called_once_with(0.6)
            self.assertIn("production_rescue_probe_simulation_error", stdout.getvalue())
            self.assertIn("time_limit_reached", stdout.getvalue())
            store = AlphaStore(db_path)
            with store.connection() as conn:
                row = conn.execute("SELECT value FROM run_state WHERE key = 'daemon'").fetchone()
            self.assertIn("time_limit", row["value"])
            self.assertNotIn("production_rescue_probe_simulation_error", row["value"])

    def test_cli_daemon_continues_after_production_rescue_duplicate_only_cycle(self):
        class FakeWorker:
            def run_once(self, cycle_plan=None):
                calls.append("run_once")
                if len(calls) > 1:
                    return {"generated": 1, "failed": 1}
                return {
                    "generated": 0,
                    "approved": 0,
                    "submitted": 0,
                    "failed": 0,
                    "pending": 0,
                    "skipped": 1,
                }

        calls = []
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "alpha.db"
            env_path = self._local_env(tmp)
            stdout = io.StringIO()
            plans = [
                {
                    "mode": "production_rescue",
                    "scope": {"region": "USA", "universe": "TOP500", "delay": 0, "neutralization": "INDUSTRY"},
                    "budget": {"batch_size": 8},
                    "reason": "quality_stop_loss_repeated",
                },
                {
                    "mode": "explore",
                    "scope": {"region": "USA", "universe": "TOP500", "delay": 0, "neutralization": "INDUSTRY"},
                    "budget": {"batch_size": 8},
                    "reason": "production_rescue_duplicate_only_recent",
                    "constraints": {"avoid_modes": ["production_rescue"]},
                },
            ]

            with (
                patch.dict(os.environ, {}, clear=True),
                patch("alpha.cli._worker", return_value=FakeWorker()),
                patch("alpha.cli.build_cycle_plan", side_effect=plans),
                patch("alpha.cli.time.monotonic", side_effect=[0.0, 0.0, 0.0, 0.1, 0.7]),
                patch("alpha.cli.time.sleep") as sleep,
                redirect_stdout(stdout),
            ):
                exit_code = main(
                    [
                        "--env-file",
                        str(env_path),
                        "--db",
                        str(db_path),
                        "daemon",
                        "--batch-size",
                        "8",
                        "--loop-seconds",
                        "60",
                        "--run-minutes",
                        "0.01",
                        "--throughput-mode",
                    ]
                )

            self.assertEqual(exit_code, 0)
            self.assertEqual(calls, ["run_once", "run_once"])
            sleep.assert_called_once_with(0.6)
            self.assertIn("production_rescue_duplicate_only", stdout.getvalue())
            self.assertIn("time_limit_reached", stdout.getvalue())
            store = AlphaStore(db_path)
            with store.connection() as conn:
                row = conn.execute("SELECT value FROM run_state WHERE key = 'daemon'").fetchone()
            self.assertIn("time_limit", row["value"])
            self.assertNotIn("production_rescue_duplicate_only", row["value"])

    def test_cli_daemon_continues_after_explore_duplicate_only_cycle(self):
        class FakeWorker:
            def run_once(self, cycle_plan=None):
                calls.append("run_once")
                if len(calls) > 1:
                    return {"generated": 1, "failed": 1}
                return {
                    "generated": 0,
                    "approved": 0,
                    "submitted": 0,
                    "failed": 0,
                    "pending": 0,
                    "skipped": 1,
                }

        calls = []
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "alpha.db"
            env_path = self._local_env(tmp)
            stdout = io.StringIO()
            plans = [
                {
                    "mode": "explore",
                    "scope": {"region": "USA", "universe": "TOP500", "delay": 0, "neutralization": "INDUSTRY"},
                    "budget": {"batch_size": 8},
                    "reason": "pending_recheck_cooldown",
                },
                {
                    "mode": "production_rescue",
                    "scope": {"region": "USA", "universe": "TOP500", "delay": 0, "neutralization": "INDUSTRY"},
                    "budget": {"batch_size": 8},
                    "reason": "explore_duplicate_only_recent",
                    "constraints": {"avoid_modes": ["explore"]},
                },
            ]

            with (
                patch.dict(os.environ, {}, clear=True),
                patch("alpha.cli._worker", return_value=FakeWorker()),
                patch("alpha.cli.build_cycle_plan", side_effect=plans),
                patch("alpha.cli.time.monotonic", side_effect=[0.0, 0.0, 0.0, 0.1, 0.7]),
                patch("alpha.cli.time.sleep") as sleep,
                redirect_stdout(stdout),
            ):
                exit_code = main(
                    [
                        "--env-file",
                        str(env_path),
                        "--db",
                        str(db_path),
                        "daemon",
                        "--batch-size",
                        "8",
                        "--loop-seconds",
                        "60",
                        "--run-minutes",
                        "0.01",
                        "--throughput-mode",
                    ]
                )

            self.assertEqual(exit_code, 0)
            self.assertEqual(calls, ["run_once", "run_once"])
            sleep.assert_called_once_with(0.6)
            self.assertIn("explore_duplicate_only", stdout.getvalue())
            self.assertIn("time_limit_reached", stdout.getvalue())
            store = AlphaStore(db_path)
            with store.connection() as conn:
                row = conn.execute("SELECT value FROM run_state WHERE key = 'daemon'").fetchone()
            self.assertIn("time_limit", row["value"])
            self.assertNotIn("explore_duplicate_only", row["value"])

    def test_cli_daemon_continues_after_standardized_probe_exhaustion_cycle(self):
        class FakeWorker:
            def run_once(self, cycle_plan=None):
                calls.append("run_once")
                return {
                    "generated": 0,
                    "approved": 0,
                    "submitted": 0,
                    "failed": 0,
                    "pending": 0,
                    "skipped": 3,
                    "standardized_probe_exhausted": 1,
                }

        calls = []
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "alpha.db"
            env_path = self._local_env(tmp)
            stdout = io.StringIO()

            with (
                patch.dict(os.environ, {}, clear=True),
                patch("alpha.cli._worker", return_value=FakeWorker()),
                patch(
                    "alpha.cli.build_cycle_plan",
                    return_value={
                        "mode": "explore",
                        "scope": {"region": "USA", "universe": "TOP500", "delay": 0, "neutralization": "INDUSTRY"},
                        "budget": {"batch_size": 8},
                        "reason": "pending_recheck_cooldown",
                    },
                ),
                patch("alpha.cli.time.sleep", side_effect=KeyboardInterrupt) as sleep,
                redirect_stdout(stdout),
            ):
                exit_code = main(
                    [
                        "--env-file",
                        str(env_path),
                        "--db",
                        str(db_path),
                        "daemon",
                        "--batch-size",
                        "8",
                        "--loop-seconds",
                        "60",
                        "--throughput-mode",
                    ]
                )

            self.assertEqual(exit_code, 0)
            self.assertEqual(calls, ["run_once"])
            sleep.assert_called_once_with(60)
            self.assertNotIn("explore_duplicate_only", stdout.getvalue())
            store = AlphaStore(db_path)
            with store.connection() as conn:
                row = conn.execute("SELECT value FROM run_state WHERE key = 'daemon'").fetchone()
            self.assertIn("interrupted", row["value"])
            self.assertNotIn("explore_duplicate_only", row["value"])

    def test_cli_daemon_continues_after_production_rescue_probe_exhaustion_cycle(self):
        class FakeWorker:
            def run_once(self, cycle_plan=None):
                calls.append("run_once")
                return {
                    "generated": 0,
                    "approved": 0,
                    "submitted": 0,
                    "failed": 0,
                    "pending": 0,
                    "skipped": 2,
                    "production_rescue_probe_exhausted": 1,
                }

        calls = []
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "alpha.db"
            env_path = self._local_env(tmp)
            stdout = io.StringIO()

            with (
                patch.dict(os.environ, {}, clear=True),
                patch("alpha.cli._worker", return_value=FakeWorker()),
                patch(
                    "alpha.cli.build_cycle_plan",
                    return_value={
                        "mode": "explore",
                        "scope": {"region": "USA", "universe": "TOP500", "delay": 0, "neutralization": "INDUSTRY"},
                        "budget": {"batch_size": 8},
                        "reason": "pending_recheck_cooldown",
                    },
                ),
                patch("alpha.cli.time.sleep", side_effect=KeyboardInterrupt) as sleep,
                redirect_stdout(stdout),
            ):
                exit_code = main(
                    [
                        "--env-file",
                        str(env_path),
                        "--db",
                        str(db_path),
                        "daemon",
                        "--batch-size",
                        "8",
                        "--loop-seconds",
                        "60",
                        "--throughput-mode",
                    ]
                )

            self.assertEqual(exit_code, 0)
            self.assertEqual(calls, ["run_once"])
            sleep.assert_called_once_with(60)
            self.assertNotIn("explore_duplicate_only", stdout.getvalue())
            store = AlphaStore(db_path)
            with store.connection() as conn:
                row = conn.execute("SELECT value FROM run_state WHERE key = 'daemon'").fetchone()
            self.assertIn("interrupted", row["value"])
            self.assertNotIn("explore_duplicate_only", row["value"])

    def test_cli_daemon_continues_after_optimize_bad_batch(self):
        class FakeWorker:
            def run_once(self, cycle_plan=None):
                calls.append("run_once")
                if len(calls) > 1:
                    return {"generated": 1, "failed": 1}
                return {
                    "generated": 4,
                    "failed": 4,
                    "quality_stop_loss": 1,
                    "quality_stop_reason": "bad_full_batch",
                }

        calls = []
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "alpha.db"
            env_path = self._local_env(tmp)
            stdout = io.StringIO()
            plans = [
                {
                    "mode": "optimize",
                    "scope": {"region": "USA", "universe": "TOP500", "delay": 0, "neutralization": "INDUSTRY"},
                    "target_candidate_id": 9906,
                    "budget": {"batch_size": 4},
                    "reason": "probe_optimize_ready_candidate_has_fixable_gap",
                },
                {
                    "mode": "explore",
                    "scope": {"region": "USA", "universe": "TOP500", "delay": 0, "neutralization": "INDUSTRY"},
                    "budget": {"batch_size": 8},
                    "reason": "optimize_quality_stop_loss_recent",
                    "constraints": {"avoid_modes": ["optimize"]},
                },
            ]

            with (
                patch.dict(os.environ, {}, clear=True),
                patch("alpha.cli._worker", return_value=FakeWorker()),
                patch("alpha.cli.build_cycle_plan", side_effect=plans),
                patch("alpha.cli.time.monotonic", side_effect=[0.0, 0.0, 0.0, 0.1, 0.7]),
                patch("alpha.cli.time.sleep") as sleep,
                redirect_stdout(stdout),
            ):
                exit_code = main(
                    [
                        "--env-file",
                        str(env_path),
                        "--db",
                        str(db_path),
                        "daemon",
                        "--batch-size",
                        "8",
                        "--loop-seconds",
                        "60",
                        "--run-minutes",
                        "0.01",
                        "--throughput-mode",
                    ]
                )

            self.assertEqual(exit_code, 0)
            self.assertEqual(calls, ["run_once", "run_once"])
            sleep.assert_called_once_with(0.6)
            self.assertIn("optimize_quality_stop_loss", stdout.getvalue())
            self.assertIn("time_limit_reached", stdout.getvalue())
            store = AlphaStore(db_path)
            with store.connection() as conn:
                row = conn.execute("SELECT value FROM run_state WHERE key = 'daemon'").fetchone()
            self.assertIn("time_limit", row["value"])
            self.assertNotIn("optimize_quality_stop_loss", row["value"])

    def test_cli_run_once_scope_args_normalize_case(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "alpha.db"
            env_path = self._local_env(tmp)

            with patch.dict(os.environ, {}, clear=True):
                main(["--env-file", str(env_path), "--db", str(db_path), "init-db"])
                exit_code = main(
                    [
                        "--env-file",
                        str(env_path),
                        "--db",
                        str(db_path),
                        "run-once",
                        "--region",
                        "chn",
                        "--universe",
                        "top2000u",
                        "--delay",
                        "0",
                        "--neutralization",
                        "subindustry",
                        "--batch-size",
                        "1",
                    ]
                )

            settings = self._latest_settings(db_path)
            self.assertEqual(exit_code, 0)
            self.assertEqual(settings["region"], "CHN")
            self.assertEqual(settings["universe"], "TOP2000U")
            self.assertEqual(settings["neutralization"], "SUBINDUSTRY")

    def test_cli_presets_prints_available_scope_choices(self):
        stdout = io.StringIO()
        with patch.dict(os.environ, {}, clear=True), redirect_stdout(stdout):
            exit_code = main(["--env-file", "/tmp/nonexistent-alpha-env", "presets"])

        output = stdout.getvalue()
        self.assertEqual(exit_code, 0)
        self.assertIn("chn-d0: region=CHN universe=TOP2000U delay=0", output)
        self.assertIn("ind: region=IND universe=TOP500 delay=1", output)
        self.assertIn("glb: region=GLB universe=TOP3000 delay=1", output)

    def test_cli_status_prints_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "alpha.db"
            env_path = self._local_env(tmp)

            with patch.dict(os.environ, {}, clear=True):
                main(["--env-file", str(env_path), "--db", str(db_path), "init-db"])
                main(["--env-file", str(env_path), "--db", str(db_path), "run-once", "--batch-size", "1"])

                stdout = io.StringIO()
                with redirect_stdout(stdout):
                    exit_code = main(["--env-file", str(env_path), "--db", str(db_path), "status"])

            self.assertEqual(exit_code, 0)
            self.assertIn("approved", stdout.getvalue())

    def test_cli_submit_approved_runs_dry_run_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "alpha.db"
            env_path = self._local_env(tmp)

            with patch.dict(os.environ, {}, clear=True):
                main(["--env-file", str(env_path), "--db", str(db_path), "init-db"])
                main(["--env-file", str(env_path), "--db", str(db_path), "run-once", "--batch-size", "1"])

                stdout = io.StringIO()
                with redirect_stdout(stdout):
                    exit_code = main(["--env-file", str(env_path), "--db", str(db_path), "submit-approved"])

            self.assertEqual(exit_code, 0)
            self.assertIn("'dry_run': 1", stdout.getvalue())

    def test_cli_check_ai_works_with_local_client(self):
        with tempfile.TemporaryDirectory() as tmp:
            env_path = Path(tmp) / ".env"
            env_path.write_text("AI_CLIENT=local\n", encoding="utf-8")

            stdout = io.StringIO()
            with patch.dict(os.environ, {}, clear=True), redirect_stdout(stdout):
                exit_code = main(["--env-file", str(env_path), "check-ai"])

            self.assertEqual(exit_code, 0)
            self.assertIn("rank(mdl_mock_score)", stdout.getvalue())

    def test_cli_check_ai_supports_multi_model_client(self):
        seen = {}

        class FakeMultiClient:
            def generate_candidates(self, batch_size, context):
                seen["batch_size"] = batch_size
                seen["context"] = context
                from alpha.models import CandidateSpec

                return [
                    CandidateSpec(
                        "group_rank(ts_rank(mdl_mock_score,22),industry)",
                        source="model:gemini",
                        metadata={"model": "gemini-3-flash-free"},
                    )
                ]

        with tempfile.TemporaryDirectory() as tmp:
            env_path = Path(tmp) / ".env"
            env_path.write_text("AI_CLIENT=multi\nBRAIN_CLIENT=local\n", encoding="utf-8")

            stdout = io.StringIO()
            with patch.dict(os.environ, {}, clear=True), patch(
                "alpha.cli.MultiModelAIClient.from_env", return_value=FakeMultiClient()
            ), redirect_stdout(stdout):
                exit_code = main(["--env-file", str(env_path), "check-ai"])

            self.assertEqual(exit_code, 0)
            self.assertIn("group_rank", stdout.getvalue())
            self.assertEqual(seen["batch_size"], 1)
            self.assertIn("research_context", seen["context"])

    def test_cli_fields_prints_scope_datafield_pool(self):
        with tempfile.TemporaryDirectory() as tmp:
            env_path = self._local_env(tmp)

            stdout = io.StringIO()
            with patch.dict(os.environ, {}, clear=True), redirect_stdout(stdout):
                exit_code = main(
                    [
                        "--env-file",
                        str(env_path),
                        "fields",
                        "--preset",
                        "ind",
                        "--limit",
                        "6",
                    ]
                )

            output = stdout.getvalue()
            self.assertEqual(exit_code, 0)
            self.assertIn("scope: region=IND universe=TOP500 delay=1", output)
            self.assertIn("field_ids:", output)
            self.assertIn("close", output)

    def test_cli_plan_next_prints_scheduler_plan(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "alpha.db"
            env_path = self._local_env(tmp)
            with patch.dict(os.environ, {}, clear=True):
                self.assertEqual(main(["--env-file", str(env_path), "--db", str(db_path), "init-db"]), 0)

                with patch("builtins.print") as printed:
                    exit_code = main(["--env-file", str(env_path), "--db", str(db_path), "plan-next"])

            self.assertEqual(exit_code, 0)
            payload = json.loads(printed.call_args.args[0])
            self.assertEqual(payload["mode"], "explore")
            self.assertIn("reason", payload)

    def test_cli_status_efficiency_prints_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "alpha.db"
            env_path = self._local_env(tmp)
            with patch.dict(os.environ, {}, clear=True):
                self.assertEqual(main(["--env-file", str(env_path), "--db", str(db_path), "init-db"]), 0)

                with patch("builtins.print") as printed:
                    exit_code = main(["--env-file", str(env_path), "--db", str(db_path), "status", "--efficiency"])

            self.assertEqual(exit_code, 0)
            output = "\n".join(str(call.args[0]) for call in printed.call_args_list)
            self.assertIn("generated:", output)
            self.assertIn("preflight_pass_rate:", output)

    def test_cli_daemon_throughput_mode_passes_cycle_plan_to_worker(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "alpha.db"
            env_path = self._local_env(tmp)
            with patch.dict(os.environ, {}, clear=True):
                self.assertEqual(main(["--env-file", str(env_path), "--db", str(db_path), "init-db"]), 0)

                with patch("alpha.cli.time.sleep", side_effect=KeyboardInterrupt):
                    exit_code = main(
                        [
                            "--env-file",
                            str(env_path),
                            "--db",
                            str(db_path),
                            "daemon",
                            "--throughput-mode",
                            "--batch-size",
                            "1",
                            "--loop-seconds",
                            "1",
                        ]
                    )

            self.assertEqual(exit_code, 0)
            events = AlphaStore(db_path).events_for_candidate(None)
            self.assertTrue(any(event["event_type"] == "cycle_plan" for event in events))

    def _latest_settings(self, db_path: Path) -> dict:
        with sqlite3.connect(db_path) as conn:
            row = conn.execute("SELECT settings_json FROM candidates ORDER BY id DESC LIMIT 1").fetchone()
        self.assertIsNotNone(row)
        return json.loads(row[0])

    def test_cli_web_defaults_to_loopback(self):
        with tempfile.TemporaryDirectory() as tmp:
            env_path = self._local_env(tmp)
            captured = {}

            def fake_run_web_app(**kwargs):
                captured.update(kwargs)
                return 0

            with patch.dict(os.environ, {}, clear=True):
                with patch("alpha.web.run_web_app", fake_run_web_app):
                    main(["--env-file", str(env_path), "--db", str(Path(tmp) / "alpha.db"), "web"])

            self.assertEqual(captured["host"], "127.0.0.1")
            self.assertEqual(captured["port"], 8080)

    def test_cli_web_host_port_env_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            env_path = self._local_env(tmp)
            captured = {}

            def fake_run_web_app(**kwargs):
                captured.update(kwargs)
                return 0

            with patch.dict(os.environ, {"ALPHA_WEB_HOST": "0.0.0.0", "ALPHA_WEB_PORT": "5000"}, clear=True):
                with patch("alpha.web.run_web_app", fake_run_web_app):
                    main(["--env-file", str(env_path), "--db", str(Path(tmp) / "alpha.db"), "web"])

            self.assertEqual(captured["host"], "0.0.0.0")
            self.assertEqual(captured["port"], 5000)

    def test_cli_web_flag_overrides_env(self):
        with tempfile.TemporaryDirectory() as tmp:
            env_path = self._local_env(tmp)
            captured = {}

            def fake_run_web_app(**kwargs):
                captured.update(kwargs)
                return 0

            with patch.dict(os.environ, {"ALPHA_WEB_HOST": "0.0.0.0"}, clear=True):
                with patch("alpha.web.run_web_app", fake_run_web_app):
                    main(["--env-file", str(env_path), "--db", str(Path(tmp) / "alpha.db"), "web", "--host", "192.168.1.5"])

            self.assertEqual(captured["host"], "192.168.1.5")


if __name__ == "__main__":
    unittest.main()
