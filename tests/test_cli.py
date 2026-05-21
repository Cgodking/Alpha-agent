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

    def _latest_settings(self, db_path: Path) -> dict:
        with sqlite3.connect(db_path) as conn:
            row = conn.execute("SELECT settings_json FROM candidates ORDER BY id DESC LIMIT 1").fetchone()
        self.assertIsNotNone(row)
        return json.loads(row[0])


if __name__ == "__main__":
    unittest.main()
