from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from alpha.db import AlphaStore
from alpha.health import daemon_health


def _iso(minutes_ago: int) -> str:
    return (datetime.now(timezone.utc) - timedelta(minutes=minutes_ago)).replace(microsecond=0).isoformat()


class HealthTests(unittest.TestCase):
    def test_daemon_health_reports_stopped_when_no_running_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = AlphaStore(Path(tmp) / "alpha.db")
            store.init()

            health = daemon_health(store)

            self.assertEqual(health["status"], "stopped")
            self.assertEqual(health["stalled"], False)

    def test_daemon_health_reports_stalled_running_daemon_without_recent_events(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = AlphaStore(Path(tmp) / "alpha.db")
            store.init()
            store.set_run_state("daemon", {"status": "running", "pid": 123, "started_at": _iso(120)})

            health = daemon_health(store, stall_minutes=60)

            self.assertEqual(health["status"], "running")
            self.assertEqual(health["stalled"], True)

    def test_daemon_health_reports_recent_block_reason(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = AlphaStore(Path(tmp) / "alpha.db")
            store.init()
            store.set_run_state("daemon", {"status": "stopped", "stop_reason": "ai_quota_blocked"})
            store.record_event(None, "daemon_stopped", {"reason": "ai_quota_blocked"})

            health = daemon_health(store)

            self.assertEqual(health["last_block_reason"], "ai_quota_blocked")

    def test_daemon_health_ignores_stale_block_reason_while_running(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = AlphaStore(Path(tmp) / "alpha.db")
            store.init()
            store.record_event(None, "daemon_stopped", {"reason": "interrupted"})
            store.set_run_state("daemon", {"status": "running", "pid": 123, "started_at": _iso(1), "stop_reason": ""})

            health = daemon_health(store)

            self.assertEqual(health["status"], "running")
            self.assertEqual(health["last_block_reason"], "")


if __name__ == "__main__":
    unittest.main()
