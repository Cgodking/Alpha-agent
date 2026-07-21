from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from alpha.db import AlphaStore
from alpha.models import DEFAULT_SETTINGS


class AlphaStoreTests(unittest.TestCase):
    def test_store_initializes_schema_and_records_candidate(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = AlphaStore(Path(tmp) / "alpha.db")
            store.init()

            candidate_id = store.insert_candidate(
                expression="rank(close)",
                settings={"region": "USA", "universe": "TOP3000", "delay": 1},
                source="local_ai",
            )

            candidate = store.get_candidate(candidate_id)
            self.assertEqual(candidate["expression"], "rank(close)")
            self.assertEqual(candidate["status"], "generated")
            self.assertEqual(json.loads(candidate["settings_json"])["region"], "USA")

    def test_store_records_structured_events(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = AlphaStore(Path(tmp) / "alpha.db")
            store.init()
            candidate_id = store.insert_candidate("rank(close)", {"region": "USA"}, "local_ai")

            store.record_event(candidate_id, "generated", {"batch": 1})

            events = store.events_for_candidate(candidate_id)
            self.assertEqual(len(events), 1)
            self.assertEqual(events[0]["event_type"], "generated")
            self.assertEqual(json.loads(events[0]["metadata_json"]), {"batch": 1})

    def test_store_reads_and_writes_run_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = AlphaStore(Path(tmp) / "alpha.db")
            store.init()

            store.set_run_state("daemon", {"pid": 123, "status": "running"})

            self.assertEqual(store.get_run_state("daemon")["pid"], 123)
            self.assertEqual(store.get_run_state("missing", {"status": "none"})["status"], "none")

    def test_store_connect_enables_wal_and_busy_timeout(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = AlphaStore(Path(tmp) / "alpha.db")
            store.init()

            conn = store.connect()
            try:
                journal_mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
                busy_timeout = conn.execute("PRAGMA busy_timeout").fetchone()[0]
                foreign_keys = conn.execute("PRAGMA foreign_keys").fetchone()[0]
            finally:
                conn.close()

            self.assertEqual(str(journal_mode).lower(), "wal")
            self.assertEqual(int(busy_timeout), 30000)
            self.assertEqual(int(foreign_keys), 1)

    def test_store_initializes_query_indexes(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = AlphaStore(Path(tmp) / "alpha.db")
            store.init()

            conn = store.connect()
            try:
                index_names = {
                    row[0]
                    for row in conn.execute(
                        "SELECT name FROM sqlite_master WHERE type = 'index'"
                    ).fetchall()
                }
            finally:
                conn.close()

            self.assertIn("idx_candidates_status_created_id", index_names)
            self.assertIn("idx_candidates_created_id", index_names)
            self.assertIn("idx_events_candidate_id_id", index_names)
            self.assertIn("idx_events_candidate_type_id", index_names)

    def test_store_finds_existing_candidate_by_expression_and_simulation_scope(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = AlphaStore(Path(tmp) / "alpha.db")
            store.init()
            settings = dict(DEFAULT_SETTINGS)
            settings.update({"region": "USA", "universe": "TOP3000", "delay": 0, "neutralization": "INDUSTRY"})
            candidate_id = store.insert_candidate("rank( close )", settings, "openai_compatible")

            duplicate = store.find_duplicate_candidate(
                "rank(close)",
                {"region": "usa", "universe": "top3000", "delay": 0, "neutralization": "industry"},
            )
            different_settings = store.find_duplicate_candidate(
                "rank(close)",
                {"region": "USA", "universe": "TOP3000", "delay": 0, "neutralization": "SUBINDUSTRY"},
            )

            self.assertIsNotNone(duplicate)
            self.assertEqual(duplicate["id"], candidate_id)
            self.assertIsNone(different_settings)

    def test_store_lists_recent_candidates_with_limit_and_status_filter(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = AlphaStore(Path(tmp) / "alpha.db")
            store.init()
            first_id = store.insert_candidate("rank(first_signal)", {"region": "USA"}, "local_ai")
            second_id = store.insert_candidate("rank(second_signal)", {"region": "USA"}, "local_ai")
            third_id = store.insert_candidate("rank(third_signal)", {"region": "USA"}, "local_ai")
            store.transition(second_id, "failed")
            store.transition(third_id, "failed")

            recent = store.list_recent_candidates(limit=2)
            failed = store.list_recent_candidates(limit=5, status="failed")

            self.assertEqual([row["id"] for row in recent], [third_id, second_id])
            self.assertEqual([row["id"] for row in failed], [third_id, second_id])
            self.assertNotIn(first_id, [row["id"] for row in recent])

    def test_store_archives_candidates_and_keeps_duplicate_memory(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = AlphaStore(Path(tmp) / "alpha.db")
            store.init()
            settings = {"region": "USA", "universe": "TOP3000", "delay": 1, "neutralization": "INDUSTRY"}
            candidate_id = store.insert_candidate("rank(stale_signal)", settings, "model:G-1")
            store.record_event(candidate_id, "generated", {"batch": 1})
            store.transition(candidate_id, "failed", {"errors": ["LOW_SHARPE:FAIL"]})

            archived = store.archive_candidates([candidate_id], "low_quality_history", {"quality_score": 0.0})
            duplicate = store.find_duplicate_candidate("rank(stale_signal)", settings)

            self.assertEqual(archived, 1)
            with self.assertRaises(KeyError):
                store.get_candidate(candidate_id)
            self.assertIsNotNone(duplicate)
            self.assertEqual(duplicate["id"], candidate_id)
            self.assertTrue(duplicate["archived"])
            with store.connection() as conn:
                archived_candidate_count = conn.execute("SELECT COUNT(*) FROM archived_candidates").fetchone()[0]
                archived_event_count = conn.execute("SELECT COUNT(*) FROM archived_events").fetchone()[0]
            self.assertEqual(archived_candidate_count, 1)
            self.assertEqual(archived_event_count, 2)


if __name__ == "__main__":
    unittest.main()
