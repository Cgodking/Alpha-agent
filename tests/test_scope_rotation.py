from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from alpha.db import AlphaStore
from alpha.scope_rotation import next_rotating_scope, parse_scope_json


class ScopeRotationTests(unittest.TestCase):
    def test_next_rotating_scope_persists_cursor_across_store_instances(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "alpha.db"
            store = AlphaStore(db_path)
            store.init()
            scopes = [
                {"region": "USA", "universe": "TOP3000", "delay": 0, "neutralization": "INDUSTRY"},
                {"region": "CHN", "universe": "TOP2000U", "delay": 0, "neutralization": "INDUSTRY"},
            ]

            first = next_rotating_scope(store, scopes)
            restarted_store = AlphaStore(db_path)
            restarted_store.init()
            second = next_rotating_scope(restarted_store, scopes)
            third = next_rotating_scope(restarted_store, scopes)

            self.assertEqual(first["region"], "USA")
            self.assertEqual(second["region"], "CHN")
            self.assertEqual(third["region"], "USA")
            state = restarted_store.get_run_state("scope_rotation")
            self.assertEqual(state["last_index"], 0)
            self.assertEqual(state["next_index"], 1)

    def test_parse_scope_json_applies_base_defaults_and_normalizes_scope(self):
        scopes = parse_scope_json(
            {"decay": 6, "truncation": 0.03},
            json.dumps([{"region": "chn", "universe": "top2000u", "delay": 0, "neutralization": "subindustry"}]),
        )

        self.assertEqual(scopes[0]["region"], "CHN")
        self.assertEqual(scopes[0]["universe"], "TOP2000U")
        self.assertEqual(scopes[0]["neutralization"], "SUBINDUSTRY")
        self.assertEqual(scopes[0]["decay"], 6)
        self.assertEqual(scopes[0]["truncation"], 0.03)


if __name__ == "__main__":
    unittest.main()
