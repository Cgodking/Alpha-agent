from __future__ import annotations

import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from alpha.field_catalog import DEFAULT_FIELD_LIMIT, build_field_catalog
from alpha.field_catalog import _cache_path, _field_search_terms


class FieldCatalogTests(unittest.TestCase):
    def test_build_field_catalog_normalizes_legacy_cached_catalog(self):
        class CachedBrain:
            pass

        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)
            target = {"instrumentType": "EQUITY", "region": "USA", "universe": "TOP3000", "delay": 0}
            env = {
                "ALPHA_FIELD_CACHE_DIR": str(cache_dir),
                "ALPHA_FIELD_SEARCHES": "model",
                "ALPHA_FIELD_LIMIT": str(DEFAULT_FIELD_LIMIT),
            }
            with patch.dict(os.environ, env, clear=False):
                cache_path = _cache_path(
                    cache_dir,
                    target,
                    _field_search_terms(),
                    DEFAULT_FIELD_LIMIT,
                    "CachedBrain",
                )
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                cache_path.write_text(
                    json.dumps(
                        {
                            "created_at": time.time(),
                            "catalog": {
                                "available": True,
                                "field_ids": ["open", "matrix_signal", "vector_signal"],
                                "fields": [
                                    {"id": "matrix_signal", "type": "MATRIX"},
                                    {"id": "vector_signal", "type": "VECTOR"},
                                ],
                                "rules": ["Use only field_ids listed here plus standard price fields."],
                            },
                        }
                    ),
                    encoding="utf-8",
                )

                catalog = build_field_catalog(CachedBrain(), target)

        self.assertEqual(catalog["field_types"]["matrix_signal"], "MATRIX")
        self.assertEqual(catalog["field_types"]["vector_signal"], "VECTOR")
        self.assertIn("matrix_signal", catalog["matrix_fields"])
        self.assertIn("vector_signal", catalog["vector_fields"])
        self.assertTrue(any("single-argument vec_* reducer" in rule for rule in catalog["rules"]))

    def test_default_field_searches_cover_non_model_non_pv_towers_before_crowded_terms(self):
        terms = _field_search_terms()

        for required in ("option", "risk", "shortinterest", "macro", "earnings"):
            self.assertIn(required, terms)
        self.assertLess(terms.index("option"), terms.index("model"))
        self.assertLess(terms.index("risk"), terms.index("pv"))
        self.assertEqual(terms[-1], "")

    def test_field_search_terms_preserves_explicit_blank_lookup_from_trailing_comma(self):
        with patch.dict(os.environ, {"ALPHA_FIELD_SEARCHES": "option,risk,"}, clear=False):
            terms = _field_search_terms()

        self.assertEqual(terms, ["option", "risk", ""])


if __name__ == "__main__":
    unittest.main()
