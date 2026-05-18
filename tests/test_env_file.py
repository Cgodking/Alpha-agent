from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from alpha.env_file import load_env_file


class EnvFileTests(unittest.TestCase):
    def test_load_env_file_sets_missing_values_without_overriding_existing_env(self):
        with tempfile.TemporaryDirectory() as tmp:
            env_path = Path(tmp) / ".env"
            env_path.write_text(
                "\n".join(
                    [
                        "AI_CLIENT=openai",
                        "AI_API_KEY=from-file",
                        "AI_MODEL=\"model-z\"",
                        "# ignored comment",
                    ]
                ),
                encoding="utf-8",
            )

            with patch.dict(os.environ, {"AI_API_KEY": "from-env"}, clear=True):
                loaded = load_env_file(env_path)

                self.assertEqual(loaded["AI_CLIENT"], "openai")
                self.assertEqual(os.environ["AI_CLIENT"], "openai")
                self.assertEqual(os.environ["AI_API_KEY"], "from-env")
                self.assertEqual(os.environ["AI_MODEL"], "model-z")


if __name__ == "__main__":
    unittest.main()
