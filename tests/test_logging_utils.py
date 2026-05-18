from __future__ import annotations

import logging
import tempfile
import unittest
from pathlib import Path

from alpha.logging_utils import setup_logging


class LoggingUtilsTests(unittest.TestCase):
    def test_setup_logging_writes_to_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "alpha.log"

            setup_logging(log_path)
            logging.getLogger("alpha.test").info("hello-log")

            text = log_path.read_text(encoding="utf-8")
            self.assertIn("hello-log", text)


if __name__ == "__main__":
    unittest.main()
