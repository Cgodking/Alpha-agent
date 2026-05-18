from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from alpha.config import load_config


class ConfigTests(unittest.TestCase):
    def test_config_defaults_to_local_clients_and_dry_run(self):
        with patch.dict(os.environ, {}, clear=True):
            cfg = load_config()

        self.assertEqual(cfg.ai_client, "local")
        self.assertEqual(cfg.brain_client, "local")
        self.assertFalse(cfg.policy.auto_submit)

    def test_config_reads_client_selection_from_env(self):
        with patch.dict(os.environ, {"AI_CLIENT": "openai", "BRAIN_CLIENT": "http", "AUTO_SUBMIT": "true"}):
            cfg = load_config()

        self.assertEqual(cfg.ai_client, "openai")
        self.assertEqual(cfg.brain_client, "http")
        self.assertTrue(cfg.policy.auto_submit)

    def test_config_reads_simulation_scope_from_env(self):
        with patch.dict(
            os.environ,
            {
                "ALPHA_REGION": "USA",
                "ALPHA_UNIVERSE": "TOP3000",
                "ALPHA_DELAY": "0",
                "ALPHA_NEUTRALIZATION": "SUBINDUSTRY",
                "ALPHA_DECAY": "6",
                "ALPHA_TRUNCATION": "0.03",
            },
            clear=True,
        ):
            cfg = load_config()

        self.assertEqual(cfg.simulation_context["region"], "USA")
        self.assertEqual(cfg.simulation_context["universe"], "TOP3000")
        self.assertEqual(cfg.simulation_context["delay"], 0)
        self.assertEqual(cfg.simulation_context["neutralization"], "SUBINDUSTRY")
        self.assertEqual(cfg.simulation_context["decay"], 6)
        self.assertEqual(cfg.simulation_context["truncation"], 0.03)


if __name__ == "__main__":
    unittest.main()
