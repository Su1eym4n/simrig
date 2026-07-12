from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from simrig.presets import resolve_small_network


class PresetTests(unittest.TestCase):
    def test_resolve_small_network_reads_run_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            checkpoint = run_dir / "policy.params"
            checkpoint.write_text("params\n", encoding="utf-8")
            (run_dir / "config.json").write_text(
                json.dumps({"config": {"small_network": True}}),
                encoding="utf-8",
            )

            self.assertTrue(resolve_small_network(checkpoint))
            self.assertFalse(resolve_small_network(checkpoint, small_network=False))

    def test_resolve_small_network_defaults_to_large(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "policy.params"
            checkpoint.write_text("params\n", encoding="utf-8")

            self.assertFalse(resolve_small_network(checkpoint))


if __name__ == "__main__":
    unittest.main()
