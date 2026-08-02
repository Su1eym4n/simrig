from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from simrig.runtime import runtime_mismatches, verify_checkpoint_runtime


class RuntimeCompatibilityTests(unittest.TestCase):
    def test_mismatch_compares_python_minor_and_packages(self) -> None:
        expected = {
            "python": "3.12.3",
            "packages": {"playground": "0.2.0", "mujoco": "3.10.0"},
        }
        current = {
            "python": "3.14.1",
            "packages": {"playground": "0.2.0", "mujoco": "3.11.0"},
        }

        mismatches = runtime_mismatches(expected, current)

        self.assertTrue(any(item.startswith("python:") for item in mismatches))
        self.assertTrue(any(item.startswith("mujoco:") for item in mismatches))
        self.assertFalse(any(item.startswith("playground:") for item in mismatches))

    def test_checkpoint_mismatch_requires_explicit_override(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / "policy.params"
            checkpoint.write_bytes(b"policy")
            (root / "config.json").write_text(
                json.dumps(
                    {
                        "config": {
                            "runtime": {
                                "python": "3.12.0",
                                "packages": {"playground": "0.2.0"},
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            current = {
                "python": "3.14.0",
                "packages": {"playground": "0.3.0"},
            }
            with (
                patch("simrig.runtime.runtime_manifest", return_value=current),
                self.assertRaises(RuntimeError),
            ):
                verify_checkpoint_runtime(checkpoint)

            with (
                patch("simrig.runtime.runtime_manifest", return_value=current),
                self.assertWarns(RuntimeWarning),
            ):
                result = verify_checkpoint_runtime(checkpoint, allow_mismatch=True)

        self.assertFalse(result["compatible"])
        self.assertEqual(len(result["mismatches"]), 2)

    def test_legacy_checkpoint_without_manifest_remains_loadable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "policy.params"
            checkpoint.write_bytes(b"policy")

            result = verify_checkpoint_runtime(checkpoint)

        self.assertFalse(result["recorded"])
        self.assertIsNone(result["compatible"])


if __name__ == "__main__":
    unittest.main()
