from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from simrig.huggingface import (
    is_huggingface_ref,
    parse_huggingface_ref,
    resolve_policy_checkpoint,
)


class HuggingFaceTests(unittest.TestCase):
    def test_parse_huggingface_policy_ref(self) -> None:
        ref = parse_huggingface_ref(
            "hf://simrig/go1-policy/checkpoints/policy.params",
            revision="main",
        )

        self.assertEqual(ref.repo_id, "simrig/go1-policy")
        self.assertEqual(ref.filename, "checkpoints/policy.params")
        self.assertEqual(ref.revision, "main")

    def test_rejects_incomplete_huggingface_ref(self) -> None:
        with self.assertRaises(ValueError):
            parse_huggingface_ref("hf://simrig/go1-policy")

    def test_local_checkpoint_resolves_without_hub_dependency(self) -> None:
        checkpoint = resolve_policy_checkpoint("runs/policy.params")

        self.assertEqual(checkpoint, Path("runs/policy.params"))
        self.assertFalse(is_huggingface_ref(str(checkpoint)))

    def test_hub_checkpoint_downloads_sibling_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            snapshot = Path(tmp) / "snapshot"
            snapshot.mkdir()
            (snapshot / "policy.params").touch()
            calls = []

            def snapshot_download(**kwargs):
                calls.append(kwargs)
                return str(snapshot)

            module = types.ModuleType("huggingface_hub")
            module.snapshot_download = snapshot_download
            with patch.dict(sys.modules, {"huggingface_hub": module}):
                checkpoint = resolve_policy_checkpoint(
                    "hf://ssuleiman/simrig-vision-cartpole/policy.params",
                    hf_revision="main",
                    hf_token="test-token",
                )

        self.assertEqual(checkpoint, snapshot / "policy.params")
        self.assertEqual(
            calls,
            [
                {
                    "repo_id": "ssuleiman/simrig-vision-cartpole",
                    "revision": "main",
                    "token": "test-token",
                    "allow_patterns": ["policy.params", "config.json"],
                }
            ],
        )


if __name__ == "__main__":
    unittest.main()
