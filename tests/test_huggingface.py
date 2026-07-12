from __future__ import annotations

from pathlib import Path
import unittest

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


if __name__ == "__main__":
    unittest.main()
