from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from simrig.scaffold import REQUIRED_SECTION_MARKERS, new_env
from simrig.validate_env import validate_env


class ScaffoldTests(unittest.TestCase):
    def test_new_env_writes_section_markers_and_methods(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = new_env("reach bot", "models/bot.xml", root=tmp)
            text = path.read_text(encoding="utf-8")

            self.assertEqual(path, Path(tmp) / "reach_bot.py")
            self.assertIn("NOT TRAINABLE YET", text)
            for marker in REQUIRED_SECTION_MARKERS:
                self.assertIn(marker, text)
            self.assertIn("def reset", text)
            self.assertIn("def step", text)
            self.assertIn("observation_size", text)
            self.assertIn("action_size", text)

    def test_new_env_refuses_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            new_env("bot", "models/bot.xml", root=tmp)
            with self.assertRaises(FileExistsError):
                new_env("bot", "models/bot.xml", root=tmp)


class ValidateEnvTests(unittest.TestCase):
    def test_validate_fresh_scaffold_passes_with_warnings(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = new_env("reach", "models/bot.xml", root=tmp)
            result = validate_env(path)

            self.assertTrue(result.passed)
            self.assertFalse(result.trainable)
            self.assertEqual(result.missing, [])
            self.assertTrue(any("NotImplementedError" in w for w in result.warnings))

    def test_validate_missing_file_fails(self) -> None:
        result = validate_env("/tmp/simrig_missing_env_does_not_exist.py")

        self.assertFalse(result.passed)
        self.assertTrue(any("file not found" in item for item in result.missing))

    def test_validate_incomplete_file_reports_missing_sections(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "broken.py"
            path.write_text(
                "MODEL_PATH = 'x'\nENV_NAME = 'y'\n\ndef default_config():\n    return {}\n\n"
                "class CustomEnv:\n    def __init__(self):\n        pass\n",
                encoding="utf-8",
            )
            result = validate_env(path)

            self.assertFalse(result.passed)
            self.assertTrue(any("section marker missing" in item for item in result.missing))
            self.assertTrue(any("CustomEnv method missing: reset" in item for item in result.missing))


if __name__ == "__main__":
    unittest.main()
