from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from simrig.mujoco_backend import list_models, resolve_model_path
from simrig.paths import find_menagerie


def _fake_menagerie(root: Path) -> Path:
    model = root / "unit_test_bot"
    model.mkdir(parents=True)
    (root / "README.md").write_text("menagerie\n", encoding="utf-8")
    (model / "README.md").write_text("bot\n", encoding="utf-8")
    (model / "LICENSE").write_text("MIT\n", encoding="utf-8")
    (model / "scene.xml").write_text("<mujoco model='x'/>\n", encoding="utf-8")
    (model / "scene_mjx.xml").write_text("<mujoco model='x'/>\n", encoding="utf-8")
    return root


class PathTests(unittest.TestCase):
    def test_find_menagerie_from_explicit_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = _fake_menagerie(Path(tmp))

            self.assertEqual(find_menagerie(root), root.resolve())

    def test_list_models_reports_xmls(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = _fake_menagerie(Path(tmp))

            entries = list_models(root)

            self.assertEqual(entries[0]["name"], "unit_test_bot")
            self.assertEqual(
                entries[0]["scene_xmls"],
                ["unit_test_bot/scene.xml", "unit_test_bot/scene_mjx.xml"],
            )
            self.assertEqual(entries[0]["mjx_xmls"], ["unit_test_bot/scene_mjx.xml"])

    def test_resolve_model_path_by_directory_name(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = _fake_menagerie(Path(tmp))

            path = resolve_model_path("unit_test_bot", menagerie=root)

            self.assertEqual(path.name, "scene_mjx.xml")


if __name__ == "__main__":
    unittest.main()

