from __future__ import annotations

from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

from simrig.preview import (
    _command_controls,
    _command_from_query,
    _checkpoint_episode_horizon,
    _episode_horizon,
    _frame_html,
    _rate_hz,
    _threejs_html,
)


class _State:
    def __init__(self, info: dict[str, object]) -> None:
        self.info = info


class PreviewMetadataTests(unittest.TestCase):
    def test_non_command_environment_has_no_controls(self) -> None:
        self.assertEqual(_command_controls(object(), _State({})), [])

    def test_three_axis_command_gets_locomotion_defaults(self) -> None:
        controls = _command_controls(
            object(),
            _State({"command": np.zeros(3, dtype=float)}),
        )

        self.assertEqual(
            [control["label"] for control in controls],
            ["Forward X", "Lateral Y", "Yaw"],
        )
        self.assertEqual([control["unit"] for control in controls], ["m/s", "m/s", "rad/s"])

    def test_environment_can_declare_arbitrary_command_names(self) -> None:
        class Env:
            def command_spec(self) -> list[str]:
                return ["a", "b", "c"]

        controls = _command_controls(Env(), _State({"command": np.zeros(3)}))

        self.assertEqual([control["key"] for control in controls], ["a", "b", "c"])
        self.assertEqual([control["label"] for control in controls], ["a", "b", "c"])

    def test_environment_can_declare_command_field_metadata(self) -> None:
        class Env:
            command_spec = {
                "controls": [
                    {"key": "jump", "label": "Jump Height", "unit": "m", "min": 0},
                    {"key": "spin", "label": "Spin", "unit": "rad/s", "step": 0.25},
                ]
            }

        controls = _command_controls(Env(), _State({"command": np.zeros(2)}))

        self.assertEqual(controls[0]["label"], "Jump Height")
        self.assertEqual(controls[0]["min"], 0)
        self.assertEqual(controls[1]["step"], 0.25)

    def test_command_spec_size_must_match_command_vector(self) -> None:
        class Env:
            command_spec = ["j", "k", "i"]

        with self.assertRaisesRegex(ValueError, "control count"):
            _command_controls(Env(), _State({"command": np.zeros(2)}))

    def test_command_query_accepts_dynamic_repeated_values(self) -> None:
        self.assertEqual(
            _command_from_query({"value": ["1.5", "-2", "0.25"]}),
            [1.5, -2.0, 0.25],
        )
        self.assertIsNone(_command_from_query({"clear": ["1"]}))

    def test_rate_and_episode_metadata(self) -> None:
        env = SimpleNamespace(config={"episode_length": 750})
        self.assertEqual(_rate_hz(0.004), 250.0)
        self.assertEqual(_episode_horizon(env), 750)

    def test_checkpoint_episode_horizon_uses_recorded_training_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "config.json").write_text(
                '{"config": {"episode_length": 1000}}',
                encoding="utf-8",
            )
            checkpoint = root / "policy.params"
            checkpoint.touch()

            self.assertEqual(_checkpoint_episode_horizon(checkpoint), 1000)

    def test_preview_pages_use_dynamic_controls_and_episode_status(self) -> None:
        for page in (_threejs_html(), _frame_html()):
            self.assertIn('id="command-section" hidden', page)
            self.assertIn('id="command-controls"', page)
            self.assertIn('id="episode-state"', page)
            self.assertIn('id="auto-reset"', page)
            self.assertNotIn('<label>Forward X</label>', page)


if __name__ == "__main__":
    unittest.main()
