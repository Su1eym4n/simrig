from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from simrig.cli import build_parser


ARM_XML = """\
<mujoco model="arm">
  <worldbody>
    <body name="link1" pos="0 0 0">
      <joint name="shoulder" type="hinge" range="-1.5 1.5"/>
      <geom type="capsule" size="0.05 0.2"/>
      <body name="link2" pos="0 0 0.4">
        <joint name="elbow" type="hinge" range="0 2"/>
        <geom type="capsule" size="0.05 0.2"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


class ModelViewTests(unittest.TestCase):
    def test_view_model_command_parses_browser_options(self) -> None:
        parser = build_parser()

        args = parser.parse_args(
            [
                "view-model",
                "unitree_go1",
                "--menagerie",
                "/tmp/menagerie",
                "--port",
                "8770",
                "--render-mode",
                "topdown",
            ]
        )

        self.assertEqual(args.model_or_xml, "unitree_go1")
        self.assertEqual(str(args.menagerie), "/tmp/menagerie")
        self.assertEqual(args.port, 8770)
        self.assertEqual(args.render_mode, "topdown")

    def test_joint_controls_use_joint_names_when_mujoco_available(self) -> None:
        try:
            import numpy  # noqa: F401
            from simrig.model_view import joint_control_specs
            from simrig.mujoco_backend import _import_mujoco

            mujoco = _import_mujoco()
        except ImportError as exc:
            self.skipTest(str(exc))
        except RuntimeError as exc:
            self.skipTest(str(exc))

        with tempfile.TemporaryDirectory() as tmp:
            xml = Path(tmp) / "arm.xml"
            xml.write_text(ARM_XML, encoding="utf-8")

            model = mujoco.MjModel.from_xml_path(str(xml))
            data = mujoco.MjData(model)
            mujoco.mj_forward(model, data)

            controls = joint_control_specs(mujoco, model, data)
            labels = [control.label for control in controls]

            self.assertEqual(labels, ["shoulder", "elbow"])
            self.assertEqual(controls[0].joint_name, "shoulder")
            self.assertEqual(controls[1].joint_name, "elbow")
            self.assertTrue(controls[0].limited)
            self.assertEqual(controls[0].min, -1.5)
            self.assertEqual(controls[0].max, 1.5)


if __name__ == "__main__":
    unittest.main()
