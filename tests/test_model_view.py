from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from simrig.cli import build_parser


ARM_XML = """\
<mujoco model="arm">
  <compiler angle="radian"/>
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

KEYFRAME_XML = """\
<mujoco model="keyframed_arm">
  <worldbody>
    <body name="link">
      <joint name="shoulder" type="hinge" range="-1 1"/>
      <geom type="capsule" size="0.05 0.2"/>
    </body>
  </worldbody>
  <keyframe>
    <key name="home" qpos="0.75"/>
  </keyframe>
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

    def test_view_model_defaults_to_threejs(self) -> None:
        parser = build_parser()

        args = parser.parse_args(["view-model", "unitree_go1"])

        self.assertEqual(args.render_mode, "threejs")

    def test_threejs_view_uses_webgl_scene_endpoint(self) -> None:
        from simrig.model_view import _html

        page = _html("threejs")

        self.assertIn("three@0.184.0", page)
        self.assertIn("OrbitControls", page)
        self.assertIn("/scene.json", page)
        self.assertIn('id="three-view"', page)

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

    def test_model_view_starts_from_first_authored_keyframe(self) -> None:
        from simrig.model_view import ModelViewSession

        with tempfile.TemporaryDirectory() as tmp:
            xml = Path(tmp) / "keyframed_arm.xml"
            xml.write_text(KEYFRAME_XML, encoding="utf-8")
            try:
                session = ModelViewSession(xml)
            except (ImportError, RuntimeError) as exc:
                self.skipTest(str(exc))
            try:
                payload = session.joints_payload()
                self.assertEqual(payload["render_mode"], "threejs")
                self.assertEqual(payload["initial_keyframe"], "home")
                self.assertAlmostEqual(payload["controls"][0]["value"], 0.75)
                session.set_joint_value(0, 0, -0.25)
                session.reset()
                self.assertAlmostEqual(session.joints_payload()["controls"][0]["value"], 0.75)
            finally:
                session.close()


if __name__ == "__main__":
    unittest.main()
