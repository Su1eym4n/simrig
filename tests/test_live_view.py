from __future__ import annotations

import unittest


SIMPLE_XML = """\
<mujoco model="live_test">
  <option timestep="0.01"/>
  <worldbody>
    <body name="tracked" pos="0 0 0.2">
      <joint name="slide" type="slide" axis="1 0 0"/>
      <geom type="sphere" size="0.05"/>
    </body>
  </worldbody>
</mujoco>
"""


class LiveWebViewerTests(unittest.TestCase):
    def test_live_page_uses_threejs_scene_and_state_endpoints(self) -> None:
        from simrig.live_view import _live_html

        page = _live_html()

        self.assertIn("three@0.184.0", page)
        self.assertIn("/scene.json", page)
        self.assertIn("/state.json", page)
        self.assertIn("Clear Trail", page)
        self.assertIn("Show Trail", page)
        self.assertIn("hide-trail", page)

    def test_live_viewer_serves_script_owned_data(self) -> None:
        try:
            import mujoco
            from simrig.live_view import LiveWebViewer
        except (ImportError, RuntimeError) as exc:
            self.skipTest(str(exc))

        model = mujoco.MjModel.from_xml_string(SIMPLE_XML)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)

        viewer = LiveWebViewer(
            model,
            data,
            name="unit script",
            tracking_body="tracked",
            mujoco_module=mujoco,
        )
        try:
            with viewer.lock:
                data.qpos[0] = 0.25
                mujoco.mj_forward(model, data)
                viewer.sync(phase="moving")

            payload = viewer.state_payload()

            self.assertEqual(payload["name"], "unit script")
            self.assertEqual(payload["phase"], "moving")
            self.assertEqual(payload["frame"], 1)
            self.assertEqual(payload["tracking_body"], "tracked")
            self.assertAlmostEqual(payload["tracking_position"][0], 0.25)
            viewer._note_client()
            self.assertTrue(viewer.wait_for_client(timeout=0.1))
        finally:
            viewer.close()


if __name__ == "__main__":
    unittest.main()
