from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from simrig.mujoco_backend import inspect_model


TINY_XML = """\
<mujoco model="tiny">
  <worldbody>
    <body name="body" pos="0 0 0.1">
      <freejoint/>
      <geom type="sphere" size="0.05" mass="1"/>
    </body>
  </worldbody>
</mujoco>
"""


class MujocoIntegrationTests(unittest.TestCase):
    def test_inspect_tiny_xml_when_mujoco_available(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            xml = Path(tmp) / "tiny.xml"
            xml.write_text(TINY_XML, encoding="utf-8")

            try:
                report = inspect_model(xml, steps=2)
            except RuntimeError as exc:
                self.skipTest(str(exc))

            self.assertTrue(report.compiled)
            self.assertTrue(report.stepped)
            self.assertTrue(report.has_freejoint)
            self.assertGreaterEqual(report.bodies, 2)


if __name__ == "__main__":
    unittest.main()

