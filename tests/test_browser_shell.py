from __future__ import annotations

import unittest

from simrig.browser_shell import (
    agent_camera_panel,
    agent_camera_script,
    agent_camera_styles,
    viewer_chrome_script,
    viewer_styles,
)


class BrowserShellTests(unittest.TestCase):
    def test_sidebar_is_glass_overlay_with_persistent_collapse_control(self) -> None:
        styles = viewer_styles(sidebar_width=320)
        script = viewer_chrome_script()

        self.assertIn("--simrig-sidebar-width: 320px", styles)
        self.assertIn("backdrop-filter: blur(18px)", styles)
        self.assertIn("body.simrig-sidebar-collapsed aside", styles)
        self.assertIn("simrig-sidebar-toggle", script)
        self.assertIn("simrig:sidebar-collapsed", script)

    def test_embed_query_hides_standard_chrome(self) -> None:
        styles = viewer_styles()
        script = viewer_chrome_script()

        self.assertIn("params.get('embed') === '1'", script)
        self.assertIn("params.get('chrome') === '0'", script)
        self.assertIn("body.simrig-embed aside", styles)
        self.assertIn("body.simrig-embed #hint", styles)

    def test_robot_view_can_collapse_and_restore(self) -> None:
        panel = agent_camera_panel()
        styles = agent_camera_styles()
        script = agent_camera_script()

        self.assertIn('id="agent-camera-toggle"', panel)
        self.assertIn('data-collapsed="true"', styles)
        self.assertIn("setAgentCameraCollapsed", script)
        self.assertIn("simrig:agent-camera-collapsed", script)
        self.assertIn("body.simrig-embed #agent-camera-panel", styles)


if __name__ == "__main__":
    unittest.main()
