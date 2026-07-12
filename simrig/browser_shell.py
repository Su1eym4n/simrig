"""Shared browser viewer HTML/CSS/JS for SimRig localhost tools."""

from __future__ import annotations


def viewer_styles(*, sidebar_width: int = 360) -> str:
    return f"""
    :root {{ color-scheme: dark; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
    body {{ margin: 0; background: #111; color: #eee; display: grid; grid-template-columns: 1fr {sidebar_width}px; min-height: 100vh; }}
    main {{ display: grid; place-items: stretch; background: #050505; position: relative; }}
    #viewport {{ position: relative; width: 100%; height: 100vh; }}
    #frame {{ width: 100%; height: 100%; object-fit: contain; background: #000; display: block; cursor: grab; user-select: none; touch-action: none; }}
    #frame.dragging {{ cursor: grabbing; }}
    #hint {{ position: absolute; left: 16px; bottom: 16px; color: #aaa; font-size: 12px; background: rgba(0,0,0,0.55); padding: 8px 10px; border-radius: 6px; pointer-events: none; }}
    aside {{ padding: 16px; border-left: 1px solid #333; background: #181818; overflow: auto; max-height: 100vh; }}
    h1 {{ font-size: 18px; margin: 0 0 12px; }}
    label {{ display: block; margin: 10px 0 4px; color: #bbb; font-size: 13px; }}
    input {{ width: 100%; box-sizing: border-box; background: #0d0d0d; color: #eee; border: 1px solid #444; padding: 8px; }}
    button {{ margin: 8px 6px 0 0; padding: 8px 10px; background: #2d6cdf; color: white; border: 0; cursor: pointer; }}
    button.secondary {{ background: #333; }}
    pre {{ white-space: pre-wrap; background: #0d0d0d; border: 1px solid #333; padding: 10px; font-size: 12px; }}
    .meta {{ color: #aaa; font-size: 12px; margin-bottom: 12px; }}
    .control {{ margin: 0 0 14px; padding-bottom: 10px; border-bottom: 1px solid #2a2a2a; }}
    .control label {{ margin-top: 0; }}
    .control input[type=range] {{ width: 100%; }}
    .value {{ color: #8ab4ff; font-size: 12px; margin-top: 4px; }}
    """


def camera_interaction_script() -> str:
    return """
    let cameraAzimuth = 135;
    let cameraElevation = -20;
    let cameraDistance = 2.4;
    let cameraInteractive = true;
    let dragging = false;
    let lastX = 0;
    let lastY = 0;
    let cameraTimer = null;

    function clamp(value, min, max) {
      return Math.max(min, Math.min(max, value));
    }

    async function pushCamera() {
      if (!cameraInteractive) return;
      const params = new URLSearchParams({
        azimuth: String(cameraAzimuth),
        elevation: String(cameraElevation),
        distance: String(cameraDistance),
      });
      try {
        await fetch('/camera?' + params.toString(), {cache: 'no-store'});
      } catch (err) {
        console.warn('camera update failed', err);
      }
    }

    function scheduleCameraPush() {
      clearTimeout(cameraTimer);
      cameraTimer = setTimeout(pushCamera, 30);
    }

    function bindCameraControls(frame) {
      frame.addEventListener('mousedown', (event) => {
        if (!cameraInteractive || event.button !== 0) return;
        dragging = true;
        frame.classList.add('dragging');
        lastX = event.clientX;
        lastY = event.clientY;
      });
      window.addEventListener('mouseup', () => {
        if (!dragging) return;
        dragging = false;
        frame.classList.remove('dragging');
        pushCamera();
      });
      window.addEventListener('mousemove', (event) => {
        if (!dragging || !cameraInteractive) return;
        const dx = event.clientX - lastX;
        const dy = event.clientY - lastY;
        lastX = event.clientX;
        lastY = event.clientY;
        cameraAzimuth = (cameraAzimuth + dx * 0.35) % 360;
        cameraElevation = clamp(cameraElevation - dy * 0.25, -89, 89);
        scheduleCameraPush();
      });
      frame.addEventListener('wheel', (event) => {
        if (!cameraInteractive) return;
        event.preventDefault();
        const scale = event.deltaY > 0 ? 1.08 : 0.92;
        cameraDistance = clamp(cameraDistance * scale, 0.4, 12.0);
        scheduleCameraPush();
      }, {passive: false});
    }

    function applyCameraFromStatus(status) {
      if (!status || !status.camera) return;
      cameraAzimuth = status.camera.azimuth ?? cameraAzimuth;
      cameraElevation = status.camera.elevation ?? cameraElevation;
      cameraDistance = status.camera.distance ?? cameraDistance;
      cameraInteractive = status.camera.interactive !== false;
    }
    """


def frame_poll_script(*, poll_ms: int = 33) -> str:
    return f"""
    let frameTimer = null;
    let frameUrl = null;

    async function refreshFrame() {{
      clearTimeout(frameTimer);
      try {{
        const res = await fetch('/frame.jpg?t=' + Date.now(), {{cache: 'no-store'}});
        if (!res.ok) throw new Error('frame request failed');
        const blob = await res.blob();
        const url = URL.createObjectURL(blob);
        frame.onload = () => {{
          if (frameUrl) URL.revokeObjectURL(frameUrl);
          frameUrl = url;
        }};
        frame.src = url;
      }} catch (err) {{
        console.warn('frame refresh failed', err);
      }} finally {{
        frameTimer = setTimeout(refreshFrame, {poll_ms});
      }}
    }}
    """
