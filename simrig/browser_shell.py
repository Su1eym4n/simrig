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


def agent_camera_styles() -> str:
    """Styles for the emulated and native named-camera inset."""
    return """
    #agent-camera-panel { position: absolute; top: 16px; right: 16px; width: min(34vw, 360px); z-index: 3; border: 1px solid rgba(148,163,184,0.5); border-radius: 10px; overflow: hidden; background: rgba(7,11,18,0.9); box-shadow: 0 12px 36px rgba(0,0,0,0.38); backdrop-filter: blur(8px); }
    #agent-camera-panel[hidden] { display: none; }
    #agent-camera-header { display: flex; align-items: center; gap: 8px; padding: 8px 10px; border-bottom: 1px solid rgba(148,163,184,0.25); }
    #agent-camera-header strong { font-size: 12px; letter-spacing: 0.02em; white-space: nowrap; flex: 1; }
    #agent-camera-mode { color: #bfdbfe; background: #172033; border: 1px solid #475569; border-radius: 999px; padding: 3px 8px; font-size: 11px; }
    #agent-camera-controls { display: flex; align-items: center; gap: 8px; padding: 7px 10px; background: rgba(15,23,42,0.72); }
    #agent-camera-controls span { color: #94a3b8; font-size: 11px; }
    #agent-camera-select { min-width: 0; flex: 1; color: #e5e7eb; background: #111827; border: 1px solid #475569; border-radius: 5px; padding: 4px 6px; }
    #agent-camera-frame, #agent-camera-three { display: block; width: 100%; aspect-ratio: 4 / 3; object-fit: contain; background: #0b1220; }
    #agent-camera-frame[hidden], #agent-camera-three[hidden] { display: none; }
    #agent-camera-status { padding: 6px 10px; color: #94a3b8; font-size: 11px; }
    @media (max-width: 760px) { #agent-camera-panel { width: min(48vw, 300px); } }
    """


def agent_camera_panel() -> str:
    """HTML for an optional emulated or native agent-camera inset."""
    return """
      <section id="agent-camera-panel" hidden aria-label="Robot camera view">
        <div id="agent-camera-header">
          <strong>Robot View</strong>
          <select id="agent-camera-mode" aria-label="Robot view source">
            <option value="emulated" selected>Emulated</option>
            <option value="sensor">Sensor</option>
          </select>
        </div>
        <div id="agent-camera-controls">
          <span>Camera</span>
          <select id="agent-camera-select" aria-label="Agent camera"></select>
        </div>
        <canvas id="agent-camera-three" aria-label="Three.js emulation of the robot camera"></canvas>
        <img id="agent-camera-frame" alt="Native MuJoCo sensor camera" hidden>
        <div id="agent-camera-status">Three.js emulation · loading…</div>
      </section>
    """


def agent_camera_script() -> str:
    """Mode switching, native polling, and camera selection for the inset."""
    return """
    const agentCameraPanel = document.getElementById('agent-camera-panel');
    const agentCameraSelect = document.getElementById('agent-camera-select');
    const agentCameraModeSelect = document.getElementById('agent-camera-mode');
    const agentCameraThree = document.getElementById('agent-camera-three');
    const agentCameraFrame = document.getElementById('agent-camera-frame');
    const agentCameraStatus = document.getElementById('agent-camera-status');
    let agentCameraTimer = null;
    let agentCameraObjectUrl = null;
    let agentCameraPollMs = 100;
    let agentCameraPayload = null;

    function agentCameraMode() {
      return agentCameraModeSelect.value;
    }

    function notifyAgentCameraChanged() {
      window.dispatchEvent(new CustomEvent('simrig-agent-camera-change', {
        detail: {
          name: agentCameraSelect.value,
          mode: agentCameraMode(),
        },
      }));
    }

    function renderAgentCameraStatus(payload) {
      agentCameraPayload = payload;
      if (agentCameraMode() === 'emulated') {
        agentCameraStatus.textContent = `Three.js emulation · ${payload.selected || 'none'} · authored pose + FOV`;
        return;
      }
      const error = payload.renderer_error ? ` · error: ${payload.renderer_error}` : '';
      agentCameraStatus.textContent = `MuJoCo sensor · ${payload.selected || 'none'} · ${payload.fps_target || 10} Hz${error}`;
    }

    function updateAgentCameraMode() {
      const sensor = agentCameraMode() === 'sensor';
      agentCameraThree.hidden = sensor;
      agentCameraFrame.hidden = !sensor;
      clearTimeout(agentCameraTimer);
      if (agentCameraPayload) renderAgentCameraStatus(agentCameraPayload);
      notifyAgentCameraChanged();
      if (sensor) refreshAgentCamera();
    }

    async function refreshAgentCamera() {
      clearTimeout(agentCameraTimer);
      if (agentCameraPanel.hidden || agentCameraMode() !== 'sensor') return;
      try {
        const res = await fetch('/agent-frame.jpg?t=' + Date.now(), {cache: 'no-store'});
        if (!res.ok) throw new Error(`camera frame failed (${res.status})`);
        const nextUrl = URL.createObjectURL(await res.blob());
        const oldUrl = agentCameraObjectUrl;
        agentCameraObjectUrl = nextUrl;
        agentCameraFrame.src = nextUrl;
        if (oldUrl) URL.revokeObjectURL(oldUrl);
      } catch (err) {
        agentCameraStatus.textContent = String(err);
      } finally {
        if (agentCameraMode() === 'sensor') {
          agentCameraTimer = setTimeout(refreshAgentCamera, agentCameraPollMs);
        }
      }
    }

    async function selectAgentCamera() {
      const params = new URLSearchParams({name: agentCameraSelect.value});
      const res = await fetch('/agent-camera?' + params.toString(), {cache: 'no-store'});
      if (!res.ok) throw new Error(`camera selection failed (${res.status})`);
      const payload = await res.json();
      renderAgentCameraStatus(payload);
      notifyAgentCameraChanged();
    }

    async function initializeAgentCamera() {
      const res = await fetch('/agent-cameras.json', {cache: 'no-store'});
      if (!res.ok) return;
      const payload = await res.json();
      if (!Array.isArray(payload.cameras) || payload.cameras.length === 0) return;
      agentCameraSelect.innerHTML = '';
      for (const name of payload.cameras) {
        const option = document.createElement('option');
        option.value = name;
        option.textContent = name;
        option.selected = name === payload.selected;
        agentCameraSelect.appendChild(option);
      }
      agentCameraPollMs = Math.max(50, Math.round(1000 / (payload.fps_target || 10)));
      renderAgentCameraStatus(payload);
      agentCameraPanel.hidden = false;
      agentCameraSelect.addEventListener('change', async () => {
        try { await selectAgentCamera(); } catch (err) { agentCameraStatus.textContent = String(err); }
      });
      agentCameraModeSelect.addEventListener('change', updateAgentCameraMode);
      updateAgentCameraMode();
    }
    """


def threejs_agent_camera_script() -> str:
    """Three.js renderer driven by authored MuJoCo camera transforms."""
    return """
    const emulatedCamera = new THREE.PerspectiveCamera(45, 4 / 3, 0.01, 1000);
    const agentRenderer = new THREE.WebGLRenderer({
      canvas: agentCameraThree,
      antialias: true,
      alpha: false,
    });
    agentRenderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    agentRenderer.shadowMap.enabled = true;
    agentRenderer.shadowMap.type = THREE.PCFShadowMap;
    agentRenderer.outputColorSpace = THREE.SRGBColorSpace;
    agentRenderer.toneMapping = THREE.ACESFilmicToneMapping;
    agentRenderer.toneMappingExposure = 1.0;
    agentRenderer.setClearColor(0x0b1220, 1);
    const authoredCameras = new Map();

    function updateAuthoredCameras(cameras) {
      for (const authored of cameras || []) {
        authoredCameras.set(authored.name, authored);
      }
      applySelectedAuthoredCamera();
    }

    function applySelectedAuthoredCamera() {
      const authored = authoredCameras.get(agentCameraSelect.value);
      if (!authored) return;
      applyTransform(emulatedCamera, authored);
      emulatedCamera.fov = authored.fovy || 45;
      emulatedCamera.updateProjectionMatrix();
    }

    function resizeAgentCamera() {
      if (agentCameraMode() !== 'emulated' || agentCameraThree.hidden) return false;
      const width = Math.max(1, agentCameraThree.clientWidth);
      const height = Math.max(1, agentCameraThree.clientHeight);
      const pixelRatio = agentRenderer.getPixelRatio();
      if (
        agentCameraThree.width !== Math.round(width * pixelRatio)
        || agentCameraThree.height !== Math.round(height * pixelRatio)
      ) {
        agentRenderer.setSize(width, height, false);
      }
      emulatedCamera.aspect = width / height;
      emulatedCamera.updateProjectionMatrix();
      return true;
    }

    function renderAgentCameraEmulation() {
      if (!authoredCameras.has(agentCameraSelect.value)) return;
      if (!resizeAgentCamera()) return;
      agentRenderer.render(scene, emulatedCamera);
    }

    window.addEventListener('simrig-agent-camera-change', () => {
      applySelectedAuthoredCamera();
      renderAgentCameraEmulation();
    });
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
