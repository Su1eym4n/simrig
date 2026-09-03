"""Shared browser viewer HTML/CSS/JS for SimRig localhost tools."""

from __future__ import annotations


def viewer_styles(*, sidebar_width: int = 360) -> str:
    return f"""
    :root {{ color-scheme: dark; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; --simrig-sidebar-width: {sidebar_width}px; }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; background: #050505; color: #eee; min-height: 100vh; overflow: hidden; }}
    main {{ display: grid; place-items: stretch; background: #050505; position: relative; min-height: 100vh; }}
    #viewport {{ position: relative; width: 100%; height: 100vh; }}
    #frame {{ width: 100%; height: 100%; object-fit: contain; background: #000; display: block; cursor: grab; user-select: none; touch-action: none; }}
    #frame.dragging {{ cursor: grabbing; }}
    #hint {{ position: absolute; left: 16px; bottom: 16px; color: #cbd5e1; font-size: 12px; background: rgba(7,11,18,0.62); border: 1px solid rgba(148,163,184,0.18); padding: 8px 10px; border-radius: 8px; pointer-events: none; backdrop-filter: blur(10px); }}
    aside {{ position: fixed; z-index: 10; top: 12px; right: 12px; bottom: 12px; width: min(var(--simrig-sidebar-width), calc(100vw - 24px)); padding: 18px; border: 1px solid rgba(148,163,184,0.28); border-radius: 14px; background: rgba(15,18,24,0.78); box-shadow: 0 18px 50px rgba(0,0,0,0.34); backdrop-filter: blur(18px) saturate(135%); overflow: auto; transition: transform 180ms ease, opacity 180ms ease; }}
    body.simrig-sidebar-collapsed aside {{ opacity: 0; pointer-events: none; transform: translateX(calc(100% + 24px)); }}
    #simrig-sidebar-toggle {{ position: fixed; z-index: 11; top: 22px; right: calc(min(var(--simrig-sidebar-width), calc(100vw - 24px)) + 20px); width: 34px; height: 34px; margin: 0; padding: 0; display: grid; place-items: center; border: 1px solid rgba(148,163,184,0.32); border-radius: 10px; color: #e2e8f0; background: rgba(15,18,24,0.76); box-shadow: 0 8px 24px rgba(0,0,0,0.28); backdrop-filter: blur(14px); transition: right 180ms ease, background 120ms ease; }}
    #simrig-sidebar-toggle:hover {{ background: rgba(37,45,59,0.92); }}
    #simrig-sidebar-toggle span {{ display: block; font-size: 22px; line-height: 1; transform: translateY(-1px); }}
    body.simrig-sidebar-collapsed #simrig-sidebar-toggle {{ right: 14px; }}
    body.simrig-embed aside, body.simrig-embed #simrig-sidebar-toggle, body.simrig-embed #hint {{ display: none; }}
    h1 {{ font-size: 18px; margin: 0 0 12px; }}
    label {{ display: block; margin: 10px 0 4px; color: #bbb; font-size: 13px; }}
    input {{ width: 100%; background: rgba(8,11,16,0.78); color: #eee; border: 1px solid rgba(148,163,184,0.30); border-radius: 7px; padding: 8px; }}
    button {{ margin: 8px 6px 0 0; padding: 8px 11px; background: #2d6cdf; color: white; border: 1px solid transparent; border-radius: 8px; cursor: pointer; font: inherit; }}
    button:hover {{ filter: brightness(1.08); }}
    button:focus-visible, input:focus-visible, select:focus-visible, summary:focus-visible {{ outline: 2px solid #60a5fa; outline-offset: 2px; }}
    button.secondary {{ background: rgba(51,65,85,0.78); border-color: rgba(148,163,184,0.18); }}
    pre {{ white-space: pre-wrap; overflow-wrap: anywhere; background: rgba(5,8,13,0.72); border: 1px solid rgba(148,163,184,0.20); border-radius: 8px; padding: 10px; font-size: 12px; }}
    .meta {{ color: #aaa; font-size: 12px; margin-bottom: 12px; }}
    .control {{ margin: 0 0 14px; padding-bottom: 10px; border-bottom: 1px solid #2a2a2a; }}
    .control label {{ margin-top: 0; }}
    .control input[type=range] {{ width: 100%; }}
    .value {{ color: #8ab4ff; font-size: 12px; margin-top: 4px; }}
    .simrig-debug {{ margin-top: 18px; border-top: 1px solid rgba(148,163,184,0.18); padding-top: 12px; }}
    .simrig-debug summary {{ color: #cbd5e1; cursor: pointer; font-size: 13px; font-weight: 600; user-select: none; }}
    .simrig-debug pre {{ margin-bottom: 0; max-height: 44vh; overflow: auto; }}
    @media (max-width: 760px) {{
      aside {{ top: 8px; right: 8px; bottom: 8px; width: calc(100vw - 16px); }}
      #simrig-sidebar-toggle {{ top: 16px; right: 16px; }}
      body:not(.simrig-sidebar-collapsed) #simrig-sidebar-toggle {{ right: 16px; }}
      #hint {{ left: 10px; bottom: 10px; }}
    }}
    @media (prefers-reduced-motion: reduce) {{ aside, #simrig-sidebar-toggle {{ transition: none; }} }}
    """


def viewer_chrome_script() -> str:
    """Shared sidebar behavior and scene-only embedding for browser viewers."""
    return """
  <script>
    (() => {
      const params = new URLSearchParams(window.location.search);
      const embed = params.get('embed') === '1' || params.get('chrome') === '0';
      if (embed) document.body.classList.add('simrig-embed');

      const sidebar = document.querySelector('aside');
      if (!sidebar || embed) return;
      sidebar.id ||= 'simrig-sidebar';

      const button = document.createElement('button');
      button.id = 'simrig-sidebar-toggle';
      button.type = 'button';
      button.setAttribute('aria-controls', sidebar.id);
      button.innerHTML = '<span aria-hidden="true">›</span>';
      document.body.appendChild(button);

      const storageKey = 'simrig:sidebar-collapsed';
      let collapsed = false;
      try { collapsed = window.localStorage.getItem(storageKey) === '1'; } catch (_) {}

      function apply(next) {
        collapsed = Boolean(next);
        document.body.classList.toggle('simrig-sidebar-collapsed', collapsed);
        button.setAttribute('aria-expanded', String(!collapsed));
        button.setAttribute('aria-label', collapsed ? 'Show sidebar' : 'Hide sidebar');
        button.title = collapsed ? 'Show sidebar' : 'Hide sidebar';
        button.firstElementChild.textContent = collapsed ? '‹' : '›';
        try { window.localStorage.setItem(storageKey, collapsed ? '1' : '0'); } catch (_) {}
        window.dispatchEvent(new CustomEvent('simrig-sidebar-change', {detail: {collapsed}}));
      }

      button.addEventListener('click', () => apply(!collapsed));
      apply(collapsed);
    })();
  </script>
    """


def agent_camera_styles() -> str:
    """Styles for the emulated and native named-camera inset."""
    return """
    #agent-camera-panel { position: absolute; top: 16px; right: calc(min(var(--simrig-sidebar-width), calc(100vw - 24px)) + 72px); width: min(34vw, 360px); z-index: 3; border: 1px solid rgba(148,163,184,0.38); border-radius: 12px; overflow: hidden; background: rgba(7,11,18,0.76); box-shadow: 0 12px 36px rgba(0,0,0,0.34); backdrop-filter: blur(14px) saturate(130%); transition: width 160ms ease, top 180ms ease, right 180ms ease, background 160ms ease; }
    #agent-camera-panel[hidden] { display: none; }
    body.simrig-sidebar-collapsed #agent-camera-panel { top: 68px; right: 16px; }
    body.simrig-embed #agent-camera-panel { display: none; }
    #agent-camera-header { display: flex; align-items: center; gap: 8px; padding: 8px 10px; border-bottom: 1px solid rgba(148,163,184,0.25); }
    #agent-camera-header strong { font-size: 12px; letter-spacing: 0.02em; white-space: nowrap; flex: 1; }
    #agent-camera-toggle { flex: 0 0 auto; display: grid; place-items: center; width: 24px; height: 24px; margin: 0; padding: 0; border: 1px solid rgba(148,163,184,0.26); border-radius: 6px; color: #cbd5e1; background: rgba(30,41,59,0.68); font-size: 16px; line-height: 1; }
    #agent-camera-mode { color: #bfdbfe; background: #172033; border: 1px solid #475569; border-radius: 999px; padding: 3px 8px; font-size: 11px; }
    #agent-camera-controls { display: flex; align-items: center; gap: 8px; padding: 7px 10px; background: rgba(15,23,42,0.72); }
    #agent-camera-controls span { color: #94a3b8; font-size: 11px; }
    #agent-camera-select { min-width: 0; flex: 1; color: #e5e7eb; background: #111827; border: 1px solid #475569; border-radius: 5px; padding: 4px 6px; }
    #agent-camera-frame, #agent-camera-three { display: block; width: 100%; aspect-ratio: 4 / 3; object-fit: contain; background: #0b1220; }
    #agent-camera-frame[hidden], #agent-camera-three[hidden] { display: none; }
    #agent-camera-status { padding: 6px 10px; color: #94a3b8; font-size: 11px; }
    #agent-camera-panel[data-collapsed="true"] { width: 44px; background: rgba(7,11,18,0.66); }
    #agent-camera-panel[data-collapsed="true"] #agent-camera-header { padding: 9px; border-bottom: 0; }
    #agent-camera-panel[data-collapsed="true"] #agent-camera-header strong,
    #agent-camera-panel[data-collapsed="true"] #agent-camera-mode,
    #agent-camera-panel[data-collapsed="true"] #agent-camera-controls,
    #agent-camera-panel[data-collapsed="true"] #agent-camera-three,
    #agent-camera-panel[data-collapsed="true"] #agent-camera-frame,
    #agent-camera-panel[data-collapsed="true"] #agent-camera-status { display: none; }
    @media (max-width: 760px) { #agent-camera-panel { right: 16px; width: min(70vw, 300px); } }
    @media (prefers-reduced-motion: reduce) { #agent-camera-panel { transition: none; } }
    """


def agent_camera_panel() -> str:
    """HTML for an optional emulated or native agent-camera inset."""
    return """
      <section id="agent-camera-panel" hidden aria-label="Robot camera view">
        <div id="agent-camera-header">
          <button id="agent-camera-toggle" type="button" aria-expanded="true" aria-label="Hide Robot View" title="Hide Robot View">›</button>
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
    const agentCameraToggle = document.getElementById('agent-camera-toggle');
    let agentCameraTimer = null;
    let agentCameraObjectUrl = null;
    let agentCameraPollMs = 100;
    let agentCameraPayload = null;

    function setAgentCameraCollapsed(collapsed) {
      const next = Boolean(collapsed);
      agentCameraPanel.dataset.collapsed = String(next);
      agentCameraToggle.setAttribute('aria-expanded', String(!next));
      agentCameraToggle.setAttribute('aria-label', next ? 'Show Robot View' : 'Hide Robot View');
      agentCameraToggle.title = next ? 'Show Robot View' : 'Hide Robot View';
      agentCameraToggle.textContent = next ? '◉' : '›';
      try { window.localStorage.setItem('simrig:agent-camera-collapsed', next ? '1' : '0'); } catch (_) {}
      clearTimeout(agentCameraTimer);
      notifyAgentCameraChanged();
      if (!next && agentCameraMode() === 'sensor') refreshAgentCamera();
    }

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
      if (agentCameraPanel.hidden || agentCameraPanel.dataset.collapsed === 'true' || agentCameraMode() !== 'sensor') return;
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
      let cameraCollapsed = false;
      try { cameraCollapsed = window.localStorage.getItem('simrig:agent-camera-collapsed') === '1'; } catch (_) {}
      setAgentCameraCollapsed(cameraCollapsed);
      agentCameraToggle.addEventListener('click', () => {
        setAgentCameraCollapsed(agentCameraPanel.dataset.collapsed !== 'true');
      });
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
