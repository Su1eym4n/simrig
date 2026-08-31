"""Browser preview server for policy rollouts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import gzip
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
import time
from typing import Any, Callable
from urllib.parse import parse_qs, urlparse

import numpy as np

from simrig.browser_render import MujocoFramePump, named_camera_names
from simrig.browser_shell import (
    agent_camera_panel,
    agent_camera_script,
    agent_camera_styles,
    camera_interaction_script,
    frame_poll_script,
    threejs_agent_camera_script,
    viewer_styles,
)
from simrig.rollout import PolicyRuntime, validate_state
from simrig.success import evaluate_task_success, load_success_spec, metric_value
from simrig.rendering import (
    CameraState,
    configure_headless_mujoco_gl,
    make_tracking_camera,
    tracking_body_id,
)
from simrig.three_scene import camera_transforms, geom_transforms, scene_payload


@dataclass
class PreviewStatus:
    env_name: str
    checkpoint: str
    step: int
    reward: float
    total_reward: float
    done: bool
    episode: int
    episode_state: str
    episode_outcome: str
    task_success: bool | None
    task_success_reason: str
    survival_steps: int
    last_episode_survival_steps: int | None
    last_episode_reward: float | None
    auto_reset: bool
    auto_reset_delay: float
    command_supported: bool
    command_controls: list[dict[str, Any]]
    command_values: list[float] | None
    command: list[float] | None
    command_applied: bool
    paused: bool
    render_mode: str
    frame_busy: bool
    renderer_error: str | None


def _preview_outcome(
    values: list[float | None], spec: Mapping[str, Any], *, done: bool,
) -> tuple[bool | None, str]:
    """Report environment success only after termination, without using reward.

    Missing samples cannot establish failure or bridge gaps in a hold criterion.
    This is the environment's own outcome, not independent acceptance evidence.
    """
    if not done:
        return None, "Episode has not ended."
    if not values or any(value is None or not np.isfinite(value) for value in values):
        return None, "Outcome unknown: no complete finite success metric is available."
    passed, reason = evaluate_task_success(values, spec or None)
    return passed, f"Environment metrics: {reason} This is not independent acceptance."


class PolicyPreviewSession:
    """Owns env, policy, renderer, and mutable rollout state."""

    def __init__(
        self,
        checkpoint: Path | str,
        *,
        env_name: str,
        backend: str = "mujoco-playground",
        small_network: bool | None = None,
        seed: int = 0,
        command: tuple[float, ...] | None = None,
        width: int = 960,
        height: int = 540,
        frame_skip: int = 1,
        camera: str | int | None = None,
        render_mode: str = "threejs",
        paused: bool = False,
        fps: int = 24,
        auto_reset: bool = False,
        auto_reset_delay: float = 1.5,
        allow_runtime_mismatch: bool = False,
        reset_transform: Callable[[Any, Any], Any] | None = None,
    ) -> None:
        configure_headless_mujoco_gl()
        self.rollout = PolicyRuntime(
            checkpoint, env_name=env_name, backend=backend, small_network=small_network,
            allow_runtime_mismatch=allow_runtime_mismatch,
        )
        self.runtime_compatibility = self.rollout.runtime_compatibility
        self.reset_transform = reset_transform
        self.env_name = env_name
        self.checkpoint = str(checkpoint)
        self.backend = backend
        self.frame_skip = max(1, int(frame_skip))
        self.command = list(command) if command is not None else None
        self.command_applied = False
        self.paused = paused
        self.step_count = 0
        self.total_reward = 0.0
        self.last_reward = 0.0
        self.done = False
        self.episode = 1
        self.last_episode_survival_steps: int | None = None
        self.last_episode_reward: float | None = None
        self.auto_reset = bool(auto_reset)
        self.auto_reset_delay = max(0.0, float(auto_reset_delay))
        self._episode_ended_at: float | None = None
        self.render_mode = render_mode.lower()
        self.fps = max(1, int(fps))
        self.renderer_error: str | None = None
        self._lock = threading.Lock()
        self._last_frame_jpeg: bytes | None = None
        self._scene_payload: dict[str, Any] | None = None

        self.jax, self.jp = self.rollout.jax, self.rollout.jp
        try:
            import mujoco  # type: ignore
            from PIL import Image  # type: ignore
            from PIL import ImageDraw  # type: ignore
        except ImportError as exc:
            raise RuntimeError("Browser preview requires MuJoCo and Pillow.") from exc

        self.mujoco = mujoco
        self.Image = Image
        self.ImageDraw = ImageDraw
        self.env = self.rollout.env
        self.success_spec = load_success_spec(env_name, self.env)
        self.success_values: list[float | None] = []
        self.state, self.rng = self.rollout.reset(seed)
        if self.reset_transform is not None:
            self.state = self.reset_transform(self.env, self.state)
            validate_state(self.state, self.env.observation_size)
        self.command_controls = _command_controls(self.env, self.state)
        if self.command is not None and len(self.command) != len(self.command_controls):
            raise ValueError(
                f"Environment accepts {len(self.command_controls)} command values; "
                f"received {len(self.command)}."
            )
        self._apply_current_command()

        self.mj_data = self.mujoco.MjData(self.env.mj_model)
        self._copy_state_to_mujoco_unlocked()
        self.width = width
        self.height = height
        self.renderer = None
        if self.render_mode not in ("threejs", "topdown", "mujoco"):
            raise ValueError("render_mode must be 'threejs', 'mujoco', or 'topdown'.")
        self.camera, self.camera_state = make_tracking_camera(
            self.mujoco,
            self.env.mj_model,
            self.mj_data,
            camera,
        )
        self._frame_pump: MujocoFramePump | None = None
        self.agent_camera_names = named_camera_names(self.mujoco, self.env.mj_model)
        self.agent_camera_name = self._initial_agent_camera(camera)
        self._agent_frame_pump: MujocoFramePump | None = None
        self._agent_pump_lock = threading.Lock()
        self._agent_width = min(360, max(160, width))
        self._agent_height = max(120, round(self._agent_width * 3 / 4))
        self._agent_fps = min(12, self.fps)
        self.physics_rate_hz = _rate_hz(getattr(self.env.mj_model.opt, "timestep", None))
        self.policy_observation_rate_hz = _rate_hz(getattr(self.env, "dt", None))
        self.episode_horizon = _episode_horizon(self.env) or _checkpoint_episode_horizon(checkpoint)
        self._rollout_thread: threading.Thread | None = None
        self._running = True
        if self.render_mode == "threejs":
            self._rollout_thread = threading.Thread(
                target=self._run_rollout,
                name="simrig-preview-rollout",
                daemon=True,
            )
            self._rollout_thread.start()
        else:
            self._frame_pump = MujocoFramePump(
                self.mujoco,
                self.env.mj_model,
                self.mj_data,
                width=width,
                height=height,
                camera=self.camera,
                camera_state=self.camera_state,
                image_module=self.Image,
                fps=self.fps,
                render_mode=self.render_mode,
                scene_lock=self._lock,
                before_render=self._advance_rollout,
                fallback_frame=self._fallback_frame,
                error_frame=self._error_frame,
            )

    def reset(self) -> None:
        with self._lock:
            self._reset_unlocked()

    def set_paused(self, paused: bool) -> None:
        with self._lock:
            self.paused = paused

    def set_command(self, command: list[float] | None) -> None:
        with self._lock:
            if command is not None:
                if not self.command_controls:
                    raise ValueError(f"Environment {self.env_name} does not support commands.")
                if len(command) != len(self.command_controls):
                    raise ValueError(
                        f"Environment accepts {len(self.command_controls)} command values; "
                        f"received {len(command)}."
                    )
            self.command = command
            self._apply_current_command()

    def set_auto_reset(self, enabled: bool) -> None:
        with self._lock:
            self.auto_reset = bool(enabled)

    def status(self) -> PreviewStatus:
        with self._lock:
            return self._status_unlocked()

    def status_payload(self) -> dict[str, Any]:
        with self._lock:
            payload = self._status_unlocked().__dict__
            payload.update(self._renderer_stats())
            payload.update(self._rate_payload())
            return payload

    def state_payload(self) -> dict[str, Any]:
        """Return current rollout metadata and world transforms for Three.js."""

        with self._lock:
            payload = self._status_unlocked().__dict__
            payload.update(self._renderer_stats())
            payload.update(self._rate_payload())
            payload["transforms"] = geom_transforms(self.env.mj_model, self.mj_data)
            payload["authored_cameras"] = camera_transforms(
                self.mujoco,
                self.env.mj_model,
                self.mj_data,
            )
            payload["tracking_position"] = np.asarray(
                self.mj_data.xpos[self._tracking_body_id()],
                dtype=float,
            ).tolist()
            return payload

    def scene_payload(self) -> dict[str, Any]:
        with self._lock:
            if self._scene_payload is None:
                self._scene_payload = scene_payload(
                    self.mujoco,
                    self.env.mj_model,
                    self.mj_data,
                    model_name=self.env_name,
                )
                self._scene_payload.pop("transforms", None)
                self._scene_payload.pop("authored_cameras", None)
            return {
                **self._scene_payload,
                "transforms": geom_transforms(self.env.mj_model, self.mj_data),
                "authored_cameras": camera_transforms(
                    self.mujoco,
                    self.env.mj_model,
                    self.mj_data,
                ),
                "tracking_position": np.asarray(
                    self.mj_data.xpos[self._tracking_body_id()],
                    dtype=float,
                ).tolist(),
                "fps_target": self.fps,
                **self._rate_payload(),
            }

    def _status_unlocked(self) -> PreviewStatus:
        success, reason = _preview_outcome(self.success_values, self.success_spec, done=self.done)
        return PreviewStatus(
            env_name=self.env_name,
            checkpoint=self.checkpoint,
            step=self.step_count,
            reward=self.last_reward,
            total_reward=self.total_reward,
            done=self.done,
            episode=self.episode,
            episode_state="ended" if self.done else "paused" if self.paused else "running",
            episode_outcome=(
                "pending" if not self.done else "success" if success is True
                else "failure" if success is False else "unknown"
            ),
            task_success=success,
            task_success_reason=reason,
            survival_steps=self.step_count,
            last_episode_survival_steps=self.last_episode_survival_steps,
            last_episode_reward=self.last_episode_reward,
            auto_reset=self.auto_reset,
            auto_reset_delay=self.auto_reset_delay,
            command_supported=bool(self.command_controls),
            command_controls=list(self.command_controls),
            command_values=self._command_values_unlocked(),
            command=list(self.command) if self.command is not None else None,
            command_applied=self.command_applied,
            paused=self.paused,
            render_mode=self.render_mode,
            frame_busy=False,
            renderer_error=self._renderer_stats()["renderer_error"],
        )

    def _rate_payload(self) -> dict[str, Any]:
        return {
            "playback_rate_hz": self.fps,
            "physics_rate_hz": self.physics_rate_hz,
            "policy_observation_rate_hz": self.policy_observation_rate_hz,
            "sensor_display_rate_hz": self._agent_fps if self.agent_camera_names else None,
            "episode_horizon": self.episode_horizon,
        }

    def _command_values_unlocked(self) -> list[float] | None:
        if not self.command_controls:
            return None
        if self.command is not None:
            return list(self.command)
        info = getattr(self.state, "info", {})
        values = info.get("command") if isinstance(info, Mapping) else None
        if values is None:
            return None
        return np.asarray(values, dtype=float).reshape(-1).tolist()

    def _renderer_stats(self) -> dict[str, Any]:
        if self._frame_pump is not None:
            return self._frame_pump.stats()
        return {
            "fps_target": self.fps,
            "renderer_error": self.renderer_error,
            "camera": {
                "interactive": True,
                "renderer": "threejs-orbit-controls",
            },
        }

    def set_camera_from_query(self, query: dict[str, list[str]]) -> None:
        if self._frame_pump is not None:
            self._frame_pump.set_camera_from_query(query)

    def frame_jpeg(self) -> bytes:
        if self._frame_pump is None:
            raise RuntimeError("Frame streaming is disabled in threejs mode.")
        return self._frame_pump.get_jpeg()

    def agent_cameras_payload(self) -> dict[str, Any]:
        stats = (
            self._agent_frame_pump.stats()
            if self._agent_frame_pump is not None
            else {"fps_target": self._agent_fps, "renderer_error": None}
        )
        return {
            "cameras": list(self.agent_camera_names),
            "selected": self.agent_camera_name,
            "fps_target": stats["fps_target"],
            "renderer_error": stats["renderer_error"],
            "source": "mujoco-offscreen",
        }

    def select_agent_camera(self, name: str) -> dict[str, Any]:
        if name not in self.agent_camera_names:
            choices = ", ".join(self.agent_camera_names) or "none"
            raise ValueError(f"Unknown agent camera: {name}. Available: {choices}")
        self.agent_camera_name = name
        if self._agent_frame_pump is not None:
            self._agent_frame_pump.select_fixed_camera(self._agent_camera_ref(name))
        return self.agent_cameras_payload()

    def agent_frame_jpeg(self) -> bytes:
        pump = self._ensure_agent_frame_pump()
        if pump is None:
            raise RuntimeError("No authored MuJoCo cameras are available.")
        return pump.get_jpeg()

    def _ensure_agent_frame_pump(self) -> MujocoFramePump | None:
        if self.agent_camera_name is None:
            return None
        with self._agent_pump_lock:
            if self._agent_frame_pump is None:
                self._agent_frame_pump = MujocoFramePump(
                    self.mujoco,
                    self.env.mj_model,
                    self.mj_data,
                    width=self._agent_width,
                    height=self._agent_height,
                    camera=self._agent_camera_ref(self.agent_camera_name),
                    camera_state=CameraState(interactive=False),
                    image_module=self.Image,
                    fps=self._agent_fps,
                    render_mode="mujoco",
                    scene_lock=self._lock,
                )
            return self._agent_frame_pump

    def close(self) -> None:
        self._running = False
        if self._frame_pump is not None:
            self._frame_pump.close()
        if self._agent_frame_pump is not None:
            self._agent_frame_pump.close()
        if self._rollout_thread is not None:
            self._rollout_thread.join(timeout=2.0)
        self.rollout.close()

    def _initial_agent_camera(self, requested: str | int | None) -> str | None:
        if not self.agent_camera_names:
            return None
        if isinstance(requested, int) or (isinstance(requested, str) and requested.isdigit()):
            index = int(requested)
            if 0 <= index < len(self.agent_camera_names):
                return self.agent_camera_names[index]
        if isinstance(requested, str) and requested in self.agent_camera_names:
            return requested
        return self.agent_camera_names[0]

    def _agent_camera_ref(self, name: str) -> str | int:
        camera_id = self.mujoco.mj_name2id(
            self.env.mj_model,
            self.mujoco.mjtObj.mjOBJ_CAMERA,
            name,
        )
        return name if camera_id >= 0 else self.agent_camera_names.index(name)

    def _run_rollout(self) -> None:
        interval = 1.0 / self.fps
        while self._running:
            started = time.monotonic()
            try:
                with self._lock:
                    self._advance_rollout()
            except Exception as exc:
                self.renderer_error = str(exc)
            elapsed = time.monotonic() - started
            time.sleep(max(0.0, interval - elapsed))

    def _advance_rollout(self) -> None:
        if (
            self.done
            and self.auto_reset
            and not self.paused
            and self._episode_ended_at is not None
            and time.monotonic() - self._episode_ended_at >= self.auto_reset_delay
        ):
            self._reset_unlocked()
        if not self.paused and not self.done:
            for _ in range(self.frame_skip):
                self._step_once_unlocked()
                if self.done:
                    break
        self._copy_state_to_mujoco_unlocked()

    def _step_once_unlocked(self) -> None:
        self.state, self.rng, _ = self.rollout.advance(
            self.state, self.rng, command=self.command,
        )
        self.command_applied = self.command is not None
        self.last_reward = float(self.state.reward)
        self.total_reward += self.last_reward
        self.step_count += 1
        self.success_values.append(metric_value(
            getattr(self.state, "metrics", None), str(self.success_spec.get("metric") or "success"),
        ))
        self.done = bool(self.state.done)
        if self.done:
            self.last_episode_survival_steps = self.step_count
            self.last_episode_reward = self.total_reward
            self._episode_ended_at = time.monotonic()

    def _reset_unlocked(self) -> None:
        self.rng, reset_rng = self.jax.random.split(self.rng)
        self.state = self.rollout.reset_key(reset_rng)
        if self.reset_transform is not None:
            self.state = self.reset_transform(self.env, self.state)
            validate_state(self.state, self.env.observation_size)
        self.episode += 1
        self.step_count = 0
        self.total_reward = 0.0
        self.last_reward = 0.0
        self.done = False
        self.success_values.clear()
        self._episode_ended_at = None
        self.command_controls = _command_controls(self.env, self.state)
        self._apply_current_command()

    def _apply_current_command(self) -> None:
        if self.command is None:
            self.command_applied = False
            return
        self.state = self.rollout.apply_command(self.state, self.command)
        self.command_applied = True

    def _copy_state_to_mujoco_unlocked(self) -> None:
        self.mj_data.qpos[:] = np.asarray(self.state.data.qpos)
        self.mj_data.qvel[:] = np.asarray(self.state.data.qvel)
        for name in ("mocap_pos", "mocap_quat"):
            source = getattr(self.state.data, name, None)
            target = getattr(self.mj_data, name, None)
            if source is not None and target is not None:
                target[:] = np.asarray(source)
        self.mujoco.mj_forward(self.env.mj_model, self.mj_data)

    def _tracking_body_id(self) -> int:
        from simrig.rendering import tracking_body_id

        return tracking_body_id(self.mujoco, self.env.mj_model, self.mj_data)

    def _fallback_frame(self) -> np.ndarray:
        """Explicit schematic mode only; not used for default MuJoCo rendering."""
        image = self.Image.new("RGB", (self.width, self.height), (8, 10, 12))
        draw = self.ImageDraw.Draw(image)
        model = self.env.mj_model
        center = np.asarray(self.mj_data.xpos[self._tracking_body_id()])[:2]
        scale = min(self.width, self.height) / 5.0

        def project(pos: np.ndarray) -> tuple[int, int]:
            xy = (np.asarray(pos)[:2] - center) * scale
            return int(self.width / 2 + xy[0]), int(self.height / 2 - xy[1])

        # Ground grid.
        grid_color = (28, 34, 38)
        for offset in np.linspace(-2.0, 2.0, 9):
            x1, y1 = project(center + np.array([offset, -2.0]))
            x2, y2 = project(center + np.array([offset, 2.0]))
            draw.line((x1, y1, x2, y2), fill=grid_color)
            x1, y1 = project(center + np.array([-2.0, offset]))
            x2, y2 = project(center + np.array([2.0, offset]))
            draw.line((x1, y1, x2, y2), fill=grid_color)

        # Parent-child body graph.
        for body_id in range(1, model.nbody):
            parent = int(model.body_parentid[body_id])
            if parent <= 0:
                continue
            x1, y1 = project(self.mj_data.xpos[parent])
            x2, y2 = project(self.mj_data.xpos[body_id])
            draw.line((x1, y1, x2, y2), fill=(96, 150, 255), width=3)

        for body_id in range(1, model.nbody):
            x, y = project(self.mj_data.xpos[body_id])
            radius = 7 if body_id == self._tracking_body_id() else 4
            fill = (255, 210, 94) if body_id == self._tracking_body_id() else (205, 224, 255)
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=fill)

        command = self.command if self.command is not None else ["-", "-", "-"]
        overlay = [
            "SimRig Preview",
            f"mode: {self.render_mode}",
            f"env: {self.env_name}",
            f"step: {self.step_count}",
            f"reward: {self.last_reward:.4f}",
            f"command: {command}",
        ]
        if self.renderer_error:
            overlay.append(f"render error: {self.renderer_error[:80]}")
        draw.rectangle((16, 16, min(self.width - 16, 720), 168), fill=(0, 0, 0))
        y = 28
        for line in overlay:
            draw.text((28, y), line, fill=(236, 240, 245))
            y += 22
        return np.asarray(image)

    def _error_frame(self, message: str) -> np.ndarray:
        image = self.Image.new("RGB", (self.width, self.height), (8, 10, 12))
        draw = self.ImageDraw.Draw(image)
        lines = [
            "SimRig Preview",
            "MuJoCo rendering failed",
            message[:240],
            "Try: export MUJOCO_GL=glfw",
            "Or use: simrig demo ... for native 3D viewer",
        ]
        y = 28
        for line in lines:
            draw.text((28, y), line, fill=(236, 240, 245))
            y += 24
        return np.asarray(image)

    def _plain_frame(self, message: str) -> np.ndarray:
        image = self.Image.new("RGB", (self.width, self.height), (8, 10, 12))
        draw = self.ImageDraw.Draw(image)
        draw.text((28, 28), "SimRig Preview", fill=(236, 240, 245))
        draw.text((28, 56), message, fill=(180, 190, 200))
        return np.asarray(image)


def _command_controls(env: Any, state: Any) -> list[dict[str, Any]]:
    """Describe editable command axes only when the environment exposes them."""
    declared: Any = None
    for target in (env, getattr(env, "unwrapped", None)):
        if target is None:
            continue
        declared = getattr(target, "command_spec", None)
        if callable(declared):
            declared = declared()
        if declared is not None:
            break

    info = getattr(state, "info", {})
    command = info.get("command") if isinstance(info, Mapping) else None
    if declared is None and command is None:
        return []

    if isinstance(declared, Mapping):
        declared = declared.get("controls", declared.get("fields"))
    if declared is None:
        size = int(np.asarray(command).size)
        if size == 3:
            declared = [
                {"key": "forward_x", "label": "Forward X", "unit": "m/s"},
                {"key": "lateral_y", "label": "Lateral Y", "unit": "m/s"},
                {"key": "yaw", "label": "Yaw", "unit": "rad/s"},
            ]
        else:
            declared = [f"Command {index + 1}" for index in range(size)]
    if isinstance(declared, (str, bytes)) or not isinstance(declared, Sequence):
        raise ValueError("command_spec() must return a sequence or a mapping with controls.")

    controls: list[dict[str, Any]] = []
    used_keys: set[str] = set()
    for index, item in enumerate(declared):
        if isinstance(item, str):
            raw = {"key": item, "label": item}
        elif isinstance(item, Mapping):
            raw = dict(item)
        else:
            raise ValueError("Each command control must be a string or mapping.")
        label = str(raw.get("label", raw.get("name", raw.get("key", f"Command {index + 1}"))))
        key = _control_key(str(raw.get("key", raw.get("name", label))), index)
        if key in used_keys:
            key = f"{key}_{index + 1}"
        used_keys.add(key)
        control: dict[str, Any] = {
            "key": key,
            "label": label,
            "step": float(raw.get("step", 0.1)),
        }
        for name in ("unit", "min", "max"):
            if raw.get(name) is not None:
                control[name] = raw[name]
        controls.append(control)

    if command is not None and int(np.asarray(command).size) != len(controls):
        raise ValueError(
            "command_spec() control count must match the environment command vector size."
        )
    return controls


def _control_key(value: str, index: int) -> str:
    key = "".join(character.lower() if character.isalnum() else "_" for character in value)
    key = "_".join(part for part in key.split("_") if part)
    return key or f"command_{index + 1}"


def _rate_hz(timestep: Any) -> float | None:
    try:
        seconds = float(timestep)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(seconds) or seconds <= 0:
        return None
    return round(1.0 / seconds, 3)


def _episode_horizon(env: Any) -> int | None:
    for target in (env, getattr(env, "unwrapped", None)):
        if target is None:
            continue
        for name in ("config", "_config"):
            config = getattr(target, name, None)
            if isinstance(config, Mapping):
                value = config.get("episode_length")
            else:
                value = getattr(config, "episode_length", None)
            if value is not None:
                return int(value)
    return None


def _checkpoint_episode_horizon(checkpoint: Path | str) -> int | None:
    checkpoint_path = Path(checkpoint).expanduser()
    config_path = (
        checkpoint_path / "config.json"
        if checkpoint_path.is_dir()
        else checkpoint_path.parent / "config.json"
    )
    if not config_path.is_file():
        return None
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        value = payload.get("config", {}).get("episode_length")
        return int(value) if value is not None else None
    except (AttributeError, TypeError, ValueError, json.JSONDecodeError):
        return None


def serve_policy_preview(
    checkpoint: Path | str,
    *,
    env_name: str,
    host: str = "127.0.0.1",
    port: int = 8765,
    backend: str = "mujoco-playground",
    small_network: bool | None = None,
    seed: int = 0,
    command: tuple[float, ...] | None = None,
    width: int = 960,
    height: int = 540,
    frame_skip: int = 1,
    camera: str | int | None = None,
    render_mode: str = "threejs",
    paused: bool = False,
    fps: int = 24,
    auto_reset: bool = False,
    auto_reset_delay: float = 1.5,
    allow_runtime_mismatch: bool = False,
    reset_transform: Callable[[Any, Any], Any] | None = None,
) -> None:
    session = PolicyPreviewSession(
        checkpoint,
        env_name=env_name,
        backend=backend,
        small_network=small_network,
        seed=seed,
        command=command,
        width=width,
        height=height,
        frame_skip=frame_skip,
        camera=camera,
        render_mode=render_mode,
        paused=paused,
        fps=fps,
        auto_reset=auto_reset,
        auto_reset_delay=auto_reset_delay,
        allow_runtime_mismatch=allow_runtime_mismatch,
        reset_transform=reset_transform,
    )

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path == "/":
                self._send_html(_html(session.render_mode))
            elif parsed.path == "/frame.jpg":
                if session.render_mode == "threejs":
                    self.send_error(HTTPStatus.NOT_FOUND, "Frame streaming disabled")
                else:
                    self._send_bytes(session.frame_jpeg(), "image/jpeg")
            elif parsed.path == "/agent-frame.jpg":
                try:
                    self._send_bytes(session.agent_frame_jpeg(), "image/jpeg")
                except RuntimeError as exc:
                    self.send_error(HTTPStatus.NOT_FOUND, str(exc))
            elif parsed.path == "/agent-cameras.json":
                self._send_json(session.agent_cameras_payload())
            elif parsed.path == "/agent-camera":
                query = parse_qs(parsed.query)
                name = query.get("name", [""])[0]
                try:
                    self._send_json(session.select_agent_camera(name))
                except (RuntimeError, ValueError) as exc:
                    self.send_error(HTTPStatus.BAD_REQUEST, str(exc))
            elif parsed.path == "/scene.json":
                self._send_json(session.scene_payload(), compress=True)
            elif parsed.path == "/state.json":
                self._send_json(session.state_payload())
            elif parsed.path == "/status.json":
                self._send_json(session.status_payload())
            elif parsed.path == "/camera":
                query = parse_qs(parsed.query)
                session.set_camera_from_query(query)
                self._send_json(session.status_payload())
            elif parsed.path == "/command":
                query = parse_qs(parsed.query)
                try:
                    command_values = _command_from_query(query)
                    session.set_command(command_values)
                    self._send_json(session.status_payload())
                except ValueError as exc:
                    self.send_error(HTTPStatus.BAD_REQUEST, str(exc))
            elif parsed.path == "/auto-reset":
                query = parse_qs(parsed.query)
                session.set_auto_reset(query.get("enabled", ["0"])[0] == "1")
                self._send_json(session.status_payload())
            elif parsed.path == "/pause":
                session.set_paused(True)
                self._send_json(session.status_payload())
            elif parsed.path == "/resume":
                session.set_paused(False)
                self._send_json(session.status_payload())
            elif parsed.path == "/reset":
                session.reset()
                self._send_json(session.status_payload())
            else:
                self.send_error(HTTPStatus.NOT_FOUND, "Not found")

        def log_message(self, format: str, *args: Any) -> None:
            return

        def _send_html(self, body: str) -> None:
            self._send_bytes(body.encode("utf-8"), "text/html; charset=utf-8")

        def _send_json(self, value: dict[str, Any], *, compress: bool = False) -> None:
            body = json.dumps(value, separators=(",", ":")).encode("utf-8")
            if compress and "gzip" in self.headers.get("Accept-Encoding", ""):
                body = gzip.compress(body, compresslevel=5)
                self._send_bytes(
                    body,
                    "application/json; charset=utf-8",
                    content_encoding="gzip",
                )
                return
            self._send_bytes(body, "application/json; charset=utf-8")

        def _send_bytes(
            self,
            body: bytes,
            content_type: str,
            *,
            content_encoding: str | None = None,
        ) -> None:
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Cache-Control", "no-store")
            if content_encoding is not None:
                self.send_header("Content-Encoding", content_encoding)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer((host, port), Handler)
    print(f"SimRig preview: http://{host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        session.close()
        server.server_close()


def _command_from_query(query: dict[str, list[str]]) -> list[float] | None:
    if "clear" in query:
        return None
    if "value" in query:
        return [float(raw) for raw in query["value"]]
    values = []
    for key in ("x", "y", "yaw"):
        raw = query.get(key, ["0"])[0]
        values.append(float(raw))
    return values


def _html(render_mode: str = "threejs") -> str:
    if render_mode == "threejs":
        return _threejs_html()
    return _frame_html()


def _threejs_html() -> str:
    return (
        """<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>SimRig Preview</title>
  <style>"""
        + viewer_styles(sidebar_width=320)
        + agent_camera_styles()
        + """
    #three-view { width: 100%; height: 100%; display: block; outline: none; }
    #loading { position: absolute; inset: 0; display: grid; place-items: center; color: #cbd5e1; background: #070b12; z-index: 2; }
    #loading.error { color: #fca5a5; padding: 28px; text-align: center; white-space: pre-wrap; }
    #render-meta { color: #94a3b8; font-size: 12px; margin: -4px 0 12px; }
    #episode-state { border: 1px solid #334155; border-radius: 8px; padding: 10px 12px; margin-bottom: 10px; background: #0f172a; }
    #episode-state.failure { border-color: #ef4444; background: #2a1118; color: #fecaca; }
    #episode-state.success { border-color: #22c55e; background: #10251b; color: #bbf7d0; }
    #episode-state.paused { border-color: #f59e0b; color: #fde68a; }
    #run-facts { color: #94a3b8; font-size: 12px; line-height: 1.55; margin-bottom: 14px; }
    #command-section[hidden] { display: none; }
    #command-section { border-top: 1px solid #263244; padding-top: 12px; margin-top: 4px; }
    #command-section h2 { font-size: 14px; margin: 0 0 10px; }
    .command-label { display: flex; justify-content: space-between; gap: 8px; }
    .unit { color: #64748b; font-weight: 400; }
    .toggle { display: flex; align-items: center; gap: 8px; margin: 10px 0; color: #cbd5e1; }
    .toggle input { width: auto; margin: 0; }
  </style>
  <script type="importmap">
    {"imports": {
      "three": "https://cdn.jsdelivr.net/npm/three@0.184.0/build/three.module.js",
      "three/addons/": "https://cdn.jsdelivr.net/npm/three@0.184.0/examples/jsm/"
    }}
  </script>
</head>
<body>
  <main>
    <div id="viewport">
      <canvas id="three-view" aria-label="Interactive SimRig policy preview"></canvas>
      <div id="loading">Loading rollout scene…</div>
      <div id="hint">Drag to orbit · scroll to zoom · right-drag to pan</div>
"""
        + agent_camera_panel()
        + """
    </div>
  </main>
  <aside>
    <h1>SimRig Preview</h1>
    <div id="render-meta">Three.js · loading rollout…</div>
    <div id="episode-state">Episode 1 · starting</div>
    <div id="run-facts">Loading simulation rates…</div>
    <section id="command-section" hidden>
      <h2>Commands</h2>
      <div id="command-controls"></div>
      <button id="set-command">Set Command</button>
      <button class="secondary" id="clear-command">Clear</button>
    </section>
    <label class="toggle"><input id="auto-reset" type="checkbox"> Auto-reset ended episodes</label>
    <button class="secondary" id="pause">Pause</button>
    <button class="secondary" id="resume">Resume</button>
    <button class="secondary" id="reset">Reset</button>
    <button class="secondary" id="reset-camera">Reset Camera</button>
    <h1 style="margin-top:18px">Status</h1>
    <pre id="status">loading</pre>
  </aside>
  <script type="module">
    import * as THREE from 'three';
    import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

    const canvas = document.getElementById('three-view');
    const viewport = document.getElementById('viewport');
    const loadingEl = document.getElementById('loading');
    const statusEl = document.getElementById('status');
    const renderMetaEl = document.getElementById('render-meta');
    const episodeStateEl = document.getElementById('episode-state');
    const runFactsEl = document.getElementById('run-facts');
    const commandSectionEl = document.getElementById('command-section');
    const commandControlsEl = document.getElementById('command-controls');
    const autoResetEl = document.getElementById('auto-reset');
    const objects = new Map();
    const meshGeometries = new Map();
    let controlsInitialized = false;
    let commandSignature = '';
    let trackingPosition = null;
    let stateTimer = null;
    let targetPollMs = 33;
"""
        + agent_camera_script()
        + """

    const renderer = new THREE.WebGLRenderer({canvas, antialias: true, alpha: false});
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFShadowMap;
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.0;
    renderer.setClearColor(0x0b1220, 1);

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0b1220);
    scene.fog = new THREE.Fog(0x0b1220, 8, 26);

    const camera = new THREE.PerspectiveCamera(42, 1, 0.01, 1000);
    camera.up.set(0, 0, 1);
    const orbit = new OrbitControls(camera, canvas);
    orbit.enableDamping = true;
    orbit.dampingFactor = 0.075;
    orbit.screenSpacePanning = false;
    orbit.minDistance = 0.08;
    orbit.maxDistance = 100;
    orbit.minPolarAngle = 0.08;
    orbit.maxPolarAngle = Math.PI / 2 - 0.04;

    scene.add(new THREE.HemisphereLight(0xbfdcff, 0x172033, 1.15));
    const keyLight = new THREE.DirectionalLight(0xffffff, 2.35);
    keyLight.position.set(4, -5, 8);
    keyLight.castShadow = true;
    keyLight.shadow.mapSize.set(2048, 2048);
    keyLight.shadow.camera.near = 0.1;
    keyLight.shadow.camera.far = 30;
    keyLight.shadow.camera.left = -5;
    keyLight.shadow.camera.right = 5;
    keyLight.shadow.camera.top = 5;
    keyLight.shadow.camera.bottom = -5;
    keyLight.shadow.bias = -0.0002;
    scene.add(keyLight, keyLight.target);
    const rimLight = new THREE.DirectionalLight(0x7aa8ff, 0.85);
    rimLight.position.set(-5, 3, 5);
    scene.add(rimLight);

    const modelRoot = new THREE.Group();
    scene.add(modelRoot);
"""
        + threejs_agent_camera_script()
        + """

    function materialFor(geom) {
      const [r, g, b, a] = geom.rgba;
      const props = geom.material || {};
      if (geom.type === 0) {
        return new THREE.MeshStandardMaterial({color: 0x182231, roughness: 0.92});
      }
      const material = new THREE.MeshPhysicalMaterial({
        color: new THREE.Color(r, g, b),
        opacity: a,
        transparent: a < 0.999,
        roughness: THREE.MathUtils.clamp(0.68 - (props.shininess || 0) * 0.32, 0.22, 0.82),
        metalness: THREE.MathUtils.clamp((props.reflectance || 0) * 0.45, 0, 0.35),
        clearcoat: THREE.MathUtils.clamp((props.specular || 0) * 0.35, 0, 0.4),
        clearcoatRoughness: 0.35,
      });
      if ((props.emission || 0) > 0) {
        material.emissive.setRGB(r, g, b);
        material.emissiveIntensity = props.emission;
      }
      return material;
    }

    function primitiveGeometry(geom) {
      const [x, y, z] = geom.size;
      switch (geom.type) {
        case 0:
          return new THREE.PlaneGeometry(200, 200);
        case 2:
          return new THREE.SphereGeometry(x, 32, 20);
        case 3: {
          const geometry = new THREE.CapsuleGeometry(x, 2 * y, 10, 24);
          geometry.rotateX(Math.PI / 2);
          return geometry;
        }
        case 4: {
          const geometry = new THREE.SphereGeometry(1, 32, 20);
          geometry.scale(x, y, z);
          return geometry;
        }
        case 5: {
          const geometry = new THREE.CylinderGeometry(x, x, 2 * y, 32);
          geometry.rotateX(Math.PI / 2);
          return geometry;
        }
        case 6:
          return new THREE.BoxGeometry(2 * x, 2 * y, 2 * z);
        case 7:
          return meshGeometries.get(geom.mesh_id) || null;
        default:
          return null;
      }
    }

    function applyTransform(object, transform) {
      if (!object || !transform) return;
      object.position.fromArray(transform.position);
      const m = transform.matrix;
      const rotation = new THREE.Matrix4();
      rotation.set(
        m[0], m[1], m[2], 0,
        m[3], m[4], m[5], 0,
        m[6], m[7], m[8], 0,
        0, 0, 0, 1,
      );
      object.quaternion.setFromRotationMatrix(rotation);
    }

    function updateTransforms(transforms) {
      for (const transform of transforms || []) {
        applyTransform(objects.get(transform.id), transform);
      }
    }

    function followRobot(rawPosition) {
      if (!Array.isArray(rawPosition)) return;
      const next = new THREE.Vector3().fromArray(rawPosition);
      if (trackingPosition !== null) {
        const delta = next.clone().sub(trackingPosition);
        camera.position.add(delta);
        orbit.target.add(delta);
        keyLight.position.add(delta);
        keyLight.target.position.add(delta);
      }
      trackingPosition = next;
    }

    function fitCamera() {
      const bounds = new THREE.Box3().setFromObject(modelRoot);
      const center = bounds.getCenter(new THREE.Vector3());
      const size = bounds.getSize(new THREE.Vector3());
      const radius = Math.max(size.x, size.y, size.z, 0.25);
      orbit.target.copy(center);
      camera.position.set(
        center.x + radius * 1.35,
        center.y - radius * 1.75,
        center.z + radius * 0.95,
      );
      camera.near = Math.max(radius / 200, 0.002);
      camera.far = Math.max(radius * 80, 100);
      camera.updateProjectionMatrix();
      orbit.update();
      trackingPosition = null;
    }

    function resize() {
      const width = Math.max(1, viewport.clientWidth);
      const height = Math.max(1, viewport.clientHeight);
      renderer.setSize(width, height, false);
      camera.aspect = width / height;
      camera.updateProjectionMatrix();
    }

    function animate() {
      orbit.update();
      renderer.render(scene, camera);
      renderAgentCameraEmulation();
      requestAnimationFrame(animate);
    }

    async function loadScene() {
      const res = await fetch('/scene.json', {cache: 'no-store'});
      if (!res.ok) throw new Error(`scene request failed (${res.status})`);
      const payload = await res.json();
      targetPollMs = Math.max(16, Math.round(1000 / (payload.fps_target || 24)));
      updateAuthoredCameras(payload.authored_cameras);

      for (const mesh of payload.meshes) {
        const geometry = new THREE.BufferGeometry();
        geometry.setAttribute('position', new THREE.Float32BufferAttribute(mesh.vertices, 3));
        geometry.setIndex(mesh.indices);
        geometry.computeVertexNormals();
        geometry.computeBoundingSphere();
        meshGeometries.set(mesh.id, geometry);
      }

      const transformById = new Map(payload.transforms.map(item => [item.id, item]));
      for (const geom of payload.geoms) {
        const geometry = primitiveGeometry(geom);
        if (!geometry || geom.rgba[3] <= 0.001) continue;
        const object = new THREE.Mesh(geometry, materialFor(geom));
        object.name = geom.name;
        object.castShadow = geom.type !== 0;
        object.receiveShadow = true;
        applyTransform(object, transformById.get(geom.id));
        if (geom.type === 0) {
          scene.add(object);
        } else {
          modelRoot.add(object);
        }
        objects.set(geom.id, object);
      }

      const grid = new THREE.GridHelper(40, 80, 0x52647a, 0x263346);
      grid.rotation.x = Math.PI / 2;
      grid.position.z = 0.001;
      grid.material.opacity = 0.42;
      grid.material.transparent = true;
      scene.add(grid);

      fitCamera();
      followRobot(payload.tracking_position);
      loadingEl.remove();
    }

    function displayStatus(status) {
      const copy = {...status};
      delete copy.transforms;
      delete copy.authored_cameras;
      delete copy.tracking_position;
      statusEl.textContent = JSON.stringify(copy, null, 2);
      renderMetaEl.textContent = `${status.env_name} · ${status.playback_rate_hz || status.fps_target || 24} Hz playback · display-rate WebGL`;
      episodeStateEl.className = status.done ? (status.episode_outcome || 'unknown') : status.episode_state;
      episodeStateEl.title = status.task_success_reason || '';
      const outcome = status.task_success === true ? 'Task succeeded'
        : status.task_success === false ? 'Task failed' : 'Ended · outcome unknown';
      episodeStateEl.textContent = `Episode ${status.episode} · ${status.done ? outcome : status.episode_state} · ${status.step} steps`;
      const rate = value => value == null ? 'n/a' : `${value} Hz`;
      const horizon = status.episode_horizon == null ? 'n/a' : `${status.episode_horizon} steps`;
      const lastEpisode = status.last_episode_survival_steps == null ? '' : ` · last episode ${status.last_episode_survival_steps} steps`;
      runFactsEl.textContent = `Physics ${rate(status.physics_rate_hz)} · policy observations ${rate(status.policy_observation_rate_hz)} · Sensor display ${rate(status.sensor_display_rate_hz)} · horizon ${horizon}${lastEpisode}`;
      autoResetEl.checked = Boolean(status.auto_reset);
      renderCommandControls(status);
    }

    function renderCommandControls(status) {
      const controls = Array.isArray(status.command_controls) ? status.command_controls : [];
      commandSectionEl.hidden = !status.command_supported || controls.length === 0;
      if (commandSectionEl.hidden) return;
      const nextSignature = JSON.stringify(controls);
      if (nextSignature !== commandSignature) {
        commandControlsEl.replaceChildren();
        controls.forEach((control, index) => {
          const label = document.createElement('label');
          label.className = 'command-label';
          label.textContent = control.label;
          if (control.unit) {
            const unit = document.createElement('span');
            unit.className = 'unit';
            unit.textContent = control.unit;
            label.appendChild(unit);
          }
          const input = document.createElement('input');
          input.type = 'number';
          input.step = control.step ?? 0.1;
          if (control.min != null) input.min = control.min;
          if (control.max != null) input.max = control.max;
          input.dataset.commandIndex = index;
          commandControlsEl.append(label, input);
        });
        commandSignature = nextSignature;
        controlsInitialized = false;
      }
      if (!controlsInitialized && Array.isArray(status.command_values)) {
        commandControlsEl.querySelectorAll('input').forEach((input, index) => {
          input.value = status.command_values[index] ?? 0;
        });
        controlsInitialized = true;
      }
    }

    async function refreshState() {
      clearTimeout(stateTimer);
      try {
        const res = await fetch('/state.json', {cache: 'no-store'});
        if (!res.ok) throw new Error(`state request failed (${res.status})`);
        const status = await res.json();
        updateTransforms(status.transforms);
        updateAuthoredCameras(status.authored_cameras);
        followRobot(status.tracking_position);
        displayStatus(status);
      } catch (err) {
        statusEl.textContent = String(err);
      } finally {
        stateTimer = setTimeout(refreshState, targetPollMs);
      }
    }

    async function call(path) {
      const res = await fetch(path, {cache: 'no-store'});
      displayStatus(await res.json());
    }

    async function setCommand() {
      const params = new URLSearchParams();
      commandControlsEl.querySelectorAll('input').forEach(input => params.append('value', input.value));
      await call('/command?' + params.toString());
    }

    async function clearCommand() {
      controlsInitialized = false;
      await call('/command?clear=1');
    }

    document.getElementById('set-command').addEventListener('click', setCommand);
    document.getElementById('clear-command').addEventListener('click', clearCommand);
    autoResetEl.addEventListener('change', () => call(`/auto-reset?enabled=${autoResetEl.checked ? 1 : 0}`));
    document.getElementById('pause').addEventListener('click', () => call('/pause'));
    document.getElementById('resume').addEventListener('click', () => call('/resume'));
    document.getElementById('reset').addEventListener('click', () => call('/reset'));
    document.getElementById('reset-camera').addEventListener('click', fitCamera);
    window.addEventListener('resize', resize);
    resize();
    animate();

    try {
      await Promise.all([loadScene(), initializeAgentCamera()]);
      await refreshState();
    } catch (err) {
      loadingEl.className = 'error';
      loadingEl.textContent = `WebGL preview failed to load.\n${err}\n\nThree.js is loaded from jsDelivr, so an internet connection is required.`;
      statusEl.textContent = String(err);
      console.error(err);
    }
  </script>
</body>
</html>
"""
    )


def _frame_html() -> str:
    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>SimRig Preview</title>
  <style>{viewer_styles(sidebar_width=320)}
    #episode-state {{ border: 1px solid #334155; border-radius: 8px; padding: 10px 12px; margin-bottom: 10px; background: #0f172a; }}
    #episode-state.failure {{ border-color: #ef4444; background: #2a1118; color: #fecaca; }}
    #episode-state.success {{ border-color: #22c55e; background: #10251b; color: #bbf7d0; }}
    #episode-state.paused {{ border-color: #f59e0b; color: #fde68a; }}
    #run-facts {{ color: #94a3b8; font-size: 12px; line-height: 1.55; margin-bottom: 14px; }}
    #command-section[hidden] {{ display: none; }}
    #command-section {{ border-top: 1px solid #263244; padding-top: 12px; margin-top: 4px; }}
    #command-section h2 {{ font-size: 14px; margin: 0 0 10px; }}
    .command-label {{ display: flex; justify-content: space-between; gap: 8px; }}
    .unit {{ color: #64748b; font-weight: 400; }}
    .toggle {{ display: flex; align-items: center; gap: 8px; margin: 10px 0; color: #cbd5e1; }}
    .toggle input {{ width: auto; margin: 0; }}
  </style>
</head>
<body>
  <main>
    <div id="viewport">
      <img id="frame" alt="SimRig rendered frame">
      <div id="hint">Drag to orbit · scroll to zoom</div>
    </div>
  </main>
  <aside>
    <h1>SimRig Preview</h1>
    <div id="episode-state">Episode 1 · starting</div>
    <div id="run-facts">Loading simulation rates…</div>
    <section id="command-section" hidden>
      <h2>Commands</h2>
      <div id="command-controls"></div>
      <button onclick="setCommand()">Set Command</button>
      <button class="secondary" onclick="clearCommand()">Clear</button>
    </section>
    <label class="toggle"><input id="auto-reset" type="checkbox"> Auto-reset ended episodes</label>
    <button class="secondary" onclick="call('/pause')">Pause</button>
    <button class="secondary" onclick="call('/resume')">Resume</button>
    <button class="secondary" onclick="call('/reset')">Reset</button>
    <h1 style="margin-top:18px">Status</h1>
    <pre id="status">loading</pre>
  </aside>
  <script>
    const frame = document.getElementById('frame');
    const statusEl = document.getElementById('status');
    const episodeStateEl = document.getElementById('episode-state');
    const runFactsEl = document.getElementById('run-facts');
    const commandSectionEl = document.getElementById('command-section');
    const commandControlsEl = document.getElementById('command-controls');
    const autoResetEl = document.getElementById('auto-reset');
    let controlsInitialized = false;
    let commandSignature = '';
    {camera_interaction_script()}
    {frame_poll_script(poll_ms=33)}
    bindCameraControls(frame);
    async function refreshStatus() {{
      try {{
        const res = await fetch('/status.json', {{cache: 'no-store'}});
        const status = await res.json();
        statusEl.textContent = JSON.stringify(status, null, 2);
        applyCameraFromStatus(status);
        episodeStateEl.className = status.done ? (status.episode_outcome || 'unknown') : status.episode_state;
        episodeStateEl.title = status.task_success_reason || '';
        const outcome = status.task_success === true ? 'Task succeeded'
          : status.task_success === false ? 'Task failed' : 'Ended · outcome unknown';
        episodeStateEl.textContent = `Episode ${{status.episode}} · ${{status.done ? outcome : status.episode_state}} · ${{status.step}} steps`;
        const rate = value => value == null ? 'n/a' : `${{value}} Hz`;
        const horizon = status.episode_horizon == null ? 'n/a' : `${{status.episode_horizon}} steps`;
        const lastEpisode = status.last_episode_survival_steps == null ? '' : ` · last episode ${{status.last_episode_survival_steps}} steps`;
        runFactsEl.textContent = `Physics ${{rate(status.physics_rate_hz)}} · policy observations ${{rate(status.policy_observation_rate_hz)}} · Sensor display ${{rate(status.sensor_display_rate_hz)}} · horizon ${{horizon}}${{lastEpisode}}`;
        autoResetEl.checked = Boolean(status.auto_reset);
        renderCommandControls(status);
      }} catch (err) {{
        statusEl.textContent = String(err);
      }}
    }}
    async function call(path) {{ await fetch(path); await refreshStatus(); }}
    function renderCommandControls(status) {{
      const controls = Array.isArray(status.command_controls) ? status.command_controls : [];
      commandSectionEl.hidden = !status.command_supported || controls.length === 0;
      if (commandSectionEl.hidden) return;
      const nextSignature = JSON.stringify(controls);
      if (nextSignature !== commandSignature) {{
        commandControlsEl.replaceChildren();
        controls.forEach((control, index) => {{
          const label = document.createElement('label');
          label.className = 'command-label';
          label.textContent = control.label;
          if (control.unit) {{
            const unit = document.createElement('span');
            unit.className = 'unit';
            unit.textContent = control.unit;
            label.appendChild(unit);
          }}
          const input = document.createElement('input');
          input.type = 'number';
          input.step = control.step ?? 0.1;
          if (control.min != null) input.min = control.min;
          if (control.max != null) input.max = control.max;
          input.dataset.commandIndex = index;
          commandControlsEl.append(label, input);
        }});
        commandSignature = nextSignature;
        controlsInitialized = false;
      }}
      if (!controlsInitialized && Array.isArray(status.command_values)) {{
        commandControlsEl.querySelectorAll('input').forEach((input, index) => {{
          input.value = status.command_values[index] ?? 0;
        }});
        controlsInitialized = true;
      }}
    }}
    async function setCommand() {{
      const params = new URLSearchParams();
      commandControlsEl.querySelectorAll('input').forEach(input => params.append('value', input.value));
      await call('/command?' + params.toString());
    }}
    autoResetEl.addEventListener('change', () => call(`/auto-reset?enabled=${{autoResetEl.checked ? 1 : 0}}`));
    async function clearCommand() {{
      controlsInitialized = false;
      await call('/command?clear=1');
    }}
    setInterval(refreshStatus, 500);
    refreshFrame();
    refreshStatus();
  </script>
</body>
</html>
"""
