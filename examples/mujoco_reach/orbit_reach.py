"""Goal-conditioned 3-DOF reaching task for the orbit-arm example.

The actor receives joint state, the world-frame end-effector target delta, and
its previous action. Actions are normalized absolute joint-position targets.
Success requires five consecutive control ticks within five centimetres.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import jax
import jax.numpy as jp
import mujoco
from flax import struct
from mujoco import mjx


MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "orbit_arm.xml"
ENV_NAME = "orbit_reach"

DEFAULT_CONFIG = {
    "episode_length": 200,
    "action_scale": 1.0,
    "max_command_delta": 0.03,
    "velocity_observation_scale": 0.1,
    "ctrl_dt": 0.02,
    "sim_dt": 0.002,
    "impl": "jax",
    "success_distance": 0.05,
    "success_hold_steps": 5,
    "target_radius_min": 0.22,
    "target_radius_max": 0.32,
    "target_height_min": 0.08,
    "target_height_max": 0.24,
    "reward_distance_temperature": 12.0,
    "reward_progress_scale": 4.0,
    "reward_in_tolerance": 1.0,
    "reward_hold_scale": 1.5,
    "reward_success": 20.0,
    "reward_action_rate_scale": 0.02,
    "reward_near_target_velocity_scale": 0.02,
    "reward_collision": -5.0,
}

TRAINING_CONFIG = {
    "num_timesteps": 2_000_000,
    "num_envs": 2048,
    "num_eval_envs": 64,
    "num_evals": 10,
    "episode_length": 200,
    "batch_size": 256,
    "unroll_length": 20,
    "num_minibatches": 16,
    "num_updates_per_batch": 4,
    "discounting": 0.97,
    "learning_rate": 0.0003,
    "entropy_cost": 0.003,
    "network_factory": {
        "policy_hidden_layer_sizes": [128, 128],
        "value_hidden_layer_sizes": [128, 128],
        "policy_obs_key": "state",
        "value_obs_key": "privileged_state",
    },
}

SUCCESS_SPEC = {"metric": "success", "threshold": 0.5, "mode": "any"}


def default_config() -> dict[str, Any]:
    return dict(DEFAULT_CONFIG)


def training_config() -> dict[str, Any]:
    return dict(TRAINING_CONFIG)


def make_env(config_overrides: dict[str, Any] | None = None) -> "CustomEnv":
    config = default_config()
    if config_overrides:
        config.update(config_overrides)
    return CustomEnv(config=config)


@struct.dataclass
class State:
    data: mjx.Data
    obs: dict[str, jax.Array]
    reward: jax.Array
    done: jax.Array
    metrics: dict[str, jax.Array]
    info: dict[str, Any]


class CustomEnv:
    """Editable MJX reach environment backed by ``orbit_arm.xml``."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        # SECTION: model loading
        self.config = config or default_config()
        if not MODEL_PATH.is_file():
            raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
        self._mj_model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
        self._mj_model.opt.timestep = float(self.config["sim_dt"])
        self._mjx_model = mjx.put_model(self._mj_model)
        self._ee_site_id = self._require_id(mujoco.mjtObj.mjOBJ_SITE, "ee_site")
        target_body_id = self._require_id(mujoco.mjtObj.mjOBJ_BODY, "target")
        self._target_mocap_id = int(self._mj_model.body_mocapid[target_body_id])
        if self._target_mocap_id < 0:
            raise ValueError("orbit_arm.xml target body must be mocap-enabled")
        self._floor_geom_id = self._require_id(mujoco.mjtObj.mjOBJ_GEOM, "floor")
        self._base_geom_id = self._require_id(mujoco.mjtObj.mjOBJ_GEOM, "base_geom")
        self._arm_geom_ids = jp.asarray([
            self._require_id(mujoco.mjtObj.mjOBJ_GEOM, "link1_geom"),
            self._require_id(mujoco.mjtObj.mjOBJ_GEOM, "link2_geom"),
        ])
        self._ctrl_low = jp.asarray(self._mj_model.actuator_ctrlrange[:, 0])
        self._ctrl_high = jp.asarray(self._mj_model.actuator_ctrlrange[:, 1])
        self._n_substeps = int(round(self.dt / self.sim_dt))
        if self._n_substeps < 1 or not bool(
            jp.isclose(self._n_substeps * self.sim_dt, self.dt)
        ):
            raise ValueError("Control interval must be an integer number of physics steps")

    def _require_id(self, object_type: mujoco.mjtObj, name: str) -> int:
        object_id = mujoco.mj_name2id(self._mj_model, object_type, name)
        if object_id < 0:
            raise ValueError(f"orbit_arm.xml must define {name}")
        return int(object_id)

    @property
    def mj_model(self) -> mujoco.MjModel:
        return self._mj_model

    @property
    def mjx_model(self) -> mjx.Model:
        return self._mjx_model

    @property
    def xml_path(self) -> str:
        return str(MODEL_PATH)

    @property
    def dt(self) -> float:
        return float(self.config["ctrl_dt"])

    @property
    def sim_dt(self) -> float:
        return float(self.config["sim_dt"])

    @property
    def action_size(self) -> int:
        return int(self._mj_model.nu)

    @property
    def observation_size(self) -> dict[str, int]:
        return {"state": 13, "privileged_state": 21}

    @property
    def unwrapped(self) -> "CustomEnv":
        return self

    def command_spec(self) -> list[dict[str, Any]]:
        return [
            {"key": "angle", "label": "Target angle", "unit": "rad", "step": 0.1,
             "min": -3.14159, "max": 3.14159},
            {"key": "radius", "label": "Target radius", "unit": "m", "step": 0.01,
             "min": self.config["target_radius_min"], "max": self.config["target_radius_max"]},
            {"key": "height", "label": "Target height", "unit": "m", "step": 0.01,
             "min": self.config["target_height_min"], "max": self.config["target_height_max"]},
        ]

    def reset(self, rng: jax.Array) -> State:
        # SECTION: reset
        rng, q_rng, angle_rng, radius_rng, height_rng = jax.random.split(rng, 5)
        data = mjx.make_data(self._mjx_model)
        q_min = jp.asarray([-jp.pi, -1.2, 0.8])
        q_max = jp.asarray([jp.pi, -0.5, 1.8])
        qpos = jax.random.uniform(q_rng, (3,), minval=q_min, maxval=q_max)
        data = data.replace(
            qpos=qpos,
            qvel=jp.zeros(self._mj_model.nv),
            ctrl=qpos,
        )
        angle = jax.random.uniform(angle_rng, (), minval=-jp.pi, maxval=jp.pi)
        radius = jax.random.uniform(
            radius_rng, (), minval=float(self.config["target_radius_min"]),
            maxval=float(self.config["target_radius_max"]),
        )
        height = jax.random.uniform(
            height_rng, (), minval=float(self.config["target_height_min"]),
            maxval=float(self.config["target_height_max"]),
        )
        target = jp.asarray([radius * jp.cos(angle), radius * jp.sin(angle), height])
        data = data.replace(
            mocap_pos=data.mocap_pos.at[self._target_mocap_id].set(target)
        )
        data = mjx.forward(self._mjx_model, data)
        distance = self._distance(data, target)
        info = {
            "rng": rng,
            "steps": jp.array(0, dtype=jp.int32),
            "hold_steps": jp.array(0, dtype=jp.int32),
            "target_pos": target,
            "command": jp.asarray([angle, radius, height]),
            "previous_action": self._normalized_qpos(qpos),
            "previous_distance": distance,
        }
        return State(
            data=data,
            obs=self._get_obs(data, info),
            reward=jp.array(0.0),
            done=jp.array(0.0),
            metrics=self._metrics_zero(distance),
            info=info,
        )

    def step(self, state: State, action: jax.Array) -> State:
        # SECTION: action mapping
        desired_action = jp.clip(
            action * float(self.config["action_scale"]), -1.0, 1.0
        )
        current_action = self._normalized_qpos(state.data.qpos[: self.action_size])
        max_delta = float(self.config["max_command_delta"])
        applied_action = jp.clip(
            desired_action,
            current_action - max_delta,
            current_action + max_delta,
        )
        ctrl = self._ctrl_low + 0.5 * (applied_action + 1.0) * (
            self._ctrl_high - self._ctrl_low
        )
        data = state.data.replace(ctrl=ctrl)

        def body_step(value, _):
            return mjx.step(self._mjx_model, value), None

        data, _ = jax.lax.scan(body_step, data, None, self._n_substeps)
        data = mjx.forward(self._mjx_model, data)
        # SECTION: observations
        raw_distance = self._distance(data, state.info["target_pos"])
        finite = (
            jp.all(jp.isfinite(data.qpos))
            & jp.all(jp.isfinite(data.qvel))
            & jp.isfinite(raw_distance)
        )
        distance = jp.nan_to_num(raw_distance, nan=1.0, posinf=1.0, neginf=1.0)
        within = distance < float(self.config["success_distance"])
        hold_steps = jp.where(within, state.info["hold_steps"] + 1, 0)
        success = hold_steps >= int(self.config["success_hold_steps"])
        collision = self._forbidden_collision(data)
        steps = state.info["steps"] + 1
        timeout = steps >= int(self.config["episode_length"])
        # SECTION: rewards
        distance_reward = jp.exp(
            -float(self.config["reward_distance_temperature"]) * distance
        )
        progress_reward = float(self.config["reward_progress_scale"]) * (
            state.info["previous_distance"] - distance
        )
        action_rate_cost = float(self.config["reward_action_rate_scale"]) * jp.sum(
            jp.square(applied_action - state.info["previous_action"])
        )
        hold_reward = float(self.config["reward_hold_scale"]) * hold_steps.astype(
            jp.float32
        )
        near_target_velocity_cost = (
            float(self.config["reward_near_target_velocity_scale"])
            * within.astype(jp.float32)
            * jp.sum(jp.square(data.qvel))
        )
        reward = (
            distance_reward
            + progress_reward
            + float(self.config["reward_in_tolerance"]) * within.astype(jp.float32)
            + hold_reward
            + float(self.config["reward_success"]) * success.astype(jp.float32)
            - action_rate_cost
            - near_target_velocity_cost
            + float(self.config["reward_collision"]) * collision.astype(jp.float32)
        )
        reward = jp.where(finite, reward, float(self.config["reward_collision"]))
        # SECTION: termination
        done = (success | collision | (~finite) | timeout).astype(jp.float32)
        info = dict(state.info)
        info.update({
            "steps": steps,
            "hold_steps": hold_steps,
            "previous_action": applied_action,
            "previous_distance": distance,
        })
        metrics = {
            "reward": reward,
            "distance": distance,
            "within_tolerance": within.astype(jp.float32),
            "hold_steps": hold_steps.astype(jp.float32),
            "success": success.astype(jp.float32),
            "collision": collision.astype(jp.float32),
            "invalid_state": (~finite).astype(jp.float32),
            "timeout": timeout.astype(jp.float32),
            "reward_distance": distance_reward,
            "reward_progress": progress_reward,
            "reward_hold": hold_reward,
            "cost_action_rate": action_rate_cost,
            "cost_near_target_velocity": near_target_velocity_cost,
        }
        return State(
            data=data,
            obs=self._get_obs(data, info),
            reward=reward,
            done=done,
            metrics=metrics,
            info=info,
        )

    def set_command(self, state: State, command: jax.Array) -> State:
        angle = jp.clip(command[0], -jp.pi, jp.pi)
        radius = jp.clip(
            command[1], self.config["target_radius_min"], self.config["target_radius_max"]
        )
        height = jp.clip(
            command[2], self.config["target_height_min"], self.config["target_height_max"]
        )
        target = jp.asarray([radius * jp.cos(angle), radius * jp.sin(angle), height])
        return self.set_target(
            state, target, command=jp.asarray([angle, radius, height]), reset_counters=True
        )

    def set_target(
        self,
        state: State,
        target: jax.Array,
        *,
        command: jax.Array | None = None,
        reset_counters: bool = True,
    ) -> State:
        target = jp.asarray(target)
        data = state.data.replace(
            mocap_pos=state.data.mocap_pos.at[self._target_mocap_id].set(target)
        )
        data = mjx.forward(self._mjx_model, data)
        distance = self._distance(data, target)
        info = dict(state.info)
        if command is None:
            angle = jp.arctan2(target[1], target[0])
            radius = jp.linalg.norm(target[:2])
            command = jp.asarray([angle, radius, target[2]])
        info.update({
            "target_pos": target,
            "command": command,
            "previous_distance": distance,
        })
        if reset_counters:
            info.update({
                "steps": jp.array(0, dtype=jp.int32),
                "hold_steps": jp.array(0, dtype=jp.int32),
                "previous_action": self._normalized_qpos(data.qpos[: self.action_size]),
            })
        return state.replace(
            data=data,
            obs=self._get_obs(data, info),
            reward=jp.array(0.0),
            done=jp.array(0.0),
            metrics=self._metrics_zero(distance),
            info=info,
        )

    def _ee_pos(self, data: mjx.Data) -> jax.Array:
        return data.site_xpos[self._ee_site_id]

    def _distance(self, data: mjx.Data, target: jax.Array) -> jax.Array:
        return jp.linalg.norm(self._ee_pos(data) - target)

    def _forbidden_collision(self, data: mjx.Data) -> jax.Array:
        contacts = data.contact
        geom_a = contacts.geom[:, 0]
        geom_b = contacts.geom[:, 1]
        active = contacts.dist < 0.0
        arm_a = jp.any(geom_a[:, None] == self._arm_geom_ids[None, :], axis=1)
        arm_b = jp.any(geom_b[:, None] == self._arm_geom_ids[None, :], axis=1)
        forbidden_a = (geom_a == self._floor_geom_id) | (geom_a == self._base_geom_id)
        forbidden_b = (geom_b == self._floor_geom_id) | (geom_b == self._base_geom_id)
        return jp.any(active & ((arm_a & forbidden_b) | (arm_b & forbidden_a)))

    def _get_obs(self, data: mjx.Data, info: dict[str, Any]) -> dict[str, jax.Array]:
        ee = self._ee_pos(data)
        target = info["target_pos"]
        delta = target - ee
        joint_state = jp.asarray([
            jp.sin(data.qpos[0]), jp.cos(data.qpos[0]), data.qpos[1], data.qpos[2]
        ])
        scaled_velocity = jp.clip(
            data.qvel * float(self.config["velocity_observation_scale"]), -5.0, 5.0
        )
        state = jp.concatenate([
            joint_state, scaled_velocity, delta, info["previous_action"]
        ])
        distance = jp.linalg.norm(delta).reshape((1,))
        collision = self._forbidden_collision(data).astype(jp.float32).reshape((1,))
        privileged = jp.concatenate([state, ee, target, distance, collision])
        return {"state": state, "privileged_state": privileged}

    def _normalized_qpos(self, qpos: jax.Array) -> jax.Array:
        return 2 * (qpos - self._ctrl_low) / (self._ctrl_high - self._ctrl_low) - 1

    def _metrics_zero(self, distance: jax.Array) -> dict[str, jax.Array]:
        zero = jp.array(0.0)
        return {
            "reward": zero,
            "distance": distance,
            "within_tolerance": zero,
            "hold_steps": zero,
            "success": zero,
            "collision": zero,
            "invalid_state": zero,
            "timeout": zero,
            "reward_distance": zero,
            "reward_progress": zero,
            "reward_hold": zero,
            "cost_action_rate": zero,
            "cost_near_target_velocity": zero,
        }
