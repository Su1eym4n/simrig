"""demo_reach — test-drive custom env for SimRig tooling.

Recreate from CLI (same idea):
  simrig new-env demo_reach --model examples/models/simple_arm.xml --root envs
  # then fill SECTION blocks (this file is the filled reference)

Run:
  simrig validate-env examples/demo_reach.py --runtime
  simrig smoke examples/demo_reach.py --steps 10
  simrig train examples/demo_reach.py --preset smoke
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import jax
import jax.numpy as jp
import mujoco
from flax import struct
from mujoco import mjx

MODEL_PATH = (Path(__file__).resolve().parent / "models" / "simple_arm.xml").expanduser()
ENV_NAME = "demo_reach"


def default_config() -> dict[str, Any]:
    return {
        "episode_length": 200,
        "action_scale": 1.0,
        "ctrl_dt": 0.02,
        "sim_dt": 0.002,
        "impl": "jax",
        "reward_distance_scale": 1.0,
        "reward_success": 1.0,
        "success_distance": 0.05,
    }


def make_env(config_overrides: dict[str, Any] | None = None) -> "CustomEnv":
    """Factory used by `simrig smoke/train/eval` for this module."""
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
    """Minimal 2-DOF reach task to exercise SimRig custom-module tooling."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or default_config()
        # SECTION: model loading
        if not MODEL_PATH.is_file():
            raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
        self._mj_model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
        self._mj_model.opt.timestep = float(self.config["sim_dt"])
        self._mjx_model = mjx.put_model(self._mj_model)
        self._ee_site_id = mujoco.mj_name2id(self._mj_model, mujoco.mjtObj.mjOBJ_SITE, "ee_site")
        if self._ee_site_id < 0:
            raise ValueError("simple_arm.xml must define site ee_site")
        low = self._mj_model.actuator_ctrlrange[:, 0]
        high = self._mj_model.actuator_ctrlrange[:, 1]
        self._ctrl_low = jp.asarray(low)
        self._ctrl_high = jp.asarray(high)
        self._n_substeps = max(
            1, int(round(float(self.config["ctrl_dt"]) / float(self.config["sim_dt"])))
        )

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
        # qpos(2) + qvel(2) + delta_ee_to_target(3) = 7
        # privileged adds ee(3) + distance(1) = 11
        return {"state": 7, "privileged_state": 11}

    @property
    def unwrapped(self) -> "CustomEnv":
        return self

    def reset(self, rng: jax.Array) -> State:
        # SECTION: reset
        rng, q_rng, target_rng = jax.random.split(rng, 3)
        data = mjx.make_data(self._mjx_model)
        qpos = data.qpos.at[:].set(
            jax.random.uniform(q_rng, (self._mj_model.nq,), minval=-0.4, maxval=0.4)
        )
        data = data.replace(qpos=qpos, qvel=jp.zeros(self._mj_model.nv))
        target_xy = jax.random.uniform(
            target_rng,
            (2,),
            minval=jp.array([0.12, -0.12]),
            maxval=jp.array([0.32, 0.12]),
        )
        target_pos = jp.array([target_xy[0], target_xy[1], 0.15])
        data = mjx.forward(self._mjx_model, data)
        info = {
            "rng": rng,
            "steps": jp.array(0, dtype=jp.int32),
            "target_pos": target_pos,
        }
        obs = self._get_obs(data, info)
        return State(
            data=data,
            obs=obs,
            reward=jp.array(0.0),
            done=jp.array(0.0),
            metrics={
                "reward": jp.array(0.0),
                "distance": self._distance(data, info),
                "success": jp.array(0.0),
            },
            info=info,
        )

    def step(self, state: State, action: jax.Array) -> State:
        # SECTION: action mapping
        action = jp.clip(action, -1.0, 1.0) * float(self.config["action_scale"])
        ctrl = self._ctrl_low + 0.5 * (action + 1.0) * (self._ctrl_high - self._ctrl_low)
        data = state.data.replace(ctrl=ctrl)

        def body_step(d, _):
            return mjx.step(self._mjx_model, d), None

        data, _ = jax.lax.scan(body_step, data, None, self._n_substeps)

        # SECTION: observations
        info = dict(state.info)
        info["steps"] = info["steps"] + 1
        obs = self._get_obs(data, info)

        # SECTION: rewards
        distance = self._distance(data, info)
        success = (distance < float(self.config["success_distance"])).astype(jp.float32)
        reward = (
            -float(self.config["reward_distance_scale"]) * distance
            + float(self.config["reward_success"]) * success
        )

        # SECTION: termination
        done = jp.where(
            (info["steps"] >= int(self.config["episode_length"])) | (success > 0),
            1.0,
            0.0,
        )
        metrics = {"reward": reward, "distance": distance, "success": success}
        return State(data=data, obs=obs, reward=reward, done=done, metrics=metrics, info=info)

    def _ee_pos(self, data: mjx.Data) -> jax.Array:
        return data.site_xpos[self._ee_site_id]

    def _distance(self, data: mjx.Data, info: dict[str, Any]) -> jax.Array:
        return jp.linalg.norm(self._ee_pos(data) - info["target_pos"])

    def _get_obs(self, data: mjx.Data, info: dict[str, Any]) -> dict[str, jax.Array]:
        ee = self._ee_pos(data)
        target = info["target_pos"]
        delta = ee - target
        distance = jp.linalg.norm(delta).reshape(())
        state = jp.concatenate([data.qpos, data.qvel, delta])
        privileged = jp.concatenate([state, ee, distance.reshape((1,))])
        return {"state": state, "privileged_state": privileged}
