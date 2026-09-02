"""Cycle a trained orbit-arm policy through the same 30 goals as the IK demo."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import mujoco
import numpy as np

from simrig import LiveWebViewer
from simrig.rollout import PolicyRuntime


ENV_PATH = Path(__file__).resolve().parent / "orbit_reach.py"
if str(ENV_PATH.parent) not in sys.path:
    sys.path.insert(0, str(ENV_PATH.parent))
from targets import seeded_targets


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--target-seed", type=int, default=0)
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument(
        "--allow-runtime-mismatch",
        action="store_true",
        help="Allow a qualitative preview when this machine differs from the training runtime.",
    )
    args = parser.parse_args()

    runtime = PolicyRuntime(
        args.policy,
        env_name=str(ENV_PATH),
        allow_runtime_mismatch=args.allow_runtime_mismatch,
    )
    state, rng = runtime.reset(args.seed)
    targets = seeded_targets(args.target_seed)
    target_index = 0
    state = runtime.env.set_target(state, targets[target_index], reset_counters=True)
    native = mujoco.MjData(runtime.env.mj_model)

    def copy_state() -> None:
        native.qpos[:] = np.asarray(state.data.qpos)
        native.qvel[:] = np.asarray(state.data.qvel)
        native.mocap_pos[:] = np.asarray(state.data.mocap_pos)
        mujoco.mj_forward(runtime.env.mj_model, native)

    copy_state()
    viewer = LiveWebViewer(
        runtime.env.mj_model,
        native,
        name="Learned 360-degree reaching",
        port=args.port,
        tracking_site="ee_site",
        allow_reset=True,
    )
    try:
        with viewer:
            viewer.wait_for_client(timeout=15)
            while True:
                if viewer.consume_reset_request():
                    state, rng = runtime.reset(args.seed)
                    target_index = 0
                    state = runtime.env.set_target(
                        state, targets[target_index], reset_counters=True
                    )
                state, rng, _ = runtime.advance(state, rng)
                reached = bool(state.metrics["success"])
                terminated = bool(state.done)
                if reached:
                    time.sleep(0.3)
                    target_index = (target_index + 1) % len(targets)
                    state = runtime.env.set_target(
                        state, targets[target_index], reset_counters=True
                    )
                elif terminated:
                    # A collision or timeout is not a success, but it should not
                    # kill a persistent demonstration. Keep the same target and
                    # start a fresh randomized arm state.
                    rng, reset_rng = runtime.jax.random.split(rng)
                    state = runtime.reset_key(reset_rng)
                    state = runtime.env.set_target(
                        state, targets[target_index], reset_counters=True
                    )
                with viewer.lock:
                    copy_state()
                    viewer.sync(
                        target_error_m=float(state.metrics["distance"]),
                        reached=reached,
                        target_index=target_index + 1,
                        target_count=len(targets),
                        phase=(
                            "next target" if reached
                            else "retrying target" if terminated
                            else "learned policy"
                        ),
                    )
                time.sleep(1.0 / max(args.fps, 1))
    except KeyboardInterrupt:
        print("\nPolicy preview stopped.")
    finally:
        runtime.close()


if __name__ == "__main__":
    main()
