# Custom MJX environments

Use this reference when filling a module created by `simrig new-env`.

## Required interface

Define `MODEL_PATH`, `ENV_NAME`, and preferably:

```python
def default_config() -> dict[str, Any]: ...
def make_env(config_overrides: dict[str, Any] | None = None) -> CustomEnv: ...
```

Expose an environment with:

- `reset(rng)` and `step(state, action)`;
- `action_size` and `observation_size`;
- `mj_model` and `mjx_model` for rendering;
- `dt`, and preferably `sim_dt`;
- state fields `.data`, `.obs`, `.reward`, `.done`, `.metrics`, and `.info`.

Prefer `mujoco_playground._src.mjx_env.MjxEnv` when it fits. A standalone
JAX/MJX environment is acceptable when it implements the same practical
contract.

## Model loading

Resolve `MODEL_PATH` from the module location when portability matters. Load
the native MuJoCo model, set simulation timestep intentionally, convert it with
`mjx.put_model`, and cache stable joint, actuator, site, body, and sensor IDs.
Check that every required named element exists.

Verify actuator control ranges and semantics. Do not assume one action per joint
or symmetric `[-1, 1]` actuator limits.

## Reset

Create fresh MJX data, randomize only within physically plausible bounds, set
task targets and phase state, run `mjx.forward`, and build the first
observation. Split JAX RNG keys explicitly and carry the updated key in
`state.info`.

Reset distributions must cover the conditions expected during evaluation
without beginning in penetrations, unstable poses, or unreachable goals.

## Action mapping

Clip normalized policy actions and map them to the intended actuator controls.
Handle actuator limits, offsets, gains, control rate, and simulation substeps.
Track the previous action when using action-rate observations or penalties.

## Observations

Return:

```python
{"state": policy_observation, "privileged_state": value_observation}
```

Keep shapes static. Include only deployable information in `state`.
`privileged_state` may add simulation-only state for the PPO value function.
Avoid unbounded angles or poorly scaled features when a stable representation
is available. Update `observation_size` whenever features change.

### Pixel observations and vision PPO

For rendered observations, keep the ordinary environment interface and add
declarative module hooks:

```python
def network_spec():
    return {
        "type": "vision_cnn",
        "factory": {
            "policy_obs_key": "state",
            "value_obs_key": "privileged_state",
        },
    }

def vision_spec():
    return {
        "pixel_keys": ["pixels/view_0"],
        "camera_names": ["fixed"],
        "resolution": [64, 64],
        "frame_stack": 3,
        "channels_per_frame": 1,
        "value_range": [-0.5, 0.5],
        "requires_impl": "warp",
        "nworld_config_key": "vision_config.nworld",
    }
```

Every pixel observation must use a `pixels/` prefix and a static HWC shape.
The vision CNN consumes all `pixels/*` entries. `policy_obs_key` may add
deployable state to the actor; it must never point to `privileged_state`.
`value_obs_key` may expose simulator-only state to the critic. Keep renderer
world count synchronized with PPO's `num_envs`; SimRig does this through the
declared `nworld_config_key`.

MJX camera rendering is currently a CUDA/MuJoCo-Warp path. Make that constraint
explicit with `requires_impl`. Do not replace rendered pixels with state-derived
proxies and call the result vision training.

## Rewards and metrics

Compute named terms separately, combine them visibly, and expose diagnostic
terms in `state.metrics`. Include a stable `reward` key. Keep signs and units
obvious.

Check for reward loopholes:

- stillness satisfying a locomotion task;
- collapse satisfying a height or pose target;
- violent impacts increasing progress;
- termination avoiding future penalties;
- energy or smoothness penalties dominating the task.

Use JAX-compatible array operations in `reset` and `step`; avoid Python
branching on traced values.

## Termination

Separate failure, success, and time-limit semantics in metrics or `info`, even
if the environment exposes one `done` value. Bound episodes. Use measurable
conditions such as forbidden contacts, non-finite state, joint violations,
fallen orientation/height, sustained success, or timeout.

Avoid terminating so early that the policy cannot learn recovery unless
recovery is outside the task.

## Validation loop

Run:

```bash
simrig validate-env envs/TASK.py
simrig validate-env envs/TASK.py --runtime
simrig smoke envs/TASK.py --steps 10
simrig train envs/TASK.py --preset smoke
```

Do not remove `NOT TRAINABLE YET` until reset and step are implemented. Static
validation only checks structure. Runtime validation constructs the env and
executes reset/step. `smoke` JITs reset/step and advances zero actions. Smoke
training verifies that the PPO pipeline can produce artifacts.

For vision environments, insert `--vision` static and runtime checks before the
generic smoke. See `examples/vision_cartpole.py` for a complete small reference.

Before a longer run, inspect resets and short rollouts for non-finite values,
unintended contacts, immediate termination, saturated actions, constant
observations, and reward terms with implausible scale.
