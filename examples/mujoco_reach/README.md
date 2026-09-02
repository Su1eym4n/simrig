# Evaluate a controller in MuJoCo

Run two ordinary Python controllers on the included two-joint arm. SimRig
freezes the task, runs the evaluator across fixed cases, derives success from
measured outcomes, and compares reports. No GPU, JAX, or training is needed.

## What this example measures

The evaluator loads [`simple_arm.xml`](../models/simple_arm.xml), resets joint
positions from a seed, applies controller outputs to MuJoCo position actuators,
and calls `mj_step`. It measures world-frame distance from `ee_site` to the
target after each 20 ms control tick (ten 2 ms physics steps).

Success uses the existing [`demo_reach.py`](../demo_reach.py) criterion: reach
within **5 cm at any tick**, within 200 ticks / 4 simulated seconds. The rollout
ends on arrival. Any MuJoCo warning fails the simulation-validity predicate.
The two targets and six seeds are declared in [`task.json`](task.json).

| Controller | What it actually does |
|---|---|
| [`controllers/ik.py`](controllers/ik.py) | Computes an elbow-up inverse-kinematics solution and commands joint positions. Expected to pass the six cases. |
| [`controllers/zero.py`](controllers/zero.py) | Outputs zero normalized action: actuator midpoint positions, **not zero torque**. Expected to fail the suite. |

These files are scripted controllers, not learned weights. Both go through the
same simulator and measurements. The evaluator reports raw distances, timing,
warnings, and target-arrival events; it does not invent reward or success labels.
SimRig's contract predicates decide success.

## Run from the repository root

```bash
python -m pip install -e ".[mujoco]"
simrig inspect-model examples/models/simple_arm.xml
simrig task validate examples/mujoco_reach/task.json

# Read the definition above before freezing this supplied example task.
simrig task freeze examples/mujoco_reach/task.json \
  --output /tmp/mujoco-reach.frozen.json
simrig eval-suite examples/mujoco_reach/controllers/ik.py \
  --contract /tmp/mujoco-reach.frozen.json --suite promotion \
  --output /tmp/reach-ik.json
```

Run the negative control separately. **Exit status 1 is expected**: the suite
fails, but its report is still written.

```bash
simrig eval-suite examples/mujoco_reach/controllers/zero.py \
  --contract /tmp/mujoco-reach.frozen.json --suite promotion \
  --output /tmp/reach-zero.json
```

Compare both reports:

```bash
simrig rank-checkpoints /tmp/reach-ik.json /tmp/reach-zero.json
```

The CLI calls the input an artifact/checkpoint, but the evaluator owns loading
it. Here it loads trusted Python controller code. A Brax `policy.params` file
cannot be substituted without a loader that reconstructs that policy and its
observations/action mapping. Other environments likewise need their own
evaluator; changing `environment.ref` does not add support.

## Watch the interactive showcase

```bash
python examples/mujoco_reach/evaluator.py --preview
python examples/mujoco_reach/evaluator.py --controller examples/mujoco_reach/controllers/zero.py --preview
```

Open the printed localhost URL. The script selects an available local port,
waits up to 15 seconds for a browser, and runs a preview-only arm with base yaw,
shoulder, and elbow joints. Its shoulder
is mounted directly on the base cylinder. The route contains 30 seeded random
positions at different radii, heights, and angles around every side of the
robot, but only the current red target is shown. Preview mode rate-limits the
controller's joint-position commands and briefly holds each arrival so the
motion is easy to inspect. **Reset Simulation** restores the centered initial
arm state, clears the trail, and restarts at target 1 at any time. Stop the
persistent preview with `Ctrl+C`.

The positive controller is **scripted inverse kinematics, not a trained
policy**. The preview uses `LiveWebViewer`; it shows live simulation, not
recorded trajectory playback. Its Three.js assets load from a CDN. Omit
`--preview` for the original two-joint, single-target headless acceptance
measurement; that evaluator and model are unchanged.

## Watch the learned 360-degree policy

The separate orbit-arm task is a real goal-conditioned PPO policy. It was
calibrated on six independently reset targets using the stricter five-tick
arrival hold, collision checks, and native MuJoCo measurements. Unlike the IK
showcase, it is learned weights rather than a planner.

Train a checkpoint on a CUDA-capable machine first:

```bash
simrig validate-env examples/mujoco_reach/orbit_reach.py --runtime
simrig smoke examples/mujoco_reach/orbit_reach.py --steps 10
simrig train examples/mujoco_reach/orbit_reach.py --preset large \
  --output runs/orbit-reach
```

For an SSH GPU host, replace the last command with `simrig remote train` after
`simrig remote prepare`; fetch the completed run into `runs/` before previewing.

```bash
python examples/mujoco_reach/policy_preview.py \
  --policy runs/remote-trained-v3c/policy.params \
  --allow-runtime-mismatch
```

Open the printed localhost URL. It moves through the same 30 target positions,
shows only the next red target, and has **Reset Simulation**. This checkpoint
was trained on the remote Linux GPU runtime, so `--allow-runtime-mismatch` is
required on this Mac and makes the preview qualitative rather than a formal
reproduction.

## Limits and adapting it

This checks arrival, **not sustained holding, collision safety, robustness, or
hardware readiness**. A fleeting target crossing can pass. Some reset states
may already be close to a target; only post-step measurements are scored.
The reset range matches the MJX reaching task, but RNG samples and this fixed
target suite differ from its training distribution. There is no reward in this
evaluator, so this is not a reward-hacking experiment.

To adapt it, define the new task's measurements and scenarios, implement the
simulator/policy loading in [`evaluator.py`](evaluator.py), test appropriate
positive and negative controls, then freeze that reviewed definition. Add
holding/contact checks if those are required by your task. Plugins and controller
code are trusted local Python and are not sandboxed. The controller receives
`model` and `data` for reading and must not modify either.
