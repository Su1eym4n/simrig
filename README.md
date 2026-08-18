# SimRig

<div align="center">

<img src="assets/g1-greeting.gif" alt="Unitree G1 waving hello in SimRig's interactive browser preview" width="100%">

<br>

<em>A pretrained Unitree G1 policy preview with a scripted greeting.</em>

</div>

Turn MuJoCo robots into trained policies with agent-guided task design, PPO
training, evaluation, and interactive previews.

SimRig combines:

- a Python CLI that inspects models, runs MuJoCo Playground environments,
  trains Brax PPO policies, evaluates checkpoints, and serves previews;
- an agent skill that teaches Codex, Claude Code, and Cursor how to use that
  pipeline safely.

Raw robot XML is not automatically a training task. SimRig helps the agent move
from a model and a requested behavior to explicit observations, actions,
rewards, resets, termination conditions, validation, training, and evaluation.

## From prompt to simulation

### Train Go1 through a smoke-gated cloud workflow

SimRig prepares the existing Go1 locomotion environment and runs local smoke
tests before requesting the user's Lambda Cloud details. The full cloud run
starts only after that handoff. The recorded result below shows the trained
checkpoint downloaded and running in SimRig's interactive browser preview.

<details>
<summary><strong>Prompt</strong></summary>

> Prepare the full Go1 robot training setup and run local smoke tests to verify
> everything works. Once the tests pass, ask me for my Lambda Cloud details
> before starting the full training run.

</details>

<img src="assets/go1-training.gif" alt="Codex running a smoke-gated Go1 training workflow and previewing the trained policy in SimRig" width="100%">

### Trace a five-pointed star with Franka Panda

This example uses a directly scripted Cartesian trajectory and inverse
kinematics—no reinforcement learning is needed. The controller keeps the
end-effector orientation fixed, moves smoothly between star vertices, and
publishes the running simulation and visible trace through SimRig's browser
viewer.

<details>
<summary><strong>Prompt</strong></summary>

> Can you configure a Franka Panda robotic arm in simulation so that its end
> effector traces a five-pointed star trajectory?
>
> Please:
>
> - Use the Franka Panda arm and gripper.
> - Move the end effector along a clear five-pointed star path in Cartesian
>   space.
> - Keep the end-effector orientation fixed and stable throughout the motion.
> - Use inverse kinematics, trajectory planning, or a direct scripted controller
>   rather than reinforcement learning unless training is genuinely necessary.
> - Add a visible trajectory trace, marker, or drawing surface so the completed
>   star can be verified.
> - Make the motion smooth, with controlled velocity and acceleration between
>   each star vertex.
> - Provide the complete runnable script and all required launch commands.
> - Explain how to modify the star size, star position, drawing plane,
>   end-effector height, motion speed, and number of repetitions.
> - State clearly whether the solution is directly scripted or trained, and
>   explain why that method is appropriate.
> - Prefer the simplest reliable direct-control implementation.

</details>

<img src="assets/franka-panda-star.gif" alt="Codex configuring a Franka Panda arm to trace a five-pointed star in SimRig's browser viewer" width="100%">

## What SimRig can do

| Goal | SimRig workflow |
|---|---|
| Train a known Playground robot | Inspect the environment, smoke-test it, train, evaluate, and preview |
| Use a custom MJCF/XML robot | Inspect the model, define the task, create an editable environment, then validate and train |
| Build locomotion or posture behaviors | Design command tracking, contacts, rewards, failures, and evaluation scenarios |
| Build custom scenes | Add terrain, props, targets, sensors, cameras, or contact rules in ordinary MJCF and Python |
| Evaluate an existing policy | Run reproducible headless rollouts and open a browser or native MuJoCo preview |

SimRig v0 uses MuJoCo and MuJoCo Playground. Isaac Lab is not currently a
supported backend.

## Installation

SimRig requires Python 3.10 or newer. Install the Playground training stack
from PyPI:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install "simrig[playground]"

simrig --version
```

For an editable source installation, clone the repository and install from its
root instead:

```bash
git clone https://github.com/Su1eym4n/simrig.git
cd simrig
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[playground]"
```

Development dependencies, tests, and contribution checks are documented in
[CONTRIBUTING.md](CONTRIBUTING.md).

## Install the agent skill

Inside this repository, Codex discovers the SimRig skill automatically through
`.agents/skills/simrig`.

To make the skill available from any project, install it globally with the
Skills CLI:

```bash
npx skills add Su1eym4n/simrig --skill simrig --global
```

The installer supports Codex, Claude Code, Cursor, and other agents. Restart the
agent after installation. See
[Agent skill installation](docs/skill-installation.md) for provider-specific
commands, manual installation, and troubleshooting.

## Use SimRig

Open a project containing a MuJoCo robot or scene and ask the agent naturally:

> Train this MuJoCo robot to walk forward.

> Create a crouching task for this robot, smoke-test it, and start a small
> training run.

> Evaluate this checkpoint across five seeds and preview the policy.

You can invoke the workflow explicitly with `$simrig`, but the skill can also
activate automatically when the request matches its description.

### Existing Playground environment

```bash
simrig list-envs --backend mujoco-playground
simrig inspect-env Go1JoystickFlatTerrain
simrig smoke Go1JoystickFlatTerrain --steps 10
simrig train Go1JoystickFlatTerrain --preset smoke --impl auto --seed 0
```

Use the `smoke` preset before a longer `local` or `cloud` configuration.
For registered Playground environments, SimRig starts from the environment's
tuned Brax PPO/network configuration and declared domain randomizer, then
bounds the expensive dimensions for `smoke` or `local`. The `cloud` preset uses
the full upstream task configuration unless you pass explicit overrides.

`--impl auto` uses the environment's default implementation when supported. It
selects MuJoCo Warp on a JAX-visible GPU when the environment defaults to Warp,
and falls back to JAX on CPU-only hosts. Use `--impl jax` or `--impl warp` to
make the choice explicit. Disable an available randomizer only for a deliberate
baseline with `--no-domain-randomization`.

### Custom robot or scene

Inspect the model before designing a task:

```bash
simrig inspect-model path/to/robot.xml --save-report
simrig view-model path/to/robot.xml --port 8766
```

Open `http://127.0.0.1:8766/` to inspect the compiled model, orbit/zoom/pan the
camera, and adjust named joints. The default `threejs` renderer sends MuJoCo's
visual meshes and primitives to a GPU-accelerated WebGL scene, so camera motion
stays smooth. When the MJCF contains named cameras, the page also shows a
**Robot View** inset. Its default **Emulated** mode uses a second Three.js camera
with the compiled MuJoCo camera's live world pose and vertical field of view,
giving it the same visual design as the orbit view. Switch to **Sensor** for the
native MuJoCo offscreen image, use the camera dropdown to switch cameras, or
choose the initial camera with `--camera NAME`. The inset follows joint-slider
changes while the main Three.js camera remains freely movable. The emulation is
for human inspection; vision policies train on MJX/Warp pixels, not this WebGL
render. If the MJCF defines a keyframe, the viewer starts from its first
authored pose and **Reset Joints** restores it.

The Three.js modules are pinned and loaded from jsDelivr, so the default viewer
needs an internet connection when the page first loads. For an entirely local
MuJoCo-rendered image stream, use:

```bash
simrig view-model path/to/robot.xml --render-mode mujoco --port 8766
```

Use `--render-mode topdown` only for the schematic debugging fallback.

### View a running MuJoCo script

Standalone controllers can publish the `MjModel` and `MjData` they already own
to the same Three.js viewer. SimRig does not step, pause, or replay the script:

```python
from simrig import LiveWebViewer

with LiveWebViewer(
    model,
    data,
    name="my controller",
    tracking_body="end_effector",
) as web:
    while running:
        with web.lock:
            data.ctrl[:] = controller(data)
            mujoco.mj_step(model, data)
            web.sync(phase="moving")
```

Open the printed `http://127.0.0.1:8767/` URL. The page receives lightweight
geom transforms while the Python script retains full control of simulation
timing and state. A named `tracking_body` also draws its live path. Use
`wait_for_client()` when motion should begin only after the page is ready.

After defining the task, scaffold and validate an editable environment:

```bash
simrig new-env my_task --model path/to/scene.xml --template mjx
simrig validate-env envs/my_task.py
simrig validate-env envs/my_task.py --runtime
simrig smoke envs/my_task.py --steps 10
simrig train envs/my_task.py --preset smoke --seed 0
```

The generated environment is a starter, not an invented task definition. The
reward, observations, actions, resets, and termination logic remain explicit
and editable in Python.

### Train from rendered pixels

Custom environments can declare a Brax vision CNN instead of the legacy MLP.
The included cartpole example renders real 64x64 MuJoCo frames, stacks three
grayscale frames, feeds pixels plus the previous action to the actor, and gives
the critic additional simulator state:

```bash
# CPU-safe metadata check
simrig validate-env examples/vision_cartpole.py --vision

# These require a JAX-visible CUDA GPU and MuJoCo Warp.
simrig validate-env examples/vision_cartpole.py --runtime --vision
simrig smoke examples/vision_cartpole.py --steps 5
simrig train examples/vision_cartpole.py --preset smoke \
  --output runs/vision-cartpole-smoke
```

A vision module declares `network_spec()`, `vision_spec()`, and optionally
`training_config()`. SimRig persists the selected `network_type` and complete
network factory in `config.json`, then reconstructs the same CNN for `eval`,
`demo`, and `preview`. Checkpoints created before vision support remain MLP by
default.

#### Run the pretrained vision reference

The published reference policy was trained for 5,079,040 PPO steps with a
1,000-step episode horizon. It completed all 1,000 requested steps without
termination for evaluation seeds 0 through 4. Install both optional extras so
SimRig can run the Playground environment and resolve the Hub artifact:

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -e ".[playground,hf]"

.venv/bin/simrig eval \
  hf://ssuleiman/simrig-vision-cartpole/policy.params \
  --env examples/vision_cartpole.py \
  --hf-revision v1 \
  --steps 1000 \
  --seed 0

.venv/bin/simrig preview \
  hf://ssuleiman/simrig-vision-cartpole/policy.params \
  --env examples/vision_cartpole.py \
  --hf-revision v1 \
  --auto-reset \
  --port 8765
```

Both commands require a JAX-visible CUDA GPU and MuJoCo Warp. Hub resolution
downloads `policy.params` with its sibling `config.json` so SimRig can rebuild
the recorded vision CNN. Exact evaluation should use the recorded Python and
package versions. `--allow-runtime-mismatch` is only for an explicitly
qualitative preview on a different compatible runtime. The checkpoint,
training configuration, metrics, environment snapshot, and five-seed report
are published at
[ssuleiman/simrig-vision-cartpole](https://huggingface.co/ssuleiman/simrig-vision-cartpole).

### Evaluate and preview

```bash
simrig eval runs/<run-dir>/policy.params \
  --env Go1JoystickFlatTerrain \
  --steps 500 \
  --seed 0 \
  --command 0.5 0.0 0.0

simrig preview runs/<run-dir>/policy.params \
  --env Go1JoystickFlatTerrain \
  --command 0.5 0.0 0.0 \
  --auto-reset \
  --port 8765
```

Open `http://127.0.0.1:8765/` to orbit, zoom, pan, change supported commands,
toggle automatic episode reset, pause, and inspect the live rollout. Preview
uses the Three.js renderer by default: the
policy advances on a server-side rollout clock while lightweight MuJoCo geom
transforms update the browser scene. The camera follows the robot without
coupling policy stepping to display rendering. Environments with named MuJoCo
cameras also get a **Robot View** inset, selectable without moving the human
orbit camera. **Emulated** renders the live authored camera pose and FOV in the
same Three.js scene; **Sensor** shows the native MuJoCo offscreen image for
comparison. Neither changes policy input: training and rollout inference keep
using the environment's configured observation pipeline. Use
`--render-mode mujoco` for the older full-page local image stream or
`--render-mode topdown` for the schematic fallback.

### Train on a Lambda Cloud GPU

After launching a Lambda On-Demand instance with an SSH key, SimRig can connect,
sync this checkout, verify JAX GPU visibility, train, monitor a detached run,
and download its artifacts:

```bash
simrig cloud lambda connect INSTANCE_IP --identity ~/Downloads/lambda-key.pem
simrig cloud lambda prepare INSTANCE_IP --identity ~/Downloads/lambda-key.pem
simrig cloud lambda smoke INSTANCE_IP Go1JoystickFlatTerrain \
  --identity ~/Downloads/lambda-key.pem
simrig cloud lambda train INSTANCE_IP Go1JoystickFlatTerrain \
  --identity ~/Downloads/lambda-key.pem \
  --preset smoke \
  --impl auto \
  --seed 0
```

Only after the environment and PPO smoke gates pass, start a detached large
run with `--preset cloud --detach`. SimRig operates on an instance you already
provisioned; it never launches or terminates billable Lambda resources. See the
complete [Lambda Cloud GPU guide](docs/lambda-cloud.md), including persistent
storage, monitoring, artifact download, and shutdown reminders.

Lambda preparation requires Python 3.11+ and installs a pinned Playground
training stack. Every run records its resolved PPO/network configuration,
implementation, seed, randomizer, source hashes, Git state, JAX devices,
precision-related environment, and package versions. Checkpoint eval, demo, and
preview reconstruct the recorded implementation and network, and reject a
different runtime unless `--allow-runtime-mismatch` is explicitly selected for
qualitative review.

## Documentation

- [Examples](examples/README.md)
- [Agent workflow](docs/agent_workflow.md)
- [Lambda Cloud GPU training](docs/lambda-cloud.md)
- [Agent skill installation](docs/skill-installation.md)
- [Contributing](CONTRIBUTING.md)
- [SimRig skill source](skills/simrig/SKILL.md)

## Outputs

SimRig writes project-local artifacts:

- `reports/` — model, environment, and evaluation reports
- `runs/` — training configuration, metrics, checkpoints, and policy parameters
- `envs/` — editable custom environment modules
- `artifacts/` and `configs/` — user-managed outputs and configuration

## License

[MIT](LICENSE)
