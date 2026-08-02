# SimRig

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
simrig train Go1JoystickFlatTerrain --preset smoke
```

Use the `smoke` preset before a longer `local` or `cloud` configuration.

### Custom robot or scene

Inspect the model before designing a task:

```bash
simrig inspect-model path/to/robot.xml --save-report
simrig view-model path/to/robot.xml --port 8766
```

Open `http://127.0.0.1:8766/` to inspect the compiled model, orbit/zoom/pan the
camera, and adjust named joints. The default `threejs` renderer sends MuJoCo's
visual meshes and primitives to a GPU-accelerated WebGL scene, so camera motion
stays smooth without streaming image frames from Python. If the MJCF defines a
keyframe, the viewer starts from its first authored pose and **Reset Joints**
restores it.

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
simrig train envs/my_task.py --preset smoke
```

The generated environment is a starter, not an invented task definition. The
reward, observations, actions, resets, and termination logic remain explicit
and editable in Python.

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
  --port 8765
```

Open `http://127.0.0.1:8765/` to orbit, zoom, pan, change commands, pause, and
inspect the live rollout. Preview uses the Three.js renderer by default: the
policy advances on a server-side rollout clock while lightweight MuJoCo geom
transforms update the browser scene. The camera follows the robot without
streaming rendered image frames. Use `--render-mode mujoco` for the older local
image-stream preview or `--render-mode topdown` for the schematic fallback.

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
  --preset smoke
```

Only after the environment and PPO smoke gates pass, start a detached large
run with `--preset cloud --detach`. SimRig operates on an instance you already
provisioned; it never launches or terminates billable Lambda resources. See the
complete [Lambda Cloud GPU guide](docs/lambda-cloud.md), including persistent
storage, monitoring, artifact download, and shutdown reminders.

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
