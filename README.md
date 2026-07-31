# SimRig

SimRig is the agent- and human-friendly way to inspect, smoke-test, train, eval,
and preview MuJoCo Playground policies.

It is deliberately honest about the line between a robot model and a trainable
task:

- A MuJoCo or Menagerie model can be inspected, compiled, and stepped.
- A trainable environment must also define reset logic, observations, actions,
  rewards, and termination.
- Existing MuJoCo Playground environments are the first-class trainable path in
  v0.1.

## Ten-minute quickstart

Install the Playground training stack, then run a short smoke training loop:

```bash
python -m pip install -e ".[playground]"

simrig list-envs --backend mujoco-playground
simrig inspect-env Go1JoystickFlatTerrain
simrig smoke Go1JoystickFlatTerrain --steps 10
simrig train Go1JoystickFlatTerrain --preset smoke
```

Evaluate and open a browser preview of the saved policy (path will match your
`runs/` output):

```bash
simrig eval runs/<run-dir>/policy.params \
  --env Go1JoystickFlatTerrain \
  --seed 0 \
  --command 0.5 0.0 0.0
simrig preview runs/<run-dir>/policy.params \
  --env Go1JoystickFlatTerrain \
  --command 0.5 0.0 0.0 \
  --port 8765
```

Open `http://127.0.0.1:8765/`. Drag to orbit and scroll to zoom.

`G1JoystickFlatTerrain` works the same way if you prefer Unitree G1.

Use `smoke` before `local` or `cloud`. Longer presets need more compute.

## Install

For local package development:

```bash
python -m pip install -e .
```

For model inspection only:

```bash
python -m pip install -e ".[mujoco]"
```

For Playground training/eval:

```bash
python -m pip install -e ".[playground]"
```

For policies stored on Hugging Face Hub:

```bash
python -m pip install -e ".[hf]"
```

Dev extras (tests):

```bash
python -m pip install -e ".[dev]"
```

## For coding agents

Prefer browser commands (`preview`, `view-model`) over the native desktop
viewer (`demo`). Agents can launch a localhost URL, read `/status.json` or
`/joints.json`, and you can inspect the render in a normal browser tab.

Follow:

- [AGENTS.md](AGENTS.md) — safety rules and command preferences
- [docs/agent_workflow.md](docs/agent_workflow.md) — step-by-step agent workflow
- [skills/simrig/SKILL.md](skills/simrig/SKILL.md) — portable skill for Claude Code / Codex / Cursor

### Install the skill globally (any folder)

Anyone can use SimRig with Claude Code / Codex / Cursor **without** opening this
repo. Full steps: [skills/README.md](skills/README.md).

```bash
pip install -e ".[playground]"   # put `simrig` on PATH; PyPI later

cp -R skills/simrig ~/.claude/skills/simrig   # Claude Code
cp -R skills/simrig ~/.agents/skills/simrig   # Codex
cp -R skills/simrig ~/.cursor/skills/simrig   # Cursor
```

Restart the agent session, then ask it to “use the simrig skill”.

## Model inspection

```bash
simrig init
simrig list-models --menagerie ~/path/to/mujoco_menagerie
simrig inspect-model unitree_g1 --save-report
simrig view-model unitree_go1 --menagerie ~/path/to/mujoco_menagerie
```

Use `MUJOCO_MENAGERIE_PATH` or `--menagerie` to point SimRig at a Menagerie
checkout. Compilation and short stepping do **not** mean a model is trainable.

## Demo and Hugging Face policies

Native desktop MuJoCo viewer (humans):

```bash
simrig demo /path/to/policy.params \
  --env Go1JoystickFlatTerrain \
  --command 0.0 0.0 0.0
```

`--command` applies only for envs that expose command-like state (joystick
locomotion). Not every Playground task has a high-level command.

Headless evaluation accepts `--seed` for reproducible rollouts and `--command`
for fixed-command tasks:

```bash
simrig eval /path/to/policy.params \
  --env Go1JoystickFlatTerrain \
  --seed 4 \
  --command 0.8 0.0 0.0
```

SimRig fails clearly if `--command` is used with an environment that does not
expose command-like state. Run multiple seeds separately when reporting task
performance.

Load policies from Hugging Face with `hf://owner/repo/path`:

```bash
simrig eval hf://my-org/go1-policy/policy.params \
  --env Go1JoystickFlatTerrain
```

Private repos: `huggingface-cli login`, `HF_TOKEN` / `HUGGING_FACE_HUB_TOKEN`,
or `--hf-token`. Pin with `--hf-revision`.

Browser preview defaults to `--render-mode mujoco`. Use `--render-mode topdown`
only for the lightweight schematic debug view.

## Custom environments

Raw Menagerie models are not automatically trainable. Scaffold a module, fill in
the task logic, then smoke/train with the **`.py` path**:

```bash
simrig inspect-model path/to/scene.xml
simrig new-env my_robot_reach --model path/to/scene.xml --template mjx
simrig validate-env envs/my_robot_reach.py
simrig validate-env envs/my_robot_reach.py --runtime
simrig smoke envs/my_robot_reach.py --steps 10
simrig train envs/my_robot_reach.py --preset smoke
```

The generated file is an editable starter. You still need to define reset,
observations, rewards, termination, and action mapping. Prefer observation keys
`state` and `privileged_state` for SimRig's PPO defaults.

- `validate-env` (static): structure checklist only
- `validate-env --runtime`: import + construct + reset/step when JAX is available
- Passing static validation does **not** mean the env is trainable

### Test-drive example

A tiny 2-DOF arm + reach task lives under [`examples/`](examples/) so you can
exercise the custom-env CLI without bringing your own model:

```bash
pip install -e ".[playground]"
simrig validate-env examples/demo_reach.py --runtime
simrig smoke examples/demo_reach.py --steps 10
simrig train examples/demo_reach.py --preset smoke
```

See [`examples/README.md`](examples/README.md).

## Outputs

SimRig writes local project artifacts by default:

- `reports/` for model/env/eval reports
- `runs/` for training run configs, checkpoints, and policy params
- `artifacts/` for user-managed outputs
- `envs/` for editable custom env templates
- `configs/` for user-managed configs

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). License: [MIT](LICENSE).
