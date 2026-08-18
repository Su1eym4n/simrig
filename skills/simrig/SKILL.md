---
name: simrig
description: Train, evaluate, and visualize MuJoCo and MuJoCo Playground robots with the SimRig CLI. Use for arbitrary MJCF/XML or Menagerie robots, existing Playground tasks, custom locomotion or manipulation tasks, crouching, jumping, reaching, custom MuJoCo scenes, editable MJX environment modules, Brax PPO training, checkpoint evaluation, browser previews, and native MuJoCo demos.
---

# SimRig

Use `simrig` as the control plane for model inspection, environment validation,
PPO training, evaluation, and visualization. Edit ordinary MJCF and Python when
a task or scene is custom; do not replace SimRig's runners or viewers.

## Establish the artifact and intent

Classify the input before acting:

| Input or request | Route |
|---|---|
| Playground environment name | Existing environment |
| Raw `.xml`, MJCF, or Menagerie robot | Inspect model, then match or build a task |
| Custom environment path ending in `.py` | Validate environment |
| `policy.params` or `hf://...` checkpoint | Evaluate, then preview |
| “Show this robot” | View model only |
| Running MuJoCo Python controller | Add `LiveWebViewer` around its existing loop |
| “Train this robot to …” | Define the task, then use an existing or custom environment |
| Custom terrain, props, targets, or contacts | Custom scene plus custom environment |

Do not treat a robot model as a task. A trainable task also requires action
mapping, observations, reset distribution, rewards, termination, and measurable
success criteria.

## Check the installation

Run:

```bash
command -v simrig
simrig --help
```

When working in the SimRig source repository and the command is unavailable,
install the relevant extra:

```bash
python3 -m pip install -e ".[mujoco]"       # inspect and view models
python3 -m pip install -e ".[playground]"   # validate, train, eval, preview
python3 -m pip install -e ".[hf]"           # resolve hf:// checkpoints
```

Request approval before installing dependencies. Resolve Menagerie with
`MUJOCO_MENAGERIE_PATH` or `--menagerie`; never recursively search a home
directory for meshes.

## Follow the training pipeline

### 1. Inspect before designing

For raw models, run:

```bash
simrig inspect-model MODEL_OR_XML --save-report
simrig view-model MODEL_OR_XML --port 8766
```

Report compilation and bounded-step results separately. Neither proves the
model is trainable. Use the browser URL `http://127.0.0.1:8766/` to review
joints, visual geometry, and the authored pose. `view-model` defaults to a
Three.js/WebGL scene with browser-local orbit, zoom, and pan controls. It loads
pinned Three.js modules from jsDelivr. Use `--render-mode mujoco` for the local
streamed renderer or `--render-mode topdown` for the schematic fallback. When
the MJCF contains keyframes, the viewer starts from the first one and Reset
Joints restores it. When the MJCF contains named cameras, the Three.js page
adds a selectable Robot View inset. Emulated mode uses the camera's compiled
live pose and vertical FOV in the Three.js scene; Sensor mode shows the native
MuJoCo offscreen render. Use `--camera NAME` to choose the initial camera. Treat
the emulation as a human-facing aid, never as evidence of exact policy pixels.

For an ordinary Python controller that already owns its `MjModel`, `MjData`,
controls, and stepping, use `simrig.LiveWebViewer` inside that script. Share
the viewer's lock around MuJoCo state mutations and call `sync()` after each
completed step. Do not precompute or replay a trajectory just to visualize it.

For a known environment, run:

```bash
simrig inspect-env ENV_NAME --save-report
```

### 2. Define a testable task

Translate the user's request into a short task contract before authoring reward
logic. Specify:

- desired behavior and command or target distribution;
- initial-state and scene randomization;
- allowed contacts and failure conditions;
- episode horizon;
- measurable evaluation scenarios and pass criteria.

Ask one focused question only when a missing choice materially changes the
task. Continue safe inspection while waiting. Do not invent task semantics from
the robot or task name. Read [task-design.md](references/task-design.md) for
locomotion, posture, jumping, manipulation, and scene-specific decisions.

### 3. Prefer an existing Playground task

Check available environments before writing a custom one:

```bash
simrig list-envs --backend mujoco-playground
simrig inspect-env ENV_NAME
simrig smoke ENV_NAME --steps 10
simrig train ENV_NAME --preset smoke --impl auto --seed 0
```

Use an existing environment only when its robot, task semantics, actions, and
scene match the contract. Do not choose one solely because the robot name
matches. Distinguish training over a command distribution from training only at
one requested command. Registered Playground tasks use their upstream tuned PPO
and network configuration plus their declared domain randomizer. `--impl auto`
uses the upstream implementation when supported and falls back from Warp to JAX
when no JAX GPU is visible. Disable randomization only for an explicit baseline.

### 4. Build a custom scene or task

Create or edit a dedicated scene XML when the task needs terrain, targets,
objects, obstacles, contact pairs, sensors, cameras, or task-specific
keyframes. Preserve relative includes and mesh paths. Inspect and view the
result before implementing the environment.

Scaffold only after the task contract is known:

```bash
simrig new-env TASK_NAME --model path/to/scene.xml --template mjx
```

Fill every `SECTION:` block in the generated module. Keep reward terms and task
assumptions readable in Python. Prefer observations with `state` and
`privileged_state`, and keep metric names stable, including `reward`. Read
[custom-environments.md](references/custom-environments.md) before editing a
custom module.

### 5. Validate in increasing order of cost

Run every gate and stop at the first failure:

```bash
simrig validate-env envs/TASK_NAME.py
simrig validate-env envs/TASK_NAME.py --runtime
simrig smoke envs/TASK_NAME.py --steps 10
simrig train envs/TASK_NAME.py --preset smoke --seed 0
```

Treat static validation as a structure check only. Treat runtime validation as
one reset/step compatibility check. Treat smoke training as pipeline evidence,
not proof that the behavior is learned.

For a pixel-observation environment, require literal `NETWORK_SPEC`,
`VISION_SPEC`, and `DEFAULT_CONFIG` mappings for import-free static checks.
Runtime `network_spec()` and `vision_spec()` hooks may enrich those declarations
after the training dependencies are available. Then add the vision gate:

```bash
simrig validate-env envs/TASK_NAME.py --vision
simrig validate-env envs/TASK_NAME.py --runtime --vision
simrig smoke envs/TASK_NAME.py --steps 5
simrig train envs/TASK_NAME.py --preset smoke
```

`--vision` checks the CNN type, declared pixel/camera contract, runtime HWC
shapes, numeric and finite values, declared range and resolution, frame change,
and actor/critic observation keys. MJX camera rendering currently requires a
JAX-visible CUDA GPU with MuJoCo Warp. A CPU-only metadata pass is not evidence
that rendered PPO can run.

Inspect state dimensions, finite values, action scaling, reward components,
termination frequency, contacts, and reset diversity before spending on a
longer run.

### 6. Scale training deliberately

Use `smoke` first. Use `local` only after smoke training produces a checkpoint
and sane metrics:

```bash
simrig train ENV_OR_PATH --preset local
```

Treat `cloud` as a large configuration, not as a remote job launcher. Do not run
it without an explicit compute destination and user approval. Use
`--timesteps`, `--num-envs`, and `--batch-size` for intentional overrides.
Record the exact run directory and preserve its `config.json`,
`final_metrics.json`, checkpoints, and `policy.params`.
Every new run should record the resolved implementation, seed, network,
randomizer, source hashes, Git state, device inventory, and runtime versions.

For an already-provisioned Lambda On-Demand Cloud GPU, read
[lambda-cloud.md](references/lambda-cloud.md). Use `simrig cloud lambda connect`
for the first interactive SSH connection, then `check`, `prepare`, remote
`smoke`, and remote `train --preset smoke` before a detached `cloud` run. SimRig
does not provision or terminate the billable instance. Fetch the complete run
directory and remind the user to terminate the instance after artifacts are
safe.

### 7. Evaluate behavior, not only reward

Run headless evaluation first:

```bash
simrig eval runs/RUN/policy.params \
  --env ENV_OR_PATH \
  --steps 500 \
  --seed 0
```

Then preview the same checkpoint and environment:

```bash
simrig preview runs/RUN/policy.params --env ENV_OR_PATH --port 8765
```

Use `--auto-reset` to begin a new episode after termination while preserving
the terminal pose briefly; adjust that pause with `--auto-reset-delay`.

Use `--command X Y YAW` with `eval` and `preview` only for environments that
expose command-like state. Repeat headless evaluation with distinct `--seed`
values when the task contract requires multiple trials. Open
`http://127.0.0.1:8765/`. Preview defaults to a Three.js/WebGL scene driven by
live geom and authored-camera transforms from a server-side rollout clock, so
browser-local orbit, zoom, and pan do not interrupt policy stepping. Named
MuJoCo cameras appear in a Robot View inset while the human orbit camera remains
independent. Emulated mode shares the Three.js design; Sensor mode is the native
MuJoCo comparison. Actual policy observations remain environment-defined.
The preview reports episode number and survival length, and distinguishes
simulator physics, policy-observation, Sensor-display, and browser-playback
rates. Command controls appear only for environments that expose commands;
custom environments may name arbitrary command fields with an instance
`command_spec()` method.
Use `--render-mode mujoco` for the local full-page stream or
`--render-mode topdown` for the schematic fallback. Prefer `preview` for
agent-visible review; use `demo` only when the user explicitly wants the native
desktop viewer.

Evaluate across multiple seeds and the scenarios from the task contract.
Compare success rate and task-specific metrics, not just total reward. Read
[evaluation-and-operations.md](references/evaluation-and-operations.md) for
checkpoint compatibility, command-specific evaluation, run artifacts, and
failure triage.

## Enforce hard rules

- Never claim that an XML is trainable because it compiles or steps.
- Never invent a complete reward, observation, or termination design from a
  model name.
- Never start long training before runtime validation, environment smoke, and
  `--preset smoke` training pass.
- Never use a policy with a different environment or network architecture
  without explicit compatibility evidence.
- Never hide project-specific research logic inside generic SimRig code.
- Never assume Isaac Lab or another backend; SimRig v0 trains through
  `mujoco-playground`.
- Never report success from reward alone; connect evaluation to the user's task
  contract and visual behavior.

## Report the handoff

Return a compact evidence summary:

- model or environment used;
- task contract and scene assumptions;
- validation and smoke results;
- exact training command, preset, and run directory;
- checkpoint and evaluation metrics;
- preview URL when running;
- remaining limitation or next experiment.
