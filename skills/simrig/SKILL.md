---
name: simrig
description: Train, evaluate, and visualize MuJoCo and MuJoCo Playground robots with the SimRig CLI. Use for arbitrary MJCF/XML or Menagerie robots, existing Playground tasks, custom locomotion or manipulation tasks, crouching, jumping, reaching, custom MuJoCo scenes, editable MJX environment modules, Brax PPO training, checkpoint evaluation, browser previews, and native MuJoCo demos.
---

# SimRig

Use `simrig` as the control plane for model inspection, environment validation,
PPO training, evaluation, and visualization. Edit ordinary MJCF and Python when
a task or scene is custom. Prefer the built-in viewers for ordinary inspection,
policy playback, and live scripts. When the task benefits from a custom project
interface, reuse SimRig's live browser scene and keep the surrounding UI in
ordinary project code. Read [browser-viewers.md](references/browser-viewers.md)
before building or modifying a custom viewer.

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
| Custom experiment UI, sensor panels, maps, or task controls | Start with the matching built-in viewer; embed its live scene in a project-local UI when the built-in page does not fit |
| “Train this robot to …” | Define the task, then use an existing or custom environment |
| GPU IP / SSH host / “train on my Linux box” | `simrig remote` (not raw ssh) |
| Custom terrain, props, targets, or contacts | Custom scene plus custom environment |

Do not treat a robot model as a task. A trainable task also requires action
mapping, observations, reset distribution, rewards, termination, and measurable
success criteria. `--preset large` is PPO scale. `simrig remote` is SSH to
another Linux GPU. Do not conflate them. Never drop to raw `ssh`/`nohup` when
`simrig remote` exists.

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
Use the normal page first. If the user asks for a task-specific dashboard, add
`?embed=1` to the viewer URL to show only the live scene, then compose the
additional interface in the active project. Let the project decide which extra
data and controls are useful rather than expanding SimRig around one example.

For a known environment, run:

```bash
simrig inspect-env ENV_NAME --save-report
```

### 2. Define physical success before implementation

For a new behavior or changed task semantics, complete a **Physical Success
Definition** before authoring rewards, termination logic, a custom environment,
or its training configuration. Inspection, evaluation with an existing reviewed
contract, and reproducing an unchanged example/upstream smoke run do not need a
new definition. Smoke runs only establish pipeline operation, not promotion.
For those new semantics, this is a mandatory gate, not a reward-design exercise.
An agent cannot
guarantee that its interpretation matches the user's intent: it must draft the
definition, check it against the model and scene, challenge it with
counterexamples, and obtain focused user confirmation for every unresolved
choice that changes what counts as success.

Start from inspection evidence for the exact model and task scene. Name the
bodies, sites, joints, objects, contacts, or sensors that will be measured; do
not use labels such as "balanced," "reached," or "placed" without defining
their physical measurements. The definition must state:

- each necessary success condition as a quantity with units, coordinate frame,
  threshold or interval, required duration/consecutive control ticks, and
  evaluation horizon;
- allowed and forbidden contacts, plus prioritized terminal failures for
  unsafe contact, invalid state, ordinary task failure, and timeout;
- the initial/reset distribution and nominal, boundary, perturbation, and
  held-out scenario-by-seed cases;
- feasibility evidence from joint limits, actuator authority, reachable
  workspace, simulator/control timing, expected noise, and sensors actually
  available to the evaluator and deployed policy;
- explicit false positives: poses, contacts, trajectories, or shortcuts that
  satisfy a naive metric while violating the requested behavior;
- negative controls (at minimum zero and random action), a known-valid positive
  control, and a deliberately exploitative control expected to earn reward or
  look plausible while failing physical success.

Separate **necessary outcome conditions** from **optimization preferences**.
Necessary conditions define independent success and promotion. Preferences
such as energy, smoothness, speed, style, clearance, or comfort belong in the
reward only when their violation should not make an otherwise valid episode
fail. If a preference is actually mandatory, promote it to a measured success
condition or terminal safety rule before reward authoring.

Map the definition into a draft task contract and an evaluator plan. The
task-owned evaluator must emit raw metrics, events, and contacts from the named
physical measurements; generic predicates and promotion gates derive success
without reading reward. Document unresolved assumptions. Ask one focused
confirmation question that exposes only the choices that change the semantics,
safety envelope, or feasibility of success. Do not freeze the contract, write
the reward/termination/environment, or train until the user confirms those
choices. Read [task-design.md](references/task-design.md) for the required
definition template, feasibility audit, counterexample analysis, and controls.

Create the machine-readable draft while developing the definition, but freeze
it only after the Physical Success Definition is complete and confirmed:

```bash
simrig task init ENV_OR_PATH --output task.json
# Populate the physical definition, evaluator plan, suites, and assumptions.
simrig task validate task.json
# Freeze only after focused user confirmation.
simrig task freeze task.json --output task.frozen.json
```

Do not freeze TODO placeholders. Use `simrig task diff` and create a new frozen
version when task semantics, resets, outcomes, or promotion suites change.
Migrate old schemas explicitly with `simrig task migrate`; never silently
reinterpret a frozen contract. Use `simrig task compatibility --policy ...`
when resume, evaluation, or report comparison spans contract versions.

Continue safe inspection while waiting for confirmation. Do not invent task
semantics from the robot or task name. The user confirmation establishes an
agreed testable interpretation, not a guarantee that prose perfectly captures
the user's real-world intent; retain assumptions and limitations in the frozen
contract and handoff.

### 3. Prefer an existing Playground task

Check available environments before writing a custom one:

```bash
simrig list-envs --backend mujoco-playground
simrig inspect-env ENV_NAME
simrig smoke ENV_NAME --steps 10
simrig train ENV_NAME --preset smoke --impl auto --seed 0 --contract task.frozen.json
```

For a user-defined behavior, use an existing environment only when its robot,
task semantics, actions, and scene match the contract. For an unchanged upstream
pipeline smoke test, `--contract` may be omitted; state that limitation.
Do not choose one solely because the robot name
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

Scaffold only after the Physical Success Definition is confirmed and the task
contract is frozen:

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
simrig train envs/TASK_NAME.py --preset smoke --seed 0 --contract task.frozen.json
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
simrig train envs/TASK_NAME.py --preset smoke --contract task.frozen.json
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
simrig train ENV_OR_PATH --preset local --contract task.frozen.json
```

`--preset large` is a large PPO configuration on the current process. It does
not SSH anywhere. A GPU on this machine uses `simrig train ENV --preset large`.
Do not run it without user approval. Use `--timesteps`, `--num-envs`, and
`--batch-size` for intentional overrides. Record the exact run directory and
preserve its `config.json`, `metrics.jsonl`, `progress.json`,
`final_metrics.json`, checkpoints, and `policy.params`.
Every new run should record the resolved implementation, seed, network,
randomizer, source hashes, Git state, device inventory, and runtime versions.

To continue a crashed or shorter run:

```bash
simrig train ENV_OR_PATH --resume runs/RUN
simrig status runs/RUN
```

For another already-running Linux GPU (workstation, lab box, or cloud VM),
read [remote-gpu.md](references/remote-gpu.md). Use `simrig remote connect` for
the first interactive SSH connection, then `check`, `prepare`, remote `smoke`,
and remote `train --preset smoke` before a detached `--preset large` run:

```bash
simrig remote connect HOST --identity KEY
simrig remote check HOST --identity KEY
simrig remote prepare HOST --identity KEY
simrig remote smoke HOST ENV_OR_PATH --identity KEY --steps 10
simrig remote train HOST ENV_OR_PATH --identity KEY --preset smoke --contract task.frozen.json
simrig remote train HOST ENV_OR_PATH --identity KEY --preset large --contract task.frozen.json --detach
simrig remote status HOST REMOTE_OUTPUT --identity KEY --lines 50
simrig remote fetch HOST REMOTE_OUTPUT --identity KEY
```

SimRig does not provision or terminate the machine. Fetch the complete run
directory and remind the user to stop a billable VM after artifacts are safe.
`--preset cloud` is a hidden alias for `large`; do not teach it.

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

`simrig preview` is `reset(seed)` plus policy stepping. Prefer that path over
one-off preview scripts when a checkpoint and environment already exist.

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
Compare success rate and task-specific metrics, not just total reward. If the
env exposes `state.metrics["success"]` or `SUCCESS_SPEC`, `simrig eval` reports
`task_success` as a boolean; otherwise it stays unknown. Never rename rollout
completion to success. Read
[evaluation-and-operations.md](references/evaluation-and-operations.md) for
checkpoint compatibility, command-specific evaluation, run artifacts, and
failure triage.

Apply the frozen contract to the independent JSON reports:

```bash
simrig gate reports/*.json --contract task.frozen.json --suite nominal
simrig reward-audit reports/*.json
```

When the contract declares an evaluator plugin, prefer the reproducible matrix
runner. The evaluator emits raw metrics/events/contacts; generic predicates
derive task success and terminal reasons independently of reward:

```bash
simrig eval-suite POLICY --contract task.frozen.json --suite promotion
simrig reward-probe reports/*.json
simrig rank-checkpoints reports/checkpoint-*.json
```

Required scenario/seed coverage is a gate. A capped `eval-suite` or
`eval-checkpoints RUN_DIR` is useful during training but cannot pass a larger
promotion matrix. Reports may be ranked together only when contract, evaluator,
and suite identities match. Read
[evaluation-and-operations.md](references/evaluation-and-operations.md) for the
plugin protocol and terminal taxonomy.

## Enforce hard rules

- Never claim that an XML is trainable because it compiles or steps.
- Never author reward, termination, a custom environment, or training settings
  before a physically grounded success definition has been drafted,
  feasibility-checked, adversarially challenged, and confirmed by the user.
- Never claim to guarantee the user's semantic intent. Record assumptions,
  identify meaning-changing choices, and obtain focused confirmation before
  freezing the task contract.
- Never invent a complete reward, observation, or termination design from a
  model name.
- Never start long training before runtime validation, environment smoke, and
  `--preset smoke` training pass.
- Never treat `--preset large` as a remote launcher; use `simrig remote` for SSH.
- Never use raw `ssh`/`nohup` when `simrig remote` exists.
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
- confirmed Physical Success Definition, including measured entities, units,
  frames, thresholds, duration/horizon, contacts, and terminal precedence;
- task contract, scene assumptions, unresolved limitations, and the user's
  meaning-changing confirmations;
- feasibility checks, false-positive challenges, and positive/negative control
  outcomes;
- validation and smoke results;
- exact training command, preset, and run directory;
- checkpoint and evaluation metrics;
- preview URL when running;
- remaining limitation or next experiment.

## Commands

| Intent | Command |
|---|---|
| See robot | `simrig view-model unitree_g1 --port 8766` |
| List / inspect env | `simrig list-envs --backend mujoco-playground` / `simrig inspect-env NAME` |
| Validate custom env | `simrig validate-env PATH.py [--runtime]` |
| Smoke | `simrig smoke NAME_OR_PATH.py --steps 10` |
| Train (this machine) | `simrig train NAME_OR_PATH.py --preset smoke --contract task.frozen.json` then `local` / `large` |
| Resume | `simrig train NAME_OR_PATH.py --resume runs/RUN` |
| Local run status | `simrig status runs/RUN` |
| Other Linux GPU | `simrig remote check` / `prepare` / `smoke` / `train HOST ENV`, then `--preset large --detach` |
| Remote status / fetch | `simrig remote status HOST RUN` / `simrig remote fetch HOST RUN` |
| Eval | `simrig eval POLICY --env NAME` |
| Independent suite | `simrig eval-suite POLICY --contract CONTRACT --suite NAME` |
| Bounded live-run check | `simrig eval-checkpoints RUN --contract CONTRACT` |
| Reward probe / ranking | `simrig reward-probe REPORTS...` / `simrig rank-checkpoints REPORTS...` |
| Policy preview | `simrig preview POLICY --env NAME --port 8765` |
| Embed a browser scene | Open the printed viewer URL with `?embed=1` and compose the surrounding UI in project code |
