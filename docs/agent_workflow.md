# Agent Workflow

Use this workflow when guiding a user with SimRig.

## Task Contract And Promotion Gates

For a new behavior, create a contract before reward implementation or any
meaningful training run:

```bash
simrig task init ENV_OR_PATH --output task.json
simrig task validate task.json
simrig task freeze task.json --output task.frozen.json
```

Do not freeze TODO placeholders. The contract must define deployable actions
and observations, train and native reset distributions, physical success and
failure, episode horizon, evaluation scenarios/seeds, grouped metric gates, and
compute/abort budgets. A semantic change requires a new frozen contract; use
`simrig task diff` before comparing or resuming checkpoints across versions.

Schema migration and compatibility decisions must be explicit:

```bash
simrig task migrate old-task.json --output task-v2.json
simrig task compatibility OLD NEW --policy training_resume
```

Pass the frozen identity into training:

```bash
simrig train ENV_OR_PATH --preset smoke --contract task.frozen.json
```

After task-specific evaluators emit JSON records, apply the independent suite:

```bash
simrig gate reports/*.json --contract task.frozen.json --suite nominal
simrig reward-audit reports/*.json
```

Prefer a contract-declared evaluator plugin when available. It executes the
whole fixed scenario-by-seed matrix and derives success from task-neutral
predicates over metrics, events, and contacts:

```bash
simrig eval-suite runs/RUN/policy.params \
  --contract task.frozen.json --suite promotion \
  --output reports/promotion.json
simrig reward-probe reports/promotion.json
simrig rank-checkpoints reports/checkpoint-*.json
```

Do not promote a capped matrix. `simrig eval-checkpoints RUN` is a bounded
one-shot diagnostic for checkpoints available during a run; required missing
coverage remains a gate failure. See
[evaluation-and-operations.md](../skills/simrig/references/evaluation-and-operations.md)
for the plugin and event schemas.

Do not put project-specific physics or semantics into SimRig's gate engine.
Expose those outcomes as stable evaluator metrics, then declare task-agnostic
aggregations and thresholds in the contract.

## Existing MuJoCo Playground Task (preferred)

1. List or confirm env:

   ```bash
   simrig list-envs --backend mujoco-playground
   simrig inspect-env ENV_NAME
   ```

2. Smoke test:

   ```bash
   simrig smoke ENV_NAME --steps 10
   ```

3. Train small first:

   ```bash
   simrig train ENV_NAME --preset smoke --impl auto --seed 0
   ```

   Existing Playground tasks use their upstream tuned PPO/network config and
   declared domain randomizer. `auto` selects the upstream implementation when
   supported and falls back from Warp to JAX when no JAX GPU is visible. Keep
   randomization enabled unless the user explicitly wants a baseline.

4. Evaluate:

   ```bash
   simrig eval runs/.../policy.params --env ENV_NAME --steps 500 --seed 0
   ```

5. Prefer browser previews when an AI coding agent needs to inspect the result:

   ```bash
   simrig preview runs/.../policy.params --env Go1JoystickFlatTerrain --command 0.5 0.0 0.0 --port 8765
   ```

   Open `http://127.0.0.1:8765/`. The default Three.js/WebGL page receives live
   geom transforms from a server-side rollout clock and exposes orbit, zoom,
   pan, reset, pause/resume, and command controls. Use `--render-mode mujoco`
   for the local streamed renderer.

6. Run an interactive desktop demo only when a human wants the native MuJoCo UI:

   ```bash
   simrig demo runs/.../policy.params --env ENV_NAME
   ```

## Raw MuJoCo Or Menagerie Model

1. Inspect model:

   ```bash
   simrig inspect-model MODEL_OR_XML --save-report
   ```

2. Explain status honestly:

   - compiled: XML is loadable
   - stepped: native MuJoCo simulation survived a short bounded-control rollout
   - trainable: only true when there is a real env/task

3. Open the model viewer:

   ```bash
   simrig view-model MODEL_OR_XML --port 8766
   ```

   The default Three.js/WebGL view renders visual geometry in the browser and
   provides orbit, zoom, pan, named joint controls, and authored-keyframe reset.
   It loads pinned Three.js modules from jsDelivr. Use `--render-mode mujoco`
   for the local streamed renderer or `--render-mode topdown` for a schematic
   fallback.

4. Check whether a Playground env already covers the robot/task before scaffolding.

## Standalone MuJoCo Script

When an ordinary Python controller already owns an `MjModel`, `MjData`, and
simulation loop, use `simrig.LiveWebViewer` to expose that live state in the
browser. Do not convert the controller into a policy preview or precompute a
trajectory just for visualization.

Guard mutations with `viewer.lock`, call `viewer.sync()` after completed steps,
and optionally select a named `tracking_body` for a browser-side path trail.
The script remains responsible for stepping and real-time pacing.

## Custom Env Module

Use this path when no Playground env fits and the user wants a custom task.

1. Scaffold after the user chooses a task:

   ```bash
   simrig new-env TASK_NAME --model MODEL_OR_XML --template mjx
   ```

2. Validate structure, then runtime:

   ```bash
   simrig validate-env envs/TASK_NAME.py
   simrig validate-env envs/TASK_NAME.py --runtime
   ```

3. Smoke and train with the **module path** (must end in `.py`):

   ```bash
   simrig smoke envs/TASK_NAME.py --steps 10
   simrig train envs/TASK_NAME.py --preset smoke --seed 0
   simrig eval runs/.../policy.params --env envs/TASK_NAME.py --seed 0
   ```

4. Rules for agents filling the scaffold:

   - Fill labeled `SECTION:` blocks; do not invent rewards from the model name.
   - Return obs as `{"state": ..., "privileged_state": ...}` for SimRig PPO defaults.
   - Prefer `make_env()` + `CustomEnv` as generated by `new-env`.
   - Remove the `NOT TRAINABLE YET` banner only after reset/step work.
   - Runtime validate + smoke must pass before proposing long training.
   - Put a 0/1 `success` term in `state.metrics` (and optional `SUCCESS_SPEC`)
     so `simrig eval` can report task success.

## Custom Env Checklist

A custom training env must define:

- model loading (`mj_model` / `mjx_model`)
- reset randomization
- action mapping and scaling
- policy observation (`state`)
- privileged/value observation (`privileged_state`) for PPO defaults
- reward terms
- termination
- metrics (`reward`, and `success` when the task has a pass criterion)
- render/eval metadata when useful

## Another Linux GPU

`--preset large` is PPO scale on the current process. `simrig remote` is SSH to
an already-running Linux GPU. Do not use raw `ssh`/`nohup`.

```bash
simrig remote connect HOST --identity KEY
simrig remote prepare HOST --identity KEY
simrig remote smoke HOST ENV_OR_PATH --identity KEY
simrig remote train HOST ENV_OR_PATH --identity KEY --preset smoke --contract task.frozen.json
simrig remote train HOST ENV_OR_PATH --identity KEY --preset large --contract task.frozen.json --detach
simrig remote status HOST REMOTE_OUTPUT --identity KEY
simrig remote fetch HOST REMOTE_OUTPUT --identity KEY
simrig status runs/FETCHED_RUN
```

Resume with `simrig train ENV --resume runs/RUN` locally, or
`simrig remote train HOST ENV --resume REMOTE_PATH` on the host. Keep the frozen
contract inside the synced project and pass it again when resuming.
