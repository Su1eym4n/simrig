---
name: simrig
description: >-
  Train, evaluate, and preview MuJoCo / MuJoCo Playground robot policies with the
  SimRig CLI. Use when the user wants to train a robot, write a custom RL env,
  smoke-test simulation, eval/preview a Brax policy, inspect Menagerie/MJCF
  models, see how a robot looks, or run Physical AI locomotion/manipulation
  workflows with an agent.
metadata:
  short-description: MuJoCo Playground train/eval via SimRig CLI
---

# SimRig

Drive the `simrig` CLI. Do **not** re-implement viewers, trainers, or mesh
hunts by hand. Prefer one clear command, then report the URL or error.

## Prerequisites (30 seconds)

```bash
command -v simrig || python -m pip install -e ".[playground]"  # or pip install simrig[playground]
simrig --help
```

Menagerie (needed to **look at** Go1/G1 meshes):

```bash
echo "${MUJOCO_MENAGERIE_PATH:-}"
# common: ~/Desktop/mujoco_menagerie  or  ~/mujoco_menagerie
```

If Menagerie is missing, **ask the user for the path or to clone it**. Do **not**
recursively search the home directory for STL files.

```bash
# user can clone once:
git clone https://github.com/google-deepmind/mujoco_menagerie.git ~/Desktop/mujoco_menagerie
export MUJOCO_MENAGERIE_PATH=~/Desktop/mujoco_menagerie
```

## Intent router (pick ONE path)

| User intent | Do this |
|-------------|---------|
| “Show me / look at G1/Go1” | **View model** (below) — not train, not inspect-env rabbit holes |
| “Train / walk / smoke” | Playground env workflow |
| “New skill / jump / reach” | Custom env workflow |
| “Try my policy” | `eval` then `preview` |

### View model (see the robot) — default for “how does it look”

```bash
# resolve menagerie once
MEN=${MUJOCO_MENAGERIE_PATH:-$HOME/Desktop/mujoco_menagerie}
test -d "$MEN/unitree_g1" || MEN=$HOME/mujoco_menagerie

simrig view-model unitree_g1 --menagerie "$MEN" --port 8766
# tell user: open http://127.0.0.1:8766/
```

Aliases: `unitree_go1`, or a path to `scene.xml`.

**Anti-patterns (do not do these for “look at G1”):**
- Do not start from `G1JoystickFlatTerrain` XML inside site-packages and chase broken mesh URIs.
- Do not `find` / search the whole home folder for meshes.
- Do not train or scaffold an env just to visualize.
- If meshes are missing: stop and ask for `MUJOCO_MENAGERIE_PATH` / clone Menagerie.

### Policy preview (see a trained policy move)

```bash
simrig preview PATH/TO/policy.params --env G1JoystickFlatTerrain --command 0.5 0 0 --port 8765
# open http://127.0.0.1:8765/
```

Needs a real policy file. Without one, use `view-model` instead.

## Hard rules

- Compiling XML ≠ trainable.
- Prefer existing Playground envs when they fit.
- Do not invent rewards from task names.
- `smoke` before long train; `--preset smoke` before `local`/`cloud`.
- Prefer `preview` / `view-model` over native `demo`.
- Custom tasks = `*.py` module + `simrig`, not one-off scripts.
- Fail fast with a clear ask to the user; do not wander the filesystem.

## Workflow A — Playground train

```bash
simrig list-envs --backend mujoco-playground
simrig inspect-env G1JoystickFlatTerrain
simrig smoke G1JoystickFlatTerrain --steps 10
simrig train G1JoystickFlatTerrain --preset smoke
simrig eval runs/.../policy.params --env G1JoystickFlatTerrain
simrig preview runs/.../policy.params --env G1JoystickFlatTerrain --command 0.5 0 0 --port 8765
```

## Workflow B — custom task

```bash
simrig inspect-model MODEL_OR_XML --save-report
simrig new-env TASK_NAME --model MODEL_OR_XML --template mjx
# edit SECTION blocks
simrig validate-env PATH.py --runtime
simrig smoke PATH.py --steps 10
simrig train PATH.py --preset smoke
```

Obs keys: prefer `state` + `privileged_state`. Keep metrics keys stable (include `reward`).

## Commands

| Intent | Command |
|--------|---------|
| **See robot** | `simrig view-model unitree_g1 --menagerie $MEN --port 8766` |
| List envs | `simrig list-envs --backend mujoco-playground` |
| Inspect env | `simrig inspect-env NAME` |
| Inspect model | `simrig inspect-model MODEL --save-report` |
| Scaffold | `simrig new-env NAME --model XML` |
| Validate | `simrig validate-env PATH.py [--runtime]` |
| Smoke | `simrig smoke NAME_OR_PATH.py --steps 10` |
| Train | `simrig train NAME_OR_PATH.py --preset smoke` |
| Eval | `simrig eval POLICY --env NAME` |
| Policy preview | `simrig preview POLICY --env NAME --port 8765` |
