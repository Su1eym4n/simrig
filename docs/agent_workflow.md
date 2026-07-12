# Agent Workflow

Use this workflow when guiding a user with SimRig.

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
   simrig train ENV_NAME --preset smoke
   ```

4. Evaluate:

   ```bash
   simrig eval runs/.../policy.params --env ENV_NAME --steps 500
   ```

5. Prefer browser previews when an AI coding agent needs to inspect the result:

   ```bash
   simrig preview runs/.../policy.params --env Go1JoystickFlatTerrain --command 0.5 0.0 0.0 --port 8765
   ```

   Open `http://127.0.0.1:8765/`. The page streams rendered frames from the
   local Python simulation and exposes reset, pause/resume, and command
   controls.

6. Run an interactive desktop demo only when a human wants the native MuJoCo UI:

   ```bash
   simrig demo runs/.../policy.params --env ENV_NAME
   ```

   For joystick locomotion policies, keep commands explicit:

   ```bash
   simrig demo runs/.../policy.params --env Go1JoystickFlatTerrain --command 0.0 0.0 0.0
   ```

## Raw MuJoCo Or Menagerie Model

1. Inspect model:

   ```bash
   simrig inspect-model MODEL_OR_XML --save-report
   ```

2. Explain status honestly:

   - compiled: XML is loadable
   - stepped: native MuJoCo simulation survived a short bounded-control rollout
   - trainable: only true when there is a real env/task (usually a Playground env)

3. Check whether a Playground env already covers the robot/task before scaffolding.

## Custom Env (basic foothold — not trainable via CLI yet)

Use this path only when no Playground env fits and the user explicitly wants a
custom task.

1. Scaffold after the user chooses a task:

   ```bash
   simrig new-env TASK_NAME --model MODEL_OR_XML --template mjx
   ```

2. Validate the static checklist (structure only):

   ```bash
   simrig validate-env envs/TASK_NAME.py
   ```

3. Tell the user clearly:

   - Passing `validate-env` means section markers and required symbols/methods
     are present.
   - It does **not** mean rewards/observations are correct or that training works.
   - `simrig train` in v0.1 only accepts MuJoCo Playground registry env names.
   - End-to-end custom env train/eval is planned for a later release.

4. When editing the scaffold, fill the labeled `SECTION:` blocks. Do not invent
   rewards from the model or task name alone.

## Custom Env Checklist

A custom training env must eventually define:

- model loading
- reset randomization
- action mapping and scaling
- policy observation
- privileged/value observation if using PPO defaults
- reward terms
- termination
- metrics
- render/eval metadata when useful
