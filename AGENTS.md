# SimRig Agent Guidelines

SimRig helps users move from robot models or known simulation tasks toward a
safe training/evaluation workflow. Prefer existing MuJoCo Playground envs when
one fits. Custom `*.py` env modules can be smoked/trained after they implement
reset/step correctly.

## Workflow

1. Inspect first.
   - For raw MJCF/XML or Menagerie models, run model inspection before proposing
     training.
   - Compilation and short stepping are useful checks, but they do not prove a
     model is trainable.
2. Prefer existing trainable environments.
   - If a MuJoCo Playground environment exists for the user's robot/task, use
     `simrig inspect-env`, then `simrig smoke`, then `simrig train`.
   - For an attached or downloaded Brax policy, use `simrig eval` for headless
     checks and `simrig demo` for a desktop MuJoCo viewer.
   - Use `simrig preview` when the agent or user needs a browser-visible
     localhost preview with orbit/zoom camera controls.
   - Use `simrig view-model` to inspect raw Menagerie/XML robots with joint sliders.
   - Use `simrig demo` only when a human wants the native desktop MuJoCo viewer.
3. Custom envs (module path ending in `.py`).
   - Scaffold with `simrig new-env` only after the user chooses a task.
   - Run `simrig validate-env PATH` (structure) then `simrig validate-env PATH --runtime`.
   - Only propose `simrig smoke PATH` / `simrig train PATH` after runtime validation
     passes (or the user explicitly wants to debug failures).
   - Do not invent rewards/observations from model names alone.
   - Prefer obs dict keys `state` and `privileged_state` for SimRig PPO defaults.
4. Keep generated code editable and ordinary.
   - Do not hide reward logic or environment assumptions in opaque configs.
5. Be backend-aware.
   - SimRig v0 supports MuJoCo and MuJoCo Playground.
   - Isaac Lab is a future backend and should not be assumed available.

## Safety Rules

- Do not claim a raw Menagerie model is trainable because it compiles.
- Do not infer rewards, observations, or termination from model names alone.
- Do not invent a complete reward function just to make `new-env` look done.
- Run smoke tests before long training runs.
- Use `smoke` preset before `local` or `large`.
- `--preset large` is PPO scale. `simrig remote` is SSH to another Linux GPU.
