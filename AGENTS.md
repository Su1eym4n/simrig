# SimRig Agent Guidelines

SimRig helps users move from robot models or known simulation tasks toward a
safe training/evaluation workflow. Prefer existing MuJoCo Playground envs.
Custom envs are a basic scaffold + checklist in v0.1 — not end-to-end training.

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
3. Custom envs (basic only in v0.1).
   - Use `simrig new-env` only as a starter template after the user chooses a task.
   - Run `simrig validate-env path/to/env.py` for a static checklist.
   - Passing validate-env does **not** mean the env is trainable.
   - Do **not** run `simrig train` on a custom env module; v0.1 trains Playground
     registry names only.
   - A real custom env must still define reset, step, action mapping,
     observations, rewards, and termination before any future train path.
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
- Use `smoke` preset before `local` or `cloud`.
- Do not copy Cortex, AXIS, Daionics, or project-specific research logic into
  generic SimRig code.
