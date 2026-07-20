# Contributing to SimRig

Thanks for helping improve SimRig. Keep changes small, honest about
trainability, and useful for both humans and coding agents.

## Development setup

```bash
python -m pip install -e ".[dev]"
```

For MuJoCo model inspection:

```bash
python -m pip install -e ".[dev,mujoco]"
```

For Playground train/eval (heavier JAX/Brax stack):

```bash
python -m pip install -e ".[dev,playground]"
```

## Tests

Core tests (no Playground required):

```bash
python -m pytest
```

CI runs the default pytest suite. Integration tests that need MuJoCo or
Playground should skip cleanly when those extras are missing.

## Pull requests

- Prefer focused PRs over large mixed changes.
- Do not claim a raw Menagerie/XML model is trainable because it compiles.
- Do not hide reward or termination logic in opaque configs.
- Update `AGENTS.md` / `docs/agent_workflow.md` when agent-facing behavior changes.
- Add or update tests for CLI and library behavior you change.

## Scope notes

- v0.1 first-class path: existing MuJoCo Playground environments.
- Custom envs: `new-env`, `validate-env` / `--runtime`, and smoke/train via `*.py` module paths.
- Isaac Lab and other heavy backends are out of scope until explicitly planned.
