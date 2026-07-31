# Contributing to SimRig

Thanks for helping improve SimRig. Keep changes small, honest about
trainability, and useful for both humans and coding agents.

## Development setup

Use Python 3.10 or newer in an isolated environment:

```bash
git clone https://github.com/Su1eym4n/simrig.git
cd simrig
python3.12 -m venv .venv  # or another installed Python >= 3.10
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

For MuJoCo model inspection:

```bash
python -m pip install -e ".[dev,mujoco]"
```

For Playground train/eval (heavier JAX/Brax stack):

```bash
python -m pip install -e ".[dev,playground]"
python -m pip check
```

## Tests

Core tests (no Playground required):

```bash
python -m pytest
```

CI runs the default pytest suite. Integration tests that need MuJoCo or
Playground should skip cleanly when those extras are missing.

With the Playground extra installed, also run the bundled end-to-end example:

```bash
simrig validate-env examples/demo_reach.py --runtime
simrig smoke examples/demo_reach.py --steps 10
simrig train examples/demo_reach.py --preset smoke
simrig eval runs/<run-dir>/policy.params \
  --env examples/demo_reach.py \
  --steps 50 \
  --seed 0
```

Replace `<run-dir>` with the directory printed by `simrig train`. The smoke
preset verifies the PPO path; it is not evidence that a useful policy has
converged.

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
