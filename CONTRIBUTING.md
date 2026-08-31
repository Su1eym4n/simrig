# Contributing to SimRig

Thanks for helping improve SimRig. Keep changes small, honest about
trainability, and useful for both humans and coding agents.

## Development setup

Use Python 3.11 or newer in an isolated environment:

```bash
git clone https://github.com/Su1eym4n/simrig.git
cd simrig
python3.12 -m venv .venv  # or another installed Python >= 3.11
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

To exercise real checkpoint loading (both final parameters and Orbax), shared
preview/eval execution, and independent physical measurements:

```bash
SIMRIG_TEST_POLICY=runs/<run-dir>/policy.params \
  python -m pytest tests/test_learned_reach.py -q
```

CI runs this against its tiny policy to validate integration, not convergence.
The [learned reaching reference](examples/learned_reach/README.md) separately
checks physical acceptance and baseline discrimination. A successful training
command or a green integration test does not imply those checks pass.

## Pull requests

- Prefer focused PRs over large mixed changes.
- Do not claim a raw Menagerie/XML model is trainable because it compiles.
- Do not hide reward or termination logic in opaque configs.
- Update `AGENTS.md` / `docs/agent_workflow.md` when agent-facing behavior changes.
- Add or update tests for CLI and library behavior you change.

## Releases

Bump `simrig/_version.py`, move notes out of `Unreleased` in `CHANGELOG.md`,
and merge to `main` with CI green. Then tag and push:

```bash
git tag -a vX.Y.Z -m "SimRig X.Y.Z"
git push origin vX.Y.Z
```

The tag must match the package version. GitHub Actions publishes to PyPI and
creates the GitHub Release.

## Scope notes

- Preferred path: existing MuJoCo Playground environments.
- Custom envs: `new-env`, `validate-env` / `--runtime`, and smoke/train via `*.py` module paths.
- Isaac Lab and other heavy backends are out of scope until explicitly planned.
