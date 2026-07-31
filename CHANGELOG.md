# Changelog

## Unreleased

- `simrig eval` accepts reproducible `--seed` rollouts and fixed
  `--command` values for command-conditioned environments.
- `simrig --version` reports the package version from one shared source.
- The reaching example samples reachable planar targets and keeps its rendered
  target marker synchronized with environment state.
- Codex discovers the repository skill through `.agents/skills/simrig`.
- Custom `*.py` env modules can be used with `inspect-env`, `smoke`, `train`, `eval`, `demo`, and `preview`.
- `validate-env --runtime` imports the module and runs construct/reset/step checks when possible.
- `new-env` scaffold includes `make_env()` and documents the smoke/train path.
- JAX 0.10 + Brax 0.14 compat shim for `device_put_replicated` so `simrig train` works.
- Example custom-env test drive: `examples/demo_reach.py` + `examples/models/simple_arm.xml`.

## 0.1.0

- Agent- and human-friendly CLI for MuJoCo Playground inspect, smoke, train, eval, and browser preview.
- Model inspection for MuJoCo / Menagerie assets with honest trainability status.
- Basic custom-env foothold: `simrig new-env` scaffold and `simrig validate-env` static checklist.
- Hugging Face `hf://` policy loading for eval/demo/preview.
- Open-source packaging: LICENSE, CONTRIBUTING, CI, and 10-minute quickstart docs.
