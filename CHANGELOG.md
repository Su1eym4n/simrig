# Changelog

## Unreleased

## 0.2.1 - 2026-07-31

- `simrig preview` now uses the Three.js/WebGL renderer by default, with a
  server-side rollout clock, live geom-transform updates, and a tracking camera
  that follows the evaluated robot without coupling orbit controls to frame
  rendering.
- `simrig preview --render-mode mujoco` retains the streamed MuJoCo policy
  preview for fully local/offline rendering.

## 0.2.0 - 2026-07-31

- `simrig view-model` now defaults to a GPU-accelerated Three.js/WebGL viewer
  with smooth orbit, zoom, pan, physically based materials, lighting, shadows,
  and responsive browser rendering.
- The browser model viewer serves compiled MuJoCo mesh and primitive geometry,
  filters collision-only geometry from the default visual scene, and updates
  geom transforms when joint controls change.
- Model viewing starts from the first authored MJCF keyframe when one exists;
  **Reset Joints** restores that pose.
- `simrig view-model --render-mode mujoco` retains the streamed MuJoCo renderer,
  and `--render-mode topdown` retains the schematic fallback.
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
