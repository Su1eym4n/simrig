# Changelog

## Unreleased

- Publish a five-seed-tested, 1,000-step vision CartPole reference checkpoint
  and download its sibling `config.json` with `hf://` policies so vision CNNs
  reconstruct from Hub artifacts without manual metadata setup.
- Make policy previews task-aware: hide unsupported command controls, accept
  environment-defined command fields, expose episode state and survival, add
  optional automatic reset, and label physics, policy-observation, sensor, and
  browser display rates separately.
- Keep Three.js previews headless-safe by lazily starting the optional native
  Sensor renderer and selecting EGL before MuJoCo import on display-less Linux.
- Pin the tested Warp 1.13 runtime and bridge MuJoCo 3.10's stale internal
  `GraphMode` import so CUDA vision environments can construct and step. The
  vision reference also normalizes Warp's unbatched singleton renderer axis so
  checkpoint evaluation receives logical HWC pixels.
- Show live, selectable named cameras inside Three.js model and policy previews:
  an authored-pose/FOV Three.js emulation by default, plus a native MuJoCo
  sensor comparison, while preserving the independent human orbit camera.
- Add declarative custom-env vision CNN selection across train/eval/demo/preview,
  rendered-frame validation with CUDA/Warp capability reporting, and a complete
  MJX-rendered cartpole vision PPO reference.
- Resolve registered MuJoCo Playground training from each task's tuned Brax PPO
  and network configuration while preserving SimRig's bounded smoke/local gates.
- Add `--impl auto|jax|warp`, GPU-aware Warp fallback, reproducible `--seed`,
  and opt-out domain randomization for local and Lambda training.
- Record the resolved implementation, network, randomizer, source hashes, Git
  state, JAX devices, precision environment, and exact CLI overrides in every run.
- Reconstruct recorded implementations and network layouts for eval, demo, and
  browser preview, while preserving legacy checkpoint compatibility.
- Add a full pinned Playground CI job that validates, smokes, trains, and reloads
  the custom reaching example.

## 0.3.0 - 2026-08-01

- Add an SSH-based Lambda On-Demand Cloud workflow with connection and GPU
  checks, project sync/setup, remote smoke and PPO training, detached-run
  status, and run-artifact download.
- Add a Lambda GPU operations guide covering SSH keys, persistent storage,
  smoke-before-cloud gates, port forwarding, and instance shutdown.
- Pin the Playground training stack and require Python 3.11+ during Lambda
  preparation instead of silently resolving an older environment on Python
  3.10.
- Record Python and package versions in every training run and refuse eval,
  demo, or preview under a different runtime unless the user explicitly allows
  a qualitative mismatch.
- Validate SSH identity files, restrict authentication to the selected key,
  fail closed when port forwarding cannot be established, and support older
  macOS `rsync` clients.
- Require the final policy, metrics, and checkpoint artifacts before reporting
  a detached Lambda run as completed; fetched runs now include a shutdown-cost
  reminder.
- Evaluation results distinguish rollout completion from task success and state
  clearly when no task-specific success evaluator is configured.

## 0.2.2 - 2026-07-31

- `LiveWebViewer` lets ordinary MuJoCo Python scripts publish their existing
  `MjModel`/`MjData` state to the interactive Three.js viewer without handing
  control or simulation stepping to SimRig.
- Live script viewers can track a named body and draw its browser-side path.
- Package metadata links PyPI users back to the repository, issues, and
  changelog.
- CI installs its NumPy test dependency, validates built distributions, and
  uses Node.js 24-compatible GitHub Actions.
- Tagged releases publish through PyPI Trusted Publishing and create a GitHub
  Release with the wheel and source distribution attached.

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
