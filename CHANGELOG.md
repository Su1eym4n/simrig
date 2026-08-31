# Changelog

## Unreleased

- Show green success, red failure, or neutral unknown outcomes in policy previews
  using environment success metrics. Keep episode completion separate from task
  success, and replace survival wording with neutral step counts.
- Share policy loading and checked rollout execution across headless eval,
  browser preview, native demo, and a reusable Playground evaluator adapter.
  Restore final parameter files and Orbax checkpoints with recorded network
  normalization and environment identity checks.
- Add task-owned reset/measurement hooks, reusable evaluator instances, unique
  trial reports, report schema v2, missing-evidence outcomes, strict seed coverage,
  and invalid numerical state/action checks. Empty contact streams require
  declared measurement coverage; mixed identities and unknown samples cannot
  silently pass promotion.
- Add a real learned-reaching acceptance reference with IK, zero, random, and
  near-miss controls, a separately reserved holdout suite, and scenario-matched
  previews. Fix stale post-integration site positions in the reaching environment
  discovered by the independent evaluator; keep its reward formula and threshold.
- Add `preview --seed` and `eval --output`; preserve unique reports by default.

- Add versioned task contracts with `simrig task init`, `validate`, `freeze`,
  and semantic `diff`. Frozen contracts use canonical content hashes and can
  enforce resolved timestep, wall-time, GPU-hour, and metric abort budgets.
- Add task-agnostic `simrig gate` promotion suites over independent JSON
  evaluation reports, including required seed/scenario coverage and grouped
  metric requirements.
- Add `simrig reward-audit` to flag obvious disagreement between reward and
  independently recorded task success.
- Write v2 `run_manifest.json` lifecycle records with contract identity,
  checkpoint lineage, bounded local source/XML/asset closure hashes, richer Git
  state, actual progress, GPU-hours, optional cost, and failure status.
- Replace the old cloud SSH path with `simrig remote`. The host is any
  already-running Linux GPU over SSH. SimRig still does not provision or stop
  machines.
- Rename the large PPO preset from `cloud` to `large`. `--preset large` is
  scale, not a remote launcher. `--preset cloud` remains a hidden alias, and
  old run `config.json` files that recorded `"preset": "cloud"` still load.
- Write `metrics.jsonl` and `progress.json` during PPO, and add `simrig status`
  plus richer `simrig remote status` (step count, eval reward, ETA).
- Add `simrig train --resume` / `simrig remote train --resume` to restore Brax
  Orbax checkpoints.
- Report `task_success` from `state.metrics["success"]` or `SUCCESS_SPEC`
  instead of always leaving it unknown.
- Add a backend-neutral evaluator plugin protocol and `simrig eval-suite` for
  frozen scenario-by-seed matrices, with evaluator/source hashes in reports and
  run manifests.
- Add generic event, sustained-signal, metric, sequence, and contact predicates
  with a stable machine-readable terminal failure taxonomy.
- Add per-condition promotion reports, coverage-failing bounded checkpoint
  evaluation, reward adversarial probes, and reward-independent checkpoint
  ranking.
- Add explicit task-contract schema migration and purpose-specific exact,
  training-resume, checkpoint-evaluation, and result-comparison policies.
- Add a native MuJoCo controller-evaluation example with explicit task semantics,
  actual actuator stepping, an IK controller, and a zero-action baseline. Keep
  the synthetic reward-trap case under test fixtures, not public checkpoints.
- Forward additional Brax PPO settings, including adaptive-KL scheduling and
  advantage/value-loss controls, from environment training configuration.
- Include examples, test fixtures, and skill documentation in source archives
  so their tests and documented examples can run outside a Git checkout.
- Strengthen the SimRig agent skill with a mandatory, user-confirmed Physical
  Success Definition gate before reward, termination, environment authoring, or
  training for new task semantics, including feasibility checks, counterexamples,
  controls, and independent promotion evidence.

## 0.4.0 - 2026-08-17

- Require Python 3.11+, test the supported boundary in CI, and make static
  vision validation parse literal metadata without importing JAX or Playground.
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
  and opt-out domain randomization for local and remote training.
- Record the resolved implementation, network, randomizer, source hashes, Git
  state, JAX devices, precision environment, and exact CLI overrides in every run.
- Reconstruct recorded implementations and network layouts for eval, demo, and
  browser preview, while preserving legacy checkpoint compatibility.
- Add a full pinned Playground CI job that validates, smokes, trains, and reloads
  the custom reaching example.

## 0.3.0 - 2026-08-01

- Add an SSH GPU workflow with connection and GPU checks, project sync/setup,
  remote smoke and PPO training, detached-run status, and run-artifact download.
- Add a remote GPU operations guide covering SSH keys, persistent storage,
  smoke-before-scale gates, port forwarding, and stopping billable VMs.
- Pin the Playground training stack and require Python 3.11+ during remote
  preparation instead of silently resolving an older environment on Python
  3.10.
- Record Python and package versions in every training run and refuse eval,
  demo, or preview under a different runtime unless the user explicitly allows
  a qualitative mismatch.
- Validate SSH identity files, restrict authentication to the selected key,
  fail closed when port forwarding cannot be established, and support older
  macOS `rsync` clients.
- Require the final policy, metrics, and checkpoint artifacts before reporting
  a detached remote run as completed; fetched runs now include a shutdown-cost
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
