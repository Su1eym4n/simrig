# Evaluation and operations

Use this reference when launching training, choosing a preset, loading a
checkpoint, or diagnosing a failed run.

## Presets

| Preset | Purpose | Current scale |
|---|---|---|
| `smoke` | Verify the full PPO path | 4,096 steps, 16 envs, small network |
| `local` | First meaningful local experiment | 1,000,000 steps, 128 envs |
| `large` | Large run on the current process | 200,000,000 steps, 8,192 envs |

Treat these values as codebase defaults, not performance guarantees. Override
only with an explicit reason. `large` does not SSH anywhere; it runs the larger
configuration in the current process. `--preset cloud` is a hidden alias for
`large` so old scripts and `config.json` files still load; do not teach it.
For another already-running Linux GPU, use `simrig remote ...` and follow
[remote-gpu.md](remote-gpu.md).

## Run artifacts

By default, training writes `runs/TIMESTAMP-ENV-PRESET/` containing:

- `config.json`: resolved environment reference and training configuration;
- `run_manifest.json`: frozen task identity, lineage, provenance, compute, and status;
- `metrics.jsonl` / `progress.json`: eval curve, step count, and ETA while the run is live;
- `checkpoints/`: Brax/Orbax checkpoints;
- `policy.params`: exported policy parameters;
- `final_metrics.json`: final trainer metrics;
- `train.log` / `train.pid`: detached remote runs.

Inspect a local or fetched run with `simrig status RUN_DIR`. Inspect a detached
remote run with `simrig remote status HOST RUN`. Resume with
`simrig train ENV --resume RUN_DIR` (or `simrig remote train HOST ENV --resume`
using a path on the remote host).

Reports are written under `reports/`. Preserve a run directory as one unit so
checkpoint architecture can be inferred from its sibling `config.json`.

## Checkpoint compatibility

Evaluate a checkpoint with the exact environment implementation and compatible
network architecture used for training. SimRig infers the small network from
the sibling `config.json`; use `--small-network` or `--no-small-network` only
when that metadata is missing and the architecture is known.

Resolve Hub checkpoints with:

```bash
simrig eval hf://OWNER/REPO/PATH --env ENV_OR_PATH --hf-revision REVISION
```

Hub resolution downloads the requested policy together with a sibling
`config.json` when the repository provides one. Publish both files in the same
Hub directory so SimRig can reconstruct recorded vision networks and runtime
requirements rather than falling back to legacy checkpoint defaults.

Use `--hf-token` only when necessary, and never print or commit the token.

## Evaluation

### Physical success is defined before it is evaluated

Evaluation implements the user-confirmed Physical Success Definition from the
frozen task contract; it does not discover the meaning of success from reward,
termination, training metrics, or visual appeal. Before trusting an evaluator,
verify that its named measurements, units, coordinate frames, thresholds,
duration, horizon, contact rules, and terminal precedence match the definition
and can be obtained from the inspected model and scene.

The agent cannot guarantee the user's semantic intent. Retain assumptions in
the contract and evaluator documentation, adversarially challenge the drafted
definition, and request focused confirmation before freezing any
meaning-changing choice. A 0/1 `success` environment metric is an integration
signal, not the authoritative definition.

Keep evaluator evidence independent of reward. The evaluator should expose raw
physical metrics, signals, event sequences, and named contacts. Express
necessary conditions with generic metric, sustained, event-count, sequence,
and forbidden-contact predicates; express grouped acceptance thresholds in the
promotion suite. Optimization preferences may remain reward terms only when
their violation does not invalidate physical success.

Calibrate the evaluator with controls before using it for promotion:

- zero-action and seeded random-action controls must fail for the expected
  physical reasons;
- a known-valid control must pass nominal, boundary, perturbation, and held-out
  scenario/seed coverage;
- a deliberately exploitative controller or policy must fail even if its
  reward is high or its motion looks plausible.

If no known-valid control exists, report evaluator calibration as incomplete;
do not turn training reward into substitute evidence. Run `simrig reward-probe`
over the positive and exploitative reports together, because a high-reward
failure probe needs successful records as its comparison baseline.

Start with:

```bash
simrig eval POLICY --env ENV_OR_PATH --steps 500
```

The built-in result reports rollout length, total reward, average reward, and
whether the episode terminated. If the environment writes `state.metrics["success"]`
or declares `SUCCESS_SPEC` / `success_spec()`, eval reports `task_success` as a
boolean. Otherwise it stays unknown. Never rename rollout completion to
“passed.”

Use `--seed N` for a reproducible rollout and `--command X Y YAW` to hold
command-like environment state fixed:

```bash
simrig eval POLICY \
  --env Go1JoystickFlatTerrain \
  --steps 500 \
  --seed 4 \
  --command 0.8 0 0
```

The command is reapplied throughout the rollout so environment resampling does
not replace it. SimRig rejects `--command` when the environment cannot accept
command-like state. Run the command once per evaluation seed.

Evaluate over the task contract's complete seed and scenario matrix. Aggregate
success rate, termination reason, task error, and episode outcome. Domain-specific
measures such as velocity tracking error still need a dedicated evaluator when
`SUCCESS_SPEC` is not enough. Compare against a simple baseline such as zero
action or the previous checkpoint when useful. Keep evaluation code separate from
the environment's reward so it can detect reward loopholes.

Apply declared coverage and promotion thresholds after independent reports
exist:

```bash
simrig gate reports/*.json --contract task.frozen.json --suite nominal
simrig reward-audit reports/*.json
```

The gate engine is task-agnostic. Task-specific evaluators expose stable
metrics; the contract supplies grouping, aggregation, operators, and thresholds.

### Independent evaluator plugins

A task-contract v2 evaluator is a Python file declaring `EVALUATOR_SPEC` with a
name, semantic version, and protocol version 1, plus `evaluate(request)`. The
request names the checkpoint, backend/environment, suite, scenario parameters,
fixed seed, horizon, contract hash, and evaluator config. The result contains
compact metrics and an event stream; SimRig applies contract predicates to
derive `task_success` and the terminal reason. Never derive either from reward.

Events use `kind`, `name`, `step`, and `active`. Contacts add `body_a`, `body_b`,
and optional `force`. Generic predicates support sustained signals, forbidden
contacts, event counts and sequences, and numeric task-owned metrics. Stable
terminal categories include success, timeout, forbidden contact, safety or
invalid-state failures, ordinary task failure, incomplete execution, evaluator
error, and unknown outcome.

Run a complete matrix with:

```bash
simrig eval-suite POLICY --contract task.frozen.json --suite promotion
```

Missing scenario/seed coverage always fails. Use `simrig eval-checkpoints RUN`
only for a bounded one-shot diagnostic while checkpoints are being produced.
Evaluator identity includes its version, config, and bounded source closure and
is recorded in suite reports and run manifests.

Inspect terminal results using an explicit precedence. Evaluator errors and
invalid states outrank safety/forbidden-contact failures, which outrank ordinary
task failure, incomplete execution, and timeout. A higher-priority failure must
not be hidden by a simultaneous target hit or large reward. Test this ordering
with multi-failure counterexamples.

Use `simrig reward-probe REPORTS...` to surface concrete high-reward failures.
Use `simrig rank-checkpoints REPORTS...` only for reports sharing the same
contract, evaluator, and suite hashes. Ranking uses promotion, worst-condition
and overall independent success, and safety failures; it excludes reward.

See the repository's `docs/independent-evaluation.md` for the complete protocol,
contract compatibility policies, and current limitations.

Preview the exact evaluated pair:

```bash
simrig preview POLICY --env ENV_OR_PATH --port 8765
```

Training run `config.json` files record the Python and package runtime. Eval,
demo, and preview reject recorded version mismatches by default. The
`--allow-runtime-mismatch` escape hatch is for qualitative compatibility review
only and must be reported as such.

The default `threejs` mode updates browser-side visual geometry from the live
rollout while keeping camera interaction local. Use `--render-mode mujoco` for
the streamed MuJoCo renderer and `--render-mode topdown` only as a schematic
debugging fallback. For raw models without a policy, use `simrig view-model`
instead.

## Failure triage

| Failure | Inspect first |
|---|---|
| XML does not compile | include paths, mesh paths, named references, schema |
| Model steps badly | initial pose, penetrations, timestep, actuator limits |
| Runtime validation fails | import path, model path, JAX tracing, state fields |
| Smoke fails | observation shapes, action size, non-finite state, JIT errors |
| PPO fails immediately | obs dictionary keys, batch/env overrides, reward NaNs |
| Reward improves but behavior is wrong | reward loopholes and term scale |
| Policy load shape mismatch | environment and network architecture compatibility |
| Resume refuses the checkpoint | recorded env/network/impl vs this command; `--allow-resume-mismatch` is qualitative |
| Preview lacks command response | environment command state or `set_command` |

Fix the earliest failing gate and rerun from that gate. Do not compensate for
environment defects by immediately increasing training steps.
