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
`simrig gate` and `simrig reward-audit` score already-written JSON reports.
`simrig eval-suite` runs a contract-declared plugin over the scenario/seed
matrix and then applies the same independent predicates.

### Independent evaluator plugins

A task-contract v2 evaluator is a trusted local Python file. It declares
`EVALUATOR_SPEC` (name, semantic version, protocol version 1) and
`evaluate(request)`. It may own simulator construction and checkpoint loading, or delegate them to the reusable Playground adapter described below.
It must not decide promotion from training reward.

See [the MuJoCo reaching evaluator](https://github.com/Su1eym4n/simrig/blob/main/examples/mujoco_reach/evaluator.py)
for an executable implementation that applies actual controller commands and
measures `ee_site` after stepping. Its declaration identifies the evaluator and
model; `evaluate(request)` returns measured `metrics` and `events`. Do not copy
constant example measurements into an evaluator and treat them as evidence.
`total_reward` is optional and is only useful for reward diagnostics; it is not
needed for independent success or ranking.

The request is a mapping with `checkpoint`, `environment`, `backend`, `suite`,
`scenario`, `parameters`, `seed`, `max_steps`, `task_contract_sha256`, and
`evaluator_config`. Return compact metrics plus the event/contact stream that
contract predicates need. Large observations, videos, or dense traces belong in
sidecar assets with compact references in the record.

Events use `kind`, `name`, `step`, and `active`. Contacts add `body_a`,
`body_b`, and optional `force`. Generic predicates:

| Type | Role |
|---|---|
| `sustained` | named signal held for consecutive steps |
| `forbidden_contact` | bound contacts on an unordered body pair |
| `event_count` | bound the number of named events |
| `sequence` | ordered event subsequence |
| `metric` | numeric task-owned metric vs a threshold |

SimRig derives `task_success` and a terminal reason from those predicates.
Never derive either from reward. Terminal reasons have `category`, `code`,
`message`, and optional `details`. Categories, highest precedence first:

`evaluator_error`, `invalid_state`, `forbidden_contact`, `safety_violation`,
`timeout`, `task_failure`, `incomplete`, `unknown`, `success`.

A higher-priority failure must not be hidden by a simultaneous target hit or
large reward. Test this ordering with multi-failure counterexamples.

SimRig hashes the evaluator declaration, config, and a bounded local source
closure (portable paths only). That identity is embedded in suite reports and
run manifests. Reports from different evaluator implementations cannot be
ranked together.

Run the complete matrix:

```bash
simrig eval-suite POLICY --contract task.frozen.json --suite promotion
```

Missing scenario/seed coverage always fails. Caps are diagnostics only; a
capped report stays failed when required promotion cells are missing:

```bash
simrig eval-suite POLICY --contract task.frozen.json --suite promotion \
  --max-scenarios 1 --max-seeds-per-scenario 1
simrig eval-checkpoints RUN --contract task.frozen.json \
  --suite promotion --max-checkpoints 1
```

`eval-checkpoints` is a one-shot look at checkpoints already in a run
directory. It does not start a monitor. Appended report paths are diagnostic
unless the full matrix passes.

```bash
simrig reward-probe reports/*.json
simrig rank-checkpoints reports/checkpoint-*.json
```

`reward-probe` lists failed scenario/seed cells whose reward reaches the
successful range. Rank only reports that share contract hash, evaluator hash,
and suite. Ranking is lexicographic over promotion pass, worst-condition
success rate, overall success rate, then lower safety-failure rate. Reward is
excluded.

The [MuJoCo controller example](https://github.com/Su1eym4n/simrig/blob/main/examples/mujoco_reach/README.md)
uses a scripted IK controller and zero-action baseline, not trained checkpoints.
Its target-arrival test does not establish sustained stability or contact safety.
The synthetic high-reward trap is a regression fixture in
`tests/fixtures/analytic_reach`, not physical acceptance evidence.

The protocol is backend-neutral, but plugins are task-specific: each must load
its supported simulator and policy format and measure its own physical signals.
An arbitrary environment name or checkpoint cannot run without that integration.

Evaluator plugins are not sandboxed. The Playground adapter below is available;
other backends still require task-owned integration. Distributed evaluator
workers, a columnar event store, and a persistent asynchronous checkpoint watcher
are not supplied. Checkpoint hashing refuses artifacts over its file-count bound
instead of silently hashing only a prefix.

### Reusable Playground evaluator

For learned PPO policies, use a task-owned `make_evaluator(config)` factory:

```python
from simrig.playground_evaluator import PlaygroundEvaluator

EVALUATOR_SPEC = {"name": "my-task", "version": "1", "protocol_version": 1}

def make_evaluator(config):
    return PlaygroundEvaluator(MyMeasurements)
```

`MyMeasurements(env, request)` creates a fresh observer for one episode:

- `reset(state)` applies scenario parameters and returns the initial state.
  Rebuild observations whenever target/command state changes. Reject unknown
  parameters instead of silently ignoring them.
- `observe(state, step)` measures the physical state after each step. Do not
  read training reward or success labels to decide the outcome.
- `result()` returns raw `metrics`, `events`, and measurement `evidence`.

The adapter reuses a compiled `PolicyRuntime` across seeds, resets each episode,
loads recorded network/normalization settings, checks state/action shapes and
finite values, and closes the runtime when the suite finishes. Learned policy
acceptance requires recorded compatible runtime metadata. Inputs may be final
parameters, a run directory, or a numeric Orbax checkpoint directory. A `.py`
input explicitly selects a trusted scripted control exposing
`make_controller(env) -> callable(state, rng)`; never describe it as learned.

See `examples/mujoco_reach/evaluator.py` and
`examples/mujoco_reach/README.md` for measured scripted-control evaluation.
Keep task-specific measurement logic in the example, not in SimRig's generic
adapter.

### Evidence and report validity

For an absence or upper-bound check, declare what was completely observed:

```json
{
  "events": [],
  "evidence": {
    "events": ["unsafe_motion"],
    "contacts": [{"body_a": "tool", "body_b": "wall", "complete": true}]
  }
}
```

This declaration is an evaluator obligation, not a substitute for implementing
measurements. Emit it only after observing the required channel throughout the
actual rollout. An observed forbidden contact can fail immediately; an empty
stream without coverage cannot pass. Missing metrics/channels produce
`task_success: null`, category `incomplete`, code `insufficient_evidence`.
Non-finite values, malformed events, and invalid simulator states fail closed.
Required gate metrics cannot discard unknown samples from a success-rate
calculation. Duplicate/unrequested trials, mixed contract/checkpoint identities,
and invalid-state/evaluator/safety failures prevent promotion.

Suite report schema v2 includes trial hashes and checkpoint/configuration
identity. Ranking requires matching report schema, contract, evaluator, and
suite. Default `eval`, `eval-suite`, and `eval-checkpoints` report paths are
unique, including repeated runs at the same seed; `eval --output PATH` can select
an explicit report destination. Preserve rejected reports as evidence.

### Contract migration and compatibility

Migration always writes an editable draft. Review and freeze it; do not
silently rewrite a frozen artifact:

```bash
simrig task migrate task-v1.json --output task-v2.json
simrig task validate task-v2.json
simrig task freeze task-v2.json --output task-v2.frozen.json
```

| Policy | Frozen fields |
|---|---|
| `exact` | entire contract semantics |
| `training_resume` | environment, behavior, interfaces, scene, reset, episode, outcomes |
| `checkpoint_evaluation` | environment, interfaces, scene, episode |
| `result_comparison` | environment, behavior, interfaces, scene, reset, episode, outcomes |

`training_resume` may change evaluation and compute budgets. Result comparison
still requires matching suite/evaluator hashes before reports can be ranked.

```bash
simrig task compatibility OLD NEW --policy training_resume
```

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

The shared Playground rollout currently requires recorded `action_repeat=1`.
Other values fail explicitly instead of silently changing the policy control
rate. CPU state-observation integration is covered by CI smoke and eval;
vision/GPU and hardware acceptance need separate evidence.
