# Evaluation and operations

Use this reference when launching training, choosing a preset, loading a
checkpoint, or diagnosing a failed run.

## Presets

| Preset | Purpose | Current scale |
|---|---|---|
| `smoke` | Verify the full PPO path | 4,096 steps, 16 envs, small network |
| `local` | First meaningful local experiment | 1,000,000 steps, 128 envs |
| `cloud` | Large explicitly provisioned run | 200,000,000 steps, 8,192 envs |

Treat these values as codebase defaults, not performance guarantees. Override
only with an explicit reason. `cloud` does not submit to a cloud service; it
runs the larger configuration in the current process. For an already-launched
Lambda GPU, use `simrig cloud lambda ...` and follow
[lambda-cloud.md](lambda-cloud.md).

## Run artifacts

By default, training writes `runs/TIMESTAMP-ENV-PRESET/` containing:

- `config.json`: resolved environment reference and training configuration;
- `checkpoints/`: Brax/Orbax checkpoints;
- `policy.params`: exported policy parameters;
- `final_metrics.json`: final trainer metrics.

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

Use `--hf-token` only when necessary, and never print or commit the token.

## Evaluation

Start with:

```bash
simrig eval POLICY --env ENV_OR_PATH --steps 500
```

The built-in result reports rollout length, total reward, average reward, and
whether the episode terminated. It records task success as unknown unless a
task-specific evaluator exists; never rename rollout completion to “passed.”

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
success rate, termination reason, task error, and episode outcome. The built-in
summary does not currently calculate domain-specific measures such as velocity
tracking error or reach success rate; add a task-specific evaluator when those
measures are required. Compare against a simple baseline such as zero action or
the previous checkpoint when useful. Keep evaluation code separate from the
environment's reward so it can detect reward loopholes.

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
| Preview lacks command response | environment command state or `set_command` |

Fix the earliest failing gate and rerun from that gate. Do not compensate for
environment defects by immediately increasing training steps.
