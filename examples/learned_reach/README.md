# Learned reaching with independent acceptance

This reference connects real Brax checkpoints to SimRig's scenario runner. It
uses the existing two-joint `demo_reach.py` task and a task-owned measurement
class. SimRig owns policy loading, observation normalization, state/action
validation, stepping, seed isolation, report identity, and acceptance gates.

**Recorded outcome:** the corrected learned policy passes all 24 development
and holdout cases, but random actions do too. The verifier therefore exits 1
because the expected baseline discrimination fails. This reference demonstrates
the complete workflow and exposes an overly easy arrival criterion; it does
not establish a useful manipulation skill. See [results.md](results.md).

## Physical definition

The controlled point is `ee_site` on `examples/models/simple_arm.xml`. Success
requires world-frame Euclidean target error **strictly below 0.05 m at any
post-step control tick**, within **200 ticks / 4 seconds**. A tick comprises ten
2 ms physics steps. Two normalized actions map to joint-position targets in
the shoulder/elbow actuator ranges. Resets retain the original JAX uniform
joint positions in [-0.4, 0.4] rad and zero velocity. Evaluation replaces the
random target with the declared fixed target and rebuilds policy observations.

The evaluator independently runs native MuJoCo forward kinematics on the
integrated joint positions. It does not read the environment's success metric
or reward. Invalid states/actions and evaluator errors fail before acceptance.
The gate requires every declared case to pass; missing or duplicate cases fail.

The target positions are [0.25, 0, 0.22] and [0.31, 0, 0.22] m. Relative to the
shoulder at [0, 0, 0.10], their radii are approximately 0.277 and 0.332 m, inside
the training radius range and the arm's 0.38 m geometric reach. The supplied
IK control checks actuator limits and provides a positive feasibility control.
The model has no physical force sensors; this reference does not certify
contact safety. Holding, smoothness, hardware transfer, and robustness outside
the declared distribution are not requirements of this arrival-only task.

`task.json` defines a six-case calibration suite, a twelve-case promotion
regression suite, and a separate twelve-case holdout suite. The holdout seeds
1001–1012 are reserved until after selecting the final policy. The fixed targets
remain inside the training distribution; this is held-out reset evaluation,
not an out-of-distribution benchmark. If holdout results inform another training
decision, treat that suite as development data and reserve a new final set.

## Reproduce from the repository root

Install `.[playground]` in an isolated environment as described in the main
README. These examples ship in the source archive, not the wheel.

```bash
simrig inspect-model examples/models/simple_arm.xml
simrig validate-env examples/demo_reach.py --runtime
simrig smoke examples/demo_reach.py --steps 10

# This freezes the supplied definition above, not an invented new task.
simrig task freeze examples/learned_reach/task.json \
  --output runs/learned-reach/task.frozen.json
simrig train examples/demo_reach.py --preset smoke --seed 5 \
  --contract runs/learned-reach/task.frozen.json \
  --output runs/learned-reach/smoke

# Only after smoke succeeds. This is a bounded local run, not a large/GPU job.
simrig train examples/demo_reach.py --preset local --timesteps 180000 \
  --num-envs 32 --batch-size 32 --seed 5 \
  --contract runs/learned-reach/task.frozen.json \
  --output runs/learned-reach/trained

python -m examples.learned_reach.verify \
  --policy runs/learned-reach/trained/policy.params \
  --contract runs/learned-reach/task.frozen.json \
  --output runs/learned-reach/evaluation
```

The verifier runs learned parameters and four ordinary Python controls through
the same MJX action/step path and independent measurements. It preserves unique
reports, ranks them without reward, and exits nonzero when the learned policy
fails or controls do not behave as expected. Do not reduce thresholds to make
the script exit successfully. PPO convergence is not guaranteed by a seed or
timestep count; compare the actual reports.

| Artifact | Purpose |
|---|---|
| `policy.params` | Actual trained PPO actor/value parameters |
| `controllers/ik.py` | Known-valid IK control; expected to pass every case |
| `controllers/zero.py` | Midpoint joint-position targets; expected to fail the full suite |
| `controllers/random.py` | Seeded uniform actions; expected to fail the full suite |
| `controllers/near_miss.py` | Aims 9 cm short; challenges loose proximity checks |

Zero, random, and near-miss controls can still cross the true target in some
episodes. Those crossings legitimately pass this arrival-only definition and
must not be relabeled failures. The controls challenge suite discrimination;
they are not evidence that the task requires holding. They are trusted code,
not sandboxed plugins, and must not mutate the environment/model.

After selecting a checkpoint without consulting holdout results:

```bash
python -m examples.learned_reach.verify \
  --policy runs/learned-reach/trained/policy.params \
  --contract runs/learned-reach/task.frozen.json --suite holdout \
  --output runs/learned-reach/holdout
```

## Preview the same case

This small task-specific script reuses the measurement class's scenario reset
and SimRig's viewer; it contains no checkpoint-loading or stepping code:

```bash
python -m examples.learned_reach.preview \
  --policy runs/learned-reach/trained/policy.params \
  --contract runs/learned-reach/task.frozen.json \
  --scenario heldout_nominal --seed 101 --port 8768
```

Open `http://127.0.0.1:8768/`. The first episode uses exactly that target and
seed. Automatic reset starts disabled; manually resetting changes the RNG, so
restart the command to reproduce the original seed. Ordinary random-target
playback remains available with `simrig preview POLICY --env
examples/demo_reach.py --seed 101 --auto-reset`.

## Regression discovered by this evaluator

Before this change, `demo_reach.py` read site positions left over from before
the last MJX integration. A policy could receive success while its actual
end-of-tick end effector was outside 5 cm. The environment now calls
`mjx.forward` after integration so observations, reward, and termination all
refer to the integrated pose. A regression test compares those values with
native MuJoCo across moving states. The reward formula and 5 cm threshold are
unchanged. Old checkpoints record a different environment source hash and must
not silently be treated as checkpoints trained on the corrected environment.

See `results.md` for the recorded local execution and its limitations.
