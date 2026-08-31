# Analytic reaching regression fixture

This dependency-free fixture exercises evaluator loading, sustained/contact
predicates, report coverage, reward probes, and ranking in
[`test_evaluation_suite.py`](../../test_evaluation_suite.py). It does not run
MuJoCo, train a policy, or establish physical robot performance.

`controllers/valid.json` and `controllers/reward_trap.json` are selectors for
hardcoded analytic controller branches, not policy checkpoints. The trap's
large reward is deliberately fabricated; the synthetic contact's `force` value
is an indicator, not a simulated force in newtons. These controlled inputs test
whether the generic evaluator rejects an unsuccessful high-reward record.

Run the regression tests from the repository root:

```bash
python -m pytest tests/test_evaluation_suite.py -q
```

For an executable example that applies actuator commands and measures their
physical result, see [the MuJoCo controller example](../../../examples/mujoco_reach/README.md).
