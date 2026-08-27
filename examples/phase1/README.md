# Phase 1 independent-evaluation acceptance example

This example is a dependency-free analytic two-link arm. It tests SimRig's
evaluation architecture, not a training backend. The valid controller reaches
nominal and boundary targets and sustains the hold. The reward-trap controller
earns a deliberately large proxy reward while missing the task and entering a
forbidden region.

```bash
simrig task validate examples/phase1/planar_reach_task.json
simrig task freeze examples/phase1/planar_reach_task.json \
  --output /tmp/planar-reach.frozen.json

simrig eval-suite examples/phase1/checkpoints/valid.json \
  --contract /tmp/planar-reach.frozen.json \
  --suite promotion \
  --output /tmp/valid-eval.json

simrig eval-suite examples/phase1/checkpoints/reward_trap.json \
  --contract /tmp/planar-reach.frozen.json \
  --suite promotion \
  --output /tmp/reward-trap-eval.json

simrig reward-probe /tmp/valid-eval.json /tmp/reward-trap-eval.json
simrig rank-checkpoints /tmp/valid-eval.json /tmp/reward-trap-eval.json
```

Expected result: the valid controller passes and ranks first. The reward trap
fails independent predicates even though its reported reward is much larger.
