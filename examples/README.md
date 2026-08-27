# Examples

Tiny assets for testing SimRig's **custom env CLI path** and independent
evaluation — not product tasks.

## demo_reach

1. Model: [`models/simple_arm.xml`](models/simple_arm.xml) (2-DOF arm)
2. Env module: [`demo_reach.py`](demo_reach.py) (filled `new-env` scaffold)

Targets are sampled inside the arm's reachable X-Z plane. The environment moves
the red mocap marker to the sampled task target so evaluation and previews show
the same goal used by the reward.

Recreate the scaffold yourself:

```bash
simrig new-env demo_reach --model examples/models/simple_arm.xml --root envs
```

Then fill the `SECTION:` blocks (see `examples/demo_reach.py` for a working fill).

Run the test drive (needs `pip install -e ".[playground]"`):

```bash
simrig validate-env examples/demo_reach.py --runtime
simrig smoke examples/demo_reach.py --steps 10
simrig train examples/demo_reach.py --preset smoke
```

## vision_cartpole

[`vision_cartpole.py`](vision_cartpole.py) is the smallest end-to-end vision
reference. It uses MuJoCo's actual MJX camera renderer, a three-frame grayscale
stack, Brax's vision CNN PPO network, and asymmetric actor/critic state keys.

Static contract validation works on any supported machine:

```bash
simrig validate-env examples/vision_cartpole.py --vision
```

Runtime rendering and training require a JAX-visible CUDA GPU plus MuJoCo Warp:

```bash
simrig validate-env examples/vision_cartpole.py --runtime --vision
simrig smoke examples/vision_cartpole.py --steps 5
simrig train examples/vision_cartpole.py --preset smoke \
  --output runs/vision-cartpole-smoke
simrig eval runs/vision-cartpole-smoke/policy.params \
  --env examples/vision_cartpole.py --steps 1000
simrig preview runs/vision-cartpole-smoke/policy.params \
  --env examples/vision_cartpole.py --port 8765
```

To run the published 1,000-step reference checkpoint instead of training one,
install `.[playground,hf]` and use:

```bash
simrig eval hf://ssuleiman/simrig-vision-cartpole/policy.params \
  --env examples/vision_cartpole.py --hf-revision v1 --steps 1000 --seed 0
simrig preview hf://ssuleiman/simrig-vision-cartpole/policy.params \
  --env examples/vision_cartpole.py --hf-revision v1 --auto-reset --port 8765
```

The policy completed the full 1,000-step horizon without termination for seeds
0 through 4 in its recorded Python 3.11/CUDA runtime. Native pixel rollout
requires a JAX-visible CUDA GPU and MuJoCo Warp. Use
`--allow-runtime-mismatch` only for a qualitative preview when the local Python
or package versions differ from the recorded configuration.

## Independent evaluation (planar arm)

[`phase1/`](phase1/README.md) is a dependency-free analytic two-link arm. It
tests SimRig's evaluator, predicates, and ranking — not a training backend.
The valid controller passes; a high-reward trap fails forbidden-contact
predicates and ranks last.

```bash
simrig task validate examples/phase1/planar_reach_task.json
simrig task freeze examples/phase1/planar_reach_task.json \
  --output /tmp/planar-reach.frozen.json
simrig eval-suite examples/phase1/checkpoints/valid.json \
  --contract /tmp/planar-reach.frozen.json --suite promotion
```
