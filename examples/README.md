# Examples

Tiny assets for testing SimRig’s **custom env CLI path** — not product tasks.

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
