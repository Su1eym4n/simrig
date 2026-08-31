"""Preview the learned reference with the exact frozen scenario target."""

import argparse
from pathlib import Path

from examples.learned_reach.evaluator import ReachMeasurements
from simrig.preview import serve_policy_preview
from simrig.task_contract import load_task_contract


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--scenario", default="heldout_nominal")
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--port", type=int, default=8768)
    args = parser.parse_args()
    contract, _ = load_task_contract(args.contract, require_frozen=True)
    scenarios = contract["evaluation"]["suites"]["promotion"]["scenarios"]
    scenario = next(item for item in scenarios if item["name"] == args.scenario)
    if args.seed not in scenario["seeds"]:
        parser.error("Choose a seed declared in the selected frozen scenario")
    if contract["name"] != "learned-demo-reach-reference":
        parser.error("This preview is for the unchanged learned reaching reference")

    def apply_scenario(env, state):
        return ReachMeasurements(env, {"parameters": scenario["parameters"]}).reset(state)

    print(f"Preview: {args.scenario}, initial seed {args.seed}; target {scenario['parameters']['target']}", flush=True)
    serve_policy_preview(
        args.policy, env_name=str(Path(__file__).resolve().parents[1] / "demo_reach.py"),
        seed=args.seed, port=args.port, auto_reset=False, reset_transform=apply_scenario,
    )


if __name__ == "__main__":
    main()
