"""Run the frozen reference matrix and controls; never train or change a task.

Run from the repository root:
  python -m examples.learned_reach.verify --policy RUN/policy.params \
    --contract task.frozen.json --output reports/learned-reach
"""

import argparse
from pathlib import Path
import statistics

from simrig.evaluation_suite import run_evaluation_suite
from simrig.io import save_json, unique_report_path
from simrig.ranking import rank_checkpoints


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--suite", default="promotion", choices=("calibration", "promotion", "holdout"))
    args = parser.parse_args()
    controls = Path(__file__).parent / "controllers"
    reports = []
    summaries = []
    for name, artifact, expected in (
        ("ik", controls / "ik.py", True),
        ("zero", controls / "zero.py", False),
        ("random", controls / "random.py", False),
        ("near_miss", controls / "near_miss.py", False),
        ("learned", args.policy, True),
    ):
        report = run_evaluation_suite(artifact, contract_path=args.contract, suite_name=args.suite)
        path = unique_report_path(name, root=args.output)
        save_json(path, report)
        records = report["records"]
        valid = all(
            record.get("terminal_reason", {}).get("category") in {"success", "task_failure"}
            and record.get("metrics", {}).get("control_steps", 0) > 0
            for record in records
        )
        summary = {
            "name": name, "report": str(path.resolve()), "passed": report["passed"],
            "expected_pass": expected, "valid_measurements": valid,
            "successes": sum(record.get("task_success") is True for record in records),
            "episodes": len(records),
            "median_episode_time_sec": statistics.median(
                record["metrics"]["simulation_time_sec"] for record in records
            ) if valid and records else None,
        }
        print(f"{name}: {summary['successes']}/{summary['episodes']} successful, gate={report['passed']}, valid={valid}", flush=True)
        reports.append(report)
        summaries.append(summary)
    ranking_path = unique_report_path("ranking", root=args.output)
    ranking_error = None
    try:
        save_json(ranking_path, rank_checkpoints(reports))
    except ValueError as exc:
        # Preserve the overall failure even if code/artifact identities changed
        # between reports. Never weaken ranking's comparability check.
        ranking_error = str(exc)
    passed = ranking_error is None and all(
        row["passed"] == row["expected_pass"] and row["valid_measurements"] for row in summaries
    )
    summary_path = unique_report_path("verification", root=args.output)
    save_json(summary_path, {
        "passed": passed, "reports": summaries,
        "ranking": str(ranking_path.resolve()) if ranking_error is None else None,
        "ranking_error": ranking_error,
    })
    print(f"Verification: {summary_path.resolve()}", flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
