# Independent evaluation

SimRig separates optimization from acceptance. A training environment may expose
reward and convenient diagnostics, but promotion is decided by a frozen task
contract, a task-owned evaluator, generic predicates, and complete scenario and
seed coverage.

## Evaluator protocol

An evaluator is a Python file with a versioned declaration and one function:

```python
EVALUATOR_SPEC = {
    "name": "my-task-evaluator",
    "version": "1.0.0",
    "protocol_version": 1,
}


def evaluate(request):
    return {
        "total_reward": 12.5,
        "metrics": {"final_error": 0.01},
        "events": [
            {
                "kind": "signal",
                "name": "at_goal",
                "step": 30,
                "active": True,
            }
        ],
    }
```

The request is backend-neutral and contains `checkpoint`, `environment`,
`backend`, `suite`, `scenario`, `parameters`, `seed`, `max_steps`,
`task_contract_sha256`, and `evaluator_config`. The plugin owns simulator
construction and checkpoint loading. It should return observations reduced to
stable metrics plus the event/contact stream required by contract predicates.
It must not decide promotion from training reward.

SimRig hashes the evaluator declaration, configuration, and bounded local source
closure. The hash is embedded in suite reports and training manifests so reports
from different evaluator implementations cannot be ranked together.

## Events, contacts, and predicates

Events use the common fields `kind`, `name`, `step`, and `active`. Contacts also
use `body_a`, `body_b`, and optionally `force`. Contract predicates turn that raw
record into `task_success` and a machine-readable terminal reason.

The initial predicate vocabulary is deliberately small:

- `sustained`: require a named signal for consecutive steps;
- `forbidden_contact`: bound contacts between an unordered body pair;
- `event_count`: bound the number of named events;
- `sequence`: require an ordered event subsequence;
- `metric`: compare a task-owned numeric metric with a threshold.

Every terminal reason has a stable `category`, task/evaluator-specific `code`,
human-readable `message`, and optional `details`. Categories are `success`,
`timeout`, `forbidden_contact`, `safety_violation`, `invalid_state`,
`task_failure`, `incomplete`, `evaluator_error`, and `unknown`. If several
required predicates fail, safety and evaluator failures take precedence over a
generic task failure.

## Scenario-by-seed suites

Declare the evaluator, predicates, scenarios, fixed seeds, grouping, and
requirements in task-contract schema v2. Freeze it, then run the full matrix:

```bash
simrig eval-suite runs/RUN/policy.params \
  --contract task.frozen.json \
  --suite promotion \
  --output reports/policy-promotion.json
```

The command fails when a predicate, a metric threshold, or required
scenario/seed coverage fails. Its report contains raw episode records,
per-condition promotion summaries, contract/checkpoint/evaluator hashes,
reward probes, and a deterministic report hash.

Use explicit caps only for bounded diagnostics. A capped report remains failed
when it omits required promotion coverage:

```bash
simrig eval-suite POLICY --contract task.frozen.json --suite promotion \
  --max-scenarios 1 --max-seeds-per-scenario 1
```

For a live or incomplete run, a one-shot bounded check can inspect the newest
available checkpoint without starting a monitor or training process:

```bash
simrig eval-checkpoints runs/RUN --contract task.frozen.json \
  --suite promotion --max-checkpoints 1
```

The resulting report paths and evaluator identity are appended to
`run_manifest.json`. This is diagnostic evidence, not promotion, unless the
full contract matrix passes.

## Reward probes and checkpoint ranking

Probe combined reports for concrete high-reward failures:

```bash
simrig reward-probe reports/*.json
```

The probe compares reward distributions with independently derived success and
lists failed scenario/seed cells whose reward reaches the successful range.
Reward is diagnostic only.

Rank comparable checkpoints after independent suite execution:

```bash
simrig rank-checkpoints reports/checkpoint-*.json \
  --output reports/ranking.json
```

Reports must have the same contract hash, evaluator hash, and suite. Ranking is
lexicographic over promotion pass, worst-condition success rate, overall success
rate, and safety-failure rate. Reward is explicitly excluded.

## Contract migration and compatibility

Migration is explicit and always produces an editable draft. Review and freeze
it rather than silently changing an existing frozen artifact:

```bash
simrig task migrate task-v1.json --output task-v2.json
simrig task validate task-v2.json
simrig task freeze task-v2.json --output task-v2.frozen.json
```

Compatibility is purpose-specific:

```bash
simrig task compatibility OLD NEW --policy exact
simrig task compatibility OLD NEW --policy training_resume
simrig task compatibility OLD NEW --policy checkpoint_evaluation
simrig task compatibility OLD NEW --policy result_comparison
```

`exact` requires identical semantics. `training_resume` permits evaluation and
compute-budget edits but freezes the environment, behavior, interfaces, scene,
reset, episode, and outcomes. Checkpoint evaluation has a narrower execution
compatibility surface. Result comparison requires the task semantics to match;
suite/evaluator hashes still govern whether evaluation reports can be ranked.

## Acceptance example

[`examples/phase1`](../examples/phase1/README.md) contains a dependency-free
analytic planar arm. Its valid artifact reaches nominal and boundary targets
for fixed seeds, sustains the required hold, and avoids forbidden contact. A
second artifact receives much higher proxy reward while failing the physical
predicates. The gate rejects it and independent ranking puts it last.

## Current limits

Evaluator plugins are trusted local Python, not sandboxed processes. SimRig
does not yet provide backend adapters, distributed evaluator workers, a
columnar event store, or a persistent asynchronous checkpoint watcher. Source
closure and checkpoint-directory hashing are intentionally bounded. Tasks with
large observations, videos, or dense contact traces should write those assets
separately and return compact references and metrics.
