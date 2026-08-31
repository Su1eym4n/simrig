# Recorded learned reaching result

The learned policy passes the frozen arrival task, but the reference **fails its baseline-discrimination check**: random actions pass every case too. This is an end-to-end integration reference and a useful diagnostic of a weak task definition, not evidence of a challenging manipulation skill.

Run on 2026-08-31 using macOS ARM CPU, Python 3.14.5, JAX 0.10.2, Brax 0.14.2, Playground 0.2.0, and MuJoCo/MJX 3.10.0. Runtime validation and a 10-step smoke passed. A 4,160-step PPO smoke preceded the corrected policy: seed 5, 32 environments, batch size 32, 180,000 requested / 184,320 actual training steps. Its recorded run took about 138 seconds, within the frozen run budget.

## Independently measured outcomes

Each suite contains 12 fixed scenario/seed cases. Success requires native-MuJoCo-measured end-effector distance strictly below 5 cm at a post-step tick, within four seconds. All measurements were valid; there were no evaluator or invalid-state failures.

| Artifact | Development successes | Holdout successes | Development median episode time | Holdout median episode time |
|---|---:|---:|---:|---:|
| learned | 12/12 | 12/12 | 0.09 s | 0.11 s |
| ik | 12/12 | 12/12 | 0.10 s | 0.10 s |
| zero | 5/12 | 2/12 | 4.00 s | 4.00 s |
| random | 12/12 | 12/12 | 0.43 s | 0.22 s |
| near_miss | 9/12 | 6/12 | 0.09 s | 2.05 s |

Time is simulated episode duration, including four-second timeouts for failures. It is diagnostic only and is not a promotion requirement. The learned policy arrives faster than random actions in these samples, but both satisfy the current success gate. No ranking tie was reinterpreted as superiority, and no threshold was tightened or relaxed after observing results.

Both `verify --suite promotion` and `verify --suite holdout` exit **1**, because random was expected to fail the complete suite. The reports and ranking are still saved. The learned policy itself passes both independent gates. Zero and the 9 cm near-miss controller fail the full suites; some of their episodes legitimately cross the target and count as successes.

## Selection, failures, and limits

Two earlier policies trained against the old environment achieved 11/12 development successes. Independent forward kinematics exposed a stale-site-position bug: one episode received training success while its actual end-of-tick error was about 6.16 cm. The environment now refreshes derived kinematics after integration. The reward formula and 5 cm threshold remain unchanged. Old checkpoints are rejected against the changed environment source. Failed runs/reports remain in the local run directory.

The corrected checkpoint was selected before consulting holdout seeds 1001–1012. No training, policy selection, target/seed filtering, or task changes followed the holdout result. Final report regeneration after an unrelated runtime guard retained the same checkpoint and outcomes; it is not additional independent evidence.

The two fixed targets remain inside the training distribution. A few reset seeds and a single training seed establish neither out-of-distribution robustness nor convergence reliability. Holding, smoothness, contact safety, perception, GPU vision, and hardware deployment were not evaluated. The shared runtime currently supports `action_repeat=1`; other rates are explicitly rejected.

To demonstrate a more useful skill, first agree on a stronger physical task and evaluation distribution, then freeze a new contract and reserve fresh final evaluation cases. More training alone cannot make this existing success gate distinguish learned control from random actions.

## Artifacts and verification

Commands are in [README.md](README.md). [results.json](results.json) records compact outcomes, runtime versions, and content identities. Full local artifacts are under `runs/learned-reach-reference-20260831/` (ignored by Git):

- `task-v2.frozen.json`: the exact frozen task used by the corrected run.
- `corrected/`: final parameters, Orbax checkpoints, configuration, provenance, and training metrics.
- `final-promotion/` and `final-holdout/`: five complete reports each, rankings, and failing reference-verification summaries.
- Earlier `trained/`, `continued/`, and evaluation directories: retained failed experiments.

Policy SHA-256: `f0d58dd7f0536828788d4c4f285764683c3e0de2bd15138413db25498fe59f58`.
Frozen contract SHA-256: `175d5c14dc28dfec6e2d1b915552522c7e6c2d761edd107b3bcddc9b702ccfe3`.

The full local regression run passed **186 tests** with the corrected checkpoint supplied through `SIMRIG_TEST_POLICY`, including final/Orbax inference parity, preview/eval parity, independent measurements, numerical failures, incomplete evidence, report hashes, and seed/identity coverage. One upstream JAXopt deprecation warning remained. Wheel/source builds and Twine checks passed. CI now runs the learned-policy integration on its tiny checkpoint; that check tests execution, not convergence.

The browser preview was checked on development scenario `heldout_nominal`, seed 101, target [0.25, 0, 0.22]. It reached in nine control ticks (0.18 s), with no renderer error. The native desktop viewer was refactored to share the same runtime but was not opened.
