# Task design

Use this reference before writing a custom scene, reward, observation,
termination rule, environment, or training configuration.

## Mandatory Physical Success Definition

Natural-language goals are underspecified. An agent cannot guarantee that its
formalization captures the user's semantic intent. Its job is to draft a
physically testable interpretation, validate that interpretation against the
available model and scene, look for counterexamples, and obtain focused user
confirmation before freezing the task contract.

Inspection comes first. Use the exact robot and task scene, not a robot name or
generic description. Record the inspected model/source revision, authored
initial pose, relevant geometry and contact names, joint ranges, actuator
limits, timestep/control rate, and available sensors. Compilation or a short
step proves only that the model loads; it does not prove that the proposed task
or success thresholds are feasible.

Write a Physical Success Definition with every row resolved or explicitly
marked as an assumption:

| Field | Required decision |
|---|---|
| Behavior | Observable physical outcome, without reward language |
| Measured entities | Exact body, site, joint, object, geom/contact pair, or sensor names |
| Quantities | Position, orientation, velocity, force, distance, overlap, event, or other measured values |
| Units and frames | SI units where possible and the world/body/object/target frame for every vector or pose |
| Thresholds | Inclusive/exclusive bounds and tolerances, including numerical slack |
| Duration | Consecutive control ticks or physical time the conditions must hold |
| Horizon | Maximum control ticks and physical seconds before timeout |
| Reset | Initial state, target/command distribution, randomization, and predecessor/native states |
| Contacts | Allowed, required, and forbidden pairs; force/impulse threshold when relevant |
| Terminal precedence | Evaluator error/invalid state and safety failures before ordinary task failure or timeout |
| Scenarios | Nominal, boundary, perturbation, and held-out cases with fixed seeds |
| Controls | Zero/random negative controls, known-valid positive control, and deliberately exploitative control |
| Provenance | Frozen contract, model/scene/evaluator source, assets, runtime, and relevant configuration identities |

Use quantities the evaluator can actually observe. A camera-only deployed
policy may have less information than an independent evaluator; document that
separation. Do not define success from privileged state that does not exist in
simulation, from a visual impression with no measurement, or from reward.

### Necessary conditions versus optimization preferences

List necessary success conditions separately from preferences:

- A **necessary condition** must hold for physical success. Examples include a
  target pose tolerance, a required grasp/contact, a stable landing interval,
  or absence of forbidden contact. Encode it as evaluator evidence and a
  predicate or promotion requirement.
- An **optimization preference** helps learning or selects among successful
  behaviors. Examples include lower energy, smoother action, faster completion,
  stylistic gait, or extra clearance. It may shape reward, but it must not be
  silently promoted to failure or used as a substitute for success.

If violating a preference should make the episode fail, it is not merely a
preference. Reclassify it, define its physical measurement and threshold, and
ask for confirmation before freezing.

### Feasibility audit

Before accepting thresholds or distributions, check:

- joint position/velocity limits, coupled motion, collision geometry, and
  kinematic reachability of every nominal and boundary target;
- actuator type, control range, gear/force/torque authority, action scaling,
  control rate, and whether the horizon permits the requested transition;
- model timestep, action repeat, sensor/update rate, latency, noise, solver
  tolerance, and a realistic numerical margin around every threshold;
- which measurements are available from simulator state, contacts, or sensors,
  and which observations are available to the deployable policy;
- reset feasibility, penetrations, object stability, and whether required or
  forbidden contacts can be identified unambiguously.

A failed feasibility check is not a reward-design problem. Narrow the task,
change the model/scene, or ask the user to choose among meaningfully different
alternatives.

### False positives and controls

For every success condition, write at least one counterexample that a naive
metric would accept. Consider collapse satisfying a height target, brief
threshold crossings satisfying a hold, wrong-object or wrong-link contact,
object proximity without grasp, goal passage without stopping, reward farming,
leaving the workspace, timeout being labeled success, and non-finite values.

Define controls before trusting the evaluator:

- zero-action and seeded random-action artifacts must fail with expected,
  stable terminal categories;
- a known-valid controller, trajectory, or policy must pass the complete
  matrix; if none exists yet, record that evaluator calibration is unresolved
  and do not claim promotion readiness;
- a deliberately exploitative or visually plausible artifact must fail the
  physical predicates even when it earns high reward or looks convincing.

Controls validate the definition and evaluator, not the reward. Keep them
small, deterministic, and safe. Do not invent a project-specific expert merely
to make a positive control appear solved; obtain or author one only when its
semantics are understood.

### Confirmation and task contract

Map the confirmed definition into Phase 1 contract fields: behavior,
interfaces, scene contacts, reset, episode horizon, physical outcomes,
evaluator, predicates, scenario/seed suites, promotion requirements, and
compute limits. Keep an explicit list of unresolved assumptions.

Ask one focused confirmation question containing only decisions that change
the meaning of success, safety envelope, or feasibility. Do not freeze while
any such decision is unresolved. Confirmation makes the interpretation
reviewable and agreed; it does not let the agent claim perfect knowledge of the
user's intent. Cosmetic choices need not block progress.

## Locomotion

Define commanded linear and angular velocity ranges, terrain, gait constraints,
allowed contacts, and fall criteria. Distinguish:

- tracking a commanded velocity;
- moving as far or fast as possible;
- reaching a destination;
- traversing terrain.

Consider policy observations for base orientation and velocity, joint position
and velocity, previous action, command, phase/contact data, and terrain sensing.
Keep privileged terrain or dynamics information out of `state` when the real
policy cannot observe it.

Use interpretable reward metrics such as command-tracking error, uprightness,
height, foot slip, action rate, energy, impact, and joint-limit cost. Choose
weights only after agreeing on priorities. Prevent survival bonuses from
outweighing the requested motion.

## Posture and crouching

Define the target by base height, named joint targets, end-effector positions,
or a reference pose. Specify whether the robot must transition into the pose,
hold it, or track a height command.

Measure pose error, base height error, balance, contact stability, action rate,
and energy. Define success as remaining within tolerances for a duration.
Avoid rewarding a low base height if collapsing can satisfy it.

## Jumping

Define vertical versus directional jumping, desired takeoff or apex, landing
zone, landing pose, and whether repeated jumps are required. Represent phases
using observable state or task state when necessary.

Measure takeoff, airborne apex, horizontal displacement, landing location,
upright landing, post-landing stability, impact, and forbidden contacts.
Require a complete takeoff-to-stable-landing event for success. Do not equate
brief base height with a successful jump.

## Manipulation and reaching

Define the controlled end effector, target distribution, object properties,
allowed contacts, success tolerance, and whether grasping or placement is
required. Put target and object state in observations only when it would be
available to the deployed policy.

Sample targets inside the robot's reachable workspace. Respect kinematic
constraints such as a planar arm's motion plane; either restrict the target
distribution or explicitly change the robot/task. Keep rendered target geometry
synchronized with the target stored in environment state.

Measure end-effector or object error, grasp/contact state, control smoothness,
collisions, joint limits, and sustained success. Avoid sparse-only rewards
unless the exploration strategy can plausibly discover success.

## Custom scenes

Create a task-owned `scene.xml` rather than mutating a vendor model when
possible. Include the robot model and add:

- terrain or world geometry;
- props and movable bodies;
- target sites or mocap bodies;
- contact exclusions or pairs;
- sensors needed by observations or metrics;
- cameras and task-relevant keyframes.

Preserve asset paths relative to the included model. After each scene change,
run `simrig inspect-model SCENE --save-report` and `simrig view-model SCENE`.
Compilation catches structural issues; visual review catches scale, placement,
camera, collision, and initial-pose errors.

## Evaluation matrix

Specify at least:

- multiple random seeds;
- nominal conditions;
- boundary commands or targets;
- perturbations or dynamics changes with declared magnitude;
- held-out targets, layouts, commands, or initial states not emphasized during
  reward tuning or training;
- success rate;
- episode length and termination reason;
- task-specific error or outcome;
- zero/random negative controls, a known-valid positive control, and a
  deliberately exploitative behavior;
- qualitative visual review.

Keep evaluation conditions reproducible. Separate training-distribution results
from held-out robustness results. Run every required scenario/seed cell with
`simrig eval-suite`; missing coverage must fail promotion. Run controls through
the same evaluator and predicates. Use `simrig reward-probe` across positive
and exploitative reports so high-reward physical failures have a successful
reward baseline for comparison.

Custom environments should put a 0/1 `success` term in `state.metrics` and may
declare `SUCCESS_SPEC` (metric, threshold, mode) so `simrig eval` can report
`task_success`. Rollout completion is not success.
