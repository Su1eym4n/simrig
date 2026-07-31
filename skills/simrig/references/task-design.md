# Task design

Use this reference before writing a custom scene, reward, observation, or
termination rule.

## Task contract

Capture the smallest contract that makes the request testable:

| Field | Decide |
|---|---|
| Behavior | What the robot must do |
| Command/goal | Fixed target or sampled distribution |
| Initial state | Nominal pose and randomization bounds |
| Scene | Ground, terrain, objects, targets, contacts, sensors |
| Success | Metric, threshold, duration, and evaluation horizon |
| Failure | Falls, forbidden contacts, joint limits, object loss, timeout |
| Generalization | Seeds and conditions not emphasized during training |

Do not block on cosmetic choices. Ask the user when choices change the learned
behavior, safety envelope, or meaning of success.

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
- mild perturbations or dynamics changes when robustness matters;
- success rate;
- episode length and termination reason;
- task-specific error or outcome;
- qualitative visual review.

Keep evaluation conditions reproducible. Separate training-distribution results
from held-out robustness results.
