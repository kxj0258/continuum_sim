# MuJoCo Low-Level Baseline Design

## Goal

Use one explicit low-level profile for the MuJoCo tracking, engine tracking,
MuJoCo navigation, and engine cleaning scenarios. The profile must preserve the
effective control behavior already restored for `single_mujoco_tracking` and
`dual_mujoco_tracking`.

## Shared Profile

Create `configs/control/mujoco_tracking_low_level.yaml` from the complete
low-level parameter set currently embedded in the MuJoCo tracking scenarios.
The profile uses position gain `1.0`, SVD projection, tendon regularization
`0.8`, maximum damping `0.1`, and disables target-speed, solver-rate, and
backend tendon protection. MuJoCo therefore uses the `actual_anchored` tendon
target mode.

## Scenario Boundaries

The following scenarios reference the new profile:

- `single_mujoco_tracking.yaml`
- `dual_mujoco_tracking.yaml`
- `single_engine_tracking.yaml`
- `dual_engine_tracking.yaml`
- `single_mujoco_navigation.yaml`
- `dual_mujoco_navigation.yaml`
- `single_engine_cleaning.yaml`
- `dual_engine_cleaning.yaml`

Task-specific upper-level behavior remains unchanged. Tracking retains its
trajectory schedule, navigation retains waypoint/clearance behavior, and
engine cleaning retains its task-space position/force controller and TCP speed
limits. Dual observer behavior remains independent from the executor. In
`dual_mujoco_tracking`, observer collision avoidance still activates below
18 mm.

## Documentation And Validation

README control-profile descriptions will identify the new baseline and explain
that the eight scenarios share it. Codex will not run tests, validation, lint,
format, build, install, or simulation commands. Scenario commands are listed
for manual validation only.
