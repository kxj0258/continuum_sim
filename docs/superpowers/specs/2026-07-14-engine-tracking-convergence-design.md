# Engine Tracking Convergence Design

## Objective

Repair the tracking-controller `NameError` and migrate single/dual MuJoCo
engine tracking to the proven MuJoCo tracking low-level baseline without asking
the continuum arms to span the full world-space distance from the origin to the
engine.

## Root cause

`WaypointTrackingController.compute_command()` publishes
`waypoint_scheduler_paused` but never defines the corresponding local variable.
The subsequent GLFW warning is cleanup noise after the controller exception.

The fixed-base engine tracking scenarios also target points approximately
0.47--0.52 m from the initial base while a straight three-segment arm is only
approximately 0.12 m long. Removing the exception alone therefore cannot make
the task reachable.

## Approved architecture (scheme A)

Engine tracking uses an explicit two-stage upper-level controller:

1. **Base approach**: translate the prescribed-twist mobile base so the current
   executor tip aligns with the first trajectory waypoint. Keep the current base
   orientation and command zero tendon rates for every arm.
2. **Fixed-base trajectory tracking**: solve the same time-parameterized
   Cartesian tracking problem and use the same shared
   `mujoco_tracking_low_level.yaml` profile as the proven MuJoCo tracking
   scenarios. Build the solver with a fixed-base copy of the assembly and force
   `base_twist_world` to zero on every tracking command.

The second stage deliberately follows the established `engine_navigation`
reaction-isolation pattern. The low-level solver cannot spend base degrees of
freedom, and the prescribed MuJoCo base receives zero velocity while tendons
move. Tendon reaction therefore does not become a commanded base wobble.

## Configuration

The tracking-control block gains an opt-in `stage_mobile_base` flag and four
base-approach parameters:

- `base_position_gain: 1.5`
- `base_orientation_gain: 2.0`
- `base_position_tolerance_m: 0.005`
- `base_orientation_tolerance_rad: 0.035`

Both engine tracking scenarios opt in, select the mobile assembly, use time
tracking for 80 s, and use no arm-generated approach samples. The base-approach
stage replaces the 40-sample arm approach used by the local MuJoCo square task.
Runtime capacity is increased to 5000 control steps so base placement does not
consume the trajectory interval.

## Single and dual behavior

The executor upper-level target and lower-level solve are identical between
single and dual engine tracking. In the dual task, the observer continues to
use collision avoidance with 18 mm activation, 20 mm release, gain 1.2, and no
dedicated avoidance-speed cap. Both arms receive zero tendon rates while the
base moves; after base arrival, the observer uses the same fixed-base tendon
control path as the dual MuJoCo tracking baseline.

## Diagnostics and completion

Base-approach commands report the stage and target pose errors, while executor
tracking error is `NaN` so relocation is excluded from arm trajectory tracking
statistics. Tracking-stage commands retain the normal tracking metadata and add
the staged-engine fields. Completion occurs only after the 80 s tracking
trajectory elapses.

## Constraints and risks

- No automatic test, validation, lint, format, build, install, viewer, or
  simulation command may be run.
- Base relocation is an open-scene translation to the first waypoint; it does
  not plan a collision-free base path.
- The base target assumes the arm shape remains unchanged while all tendon rates
  are zero. Residual MuJoCo tendon settling can create a small initial offset,
  which the fixed-base tracking servo must remove.
- Equivalent controller structure and parameters do not guarantee the same
  numerical error as an empty-scene square because the engine trajectory,
  geometry queries, and reachable Jacobian directions differ.
