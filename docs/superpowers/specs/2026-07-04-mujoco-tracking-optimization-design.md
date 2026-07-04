# MuJoCo Tracking Optimization Design

## Goal

Reduce avoidable tracking transients, make tracking-controller behavior configurable, prevent observer-arm conditioning from throttling the executor arm, and export enough diagnostics to distinguish path-discretization error from actuator/model limitations.

## Scope

This change applies to scenario-native tracking runs launched through `scripts/run_scenario.py`. Legacy task YAML files and legacy `runtime/mujoco_tracking_runtime.py` remain unchanged. Navigation, wiping, and engine-cleaning controllers retain their current defaults.

## Design

### Tracking metrics

`WaypointTrackingController` will report both the error evaluated before waypoint scheduling and the error to the command target after scheduling. It will also mark waypoint transitions, completed waypoints, approach samples, and task completion. The final completion command retains tracking metadata instead of returning an anonymous zero command.

Artifacts will preserve `tracking_error_m` as the commanded-target error for compatibility and add achieved-waypoint error, transition flags, completion flags, per-arm singularity protection, saturation scale, tendon target error, and peak actuator force. Summary metrics will use achieved errors for waypoint-accuracy statistics while retaining command-error statistics.

### Smooth approach and feedforward

Tracking trajectories may prepend a quintic smooth-step interpolation from the straight executor tip to the first requested waypoint. The original trajectory remains identifiable through an approach mask and source waypoint index.

The tracking target carries an optional world-frame feedforward velocity. For each non-final waypoint this velocity follows the direction to the next waypoint at a configured speed. Position feedback and feedforward are summed and clipped to a configured Cartesian target-speed limit.

### Configuration

Scenario `task.tracking_control` exposes:

- approach sample count;
- executor and observer position gains;
- feedforward and maximum Cartesian speed;
- whole-body task and regularization weights;
- singularity thresholds, damping limits, and minimum velocity scale;
- fixed-base per-arm singularity decoupling.

Defaults preserve existing behavior except where the single/dual MuJoCo tracking scenarios explicitly enable approach, feedforward, and per-arm decoupling.

### Per-arm singularity protection

For a fixed base, the weighted Jacobian is block-separated by arm. The solver will calculate damping and velocity scaling independently for each arm slice, apply block-diagonal damping, and scale each arm’s solved velocity independently. A movable-base assembly retains the global protection because the shared base couples arm tasks.

The existing whole-body singularity report remains available. New per-arm reports reveal which arm caused damping or speed reduction.

### Diagnostics

Each recorded step exports:

- per-arm singularity minimum value, condition number, damping, and velocity scale;
- backend saturation common scale;
- norm and maximum absolute value of tendon target minus actual displacement;
- peak absolute actuator force;
- pre-scheduling achieved error, commanded-target error, transition, approach, and completion flags.

No actuator gains, force limits, or tendon limits are automatically increased. Those parameters require evidence from the new diagnostics.

## Error handling and compatibility

All new numeric configuration values are validated as finite and within their physical domains. Approach generation requires at least two samples when enabled. Feedforward is zero at the final waypoint and when disabled. Existing scenario files without `tracking_control` load with legacy-compatible defaults.

## Testing strategy

Tests cover configuration parsing and rejection, smooth approach generation, waypoint metric semantics, final metadata retention, feedforward clipping, fixed-base per-arm singularity decoupling, and artifact flattening. Per user instruction, tests are authored but not executed automatically.

