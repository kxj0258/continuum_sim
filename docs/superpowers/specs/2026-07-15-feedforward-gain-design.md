# Feedforward Gain Design

## Objective

Add a dimensionless `feedforward_gain` to the shared task-space control path so
MuJoCo tracking, navigation, engine tracking, and other position-controlled
tasks can tune trajectory feedforward independently from Cartesian position
feedback.

The default value is `1.0`, preserving current behavior. A value of `0.0`
disables trajectory feedforward, values between zero and one attenuate it, and
values greater than one amplify it.

## Semantic boundary

`CartesianTaskIntent.feedforward_velocity_world` currently carries two
different kinds of velocity:

1. In `control_mode="position"`, it is genuine trajectory or waypoint
   feedforward and is scaled by `feedforward_gain`.
2. In `control_mode="velocity"`, it is the complete direct velocity command
   and is not scaled.

This preserves the authority of engine-cleaning velocity control and navigation
velocity overrides such as CBF avoidance. If direct velocity scaling is needed
later, it must use a separately named parameter.

## Control law

For executor position control, the coordinated target velocity is

```text
v_target = Kp * (p_target - p_measured) + Kff * v_feedforward
```

For executor velocity control, the servo target remains the measured position,
so the position-error term is zero and the direct velocity is forwarded without
the feedforward gain:

```text
v_target = v_direct
```

The existing optional Cartesian target-speed limit remains downstream of this
sum and therefore applies to the final requested velocity exactly as before.

## Architecture

`ScenarioTrackingControlConfig` owns the public `feedforward_gain` field. The
scenario loader merges values in the existing order:

```text
code default < low_level_control profile < task.tracking_control override
```

`application._tracking_coordinated_config()` transfers the resolved value into
`CoordinatedTrackingConfig`. `UnifiedLowLevelController` is the semantic
boundary that still knows `control_mode`; it scales the executor velocity only
for position mode before constructing `CoordinatedTrackingTarget`.

This centralized implementation covers timed trajectories, waypoint tracking,
navigation, staged engine tracking, and wiping without duplicating logic in
each upper-level controller.

## Configuration exposure

Both shared low-level profiles explicitly contain `feedforward_gain: 1.0`:

- `configs/control/mujoco_tracking_low_level.yaml`
- `configs/control/spatial_low_level.yaml`

The spatial profile is included because some MuJoCo wiping and engine-navigation
scenarios currently reference it. Scenario-local overrides remain supported via
`task.tracking_control.feedforward_gain`.

## Validation and diagnostics

The gain must be finite and non-negative. Invalid values fail during scenario
configuration construction.

Existing raw task-intent velocity metadata is retained. Commands additionally
report:

- `executor_feedforward_gain`
- `executor_scaled_feedforward_velocity_world`

The NPZ artifact writer exports both fields for every command, allowing raw
feedforward, scaled feedforward, and final target velocity to be compared.

## Compatibility and risks

- The default and committed profile value `1.0` preserve numerical behavior.
- A gain of zero leaves position feedback active; it does not stop the task.
- Velocity-mode commands intentionally ignore this gain.
- Scaling occurs before target-speed limiting, so a saturated final target may
  show less than the expected proportional change.
- No tests, validation, lint, format, build, installation, or simulation are to
  be run automatically for this change.
