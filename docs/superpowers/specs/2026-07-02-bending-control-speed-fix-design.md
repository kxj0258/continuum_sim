# Bending-Control Speed Fix Design

## Goal

Remove unintended numerical throttling from single-arm executor and dual-arm
executor tracking while retaining bending-space tendon compatibility.

## Confirmed Changes

### Dual-arm clearance

Change the coordinated controller's minimum inter-arm distance from `0.025 m`
to `0.010 m`.

Fixed mount/root samples must not participate in arm-arm collision selection.
The first centerline sample of each arm is fixed by the mount and has no bending
Jacobian. Selecting those samples currently creates an impossible avoidance
task.

After selecting movable samples, an avoidance task is actionable only when:

- distance is below the configured minimum distance;
- its relative bending Jacobian has a norm above the numerical rank tolerance;
- the requested separation velocity is positive.

If no actionable avoidance task exists, observer tracking remains enabled.

### Singularity handling

The singularity report continues to record matrix rank and whether the complete
task matrix is full rank. Damping and global velocity scaling, however, use the
smallest singular value above `rank_tolerance`.

Structural zero directions, such as axial tip velocity at a straight
bending-only arm or a zero collision row, must not force the entire controllable
command to the minimum `0.05` scale.

If a matrix has no positive singular values, it remains fully damped and uses
the minimum velocity scale.

### Bending-space regularization

Arm regularization penalizes physical compatible tendon effort:

```text
sqrt(lambda_tendon) C_b b_dot
```

It no longer penalizes curvature rate directly with
`sqrt(lambda_tendon) I`. Base regularization remains an identity-weighted base
twist penalty.

This preserves the existing configuration value while giving it the same
physical meaning it had before the optimization variables changed from tendon
rate to bending rate.

### Trajectory placement

Do not change the current square trajectory z placement. Its first waypoint has
both lateral and axial error. Once structurally zero singular directions no
longer throttle controllable lateral motion, the arm can leave the straight
configuration and acquire axial sensitivity naturally.

## Diagnostics

Retain existing rank, damping, velocity-scale, residual, command-rate, and
mapping diagnostics. Add no new runtime dependency.

An inactive or rejected inter-arm collision task should be distinguishable from
an active task in controller metadata through the existing
`observer_collision_active` and `observer_tracking_active` fields.

## Test Source Updates

Add or update tests for:

- rank-deficient matrices using their smallest positive singular value;
- all-zero matrices retaining maximum damping and minimum scale;
- tendon-space regularization block construction;
- fixed root samples not producing an avoidance task;
- the `0.010 m` minimum distance default;
- observer tracking remaining active when no actionable collision exists.

Per project instruction, implementation will not automatically run tests,
builds, linters, formatters, viewers, or simulations.

## Risks

- Removing global throttling can substantially increase tendon commands. The
  existing compatible common-scale rate and displacement limits remain the
  final safety boundary.
- A `10 mm` arm-arm distance may be smaller than real tool or tube radii; this
  value is a controller setting requested for the current model, not a
  validated physical safety clearance.
- Collision avoidance remains observer-only and cannot move fixed mounts.
