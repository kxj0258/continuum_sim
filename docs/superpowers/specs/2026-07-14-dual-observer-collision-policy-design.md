# Dual Observer Collision Policy Design

## Goal

Give every dual-arm scenario the same explicit observer inter-arm collision
policy as `dual_mujoco_tracking`, with updated gain and hysteresis parameters
and no avoidance-specific speed cap.

## Shared Effective Policy

Every dual scenario uses `observer_control_mode: collision_avoidance` and the
following effective policy:

- minimum/diagnostic distance: 10 mm;
- avoidance activation distance: 18 mm;
- critical/diagnostic distance: 8 mm;
- release margin: 2 mm, producing a 20 mm release threshold;
- avoidance gain: 1.2;
- no avoidance-specific speed limit;
- six centerline samples per segment;
- observer collision task weight: 80.

The policy remains a soft observer-only velocity task. The 10 mm and 8 mm
values do not become hard constraints, executor freeze, or automatic stop
conditions.

## Speed-Limit Semantics

`CoordinatedTrackingConfig.inter_arm_max_avoidance_speed_mps = None` will mean
that collision avoidance has no task-specific speed cap. It will no longer
fall back to the general Cartesian `max_target_speed_mps`, allowing the
collision policy to remain identical across protected and actual-anchored
low-level profiles.

An explicit numeric avoidance limit will continue to clip the requested
separation speed.

## Engine Navigation

Engine navigation has a separate nested observer specification. Its collision
fields will be aligned with the common policy, its maximum avoidance speed will
be optional, and the staged controller will pass that value through instead of
discarding it. Base-motion phases continue to command zero tendon rates; the
collision policy applies during local executor-path and rejoin phases.

## Validation Constraint

Codex will not run tests, validation, lint, format, build, install, or
simulation commands. Manual scenario execution remains the user's validation
step.
