# Navigation Quaternion Pose Servo

`task.type: navigation` can enable waypoint pose servo:

```yaml
task:
  pose_servo:
    enabled: true
    orientation_source: explicit_directions
    waypoint_directions_world:
      - [1.0, 1.0, 1.0]
      - [1.0, -1.0, 1.0]
      - [-1.0, -1.0, 1.0]
    orientation_tolerance_rad: 3.2
    roll_reference_world: [0.0, 0.0, 1.0]
  tracking_control:
    stage_mobile_base: true
    base_approach_standoff_m: 0.030
    base_approach_z_bias: 1.0
    task_space_servo:
      orientation_gain: 2.0
      max_angular_speed_rad_s: 1.0
    tendon_command:
      executor_orientation_tracking_weight: 20.0
```

- `orientation_source: explicit_directions` uses `waypoint_directions_world`
  as the target direction of the executor tip local `+Z` axis. Each direction
  is converted to a normalized quaternion with `roll_reference_world`.
- Explicit `waypoint_orientations_world_wxyz` is still supported. It cannot be
  combined with `waypoint_directions_world`.
- `orientation_source: nearest_clearance` queries the nearest structured-scene
  clearance for each waypoint and uses `-clearance.normal` as the tip look
  direction.
- Staged mobile-base navigation is per waypoint: first translate the base, then
  fix the base and servo the executor with tendons. The base target keeps the
  current base rotation, so navigation uses only XYZ translation.
- For a pose target direction `d = [dx, dy, dz]`, base approach places the
  current tip near `waypoint + standoff * normalize([-dx, -dy, z_bias])`.
  This starts the tip on the side from which continuum bending can rotate the
  tip toward the requested pose while preserving usable local workspace.
- If waypoint completion should temporarily ignore orientation reachability, set
  `pose_servo.orientation_tolerance_rad` larger than `pi`, for example `3.2`.
  The orientation task still participates in tendon solving, but waypoint
  advancement is governed by position error.
- The observer arm `look_at_executor_tip` uses quaternion look-at servo. In
  collision-avoidance mode, the look-at task is projected through the avoidance
  nullspace so it does not override the primary collision-avoidance task.
- MuJoCo structured-scene targets can be visualized with a green sphere and a
  small direction arrow. With `nearest_clearance`, the arrow direction is
  `-clearance.normal`; explicit direction targets are converted into waypoint
  pose targets for control.
