# dual_mujoco_wiping control migration

This scenario now uses the same scenario-native control layering as the
dual-arm MuJoCo navigation and tracking scenarios.

## Control chain

1. `WipingController` owns the wiping task state, raster waypoint phase, contact
   distance estimate, and force error estimate.
2. Executor position servo remains the primary task. During contact force
   control, its tracking Jacobian is projected onto the board tangent plane, so
   it owns path following while leaving the normal direction available.
3. Hybrid force-position wiping adds a closed-loop `ContactTaskIntent` instead
   of only moving the waypoint normal to the board. The low-level controller
   turns this into an executor normal-velocity whole-body task.
4. `CoordinatedTrackingController` solves executor tangent position first, then
   the normal force-control task in the position task nullspace.
5. The observer arm keeps the navigation-style priority order: inter-arm
   collision avoidance first, then scene/board avoidance in the remaining
   nullspace.
6. `MujocoSystemBackend` still receives tendon-rate commands and converts them
   to absolute MuJoCo tendon position targets through the
   `mujoco_tendon_position` execution adapter.

## Scenario parameters

`configs/scenarios/dual_mujoco_wiping.yaml` now uses `wiping_path` instead of
manual waypoints. The path mirrors the old task-level raster wiping setup:

- `surface_id: board_surface`
- `patch_id: center_patch`
- `line_count: 5`
- `samples_per_line: 30`
- `approach_offset_m: 0.005`
- `contact_offset_m: -0.0025`

The tracking mode is tolerance-based waypoint tracking, with
`max_steps_per_waypoint: 500`, matching the dual tracking style.

The closed-loop force task is controlled by:

- `target_normal_force_n`
- `normal_force_gain`
- `force_proxy_stiffness_n_m`
- `max_normal_velocity_m_s`
- `force_control_weight`

The normal velocity command is clipped by `max_normal_velocity_m_s` before it
is passed into the nullspace force task.
