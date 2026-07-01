# Mobile Base 6D Control

> This document records the earlier base scaffold. The authoritative command
> contract is now `docs/coordinate_conventions.md`: world-frame 6D twist only,
> integrated as a prescribed pose with calibration limits supplied by assembly
> configuration.

## Coordinate Chain

`T_world_tip = T_world_mobile_base * T_mobile_base_mount * T_mount_tip`

- `T_world_mobile_base` is the 6D pose of the mobile base in the MuJoCo world frame.
- `T_mobile_base_mount` is the fixed mount transform from the base body to the continuum arm root.
- `T_mount_tip` is the local arm FK result.

When `base_pose` is identity and mount pose is identity or the project default, the existing single-arm pose chain remains compatible.

## Configuration

The mobile-base scaffold lives in `configs/robots/mobile_base_pose.yaml`.

- `mobile_base.pose` uses `position_m` and `quat_wxyz`.
- `mobile_base.limits` defines translation, Euler-angle, and velocity bounds.
- `mobile_base.manual_control` defines coarse and fine UI step sizes.
- `mobile_base.visualization` defines the optional MuJoCo base box.
- `mounts.arm_mount` defines the default `mobile_base -> continuum_arm_root` transform.

`src/continuum_sim/model/mount_frame.py` keeps backward compatibility with the legacy single `mount:` shape while also supporting the new `mounts:` mapping.

## BasePose

`src/continuum_sim/model/base_pose.py` extends `Pose6D` with:

- `identity()`
- `from_dict()` with `position` / `position_m`, `quat` / `quat_wxyz`, and `rpy_deg` / `rpy_rad`
- `as_matrix()` / `to_transform()`
- `inverse()`
- `compose()`
- `transform_point()`
- `transform_vector()`
- `transform_pose()`
- `to_dict()`

Quaternion order is consistently `[w, x, y, z]`.

## World Kinematics

`src/continuum_sim/kinematics/world_kinematics.py` now exposes a slightly clearer world-frame API:

- `compose_base_mount_pose(...)`
- `compute_world_tip_pose(...)`
- `transform_pose_to_world(...)`
- `transform_centerline_to_world(...)`
- `compose_world_tip_pose(...)` remains as a compatibility alias

The centerline helper still applies batched `N x 3` transforms.

## MuJoCo Visualization

`src/continuum_sim/scenes/scene_builder.py` now supports an optional `mobile_base` wrapper body when a `mobile_base_config_path` is provided through MuJoCo config loading.

The generated wrapper adds:

- `<body name="mobile_base" ...>`
- `<geom name="mobile_base_box" type="box" ...>` when visualization is enabled
- `<site name="mobile_base_frame" />`
- mount marker sites derived from the loaded `mounts`

Important MuJoCo note:

- MuJoCo `box` `size` is half-size.
- YAML `size_m` is treated as full size and divided by `2` during XML generation.
- The phase-2 base box is visualization-only with `contype="0"` and `conaffinity="0"`.

## Baseline Integration

The old fixed-base architecture is now treated as the local arm kernel:

- `motor -> tendon_delta -> q_arm -> local FK / Jacobian / controller`

The new primary outer chain is:

- `world -> mobile_base -> mount -> local arm`

当前 baseline 通过 scenario 入口覆盖：

- `configs/scenarios/single_analytic_*.yaml`
- `configs/scenarios/dual_analytic_tracking.yaml`
- `configs/scenarios/single_mujoco_*.yaml`
- `configs/scenarios/dual_mujoco_*.yaml`
- `configs/scenarios/*_engine_tracking.yaml`

Current behavior by layer:

- Pure PCC and offline motor-chain tools still solve the arm locally, then render in world coordinates through the base/mount context.
- MuJoCo viewer and tendon-debug flows now resolve runtime XML through the mobile-base wrapper when configured.
- MuJoCo trajectory tracking now keeps local target generation for the PCC controller side while comparing and displaying world targets/tip poses consistently.

## Manual Control

The current phase adds state and API scaffolding first, not a full viewer keymap implementation.

Recommended key layout for a later UI hookup:

- `W/S`: base `+x / -x`
- `A/D`: base `+y / -y`
- `R/F`: base `+z / -z`
- `I/K`: pitch `+ / -`
- `J/L`: yaw `+ / -`
- `U/O`: roll `+ / -`
- `B`: toggle base lock
- `H`: reset base pose
- `Shift`: fine/coarse step switch

The shared viewer state now reserves `base_state`, `base lock`, and `reset` handling hooks, but translational and rotational hotkeys are intentionally left for a later integration pass.

## Controller Interface

`src/continuum_sim/control/mobile_base_controller.py` introduces the lightweight base/whole-body scaffold:

- `MobileBaseCommand`
- `MobileBaseState`
- `WholeBodyCommand`
- `clip_base_twist(...)`
- `resolve_mobile_base_command(...)`
- `integrate_base_pose(...)`
- `reset_mobile_base_state(...)`
- `set_mobile_base_locked(...)`

The intended velocity convention is:

`base_twist = [vx, vy, vz, wx, wy, wz]`

And the future whole-body command shape is:

`u = [base_twist, arm_command]`

Current lock behavior:

- base locked: resolved twist becomes zero
- arm lock: flag is reserved in the command/viewer state layer

## Current Limitations

- This phase only establishes the 6D mobile-base support framework.
- Full whole-body IK is not implemented.
- Existing passive-viewer movement keys are not fully mapped to base motion yet.
- The current baseline still treats the base as static/locked during task execution.
- Advanced MuJoCo task flows are stabilized around the default `mujoco_actual` feedback path; full whole-body reformulation for every fallback controller path is still future work.
- No tests, simulation, lint, format, build, or install commands were run in this task.
