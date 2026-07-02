# Coordinate and Command Conventions

## Bending coordinates and tendon compatibility

For segment `i`, the normal control coordinate order is `[kx_i, ky_i]`, in
inverse metres. A three-segment arm therefore uses:

```text
[kx_1, ky_1, kx_2, ky_2, kx_3, ky_3]
```

The existing PCC order remains `[kx_i, ky_i, eps_i]`, but normal control inserts
`eps_i = 0`. `C_b` maps bending coordinates to tendon displacement in metres;
its time derivative maps bending rates to tendon rates in metres per second.

Positive/negative curvature directions are arm-local. Mount-pose rotation then
expresses Cartesian Jacobians in the MuJoCo world frame. The tendon mapping
itself is not rebuilt in world coordinates: tendon routing belongs to the arm
local frame.

A tendon vector is compatible when its residual
`r = delta_l - C_b pinv(C_b) delta_l` is within the configured numerical
tolerance. Normal commands must satisfy this condition. Measured state may have
a nonzero residual, which is reported before projection.

This document is normative for the composable spatial-arm system.

## Frames and transforms

`T_A_B` maps a point represented in frame `B` into frame `A`.

```text
T_W_tip = T_W_base * T_base_mount * T_mount_tip
```

- `W`: MuJoCo world frame.
- `base`: prescribed 6D mobile-base frame.
- `mount`: fixed arm mount frame.
- `tip`: continuum-arm tip or attachment frame.

Positions use metres. Rotation matrices are right-handed. Quaternions always
use `[w, x, y, z]`.

## Base twist

The only accepted base command is a world-frame spatial twist:

```text
V_W_base = [vx, vy, vz, wx, wy, wz]
```

Linear velocity uses metres per second. Angular velocity uses radians per
second. Body-frame twist commands are intentionally unsupported.

The initial implementation integrates this twist into a prescribed pose. It is
not a dynamic force/torque actuator model.

Assembly base limits include `calibrated: false` until measured engine/base
workspace, linear-speed, and angular-speed values replace the placeholders.

## Spatial-arm command

Each arm command is the arm-local tendon-length change rate:

```text
delta_l_dot = [dl1/dt, ..., dl9/dt]  # metres per second
```

There is no motor/spool/gear stage in the spatial MuJoCo control path.

```text
rate_limited = clip(delta_l_dot, rate_limits)
delta_l_next = clip(delta_l + dt * rate_limited, displacement_limits)
ctrl = neutral_tendon_length + delta_l_next
```

## Jacobians

Controller Jacobians produce world-frame task velocities:

```text
tip_linear_velocity_W = J_system * system_velocity
```

For direct tendon control:

```text
J_tip_tendon = J_tip_shape * pinv(C_tendon_shape)
```

The base point contribution is:

```text
v_point = v_base + omega_base x (p_point - p_base)
```

Euler-angle subtraction is not a pose error. Controllers must use rotation
matrices, quaternions, or an SE(3) error. Euler angles are allowed only in YAML
input and MuJoCo freejoint boundary conversion.

## System layouts

```text
single = [base_twist(6), executor_tendon_rate(9)]                  # 15D
dual   = [base_twist(6), executor_tendon_rate(9),
          observer_tendon_rate(9)]                                # 24D
```

`ControlLayout` owns flat slices. Controllers, tasks, and scenes address arms
by name and must not concatenate hard-coded vectors.
