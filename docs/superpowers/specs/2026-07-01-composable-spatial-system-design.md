# Composable Spatial Continuum System Design

## Goal

Refactor the simulator around one composable system model:

```text
world-frame 6D base + N spatial tendon-driven continuum arms + scene
```

Single-arm and dual-arm engine simulations differ only in assembly
configuration. Controllers operate on named system state and emit world-frame
base twist plus per-arm tendon-length rates.

## Decisions

- Base commands are 6D twists expressed in the world frame only.
- The first base implementation is prescribed twist integrated into pose.
- Spatial-arm controllers output tendon-length change rates directly.
- Motor transmission is not part of the new spatial simulation control path.
- Tendon-rate commands are rate-limited, integrated, displacement-limited, then
  converted to absolute MuJoCo tendon-length targets at the backend boundary.
- Tendon tension and slack are out of scope.
- Singularity handling uses rank, singular values, condition number, adaptive
  damping, and velocity scaling.
- Observer priority is executor tracking and executor collision avoidance.
- Base workspace and speed limits are configurable placeholders until engine
  calibration values are available.
- Engine control geometry initially uses existing primitive collision geoms.
- Engine MJCF composition belongs to a scene adapter, not a preview script.

## Coordinate Contract

- `T_W_B` maps coordinates from base frame `B` into world frame `W`.
- Poses use position in metres and quaternion `wxyz`.
- Twists use `[vx, vy, vz, wx, wy, wz]` and are expressed in world frame.
- Tendon displacement uses metres; tendon rate uses metres per second.
- Controller Jacobians map system velocity variables to world-frame task
  velocities.
- Euler angles are allowed only in configuration and MuJoCo boundary adapters.

## Domain Model

`RobotAssemblyConfig` is the composition root for the robot:

```text
RobotAssemblyConfig
├── base: fixed | prescribed_twist
└── arms[name]
    ├── role: executor | observer
    ├── spatial_arm_config_path
    ├── mount pose
    └── attachment
```

Spatial arm files use local tendon indices. Global actuator offsets are derived
by `ControlLayout`; they are not stored in reusable arm files.

`RobotSystemState` contains base state and named `ArmSystemState` objects.
`RobotSystemCommand` contains one world-frame base twist and named tendon-rate
commands. The controller never sees a flat MuJoCo control array.

## Control Layout

The generalized system velocity is:

```text
single: [base_twist_6D, executor_tendon_rate_9D] = 15D
dual:   [base_twist_6D, executor_tendon_rate_9D,
         observer_tendon_rate_9D] = 24D
```

For a fixed base, the same layout may be used with the base block locked, while
the effective optimization variables are the enabled arm blocks.

## Actuation Path

For each arm:

```text
rate = clip(requested_rate, +/- max_tendon_rate)
delta_next = clip(delta_current + dt * rate, delta_min, delta_max)
absolute_target = neutral_tendon_length + delta_next
```

The integrator stores the clipped displacement so saturation does not wind up.
Reset establishes neutral tendon lengths from the backend observation.

## Kinematics and Singularity Handling

Each arm exposes a world-frame tendon Jacobian:

```text
J_tip_tendon = J_tip_shape * dq_d_tendon
```

The whole-body Jacobian inserts the world-frame base point Jacobian and named
arm block:

```text
executor = [J_base_executor, J_executor, 0]
observer = [J_base_observer, 0, J_observer]
```

SVD diagnostics report numerical rank, singular values, condition number, and a
velocity scale. Adaptive damping increases and the velocity scale decreases as
the minimum singular value approaches a configured threshold.

## Whole-Body Objectives

The initial solver is weighted damped least squares:

1. Executor tool tracking.
2. Observer tracking of the executor tool and engine ROI.
3. Observer/executor collision avoidance.
4. Base-motion and neutral-tendon regularization.

The solver interface permits a later hierarchical QP implementation without
changing runtime or backend contracts.

## Scene Composition

`EngineMjcfAdapter` composes a robot MJCF file with:

- engine visual mesh;
- optional engine collision mesh;
- enabled primitive collision geoms.

The same adapter accepts either single-arm or dual-arm robot XML. Scene control
queries use primitive geometry first. Preview scripts become clients of the
adapter.

## Runtime

The generic simulation loop depends on protocols:

```text
SystemBackend -> RobotSystemState
SystemController(RobotSystemState) -> RobotSystemCommand
SimulationLoop -> controller, backend, hooks
```

Tracking, engine cleaning, recording, and viewer behavior are hooks or task
providers rather than separate backend-owning loops.

## Breaking Migration

The system API is the only supported runtime/backend contract after this
refactor. `BackendState`, flat-array backend commands, `default_arm`,
`DualArmCommandAdapter`, and motor-based spatial control are not compatibility
targets. Existing callers must migrate to named assembly, state, command, and
layout types.

## Deferred Work

- tendon tension and slack;
- dynamic-actuated base;
- image-based observer perception;
- full mesh distance queries;
- constrained hierarchical QP;
- calibrated engine limits.
