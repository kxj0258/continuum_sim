# Four-Layer Control Architecture

This project now treats scenario control as four explicit layers. The stable
system command boundary remains `RobotSystemCommand`: world-frame base twist plus
per-arm tendon-rate references.

## Layer 1: Task Reference

Task code decides what the robot should do next. Examples are trajectory
samples, point-servo goals, wiping phases, engine-cleaning waypoints, observer
ROI policy, and collision-avoidance intent. This layer owns task phase and
completion state. It must not compute MuJoCo actuator targets.

Current entry points still live in the scenario task controllers and staged
engine controllers. They emit `TaskStep(SystemTaskIntent, TaskStatus)`.

## Layer 2: Task-Space Servo

`TaskSpaceServo` converts the executor reference into a Cartesian TCP velocity.

Position mode:

```text
tcp_velocity = position_gain * (target_position - measured_position)
             + feedforward_gain * feedforward_velocity
```

Velocity mode:

```text
tcp_velocity = feedforward_velocity
```

Optional speed limiting is applied here. This is where trajectory tracking and
point servo behavior should diverge: tracking uses position plus feedforward
velocity, while point servo uses position feedback and settling/hold policy.

## Layer 3: Kinematic IK / Tendon Command

`TendonCommandController` converts task-space velocities into tendon-rate
references. It delegates the weighted Jacobian solve to the existing
`WholeBodyController`.

The layer solves in base-plus-bending coordinates:

```text
J v ~= tcp_velocity
v = [base_twist, executor_bending_rate, observer_bending_rate]
tendon_rate_ref = C_b * bending_rate
```

The output is `RobotSystemCommand`. This is the last backend-independent
control output and should be interpreted as tendon-level reference, not as a
MuJoCo actuator target.

## Layer 4: Backend Execution

Execution adapters convert `RobotSystemCommand` into backend-specific action.

For MuJoCo tendon-position actuators, `MujocoTendonPositionExecutionAdapter`
converts tendon-rate references into absolute tendon position targets:

```text
tendon_rate_ref + actual_tendon_length + actuator_force
  -> tendon_position_target
  -> MuJoCo ctrl
```

MuJoCo `tendon_position` does not teleport tendons to the target length. It is a
position actuator: the control value is a target actuator length, and MuJoCo
generates actuator force from the target/current error subject to `kp`, force
limits, dynamics, contacts, and solver behavior.

The execution layer may enforce tendon bounds, target lead, slew, and force
guards. It must not change task targets, solve IK, or implement task-space
trajectory correction.

## Configuration Layout

Low-level profiles use the same four-layer naming:

```yaml
low_level_control:
  task_space_servo:
    position_gain: 1.0
    feedforward_gain: 1.0
    max_speed_mps:
    enforce_speed_limit: false

  tendon_command:
    solver: weighted_svd
    singularity_strategy: svd_projection
    tendon_regularization_weight: 0.8
    enforce_velocity_limits: false

  execution:
    backend_adapter: mujoco_tendon_position
    enforce_tendon_limits: false
    tendon_inner_loop:
      mode: bending_rate_servo
```

`configs/control/spatial_low_level.yaml` is the conservative profile. It enables
task-space speed limiting, solver velocity limits, and protected execution.

`configs/control/mujoco_tracking_low_level.yaml` is the MuJoCo tracking profile.
It leaves solver/backend legacy velocity limits off and relies on the MuJoCo
tendon-position execution adapter with `bending_rate_servo` compensation.

## Result Analysis Layout

Saved scenario runs expose the same four-layer boundary in `result.npz` and
`metadata.json`.

Layer 1 task reference:

```text
layer1_task_target_position_world
layer1_task_feedforward_velocity_world
layer1_task_control_mode
layer1_task_phase
layer1_task_active_index
layer1_task_complete
layer1_task_reference_jump_m
```

Use these fields to check whether the task produced a smooth and feasible
reference. Large reference jumps usually indicate waypoint spacing, phase
handoff, or trajectory sampling issues.

Layer 2 task-space servo:

```text
layer2_servo_position_error_world
layer2_servo_position_error_norm_m
layer2_servo_raw_velocity_world
layer2_servo_velocity_world
layer2_servo_velocity_norm_mps
layer2_servo_speed_limited
```

Use these fields to separate target/reference quality from servo behavior. If
Layer 2 position error is large while `layer2_servo_speed_limited` is often
true, the bottleneck is the task-space speed policy rather than IK.

Layer 3 IK / tendon command:

```text
layer3_ik_tcp_velocity_ref_world
layer3_ik_base_twist_world
layer3_ik_condition_number
layer3_ik_min_singular_value
layer3_ik_velocity_scale
layer3_ik_residual_norm
layer3_ik_projection_residual_norm
layer3_ik_arm_<name>_tendon_rate_ref_mps
```

Use these fields to identify Jacobian conditioning, singularity protection, and
unachievable task-space velocity requests.

Layer 4 backend execution:

```text
layer4_execution_arm_<name>_tendon_rate_ref_mps
layer4_execution_arm_<name>_applied_rate_mps
layer4_execution_arm_<name>_realized_rate_mps
layer4_execution_arm_<name>_tendon_position_target_m
layer4_execution_arm_<name>_tendon_displacement_m
layer4_execution_arm_<name>_tendon_position_error_m
layer4_execution_arm_<name>_force_utilization
layer4_execution_arm_<name>_saturation_active
layer4_execution_arm_<name>_inner_loop_mode
```

Use these fields to determine whether MuJoCo tendon-position execution follows
the tendon-rate reference. A good Layer 3 command with large Layer 4
`applied_minus_realized_rate` or force utilization indicates actuator, tendon
target, force, or contact limitations.

`metadata.json.metrics.control_layers` stores nested summaries for these same
signals. Static artifacts also include `plots/four_layer_control_diagnostics.png`
when layer data is available.
