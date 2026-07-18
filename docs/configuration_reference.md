# 配置参考

清理后的主任务入口位于 `configs/scenarios/`：

- `mujoco_tracking.yaml`
- `mujoco_navigation.yaml`
- `engine_navigation.yaml`
- `mujoco_wiping.yaml`
- `mujoco_point_servo.yaml`

除非特别说明，长度单位为米，时间单位为秒，力单位为牛。

## scenario

```yaml
scenario:
  name: mujoco_tracking
  arm_mode: dual
  low_level_control_path: ../control/mujoco_tracking_low_level.yaml
  backend:
    type: mujoco
```

## scenario.backend.viewer.camera

`scenario.backend.viewer.camera` can override the shared MuJoCo viewer/live-video
camera for one scenario without changing `configs/mujoco_dual.yaml`.

```yaml
backend:
  type: mujoco
  viewer:
    camera:
      lookat: [0.0375, 0.005, 0.115]
      distance: 0.28
      azimuth: 30.0
      elevation: -18.0
      follow: none
```

- `arm_mode`：`dual` 或 `single`。`single` 只保留 executor 主臂，`dual` 同时保留 executor 和 observer。
- `low_level_control_path`：共享低层控制参数，当前主任务统一使用 `configs/control/mujoco_tracking_low_level.yaml`。
- `backend.type`：清理后只支持 `mujoco`。
- `backend.generated_xml_path`：生成后的场景 XML。`arm_mode: single` 时会自动把 `scenario_dual_...xml` 切换为 `scenario_single_...xml`。

## arm_mode 与装配

`arm_mode` 自动选择装配文件：

| 任务类型 | dual | single |
| --- | --- | --- |
| 固定基座任务 | `configs/robots/assemblies/dual_spatial.yaml` | `configs/robots/assemblies/single_spatial.yaml` |
| 移动基座任务 | `configs/robots/assemblies/dual_spatial_mobile.yaml` | `configs/robots/assemblies/single_spatial_mobile.yaml` |

移动基座任务目前指 `navigation` 和 `engine_navigation`。普通 `tracking`、`wiping`、`mujoco_point_servo` 使用固定基座装配。

## task 类型

- `tracking`：用于 `mujoco_tracking.yaml` 和 `mujoco_point_servo.yaml`。
- `navigation`：用于 `mujoco_navigation.yaml`。
- `engine_navigation`：用于 `engine_navigation.yaml`。
- `wiping`：用于 `mujoco_wiping.yaml`。

## tracking_control

常用字段：

- `approach_samples`：任务开始时插入的 approach 点数量。
- `tracking_mode`：`waypoint` 或 `time`。
- `max_steps_per_waypoint`：waypoint 模式下每个点最多停留的控制周期数。
- `feedforward_speed_mps`：沿目标方向的前馈速度。
- `task_space_servo.max_speed_mps`：TCP 目标速度上限。
- `task_space_servo.enforce_speed_limit`：是否启用 TCP 速度上限。
- `online_reachability`：覆盖共享在线评分参数。

## wiping_path

`mujoco_wiping.yaml` 使用场景原生擦拭路径：

```yaml
wiping_path:
  surface_id: board_surface
  patch_id: center_patch
  center_m: [0.0375, 0.005, 0.115]
  line_count: 5
  samples_per_line: 30
  approach_reference: first_contact
  approach_offset_m: 0.005
  contact_offset_m: -0.0025
```

- `approach_reference: first_contact`：预接触点靠近第一条 contact 轨迹点，并位于黑板外侧。
- `approach_offset_m`：预接触点离板面的外侧距离。
- `contact_offset_m`：接触轨迹相对板面的法向偏移。

## wiping 控制策略

`mujoco_wiping.yaml` 默认使用当前主线方案：

```yaml
wiping_control_type: hybrid_force_position
force_strategy:
  type: kinematic_hybrid
```

可选方案一：动态自适应阻抗预测。它先执行运动学混合力位修正，再用缩减 PCC dynamics 对法向修正量做一阶预测。

```yaml
wiping_control_type: dynamic_adaptive_impedance
force_strategy:
  type: dynamic_adaptive_impedance
dynamics_config_path: ../dynamics/pcc_reduced.yaml
```

可选方案二：接触触发导纳。它在接触力超过阈值后启用导纳状态，依据力误差更新法向偏移，并由自身稳定条件或最大步数控制 waypoint 推进。

```yaml
wiping_control_type: contact_triggered_admittance
force_strategy:
  type: contact_triggered_admittance
contact_admittance:
  target_normal_force_n: 1.5
  contact_force_threshold_n: 0.1
  tangent_tolerance_m: 0.001
  force_tolerance_n: 0.08
  stable_steps_required: 1
  max_steps_per_target: 100
  position_gain: 10.0
  kp_force: 0.5
  ki_force: 0.012
  admittance_mass: 1.0
  admittance_damping: 20.0
  admittance_stiffness: 5.0
  admittance_clip_m: 0.012
  force_deadband_n: 0.03
  force_filter_alpha: 0.1
  enforce_velocity_limits: false
```

三种策略都接入当前 `WipingController -> WipingForceStrategy -> UnifiedLowLevelController` 架构；默认主线不会因为保留可选策略而改变。

## 在线评分

共享评分参数位于：

```text
configs/control/mujoco_tracking_low_level.yaml
```

当前分为两个分数：

- `reachability_score = progress * alignment * model`
- `execution_score = tendon`

自动跳点依据 `reachability_score`，而不是 MuJoCo actuator 是否完全跟上的 `execution_score`。详细公式见 `docs/online_waypoint_reachability.md`。
