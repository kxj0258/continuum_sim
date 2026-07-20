# 配置参考

当前所有主任务都使用 `configs/scenarios/*.yaml`。一个场景文件同时声明机器人模式、后端、场景、任务、运行时 hooks 和 artifact 输出。

## 顶层结构

```yaml
schema_version: 1
scenario:
  name: mujoco_tracking
  arm_mode: dual
  low_level_control_path: ../control/mujoco_tracking_low_level.yaml
  backend: {}
  scene: {}
  task: {}
  runtime: {}
  hooks: {}
  artifacts: {}
```

## scenario

- `name`：运行名，也会用于 `output/runs/<name>_<timestamp>/`。
- `arm_mode`：`dual` 或 `single`。
- `low_level_control_path`：共享低层控制配置，当前主线统一使用 `configs/control/mujoco_tracking_low_level.yaml`。

## backend

当前主线后端为 MuJoCo：

```yaml
backend:
  type: mujoco
  kinematics_mode: discrete_hinge
  mujoco_config_path: ../mujoco_dual.yaml
  source_xml_path: ../../assets/mujoco/dual_three_segment_arm_tendon_with_visuals_mobile_base.xml
  generated_xml_path: ../../output/generated/scenario_dual_tracking.xml
```

- `type`：当前主场景使用 `mujoco`。
- `kinematics_mode`：当前主线使用 `discrete_hinge`。
- `source_xml_path`：基础 MuJoCo XML。
- `generated_xml_path`：场景运行前生成的 XML。`arm_mode: single` 时会自动切换到 single 命名。

单个场景可以覆盖 MuJoCo viewer/live-video 相机：

```yaml
backend:
  viewer:
    camera:
      lookat: [0.0375, 0.005, 0.115]
      distance: 0.28
      azimuth: 300.0
      elevation: -18.0
      follow: none
```

## scene

结构化场景和发动机场景二选一：

```yaml
scene:
  structured_config_path: ../scenes/wiping_board.yaml
```

```yaml
scene:
  engine_config_path: ../scenes/engine_scene.yaml
```

`mujoco_tracking.yaml` 和 `mujoco_point_servo.yaml` 可以使用空场景：

```yaml
scene: {}
```

## task

当前任务类型：

- `tracking`：轨迹跟踪和点伺服。
- `navigation`：移动基座导航和结构化场景避障。
- `engine_navigation`：发动机入口、插入和局部路径导航。
- `wiping`：黑板接触擦拭。

常见字段：

- `waypoint_tolerance_m`：waypoint 到达容差。
- `target_advance_mode`：waypoint 推进模式。
- `observer_control_mode`：`collision_avoidance` 或 `visual_servo`。
- `observer_control`：observer 从臂避碰、看向、视觉伺服参数。
- `scene_avoidance`：环境避障参数。
- `tracking_control`：task-space servo、base staging、tendon command 覆盖项。

## tracking_control

```yaml
tracking_control:
  approach_samples: 5
  tracking_mode: waypoint
  max_steps_per_waypoint: 500
  feedforward_speed_mps: 0.0
  task_space_servo:
    max_speed_mps: 0.015
    enforce_speed_limit: true
```

- `approach_samples`：任务开始前插入的 approach 点数量。
- `tracking_mode`：当前主线主要使用 `waypoint`。
- `max_steps_per_waypoint`：单个 waypoint 最多控制周期数。
- `feedforward_speed_mps`：沿目标方向的前馈速度。
- `task_space_servo`：覆盖共享 task-space servo 参数。
- `tendon_command`：覆盖 IK/tendon command 参数。

## wiping_path

`mujoco_wiping.yaml` 使用场景原生擦拭路径：

```yaml
wiping_path:
  surface_id: board_surface
  patch_id: center_patch
  center_m: [0.0375, 0.005, 0.115]
  width_m: 0.040
  height_m: 0.025
  line_count: 5
  samples_per_line: 5
  approach_reference: first_contact
  approach_offset_m: 0.005
  contact_offset_m: -0.0025
```

`surface_normal_world` 定义接触法向，`contact_offset_m` 和 `approach_offset_m` 都沿该法向解释。

## wiping 控制策略

默认主线：

```yaml
wiping_control_type: hybrid_force_position
force_strategy:
  type: kinematic_hybrid
```

可选动态自适应阻抗：

```yaml
wiping_control_type: dynamic_adaptive_impedance
force_strategy:
  type: dynamic_adaptive_impedance
dynamics_config_path: ../dynamics/pcc_reduced.yaml
```

可选接触触发导纳：

```yaml
wiping_control_type: contact_triggered_admittance
force_strategy:
  type: contact_triggered_admittance
contact_admittance:
  target_normal_force_n: 1.5
  contact_force_threshold_n: 0.1
```

三种策略都接入当前 `WipingController -> WipingForceStrategy -> UnifiedLowLevelController` 主线。

## runtime

```yaml
runtime:
  controller_dt_s: 0.02
  n_substeps: 10
  max_steps: 10000
```

- `controller_dt_s`：控制周期。
- `n_substeps`：每个控制周期内的 MuJoCo 子步数。
- `max_steps`：最大控制步数。

## hooks

```yaml
hooks:
  recorder: true
  tendon_debug: true
  tendon_debug_stride: 5
  show_live_tendon_panel: true
  show_live_diagnostics_panel: true
  live_diagnostics_panel_stride: 5
  show_observer_camera: true
  viewer: mujoco
  keep_viewer_open: false
```

常用 hook：

- `recorder`：记录状态和任务诊断。
- `tendon_debug`：采样 tendon 诊断文本。
- `show_live_tendon_panel`：打开 tendon 实时窗口。
- `show_live_force_panel`：擦拭任务接触力窗口。
- `show_live_diagnostics_panel`：实时控制层诊断窗口；保存 plots 时会落盘最终图。
- `show_observer_camera`：显示 observer 相机窗口。
- `viewer`：`none`、`matplotlib` 或 `mujoco`。

## artifacts

```yaml
artifacts:
  enabled: true
  save_npz: true
  save_plots: true
  save_gif: true
  save_mp4: true
  video_mode: live_mujoco
  video_fps: 10
  video_stride: 10
```

输出目录：

```text
output/runs/<scenario_name>_<timestamp>/
```

`save_plots: true` 会保存轨迹图、四层控制诊断图、PCC/MuJoCo 诊断图，以及可用的 live panel 最终图。

## 在线评分

共享参数位于：

```text
configs/control/mujoco_tracking_low_level.yaml
```

自动跳点主要使用 `reachability_score`，执行器跟踪情况单独记录为 `execution_score`。详细说明见 `docs/online_waypoint_reachability.md`。
