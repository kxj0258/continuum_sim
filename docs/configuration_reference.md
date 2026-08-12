# 配置参考

场景文件位于 `configs/scenarios/*.yaml`。一个场景同时声明机器人装配、MuJoCo 后端、环境、任务、运行参数、实时窗口和输出产物。

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

- `name`：运行名称，也是输出目录名称的一部分。
- `arm_mode`：`dual` 或 `single`。
- `low_level_control_path`：低层控制参数文件。

`dual` 启用 `executor` 和 `observer`，`single` 启用 `executor`。

## backend

```yaml
backend:
  type: mujoco
  kinematics_mode: discrete_hinge
  mujoco_config_path: ../mujoco_dual.yaml
  source_xml_path: ../../assets/mujoco/dual_three_segment_arm_tendon_with_visuals_mobile_base.xml
  generated_xml_path: ../../output/generated/scenario_dual_tracking.xml
  viewer:
    camera:
      lookat: [0.0375, 0.005, 0.115]
      distance: 0.28
      azimuth: 300.0
      elevation: -18.0
      follow: none
```

- `type`：MuJoCo 后端标识。
- `kinematics_mode`：系统状态和雅可比使用的运动学模式。
- `mujoco_config_path`：求解器、执行器、渲染和 viewer 参数。
- `source_xml_path`：机械臂基础 MJCF。
- `generated_xml_path`：注入场景、附件、相机和传感器后的 MJCF。
- `viewer.camera`：MuJoCo 主窗口相机。

## scene

结构化场景：

```yaml
scene:
  structured_config_path: ../scenes/wiping_board.yaml
```

发动机场景：

```yaml
scene:
  engine_config_path: ../scenes/engine_scene.yaml
```

无环境物体：

```yaml
scene: {}
```

### 发动机材质

`configs/scenes/engine_scene.yaml`：

```yaml
preview_visualization:
  visual_mesh_rgba: [0.66, 0.68, 0.71, 1.0]
  visual_material:
    name: engine_silver
    emission: 0.0
    specular: 0.72
    shininess: 0.48
```

`rgba` 的第四项为不透明度。材质绑定在发动机可视网格上，场景构建器同时设置中性 headlight、主光和补光，主窗口与 observer 相机共享渲染结果。

## task

支持的任务类型：

- `idle`：不生成自动运动命令，用于手动控制。
- `tracking`：轨迹跟踪和点伺服。
- `navigation`：移动基座接近和结构化场景导航。
- `engine_navigation`：发动机入口、插入和内部路径导航。
- `wiping`：接触擦拭。

常用字段：

- `waypoint_tolerance_m`：waypoint 到达容差。
- `target_advance_mode`：目标推进方式。
- `observer_control_mode`：`collision_avoidance` 或 `visual_servo`。
- `observer_control`：从臂避碰、看向和视觉伺服参数。
- `scene_avoidance`：环境避障参数。
- `tracking_control`：任务空间伺服和肌腱命令覆盖参数。

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

- `approach_samples`：进入任务轨迹前生成的接近点数量。
- `tracking_mode`：目标跟踪模式。
- `max_steps_per_waypoint`：单个目标最多运行的控制周期数。
- `feedforward_speed_mps`：目标方向前馈速度。
- `task_space_servo`：任务空间速度与限幅参数。
- `tendon_command`：弯曲空间、雅可比和肌腱命令参数。

## wiping

擦拭路径示例：

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

力反馈与控制：

```yaml
wiping_control_type: hybrid_force_position
force_strategy:
  type: kinematic_hybrid
surface_normal_world: [-1.0, 0.0, 0.0]
target_normal_force_n: 1.5
force_feedback_mode: tool_wrench_sensor
max_contact_force_n: 5.0
```

`tool_wrench_sensor` 从执行臂末端六维力传感器读取力和力矩。控制器把传感器力转换到世界坐标，并将其投影到 `surface_normal_world`，得到擦拭法向力。

系统还提供 `dynamic_adaptive_impedance` 和 `contact_triggered_admittance` 两种擦拭控制策略，参数分别位于 `force_strategy`、`dynamics_config_path` 和 `contact_admittance`。

## runtime

```yaml
runtime:
  controller_dt_s: 0.02
  n_substeps: 20
  max_steps: 10000
```

- `controller_dt_s`：控制周期。
- `n_substeps`：每个控制周期的 MuJoCo 子步数。
- `max_steps`：最大控制周期数。

MuJoCo 配置使用 `solver.timestep: 0.001`，因此必须满足：

```text
n_substeps × solver.timestep = controller_dt_s
20 × 0.001 s = 0.02 s
```

场景构建时会检查该关系，不一致时停止启动并报告配置错误。

## hooks

```yaml
hooks:
  recorder: true
  tendon_debug: true
  tendon_debug_stride: 5
  show_live_tendon_panel: true
  show_live_force_panel: false
  show_live_diagnostics_panel: true
  show_observer_camera: true
  observer_camera_stride: 1
  viewer: mujoco
  keep_viewer_open: false
```

- `recorder`：记录系统状态、命令和任务诊断。
- `tendon_debug`：输出肌腱诊断采样。
- `show_live_tendon_panel`：肌腱状态窗口。
- `show_live_force_panel`：擦拭力窗口。
- `show_live_diagnostics_panel`：控制层诊断窗口。
- `show_observer_camera`：观测臂相机窗口。
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

输出内容由开关决定，可包含 `result.npz`、`metadata.json`、配置副本、生成模型、诊断图和视频。

## 工具附件

执行臂在机器人装配配置中通过 `attachment: carbon_remover` 选择工具。`configs/tools/carbon_remover.yaml` 的主要字段：

```yaml
tip_to_attachment:
  position: [0.0, 0.0, 0.004]
tcp_pose:
  position: [0.0, 0.0, 0.014]
collision:
  type: sphere
  radius_m: 0.009
  position: [0.0, 0.0, 0.005]
force_torque_sensor:
  size_m: [0.015, 0.015, 0.008]
  filter_cutoff_hz: 15.0
  tare_on_reset: true
  gravity_compensation: true
```

`tip_to_attachment` 将传感器中心放在裸臂末端前方 `4 mm`；球心相对传感器中心前移 `5 mm`，半径为 `9 mm`，所以球体后端与传感器后表面相切。`tcp_pose` 相对传感器中心前移 `14 mm`，因此球面 TCP 距裸臂末端 `18 mm`。

## 在线评分

在线目标推进使用 `reachability_score`，执行层跟踪质量使用 `execution_score`。配置位于 `configs/control/mujoco_tracking_low_level.yaml`，详见 [在线 waypoint 可达性评分](online_waypoint_reachability.md)。
