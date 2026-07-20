# 新手入门指南

本文面向第一次接触 `continuum_sim` 的读者，目标是帮助你从“能看懂项目在做什么”逐步走到“能改配置、能调参、能扩展任务”。如果只想快速运行场景，可以先看 `README.md`；如果想理解项目结构和控制链路，建议从本文开始。

## 一句话理解项目

`continuum_sim` 是一个面向空间连续体机械臂的 MuJoCo 仿真与控制项目。当前主线用一个场景 YAML 描述任务，再由 `SimulationApplication` 组装机器人、场景、控制器、MuJoCo 后端、实时诊断窗口和输出产物。

核心运行链路是：

```text
configs/scenarios/*.yaml
  -> load_scenario_config
  -> SimulationApplication
  -> task plan
  -> task controller
  -> UnifiedLowLevelController
  -> WholeBodyController / tendon command
  -> MujocoSystemBackend
  -> hooks / artifacts
```

当前只维护 5 个主任务入口：

```powershell
python scripts/run_scenario.py configs/scenarios/mujoco_tracking.yaml
python scripts/run_scenario.py configs/scenarios/mujoco_navigation.yaml
python scripts/run_scenario.py configs/scenarios/engine_navigation.yaml
python scripts/run_scenario.py configs/scenarios/mujoco_wiping.yaml
python scripts/run_scenario.py configs/scenarios/mujoco_point_servo.yaml
```

## 推荐学习路线

建议按这个顺序读项目：

1. 先看 `README.md`，知道当前保留哪些场景、怎么运行、输出在哪里。
2. 看 `configs/scenarios/mujoco_point_servo.yaml`，理解最小任务结构。
3. 看 `configs/scenarios/mujoco_tracking.yaml`，理解 executor/observer 双臂跟踪和避碰。
4. 看 `docs/configuration_reference.md`，熟悉 YAML 字段。
5. 看 `src/continuum_sim/application/application.py`，理解 composition root。
6. 看 `src/continuum_sim/application/controller_factory.py`，理解不同任务如何选择控制器。
7. 看 `src/continuum_sim/control/scenario_controllers.py` 和 `src/continuum_sim/control/coordinated_tracking.py`，理解上层任务和双臂协调。
8. 看 `src/continuum_sim/kinematics/pcc.py`、`src/continuum_sim/kinematics/whole_body.py`，理解 PCC、Jacobian 和 whole-body 求解。
9. 最后看 `src/continuum_sim/backends/mujoco_system_backend.py` 和 `src/continuum_sim/execution/`，理解命令如何进入 MuJoCo。

## 目录地图

```text
configs/
  scenarios/       当前 5 个主任务场景
  control/         共享低层控制参数
  robots/          单臂、双臂、移动基座、装配和安装位姿
  scenes/          wiping board、rocket、engine 等场景
  tools/           喷嘴、相机、清理工具等末端附件

assets/
  mujoco/          MuJoCo XML 基础模型
  meshes/          机器人和双臂 mesh
  engine/          发动机视觉/碰撞模型
  cad/             CAD 源文件

scripts/
  run_scenario.py          单场景入口
  run_all_scenarios.py     批量运行主场景
  export_replay_video.py   从结果导出视频
  build_mujoco_dual_arm_model.py  生成 MuJoCo 双臂模型

src/continuum_sim/
  application/     场景加载和组装入口
  model/           机器人参数、装配、肌腱、base pose
  kinematics/      PCC、Jacobian、whole-body 映射
  control/         上层任务控制器、低层 intent resolver、求解器
  execution/       tendon-rate 到 MuJoCo tendon-position 的执行适配
  backends/        MuJoCo 系统后端
  runtime/         simulation loop、hooks、实时窗口和记录器
  io/              artifact 导出、npz、metadata、plots、video
  scenes/          场景配置、查询、MJCF adapter
  tasks/           task plan、轨迹、擦拭路径、发动机导航路径
  sensing/         observer 相机和视觉反馈
  visualization/   live panel、视频、调试图
```

## 应用层架构

当前项目把 `SimulationApplication` 作为唯一主线入口。它负责：

1. 读取 `configs/scenarios/*.yaml`。
2. 加载机器人装配。
3. 加载结构化场景或发动机场景。
4. 生成 task plan。
5. 构建控制器。
6. 构建运行时 hooks。
7. 创建 `SimulationLoop`。
8. 运行后保存 artifacts。

关键文件：

- `src/continuum_sim/application/scenario.py`：场景 YAML 解析和 dataclass 配置。
- `src/continuum_sim/application/application.py`：composition root。
- `src/continuum_sim/application/backend_factory.py`：MuJoCo 后端和 XML 生成。
- `src/continuum_sim/application/task_plan_factory.py`：轨迹、导航、擦拭路径转换成 task plan。
- `src/continuum_sim/application/controller_factory.py`：根据 `task.type` 选择上层控制器。
- `src/continuum_sim/application/hook_factory.py`：构建 recorder、live panel、viewer、video、observer camera 等 hooks。

理解项目时，可以把 `application/` 看成“把所有零件接起来”的地方；具体算法通常不放在这里。

## 机器人和 PCC 模型

每条连续体臂由 3 段组成。每段的 PCC 状态是：

```text
[kx_i, ky_i, eps_i]
```

整条臂是：

```text
[kx1, ky1, eps1, kx2, ky2, eps2, kx3, ky3, eps3]
```

当前默认运动学模式是：

```text
discrete_hinge
```

它把一段连续体近似成实际柔性铰链序列，默认顺序为：

```text
Y / X / Y / X
```

相关文件：

- `src/continuum_sim/model/robot_params.py`：段长、柔性段长度、直段长度、肌腱半径、铰链轴等参数。
- `src/continuum_sim/kinematics/pcc.py`：PCC 正运动学、中心线采样、三种 kinematics mode。
- `src/continuum_sim/kinematics/tendon_mapping.py`：PCC 状态和 tendon length delta 的映射。
- `src/continuum_sim/kinematics/whole_body.py`：tip、centerline、orientation、base 的 Jacobian。

当前 PCC 相比旧理想常曲率模型更贴近实物：

- 区分 `flexure_length` 和 `distal_straight_length`。
- 默认使用离散铰链布局表达实际柔性结构。
- tendon 映射使用有效柔性段长度，而不是整段总长。
- whole-body Jacobian、控制器和 MuJoCo 后端使用同一套 `kinematics_mode`。

## 系统状态和命令

系统状态类型在 `src/continuum_sim/system/types.py`。

一个控制周期内，控制器读取：

```text
RobotSystemState
  base pose
  executor arm state
  observer arm state
  metadata
```

输出：

```text
RobotSystemCommand
  base_twist_world
  arms[executor].tendon_rate_mps
  arms[observer].tendon_rate_mps
  metadata
```

移动基座命令是世界系 6D twist：

```text
[vx, vy, vz, wx, wy, wz]
```

每条臂的底层命令是 tendon rate，单位 m/s。MuJoCo 后端不会直接把 tendon rate 塞给 actuator，而是通过执行层转换成 tendon position target。

## 控制链路

当前控制链路分成四层更容易理解：

```text
Layer 1: task controller
  选择当前目标点、任务阶段、接触状态、observer 策略

Layer 2: task-space servo
  把 executor 目标位置/姿态误差转换成 TCP 速度

Layer 3: tendon command / whole-body solver
  把 TCP 速度、observer 避碰、视觉伺服、场景避障转换成 tendon-rate reference

Layer 4: MuJoCo tendon-position execution
  把 tendon-rate reference 转成 MuJoCo tendon position target
```

对应代码：

- Layer 1：`scenario_controllers.py`、`staged_navigation.py`、`staged_engine_navigation.py`
- Layer 2：`task_space_servo.py`
- Layer 3：`intent_resolver.py`、`tendon_command_controller.py`、`whole_body_controller.py`
- Layer 4：`execution/`、`mujoco_system_backend.py`

## 主臂和从臂

当前双臂架构中有两个固定角色：

```text
executor：主臂
observer：从臂
```

executor 主臂负责：

- 跟踪轨迹和 waypoint。
- 执行 navigation 和 engine navigation 的目标。
- 执行擦拭接触任务。
- 可选姿态控制、法向力/位控制、场景避障。

observer 从臂负责：

- 避免与 executor 碰撞。
- 观察 ROI 或 executor tip。
- 执行 visual servo。
- 参与自身场景避障。

核心设计原则是：executor tracking 是主任务，observer 的避碰和观察任务只作用在 observer tendon 空间中，不把 executor 或 base 拉离主任务。

## 从臂避碰逻辑

从臂避碰在 `coordinated_tracking.py` 中实现。

每个周期：

1. 用当前 tendon displacement 估计 executor 和 observer 的 bending 状态。
2. 用 PCC 正运动学采样两条臂中心线。
3. 计算两条中心线采样点之间的最近距离。
4. 当距离小于 `inter_arm_influence_distance_m` 时激活避碰。
5. 期望分离速度为：

```text
desired_speed = avoidance_gain * max(influence_distance - distance, 0)
```

6. 把分离速度做成 `executor_observer_collision_avoidance` 任务。
7. 用 `_observer_only_jacobian()` 清掉非 observer 变量，只允许从臂动。
8. observer priority stack 优先求解 `interarm_avoidance`。

默认从臂优先级：

```yaml
observer:
  - tasks: [interarm_avoidance]
  - tasks: [observer_tracking]
  - tasks: [look_at]
  - tasks: [scene_avoidance]
```

这意味着避碰优先级高于观察和跟踪。若开启 visual servo，视觉任务也会服从避碰。

## 五个主任务

### mujoco_point_servo.yaml

用途：最小单点伺服调试。

默认模式：

```yaml
arm_mode: single
```

控制器：

```text
WaypointTrackingController
  -> UnifiedLowLevelController
```

适合新手先理解：

- 一个 target 如何变成 executor tendon-rate。
- PCC/IK 是否大致工作。
- MuJoCo tendon actuator 是否跟随。

### mujoco_tracking.yaml

用途：固定基座双臂轨迹跟踪。

控制器：

```text
WaypointTrackingController
  -> UnifiedLowLevelController
  -> CoordinatedTracking / WholeBodyController
```

executor：

- 跟踪方形轨迹或配置中的 trajectory。
- 可选姿态控制。

observer：

- 默认 collision avoidance。
- 可选 look-at 或 ROI tracking。

重点看：

- `task.trajectory`
- `task.tracking_control`
- `task.observer_control`
- live diagnostics 的 reachability 和 tendon error。

### mujoco_navigation.yaml

用途：结构化场景导航。

控制器：

```text
StagedNavigationController
  -> base_approach
  -> NavigationController
  -> UnifiedLowLevelController
```

控制流程：

1. 移动基座先靠近当前 waypoint。
2. 到位后切换到局部机械臂 waypoint 跟踪。
3. executor 跟踪目标点。
4. observer 做避碰/观察。
5. scene avoidance 根据结构化场景距离提供避障项。

重点参数：

- `tracking_control.stage_mobile_base`
- `base_position_gain`
- `base_orientation_gain`
- `base_approach_standoff_m`
- `scene_avoidance`

### engine_navigation.yaml

用途：发动机场景入口、插入和局部路径导航。

控制器：

```text
StagedEngineNavigationController
  -> base_approach
  -> base_insertion
  -> executor_navigation
      -> WaypointTrackingController
```

阶段：

- `base_approach`：移动基座到预入口位姿。
- `base_insertion`：沿发动机入口路径推进基座。
- `executor_navigation`：主臂执行局部路径，例如圆、椭圆、方形。
- `complete` / `failed`：完成或失败。

executor：

- 执行插入后的局部导航路径。

observer：

- 通常使用 visual servo，看向 ROI。
- 同时保留臂间避碰。

重点参数：

- `task.engine_navigation.entry_region`
- `task.engine_navigation.insertion_path`
- `task.engine_navigation.intermediate_local_paths`
- `task.engine_navigation.local_path`
- `observer_control.visual_servo`

### mujoco_wiping.yaml

用途：黑板接触擦拭。

控制器：

```text
WipingController
  -> WipingForceStrategy
  -> UnifiedLowLevelController
```

阶段通常包括：

- approach：接近接触区域。
- contact wiping：沿擦拭路径接触运动。
- retreat：离开接触面。

默认控制策略：

```yaml
wiping_control_type: hybrid_force_position
force_strategy:
  type: kinematic_hybrid
```

含义：

- 切向跟踪擦拭路径。
- 法向根据接触距离/力误差修正。

重点参数：

- `surface_normal_world`
- `target_contact_distance_m`
- `target_normal_force_n`
- `normal_force_gain`
- `force_control_weight`
- `wiping_path`
- `show_live_force_panel`

## 场景 YAML 怎么读

一个主场景通常长这样：

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

读 YAML 时建议按顺序看：

1. `name`：输出目录和运行名。
2. `arm_mode`：single 还是 dual。
3. `backend`：MuJoCo XML、kinematics mode、viewer camera。
4. `scene`：结构化场景或 engine scene。
5. `task`：任务类型和目标。
6. `runtime`：控制周期和步数。
7. `hooks`：实时窗口、viewer、recorder。
8. `artifacts`：npz、plots、videos。

## 共享低层控制参数

共享低层参数在：

```text
configs/control/mujoco_tracking_low_level.yaml
```

### task_space_servo

```yaml
task_space_servo:
  position_gain: 1.0
  observer_position_gain: 1.0
  feedforward_gain: 1.0
  feedforward_speed_mps: 0.0
  max_speed_mps:
  enforce_speed_limit: false
```

含义：

- `position_gain`：executor 位置误差到目标速度的比例增益。
- `observer_position_gain`：observer tracking 的位置增益。
- `feedforward_gain`：前馈速度缩放。
- `feedforward_speed_mps`：沿 waypoint 方向的前馈速度。
- `max_speed_mps`：TCP 速度上限。
- `enforce_speed_limit`：是否强制限速。

调参直觉：

- 跟踪慢：提高 `position_gain` 或前馈速度。
- 抖动明显：降低 `position_gain`，开启并降低 `max_speed_mps`。
- waypoint 附近来回震荡：降低增益或提高阻尼/限速。

### tendon_command

```yaml
tendon_command:
  executor_tracking_weight: 100.0
  observer_tracking_weight: 40.0
  collision_avoidance_weight: 80.0
  executor_orientation_tracking_mode: nullspace
  base_regularization_weight: 1.0
  tendon_regularization_weight: 0.8
  singularity_strategy: svd_projection
```

含义：

- `executor_tracking_weight`：主臂跟踪任务权重。
- `observer_tracking_weight`：从臂观察/跟踪任务权重。
- `collision_avoidance_weight`：避碰任务权重。
- `base_regularization_weight`：移动基座速度正则。
- `tendon_regularization_weight`：肌腱速度正则。
- `singularity_strategy`：奇异附近的求解策略。
- `minimum_singular_value`、`maximum_damping`、`minimum_velocity_scale`：奇异保护参数。

调参直觉：

- 主臂任务被其它任务干扰：提高 `executor_tracking_weight`，或检查 priority stack。
- 从臂避碰不积极：提高 `collision_avoidance_weight` 或 `avoidance_gain`。
- tendon 命令过大：提高 `tendon_regularization_weight`，开启速度限制。
- 奇异附近速度突然变慢：检查 `minimum_singular_value` 和 `minimum_velocity_scale`。

### priority_stack

```yaml
priority_stack:
  executor:
    - tasks: [position_servo]
    - tasks: [normal_force_control]
    - tasks: [orientation_servo]
    - tasks: [scene_avoidance]
  observer:
    - tasks: [interarm_avoidance]
    - tasks: [observer_tracking]
    - tasks: [look_at]
    - tasks: [scene_avoidance]
```

优先级栈表示：前面的任务优先满足，后面的任务在前面任务的零空间里尽量满足。

常见调整：

- 希望从臂绝对先避碰：保持 `interarm_avoidance` 第一。
- 希望从臂优先看 ROI：把 `visual_servo` 或 `look_at` 放在 `observer_tracking` 前。
- 希望主臂优先避障：把 `scene_avoidance` 提到 executor 更前面，但这可能牺牲轨迹跟踪。

### execution

```yaml
execution:
  backend_adapter: mujoco_tendon_position
  tendon_inner_loop:
    mode: bending_rate_servo
    rate_filter_time_constant_s: 0.04
    feedforward_lead_time_s: 0.02
    rate_integral_gain: 0.1
```

含义：

- 上层输出 tendon-rate reference。
- 执行层把 tendon rate 转成 MuJoCo tendon position target。
- `bending_rate_servo` 会把命令约束到更兼容的 bending 子空间。

调参直觉：

- actuator 跟不上：看 live diagnostics 里的 `execution_score`、tendon error、force utilization。
- target lead 太大：考虑启用 target lead limit。
- 速度噪声大：增加 `rate_filter_time_constant_s`。
- 稳态误差大：适度调整 `rate_integral_gain`。

## observer_control 参数

场景里的 `task.observer_control` 定义从臂行为。

```yaml
observer_control:
  minimum_distance_m: 0.010
  influence_distance_m: 0.019
  critical_distance_m: 0.008
  release_margin_m: 0.001
  avoidance_gain: 5.0
  collision_pair_count: 3
```

含义：

- `minimum_distance_m`：期望安全距离。
- `influence_distance_m`：进入该距离后开始避碰。
- `critical_distance_m`：危险距离，诊断进入 critical。
- `release_margin_m`：避碰释放滞回，避免开关抖动。
- `avoidance_gain`：距离越近，分离速度越大。
- `collision_pair_count`：选多少组最近中心线点对参与避碰。

调参直觉：

- 从臂反应太晚：增大 `influence_distance_m`。
- 从臂让得不够快：增大 `avoidance_gain`。
- 避碰抖动：增大 `release_margin_m` 或降低 `avoidance_gain`。
- 只避开 tip 但中段接近：增大 `collision_pair_count`。

## hooks 和实时窗口

常用 hooks：

```yaml
hooks:
  recorder: true
  tendon_debug: true
  show_live_tendon_panel: true
  show_live_force_panel: true
  show_live_diagnostics_panel: true
  show_observer_camera: true
  viewer: mujoco
```

用途：

- `recorder`：记录状态、命令和诊断。
- `tendon_debug`：采样 tendon 诊断。
- `show_live_tendon_panel`：实时 tendon target/actual/force。
- `show_live_force_panel`：擦拭接触力窗口。
- `show_live_diagnostics_panel`：四层控制和在线可达性诊断。
- `show_observer_camera`：observer 相机窗口。
- `viewer`：打开 MuJoCo 或 matplotlib viewer。

如果 `artifacts.save_plots: true`，运行结束后会保存当前可用的诊断图，包括 `plots/live_diagnostics_panel.png`。

## artifacts 输出

常见输出：

```text
output/runs/<scenario_name>_<timestamp>/
  result.npz
  metadata.json
  configs/
  model/
  plots/
  videos/
```

建议新手先看：

- `metadata.json`：有没有 early stop、错误、主要指标。
- `plots/four_layer_control_diagnostics.png`：四层控制诊断。
- `plots/live_diagnostics_panel.png`：实时窗口最终图。
- `videos/simulation.mp4` 或 `simulation.gif`：整体运动。
- `result.npz`：需要做数据分析时再看。

## 常见调参路径

### 跟踪误差大

先看：

- tip error 是否持续下降。
- tendon error 是否很大。
- execution score 是否低。
- reachability score 是低在 progress、alignment 还是 model。

可能调整：

- 增大 `task_space_servo.position_gain`。
- 启用并设置 `max_speed_mps`。
- 增大 `executor_tracking_weight`。
- 检查 waypoint 是否过密、过远或几何不可达。
- 检查 `kinematics_mode` 是否与模型一致。

### 从臂和主臂距离太近

先看 metadata/live diagnostics：

- `inter_arm_distance_m`
- `observer_collision_active`
- `inter_arm_safety_mode`
- `observer_avoidance_desired_speed_mps`

可能调整：

- 增大 `observer_control.influence_distance_m`。
- 增大 `observer_control.avoidance_gain`。
- 增大 `collision_avoidance_weight`。
- 增大 `collision_pair_count`。
- 把 observer priority stack 中 `interarm_avoidance` 保持在第一位。

### 从臂观察目标不稳定

可能调整：

- 降低 `observer_visual_servo_center_gain`。
- 降低 `observer_visual_servo_depth_gain`。
- 设置 `observer_visual_servo_max_speed_mps`。
- 设置 `observer_visual_servo_max_angular_speed_rad_s`。
- 确认 visual servo 任务没有和 interarm avoidance 冲突。

### 擦拭接触力不合适

先看：

- live force panel。
- `target_normal_force_n`
- `measured_normal_force_n`
- `force_error_n`
- `contact_distance_m`

可能调整：

- `normal_force_gain`
- `force_control_weight`
- `force_proxy_stiffness_n_m`
- `max_normal_velocity_m_s`
- `target_contact_distance_m`
- `wiping_path.contact_offset_m`

### 执行层跟不上

先看：

- tendon target error。
- force utilization。
- saturation active。
- actuator force at limit。

可能调整：

- 降低 task-space 速度。
- 增加 tendon 正则。
- 调整 tendon inner loop 滤波和积分。
- 检查 tendon displacement/rate/target lead 限制。

## 如何新增一个场景

最稳的方式是复制现有场景：

1. 从最接近的 YAML 复制一个新文件到 `configs/scenarios/`。
2. 修改 `scenario.name`。
3. 修改 `scene`。
4. 修改 `task` 目标和任务参数。
5. 保持 `low_level_control_path` 指向共享低层配置。
6. 先关闭复杂 hooks 或视频，确认基本运动后再打开。
7. 手动运行新场景。

新场景建议从这些模板选：

- 点伺服：复制 `mujoco_point_servo.yaml`。
- 轨迹跟踪：复制 `mujoco_tracking.yaml`。
- 结构化导航：复制 `mujoco_navigation.yaml`。
- 发动机任务：复制 `engine_navigation.yaml`。
- 接触擦拭：复制 `mujoco_wiping.yaml`。

## 如何新增一个任务类型

如果要新增 `task.type`，通常需要改这些地方：

1. `src/continuum_sim/application/scenario.py`
   - 增加 task type 字符串。
   - 增加配置字段解析。

2. `src/continuum_sim/application/task_plan_factory.py`
   - 把 YAML 目标转换成 `TaskPlan` 或新任务 plan。

3. `src/continuum_sim/application/controller_factory.py`
   - 根据新 `task.type` 构建新控制器。

4. `src/continuum_sim/control/`
   - 实现新上层控制器。
   - 输出 `TaskStep` 或直接输出 `RobotSystemCommand`。

5. `src/continuum_sim/io/scenario_artifacts.py`
   - 如果新任务有新诊断字段，加入 artifact 导出。

6. `docs/`
   - 更新配置说明和主场景说明。

推荐优先复用：

- `UnifiedLowLevelController`
- `TaskSpaceServo`
- `WaypointScheduler`
- `CoordinatedTrackingConfig`
- `WholeBodyController`

不要一开始就重写底层 tendon 求解器，除非新任务确实需要完全不同的执行接口。

## 如何新增一种擦拭力控策略

擦拭策略在：

```text
src/continuum_sim/control/wiping_force_strategies.py
```

接入点在：

```text
src/continuum_sim/application/control_config_factory.py
```

基本步骤：

1. 定义新的 `WipingForceStrategy` 子类或同接口对象。
2. 在 `build_wiping_force_strategy()` 中识别新的 `force_strategy.type`。
3. 在 `scenario.py` 中解析需要的新参数。
4. 在 `mujoco_wiping.yaml` 中配置新策略。
5. 在 artifacts metadata 中记录关键诊断字段。

## 如何新增工具或附件

工具配置放在：

```text
configs/tools/
```

当前例子：

- `carbon_remover.yaml`
- `eye_camera_air_gun.yaml`

机器人 arm 配置可以通过 `attachment` 引用工具。工具加载逻辑在：

```text
src/continuum_sim/tools/attachments.py
```

如果工具影响几何、相机或控制目标，需要同步检查：

- robot arm 配置中的 attachment。
- observer camera hook。
- scene/query 中的碰撞或 ROI。
- controller metadata。

## 如何新增场景

结构化场景在：

```text
configs/scenes/
```

主要类型：

- `wiping_board.yaml`
- `rocket_nozzle_entry.yaml`
- `rocket_wall_inspection.yaml`
- `engine_scene.yaml`

代码位置：

- `src/continuum_sim/scenes/scene_config.py`
- `src/continuum_sim/scenes/scene_builder.py`
- `src/continuum_sim/scenes/structured_query.py`
- `src/continuum_sim/scenes/engine_scene.py`
- `src/continuum_sim/scenes/engine_query.py`

如果只是换 waypoint 或 surface，优先改 YAML。只有需要新增场景元素类型或查询方式时才改代码。

## 阅读输出诊断

建议按这个顺序看一次运行结果：

1. `metadata.json`
   - `stopped_early`
   - `stop_reason`
   - `errors`
   - `metrics`

2. `plots/trajectory.png`
   - 目标轨迹和实际轨迹是否大体一致。

3. `plots/four_layer_control_diagnostics.png`
   - Layer 1 target 是否跳变。
   - Layer 2 servo error 是否下降。
   - Layer 3 IK residual 是否异常。
   - Layer 4 tendon error 是否过大。

4. `plots/live_diagnostics_panel.png`
   - reachability 分数。
   - execution 分数。
   - observer 模式。
   - inter-arm distance。

5. `videos/`
   - 观察实际运动是否和指标一致。

## 常见代码入口速查

```text
运行入口:
  scripts/run_scenario.py

应用组装:
  src/continuum_sim/application/application.py
  src/continuum_sim/application/controller_factory.py
  src/continuum_sim/application/hook_factory.py

配置解析:
  src/continuum_sim/application/scenario.py

任务计划:
  src/continuum_sim/application/task_plan_factory.py
  src/continuum_sim/tasks/

控制器:
  src/continuum_sim/control/scenario_controllers.py
  src/continuum_sim/control/staged_navigation.py
  src/continuum_sim/control/staged_engine_navigation.py
  src/continuum_sim/control/coordinated_tracking.py

低层控制:
  src/continuum_sim/control/unified_low_level.py
  src/continuum_sim/control/intent_resolver.py
  src/continuum_sim/control/tendon_command_controller.py
  src/continuum_sim/control/whole_body_controller.py

PCC 和 Jacobian:
  src/continuum_sim/kinematics/pcc.py
  src/continuum_sim/kinematics/whole_body.py
  src/continuum_sim/kinematics/tendon_mapping.py

MuJoCo 后端:
  src/continuum_sim/backends/mujoco_system_backend.py
  src/continuum_sim/execution/

输出:
  src/continuum_sim/io/scenario_artifacts.py

实时诊断:
  src/continuum_sim/runtime/live_panel_hooks.py
```

## 新手改动建议

刚开始不要同时改很多层。推荐按这个顺序练习：

1. 改 `mujoco_point_servo.yaml` 的目标点。
2. 改 `mujoco_tracking.yaml` 的 trajectory 半径或 samples。
3. 改 `task_space_servo.position_gain`，观察误差和抖动。
4. 改 observer 的 `influence_distance_m` 和 `avoidance_gain`，观察臂间距离。
5. 改 `mujoco_wiping.yaml` 的 `target_normal_force_n` 和 `normal_force_gain`。
6. 新增一个复制自现有场景的 YAML。
7. 最后再考虑新增控制器或策略。

每次改动只改一两个参数，运行后看 metadata、plots 和 video。这样最容易建立参数和行为之间的对应关系。

## 手动验证建议

本文只说明项目结构和调参路径，不要求自动运行任何命令。需要验证时可手动执行：

```powershell
python scripts/run_scenario.py configs/scenarios/mujoco_point_servo.yaml
python scripts/run_scenario.py configs/scenarios/mujoco_tracking.yaml
python scripts/run_scenario.py configs/scenarios/mujoco_wiping.yaml
```

需要批量跑主场景时：

```powershell
python scripts/run_all_scenarios.py
```
