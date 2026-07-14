# 架构概览

本文描述当前推荐维护路径。旧运行时和旧任务配置仍保留用于兼容、测试和参考，但新的实验应优先使用
`configs/scenarios/*.yaml` 与 `SimulationApplication`。

## 顶层数据流

```text
场景 YAML
  -> load_scenario_config()
  -> SimulationApplication.from_config()
  -> 装配 + 后端 + 场景 + 任务 + hooks
  -> SimulationLoop
  -> RobotSystemState / RobotSystemCommand
  -> 可选运行产物
```

应用层是组合根：它负责读取配置、生成或注入 MuJoCo XML、创建控制器、注册 hooks，并把后端
和控制器交给统一的仿真循环。业务逻辑应尽量留在 model、tasks、control、scenes 等模块中，
避免在脚本里堆临时分支。

## 主要模块边界

```text
continuum_sim.application
  场景 dataclass、YAML 加载器、SimulationApplication 组合根

continuum_sim.model
  机器人装配、底座位姿、多臂状态、tendon routing、bending-space 模型

continuum_sim.system
  ControlLayout、RobotSystemState、RobotSystemCommand 等系统层协议

continuum_sim.tasks
  轨迹、导航任务、擦拭路径、发动机导航计划解析

continuum_sim.control
  路点跟踪、导航、擦拭、发动机清洗、分阶段发动机导航和实验控制器

continuum_sim.dynamics
  实验性 PCC 降阶动力学；当前主要由旧 wiping runtime 使用

continuum_sim.backends
  analytic 和 MuJoCo 系统后端

continuum_sim.scenes
  发动机和结构化场景配置、查询和 MJCF 注入

continuum_sim.runtime
  后端无关循环、viewer、视频、记录器和诊断 hooks

continuum_sim.io
  场景运行产物、NPZ、metadata、图表、视频回放导出
```

## 场景配置职责

一个场景配置应该回答这些问题：

- 使用哪个机器人装配：`assembly_config_path`
- 使用哪个共享底层控制 profile：`low_level_control_path`
- 使用哪个后端：`backend.type`
- 是否需要 MuJoCo XML 注入：`source_xml_path`、`generated_xml_path`
- 使用哪个场景查询：`scene.engine_config_path` 或 `scene.structured_config_path`
- 执行什么任务：`task.type` 以及对应的 trajectory、mission、wiping、engine_navigation
- 运行时节拍和最长步数：`runtime.controller_dt_s`、`n_substeps`、`max_steps`
- 需要哪些观察器：`hooks.recorder`、`tendon_debug`、viewer、实时面板
- 是否写产物：`artifacts.enabled`、`save_npz`、`save_plots`、`save_gif`、`save_model`

## MuJoCo 组合路径

```text
source_xml_path
  -> 重设文件资源相对路径
  -> 可选保留指定空间臂
  -> 可选注入发动机或结构化场景
  -> 可选锁定固定底座 freejoint
  -> generated_xml_path
  -> MujocoSystemBackend
```

`generated_xml_path` 是输出位置，不应被当成手写源文件。固定基线 XML 位于 `assets/mujoco/`；
场景运行期间生成或覆盖的 XML 通常位于 `output/generated/` 或运行产物目录。

## 控制路径

主线控制链路：

```text
tracking / navigation / wiping / cleaning / engine-navigation 上层策略
  -> TaskStep(SystemTaskIntent, TaskStatus)
  -> UnifiedLowLevelController
  -> CoordinatedTrackingController（共享 Cartesian servo / observer 上层策略）
  -> executor active-subspace solve + observer active-subspace solve
  -> WholeBodyController（每臂独立 Jacobian / SVD / 正则化 / 限幅）
  -> ControlLayout 映射 base twist 和相容 tendon-rate
  -> 后端应用底座位姿和 tendon 目标
  -> RobotSystemState 回报末端位姿、tendon 状态、metadata
  -> hooks 记录诊断数据和运行产物
```

`SystemTaskIntent` 是上层到下层的稳定边界。位置模式携带目标位置和速度前馈；速度模式由底层把
位置伺服锚定到当前 TCP，只执行速度意图。Engine cleaning 使用后者，因此 task-space cleaning
controller 的闭环速度不会再与通用位置 P 重复叠加。

双臂控制不再把两臂任务堆叠到一次全局 SVD。executor solve 允许 base 和 executor，自身数学路径与
single 保持一致；observer solve 只允许 observer tendon。observer 的 collision-only 策略从实际中心线
最近点构造排斥速度，在 influence distance 外输出零速度，并且不能触发 executor freeze 或全局 hard stop。

共享低层参数只在 `configs/control/spatial_low_level.yaml` 定义，包含笛卡尔增益、速度上限、
whole-body 权重、奇异性策略、阻尼和 tendon/backend 限制开关。任务 YAML 保留路径、时序、
waypoint 推进、clearance、接触/力目标和阶段状态。旧配置可用 `task.tracking_control` 覆盖 profile，
该能力仅用于兼容，不建议新场景复制底层参数。

发动机导航使用 `StagedEngineNavigationController`，按预进入、插入、局部执行臂路径、回归和终止阶段推进，
并通过 metadata 暴露活动目标、底座路径、执行臂路径、observer ROI 等叠加层和运行产物需要的数据。
局部执行臂路径同样进入 `UnifiedLowLevelController`；单臂和双臂场景共用该状态机，observer 为可选角色。

需要注意：`control` 目录中还保留了旧 motor-space 差分 IK、navigation CBF-QP、
contact-triggered admittance、engine-cleaning task-space scaffold 和 dynamic adaptive impedance。
这些模块并不都作为 scenario 主入口的一等控制器启用。当前状态和迁移建议见
`docs/current_status.md`。

## Hooks 与调试数据

`runtime.hooks` 是调试便利性的核心模块，包含：

- `StateRecorderHook`：记录状态、目标、误差、force/contact、发动机导航阶段。
- `TendonDiagnosticHook`：按 stride 打印 tendon 目标误差和饱和信息。
- `MujocoReplayRecorderHook`：记录 qpos、qvel、mocap，用于离屏回放视频。
- `MujocoLiveVideoRecorderHook`：仿真过程中直接采集 MuJoCo 帧。
- `MujocoViewerHook`、`MatplotlibSystemViewerHook`：交互式查看。
- 实时面板：tendon、擦拭力、诊断数据。

运行产物按 `time_s`（state）和 `command_time_s`（command）分别对齐，逐臂保存 tendon target/actual、
requested/applied rate、实际速度和 actuator force；双臂命令 metadata 还保存 observer mode、最近点、
臂间距离、阈值和避碰速度，用于同步控制图。

新增调试字段时，优先放入控制器命令 metadata，再由 recorder/hooks 消费。这样不会把具体任务逻辑
塞进 IO 或 viewer 代码里。

## 可扩展性约定

- 新任务类型：先在 `tasks` 中定义 spec/plan，再在 `control` 中实现控制器，最后在 `application` 中组合。
- 新场景：优先实现结构化 YAML 和查询对象，MuJoCo 注入只负责渲染和碰撞表达。
- 新运行产物：优先扩展 `io.scenario_artifacts`，不要在控制器中直接写文件。
- 新可视化：优先通过 hooks 和 metadata 接入，不要让仿真循环依赖某个 viewer。
- 新配置字段：在 dataclass 中给出默认值和 `__post_init__` 校验，并同步更新
  `docs/configuration_reference.md`。
