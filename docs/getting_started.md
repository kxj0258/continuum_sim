# 入门与代码结构

## 安装

项目需要 Python 3.11 或更高版本。

```powershell
python -m pip install -e ".[mujoco]"
```

安装测试依赖：

```powershell
python -m pip install -e ".[mujoco,dev]"
```

## 第一次运行

手动操作双臂：

```powershell
python scripts/run_manual_control.py
```

运行轨迹跟踪：

```powershell
python scripts/run_scenario.py configs/scenarios/mujoco_tracking.yaml
```

运行擦拭任务：

```powershell
python scripts/run_scenario.py configs/scenarios/mujoco_wiping.yaml
```

## 目录地图

```text
configs/
  scenarios/       完整运行场景
  control/         低层控制、优先级和执行层参数
  robots/          机械臂、安装位姿、基座和装配
  scenes/          结构化环境、擦拭板和发动机
  tools/           执行工具、喷嘴和 observer 相机
  dynamics/        擦拭动力学参数
assets/
  mujoco/          机械臂基础 MJCF
  meshes/          双臂可视网格
  engine/          发动机可视和碰撞资产
scripts/
  run_scenario.py              自动任务入口
  run_manual_control.py        三窗口手动控制入口
  run_all_scenarios.py         批量任务入口
  export_replay_video.py       回放视频导出
  build_mujoco_dual_arm_model.py 机械臂模型生成
src/continuum_sim/
  application/      场景解析和应用组装
  control/          任务控制、优先级控制和肌腱命令
  model/            PCC、离散运动学、安装位姿和弯曲模型
  execution/        肌腱速率到 actuator 位置目标
  backends/         MuJoCo 物理与系统状态后端
  scenes/           场景、发动机、工具和相机 MJCF 注入
  runtime/          运行循环、hooks、相机和实时窗口
  visualization/    手动控制和诊断面板
  io/               运行产物
tests/               单元和集成测试
```

## 应用组装

`SimulationApplication.from_yaml()` 读取一个场景文件，并完成以下组装：

```text
场景配置
  -> 选择固定/移动、单臂/双臂装配
  -> 读取 MuJoCo 求解器配置
  -> 从基础 MJCF 注入环境、发动机、工具、传感器和相机
  -> 构建 MujocoSystemBackend
  -> 构建任务控制器和 UnifiedLowLevelController
  -> 构建运行 hooks 与 artifacts
  -> SimulationLoop
```

关键入口：

- `src/continuum_sim/application/application.py`：应用对象。
- `src/continuum_sim/application/scenario.py`：场景配置数据结构和解析。
- `src/continuum_sim/application/backend_factory.py`：MJCF 组合与后端构建。
- `src/continuum_sim/application/controller_factory.py`：任务控制器构建。
- `src/continuum_sim/application/hook_factory.py`：实时窗口、相机和记录器构建。

## 机器人模型

每条空间连续体臂包含三段，每段由四个离散链接和两自由度弯曲关节近似。每段配置三根肌腱，整条臂共九根肌腱。

控制层使用每段两个弯曲分量：

```text
b = [kx1, ky1, kx2, ky2, kx3, ky3]
```

弯曲模型提供：

- 从分段曲率生成九维肌腱位移。
- 从肌腱位移估计实际分段曲率。
- 将九维肌腱目标投影到弯曲子空间。
- 在肌腱行程和速率限制内生成命令。

MuJoCo 系统后端读取关节、肌腱、执行器、site、相机和传感器数据，组装为 `RobotSystemState`。

## 状态和命令

`RobotSystemState` 包含：

- 仿真时间。
- 基座位姿和速度。
- 每条臂的裸臂末端位姿。
- 每段末端位姿。
- 肌腱位移、速度、目标和执行器力。
- 工具 TCP 位姿。
- 执行臂六维力状态。
- 任务、视觉、避碰和执行层诊断 metadata。

每条臂的标准底层命令为九维肌腱速率。执行层将速率积分为 MuJoCo tendon position actuator 的目标位置。

## 控制链路

```text
任务控制器
  -> 生成目标点、方向、接触要求和 observer 意图
  -> UnifiedLowLevelController
  -> 任务空间误差和速度命令
  -> 主臂/从臂优先级任务与避碰
  -> 系统雅可比和弯曲空间映射
  -> 九维肌腱速率
  -> BendingRateServo / tendon position target
  -> MuJoCo 物理步进
```

共享参数文件 `configs/control/mujoco_tracking_low_level.yaml` 管理：

- `task_space_servo`：位置增益、速度上限和阻尼。
- `tendon_command`：雅可比求解、正则化、弯曲和肌腱限制。
- `priority_stack`：主任务、臂间避碰、环境避障和 observer 任务顺序。
- `tendon_inner_loop`：速率滤波、积分、目标领先限制和执行器力约束。
- `online_reachability`：目标收敛评分和自动跳点。

## 双臂职责

- `executor`：执行轨迹、导航、插入和擦拭任务。
- `observer`：维持安全距离、调整观察方向并输出末端相机图像。

双臂场景会计算臂间链接距离，在影响距离内生成避碰任务。observer 还可以根据任务配置执行看向 executor TCP、观察区域或视觉目标的控制。

## 场景任务

### 手动控制

`mujoco_manual_control.yaml` 使用 `idle` 任务。手动程序直接向两条臂发送肌腱速率命令，并同步控制面板、MuJoCo viewer 和 observer 相机。详见 [双臂手动控制](manual_control.md)。

### 轨迹跟踪与点伺服

`mujoco_tracking.yaml` 跟踪多点轨迹；`mujoco_point_servo.yaml` 运行单点目标。控制器根据位置误差、前馈速度、waypoint 容差和在线评分推进目标。

### 结构化场景导航

`mujoco_navigation.yaml` 使用移动基座分阶段接近目标区域，同时执行机械臂跟踪、observer 协同和场景避障。

### 发动机导航

`engine_navigation.yaml` 加载发动机网格、入口、局部坐标标注和探索路径。任务包括基座接近、入口对齐、轴向插入和路径跟随。

发动机可视网格使用 `engine_silver` 材质。MJCF 构建时写入不透明灰银色、镜面反射参数和中性灯光，viewer 与 observer 相机共享该场景。

### 擦拭

`mujoco_wiping.yaml` 从擦拭板 surface patch 生成往复路径。执行臂使用球面 TCP 做任务空间控制，并使用末端六维力传感器测量表面法向力。

控制器支持力/位混合、动态自适应阻抗和接触触发导纳。默认场景使用 `hybrid_force_position` 与 `tool_wrench_sensor`。

## 末端六维力工具

`configs/tools/carbon_remover.yaml` 定义：

```text
裸臂 tip
  -> 15 × 15 × 8 mm 六维力传感器
  -> 隐藏连接垫
  -> 直径 18 mm 包覆式球形擦拭工具
  -> 球面 TCP（距裸臂 tip 18 mm）
```

传感器 body、site、球体 geom、TCP site、`force` sensor 和 `torque` sensor 会自动注入生成的 MJCF。系统后端完成复位置零、滤波、坐标转换、限幅和重力补偿，并把结果写入 `ArmSystemState.tool_wrench`。

擦拭控制器使用工具 TCP 计算接触距离和任务误差。系统雅可比包含从裸臂 tip 到工具 TCP 的杠杆臂项。

## 手动控制界面

`scripts/run_manual_control.py` 构建三个窗口：

- Matplotlib 控制和诊断面板。
- MuJoCo passive viewer。
- observer 相机图像窗口。

每条臂的三段均可直接执行 `+kx/-kx/+ky/-ky`。九根肌腱滑块支持弯曲子空间投影。界面实时显示目标/实际曲率、三段末端世界位置，以及执行臂六维力。

## 仿真时钟

场景配置：

```yaml
runtime:
  controller_dt_s: 0.02
  n_substeps: 20
```

MuJoCo 配置：

```yaml
solver:
  timestep: 0.001
```

因此一个控制周期恰好推进 `20 × 0.001 s = 0.02 s`。`backend_factory` 在加载时检查该关系。

## 实时窗口和输出

场景 `hooks` 可以启用：

- MuJoCo 或 Matplotlib viewer。
- observer 相机。
- 肌腱监控。
- 擦拭力监控。
- 综合控制诊断。
- 状态和命令记录器。

场景 `artifacts` 控制 `npz`、元数据、配置副本、模型、诊断图、GIF 和 MP4。输出目录为：

```text
output/runs/<scenario_name>_<timestamp>/
```

## 修改项目的常用入口

### 修改目标和场景行为

编辑 `configs/scenarios/*.yaml` 的 `task`、`scene` 和 `runtime`。

### 修改控制器响应

编辑 `configs/control/mujoco_tracking_low_level.yaml` 的任务空间增益、速度限制、雅可比正则化、优先级和执行器内环。

### 修改发动机外观或位置

编辑 `configs/scenes/engine_scene.yaml` 的：

- `engine.pose`：发动机位置与姿态。
- `preview_visualization.visual_mesh_rgba`：灰银底色和透明度。
- `preview_visualization.visual_material`：镜面反射和高光。
- `regions`、`exploration_start`、`exploration_paths`：发动机局部标注。

### 修改工具

编辑 `configs/tools/carbon_remover.yaml` 的传感器尺寸、球体半径、TCP、质量、摩擦、滤波和量程。MJCF 注入逻辑位于 `src/continuum_sim/scenes/tool_mjcf_adapter.py`，力状态读取位于 `src/continuum_sim/backends/mujoco_system_backend.py`。

### 新增场景

1. 在 `configs/scenarios/` 新建 YAML。
2. 选择 `arm_mode`、MuJoCo 配置和基础 XML。
3. 配置 `scene`、`task`、`runtime`、`hooks` 和 `artifacts`。
4. 保证 `n_substeps × timestep = controller_dt_s`。
5. 使用 `scripts/run_scenario.py` 运行。

### 新增任务类型

任务配置解析位于 `application/scenario.py`，控制器创建位于 `application/controller_factory.py`，任务控制器位于 `control/`。任务输出应转换为统一的系统命令，使后端、hooks 和 artifacts 可以使用同一状态接口。

## 功能测试

发动机材质、工具注入、传感器读取、时钟一致性和手动场景集成测试：

```powershell
python -m pytest tests/test_backend_timing.py tests/test_engine_material.py tests/test_tool_mjcf_adapter.py tests/test_manual_control_integration.py
```

场景和相机相关测试：

```powershell
python -m pytest tests/test_scenario_mujoco_composition.py tests/test_observer_camera_hook.py tests/test_engine_navigation_application.py
```
