# continuum_sim

`continuum_sim` 是一个基于 MuJoCo 的空间连续体机械臂仿真与控制项目。系统支持一条执行臂（`executor`）和一条观测臂（`observer`），每条臂包含 3 段、9 根驱动肌腱，可运行轨迹跟踪、点伺服、场景导航、发动机内部导航、接触擦拭和双臂手动控制。

## 主要功能

- 使用 `SimulationApplication` 从场景 YAML 组装机器人、场景、控制器、MuJoCo 后端、实时窗口和运行产物。
- 使用离散铰链 MuJoCo 模型模拟三段肌腱驱动连续体臂，同时使用弯曲空间模型描述每段 `kx、ky`。
- 支持固定基座和六自由度移动基座任务。
- 支持双臂协同、臂间避碰、环境避障、观测臂看向控制和视觉伺服。
- 支持发动机网格、入口、探索路径和碰撞控制几何的场景注入。
- 发动机使用不透明灰银色材质、镜面高光和中性补光；MuJoCo 主窗口与观测臂相机使用同一套场景材质和灯光。
- 执行臂末端装配 `15 × 15 × 8 mm` 六维力传感器和直径 `18 mm` 的包覆式球形擦拭工具。
- 擦拭控制直接读取 MuJoCo `force`/`torque` 传感器，提供置零、低通滤波、坐标变换、限幅和重力补偿。
- 提供三窗口手动控制工具：控制面板、MuJoCo 三维窗口、观测臂相机窗口。
- 提供单姿态 64 角点最坏情况分析和 10,000 姿态工作空间编码器精度统计，可导出 CSV、NPZ 与 TCP 误差热力图。
- 控制周期为 `0.02 s`，MuJoCo 步长为 `0.001 s`，每周期执行 20 个物理子步；启动时会检查两套时钟是否一致。

## 安装

需要 Python 3.11 或更高版本。

```powershell
python -m pip install -e ".[mujoco]"
```

开发和测试环境：

```powershell
python -m pip install -e ".[mujoco,dev]"
```

## 快速开始

启动双臂手动控制：

```powershell
python scripts/run_manual_control.py
```

运行编码器精度分析：

```powershell
python scripts/cal_accuracy.py
python scripts/cal_accuracy_workspace.py --samples 10000 --accuracy-deg 0.5 0.25 0.1 0.05
```

两套脚本直接读取执行臂、工具、场景和 MuJoCo 配置。工作空间脚本按配置的 `±30°` 总弯曲角范围采样，并使用九根物理肌腱行程筛除不可行姿态，结果写入 `output/accuracy_workspace/`。

运行一个自动任务：

```powershell
python scripts/run_scenario.py configs/scenarios/mujoco_tracking.yaml
```

批量运行所有非空闲任务：

```powershell
python scripts/run_all_scenarios.py
```

## 场景入口

| 场景 | 任务 | 默认装配 | 主要用途 |
| --- | --- | --- | --- |
| `mujoco_manual_control.yaml` | `idle` | 双臂六自由度移动基座 | 无发动机轻量场景、三窗口手动控制与基座调姿 |
| `mujoco_tracking.yaml` | `tracking` | 双臂固定基座 | 方形轨迹跟踪、双臂协同和实时诊断 |
| `mujoco_point_servo.yaml` | `tracking` | 单臂固定基座 | 单点伺服和控制链调试 |
| `mujoco_navigation.yaml` | `navigation` | 双臂移动基座 | 结构化场景导航、避障和相机观察 |
| `engine_navigation.yaml` | `engine_navigation` | 双臂移动基座 | 发动机入口接近、轴向插入和内部路径导航 |
| `mujoco_wiping.yaml` | `wiping` | 双臂固定基座 | 球形工具接触擦拭和六维力反馈 |

自动任务统一使用：

```powershell
python scripts/run_scenario.py <场景 YAML>
```

手动控制场景使用：

```powershell
python scripts/run_manual_control.py [场景 YAML]
```

## 系统链路

```text
场景 YAML
  -> SimulationApplication
  -> 任务控制器
  -> UnifiedLowLevelController
  -> 弯曲空间 / 雅可比 / 肌腱速率命令
  -> MuJoCo tendon position 执行层
  -> RobotSystemState
  -> 实时窗口、相机反馈和运行产物
```

共享低层控制参数位于：

```text
configs/control/mujoco_tracking_low_level.yaml
```

这里集中配置任务空间伺服、弯曲和肌腱命令、优先级任务、执行器内环、在线可达性评分及自动跳点。

## 手动控制

控制面板同时管理 `executor` 和 `observer`。每一段提供 `+kx/-kx/+ky/-ky` 按钮，方向定义在该段的机械臂局部弯曲坐标中，按钮直接修改对应曲率分量，不执行世界坐标或安装坐标逆解。基座区提供世界坐标中的 `X/Y/Z/Roll/Pitch/Yaw` 目标调节以及粗调、微调切换。

面板还提供九根肌腱目标滑块。`compatible` 模式会把滑动结果投影到弯曲子空间并同步关联肌腱；`raw tendon` 模式用于直接设置各肌腱。诊断区实时显示：

- 每段目标和实际 `kx、ky`，单位 `1/m`。
- 每段末端在 MuJoCo 世界坐标系中的实际位置，单位 `m`。
- 执行臂三轴力、三轴力矩和传感器饱和状态。

详细操作见 [双臂手动控制](docs/manual_control.md)。

## 发动机显示

发动机外观由 `configs/scenes/engine_scene.yaml` 的 `preview_visualization.visual_material` 配置。默认材质为：

```yaml
visual_mesh_rgba: [0.66, 0.68, 0.71, 1.0]
visual_material:
  name: engine_silver
  emission: 0.0
  specular: 0.72
  shininess: 0.48
```

场景构建时会将材质绑定到发动机可视网格，并加入中性主光、补光和 headlight，因此主窗口和相机渲染看到相同的灰银色表面。

## 执行臂末端附件

附件配置位于 `configs/tools/carbon_remover.yaml`：

- 六维力传感器：`15 × 15 × 8 mm`，中心位于裸臂末端前方 `4 mm`。
- 球形工具：直径 `18 mm`，球心相对传感器中心前移 `5 mm`；球体后端与传感器后表面相切，并与传感器主体重叠形成包覆效果。
- 球面 TCP：位于裸臂末端前方 `18 mm`。
- 传感器外壳只参与显示；球形工具参与碰撞。
- MuJoCo 在传感器 site 输出三轴力和三轴力矩，控制器使用沿接触法向的实测分量进行擦拭力闭环。

观测臂末端装有直径 `7.5 mm` 的可视半球相机外壳，直径为末端直径的一半；镜头作为独立蓝色圆面显示，成像相机位于外壳前端。

## 配置和目录

```text
configs/scenarios/   可运行场景
configs/control/     共享低层控制参数
configs/robots/      机械臂、基座和装配配置
configs/scenes/      结构化场景与发动机场景
configs/tools/       工具、喷嘴和相机附件
assets/              MuJoCo 基础模型、网格和发动机资产
scripts/             场景运行、手动控制、批量运行和视频导出
src/continuum_sim/   应用、控制、后端、模型、运行时和可视化代码
docs/                项目使用与配置说明
tests/               单元测试和 MuJoCo 集成测试
```

每次启用产物保存的任务会写入：

```text
output/runs/<scenario_name>_<timestamp>/
```

其中可包含状态与命令数据、运行元数据、配置副本、生成的 MuJoCo 模型、诊断图和视频。

## 文档

- [入门与代码结构](docs/getting_started.md)
- [场景说明](docs/main_scenarios.md)
- [配置参考](docs/configuration_reference.md)
- [坐标和命令约定](docs/coordinate_conventions.md)
- [双臂手动控制](docs/manual_control.md)
- [编码器精度与工作空间统计](docs/encoder_accuracy_calculation.md)
- [在线 waypoint 可达性评分](docs/online_waypoint_reachability.md)

## 功能验证

```powershell
python -m pytest tests/test_backend_timing.py tests/test_engine_material.py tests/test_tool_mjcf_adapter.py tests/test_manual_control_integration.py
```
