# continuum_sim

面向空间连续体机械臂的组合式仿真项目。当前推荐使用
`scripts/run_scenario.py + configs/scenarios/*.yaml` 作为唯一实验入口；
Python 代码中使用 `continuum_sim.application.SimulationApplication` 调用同一套组合逻辑。

## 快速运行

在项目根目录执行。MuJoCo 场景会打开 viewer 的配置可能需要手动关闭窗口。

```powershell
# 最小冒烟场景
python scripts/run_scenario.py configs/scenarios/single_analytic_smoke.yaml
python scripts/run_scenario.py configs/scenarios/single_mujoco_smoke.yaml

# 单臂 / 双臂 Analytic
python scripts/run_scenario.py configs/scenarios/single_analytic_preview.yaml
python scripts/run_scenario.py configs/scenarios/single_analytic_tracking.yaml
python scripts/run_scenario.py configs/scenarios/dual_analytic_tracking.yaml

# 单臂 / 双臂 MuJoCo
python scripts/run_scenario.py configs/scenarios/single_mujoco_view.yaml
python scripts/run_scenario.py configs/scenarios/single_mujoco_tracking.yaml
python scripts/run_scenario.py configs/scenarios/dual_mujoco_tracking.yaml

# Navigation 与 wiping path
python scripts/run_scenario.py configs/scenarios/single_analytic_navigation.yaml
python scripts/run_scenario.py configs/scenarios/single_analytic_wiping.yaml
python scripts/run_scenario.py configs/scenarios/single_mujoco_navigation.yaml
python scripts/run_scenario.py configs/scenarios/dual_mujoco_navigation.yaml
python scripts/run_scenario.py configs/scenarios/single_mujoco_wiping.yaml
python scripts/run_scenario.py configs/scenarios/dual_mujoco_wiping.yaml

# 发动机场景
python scripts/run_scenario.py configs/scenarios/single_engine_tracking.yaml
python scripts/run_scenario.py configs/scenarios/dual_engine_tracking.yaml
```

Python API：

```python
from continuum_sim.application import SimulationApplication

application = SimulationApplication.from_yaml(
    "configs/scenarios/dual_mujoco_tracking.yaml"
)
result = application.run()
print(len(result.states))
print(application.last_artifacts.run_dir)
```

## 项目作用

项目用于组合和评估空间连续体机械臂仿真能力：

- 单臂与 executor/observer 双臂系统；
- world-frame 6D base twist；
- 直接 tendon-length-rate 控制；
- Analytic PCC 与 MuJoCo backend；
- primitive 场景避碰、发动机 MJCF 组合；
- tracking、navigation、wiping 任务；
- NPZ、metadata、图表、组合 MJCF 和可选 GIF 产物。

旧 `cli.py` 入口、`configs/main_config*.yaml` 索引式入口和围绕旧 CLI 的说明文档已清理。
底层 task/runtime 模块仍保留，用于复用已有配置解析、控制器和测试覆盖。

## 运行产物

非 idle 场景默认保存到：

```text
output/runs/<scenario>_<timestamp>/
├── result.npz
├── metadata.json
├── configs/
│   ├── scenario.yaml
│   ├── assembly.yaml
│   └── mujoco.yaml
├── model/
│   └── scene.xml
├── plots/
│   ├── trajectory.png
│   ├── tracking_error.png
│   ├── arm_*_tendon_displacement_m.png
│   ├── min_clearance_m.png
│   └── contact_distance_m.png
└── videos/
    ├── simulation.gif
    └── video_error.txt
```

`result.npz` 使用命名字段，避免依赖 9/18/24 维隐式切片。常见字段包括：

```text
time_s
base_position_m
base_quat_wxyz
base_twist_world
arm_executor_tip_position_m
arm_executor_tendon_displacement_m
arm_executor_command_rate_mps
arm_observer_tip_position_m
target_position_m
tracking_error_m
min_clearance_m
qpos / qvel
```

GIF 导出失败不会丢弃数值数据；失败原因写入 `videos/video_error.txt`。
MuJoCo GIF 需要可用的离屏渲染环境和 `imageio`。

产物行为可在 scenario 中配置：

```yaml
scenario:
  artifacts:
    enabled: true
    output_root: ../../output/runs
    save_npz: true
    save_plots: true
    save_gif: true
    save_model: true
    video_fps: 20
    video_stride: 5
```

## 架构概览

```text
ScenarioConfig
  ├── RobotAssemblyConfig
  │   ├── Base: fixed | prescribed_twist
  │   └── Arms: executor + optional observer
  ├── Task Controller
  │   ├── idle
  │   ├── tracking
  │   ├── navigation
  │   └── wiping
  ├── SystemBackend
  │   ├── AnalyticSystemBackend
  │   └── MujocoSystemBackend
  ├── SceneQuery
  │   ├── StructuredSceneQuery
  │   └── EnginePrimitiveSceneQuery
  ├── SimulationLoop
  └── Hooks + ScenarioArtifactWriter
```

系统命令布局：

```text
single = [base_twist_6D, executor_tendon_rate_9D]
dual   = [base_twist_6D, executor_tendon_rate_9D, observer_tendon_rate_9D]
```

固定 base 仍保留 6D 布局，但对应速度被约束为零。MuJoCo backend 负责积分腱长速度、
执行速率/行程限幅，并将 `neutral_tendon_length + tendon_displacement`
写入位置 actuator。坐标约定以 [docs/coordinate_conventions.md](docs/coordinate_conventions.md) 为准。

## 模块职责

- `src/continuum_sim/application`：scenario 解析、组合根和主运行 API。
- `src/continuum_sim/system`：整机状态、命令布局和组合接口。
- `src/continuum_sim/model`：机器人参数、装配、移动基座、双臂、腱路径和安装框架。
- `src/continuum_sim/kinematics`：PCC FK、中心线采样、Jacobian、SDF 与整机运动学。
- `src/continuum_sim/control`：scenario 控制器、differential IK、whole-body、navigation、wiping 与阻抗控制实验。
- `src/continuum_sim/backends`：Analytic 与 MuJoCo backend，以及 MJCF/PCC 适配。
- `src/continuum_sim/scenes`：结构化场景、发动机场景、collision primitive 和 MJCF 注入。
- `src/continuum_sim/runtime`：仿真循环、hooks，以及仍被测试/脚本复用的低层 MuJoCo runtime。
- `src/continuum_sim/tasks`：旧 task YAML loader 和路径生成逻辑，作为底层能力保留。
- `src/continuum_sim/io`：scenario 产物与低层 run artifact 保存。
- `src/continuum_sim/visualization`：绘图、viewer、视频和诊断面板。
- `scripts/`：当前 scenario 入口、资产检查、MJCF 构建、报告/导出辅助脚本。
- `configs/scenarios/`：推荐实验入口配置。
- `configs/robots/`、`configs/systems/`、`configs/scenes/`、`configs/tasks/`：机器人、系统、场景和底层任务配置。
- `assets/`：MuJoCo XML、STL/GLB mesh、发动机模型和 CAD 源文件。

## 任务能力

### Tracking

- executor 世界坐标位置跟踪；
- observer 跟踪 executor 与 engine ROI 的组合目标；
- executor/observer 中心线避碰；
- primitive 场景距离任务；
- Jacobian 数值秩、条件数、自适应阻尼和速度缩放；
- 轨迹、误差、腱长与 GIF 输出。

### Navigation

- 有序 waypoint 推进；
- 连续体中心线最小间隙查询；
- primitive 避障任务；
- 间隙违规记录与可选提前终止；
- 最小间隙曲线输出。

### Wiping

- approach/contact/retract 阶段标记；
- 擦拭路径跟踪；
- 场景可查询时记录接触距离和接触误差；
- 接触距离曲线输出。

当前 wiping 使用几何接触距离代理，不代表可靠的 MuJoCo 腱索张力或真实六维力传感器反馈。
真实接触力闭环仍需后续标定接触模型。

## 发动机场景

Engine scene adapter 负责：

- 将机器人 MJCF 与 engine scene 组合；
- 重写移动后 XML 的 mesh 相对路径；
- 处理 MuJoCo STL 200,000 面限制；
- 注入 visual mesh；
- 注入实时控制使用的 primitive collision geoms。

当前实时控制只查询 primitive collision geoms，不在控制周期中对高面数 engine mesh 做距离计算。

## 环境

推荐在已有 `continuum_sim` 环境中运行。项目不会在 scenario 运行时自动安装依赖。
`mujoco` 是可选依赖；缺少 MuJoCo 时只能使用 analytic 场景和不依赖 MuJoCo 的工具。

## 手动验证建议

本次仓库清理没有自动运行验证。建议你按需要手动执行：

```powershell
python -m compileall src scripts/run_scenario.py
pytest tests/test_scenario_import_boundaries.py tests/test_scenario_mujoco_composition.py tests/test_run_artifacts.py

python scripts/run_scenario.py configs/scenarios/single_analytic_smoke.yaml
python scripts/run_scenario.py configs/scenarios/single_analytic_tracking.yaml
python scripts/run_scenario.py configs/scenarios/dual_analytic_tracking.yaml
python scripts/run_scenario.py configs/scenarios/single_mujoco_smoke.yaml
python scripts/run_scenario.py configs/scenarios/dual_mujoco_tracking.yaml
python scripts/run_scenario.py configs/scenarios/single_engine_tracking.yaml
```

重点人工检查：

- `output/runs/` 是否包含 NPZ、metadata、plots、model 和可选 GIF；
- 单/双臂字段维度是否符合命名布局；
- viewer 是否能看到连续体、目标和场景；
- observer 是否跟踪 executor/ROI 且没有穿过 executor；
- engine primitive 的坐标、缩放和 clearance 是否合理；
- base twist 和腱长速度是否触发预期限幅。

## 当前限制

- base 使用 prescribed world-frame twist，不是受力驱动的动态底座；
- whole-body solver 为加权阻尼最小二乘，尚未升级为分层约束 QP；
- primitive 避碰是采样式近似；
- wiping 接触力仍是距离代理；
- MuJoCo 腱索张力不作为可靠控制反馈；
- 旧 motor-chain viewer 不再作为项目入口维护，因为当前空间连续体主链已删除电机层。
