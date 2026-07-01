# continuum_sim

面向空间连续体机械臂的组合式仿真项目。当前主架构统一支持：

- 单臂或 executor/observer 双臂；
- world-frame 6D base twist；
- 控制器直接输出腱长变化速度；
- Analytic PCC 与 MuJoCo backend；
- primitive 场景避碰与发动机 MJCF 组合；
- tracking、navigation、wiping-path 任务；
- NPZ、指标、图表和 GIF 实验产物。

## 快速运行

在项目根目录执行：

```powershell
# 最小检查
python scripts/run_scenario.py configs/scenarios/single_analytic_smoke.yaml
python scripts/run_scenario.py configs/scenarios/single_mujoco_smoke.yaml

# 单臂/双臂 Analytic
python scripts/run_scenario.py configs/scenarios/single_analytic_preview.yaml
python scripts/run_scenario.py configs/scenarios/single_analytic_tracking.yaml
python scripts/run_scenario.py configs/scenarios/dual_analytic_tracking.yaml

# 单臂/双臂 MuJoCo
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

Python API 与脚本使用同一个组合入口：

```python
from continuum_sim.application import SimulationApplication

application = SimulationApplication.from_yaml(
    "configs/scenarios/dual_mujoco_tracking.yaml"
)
result = application.run()
print(application.last_artifacts.run_dir)
```

旧 `cli.py`、`main_config*.yaml` 和任务专用 runtime 不再是推荐入口。

## 运行结果

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

`result.npz` 使用命名字段，不依赖 9/18/24 维隐式切片，例如：

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

GIF 导出失败不会丢弃仿真数据，原因会写入
`videos/video_error.txt`。MuJoCo GIF 需要可用的离屏渲染环境和
`imageio`。

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

## 架构

```text
ScenarioConfig
  ├── RobotAssemblyConfig
  │   ├── Base: fixed | prescribed_twist
  │   └── Arms: executor + optional observer
  ├── Task Controller
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

系统控制变量为：

```text
single = [base_twist_6D, executor_tendon_rate_9D]
dual   = [base_twist_6D, executor_tendon_rate_9D, observer_tendon_rate_9D]
```

固定 base 时仍保留 6D 布局，但对应速度被约束为零。MuJoCo backend
负责积分腱长速度、执行速率与行程限幅，并将
`neutral_tendon_length + tendon_displacement` 写入位置 actuator。

坐标约定以 [docs/coordinate_conventions.md](docs/coordinate_conventions.md)
为准。

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

当前 wiping 使用几何接触距离代理，不代表可靠的 MuJoCo 腱索张力或真实
六维力传感器反馈。真实接触力闭环仍需后续标定接触模型。

## Viewer 与诊断

MuJoCo viewer 使用 `configs/mujoco_dual.yaml` 中的 camera、geom group 和
realtime 设置。`keep_viewer_open: true` 时，任务结束后等待用户关闭窗口。

Matplotlib viewer 显示：

- 单/双臂中心线；
- executor 当前目标；
- executor 末端轨迹；
- 目标轨迹。

`tendon_debug: true` 会采集每条臂的腱位移、腱速度、限幅信息以及控制器
提供的奇异性诊断。诊断数据当前随运行保留在内存；主要数值历史已经写入
`result.npz`。

## 主要配置

```text
configs/robots/spatial_arm_executor.yaml
configs/robots/spatial_arm_observer.yaml
configs/robots/assemblies/single_spatial.yaml
configs/robots/assemblies/dual_spatial.yaml
configs/scenarios/
configs/scenes/
configs/mujoco_dual.yaml
```

单臂系统直接复用双臂主臂 executor 的 spatial 配置。单臂 MuJoCo 场景由
双臂 MJCF 裁剪得到，裁剪过程会同步删除 observer body、tendon、actuator
和 sensor 引用。

## 发动机场景

Engine scene adapter 负责：

- 将机器人 MJCF 与 engine scene 组合；
- 重写移动后 XML 的 mesh 相对路径；
- 处理 MuJoCo STL 200,000 面限制；
- 注入 visual mesh；
- 注入实时控制使用的 primitive collision geoms。

当前实时控制只查询 primitive collision geoms，不对高面数 engine mesh
执行逐控制周期距离计算。

## 环境与手动验证

推荐在已有 `continuum_sim` 环境中运行。安装方式取决于本地环境，本项目
不会在运行 scenario 时自动安装依赖。

建议修改后手动执行：

```powershell
python -m compileall src scripts/run_scenario.py
pytest tests/test_scenario_mujoco_composition.py

python scripts/run_scenario.py configs/scenarios/single_analytic_tracking.yaml
python scripts/run_scenario.py configs/scenarios/dual_analytic_tracking.yaml
python scripts/run_scenario.py configs/scenarios/single_mujoco_smoke.yaml
python scripts/run_scenario.py configs/scenarios/dual_mujoco_tracking.yaml
python scripts/run_scenario.py configs/scenarios/single_engine_tracking.yaml
```

重点人工检查：

- `output/runs/` 是否包含 NPZ、metadata、plots 和 GIF；
- 单/双臂字段维度是否分别符合命名布局；
- viewer 是否能看到连续体及目标；
- observer 是否跟踪 executor/ROI 且没有穿过 executor；
- engine primitive 的坐标、缩放和 clearance 是否合理；
- base twist 和腱长速度是否触发预期限幅。

## 当前限制

- base 使用 prescribed world-frame twist，不是受力驱动的动态底座；
- whole-body solver 为加权阻尼最小二乘，尚未升级为分层约束 QP；
- primitive 避碰是采样式近似；
- wiping 接触力仍是距离代理；
- MuJoCo 腱索张力不作为可靠控制反馈；
- 旧 motor-chain viewer 不迁移，因为新空间连续体链路已删除电机层。
