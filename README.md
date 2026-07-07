# continuum_sim

`continuum_sim` 是面向空间连续体机械臂的仿真、控制和场景编排项目。当前推荐入口是
`configs/scenarios/*.yaml`：一个场景配置同时声明机器人装配、后端、场景、任务、运行时、hooks
和运行产物。

## 推荐入口

```powershell
python scripts/run_scenario.py configs/scenarios/<scenario>.yaml
```

Python 代码中也可以直接组合应用层：

```python
from continuum_sim.application import SimulationApplication

application = SimulationApplication.from_yaml(
    "configs/scenarios/dual_engine_navigation.yaml"
)
result = application.run()
print(len(result.states))
print(application.last_artifacts.run_dir)
```

## 项目结构

```text
configs/scenarios/         推荐运行入口，组合装配、后端、场景、任务、运行时和 hooks
configs/robots/            单臂、双臂、移动底座和装配配置
configs/scenes/            结构化场景，例如发动机、火箭喷管、擦拭板
configs/tasks/             旧运行时兼容配置和任务片段
assets/mujoco/             固定 MuJoCo XML 基线模型
src/continuum_sim/application
                            场景解析和 SimulationApplication 组合根
src/continuum_sim/control  跟踪、导航、擦拭、发动机导航控制器
src/continuum_sim/model    机器人参数、装配、tendon、bending-space 模型
src/continuum_sim/runtime  后端无关仿真循环和 hooks
src/continuum_sim/io       运行产物、图表、metadata、视频导出
docs/                      架构、配置、调试、双臂和当前能力状态说明
```

## 常用场景

```powershell
# 轻量 analytic 冒烟场景
python scripts/run_scenario.py configs/scenarios/single_analytic_smoke.yaml

# MuJoCo 冒烟场景
python scripts/run_scenario.py configs/scenarios/single_mujoco_smoke.yaml

# 单臂/双臂跟踪
python scripts/run_scenario.py configs/scenarios/single_analytic_tracking.yaml
python scripts/run_scenario.py configs/scenarios/dual_analytic_tracking.yaml
python scripts/run_scenario.py configs/scenarios/single_mujoco_tracking.yaml
python scripts/run_scenario.py configs/scenarios/dual_mujoco_tracking.yaml

# 导航、擦拭和发动机场景
python scripts/run_scenario.py configs/scenarios/single_mujoco_navigation.yaml
python scripts/run_scenario.py configs/scenarios/dual_mujoco_navigation.yaml
python scripts/run_scenario.py configs/scenarios/single_mujoco_wiping.yaml
python scripts/run_scenario.py configs/scenarios/dual_mujoco_wiping.yaml
python scripts/run_scenario.py configs/scenarios/single_engine_cleaning.yaml
python scripts/run_scenario.py configs/scenarios/dual_engine_navigation.yaml
```

带 `hooks.viewer: mujoco` 或 `hooks.viewer: matplotlib` 的场景会打开窗口。自动化排查时建议先把
viewer 设为 `none`，只保留 recorder、调试面板或运行产物。

## 控制约定

常规机械臂命令使用每臂 6 维 bending-space：

```text
b = [kx_1, ky_1, kx_2, ky_2, kx_3, ky_3]
q = S_b b
delta_l = C_b b,  C_b = C_q S_b
```

轴向应变默认固定为 0，并与 MuJoCo 配置 `tendon_model.include_axial_strain: false`
保持一致。跟踪、导航、擦拭、发动机导航和 observer 协同任务都会先求 `b_dot`，
再由 `C_b` 生成 tendon 相容速度。

## 调试入口

优先从场景配置开始排查：

1. 检查 `configs/scenarios/<name>.yaml` 的 `backend`、`task`、`runtime`、`hooks`。
2. 打开 `artifacts.enabled`、`save_npz`、`save_plots`，关闭 viewer 复现无窗口问题。
3. 需要看实时误差时再开启 `show_live_diagnostics_panel` 或 MuJoCo viewer 叠加层。
4. 查看 `output/runs/<scenario>_<timestamp>/metadata.json`、`result.npz`、`plots/`、`videos/video_error.txt`。

更多细节见 [docs/debugging_guide.md](docs/debugging_guide.md)。

## 运行产物

默认运行产物写入：

```text
output/runs/<scenario>_<timestamp>/
  result.npz
  metadata.json
  configs/
  model/
  plots/
  videos/simulation.gif
```

`output/runs/` 是本地运行产物，不应提交。`output/generated/` 中已有文件是当前仓库保留的生成基线，
部分场景配置会引用这些路径作为 MuJoCo XML 输出位置。

## 文档索引

- [docs/architecture_overview.md](docs/architecture_overview.md)：模块边界和数据流。
- [docs/configuration_reference.md](docs/configuration_reference.md)：场景、MuJoCo、任务配置字段。
- [docs/debugging_guide.md](docs/debugging_guide.md)：调试 hooks、运行产物和手动检查建议。
- [docs/coordinate_conventions.md](docs/coordinate_conventions.md)：坐标系约定。
- [docs/dual_arm_mujoco_landing.md](docs/dual_arm_mujoco_landing.md)：双臂 spatial tendon、孔位和 MuJoCo 资产说明。
- [docs/current_status.md](docs/current_status.md)：当前动力学、未接入主接口的控制能力和迁移建议。

## 手动验证建议

本项目默认不自动运行测试或仿真。修改后建议按需要手动执行：

```powershell
pytest tests/test_robot_config.py tests/test_scenario_artifacts.py
pytest tests/test_engine_navigation.py tests/test_staged_engine_navigation.py
python scripts/run_scenario.py configs/scenarios/single_analytic_smoke.yaml
python scripts/run_scenario.py configs/scenarios/dual_engine_navigation.yaml
```

如果只想确认 MuJoCo 渲染环境，再单独运行：

```powershell
python scripts/check_mujoco_offscreen_renderer.py configs/mujoco_dual.yaml
```
