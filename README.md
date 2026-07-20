# continuum_sim

`continuum_sim` 是一个面向空间连续体机械臂的 MuJoCo 场景仿真与任务控制项目。当前主线以 `SimulationApplication` 为唯一应用入口，通过 `configs/scenarios/*.yaml` 描述机器人装配、MuJoCo 后端、任务、运行时 hooks 和输出产物。

## 当前主线

主线运行链路如下：

```text
场景 YAML
  -> SimulationApplication
  -> 任务控制器
  -> UnifiedLowLevelController
  -> MuJoCo system backend
  -> hooks / artifacts / live diagnostics
```

当前保留 5 个场景入口：

```powershell
python scripts/run_scenario.py configs/scenarios/mujoco_tracking.yaml
python scripts/run_scenario.py configs/scenarios/mujoco_navigation.yaml
python scripts/run_scenario.py configs/scenarios/engine_navigation.yaml
python scripts/run_scenario.py configs/scenarios/mujoco_wiping.yaml
python scripts/run_scenario.py configs/scenarios/mujoco_point_servo.yaml
```

`single` 和 `dual` 不再维护两套任务文件。需要切换单臂/双臂时，只修改场景中的：

```yaml
scenario:
  arm_mode: dual   # dual 或 single
```

`single` 模式只保留 executor 主臂；`dual` 模式启用 executor 主臂和 observer 从臂，并可叠加避碰、观察相机和可视化反馈。

## 场景说明

| 场景 | 任务类型 | 默认模式 | 用途 |
| --- | --- | --- | --- |
| `mujoco_tracking.yaml` | `tracking` | `dual` | 固定基座轨迹跟踪 |
| `mujoco_navigation.yaml` | `navigation` | `dual` | 结构化火箭喷管入口导航 |
| `engine_navigation.yaml` | `engine_navigation` | `dual` | 发动机场景入口、插入和局部路径导航 |
| `mujoco_wiping.yaml` | `wiping` | `dual` | 黑板接触擦拭任务 |
| `mujoco_point_servo.yaml` | `tracking` | `single` | 单点伺服调试 |

共享低层控制参数位于：

```text
configs/control/mujoco_tracking_low_level.yaml
```

该文件集中维护 task-space servo、IK/tendon command、MuJoCo tendon position actuator、在线可达性评分和自动跳点参数。

## 输出产物

默认输出目录：

```text
output/runs/<scenario_name>_<timestamp>/
```

常见内容：

- `result.npz`：状态、命令、诊断字段。
- `metadata.json`：运行摘要、指标、输出文件索引和 artifact 错误。
- `configs/`：本次运行使用的配置副本。
- `model/`：生成或复制的 MuJoCo 模型。
- `plots/`：轨迹、控制层诊断、PCC/MuJoCo 诊断和 live diagnostics 最终图。
- `videos/`：MuJoCo live/replay 视频和 observer camera 视频。

`mujoco_wiping.yaml` 会打开擦拭力监控窗口；开启 `artifacts.save_plots: true` 时会保存最终接触力图。开启 `show_live_diagnostics_panel` 且 `save_plots` 时，会保存 `plots/live_diagnostics_panel.png`。

## 目录结构

```text
configs/scenarios/   当前主任务 YAML
configs/control/     共享低层控制参数
configs/robots/      单臂、双臂、移动基座和装配配置
configs/scenes/      wiping board、engine、rocket 等场景配置
configs/tools/       末端工具、喷嘴、相机附件配置
assets/              MuJoCo XML、网格、发动机模型和 CAD 源文件
scripts/             场景运行、批量运行、视频导出和模型生成脚本
src/continuum_sim/   应用层、控制器、后端、运行循环、IO 和可视化代码
docs/                当前中文说明文档
tests/               当前主线仍维护的单元和集成测试
```

## 已清理的旧入口

当前项目不再维护以下旧入口：

- analytic backend 场景入口。
- `configs/tasks/*.yaml` 旧任务配置。
- engine cleaning 专用任务配置、控制器和路径接口。
- `continuum_sim.runtime.hooks` / `hooks_impl` 兼容导出层。
- 迁移过程文档、阶段性 handoff、findings 和 progress 记录。

如果需要新增能力，请优先接入当前 `configs/scenarios/*.yaml -> SimulationApplication` 主线。

## 相关文档

- `docs/main_scenarios.md`：5 个主场景的职责和运行方式。
- `docs/configuration_reference.md`：场景 YAML 关键字段说明。
- `docs/coordinate_conventions.md`：坐标、命令和法向约定。
- `docs/online_waypoint_reachability.md`：在线 waypoint 可达性评分。

## 手动验证建议

本项目不会默认自动运行测试或仿真。修改后可按需手动执行：

```powershell
python scripts/run_scenario.py configs/scenarios/mujoco_tracking.yaml
python scripts/run_scenario.py configs/scenarios/mujoco_navigation.yaml
python scripts/run_scenario.py configs/scenarios/engine_navigation.yaml
python scripts/run_scenario.py configs/scenarios/mujoco_wiping.yaml
python scripts/run_scenario.py configs/scenarios/mujoco_point_servo.yaml
```

批量运行当前主场景：

```powershell
python scripts/run_all_scenarios.py
```
