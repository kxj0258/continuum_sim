# continuum_sim

`continuum_sim` 是面向空间连续体机械臂的 MuJoCo 仿真、任务控制和场景编排项目。清理后的主线入口统一为 `configs/scenarios/*.yaml`，每个场景文件同时声明机械臂模式、后端、任务、运行时、hooks 和产物输出。

## 主任务命令

当前保留 5 个主任务场景：

```powershell
python scripts/run_scenario.py configs/scenarios/mujoco_tracking.yaml
python scripts/run_scenario.py configs/scenarios/mujoco_navigation.yaml
python scripts/run_scenario.py configs/scenarios/engine_navigation.yaml
python scripts/run_scenario.py configs/scenarios/mujoco_wiping.yaml
python scripts/run_scenario.py configs/scenarios/mujoco_point_servo.yaml
```

`single` 和 `dual` 不再使用两套 YAML。需要切换单臂/双臂时，修改场景中的：

```yaml
scenario:
  arm_mode: dual   # dual 或 single
```

`single` 模式只保留主臂 executor；控制逻辑、低层参数、评分参数和执行链路与 `dual` 模式保持一致。`dual` 模式额外启用 observer 从臂的可视化、避碰和监控。

## 当前控制主线

清理后的控制流程只保留当前已达到预期的 MuJoCo 主线：

```text
scenario yaml
  -> SimulationApplication
  -> task controller
  -> UnifiedLowLevelController
  -> MuJoCo system backend
  -> hooks / artifacts / live diagnostics
```

主任务类型为：

- `tracking`：轨迹跟踪和点伺服共用该类型。
- `navigation`：移动基座导航与主臂跟随。
- `engine_navigation`：发动机场景中的 staged navigation。
- `wiping`：黑板 approach、contact wiping、retreat 擦拭任务。

analytic backend、旧 single/dual YAML、旧任务配置文件、旧 standalone runtime 和 engine cleaning 专用控制器已从主线中移除。`mujoco_wiping.yaml` 默认使用运动学混合力位策略，同时保留动态自适应阻抗和接触触发导纳作为 YAML 可选方案。

## 关键配置

共享低层控制参数位于：

```text
configs/control/mujoco_tracking_low_level.yaml
```

该文件集中定义：

- task-space servo 参数。
- tendon command / IK 参数。
- MuJoCo tendon position actuator 执行参数。
- 在线 waypoint reachability / execution 评分参数。
- 自动跳点阈值和窗口参数。

主场景说明见：

- `docs/main_scenarios.md`
- `docs/configuration_reference.md`
- `docs/online_waypoint_reachability.md`
- `docs/coordinate_conventions.md`

## 目录结构

```text
configs/scenarios/   当前主任务 YAML
configs/control/     共享低层控制参数
configs/robots/      单臂、双臂、移动基座和装配配置
configs/scenes/      wiping board、engine、rocket 等结构化场景
configs/tools/       工具/喷嘴/相机等任务附件配置
assets/mujoco/       MuJoCo XML 基础模型
scripts/             当前保留的场景运行、批量运行、视频导出和模型生成脚本
src/continuum_sim/   应用层、控制器、后端、运行循环、IO 和可视化代码
docs/                中文配置和主线说明文档
```

## 输出产物

默认输出目录：

```text
output/runs/<scenario_name>_<timestamp>/
```

常见内容包括：

- `result.npz`
- `metadata.json`
- `configs/`
- `model/`
- `plots/`
- `videos/`

`output/runs/` 是本地运行产物，不应提交。

## 手动验证建议

本项目不默认自动运行测试或仿真。修改后可按需手动执行：

```powershell
python scripts/run_scenario.py configs/scenarios/mujoco_tracking.yaml
python scripts/run_scenario.py configs/scenarios/mujoco_navigation.yaml
python scripts/run_scenario.py configs/scenarios/engine_navigation.yaml
python scripts/run_scenario.py configs/scenarios/mujoco_wiping.yaml
python scripts/run_scenario.py configs/scenarios/mujoco_point_servo.yaml
```

如需批量运行当前主场景：

```powershell
python scripts/run_all_scenarios.py
```
