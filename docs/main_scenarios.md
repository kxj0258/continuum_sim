# 主场景说明

当前项目只维护 `configs/scenarios/` 下的 5 个主场景。所有主场景都通过 `scripts/run_scenario.py` 进入 `SimulationApplication`，不再使用旧的 `configs/tasks/*.yaml` 任务配置。

## 统一运行入口

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

## 单臂和双臂

每个场景通过 `scenario.arm_mode` 选择装配：

```yaml
scenario:
  arm_mode: dual   # dual 或 single
```

- `dual`：启用 executor 和 observer 两条臂。
- `single`：只保留 executor，适合点伺服、基础跟踪和快速调参。
- 移动基座任务会自动选择 mobile assembly；固定基座任务选择 fixed assembly。

## 场景列表

| 场景文件 | 任务类型 | 默认模式 | 主要能力 |
| --- | --- | --- | --- |
| `mujoco_tracking.yaml` | `tracking` | `dual` | 方形轨迹跟踪、observer 避碰、live diagnostics |
| `mujoco_navigation.yaml` | `navigation` | `dual` | 移动基座阶段化接近、结构化场景避障、observer 相机 |
| `engine_navigation.yaml` | `engine_navigation` | `dual` | 发动机入口定位、轴向插入、中途局部路径和端点局部路径 |
| `mujoco_wiping.yaml` | `wiping` | `dual` | 黑板接触擦拭、力/位混合控制、接触力监控 |
| `mujoco_point_servo.yaml` | `tracking` | `single` | 单点伺服和底层链路调试 |

## 推荐改动位置

- 调整任务目标：优先改 `configs/scenarios/*.yaml` 的 `task` 段。
- 调整低层执行：改 `configs/control/mujoco_tracking_low_level.yaml`。
- 调整机器人装配：改 `configs/robots/assemblies/*.yaml` 或对应 arm 配置。
- 调整场景：改 `configs/scenes/*.yaml`。
- 调整输出：改 `scenario.artifacts`。
- 调整实时窗口：改 `scenario.hooks`。

## 不再维护的入口

以下旧入口已经从当前主线清理：

- 旧 `dual_*` / `single_*` 场景副本。
- 旧 `*_analytic_*` 场景。
- `configs/tasks/*.yaml` 任务配置。
- engine cleaning 专用控制器、任务配置和 surface path 接口。
- 旧 runtime hook 兼容导出层。
