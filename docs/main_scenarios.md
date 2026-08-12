# 场景说明

项目在 `configs/scenarios/` 中提供 6 个可运行场景。自动任务由 `scripts/run_scenario.py` 启动，手动控制由 `scripts/run_manual_control.py` 启动。

## 场景列表

| 场景文件 | 任务类型 | 默认模式 | 基座 | 主要能力 |
| --- | --- | --- | --- | --- |
| `mujoco_manual_control.yaml` | `idle` | `dual` | 移动 | 双臂局部曲率、肌腱和基座六自由度手动控制、三窗口显示 |
| `mujoco_tracking.yaml` | `tracking` | `dual` | 固定 | 方形轨迹跟踪、observer 协同、实时诊断 |
| `mujoco_point_servo.yaml` | `tracking` | `single` | 固定 | 单点伺服和低层控制调试 |
| `mujoco_navigation.yaml` | `navigation` | `dual` | 移动 | 基座阶段化接近、结构化场景避障、observer 相机 |
| `engine_navigation.yaml` | `engine_navigation` | `dual` | 移动 | 发动机入口定位、轴向插入和内部探索路径 |
| `mujoco_wiping.yaml` | `wiping` | `dual` | 固定 | 球形工具擦拭、六维力反馈、力/位混合控制 |

## 运行命令

```powershell
python scripts/run_scenario.py configs/scenarios/mujoco_tracking.yaml
python scripts/run_scenario.py configs/scenarios/mujoco_navigation.yaml
python scripts/run_scenario.py configs/scenarios/engine_navigation.yaml
python scripts/run_scenario.py configs/scenarios/mujoco_wiping.yaml
python scripts/run_scenario.py configs/scenarios/mujoco_point_servo.yaml
python scripts/run_manual_control.py
```

批量运行非空闲任务：

```powershell
python scripts/run_all_scenarios.py
```

## 单臂和双臂

场景通过 `scenario.arm_mode` 选择装配：

```yaml
scenario:
  arm_mode: dual
```

- `dual`：启用 `executor` 和 `observer`。
- `single`：启用 `executor`。
- `manual_control`、`navigation` 和 `engine_navigation` 选择移动基座装配。
- 轨迹跟踪、点伺服和擦拭任务选择固定基座装配。

## 配置职责

- 任务目标与任务模式：`configs/scenarios/*.yaml` 的 `task`。
- 控制器增益、优先级和执行层：`configs/control/mujoco_tracking_low_level.yaml`。
- 机械臂和装配：`configs/robots/`。
- 环境、擦拭板和发动机：`configs/scenes/`。
- 末端工具和相机：`configs/tools/`。
- 实时窗口：场景的 `hooks`。
- 运行数据和图像：场景的 `artifacts`。

## 时间设置

所有 MuJoCo 场景使用：

```yaml
runtime:
  controller_dt_s: 0.02
  n_substeps: 20
```

配合 `0.001 s` MuJoCo timestep，每个控制周期推进 `0.02 s` 物理时间。场景构建时会验证这个等式。
