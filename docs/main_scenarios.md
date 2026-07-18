# 主任务命令

当前项目只维护 5 个主任务入口。每个任务使用一个 YAML，通过
`scenario.arm_mode` 在双臂和单臂之间切换。

```yaml
scenario:
  arm_mode: dual   # 可选 dual 或 single
```

`dual` 模式使用主臂 executor 和从臂 observer。`single` 模式使用仅包含
executor 的装配，并在生成 MuJoCo XML 时只保留主臂。除从臂是否存在外，
轨迹、控制参数、runtime、hooks 和 artifacts 都来自同一个 YAML。

## 推荐命令

```powershell
python scripts/run_scenario.py configs/scenarios/mujoco_tracking.yaml
python scripts/run_scenario.py configs/scenarios/mujoco_navigation.yaml
python scripts/run_scenario.py configs/scenarios/engine_navigation.yaml
python scripts/run_scenario.py configs/scenarios/mujoco_wiping.yaml
python scripts/run_scenario.py configs/scenarios/mujoco_point_servo.yaml
```

## 默认模式

| 场景 | 默认模式 | 说明 |
| --- | --- | --- |
| `mujoco_tracking.yaml` | `dual` | 固定基座方形轨迹跟踪 |
| `mujoco_navigation.yaml` | `dual` | 火箭喷管入口结构化场景导航 |
| `engine_navigation.yaml` | `dual` | 发动机入口、插入和局部路径导航 |
| `mujoco_wiping.yaml` | `dual` | 黑板擦拭接触任务 |
| `mujoco_point_servo.yaml` | `single` | 单臂点伺服；切到 dual 时启用从臂避碰 |

## 清理后的边界

旧的 `dual_*`、`single_*`、`*_analytic_*` 场景文件、`configs/tasks/*.yaml`
旧任务配置、旧 MuJoCo runtime 和旧指标脚本已经移除。主线只通过
`SimulationApplication` 和 `scripts/run_scenario.py` 运行。
