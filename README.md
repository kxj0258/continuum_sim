# continuum_sim

`continuum_sim` 是面向空间连续体机械臂的仿真、控制和场景编排项目。当前推荐入口是 `configs/scenarios/*.yaml`：一个场景配置同时声明机器人装配、后端、场景、任务、运行时、hooks 和运行产物。

## 快速运行

```powershell
python scripts/run_scenario.py configs/scenarios/<scenario>.yaml
```

常用场景：

```powershell
# 单臂 MuJoCo 轨迹跟踪
python scripts/run_scenario.py configs/scenarios/single_mujoco_tracking.yaml

# 单臂 / 双臂擦拭
python scripts/run_scenario.py configs/scenarios/single_mujoco_wiping.yaml
python scripts/run_scenario.py configs/scenarios/single_mujoco_wiping_admittance.yaml
python scripts/run_scenario.py configs/scenarios/dual_mujoco_wiping.yaml

# 导航与发动机清洗
python scripts/run_scenario.py configs/scenarios/single_mujoco_navigation.yaml
python scripts/run_scenario.py configs/scenarios/single_engine_cleaning.yaml
python scripts/run_scenario.py configs/scenarios/dual_engine_navigation.yaml
```

Python 代码中也可以直接组合应用层：

```python
from continuum_sim.application import SimulationApplication

app = SimulationApplication.from_yaml(
    "configs/scenarios/single_mujoco_tracking.yaml"
)
result = app.run()
print(len(result.states))
print(app.last_artifacts.run_dir)
```

## 当前主线控制策略

当前 scenario 主线任务，包括 `tracking`、`navigation`、`wiping`、`engine_cleaning` 和 `engine_navigation`，默认采用以下策略：

- 使用解析 PCC 雅可比，不再在主线 whole-body 控制里依赖末端位置数值差分。
- 控制变量使用 bending space：

```text
b = [kx_1, ky_1, kx_2, ky_2, kx_3, ky_3]
q = S_b b
delta_l = C_b b
```

- `eps` 轴向应变默认不作为在线控制自由度。
- 奇异性保护默认使用严格 SVD 方向投影：

```text
xdot_projected = U_valid U_valid^T xdot_des
```

这会丢弃 Jacobian 弱可控方向上的目标速度分量，而不是继续使用 damping/velocity-scale 硬追。

- 默认关闭控制层速度限幅：
  - 目标末端速度限幅关闭。
  - whole-body solver 的 base/tendon rate 限幅关闭。
  - backend tendon target 的 rate/displacement/target-lead 保护关闭。
  - staged engine navigation 的基座 pose controller 速度裁剪关闭。
  - 接触导纳内部的切向/法向速度裁剪默认关闭。

- MuJoCo system backend 默认使用 `actual_anchored` tendon target 模式：

```text
tendon_target_next = actual_tendon_length + dt * compatible_tendon_rate
```

该模式避免 tendon target 自由积分后长期漂离实际 tendon length。

如果需要恢复旧保护，可在场景 YAML 中显式打开：

```yaml
task:
  tracking_control:
    singularity_strategy: damping_scale
    enforce_target_speed_limit: true
    enforce_solver_velocity_limits: true
    enforce_backend_tendon_limits: true
    max_target_speed_mps: 0.015
```

接触导纳速度裁剪可通过：

```yaml
task:
  admittance:
    enforce_velocity_limits: true
```

## 时间轨迹跟踪

`single_mujoco_tracking.yaml` 当前使用时间参数化轨迹：

```yaml
tracking_control:
  tracking_mode: time
  trajectory_duration_s: 80.0
```

控制器每个控制周期按仿真时间采样：

```text
p_d(t), p_dot_d(t)
v_des = p_dot_d(t) + Kp * (p_d(t) - p_tip)
```

这与旧的逐 waypoint 容差推进不同。旧模式仍可通过：

```yaml
tracking_control:
  tracking_mode: waypoint
```

恢复。

## 项目结构

```text
configs/scenarios/         推荐运行入口
configs/robots/            单臂、双臂、移动基座和装配配置
configs/scenes/            结构化场景，例如发动机、喷管、擦拭板
configs/tasks/             旧任务配置和兼容运行时配置
assets/mujoco/             MuJoCo XML 基线模型
scripts/                   运行、检查和调试脚本
src/continuum_sim/application
                            场景解析和 SimulationApplication 组合根
src/continuum_sim/control   跟踪、导航、擦拭、发动机导航控制器
src/continuum_sim/kinematics
                            PCC FK、解析雅可比、whole-body Jacobian
src/continuum_sim/backends  analytic / MuJoCo 系统后端
src/continuum_sim/runtime   仿真循环和 hooks
src/continuum_sim/io        运行产物、图表、metadata、视频导出
docs/                       架构、配置、调试和迁移说明
```

## 坐标与安装高度检查

检查解析直臂末端和 MuJoCo reset 后 tip 是否一致：

```powershell
python scripts/compare_analytic_mujoco_tip.py
```

也可以指定场景：

```powershell
python scripts/compare_analytic_mujoco_tip.py configs/scenarios/dual_mujoco_wiping.yaml
```

输出中的：

```text
analytic_straight_tip_world_m
mujoco_reset_tip_world_m
difference_mujoco_minus_analytic_m
```

应接近一致。当前单臂装配的 `mount_pose.position_m.z = 0.02` 已经包含连续体安装高度 20 mm，不应再重复加到 PCC FK 内部。

## 运行产物

默认运行产物写入：

```text
output/runs/<scenario>_<timestamp>/
  result.npz
  metadata.json
  configs/
  model/
  plots/
  videos/
```

`output/runs/` 是本地运行产物，不应提交。`output/generated/` 中的 XML 是场景运行前生成或保留的 MuJoCo 模型文件。

## 调试建议

推荐从场景配置开始排查：

1. 检查 `configs/scenarios/<name>.yaml` 的 `backend`、`task`、`runtime` 和 `hooks`。
2. 打开 `recorder`、`tendon_debug` 和 `show_live_diagnostics_panel`，观察误差、奇异值、投影残差、tendon target/current。
3. 查看 `output/runs/<scenario>_<timestamp>/metadata.json`、`result.npz` 和 `plots/`。
4. 如果 tendon target 和 current 差距异常，优先看 tendon monitor 中的 `target actual_anchored`、actuator force 和 target-current error。

更多细节见：

- [docs/configuration_reference.md](docs/configuration_reference.md)
- [docs/debugging_guide.md](docs/debugging_guide.md)
- [docs/coordinate_conventions.md](docs/coordinate_conventions.md)
- [docs/mainline_migration_plan.md](docs/mainline_migration_plan.md)

## 手动验证建议

本项目默认不自动运行测试或仿真。修改后可按需手动执行：

```powershell
python scripts/compare_analytic_mujoco_tip.py
python scripts/run_scenario.py configs/scenarios/single_mujoco_tracking.yaml
python scripts/run_scenario.py configs/scenarios/single_mujoco_navigation.yaml
python scripts/run_scenario.py configs/scenarios/single_mujoco_wiping.yaml
python scripts/run_scenario.py configs/scenarios/dual_mujoco_wiping.yaml
python scripts/run_scenario.py configs/scenarios/single_engine_cleaning.yaml
python scripts/run_scenario.py configs/scenarios/dual_engine_navigation.yaml
```

如需运行单元测试，可手动执行：

```powershell
pytest
```
