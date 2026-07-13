# continuum_sim

`continuum_sim` 是面向空间连续体机械臂的仿真、控制和场景编排项目。当前推荐入口是 `configs/scenarios/*.yaml`：一个场景配置同时声明机器人装配、后端、场景、任务、运行时、hooks 和运行产物。

## 快速运行

```powershell
python scripts/run_scenario.py configs/scenarios/<scenario>.yaml
```

任务场景命令：

```powershell
# Tracking：analytic / MuJoCo / engine scene
python scripts/run_scenario.py configs/scenarios/single_analytic_tracking.yaml
python scripts/run_scenario.py configs/scenarios/dual_analytic_tracking.yaml
python scripts/run_scenario.py configs/scenarios/single_mujoco_tracking.yaml
python scripts/run_scenario.py configs/scenarios/dual_mujoco_tracking.yaml
python scripts/run_scenario.py configs/scenarios/single_engine_tracking.yaml
python scripts/run_scenario.py configs/scenarios/dual_engine_tracking.yaml

# Navigation：analytic / MuJoCo
python scripts/run_scenario.py configs/scenarios/single_analytic_navigation.yaml
python scripts/run_scenario.py configs/scenarios/dual_analytic_navigation.yaml
python scripts/run_scenario.py configs/scenarios/single_mujoco_navigation.yaml
python scripts/run_scenario.py configs/scenarios/dual_mujoco_navigation.yaml

# Wiping：analytic / MuJoCo
python scripts/run_scenario.py configs/scenarios/single_analytic_wiping.yaml
python scripts/run_scenario.py configs/scenarios/dual_analytic_wiping.yaml
python scripts/run_scenario.py configs/scenarios/single_mujoco_wiping.yaml
python scripts/run_scenario.py configs/scenarios/dual_mujoco_wiping.yaml

# Engine cleaning
python scripts/run_scenario.py configs/scenarios/single_engine_cleaning.yaml
python scripts/run_scenario.py configs/scenarios/dual_engine_cleaning.yaml

# Engine navigation
python scripts/run_scenario.py configs/scenarios/single_engine_navigation.yaml
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

- 共享底层 profile 使用保守的末端目标速度限幅：
  - 目标末端速度上限为 `0.015 m/s`。
  - whole-body solver 的 base/tendon rate 限幅关闭。
  - backend tendon target 的 rate/displacement/target-lead 保护关闭。
  - staged engine navigation 的基座 pose controller 速度裁剪关闭。
  - 接触导纳内部的切向/法向速度裁剪默认关闭。

- MuJoCo system backend 默认使用 `actual_anchored` tendon target 模式：

```text
tendon_target_next = actual_tendon_length + dt * compatible_tendon_rate
```

该模式避免 tendon target 自由积分后长期漂离实际 tendon length。

底层参数统一定义在 `configs/control/spatial_low_level.yaml`。场景通过
`scenario.low_level_control_path` 引用它；`task.tracking_control` 只应保留
`approach_samples`、`tracking_mode`、`trajectory_duration_s` 等上层调度参数。
旧场景仍允许显式覆盖底层字段，但新场景不建议这样做。

如果需要启用 solver/backend 的物理限幅，可在共享 profile 中打开：

```yaml
low_level_control:
  enforce_solver_velocity_limits: true
  enforce_backend_tendon_limits: true
```

接触导纳速度裁剪可通过：

```yaml
task:
  admittance:
    enforce_velocity_limits: true
```

## 统一上层 / 底层控制架构

所有 scenario 主线任务保持同一个运行和驱动接口：

```text
独立上层任务控制器
  -> TaskStep(SystemTaskIntent, TaskStatus)
  -> UnifiedLowLevelController
  -> Cartesian servo + observer coordination
  -> WholeBodyController(Jacobian / SVD / regularization)
  -> RobotSystemCommand(base twist + compatible tendon rates)
  -> analytic backend 或 MuJoCo actual-anchored tendon targets
```

各任务的上层控制过程如下：

- `tracking`：轨迹生成器或 waypoint scheduler 输出目标位置和速度前馈；共享底层只计算一次位置闭环。
- `navigation`：上层增加路径推进、完整中心线 clearance 和可选 CBF-QP；未触发安全终止时复用同一底层运动控制。
- `wiping`：上层管理 approach/contact/retreat，并由选定的 force strategy 修正接触阶段目标；修正后的位置 intent 交给共享底层。
- `engine_cleaning`：上层 task-space cleaning controller 根据 waypoint、接触距离和法向力输出 TCP velocity intent；底层把位置伺服锚定到当前 TCP，避免旧实现的双重 P 控制。
- `engine_navigation`：上层依次执行 base approach、base insertion、局部 executor path、rejoin 和 complete；局部机械臂路径仍通过统一 intent 和共享底层求解。

`CartesianTaskIntent` 的 `position` 模式表示“位置目标 + 速度前馈”，`velocity`
模式表示“直接速度目标”。`TaskStatus` 统一记录任务类型、阶段、活动索引、完成状态和停止原因。

## Wiping 导纳可选策略

导纳控制仍然有意义：接触建立后，它根据法向力误差调节法向位移，适合接触刚度不确定或需要减小冲击的实验。为避免两份场景漂移，已删除独立的
`single_mujoco_wiping_admittance.yaml`；完整参数保留在
`single_mujoco_wiping.yaml` 的 `task.admittance` 中。启用时只需把同一文件中的两项都改为：

```yaml
task:
  wiping_control_type: contact_triggered_admittance
  force_strategy:
    type: contact_triggered_admittance
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
python scripts/run_scenario.py configs/scenarios/dual_analytic_navigation.yaml
python scripts/run_scenario.py configs/scenarios/dual_analytic_wiping.yaml
python scripts/run_scenario.py configs/scenarios/single_mujoco_tracking.yaml
python scripts/run_scenario.py configs/scenarios/single_mujoco_navigation.yaml
python scripts/run_scenario.py configs/scenarios/single_mujoco_wiping.yaml
python scripts/run_scenario.py configs/scenarios/dual_mujoco_wiping.yaml
python scripts/run_scenario.py configs/scenarios/single_engine_cleaning.yaml
python scripts/run_scenario.py configs/scenarios/dual_engine_cleaning.yaml
python scripts/run_scenario.py configs/scenarios/single_engine_navigation.yaml
python scripts/run_scenario.py configs/scenarios/dual_engine_navigation.yaml
```

如需运行单元测试，可手动执行：

```powershell
pytest
```
