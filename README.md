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

- 通用 `spatial_low_level.yaml` profile 使用保守的末端目标速度限幅：
  - 目标末端速度上限为 `0.015 m/s`。
  - whole-body solver 的 base/tendon rate 限幅开启。
  - backend tendon target 的 rate/displacement/target-lead 保护开启。
  - staged engine navigation 的基座 pose controller 速度裁剪关闭。
  - 接触导纳内部的切向/法向速度裁剪默认关闭。

- 通用 spatial profile 继续使用 measured-feedback `protected` tendon target 模式：

```text
tendon_rate_applied = clip_compatible_rate(tendon_rate_requested)
tendon_target_next = clip(integrated_target,
                          actual_tendon_length ± target_lead)
```

MuJoCo tracking profile 改用 `bending_rate_servo`。它用相邻控制周期的实际 tendon 位移有限差分
估计实现速度，投影到 6 维 bending space 并低通滤波，再由 command/realized bending-rate 误差生成
受限的位置 lead：

```text
b_dot_realized = LPF(C_b^+ (l_actual[k] - l_actual[k-1]) / dt)
b_dot_ref = project_rate_and_displacement_limits(b_dot_command)
e_rate = b_dot_ref - b_dot_realized
delta_b_lead = T_ff b_dot_ref + T_p e_rate + K_i integral(e_rate)
l_target = l_actual + C_b project(delta_b_lead)
```

最终 target 同时受 tendon rate、绝对位移、target lead 和 actuator force guard 约束；投影修改会通过
向量 back-calculation 回写积分状态，避免 actuator 已受限时继续 windup。`actual_anchored`、
`free_integrated` 和 `protected` 仍保留为兼容模式。

底层参数通过 `scenario.low_level_control_path` 引用共享 profile；
`task.tracking_control` 只保留 `approach_samples`、`tracking_mode`、
`trajectory_duration_s` 等上层调度参数。当前提供两套明确的底层版本：

- `configs/control/spatial_low_level.yaml`：通用 protected 版本，启用目标速度、solver 和 backend 保护。
- `configs/control/mujoco_tracking_low_level.yaml`：MuJoCo tracking 版本，保留 `1.0` 位置增益和
  SVD/权重参数，并为 tracking task 显式选择自带联合 guard 的 `bending_rate_servo`。

single/dual 的 MuJoCo tracking、engine tracking、MuJoCo navigation 和 engine cleaning 共 8 个场景
统一引用该文件，但新 servo 只在 `task.type: tracking` 时装配。因此四个 tracking 场景使用新内环，
navigation 和 cleaning 仍保持原 target policy；dual 中 executor 与 observer 使用各自独立的 servo 状态。

共享 MuJoCo profile 保持 solver/backend legacy 限制开关关闭，以隔离本次 tendon 内环变量；
tracking servo 不依赖这两个 legacy 开关，始终执行自身联合 guard：

```yaml
low_level_control:
  enforce_solver_velocity_limits: false
  enforce_backend_tendon_limits: false
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
  -> shared Cartesian servo + per-arm speed limit
  -> isolated executor / observer Jacobian-SVD solves
  -> RobotSystemCommand(base twist + compatible tendon rates)
  -> analytic target integration 或 MuJoCo tendon target policy
     (actual_anchored / protected / free_integrated / bending_rate_servo)
```

各任务的上层控制过程如下：

- `tracking`：轨迹生成器或 waypoint scheduler 输出目标位置和速度前馈；共享底层只计算一次位置闭环。
- `engine tracking`：先由移动基座把直臂末端送到发动机轨迹起点，再冻结基座自由度并复用 time tracking 底层。
- `navigation`：上层增加路径推进、完整中心线 clearance 和可选 CBF-QP；未触发安全终止时复用同一底层运动控制。
- `wiping`：上层管理 approach/contact/retreat，并由选定的 force strategy 修正接触阶段目标；修正后的位置 intent 交给共享底层。
- `engine_cleaning`：上层 task-space cleaning controller 根据 waypoint、接触距离和法向力输出 TCP velocity intent；底层把位置伺服锚定到当前 TCP，避免旧实现的双重 P 控制。
- `engine_navigation`：上层依次执行 base approach、base insertion、局部 executor path、rejoin 和 complete；局部机械臂路径仍通过统一 intent 和共享底层求解。

`CartesianTaskIntent` 的 `position` 模式表示“位置目标 + 速度前馈”，`velocity`
模式表示“直接速度目标”。`TaskStatus` 统一记录任务类型、阶段、活动索引、完成状态和停止原因。

双臂场景中 executor 与 observer 分别求解。executor solve 只包含 base/executor 自由度，observer
solve 只包含 observer tendon 自由度且不允许使用 base；两个结果只在 `RobotSystemCommand` 组装时合并。
因此 observer tracking、避碰和奇异性不会进入 executor 的 SVD 矩阵，也不会冻结或 hard-stop executor。

所有 `dual_*.yaml` 场景都显式使用同一套 observer `collision_avoidance` 上层策略：两臂中心线最近
距离低于 `0.018 m` 时激活，超过 `0.020 m` 时释放，排斥速度按
`1.2 * (0.018 - distance)` 计算且不设置避碰专用速度上限。`0.010 m` minimum 和 `0.008 m`
critical 当前用于配置校验与诊断，不是硬性间距、冻结或急停条件。ROI/offset 跟随仍可通过
`tracking` 模式显式选择，但不用于当前 dual 场景。

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

`single_mujoco_tracking.yaml` 和 `dual_mujoco_tracking.yaml` 当前使用相同的时间参数化轨迹：

```yaml
tracking_control:
  tracking_mode: time
  trajectory_duration_s: 30.0
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

`mujoco_tracking_low_level.yaml` 统一定义 `1.0` 位置增益、SVD 参数、权重和
`bending_rate_servo` 参数。Cartesian target 与 solver/legacy backend 限幅保持关闭；servo 自身的
rate/displacement/lead/force guard 始终开启。
所有 tracking 场景中的每条机械臂各自维护实际速度滤波、rate-error 积分和 anti-windup 状态；
observer 的上层目标仍来自独立 intent。继续引用 `spatial_low_level.yaml` 的其他场景仍使用
`arm_position_gain: 1.5` 和 protected tendon target 模式。

### Engine tracking 的移动基座与反作用隔离

`single_engine_tracking.yaml` 和 `dual_engine_tracking.yaml` 使用两阶段上层控制：

```text
base_approach
  -> MobileBasePoseController
  -> base twist；所有机械臂 tendon rate = 0
  -> 到位条件：位置误差 <= 5 mm，姿态误差 <= 0.035 rad

tracking
  -> TimedTrajectoryTrackingController（30 s）
  -> UnifiedLowLevelController + mujoco_tracking_low_level.yaml
  -> 固定基座装配模型求解 tendon rate
  -> 发给 MuJoCo 的 base_twist_world 恒为 0
```

基座目标在第一个状态处计算：保持当前基座姿态，仅平移基座，使当前 executor tip 对齐第一个发动机
waypoint。基座阶段使用 `base_position_gain: 1.5`、`base_orientation_gain: 2.0`，且与
`engine_navigation` 一样不在 pose controller 内裁剪速度。轨迹阶段同时采用“求解模型无基座自由度”与
“MuJoCo prescribed base 收到零速度”两层约束，避免肌腱动作及其反作用被转换为基座晃动。
基座阶段的 `executor_error_m` 记录为 `NaN`，因此命令行输出的 final/mean/max tracking error 只统计
实际的机械臂轨迹阶段，不会被基座移动阶段稀释。

这里的 `approach_samples: 0` 是有意设置：普通 MuJoCo tracking 用 40 个机械臂 approach 样本连接本地
方形轨迹；engine tracking 已由基座阶段完成远距离接近，若再次从初始原点 prepend arm approach，基座
移动后这些世界坐标样本会失去意义。

dual engine tracking 的 executor 与 single 使用相同的时间轨迹和底层参数；observer 在基座阶段保持
零肌腱速度，进入 tracking 后使用独立的 18 mm 激活避碰上层 intent，并通过相同的固定基座肌腱底层执行。

dual 运行产物额外保存 observer target/actual tendon、requested/applied rate、实际速度、逐 tendon
actuator force、observer mode、最近臂间距离和避碰目标速度，并生成：

- `arm_executor_synchronized_control.png`
- `arm_observer_synchronized_control.png`
- `dual_arm_synchronized_safety.png`

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
4. 如果 tendon target 和 current 差距异常，依次比较 requested、constrained、target FD、realized FD、
   servo 使用的 filtered measured rate 和 raw sensor rate，再检查 target lead、anti-windup、actuator
   force utilization 与 guard feasibility。

`applied_rate_mps` 只是软件内环接受的兼容 reference，不代表物理实现速度；
`tendon_velocity_mps` 是 MuJoCo 周期末的瞬时 sensor。判断执行速度应优先使用跨 state 的
`tendon_realized_rate_fd_mps`。

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
python scripts/run_scenario.py configs/scenarios/dual_mujoco_tracking.yaml
python scripts/run_scenario.py configs/scenarios/single_engine_tracking.yaml
python scripts/run_scenario.py configs/scenarios/dual_engine_tracking.yaml
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
pytest tests/test_tracking_optimization.py tests/test_scenario_migrated_task_features.py tests/test_staged_engine_navigation.py
```

## 交互式 PCC–MuJoCo 末端对比

使用以下专用入口加载双臂 MuJoCo XML，并手动控制两条连续体臂的全部肌腱：

```powershell
python scripts/debug_mujoco_pcc.py configs/scenarios/dual_mujoco_tracking.yaml
```

该入口只复用场景中的 assembly、MuJoCo backend、控制周期和低层肌腱参数，
不会启动 tracking task controller。MuJoCo 窗口中紫色表示 PCC 计算的中心线和
末端，青色表示 MuJoCo 实际中心线和末端 site，红线表示两个末端之间的误差。
控制面板显示两种末端的世界坐标、安装坐标系下的分轴误差、三维误差模长和肌腱
兼容性残差。

PCC 始终使用 MuJoCo 当前实际肌腱位移计算，而不是滑块目标值。调整滑块后应点击
`Run`，等待 target/current 柱状图和末端位置基本稳定，再点击 `Pause` 读取误差。
默认 `compatible` 模式适合检查 PCC 参数；`raw tendon` 允许任意单根肌腱输入，
其非零兼容性残差代表形变已经超出纯 PCC 弯曲子空间。

`Save CSV` 仅在手动点击时写入数据，默认文件位于
`output/diagnostics/mujoco_pcc_manual_<timestamp>.csv`。详细说明见
[docs/debugging_guide.md](docs/debugging_guide.md)。
