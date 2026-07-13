# 统一控制架构（方案 B）设计

## 目标

在保持 `python scripts/run_scenario.py <scenario.yaml>` 入口不变的前提下，将 tracking、navigation、wiping、engine cleaning 和 engine navigation 统一为：

1. 独立的上层任务控制器负责路径、阶段、接触和任务终止语义。
2. 上层统一输出强类型 `TaskStep(TaskIntent, TaskStatus)`。
3. 共享底层控制器负责笛卡尔闭环、observer 协同、Jacobian、SVD/正则化、whole-body 求解及 tendon-rate 命令。
4. MuJoCo backend 继续负责 actual-anchored tendon target 积分和 actuator target 写入。

## 控制边界

```text
task controller
  -> TaskStep(TaskIntent, TaskStatus)
  -> UnifiedLowLevelController
  -> CoordinatedTrackingController
  -> WholeBodyController
  -> RobotSystemCommand(base twist + tendon rates)
  -> backend tendon integration / MuJoCo actuator targets
```

`CartesianTaskIntent` 支持两种互斥语义：

- `position`：目标位置 + 速度前馈，位置误差只在共享底层乘一次增益。
- `velocity`：直接速度意图；底层把内部位置锚定到当前 TCP，避免 engine cleaning 出现双重 P 控制。

`TaskStatus` 统一承载 phase、active index、complete 和 stop reason。任务特有诊断仍可放入 metadata，但不再用 metadata 传递核心控制语义。

## 参数归属

- `configs/control/spatial_low_level.yaml`：共享笛卡尔增益、速度限制、whole-body 权重、SVD/阻尼、求解器及 backend tendon-limit 开关。
- 场景 YAML：轨迹形状、任务时长、waypoint 推进、clearance、接触目标、力控策略、任务阶段、场景和 backend。
- robot/assembly YAML：机构尺寸、安装位姿、base/tendon 物理限制。
- MuJoCo YAML/MJCF：仿真步长、actuator、传感器和渲染。

场景中的 `tracking_control` 只保留任务调度参数；为了兼容旧配置，仍允许显式覆盖共享底层字段，但新配置不应这样做。

## Admittance 配置决策

`contact_triggered_admittance` 有明确意义：它在检测到接触后，用法向力误差调整法向位移，同时保持切向轨迹推进，适合接触刚度未知或希望限制冲击的擦拭实验。因此保留其实现和参数。

但 `single_mujoco_wiping_admittance.yaml` 与普通擦拭场景仅策略和参数不同，复制了 backend、路径、runtime 和 hooks，容易漂移。删除该重复场景；在 `single_mujoco_wiping.yaml` 中保留完整 `admittance` 参数块和可选策略说明，切换 `wiping_control_type` 与 `force_strategy.type` 即可启用。

## 兼容与风险

- 现有 controller 对 `SimulationLoop` 仍输出 `RobotSystemCommand`，所以 runtime/backend 接口不变。
- Engine cleaning 改为 velocity intent 后，不再叠加通用位置 P；行为会比旧实现温和，原有增益需要重新手工标定。
- 共享参数会减少场景逐项调参，但 analytic 与 MuJoCo 模型差异仍可能要求以后增加少量经过审查的 profile，而不是在每个任务中复制底层字段。
- 本轮按用户约束不自动运行任何验证。
