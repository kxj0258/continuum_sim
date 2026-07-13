# Findings & Decisions

## Requirements
- 解释每个已运行控制任务的控制过程。
- 识别上层和底层控制器，以及各自输入输出。
- 说明相关参数定义与作用。
- 设计“统一底层控制 + 独立上层控制器”的优化方向。
- 读取现有运行结果，分析总体误差和 single_mujoco_tracking 局部 z 稳态偏差。
- 不运行任何测试、验证、lint、format、build、install 或仿真命令。
- 保持现有 `scripts/run_scenario.py <scenario.yaml>` 命令入口稳定。
- 统一底层控制参数、控制求解、tendon rate/position 驱动与 MuJoCo backend 接口。
- tracking/navigation/wiping/cleaning/engine navigation 保留独立上层任务策略，但输出统一命令协议。

## Research Findings
- 用户提供的 tracking 误差约 3–8 mm；wiping、navigation、cleaning 误差更大且全部报告 stopped_early。
- `single_mujoco_navigation` 仅 5 states，`single_engine_cleaning` 仅 202 states，优先怀疑终止/阶段切换条件，而非单纯稳态精度。
- HANDOFF 说明控制目标倾向使用 `actual_anchored` tendon state、解析雅可比、严格 SVD 投影；wiping 已改为时间轨迹并运行时生成 approach。
- 主要模块边界包括 `application`、`tasks`、`control`、`runtime`、`backends`、`kinematics`、`system`、`model` 和 `io`；控制相关实现分散在 `control` 与若干 `runtime/mujoco_*_runtime.py` 中。
- 六个已读取运行目录均生成 `videos/simulation.gif`，metadata 显示 `video_mode: live_mujoco`、`video_status: ok`、`errors: []`，因此本批次录制路径已正常工作。
- `single_mujoco_tracking` 与 `dual_mujoco_tracking` 的最终误差都约 3.25 mm，但二者都记录 `maximum_whole_body_condition_number: Infinity`；需进一步检查 rank、最小奇异值和残差。
- `single_mujoco_navigation` 仅运行 4 个 command，不能把 68.5 mm 最终误差解释成稳态控制精度；必须先找到 done/停止原因。
- metadata 没有 `summary.json`，核心数值序列位于 `result.npz`。
- `single_mujoco_tracking` 末 200 个控制周期的世界坐标误差均值约 `[+0.088, +2.378, -2.551] mm`，各轴标准差均小于 `0.1 mm`，属于稳定偏置；末时 tendon target error norm 仅约 `0.012 mm`。
- `single_mujoco_tracking` 使用 time mode，80.02 s 时 `tracking_complete=True`；其 `stopped_early` 是完成 hook 在 `max_steps` 前正常停止。
- 两个 wiping run 分别在 35.02 s 与 20.02 s 结束，均 `tracking_complete=True`；这与配置的 `trajectory_duration_s` 一致。
- `dual_mujoco_tracking` 走 tolerance 推进，结束时 waypoint 48、`tracking_complete=False`，停止原因尚需 hook 证据。
- wiping 的 `task_phase` 在 NPZ 全程为 `approach`，与 waypoint index 随时间递增并最终完成不一致，倾向于 metadata 标签未更新而非真实阶段未切换。
- `single_mujoco_navigation` 配置 clearance 下限 10 mm，NPZ 记录最小值约 19.46 mm；仅凭这一字段不能支持 clearance violation 根因。
- `SimulationLoop.stopped_early` 仅表示任一 hook 在 `max_steps` 前要求停止；来源可能是 controller completion，也可能是 viewer 关闭，不能直接解释成异常。
- `SimulationApplication.from_config()` 是组合根：按 task type 创建 `NavigationController`、`WipingController`、`EngineCleaningSystemController`、time/waypoint tracking controller，并统一接入同一 `SimulationLoop` 与 `RobotSystemCommand` 协议。
- `StateRecorderHook` 的 `task_phase` 实际读取 command metadata 的 `wiping_phase`，所以非 wiping 为空；wiping 全程 approach 说明 controller metadata 的 phase 值需要进一步追踪。
- 已证实：`TimedTrajectoryTrackingController.active_index` 依赖 `_elapsed_or_zero()`，后者固定返回 0；`WipingController` 因此始终使用 index 0 和 phase `approach`，time-mode wiping 的 contact force strategy 实际不会进入 contact 修正分支。
- 已证实：`WaypointTrackingController.compute_command(..., advance=...)` 未使用 `advance` 参数，且 scheduler update 未检查 `self.advance_enabled`；wiping/cleaning 试图暂停通用 scheduler 的配置目前无效。
- `NavigationController.done = tracking.done OR (terminate_on_clearance_violation AND clearance_violated)`；0.08 s 停止必然是某一步内部 clearance 小于 10 mm 或 tracking scheduler 异常完成。NPZ 只记录含 executor target 的 command，需检查逐步最小值/中心线查询差异。
- `EngineCleaningSystemController.done = task controller is_done OR safety_stop`，并将 task controller 的 active index/velocity 写入通用 tracker；但通用 tracker仍可能自行推进，存在双状态机竞争。
- tracking 的任务速度为 `executor_position_gain * (target - actual) + trajectory/feedforward velocity`，然后由 `WholeBodyController` 解加权线性速度任务。
- `svd_projection` 会把低于 `minimum_singular_value` 的 weighted task-space SVD 方向从目标中直接删除；single tracking 初始 rank=2，轨迹中最小奇异值曾低至 `2.8e-9`，而阈值为 `1e-5`，弱方向误差会在这些时段积累。
- `WholeBodyController` 还对 base/tendon velocity 加 Tikhonov 型正则；这会降低响应但在静态目标上原则上不应单独造成永久偏差，除非方向被投影、命令被提前终止或模型/反馈映射不一致。
- `TimedTrajectoryTrackingController` 在 elapsed 达到 duration 的同一周期立即将 command 置零并 done，不提供终点 settling 阶段；single tracking 因而把运动中的约 3 mm 跟随误差直接冻结为最终误差。
- Coordinated tracking 的 Jacobian 根据解析 PCC 模型和 tendon displacement 估计 bending state，而反馈位置来自 MuJoCo 实际 tip；解析模型与离散柔性 MuJoCo、重力/预张力之间的差异会表现为任务空间模型误差，但闭环若有足够 settling 时间应可继续纠正其可控分量。
- MuJoCo backend 将 bending-compatible tendon rate 通过 `CompatibleTendonRateIntegrator` 转成 tendon position target；当 command metadata 禁用 backend limits 时采用 `actual_anchored`，每步锚定 MuJoCo 实际 tendon length，而不是盲目累计旧目标。
- single assembly 的 fixed base、executor mount quaternion 都为单位旋转，因此本运行中 arm mount 局部 z 与世界 z 一致；末段 target-tip z 均值约 -2.55 mm 表示实际 tip 比目标高约 2.55 mm。
- MuJoCo 使用 tendon-position actuator，`kp=40000`、force limit ±30 N、gravity disabled；低层目标误差微小与该高增益位置执行链一致。
- cleaning 结束时 active waypoint=0、tracking_complete=False、误差约 0.46 m，故不可能是路径完成；结合唯一 done 分支可确认是 `max_contact_force_exceeded` safety stop。
- cleaning 的 stop reason、专用 contact distance/force 使用 `engine_cleaning_*` metadata 名，而 StateRecorder 只记录通用 wiping/contact 字段且要求存在 `executor_target_world`；当前 NPZ 因而丢失终止周期的关键诊断。
- cleaning 第一个目标 `[0.099155, 0.515382, 0] m` 与初始 tip `[0, 0.01, 0.14] m` 相距约 0.534 m，明显超出 0.14 m 级单臂局部工作范围；需追踪 engine scene/path 坐标变换。
- 已证实 cleaning patch 配置为 world frame，中心 `y=0.530382 m`；path builder 直接使用 region.position_m，不施加 engine/base/arm 变换。fixed-base 单臂任务因此先天不可达，属于场景与任务可达性配置错误。
- navigation 场景未定义 tracking_control，继承默认 Kp=4、无 target speed limit、无 solver velocity limits、无 backend tendon limits；NPZ tendon rate 峰值约 1.36 m/s、tendon target error norm 峰值约 50.8 mm，导致极激进的目标跳变。
- navigation 的 violating command 被替换成 `RobotSystemCommand.zeros()`，从而丢失 `executor_target_world`；StateRecorder 把 min_clearance 的记录嵌套在 target 存在条件内，所以恰好漏掉最后的 clearance violation 样本。
- dual tracking 结束时 tracking_complete=False；除 completion 外唯一会请求停止的已启用 hook 是 `MujocoViewerHook` 的 `not viewer.is_running()`，因此该 run 是 viewer 被关闭而停止。
- single tracking 最后 4 s 目标速度约 2.975 mm/s，tip 速度量级相近但保持约 `[+0.03,+2.36,-2.55] mm` target-minus-tip 偏置；目标直到结束都未静止。
- single tracking 的弱奇异阈值触发集中在前 3.42 s（171/4001 样本）；结束时 min singular≈0.00717、condition≈5.83，最终偏差不是“终点仍奇异”。
- 最后一条非零 tendon-rate command 范数约 1.10 mm/s，duration 到期后立即变 0；终点没有静态闭环收敛窗口。
- StateRecorder 的 `tracking_error_m` 来自 controller 计算 command 前的 state，而 `target_actual_position_m` 记录 command 执行后的 next_state；两个序列存在一个控制周期的对齐差异，精确分量分析应显式对齐。
- `EngineCleaningController` 已经用 approach/contact gains 生成闭环 TCP velocity，但 adapter 又把同一 waypoint position 交给默认 Kp=4 的 generic tracker；后者再次叠加位置反馈，绕过 cleaning 的 0.03 m/s TCP 限速，形成双重 P 控制。
- MuJoCo backend 未填充 `ArmSystemState.centerline_world`；NavigationController 的 clearance termination 因此退化成只查询 tip，而不是完整连续体中心线。
- dual tracking 虽配置 `max_target_speed_mps: 0.030`，但缺少 `enforce_target_speed_limit: true`，该上限实际不生效。

## Proposed Architecture
- 上层任务控制器只管理 trajectory/waypoint/phase/force mission，输出强类型 `TaskIntent` 与 `TaskStatus`。
- 统一 `CartesianServo` 接受互斥的 position-setpoint 或 velocity-intent，避免 cleaning 双重 P。
- 统一 `WholeBodyMotionController` 负责 Jacobian、任务权重、SVD、正则和 compatible tendon rate。
- 统一 `SafetySupervisor` 处理完整 centerline clearance、contact force 与命令限幅，不由各任务重复实现。
- `ActuationPipeline` 统一 rate limit、actual-anchored target integration 和 backend actuator target。
- `SimulationStopReason` 区分 completed、duration_elapsed、clearance_violation、force_limit、viewer_closed、max_steps。
- recorder 独立记录 state、intent、command、status，采用同一周期语义并保存终止样本。

## Architecture B Implementation
- 上层/底层边界已实现为 `TaskStep(SystemTaskIntent, TaskStatus)`。
- `UnifiedLowLevelController` 统一进入 coordinated tracking、whole-body solver 和 compatible tendon-rate 链路。
- Engine cleaning 使用 velocity intent，避免 task-space 速度闭环与通用位置 P 重复叠加。
- `configs/control/spatial_low_level.yaml` 成为主线场景共享低层参数源；场景级 tracking_control 只保留调度字段。
- time trajectory active index 和 waypoint advance 开关已按实际控制语义修正。
- MuJoCo arm state 已填充 centerline，navigation clearance 不再退化为 tip-only。
- contact-triggered admittance 作为 `single_mujoco_wiping.yaml` 的可选策略保留，重复场景文件已删除。

## Technical Decisions
| Decision | Rationale |
|----------|-----------|
| 先追踪运行产物中的停止原因和误差分量，再提出优化 | 避免把表现症状误判成控制器根因 |
| 架构建议聚焦控制命令协议和反馈快照边界 | 统一底层不应抹平 navigation/wiping/cleaning 的任务策略差异 |
| 采用渐进式 adapter 迁移 | 先修复诊断和命令协议，再逐个迁移 task，可降低一次性改写控制语义的风险 |
| 推荐以强类型 TaskIntent 为上层统一输出 | 能同时承载位置、速度前馈、接触力、observer/base 目标，又避免 metadata 隐式协议 |

## Issues Encountered
| Issue | Resolution |
|-------|------------|
| 截图只能显示几何现象，不能直接量化局部 z 偏差 | 结合 metadata、state/target 序列和参考系变换代码分析 |
| 初次枚举引用了不存在的 `configs/controllers` | 改按实际目录和具体场景 YAML 读取 |
| `stopped_early` 的语义不明确 | 读取 `SimulationLoop` 与 controller 的 `done` 条件后再判断是否异常 |

## Resources
- `HANDOFF.md`
- `src/continuum_sim/application/`
- `src/continuum_sim/control/`
- `src/continuum_sim/backends/`
- `src/continuum_sim/runtime/`
- `configs/scenarios/`
- `output/runs/`

## Visual/Browser Findings
- 截图为 `single_spatial_executor` 的 MuJoCo viewer，机械臂已弯曲到目标附近；画面中的 RGB 轴受视角透视影响，无法仅凭截图断定偏差属于世界 z 或执行器局部 z。
- 末端看起来形成稳定位置偏置而非明显发散，需重点检查 SVD 可控子空间、目标/测量参考点与 tendon 饱和状态。
