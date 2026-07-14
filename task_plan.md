# Task Plan: 控制流程、架构与跟踪误差分析

## Goal
基于源码、YAML 与用户现有运行产物，说明各控制任务的上下层控制流程、输入输出和参数，并给出统一底层控制架构与误差根因建议。

## Current Phase
Phase 8 (in progress)

## Phases

### Phase 1: 范围与证据收集
- [x] 记录用户约束与运行结果
- [x] 读取项目结构、控制入口、配置和运行产物
- [x] 建立任务到控制器的顶层数据流
- **Status:** complete

### Phase 2: 控制流程分析
- [x] 分析 tracking/navigation/wiping/cleaning 的上下层控制器
- [x] 整理各层输入、输出和配置参数
- [x] 整理统一底层 tendon rate → actual-anchored target → MuJoCo feedback 链路
- **Status:** complete

### Phase 3: 误差根因分析
- [x] 对比各运行的停止原因和误差分量
- [x] 追踪 single_mujoco_tracking 局部 z 稳态偏差
- [x] 区分已证实根因、强假设和待验证项
- **Status:** complete

### Phase 4: 架构优化建议
- [x] 绘制现有依赖关系
- [x] 设计统一底层控制 + 独立上层控制器边界
- [x] 给出低风险渐进迁移顺序
- **Status:** complete

### Phase 5: 交付
- [x] 汇总结论、风险及建议手动验证命令
- [x] 明确未运行任何测试或验证
- **Status:** complete

### Phase 6: 统一控制架构设计
- [x] 提交现有工作区作为重构前基线
- [x] 恢复现有控制链与已知问题上下文
- [x] 对比 2–3 种统一架构方案
- [x] 给出推荐架构、任务命令、任务流程与参数归属
- [x] 用户确认采用方案 B
- **Status:** complete

### Phase 7: 方案 B 实现
- [x] 新增强类型 TaskIntent / TaskStatus / TaskStep
- [x] 新增 UnifiedLowLevelController 并迁移 waypoint/time/cleaning 控制链
- [x] 统一 low-level YAML profile 并接入场景加载
- [x] 修正 waypoint advance 与 timed active index
- [x] 为 MuJoCo 状态补充 centerline clearance 数据
- [x] 新增 dual analytic navigation/wiping、dual cleaning、single engine navigation
- [x] 将 admittance 合并为 single MuJoCo wiping 的可选策略
- [x] 完成文档与静态交接整理
- **Status:** complete

### Phase 8: 双臂隔离控制与同步诊断
- [x] 统一 single/dual tracking 的 executor 时间轨迹配置与控制器类型
- [x] 将 executor 与 observer 的 Jacobian/SVD 求解完全隔离
- [x] 将 observer 改为 collision-avoidance-only 上层策略
- [x] 对两臂统一施加 Cartesian 速度和 tendon rate/lead 限制
- [x] 禁止 observer 冻结或 hard-stop executor
- [x] 持久化 observer 目标、误差、模式、臂间距离、限幅与 tendon target
- [x] 增加双臂同步时序图并更新架构/配置文档
- [x] 仅记录建议手动验证命令，不执行验证
- **Status:** complete

## Key Questions
1. 各场景实际实例化了哪个任务控制器、跟踪器和执行器控制器？
2. 上层目标如何变成 tendon length/rate 或 base command？
3. stopped_early 的实际原因是什么，各误差统计分别度量什么？
4. single_mujoco_tracking 的稳定偏差来自参考系、不可控方向、限幅、模型差异还是终止条件？
5. 如何在不破坏场景策略的前提下统一底层控制接口？

## Decisions Made
| Decision | Rationale |
|----------|-----------|
| 仅做静态读取和现有产物分析 | 用户明确禁止自动运行任何测试、验证或仿真命令 |
| 将截图现象视为待证实观察 | 单张透视图无法独立证明局部 z 误差或其根因 |
| 先提交再开展新架构设计 | 用户明确要求先建立可回退的 Git 基线 |
| executor 与 observer 分臂求解 | 保证 dual executor 的控制数学路径不受 observer 任务堆叠影响 |
| observer 默认 collision-only | 本任务不再跟踪 executor offset/ROI，只在安全影响区内产生回避速度 |
| 两臂共用同一限幅策略 | 避免 observer 未限幅命令导致 MuJoCo actuator 长期饱和 |

## Errors Encountered
| Error | Attempt | Resolution |
|-------|---------|------------|
| HANDOFF.md 首次读取乱码 | 1 | 改用显式 UTF-8 读取 |
| 文件枚举包含不存在的 `configs/controllers` | 1 | 保留已获得的模块清单，后续按实际目录读取 |
| 一次跨多个 controller 的补丁上下文不唯一 | 1 | 拆成按 class 定位的小补丁分别应用 |
| observer dataclass 插入点截断 tracking 校验尾部 | 1 | 静态复读后将 damping/velocity-scale 校验移回 tracking config |
| debugging guide 预期英文标题不存在 | 1 | 枚举实际中文标题后插入“运行产物排查点”章节 |

## Notes
- 不运行 pytest、脚本验证、lint、format、build、install 或仿真。
- 本轮按已确认的方案 B 修改控制代码、配置和文档。
- 当前基线提交：`205bba7 feat: stabilize scenario control and MuJoCo artifacts`。
