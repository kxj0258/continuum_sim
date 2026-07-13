# Task Plan: 控制流程、架构与跟踪误差分析

## Goal
基于源码、YAML 与用户现有运行产物，说明各控制任务的上下层控制流程、输入输出和参数，并给出统一底层控制架构与误差根因建议。

## Current Phase
Phase 5

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

## Errors Encountered
| Error | Attempt | Resolution |
|-------|---------|------------|
| HANDOFF.md 首次读取乱码 | 1 | 改用显式 UTF-8 读取 |
| 文件枚举包含不存在的 `configs/controllers` | 1 | 保留已获得的模块清单，后续按实际目录读取 |

## Notes
- 不运行 pytest、脚本验证、lint、format、build、install 或仿真。
- 不修改控制代码；本轮交付以分析与架构建议为主。
