# 当前分支架构重构优化方案

## 目标

在保留当前双臂 MuJoCo 主任务、发动机导航、擦拭清洁、observer 视觉反馈和视频记录能力的前提下，逐步降低 `application`、`control`、`runtime`、`io` 的耦合度，使项目更容易阅读、扩展、复用和合并回 `main` 分支。

## 总体分层

推荐将项目稳定为以下单向依赖方向：

1. `model` / `system.types`：只放纯数据模型、控制命令和状态类型。
2. `kinematics` / `actuation` / `dynamics`：依赖基础模型，提供算法能力。
3. `scenes` / `tasks`：定义场景查询、任务规格和 waypoint/task plan 生成。
4. `control`：消费任务计划、场景查询和系统状态，输出标准 `RobotSystemCommand`。
5. `backends`：只负责仿真/硬件状态同步和命令执行，不反向理解具体控制器。
6. `runtime`：只负责循环、hook 生命周期和运行时观测。
7. `io`：只负责 artifact 输出、数据转换和结果归档。
8. `application`：唯一 composition root，负责把配置、backend、task、controller、hooks 组装起来。

## 分阶段方案

### 阶段 1：低风险边界整理

本阶段只做结构拆分，不改变控制行为。

- 抽出 runtime profile，例如 windowless batch profile，避免批量脚本各自手写关闭 viewer、live panel、observer window 的逻辑。
- 抽出 hook factory，让 `SimulationApplication` 不直接持有所有 hook 类型和 observer camera 记录细节。
- 抽出 metadata schema，把控制器和 artifact 之间的字符串 key 先集中管理。
- 保留现有 public API：`SimulationApplication.from_yaml`、`SimulationApplication.from_config`、`load_scenario_config` 不变。

### 阶段 2：application composition root 拆分

- 将 backend 构建逻辑拆到 `application/backend_factory.py`。
- 将 task plan 解析拆到 `application/task_plan_factory.py`。
- 将 controller 构建拆到 `application/controller_factory.py`。
- 将 observer camera target 策略拆到 `application/observer_policy.py`。
- `application.py` 最终只保留加载配置、调用工厂、创建 `SimulationLoop`、保存 artifacts。

### 阶段 3：runtime 和 io 拆分

- 将 `runtime/hooks.py` 拆成 `hook_base.py`、`recording_hooks.py`、`viewer_hooks.py`、`video_hooks.py`、`observer_camera_hook.py`、`diagnostic_hooks.py`。
- 将 `io/scenario_artifacts.py` 拆成 `metadata_exporter.py`、`npz_exporter.py`、`plot_exporter.py`、`video_exporter.py` 和任务专属 artifact 插件。
- 保持 artifact 文件名和输出目录结构兼容，避免破坏已有批处理分析流程。

### 阶段 4：control 模块职责拆分

- 将 `scenario_controllers.py` 拆为 tracking、navigation、wiping、timed trajectory、online reachability。
- 将 `coordinated_tracking.py` 拆为 task stack builder、observer visual servo、inter-arm avoidance、scene avoidance、force/position intent 生成。
- 将 force strategy 与 waypoint 修正、法向速度控制的边界显式化，避免擦拭任务调参时难以判断误差来源。

### 阶段 5：打破核心循环依赖

- 优先移除 `model -> kinematics` 的反向依赖。
- 让 `backends` 只依赖 `system`/`model` 的稳定接口，不直接依赖具体控制策略。
- 将跨层 protocol 放到更稳定的位置，例如 `system.protocols` 或 `scenes.protocols`。

### 阶段 6：架构守护

- 增加轻量 import boundary 检查，防止 `model`、`system`、`backends` 再次反向依赖高层模块。
- 增加 metadata schema 覆盖检查，防止 artifact 输出字段静默丢失。
- 保持五个主任务作为行为基准：tracking、navigation、engine_navigation、wiping、point_servo。

## 当前提交落地范围

本次提交先实现阶段 1 的低风险部分：

- 新增统一 windowless batch runtime profile。
- 批量运行脚本统一使用该 profile。
- 新增 runtime metadata schema，并让 artifact 导出从集中 schema 读取 observer/visual-servo 字段。
- 新增 application hook factory，隔离 hook 构建和视频路径策略。

## 后续建议

下一步优先拆 `application/backend_factory.py` 和 `application/task_plan_factory.py`，因为这两部分仍然让 `application.py` 持有较高 fan-out。等五个主任务输出稳定后，再拆 control 内部求解逻辑。
