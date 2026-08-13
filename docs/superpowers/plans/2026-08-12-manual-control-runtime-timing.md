# 手动控制运行计时实施计划

> **执行说明：** 按任务逐项实施，并使用复选框（`- [ ]`）跟踪进度。

**目标：** 增加低开销的终端滚动计时，用于识别降低手动控制响应频率的具体阶段。

**架构：** 一个 `RuntimeTimingReporter` 接收手动控制界面、系统后端、MuJoCo 后端和被动三维窗口的阶段耗时。它每 `0.5 s` 聚合并输出平均值/最大值；`kx/ky` 滑块或数值输入事件则在下一个完成的控制周期后输出输入延迟。手动控制拆分后不再启动观测臂相机，因此相机计时只保留在自动场景路径。

**技术栈：** Python、`time.perf_counter`、上下文管理器、pytest。

## 全局约束

- 仅由两个手动控制入口启用终端计时。
- 每 `0.5 s` 输出一次滚动汇总，不在每个控制周期输出。
- 在下一个控制周期完成后输出 `kx/ky` 输入延迟。
- 未经明确授权，不增加依赖，也不运行测试或仿真。

---

### 任务 1：滚动计时报告器

**文件：**
- 新建：`src/continuum_sim/utils/runtime_timing.py`
- 新建：`tests/test_runtime_timing.py`

**接口：**
- 提供：`RuntimeTimingReporter.measure(stage)`、`record(stage, duration_s)`、`mark_input(label)`、`start_cycle()` 和 `finish_cycle()`。

- [ ] 编写滚动平均值/最大值和下一周期输入延迟测试。
- [ ] 使用 `perf_counter()` 与 `0.5 s` 报告窗口实现聚合。
- [ ] 获得授权后使用 `pytest tests/test_runtime_timing.py -q` 验证。

### 任务 2：记录控制与物理阶段耗时

**文件：**
- 修改：`src/continuum_sim/visualization/mujoco_system_debug_viewer.py`
- 修改：`src/continuum_sim/backends/mujoco_system_backend.py`
- 修改：`src/continuum_sim/backends/mujoco_backend.py`
- 修改：`tests/test_mujoco_system_debug_viewer.py`

**接口：**
- 输入：由所有层共享的可选 `RuntimeTimingReporter`。
- 输出：输入回调、命令准备、内层循环、MuJoCo 子步、最终前向计算、状态构建和周期总耗时。

- [ ] 测试 `kx/ky` 滑块和数值输入回调会标记输入事件。
- [ ] 增加可选计时注入，不改变非手动路径的行为。
- [ ] 用报告器的 `measure()` 上下文管理器包裹各命名阶段。
- [ ] 获得授权后验证相关测试。

### 任务 3：记录展示耗时并连接手动入口

**文件：**
- 修改：`src/continuum_sim/visualization/manual_control_app.py`

**接口：**
- 输入：由曲率和拉线手动入口创建的共享报告器。
- 输出：控制面板、状态窗口、拉线监测窗口和被动三维窗口耗时。

- [ ] 在手动控制组合根创建一个报告器。
- [ ] 将其传给界面、后端和窗口管理器。
- [ ] 记录展示耗时，但不增加逐帧输出。
- [ ] 获得授权后分别使用 `python scripts/run_manual_curvature_control.py` 和 `python scripts/run_manual_tendon_control.py` 手动验证。
