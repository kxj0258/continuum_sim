# 双通道仿真运行时实施计划

> **执行约束：** 分两批测试先行实现；每批只提交自己的范围。仅在获得授权后运行本计划列出的单元测试，不启动 MuJoCo 三维窗口、场景仿真或长时间进程。

**目标：** 将 `50 Hz` 控制关键路径与图形界面、MuJoCo 三维窗口、相机渲染、终端输出和视频编码隔离，同时保持控制器、状态机和物理参数不变。

**总体结构：** 控制线程独占实时 `MjData` 写入，目标与完成状态通过最新值邮箱交换；展示消费者各自持有 `MjData` 副本。自动场景仍同步执行控制关键钩子，展示与录像钩子使用独立时钟和有界队列。

**技术栈：** Python、线程/队列、NumPy、MuJoCo、Matplotlib、pytest。

---

## 第一批：手动控制与公共运行时基础

### 任务 1：公共并发原语

**文件：**

- 新建：`src/continuum_sim/runtime/concurrency.py`
- 修改：`src/continuum_sim/runtime/__init__.py`
- 新建：`tests/test_runtime_concurrency.py`

**先写失败测试：**

- `LatestValueSlot` 发布新值时版本递增，慢消费者只读取最新值。
- `TimeRateGate` 使用绝对时间推进，迟到时跳过过期截止点。
- `MonotonicRateRunner` 回调异常可查询，并且 `stop()` 幂等。
- `AsyncLinePrinter.write()` 不在调用线程执行目标 printer，`close(drain=True)` 会排空。

**实现接口：**

```python
class LatestValueSlot(Generic[T]):
    def publish(self, value: T) -> int: ...
    def snapshot(self) -> tuple[T, int]: ...
    def consume_after(self, version: int) -> tuple[T, int] | None: ...

class TimeRateGate:
    def due(self, now_s: float | None = None) -> bool: ...
    def reset(self, now_s: float | None = None) -> None: ...

class MonotonicRateRunner:
    def start(self) -> None: ...
    def stop(self) -> None: ...
    def raise_if_failed(self) -> None: ...

class AsyncLinePrinter:
    def write(self, line: str) -> None: ...
    def close(self, *, drain: bool = True) -> None: ...
```

调度器使用绝对截止时间；较长剩余时间用可中断 `Event.wait()`，最后的短窗口用高分辨率等待；若已错过多个周期，则直接跳到未来最近截止点，不补跑旧周期。

**最小测试命令：**

```powershell
pytest -q tests/test_runtime_concurrency.py
```

### 任务 2：MuJoCo 动态状态副本

**文件：**

- 新建：`src/continuum_sim/runtime/mujoco_state_copy.py`
- 新建：`tests/test_mujoco_state_copy.py`

**先写失败测试：**

- 复制 `time/qpos/qvel/act/ctrl/mocap_pos/mocap_quat/userdata`，目标对象与源数组不共享内存。
- 对无执行器、无 mocap 或无 userdata 的零长度数组安全。

**实现接口：**

```python
MUJOCO_DYNAMIC_ARRAY_FIELDS = (...)

def copy_mujoco_dynamic_state(source: object, destination: object) -> None:
    destination.time = source.time
    for field_name in MUJOCO_DYNAMIC_ARRAY_FIELDS:
        np.copyto(getattr(destination, field_name), getattr(source, field_name))
```

复制函数只处理状态输入；消费者随后在自己的 `MjData` 上调用 `mujoco.mj_forward()` 生成派生量。

**最小测试命令：**

```powershell
pytest -q tests/test_mujoco_state_copy.py
```

### 任务 3：目标与状态最新值邮箱，缩短实时数据锁

**文件：**

- 修改：`src/continuum_sim/visualization/mujoco_system_debug_viewer.py`
- 修改：`tests/test_mujoco_system_debug_viewer.py`

**先写失败测试：**

- `kx/ky` 滑块和数值输入框回调只获取目标锁；当测试人为占用 MuJoCo 锁时回调仍立即完成。
- `step()` 从目标邮箱取得最新快照，只在 `backend.step_system()` 周围获取 MuJoCo 锁。
- 控制步完成后发布最新 `RobotSystemState`；图形界面获取状态不再获取 MuJoCo 锁。
- `set_targets()` 仍只同步实际变化的臂和控件。

**实施内容：**

```python
self._target_lock = RLock()
self._target_slot = LatestValueSlot(copy_targets(self.targets))
self._state_slot = LatestValueSlot(self.state)

def _publish_targets_locked(self) -> None:
    self._target_slot.publish(copy_targets(self.targets))

def step(self) -> RobotSystemState:
    targets, _ = self._target_slot.snapshot()
    state, _ = self._state_slot.snapshot()
    command = self._build_command(state, targets)
    with self._simulation_lock:
        next_state = self.backend.step_system(state, command, self.dt)
    self._state_slot.publish(next_state)
    self.state = next_state  # 仅保留兼容读取，不作为线程交换通道
    return next_state
```

用 `MonotonicRateRunner` 替换文件内 `_FixedRateControlWorker`。面板定时器只消费最新状态和待刷新控件，不参与控制时钟。

**最小测试命令：**

```powershell
pytest -q tests/test_mujoco_system_debug_viewer.py
```

### 任务 4：独立三维窗口数据与手动相机路径移除

**文件：**

- 修改：`src/continuum_sim/visualization/manual_control_app.py`
- 修改：`tests/test_manual_control_app.py`

**先写失败测试：**

- 被动三维窗口使用新建的 `MjData`，到期时仅在锁内复制，`mj_forward()` 与 `viewer.sync()` 在锁外。
- 手动控制窗口管理器不创建观测臂相机渲染线程、帧邮箱或视觉反馈图像窗口。
- 三维窗口只消费最新状态，按独立时间刷新门同步；控制线程不调用窗口接口。
- `close()` 停止控制线程后销毁状态窗口、可选拉线监测窗口和三维窗口。

**实施内容：**

- `ManualControlWindows` 只持有实时数据和三维窗口数据副本；三维窗口在独立副本上调用 `mj_forward()`。
- 手动控制组成根不查询相机附件，不创建渲染器，也不调度相机帧。
- 图形界面定时器使用独立 `TimeRateGate`，不执行离屏渲染。

**最小测试命令：**

```powershell
pytest -q tests/test_manual_control_app.py
```

### 任务 5：异步计时输出与第一批提交

**文件：**

- 修改：`scripts/run_manual_curvature_control.py`
- 修改：`scripts/run_manual_tendon_control.py`
- 修改：`src/continuum_sim/utils/runtime_timing.py`（仅在需要暴露 close/诊断时）
- 修改：`tests/test_runtime_timing.py`
- 修改：`docs/manual_control.md`

**先写失败测试：**

- `finish_cycle()` 调用的输出器是非阻塞入队器。
- 关闭时排空已生成的计时文本，不丢失最后一批统计。

**实施内容：**

- 命令行入口创建一个 `AsyncLinePrinter(print)`，传入 `RuntimeTimingReporter(printer=printer.write)`。
- 正常退出和异常退出都按控制线程 → 异步输出器 → 图形窗口的顺序关闭。
- 文档说明 `50 Hz` 是控制调度目标，控制面板、状态窗口和三维窗口是可丢帧显示通道。

**第一批回归命令：**

```powershell
pytest -q tests/test_runtime_concurrency.py tests/test_mujoco_state_copy.py tests/test_mujoco_system_debug_viewer.py tests/test_manual_control_app.py tests/test_runtime_timing.py
```

**第一批提交：**

```powershell
git add src/continuum_sim/runtime/concurrency.py src/continuum_sim/runtime/mujoco_state_copy.py src/continuum_sim/runtime/__init__.py src/continuum_sim/visualization/manual_control_app.py src/continuum_sim/visualization/mujoco_system_debug_viewer.py src/continuum_sim/utils/runtime_timing.py scripts/run_manual_curvature_control.py scripts/run_manual_tendon_control.py tests/test_runtime_concurrency.py tests/test_mujoco_state_copy.py tests/test_mujoco_system_debug_viewer.py tests/test_manual_control_app.py tests/test_runtime_timing.py docs/manual_control.md docs/superpowers/plans/2026-08-12-dual-channel-runtime-implementation.md
git commit -m "性能(手动控制): 隔离控制循环与渲染线程"
```

---

## 第二批：自动场景展示与录像管线

### 任务 6：Hook 分类与独立实时限速器

**文件：**

- 修改：`src/continuum_sim/runtime/simulation_loop.py`
- 修改：`src/continuum_sim/runtime/viewer_hooks.py`
- 修改：`src/continuum_sim/application/hook_factory.py`
- 修改：`tests/test_simulation_loop_state_enrichment.py`
- 修改：`tests/test_mujoco_tracking_runtime.py`

**先写失败测试：**

- 控制关键 `enrich_state()` 和 completion Hook 保持原顺序、每周期同步执行。
- `RealtimePacerHook` 每周期按仿真时间限速，即使 viewer 只低频刷新也不改变实时倍率。
- `MujocoViewerHook` 用时间 gate 更新独立 `MjData`，不再负责 realtime sleep。

**实施内容：**

```python
class RealtimePacerHook:
    def on_reset(self, state): ...
    def on_step(self, state, command, index):
        sleep_until_sim_deadline(...)

class MujocoViewerHook:
    def on_step(...):
        if not self._display_gate.due():
            return
        copy_live_data_under_lock()
        mujoco.mj_forward(self.model, self._viewer_data)
        self._viewer.sync()
```

### 任务 7：observer 反馈与画面路径分离

**文件：**

- 修改：`src/continuum_sim/runtime/observer_camera_hooks.py`
- 修改：`tests/test_observer_camera_hook.py`

**先写失败测试：**

- 几何反馈 metadata 每控制周期更新，和 display/video FPS 无关。
- 离屏渲染只按时间 gate 提交，慢渲染时输入被 latest-value 合并。
- GUI 关闭只禁用 presentation，不阻断同步反馈。

**实施内容：**

- `enrich_state()` 只计算相机位姿、ROI、深度与像素误差。
- renderer worker 独占 renderer 与独立 `MjData`；展示端只消费最新 RGB。
- `on_finish()` 排空录像帧并显式关闭 worker，错误写入现有 artifact 错误路径。

### 任务 8：视频有界 FIFO 与 backpressure

**文件：**

- 修改：`src/continuum_sim/runtime/video_hooks.py`
- 修改：`src/continuum_sim/runtime/video_utils.py`
- 修改：`tests/test_mujoco_video_export.py`

**先写失败测试：**

- worker 按序写入所有已接收帧。
- 队列满时不静默丢帧：记录 overload 次数并向调用方暴露。
- `on_finish()` 排空队列后关闭 GIF/MP4 writer；异常路径也关闭 writer。

**实施内容：**

- 仿真线程只捕获带序号的动态状态快照并入有界 FIFO。
- 视频 worker 独占 `MjData`、renderer 和 writer。
- 保留现有 artifact 文件名与错误文件语义。

### 任务 9：实时面板复用 Artist

**文件：**

- 修改：`src/continuum_sim/runtime/live_panel_hooks.py`
- 修改：`src/continuum_sim/runtime/viewer_hooks.py`
- 修改：`src/continuum_sim/visualization/wiping_force_panel.py`
- 修改：`tests/test_wiping_force_panel.py`
- 新建或修改：`tests/test_live_panel_hooks.py`

**先写失败测试：**

- 首次 reset 创建 Line2D/Text/reference artists，后续刷新只调用 `set_data()`/`set_text()`。
- 运行期不调用 `Axes.clear()`、`Axes.cla()` 或 `canvas.flush_events()`。
- 所有面板用 `TimeRateGate`，20 FPS 可产生 2/3 控制周期交替间隔。

**实施内容：**

- tendon bar 保持现有 persistent bar。
- wiping、tracking 与 diagnostics panel 缓存 line/text/reference artists；动态事件标记维护独立 artist 列表并局部 remove/recreate，不清空轴。
- 每次刷新最后只调用一次 `draw_idle()`。

### 任务 10：交互场景运行协调与第二批提交

**文件：**

- 修改：`src/continuum_sim/application/application.py`
- 修改：`src/continuum_sim/runtime/simulation_loop.py`
- 修改：`tests/test_mujoco_wiping_runtime.py`
- 修改：`docs/runtime.md` 或现有对应运行文档

**先写失败测试：**

- 启用 Matplotlib 实时面板时，仿真循环在工作线程，主线程只调度图形界面展示。
- 无界面/无面板路径仍同步运行，不创建图形界面工作线程。
- 工作线程异常传播到应用；正常和异常退出都关闭所有展示/视频工作线程。

**实施内容：**

- 增加仅在存在 Matplotlib 展示输出端时启用的交互协调器。
- 工作线程发布最新 `(state, command, step_index)`；主线程用图形界面定时器消费并更新面板。
- 仿真完成后主线程执行最后一次展示更新并按既定顺序关闭资源。

**第二批回归命令：**

```powershell
pytest -q tests/test_simulation_loop_state_enrichment.py tests/test_mujoco_tracking_runtime.py tests/test_observer_camera_hook.py tests/test_mujoco_video_export.py tests/test_wiping_force_panel.py tests/test_live_panel_hooks.py tests/test_mujoco_wiping_runtime.py
```

**第二批提交：**

```powershell
git add src/continuum_sim/runtime src/continuum_sim/application src/continuum_sim/visualization/wiping_force_panel.py tests docs
git commit -m "性能(运行时): 解耦场景展示与视频管线"
```

## 不在自动执行范围内的手动验收

两批提交完成后，建议由用户手动运行：

```powershell
python scripts/run_manual_curvature_control.py
python scripts/run_manual_tendon_control.py
```

观察终端计时：`cycle.interval` 接近 `20 ms`、控制频率接近 `50 Hz`，`mujoco.steps` 不再包含三维窗口、相机、图形界面或终端输出耗时；连续拖动 `kx/ky` 滑块或在数值输入框中提交目标时，回调到下一控制周期通常不超过一个控制周期。自动场景另选择一个现有短场景，分别以图形界面和 `--headless` 模式运行，核对任务结果与产物文件兼容性。
