# 双臂 MuJoCo 手动控制

手动控制拆分为两个独立程序：

- `run_manual_curvature_control.py`：按机械臂分段控制 `kx/ky`。
- `run_manual_tendon_control.py`：直接控制每根拉线的目标位移。

两个程序默认打开三个彼此独立的窗口：

1. 当前模式的控制窗口。
2. 只读实时状态窗口。
3. MuJoCo 三维仿真窗口。

手动控制不会启动观察臂相机渲染，也不会打开视觉反馈图像窗口。拉线长度和拉力图位于独立诊断窗口中，默认关闭。

## 安装与运行

```powershell
python -m pip install -e ".[mujoco]"
python scripts/run_manual_curvature_control.py
python scripts/run_manual_tendon_control.py
```

默认场景为：

```text
configs/scenarios/mujoco_manual_control.yaml
```

也可以传入其他双臂 MuJoCo 场景：

```powershell
python scripts/run_manual_curvature_control.py configs/scenarios/mujoco_wiping.yaml
python scripts/run_manual_tendon_control.py configs/scenarios/mujoco_wiping.yaml
```

常用参数：

```powershell
python scripts/run_manual_curvature_control.py `
  --panel-fps 15 --status-fps 5 --viewer-fps 15 --curvature-step 0.5

python scripts/run_manual_tendon_control.py `
  --panel-fps 15 --status-fps 5 --viewer-fps 15
```

- `--panel-fps`：控制窗口的目标刷新率，默认 `15 Hz`。
- `--status-fps`：只读状态及可选拉线监测窗口的目标刷新率，默认 `5 Hz`。
- `--viewer-fps`：MuJoCo 三维窗口的目标刷新率，默认 `15 Hz`。
- `--curvature-step`：曲率程序每次按钮操作的增量，单位 `1/m`。
- `--show-tendon-monitor`：额外打开拉线长度和拉力图，默认不打开。

两个入口都会每 `0.5 s` 在终端输出一次时序统计。50 Hz 控制线程使用独立单调时钟，不依赖 Matplotlib 定时器；控制窗口、状态窗口、可选拉线监测窗口和 MuJoCo viewer 只消费最新状态，显示较慢时不会积压历史帧。

## 曲率控制程序

两条机械臂的每一段均提供四个按钮：

```text
+kx：kx += Δκ
-kx：kx -= Δκ
+ky：ky += Δκ
-ky：ky -= Δκ
```

`kx/ky` 是机械臂分段局部弯曲坐标。回调只更新目标值和 dirty 标记；弯曲模型将六维分段曲率转换为九维拉线目标，并统一缩放到拉线行程范围内。控制窗口不创建任何拉线 Slider/TextBox。

`curvature step` 文本框可在运行时修改 `Δκ`。该程序固定使用 `bending_compatible` 控制空间，不提供模式切换。

## 拉线控制程序

每条机械臂提供九个拉线 Slider 和数值输入框，单位为 `mm`。该程序固定使用 `raw_tendon_debug` 控制空间，不创建曲率按钮，也不把独立拉线目标投影回弯曲子空间。

命名预设可发送单根拉线、第一段三根拉线或全部拉线的调试目标。

## 通用控制

两个控制程序都保留：

- `Reset`：复位 MuJoCo、机械臂目标和基座目标。
- `Zero arms`：将两条机械臂目标设为零。
- `Zero base`：把基座目标恢复到本次复位后的初始位姿。
- `Step`：推进一个 `0.02 s` 控制周期。
- `Run/Pause`：按 50 Hz 独立控制时钟连续运行或暂停。

基座区域可设置世界坐标系中的六自由度目标：`X/Y/Z` 使用米，`Roll/Pitch/Yaw` 使用度。`coarse/fine` 用于切换粗调和微调步长。

## 实时状态窗口

实时状态窗口只包含文本，不绘制拉线长度或拉力。它显示：

- 仿真时间和固定控制模式。
- 基座目标/实际位置与姿态。
- 各臂各段目标/实际曲率。
- 各段末端世界坐标位置。
- 可用时显示末端力、力矩及饱和状态。

单独关闭状态窗口不会停止控制，关闭控制窗口会停止控制线程并关闭本程序创建的其他窗口。

## 可选拉线监测窗口

需要观察拉线长度和拉力时显式启动：

```powershell
python scripts/run_manual_curvature_control.py --show-tendon-monitor
python scripts/run_manual_tendon_control.py --show-tendon-monitor
```

该窗口只创建一次柱状图 Artist，后续更新柱高和纵轴范围，不清空坐标轴，也不重建图例和刻度。默认不创建该窗口，因此常规手动控制不承担这部分计算和重绘开销。

## 仿真时钟

```text
controller_dt_s = 0.02 s
mujoco_timestep = 0.001 s
n_substeps = 20
```

每次 `Step` 或连续控制周期推进 20 个 MuJoCo 子步。程序启动时验证：

```text
n_substeps × mujoco_timestep == controller_dt_s
```
