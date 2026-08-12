# 双臂 MuJoCo 手动控制

手动控制程序同时打开三个窗口：

1. 双臂控制与诊断面板。
2. MuJoCo 三维仿真窗口。
3. `observer` 末端相机图像窗口。

## 安装与运行

```powershell
python -m pip install -e ".[mujoco]"
python scripts/run_manual_control.py
```

The manual-control entry point prints a timing summary every 0.5 seconds.
Stage values use `average/maximum` milliseconds. The control loop runs on an
independent 50 Hz worker clock; `cycle.interval` is its real start-to-start
interval, `cycle.total` is control work time, and `control.wait` is its idle
time. The Matplotlib panel, passive viewer, and observer camera are refreshed
independently at lower rates. Clicking a `kx` or `ky` button also prints
`callback->cycle` and `callback->complete` latency for the next completed
control cycle.

默认场景：

```text
configs/scenarios/mujoco_manual_control.yaml
```

也可以把其他双臂 MuJoCo 场景交给手动控制器：

```powershell
python scripts/run_manual_control.py configs/scenarios/mujoco_wiping.yaml
```

命令行参数：

```powershell
python scripts/run_manual_control.py --panel-fps 15 --viewer-fps 15 --camera-fps 10 --curvature-step 0.5
```

- `--panel-fps`：控制与诊断面板目标刷新率，默认 `15 Hz`。
- `--viewer-fps`：MuJoCo 三维窗口目标刷新率，默认 `15 Hz`。
- `--camera-fps`：观测相机窗口目标刷新率。
- `--curvature-step`：每次曲率按钮的增量，单位 `1/m`。

## 局部曲率按钮

两条臂的每一段都有四个按钮：

```text
+kx：kx += Δκ
-kx：kx -= Δκ
+ky：ky += Δκ
-ky：ky -= Δκ
```

`kx、ky` 是该段的机械臂局部弯曲坐标。按钮直接修改所选段的曲率，不进行世界坐标变换、末端目标规划或逆运动学求解。弯曲模型随后把六维分段曲率转换为九根肌腱目标，并沿目标方向统一缩放以满足肌腱行程限制。

面板中的 `curvature step` 文本框可以在运行时修改 `Δκ`。

## 肌腱控制

每条臂提供九个肌腱滑块和数值输入框，单位为 `mm`。

- `compatible`：将修改后的九维目标投影到弯曲子空间，并同步更新关联肌腱。
- `raw tendon`：直接设置各根肌腱目标。
- `Zero`：将两条臂的目标设为零。
- `Reset`：复位 MuJoCo 状态、肌腱目标和六维力传感器零点。
- `Step`：推进一个 `0.02 s` 控制周期。
- `Run/Pause`：按控制周期连续运行或暂停。

右侧预设目标可以发送单根肌腱、第一段三根肌腱或全部肌腱的测试目标。

## 基座六自由度控制

右侧 `BASE 6-DOF` 区域直接设置移动基座在世界坐标系中的目标位姿：

- `X/Y/Z`：平移目标，输入框单位为 `m`。
- `Roll/Pitch/Yaw`：旋转目标，输入框单位为 `deg`。
- `coarse/fine`：切换粗调和微调步长；步长及位姿范围由移动基座配置给出。
- `Zero base`：把基座目标恢复到本次复位后的初始位姿。

程序把当前位姿到目标位姿的误差转换为六维速度指令，并按照基座线速度、角速度和位姿范围限幅。基座移动不会改变双臂局部 `kx/ky` 按钮的定义。

## 实时显示

诊断区按机械臂和分段显示：

```text
S1 k=(target_kx,target_ky) act=(actual_kx,actual_ky)
   pW=(x,y,z) m
```

- `k`：目标局部曲率，单位 `1/m`。
- `act`：根据实际肌腱位移估计的局部曲率。
- `pW`：该段末端在 MuJoCo 世界坐标系中的实际位置。

执行臂还显示：

```text
F=(Fx,Fy,Fz) N
M=(Mx,My,Mz) Nm
wrench: ok / SATURATED
```

## 末端工具

执行臂附件由 `configs/tools/carbon_remover.yaml` 定义：

- `15 × 15 × 8 mm` 方形六维力传感器。
- 直径 `18 mm` 的包覆式球形擦拭工具。
- 球心位于传感器中心前方 `5 mm`，球体后端与传感器后表面相切。
- 距裸臂末端 `18 mm` 的球面 TCP。

方形传感器用于显示和六维力测量，球形工具参与环境碰撞。MuJoCo 的 `force`/`torque` 传感器输出经过复位置零、15 Hz 低通滤波、输出方向处理、坐标转换、限幅和重力补偿后写入系统状态。

## 轻量场景与相机

默认手动场景不加载发动机或其他任务环境，只保留双臂、移动基座、末端工具、传感器和 observer 相机。这样手动调节曲率时可以减少主窗口与相机的渲染负载；如需在特定环境中调试，仍可将对应场景 YAML 作为位置参数传入。

observer 相机刚性安装在观测臂末端。模型显示一个直径 `7.5 mm` 的半球相机外壳和独立镜头，外壳直径为机械臂末端直径的一半。通过 observer 的分段曲率按钮调整视线，图像窗口会按 `--camera-fps` 指定频率刷新。

## 仿真时钟

```text
controller_dt_s = 0.02 s
mujoco_timestep = 0.001 s
n_substeps = 20
```

每次 `Step` 或连续运行周期推进 20 个 MuJoCo 子步。程序启动时验证：

```text
n_substeps × mujoco_timestep == controller_dt_s
```
