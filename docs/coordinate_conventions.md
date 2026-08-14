# 坐标和命令约定

## 单位

- 长度：`m`。
- 时间：`s`。
- 曲率：`1/m`。
- 线速度：`m/s`。
- 角速度：`rad/s`。
- 力：`N`。
- 力矩：`N·m`。
- 四元数：`[w, x, y, z]`。

## 世界、基座和安装坐标

MuJoCo 使用右手世界坐标系。机械臂末端位姿的变换链为：

```text
T_W_tip = T_W_base · T_base_mount · T_mount_tip
```

- `W`：MuJoCo 世界坐标系。
- `base`：固定或移动基座坐标系。
- `mount`：每条机械臂的安装坐标系。
- `tip`：裸臂末端坐标系。

移动基座命令在世界坐标系中表达：

```text
V_W_base = [vx, vy, vz, wx, wy, wz]
```

## 每段弯曲坐标

每段使用两个曲率分量：

```text
[kx_i, ky_i]
```

三段臂按以下顺序排列：

```text
[kx_1, ky_1, kx_2, ky_2, kx_3, ky_3]
```

手动界面的 `kx/ky` 滑块与数值输入框定义在各段的机械臂局部弯曲坐标中。控件设置对应分量的绝对目标，再由弯曲模型生成九根肌腱目标；该操作不把方向转换为世界坐标，也不求解末端目标。

## 肌腱命令

每条臂的底层命令为九根肌腱长度变化率：

```text
delta_l_dot = [dl1/dt, ..., dl9/dt]
```

MuJoCo actuator 接收肌腱位置目标。执行层根据肌腱速率命令、当前目标、速率限制和执行器限制积分得到新的位置目标。

手动界面的 `compatible` 模式将目标投影到弯曲模型可表达的子空间；`raw tendon` 模式直接操作九维肌腱目标。

## 工具坐标

执行臂末端变换链：

```text
T_W_sensor = T_W_tip · T_tip_sensor
T_W_tcp    = T_W_sensor · T_sensor_tcp
```

配置值为：

```text
tip -> sensor center: 4 mm, 沿局部 +Z
sensor center -> TCP: 14 mm, 沿局部 +Z
tip -> TCP: 18 mm
```

六维力传感器尺寸为 `15 × 15 × 8 mm`。直径 `18 mm` 的球形擦拭工具与传感器重叠包覆，球体后端与传感器后表面相切，TCP 位于球面最前端。

所有自动任务及其对应雅可比统一使用工具 TCP，不使用裸臂 tip。工具偏置产生的线速度项按：

```text
v_tcp = v_tip + omega_tip × r_tip_to_tcp
```

加入系统雅可比。

## 六维力坐标

MuJoCo `force` 和 `torque` 传感器在传感器 site 坐标系中输出：

```text
F_S = [Fx, Fy, Fz]
M_S = [Mx, My, Mz]
```

系统状态同时提供传感器坐标和世界坐标结果：

```text
F_W = R_W_S · F_S
M_W = R_W_S · M_S
```

擦拭法向力由世界坐标力在表面法向上的投影得到。传感器在复位时置零，并对输出执行低通滤波、限幅和重力补偿。

## 擦拭表面法向

`configs/scenarios/mujoco_wiping.yaml`：

```yaml
surface_normal_world: [-1.0, 0.0, 0.0]
```

- `approach_offset_m`：沿表面法向定义预接触点。
- `contact_offset_m`：沿表面法向定义接触轨迹偏移。
- `target_contact_distance_m`：目标接触距离，负值表示向表面内部压入。

## 发动机坐标

发动机网格位姿由 `configs/scenes/engine_scene.yaml` 的 `engine.pose` 定义。标记为 `frame: engine` 的入口、起点和探索路径先在发动机局部坐标系中解释，再通过发动机位置、`frame_offset_m` 和四元数转换到世界坐标系。

`diagnostics.bbox` 用于诊断显示。发动机局部坐标轴由 `preview_visualization` 中的轴长、半径和颜色配置。
