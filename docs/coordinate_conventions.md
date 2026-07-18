# 坐标和命令约定

## 世界坐标

主线 MuJoCo 场景使用右手世界坐标系，长度单位为米。位姿中的四元数统一采用：

```text
[w, x, y, z]
```

常见变换链为：

```text
T_W_tip = T_W_base * T_base_mount * T_mount_tip
```

- `W`：MuJoCo 世界坐标。
- `base`：移动基座坐标。
- `mount`：机械臂安装坐标。
- `tip`：连续体机械臂末端或工具坐标。

## 移动基座命令

移动基座命令为世界坐标下的空间速度：

```text
V_W_base = [vx, vy, vz, wx, wy, wz]
```

线速度单位为 m/s，角速度单位为 rad/s。

## 肌腱命令

每条臂的底层命令是臂局部肌腱长度变化率：

```text
delta_l_dot = [dl1/dt, ..., dl9/dt]
```

单位为 m/s。MuJoCo position actuator 最终接收的是肌腱位置 target，
该 target 由 tendon inner loop 根据命令速度、实际肌腱状态和限幅约束生成。

## 弯曲坐标和相容性

每段的控制弯曲坐标为：

```text
[kx_i, ky_i]
```

三段臂对应：

```text
[kx_1, ky_1, kx_2, ky_2, kx_3, ky_3]
```

PCC 内部仍可使用 `[kx_i, ky_i, eps_i]`，但主线控制中轴向伸长项通常置零。
肌腱位移必须尽量落在弯曲模型可解释的相容子空间内。相容残差越大，说明实际肌腱状态
越难由当前弯曲坐标解释。

## 擦拭任务法向

`mujoco_wiping.yaml` 中黑板法向为：

```yaml
surface_normal_world: [-1.0, 0.0, 0.0]
```

`contact_offset_m` 沿该法向定义接触轨迹偏移，`approach_offset_m` 沿该法向定义板外侧预接触点。
