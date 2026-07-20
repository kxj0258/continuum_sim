# 坐标和命令约定

本文记录当前 MuJoCo 主线使用的坐标、位姿和命令约定。

## 基本单位

- 长度：米。
- 时间：秒。
- 角度：弧度，配置里的 viewer 角度除外。
- 力：牛。
- 四元数顺序：`[w, x, y, z]`。

## 世界坐标

主线场景使用 MuJoCo 右手世界坐标系。常见变换链为：

```text
T_W_tip = T_W_base * T_base_mount * T_mount_tip
```

- `W`：MuJoCo 世界坐标。
- `base`：移动基座坐标。
- `mount`：机械臂安装坐标。
- `tip`：连续体机械臂末端或工具坐标。

固定基座任务中，`T_W_base` 通常为单位位姿或配置给定的固定基座位姿。移动基座任务中，控制器会直接更新 base pose。

## 移动基座命令

移动基座命令是世界坐标下的空间速度：

```text
V_W_base = [vx, vy, vz, wx, wy, wz]
```

- 线速度单位：m/s。
- 角速度单位：rad/s。
- 当前 `navigation` 和 `engine_navigation` 会使用移动基座阶段化接近逻辑。

## 肌腱命令

每条臂的标准底层命令是臂局部肌腱长度变化率：

```text
delta_l_dot = [dl1/dt, ..., dl9/dt]
```

单位为 m/s。MuJoCo tendon position actuator 最终接收肌腱位置 target；该 target 由低层 tendon inner loop 根据命令速度、实际肌腱状态、限幅和力约束生成。

## 弯曲坐标

每段的控制弯曲坐标为：

```text
[kx_i, ky_i]
```

三段臂对应：

```text
[kx_1, ky_1, kx_2, ky_2, kx_3, ky_3]
```

PCC 内部仍可使用 `[kx_i, ky_i, eps_i]`，但当前控制主线通常不主动使用轴向伸长项。肌腱位移应尽量落在弯曲模型可解释的相容子空间内；相容残差越大，说明实际肌腱状态越难由当前弯曲坐标解释。

## 工具和附件

工具、喷嘴和相机附件由 `configs/tools/*.yaml` 描述，并挂接到 `configs/robots/*.yaml` 中的 arm 配置。observer 相机反馈会把视觉伺服相关字段写入 state metadata，供控制器、overlay 和 artifacts 使用。

## 擦拭任务法向

`mujoco_wiping.yaml` 中黑板法向为：

```yaml
surface_normal_world: [-1.0, 0.0, 0.0]
```

- `contact_offset_m`：沿该法向定义接触轨迹偏移。
- `approach_offset_m`：沿该法向定义板外侧预接触点。
- `target_contact_distance_m`：目标接触距离，负值表示进入接触/压入代理距离。

## 发动机场景坐标

发动机场景配置位于：

```text
configs/scenes/engine_scene.yaml
```

配置中的 `frame: engine` 注解会先在发动机局部坐标下解释，再结合 `engine.pose.position_m`、`frame_offset_m` 和 `quat_wxyz` 映射到世界坐标。`diagnostics.bbox` 仅用于诊断和可视化，不定义世界原点。
