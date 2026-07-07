# 双臂连续体 MuJoCo Spatial Tendon 说明

本文档说明当前双臂 SolidWorks STL、12 孔模板、MuJoCo spatial tendon，以及 6D 基座控制接口。

## 配置入口

- `configs/mujoco_dual.yaml`：双臂 MuJoCo 后端入口。
- `configs/robots/dual_arm_3seg.yaml`：双臂机器人、18 根肌腱和 18 个 motor 定义。
- `configs/robots/dual_arm_hole_pattern.yaml`：base、奇数 link、偶数 link 三类 12 孔模板。
- `configs/robots/dual_arm_meshes.yaml`：双臂 STL mesh 清单。
- `configs/robots/dual_mobile_base_pose.yaml`：统一 6D mobile base 初始位姿、限位和手动控制步长。

## 肌腱数量

每条臂 3 段，每段 3 根肌腱，所以每条臂 9 根肌腱，双臂共 18 根肌腱。

```text
0..8     executor_tendon_1 .. executor_tendon_9
9..17    observer_tendon_1 .. observer_tendon_9
```

每个 link 和每条臂基座仍保留 12 个物理孔位 site。当前 tendon 只使用其中 9 个孔位：

```text
segment 1: hole_01, hole_05, hole_09
segment 2: hole_03, hole_07, hole_11
segment 3: hole_04, hole_08, hole_12
```

## 孔位模板

`configs/robots/dual_arm_hole_pattern.yaml` 分为三类模板：

- `hole_pattern.link_odd.holes[]`：每条臂全局编号为奇数的 link 上的 12 个孔。
- `hole_pattern.link_even.holes[]`：每条臂全局编号为偶数的 link 上的 12 个孔。
- `hole_pattern.base.holes[]`：每条臂基座上的 12 个孔。

两层 hole 都必须定义：

- `xy_m`：孔在对应局部截面内的中心坐标。
- `in_z_m`：入口截面的局部 z 坐标。
- `out_z_m`：出口截面的局部 z 坐标。

`hole_pattern.site_generation` 为入口和出口 site 分别定义颜色：

```yaml
site_generation:
  site_size_m: 0.0006
  in_site_rgba: [1.0, 0.2, 0.1, 1.0]
  out_site_rgba: [0.1, 0.4, 1.0, 1.0]
```

`in_site_rgba` 用于所有 base/link 的 `*_in` site，`out_site_rgba` 用于所有 base/link 的 `*_out` site。两个颜色字段均为必填项；旧的 `site_rgba` 字段不再支持。

link 从每条臂的基座向末端按 1-based 全局顺序编号。当前每段 4 个 link，因此三个 segment 依次对应全局 link 1–4、5–8、9–12。入口和出口使用交叉轴距，Y 轴距离为 `abs(x_m)`，X 轴距离为 `abs(y_m)`：

```text
even link:
  in_z_m  = abs(y_m) * tan(15deg)
  out_z_m = link_length_m - abs(x_m) * tan(15deg)

odd link:
  in_z_m  = abs(x_m) * tan(15deg)
  out_z_m = link_length_m - abs(y_m) * tan(15deg)
```

加载器、MuJoCo XML 生成器和 tendon path overlay 都把 YAML 中的 `in_z_m`、`out_z_m` 作为独立坐标直接使用，不会在代码中用其中一个值推导另一个值。

### 每段末端 link

每个 segment 的 link4 使用 `segment_terminal_links` 单独定义两条臂的三个 7 mm 出口：

| Segment | Executor | Observer | 出口模式 |
|---|---|---|---|
| 1 | hole 03/07/11 | hole 01/05/09 | 非独占 |
| 2 | hole 01/05/09 | hole 03/07/11 | 非独占 |
| 3 | hole 04/08/12 | hole 02/06/10 | 独占 |

“非独占”表示列出的三个 out hole 使用绝对 `z_m: 0.007`，其他远端 tendon 累计经过的 out hole 继续使用普通 `link_even` 坐标。“独占”表示 segment 3 link4 只存在该 arm 列出的三个 out hole。所有 link4 的 12 个 in hole 仍来自 `link_even`。

### 双臂驱动绳布局

两条臂按上述孔组同步设置 `segments[].tendon_angles_deg` 和 `physical_tendons[].angle_deg/hole_index`：

```text
executor:
  segment 1: hole 03/07/11 = [60, 180, 300] deg
  segment 2: hole 01/05/09 = [0, 120, 240] deg
  segment 3: hole 04/08/12 = [90, 210, 330] deg

observer:
  segment 1: hole 01/05/09 = [0, 120, 240] deg
  segment 2: hole 03/07/11 = [60, 180, 300] deg
  segment 3: hole 02/06/10 = [30, 150, 270] deg
```

`path_segment_indices` 保持累计：第二段 tendon 仍经过 segment 1，第三段 tendon 仍经过 segment 1 和 2。

### 孔与拉绳可视化开关

`hole_pattern.visualization` 控制生成 XML 和 viewer overlay 的显示策略：

```yaml
visualization:
  hole_display: routed
  show_tendons: true
```

`hole_display` 支持：

- `none`：所有 hole site 隐藏；拉绳引用的 site 仍以透明形式保留。
- `routed`：只生成并显示当前 link 上确实有 physical tendon 经过的 hole site，默认使用此模式。
- `all`：显示该 link 物理定义中的全部 hole site；独占的 segment 3 link4 仍只显示对应 arm 的三个 out hole。

`show_tendons: false` 会同时隐藏 XML 原生 spatial tendon 和 viewer tendon path overlay，但不会删除 tendon、actuator、sensor，也不会改变仿真动力学。现有 `viewer.overlays.tendon_paths` 仍可单独关闭 viewer overlay；两项都开启时 overlay 才会显示。

当前 base hole 的 `xy_m` 和 link hole 的 `xy_m` 完全相同。基座高度为 `0.020m`，且基座 z 轴朝下，所以 base hole 使用：

```text
in_z_m = 0.020
out_z_m = abs(y_m) * tan(15deg)
```

生成器会把 base hole 写成 `*_base_hole_XX_in/out`，把 link hole 写成 `*_segment_*_link_*_hole_XX_in/out`。每根 spatial tendon 从 base in/out site 开始，再串联对应 link in/out site。

## Spatial Tendon 生成

生成命令：

```bash
python scripts/build_mujoco_dual_arm_model.py --config configs/mujoco_dual.yaml
```

输出：

```text
assets/mujoco/dual_three_segment_arm_tendon_with_visuals.xml
assets/mujoco/dual_three_segment_arm_tendon_with_visuals_mobile_base.xml
```

生成器先写入基础双臂模型，再根据
`configs/robots/dual_mobile_base_pose.yaml` 包装 mobile base。默认输出路径分别由
`tendon_xml_path` 和 `mobile_base_xml_path` 决定；临时生成时可以同时使用
`--output` 和 `--mobile-base-output` 覆盖。

两臂 tendon position actuator 当前统一使用 `kp=40000 N/m` 和
`forcerange=[-30, 30] N`。由于 MuJoCo actuator 接收的是
`neutral_tendon_length + relative_delta` 绝对长度，MJCF 的
`ctrllimited` 必须保持 `false`；`ctrlrange_m` 仍作为软件侧相对位移限幅。

每根 tendon 会写入独立颜色和更细的可视化宽度。`configs/mujoco_dual.yaml` 中的 overlay 半径现在为 `0.0002m`。

## Tendon Path Overlay

`configs/mujoco_dual.yaml` 通过 `viewer.overlays.tendon_path_arms` 控制解释性走线 overlay：

- `default`：只绘制 `dual_robot.default_arm`，当前为主臂 `executor`。
- `observer`：只绘制从臂 `observer`。
- `both`：同时绘制主臂和从臂。
- `none`：不绘制 tendon path overlay。

`viewer.overlays.tendon_paths: false` 仍然是总开关，会直接关闭所有 tendon path overlay。

## 控制接口

MuJoCo spatial tendon 的 `ten_length` 是绝对长度。项目控制器仍使用“相对中立位的长度增量”：

```text
data.ctrl = neutral_tendon_length + tendon_delta_command
BackendState.tendon_length = data.ten_length - neutral_tendon_length
```

低层 `MujocoBackend.step()` 的 raw 控制向量为：

```text
[18D tendon_delta, 6D mobile_base_xyz_rpy]
```

其中 base 6D 为：

```text
[x, y, z, roll, pitch, yaw]
```

`scene_builder.inject_mobile_base_wrapper()` 会给 `mobile_base` body 注入 `mobile_base_freejoint`。`MujocoBackend.step()` 接收到 24 维整机命令时，会先写入 6D freejoint qpos，再写入 18 维 tendon actuator 控制。

scenario 主接口不直接拼接这个 raw 向量，而是使用 `RobotSystemCommand`：

```text
[base_twist(6), executor_tendon_rate(9), observer_tendon_rate(9)]
```

`MujocoSystemBackend` 会把每条臂的 tendon-rate 命令积分为相容 tendon target，再按低层 MuJoCo
raw 控制顺序写入 `[tendon_target, base_pose_rpy]`。旧单臂 runtime 仍可能通过
`DualArmCommandAdapter` 将默认主臂 9 维命令扩展到双臂 XML；新控制器应优先输出命名的
`RobotSystemCommand`，不要依赖默认主臂扩展。

## 诊断视图

旧 tendon debug CLI 已删除；整机调试能力现在应通过 scenario hooks 和 MuJoCo viewer 配置继续收敛：

- 支持 18 根 tendon 的滑条。
- 支持 6D mobile base 滑条。
- executor/observer 的 tendon 分开标注。
- tendon path overlay 使用多色调色板，便于确认每根肌腱是否穿过正确 base/link 孔位。

## 主要风险

- `dual_arm_hole_pattern.yaml` 的 base/link 坐标必须和 CAD 局部坐标一致，否则 tendon 会穿错孔。
- mobile base 当前通过 freejoint qpos 直接写入，适合调试和任务接口打通；如果后续需要动力学 actuator，需要另建 6D actuator 模型。
- 当前控制器仍使用 PCC 近似耦合矩阵估计 q，不是直接用 spatial tendon 的解析 Jacobian。
