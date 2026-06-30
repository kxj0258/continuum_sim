# 基线命令实现与工作流导览

本文说明 README 中 8 条基线命令如何实现，以及 YAML 配置如何进入 PCC 模型、MuJoCo runtime、6D mobile base 和三段腱驱连续体臂。

手动运行这些命令前，请先激活项目环境：

```powershell
conda activate continuum_sim
```

本文只描述实现链路，不要求自动运行测试、viewer、安装或仿真命令。

## 总入口

所有基线命令都从 `cli.py` 进入：

```text
parse_args()
  -> COMMAND_HANDLERS[command]
  -> command-specific function
```

`configs/main_config.yaml` 是默认索引文件。它把各命令族分发到更具体的配置：

- `robot_config: configs/robot_3seg.yaml`
- `mobile_base_config: configs/robots/mobile_base_pose.yaml`
- `mujoco_backend_config: configs/mujoco.yaml`
- `pcc_tracking_config: configs/tasks/pcc_trajectory_tracking.yaml`
- `mujoco_tracking_config: configs/tasks/mujoco_trajectory_tracking.yaml`
- `mujoco_navigation_config: configs/tasks/mujoco_navigation_rocket.yaml`
- `mujoco_wiping_config: configs/tasks/mujoco_wiping_board.yaml`

`cli.py::_resolve_path()` 会先按“包含该路径的 YAML 文件所在目录”解析相对路径。如果该候选路径不存在，再按当前工作目录解析。`_tracking_config_path()` 和 `_indexed_config()` 让同一个命令既能接收 `configs/main_config.yaml`，也能直接接收命令专用 YAML。

## 6D Mobile Base 定义

静态 mobile base 和 mount 定义在 `configs/robots/mobile_base_pose.yaml`：

- `mobile_base.pose.position_m` 和 `quat_wxyz` 定义 `T_world_mobile_base`。
- `mounts.arm_mount.position_m` 和 `quat_wxyz` 定义 `T_mobile_base_mount`。
- `limits`、`manual_control` 和 `visualization` 定义位姿限位、建议手动步长和可选 MuJoCo base box。

加载代码分布在：

- `src/continuum_sim/model/base_pose.py`：`Pose6D`，负责四元数归一化、RPY 转换、矩阵转换、compose、inverse、点/pose 变换。
- `src/continuum_sim/model/mount_frame.py`：`load_mobile_base_mount_config()`，把 YAML 加载成 `MobileBaseMountConfig`。
- `src/continuum_sim/model/mobile_base_context.py`：`MobileBaseArmContext`，组合 base pose 和选中的 mount pose。

当前预期坐标链为：

```text
T_world_tip
= T_world_mobile_base
* T_mobile_base_mount
* T_mount_tip
```

其中 `T_mount_tip` 由局部 PCC 或 MuJoCo arm 模型产生。

## 连续体臂定义

三段腱驱连续体臂定义在 `configs/robot_3seg.yaml`：

- `segments`：三段 40 mm 连续体段，包含长度、腱半径、弯曲/扭转参数和腱角。
- `physical_tendons`：9 根物理腱，包含 global index、motor index、锚定段、径向偏置、角度和经过的段索引。
- `motors.items`：9 个电机，包含 spool radius、gear ratio、direction sign、zero position 和对应腱。
- `actuation.limits`：腱长增量和张力限制。

主要加载链路为：

```text
configs/robot_3seg.yaml
  -> ThreeSegmentRobotParams.from_yaml()
  -> load_physical_tendons_from_yaml()
  -> load_motor_params_from_yaml()
```

analytic 主链路为：

```text
motor_position
  -> motor_position_to_tendon_delta()
  -> physical_tendon_delta_to_q()
  -> forward_kinematics()
  -> local tip / centerline
  -> MobileBaseArmContext.local_*_to_world()
```

PCC 状态 `q` 有 9 个量：

```text
[kx1, ky1, eps1, kx2, ky2, eps2, kx3, ky3, eps3]
```

`src/continuum_sim/model/tendon_coupling.py` 构建物理腱耦合矩阵。每根腱会根据自己的 `path_segment_indices` 作用到多个连续体段，`physical_tendon_delta_to_q()` 用伪逆从 9 根腱长增量估计 9D PCC 状态。

## 命令工作流

### `view-pcc`

实现入口：`cli.py::view_pcc()`

```text
main_config.yaml
  -> robot_config
  -> mobile_base_config
  -> ThreeSegmentRobotParams.from_yaml()
  -> MobileBaseArmContext.from_config_path()
  -> PCCInteractiveViewer
  -> forward_kinematics(q)
  -> local centerline/tendon points -> world geometry
```

该命令不使用电机或腱耦合。viewer 使用命名 PCC 状态，例如 `straight`，在 arm mount 局部坐标系中做 FK，然后通过 `MobileBaseArmContext` 把中心线、腱导向点和 mount frame 渲染到 world 坐标中。

### `view-motor-chain`

实现入口：`cli.py::view_motor_chain()`

```text
main_config.yaml
  -> robot_config + mobile_base_config
  -> params + physical_tendons + motor_params
  -> MotorChainInteractiveViewer
  -> motor position / velocity
  -> tendon_delta
  -> q_est
  -> PCC FK
  -> local geometry -> world geometry
```

该命令展示完整 analytic 电机到机械臂链路。`motor_chain_viewer` 和 `simulation` YAML 字段提供初始电机状态、限位、`dt` 和绘图采样数。

### `run-tracking`

实现入口：`cli.py::run_tracking()`

```text
main_config.yaml
  -> pcc_tracking_config
  -> load_tracking_config()
  -> ContinuumKinematicsChain.from_robot_config()
  -> build_target_positions()
  -> simulate_tracking()
  -> plot/animation with MobileBaseArmContext
```

`ContinuumKinematicsChain` 会从 robot YAML 一次性加载 robot params、physical tendons 和 motor params。控制循环是 motor space 中的 damped least-squares differential IK：计算电机速度，积分电机位置，估计 tendon delta 和 PCC `q`，再计算 PCC FK。目标轨迹在 arm local 坐标下生成；绘图和动画阶段再把目标与结果转换到 world 坐标。

### `view-mujoco`

实现入口：`cli.py::view_mujoco()`

```text
main_config.yaml
  -> mujoco_backend_config
  -> load_mujoco_config()
  -> resolve_runtime_xml_path()
  -> MujocoBackend.from_config()
  -> reset()
  -> zero-control step loop / passive viewer
```

`configs/mujoco.yaml` 定义 robot config path、mobile-base config path、原始 XML 路径、generated visual XML 路径、control mode、solver、gravity、joints、tendon model、actuators、sensors、viewer 和 overlays。

`resolve_runtime_xml_path()` 会根据 `control_mode` 和 `viewer.use_segment_visuals` 选择 runtime XML。如果配置了 `mobile_base_config_path`，它会调用 `inject_mobile_base_wrapper()` 并使用生成的 `*_mobile_base.xml`。

### `debug-mujoco-tendons`

实现入口：`cli.py::debug_mujoco_tendons()`

```text
main_config.yaml
  -> mujoco_backend_config
  -> load_mujoco_config()
  -> require control_mode == tendon_position
  -> params + physical_tendons
  -> resolve_runtime_xml_path()
  -> MujocoBackend
  -> MujocoTendonDebugViewer
```

该命令复用 MuJoCo backend，但把 rollout 换成 tendon debug panel。它显示 commanded tendon deltas、actual tendon lengths、actuator forces、tip position 和 q estimates。如果启用了 tendon-path overlays，还会使用 robot physical tendon 定义绘制解释性腱路径。

### `run-mujoco-tracking`

实现入口：`cli.py::run_mujoco_tracking()`。运行主体在 `src/continuum_sim/runtime/mujoco_tracking_runtime.py`。

```text
main_config.yaml
  -> mujoco_tracking_config
  -> task YAML may override mujoco_backend_config
  -> load_mujoco_tracking_config()
  -> load_mujoco_config()
  -> params + physical_tendons + motor_params
  -> MobileBaseArmContext
  -> local target trajectory -> world target trajectory
  -> resolve_runtime_xml_path()
  -> MujocoBackend
  -> control loop
```

控制有两条路径：

- `position_joint`：analytic controller 计算 motor velocity，估计 `q`，再通过 `pcc_q_to_joint_targets()` 把 `q` 转成 hinge joint targets。
- `tendon_position`：当前基线路径。根据 `feedback_mode` 使用 commanded PCC 状态或 MuJoCo 实测 tendon lengths，然后输出 9D tendon-position actuator command。

tracking error 在 world 坐标下计算。local targets 会通过 `MobileBaseArmContext.local_points_to_world()` 转成 world targets；MuJoCo tip pose 则已经是 wrapped runtime XML 下的 world-frame 值。

### `run-mujoco-navigation`

实现入口：`cli.py::run_mujoco_navigation_cli()`。运行主体在 `src/continuum_sim/runtime/mujoco_navigation_runtime.py`。

```text
main_config.yaml
  -> mujoco_navigation_config
  -> load_mujoco_navigation_config()
  -> load_mujoco_config()
  -> load_navigation_scene_config()
  -> build_mujoco_scene_xml()
  -> MujocoBackend
  -> navigation control loop
```

navigation 与 tracking 的主要区别：

- 目标来自 scene YAML 中的 inspection targets 和 ordered mission waypoints。
- scene builder 会把结构化 shell、box、cylinder primitives 注入 generated XML。
- 同一份 scene config 也提供 clearance primitives，供 navigation controller 查询中心线 clearance。

生成 scene XML 时会传入 `mobile_base_config_path`，所以 mobile-base wrapper 和 scene geoms 会插入同一份 runtime XML。

### `run-mujoco-wiping`

实现入口：`cli.py::run_mujoco_wiping_cli()`。运行主体在 `src/continuum_sim/runtime/mujoco_wiping_runtime.py`。

```text
main_config.yaml
  -> mujoco_wiping_config
  -> load_mujoco_wiping_config()
  -> load_mujoco_config()
  -> load_navigation_scene_config()
  -> build_raster_wiping_path()
  -> build_mujoco_wiping_xml()
  -> MujocoBackend
  -> hybrid force-position wiping loop
```

wiping 在 navigation 基础上增加 work-surface 元数据、wipe patches、raster path、注入 XML 的 tip-mounted contact pad、normal-force/contact-proxy 记录、phase tracking，以及可选 live force panel。

当前 wiping runtime 要求 `control_mode='tendon_position'`。

## MuJoCo Mobile-Base XML Wrapper

`src/continuum_sim/scenes/scene_builder.py::inject_mobile_base_wrapper()` 负责把 6D base 写入 MuJoCo XML。

处理流程：

1. 解析 base XML。
2. 查找 `worldbody` 下名为 `base` 的 top-level body；如果不存在，就使用第一个 top-level body。
3. 读取原始 robot root 的 `pos` / `quat`。
4. 加载 `configs/robots/mobile_base_pose.yaml`。
5. 创建名为 `mobile_base` 的新 top-level body，使用配置中的 base pose。
6. 添加 `mobile_base_frame`、可选 `mobile_base_box` 和 mount sites。
7. 从 `worldbody` 移除原始 robot root。
8. 把原始 robot root 挂到 `mobile_base` 下，并把其 pose 设为：

```text
T_mobile_base_robot_root = T_mobile_base_mount * T_original_robot_root
```

最终层级为：

```text
worldbody
  mobile_base body
    mount site(s)
    original robot root body
      continuum arm bodies / sites / tendons / actuators
```

`view-mujoco`、`debug-mujoco-tendons` 和 `run-mujoco-tracking` 通过 `resolve_runtime_xml_path()` 使用这条 wrapper 路径。`run-mujoco-navigation` 和 `run-mujoco-wiping` 则把同一个 mobile-base config 传给 scene XML builders。

## 当前风险

- `run_mujoco_trajectory_tracking()` 的 tolerance-advance 分支引用了 `target_positions`，但函数中实际定义的是 `target_positions_local` 和 `target_positions_world`。当 `mujoco.target_advance_mode: tolerance` 且到达 waypoint 时，可能触发 `NameError`。
- mobile base 当前是 prescribed/static pose，不是 MuJoCo free joint 或动态底盘控制器。
- `inject_mobile_base_wrapper()` 假设 robot root 是名为 `base` 的 top-level body，或者退回到第一个 top-level body。未来如果 XML 有多个 top-level robot body，可能需要更显式的选择规则。
- analytic PCC tracking 在 arm-local 坐标下控制，6D base 主要用于 world-frame 可视化；MuJoCo tracking 会把目标转换到 world 坐标，并与 world-frame MuJoCo tip pose 比较。
- runtime XML 解析和 scene builders 会写出 generated XML 文件。提交前应检查这些生成资产是否需要纳入版本控制。

## 建议手动验证

只在你需要验证行为时手动运行：

```powershell
conda activate continuum_sim
python cli.py view-pcc --config configs/main_config.yaml
python cli.py view-motor-chain --config configs/main_config.yaml
python cli.py run-tracking --config configs/main_config.yaml
python cli.py view-mujoco --config configs/main_config.yaml
python cli.py debug-mujoco-tendons --config configs/main_config.yaml
python cli.py run-mujoco-tracking --config configs/main_config.yaml
python cli.py run-mujoco-navigation --config configs/main_config.yaml
python cli.py run-mujoco-wiping --config configs/main_config.yaml
```

针对代码层检查，可手动运行：

```powershell
python -m pytest tests/test_base_pose.py tests/test_world_kinematics.py -v
python -m pytest tests/test_robot_config.py tests/test_motor_mapping.py tests/test_tendon_coupling.py -v
python -m pytest tests/test_cli_smoke.py -v
python -m pytest -m mujoco
```
