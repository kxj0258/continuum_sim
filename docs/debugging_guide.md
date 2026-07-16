# 调试指南

本文整理当前项目最实用的排查入口。目标是先用可复现的场景配置和运行产物定位问题，再按需要打开
viewer 或更重的 MuJoCo 工具。

## 快速定位顺序

1. 从 `configs/scenarios/<name>.yaml` 开始，确认 `backend.type`、`task.type` 和 `runtime.max_steps`。
2. 关闭交互 viewer：`hooks.viewer: none`，避免窗口生命周期影响复现。
3. 开启轻量记录：`hooks.recorder: true`、`artifacts.save_npz: true`、`artifacts.save_plots: true`。
4. 运行后查看 `output/runs/<scenario>_<timestamp>/metadata.json` 和 `result.npz`。
5. 如果 MuJoCo GIF 失败，先看 `videos/video_error.txt`，数值产物通常仍然可用。

## 常用 hooks

```yaml
hooks:
  recorder: true
  tendon_debug: true
  tendon_debug_stride: 5
  viewer: none
  keep_viewer_open: false
  show_live_tendon_panel: false
  show_live_force_panel: false
  show_live_diagnostics_panel: false
```

需要实时观察时再打开：

```yaml
hooks:
  viewer: mujoco
  keep_viewer_open: true
  show_live_tendon_panel: true
  live_tendon_panel_stride: 5
  show_live_force_panel: true
  live_force_panel_stride: 5
  show_live_diagnostics_panel: true
  live_diagnostics_panel_stride: 5
```

## 运行产物排查点

双臂 tracking 优先查看 `result.npz` 中的：

- `arm_observer_tendon_target_m` / `arm_observer_tendon_displacement_m`
- `arm_observer_tendon_inner_loop_mode` / `arm_observer_tendon_servo_evaluated`
- `arm_observer_command_rate_mps` / `arm_observer_constrained_command_rate_mps`
- `arm_observer_tendon_target_rate_fd_mps` / `arm_observer_tendon_realized_rate_fd_mps`
- `arm_observer_tendon_measured_rate_filtered_mps`
- `arm_observer_tendon_velocity_sensor_raw_mps` / `arm_observer_actuator_force_n`
- `observer_control_mode` / `observer_collision_active`
- `inter_arm_distance_m` 与 influence/minimum/critical 阈值

对应同步图为 `arm_observer_synchronized_control.png` 和
`dual_arm_synchronized_safety.png`。command 信号使用 `command_time_s`，state/actuator 信号使用
`time_s`，不要把两者按相同数组下标误认为同一时刻。

```text
output/runs/<scenario>_<timestamp>/
  metadata.json              场景、后端、运行产物错误和视频错误摘要
  result.npz                 time、tip、target、error、force/contact 等数组
  configs/                   运行时配置副本
  model/                     运行用 MuJoCo XML 副本
  plots/                     静态诊断图
  videos/simulation.gif      回放或 live_mujoco GIF
  videos/video_error.txt     视频导出失败原因
```

如果只改控制器，优先比较 `tracking_error_m`、`achieved_waypoint_error_m`、
`arm_tendon_target_error_norm_m` 和 `arm_saturation_scale`。如果只改场景或路径生成，优先看
waypoints、local path plots 和 generated XML。

### Bending-rate servo 排查

按下面的链路逐层比较，避免把不同含义的 rate 混在一起：

```text
requested command
  -> constrained command
  -> actuator target finite difference
  -> realized tendon displacement finite difference
     -> filtered measured rate（下一次 servo 反馈）
     \-> raw MuJoCo velocity sensor（并列的瞬时对照）
```

先用 `arm_<name>_tendon_servo_evaluated` 过滤实际执行 servo 的 transition；`raw_debug`、`legacy`
或 `unknown` 周期中的 guard/anti-windup 默认占位值不能解释为控制故障。

- command 高、constrained 低：入口 rate/displacement/lead 约束在裁剪，先查 `rate_saturated` 和
  `tendon_guard_feasible`，不要调 actuator gain。
- constrained 高、target FD 低：target lead/force guard 正在限制当前 target，anti-windup 随后回算响应；检查
  `tendon_target_lead_utilization`、`tendon_force_constraint_active` 和 force utilization。
- target FD 高、realized FD 低：position actuator/结构动力学没有实现 target；检查实际 force 是否接近
  physical limit。此时继续提高积分只会触发 guard，应降低 reference speed 或增加上层 governor。
- realized FD 与 `arm_<name>_tendon_measured_rate_filtered_mps` 差异大且高频抖动：适当增大
  `rate_filter_time_constant_s`，同时确认 `controller_dt_s == n_substeps * timestep`。
- command 归零后 target 仍缓慢漂移：检查 `zero_command_mode`、rate-error integral 和 anti-windup；
  `hold` 请求保持上一 target，`relax` 请求受限地卸力回到 actual；两者都可能被联合 guard 修正。
- `tendon_guard_feasible` 为 false 或 constraint violation 非零：不要继续调增益。先检查 actual tendon
  是否越过绝对位移边界、assembly lead 与 actuator force/`kp` 推导的 lead 是否冲突。此时 backend
  会优先执行逐 tendon 安全回撤；若 `tendon_compatibility_bypassed_for_safety` 为 true，应把该周期视为
  fault recovery，而不是正常 tracking 样本。
- PCC–MuJoCo tip residual 大但 tendon target/realized 链路正常：这是模型映射/物理参数问题，转到
  本文后面的 PCC–MuJoCo 诊断，不要让 tendon servo 补偿 Cartesian 模型误差。

`arm_<name>_applied_rate_mps` 是 constrained reference 的兼容别名，不是物理 realized rate；
`arm_<name>_tendon_velocity_mps` 是 raw sensor 的兼容别名。定量比较应使用显式命名的新字段。

## 发动机导航排查

重点 metadata 字段：

- `engine_navigation_phase`
- `engine_navigation_progress`
- `engine_navigation_active_target_m`
- `engine_navigation_active_target_kind`
- `engine_navigation_base_path_m`
- `engine_navigation_insertion_path_m`
- `engine_navigation_executor_paths_m`
- `engine_navigation_observer_roi_m`
- `base_position_error_m`
- `base_orientation_error_rad`

局部路径异常时，先检查 `intermediate_local_paths` 的 `at_fraction`、`type`、`samples`、
`axial_retraction_m` 和 `local_tracking` 推进参数。

## 手动验证命令

本仓库不建议在清理或文档任务里自动运行验证。需要人工检查时，可按风险从轻到重选择：

```powershell
pytest tests/test_tracking_optimization.py tests/test_scenario_migrated_task_features.py
pytest tests/test_bending_space.py
pytest tests/test_scenario_artifacts.py tests/test_staged_engine_navigation.py
python scripts/run_scenario.py configs/scenarios/single_mujoco_tracking.yaml
python scripts/run_scenario.py configs/scenarios/dual_mujoco_tracking.yaml
python scripts/run_scenario.py configs/scenarios/single_analytic_smoke.yaml
python scripts/run_scenario.py configs/scenarios/single_mujoco_smoke.yaml
python scripts/run_scenario.py configs/scenarios/dual_engine_navigation.yaml
python scripts/check_mujoco_offscreen_renderer.py configs/mujoco_dual.yaml
```

## PCC 与 MuJoCo 末端的交互式诊断

### 启动方式

双臂场景：

```powershell
python scripts/debug_mujoco_pcc.py configs/scenarios/dual_mujoco_tracking.yaml
```

同一入口也可加载单臂 MuJoCo 场景：

```powershell
python scripts/debug_mujoco_pcc.py configs/scenarios/single_mujoco_tracking.yaml
```

这是一项需要人工操作窗口的诊断，不应放入自动测试或无人值守脚本。它不会运行
场景的 tracking controller；MuJoCo 只在点击 `Step` 或打开 `Run` 后前进。

### 显示含义

- 紫色中心线和球体：由当前实际肌腱位移重建的 PCC 模型及其末端；
- 青色中心线和球体：MuJoCo link/site 给出的实际中心线和末端；
- 红色连线：从 MuJoCo 末端 site 到 PCC 末端的误差向量；
- `PCC` / `MJ`：两种末端的世界坐标，单位为 mm；
- `dM`：`PCC - MuJoCo` 在该臂安装坐标系下的 XYZ 分量，单位为 mm；
- `|d|`：三维欧氏距离，单位为 mm；
- `compatibility residual`：实际肌腱位移中无法由 PCC 六维弯曲状态解释的分量。

`executor_tip` 和 `observer_tip` 都定义在各自最后一个 link body 的
`pos="0 0 0.01"`，即从最后一个 10 mm link 的近端 body 原点沿局部 Z 轴偏移到
远端。该 link 的 collision geom 同样从局部 Z=0 延伸到 Z=10 mm，因此这个 site
不是在 120 mm 总臂长外额外增加的 10 mm。诊断工具直接使用这个 site，不再手动
加减偏置；在三段各 40 mm、MuJoCo 共 12 个 10 mm link 的直臂状态下，两者理论
长度都应是 120 mm。若直臂仍有明显误差，应继续检查安装变换、关节静态平衡和
离散 link/PCC 曲线定义，而不能直接归因于 site 的 `0.01`。

### 推荐操作顺序

1. 保持 `compatible` 模式并点击 `Reset`，记录两臂零输入误差；
2. 调整一条臂的目标滑块，点击 `Run`；
3. 等待橙色 target 和蓝色 current 柱状图基本重合、末端不再明显运动；
4. 点击 `Pause`，比较 PCC/MuJoCo 坐标和 `dM`；
5. 依次测试第一、第二、第三段主要弯曲方向，并观察误差从哪一段开始增长；
6. 必要时切换 `raw tendon`，但应同时检查兼容性残差，不能把非兼容形变直接归因于
   PCC 几何参数；
7. 点击 `Save CSV` 保存当前会话的全部样本。

`Reset` 后已有样本不会被删除，CSV 中的 `session` 会在仿真时间回退时递增。
CSV 每条臂每个状态一行，包含实际肌腱位移、PCC/MuJoCo 世界坐标、世界/安装坐标
误差、误差模长和兼容性残差。只有点击 `Save CSV` 才会创建目录和文件。

可选参数：

```powershell
python scripts/debug_mujoco_pcc.py configs/scenarios/dual_mujoco_tracking.yaml `
  --samples-per-segment 21 `
  --output output/diagnostics/my_pcc_check.csv
```

`--samples-per-segment` 只改变紫色 PCC 中心线的显示采样密度，不改变末端正运动学
结果。采样过高可能超过 MuJoCo user scene 的 overlay 几何容量，工具会明确报错。

如果目标是检查 viewer 或实时面板，建议先复制一个场景配置，把 `hooks.viewer` 改成 `mujoco`，
并把 `runtime.max_steps` 降低到较小值。
