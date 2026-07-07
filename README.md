# continuum_sim

`continuum_sim` 是一个面向空间连续体机械臂的仿真与控制项目。当前主入口是
scenario YAML，通过同一套应用层组合机器人装配、任务、控制器、后端、hooks
和运行产物。

推荐入口：

```powershell
python scripts/run_scenario.py configs/scenarios/<scenario>.yaml
```

Python 中也可以直接使用：

```python
from continuum_sim.application import SimulationApplication

application = SimulationApplication.from_yaml(
    "configs/scenarios/dual_engine_navigation.yaml"
)
result = application.run()
print(len(result.states))
print(application.last_artifacts.run_dir)
```

## 快速运行

```powershell
# smoke
python scripts/run_scenario.py configs/scenarios/single_analytic_smoke.yaml
python scripts/run_scenario.py configs/scenarios/single_mujoco_smoke.yaml

# tracking
python scripts/run_scenario.py configs/scenarios/single_analytic_tracking.yaml
python scripts/run_scenario.py configs/scenarios/dual_analytic_tracking.yaml
python scripts/run_scenario.py configs/scenarios/single_mujoco_tracking.yaml
python scripts/run_scenario.py configs/scenarios/dual_mujoco_tracking.yaml

# navigation / wiping / engine
python scripts/run_scenario.py configs/scenarios/single_mujoco_navigation.yaml
python scripts/run_scenario.py configs/scenarios/dual_mujoco_navigation.yaml
python scripts/run_scenario.py configs/scenarios/single_mujoco_wiping.yaml
python scripts/run_scenario.py configs/scenarios/dual_mujoco_wiping.yaml
python scripts/run_scenario.py configs/scenarios/single_engine_cleaning.yaml
python scripts/run_scenario.py configs/scenarios/dual_engine_navigation.yaml
```

带 `viewer: mujoco` 或 `viewer: matplotlib` 的场景会打开可视化窗口，可能需要手动关闭。

## 核心控制约定

正常任务统一在每臂 6 维 bending-space 中求解：

```text
b = [kx_1, ky_1, kx_2, ky_2, kx_3, ky_3]
q = S_b b
delta_l = C_b b,  C_b = C_q S_b
```

每段轴向应变固定为零，并与 MuJoCo 配置
`tendon_model.include_axial_strain: false` 保持一致。`tracking`、
`navigation`、`wiping`、`engine_cleaning`、双臂 observer 协同和避障任务均先求
`b_dot`，再由 `C_b` 一次性生成 9 根 tendon 的相容速度。

限速和目标位移限幅使用每臂统一缩放系数，不逐根 tendon 裁剪，避免破坏 tendon
比例。MuJoCo 实际绳长仍可能因为弹性、动力学滞后和求解误差产生小量残差；相关
残差会进入 metadata 和 tendon debug 产物。

## Scenario 配置

推荐配置位于 `configs/scenarios/*.yaml`。旧的 `configs/tasks/*.yaml` 和
`runtime/mujoco_*_runtime.py` 主要保留作兼容与参考。

### Tracking

tracking 支持手写 `waypoints_world`，也支持自动生成轨迹：

```yaml
task:
  type: tracking
  trajectory:
    type: ellipse  # circle, figure-eight, ellipse, line, square, lissajous, helix, dmp
    samples: 80
    radius_m: 0.018
    placement:
      center_mode: straight_tip_xy
      z_mode: straight_tip_minus_radius
      plane: xy
    shape:
      radius_x_m: 0.020
      radius_y_m: 0.012
```

目标推进方式：

```yaml
task:
  target_advance_mode: tolerance  # tolerance 或 time
  waypoint_tolerance_m: 0.002
  advance_steps: 40
  advance_time_s: 0.8
```

`tracking_error_m` 表示当前命令目标误差；`achieved_waypoint_error_m` 只在目标点
达成时有效，更适合作为实际到点精度指标。

### Engine Navigation

`configs/scenarios/dual_engine_navigation.yaml` 是当前发动机插入导航主场景。它包含：

1. 移动底座对准发动机入口。
2. 底座沿 `nozzle_axis_entry` 插入路径推进。
3. 在中间插入点暂停底座，executor 执行局部横向轨迹，observer 跟随 ROI。
4. executor 回到进入局部轨迹时的原插入轴目标，底座目标先保持该点，再继续推进。
5. 终点执行局部轨迹并结束。

局部路径示例：

```yaml
intermediate_local_paths:
  - name: one_third_circle
    at_fraction: 0.3333333333
    type: transverse_circle
    radius_m: 0.075
    samples: 60
    axial_retraction_m: 0.045
  - name: two_thirds_ellipse
    at_fraction: 0.6666666667
    type: ellipse
    radius_m: 0.075
    shape:
      radius_x_m: 0.075
      radius_y_m: 0.045
    samples: 75
    axial_retraction_m: 0.045
```

支持的局部路径类型：

```text
circle / transverse_circle
ellipse / transverse_ellipse
line / transverse_line
square / transverse_square
figure-eight / figure_eight / transverse_figure_eight
lissajous / transverse_lissajous
```

`shape` 可提供：

```text
radius_x_m, radius_y_m       # ellipse / figure-eight / lissajous
length_m                     # line
side_length_m                # square
lissajous_frequency_x
lissajous_frequency_y
lissajous_phase_deg
```

局部轨迹中心计算为：

```text
center = insertion_target - axial_retraction_m * insertion_direction
```

增大 `axial_retraction_m` 会让局部轨迹更靠近机械臂基座；设为 `0.0` 则以插入目标点为中心。

局部路径目标推进：

```yaml
local_tracking:
  advance_mode: tolerance  # tolerance, time 或 steps
  waypoint_tolerance_m: 0.005
  rejoin_tolerance_m: 0.005
  advance_time_s: 0.20
  advance_steps: 10
  max_steps_per_waypoint: 25
  transition_samples: 20
```

`rejoin_tolerance_m` 只控制中间局部轨迹结束后回到插入轴目标的判定。未设置时优先复用
`local_tracking.waypoint_tolerance_m`，再回退到外层 `task.waypoint_tolerance_m`。

### Observer 避碰

observer 相关参数位于 `engine_navigation.observer_control`：

```yaml
observer_control:
  position_gain: 3.0
  executor_offset_world_m: [0.0, -0.04, 0.02]
  roi_blend: 0.25
  inter_arm_influence_distance_m: 0.018
  inter_arm_safe_distance_m: 0.014
  inter_arm_critical_distance_m: 0.009
  inter_arm_release_margin_m: 0.002
  inter_arm_avoidance_gain: 6.0
  inter_arm_max_avoidance_speed_mps: 0.03
  centerline_samples_per_segment: 8
  observer_tracking_weight: 20.0
  observer_collision_weight: 250.0
  stop_all_on_critical_distance: false
```

距离满足 `critical < safe < influence`。默认 `stop_all_on_critical_distance: false`，避碰只改变
observer 驱动，不直接停止 executor 目标推进。

## MuJoCo 录屏与可视化

运行产物默认写入：

```text
output/runs/<scenario>_<timestamp>/
  result.npz
  metadata.json
  configs/
  model/
  plots/
  videos/simulation.gif
```

MuJoCo GIF 有两种模式：

```yaml
artifacts:
  save_gif: true
  video_mode: replay       # 仿真结束后用 qpos/qvel 离屏回放
  # video_mode: live_mujoco # 仿真过程中实时采集 MuJoCo 画面
  video_fps: 10
  video_stride: 10
```

相机来自 `configs/mujoco_dual.yaml` 的 `viewer.camera`。当前双臂默认使用基座跟随视角，
并把距离调到 `0.50`，比原 `0.25` 视野更大，画面主体约缩小一半：

```yaml
viewer:
  camera:
    lookat: [0.025, 0.0, 0.095]
    distance: 0.50
    azimuth: 315.0
    elevation: -25.0
    follow: base  # none, base 或 executor_tip
```

`follow: base` 同时作用于 MuJoCo viewer、`live_mujoco` GIF 和 replay 导出的
`simulation.gif`。如果录屏失败，仿真、NPZ、metadata 和 plots 仍会保存，错误写入
`metadata.json.errors` 或 `videos/video_error.txt`。

可视化叠加由 `viewer.overlays` 控制。engine navigation 默认显示底座规划路径、插入路径、
局部 executor 路径、当前目标、observer ROI、底座历史和 executor 历史。
`viewer.overlays.error_vector` 会在 MuJoCo viewer 中同步显示当前执行点到当前目标的误差向量；
tracking/wiping 使用 executor tip 到目标点，engine navigation 会根据当前目标类型在 base 或 executor
之间切换。场景 hook 还可以开启 `show_live_diagnostics_panel`，同步显示 tracking/base error、
clearance/contact/force error、奇异条件数、限速比例和 tendon target error，便于运行中定位误差来源。

## 调试工具

独立 MuJoCo 肌腱调试入口：

```powershell
python scripts/debug_mujoco.py configs/scenarios/dual_mujoco_tracking.yaml
python scripts/debug_mujoco.py configs/scenarios/single_mujoco_tracking.yaml
python scripts/debug_mujoco.py configs/scenarios/dual_mujoco_tracking.yaml --panel-only
```

常用 hook：

```yaml
hooks:
  recorder: true
  tendon_debug: true
  tendon_debug_stride: 5
  show_live_tendon_panel: true
  live_tendon_panel_stride: 5
  show_live_force_panel: true
  live_force_panel_stride: 5
  show_live_diagnostics_panel: true
  live_diagnostics_panel_stride: 5
  live_diagnostics_panel_history_points: 300
```

## 关键目录

```text
src/continuum_sim/application   scenario 解析与应用组装
src/continuum_sim/tasks         轨迹、mission、engine navigation 与清洗路径生成
src/continuum_sim/control       tracking、navigation、wiping、engine navigation 控制器
src/continuum_sim/runtime       backend-independent loop 与 hooks
src/continuum_sim/io            NPZ、plots、GIF、metadata 产物
configs/scenarios               推荐运行入口
configs/mujoco_dual.yaml        双臂 MuJoCo、viewer、overlay、录屏相机配置
```

## 手动验证建议

本项目默认不自动运行测试或仿真。修改后建议按需手动执行：

```powershell
pytest tests/test_robot_config.py tests/test_mujoco_video_export.py tests/test_scenario_artifacts.py
pytest tests/test_engine_navigation.py tests/test_staged_engine_navigation.py
python scripts/run_scenario.py configs/scenarios/dual_engine_navigation.yaml
```

重点检查：

- `videos/simulation.gif` 是否跟随基座且视野足够大。
- `engine_navigation_local_path_<name>.png` 中局部轨迹形状和误差是否符合预期。
- `metadata.json.errors` 是否为空或只包含可接受的视频降级信息。
- `result.npz` 中 `tracking_error_m`、`achieved_waypoint_error_m`、`base_position_m` 是否完整。

## 当前限制

- 高级 wiping / engine cleaning 已接入 scenario，但真实接触力闭环仍依赖 MuJoCo 接触模型和力反馈标定。
- `configs/tasks/*.yaml` 与旧 `runtime/mujoco_*_runtime.py` 不再是推荐主入口。
- 大半径椭圆、figure-eight 和 lissajous 局部轨迹可能需要降低速度、增大容差或增加样本数才能稳定跟踪。
