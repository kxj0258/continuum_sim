# continuum_sim

## Bending-space 相容控制

正常控制任务统一在每臂 6 维 bending-space 中求解：

```text
b = [kx_1, ky_1, kx_2, ky_2, kx_3, ky_3]
q = S_b b
delta_l = C_b b,  C_b = C_q S_b
```

每段轴向应变固定为零，和 MuJoCo
`tendon_model.include_axial_strain: false` 一致。`tracking`、`navigation`、
`wiping`、`engine_cleaning`、双臂 observer 协同和避障任务均先求
`b_dot`，再由 `C_b` 一次性生成 9 根 tendon 的相容速度。

限速和目标位移限幅使用每臂统一缩放系数，不再逐根裁剪，因此不会破坏
tendon 比例。MuJoCo 实际绳长仍可能因弹性、动力学滞后和求解误差产生小量
不相容残差；控制器使用其 bending-space 投影作为状态，并在 metadata 中记录
原始 residual。

系统 tendon debug 默认使用 `compatible` 模式：逐根输入会投影到可实现的
bending 子空间。只有显式选择 `raw tendon` 才会保留独立 tendon 命令；该模式
专用于检查 routing、方向和 actuator force，不应作为任务控制策略。

建议手动验证：

```powershell
conda activate continuum_sim
pytest tests/test_bending_space.py tests/test_differential_ik.py tests/test_navigation_controller.py tests/test_hybrid_force_position.py tests/test_adaptive_impedance.py
pytest tests/test_system_tendon_debug.py tests/test_mujoco_system_debug_viewer.py
pytest tests/test_scenario_migrated_task_features.py tests/test_scenario_mujoco_composition.py
python -m compileall src scripts
```

本次实现过程未自动执行上述命令。

面向空间连续体机械臂的组合式仿真项目。当前主入口统一为：

```powershell
python scripts/run_scenario.py configs/scenarios/<scenario>.yaml
```

Python 代码中使用 `continuum_sim.application.SimulationApplication` 调用同一套 scenario 架构。

## 快速运行

在项目根目录执行。带 `viewer: mujoco` 或 `viewer: matplotlib` 的场景会打开可视化窗口，可能需要手动关闭窗口。

```powershell
# 最小 smoke
python scripts/run_scenario.py configs/scenarios/single_analytic_smoke.yaml
python scripts/run_scenario.py configs/scenarios/single_mujoco_smoke.yaml

# tracking: 支持手写 waypoints_world 或 task.trajectory 自动生成
python scripts/run_scenario.py configs/scenarios/single_analytic_tracking.yaml
python scripts/run_scenario.py configs/scenarios/dual_analytic_tracking.yaml
python scripts/run_scenario.py configs/scenarios/single_mujoco_tracking.yaml
python scripts/run_scenario.py configs/scenarios/dual_mujoco_tracking.yaml

# navigation: 支持 scene inspection target id mission
python scripts/run_scenario.py configs/scenarios/single_analytic_navigation.yaml
python scripts/run_scenario.py configs/scenarios/single_mujoco_navigation.yaml
python scripts/run_scenario.py configs/scenarios/dual_mujoco_navigation.yaml

# wiping: 支持 structured scene raster wiping_path
python scripts/run_scenario.py configs/scenarios/single_analytic_wiping.yaml
python scripts/run_scenario.py configs/scenarios/single_mujoco_wiping.yaml
python scripts/run_scenario.py configs/scenarios/dual_mujoco_wiping.yaml

# engine scene / engine cleaning
python scripts/run_scenario.py configs/scenarios/single_engine_tracking.yaml
python scripts/run_scenario.py configs/scenarios/dual_engine_tracking.yaml
python scripts/run_scenario.py configs/scenarios/single_engine_cleaning.yaml
```

Python API:

```python
from continuum_sim.application import SimulationApplication

application = SimulationApplication.from_yaml(
    "configs/scenarios/dual_mujoco_tracking.yaml"
)
result = application.run()
print(len(result.states))
print(application.last_artifacts.run_dir)
```

## 新版 Scenario 能力

`configs/scenarios/*.yaml` 是推荐配置入口。旧 `configs/tasks/*.yaml` 和旧 `runtime/mujoco_*_runtime.py` 仍保留给历史测试和参考，但不再作为主运行路径。

### Tracking 轨迹生成

`task` 支持两种目标来源，二选一：

```yaml
task:
  type: tracking
  waypoints_world:
    - [0.0, 0.0, 0.14]
    - [0.01, 0.0, 0.14]
```

或：

```yaml
task:
  type: tracking
  trajectory:
    type: square        # circle, figure-eight, ellipse, line, square, lissajous, helix, dmp
    samples: 80
    radius_m: 0.018
    placement:
      center_mode: straight_tip_xy
      z_mode: straight_tip_minus_radius
      plane: xy
      yaw_deg: 15.0
    shape:
      side_length_m: 0.04
```

目标点更新策略：

```yaml
task:
  target_advance_mode: tolerance  # tolerance 或 time
  waypoint_tolerance_m: 0.002
```

按时间/步数更新：

```yaml
task:
  target_advance_mode: time
  advance_steps: 40
  # 或 advance_time_s: 0.8
```

### Navigation Mission

navigation 可以继续手写 `waypoints_world`，也可以从 structured scene 的 inspection target id 生成：

```yaml
scene:
  structured_config_path: ../scenes/rocket_nozzle_entry.yaml
task:
  type: navigation
  mission:
    waypoint_ids:
      - entry_wall_30deg
      - rib_gap_center
      - throat_wall_210deg
```

### Wiping Path 与高级控制字段

wiping 可以用 structured scene 中的 work surface / patch 自动生成 raster path：

```yaml
scene:
  structured_config_path: ../scenes/wiping_board.yaml
task:
  type: wiping
  wiping_control_type: dynamic_adaptive_impedance
  feedback_mode: mujoco_actual
  target_normal_force_n: 1.5
  normal_force_gain: 0.075
  force_proxy_stiffness_n_m: 600.0
  target_contact_distance_m: -0.0025
  wiping_path:
    surface_id: board_surface
    patch_id: center_patch
    line_count: 5
    samples_per_line: 30
    approach_offset_m: 0.005
    contact_offset_m: -0.0025
```

当前新版 scenario 控制器仍输出统一的 `RobotSystemCommand`，高级力控字段会进入 task metadata、记录和 artifacts；真实法向力闭环仍依赖后续 MuJoCo 接触/力反馈标定。

### Engine Cleaning Path

engine cleaning 现在作为 scenario task type 接入：

```yaml
scene:
  engine_config_path: ../scenes/engine_cleaning.yaml
task:
  type: engine_cleaning
  wiping_control_type: hybrid_force_position
  engine_cleaning:
    region_name: cleaning_patch
    num_passes_u: 4
    num_passes_v: 3
    approach_distance_m: 0.015
    retreat_distance_m: 0.020
    target_force_n: 1.2
    standoff_distance_m: 0.005
```

### Live Debug Panel

富肌腱监控面板已经迁入 scenario hook 体系，不再依赖旧 runtime loop。开启后会按
`executor:1`、`observer:1` 这样的命名显示每根肌腱：

- 目标位移与 MuJoCo 当前位移，单位为 mm；
- 目标误差，单位为 mm；
- MuJoCo `actuator_force`，单位为 N；
- 当前时间以及速率/位移饱和数量。

这里的长度是相对于 reset 中性长度的变化量，与 direct tendon-rate 控制器的目标
定义一致。单臂和双臂场景会根据 assembly 自动生成 9 根或 18 根肌腱标签。

```yaml
hooks:
  show_live_tendon_panel: true
  live_tendon_panel_stride: 5
  show_live_force_panel: true
  live_force_panel_stride: 5
  live_force_panel_history_points: 300
```

交互式 MuJoCo 场景默认打开肌腱面板；批处理时可将
`show_live_tendon_panel` 改为 `false`。关闭肌腱面板不会终止场景任务。

### 独立 MuJoCo 肌腱调试

新的调试入口直接读取 scenario YAML，因此使用的 assembly、生成 XML、单/双臂
布局和普通任务完全一致：

```powershell
# 双臂调试：MuJoCo 3D viewer + 肌腱控制/监控面板
python scripts/debug_mujoco.py configs/scenarios/dual_mujoco_tracking.yaml

# 单臂调试
python scripts/debug_mujoco.py configs/scenarios/single_mujoco_tracking.yaml

# 只打开 Matplotlib 控制/监控面板
python scripts/debug_mujoco.py configs/scenarios/dual_mujoco_tracking.yaml --panel-only
```

调试器为每根肌腱提供同步的目标位移滑块和数值输入框，二者统一使用 mm。拖动
滑块会刷新输入框；在输入框按 Enter 会更新并裁剪对应滑块，但不会自动推进
仿真，仍需使用 Step 或 Run。Reset、Zero 和预设命令也会同步刷新两种控件。

界面还提供单根肌腱、第一段三根肌腱和全部肌腱的预设命令。UI 在边界处把 mm
转换为 m，再生成受 assembly `max_tendon_rate_mps` 限制的
`RobotSystemCommand`，不会改变后端的 SI 单位或绕开新的系统控制边界。

## 运行产物

非 idle 场景默认保存到：

```text
output/runs/<scenario>_<timestamp>/
  result.npz
  metadata.json
  configs/
  model/
  plots/
  videos/
```

常见 `result.npz` 字段：

```text
time_s
base_position_m
base_quat_wxyz
arm_executor_tip_position_m
arm_executor_tendon_displacement_m
arm_executor_command_rate_mps
target_position_m
tracking_error_m
waypoint_index
min_clearance_m
contact_distance_m
contact_error_m
target_force_n
task_phase
qpos / qvel
```

GIF 导出失败不会丢弃 NPZ、metadata 和 plots；错误会写入 `metadata.json.errors`。

MuJoCo 场景视频支持两种模式：

```yaml
artifacts:
  save_gif: true
  video_mode: replay       # 默认：仿真结束后用 qpos/qvel 离屏重放
  # video_mode: live_mujoco # 运行过程中实时采集 MuJoCo 场景帧
  video_fps: 10
  video_stride: 10
```

`live_mujoco` 会在仿真循环中写入真实 MuJoCo 场景画面，并叠加与 viewer 一致的目标点、目标轨迹和 executor 实际轨迹；结束后把临时 GIF 移入本次 `output/runs/.../videos/`。如果本机 MuJoCo `Renderer` / OpenGL 上下文初始化失败，仿真和 NPZ、metadata、plots 仍会继续保存，视频错误会记录到 `metadata.json.errors` 和 `videos/video_error.txt`。

### MuJoCo overlay 标记

`configs/mujoco_dual.yaml` 的 `viewer.overlays.segment_endpoints` 用于在运行时叠加每个 segment 末端标记；默认 executor 为红色，observer 为黄色。该标记不修改 MuJoCo XML，也不影响 tendon 走线或控制逻辑；使用 `artifacts.video_mode: live_mujoco` 时会一并录入 `simulation.gif`。

### Dual tracking 控制逻辑

双臂 tracking 采用 executor-primary 控制：executor 主臂只执行目标轨迹追踪主任务；observer 从臂的避碰和观测任务只作用在 observer 自身 tendon 上，不能反向拉动 executor 或共享 base。observer 与 executor 距离进入避碰影响范围时，observer 优先执行双臂避碰；距离安全后再执行相对主臂/ROI 的观测任务。

当 assembly 的 `base.control_mode: fixed` 时，控制布局会移除 base DOF，whole-body Jacobian 只包含各臂 tendon-rate 变量；此时控制器按纯肌腱驱动求解，而不是先求 base 速度再清零。

## 架构概览

```text
ScenarioConfig
  -> RobotAssemblyConfig
  -> task builders: trajectory / mission / wiping_path / engine_cleaning
  -> controller: tracking / navigation / wiping / engine_cleaning
  -> backend: AnalyticSystemBackend / MujocoSystemBackend
  -> SimulationLoop
  -> hooks + ScenarioArtifactWriter
```

关键目录：

- `src/continuum_sim/application`: scenario 解析和组合根。
- `src/continuum_sim/tasks`: 新版任务目标/路径生成，以及保留的旧 task loader。
- `src/continuum_sim/control`: whole-body controller、scenario controllers、waypoint scheduler。
- `src/continuum_sim/runtime`: backend-independent loop 和 hooks。
- `src/continuum_sim/io`: scenario artifacts。
- `configs/scenarios`: 推荐运行配置。
- `configs/tasks`: 历史任务配置，保留作兼容参考。

## 手动验证建议

本次不自动运行测试或仿真。建议你按需要手动执行：

```powershell
python -m compileall src scripts/run_scenario.py
pytest tests/test_system_tendon_debug.py tests/test_mujoco_system_debug_viewer.py
pytest tests/test_scenario_migrated_task_features.py
pytest tests/test_scenario_import_boundaries.py tests/test_scenario_mujoco_composition.py tests/test_scenario_artifacts.py

python scripts/run_scenario.py configs/scenarios/single_analytic_tracking.yaml
python scripts/run_scenario.py configs/scenarios/single_mujoco_tracking.yaml
python scripts/run_scenario.py configs/scenarios/dual_mujoco_tracking.yaml
python scripts/debug_mujoco.py configs/scenarios/dual_mujoco_tracking.yaml
python scripts/run_scenario.py configs/scenarios/single_mujoco_navigation.yaml
python scripts/run_scenario.py configs/scenarios/single_mujoco_wiping.yaml
python scripts/run_scenario.py configs/scenarios/single_engine_cleaning.yaml
```

重点人工检查：

- 自动生成轨迹是否符合预期形状和位置。
- `target_advance_mode: tolerance/time` 是否按预期推进目标点。
- wiping/engine cleaning 的 `task_phase`、`target_force_n`、`contact_distance_m` 是否写入产物。
- live tendon/force panel 是否只在 hook 开关启用时出现。
- MuJoCo 场景中的目标点、executor、observer 和 engine/structured scene 坐标是否合理。

## 当前限制

- 新版高级 wiping/engine cleaning 已迁入 scenario 架构，但真实接触力闭环仍依赖 MuJoCo 接触模型和力反馈标定。
- `feedback_mode` 目前作为 scenario 任务语义和产物字段保留，控制主链仍是统一的 direct tendon-rate `RobotSystemCommand`。
- 旧 `configs/tasks/*.yaml` 和旧 `runtime/mujoco_*_runtime.py` 不再是推荐主入口。
