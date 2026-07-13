# continuum_sim 交接文档

本文写给完全没有本轮对话上下文的新会话。请先读完再继续改代码。

## 1. 当前在做什么

当前主线任务是把 `continuum_sim` 的 MuJoCo 连续体机械臂任务流程整理到更稳定的控制和可视化路径上，重点包括：

- tracking / navigation / wiping / cleaning 等任务统一使用更贴近 MuJoCo 实际肌腱长度的 `actual_anchored` 控制思路。
- 控制器使用解析雅可比和严格 SVD 方向投影，尽量避免数值雅可比、阻尼缩放和限幅策略带来的额外误差。
- 将硬编码控制参数改成 YAML 可配置。
- wiping 任务从逐点跳转改成时间轨迹，并在运行时根据当前真实末端位置生成平滑 approach 段。
- MuJoCo viewer 运行结束后自动保存结果并关闭窗口。
- 所有打开 MuJoCo 画面的录制任务尽量走 `live_mujoco` 视频路径，避免 replay renderer 在当前 Windows/OpenGL 环境下崩溃。

非常重要：用户明确要求不要自动运行测试、验证、lint、format、build、install、仿真命令。后续会话除非用户明确说“运行测试”或“运行验证”，否则只做代码/文档修改，最后给出建议手动运行的命令。

## 2. 已经完成了什么

### 2.1 场景配置加载修复

此前 `load_scenario_config()` 有几处迁移残留问题，用户运行时报过：

- `SyntaxError: keyword argument repeated: dynamics_config_path`
- `NameError: _load_contact_admittance_config is not defined`
- `NameError: navigation_control_type is not defined`

已经修复方向：

- 在 `src/continuum_sim/application/scenario.py` 中补齐 navigation 控制类型解析。
- 对齐 contact/admittance 配置加载函数名。
- 避免重复传入 `dynamics_config_path`。
- `src/continuum_sim/application/application.py` 已把 navigation 相关参数传入对应 controller。

### 2.2 WipingController 接口修复

此前运行 wiping / engine_cleaning 报错：

```text
TypeError: WipingController.__init__() got an unexpected keyword argument 'force_strategy'
```

已经修复方向：

- `src/continuum_sim/control/scenario_controllers.py`
  - `WipingController` 增加 `force_strategy`、tracking 增益、速度上限、solver 配置、tendon limit 开关、dynamics/admittance 配置等入参。
  - `compute_command()` 中通过 `WipingForceContext` 调用 force strategy。
- `src/continuum_sim/application/application.py`
  - 拆分 `wiping` 和 `engine_cleaning` 分支。
  - `wiping` 使用 `WipingController`。
  - `engine_cleaning` 使用 `EngineCleaningSystemController`。

### 2.3 移动基座漂移处理

用户发现 `dual_engine_navigation` 局部轨迹跟踪阶段，肌腱反作用力会让移动基座偏转。

当前处理：

- 在 `src/continuum_sim/backends/mujoco_backend.py` 中，当上层传入 `base_pose_rpy` 时，在每个 MuJoCo substep 前后调用 `set_mobile_base_pose_rpy(base_pose_rpy)`。
- 这会把 prescribed pose 阶段的移动基座钉在控制器给定姿态，避免 freejoint 在仿真子步里被 tendon/contact reaction 推偏。

风险：

- 这是一种“运动学锁定”方案，会隐藏真实移动基座动力学响应。
- 如果以后要研究真实移动底盘质量、轮系、接触反力和地面约束，就应该改成可配置开关，而不是永久强制锁死。

### 2.4 实时诊断可视化

已经在 `src/continuum_sim/runtime/hooks.py` 增加实时 tip 跟踪误差显示：

- 读取 metadata 中的 `executor_target_world` 或 `engine_navigation_active_target_m`。
- 显示 tip target error norm。
- 显示 x/y/z 三方向误差。
- 在诊断面板文本里显示 `tip_err_xyz_m`。

同时修复了 condition 数据全非正时 `semilogy` 触发的 Matplotlib warning：

```text
UserWarning: Data has no positive values, and therefore cannot be log-scaled.
```

当前逻辑是在没有正数 condition 时退回普通线性 plot。

### 2.5 wiping 接触点和轨迹配置调整

之前黑板表面和擦拭点存在几何不一致：

- 黑板表面大致在 `x = 0.0475`。
- normal 是 `[-1, 0, 0]`。
- 旧的 `contact_offset_m: -0.0025` 会把目标点推到黑板背面，导致 approach 点或 contact 点穿透黑板。

已经调整：

- `configs/scenarios/single_mujoco_wiping.yaml`
- `configs/scenarios/single_mujoco_wiping_admittance.yaml`
- `configs/scenarios/dual_mujoco_wiping.yaml`

建议值：

```yaml
target_contact_distance_m: 0.0
contact_offset_m: 0.0
approach_offset_m: 0.005
```

并启用：

```yaml
enforce_target_speed_limit: true
max_target_speed_mps: 0.015
```

注意：如果只写 `max_target_speed_mps` 但没有启用 `enforce_target_speed_limit`，速度上限不会实际生效。

### 2.6 engine navigation local_tracking 参数 YAML 化

用户要求把这些硬编码参数改成 YAML 可配置：

- `executor_position_gain`
- `max_target_speed_mps`
- `enforce_tendon_rate_limits`

已经完成方向：

- `src/continuum_sim/tasks/engine_navigation.py`
  - `EngineNavigationLocalTrackingSpec` 增加上述字段。
  - `_load_local_tracking()` 解析并校验这些字段。
- `src/continuum_sim/control/staged_engine_navigation.py`
  - `_make_tracker()` 不再硬编码 `3.0`、`None`、`False`。
  - 改从 `local_tracking` 配置读取。
- `configs/scenarios/dual_engine_navigation.yaml`
  - 已改为更慢的 time advance 模式：

```yaml
local_tracking:
  advance_mode: time
  advance_time_s: 0.20
  executor_position_gain: 1.5
  max_target_speed_mps: 0.015
  enforce_tendon_rate_limits: false
```

### 2.7 wiping 改成时间轨迹和运行时平滑 approach

用户要求：

- 运行时生成 approach smooth segment。
- wiping 改成时间轨迹。
- 如果估算接触力超过 `max_contact_force_n`，先不要提前结束仿真。

已经完成方向：

- `src/continuum_sim/control/scenario_controllers.py`
  - `WipingController` 增加：
    - `tracking_mode`
    - `trajectory_duration_s`
    - `approach_samples`
  - 支持 `tracking_mode: waypoint` 和 `tracking_mode: time`。
  - 新增运行时 approach 构造：
    - 第一次 `compute_command()` 时读取当前真实 executor tip 位置作为起点。
    - 原始第一个 waypoint 作为 approach 终点。
    - 用 smoothstep `alpha = 3*s**2 - 2*s**3` 生成平滑 approach samples。
    - 用 approach samples 加原始擦拭 waypoint 重建内部 tracker。
  - `done` 不再因为 `force_limit_exceeded` 提前结束，只由 tracking 是否完成决定。
  - `force_limit_exceeded` 仍写入 metadata 供诊断。
- `src/continuum_sim/application/application.py`
  - 向 `WipingController` 传入新的 tracking 配置。
- wiping 相关 YAML 已加入：

```yaml
approach_samples: 30
tracking_mode: time
trajectory_duration_s: 35.0
```

`dual_mujoco_wiping.yaml` 的 duration 当前较短，约为 `20.0`。

风险：

- time mode 下，现有 force strategy 仍更偏 waypoint 语义，接触力修正不一定完全跟随连续插值点。
- `contact_triggered_admittance` 原有“到达某 waypoint 后再 advance”的语义在 time trajectory 模式下会被弱化。

### 2.8 MuJoCo viewer 自动关闭和 live_mujoco 视频路径

用户希望所有打开 MuJoCo 的运行命令在结束后自动保存结果并关闭窗口。

已经处理方向：

- 对 MuJoCo viewer 场景设置：

```yaml
viewer:
  keep_viewer_open: false
```

用户还遇到 video export 错误：

```text
video export failed during creating MuJoCo renderer:
OSError: exception: access violation reading 0x0000000000031E78
fallback_saved: matplotlib trajectory animation ...
```

分析结论：

- `dual_engine_navigation` 能正常保存 MuJoCo GIF，是因为它显式配置了 `artifacts.video_mode: live_mujoco`。
- wiping 场景之前没有显式 artifacts，默认走 `replay` 路径。
- replay 路径会重新创建 `mujoco.Renderer`，在当前 Windows/OpenGL/GLFW 环境下容易 access violation。
- `Exception ignored in: GLContext.__del__` 多半是 MuJoCo/GLFW 第三方析构噪声，根因仍是 offscreen replay renderer 创建失败。

已经把打开 MuJoCo 且需要录制的任务统一改成：

```yaml
artifacts:
  enabled: true
  save_gif: true
  video_mode: live_mujoco
  video_fps: 10
  video_stride: 10
```

涉及场景至少包括：

- `configs/scenarios/single_mujoco_wiping.yaml`
- `configs/scenarios/single_mujoco_wiping_admittance.yaml`
- `configs/scenarios/dual_mujoco_wiping.yaml`
- `configs/scenarios/single_mujoco_tracking.yaml`
- `configs/scenarios/dual_mujoco_tracking.yaml`
- `configs/scenarios/single_mujoco_navigation.yaml`
- `configs/scenarios/dual_mujoco_navigation.yaml`
- `configs/scenarios/dual_engine_navigation.yaml`
- `configs/scenarios/dual_engine_tracking.yaml`
- `configs/scenarios/single_engine_cleaning.yaml`
- `configs/scenarios/single_engine_tracking.yaml`

`single_mujoco_view.yaml` 是 idle preview，当前更适合：

```yaml
artifacts:
  enabled: false
  save_gif: false
  video_mode: live_mujoco
```

## 3. 当前卡在哪里

当前最后一个用户请求是写交接文档，因此此处只做文档交接，没有继续验证。

潜在未确认状态：

- 所有修改都还没有经过自动测试或仿真验证，因为用户明确禁止自动运行。
- wiping 的 `live_mujoco` 录制路径是否彻底绕开 replay renderer，需要用户手动运行确认。
- `WipingController` time mode 和 admittance/force strategy 的语义是否完全符合预期，还需要结合运行轨迹观察。
- 移动基座每个 substep 强制回写 pose 是否会影响其他任务中的真实动力学表现，需要用户根据目标决定是否保留为默认行为。

## 4. 下一步计划

建议后续会话按这个顺序推进：

1. 先静态检查最近改过的配置文件，确认所有 MuJoCo viewer 场景都有 `keep_viewer_open: false`，有录制需求的都有 `artifacts.video_mode: live_mujoco`。
2. 由用户手动运行 wiping 三个主命令，确认窗口能自动关闭并保存 GIF。
3. 如果 wiping 仍提前结束，优先检查 run 目录中的 metadata、summary 和 `force_limit_exceeded`，不要先怀疑 OpenGL。
4. 如果 `live_mujoco` 仍无 GIF，检查 runtime hooks 中 live frame capture 是否只在 viewer loop 下采样，避免再次走 replay renderer。
5. 单独整理一个可配置项：是否把 mobile base 在 prescribed pose 阶段强制锁定。建议命名为 `lock_prescribed_base_pose` 或类似字段。
6. 后续再考虑把 wiping force/admittance 从 waypoint 语义改成真正的连续时间轨迹语义。

## 5. 绝不能再踩的坑

- 不要自动运行仿真、测试、lint、format、build、install 命令。用户已经多次强调。
- 不要再让 MuJoCo 录制任务默认走 replay renderer。当前 Windows/OpenGL 环境下 replay 创建 `mujoco.Renderer` 可能 access violation。
- 不要把 `GLContext.__del__` 的 `_context` AttributeError 当成主要控制器错误。它通常是第三方 GLContext 初始化失败后的析构噪声。
- 不要只设置 `max_target_speed_mps` 就以为速度限幅生效；对应的 `enforce_target_speed_limit` 也要打开。
- 不要让 wiping 的 `contact_offset_m` 为负导致目标点穿透黑板背面。当前更合理默认是 `0.0`。
- 不要把移动基座漂移简单解释成 controller 输出了非零 twist。即使 controller 输出零 twist，MuJoCo freejoint 也可能在子步里被 tendon/contact reaction 推动。
- 不要在不说明风险的情况下把 base 永久固定。固定基座适合稳定局部轨迹，但会牺牲真实动力学。
- 不要再把 `wiping` 和 `engine_cleaning` 都硬塞进 `WipingController` 分支；两者当前控制器不同。
- 不要提交或推送，除非用户明确要求。当前用户最近要求过“修复后不用自动提交和推送”，本次只要求写文档。

## 6. 建议用户手动验证的命令

后续应由用户手动运行，不要由助手自动运行：

```powershell
python scripts/run_scenario.py configs/scenarios/single_mujoco_wiping.yaml
python scripts/run_scenario.py configs/scenarios/single_mujoco_wiping_admittance.yaml
python scripts/run_scenario.py configs/scenarios/dual_mujoco_wiping.yaml
python scripts/run_scenario.py configs/scenarios/single_mujoco_tracking.yaml
python scripts/run_scenario.py configs/scenarios/single_mujoco_navigation.yaml
python scripts/run_scenario.py configs/scenarios/dual_engine_navigation.yaml
python scripts/run_scenario.py configs/scenarios/single_engine_cleaning.yaml
```

如果视频仍失败，优先查看：

```text
output/runs/<run_name>/videos/video_error.txt
output/runs/<run_name>/metadata.json
output/runs/<run_name>/summary.json
```

重点确认：

- 是否还出现 `video_mode: replay` 相关路径。
- 是否生成 MuJoCo live GIF。
- viewer 是否在任务结束后自动关闭。
- `force_limit_exceeded` 是否只记录，不再触发提前结束。
- base pose 是否仍在局部 tracking 阶段明显偏转。

## 7. 重要文件清单

核心代码：

- `src/continuum_sim/application/scenario.py`
- `src/continuum_sim/application/application.py`
- `src/continuum_sim/control/scenario_controllers.py`
- `src/continuum_sim/control/staged_engine_navigation.py`
- `src/continuum_sim/tasks/engine_navigation.py`
- `src/continuum_sim/backends/mujoco_backend.py`
- `src/continuum_sim/runtime/hooks.py`

核心配置：

- `configs/scenarios/single_mujoco_wiping.yaml`
- `configs/scenarios/single_mujoco_wiping_admittance.yaml`
- `configs/scenarios/dual_mujoco_wiping.yaml`
- `configs/scenarios/single_mujoco_tracking.yaml`
- `configs/scenarios/dual_mujoco_tracking.yaml`
- `configs/scenarios/single_mujoco_navigation.yaml`
- `configs/scenarios/dual_mujoco_navigation.yaml`
- `configs/scenarios/dual_engine_navigation.yaml`
- `configs/scenarios/dual_engine_tracking.yaml`
- `configs/scenarios/single_engine_cleaning.yaml`
- `configs/scenarios/single_engine_tracking.yaml`
- `configs/scenarios/single_mujoco_view.yaml`

参考输出：

- `output/runs/*/videos/video_error.txt`
- `output/runs/*/videos/simulation.gif`
- `output/runs/*/metadata.json`
- `output/runs/*/summary.json`

## 8. 当前交接结论

本轮主要目标已经从“控制逻辑修正”推进到“MuJoCo live 录制和自动关闭路径统一”。最后一步尚未由用户手动验证。新会话接手时，建议先不要继续大改控制器，而是围绕 wiping / navigation 的手动运行结果做小步修复。

本文件创建时未运行任何测试、验证、仿真、lint、format、build 或 install 命令。
