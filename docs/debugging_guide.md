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
pytest tests/test_robot_config.py tests/test_scenario_artifacts.py
pytest tests/test_engine_navigation.py tests/test_staged_engine_navigation.py
python scripts/run_scenario.py configs/scenarios/single_analytic_smoke.yaml
python scripts/run_scenario.py configs/scenarios/single_mujoco_smoke.yaml
python scripts/run_scenario.py configs/scenarios/dual_engine_navigation.yaml
python scripts/check_mujoco_offscreen_renderer.py configs/mujoco_dual.yaml
```

如果目标是检查 viewer 或实时面板，建议先复制一个场景配置，把 `hooks.viewer` 改成 `mujoco`，
并把 `runtime.max_steps` 降低到较小值。
