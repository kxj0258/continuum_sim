# 在线 waypoint 可达性评分

在线可达性评分用于 waypoint 伺服过程中判断当前目标点是否正在收敛、是否容易继续跟随，以及是否需要自动跳到下一个 waypoint。

## 分数定义

当前主线将“目标是否可继续推进”和“执行器是否跟上命令”分开记录：

```text
reachability_score = progress_component
                   * alignment_component
                   * model_component

execution_score = tendon_component

combined_score = reachability_score * execution_score
```

自动跳点只使用 `reachability_score`。这样可以避免把 MuJoCo tendon position actuator 的滞后误判为目标几何不可达。

## 分量含义

- `progress_component`：最近窗口内 tip error 是否下降。
- `alignment_component`：实际 tip 运动方向是否朝向当前目标。
- `model_component`：模型残差是否较小，反映 IK/雅可比/投影对当前命令的解释能力。
- `tendon_component`：实际肌腱速度与上一周期命令速度的比例，反映执行层是否跟得上。

## 自动跳点条件

控制器只有在满足以下条件时才会自动跳点：

- 在线评分启用。
- 自动跳点启用。
- 当前 waypoint 已经运行超过 `min_steps_before_auto_advance`。
- 最近连续 `low_score_patience_steps` 个周期的 `reachability_score` 低于阈值。
- 当前点尚未达到 waypoint 容差。

默认参数位于：

```text
configs/control/mujoco_tracking_low_level.yaml
```

单个场景可以覆盖：

```yaml
task:
  tracking_control:
    online_reachability:
      score_threshold: 0.3
      window_steps: 25
      low_score_patience_steps: 25
```

## 实时诊断

开启 live diagnostics：

```yaml
hooks:
  show_live_diagnostics_panel: true
  live_diagnostics_panel_stride: 5
  live_diagnostics_panel_history_points: 300
```

实时窗口会显示：

- 当前 `reachability_score`。
- `progress`、`alignment`、`model` 三项分量。
- `execution_score` 和 tendon 误差。
- 自动跳点事件、当前 waypoint、阶段和主要瓶颈。
- observer 模式、从臂误差和臂间距离等运行诊断。

如果同时开启：

```yaml
artifacts:
  save_plots: true
```

运行结束后会在当前 run 目录保存：

```text
plots/live_diagnostics_panel.png
```

该图保存的是窗口关闭前的最终显示内容；如果 `live_diagnostics_panel_history_points` 只保留最近 N 个点，落盘图也只包含这段历史。

## 读图建议

- `reachability_score` 低且 `execution_score` 也低：优先看执行器响应、tendon position actuator 和限幅。
- `reachability_score` 低但 `execution_score` 正常：优先看目标几何、方向对齐、模型残差和 waypoint 布局。
- `progress_component` 低：tip error 没有明显下降。
- `alignment_component` 低：运动方向偏离目标方向。
- `model_component` 低：IK/投影/模型残差偏大。
