# 在线 waypoint 可达性评分

在线评分用于 waypoint 伺服过程中判断当前目标点是否容易跟随、是否正在收敛，
以及是否需要自动跳到下一个 waypoint。

## 分数定义

评分拆成两类：

```text
reachability_score = progress_component
                   * alignment_component
                   * model_component

execution_score = tendon_component

combined_score = reachability_score * execution_score
```

自动跳点只使用 `reachability_score`。这样可以把“几何/控制上难以到达”和
“MuJoCo position actuator 跟得慢”分开分析。

## 各项指标

- `progress_component`：最近窗口内 tip error 是否在下降。下降越快，分数越高。
- `alignment_component`：实际 tip 运动方向是否指向当前目标点。侧向运动或反向运动会降低分数。
- `model_component`：底层模型残差是否较小。残差大说明当前模型/雅可比/投影无法很好解释命令。
- `tendon_component`：实际肌腱速度与上一周期命令速度的比例。它反映执行器是否跟得上命令。

## 自动跳点条件

控制器只有在满足以下条件时才会自动跳点：

- 在线评分启用。
- 自动跳点启用。
- 当前 waypoint 已经运行超过 `min_steps_before_auto_advance`。
- 最近连续 `low_score_patience_steps` 个周期的 `reachability_score` 低于阈值。
- 当前点还没有达到 waypoint 容差。

共享默认参数在：

```text
configs/control/mujoco_tracking_low_level.yaml
```

可以在单个场景中覆盖：

```yaml
task:
  tracking_control:
    online_reachability:
      score_threshold: 0.3
      window_steps: 25
      low_score_patience_steps: 25
```

## 可视化

实时诊断窗口会显示：

- 当前 `reachability_score`。
- `progress`、`alignment`、`model` 三项分量。
- `execution_score`，用于观察 MuJoCo tendon actuator 是否跟上命令。
- 自动跳点事件、当前 waypoint、阶段和主要瓶颈。

如果某个点 `reachability_score` 低但 `execution_score` 也低，通常说明执行器响应慢；
如果 `reachability_score` 低而 `execution_score` 正常，则更像是目标点几何可达性、
方向对齐或模型映射问题。
