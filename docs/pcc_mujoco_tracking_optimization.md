# PCC–MuJoCo tracking 优化基线

## 目标与物理边界

本轮优化把真实硬件的 tendon 位移、速度和力响应作为最终物理权威。在取得硬件辨识数据前，
MuJoCo 的关节阻尼、armature、tendon actuator `kp`、`forcerange`、质量和刚度保持不变；不能为了
降低仿真 tracking error 而把 plant 人为调快。

生产控制仍以可迁移的 PCC/标定 PCC 为主。MuJoCo-native Jacobian 只能作为仿真性能上界和问题
隔离工具，不能直接视为可迁移到真实机器人的解决方案。

## 现有运行证据

代表性结果表明模型误差与执行误差是两条独立链路：

| run | tracking mean/max | PCC–MuJoCo position mean/max | command–MuJoCo velocity mean |
|---|---:|---:|---:|
| `20260714_182057` | `7.518 mm` / — | `13.285/19.741 mm` | `7.737 mm/s` |
| `20260715_111107`，gain=1 | `8.099/15.831 mm` | `4.815/7.059 mm` | `8.321 mm/s` |
| `20260715_121229`，gain=2 | `7.568/15.96 mm` | `4.829/6.890 mm` | `15.311 mm/s` |
| `20260715_122120`，gain=5 | `6.755/15.125 mm` | `4.873/7.083 mm` | `33.862 mm/s` |

最新 run 中，PCC Jacobian 线性化残差均值约 `0.00683 mm/s`，同一实际 tendon 增量下的
PCC–MuJoCo 模型速度残差约 `0.684 mm/s`，而 command–MuJoCo 速度残差约 `33.862 mm/s`。
因此当前首要问题位于 command → tendon target → 实际 tendon 运动链，约 `5 mm` 的绝对位置偏差
是需要单独标定的第二问题。

gain 从 1 提高到 5 时，平均 tracking error 仅改善约 `16.6%`，command velocity residual 却增至
约 `4.07` 倍。继续增加外环增益不是有效方向。

## 当前控制基线

### 参考轨迹

- approach 使用首帧实测 executor tip 作为起点，不再假设解析直臂 tip 与 MuJoCo 完全一致。
- approach 使用独立 `20 s`；`trajectory_duration_s: 80.0` 只表示正式方形路径时间。
- 方形显式闭合，按弧长分配时间。
- `corner_stop_hermite` 保持精确直边和角点，并令 90° 角点速度为零。
- reference governor 根据实际 tracking error 和 executor tendon lead 利用率推进虚拟时间；落后时参考
  会减速或暂停，恢复时按有限斜率加速。
- governor 启用时，完成条件是参考到达终点且末端进入 `waypoint_tolerance_m`，不是墙钟到点即结束。

### 底层控制

`configs/control/mujoco_tracking_low_level.yaml` 当前基线为：

```yaml
arm_position_gain: 1.0
feedforward_gain: 1.0
max_target_speed_mps: 0.006
enforce_target_speed_limit: true
enforce_solver_velocity_limits: true
enforce_backend_tendon_limits: true
backend_tendon_target_mode: protected
```

`protected` 使用有界持久 target：未完成的 target 位移不会像 `actual_anchored` 一样每拍丢弃，但仍受
tendon rate、绝对位移和 target lead 限制。executor/observer 的 target lead 均为 `0.25 mm`；按当前
`kp=100000 N/m` 估算，单 tendon 的位置误差力预算约为 `25 N`，低于 `±30 N` actuator force range。

该 profile 只用于 MuJoCo/engine tracking 和匹配的 analytic A/B 场景。navigation 与 engine-cleaning 的
直接速度尺度不同，已隔离到 `configs/control/mujoco_direct_task_low_level.yaml`，避免本轮 tracking
限速改变其他任务行为。

该 `25 N` 只是 position actuator 的静态上界估算，不等同于真实电机张力模型。取得硬件数据后必须
重新校准 lead、`kp` 和 force range，不能把当前值永久当作硬件真值。

### 速率诊断语义

必须分别观察以下信号：

- `command_rate_mps`：控制器请求；
- `constrained_command_rate_mps`：solver/backend 约束后的命令，旧 `applied_rate_mps` 仅为兼容别名；
- `tendon_target_rate_fd_mps`：相邻状态 target 的有限差分；
- `tendon_realized_rate_fd_mps`：相邻状态实际 tendon displacement 的有限差分；
- `tendon_velocity_sensor_raw_mps`：MuJoCo/backend 的瞬时原始 sensor。

不能再用 `applied_rate_mps` 或周期末瞬时 sensor 代替实际周期平均速度。每次运行还会记录 Git commit、
dirty 状态、输入文件清单和 SHA-256，避免相同 YAML 快照对应不同源码却被当成可比实验。

## PCC 标定阶段

当前不写入任何猜测补偿系数。标定应使用慢速、准静态、覆盖工作区的数据，并执行以下顺序：

1. 确认零输入 neutral/mount bias；
2. 检查每段主要弯曲方向的 tendon-to-bending scale、有效 routing radius 和交叉耦合；
3. 使用 hold-out 状态拟合 `p_mujoco - p_pcc` 的低阶、可微 residual；
4. 控制器同时使用 residual position 与其解析导数，不能只平移 FK 而继续使用未修正 Jacobian；
5. calibration 文件必须记录训练工作区、输入文件 SHA-256、模型/assembly hash 和版本；
6. 若低阶标定后速度残差仍明显，再升级为与 12 个离散 flexure link 一致的 pseudo-rigid-body 模型。

只有同时满足以下门槛，标定结果才允许进入控制器：

- 独立 hold-out 的位置残差相对未标定 PCC 明显下降；
- hold-out 的 Jacobian/速度残差没有恶化；
- 工作区外有显式拒绝或退回未标定 PCC 的策略；
- 真实硬件与 MuJoCo 的同方向准静态响应一致。

`configs/scenarios/single_analytic_tracking_matched.yaml` 与
`configs/scenarios/single_mujoco_tracking.yaml` 使用相同 assembly、低层 profile、轨迹几何、参考调度和
控制周期，用于建立严格 A/B。不能继续拿旧的 20 mm、waypoint-mode analytic 场景与 40 mm、time-mode
MuJoCo 场景直接比较 tracking error。

## 分阶段验收

1. **80 s 可实现性基线**：先确认 reference scale 大部分时间接近 1，角点仅短时降速，force 不持续
   接近上限，target FD 与 realized FD 同方向且幅值接近。
2. **命令链验收**：比较 requested、constrained、target FD、realized FD；若差异发生在 solver/backend，
   调约束和 governor；若 target FD 已正确但 realized FD 落后，再辨识 plant 带宽。
3. **PCC 准静态标定**：只使用低速数据评价绝对 position/Jacobian residual，不混入 tracking controller
   的动态误差。
4. **30 s 性能目标**：80 s 通过后再逐步缩短 path duration，并记录每一级的 tracking、lead、force、
   realized-rate 和 reference-scale 指标；不能一步回到 30 s 后再提高 gain。
5. **硬件迁移**：用相同 command/target/realized 字段记录真实执行器响应，再据此校准 MuJoCo 动态参数。

## 尚未自动验证

本轮没有运行测试、lint、format、build、安装、MuJoCo viewer 或仿真。所有验收命令均应由用户手动执行。
