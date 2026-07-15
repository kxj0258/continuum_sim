# 配置参考

除非字段另有说明，所有数值都使用 SI 单位。路径通常相对声明它的 YAML 文件解析；在适用时，也会回退到项目工作目录解析。

## 推荐入口：`configs/scenarios/*.yaml`

当前推荐通过 scenario 配置运行项目：

```powershell
python scripts/run_scenario.py configs/scenarios/single_analytic_smoke.yaml
python scripts/run_scenario.py configs/scenarios/dual_mujoco_tracking.yaml
```

scenario YAML 负责组合 assembly、backend、scene、task、runtime、hooks 和 artifacts。
旧 `configs/main_config*.yaml` 索引入口已不再作为运行入口维护。

## 四个 MuJoCo tracking 控制基线

以下四个场景共享 `configs/control/mujoco_tracking_low_level.yaml`。2026-07-15 之前保存的误差表对应旧的
uniform-time、`actual_anchored`、无速度保护基线，不能作为当前配置的验收值。当前问题分解、历史代表性
run 和分阶段验收见 `docs/pcc_mujoco_tracking_optimization.md`；本轮修改后尚未自动运行新基线。

未启用 reference governor 时，`stopped_early: True` 与 `stop_reason: duration_elapsed` 表示正常完成。
当前普通 MuJoCo tracking 启用 governor，正常完成原因为 `reference_complete`：参考走到终点并且末端进入
容差后置 `done`。`max_steps` 仍是独立的运行时上限，不代表轨迹正常完成。
每次运行先保存 reset state，所以通常有 `states = commands + 1`。governor 会改变命令条数，不能再用
旧的固定 4001/4150 行数判断运行是否正确。

### 公共控制链路

四条命令都经过同一组合根和闭环：

1. `scripts/run_scenario.py` 将 scenario YAML 交给 `SimulationApplication.from_yaml()`。
2. application 加载 assembly、共享 low-level profile、MuJoCo 配置及可选 engine scene。
3. application 从源 MJCF 生成本次场景 XML：单臂场景移除另一条臂，engine 场景注入发动机几何，
   固定基座场景锁定 mobile-base freejoint。
4. task 生成或读取世界坐标 waypoint，并按 `tracking_control` 选择 time controller 或 staged controller。
5. time controller 使用独立 approach/path 时间、弧长参数化和角点零速 Hermite 插值生成参考；reference
   governor 可根据实际 tracking error 和 tendon lead 利用率减慢虚拟时间：

   ```text
   p_d(t), p_dot_d(t)
   v_executor = p_dot_d(t) + arm_position_gain * (p_d(t) - p_tip)
   ```

6. `UnifiedLowLevelController` 构造 executor tracking、可选 observer 避碰和 engine clearance 任务；
   `WholeBodyController` 用加权最小二乘/SVD 将笛卡尔速度转换为兼容的 tendon rate。
7. MuJoCo backend 按 `controller_dt_s` 积分 tendon target，再执行 `n_substeps` 个 MuJoCo 步；hooks
   同步记录状态、诊断、viewer 和视频，直到 duration、viewer 或 `max_steps` 结束运行。

当前 `enforce_backend_tendon_limits: true` 且 `backend_tendon_target_mode: protected`。backend 持久积分
tendon target，并共同应用 rate、displacement 和 target-lead 保护。旧 `actual_anchored` 仍可显式选择，
但不再是 MuJoCo tracking 基线。

### 每条命令的具体流程

#### `single_mujoco_tracking.yaml`

1. 加载固定基座 `single_spatial.yaml`，并从 dual 源 XML 只保留 `executor`。
2. 以直臂末端为放置参考生成 80 个方形独立样本并追加首点闭合；边长 40 mm，平面内旋转 15°。
3. 在方形第一个点前添加 40 个五次平滑直线 approach 点；首帧再用实测 executor tip 重建 approach 起点。
4. approach 独立使用 20 s；正式闭合方形按弧长使用 80 s。角点采用零速度 Hermite 插值，并由
   reference governor 在执行链落后时减慢参考。
5. 固定基座 whole-body solve 只输出 executor tendon rate，完成时输出零命令。

#### `dual_mujoco_tracking.yaml`

executor 的轨迹、计时和底层参数与 single 完全相同。区别是保留 observer，并启用
`collision_avoidance`：

- executor tracking 单独求解，只允许 executor tendon（固定基座下 base 也被置零）。
- observer 的最近中心线点距小于 18 mm 时，单独求解沿分离法线的 observer tendon 速度。
- observer 任务不能拉动 executor 或共享基座，所以无碰撞激活或数值耦合时，single/dual 的 executor
  误差几乎一致是预期结果。
- 距离超过 `18 + 2 = 20 mm` 后解除避碰；8 mm 只标记 critical diagnostic，不冻结 executor，
  也不会自动停止场景。

#### `single_engine_tracking.yaml`

1. 加载 `single_spatial_mobile.yaml` 和 engine scene，把发动机几何注入生成的 XML。
2. 使用两个显式世界坐标 waypoint，不生成方形，也不 prepend arm approach。
3. `StagedEngineTrackingController` 首次收到状态时计算一次基座目标：

   ```text
   base_target_position = base_position + first_waypoint - current_executor_tip
   base_target_orientation = current_base_orientation
   ```

4. `base_approach` 阶段只输出世界坐标 base twist，所有 tendon rate 为零；达到 5 mm 与 0.035 rad
   容差后切换。该阶段是直线位姿闭环，不含移动基座碰撞规划。
5. `tracking` 阶段用固定基座 assembly 副本在 80 s 内线性插值两个 waypoint；求解器不含 base 自由度，
   发给 MuJoCo 的 `base_twist_world` 也始终为零，以隔离 tendon 反作用。
6. 基座阶段的 tracking error 记录为 NaN，所以命令行 final/mean/max 只统计机械臂 tracking 阶段。

#### `dual_engine_tracking.yaml`

基座接近和 executor 两点跟踪与 single engine 相同。base approach 时两臂 tendon 都保持零；进入
tracking 后 executor 与 observer 分开求解。observer 使用与 dual MuJoCo 相同的 18 mm 避碰策略。
`observer_roi_world` 仍被传入 controller，但在当前 `observer_control_mode: collision_avoidance` 下不参与
目标计算；只有切换为 `tracking` 时它才与默认 executor offset 混合生成 observer 观察目标。

### 四份 scenario YAML 的全部字段

#### 顶层、backend 与 scene

| 字段 | 当前值/使用场景 | 含义与调参建议 |
|---|---|---|
| `schema_version` | `1`，全部 | 配置格式版本，不是控制参数；不要用它调性能。 |
| `scenario.name` | 四个不同名称 | 日志标识和 `output/runs/<name>_<timestamp>` 前缀；改名不影响控制。 |
| `assembly_config_path` | fixed 或 mobile、single 或 dual assembly | 决定 base 是否可动、启用哪些臂、角色、安装位姿和限幅。切换它等于切换机器人系统，不应作为小幅调参。 |
| `low_level_control_path` | `mujoco_tracking_low_level.yaml` | 四个基线共享的底层增益、权重、奇异保护和限幅策略。相对 scenario YAML 解析。 |
| `backend.type` | `mujoco` | 选择 MuJoCo system backend。 |
| `backend.mujoco_config_path` | `../mujoco_dual.yaml` | 物理 timestep、执行器、solver、viewer camera 和渲染设置。 |
| `backend.source_xml_path` | dual mobile-base MJCF | 每次运行用于生成场景 XML 的模板。 |
| `backend.generated_xml_path` | 每个场景独立输出路径 | 实际交给 MuJoCo 加载的生成 XML；应保持场景间路径不同，避免并行运行互相覆盖。 |
| `backend.retain_arm` | single 场景为 `executor`；dual 省略 | 从 dual 源 XML 中只保留指定臂，使 single assembly 与模型一致。不要在 dual 场景设置。 |
| `scene: {}` | 两个普通 MuJoCo tracking | 不注入额外环境，也没有 engine clearance query。 |
| `scene.engine_config_path` | 两个 engine tracking | 同一份配置同时用于注入发动机几何和创建控制器 distance query；改模型位姿会直接改变碰撞距离和 waypoint 语义。 |

#### task、time tracking 与 trajectory

| 字段 | 当前值 | 实际作用与调参建议 |
|---|---:|---|
| `task.type` | `tracking` | 选择 tracking controller 分支。 |
| `task.waypoint_tolerance_m` | 普通 `0.003`；engine `0.002` | waypoint 模式的到点阈值。当前四个场景都是 time 模式，因此不决定推进或结束，仅保留在 controller 配置中。 |
| `task.target_advance_mode` | 普通场景显式 `tolerance`；engine 使用默认值 | 仅 waypoint 模式控制按 tolerance/time/steps 推进；time 模式忽略。 |
| `task.loop` | `false` | false 时在参考完成后结束；governor 开启时墙钟用时可以超过名义 duration。loop 当前只支持旧的 uniform/linear time sampling。 |
| `tracking_control.approach_samples` | 普通 `40`；engine `0` | 普通场景在正式路径前生成五次平滑接近样本；首帧起点使用实测 executor tip。engine 已用 base approach，不应再添加 arm approach。 |
| `tracking_control.approach_duration_s` | 普通 `20.0`；engine `0` | 单独控制接近段时长，不再占用正式路径的 `trajectory_duration_s`。 |
| `tracking_control.tracking_mode` | `time` | 按仿真时间连续插值；改为 `waypoint` 后才使用 tolerance/advance 参数，并改用 waypoint 前馈逻辑。 |
| `tracking_control.trajectory_duration_s` | `80.0` | 正式路径名义时长，不含普通 arm approach 或 engine base approach。governor 可增加实际墙钟用时。 |
| `tracking_control.time_parameterization` | 普通 `arc_length` | `arc_length` 按路径弦长分配时间，避免 waypoint 密度改变局部速度；旧默认是 `uniform_waypoint`。 |
| `tracking_control.trajectory_interpolation` | 普通 `corner_stop_hermite` | 保持精确直边，并在检测到方向突变的 waypoint 将速度连续降为零；旧默认是 `linear`。 |
| `tracking_control.reference_governor_enabled` | 普通 `true` | 根据 tracking error 与 executor tendon lead 利用率推进虚拟参考时间；engine 当前保持 false。 |
| `tracking_control.stage_mobile_base` | engine 为 `true`；普通场景省略/false | 在 time tracking 前执行一次 base-only 接近。只允许 mobile assembly；普通 fixed-base 场景不能开启。 |
| `base_position_gain` | `1.5 s^-1` | `v_base = gain * position_error`。增大可缩短接近时间，但当前 staged controller 不做速度裁剪，过大可能跳动或穿越场景。 |
| `base_orientation_gain` | `2.0 s^-1` | 旋转向量误差到角速度的比例。当前目标姿态等于初始姿态，正常情况下误差接近零；有姿态扰动时才明显生效。 |
| `base_position_tolerance_m` | `0.005` | base approach 切换阈值。减小可提高交接位置精度，但会增加接近步数，并可能因噪声迟迟不切换。 |
| `base_orientation_tolerance_rad` | `0.035` | 姿态切换阈值，约 2°。与 position tolerance 必须同时满足。 |
| `task.trajectory.type` | `square`，普通场景 | 选择程序化方形 waypoint 生成器。engine 使用显式点，不含该块。 |
| `task.trajectory.samples` | `80` | 方形独立样本数；`closed: true` 会额外追加首点。arc-length 模式下改变采样密度不会改变名义路径速度。 |
| `task.trajectory.closed` | `true` | 显式追加首点，确保非 loop 的方形也走完最后一条边。 |
| `task.trajectory.radius_m` | `0.018` | square 已显式给 `side_length_m`，所以它不决定边长；它仍参与 `z_mode` 的参考尺度计算。 |
| `trajectory.placement.center_mode` | `straight_tip_xy` | x/y 取直臂 executor tip，z 由 `z_mode` 计算。改为 explicit 时需要提供中心坐标。 |
| `trajectory.placement.z_mode` | `straight_tip_minus_radius` | 中心 z 为直臂 tip z 减去所有有效轨迹尺度的最大值。当前 `side_length/2 = 0.020 m` 大于 `radius_m`，实际下移 20 mm。 |
| `trajectory.placement.plane` | `xy` | 方形位于世界 xy 平面；`xz`/`yz` 会改变运动平面和可达性。 |
| `trajectory.placement.yaw_deg` | `15.0` | 在所选平面内旋转轨迹，不旋转机器人基座。 |
| `trajectory.shape.side_length_m` | `0.040` | 方形边长。增大会扩大工作空间和速度需求；固定 duration 下近似按比例提高路径速度。 |
| `task.waypoints_world` | engine 两个 `[x,y,z]` 点 | 显式世界坐标路径；time controller 在两点间做 80 s 直线插值。先核对 engine/world 坐标约定，再改位置。 |

#### dual observer 策略

| 字段 | 当前值 | 实际作用与调参建议 |
|---|---:|---|
| `observer_control_mode` | `collision_avoidance` | `tracking` 跟随观察目标，`collision_avoidance` 只在接近时驱动 observer，`disabled` 始终给 observer 零任务。 |
| `observer_roi_world` | 仅 dual engine：`[0.115, 0.500, 0.060]` | 当前 collision mode 下不参与控制；切为 tracking 后与 executor-offset 目标混合。 |
| `observer_control.minimum_distance_m` | `0.010` | 期望安全距离，当前实现主要用于诊断显示；避碰激活实际由 `influence_distance_m` 决定。 |
| `observer_control.influence_distance_m` | `0.018` | 最近两臂中心线点距低于此值时激活 observer 回避。增大可更早避让，但可能增加无谓运动。必须大于 `minimum_distance_m`。 |
| `observer_control.critical_distance_m` | `0.008` | critical 诊断阈值，必须不大于 minimum。当前策略不会因此冻结 executor 或 stop all。 |
| `observer_control.release_margin_m` | `0.002` | 避碰激活后的滞回；当前释放距离为 20 mm。增大可减少阈值附近频繁切换，但延长避让状态。 |
| `observer_control.avoidance_gain` | `1.2 s^-1` | `avoidance_speed = gain * max(influence_distance - distance, 0)`。增大可更快分离，也会增大 tendon 速度和抖动风险。 |
| `observer_control.max_avoidance_speed_mps` | 省略/`None` | 不设 observer 避碰专用速度上限。需要限制峰值时添加正值；它独立于 Cartesian target speed 开关。 |

#### runtime 与时间一致性

| 字段 | 当前值 | 实际作用与调参建议 |
|---|---:|---|
| `runtime.controller_dt_s` | `0.02 s` | 控制命令周期，也是 base/tendon target 的软件积分步长。减小可提高控制更新率，但增加计算与记录量。 |
| `runtime.n_substeps` | `20` | 每条控制命令执行的 MuJoCo 物理步数。`mujoco_dual.yaml` timestep 为 0.001 s，当前正好 `20 * 0.001 = 0.02 s`。 |
| `runtime.max_steps` | 普通 `9000`；engine `6500` | 分别约 180 s/130 s 的独立安全上限，为 governor 或 base approach 留余量；达到它不等于任务正常完成。 |

必须优先保持：

```text
controller_dt_s == n_substeps * mujoco.solver.timestep
```

backend 用 `controller_dt_s` 积分 base/tendon target，但 time controller 的时钟来自 MuJoCo state time。
两者不一致会让目标积分、物理时间和轨迹采样产生系统性偏差。修改 timestep 时应成组调整，不要只改一个字段。

#### hooks 与 artifacts

| 字段 | 当前值 | 作用与调参建议 |
|---|---:|---|
| `hooks.recorder` | `true` | 保存状态、命令和误差；命令行 final/mean/max 依赖它。关闭会减少内存，但也失去主要结果与 plots 数据。 |
| `hooks.tendon_debug` | `true` | 保存 tendon target/actual/rate/force、solver metadata 等快照。只影响诊断开销，不改变控制律。 |
| `hooks.tendon_debug_stride` | `5` | 每 5 个 step 采一帧，外加 reset 快照。增大可减小内存/文件，减小可提高瞬态分辨率。 |
| `hooks.show_live_tendon_panel` | `true` | 打开 tendon 实时面板。只影响 UI 与运行开销。 |
| `hooks.live_tendon_panel_stride` | `5` | tendon panel 刷新间隔；GUI 卡顿时增大。 |
| `hooks.show_live_diagnostics_panel` | 普通显式 `true`；engine 省略但 task 非 idle 时默认 `true` | 打开 tracking/safety/solver/actuator 综合面板。 |
| `hooks.live_diagnostics_panel_stride` | 普通显式 `5`；engine 默认 `5` | 综合面板刷新间隔。 |
| `hooks.viewer` | `mujoco` | 启动 passive MuJoCo viewer；关闭窗口会以 `viewer_closed` 提前结束。 |
| `hooks.keep_viewer_open` | `false` | 完成后自动关闭 viewer；true 会阻塞等待用户关闭窗口。 |
| `artifacts.enabled` | engine 显式 `true`；普通场景默认 `true` | 总产物开关。false 时不保存 run directory 内容，且不会注册 live video hook。 |
| `artifacts.save_npz` | engine 显式 `true`；普通默认 `true` | 保存数值历史。 |
| `artifacts.save_plots` | engine 显式 `true`；普通默认 `true` | 生成误差、tendon 和 safety 图。 |
| `artifacts.save_gif` | `true` | 启用 GIF；与 `video_mode` 一起决定 live/replay 路径。 |
| `artifacts.video_mode` | `live_mujoco` | 仿真运行时直接抓取 MuJoCo 帧，不在结束后重放 qpos。 |
| `artifacts.video_fps` | `10` | 输出 GIF 播放帧率，不是控制频率。 |
| `artifacts.video_stride` | `10` | 每 10 个 controller step 抓一帧；当前每 0.2 s 仿真时间取一帧，即 5 capture fps，以 10 fps 播放约为 2 倍速。 |

四份 YAML 未显式写出的 artifact 默认值还包括 `output_root: ../../output/runs`、`save_model: true`；
普通场景的 `save_npz`/`save_plots` 也默认为 true。省略字段不等于关闭。

### 共享 `mujoco_tracking_low_level.yaml` 参数与调参

加载优先级为“代码默认值 < `low_level_control` profile < `task.tracking_control`”。四个基线没有在
task 中覆盖底层字段，因此实际值来自同一个 profile。

| 字段 | 当前值 | 控制作用 | 调参方向与风险 |
|---|---:|---|---|
| `arm_position_gain` | `1.0 s^-1` | 同时覆盖 executor/observer 的笛卡尔位置反馈增益。position 模式使用 `feedforward_gain*v_ff + Kp*error`。 | 当前低增益与 governor 配合；不要用继续增大 gain 代替参考降速。 |
| `feedforward_gain` | `1.0` | 无量纲的轨迹前馈比例，作用于 time 模式的轨迹导数和 waypoint 模式的前馈速度。 | `0` 关闭前馈但保留位置反馈；`0~1` 降低前馈；`1` 保持原始轨迹速度；大于 `1` 放大前馈并可能增加超调和 tendon 控制量。 |
| `feedforward_speed_mps` | `0.0` | 仅 waypoint 模式沿下一段提供常速前馈。 | 当前 time 模式由轨迹导数自动提供前馈，此字段不生效。 |
| `max_target_speed_mps` | `0.006` | 反馈与前馈相加后的 Cartesian 总目标速度上限。 | 80 s 方形平均边速度约 2 mm/s，其余预算留给位置反馈。 |
| `enforce_target_speed_limit` | `true` | 是否裁剪 executor/observer 的合成 Cartesian 速度。 | 当前开启，防止位置误差被高增益放大成不可实现命令。 |
| `executor_tracking_weight` | `100.0` | executor tracking 相对 tendon 正则与环境避碰的权重。 | 增大偏向跟踪，可能放大控制量；降低更保守但误差增大。权重只有相对比例有意义。 |
| `observer_tracking_weight` | `40.0` | observer 为 tracking 模式时的任务权重。 | 当前 collision mode 不使用 observer tracking task。 |
| `executor_collision_avoidance_weight` | `80.0` | engine clearance task 权重；未单设 observer collision weight 时也作为 observer 避碰权重。 | 提高避障相对正则的优先级；过高可能产生激烈回避。它不会把 observer 任务耦合到 executor solve。 |
| `base_regularization_weight` | `1.0` | 抑制 whole-body solve 中 base 速度。 | 这四个场景的 tracking solve 都是 fixed base，engine base approach 又绕过 whole-body solver，因此当前基本不影响结果。 |
| `tendon_regularization_weight` | `0.8` | 惩罚 tendon effort，平衡 tracking 权重。 | 增大更平滑、控制量更小但误差增大；减小更积极，也更容易放大奇异和噪声。 |
| `singularity_strategy` | `svd_projection` | 将任务速度投影到可控 SVD 子空间，丢弃弱方向。 | 适合当前基线。切换 `damping_scale` 会启用下面的自适应阻尼/缩放，属于策略变化，不宜与其他调参同时进行。 |
| `rank_tolerance` | `1e-9` | SVD 秩诊断阈值，也用于判断避碰 Jacobian 是否近零。 | 提高会更早判定方向不可控；过高可能误删有效避碰任务。 |
| `minimum_singular_value` | `1e-5` | `svd_projection` 保留弱方向的阈值。 | 增大更保守、投影残差可能上升；减小可追踪更弱方向，但 tendon rate 和数值敏感性可能升高。 |
| `nominal_damping` | `1e-3` | `damping_scale` 在可控状态的阻尼。 | 当前 SVD projection solve 不使用该阻尼，只保留诊断计算。切换策略后再调。 |
| `maximum_damping` | `0.1` | `damping_scale` 接近奇异时的最大阻尼。 | 当前策略下不参与 solve；必须不小于 nominal。 |
| `minimum_velocity_scale` | `0.05` | `damping_scale` 接近奇异时的最低速度比例。 | 当前策略下不参与 solve；增大响应更强，降低更保守。 |
| `decouple_arm_singularity` | `true` | fixed-base 且 `damping_scale` 时按臂分别应用阻尼/缩放。 | 当前 `svd_projection` 下不改变求解结果。 |
| `enforce_solver_velocity_limits` | `true` | 在 solver 输出端按臂统一缩放，满足 assembly tendon rate limit。 | 统一缩放保持 tendon 方向；诊断需观察 constrained command。 |
| `enforce_backend_tendon_limits` | `true` | backend 应用 rate、displacement 与 target-lead 保护。 | 与显式 target mode 一起构成当前安全基线。 |
| `backend_tendon_target_mode` | `protected` | 使用有界持久 target；旧的 `actual_anchored` 每拍重新锚定实际位移。 | `free_integrated` 可能积累不可达目标，不能作为当前基线。 |

`feedforward_gain` 与 `arm_position_gain` 控制不同部分：

```text
position 模式：v_target = arm_position_gain * (p_target - p_measured)
                        + feedforward_gain * v_feedforward
velocity 模式：v_target = v_direct
```

因此 tracking、navigation、engine tracking 和 wiping 中由轨迹导数或 waypoint 方向产生的前馈会被缩放；
engine cleaning 以及 navigation 安全控制生成的直接 velocity intent 不会被缩放。最终的
`max_target_speed_mps` 裁剪发生在反馈与缩放后前馈相加之后，所以合成速度已经饱和时，降低
`feedforward_gain` 不一定产生同等比例的最终速度变化。

默认值和两个共享 profile 都是 `1.0`，用于保持已有行为。全局调节位置为
`configs/control/mujoco_tracking_low_level.yaml` 或 `configs/control/spatial_low_level.yaml`；单个任务可在
scenario 的 `task.tracking_control.feedforward_gain` 中覆盖。运行产物 `result.npz` 记录：

- `task_intent_velocity_world`：上层给出的原始前馈或直接速度；
- `executor_feedforward_gain`：当前配置比例；
- `executor_scaled_feedforward_velocity_world`：按控制模式处理后的速度；
- `executor_target_velocity_world`：与位置反馈相加并经过可选限速后的最终笛卡尔目标速度。

### 推荐调参顺序

1. **先固定比较条件**：保持轨迹几何、seedless 初始状态、`controller_dt_s = n_substeps * timestep`、
   viewer/video 开关一致；一次只改一个控制变量。
2. **先调速度，再调反馈**：误差整体偏大且 actuator/tendon 未饱和时，先增大
   `trajectory_duration_s`；仍有相位滞后再将 `arm_position_gain` 以小步幅提高。若振荡或 force 峰值增大，反向调整。
3. **普通方形先分清 approach 与主路径**：`approach_samples` 会占用总时长。希望保留更多主轨迹时间时，
   减少 approach 点或同步增加 duration；不要只增加 samples 后直接比较 mean error。
4. **engine 先调基座交接**：base approach 太慢时小幅提高 `base_position_gain` 或放宽 tolerance；
   交接跳变时降低 gain 或收紧 tolerance。由于当前无速度裁剪，应优先观察 base twist/position，不做大幅增加。
5. **dual 再调 observer**：先用 `influence_distance_m` 决定何时避让，再用 `avoidance_gain` 决定避让速度，
   最后按需添加 `max_avoidance_speed_mps`。同时检查最小臂间距和 observer tendon/force，不用 executor error 单独判断安全性。
6. **最后调 solver 与限幅**：只有诊断显示 weak singular directions、projection residual、rate/force 或 target error
   异常时，才调整 SVD 阈值、regularization 或保护开关；每次策略变化都重新建立 baseline。

比较时至少同时观察：final/mean/max executor error、分阶段误差、最小臂间距离、SVD singular value 与
projection residual、requested/applied tendon rate、target/actual tendon error、limit scale 和 actuator force。

## 旧索引配置：`configs/main_config.yaml`

| 字段                     | 类型 | 说明                          |
|--------------------------|------|-------------------------------|
| `schema_version`         | int  | 配置 schema 标记。当前值为 `1`。 |
| `robot_config`           | path | 规范机器人 YAML。             |
| `pcc_backend_config`     | path | Analytic PCC 后端 YAML。      |
| `mujoco_backend_config`  | path | MuJoCo 后端 YAML。            |
| `pcc_tracking_config`    | path | PCC tracking 任务 YAML。      |
| `mujoco_tracking_config` | path | MuJoCo tracking 任务 YAML。   |
| `mujoco_navigation_config` | path | MuJoCo navigation 任务 YAML。 |
| `mujoco_wiping_config` | path | MuJoCo wiping 任务 YAML。 |

## 运行产物导出

带 artifacts 的 scenario 会把本次运行写入 `output/runs/<scenario>_<timestamp>/`：

```powershell
python scripts/run_scenario.py configs/scenarios/single_mujoco_wiping.yaml
```

保存产物包括 `result.npz`、`metadata.json`、复制后的 YAML 配置、PNG 曲线图、可用时生成的
MuJoCo 场景 XML，以及 `videos/simulation.gif`。MuJoCo 回放视频会在独立子进程中根据保存的
`qpos/qvel` 历史和归档的 `model/scene.xml` 导出，尺寸使用 MuJoCo 后端中的
`rendering.offscreen_*`，相机使用 `viewer.camera`。如果离屏渲染或视频编码不可用，命令仍会保存
NPZ/PNG，失败原因写入 `videos/video_error.txt`，并同步汇总到 `metadata.json.errors`；
`metadata.json.video_status` 会标记为 `ok`、`failed` 或 `disabled`。

### MuJoCo–PCC 模型与雅可比诊断

所有 `backend.type: mujoco` 且 `artifacts.enabled: true` 的任务都会在运行结束导出产物时计算
MuJoCo–PCC 对照诊断。该功能默认启用，不新增 YAML 开关，也不依赖 `hooks.recorder`；它只读取已经
产生的状态和命令，不参与控制求解。安装座坐标优先使用 state metadata 中的 MuJoCo 实际
`mobile_base_frame` 位姿，缺失时才回退到软件 base pose。`artifacts.enabled: false` 的纯 viewer
场景仍不创建运行产物。

每条启用机械臂使用独立的 `arm_<name>_` 前缀。主要 NPZ 字段如下：

| 字段后缀 | 对齐 | 含义 |
|---|---|---|
| `mujoco_tip_position_world_m` / `mount_m` | state | MuJoCo site 末端的世界坐标位置，以及转换到当前机械臂安装座坐标系的位置。 |
| `pcc_bending_state_rad_per_m` | state | 根据实际 MuJoCo 腱位移经 bending-space 伪逆估计的 PCC 弯曲状态。 |
| `pcc_tip_position_mount_m` / `pcc_tip_position_world_m` | state | 同一弯曲状态经 PCC FK 得到的局部/世界坐标末端。 |
| `pcc_mujoco_tip_error_world_m` / `mount_m` | state | `PCC FK - MuJoCo` 的世界/安装座 XYZ 位置误差；对应坐标系中正 Z 表示 PCC 预测比 MuJoCo 末端更高。 |
| `pcc_mujoco_tip_error_norm_m` | state | 上述位置误差的二范数。 |
| `pcc_bending_jacobian_mount` / `world` | state | 弯曲空间 PCC 位置雅可比；world 版本使用 MuJoCo 实际 base frame，形状为 `N×3×6`。 |
| `pcc_tendon_jacobian_mount` / `world` | state | 弯曲雅可比乘 tendon-to-bending 伪逆后的物理腱雅可比，形状为 `N×3×tendon_count`。 |
| `pcc_software_frame_bending_jacobian_world` | state | 使用软件 base pose 旋转到世界坐标的弯曲雅可比，与协调控制器的坐标来源一致。 |
| `pcc_reconstructed_system_jacobian_world` | state | 按后端 assembly 的统一 layout 重构的完整 system Jacobian，包含可用 base 列、当前臂弯曲列及其他臂零列；direct/staged 阶段不一定把它作为求解任务。 |
| `pcc_jacobian_singular_values`、`rank`、`condition_number` | state | PCC 弯曲雅可比的 SVD 可控性诊断。 |
| `pcc_jacobian_mount_row_norm` / `world_row_norm` | state | XYZ 每一行的灵敏度；world 数组第三列是世界 Z 方向的一阶灵敏度。 |
| `mujoco_tip_velocity_fd_mount_mps` / `world_mps` | command transition | 相邻 MuJoCo state 末端位置的有限差分速度。 |
| `pcc_tip_velocity_fd_mount_mps` | command transition | 相邻实际腱位移分别做 PCC FK 后的有限差分速度。 |
| `pcc_tip_velocity_from_tendon_delta_mount_mps` | command transition | 起点 PCC 雅可比乘本周期实际腱位移增量所预测的末端速度。 |
| `pcc_tip_velocity_from_measured_tendon_*` | state | PCC 雅可比乘 MuJoCo 报告的瞬时腱速度。 |
| `pcc_command_arm_tip_velocity_*` | command transition | PCC 雅可比乘控制器请求腱速度，不含底座贡献。 |
| `base_command_tip_velocity_world_mps` | command transition | `base_twist_world` 对该臂末端世界速度的单独贡献。 |
| `pcc_command_total_tip_velocity_world_mps` | command transition | 机械臂命令预测与底座命令贡献之和。 |

`state` 字段长度等于 `states`；`command transition` 字段长度等于可用命令数，索引 `i` 表示
`state[i] --command[i]--> state[i+1]`。三个残差族用于拆分根因：

- `pcc_jacobian_linearization_residual_* = Jacobian(actual tendon delta) - PCC FK finite difference`。
  它很小而后续模型残差较大时，解析雅可比通常不是主要问题。
- `pcc_mujoco_model_velocity_residual_* = PCC FK finite difference - MuJoCo finite difference`。
  它直接衡量相同实际腱增量在 PCC 与 MuJoCo 中产生的末端运动差异。
- `pcc_command_mujoco_velocity_residual_* = command prediction - MuJoCo finite difference`。
  它同时包含执行器动力学、tendon target 跟随、模型差异和命令/实际腱速度差异，不能单独用于判定雅可比错误。

每条臂会额外生成三张图：

- `arm_<name>_pcc_mujoco_position.png`：安装座坐标系中的 MuJoCo/PCC XYZ 位置及位置误差。
- `arm_<name>_pcc_mujoco_velocity.png`：MuJoCo 有限差分、PCC FK 有限差分、雅可比预测、命令预测和三类残差。
- `arm_<name>_pcc_jacobian.png`：奇异值、世界 XYZ 行灵敏度、条件数与秩。

建议按以下顺序判读：

1. 先看 `pcc_mujoco_tip_error_mount_m`，确认固定方向偏差是否在安装座坐标系中仍然存在；若存在，
   它不是移动底座世界位姿造成的假象。
2. 若雅可比线性化残差远小于 PCC–MuJoCo 模型速度残差，优先检查 PCC 常曲率假设、MuJoCo
   分布式关节、spatial tendon 路径、neutral tendon length 和安装/末端 site 定义。
3. 若雅可比线性化残差本身也大，先检查 tendon-to-bending 映射、雅可比坐标系、状态/命令时间对齐；
   同时减小单周期运动量，区分实现错误与局部线性化误差。
4. 若实际腱增量的 PCC 预测接近 MuJoCo，但命令预测偏差大，优先检查 actuator force、tendon target
   模式、requested/applied rate、target lead 和物理响应滞后。
5. 若世界 Z 行范数或最小奇异值长期接近零，当前构型对世界 Z 方向的一阶控制能力很弱；直臂附近
   Z 缩短主要是曲率的二阶效应，不应先通过降低 SVD 阈值或大幅提高位置增益处理。

位置误差、雅可比线性化速度残差、PCC–MuJoCo 模型速度残差和命令响应残差的 final/mean/max
同时写入 `metadata.json.metrics`。若诊断计算失败，原有运行产物仍会保存，错误写入
`metadata.json.errors`。

已保存的 MuJoCo 运行结果可以在不重新仿真的情况下再次导出视频：

```powershell
python scripts/export_replay_video.py `
  --result-npz output/runs/<task_name>_<timestamp>/result.npz `
  --scene-xml output/runs/<task_name>_<timestamp>/model/scene.xml `
  --output output/runs/<task_name>_<timestamp>/videos/replay.gif
```

批量导出前，可以先检查当前 XML 是否能创建指定尺寸的离屏渲染器：

```powershell
python scripts/check_mujoco_offscreen_renderer.py --config configs/mujoco.yaml
```

## 运行期同步诊断

scenario 的 `hooks` 可打开同步诊断窗口：

```yaml
hooks:
  show_live_diagnostics_panel: true
  live_diagnostics_panel_stride: 5
  live_diagnostics_panel_history_points: 300
```

该面板同步显示 tracking/base error、clearance/contact/force error、奇异条件数、
限速/限幅比例和 tendon target error。tracking、navigation、wiping 运行时可用同一面板对照
目标误差、环境距离、接触力代理和 tendon 执行滞后，定位误差来自控制目标、避障/接触约束、
奇异性还是执行器饱和。

## `configs/robot_3seg.yaml`

| 字段                                     | 类型        | 说明                                 |
|------------------------------------------|-------------|--------------------------------------|
| `schema_version`                         | int         | 配置 schema 标记。                   |
| `name`                                   | string      | 机器人标识。                         |
| `units.*`                                | string      | 单位声明，用作文档和校验上下文。     |
| `robot.type`                             | string      | 机器人类型标识。                     |
| `robot.segment_count`                    | int         | 连续体段数。                         |
| `robot.tendons_per_segment`              | int         | 每段名义局部腱数。                   |
| `robot.total_tendon_count`               | int         | 物理腱总数。                         |
| `robot.base_frame`                       | string      | 基座坐标系名称。                     |
| `robot.tip_frame`                        | string      | 末端坐标系名称。                     |
| `materials.backbone.*`                   | scalar      | backbone 材料元数据。                |
| `materials.tendons.*`                    | scalar      | tendon 材料元数据。                  |
| `segments[].id`                          | string      | 段标识。                             |
| `segments[].index`                       | int         | 从 0 开始的段索引。                  |
| `segments[].length`                      | float, m    | 段长。                               |
| `segments[].backbone_radius`             | float, m    | backbone 的视觉/物理半径。           |
| `segments[].tendon_radius`               | float, m    | 局部腱路径半径。                     |
| `segments[].mass`                        | float, kg   | 段质量元数据。                       |
| `segments[].bending_stiffness`           | float       | 弯曲刚度元数据。                     |
| `segments[].torsional_stiffness`         | float       | 扭转刚度元数据。                     |
| `segments[].tendon_angles_deg`           | list[float] | 名义局部腱角度。                     |
| `segments[].tendons[].*`                 | scalar      | 局部腱 ID、索引、角度和径向偏置。    |
| `physical_tendons[].id`                  | string      | 物理腱标识。                         |
| `physical_tendons[].global_index`        | int         | 腱向量中的索引。                     |
| `physical_tendons[].motor_index`         | int         | 驱动该腱的电机索引。                 |
| `physical_tendons[].anchor_segment_index`| int         | 腱锚定/终止的段索引。                |
| `physical_tendons[].angle_deg`           | float, deg  | 耦合矩阵使用的腱角度。               |
| `physical_tendons[].radial_offset`       | float, m    | 耦合矩阵使用的径向偏置。             |
| `physical_tendons[].path_segment_indices`| list[int]   | 该腱经过的段索引。                   |
| `motors.position_unit`                   | string      | 电机位置单位。                       |
| `motors.velocity_unit`                   | string      | 电机速度单位。                       |
| `motors.length_unit`                     | string      | 腱长单位。                           |
| `motors.items[].id`                      | string      | 电机标识。                           |
| `motors.items[].motor_index`             | int         | 电机向量中的索引。                   |
| `motors.items[].tendon_global_index`     | int         | 该电机驱动的物理腱索引。             |
| `motors.items[].spool_radius`            | float, m    | 卷筒半径。                           |
| `motors.items[].gear_ratio`              | float       | 传动比乘子。                         |
| `motors.items[].direction_sign`          | float       | 电机到腱长映射的方向符号约定。       |
| `motors.items[].zero_position`           | float, rad  | 电机零位偏置。                       |
| `actuation.command_type`                 | string      | 命令约定。                           |
| `actuation.tendon_count`                 | int         | 期望腱数。                           |
| `actuation.limits.min_length_delta`      | float, m    | 腱长命令下限。                       |
| `actuation.limits.max_length_delta`      | float, m    | 腱长命令上限。                       |
| `actuation.limits.max_tension`           | float, N    | 张力元数据。                         |

Spatial-arm assembly configs also accept `spatial_arm.limits.target_lead_m`.
This positive scalar or per-tendon vector limits how far a tendon-position
target may lead the measured tendon length. The executor / observer configs use
the same `[-0.020, 0.020]` m displacement range, `0.005` m/s rate limit, and
`0.00025` m target lead. The code default remains `0.0005` m for older configs
that omit this field.

## `configs/pcc.yaml`

| 字段                                  | 类型         | 说明                             |
|---------------------------------------|--------------|----------------------------------|
| `schema_version`                      | int          | 配置 schema 标记。               |
| `backend`                             | string       | 后端名称。当前值为 `pcc`。       |
| `enabled`                             | bool         | 该后端配置是否启用。             |
| `robot_config_path`                   | path         | 后端使用的机器人 YAML。          |
| `model.assumption`                    | string       | 建模假设。                       |
| `model.segment_count`                 | int          | 段数。                           |
| `model.state_variables`               | list[string] | 每段 PCC 状态变量名称。          |
| `model.integration.samples_per_segment` | int        | 中心线/FK 采样密度。             |
| `model.integration.method`            | string       | 积分方法标签。                   |
| `solver.mode`                         | string       | solver 模式标签。                |
| `solver.tolerance`                    | float        | 数值容差。                       |
| `solver.max_iterations`               | int          | 最大 solver 迭代次数。           |
| `output.save_centerline`              | bool         | 是否保留 centerline 输出。       |
| `output.save_tip_pose`                | bool         | 是否保留 tip pose 输出。         |
| `runtime.timestep`                    | float, s     | 运行时 timestep 元数据。         |

## `configs/mujoco.yaml`

| 字段                              | 类型         | 说明                                             |
|-----------------------------------|--------------|--------------------------------------------------|
| `schema_version`                  | int          | 配置 schema 标记。                               |
| `backend`                         | string       | 后端名称。当前值为 `mujoco`。                    |
| `enabled`                         | bool         | 该后端配置是否启用。                             |
| `robot_config_path`               | path         | 用于映射和 overlay 的机器人 YAML。               |
| `xml_path`                        | path         | 基础 MuJoCo XML。                                |
| `tendon_xml_path`                 | path         | 启用 tendon 的 MuJoCo XML。                      |
| `generated_xml_path`              | path         | 生成的视觉 XML 输出路径。                        |
| `tendon_generated_xml_path`       | path         | 生成的 tendon 视觉 XML 输出路径。                |
| `asset_scale`                     | float        | 导入资产的缩放比例。                             |
| `links_per_segment`               | int          | 每个连续体段的降阶 link 数。                     |
| `control_mode`                    | string       | `tendon_position` 或 `position_joint`。          |
| `visuals.enabled`                 | bool         | 是否启用 segmented visual 生成/使用。            |
| `visuals.frame_mode`              | string       | 网格坐标系约定。                                 |
| `visuals.cad_origin_mm`           | list[float]  | CAD 原点偏置，单位 mm。                          |
| `visuals.mesh_unit`               | string       | 源网格单位标签。                                 |
| `visuals.mesh_scale`              | float        | 网格缩放到米的比例。                             |
| `visuals.directory`               | path         | 网格目录。                                       |
| `visuals.template_path`           | path         | 视觉 XML 模板。                                  |
| `visuals.collision_mode`          | string       | 碰撞几何模式。                                   |
| `visuals.visual_geom_group`       | int          | MuJoCo visual group 索引。                       |
| `visuals.collision_geom_group`    | int          | MuJoCo collision group 索引。                    |
| `visuals.expected_meshes`         | list[string] | 检查时预期存在的网格文件名。                     |
| `viewer.show`                     | bool         | 为 true 时打开 passive viewer。                  |
| `viewer.steps`                    | int          | viewer 运行时 backend 推进次数。                 |
| `viewer.use_segment_visuals`      | bool         | 可用时使用生成的视觉 XML。                       |
| `viewer.show_collision_geoms`     | bool         | 是否显示碰撞几何 group。                         |
| `viewer.sync_interval_steps`      | int          | viewer 同步间隔。                                |
| `viewer.realtime`                 | bool         | 是否按近似实时速度 sleep。                       |
| `viewer.realtime_factor`          | float        | 实时播放倍率。                                   |
| `viewer.camera.lookat`            | list[float]  | 相机 lookat 位置。                               |
| `viewer.camera.distance`          | float        | 相机距离。                                       |
| `viewer.camera.azimuth`           | float, deg   | 相机方位角。                                     |
| `viewer.camera.elevation`         | float, deg   | 相机俯仰角。                                     |
| `viewer.overlays.*`               | scalar       | 目标 marker、轨迹 trail 和 tendon path overlay 设置。 |
| `viewer.overlays.error_vector`     | bool         | 在 viewer 中显示当前执行点到当前目标的误差向量。 |
| `viewer.overlays.error_vector_radius` | float, m  | 误差向量 capsule 半径。                         |
| `viewer.overlays.error_vector_rgba` | list[float] | 误差向量颜色。                                  |
| `solver.timestep`                 | float, s     | MuJoCo 积分 timestep。                           |
| `solver.integrator`               | string       | MuJoCo integrator。                              |
| `solver.iterations`               | int          | MuJoCo solver 迭代次数。                         |
| `gravity.enabled`                 | bool         | 是否启用重力。                                   |
| `gravity.vector_m_s2`             | list[float]  | 重力向量。                                       |
| `joints.hinge.*`                  | scalar       | hinge damping、armature、limit、range、stiffness 和 springref。 |
| `tendon_model.*`                  | scalar       | tendon model 类型、数量、限幅、damping、stiffness 和 coefficient source。 |
| `actuators.tendon_position.*`     | scalar       | tendon position gain、命令范围和力范围。         |
| `actuators.tendon_position.kp`    | float, N/m   | Tendon position gain. Dual-arm config uses `100000.0` with `forcerange_n: [-30.0, 30.0]`. |
| `actuators.tendon_position.ctrllimited` | bool | Spatial-tendon MJCF uses `false` because MuJoCo receives absolute tendon lengths; `ctrlrange_m` remains the software-side relative displacement range. |
| `mobile_base_xml_path`            | path         | Optional committed output for the mobile-base-wrapped model generated after `tendon_xml_path`. |
| `visuals.world_frame.*`           | mapping      | Optional world-origin and RGB-axis MJCF site dimensions, colors, and geom group. |
| `actuators.joint_position.*`      | scalar       | joint position gain、命令范围和力矩范围。        |
| `sensors.*`                       | bool         | tendon length、velocity 和 actuator force sensor 开关。 |
| `smoke_tests.*`                   | scalar       | MuJoCo smoke 路径使用的轻量数值检查参数。        |
| `rendering.offscreen_width`       | int          | MuJoCo replay video 离屏 framebuffer 宽度。      |
| `rendering.offscreen_height`      | int          | MuJoCo replay video 离屏 framebuffer 高度。      |
| `site_names.*`                    | string/list  | base、segment tip 和最终 tip site 名称。         |
| `notes`                           | list[string] | 面向人的备注。                                   |

`viewer.camera.*` 同时影响 passive viewer、生成 XML 的默认 MuJoCo visual camera，以及
scenario artifacts 的 MuJoCo replay GIF。`rendering.offscreen_*` 只控制离屏导出 framebuffer
尺寸，不改变仿真本身。

## Tracking YAML 配置

`configs/tasks/pcc_trajectory_tracking.yaml` 和 `configs/tasks/mujoco_trajectory_tracking.yaml` 共享以下字段：

| 字段                                          | 类型         | 说明                               |
|-----------------------------------------------|--------------|------------------------------------|
| `schema_version`                              | int          | 配置 schema 标记。                 |
| `name`                                        | string       | 任务标识。                         |
| `robot.config_path`                           | path         | 机器人 YAML。                      |
| `simulation.dt`                               | float, s     | 控制器 timestep。                  |
| `simulation.max_steps`                        | int          | tracking loop 最大迭代次数。       |
| `simulation.stop_on_completion`               | bool         | 目标序列完成后是否停止。           |
| `simulation.position_limit_rad`               | float, rad   | 电机位置裁剪限幅。                 |
| `simulation.initial_motor_position_rad`       | list[float]  | 初始电机向量。                     |
| `controller.type`                             | string       | 控制器标识。当前值为 `differential_ik`。 |
| `controller.damping`                          | float        | 阻尼最小二乘 damping。             |
| `controller.position_gain`                    | float        | 末端位置反馈增益。                 |
| `controller.max_motor_velocity_rad_s`         | float        | 电机速度限幅。                     |
| `controller.position_tolerance_m`             | float, m     | 目标完成容差。                     |
| `trajectory.type`                             | string       | `circle`、`figure-eight`、`ellipse`、`line`、`square`、`lissajous` 或 `helix`。 |
| `trajectory.samples`                          | int          | 目标采样数量。                     |
| `trajectory.radius_m`                         | float, m     | 通用尺度参数；对 circle / helix 直接使用，也可作为其他轨迹的默认尺度。 |
| `trajectory.placement.center_mode`            | string       | `straight_tip_xy`、`straight_tip` 或 `explicit`。 |
| `trajectory.placement.z_mode`                 | string       | `straight_tip_minus_radius`、`center` 或 `explicit`。 |
| `trajectory.placement.plane`                  | string       | `xy`、`xz` 或 `yz`。               |
| `trajectory.placement.yaw_deg`                | float, deg   | 在所选平面内的旋转角。             |
| `trajectory.placement.offset_xyz_m`           | list[float]  | 在最终中心点上叠加的三维偏移。     |
| `trajectory.placement.center_xyz_m`           | list[float]  | `center_mode: explicit` 时使用的显式中心点。 |
| `trajectory.placement.z_value_m`              | float, m     | `z_mode: explicit` 时使用的显式高度。 |
| `trajectory.shape.radius_x_m`                 | float, m     | ellipse / figure-eight / lissajous 的 x 向半轴或振幅。 |
| `trajectory.shape.radius_y_m`                 | float, m     | ellipse / figure-eight / lissajous 的 y 向半轴或振幅。 |
| `trajectory.shape.length_m`                   | float, m     | line 的总长度。                    |
| `trajectory.shape.side_length_m`              | float, m     | square 的边长。                    |
| `trajectory.shape.turns`                      | float        | helix 的圈数。                     |
| `trajectory.shape.pitch_m`                    | float, m     | helix 的螺距。                     |
| `trajectory.shape.lissajous_frequency_x`      | int          | lissajous 在 x 向的频率系数。      |
| `trajectory.shape.lissajous_frequency_y`      | int          | lissajous 在 y 向的频率系数。      |
| `trajectory.shape.lissajous_phase_deg`        | float, deg   | lissajous 的相位差。               |
| `visualization.mode`                          | string       | `static` 或 `animation`。          |
| `visualization.show`                          | bool         | 为 true 时打开 matplotlib UI。     |
| `visualization.show_summary_after_animation`  | bool         | 动画后是否显示 summary。           |
| `visualization.animation.interval_ms`         | int          | 动画帧间隔。                       |
| `visualization.animation.stride`              | int          | 动画采样 stride。                  |
| `visualization.animation.samples_per_segment` | int          | 中心线绘制密度。                   |

轨迹字段兼容说明：

- 旧写法里的 `trajectory.center_mode`、`trajectory.z_mode`、`trajectory.plane`、`trajectory.yaw_deg` 仍然会被接受。
- 旧写法里的 `trajectory.radius_x_m`、`trajectory.radius_y_m`、`trajectory.length_m` 等 shape 字段也仍然会被接受。
- 新增字段推荐统一放到 `trajectory.placement` 和 `trajectory.shape` 下，便于后续继续扩展。

各轨迹的典型参数组合：

- `circle`：使用 `radius_m`。
- `figure-eight`：默认使用 `radius_m` 作为 x 向尺度，`0.5 * radius_m` 作为 y 向尺度；也可显式提供 `radius_x_m`、`radius_y_m`。
- `ellipse`：建议提供 `radius_x_m`、`radius_y_m`。
- `line`：建议提供 `length_m`；未提供时回退到 `2 * radius_m`。
- `square`：建议提供 `side_length_m`；未提供时回退到 `2 * radius_m`。
- `lissajous`：建议提供 `radius_x_m`、`radius_y_m`、`lissajous_frequency_x`、`lissajous_frequency_y`、`lissajous_phase_deg`。
- `helix`：使用 `radius_m`，并建议补充 `pitch_m`、`turns`。

实现约定：

- 所有轨迹最终都会被离散成 `N x 3` 的目标点列。
- 平面闭合轨迹会先在局部坐标系生成，再做平面放置和旋转。
- 曲线会按弧长重新采样，以减少不同参数化方式带来的 waypoint 密度偏差。

`configs/tasks/mujoco_trajectory_tracking.yaml` 额外包含：

| 字段                                 | 类型   | 说明                                     |
|--------------------------------------|--------|------------------------------------------|
| `mujoco_backend_config`              | path   | MuJoCo 后端 YAML。                       |
| `mujoco.target_advance_mode`         | string | `time` 或 `tolerance`。                  |
| `mujoco.feedback_mode`               | string | `mujoco_actual` 或 `pcc_command`。       |
| `mujoco.show_live_tendon_panel`      | bool   | 是否随 viewer 打开 live tendon monitor。 |
| `mujoco.live_tendon_panel_stride`    | int    | monitor 更新 stride。                    |
| `mujoco.hold_viewer_open_after_run`  | bool   | tracking 结束后是否保持 viewer 打开。    |
| `mujoco.show_summary`                | bool   | 是否显示 tracking summary figure。       |

## Structured Scene YAML

`configs/scenes/rocket_*.yaml` 描述火箭发动机腔体检修场景。场景会被同时用于两件事：生成带障碍物的 MuJoCo XML，以及给导航控制器提供中心线 clearance 查询。

| 字段                               | 类型        | 说明                                      |
|------------------------------------|-------------|-------------------------------------------|
| `schema_version`                   | int         | 配置 schema 标记。                        |
| `name`                             | string      | 场景标识，也会用于生成 XML 中的 body 名称。 |
| `description`                      | string      | 面向人的场景说明。                        |
| `builder.shell_approx_sides`       | int         | 用多少个薄 box 近似圆形/锥形内壁。        |
| `builder.shell_axial_slices`       | int         | 锥形/圆柱壳沿 z 方向的离散段数。          |
| `builder.wall_thickness_m`         | float, m    | 壳体可视/碰撞壁厚。                       |
| `builder.geom_group`               | int         | 注入 MuJoCo geom/site 使用的 group。       |
| `builder.shell_rgba`               | list[float] | 壳体默认颜色。                            |
| `builder.obstacle_rgba`            | list[float] | 障碍物默认颜色。                          |
| `builder.target_rgba`              | list[float] | 巡检目标 site 颜色。                      |
| `builder.target_radius_m`          | float, m    | 巡检目标 site 半径。                      |
| `builder.contype`                  | int         | 注入 geom 的 MuJoCo contact type。         |
| `builder.conaffinity`              | int         | 注入 geom 的 MuJoCo contact affinity。     |
| `scene.primitives[].id`            | string      | 场景 primitive 标识。                     |
| `scene.primitives[].type`          | string      | `cylindrical_shell_segment`、`frustum_shell_segment`、`cylinder_obstacle`、`box_obstacle` 或 `box_surface`。 |
| `scene.primitives[].z_min_m`       | float, m    | 壳体段起始 z。                            |
| `scene.primitives[].z_max_m`       | float, m    | 壳体段结束 z。                            |
| `scene.primitives[].radius_m`      | float, m    | 圆柱壳或圆柱障碍半径。                    |
| `scene.primitives[].radius_start_m`| float, m    | 锥形壳起点半径。                          |
| `scene.primitives[].radius_end_m`  | float, m    | 锥形壳终点半径。                          |
| `scene.primitives[].center_m`      | list[float] | 圆柱/盒子障碍中心。                       |
| `scene.primitives[].half_length_m` | float, m    | 圆柱障碍半长。                            |
| `scene.primitives[].axis`          | string      | 圆柱轴向：`x`、`y` 或 `z`。               |
| `scene.primitives[].half_size_m`   | list[float] | box 障碍半尺寸。                          |
| `scene.primitives[].rgba`          | list[float] | 该 primitive 的颜色覆盖值。               |
| `scene.inspection_targets[].id`    | string      | 巡检 waypoint 标识。                      |
| `scene.inspection_targets[].type`  | string      | `point` 或 `wall_point`。                 |
| `scene.inspection_targets[].pos_m` | list[float] | `point` 类型的显式目标点。                |
| `scene.inspection_targets[].section_id` | string | `wall_point` 引用的壳体 primitive。       |
| `scene.inspection_targets[].theta_deg`  | float, deg | `wall_point` 的圆周角。               |
| `scene.inspection_targets[].z_m`         | float, m   | `wall_point` 的轴向位置。             |
| `scene.inspection_targets[].inward_offset_m` | float, m | 目标点相对内壁向腔体内部偏移距离。 |
| `scene.work_surfaces[].id`        | string      | 作业面 frame 标识。                     |
| `scene.work_surfaces[].primitive_id` | string   | 对应的可碰撞 surface primitive。        |
| `scene.work_surfaces[].center_m`  | list[float] | 作业面 frame 原点。                     |
| `scene.work_surfaces[].normal`    | list[float] | 指向自由空间的单位法向，loader 会归一化。 |
| `scene.work_surfaces[].tangent_u` | list[float] | 作业面第一切向轴，loader 会正交化。      |
| `scene.work_surfaces[].width_m`   | float, m    | 作业面元数据宽度。                     |
| `scene.work_surfaces[].height_m`  | float, m    | 作业面元数据高度。                     |
| `scene.wipe_patches[].id`         | string      | 擦拭 patch 标识。                       |
| `scene.wipe_patches[].surface_id` | string      | patch 所属 work surface。               |
| `scene.wipe_patches[].center_m`   | list[float] | patch 中心点。                         |
| `scene.wipe_patches[].width_m`    | float, m    | patch 宽度。                            |
| `scene.wipe_patches[].height_m`   | float, m    | patch 高度。                            |

## MuJoCo Navigation YAML 配置

`configs/tasks/mujoco_navigation_rocket.yaml` 是任务 1 的默认入口，当前用于火箭发动机腔体结构化障碍中的适形导航。

| 字段                                          | 类型         | 说明                                      |
|-----------------------------------------------|--------------|-------------------------------------------|
| `schema_version`                              | int          | 配置 schema 标记。                        |
| `name`                                        | string       | 任务标识。                                |
| `mujoco_backend_config`                       | path         | MuJoCo 后端 YAML。                        |
| `robot.config_path`                           | path         | 机器人 YAML。                             |
| `scene.config_path`                           | path         | 结构化场景 YAML。                         |
| `scene.generated_xml_path`                    | path         | 注入场景后的 MuJoCo XML 输出路径。        |
| `simulation.dt`                               | float, s     | 控制器 timestep。                         |
| `simulation.max_steps`                        | int          | navigation loop 最大迭代次数。            |
| `simulation.stop_on_completion`               | bool         | 最后一个 waypoint 完成后是否停止。        |
| `simulation.position_limit_rad`               | float, rad   | 电机位置裁剪限幅。                        |
| `simulation.initial_motor_position_rad`       | list[float]  | 初始电机向量。                            |
| `controller.type`                             | string       | 当前值为 `navigation_differential_ik`。    |
| `controller.damping`                          | float        | 阻尼最小二乘 damping。                    |
| `controller.position_gain`                    | float        | 末端目标跟踪增益。                        |
| `controller.clearance_gain`                   | float        | 中心线避障/保距修正增益。                 |
| `controller.clearance_min_m`                  | float, m     | 最小允许 clearance。                      |
| `controller.avoidance_influence_m`            | float, m     | clearance 低于该值时开始产生避障项。      |
| `controller.max_motor_velocity_rad_s`         | float        | 电机速度限幅。                            |
| `controller.position_tolerance_m`             | float, m     | waypoint 完成容差。                       |
| `controller.centerline_samples_per_segment`   | int          | 每段中心线 clearance 采样数量。           |
| `controller.finite_difference_step_rad`       | float        | 中心线点 Jacobian 有限差分步长。          |
| `mission.type`                                | string       | 当前值为 `ordered_inspection`。            |
| `mission.waypoint_ids`                        | list[string] | 按顺序访问的 `inspection_targets[].id`。   |
| `mission.terminate_on_clearance_violation`    | bool         | clearance 低于最小值时是否立即终止。      |
| `mujoco.feedback_mode`                        | string       | `mujoco_actual` 或 `pcc_command`。         |
| `mujoco.show_live_tendon_panel`               | bool         | 是否随 viewer 打开 live tendon monitor。  |
| `mujoco.live_tendon_panel_stride`             | int          | monitor 更新 stride。                     |
| `mujoco.hold_viewer_open_after_run`           | bool         | navigation 结束后是否保持 viewer 打开。   |
| `mujoco.show_summary`                         | bool         | 当前仅输出命令行 summary 指标。           |
| `visualization.show`                          | bool         | 为 true 时允许打开 viewer。               |

## MuJoCo Wiping YAML 配置

`configs/tasks/mujoco_wiping_board.yaml` 是任务 2 的默认入口，用于在指定作业面上执行 raster 擦拭和法向接触力调节。

| 字段                                          | 类型         | 说明                                      |
|-----------------------------------------------|--------------|-------------------------------------------|
| `schema_version`                              | int          | 配置 schema 标记。                         |
| `name`                                        | string       | 任务标识。                                |
| `mujoco_backend_config`                       | path         | MuJoCo 后端 YAML。                         |
| `robot.config_path`                           | path         | 机器人 YAML。                              |
| `scene.config_path`                           | path         | 包含 work surface 和 patch 的场景 YAML。    |
| `scene.generated_xml_path`                    | path         | 注入场景和 tool pad 后的 XML 输出路径。     |
| `tool.type`                                   | string       | `spherical_pad` 或 `capsule_pad`。          |
| `tool.radius_m`                               | float, m     | contact pad 半径。                         |
| `tool.length_m`                               | float, m     | capsule pad 长度；sphere 可省略。           |
| `tool.offset_m`                               | list[float]  | tool body 相对 tip body/site 的偏移。       |
| `tool.rgba`                                   | list[float]  | pad 和 contact site 颜色。                 |
| `tool.geom_name`                              | string       | contact pad geom 名称。                    |
| `tool.body_name`                              | string       | 注入的 tool body 名称。                    |
| `tool.contact_site_name`                      | string       | tool contact site 名称。                   |
| `simulation.dt`                               | float, s     | 控制器 timestep。                          |
| `simulation.max_steps`                        | int          | wiping loop 最大迭代次数。                 |
| `simulation.stop_on_completion`               | bool         | 最后一个 waypoint 完成后是否停止。          |
| `simulation.position_limit_rad`               | float, rad   | 电机位置裁剪限幅。                         |
| `simulation.initial_motor_position_rad`       | list[float]  | 初始 9 维电机向量。                        |
| `controller.type`                             | string       | 当前值为 `hybrid_force_position`。          |
| `controller.damping`                          | float        | 阻尼最小二乘 damping。                     |
| `controller.tangent_position_gain`            | float        | 作业面切向位置跟踪增益。                   |
| `controller.normal_force_gain`                | float        | 法向力误差到法向速度的增益。               |
| `controller.normal_position_gain`             | float        | 接触距离/压入量 proxy 的法向位置增益。      |
| `controller.target_normal_force_n`            | float, N     | 目标法向接触力。                           |
| `controller.force_proxy_stiffness_n_m`        | float, N/m   | 用压入量估算法向力时的 proxy 刚度。         |
| `controller.target_contact_distance_m`        | float, m     | pad 表面相对作业面的目标 signed distance；负值表示压入。 |
| `controller.max_normal_velocity_m_s`          | float, m/s   | 法向速度限幅。                             |
| `controller.max_tangent_velocity_m_s`         | float, m/s   | 切向速度限幅。                             |
| `controller.max_motor_velocity_rad_s`         | float        | 输出 motor velocity 限幅。                 |
| `controller.position_tolerance_m`             | float, m     | 位置误差记录/完成容差。                    |
| `controller.force_tolerance_n`                | float, N     | 力误差记录容差。                           |
| `controller.max_contact_force_n`              | float, N     | 超过该力时 runtime 停止。                  |
| `controller.contact_loss_tolerance_steps`     | int          | contact phase 连续失联步数容忍度。         |
| `controller.finite_difference_step_rad`       | float        | motor Jacobian 有限差分步长。              |
| `motion.type`                                 | string       | 当前值为 `raster_wipe`。                   |
| `motion.surface_id`                           | string       | 引用 `scene.work_surfaces[].id`。          |
| `motion.patch_id`                             | string       | 可选 patch 元数据引用。                    |
| `motion.center_m`                             | list[float]  | raster 中心点；省略时使用 surface 中心。    |
| `motion.width_m`                              | float, m     | raster 宽度。                              |
| `motion.height_m`                             | float, m     | raster 高度。                              |
| `motion.line_count`                           | int          | raster 行数。                              |
| `motion.samples_per_line`                     | int          | 每行 waypoint 数。                         |
| `motion.approach_offset_m`                    | float, m     | approach waypoint 沿 surface normal 外偏距离。 |
| `motion.contact_offset_m`                     | float, m     | contact waypoint 的 pad 表面 signed offset；runtime 会自动加上 pad 半径。 |
| `motion.waypoint_tolerance_m`                 | float, m     | waypoint 切换容差。                        |
| `mujoco.feedback_mode`                        | string       | `mujoco_actual` 或 `pcc_command`。         |
| `mujoco.show_live_tendon_panel`               | bool         | 是否随 viewer 打开 live tendon monitor。   |
| `mujoco.live_tendon_panel_stride`             | int          | monitor 更新 stride。                      |
| `mujoco.show_live_force_panel`                | bool         | 是否随 viewer 打开 wiping force monitor；仅 wiping runtime 使用。 |
| `mujoco.live_force_panel_stride`              | int          | force monitor 更新 stride。                |
| `mujoco.live_force_panel_history_points`      | int          | force monitor 保留的历史采样点数量。       |
| `mujoco.hold_viewer_open_after_run`           | bool         | wiping 结束后是否保持 viewer 打开。        |
| `mujoco.show_summary`                         | bool         | 当前仅输出命令行 summary 指标。            |
| `visualization.show`                          | bool         | 为 true 时允许打开 viewer。                |

## MuJoCo segment-2DOF follower 模型

默认的 `configs/mujoco.yaml` 使用 `model.type: distributed_links`，也就是原有三段、每段四个
物理 link 的 MuJoCo 模型。它保留 24 个物理弯曲 DOF 和既有 `tendon_position` 行为。

`configs/mujoco_segment_2dof.yaml` 使用 `model.type: segment_2dof_followers`。它的物理
`q` 是 6 维，顺序如下：

```text
segment_1_x, segment_1_y,
segment_2_x, segment_2_y,
segment_3_x, segment_3_y
```

每个 x/y 对表示一段的总 hinge 角。运行时 follower mocap body 由 PCC 模型采样得到，每段采样数量由
`model.follower_samples_per_segment` 控制。这些 follower 视觉/碰撞体不会增加 MuJoCo 物理 DOF。

额外的 `model.*` 字段：

| 字段 | 类型 | 说明 |
|-------|------|---------|
| `model.follower_collision` | bool | 生成 follower 碰撞 capsule。 |
| `model.follower_visuals` | bool | 在视觉 XML 中生成 follower 视觉 capsule。 |
| `model.contact_force_projection` | bool | 使用 follower 接触扳手投影作为 wiping 力反馈。 |
| `model.apply_projected_qfrc` | bool | 将投影后的广义力写入 6 个 segment DOF；默认 false。 |

生成已提交的 2DOF XML 资产：

```powershell
python scripts/build_mujoco_segment_2dof_model.py --config configs/mujoco_segment_2dof.yaml
```

当前限制：接触投影使用有限差分 Jacobian；`apply_projected_qfrc` 默认关闭；follower 接触力目前主要作为
wiping 力反馈信号使用，若要通过 `qfrc_applied` 加入动力学反作用，应谨慎启用并单独验证。

## 高级控制扩展

本节区分两类入口：

- scenario 主入口：`configs/scenarios/*.yaml` 与 `SimulationApplication`。
- 旧任务 runtime：`configs/tasks/*.yaml` 与 `src/continuum_sim/runtime/mujoco_*_runtime.py`。

如果某个模式只在旧 runtime 中启用，本节会明确标注。

### DMP Tracking 轨迹

scenario tracking 支持 `trajectory.type: dmp`：

```yaml
trajectory:
  type: dmp
  samples: 100
  demo_path: path/to/demo.csv
  start_xyz_m: [0.0, 0.0, 0.12]
  goal_xyz_m: [0.04, 0.0, 0.15]
  tau: 1.0
  basis_count: 24
```

`demo_path` 可以指向包含 `x,y,z` 或 `time,x,y,z` 列的 CSV/text 文件，也可以指向包含 `time` 和
`trajectory` 数组的 NPZ 文件。

### Navigation CBF 模式

scenario navigation 支持：

```yaml
task:
  type: navigation
  navigation_control_type: navigation_cbf_qp
  navigation_cbf_gain: 4.0
  navigation_cbf_influence_distance_m: 0.025
```

当前实现使用小规模 NumPy 投影处理 CBF 半空间约束，对 `WaypointTrackingController` /
`CoordinatedTrackingController` 生成的 whole-body command 做后处理。旧 `controller.type:
navigation_cbf_qp` 写法在 scenario 中也会被识别为 `navigation_control_type:
navigation_cbf_qp`。未显式配置 `navigation_cbf_influence_distance_m` 时，控制器会使用略大于
`min_clearance_m` 的默认影响距离提前介入。

### PCC 降阶动力学

`configs/dynamics/pcc_reduced.yaml` 保存实验性动力学擦拭控制器使用的工程估计参数。
旧 `run_mujoco_wiping(...)` runtime 和 scenario 主入口都可调用该模型。scenario 写法：

```yaml
task:
  type: wiping
  wiping_control_type: dynamic_adaptive_impedance
  dynamics_config_path: ../dynamics/pcc_reduced.yaml
```

scenario 中的系统级控制器会把预测的 executor `qdot` 映射为相容 tendon-rate，并在 metadata 中记录
`wiping_dynamic_system_controller_active`。

| 字段 | 类型 | 说明 |
|-------|------|---------|
| `dynamics.segment_masses_kg` | list[float] | 每个 PCC 段的质量估计。 |
| `dynamics.bending_stiffness` | list[float] | 每段弯曲刚度估计。 |
| `dynamics.axial_stiffness` | list[float] | 每段轴向应变刚度估计。 |
| `dynamics.damping` | list[float] | 9 维 PCC 广义坐标中的对角阻尼。 |
| `dynamics.mass_regularization` | float | 求解质量矩阵时使用的小正则项。 |
| `dynamics.centerline_samples_per_segment` | int | 质量矩阵 Jacobian 积分时每段中心线采样点数。 |

降阶模型为：

```text
M(q) qddot + D qdot + K q = tau + J_tip(q).T F_contact
```

### Wiping 控制器模式

- `contact_distance`：scenario 主入口的默认接触距离修正模式。
- `hybrid_force_position`：旧 runtime 的运动学级切向位置/法向力控制器；scenario 中作为接触修正模式保留。
- `dynamic_adaptive_impedance`：先使用 PCC 降阶动力学预测降阶状态速度，再映射回 scenario tendon-rate。
- `contact_triggered_admittance`：接触后启用目标法向力，使用导纳状态修正目标点并输出期望 TCP 速度。

接触导纳示例：

```yaml
task:
  type: wiping
  wiping_control_type: contact_triggered_admittance
  contact_admittance:
    target_normal_force_n: 1.5
    contact_force_threshold_n: 0.1
    tangent_tolerance_m: 0.001
    force_tolerance_n: 0.08
```

### Engine Cleaning 控制器

`task.type: engine_cleaning` 默认使用 task-space engine cleaning controller。可通过
`engine_cleaning_control` 覆盖增益：

```yaml
task:
  type: engine_cleaning
  engine_cleaning_control:
    tangential_position_gain: 8.0
    normal_position_gain: 3.0
    normal_force_gain: 0.001
    approach_position_gain: 5.0
    retreat_position_gain: 5.0
    max_tcp_speed_mps: 0.03
    max_normal_speed_mps: 0.01
    waypoint_tolerance_m: 0.002
    max_contact_force_n: 5.0
    force_deadband_n: 0.05
```

旧 runtime 动力学实验任务示例：

```text
configs/tasks/mujoco_wiping_board_dynamic.yaml
```

### 验收脚本

专用报告脚本位于 `scripts/`：

```powershell
python scripts/test_indicator_3_1_positioning.py --result-npz output/runs/<run>/result.npz
python scripts/test_indicator_3_3_disturbance.py --result-npz output/runs/<run>/result.npz
python scripts/test_indicator_3_3_force_tracking.py --result-npz output/runs/<run>/result.npz
```

每个脚本都会写出 Markdown 报告、PNG 曲线图、指标表、阈值结论和 CNAS/CMA 盖章预留区。推荐先通过
scenario 生成 `result.npz`，再传入脚本分析：

```powershell
python scripts/run_scenario.py configs/scenarios/single_mujoco_wiping.yaml
python scripts/test_indicator_3_3_force_tracking.py --result-npz output/runs/<run>/result.npz
```

如果不提供 `--result-npz`，脚本会尝试无窗口运行对应 MuJoCo 任务。动态阻抗模式需要在每个控制步计算
PCC 质量矩阵，耗时会明显长于只分析已保存 NPZ。
### Scenario 擦拭力控策略

当前 scenario 主线支持通过 `task.force_strategy.type` 显式选择擦拭力控策略。默认不改变已有控制器行为。

| 字段 | 类型 | 说明 |
|------|------|------|
| `task.wiping_control_type` | string | 擦拭控制类型。支持 `contact_distance`、`hybrid_force_position`、`dynamic_adaptive_impedance`、`contact_triggered_admittance`。 |
| `task.force_strategy.type` | string | 主线策略选择。支持 `contact_distance`、`kinematic_hybrid`、`dynamic_adaptive_impedance`、`contact_triggered_admittance`。未配置时会根据 `wiping_control_type` 自动推导。 |
| `task.dynamics_config_path` | path | `dynamic_adaptive_impedance` 使用的 PCC 降阶动力学参数路径。 |
| `task.admittance.*` | mapping | `contact_triggered_admittance` 使用的导纳控制参数。 |

动态策略示例：

```yaml
task:
  type: wiping
  wiping_control_type: dynamic_adaptive_impedance
  dynamics_config_path: ../dynamics/pcc_reduced.yaml
  force_strategy:
    type: dynamic_adaptive_impedance
```

导纳策略示例：

```yaml
task:
  type: wiping
  wiping_control_type: contact_triggered_admittance
  force_strategy:
    type: contact_triggered_admittance
  admittance:
    target_normal_force_n: 1.5
    contact_force_threshold_n: 0.1
    tangent_tolerance_m: 0.001
    force_tolerance_n: 0.08
    stable_steps_required: 3
    max_steps_per_target: 80
    admittance_mass: 1.0
    admittance_damping: 20.0
    admittance_stiffness: 5.0
    admittance_clip_m: 0.012
```

`contact_triggered_admittance` 保留为有意义的可选力控策略，但不再使用独立的
`single_mujoco_wiping_admittance.yaml`。请在 `single_mujoco_wiping.yaml` 中同时切换
`wiping_control_type` 与 `force_strategy.type`；该文件已经保留完整 `task.admittance` 参数块。

运行产物会额外记录：

```text
measured_force_n
normal_force_source
admittance_position_m
admittance_velocity_m_s
dynamic_normal_correction_m
wiping_dynamic_active
```

### Wiping 轨迹与跟踪调参

scenario 主线中的 `wiping` 任务复用统一底层。任务时序保留在 `task.tracking_control`，底层调参位于
`configs/control/spatial_low_level.yaml`：

| 字段 | 建议用途 |
| ---- | -------- |
| `low_level_control.executor_position_gain` | 末端位置误差到目标速度的共享比例增益。 |
| `low_level_control.feedforward_gain` | position 模式的轨迹/waypoint 前馈比例；不缩放 velocity 模式直接命令。 |
| `low_level_control.feedforward_speed_mps` | waypoint 模式沿下一 waypoint 的前馈速度。 |
| `low_level_control.max_target_speed_mps` | 所有任务共用的末端目标速度上限。 |
| `low_level_control.tendon_regularization_weight` | tendon 速度正则权重。增大后动作更慢、更稳。 |
| `low_level_control.nominal_damping` / `maximum_damping` | 奇异或病态映射附近的阻尼。 |
| `low_level_control.decouple_arm_singularity` | 是否对单臂奇异性进行解耦保护。 |

`configs/scenes/wiping_board.yaml` 中有两类位置需要区分：

- `scene.primitives[].center_m`：MuJoCo 中可见黑板和边框几何体的位置。
- `scene.work_surfaces[].center_m` / `scene.wipe_patches[].center_m`：擦拭轨迹和接触法向使用的作业面位置。

对于当前黑板：

```yaml
board_surface_geom:
  center_m: [0.050, 0.0, 0.095]
  half_size_m: [0.0025, 0.035, 0.030]
board_surface:
  center_m: [0.0475, 0.0, 0.095]
  normal: [-1.0, 0.0, 0.0]
```

这表示黑板主体中心在 `x=0.050`，厚度半宽为 `0.0025`，面向执行臂的前表面在 `x=0.0475`。`normal` 指向自由空间侧。擦拭路径按下面公式生成：

```text
contact_origin = center_m + contact_offset_m * normal
approach_point = contact_origin + approach_offset_m * normal
```

因此当 `normal: [-1, 0, 0]`、`contact_offset_m: -0.0025`、`approach_offset_m: 0.005` 时，接触点会进入黑板 `2.5 mm`，接近点会退到自由空间侧 `2.5 mm`。

通过 `wiping_path` 生成的 scenario 会把 `work_surfaces[].center_m` 作为作业面平面点传入控制器。接触误差使用该平面的 signed distance，而不是 MuJoCo box 最近面的距离；这样即使末端越过黑板厚度中面，法向误差方向也不会因为最近面切换而反向。
## PCC–MuJoCo 物理参数统一约定

当前三段空间连续体机械臂使用下列统一参数语义：

| 参数 | 当前值 | 含义 |
|------|--------|------|
| `tendon_radius` / `tendon_radius_m` | `0.005 m` | 腱中心线相对 backbone 轴线的径向偏置。PCC 腱长耦合使用该值。 |
| `radial_offset` / `radial_offset_m` | `0.005 m` | 单根物理腱的中心线偏置；应与对应孔位半径一致。 |
| `collision_radius` / `collision_radius_m` | `0.00625 m` | MuJoCo link capsule 的碰撞/惯量几何半径，不参与 PCC 腱长耦合。 |
| `mass` / `mass_kg` | `0.00989929 kg` | 单个 40 mm 段的总质量。四 link 模型为每个 link 分配约 `0.00247482 kg`。 |
| `bending_stiffness` / `bending_stiffness_n_m2` | `0.0002 N·m²` | 连续体弯曲刚度 `EI`，不是 MuJoCo hinge 的转动刚度。 |

旧机器人 YAML 未提供 `collision_radius` 时，加载器回退到
`tendon_radius`；未提供段质量时，MuJoCo 生成器保留原有的基于 geom
density 的质量计算；未提供段 `bending_stiffness` 时，hinge 继续继承
MuJoCo backend 配置中的全局默认刚度。

### 刚度换算

MuJoCo 分布式 link 模型按以下关系从连续体 `EI` 得到单关节刚度：

```text
k_joint = EI / link_length
```

当前 `EI = 0.0002 N·m²`、`link_length = 0.01 m`，因此每个单轴 flexure
hinge 保持为 `0.02 N·m/rad`。

PCC 状态使用曲率 `kappa`，因此曲率坐标中的对角刚度为：

```text
K_kappa = EI * segment_length
```

当前每段 `segment_length = 0.04 m`，对应 `K_kappa = 8e-6`。轴向和扭转
刚度没有按该公式映射，因为当前 distributed-link MuJoCo 臂没有对应的
独立轴向或扭转关节自由度。

### 双臂单轴柔性关节拓扑

`configs/robots/dual_arm_3seg.yaml` 的
`dual_robot.flexure_joint_axis_pattern` 定义真实小节近端柔性铰链的局部材料轴：

```yaml
flexure_joint_axis_pattern:
  - [0.0, 1.0, 0.0]
  - [1.0, 0.0, 0.0]
```

executor 和 observer 都有 12 个刚性小节，每个小节只生成一个 proximal
hinge。全臂第 1、3、5…小节绕局部 Y 轴转动，对应局部 X 方向弯曲；第
2、4、6…小节绕局部 X 轴转动，对应局部 Y 方向弯曲。第一个小节与基座之间
同样保留 Y 轴转动自由度。轴序列沿全臂连续取模，两条臂完全一致。

这里的 axis 属于对应 body 的局部坐标系，会随上游刚体姿态一起旋转；它不是
运行期间固定在世界坐标系中的转轴。

#### 单轴铰链的等效刚度

每个 segment 的 `bending_stiffness` 仍表示连续模型的抗弯刚度 `EI`，单位为
`N·m²`，不是单个 MuJoCo hinge 的转动刚度。生成器会在每个 segment 内统计与
当前 hinge 平行或反向平行的串联关节数量 `n_axis`，并计算：

```text
k_joint = n_axis * EI / segment_length
k_segment = k_joint / n_axis = EI / segment_length
```

当前每个 40 mm segment 含 Y/X/Y/X 四个小节，因此 X、Y 方向分别有两个串联
hinge。使用 `EI = 0.0002 N·m²` 时：

```text
k_joint = 2 * 0.0002 / 0.04 = 0.01 N·m/rad
k_segment = 0.01 / 2 = 0.005 N·m/rad
```

`configs/mujoco_dual.yaml` 中 `joints.hinge.stiffness: 0.01` 是缺少段级
`bending_stiffness` 时的 fallback；正常生成的柔性关节仍以机器人 YAML 中的
segment `EI` 和上述公式为准，并显式写入每个 `<joint>`。

如果继续沿用双自由度模型时期的 `0.02 N·m/rad` 单关节值，每个方向只有两个
串联关节，会使段级等效刚度变成 `0.01 N·m/rad`，即 PCC 目标值的两倍。

刚度会直接影响肌腱控制：刚度越大，相同肌腱长度增量需要的执行器力越大，实际
肌腱速度和末端速度越容易落后于 PCC 命令；刚度降低则会提高响应，但也可能增加
形变、过冲和振荡。本修正只恢复段级等效刚度，不改变 pivot、阻尼、关节范围、
质量、肌腱路径或控制增益。

### MuJoCo 末端 site

`scripts/build_mujoco_dual_arm_model.py` 在最后一个 link 的局部坐标系中定义：

```xml
<site name="executor_tip" pos="0 0 0.01" />
```

`observer_tip` 使用相同规则。局部 Z 方向的 10 mm 位移等于最后一根 link
自身的长度，site 位于该 link 的远端；它不是机械臂实体之外额外增加的
10 mm 工具偏置。三段、每段四根 10 mm link 的直线总长仍为 120 mm。
