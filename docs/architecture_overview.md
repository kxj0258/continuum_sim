# 架构概览

## 可组合 spatial system

当前主架构由 `RobotAssemblyConfig + EngineSceneConfig` 组合：

```text
RobotAssemblyConfig + EngineSceneConfig
        -> ControlLayout
        -> RobotSystemState
        -> CoordinatedTrackingController
        -> world base twist + named tendon-length rates
        -> MujocoSystemBackend
        -> prescribed freejoint pose + absolute tendon-length targets
```

单臂和双臂复用同一 controller、backend、simulation loop 和 engine scene
adapter，仅由 assembly 配置决定启用哪些命名臂。单臂布局为15D，双臂布局为
24D。

spatial 控制链不再包含电机、卷盘或减速器映射。奇异规避基于
tendon-to-shape 与任务 Jacobian 的 SVD。engine 实时 clearance 暂时使用
`primitive_collision_geoms`，visual mesh 不参与控制查询。

坐标系和命令语义以 `docs/coordinate_conventions.md` 为准。

## Scenario应用层

推荐入口为：

```text
ScenarioConfig
  -> SimulationApplication
  -> assembly/backend/scene/task/hooks
  -> SimulationLoop
```

`configs/scenarios/`分别提供单臂、双臂、analytic、MuJoCo和engine组合。旧 CLI
入口已删除，不再决定模块边界。

## 旧版研究模块

早期模块使用 PCC 运动学、differential IK 和可选 MuJoCo 降阶后端建模三段
腱驱连续体机械臂。它们不再构成新的 system runtime 公共接口。

## 分层

```text
YAML 配置
  -> 机器人与腱模型
  -> 驱动映射
  -> PCC 运动学
  -> differential IK
  -> analytic 或 MuJoCo 后端
  -> 可视化与运行命令
```

## 主要模块

- `continuum_sim.model`：机器人段参数、物理腱路径，以及物理腱长与 PCC 状态 `q` 之间的耦合矩阵。
- `continuum_sim.actuation`：根据 spool radius、gear ratio、direction 和 zero offset，在电机位置/速度与腱长/腱速之间转换。
- `continuum_sim.kinematics`：PCC 变换、中心线采样、末端位置有限差分 Jacobian，以及 `ContinuumKinematicsChain`。
- `continuum_sim.control`：阻尼最小二乘 differential IK 和离线 tracking 仿真。
- `continuum_sim.backends`：`AnalyticBackend`、`MujocoBackend`、共享 backend protocol/state，以及 PCC 到 MuJoCo target 的转换。
- `continuum_sim.scenes`：结构化场景 YAML、clearance 查询原语，以及把障碍物注入 MuJoCo XML 的 builder。
- `continuum_sim.tasks`：PCC/MuJoCo tracking 与 MuJoCo navigation 的 YAML loader，以及目标轨迹/巡检 waypoint 生成。
- `continuum_sim.runtime`：MuJoCo tracking/navigation 运行编排和共享 viewer helper。
- `continuum_sim.visualization`：PCC、motor-chain、MuJoCo tendon monitor 和 tracking 结果绘图工具。

当前 tracking 任务的目标轨迹生成已经从单一圆/8 字扩展为一组离散空间 waypoint 生成器，支持 `circle`、`figure-eight`、`ellipse`、`line`、`square`、`lissajous` 和 `helix`，并通过 `trajectory.placement` 与 `trajectory.shape` 两层配置描述轨迹放置和形状参数。

## 命令边界

当前维护的运行入口是 `scripts/run_scenario.py`：

```powershell
python scripts/run_scenario.py configs/scenarios/single_analytic_smoke.yaml
python scripts/run_scenario.py configs/scenarios/single_mujoco_smoke.yaml
python scripts/run_scenario.py configs/scenarios/dual_mujoco_tracking.yaml
python scripts/run_scenario.py configs/scenarios/single_engine_tracking.yaml
```

## 结构化导航流程

火箭发动机腔体检修任务新增了一条 MuJoCo navigation 链路：

```text
scene YAML
  -> shell/obstacle clearance primitives
  -> generated MuJoCo XML with chamber geoms
  -> ordered inspection waypoints
  -> tip tracking + centerline clearance differential IK
  -> tendon-position or joint-position MuJoCo control
```

`configs/scenes/rocket_*.yaml` 定义腔体内壁、筋条、凸台和喷注器柱等结构；同一份场景既用于生成 MuJoCo XML，也用于控制器查询机械臂中心线到环境的最小 clearance。默认任务入口是 `configs/tasks/mujoco_navigation_rocket.yaml`。

## 作业面擦拭流程

板面擦拭任务复用 structured scene 和 XML builder，但把环境查询从 clearance 扩展到 planar work surface：

```text
scene YAML
  -> collidable board/frame geoms + surface/patch metadata
  -> generated MuJoCo XML with tip-mounted contact pad
  -> raster wipe waypoints in the surface frame
  -> 切向位置跟踪 + 法向力/接触 proxy 调节
  -> tendon-position MuJoCo control
```

`configs/scenes/wiping_board.yaml` 定义可碰撞作业面、辅助边框、`work_surfaces` 和 `wipe_patches`；`configs/tasks/mujoco_wiping_board.yaml` 绑定 tool pad、raster motion、hybrid force-position controller 和 MuJoCo runtime 记录字段。

当 MuJoCo viewer 启用且 `mujoco.show_live_force_panel` 为 true 时，wiping runtime 会在同一循环中创建 `WipingForceMonitorPanel`，按 stride 显示法向接触力、目标力、force error、contact proxy、phase 和 waypoint 状态；它与 live tendon panel 独立，可同时打开。

运行行为写在 scenario YAML 中，这样实验可以通过提交 YAML 复现，而不是依赖临时命令行覆盖项。

## 规范配置

- `configs/scenarios/`：推荐运行入口，组合机器人、后端、场景、任务、runtime 和 hooks。
- `configs/robot_3seg.yaml`：几何、物理腱、电机和执行限幅。
- `configs/pcc.yaml`：analytic PCC 后端设置。
- `configs/mujoco.yaml`：MuJoCo XML 路径、solver、控制模式、actuator、sensor、viewer 和 overlay。
- `configs/tasks/pcc_trajectory_tracking.yaml`：PCC tracking 运行配置。
- `configs/tasks/mujoco_trajectory_tracking.yaml`：MuJoCo tracking 运行配置。
- `configs/tasks/mujoco_navigation_rocket.yaml`：结构化障碍 navigation 运行配置。
- `configs/tasks/mujoco_wiping_board.yaml`：作业面擦拭 hybrid force-position 运行配置。
- `configs/scenes/wiping_board.yaml`：板面作业场景和 patch 元数据。

## 离线 PCC 流程

```text
motor_position
  -> motor_position_to_tendon_delta
  -> physical_tendon_delta_to_q
  -> forward_kinematics
  -> tip position error
  -> motor_position_jacobian
  -> damped least-squares motor velocity
```

这条路径确定性强、运行快，并由 `core` 测试标记覆盖。

## MuJoCo 流程

```text
tracking target
  -> differential IK motor velocity
  -> motor/tendon integration
  -> MuJoCo control vector
  -> MujocoBackend.step()
  -> tip pose, segment poses, sensors, force history
```

`configs/mujoco.yaml` 选择 `tendon_position` 或 `position_joint` 模式。MuJoCo 是可选依赖；需要 MuJoCo 的测试在未安装该包时会干净跳过。

## Segment-2DOF follower MuJoCo 模型

`MujocoBackend` 支持两种模型拓扑：

- `distributed_links`：现有默认模型。它有三段，每段四个物理 link，每个 link 两个 hinge joint，以及 9 个 tendon-position actuator。
- `segment_2dof_followers`：并行的 2DOF 降阶模型，每段只有两个物理 hinge DOF。6 个物理 `q` 是各段总 hinge 角，顺序为 `segment_1_x`、`segment_1_y`、...、`segment_3_y`。

对于 `segment_2dof_followers`，运行时 follower body 是名为 `follower_segment_<i>_sample_<k>` 的 MuJoCo mocap body。它们的位姿由 `continuum_sim.model.segment_followers` 中的 PCC 模型采样得到，并在 reset、step 和 replay 时写入 `data.mocap_pos` / `data.mocap_quat`。这些 body 只提供视觉/碰撞采样，不增加广义坐标。

擦拭力反馈路径为：

```text
MuJoCo contacts
  -> follower 碰撞 geom 筛选
  -> mj_contactForce 接触坐标系扳手
  -> 世界坐标系力/力矩
  -> 相对 6 维 segment q 的 follower/contact-point 有限差分 Jacobian
  -> normal_force_n + projected_generalized_force_q
  -> 混合力位擦拭控制器
```

`mujoco_actual` 擦拭反馈优先使用 `mujoco_follower_contact_projection`；如果没有 follower 接触，则回退到已有 tool-pad MuJoCo 接触力，再回退到距离 proxy。`apply_projected_qfrc` 可用但默认关闭，因为当前首要用途是力反馈，而不是接触反作用动力学。

## 运行产物导出

Scenario 产物导出刻意放在 runtime 控制循环之外。运行命令会先返回内存中的 result dataclass；当 scenario 启用 artifacts 时，`continuum_sim.io.scenario_artifacts` 会创建 `output/runs/<scenario>_<timestamp>/`，写出 NPZ 数据、metadata、配置副本、静态 PNG 曲线图、生成的 scene XML，以及可选的 `videos/simulation.gif`。

对 MuJoCo 结果，回放视频导出被隔离在 `scripts/export_replay_video.py`，这样渲染器会在新进程中创建。导出器用保存的 `qpos/qvel` 复现归档的 `model/scene.xml`，尺寸使用 `configs/mujoco.yaml` 的 `rendering.offscreen_*`，相机使用 `viewer.camera`，与 passive viewer 的前侧斜视角保持一致。如果视频导出失败，原因会记录到 `videos/video_error.txt`，数值和曲线产物仍会保留。

如果要在完整运行前诊断某台机器上的渲染环境问题，可以先运行 `scripts/check_mujoco_offscreen_renderer.py`。它会直接探测配置中的 XML，并报告失败发生在 XML 加载、渲染器创建还是帧渲染阶段。
