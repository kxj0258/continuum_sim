# 主线迁移方案

本文把旧基线和实验功能统一整理到当前 scenario 主线下。目标不是替换已经稳定的控制器，而是把未接入的能力做成可选策略，让每个场景通过 YAML 显式选择。

## 当前实现状态

本轮迁移已经完成第一步主线接入：

- 新增 `src/continuum_sim/control/wiping_force_strategies.py`，把擦拭任务的力控差异收敛为策略对象。
- `WipingController` 保持为主线控制器，但不再硬编码所有法向修正逻辑。
- 默认 `contact_distance` 和 `hybrid_force_position` 行为保持兼容。
- `dynamic_adaptive_impedance` 现在会在 scenario 主线中加载 `configs/dynamics/pcc_reduced.yaml`，执行 PCC 降阶动力学预测，并把预测修正和诊断写入 metadata。
- `contact_triggered_admittance` 现在可通过 scenario YAML 打开，使用 `ContactTriggeredAdmittanceTracker` 控制 corrected target 和 waypoint 推进。
- 新增 `configs/scenarios/single_mujoco_wiping_admittance.yaml` 作为导纳控制示例。
- recorder/artifacts 会记录 `measured_force_n`、`normal_force_source`、`admittance_position_m`、`admittance_velocity_m_s`、`dynamic_normal_correction_m`、`wiping_dynamic_active`。

仍未做的深化项：

- MuJoCo 真实接触力目前还没有直接注入 scenario `WipingController`，主线导纳策略先使用距离 proxy 估计力。
- 动态策略先作为 predictive correction 接入，未直接替换 whole-body 求解器或底层 tendon rate 生成。
- CBF-QP、DMP、移动底座位姿控制和发动机清洗 task-space 控制器仍按下文路线分阶段接入。

## 当前结论

项目里仍然保留了连续体动力学相关实现，但它没有完整进入当前 scenario 主线：

- `src/continuum_sim/dynamics/pcc_dynamics.py` 实现了 9 维 PCC 降阶动力学。
- `configs/dynamics/pcc_reduced.yaml` 保存工程估计参数。
- `src/continuum_sim/control/adaptive_impedance.py` 使用该动力学做擦拭力位控制实验。
- 旧入口 `src/continuum_sim/runtime/mujoco_wiping_runtime.py` 会在 `controller.type: dynamic_adaptive_impedance` 时真正调用动力学控制器。
- 当前主线 `configs/scenarios/*.yaml` 使用 `SimulationApplication` 和 `control.scenario_controllers.WipingController`，其中 `wiping_control_type: dynamic_adaptive_impedance` 只被记录为请求；metadata 里仍标记 `wiping_dynamic_system_controller_active: False`。

最新导入的接触触发导纳控制器也还没有接入主线：

- `src/continuum_sim/control/contact_triggered_admittance.py` 是后端无关的 waypoint/force 状态机。
- 它能在接触前关闭力控，接触后根据法向力误差生成导纳位移，并用切向误差和力误差共同决定是否推进路点。
- 当前只有单元测试直接覆盖它，scenario YAML、`WipingController` 和运行产物还不能选择它。

## 文档清理结果

已删除的文档属于历史过程记录或已经被现行文档取代：

- `docs/control_upgrade.md`
- `docs/development_log_template.md`
- `docs/dual_arm_mujoco_landing.md`
- `docs/long_term_engine_dual_arm_plan.md`
- `docs/mobile_base_6d_control.md`
- `docs/logs/`
- `docs/superpowers/`

保留的现行文档是：

- `docs/architecture_overview.md`
- `docs/configuration_reference.md`
- `docs/coordinate_conventions.md`
- `docs/debugging_guide.md`
- `docs/mainline_migration_plan.md`

## 债务清单

| 债务项 | 位置 | 风险 | 影响 |
|---|---|---|---|
| 动态自适应阻抗只在旧 MuJoCo wiping runtime 中真正生效 | `runtime/mujoco_wiping_runtime.py`、`control/scenario_controllers.py` | 高 | scenario 主线配置看起来可选动态控制，但实际未激活 |
| 接触触发导纳控制未接入 scenario | `control/contact_triggered_admittance.py` | 中 | 新力位控制能力不能被主线任务复用 |
| 旧 task 配置和 scenario 配置并存 | `configs/tasks/*.yaml`、`configs/scenarios/*.yaml` | 中 | 同名能力在两套入口里语义不同 |
| CBF-QP、DMP、移动底座位姿控制等功能是库级能力，主线任务只部分使用 | `control/cbf_qp_kinematics.py`、`tasks/dmp_trajectory.py`、`control/mobile_base_pose_control.py` | 中 | 代码可测但不容易从主线任务打开 |
| 发动机清洗 task-space 控制器与当前 staged engine navigation 分离 | `control/engine_cleaning_controller.py`、`control/staged_engine_navigation.py` | 中 | 清洗控制意图还没有并入当前发动机任务状态机 |

## 迁移原则

1. 不覆盖现有控制器。`WaypointTrackingController`、`WipingController`、`StagedEngineNavigationController` 继续作为稳定默认实现。
2. 新能力以策略对象接入。主线控制器只负责选择和编排，具体动力学、导纳、CBF、DMP 逻辑放在独立模块。
3. YAML 显式选择。任何实验功能都必须通过场景配置字段打开，默认仍走现有稳定路径。
4. metadata 真实反映执行路径。若动态控制没有激活，metadata 不能让用户误以为已生效。
5. 先接入单臂擦拭，再迁移到发动机和双臂任务。力控闭环的风险高，应从最小工作面开始。

## 动力学控制迁移

### 现状

降阶动力学模型使用：

```text
M(q) qddot + D qdot + K q = tau + J_tip(q).T F_contact
```

它仍按旧 9 维 PCC 坐标工作：

```text
q = [kx_1, ky_1, eps_1, kx_2, ky_2, eps_2, kx_3, ky_3, eps_3]
```

当前主线控制已经转向每臂 6 维 bending-space，相容控制默认去掉轴向应变。因此迁移时不能直接把旧 `qdot` 作为主线命令，需要在边界处显式投影：

```text
measured tendon delta
  -> bending estimate
  -> embed to 9D q with eps = 0
  -> dynamics predicts 9D qdot
  -> drop eps dof
  -> map 6D bending rate to RobotSystemCommand arm command
```

### 推荐实现

新增一个主线专用策略模块，例如：

```text
src/continuum_sim/control/wiping_force_strategies.py
```

其中定义统一接口：

```python
class WipingForceStrategy(Protocol):
    def compute_target_or_velocity(...):
        ...
```

首批策略：

- `KinematicHybridForceStrategy`：封装当前 `WipingController` 的法向修正逻辑，作为默认策略。
- `DynamicAdaptiveImpedanceStrategy`：复用 `adaptive_impedance.py`，把旧 motor/tendon 输出改造成主线 `RobotSystemCommand` 可接受的 bending rate。

场景配置建议：

```yaml
task:
  wiping_control_type: hybrid_force_position
  force_strategy:
    type: kinematic_hybrid
```

动态实验显式打开：

```yaml
task:
  wiping_control_type: dynamic_adaptive_impedance
  dynamics_config_path: ../dynamics/pcc_reduced.yaml
  force_strategy:
    type: dynamic_adaptive_impedance
```

为了兼容旧字段，`wiping_control_type: dynamic_adaptive_impedance` 可以自动补齐 `force_strategy.type`，但文档中应推荐新字段。

### 接入点

优先修改：

- `application/scenario.py`：给 `ScenarioTaskConfig` 增加 `dynamics_config_path` 和 `force_strategy` 子配置。
- `control/scenario_controllers.py`：让 `WipingController` 持有策略对象，而不是直接在 `compute_command()` 中写死法向修正。
- `application/application.py`：根据场景配置构造策略对象并注入 `WipingController`。
- `io/scenario_artifacts.py` 和 `runtime/hooks.py`：记录 `dynamic_active`、`predicted_qdot`、`contact_generalized_force` 等诊断字段。

## 接触触发导纳控制迁移

### 现状

`ContactTriggeredAdmittanceTracker` 已经具备主线需要的状态：

- 接触触发：未接触时目标法向力为 0。
- 导纳位移：接触后用法向力误差更新 `admittance_position_m`。
- 路点门控：切向误差和力误差同时稳定后推进。
- 后端无关：输入 tip 位置、目标点、法向、测量力和 dt 即可。

缺口是它没有连接到 scenario 配置、`WipingController` 和 recorder metadata。

### 推荐实现

把它作为另一种力位策略接入：

```yaml
task:
  wiping_control_type: contact_triggered_admittance
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

主线接口应保持后端无关：

```text
RobotSystemState + scene/query/contact metadata
  -> ContactTriggeredAdmittanceTracker.step()
  -> corrected target + desired velocity + metadata
  -> existing whole-body/bending solver
```

这样不会覆盖当前 `hybrid_force_position`，也不会强迫所有擦拭任务使用导纳控制。

### 需要补齐的数据

当前 scenario `WipingController` 主要使用距离 proxy 估计法向力。导纳控制更适合使用真实 MuJoCo 接触力，因此需要定义统一测量字段优先级：

1. MuJoCo follower contact projection 的法向力。
2. tool pad 接触力。
3. structured scene 距离 proxy。

无论来源是什么，都写入统一 metadata：

- `measured_normal_force_n`
- `normal_force_source`
- `admittance_position_m`
- `admittance_velocity_m_s`
- `force_control_active`
- `waypoint_advance_reason`

## 其他未完全接入主线的功能

### CBF-QP 速度投影

`control/cbf_qp_kinematics.py` 已提供小型线性约束投影。建议把它接入 `WholeBodyController` 或导航安全策略，而不是单独替换导航控制器：

```yaml
task:
  safety_filter:
    type: cbf_qp
    safe_distance_m: 0.014
    gamma: 4.0
```

默认关闭。打开后只对参考速度做后处理，避免覆盖主线 tracking/navigation 求解器。

### DMP 轨迹

`tasks/dmp_trajectory.py` 已有 DMP 拟合与 rollout。建议继续通过 `TrajectorySpec` 接入 tracking，而不是新增 runtime：

```yaml
task:
  trajectory:
    type: dmp
    demo_path: path/to/demo.csv
```

迁移重点是让文档明确 DMP 是轨迹源，不是控制器。

### 移动底座位姿控制

`control/mobile_base_pose_control.py` 已能从目标 SE(3) 位姿计算世界系 twist。当前 staged engine navigation 已经有底座推进逻辑，建议把移动底座位姿控制作为 `base_motion_strategy`：

```yaml
task:
  base_motion_strategy:
    type: pose_servo
    position_gain: 3.0
    orientation_gain: 2.0
```

默认继续使用现有 staged navigation 的底座控制，不改变已有场景。

### 发动机清洗 task-space 控制器

`control/engine_cleaning_controller.py` 输出 TCP 速度意图，适合作为 staged engine navigation 的局部执行臂子策略：

```yaml
task:
  engine_navigation:
    local_executor_strategy:
      type: engine_cleaning_task_space
```

它不应该替换 `StagedEngineNavigationController`。更合适的边界是：staged controller 管阶段，engine cleaning controller 管接触段的 TCP 速度意图。

## 分阶段实施路线

### 第一阶段：配置和 metadata 对齐

- 在 `ScenarioTaskConfig` 中增加 `force_strategy`、`dynamics_config_path`、`admittance` 子配置。
- 让 `WipingController` metadata 区分 requested、configured、active 三个状态。
- 当前默认行为不变。

### 第二阶段：导纳控制接入

- 注入 `ContactTriggeredAdmittanceTracker`。
- 复用现有 waypoint/bending 求解器执行 corrected target 或 desired velocity。
- 新增 `wiping_control_type: contact_triggered_admittance`。

### 第三阶段：动力学控制接入

- 把旧 `adaptive_impedance.py` 包装成主线策略。
- 显式处理 9D PCC 与 6D bending-space 的投影边界。
- 只在单臂 MuJoCo 擦拭场景中默认提供示例。

### 第四阶段：安全和发动机场景扩展

- 将 CBF-QP 做成可选 safety filter。
- 将 `EngineCleaningController` 接到 staged engine navigation 的局部接触阶段。
- 将 `MobileBasePoseController` 做成可选 base motion strategy。

## 风险和防护

- 动力学参数目前是工程估计，不能默认替代稳定控制器。
- 旧 9D PCC 包含轴向应变，主线 6D bending-space 不包含轴向应变，边界投影必须可追踪。
- MuJoCo 真实接触力、follower 投影力和距离 proxy 的量纲与噪声不同，必须记录 `normal_force_source`。
- 所有实验策略必须默认关闭，并且在 metadata 中暴露是否实际激活。

## 建议人工验证

实现迁移后建议手动运行：

```powershell
pytest tests/test_contact_triggered_admittance.py tests/test_adaptive_impedance.py tests/test_pcc_dynamics.py
pytest tests/test_scenario_artifacts.py tests/test_mujoco_wiping_runtime.py
python scripts/run_scenario.py configs/scenarios/single_mujoco_wiping.yaml
```

如果只改文档，不需要运行以上命令。
