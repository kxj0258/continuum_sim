# 当前能力状态与迁移建议

本文只描述当前推荐入口 `configs/scenarios/*.yaml` 与
`SimulationApplication` 的真实状态。旧任务 YAML 和旧 runtime 仍可作为兼容层或实验参考，
但不再作为新实验的优先入口。

## 文档范围

当前建议保留并维护的文档：

- `README.md`：项目入口、目录和常用场景。
- `docs/architecture_overview.md`：组合根、模块边界和数据流。
- `docs/configuration_reference.md`：配置字段参考。
- `docs/coordinate_conventions.md`：坐标、base twist 和 tendon 命令约定。
- `docs/debugging_guide.md`：排查顺序、hooks 和运行产物。
- `docs/dual_arm_mujoco_landing.md`：双臂 MuJoCo spatial tendon 与孔位资产说明。
- `docs/current_status.md`：当前动力学、控制接口和迁移建议。

已移除的文档类别：

- 历史实施计划、spec、阶段日志和开发日志模板。
- 已被 scenario 主入口取代的阶段性说明。
- 与当前代码状态不一致的“新增控制扩展”说明。

## 连续体臂动力学建模

仓库中有连续体臂动力学建模。它仍是实验性的单臂 PCC 降阶模型，但现在已经可以通过
scenario 主接口的 `task.wiping_control_type: dynamic_adaptive_impedance` 调用。

实现位置：

- `src/continuum_sim/dynamics/pcc_dynamics.py`
- `src/continuum_sim/control/adaptive_impedance.py`
- `configs/dynamics/pcc_reduced.yaml`
- 旧 runtime 调用点：`src/continuum_sim/runtime/mujoco_wiping_runtime.py`

模型状态向量沿用 9 维 PCC 坐标：

```text
q = [kx_1, ky_1, eps_1, kx_2, ky_2, eps_2, kx_3, ky_3, eps_3]
```

动力学形式为：

```text
M(q) qddot + D qdot + K q = tau + J_tip(q).T F_contact
```

`dynamic_adaptive_impedance` 在旧 MuJoCo wiping runtime 中会输出 motor velocity；scenario 主入口中
则通过 `RobotSystemCommand` 输出 executor 的 tendon-rate。迁移后的系统级实现会在每个控制步估计
executor 的 `q/qdot`、求质量矩阵、预测 `qdot`，再映射到相容 tendon-rate；metadata 中的
`wiping_dynamic_system_controller_active` 会标记实际启用状态。

## 是否可应用到当前双臂

已经完成第一阶段迁移：动力学擦拭可作用于当前双臂装配中的 executor 单臂，observer 和 base
仍沿用现有系统级控制输出。它还不能视为完整双臂动力学模型。原因是现有动力学模型是“单条三段臂、臂局部 PCC
坐标、无移动底座动力学、无双臂接触耦合”的工程估计模型；当前双臂主接口使用
`RobotSystemState`、`RobotSystemCommand`、`ControlLayout`、`WholeBodyController` 和每臂
bending-space/tendon-rate 命令。

后续深化路径：

1. 用实验或 MuJoCo 数据辨识 `configs/dynamics/pcc_reduced.yaml`。
2. 将 executor 动力学控制变成 executor active-subspace 中的目标或约束；observer ROI/避碰保持在
   observer active-subspace，不能重新并入 executor 的 SVD。
3. 如果需要两臂同时接触，再扩展为 per-arm 动力学状态和多接触 generalized force 聚合。

迁移风险：

- `configs/dynamics/pcc_reduced.yaml` 是工程估计，不是辨识参数。
- 现有 `mass_matrix(...)` 用有限差分和中心线采样，双臂实时运行成本会随每臂调用次数上升。
- 当前模型没有 base 质量、base actuator、双臂结构耦合和 MuJoCo spatial tendon 解析动力学。
- 如果两条臂都进入接触，单臂 `J_tip.T F_contact` 近似不足以表达整机接触反作用。

## 旧接口迁移状态

以下能力已经接入 `SimulationApplication` 主入口：

- Dynamic adaptive impedance：`task.wiping_control_type: dynamic_adaptive_impedance`，
  可选 `task.dynamics_config_path`。
- Contact-triggered admittance：`task.wiping_control_type: contact_triggered_admittance`，
  参数位于 `task.contact_admittance`。
- Navigation CBF-QP：`task.navigation_control_type: navigation_cbf_qp`，作为 whole-body command
  后处理投影接入。
- EngineCleaningController：`task.type: engine_cleaning` 默认使用 task-space controller；
  可用 `task.engine_cleaning_control` 覆盖增益。
- DMP trajectory：scenario tracking 的 `trajectory.type: dmp`。

以下能力仍作为兼容或底层工具保留：

- 旧单臂 motor-space 差分 IK 跟踪 runtime。
- 旧 MuJoCo navigation/wiping runtime。
- `mobile_base_controller.py` 的底座命令工具；主接口仍通过 world-frame base twist 和后端积分使用。

已经进入主接口的关键能力：

- scenario 组合入口：`SimulationApplication.from_yaml(...)`。
- 双臂/单臂 assembly、mobile base、engine/structured scene 注入。
- tracking、navigation、wiping、engine_navigation 的系统级控制器。
- executor/observer 分臂求解、双臂中心线避碰、发动机 clearance 避障、奇异性保护和 tendon 相容命令。
- DMP trajectory 作为 scenario tracking 的 `trajectory.type: dmp`。

## 迁移优先级

建议优先级：

1. 为新增 scenario 控制模式补齐手动验证和回归测试。
2. 把动态擦拭从“覆盖 executor tendon-rate”进一步融合进 `WholeBodyController`。
3. 为 engine cleaning 加入真实接触力来源或工具传感器接口，替代当前 signed-distance force proxy。
