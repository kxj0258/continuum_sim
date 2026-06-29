# continuum_sim

`continuum_sim` 是一个面向三段腱驱连续体机械臂的轻量研究仿真代码库。项目当前维护两条并行能力：

1. 原有稳定主链：单连续体 PCC / MuJoCo tracking、navigation、wiping 仿真；
2. `feat/engine-dual-arm-foundation` 分支新增扩展链：真实发动机场景、6D 移动基座、双连续体观测/执行结构、末端工具与相机附件、发动机局部清理路径、执行臂任务空间清理控制器，以及真实 engine mesh 检查和 MuJoCo 预览工具。

本项目适合用于控制链验证、几何建模、MuJoCo 可视化调试、任务路径生成、回归测试和后续 engine dual-arm cleaning runtime 的逐步搭建。它不是高保真 FEM 软体仿真器，也不是硬件驱动程序。

---

## 1. 安装

推荐使用 Conda：

```bash
conda env create -f environment.yml
conda activate continuum_sim
```

已有环境时：

```bash
conda env update -n continuum_sim -f environment.yml
conda activate continuum_sim
```

不使用 Conda：

```bash
python -m pip install -e .
```

MuJoCo 是可选依赖：

```bash
python -m pip install -e .[mujoco]
```

---

## 2. 原有快速运行入口

所有维护中的原有入口都在 `cli.py`，默认从 `configs/main_config.yaml` 开始解析：

```bash
python cli.py view-pcc --config configs/main_config.yaml
python cli.py view-motor-chain --config configs/main_config.yaml
python cli.py run-tracking --config configs/main_config.yaml
python cli.py view-mujoco --config configs/main_config.yaml
python cli.py debug-mujoco-tendons --config configs/main_config.yaml
python cli.py run-mujoco-tracking --config configs/main_config.yaml
python cli.py run-mujoco-navigation --config configs/main_config.yaml
python cli.py run-mujoco-wiping --config configs/main_config.yaml
```

需要无窗口运行时，修改对应 YAML 里的 `visualization.show` 或 `viewer.show`。

---

## 3. 原有功能概览

当前主线保留以下能力：

* PCC 正运动学、中心线采样、有限差分 Jacobian；
* 电机位置/速度与物理腱长/腱速映射；
* 末端位置 tracking 的 damped least-squares differential IK；
* `AnalyticBackend` 和可选 `MujocoBackend`；
* MuJoCo tendon-position 和 legacy joint-position 控制模式；
* 结构化 navigation 场景：火箭腔体、障碍物、中心线 clearance 查询；
* wiping 任务：板面作业面、tip contact pad、raster 路径、法向力/距离 proxy、live force panel；
* `--save-run` 导出 NPZ、metadata、配置副本、图和可选 replay video；
* dynamic adaptive impedance 实验控制器。

---

## 4. Engine Dual-Arm Foundation 分支新增内容

`feat/engine-dual-arm-foundation` 分支的目标是把项目从“单连续体板面/腔体任务”扩展为“发动机内部双连续体观测-执行协同任务”研究平台。

当前新增内容仍以 scaffold、配置、几何和控制数学层为主，尚未接入完整 dual-arm MuJoCo runtime。

### 4.1 长期规划文档

新增：

```text
docs/long_term_engine_dual_arm_plan.md
```

该文档定义了长期路线：

* M0 baseline audit；
* M1 engine scene loader；
* M1.5 real engine mesh asset integration；
* M1.6 engine mesh preview diagnostics；
* M1.7 engine alignment；
* M1.8 nozzle primitive collision diagnostics；
* M2 6D mobile base；
* M3 dual continuum arms；
* M4 tool/camera attachments；
* M5 engine surface path generation；
* M6 executor engine cleaning controller；
* M7 observer ground-truth perception and visual servo；
* M8 dual-arm collision avoidance；
* M9 full task state machine；
* M10 sim2real noise / latency / hardware interface。

---

## 5. Engine scene 与真实发动机 mesh

### 5.1 配置文件

当前 engine scene 相关配置包括：

```text
configs/scenes/engine_cleaning.yaml
configs/scenes/engine_cleaning_aligned.yaml
configs/scenes/engine_cleaning_nozzle_collision.yaml
```

其中：

* `engine_cleaning.yaml`：原始真实 engine mesh 接入配置；
* `engine_cleaning_aligned.yaml`：根据 mesh bbox 生成的 grounded 对齐配置；
* `engine_cleaning_nozzle_collision.yaml`：在 aligned 配置基础上增加喷管 primitive collision hint。

### 5.2 真实资产目录

真实 CAD / mesh 推荐放在：

```text
assets/engine/
  raw/
    engine.step
    engine_for_sim.sldprt
  meshes/
    engine_visual.stl
  collision/
    engine_collision.stl
```

注意：

* STEP / SLDPRT / SLDASM 不能直接进入 MuJoCo，需要从 SolidWorks 导出 STL / OBJ / MSH；
* visual mesh 和 collision mesh 应分离；
* 复杂整体发动机 mesh 不建议直接作为唯一碰撞模型；
* 大型 CAD/mesh 不建议直接提交 git，必要时使用 Git LFS；
* 如果 SolidWorks 导出单位为 mm，通常在 YAML 中使用 `scale: 0.001`。

### 5.3 资产检查脚本

检查真实 engine mesh：

```bash
python scripts/check_engine_assets.py --config configs/scenes/engine_cleaning.yaml
python scripts/check_engine_assets.py --config configs/scenes/engine_cleaning_aligned.yaml
python scripts/check_engine_assets.py --config configs/scenes/engine_cleaning_nozzle_collision.yaml
```

严格检查资产是否存在：

```bash
python scripts/check_engine_assets.py --config configs/scenes/engine_cleaning_aligned.yaml --strict-assets
```

输出 JSON：

```bash
python scripts/check_engine_assets.py --config configs/scenes/engine_cleaning_aligned.yaml --json
```

该脚本会输出：

* visual/collision mesh 是否存在；
* 文件大小；
* 顶点数 / 面数；
* raw bbox；
* scaled bbox；
* world bbox；
* region 与 bbox 的距离；
* primitive collision hint 的 bbox 与相交关系；
* 单位和 face count warning。

### 5.4 engine pose 建议

根据 visual mesh bbox 生成对齐 pose：

```bash
python scripts/suggest_engine_pose.py --config configs/scenes/engine_cleaning.yaml --mode grounded
```

写出建议配置，不覆盖原始文件：

```bash
python scripts/suggest_engine_pose.py --config configs/scenes/engine_cleaning.yaml --mode grounded --write-aligned-config configs/scenes/engine_cleaning_aligned.yaml
```

### 5.5 MuJoCo 预览 engine scene

headless 检查：

```bash
python scripts/preview_engine_scene_mujoco.py --config configs/scenes/engine_cleaning_aligned.yaml --headless-check
```

打开 viewer：

```bash
python scripts/preview_engine_scene_mujoco.py --config configs/scenes/engine_cleaning_aligned.yaml --viewer --show-bbox --show-regions --show-axes
```

预览喷管 primitive collision hint：

```bash
python scripts/preview_engine_scene_mujoco.py --config configs/scenes/engine_cleaning_nozzle_collision.yaml --viewer --show-bbox --show-regions --show-axes --show-primitive-collision --show-disabled-hints
```

### 5.6 喷管 primitive collision 建议

从 collision mesh bbox 自动生成 capsule/box hint：

```bash
python scripts/suggest_nozzle_collision.py --config configs/scenes/engine_cleaning_aligned.yaml --source collision --primitive capsule --output-config configs/scenes/engine_cleaning_nozzle_collision.yaml
```

可选启用 hint：

```bash
python scripts/suggest_nozzle_collision.py --config configs/scenes/engine_cleaning_aligned.yaml --source collision --primitive capsule --output-config configs/scenes/engine_cleaning_nozzle_collision.yaml --enable-hint
```

默认生成的 primitive collision hint 是辅助可视化和诊断，不建议未经人工确认就作为最终接触碰撞模型。

---

## 6. 6D mobile base 与 mount 坐标系

新增模块：

```text
src/continuum_sim/model/base_pose.py
src/continuum_sim/model/mount_frame.py
src/continuum_sim/kinematics/world_kinematics.py
configs/robots/mobile_base_pose.yaml
```

核心变换链：

```text
T_world_tip
=
T_world_mobile_base
*
T_mobile_base_mount
*
T_mount_tip
```

当前支持：

* `Pose6D`；
* 四元数顺序 `[w, x, y, z]`；
* pose 到 4x4 齐次矩阵；
* inverse；
* compose；
* 点 / 点集 / pose 变换；
* local tip pose 到 world tip pose；
* local centerline 到 world centerline。

---

## 7. 双连续体 observer / executor 配置

新增配置：

```text
configs/robots/dual_continuum.yaml
```

当前结构：

```text
mobile_base
  ├── observer arm
  │     └── attachment: eye_camera_air_gun
  └── executor arm
        └── attachment: carbon_removal_tool
```

新增模块：

```text
src/continuum_sim/model/multi_arm.py
src/continuum_sim/runtime/multi_arm_state.py
```

当前支持：

* `ArmConfig`；
* `MultiArmConfig`；
* observer / executor role 校验；
* mount offset；
* enabled arm 过滤；
* 根据 role 查询 arm；
* 多臂 runtime state；
* 每根 arm 的 world tip pose；
* 每根 arm的 world centerline。

当前尚未实现双臂控制器和 dual-arm MuJoCo runtime。

---

## 8. 工具、相机和气枪附件

新增配置：

```text
configs/tools/carbon_remover.yaml
configs/tools/eye_camera_air_gun.yaml
```

新增模块：

```text
src/continuum_sim/tools/attachments.py
src/continuum_sim/tools/tool_frames.py
src/continuum_sim/sensing/camera_model.py
```

### 8.1 executor 工具

`carbon_removal_tool` 当前表示为 contact sphere tool，包含：

* tip 到 attachment 位姿；
* TCP pose；
* sphere collision radius；
* mass；
* target normal force；
* max normal force；
* standoff distance。

### 8.2 observer 工具

`eye_camera_air_gun` 包含：

* tip 到 attachment 位姿；
* camera intrinsics；
* tip 到 camera 位姿；
* nozzle pose；
* airgun standoff distance。

### 8.3 坐标系计算

支持：

```text
world_tip -> world_attachment
world_tip -> world_tcp
world_tip -> world_camera
world_tip -> world_nozzle
```

当前还没有实现 MuJoCo camera renderer、真实图像识别或视觉伺服。

---

## 9. Engine surface path generation

新增配置：

```text
configs/tasks/engine_surface_path.yaml
```

新增模块：

```text
src/continuum_sim/scenes/engine_surfaces.py
src/continuum_sim/tasks/engine_surface_path.py
src/continuum_sim/tasks/engine_cleaning_config.py
```

当前支持：

* `plane_patch`；
* `sphere_patch`；
* `annotated_mesh_patch` 占位；
* surface frame；
* surface point sampling；
* raster cleaning path；
* `CleaningWaypoint`；
* approach / contact / retreat waypoint；
* snake pattern；
* target force；
* standoff distance。

当前还没有把 path 接入 MuJoCo runtime。

---

## 10. Executor engine cleaning controller

新增配置：

```text
configs/control/engine_cleaning_controller.yaml
```

新增模块：

```text
src/continuum_sim/control/engine_cleaning_types.py
src/continuum_sim/control/engine_cleaning_controller.py
```

当前控制器是 task-space scaffold。

输入：

```text
CleaningWaypoint sequence
current TCP pose
measured normal force
contact distance
contact state
```

输出：

```text
desired_tcp_velocity_world
active_waypoint_index
phase
waypoint_reached
safety_stop
stop_reason
```

当前支持：

* approach 阶段位置伺服；
* contact 阶段切向位置控制 + 法向 gap / force 混合控制；
* retreat 阶段位置伺服；
* max TCP speed 限制；
* max normal speed 限制；
* max contact force safety stop。

当前不输出：

* qdot；
* tendon velocity；
* motor command；
* MuJoCo actuator target；
* real hardware command。

---

## 11. 当前还没有实现的内容

当前 `feat/engine-dual-arm-foundation` 分支已经完成基础架构层，但尚未完成完整仿真闭环。

未完成项包括：

* 双连续体 MuJoCo runtime；
* engine cleaning runtime；
* observer ground-truth perception；
* visual servo；
* MuJoCo camera renderer；
* RGB-D 识别；
* 双臂避碰；
* executor task-space command 到 qdot/tendon/motor 的映射；
* 蛇形臂本体动力学；
* sim2real 噪声、时延和硬件接口；
* 真实喷管 collision 的人工确认与启用；
* low-poly visual mesh 的正式导出。

---

## 12. 推荐后续开发路线

建议继续按低风险顺序推进：

### M1.9：engine region manual calibration

目标：

* 在 viewer 中人工确认 entry_port、inspection_roi、carbon_deposit_region、forbidden_zone；
* 让 region 更贴合真实喷管和碳沉积目标区域；
* 输出 `engine_cleaning_regions_calibrated.yaml`。

### M6.5：executor command adapter scaffold

目标：

```text
desired_tcp_velocity_world
  -> executor Jacobian
  -> qdot
  -> tendon velocity
  -> motor / MuJoCo command placeholder
```

仍然可以先不接 runtime，只做 math adapter 和测试。

### M7：observer ground-truth perception + visual servo scaffold

目标：

* 从 engine scene region 生成 target observation；
* 计算 target in camera frame；
* 输出 observer camera velocity command；
* 先不做 RGB-D 图像识别。

### M8：dual-arm collision avoidance

目标：

* observer centerline vs executor centerline；
* arm/tool vs engine collision；
* capsule/centerline clearance；
* velocity scaling / safety stop filter。

### M9：engine cleaning MuJoCo runtime

目标：

* engine scene；
* executor arm；
* carbon removal tool；
* surface path；
* task-space controller；
* MuJoCo viewer / headless rollout；
* 保存 result、metadata、plots、GIF。

### M10：sim2real interface

目标：

* actuator delay；
* tendon noise；
* camera latency；
* force sensor noise；
* hardware interface stubs。

---

## 13. 常用测试命令

### 13.1 全量测试

```bash
python -m pytest
```

### 13.2 原有核心测试

```bash
python -m pytest -m core
python -m pytest -m baseline
```

### 13.3 Engine scene / asset / pose / nozzle collision

```bash
python -m pytest tests/test_engine_scene.py -v
python -m pytest tests/test_engine_aligned_scene.py -v
python -m pytest tests/test_engine_asset_checks.py -v
python -m pytest tests/test_engine_pose_suggestions.py -v
python -m pytest tests/test_primitive_collision_geoms.py -v
python -m pytest tests/test_nozzle_collision_suggestions.py -v
```

### 13.4 6D base / world kinematics

```bash
python -m pytest tests/test_base_pose.py tests/test_world_kinematics.py -v
```

### 13.5 Dual-arm config / state

```bash
python -m pytest tests/test_multi_arm_model.py tests/test_multi_arm_state.py -v
```

### 13.6 Tool / camera attachments

```bash
python -m pytest tests/test_tool_attachments.py tests/test_tool_frames.py tests/test_camera_model.py -v
```

### 13.7 Engine surface path

```bash
python -m pytest tests/test_engine_surfaces.py tests/test_engine_surface_path.py tests/test_engine_cleaning_config.py -v
```

### 13.8 Engine cleaning controller

```bash
python -m pytest tests/test_engine_cleaning_controller.py tests/test_engine_cleaning_controller_config.py -v
```

### 13.9 MuJoCo 相关测试

```bash
python -m pytest -m mujoco
```

如果没有安装 MuJoCo，相关测试可能会跳过。

---

## 14. Engine 资产诊断与预览命令

### 14.1 检查原始 engine 配置

```bash
python scripts/check_engine_assets.py --config configs/scenes/engine_cleaning.yaml
```

### 14.2 检查 aligned engine 配置

```bash
python scripts/check_engine_assets.py --config configs/scenes/engine_cleaning_aligned.yaml
```

### 14.3 检查 nozzle collision 配置

```bash
python scripts/check_engine_assets.py --config configs/scenes/engine_cleaning_nozzle_collision.yaml
```

### 14.4 生成 engine pose 建议

```bash
python scripts/suggest_engine_pose.py --config configs/scenes/engine_cleaning.yaml --mode grounded
```

### 14.5 生成 nozzle primitive collision hint

```bash
python scripts/suggest_nozzle_collision.py --config configs/scenes/engine_cleaning_aligned.yaml --source collision --primitive capsule --output-config configs/scenes/engine_cleaning_nozzle_collision.yaml
```

### 14.6 MuJoCo headless 预览检查

```bash
python scripts/preview_engine_scene_mujoco.py --config configs/scenes/engine_cleaning_aligned.yaml --headless-check
```

### 14.7 MuJoCo viewer 预览 aligned scene

```bash
python scripts/preview_engine_scene_mujoco.py --config configs/scenes/engine_cleaning_aligned.yaml --viewer --show-bbox --show-regions --show-axes
```

### 14.8 MuJoCo viewer 预览 nozzle collision hint

```bash
python scripts/preview_engine_scene_mujoco.py --config configs/scenes/engine_cleaning_nozzle_collision.yaml --viewer --show-bbox --show-regions --show-axes --show-primitive-collision --show-disabled-hints
```

---

## 15. 推荐仓库结构

```text
configs/
  control/
    engine_cleaning_controller.yaml
  robots/
    mobile_base_pose.yaml
    dual_continuum.yaml
  scenes/
    engine_cleaning.yaml
    engine_cleaning_aligned.yaml
    engine_cleaning_nozzle_collision.yaml
  tasks/
    engine_surface_path.yaml
  tools/
    carbon_remover.yaml
    eye_camera_air_gun.yaml

assets/
  engine/
    README.md
    raw/
    meshes/
    collision/

scripts/
  check_engine_assets.py
  preview_engine_scene_mujoco.py
  suggest_engine_pose.py
  suggest_nozzle_collision.py

src/continuum_sim/
  model/
    base_pose.py
    mount_frame.py
    multi_arm.py
  kinematics/
    world_kinematics.py
  scenes/
    engine_scene.py
    engine_surfaces.py
    primitive_collision.py
  tasks/
    engine_surface_path.py
    engine_cleaning_config.py
  tools/
    attachments.py
    tool_frames.py
  sensing/
    camera_model.py
  control/
    engine_cleaning_types.py
    engine_cleaning_controller.py
  runtime/
    multi_arm_state.py
```

---

## 16. 开发注意事项

* 不要把大型 CAD / mesh 文件直接提交到 git；
* 若需要版本化真实资产，建议使用 Git LFS；
* 不要用高面数整体 engine visual mesh 作为唯一 collision；
* visual mesh 和 collision mesh 应分离；
* 当前 `engine_visual.stl` 仍建议后续导出 low-poly 版本；
* 当前 nozzle primitive collision hint 默认 disabled，需要人工 viewer 确认后再启用；
* 当前 engine path / controller 仍未接入完整 MuJoCo runtime；
* 当前 observer 相机还没有真实渲染和视觉伺服；
* 当前 dual-arm 只完成配置和状态抽象，还没有协同控制闭环。

---

## 17. 当前分支定位

`feat/engine-dual-arm-foundation` 当前不是最终可演示的双臂发动机清理 demo，而是一个“发动机双臂任务基础设施分支”。

已经完成：

```text
engine scene + real mesh diagnostics
6D base pose
dual-arm config/state
tool/camera attachments
engine surface path
executor task-space controller
primitive nozzle collision hint
```

尚未完成：

```text
dual-arm MuJoCo runtime
visual servo
collision avoidance
Jacobian/motor command adapter
full task state machine
sim2real
```

下一步建议优先完成：

```text
M1.9 region calibration
M6.5 executor command adapter
M7 observer ground-truth perception and visual servo scaffold
```

---

## Development Roadmap

The long-term development baseline is documented in:

- `docs/development_baseline.md`
- `docs/development_log_template.md`

For future engine scene alignment, mobile base control, dual-arm import, and engine interaction tasks, use `feat/engine-dual-arm-foundation` as the main development branch.
