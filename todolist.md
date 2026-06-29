# continuum_sim 长期开发 TODO 大纲

## 0. 项目开发基线与协作约定

### 0.1 分支基线

* [ ] 确认 `main` 分支当前可运行状态
* [ ] 确认 `feat/engine-dual-arm-foundation` 分支当前可运行状态
* [ ] 将 `feat/engine-dual-arm-foundation` 作为后续主要开发分支
* [ ] 后续每个功能点从该分支拆出独立 feature 分支

建议分支命名：

```bash
feat/engine-frame-alignment
feat/mobile-base-6d-control
feat/dual-arm-import-passive-observer
feat/engine-arm-task-integration
feat/observer-camera-feedback
```

### 0.2 基础测试命令记录

* [ ] 记录当前可以通过的测试
* [ ] 记录当前失败的测试及原因
* [ ] 建立每次修改后的最小回归测试集合

建议命令：

```bash
python -m pytest -m core
python -m pytest tests/test_base_pose.py tests/test_world_kinematics.py
python -m pytest tests/test_engine_scene.py tests/test_engine_cleaning_config.py
python -m pytest tests/test_multi_arm_model.py tests/test_multi_arm_state.py
python -m pytest tests/test_camera_model.py
```

### 0.3 开发记录规范

每完成一个小任务，记录：

```markdown
## 日期：YYYY-MM-DD

### 完成内容

### 修改文件

### 运行命令

### 测试结果

### 当前问题

### 下一步计划
```

---

# 1. 发动机模型坐标系对齐与入口路径可视化

## 1.1 目标

在已有发动机模型场景下，明确以下坐标关系：

```text
CAD / mesh 原始坐标系
        ↓
engine 局部坐标系
        ↓
MuJoCo world 坐标系
        ↓
连续体臂基座 / 末端 / 工具坐标系
```

最终应准确给出：

* [ ] 发动机 mesh 在 MuJoCo world 下的位置和姿态
* [ ] 发动机 bbox 在 MuJoCo world 下的坐标
* [ ] 导航入口点 entry point 的 MuJoCo 坐标
* [ ] entry normal 的 MuJoCo 表达
* [ ] 初始导航路径 waypoints 的 MuJoCo 坐标
* [ ] 可视化入口点、路径、坐标轴、bbox

---

## 1.2 配置文件整理

* [ ] 检查 `configs/scenes/engine_cleaning.yaml`
* [ ] 检查 `configs/scenes/engine_cleaning_aligned.yaml`
* [ ] 检查 `configs/scenes/engine_cleaning_nozzle_collision.yaml`
* [ ] 统一 `regions`、`entry_port`、`exploration_paths` 的 frame 语义
* [ ] 所有坐标字段显式增加 `frame`

建议格式：

```yaml
entry_port:
  frame: engine
  center_m: [...]
  normal: [...]
  radius_m: 0.045
```

---

## 1.3 坐标变换工具

* [ ] 新增或完善 `Transform` 工具类
* [ ] 支持 position + quaternion
* [ ] 支持 engine frame 到 MuJoCo world frame 的转换
* [ ] 支持 path points 批量转换
* [ ] 支持 normal/vector 的旋转转换
* [ ] 编写单元测试验证变换正确性

可能涉及文件：

```text
src/continuum_sim/model/base_pose.py
src/continuum_sim/kinematics/world_kinematics.py
src/continuum_sim/scenes/scene_config.py
src/continuum_sim/scenes/engine_scene.py
```

---

## 1.4 发动机坐标报告脚本

* [ ] 新增 `scripts/report_engine_alignment.py`
* [ ] 输出 mesh 原始 bbox
* [ ] 输出 scale 后 bbox
* [ ] 输出 MuJoCo world 下 bbox
* [ ] 输出 entry point 的 engine 坐标和 MuJoCo 坐标
* [ ] 输出 initial path 的 engine 坐标和 MuJoCo 坐标
* [ ] 检查 normal 是否单位化
* [ ] 检查 path 起点是否接近 entry point
* [ ] 检查可能的坐标轴反向或符号错误

建议命令：

```bash
python scripts/report_engine_alignment.py \
  --scene configs/scenes/engine_cleaning_nozzle_collision.yaml \
  --out reports/engine_alignment_report.md \
  --json reports/engine_alignment_report.json
```

---

## 1.5 场景可视化

* [ ] 可视化发动机 bbox
* [ ] 可视化 entry point
* [ ] 可视化 entry normal
* [ ] 可视化 initial path
* [ ] 可视化 engine frame 三轴
* [ ] 可视化 MuJoCo world frame 三轴
* [ ] 确保可视化 geom 不参与碰撞

MuJoCo 可视化 geom 建议：

```xml
contype="0"
conaffinity="0"
```

建议命令：

```bash
python scripts/preview_engine_scene_mujoco.py \
  --scene configs/scenes/engine_cleaning_nozzle_collision.yaml \
  --show-entry \
  --show-path \
  --show-frames
```

---

## 1.6 验收标准

* [ ] 能输出 entry point 的 MuJoCo 坐标
* [ ] 能输出 initial path 每个 waypoint 的 MuJoCo 坐标
* [ ] 在 MuJoCo viewer 中能看到入口点、路径和坐标轴
* [ ] 可视化位置与发动机 mesh 对齐
* [ ] engine pose 改变后，entry/path 能自动跟随变换
* [ ] 相关测试通过

---

# 2. 连续体机械臂增加 6D 基座自由度

## 2.1 目标

在已有连续体机械臂模型基础上增加移动基座自由度：

```text
base_x
base_y
base_z
base_roll
base_pitch
base_yaw
```

使完整系统状态由：

```text
q_arm
```

扩展为：

```text
q_full = [base_pose, q_arm]
```

---

## 2.2 BasePose 数据结构

* [ ] 新增或完善 `BasePose`
* [ ] 支持 position
* [ ] 支持 quaternion
* [ ] 支持 rpy 与 quaternion 互转
* [ ] 支持 pose composition
* [ ] 支持 pose inverse
* [ ] 支持 transform point / vector
* [ ] 编写单元测试

可能涉及文件：

```text
src/continuum_sim/model/base_pose.py
tests/test_base_pose.py
```

---

## 2.3 World Kinematics

* [ ] 将连续体臂 FK 从局部坐标扩展到 world 坐标
* [ ] 支持 `T_world_tip = T_world_base * T_base_arm * T_arm_tip`
* [ ] 支持 base 改变后 centerline 跟随变化
* [ ] 支持 tip pose 跟随 base 位姿变化
* [ ] 编写测试验证 identity base 时原始行为不变

可能涉及文件：

```text
src/continuum_sim/kinematics/world_kinematics.py
tests/test_world_kinematics.py
```

---

## 2.4 MuJoCo 模型结构修改

* [ ] 在 MuJoCo XML 中增加 base body
* [ ] 将连续体臂 root 挂载到 base body 下
* [ ] 增加 base box 可视化
* [ ] 增加 base frame site
* [ ] 增加 arm mount site
* [ ] 支持从配置文件读取 base 初始 pose

建议结构：

```xml
<body name="mobile_base" pos="..." quat="...">
  <geom name="mobile_base_box" type="box" size="..." rgba="..."/>
  <site name="base_frame"/>
  <site name="arm_mount"/>
  <!-- continuum arm body -->
</body>
```

---

## 2.5 手动控制 UI

* [ ] UI 中增加 base x/y/z 控制
* [ ] UI 中增加 base roll/pitch/yaw 控制
* [ ] 增加 reset base pose 功能
* [ ] 增加 lock base 功能
* [ ] 增加 lock arm 功能
* [ ] 支持 coarse/fine step size 切换

建议快捷键：

```text
W/S: base +x / -x
A/D: base +y / -y
R/F: base +z / -z
I/K: pitch
J/L: yaw
U/O: roll
B: toggle base lock
H: reset base pose
```

---

## 2.6 控制器扩展

* [ ] 将控制变量扩展为 base twist + arm command
* [ ] 构建 full Jacobian
* [ ] 支持 base 与 arm 的加权控制
* [ ] 支持只动 base
* [ ] 支持只动 arm
* [ ] 支持 base + arm 协同控制

控制形式：

```text
x_dot_tip = J_base * v_base + J_arm * q_dot_arm
```

优化形式：

```text
min ||J_full u - x_dot_des||²
    + λ_base ||u_base||²
    + λ_arm ||u_arm||²
```

---

## 2.7 验收标准

* [ ] base 平移时，整条机械臂跟随移动
* [ ] base 旋转时，整条机械臂跟随旋转
* [ ] tip world pose 计算正确
* [ ] UI 可以控制 6D base
* [ ] base box 在场景中显示正确
* [ ] base pose 为 identity 时，原有单臂控制行为不变
* [ ] 测试通过

---

# 3. 双连续体臂 SolidWorks 模型导入

## 3.1 目标

将双臂组合基座 SolidWorks 文件导入 MuJoCo，实现：

* [ ] 一个主执行臂 executor arm
* [ ] 一个观测从臂 observer arm
* [ ] 主臂参与控制
* [ ] 主臂参与碰撞检测
* [ ] 从臂初期仅可视化
* [ ] 从臂初期不参与碰撞
* [ ] 从臂初期不参与任务执行
* [ ] 后续从臂支持碰撞、任务执行和手眼相机

---

## 3.2 CAD 文件整理

* [ ] 整理 SolidWorks 源文件
* [ ] 明确装配体坐标系
* [ ] 明确主臂安装坐标系
* [ ] 明确从臂安装坐标系
* [ ] 明确相机安装坐标系
* [ ] 导出 visual mesh
* [ ] 导出 simplified collision mesh
* [ ] 统一单位为 m

建议目录：

```text
assets/cad/source/dual_arm_assembly/
assets/meshes/dual_arm/
assets/mujoco/dual_arm/
```

---

## 3.3 CAD 到 MuJoCo 坐标标定

* [ ] 建立 `T_mujoco_cad`
* [ ] 建立 `T_base_executor_mount`
* [ ] 建立 `T_base_observer_mount`
* [ ] 建立 `T_observer_camera`
* [ ] 编写坐标报告脚本
* [ ] 可视化 CAD frame、base frame、mount frame

建议配置：

```yaml
dual_arm_assembly:
  cad_unit: mm
  scale_to_m: 0.001
  T_mujoco_cad:
    position_m: [...]
    quat_wxyz: [...]
```

---

## 3.4 主臂 active，从臂 visual-only

* [ ] 配置 executor arm 为 active
* [ ] executor arm 接入已有 tendon/PCC/MuJoCo 控制
* [ ] executor arm 参与 collision
* [ ] observer arm 只加载 visual mesh
* [ ] observer arm 设置 `contype=0`
* [ ] observer arm 设置 `conaffinity=0`
* [ ] observer arm 不加入 actuator
* [ ] observer arm 不加入 controller state
* [ ] observer arm 不加入任务执行

建议配置：

```yaml
arms:
  executor:
    role: executor
    enabled: true
    visual: true
    collision: true
    actuated: true
    task_enabled: true

  observer:
    role: observer
    enabled: true
    visual: true
    collision: false
    actuated: false
    task_enabled: false
```

---

## 3.5 双臂状态管理

* [ ] 扩展 `MultiArmState`
* [ ] 支持 active arm 列表
* [ ] 支持 passive arm 列表
* [ ] 支持 executor arm state
* [ ] 支持 observer arm visual state
* [ ] 支持后续 observer camera state
* [ ] 编写测试

可能涉及文件：

```text
src/continuum_sim/runtime/multi_arm_state.py
src/continuum_sim/model/multi_arm_model.py
tests/test_multi_arm_model.py
tests/test_multi_arm_state.py
```

---

## 3.6 从臂相机预留

* [ ] 在 observer arm 上增加 camera site
* [ ] 增加 camera frame
* [ ] 增加 camera config
* [ ] 支持后续 RGB 渲染
* [ ] 支持后续 depth 渲染
* [ ] 支持后续 segmentation 渲染
* [ ] 支持输出 camera extrinsics

建议结构：

```text
observer_arm
  └── observer_tip
        └── observer_camera_site
```

---

## 3.7 验收标准

* [ ] 双臂基座能在 MuJoCo 中正确显示
* [ ] 主臂位置和安装方向正确
* [ ] 从臂位置和安装方向正确
* [ ] 主臂可控制
* [ ] 主臂可碰撞
* [ ] 从臂仅可视化，不产生碰撞
* [ ] 从臂不影响控制器
* [ ] 相机安装点可视化正确
* [ ] 相关测试通过

---

# 4. 连续体臂导入发动机场景

## 4.1 目标

将连续体臂或双臂系统导入发动机模型场景，构建实际任务仿真环境，支持：

* [ ] 设定机器人与发动机的初始相对位置
* [ ] 基座 6D 自由度控制
* [ ] 连续体臂绳驱控制
* [ ] 入口导航
* [ ] 内部路径跟踪
* [ ] 接触检测
* [ ] 接触交互
* [ ] 后续视觉反馈

---

## 4.2 场景结构

目标结构：

```text
world
  ├── engine_model
  │     ├── visual mesh
  │     ├── collision mesh
  │     ├── entry_port marker
  │     ├── initial_path marker
  │     └── target_region
  │
  └── mobile_dual_arm_base
        ├── base_box
        ├── executor_arm_mount
        │     └── executor_continuum_arm
        │           └── tool
        └── observer_arm_mount
              └── observer_arm_visual / observer_camera
```

---

## 4.3 初始相对位姿配置

* [ ] 支持 world 绝对位姿
* [ ] 支持相对于 engine entry_port 的位姿
* [ ] 支持相对于 engine frame 的位姿
* [ ] 支持配置 base offset
* [ ] 支持配置 base rpy
* [ ] 支持可视化初始位姿

推荐配置：

```yaml
robot_initial_pose:
  frame: engine_entry_port
  offset_m: [-0.30, 0.00, 0.00]
  rpy_deg: [0.0, 0.0, 0.0]
```

---

## 4.4 任务配置文件

* [ ] 新增 `configs/tasks/mujoco_engine_navigation.yaml`
* [ ] 指定 engine scene
* [ ] 指定 robot config
* [ ] 指定 active arms
* [ ] 指定 initial pose
* [ ] 指定 entry region
* [ ] 指定 initial path
* [ ] 指定 target region
* [ ] 指定 tool
* [ ] 指定接触力阈值
* [ ] 指定 clearance 阈值

建议配置：

```yaml
task:
  type: engine_navigation_interaction
  scene_config_path: configs/scenes/engine_cleaning_nozzle_collision.yaml
  robot_config_path: configs/robots/dual_continuum.yaml

active_arms:
  executor:
    control: true
    collision: true
    task: true

  observer:
    visual: true
    collision: false
    camera: false

navigation:
  entry_region: entry_port
  path_name: nozzle_axis_entry
  clearance_m: 0.01
  max_tip_speed_mps: 0.02

interaction:
  target_region: carbon_deposit_region
  target_force_n: 1.0
  max_force_n: 2.0
```

---

## 4.5 导航阶段

### Stage 1：基座移动到入口前

* [ ] 根据 entry_port 计算 base prepose
* [ ] base 6D 控制到入口前方
* [ ] 确保机械臂未碰撞发动机
* [ ] 可视化 base 目标位姿

### Stage 2：主臂进入入口

* [ ] tip 跟踪 entry point
* [ ] 沿 initial path 进入发动机内部
* [ ] 检查碰撞
* [ ] 检查 clearance
* [ ] 记录 tip trajectory

### Stage 3：接近目标区域

* [ ] 识别 target region
* [ ] 生成 approach pose
* [ ] 控制 tip 到达目标附近
* [ ] 建立轻微接触

### Stage 4：表面交互

* [ ] 沿 surface path 运动
* [ ] 保持接触力不超过阈值
* [ ] 记录接触力
* [ ] 记录穿透深度
* [ ] 记录最小距离
* [ ] 完成 retreat

---

## 4.6 控制器集成

* [ ] 实现 stage manager
* [ ] 实现 base controller
* [ ] 实现 executor arm controller
* [ ] 实现 base + arm 协同控制
* [ ] 实现 collision constraint
* [ ] 实现 contact force constraint
* [ ] 实现 task logging

控制变量：

```text
u = [base_twist, tendon_velocity]
```

控制目标：

```text
tip pose tracking
collision avoidance
contact force limiting
surface path following
```

---

## 4.7 运行脚本

* [ ] 新增 `scripts/run_engine_navigation.py`
* [ ] 新增 `scripts/run_engine_surface_interaction.py`
* [ ] 支持保存运行日志
* [ ] 支持保存 tip trajectory
* [ ] 支持保存 contact log
* [ ] 支持保存 camera observation

建议命令：

```bash
python scripts/run_engine_navigation.py \
  --task configs/tasks/mujoco_engine_navigation.yaml \
  --save-run outputs/engine_navigation_smoke
```

```bash
python scripts/run_engine_surface_interaction.py \
  --task configs/tasks/engine_surface_path.yaml \
  --save-run outputs/engine_surface_interaction
```

---

## 4.8 验收标准

* [ ] 发动机和机械臂在同一个 MuJoCo 场景中正确显示
* [ ] 初始相对位姿可配置
* [ ] base 能移动到入口前方
* [ ] 主臂能沿预设路径进入
* [ ] 能检测主臂与发动机碰撞
* [ ] 能记录接触力、穿透深度、最小距离
* [ ] 能完成一次最小闭环任务演示
* [ ] 相关测试通过

---

# 5. 从臂手眼相机与视觉反馈

## 5.1 目标

在 observer arm 上搭载手眼相机，支持 MuJoCo 中的视觉信息反馈。

---

## 5.2 Camera Model

* [ ] 定义 camera intrinsics
* [ ] 定义 camera extrinsics
* [ ] 输出 RGB image
* [ ] 输出 depth image
* [ ] 输出 segmentation image
* [ ] 输出 `T_world_camera`
* [ ] 输出 `T_base_camera`
* [ ] 编写 camera observation 数据结构

建议结构：

```python
CameraObservation:
    rgb
    depth
    segmentation
    intrinsics
    T_world_camera
    timestamp
```

---

## 5.3 MuJoCo 渲染接口

* [ ] 从 MuJoCo camera 读取 RGB
* [ ] 从 MuJoCo camera 读取 depth
* [ ] 支持指定 camera name
* [ ] 支持保存图像
* [ ] 支持离线 replay
* [ ] 支持与任务日志同步

---

## 5.4 视觉反馈任务

初期目标：

* [ ] 只做图像获取
* [ ] 只验证 observer camera 视野
* [ ] 只保存 RGB/depth

中期目标：

* [ ] 检测发动机入口
* [ ] 检测主臂末端
* [ ] 检测目标区域
* [ ] 将图像点转换为 world 3D 坐标

后期目标：

* [ ] 根据视觉反馈修正 entry point
* [ ] 根据视觉反馈修正 surface path
* [ ] 根据视觉反馈闭环控制主臂

---

## 5.5 验收标准

* [ ] observer arm 上 camera 位姿正确
* [ ] camera 能看到主臂或发动机目标区域
* [ ] RGB 图像正常
* [ ] depth 图像正常
* [ ] camera extrinsics 正确
* [ ] 图像和 MuJoCo world 坐标能对应
* [ ] 相关测试通过

---

# 6. 接触检测、碰撞检测与交互控制

## 6.1 碰撞检测

* [ ] 主臂与发动机碰撞检测
* [ ] 主臂与基座碰撞检测
* [ ] 主臂与从臂碰撞检测，后续开启
* [ ] 从臂与发动机碰撞检测，后续开启
* [ ] 工具与发动机碰撞检测
* [ ] 可视化接触点

---

## 6.2 接触信息记录

* [ ] 接触 body / geom 名称
* [ ] 接触点位置
* [ ] 接触法向
* [ ] 接触力
* [ ] 穿透深度
* [ ] 最小距离
* [ ] 时间戳

---

## 6.3 接触约束控制

* [ ] 设置最大接触力
* [ ] 设置最小 clearance
* [ ] 设置最大穿透深度
* [ ] 接触过大时 retreat
* [ ] 接触过大时降低 base/arm 速度
* [ ] 接触过大时切换任务阶段

---

## 6.4 验收标准

* [ ] 能稳定读取 MuJoCo contact 信息
* [ ] 能保存 contact log
* [ ] 能在 viewer 中显示接触点
* [ ] 接触力超过阈值时控制器能响应
* [ ] 接触任务不会导致仿真发散

---

# 7. 文档与演示

## 7.1 README 更新

* [ ] 增加项目整体功能介绍
* [ ] 增加分支说明
* [ ] 增加环境安装说明
* [ ] 增加基础运行命令
* [ ] 增加发动机场景说明
* [ ] 增加 6D base 控制说明
* [ ] 增加双臂模型说明
* [ ] 增加任务运行说明
* [ ] 增加测试命令

---

## 7.2 开发文档

建议新增：

```text
docs/engine_frame_alignment.md
docs/mobile_base_6d_control.md
docs/dual_arm_import.md
docs/engine_navigation_task.md
docs/observer_camera_feedback.md
docs/contact_interaction_control.md
```

---

## 7.3 演示材料

* [ ] 发动机坐标系可视化截图
* [ ] entry/path 可视化截图
* [ ] base 6D 控制演示
* [ ] 双臂导入演示
* [ ] 主臂进入发动机演示
* [ ] 接触交互演示
* [ ] observer camera 图像演示

---

# 8. 推荐长期实现顺序

```text
1. 固定开发基线，跑通当前测试
2. 发动机坐标系对齐
3. entry point / initial path 坐标报告
4. entry/path/bbox/frame 可视化
5. 单臂增加 6D base
6. 手动 UI 控制 6D base
7. base + arm 协同 FK / IK
8. 单臂导入发动机场景
9. base 移动到入口前方
10. 主臂沿 initial path 进入发动机
11. SolidWorks 双臂模型导入
12. 主臂 active，从臂 visual-only
13. 从臂 camera site 和 MuJoCo camera
14. observer camera RGB/depth 反馈
15. 主臂与发动机接触检测
16. 接触力约束控制
17. 表面路径跟踪和交互任务
18. 从臂参与碰撞检测
19. 从臂参与视觉闭环
20. 完整 engine navigation + interaction demo
```

---

# 9. 当前优先级最高的近期 TODO

## P0：马上做

* [ ] 确认当前 feature 分支所有相关测试结果
* [ ] 梳理 engine scene 配置中的坐标 frame
* [ ] 写 engine alignment report 脚本
* [ ] 在 MuJoCo 中可视化 entry point 和 initial path
* [ ] 确认 entry/path 坐标是否和发动机 mesh 对齐

## P1：坐标系稳定后做

* [ ] 实现单臂 6D base
* [ ] 增加 base box 可视化
* [ ] 增加 UI 手动控制
* [ ] 将 base pose 纳入 world kinematics
* [ ] 将 base twist 纳入控制器

## P2：单臂 + base 稳定后做

* [ ] 导入双臂 SolidWorks visual mesh
* [ ] 配置主臂 active
* [ ] 配置从臂 visual-only
* [ ] 预留 observer camera site
* [ ] 测试主臂控制不受从臂影响

## P3：集成阶段

* [ ] 连续体臂放入发动机场景
* [ ] 支持相对于 entry_port 的初始位姿
* [ ] 实现 base 到入口前方的导航
* [ ] 实现主臂沿 initial path 进入
* [ ] 实现接触检测和日志记录

## P4：高级功能

* [ ] 接触力约束控制
* [ ] 表面路径跟踪
* [ ] observer camera RGB/depth
* [ ] 视觉反馈闭环
* [ ] 从臂参与碰撞和任务执行
