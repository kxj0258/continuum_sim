# 配置参考

除非字段另有说明，所有数值都使用 SI 单位。路径通常相对声明它的 YAML 文件解析；在适用时，也会回退到项目工作目录解析。

## `configs/main_config.yaml`

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

## CLI 运行产物导出

`run-tracking`、`run-mujoco-tracking`、`run-mujoco-navigation` 和
`run-mujoco-wiping` 默认不保存 rollout 数据。加入 `--save-run` 后，会把本次运行写入
`output/runs/<task_name>_<timestamp>/`：

```powershell
python cli.py run-mujoco-wiping --config configs/main_config.yaml --save-run
```

保存产物包括 `result.npz`、`metadata.json`、复制后的 YAML 配置、PNG 曲线图、可用时生成的
MuJoCo 场景 XML，以及 `videos/simulation.gif`。MuJoCo 回放视频会在独立子进程中根据保存的
`qpos/qvel` 历史和归档的 `model/scene.xml` 导出，尺寸使用 MuJoCo 后端中的
`rendering.offscreen_*`，相机使用 `viewer.camera`。如果离屏渲染或视频编码不可用，命令仍会保存
NPZ/PNG，失败原因写入 `videos/video_error.txt`。

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
| `solver.timestep`                 | float, s     | MuJoCo 积分 timestep。                           |
| `solver.integrator`               | string       | MuJoCo integrator。                              |
| `solver.iterations`               | int          | MuJoCo solver 迭代次数。                         |
| `gravity.enabled`                 | bool         | 是否启用重力。                                   |
| `gravity.vector_m_s2`             | list[float]  | 重力向量。                                       |
| `joints.hinge.*`                  | scalar       | hinge damping、armature、limit、range、stiffness 和 springref。 |
| `tendon_model.*`                  | scalar       | fixed tendon 数量、限幅、damping、stiffness 和 coefficient source。 |
| `actuators.tendon_position.*`     | scalar       | tendon position gain、命令范围和力范围。         |
| `actuators.joint_position.*`      | scalar       | joint position gain、命令范围和力矩范围。        |
| `sensors.*`                       | bool         | tendon length、velocity 和 actuator force sensor 开关。 |
| `smoke_tests.*`                   | scalar       | MuJoCo smoke 路径使用的轻量数值检查参数。        |
| `rendering.offscreen_width`       | int          | MuJoCo replay video 离屏 framebuffer 宽度。      |
| `rendering.offscreen_height`      | int          | MuJoCo replay video 离屏 framebuffer 高度。      |
| `site_names.*`                    | string/list  | base、segment tip 和最终 tip site 名称。         |
| `notes`                           | list[string] | 面向人的备注。                                   |

`viewer.camera.*` 同时影响 passive viewer、生成 XML 的默认 MuJoCo visual camera，以及
`--save-run` 的 MuJoCo replay GIF。`rendering.offscreen_*` 只控制离屏导出 framebuffer
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

### DMP Tracking 轨迹

Tracking YAML 支持 `trajectory.type: dmp`：

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

MuJoCo navigation YAML 支持：

```yaml
controller:
  type: navigation_cbf_qp
```

当前实现使用小规模 NumPy 投影处理 CBF 半空间约束。后续如果加入大量同步约束，可以在不改变
navigation runtime API 的前提下替换为 OSQP。

### PCC 降阶动力学

`configs/dynamics/pcc_reduced.yaml` 保存实验性动力学擦拭控制器使用的工程估计参数。

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

- `hybrid_force_position`：原有运动学级切向位置/法向力控制器，仍然是默认基线。
- `dynamic_adaptive_impedance`：实验控制器，先使用 PCC 降阶动力学预测降阶状态速度，再映射回电机速度。

动力学实验任务示例：

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
CLI `--save-run` 生成 `result.npz`，再传入脚本分析：

```powershell
python cli.py run-mujoco-wiping --config configs/main_config_dynamic_wiping.yaml --save-run
python scripts/test_indicator_3_3_force_tracking.py --result-npz output/runs/<run>/result.npz
```

如果不提供 `--result-npz`，脚本会尝试无窗口运行对应 MuJoCo 任务。动态阻抗模式需要在每个控制步计算
PCC 质量矩阵，耗时会明显长于只分析已保存 NPZ。
