# continuum_sim

`continuum_sim` 是一个面向三段腱驱连续体机械臂的轻量研究仿真代码库。它维护一条清晰主链：

```text
YAML 配置
  -> 电机/物理腱映射
  -> PCC 正运动学和微分运动学
  -> differential IK 控制
  -> analytic PCC 或 MuJoCo 后端
  -> tracking / navigation / wiping 任务
```

适合用于控制链验证、回归测试、MuJoCo 可视化调试和实验结果保存。它不是高保真 FEM 软体仿真器，也不是硬件驱动程序。

## 安装

推荐使用 Conda：

```powershell
conda env create -f environment.yml
conda activate continuum_sim
```

已有环境时：

```powershell
conda env update -n continuum_sim -f environment.yml
conda activate continuum_sim
```

不使用 Conda：

```powershell
python -m pip install -e .
```

MuJoCo 是可选依赖：

```powershell
python -m pip install -e .[mujoco]
```

## 快速运行

所有维护中的入口都在 `cli.py`，默认从 `configs/main_config.yaml` 开始解析。命令行只保留少量入口参数，具体行为写在 YAML 里，便于复现实验。

```powershell
python cli.py view-pcc --config configs/main_config.yaml
python cli.py view-motor-chain --config configs/main_config.yaml
python cli.py run-tracking --config configs/main_config.yaml
python cli.py view-mujoco --config configs/main_config.yaml
python cli.py debug-mujoco-tendons --config configs/main_config.yaml
python cli.py run-mujoco-tracking --config configs/main_config.yaml
python cli.py run-mujoco-navigation --config configs/main_config.yaml
python cli.py run-mujoco-wiping --config configs/main_config.yaml
```

需要无窗口运行时，改对应 YAML 里的 `visualization.show` 或 `viewer.show`。

`debug-mujoco-tendons` 会打开 MuJoCo passive viewer 和 tendon debug panel，用来检查 tendon-position 命令、传感器读数和 actuator force；如果只想做无窗口 smoke check，可以把对应 MuJoCo YAML 里的 `viewer.show` 设为 `false`。

## 当前功能

- PCC 正运动学、中心线采样、有限差分 Jacobian。
- 电机位置/速度与物理腱长/腱速映射。
- 末端位置 tracking 的 damped least-squares differential IK。
- `AnalyticBackend` 和可选 `MujocoBackend`。
- MuJoCo tendon-position 和 legacy joint-position 控制模式。
- 结构化 navigation 场景：火箭腔体、障碍物、中心线 clearance 查询。
- wiping 任务：板面作业面、tip contact pad、raster 路径、法向力/距离 proxy、live force panel。
- `--save-run` 导出 NPZ、metadata、配置副本、图和可选 replay video。

## MuJoCo 模型

默认配置是：

```text
configs/mujoco.yaml
  model.type: distributed_links
```

它使用三段、每段四个降阶 link、每个 link 两个 hinge joint，以及 9 个 tendon-position actuator。

另一个模型是：

```text
configs/mujoco_segment_2dof.yaml
  model.type: segment_2dof_followers
```

它每段只有 x/y 两个物理弯曲 DOF，三段共 6 个 q。视觉和碰撞由 runtime follower mocap bodies 采样得到，不额外增加广义坐标。wiping 的 `mujoco_actual` feedback 会优先使用 follower contact projection，失败时回退到 tool-pad MuJoCo contact force，再回退到距离 proxy。

生成或检查 MuJoCo 资产：

```powershell
python scripts/build_mujoco_tendon_model.py --config configs/mujoco.yaml
python scripts/check_mujoco_segment_visuals.py --config configs/mujoco.yaml
python scripts/build_mujoco_with_segment_visuals.py --config configs/mujoco.yaml
python scripts/build_mujoco_segment_2dof_model.py --config configs/mujoco_segment_2dof.yaml
python scripts/check_mujoco_offscreen_renderer.py --config configs/mujoco.yaml
```

默认观察相机写在 `configs/mujoco.yaml` 的 `viewer.camera`。这组参数同时用于 passive viewer、生成 XML 中的默认 MuJoCo visual camera，以及 `--save-run` 保存的 MuJoCo replay GIF。板面 wiping 场景默认采用从工作面前侧斜上方观察的视角，避免黑板遮挡机械臂；如果改了相机参数，建议重新运行上面的 XML 生成脚本保持资产同步。

## 配置入口

从这里开始看：

```text
configs/main_config.yaml
```

常改文件：

- `configs/robot_3seg.yaml`：机器人几何、物理腱、电机、限幅。
- `configs/mujoco.yaml`：MuJoCo XML、solver、actuator、viewer、overlay。
- `configs/mujoco_segment_2dof.yaml`：2DOF follower MuJoCo 模型。
- `configs/tasks/pcc_trajectory_tracking.yaml`：离线 PCC tracking。
- `configs/tasks/mujoco_trajectory_tracking.yaml`：MuJoCo tracking。
- `configs/tasks/mujoco_navigation_rocket.yaml`：结构化障碍 navigation。
- `configs/tasks/mujoco_wiping_board.yaml`：板面 wiping 力位混合控制。
- `configs/scenes/*.yaml`：navigation/wiping 场景。

完整字段说明见 [docs/configuration_reference.md](docs/configuration_reference.md)。

## 保存运行结果

run 类命令默认只打印摘要，不落盘。需要保存时加 `--save-run`：

```powershell
python cli.py run-mujoco-wiping --config configs/main_config.yaml --save-run
```

输出目录：

```text
output/runs/<task_name>_<timestamp>/
```

常见产物：

- `result.npz`：rollout 数组。
- `metadata.json`：命令、任务名、样本数、产物路径。
- `configs/`：本次运行用到的配置副本。
- `model/scene.xml`：navigation/wiping 生成的 MuJoCo scene XML。
- `plots/`：轨迹、误差、电机速度、腱长、wiping 力曲线。
- `videos/simulation.gif`：可选 MuJoCo replay animation。

MuJoCo replay GIF 会从保存的 `qpos/qvel` 和归档的 `model/scene.xml` 在独立子进程中离屏渲染，使用 `configs/mujoco.yaml` 里的 `rendering.offscreen_*` 和 `viewer.camera`。如果 MuJoCo offscreen renderer 或视频编码不可用，数值和图片仍会保存，错误会写入 `videos/video_error.txt`；能退回轨迹动画时，fallback GIF 的 xyz 三轴使用相同比例尺。

想单独检查离屏渲染环境时，可以运行：

```powershell
python scripts/check_mujoco_offscreen_renderer.py --config configs/mujoco.yaml
```

想从已保存的 rollout 重新导出 GIF 时，可以运行：

```powershell
python scripts/export_replay_video.py `
  --result-npz output/runs/<task_name>_<timestamp>/result.npz `
  --scene-xml output/runs/<task_name>_<timestamp>/model/scene.xml `
  --output output/runs/<task_name>_<timestamp>/videos/replay.gif
```

## 测试

```powershell
python -m pytest -m core
python -m pytest -m baseline
python -m pytest
```

常用 marker：

- `core`：快速库层检查，覆盖配置、运动学、驱动映射和控制。
- `baseline`：稳定回归范围，包含 CLI smoke 和核心 MuJoCo 资产检查。
- `cli_smoke`：命令入口子进程检查。
- `mujoco`：MuJoCo 相关测试；未安装 MuJoCo 时会跳过。
- `slow`：较慢回归。

## 仓库结构

```text
configs/                  YAML 配置
assets/                   MuJoCo XML、网格、CAD/mesh 资产
scripts/                  MuJoCo 资产生成与检查脚本
src/continuum_sim/
  actuation/              电机到腱长映射
  backends/               Analytic PCC / MuJoCo 后端
  control/                differential IK、navigation、wiping 控制
  io/                     run artifact 导出
  kinematics/             PCC FK、微分运动学、核心链 facade
  model/                  机器人参数、物理腱、耦合矩阵、2DOF followers
  runtime/                MuJoCo tracking/navigation/wiping 编排
  scenes/                 结构化场景、clearance、作业面、XML 注入
  tasks/                  YAML task loader 和轨迹/路径生成
  visualization/          Matplotlib viewer、panel、plot、video
tests/                    Pytest 回归测试
docs/                     架构与配置文档
```

## 进一步阅读

- [docs/architecture_overview.md](docs/architecture_overview.md)
- [docs/configuration_reference.md](docs/configuration_reference.md)

## 高级控制扩展

原有运动学级擦拭控制器保持不变，仍然是默认回归基线：

```yaml
controller:
  type: hybrid_force_position
```

新增的 PCC 降阶动力学控制器作为实验模式提供，用于力位混合控制研究：

```yaml
controller:
  type: dynamic_adaptive_impedance
  dynamics_config_path: ../dynamics/pcc_reduced.yaml
```

相关示例文件：

- `configs/dynamics/pcc_reduced.yaml`
- `configs/tasks/mujoco_wiping_board_dynamic.yaml`
- `configs/main_config_dynamic_wiping.yaml`

建议先用 CLI 保存一次完整仿真，再把生成的 `result.npz` 交给验收脚本分析。这样比直接运行验收脚本更清楚，也能复现实验数据：

```powershell
python cli.py run-mujoco-navigation --config configs/main_config.yaml --save-run
python scripts/test_indicator_3_1_positioning.py --result-npz output/runs/<run>/result.npz
```

运动学级擦拭基线：

```powershell
python cli.py run-mujoco-wiping --config configs/main_config.yaml --save-run
python scripts/test_indicator_3_3_force_tracking.py --result-npz output/runs/<run>/result.npz
```

动力学级擦拭实验模式：

```powershell
python cli.py run-mujoco-wiping --config configs/main_config_dynamic_wiping.yaml --save-run
python scripts/test_indicator_3_3_force_tracking.py --result-npz output/runs/<run>/result.npz
python scripts/test_indicator_3_3_disturbance.py --result-npz output/runs/<run>/result.npz
```

如果省略 `--result-npz`，验收脚本会自行启动完整 MuJoCo 仿真。动态阻抗模式每个控制步都要计算 PCC 质量矩阵，运行时间会明显更长；看到堆栈停在 `mass_matrix`、`forward_kinematics` 一类函数时通常表示仍在计算，不一定是崩溃。

扰动验收脚本目前记录可配置工程扰动工况；第三方尚未固定扰动力大小、方向和持续时间，因此暂未把某一组 `mj_applyFT` 扰动口径写死。
