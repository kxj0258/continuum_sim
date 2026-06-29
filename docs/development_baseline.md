# Development Baseline

This baseline was established on `2026-06-29`.
All test commands below assume the project environment is active first:

```bash
conda activate continuum_sim
```

## 1. Branch Overview

### `main`

- 当前定位：
  单臂连续体机械臂仿真主线，覆盖 PCC、MuJoCo tracking / navigation / wiping，以及稳定的 CLI 与回归测试体系。
- 主要功能：
  `AnalyticBackend` / `MujocoBackend`、PCC 运动学、tendon / motor 映射、结构化 navigation scene、wiping 任务、运行结果导出与 replay。
- 关键目录：
  `configs/`、`scripts/`、`src/continuum_sim/{actuation,backends,control,kinematics,model,runtime,scenes,tasks,visualization}`、`tests/`、`docs/`
- 关键运行命令：
  `python cli.py view-pcc --config configs/main_config.yaml`
  `python cli.py view-mujoco --config configs/main_config.yaml`
  `python cli.py run-mujoco-tracking --config configs/main_config.yaml`
  `python cli.py run-mujoco-navigation --config configs/main_config.yaml`
  `python cli.py run-mujoco-wiping --config configs/main_config.yaml`
- 当前测试状态：
  `conda run -n continuum_sim python -m pytest --basetemp .tmp_pytest_full -p no:cacheprovider`
  结果：`218 passed`

### `feat/engine-dual-arm-foundation`

- 当前定位：
  在 `main` 稳定主线之上，向 engine scene、mobile base、dual arm、tool / camera attachment、engine task scaffold 扩展的长期研发分支。
- 相比 `main` 的新增内容：
  新增 engine mesh 资产、engine scene YAML、engine 诊断脚本、6D base pose / mount frame、multi-arm 配置与状态、camera intrinsics scaffold、tool attachment 配置、engine task/config/controller scaffold，以及对应测试。
- 主要功能：
  保留 `main` 全部单臂能力，同时增加：
  engine scene loader / preview / asset diagnostics、
  mobile base pose 与 mount transform、
  dual continuum arm 配置模型、
  observer camera / tool attachment scaffold、
  engine surface path 与清理控制 scaffold。
- 关键目录：
  `assets/engine/`
  `configs/scenes/engine_*.yaml`
  `configs/robots/{mobile_base_pose.yaml,dual_continuum.yaml}`
  `configs/tools/`
  `scripts/{check_engine_assets.py,preview_engine_scene_mujoco.py,suggest_engine_pose.py,suggest_nozzle_collision.py}`
  `src/continuum_sim/{scenes,model,runtime,sensing,tools,control,tasks}`
  `tests/test_engine_*.py`
  `tests/test_{base_pose,world_kinematics,multi_arm_model,multi_arm_state,camera_model}.py`
- 关键运行命令：
  `python cli.py run-mujoco-navigation --config configs/main_config.yaml`
  `python scripts/check_engine_assets.py --config configs/scenes/engine_cleaning_aligned.yaml`
  `python scripts/preview_engine_scene_mujoco.py --config configs/scenes/engine_cleaning_aligned.yaml --headless-check`
  `python scripts/preview_engine_scene_mujoco.py --config configs/scenes/engine_cleaning_aligned.yaml --viewer --show-bbox --show-regions --show-axes`
  `python scripts/suggest_engine_pose.py --config configs/scenes/engine_cleaning.yaml --mode grounded --write-aligned-config configs/scenes/engine_cleaning_aligned.yaml`
- 当前测试状态：
  `conda run -n continuum_sim python -m pytest --basetemp .tmp_pytest_full -p no:cacheprovider`
  结果：`362 passed, 2 failed`
  当前失败集中在 `tests/test_engine_aligned_scene.py`，反映 engine aligned / nozzle collision 配置中的 `engine.pose.position_m` 与测试期望不一致。

## 2. Branch Comparison Summary

| 模块 | `main` | `feat/engine-dual-arm-foundation` | 后续建议 |
|---|---|---|---|
| core model | 单臂 PCC / tendon / motor 主链稳定 | 保持兼容，并新增 `Pose6D`、mount、multi-arm scaffold | 继续以 `main` 的稳定主链为兼容基线 |
| MuJoCo backend | tracking / navigation / wiping 完整可用 | 继承 `main`，并增加 engine preview / diagnostics 脚本 | 不重写已有 MuJoCo runtime，新增能力尽量旁路扩展 |
| engine scene | 无 engine scene 专项模块 | 新增 engine assets、scene loader、surface/path、preview、pose/nozzle 建议脚本 | 后续 engine 相关开发全部落在该分支继续推进 |
| mobile base | 无 | 新增 `base_pose.py`、`mount_frame.py`、`world_kinematics.py`、`configs/robots/mobile_base_pose.yaml` | 后续 6D base 控制从这里继续扩展，不回灌到 `main` |
| dual arm / multi arm | 无 | 新增 `multi_arm.py`、`multi_arm_state.py`、`configs/robots/dual_continuum.yaml` | 以 scaffold 为基础，先做主臂可控、从臂被动显示 |
| camera / sensing | 无独立 camera 模块 | 新增 `sensing/camera_model.py` 与 `configs/tools/eye_camera_air_gun.yaml` | 先补 MuJoCo camera 挂载与图像输出，再做视觉反馈 |
| task configs | 以 tracking / navigation / wiping 为主 | 新增 engine cleaning controller / surface path / tool / robot / scene configs | engine 任务独立扩展，不污染老 task config |
| tests | `218` 项，聚焦单臂主线 | `364` 项，新增 engine / mobile base / dual arm / camera / tool 相关测试 | 后续长期开发应维护并扩展 `feat/...` 的新增测试层 |

## 3. Recommended Development Branch

建议后续以 `feat/engine-dual-arm-foundation` 作为长期开发主线。

原因：

1. 该分支已经具备 engine scene、mobile base、dual arm、camera / tool attachment 的最小结构与配置入口，而 `main` 没有这些开发支点。
2. 该分支保留了 `main` 的绝大部分稳定能力，并新增了成体系的测试文件，可支撑后续阶段化开发。
3. 后续目标集中在发动机场景、6D 基座、双臂导入和观测相机，这些方向都已经在 `feat/engine-dual-arm-foundation` 中形成了可继续演进的 scaffold。

## 4. Minimal Regression Test Set

后续每次开发完成后，建议至少运行：

```bash
python -m pytest -m core --basetemp .tmp_pytest_core -p no:cacheprovider
python -m pytest tests/test_base_pose.py tests/test_world_kinematics.py --basetemp .tmp_pytest_pose -p no:cacheprovider
python -m pytest tests/test_engine_scene.py tests/test_engine_cleaning_config.py --basetemp .tmp_pytest_engine_cfg -p no:cacheprovider
python -m pytest tests/test_multi_arm_model.py tests/test_multi_arm_state.py --basetemp .tmp_pytest_multi_arm -p no:cacheprovider
python -m pytest tests/test_camera_model.py --basetemp .tmp_pytest_camera -p no:cacheprovider
```

当前核查结果：

- `feat/engine-dual-arm-foundation`
  - `-m core`：`91 passed`
  - `tests/test_base_pose.py tests/test_world_kinematics.py`：`17 passed`
  - `tests/test_engine_scene.py tests/test_engine_cleaning_config.py`：`13 passed`
  - `tests/test_multi_arm_model.py tests/test_multi_arm_state.py`：`15 passed`
  - `tests/test_camera_model.py`：`8 passed`
- `main`
  - `tests/test_base_pose.py`：当前分支不存在该测试文件
  - `tests/test_world_kinematics.py`：当前分支不存在该测试文件
  - `tests/test_engine_scene.py`：当前分支不存在该测试文件
  - `tests/test_engine_cleaning_config.py`：当前分支不存在该测试文件
  - `tests/test_multi_arm_model.py`：当前分支不存在该测试文件
  - `tests/test_multi_arm_state.py`：当前分支不存在该测试文件
  - `tests/test_camera_model.py`：当前分支不存在该测试文件

补充建议：

```bash
python -m pytest tests/test_engine_aligned_scene.py --basetemp .tmp_pytest_engine_aligned -p no:cacheprovider
```

说明：
当前 `feat/engine-dual-arm-foundation` 的全量失败全部集中在 `tests/test_engine_aligned_scene.py`。
如果后续开发涉及 engine aligned pose、nozzle collision config 或 preview 对齐逻辑，应把该测试加入阶段性回归集合。

## 5. Current Known Issues

| 问题 | 所在分支 | 复现命令 | 报错摘要 | 初步判断 | 后续处理建议 |
| -- | ---- | ---- | ---- | ---- | ------ |
| engine aligned scene 测试与当前配置不一致 | `feat/engine-dual-arm-foundation` | `python -m pytest tests/test_engine_aligned_scene.py --basetemp .tmp_pytest_engine_aligned -p no:cacheprovider` | `config.engine.pose.position_m` 实际为 `[0.0, 0.0, 0.0]`，测试期望为 `[4.043, 1.12127, 0.0]` | 更像是配置语义或测试期望没有同步更新，而不是运行时崩溃 | 在进入 Phase 1 发动机坐标系对齐前，先统一 aligned / nozzle collision 配置与测试期望 |
| `main` 缺少 engine / mobile base / dual arm / camera 专项测试 | `main` | 文件存在性检查 | 相关测试文件不存在 | `main` 本身未承载这些能力 | 后续不要在 `main` 上直接推进长期功能开发 |
| 并行 `conda run` 在 Windows 下会争抢临时激活文件 | 环境层问题 | 并行执行多个 `conda run -n continuum_sim ...` | `The process cannot access the file because it is being used by another process.` | Conda 临时激活文件竞争，不是项目逻辑问题 | Windows 下回归测试建议串行执行 |

## 6. Suggested Feature Branches

```bash
feat/engine-frame-alignment
feat/mobile-base-6d-control
feat/dual-arm-import-passive-observer
feat/engine-arm-task-integration
feat/observer-camera-feedback
```

## 7. Development Rules

* 不直接在 `main` 上开发长期功能。
* 每个功能点从 `feat/engine-dual-arm-foundation` 拆出独立 feature 分支。
* 每次修改前先确认 `git status`。
* 每次提交前至少运行最小回归测试集合。
* 涉及 engine aligned pose / nozzle collision 的修改，额外运行 `tests/test_engine_aligned_scene.py`。
* 每个阶段完成后更新开发记录。
