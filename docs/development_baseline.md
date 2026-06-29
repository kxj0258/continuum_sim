# Development Baseline

This baseline was established on `2026-06-29` and refreshed on `2026-06-29`
after the engine preview/frame cleanup. All test commands below assume the
project environment is active first:

```bash
conda activate continuum_sim
```

## 1. Branch Overview

### `main`

- 当前定位：
  单臂连续体机械臂仿真主线，覆盖 PCC、MuJoCo tracking / navigation /
  wiping，以及稳定的 CLI 与回归测试体系。
- 主要功能：
  `AnalyticBackend` / `MujocoBackend`、PCC 运动学、tendon / motor 映射、
  navigation scene、wiping task、运行结果导出与 replay。
- 关键目录：
  `configs/`、`scripts/`、
  `src/continuum_sim/{actuation,backends,control,kinematics,model,runtime,scenes,tasks,visualization}`、
  `tests/`、`docs/`
- 关键运行命令：
  `python cli.py view-pcc --config configs/main_config.yaml`
  `python cli.py view-mujoco --config configs/main_config.yaml`
  `python cli.py run-mujoco-tracking --config configs/main_config.yaml`
  `python cli.py run-mujoco-navigation --config configs/main_config.yaml`
  `python cli.py run-mujoco-wiping --config configs/main_config.yaml`
- 当前测试状态：
  历史基线检查命令：
  `conda run -n continuum_sim python -m pytest --basetemp .tmp_pytest_full -p no:cacheprovider`
  结果：`218 passed`

### `feat/engine-dual-arm-foundation`

- 当前定位：
  在 `main` 稳定主线上，向 engine scene、mobile base、dual arm、
  tool / camera attachment、engine task scaffold 扩展的长期研发分支。
- 相比 `main` 的新增内容：
  engine mesh 资产、engine scene loader / preview / diagnostics、
  `Pose6D` / `world_kinematics`、multi-arm config/state、tool/camera
  scaffold、engine cleaning task/config/controller scaffold。
- 主要功能：
  保留 `main` 全部单臂能力，同时新增 engine scene 可视化与诊断、
  engine-frame 路径标注、6D base pose、双臂配置骨架、相机/工具附件骨架。
- 关键目录：
  `assets/engine/`
  `configs/scenes/engine_cleaning.yaml`
  `configs/robots/{mobile_base_pose.yaml,dual_continuum.yaml}`
  `configs/tools/`
  `scripts/{check_engine_assets.py,preview_engine_scene_mujoco.py,suggest_engine_pose.py,suggest_nozzle_collision.py}`
  `src/continuum_sim/{scenes,model,runtime,sensing,tools,control,tasks}`
  `tests/test_engine_*.py`
  `tests/test_{base_pose,world_kinematics,multi_arm_model,multi_arm_state,camera_model}.py`
- 关键运行命令：
  `python cli.py run-mujoco-navigation --config configs/main_config.yaml`
  `python scripts/check_engine_assets.py --config configs/scenes/engine_cleaning.yaml`
  `python scripts/preview_engine_scene_mujoco.py --config configs/scenes/engine_cleaning.yaml --headless-check`
  `python scripts/preview_engine_scene_mujoco.py --config configs/scenes/engine_cleaning.yaml --viewer`
  `python scripts/suggest_engine_pose.py --config configs/scenes/engine_cleaning.yaml --mode grounded --write-aligned-config configs/scenes/engine_cleaning_grounded.generated.yaml`
  `python scripts/suggest_nozzle_collision.py --config configs/scenes/engine_cleaning.yaml --source collision --primitive capsule --output-config configs/scenes/engine_cleaning_with_nozzle_collision.yaml`
- 当前测试状态：
  本次刷新已重新验证 engine preview / scene 相关子集，详见
  “Minimal Regression Test Set” 和“Current Known Issues”。

## 2. Branch Comparison Summary

| 模块 | `main` | `feat/engine-dual-arm-foundation` | 后续建议 |
|---|---|---|---|
| core model | 单臂 PCC / tendon / motor 主链稳定 | 保持兼容，并新增 `Pose6D`、mount、multi-arm scaffold | 继续以 `main` 的稳定主链为兼容基线 |
| MuJoCo backend | tracking / navigation / wiping 完整可用 | 继承 `main`，并增加 engine preview / diagnostics 脚本 | 不重写已有 MuJoCo runtime，新能力尽量旁路扩展 |
| engine scene | 无 engine scene 专项模块 | 新增 engine assets、scene loader、preview、pose / nozzle 建议脚本 | engine 相关功能继续在该分支推进 |
| mobile base | 无 | 新增 `base_pose.py`、`mount_frame.py`、`world_kinematics.py` | 后续 6D base 控制从这里扩展 |
| dual arm / multi arm | 无 | 新增 `multi_arm.py`、`multi_arm_state.py`、`dual_continuum.yaml` | 先做主臂可控、从臂被动显示 |
| camera / sensing | 无独立 camera 模块 | 新增 `sensing/camera_model.py` 和 observer attachment scaffold | 先补 MuJoCo camera 挂载与图像输出 |
| task configs | 以 tracking / navigation / wiping 为主 | 新增 engine cleaning controller / surface path / tool / robot / scene configs | engine 任务独立扩展，不污染原 task config |
| tests | 聚焦单臂主线 | 新增 engine / mobile base / dual arm / camera / tool 测试 | 长期开发应维护并扩展该分支测试层 |

## 3. Recommended Development Branch

建议后续以 `feat/engine-dual-arm-foundation` 作为长期开发主线。

原因：

1. 该分支已经具备 engine scene、mobile base、dual arm、camera / tool
   attachment 的最小结构与配置入口，而 `main` 没有这些开发支点。
2. 该分支保留了 `main` 的稳定能力，同时补齐了后续阶段需要的 scene、
   pose、preview、task scaffold 与对应测试。
3. 你接下来的工作重点正是发动机坐标系对齐、6D 基座、双臂导入和视觉反馈，
   这些都已经在该分支形成可持续演进的基础。

## 4. Minimal Regression Test Set

后续每次开发完成后，建议至少运行：

```bash
python -m pytest -m core --basetemp .tmp_pytest_core -p no:cacheprovider
python -m pytest tests/test_base_pose.py tests/test_world_kinematics.py --basetemp .tmp_pytest_pose -p no:cacheprovider
python -m pytest tests/test_engine_scene.py tests/test_engine_cleaning_config.py --basetemp .tmp_pytest_engine_cfg -p no:cacheprovider
python -m pytest tests/test_multi_arm_model.py tests/test_multi_arm_state.py --basetemp .tmp_pytest_multi_arm -p no:cacheprovider
python -m pytest tests/test_camera_model.py --basetemp .tmp_pytest_camera -p no:cacheprovider
```

如果修改涉及 engine preview / engine-frame overlay / nozzle hint，额外运行：

```bash
python -m pytest tests/test_engine_asset_checks.py tests/test_engine_aligned_scene.py tests/test_nozzle_collision_suggestions.py tests/test_engine_pose_suggestions.py --basetemp .tmp_pytest_engine_preview -p no:cacheprovider
```

当前核查结论：

- `main`
  - 上述 engine / base / multi-arm / camera 专项测试文件在该分支多数不存在。
- `feat/engine-dual-arm-foundation`
  - 需要长期保留上述最小回归集合。
  - engine preview 相关修改后，建议把额外的 preview 子集一起纳入阶段回归。

## 5. Current Known Issues

| 问题 | 所在分支 | 复现命令 | 报错摘要 | 初步判断 | 后续处理建议 |
| -- | ---- | ---- | ---- | ---- | ------ |
| 并行 `conda run` 在 Windows 下会争抢临时激活文件 | 环境层问题 | 并行执行多个 `conda run -n continuum_sim ...` | `The process cannot access the file because it is being used by another process.` | Conda 临时激活文件竞争，不是项目逻辑问题 | Windows 下回归测试建议串行执行 |
| nozzle primitive collision hint 仍需人工确认 | `feat/engine-dual-arm-foundation` | `python scripts/preview_engine_scene_mujoco.py --config configs/scenes/engine_cleaning_with_nozzle_collision.yaml --viewer --show-primitive-collision --show-disabled-hints` | 无自动报错，但 hint 仅基于 bbox 生成 | 当前只是诊断/初始化辅助几何，不应直接当作最终碰撞模型 | 在 viewer 中确认尺寸、方向、启用策略后再纳入任务场景 |

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
* 涉及 engine preview / frame overlay / nozzle hint 的修改，额外运行 preview 子集测试。
* 每个阶段完成后更新开发记录。
