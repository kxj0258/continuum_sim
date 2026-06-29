# Phase 1.0 Engine Aligned Baseline

## 日期

2026-06-29

## 当前分支

feat/engine-dual-arm-foundation

## 本次目标

- 修复 `tests/test_engine_aligned_scene.py` 与当前 engine aligned / nozzle collision 配置语义不一致的问题。
- 为 Phase 1 发动机坐标系对齐建立干净测试基线。

## 问题现象

全量测试原始结果为：

```text
362 passed, 2 failed
```

失败集中在：

```text
tests/test_engine_aligned_scene.py
```

原始失败摘要：

```text
config.engine.pose.position_m 实际为 [0.0, 0.0, 0.0]
测试仍期望 [4.043, 1.12127, 0.0]
```

## 原因分析

当前 `configs/scenes/engine_cleaning_aligned.yaml` 与
`configs/scenes/engine_cleaning_nozzle_collision.yaml` 都显式使用：

```text
engine.pose.position_m = [0.0, 0.0, 0.0]
engine.pose.quat_wxyz = [1.0, 0.0, 0.0, 0.0]
```

并且最近提交 `93621a4` 同时把这两个配置改成了当前 origin / identity pose 语义，
但 `tests/test_engine_aligned_scene.py` 却被改成了另一套 `y-forward` 硬编码 pose：

```text
position_m = [4.043, 1.12127, 0.0]
quat_wxyz = [0.5, 0.5, -0.5, -0.5]
```

这说明本次失败的根因不是当前配置加载逻辑错误，而是测试期望没有和当前配置语义同步。

因此本次选择的修复方向是：

- 不修改配置；
- 只修正测试，使其真实表达当前基线语义；
- 同时保留足够具体的断言，确保测试不是“放水通过”。

## 修改文件

```text
tests/test_engine_aligned_scene.py
docs/logs/2026-06-29-phase-1-0-engine-aligned-baseline.md
```

## 修改说明

### 1. `tests/test_engine_aligned_scene.py`

- 将 aligned scene 的 pose 断言更新为当前配置真实值：
  - `position_m == [0.0, 0.0, 0.0]`
  - `quat_wxyz == [1.0, 0.0, 0.0, 0.0]`
- 将 nozzle collision scene 的 pose 检查改为：
  - 与 aligned scene 的 engine pose 保持一致；
  - 并且当前仍固定为 origin / identity baseline。
- 删除了对过时 `y-forward` pose 的硬编码依赖。
- 保留并补强了 nozzle collision scene 的真实约束检查：
  - primitive collision hints 名称与 disabled 状态；
  - `collision_mesh_offset_m == [0.0, 0.0, 0.0]`；
  - capsule hint 的 `fromto_m`；
  - exploration path `nozzle_axis_entry` 的 frame 与点位；
  - 当前 nozzle collision scene 只保留 `entry_port` 区域这一事实。

### 2. 开发日志

- 新增本文件，记录本次基线收口的原因、修改和测试结果。

## 运行命令

```bash
git branch --show-current
git status --short --branch
```

```bash
conda run -n continuum_sim python -m pytest tests/test_engine_aligned_scene.py --basetemp .tmp_pytest_engine_aligned -p no:cacheprovider -v
conda run -n continuum_sim python -m pytest -m core --basetemp .tmp_pytest_core -p no:cacheprovider
conda run -n continuum_sim python -m pytest tests/test_base_pose.py tests/test_world_kinematics.py --basetemp .tmp_pytest_pose -p no:cacheprovider
conda run -n continuum_sim python -m pytest tests/test_engine_scene.py tests/test_engine_cleaning_config.py --basetemp .tmp_pytest_engine_cfg -p no:cacheprovider
conda run -n continuum_sim python -m pytest tests/test_multi_arm_model.py tests/test_multi_arm_state.py --basetemp .tmp_pytest_multi_arm -p no:cacheprovider
conda run -n continuum_sim python -m pytest tests/test_camera_model.py --basetemp .tmp_pytest_camera -p no:cacheprovider
conda run -n continuum_sim python -m pytest --basetemp .tmp_pytest_full -p no:cacheprovider
```

## 测试结果

```text
tests/test_engine_aligned_scene.py: 7 passed
-m core: 91 passed, 273 deselected
tests/test_base_pose.py tests/test_world_kinematics.py: 17 passed
tests/test_engine_scene.py tests/test_engine_cleaning_config.py: 13 passed
tests/test_multi_arm_model.py tests/test_multi_arm_state.py: 15 passed
tests/test_camera_model.py: 8 passed
full pytest: 364 passed
```

## 当前剩余问题

暂无。

## 下一步建议

建议进入 Phase 1：发动机坐标系对齐，包括：

1. 统一 scene config 中的 frame 语义。
2. 增加 entry/path 的 MuJoCo/world 坐标报告。
3. 增加 entry point、normal、path、bbox、frame 可视化。
