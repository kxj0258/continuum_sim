# Phase 2 Mobile Base 6D

## Date

2026-06-29

## Current Branch

`feat/engine-dual-arm-foundation`

## Goal

- Add a 6D mobile-base support scaffold for the continuum arm stack.
- Extend base pose, mount frame, and world-frame kinematics.
- Add an optional MuJoCo mobile-base wrapper and visualization-only base box.
- Reserve extension points for manual UI control and future whole-body control.

## Coordinate Constraint

`T_world_tip = T_world_mobile_base * T_mobile_base_mount * T_mount_tip`

## Modified Files

- `src/continuum_sim/model/base_pose.py`
- `src/continuum_sim/model/mount_frame.py`
- `src/continuum_sim/kinematics/world_kinematics.py`
- `src/continuum_sim/control/mobile_base_controller.py`
- `src/continuum_sim/control/__init__.py`
- `src/continuum_sim/config.py`
- `src/continuum_sim/scenes/scene_builder.py`
- `src/continuum_sim/scenes/__init__.py`
- `src/continuum_sim/runtime/mujoco_navigation_runtime.py`
- `src/continuum_sim/runtime/mujoco_wiping_runtime.py`
- `src/continuum_sim/runtime/mujoco_viewer_runtime.py`
- `configs/robots/mobile_base_pose.yaml`
- `docs/mobile_base_6d_control.md`
- `docs/logs/2026-06-29-phase-2-mobile-base-6d.md`

## Change Notes

- Expanded `Pose6D` to support richer YAML loading, matrix export, vector/pose transforms, dict export, and RPY input conversion.
- Extended mobile-base mount loading to support the new `mobile_base + mounts` schema while keeping legacy `mount` compatibility.
- Added clearer world-frame composition helpers without breaking the old `compose_world_tip_pose(...)` entry point.
- Added a new lightweight mobile-base controller scaffold with command/state containers, clipping, pose integration, reset, and lock handling.
- Added optional `mobile_base_config_path` support to MuJoCo config loading.
- Added an opt-in MuJoCo wrapper body generator that can place the existing arm root under a `mobile_base` body and visualize a base box.
- Passed the optional mobile-base config into navigation/wiping scene generation.
- Reserved viewer state fields for base state, base lock, and base reset hooks.
- Added user-facing documentation for configuration, pose conventions, MuJoCo wrapper behavior, and manual-control recommendations.

## Verification Note

Per user instruction, no tests, validation, lint, format, build, install, viewer, or simulation commands were run.

## Risks And Follow-Up Verification

- The new mobile-base YAML shape is broader than the earlier test fixture and may require follow-up test updates.
- The MuJoCo wrapper assumes the arm root is a top-level body named `base`, or else falls back to the first top-level body.
- Viewer hotkeys for active 6D base motion are not fully wired yet; only lock/reset state hooks are reserved.
- The base pose clamp currently limits translation only; Euler-angle limit enforcement is left for a later controller pass.

## Suggested Manual Verification Commands

```bash
conda activate continuum_sim
python -m pytest tests/test_base_pose.py tests/test_world_kinematics.py --basetemp .tmp_pytest_pose -p no:cacheprovider
python -m pytest tests/test_multi_arm_model.py tests/test_multi_arm_state.py --basetemp .tmp_pytest_multi_arm -p no:cacheprovider
python -m pytest -m core --basetemp .tmp_pytest_core -p no:cacheprovider
python scripts/preview_engine_scene_mujoco.py --config configs/scenes/engine_cleaning.yaml --viewer --show-bbox --show-regions --show-axes
```
