# Engine World Offset Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a separate calibrated engine world-offset field so engine-frame annotations shift into MuJoCo world space without rewriting engine-local coordinates.

**Architecture:** Keep `engine.pose.position_m` as the authored scene pose and add an independent offset vector that is summed into the effective world translation used by diagnostics and preview MJCF generation. Reuse the existing engine-frame transform path so regions, exploration start, exploration paths, and engine-frame primitive hints move consistently with the mesh.

**Tech Stack:** Python, YAML config loading, pytest, MuJoCo MJCF text generation

## Global Constraints

- Keep the change focused on engine scene config parsing and preview/diagnostic transforms only.
- Preserve existing config semantics when the new offset field is omitted.
- Do not modify unrelated runtime, control, or robot model code.

---

### Task 1: Add failing tests for engine world offset

**Files:**
- Modify: `tests/test_engine_scene.py`
- Modify: `tests/test_engine_asset_checks.py`

**Interfaces:**
- Consumes: `load_engine_scene_config(path) -> EngineSceneConfig`
- Consumes: `build_engine_preview_mjcf(config_path, ...) -> str`
- Consumes: `collect_engine_scene_diagnostics(config_path, ...) -> EngineSceneDiagnostics`
- Produces: regression coverage for config parsing and world-transform behavior

### Task 2: Implement the new offset field in engine scene parsing

**Files:**
- Modify: `src/continuum_sim/scenes/engine_scene.py`

**Interfaces:**
- Produces: `EnginePoseConfig.world_offset_m: np.ndarray | None`
- Produces: helper returning effective engine world translation

### Task 3: Use the effective world translation in diagnostics and preview generation

**Files:**
- Modify: `scripts/check_engine_assets.py`
- Modify: `scripts/preview_engine_scene_mujoco.py`

**Interfaces:**
- Consumes: effective engine world translation helper
- Produces: consistent world placement for mesh, bbox, regions, exploration start, exploration paths, and engine-frame primitive hints

### Task 4: Update the engine scene config and verify

**Files:**
- Modify: `configs/scenes/engine_cleaning.yaml`

**Interfaces:**
- Produces: calibrated engine world offset in config
- Test: `C:\Users\kxj\.conda\envs\continuum_sim\python.exe -m pytest tests/test_engine_scene.py tests/test_engine_asset_checks.py -q`
