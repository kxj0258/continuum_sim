# Engine Navigation Visualization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add configurable planned-route, active-target, and history overlays to the staged dual-arm engine-navigation viewer and live MuJoCo recording.

**Architecture:** The staged controller publishes resolved plan geometry and the active target through command metadata. A focused runtime overlay-state object captures bounded dynamic histories, while the existing shared MuJoCo scene renderer draws static plans and dynamic histories for both the passive viewer and live video.

**Tech Stack:** Python dataclasses, NumPy, MuJoCo passive viewer/render scene API, YAML configuration, pytest.

## Global Constraints

- Do not add dependencies.
- Do not write visualization geometry into generated MJCF; use runtime overlays.
- Preserve existing overlay behavior for non-engine tasks.
- Do not run tests, builds, linters, formatters, installers, or simulations during implementation.
- Any commands listed under manual validation are suggestions for the user only.

---

### Task 1: Add Typed Engine-Navigation Overlay Configuration

**Files:**
- Modify: `src/continuum_sim/config.py`
- Modify: `configs/mujoco_dual.yaml`
- Modify: `tests/test_robot_config.py`

**Interfaces:**
- Produces: `MujocoEngineNavigationOverlayConfig`
- Produces: `MujocoViewerOverlayConfig.engine_navigation`
- Consumes: existing `_bool`, `_positive_float_value`, `_positive_int_value`, and `_rgba_tuple` configuration helpers

- [x] **Step 1: Describe the YAML contract in a config-loading test**

Extend `test_mujoco_viewer_config_loads_overlay_settings` to assert the master
switches, route/waypoint strides, radii, and representative RGBA values from
`viewer.overlays.engine_navigation`.

- [x] **Step 2: Add the typed nested dataclass**

Add `MujocoEngineNavigationOverlayConfig` with switches for planned paths,
waypoints, observer ROI, current target, base history, executor history, and
target history; add the associated radii, RGBA values, and sampling strides.

- [x] **Step 3: Parse the nested configuration**

Add `_load_mujoco_engine_navigation_overlay_config(values)` and assign its
result to `MujocoViewerOverlayConfig.engine_navigation`. Missing sections use
safe defaults with `enabled=False`.

- [x] **Step 4: Add explicit dual-engine display settings**

Add `viewer.overlays.engine_navigation` to `configs/mujoco_dual.yaml`, enable
all requested elements, and select visually distinct colors:

```yaml
engine_navigation:
  enabled: true
  planned_paths: true
  insertion_waypoints: true
  observer_roi: true
  current_target: true
  base_history: true
  executor_history: true
  target_history: true
  path_stride: 1
  waypoint_stride: 1
```

Provide explicit radii and RGBA arrays beside these switches.

### Task 2: Publish the Navigation Visualization Contract

**Files:**
- Modify: `src/continuum_sim/control/staged_engine_navigation.py`
- Modify: `tests/test_staged_engine_navigation.py`

**Interfaces:**
- Consumes: `EngineNavigationPlan`
- Produces command metadata keys:
  `engine_navigation_pre_entry_target_m`,
  `engine_navigation_base_path_m`,
  `engine_navigation_insertion_path_m`,
  `engine_navigation_executor_path_m`,
  `engine_navigation_observer_roi_m`,
  `engine_navigation_active_target_m`,
  `engine_navigation_active_target_kind`

- [x] **Step 1: Add metadata-contract assertions**

Extend the base-motion and executor-motion controller tests to assert that
plan arrays are copied into metadata, base phases identify the active target
as `base`, and executor navigation identifies it as `executor`.

- [x] **Step 2: Publish immutable plan geometry**

Extend `_metadata()` so every command includes copies of the resolved pre-entry
target, mobile-base path, insertion tip path, executor local path, and observer
ROI.

- [x] **Step 3: Publish phase-aware active target**

Pass `active_target_kind="base"` from base phases and
`active_target_kind="executor"` from executor phases. During executor
navigation use `tracked.metadata["executor_target_world"]` as the active
target; terminal commands keep the last meaningful target.

### Task 3: Share Dynamic Overlay State Between Viewer and Video

**Files:**
- Modify: `src/continuum_sim/runtime/hooks.py`
- Create: `tests/test_engine_navigation_overlay.py`

**Interfaces:**
- Produces: private `_TrackingOverlayState`
- Produces: `_TrackingOverlayState.capture(state, command)`
- Consumes: `RobotSystemState`, `RobotSystemCommand`, and overlay configuration

- [x] **Step 1: Specify history capture behavior**

Create unit tests that construct minimal state/command objects and assert:

- engine navigation captures base positions;
- executor tip positions continue to be captured;
- active target history uses `engine_navigation_active_target_m`;
- non-engine commands fall back to `executor_target_world`;
- reset clears all histories and cached metadata.

- [x] **Step 2: Add focused overlay-state storage**

Create `_TrackingOverlayState` with:

```python
tip_trail: list[np.ndarray]
target_trail: list[np.ndarray]
base_trail: list[np.ndarray]
navigation_metadata: dict[str, object]
```

Its `capture()` method copies arrays from command/state and its `clear()` method
resets the state.

- [x] **Step 3: Replace duplicated viewer/video history fields**

Use `_TrackingOverlayState` in `MujocoViewerHook` and
`MujocoLiveVideoRecorderHook`. Both hooks call `capture()` before rendering and
pass the state object to the shared renderer.

### Task 4: Render Static Plans, Active Target, and Histories

**Files:**
- Modify: `src/continuum_sim/runtime/hooks.py`
- Modify: `tests/test_engine_navigation_overlay.py`

**Interfaces:**
- Consumes: `_TrackingOverlayState`
- Produces: private `_draw_engine_navigation_overlay_scene(...)`
- Reuses: `_add_overlay_sphere(...)` and `_add_overlay_trail(...)`

- [x] **Step 1: Specify geometry selection without requiring MuJoCo**

Test small metadata-extraction and point-sampling helpers directly: valid
`(N, 3)` arrays are accepted, malformed values are ignored, and configured
strides preserve the final point.

- [x] **Step 2: Draw static navigation geometry**

When `config.engine_navigation.enabled` is true, draw:

- pre-entry sphere;
- mobile-base planned path capsule trail;
- insertion path capsule trail;
- insertion waypoint spheres;
- executor local path capsule trail;
- observer ROI sphere.

Use the configured route and waypoint strides while always retaining path end
points.

- [x] **Step 3: Draw dynamic navigation geometry**

Draw the phase-aware current target, bounded base history, bounded executor
history, and bounded active-target history. Use the existing
`trail_max_points` and `trail_stride` limits for histories.

- [x] **Step 4: Preserve generic overlays**

Keep generic target/tip overlays for non-engine tasks. For engine navigation,
avoid duplicate generic target/tip geometry when the corresponding
engine-navigation current-target, executor-history, or target-history switch
is enabled.

- [x] **Step 5: Keep scene-capacity behavior safe**

Continue relying on `_add_overlay_sphere` and `_add_overlay_trail` capacity
checks, and order current/critical target geometry before decorative history
geometry so the most useful marker survives a crowded scene.

### Task 5: Document Tuning and Manual Validation

**Files:**
- Modify: `README.md`

**Interfaces:**
- Documents: `viewer.overlays.engine_navigation`
- Documents manual command:
  `python scripts/run_scenario.py configs/scenarios/dual_engine_navigation.yaml`

- [x] **Step 1: Document the visual legend**

Describe what each default color represents and how the visualization changes
between approach, insertion, and executor-navigation phases.

- [x] **Step 2: Document every tuning category**

Explain master/element switches, radii, RGBA values, route/waypoint strides,
and the shared history limit/stride.

- [x] **Step 3: Provide manual validation guidance**

Tell the user to run:

```powershell
python scripts/run_scenario.py configs/scenarios/dual_engine_navigation.yaml
```

Ask them to inspect both the passive viewer and the saved live MuJoCo GIF.
Explicitly note that no automated validation was run during implementation.

## Suggested Manual Checks (Not Run by Codex)

```powershell
pytest tests/test_robot_config.py tests/test_staged_engine_navigation.py tests/test_engine_navigation_overlay.py
python scripts/run_scenario.py configs/scenarios/dual_engine_navigation.yaml
```
