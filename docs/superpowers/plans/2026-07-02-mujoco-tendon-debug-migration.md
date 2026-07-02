# MuJoCo Tendon Debug Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restore standalone MuJoCo tendon debugging and rich per-tendon monitoring in scenario-driven runs.

**Architecture:** Add target and force diagnostics to the backend-independent arm state, then build one named-arm monitor data model used by the scenario hook and standalone debugger. The standalone debugger loads a scenario and drives `MujocoSystemBackend` through rate commands instead of raw actuator controls.

**Tech Stack:** Python, NumPy, Matplotlib widgets, MuJoCo passive viewer, pytest tests.

## Global Constraints

- Do not run tests, verification, lint, format, build, install, viewer, demo, or simulation commands.
- Keep live panels controlled by scenario YAML.
- Enable the tendon panel by default in relevant MuJoCo scenario files.
- Document commands in `README.md`.

---

### Task 1: System diagnostic state

**Files:**
- Modify: `src/continuum_sim/system/types.py`
- Modify: `src/continuum_sim/backends/mujoco_system_backend.py`
- Modify: `src/continuum_sim/backends/analytic_system_backend.py`
- Test: `tests/test_multi_arm_state.py`

**Interfaces:**
- Produces: `ArmSystemState.tendon_target_m: np.ndarray`
- Produces: `ArmSystemState.actuator_force_n: np.ndarray`

- [ ] **Step 1: Write state validation tests**

Add tests constructing `ArmSystemState` with matching target, displacement,
velocity, and force arrays, plus a mismatched force-array rejection case.

- [ ] **Step 2: Manual RED command**

Suggested only:

```powershell
pytest tests/test_multi_arm_state.py
```

- [ ] **Step 3: Add state fields and backend propagation**

Validate all four tendon vectors as matching one-dimensional arrays.
Populate MuJoCo targets from each `TendonRateIntegrator.displacement_m`,
forces from `physics.get_actuator_force()`, and analytic targets from its
integrators with zero force.

- [ ] **Step 4: Manual GREEN command**

Suggested only:

```powershell
pytest tests/test_multi_arm_state.py tests/test_multi_arm_model.py
```

### Task 2: Named system tendon monitor

**Files:**
- Create: `src/continuum_sim/visualization/system_tendon_debug.py`
- Modify: `src/continuum_sim/runtime/hooks.py`
- Modify: `src/continuum_sim/application/application.py`
- Test: `tests/test_system_tendon_debug.py`

**Interfaces:**
- Produces: `SystemTendonViewData`
- Produces: `system_tendon_view_data(state: RobotSystemState) -> SystemTendonViewData`
- Produces: `SystemTendonMonitorPanel.update(state: RobotSystemState)`

- [ ] **Step 1: Write flattening and hook tests**

Test stable executor/observer labels, target/current/error/force flattening,
and panel hook update stride without creating a GUI.

- [ ] **Step 2: Manual RED command**

Suggested only:

```powershell
pytest tests/test_system_tendon_debug.py
```

- [ ] **Step 3: Implement monitor data and panel**

Render target/current displacement bars, actuator-force bars, error text,
per-arm grouping, time, and saturation information. Replace the simplified
history-only `LiveTendonPanelHook` internals with this panel.

- [ ] **Step 4: Manual GREEN command**

Suggested only:

```powershell
pytest tests/test_system_tendon_debug.py tests/test_scenario_migrated_task_features.py
```

### Task 3: Scenario-based standalone debugger

**Files:**
- Create: `src/continuum_sim/visualization/mujoco_system_debug_viewer.py`
- Create: `scripts/debug_mujoco.py`
- Test: `tests/test_mujoco_system_debug_viewer.py`

**Interfaces:**
- Produces: `target_rates(target_m, current_target_m, max_rate_mps, dt) -> np.ndarray`
- Produces: `MujocoSystemDebugViewer`
- Consumes: `SimulationApplication.from_yaml(path).loop.backend`

- [ ] **Step 1: Write target-rate and scenario validation tests**

Cover target-rate clipping, zero error, named preset construction, and
rejection of analytic scenarios.

- [ ] **Step 2: Manual RED command**

Suggested only:

```powershell
pytest tests/test_mujoco_system_debug_viewer.py
```

- [ ] **Step 3: Implement viewer and CLI**

Create named tendon sliders, reset/zero/step/run controls, preset commands,
MuJoCo passive-viewer synchronization, and clean shutdown. Drive the backend
with `RobotSystemCommand` values computed from slider target error.

- [ ] **Step 4: Manual GREEN command**

Suggested only:

```powershell
pytest tests/test_mujoco_system_debug_viewer.py
python scripts/debug_mujoco.py configs/scenarios/dual_mujoco_tracking.yaml
```

### Task 4: Configuration and documentation

**Files:**
- Modify: `configs/scenarios/single_mujoco_tracking.yaml`
- Modify: `configs/scenarios/dual_mujoco_tracking.yaml`
- Modify: `configs/scenarios/single_mujoco_navigation.yaml`
- Modify: `configs/scenarios/dual_mujoco_navigation.yaml`
- Modify: `configs/scenarios/single_mujoco_wiping.yaml`
- Modify: `configs/scenarios/dual_mujoco_wiping.yaml`
- Modify: `configs/scenarios/single_engine_cleaning.yaml`
- Modify: `README.md`

**Interfaces:**
- Documents: scenario hook configuration and standalone debugger command.

- [ ] **Step 1: Enable relevant panels**

Set `show_live_tendon_panel: true` and retain an explicit stride in interactive
MuJoCo scenarios.

- [ ] **Step 2: Document usage**

Explain target/current/force semantics, single/dual labels, disabling the
panel, and standalone commands for single and dual scenarios.

- [ ] **Step 3: Manual final verification commands**

Suggested only:

```powershell
pytest tests/test_multi_arm_state.py tests/test_system_tendon_debug.py tests/test_mujoco_system_debug_viewer.py
python scripts/run_scenario.py configs/scenarios/dual_mujoco_tracking.yaml
python scripts/debug_mujoco.py configs/scenarios/dual_mujoco_tracking.yaml
```
