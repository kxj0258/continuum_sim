# Scenario Application Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement one scenario API that composes single/dual assemblies, analytic/MuJoCo backends, tasks, and hooks.

**Architecture:** `ScenarioFactory` is the only composition root. Runtime behavior is selected through task and hook protocols rather than separate runtime files or CLI handlers.

**Tech Stack:** Python 3.11, NumPy, PyYAML, Matplotlib, optional MuJoCo.

## Global Constraints

- Use world-frame base twist commands.
- Use direct tendon-length rates for spatial arms.
- Do not preserve old package-level Python APIs.
- Do not run tests, lint, format, build, install, viewers, or simulations automatically.

---

### Task 1: Scenario configuration and application API

**Files:**
- Create: `src/continuum_sim/application/scenario.py`
- Create: `src/continuum_sim/application/application.py`
- Create: `src/continuum_sim/application/__init__.py`

**Interfaces:**
- Produces: `load_scenario_config()`, `SimulationApplication.from_yaml()`,
  `SimulationApplication.run()`.

- [ ] Parse assembly, backend, scene, task, runtime, and hook sections.
- [ ] Build dependencies in one factory without package facade imports.
- [ ] Keep the script entry point as a one-argument adapter.

### Task 2: Analytic system backend and scenario controllers

**Files:**
- Create: `src/continuum_sim/backends/analytic_system_backend.py`
- Create: `src/continuum_sim/control/scenario_controllers.py`

**Interfaces:**
- Produces: `AnalyticSystemBackend`, `ZeroSystemController`, and
  `WaypointTrackingController`.

- [ ] Integrate direct tendon rates and world base twist.
- [ ] Compute named arm tip/centerline state using PCC FK.
- [ ] Advance tracking/navigation waypoints by executor tolerance.

### Task 3: Hooks

**Files:**
- Create: `src/continuum_sim/runtime/hooks.py`

**Interfaces:**
- Produces: recorder, tendon diagnostic, and optional MuJoCo viewer hooks.

- [ ] Record named state and command samples.
- [ ] Collect tendon saturation/rank diagnostic snapshots.
- [ ] Isolate optional viewer imports inside the viewer hook.

### Task 4: MJCF asset boundary

**Files:**
- Modify: `src/continuum_sim/scenes/engine_mjcf_adapter.py`
- Modify: `src/continuum_sim/system/composition.py`

**Interfaces:**
- Produces: rebased robot assets and prepared engine STL paths.

- [ ] Rebase every source MJCF file reference before writing output.
- [ ] Move binary STL face limiting from preview code into the scene adapter.
- [ ] Use prepared engine mesh paths for both preview and scenario composition.

### Task 5: Scenario configurations

**Files:**
- Create: `configs/scenarios/single_analytic_tracking.yaml`
- Create: `configs/scenarios/dual_analytic_tracking.yaml`
- Create: `configs/scenarios/single_mujoco_view.yaml`
- Create: `configs/scenarios/dual_mujoco_tracking.yaml`
- Create: `configs/scenarios/single_engine_tracking.yaml`
- Create: `configs/scenarios/dual_engine_tracking.yaml`

**Interfaces:**
- Produces: reproducible scenario examples for baseline and engine workflows.

- [ ] Add idle, tracking, and waypoint examples.
- [ ] Select single/dual assembly only through configuration.
- [ ] Keep calibration fields explicit.

### Task 6: Entry point and documentation

**Files:**
- Create: `scripts/run_scenario.py`
- Modify: `README.md`
- Modify: `docs/architecture_overview.md`

**Interfaces:**
- Produces: Python API and single convenience script documentation.

- [ ] Document scenario usage and capability mapping.
- [ ] Mark old CLI/runtime paths as superseded.
- [ ] Record manual verification commands without executing them.

