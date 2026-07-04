# MuJoCo Tracking Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Improve scenario-native MuJoCo trajectory tracking metrics, transients, configurability, dual-arm isolation, and diagnostics.

**Architecture:** Add a typed tracking-control configuration at the scenario boundary, carry approach/feedforward metadata through the waypoint controller, apply fixed-base per-arm singularity protection inside the existing whole-body solver, and flatten controller/backend diagnostics into scenario artifacts. Existing controller and backend interfaces remain intact.

**Tech Stack:** Python, NumPy, dataclasses, YAML, pytest.

## Global Constraints

- Do not run tests, verification, lint, format, build, install, or simulation commands.
- Preserve existing user changes in generated XML files.
- Keep legacy MuJoCo tracking runtime and task YAML behavior unchanged.
- Commit only files belonging to this optimization.

---

### Task 1: Tracking configuration and approach path

**Files:**
- Modify: `src/continuum_sim/application/scenario.py`
- Modify: `src/continuum_sim/tasks/trajectory_generation.py`
- Modify: `src/continuum_sim/application/application.py`
- Test: `tests/test_scenario_migrated_task_features.py`

**Interfaces:**
- Produces: `ScenarioTrackingControlConfig`
- Produces: `prepend_tracking_approach(waypoints, assembly, samples)`

- [ ] Add tests for nested configuration defaults/overrides and quintic approach endpoints.
- [ ] Add the typed tracking-control config and validation.
- [ ] Add approach generation and pass the generated waypoint metadata to tracking composition.

### Task 2: Metric semantics and feedforward

**Files:**
- Modify: `src/continuum_sim/control/scenario_controllers.py`
- Modify: `src/continuum_sim/control/coordinated_tracking.py`
- Test: `tests/test_tracking_optimization.py`

**Interfaces:**
- Extends: `CoordinatedTrackingTarget.executor_velocity_world`
- Produces command metadata: `achieved_waypoint_error_m`, `waypoint_advanced`, `tracking_complete`, `tracking_approach`

- [ ] Add tests that distinguish pre-schedule achieved error from post-schedule command error.
- [ ] Add tests for final completion metadata and Cartesian feedforward clipping.
- [ ] Implement metadata-preserving completion and feedforward target velocity.

### Task 3: Fixed-base per-arm singularity protection

**Files:**
- Modify: `src/continuum_sim/control/whole_body_controller.py`
- Modify: `src/continuum_sim/control/coordinated_tracking.py`
- Test: `tests/test_tracking_optimization.py`

**Interfaces:**
- Extends: `WholeBodyControllerConfig.decouple_arm_singularity`
- Extends: `WholeBodySolveResult.arm_singularities`

- [ ] Add a block-diagonal test where poor observer conditioning does not reduce executor velocity.
- [ ] Compute per-arm reports for fixed-base systems and apply block damping/scaling.
- [ ] Preserve global singularity behavior for movable-base systems and disabled decoupling.

### Task 4: Recorder and artifacts

**Files:**
- Modify: `src/continuum_sim/runtime/hooks.py`
- Modify: `src/continuum_sim/io/scenario_artifacts.py`
- Modify: `scripts/run_scenario.py`
- Test: `tests/test_scenario_artifacts.py`

**Interfaces:**
- Produces NPZ arrays for tracking semantics, per-arm solver reports, saturation, tendon lag, and force.

- [ ] Add artifact tests for the new arrays and metric names.
- [ ] Record tracking semantics and state-backed actuator diagnostics.
- [ ] Flatten diagnostics and print achieved-waypoint summary metrics.

### Task 5: Scenario tuning and documentation

**Files:**
- Modify: `configs/scenarios/single_mujoco_tracking.yaml`
- Modify: `configs/scenarios/dual_mujoco_tracking.yaml`
- Modify: `README.md`

- [ ] Enable a 20-sample approach and conservative feedforward in both tracking scenarios.
- [ ] Enable per-arm singularity protection in dual tracking.
- [ ] Document parameter meanings, diagnostic arrays, risks, and manual verification commands.
- [ ] Review the staged diff without executing tests and commit with `feat(control): optimize mujoco trajectory tracking`.

