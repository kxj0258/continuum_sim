# Unified Control Architecture Implementation Plan

> **For Codex:** Execute this plan directly in the current workspace. The user explicitly forbids automatic tests, builds, lint, format, install, simulation, and verification commands. Record manual validation commands only; do not run them and do not auto-commit.

**Goal:** Implement architecture B with strongly typed task intents, a shared low-level motion controller/profile, and the requested analytic/MuJoCo scenario matrix.

**Architecture:** Existing task adapters retain task scheduling and force/mission logic, but construct `TaskStep` objects. `UnifiedLowLevelController` converts those intents into the existing coordinated whole-body solver and returns the unchanged `RobotSystemCommand` required by the simulation loop and backends.

**Tech Stack:** Python dataclasses/enums, NumPy, YAML scenario configuration, existing continuum_sim control/runtime APIs.

---

### Task 1: Add the typed task-control contract

**Files:**
- Create: `src/continuum_sim/control/task_intent.py`
- Create: `src/continuum_sim/control/unified_low_level.py`

Define validated position/velocity Cartesian intent, observer intent, task status and task step. Wrap `CoordinatedTrackingController` so all task adapters use one lower-level entry point.

### Task 2: Migrate task adapters to the shared lower controller

**Files:**
- Modify: `src/continuum_sim/control/scenario_controllers.py`
- Modify: `src/continuum_sim/control/staged_engine_navigation.py`

Make waypoint and timed trackers produce task steps. Honor waypoint `advance` and `advance_enabled`, persist timed active index, and represent engine-cleaning output as a velocity intent to eliminate the double position feedback term.

### Task 3: Add shared lower-level configuration

**Files:**
- Create: `configs/control/spatial_low_level.yaml`
- Modify: `src/continuum_sim/application/scenario.py`
- Modify: `src/continuum_sim/application/application.py`
- Modify: relevant files under `configs/scenarios/`

Load `scenario.low_level_control_path`, merge its `low_level_control` values beneath any legacy scenario override, and pass the resolved configuration into all standard task controllers and staged engine navigation.

### Task 4: Complete requested scenario coverage

**Files:**
- Create: `configs/robots/assemblies/single_spatial_mobile.yaml`
- Create: `configs/scenarios/dual_analytic_navigation.yaml`
- Create: `configs/scenarios/dual_analytic_wiping.yaml`
- Create: `configs/scenarios/dual_engine_cleaning.yaml`
- Create: `configs/scenarios/single_engine_navigation.yaml`
- Modify: `configs/scenarios/single_mujoco_wiping.yaml`
- Delete: `configs/scenarios/single_mujoco_wiping_admittance.yaml`

Base each new scenario on its existing nearest equivalent, keep the CLI uniform, and consolidate admittance into the primary single MuJoCo wiping configuration as a selectable strategy.

### Task 5: Document execution and parameter ownership

**Files:**
- Modify: `README.md`
- Modify: `docs/architecture_overview.md`
- Modify: `docs/configuration_reference.md`
- Modify: `task_plan.md`
- Modify: `progress.md`

Document every requested command, each task flow, the shared lower-level profile, task-specific upper-level configuration, admittance switching, risks, and manual validation commands.

### Task 6: Manual handoff only

Do not execute verification. Report modified files, exact behavior changes, risks, and suggested manual commands. Do not claim tests pass and do not commit unless the user asks separately.

### Task 7: Isolate executor and observer control

**Files:**
- Modify: `src/continuum_sim/control/task_intent.py`
- Modify: `src/continuum_sim/control/unified_low_level.py`
- Modify: `src/continuum_sim/control/coordinated_tracking.py`
- Modify: `src/continuum_sim/control/whole_body_controller.py`

Represent observer collision avoidance as an explicit upper-level intent. Solve executor and observer tasks independently so observer rows and columns never enter the executor solve. Keep executor freeze and observer-triggered global hard stop disabled.

### Task 8: Unify tracking and actuation limits

**Files:**
- Modify: `configs/scenarios/dual_mujoco_tracking.yaml`
- Modify: `configs/control/spatial_low_level.yaml`
- Modify: `src/continuum_sim/application/scenario.py`
- Modify: `src/continuum_sim/application/application.py`

Make dual executor use the same timed trajectory parameters as single. Apply the shared Cartesian speed limit and tendon-rate/target-lead protections to both arms.

### Task 9: Record and plot synchronized observer diagnostics

**Files:**
- Modify: `src/continuum_sim/runtime/hooks.py`
- Modify: `src/continuum_sim/io/scenario_artifacts.py`
- Modify: `README.md`
- Modify: `docs/architecture_overview.md`
- Modify: `docs/configuration_reference.md`

Persist arm targets, requested/applied rates, observer safety state and inter-arm clearance. Add time-aligned target-vs-actual, command-vs-velocity, force/error and clearance plots.
