# Engine Tracking Convergence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Repair the tracking exception and give single/dual engine tracking a staged mobile-base approach followed by the proven fixed-base MuJoCo tendon tracking controller.

**Architecture:** An opt-in scenario flag selects a focused staged controller. The controller moves the prescribed base with all tendon rates zero, then delegates the trajectory to `TimedTrajectoryTrackingController` constructed from a fixed-base assembly copy and overrides its base command to zero.

**Tech Stack:** Python dataclasses, NumPy, YAML scenario configuration, MuJoCo system backend.

## Global Constraints

- Do not automatically run tests, verification, lint, format, build, install, viewer, simulation, or long-running commands.
- Manual validation commands are documentation only and must not be executed by Codex.
- Preserve the shared `configs/control/mujoco_tracking_low_level.yaml` low-level baseline.

---

### Task 1: Repair waypoint scheduler metadata

**Files:**
- Modify: `src/continuum_sim/control/scenario_controllers.py`

**Interfaces:**
- Produces: `waypoint_scheduler_paused: bool`, true exactly when scheduler advancement is disabled for the current call.

- [x] Define `scheduler_paused = not (advance and self.advance_enabled)` before scheduler update.
- [x] Guard `scheduler.update(...)` with `if not scheduler_paused`.
- [x] Keep the existing metadata key and all scheduler semantics unchanged.

### Task 2: Add opt-in staged mobile tracking configuration

**Files:**
- Modify: `src/continuum_sim/application/scenario.py`

**Interfaces:**
- Produces: `ScenarioTrackingControlConfig.stage_mobile_base` and base gain/tolerance fields loaded from `task.tracking_control`.

- [x] Add the boolean opt-in and engine-navigation-compatible default values.
- [x] Validate all four base approach values as positive and finite.
- [x] Load all new fields without changing scenarios that omit the opt-in.

### Task 3: Implement reaction-isolated engine tracking

**Files:**
- Create: `src/continuum_sim/control/staged_engine_tracking.py`

**Interfaces:**
- Consumes: mobile `RobotAssemblyConfig`, trajectory waypoints, normal tracking controller parameters, and base approach gains/tolerances.
- Produces: `StagedEngineTrackingController.compute_command(state) -> RobotSystemCommand`, plus `done`, `terminal_reason`, and `last_diagnostics`.

- [x] Derive the base target on the first state by translating the current base by `first_waypoint - current_executor_tip` while preserving orientation.
- [x] During `base_approach`, use `MobileBasePoseController` and return zero arm tendon rates.
- [x] Construct the delegated time tracker with a fixed-base assembly copy.
- [x] During `tracking`, forward only arm commands and always return a zero base twist.
- [x] Merge stage diagnostics into normal tracking metadata and delegate trajectory completion.

### Task 4: Compose and configure engine tracking

**Files:**
- Modify: `src/continuum_sim/application/application.py`
- Modify: `configs/scenarios/single_engine_tracking.yaml`
- Modify: `configs/scenarios/dual_engine_tracking.yaml`

**Interfaces:**
- Consumes: `tracking_control.stage_mobile_base`.
- Produces: explicit staged behavior only for scenarios that opt in.

- [x] Select `StagedEngineTrackingController` before the normal time tracker when the opt-in is true.
- [x] Reject a staged request on a fixed assembly or non-time tracking mode with a clear configuration error.
- [x] Switch both engine scenarios to their mobile assemblies.
- [x] Configure time tracking for 80 s, zero arm approach samples, and the engine-navigation base gains/tolerances.
- [x] Increase runtime to 5000 control steps and keep the shared MuJoCo tracking low-level profile.

### Task 5: Document flow, parameters, risks, and manual validation

**Files:**
- Modify: `README.md`
- Modify: `docs/configuration_reference.md`

**Interfaces:**
- Produces: operator documentation for the staged engine tracking mode.

- [x] Describe the base-approach and fixed-base tracking phases.
- [x] Explain why zero base twist and a fixed solver assembly isolate tendon reaction from base motion.
- [x] List the configuration fields and dual observer policy.
- [x] List, but do not execute, these manual commands:

```powershell
python scripts/run_scenario.py configs/scenarios/single_engine_tracking.yaml
python scripts/run_scenario.py configs/scenarios/dual_engine_tracking.yaml
```
