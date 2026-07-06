# Engine Navigation Local Tracking and Observer Safety Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Configure local waypoint advancement from YAML and prioritize observer-only inter-arm collision avoidance without changing executor tracking.

**Architecture:** Parse engine-specific local tracking and observer safety specs. Pass them through the staged tracker into coordinated tracking, where a hysteretic centerline-distance state machine selects observation or observer-only avoidance while per-arm singularity protection isolates executor commands.

**Tech Stack:** Python dataclasses, NumPy, YAML, weighted whole-body control, pytest.

## Global Constraints

- Intermediate rejoin always uses tolerance mode.
- Ordinary avoidance moves observer tendons only.
- Observer avoidance never freezes, pauses, zeros, or terminates executor tracking.
- Existing non-engine controllers retain current defaults.
- Do not run tests, builds, linters, formatters, installers, or simulations.

---

### Task 1: Parse YAML Policies

**Files:**
- Modify: `src/continuum_sim/tasks/engine_navigation.py`
- Modify: `configs/scenarios/dual_engine_navigation.yaml`
- Modify: `tests/test_engine_navigation.py`

- [x] Add validated local-tracking specification.
- [x] Add validated observer-control specification.
- [x] Configure tolerance mode and observer-priority safety defaults.

### Task 2: Wire Local Tracking Mode

**Files:**
- Modify: `src/continuum_sim/control/staged_engine_navigation.py`
- Modify: `src/continuum_sim/control/scenario_controllers.py`
- Modify: `tests/test_staged_engine_navigation.py`

- [x] Pass selected scheduler mode/time/steps to local path trackers.
- [x] Force tolerance scheduling for rejoin trackers.
- [x] Pass observer target and coordinated-control policies.

### Task 3: Implement Observer Safety State Machine

**Files:**
- Modify: `src/continuum_sim/control/coordinated_tracking.py`
- Modify: `src/continuum_sim/control/whole_body_controller.py`
- Modify: `tests/test_bending_space.py`

- [x] Compute nearest centerline distance and closest indices.
- [x] Activate observer-only avoidance inside influence distance.
- [x] Add release hysteresis.
- [x] Keep executor target velocity and scheduler independent of avoidance.
- [x] Use maximum observer retreat inside critical distance.
- [x] Publish safety diagnostics.

### Task 4: Stage Termination and Documentation

**Files:**
- Modify: `src/continuum_sim/control/staged_engine_navigation.py`
- Modify: `README.md`

- [x] Enable fixed-base per-arm singularity decoupling.
- [x] Add executor-command invariance regression coverage.
- [x] Pass engine scene query into local trackers.
- [x] Document mode selection, timing conversion, and observer safety tuning.

## Suggested Manual Checks (Not Run by Codex)

```powershell
pytest tests/test_engine_navigation.py tests/test_staged_engine_navigation.py tests/test_bending_space.py
python scripts/run_scenario.py configs/scenarios/dual_engine_navigation.yaml
```
