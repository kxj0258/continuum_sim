# Engine Navigation Intermediate Local Paths Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add retracted circle and figure-eight executor events at one-third and two-thirds of the insertion route while retaining the endpoint square.

**Architecture:** Resolve YAML local-path specifications into ordered path events tied to insertion waypoint indices. The staged controller pauses base motion for each event, executes its path, rejoins the straight insertion target for intermediate events, and resumes insertion.

**Tech Stack:** Python dataclasses, NumPy, YAML, MuJoCo runtime overlays, pytest.

## Global Constraints

- Preserve endpoint-only behavior when no intermediate paths are configured.
- Lock the base during local-path and rejoin commands.
- Rejoin intermediate events before resuming base motion.
- Do not run tests, builds, linters, formatters, installers, or simulations.

---

### Task 1: Configuration and Geometry

**Files:**
- Modify: `src/continuum_sim/tasks/engine_navigation.py`
- Modify: `configs/scenarios/dual_engine_navigation.yaml`
- Modify: `tests/test_engine_navigation.py`

- [x] Parse and validate ordered `intermediate_local_paths`.
- [x] Add circle and figure-eight transverse generators.
- [x] Resolve one-third, two-thirds, and endpoint path events.
- [x] Keep endpoint compatibility fields.

### Task 2: Staged Controller

**Files:**
- Modify: `src/continuum_sim/control/staged_engine_navigation.py`
- Modify: `tests/test_staged_engine_navigation.py`

- [x] Replace the single persistent tracker with an event-specific tracker.
- [x] Pause at each event insertion index.
- [x] Execute local path with base locked.
- [x] Rejoin the insertion-axis target after intermediate paths.
- [x] Resume insertion or complete after the endpoint path.
- [x] Publish event and subphase metadata.

### Task 3: Visualization

**Files:**
- Modify: `src/continuum_sim/runtime/hooks.py`
- Modify: `tests/test_engine_navigation_overlay.py`

- [x] Publish all local path arrays in metadata.
- [x] Copy collections safely in overlay state.
- [x] Render each local path separately.

### Task 4: Documentation

**Files:**
- Modify: `README.md`

- [x] Document the three-event sequence and every path parameter.
- [x] Document that intermediate paths rejoin before base motion resumes.

## Suggested Manual Checks (Not Run by Codex)

```powershell
pytest tests/test_engine_navigation.py tests/test_staged_engine_navigation.py tests/test_engine_navigation_overlay.py
python scripts/run_scenario.py configs/scenarios/dual_engine_navigation.yaml
```
