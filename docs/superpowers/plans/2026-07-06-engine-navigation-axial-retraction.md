# Engine Navigation Axial Retraction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move the phase-three executor path center back from the fully extended insertion endpoint by a configurable distance along the negative insertion axis.

**Architecture:** Parse `local_path.axial_retraction_m` into `EngineNavigationSpec`. During plan resolution, compute one retracted local-path center and use it for both executor waypoints and observer ROI.

**Tech Stack:** Python dataclasses, NumPy, YAML, pytest.

## Global Constraints

- `axial_retraction_m` is finite and non-negative.
- Positive values move toward the arm base: `center = endpoint - retraction * insertion_direction`.
- A zero value preserves the old endpoint-centered behavior.
- Do not run tests, builds, linters, formatters, installers, or simulations.

---

### Task 1: Parse and Validate the Parameter

**Files:**
- Modify: `src/continuum_sim/tasks/engine_navigation.py`
- Modify: `configs/scenarios/dual_engine_navigation.yaml`

- [x] Add `local_path_axial_retraction_m: float = 0.01` to `EngineNavigationSpec`.
- [x] Parse `local_path.axial_retraction_m`.
- [x] Reject non-finite or negative values while allowing zero.
- [x] Set `axial_retraction_m: 0.010` explicitly in the scenario YAML.

### Task 2: Apply Retraction to the Resolved Plan

**Files:**
- Modify: `src/continuum_sim/tasks/engine_navigation.py`

- [x] Compute:

```python
local_path_center = (
    insertion_waypoints[-1]
    - spec.local_path_axial_retraction_m * insertion_direction
)
```

- [x] Generate the transverse square around `local_path_center`.
- [x] Set `observer_roi_world` to the same retracted center.

### Task 3: Update Contract Tests and Documentation

**Files:**
- Modify: `tests/test_engine_navigation.py`
- Modify: `README.md`

- [x] Assert the parsed value is `0.01`.
- [x] Assert the mean square position and observer ROI equal the retracted center.
- [x] Document the parameter formula, units, and tuning direction.

## Suggested Manual Checks (Not Run by Codex)

```powershell
pytest tests/test_engine_navigation.py
python scripts/run_scenario.py configs/scenarios/dual_engine_navigation.yaml
```
