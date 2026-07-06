# Engine Navigation Local Path Plots Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate one scale-independent local trajectory/error image for each engine-navigation maneuver and correct target/actual alignment in the whole-run trajectory plot.

**Architecture:** Record actual tip and local-path context alongside each executor target. Project each named path into its transverse local frame during artifact generation and save a two-panel plot.

**Tech Stack:** Python dataclasses, NumPy, Matplotlib, pytest.

## Global Constraints

- Exclude rejoin samples from local shape plots.
- Keep existing artifacts and non-engine behavior compatible.
- Skip malformed optional local context without failing artifact generation.
- Do not run tests, builds, linters, formatters, installers, or simulations.

---

### Task 1: Target-Aligned Recording

**Files:**
- Modify: `src/continuum_sim/runtime/hooks.py`
- Modify: `src/continuum_sim/io/scenario_artifacts.py`

- [x] Record actual executor tip with every executor target.
- [x] Record local-path name, type, subphase, center, and insertion direction.
- [x] Export the aligned arrays to `data.npz`.
- [x] Use aligned actual positions in `trajectory.png`.

### Task 2: Per-Path Local Plots

**Files:**
- Modify: `src/continuum_sim/io/scenario_artifacts.py`

- [x] Build a stable transverse basis from insertion direction.
- [x] Select `path` samples by local-path name.
- [x] Project target and actual trajectories relative to the local center.
- [x] Plot local XY trajectory and error in millimetres.
- [x] Save one sanitized file per local path.

### Task 3: Contract Tests and Documentation

**Files:**
- Modify: `tests/test_scenario_artifacts.py`
- Modify: `README.md`

- [x] Populate representative circle and square aligned samples.
- [x] Assert local plot files and NPZ arrays are produced.
- [x] Document every existing plot and the new local plot layout.

## Suggested Manual Checks (Not Run by Codex)

```powershell
pytest tests/test_scenario_artifacts.py
python scripts/run_scenario.py configs/scenarios/dual_engine_navigation.yaml
```
