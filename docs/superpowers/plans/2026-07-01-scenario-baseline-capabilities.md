# Scenario Baseline Capabilities Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Migrate reproducible tracking, navigation, wiping, diagnostics, plots, and GIF artifacts to the Scenario Application.

**Architecture:** Scenario remains the only composition root. Controllers emit named system commands and diagnostics; hooks record backend-specific replay state; a post-run artifact writer serializes backend-neutral histories and optional MuJoCo replay state.

**Tech Stack:** Python 3.11, NumPy, PyYAML, Matplotlib, imageio, optional MuJoCo.

## Global Constraints

- Preserve direct tendon-length-rate commands and world-frame base twist.
- Do not restore the old CLI or motor-space control path.
- Single- and dual-arm scenarios use identical runtime and artifact code.
- Do not automatically run tests, lint, format, build, install, or simulations.

---

### Task 1: Scenario artifact configuration and histories

**Files:** `src/continuum_sim/application/scenario.py`, `src/continuum_sim/runtime/hooks.py`

- [ ] Parse artifact enablement, output path, plots, GIF, FPS, and stride.
- [ ] Record named state, target, command, tendon, and MuJoCo replay histories.

### Task 2: Native artifact writer

**Files:** `src/continuum_sim/io/scenario_artifacts.py`, `src/continuum_sim/application/application.py`, `scripts/run_scenario.py`

- [ ] Flatten `RobotSystemState` and `RobotSystemCommand` into compressed NPZ arrays.
- [ ] Save metadata, scenario/config snapshots, composed MJCF, plots, and optional GIF.
- [ ] Print the run directory and tracking summary.

### Task 3: Task policies and diagnostics

**Files:** `src/continuum_sim/control/scenario_controllers.py`

- [ ] Preserve tracking targets and errors in command metadata.
- [ ] Add navigation clearance diagnostics and configurable violation stop.
- [ ] Add wiping approach/contact/retract phases and normal-contact proxy diagnostics.

### Task 4: Viewer and tendon diagnostics

**Files:** `src/continuum_sim/runtime/hooks.py`

- [ ] Draw target/tip trajectory overlays in the analytic viewer.
- [ ] Preserve tendon displacement, rate, saturation, rank, and condition snapshots.

### Task 5: Scenarios and Chinese README

**Files:** `configs/scenarios/*.yaml`, `README.md`

- [ ] Enable artifacts for maintained task scenarios.
- [ ] Put run commands first and document outputs, architecture, and limitations in Chinese.
