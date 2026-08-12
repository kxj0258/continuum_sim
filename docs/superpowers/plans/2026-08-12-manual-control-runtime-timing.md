# Manual-Control Runtime Timing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add low-overhead rolling terminal timing that identifies which manual-control stage reduces response frequency.

**Architecture:** One `RuntimeTimingReporter` receives stage durations from the manual-control UI, system backend, MuJoCo backend, passive viewer, and observer camera. It aggregates measurements for 0.5 seconds before printing average/maximum timings, while a `kx/ky` input marker prints its latency after the next completed control cycle.

**Tech Stack:** Python, `time.perf_counter`, context managers, pytest.

## Global Constraints

- Enable terminal timing only from `run_manual_control.py`.
- Print one rolling summary every 0.5 seconds, not once per control cycle.
- Print `kx/ky` input latency after the next completed control cycle.
- Do not add dependencies or run tests/simulations without explicit authorization.

---

### Task 1: Rolling timing reporter

**Files:**
- Create: `src/continuum_sim/utils/runtime_timing.py`
- Create: `tests/test_runtime_timing.py`

**Interfaces:**
- Produces: `RuntimeTimingReporter.measure(stage)`, `record(stage, duration_s)`, `mark_input(label)`, `start_cycle()`, and `finish_cycle()`.

- [ ] Write tests for rolling average/maximum values and next-cycle input latency.
- [ ] Implement aggregation with `perf_counter()` and a 0.5-second reporting window.
- [ ] Verify with `pytest tests/test_runtime_timing.py -q` after authorization.

### Task 2: Instrument control and physics stages

**Files:**
- Modify: `src/continuum_sim/visualization/mujoco_system_debug_viewer.py`
- Modify: `src/continuum_sim/backends/mujoco_system_backend.py`
- Modify: `src/continuum_sim/backends/mujoco_backend.py`
- Modify: `tests/test_mujoco_system_debug_viewer.py`

**Interfaces:**
- Consumes: one optional `RuntimeTimingReporter` shared by all layers.
- Produces: timings for input callback, command preparation, inner loop, MuJoCo substeps, final forward, state construction, and cycle total.

- [ ] Test that a `kx/ky` callback marks an input event.
- [ ] Add optional timing injection without changing non-manual behavior.
- [ ] Wrap the named stages with the reporter's `measure()` context manager.
- [ ] Verify relevant tests after authorization.

### Task 3: Instrument rendering and wire the manual entry point

**Files:**
- Modify: `src/continuum_sim/visualization/manual_control_app.py`
- Modify: `src/continuum_sim/runtime/observer_camera_hooks.py`

**Interfaces:**
- Consumes: the shared reporter created by `run_manual_control()`.
- Produces: panel, passive-viewer, camera-forward, camera-render, and camera-present timings.

- [ ] Create one reporter at the manual-control composition root.
- [ ] Pass it to the UI, backend, window owner, and camera hook.
- [ ] Instrument rendering without adding per-frame output.
- [ ] Manually verify with `python scripts/run_manual_control.py` after authorization.
