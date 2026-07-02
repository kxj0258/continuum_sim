# MuJoCo Debug Millimetre Input Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add synchronized millimetre sliders and numeric inputs for every tendon target in the standalone MuJoCo debugger.

**Architecture:** Keep backend targets in metres and convert only at the Matplotlib widget boundary. Pair each existing slider with a `TextBox`, route both through one clipped target update method, and synchronize all widgets after reset, zero, presets, slider movement, or text submission.

**Tech Stack:** Python, NumPy, Matplotlib `Slider` and `TextBox`, pytest.

## Global Constraints

- Do not run tests, verification, lint, format, build, install, viewer, demo, or simulation commands.
- UI target values and limits use millimetres.
- Backend and command values remain metres.
- Editing a target does not advance simulation.

---

### Task 1: Millimetre target normalization

**Files:**
- Modify: `tests/test_mujoco_system_debug_viewer.py`
- Modify: `src/continuum_sim/visualization/mujoco_system_debug_viewer.py`

**Interfaces:**
- Produces: `normalize_target_mm(value, minimum_mm, maximum_mm, fallback_mm) -> float`

- [ ] **Step 1: Add normalization tests**

```python
def test_normalize_target_mm_clips_and_rejects_invalid_input():
    assert normalize_target_mm("12.5", -20.0, 20.0, 0.0) == 12.5
    assert normalize_target_mm("25", -20.0, 20.0, 0.0) == 20.0
    assert normalize_target_mm("bad", -20.0, 20.0, 3.0) == 3.0
```

- [ ] **Step 2: Suggested RED command**

```powershell
pytest tests/test_mujoco_system_debug_viewer.py
```

- [ ] **Step 3: Implement normalization**

Parse finite floats, restore the fallback for invalid values, and clip valid
millimetre input to the configured millimetre limits.

### Task 2: Synchronized Slider and TextBox widgets

**Files:**
- Modify: `src/continuum_sim/visualization/mujoco_system_debug_viewer.py`
- Modify: `README.md`

**Interfaces:**
- Produces: `MujocoSystemDebugViewer.target_inputs`
- Consumes: `normalize_target_mm(...)`

- [ ] **Step 1: Build controls in millimetres**

Create one `TextBox` beside each slider. Multiply configured metre limits by
`1000`, and divide submitted millimetre targets by `1000` before storing them.

- [ ] **Step 2: Synchronize every update path**

Slider changes update the text input; text submission updates the slider;
`set_targets()` updates both without recursive callbacks.

- [ ] **Step 3: Document units and input behavior**

Update the standalone debugger section in `README.md` to state that sliders and
text boxes use millimetres and Enter only changes the target.

- [ ] **Step 4: Suggested manual verification**

```powershell
pytest tests/test_mujoco_system_debug_viewer.py
python scripts/debug_mujoco.py configs/scenarios/dual_mujoco_tracking.yaml
```
