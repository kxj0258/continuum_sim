# Feedforward Gain Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a configurable, dimensionless feedforward gain to every shared position-mode task-space control path while preserving direct velocity commands.

**Architecture:** Load `feedforward_gain` through the existing scenario/profile merge and transfer it to `CoordinatedTrackingConfig`. Apply it once in `UnifiedLowLevelController`, where position and velocity intent semantics are still distinguishable, then export raw, scaled, and final velocities for diagnosis.

**Tech Stack:** Python dataclasses, NumPy, YAML configuration, pytest-style unit tests, Markdown documentation.

## Global Constraints

- `feedforward_gain` defaults to exactly `1.0` and must be finite and non-negative.
- Scale only `CartesianTaskIntent(control_mode="position")` feedforward velocity.
- Do not scale `control_mode="velocity"` direct commands.
- Do not run tests, validation, lint, format, build, installation, simulation, viewers, demos, or project entry points.
- Do not commit changes unless the user separately authorizes a commit.

---

### Task 1: Configuration contract

**Files:**
- Modify: `src/continuum_sim/application/scenario.py`
- Modify: `src/continuum_sim/control/coordinated_tracking.py`
- Test: `tests/test_feedforward_gain.py`

**Interfaces:**
- Consumes: `low_level_control.feedforward_gain` or `task.tracking_control.feedforward_gain`.
- Produces: `ScenarioTrackingControlConfig.feedforward_gain: float` and `CoordinatedTrackingConfig.feedforward_gain: float`.

- [x] **Step 1: Write configuration tests before production changes**

Add tests asserting that the MuJoCo shared profile resolves to `1.0`, a task-local override wins, and negative/non-finite values raise `ValueError`.

- [x] **Step 2: Do not run the failing tests automatically**

Suggested manual command only:

```powershell
pytest tests/test_feedforward_gain.py -v
```

Before implementation, the expected result is failure because the fields do not yet exist.

- [x] **Step 3: Add the configuration fields and validation**

Add `feedforward_gain: float = 1.0` to both dataclasses. Validate with:

```python
if not np.isfinite(self.feedforward_gain) or self.feedforward_gain < 0.0:
    raise ValueError("...feedforward_gain must be non-negative and finite.")
```

Load the public field in `_load_tracking_control_config()` using
`float(values.get("feedforward_gain", 1.0))`.

- [x] **Step 4: Transfer the resolved value into the coordinated config**

In `_tracking_coordinated_config()`, construct:

```python
CoordinatedTrackingConfig(
    executor_position_gain=tracking.executor_position_gain,
    feedforward_gain=tracking.feedforward_gain,
    ...
)
```

### Task 2: Mode-aware scaling and diagnostics

**Files:**
- Modify: `src/continuum_sim/control/unified_low_level.py`
- Test: `tests/test_feedforward_gain.py`

**Interfaces:**
- Consumes: `CoordinatedTrackingConfig.feedforward_gain` and `CartesianTaskIntent.control_mode`.
- Produces: a scaled `CoordinatedTrackingTarget.executor_velocity_world` plus command metadata `executor_feedforward_gain` and `executor_scaled_feedforward_velocity_world`.

- [x] **Step 1: Write behavior tests before production changes**

Use a real single-arm analytic state with zero Cartesian feedback error. Assert
that position mode with gain `0.25` changes `[0.004, 0, 0]` into final target
velocity `[0.001, 0, 0]`, while velocity mode leaves `[0.004, 0, 0]`
unchanged. Assert raw and scaled metadata separately.

- [x] **Step 2: Do not run the failing tests automatically**

Suggested manual command only:

```powershell
pytest tests/test_feedforward_gain.py -v
```

Before implementation, the expected result is failure because no scaling or
new metadata exists.

- [x] **Step 3: Implement scaling at the semantic boundary**

In `UnifiedLowLevelController.compute_command()` calculate:

```python
scaled_feedforward_velocity = (
    executor.feedforward_velocity_world * self.config.feedforward_gain
    if executor.control_mode == "position"
    else executor.feedforward_velocity_world.copy()
)
```

Pass that vector to `CoordinatedTrackingTarget` and add the configured gain and
scaled vector to command metadata. Keep `task_intent_velocity_world` raw.

### Task 3: Artifact export and configuration profiles

**Files:**
- Modify: `src/continuum_sim/io/scenario_artifacts.py`
- Modify: `configs/control/mujoco_tracking_low_level.yaml`
- Modify: `configs/control/spatial_low_level.yaml`
- Test: `tests/test_feedforward_gain.py`

**Interfaces:**
- Consumes: command metadata from Task 2.
- Produces: NPZ arrays `executor_feedforward_gain`, `task_intent_velocity_world`, and `executor_scaled_feedforward_velocity_world`.

- [x] **Step 1: Add an artifact-key expectation to the test coverage**

Assert the exact metadata names used by the exporter so a future rename cannot
silently remove the diagnostic chain.

- [x] **Step 2: Export scalar and vector diagnostics**

Add `executor_feedforward_gain` to the scalar metadata loop and add the raw and
scaled velocity names to the three-vector loop in `_result_arrays()`.

- [x] **Step 3: Expose the backward-compatible value in both profiles**

Add directly below `arm_position_gain`:

```yaml
feedforward_gain: 1.0
```

This reaches MuJoCo tasks using either shared profile without changing current
numerical behavior.

### Task 4: Parameter documentation

**Files:**
- Modify: `docs/configuration_reference.md`
- Modify: `docs/superpowers/specs/2026-07-15-feedforward-gain-design.md`

**Interfaces:**
- Consumes: the implemented configuration and metadata names.
- Produces: tuning guidance and an explicit explanation of the position/velocity boundary.

- [x] **Step 1: Document the control law and tuning meaning**

Document `v_target = Kp * position_error + feedforward_gain * v_feedforward`,
the meaning of `0`, `(0, 1)`, `1`, and values above `1`, profile/scenario
override locations, and the fact that target-speed limiting occurs afterward.

- [x] **Step 2: Document velocity-mode exclusion and diagnostics**

State that engine-cleaning and safety velocity overrides are not scaled. List
the raw, scaled, configured-gain, and final target-velocity NPZ fields.

- [x] **Step 3: Do not run verification automatically**

Suggested manual checks after implementation:

```powershell
pytest tests/test_feedforward_gain.py -v
python scripts/run_scenario.py configs/scenarios/single_mujoco_tracking.yaml
python scripts/run_scenario.py configs/scenarios/single_mujoco_navigation.yaml
python scripts/run_scenario.py configs/scenarios/single_engine_tracking.yaml
```

The user should compare `feedforward_gain: 1.0` against `0.75` one scenario at
a time and inspect raw/scaled/final Cartesian velocity arrays in `result.npz`.
