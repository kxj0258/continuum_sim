# Bending-Control Speed Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove unintended numerical throttling from single- and dual-arm bending-space tracking.

**Architecture:** Keep the compatible command boundary unchanged. Correct task construction so fixed roots cannot create avoidance objectives, evaluate damping on the controllable singular subspace, and regularize mapped tendon effort rather than curvature-rate magnitude.

**Tech Stack:** Python 3.11, NumPy, existing dataclass-based controllers, pytest test sources.

## Global Constraints

- Dual-arm minimum distance is exactly `0.010 m`.
- No new runtime dependency.
- Existing compatible rate/displacement limits remain active.
- Do not change square trajectory z placement.
- Do not run tests, builds, linters, formatters, viewers, simulations, or other verification commands.

---

### Task 1: Controllable-subspace singularity handling

**Files:**
- Modify: `src/continuum_sim/kinematics/whole_body.py`
- Modify: `tests/test_bending_space.py`

**Interfaces:**
- Consumes: `SingularityConfig.rank_tolerance`.
- Produces: `analyze_singularity(matrix, config)` whose damping and velocity scale use the smallest singular value greater than `rank_tolerance`.

- [ ] Add test source asserting a rank-deficient matrix with singular values `[1, 0]` retains nominal damping and full velocity scale.
- [ ] Add test source asserting an all-zero matrix retains maximum damping and minimum velocity scale.
- [ ] Filter singular values by `rank_tolerance` for control scaling while retaining original rank/full-rank diagnostics.
- [ ] Record, but do not execute, `pytest tests/test_bending_space.py -v`.

### Task 2: Tendon-effort regularization

**Files:**
- Modify: `src/continuum_sim/control/whole_body_controller.py`
- Modify: `tests/test_bending_space.py`

**Interfaces:**
- Consumes: `ControlLayout.bending_models[name].coupling_matrix`.
- Produces: `_regularization_matrix()` with base identity rows and per-arm `sqrt(weight) * C_b` rows.

- [ ] Add test source checking the arm regularization block equals the bending coupling matrix times the configured square-root weight.
- [ ] Replace the square diagonal curvature penalty with separately assembled base and tendon-effort row blocks.
- [ ] Keep `tendon_regularization_weight` as the public compatibility field.
- [ ] Record, but do not execute, the focused whole-body controller tests.

### Task 3: Actionable dual-arm collision avoidance and documentation

**Files:**
- Modify: `src/continuum_sim/control/coordinated_tracking.py`
- Modify: `README.md`
- Modify: `tests/test_bending_space.py`

**Interfaces:**
- Produces: `CoordinatedTrackingConfig.inter_arm_min_distance_m == 0.010`.
- Produces: collision tasks only for movable samples, positive violation speed, and nonzero relative Jacobian.

- [ ] Change the minimum distance default to `0.010`.
- [ ] Exclude centerline index zero for both arms when choosing the closest pair.
- [ ] Return no collision task when distance is not below the minimum.
- [ ] Return no collision task when the relative Jacobian norm is at or below the solver rank tolerance.
- [ ] Preserve observer tracking whenever no actionable collision task exists.
- [ ] Document the speed fix, physical meaning of the 10 mm setting, and diagnostics to inspect.
- [ ] Commit code, test sources, plan, and documentation with a focused message.
