# Task Plan: PCC Dynamics-Based Hybrid Force-Position Control Upgrade

## Goal
Upgrade `continuum_sim` to support acceptance-oriented continuum robot control using DMP trajectory generation, CBF/SDF self-avoidance, and a new optional PCC reduced-dynamics hybrid force-position controller while keeping the existing kinematic `hybrid_force_position` controller as the stable baseline.

## Current Phase
Complete

## Phases

### Phase 1: Requirements and Architecture Baseline
- [x] Capture user decisions.
- [x] Identify current baseline modules to preserve.
- [x] Document requirements and constraints in `findings.md`.
- **Status:** complete

### Phase 2: DMP Trajectory Layer
- [x] Add `src/continuum_sim/tasks/dmp_trajectory.py`.
- [x] Implement typed `DiscreteDMP.imitate(time, trajectory)` and `rollout(start_pos, goal_pos, tau)`.
- [x] Extend `trajectory_tracking_config.py` so `trajectory.type: dmp` can be loaded from YAML while preserving existing trajectory types.
- [x] Add DMP unit tests for shape learning, start/goal adaptation, and numerical stability.
- **Status:** complete

### Phase 3: CBF-QP and SDF Self-Avoidance Layer
- [x] Add a CBF-QP-compatible velocity projection helper without removing the existing DLS controller.
- [x] Add SDF/clearance gradient utilities and null-space projection helpers in `src/continuum_sim/kinematics/`.
- [x] Integrate CBF/SDF into navigation through a new `controller.type` option.
- [x] Add tests for clearance constraints, null-space projection, zero gradients, and no-obstacle scenes.
- **Status:** complete

### Phase 4: PCC Reduced Dynamics Model
- [x] Create `src/continuum_sim/dynamics/`.
- [x] Implement a first-pass reduced model: `M(q) qddot + D qdot + K(q-q0) = tau_tendon + J_tip.T F_contact + tau_disturbance`.
- [x] Use YAML engineering estimates for mass, stiffness, damping, and integration parameters.
- [x] Reuse existing PCC FK, centerline sampling, and finite-difference Jacobians before considering C++/Eigen acceleration.
- [x] Add `tests/test_pcc_dynamics.py` for matrix shapes, positive definiteness, contact-force projection, and stable integration.
- **Status:** complete

### Phase 5: Optional Dynamics-Based Hybrid Force-Position Control
- [x] Keep `controller.type: hybrid_force_position` as the current default and regression baseline.
- [x] Add a new experimental `controller.type`, for example `dynamic_adaptive_impedance`.
- [x] Implement `src/continuum_sim/control/adaptive_impedance.py` using the PCC reduced dynamics model.
- [x] Add a minimal iLQR-ready data path where stiffness and damping diagonals are recorded.
- [x] Integrate the new controller into `mujoco_wiping_runtime.py` through a clean type-based branch.
- **Status:** complete

### Phase 6: Acceptance Automation and Reporting
- [x] Add positioning acceptance script for indicator 3.1 with final/steady-state error below 2 cm.
- [x] Add disturbance acceptance script for indicator 3.3 with configurable wrench magnitude, direction, and duration, defaulting to an engineering test profile until a third-party protocol exists.
- [x] Add force tracking acceptance script for indicator 3.3 with contact-force mean/RMSE error below 1 N.
- [x] Generate Markdown reports with plots, metrics tables, configuration snapshots, and reserved CNAS/CMA stamp area.
- [x] Run `python -m pytest -m core`, relevant MuJoCo tests, and all acceptance scripts.
- **Status:** complete

### Phase 7: Documentation and Rollout
- [x] Update `README.md` and `docs/configuration_reference.md` with the new controller modes and YAML fields.
- [x] Document when to use the kinematic controller versus the dynamics-based controller.
- [x] Add example configs for baseline and experimental dynamic impedance runs.
- [x] Record limitations, especially that third-party disturbance protocol is not yet fixed.
- **Status:** complete

### Phase 8: Current Workspace Completion
- [x] Reproduce the currently added tests in `tests/test_adaptive_impedance.py` and `tests/test_wiping_config.py`.
- [x] Trace root causes for any failing requirements.
- [x] Apply focused fixes without reverting existing workspace changes.
- [x] Re-run focused tests and an appropriate broader regression gate through `conda activate continuum_sim`.
- **Status:** complete

### Phase 9: Git History Reinitialization
- [x] Verify the existing `continuum_sim_history_backup_20260616_112931.bundle`.
- [x] Create a fresh local history backup bundle for the current `main`.
- [x] Preserve the current history as a local-only branch.
- [x] Reinitialize `main` as a clean single-root history from the current tree.
- [x] Validate the rewritten tree.
- [x] Force-push the rewritten `main` to `origin/main`.
- **Status:** complete

## Key Questions
1. What exact third-party disturbance profile should be used once it becomes available?
2. Which dynamic controller should become the default after validation, if any?
3. When performance bottlenecks appear, should acceleration be done with vectorized NumPy first or C++/Eigen immediately?

## Decisions Made
| Decision | Rationale |
|----------|-----------|
| Keep `hybrid_force_position` as the baseline controller. | It already works as a kinematic force-position controller and provides a stable regression target. |
| Add PCC reduced dynamics instead of full FEM/Cosserat dynamics. | It matches the existing PCC state representation and is practical for iLQR and impedance control. |
| Use YAML engineering estimates for dynamics parameters first. | The user confirmed this is acceptable and it allows implementation before identification data exists. |
| Treat disturbance profile as configurable. | No third-party protocol exists yet, so force magnitude, direction, and duration must be easy to change. |
| Add the dynamics-based controller as an optional experimental `controller.type`. | The user requested preserving the current controller while enabling the new path for experiments. |

## Errors Encountered
| Error | Attempt | Resolution |
|-------|---------|------------|
| None in this planning session. | 1 | Not applicable. |

## Notes
- Acceptance indicators from the current project discussion:
  - Indicator 3.1: tip positioning error less than or equal to 2 cm.
  - Indicator 3.3 disturbance: maximum load/disturbance tip displacement deviation less than or equal to 4 cm.
  - Indicator 3.3 force tracking: contact interaction force average error less than or equal to 1 N.
  - Indicator 3.3 safety: body and environment cavity maintain the defined minimum safety distance.
- The old kinematic controller should remain runnable with existing configs.
- New dynamic control should be introduced by adding config modes, not by changing old behavior silently.
