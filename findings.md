# Findings and Decisions

## Requirements
- Preserve the existing kinematic hybrid force-position control implementation.
- Add a new optional dynamics-based hybrid force-position controller using a PCC reduced dynamics model.
- Use YAML engineering estimates for dynamics parameters initially.
- Disturbance acceptance tests must be configurable because no third-party disturbance profile exists yet.
- The new dynamics controller should be selected through `controller.type` as an experimental mode.
- Continue supporting the broader project goals:
  - DMP trajectory generation for adaptable smooth end-effector paths.
  - CBF-QP conformal positioning in narrow cavity environments.
  - SDF gradient and null-space projection for body self-avoidance.
  - iLQR and adaptive impedance control for disturbance rejection and force tracking.
  - Automated acceptance scripts and Markdown reports.

## Research Findings
- Current trajectory generation is centered in `src/continuum_sim/tasks/trajectory_tracking_config.py`.
- Existing kinematic force-position control is in `src/continuum_sim/control/hybrid_force_position.py`.
- Existing wiping controller config accepts only `controller.type: hybrid_force_position` through `WIPING_CONTROLLER_TYPES`.
- Existing MuJoCo wiping runtime records normal force, force error, contact source, motor state, tendon state, and MuJoCo state in `MujocoWipingResult`.
- Existing PCC state has 3 values per segment and 3 segments, so the reduced dynamics generalized coordinate size is 9.
- Existing kinematics already provide FK, finite-difference Jacobians, motor-to-qdot mapping, and contact projection utilities that can be reused for dynamics.
- Existing navigation controller already queries centerline clearance and computes a clearance avoidance term; this is a natural insertion point for CBF/SDF upgrades.

## Technical Decisions
| Decision | Rationale |
|----------|-----------|
| Use a PCC reduced dynamics model with 9 generalized coordinates. | It matches the current `q = [kx, ky, eps] * 3` representation. |
| Start with `M(q) qddot + D qdot + K(q-q0) = tau_tendon + J_tip.T F_contact + tau_disturbance`. | This is sufficient for adaptive impedance and iLQR without requiring a full high-fidelity soft-body model. |
| Compute mass matrix from centerline sample Jacobians initially. | It reuses existing finite-difference tools and keeps implementation simple. |
| Keep the existing kinematic controller as the default. | It protects current demos, tests, and saved run behavior. |
| Add `dynamic_adaptive_impedance` as a new optional controller type. | It allows side-by-side comparison and gradual validation. |
| Keep disturbance parameters in YAML/scripts. | Third-party testing details are unknown, so the test profile must remain adjustable. |

## Issues Encountered
| Issue | Resolution |
|-------|------------|
| No third-party disturbance protocol yet. | Plan acceptance scripts with configurable default engineering profiles. |
| Dynamics parameters are not identified yet. | Use YAML engineering estimates first, with a later parameter-identification phase if needed. |

## 2026-06-18 Workspace Findings
- The previous plan is marked complete, but the working tree contains uncommitted changes in `tests/test_adaptive_impedance.py` and `tests/test_wiping_config.py`.
- `tests/test_adaptive_impedance.py` adds an expectation that the dynamic impedance prediction keeps axial strain coordinates (`q[2::3]`, `qdot[2::3]`, `qddot[2::3]`) at zero.
- `tests/test_wiping_config.py` adds an expectation that the MuJoCo tendon-position actuator `kp` is `500000.0` while legacy joint-position `kp` remains `2.0`.
- The next step is to run these tests in the `continuum_sim` conda environment and treat any failures as the current unfinished work.
- Focused tests failed before implementation with axial predictions near `-0.00499896` and `tendon_position.kp == 1000000.0`.
- Root cause: the dynamic wiping controller solved the DLS command over all 9 PCC coordinates even though the active MuJoCo tendon model has no axial strain DOF (`include_axial_strain: false`).
- Root cause: `configs/mujoco.yaml` retained the previous tendon-position gain instead of the lower `500000.0` setting.

## 2026-06-18 Git History Reinitialization Findings
- Existing local history bundle: `C:\work_kxj\continuum_sim_history_backup_20260616_112931.bundle`.
- `git bundle verify` reports that the existing bundle records a complete history and contains refs for `main`, `refactor/task3-off-minimal-core`, and their `origin/*` counterparts.
- Current remote is `origin git@github.com:kxj0258/continuum_sim.git`.
- Current `main` and `origin/main` both point to `47ec19e Tune wiping force control and MuJoCo actuators` before the new rewrite.
- Safety strategy: keep the 2026-06-16 bundle, create a new 2026-06-18 bundle for current history, create a local-only `history/main-before-rewrite-*` branch, then force-push only the new compact `main`.

## Resources
- `src/continuum_sim/tasks/trajectory_tracking_config.py`
- `src/continuum_sim/control/hybrid_force_position.py`
- `src/continuum_sim/control/navigation_controller.py`
- `src/continuum_sim/kinematics/pcc.py`
- `src/continuum_sim/kinematics/differential.py`
- `src/continuum_sim/model/robot_params.py`
- `src/continuum_sim/runtime/mujoco_wiping_runtime.py`
- `configs/tasks/mujoco_wiping_board.yaml`
- `configs/robot_3seg.yaml`

## Visual/Browser Findings
- PDF-derived task context from earlier planning:
  - Task 3 subtask 1 asks for conformal positioning control in geometrically constrained environments, including DMP and barrier-function optimization.
  - Task 3 subtask 3 asks for iLQR-based adaptive impedance, SDF costs, dynamic-consistent null-space projection, and body safety control.
  - Final acceptance indicator 3.1 requires flexible continuum tip positioning error less than or equal to 2 cm.
  - Final acceptance indicator 3.3 requires disturbance displacement deviation less than or equal to 4 cm, body safety distance maintenance, and contact-force average error less than or equal to 1 N.

---
Update this file after every 2 view/browser/search operations.
