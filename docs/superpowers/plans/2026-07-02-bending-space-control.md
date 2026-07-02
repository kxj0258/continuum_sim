# Bending-Space Control Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make every normal continuum-arm task solve and integrate commands in the six-dimensional bending space while retaining an explicit raw-tendon debug mode.

**Architecture:** A model-level `BendingSpaceModel` owns the six-to-nine dimensional mapping and compatibility diagnostics. Whole-body and retained motor controllers solve bending rates; system backends enforce compatible commands with common-scale limiting and integrate bending targets before producing tendon-position targets.

**Tech Stack:** Python 3.11, NumPy, existing dataclasses and controller/runtime interfaces, pytest test sources, Matplotlib debug UI.

## Global Constraints

- Cover scenario idle, tracking, navigation, wiping, and engine-cleaning tasks, plus retained motor-space APIs.
- Axial strain is fixed to zero in normal control and state estimation.
- Do not add a third-party optimization dependency.
- Keep `RobotSystemCommand` as the runtime command container.
- Default arm commands are bending-compatible; raw tendon control requires an explicit debug mode.
- Do not run tests, builds, linters, formatters, installers, viewers, simulations, or other verification commands during implementation.
- Preserve unrelated working-tree changes and stage only files belonging to this feature.

---

### Task 1: Bending-space model and compatibility contract

**Files:**
- Create: `src/continuum_sim/model/bending_space.py`
- Modify: `src/continuum_sim/model/__init__.py`
- Modify: `src/continuum_sim/system/types.py`
- Test: `tests/test_bending_space.py`

**Interfaces:**
- Produces: `BendingSpaceModel.from_arm(params, tendons)`, `to_q`, `to_tendon`, `estimate`, `project`, `residual`, and `is_compatible`.
- Produces: `ArmTendonRateCommand.control_space` with values `bending_compatible` and `raw_tendon_debug`.

- [ ] Add tests defining the six-coordinate ordering, zero axial entries, exact forward mapping, least-squares inverse, projection, residual, and rank validation.
- [ ] Implement immutable mapping arrays from the existing full coupling matrix and a six-column selection matrix.
- [ ] Add strict finite/shape validation and absolute-plus-relative compatibility tolerance.
- [ ] Extend the arm command dataclass with a safe default control-space value and validate allowed modes.
- [ ] Export the new model API.
- [ ] Record the manual command `pytest tests/test_bending_space.py tests/test_system_types.py -v` without executing it.

### Task 2: Bending layout, kinematics, and whole-body solve

**Files:**
- Modify: `src/continuum_sim/system/control_layout.py`
- Modify: `src/continuum_sim/kinematics/whole_body.py`
- Modify: `src/continuum_sim/kinematics/__init__.py`
- Modify: `src/continuum_sim/control/whole_body_controller.py`
- Test: `tests/test_control_layout.py`
- Test: `tests/test_whole_body_kinematics.py`
- Test: `tests/test_whole_body_controller.py`

**Interfaces:**
- Produces: `ControlLayout` optimization slices containing six bending coordinates per arm.
- Produces: `bending_position_jacobian` and `centerline_point_bending_jacobian`.
- Consumes: `BendingSpaceModel.to_tendon(bending_rate)`.

- [ ] Change optimization layout sizing from tendon count to `2 * segment_count`, while retaining physical tendon slices for backend vectors.
- [ ] Add direct `J_q @ S_b` bending Jacobians for tip and centerline points.
- [ ] Assemble all task matrices in base-plus-bending layout.
- [ ] Solve and regularize bending rates, map them to tendon rates, and place those rates in `RobotSystemCommand`.
- [ ] Replace componentwise tendon clipping with common per-arm scaling derived from physical tendon-rate limits.
- [ ] Add solver diagnostics for requested/applied bending rates, mapping rank/condition, scale, and compatibility residual.
- [ ] Adapt dimension and compatibility tests.
- [ ] Record the manual command `pytest tests/test_control_layout.py tests/test_whole_body_kinematics.py tests/test_whole_body_controller.py -v` without executing it.

### Task 3: Coordinated scenario controllers

**Files:**
- Modify: `src/continuum_sim/control/coordinated_tracking.py`
- Modify: `src/continuum_sim/control/scenario_controllers.py`
- Test: `tests/test_coordinated_tracking.py`
- Test: `tests/test_scenario_controllers.py`

**Interfaces:**
- Consumes: bending layout and bending Jacobians from Task 2.
- Produces: compatible commands for idle, tracking, navigation, wiping, and engine cleaning.

- [ ] Estimate each arm state through `BendingSpaceModel.estimate` and reconstruct zero-strain PCC state.
- [ ] Replace tip and centerline tendon Jacobians with bending Jacobians.
- [ ] Preserve executor/observer column masking using bending slices.
- [ ] Attach measured compatibility residuals and mapping diagnostics to controller metadata.
- [ ] Ensure zero/termination commands use the default compatible control mode.
- [ ] Adapt scenario controller tests for six-variable internal layouts and compatible nine-tendon outputs.
- [ ] Record the manual command `pytest tests/test_coordinated_tracking.py tests/test_scenario_controllers.py -v` without executing it.

### Task 4: Compatible backend integration and limiting

**Files:**
- Modify: `src/continuum_sim/control/tendon_rate_control.py`
- Modify: `src/continuum_sim/control/__init__.py`
- Modify: `src/continuum_sim/backends/analytic_system_backend.py`
- Modify: `src/continuum_sim/backends/mujoco_system_backend.py`
- Test: `tests/test_tendon_rate_control.py`
- Test: `tests/test_analytic_system_backend.py`
- Test: `tests/test_mujoco_system_backend.py`

**Interfaces:**
- Produces: `BendingRateIntegrator(model, limits)` and a step result containing bending/tendon rates, common scale, residual, and saturation arrays.
- Consumes: `ArmTendonRateCommand.control_space`.

- [ ] Implement compatible target integration with one scale satisfying all rate and next-position limits.
- [ ] Keep the existing independent `TendonRateIntegrator` only for explicit raw debug commands.
- [ ] Add deterministic mode switching that projects the current raw target before compatible integration.
- [ ] Make analytic state estimation bending-only and expose compatibility metadata.
- [ ] Make MuJoCo targets derive from compatible bending state and expose scale/saturation metadata.
- [ ] Reject incompatible normal commands before stepping either backend.
- [ ] Adapt backend tests for compatible and raw modes.
- [ ] Record the manual command `pytest tests/test_tendon_rate_control.py tests/test_analytic_system_backend.py tests/test_mujoco_system_backend.py -v` without executing it.

### Task 5: Retained motor-space controllers

**Files:**
- Modify: `src/continuum_sim/kinematics/differential.py`
- Modify: `src/continuum_sim/control/differential_ik.py`
- Modify: `src/continuum_sim/control/navigation_controller.py`
- Modify: `src/continuum_sim/control/hybrid_force_position.py`
- Modify: `src/continuum_sim/control/dynamic_adaptive_impedance.py`
- Modify: `src/continuum_sim/control/engine_cleaning_controller.py`
- Test: `tests/test_differential_ik.py`
- Test: `tests/test_navigation_controller.py`
- Test: `tests/test_hybrid_force_position.py`
- Test: `tests/test_dynamic_adaptive_impedance.py`
- Test: `tests/test_engine_cleaning_controller.py`

**Interfaces:**
- Produces: motor commands whose tendon-equivalent rates lie in `range(C_b)`.
- Consumes: existing motor spool/sign mapping and `BendingSpaceModel`.

- [ ] Add bending-state Cartesian Jacobians and a tendon-rate-to-motor-rate mapping using configured spool radii/signs.
- [ ] Change legacy state estimation to bending projection with zero axial strain.
- [ ] Solve tracking, clearance, wiping, adaptive impedance, and engine-cleaning motion in bending rate.
- [ ] Map once to compatible tendon rate and then to motor rate.
- [ ] Apply a common scale for motor and tendon limits instead of clipping motors independently.
- [ ] Preserve public return shapes and diagnostic dictionaries, adding bending and compatibility fields.
- [ ] Adapt each legacy controller test to assert tendon-equivalent compatibility.
- [ ] Record the manual command `pytest tests/test_differential_ik.py tests/test_navigation_controller.py tests/test_hybrid_force_position.py tests/test_dynamic_adaptive_impedance.py tests/test_engine_cleaning_controller.py -v` without executing it.

### Task 6: Debug modes and compatibility display

**Files:**
- Modify: `src/continuum_sim/visualization/mujoco_system_debug_viewer.py`
- Modify: `scripts/debug_mujoco.py`
- Test: `tests/test_mujoco_system_debug_viewer.py`
- Test: `tests/test_system_tendon_debug.py`

**Interfaces:**
- Produces: default compatible bending inputs and explicit raw-tendon diagnostic inputs.
- Consumes: command control-space modes and bending mapping diagnostics.

- [ ] Add compatible/raw mode selection, defaulting to compatible.
- [ ] In compatible mode expose segment `kx`/`ky` targets and map them to synchronized tendon millimetre fields/sliders.
- [ ] In raw mode retain independent tendon millimetre sliders and text boxes.
- [ ] Mark raw commands with `raw_tendon_debug`; keep normal commands compatible.
- [ ] Display compatibility residual, common scale, and a clear nonphysical raw-mode warning.
- [ ] Adapt UI state and command-construction tests.
- [ ] Record the manual command `pytest tests/test_mujoco_system_debug_viewer.py tests/test_system_tendon_debug.py -v` without executing it.

### Task 7: Documentation, configuration semantics, and handoff

**Files:**
- Modify: `README.md`
- Modify: `docs/architecture_overview.md`
- Modify: `docs/coordinate_conventions.md`
- Modify: `configs/scenarios/dual_mujoco_tracking.yaml`

**Interfaces:**
- Documents all interfaces and operational modes introduced by Tasks 1-6.

- [ ] Document the six-coordinate ordering, equations, zero-strain assumption, and end-to-end data flow.
- [ ] Document common-scale limiting and the difference between commanded compatibility and measured residual.
- [ ] Document scenario-wide coverage and retained legacy API behavior.
- [ ] Document default compatible debug mode, explicit raw diagnostic mode, units, and warnings.
- [ ] Add conservative compatibility tolerance configuration only if implementation requires it; otherwise document fixed numerical defaults.
- [ ] List focused and full manual validation commands without executing them.
- [ ] Stage only feature-owned files and create an implementation commit with an appropriate message.
