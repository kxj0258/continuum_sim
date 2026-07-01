# Composable Spatial Continuum System Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a named base-plus-arms system API with direct tendon-rate control and reusable single/dual-arm engine composition.

**Architecture:** Pure assembly and system types define policy-facing state and commands. Kinematics and control operate on named blocks, while MuJoCo-specific flattening, neutral-length offsets, and freejoint pose updates remain in the backend adapter.

**Tech Stack:** Python 3.11, dataclasses, NumPy, PyYAML, MuJoCo MJCF/XML.

## Global Constraints

- Base twists are world-frame `[vx, vy, vz, wx, wy, wz]`.
- Spatial simulation controllers output tendon-length rates in metres per second.
- Do not model tendon tension or slack in this phase.
- Engine control geometry uses primitive collision geoms.
- Do not automatically run tests, linters, formatters, builds, installers, or simulations.
- Do not preserve legacy backend, default-arm, flat-array, or motor-based
  spatial-control interfaces.

---

### Task 1: Assembly and system contracts

**Files:**
- Create: `src/continuum_sim/model/robot_assembly.py`
- Create: `src/continuum_sim/system/types.py`
- Create: `src/continuum_sim/system/control_layout.py`
- Create: `src/continuum_sim/system/__init__.py`
- Create: `configs/robots/assemblies/single_spatial.yaml`
- Create: `configs/robots/assemblies/dual_spatial.yaml`

**Interfaces:**
- Produces: `load_robot_assembly_config()`, `RobotSystemState`,
  `RobotSystemCommand`, and `ControlLayout`.

- [ ] Define base and arm assembly dataclasses and validate world-frame twist mode.
- [ ] Define immutable named state and tendon-rate command dataclasses.
- [ ] Derive named base and tendon slices without global indices in arm configs.
- [ ] Add single and dual assembly YAML composition roots.

### Task 2: Direct tendon-rate actuation

**Files:**
- Create: `src/continuum_sim/control/tendon_rate_control.py`
- Modify: `src/continuum_sim/control/__init__.py`

**Interfaces:**
- Consumes: named tendon-rate commands.
- Produces: `TendonRateIntegrator.step(rate, dt)` and clipped tendon displacement.

- [ ] Validate per-tendon rate and displacement limits.
- [ ] Integrate clipped rates into clipped tendon displacement without windup.
- [ ] Support reset from zero displacement or observed displacement.

### Task 3: Whole-body kinematics and singularity diagnostics

**Files:**
- Create: `src/continuum_sim/kinematics/whole_body.py`
- Modify: `src/continuum_sim/kinematics/__init__.py`

**Interfaces:**
- Produces: `base_point_jacobian_world()`,
  `assemble_whole_body_jacobian()`, and `analyze_singularity()`.

- [ ] Implement the world-frame 6D base point Jacobian.
- [ ] Insert per-arm tendon Jacobians into named layout blocks.
- [ ] Compute SVD rank, minimum singular value, condition number, adaptive
  damping, and velocity scale.

### Task 4: Whole-body controller

**Files:**
- Create: `src/continuum_sim/control/whole_body_controller.py`
- Modify: `src/continuum_sim/control/__init__.py`

**Interfaces:**
- Consumes: named weighted tasks and a `ControlLayout`.
- Produces: `RobotSystemCommand` with base twist and named tendon rates.

- [ ] Stack executor, observer, collision-avoidance, and regularization tasks.
- [ ] Solve weighted adaptive damped least squares.
- [ ] Apply base and per-arm velocity limits.
- [ ] Keep observer tracking and executor collision avoidance configurable as
  higher-priority weights.

### Task 5: MuJoCo system backend adapter

**Files:**
- Modify: `src/continuum_sim/backends/base_types.py`
- Modify: `src/continuum_sim/backends/mujoco_backend.py`

**Interfaces:**
- Produces: `reset_system()`, `get_system_state()`, and `step_system()`.

- [ ] Resolve named executor/observer tip and segment sites.
- [ ] Split observed tendon state by `ControlLayout`.
- [ ] Integrate tendon rates and write neutral-plus-displacement targets.
- [ ] Integrate world-frame base twist into prescribed freejoint pose.
- [ ] Replace the legacy ndarray `step()` API with the named system command API.

### Task 6: Generic simulation loop

**Files:**
- Create: `src/continuum_sim/runtime/simulation_loop.py`
- Modify: `src/continuum_sim/runtime/__init__.py`

**Interfaces:**
- Produces: `SimulationLoop`, `SystemControllerProtocol`, and hook protocols.

- [ ] Make the loop depend on system backend/controller protocols.
- [ ] Keep recording, viewer sync, and stop conditions optional.
- [ ] Return named system-state history without importing MuJoCo.

### Task 7: Engine scene composition

**Files:**
- Create: `src/continuum_sim/scenes/engine_mjcf_adapter.py`
- Modify: `src/continuum_sim/scenes/__init__.py`
- Modify: `scripts/preview_engine_scene_mujoco.py`

**Interfaces:**
- Produces: `build_engine_mujoco_scene_xml()`.

- [ ] Inject visual and optional collision mesh assets into a robot MJCF.
- [ ] Convert enabled primitive collision geoms to MuJoCo geoms.
- [ ] Apply engine/world frame transforms consistently.
- [ ] Make the preview script call the reusable scene adapter.

### Task 8: Single/dual engine compositions and documentation

**Files:**
- Create: `configs/systems/single_spatial_engine.yaml`
- Create: `configs/systems/dual_spatial_engine.yaml`
- Modify: `README.md`
- Modify: `docs/architecture_overview.md`

**Interfaces:**
- Produces: documented composition files for both engine variants.

- [ ] Bind assembly, MuJoCo backend, engine scene, and task configuration.
- [ ] Document the 15D single-arm and 24D dual-arm layouts.
- [ ] Document deferred calibration fields and migration from motor commands.
- [ ] Record manual validation commands without executing them.
