# Engine Dual-Arm Long-Term Plan

## Current Baseline

The current `continuum_sim` codebase already provides a stable YAML-first simulation stack built around a single three-segment tendon-driven continuum arm. The baseline includes PCC forward and differential kinematics, tendon and motor mapping, analytic and MuJoCo backends, tracking/navigation/wiping runtimes, hybrid force-position control, dynamic adaptive impedance experiments, artifact saving, replay support, and a broad pytest regression suite.

This baseline is valuable because future engine and dual-arm work does not need to rebuild the infrastructure for configuration loading, motion generation, runtime orchestration, validation, or regression testing. The safest extension path is to add new scene/config/runtime modules beside the existing wiping and navigation workflows rather than rewriting them.

## Long-Term Goal Overview

The long-term roadmap expands the current single-continuum sandbox into an engine cleaning research platform with the following capabilities:

- engine scene support with imported 3D assets
- mobile 6D base pose abstraction for the continuum mounting point
- dual continuum arms attached to a shared snake-arm end-effector
- observer arm with hand-eye camera and airgun
- executor arm with carbon-removal tool
- visual servoing for observation-guided execution
- dual-arm collision avoidance and interference suppression
- engine cleaning task runtime
- dual-arm engine cleaning task runtime and task-level state machine
- sim2real-facing noise, latency, and hardware interface layers

## Design Principles

- The first engine model version must separate `visual mesh` from `simplified collision geoms`.
- The first snake-arm integration version should abstract the mount as a controllable `6D base pose`, not a full snake-arm dynamics model.
- The first vision version should use `ground-truth perception` before introducing RGB/depth recognition.
- The first dual-arm collision safety layer should use a `centerline/capsule clearance safety filter` before any null-space avoidance or MPC upgrade.

## Milestone Plan

### M0 baseline audit

**Goal**

Capture the current stable system boundaries, identify reusable modules, and define extension constraints so new work does not destabilize tracking, navigation, or wiping.

**Recommended new files**

- `docs/long_term_engine_dual_arm_plan.md`
- optional future audit notes under `docs/`

**Testing approach**

- run core and full regression suites before and after foundation changes
- inspect CLI and YAML compatibility manually from the baseline documentation

**Acceptance criteria**

- baseline capabilities are documented clearly
- protected legacy subsystems are identified explicitly
- future milestones are sequenced with low-risk dependency order

### M1 engine scene loader

**Goal**

Add a standalone engine scene configuration format and loader so future runtimes can consume engine assets and named task regions without modifying existing scene loaders.

**Recommended new files**

- `configs/scenes/engine_cleaning.yaml`
- `src/continuum_sim/scenes/engine_scene.py`
- `tests/test_engine_scene.py`

**Testing approach**

- headless YAML loading tests
- strict vs non-strict asset validation tests
- region parsing and path resolution tests

**Acceptance criteria**

- engine scene YAML loads successfully
- placeholder assets can be tolerated with warnings in non-strict mode
- strict mode fails with clear asset errors
- no existing scene/runtime module depends on the new loader

### M2 6D mobile base

**Goal**

Introduce a mount-frame abstraction that lets future observer/executor continua move with a controllable six-degree-of-freedom base pose.

**Recommended new files**

- `src/continuum_sim/model/mobile_base_pose.py`
- `src/continuum_sim/kinematics/mobile_mount.py`
- `tests/test_mobile_base_pose.py`
- optional `configs/tasks/engine_*` task stubs

**Testing approach**

- unit tests for pose composition and frame transforms
- regression checks showing existing PCC kinematics remain unchanged without a mobile base

**Acceptance criteria**

- mount pose composes cleanly with continuum base frame
- APIs are independent from legacy wiping/navigation loaders
- future runtimes can consume the pose object without special-case hacks

### M3 dual continuum arms

**Goal**

Represent two continua with separate identities, attachment metadata, and independent actuation/configuration while preserving shared infrastructure patterns.

**Recommended new files**

- `src/continuum_sim/model/dual_arm_config.py`
- `src/continuum_sim/kinematics/dual_continuum_chain.py`
- `tests/test_dual_arm_config.py`
- `tests/test_dual_continuum_chain.py`

**Testing approach**

- configuration loading tests for observer/executor arms
- kinematic transform tests with shared mount pose
- regression tests proving single-arm paths still work

**Acceptance criteria**

- two-arm configuration loads cleanly
- each arm can be addressed independently
- single-arm modules remain untouched and compatible

### M4 tool and camera attachments

**Goal**

Add structured attachment metadata for observer camera, airgun, and executor cleaning tool so future runtimes and perception modules use consistent mount definitions.

**Recommended new files**

- `src/continuum_sim/model/attachment_config.py`
- `configs/tools/observer_camera.yaml`
- `configs/tools/executor_tool.yaml`
- `tests/test_attachment_config.py`

**Testing approach**

- unit tests for attachment pose offsets and validation
- fixture-based checks for camera/tool metadata parsing

**Acceptance criteria**

- attachment poses are expressible in YAML
- observer and executor attachments can be loaded independently
- no runtime coupling is introduced prematurely

### M5 engine surface path generation

**Goal**

Generate engine-specific inspection and cleaning paths from scene regions and simplified surface descriptors.

**Recommended new files**

- `src/continuum_sim/tasks/engine_path_generation.py`
- `tests/test_engine_path_generation.py`
- additional scene region examples under `configs/scenes/`

**Testing approach**

- deterministic path generation tests from named regions
- geometry edge-case coverage for entry and deposit surfaces

**Acceptance criteria**

- inspection and cleaning paths can be created from engine regions
- output path format is compatible with future runtimes
- no dependency on full CAD collision is required

### M6 executor engine cleaning controller

**Goal**

Create a first executor-focused cleaning controller that adapts existing tangent/normal regulation ideas to engine contact tasks.

**Recommended new files**

- `src/continuum_sim/control/engine_cleaning_controller.py`
- `tests/test_engine_cleaning_controller.py`
- future executor task YAMLs under `configs/tasks/`

**Testing approach**

- controller unit tests using synthetic engine surface normals and targets
- regression tests comparing behavior with existing wiping control assumptions where applicable

**Acceptance criteria**

- executor controller produces stable command outputs from engine path targets
- controller remains decoupled from observer and state-machine logic
- legacy wiping controller remains unchanged

### M7 observer camera and visual servo

**Goal**

Introduce ground-truth perception and a first visual-servo loop for the observer arm before image-based perception is attempted.

**Recommended new files**

- `src/continuum_sim/perception/ground_truth_engine_perception.py`
- `src/continuum_sim/control/visual_servo.py`
- `tests/test_ground_truth_engine_perception.py`
- `tests/test_visual_servo.py`

**Testing approach**

- synthetic perception tests from known scene/object poses
- closed-loop unit tests for observer target correction

**Acceptance criteria**

- observer arm can receive deterministic scene observations
- servo outputs are testable without image rendering
- APIs leave room for later RGB/depth replacement

### M8 dual-arm collision avoidance

**Goal**

Add a first safety filter that uses centerline and capsule clearance to suppress dual-arm interference.

**Recommended new files**

- `src/continuum_sim/control/dual_arm_clearance_filter.py`
- `tests/test_dual_arm_clearance_filter.py`
- optional shared geometry helpers under `src/continuum_sim/kinematics/`

**Testing approach**

- near-collision synthetic cases
- safe-pass-through and rejection tests
- regression tests to ensure the filter can be disabled cleanly

**Acceptance criteria**

- unsafe pairwise arm configurations are detected reliably
- filter works with centerline/capsule abstractions only
- no MPC or null-space machinery is required in the first version

### M9 full task state machine

**Goal**

Orchestrate entry, observation, servo alignment, cleaning, retreat, and retry logic for the combined engine task.

**Recommended new files**

- `src/continuum_sim/runtime/mujoco_engine_cleaning_runtime.py`
- `src/continuum_sim/runtime/mujoco_dual_arm_engine_cleaning_runtime.py`
- `src/continuum_sim/tasks/engine_mission_config.py`
- `tests/test_engine_cleaning_runtime.py`

**Testing approach**

- headless runtime tests with mocked or analytic subsystems
- state transition coverage for success, retry, and abort flows

**Acceptance criteria**

- task phases are explicit and testable
- observer/executor coordination is managed by a stable state machine
- existing runtimes continue to operate unchanged

### M10 sim2real noise, latency, and hardware interface

**Goal**

Prepare the research stack for deployment-oriented experiments by introducing configurable noise, delay, and interface boundaries.

**Recommended new files**

- `src/continuum_sim/io/hardware_interface.py`
- `src/continuum_sim/validation/sim2real_noise.py`
- `tests/test_hardware_interface.py`
- `tests/test_sim2real_noise.py`

**Testing approach**

- deterministic noise injection tests
- latency simulation tests
- stub hardware adapter tests

**Acceptance criteria**

- perception and actuation paths can be wrapped with configurable delay/noise
- hardware-facing APIs are explicit and mockable
- simulation-only and hardware-oriented code paths remain separable

## Recommended Execution Strategy

The safest sequence is to complete M1 and M2 before attempting any dual-arm runtime work. M1 defines the engine-scene contract. M2 establishes the 6D mount abstraction that replaces the temptation to immediately simulate full snake-arm dynamics. Only after those boundaries exist should the project add dual-arm modeling, tool attachments, and engine-specific task logic.

In practical terms, the near-term next step after this foundation round should be M2: a mobile-base abstraction with clean pose composition and tests, still without touching the existing tracking, navigation, or wiping runtime contracts.
