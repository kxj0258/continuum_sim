# Dual-Arm Engine Staged Navigation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a scenario-native dual-arm engine navigation task that approaches and inserts with a prescribed mobile base, then holds the base while the executor performs a local inspection path and the observer follows.

**Architecture:** A pure task-plan module resolves engine-frame annotations into world-frame base and arm targets. A focused mobile-base pose controller and staged controller own phase execution, while executor/observer motion delegates to the existing fixed-base waypoint tracker. `SimulationApplication` remains the composition root.

**Tech Stack:** Python dataclasses, NumPy, YAML configuration, existing PCC/whole-body control, MuJoCo scenario backend, pytest-style tests.

## Global Constraints

- Do not add mesh path planning, mesh SDF queries, online replanning, contact, or cleaning.
- Engine annotation fields named `*_m` are metres and are not multiplied by `engine.scale`.
- Base insertion follows the authored `nozzle_axis_entry` path.
- Base motion is prescribed and stops during executor navigation.
- Existing uncommitted wiping/MuJoCo changes must not be overwritten or staged.
- Tests and validation commands are documented but not run automatically.

---

### Task 1: Engine navigation configuration and plan resolution

**Files:**
- Create: `src/continuum_sim/tasks/engine_navigation.py`
- Modify: `src/continuum_sim/application/scenario.py`
- Test: `tests/test_engine_navigation.py`

**Interfaces:**
- Produces: `EngineNavigationSpec.from_mapping(values)`.
- Produces: `EngineNavigationPlan`.
- Produces: `resolve_engine_navigation_plan(spec, scene, assembly)`.
- Consumes: `EngineSceneConfig`, `RobotAssemblyConfig`, `Pose6D`.

- [ ] **Step 1: Add focused tests for frame resolution, path resampling, pose construction, and invalid names**

```python
def test_resolve_engine_navigation_plan_builds_world_targets(engine_scene, mobile_assembly):
    spec = EngineNavigationSpec.from_mapping({
        "entry_region": "entry_port",
        "insertion_path": "nozzle_axis_entry",
        "pre_entry_standoff_m": 0.05,
        "insertion_waypoint_spacing_m": 0.02,
        "local_path": {"type": "transverse_square", "radius_m": 0.01, "samples": 40},
    })
    plan = resolve_engine_navigation_plan(spec, engine_scene, mobile_assembly)
    assert plan.insertion_tip_waypoints_world.shape[1] == 3
    assert len(plan.insertion_base_poses) == len(plan.insertion_tip_waypoints_world)
    assert plan.executor_waypoints_world.shape == (40, 3)
    assert np.allclose(
        plan.pre_entry_tip_world,
        plan.insertion_tip_waypoints_world[0] - 0.05 * plan.insertion_direction_world,
    )
```

- [ ] **Step 2: Document the test commands without running them**

```powershell
pytest tests/test_engine_navigation.py -q
```

Expected after implementation: all engine navigation plan tests pass.

- [ ] **Step 3: Implement validated dataclasses and pure geometry helpers**

```python
@dataclass(frozen=True)
class EngineNavigationSpec:
    entry_region: str
    insertion_path: str
    pre_entry_standoff_m: float
    insertion_waypoint_spacing_m: float
    base_position_tolerance_m: float
    base_orientation_tolerance_rad: float
    base_position_gain: float
    base_orientation_gain: float
    local_path_radius_m: float
    local_path_samples: int
    phase_timeout_steps: int

@dataclass(frozen=True)
class EngineNavigationPlan:
    pre_entry_tip_world: np.ndarray
    insertion_direction_world: np.ndarray
    insertion_tip_waypoints_world: np.ndarray
    pre_entry_base_pose: Pose6D
    insertion_base_poses: tuple[Pose6D, ...]
    executor_waypoints_world: np.ndarray
    observer_roi_world: np.ndarray
```

Implementation requirements:

- resolve named entry region and enabled exploration path;
- transform engine-frame annotations with effective engine-frame pose and no scale;
- verify entry and path start agree within 10 mm;
- resample each segment with endpoint preservation;
- construct a stable frame whose `+Z` follows insertion direction;
- compute base targets using the inverse straight executor-tip transform;
- generate a transverse square in the plane perpendicular to insertion direction.

- [ ] **Step 4: Parse `task.engine_navigation` into `ScenarioTaskConfig`**

Add `engine_navigation: EngineNavigationSpec | None`, add `engine_navigation` to
the exclusive waypoint-source list, and accept task type `engine_navigation`.

- [ ] **Step 5: Commit the configuration and plan boundary**

```powershell
git add src/continuum_sim/tasks/engine_navigation.py src/continuum_sim/application/scenario.py tests/test_engine_navigation.py
git commit -m "feat(engine): resolve staged navigation plans"
```

### Task 2: Mobile-base pose and staged navigation controllers

**Files:**
- Create: `src/continuum_sim/control/mobile_base_pose_control.py`
- Create: `src/continuum_sim/control/staged_engine_navigation.py`
- Test: `tests/test_staged_engine_navigation.py`

**Interfaces:**
- Consumes: `EngineNavigationPlan`, `RobotAssemblyConfig`, `EngineSceneQueryProtocol`.
- Produces: `MobileBasePoseController.compute_twist(current, target)`.
- Produces: `StagedEngineNavigationController.compute_command(state)`.
- Produces controller properties: `done`, `failed`, `phase`, `terminal_reason`.

- [ ] **Step 1: Add tests for bounded pose commands and deterministic phase transitions**

```python
def test_mobile_base_pose_controller_limits_twist():
    controller = MobileBasePoseController(position_gain=2.0, orientation_gain=2.0)
    twist, position_error, orientation_error = controller.compute_twist(
        Pose6D.identity(),
        Pose6D.from_rpy_rad(position=(1.0, 0.0, 0.0), rpy_rad=(0.0, 0.0, 1.0)),
        max_linear_speed=0.05,
        max_angular_speed=0.30,
    )
    assert np.linalg.norm(twist[:3]) <= 0.05
    assert np.linalg.norm(twist[3:]) <= 0.30
    assert position_error > 0.0
    assert orientation_error > 0.0

def test_staged_controller_holds_arms_during_base_approach(
    staged_controller,
    initial_robot_state,
):
    command = controller.compute_command(initial_state)
    assert controller.phase == "base_approach"
    assert np.any(command.base_twist_world)
    assert all(np.allclose(arm.tendon_rate_mps, 0.0) for arm in command.arms.values())
```

- [ ] **Step 2: Document the controller test command without running it**

```powershell
pytest tests/test_staged_engine_navigation.py -q
```

Expected after implementation: pose limits, phase transitions, base hold, timeouts,
and clearance termination tests pass.

- [ ] **Step 3: Implement norm-limited SE(3) pose control**

```python
class MobileBasePoseController:
    def compute_twist(
        self,
        current: Pose6D,
        target: Pose6D,
        *,
        max_linear_speed: float,
        max_angular_speed: float,
    ) -> tuple[np.ndarray, float, float]:
        position_delta = target.position - current.position
        rotation_delta = target.as_matrix()[:3, :3] @ current.as_matrix()[:3, :3].T
        rotation_vector = rotation_matrix_to_vector(rotation_delta)
        linear = limit_norm(self.position_gain * position_delta, max_linear_speed)
        angular = limit_norm(self.orientation_gain * rotation_vector, max_angular_speed)
        return (
            np.concatenate((linear, angular)),
            float(np.linalg.norm(position_delta)),
            float(np.linalg.norm(rotation_vector)),
        )
```

Use the rotation-vector logarithm of
`R_target @ R_current.T`, multiply position and orientation errors by their
gains, and norm-limit the linear and angular vectors independently.

- [ ] **Step 4: Implement the staged state machine**

```python
ENGINE_NAVIGATION_PHASES = (
    "base_approach",
    "base_insertion",
    "executor_navigation",
    "complete",
    "failed",
)
```

Behavior:

- base phases emit zero tendon rates;
- approach advances when both base pose tolerances pass;
- insertion advances through resolved base poses;
- executor phase delegates to `WaypointTrackingController` built from a
  `dataclasses.replace` copy of the assembly whose base is fixed;
- delegated base twist is discarded;
- clearance and phase timeout produce zero commands and a failure reason;
- metadata includes phase, target pose, errors, path progress, clearance, and reason.

- [ ] **Step 5: Commit controller behavior**

```powershell
git add src/continuum_sim/control/mobile_base_pose_control.py src/continuum_sim/control/staged_engine_navigation.py tests/test_staged_engine_navigation.py
git commit -m "feat(control): add staged engine navigation controller"
```

### Task 3: Application composition and runnable scenario

**Files:**
- Create: `configs/robots/assemblies/dual_spatial_mobile.yaml`
- Create: `configs/scenarios/dual_engine_navigation.yaml`
- Modify: `src/continuum_sim/application/application.py`
- Test: `tests/test_engine_navigation_application.py`

**Interfaces:**
- Consumes: `resolve_engine_navigation_plan` and `StagedEngineNavigationController`.
- Produces: `SimulationApplication.from_yaml("configs/scenarios/dual_engine_navigation.yaml")`.

- [ ] **Step 1: Add application composition tests**

```python
def test_dual_engine_navigation_composes_staged_controller():
    app = SimulationApplication.from_yaml(
        Path("configs/scenarios/dual_engine_navigation.yaml")
    )
    assert isinstance(app.loop.controller, StagedEngineNavigationController)
    assert app.config.task.type == "engine_navigation"
```

Also cover rejection of fixed-base assemblies and missing engine scenes.

- [ ] **Step 2: Add a mobile dual-arm assembly**

Copy the existing dual arm mounts and references, set:

```yaml
base:
  control_mode: prescribed_twist
  limits:
    calibrated: false
    position_min_m: [-3.0, -3.0, -3.0]
    position_max_m: [3.0, 3.0, 3.0]
    max_linear_speed_mps: 0.05
    max_angular_speed_rad_s: 0.30
```

- [ ] **Step 3: Add the engine navigation scenario**

Use the schema in the design, `max_steps: 8000`, the existing engine scene,
mobile-base-wrapped source XML, and a dedicated generated XML path.

- [ ] **Step 4: Compose the plan and controller in `SimulationApplication`**

Before ordinary task branches:

```python
elif config.task.type == "engine_navigation":
    if engine_scene is None or config.task.engine_navigation is None:
        raise ValueError("engine_navigation requires an engine scene and task specification.")
    plan = resolve_engine_navigation_plan(
        config.task.engine_navigation,
        engine_scene,
        assembly,
    )
    controller = StagedEngineNavigationController(
        assembly,
        plan,
        config.task.engine_navigation,
        scene_query=scene_query,
        waypoint_tolerance_m=config.task.waypoint_tolerance_m,
        min_clearance_m=config.task.min_clearance_m,
        terminate_on_clearance_violation=config.task.terminate_on_clearance_violation,
    )
```

Do not alter ordinary navigation, tracking, or wiping behavior.

- [ ] **Step 5: Document the composition test command without running it**

```powershell
pytest tests/test_engine_navigation_application.py -q
```

- [ ] **Step 6: Commit the runnable composition**

```powershell
git add configs/robots/assemblies/dual_spatial_mobile.yaml configs/scenarios/dual_engine_navigation.yaml src/continuum_sim/application/application.py tests/test_engine_navigation_application.py
git commit -m "feat(engine): compose dual-arm staged navigation scenario"
```

### Task 4: Artifact observability and documentation

**Files:**
- Modify: `src/continuum_sim/runtime/hooks.py`
- Modify: `src/continuum_sim/io/scenario_artifacts.py`
- Modify: `README.md`
- Test: `tests/test_scenario_artifacts.py`

**Interfaces:**
- Consumes staged command metadata.
- Produces NPZ keys `engine_navigation_phase`, `base_target_position_m`,
  `base_position_error_m`, `base_orientation_error_rad`,
  `engine_navigation_progress`, and `engine_navigation_terminal_reason`.

- [ ] **Step 1: Extend recorder/artifact tests**

```python
assert arrays["engine_navigation_phase"].tolist() == ["base_approach"]
assert arrays["base_target_position_m"].shape == (1, 3)
assert arrays["base_position_error_m"].tolist() == pytest.approx([0.1])
```

- [ ] **Step 2: Record staged metadata**

Add focused fields to `StateRecorderHook`, clear them on reset, append them
only when command metadata reports `task_type == "engine_navigation"`, and
export them from `scenario_artifacts`.

- [ ] **Step 3: Document usage and limitations**

Add:

```powershell
python scripts/run_scenario.py configs/scenarios/dual_engine_navigation.yaml
```

Document the three phases, authored-path safety assumption, and the lack of
full collision-mesh path planning.

- [ ] **Step 4: Document the artifact test command without running it**

```powershell
pytest tests/test_scenario_artifacts.py -q
```

- [ ] **Step 5: Commit observability and docs**

```powershell
git add src/continuum_sim/runtime/hooks.py src/continuum_sim/io/scenario_artifacts.py tests/test_scenario_artifacts.py README.md
git commit -m "docs(engine): expose staged navigation diagnostics"
```

### Task 5: Manual verification handoff

**Files:**
- Modify: `docs/superpowers/plans/2026-07-06-dual-engine-staged-navigation.md`

- [ ] **Step 1: Review the final diff without changing unrelated user files**

Confirm only the files named by Tasks 1–4 plus this plan and its design spec
belong to the feature.

- [ ] **Step 2: Do not run automated or simulation verification**

Per project instruction, leave these commands for the user:

```powershell
pytest tests/test_engine_navigation.py tests/test_staged_engine_navigation.py tests/test_engine_navigation_application.py tests/test_scenario_artifacts.py -q
python scripts/run_scenario.py configs/scenarios/dual_engine_navigation.yaml
```

- [ ] **Step 3: Report risks**

Report that the authored path is assumed safe, primitive coverage is incomplete,
the mobile bounds are simulation-only, and the first manual run may require
base speed/timeout and frame calibration.
