# Dual-Arm Engine Staged Navigation Design

## Goal

Add one scenario-native MuJoCo task that combines the dual spatial-arm assembly
with the existing engine scene and performs deterministic staged navigation:

1. move and orient the mobile base to a pre-entry pose;
2. insert the base along the configured `nozzle_axis_entry` path while both arms
   hold their nominal shapes;
3. hold the base pose and navigate the executor through a small local inspection
   path while the observer follows an observation target;
4. stop on completion, timeout, or a configured clearance violation.

The first version uses authored engine annotations and primitive clearance
queries. It does not attempt collision-mesh path planning or claim full
collision-mesh avoidance.

## Existing Constraints

- `dual_spatial.yaml` currently fixes the base, so the new scenario needs a
  mobile variant with `control_mode: prescribed_twist`.
- The configured entry/path envelope exceeds the current assembly translation
  limits of +/-1 m. The mobile engine assembly must use calibrated limits that
  contain the resolved path plus a safety margin; the initial implementation
  uses +/-3 m pending tighter calibration.
- The executor arm is approximately 0.12 m long.
- `nozzle_axis_entry` is approximately 1.25 m long, so it cannot be traversed
  by a fixed-base continuum arm.
- Engine annotation values whose `frame` is `engine` are already expressed in
  metres. They are rotated and translated by the effective engine frame; they
  must not be multiplied by `engine.scale`.
- The current engine control query is backed by authored primitive collision
  geometry. The full collision mesh is not available as a signed-distance query.
- The MuJoCo mobile base is prescribed by writing the free-joint pose each
  control step. Holding zero base twist after insertion therefore holds the base
  deterministically.

## Considered Approaches

### Extend the existing `NavigationController`

This minimizes new classes, but mixes base pose control, phase transitions,
engine-frame resolution, arm tracking, and ordinary waypoint navigation in one
controller. It also makes ordinary structured-scene navigation depend on engine
concepts.

### Add `StagedEngineNavigationController` — selected

A dedicated controller owns the phase state machine and delegates arm motion to
the existing waypoint/coordinated tracking controller. Engine path resolution
stays in a task-plan module, and mobile-base pose control stays in a small
controller helper.

This preserves the current application architecture while making phase
transitions explicit and observable.

### Chain multiple scenarios

Running base approach, insertion, and arm navigation as separate scenarios is
simple, but each scenario reset loses physical and controller state. It is not
appropriate for one continuous navigation experiment.

## Scenario Model

Add task type `engine_navigation` and an engine-navigation specification:

```yaml
task:
  type: engine_navigation
  waypoint_tolerance_m: 0.003
  min_clearance_m: 0.010
  terminate_on_clearance_violation: true
  engine_navigation:
    entry_region: entry_port
    insertion_path: nozzle_axis_entry
    pre_entry_standoff_m: 0.050
    insertion_waypoint_spacing_m: 0.020
    base_position_tolerance_m: 0.005
    base_orientation_tolerance_rad: 0.035
    base_position_gain: 1.5
    base_orientation_gain: 2.0
    local_path:
      type: transverse_square
      radius_m: 0.010
      samples: 40
    observer:
      executor_offset_world: [0.0, -0.04, 0.02]
      roi_blend: 0.25
```

The new scenario uses:

- `assembly_config_path: ../robots/assemblies/dual_spatial_mobile.yaml`;
- `scene.engine_config_path: ../scenes/engine_cleaning.yaml`;
- the existing mobile-base-wrapped dual-arm source XML;
- a dedicated generated XML path such as
  `output/generated/scenario_dual_engine_navigation.xml`.

The mobile assembly keeps the existing arm mounts and tendon configs, changes
the base control mode to `prescribed_twist`, and uses translation bounds of
`[-3, -3, -3]` to `[3, 3, 3]`. These are simulation bounds, not a claim about a
physical platform workspace.

## Engine-Frame Resolution

Add a single engine annotation resolver responsible for:

- engine-frame point to world point;
- engine-frame direction to world unit direction;
- named region position/center and normal resolution;
- named enabled exploration path resolution.

For an engine-frame point:

```text
p_world = effective_engine_frame_pose * p_engine
```

No `engine.scale` is applied because the annotation fields are named `*_m` and
the current preview/runtime behavior treats them as metres.

The resolver validates:

- named region and path existence;
- required center/position and normal fields;
- at least two non-duplicate path points;
- the insertion path start being consistent with the entry-region center within
  a configurable tolerance.

## Pose Construction

The entry-region normal and the first insertion segment define the desired base
forward direction. The insertion path direction wins when the two differ
slightly.

The desired executor straight-tip frame is built with:

- local `+Z` aligned with the insertion direction;
- local `+X` selected from a stable projected world reference axis;
- local `+Y` completing a right-handed frame.

The pre-entry executor target is:

```text
entry_point - pre_entry_standoff_m * insertion_direction
```

The corresponding base target is computed from the executor's straight
kinematic tip transform:

```text
T_world_base = T_world_executor_tip_target
               * inverse(T_base_executor_straight_tip)
```

This avoids hard-coded mount offsets and remains valid if the executor mount
pose changes.

## Phase State Machine

### `base_approach`

- Arms receive zero tendon-rate commands and retain their current targets.
- A proportional SE(3) pose controller commands world-frame base twist.
- Linear and angular commands are norm-limited by assembly base limits.
- Completion requires both position and orientation tolerances.
- Clearance is monitored but the authored insertion path is treated as the
  primary safety contract.

### `base_insertion`

- The configured exploration polyline is resampled at approximately
  `insertion_waypoint_spacing_m`.
- Each insertion waypoint is converted into a base pose using the straight
  executor-tip transform.
- Arms continue to hold zero tendon rates.
- The base advances by pose tolerance.
- On the final insertion waypoint, the exact achieved base pose is retained as
  the hold pose.

### `executor_navigation`

- Base twist is always zero.
- A local transverse square is generated around the insertion endpoint in the
  plane normal to the insertion direction.
- The executor uses the existing coordinated waypoint tracking controller.
- The observer target blends an executor-relative observation offset with the
  insertion endpoint ROI.
- Inter-arm avoidance remains observer-only, preserving executor priority.

### Terminal states

- `complete`: all local executor waypoints achieved;
- `clearance_violation`: configured primitive clearance falls below threshold;
- `phase_timeout`: a phase exceeds its configured maximum step count;
- `invalid_plan`: frame/path validation fails before simulation starts.

Terminal failure states emit zero base and tendon commands and are included in
command metadata and saved artifacts.

## Controller Boundaries

Introduce focused components:

- `EngineNavigationSpec`: validated YAML values;
- `EngineNavigationPlan`: resolved world-frame entry, insertion path, base
  poses, local executor path, and observer ROI;
- `resolve_engine_navigation_plan(...)`: pure configuration/geometry function;
- `MobileBasePoseController`: proportional position/orientation error to
  bounded world twist;
- `StagedEngineNavigationController`: phase state machine and command
  composition.

The staged controller delegates executor/observer motion rather than copying
whole-body Jacobian logic.

## Application Composition

`SimulationApplication` performs these steps:

1. load the mobile dual-arm assembly and engine scene;
2. inject engine visual assets and enabled primitive control geometry;
3. resolve the engine navigation plan;
4. construct the staged controller;
5. register normal recorder, tendon diagnostics, viewer, and completion hooks;
6. run the existing generic `SimulationLoop`.

The application must reject `engine_navigation` when:

- no engine scene is selected;
- the assembly has no executor or no observer;
- the base is fixed;
- the mobile-base free joint is absent;
- the named region or path cannot be resolved.

## Collision Scope

First-version collision behavior is deliberately limited:

- primitive clearance queries provide controller diagnostics and termination;
- the authored insertion path is assumed to be calibrated and safe;
- the visual mesh is not treated as a control-query surface;
- no RRT, trajectory optimization, mesh SDF, or online replanning is added.

Before claiming general engine navigation, the temporary `debug_box_1` must be
replaced or supplemented with calibrated primitive geometry around the
insertion corridor.

## Configuration Defaults and Tuning

- `pre_entry_standoff_m`: start at 0.05 m; increase for safer alignment.
- `insertion_waypoint_spacing_m`: start at 0.02 m; reduce for smoother base
  motion.
- base position gain: start at 1.5 s^-1.
- base orientation gain: start at 2.0 s^-1.
- base speed limits remain owned by the assembly config.
- base position tolerance: 5 mm.
- base orientation tolerance: 0.035 rad, approximately 2 degrees.
- executor waypoint tolerance: 3 mm.
- local square radius: 10 mm, safely below the arm's lateral workspace.
- local path samples: 40.
- existing executor/observer tendon lead remains 0.5 mm.

`controller_dt_s` remains 0.02 s and MuJoCo uses 20 substeps at 0.001 s.
The initial scenario uses `max_steps: 8000`. Base approach and insertion each
receive explicit phase timeouts derived from path length, configured speed, and
a two-times settling margin; executor navigation receives its own fixed timeout.
This prevents a long translation from being mistaken for controller failure.

## Observability

Every command records:

- current stage and stage waypoint index;
- base target pose;
- base position and orientation error;
- insertion-path progress;
- executor navigation error;
- minimum primitive clearance;
- completion or failure reason.

Artifacts add stage arrays and a base-path plot while retaining existing
tracking and tendon diagnostics.

## Validation Strategy

Automated validation to add during implementation:

- engine annotation frame resolution;
- entry/path consistency validation;
- path resampling;
- base target pose construction from mount geometry;
- mobile-base pose-controller limits and convergence direction;
- deterministic phase transitions;
- zero arm commands during base phases;
- zero base command during executor phase;
- clearance and timeout terminal behavior;
- scenario parsing and application composition.

Manual validation commands will be documented, but are not run automatically:

```powershell
python scripts/run_scenario.py configs/scenarios/dual_engine_navigation.yaml
```

Success criteria:

- all phases complete before their timeouts;
- base remains fixed during executor navigation;
- executor completes the local path;
- observer remains enabled without changing executor tracking behavior;
- no configured primitive clearance violation occurs;
- artifacts clearly show phase transitions and target/error histories.

## Out of Scope

- automatic planning from the engine triangle mesh;
- full collision-mesh signed-distance queries;
- simultaneous base/arm whole-body motion during local inspection;
- physical contact or cleaning;
- autonomous recovery and replanning after a blocked path;
- moving the engine model or recalibrating its mesh frame.
