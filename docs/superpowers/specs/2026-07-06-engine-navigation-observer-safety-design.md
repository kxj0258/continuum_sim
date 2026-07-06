# Engine Navigation Local Tracking and Observer Safety Design

## Goal

Make local executor waypoint advancement selectable from YAML and make
observer-to-executor collision avoidance the observer arm's primary objective
without changing executor tracking commands or waypoint progression.

## Local Tracking Configuration

Add this engine-navigation section:

```yaml
local_tracking:
  advance_mode: tolerance
  advance_time_s: null
  advance_steps: null
```

`advance_mode` accepts `tolerance` or `time`. Tolerance mode uses the existing
scenario `task.waypoint_tolerance_m`. Time mode requires exactly one positive
`advance_time_s` or `advance_steps`. If seconds are selected, the scheduler
converts them to controller steps using `runtime.controller_dt_s`.

Only circle, figure-eight, and endpoint-square paths use the selected mode.
Intermediate `rejoin` trackers always use tolerance mode, because the base must
not resume insertion until the executor has actually returned to the insertion
axis.

## Observer Safety Configuration

Add:

```yaml
observer_control:
  position_gain: 3.0
  executor_offset_world_m: [0.0, -0.04, 0.02]
  roi_blend: 0.25
  inter_arm_influence_distance_m: 0.018
  inter_arm_safe_distance_m: 0.014
  inter_arm_critical_distance_m: 0.009
  inter_arm_release_margin_m: 0.002
  inter_arm_avoidance_gain: 6.0
  inter_arm_max_avoidance_speed_mps: 0.03
  centerline_samples_per_segment: 8
  observer_tracking_weight: 20.0
  observer_collision_weight: 250.0
  stop_all_on_critical_distance: false
```

Distances are centerline-to-centerline. Validation requires:

```text
0 < critical < safe < influence
```

The release margin must be non-negative. Gains, weights, speeds, and sampling
counts must be positive. ROI blend remains in `[0, 1]`.

## Safety Modes

The coordinated controller computes the nearest pair over sampled executor and
observer centerlines.

- `tracking`: distance is outside the influence zone; observer follows its
  observation target.
- `avoidance`: distance is inside the influence zone; observer tracking is
  removed and only observer tendons generate separating velocity.
- `critical_avoidance`: distance is at or below the critical distance;
  observer avoidance remains at its configured maximum while executor tracking
  and waypoint progression continue unchanged.

Avoidance uses hysteresis. Once active, it remains active until distance exceeds
`influence + release_margin`, reducing rapid switching around the influence
boundary.

The collision Jacobian remains observer-only. Executor motion cannot be used
to satisfy avoidance. Observer tracking, observer scene avoidance, and
inter-arm avoidance contain no executor columns.

The fixed-base whole-body solver enables per-arm singularity decoupling so an
observer collision task cannot indirectly scale executor tendon velocity
through global singularity protection.

`stop_all_on_critical_distance` remains an explicit opt-in emergency policy,
but is `false` for engine navigation. With that setting, no observer safety
mode freezes executor velocity, pauses its scheduler, zeros its command, or
terminates the staged task.

## Solver and Scene Integration

Add an explicit observer collision weight while preserving the old generic
collision weight as a compatibility fallback. Pass the engine scene query into
local trackers so engine primitive avoidance is active during local paths.

## Diagnostics

Command metadata exposes:

- minimum inter-arm centerline distance;
- safety mode;
- avoidance-active flag;
- critical-distance flag;
- optional stop-all flag;
- closest centerline indices.

These values support later plots and parameter tuning. A regression contract
compares executor tendon commands with observer avoidance enabled and disabled;
they must match for the same fixed-base executor state and target.

## Manual Validation

Suggested commands:

```powershell
pytest tests/test_engine_navigation.py tests/test_staged_engine_navigation.py tests/test_bending_space.py
python scripts/run_scenario.py configs/scenarios/dual_engine_navigation.yaml
```

Codex does not run tests, lint, format, build, installation, or simulation
commands during implementation.
