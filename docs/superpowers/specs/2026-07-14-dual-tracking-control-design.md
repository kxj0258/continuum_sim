# Dual MuJoCo Tracking Control Design

## Goal

Make the dual MuJoCo tracking scenario use the same proven low-level control
parameters as `single_mujoco_tracking`, while keeping independent upper-level
tasks:

- executor: time-parameterized Cartesian trajectory tracking;
- observer: inter-arm collision avoidance activated at 18 mm.

## Configuration

`dual_mujoco_tracking.yaml` stops loading the protected
`spatial_low_level.yaml` profile and instead references
`mujoco_tracking_low_level.yaml`. That profile contains the complete control
parameter set restored from `single_mujoco_tracking.yaml`. Because the tracking
control configuration is shared by the coordinated low-level controller, both
executor and observer use the same Cartesian-to-tendon solver settings and the
same MuJoCo tendon target mode.

The observer-specific task configuration remains independent:

- `observer_control_mode: collision_avoidance`;
- `minimum_distance_m: 0.010` remains unchanged;
- `influence_distance_m: 0.018`, so avoidance activates below 18 mm;
- `release_margin_m: 0.002`, so avoidance releases above 20 mm;
- `avoidance_gain: 1.2`;
- no avoidance-specific maximum speed.

## Collision-Avoidance Semantics

The observer-only avoidance task remains a soft velocity task. It is activated
when the closest centerline distance falls below 18 mm and commands separation
along the closest-point normal. Its requested speed is
`1.2 * max(0.018 - distance, 0)`. An unset avoidance speed limit means no
avoidance-specific clipping and does not fall back to the general Cartesian
target-speed limit. The executor and observer continue to be solved
independently, so observer avoidance does not consume executor tendon degrees
of freedom.

No hard-stop or executor-freeze behavior is added in this change. Therefore
18 mm is the activation threshold, not a mathematically hard minimum-distance
guarantee under actuator saturation, contact, or infeasible geometry.

## Files

- `configs/control/mujoco_tracking_low_level.yaml`: own the single-compatible
  MuJoCo low-level baseline.
- `configs/scenarios/dual_mujoco_tracking.yaml`: reference the baseline and set
  the observer avoidance activation distance to 18 mm.
- `README.md`: document the dual tracking task split, shared low-level behavior,
  and 18 mm activation-distance semantics.

## Manual Validation

No automated tests or simulations will be run by Codex. The user can manually
run:

```powershell
python scripts/run_scenario.py configs/scenarios/single_mujoco_tracking.yaml
python scripts/run_scenario.py configs/scenarios/dual_mujoco_tracking.yaml
```

The dual result should be inspected for executor tracking error, executor and
observer saturation scale, inter-arm distance, and collision-avoidance state.
