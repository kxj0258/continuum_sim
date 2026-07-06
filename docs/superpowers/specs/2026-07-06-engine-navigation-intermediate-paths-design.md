# Engine Navigation Intermediate Local Paths Design

## Goal

Execute local executor trajectories at one-third and two-thirds of the engine
insertion route, then execute the existing square at the endpoint. Every local
trajectory is retracted along the arm's local negative Z axis so it remains
inside the continuum arm workspace.

## Configuration

Keep the endpoint `local_path` for backward compatibility and add
`intermediate_local_paths`:

```yaml
intermediate_local_paths:
  - name: one_third_circle
    at_fraction: 0.3333333333
    type: transverse_circle
    radius_m: 0.010
    samples: 40
    axial_retraction_m: 0.010
  - name: two_thirds_figure_eight
    at_fraction: 0.6666666667
    type: transverse_figure_eight
    radius_m: 0.010
    samples: 60
    axial_retraction_m: 0.010
```

Intermediate fractions must be strictly between zero and one, unique, and
in ascending order. Names must be non-empty and unique. Radii and sample counts
must be positive; axial retraction must be finite and non-negative.

The existing endpoint `local_path` is treated as an event at fraction `1.0`.

## Planning

The insertion polyline is already resampled into base targets. Each requested
fraction selects the resampled target whose cumulative path fraction is
closest to the requested fraction. The resolved event stores:

- name and path type;
- requested fraction;
- insertion waypoint index;
- unretracted insertion-axis target;
- retracted local-path center;
- local executor waypoints.

For every event:

```text
center = insertion_target - axial_retraction_m * insertion_direction
```

Circle, figure-eight, and square paths are generated in the plane normal to
the insertion direction. They contain the configured number of samples and
close at their starting point.

## Control Sequence

The controller runs:

1. base approach;
2. base insertion until the one-third event;
3. base locked, executor circle;
4. executor returns to the unretracted insertion-axis target;
5. base insertion resumes until the two-thirds event;
6. base locked, executor figure-eight;
7. executor returns to the unretracted insertion-axis target;
8. base insertion resumes to the endpoint;
9. base locked, executor square;
10. complete.

The rejoin step is mandatory for intermediate events. It restores the straight
tip target assumed by subsequent base poses and avoids moving the base while
the arm remains bent. The endpoint event completes without an extra rejoin.

The existing `executor_navigation` phase remains the public phase name.
Metadata adds the active local-path name, type, event index, requested
fraction, and subphase (`path` or `rejoin`).

## Visualization

All resolved local paths are published as a collection in command metadata.
The MuJoCo viewer and live recording draw each path separately so there are no
false connector segments between the circle, figure-eight, and square.

The current active target and histories continue to use the existing dynamic
overlay behavior. Observer ROI follows the currently active retracted local
path center.

## Compatibility

If `intermediate_local_paths` is omitted, behavior remains the existing
endpoint-only square. The existing final `executor_waypoints_world` and
`observer_roi_world` plan fields continue to refer to the endpoint event for
compatibility with current code and tests.

## Manual Validation

Suggested commands for the user:

```powershell
pytest tests/test_engine_navigation.py tests/test_staged_engine_navigation.py tests/test_engine_navigation_overlay.py
python scripts/run_scenario.py configs/scenarios/dual_engine_navigation.yaml
```

No automated test, lint, format, build, install, or simulation command is run
by Codex.
