# Engine Exploration Path Design

## Goal

Add a reusable engine-relative exploration path to the engine scene model and
MuJoCo preview. The initial path follows the measured nozzle axis from outer
arc center P1 to inner arc center P2. This work is visualization and scene
configuration only; it does not connect the path to either arm controller.

## Coordinate Definition

The measured CAD coordinates are in millimeters and map directly to the current
engine-local axes. Store them in meters:

- P1, the arm entry point: `[0.442, 1.58169, 1.74693]`
- P2, the initial exploration target: `[0.442, 0.37281, 2.07857]`
- P1 to P2 unit direction: `[0.0, -0.96436878, 0.26456163]`
- Path length: approximately `1.25354535 m`
- Inclination: approximately `15.3409 degrees` from engine-local `-Y` toward
  engine-local `+Z`

Both the exploration path and entry region use `frame: engine`. Their points
and directions must be transformed by the configured engine position and
quaternion when displayed in the world.

## Configuration

Add an optional top-level `exploration_paths` list:

```yaml
exploration_paths:
  - name: nozzle_axis_entry
    type: polyline
    frame: engine
    enabled: true
    points_m:
      - [0.442, 1.58169, 1.74693]
      - [0.442, 0.37281, 2.07857]
    radius_m: 0.008
    rgba: [0.1, 1.0, 0.3, 0.8]
```

The polyline accepts two or more points so later planning code can replace the
initial straight axis with a sampled or generated route.

Update `engine_cleaning_nozzle_collision.yaml` so `regions` contains only:

```yaml
regions:
  entry_port:
    type: circular_port
    frame: engine
    center_m: [0.442, 1.58169, 1.74693]
    normal: [0.0, -0.96436878, 0.26456163]
    radius_m: 0.045
```

Existing scene files without `exploration_paths` or region `frame` remain
valid. Existing regions default to `frame: world`.

## Scene Model

Introduce an `ExplorationPathConfig` with:

- `name`
- `type`
- `frame`
- `enabled`
- `points_m`
- `radius_m`
- `rgba`

Initially support only `type: polyline` and frames `engine` and `world`.
Validation requires a non-empty name, at least two finite 3D points, positive
radius, valid RGBA values, and no zero-length adjacent segment.

Expose loaded paths through `EngineSceneConfig.exploration_paths`. Code may
construct or replace these immutable configuration values without rewriting
the YAML loader or preview geometry logic.

Add `frame` to `EngineRegionConfig`, defaulting to `world`. Engine-frame region
positions and normals use the same transform convention as exploration paths.
Normals rotate but do not translate or scale.

## MuJoCo Preview

Render each enabled polyline as non-colliding capsule segments:

- `contype: 0`
- `conaffinity: 0`
- configured radius and color

Render a start marker at the first point and an end marker at the last point
using distinct colors so path direction is visible. Transform engine-frame
points through the engine pose before emitting MJCF. World-frame paths are
emitted unchanged.

Keep this separate from `primitive_collision_geoms`; exploration geometry must
not be interpreted as physical collision.

Add a preview switch that allows exploration paths to be shown or hidden while
preserving the existing default preview behavior.

## Tests

Add focused tests for:

- loading and validating a polyline exploration path;
- backward compatibility when the section is absent;
- engine-frame point transformation under translation and rotation;
- engine-frame entry region position and normal transformation;
- generated MJCF capsule segments and endpoint markers;
- the nozzle configuration values for P1, P2, and the normalized direction;
- rejection of malformed points and zero-length segments.

Run the focused engine scene and preview tests, the full test suite, the asset
checker, and the MuJoCo headless scene load.

## Scope Boundaries

This change does not:

- execute the exploration path;
- connect either continuum arm;
- modify the M5 surface cleaning path;
- modify the M6 cleaning controller;
- add visual servoing or collision avoidance;
- enable the existing primitive collision hints.
