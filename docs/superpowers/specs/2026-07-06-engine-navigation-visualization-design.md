# Engine Navigation Visualization Design

## Goal

Make the staged dual-arm engine-navigation task understandable while it is
running. The MuJoCo passive viewer and live MuJoCo video must show the complete
planned route, the active target, and relevant actual/history trails. All
visual elements must be independently configurable from YAML.

## Scope

The visualization covers the `engine_navigation` task:

- pre-entry target;
- mobile-base planned path;
- insertion planned path and insertion waypoints;
- executor local planned path;
- observer region of interest;
- current base or executor target;
- actual mobile-base history;
- actual executor-tip history;
- visited target history.

The executor local path center is not the fully extended insertion endpoint.
`task.engine_navigation.local_path.axial_retraction_m` moves that center back
toward the arm base along the negative insertion axis. The observer ROI follows
the retracted center so planning, control, and visualization stay consistent.

The existing generic target marker, target trail, tip trail, tendon paths, and
segment endpoint overlays remain compatible with tracking, navigation, and
wiping scenarios.

## Architecture

Use runtime MuJoCo overlays rather than injecting task markers into generated
MJCF.

The staged controller owns the resolved navigation plan, so it publishes a
small visualization payload in command metadata. The viewer and video hooks
consume the same payload and keep their own bounded actual-history buffers.
The shared overlay renderer draws both live viewer and recorded-video scenes.

This keeps planning data in the task/controller layer, rendering policy in the
runtime layer, and appearance in the MuJoCo YAML configuration.

## Metadata Contract

Each engine-navigation command exposes:

- `engine_navigation_pre_entry_target_m`;
- `engine_navigation_base_path_m`;
- `engine_navigation_insertion_path_m`;
- `engine_navigation_executor_path_m`;
- `engine_navigation_observer_roi_m`;
- `engine_navigation_active_target_m`;
- `engine_navigation_active_target_kind`, either `base` or `executor`;
- existing `engine_navigation_phase` and `base_target_position_m`;
- existing `executor_target_world` during executor navigation.

The arrays describe immutable plan geometry. They are included in command
metadata so hooks do not need to depend directly on task/controller classes.

## Rendering

The shared MuJoCo overlay renderer draws:

- pre-entry target as a sphere;
- mobile-base planned route as a line of spheres;
- insertion route as a line of spheres;
- insertion waypoints as larger spheres;
- executor local route as a line of spheres;
- observer ROI as a sphere;
- active target as a prominent sphere whose color identifies base or executor;
- base actual history as a bounded trail;
- executor actual history using the existing tip trail;
- active-target history as a bounded trail.

Static routes are sampled by a configurable stride. Dynamic histories use the
existing global history limit and stride. Overlay geometry capacity is
respected: drawing stops cleanly when the MuJoCo scene has no free geometry
slots.

## Configuration

Add an `engine_navigation` subsection below `viewer.overlays` in
`configs/mujoco_dual.yaml`. It provides:

- a master `enabled` switch;
- independent switches for planned routes, waypoints, ROI, current target,
  base history, executor history, and target history;
- radius and RGBA values for every visual category;
- route and waypoint strides.

Defaults preserve existing behavior when the subsection is absent. Engine
navigation overlays default to disabled in the generic parser and are enabled
explicitly by the dual MuJoCo configuration.

## Compatibility and Failure Handling

Non-engine tasks do not publish the new metadata, so they retain their current
rendering. Missing or malformed optional visualization metadata is ignored
rather than affecting control. Configuration values are validated with the
same positive-number, positive-integer, boolean, and RGBA helpers used by the
existing overlay settings.

`axial_retraction_m` must be finite and non-negative. A value of zero restores
the original endpoint-centered behavior.

## Manual Validation

The user can run:

```powershell
python scripts/run_scenario.py configs/scenarios/dual_engine_navigation.yaml
```

Expected visual sequence:

1. approach phase shows the pre-entry and active base targets;
2. insertion phase advances the active base target along the insertion route;
3. executor phase switches the active target to the executor local route;
4. base, executor, and target histories remain visible according to YAML
   limits and switches;
5. saved live MuJoCo GIF contains the same overlays as the viewer.

No automated test, build, lint, format, or simulation command is part of the
implementation workflow unless explicitly requested.
