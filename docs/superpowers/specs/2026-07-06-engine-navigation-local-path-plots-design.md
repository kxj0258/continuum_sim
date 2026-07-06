# Engine Navigation Local Path Plots Design

## Goal

Make circle, figure-eight, and square tracking quality readable without the
roughly metre-scale insertion route compressing centimetre-scale local paths.
Also align every plotted executor target with the actual tip position sampled
at the same control step.

## Existing Artifacts

The current artifact writer can produce:

- `trajectory.png`: world-frame executor target and arm-tip trajectories;
- `engine_navigation_base_path.png`: base target and actual base path;
- `tracking_error.png`: executor target error over target-bearing commands;
- one tendon-displacement plot per arm;
- `min_clearance_m.png` when finite clearance data exists;
- `whole_body_singularity.png` when condition-number data exists;
- contact/force plots for wiping tasks.

The world-frame trajectory remains useful for whole-run context, but it is not
the right scale for local engine maneuvers.

## Recording Contract

Whenever a command contains `executor_target_world`, the recorder also stores:

- the executor tip position from that same state;
- active engine local-path name and type;
- executor subphase (`path` or `rejoin`);
- active local-path center;
- insertion direction.

These arrays are target-sample aligned and therefore do not rely on slicing the
full state history by target count.

## Local Plot

For each non-empty local-path name, select samples whose subphase is `path`.
Build the same transverse basis used by planning from the insertion direction,
subtract the active path center, and project target and actual positions onto
that basis.

Save:

```text
engine_navigation_local_path_<safe-name>.png
```

Each image contains:

1. an equal-aspect local XY trajectory panel with target, actual, start, and
   end markers;
2. a same-sample Euclidean error panel in millimetres with mean, RMS, and
   maximum error annotations.

Rejoin samples are excluded so the axial return does not distort the local
shape plot.

## Existing Plot Correction

`trajectory.png` uses the new target-aligned actual-tip array when available.
Non-engine scenarios without the new array retain the existing fallback.

## Failure Handling

Missing, empty, malformed, or non-finite local context skips only the affected
local plot. Artifact generation continues for all other plots. File names are
sanitized to alphanumeric, dash, and underscore characters.

## Manual Validation

Suggested user commands:

```powershell
pytest tests/test_scenario_artifacts.py
python scripts/run_scenario.py configs/scenarios/dual_engine_navigation.yaml
```

Codex does not run tests, lint, formatting, builds, installation, or simulation
commands during implementation.
