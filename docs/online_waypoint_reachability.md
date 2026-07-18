# Online Waypoint Reachability

This feature scores the currently active waypoint during waypoint servoing. It is
intended to separate easy-to-follow waypoints from waypoints that are slow,
poorly aligned, actuator-limited, or poorly explained by the current model. When
the score stays below a threshold, the waypoint controller can automatically
advance to the next waypoint.

## Scope

Online reachability is used by waypoint-mode tracking controllers, including the
tracking controller used inside navigation and wiping waypoint phases. It does
not change timed trajectory interpolation.

MuJoCo scenarios normally inherit the shared defaults from
`configs/control/mujoco_tracking_low_level.yaml`:

```yaml
low_level_control:
  online_reachability:
    enabled: true
    auto_advance_enabled: true
    score_threshold: 0.3
    window_steps: 25
    min_steps_before_auto_advance: 50
    low_score_patience_steps: 25
    good_progress_mps: 0.001
    good_tendon_speed_ratio: 0.75
    good_alignment: 0.8
    bad_model_residual_mps: 0.005
```

Individual scenarios may still override any field under
`scenario.task.tracking_control.online_reachability`; scenario values take
precedence over the shared low-level profile.

`configs/scenarios/dual_mujoco_tracking.yaml` enables the live diagnostics panel
by default so the score can be watched during a run.

## Scores

The online evaluator separates geometric/control reachability from backend
execution:

```text
reachability_score = progress_component
                   * alignment_component
                   * model_component

execution_score = tendon_component

combined_score = reachability_score * execution_score
```

Automatic waypoint advance uses `reachability_score`, not `execution_score`.
This keeps slow MuJoCo actuator response from being treated as an unreachable
waypoint when the tip is still moving toward the target.

`progress_component` measures whether the tip error is decreasing over the
recent sample window. It compares recent error reduction speed against
`good_progress_mps`.

`alignment_component` measures whether the actual tip motion points toward the
active target. A value near 1 means motion is well aligned with the target
direction. Negative or sideways motion lowers this component.

`tendon_component` compares the measured executor tendon speed with the previous
commanded tendon-rate reference. A value below 1 means the backend actuator
response is lagging the command. It is reported as `execution_score`, but it is
not used by the automatic waypoint advance decision.

`model_component` uses the previous command metadata residual when available.
Large residuals lower this component because the current model/solver mapping is
not explaining the requested motion well. If no residual is available, this
component defaults to 1.

## Automatic Advance

The controller requests an automatic advance only when all of these are true:

```text
enabled
auto_advance_enabled
steps_on_waypoint >= min_steps_before_auto_advance
current_error > waypoint_tolerance_m
reachability_score < score_threshold
low score has persisted for low_score_patience_steps
```

When this happens, the waypoint scheduler advance reason is:

```text
online_reachability_low
```

The patience and minimum-step gates avoid skipping a waypoint because of startup
transients at the beginning of a servo segment.

## Live Diagnostics

`LiveDiagnosticsPanelHook` shows the online score in the context of the full
control stack. It also annotates events so slow waypoints can be identified
without post-processing:

```text
Gray background:
  approach waypoint samples.

Light red background:
  reachability_score is below the configured threshold.

Dotted vertical lines:
  waypoint index changes.

Red downward markers in the score panel:
  automatic advance was requested.

Dark tick markers at the top of the score panel:
  normal waypoint advance events.

Top status bar:
  current waypoint, reachability score, execution score, lowest reachability
  component, low-score patience counter, tip error, and tendon target error.
```

The two reachability-specific panels are:

```text
Online reachability score:
  reachability score
  progress_component
  alignment_component
  model_component
  execution_score
  combined_score
  threshold line at 0.3

Reachability drivers:
  progress_rate_mps, displayed as mm/s
  target_alignment
  tendon_speed_ratio
  model_residual_mps, displayed as mm/s
  auto_advance_requested
```

The score panel highlights the current lowest reachability component by making
that curve thicker. `execution_score` and `combined_score` are shown for context,
but they are not reachability bottlenecks. The drivers panel uses a split y-axis:
progress and model residual are shown in mm/s on the left axis, while alignment,
tendon ratio, and automatic advance are shown on the right axis as normalized
ratios or active flags.

Use these plots to determine why a waypoint is hard:

```text
Low progress, good alignment:
  moving in the right direction, but too slowly. This often points to actuator
  lag, conservative actuator limits, low task speed, or low gains.

Low alignment:
  actual motion is not aimed at the target. This often points to poor local
  Jacobian conditioning, competing tasks, contact/avoidance effects, or a target
  outside the locally reachable direction.

Low tendon ratio:
  commanded tendon speed is not being realized. This points to MuJoCo actuator
  response, actuator gain/force limits, tendon target policy, lead limits, or
  rate/displacement saturation. This lowers execution_score, not
  reachability_score.

Low model component:
  the requested Cartesian motion does not map cleanly through the current
  model/solver state. This points to model mismatch, bad conditioning, task
  conflicts, or local kinematic singularity.
```

## Saved Result Arrays

When the recorder is enabled, these arrays are written to `result.npz` alongside
the existing tracking arrays:

```text
online_reachability_score
online_reachability_execution_score
online_reachability_combined_score
online_reachability_progress_component
online_reachability_alignment_component
online_reachability_tendon_component
online_reachability_model_component
online_reachability_progress_rate_mps
online_reachability_target_alignment
online_reachability_tendon_speed_ratio
online_reachability_model_residual_mps
online_reachability_low_score_steps
online_reachability_auto_advance_requested
```

## Tuning Notes

Skipping is based on `reachability_score`. For more aggressive skipping, raise
`score_threshold`, lower
`min_steps_before_auto_advance`, or lower `low_score_patience_steps`.

For less aggressive skipping, lower `score_threshold`, raise
`min_steps_before_auto_advance`, or raise `low_score_patience_steps`.

If normal reachable points are being judged too harshly, adjust the normalization
constants first:

```text
good_progress_mps:
  lower this if the expected stable convergence speed is below 1 mm/s.

good_tendon_speed_ratio:
  lower this if the backend normally realizes less than 75% of the requested
  tendon speed even on successful points. This affects execution_score and
  diagnostics, not automatic waypoint advance.

good_alignment:
  lower this if curved or constrained local motion is expected and still useful.

bad_model_residual_mps:
  raise this if normal solver residuals are frequently above 5 mm/s but still
  lead to convergence.
```
