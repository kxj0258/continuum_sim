# MuJoCo Tendon Debug Migration Design

## Goal

Restore the former MuJoCo tendon debugging workflow within the scenario-based
application architecture. Both normal MuJoCo tasks and a standalone debug
entry point must expose commanded tendon targets, measured tendon displacement,
and actuator force for every enabled arm.

## Architecture

`RobotSystemState` is the shared diagnostic boundary. Each `ArmSystemState`
reports the current tendon target and actuator force alongside the existing
measured displacement and velocity. `MujocoSystemBackend` obtains these values
from its tendon-rate integrators and the underlying MuJoCo state. The analytic
backend supplies its current integrator target and zero force so consumers do
not require backend-specific branches.

A system-level visualization module converts named arm states into flat,
labelled monitor data. The scenario `LiveTendonPanelHook` and the standalone
debug viewer use the same monitor panel.

## Task Monitor

The live panel remains explicitly controlled by scenario YAML:

```yaml
hooks:
  show_live_tendon_panel: true
  live_tendon_panel_stride: 5
```

Relevant MuJoCo scenarios enable it by default. The panel displays:

- target and measured tendon displacement in millimetres;
- target error per tendon;
- actuator force in newtons;
- stable `executor:1` / `observer:1` style labels;
- current simulation time and per-arm saturation information.

Single-arm and dual-arm layouts are derived from the named system state rather
than hard-coded tendon offsets.

## Standalone Debug Workflow

The new entry point is:

```powershell
python scripts/debug_mujoco.py configs/scenarios/dual_mujoco_tracking.yaml
```

It loads the complete scenario composition, requires a MuJoCo backend, and
opens the MuJoCo passive viewer plus a Matplotlib control/monitor window. The
controls include per-tendon target sliders, reset, zero, single-step,
run/pause, and the useful named commands from the former baseline.

Slider targets are approached through `RobotSystemCommand` tendon rates,
clipped by the configured arm rate limits. The debug UI therefore uses the new
system backend instead of bypassing it with raw MuJoCo controls.

## Error Handling

- Standalone debug rejects analytic scenarios with a clear message.
- State vectors must match the tendon count of their named arm.
- Closed GUI windows stop interactive updates cleanly.
- An unavailable interactive Matplotlib backend is reported without starting a
  long-running loop.

## Tests and Verification

Tests cover state validation, MuJoCo diagnostic propagation, named monitor-data
flattening, target-to-rate conversion, hook updates, and CLI construction.
Per project instruction, Codex writes the tests but does not execute tests,
lint, format, build, installation, viewers, demos, or simulations.
