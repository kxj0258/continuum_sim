# Scenario Application Design

## Goal

Provide the original research capabilities through one scenario-driven
application API instead of restoring the old CLI, package facades, and
task-specific runtime loops.

## Primary API

```python
application = SimulationApplication.from_yaml(
    "configs/scenarios/single_mujoco_view.yaml"
)
result = application.run()
```

A thin script may call the same API:

```powershell
python scripts/run_scenario.py configs/scenarios/single_mujoco_view.yaml
```

## Scenario Model

```text
ScenarioConfig
├── assembly
├── backend: analytic | mujoco
├── scene: none | engine
├── task: idle | tracking | navigation | wiping
├── runtime
└── hooks
    ├── viewer
    ├── tendon_debug
    └── recorder
```

Assembly configuration determines single- or dual-arm operation. Tasks and
hooks cannot instantiate backends.

## Capability Mapping

- PCC preview: analytic backend plus recorded kinematic state.
- Tendon/shape diagnostics: direct tendon displacement to PCC shape and FK.
- Offline tracking: analytic backend plus coordinated tracking controller.
- MuJoCo view: MuJoCo backend plus zero controller and optional viewer hook.
- Tendon debug: tendon diagnostic hook.
- MuJoCo tracking: coordinated tracking controller with MuJoCo backend.
- Navigation: waypoint controller plus primitive clearance tasks.
- Wiping: waypoint/contact task provider on the same loop.
- Engine work: engine scene composition selected by scenario.

The motor transmission path is not part of the spatial system. Any future
motor study is an independent diagnostic.

## Boundaries

`ScenarioFactory` is the composition root. `SimulationApplication` owns the
loop lifecycle. Backends return `RobotSystemState`; controllers return
`RobotSystemCommand`; hooks observe but do not alter control.

## Asset Composition

When MJCF output moves to another directory, every pre-existing file reference
is resolved relative to the source MJCF and rewritten relative to the output.
Engine STL preparation is owned by the scene adapter, including MuJoCo's
200,000-face limit.

## Constraints

- World-frame base twist only.
- Direct tendon-length-rate control only.
- Single and dual systems share application/runtime code.
- No compatibility requirement for old Python APIs.
- No automatic tests, builds, lint, install, or simulation commands.

## Baseline Capability Migration

Scenario runs are reproducible experiments rather than transient viewers.
Every non-idle task enables a native artifact writer by default. One run
directory contains flattened named-system histories, summary metrics, input
configuration, the composed MJCF when applicable, static plots, and an
optional GIF replay.

Tracking, navigation, and wiping remain separate task policies on the same
loop. Navigation adds clearance diagnostics and violation termination.
Wiping adds approach/contact/retract phases and a configurable normal-contact
proxy; it does not reintroduce motor-space commands. Viewer overlays and
tendon diagnostics consume the same named state and command histories.

The old motor-chain viewer is intentionally not migrated because direct
tendon-length-rate control removed that actuator layer.
