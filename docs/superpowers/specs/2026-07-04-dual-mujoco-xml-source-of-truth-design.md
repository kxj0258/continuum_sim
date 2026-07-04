# Dual MuJoCo XML Source-of-Truth Design

## Goal

Make `configs/mujoco_dual.yaml` and its referenced robot/mobile-base YAML files the reproducible source of truth for both committed dual-arm MJCF files, while raising all tendon actuator force limits to ±30 N.

## Current problem

`scripts/build_mujoco_dual_arm_model.py` generates only the unwrapped dual-arm XML. Scenario files consume the separately committed `_mobile_base.xml`, which is currently produced only as a runtime side effect. Several behavior-relevant values also disagree:

- YAML declares tendon actuator control limiting while MJCF disables it because controls are absolute tendon lengths.
- YAML declares follower collision while both arm collision geoms are disabled by the mesh manifest.
- World-frame marker sites exist in committed XML but are absent from the generator and YAML.
- Robot actuation metadata still declares a 20 N maximum tension.

## Configuration

`configs/mujoco_dual.yaml` will:

- set the global tendon actuator force range to `[-30.0, 30.0]`;
- set `actuators.tendon_position.ctrllimited` to `false`;
- retain `ctrlrange_m` as the software-side relative tendon command range;
- set `model.follower_collision` to `false`;
- declare `mobile_base_xml_path`;
- configure optional world-frame marker sites under `visuals.world_frame`.

`configs/robots/dual_arm_3seg.yaml` will synchronize executor, observer, and aggregate `max_tension` metadata to `30.0`.

## Generator behavior

The dual-arm builder will generate the base XML first, then use the existing mobile-base wrapper to generate the configured mobile-base XML. `--output` derives a sibling `_mobile_base.xml`; `--mobile-base-output` may override it. The CLI prints both paths.

Actuator `ctrllimited` will come from configuration instead of a hardcoded value. World origin and RGB axis sites will be generated from typed visual configuration. Stable naming, tendon colors, hole-site routing, mesh placement, and cosmetic MJCF defaults remain deterministic generator policy.

## Compatibility and safety

Both executor and observer receive ±30 N. Position gain remains 40000 N/m. No joint, tendon routing, mesh, site, or mobile-base pose changes are introduced. Existing user changes to tracking speed and generated scenario XML files are preserved.

## Tests

Tests will cover configuration loading, all 18 actuator attributes, world-frame markers, mobile-base output derivation, wrapper creation, and agreement between committed base/mobile XML files and YAML. Tests are written but not run automatically.

