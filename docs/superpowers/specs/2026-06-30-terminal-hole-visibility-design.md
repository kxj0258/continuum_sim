# Terminal Hole and Visibility Controls

## Goal

Model the physical outlet holes on each arm's `segment_3_link_4` and provide YAML controls for hole-site and tendon visualization without changing tendon physics.

## Terminal Link Geometry

Add a `hole_pattern.terminal_link` section:

```yaml
terminal_link:
  segment_number: 3
  link_number: 4
  in_holes_from: link_even
  out_holes:
    - {id: hole_04, index: 3, z_m: 0.007}
    - {id: hole_08, index: 7, z_m: 0.007}
    - {id: hole_12, index: 11, z_m: 0.007}
```

The terminal link keeps all twelve inlet holes from the even-link template. Its outlet set contains only holes 04, 08, and 12, each at local z = 0.007 m. Outlet x/y coordinates come from the matching inlet hole, avoiding duplicated coordinates.

The loader must reject duplicate terminal outlet indices, unknown indices, mismatched ids, or an unsupported `in_holes_from` value.

## Visualization Configuration

Add:

```yaml
visualization:
  hole_display: routed
  show_tendons: true
```

`hole_display` supports:

- `none`: required routed sites remain in the XML with alpha zero; unused sites are omitted.
- `routed`: only sites referenced by a physical tendon are emitted visibly.
- `all`: all physically defined sites are emitted visibly. The terminal link still has only three outlet sites.

The default project configuration uses `routed`.

`show_tendons` is a visualization-only master switch:

- `true`: native MuJoCo spatial tendons retain their configured colors, and the viewer overlay may render when its existing overlay setting is enabled.
- `false`: native spatial tendon alpha is zero and the viewer overlay is skipped.

Spatial tendons, actuators, sensors, and all physics remain present in both states.

## Routing and Generation

For each arm, routed hole indices are derived from its `physical_tendons`:

- Base sites use every tendon hole index on that arm.
- A link uses tendon hole indices whose `path_segment_indices` include that link's segment.

The XML generator always emits every site referenced by a spatial tendon. Visibility affects alpha and omission of unused sites only.

The runtime tendon overlay uses the same terminal outlet override and honors `show_tendons`.

## Documentation and Validation

Update the dual-arm landing documentation with the terminal geometry and switch behavior.

No tests, builds, linters, formatters, installers, model generation, viewers, or simulations will be run automatically. Manual validation commands will be supplied after implementation.
