# Link Outlet Offsets and Observer Rotation

> Superseded by `2026-06-30-segment-terminal-routing-design.md`.

## Goal

Move the 0.5 mm outlet adjustment from shared odd/even templates to the two physical links where it applies, and rotate the observer arm's tendon layout by 30 degrees.

## Link-Specific Outlet Offsets

Restore the shared link templates to their normal outlet geometry:

- `link_odd` holes 01 and 07 return to their unadjusted outlet z.
- `link_even` holes 04 and 10 return to their unadjusted outlet z.
- Existing base-hole adjustments remain unchanged.

Add `hole_pattern.link_out_offsets`:

```yaml
link_out_offsets:
  - segment_number: 1
    link_number: 4
    delta_z_m: -0.0005
    holes:
      - {id: hole_01, index: 0}
      - {id: hole_07, index: 6}
  - segment_number: 2
    link_number: 4
    delta_z_m: -0.0005
    holes:
      - {id: hole_02, index: 1}
      - {id: hole_08, index: 7}
```

The loader validates positive segment/link numbers, known and matching hole ids/indices, and duplicate targets. The endpoint accessor applies matching deltas after selecting the shared odd/even template.

The explicit `terminal_link` outlet set has higher priority and does not receive link offsets.

## Observer Tendon Rotation

Keep executor configuration unchanged. Rotate observer geometry by +30 degrees:

- Segment tendon angles become `[30, 150, 270]`, `[90, 210, 330]`, and `[120, 240, 360]`.
- Every observer physical tendon's `angle_deg` increases by 30 degrees.
- Every observer physical tendon's `hole_index` increases by one modulo 12.

This keeps segment parameters, fallback geometry, explicit hole routing, generated spatial tendons, and viewer overlays consistent.

The observer terminal outlet set rotates with the route to holes 05, 09, and 01. The executor terminal outlet set remains holes 04, 08, and 12.

## Validation

No tests, builds, linters, formatters, installers, XML generation, viewers, or simulations will be run automatically.
