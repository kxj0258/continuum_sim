# Segment Terminal Routing

## Goal

Define each segment's final-link outlet geometry per arm with absolute coordinates, while retaining cumulative tendon paths.

## Configuration

Replace `terminal_link` and `link_out_offsets` with `segment_terminal_links`. Each entry identifies one segment's link 4, uses the even-link inlet template, and supplies per-arm outlet overrides at local z = 0.007 m.

- Segment 1:
  - executor: holes 03, 07, 11
  - observer: holes 01, 05, 09
- Segment 2:
  - executor: holes 01, 05, 09
  - observer: holes 03, 07, 11
- Segment 3:
  - executor: holes 04, 08, 12
  - observer: holes 02, 06, 10

Segments 1 and 2 use `exclusive_out_holes: false`: listed holes receive z = 0.007 m while other outlets retain the normal even-link coordinates required by cumulative distal tendons.

Segment 3 uses `exclusive_out_holes: true`: only the three listed arm-specific outlets physically exist.

The loader rejects duplicate segment/link entries, unknown arms, duplicate outlet indices, unknown or mismatched ids, and missing arm entries.

## Tendon Routing

`path_segment_indices` remains cumulative:

- Segment 1 tendons: `[0]`
- Segment 2 tendons: `[0, 1]`
- Segment 3 tendons: `[0, 1, 2]`

Reassign each arm's three tendon groups to the configured holes. Synchronize `segments[].tendon_angles_deg`, `physical_tendons[].angle_deg`, and `physical_tendons[].hole_index`.

The new mapping supersedes the earlier uniform observer +30-degree rule.

## Consumers

The XML generator and viewer overlay continue to consume inlet/outlet endpoint accessors. The endpoint model resolves the matching segment-terminal entry and arm-specific outlet set before returning coordinates.

## Validation

No tests, builds, linters, formatters, installers, XML generation, viewers, or simulations will be run automatically.
