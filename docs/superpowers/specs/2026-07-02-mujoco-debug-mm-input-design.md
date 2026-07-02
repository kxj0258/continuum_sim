# MuJoCo Debug Millimetre Input Design

## Goal

Allow every tendon target in the standalone MuJoCo debug interface to be set
either by slider or by an exact numeric text input, with all UI values shown in
millimetres.

## Interaction

- Each tendon row contains one slider and one text box.
- Slider limits and displayed values use millimetres.
- Dragging a slider updates its text box immediately.
- Submitting a text box with Enter updates the matching slider and target.
- Text input is clipped to the configured tendon displacement range.
- Invalid or non-finite input restores the previous valid value.
- Editing a target does not advance simulation; `Step` and `Run` retain their
  existing behavior.
- Reset, Zero, and named presets synchronize both controls.

## Data Boundary

The visualization layer converts millimetres to metres before writing
`MujocoSystemDebugViewer.targets`. The backend, system state, actuator
configuration, and control commands remain SI-native and unchanged.

## Tests

Focused tests cover millimetre/metre conversion, clipping, invalid input, and
slider/text synchronization. Per project instruction, Codex will not execute
tests, lint, formatting, builds, installation, viewers, or simulations.
