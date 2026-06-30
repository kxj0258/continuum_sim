# Terminal Hole and Visibility Controls Implementation Plan

> Terminal geometry steps are superseded by `2026-06-30-segment-terminal-routing.md`; visibility steps remain applicable.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the three physical terminal outlet holes and YAML-controlled hole/tendon visualization.

**Architecture:** The hole-pattern model owns terminal endpoint geometry and visualization policy. The XML generator derives routed hole indices from each arm's physical tendons, while the runtime overlay treats the YAML tendon switch as a visualization-only master switch.

**Tech Stack:** Python dataclasses, YAML, `xml.etree.ElementTree`, MuJoCo user-scene overlays

## Global Constraints

- `segment_3_link_4` keeps all inlet holes from `link_even`.
- Executor terminal outlets are holes 04, 08, and 12; observer terminal outlets are holes 05, 09, and 01 after its 30-degree route rotation. All use z = 0.007 m.
- Hole display modes are `none`, `routed`, and `all`, defaulting to `routed`.
- Hiding tendons must not remove spatial tendons, actuators, sensors, or physics.
- Do not run tests, builds, linters, formatters, installers, model generation, viewers, or simulations.

---

### Task 1: Model YAML Geometry and Visualization

**Files:**
- Modify: `configs/robots/dual_arm_hole_pattern.yaml`
- Modify: `src/continuum_sim/model/hole_pattern.py`

**Interfaces:**
- Consumes: `hole_pattern.terminal_link` and `hole_pattern.visualization`
- Produces: terminal in/out endpoint accessors and visualization settings

- [ ] Add `visualization.hole_display`, `visualization.show_tendons`, and the three terminal outlet definitions.
- [ ] Parse and validate display modes, booleans, terminal location, template name, ids, indices, and outlet z values.
- [ ] Expose separate inlet/outlet endpoint lookup so terminal outlets do not inherit the normal even-link outlet coordinates.

### Task 2: Generate Routed or Hidden Sites

**Files:**
- Modify: `scripts/build_mujoco_dual_arm_model.py`

**Interfaces:**
- Consumes: arm physical-tendon routes and hole-pattern endpoint/visualization settings
- Produces: required XML sites with mode-dependent visibility

- [ ] Derive routed base/link hole-index sets from `PhysicalTendonPath`.
- [ ] Emit only routed sites for `routed` and `none`; emit all physically defined sites for `all`.
- [ ] Set site alpha to zero in `none` while retaining every tendon-referenced site.
- [ ] Apply the terminal inlet/outlet split and reject routes that reference an absent terminal outlet.
- [ ] Set native spatial tendon alpha to zero when `show_tendons` is false without removing physical elements.

### Task 3: Align Viewer Overlay

**Files:**
- Modify: `src/continuum_sim/visualization/mujoco_tendon_path_overlay.py`
- Modify: `src/continuum_sim/runtime/mujoco_runtime_utils.py`

**Interfaces:**
- Consumes: terminal endpoint accessors and `show_tendons`
- Produces: correct terminal path points and master overlay suppression

- [ ] Pass segment/link identity into endpoint lookup and use the terminal outlet override.
- [ ] Skip dual-arm tendon overlay drawing when `show_tendons` is false.

### Task 4: Document Behavior

**Files:**
- Modify: `docs/dual_arm_mujoco_landing.md`

- [ ] Document terminal outlet geometry, hole-display modes, tendon visibility precedence, and preserved physics.
- [ ] Hand off manual model-generation and viewer commands without executing them.
