# Link Outlet Offsets and Observer Rotation Implementation Plan

> Superseded by `2026-06-30-segment-terminal-routing.md`.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Apply two 0.5 mm link-specific outlet adjustments and rotate all observer tendon geometry by 30 degrees.

**Architecture:** YAML remains the geometry source of truth. The hole-pattern loader validates segment/link outlet overrides, and its endpoint accessor applies them before the existing XML generator and overlay consume the coordinates.

**Tech Stack:** YAML, Python dataclasses

## Global Constraints

- Keep executor tendon geometry unchanged.
- Preserve existing base-hole adjustments.
- Terminal outlet geometry overrides ordinary link offsets.
- Do not run tests, builds, linters, formatters, installers, XML generation, viewers, or simulations.

---

### Task 1: Configure Link-Specific Offsets

**Files:**
- Modify: `configs/robots/dual_arm_hole_pattern.yaml`
- Modify: `src/continuum_sim/model/hole_pattern.py`

- [ ] Restore shared odd/even outlet values.
- [ ] Add the two `link_out_offsets` entries.
- [ ] Parse, validate, and apply offsets in non-terminal outlet endpoint lookup.

### Task 2: Rotate Observer Tendons

**Files:**
- Modify: `configs/robots/dual_arm_3seg.yaml`

- [ ] Add 30 degrees to observer segment tendon-angle sets.
- [ ] Add 30 degrees to all observer physical-tendon angles.
- [ ] Increment all observer hole indices modulo 12.
- [ ] Rotate observer terminal outlets to holes 05, 09, and 01.

### Task 3: Document Geometry

**Files:**
- Modify: `docs/dual_arm_mujoco_landing.md`

- [ ] Document link-specific outlet adjustments and observer-only rotation.
- [ ] Provide manual generation and visualization commands without executing them.
