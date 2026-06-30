# Segment Terminal Routing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace delta-based outlet adjustments with per-segment, per-arm absolute terminal outlet definitions and align all tendon groups.

**Architecture:** `segment_terminal_links` becomes the only special link-end schema. Endpoint lookup merges non-exclusive segment 1/2 overrides with ordinary even-link outlets and returns only explicit outlets for exclusive segment 3.

**Tech Stack:** YAML, Python dataclasses

## Global Constraints

- Do not preserve the old `terminal_link` or `link_out_offsets` interfaces.
- Keep cumulative `path_segment_indices`.
- Do not run tests, builds, linters, formatters, installers, XML generation, viewers, or simulations.

---

### Task 1: Replace Hole Schema and Loader

**Files:**
- Modify: `configs/robots/dual_arm_hole_pattern.yaml`
- Modify: `src/continuum_sim/model/hole_pattern.py`

- [ ] Define all three segment terminal links by arm with absolute 7 mm outlets.
- [ ] Replace old data classes and parsers with `TendonSegmentTerminalLink`.
- [ ] Merge non-exclusive outlet overrides and enforce exclusive segment-3 outlets.

### Task 2: Align Physical Tendons

**Files:**
- Modify: `configs/robots/dual_arm_3seg.yaml`

- [ ] Set executor groups to holes 03/07/11, 01/05/09, and 04/08/12.
- [ ] Set observer groups to holes 01/05/09, 03/07/11, and 02/06/10.
- [ ] Synchronize segment and physical-tendon angles without changing cumulative paths.

### Task 3: Update Documentation

**Files:**
- Modify: `docs/dual_arm_mujoco_landing.md`

- [ ] Replace obsolete terminal and delta descriptions with the unified routing table.
