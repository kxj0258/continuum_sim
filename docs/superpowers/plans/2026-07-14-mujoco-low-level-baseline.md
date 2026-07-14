# MuJoCo Low-Level Baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Centralize the restored MuJoCo tracking low-level parameters and apply them to eight MuJoCo task scenarios.

**Architecture:** A new YAML profile owns Cartesian servo, whole-body solver, singularity, and tendon-target settings. Scenario YAML files reference it while retaining only task-specific upper-level behavior and scheduling.

**Tech Stack:** YAML configuration, Markdown documentation

## Global Constraints

- Do not run tests, validation, lint, format, build, install, or simulation commands.
- Preserve every task-specific upper-level controller and dual observer mode.
- Preserve the dual MuJoCo tracking collision-avoidance activation threshold at 18 mm.

---

### Task 1: Create And Adopt The Baseline Profile

**Files:**
- Create: `configs/control/mujoco_tracking_low_level.yaml`
- Modify: eight scenario YAML files listed in the design specification

**Interfaces:**
- Consumes: `scenario.low_level_control_path` and `low_level_control` schema fields.
- Produces: one effective low-level parameter set for all eight scenarios.

- [ ] Create the profile from the complete effective `single_mujoco_tracking` parameter set.
- [ ] Point all eight scenarios to `../control/mujoco_tracking_low_level.yaml`.
- [ ] Remove duplicated low-level fields from the two MuJoCo tracking scenarios while retaining tracking schedule fields.

### Task 2: Synchronize Documentation

**Files:**
- Modify: `README.md`
- Modify: `docs/superpowers/specs/2026-07-14-dual-tracking-control-design.md`
- Modify: `docs/superpowers/plans/2026-07-14-dual-tracking-control.md`

**Interfaces:**
- Consumes: the final shared-profile structure.
- Produces: documentation that does not describe the tracking scenarios as self-contained profile exceptions.

- [ ] Document the baseline profile, its eight consumers, and unchanged task-level differences.
- [ ] Update the earlier dual-tracking design and plan so they do not contradict the shared-profile architecture.
- [ ] List manual scenario commands without executing them.
