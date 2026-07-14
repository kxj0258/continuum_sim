# Dual MuJoCo Tracking Control Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make both dual tracking arms use the proven single tracking low-level parameters while using the common uncapped observer collision policy below 18 mm.

**Architecture:** Keep the existing independent executor trajectory intent and observer collision-avoidance intent. Configure both through the existing unified low-level controller by referencing the shared single-compatible `mujoco_tracking_low_level.yaml` profile.

**Tech Stack:** YAML scenario configuration, Markdown documentation

## Global Constraints

- Do not run tests, validation, lint, format, build, install, or simulation commands.
- Preserve `minimum_distance_m: 0.010` and `critical_distance_m: 0.008`.
- Set collision-avoidance activation distance to `influence_distance_m: 0.018`.

---

### Task 1: Configure dual tracking control

**Files:**
- Modify: `configs/scenarios/dual_mujoco_tracking.yaml`

**Interfaces:**
- Consumes: `ScenarioTrackingControlConfig` and `ScenarioObserverControlConfig` YAML fields.
- Produces: a self-contained dual tracking scenario with shared single-compatible low-level parameters.

- [ ] Replace the protected profile reference with `../control/mujoco_tracking_low_level.yaml`.
- [ ] Keep only task scheduling fields in the scenario `tracking_control` block.
- [ ] Set the common observer collision policy: 18 mm activation, 2 mm release margin, gain 1.2, and no avoidance-specific speed cap.

### Task 2: Document behavior

**Files:**
- Modify: `README.md`

**Interfaces:**
- Consumes: the final dual scenario configuration.
- Produces: user-facing documentation of executor tracking, observer avoidance, shared low-level control, and the 18 mm activation threshold.

- [ ] Update the dual tracking documentation without changing its execution command.
- [ ] Record manual validation commands but do not execute them.
