# Dual Observer Collision Policy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Apply one explicit, uncapped observer collision-avoidance policy to every dual-arm scenario.

**Architecture:** The coordinated controller owns the meaning of an optional avoidance speed limit. Scenario YAML files declare one common policy, while engine navigation maps its nested observer specification into the same coordinated-controller fields.

**Tech Stack:** Python dataclasses and controller composition, YAML configuration, Markdown documentation

## Global Constraints

- Do not run tests, validation, lint, format, build, install, or simulation commands.
- Preserve all executor task behavior and low-level profile selection.
- Preserve observer-only solving and soft-avoidance behavior.

---

### Task 1: Define Uncapped Avoidance Semantics

**Files:**
- Modify: `src/continuum_sim/control/coordinated_tracking.py`
- Modify: `src/continuum_sim/tasks/engine_navigation.py`
- Modify: `src/continuum_sim/control/staged_engine_navigation.py`

**Interfaces:**
- Consumes: `inter_arm_max_avoidance_speed_mps: float | None`.
- Produces: `None` means no avoidance-specific speed clipping; numeric values clip the separation speed.

- [ ] Remove fallback from the optional avoidance speed to general target speed.
- [ ] Make the engine-navigation nested avoidance speed optional and validate it only when numeric.
- [ ] Pass the nested engine-navigation avoidance limit through to the coordinated controller.

### Task 2: Align Every Dual Scenario

**Files:**
- Modify: all nine `configs/scenarios/dual_*.yaml` files.

**Interfaces:**
- Consumes: generic `observer_control` fields and engine-navigation nested observer fields.
- Produces: an explicit collision policy with 18 mm activation, 10 mm minimum diagnostic distance, 8 mm critical diagnostic distance, 2 mm release margin, gain 1.2, six samples per segment, weight 80, and no speed cap.

- [ ] Add or update the explicit generic observer policy in every dual scenario.
- [ ] Align the engine-navigation nested collision fields and remove its numeric avoidance limit.

### Task 3: Document Effective Behavior

**Files:**
- Modify: `README.md`
- Modify: relevant dual-tracking design documentation.

**Interfaces:**
- Consumes: the final controller and scenario semantics.
- Produces: documentation distinguishing activation, release, diagnostic thresholds, and uncapped soft avoidance.

- [ ] Document the common policy and formula.
- [ ] Record manual scenario commands without executing them.
