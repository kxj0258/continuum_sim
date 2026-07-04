# Dual MuJoCo XML Source-of-Truth Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make one command reproducibly generate both dual-arm MJCF assets from YAML with ±30 N tendon actuators.

**Architecture:** Extend the validated MuJoCo configuration with an optional mobile-base output and world-frame marker style. Update the existing dual-arm builder to emit the base model and then call the existing mobile-base wrapper.

**Tech Stack:** Python, dataclasses, PyYAML, ElementTree, pytest.

## Global Constraints

- Do not run tests, generators, simulation, lint, format, build, or install commands.
- Preserve user changes in both tracking scenario YAML files.
- Preserve user changes in generated scenario XML files.
- Do not commit unless explicitly requested.

---

### Task 1: Configuration contract

**Files:**
- Modify: `src/continuum_sim/config.py`
- Modify: `configs/mujoco_dual.yaml`
- Modify: `configs/robots/dual_arm_3seg.yaml`
- Test: `tests/test_robot_config.py`

- [ ] Add tests for `mobile_base_xml_path`, world-frame marker configuration, disabled MJCF control limiting, disabled follower collision, and ±30 N force limits.
- [ ] Add typed optional output and world-frame configuration with validation.
- [ ] Synchronize the dual MuJoCo and robot metadata YAML values.

### Task 2: Reproducible dual output

**Files:**
- Modify: `scripts/build_mujoco_dual_arm_model.py`
- Test: `tests/test_mujoco_dual_arm_model.py`

- [ ] Add tests that expect configured world-frame sites and both output paths.
- [ ] Generate actuator limiting attributes from YAML.
- [ ] Emit world-frame markers from configuration.
- [ ] Generate the mobile-base wrapper after the base XML and print both paths.

### Task 3: Committed assets and documentation

**Files:**
- Modify: `assets/mujoco/dual_three_segment_arm_tendon_with_visuals.xml`
- Modify: `assets/mujoco/dual_three_segment_arm_tendon_with_visuals_mobile_base.xml`
- Modify: `docs/dual_arm_mujoco_landing.md`
- Modify: `docs/configuration_reference.md`

- [ ] Change all 18 actuator force ranges to ±30 N without altering other MJCF structure.
- [ ] Document the authoritative YAML fields and one-command two-output workflow.
- [ ] Document that `ctrlrange_m` is a relative software limit while MJCF control limiting is disabled for absolute-length controls.
- [ ] Review only the source diff and leave verification commands for manual execution.

