# Equivalent Flexure Stiffness Implementation Plan

> **For agentic workers:** Implement inline in the current workspace. Do not dispatch subagents.

**Goal:** Preserve the configured segment-level `EI/L` bending stiffness after replacing co-located X/Y hinges with alternating single-axis hinges.

**Architecture:** The dual-arm MJCF builder counts parallel hinge axes within each segment and converts segment flexural rigidity to per-joint rotational stiffness using `k_joint = n_axis * EI / L`. Tests assert the resulting `0.01 N m/rad` value in generated and committed MJCF.

**Tech Stack:** Python, `xml.etree.ElementTree`, YAML, MJCF, pytest.

## Global Constraints

- Modify only code, tests, configuration documentation, and canonical source MJCF.
- Do not run tests, lint, format, build, install, viewer, or simulation commands.
- Preserve damping, masses, tendon routing, control gains, and output history.

---

### Task 1: Lock the stiffness contract in tests

**Files:**
- Modify: `tests/test_mujoco_dual_arm_model.py`

- [ ] Extend the existing alternating-flexure assertion so every hinge has `stiffness="0.01"`.
- [ ] Manual validation command, not run by Codex: `pytest tests/test_mujoco_dual_arm_model.py`.

### Task 2: Generate axis-count-aware hinge stiffness

**Files:**
- Modify: `scripts/build_mujoco_dual_arm_model.py`

- [ ] Build the segment's complete axis sequence before creating its links.
- [ ] Count axes that are parallel or antiparallel to the current hinge axis.
- [ ] Set `k_joint = n_axis * segment.bending_stiffness / segment.length`.
- [ ] Keep the joint unchanged when `bending_stiffness` is absent.

### Task 3: Synchronize source MJCF and documentation

**Files:**
- Modify: `assets/mujoco/dual_three_segment_arm_tendon_with_visuals.xml`
- Modify: `assets/mujoco/dual_three_segment_arm_tendon_with_visuals_mobile_base.xml`
- Modify: `docs/configuration_reference.md`

- [ ] Change all 24 canonical flexure hinge stiffness values from `0.02` to `0.01` in each model.
- [ ] Document the segment-equivalent stiffness equation and its effect on control response.
- [ ] Leave runtime-generated and historical output files unchanged.

### Task 4: Manual validation handoff

- [ ] Manual generator command, not run by Codex: `python scripts/build_mujoco_dual_arm_model.py --config configs/mujoco_dual.yaml`.
- [ ] Manual focused test, not run by Codex: `pytest tests/test_mujoco_dual_arm_model.py`.
- [ ] Manual scenario command, not run by Codex: `python scripts/run_scenario.py configs/scenarios/single_mujoco_tracking.yaml`.
