# Alternating Single-Axis Flexure Joints Implementation Plan

> **For agentic workers:** Execute inline in the current workspace. Do not
> dispatch subagents and do not run automatic verification.

**Goal:** Replace every co-located X/Y MuJoCo joint pair with the real Y/X
alternating single-axis flexure topology on both twelve-link arms.

**Architecture:** Store the shared axis pattern in the physical dual-arm robot
configuration, validate it in the dual-arm loader, and consume it in the MJCF
builder. Keep committed source MJCF assets synchronized with builder output.

**Tech Stack:** Python dataclasses, YAML, ElementTree MJCF generation, pytest
contract tests.

## Global Constraints

- Modify only code, configuration, XML assets, tests, and documentation.
- Do not run tests, validation, lint, format, build, install, simulation, or a
  MuJoCo viewer.
- Do not edit historical files under `output/generated/`.
- Do not modify stiffness, damping, tendon, mass, controller, or PCC behavior.

---

### Task 1: Write the topology contract test

**Files:**
- Modify: `tests/test_mujoco_dual_arm_model.py`

**Interfaces:**
- Consumes: generated and committed dual-arm MJCF.
- Produces: assertions for twelve single hinges per arm in Y/X order.

- [ ] Add a helper that maps global link numbers 1..12 to segment/link names.
- [ ] Assert odd links contain only `_y` with `axis="0 1 0"`.
- [ ] Assert even links contain only `_x` with `axis="1 0 0"`.
- [ ] Apply the same assertions to executor, observer, base XML, and mobile XML.
- [ ] Do not run the test; recommend it for manual execution.

### Task 2: Load the physical axis pattern

**Files:**
- Modify: `configs/robots/dual_arm_3seg.yaml`
- Modify: `src/continuum_sim/model/dual_arm_robot.py`

**Interfaces:**
- Produces: `DualArmRobotConfig.flexure_joint_axis_pattern` as normalized
  `tuple[tuple[float, float, float], ...]`.

- [ ] Add `[[0,1,0],[1,0,0]]` under `dual_robot`.
- [ ] Parse and validate non-empty finite non-zero three-vectors.
- [ ] Normalize vectors so MJCF generation is independent of YAML magnitude.

### Task 3: Generate one hinge per subsection

**Files:**
- Modify: `scripts/build_mujoco_dual_arm_model.py`

**Interfaces:**
- Consumes: `dual_robot.flexure_joint_axis_pattern` and global link index.
- Produces: one `_x`, `_y`, or `_bend` hinge per rigid subsection.

- [ ] Select the axis by global-link modulo pattern length.
- [ ] Name cardinal X/Y axes `_x`/`_y`; reserve `_bend` for non-cardinal future
  axes.
- [ ] Reuse `_joint_attrs()` so existing flexural stiffness remains unchanged.
- [ ] Remove unconditional creation of the second orthogonal hinge.

### Task 4: Synchronize committed source MJCF and docs

**Files:**
- Modify: `assets/mujoco/dual_three_segment_arm_tendon_with_visuals.xml`
- Modify: `assets/mujoco/dual_three_segment_arm_tendon_with_visuals_mobile_base.xml`
- Modify: `docs/configuration_reference.md`

- [ ] Remove `_x` on odd global links and `_y` on even global links for both
  arms in both XML assets.
- [ ] Document the shared local-axis pattern and one-hinge-per-link semantics.
- [ ] Leave `output/generated/` untouched for the user's next manual run.

### Task 5: User-run validation

Codex must not execute these commands. Recommend:

```powershell
pytest tests/test_mujoco_dual_arm_model.py
python scripts/build_mujoco_dual_arm_model.py --config configs/mujoco_dual.yaml
python scripts/debug_mujoco_pcc.py configs/scenarios/dual_mujoco_tracking.yaml
```
