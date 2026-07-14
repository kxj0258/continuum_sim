# PCC–MuJoCo Physical Parameter Unification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give every PCC task a 5 mm tendon-routing radius and make the current distributed-link MuJoCo model and reduced PCC dynamics share explicit segment mass and bending-rigidity semantics.

**Architecture:** Extend the existing segment parameter object with optional physical properties while preserving old YAML compatibility. Keep tendon routing, collision radius, segment mass, and continuum rigidity as separate concepts; derive MuJoCo link mass and hinge stiffness from segment properties and derive PCC curvature stiffness from the same continuum rigidity.

**Tech Stack:** Python dataclasses, NumPy, YAML, MJCF XML generation, Markdown documentation.

## Global Constraints

- Do not run tests, validation, lint, format, build, install, viewer, simulation, demo, or long-running commands.
- Do not run `pip install -e .`, `pytest`, `python scripts/verify_feagine_install.py`, or `python scripts/inspect_feagine_scene.py`.
- Do not edit committed or runtime-generated MJCF files; the user will regenerate and validate manually.
- Do not create a Git commit unless the user explicitly asks.

---

### Task 1: Separate tendon routing from physical segment properties

**Files:**
- Modify: `src/continuum_sim/model/robot_params.py`
- Modify: `src/continuum_sim/model/dual_arm_robot.py`
- Modify: `src/continuum_sim/model/robot_assembly.py`

**Interfaces:**
- Produces: optional `SegmentParams.collision_radius`, `SegmentParams.mass`, and `SegmentParams.bending_stiffness` values.
- Preserves: existing positional construction with `length`, `tendon_radius`, and optional `tendon_angles_deg`.

- [ ] **Step 1: Extend `SegmentParams` with compatible optional fields**

```python
@dataclass(frozen=True)
class SegmentParams:
    length: float
    tendon_radius: float
    tendon_angles_deg: tuple[float, ...] = (0.0, 120.0, 240.0)
    collision_radius: float | None = None
    mass: float | None = None
    bending_stiffness: float | None = None

    @property
    def effective_collision_radius(self) -> float:
        return self.tendon_radius if self.collision_radius is None else self.collision_radius
```

- [ ] **Step 2: Load dual-arm physical fields without changing old-file behavior**

Construct each segment with:

```python
collision_radius=(
    None
    if segment_raw.get("collision_radius") is None
    else float(segment_raw["collision_radius"])
),
mass=None if segment_raw.get("mass") is None else float(segment_raw["mass"]),
bending_stiffness=(
    None
    if segment_raw.get("bending_stiffness") is None
    else float(segment_raw["bending_stiffness"])
),
```

- [ ] **Step 3: Accept equivalent optional spatial-arm keys**

The spatial-arm loader accepts `collision_radius_m`, `mass_kg`, and
`bending_stiffness_n_m2`, leaving all three as `None` when omitted.

### Task 2: Unify all current robot parameter values

**Files:**
- Modify: `configs/robots/spatial_arm_executor.yaml`
- Modify: `configs/robots/spatial_arm_observer.yaml`
- Modify: `configs/robots/dual_arm_3seg.yaml`
- Modify: `configs/robot_3seg.yaml`
- Modify: `configs/dynamics/pcc_reduced.yaml`

**Interfaces:**
- Produces: `tendon_radius = radial_offset = 0.005 m` for all PCC task inputs.
- Produces: `collision_radius = 0.00625 m`, `mass = 0.00989929 kg`, and `EI = 0.0002 N m^2` for the current distributed-link robot.

- [ ] **Step 1: Change executor and observer PCC routing values**

Change every `tendon_radius_m: 0.00625` and `radial_offset_m: 0.00625` to
`0.005` in both spatial-arm files.

- [ ] **Step 2: Change dual-arm routing while retaining collision geometry**

For all six dual-arm segments use:

```yaml
tendon_radius: 0.005
collision_radius: 0.00625
mass: 0.00989929
bending_stiffness: 0.0002
```

Change all 18 `physical_tendons[].radial_offset` values to `0.005`.

- [ ] **Step 3: Update legacy PCC metadata and reduced dynamics**

In `configs/robot_3seg.yaml`, retain its existing 5 mm tendon routing and set
each segment to:

```yaml
mass: 0.00989929
bending_stiffness: 0.0002
```

In `configs/dynamics/pcc_reduced.yaml` use:

```yaml
segment_masses_kg: [0.00989929, 0.00989929, 0.00989929]
bending_stiffness: [0.0002, 0.0002, 0.0002]
```

### Task 3: Derive MuJoCo properties from segment physics

**Files:**
- Modify: `scripts/build_mujoco_dual_arm_model.py`

**Interfaces:**
- Consumes: `SegmentParams.effective_collision_radius`, `.mass`, `.bending_stiffness`.
- Produces: explicit per-link geom mass and per-link hinge stiffness in generated MJCF.

- [ ] **Step 1: Keep capsule radius independent from tendon routing**

Replace capsule size use of `segment.tendon_radius` with:

```python
"size": _format_float(segment.effective_collision_radius),
```

- [ ] **Step 2: Write explicit per-link mass when configured**

Before creating the collision geom:

```python
if segment.mass is not None:
    geom_attrs["mass"] = _format_float(segment.mass / float(config.links_per_segment))
```

This yields approximately `0.0024748225 kg` per link for the current robot.

- [ ] **Step 3: Derive hinge stiffness from continuum rigidity**

Build both x/y joint attribute dictionaries with:

```python
if segment.bending_stiffness is not None:
    joint_attrs["stiffness"] = _format_float(
        segment.bending_stiffness / link_length
    )
```

The current `0.0002 / 0.01` calculation preserves `0.02 N m/rad`. Missing
segment rigidity continues to inherit `joints.hinge.stiffness` from the MJCF
default.

### Task 4: Give PCC curvature stiffness the same physical meaning

**Files:**
- Modify: `src/continuum_sim/dynamics/pcc_dynamics.py`

**Interfaces:**
- Consumes: `PCCDynamicsConfig.bending_stiffness` as continuum `EI` in `N m^2`.
- Produces: diagonal curvature-coordinate stiffness `EI * segment_length`.

- [ ] **Step 1: Multiply continuum rigidity by segment length**

Update the two bending entries per segment to:

```python
bending_stiffness = (
    config.bending_stiffness[segment_index]
    * params.segments[segment_index].length
)
diagonal[base : base + 3] = (
    bending_stiffness,
    bending_stiffness,
    config.axial_stiffness[segment_index],
)
```

- [ ] **Step 2: Clarify units in docstrings**

Document that `PCCDynamicsConfig.bending_stiffness` contains `EI` values and
that `stiffness_matrix()` converts them for curvature coordinates.

### Task 5: Document parameter semantics and endpoint placement

**Files:**
- Modify: `docs/configuration_reference.md`

**Interfaces:**
- Documents all public YAML fields and the unchanged endpoint convention.

- [ ] **Step 1: Document segment physical fields**

Document:

- `tendon_radius[_m]`: tendon centerline offset, current value 5 mm;
- `collision_radius[_m]`: MuJoCo capsule radius, current value 6.25 mm;
- `mass[_kg]`: total segment mass, divided among generated links;
- `bending_stiffness[_n_m2]`: continuum `EI`.

- [ ] **Step 2: Document conversion formulas**

Include `k_joint = EI / link_length` and `K_kappa = EI * segment_length`, with
the current numerical examples `0.02 N m/rad` and `8e-6` respectively.

- [ ] **Step 3: Document endpoint site placement**

State that `executor_tip`/`observer_tip` use `pos="0 0 0.01"` in the final-link
frame. The 10 mm offset reaches the distal end of the final link and does not
extend the 120 mm arm to 130 mm.

## Manual Validation For The User

Codex must not run these commands. After reviewing the edits, the user may run:

```powershell
python scripts/build_mujoco_dual_arm_model.py --config configs/mujoco_dual.yaml --output output/generated/dual_model_check.xml --mobile-base-output output/generated/dual_model_mobile_check.xml
python scripts/run_scenario.py configs/scenarios/single_mujoco_tracking.yaml
python scripts/run_scenario.py configs/scenarios/dual_mujoco_tracking.yaml
python scripts/run_scenario.py configs/scenarios/single_engine_tracking.yaml
python scripts/run_scenario.py configs/scenarios/dual_engine_tracking.yaml
```

