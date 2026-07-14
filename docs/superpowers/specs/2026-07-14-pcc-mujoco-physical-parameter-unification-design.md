# PCC–MuJoCo Physical Parameter Unification Design

## Objective

Unify the tendon-routing radius, segment mass, and bending stiffness used by
the PCC and current `dual_distributed_links` MuJoCo models without changing the
active MuJoCo arm shape, endpoint definition, or runtime model-generation
workflow.

No tests, validation commands, lint, formatting, builds, installs, viewers, or
simulations are to be run by Codex during this change.

## Scope

The change covers:

- all PCC task entry points using the executor and observer spatial-arm YAMLs;
- the dual-arm robot metadata used by the MuJoCo model generator;
- the shared reduced PCC dynamics configuration;
- the dual distributed-link MuJoCo generator;
- the configuration reference documentation.

Legacy standalone MuJoCo prototypes such as `mujoco.yaml` and
`mujoco_segment_2dof.yaml` retain their independent calibration values.

## Authoritative Physical Parameters

### Tendon routing

- Tendon centerline radius: `0.005 m`.
- `tendon_radius` and physical-tendon `radial_offset` represent only the tendon
  centerline offset from the backbone axis.
- MuJoCo collision-body radius remains `0.00625 m` and is represented by a new
  independent `collision_radius` parameter.

### Segment mass

The current MuJoCo capsule estimate is retained as the authority:

- link length: `0.01 m`;
- capsule radius: `0.00625 m`;
- density used by the current XML: `1100 kg/m^3`;
- mass per link: approximately `0.00247482 kg`;
- mass per four-link segment: `0.00989929 kg`.

The robot segment stores `0.00989929 kg`. The MuJoCo generator divides the
segment mass equally among its four links and writes explicit geom masses so
future density or collision-radius edits cannot silently change the dynamics.
The reduced PCC dynamics configuration uses the same per-segment mass.

### Bending stiffness

The current MuJoCo hinge stiffness is retained as the authority:

- hinge stiffness: `0.02 N m/rad`;
- link length: `0.01 m`;
- equivalent continuum rigidity: `EI = 0.0002 N m^2`.

Robot and PCC dynamics configuration store `EI`, not discrete hinge stiffness.
The MuJoCo generator computes:

```text
k_joint = EI / link_length
```

The reduced PCC curvature-coordinate stiffness matrix computes:

```text
K_kappa = EI * segment_length
```

For a `0.04 m` segment this gives `8e-6` in the curvature-coordinate stiffness
matrix. The global MuJoCo `joints.hinge.stiffness` remains a compatibility
fallback for robot configurations that do not provide segment `EI`.

Axial and torsional stiffness values are not remapped because the current
MuJoCo distributed-link arm has no equivalent axial or torsional joint degree
of freedom.

## Data Model And Compatibility

`SegmentParams` gains optional physical fields for collision radius, segment
mass, and continuum bending rigidity. Loaders accept the new fields from both
dual-arm and spatial-arm robot schemas. Missing fields preserve the prior
behavior:

- collision radius falls back to tendon radius;
- missing segment mass leaves the generator's previous density-based mass
  behavior available;
- missing segment bending rigidity leaves the global MuJoCo hinge stiffness as
  the fallback.

This preserves compatibility with older robot YAML files while allowing the
current robot to use one explicit physical definition.

## Configuration Changes

The executor and observer spatial-arm configurations change every segment
`tendon_radius_m` and every tendon `radial_offset_m` from `0.00625` to `0.005`.

The dual-arm robot configuration changes:

- segment `tendon_radius` to `0.005`;
- physical-tendon `radial_offset` to `0.005`;
- adds `collision_radius: 0.00625`;
- segment `mass` to `0.00989929`;
- segment `bending_stiffness` to `0.0002`.

The legacy PCC robot configuration already uses a `0.005 m` tendon radius. Its
mass and bending-rigidity metadata are updated to the same authoritative
values. The shared `pcc_reduced.yaml` dynamics values are updated likewise.

## MuJoCo Endpoint Definition

The endpoint definition is unchanged. The generator creates `executor_tip` and
`observer_tip` on the final link at:

```text
pos = [0, 0, link_length] = [0, 0, 0.01] m
```

The 10 mm local-Z offset places the site at the distal end of the final 10 mm
link. It is part of the 120 mm arm length and is not an additional 10 mm tool
offset beyond the arm.

## Files Expected To Change

- `configs/robots/spatial_arm_executor.yaml`
- `configs/robots/spatial_arm_observer.yaml`
- `configs/robots/dual_arm_3seg.yaml`
- `configs/robot_3seg.yaml`
- `configs/dynamics/pcc_reduced.yaml`
- `src/continuum_sim/model/robot_params.py`
- `src/continuum_sim/model/dual_arm_robot.py`
- `src/continuum_sim/model/robot_assembly.py`
- `src/continuum_sim/dynamics/pcc_dynamics.py`
- `scripts/build_mujoco_dual_arm_model.py`
- `docs/configuration_reference.md`

Committed and runtime-generated MJCF files are not edited. Their current
capsule radius, implicit mass estimate, hinge stiffness, and endpoint position
already match the target MuJoCo values.

## Risks

- Changing the PCC tendon radius changes the tendon-to-curvature estimate and
  therefore all PCC Jacobians and control commands.
- The reduced PCC dynamic controller will become substantially more compliant
  because its previous stiffness value was used without the required segment
  length conversion.
- Explicit link mass can expose small differences from MuJoCo's exact internal
  capsule-inertia calculation if rounded values are used.
- Old generated XML remains usable but will not show the new explicit mass
  attributes until the user manually regenerates it.

## Manual Validation

After implementation, the user should manually inspect or regenerate a model
to a temporary output path, then run the desired scenarios. Codex will not run
these commands automatically.

