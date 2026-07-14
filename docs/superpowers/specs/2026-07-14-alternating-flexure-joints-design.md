# Alternating Single-Axis Flexure Joints Design

## Goal

Make the dual-arm MuJoCo topology match the first-order real mechanism: each
arm has twelve rigid subsections, and each subsection has one proximal flexure
hinge rather than a co-located X/Y hinge pair.

## Confirmed Physical Topology

- Both executor and observer contain twelve rigid subsections.
- The first subsection is hinged to the base and rotates about its local Y axis,
  producing bending in the local X direction.
- The second subsection rotates about its local X axis, producing bending in
  the local Y direction.
- The axes continue Y, X, Y, X across all twelve subsections.
- Executor and observer use the same axis sequence.
- A hinge axis is expressed in the subsection's local material frame and moves
  with upstream bodies.

## Configuration

Add a required shared physical-topology field to `dual_robot`:

```yaml
flexure_joint_axis_pattern:
  - [0.0, 1.0, 0.0]
  - [1.0, 0.0, 0.0]
```

The loader validates that the pattern is non-empty, every axis is a finite
three-vector, every vector has non-zero length, and every axis is normalized.
The builder chooses `pattern[(global_link_index - 1) % len(pattern)]`.

## Generated MJCF

For each arm and link, generate one hinge named after its dominant cardinal
axis. Odd global links use `<joint ..._y axis="0 1 0">`; even global links use
`<joint ..._x axis="1 0 0">`. Preserve the existing pivot, stiffness,
damping, armature, range, spring reference, rigid link transforms, tendon
sites, mass, visuals, and actuators.

The two committed source XML assets are updated to match the generator. Files
under `output/generated/` are historical runtime products and are not edited.

## Scope Exclusions

This phase does not change flexure pivot placement, stiffness calibration,
visual deformation, collision, tendon pretension, gravity, attachment mass,
PCC equations, or control gains.

## Validation Contract

Builder tests inspect both freshly generated and committed XML. Each arm must
contain exactly twelve flexure joints, ordered Y/X for global links 1 through
12, with no second orthogonal joint on any link. Codex will write these tests
but will not run them under the user's no-automatic-verification constraint.
