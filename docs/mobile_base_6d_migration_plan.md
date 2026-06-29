# Mobile Base 6D Migration Plan

## Goal

Make `6D mobile base + tendon-driven continuum arm` the primary architecture, while preserving the existing fixed-base baseline as the identity/locked special case.

## Migration Strategy

1. Keep the tendon-driven single-arm PCC chain as the local arm kernel.
2. Add a world/base/mount layer around that kernel.
3. Route all baseline CLI entry points through the new architecture.
4. Preserve baseline behavior by defaulting the mobile base to identity and locked semantics.

## In Scope

- Add a reusable mobile-base arm context for local/world transforms.
- Wire `configs/main_config.yaml` and backend configs into the new mobile-base context.
- Rebuild the baseline commands on top of the new architecture:
  - `view-pcc`
  - `view-motor-chain`
  - `run-tracking`
  - `view-mujoco`
  - `debug-mujoco-tendons`
  - `run-mujoco-tracking`
  - `run-mujoco-navigation`
  - `run-mujoco-wiping`
- Keep current base behavior compatible through identity defaults.

## Out Of Scope

- Full mobile-base whole-body IK.
- Dynamic mobile-base actuation inside MuJoCo.
- Replacing the existing tendon/PCC solvers.

## Execution Steps

1. Add a shared mobile-base arm context that can convert points and poses between local arm and world frames.
2. Add mobile-base config discovery to the CLI and backend config flow.
3. Update the pure-PCC viewers and tracking flow to render/report through the world/base/mount context.
4. Update MuJoCo runtime/model path resolution so tracking and viewer paths also use the mobile-base wrapper.
5. Update docs and README so the new primary architecture and command baseline are described consistently.
6. Commit the migration in logical git checkpoints.

## Current Compatibility Rule

If `base_pose` is identity and the base is effectively locked, all existing baseline commands should behave the same as the previous fixed-base architecture, but now under the new `base + mount + arm` structure.

## Migration Status

Integrated in this round:

- Shared `MobileBaseArmContext` for local/world pose and point transforms.
- `configs/main_config.yaml`, `configs/pcc.yaml`, and `configs/mujoco.yaml` wiring for `mobile_base_config`.
- `view-pcc`, `view-motor-chain`, and `run-tracking` world-frame visualization/reporting on top of the local PCC kernel.
- `view-mujoco`, `debug-mujoco-tendons`, and `run-mujoco-tracking` runtime XML resolution through the mobile-base wrapper.
- Existing `run-mujoco-navigation` and `run-mujoco-wiping` scene builders kept compatible with the same wrapper path.

Mergeable now:

- Static 6D base pose + tendon-driven continuum arm as the primary single-arm architecture.
- Fixed-base baseline as the identity/locked special case.
- Baseline CLI entry points using a shared mobile-base config path.

Still intentionally deferred:

- Dynamic mobile-base actuation inside MuJoCo.
- Whole-body IK that simultaneously commands base and arm.
- Reworking the non-default `pcc_command` fallback paths in the more advanced MuJoCo task controllers into a full whole-body formulation.
