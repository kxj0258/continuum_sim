# MuJoCo–PCC Interactive Debugger Design

## Goal

Add a dedicated manual diagnostic entry point that loads a scenario-composed
MuJoCo dual-arm model, lets the operator command all tendons, and compares the
PCC-predicted tip with the corresponding MuJoCo tip site in real time.

## Decisions

- Add a new `scripts/debug_mujoco_pcc.py` entry point instead of expanding the
  responsibilities of `scripts/debug_mujoco.py`.
- Compute PCC kinematics from each arm's current measured MuJoCo tendon
  displacement, not from its command target.
- Display both world-frame coordinates and mount-frame error. World coordinates
  drive MuJoCo rendering; mount-frame error isolates arm-model mismatch from
  base and mount placement.
- Preserve the existing `compatible` and `raw tendon` control modes. Compatible
  mode remains the default; raw mode is explicitly treated as a model-boundary
  experiment.
- Render the PCC centerline and tip, the MuJoCo measured centerline and tip site,
  and a line joining the two tips for each enabled arm.
- Keep samples in memory and write a CSV only after the operator clicks an
  explicit save button.
- Do not add YAML configuration. Marker style and sampling density are
  diagnostic-code constants so all MuJoCo scenarios behave consistently.

## Data Flow

For every state update and every enabled arm:

1. Read `ArmSystemState.tendon_displacement_m` from the MuJoCo backend.
2. Build the configured `BendingSpaceModel` and estimate its bending state.
3. Convert the bending state to PCC `q`, then run `forward_kinematics`.
4. Compose the MuJoCo mobile-base site pose with the configured arm mount pose.
5. Transform the PCC centerline and tip from mount coordinates into world
   coordinates.
6. Compare the transformed PCC tip with `ArmSystemState.tip_pose_world`, which
   is sourced from `executor_tip` or `observer_tip` by
   `MujocoSystemBackend.get_system_state()`.
7. Render both representations and publish numeric diagnostics in the existing
   Matplotlib tendon-control window.

The MuJoCo XML currently places both `executor_tip` and `observer_tip` at
`pos="0 0 0.01"` in the final link body. That body origin is the proximal end
of a 10 mm link whose collision geometry spans local Z from 0 to 10 mm, so the
site represents the link's distal end; it is not automatically an extra 10 mm
beyond the 120 mm arm. The comparison deliberately uses the reported site
position without adding or subtracting an offset.

## UI

- Reuse the existing two-arm tendon sliders, numeric inputs, named targets,
  Reset, Zero, Step, Run/Pause, and compatible/raw selector.
- Add `Save CSV` in the unused control area.
- In the MuJoCo window:
  - PCC centerline and tip: purple;
  - MuJoCo centerline and tip: cyan;
  - PCC-to-MuJoCo tip error: red;
  - executor and observer use different brightness while preserving semantics.
- In the text panel, show for each arm:
  - PCC tip XYZ in world coordinates;
  - MuJoCo tip XYZ in world coordinates;
  - PCC minus MuJoCo XYZ in mount coordinates;
  - Euclidean tip error;
  - tendon compatibility-residual norm.

## Recording

Each displayed state contributes one CSV row per arm. Rows contain a session
number, state time, arm name, control mode, nine current tendon displacements,
PCC and MuJoCo world XYZ, world and mount XYZ errors, error norm, and
compatibility-residual norm. A time rollback caused by Reset increments the
session number instead of discarding earlier samples.

CSV files are created only after an explicit click and default to
`output/diagnostics/mujoco_pcc_manual_<timestamp>.csv`.

## Error Handling

- Reject non-MuJoCo scenarios before opening a viewer.
- Reject states whose enabled-arm names do not match the assembly.
- Reject non-finite or incorrectly shaped tendon and pose values with the arm
  name in the error.
- Do not silently fall back from the MuJoCo mobile-base frame to the software
  base pose in this interactive tool; a missing frame indicates a diagnostic
  setup error.
- Do not hide overlay-capacity exhaustion; raise a clear error advising the
  operator to reduce centerline sampling.

## Documentation and Manual Validation

Document the command, color legend, coordinate conventions, control modes,
settling procedure, CSV fields, and interpretation risks. Codex must not launch
the viewer or run tests; the user performs all validation manually.
