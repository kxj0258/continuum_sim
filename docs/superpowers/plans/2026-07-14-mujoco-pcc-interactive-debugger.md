# MuJoCo–PCC Interactive Debugger Implementation Plan

> **For agentic workers:** Implement inline in the current workspace. Do not
> dispatch subagents for this task.

**Goal:** Build a dual-arm interactive MuJoCo viewer that compares PCC and
MuJoCo tips under manual tendon commands and can explicitly export samples.

**Architecture:** A focused visualization module owns state comparison,
overlay geometry, diagnostic formatting, and CSV recording. A new script
composes it with the existing scenario loader and tendon-control panel. The
generic control panel gains only optional diagnostic-text support.

**Tech Stack:** Python, NumPy, MuJoCo passive viewer, Matplotlib widgets, CSV
standard library.

## Global Constraints

- Do not automatically run tests, validation, lint, format, build, install,
  simulation, demos, or a MuJoCo viewer.
- Do not add a YAML configuration switch.
- Do not overwrite unrelated dirty-worktree changes.
- Do not commit changes automatically.

---

### Task 1: Real-time comparison model

**Files:**
- Create: `src/continuum_sim/visualization/mujoco_pcc_debug.py`

**Interfaces:**
- Produces: `ArmPccMujocoComparison`, `compare_pcc_mujoco_state()`, and
  `format_pcc_mujoco_diagnostics()`.
- Consumes: `RobotAssemblyConfig`, `RobotSystemState`, `BendingSpaceModel`,
  `forward_kinematics`, and the MuJoCo mobile-base frame metadata.

- [ ] Define an immutable per-arm comparison record containing measured tendon
  values, bending state, PCC/MuJoCo centerlines and tips, frame-specific errors,
  and compatibility residual.
- [ ] Validate enabled arms and all array shapes before computing results.
- [ ] Transform PCC samples with `mujoco_mobile_base_frame_pose @ mount_pose`.
- [ ] Format compact per-arm millimetre readouts for the existing information
  panel.

### Task 2: MuJoCo overlay and explicit CSV recorder

**Files:**
- Modify: `src/continuum_sim/visualization/mujoco_pcc_debug.py`

**Interfaces:**
- Produces: `MujocoPccOverlay`, `PccMujocoSampleRecorder`, and
  `default_pcc_debug_csv_path()`.

- [ ] Draw colored spheres for both tips and capsules for centerlines and error
  vectors in `viewer.user_scn`.
- [ ] Clear only the dedicated user scene before redrawing and check `maxgeom`
  before every geometry append.
- [ ] Deduplicate samples by state object, retain Reset-separated sessions, and
  serialize stable CSV columns with the standard library.
- [ ] Create output directories only when `save_csv()` is called.

### Task 3: Optional diagnostic text in the existing panel

**Files:**
- Modify: `src/continuum_sim/visualization/system_tendon_debug.py`
- Modify: `src/continuum_sim/visualization/mujoco_system_debug_viewer.py`

**Interfaces:**
- `SystemTendonMonitorPanel.update(..., info_text: str | None = None)` replaces
  the default per-tendon text only when provided.
- `MujocoSystemDebugViewer(..., diagnostic_text_provider=...)` supplies that
  text while leaving all existing callers unchanged.

- [ ] Add the optional keyword without changing the existing default view.
- [ ] Use the provider on initialization, Step, Run, and Reset updates.
- [ ] Leave the current `state_update_callback` contract intact for MuJoCo
  synchronization and overlay drawing.

### Task 4: Dedicated manual diagnostic script

**Files:**
- Create: `scripts/debug_mujoco_pcc.py`

**Interfaces:**
- Consumes: a scenario YAML path and optional `--samples-per-segment` and
  `--output` arguments.
- Produces: an interactive MuJoCo window, tendon-control panel, and explicit
  `Save CSV` action.

- [ ] Load the application and reject any backend other than
  `MujocoSystemBackend`.
- [ ] Compose comparison, overlay, recorder, numeric-text provider, and viewer
  synchronization without starting the scenario tracking controller.
- [ ] Add the save button and display the saved path or save error in the panel.
- [ ] Keep the command blocked only while the operator has the UI open.

### Task 5: Documentation

**Files:**
- Modify: `README.md`
- Modify: `docs/debugging_guide.md`

- [ ] Add the dual-arm command and explain that it starts an interactive viewer.
- [ ] Document colors, coordinates, actual-vs-target tendon semantics,
  compatible/raw interpretation, steady-state procedure, and CSV behavior.
- [ ] Document that the MuJoCo tip site is 10 mm beyond the final link body
  origin and that this offset is intentionally included in the comparison.

### Task 6: User-run validation

Codex must not execute these commands. Recommend that the user run them after
reviewing the changes:

```powershell
python scripts/debug_mujoco_pcc.py configs/scenarios/dual_mujoco_tracking.yaml
python scripts/debug_mujoco_pcc.py configs/scenarios/single_mujoco_tracking.yaml
pytest tests/test_mujoco_system_debug_viewer.py
```

Expected manual behavior: the UI opens with both arms at zero target; Step and
Run update all overlay geometries and numeric values; compatible and raw modes
remain selectable; Save CSV writes a timestamped file only after being clicked.
