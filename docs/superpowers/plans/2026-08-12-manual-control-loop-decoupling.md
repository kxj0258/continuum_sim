# Manual Control Loop Decoupling Plan

**Goal:** Keep MuJoCo control at 50 Hz while preventing Matplotlib controls,
the passive viewer, and observer-camera presentation from blocking it.

## Design

- Run `step()` from a monotonic fixed-rate worker thread; keep Matplotlib calls
  exclusively on the GUI thread.
- Protect live MuJoCo data and target/state handoff with one shared re-entrant
  lock. The UI consumes the latest completed state rather than every state.
- Make target callbacks data-only. Synchronize dirty widgets during the panel
  refresh with widget drawing and events disabled, then issue one panel redraw.
- Schedule panel, passive viewer, and camera by independent wall-clock
  deadlines (defaults: 15/15/10 Hz).
- Render camera frames from a copied `MjData` snapshot so the live-data lock is
  held only while copying dynamic arrays. Reuse one Matplotlib `AxesImage` and
  request asynchronous drawing without clearing axes or flushing events.

## Regression coverage

- Target updates do not call widget `set_val()` before UI synchronization.
- Batched widget updates run with `drawon=False` and `eventson=False`.
- Camera Matplotlib fallback creates one image artist and later calls
  `set_data()`.
- Viewer and camera wall-clock deadlines advance independently.
- Runtime timing remains safe when control and UI stages record concurrently.

Per repository policy, tests and the interactive simulation are not run unless
the user explicitly authorizes those operations.
