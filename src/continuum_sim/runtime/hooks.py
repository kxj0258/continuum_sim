"""Compatibility exports for runtime hooks.

New code should prefer the focused modules in this package:
`recording_hooks`, `observer_camera_hooks`, `video_hooks`,
`diagnostic_hooks`, `live_panel_hooks`, `viewer_hooks`, and
`completion_hooks`.
"""

from __future__ import annotations

from continuum_sim.runtime.completion_hooks import ControllerCompletionHook
from continuum_sim.runtime.diagnostic_hooks import TendonDiagnosticHook
from continuum_sim.runtime.live_panel_hooks import (
    LiveDiagnosticsPanelHook,
    LiveTendonPanelHook,
    LiveWipingForcePanelHook,
)
from continuum_sim.runtime.observer_camera_hooks import (
    MujocoObserverCameraFeedbackHook,
)
from continuum_sim.runtime.recording_hooks import (
    MujocoReplayRecorderHook,
    StateRecorderHook,
)
from continuum_sim.runtime.video_hooks import MujocoLiveVideoRecorderHook
from continuum_sim.runtime.viewer_hooks import (
    MatplotlibSystemViewerHook,
    MujocoViewerHook,
    _TrackingOverlayState,
    _configure_mujoco_viewer,
    _metadata_path,
    _metadata_paths,
    _metadata_point,
    _sample_overlay_points,
    _split_target_history,
)

__all__ = [
    "ControllerCompletionHook",
    "LiveDiagnosticsPanelHook",
    "LiveTendonPanelHook",
    "LiveWipingForcePanelHook",
    "MatplotlibSystemViewerHook",
    "MujocoLiveVideoRecorderHook",
    "MujocoObserverCameraFeedbackHook",
    "MujocoReplayRecorderHook",
    "MujocoViewerHook",
    "StateRecorderHook",
    "TendonDiagnosticHook",
    "_TrackingOverlayState",
    "_configure_mujoco_viewer",
    "_metadata_path",
    "_metadata_paths",
    "_metadata_point",
    "_sample_overlay_points",
    "_split_target_history",
]
