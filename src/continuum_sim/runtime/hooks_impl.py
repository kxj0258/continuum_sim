"""Compatibility exports for legacy runtime hook imports."""

from __future__ import annotations

from continuum_sim.runtime.hook_utils import (
    metadata_path as _metadata_path,
    metadata_paths as _metadata_paths,
    metadata_point as _metadata_point,
    sample_overlay_points as _sample_overlay_points,
    split_target_history as _split_target_history,
)
from continuum_sim.runtime.mujoco_overlay_utils import (
    _TrackingOverlayState,
    _draw_mujoco_tracking_overlay,
    _draw_tracking_overlay_scene,
    _update_follow_camera,
)
from continuum_sim.runtime.viewer_hooks import (
    MatplotlibSystemViewerHook,
    MujocoViewerHook,
    _configure_mujoco_viewer,
    _sleep_until_simulation_time,
)

__all__ = [
    "MatplotlibSystemViewerHook",
    "MujocoViewerHook",
    "_TrackingOverlayState",
    "_configure_mujoco_viewer",
    "_draw_mujoco_tracking_overlay",
    "_draw_tracking_overlay_scene",
    "_metadata_path",
    "_metadata_paths",
    "_metadata_point",
    "_sample_overlay_points",
    "_sleep_until_simulation_time",
    "_split_target_history",
    "_update_follow_camera",
]
