"""Three-window manual control application for the dual MuJoCo system."""

from __future__ import annotations

from pathlib import Path

from continuum_sim.application import SimulationApplication
from continuum_sim.application.hook_factory import observer_camera_attachment_config
from continuum_sim.runtime.observer_camera_hooks import MujocoObserverCameraFeedbackHook
from continuum_sim.runtime.viewer_hooks import _configure_mujoco_viewer
from continuum_sim.system.types import RobotSystemState
from continuum_sim.visualization.mujoco_system_debug_viewer import (
    MujocoSystemDebugViewer,
)


class ManualControlWindows:
    """Own the passive MuJoCo and observer-camera windows beside the controls."""

    def __init__(self, backend, *, control_dt_s: float, camera_fps: float = 20.0) -> None:
        if camera_fps <= 0.0:
            raise ValueError("camera_fps must be positive.")
        self.backend = backend
        self.control_dt_s = float(control_dt_s)
        self.camera_fps = float(camera_fps)
        self._viewer = None
        self._camera_hook = None
        self._active = False
        self._last_state: RobotSystemState | None = None

    def start(self, state: RobotSystemState) -> None:
        import mujoco.viewer

        self._last_state = state
        self._viewer = mujoco.viewer.launch_passive(
            self.backend.physics.model,
            self.backend.physics.data,
        )
        _configure_mujoco_viewer(self._viewer, self.backend.config)
        self._viewer.sync()

        camera_attachment = observer_camera_attachment_config(self.backend.assembly)
        if camera_attachment is not None and camera_attachment.camera is not None:
            control_hz = 1.0 / self.control_dt_s
            stride = max(1, round(control_hz / self.camera_fps))
            self._camera_hook = MujocoObserverCameraFeedbackHook(
                self.backend,
                camera_name=camera_attachment.camera.name,
                intrinsics=camera_attachment.camera.intrinsics,
                show_window=True,
                stride=stride,
                video_output_paths=None,
            )
            self._camera_hook.on_reset(state)
            self._camera_hook.enrich_state(state)
        self._active = True

    def update(self, state: RobotSystemState) -> None:
        self._last_state = state
        if not self._active:
            return
        if self._viewer is not None and self._viewer.is_running():
            self._viewer.sync()
        if self._camera_hook is not None:
            self._camera_hook.enrich_state(state)

    def close(self) -> None:
        self._active = False
        state = self._last_state
        if self._camera_hook is not None and state is not None:
            self._camera_hook.on_finish(state)
        self._camera_hook = None
        if self._viewer is not None:
            self._viewer.close()
        self._viewer = None


def run_manual_control(
    scenario_path: str | Path,
    *,
    camera_fps: float = 20.0,
    curvature_step_1_per_m: float = 0.5,
) -> None:
    """Compose a scenario backend and run all three manual-control windows."""

    application = SimulationApplication.from_yaml(scenario_path)
    backend = application.loop.backend
    windows = ManualControlWindows(
        backend,
        control_dt_s=application.config.runtime.controller_dt_s,
        camera_fps=camera_fps,
    )
    viewer = MujocoSystemDebugViewer(
        backend,
        control_dt_s=application.config.runtime.controller_dt_s,
        n_substeps=application.config.runtime.n_substeps,
        state_update_callback=windows.update,
        curvature_step_1_per_m=curvature_step_1_per_m,
    )
    windows.start(viewer.state)
    try:
        viewer.show()
    finally:
        windows.close()
        viewer.close()


__all__ = ["ManualControlWindows", "run_manual_control"]
