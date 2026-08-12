"""Three-window manual control application for the dual MuJoCo system."""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from threading import RLock
from time import perf_counter

from continuum_sim.application import SimulationApplication
from continuum_sim.application.hook_factory import observer_camera_attachment_config
from continuum_sim.runtime.observer_camera_hooks import MujocoObserverCameraFeedbackHook
from continuum_sim.runtime.viewer_hooks import _configure_mujoco_viewer
from continuum_sim.system.types import RobotSystemState
from continuum_sim.utils.runtime_timing import RuntimeTimingReporter
from continuum_sim.visualization.mujoco_system_debug_viewer import (
    MujocoSystemDebugViewer,
)


class ManualControlWindows:
    """Own the passive MuJoCo and observer-camera windows beside the controls."""

    def __init__(
        self,
        backend,
        *,
        control_dt_s: float,
        viewer_fps: float = 15.0,
        camera_fps: float = 10.0,
        simulation_lock=None,
        clock=perf_counter,
        runtime_timing=None,
    ) -> None:
        if viewer_fps <= 0.0:
            raise ValueError("viewer_fps must be positive.")
        if camera_fps <= 0.0:
            raise ValueError("camera_fps must be positive.")
        self.backend = backend
        self.control_dt_s = float(control_dt_s)
        self.viewer_fps = float(viewer_fps)
        self.camera_fps = float(camera_fps)
        self._simulation_lock = RLock() if simulation_lock is None else simulation_lock
        self._clock = clock
        self.runtime_timing = runtime_timing
        self._viewer = None
        self._camera_hook = None
        self._active = False
        self._last_state: RobotSystemState | None = None
        self._next_viewer_update_s: float | None = None
        self._next_camera_update_s: float | None = None

    def start(self, state: RobotSystemState) -> None:
        import mujoco.viewer

        self._last_state = state
        self._viewer = mujoco.viewer.launch_passive(
            self.backend.physics.model,
            self.backend.physics.data,
        )
        _configure_mujoco_viewer(self._viewer, self.backend.config)
        with self._simulation_lock:
            self._viewer.sync()

        camera_attachment = observer_camera_attachment_config(self.backend.assembly)
        if camera_attachment is not None and camera_attachment.camera is not None:
            self._camera_hook = MujocoObserverCameraFeedbackHook(
                self.backend,
                camera_name=camera_attachment.camera.name,
                intrinsics=camera_attachment.camera.intrinsics,
                show_window=True,
                stride=1,
                video_output_paths=None,
                data_lock=self._simulation_lock,
                runtime_timing=self.runtime_timing,
            )
            self._camera_hook.on_reset(state)
            self._camera_hook.enrich_state(state)
        self._active = True
        now_s = self._clock()
        self._next_viewer_update_s = now_s + 1.0 / self.viewer_fps
        self._next_camera_update_s = now_s + 1.0 / self.camera_fps

    def update(self, state: RobotSystemState) -> None:
        self._last_state = state
        if not self._active:
            return
        timing = self.runtime_timing
        now_s = self._clock()
        if self._display_due("viewer", now_s, 1.0 / self.viewer_fps):
            with (
                nullcontext()
                if timing is None
                else timing.measure("viewer.sync")
            ):
                if self._viewer is not None and self._viewer.is_running():
                    with self._simulation_lock:
                        self._viewer.sync()
        if self._display_due("camera", now_s, 1.0 / self.camera_fps):
            with (
                nullcontext()
                if timing is None
                else timing.measure("camera.total")
            ):
                if self._camera_hook is not None:
                    self._camera_hook.enrich_state(state)

    def _display_due(self, name: str, now_s: float, interval_s: float) -> bool:
        attribute = f"_next_{name}_update_s"
        deadline_s = getattr(self, attribute)
        if deadline_s is None:
            setattr(self, attribute, now_s + interval_s)
            return True
        tolerance_s = 1.0e-12
        if now_s + tolerance_s < deadline_s:
            return False
        intervals = max(
            1,
            int((now_s + tolerance_s - deadline_s) // interval_s) + 1,
        )
        setattr(self, attribute, deadline_s + intervals * interval_s)
        return True

    def close(self) -> None:
        self._active = False
        self._next_viewer_update_s = None
        self._next_camera_update_s = None
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
    panel_fps: float = 15.0,
    viewer_fps: float = 15.0,
    camera_fps: float = 10.0,
    curvature_step_1_per_m: float = 0.5,
) -> None:
    """Compose a scenario backend and run all three manual-control windows."""

    application = SimulationApplication.from_yaml(scenario_path)
    backend = application.loop.backend
    simulation_lock = RLock()
    runtime_timing = RuntimeTimingReporter(report_interval_s=0.5)
    windows = ManualControlWindows(
        backend,
        control_dt_s=application.config.runtime.controller_dt_s,
        viewer_fps=viewer_fps,
        camera_fps=camera_fps,
        simulation_lock=simulation_lock,
        runtime_timing=runtime_timing,
    )
    viewer = MujocoSystemDebugViewer(
        backend,
        control_dt_s=application.config.runtime.controller_dt_s,
        n_substeps=application.config.runtime.n_substeps,
        state_update_callback=windows.update,
        curvature_step_1_per_m=curvature_step_1_per_m,
        panel_fps=panel_fps,
        simulation_lock=simulation_lock,
        runtime_timing=runtime_timing,
    )
    windows.start(viewer.state)
    runtime_timing.reset()
    print(
        "[manual-timing] enabled; summary interval=0.5 s; "
        "stage values are avg/max milliseconds",
        flush=True,
    )
    try:
        viewer.show()
    finally:
        viewer.close()
        windows.close()


__all__ = ["ManualControlWindows", "run_manual_control"]
