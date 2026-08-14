"""Manual curvature and tendon control applications for the MuJoCo system."""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from threading import RLock
from time import perf_counter

from continuum_sim.application import SimulationApplication
from continuum_sim.runtime.concurrency import AsyncLinePrinter, TimeRateGate
from continuum_sim.runtime.mujoco_state_copy import copy_mujoco_dynamic_state
from continuum_sim.runtime.viewer_hooks import _configure_mujoco_viewer
from continuum_sim.system.types import RobotSystemState
from continuum_sim.utils.runtime_timing import RuntimeTimingReporter
from continuum_sim.visualization.mujoco_system_debug_viewer import (
    MujocoSystemDebugViewer,
)


class ManualControlWindows:
    """Own only the passive MuJoCo viewer beside the manual-control windows."""

    def __init__(
        self,
        backend,
        *,
        viewer_fps: float = 15.0,
        simulation_lock=None,
        clock=perf_counter,
        runtime_timing=None,
    ) -> None:
        if viewer_fps <= 0.0:
            raise ValueError("viewer_fps must be positive.")
        self.backend = backend
        self.viewer_fps = float(viewer_fps)
        self._simulation_lock = RLock() if simulation_lock is None else simulation_lock
        self._clock = clock
        self.runtime_timing = runtime_timing
        self._mujoco = None
        self._viewer = None
        self._viewer_data = None
        self._active = False
        self._last_state: RobotSystemState | None = None
        self._viewer_gate = TimeRateGate(
            1.0 / self.viewer_fps,
            clock=self._clock,
        )

    def start(self, state: RobotSystemState) -> None:
        import mujoco
        import mujoco.viewer

        self._last_state = state
        self._mujoco = mujoco
        self._viewer_data = mujoco.MjData(self.backend.physics.model)
        self._viewer = mujoco.viewer.launch_passive(
            self.backend.physics.model,
            self._viewer_data,
            show_left_ui=self.backend.config.viewer.show_left_ui,
            show_right_ui=self.backend.config.viewer.show_right_ui,
        )
        _configure_mujoco_viewer(self._viewer, self.backend.config)
        self._sync_viewer()
        self._active = True
        self._viewer_gate.reset(self._clock())

    def update(self, state: RobotSystemState) -> None:
        self._last_state = state
        if not self._active or not self._viewer_gate.due(self._clock()):
            return
        timing = self.runtime_timing
        with (
            nullcontext()
            if timing is None
            else timing.measure("viewer.sync")
        ):
            if self._viewer is not None and self._viewer.is_running():
                self._sync_viewer()

    def _sync_viewer(self) -> None:
        with self._simulation_lock:
            copy_mujoco_dynamic_state(
                self.backend.physics.data,
                self._viewer_data,
            )
        self._mujoco.mj_forward(self.backend.physics.model, self._viewer_data)
        self._viewer.sync()

    def close(self) -> None:
        self._active = False
        if self._viewer is not None:
            self._viewer.close()
        self._viewer = None
        self._viewer_data = None


def run_manual_curvature_control(
    scenario_path: str | Path,
    *,
    panel_fps: float = 15.0,
    status_fps: float = 5.0,
    viewer_fps: float = 15.0,
    show_tendon_monitor: bool = False,
) -> None:
    """Run the curvature-only manual-control interface."""

    _run_manual_control(
        scenario_path,
        control_mode="curvature",
        panel_fps=panel_fps,
        status_fps=status_fps,
        viewer_fps=viewer_fps,
        show_tendon_monitor=show_tendon_monitor,
    )


def run_manual_tendon_control(
    scenario_path: str | Path,
    *,
    panel_fps: float = 15.0,
    status_fps: float = 5.0,
    viewer_fps: float = 15.0,
    show_tendon_monitor: bool = False,
) -> None:
    """Run the raw-tendon manual-control interface."""

    _run_manual_control(
        scenario_path,
        control_mode="tendon",
        panel_fps=panel_fps,
        status_fps=status_fps,
        viewer_fps=viewer_fps,
        show_tendon_monitor=show_tendon_monitor,
    )


def _run_manual_control(
    scenario_path: str | Path,
    *,
    control_mode: str,
    panel_fps: float,
    status_fps: float,
    viewer_fps: float,
    show_tendon_monitor: bool,
) -> None:
    application = SimulationApplication.from_yaml(scenario_path)
    backend = application.loop.backend
    simulation_lock = RLock()
    async_printer = AsyncLinePrinter(lambda line: print(line, flush=True))
    runtime_timing = RuntimeTimingReporter(
        report_interval_s=0.5,
        printer=async_printer.write,
    )
    windows = ManualControlWindows(
        backend,
        viewer_fps=viewer_fps,
        simulation_lock=simulation_lock,
        runtime_timing=runtime_timing,
    )
    viewer = MujocoSystemDebugViewer(
        backend,
        control_dt_s=application.config.runtime.controller_dt_s,
        n_substeps=application.config.runtime.n_substeps,
        state_update_callback=windows.update,
        control_mode=control_mode,
        panel_fps=panel_fps,
        status_fps=status_fps,
        show_tendon_monitor=show_tendon_monitor,
        simulation_lock=simulation_lock,
        runtime_timing=runtime_timing,
    )
    windows.start(viewer.state)
    runtime_timing.reset()
    async_printer.write(
        "[manual-timing] enabled; summary interval=0.5 s; "
        "stage values are avg/max milliseconds"
    )
    try:
        viewer.show()
    finally:
        viewer.close()
        windows.close()
        async_printer.close(drain=True)


__all__ = [
    "ManualControlWindows",
    "run_manual_curvature_control",
    "run_manual_tendon_control",
]
