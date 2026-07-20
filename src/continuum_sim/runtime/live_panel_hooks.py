"""Interactive live-panel hooks."""

from __future__ import annotations

import numpy as np

from continuum_sim.runtime.hook_utils import finite_metadata_float as _finite_metadata_float
from continuum_sim.system.types import RobotSystemCommand, RobotSystemState

from continuum_sim.runtime.hooks_impl import LiveDiagnosticsPanelHook


class LiveTendonPanelHook:
    """Optional rich tendon monitor attached to the scenario hook lifecycle."""

    def __init__(self, *, stride: int = 1, history_points: int = 300) -> None:
        if stride <= 0:
            raise ValueError("LiveTendonPanelHook stride must be positive.")
        self.stride = stride
        self.history_points = history_points
        self._panel = None

    def on_reset(self, state: RobotSystemState) -> None:
        from continuum_sim.visualization.system_tendon_debug import (
            SystemTendonMonitorPanel,
        )

        if self._panel is not None:
            self._panel.close()
        self._panel = SystemTendonMonitorPanel()
        self._panel.update(state, redraw=False)
        self._panel.show(block=False)

    def on_step(
        self,
        state: RobotSystemState,
        command: RobotSystemCommand,
        step_index: int,
    ) -> None:
        del command
        if step_index % self.stride == 0:
            if self._panel is not None and self._panel.is_open():
                self._panel.update(state)
                self._panel.flush_events()

    def should_stop(self, state: RobotSystemState, step_index: int) -> bool:
        del state, step_index
        return False

    def on_finish(self, state: RobotSystemState) -> None:
        del state
        if self._panel is not None:
            _safe_panel_call(self._panel, "flush_events")
            _safe_panel_call(self._panel, "close")
            self._panel = None


def _safe_panel_call(panel: object, method_name: str) -> None:
    method = getattr(panel, method_name, None)
    if not callable(method):
        return
    try:
        method()
    except Exception:
        pass


class LiveWipingForcePanelHook:
    """Optional live panel for scenario wiping force/contact metadata."""

    def __init__(self, *, stride: int = 1, history_points: int = 300) -> None:
        if stride <= 0:
            raise ValueError("LiveWipingForcePanelHook stride must be positive.")
        self.stride = stride
        self.history_points = history_points
        self._plt = None
        self._figure = None
        self._axes = None
        self._time: list[float] = []
        self._target_force: list[float] = []
        self._current_force: list[float] = []
        self._force_error: list[float] = []
        self._contact_distance: list[float] = []

    def on_reset(self, state: RobotSystemState) -> None:
        import matplotlib.pyplot as plt

        self._plt = plt
        self._figure, self._axes = plt.subplots(2, 1, figsize=(9.5, 6.8), sharex=True)
        manager = getattr(self._figure.canvas, "manager", None)
        if manager is not None:
            manager.set_window_title("continuum_sim wiping contact force")
        self._time.clear()
        self._target_force.clear()
        self._current_force.clear()
        self._force_error.clear()
        self._contact_distance.clear()
        plt.ion()
        plt.show(block=False)

    def on_step(
        self,
        state: RobotSystemState,
        command: RobotSystemCommand,
        step_index: int,
    ) -> None:
        if step_index % self.stride != 0:
            return
        self._time.append(float(state.time_s))
        self._target_force.append(
            float(command.metadata.get("target_normal_force_n", np.nan))
        )
        self._current_force.append(
            _finite_metadata_float(
                command.metadata,
                "measured_normal_force_n",
                fallback_key="estimated_normal_force_n",
            )
        )
        self._force_error.append(float(command.metadata.get("force_error_n", np.nan)))
        self._contact_distance.append(
            float(command.metadata.get("contact_distance_m", np.nan))
        )
        self._trim()
        self._draw()

    def should_stop(self, state: RobotSystemState, step_index: int) -> bool:
        del state, step_index
        return False

    def on_finish(self, state: RobotSystemState) -> None:
        del state
        if self._plt is not None:
            try:
                self._plt.ioff()
                if self._figure is not None:
                    self._plt.close(self._figure)
            except Exception:
                pass
        self._figure = None
        self._axes = None

    def _trim(self) -> None:
        if len(self._time) <= self.history_points:
            return
        excess = len(self._time) - self.history_points
        del self._time[:excess]
        del self._target_force[:excess]
        del self._current_force[:excess]
        del self._force_error[:excess]
        del self._contact_distance[:excess]

    def _draw(self) -> None:
        if self._axes is None or self._figure is None:
            return
        force_axis, contact_axis = self._axes
        for axis in self._axes:
            axis.clear()
            axis.grid(True, alpha=0.25)
        time_s = np.asarray(self._time, dtype=float)
        target_force = np.asarray(self._target_force, dtype=float)
        current_force = np.asarray(self._current_force, dtype=float)
        force_error = np.asarray(self._force_error, dtype=float)
        contact_distance_mm = 1000.0 * np.asarray(self._contact_distance, dtype=float)
        penetration_mm = 1000.0 * np.maximum(
            0.0,
            -np.asarray(self._contact_distance, dtype=float),
        )

        force_axis.plot(time_s, target_force, "--", label="target force [N]")
        force_axis.plot(time_s, current_force, label="current contact force [N]")
        force_axis.plot(time_s, force_error, label="force error [N]")
        force_axis.set_ylabel("force [N]")
        force_axis.set_title("Wiping contact force")
        force_axis.legend(loc="upper right", fontsize=8)

        contact_axis.plot(time_s, contact_distance_mm, label="contact distance [mm]")
        contact_axis.plot(time_s, penetration_mm, label="penetration proxy [mm]")
        contact_axis.axhline(0.0, color="0.35", linestyle="--", linewidth=0.9)
        contact_axis.set_xlabel("time [s]")
        contact_axis.set_ylabel("distance [mm]")
        contact_axis.set_title("Contact distance / penetration proxy")
        contact_axis.legend(loc="upper right", fontsize=8)
        self._figure.tight_layout()
        self._figure.canvas.draw_idle()
        self._figure.canvas.flush_events()


__all__ = [
    "LiveDiagnosticsPanelHook",
    "LiveTendonPanelHook",
    "LiveWipingForcePanelHook",
]
