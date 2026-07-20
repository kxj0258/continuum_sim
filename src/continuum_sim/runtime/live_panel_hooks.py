"""Interactive live-panel hooks."""

from __future__ import annotations

from continuum_sim.system.types import RobotSystemCommand, RobotSystemState

from continuum_sim.runtime.hooks_impl import (
    LiveDiagnosticsPanelHook,
    LiveWipingForcePanelHook,
)


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


__all__ = [
    "LiveDiagnosticsPanelHook",
    "LiveTendonPanelHook",
    "LiveWipingForcePanelHook",
]
