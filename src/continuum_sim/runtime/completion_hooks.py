"""Controller completion hooks."""

from __future__ import annotations

from continuum_sim.system.types import RobotSystemCommand, RobotSystemState


class ControllerCompletionHook:
    """Stop a scenario when a waypoint-style controller reports completion."""

    def __init__(self, controller) -> None:
        self.controller = controller

    def on_reset(self, state: RobotSystemState) -> None:
        del state

    def on_step(
        self,
        state: RobotSystemState,
        command: RobotSystemCommand,
        step_index: int,
    ) -> None:
        del state, command, step_index

    def should_stop(self, state: RobotSystemState, step_index: int) -> bool:
        del state, step_index
        return bool(getattr(self.controller, "done", False))

    def on_finish(self, state: RobotSystemState) -> None:
        del state

__all__ = ["ControllerCompletionHook"]
