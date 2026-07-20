"""Diagnostic sampling hooks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from continuum_sim.system.types import RobotSystemCommand, RobotSystemState


@dataclass
class TendonDiagnosticHook:
    """Collect tendon and singularity snapshots without a GUI dependency."""

    stride: int = 1
    samples: list[dict[str, Any]] = field(default_factory=list)

    def on_reset(self, state: RobotSystemState) -> None:
        self.samples.clear()
        self.samples.append(self._snapshot(state, None, -1))

    def on_step(
        self,
        state: RobotSystemState,
        command: RobotSystemCommand,
        step_index: int,
    ) -> None:
        if step_index % self.stride == 0:
            self.samples.append(self._snapshot(state, command, step_index))

    def should_stop(self, state: RobotSystemState, step_index: int) -> bool:
        del state, step_index
        return False

    def on_finish(self, state: RobotSystemState) -> None:
        del state

    def _snapshot(
        self,
        state: RobotSystemState,
        command: RobotSystemCommand | None,
        step_index: int,
    ) -> dict[str, Any]:
        return {
            "step_index": step_index,
            "time_s": state.time_s,
            "arms": {
                name: {
                    "tendon_target_m": arm.tendon_target_m.copy(),
                    "tendon_displacement_m": arm.tendon_displacement_m.copy(),
                    "tendon_velocity_mps": arm.tendon_velocity_mps.copy(),
                    "actuator_force_n": arm.actuator_force_n.copy(),
                }
                for name, arm in state.arms.items()
            },
            "command_metadata": {} if command is None else dict(command.metadata),
            "state_metadata": dict(state.metadata),
        }

__all__ = ["TendonDiagnosticHook"]
