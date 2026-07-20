"""Recording-oriented runtime hooks."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from continuum_sim.runtime.hooks_impl import StateRecorderHook
from continuum_sim.system.types import RobotSystemCommand, RobotSystemState


@dataclass
class MujocoReplayRecorderHook:
    """Record generalized state needed for deterministic offscreen replay."""

    backend: object
    qpos: list[np.ndarray] = field(default_factory=list)
    qvel: list[np.ndarray] = field(default_factory=list)
    mocap_pos: list[np.ndarray] = field(default_factory=list)
    mocap_quat: list[np.ndarray] = field(default_factory=list)

    def on_reset(self, state: RobotSystemState) -> None:
        del state
        self.qpos.clear()
        self.qvel.clear()
        self.mocap_pos.clear()
        self.mocap_quat.clear()
        self._append()

    def on_step(
        self,
        state: RobotSystemState,
        command: RobotSystemCommand,
        step_index: int,
    ) -> None:
        del state, command, step_index
        self._append()

    def should_stop(self, state: RobotSystemState, step_index: int) -> bool:
        del state, step_index
        return False

    def on_finish(self, state: RobotSystemState) -> None:
        del state

    def _append(self) -> None:
        data = self.backend.physics.data
        self.qpos.append(np.asarray(data.qpos, dtype=float).copy())
        self.qvel.append(np.asarray(data.qvel, dtype=float).copy())
        self.mocap_pos.append(np.asarray(data.mocap_pos, dtype=float).copy())
        self.mocap_quat.append(np.asarray(data.mocap_quat, dtype=float).copy())

__all__ = [
    "MujocoReplayRecorderHook",
    "StateRecorderHook",
]
