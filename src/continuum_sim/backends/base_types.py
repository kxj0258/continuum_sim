"""Common backend-facing types shared across runtime adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class BackendState:
    """Common state returned by simulation backends."""

    time: float
    tip_pose: np.ndarray
    segment_poses: np.ndarray
    qpos: np.ndarray
    qvel: np.ndarray
    tendon_length: np.ndarray
    tendon_velocity: np.ndarray
    actuator_force: np.ndarray
    mocap_pos: np.ndarray | None = None
    mocap_quat: np.ndarray | None = None


class BackendProtocol(Protocol):
    """Minimal protocol shared by analytic and physics-backed runtimes."""

    def reset(self) -> BackendState:
        """Reset internal state and return the initial backend snapshot."""

    def step(
        self,
        control: np.ndarray | None = None,
        n_substeps: int = 1,
    ) -> BackendState:
        """Advance the backend and return the resulting state."""

    def get_state(self) -> BackendState:
        """Return the latest backend snapshot without mutating state."""
