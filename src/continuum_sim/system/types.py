"""Backend-independent state and command types for composable robot systems."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from continuum_sim.model.base_pose import Pose6D


@dataclass(frozen=True)
class BaseSystemState:
    """Observed world-frame state of the prescribed 6D base."""

    pose: Pose6D
    twist_world: np.ndarray = field(default_factory=lambda: np.zeros(6, dtype=float))

    def __post_init__(self) -> None:
        object.__setattr__(self, "twist_world", _vector(self.twist_world, 6, "twist_world"))


@dataclass(frozen=True)
class ArmSystemState:
    """Observed state of one named direct-tendon spatial arm."""

    name: str
    role: str
    tip_pose_world: Pose6D
    segment_poses_world: np.ndarray
    tendon_displacement_m: np.ndarray
    tendon_velocity_mps: np.ndarray
    centerline_world: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        segment_poses = np.asarray(self.segment_poses_world, dtype=float)
        if segment_poses.ndim != 3 or segment_poses.shape[1:] != (4, 4):
            raise ValueError(
                "segment_poses_world must have shape (N, 4, 4), "
                f"got {segment_poses.shape}."
            )
        displacement = np.asarray(self.tendon_displacement_m, dtype=float)
        velocity = np.asarray(self.tendon_velocity_mps, dtype=float)
        if displacement.ndim != 1 or velocity.shape != displacement.shape:
            raise ValueError("Tendon displacement and velocity must be matching 1D arrays.")
        object.__setattr__(self, "segment_poses_world", segment_poses.copy())
        object.__setattr__(self, "tendon_displacement_m", displacement.copy())
        object.__setattr__(self, "tendon_velocity_mps", velocity.copy())
        if self.centerline_world is not None:
            centerline = np.asarray(self.centerline_world, dtype=float)
            if centerline.ndim != 2 or centerline.shape[1] != 3:
                raise ValueError("centerline_world must have shape (N, 3).")
            object.__setattr__(self, "centerline_world", centerline.copy())


@dataclass(frozen=True)
class RobotSystemState:
    """Named snapshot returned by analytic or physics backends."""

    time_s: float
    base: BaseSystemState
    arms: dict[str, ArmSystemState]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ArmTendonRateCommand:
    """Direct tendon-length change rate for one arm, in metres per second."""

    tendon_rate_mps: np.ndarray

    def __post_init__(self) -> None:
        values = np.asarray(self.tendon_rate_mps, dtype=float)
        if values.ndim != 1 or not np.all(np.isfinite(values)):
            raise ValueError("tendon_rate_mps must be a finite 1D array.")
        object.__setattr__(self, "tendon_rate_mps", values.copy())


@dataclass(frozen=True)
class RobotSystemCommand:
    """World-frame base twist plus named direct tendon-rate commands."""

    base_twist_world: np.ndarray
    arms: dict[str, ArmTendonRateCommand]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "base_twist_world",
            _vector(self.base_twist_world, 6, "base_twist_world"),
        )

    @classmethod
    def zeros(cls, tendon_counts: dict[str, int]) -> "RobotSystemCommand":
        return cls(
            base_twist_world=np.zeros(6, dtype=float),
            arms={
                name: ArmTendonRateCommand(np.zeros(count, dtype=float))
                for name, count in tendon_counts.items()
            },
        )


def _vector(values: np.ndarray, size: int, name: str) -> np.ndarray:
    result = np.asarray(values, dtype=float)
    if result.shape != (size,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite vector with shape ({size},).")
    return result.copy()

