"""Command adapters for the dual-arm MuJoCo tendon model."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from continuum_sim.model.dual_arm_robot import DualArmRobotConfig, load_dual_arm_robot_config


@dataclass(frozen=True)
class DualArmCommandAdapter:
    """Map task-arm commands into the full dual-arm actuator vector."""

    arm_names: tuple[str, ...]
    target_arm: str
    tendons_per_arm: int

    @classmethod
    def from_robot_config(
        cls,
        robot_config: str | Path | DualArmRobotConfig,
        *,
        target_arm: str | None = None,
    ) -> "DualArmCommandAdapter":
        config = (
            robot_config
            if isinstance(robot_config, DualArmRobotConfig)
            else load_dual_arm_robot_config(robot_config)
        )
        arm_name = target_arm or config.default_arm
        if arm_name not in config.arm_names:
            raise ValueError(f"target_arm {arm_name!r} is not defined in dual robot config.")
        tendon_counts = {name: len(config.tendons_by_arm[name]) for name in config.arm_names}
        if len(set(tendon_counts.values())) != 1:
            raise ValueError(f"Dual-arm tendon counts must match per arm, got {tendon_counts}.")
        tendons_per_arm = next(iter(tendon_counts.values()))
        return cls(
            arm_names=config.arm_names,
            target_arm=arm_name,
            tendons_per_arm=tendons_per_arm,
        )

    @property
    def total_tendon_count(self) -> int:
        return len(self.arm_names) * self.tendons_per_arm

    def adapt(self, command: np.ndarray) -> np.ndarray:
        """Return a full dual-arm command.

        A full dual-arm vector is passed through unchanged. A single-arm vector
        is placed into the configured target arm slice, with the other arm held
        at zero command.
        """

        command_array = np.asarray(command, dtype=float)
        if command_array.shape == (self.total_tendon_count,):
            return command_array
        if command_array.shape != (self.tendons_per_arm,):
            raise ValueError(
                "Expected dual-arm command with shape "
                f"({self.total_tendon_count},) or single-arm command with shape "
                f"({self.tendons_per_arm},), got {command_array.shape}."
            )
        expanded = np.zeros((self.total_tendon_count,), dtype=float)
        start = self.arm_names.index(self.target_arm) * self.tendons_per_arm
        expanded[start : start + self.tendons_per_arm] = command_array
        return expanded

    def split(self, command: np.ndarray) -> dict[str, np.ndarray]:
        """Split a full dual-arm command into per-arm views."""

        command_array = np.asarray(command, dtype=float)
        if command_array.shape != (self.total_tendon_count,):
            raise ValueError(
                f"Expected dual-arm command with shape ({self.total_tendon_count},), "
                f"got {command_array.shape}."
            )
        return {
            arm_name: command_array[
                index * self.tendons_per_arm : (index + 1) * self.tendons_per_arm
            ].copy()
            for index, arm_name in enumerate(self.arm_names)
        }

    def target_arm_view(self, command: np.ndarray) -> np.ndarray:
        """Return the configured target arm slice from a full dual-arm command."""

        return self.split(command)[self.target_arm]
