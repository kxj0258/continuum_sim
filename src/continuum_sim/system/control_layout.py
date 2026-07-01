"""Named system-variable layout and flat-array boundary helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from continuum_sim.model.robot_assembly import RobotAssemblyConfig
from continuum_sim.system.types import ArmTendonRateCommand, RobotSystemCommand


@dataclass(frozen=True)
class ControlLayout:
    """Slices for base twist and each enabled arm tendon-rate block."""

    base: slice
    arms: dict[str, slice]
    size: int

    @classmethod
    def from_assembly(cls, assembly: RobotAssemblyConfig) -> "ControlLayout":
        offset = 6
        arm_slices: dict[str, slice] = {}
        for arm in assembly.enabled_arms:
            count = arm.spatial_arm.tendon_count
            arm_slices[arm.name] = slice(offset, offset + count)
            offset += count
        return cls(base=slice(0, 6), arms=arm_slices, size=offset)

    @property
    def tendon_size(self) -> int:
        return self.size - 6

    def flatten(self, command: RobotSystemCommand) -> np.ndarray:
        result = np.zeros(self.size, dtype=float)
        result[self.base] = command.base_twist_world
        unknown = set(command.arms).difference(self.arms)
        if unknown:
            raise KeyError(f"Command contains arms outside the control layout: {sorted(unknown)}")
        for name, arm_slice in self.arms.items():
            arm_command = command.arms.get(name)
            if arm_command is None:
                continue
            expected = arm_slice.stop - arm_slice.start
            if arm_command.tendon_rate_mps.shape != (expected,):
                raise ValueError(
                    f"Arm {name!r} tendon rate must have shape ({expected},), "
                    f"got {arm_command.tendon_rate_mps.shape}."
                )
            result[arm_slice] = arm_command.tendon_rate_mps
        return result

    def unflatten(self, values: np.ndarray) -> RobotSystemCommand:
        vector = np.asarray(values, dtype=float)
        if vector.shape != (self.size,):
            raise ValueError(f"Expected system vector with shape ({self.size},), got {vector.shape}.")
        return RobotSystemCommand(
            base_twist_world=vector[self.base],
            arms={
                name: ArmTendonRateCommand(vector[arm_slice])
                for name, arm_slice in self.arms.items()
            },
        )

    def tendon_slice(self, arm_name: str) -> slice:
        try:
            system_slice = self.arms[arm_name]
        except KeyError as exc:
            raise KeyError(f"Unknown arm {arm_name!r}.") from exc
        return slice(system_slice.start - 6, system_slice.stop - 6)

