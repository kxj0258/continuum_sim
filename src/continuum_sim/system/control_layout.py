"""Named system-variable layout and flat-array boundary helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from continuum_sim.model.bending_space import BendingSpaceModel
from continuum_sim.model.robot_assembly import RobotAssemblyConfig
from continuum_sim.system.types import ArmTendonRateCommand, RobotSystemCommand


@dataclass(frozen=True)
class ControlLayout:
    """Slices for enabled control variables.

    Fixed-base assemblies omit base DOFs from the flat control vector, so
    whole-body solves become bending-only instead of solving base motion and
    clearing it afterward.
    """

    base: slice
    arms: dict[str, slice]
    physical_tendons: dict[str, slice]
    bending_models: dict[str, BendingSpaceModel]
    size: int

    @classmethod
    def from_assembly(cls, assembly: RobotAssemblyConfig) -> "ControlLayout":
        base_size = 0 if assembly.base.control_mode == "fixed" else 6
        offset = base_size
        arm_slices: dict[str, slice] = {}
        tendon_slices: dict[str, slice] = {}
        bending_models: dict[str, BendingSpaceModel] = {}
        tendon_offset = 0
        for arm in assembly.enabled_arms:
            model = BendingSpaceModel.from_arm(
                arm.spatial_arm.params,
                arm.spatial_arm.tendons,
            )
            arm_slices[arm.name] = slice(offset, offset + model.bending_size)
            tendon_slices[arm.name] = slice(
                tendon_offset,
                tendon_offset + model.tendon_count,
            )
            bending_models[arm.name] = model
            offset += model.bending_size
            tendon_offset += model.tendon_count
        return cls(
            base=slice(0, base_size),
            arms=arm_slices,
            physical_tendons=tendon_slices,
            bending_models=bending_models,
            size=offset,
        )

    @property
    def base_size(self) -> int:
        return self.base.stop - self.base.start

    @property
    def tendon_size(self) -> int:
        return sum(
            tendon_slice.stop - tendon_slice.start
            for tendon_slice in self.physical_tendons.values()
        )

    def flatten(self, command: RobotSystemCommand) -> np.ndarray:
        result = np.zeros(self.size, dtype=float)
        if self.base_size:
            result[self.base] = command.base_twist_world
        unknown = set(command.arms).difference(self.arms)
        if unknown:
            raise KeyError(f"Command contains arms outside the control layout: {sorted(unknown)}")
        for name, arm_slice in self.arms.items():
            arm_command = command.arms.get(name)
            if arm_command is None:
                continue
            model = self.bending_models[name]
            if arm_command.tendon_rate_mps.shape != (model.tendon_count,):
                raise ValueError(
                    f"Arm {name!r} tendon rate must have shape ({model.tendon_count},), "
                    f"got {arm_command.tendon_rate_mps.shape}."
                )
            if arm_command.control_space != "bending_compatible":
                raise ValueError("Raw tendon debug commands cannot be flattened.")
            result[arm_slice] = model.estimate(arm_command.tendon_rate_mps)
        return result

    def unflatten(self, values: np.ndarray) -> RobotSystemCommand:
        vector = np.asarray(values, dtype=float)
        if vector.shape != (self.size,):
            raise ValueError(f"Expected system vector with shape ({self.size},), got {vector.shape}.")
        return RobotSystemCommand(
            base_twist_world=(
                vector[self.base].copy()
                if self.base_size
                else np.zeros(6, dtype=float)
            ),
            arms={
                name: ArmTendonRateCommand(
                    self.bending_models[name].to_tendon(vector[arm_slice])
                )
                for name, arm_slice in self.arms.items()
            },
        )

    def tendon_slice(self, arm_name: str) -> slice:
        try:
            tendon_slice = self.physical_tendons[arm_name]
        except KeyError as exc:
            raise KeyError(f"Unknown arm {arm_name!r}.") from exc
        return tendon_slice
