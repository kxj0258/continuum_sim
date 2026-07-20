"""Actuation compatibility layer for bending-coupled tendon drives."""

from __future__ import annotations

import numpy as np

from continuum_sim.config import MujocoConfig
from continuum_sim.execution.tendon_rate_control import BendingRateServoConfig
from continuum_sim.execution.mujoco_tendon_position_adapter import (
    MujocoTendonPositionExecutionAdapter,
    TendonPositionExecutionStep,
)
from continuum_sim.model.robot_assembly import RobotAssemblyConfig
from continuum_sim.system.control_layout import ControlLayout
from continuum_sim.system.types import RobotSystemCommand


class ActuationCompatibilityLayer:
    """Project raw tendon-rate references into executable actuator targets.

    The layer owns bending-space compatibility, antagonistic tendon-rate
    projection, rate/lead/force limiting, and MuJoCo tendon-position target
    tracking.  Intent resolving stops before this layer.
    """

    def __init__(
        self,
        assembly: RobotAssemblyConfig,
        layout: ControlLayout,
        mujoco_config: MujocoConfig,
        tendon_rate_servo_config: BendingRateServoConfig | None,
    ) -> None:
        self._adapter = MujocoTendonPositionExecutionAdapter(
            assembly,
            layout,
            mujoco_config,
            tendon_rate_servo_config,
        )

    @property
    def last_applied_rates(self) -> dict[str, np.ndarray]:
        return self._adapter.last_applied_rates

    @property
    def last_tendon_targets(self) -> dict[str, np.ndarray]:
        return self._adapter.last_tendon_targets

    def reset(self, actual_tendon_displacement_m: np.ndarray) -> None:
        self._adapter.reset(actual_tendon_displacement_m)

    def project_and_track(
        self,
        command: RobotSystemCommand,
        *,
        dt: float,
        actual_tendon_displacement_m: np.ndarray,
        actuator_force_n: np.ndarray,
    ) -> TendonPositionExecutionStep:
        return self._adapter.step(
            command,
            dt=dt,
            actual_tendon_displacement_m=actual_tendon_displacement_m,
            actuator_force_n=actuator_force_n,
        )

    def finalize_step(
        self,
        step: TendonPositionExecutionStep,
        *,
        dt: float,
        previous_actual_tendon_displacement_m: np.ndarray,
        previous_tendon_targets: dict[str, np.ndarray],
        state_arms: dict[str, object],
    ) -> dict[str, dict[str, object]]:
        saturation = self._adapter.finalize_step(
            step,
            dt=dt,
            previous_actual_tendon_displacement_m=previous_actual_tendon_displacement_m,
            previous_tendon_targets=previous_tendon_targets,
            state_arms=state_arms,
        )
        for arm_saturation in saturation.values():
            arm_saturation["actuation_compatibility_layer"] = type(self).__name__
            arm_saturation["actuation_compatibility_output"] = (
                "mujoco_tendon_position_target"
            )
        return saturation
