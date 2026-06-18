"""Facade for the motor-to-PCC kinematics chain."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from continuum_sim.actuation.motor_mapping import (
    MotorParams,
    load_motor_params_from_yaml,
    motor_position_to_tendon_delta,
)
from continuum_sim.control.differential_ik import (
    DifferentialIKConfig,
    TrackingResult,
    simulate_position_tracking,
)
from continuum_sim.kinematics.differential import motor_position_jacobian, tip_position_from_q
from continuum_sim.kinematics.pcc import PCCForwardKinematicsResult, forward_kinematics
from continuum_sim.model.physical_tendon import (
    PhysicalTendonPath,
    load_physical_tendons_from_yaml,
)
from continuum_sim.model.robot_params import ThreeSegmentRobotParams
from continuum_sim.model.tendon_coupling import physical_tendon_delta_to_q


@dataclass(frozen=True)
class ContinuumKinematicsChain:
    """Small facade for the stable motor -> tendon -> q -> FK/Jacobian chain."""

    params: ThreeSegmentRobotParams
    physical_tendons: tuple[PhysicalTendonPath, ...]
    motor_params: tuple[MotorParams, ...]

    @classmethod
    def from_robot_config(cls, path: str | Path) -> "ContinuumKinematicsChain":
        """Load robot, physical tendon, and motor parameters from one YAML file."""
        robot_config_path = Path(path)
        return cls(
            params=ThreeSegmentRobotParams.from_yaml(robot_config_path),
            physical_tendons=load_physical_tendons_from_yaml(robot_config_path),
            motor_params=load_motor_params_from_yaml(robot_config_path),
        )

    @property
    def motor_size(self) -> int:
        return len(self.motor_params)

    @property
    def q_size(self) -> int:
        return self.params.q_size

    def motor_position_to_tendon_delta(self, motor_position: np.ndarray) -> np.ndarray:
        """Map motor positions in rad to physical tendon length deltas in meters."""
        return motor_position_to_tendon_delta(
            self._as_motor_vector(motor_position, "motor_position"),
            self.motor_params,
        )

    def motor_position_to_q(self, motor_position: np.ndarray) -> np.ndarray:
        """Estimate PCC q from motor positions."""
        tendon_delta = self.motor_position_to_tendon_delta(motor_position)
        return physical_tendon_delta_to_q(tendon_delta, self.params, self.physical_tendons)

    def forward_kinematics_from_motor(
        self,
        motor_position: np.ndarray,
        *,
        samples_per_segment: int = 21,
    ) -> PCCForwardKinematicsResult:
        """Compute PCC forward kinematics from motor positions."""
        q = self.motor_position_to_q(motor_position)
        return forward_kinematics(
            q,
            self.params,
            samples_per_segment=samples_per_segment,
        )

    def tip_position_from_motor(self, motor_position: np.ndarray) -> np.ndarray:
        """Return the tip position implied by motor positions."""
        q = self.motor_position_to_q(motor_position)
        return tip_position_from_q(q, self.params)

    def motor_position_jacobian(
        self,
        motor_position: np.ndarray,
        *,
        step: float = 1.0e-5,
    ) -> np.ndarray:
        """Return d tip_position / d motor_velocity at a motor position."""
        q = self.motor_position_to_q(motor_position)
        return motor_position_jacobian(
            q,
            self.params,
            self.physical_tendons,
            self.motor_params,
            step=step,
        )

    def simulate_tracking(
        self,
        initial_motor_position: np.ndarray,
        target_positions: np.ndarray,
        config: DifferentialIKConfig,
        *,
        position_limit_rad: float = 2.0,
        stop_on_completion: bool = False,
    ) -> TrackingResult:
        """Run the existing differential-IK tracker through this loaded chain."""
        return simulate_position_tracking(
            self._as_motor_vector(initial_motor_position, "initial_motor_position"),
            target_positions,
            self.params,
            self.physical_tendons,
            self.motor_params,
            config,
            position_limit_rad=position_limit_rad,
            stop_on_completion=stop_on_completion,
        )

    def _as_motor_vector(self, values: np.ndarray, name: str) -> np.ndarray:
        array = np.asarray(values, dtype=float)
        if array.shape != (self.motor_size,):
            raise ValueError(f"Expected {name} with shape ({self.motor_size},), got {array.shape}.")
        return array
