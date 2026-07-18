"""Shared mobile-base approach stage used by staged task controllers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from continuum_sim.control.mobile_base_pose_control import MobileBasePoseController
from continuum_sim.model.base_pose import Pose6D, quaternion_wxyz_to_rotation_matrix
from continuum_sim.system.types import RobotSystemState


@dataclass(frozen=True)
class BaseApproachResult:
    """One base servo sample."""

    twist_world: np.ndarray
    position_error_m: float
    orientation_error_rad: float
    reached: bool


class BaseApproachStage:
    """Reusable mobile-base pose servo and tip-target staging helper."""

    def __init__(
        self,
        *,
        position_gain: float,
        orientation_gain: float,
        position_tolerance_m: float,
        orientation_tolerance_rad: float,
        standoff_m: float = 0.0,
        z_bias: float = 1.0,
    ) -> None:
        self.position_tolerance_m = float(position_tolerance_m)
        self.orientation_tolerance_rad = float(orientation_tolerance_rad)
        self.standoff_m = float(standoff_m)
        self.z_bias = float(z_bias)
        if (
            not np.isfinite(self.position_tolerance_m)
            or self.position_tolerance_m < 0.0
        ):
            raise ValueError("position_tolerance_m must be non-negative and finite.")
        if (
            not np.isfinite(self.orientation_tolerance_rad)
            or self.orientation_tolerance_rad < 0.0
        ):
            raise ValueError(
                "orientation_tolerance_rad must be non-negative and finite."
            )
        if not np.isfinite(self.standoff_m) or self.standoff_m < 0.0:
            raise ValueError("standoff_m must be non-negative and finite.")
        if not np.isfinite(self.z_bias):
            raise ValueError("z_bias must be finite.")
        self._pose_controller = MobileBasePoseController(
            position_gain=position_gain,
            orientation_gain=orientation_gain,
        )

    def compute_to_pose(
        self,
        state: RobotSystemState,
        target: Pose6D,
        *,
        max_linear_speed: float | None,
        max_angular_speed: float | None,
        ignore_orientation: bool = False,
    ) -> BaseApproachResult:
        twist, position_error, orientation_error = (
            self._pose_controller.compute_twist(
                state.base.pose,
                target,
                max_linear_speed=max_linear_speed,
                max_angular_speed=max_angular_speed,
            )
        )
        if ignore_orientation:
            twist = twist.copy()
            twist[3:] = 0.0
            orientation_error = 0.0
        reached = (
            position_error <= self.position_tolerance_m
            and orientation_error <= self.orientation_tolerance_rad
        )
        return BaseApproachResult(
            twist_world=twist,
            position_error_m=float(position_error),
            orientation_error_rad=float(orientation_error),
            reached=bool(reached),
        )

    def staged_tip_target(
        self,
        waypoint_world: np.ndarray,
        target_direction_world: np.ndarray | None,
    ) -> np.ndarray:
        waypoint = np.asarray(waypoint_world, dtype=float)
        if self.standoff_m <= 0.0:
            return waypoint.copy()
        direction = self.approach_direction(target_direction_world)
        if np.linalg.norm(direction) <= 1.0e-12:
            return waypoint.copy()
        return waypoint + self.standoff_m * direction

    def approach_direction(
        self,
        target_direction_world: np.ndarray | None,
    ) -> np.ndarray:
        if target_direction_world is None:
            return np.zeros(3, dtype=float)
        target_direction = np.asarray(target_direction_world, dtype=float)
        if target_direction.shape != (3,) or not np.all(np.isfinite(target_direction)):
            return np.zeros(3, dtype=float)
        if np.linalg.norm(target_direction) <= 1.0e-12:
            return np.zeros(3, dtype=float)
        target_direction = target_direction / np.linalg.norm(target_direction)
        approach = np.array(
            [-target_direction[0], -target_direction[1], self.z_bias],
            dtype=float,
        )
        norm = float(np.linalg.norm(approach))
        if not np.isfinite(norm) or norm <= 1.0e-12:
            approach = -target_direction.copy()
            norm = float(np.linalg.norm(approach))
        if not np.isfinite(norm) or norm <= 1.0e-12:
            return np.zeros(3, dtype=float)
        return approach / norm

    def base_pose_for_tip_target(
        self,
        state: RobotSystemState,
        executor_name: str,
        target_tip_position: np.ndarray,
        target_base_quat: np.ndarray,
    ) -> Pose6D:
        current_base_rotation = quaternion_wxyz_to_rotation_matrix(
            state.base.pose.quat
        )
        target_base_rotation = quaternion_wxyz_to_rotation_matrix(target_base_quat)
        tip_position = state.arms[executor_name].tip_pose_world.position
        tip_offset_base = current_base_rotation.T @ (
            tip_position - state.base.pose.position
        )
        return Pose6D(
            position=(
                np.asarray(target_tip_position, dtype=float)
                - target_base_rotation @ tip_offset_base
            ),
            quat=np.asarray(target_base_quat, dtype=float).copy(),
        )
