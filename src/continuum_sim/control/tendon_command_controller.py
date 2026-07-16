"""Kinematic layer that converts task velocities to tendon-rate commands."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from continuum_sim.control.coordinated_tracking import (
    CoordinatedTrackingConfig,
    CoordinatedTrackingController,
    CoordinatedTrackingTarget,
)
from continuum_sim.control.task_space_servo import TaskSpaceVelocityCommand
from continuum_sim.control.whole_body_controller import WholeBodyControllerConfig
from continuum_sim.model.robot_assembly import RobotAssemblyConfig
from continuum_sim.scenes.engine_query import EngineSceneQueryProtocol
from continuum_sim.system.types import RobotSystemCommand, RobotSystemState


@dataclass(frozen=True)
class ObserverCommandReference:
    """Observer policy passed through the tendon command layer."""

    control_mode: str = "disabled"
    executor_offset_world: np.ndarray = field(
        default_factory=lambda: np.array([0.0, -0.04, 0.02], dtype=float)
    )
    roi_position_world: np.ndarray | None = None
    roi_blend: float = 0.25

    def __post_init__(self) -> None:
        if self.control_mode not in ("tracking", "collision_avoidance", "disabled"):
            raise ValueError("Unsupported observer control_mode.")
        object.__setattr__(
            self,
            "executor_offset_world",
            _vector3(self.executor_offset_world, "executor_offset_world"),
        )
        if self.roi_position_world is not None:
            object.__setattr__(
                self,
                "roi_position_world",
                _vector3(self.roi_position_world, "roi_position_world"),
            )
        if not np.isfinite(self.roi_blend) or not 0.0 <= self.roi_blend <= 1.0:
            raise ValueError("roi_blend must be finite and in [0, 1].")


class TendonCommandController:
    """Layer-3 controller producing base twist and tendon-rate references."""

    def __init__(
        self,
        assembly: RobotAssemblyConfig,
        *,
        coordinated_config: CoordinatedTrackingConfig = CoordinatedTrackingConfig(),
        solver_config: WholeBodyControllerConfig = WholeBodyControllerConfig(),
        scene_query: EngineSceneQueryProtocol | None = None,
    ) -> None:
        self._controller = CoordinatedTrackingController(
            assembly,
            CoordinatedTrackingTarget(
                executor_position_world=np.zeros(3, dtype=float),
                executor_velocity_world=np.zeros(3, dtype=float),
            ),
            config=coordinated_config,
            solver_config=solver_config,
            scene_query=scene_query,
        )

    @property
    def solver(self):
        return self._controller.solver

    @property
    def config(self) -> CoordinatedTrackingConfig:
        return self._controller.config

    @property
    def last_diagnostics(self) -> dict[str, object]:
        return self._controller.last_diagnostics

    def compute_command(
        self,
        state: RobotSystemState,
        task_velocity: TaskSpaceVelocityCommand,
        observer: ObserverCommandReference,
    ) -> RobotSystemCommand:
        self._controller.set_target(
            CoordinatedTrackingTarget(
                executor_position_world=task_velocity.servo_anchor_position_world,
                executor_velocity_world=task_velocity.tcp_velocity_world,
                observer_roi_position_world=observer.roi_position_world,
                observer_executor_offset_world=observer.executor_offset_world,
                observer_roi_blend=observer.roi_blend,
                observer_control_mode=observer.control_mode,
            )
        )
        command = self._controller.compute_command(state)
        return RobotSystemCommand(
            base_twist_world=command.base_twist_world,
            arms=command.arms,
            metadata={
                **command.metadata,
                "layer3_output": "tendon_rate_ref",
                "tendon_command_controller": type(self).__name__,
            },
        )


def _vector3(values: np.ndarray, name: str) -> np.ndarray:
    result = np.asarray(values, dtype=float)
    if result.shape != (3,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite vector with shape (3,).")
    return result.copy()
