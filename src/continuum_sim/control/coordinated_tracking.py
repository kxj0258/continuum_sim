"""Executor-primary tracking with observer-only avoidance and observation."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from continuum_sim.control.whole_body_controller import (
    WholeBodyController,
    WholeBodyControllerConfig,
    WholeBodyTask,
)
from continuum_sim.kinematics.whole_body import (
    analyze_tendon_mapping,
    assemble_whole_body_jacobian,
    base_point_jacobian_world,
    centerline_point_tendon_jacobian,
    rotate_position_jacobian_to_world,
    tendon_position_jacobian,
)
from continuum_sim.kinematics.pcc import forward_kinematics
from continuum_sim.model.robot_assembly import RobotAssemblyConfig
from continuum_sim.model.tendon_coupling import physical_tendon_delta_to_q
from continuum_sim.scenes.engine_query import EngineSceneQueryProtocol
from continuum_sim.system.types import RobotSystemCommand, RobotSystemState


@dataclass(frozen=True)
class CoordinatedTrackingTarget:
    """World-frame executor target and observer tracking policy."""

    executor_position_world: np.ndarray
    observer_roi_position_world: np.ndarray | None = None
    observer_executor_offset_world: np.ndarray = field(
        default_factory=lambda: np.array([0.0, -0.04, 0.02], dtype=float)
    )
    observer_roi_blend: float = 0.25

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "executor_position_world",
            _vector3(self.executor_position_world, "executor_position_world"),
        )
        object.__setattr__(
            self,
            "observer_executor_offset_world",
            _vector3(
                self.observer_executor_offset_world,
                "observer_executor_offset_world",
            ),
        )
        if self.observer_roi_position_world is not None:
            object.__setattr__(
                self,
                "observer_roi_position_world",
                _vector3(
                    self.observer_roi_position_world,
                    "observer_roi_position_world",
                ),
            )
        if not 0.0 <= self.observer_roi_blend <= 1.0:
            raise ValueError("observer_roi_blend must be in [0, 1].")


@dataclass(frozen=True)
class CoordinatedTrackingConfig:
    """Task gains and executor/observer safety distances.

    Executor tracking is the primary task. Observer avoidance and observation
    objectives are projected onto observer tendons only, so they cannot pull the
    executor or shared base away from the tracking trajectory.
    """

    executor_position_gain: float = 4.0
    observer_position_gain: float = 5.0
    inter_arm_min_distance_m: float = 0.025
    inter_arm_influence_distance_m: float = 0.05
    inter_arm_avoidance_gain: float = 4.0
    centerline_samples_per_segment: int = 6
    engine_min_clearance_m: float = 0.01
    engine_influence_distance_m: float = 0.025
    engine_avoidance_gain: float = 4.0


class CoordinatedTrackingController:
    """Build executor-primary tracking and observer-only secondary tasks."""

    def __init__(
        self,
        assembly: RobotAssemblyConfig,
        target: CoordinatedTrackingTarget,
        *,
        config: CoordinatedTrackingConfig = CoordinatedTrackingConfig(),
        solver_config: WholeBodyControllerConfig = WholeBodyControllerConfig(),
        scene_query: EngineSceneQueryProtocol | None = None,
    ) -> None:
        self.assembly = assembly
        self.target = target
        self.config = config
        self.solver = WholeBodyController(assembly, solver_config)
        self.scene_query = scene_query
        self.last_diagnostics: dict[str, object] = {}

    def set_target(self, target: CoordinatedTrackingTarget) -> None:
        self.target = target

    def compute_command(self, state: RobotSystemState) -> RobotSystemCommand:
        jacobians = {
            arm.name: self._arm_system_jacobian(state, arm.name)
            for arm in self.assembly.enabled_arms
        }
        tasks: list[WholeBodyTask] = []
        observer_collision_active = False
        observer_tracking_active = False
        executor_name = self._arm_name_for_role("executor")
        executor_state = state.arms[executor_name]
        executor_error = (
            self.target.executor_position_world
            - executor_state.tip_pose_world.position
        )
        tasks.append(
            WholeBodyTask(
                name="executor_tracking",
                jacobian=jacobians[executor_name],
                target_velocity=self.config.executor_position_gain * executor_error,
                weight=self.solver.weight_for("executor_tracking"),
            )
        )

        observer_name = self._optional_arm_name_for_role("observer")
        if observer_name is not None:
            collision_task = self._inter_arm_collision_task(
                state,
                executor_name,
                observer_name,
            )
            if collision_task is not None:
                tasks.append(collision_task)
                observer_collision_active = True
            else:
                observer_state = state.arms[observer_name]
                desired_observer = (
                    executor_state.tip_pose_world.position
                    + self.target.observer_executor_offset_world
                )
                if self.target.observer_roi_position_world is not None:
                    blend = self.target.observer_roi_blend
                    desired_observer = (
                        (1.0 - blend) * desired_observer
                        + blend * self.target.observer_roi_position_world
                    )
                tasks.append(
                    WholeBodyTask(
                        name="observer_tracking",
                        jacobian=self._observer_only_jacobian(
                            jacobians[observer_name],
                            observer_name,
                        ),
                        target_velocity=self.config.observer_position_gain
                        * (desired_observer - observer_state.tip_pose_world.position),
                        weight=self.solver.weight_for("observer_tracking"),
                    )
                )
                observer_tracking_active = True

        if self.scene_query is not None:
            for arm in self.assembly.enabled_arms:
                scene_task = self._engine_collision_task(
                    state,
                    arm.name,
                )
                if scene_task is not None:
                    tasks.append(scene_task)
        result = self.solver.solve(tasks)
        self.last_diagnostics = {
            "whole_body_singularity": result.singularity,
            "residual_norm": result.residual_norm,
            "observer_collision_active": observer_collision_active,
            "observer_tracking_active": observer_tracking_active,
            "tendon_mapping_singularity": {
                arm.name: analyze_tendon_mapping(
                    arm.spatial_arm.params,
                    arm.spatial_arm.tendons,
                    self.solver.config.singularity,
                )
                for arm in self.assembly.enabled_arms
            },
        }
        return RobotSystemCommand(
            base_twist_world=result.command.base_twist_world,
            arms=result.command.arms,
            metadata=self.last_diagnostics,
        )

    def _arm_system_jacobian(
        self,
        state: RobotSystemState,
        arm_name: str,
    ) -> np.ndarray:
        arm_config = self.assembly.arms[arm_name]
        arm_state = state.arms[arm_name]
        q = physical_tendon_delta_to_q(
            arm_state.tendon_displacement_m,
            arm_config.spatial_arm.params,
            arm_config.spatial_arm.tendons,
        )
        jacobian_local = tendon_position_jacobian(
            q,
            arm_config.spatial_arm.params,
            arm_config.spatial_arm.tendons,
        )
        world_mount = state.base.pose.compose(arm_config.mount_pose)
        jacobian_world = rotate_position_jacobian_to_world(
            jacobian_local,
            world_mount.as_matrix()[:3, :3],
        )
        base_jacobian = base_point_jacobian_world(
            arm_state.tip_pose_world.position,
            state.base.pose.position,
        )
        return assemble_whole_body_jacobian(
            self.solver.layout,
            arm_name,
            base_jacobian,
            jacobian_world,
        )

    def _observer_only_jacobian(
        self,
        jacobian: np.ndarray,
        observer_name: str,
    ) -> np.ndarray:
        result = np.zeros_like(jacobian)
        observer_slice = self.solver.layout.arms[observer_name]
        result[:, observer_slice] = jacobian[:, observer_slice]
        return result

    def _inter_arm_collision_task(
        self,
        state: RobotSystemState,
        executor_name: str,
        observer_name: str,
    ) -> WholeBodyTask | None:
        executor_centerline, _, _ = self._world_centerline(
            state,
            executor_name,
        )
        observer_centerline, observer_q, observer_mount = self._world_centerline(
            state,
            observer_name,
        )
        pairwise = observer_centerline[:, None, :] - executor_centerline[None, :, :]
        distances = np.linalg.norm(pairwise, axis=2)
        observer_index, executor_index = np.unravel_index(
            int(np.argmin(distances)),
            distances.shape,
        )
        executor_point = executor_centerline[executor_index]
        observer_point = observer_centerline[observer_index]
        separation = observer_point - executor_point
        distance = float(np.linalg.norm(separation))
        if distance >= self.config.inter_arm_influence_distance_m:
            return None
        normal = (
            separation / distance
            if distance > 1.0e-12
            else np.array([0.0, -1.0, 0.0], dtype=float)
        )
        observer_point_jacobian = self._centerline_system_jacobian(
            state,
            observer_name,
            observer_q,
            observer_mount,
            int(observer_index),
            observer_point,
        )
        relative_jacobian = normal[None, :] @ self._observer_only_jacobian(
            observer_point_jacobian,
            observer_name,
        )
        desired_speed = self.config.inter_arm_avoidance_gain * max(
            self.config.inter_arm_min_distance_m - distance,
            0.0,
        )
        return WholeBodyTask(
            name="executor_observer_collision_avoidance",
            jacobian=relative_jacobian,
            target_velocity=np.array([desired_speed], dtype=float),
            weight=self.solver.weight_for("executor_collision_avoidance"),
        )

    def _world_centerline(
        self,
        state: RobotSystemState,
        arm_name: str,
    ) -> tuple[np.ndarray, np.ndarray, object]:
        arm = self.assembly.arms[arm_name]
        arm_state = state.arms[arm_name]
        q = physical_tendon_delta_to_q(
            arm_state.tendon_displacement_m,
            arm.spatial_arm.params,
            arm.spatial_arm.tendons,
        )
        local = forward_kinematics(
            q,
            arm.spatial_arm.params,
            samples_per_segment=self.config.centerline_samples_per_segment,
        ).centerline
        mount = state.base.pose.compose(arm.mount_pose)
        return mount.transform_points(local), q, mount

    def _centerline_system_jacobian(
        self,
        state: RobotSystemState,
        arm_name: str,
        q: np.ndarray,
        mount,
        centerline_index: int,
        point_world: np.ndarray,
    ) -> np.ndarray:
        arm = self.assembly.arms[arm_name]
        local_jacobian = centerline_point_tendon_jacobian(
            q,
            centerline_index,
            arm.spatial_arm.params,
            arm.spatial_arm.tendons,
            samples_per_segment=self.config.centerline_samples_per_segment,
        )
        world_jacobian = rotate_position_jacobian_to_world(
            local_jacobian,
            mount.as_matrix()[:3, :3],
        )
        return assemble_whole_body_jacobian(
            self.solver.layout,
            arm_name,
            base_point_jacobian_world(point_world, state.base.pose.position),
            world_jacobian,
        )

    def _engine_collision_task(
        self,
        state: RobotSystemState,
        arm_name: str,
    ) -> WholeBodyTask | None:
        centerline, q, mount = self._world_centerline(state, arm_name)
        queries = [self.scene_query.nearest_distance(point) for point in centerline]
        centerline_index = int(np.argmin([query.distance_m for query in queries]))
        query = queries[centerline_index]
        if query.distance_m >= self.config.engine_influence_distance_m:
            return None
        point_jacobian = self._centerline_system_jacobian(
            state,
            arm_name,
            q,
            mount,
            centerline_index,
            centerline[centerline_index],
        )
        if self.assembly.arms[arm_name].role == "observer":
            point_jacobian = self._observer_only_jacobian(point_jacobian, arm_name)
        desired_speed = self.config.engine_avoidance_gain * max(
            self.config.engine_min_clearance_m - query.distance_m,
            0.0,
        )
        return WholeBodyTask(
            name=f"{arm_name}_engine_collision_avoidance",
            jacobian=query.normal[None, :] @ point_jacobian,
            target_velocity=np.array([desired_speed], dtype=float),
            weight=self.solver.weight_for("executor_collision_avoidance"),
        )

    def _arm_name_for_role(self, role: str) -> str:
        name = self._optional_arm_name_for_role(role)
        if name is None:
            raise ValueError(f"Assembly has no enabled {role!r} arm.")
        return name

    def _optional_arm_name_for_role(self, role: str) -> str | None:
        names = [arm.name for arm in self.assembly.enabled_arms if arm.role == role]
        if len(names) > 1:
            raise ValueError(f"Assembly has multiple enabled {role!r} arms.")
        return names[0] if names else None


def _vector3(values: np.ndarray, name: str) -> np.ndarray:
    result = np.asarray(values, dtype=float)
    if result.shape != (3,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite vector with shape (3,).")
    return result.copy()
