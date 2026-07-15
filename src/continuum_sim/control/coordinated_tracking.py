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
    bending_position_jacobian,
    centerline_point_bending_jacobian,
    rotate_position_jacobian_to_world,
)
from continuum_sim.kinematics.pcc import forward_kinematics
from continuum_sim.model.robot_assembly import RobotAssemblyConfig
from continuum_sim.scenes.engine_query import EngineSceneQueryProtocol
from continuum_sim.system.types import RobotSystemCommand, RobotSystemState


@dataclass(frozen=True)
class CoordinatedTrackingTarget:
    """World-frame executor target and observer tracking policy."""

    executor_position_world: np.ndarray
    executor_velocity_world: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=float)
    )
    observer_roi_position_world: np.ndarray | None = None
    observer_executor_offset_world: np.ndarray = field(
        default_factory=lambda: np.array([0.0, -0.04, 0.02], dtype=float)
    )
    observer_roi_blend: float = 0.25
    observer_control_mode: str = "tracking"

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "executor_position_world",
            _vector3(self.executor_position_world, "executor_position_world"),
        )
        object.__setattr__(
            self,
            "executor_velocity_world",
            _vector3(self.executor_velocity_world, "executor_velocity_world"),
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
        if self.observer_control_mode not in (
            "tracking",
            "collision_avoidance",
            "disabled",
        ):
            raise ValueError("Unsupported observer_control_mode.")


@dataclass(frozen=True)
class CoordinatedTrackingConfig:
    """Task gains and executor/observer safety distances.

    Executor tracking is the primary task. Observer avoidance and observation
    objectives are projected onto observer tendons only, so they cannot pull the
    executor or shared base away from the tracking trajectory.
    """

    executor_position_gain: float = 4.0
    observer_position_gain: float = 5.0
    feedforward_gain: float = 1.0
    max_target_speed_mps: float | None = None
    inter_arm_min_distance_m: float = 0.010
    inter_arm_influence_distance_m: float = 0.05
    inter_arm_hard_stop_distance_m: float = 0.008
    inter_arm_release_margin_m: float = 0.002
    inter_arm_avoidance_gain: float = 0.4
    inter_arm_max_avoidance_speed_mps: float | None = None
    observer_collision_priority: bool = False
    freeze_executor_inside_safe_distance: bool = False
    stop_all_on_critical_distance: bool = False
    centerline_samples_per_segment: int = 6
    engine_min_clearance_m: float = 0.01
    engine_influence_distance_m: float = 0.025
    engine_avoidance_gain: float = 4.0
    enforce_backend_tendon_limits: bool = False

    def __post_init__(self) -> None:
        if not np.isfinite(self.feedforward_gain) or self.feedforward_gain < 0.0:
            raise ValueError("feedforward_gain must be non-negative and finite.")


@dataclass(frozen=True)
class _InterArmCollisionResult:
    """Nearest centerline pair and optional observer-only avoidance task."""

    task: WholeBodyTask | None
    distance_m: float
    observer_index: int
    executor_index: int
    observer_point_world: np.ndarray
    executor_point_world: np.ndarray
    separation_normal_world: np.ndarray
    desired_speed_mps: float


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
        self._observer_avoidance_active = False

    def set_target(self, target: CoordinatedTrackingTarget) -> None:
        self.target = target

    def compute_command(self, state: RobotSystemState) -> RobotSystemCommand:
        jacobians = {
            arm.name: self._arm_system_jacobian(state, arm.name)
            for arm in self.assembly.enabled_arms
        }
        executor_tasks: list[WholeBodyTask] = []
        observer_tasks: list[WholeBodyTask] = []
        observer_collision_active = False
        observer_tracking_active = False
        executor_name = self._arm_name_for_role("executor")
        executor_state = state.arms[executor_name]
        observer_name = self._optional_arm_name_for_role("observer")
        observer_mode = self.target.observer_control_mode
        collision_result = (
            None
            if observer_name is None
            else self._inter_arm_collision_task(
                state,
                executor_name,
                observer_name,
            )
        )
        inter_arm_distance = (
            float("inf")
            if collision_result is None
            else collision_result.distance_m
        )
        if collision_result is not None and observer_mode == "collision_avoidance":
            activation_distance = self.config.inter_arm_influence_distance_m
            if self._observer_avoidance_active:
                release_distance = (
                    activation_distance + self.config.inter_arm_release_margin_m
                )
                if inter_arm_distance > release_distance:
                    self._observer_avoidance_active = False
            elif inter_arm_distance < activation_distance:
                self._observer_avoidance_active = True
        else:
            self._observer_avoidance_active = False
        critical_distance = (
            collision_result is not None
            and observer_mode == "collision_avoidance"
            and inter_arm_distance <= self.config.inter_arm_hard_stop_distance_m
        )
        executor_error = (
            self.target.executor_position_world
            - executor_state.tip_pose_world.position
        )
        executor_target_velocity = self._executor_target_velocity(executor_error)
        executor_tasks.append(
            WholeBodyTask(
                name="executor_tracking",
                jacobian=jacobians[executor_name],
                target_velocity=executor_target_velocity,
                weight=self.solver.weight_for("executor_tracking"),
            )
        )

        observer_target_position = np.full(3, np.nan, dtype=float)
        observer_target_velocity = np.zeros(3, dtype=float)
        observer_target_error = np.full(3, np.nan, dtype=float)
        if observer_name is not None:
            if (
                observer_mode == "collision_avoidance"
                and self._observer_avoidance_active
            ):
                if collision_result is not None and collision_result.task is not None:
                    observer_tasks.append(collision_result.task)
                    observer_target_velocity = (
                        collision_result.separation_normal_world
                        * collision_result.desired_speed_mps
                    )
                observer_collision_active = True
            elif observer_mode == "tracking":
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
                observer_target_position = desired_observer.copy()
                observer_target_error = (
                    desired_observer - observer_state.tip_pose_world.position
                )
                observer_target_velocity = self._limit_cartesian_velocity(
                    self.config.observer_position_gain
                    * observer_target_error
                )
                observer_tasks.append(
                    WholeBodyTask(
                        name="observer_tracking",
                        jacobian=self._observer_only_jacobian(
                            jacobians[observer_name],
                            observer_name,
                        ),
                        target_velocity=observer_target_velocity,
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
                    if arm.name == observer_name:
                        observer_tasks.append(scene_task)
                    else:
                        executor_tasks.append(scene_task)
        executor_result = self.solver.solve(
            executor_tasks,
            active_arm_names=(executor_name,),
            include_base=True,
        )
        observer_result = (
            None
            if observer_name is None
            else self.solver.solve(
                observer_tasks,
                active_arm_names=(observer_name,),
                include_base=False,
            )
        )
        zero_command = RobotSystemCommand.zeros(
            {
                arm.name: arm.spatial_arm.tendon_count
                for arm in self.assembly.enabled_arms
            }
        )
        output_arms = dict(zero_command.arms)
        output_arms[executor_name] = executor_result.command.arms[executor_name]
        if observer_name is not None and observer_result is not None:
            output_arms[observer_name] = observer_result.command.arms[observer_name]
        if critical_distance:
            safety_mode = "critical_avoidance"
        elif observer_collision_active:
            safety_mode = "avoidance"
        elif observer_tracking_active:
            safety_mode = "tracking"
        else:
            safety_mode = "idle"
        bending_control = {
            executor_name: executor_result.arm_diagnostics[executor_name]
        }
        arm_singularities = {
            executor_name: executor_result.arm_singularities[executor_name]
        }
        if observer_name is not None and observer_result is not None:
            bending_control[observer_name] = observer_result.arm_diagnostics[
                observer_name
            ]
            arm_singularities[observer_name] = observer_result.arm_singularities[
                observer_name
            ]
        self.last_diagnostics = {
            "whole_body_singularity": executor_result.singularity,
            "observer_singularity": (
                None if observer_result is None else observer_result.singularity
            ),
            "residual_norm": executor_result.residual_norm,
            "observer_residual_norm": (
                0.0 if observer_result is None else observer_result.residual_norm
            ),
            "observer_control_mode": observer_mode,
            "observer_collision_active": observer_collision_active,
            "observer_tracking_active": observer_tracking_active,
            "inter_arm_distance_m": inter_arm_distance,
            "inter_arm_min_distance_m": self.config.inter_arm_min_distance_m,
            "inter_arm_influence_distance_m": (
                self.config.inter_arm_influence_distance_m
            ),
            "inter_arm_hard_stop_distance_m": (
                self.config.inter_arm_hard_stop_distance_m
            ),
            "inter_arm_release_margin_m": self.config.inter_arm_release_margin_m,
            "inter_arm_safety_mode": safety_mode,
            "inter_arm_executor_frozen": False,
            "inter_arm_critical_distance": critical_distance,
            "inter_arm_hard_stop": False,
            "inter_arm_closest_observer_index": (
                -1
                if collision_result is None
                else collision_result.observer_index
            ),
            "inter_arm_closest_executor_index": (
                -1
                if collision_result is None
                else collision_result.executor_index
            ),
            "inter_arm_closest_observer_point_world": (
                np.full(3, np.nan, dtype=float)
                if collision_result is None
                else collision_result.observer_point_world.copy()
            ),
            "inter_arm_closest_executor_point_world": (
                np.full(3, np.nan, dtype=float)
                if collision_result is None
                else collision_result.executor_point_world.copy()
            ),
            "observer_target_position_world": observer_target_position,
            "observer_target_error_world": observer_target_error,
            "observer_target_velocity_world": observer_target_velocity,
            "observer_avoidance_desired_speed_mps": (
                0.0
                if collision_result is None
                else collision_result.desired_speed_mps
            ),
            "tendon_mapping_singularity": {
                arm.name: analyze_tendon_mapping(
                    arm.spatial_arm.params,
                    arm.spatial_arm.tendons,
                    self.solver.config.singularity,
                )
                for arm in self.assembly.enabled_arms
            },
            "bending_control": bending_control,
            "measured_compatibility": {
                arm.name: self._measured_compatibility(state, arm.name)
                for arm in self.assembly.enabled_arms
            },
            "executor_target_velocity_world": executor_target_velocity,
            "arm_singularities": arm_singularities,
            "whole_body_solver": executor_result.solver_diagnostics or {},
            "observer_solver": (
                {}
                if observer_result is None
                else observer_result.solver_diagnostics or {}
            ),
            "disable_backend_tendon_limits": (
                not self.config.enforce_backend_tendon_limits
            ),
        }
        return RobotSystemCommand(
            base_twist_world=executor_result.command.base_twist_world,
            arms=output_arms,
            metadata=self.last_diagnostics,
        )

    def _executor_target_velocity(self, executor_error: np.ndarray) -> np.ndarray:
        velocity = (
            self.config.executor_position_gain * executor_error
            + self.target.executor_velocity_world
        )
        return self._limit_cartesian_velocity(velocity)

    def _limit_cartesian_velocity(self, velocity: np.ndarray) -> np.ndarray:
        velocity = np.asarray(velocity, dtype=float).copy()
        limit = self.config.max_target_speed_mps
        norm = float(np.linalg.norm(velocity))
        if limit is not None and norm > limit:
            velocity = velocity * (limit / norm)
        return velocity

    def _arm_system_jacobian(
        self,
        state: RobotSystemState,
        arm_name: str,
    ) -> np.ndarray:
        arm_config = self.assembly.arms[arm_name]
        arm_state = state.arms[arm_name]
        model = self.solver.layout.bending_models[arm_name]
        q = model.to_q(model.estimate(arm_state.tendon_displacement_m))
        jacobian_local = bending_position_jacobian(
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
    ) -> _InterArmCollisionResult:
        executor_centerline, _, _ = self._world_centerline(
            state,
            executor_name,
        )
        observer_centerline, observer_q, observer_mount = self._world_centerline(
            state,
            observer_name,
        )
        movable_executor = executor_centerline[1:]
        movable_observer = observer_centerline[1:]
        pairwise = (
            movable_observer[:, None, :]
            - movable_executor[None, :, :]
        )
        distances = np.linalg.norm(pairwise, axis=2)
        observer_offset, executor_offset = np.unravel_index(
            int(np.argmin(distances)),
            distances.shape,
        )
        observer_index = int(observer_offset + 1)
        executor_index = int(executor_offset + 1)
        executor_point = executor_centerline[executor_index]
        observer_point = observer_centerline[observer_index]
        separation = observer_point - executor_point
        distance = float(np.linalg.norm(separation))
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
        activation_distance = self.config.inter_arm_influence_distance_m
        desired_speed = self.config.inter_arm_avoidance_gain * max(
            activation_distance - distance,
            0.0,
        )
        avoidance_speed_limit = self.config.inter_arm_max_avoidance_speed_mps
        if avoidance_speed_limit is not None:
            desired_speed = min(
                float(avoidance_speed_limit),
                desired_speed,
            )
        if (
            desired_speed <= 0.0
            or np.linalg.norm(relative_jacobian)
            <= self.solver.config.singularity.rank_tolerance
        ):
            task = None
        else:
            task = WholeBodyTask(
                name="executor_observer_collision_avoidance",
                jacobian=relative_jacobian,
                target_velocity=np.array([desired_speed], dtype=float),
                weight=self.solver.weight_for("observer_collision_avoidance"),
            )
        return _InterArmCollisionResult(
            task=task,
            distance_m=distance,
            observer_index=observer_index,
            executor_index=executor_index,
            observer_point_world=observer_point.copy(),
            executor_point_world=executor_point.copy(),
            separation_normal_world=normal.copy(),
            desired_speed_mps=float(desired_speed),
        )

    def _world_centerline(
        self,
        state: RobotSystemState,
        arm_name: str,
    ) -> tuple[np.ndarray, np.ndarray, object]:
        arm = self.assembly.arms[arm_name]
        arm_state = state.arms[arm_name]
        model = self.solver.layout.bending_models[arm_name]
        q = model.to_q(model.estimate(arm_state.tendon_displacement_m))
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
        local_jacobian = centerline_point_bending_jacobian(
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

    def _measured_compatibility(
        self,
        state: RobotSystemState,
        arm_name: str,
    ) -> dict[str, object]:
        displacement = state.arms[arm_name].tendon_displacement_m
        model = self.solver.layout.bending_models[arm_name]
        residual = model.residual(displacement)
        return {
            "residual_m": residual,
            "residual_norm_m": float(np.linalg.norm(residual)),
            "tolerance_m": model.compatibility_tolerance(displacement),
            "compatible": model.is_compatible(displacement),
        }

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
