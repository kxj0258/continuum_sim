"""Executor-primary tracking with observer-only avoidance and observation."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from continuum_sim.control.whole_body_controller import (
    WholeBodyController,
    WholeBodyControllerConfig,
    WholeBodyTask,
)
from continuum_sim.control.priority_stack import PriorityStackConfig
from continuum_sim.kinematics.whole_body import (
    analyze_tendon_mapping,
    assemble_whole_body_jacobian,
    base_orientation_jacobian_world,
    base_point_jacobian_world,
    bending_orientation_jacobian,
    bending_position_jacobian,
    centerline_point_bending_jacobian,
    rotate_angular_jacobian_to_world,
    rotate_position_jacobian_to_world,
)
from continuum_sim.kinematics.pcc import (
    DEFAULT_PCC_KINEMATICS_MODE,
    PCCKinematicsMode,
    forward_kinematics,
)
from continuum_sim.utils.math_utils import skew
from continuum_sim.model.robot_assembly import RobotAssemblyConfig
from continuum_sim.model.base_pose import (
    look_rotation_quaternion_wxyz,
    quaternion_error_rotation_vector,
    quaternion_wxyz_to_rotation_matrix,
)
from continuum_sim.scenes.engine_query import EngineSceneQueryProtocol
from continuum_sim.system.types import RobotSystemCommand, RobotSystemState


EXECUTOR_ORIENTATION_TRACKING_MODES = ("disabled", "weighted", "nullspace")
OBSERVER_CONTROL_MODES = (
    "tracking",
    "collision_avoidance",
    "visual_servo",
    "disabled",
)
OBSERVER_INTERARM_AVOIDANCE_MODES = ("collision_avoidance", "visual_servo")
EXECUTOR_SCENE_AVOIDANCE_MODES = ("disabled", "nullspace_after_pose")
OBSERVER_SCENE_AVOIDANCE_MODES = (
    "disabled",
    "nullspace_after_interarm_lookat",
)


@dataclass(frozen=True)
class _SceneClearanceData:
    query: object
    centerline: np.ndarray
    q: np.ndarray
    mount: object
    centerline_index: int


@dataclass(frozen=True)
class CoordinatedTrackingTarget:
    """World-frame executor target and observer tracking policy."""

    executor_position_world: np.ndarray
    executor_velocity_world: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=float)
    )
    executor_orientation_world_wxyz: np.ndarray | None = None
    executor_angular_velocity_world: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=float)
    )
    executor_orientation_control_mode: str = "disabled"
    executor_force_normal_world: np.ndarray | None = None
    executor_force_velocity_mps: float = 0.0
    executor_force_control_weight: float = 20.0
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
        if self.executor_orientation_world_wxyz is not None:
            object.__setattr__(
                self,
                "executor_orientation_world_wxyz",
                _quat4(
                    self.executor_orientation_world_wxyz,
                    "executor_orientation_world_wxyz",
                ),
            )
        object.__setattr__(
            self,
            "executor_angular_velocity_world",
            _vector3(
                self.executor_angular_velocity_world,
                "executor_angular_velocity_world",
            ),
        )
        if self.executor_orientation_control_mode not in ("disabled", "quaternion"):
            raise ValueError("Unsupported executor_orientation_control_mode.")
        if (
            self.executor_orientation_control_mode != "disabled"
            and self.executor_orientation_world_wxyz is None
        ):
            raise ValueError(
                "executor_orientation_world_wxyz is required when executor "
                "orientation control is enabled."
            )
        if self.executor_force_normal_world is not None:
            force_normal = _vector3(
                self.executor_force_normal_world,
                "executor_force_normal_world",
            )
            force_norm = float(np.linalg.norm(force_normal))
            if force_norm <= 1.0e-12:
                raise ValueError("executor_force_normal_world must be nonzero.")
            object.__setattr__(
                self,
                "executor_force_normal_world",
                force_normal / force_norm,
            )
        if not np.isfinite(self.executor_force_velocity_mps):
            raise ValueError("executor_force_velocity_mps must be finite.")
        if (
            not np.isfinite(self.executor_force_control_weight)
            or self.executor_force_control_weight <= 0.0
        ):
            raise ValueError("executor_force_control_weight must be positive.")
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
        if self.observer_control_mode not in OBSERVER_CONTROL_MODES:
            raise ValueError("Unsupported observer_control_mode.")


@dataclass(frozen=True)
class CoordinatedTrackingConfig:
    """Task gains and executor/observer safety distances.

    Executor tracking is the primary task. Observer avoidance and observation
    objectives are projected onto observer tendons only, so they cannot pull the
    executor or shared base away from the tracking trajectory.
    """

    kinematics_mode: PCCKinematicsMode = DEFAULT_PCC_KINEMATICS_MODE
    executor_position_gain: float = 4.0
    executor_orientation_gain: float = 2.0
    observer_position_gain: float = 5.0
    executor_orientation_tracking_weight: float = 20.0
    executor_orientation_tracking_mode: str = "nullspace"
    feedforward_gain: float = 1.0
    max_target_speed_mps: float | None = None
    max_target_angular_speed_rad_s: float | None = None
    inter_arm_min_distance_m: float = 0.010
    inter_arm_influence_distance_m: float = 0.05
    inter_arm_hard_stop_distance_m: float = 0.008
    inter_arm_release_margin_m: float = 0.002
    inter_arm_avoidance_gain: float = 0.4
    inter_arm_max_avoidance_speed_mps: float | None = None
    inter_arm_collision_pair_count: int = 1
    inter_arm_collision_pair_index_separation: int = 1
    observer_look_at_executor_tip: bool = False
    observer_look_at_gain: float = 1.0
    observer_look_at_weight: float = 10.0
    observer_look_at_distance_m: float = 0.010
    observer_look_at_max_speed_mps: float | None = 0.005
    observer_visual_servo_center_gain: float = 1.0
    observer_visual_servo_depth_gain: float = 1.0
    observer_visual_servo_depth_target_m: float = 0.08
    observer_visual_servo_max_speed_mps: float | None = 0.010
    observer_visual_servo_max_angular_speed_rad_s: float | None = 1.0
    observer_collision_priority: bool = False
    freeze_executor_inside_safe_distance: bool = False
    stop_all_on_critical_distance: bool = False
    centerline_samples_per_segment: int = 6
    scene_avoidance_enabled: bool = True
    executor_scene_avoidance_mode: str = "nullspace_after_pose"
    observer_scene_avoidance_mode: str = "nullspace_after_interarm_lookat"
    engine_min_clearance_m: float = 0.01
    engine_influence_distance_m: float = 0.025
    engine_avoidance_gain: float = 4.0
    enforce_backend_tendon_limits: bool = False
    priority_stack: PriorityStackConfig = field(
        default_factory=PriorityStackConfig
    )

    def __post_init__(self) -> None:
        if self.executor_orientation_tracking_mode not in EXECUTOR_ORIENTATION_TRACKING_MODES:
            raise ValueError(
                "executor_orientation_tracking_mode must be one of "
                f"{EXECUTOR_ORIENTATION_TRACKING_MODES}."
            )
        if self.executor_scene_avoidance_mode not in EXECUTOR_SCENE_AVOIDANCE_MODES:
            raise ValueError(
                "executor_scene_avoidance_mode must be one of "
                f"{EXECUTOR_SCENE_AVOIDANCE_MODES}."
            )
        if self.observer_scene_avoidance_mode not in OBSERVER_SCENE_AVOIDANCE_MODES:
            raise ValueError(
                "observer_scene_avoidance_mode must be one of "
                f"{OBSERVER_SCENE_AVOIDANCE_MODES}."
            )
        if not np.isfinite(self.feedforward_gain) or self.feedforward_gain < 0.0:
            raise ValueError("feedforward_gain must be non-negative and finite.")
        if self.inter_arm_collision_pair_count <= 0:
            raise ValueError("inter_arm_collision_pair_count must be positive.")
        if self.inter_arm_collision_pair_index_separation < 1:
            raise ValueError(
                "inter_arm_collision_pair_index_separation must be at least 1."
            )
        positive = {
            "observer_look_at_gain": self.observer_look_at_gain,
            "observer_look_at_weight": self.observer_look_at_weight,
            "observer_look_at_distance_m": self.observer_look_at_distance_m,
            "observer_visual_servo_center_gain": (
                self.observer_visual_servo_center_gain
            ),
            "observer_visual_servo_depth_gain": (
                self.observer_visual_servo_depth_gain
            ),
            "observer_visual_servo_depth_target_m": (
                self.observer_visual_servo_depth_target_m
            ),
            "executor_orientation_tracking_weight": (
                self.executor_orientation_tracking_weight
            ),
        }
        for name, value in positive.items():
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite.")
        if self.observer_look_at_max_speed_mps is not None and (
            not np.isfinite(self.observer_look_at_max_speed_mps)
            or self.observer_look_at_max_speed_mps <= 0.0
        ):
            raise ValueError(
                "observer_look_at_max_speed_mps must be positive and finite."
            )
        if self.observer_visual_servo_max_speed_mps is not None and (
            not np.isfinite(self.observer_visual_servo_max_speed_mps)
            or self.observer_visual_servo_max_speed_mps <= 0.0
        ):
            raise ValueError(
                "observer_visual_servo_max_speed_mps must be positive and finite."
            )
        if self.observer_visual_servo_max_angular_speed_rad_s is not None and (
            not np.isfinite(self.observer_visual_servo_max_angular_speed_rad_s)
            or self.observer_visual_servo_max_angular_speed_rad_s <= 0.0
        ):
            raise ValueError(
                "observer_visual_servo_max_angular_speed_rad_s must be positive and finite."
            )
        if not np.isfinite(self.executor_orientation_gain) or self.executor_orientation_gain < 0.0:
            raise ValueError("executor_orientation_gain must be non-negative and finite.")
        if self.max_target_angular_speed_rad_s is not None and (
            not np.isfinite(self.max_target_angular_speed_rad_s)
            or self.max_target_angular_speed_rad_s <= 0.0
        ):
            raise ValueError(
                "max_target_angular_speed_rad_s must be positive and finite."
            )
        if not np.isfinite(self.engine_min_clearance_m) or self.engine_min_clearance_m < 0.0:
            raise ValueError("engine_min_clearance_m must be non-negative and finite.")
        if (
            not np.isfinite(self.engine_influence_distance_m)
            or self.engine_influence_distance_m < 0.0
        ):
            raise ValueError(
                "engine_influence_distance_m must be non-negative and finite."
            )
        if self.engine_influence_distance_m < self.engine_min_clearance_m:
            raise ValueError(
                "engine_influence_distance_m must be at least engine_min_clearance_m."
            )
        if not np.isfinite(self.engine_avoidance_gain) or self.engine_avoidance_gain <= 0.0:
            raise ValueError("engine_avoidance_gain must be positive and finite.")


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
    pair_observer_indices: np.ndarray
    pair_executor_indices: np.ndarray
    pair_distances_m: np.ndarray
    pair_desired_speeds_mps: np.ndarray


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
        orientation_jacobians = {
            arm.name: self._arm_orientation_system_jacobian(state, arm.name)
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
        observer_uses_interarm_avoidance = (
            observer_mode in OBSERVER_INTERARM_AVOIDANCE_MODES
        )
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
        if collision_result is not None and observer_uses_interarm_avoidance:
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
            and observer_uses_interarm_avoidance
            and inter_arm_distance <= self.config.inter_arm_hard_stop_distance_m
        )
        executor_error = (
            self.target.executor_position_world
            - executor_state.tip_pose_world.position
        )
        executor_target_velocity = self._executor_target_velocity(executor_error)
        executor_tracking_jacobian = jacobians[executor_name]
        executor_tracking_velocity = executor_target_velocity
        force_normal = self.target.executor_force_normal_world
        executor_tracking_projected = False
        if force_normal is not None:
            tangent_projection = np.eye(3, dtype=float) - np.outer(
                force_normal,
                force_normal,
            )
            executor_tracking_jacobian = tangent_projection @ executor_tracking_jacobian
            executor_tracking_velocity = tangent_projection @ executor_target_velocity
            executor_tracking_projected = True
        executor_tasks.append(
            WholeBodyTask(
                name="executor_tracking",
                jacobian=executor_tracking_jacobian,
                target_velocity=executor_tracking_velocity,
                weight=self.solver.weight_for("executor_tracking"),
            )
        )
        executor_force_task = self._executor_force_control_task(
            jacobians[executor_name]
        )
        executor_force_tasks = (
            () if executor_force_task is None else (executor_force_task,)
        )
        executor_orientation_error = np.zeros(3, dtype=float)
        executor_orientation_velocity = np.zeros(3, dtype=float)
        executor_orientation_requested = (
            self.target.executor_orientation_control_mode == "quaternion"
            and self.target.executor_orientation_world_wxyz is not None
        )
        executor_orientation_active = (
            executor_orientation_requested
            and self.config.executor_orientation_tracking_mode != "disabled"
        )
        executor_orientation_task = None
        if executor_orientation_requested:
            executor_orientation_error = quaternion_error_rotation_vector(
                self.target.executor_orientation_world_wxyz,
                executor_state.tip_pose_world.quat,
            )
            executor_orientation_velocity = (
                self.target.executor_angular_velocity_world.copy()
            )
            executor_orientation_task = WholeBodyTask(
                name="executor_orientation_tracking",
                jacobian=orientation_jacobians[executor_name],
                target_velocity=executor_orientation_velocity,
                weight=self.config.executor_orientation_tracking_weight,
            )

        observer_target_position = np.full(3, np.nan, dtype=float)
        observer_target_velocity = np.zeros(3, dtype=float)
        observer_target_error = np.full(3, np.nan, dtype=float)
        observer_look_at_active = False
        observer_visual_servo_active = False
        observer_look_at_error = np.full(3, np.nan, dtype=float)
        observer_look_at_velocity = np.zeros(3, dtype=float)
        observer_visual_servo_pixel_error = np.full(2, np.nan, dtype=float)
        observer_visual_servo_angular_velocity = np.zeros(3, dtype=float)
        observer_visual_servo_depth_error = float("nan")
        observer_visual_servo_position_velocity = np.zeros(3, dtype=float)
        observer_avoidance_tasks: list[WholeBodyTask] = []
        observer_tracking_tasks: list[WholeBodyTask] = []
        observer_look_at_tasks: list[WholeBodyTask] = []
        observer_visual_servo_tasks: list[WholeBodyTask] = []
        executor_scene_tasks: list[WholeBodyTask] = []
        observer_scene_tasks: list[WholeBodyTask] = []
        scene_clearance_by_arm: dict[str, _SceneClearanceData] = {}
        if observer_name is not None:
            if (
                observer_uses_interarm_avoidance
                and self._observer_avoidance_active
            ):
                if collision_result is not None and collision_result.task is not None:
                    observer_tasks.append(collision_result.task)
                    observer_avoidance_tasks.append(collision_result.task)
                    observer_target_velocity = (
                        collision_result.separation_normal_world
                        * collision_result.desired_speed_mps
                    )
                observer_collision_active = True
            if observer_mode == "tracking":
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
                observer_tracking_task = WholeBodyTask(
                    name="observer_tracking",
                    jacobian=self._observer_only_jacobian(
                        jacobians[observer_name],
                        observer_name,
                    ),
                    target_velocity=observer_target_velocity,
                    weight=self.solver.weight_for("observer_tracking"),
                )
                observer_tasks.append(observer_tracking_task)
                observer_tracking_tasks.append(observer_tracking_task)
                observer_tracking_active = True
            elif observer_mode == "visual_servo":
                (
                    visual_tasks,
                    visual_pixel_error,
                    visual_angular_velocity,
                    visual_depth_error,
                    visual_position_velocity,
                ) = self._observer_visual_servo_tasks(
                    state,
                    observer_name,
                    avoidance_tasks=tuple(observer_avoidance_tasks),
                )
                observer_visual_servo_tasks.extend(visual_tasks)
                observer_visual_servo_active = bool(visual_tasks)
                observer_visual_servo_pixel_error = visual_pixel_error
                observer_visual_servo_angular_velocity = visual_angular_velocity
                observer_visual_servo_depth_error = visual_depth_error
                observer_visual_servo_position_velocity = visual_position_velocity
                if observer_visual_servo_active:
                    observer_target_velocity = visual_position_velocity.copy()

        if self.scene_query is not None:
            for arm in self.assembly.enabled_arms:
                scene_clearance_by_arm[arm.name] = self._engine_clearance(
                    state,
                    arm.name,
                )
        if self.config.scene_avoidance_enabled and self.scene_query is not None:
            for arm in self.assembly.enabled_arms:
                scene_task = self._engine_collision_task(
                    state,
                    arm.name,
                    scene_clearance_by_arm[arm.name],
                )
                if scene_task is not None:
                    if (
                        arm.name == observer_name
                        and self.config.observer_scene_avoidance_mode != "disabled"
                    ):
                        observer_scene_tasks.append(scene_task)
                    elif (
                        arm.name == executor_name
                        and self.config.executor_scene_avoidance_mode != "disabled"
                    ):
                        executor_scene_tasks.append(scene_task)

        if observer_name is not None:
            if (
                observer_mode in ("tracking", "collision_avoidance")
                and self.config.observer_look_at_executor_tip
            ):
                look_at_task, look_at_error, look_at_velocity = (
                    self._observer_look_at_task(
                        state,
                        executor_name,
                        observer_name,
                        avoidance_tasks=(),
                    )
                )
                if look_at_task is not None:
                    observer_look_at_tasks.append(look_at_task)
                    observer_look_at_active = True
                    observer_look_at_error = look_at_error
                    observer_look_at_velocity = look_at_velocity
                    if not observer_collision_active and not observer_tracking_active:
                        observer_target_velocity = look_at_velocity.copy()
        executor_position_tasks = list(executor_tasks)
        executor_orientation_tasks: tuple[WholeBodyTask, ...] = ()
        if executor_orientation_task is not None:
            if self.config.executor_orientation_tracking_mode == "weighted":
                executor_position_tasks.append(executor_orientation_task)
            elif self.config.executor_orientation_tracking_mode == "nullspace":
                executor_orientation_tasks = (executor_orientation_task,)
        executor_result = self.solver.solve_priority_stack(
            _priority_levels(
                self.config.priority_stack.executor,
                {
                    "position_servo": tuple(executor_position_tasks),
                    "normal_force_control": tuple(executor_force_tasks),
                    "orientation_servo": executor_orientation_tasks,
                    "scene_avoidance": tuple(executor_scene_tasks),
                },
            ),
            active_arm_names=(executor_name,),
            include_base=True,
        )
        observer_result = (
            None
            if observer_name is None
            else self.solver.solve_priority_stack(
                _priority_levels(
                    self.config.priority_stack.observer,
                    {
                        "interarm_avoidance": tuple(observer_avoidance_tasks),
                        "observer_tracking": tuple(observer_tracking_tasks),
                        "visual_servo": tuple(observer_visual_servo_tasks),
                        "look_at": tuple(observer_look_at_tasks),
                        "scene_avoidance": tuple(observer_scene_tasks),
                    },
                ),
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
        elif observer_visual_servo_active:
            safety_mode = "visual_servo"
        elif observer_look_at_active:
            safety_mode = "look_at"
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
        visual_servo_normalized_error = _metadata_vector(
            state.metadata,
            "visual_servo_normalized_error",
            2,
            allow_nan=True,
        )
        visual_servo_roi_world = _metadata_vector(
            state.metadata,
            "visual_servo_roi_world",
            3,
        )
        visual_servo_camera_position_world = _metadata_vector(
            state.metadata,
            "visual_servo_camera_position_world",
            3,
        )
        self.last_diagnostics = {
            "whole_body_singularity": executor_result.singularity,
            "priority_stack": self.config.priority_stack.as_metadata(),
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
            "observer_look_at_active": observer_look_at_active,
            "observer_visual_servo_active": observer_visual_servo_active,
            "observer_visual_servo_pixel_error_px": (
                observer_visual_servo_pixel_error
            ),
            "observer_visual_servo_angular_velocity_world": (
                observer_visual_servo_angular_velocity
            ),
            "observer_visual_servo_depth_error_m": (
                observer_visual_servo_depth_error
            ),
            "observer_visual_servo_position_velocity_world": (
                observer_visual_servo_position_velocity
            ),
            "visual_servo_target_visible": bool(
                state.metadata.get("visual_servo_target_visible", False)
            ),
            "visual_servo_depth_m": _metadata_float(
                state.metadata,
                "visual_servo_depth_m",
            ),
            "visual_servo_normalized_error": (
                visual_servo_normalized_error
                if visual_servo_normalized_error is not None
                else np.full(2, np.nan, dtype=float)
            ),
            "visual_servo_roi_world": (
                visual_servo_roi_world
                if visual_servo_roi_world is not None
                else np.full(3, np.nan, dtype=float)
            ),
            "visual_servo_camera_position_world": (
                visual_servo_camera_position_world
                if visual_servo_camera_position_world is not None
                else np.full(3, np.nan, dtype=float)
            ),
            "observer_look_at_nullspace_tasks": tuple(
                task.name for task in observer_avoidance_tasks
            ),
            "scene_avoidance_enabled": self.config.scene_avoidance_enabled,
            "executor_scene_avoidance_mode": (
                self.config.executor_scene_avoidance_mode
            ),
            "observer_scene_avoidance_mode": (
                self.config.observer_scene_avoidance_mode
            ),
            "engine_min_clearance_m": self.config.engine_min_clearance_m,
            "engine_influence_distance_m": (
                self.config.engine_influence_distance_m
            ),
            "engine_avoidance_gain": self.config.engine_avoidance_gain,
            "executor_scene_collision_active": bool(executor_scene_tasks),
            "observer_scene_collision_active": bool(observer_scene_tasks),
            "executor_force_control_active": bool(executor_force_tasks),
            "executor_force_control_velocity_mps": (
                self.target.executor_force_velocity_mps
            ),
            "executor_force_control_normal_world": (
                np.full(3, np.nan, dtype=float)
                if self.target.executor_force_normal_world is None
                else self.target.executor_force_normal_world.copy()
            ),
            "executor_force_control_weight": (
                self.target.executor_force_control_weight
            ),
            "executor_tracking_projected_for_force_control": (
                executor_tracking_projected
            ),
            "executor_clearance_m": _scene_clearance_distance(
                scene_clearance_by_arm,
                executor_name,
            ),
            "observer_clearance_m": _scene_clearance_distance(
                scene_clearance_by_arm,
                observer_name,
            ),
            "observer_look_at_error_world": observer_look_at_error,
            "observer_look_at_velocity_world": observer_look_at_velocity,
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
            "inter_arm_collision_pair_observer_indices": (
                np.zeros(0, dtype=int)
                if collision_result is None
                else collision_result.pair_observer_indices.copy()
            ),
            "inter_arm_collision_pair_executor_indices": (
                np.zeros(0, dtype=int)
                if collision_result is None
                else collision_result.pair_executor_indices.copy()
            ),
            "inter_arm_collision_pair_distances_m": (
                np.zeros(0, dtype=float)
                if collision_result is None
                else collision_result.pair_distances_m.copy()
            ),
            "inter_arm_collision_pair_desired_speeds_mps": (
                np.zeros(0, dtype=float)
                if collision_result is None
                else collision_result.pair_desired_speeds_mps.copy()
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
            "executor_orientation_control_active": executor_orientation_active,
            "executor_orientation_tracking_mode": (
                self.config.executor_orientation_tracking_mode
            ),
            "executor_target_orientation_world_wxyz": (
                np.full(4, np.nan, dtype=float)
                if self.target.executor_orientation_world_wxyz is None
                else self.target.executor_orientation_world_wxyz.copy()
            ),
            "executor_orientation_error_world": executor_orientation_error,
            "executor_orientation_error_rad": float(
                np.linalg.norm(executor_orientation_error)
            ),
            "executor_target_angular_velocity_world": executor_orientation_velocity,
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
            "kinematics_mode": self.solver.config.kinematics_mode,
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

    def _limit_observer_look_at_velocity(self, velocity: np.ndarray) -> np.ndarray:
        velocity = np.asarray(velocity, dtype=float).copy()
        limit = self.config.observer_look_at_max_speed_mps
        norm = float(np.linalg.norm(velocity))
        if limit is not None and norm > limit:
            velocity = velocity * (limit / norm)
        return velocity

    def _limit_observer_visual_linear_velocity(self, velocity: np.ndarray) -> np.ndarray:
        velocity = np.asarray(velocity, dtype=float).copy()
        limit = self.config.observer_visual_servo_max_speed_mps
        norm = float(np.linalg.norm(velocity))
        if limit is not None and norm > limit:
            velocity = velocity * (limit / norm)
        return velocity

    def _limit_observer_visual_angular_velocity(self, velocity: np.ndarray) -> np.ndarray:
        velocity = np.asarray(velocity, dtype=float).copy()
        limit = self.config.observer_visual_servo_max_angular_speed_rad_s
        norm = float(np.linalg.norm(velocity))
        if limit is not None and norm > limit:
            velocity = velocity * (limit / norm)
        return velocity

    def _observer_visual_servo_tasks(
        self,
        state: RobotSystemState,
        observer_name: str,
        *,
        avoidance_tasks: tuple[WholeBodyTask, ...],
    ) -> tuple[list[WholeBodyTask], np.ndarray, np.ndarray, float, np.ndarray]:
        roi = _metadata_vector(state.metadata, "visual_servo_roi_world", 3)
        if roi is None and self.target.observer_roi_position_world is not None:
            roi = self.target.observer_roi_position_world.copy()
        if roi is None:
            return (
                [],
                np.full(2, np.nan, dtype=float),
                np.zeros(3, dtype=float),
                float("nan"),
                np.zeros(3, dtype=float),
            )
        observer_state = state.arms[observer_name]
        camera_position = _metadata_vector(
            state.metadata,
            "visual_servo_camera_position_world",
            3,
        )
        if camera_position is None:
            camera_position = observer_state.tip_pose_world.position.copy()
        camera_quat = _metadata_vector(
            state.metadata,
            "visual_servo_camera_quat_world_wxyz",
            4,
        )
        if camera_quat is None:
            camera_quat = observer_state.tip_pose_world.quat.copy()
        camera_rotation = quaternion_wxyz_to_rotation_matrix(camera_quat)
        camera_right_world = camera_rotation[:, 0]
        camera_up_world = camera_rotation[:, 1]
        camera_forward_world = -camera_rotation[:, 2]
        measured_depth = _metadata_float(state.metadata, "visual_servo_depth_m")
        geometric_depth = float(np.dot(roi - camera_position, camera_forward_world))
        depth = measured_depth if np.isfinite(measured_depth) else geometric_depth
        depth_error = depth - self.config.observer_visual_servo_depth_target_m
        position_velocity = self._limit_observer_visual_linear_velocity(
            self.config.observer_visual_servo_depth_gain
            * depth_error
            * camera_forward_world
        )
        tasks: list[WholeBodyTask] = []
        if np.linalg.norm(position_velocity) > 1.0e-12:
            tasks.append(
                WholeBodyTask(
                    name="observer_visual_depth",
                    jacobian=self._observer_only_jacobian(
                        self._arm_system_jacobian(state, observer_name),
                        observer_name,
                    ),
                    target_velocity=position_velocity,
                    weight=self.solver.weight_for("observer_tracking"),
                )
            )

        pixel_error = _metadata_vector(
            state.metadata,
            "visual_servo_pixel_error_px",
            2,
            allow_nan=True,
        )
        if pixel_error is None:
            pixel_error = np.full(2, np.nan, dtype=float)
        normalized_error = _metadata_vector(
            state.metadata,
            "visual_servo_normalized_error",
            2,
            allow_nan=True,
        )
        target_visible = bool(state.metadata.get("visual_servo_target_visible", False))
        if (
            target_visible
            and normalized_error is not None
            and np.all(np.isfinite(normalized_error))
        ):
            desired_direction = (
                camera_forward_world
                + normalized_error[0] * camera_right_world
                - normalized_error[1] * camera_up_world
            )
            desired_direction_norm = float(np.linalg.norm(desired_direction))
            if desired_direction_norm <= 1.0e-12:
                angular_velocity = np.zeros(3, dtype=float)
            else:
                desired_direction /= desired_direction_norm
                angular_velocity = (
                    self.config.observer_visual_servo_center_gain
                    * np.cross(camera_forward_world, desired_direction)
                )
        else:
            desired_direction = roi - camera_position
            desired_direction_norm = float(np.linalg.norm(desired_direction))
            if desired_direction_norm <= 1.0e-12:
                angular_velocity = np.zeros(3, dtype=float)
            else:
                desired_direction /= desired_direction_norm
                angular_velocity = (
                    self.config.observer_visual_servo_center_gain
                    * np.cross(camera_forward_world, desired_direction)
                )
        angular_velocity = self._limit_observer_visual_angular_velocity(
            angular_velocity
        )
        if np.linalg.norm(angular_velocity) > 1.0e-12:
            jacobian = self._observer_only_jacobian(
                self._arm_orientation_system_jacobian(state, observer_name),
                observer_name,
            )
            avoidance_jacobian = _stack_task_jacobians(avoidance_tasks)
            if avoidance_jacobian is not None:
                jacobian = self._project_observer_task_to_avoidance_nullspace(
                    jacobian,
                    avoidance_jacobian,
                    observer_name,
                )
            if np.linalg.norm(jacobian) > self.solver.config.singularity.rank_tolerance:
                tasks.append(
                    WholeBodyTask(
                        name="observer_visual_centering",
                        jacobian=jacobian,
                        target_velocity=angular_velocity,
                        weight=self.config.observer_look_at_weight,
                    )
                )
        return (
            tasks,
            pixel_error,
            angular_velocity,
            float(depth_error),
            position_velocity,
        )

    def _observer_look_at_task(
        self,
        state: RobotSystemState,
        executor_name: str,
        observer_name: str,
        *,
        avoidance_tasks: tuple[WholeBodyTask, ...],
    ) -> tuple[WholeBodyTask | None, np.ndarray, np.ndarray]:
        observer_state = state.arms[observer_name]
        executor_tip = state.arms[executor_name].tip_pose_world.position
        observer_tip = observer_state.tip_pose_world.position
        target_direction = executor_tip - observer_tip
        target_distance = float(np.linalg.norm(target_direction))
        if target_distance <= 1.0e-12:
            nan_error = np.full(3, np.nan, dtype=float)
            return None, nan_error, np.zeros(3, dtype=float)
        target_direction /= target_distance
        target_orientation = look_rotation_quaternion_wxyz(target_direction)
        error = quaternion_error_rotation_vector(
            target_orientation,
            observer_state.tip_pose_world.quat,
        )
        velocity = self._limit_observer_look_at_velocity(
            self.config.observer_look_at_gain * error
        )
        if np.linalg.norm(velocity) <= 1.0e-12:
            return None, error, velocity
        jacobian = self._observer_only_jacobian(
            self._arm_orientation_system_jacobian(state, observer_name),
            observer_name,
        )
        avoidance_jacobian = _stack_task_jacobians(avoidance_tasks)
        if avoidance_jacobian is not None:
            jacobian = self._project_observer_task_to_avoidance_nullspace(
                jacobian,
                avoidance_jacobian,
                observer_name,
            )
        if np.linalg.norm(jacobian) <= self.solver.config.singularity.rank_tolerance:
            return None, error, velocity
        return (
            WholeBodyTask(
                name="observer_quaternion_look_at_executor_tip",
                jacobian=jacobian,
                target_velocity=velocity,
                weight=self.config.observer_look_at_weight,
            ),
            error,
            velocity,
        )

    def _observer_tip_forward_point_jacobian(
        self,
        state: RobotSystemState,
        observer_name: str,
        look_at_distance_m: float,
    ) -> np.ndarray:
        arm_config = self.assembly.arms[observer_name]
        arm_state = state.arms[observer_name]
        model = self.solver.layout.bending_models[observer_name]
        bending = model.estimate(arm_state.tendon_displacement_m)
        mount = state.base.pose.compose(arm_config.mount_pose)

        def point_from_bending(values: np.ndarray) -> np.ndarray:
            q = model.to_q(values)
            tip_pose_mount = forward_kinematics(
                q,
                arm_config.spatial_arm.params,
                kinematics_mode=self.solver.config.kinematics_mode,
            ).tip_pose
            tip_pose_world = mount.as_matrix() @ tip_pose_mount
            return (
                tip_pose_world[:3, 3]
                + look_at_distance_m * tip_pose_world[:3, 2]
            )

        local = np.empty((3, model.bending_size), dtype=float)
        for index in range(model.bending_size):
            step = 1.0e-6 * max(1.0, abs(float(bending[index])))
            plus = bending.copy()
            minus = bending.copy()
            plus[index] += step
            minus[index] -= step
            local[:, index] = (
                point_from_bending(plus) - point_from_bending(minus)
            ) / (2.0 * step)
        result = np.zeros((3, self.solver.layout.size), dtype=float)
        result[:, self.solver.layout.arms[observer_name]] = local
        return result

    def _project_observer_task_to_avoidance_nullspace(
        self,
        task_jacobian: np.ndarray,
        avoidance_jacobian: np.ndarray,
        observer_name: str,
    ) -> np.ndarray:
        observer_slice = self.solver.layout.arms[observer_name]
        active_avoidance = avoidance_jacobian[:, observer_slice]
        active_task = task_jacobian[:, observer_slice]
        if not active_avoidance.size or np.linalg.norm(active_avoidance) <= 0.0:
            return task_jacobian
        nullspace = (
            np.eye(active_avoidance.shape[1], dtype=float)
            - np.linalg.pinv(
                active_avoidance,
                rcond=self.solver.config.singularity.minimum_singular_value,
            )
            @ active_avoidance
        )
        projected = np.zeros_like(task_jacobian)
        projected[:, observer_slice] = active_task @ nullspace
        return projected

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
            kinematics_mode=self.solver.config.kinematics_mode,
        )
        world_mount = state.base.pose.compose(arm_config.mount_pose)
        jacobian_world = rotate_position_jacobian_to_world(
            jacobian_local,
            world_mount.as_matrix()[:3, :3],
        )
        if (
            arm_state.tool_pose_world is not None
            and np.allclose(
                arm_state.tip_pose_world.position,
                arm_state.tool_pose_world.position,
                rtol=0.0,
                atol=1.0e-12,
            )
        ):
            raw_tip_pose = np.asarray(
                arm_state.metadata.get("arm_tip_pose_world", np.eye(4)),
                dtype=float,
            )
            if raw_tip_pose.shape == (4, 4):
                offset_world = (
                    arm_state.tool_pose_world.position - raw_tip_pose[:3, 3]
                )
                angular_local = bending_orientation_jacobian(
                    q,
                    arm_config.spatial_arm.params,
                    arm_config.spatial_arm.tendons,
                    kinematics_mode=self.solver.config.kinematics_mode,
                )
                angular_world = rotate_angular_jacobian_to_world(
                    angular_local,
                    world_mount.as_matrix()[:3, :3],
                )
                jacobian_world = jacobian_world - skew(offset_world) @ angular_world
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

    def _arm_orientation_system_jacobian(
        self,
        state: RobotSystemState,
        arm_name: str,
    ) -> np.ndarray:
        arm_config = self.assembly.arms[arm_name]
        arm_state = state.arms[arm_name]
        model = self.solver.layout.bending_models[arm_name]
        q = model.to_q(model.estimate(arm_state.tendon_displacement_m))
        jacobian_local = bending_orientation_jacobian(
            q,
            arm_config.spatial_arm.params,
            arm_config.spatial_arm.tendons,
            kinematics_mode=self.solver.config.kinematics_mode,
        )
        world_mount = state.base.pose.compose(arm_config.mount_pose)
        jacobian_world = rotate_angular_jacobian_to_world(
            jacobian_local,
            world_mount.as_matrix()[:3, :3],
        )
        return assemble_whole_body_jacobian(
            self.solver.layout,
            arm_name,
            base_orientation_jacobian_world(),
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

    def _executor_force_control_task(
        self,
        executor_jacobian: np.ndarray,
    ) -> WholeBodyTask | None:
        normal = self.target.executor_force_normal_world
        velocity = float(self.target.executor_force_velocity_mps)
        if normal is None or abs(velocity) <= 1.0e-12:
            return None
        jacobian = normal.reshape(1, 3) @ executor_jacobian
        if np.linalg.norm(jacobian) <= self.solver.config.singularity.rank_tolerance:
            return None
        return WholeBodyTask(
            name="executor_normal_force_control",
            jacobian=jacobian,
            target_velocity=np.array([velocity], dtype=float),
            weight=self.target.executor_force_control_weight,
        )

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
        selected_pairs = self._select_collision_pairs(distances)
        closest_observer_offset, closest_executor_offset = selected_pairs[0]
        observer_index = int(closest_observer_offset + 1)
        executor_index = int(closest_executor_offset + 1)
        executor_point = executor_centerline[executor_index]
        observer_point = observer_centerline[observer_index]
        separation = observer_point - executor_point
        distance = float(np.linalg.norm(separation))
        normal = (
            separation / distance
            if distance > 1.0e-12
            else np.array([0.0, -1.0, 0.0], dtype=float)
        )
        activation_distance = self.config.inter_arm_influence_distance_m
        rows: list[np.ndarray] = []
        targets: list[float] = []
        pair_observer_indices: list[int] = []
        pair_executor_indices: list[int] = []
        pair_distances: list[float] = []
        pair_speeds: list[float] = []
        for observer_offset, executor_offset in selected_pairs:
            pair_observer_index = int(observer_offset + 1)
            pair_executor_index = int(executor_offset + 1)
            pair_observer_point = observer_centerline[pair_observer_index]
            pair_executor_point = executor_centerline[pair_executor_index]
            pair_separation = pair_observer_point - pair_executor_point
            pair_distance = float(np.linalg.norm(pair_separation))
            pair_normal = (
                pair_separation / pair_distance
                if pair_distance > 1.0e-12
                else np.array([0.0, -1.0, 0.0], dtype=float)
            )
            desired_speed = self.config.inter_arm_avoidance_gain * max(
                activation_distance - pair_distance,
                0.0,
            )
            avoidance_speed_limit = self.config.inter_arm_max_avoidance_speed_mps
            if avoidance_speed_limit is not None:
                desired_speed = min(float(avoidance_speed_limit), desired_speed)
            pair_observer_indices.append(pair_observer_index)
            pair_executor_indices.append(pair_executor_index)
            pair_distances.append(pair_distance)
            pair_speeds.append(float(desired_speed))
            if desired_speed <= 0.0:
                continue
            observer_point_jacobian = self._centerline_system_jacobian(
                state,
                observer_name,
                observer_q,
                observer_mount,
                pair_observer_index,
                pair_observer_point,
            )
            relative_jacobian = pair_normal[None, :] @ self._observer_only_jacobian(
                observer_point_jacobian,
                observer_name,
            )
            if (
                np.linalg.norm(relative_jacobian)
                <= self.solver.config.singularity.rank_tolerance
            ):
                continue
            rows.append(relative_jacobian)
            targets.append(float(desired_speed))
        task = None
        if rows:
            task = WholeBodyTask(
                name="executor_observer_collision_avoidance",
                jacobian=np.vstack(rows),
                target_velocity=np.asarray(targets, dtype=float),
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
            desired_speed_mps=(
                0.0 if not pair_speeds else float(np.max(pair_speeds))
            ),
            pair_observer_indices=np.asarray(pair_observer_indices, dtype=int),
            pair_executor_indices=np.asarray(pair_executor_indices, dtype=int),
            pair_distances_m=np.asarray(pair_distances, dtype=float),
            pair_desired_speeds_mps=np.asarray(pair_speeds, dtype=float),
        )

    def _select_collision_pairs(
        self,
        distances: np.ndarray,
    ) -> list[tuple[int, int]]:
        flat_order = np.argsort(distances, axis=None)
        selected: list[tuple[int, int]] = []
        separation = self.config.inter_arm_collision_pair_index_separation
        for flat_index in flat_order:
            observer_offset, executor_offset = np.unravel_index(
                int(flat_index),
                distances.shape,
            )
            if any(
                abs(observer_offset - chosen_observer) < separation
                and abs(executor_offset - chosen_executor) < separation
                for chosen_observer, chosen_executor in selected
            ):
                continue
            selected.append((int(observer_offset), int(executor_offset)))
            if len(selected) >= self.config.inter_arm_collision_pair_count:
                break
        if not selected:
            observer_offset, executor_offset = np.unravel_index(
                int(np.argmin(distances)),
                distances.shape,
            )
            selected.append((int(observer_offset), int(executor_offset)))
        return selected

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
            kinematics_mode=self.solver.config.kinematics_mode,
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
            kinematics_mode=self.solver.config.kinematics_mode,
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
        clearance: _SceneClearanceData,
    ) -> WholeBodyTask | None:
        query = clearance.query
        if query.distance_m >= self.config.engine_influence_distance_m:
            return None
        point_jacobian = self._centerline_system_jacobian(
            state,
            arm_name,
            clearance.q,
            clearance.mount,
            clearance.centerline_index,
            clearance.centerline[clearance.centerline_index],
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

    def _engine_clearance(
        self,
        state: RobotSystemState,
        arm_name: str,
    ) -> _SceneClearanceData:
        if self.scene_query is None:
            raise RuntimeError("Scene clearance requires a scene query.")
        centerline, q, mount = self._world_centerline(state, arm_name)
        queries = [self.scene_query.nearest_distance(point) for point in centerline]
        centerline_index = int(np.argmin([query.distance_m for query in queries]))
        return _SceneClearanceData(
            query=queries[centerline_index],
            centerline=centerline,
            q=q,
            mount=mount,
            centerline_index=centerline_index,
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


def _quat4(values: np.ndarray, name: str) -> np.ndarray:
    result = np.asarray(values, dtype=float)
    if result.shape != (4,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite vector with shape (4,).")
    norm = float(np.linalg.norm(result))
    if norm <= 1.0e-12:
        raise ValueError(f"{name} must have non-zero length.")
    return (result / norm).copy()


def _scene_clearance_distance(
    clearances: dict[str, _SceneClearanceData],
    arm_name: str | None,
) -> float:
    if arm_name is None or arm_name not in clearances:
        return float("nan")
    return float(clearances[arm_name].query.distance_m)


def _stack_task_jacobians(
    tasks: tuple[WholeBodyTask, ...],
) -> np.ndarray | None:
    rows = [task.jacobian for task in tasks if task.jacobian.size]
    if not rows:
        return None
    return np.vstack(rows)


def _metadata_vector(
    metadata: dict[str, object],
    key: str,
    size: int,
    *,
    allow_nan: bool = False,
) -> np.ndarray | None:
    value = metadata.get(key)
    if value is None:
        return None
    try:
        result = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return None
    if result.shape != (size,):
        return None
    if allow_nan:
        return None if np.any(np.isinf(result)) else result.copy()
    return result.copy() if np.all(np.isfinite(result)) else None


def _metadata_float(metadata: dict[str, object], key: str) -> float:
    value = metadata.get(key)
    if value is None:
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _priority_levels(
    configured_levels,
    tasks_by_group: dict[str, tuple[WholeBodyTask, ...]],
) -> tuple[tuple[WholeBodyTask, ...], ...]:
    levels: list[tuple[WholeBodyTask, ...]] = []
    for level in configured_levels:
        tasks: list[WholeBodyTask] = []
        for group in level.tasks:
            tasks.extend(tasks_by_group.get(group, ()))
        levels.append(tuple(tasks))
    if not levels:
        return ((),)
    return tuple(levels)
