"""Scenario composition root and primary simulation application API."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from continuum_sim.application.scenario import (
    ScenarioConfig,
    ScenarioObserverControlConfig,
    ScenarioSceneAvoidanceConfig,
    ScenarioTrackingControlConfig,
    load_scenario_config,
)
from continuum_sim.backends.analytic_system_backend import AnalyticSystemBackend
from continuum_sim.backends.mujoco_system_backend import MujocoSystemBackend
from continuum_sim.config import load_mujoco_config
from continuum_sim.control.scenario_controllers import (
    EngineCleaningSystemController,
    NavigationController,
    TimedTrajectoryTrackingController,
    WaypointTrackingController,
    WipingController,
    ZeroSystemController,
)
from continuum_sim.control.contact_triggered_admittance import (
    ContactTriggeredAdmittanceConfig,
)
from continuum_sim.control.tendon_rate_control import BendingRateServoConfig
from continuum_sim.control.wiping_force_strategies import (
    ContactDistanceStrategy,
    ContactTriggeredAdmittanceStrategy,
    DynamicAdaptiveImpedanceStrategy,
    KinematicHybridForceStrategy,
)
from continuum_sim.control.staged_engine_navigation import (
    StagedEngineNavigationController,
)
from continuum_sim.control.staged_engine_tracking import (
    StagedEngineTrackingController,
)
from continuum_sim.control.staged_navigation import StagedNavigationController
from continuum_sim.control.coordinated_tracking import CoordinatedTrackingConfig
from continuum_sim.control.whole_body_controller import WholeBodyControllerConfig
from continuum_sim.dynamics import load_pcc_dynamics_config
from continuum_sim.kinematics.whole_body import SingularityConfig
from continuum_sim.model.base_pose import look_rotation_quaternion_wxyz
from continuum_sim.model.robot_assembly import load_robot_assembly_config
from continuum_sim.runtime.hooks import (
    ControllerCompletionHook,
    LiveTendonPanelHook,
    LiveDiagnosticsPanelHook,
    LiveWipingForcePanelHook,
    MatplotlibSystemViewerHook,
    MujocoLiveVideoRecorderHook,
    MujocoViewerHook,
    MujocoReplayRecorderHook,
    StateRecorderHook,
    TendonDiagnosticHook,
)
from continuum_sim.io.scenario_artifacts import save_scenario_artifacts
from continuum_sim.runtime.simulation_loop import (
    SimulationLoop,
    SimulationLoopConfig,
    SimulationLoopResult,
)
from continuum_sim.scenes.engine_mjcf_adapter import (
    inject_engine_scene,
    rebase_mjcf_file_assets,
    retain_spatial_arm,
)
from continuum_sim.scenes.engine_query import EnginePrimitiveSceneQuery
from continuum_sim.scenes.engine_scene import load_engine_scene_config
from continuum_sim.scenes.scene_config import (
    InspectionTargetConfig,
    load_navigation_scene_config,
)
from continuum_sim.scenes.structured_query import StructuredSceneQuery
from continuum_sim.scenes.scene_builder import (
    inject_structured_scene,
    lock_mobile_base_freejoint,
)
from continuum_sim.tasks.engine_cleaning_path import build_engine_cleaning_plan
from continuum_sim.tasks.engine_navigation import resolve_engine_navigation_plan
from continuum_sim.tasks.navigation_mission import resolve_navigation_waypoints
from continuum_sim.tasks.trajectory_generation import (
    generate_trajectory_waypoints,
    prepend_tracking_approach,
)
from continuum_sim.tasks.wiping_path import build_wiping_plan


@dataclass
class SimulationApplication:
    """Fully composed backend, controller, and hook lifecycle."""

    config: ScenarioConfig
    loop: SimulationLoop
    hooks_by_name: dict[str, object]
    last_artifacts: object | None = None

    @classmethod
    def from_yaml(cls, path: str | Path) -> "SimulationApplication":
        return cls.from_config(load_scenario_config(path))

    @classmethod
    def from_config(cls, config: ScenarioConfig) -> "SimulationApplication":
        assembly = load_robot_assembly_config(config.assembly_config_path)
        engine_scene = (
            None
            if config.scene.engine_config_path is None
            else load_engine_scene_config(config.scene.engine_config_path)
        )
        structured_scene = (
            None
            if config.scene.structured_config_path is None
            else load_navigation_scene_config(config.scene.structured_config_path)
        )
        if engine_scene is not None and structured_scene is not None:
            raise ValueError("A scenario cannot select engine and structured scenes together.")
        if config.backend.type == "analytic":
            backend = AnalyticSystemBackend(
                assembly,
                kinematics_mode=config.backend.kinematics_mode,
            )
        else:
            backend = _build_mujoco_backend(
                config,
                assembly,
                engine_scene,
                structured_scene,
            )
        if engine_scene is not None:
            scene_query = EnginePrimitiveSceneQuery(engine_scene)
        elif structured_scene is not None:
            scene_query = StructuredSceneQuery(structured_scene)
        else:
            scene_query = None
        task_plan = _resolve_task_plan(config, assembly, engine_scene, structured_scene)
        if config.task.type == "idle":
            controller = ZeroSystemController(assembly)
        elif config.task.type == "engine_navigation":
            if engine_scene is None or config.task.engine_navigation is None:
                raise ValueError(
                    "engine_navigation requires an engine scene and task specification."
                )
            plan = resolve_engine_navigation_plan(
                config.task.engine_navigation,
                engine_scene,
                assembly,
            )
            tracking = config.task.tracking_control
            controller = StagedEngineNavigationController(
                assembly,
                plan,
                config.task.engine_navigation,
                scene_query=scene_query,
                waypoint_tolerance_m=config.task.waypoint_tolerance_m,
                min_clearance_m=config.task.min_clearance_m,
                terminate_on_clearance_violation=(
                    config.task.terminate_on_clearance_violation
                ),
                observer_control_mode=config.task.observer_control_mode,
                controller_dt_s=config.runtime.controller_dt_s,
                low_level_coordinated_config=(
                    _tracking_coordinated_config(
                        tracking,
                        config.task.observer_control,
                        config.task.scene_avoidance,
                    )
                ),
                low_level_solver_config=_tracking_solver_config(tracking),
            )
        elif config.task.type == "navigation":
            tracking = config.task.tracking_control
            navigation_kwargs = {
                "waypoint_tolerance_m": config.task.waypoint_tolerance_m,
                "observer_roi_world": config.task.observer_roi_world,
                "observer_control_mode": config.task.observer_control_mode,
                "scene_query": scene_query,
                "min_clearance_m": config.task.min_clearance_m,
                "terminate_on_clearance_violation": (
                    config.task.terminate_on_clearance_violation
                ),
                "target_advance_mode": config.task.target_advance_mode,
                "controller_dt_s": config.runtime.controller_dt_s,
                "advance_time_s": config.task.advance_time_s,
                "advance_steps": config.task.advance_steps,
                "max_steps_per_waypoint": tracking.max_steps_per_waypoint,
                "executor_position_gain": tracking.executor_position_gain,
                "observer_position_gain": tracking.observer_position_gain,
                "feedforward_speed_mps": tracking.feedforward_speed_mps,
                "max_target_speed_mps": _tracking_target_speed_limit(tracking),
                "waypoint_orientations_world_wxyz": (
                    task_plan["waypoint_orientations_world_wxyz"]
                ),
                "orientation_tolerance_rad": (
                    config.task.orientation_tolerance_rad
                ),
                "solver_config": _tracking_solver_config(tracking),
                "enforce_backend_tendon_limits": (
                    tracking.enforce_backend_tendon_limits
                ),
                "coordinated_config": _tracking_coordinated_config(
                    tracking,
                    config.task.observer_control,
                    config.task.scene_avoidance,
                ),
                "control_type": config.task.navigation_control_type,
                "cbf_gain": config.task.navigation_cbf_gain,
                "cbf_influence_distance_m": (
                    config.task.navigation_cbf_influence_distance_m
                ),
            }
            if tracking.stage_mobile_base:
                controller = StagedNavigationController(
                    assembly,
                    task_plan["waypoints_world"],
                    **navigation_kwargs,
                    base_position_gain=tracking.base_position_gain,
                    base_orientation_gain=tracking.base_orientation_gain,
                    waypoint_directions_world=(
                        config.task.waypoint_directions_world
                    ),
                    base_position_tolerance_m=tracking.base_position_tolerance_m,
                    base_orientation_tolerance_rad=(
                        tracking.base_orientation_tolerance_rad
                    ),
                    base_approach_standoff_m=(
                        tracking.base_approach_standoff_m
                    ),
                    base_approach_z_bias=tracking.base_approach_z_bias,
                    intermediate_waypoints_per_waypoint=(
                        tracking.intermediate_waypoints_per_waypoint
                    ),
                )
            else:
                controller = NavigationController(
                    assembly,
                    task_plan["waypoints_world"],
                    **navigation_kwargs,
                )
        elif config.task.type == "wiping":
            tracking = config.task.tracking_control
            controller = WipingController(
                assembly,
                task_plan["waypoints_world"],
                waypoint_tolerance_m=config.task.waypoint_tolerance_m,
                scene_query=scene_query,
                surface_normal_world=task_plan["surface_normal_world"],
                surface_point_world=task_plan["surface_point_world"],
                target_contact_distance_m=config.task.target_contact_distance_m,
                contact_tolerance_m=config.task.contact_tolerance_m,
                target_advance_mode=config.task.target_advance_mode,
                controller_dt_s=config.runtime.controller_dt_s,
                advance_time_s=config.task.advance_time_s,
                advance_steps=config.task.advance_steps,
                phases=task_plan["phases"],
                target_force_n=task_plan["target_force_n"],
                control_type=config.task.wiping_control_type,
                normal_force_gain=config.task.normal_force_gain,
                force_proxy_stiffness_n_m=config.task.force_proxy_stiffness_n_m,
                max_contact_force_n=config.task.max_contact_force_n,
                force_strategy=_build_wiping_force_strategy(config, assembly),
                tracking_mode=tracking.tracking_mode,
                trajectory_duration_s=tracking.trajectory_duration_s,
                approach_samples=tracking.approach_samples,
                observer_control_mode=config.task.observer_control_mode,
                executor_position_gain=tracking.executor_position_gain,
                observer_position_gain=tracking.observer_position_gain,
                feedforward_speed_mps=tracking.feedforward_speed_mps,
                max_target_speed_mps=_tracking_target_speed_limit(tracking),
                solver_config=_tracking_solver_config(tracking),
                enforce_backend_tendon_limits=tracking.enforce_backend_tendon_limits,
                coordinated_config=_tracking_coordinated_config(
                    tracking,
                    config.task.observer_control,
                    config.task.scene_avoidance,
                ),
            )
        elif config.task.type == "engine_cleaning":
            tracking = config.task.tracking_control
            controller = EngineCleaningSystemController(
                assembly,
                task_plan["waypoints_world"],
                task_plan["normals_world"],
                task_plan["phases"],
                task_plan["target_force_n"],
                task_plan["standoff_distance_m"],
                scene_query=scene_query,
                gains=(
                    config.task.engine_cleaning_control
                    if config.task.engine_cleaning_control is not None
                    else _default_engine_cleaning_gains(config)
                ),
                controller_dt_s=config.runtime.controller_dt_s,
                observer_roi_world=config.task.observer_roi_world,
                observer_control_mode=config.task.observer_control_mode,
                executor_position_gain=tracking.executor_position_gain,
                observer_position_gain=tracking.observer_position_gain,
                max_target_speed_mps=_tracking_target_speed_limit(tracking),
                solver_config=_tracking_solver_config(tracking),
                enforce_backend_tendon_limits=(
                    tracking.enforce_backend_tendon_limits
                ),
                coordinated_config=_tracking_coordinated_config(
                    tracking,
                    config.task.observer_control,
                    config.task.scene_avoidance,
                ),
            )
        else:
            tracking = config.task.tracking_control
            if tracking.tracking_mode == "time":
                timed_controller_type = (
                    StagedEngineTrackingController
                    if tracking.stage_mobile_base
                    else TimedTrajectoryTrackingController
                )
                controller = timed_controller_type(
                    assembly,
                    task_plan["waypoints_world"],
                    trajectory_duration_s=float(tracking.trajectory_duration_s),
                    waypoint_tolerance_m=config.task.waypoint_tolerance_m,
                    observer_roi_world=config.task.observer_roi_world,
                    observer_control_mode=config.task.observer_control_mode,
                    loop=config.task.loop,
                    scene_query=scene_query,
                    approach_mask=task_plan["approach_mask"],
                    source_waypoint_index=task_plan["source_waypoint_index"],
                    executor_position_gain=tracking.executor_position_gain,
                    observer_position_gain=tracking.observer_position_gain,
                    max_target_speed_mps=_tracking_target_speed_limit(tracking),
                    solver_config=_tracking_solver_config(tracking),
                    enforce_backend_tendon_limits=tracking.enforce_backend_tendon_limits,
                    coordinated_config=_tracking_coordinated_config(
                        tracking,
                        config.task.observer_control,
                        config.task.scene_avoidance,
                    ),
                    **(
                        {
                            "base_position_gain": tracking.base_position_gain,
                            "base_orientation_gain": tracking.base_orientation_gain,
                            "base_position_tolerance_m": (
                                tracking.base_position_tolerance_m
                            ),
                            "base_orientation_tolerance_rad": (
                                tracking.base_orientation_tolerance_rad
                            ),
                        }
                        if tracking.stage_mobile_base
                        else {}
                    ),
                )
            else:
                if tracking.stage_mobile_base:
                    raise ValueError(
                        "tracking_control.stage_mobile_base requires "
                        "tracking_mode='time'."
                    )
                controller = WaypointTrackingController(
                    assembly,
                    task_plan["waypoints_world"],
                    waypoint_tolerance_m=config.task.waypoint_tolerance_m,
                    waypoint_orientations_world_wxyz=(
                        task_plan["waypoint_orientations_world_wxyz"]
                    ),
                    orientation_tolerance_rad=(
                        config.task.orientation_tolerance_rad
                    ),
                    observer_roi_world=config.task.observer_roi_world,
                    observer_control_mode=config.task.observer_control_mode,
                    loop=config.task.loop,
                    target_advance_mode=config.task.target_advance_mode,
                    controller_dt_s=config.runtime.controller_dt_s,
                    advance_time_s=config.task.advance_time_s,
                    advance_steps=config.task.advance_steps,
                    max_steps_per_waypoint=tracking.max_steps_per_waypoint,
                    scene_query=scene_query,
                    approach_mask=task_plan["approach_mask"],
                    source_waypoint_index=task_plan["source_waypoint_index"],
                    executor_position_gain=tracking.executor_position_gain,
                    observer_position_gain=tracking.observer_position_gain,
                    feedforward_speed_mps=tracking.feedforward_speed_mps,
                    max_target_speed_mps=_tracking_target_speed_limit(tracking),
                    solver_config=_tracking_solver_config(tracking),
                    enforce_backend_tendon_limits=tracking.enforce_backend_tendon_limits,
                    coordinated_config=_tracking_coordinated_config(
                        tracking,
                        config.task.observer_control,
                        config.task.scene_avoidance,
                    ),
                )
        hooks: list[object] = []
        hooks_by_name: dict[str, object] = {}
        if config.hooks.recorder:
            hooks_by_name["recorder"] = StateRecorderHook()
        if config.hooks.tendon_debug:
            hooks_by_name["tendon_debug"] = TendonDiagnosticHook(
                stride=config.hooks.tendon_debug_stride
            )
        if config.hooks.show_live_tendon_panel:
            hooks_by_name["live_tendon_panel"] = LiveTendonPanelHook(
                stride=config.hooks.live_tendon_panel_stride,
                history_points=config.hooks.live_force_panel_history_points,
            )
        if config.hooks.show_live_force_panel:
            hooks_by_name["live_force_panel"] = LiveWipingForcePanelHook(
                stride=config.hooks.live_force_panel_stride,
                history_points=config.hooks.live_force_panel_history_points,
            )
        if config.hooks.show_live_diagnostics_panel:
            hooks_by_name["live_diagnostics_panel"] = LiveDiagnosticsPanelHook(
                stride=config.hooks.live_diagnostics_panel_stride,
                history_points=config.hooks.live_diagnostics_panel_history_points,
            )
        if (
            config.backend.type == "mujoco"
            and config.artifacts.enabled
            and _video_artifacts_enabled(config.artifacts)
            and config.artifacts.video_mode == "live_mujoco"
        ):
            hooks_by_name["live_mujoco_video"] = MujocoLiveVideoRecorderHook(
                backend,
                _live_mujoco_pending_video_paths(config),
                fps=config.artifacts.video_fps,
                stride=config.artifacts.video_stride,
                width=backend.config.rendering.offscreen_width,
                height=backend.config.rendering.offscreen_height,
            )
        if (
            config.backend.type == "mujoco"
            and config.artifacts.enabled
            and config.artifacts.video_mode == "replay"
        ):
            hooks_by_name["mujoco_replay"] = MujocoReplayRecorderHook(backend)
        if config.hooks.viewer == "matplotlib":
            hooks_by_name["viewer"] = MatplotlibSystemViewerHook(
                keep_open=config.hooks.keep_viewer_open
            )
        elif config.hooks.viewer == "mujoco":
            if config.backend.type != "mujoco":
                raise ValueError("The MuJoCo viewer requires a MuJoCo backend.")
            hooks_by_name["viewer"] = MujocoViewerHook(
                backend,
                keep_open=config.hooks.keep_viewer_open,
            )
        if hasattr(controller, "done"):
            hooks_by_name["completion"] = ControllerCompletionHook(controller)
        hooks.extend(hooks_by_name.values())
        loop = SimulationLoop(
            backend,
            controller,
            SimulationLoopConfig(
                controller_dt_s=config.runtime.controller_dt_s,
                n_substeps=config.runtime.n_substeps,
                max_steps=config.runtime.max_steps,
            ),
            hooks=tuple(hooks),
        )
        return cls(config=config, loop=loop, hooks_by_name=hooks_by_name)

    def run(self) -> SimulationLoopResult:
        result = self.loop.run()
        self.last_artifacts = save_scenario_artifacts(self, result)
        return result


def _build_mujoco_backend(config, assembly, engine_scene, structured_scene):
    backend = config.backend
    if (
        backend.mujoco_config_path is None
        or backend.source_xml_path is None
        or backend.generated_xml_path is None
    ):
        raise ValueError(
            "MuJoCo scenarios require mujoco_config_path, source_xml_path, "
            "and generated_xml_path."
        )
    mujoco_config = load_mujoco_config(backend.mujoco_config_path)
    output_path = backend.generated_xml_path
    tree = ET.parse(backend.source_xml_path)
    root = tree.getroot()
    rebase_mjcf_file_assets(
        root,
        backend.source_xml_path.parent,
        output_path.parent,
    )
    if backend.retain_arm is not None:
        retain_spatial_arm(root, backend.retain_arm)
    if engine_scene is not None:
        inject_engine_scene(
            root,
            engine_scene,
            output_dir=output_path.parent,
            include_visual_mesh=True,
            include_collision_mesh=False,
            include_control_primitives=True,
        )
    visual_structured_scene = _structured_scene_for_mujoco_visuals(
        config,
        structured_scene,
    )
    if visual_structured_scene is not None:
        inject_structured_scene(root, visual_structured_scene)
    if assembly.base.control_mode == "fixed":
        lock_mobile_base_freejoint(root)
    _apply_mujoco_offscreen_rendering_config(root, mujoco_config)
    _apply_mujoco_tendon_position_actuator_config(root, mujoco_config)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(tree)
    tree.write(output_path, encoding="utf-8", xml_declaration=False)
    inner_loop = config.task.tracking_control.tendon_inner_loop
    tendon_rate_servo_config = None
    if inner_loop.mode == "bending_rate_servo":
        tendon_rate_servo_config = BendingRateServoConfig(
            rate_filter_time_constant_s=inner_loop.rate_filter_time_constant_s,
            feedforward_lead_time_s=inner_loop.feedforward_lead_time_s,
            rate_proportional_time_s=inner_loop.rate_proportional_time_s,
            rate_integral_gain=inner_loop.rate_integral_gain,
            anti_windup_gain=inner_loop.anti_windup_gain,
            enforce_target_lead_limit=inner_loop.enforce_target_lead_limit,
            max_target_lead_m=inner_loop.max_target_lead_m,
            soft_force_limit_n=inner_loop.soft_force_limit_n,
            hard_force_limit_n=inner_loop.hard_force_limit_n,
            zero_command_mode=inner_loop.zero_command_mode,
            zero_rate_tolerance_mps=inner_loop.zero_rate_tolerance_mps,
        )
    return MujocoSystemBackend(
        mujoco_config,
        assembly,
        xml_path=output_path,
        tendon_rate_servo_config=tendon_rate_servo_config,
        kinematics_mode=config.backend.kinematics_mode,
    )


def _structured_scene_for_mujoco_visuals(config, structured_scene):
    if structured_scene is None:
        return None
    task = config.task
    if task.waypoint_source != "waypoints_world" or task.waypoints_world.size == 0:
        return structured_scene
    waypoints = np.asarray(task.waypoints_world, dtype=float)
    if waypoints.ndim != 2 or waypoints.shape[1] != 3 or waypoints.shape[0] == 0:
        return structured_scene
    directions = task.waypoint_directions_world
    if directions.shape != waypoints.shape:
        directions = np.zeros((0, 3), dtype=float)
    existing_targets = structured_scene.inspection_targets
    visual_targets = []
    for index, waypoint in enumerate(waypoints):
        target_id = (
            existing_targets[index].id
            if index < len(existing_targets)
            else f"waypoint_{index + 1}"
        )
        visual_targets.append(
            InspectionTargetConfig(
                id=target_id,
                type="point",
                pos_m=waypoint.copy(),
                direction_world=(
                    directions[index].copy()
                    if directions.shape[0] == waypoints.shape[0]
                    else None
                ),
            )
        )
    return replace(structured_scene, inspection_targets=tuple(visual_targets))


def _apply_mujoco_offscreen_rendering_config(root, mujoco_config) -> None:
    rendering = mujoco_config.rendering
    visual = root.find("visual")
    if visual is None:
        visual = ET.Element("visual")
        insert_index = 1 if root.find("option") is not None else 0
        root.insert(insert_index, visual)
    global_visual = visual.find("global")
    if global_visual is None:
        global_visual = ET.Element("global")
        visual.insert(0, global_visual)
    global_visual.set("offwidth", str(int(rendering.offscreen_width)))
    global_visual.set("offheight", str(int(rendering.offscreen_height)))


def _apply_mujoco_tendon_position_actuator_config(root, mujoco_config) -> None:
    actuator_config = mujoco_config.actuators.tendon_position
    for position in root.findall("./actuator/position"):
        if position.get("tendon") is None:
            continue
        position.set("kp", f"{actuator_config.kp:g}")
        position.set("ctrllimited", str(actuator_config.ctrllimited).lower())
        if actuator_config.ctrllimited:
            position.set(
                "ctrlrange",
                f"{actuator_config.ctrlrange_m[0]:g} {actuator_config.ctrlrange_m[1]:g}",
            )
        elif "ctrlrange" in position.attrib:
            del position.attrib["ctrlrange"]
        position.set("forcelimited", str(actuator_config.forcelimited).lower())
        position.set(
            "forcerange",
            f"{actuator_config.forcerange_n[0]:g} {actuator_config.forcerange_n[1]:g}",
        )


def _build_wiping_force_strategy(config, assembly):
    strategy_type = config.task.force_strategy.type
    if strategy_type == "contact_distance":
        return ContactDistanceStrategy()
    if strategy_type == "kinematic_hybrid":
        return KinematicHybridForceStrategy()
    if strategy_type == "contact_triggered_admittance":
        admittance = config.task.admittance
        return ContactTriggeredAdmittanceStrategy(
            ContactTriggeredAdmittanceConfig(
                target_normal_force_n=admittance.target_normal_force_n,
                contact_force_threshold_n=admittance.contact_force_threshold_n,
                tangent_tolerance_m=admittance.tangent_tolerance_m,
                force_tolerance_n=admittance.force_tolerance_n,
                stable_steps_required=admittance.stable_steps_required,
                max_steps_per_target=admittance.max_steps_per_target,
                position_gain=admittance.position_gain,
                kp_force=admittance.kp_force,
                ki_force=admittance.ki_force,
                admittance_mass=admittance.admittance_mass,
                admittance_damping=admittance.admittance_damping,
                admittance_stiffness=admittance.admittance_stiffness,
                admittance_clip_m=admittance.admittance_clip_m,
                force_deadband_n=admittance.force_deadband_n,
                force_filter_alpha=admittance.force_filter_alpha,
                max_tangent_velocity_m_s=admittance.max_tangent_velocity_m_s,
                max_normal_velocity_m_s=admittance.max_normal_velocity_m_s,
                enforce_velocity_limits=admittance.enforce_velocity_limits,
            )
        )
    if strategy_type == "dynamic_adaptive_impedance":
        return DynamicAdaptiveImpedanceStrategy(
            assembly,
            dynamics_config_path=(
                None
                if config.task.dynamics_config_path is None
                else str(config.task.dynamics_config_path)
            ),
            kinematics_mode=config.backend.kinematics_mode,
        )
    raise ValueError(f"Unsupported wiping force strategy {strategy_type!r}.")


def _tracking_solver_config(
    tracking: ScenarioTrackingControlConfig,
) -> WholeBodyControllerConfig:
    return WholeBodyControllerConfig(
        executor_tracking_weight=tracking.executor_tracking_weight,
        observer_tracking_weight=tracking.observer_tracking_weight,
        executor_collision_avoidance_weight=(
            tracking.executor_collision_avoidance_weight
        ),
        base_regularization_weight=tracking.base_regularization_weight,
        tendon_regularization_weight=tracking.tendon_regularization_weight,
        singularity=SingularityConfig(
            rank_tolerance=tracking.rank_tolerance,
            minimum_singular_value=tracking.minimum_singular_value,
            nominal_damping=tracking.nominal_damping,
            maximum_damping=tracking.maximum_damping,
            minimum_velocity_scale=tracking.minimum_velocity_scale,
        ),
        decouple_arm_singularity=tracking.decouple_arm_singularity,
        singularity_strategy=tracking.singularity_strategy,
        enforce_base_velocity_limits=tracking.enforce_solver_velocity_limits,
        enforce_tendon_rate_limits=tracking.enforce_solver_velocity_limits,
        kinematics_mode=tracking.kinematics_mode,
    )


def _tracking_coordinated_config(
    tracking: ScenarioTrackingControlConfig,
    observer: ScenarioObserverControlConfig | None = None,
    scene_avoidance: ScenarioSceneAvoidanceConfig | None = None,
) -> CoordinatedTrackingConfig:
    observer = ScenarioObserverControlConfig() if observer is None else observer
    scene_avoidance = (
        ScenarioSceneAvoidanceConfig()
        if scene_avoidance is None
        else scene_avoidance
    )
    return CoordinatedTrackingConfig(
        kinematics_mode=tracking.kinematics_mode,
        executor_position_gain=tracking.executor_position_gain,
        executor_orientation_gain=tracking.executor_orientation_gain,
        observer_position_gain=tracking.observer_position_gain,
        feedforward_gain=tracking.feedforward_gain,
        max_target_speed_mps=_tracking_target_speed_limit(tracking),
        max_target_angular_speed_rad_s=tracking.max_target_angular_speed_rad_s,
        executor_orientation_tracking_weight=(
            tracking.executor_orientation_tracking_weight
        ),
        executor_orientation_tracking_mode=(
            tracking.executor_orientation_tracking_mode
        ),
        inter_arm_min_distance_m=observer.minimum_distance_m,
        inter_arm_influence_distance_m=observer.influence_distance_m,
        inter_arm_hard_stop_distance_m=observer.critical_distance_m,
        inter_arm_release_margin_m=observer.release_margin_m,
        inter_arm_avoidance_gain=observer.avoidance_gain,
        inter_arm_max_avoidance_speed_mps=observer.max_avoidance_speed_mps,
        inter_arm_collision_pair_count=observer.collision_pair_count,
        inter_arm_collision_pair_index_separation=(
            observer.collision_pair_index_separation
        ),
        observer_look_at_executor_tip=observer.look_at_executor_tip,
        observer_look_at_gain=observer.look_at_gain,
        observer_look_at_weight=observer.look_at_weight,
        observer_look_at_distance_m=observer.look_at_distance_m,
        observer_look_at_max_speed_mps=(
            observer.look_at_max_angular_speed_rad_s
            if observer.look_at_max_angular_speed_rad_s is not None
            else observer.look_at_max_speed_mps
        ),
        observer_collision_priority=True,
        freeze_executor_inside_safe_distance=False,
        stop_all_on_critical_distance=False,
        scene_avoidance_enabled=scene_avoidance.enabled,
        executor_scene_avoidance_mode=scene_avoidance.executor_mode,
        observer_scene_avoidance_mode=scene_avoidance.observer_mode,
        engine_min_clearance_m=scene_avoidance.engine_min_clearance_m,
        engine_influence_distance_m=(
            scene_avoidance.engine_influence_distance_m
        ),
        engine_avoidance_gain=scene_avoidance.engine_avoidance_gain,
        enforce_backend_tendon_limits=tracking.enforce_backend_tendon_limits,
    )


def _tracking_target_speed_limit(
    tracking: ScenarioTrackingControlConfig,
) -> float | None:
    if not tracking.enforce_target_speed_limit:
        return None
    return tracking.max_target_speed_mps


def _video_artifacts_enabled(artifacts) -> bool:
    return bool(artifacts.save_gif or artifacts.save_mp4)


def _live_mujoco_pending_video_paths(config) -> list[Path]:
    suffixes: list[str] = []
    if config.artifacts.save_gif:
        suffixes.append("gif")
    if config.artifacts.save_mp4:
        suffixes.append("mp4")
    return [
        config.artifacts.output_root
        / f"_{config.name}_live_mujoco_pending.{suffix}"
        for suffix in suffixes
    ]


def _resolve_task_plan(config, assembly, engine_scene, structured_scene):
    task = config.task
    if task.type in ("idle", "engine_navigation"):
        waypoints = task.waypoints_world
        phases: tuple[str, ...] = ()
        target_force = task.target_force_n
        normal = task.surface_normal_world
        surface_point = None
    elif task.trajectory is not None:
        waypoints = generate_trajectory_waypoints(task.trajectory, assembly)
        phases = task.waypoint_phases
        target_force = task.target_force_n
        normal = task.surface_normal_world
        surface_point = None
    elif task.mission is not None:
        if structured_scene is None:
            raise ValueError("scenario.task.mission requires scenario.scene.structured_config_path.")
        waypoints = resolve_navigation_waypoints(task.mission, structured_scene)
        phases = task.waypoint_phases
        target_force = task.target_force_n
        normal = task.surface_normal_world
        surface_point = None
    elif task.wiping_path is not None:
        if structured_scene is None:
            raise ValueError("scenario.task.wiping_path requires scenario.scene.structured_config_path.")
        plan = build_wiping_plan(task.wiping_path, structured_scene)
        waypoints = plan.waypoints_world
        phases = plan.phases
        target_force = task.target_force_n
        normal = plan.surface_normal_world
        surface_point = plan.surface_point_world
    elif task.engine_cleaning is not None:
        if engine_scene is None:
            raise ValueError("scenario.task.engine_cleaning requires scenario.scene.engine_config_path.")
        plan = build_engine_cleaning_plan(task.engine_cleaning, engine_scene)
        waypoints = plan.waypoints_world
        normals = plan.normals_world
        phases = plan.phases
        target_force = plan.target_force_n
        normal = plan.normals_world[0]
        standoff_distance = plan.standoff_distance_m
        surface_point = None
    else:
        waypoints = task.waypoints_world
        phases = task.waypoint_phases
        target_force = task.target_force_n
        normal = task.surface_normal_world
        surface_point = None
    use_arm_approach = task.type in ("tracking", "navigation") and not (
        task.type == "navigation" and task.tracking_control.stage_mobile_base
    )
    if use_arm_approach:
        tracking_plan = prepend_tracking_approach(
            waypoints,
            assembly,
            samples=task.tracking_control.approach_samples,
        )
        waypoints = tracking_plan.waypoints_world
        approach_mask = tracking_plan.approach_mask
        source_waypoint_index = tracking_plan.source_waypoint_index
    else:
        approach_mask = np.zeros(waypoints.shape[0], dtype=bool)
        source_waypoint_index = np.arange(waypoints.shape[0], dtype=int)
    if "normals" not in locals() or normals.shape[0] != waypoints.shape[0]:
        normals = np.tile(normal, (waypoints.shape[0], 1))
    if (
        "standoff_distance" not in locals()
        or standoff_distance.shape != (waypoints.shape[0],)
    ):
        standoff_distance = np.zeros(waypoints.shape[0], dtype=float)
    if phases and len(phases) != waypoints.shape[0]:
        raise ValueError("scenario.task.waypoint_phases must match waypoint count.")
    if target_force.size == 0:
        target_force = np.zeros(waypoints.shape[0], dtype=float)
        if task.target_normal_force_n > 0.0:
            for index, phase in enumerate(phases):
                if phase == "contact":
                    target_force[index] = task.target_normal_force_n
    elif target_force.shape != (waypoints.shape[0],):
        raise ValueError("scenario.task.target_force_n must match waypoint count.")
    waypoint_orientations = _resolve_waypoint_orientations(
        task,
        waypoints,
        structured_scene,
    )
    return {
        "waypoints_world": waypoints,
        "waypoint_orientations_world_wxyz": waypoint_orientations,
        "phases": phases,
        "target_force_n": target_force,
        "surface_normal_world": normal,
        "surface_point_world": surface_point,
        "normals_world": normals,
        "standoff_distance_m": standoff_distance,
        "approach_mask": approach_mask,
        "source_waypoint_index": source_waypoint_index,
    }


def _resolve_waypoint_orientations(task, waypoints, structured_scene) -> np.ndarray:
    if not task.pose_servo_enabled or task.waypoint_orientation_source == "none":
        return np.zeros((0, 4), dtype=float)
    explicit = task.waypoint_orientations_world_wxyz
    if explicit.shape[0] > 0:
        if explicit.shape[0] != waypoints.shape[0]:
            raise ValueError(
                "scenario.task.waypoint_orientations_world_wxyz must match "
                "resolved waypoint count."
            )
        return explicit.copy()
    directions = task.waypoint_directions_world
    if directions.shape[0] > 0:
        if directions.shape[0] != waypoints.shape[0]:
            raise ValueError(
                "scenario.task.pose_servo.waypoint_directions_world must match "
                "resolved waypoint count."
            )
        return np.asarray(
            [
                look_rotation_quaternion_wxyz(
                    direction,
                    task.orientation_roll_reference_world,
                )
                for direction in directions
            ],
            dtype=float,
        )
    if task.waypoint_orientation_source == "explicit_directions":
        raise ValueError(
            "scenario.task.pose_servo.orientation_source='explicit_directions' "
            "requires waypoint_directions_world."
        )
    if task.waypoint_orientation_source != "nearest_clearance":
        raise ValueError(
            "Unsupported waypoint orientation source "
            f"{task.waypoint_orientation_source!r}."
        )
    if structured_scene is None:
        raise ValueError(
            "scenario.task.pose_servo.orientation_source='nearest_clearance' "
            "requires scenario.scene.structured_config_path."
        )
    query = StructuredSceneQuery(structured_scene)
    orientations = []
    for waypoint in waypoints:
        clearance = query.nearest_distance(waypoint)
        direction = -np.asarray(clearance.normal, dtype=float)
        if np.linalg.norm(direction) <= 1.0e-12:
            direction = np.array([1.0, 0.0, 0.0], dtype=float)
        orientations.append(
            look_rotation_quaternion_wxyz(
                direction,
                task.orientation_roll_reference_world,
            )
        )
    return np.asarray(orientations, dtype=float)


def _load_task_dynamics_config(config, assembly):
    if config.task.dynamics_config_path is None:
        return None
    executor_names = [arm.name for arm in assembly.enabled_arms if arm.role == "executor"]
    if len(executor_names) != 1:
        raise ValueError("Task dynamics requires exactly one executor arm.")
    params = assembly.arms[executor_names[0]].spatial_arm.params
    return load_pcc_dynamics_config(config.task.dynamics_config_path, params)


def _default_engine_cleaning_gains(config) -> EngineCleaningControllerGains:
    return EngineCleaningControllerGains(
        tangential_position_gain=8.0,
        normal_position_gain=3.0,
        normal_force_gain=max(config.task.normal_force_gain, 0.001),
        approach_position_gain=5.0,
        retreat_position_gain=5.0,
        max_tcp_speed_mps=0.03,
        max_normal_speed_mps=0.01,
        waypoint_tolerance_m=config.task.waypoint_tolerance_m,
        max_contact_force_n=(
            5.0
            if config.task.max_contact_force_n is None
            else config.task.max_contact_force_n
        ),
        force_deadband_n=0.05,
        min_clearance_m=config.task.min_clearance_m,
    )
