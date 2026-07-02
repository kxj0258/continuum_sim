"""Scenario composition root and primary simulation application API."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from continuum_sim.application.scenario import ScenarioConfig, load_scenario_config
from continuum_sim.backends.analytic_system_backend import AnalyticSystemBackend
from continuum_sim.backends.mujoco_system_backend import MujocoSystemBackend
from continuum_sim.config import load_mujoco_config
from continuum_sim.control.scenario_controllers import (
    NavigationController,
    WaypointTrackingController,
    WipingController,
    ZeroSystemController,
)
from continuum_sim.model.robot_assembly import load_robot_assembly_config
from continuum_sim.runtime.hooks import (
    ControllerCompletionHook,
    LiveTendonPanelHook,
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
from continuum_sim.scenes.scene_config import load_navigation_scene_config
from continuum_sim.scenes.structured_query import StructuredSceneQuery
from continuum_sim.scenes.scene_builder import (
    inject_structured_scene,
    lock_mobile_base_freejoint,
)
from continuum_sim.tasks.engine_cleaning_path import build_engine_cleaning_plan
from continuum_sim.tasks.navigation_mission import resolve_navigation_waypoints
from continuum_sim.tasks.trajectory_generation import generate_trajectory_waypoints
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
            backend = AnalyticSystemBackend(assembly)
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
        elif config.task.type == "navigation":
            controller = NavigationController(
                assembly,
                task_plan["waypoints_world"],
                waypoint_tolerance_m=config.task.waypoint_tolerance_m,
                observer_roi_world=config.task.observer_roi_world,
                scene_query=scene_query,
                min_clearance_m=config.task.min_clearance_m,
                terminate_on_clearance_violation=(
                    config.task.terminate_on_clearance_violation
                ),
                target_advance_mode=config.task.target_advance_mode,
                controller_dt_s=config.runtime.controller_dt_s,
                advance_time_s=config.task.advance_time_s,
                advance_steps=config.task.advance_steps,
            )
        elif config.task.type in ("wiping", "engine_cleaning"):
            controller = WipingController(
                assembly,
                task_plan["waypoints_world"],
                waypoint_tolerance_m=config.task.waypoint_tolerance_m,
                scene_query=scene_query,
                surface_normal_world=task_plan["surface_normal_world"],
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
            )
        else:
            controller = WaypointTrackingController(
                assembly,
                task_plan["waypoints_world"],
                waypoint_tolerance_m=config.task.waypoint_tolerance_m,
                observer_roi_world=config.task.observer_roi_world,
                loop=config.task.loop,
                target_advance_mode=config.task.target_advance_mode,
                controller_dt_s=config.runtime.controller_dt_s,
                advance_time_s=config.task.advance_time_s,
                advance_steps=config.task.advance_steps,
                scene_query=scene_query,
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
        if (
            config.backend.type == "mujoco"
            and config.artifacts.enabled
            and config.artifacts.save_gif
            and config.artifacts.video_mode == "live_mujoco"
        ):
            hooks_by_name["live_mujoco_video"] = MujocoLiveVideoRecorderHook(
                backend,
                config.artifacts.output_root / f"_{config.name}_live_mujoco_pending.gif",
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
    if structured_scene is not None:
        inject_structured_scene(root, structured_scene)
    if assembly.base.control_mode == "fixed":
        lock_mobile_base_freejoint(root)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(tree)
    tree.write(output_path, encoding="utf-8", xml_declaration=False)
    return MujocoSystemBackend(
        load_mujoco_config(backend.mujoco_config_path),
        assembly,
        xml_path=output_path,
    )


def _resolve_task_plan(config, assembly, engine_scene, structured_scene):
    task = config.task
    if task.type == "idle":
        waypoints = task.waypoints_world
        phases: tuple[str, ...] = ()
        target_force = task.target_force_n
        normal = task.surface_normal_world
    elif task.trajectory is not None:
        waypoints = generate_trajectory_waypoints(task.trajectory, assembly)
        phases = task.waypoint_phases
        target_force = task.target_force_n
        normal = task.surface_normal_world
    elif task.mission is not None:
        if structured_scene is None:
            raise ValueError("scenario.task.mission requires scenario.scene.structured_config_path.")
        waypoints = resolve_navigation_waypoints(task.mission, structured_scene)
        phases = task.waypoint_phases
        target_force = task.target_force_n
        normal = task.surface_normal_world
    elif task.wiping_path is not None:
        if structured_scene is None:
            raise ValueError("scenario.task.wiping_path requires scenario.scene.structured_config_path.")
        plan = build_wiping_plan(task.wiping_path, structured_scene)
        waypoints = plan.waypoints_world
        phases = plan.phases
        target_force = task.target_force_n
        normal = plan.surface_normal_world
    elif task.engine_cleaning is not None:
        if engine_scene is None:
            raise ValueError("scenario.task.engine_cleaning requires scenario.scene.engine_config_path.")
        plan = build_engine_cleaning_plan(task.engine_cleaning, engine_scene)
        waypoints = plan.waypoints_world
        phases = plan.phases
        target_force = plan.target_force_n
        normal = plan.normals_world[0]
    else:
        waypoints = task.waypoints_world
        phases = task.waypoint_phases
        target_force = task.target_force_n
        normal = task.surface_normal_world
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
    return {
        "waypoints_world": waypoints,
        "phases": phases,
        "target_force_n": target_force,
        "surface_normal_world": normal,
    }
