"""Scenario composition root and primary simulation application API."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET

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
    MatplotlibSystemViewerHook,
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
from continuum_sim.scenes.scene_builder import inject_structured_scene


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
        if config.task.type == "idle":
            controller = ZeroSystemController(assembly)
        elif config.task.type == "navigation":
            controller = NavigationController(
                assembly,
                config.task.waypoints_world,
                waypoint_tolerance_m=config.task.waypoint_tolerance_m,
                observer_roi_world=config.task.observer_roi_world,
                scene_query=scene_query,
                min_clearance_m=config.task.min_clearance_m,
                terminate_on_clearance_violation=(
                    config.task.terminate_on_clearance_violation
                ),
            )
        elif config.task.type == "wiping":
            controller = WipingController(
                assembly,
                config.task.waypoints_world,
                waypoint_tolerance_m=config.task.waypoint_tolerance_m,
                scene_query=scene_query,
                surface_normal_world=config.task.surface_normal_world,
                target_contact_distance_m=config.task.target_contact_distance_m,
                contact_tolerance_m=config.task.contact_tolerance_m,
            )
        else:
            controller = WaypointTrackingController(
                assembly,
                config.task.waypoints_world,
                waypoint_tolerance_m=config.task.waypoint_tolerance_m,
                observer_roi_world=config.task.observer_roi_world,
                loop=config.task.loop,
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
        if config.backend.type == "mujoco" and config.artifacts.enabled:
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
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(tree)
    tree.write(output_path, encoding="utf-8", xml_declaration=False)
    return MujocoSystemBackend(
        load_mujoco_config(backend.mujoco_config_path),
        assembly,
        xml_path=output_path,
    )
