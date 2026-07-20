"""Backend construction for scenario applications."""

from __future__ import annotations

from dataclasses import replace
import xml.etree.ElementTree as ET

import numpy as np

from continuum_sim.application.hook_factory import observer_camera_attachment_config
from continuum_sim.backends.mujoco_system_backend import MujocoSystemBackend
from continuum_sim.config import load_mujoco_config
from continuum_sim.control.tendon_rate_control import BendingRateServoConfig
from continuum_sim.scenes.engine_mjcf_adapter import (
    inject_engine_scene,
    rebase_mjcf_file_assets,
    retain_spatial_arm,
)
from continuum_sim.scenes.scene_config import InspectionTargetConfig
from continuum_sim.scenes.scene_builder import (
    inject_structured_scene,
    inject_tip_camera,
    lock_mobile_base_freejoint,
)


def build_mujoco_backend(config, assembly, engine_scene, structured_scene):
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
    if backend.mujoco_viewer_camera is not None:
        mujoco_config = replace(
            mujoco_config,
            viewer=replace(
                mujoco_config.viewer,
                camera=backend.mujoco_viewer_camera,
            ),
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
    visual_structured_scene = _structured_scene_for_mujoco_visuals(
        config,
        structured_scene,
    )
    if visual_structured_scene is not None:
        inject_structured_scene(root, visual_structured_scene)
    observer_camera = observer_camera_attachment_config(assembly)
    if observer_camera is not None and observer_camera.camera is not None:
        inject_tip_camera(
            root,
            tip_site_name="observer_tip",
            camera_name=observer_camera.camera.name,
            tip_to_camera=observer_camera.camera.tip_to_camera,
            fovy_deg=observer_camera.camera.intrinsics.fovy_deg,
        )
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
