"""Build MuJoCo XML scene variants from structured scene YAML."""

from __future__ import annotations

import math
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from continuum_sim.model.base_pose import Pose6D, quaternion_wxyz_multiply
from continuum_sim.model.mount_frame import load_mobile_base_mount_config
from continuum_sim.scenes.scene_config import (
    InspectionTargetConfig,
    NavigationSceneConfig,
    ScenePrimitiveConfig,
)


@dataclass(frozen=True)
class ToolPadXmlConfig:
    """MuJoCo XML attributes for a tip-mounted contact pad."""

    type: str
    radius_m: float
    length_m: float
    offset_m: np.ndarray
    rgba: tuple[float, float, float, float]
    geom_name: str
    body_name: str
    contact_site_name: str
    site_radius_m: float
    contype: int
    conaffinity: int


def build_mujoco_scene_xml(
    base_xml_path: str | Path,
    scene_config: NavigationSceneConfig,
    output_xml_path: str | Path,
    *,
    offscreen_size: tuple[int, int] | None = None,
    mobile_base_config_path: str | Path | None = None,
) -> Path:
    """Inject structured scene geoms into a base MuJoCo XML file."""

    base_path = Path(base_xml_path).resolve()
    output_path = Path(output_xml_path).resolve()
    if not base_path.is_file():
        raise FileNotFoundError(f"Base MuJoCo XML does not exist: {base_path}")

    tree = ET.parse(base_path)
    root = tree.getroot()
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError(f"{base_path} is missing a <worldbody> element.")
    _rebase_asset_file_paths(root, base_path.parent, output_path.parent)
    if offscreen_size is not None:
        _set_offscreen_framebuffer(root, offscreen_size)
    if mobile_base_config_path is not None:
        _attach_mobile_base_wrapper(
            root,
            mobile_base_config_path,
        )

    inject_structured_scene(root, scene_config)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    _indent(root)
    tree.write(output_path, encoding="utf-8", xml_declaration=False)
    return output_path


def inject_structured_scene(
    root: ET.Element,
    scene_config: NavigationSceneConfig,
) -> ET.Element:
    """Inject navigation/wiping primitives into an existing MJCF tree."""

    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError("MJCF is missing a <worldbody> element.")
    scene_body = ET.SubElement(
        worldbody,
        "body",
        {
            "name": f"scene_{_safe_name(scene_config.name)}",
            "pos": "0 0 0",
        },
    )
    for primitive in scene_config.primitives:
        _append_primitive(scene_body, scene_config, primitive)
    for target in scene_config.inspection_targets:
        _append_target_marker(scene_body, scene_config, target)
    return root


def build_mujoco_wiping_xml(
    base_xml_path: str | Path,
    scene_config: NavigationSceneConfig,
    tool_config: ToolPadXmlConfig,
    output_xml_path: str | Path,
    *,
    tip_site_name: str = "tip",
    offscreen_size: tuple[int, int] | None = None,
    mobile_base_config_path: str | Path | None = None,
) -> Path:
    """Inject structured scene geoms and a tip-mounted wiping tool."""

    output_path = build_mujoco_scene_xml(
        base_xml_path,
        scene_config,
        output_xml_path,
        offscreen_size=offscreen_size,
        mobile_base_config_path=mobile_base_config_path,
    )
    inject_tool_contact_pad(
        output_path,
        output_path,
        tool_config,
        tip_site_name=tip_site_name,
    )
    return output_path


def inject_mobile_base_wrapper(
    base_xml_path: str | Path,
    output_xml_path: str | Path,
    mobile_base_config_path: str | Path,
) -> Path:
    """Wrap the arm root under an optional mobile-base body."""

    base_path = Path(base_xml_path).resolve()
    output_path = Path(output_xml_path).resolve()
    if not base_path.is_file():
        raise FileNotFoundError(f"Base MuJoCo XML does not exist: {base_path}")

    tree = ET.parse(base_path)
    root = tree.getroot()
    _rebase_asset_file_paths(root, base_path.parent, output_path.parent)
    _attach_mobile_base_wrapper(root, mobile_base_config_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _indent(root)
    tree.write(output_path, encoding="utf-8", xml_declaration=False)
    return output_path


def lock_mobile_base_freejoint(root: ET.Element) -> ET.Element:
    """Remove the optional mobile-base freejoint while keeping the wrapper body."""

    for body in root.findall(".//body"):
        for child in list(body):
            if (
                child.tag == "freejoint"
                and child.attrib.get("name") == "mobile_base_freejoint"
            ):
                body.remove(child)
    return root


def _set_offscreen_framebuffer(
    root: ET.Element,
    offscreen_size: tuple[int, int],
) -> None:
    width, height = offscreen_size
    if width <= 0 or height <= 0:
        raise ValueError(f"offscreen_size must be positive, got {offscreen_size}.")
    visual = root.find("visual")
    if visual is None:
        visual = ET.Element("visual")
        insert_index = 1 if root.find("option") is not None else 0
        root.insert(insert_index, visual)
    global_visual = visual.find("global")
    if global_visual is None:
        global_visual = ET.Element("global")
        visual.insert(0, global_visual)
    global_visual.set("offwidth", str(int(width)))
    global_visual.set("offheight", str(int(height)))


def _attach_mobile_base_wrapper(
    root: ET.Element,
    mobile_base_config_path: str | Path,
) -> None:
    config = load_mobile_base_mount_config(mobile_base_config_path)
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError("Base XML is missing a <worldbody> section.")
    robot_root = _find_robot_root_body(worldbody)
    robot_root_pose = _body_pose_from_xml(robot_root)
    primary_mount = config.mount
    wrapped_pose = primary_mount.pose.compose(robot_root_pose)

    worldbody.remove(robot_root)
    mobile_base_body = ET.Element(
        "body",
        {
            "name": "mobile_base",
            "pos": _format_vec(config.mobile_base.pose.position),
            "quat": _format_tuple(tuple(float(value) for value in config.mobile_base.pose.quat)),
        },
    )
    ET.SubElement(
        mobile_base_body,
        "inertial",
        {
            "pos": "0 0 0",
            "mass": _format_float(config.mobile_base.inertial.mass_kg),
            "diaginertia": _format_vec(config.mobile_base.inertial.diaginertia_kg_m2),
        },
    )
    ET.SubElement(
        mobile_base_body,
        "freejoint",
        {
            "name": "mobile_base_freejoint",
        },
    )
    ET.SubElement(
        mobile_base_body,
        "site",
        {
            "name": "mobile_base_frame",
            "type": "sphere",
            "pos": "0 0 0",
            "size": "0.002",
            "rgba": "0.2 0.6 1 0.8",
            "group": "2",
        },
    )
    if config.mobile_base.visualization.enabled:
        _append_mobile_base_geom(mobile_base_body, config)
    for mount in config.mounts.values():
        ET.SubElement(
            mobile_base_body,
            "site",
            {
                "name": mount.name,
                "type": "sphere",
                "pos": _format_vec(mount.pose.position),
                "quat": _format_tuple(tuple(float(value) for value in mount.pose.quat)),
                "size": "0.002",
                "rgba": "0.95 0.55 0.12 0.85",
                "group": "2",
            },
        )

    robot_root.set("pos", _format_vec(wrapped_pose.position))
    robot_root.set("quat", _format_tuple(tuple(float(value) for value in wrapped_pose.quat)))
    mobile_base_body.append(robot_root)
    worldbody.insert(0, mobile_base_body)


def _append_mobile_base_geom(parent: ET.Element, config) -> None:
    visualization = config.mobile_base.visualization
    if visualization.type != "box":
        raise ValueError(
            "Only mobile_base.visualization.type='box' is supported in this phase, "
            f"got {visualization.type!r}."
        )
    half_size = 0.5 * np.asarray(visualization.size_m, dtype=float)
    ET.SubElement(
        parent,
        "geom",
        {
            "name": "mobile_base_box",
            "type": "box",
            "size": _format_vec(half_size),
            "rgba": _format_tuple(visualization.rgba),
            "contype": "0",
            "conaffinity": "0",
            "group": "2",
        },
    )


def _find_robot_root_body(worldbody: ET.Element) -> ET.Element:
    top_level_bodies = [child for child in worldbody if child.tag == "body"]
    if not top_level_bodies:
        raise ValueError("Base XML does not contain any top-level <body> elements.")
    for body in top_level_bodies:
        if body.attrib.get("name") == "base":
            return body
    return top_level_bodies[0]


def _body_pose_from_xml(body: ET.Element):
    from continuum_sim.model.base_pose import Pose6D

    pos = _parse_vec(body.attrib.get("pos", "0 0 0"))
    quat = np.fromstring(body.attrib.get("quat", "1 0 0 0"), sep=" ", dtype=float)
    if quat.shape != (4,):
        raise ValueError(
            f"Expected body quaternion with 4 entries, got {body.attrib.get('quat', '')!r}."
        )
    return Pose6D(position=pos, quat=quat)


def inject_tool_contact_pad(
    base_xml_path: str | Path,
    output_xml_path: str | Path,
    tool_config: ToolPadXmlConfig,
    *,
    tip_site_name: str = "tip",
) -> Path:
    """Insert a spherical or capsule contact pad into the body that owns ``tip``."""

    base_path = Path(base_xml_path).resolve()
    output_path = Path(output_xml_path).resolve()
    if not base_path.is_file():
        raise FileNotFoundError(f"Base MuJoCo XML does not exist: {base_path}")

    tree = ET.parse(base_path)
    root = tree.getroot()
    _rebase_asset_file_paths(root, base_path.parent, output_path.parent)
    tip_body, tip_site = _find_site_parent_body_and_site(root, tip_site_name)
    if tip_body is None or tip_site is None:
        raise ValueError(f"MuJoCo model is missing tip site {tip_site_name!r}.")
    _remove_existing_named_child(tip_body, "body", tool_config.body_name)
    tip_site_pos = _parse_vec(tip_site.attrib.get("pos", "0 0 0"))
    pad_pos = tip_site_pos + np.asarray(tool_config.offset_m, dtype=float)
    pad_body = ET.SubElement(
        tip_body,
        "body",
        {
            "name": tool_config.body_name,
            "pos": _format_vec(pad_pos),
        },
    )
    geom_attrs = {
        "name": tool_config.geom_name,
        "type": tool_config.type,
        "rgba": _format_tuple(tool_config.rgba),
        "contype": str(tool_config.contype),
        "conaffinity": str(tool_config.conaffinity),
    }
    if tool_config.type == "sphere":
        geom_attrs["size"] = _format_float(tool_config.radius_m)
    elif tool_config.type == "capsule":
        half_length = max(0.5 * tool_config.length_m, 1.0e-9)
        geom_attrs["fromto"] = _format_tuple(
            (0.0, 0.0, -half_length, 0.0, 0.0, half_length)
        )
        geom_attrs["size"] = _format_float(tool_config.radius_m)
    else:
        raise ValueError(f"Unsupported tool pad type {tool_config.type!r}.")
    ET.SubElement(pad_body, "geom", geom_attrs)
    ET.SubElement(
        pad_body,
        "site",
        {
            "name": tool_config.contact_site_name,
            "type": "sphere",
            "pos": "0 0 0",
            "size": _format_float(tool_config.site_radius_m),
            "rgba": _format_tuple(tool_config.rgba),
        },
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    _indent(root)
    tree.write(output_path, encoding="utf-8", xml_declaration=False)
    return output_path


def inject_tip_camera(
    root: ET.Element,
    *,
    tip_site_name: str,
    camera_name: str,
    tip_to_camera: Pose6D,
    fovy_deg: float,
    camera_visual=None,
) -> None:
    """Insert an observer camera and its optional tip-mounted dome."""

    tip_body, tip_site = _find_site_parent_body_and_site(root, tip_site_name)
    if tip_body is None or tip_site is None:
        raise ValueError(f"MuJoCo model is missing tip site {tip_site_name!r}.")
    camera_quat = quaternion_wxyz_multiply(
        tip_to_camera.quat,
        np.array([0.0, 1.0, 0.0, 0.0], dtype=float),
    )
    mount_name = f"{camera_name}_mount"
    _remove_existing_named_child(tip_body, "body", mount_name)
    tip_site_pos = _parse_vec(tip_site.attrib.get("pos", "0 0 0"))
    tip_site_quat = tip_site.attrib.get("quat", "1 0 0 0")
    mount_body = ET.SubElement(
        tip_body,
        "body",
        {
            "name": mount_name,
            "pos": _format_vec(tip_site_pos),
            "quat": tip_site_quat,
        },
    )
    if camera_visual is not None:
        radius = float(camera_visual.radius_m)
        lens_depth = float(camera_visual.lens_depth_m)
        ET.SubElement(
            mount_body,
            "geom",
            {
                "name": f"{camera_name}_dome_visual",
                "type": "sphere",
                # The sphere centre lies on the arm end plane. Its rear half is
                # embedded in the opaque terminal link, leaving a hemisphere.
                "pos": "0 0 0",
                "size": _format_float(radius),
                "rgba": _format_tuple(camera_visual.rgba),
                "contype": "0",
                "conaffinity": "0",
                "density": "0",
                "group": "1",
            },
        )
        ET.SubElement(
            mount_body,
            "geom",
            {
                "name": f"{camera_name}_lens_visual",
                "type": "cylinder",
                "fromto": (
                    f"0 0 {_format_float(radius - 0.5 * lens_depth)} "
                    f"0 0 {_format_float(radius + 0.5 * lens_depth)}"
                ),
                "size": _format_float(float(camera_visual.lens_radius_m)),
                "rgba": _format_tuple(camera_visual.lens_rgba),
                "contype": "0",
                "conaffinity": "0",
                "density": "0",
                "group": "1",
            },
        )
    ET.SubElement(
        mount_body,
        "camera",
        {
            "name": camera_name,
            "pos": _format_vec(tip_to_camera.position),
            "quat": _format_vec(camera_quat),
            "fovy": _format_float(fovy_deg),
        },
    )


def _rebase_asset_file_paths(
    root: ET.Element,
    base_xml_dir: Path,
    output_xml_dir: Path,
) -> None:
    asset = root.find("asset")
    if asset is None:
        return
    for element in asset.iter():
        raw_file = element.attrib.get("file")
        if not raw_file:
            continue
        raw_path = Path(raw_file)
        if raw_path.is_absolute():
            continue
        source_path = (base_xml_dir / raw_path).resolve()
        element.set(
            "file",
            Path(os.path.relpath(source_path, output_xml_dir.resolve())).as_posix(),
        )


def _append_primitive(
    parent: ET.Element,
    scene_config: NavigationSceneConfig,
    primitive: ScenePrimitiveConfig,
) -> None:
    if primitive.type in ("cylindrical_shell_segment", "frustum_shell_segment"):
        _append_shell(parent, scene_config, primitive)
        return
    if primitive.type == "cylinder_obstacle":
        _append_cylinder_obstacle(parent, scene_config, primitive)
        return
    if primitive.type in ("box_obstacle", "box_surface"):
        _append_box_obstacle(parent, scene_config, primitive)
        return
    raise ValueError(f"Unsupported scene primitive type {primitive.type!r}.")


def _append_target_marker(
    parent: ET.Element,
    scene_config: NavigationSceneConfig,
    target: InspectionTargetConfig,
) -> None:
    target_name = _safe_name(target.id)
    radius = scene_config.builder.target_radius_m
    ET.SubElement(
        parent,
        "site",
        {
            "name": f"scene_target_{target_name}",
            "type": "sphere",
            "pos": _format_vec(target.pos_m),
            "size": _format_float(radius),
            "rgba": _format_tuple(scene_config.builder.target_rgba),
            "group": str(scene_config.builder.geom_group),
        },
    )
    direction = _target_marker_direction(scene_config, target)
    arrow_length = max(6.0 * radius, 0.018)
    shaft_start = target.pos_m + direction * (1.35 * radius)
    shaft_end = target.pos_m + direction * arrow_length
    ET.SubElement(
        parent,
        "geom",
        {
            "name": f"scene_target_{target_name}_orientation_shaft",
            "type": "capsule",
            "fromto": _format_tuple(
                tuple(float(value) for value in np.concatenate((shaft_start, shaft_end)))
            ),
            "size": _format_float(max(0.25 * radius, 0.0006)),
            "rgba": _format_tuple(scene_config.builder.target_rgba),
            "group": str(scene_config.builder.geom_group),
            "contype": "0",
            "conaffinity": "0",
        },
    )
    ET.SubElement(
        parent,
        "site",
        {
            "name": f"scene_target_{target_name}_orientation_tip",
            "type": "sphere",
            "pos": _format_vec(shaft_end),
            "size": _format_float(max(0.65 * radius, 0.0015)),
            "rgba": _format_tuple(scene_config.builder.target_rgba),
            "group": str(scene_config.builder.geom_group),
        },
    )


def _target_marker_direction(
    scene_config: NavigationSceneConfig,
    target: InspectionTargetConfig,
) -> np.ndarray:
    if target.direction_world is not None:
        direction = np.asarray(target.direction_world, dtype=float)
        norm = np.linalg.norm(direction)
        if np.isfinite(norm) and norm > 1.0e-12:
            return direction / norm
    best_distance = float("inf")
    best_normal: np.ndarray | None = None
    for primitive in scene_config.clearance_primitives:
        query = primitive.clearance(target.pos_m)
        normal_norm = float(np.linalg.norm(query.normal))
        if not np.isfinite(query.distance_m) or normal_norm <= 1.0e-12:
            continue
        if query.distance_m < best_distance:
            best_distance = float(query.distance_m)
            best_normal = query.normal / normal_norm
    if best_normal is None:
        return np.array([0.0, 0.0, 1.0], dtype=float)
    return -best_normal


def _append_shell(
    parent: ET.Element,
    scene_config: NavigationSceneConfig,
    primitive: ScenePrimitiveConfig,
) -> None:
    sides = scene_config.builder.shell_approx_sides
    slices = scene_config.builder.shell_axial_slices
    z_min = _required(primitive.z_min_m, f"{primitive.id}.z_min_m")
    z_max = _required(primitive.z_max_m, f"{primitive.id}.z_max_m")
    wall_thickness = scene_config.builder.wall_thickness_m
    for slice_index in range(slices):
        z0 = z_min + (z_max - z_min) * slice_index / slices
        z1 = z_min + (z_max - z_min) * (slice_index + 1) / slices
        z_mid = 0.5 * (z0 + z1)
        radius = _shell_radius_at(primitive, z_mid)
        wall_radius = radius + 0.5 * wall_thickness
        axial_half = max(0.5 * (z1 - z0), 1.0e-6)
        tangent_half = max(wall_radius * math.sin(math.pi / sides), 1.0e-6)
        for side_index in range(sides):
            theta = 2.0 * math.pi * side_index / sides
            pos = np.array(
                [
                    wall_radius * math.cos(theta),
                    wall_radius * math.sin(theta),
                    z_mid,
                ],
                dtype=float,
            )
            yaw = theta + 0.5 * math.pi
            ET.SubElement(
                parent,
                "geom",
                {
                    "name": (
                        f"scene_{_safe_name(primitive.id)}_"
                        f"s{slice_index:02d}_{side_index:02d}"
                    ),
                    "type": "box",
                    "pos": _format_vec(pos),
                    "quat": _yaw_quat(yaw),
                    "size": _format_vec(
                        np.array(
                            [tangent_half, 0.5 * wall_thickness, axial_half],
                            dtype=float,
                        )
                    ),
                    "rgba": _format_tuple(primitive.rgba or scene_config.builder.shell_rgba),
                    "group": str(scene_config.builder.geom_group),
                    "contype": str(scene_config.builder.contype),
                    "conaffinity": str(scene_config.builder.conaffinity),
                },
            )


def _append_cylinder_obstacle(
    parent: ET.Element,
    scene_config: NavigationSceneConfig,
    primitive: ScenePrimitiveConfig,
) -> None:
    center = _required_array(primitive.center_m, f"{primitive.id}.center_m")
    attrs = {
        "name": f"scene_{_safe_name(primitive.id)}",
        "type": "cylinder",
        "pos": _format_vec(center),
        "size": _format_tuple(
            (
                _required(primitive.radius_m, f"{primitive.id}.radius_m"),
                _required(primitive.half_length_m, f"{primitive.id}.half_length_m"),
            )
        ),
        "rgba": _format_tuple(primitive.rgba or scene_config.builder.obstacle_rgba),
        "group": str(scene_config.builder.geom_group),
        "contype": str(scene_config.builder.contype),
        "conaffinity": str(scene_config.builder.conaffinity),
    }
    quat = _axis_quat(primitive.axis)
    if quat is not None:
        attrs["quat"] = quat
    ET.SubElement(parent, "geom", attrs)


def _append_box_obstacle(
    parent: ET.Element,
    scene_config: NavigationSceneConfig,
    primitive: ScenePrimitiveConfig,
) -> None:
    ET.SubElement(
        parent,
        "geom",
        {
            "name": f"scene_{_safe_name(primitive.id)}",
            "type": "box",
            "pos": _format_vec(_required_array(primitive.center_m, f"{primitive.id}.center_m")),
            "size": _format_vec(
                _required_array(primitive.half_size_m, f"{primitive.id}.half_size_m")
            ),
            "rgba": _format_tuple(primitive.rgba or scene_config.builder.obstacle_rgba),
            "group": str(scene_config.builder.geom_group),
            "contype": str(scene_config.builder.contype),
            "conaffinity": str(scene_config.builder.conaffinity),
        },
    )


def _shell_radius_at(primitive: ScenePrimitiveConfig, z_m: float) -> float:
    if primitive.type == "cylindrical_shell_segment":
        return _required(primitive.radius_m, f"{primitive.id}.radius_m")
    z_min = _required(primitive.z_min_m, f"{primitive.id}.z_min_m")
    z_max = _required(primitive.z_max_m, f"{primitive.id}.z_max_m")
    radius_start = _required(primitive.radius_start_m, f"{primitive.id}.radius_start_m")
    radius_end = _required(primitive.radius_end_m, f"{primitive.id}.radius_end_m")
    alpha = (z_m - z_min) / (z_max - z_min)
    return float((1.0 - alpha) * radius_start + alpha * radius_end)


def _axis_quat(axis: str) -> str | None:
    if axis == "z":
        return None
    if axis == "x":
        return _format_tuple((math.sqrt(0.5), 0.0, math.sqrt(0.5), 0.0))
    if axis == "y":
        return _format_tuple((math.sqrt(0.5), -math.sqrt(0.5), 0.0, 0.0))
    raise ValueError(f"axis must be one of ('x', 'y', 'z'), got {axis!r}.")


def _find_site_parent_body_and_site(
    root: ET.Element,
    site_name: str,
) -> tuple[ET.Element | None, ET.Element | None]:
    worldbody = root.find("worldbody")
    if worldbody is None:
        return None, None
    return _find_site_parent_body_and_site_recursive(worldbody, site_name)


def _find_site_parent_body_and_site_recursive(
    element: ET.Element,
    site_name: str,
) -> tuple[ET.Element | None, ET.Element | None]:
    if element.tag == "body":
        for child in element:
            if child.tag == "site" and child.attrib.get("name") == site_name:
                return element, child
    for child in element:
        result = _find_site_parent_body_and_site_recursive(child, site_name)
        if result[0] is not None:
            return result
    return None, None


def _remove_existing_named_child(parent: ET.Element, tag: str, name: str) -> None:
    for child in list(parent):
        if child.tag == tag and child.attrib.get("name") == name:
            parent.remove(child)


def _yaw_quat(yaw_rad: float) -> str:
    return _format_tuple((math.cos(0.5 * yaw_rad), 0.0, 0.0, math.sin(0.5 * yaw_rad)))


def _format_vec(values: np.ndarray) -> str:
    return _format_tuple(tuple(float(value) for value in values))


def _parse_vec(raw_value: str) -> np.ndarray:
    values = np.fromstring(raw_value, sep=" ", dtype=float)
    if values.shape != (3,):
        raise ValueError(f"Expected XML vector with 3 entries, got {raw_value!r}.")
    return values


def _format_tuple(values: tuple[float, ...]) -> str:
    return " ".join(_format_float(value) for value in values)


def _format_float(value: float) -> str:
    return f"{float(value):.9g}"


def _safe_name(value: str) -> str:
    return "".join(char if char.isalnum() or char == "_" else "_" for char in value)


def _required(value: float | None, name: str) -> float:
    if value is None:
        raise ValueError(f"Missing required config field {name!r}.")
    return float(value)


def _required_array(value: np.ndarray | None, name: str) -> np.ndarray:
    if value is None:
        raise ValueError(f"Missing required config field {name!r}.")
    return np.asarray(value, dtype=float)


def _indent(element: ET.Element, level: int = 0) -> None:
    spaces = "\n" + level * "  "
    if len(element):
        if not element.text or not element.text.strip():
            element.text = spaces + "  "
        for child in element:
            _indent(child, level + 1)
        if not element[-1].tail or not element[-1].tail.strip():
            element[-1].tail = spaces
    if level and (not element.tail or not element.tail.strip()):
        element.tail = spaces
