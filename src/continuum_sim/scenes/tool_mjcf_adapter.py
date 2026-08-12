"""MuJoCo composition for arm-tip force sensors and contact tools."""

from __future__ import annotations

from collections.abc import Mapping
import xml.etree.ElementTree as ET

import numpy as np

from continuum_sim.model.base_pose import Pose6D
from continuum_sim.tools.attachments import AttachmentConfig


def inject_enabled_tip_tools(
    root: ET.Element,
    assembly,
    attachments: Mapping[str, AttachmentConfig],
) -> ET.Element:
    """Inject supported enabled-arm tools into an MJCF tree in place."""

    for arm in assembly.enabled_arms:
        attachment = attachments.get(arm.name)
        if attachment is None or not attachment.enabled:
            continue
        if attachment.type == "contact_sphere_tool":
            if arm.role != "executor":
                raise ValueError(
                    "contact_sphere_tool must be mounted on the executor arm, "
                    f"not role {arm.role!r}."
                )
            inject_force_sensor_sphere_tool(
                root,
                arm_name=arm.name,
                tip_site_name=f"{arm.name}_tip",
                config=attachment,
            )
    return root


def inject_force_sensor_sphere_tool(
    root: ET.Element,
    *,
    arm_name: str,
    tip_site_name: str,
    config: AttachmentConfig,
) -> ET.Element:
    """Mount a box F/T sensor followed by one spherical collision pad."""

    if config.collision is None or config.collision.radius_m is None:
        raise ValueError("A spherical tool requires collision.radius_m.")
    if config.tcp_pose is None:
        raise ValueError("A spherical tool requires tcp_pose.")
    sensor_config = config.force_torque_sensor
    if sensor_config is None:
        raise ValueError(
            "The primary MuJoCo tool path requires tool.force_torque_sensor."
        )

    tip_body, tip_site = _find_site_parent_body_and_site(root, tip_site_name)
    if tip_body is None or tip_site is None:
        raise ValueError(f"MuJoCo model is missing tip site {tip_site_name!r}.")

    sensor_body_name = f"{arm_name}_ft_sensor_body"
    _remove_named_child(tip_body, "body", sensor_body_name)
    tip_pose = Pose6D(
        position=_parse_vector(tip_site.get("pos", "0 0 0")),
        quat=_parse_quaternion(tip_site.get("quat", "1 0 0 0")),
    )
    sensor_pose = tip_pose.compose(config.tip_to_attachment)
    sensor_body = ET.SubElement(
        tip_body,
        "body",
        {
            "name": sensor_body_name,
            "pos": _format_vector(sensor_pose.position),
            "quat": _format_vector(sensor_pose.quat),
        },
    )
    ET.SubElement(
        sensor_body,
        "geom",
        {
            "name": f"{arm_name}_ft_sensor_visual",
            "type": "box",
            "size": _format_vector(0.5 * sensor_config.size_m),
            "mass": _format_float(sensor_config.mass_kg),
            "rgba": _format_tuple(sensor_config.rgba),
            "contype": "0",
            "conaffinity": "0",
            "group": "1",
        },
    )
    sensor_site_name = f"{arm_name}_ft_sensor_site"
    ET.SubElement(
        sensor_body,
        "site",
        {
            "name": sensor_site_name,
            "type": "sphere",
            "pos": "0 0 0",
            "size": "0.001",
            "rgba": "0.95 0.75 0.1 0.9",
            "group": "2",
        },
    )

    sphere_position = np.asarray(config.collision.position, dtype=float)
    sensor_front_z = 0.5 * float(sensor_config.size_m[2])
    sphere_back_z = float(sphere_position[2] - config.collision.radius_m)
    if sphere_back_z > sensor_front_z:
        ET.SubElement(
            sensor_body,
            "geom",
            {
                "name": f"{arm_name}_tool_connector_visual",
                "type": "cylinder",
                "fromto": (
                    f"0 0 {_format_float(sensor_front_z)} "
                    f"0 0 {_format_float(sphere_back_z)}"
                ),
                "size": "0.003",
                "mass": "0.0001",
                "rgba": "0.12 0.13 0.14 1",
                "contype": "0",
                "conaffinity": "0",
                "group": "1",
            },
        )

    tool_body = ET.SubElement(
        sensor_body,
        "body",
        {
            "name": f"{arm_name}_wiping_tool_body",
            "pos": _format_vector(sphere_position),
        },
    )
    ET.SubElement(
        tool_body,
        "geom",
        {
            "name": f"{arm_name}_wiping_sphere",
            "type": "sphere",
            "size": _format_float(config.collision.radius_m),
            "mass": _format_float(config.mass_kg or 0.02),
            "rgba": _format_tuple(config.collision.rgba),
            "friction": _format_tuple(config.collision.friction),
            "contype": str(config.collision.contype),
            "conaffinity": str(config.collision.conaffinity),
            "group": "1",
        },
    )
    tcp_local = config.tcp_pose.position - sphere_position
    ET.SubElement(
        tool_body,
        "site",
        {
            "name": f"{arm_name}_tool_tcp",
            "type": "sphere",
            "pos": _format_vector(tcp_local),
            "quat": _format_vector(config.tcp_pose.quat),
            "size": "0.0015",
            "rgba": "0.1 0.95 0.2 0.95",
            "group": "2",
        },
    )

    sensors = root.find("sensor")
    if sensors is None:
        sensors = ET.SubElement(root, "sensor")
    force_name = f"{arm_name}_ft_force"
    torque_name = f"{arm_name}_ft_torque"
    _remove_named_child(sensors, "force", force_name)
    _remove_named_child(sensors, "torque", torque_name)
    ET.SubElement(
        sensors,
        "force",
        {"name": force_name, "site": sensor_site_name},
    )
    ET.SubElement(
        sensors,
        "torque",
        {"name": torque_name, "site": sensor_site_name},
    )
    return root


def _find_site_parent_body_and_site(
    root: ET.Element,
    site_name: str,
) -> tuple[ET.Element | None, ET.Element | None]:
    worldbody = root.find("worldbody")
    if worldbody is None:
        return None, None
    return _find_site_recursive(worldbody, site_name)


def _find_site_recursive(
    element: ET.Element,
    site_name: str,
) -> tuple[ET.Element | None, ET.Element | None]:
    if element.tag == "body":
        for child in element:
            if child.tag == "site" and child.get("name") == site_name:
                return element, child
    for child in element:
        body, site = _find_site_recursive(child, site_name)
        if body is not None:
            return body, site
    return None, None


def _remove_named_child(parent: ET.Element, tag: str, name: str) -> None:
    for child in list(parent):
        if child.tag == tag and child.get("name") == name:
            parent.remove(child)


def _parse_vector(raw_value: str) -> np.ndarray:
    values = np.fromstring(raw_value, sep=" ", dtype=float)
    if values.shape != (3,):
        raise ValueError(f"Expected a 3-vector, got {raw_value!r}.")
    return values


def _parse_quaternion(raw_value: str) -> np.ndarray:
    values = np.fromstring(raw_value, sep=" ", dtype=float)
    if values.shape != (4,):
        raise ValueError(f"Expected a quaternion, got {raw_value!r}.")
    return values


def _format_vector(values: np.ndarray) -> str:
    return _format_tuple(tuple(float(value) for value in values))


def _format_tuple(values: tuple[float, ...]) -> str:
    return " ".join(_format_float(value) for value in values)


def _format_float(value: float) -> str:
    return f"{float(value):.9g}"


__all__ = ["inject_enabled_tip_tools", "inject_force_sensor_sphere_tool"]
