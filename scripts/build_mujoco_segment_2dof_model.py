"""Generate the 2DOF-per-segment follower MuJoCo tendon model."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from xml.etree import ElementTree

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from continuum_sim.config import load_mujoco_config
from continuum_sim.model import ThreeSegmentRobotParams, load_physical_tendons_from_yaml
from continuum_sim.model.physical_tendon import PhysicalTendonPath


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/mujoco_segment_2dof.yaml"),
        help="Path to the MuJoCo segment-2DOF YAML config.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Override the generated 2DOF tendon XML output path.",
    )
    parser.add_argument(
        "--visual-output",
        type=Path,
        default=None,
        help="Override the generated 2DOF tendon XML with follower visuals.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        tendon_path, visual_path = build_mujoco_segment_2dof_model(
            config_path=args.config,
            output_path=args.output,
            visual_output_path=args.visual_output,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"Failed to generate segment-2DOF MuJoCo XML: {exc}", file=sys.stderr)
        return 1

    print(f"segment_2dof_tendon_xml_path: {tendon_path}")
    print(f"segment_2dof_tendon_visual_xml_path: {visual_path}")
    return 0


def build_mujoco_segment_2dof_model(
    *,
    config_path: Path,
    output_path: Path | None = None,
    visual_output_path: Path | None = None,
) -> tuple[Path, Path]:
    config = load_mujoco_config(
        config_path,
        require_xml=False,
        require_tendon_xml=False,
        require_visual_meshes=False,
    )
    if config.model.type != "segment_2dof_followers":
        raise ValueError("model.type must be 'segment_2dof_followers'.")
    if not config.tendon_model.enabled:
        raise ValueError("tendon_model.enabled must be true before generating tendon XML.")

    params = ThreeSegmentRobotParams.from_yaml(config.robot_config_path)
    physical_tendons = load_physical_tendons_from_yaml(config.robot_config_path)
    tendon_path = (output_path or config.tendon_xml_path).resolve()
    visual_path = (visual_output_path or config.tendon_generated_xml_path).resolve()

    base_root = _build_root(config, params, physical_tendons, include_visual_geoms=False)
    _write_xml(base_root, tendon_path)
    visual_root = _build_root(config, params, physical_tendons, include_visual_geoms=True)
    _write_xml(visual_root, visual_path)
    return tendon_path, visual_path


def _build_root(
    config,
    params: ThreeSegmentRobotParams,
    physical_tendons: tuple[PhysicalTendonPath, ...],
    *,
    include_visual_geoms: bool,
) -> ElementTree.Element:
    root = ElementTree.Element(
        "mujoco",
        {"model": "three_segment_2dof_follower_arm"},
    )
    ElementTree.SubElement(root, "compiler", {"angle": "radian"})
    ElementTree.SubElement(
        root,
        "option",
        {
            "timestep": _format_float(config.solver.timestep),
            "integrator": config.solver.integrator,
            "iterations": str(config.solver.iterations),
            "gravity": _format_vec(_effective_gravity_vector(config)),
        },
    )
    visual = ElementTree.SubElement(root, "visual")
    ElementTree.SubElement(
        visual,
        "global",
        {
            "azimuth": _format_float(config.viewer.camera.azimuth),
            "elevation": _format_float(config.viewer.camera.elevation),
            "offwidth": str(config.rendering.offscreen_width),
            "offheight": str(config.rendering.offscreen_height),
        },
    )
    ElementTree.SubElement(
        visual,
        "headlight",
        {
            "ambient": "0.35 0.35 0.35",
            "diffuse": "0.75 0.75 0.75",
            "specular": "0.2 0.2 0.2",
        },
    )
    ElementTree.SubElement(visual, "map", {"znear": "0.001", "zfar": "2"})
    ElementTree.SubElement(visual, "quality", {"shadowsize": "2048"})
    _append_defaults(root, config)
    _append_worldbody(root, config, params, include_visual_geoms=include_visual_geoms)
    _append_tendons(root, config, params, physical_tendons)
    _append_actuators(root, config, physical_tendons)
    _append_sensors(root, config, physical_tendons)
    return root


def _append_defaults(root: ElementTree.Element, config) -> None:
    default = ElementTree.SubElement(root, "default")
    hinge = config.joints.hinge
    ElementTree.SubElement(
        default,
        "joint",
        {
            "damping": _format_float(hinge.damping),
            "armature": _format_float(hinge.armature),
            "limited": _format_bool(hinge.limited),
            "range": _format_vec(hinge.range_rad),
            "stiffness": _format_float(hinge.stiffness),
            "springref": _format_float(hinge.springref),
        },
    )
    ElementTree.SubElement(
        default,
        "geom",
        {
            "type": "capsule",
            "size": "0.004",
            "density": "1100",
            "rgba": "0.2 0.45 0.85 1",
            "group": str(config.visuals.collision_geom_group),
        },
    )
    ElementTree.SubElement(default, "site", {"size": "0.003", "rgba": "0.95 0.25 0.2 1"})
    actuator = config.actuators.tendon_position
    ElementTree.SubElement(
        default,
        "position",
        {
            "kp": _format_float(actuator.kp),
            "ctrllimited": _format_bool(actuator.ctrllimited),
            "ctrlrange": _format_vec(actuator.ctrlrange_m),
            "forcelimited": _format_bool(actuator.forcelimited),
            "forcerange": _format_vec(actuator.forcerange_n),
        },
    )


def _append_worldbody(
    root: ElementTree.Element,
    config,
    params: ThreeSegmentRobotParams,
    *,
    include_visual_geoms: bool,
) -> None:
    worldbody = ElementTree.SubElement(root, "worldbody")
    base = ElementTree.SubElement(worldbody, "body", {"name": "base", "pos": "0 0 0"})
    ElementTree.SubElement(
        base,
        "geom",
        {
            "name": "base_geom",
            "type": "cylinder",
            "size": "0.012 0.004",
            "pos": "0 0 -0.004",
            "rgba": "0.12 0.12 0.12 1",
            "group": str(config.visuals.collision_geom_group),
            "contype": "0",
            "conaffinity": "0",
        },
    )
    ElementTree.SubElement(base, "site", {"name": "base_site", "pos": "0 0 0", "quat": "1 0 0 0"})

    parent = base
    for segment_index, segment in enumerate(params.segments):
        segment_number = segment_index + 1
        body = ElementTree.SubElement(
            parent,
            "body",
            {
                "name": f"segment_{segment_number}",
                "pos": "0 0 0",
            },
        )
        ElementTree.SubElement(
            body,
            "joint",
            {
                "name": f"segment_{segment_number}_x",
                "type": "hinge",
                "axis": "1 0 0",
            },
        )
        ElementTree.SubElement(
            body,
            "joint",
            {
                "name": f"segment_{segment_number}_y",
                "type": "hinge",
                "axis": "0 1 0",
            },
        )
        ElementTree.SubElement(
            body,
            "geom",
            {
                "name": f"segment_{segment_number}_backbone_geom",
                "fromto": _format_vec((0.0, 0.0, 0.0, 0.0, 0.0, segment.length)),
                "size": _format_float(segment.tendon_radius),
                "group": str(config.visuals.collision_geom_group),
                "contype": "0",
                "conaffinity": "0",
            },
        )
        ElementTree.SubElement(
            body,
            "site",
            {
                "name": f"segment_{segment_number}_tip",
                "pos": _format_vec((0.0, 0.0, segment.length)),
                "quat": "1 0 0 0",
            },
        )
        if segment_index == params.segment_count - 1:
            ElementTree.SubElement(
                body,
                "site",
                {
                    "name": "tip",
                    "pos": _format_vec((0.0, 0.0, segment.length)),
                    "quat": "1 0 0 0",
                },
            )
        parent = ElementTree.SubElement(
            body,
            "body",
            {
                "name": f"segment_{segment_number}_tip_frame",
                "pos": _format_vec((0.0, 0.0, segment.length)),
            },
        )

    _append_follower_bodies(
        worldbody,
        config,
        params,
        include_visual_geoms=include_visual_geoms,
    )


def _append_follower_bodies(
    worldbody: ElementTree.Element,
    config,
    params: ThreeSegmentRobotParams,
    *,
    include_visual_geoms: bool,
) -> None:
    if not config.model.follower_collision and not include_visual_geoms:
        return
    samples = config.model.follower_samples_per_segment
    for segment_index, segment in enumerate(params.segments):
        sample_length = segment.length / float(samples)
        half_length = max(0.5 * sample_length, 1.0e-9)
        for sample_index in range(samples):
            body_name = f"follower_segment_{segment_index + 1}_sample_{sample_index + 1}"
            body = ElementTree.SubElement(
                worldbody,
                "body",
                {
                    "name": body_name,
                    "mocap": "true",
                    "pos": _format_vec(
                        (
                            0.0,
                            0.0,
                            sum(s.length for s in params.segments[:segment_index])
                            + (sample_index + 0.5) * sample_length,
                        )
                    ),
                },
            )
            if config.model.follower_collision:
                ElementTree.SubElement(
                    body,
                    "geom",
                    {
                        "name": f"{body_name}_collision",
                        "type": "capsule",
                        "fromto": _format_vec(
                            (0.0, 0.0, -half_length, 0.0, 0.0, half_length)
                        ),
                        "size": _format_float(segment.tendon_radius),
                        "rgba": "0.15 0.45 0.85 0.28",
                        "group": str(config.visuals.collision_geom_group),
                        "contype": "1",
                        "conaffinity": "1",
                    },
                )
            if include_visual_geoms and config.model.follower_visuals:
                ElementTree.SubElement(
                    body,
                    "geom",
                    {
                        "name": f"{body_name}_visual",
                        "type": "capsule",
                        "fromto": _format_vec(
                            (0.0, 0.0, -half_length, 0.0, 0.0, half_length)
                        ),
                        "size": _format_float(segment.tendon_radius * 0.82),
                        "rgba": "0.05 0.7 0.9 0.42",
                        "group": str(config.visuals.visual_geom_group),
                        "contype": "0",
                        "conaffinity": "0",
                        "density": "0",
                    },
                )


def _append_tendons(
    root: ElementTree.Element,
    config,
    params: ThreeSegmentRobotParams,
    physical_tendons: tuple[PhysicalTendonPath, ...],
) -> None:
    tendon_section = ElementTree.SubElement(root, "tendon")
    for tendon in physical_tendons:
        fixed = ElementTree.SubElement(tendon_section, "fixed")
        fixed.set("name", tendon.id)
        fixed.set("limited", _format_bool(config.tendon_model.limited))
        fixed.set("range", _format_vec(config.tendon_model.length_range_m))
        fixed.set("damping", _format_float(config.tendon_model.damping))
        fixed.set("stiffness", _format_float(config.tendon_model.stiffness))
        for joint_name, coef in _segment_tendon_joint_coefficients(tendon, params):
            joint = ElementTree.SubElement(fixed, "joint")
            joint.set("joint", joint_name)
            joint.set("coef", _format_float(coef))


def _append_actuators(
    root: ElementTree.Element,
    config,
    physical_tendons: tuple[PhysicalTendonPath, ...],
) -> None:
    actuator_section = ElementTree.SubElement(root, "actuator")
    actuator = config.actuators.tendon_position
    for tendon in physical_tendons:
        position = ElementTree.SubElement(actuator_section, "position")
        position.set("name", f"act_{tendon.id}")
        position.set("tendon", tendon.id)
        position.set("kp", _format_float(actuator.kp))
        position.set("ctrllimited", _format_bool(actuator.ctrllimited))
        position.set("ctrlrange", _format_vec(actuator.ctrlrange_m))
        position.set("forcelimited", _format_bool(actuator.forcelimited))
        position.set("forcerange", _format_vec(actuator.forcerange_n))


def _append_sensors(
    root: ElementTree.Element,
    config,
    physical_tendons: tuple[PhysicalTendonPath, ...],
) -> None:
    sensors = config.sensors
    if not (sensors.tendon_length or sensors.tendon_velocity or sensors.actuator_force):
        return
    sensor_section = ElementTree.SubElement(root, "sensor")
    for tendon in physical_tendons:
        if sensors.tendon_length:
            sensor = ElementTree.SubElement(sensor_section, "tendonpos")
            sensor.set("name", f"sensor_{tendon.id}_length")
            sensor.set("tendon", tendon.id)
        if sensors.tendon_velocity:
            sensor = ElementTree.SubElement(sensor_section, "tendonvel")
            sensor.set("name", f"sensor_{tendon.id}_velocity")
            sensor.set("tendon", tendon.id)
        if sensors.actuator_force:
            sensor = ElementTree.SubElement(sensor_section, "actuatorfrc")
            sensor.set("name", f"sensor_{tendon.id}_actuator_force")
            sensor.set("actuator", f"act_{tendon.id}")


def _segment_tendon_joint_coefficients(
    tendon: PhysicalTendonPath,
    params: ThreeSegmentRobotParams,
) -> list[tuple[str, float]]:
    theta_rad = math.radians(tendon.angle_deg)
    coef_x = tendon.radial_offset * math.sin(theta_rad)
    coef_y = -tendon.radial_offset * math.cos(theta_rad)
    coefficients: list[tuple[str, float]] = []
    for segment_index in tendon.path_segment_indices:
        if segment_index < 0 or segment_index >= len(params.segments):
            raise ValueError(
                f"{tendon.id} references invalid segment index {segment_index}."
            )
        segment_number = segment_index + 1
        coefficients.append((f"segment_{segment_number}_x", coef_x))
        coefficients.append((f"segment_{segment_number}_y", coef_y))
    return coefficients


def _write_xml(root: ElementTree.Element, target_path: Path) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    _indent(root)
    ElementTree.ElementTree(root).write(
        target_path,
        encoding="utf-8",
        xml_declaration=False,
        short_empty_elements=True,
    )


def _format_bool(value: bool) -> str:
    return "true" if value else "false"


def _format_float(value: float) -> str:
    return f"{float(value):.12g}"


def _format_vec(values: tuple[float, ...]) -> str:
    return " ".join(_format_float(value) for value in values)


def _effective_gravity_vector(config) -> tuple[float, float, float]:
    if not config.gravity.enabled:
        return (0.0, 0.0, 0.0)
    return config.gravity.vector_m_s2


def _indent(element: ElementTree.Element, level: int = 0) -> None:
    prefix = "\n" + level * "  "
    child_prefix = "\n" + (level + 1) * "  "
    children = list(element)
    if children:
        if not element.text or not element.text.strip():
            element.text = child_prefix
        for child in children:
            _indent(child, level + 1)
        if not children[-1].tail or not children[-1].tail.strip():
            children[-1].tail = prefix
    if level and (not element.tail or not element.tail.strip()):
        element.tail = prefix


if __name__ == "__main__":
    raise SystemExit(main())
