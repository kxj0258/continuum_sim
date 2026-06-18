"""Generate the tendon-driven MuJoCo reduced-order model."""

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
        default=Path("configs/mujoco.yaml"),
        help="Path to the MuJoCo backend YAML config.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Override the generated tendon XML output path.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        output_path = build_mujoco_tendon_model(
            config_path=args.config,
            output_path=args.output,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"Failed to generate tendon MuJoCo XML: {exc}", file=sys.stderr)
        return 1

    print(f"tendon_xml_path: {output_path}")
    return 0


def build_mujoco_tendon_model(
    *,
    config_path: Path,
    output_path: Path | None = None,
) -> Path:
    config = load_mujoco_config(config_path, require_xml=True)
    if not config.tendon_model.enabled:
        raise ValueError("tendon_model.enabled must be true before generating tendon XML.")

    params = ThreeSegmentRobotParams.from_yaml(config.robot_config_path)
    physical_tendons = load_physical_tendons_from_yaml(config.robot_config_path)
    target_path = (output_path or config.tendon_xml_path).resolve()

    root = ElementTree.parse(config.xml_path).getroot()
    _apply_solver_config(root, config)
    _apply_rendering_config(root, config)
    _apply_joint_defaults(root, config)
    _apply_position_actuator_defaults(root, config)
    _replace_tendon_section(root, config, params, physical_tendons)
    _replace_actuator_section(root, config, physical_tendons)
    _replace_sensor_section(root, config, physical_tendons)

    target_path.parent.mkdir(parents=True, exist_ok=True)
    _indent(root)
    ElementTree.ElementTree(root).write(
        target_path,
        encoding="utf-8",
        xml_declaration=False,
        short_empty_elements=True,
    )
    return target_path


def _apply_solver_config(root: ElementTree.Element, config) -> None:
    option = root.find("option")
    if option is None:
        option = ElementTree.Element("option")
        root.insert(0, option)
    option.set("timestep", _format_float(config.solver.timestep))
    option.set("integrator", config.solver.integrator)
    option.set("iterations", str(config.solver.iterations))
    option.set("gravity", _format_vec(_effective_gravity_vector(config)))


def _apply_rendering_config(root: ElementTree.Element, config) -> None:
    visual = root.find("visual")
    if visual is None:
        visual = ElementTree.Element("visual")
        insert_index = 1 if root.find("option") is not None else 0
        root.insert(insert_index, visual)
    global_visual = visual.find("global")
    if global_visual is None:
        global_visual = ElementTree.Element("global")
        visual.insert(0, global_visual)
    global_visual.set("azimuth", _format_float(config.viewer.camera.azimuth))
    global_visual.set("elevation", _format_float(config.viewer.camera.elevation))
    global_visual.set("offwidth", str(config.rendering.offscreen_width))
    global_visual.set("offheight", str(config.rendering.offscreen_height))


def _apply_joint_defaults(root: ElementTree.Element, config) -> None:
    default = _default_section(root)
    joint = default.find("joint")
    if joint is None:
        joint = ElementTree.Element("joint")
        default.insert(0, joint)
    hinge = config.joints.hinge
    joint.set("damping", _format_float(hinge.damping))
    joint.set("armature", _format_float(hinge.armature))
    joint.set("limited", _format_bool(hinge.limited))
    joint.set("range", _format_vec(hinge.range_rad))
    joint.set("stiffness", _format_float(hinge.stiffness))
    joint.set("springref", _format_float(hinge.springref))


def _apply_position_actuator_defaults(root: ElementTree.Element, config) -> None:
    default = _default_section(root)
    position = default.find("position")
    if position is None:
        position = ElementTree.Element("position")
        default.append(position)
    actuator = config.actuators.tendon_position
    position.set("kp", _format_float(actuator.kp))
    position.set("ctrllimited", _format_bool(actuator.ctrllimited))
    position.set("ctrlrange", _format_vec(actuator.ctrlrange_m))
    position.set("forcelimited", _format_bool(actuator.forcelimited))
    position.set("forcerange", _format_vec(actuator.forcerange_n))


def _replace_tendon_section(
    root: ElementTree.Element,
    config,
    params: ThreeSegmentRobotParams,
    physical_tendons: tuple[PhysicalTendonPath, ...],
) -> None:
    _remove_direct_children(root, "tendon")
    tendon_section = ElementTree.Element("tendon")
    joint_names = _joint_names(root)

    for tendon in physical_tendons:
        fixed = ElementTree.SubElement(tendon_section, "fixed")
        fixed.set("name", tendon.id)
        fixed.set("limited", _format_bool(config.tendon_model.limited))
        fixed.set("range", _format_vec(config.tendon_model.length_range_m))
        fixed.set("damping", _format_float(config.tendon_model.damping))
        fixed.set("stiffness", _format_float(config.tendon_model.stiffness))
        for joint_name, coef in _fixed_tendon_joint_coefficients(
            tendon,
            params,
            config.links_per_segment,
        ):
            if joint_name not in joint_names:
                raise ValueError(
                    f"Base MuJoCo XML is missing joint {joint_name!r} "
                    f"referenced by {tendon.id}."
                )
            joint = ElementTree.SubElement(fixed, "joint")
            joint.set("joint", joint_name)
            joint.set("coef", _format_float(coef))

    _insert_before(root, tendon_section, "actuator")


def _replace_actuator_section(
    root: ElementTree.Element,
    config,
    physical_tendons: tuple[PhysicalTendonPath, ...],
) -> None:
    _remove_direct_children(root, "actuator")
    actuator_section = ElementTree.Element("actuator")
    actuator = config.actuators.tendon_position
    for tendon in physical_tendons:
        position = ElementTree.SubElement(actuator_section, "position")
        position.set("name", _actuator_name(tendon))
        position.set("tendon", tendon.id)
        position.set("kp", _format_float(actuator.kp))
        position.set("ctrllimited", _format_bool(actuator.ctrllimited))
        position.set("ctrlrange", _format_vec(actuator.ctrlrange_m))
        position.set("forcelimited", _format_bool(actuator.forcelimited))
        position.set("forcerange", _format_vec(actuator.forcerange_n))
    root.append(actuator_section)


def _replace_sensor_section(
    root: ElementTree.Element,
    config,
    physical_tendons: tuple[PhysicalTendonPath, ...],
) -> None:
    _remove_direct_children(root, "sensor")
    sensors = config.sensors
    if not (sensors.tendon_length or sensors.tendon_velocity or sensors.actuator_force):
        return

    sensor_section = ElementTree.Element("sensor")
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
            sensor.set("actuator", _actuator_name(tendon))
    root.append(sensor_section)


def _fixed_tendon_joint_coefficients(
    tendon: PhysicalTendonPath,
    params: ThreeSegmentRobotParams,
    links_per_segment: int,
) -> list[tuple[str, float]]:
    if links_per_segment <= 0:
        raise ValueError(f"links_per_segment must be positive, got {links_per_segment}.")
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
        for link_index in range(links_per_segment):
            link_number = link_index + 1
            coefficients.append(
                (f"segment_{segment_number}_link_{link_number}_x", coef_x)
            )
            coefficients.append(
                (f"segment_{segment_number}_link_{link_number}_y", coef_y)
            )
    return coefficients


def _default_section(root: ElementTree.Element) -> ElementTree.Element:
    default = root.find("default")
    if default is None:
        default = ElementTree.Element("default")
        root.insert(0, default)
    return default


def _joint_names(root: ElementTree.Element) -> set[str]:
    return {
        str(joint.get("name"))
        for joint in root.findall(".//joint")
        if joint.get("name") is not None
    }


def _remove_direct_children(root: ElementTree.Element, tag: str) -> None:
    for child in list(root):
        if child.tag == tag:
            root.remove(child)


def _insert_before(
    root: ElementTree.Element,
    element: ElementTree.Element,
    before_tag: str,
) -> None:
    children = list(root)
    insert_index = next(
        (index for index, child in enumerate(children) if child.tag == before_tag),
        len(children),
    )
    root.insert(insert_index, element)


def _actuator_name(tendon: PhysicalTendonPath) -> str:
    return f"act_{tendon.id}"


def _format_bool(value: bool) -> str:
    return "true" if value else "false"


def _format_float(value: float) -> str:
    return f"{value:.12g}"


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
