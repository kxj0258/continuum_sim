"""Generate the dual-arm MuJoCo tendon model with SolidWorks visual meshes."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
import sys
from pathlib import Path
from xml.etree import ElementTree

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from continuum_sim.config import load_mujoco_config, load_yaml
from continuum_sim.model import ThreeSegmentRobotParams, load_dual_arm_robot_config
from continuum_sim.model.dual_arm_robot import DualArmRobotConfig
from continuum_sim.model.hole_pattern import TendonHolePattern, load_tendon_hole_pattern
from continuum_sim.model.multi_arm import ArmConfig, load_multi_arm_config
from continuum_sim.model.physical_tendon import PhysicalTendonPath
from continuum_sim.scenes.scene_builder import inject_mobile_base_wrapper


@dataclass(frozen=True)
class DualArmModelBuildResult:
    """Paths written by one reproducible dual-arm model build."""

    base_xml_path: Path
    mobile_base_xml_path: Path | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/mujoco_dual.yaml"),
        help="Path to the dual-arm MuJoCo backend YAML config.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Override the generated dual-arm XML output path.",
    )
    parser.add_argument(
        "--mobile-base-output",
        type=Path,
        default=None,
        help="Override the generated mobile-base-wrapped XML output path.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        result = build_mujoco_dual_arm_model(
            config_path=args.config,
            output_path=args.output,
            mobile_base_output_path=args.mobile_base_output,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"Failed to generate dual-arm MuJoCo XML: {exc}", file=sys.stderr)
        return 1

    print(f"dual_arm_tendon_xml_path: {result.base_xml_path}")
    if result.mobile_base_xml_path is not None:
        print(f"dual_arm_mobile_base_xml_path: {result.mobile_base_xml_path}")
    return 0


def build_mujoco_dual_arm_model(
    *,
    config_path: Path,
    output_path: Path | None = None,
    mobile_base_output_path: Path | None = None,
) -> DualArmModelBuildResult:
    config = load_mujoco_config(
        config_path,
        require_xml=False,
        require_tendon_xml=False,
    )
    if config.model.type != "dual_distributed_links":
        raise ValueError("model.type must be 'dual_distributed_links'.")
    if not config.tendon_model.enabled:
        raise ValueError("tendon_model.enabled must be true before generating tendon XML.")
    if config.tendon_model.type != "spatial":
        raise ValueError("dual-arm tendon_model.type must be 'spatial'.")
    if config.multi_arm_config_path is None:
        raise ValueError("multi_arm_config_path is required for dual-arm generation.")
    if config.dual_arm_mesh_config_path is None:
        raise ValueError("dual_arm_mesh_config_path is required for dual-arm generation.")
    if config.dual_arm_hole_pattern_config_path is None:
        raise ValueError("dual_arm_hole_pattern_config_path is required for spatial tendon generation.")

    dual_robot = load_dual_arm_robot_config(config.robot_config_path)
    multi_arm = load_multi_arm_config(config.multi_arm_config_path, strict_paths=False)
    mesh_manifest = _load_mesh_manifest(config.dual_arm_mesh_config_path)
    hole_pattern = load_tendon_hole_pattern(config.dual_arm_hole_pattern_config_path)
    target_path = (output_path or config.tendon_xml_path).resolve()
    arms = _ordered_enabled_arms(multi_arm)
    _require_matching_arm_configs(dual_robot, arms)

    root = _build_root(
        config=config,
        dual_robot=dual_robot,
        arms=arms,
        mesh_manifest=mesh_manifest,
        hole_pattern=hole_pattern,
        target_path=target_path,
    )
    _write_xml(root, target_path)
    mobile_target = _mobile_base_output_path(
        config,
        target_path,
        output_overridden=output_path is not None,
        override=mobile_base_output_path,
    )
    if mobile_target is not None:
        if config.mobile_base_config_path is None:
            raise ValueError(
                "mobile_base_config_path is required to generate mobile-base XML."
            )
        inject_mobile_base_wrapper(
            target_path,
            mobile_target,
            config.mobile_base_config_path,
        )
    return DualArmModelBuildResult(
        base_xml_path=target_path,
        mobile_base_xml_path=mobile_target,
    )


def _build_root(
    *,
    config,
    dual_robot: DualArmRobotConfig,
    arms: tuple[ArmConfig, ...],
    mesh_manifest: dict[str, object],
    hole_pattern: TendonHolePattern,
    target_path: Path,
) -> ElementTree.Element:
    root = ElementTree.Element("mujoco", {"model": "dual_three_segment_arm"})
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
    _append_visual_settings(root, config)
    _append_assets(root, config, mesh_manifest, target_path)
    _append_defaults(root, config)
    _append_worldbody(root, config, dual_robot, arms, mesh_manifest, hole_pattern)
    _append_tendons(root, config, dual_robot, arms, hole_pattern)
    _append_actuators(root, config, dual_robot, arms)
    _append_sensors(root, config, dual_robot, arms)
    return root


def _append_visual_settings(root: ElementTree.Element, config) -> None:
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


def _append_assets(
    root: ElementTree.Element,
    config,
    mesh_manifest: dict[str, object],
    target_path: Path,
) -> None:
    asset = ElementTree.SubElement(root, "asset")
    for mesh_name, mesh_path in _iter_mesh_assets(mesh_manifest):
        ElementTree.SubElement(
            asset,
            "mesh",
            {
                "name": mesh_name,
                "file": _mesh_file_reference(mesh_path, target_path.parent),
                "scale": _format_vec((config.visuals.mesh_scale,) * 3),
            },
        )


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


def _append_worldbody(
    root: ElementTree.Element,
    config,
    dual_robot: DualArmRobotConfig,
    arms: tuple[ArmConfig, ...],
    mesh_manifest: dict[str, object],
    hole_pattern: TendonHolePattern,
) -> None:
    worldbody = ElementTree.SubElement(root, "worldbody")
    _append_world_frame_sites(worldbody, config)
    base_body = ElementTree.SubElement(
        worldbody,
        "body",
        {"name": "dual_mobile_base", "pos": "0 0 0", "quat": "1 0 0 0"},
    )
    ElementTree.SubElement(
        base_body,
        "site",
        {"name": "dual_mobile_base_frame", "pos": "0 0 0", "quat": "1 0 0 0"},
    )
    _append_visual_geom(
        body=base_body,
        name="dual_mobile_base_visual",
        mesh_name="shared_base_visual_mesh",
        config=config,
        body_origin=(0.0, 0.0, 0.0),
        mesh_manifest=mesh_manifest,
    )

    for arm in arms:
        _append_arm(
            base_body=base_body,
            config=config,
            params=dual_robot.params_by_arm[arm.name],
            physical_tendons=dual_robot.tendons_by_arm[arm.name],
            arm=arm,
            mesh_manifest=mesh_manifest,
            hole_pattern=hole_pattern,
        )


def _append_world_frame_sites(
    worldbody: ElementTree.Element,
    config,
) -> None:
    frame = config.visuals.world_frame
    if not frame.enabled:
        return
    ElementTree.SubElement(
        worldbody,
        "site",
        {
            "name": "world_origin",
            "type": "sphere",
            "pos": "0 0 0",
            "size": _format_float(frame.origin_radius_m),
            "rgba": _format_vec(frame.origin_rgba),
            "group": str(frame.geom_group),
        },
    )
    axes = (
        ("x", (frame.axis_length_m, 0.0, 0.0), frame.x_rgba),
        ("y", (0.0, frame.axis_length_m, 0.0), frame.y_rgba),
        ("z", (0.0, 0.0, frame.axis_length_m), frame.z_rgba),
    )
    for axis, endpoint, rgba in axes:
        ElementTree.SubElement(
            worldbody,
            "site",
            {
                "name": f"world_{axis}_axis",
                "type": "cylinder",
                "fromto": _format_vec((0.0, 0.0, 0.0, *endpoint)),
                "size": _format_float(frame.axis_radius_m),
                "rgba": _format_vec(rgba),
                "group": str(frame.geom_group),
            },
        )


def _append_arm(
    *,
    base_body: ElementTree.Element,
    config,
    params: ThreeSegmentRobotParams,
    physical_tendons: tuple[PhysicalTendonPath, ...],
    arm: ArmConfig,
    mesh_manifest: dict[str, object],
    hole_pattern: TendonHolePattern,
) -> None:
    arm_name = arm.name
    arm_mesh = _arm_mesh_entry(mesh_manifest, arm_name)
    collision_enabled = bool(arm_mesh.get("collision_enabled", arm.role == "executor"))
    mount_pos = tuple(float(value) for value in arm.mount.pose.position)
    mount_quat = tuple(float(value) for value in arm.mount.pose.quat)
    arm_body = ElementTree.SubElement(
        base_body,
        "body",
        {
            "name": f"{arm_name}_base",
            "pos": _format_vec(mount_pos),
            "quat": _format_vec(mount_quat),
        },
    )
    ElementTree.SubElement(
        arm_body,
        "site",
        {"name": f"{arm_name}_base_site", "pos": "0 0 0", "quat": "1 0 0 0"},
    )
    _append_tendon_base_sites(
        arm_body,
        arm_name,
        hole_pattern,
        routed_hole_indices=_routed_hole_indices(
            physical_tendons,
            segment_index=None,
        ),
    )
    _append_visual_geom(
        body=arm_body,
        name=f"{arm_name}_base_visual",
        mesh_name=f"{arm_name}_base_visual_mesh",
        config=config,
        body_origin=mount_pos,
        mesh_manifest=mesh_manifest,
    )

    parent = arm_body
    straight_z = 0.0
    link_index = 0
    for segment_index, segment in enumerate(params.segments):
        segment_number = segment_index + 1
        link_length = segment.length / float(config.links_per_segment)
        for local_link_index in range(config.links_per_segment):
            link_number = local_link_index + 1
            link_index += 1
            body_origin = (
                mount_pos[0],
                mount_pos[1],
                mount_pos[2] + straight_z,
            )
            body_name = f"{arm_name}_segment_{segment_number}_link_{link_number}"
            body_pos = (0.0, 0.0, 0.0) if link_index == 1 else (0.0, 0.0, link_length)
            body = ElementTree.SubElement(
                parent,
                "body",
                {"name": body_name, "pos": _format_vec(body_pos)},
            )
            ElementTree.SubElement(
                body,
                "joint",
                {
                    "name": f"{body_name}_x",
                    "type": "hinge",
                    "axis": "1 0 0",
                },
            )
            ElementTree.SubElement(
                body,
                "joint",
                {
                    "name": f"{body_name}_y",
                    "type": "hinge",
                    "axis": "0 1 0",
                },
            )
            geom_attrs = {
                "name": f"{body_name}_collision",
                "type": "capsule",
                "fromto": _format_vec((0.0, 0.0, 0.0, 0.0, 0.0, link_length)),
                "size": _format_float(segment.tendon_radius),
                "group": str(config.visuals.collision_geom_group),
            }
            if not collision_enabled:
                geom_attrs.update(
                    {
                        "contype": "0",
                        "conaffinity": "0",
                        "rgba": "0.2 0.45 0.85 0",
                    }
                )
            ElementTree.SubElement(body, "geom", geom_attrs)

            _append_visual_geom(
                body=body,
                name=f"{body_name}_visual",
                mesh_name=f"{body_name}_visual_mesh",
                config=config,
                body_origin=body_origin,
                mesh_manifest=mesh_manifest,
            )
            if link_number == config.links_per_segment:
                ElementTree.SubElement(
                    body,
                    "site",
                    {
                        "name": f"{arm_name}_segment_{segment_number}_tip",
                        "pos": _format_vec((0.0, 0.0, link_length)),
                        "quat": "1 0 0 0",
                    },
                )
            if link_index == params.segment_count * config.links_per_segment:
                ElementTree.SubElement(
                    body,
                    "site",
                    {
                        "name": f"{arm_name}_tip",
                        "pos": _format_vec((0.0, 0.0, link_length)),
                        "quat": "1 0 0 0",
                    },
                )
            _append_tendon_link_sites(
                body=body,
                body_name=body_name,
                arm_name=arm_name,
                global_link_number=link_index,
                segment_number=segment_number,
                segment_link_number=link_number,
                hole_pattern=hole_pattern,
                routed_hole_indices=_routed_hole_indices(
                    physical_tendons,
                    segment_index=segment_index,
                ),
            )
            straight_z += link_length
            parent = body


def _append_tendon_base_sites(
    body: ElementTree.Element,
    arm_name: str,
    hole_pattern: TendonHolePattern,
    *,
    routed_hole_indices: set[int],
) -> None:
    for hole in hole_pattern.base_holes:
        if not _emit_hole_site(
            hole_pattern,
            hole.index,
            routed_hole_indices,
        ):
            continue
        for suffix, z in (
            ("in", hole.in_z_m),
            ("out", hole.out_z_m),
        ):
            ElementTree.SubElement(
                body,
                "site",
                {
                    "name": _base_hole_site_name(arm_name, hole.index, suffix),
                    "pos": _format_vec((hole.xy_m[0], hole.xy_m[1], z)),
                    "size": _format_float(hole_pattern.site_generation.site_size_m),
                    "rgba": _format_vec(
                        _hole_site_rgba(
                            hole_pattern,
                            suffix,
                            visible=_hole_site_visible(hole_pattern),
                        )
                    ),
                },
            )


def _append_tendon_link_sites(
    *,
    body: ElementTree.Element,
    body_name: str,
    arm_name: str,
    global_link_number: int,
    segment_number: int,
    segment_link_number: int,
    hole_pattern: TendonHolePattern,
    routed_hole_indices: set[int],
) -> None:
    endpoints_by_suffix = (
        (
            "in",
            hole_pattern.link_in_endpoints(
                global_link_number=global_link_number,
                segment_number=segment_number,
                segment_link_number=segment_link_number,
            ),
        ),
        (
            "out",
            hole_pattern.link_out_endpoints(
                arm_name=arm_name,
                global_link_number=global_link_number,
                segment_number=segment_number,
                segment_link_number=segment_link_number,
            ),
        ),
    )
    for suffix, endpoints in endpoints_by_suffix:
        endpoint_indices = {endpoint.index for endpoint in endpoints}
        missing_routed_indices = routed_hole_indices - endpoint_indices
        if missing_routed_indices:
            raise ValueError(
                f"{body_name} is missing routed {suffix} hole indices "
                f"{sorted(missing_routed_indices)}."
            )
        for endpoint in endpoints:
            if not _emit_hole_site(
                hole_pattern,
                endpoint.index,
                routed_hole_indices,
            ):
                continue
            ElementTree.SubElement(
                body,
                "site",
                {
                    "name": _link_hole_site_name(
                        body_name,
                        endpoint.index,
                        suffix,
                    ),
                    "pos": _format_vec(
                        (endpoint.xy_m[0], endpoint.xy_m[1], endpoint.z_m)
                    ),
                    "size": _format_float(
                        hole_pattern.site_generation.site_size_m
                    ),
                    "rgba": _format_vec(
                        _hole_site_rgba(
                            hole_pattern,
                            suffix,
                            visible=_hole_site_visible(hole_pattern),
                        )
                    ),
                },
            )


def _hole_site_rgba(
    hole_pattern: TendonHolePattern,
    suffix: str,
    *,
    visible: bool,
) -> tuple[float, float, float, float]:
    if suffix == "in":
        rgba = hole_pattern.site_generation.in_site_rgba
    elif suffix == "out":
        rgba = hole_pattern.site_generation.out_site_rgba
    else:
        raise ValueError(f"Unknown hole site suffix {suffix!r}.")
    if visible:
        return rgba
    return (rgba[0], rgba[1], rgba[2], 0.0)


def _routed_hole_indices(
    physical_tendons: tuple[PhysicalTendonPath, ...],
    *,
    segment_index: int | None,
) -> set[int]:
    return {
        tendon.hole_index
        for tendon in physical_tendons
        if tendon.hole_index is not None
        and (
            segment_index is None
            or segment_index in tendon.path_segment_indices
        )
    }


def _emit_hole_site(
    hole_pattern: TendonHolePattern,
    hole_index: int,
    routed_hole_indices: set[int],
) -> bool:
    if hole_pattern.visualization.hole_display == "all":
        return True
    return hole_index in routed_hole_indices


def _hole_site_visible(hole_pattern: TendonHolePattern) -> bool:
    return hole_pattern.visualization.hole_display != "none"


def _append_visual_geom(
    *,
    body: ElementTree.Element,
    name: str,
    mesh_name: str,
    config,
    body_origin: tuple[float, float, float],
    mesh_manifest: dict[str, object],
) -> None:
    ElementTree.SubElement(
        body,
        "geom",
        {
            "name": name,
            "type": "mesh",
            "mesh": mesh_name,
            "pos": _format_vec(
                _visual_geom_pos(
                    frame_mode=_alignment_frame_mode(mesh_manifest),
                    cad_origin_mm=_alignment_cad_origin_mm(mesh_manifest),
                    mesh_scale=config.visuals.mesh_scale,
                    body_origin=body_origin,
                )
            ),
            "contype": "0",
            "conaffinity": "0",
            "density": "0",
            "group": str(config.visuals.visual_geom_group),
        },
    )


def _append_tendons(
    root: ElementTree.Element,
    config,
    dual_robot: DualArmRobotConfig,
    arms: tuple[ArmConfig, ...],
    hole_pattern: TendonHolePattern,
) -> None:
    tendon_section = ElementTree.SubElement(root, "tendon")
    for arm in arms:
        params = dual_robot.params_by_arm[arm.name]
        for tendon in dual_robot.tendons_by_arm[arm.name]:
            tendon_name = _tendon_xml_name(arm.name, tendon)
            tendon_rgba = _tendon_rgba(tendon.global_index)
            if not hole_pattern.visualization.show_tendons:
                tendon_rgba = (
                    tendon_rgba[0],
                    tendon_rgba[1],
                    tendon_rgba[2],
                    0.0,
                )
            spatial = ElementTree.SubElement(
                tendon_section,
                "spatial",
                {
                    "name": tendon_name,
                    "damping": _format_float(config.tendon_model.damping),
                    "stiffness": _format_float(config.tendon_model.stiffness),
                    "width": _format_float(config.viewer.overlays.tendon_path_radius),
                    "rgba": _format_vec(tendon_rgba),
                },
            )
            for site_name in _spatial_tendon_site_names(
                arm.name,
                tendon,
                params,
                config.links_per_segment,
            ):
                ElementTree.SubElement(
                    spatial,
                    "site",
                    {"site": site_name},
                )


def _append_actuators(
    root: ElementTree.Element,
    config,
    dual_robot: DualArmRobotConfig,
    arms: tuple[ArmConfig, ...],
) -> None:
    actuator_section = ElementTree.SubElement(root, "actuator")
    actuator = config.actuators.tendon_position
    for arm in arms:
        for tendon in dual_robot.tendons_by_arm[arm.name]:
            tendon_name = _tendon_xml_name(arm.name, tendon)
            ElementTree.SubElement(
                actuator_section,
                "position",
                _actuator_attrs_for_tendon(
                    config,
                    actuator,
                    tendon_name,
                ),
            )


def _actuator_attrs_for_tendon(config, actuator, tendon_name: str) -> dict[str, str]:
    attrs = {
        "name": f"act_{tendon_name}",
        "tendon": tendon_name,
        "kp": _format_float(actuator.kp),
        "forcelimited": _format_bool(actuator.forcelimited),
        "forcerange": _format_vec(actuator.forcerange_n),
        "ctrllimited": _format_bool(actuator.ctrllimited),
    }
    return attrs


def _mobile_base_output_path(
    config,
    base_xml_path: Path,
    *,
    output_overridden: bool,
    override: Path | None,
) -> Path | None:
    if override is not None:
        return override.resolve()
    if config.mobile_base_config_path is None:
        return None
    if not output_overridden and config.mobile_base_xml_path is not None:
        return config.mobile_base_xml_path.resolve()
    return base_xml_path.with_name(
        f"{base_xml_path.stem}_mobile_base{base_xml_path.suffix}"
    )


def _append_sensors(
    root: ElementTree.Element,
    config,
    dual_robot: DualArmRobotConfig,
    arms: tuple[ArmConfig, ...],
) -> None:
    sensors = config.sensors
    if not (sensors.tendon_length or sensors.tendon_velocity or sensors.actuator_force):
        return
    sensor_section = ElementTree.SubElement(root, "sensor")
    for arm in arms:
        for tendon in dual_robot.tendons_by_arm[arm.name]:
            tendon_name = _tendon_xml_name(arm.name, tendon)
            actuator_name = f"act_{tendon_name}"
            if sensors.tendon_length:
                ElementTree.SubElement(
                    sensor_section,
                    "tendonpos",
                    {"name": f"sensor_{tendon_name}_length", "tendon": tendon_name},
                )
            if sensors.tendon_velocity:
                ElementTree.SubElement(
                    sensor_section,
                    "tendonvel",
                    {"name": f"sensor_{tendon_name}_velocity", "tendon": tendon_name},
                )
            if sensors.actuator_force:
                ElementTree.SubElement(
                    sensor_section,
                    "actuatorfrc",
                    {"name": f"sensor_{tendon_name}_actuator_force", "actuator": actuator_name},
                )


def _spatial_tendon_site_names(
    arm_name: str,
    tendon: PhysicalTendonPath,
    params: ThreeSegmentRobotParams,
    links_per_segment: int,
) -> list[str]:
    if tendon.hole_index is None:
        raise ValueError(f"{tendon.id} must define hole_index for spatial tendons.")
    site_names = [
        _base_hole_site_name(arm_name, tendon.hole_index, "in"),
        _base_hole_site_name(arm_name, tendon.hole_index, "out"),
    ]
    for segment_index in tendon.path_segment_indices:
        if segment_index < 0 or segment_index >= len(params.segments):
            raise ValueError(
                f"{tendon.id} references invalid segment index {segment_index}."
            )
        segment_number = segment_index + 1
        for link_number in range(1, links_per_segment + 1):
            prefix = f"{arm_name}_segment_{segment_number}_link_{link_number}"
            site_names.append(_link_hole_site_name(prefix, tendon.hole_index, "in"))
            site_names.append(_link_hole_site_name(prefix, tendon.hole_index, "out"))
    return site_names


def _ordered_enabled_arms(multi_arm) -> tuple[ArmConfig, ...]:
    arms = [arm for arm in multi_arm.arms.values() if arm.enabled]
    default_arm = multi_arm.default_arm
    if default_arm is None:
        return tuple(arms)
    return tuple(
        sorted(
            arms,
            key=lambda arm: (0 if arm.name == default_arm else 1, arm.name),
        )
    )


def _require_matching_arm_configs(
    dual_robot: DualArmRobotConfig,
    arms: tuple[ArmConfig, ...],
) -> None:
    enabled_arm_names = {arm.name for arm in arms}
    robot_arm_names = set(dual_robot.arm_names)
    missing_in_robot = enabled_arm_names - robot_arm_names
    if missing_in_robot:
        raise ValueError(
            "multi_arm_config_path enables arms missing from robot_config_path: "
            f"{sorted(missing_in_robot)}."
        )
    missing_in_multi_arm = robot_arm_names - enabled_arm_names
    if missing_in_multi_arm:
        raise ValueError(
            "robot_config_path defines arms not enabled by multi_arm_config_path: "
            f"{sorted(missing_in_multi_arm)}."
        )
    enabled_order = tuple(arm.name for arm in arms)
    if enabled_order != dual_robot.arm_names:
        raise ValueError(
            "Enabled multi-arm order must match robot_config_path arm order so "
            "MuJoCo actuator order matches dual-arm spatial tendon indices, got "
            f"{enabled_order} and {dual_robot.arm_names}."
        )


def _load_mesh_manifest(path: Path) -> dict[str, object]:
    raw = load_yaml(path)
    root = Path(str(raw.get("root", "")))
    if not root.is_absolute():
        root = (path.parent / root).resolve()
    raw["root"] = root
    _require_mesh_path(root / _shared_base_mesh(raw), "shared_base.visual_mesh")
    for arm_name, arm_raw in _arms_mesh_entries(raw).items():
        if not isinstance(arm_raw, dict):
            raise ValueError(f"dual_arm_meshes arms.{arm_name} must be a mapping.")
        _require_mesh_path(root / str(arm_raw["base_visual_mesh"]), f"arms.{arm_name}.base_visual_mesh")
        for segment_number in range(1, 4):
            for link_number in range(1, 5):
                rel = _link_mesh_file(arm_raw, segment_number, link_number)
                _require_mesh_path(
                    root / rel,
                    f"arms.{arm_name}.link_visual_meshes.segment_{segment_number}.link_{link_number}",
                )
    return raw


def _iter_mesh_assets(mesh_manifest: dict[str, object]) -> list[tuple[str, Path]]:
    root = Path(mesh_manifest["root"])
    assets = [("shared_base_visual_mesh", root / _shared_base_mesh(mesh_manifest))]
    for arm_name, arm_raw in _arms_mesh_entries(mesh_manifest).items():
        if not isinstance(arm_raw, dict):
            raise ValueError(f"arms.{arm_name} must be a mapping.")
        assets.append((f"{arm_name}_base_visual_mesh", root / str(arm_raw["base_visual_mesh"])))
        for segment_number in range(1, 4):
            for link_number in range(1, 5):
                assets.append(
                    (
                        f"{arm_name}_segment_{segment_number}_link_{link_number}_visual_mesh",
                        root / _link_mesh_file(arm_raw, segment_number, link_number),
                    )
                )
    return assets


def _shared_base_mesh(mesh_manifest: dict[str, object]) -> str:
    shared_base = mesh_manifest.get("shared_base", {})
    if not isinstance(shared_base, dict):
        raise ValueError("dual_arm_meshes.shared_base must be a mapping.")
    return str(shared_base["visual_mesh"])


def _arms_mesh_entries(mesh_manifest: dict[str, object]) -> dict[str, object]:
    arms = mesh_manifest.get("arms", {})
    if not isinstance(arms, dict) or not arms:
        raise ValueError("dual_arm_meshes.arms must be a non-empty mapping.")
    return arms


def _arm_mesh_entry(mesh_manifest: dict[str, object], arm_name: str) -> dict[str, object]:
    arms = _arms_mesh_entries(mesh_manifest)
    if arm_name not in arms or not isinstance(arms[arm_name], dict):
        raise ValueError(f"dual_arm_meshes.arms is missing arm {arm_name!r}.")
    return arms[arm_name]  # type: ignore[return-value]


def _link_mesh_file(arm_raw: dict[str, object], segment_number: int, link_number: int) -> str:
    link_visual_meshes = arm_raw.get("link_visual_meshes", {})
    if not isinstance(link_visual_meshes, dict):
        raise ValueError("link_visual_meshes must be a mapping.")
    segment_raw = link_visual_meshes.get(f"segment_{segment_number}", {})
    if not isinstance(segment_raw, dict):
        raise ValueError(f"segment_{segment_number} must be a mapping.")
    return str(segment_raw[f"link_{link_number}"])


def _alignment_frame_mode(mesh_manifest: dict[str, object]) -> str:
    alignment = mesh_manifest.get("alignment", {})
    if not isinstance(alignment, dict):
        return "cad_global"
    return str(alignment.get("frame_mode", "cad_global"))


def _alignment_cad_origin_mm(mesh_manifest: dict[str, object]) -> tuple[float, float, float]:
    alignment = mesh_manifest.get("alignment", {})
    if not isinstance(alignment, dict):
        return (0.0, 0.0, 0.0)
    raw = alignment.get("cad_origin_mm", (0.0, 0.0, 0.0))
    if not isinstance(raw, list | tuple) or len(raw) != 3:
        raise ValueError("alignment.cad_origin_mm must contain exactly 3 numbers.")
    return tuple(float(value) for value in raw)  # type: ignore[return-value]


def _visual_geom_pos(
    *,
    frame_mode: str,
    cad_origin_mm: tuple[float, float, float],
    mesh_scale: float,
    body_origin: tuple[float, float, float],
) -> tuple[float, float, float]:
    if frame_mode == "body_local":
        return (0.0, 0.0, 0.0)
    if frame_mode != "cad_global":
        raise ValueError(f"Unsupported dual-arm mesh alignment.frame_mode: {frame_mode!r}")
    cad_origin_model = tuple(value * mesh_scale for value in cad_origin_mm)
    return tuple(
        -cad_origin_model[index] - body_origin[index]
        for index in range(3)
    )


def _tendon_xml_name(arm_name: str, tendon: PhysicalTendonPath) -> str:
    if tendon.id.startswith(f"{arm_name}_"):
        return tendon.id
    return f"{arm_name}_{tendon.id}"


def _base_hole_site_name(arm_name: str, hole_index: int, suffix: str) -> str:
    return f"{arm_name}_base_hole_{hole_index + 1:02d}_{suffix}"


def _link_hole_site_name(body_name: str, hole_index: int, suffix: str) -> str:
    return f"{body_name}_hole_{hole_index + 1:02d}_{suffix}"


def _tendon_rgba(index: int) -> tuple[float, float, float, float]:
    palette = (
        (0.92, 0.20, 0.16, 1.0),
        (0.95, 0.55, 0.12, 1.0),
        (0.98, 0.84, 0.20, 1.0),
        (0.36, 0.73, 0.19, 1.0),
        (0.10, 0.73, 0.61, 1.0),
        (0.09, 0.60, 0.92, 1.0),
        (0.29, 0.43, 0.95, 1.0),
        (0.64, 0.33, 0.94, 1.0),
        (0.90, 0.26, 0.70, 1.0),
        (0.55, 0.32, 0.16, 1.0),
        (0.15, 0.15, 0.15, 1.0),
        (0.55, 0.70, 0.90, 1.0),
    )
    rgba = palette[index % len(palette)]
    if index >= len(palette):
        return (0.75 * rgba[0], 0.75 * rgba[1], 0.75 * rgba[2], rgba[3])
    return rgba


def _require_mesh_path(path: Path, field_name: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{field_name} does not exist: {path}")


def _write_xml(root: ElementTree.Element, target_path: Path) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    _indent(root)
    ElementTree.ElementTree(root).write(
        target_path,
        encoding="utf-8",
        xml_declaration=False,
        short_empty_elements=True,
    )


def _mesh_file_reference(mesh_path: Path, xml_directory: Path) -> str:
    mesh_path = mesh_path.resolve()
    try:
        relpath = os.path.relpath(mesh_path, xml_directory.resolve())
    except ValueError:
        return mesh_path.as_posix()
    return relpath.replace(os.sep, "/")


def _effective_gravity_vector(config) -> tuple[float, float, float]:
    if not config.gravity.enabled:
        return (0.0, 0.0, 0.0)
    return config.gravity.vector_m_s2


def _format_bool(value: bool) -> str:
    return "true" if value else "false"


def _format_float(value: float) -> str:
    return f"{float(value):.12g}"


def _format_vec(values: tuple[float, ...]) -> str:
    return " ".join(_format_float(value) for value in values)


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
