"""Reusable engine-scene composition for robot MuJoCo models."""

from __future__ import annotations

import os
from pathlib import Path
import struct
import xml.etree.ElementTree as ET

import numpy as np

from continuum_sim.model.base_pose import Pose6D
from continuum_sim.scenes.engine_scene import (
    EngineSceneConfig,
    effective_engine_frame_position,
    load_engine_scene_config,
    resolve_engine_asset_paths,
)
from continuum_sim.scenes.primitive_collision import PrimitiveCollisionGeomConfig


def build_engine_mujoco_scene_xml(
    robot_xml_path: str | Path,
    engine_config: str | Path | EngineSceneConfig,
    output_xml_path: str | Path,
    *,
    include_visual_mesh: bool = True,
    include_collision_mesh: bool = False,
    include_control_primitives: bool = True,
) -> Path:
    """Compose one single- or dual-arm robot MJCF with the engine scene."""

    robot_path = Path(robot_xml_path).resolve()
    output_path = Path(output_xml_path).resolve()
    config = (
        engine_config
        if isinstance(engine_config, EngineSceneConfig)
        else load_engine_scene_config(engine_config)
    )
    tree = ET.parse(robot_path)
    inject_engine_scene(
        tree.getroot(),
        config,
        output_dir=output_path.parent,
        include_visual_mesh=include_visual_mesh,
        include_collision_mesh=include_collision_mesh,
        include_control_primitives=include_control_primitives,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(tree)
    tree.write(output_path, encoding="utf-8", xml_declaration=False)
    return output_path


def retain_spatial_arm(root: ET.Element, arm_name: str) -> ET.Element:
    """Remove other named arms from a dual-arm MJCF tree in place."""

    arm_names = ("executor", "observer")
    if arm_name not in arm_names:
        raise ValueError(f"arm_name must be one of {arm_names}.")
    removed_names = tuple(name for name in arm_names if name != arm_name)
    for parent in root.iter():
        for child in list(parent):
            if _element_belongs_to_arm(child, removed_names):
                parent.remove(child)
    root.set("model", f"single_spatial_{arm_name}")
    return root


def _element_belongs_to_arm(
    element: ET.Element,
    arm_names: tuple[str, ...],
) -> bool:
    """Return whether any MJCF name or reference belongs to a removed arm."""

    for value in element.attrib.values():
        for arm_name in arm_names:
            if value.startswith(f"{arm_name}_") or value.startswith(f"act_{arm_name}_"):
                return True
    return False


def inject_engine_scene(
    root: ET.Element,
    config: EngineSceneConfig,
    *,
    output_dir: str | Path | None,
    include_visual_mesh: bool = True,
    include_collision_mesh: bool = False,
    include_control_primitives: bool = True,
    mesh_overrides: dict[str, Path] | None = None,
    primitive_collision_enabled: bool = True,
) -> ET.Element:
    """Inject engine assets, body, and enabled control primitives into MJCF."""

    output = None if output_dir is None else Path(output_dir).resolve()
    assets = resolve_engine_asset_paths(config, config.path.parent)
    overrides = mesh_overrides or {}
    visual_mesh = overrides.get("visual_mesh", assets.visual_mesh)
    collision_mesh = overrides.get("collision_mesh", assets.collision_mesh)
    if output is not None:
        visual_mesh = prepare_mujoco_stl(
            visual_mesh,
            output,
            "engine_visual_mujoco",
        )
        if collision_mesh is not None:
            collision_mesh = prepare_mujoco_stl(
                collision_mesh,
                output,
                "engine_collision_mujoco",
            )
    asset = root.find("asset")
    if asset is None:
        asset = ET.SubElement(root, "asset")
    worldbody = root.find("worldbody")
    if worldbody is None:
        worldbody = ET.SubElement(root, "worldbody")
    scale = _vec((config.engine.scale,) * 3)
    if include_visual_mesh:
        ET.SubElement(
            asset,
            "mesh",
            {
                "name": "engine_visual_mesh",
                "file": _relative_path(visual_mesh, output),
                "scale": scale,
            },
        )
    if include_collision_mesh and collision_mesh is not None:
        ET.SubElement(
            asset,
            "mesh",
            {
                "name": "engine_collision_mesh",
                "file": _relative_path(collision_mesh, output),
                "scale": scale,
            },
        )
    engine_body = ET.SubElement(
        worldbody,
        "body",
        {
            "name": "engine",
            "pos": _vec(config.engine.pose.position_m),
            "quat": _vec(config.engine.pose.quat_wxyz),
        },
    )
    if include_visual_mesh:
        ET.SubElement(
            engine_body,
            "geom",
            {
                "name": "engine_visual",
                "type": "mesh",
                "mesh": "engine_visual_mesh",
                "contype": "0",
                "conaffinity": "0",
                "group": "1",
                "rgba": _vec(config.preview_visualization.visual_mesh_rgba),
            },
        )
    if include_collision_mesh and collision_mesh is not None:
        attrs = {
            "name": "engine_collision_mesh_geom",
            "type": "mesh",
            "mesh": "engine_collision_mesh",
            "group": "0",
        }
        if config.engine.assets.collision_mesh_offset_m is not None:
            attrs["pos"] = _vec(config.engine.assets.collision_mesh_offset_m)
        ET.SubElement(engine_body, "geom", attrs)
    if include_control_primitives:
        for geom in config.primitive_collision_geoms:
            if geom.enabled:
                _add_primitive(
                    worldbody,
                    geom,
                    config,
                    collision_enabled=primitive_collision_enabled,
                )
    return root


def rebase_mjcf_file_assets(
    root: ET.Element,
    source_xml_dir: str | Path,
    output_xml_dir: str | Path,
) -> ET.Element:
    """Rebase all existing MJCF ``file`` attributes for a moved document."""

    source_dir = Path(source_xml_dir).resolve()
    output_dir = Path(output_xml_dir).resolve()
    for element in root.iter():
        raw_path = element.get("file")
        if not raw_path:
            continue
        path = Path(raw_path)
        resolved = path if path.is_absolute() else (source_dir / path).resolve()
        element.set("file", _relative_path(resolved, output_dir))
    return root


def prepare_mujoco_stl(
    path: str | Path,
    output_dir: str | Path,
    stem: str,
    *,
    max_faces: int = 200_000,
) -> Path:
    """Return a MuJoCo-loadable binary STL, limiting face count when needed."""

    source = Path(path).resolve()
    if source.suffix.lower() != ".stl":
        return source
    data = source.read_bytes()
    if len(data) < 84:
        return source
    triangle_count = struct.unpack("<I", data[80:84])[0]
    if len(data) != 84 + triangle_count * 50:
        return source
    if triangle_count <= max_faces:
        return source
    target_dir = Path(output_dir).resolve()
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"{stem}_max_{max_faces}.stl"
    with target.open("wb") as stream:
        stream.write(data[:80])
        stream.write(struct.pack("<I", max_faces))
        for index in range(max_faces):
            source_index = int(index * triangle_count / max_faces)
            start = 84 + source_index * 50
            stream.write(data[start : start + 50])
    return target


def _add_primitive(
    worldbody: ET.Element,
    geom: PrimitiveCollisionGeomConfig,
    config: EngineSceneConfig,
    *,
    collision_enabled: bool,
) -> None:
    common = {
        "name": f"engine_primitive_{geom.name}",
        "type": geom.type,
        "group": "0",
        "contype": "1" if collision_enabled else "0",
        "conaffinity": "1" if collision_enabled else "0",
        "rgba": _vec(geom.rgba or (0.9, 0.2, 0.1, 0.35)),
    }
    if geom.type == "capsule" and geom.fromto_m is not None:
        start, end, scale = _fromto_world(geom, config)
        common["fromto"] = _vec((*start, *end))
        common["size"] = f"{float(geom.radius_m) * scale:.12g}"
        ET.SubElement(worldbody, "geom", common)
        return
    pose, dimension_scale = _primitive_pose_world(geom, config)
    body = ET.SubElement(
        worldbody,
        "body",
        {
            "name": f"engine_primitive_body_{geom.name}",
            "pos": _vec(pose.position),
            "quat": _vec(pose.quat),
        },
    )
    if geom.type == "box":
        common["size"] = _vec(0.5 * np.asarray(geom.size_m) * dimension_scale)
    elif geom.type == "sphere":
        common["size"] = f"{float(geom.radius_m) * dimension_scale:.12g}"
    elif geom.type in ("cylinder", "capsule"):
        common["size"] = _vec(
            (
                float(geom.radius_m) * dimension_scale,
                0.5 * float(geom.length_m) * dimension_scale,
            )
        )
    ET.SubElement(body, "geom", common)


def _primitive_pose_world(
    geom: PrimitiveCollisionGeomConfig,
    config: EngineSceneConfig,
) -> tuple[Pose6D, float]:
    local = Pose6D(
        position=np.asarray(geom.position_m, dtype=float),
        quat=(
            np.array([1.0, 0.0, 0.0, 0.0])
            if geom.quat_wxyz is None
            else geom.quat_wxyz
        ),
    )
    if geom.frame == "world":
        return local, 1.0
    engine = Pose6D(
        position=effective_engine_frame_position(config),
        quat=config.engine.pose.quat_wxyz,
    )
    return engine.compose(
        Pose6D(position=local.position * config.engine.scale, quat=local.quat)
    ), float(config.engine.scale)


def _fromto_world(
    geom: PrimitiveCollisionGeomConfig,
    config: EngineSceneConfig,
) -> tuple[np.ndarray, np.ndarray, float]:
    points = np.asarray(geom.fromto_m, dtype=float).reshape(2, 3)
    if geom.frame == "world":
        return points[0], points[1], 1.0
    engine = Pose6D(
        position=effective_engine_frame_position(config),
        quat=config.engine.pose.quat_wxyz,
    )
    transformed = engine.transform_points(points * config.engine.scale)
    return transformed[0], transformed[1], float(config.engine.scale)


def _relative_path(path: Path, output_dir: Path | None) -> str:
    if output_dir is None:
        return path.resolve().as_posix()
    return Path(os.path.relpath(path.resolve(), output_dir)).as_posix()


def _vec(values: object) -> str:
    return " ".join(f"{float(value):.12g}" for value in values)
