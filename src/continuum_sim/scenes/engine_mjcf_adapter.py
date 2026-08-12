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
    visual_meshes = (visual_mesh,)
    collision_meshes = () if collision_mesh is None else (collision_mesh,)
    if output is not None:
        visual_meshes = prepare_mujoco_stl_parts(
            visual_mesh,
            output,
            "engine_visual_mujoco",
        )
        if collision_mesh is not None:
            collision_meshes = prepare_mujoco_stl_parts(
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
        visual_mesh_names = _add_mesh_assets(
            asset,
            base_name="engine_visual_mesh",
            paths=visual_meshes,
            output_dir=output,
            scale=scale,
        )
        material = config.preview_visualization.visual_material
        if material is not None:
            _remove_named_asset(asset, "material", material.name)
            ET.SubElement(
                asset,
                "material",
                {
                    "name": material.name,
                    "rgba": _vec(config.preview_visualization.visual_mesh_rgba),
                    "emission": f"{material.emission:g}",
                    "specular": f"{material.specular:g}",
                    "shininess": f"{material.shininess:g}",
                },
            )
    else:
        visual_mesh_names = ()
    if include_collision_mesh and collision_meshes:
        collision_mesh_names = _add_mesh_assets(
            asset,
            base_name="engine_collision_mesh",
            paths=collision_meshes,
            output_dir=output,
            scale=scale,
        )
    else:
        collision_mesh_names = ()
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
        material = config.preview_visualization.visual_material
        for index, mesh_name in enumerate(visual_mesh_names):
            visual_attrs = {
                "name": _part_name("engine_visual", index, len(visual_mesh_names)),
                "type": "mesh",
                "mesh": mesh_name,
                "contype": "0",
                "conaffinity": "0",
                "group": "1",
                # Explicit RGBA prevents the robot model's blue default geom
                # color from overriding the engine material.
                "rgba": _vec(config.preview_visualization.visual_mesh_rgba),
            }
            if material is not None:
                visual_attrs["material"] = material.name
            ET.SubElement(engine_body, "geom", visual_attrs)
    if include_collision_mesh and collision_mesh_names:
        for index, mesh_name in enumerate(collision_mesh_names):
            attrs = {
                "name": _part_name(
                    "engine_collision_mesh_geom",
                    index,
                    len(collision_mesh_names),
                ),
                "type": "mesh",
                "mesh": mesh_name,
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


def _remove_named_asset(parent: ET.Element, tag: str, name: str) -> None:
    for child in list(parent):
        if child.tag == tag and child.get("name") == name:
            parent.remove(child)


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


def prepare_mujoco_stl_parts(
    path: str | Path,
    output_dir: str | Path,
    stem: str,
    *,
    max_faces: int = 200_000,
) -> tuple[Path, ...]:
    """Return lossless STL parts whose individual face counts fit MuJoCo."""

    if max_faces <= 0:
        raise ValueError("max_faces must be positive.")

    source = Path(path).resolve()
    if source.suffix.lower() != ".stl":
        return (source,)
    data = source.read_bytes()
    if len(data) < 84:
        return (source,)
    triangle_count = struct.unpack("<I", data[80:84])[0]
    if len(data) != 84 + triangle_count * 50:
        return (source,)
    if triangle_count <= max_faces:
        return (source,)
    target_dir = Path(output_dir).resolve()
    target_dir.mkdir(parents=True, exist_ok=True)
    part_count = (triangle_count + max_faces - 1) // max_faces
    targets: list[Path] = []
    for part_index in range(part_count):
        first_face = part_index * max_faces
        face_count = min(max_faces, triangle_count - first_face)
        target = target_dir / f"{stem}_part_{part_index + 1:02d}.stl"
        start = 84 + first_face * 50
        stop = start + face_count * 50
        with target.open("wb") as stream:
            stream.write(data[:80])
            stream.write(struct.pack("<I", face_count))
            stream.write(data[start:stop])
        targets.append(target)
    return tuple(targets)


def _add_mesh_assets(
    asset: ET.Element,
    *,
    base_name: str,
    paths: tuple[Path, ...],
    output_dir: Path | None,
    scale: str,
) -> tuple[str, ...]:
    names = tuple(_part_name(base_name, index, len(paths)) for index in range(len(paths)))
    for name, path in zip(names, paths, strict=True):
        ET.SubElement(
            asset,
            "mesh",
            {
                "name": name,
                "file": _relative_path(path, output_dir),
                "scale": scale,
            },
        )
    return names


def _part_name(base_name: str, index: int, count: int) -> str:
    return base_name if count == 1 else f"{base_name}_part_{index + 1}"


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
