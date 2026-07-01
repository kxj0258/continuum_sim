"""Preview the configured engine mesh and region markers in a minimal MuJoCo scene."""

from __future__ import annotations

import argparse
import struct
import sys
import tempfile
from pathlib import Path
from xml.etree import ElementTree

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from continuum_sim.config import load_yaml  # noqa: E402
from continuum_sim.scenes.engine_scene import (  # noqa: E402
    EngineSceneConfig,
    EngineRegionConfig,
    effective_engine_frame_position,
    load_engine_scene_config,
    resolve_engine_asset_paths,
)
from continuum_sim.scenes.engine_mjcf_adapter import inject_engine_scene  # noqa: E402
from continuum_sim.scenes.primitive_collision import (  # noqa: E402
    PrimitiveCollisionGeomConfig,
    load_primitive_collision_geoms,
)
from scripts.check_engine_assets import collect_engine_scene_diagnostics, transform_points  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if not args.headless_check and not args.viewer:
        print("Pass --headless-check or --viewer.")
        return 1
    if args.visual_only and args.collision_only:
        print("ERROR: --visual-only and --collision-only are mutually exclusive.", file=sys.stderr)
        return 1

    config = load_engine_scene_config(args.config)
    asset_paths = resolve_engine_asset_paths(config, config.path.parent)
    if not args.collision_only and not asset_paths.visual_mesh.exists():
        print(f"ERROR: visual mesh does not exist: {asset_paths.visual_mesh}", file=sys.stderr)
        return 1
    if not args.visual_only and asset_paths.collision_mesh is not None and not asset_paths.collision_mesh.exists():
        print(f"ERROR: collision mesh does not exist: {asset_paths.collision_mesh}", file=sys.stderr)
        return 1

    try:
        import mujoco
    except ImportError:
        print("ERROR: Python package 'mujoco' is not installed in this environment.", file=sys.stderr)
        return 1

    with tempfile.TemporaryDirectory(prefix="engine_preview_mujoco_") as temp_dir:
        mesh_overrides = _prepare_preview_meshes(
            config,
            Path(temp_dir),
            include_visual=not args.collision_only,
            include_collision=not args.visual_only and not args.hide_mesh_collision,
        )
        xml_text = build_engine_preview_mjcf(
            config_path=args.config,
            mesh_overrides=mesh_overrides,
            visual_only=args.visual_only,
            collision_only=args.collision_only,
            alpha_visual=args.alpha_visual,
            alpha_collision=args.alpha_collision,
            show_primitive_collision=args.show_primitive_collision,
            show_disabled_hints=args.show_disabled_hints,
            primitive_alpha=args.primitive_alpha,
            hide_mesh_collision=args.hide_mesh_collision,
            show_exploration_paths=not args.hide_exploration_paths,
        )
        xml_path = Path(temp_dir) / "engine_preview.xml"
        xml_path.write_text(xml_text, encoding="utf-8")
        try:
            model = mujoco.MjModel.from_xml_path(str(xml_path))
        except Exception as exc:
            print(f"ERROR: MuJoCo failed to load generated MJCF: {exc}", file=sys.stderr)
            return 1
        print(
            "MuJoCo model loaded: "
            f"nbody={model.nbody}, ngeom={model.ngeom}, nsite={model.nsite}, nmesh={model.nmesh}"
        )
        camera = _suggest_camera(args.config)
        print(
            "Recommended camera: "
            f"lookat={_mujoco_vec(camera['lookat'])}, distance={camera['distance']:.6g}, "
            f"azimuth={camera['azimuth']:.6g}, elevation={camera['elevation']:.6g}"
        )

        if args.viewer:
            try:
                import mujoco.viewer
            except ImportError:
                print("ERROR: mujoco.viewer is not available in this environment.", file=sys.stderr)
                return 1
            print("Opening MuJoCo viewer. Close the viewer window to exit.")
            data = mujoco.MjData(model)
            with mujoco.viewer.launch_passive(model, data) as viewer:
                viewer.cam.lookat[:] = camera["lookat"]
                viewer.cam.distance = camera["distance"]
                viewer.cam.azimuth = camera["azimuth"]
                viewer.cam.elevation = camera["elevation"]
                while viewer.is_running():
                    viewer.sync()

    return 0


def build_engine_preview_mjcf(
    config_path: str | Path = Path("configs/scenes/engine_cleaning.yaml"),
    *,
    mesh_overrides: dict[str, Path] | None = None,
    visual_only: bool = False,
    collision_only: bool = False,
    show_bbox: bool = True,
    show_regions: bool = True,
    show_axes: bool = True,
    alpha_visual: float | None = None,
    alpha_collision: float | None = None,
    show_primitive_collision: bool = True,
    show_disabled_hints: bool = False,
    primitive_alpha: float = 0.55,
    hide_mesh_collision: bool = False,
    show_exploration_paths: bool = True,
) -> str:
    """Build a minimal MJCF document for the configured engine mesh."""

    config = load_engine_scene_config(config_path)
    asset_paths = resolve_engine_asset_paths(config, config.path.parent)
    overrides = mesh_overrides or {}
    visual_mesh = overrides.get("visual_mesh", asset_paths.visual_mesh)
    collision_mesh = overrides.get("collision_mesh", asset_paths.collision_mesh)
    include_visual = not collision_only
    include_collision = not visual_only and not hide_mesh_collision and collision_mesh is not None
    root = ElementTree.Element("mujoco", {"model": "engine_preview"})
    compiler = ElementTree.SubElement(root, "compiler")
    compiler.set("angle", "radian")
    compiler.set("autolimits", "true")
    ElementTree.SubElement(root, "option", {"timestep": "0.01"})

    inject_engine_scene(
        root,
        config,
        output_dir=None,
        include_visual_mesh=include_visual,
        include_collision_mesh=include_collision,
        include_control_primitives=show_primitive_collision,
        mesh_overrides={
            "visual_mesh": visual_mesh,
            **(
                {"collision_mesh": collision_mesh}
                if collision_mesh is not None
                else {}
            ),
        },
        primitive_collision_enabled=False,
    )
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError("Engine scene adapter did not create a worldbody.")
    diagnostics = collect_engine_scene_diagnostics(config_path)
    visual_report = next(
        (report for report in diagnostics.asset_reports if report.asset_name == "visual_mesh"),
        None,
    )
    preview = config.preview_visualization
    axis_length = _marker_scale_from_bbox(visual_report)
    if show_axes:
        _add_engine_axes(worldbody, config, axis_length=axis_length)

    if show_bbox and visual_report is not None:
        _add_bbox_marker(worldbody, visual_report, config)

    if show_regions:
        for region in config.regions.values():
            _add_region_site(worldbody, region, config=config)

    _add_exploration_start(worldbody, config, axis_length=axis_length)

    if show_exploration_paths:
        _add_exploration_paths(worldbody, config)

    ElementTree.indent(root)
    return ElementTree.tostring(root, encoding="unicode")


def _prepare_preview_meshes(
    config,
    temp_dir: Path,
    *,
    include_visual: bool = True,
    include_collision: bool = True,
) -> dict[str, Path]:
    asset_paths = resolve_engine_asset_paths(config, config.path.parent)
    overrides: dict[str, Path] = {}
    if include_visual:
        visual_mesh = _mujoco_ready_stl(asset_paths.visual_mesh, temp_dir, "engine_visual_preview")
        if visual_mesh != asset_paths.visual_mesh:
            overrides["visual_mesh"] = visual_mesh
    if include_collision and asset_paths.collision_mesh is not None:
        collision_mesh = _mujoco_ready_stl(
            asset_paths.collision_mesh,
            temp_dir,
            "engine_collision_preview",
        )
        if collision_mesh != asset_paths.collision_mesh:
            overrides["collision_mesh"] = collision_mesh
    return overrides


def _mujoco_ready_stl(path: Path, temp_dir: Path, stem: str) -> Path:
    if path.suffix.lower() != ".stl":
        return path
    data = path.read_bytes()
    if len(data) < 84:
        return path
    triangle_count = struct.unpack("<I", data[80:84])[0]
    if len(data) != 84 + triangle_count * 50:
        return path
    if triangle_count <= 200_000:
        return path

    target_count = 200_000
    output_path = temp_dir / f"{stem}_max_{target_count}.stl"
    with output_path.open("wb") as stream:
        stream.write(data[:80])
        stream.write(struct.pack("<I", target_count))
        for index in range(target_count):
            source_index = int(index * triangle_count / target_count)
            start = 84 + source_index * 50
            stream.write(data[start : start + 50])
    print(
        f"Generated temporary MuJoCo preview STL for {path.name}: "
        f"{triangle_count} -> {target_count} faces."
    )
    return output_path


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/scenes/engine_cleaning.yaml"),
        help="Engine scene YAML config to preview.",
    )
    parser.add_argument(
        "--headless-check",
        action="store_true",
        help="Generate MJCF and verify MuJoCo can load it without opening a viewer.",
    )
    parser.add_argument("--viewer", action="store_true", help="Open an interactive MuJoCo viewer.")
    parser.add_argument("--visual-only", action="store_true", help="Show only the visual mesh geom.")
    parser.add_argument("--collision-only", action="store_true", help="Show only the collision mesh geom.")
    parser.add_argument(
        "--alpha-visual",
        type=float,
        default=None,
        help="Override the alpha channel of preview_visualization.visual_mesh_rgba.",
    )
    parser.add_argument(
        "--alpha-collision",
        type=float,
        default=None,
        help="Override the alpha channel of preview_visualization.collision_mesh_rgba.",
    )
    parser.add_argument(
        "--show-primitive-collision",
        action="store_true",
        default=True,
        help="Show enabled primitive_collision_geoms hints.",
    )
    parser.add_argument(
        "--show-disabled-hints",
        action="store_true",
        help="Show disabled primitive_collision_geoms hints as low-alpha preview geometry.",
    )
    parser.add_argument("--primitive-alpha", type=float, default=0.55, help="Primitive hint alpha.")
    parser.add_argument(
        "--hide-mesh-collision",
        action="store_true",
        help="Hide configured mesh collision while keeping primitive hints visible.",
    )
    parser.add_argument(
        "--hide-exploration-paths",
        action="store_true",
        help="Hide configured exploration path overlays.",
    )
    return parser.parse_args(argv)


def _add_engine_axes(
    worldbody: ElementTree.Element,
    config: EngineSceneConfig,
    *,
    axis_length: float,
) -> None:
    preview = config.preview_visualization
    axis_length = preview.engine_axis_length_m or axis_length
    radius = preview.engine_axis_radius_m or max(axis_length * 0.01, 0.003)
    origin = effective_engine_frame_position(config)
    quat = tuple(float(value) for value in config.engine.pose.quat_wxyz)
    x_axis = transform_points(
        [[0.0, 0.0, 0.0], [axis_length, 0.0, 0.0]],
        position=tuple(float(value) for value in origin),
        quat_wxyz=quat,
        scale=1.0,
    )
    y_axis = transform_points(
        [[0.0, 0.0, 0.0], [0.0, axis_length, 0.0]],
        position=tuple(float(value) for value in origin),
        quat_wxyz=quat,
        scale=1.0,
    )
    z_axis = transform_points(
        [[0.0, 0.0, 0.0], [0.0, 0.0, axis_length]],
        position=tuple(float(value) for value in origin),
        quat_wxyz=quat,
        scale=1.0,
    )
    ElementTree.SubElement(
        worldbody,
        "site",
        {
            "name": "engine_x_axis",
            "type": "capsule",
            "fromto": f"{_mujoco_vec(x_axis[0])} {_mujoco_vec(x_axis[1])}",
            "size": f"{radius:.12g}",
            "rgba": _mujoco_vec(preview.engine_x_rgba),
        },
    )
    ElementTree.SubElement(
        worldbody,
        "site",
        {
            "name": "engine_y_axis",
            "type": "capsule",
            "fromto": f"{_mujoco_vec(y_axis[0])} {_mujoco_vec(y_axis[1])}",
            "size": f"{radius:.12g}",
            "rgba": _mujoco_vec(preview.engine_y_rgba),
        },
    )
    ElementTree.SubElement(
        worldbody,
        "site",
        {
            "name": "engine_z_axis",
            "type": "capsule",
            "fromto": f"{_mujoco_vec(z_axis[0])} {_mujoco_vec(z_axis[1])}",
            "size": f"{radius:.12g}",
            "rgba": _mujoco_vec(preview.engine_z_rgba),
        },
    )


def _add_bbox_marker(worldbody: ElementTree.Element, visual_report, config: EngineSceneConfig) -> None:
    bbox_min = visual_report.bbox_min_world
    bbox_max = visual_report.bbox_max_world
    bbox_size = visual_report.bbox_size_world
    if bbox_min is None or bbox_max is None or bbox_size is None:
        return
    x0, y0, z0 = bbox_min
    x1, y1, z1 = bbox_max
    corners = [
        (x0, y0, z0),
        (x1, y0, z0),
        (x1, y1, z0),
        (x0, y1, z0),
        (x0, y0, z1),
        (x1, y0, z1),
        (x1, y1, z1),
        (x0, y1, z1),
    ]
    edges = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    ]
    preview = config.preview_visualization
    radius = preview.bbox_edge_radius_m or max(max(bbox_size) * 0.002, 0.002)
    for index, (start_index, end_index) in enumerate(edges):
        ElementTree.SubElement(
            worldbody,
            "site",
            {
                "name": f"bbox_edge_{index}",
                "type": "capsule",
                "fromto": f"{_mujoco_vec(corners[start_index])} {_mujoco_vec(corners[end_index])}",
                "size": f"{radius:.12g}",
                "rgba": _mujoco_vec(preview.bbox_rgba),
            },
        )


def _add_region_site(
    worldbody: ElementTree.Element,
    region: EngineRegionConfig,
    *,
    config: EngineSceneConfig,
) -> None:
    attrs = {
        "name": f"region_{region.name}",
        "rgba": _region_rgba(region, config),
    }
    if region.type == "circular_port" and region.center_m is not None:
        attrs.update(
            {
                "type": "sphere",
                "pos": _mujoco_vec(_region_point_world(region.center_m, region, config)),
                "size": str(region.radius_m or 0.01),
            }
        )
    elif region.type in ("roi_box", "box") and region.center_m is not None and region.size_m is not None:
        attrs.update(
            {
                "type": "box",
                "pos": _mujoco_vec(_region_point_world(region.center_m, region, config)),
                "size": _mujoco_vec(tuple(float(value) * 0.5 for value in region.size_m)),
            }
        )
    elif region.type == "surface_patch" and region.position_m is not None and region.extents_m is not None:
        attrs.update(
            {
                "type": "box",
                "pos": _mujoco_vec(_region_point_world(region.position_m, region, config)),
                "size": _mujoco_vec(tuple(max(float(value) * 0.5, 0.002) for value in region.extents_m)),
            }
        )
    elif region.center_m is not None:
        attrs.update(
            {
                "type": "sphere",
                "pos": _mujoco_vec(_region_point_world(region.center_m, region, config)),
                "size": "0.01",
            }
        )
    else:
        return
    ElementTree.SubElement(worldbody, "site", attrs)


def _add_exploration_start(
    worldbody: ElementTree.Element,
    config: EngineSceneConfig,
    *,
    axis_length: float,
) -> None:
    start = config.exploration_start
    if start is None:
        return
    preview = config.preview_visualization
    point = _exploration_point_world(start.point_m, start.frame, config)
    normal = _exploration_normal_world(start.normal, start.frame, config)
    marker_size = start.point_radius_m or preview.exploration_start_point_radius_m or max(axis_length * 0.025, 0.008)
    normal_length = start.normal_length_m or preview.exploration_start_normal_length_m or max(axis_length * 0.35, 0.12)
    normal_radius = (
        start.normal_radius_m
        or preview.exploration_start_normal_radius_m
        or max(marker_size * 0.35, 0.003)
    )
    group = start.group if start.group is not None else preview.exploration_start_group
    normal_end = point + normal * normal_length

    ElementTree.SubElement(
        worldbody,
        "site",
        {
            "name": "exploration_start_point",
            "type": "sphere",
            "pos": _mujoco_vec(point),
            "size": f"{marker_size:.12g}",
            "rgba": _mujoco_vec(start.point_rgba or preview.exploration_start_point_rgba),
            "group": str(group),
        },
    )
    ElementTree.SubElement(
        worldbody,
        "site",
        {
            "name": "exploration_start_normal",
            "type": "capsule",
            "fromto": f"{_mujoco_vec(point)} {_mujoco_vec(normal_end)}",
            "size": f"{normal_radius:.12g}",
            "rgba": _mujoco_vec(start.normal_rgba or preview.exploration_start_normal_rgba),
            "group": str(group),
        },
    )


def _add_exploration_paths(
    worldbody: ElementTree.Element,
    config: EngineSceneConfig,
) -> None:
    preview = config.preview_visualization
    for path in config.exploration_paths:
        if not path.enabled:
            continue
        group = path.group if path.group is not None else preview.exploration_path_group
        points = (
            _engine_frame_points_to_world(path.points_m, config)
            if path.frame == "engine"
            else np.asarray(path.points_m, dtype=float)
        )
        for index, (start, end) in enumerate(zip(points[:-1], points[1:], strict=True)):
            ElementTree.SubElement(
                worldbody,
                "geom",
                {
                    "name": f"exploration_{path.name}_segment_{index}",
                    "type": "capsule",
                    "fromto": _mujoco_vec((*start, *end)),
                    "size": f"{path.radius_m:.12g}",
                    "rgba": _mujoco_vec(path.rgba),
                    "contype": "0",
                    "conaffinity": "0",
                    "group": str(group),
                },
            )
        marker_size = (
            path.marker_radius_m
            or preview.exploration_path_marker_radius_m
            or max(path.radius_m * 1.8, 0.006)
        )
        ElementTree.SubElement(
            worldbody,
            "site",
            {
                "name": f"exploration_{path.name}_start",
                "type": "sphere",
                "pos": _mujoco_vec(points[0]),
                "size": f"{marker_size:.12g}",
                "rgba": _mujoco_vec(path.start_marker_rgba or preview.exploration_path_start_marker_rgba),
                "group": str(group),
            },
        )
        ElementTree.SubElement(
            worldbody,
            "site",
            {
                "name": f"exploration_{path.name}_end",
                "type": "sphere",
                "pos": _mujoco_vec(points[-1]),
                "size": f"{marker_size:.12g}",
                "rgba": _mujoco_vec(path.end_marker_rgba or preview.exploration_path_end_marker_rgba),
                "group": str(group),
            },
        )


def _region_point_world(
    point_m: object,
    region: EngineRegionConfig,
    config: EngineSceneConfig,
) -> np.ndarray:
    if region.frame == "engine":
        return _engine_frame_points_to_world([point_m], config)[0]
    return np.asarray(point_m, dtype=float)


def _exploration_point_world(
    point_m: object,
    frame: str,
    config: EngineSceneConfig,
) -> np.ndarray:
    if frame == "engine":
        return _engine_frame_points_to_world([point_m], config)[0]
    return np.asarray(point_m, dtype=float)


def _exploration_normal_world(
    normal: object,
    frame: str,
    config: EngineSceneConfig,
) -> np.ndarray:
    vector = np.asarray(normal, dtype=float)
    if frame == "engine":
        vector = transform_points(
            [vector],
            position=(0.0, 0.0, 0.0),
            quat_wxyz=tuple(float(value) for value in config.engine.pose.quat_wxyz),
            scale=1.0,
        )[0]
    norm = float(np.linalg.norm(vector))
    if norm <= 1.0e-12:
        return np.array([1.0, 0.0, 0.0], dtype=float)
    return vector / norm


def _engine_frame_points_to_world(
    points_m: object,
    config: EngineSceneConfig,
) -> np.ndarray:
    return transform_points(
        points_m,
        position=tuple(float(value) for value in effective_engine_frame_position(config)),
        quat_wxyz=tuple(float(value) for value in config.engine.pose.quat_wxyz),
        scale=1.0,
    )


def _add_primitive_collision_hints(
    worldbody: ElementTree.Element,
    raw_config: dict,
    *,
    config: EngineSceneConfig,
    show_disabled_hints: bool,
    primitive_alpha: float,
) -> None:
    geoms = load_primitive_collision_geoms(raw_config.get("primitive_collision_geoms"))
    for geom in geoms:
        if not geom.enabled and not show_disabled_hints:
            continue
        alpha = primitive_alpha if geom.enabled else min(primitive_alpha, 0.22)
        _add_primitive_collision_hint(worldbody, geom, config=config, alpha=alpha)


def _add_primitive_collision_hint(
    worldbody: ElementTree.Element,
    geom: PrimitiveCollisionGeomConfig,
    *,
    config: EngineSceneConfig,
    alpha: float,
) -> None:
    rgba = _primitive_rgba(geom, alpha)
    if geom.type == "capsule" and geom.fromto_m is not None:
        ElementTree.SubElement(
            worldbody,
            "geom",
            {
                "name": f"hint_{geom.name}",
                "type": "capsule",
                "fromto": _mujoco_vec(_primitive_fromto_world(geom, config)),
                "size": f"{float(geom.radius_m) * _primitive_dimension_scale(geom, config):.12g}",
                "rgba": rgba,
                "contype": "0",
                "conaffinity": "0",
                "group": "2",
            },
        )
        return

    position = _primitive_position_world(geom, config)
    quat = _primitive_quat_world(geom, config)
    body = ElementTree.SubElement(
        worldbody,
        "body",
        {
            "name": f"hint_body_{geom.name}",
            "pos": _mujoco_vec(position),
            "quat": _mujoco_vec(quat),
        },
    )
    attrs = {
        "name": f"hint_{geom.name}",
        "type": geom.type,
        "rgba": rgba,
        "contype": "0",
        "conaffinity": "0",
        "group": "2",
    }
    if geom.type == "capsule":
        dimension_scale = _primitive_dimension_scale(geom, config)
        radius = float(geom.radius_m) * dimension_scale
        half_length = float(geom.length_m) * 0.5 * dimension_scale
        attrs["fromto"] = f"0 0 {-half_length:.12g} 0 0 {half_length:.12g}"
        attrs["size"] = f"{radius:.12g}"
    elif geom.type == "cylinder":
        dimension_scale = _primitive_dimension_scale(geom, config)
        radius = float(geom.radius_m) * dimension_scale
        half_length = float(geom.length_m) * 0.5 * dimension_scale
        attrs["size"] = f"{radius:.12g} {half_length:.12g}"
    elif geom.type == "sphere":
        attrs["size"] = f"{float(geom.radius_m) * _primitive_dimension_scale(geom, config):.12g}"
    elif geom.type == "box":
        dimension_scale = _primitive_dimension_scale(geom, config)
        size = tuple(float(value) * dimension_scale for value in geom.size_m)
        attrs["size"] = _mujoco_vec(tuple(float(value) * 0.5 for value in size))
    else:
        return
    ElementTree.SubElement(body, "geom", attrs)


def _primitive_position_world(
    geom: PrimitiveCollisionGeomConfig,
    config: EngineSceneConfig,
) -> tuple[float, float, float]:
    if geom.position_m is None:
        return (0.0, 0.0, 0.0)
    if geom.frame == "world":
        return tuple(float(value) for value in geom.position_m)  # type: ignore[return-value]
    return _engine_local_to_world(geom.position_m, config)


def _primitive_fromto_world(
    geom: PrimitiveCollisionGeomConfig,
    config: EngineSceneConfig,
) -> tuple[float, float, float, float, float, float]:
    values = geom.fromto_m
    if values is None:
        raise ValueError(f"Primitive collision hint {geom.name!r} has no fromto_m.")
    if geom.frame == "world":
        return tuple(float(value) for value in values)  # type: ignore[return-value]
    start = _engine_local_to_world(values[:3], config)
    end = _engine_local_to_world(values[3:], config)
    return (*start, *end)


def _engine_local_to_world(values: object, config: EngineSceneConfig) -> tuple[float, float, float]:
    transformed = transform_points(
        [values],
        position=tuple(float(value) for value in effective_engine_frame_position(config)),
        quat_wxyz=tuple(float(value) for value in config.engine.pose.quat_wxyz),
        scale=float(config.engine.scale),
    )
    return tuple(float(value) for value in transformed[0])  # type: ignore[return-value]


def _primitive_dimension_scale(geom: PrimitiveCollisionGeomConfig, config: EngineSceneConfig) -> float:
    if geom.frame == "world":
        return 1.0
    return float(config.engine.scale)


def _primitive_quat_world(
    geom: PrimitiveCollisionGeomConfig,
    config: EngineSceneConfig,
) -> tuple[float, float, float, float]:
    local_quat = (
        tuple(float(value) for value in geom.quat_wxyz)
        if geom.quat_wxyz is not None
        else (1.0, 0.0, 0.0, 0.0)
    )
    if geom.frame == "world":
        return local_quat  # type: ignore[return-value]
    return _quat_multiply(tuple(float(value) for value in config.engine.pose.quat_wxyz), local_quat)


def _quat_multiply(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    lw, lx, ly, lz = left
    rw, rx, ry, rz = right
    quat = (
        lw * rw - lx * rx - ly * ry - lz * rz,
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
    )
    norm = sum(value * value for value in quat) ** 0.5
    if norm <= 1.0e-12:
        return (1.0, 0.0, 0.0, 0.0)
    return tuple(value / norm for value in quat)  # type: ignore[return-value]


def _primitive_rgba(geom: PrimitiveCollisionGeomConfig, alpha: float) -> str:
    rgba = geom.rgba or (1.0, 0.25, 0.1, alpha)
    return _mujoco_vec((rgba[0], rgba[1], rgba[2], alpha))


def _region_rgba(region: EngineRegionConfig, config: EngineSceneConfig) -> str:
    rgba = region.preview_rgba
    if rgba is None:
        rgba = config.preview_visualization.region_default_rgba_by_type.get(
            region.type,
            (0.2, 1.0, 0.4, 0.35),
        )
    return _mujoco_vec(rgba)


def _mujoco_vec(values: object) -> str:
    return " ".join(f"{float(value):.12g}" for value in values)


def _rgba_with_optional_alpha(
    rgba: tuple[float, float, float, float],
    alpha_override: float | None,
) -> str:
    if alpha_override is None:
        return _mujoco_vec(rgba)
    return _mujoco_vec((rgba[0], rgba[1], rgba[2], float(alpha_override)))


def _marker_scale_from_bbox(visual_report) -> float:
    if visual_report is None or visual_report.bbox_size_world is None:
        return 0.15
    return max(max(visual_report.bbox_size_world) * 0.35, 0.15)


def _suggest_camera(config_path: Path) -> dict[str, object]:
    config = load_engine_scene_config(config_path)
    diagnostics = collect_engine_scene_diagnostics(config_path)
    visual_report = next(
        (report for report in diagnostics.asset_reports if report.asset_name == "visual_mesh"),
        None,
    )
    engine_position = tuple(float(value) for value in config.engine.pose.position_m)
    if visual_report is None or visual_report.bbox_size_world is None:
        return {
            "lookat": engine_position,
            "distance": 1.0,
            "azimuth": 135.0,
            "elevation": -25.0,
        }
    max_size = max(visual_report.bbox_size_world)
    return {
        "lookat": engine_position,
        "distance": max(max_size * 2.2, 0.5),
        "azimuth": 135.0,
        "elevation": -25.0,
    }


if __name__ == "__main__":
    raise SystemExit(main())
