"""Preview the configured engine mesh and region markers in a minimal MuJoCo scene."""

from __future__ import annotations

import argparse
import struct
import sys
import tempfile
from pathlib import Path
from xml.etree import ElementTree

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
    load_engine_scene_config,
    resolve_engine_asset_paths,
)
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
            show_bbox=args.show_bbox,
            show_regions=args.show_regions,
            show_axes=args.show_axes,
            alpha_visual=args.alpha_visual,
            alpha_collision=args.alpha_collision,
            show_primitive_collision=args.show_primitive_collision,
            show_disabled_hints=args.show_disabled_hints,
            primitive_alpha=args.primitive_alpha,
            hide_mesh_collision=args.hide_mesh_collision,
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
    alpha_visual: float = 0.45,
    alpha_collision: float = 0.28,
    show_primitive_collision: bool = True,
    show_disabled_hints: bool = False,
    primitive_alpha: float = 0.55,
    hide_mesh_collision: bool = False,
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

    asset = ElementTree.SubElement(root, "asset")
    scale = _mujoco_vec((config.engine.scale, config.engine.scale, config.engine.scale))
    if include_visual:
        ElementTree.SubElement(
            asset,
            "mesh",
            {
                "name": "engine_visual_mesh",
                "file": str(visual_mesh),
                "scale": scale,
            },
        )
    if include_collision:
        ElementTree.SubElement(
            asset,
            "mesh",
            {
                "name": "engine_collision_mesh",
                "file": str(collision_mesh),
                "scale": scale,
            },
        )

    worldbody = ElementTree.SubElement(root, "worldbody")
    diagnostics = collect_engine_scene_diagnostics(config_path)
    visual_report = next(
        (report for report in diagnostics.asset_reports if report.asset_name == "visual_mesh"),
        None,
    )
    axis_length = _marker_scale_from_bbox(visual_report)
    if show_axes:
        _add_world_axes(worldbody, axis_length=axis_length)
    body = ElementTree.SubElement(
        worldbody,
        "body",
        {
            "name": "engine",
            "pos": _mujoco_vec(config.engine.pose.position_m),
            "quat": _mujoco_vec(config.engine.pose.quat_wxyz),
        },
    )
    if include_visual:
        ElementTree.SubElement(
            body,
            "geom",
            {
                "name": "engine_visual",
                "type": "mesh",
                "mesh": "engine_visual_mesh",
                "contype": "0",
                "conaffinity": "0",
                "group": "1",
                "rgba": f"0.72 0.76 0.80 {float(alpha_visual):.6g}",
            },
        )
    if include_collision:
        ElementTree.SubElement(
            body,
            "geom",
            {
                "name": "engine_collision",
                "type": "mesh",
                "mesh": "engine_collision_mesh",
                "group": "0",
                "rgba": f"0.9 0.2 0.15 {float(alpha_collision):.6g}",
            },
        )

    if show_bbox and visual_report is not None:
        _add_bbox_marker(worldbody, visual_report)

    if show_regions:
        for region in config.regions.values():
            _add_region_site(worldbody, region)

    if show_primitive_collision:
        _add_primitive_collision_hints(
            worldbody,
            load_yaml(Path(config_path).resolve()),
            config=config,
            show_disabled_hints=show_disabled_hints,
            primitive_alpha=primitive_alpha,
        )

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
        "--show-bbox",
        action="store_true",
        default=True,
        help="Show visual mesh world bbox markers.",
    )
    parser.add_argument(
        "--show-regions",
        action="store_true",
        default=True,
        help="Show configured engine region markers.",
    )
    parser.add_argument(
        "--show-axes",
        action="store_true",
        default=True,
        help="Show world X/Y/Z axes markers.",
    )
    parser.add_argument("--alpha-visual", type=float, default=0.45, help="Visual mesh alpha.")
    parser.add_argument("--alpha-collision", type=float, default=0.28, help="Collision mesh alpha.")
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
    return parser.parse_args(argv)


def _add_world_axes(worldbody: ElementTree.Element, *, axis_length: float) -> None:
    radius = max(axis_length * 0.01, 0.003)
    ElementTree.SubElement(
        worldbody,
        "site",
        {
            "name": "world_x_axis",
            "type": "capsule",
            "fromto": f"0 0 0 {axis_length:.12g} 0 0",
            "size": f"{radius:.12g}",
            "rgba": "1 0 0 1",
        },
    )
    ElementTree.SubElement(
        worldbody,
        "site",
        {
            "name": "world_y_axis",
            "type": "capsule",
            "fromto": f"0 0 0 0 {axis_length:.12g} 0",
            "size": f"{radius:.12g}",
            "rgba": "0 0.8 0 1",
        },
    )
    ElementTree.SubElement(
        worldbody,
        "site",
        {
            "name": "world_z_axis",
            "type": "capsule",
            "fromto": f"0 0 0 0 0 {axis_length:.12g}",
            "size": f"{radius:.12g}",
            "rgba": "0.1 0.2 1 1",
        },
    )


def _add_bbox_marker(worldbody: ElementTree.Element, visual_report) -> None:
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
    radius = max(max(bbox_size) * 0.002, 0.002)
    for index, (start_index, end_index) in enumerate(edges):
        ElementTree.SubElement(
            worldbody,
            "site",
            {
                "name": f"bbox_edge_{index}",
                "type": "capsule",
                "fromto": f"{_mujoco_vec(corners[start_index])} {_mujoco_vec(corners[end_index])}",
                "size": f"{radius:.12g}",
                "rgba": "1 1 0 0.85",
            },
        )


def _add_region_site(worldbody: ElementTree.Element, region: EngineRegionConfig) -> None:
    attrs = {
        "name": f"region_{region.name}",
        "rgba": _region_rgba(region.type),
    }
    if region.type == "circular_port" and region.center_m is not None:
        attrs.update(
            {
                "type": "sphere",
                "pos": _mujoco_vec(region.center_m),
                "size": str(region.radius_m or 0.01),
            }
        )
    elif region.type in ("roi_box", "box") and region.center_m is not None and region.size_m is not None:
        attrs.update(
            {
                "type": "box",
                "pos": _mujoco_vec(region.center_m),
                "size": _mujoco_vec(tuple(float(value) * 0.5 for value in region.size_m)),
            }
        )
    elif region.type == "surface_patch" and region.position_m is not None and region.extents_m is not None:
        attrs.update(
            {
                "type": "box",
                "pos": _mujoco_vec(region.position_m),
                "size": _mujoco_vec(tuple(max(float(value) * 0.5, 0.002) for value in region.extents_m)),
            }
        )
    elif region.center_m is not None:
        attrs.update({"type": "sphere", "pos": _mujoco_vec(region.center_m), "size": "0.01"})
    else:
        return
    ElementTree.SubElement(worldbody, "site", attrs)


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
        position=tuple(float(value) for value in config.engine.pose.position_m),
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


def _region_rgba(region_type: str) -> str:
    if region_type == "circular_port":
        return "0.1 0.6 1.0 0.6"
    if region_type == "surface_patch":
        return "1.0 0.7 0.1 0.55"
    if region_type == "box":
        return "1.0 0.1 0.1 0.35"
    return "0.2 1.0 0.4 0.35"


def _mujoco_vec(values: object) -> str:
    return " ".join(f"{float(value):.12g}" for value in values)


def _marker_scale_from_bbox(visual_report) -> float:
    if visual_report is None or visual_report.bbox_size_world is None:
        return 0.15
    return max(max(visual_report.bbox_size_world) * 0.35, 0.15)


def _suggest_camera(config_path: Path) -> dict[str, object]:
    diagnostics = collect_engine_scene_diagnostics(config_path)
    visual_report = next(
        (report for report in diagnostics.asset_reports if report.asset_name == "visual_mesh"),
        None,
    )
    if visual_report is None or visual_report.bbox_center_world is None or visual_report.bbox_size_world is None:
        return {
            "lookat": (0.0, 0.0, 0.0),
            "distance": 1.0,
            "azimuth": 135.0,
            "elevation": -25.0,
        }
    max_size = max(visual_report.bbox_size_world)
    return {
        "lookat": visual_report.bbox_center_world,
        "distance": max(max_size * 2.2, 0.5),
        "azimuth": 135.0,
        "elevation": -25.0,
    }


if __name__ == "__main__":
    raise SystemExit(main())
