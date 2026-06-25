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
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from continuum_sim.scenes.engine_scene import (  # noqa: E402
    EngineRegionConfig,
    load_engine_scene_config,
    resolve_engine_asset_paths,
)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if not args.headless_check and not args.viewer:
        print("Pass --headless-check or --viewer.")
        return 1

    config = load_engine_scene_config(args.config)
    asset_paths = resolve_engine_asset_paths(config, config.path.parent)
    if not asset_paths.visual_mesh.exists():
        print(f"ERROR: visual mesh does not exist: {asset_paths.visual_mesh}", file=sys.stderr)
        return 1
    if asset_paths.collision_mesh is not None and not asset_paths.collision_mesh.exists():
        print(f"ERROR: collision mesh does not exist: {asset_paths.collision_mesh}", file=sys.stderr)
        return 1

    try:
        import mujoco
    except ImportError:
        print("ERROR: Python package 'mujoco' is not installed in this environment.", file=sys.stderr)
        return 1

    with tempfile.TemporaryDirectory(prefix="engine_preview_mujoco_") as temp_dir:
        mesh_overrides = _prepare_preview_meshes(config, Path(temp_dir))
        xml_text = build_engine_preview_mjcf(
            config_path=args.config,
            mesh_overrides=mesh_overrides,
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

        if args.viewer:
            try:
                import mujoco.viewer
            except ImportError:
                print("ERROR: mujoco.viewer is not available in this environment.", file=sys.stderr)
                return 1
            print("Opening MuJoCo viewer. Close the viewer window to exit.")
            data = mujoco.MjData(model)
            with mujoco.viewer.launch_passive(model, data) as viewer:
                while viewer.is_running():
                    viewer.sync()

    return 0


def build_engine_preview_mjcf(
    config_path: str | Path = Path("configs/scenes/engine_cleaning.yaml"),
    *,
    mesh_overrides: dict[str, Path] | None = None,
) -> str:
    """Build a minimal MJCF document for the configured engine mesh."""

    config = load_engine_scene_config(config_path)
    asset_paths = resolve_engine_asset_paths(config, config.path.parent)
    overrides = mesh_overrides or {}
    visual_mesh = overrides.get("visual_mesh", asset_paths.visual_mesh)
    collision_mesh = overrides.get("collision_mesh", asset_paths.collision_mesh)
    root = ElementTree.Element("mujoco", {"model": "engine_preview"})
    compiler = ElementTree.SubElement(root, "compiler")
    compiler.set("angle", "radian")
    compiler.set("autolimits", "true")
    ElementTree.SubElement(root, "option", {"timestep": "0.01"})

    asset = ElementTree.SubElement(root, "asset")
    scale = _mujoco_vec((config.engine.scale, config.engine.scale, config.engine.scale))
    ElementTree.SubElement(
        asset,
        "mesh",
        {
            "name": "engine_visual_mesh",
            "file": str(visual_mesh),
            "scale": scale,
        },
    )
    if collision_mesh is not None:
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
    _add_world_axes(worldbody)
    body = ElementTree.SubElement(
        worldbody,
        "body",
        {
            "name": "engine",
            "pos": _mujoco_vec(config.engine.pose.position_m),
            "quat": _mujoco_vec(config.engine.pose.quat_wxyz),
        },
    )
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
            "rgba": "0.72 0.76 0.80 1.0",
        },
    )
    if collision_mesh is not None:
        ElementTree.SubElement(
            body,
            "geom",
            {
                "name": "engine_collision",
                "type": "mesh",
                "mesh": "engine_collision_mesh",
                "group": "0",
                "rgba": "0.9 0.2 0.15 0.25",
            },
        )

    for region in config.regions.values():
        _add_region_site(worldbody, region)

    ElementTree.indent(root)
    return ElementTree.tostring(root, encoding="unicode")


def _prepare_preview_meshes(config, temp_dir: Path) -> dict[str, Path]:
    asset_paths = resolve_engine_asset_paths(config, config.path.parent)
    overrides: dict[str, Path] = {}
    visual_mesh = _mujoco_ready_stl(asset_paths.visual_mesh, temp_dir, "engine_visual_preview")
    if visual_mesh != asset_paths.visual_mesh:
        overrides["visual_mesh"] = visual_mesh
    if asset_paths.collision_mesh is not None:
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
    return parser.parse_args(argv)


def _add_world_axes(worldbody: ElementTree.Element) -> None:
    ElementTree.SubElement(
        worldbody,
        "site",
        {"name": "world_x_axis", "type": "capsule", "fromto": "0 0 0 0.15 0 0", "size": "0.003", "rgba": "1 0 0 1"},
    )
    ElementTree.SubElement(
        worldbody,
        "site",
        {"name": "world_y_axis", "type": "capsule", "fromto": "0 0 0 0 0.15 0", "size": "0.003", "rgba": "0 0.8 0 1"},
    )
    ElementTree.SubElement(
        worldbody,
        "site",
        {"name": "world_z_axis", "type": "capsule", "fromto": "0 0 0 0 0 0.15", "size": "0.003", "rgba": "0.1 0.2 1 1"},
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


if __name__ == "__main__":
    raise SystemExit(main())
