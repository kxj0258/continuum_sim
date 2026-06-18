"""Generate a MuJoCo XML with segmented visual mesh geoms attached."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from xml.etree import ElementTree

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from continuum_sim.config import load_mujoco_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/mujoco.yaml"),
        help="Path to the MuJoCo backend YAML config.",
    )
    parser.add_argument(
        "--template",
        type=Path,
        default=None,
        help=(
            "Optional legacy segmented-visual template override. The generator "
            "now reads mesh/body settings from the YAML config."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Override the generated XML output path.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        output_path = build_xml_with_segment_visuals(
            config_path=args.config,
            template_path=args.template,
            output_path=args.output,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"Failed to generate segmented-visual MuJoCo XML: {exc}", file=sys.stderr)
        return 1

    print(f"generated_xml_path: {output_path}")
    return 0


def build_xml_with_segment_visuals(
    *,
    config_path: Path,
    template_path: Path | None = None,
    output_path: Path | None = None,
) -> Path:
    config = load_mujoco_config(
        config_path,
        require_xml=False,
        require_tendon_xml=False,
    )
    if not config.visuals.enabled:
        raise ValueError("visuals.enabled must be true before generating visual XML.")

    source_xml_path, default_output_path = _visual_xml_paths_for_mode(config)
    if not source_xml_path.is_file():
        raise FileNotFoundError(f"MuJoCo source XML does not exist: {source_xml_path}")
    target_path = (output_path or default_output_path).resolve()
    base_root = ElementTree.parse(source_xml_path).getroot()
    resolved_template_path = (template_path or config.visuals.template_path).resolve()
    if template_path is not None and not resolved_template_path.is_file():
        raise FileNotFoundError(
            f"Segmented visuals template does not exist: {resolved_template_path}"
        )
    if resolved_template_path.is_file():
        _validate_legacy_template(resolved_template_path, config.visuals.expected_meshes)

    _set_collision_geom_groups(base_root, config.visuals.collision_geom_group)
    _add_segment_visuals(base_root, config, target_path)

    target_path.parent.mkdir(parents=True, exist_ok=True)
    _indent(base_root)
    ElementTree.ElementTree(base_root).write(
        target_path,
        encoding="utf-8",
        xml_declaration=False,
        short_empty_elements=True,
    )
    return target_path


def _visual_xml_paths_for_mode(config) -> tuple[Path, Path]:
    if config.control_mode == "position_joint":
        return config.xml_path, config.generated_xml_path
    if config.control_mode == "tendon_position":
        return config.tendon_xml_path, config.tendon_generated_xml_path
    raise ValueError(f"Unsupported MuJoCo control_mode {config.control_mode!r}.")


def _validate_legacy_template(
    template_path: Path,
    expected_meshes: tuple[str, ...],
) -> None:
    template_root = ElementTree.parse(template_path).getroot()
    template_asset = template_root.find("asset")
    if template_asset is None:
        return
    template_meshes = {
        Path(str(mesh.get("file"))).name
        for mesh in template_asset.findall("mesh")
        if mesh.get("file") is not None
    }
    if template_meshes and template_meshes != set(expected_meshes):
        raise ValueError(
            "Segmented visuals template mesh files do not match "
            f"visuals.expected_meshes: {template_path}"
        )


def _set_collision_geom_groups(
    base_root: ElementTree.Element,
    collision_geom_group: int,
) -> None:
    for geom in base_root.findall(".//geom"):
        geom.set("group", str(collision_geom_group))


def _add_segment_visuals(
    base_root: ElementTree.Element,
    config,
    target_path: Path,
) -> None:
    visuals = config.visuals
    body_origins = _straight_pose_body_origins(base_root)
    base_bodies = {
        str(body.get("name")): body
        for body in base_root.findall(".//body")
        if body.get("name") is not None
    }

    missing_bodies: list[str] = []
    for mesh_filename in visuals.expected_meshes:
        body_name = _body_name_from_visual_mesh(mesh_filename)
        if body_name not in base_bodies:
            missing_bodies.append(body_name)
    if missing_bodies:
        raise ValueError(f"Base XML is missing visual mesh bodies: {sorted(missing_bodies)}")

    base_asset = _asset_section(base_root)
    for mesh_filename in visuals.expected_meshes:
        body_name = _body_name_from_visual_mesh(mesh_filename)
        mesh_path = visuals.directory / mesh_filename
        if not mesh_path.is_file():
            raise FileNotFoundError(f"Segmented visual mesh does not exist: {mesh_path}")

        mesh_name = f"{body_name}_visual_mesh"
        geom_name = f"{body_name}_visual_geom"
        mesh_element = _mesh_element(base_asset, mesh_name)
        mesh_element.set("name", mesh_name)
        mesh_element.set("file", _mesh_file_reference(mesh_path, target_path.parent))
        mesh_element.set("scale", _format_vec((visuals.mesh_scale,) * 3))

        visual_pos = _visual_geom_pos(
            frame_mode=visuals.frame_mode,
            cad_origin_mm=visuals.cad_origin_mm,
            mesh_scale=visuals.mesh_scale,
            body_origin=body_origins[body_name],
        )
        body = base_bodies[body_name]
        geom_element = _geom_element(body, geom_name)
        geom_element.set("name", geom_name)
        geom_element.set("type", "mesh")
        geom_element.set("mesh", mesh_name)
        geom_element.set("pos", _format_vec(visual_pos))
        geom_element.set("contype", "0")
        geom_element.set("conaffinity", "0")
        geom_element.set("density", "0")
        geom_element.set("group", str(visuals.visual_geom_group))


def _asset_section(base_root: ElementTree.Element) -> ElementTree.Element:
    base_asset = base_root.find("asset")
    if base_asset is None:
        base_asset = ElementTree.Element("asset")
        base_root.insert(0, base_asset)
    return base_asset


def _mesh_element(asset: ElementTree.Element, mesh_name: str) -> ElementTree.Element:
    existing = asset.find(f"./mesh[@name='{mesh_name}']")
    if existing is not None:
        return existing
    mesh = ElementTree.Element("mesh")
    asset.append(mesh)
    return mesh


def _geom_element(body: ElementTree.Element, geom_name: str) -> ElementTree.Element:
    existing = body.find(f"./geom[@name='{geom_name}']")
    if existing is not None:
        return existing
    geom = ElementTree.Element("geom")
    children = list(body)
    insert_index = next(
        (index for index, child in enumerate(children) if child.tag == "body"),
        len(children),
    )
    body.insert(insert_index, geom)
    return geom


def _body_name_from_visual_mesh(mesh_filename: str) -> str:
    stem = Path(mesh_filename).stem
    if not stem.endswith("_visual"):
        raise ValueError(
            "Segmented visual mesh filenames must end with '_visual.stl', "
            f"got {mesh_filename!r}."
        )
    return stem[: -len("_visual")]


def _straight_pose_body_origins(
    root: ElementTree.Element,
) -> dict[str, tuple[float, float, float]]:
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError("Base XML is missing a <worldbody> section.")

    origins: dict[str, tuple[float, float, float]] = {}

    def walk(body: ElementTree.Element, parent_origin: tuple[float, float, float]) -> None:
        _reject_rotated_body(body)
        body_name = body.get("name")
        body_pos = _parse_vec(body.get("pos", "0 0 0"), "body.pos")
        origin = tuple(parent_origin[index] + body_pos[index] for index in range(3))
        if body_name is not None:
            origins[str(body_name)] = origin
        for child in body.findall("body"):
            walk(child, origin)

    for body in worldbody.findall("body"):
        walk(body, (0.0, 0.0, 0.0))
    return origins


def _reject_rotated_body(body: ElementTree.Element) -> None:
    rotation_attrs = ("quat", "axisangle", "xyaxes", "zaxis", "euler")
    present = [name for name in rotation_attrs if body.get(name) is not None]
    if present:
        body_name = body.get("name", "<unnamed>")
        raise ValueError(
            "CAD-global visual offset generation currently supports straight-pose "
            f"body origins without body rotations; body {body_name!r} has {present}."
        )


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
        raise ValueError(f"Unsupported visuals.frame_mode: {frame_mode!r}")
    cad_origin_model = tuple(value * mesh_scale for value in cad_origin_mm)
    return tuple(
        -cad_origin_model[index] - body_origin[index]
        for index in range(3)
    )


def _parse_vec(raw_value: str, name: str) -> tuple[float, float, float]:
    parts = raw_value.split()
    if len(parts) != 3:
        raise ValueError(f"{name} must contain exactly 3 numbers, got {raw_value!r}.")
    return tuple(float(part) for part in parts)  # type: ignore[return-value]


def _mesh_file_reference(mesh_path: Path, xml_directory: Path) -> str:
    mesh_path = mesh_path.resolve()
    try:
        relpath = os.path.relpath(mesh_path, xml_directory.resolve())
    except ValueError:
        return mesh_path.as_posix()
    return relpath.replace(os.sep, "/")


def _format_vec(values: tuple[float, ...]) -> str:
    return " ".join(f"{value:.12g}" for value in values)


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
