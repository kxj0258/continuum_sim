"""Inspect engine scene mesh assets and report lightweight geometry bounds."""

from __future__ import annotations

import argparse
import json
import struct
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable
import warnings

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from continuum_sim.config import load_yaml  # noqa: E402
from continuum_sim.scenes.engine_scene import (  # noqa: E402
    effective_engine_frame_position,
    load_engine_scene_config,
    resolve_engine_asset_paths,
    validate_engine_scene_config,
)
from continuum_sim.scenes.primitive_collision import (  # noqa: E402
    PrimitiveCollisionGeomConfig,
    load_primitive_collision_geoms,
    primitive_geom_bbox,
)


SUPPORTED_MESH_EXTENSIONS = {".obj", ".stl", ".msh"}
SUPPORTED_METADATA_EXTENSIONS = {".yaml", ".yml"}


@dataclass(frozen=True)
class MeshGeometry:
    """Lightweight mesh geometry summary in the mesh file's native units."""

    vertex_count: int | None
    face_count: int | None
    bbox_min: tuple[float, float, float] | None
    bbox_max: tuple[float, float, float] | None
    size: tuple[float, float, float] | None
    center: tuple[float, float, float] | None
    parser: str
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class AssetReport:
    """Inspection report for one configured engine asset."""

    asset_name: str
    role: str
    path: Path
    exists: bool
    extension: str
    supported: bool
    size_bytes: int | None
    geometry: MeshGeometry | None
    warnings: tuple[str, ...]
    scale: float | None = None
    pose_position: tuple[float, float, float] | None = None
    local_offset_m: tuple[float, float, float] | None = None
    bbox_min_raw: tuple[float, float, float] | None = None
    bbox_max_raw: tuple[float, float, float] | None = None
    bbox_size_raw: tuple[float, float, float] | None = None
    bbox_center_raw: tuple[float, float, float] | None = None
    bbox_min_scaled: tuple[float, float, float] | None = None
    bbox_max_scaled: tuple[float, float, float] | None = None
    bbox_size_scaled: tuple[float, float, float] | None = None
    bbox_center_scaled: tuple[float, float, float] | None = None
    bbox_min_world: tuple[float, float, float] | None = None
    bbox_max_world: tuple[float, float, float] | None = None
    bbox_size_world: tuple[float, float, float] | None = None
    bbox_center_world: tuple[float, float, float] | None = None
    recommended_pose_position: tuple[float, float, float] | None = None
    recommended_grounded_pose_position: tuple[float, float, float] | None = None


@dataclass(frozen=True)
class RegionReport:
    """Relationship between a configured region marker and the visual bbox."""

    name: str
    type: str
    reference_point: tuple[float, float, float] | None
    distance_to_visual_bbox: float | None
    inside_visual_bbox: bool | None
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class PrimitiveCollisionHintReport:
    """Approximate diagnostic report for a primitive collision hint."""

    name: str
    type: str
    enabled: bool
    frame: str
    position_m: tuple[float, float, float] | None
    quat_wxyz: tuple[float, float, float, float] | None
    fromto_m: tuple[float, float, float, float, float, float] | None
    radius_m: float | None
    length_m: float | None
    size_m: tuple[float, float, float] | None
    rgba: tuple[float, float, float, float] | None
    bbox_min: tuple[float, float, float] | None
    bbox_max: tuple[float, float, float] | None
    intersects_visual_bbox: bool | None
    intersects_collision_bbox: bool | None
    warnings: tuple[str, ...]
    note: str = ""


@dataclass(frozen=True)
class EngineSceneDiagnostics:
    """Full diagnostic report for configured engine assets and regions."""

    config_path: Path
    scale: float
    pose_position: tuple[float, float, float]
    pose_quat_wxyz: tuple[float, float, float, float]
    asset_reports: list[AssetReport]
    region_reports: list[RegionReport]
    primitive_hint_reports: list[PrimitiveCollisionHintReport]


def parse_mesh_geometry(path: str | Path) -> MeshGeometry:
    """Parse lightweight geometry from OBJ/STL meshes without heavy dependencies."""

    mesh_path = Path(path)
    extension = mesh_path.suffix.lower()
    if extension == ".obj":
        return _parse_obj_geometry(mesh_path)
    if extension == ".stl":
        return _parse_stl_geometry(mesh_path)
    if extension == ".msh":
        return MeshGeometry(
            vertex_count=None,
            face_count=None,
            bbox_min=None,
            bbox_max=None,
            size=None,
            center=None,
            parser="msh-unparsed",
            warnings=("MSH existence is checked, but bbox parsing is not implemented.",),
        )
    raise ValueError(f"Unsupported mesh extension for {mesh_path}: {extension}")


def collect_engine_asset_reports(
    config_path: str | Path = Path("configs/scenes/engine_cleaning.yaml"),
    *,
    strict_assets: bool = False,
    root: str | Path | None = None,
) -> list[AssetReport]:
    """Load an engine scene config and inspect configured visual/collision assets."""

    return collect_engine_scene_diagnostics(
        config_path,
        strict_assets=strict_assets,
        root=root,
    ).asset_reports


def collect_engine_scene_diagnostics(
    config_path: str | Path = Path("configs/scenes/engine_cleaning.yaml"),
    *,
    strict_assets: bool = False,
    root: str | Path | None = None,
) -> EngineSceneDiagnostics:
    """Load an engine scene config and inspect assets plus region-to-bbox offsets."""

    config = load_engine_scene_config(config_path)
    raw = load_yaml(config.path)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        validate_engine_scene_config(config, strict_assets=False)

    root_dir = Path(root).resolve() if root is not None else config.path.parent
    resolved = resolve_engine_asset_paths(config, root_dir)
    scale = float(config.engine.scale)
    pose_position = _tuple3(config.engine.pose.position_m)
    frame_pose_position = _tuple3(effective_engine_frame_position(config))
    pose_quat_wxyz = _tuple4(config.engine.pose.quat_wxyz)
    collision_mesh_offset_m = (
        _tuple3(config.engine.assets.collision_mesh_offset_m)
        if config.engine.assets.collision_mesh_offset_m is not None
        else (0.0, 0.0, 0.0)
    )
    assets = [
        ("visual_mesh", "visual mesh", resolved.visual_mesh, (0.0, 0.0, 0.0)),
        ("collision_mesh", "collision mesh", resolved.collision_mesh, collision_mesh_offset_m),
        ("collision_geoms", "collision geometry metadata", resolved.collision_geoms, (0.0, 0.0, 0.0)),
    ]

    reports: list[AssetReport] = []
    for asset_name, role, asset_path, local_offset_m in assets:
        if asset_path is None:
            continue
        report = _inspect_asset(
            asset_name,
            role,
            asset_path,
            scale=scale,
            pose_position=pose_position,
            pose_quat_wxyz=pose_quat_wxyz,
            local_offset_m=local_offset_m,
        )
        if strict_assets and not report.exists:
            raise FileNotFoundError(
                f"Engine asset {asset_name!r} does not exist: {asset_path}."
            )
        if strict_assets and not report.supported:
            raise ValueError(
                f"Engine asset {asset_name!r} has unsupported extension "
                f"{report.extension!r}: {asset_path}."
            )
        reports.append(report)

    visual_report = next((report for report in reports if report.asset_name == "visual_mesh"), None)
    region_reports = _collect_region_reports(
        config.regions.values(),
        visual_report,
        pose_position=frame_pose_position,
        pose_quat_wxyz=pose_quat_wxyz,
    )
    collision_report = next((report for report in reports if report.asset_name == "collision_mesh"), None)
    primitive_hint_reports = _collect_primitive_hint_reports(
        raw.get("primitive_collision_geoms", []),
        visual_report=visual_report,
        collision_report=collision_report,
        scale=scale,
        pose_position=frame_pose_position,
        pose_quat_wxyz=pose_quat_wxyz,
    )
    return EngineSceneDiagnostics(
        config_path=config.path,
        scale=scale,
        pose_position=pose_position,
        pose_quat_wxyz=pose_quat_wxyz,
        asset_reports=reports,
        region_reports=region_reports,
        primitive_hint_reports=primitive_hint_reports,
    )


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        diagnostics = collect_engine_scene_diagnostics(
            args.config,
            strict_assets=args.strict_assets,
            root=args.root,
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(_diagnostics_to_dict(diagnostics), indent=2))
        return 0

    _print_diagnostics(diagnostics, display_root=Path(args.root).resolve() if args.root else PROJECT_ROOT)
    return 0


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/scenes/engine_cleaning.yaml"),
        help="Engine scene YAML config to inspect.",
    )
    parser.add_argument(
        "--strict-assets",
        action="store_true",
        help="Return a non-zero exit code if configured assets are missing or unsupported.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Optional root used when re-resolving relative configured asset paths.",
    )
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    return parser.parse_args(argv)


def _inspect_asset(
    asset_name: str,
    role: str,
    path: Path,
    *,
    scale: float,
    pose_position: tuple[float, float, float],
    pose_quat_wxyz: tuple[float, float, float, float],
    local_offset_m: tuple[float, float, float],
) -> AssetReport:
    extension = path.suffix.lower()
    supported = extension in SUPPORTED_MESH_EXTENSIONS
    if asset_name == "collision_geoms":
        supported = extension in SUPPORTED_METADATA_EXTENSIONS

    warnings_list: list[str] = []
    if not path.exists():
        warnings_list.append(f"{role} does not exist: {path}")
        return AssetReport(
            asset_name=asset_name,
            role=role,
            path=path,
            exists=False,
            extension=extension,
            supported=supported,
            size_bytes=None,
            geometry=None,
            warnings=tuple(warnings_list),
            scale=scale,
            pose_position=pose_position,
            local_offset_m=local_offset_m,
        )

    geometry: MeshGeometry | None = None
    if supported and extension in SUPPORTED_MESH_EXTENSIONS:
        geometry = parse_mesh_geometry(path)
        warnings_list.extend(geometry.warnings)
        warnings_list.extend(_scale_warnings(geometry))
    elif not supported:
        warnings_list.append(f"Unsupported extension {extension!r} for {role}.")

    bbox_values = _asset_bbox_values(
        geometry,
        scale=scale,
        pose_position=pose_position,
        pose_quat_wxyz=pose_quat_wxyz,
        local_offset_m=local_offset_m,
    )
    return AssetReport(
        asset_name=asset_name,
        role=role,
        path=path,
        exists=True,
        extension=extension,
        supported=supported,
        size_bytes=path.stat().st_size,
        geometry=geometry,
        warnings=tuple(warnings_list),
        scale=scale,
        pose_position=pose_position,
        local_offset_m=local_offset_m,
        **bbox_values,
    )


def _asset_bbox_values(
    geometry: MeshGeometry | None,
    *,
    scale: float,
    pose_position: tuple[float, float, float],
    pose_quat_wxyz: tuple[float, float, float, float],
    local_offset_m: tuple[float, float, float],
) -> dict[str, tuple[float, float, float] | None]:
    empty: dict[str, tuple[float, float, float] | None] = {
        "bbox_min_raw": None,
        "bbox_max_raw": None,
        "bbox_size_raw": None,
        "bbox_center_raw": None,
        "bbox_min_scaled": None,
        "bbox_max_scaled": None,
        "bbox_size_scaled": None,
        "bbox_center_scaled": None,
        "bbox_min_world": None,
        "bbox_max_world": None,
        "bbox_size_world": None,
        "bbox_center_world": None,
        "recommended_pose_position": None,
        "recommended_grounded_pose_position": None,
    }
    if geometry is None or geometry.bbox_min is None or geometry.bbox_max is None:
        return empty

    bbox_min_scaled = _scale_tuple(geometry.bbox_min, scale)
    bbox_max_scaled = _scale_tuple(geometry.bbox_max, scale)
    bbox_center_scaled = _scale_tuple(geometry.center, scale) if geometry.center else None
    bbox_size_scaled = _scale_tuple(geometry.size, abs(scale)) if geometry.size else None
    scaled_corners = bbox_corners(bbox_min_scaled, bbox_max_scaled)
    offset_corners = scaled_corners + np.asarray(local_offset_m, dtype=float)
    world_corners = transform_points(offset_corners, position=pose_position, quat_wxyz=pose_quat_wxyz)
    bbox_min_world, bbox_max_world, bbox_size_world, bbox_center_world = bbox_from_points(world_corners)
    zero_pose_corners = transform_points(
        offset_corners,
        position=(0.0, 0.0, 0.0),
        quat_wxyz=pose_quat_wxyz,
    )
    recommended_min, _recommended_max, _recommended_size, recommended_center = bbox_from_points(zero_pose_corners)
    return {
        "bbox_min_raw": geometry.bbox_min,
        "bbox_max_raw": geometry.bbox_max,
        "bbox_size_raw": geometry.size,
        "bbox_center_raw": geometry.center,
        "bbox_min_scaled": bbox_min_scaled,
        "bbox_max_scaled": bbox_max_scaled,
        "bbox_size_scaled": bbox_size_scaled,
        "bbox_center_scaled": bbox_center_scaled,
        "bbox_min_world": bbox_min_world,
        "bbox_max_world": bbox_max_world,
        "bbox_size_world": bbox_size_world,
        "bbox_center_world": bbox_center_world,
        "recommended_pose_position": _neg_tuple(recommended_center),
        "recommended_grounded_pose_position": (
            -recommended_center[0],
            -recommended_center[1],
            -recommended_min[2],
        ),
    }


def _collect_region_reports(
    regions,
    visual_report: AssetReport | None,
    *,
    pose_position: tuple[float, float, float],
    pose_quat_wxyz: tuple[float, float, float, float],
) -> list[RegionReport]:
    reports: list[RegionReport] = []
    visual_min = visual_report.bbox_min_world if visual_report else None
    visual_max = visual_report.bbox_max_world if visual_report else None
    visual_size = visual_report.bbox_size_world if visual_report else None
    near_threshold = 0.5

    for region in regions:
        reference_point = _region_reference_point(region)
        if reference_point is not None and region.frame == "engine":
            reference_point = _tuple3(
                transform_points(
                    [reference_point],
                    position=pose_position,
                    quat_wxyz=pose_quat_wxyz,
                    scale=1.0,
                )[0]
            )
        distance = (
            _distance_to_bbox(reference_point, visual_min, visual_max)
            if reference_point is not None and visual_min is not None and visual_max is not None
            else None
        )
        inside = distance == 0.0 if distance is not None else None
        warning_list: list[str] = []
        if distance is None:
            warning_list.append("Region has no comparable reference point or visual bbox is unavailable.")
        elif distance > near_threshold:
            warning_list.append(
                f"Region is {distance:.6g} m from visual mesh world bbox; pose or region alignment may need review."
            )
        reports.append(
            RegionReport(
                name=region.name,
                type=region.type,
                reference_point=reference_point,
                distance_to_visual_bbox=distance,
                inside_visual_bbox=inside,
                warnings=tuple(warning_list),
            )
        )
    return reports


def _parse_obj_geometry(path: Path) -> MeshGeometry:
    vertices: list[tuple[float, float, float]] = []
    face_count = 0
    with path.open("r", encoding="utf-8", errors="ignore") as stream:
        for line in stream:
            stripped = line.strip()
            if stripped.startswith("v "):
                parts = stripped.split()
                if len(parts) >= 4:
                    vertices.append((float(parts[1]), float(parts[2]), float(parts[3])))
            elif stripped.startswith("f "):
                face_count += 1
    return _geometry_from_vertices(vertices, face_count=face_count, parser="obj")


def _parse_stl_geometry(path: Path) -> MeshGeometry:
    data = path.read_bytes()
    if _looks_like_binary_stl(data):
        return _parse_binary_stl_geometry(data)
    return _parse_ascii_stl_geometry(data)


def _looks_like_binary_stl(data: bytes) -> bool:
    if len(data) < 84:
        return False
    triangle_count = struct.unpack("<I", data[80:84])[0]
    return len(data) == 84 + triangle_count * 50


def _parse_binary_stl_geometry(data: bytes) -> MeshGeometry:
    triangle_count = struct.unpack("<I", data[80:84])[0]
    vertices: list[tuple[float, float, float]] = []
    offset = 84
    for _ in range(triangle_count):
        offset += 12
        for _vertex_index in range(3):
            vertices.append(struct.unpack("<3f", data[offset : offset + 12]))
            offset += 12
        offset += 2
    return _geometry_from_vertices(vertices, face_count=triangle_count, parser="binary-stl")


def _parse_ascii_stl_geometry(data: bytes) -> MeshGeometry:
    vertices: list[tuple[float, float, float]] = []
    for raw_line in data.decode("utf-8", errors="ignore").splitlines():
        parts = raw_line.strip().split()
        if len(parts) == 4 and parts[0].lower() == "vertex":
            vertices.append((float(parts[1]), float(parts[2]), float(parts[3])))
    face_count = len(vertices) // 3 if vertices else 0
    return _geometry_from_vertices(vertices, face_count=face_count, parser="ascii-stl")


def _geometry_from_vertices(
    vertices: Iterable[tuple[float, float, float]],
    *,
    face_count: int | None,
    parser: str,
) -> MeshGeometry:
    vertex_list = list(vertices)
    if not vertex_list:
        return MeshGeometry(
            vertex_count=0,
            face_count=face_count,
            bbox_min=None,
            bbox_max=None,
            size=None,
            center=None,
            parser=parser,
            warnings=("No vertices were found; bbox is unavailable.",),
        )
    mins = tuple(min(vertex[axis] for vertex in vertex_list) for axis in range(3))
    maxs = tuple(max(vertex[axis] for vertex in vertex_list) for axis in range(3))
    size = tuple(maxs[axis] - mins[axis] for axis in range(3))
    center = tuple((mins[axis] + maxs[axis]) * 0.5 for axis in range(3))
    return MeshGeometry(
        vertex_count=len(vertex_list),
        face_count=face_count,
        bbox_min=mins,
        bbox_max=maxs,
        size=size,
        center=center,
        parser=parser,
    )


def _scale_warnings(geometry: MeshGeometry) -> list[str]:
    warnings_list: list[str] = []
    if geometry.face_count is not None and geometry.face_count > 200_000:
        warnings_list.append(
            "face_count is > 200000; MuJoCo may reject this STL unless it is "
            "simplified for preview or exported at lower resolution."
        )
    if geometry.size is None:
        return warnings_list
    max_dimension = max(geometry.size)
    if max_dimension > 20.0:
        warnings_list.append(
            "bbox max dimension is > 20 in mesh units; this may be millimeters. "
            "Consider engine.scale: 0.001 if exported from CAD in mm."
        )
    elif max_dimension < 0.01:
        warnings_list.append(
            "bbox max dimension is < 0.01 in mesh units; scale may be too small "
            "or the model units may be unusual."
        )
    else:
        warnings_list.append(
            "bbox dimensions look meter-scale; engine.scale: 1.0 is usually appropriate."
        )
    return warnings_list


def _print_diagnostics(diagnostics: EngineSceneDiagnostics, *, display_root: Path) -> None:
    print(f"config: {_relative_to(diagnostics.config_path, display_root)}")
    print(f"scale: {diagnostics.scale}")
    print(f"pose_position: {_format_tuple(diagnostics.pose_position)}")
    print()
    for report in diagnostics.asset_reports:
        print(f"{report.asset_name} ({report.role})")
        print(f"  path: {_relative_to(report.path, display_root)}")
        print(f"  exists: {report.exists}")
        print(f"  extension: {report.extension or '<none>'}")
        print(f"  supported: {report.supported}")
        if report.size_bytes is not None:
            print(f"  size_bytes: {report.size_bytes}")
        if report.geometry is not None:
            geometry = report.geometry
            print(f"  parser: {geometry.parser}")
            print(f"  vertex_count: {geometry.vertex_count}")
            print(f"  face_count: {geometry.face_count}")
            print(f"  bbox_min_raw: {_format_tuple(report.bbox_min_raw)}")
            print(f"  bbox_max_raw: {_format_tuple(report.bbox_max_raw)}")
            print(f"  bbox_size_raw: {_format_tuple(report.bbox_size_raw)}")
            print(f"  bbox_center_raw: {_format_tuple(report.bbox_center_raw)}")
            print(f"  local_offset_m: {_format_tuple(report.local_offset_m)}")
            print(f"  bbox_min_scaled: {_format_tuple(report.bbox_min_scaled)}")
            print(f"  bbox_max_scaled: {_format_tuple(report.bbox_max_scaled)}")
            print(f"  bbox_size_scaled: {_format_tuple(report.bbox_size_scaled)}")
            print(f"  bbox_center_scaled: {_format_tuple(report.bbox_center_scaled)}")
            print(f"  bbox_min_world: {_format_tuple(report.bbox_min_world)}")
            print(f"  bbox_max_world: {_format_tuple(report.bbox_max_world)}")
            print(f"  bbox_size_world: {_format_tuple(report.bbox_size_world)}")
            print(f"  bbox_center_world: {_format_tuple(report.bbox_center_world)}")
            print(f"  recommended_pose_position: {_format_tuple(report.recommended_pose_position)}")
            print(
                "  recommended_grounded_pose_position: "
                f"{_format_tuple(report.recommended_grounded_pose_position)}"
            )
        for warning in report.warnings:
            print(f"  warning: {warning}")
        print()

    print("regions")
    for report in diagnostics.region_reports:
        print(f"  {report.name} ({report.type})")
        print(f"    reference_point: {_format_tuple(report.reference_point)}")
        print(f"    distance_to_visual_bbox: {_format_float(report.distance_to_visual_bbox)}")
        print(f"    inside_visual_bbox: {report.inside_visual_bbox}")
        for warning in report.warnings:
            print(f"    warning: {warning}")
        if not report.warnings and report.distance_to_visual_bbox is not None:
            print("    OK: region is inside or near the visual mesh bbox.")

    if diagnostics.primitive_hint_reports:
        print()
        print("primitive_collision_hints")
        for report in diagnostics.primitive_hint_reports:
            print(f"  {report.name} ({report.type})")
            print(f"    enabled: {report.enabled}")
            print(f"    frame: {report.frame}")
            print(f"    position_m: {_format_tuple(report.position_m)}")
            print(f"    fromto_m: {_format_tuple6(report.fromto_m)}")
            print(f"    radius_m: {_format_float(report.radius_m)}")
            print(f"    length_m: {_format_float(report.length_m)}")
            print(f"    size_m: {_format_tuple(report.size_m)}")
            print(f"    bbox_min: {_format_tuple(report.bbox_min)}")
            print(f"    bbox_max: {_format_tuple(report.bbox_max)}")
            print(f"    intersects_visual_bbox: {report.intersects_visual_bbox}")
            print(f"    intersects_collision_bbox: {report.intersects_collision_bbox}")
            if report.note:
                print(f"    note: {report.note}")
            for warning in report.warnings:
                print(f"    warning: {warning}")


def _format_tuple(values: tuple[float, float, float] | None) -> str:
    if values is None:
        return "<unavailable>"
    return "[" + ", ".join(f"{value:.6g}" for value in values) + "]"


def _format_float(value: float | None) -> str:
    if value is None:
        return "<unavailable>"
    return f"{value:.6g}"


def _format_tuple6(values: tuple[float, float, float, float, float, float] | None) -> str:
    if values is None:
        return "<unavailable>"
    return "[" + ", ".join(f"{value:.6g}" for value in values) + "]"


def _relative_to(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return str(path)


def _diagnostics_to_dict(diagnostics: EngineSceneDiagnostics) -> dict[str, object]:
    return {
        "config_path": str(diagnostics.config_path),
        "scale": diagnostics.scale,
        "pose_position": diagnostics.pose_position,
        "pose_quat_wxyz": diagnostics.pose_quat_wxyz,
        "asset_reports": [_report_to_dict(report) for report in diagnostics.asset_reports],
        "region_reports": [asdict(report) for report in diagnostics.region_reports],
        "primitive_hint_reports": [asdict(report) for report in diagnostics.primitive_hint_reports],
    }


def _report_to_dict(report: AssetReport) -> dict[str, object]:
    values = asdict(report)
    values["path"] = str(report.path)
    return values


def _tuple3(values: object) -> tuple[float, float, float]:
    return tuple(float(value) for value in values)  # type: ignore[return-value]


def _tuple4(values: object) -> tuple[float, float, float, float]:
    return tuple(float(value) for value in values)  # type: ignore[return-value]


def quat_wxyz_to_matrix(quat: object) -> np.ndarray:
    """Return a 3x3 rotation matrix for a `[w, x, y, z]` quaternion."""

    q = np.asarray(quat, dtype=float)
    if q.shape != (4,):
        raise ValueError(f"Expected quaternion with shape (4,), got {q.shape}.")
    norm = float(np.linalg.norm(q))
    if norm <= 1.0e-12:
        raise ValueError("quat_wxyz must have non-zero length.")
    w, x, y, z = q / norm
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=float,
    )


def bbox_corners(
    bbox_min: tuple[float, float, float],
    bbox_max: tuple[float, float, float],
) -> np.ndarray:
    """Return the eight corners of an axis-aligned bbox."""

    x0, y0, z0 = bbox_min
    x1, y1, z1 = bbox_max
    return np.array(
        [
            [x0, y0, z0],
            [x1, y0, z0],
            [x1, y1, z0],
            [x0, y1, z0],
            [x0, y0, z1],
            [x1, y0, z1],
            [x1, y1, z1],
            [x0, y1, z1],
        ],
        dtype=float,
    )


def transform_points(
    points: object,
    *,
    position: tuple[float, float, float],
    quat_wxyz: tuple[float, float, float, float],
    scale: float = 1.0,
) -> np.ndarray:
    """Transform points by optional scale, `[w, x, y, z]` rotation, and translation."""

    point_array = np.asarray(points, dtype=float)
    if point_array.ndim != 2 or point_array.shape[1] != 3:
        raise ValueError(f"Expected points with shape (N, 3), got {point_array.shape}.")
    rotation = quat_wxyz_to_matrix(quat_wxyz)
    translation = np.asarray(position, dtype=float)
    return (rotation @ (point_array * float(scale)).T).T + translation


def bbox_from_points(
    points: object,
) -> tuple[
    tuple[float, float, float],
    tuple[float, float, float],
    tuple[float, float, float],
    tuple[float, float, float],
]:
    """Return min, max, size, and center for an `N x 3` point cloud."""

    point_array = np.asarray(points, dtype=float)
    if point_array.ndim != 2 or point_array.shape[1] != 3 or point_array.shape[0] == 0:
        raise ValueError(f"Expected non-empty points with shape (N, 3), got {point_array.shape}.")
    minimum = np.min(point_array, axis=0)
    maximum = np.max(point_array, axis=0)
    size = maximum - minimum
    center = (minimum + maximum) * 0.5
    return _tuple3(minimum), _tuple3(maximum), _tuple3(size), _tuple3(center)


def _scale_tuple(values: tuple[float, float, float] | None, scale: float) -> tuple[float, float, float] | None:
    if values is None:
        return None
    return tuple(float(value) * scale for value in values)  # type: ignore[return-value]


def _add_tuple(
    left: tuple[float, float, float] | None,
    right: tuple[float, float, float],
) -> tuple[float, float, float] | None:
    if left is None:
        return None
    return tuple(left[index] + right[index] for index in range(3))  # type: ignore[return-value]


def _neg_tuple(values: tuple[float, float, float]) -> tuple[float, float, float]:
    return tuple(-value for value in values)  # type: ignore[return-value]


def _region_reference_point(region) -> tuple[float, float, float] | None:
    if region.center_m is not None:
        return _tuple3(region.center_m)
    if region.position_m is not None:
        return _tuple3(region.position_m)
    return None


def _distance_to_bbox(
    point: tuple[float, float, float],
    bbox_min: tuple[float, float, float],
    bbox_max: tuple[float, float, float],
) -> float:
    squared = 0.0
    for axis in range(3):
        if point[axis] < bbox_min[axis]:
            squared += (bbox_min[axis] - point[axis]) ** 2
        elif point[axis] > bbox_max[axis]:
            squared += (point[axis] - bbox_max[axis]) ** 2
    return squared**0.5


def _collect_primitive_hint_reports(
    raw_hints: object,
    *,
    visual_report: AssetReport | None,
    collision_report: AssetReport | None,
    scale: float,
    pose_position: tuple[float, float, float],
    pose_quat_wxyz: tuple[float, float, float, float],
) -> list[PrimitiveCollisionHintReport]:
    geoms = load_primitive_collision_geoms(raw_hints)
    return [
        _primitive_hint_report(
            geom,
            visual_report=visual_report,
            collision_report=collision_report,
            scale=scale,
            pose_position=pose_position,
            pose_quat_wxyz=pose_quat_wxyz,
        )
        for geom in geoms
    ]


def _primitive_hint_report(
    geom: PrimitiveCollisionGeomConfig,
    *,
    visual_report: AssetReport | None,
    collision_report: AssetReport | None,
    scale: float,
    pose_position: tuple[float, float, float],
    pose_quat_wxyz: tuple[float, float, float, float],
) -> PrimitiveCollisionHintReport:
    bbox = _primitive_geom_bbox_world(
        geom,
        scale=scale,
        pose_position=pose_position,
        pose_quat_wxyz=pose_quat_wxyz,
    )
    intersects_visual = _bbox_intersects_report(bbox.minimum, bbox.maximum, visual_report)
    intersects_collision = _bbox_intersects_report(bbox.minimum, bbox.maximum, collision_report)
    warnings_list: list[str] = []
    if intersects_visual is False and intersects_collision is False:
        warnings_list.append("Primitive hint bbox does not intersect visual or collision mesh world bboxes.")
    return PrimitiveCollisionHintReport(
        name=geom.name,
        type=geom.type,
        enabled=geom.enabled,
        frame=geom.frame,
        position_m=_tuple_or_none(geom.position_m),
        quat_wxyz=_tuple4_or_none(geom.quat_wxyz),
        fromto_m=_tuple6_or_none(geom.fromto_m),
        radius_m=geom.radius_m,
        length_m=geom.length_m,
        size_m=_tuple_or_none(geom.size_m),
        rgba=geom.rgba,
        bbox_min=bbox.minimum,
        bbox_max=bbox.maximum,
        intersects_visual_bbox=intersects_visual,
        intersects_collision_bbox=intersects_collision,
        warnings=tuple(warnings_list),
        note=geom.note or "",
    )


def _primitive_geom_bbox_world(
    geom: PrimitiveCollisionGeomConfig,
    *,
    scale: float,
    pose_position: tuple[float, float, float],
    pose_quat_wxyz: tuple[float, float, float, float],
):
    bbox = primitive_geom_bbox(geom)
    if geom.frame == "world":
        return bbox
    world_corners = transform_points(
        bbox_corners(bbox.minimum, bbox.maximum),
        position=pose_position,
        quat_wxyz=pose_quat_wxyz,
        scale=scale,
    )
    bbox_min, bbox_max, _bbox_size, _bbox_center = bbox_from_points(world_corners)
    return type(bbox)(minimum=bbox_min, maximum=bbox_max)


def _bbox_intersects_report(
    bbox_min: tuple[float, float, float],
    bbox_max: tuple[float, float, float],
    report: AssetReport | None,
) -> bool | None:
    if report is None or report.bbox_min_world is None or report.bbox_max_world is None:
        return None
    return _bbox_intersects(bbox_min, bbox_max, report.bbox_min_world, report.bbox_max_world)


def _bbox_intersects(
    a_min: tuple[float, float, float],
    a_max: tuple[float, float, float],
    b_min: tuple[float, float, float],
    b_max: tuple[float, float, float],
) -> bool:
    return all(a_min[index] <= b_max[index] and b_min[index] <= a_max[index] for index in range(3))


def _tuple_or_none(values: object) -> tuple[float, float, float] | None:
    if values is None:
        return None
    return tuple(float(value) for value in values)  # type: ignore[return-value]


def _tuple4_or_none(values: object) -> tuple[float, float, float, float] | None:
    if values is None:
        return None
    return tuple(float(value) for value in values)  # type: ignore[return-value]


def _tuple6_or_none(values: object) -> tuple[float, float, float, float, float, float] | None:
    if values is None:
        return None
    return tuple(float(value) for value in values)  # type: ignore[return-value]


if __name__ == "__main__":
    raise SystemExit(main())
