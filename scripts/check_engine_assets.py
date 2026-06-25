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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from continuum_sim.scenes.engine_scene import (  # noqa: E402
    load_engine_scene_config,
    resolve_engine_asset_paths,
    validate_engine_scene_config,
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

    config = load_engine_scene_config(config_path)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        validate_engine_scene_config(config, strict_assets=False)

    root_dir = Path(root).resolve() if root is not None else config.path.parent
    resolved = resolve_engine_asset_paths(config, root_dir)
    assets = [
        ("visual_mesh", "visual mesh", resolved.visual_mesh),
        ("collision_mesh", "collision mesh", resolved.collision_mesh),
        ("collision_geoms", "collision geometry metadata", resolved.collision_geoms),
    ]

    reports: list[AssetReport] = []
    for asset_name, role, asset_path in assets:
        if asset_path is None:
            continue
        report = _inspect_asset(asset_name, role, asset_path)
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
    return reports


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        reports = collect_engine_asset_reports(
            args.config,
            strict_assets=args.strict_assets,
            root=args.root,
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps([_report_to_dict(report) for report in reports], indent=2))
        return 0

    _print_reports(reports, display_root=Path(args.root).resolve() if args.root else PROJECT_ROOT)
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


def _inspect_asset(asset_name: str, role: str, path: Path) -> AssetReport:
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
        )

    geometry: MeshGeometry | None = None
    if supported and extension in SUPPORTED_MESH_EXTENSIONS:
        geometry = parse_mesh_geometry(path)
        warnings_list.extend(geometry.warnings)
        warnings_list.extend(_scale_warnings(geometry))
    elif not supported:
        warnings_list.append(f"Unsupported extension {extension!r} for {role}.")

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
    )


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


def _print_reports(reports: list[AssetReport], *, display_root: Path) -> None:
    for report in reports:
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
            print(f"  bbox_min: {_format_tuple(geometry.bbox_min)}")
            print(f"  bbox_max: {_format_tuple(geometry.bbox_max)}")
            print(f"  size: {_format_tuple(geometry.size)}")
            print(f"  center: {_format_tuple(geometry.center)}")
        for warning in report.warnings:
            print(f"  warning: {warning}")


def _format_tuple(values: tuple[float, float, float] | None) -> str:
    if values is None:
        return "<unavailable>"
    return "[" + ", ".join(f"{value:.6g}" for value in values) + "]"


def _relative_to(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return str(path)


def _report_to_dict(report: AssetReport) -> dict[str, object]:
    values = asdict(report)
    values["path"] = str(report.path)
    return values


if __name__ == "__main__":
    raise SystemExit(main())
