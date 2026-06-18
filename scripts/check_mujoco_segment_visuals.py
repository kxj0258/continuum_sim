"""Check segmented MuJoCo visual mesh exports."""

from __future__ import annotations

import argparse
import struct
import sys
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from continuum_sim.config import load_mujoco_config


PLACEHOLDER_README = "README.md"


@dataclass(frozen=True)
class StlBounds:
    """Axis-aligned STL bounds in the mesh file's coordinate units."""

    triangle_count: int
    minimum: tuple[float, float, float]
    maximum: tuple[float, float, float]

    @property
    def center(self) -> tuple[float, float, float]:
        return tuple(
            0.5 * (self.minimum[index] + self.maximum[index])
            for index in range(3)
        )

    @property
    def extent(self) -> tuple[float, float, float]:
        return tuple(
            self.maximum[index] - self.minimum[index]
            for index in range(3)
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/mujoco.yaml"),
        help="Path to the MuJoCo backend YAML config.",
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Return success even when expected STL files are missing.",
    )
    parser.add_argument(
        "--create-placeholders",
        action="store_true",
        help="Create the visual directory and a README placeholder, but no STL files.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_mujoco_config(
        args.config,
        require_xml=False,
        require_visual_meshes=False,
    )
    visuals = config.visuals

    print(f"visuals_enabled: {visuals.enabled}")
    print(f"frame_mode: {visuals.frame_mode}")
    print(f"cad_origin_mm: {_format_vec(visuals.cad_origin_mm)}")
    print(f"mesh_unit: {visuals.mesh_unit}")
    print(f"mesh_scale: {visuals.mesh_scale:g}")
    print(f"collision_mode: {visuals.collision_mode}")
    print(f"visual_geom_group: {visuals.visual_geom_group}")
    print(f"collision_geom_group: {visuals.collision_geom_group}")
    print(f"directory: {visuals.directory}")
    print("expected_meshes:")
    for mesh_name in visuals.expected_meshes:
        print(f"  - {mesh_name}")

    if args.create_placeholders:
        _create_placeholder_directory(visuals.directory)

    parsed_bounds: dict[str, StlBounds] = {}
    parse_errors: dict[str, str] = {}
    missing = [
        mesh_name
        for mesh_name in visuals.expected_meshes
        if not (visuals.directory / mesh_name).is_file()
    ]
    for mesh_name in visuals.expected_meshes:
        mesh_path = visuals.directory / mesh_name
        if mesh_name in missing:
            continue
        try:
            parsed_bounds[mesh_name] = read_stl_bounds(mesh_path)
        except ValueError as exc:
            parse_errors[mesh_name] = str(exc)

    if parsed_bounds:
        print("")
        print("mesh_bounds:")
        for mesh_name in visuals.expected_meshes:
            bounds = parsed_bounds.get(mesh_name)
            if bounds is None:
                continue
            minimum_m = tuple(value * visuals.mesh_scale for value in bounds.minimum)
            maximum_m = tuple(value * visuals.mesh_scale for value in bounds.maximum)
            print(f"  - name: {mesh_name}")
            print(f"    triangles: {bounds.triangle_count}")
            print(f"    min_{visuals.mesh_unit}: {_format_vec(bounds.minimum)}")
            print(f"    max_{visuals.mesh_unit}: {_format_vec(bounds.maximum)}")
            print(f"    min_model_m: {_format_vec(minimum_m)}")
            print(f"    max_model_m: {_format_vec(maximum_m)}")

    warnings = _frame_warnings(visuals.frame_mode, visuals.mesh_scale, parsed_bounds)
    if warnings:
        print("")
        print("frame_warnings:")
        for warning in warnings:
            print(f"  - {warning}")

    if parse_errors:
        print("")
        print("invalid_meshes:")
        for mesh_name, message in parse_errors.items():
            print(f"  - {mesh_name}: {message}")
        return 1

    if missing:
        print("")
        print("missing_meshes:")
        for mesh_name in missing:
            print(f"  - {mesh_name}")
        print(
            "Segmented visual meshes are incomplete. Do not enable true "
            "per-link visual following until these STL files are exported "
            "from CAD and placed in the configured directory."
        )
        return 0 if args.allow_missing else 1

    if warnings:
        return 1

    print("")
    print("All segmented visual mesh files are present.")
    return 0


def read_stl_bounds(path: Path) -> StlBounds:
    data = path.read_bytes()
    if len(data) < 84:
        raise ValueError("STL file is too small to contain binary mesh data.")

    triangle_count = struct.unpack_from("<I", data, 80)[0]
    expected_binary_size = 84 + triangle_count * 50
    if expected_binary_size == len(data):
        return _read_binary_stl_bounds(data, triangle_count)
    return _read_ascii_stl_bounds(data)


def _read_binary_stl_bounds(data: bytes, triangle_count: int) -> StlBounds:
    minimum = [float("inf"), float("inf"), float("inf")]
    maximum = [float("-inf"), float("-inf"), float("-inf")]
    offset = 84
    for _ in range(triangle_count):
        values = struct.unpack_from("<12f", data, offset)
        for vertex_index in range(3):
            vertex = values[3 + vertex_index * 3 : 3 + (vertex_index + 1) * 3]
            for axis, value in enumerate(vertex):
                minimum[axis] = min(minimum[axis], value)
                maximum[axis] = max(maximum[axis], value)
        offset += 50
    if triangle_count <= 0:
        raise ValueError("STL file contains no triangles.")
    return StlBounds(
        triangle_count=triangle_count,
        minimum=tuple(minimum),  # type: ignore[arg-type]
        maximum=tuple(maximum),  # type: ignore[arg-type]
    )


def _read_ascii_stl_bounds(data: bytes) -> StlBounds:
    minimum = [float("inf"), float("inf"), float("inf")]
    maximum = [float("-inf"), float("-inf"), float("-inf")]
    vertex_count = 0
    text = data.decode("utf-8", errors="ignore")
    for line in text.splitlines():
        parts = line.strip().split()
        if len(parts) != 4 or parts[0].lower() != "vertex":
            continue
        vertex = tuple(float(part) for part in parts[1:])
        vertex_count += 1
        for axis, value in enumerate(vertex):
            minimum[axis] = min(minimum[axis], value)
            maximum[axis] = max(maximum[axis], value)
    if vertex_count == 0:
        raise ValueError("STL file contains no parseable vertices.")
    return StlBounds(
        triangle_count=vertex_count // 3,
        minimum=tuple(minimum),  # type: ignore[arg-type]
        maximum=tuple(maximum),  # type: ignore[arg-type]
    )


def _frame_warnings(
    frame_mode: str,
    mesh_scale: float,
    bounds_by_name: dict[str, StlBounds],
) -> list[str]:
    if frame_mode != "body_local":
        return []

    link_min_z = [
        bounds.minimum[2] * mesh_scale
        for mesh_name, bounds in bounds_by_name.items()
        if mesh_name.startswith("segment_")
    ]
    if len(link_min_z) < 2:
        return []
    min_z_span = max(link_min_z) - min(link_min_z)
    if min_z_span <= 0.02:
        return []
    return [
        "visuals.frame_mode is body_local, but segment STL z-bounds span "
        f"{min_z_span:.6g} m after scaling. That looks like CAD-global STL "
        "coordinates; use frame_mode: cad_global with cad_origin_mm, or "
        "re-export each STL in its MuJoCo body-local frame."
    ]


def _format_vec(values: tuple[float, ...]) -> str:
    return "[" + ", ".join(f"{value:.6g}" for value in values) + "]"


def _create_placeholder_directory(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    readme_path = directory / PLACEHOLDER_README
    if not readme_path.exists():
        readme_path.write_text(
            "# MuJoCo Segmented Visual Meshes\n\n"
            "Place CAD-exported per-body STL files in this directory using the\n"
            "filenames listed by `scripts/check_mujoco_segment_visuals.py`.\n"
            "This placeholder intentionally does not include fake STL files.\n",
            encoding="utf-8",
        )
    print(f"placeholder_directory_ready: {directory}")
    print(f"placeholder_readme: {readme_path}")


if __name__ == "__main__":
    raise SystemExit(main())
