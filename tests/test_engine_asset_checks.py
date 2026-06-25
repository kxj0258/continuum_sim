from __future__ import annotations

import struct
from pathlib import Path

import pytest
import yaml

from scripts.check_engine_assets import collect_engine_asset_reports, parse_mesh_geometry


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCENE_CONFIG = PROJECT_ROOT / "configs" / "scenes" / "engine_cleaning.yaml"


def test_parse_ascii_stl_bounds(tmp_path: Path) -> None:
    mesh_path = tmp_path / "triangle_ascii.stl"
    mesh_path.write_text(
        """solid triangle
  facet normal 0 0 1
    outer loop
      vertex 1 2 3
      vertex 4 2 3
      vertex 1 6 3
    endloop
  endfacet
endsolid triangle
""",
        encoding="utf-8",
    )

    geometry = parse_mesh_geometry(mesh_path)

    assert geometry.vertex_count == 3
    assert geometry.face_count == 1
    assert geometry.bbox_min == pytest.approx((1.0, 2.0, 3.0))
    assert geometry.bbox_max == pytest.approx((4.0, 6.0, 3.0))
    assert geometry.size == pytest.approx((3.0, 4.0, 0.0))
    assert geometry.center == pytest.approx((2.5, 4.0, 3.0))


def test_parse_binary_stl_bounds(tmp_path: Path) -> None:
    mesh_path = tmp_path / "triangle_binary.stl"
    vertices = (
        (0.0, 0.0, 0.0),
        (2.0, 0.0, 0.0),
        (0.0, 3.0, 4.0),
    )
    record = struct.pack("<3f", 0.0, 0.0, 1.0)
    for vertex in vertices:
        record += struct.pack("<3f", *vertex)
    record += struct.pack("<H", 0)
    mesh_path.write_bytes(b"binary test".ljust(80, b" ") + struct.pack("<I", 1) + record)

    geometry = parse_mesh_geometry(mesh_path)

    assert geometry.vertex_count == 3
    assert geometry.face_count == 1
    assert geometry.bbox_min == pytest.approx((0.0, 0.0, 0.0))
    assert geometry.bbox_max == pytest.approx((2.0, 3.0, 4.0))
    assert geometry.size == pytest.approx((2.0, 3.0, 4.0))
    assert geometry.center == pytest.approx((1.0, 1.5, 2.0))


def test_parse_obj_bounds_and_faces(tmp_path: Path) -> None:
    mesh_path = tmp_path / "mesh.obj"
    mesh_path.write_text(
        """v -1.0 0.0 2.0
v 3.0 4.0 6.0
v 0.0 1.0 2.0
f 1 2 3
""",
        encoding="utf-8",
    )

    geometry = parse_mesh_geometry(mesh_path)

    assert geometry.vertex_count == 3
    assert geometry.face_count == 1
    assert geometry.bbox_min == pytest.approx((-1.0, 0.0, 2.0))
    assert geometry.bbox_max == pytest.approx((3.0, 4.0, 6.0))
    assert geometry.size == pytest.approx((4.0, 4.0, 4.0))
    assert geometry.center == pytest.approx((1.0, 2.0, 4.0))


def test_collect_asset_reports_allows_missing_assets_when_not_strict(tmp_path: Path) -> None:
    config_path = _write_scene_config(tmp_path, "missing_visual.stl", "missing_collision.stl")

    reports = collect_engine_asset_reports(config_path, strict_assets=False)

    assert [report.asset_name for report in reports] == ["visual_mesh", "collision_mesh"]
    assert [report.exists for report in reports] == [False, False]
    assert any("does not exist" in warning for report in reports for warning in report.warnings)


def test_collect_asset_reports_rejects_missing_assets_when_strict(tmp_path: Path) -> None:
    config_path = _write_scene_config(tmp_path, "missing_visual.stl", "missing_collision.stl")

    with pytest.raises(FileNotFoundError, match="visual_mesh.*does not exist"):
        collect_engine_asset_reports(config_path, strict_assets=True)


def test_repository_engine_config_loads_with_non_strict_assets() -> None:
    reports = collect_engine_asset_reports(SCENE_CONFIG, strict_assets=False)

    assert any(report.asset_name == "visual_mesh" for report in reports)


def _write_scene_config(tmp_path: Path, visual_mesh: str, collision_mesh: str) -> Path:
    config_path = tmp_path / "engine_scene.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "name": "test_engine",
                "scene_type": "engine_cleaning",
                "engine": {
                    "assets": {
                        "visual_mesh": visual_mesh,
                        "collision_mesh": collision_mesh,
                    },
                    "scale": 1.0,
                    "pose": {
                        "position_m": [0.0, 0.0, 0.0],
                        "quat_wxyz": [1.0, 0.0, 0.0, 0.0],
                    },
                },
                "regions": {
                    "entry_port": {
                        "type": "circular_port",
                        "center_m": [0.0, 0.0, 0.0],
                        "normal": [1.0, 0.0, 0.0],
                        "radius_m": 0.1,
                    }
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return config_path
