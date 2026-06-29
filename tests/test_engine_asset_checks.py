from __future__ import annotations

import struct
from pathlib import Path

import pytest
import yaml

from scripts.check_engine_assets import (
    collect_engine_asset_reports,
    collect_engine_scene_diagnostics,
    main as check_engine_assets_main,
    parse_mesh_geometry,
)
from scripts.preview_engine_scene_mujoco import build_engine_preview_mjcf


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


def test_collect_scene_diagnostics_computes_scaled_world_bbox_and_recenter_pose(
    tmp_path: Path,
) -> None:
    visual_mesh = tmp_path / "visual.obj"
    visual_mesh.write_text(
        """v 0 0 0
v 2 4 6
v 1 1 1
f 1 2 3
""",
        encoding="utf-8",
    )
    config_path = _write_scene_config(
        tmp_path,
        visual_mesh.name,
        "",
        scale=0.5,
        position_m=[1.0, 2.0, 3.0],
    )

    diagnostics = collect_engine_scene_diagnostics(config_path, strict_assets=False)
    visual = diagnostics.asset_reports[0]

    assert visual.bbox_min_raw == pytest.approx((0.0, 0.0, 0.0))
    assert visual.bbox_max_raw == pytest.approx((2.0, 4.0, 6.0))
    assert visual.bbox_min_scaled == pytest.approx((0.0, 0.0, 0.0))
    assert visual.bbox_max_scaled == pytest.approx((1.0, 2.0, 3.0))
    assert visual.bbox_min_world == pytest.approx((1.0, 2.0, 3.0))
    assert visual.bbox_max_world == pytest.approx((2.0, 4.0, 6.0))
    assert visual.recommended_pose_position == pytest.approx((-0.5, -1.0, -1.5))
    assert visual.recommended_grounded_pose_position == pytest.approx((-0.5, -1.0, 0.0))


def test_collision_mesh_offset_aligns_collision_bbox_center_to_visual_bbox_center(
    tmp_path: Path,
) -> None:
    visual_mesh = tmp_path / "visual.obj"
    collision_mesh = tmp_path / "collision.obj"
    visual_mesh.write_text(
        """v 0 0 0
v 2 2 2
v 0 2 0
f 1 2 3
""",
        encoding="utf-8",
    )
    collision_mesh.write_text(
        """v 10 0 0
v 12 2 2
v 10 2 0
f 1 2 3
""",
        encoding="utf-8",
    )
    config_path = _write_scene_config(
        tmp_path,
        visual_mesh.name,
        collision_mesh.name,
        collision_mesh_offset_m=[-10.0, 0.0, 0.0],
    )

    diagnostics = collect_engine_scene_diagnostics(config_path, strict_assets=False)
    visual = next(report for report in diagnostics.asset_reports if report.asset_name == "visual_mesh")
    collision = next(report for report in diagnostics.asset_reports if report.asset_name == "collision_mesh")

    assert collision.local_offset_m == pytest.approx((-10.0, 0.0, 0.0))
    assert collision.bbox_center_world == pytest.approx(visual.bbox_center_world)
    assert collision.bbox_min_world == pytest.approx(visual.bbox_min_world)
    assert collision.bbox_max_world == pytest.approx(visual.bbox_max_world)


def test_region_diagnostics_warn_when_regions_are_far_from_visual_bbox(tmp_path: Path) -> None:
    visual_mesh = tmp_path / "visual.obj"
    visual_mesh.write_text(
        """v 0 0 0
v 1 1 1
v 0 1 0
f 1 2 3
""",
        encoding="utf-8",
    )
    config_path = _write_scene_config(
        tmp_path,
        visual_mesh.name,
        "",
        regions={
            "entry_port": {
                "type": "circular_port",
                "center_m": [10.0, 0.0, 0.0],
                "normal": [1.0, 0.0, 0.0],
                "radius_m": 0.1,
            }
        },
    )

    diagnostics = collect_engine_scene_diagnostics(config_path, strict_assets=False)

    assert diagnostics.region_reports[0].name == "entry_port"
    assert diagnostics.region_reports[0].distance_to_visual_bbox > 0.0
    assert diagnostics.region_reports[0].warnings


def test_json_output_contains_scaled_bbox_and_scale(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    visual_mesh = tmp_path / "visual.obj"
    visual_mesh.write_text(
        """v 0 0 0
v 2 2 2
v 0 2 0
f 1 2 3
""",
        encoding="utf-8",
    )
    config_path = _write_scene_config(tmp_path, visual_mesh.name, "", scale=0.25)

    assert check_engine_assets_main(["--config", str(config_path), "--json"]) == 0
    output = capsys.readouterr().out

    assert '"scale": 0.25' in output
    assert '"bbox_min_scaled"' in output
    assert '"bbox_max_world"' in output


def test_preview_mjcf_supports_visibility_modes_and_markers(tmp_path: Path) -> None:
    visual_mesh = tmp_path / "visual.obj"
    collision_mesh = tmp_path / "collision.obj"
    mesh_text = """v 0 0 0
v 1 1 1
v 0 1 0
f 1 2 3
"""
    visual_mesh.write_text(mesh_text, encoding="utf-8")
    collision_mesh.write_text(mesh_text, encoding="utf-8")
    config_path = _write_scene_config(tmp_path, visual_mesh.name, collision_mesh.name)

    both = build_engine_preview_mjcf(
        config_path,
        show_bbox=True,
        show_regions=True,
        show_axes=True,
        alpha_visual=0.4,
        alpha_collision=0.2,
    )
    visual_only = build_engine_preview_mjcf(config_path, visual_only=True)
    collision_only = build_engine_preview_mjcf(config_path, collision_only=True)

    assert 'name="engine_visual"' in both
    assert 'name="engine_collision"' in both
    assert 'name="bbox_edge_0"' in both
    assert 'name="region_entry_port"' in both
    assert 'name="world_x_axis"' in both
    assert "0.72 0.76 0.80 0.4" in both
    assert "0.9 0.2 0.15 0.2" in both
    assert 'name="engine_collision"' not in visual_only
    assert 'name="engine_visual"' not in collision_only


def test_preview_mjcf_applies_collision_mesh_offset_to_collision_geom(tmp_path: Path) -> None:
    visual_mesh = tmp_path / "visual.obj"
    collision_mesh = tmp_path / "collision.obj"
    mesh_text = """v 0 0 0
v 1 1 1
v 0 1 0
f 1 2 3
"""
    visual_mesh.write_text(mesh_text, encoding="utf-8")
    collision_mesh.write_text(mesh_text, encoding="utf-8")
    config_path = _write_scene_config(
        tmp_path,
        visual_mesh.name,
        collision_mesh.name,
        collision_mesh_offset_m=[0.25, -0.5, 0.0],
    )

    xml_text = build_engine_preview_mjcf(config_path)

    assert 'name="engine_collision"' in xml_text
    assert 'pos="0.25 -0.5 0"' in xml_text


def _write_scene_config(
    tmp_path: Path,
    visual_mesh: str,
    collision_mesh: str,
    *,
    scale: float = 1.0,
    position_m: list[float] | None = None,
    collision_mesh_offset_m: list[float] | None = None,
    regions: dict[str, object] | None = None,
) -> Path:
    config_path = tmp_path / "engine_scene.yaml"
    assets = {"visual_mesh": visual_mesh}
    if collision_mesh:
        assets["collision_mesh"] = collision_mesh
    if collision_mesh_offset_m is not None:
        assets["collision_mesh_offset_m"] = collision_mesh_offset_m
    config_path.write_text(
        yaml.safe_dump(
            {
                "name": "test_engine",
                "scene_type": "engine_cleaning",
                "engine": {
                    "assets": assets,
                    "scale": scale,
                    "pose": {
                        "position_m": position_m or [0.0, 0.0, 0.0],
                        "quat_wxyz": [1.0, 0.0, 0.0, 0.0],
                    },
                },
                "regions": regions
                or {
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
