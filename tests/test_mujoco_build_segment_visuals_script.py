import subprocess
import sys
from pathlib import Path
from xml.etree import ElementTree

import pytest
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MUJOCO_CONFIG = PROJECT_ROOT / "configs" / "mujoco.yaml"
BASE_XML = PROJECT_ROOT / "assets" / "mujoco" / "three_segment_arm.xml"
TENDON_XML = PROJECT_ROOT / "assets" / "mujoco" / "three_segment_arm_tendon.xml"
SCRIPT = PROJECT_ROOT / "scripts" / "build_mujoco_with_segment_visuals.py"


def test_build_segment_visuals_generates_xml_without_overwriting_base(tmp_path: Path) -> None:
    visual_dir = tmp_path / "visuals"
    output_path = tmp_path / "with_visuals.xml"
    config_path = _write_enabled_config(
        tmp_path,
        visual_dir,
        output_path,
        control_mode="position_joint",
    )
    original_base_text = BASE_XML.read_text(encoding="utf-8")

    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--config", str(config_path)],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert f"generated_xml_path: {output_path.resolve()}" in result.stdout
    assert output_path.is_file()
    assert BASE_XML.read_text(encoding="utf-8") == original_base_text

    root = ElementTree.parse(output_path).getroot()
    assert root.find("./asset/mesh[@name='segment_1_link_1_visual_mesh']") is not None
    visual_geom = root.find(".//geom[@name='segment_1_link_1_visual_geom']")
    assert visual_geom is not None
    assert visual_geom.get("contype") == "0"
    assert visual_geom.get("conaffinity") == "0"
    assert visual_geom.get("group") == str(raw_visual_group(config_path))
    collision_geom = root.find(".//geom[@name='segment_1_link_1_geom']")
    assert collision_geom is not None
    assert collision_geom.get("group") == str(raw_collision_group(config_path))


def test_build_segment_visuals_writes_cad_global_body_local_offsets(
    tmp_path: Path,
) -> None:
    visual_dir = tmp_path / "visuals"
    output_path = tmp_path / "with_visuals.xml"
    config_path = _write_enabled_config(
        tmp_path,
        visual_dir,
        output_path,
        control_mode="position_joint",
    )

    subprocess.run(
        [sys.executable, str(SCRIPT), "--config", str(config_path)],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    root = ElementTree.parse(output_path).getroot()
    first_link_geom = root.find(".//geom[@name='segment_1_link_1_visual_geom']")
    second_segment_geom = root.find(".//geom[@name='segment_2_link_1_visual_geom']")
    assert first_link_geom is not None
    assert second_segment_geom is not None

    first_pos = _parse_vec(first_link_geom.get("pos"))
    second_segment_pos = _parse_vec(second_segment_geom.get("pos"))

    assert any(abs(value) > 1.0e-12 for value in first_pos)
    assert first_pos == pytest.approx((-0.011160794, -0.010092945, -0.020345005))
    assert second_segment_pos == pytest.approx(
        (-0.011160794, -0.010092945, -0.060345005)
    )


def test_build_segment_visuals_fails_when_mesh_missing(tmp_path: Path) -> None:
    visual_dir = tmp_path / "visuals"
    output_path = tmp_path / "with_visuals.xml"
    config_path = _write_enabled_config(
        tmp_path,
        visual_dir,
        output_path,
        control_mode="position_joint",
        skip_mesh="segment_3_link_4_visual.stl",
    )

    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--config", str(config_path)],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "segment_3_link_4_visual.stl" in result.stderr
    assert not output_path.exists()


def test_build_segment_visuals_fails_when_visuals_disabled(tmp_path: Path) -> None:
    visual_dir = tmp_path / "visuals"
    output_path = tmp_path / "with_visuals.xml"
    config_path = _write_enabled_config(
        tmp_path,
        visual_dir,
        output_path,
        control_mode="position_joint",
    )
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    raw["visuals"]["enabled"] = False
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--config", str(config_path)],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "visuals.enabled must be true" in result.stderr
    assert not output_path.exists()


def test_build_segment_visuals_supports_tendon_xml_input(tmp_path: Path) -> None:
    visual_dir = tmp_path / "visuals"
    output_path = tmp_path / "with_tendon_visuals.xml"
    config_path = _write_enabled_config(
        tmp_path,
        visual_dir,
        output_path,
        control_mode="tendon_position",
        source_xml_path=TENDON_XML,
    )
    original_tendon_text = TENDON_XML.read_text(encoding="utf-8")

    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--config", str(config_path)],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert f"generated_xml_path: {output_path.resolve()}" in result.stdout
    assert output_path.is_file()
    assert TENDON_XML.read_text(encoding="utf-8") == original_tendon_text

    root = ElementTree.parse(output_path).getroot()
    tendon_geom = root.find(".//geom[@name='segment_3_link_4_visual_geom']")
    actuator = root.find("./actuator/position[@name='act_tendon_9']")
    assert tendon_geom is not None
    assert tendon_geom.get("density") == "0"
    assert tendon_geom.get("group") == str(raw_visual_group(config_path))
    assert actuator is not None
    assert actuator.get("tendon") == "tendon_9"


def _write_enabled_config(
    tmp_path: Path,
    visual_dir: Path,
    output_path: Path,
    *,
    control_mode: str,
    source_xml_path: Path | None = None,
    skip_mesh: str | None = None,
) -> Path:
    raw = yaml.safe_load(MUJOCO_CONFIG.read_text(encoding="utf-8"))
    visual_dir.mkdir()
    for mesh_name in raw["visuals"]["expected_meshes"]:
        if mesh_name == skip_mesh:
            continue
        (visual_dir / mesh_name).write_text("", encoding="utf-8")
    raw["robot_config_path"] = str(PROJECT_ROOT / "configs" / "robot_3seg.yaml")
    raw["control_mode"] = control_mode
    if control_mode == "position_joint":
        raw["xml_path"] = str(source_xml_path or BASE_XML)
        raw["generated_xml_path"] = str(output_path)
    elif control_mode == "tendon_position":
        raw["tendon_xml_path"] = str(source_xml_path or TENDON_XML)
        raw["tendon_generated_xml_path"] = str(output_path)
    else:
        raise AssertionError(f"Unexpected control_mode {control_mode!r}.")
    raw["visuals"]["enabled"] = True
    raw["visuals"]["directory"] = str(visual_dir)
    config_path = tmp_path / "mujoco.yaml"
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return config_path


def raw_visual_group(config_path: Path) -> int:
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    return int(raw["visuals"]["visual_geom_group"])


def raw_collision_group(config_path: Path) -> int:
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    return int(raw["visuals"]["collision_geom_group"])


def _parse_vec(raw_value: str | None) -> tuple[float, float, float]:
    assert raw_value is not None
    parts = tuple(float(part) for part in raw_value.split())
    assert len(parts) == 3
    return parts
