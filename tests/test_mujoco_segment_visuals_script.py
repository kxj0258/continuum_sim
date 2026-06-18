import subprocess
import sys
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MUJOCO_CONFIG = PROJECT_ROOT / "configs" / "mujoco.yaml"
SCRIPT = PROJECT_ROOT / "scripts" / "check_mujoco_segment_visuals.py"


def test_check_segment_visuals_allow_missing_returns_success(tmp_path: Path) -> None:
    config_path = _write_visual_config(tmp_path, tmp_path / "missing")

    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--config", str(config_path), "--allow-missing"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "expected_meshes:" in result.stdout
    assert "missing_meshes:" in result.stdout
    assert "Do not enable true per-link visual following" in result.stdout


def test_check_segment_visuals_missing_returns_nonzero(tmp_path: Path) -> None:
    config_path = _write_visual_config(tmp_path, tmp_path / "missing")

    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--config", str(config_path)],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "missing_meshes:" in result.stdout


def test_check_segment_visuals_create_placeholders_does_not_create_fake_stl(
    tmp_path: Path,
) -> None:
    visual_dir = tmp_path / "visuals"
    config_path = _write_visual_config(tmp_path, visual_dir)

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--config",
            str(config_path),
            "--create-placeholders",
            "--allow-missing",
        ],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "placeholder_directory_ready" in result.stdout
    assert visual_dir.is_dir()
    assert (visual_dir / "README.md").is_file()
    assert not list(visual_dir.glob("*.stl"))


def test_check_segment_visuals_reports_bounds_for_existing_meshes() -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--config", str(MUJOCO_CONFIG)],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "mesh_bounds:" in result.stdout
    assert "segment_1_link_1_visual.stl" in result.stdout
    assert "min_model_m:" in result.stdout
    assert "All segmented visual mesh files are present." in result.stdout


def test_check_segment_visuals_warns_when_cad_global_meshes_are_body_local(
    tmp_path: Path,
) -> None:
    raw = yaml.safe_load(MUJOCO_CONFIG.read_text(encoding="utf-8"))
    raw["robot_config_path"] = str(PROJECT_ROOT / "configs" / "robot_3seg.yaml")
    raw["visuals"]["frame_mode"] = "body_local"
    raw["xml_path"] = str(PROJECT_ROOT / "assets" / "mujoco" / "three_segment_arm.xml")
    raw["visuals"]["directory"] = str(
        PROJECT_ROOT / "assets" / "meshes" / "mujoco_visual_segments"
    )
    raw["visuals"]["template_path"] = str(
        PROJECT_ROOT / "assets" / "mujoco" / "segmented_visuals_template.xml"
    )
    config_path = tmp_path / "mujoco.yaml"
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--config", str(config_path)],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "frame_warnings:" in result.stdout
    assert "CAD-global STL coordinates" in result.stdout


def _write_visual_config(tmp_path: Path, visual_dir: Path) -> Path:
    raw = yaml.safe_load(MUJOCO_CONFIG.read_text(encoding="utf-8"))
    raw["robot_config_path"] = str(PROJECT_ROOT / "configs" / "robot_3seg.yaml")
    raw["visuals"]["directory"] = str(visual_dir)
    config_path = tmp_path / "mujoco.yaml"
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return config_path
