import importlib.util
import subprocess
import sys
from pathlib import Path
from xml.etree import ElementTree

import pytest
import yaml

from continuum_sim import load_mujoco_config, load_yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MUJOCO_CONFIG = PROJECT_ROOT / "configs" / "mujoco_segment_2dof.yaml"
ROBOT_CONFIG = PROJECT_ROOT / "configs" / "robot_3seg.yaml"
SCRIPT = PROJECT_ROOT / "scripts" / "build_mujoco_segment_2dof_model.py"


def test_build_segment_2dof_model_generates_xml(tmp_path: Path) -> None:
    output_path = tmp_path / "three_segment_arm_2dof_tendon.xml"
    visual_path = tmp_path / "three_segment_arm_2dof_tendon_with_visuals.xml"
    config_path = _write_segment_2dof_build_config(tmp_path, output_path, visual_path)

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--config",
            str(config_path),
        ],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert f"segment_2dof_tendon_xml_path: {output_path.resolve()}" in result.stdout
    assert f"segment_2dof_tendon_visual_xml_path: {visual_path.resolve()}" in result.stdout
    assert output_path.is_file()
    assert visual_path.is_file()


def test_committed_segment_2dof_mjcf_contains_expected_topology() -> None:
    config = load_mujoco_config(
        MUJOCO_CONFIG,
        require_xml=True,
        require_tendon_xml=True,
        require_visual_meshes=False,
    )
    root = ElementTree.parse(config.tendon_xml_path).getroot()

    hinges = [joint for joint in root.findall(".//joint") if joint.get("type") == "hinge"]
    fixed_tendons = root.findall("./tendon/fixed")
    actuators = root.findall("./actuator/position")
    follower_bodies = [
        body
        for body in root.findall(".//body")
        if str(body.get("name", "")).startswith("follower_segment_")
    ]
    follower_collisions = [
        geom
        for geom in root.findall(".//geom")
        if str(geom.get("name", "")).endswith("_collision")
    ]
    sites = {site.get("name") for site in root.findall(".//site")}
    tendon_joint_names = {
        joint.get("joint")
        for fixed in fixed_tendons
        for joint in fixed.findall("joint")
    }

    assert [joint.get("name") for joint in hinges] == [
        "segment_1_x",
        "segment_1_y",
        "segment_2_x",
        "segment_2_y",
        "segment_3_x",
        "segment_3_y",
    ]
    assert len(fixed_tendons) == 9
    assert len(actuators) == 9
    assert all(actuator.get("tendon") for actuator in actuators)
    assert len(follower_bodies) == 12
    assert all(body.get("mocap") == "true" for body in follower_bodies)
    assert len(follower_collisions) == 12
    assert tendon_joint_names == {
        "segment_1_x",
        "segment_1_y",
        "segment_2_x",
        "segment_2_y",
        "segment_3_x",
        "segment_3_y",
    }
    assert {
        "base_site",
        "segment_1_tip",
        "segment_2_tip",
        "segment_3_tip",
        "tip",
    }.issubset(sites)


def test_committed_segment_2dof_visual_mjcf_adds_follower_visuals() -> None:
    config = load_mujoco_config(
        MUJOCO_CONFIG,
        require_xml=True,
        require_tendon_xml=True,
        require_visual_meshes=False,
    )
    root = ElementTree.parse(config.tendon_generated_xml_path).getroot()

    follower_visuals = [
        geom
        for geom in root.findall(".//geom")
        if str(geom.get("name", "")).endswith("_visual")
    ]

    assert len(follower_visuals) == 12
    assert all(geom.get("contype") == "0" for geom in follower_visuals)
    assert all(geom.get("conaffinity") == "0" for geom in follower_visuals)
    assert all(geom.get("group") == str(config.visuals.visual_geom_group) for geom in follower_visuals)


def test_segment_2dof_mjcf_loads_in_mujoco_when_available() -> None:
    mujoco = pytest.importorskip("mujoco")
    config = load_mujoco_config(
        MUJOCO_CONFIG,
        require_xml=True,
        require_tendon_xml=True,
        require_visual_meshes=False,
    )

    model = mujoco.MjModel.from_xml_path(str(config.tendon_xml_path))

    assert model.nv == 6
    assert model.nu == 9
    assert model.ntendon == 9
    assert model.nmocap == 12


def test_build_segment_2dof_model_rejects_wrong_model_type(tmp_path: Path) -> None:
    module = _load_script_module(SCRIPT, "build_mujoco_segment_2dof_model_for_test")
    output_path = tmp_path / "segment_2dof.xml"
    visual_path = tmp_path / "segment_2dof_visual.xml"
    config_path = _write_segment_2dof_build_config(tmp_path, output_path, visual_path)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    raw["model"]["type"] = "distributed_links"
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="segment_2dof_followers"):
        module.build_mujoco_segment_2dof_model(config_path=config_path)


def _write_segment_2dof_build_config(
    tmp_path: Path,
    output_path: Path,
    visual_path: Path,
) -> Path:
    raw = load_yaml(MUJOCO_CONFIG)
    raw["robot_config_path"] = str(ROBOT_CONFIG)
    raw["xml_path"] = str(output_path)
    raw["tendon_xml_path"] = str(output_path)
    raw["generated_xml_path"] = str(visual_path)
    raw["tendon_generated_xml_path"] = str(visual_path)
    config_path = tmp_path / "mujoco_segment_2dof.yaml"
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return config_path


def _load_script_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module
