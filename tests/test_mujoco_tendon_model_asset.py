import importlib.util
import math
import subprocess
import sys
from pathlib import Path
from xml.etree import ElementTree

import pytest
import yaml

from continuum_sim import load_mujoco_config, load_yaml
from continuum_sim.model import ThreeSegmentRobotParams, load_physical_tendons_from_yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MUJOCO_CONFIG = PROJECT_ROOT / "configs" / "mujoco.yaml"
ROBOT_CONFIG = PROJECT_ROOT / "configs" / "robot_3seg.yaml"
BASE_XML = PROJECT_ROOT / "assets" / "mujoco" / "three_segment_arm.xml"
SCRIPT = PROJECT_ROOT / "scripts" / "build_mujoco_tendon_model.py"


def test_build_tendon_model_generates_xml_without_overwriting_base(tmp_path: Path) -> None:
    output_path = tmp_path / "three_segment_arm_tendon.xml"
    config_path = _write_tendon_build_config(tmp_path, output_path)
    original_base_text = BASE_XML.read_text(encoding="utf-8")

    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--config", str(config_path)],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert f"tendon_xml_path: {output_path.resolve()}" in result.stdout
    assert output_path.is_file()
    assert BASE_XML.read_text(encoding="utf-8") == original_base_text


def test_committed_tendon_mjcf_contains_nine_fixed_tendons_and_actuators() -> None:
    config = load_mujoco_config(MUJOCO_CONFIG, require_tendon_xml=True)
    root = ElementTree.parse(config.tendon_xml_path).getroot()

    fixed_tendons = root.findall("./tendon/fixed")
    actuator_positions = root.findall("./actuator/position")
    sites = {site.get("name") for site in root.findall(".//site")}
    global_visual = root.find("./visual/global")
    default_joint = root.find("./default/joint")

    assert global_visual is not None
    assert float(str(global_visual.get("azimuth"))) == pytest.approx(
        config.viewer.camera.azimuth
    )
    assert float(str(global_visual.get("elevation"))) == pytest.approx(
        config.viewer.camera.elevation
    )
    assert int(str(global_visual.get("offwidth"))) == config.rendering.offscreen_width
    assert int(str(global_visual.get("offheight"))) == config.rendering.offscreen_height
    assert default_joint is not None
    assert float(str(default_joint.get("damping"))) == pytest.approx(
        config.joints.hinge.damping
    )
    assert float(str(default_joint.get("armature"))) == pytest.approx(
        config.joints.hinge.armature
    )
    assert default_joint.get("limited") == str(config.joints.hinge.limited).lower()
    assert _parse_vec(default_joint.get("range")) == pytest.approx(
        config.joints.hinge.range_rad
    )
    assert float(str(default_joint.get("stiffness"))) == pytest.approx(
        config.joints.hinge.stiffness
    )
    assert float(str(default_joint.get("springref"))) == pytest.approx(
        config.joints.hinge.springref
    )
    assert len(fixed_tendons) == config.tendon_model.count
    assert len(actuator_positions) == config.tendon_model.count
    assert {
        "base_site",
        "segment_1_tip",
        "segment_2_tip",
        "segment_3_tip",
        "tip",
    }.issubset(sites)
    assert all(actuator.get("joint") is None for actuator in actuator_positions)
    assert all(actuator.get("tendon") is not None for actuator in actuator_positions)

    for fixed in fixed_tendons:
        assert fixed.get("limited") == str(config.tendon_model.limited).lower()
        assert _parse_vec(fixed.get("range")) == pytest.approx(
            config.tendon_model.length_range_m
        )
        assert float(str(fixed.get("damping"))) == pytest.approx(
            config.tendon_model.damping
        )
        assert float(str(fixed.get("stiffness"))) == pytest.approx(
            config.tendon_model.stiffness
        )

    for actuator in actuator_positions:
        assert float(str(actuator.get("kp"))) == pytest.approx(
            config.actuators.tendon_position.kp
        )
        assert actuator.get("ctrllimited") == str(
            config.actuators.tendon_position.ctrllimited
        ).lower()
        assert _parse_vec(actuator.get("ctrlrange")) == pytest.approx(
            config.actuators.tendon_position.ctrlrange_m
        )
        assert actuator.get("forcelimited") == str(
            config.actuators.tendon_position.forcelimited
        ).lower()
        assert _parse_vec(actuator.get("forcerange")) == pytest.approx(
            config.actuators.tendon_position.forcerange_n
        )


def test_committed_tendon_visual_mjcf_preserves_fixed_tendons_and_adds_meshes() -> None:
    config = load_mujoco_config(MUJOCO_CONFIG, require_tendon_xml=True)
    assert config.tendon_generated_xml_path.is_file()

    root = ElementTree.parse(config.tendon_generated_xml_path).getroot()
    fixed_tendons = root.findall("./tendon/fixed")
    visual_geoms = root.findall(".//geom[@type='mesh']")

    assert len(fixed_tendons) == config.tendon_model.count
    assert len(visual_geoms) == len(config.visuals.expected_meshes)
    assert all(geom.get("contype") == "0" for geom in visual_geoms)
    assert all(geom.get("conaffinity") == "0" for geom in visual_geoms)
    assert all(geom.get("density") == "0" for geom in visual_geoms)
    assert all(geom.get("group") == str(config.visuals.visual_geom_group) for geom in visual_geoms)


def test_fixed_tendon_coefficients_follow_physical_tendon_paths() -> None:
    config = load_mujoco_config(MUJOCO_CONFIG, require_tendon_xml=True)
    params = ThreeSegmentRobotParams.from_yaml(ROBOT_CONFIG)
    physical_tendons = load_physical_tendons_from_yaml(ROBOT_CONFIG)
    root = ElementTree.parse(config.tendon_xml_path).getroot()
    joint_names = {
        str(joint.get("name"))
        for joint in root.findall(".//joint")
        if joint.get("name") is not None
    }
    fixed_by_name = {
        str(fixed.get("name")): fixed
        for fixed in root.findall("./tendon/fixed")
        if fixed.get("name") is not None
    }

    for tendon in physical_tendons:
        fixed = fixed_by_name[tendon.id]
        fixed_joints = fixed.findall("joint")
        theta_rad = math.radians(tendon.angle_deg)
        coef_x = tendon.radial_offset * math.sin(theta_rad)
        coef_y = -tendon.radial_offset * math.cos(theta_rad)
        expected: list[tuple[str, float]] = []
        for segment_index in tendon.path_segment_indices:
            assert segment_index < len(params.segments)
            segment_number = segment_index + 1
            for link_index in range(config.links_per_segment):
                link_number = link_index + 1
                expected.append((f"segment_{segment_number}_link_{link_number}_x", coef_x))
                expected.append((f"segment_{segment_number}_link_{link_number}_y", coef_y))

        actual = [
            (str(joint.get("joint")), float(str(joint.get("coef"))))
            for joint in fixed_joints
        ]

        assert len(actual) == len(expected)
        assert [name for name, _coef in actual] == [name for name, _coef in expected]
        assert all(name in joint_names for name, _coef in actual)
        assert [coef for _name, coef in actual] == pytest.approx(
            [coef for _name, coef in expected]
        )


def test_tendon_mjcf_loads_in_mujoco_when_available() -> None:
    mujoco = pytest.importorskip("mujoco")
    config = load_mujoco_config(MUJOCO_CONFIG, require_tendon_xml=True)

    model = mujoco.MjModel.from_xml_path(str(config.tendon_xml_path))

    assert model.nu == config.tendon_model.count
    assert model.ntendon == config.tendon_model.count
    for site_name in (
        config.site_names.base,
        *config.site_names.segments,
        config.site_names.tip,
    ):
        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        assert site_id >= 0


def test_build_tendon_model_helper_rejects_disabled_tendon_model(tmp_path: Path) -> None:
    module = _load_script_module(SCRIPT, "build_mujoco_tendon_model_for_test")
    output_path = tmp_path / "three_segment_arm_tendon.xml"
    config_path = _write_tendon_build_config(tmp_path, output_path)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    raw["tendon_model"]["enabled"] = False
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="tendon_model.enabled"):
        module.build_mujoco_tendon_model(config_path=config_path)


def test_build_tendon_model_writes_zero_gravity_when_disabled(tmp_path: Path) -> None:
    module = _load_script_module(SCRIPT, "build_mujoco_tendon_model_gravity_for_test")
    output_path = tmp_path / "three_segment_arm_tendon_zero_gravity.xml"
    config_path = _write_tendon_build_config(tmp_path, output_path)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    raw["gravity"] = {
        "enabled": False,
        "vector_m_s2": [0.0, 0.0, -9.81],
    }
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    built_path = module.build_mujoco_tendon_model(config_path=config_path)
    root = ElementTree.parse(built_path).getroot()
    option = root.find("./option")

    assert option is not None
    assert _parse_vec(option.get("gravity")) == pytest.approx((0.0, 0.0, 0.0))


def _write_tendon_build_config(tmp_path: Path, output_path: Path) -> Path:
    raw = load_yaml(MUJOCO_CONFIG)
    raw["robot_config_path"] = str(ROBOT_CONFIG)
    raw["xml_path"] = str(BASE_XML)
    raw["tendon_xml_path"] = str(output_path)
    raw["visuals"]["directory"] = str(
        PROJECT_ROOT / "assets" / "meshes" / "mujoco_visual_segments"
    )
    raw["visuals"]["template_path"] = str(
        PROJECT_ROOT / "assets" / "mujoco" / "segmented_visuals_template.xml"
    )
    config_path = tmp_path / "mujoco.yaml"
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return config_path


def _parse_vec(raw_value: str | None) -> tuple[float, ...]:
    assert raw_value is not None
    return tuple(float(part) for part in raw_value.split())


def _load_script_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module
