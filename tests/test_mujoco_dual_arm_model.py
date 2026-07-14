from __future__ import annotations

from pathlib import Path
import xml.etree.ElementTree as ET

from scripts.build_mujoco_dual_arm_model import build_mujoco_dual_arm_model


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "mujoco_dual.yaml"


def _assert_alternating_single_axis_flexures(root: ET.Element) -> None:
    for arm_name in ("executor", "observer"):
        for global_link_index in range(1, 13):
            segment_number = (global_link_index - 1) // 4 + 1
            link_number = (global_link_index - 1) % 4 + 1
            body_name = (
                f"{arm_name}_segment_{segment_number}_link_{link_number}"
            )
            body = root.find(f".//body[@name='{body_name}']")
            assert body is not None
            joints = body.findall("./joint")
            assert len(joints) == 1
            expected_axis_name = "y" if global_link_index % 2 else "x"
            expected_axis = "0 1 0" if global_link_index % 2 else "1 0 0"
            assert joints[0].get("name") == f"{body_name}_{expected_axis_name}"
            assert joints[0].get("type") == "hinge"
            assert joints[0].get("axis") == expected_axis


def test_dual_builder_generates_base_and_mobile_models_from_yaml(
    tmp_path: Path,
) -> None:
    base_path = tmp_path / "dual.xml"
    mobile_path = tmp_path / "dual_mobile_base.xml"

    result = build_mujoco_dual_arm_model(
        config_path=CONFIG_PATH,
        output_path=base_path,
        mobile_base_output_path=mobile_path,
    )

    assert result.base_xml_path == base_path
    assert result.mobile_base_xml_path == mobile_path
    assert base_path.is_file()
    assert mobile_path.is_file()

    base_root = ET.parse(base_path).getroot()
    mobile_root = ET.parse(mobile_path).getroot()
    actuators = base_root.findall("./actuator/position")
    assert len(actuators) == 18
    assert {actuator.get("forcerange") for actuator in actuators} == {"-30 30"}
    assert {actuator.get("ctrllimited") for actuator in actuators} == {"false"}
    assert base_root.find("./worldbody/site[@name='world_origin']") is not None
    assert base_root.find("./worldbody/site[@name='world_x_axis']") is not None
    assert base_root.find("./worldbody/site[@name='world_y_axis']") is not None
    assert base_root.find("./worldbody/site[@name='world_z_axis']") is not None
    assert (
        mobile_root.find(".//freejoint[@name='mobile_base_freejoint']")
        is not None
    )
    _assert_alternating_single_axis_flexures(base_root)
    _assert_alternating_single_axis_flexures(mobile_root)


def test_committed_dual_models_keep_yaml_actuator_force_range() -> None:
    for filename in (
        "dual_three_segment_arm_tendon_with_visuals.xml",
        "dual_three_segment_arm_tendon_with_visuals_mobile_base.xml",
    ):
        root = ET.parse(PROJECT_ROOT / "assets" / "mujoco" / filename).getroot()
        actuators = root.findall("./actuator/position")
        assert len(actuators) == 18
        assert {actuator.get("forcerange") for actuator in actuators} == {"-30 30"}
        assert {actuator.get("ctrllimited") for actuator in actuators} == {"false"}
        assert root.find(".//site[@name='world_origin']") is not None
        _assert_alternating_single_axis_flexures(root)
