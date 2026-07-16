from pathlib import Path
from xml.etree import ElementTree

import numpy as np
import pytest
from numpy.testing import assert_allclose

from continuum_sim import load_mujoco_config
from continuum_sim.backends import pcc_q_to_joint_targets
from continuum_sim.model import ThreeSegmentRobotParams


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MUJOCO_CONFIG = PROJECT_ROOT / "configs" / "mujoco.yaml"
ROBOT_CONFIG = PROJECT_ROOT / "configs" / "robot_3seg.yaml"


def _params_and_links() -> tuple[ThreeSegmentRobotParams, int]:
    config = load_mujoco_config(MUJOCO_CONFIG)
    return ThreeSegmentRobotParams.from_yaml(ROBOT_CONFIG), config.links_per_segment


def test_zero_pcc_q_maps_to_zero_joint_targets() -> None:
    params, links_per_segment = _params_and_links()

    targets = pcc_q_to_joint_targets(np.zeros(9), params, links_per_segment)

    assert targets.shape == (3 * links_per_segment * 2,)
    assert_allclose(targets, 0.0, atol=1.0e-14)


def test_single_segment_bending_only_affects_matching_joint_targets() -> None:
    params, links_per_segment = _params_and_links()
    q = np.zeros(9)
    q[3] = 6.0
    q[4] = -3.0
    q[5] = 0.25

    targets = pcc_q_to_joint_targets(q, params, links_per_segment)
    per_segment = targets.reshape(3, links_per_segment, 2)
    segment = params.segments[1]
    expected_link_targets = np.array(
        [
            [0.0, q[3] * segment.effective_flexure_length / 2.0],
            [-q[4] * segment.effective_flexure_length / 2.0, 0.0],
            [0.0, q[3] * segment.effective_flexure_length / 2.0],
            [-q[4] * segment.effective_flexure_length / 2.0, 0.0],
        ],
        dtype=float,
    )

    assert_allclose(per_segment[0], 0.0, atol=1.0e-14)
    assert_allclose(per_segment[1], expected_link_targets, atol=1.0e-14)
    assert_allclose(per_segment[2], 0.0, atol=1.0e-14)


def test_joint_target_shape_matches_mjcf_actuator_count() -> None:
    config = load_mujoco_config(MUJOCO_CONFIG)
    params = ThreeSegmentRobotParams.from_yaml(ROBOT_CONFIG)
    root = ElementTree.parse(config.xml_path).getroot()
    actuator_root = root.find("actuator")
    assert actuator_root is not None

    targets = pcc_q_to_joint_targets(np.zeros(9), params, config.links_per_segment)

    assert targets.shape == (len(list(actuator_root)),)


def test_pcc_q_to_joint_targets_rejects_bad_inputs() -> None:
    params, links_per_segment = _params_and_links()

    with pytest.raises(ValueError, match="q"):
        pcc_q_to_joint_targets(np.zeros((3, 3)), params, links_per_segment)

    with pytest.raises(ValueError, match="links_per_segment"):
        pcc_q_to_joint_targets(np.zeros(9), params, 0)
