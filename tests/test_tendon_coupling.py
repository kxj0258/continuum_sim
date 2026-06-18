from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose

from continuum_sim.model.physical_tendon import load_physical_tendons_from_yaml
from continuum_sim.model.robot_params import ThreeSegmentRobotParams
from continuum_sim.model.tendon_coupling import (
    build_coupling_matrix,
    coupling_diagnostics,
    physical_tendon_delta_to_q,
    q_to_physical_tendon_delta,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROBOT_CONFIG = PROJECT_ROOT / "configs" / "robot_3seg.yaml"


def test_build_coupling_matrix_has_expected_shape_and_diagnostics() -> None:
    params = ThreeSegmentRobotParams.from_yaml(ROBOT_CONFIG)
    physical_tendons = load_physical_tendons_from_yaml(ROBOT_CONFIG)

    C = build_coupling_matrix(params, physical_tendons)
    diagnostics = coupling_diagnostics(C)

    assert C.shape == (9, 9)
    assert "rank" in diagnostics
    assert "condition_number" in diagnostics
    assert "is_full_rank" in diagnostics
    assert diagnostics["rank"] == 9
    assert diagnostics["is_full_rank"] is True


def test_zero_q_maps_to_zero_physical_tendon_delta() -> None:
    params = ThreeSegmentRobotParams.from_yaml(ROBOT_CONFIG)
    physical_tendons = load_physical_tendons_from_yaml(ROBOT_CONFIG)

    tendon_delta = q_to_physical_tendon_delta(np.zeros(9), params, physical_tendons)

    assert_allclose(tendon_delta, np.zeros(9), atol=1.0e-14)


def test_physical_tendon_mapping_round_trip() -> None:
    params = ThreeSegmentRobotParams.from_yaml(ROBOT_CONFIG)
    physical_tendons = load_physical_tendons_from_yaml(ROBOT_CONFIG)
    q = np.array(
        [
            6.0,
            -2.0,
            0.01,
            -4.0,
            5.0,
            -0.004,
            3.0,
            1.5,
            0.002,
        ]
    )

    tendon_delta = q_to_physical_tendon_delta(q, params, physical_tendons)
    recovered_q = physical_tendon_delta_to_q(tendon_delta, params, physical_tendons)

    assert_allclose(recovered_q, q, atol=1.0e-10)


def test_coupling_matrix_matches_physical_path_blocks() -> None:
    params = ThreeSegmentRobotParams.from_yaml(ROBOT_CONFIG)
    physical_tendons = load_physical_tendons_from_yaml(ROBOT_CONFIG)
    C = build_coupling_matrix(params, physical_tendons)

    for row in range(3):
        assert np.linalg.norm(C[row, 0:3]) > 0.0
        assert_allclose(C[row, 3:9], np.zeros(6), atol=1.0e-14)

    for row in range(3, 6):
        assert np.linalg.norm(C[row, 0:3]) > 0.0
        assert np.linalg.norm(C[row, 3:6]) > 0.0
        assert_allclose(C[row, 6:9], np.zeros(3), atol=1.0e-14)

    for row in range(6, 9):
        assert np.linalg.norm(C[row, 0:3]) > 0.0
        assert np.linalg.norm(C[row, 3:6]) > 0.0
        assert np.linalg.norm(C[row, 6:9]) > 0.0


def test_coupling_mapping_rejects_wrong_shapes() -> None:
    params = ThreeSegmentRobotParams.from_yaml(ROBOT_CONFIG)
    physical_tendons = load_physical_tendons_from_yaml(ROBOT_CONFIG)

    with np.testing.assert_raises(ValueError):
        q_to_physical_tendon_delta(np.zeros((9, 1)), params, physical_tendons)

    with np.testing.assert_raises(ValueError):
        physical_tendon_delta_to_q(np.zeros(8), params, physical_tendons)
