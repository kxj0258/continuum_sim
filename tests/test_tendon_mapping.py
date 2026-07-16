import numpy as np
from numpy.testing import assert_allclose

from continuum_sim.kinematics.tendon_mapping import q_to_tendon_delta, tendon_delta_to_q
from continuum_sim.model.robot_params import ThreeSegmentRobotParams


def test_zero_q_maps_to_zero_tendon_delta() -> None:
    tendon_delta = q_to_tendon_delta(np.zeros(9))

    assert_allclose(tendon_delta, np.zeros(9), atol=1.0e-14)


def test_single_segment_tendon_mapping_matches_closed_form() -> None:
    params = ThreeSegmentRobotParams.default()
    segment = params.segments[0]
    q = np.zeros(9)
    q[0] = 10.0

    tendon_delta = q_to_tendon_delta(q, params)

    expected_first_segment = np.array([-0.001825, 0.0009125, 0.0009125])
    assert_allclose(tendon_delta[:3], expected_first_segment, atol=1.0e-12)
    assert_allclose(tendon_delta[3:], np.zeros(6), atol=1.0e-14)

    expected_sum = 3.0 * segment.effective_flexure_length * q[2]
    assert_allclose(np.sum(tendon_delta[:3]), expected_sum, atol=1.0e-14)


def test_uniform_tendon_delta_maps_to_axial_extension() -> None:
    params = ThreeSegmentRobotParams.default()
    segment = params.segments[0]
    eps = 0.01
    tendon_delta = np.full(9, segment.effective_flexure_length * eps)

    q = tendon_delta_to_q(tendon_delta, params)

    expected = np.tile(np.array([0.0, 0.0, eps]), 3)
    assert_allclose(q, expected, atol=1.0e-12)


def test_q_to_tendon_delta_to_q_round_trip() -> None:
    q = np.array(
        [
            8.0,
            -4.0,
            0.01,
            -3.0,
            6.0,
            -0.005,
            2.0,
            1.0,
            0.0,
        ]
    )

    recovered_q = tendon_delta_to_q(q_to_tendon_delta(q))

    assert_allclose(recovered_q, q, atol=1.0e-12)
