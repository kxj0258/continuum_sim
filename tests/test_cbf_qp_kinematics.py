import numpy as np
from numpy.testing import assert_allclose

from continuum_sim.control.cbf_qp_kinematics import solve_cbf_qp_velocity


def test_cbf_qp_velocity_projects_reference_onto_barrier_halfspace() -> None:
    velocity = solve_cbf_qp_velocity(
        reference_velocity=np.array([-1.0, 0.25], dtype=float),
        barrier_jacobian=np.array([[1.0, 0.0]], dtype=float),
        barrier_lower_bound=np.array([0.2], dtype=float),
    )

    assert velocity[0] >= 0.2 - 1.0e-9
    assert_allclose(velocity[1], 0.25)


def test_cbf_qp_velocity_honors_linear_equality_first() -> None:
    velocity = solve_cbf_qp_velocity(
        reference_velocity=np.array([1.0, 2.0], dtype=float),
        equality_matrix=np.array([[1.0, 1.0]], dtype=float),
        equality_target=np.array([0.0], dtype=float),
    )

    assert_allclose(np.sum(velocity), 0.0, atol=1.0e-9)
    assert_allclose(velocity, [-0.5, 0.5], atol=1.0e-9)
