import numpy as np
from numpy.testing import assert_allclose

from continuum_sim.kinematics.sdf import (
    damped_pseudoinverse,
    fuse_task_and_nullspace_velocity,
    nullspace_projector,
    sdf_repulsive_velocity,
)


def test_nullspace_projector_preserves_primary_task() -> None:
    jacobian = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=float)
    task_velocity = np.array([0.03, -0.02], dtype=float)
    repulsive_qdot = np.array([1.0, 1.0, 0.5], dtype=float)

    projector = nullspace_projector(jacobian)
    qdot = fuse_task_and_nullspace_velocity(jacobian, task_velocity, repulsive_qdot)

    assert_allclose(jacobian @ projector, np.zeros((2, 3)), atol=1.0e-9)
    assert_allclose(jacobian @ qdot, task_velocity, atol=1.0e-9)
    assert qdot[2] != 0.0


def test_sdf_repulsive_velocity_activates_only_inside_influence_distance() -> None:
    gradient = np.array([2.0, 0.0, 0.0], dtype=float)

    active = sdf_repulsive_velocity(
        distance_m=0.006,
        gradient=gradient,
        safe_distance_m=0.010,
        influence_distance_m=0.030,
        gain=0.5,
    )
    inactive = sdf_repulsive_velocity(
        distance_m=0.050,
        gradient=gradient,
        safe_distance_m=0.010,
        influence_distance_m=0.030,
        gain=0.5,
    )

    assert active[0] > 0.0
    assert_allclose(inactive, np.zeros(3, dtype=float))
    assert_allclose(damped_pseudoinverse(np.eye(2)), np.eye(2), atol=1.0e-9)
