from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose

from continuum_sim.dynamics import (
    PCCDynamicsConfig,
    PCCDynamicsState,
    contact_generalized_force,
    mass_matrix,
    step_dynamics,
    stiffness_matrix,
)
from continuum_sim.model import ThreeSegmentRobotParams


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROBOT_CONFIG = PROJECT_ROOT / "configs" / "robot_3seg.yaml"


def test_pcc_mass_matrix_is_positive_definite() -> None:
    params = ThreeSegmentRobotParams.from_yaml(ROBOT_CONFIG)
    config = PCCDynamicsConfig.default(params)
    matrix = mass_matrix(np.zeros(params.q_size, dtype=float), params, config)

    assert matrix.shape == (params.q_size, params.q_size)
    assert np.all(np.linalg.eigvalsh(matrix) > 0.0)


def test_contact_force_projects_to_generalized_coordinates() -> None:
    params = ThreeSegmentRobotParams.from_yaml(ROBOT_CONFIG)
    q = np.zeros(params.q_size, dtype=float)
    tau = contact_generalized_force(q, np.array([1.0, 0.0, 0.0], dtype=float), params)

    assert tau.shape == (params.q_size,)
    assert np.all(np.isfinite(tau))
    assert float(np.linalg.norm(tau)) > 0.0


def test_zero_state_remains_at_rest_without_loads() -> None:
    params = ThreeSegmentRobotParams.from_yaml(ROBOT_CONFIG)
    config = PCCDynamicsConfig.default(params)
    state = PCCDynamicsState(
        q=np.zeros(params.q_size, dtype=float),
        qdot=np.zeros(params.q_size, dtype=float),
    )

    next_state, info = step_dynamics(
        state,
        applied_generalized_force=np.zeros(params.q_size, dtype=float),
        params=params,
        config=config,
        dt=0.01,
    )

    assert_allclose(next_state.q, state.q)
    assert_allclose(next_state.qdot, state.qdot)
    assert stiffness_matrix(params, config).shape == (params.q_size, params.q_size)
    assert info["qddot"].shape == (params.q_size,)
