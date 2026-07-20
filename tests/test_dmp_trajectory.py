import numpy as np
from numpy.testing import assert_allclose

from continuum_sim.tasks.dmp_trajectory import DiscreteDMP


def test_discrete_dmp_rollout_adapts_start_and_goal() -> None:
    time = np.linspace(0.0, 1.0, 60)
    demo = np.column_stack(
        (
            0.04 * time,
            0.01 * np.sin(np.pi * time),
            0.12 + 0.02 * time,
        )
    )
    dmp = DiscreteDMP(basis_count=16, samples=80).imitate(time, demo)

    start = np.array([0.02, -0.01, 0.10], dtype=float)
    goal = np.array([0.08, 0.01, 0.18], dtype=float)
    rollout = dmp.rollout(start, goal, tau=1.5)

    assert rollout.position.shape == (80, 3)
    assert np.all(np.isfinite(rollout.position))
    assert_allclose(rollout.position[0], start, atol=1.0e-12)
    assert_allclose(rollout.position[-1], goal, atol=3.0e-3)
    assert float(np.max(rollout.position[:, 1]) - np.min(rollout.position[:, 1])) > 0.005
