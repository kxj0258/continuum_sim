from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose

from continuum_sim.backends import (
    AnalyticBackend,
    BackendState,
    load_analytic_backend_config,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PCC_CONFIG = PROJECT_ROOT / "configs" / "pcc.yaml"


def test_load_analytic_backend_config_resolves_robot_path() -> None:
    config = load_analytic_backend_config(PCC_CONFIG)

    assert config.path == PCC_CONFIG.resolve()
    assert config.robot_config_path == (PROJECT_ROOT / "configs" / "robot_3seg.yaml").resolve()
    assert config.samples_per_segment == 20
    assert config.timestep == pytest.approx(1.0)


def test_analytic_backend_reset_returns_straight_pose_state() -> None:
    backend = AnalyticBackend.from_config(PCC_CONFIG)

    state = backend.reset()

    assert isinstance(state, BackendState)
    assert state.time == pytest.approx(0.0)
    assert state.tip_pose.shape == (4, 4)
    assert state.segment_poses.shape == (3, 4, 4)
    assert state.qpos.shape == (9,)
    assert state.qvel.shape == (9,)
    assert state.tendon_length.shape == (9,)
    assert state.tendon_velocity.shape == (9,)
    assert state.actuator_force.shape == (9,)
    assert_allclose(state.tip_pose[:3, 3], [0.0, 0.0, 0.12], atol=1.0e-12)
    assert_allclose(state.segment_poses[:, 2, 3], [0.04, 0.08, 0.12], atol=1.0e-12)


def test_analytic_backend_step_tracks_motor_position_command() -> None:
    backend = AnalyticBackend.from_config(PCC_CONFIG)
    command = np.zeros(9, dtype=float)
    command[0] = 0.1

    state = backend.step(command, n_substeps=2)

    assert state.time == pytest.approx(2.0)
    assert np.linalg.norm(state.tendon_length) > 0.0
    assert np.linalg.norm(state.qpos) > 0.0
    assert np.all(np.isfinite(state.tip_pose))
    assert np.all(np.isfinite(state.segment_poses))


def test_analytic_backend_from_config_returns_backend() -> None:
    backend = AnalyticBackend.from_config(PCC_CONFIG)

    assert isinstance(backend, AnalyticBackend)
    assert backend.step(np.zeros(9, dtype=float), n_substeps=2).time == pytest.approx(2.0)
