from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose

from continuum_sim import load_mujoco_config
from continuum_sim.backends import BackendState, MujocoBackend
from continuum_sim.backends.mujoco_backend import (
    _effective_gravity_vector,
    _model_xml_path_for_control_mode,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MUJOCO_CONFIG = PROJECT_ROOT / "configs" / "mujoco.yaml"
SEGMENT_2DOF_CONFIG = PROJECT_ROOT / "configs" / "mujoco_segment_2dof.yaml"


@pytest.fixture
def _mujoco_available() -> None:
    pytest.importorskip("mujoco")


def test_mujoco_backend_reset_returns_pose_state(_mujoco_available: None) -> None:
    backend = MujocoBackend.from_config(MUJOCO_CONFIG)
    state = backend.reset()

    assert isinstance(state, BackendState)
    assert state.time == pytest.approx(0.0)
    assert state.tip_pose.shape == (4, 4)
    assert state.segment_poses.shape == (3, 4, 4)
    assert state.qpos.shape == (backend.model.nq,)
    assert state.qvel.shape == (backend.model.nv,)
    assert state.tendon_length.shape == (backend.model.ntendon,)
    assert state.tendon_velocity.shape == (backend.model.ntendon,)
    assert state.actuator_force.shape == (backend.model.nu,)
    assert_allclose(state.tip_pose, backend.get_tip_pose())
    assert_allclose(state.segment_poses, backend.get_segment_poses())
    assert_allclose(state.tip_pose[:3, 3], [0.0, 0.0, 0.12], atol=1.0e-12)
    assert_allclose(state.segment_poses[:, 2, 3], [0.04, 0.08, 0.12], atol=1.0e-12)


def test_mujoco_backend_step_accepts_actuator_control_vector(_mujoco_available: None) -> None:
    backend = MujocoBackend(load_mujoco_config(MUJOCO_CONFIG))
    control = np.zeros(backend.model.nu, dtype=float)
    control[0] = 0.05

    state = backend.step(control, n_substeps=2)

    assert state.time > 0.0
    assert state.tip_pose.shape == (4, 4)
    assert state.segment_poses.shape == (3, 4, 4)
    assert state.qpos.shape == (backend.model.nq,)
    assert state.qvel.shape == (backend.model.nv,)
    assert state.tendon_length.shape == (backend.model.ntendon,)
    assert state.tendon_velocity.shape == (backend.model.ntendon,)
    assert state.actuator_force.shape == (backend.model.nu,)
    assert np.all(np.isfinite(state.tip_pose))
    assert np.all(np.isfinite(state.segment_poses))


def test_mujoco_backend_tendon_position_mode_reset_and_step(
    _mujoco_available: None,
) -> None:
    config = replace(load_mujoco_config(MUJOCO_CONFIG), control_mode="tendon_position")
    backend = MujocoBackend.from_config(config)

    state = backend.reset()
    assert backend.model.nu == config.tendon_model.count
    assert backend.model.ntendon == config.tendon_model.count
    assert state.tip_pose.shape == (4, 4)
    assert state.segment_poses.shape == (3, 4, 4)
    assert state.qpos.shape == (backend.model.nq,)
    assert state.qvel.shape == (backend.model.nv,)
    assert state.tendon_length.shape == (backend.model.ntendon,)
    assert state.tendon_velocity.shape == (backend.model.ntendon,)
    assert state.actuator_force.shape == (backend.model.nu,)

    state = backend.step(np.zeros(config.tendon_model.count), n_substeps=2)

    assert state.time > 0.0
    assert np.all(np.isfinite(state.tendon_length))
    assert np.all(np.isfinite(state.tendon_velocity))
    assert np.all(np.isfinite(state.actuator_force))


def test_mujoco_backend_segment_2dof_followers_reset_and_step(
    _mujoco_available: None,
) -> None:
    config = load_mujoco_config(
        SEGMENT_2DOF_CONFIG,
        require_xml=True,
        require_tendon_xml=True,
        require_visual_meshes=False,
    )
    backend = MujocoBackend.from_config(config)

    state = backend.reset()
    assert backend.model.nv == 6
    assert backend.model.nu == 9
    assert state.qpos.shape == (6,)
    assert state.qvel.shape == (6,)
    assert state.mocap_pos is not None
    assert state.mocap_quat is not None
    assert state.mocap_pos.shape == (12, 3)
    assert state.mocap_quat.shape == (12, 4)
    assert_allclose(state.tip_pose[:3, 3], [0.0, 0.0, 0.12], atol=1.0e-12)
    assert_allclose(state.segment_poses[:, 2, 3], [0.04, 0.08, 0.12], atol=1.0e-12)

    state = backend.step(np.zeros(9, dtype=float), n_substeps=2)

    assert state.time > 0.0
    assert state.qpos.shape == (6,)
    assert state.tendon_length.shape == (9,)


def test_mujoco_backend_overrides_model_gravity_from_config(
    _mujoco_available: None,
) -> None:
    config = load_mujoco_config(MUJOCO_CONFIG)
    zero_gravity = replace(config.gravity, enabled=False)
    backend = MujocoBackend.from_config(replace(config, gravity=zero_gravity))

    assert_allclose(backend.model.opt.gravity, [0.0, 0.0, 0.0], atol=1.0e-12)
    assert _effective_gravity_vector(replace(config, gravity=zero_gravity)) == (
        0.0,
        0.0,
        0.0,
    )


def test_mujoco_backend_rejects_wrong_control_shape(_mujoco_available: None) -> None:
    backend = MujocoBackend.from_config(MUJOCO_CONFIG)

    with pytest.raises(ValueError, match="control"):
        backend.step(np.zeros(backend.model.nu + 1))


def test_mujoco_backend_selects_xml_path_from_control_mode() -> None:
    config = load_mujoco_config(MUJOCO_CONFIG)

    assert (
        _model_xml_path_for_control_mode(replace(config, control_mode="position_joint"))
        == config.xml_path
    )
    assert (
        _model_xml_path_for_control_mode(
            replace(config, control_mode="tendon_position")
        )
        == config.tendon_xml_path
    )


def test_mujoco_backend_from_config_passes_xml_override(monkeypatch, tmp_path: Path) -> None:
    config = load_mujoco_config(MUJOCO_CONFIG)
    override_xml_path = tmp_path / "with_visuals.xml"
    captured = {}

    def fake_init(self, config_arg, mujoco_module=None, *, xml_path=None):
        captured["config"] = config_arg
        captured["mujoco_module"] = mujoco_module
        captured["xml_path"] = xml_path

    monkeypatch.setattr(MujocoBackend, "__init__", fake_init)

    backend = MujocoBackend.from_config(config, override_xml_path=override_xml_path)

    assert isinstance(backend, MujocoBackend)
    assert captured["config"] is config
    assert captured["mujoco_module"] is None
    assert captured["xml_path"] == override_xml_path
