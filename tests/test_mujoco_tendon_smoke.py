from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from continuum_sim import load_mujoco_config, load_yaml
from continuum_sim.backends import MujocoBackend
from continuum_sim.model import load_physical_tendons_from_yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MUJOCO_CONFIG = PROJECT_ROOT / "configs" / "mujoco.yaml"


@pytest.fixture
def _mujoco_available() -> None:
    pytest.importorskip("mujoco")


def test_tendon_position_zero_command_stays_nearly_straight(
    _mujoco_available: None,
) -> None:
    config, backend, initial_tip = _tendon_backend()
    control = np.zeros(config.tendon_model.count, dtype=float)

    state = backend.step(control, n_substeps=_smoke_substeps(config))
    lateral_delta = state.tip_pose[:2, 3] - initial_tip[:2]

    assert np.linalg.norm(lateral_delta) <= config.smoke_tests.zero_command_tolerance_m


def test_tendon_position_single_tendon_shortening_bends_expected_direction(
    _mujoco_available: None,
) -> None:
    config, backend, initial_tip = _tendon_backend()
    physical_tendons = load_physical_tendons_from_yaml(config.robot_config_path)
    tendon = physical_tendons[0]
    command = config.smoke_tests.single_tendon_delta_m
    assert command != 0.0

    zero_state = backend.step(
        np.zeros(config.tendon_model.count, dtype=float),
        n_substeps=_smoke_substeps(config),
    )
    backend.reset()
    control = np.zeros(config.tendon_model.count, dtype=float)
    control[tendon.global_index] = command
    state = backend.step(control, n_substeps=_smoke_substeps(config))

    expected_direction = -np.sign(command) * np.array(
        [
            np.cos(np.deg2rad(tendon.angle_deg)),
            np.sin(np.deg2rad(tendon.angle_deg)),
        ],
        dtype=float,
    )
    zero_projection = float(np.dot(zero_state.tip_pose[:2, 3] - initial_tip[:2], expected_direction))
    single_projection = float(np.dot(state.tip_pose[:2, 3] - initial_tip[:2], expected_direction))

    assert single_projection > zero_projection


def test_tendon_position_symmetric_section_command_has_small_lateral_motion(
    _mujoco_available: None,
) -> None:
    config, backend, initial_tip = _tendon_backend()
    physical_tendons = load_physical_tendons_from_yaml(config.robot_config_path)
    robot = load_yaml(config.robot_config_path)
    tendons_per_segment = int(robot["robot"]["tendons_per_segment"])
    anchor_segment_index = physical_tendons[0].anchor_segment_index
    section_tendons = tuple(
        tendon
        for tendon in physical_tendons
        if tendon.anchor_segment_index == anchor_segment_index
    )
    assert len(section_tendons) == tendons_per_segment

    control = np.zeros(config.tendon_model.count, dtype=float)
    for tendon in section_tendons:
        control[tendon.global_index] = config.smoke_tests.symmetric_tendon_delta_m

    state = backend.step(control, n_substeps=_smoke_substeps(config))
    lateral_delta = state.tip_pose[:2, 3] - initial_tip[:2]

    assert np.linalg.norm(lateral_delta) <= config.smoke_tests.zero_command_tolerance_m


def _tendon_backend() -> tuple:
    config = replace(load_mujoco_config(MUJOCO_CONFIG), control_mode="tendon_position")
    backend = MujocoBackend.from_config(config)
    initial_state = backend.reset()
    return config, backend, initial_state.tip_pose[:3, 3].copy()


def _smoke_substeps(config) -> int:
    return max(1, round(config.smoke_tests.duration_s / config.solver.timestep))
