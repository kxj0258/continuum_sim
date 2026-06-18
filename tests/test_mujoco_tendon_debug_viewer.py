from types import SimpleNamespace
from pathlib import Path

import matplotlib
import numpy as np
from numpy.testing import assert_allclose

matplotlib.use("Agg")

from continuum_sim.backends import BackendState
from continuum_sim.config import load_mujoco_config
from continuum_sim.model import ThreeSegmentRobotParams, load_physical_tendons_from_yaml
from continuum_sim.visualization.mujoco_tendon_debug_viewer import (
    MujocoTendonDebugViewer,
    MujocoTendonMonitorPanel,
    _is_noninteractive_matplotlib_backend,
    clip_tendon_command,
    compute_mujoco_tendon_debug_view_data,
    named_tendon_command,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MUJOCO_CONFIG = PROJECT_ROOT / "configs" / "mujoco.yaml"
ROBOT_CONFIG = PROJECT_ROOT / "configs" / "robot_3seg.yaml"


def _robot_and_mujoco_config():
    config = load_mujoco_config(MUJOCO_CONFIG)
    params = ThreeSegmentRobotParams.from_yaml(ROBOT_CONFIG)
    physical_tendons = load_physical_tendons_from_yaml(ROBOT_CONFIG)
    return config, params, physical_tendons


def test_compute_mujoco_tendon_debug_view_data_uses_actual_state_readback() -> None:
    _config, params, physical_tendons = _robot_and_mujoco_config()
    command = np.linspace(-0.002, 0.002, 9, dtype=float)
    actual_tendon_length = np.linspace(-0.001, 0.003, 9, dtype=float)
    actuator_force = np.linspace(0.0, 4.0, 9, dtype=float)
    tip_pose = np.eye(4, dtype=float)
    tip_pose[:3, 3] = np.array([0.01, -0.02, 0.11], dtype=float)
    state = BackendState(
        time=0.123,
        tip_pose=tip_pose,
        segment_poses=np.repeat(np.eye(4, dtype=float)[None, :, :], 3, axis=0),
        qpos=np.zeros(24, dtype=float),
        qvel=np.zeros(24, dtype=float),
        tendon_length=actual_tendon_length,
        tendon_velocity=np.zeros(9, dtype=float),
        actuator_force=actuator_force,
    )

    view_data = compute_mujoco_tendon_debug_view_data(
        command,
        state,
        params,
        physical_tendons,
    )

    assert view_data.time_s == 0.123
    assert_allclose(view_data.commanded_tendon_delta, command)
    assert_allclose(view_data.actual_tendon_length, actual_tendon_length)
    assert_allclose(view_data.actuator_force, actuator_force)
    assert_allclose(view_data.tip_position, [0.01, -0.02, 0.11])
    assert_allclose(view_data.tendon_error, command - actual_tendon_length)
    assert view_data.q_est.shape == (9,)


def test_named_tendon_command_uses_existing_mujoco_config_ranges() -> None:
    config, _params, _physical_tendons = _robot_and_mujoco_config()

    single = named_tendon_command("tendon_4_pull", config)
    segment = named_tendon_command("segment_3_triplet", config)

    assert single.shape == (9,)
    assert single[3] == config.smoke_tests.single_tendon_delta_m
    assert np.count_nonzero(single) == 1
    assert_allclose(segment[6:9], np.full(3, config.smoke_tests.symmetric_tendon_delta_m))
    assert_allclose(
        clip_tendon_command(np.full(9, 1.0, dtype=float), config.actuators.tendon_position.ctrlrange_m),
        np.full(9, config.actuators.tendon_position.ctrlrange_m[1], dtype=float),
    )


def test_mujoco_tendon_debug_viewer_runs_without_gui_display() -> None:
    config, params, physical_tendons = _robot_and_mujoco_config()
    backend = _FakeTendonBackend(config.tendon_model.count)
    callback_times: list[float] = []
    viewer = MujocoTendonDebugViewer(
        backend,
        config,
        params,
        physical_tendons,
        control_dt=0.02,
        state_update_callback=lambda state: callback_times.append(float(state.time)),
    )
    try:
        zero_data = viewer.update_view(redraw=False)
        assert_allclose(zero_data.commanded_tendon_delta, np.zeros(9), atol=1.0e-12)
        assert zero_data.tip_position.shape == (3,)

        command = np.zeros(9, dtype=float)
        command[0] = config.smoke_tests.single_tendon_delta_m
        command_data = viewer.set_command(command, simulate=True)

        assert backend.last_n_substeps == 20
        assert_allclose(command_data.commanded_tendon_delta, command)
        assert_allclose(command_data.actual_tendon_length, 0.8 * command)
        assert np.linalg.norm(command_data.actuator_force) > 0.0
        force_ylim = viewer.force_ax.get_ylim()
        assert force_ylim[1] < config.actuators.tendon_position.forcerange_n[1]
        assert force_ylim[1] > float(np.max(command_data.actuator_force))
        assert len(callback_times) >= 2

        named_data = viewer.apply_named_command("segment_2_triplet")
        assert_allclose(
            named_data.commanded_tendon_delta[3:6],
            np.full(3, config.smoke_tests.symmetric_tendon_delta_m),
        )

        reset_data = viewer.reset()
        assert_allclose(reset_data.commanded_tendon_delta, np.zeros(9), atol=1.0e-12)
        assert_allclose(reset_data.actual_tendon_length, np.zeros(9), atol=1.0e-12)
    finally:
        viewer.close()


def test_mujoco_tendon_monitor_panel_updates_from_state_without_controls() -> None:
    config, params, physical_tendons = _robot_and_mujoco_config()
    panel = MujocoTendonMonitorPanel(config, params, physical_tendons)
    command = np.zeros(9, dtype=float)
    command[0] = config.smoke_tests.single_tendon_delta_m
    state = _make_state(
        0.02,
        0.7 * command,
        np.linspace(0.0, 0.6, 9, dtype=float),
    )
    try:
        view_data = panel.update_from_state(command, state, redraw=False)
        panel.show(block=False)
        panel.flush_events()

        assert_allclose(view_data.commanded_tendon_delta, command)
        assert_allclose(view_data.actual_tendon_length, 0.7 * command)
        assert_allclose(view_data.actuator_force, np.linspace(0.0, 0.6, 9, dtype=float))
    finally:
        panel.close()


def test_noninteractive_backend_detection_distinguishes_qtagg_from_agg() -> None:
    assert _is_noninteractive_matplotlib_backend("agg")
    assert _is_noninteractive_matplotlib_backend("module://matplotlib_inline.backend_inline")
    assert not _is_noninteractive_matplotlib_backend("qtagg")
    assert not _is_noninteractive_matplotlib_backend("Qt5Agg")


def test_mujoco_tendon_monitor_panel_show_uses_interactive_qtagg_backend(
    monkeypatch,
) -> None:
    config, params, physical_tendons = _robot_and_mujoco_config()
    panel = MujocoTendonMonitorPanel(config, params, physical_tendons)
    show_calls: list[bool] = []
    flush_calls: list[str] = []
    try:
        monkeypatch.setattr(matplotlib.pyplot, "get_backend", lambda: "qtagg")
        monkeypatch.setattr(
            matplotlib.pyplot,
            "show",
            lambda *, block: show_calls.append(bool(block)),
        )
        monkeypatch.setattr(panel, "flush_events", lambda: flush_calls.append("flush"))

        panel.show(block=False)

        assert show_calls == [False]
        assert flush_calls == ["flush"]
    finally:
        panel.close()


def test_mujoco_tendon_monitor_panel_show_skips_noninteractive_agg_backend(
    monkeypatch,
) -> None:
    config, params, physical_tendons = _robot_and_mujoco_config()
    panel = MujocoTendonMonitorPanel(config, params, physical_tendons)
    show_calls: list[bool] = []
    try:
        monkeypatch.setattr(matplotlib.pyplot, "get_backend", lambda: "agg")
        monkeypatch.setattr(
            matplotlib.pyplot,
            "show",
            lambda *, block: show_calls.append(bool(block)),
        )

        panel.show(block=False)

        assert show_calls == []
    finally:
        panel.close()


class _FakeTendonBackend:
    def __init__(self, tendon_count: int) -> None:
        self.tendon_count = tendon_count
        self.model = SimpleNamespace(nu=tendon_count)
        self.last_n_substeps = 0
        self._time = 0.0

    def reset(self) -> BackendState:
        self._time = 0.0
        return _make_state(
            self._time,
            np.zeros(self.tendon_count, dtype=float),
            np.zeros(self.tendon_count, dtype=float),
        )

    def step(self, control: np.ndarray, n_substeps: int = 20) -> BackendState:
        control_array = np.asarray(control, dtype=float)
        self.last_n_substeps = n_substeps
        self._time += 0.001 * n_substeps
        actual_tendon = 0.8 * control_array
        actuator_force = 1200.0 * np.abs(control_array)
        return _make_state(self._time, actual_tendon, actuator_force)


def _make_state(
    time_s: float,
    tendon_length: np.ndarray,
    actuator_force: np.ndarray,
) -> BackendState:
    tip_pose = np.eye(4, dtype=float)
    lateral = float(np.sum(tendon_length[0:3]) - np.sum(tendon_length[6:9]))
    tip_pose[:3, 3] = np.array([8.0 * lateral, 0.0, 0.12 - 2.0 * abs(lateral)], dtype=float)
    return BackendState(
        time=time_s,
        tip_pose=tip_pose,
        segment_poses=np.repeat(np.eye(4, dtype=float)[None, :, :], 3, axis=0),
        qpos=np.zeros(24, dtype=float),
        qvel=np.zeros(24, dtype=float),
        tendon_length=np.asarray(tendon_length, dtype=float).copy(),
        tendon_velocity=np.zeros_like(np.asarray(tendon_length, dtype=float)),
        actuator_force=np.asarray(actuator_force, dtype=float).copy(),
    )
