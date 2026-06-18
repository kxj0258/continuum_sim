import importlib
from pathlib import Path

import numpy as np
import yaml
from numpy.testing import assert_allclose

from continuum_sim.actuation import load_motor_params_from_yaml
from continuum_sim.actuation.motor_mapping import (
    motor_velocity_to_tendon_velocity,
    tendon_delta_to_motor_position,
)
from continuum_sim.backends.mujoco_backend import BackendState


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MUJOCO_CONFIG = PROJECT_ROOT / "configs" / "mujoco.yaml"
TASK_CONFIG = PROJECT_ROOT / "configs" / "tasks" / "mujoco_trajectory_tracking.yaml"


def _load_tracking_runtime_module():
    module = importlib.import_module("continuum_sim.runtime.mujoco_tracking_runtime")
    return importlib.reload(module)


def test_mujoco_tracking_result_exposes_pose_histories() -> None:
    module = _load_tracking_runtime_module()

    fields = set(module.MujocoTrackingResult.__dataclass_fields__)

    assert {
        "tip_pose",
        "segment_poses",
        "error_norm",
        "motor_velocity",
        "tendon_delta",
        "q_est",
        "mujoco_control",
        "qpos",
        "qvel",
        "tendon_length",
        "tendon_velocity",
        "actuator_force",
    }.issubset(fields)
    assert callable(module.run_mujoco_trajectory_tracking)


def test_mujoco_tracking_summary_draws_with_actual_tip_history() -> None:
    module = _load_tracking_runtime_module()
    sample_count = 3
    tip_pose = np.repeat(np.eye(4, dtype=float)[None, :, :], sample_count, axis=0)
    tip_pose[:, :3, 3] = np.array(
        [
            [0.0, 0.0, 0.12],
            [0.01, 0.0, 0.11],
            [0.02, 0.0, 0.10],
        ],
        dtype=float,
    )
    result = module.MujocoTrackingResult(
        time=np.array([0.0, 0.02, 0.04], dtype=float),
        target_position=np.array(
            [
                [0.0, 0.0, 0.10],
                [0.01, 0.0, 0.10],
                [0.02, 0.0, 0.10],
            ],
            dtype=float,
        ),
        tip_pose=tip_pose,
        segment_poses=np.zeros((sample_count, 3, 4, 4), dtype=float),
        error_norm=np.array([0.02, 0.01, 0.0], dtype=float),
        motor_position=np.zeros((sample_count, 9), dtype=float),
        motor_velocity=np.zeros((sample_count, 9), dtype=float),
        tendon_delta=np.zeros((sample_count, 9), dtype=float),
        q_est=np.zeros((sample_count, 9), dtype=float),
        joint_targets=np.zeros((sample_count, 24), dtype=float),
        mujoco_control=np.zeros((sample_count, 24), dtype=float),
        qpos=np.zeros((sample_count, 24), dtype=float),
        qvel=np.zeros((sample_count, 24), dtype=float),
        mocap_pos=np.zeros((sample_count, 0, 3), dtype=float),
        mocap_quat=np.zeros((sample_count, 0, 4), dtype=float),
        tendon_length=np.zeros((sample_count, 0), dtype=float),
        tendon_velocity=np.zeros((sample_count, 0), dtype=float),
        actuator_force=np.zeros((sample_count, 24), dtype=float),
        scene_xml_path=MUJOCO_CONFIG,
    )

    module._show_mujoco_tracking_summary(result, TASK_CONFIG, show=False)


def test_mujoco_control_substeps_aligns_controller_dt_to_timestep() -> None:
    module = _load_tracking_runtime_module()

    assert module.compute_mujoco_control_substeps(0.02, 0.001) == 20
    assert module.compute_mujoco_control_substeps(0.0004, 0.001) == 1


def test_time_advanced_target_index_spans_trajectory_over_max_steps() -> None:
    module = _load_tracking_runtime_module()

    indices = [
        module._time_advanced_target_index(step, target_count=500, max_steps=2000)
        for step in (0, 3, 4, 1998, 1999)
    ]

    assert indices == [0, 0, 1, 499, 499]
    assert module._target_advance_mode("time") == "time"
    assert module._target_advance_mode("tolerance") == "tolerance"


def test_viewer_keyboard_state_updates_pause_speed_and_replay() -> None:
    module = _load_tracking_runtime_module()
    control = module.ViewerControlState()

    module._handle_viewer_key(control, ord(" "))
    assert control.paused is True

    module._handle_viewer_key(control, ord("."))
    assert control.paused is True
    assert control.step_once is True

    module._handle_viewer_key(control, ord("+"))
    assert control.speed > 1.0
    module._handle_viewer_key(control, ord("-"))
    assert np.isclose(control.speed, 1.0)

    module._handle_viewer_key(control, ord("r"))
    assert control.replay_requested is True
    assert control.replay_index == 0


def test_trail_helpers_stride_limit_and_skip_degenerate_segments() -> None:
    module = _load_tracking_runtime_module()
    trail: list[np.ndarray] = []
    for index in range(6):
        module._append_trail_sample(
            trail,
            np.array([float(index), 0.0, 0.0]),
            index,
            stride=2,
            max_points=2,
        )

    assert [point[0] for point in trail] == [2.0, 4.0]

    points = [
        np.array([0.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 0.0]),
        np.array([1.0, 0.0, 0.0]),
        np.array([2.0, 0.0, 0.0]),
        np.array([3.0, 0.0, 0.0]),
    ]
    segments = module._select_trail_segments(points, max_segments=2)

    assert len(segments) == 2
    assert np.allclose(segments[0][0], [1.0, 0.0, 0.0])
    assert np.allclose(segments[1][1], [3.0, 0.0, 0.0])


def test_capsule_connector_supports_python_and_legacy_mujoco_signatures() -> None:
    module = _load_tracking_runtime_module()

    class GeomType:
        mjGEOM_CAPSULE = 3

    class PythonConnectorApi:
        mjtGeom = GeomType

        def __init__(self) -> None:
            self.calls = []

        def mjv_connector(self, geom, geom_type, width, from_, to) -> None:
            self.calls.append((geom, geom_type, width, from_, to))

    class LegacyConnectorApi:
        mjtGeom = GeomType

        def __init__(self) -> None:
            self.calls = []

        def mjv_makeConnector(self, *args) -> None:
            self.calls.append(args)

    start = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    end = np.array([4.0, 5.0, 6.0], dtype=np.float32)

    python_api = PythonConnectorApi()
    module._connect_capsule_geom(python_api, "geom", 0.0012, start, end)

    assert len(python_api.calls) == 1
    _, geom_type, width, from_point, to_point = python_api.calls[0]
    assert geom_type == GeomType.mjGEOM_CAPSULE
    assert width == 0.0012
    assert from_point.dtype == np.float64
    assert to_point.dtype == np.float64
    np.testing.assert_allclose(from_point, [1.0, 2.0, 3.0])
    np.testing.assert_allclose(to_point, [4.0, 5.0, 6.0])

    legacy_api = LegacyConnectorApi()
    module._connect_capsule_geom(legacy_api, "geom", 0.0012, start, end)

    assert legacy_api.calls == [
        (
            "geom",
            GeomType.mjGEOM_CAPSULE,
            0.0012,
            1.0,
            2.0,
            3.0,
            4.0,
            5.0,
            6.0,
        )
    ]


def test_mujoco_tracking_tendon_mode_uses_actual_observation_feedback(
    monkeypatch,
    tmp_path: Path,
) -> None:
    module = _load_tracking_runtime_module()
    task_config_path = _write_task_config(
        tmp_path,
        max_steps=2,
        feedback_mode="mujoco_actual",
    )
    mujoco_config_path = _write_mujoco_config(
        tmp_path,
        use_segment_visuals=False,
        control_mode="tendon_position",
    )
    state0 = _make_backend_state(
        tip_position=np.array([0.0, 0.0, 0.12], dtype=float),
        tendon_length=np.zeros(9, dtype=float),
        nu=9,
    )
    state1 = _make_backend_state(
        tip_position=np.array([0.001, 0.0, 0.119], dtype=float),
        tendon_length=np.full(9, 0.003, dtype=float),
        nu=9,
    )
    state2 = _make_backend_state(
        tip_position=np.array([0.0015, 0.0, 0.1185], dtype=float),
        tendon_length=np.full(9, 0.004, dtype=float),
        nu=9,
    )
    backend = _FakeTrackingBackend([state0, state1, state2])
    motor_params = load_motor_params_from_yaml(PROJECT_ROOT / "configs" / "robot_3seg.yaml")
    motor_velocity_commands = [
        np.full(9, 0.1, dtype=float),
        np.full(9, -0.05, dtype=float),
    ]
    observed_calls: list[tuple[np.ndarray, np.ndarray]] = []

    class BackendFactory:
        @staticmethod
        def from_config(config, *, override_xml_path=None):
            return backend

    def fake_observation_helper(
        actual_tip_position,
        actual_tendon_delta,
        target_position,
        params,
        physical_tendons,
        motor_params_arg,
        config,
    ):
        index = len(observed_calls)
        observed_calls.append(
            (
                np.asarray(actual_tip_position, dtype=float).copy(),
                np.asarray(actual_tendon_delta, dtype=float).copy(),
            )
        )
        return (
            motor_velocity_commands[index].copy(),
            {
                "q_est": np.full(9, float(index + 1), dtype=float),
                "tip_position": np.asarray(actual_tip_position, dtype=float).copy(),
                "position_error": np.asarray(target_position, dtype=float)
                - np.asarray(actual_tip_position, dtype=float),
                "error_norm": float(
                    np.linalg.norm(
                        np.asarray(target_position, dtype=float)
                        - np.asarray(actual_tip_position, dtype=float)
                    )
                ),
                "desired_tip_velocity": np.zeros(3, dtype=float),
                "J_motor": np.zeros((3, 9), dtype=float),
            },
        )

    monkeypatch.setattr(module, "MujocoBackend", BackendFactory)
    monkeypatch.setattr(
        module,
        "build_target_positions",
        lambda config, params: np.array([[0.0, 0.0, 0.12]], dtype=float),
    )
    monkeypatch.setattr(
        module,
        "compute_motor_velocity_command_from_observation",
        fake_observation_helper,
    )
    monkeypatch.setattr(
        module,
        "compute_motor_velocity_command",
        lambda *args, **kwargs: (_unexpected_legacy_controller_call()),
    )

    result = module.run_mujoco_trajectory_tracking(
        task_config_path,
        mujoco_config_path,
        show=False,
    )

    assert len(observed_calls) == 2
    assert_allclose(observed_calls[0][0], state0.tip_pose[:3, 3])
    assert_allclose(observed_calls[0][1], state0.tendon_length)
    assert_allclose(observed_calls[1][0], state1.tip_pose[:3, 3])
    assert_allclose(observed_calls[1][1], state1.tendon_length)
    assert_allclose(result.tendon_delta[0], state0.tendon_length)
    assert_allclose(result.tendon_delta[1], state1.tendon_length)
    assert_allclose(
        result.motor_position[1],
        tendon_delta_to_motor_position(state1.tendon_length, motor_params),
    )
    expected_tendon_velocity_0 = motor_velocity_to_tendon_velocity(
        motor_velocity_commands[0],
        motor_params,
    )
    expected_tendon_velocity_1 = motor_velocity_to_tendon_velocity(
        motor_velocity_commands[1],
        motor_params,
    )
    assert_allclose(
        backend.controls[0],
        state0.tendon_length + 0.02 * expected_tendon_velocity_0,
    )
    assert_allclose(
        result.mujoco_control[1],
        state1.tendon_length + 0.02 * expected_tendon_velocity_1,
    )


def test_mujoco_tracking_position_joint_mode_keeps_legacy_controller_path(
    monkeypatch,
    tmp_path: Path,
) -> None:
    module = _load_tracking_runtime_module()
    task_config_path = _write_task_config(
        tmp_path,
        max_steps=1,
        feedback_mode="mujoco_actual",
    )
    mujoco_config_path = _write_mujoco_config(
        tmp_path,
        use_segment_visuals=False,
        control_mode="position_joint",
    )
    state0 = _make_backend_state(
        tip_position=np.array([0.0, 0.0, 0.12], dtype=float),
        tendon_length=np.zeros(0, dtype=float),
        nu=24,
    )
    state1 = _make_backend_state(
        tip_position=np.array([0.0, 0.0, 0.12], dtype=float),
        tendon_length=np.zeros(0, dtype=float),
        nu=24,
    )
    backend = _FakeTrackingBackend([state0, state1])
    legacy_calls: list[np.ndarray] = []
    joint_targets = np.linspace(-0.2, 0.2, 24, dtype=float)

    class BackendFactory:
        @staticmethod
        def from_config(config, *, override_xml_path=None):
            return backend

    def fake_legacy_helper(
        motor_position,
        target_position,
        params,
        physical_tendons,
        motor_params,
        config,
    ):
        legacy_calls.append(np.asarray(motor_position, dtype=float).copy())
        return (
            np.zeros(9, dtype=float),
            {
                "q_est": np.zeros(9, dtype=float),
                "tip_position": np.asarray(target_position, dtype=float).copy(),
                "position_error": np.zeros(3, dtype=float),
                "error_norm": 0.0,
                "desired_tip_velocity": np.zeros(3, dtype=float),
                "J_motor": np.zeros((3, 9), dtype=float),
            },
        )

    monkeypatch.setattr(module, "MujocoBackend", BackendFactory)
    monkeypatch.setattr(
        module,
        "build_target_positions",
        lambda config, params: np.array([[0.0, 0.0, 0.12]], dtype=float),
    )
    monkeypatch.setattr(module, "compute_motor_velocity_command", fake_legacy_helper)
    monkeypatch.setattr(
        module,
        "compute_motor_velocity_command_from_observation",
        lambda *args, **kwargs: (_unexpected_observation_controller_call()),
    )
    monkeypatch.setattr(module, "pcc_q_to_joint_targets", lambda *args, **kwargs: joint_targets)

    result = module.run_mujoco_trajectory_tracking(
        task_config_path,
        mujoco_config_path,
        show=False,
    )

    assert len(legacy_calls) == 1
    assert_allclose(result.joint_targets[0], joint_targets)
    assert_allclose(result.mujoco_control[0], joint_targets)


def test_create_tendon_live_panel_returns_none_when_disabled() -> None:
    module = _load_tracking_runtime_module()
    panel = module._create_tendon_live_panel(
        show_viewer=False,
        control_mode="tendon_position",
        show_live_tendon_panel=True,
        config=object(),
        params=object(),
        physical_tendons=(),
        initial_state=_make_backend_state(
            tip_position=np.array([0.0, 0.0, 0.12], dtype=float),
            tendon_length=np.zeros(9, dtype=float),
            nu=9,
        ),
    )

    assert panel is None


def test_sync_tendon_live_panel_uses_mujoco_control_and_observed_state() -> None:
    module = _load_tracking_runtime_module()
    panel = _FakeTendonMonitorPanel()
    command = np.linspace(-0.004, 0.004, 9, dtype=float)
    state = _make_backend_state(
        tip_position=np.array([0.001, -0.002, 0.119], dtype=float),
        tendon_length=np.linspace(-0.003, 0.005, 9, dtype=float),
        nu=9,
    )
    state = BackendState(
        time=0.02,
        tip_pose=state.tip_pose,
        segment_poses=state.segment_poses,
        qpos=state.qpos,
        qvel=state.qvel,
        tendon_length=state.tendon_length,
        tendon_velocity=np.zeros(9, dtype=float),
        actuator_force=np.linspace(0.0, 0.9, 9, dtype=float),
    )

    module._sync_tendon_live_panel(panel, 2, command, state, sample_index=4)

    assert len(panel.update_calls) == 1
    assert_allclose(panel.update_calls[0][0], command)
    assert_allclose(panel.update_calls[0][1].tendon_length, state.tendon_length)
    assert_allclose(panel.update_calls[0][1].actuator_force, state.actuator_force)
    assert panel.flush_count == 1


def test_update_tendon_live_panel_from_history_uses_recorded_force_and_length() -> None:
    module = _load_tracking_runtime_module()
    panel = _FakeTendonMonitorPanel()
    command_history = [
        np.zeros(9, dtype=float),
        np.linspace(-0.003, 0.003, 9, dtype=float),
    ]
    tip_pose_history = np.repeat(np.eye(4, dtype=float)[None, :, :], 2, axis=0)
    tip_pose_history[1, :3, 3] = np.array([0.01, 0.0, 0.11], dtype=float)
    qpos_history = np.zeros((2, 24), dtype=float)
    qvel_history = np.zeros((2, 24), dtype=float)
    tendon_length_history = [
        np.zeros(9, dtype=float),
        np.linspace(-0.002, 0.004, 9, dtype=float),
    ]
    actuator_force_history = [
        np.zeros(9, dtype=float),
        np.linspace(0.1, 0.9, 9, dtype=float),
    ]

    module._update_tendon_live_panel_from_history(
        panel,
        index=1,
        controller_dt=0.02,
        mujoco_control_history=command_history,
        tip_pose_history=tip_pose_history,
        qpos_history=qpos_history,
        qvel_history=qvel_history,
        tendon_length_history=tendon_length_history,
        actuator_force_history=actuator_force_history,
    )

    assert len(panel.update_calls) == 1
    command, state, redraw = panel.update_calls[0]
    assert redraw is True
    assert_allclose(command, command_history[1])
    assert_allclose(state.tendon_length, tendon_length_history[1])
    assert_allclose(state.actuator_force, actuator_force_history[1])
    assert state.time == 0.02
    assert panel.flush_count == 1


def test_maybe_hold_tracking_viewer_open_after_run_only_when_enabled(
    monkeypatch,
) -> None:
    module = _load_tracking_runtime_module()
    idle_calls: list[dict[str, object]] = []

    def fake_idle_viewer(*args, **kwargs) -> None:
        idle_calls.append({"args": args, "kwargs": kwargs})

    monkeypatch.setattr(module, "_idle_tracking_viewer", fake_idle_viewer)

    kwargs = dict(
        viewer=object(),
        backend=object(),
        mujoco_module=object(),
        control=module.ViewerControlState(),
        tendon_overlay=None,
        tendon_monitor_panel=None,
        overlay_config=object(),
        target_history=[np.zeros(3, dtype=float)],
        tip_pose_history=[np.eye(4, dtype=float)],
        qpos_history=[np.zeros(24, dtype=float)],
        qvel_history=[np.zeros(24, dtype=float)],
        mujoco_control_history=[np.zeros(9, dtype=float)],
        tendon_length_history=[np.zeros(9, dtype=float)],
        actuator_force_history=[np.zeros(9, dtype=float)],
        controller_dt=0.02,
        realtime=True,
        realtime_factor=1.0,
    )

    module._maybe_hold_tracking_viewer_open_after_run(
        keep_viewer_open_after_run=False,
        **kwargs,
    )
    assert idle_calls == []

    module._maybe_hold_tracking_viewer_open_after_run(
        keep_viewer_open_after_run=True,
        **kwargs,
    )
    assert len(idle_calls) == 1


def _write_mujoco_config(
    tmp_path: Path,
    *,
    use_segment_visuals: bool,
    control_mode: str = "tendon_position",
) -> Path:
    raw = yaml.safe_load(MUJOCO_CONFIG.read_text(encoding="utf-8"))
    raw["robot_config_path"] = str(PROJECT_ROOT / "configs" / "robot_3seg.yaml")
    raw["xml_path"] = str(PROJECT_ROOT / "assets" / "mujoco" / "three_segment_arm.xml")
    raw["tendon_xml_path"] = str(
        PROJECT_ROOT / "assets" / "mujoco" / "three_segment_arm_tendon.xml"
    )
    raw["generated_xml_path"] = str(tmp_path / "missing_generated.xml")
    raw["tendon_generated_xml_path"] = str(tmp_path / "missing_tendon_generated.xml")
    raw["control_mode"] = control_mode
    raw["visuals"]["directory"] = str(
        PROJECT_ROOT / "assets" / "meshes" / "mujoco_visual_segments"
    )
    raw["visuals"]["template_path"] = str(
        PROJECT_ROOT / "assets" / "mujoco" / "segmented_visuals_template.xml"
    )
    raw["viewer"]["show"] = False
    raw["viewer"]["steps"] = 2
    raw["viewer"]["use_segment_visuals"] = use_segment_visuals
    config_path = tmp_path / "mujoco.yaml"
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return config_path


def _write_task_config(
    tmp_path: Path,
    *,
    source_config: Path = TASK_CONFIG,
    max_steps: int = 3,
    feedback_mode: str = "mujoco_actual",
    show_live_tendon_panel: bool = True,
    hold_viewer_open_after_run: bool = False,
) -> Path:
    raw = yaml.safe_load(source_config.read_text(encoding="utf-8"))
    raw["robot"]["config_path"] = str(PROJECT_ROOT / "configs" / "robot_3seg.yaml")
    raw["simulation"]["max_steps"] = max_steps
    raw["simulation"]["stop_on_completion"] = False
    raw["trajectory"]["samples"] = 3
    if "mujoco" in raw:
        raw["mujoco"]["feedback_mode"] = feedback_mode
        raw["mujoco"]["show_live_tendon_panel"] = show_live_tendon_panel
        raw["mujoco"]["hold_viewer_open_after_run"] = hold_viewer_open_after_run
    config_path = tmp_path / "tracking.yaml"
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return config_path


class _FakeTrackingBackend:
    def __init__(self, states: list[BackendState]) -> None:
        self._states = states
        self._index = 0
        self.controls: list[np.ndarray] = []
        self.model = object()
        self.data = object()

    def reset(self) -> BackendState:
        self._index = 0
        return self._states[0]

    def step(self, control: np.ndarray, n_substeps: int = 20) -> BackendState:
        self.controls.append(np.asarray(control, dtype=float).copy())
        self._index = min(self._index + 1, len(self._states) - 1)
        return self._states[self._index]


class _FakeTendonMonitorPanel:
    def __init__(self) -> None:
        self.update_calls: list[tuple[np.ndarray, BackendState, bool]] = []
        self.flush_count = 0

    def update_from_state(
        self,
        commanded_tendon_delta: np.ndarray,
        state: BackendState,
        *,
        redraw: bool = True,
    ) -> None:
        self.update_calls.append(
            (
                np.asarray(commanded_tendon_delta, dtype=float).copy(),
                state,
                bool(redraw),
            )
        )

    def flush_events(self) -> None:
        self.flush_count += 1


def _make_backend_state(
    *,
    tip_position: np.ndarray,
    tendon_length: np.ndarray,
    nu: int,
) -> BackendState:
    tip_pose = np.eye(4, dtype=float)
    tip_pose[:3, 3] = np.asarray(tip_position, dtype=float)
    return BackendState(
        time=0.0,
        tip_pose=tip_pose,
        segment_poses=np.repeat(np.eye(4, dtype=float)[None, :, :], 3, axis=0),
        qpos=np.zeros(24, dtype=float),
        qvel=np.zeros(24, dtype=float),
        tendon_length=np.asarray(tendon_length, dtype=float).copy(),
        tendon_velocity=np.zeros_like(np.asarray(tendon_length, dtype=float)),
        actuator_force=np.zeros(nu, dtype=float),
    )


def _unexpected_legacy_controller_call():
    raise AssertionError("Legacy PCC command helper should not run in mujoco_actual tendon mode.")


def _unexpected_observation_controller_call():
    raise AssertionError(
        "Observation-based helper should not run in position_joint control mode."
    )
