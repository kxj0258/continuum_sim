from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np

import continuum_sim.io.scenario_artifacts as scenario_artifacts
from continuum_sim.application.scenario import (
    ScenarioArtifactConfig,
    ScenarioBackendConfig,
    ScenarioConfig,
    ScenarioHookConfig,
    ScenarioRuntimeConfig,
    ScenarioSceneConfig,
    ScenarioTaskConfig,
)
from continuum_sim.model.base_pose import Pose6D
from continuum_sim.runtime.simulation_loop import SimulationLoopResult
from continuum_sim.system.types import (
    ArmSystemState,
    ArmTendonRateCommand,
    BaseSystemState,
    RobotSystemCommand,
    RobotSystemState,
)


def test_scenario_artifacts_keep_npz_metadata_and_plots_when_gif_fails(
    tmp_path,
    monkeypatch,
) -> None:
    scenario_path = tmp_path / "scenario.yaml"
    assembly_path = tmp_path / "assembly.yaml"
    scenario_path.write_text("schema_version: 1\n", encoding="utf-8")
    assembly_path.write_text("schema_version: 1\n", encoding="utf-8")
    config = ScenarioConfig(
        path=scenario_path,
        name="gif_failure",
        assembly_config_path=assembly_path,
        backend=ScenarioBackendConfig(type="analytic"),
        scene=ScenarioSceneConfig(),
        task=ScenarioTaskConfig(
            type="tracking",
            waypoints_world=np.zeros((1, 3), dtype=float),
            waypoint_tolerance_m=0.001,
            observer_roi_world=None,
            loop=False,
            min_clearance_m=0.01,
            terminate_on_clearance_violation=True,
            surface_normal_world=np.array([0.0, 0.0, 1.0], dtype=float),
            target_contact_distance_m=0.0,
            contact_tolerance_m=0.002,
        ),
        runtime=ScenarioRuntimeConfig(
            controller_dt_s=0.02,
            n_substeps=1,
            max_steps=1,
        ),
        hooks=ScenarioHookConfig(
            recorder=True,
            tendon_debug=False,
            tendon_debug_stride=1,
            viewer="none",
            keep_viewer_open=False,
        ),
        artifacts=ScenarioArtifactConfig(
            enabled=True,
            output_root=tmp_path / "runs",
            save_npz=True,
            save_plots=True,
            save_gif=True,
            save_model=True,
            video_fps=20,
            video_stride=10,
        ),
    )
    state0 = _state(0.0, [0.0, 0.0, 0.10])
    state1 = _state(
        0.02,
        [0.01, 0.0, 0.10],
        tendon_displacement_m=np.full(9, 0.00002, dtype=float),
        tendon_velocity_mps=np.full(9, 0.001, dtype=float),
        tendon_target_m=np.full(9, 0.00004, dtype=float),
        saturation={
            "inner_loop_mode": "bending_rate_servo",
            "target_mode": "bending_rate_servo",
            "constrained_rate_mps": np.full(9, 0.0015, dtype=float),
            "measured_rate_mps": np.full(9, 0.0008, dtype=float),
            "guard_feasible": True,
        },
    )
    result = SimulationLoopResult(
        states=(state0, state1),
        commands=(
            RobotSystemCommand(
                base_twist_world=np.zeros(6, dtype=float),
                arms={
                    "executor": ArmTendonRateCommand(
                        np.full(9, 0.0018, dtype=float)
                    )
                },
                metadata={
                    "task_intent_control_mode": "position",
                    "task_intent_target_world": np.array(
                        [0.01, 0.0, 0.10], dtype=float
                    ),
                    "executor_feedforward_gain": 0.25,
                    "task_intent_velocity_world": np.array(
                        [0.004, 0.0, 0.0], dtype=float
                    ),
                    "task_space_position_error_world": np.array(
                        [0.001, 0.0, 0.0], dtype=float
                    ),
                    "task_space_raw_velocity_world": np.array(
                        [0.002, 0.0, 0.0], dtype=float
                    ),
                    "task_space_velocity_world": np.array(
                        [0.002, 0.0, 0.0], dtype=float
                    ),
                    "task_space_speed_limited": False,
                    "executor_scaled_feedforward_velocity_world": np.array(
                        [0.001, 0.0, 0.0], dtype=float
                    ),
                    "task_status_phase": "tracking",
                    "task_status_active_index": 0,
                    "task_status_complete": False,
                    "residual_norm": 0.0002,
                    "whole_body_solver": {
                        "singularity_strategy": "svd_projection",
                        "target_projection_residual_norm": 0.0001,
                    },
                },
            ),
        ),
        stopped_early=False,
    )
    recorder = SimpleNamespace(
        target_position_m=[np.array([0.01, 0.0, 0.10], dtype=float)],
        target_actual_position_m=[np.array([0.009, 0.0, 0.10], dtype=float)],
        target_engine_local_path_name=["one_third_circle"],
        target_engine_local_path_type=["transverse_circle"],
        target_engine_executor_subphase=["path"],
        target_engine_local_path_center_m=[
            np.array([0.0, 0.0, 0.10], dtype=float)
        ],
        target_engine_insertion_direction_world=[
            np.array([0.0, 0.0, 1.0], dtype=float)
        ],
        tracking_error_m=[0.0],
        achieved_waypoint_error_m=[0.0005],
        waypoint_advanced=[True],
        tracking_complete=[True],
        tracking_approach=[False],
        arm_saturation_scale={"executor": [0.75]},
        arm_tendon_target_error_norm_m={"executor": [0.0004]},
        arm_tendon_target_error_max_m={"executor": [0.0003]},
        arm_peak_actuator_force_n={"executor": [3.0]},
        waypoint_index=[0],
        min_clearance_m=[np.nan],
        contact_distance_m=[np.nan],
        contact_error_m=[np.nan],
        target_force_n=[np.nan],
        estimated_force_n=[np.nan],
        force_error_n=[np.nan],
        task_phase=["tracking"],
        engine_navigation_phase=["base_approach"],
        engine_navigation_terminal_reason=[""],
        engine_navigation_progress=[0.0],
        base_target_position_m=[np.array([0.1, 0.2, 0.3])],
        base_position_error_m=[0.1],
        base_orientation_error_rad=[0.2],
    )

    class LiveDiagnosticsPanel:
        errors: list[str] = []

        def save_snapshot(self, path):
            path.write_bytes(b"live diagnostics png")
            return path

    application = SimpleNamespace(
        config=config,
        hooks_by_name={
            "recorder": recorder,
            "live_diagnostics_panel": LiveDiagnosticsPanel(),
        },
        loop=SimpleNamespace(backend=SimpleNamespace()),
    )

    def fail_video(*args, **kwargs):
        del args, kwargs
        raise RuntimeError("gif encoder unavailable")

    monkeypatch.setattr(scenario_artifacts, "save_replay_video", fail_video)

    paths = scenario_artifacts.save_scenario_artifacts(application, result)

    assert paths is not None
    assert paths.result_npz.is_file()
    assert paths.metadata_json.is_file()
    assert (paths.plots_dir / "trajectory.png").is_file()
    assert (paths.plots_dir / "four_layer_control_diagnostics.png").is_file()
    assert (paths.plots_dir / "live_diagnostics_panel.png").is_file()
    assert (
        paths.plots_dir
        / "engine_navigation_local_path_one_third_circle.png"
    ).is_file()
    metadata = json.loads(paths.metadata_json.read_text(encoding="utf-8"))
    assert metadata["video"] is None
    assert metadata["errors"] == ["video: RuntimeError: gif encoder unavailable"]
    assert any(
        path.endswith("plots/live_diagnostics_panel.png")
        for path in metadata["plots"]
    )
    assert metadata["metrics"]["final_achieved_waypoint_error_m"] == 0.0005
    assert (
        metadata["metrics"]["control_layers"]["layer2_task_space_servo"][
            "final_position_error_m"
        ]
        == 0.001
    )
    with np.load(paths.result_npz) as arrays:
        assert arrays["arm_executor_saturation_scale"].tolist() == [0.75]
        assert arrays["arm_executor_peak_actuator_force_n"].tolist() == [3.0]
        assert arrays["engine_navigation_phase"].tolist() == ["base_approach"]
        assert arrays["base_target_position_m"].shape == (1, 3)
        assert arrays["base_position_error_m"].tolist() == [0.1]
        assert arrays["target_actual_position_m"].shape == (1, 3)
        assert arrays["target_engine_local_path_name"].tolist() == [
            "one_third_circle"
        ]
        assert arrays["executor_feedforward_gain"].tolist() == [0.25]
        assert arrays["task_intent_velocity_world"].tolist() == [
            [0.004, 0.0, 0.0]
        ]
        assert arrays["layer1_task_control_mode"].tolist() == ["position"]
        assert arrays["layer1_task_phase"].tolist() == ["tracking"]
        np.testing.assert_allclose(
            arrays["layer1_task_target_position_world"],
            [[0.01, 0.0, 0.10]],
        )
        np.testing.assert_allclose(
            arrays["layer2_servo_position_error_world"],
            [[0.001, 0.0, 0.0]],
        )
        np.testing.assert_allclose(
            arrays["layer2_servo_velocity_world"],
            [[0.002, 0.0, 0.0]],
        )
        assert arrays["layer2_servo_speed_limited"].tolist() == [False]
        np.testing.assert_allclose(
            arrays["layer3_ik_arm_executor_tendon_rate_ref_mps"],
            np.full((1, 9), 0.0018),
        )
        assert arrays[
            "executor_scaled_feedforward_velocity_world"
        ].tolist() == [[0.001, 0.0, 0.0]]
        np.testing.assert_allclose(
            arrays["arm_executor_constrained_command_rate_mps"],
            np.full((1, 9), 0.0015),
        )
        np.testing.assert_allclose(
            arrays["arm_executor_tendon_target_rate_fd_mps"],
            np.full((1, 9), 0.002),
        )
        np.testing.assert_allclose(
            arrays["arm_executor_tendon_realized_rate_fd_mps"],
            np.full((1, 9), 0.001),
        )
        np.testing.assert_allclose(
            arrays["arm_executor_tendon_velocity_sensor_raw_mps"],
            np.vstack((np.zeros(9), np.full(9, 0.001))),
        )
        np.testing.assert_allclose(
            arrays["arm_executor_tendon_target_lead_m"],
            np.full((1, 9), 0.00002),
        )
        np.testing.assert_allclose(
            arrays["arm_executor_tendon_measured_rate_filtered_mps"],
            np.full((1, 9), 0.0008),
        )
        assert arrays["arm_executor_tendon_inner_loop_mode"].tolist() == [
            "bending_rate_servo"
        ]
        np.testing.assert_allclose(
            arrays["layer4_execution_arm_executor_applied_rate_mps"],
            np.full((1, 9), 0.0015),
        )
        np.testing.assert_allclose(
            arrays["layer4_execution_arm_executor_realized_rate_mps"],
            np.full((1, 9), 0.001),
        )
        np.testing.assert_allclose(
            arrays["layer4_execution_arm_executor_tendon_position_error_norm_m"],
            [0.0, 0.00006],
        )
        assert arrays["arm_executor_tendon_servo_evaluated"].tolist() == [True]
        assert arrays["arm_executor_tendon_guard_feasible"].tolist() == [True]


def test_scenario_artifacts_collects_observer_camera_videos_separately(
    tmp_path,
) -> None:
    scenario_path = tmp_path / "scenario.yaml"
    assembly_path = tmp_path / "assembly.yaml"
    scenario_path.write_text("schema_version: 1\n", encoding="utf-8")
    assembly_path.write_text("schema_version: 1\n", encoding="utf-8")
    pending_main_gif = tmp_path / "pending" / "main.gif"
    pending_main_mp4 = tmp_path / "pending" / "main.mp4"
    pending_observer_gif = tmp_path / "pending" / "observer_eye_camera.gif"
    pending_observer_mp4 = tmp_path / "pending" / "observer_eye_camera.mp4"
    for path in (
        pending_main_gif,
        pending_main_mp4,
        pending_observer_gif,
        pending_observer_mp4,
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(path.name.encode("ascii"))
    config = ScenarioConfig(
        path=scenario_path,
        name="observer_camera_video",
        assembly_config_path=assembly_path,
        backend=ScenarioBackendConfig(type="mujoco"),
        scene=ScenarioSceneConfig(),
        task=ScenarioTaskConfig(
            type="tracking",
            waypoints_world=np.zeros((1, 3), dtype=float),
            waypoint_tolerance_m=0.001,
            observer_roi_world=None,
            loop=False,
            min_clearance_m=0.01,
            terminate_on_clearance_violation=True,
            surface_normal_world=np.array([0.0, 0.0, 1.0], dtype=float),
            target_contact_distance_m=0.0,
            contact_tolerance_m=0.002,
        ),
        runtime=ScenarioRuntimeConfig(
            controller_dt_s=0.02,
            n_substeps=1,
            max_steps=1,
        ),
        hooks=ScenarioHookConfig(
            recorder=True,
            tendon_debug=False,
            tendon_debug_stride=1,
            viewer="none",
            keep_viewer_open=False,
        ),
        artifacts=ScenarioArtifactConfig(
            enabled=True,
            output_root=tmp_path / "runs",
            save_npz=False,
            save_plots=False,
            save_gif=True,
            save_mp4=True,
            save_model=False,
            save_mujoco_pcc_diagnostics=False,
            video_mode="live_mujoco",
            video_fps=20,
            video_stride=10,
        ),
    )
    application = SimpleNamespace(
        config=config,
        hooks_by_name={
            "live_mujoco_video": SimpleNamespace(
                output_paths=(pending_main_gif, pending_main_mp4),
                paths=[pending_main_gif, pending_main_mp4],
                errors=[],
                frame_count=2,
            ),
            "observer_camera": SimpleNamespace(
                camera_name="observer_eye_camera",
                output_paths=(pending_observer_gif, pending_observer_mp4),
                paths=[pending_observer_gif, pending_observer_mp4],
                errors=[],
                frame_count=2,
            ),
        },
        loop=SimpleNamespace(backend=SimpleNamespace(config=SimpleNamespace())),
    )
    result = SimulationLoopResult(
        states=(_state(0.0, [0.0, 0.0, 0.10]),),
        commands=(),
        stopped_early=False,
        metadata={"stop_reason": "max_steps"},
    )

    paths = scenario_artifacts.save_scenario_artifacts(application, result)

    metadata = json.loads(paths.metadata_json.read_text(encoding="utf-8"))
    assert metadata["videos"]["gif"].endswith("videos/simulation.gif")
    assert metadata["videos"]["mp4"].endswith("videos/simulation.mp4")
    assert metadata["observer_videos"]["gif"].endswith(
        "videos/observer_eye_camera.gif"
    )
    assert metadata["observer_videos"]["mp4"].endswith(
        "videos/observer_eye_camera.mp4"
    )
    assert metadata["video_frames"] == 2
    assert metadata["observer_video_frames"] == 2


def test_scenario_artifacts_save_separate_named_local_path_plots(tmp_path) -> None:
    target = np.array(
        [
            [0.010, 0.000, 0.100],
            [0.000, 0.010, 0.100],
            [0.010, 0.010, 0.100],
            [-0.010, 0.010, 0.100],
        ],
        dtype=float,
    )
    actual = target + np.array([0.0005, -0.0002, 0.0001])
    arrays = {
        "target_position_m": target,
        "target_actual_position_m": actual,
        "arm_executor_tip_position_m": np.vstack((actual[:1], actual)),
        "target_engine_local_path_name": np.array(
            ["one_third_circle", "one_third_circle", "endpoint_square", "endpoint_square"]
        ),
        "target_engine_executor_subphase": np.array(
            ["path", "path", "path", "path"]
        ),
        "target_engine_local_path_center_m": np.repeat(
            np.array([[0.0, 0.0, 0.100]]),
            4,
            axis=0,
        ),
        "target_engine_insertion_direction_world": np.repeat(
            np.array([[0.0, 0.0, 1.0]]),
            4,
            axis=0,
        ),
    }

    saved = scenario_artifacts._save_plots(arrays, tmp_path)

    assert tmp_path / "engine_navigation_local_path_one_third_circle.png" in saved
    assert tmp_path / "engine_navigation_local_path_endpoint_square.png" in saved


def _state(
    time_s: float,
    tip_position: list[float],
    *,
    tendon_displacement_m: np.ndarray | None = None,
    tendon_velocity_mps: np.ndarray | None = None,
    tendon_target_m: np.ndarray | None = None,
    saturation: dict[str, object] | None = None,
) -> RobotSystemState:
    pose = Pose6D(position=np.asarray(tip_position, dtype=float), quat=np.array([1.0, 0.0, 0.0, 0.0]))
    displacement = (
        np.zeros(9, dtype=float)
        if tendon_displacement_m is None
        else tendon_displacement_m
    )
    velocity = (
        np.zeros(9, dtype=float)
        if tendon_velocity_mps is None
        else tendon_velocity_mps
    )
    return RobotSystemState(
        time_s=time_s,
        base=BaseSystemState(Pose6D.identity()),
        arms={
            "executor": ArmSystemState(
                name="executor",
                role="executor",
                tip_pose_world=pose,
                segment_poses_world=np.repeat(np.eye(4)[None, :, :], 2, axis=0),
                tendon_displacement_m=displacement,
                tendon_velocity_mps=velocity,
                tendon_target_m=(
                    displacement if tendon_target_m is None else tendon_target_m
                ),
            )
        },
        metadata={
            "saturation": {"executor": saturation}
            if saturation is not None
            else {}
        },
    )
