from __future__ import annotations

import numpy as np

from continuum_sim.model.base_pose import Pose6D
from continuum_sim.runtime.hooks import (
    _TrackingOverlayState,
    _metadata_path,
    _metadata_paths,
    _metadata_point,
    _sample_overlay_points,
    _split_target_history,
)
from continuum_sim.system.types import (
    ArmTendonRateCommand,
    ArmSystemState,
    BaseSystemState,
    RobotSystemCommand,
    RobotSystemState,
)


def test_overlay_state_captures_engine_navigation_histories() -> None:
    state = _state()
    command = RobotSystemCommand(
        base_twist_world=np.zeros(6),
        arms={"executor": ArmTendonRateCommand(tendon_rate_mps=np.zeros(3))},
        metadata={
            "task_type": "engine_navigation",
            "engine_navigation_active_target_m": np.array([0.3, 0.2, 0.1]),
            "engine_navigation_active_target_kind": "base",
            "engine_navigation_base_path_m": np.array(
                [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]]
            ),
        },
    )
    overlay = _TrackingOverlayState()

    overlay.capture(state, command, max_points=10)

    np.testing.assert_allclose(overlay.base_trail[-1], [0.1, 0.2, 0.3])
    np.testing.assert_allclose(overlay.tip_trail[-1], [0.2, 0.3, 0.4])
    np.testing.assert_allclose(overlay.target_trail[-1], [0.3, 0.2, 0.1])
    assert overlay.target_trail_kinds == ["base"]
    assert overlay.navigation_metadata[
        "engine_navigation_active_target_kind"
    ] == "base"


def test_overlay_state_falls_back_to_generic_executor_target() -> None:
    overlay = _TrackingOverlayState()
    command = RobotSystemCommand(
        base_twist_world=np.zeros(6),
        arms={"executor": ArmTendonRateCommand(tendon_rate_mps=np.zeros(3))},
        metadata={"executor_target_world": np.array([0.4, 0.5, 0.6])},
    )

    overlay.capture(_state(), command, max_points=10)

    np.testing.assert_allclose(overlay.target_trail[-1], [0.4, 0.5, 0.6])
    assert overlay.navigation_metadata == {}
    assert overlay.base_trail == []


def test_overlay_state_captures_generic_observer_roi() -> None:
    overlay = _TrackingOverlayState()
    command = RobotSystemCommand(
        base_twist_world=np.zeros(6),
        arms={"executor": ArmTendonRateCommand(tendon_rate_mps=np.zeros(3))},
        metadata={
            "executor_target_world": np.array([0.4, 0.5, 0.6]),
            "visual_servo_roi_world": np.array([0.1, 0.2, 0.3]),
        },
    )

    overlay.capture(_state(), command, max_points=10)

    np.testing.assert_allclose(overlay.observer_roi_world, [0.1, 0.2, 0.3])


def test_overlay_state_bounds_and_clears_histories() -> None:
    overlay = _TrackingOverlayState()
    command = RobotSystemCommand(
        base_twist_world=np.zeros(6),
        arms={"executor": ArmTendonRateCommand(tendon_rate_mps=np.zeros(3))},
        metadata={"executor_target_world": np.array([0.4, 0.5, 0.6])},
    )
    for index in range(4):
        command.metadata["executor_target_world"] = np.array(
            [0.4 + index, 0.5, 0.6]
        )
        overlay.capture(_state(), command, max_points=2)

    assert len(overlay.tip_trail) == 2
    assert len(overlay.target_trail) == 2

    overlay.clear()

    assert overlay.tip_trail == []
    assert overlay.target_trail == []
    assert overlay.target_trail_kinds == []
    assert overlay.base_trail == []
    assert overlay.observer_roi_world is None
    assert overlay.navigation_metadata == {}


def test_overlay_metadata_helpers_reject_invalid_shapes_and_keep_path_end() -> None:
    metadata = {
        "point": np.array([1.0, 2.0, 3.0]),
        "path": np.arange(15, dtype=float).reshape(5, 3),
        "bad_point": np.zeros(2),
        "bad_path": np.zeros((2, 2)),
    }

    np.testing.assert_allclose(_metadata_point(metadata, "point"), metadata["point"])
    np.testing.assert_allclose(_metadata_path(metadata, "path"), metadata["path"])
    assert _metadata_point(metadata, "bad_point") is None
    assert _metadata_path(metadata, "bad_path") is None
    sampled = _sample_overlay_points(metadata["path"], 3)
    np.testing.assert_allclose(sampled, metadata["path"][[0, 3, 4]])
    paths = _metadata_paths(
        {
            "paths": (
                metadata["path"],
                metadata["bad_path"],
            )
        },
        "paths",
    )
    assert len(paths) == 1
    np.testing.assert_allclose(paths[0], metadata["path"])


def test_target_history_is_split_when_active_target_kind_changes() -> None:
    points = [np.array([float(index), 0.0, 0.0]) for index in range(5)]

    segments = _split_target_history(
        points,
        ["base", "base", "base", "executor", "executor"],
        stride=2,
    )

    assert len(segments) == 2
    np.testing.assert_allclose(segments[0], points[:3:2])
    np.testing.assert_allclose(segments[1], points[3:])


def _state() -> RobotSystemState:
    arm = ArmSystemState(
        name="executor",
        role="executor",
        tip_pose_world=Pose6D(position=np.array([0.2, 0.3, 0.4])),
        segment_poses_world=np.zeros((0, 4, 4)),
        tendon_displacement_m=np.zeros(3),
        tendon_velocity_mps=np.zeros(3),
    )
    return RobotSystemState(
        time_s=0.0,
        base=BaseSystemState(
            pose=Pose6D(position=np.array([0.1, 0.2, 0.3]))
        ),
        arms={"executor": arm},
    )
