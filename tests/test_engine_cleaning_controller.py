from __future__ import annotations

import numpy as np
from numpy.testing import assert_allclose
import pytest

from continuum_sim.control.engine_cleaning_controller import (
    EngineCleaningController,
    limit_tcp_velocity,
)
from continuum_sim.control.engine_cleaning_types import (
    EngineCleaningControllerGains,
    EngineCleaningFeedback,
)
from continuum_sim.model.base_pose import Pose6D
from continuum_sim.scenes.engine_surfaces import load_surface_patch_config
from continuum_sim.tasks.engine_surface_path import (
    CleaningWaypoint,
    EngineSurfacePathConfig,
    build_raster_cleaning_path,
)


def test_controller_rejects_empty_waypoint_list() -> None:
    with pytest.raises(ValueError, match="waypoints"):
        EngineCleaningController(_gains(), [])


def test_approach_phase_velocity_points_to_waypoint() -> None:
    controller = EngineCleaningController(_gains(), [_waypoint("approach", [0.0, 0.0, 0.1])])

    command = controller.step(_feedback([0.0, 0.0, 0.0]))

    assert command.phase == "approach"
    assert command.active_waypoint_index == 0
    assert command.desired_tcp_velocity_world[2] > 0.0
    assert_allclose(command.desired_tcp_velocity_world[:2], [0.0, 0.0])


def test_approach_reached_advances_to_contact_waypoint() -> None:
    controller = EngineCleaningController(
        _gains(),
        [
            _waypoint("approach", [0.0, 0.0, 0.0], index=0),
            _waypoint("contact", [0.1, 0.0, 0.0], index=1),
        ],
    )

    command = controller.step(_feedback([0.001, 0.0, 0.0]))

    assert command.waypoint_reached is True
    assert command.active_waypoint_index == 0
    assert controller.active_index == 1
    assert controller.current_waypoint().phase == "contact"


def test_contact_phase_projects_tangential_error_to_tangent_plane() -> None:
    controller = EngineCleaningController(_gains(), [_waypoint("contact", [0.1, 0.0, 0.0])])

    command = controller.step(_feedback([0.0, 0.0, 0.0], measured_normal_force_n=1.0))

    assert command.phase == "contact"
    assert command.desired_tcp_velocity_world[0] > 0.0
    assert abs(command.metadata["tangent_velocity"][2]) < 1.0e-12


def test_contact_phase_tool_too_far_moves_along_negative_normal() -> None:
    waypoint = _waypoint("contact", [0.0, 0.0, 0.0], standoff_distance_m=0.01)
    controller = EngineCleaningController(_gains(), [waypoint])

    command = controller.step(_feedback([0.0, 0.0, 0.05], measured_normal_force_n=1.0))

    assert command.desired_tcp_velocity_world[2] < 0.0
    assert command.metadata["signed_gap_m"] == pytest.approx(0.05)


def test_contact_phase_low_force_moves_along_negative_normal() -> None:
    controller = EngineCleaningController(_gains(), [_waypoint("contact", [0.0, 0.0, 0.0])])

    command = controller.step(_feedback([0.0, 0.0, 0.0], measured_normal_force_n=0.0))

    assert command.desired_tcp_velocity_world[2] < 0.0
    assert command.metadata["force_error_n"] > 0.0


def test_contact_phase_high_force_below_safety_limit_moves_along_positive_normal() -> None:
    controller = EngineCleaningController(_gains(), [_waypoint("contact", [0.0, 0.0, 0.0])])

    command = controller.step(_feedback([0.0, 0.0, 0.0], measured_normal_force_n=2.0))

    assert command.desired_tcp_velocity_world[2] > 0.0
    assert command.safety_stop is False


def test_contact_reached_advances_to_retreat_waypoint() -> None:
    controller = EngineCleaningController(
        _gains(),
        [
            _waypoint("contact", [0.0, 0.0, 0.0], index=0),
            _waypoint("retreat", [0.0, 0.0, 0.1], index=1),
        ],
    )

    command = controller.step(_feedback([0.001, 0.001, 0.02], measured_normal_force_n=1.0))

    assert command.waypoint_reached is True
    assert controller.active_index == 1
    assert controller.current_waypoint().phase == "retreat"


def test_retreat_phase_velocity_points_to_retreat_waypoint() -> None:
    controller = EngineCleaningController(_gains(), [_waypoint("retreat", [0.0, 0.0, 0.1])])

    command = controller.step(_feedback([0.0, 0.0, 0.0]))

    assert command.phase == "retreat"
    assert command.desired_tcp_velocity_world[2] > 0.0


def test_retreat_reached_marks_controller_done() -> None:
    controller = EngineCleaningController(_gains(), [_waypoint("retreat", [0.0, 0.0, 0.0])])

    command = controller.step(_feedback([0.0, 0.0, 0.001]))

    assert command.waypoint_reached is True
    assert controller.is_done() is True


def test_velocity_norm_is_limited_to_max_tcp_speed() -> None:
    velocity = limit_tcp_velocity(
        np.array([1.0, 1.0, 0.0], dtype=float),
        np.array([0.0, 0.0, 1.0], dtype=float),
        max_tcp_speed_mps=0.04,
        max_normal_speed_mps=0.015,
    )

    assert np.linalg.norm(velocity) <= 0.04 + 1.0e-12


def test_normal_velocity_is_limited_to_max_normal_speed() -> None:
    velocity = limit_tcp_velocity(
        np.array([0.0, 0.0, -1.0], dtype=float),
        np.array([0.0, 0.0, 1.0], dtype=float),
        max_tcp_speed_mps=0.04,
        max_normal_speed_mps=0.015,
    )

    assert abs(velocity[2]) <= 0.015 + 1.0e-12


def test_exceeding_max_contact_force_triggers_safety_stop() -> None:
    controller = EngineCleaningController(_gains(), [_waypoint("contact", [0.0, 0.0, 0.0])])

    command = controller.step(_feedback([0.0, 0.0, 0.0], measured_normal_force_n=5.0))

    assert command.safety_stop is True
    assert command.stop_reason == "max_contact_force_exceeded"
    assert_allclose(command.desired_tcp_velocity_world, [0.0, 0.0, 0.0])
    assert controller.active_index == 0


def test_safety_stop_persists_with_zero_velocity() -> None:
    controller = EngineCleaningController(_gains(), [_waypoint("contact", [0.0, 0.0, 0.0])])
    controller.step(_feedback([0.0, 0.0, 0.0], measured_normal_force_n=5.0))

    command = controller.step(_feedback([0.0, 0.0, 0.0], measured_normal_force_n=0.0))

    assert command.safety_stop is True
    assert command.stop_reason == "max_contact_force_exceeded"
    assert_allclose(command.desired_tcp_velocity_world, [0.0, 0.0, 0.0])


def test_unknown_waypoint_phase_raises_clear_error() -> None:
    controller = EngineCleaningController(_gains(), [_waypoint("hover", [0.0, 0.0, 0.0])])

    with pytest.raises(ValueError, match="waypoint phase"):
        controller.step(_feedback([0.0, 0.0, 0.0]))


def test_command_reports_phase_index_reached_and_metadata() -> None:
    controller = EngineCleaningController(_gains(), [_waypoint("contact", [0.0, 0.0, 0.0], index=7)])

    command = controller.step(_feedback([0.0, 0.0, 0.0], measured_normal_force_n=1.0))

    assert command.active_waypoint_index == 0
    assert command.phase == "contact"
    assert command.waypoint_reached is True
    assert command.stop_reason is None
    assert command.metadata["waypoint_path_index"] == 7


def test_controller_accepts_m5_raster_cleaning_path_waypoints() -> None:
    patch = load_surface_patch_config(
        {
            "name": "plane_path",
            "type": "plane_patch",
            "center": [0.0, 0.0, 0.0],
            "normal": [0.0, 0.0, 1.0],
            "tangent_u": [1.0, 0.0, 0.0],
            "size_u_m": 0.02,
            "size_v_m": 0.02,
        }
    )
    waypoints = build_raster_cleaning_path(
        patch,
        EngineSurfacePathConfig(
            patch_name="plane_path",
            num_passes_u=2,
            num_passes_v=1,
            approach_distance_m=0.01,
            retreat_distance_m=0.01,
            target_force_n=1.0,
            standoff_distance_m=0.0,
        ),
    )
    controller = EngineCleaningController(_gains(), waypoints)

    command = controller.step(_feedback(waypoints[0].position))

    assert command.phase == "approach"
    assert command.waypoint_reached is True
    assert controller.current_waypoint().phase == "contact"


def _gains() -> EngineCleaningControllerGains:
    return EngineCleaningControllerGains(
        tangential_position_gain=1.5,
        normal_position_gain=1.0,
        normal_force_gain=0.2,
        approach_position_gain=1.2,
        retreat_position_gain=1.2,
        max_tcp_speed_mps=0.04,
        max_normal_speed_mps=0.015,
        waypoint_tolerance_m=0.005,
        max_contact_force_n=3.0,
        force_deadband_n=0.05,
        min_clearance_m=0.01,
    )


def _feedback(
    position: list[float] | np.ndarray,
    *,
    measured_normal_force_n: float = 0.0,
) -> EngineCleaningFeedback:
    return EngineCleaningFeedback(
        tcp_pose=Pose6D.from_dict(
            {
                "position": list(np.asarray(position, dtype=float)),
                "quat": [1.0, 0.0, 0.0, 0.0],
            }
        ),
        measured_normal_force_n=measured_normal_force_n,
    )


def _waypoint(
    phase: str,
    position: list[float],
    *,
    index: int = 0,
    target_force_n: float = 1.0,
    standoff_distance_m: float = 0.0,
) -> CleaningWaypoint:
    return CleaningWaypoint(
        position=np.asarray(position, dtype=float),
        normal=np.array([0.0, 0.0, 1.0], dtype=float),
        tangent_u=np.array([1.0, 0.0, 0.0], dtype=float),
        tangent_v=np.array([0.0, 1.0, 0.0], dtype=float),
        phase=phase,
        target_force_n=target_force_n,
        standoff_distance_m=standoff_distance_m,
        index=index,
        metadata={},
    )
