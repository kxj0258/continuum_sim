import numpy as np
from numpy.testing import assert_allclose

from continuum_sim.control import (
    ContactTriggeredAdmittanceConfig,
    ContactTriggeredAdmittanceTracker,
)


def test_tracker_keeps_force_target_disabled_before_contact() -> None:
    tracker = ContactTriggeredAdmittanceTracker(
        ContactTriggeredAdmittanceConfig(target_normal_force_n=0.5)
    )

    command = tracker.step(
        tip_position=np.zeros(3),
        target_positions=np.array([[0.0, 0.0, 0.0], [0.0, 0.002, 0.0]]),
        normal=np.array([1.0, 0.0, 0.0]),
        measured_normal_force_n=0.0,
        dt=0.01,
    )

    assert command.contact_active is False
    assert command.target_normal_force_n == 0.0
    assert command.waypoint_advanced is False
    assert tracker.target_index == 0
    assert_allclose(command.corrected_target_position, command.target_position)


def test_tracker_enables_force_control_on_contact_and_offsets_normal_target() -> None:
    tracker = ContactTriggeredAdmittanceTracker(
        ContactTriggeredAdmittanceConfig(
            target_normal_force_n=0.5,
            force_filter_alpha=1.0,
            kp_force=1.0,
            ki_force=0.0,
            admittance_damping=0.0,
            admittance_stiffness=0.0,
            admittance_clip_m=0.1,
        )
    )

    command = tracker.step(
        tip_position=np.zeros(3),
        target_positions=np.array([[0.0, 0.0, 0.0]]),
        normal=np.array([1.0, 0.0, 0.0]),
        measured_normal_force_n=0.2,
        dt=0.1,
    )

    assert command.contact_active is True
    assert command.target_normal_force_n == 0.5
    assert command.admittance_position_m < 0.0
    assert command.corrected_target_position[0] < command.target_position[0]


def test_tracker_advances_after_stable_tangent_and_force_tracking() -> None:
    tracker = ContactTriggeredAdmittanceTracker(
        ContactTriggeredAdmittanceConfig(
            target_normal_force_n=0.5,
            tangent_tolerance_m=0.001,
            force_tolerance_n=0.05,
            stable_steps_required=2,
            max_steps_per_target=10,
            force_filter_alpha=1.0,
        )
    )
    targets = np.array([[0.0, 0.0, 0.0], [0.0, 0.01, 0.0]])

    first = tracker.step(
        tip_position=targets[0],
        target_positions=targets,
        normal=np.array([1.0, 0.0, 0.0]),
        measured_normal_force_n=0.5,
        dt=0.01,
    )
    second = tracker.step(
        tip_position=targets[0],
        target_positions=targets,
        normal=np.array([1.0, 0.0, 0.0]),
        measured_normal_force_n=0.5,
        dt=0.01,
    )

    assert first.waypoint_advanced is False
    assert second.waypoint_advanced is True
    assert second.advance_reason == "stable"
    assert second.target_index == 0
    assert tracker.target_index == 1


def test_tracker_advances_on_max_steps_when_force_gate_is_not_met() -> None:
    tracker = ContactTriggeredAdmittanceTracker(
        ContactTriggeredAdmittanceConfig(
            target_normal_force_n=0.5,
            tangent_tolerance_m=0.001,
            force_tolerance_n=0.01,
            stable_steps_required=5,
            max_steps_per_target=3,
            force_filter_alpha=1.0,
        )
    )
    targets = np.array([[0.0, 0.0, 0.0], [0.0, 0.01, 0.0]])
    command = None

    for _ in range(3):
        command = tracker.step(
            tip_position=targets[0],
            target_positions=targets,
            normal=np.array([1.0, 0.0, 0.0]),
            measured_normal_force_n=0.2,
            dt=0.01,
        )

    assert command is not None
    assert command.waypoint_advanced is True
    assert command.advance_reason == "max_steps"
    assert tracker.target_index == 1
