import numpy as np
from numpy.testing import assert_allclose

from continuum_sim.control.contact_triggered_admittance import (
    ContactTriggeredAdmittanceConfig,
)
from continuum_sim.control.wiping_force_strategies import (
    ContactTriggeredAdmittanceStrategy,
    KinematicHybridForceStrategy,
    WipingForceContext,
)
from continuum_sim.model.base_pose import Pose6D
from continuum_sim.system.types import ArmSystemState


def test_kinematic_hybrid_strategy_keeps_existing_force_distance_correction() -> None:
    strategy = KinematicHybridForceStrategy()
    context = _context(
        phase="contact",
        contact_error_m=-0.002,
        estimated_force_n=0.5,
        target_force_n=1.5,
        normal_force_gain=0.075,
    )

    result = strategy.compute(context)

    expected_correction = -0.002 + 0.075 * ((1.5 - 0.5) / 600.0)
    assert_allclose(
        result.corrected_waypoint,
        np.array([0.0, 0.0, expected_correction]),
    )
    assert result.metadata["wiping_force_strategy"] == "kinematic_hybrid"


def test_contact_triggered_admittance_strategy_controls_waypoint_advance() -> None:
    strategy = ContactTriggeredAdmittanceStrategy(
        ContactTriggeredAdmittanceConfig(
            target_normal_force_n=0.5,
            force_filter_alpha=1.0,
            tangent_tolerance_m=0.001,
            force_tolerance_n=0.05,
            stable_steps_required=1,
        )
    )
    context = _context(
        phase="contact",
        estimated_force_n=0.5,
        target_force_n=0.5,
        waypoints_world=np.array(
            [[0.0, 0.0, 0.0], [0.0, 0.01, 0.0]],
            dtype=float,
        ),
    )

    result = strategy.compute(context)

    assert result.controls_waypoint_advance is True
    assert result.waypoint_advanced is True
    assert result.metadata["force_control_active"] is True
    assert result.metadata["normal_force_source"] == "distance_proxy"


def test_contact_triggered_admittance_strategy_leaves_approach_advance_to_tracker() -> None:
    strategy = ContactTriggeredAdmittanceStrategy(ContactTriggeredAdmittanceConfig())
    context = _context(phase="approach")

    result = strategy.compute(context)

    assert result.controls_waypoint_advance is False
    assert result.waypoint_advanced is False
    assert result.metadata["force_control_active"] is False
    assert_allclose(result.corrected_waypoint, context.waypoint)


def _context(
    *,
    phase: str,
    contact_error_m: float = 0.0,
    estimated_force_n: float = 0.0,
    target_force_n: float = 0.0,
    normal_force_gain: float = 0.0,
    waypoints_world: np.ndarray | None = None,
) -> WipingForceContext:
    return WipingForceContext(
        executor=ArmSystemState(
            name="executor",
            role="executor",
            tip_pose_world=Pose6D.identity(),
            segment_poses_world=np.repeat(np.eye(4)[None, :, :], 3, axis=0),
            tendon_displacement_m=np.zeros(9, dtype=float),
            tendon_velocity_mps=np.zeros(9, dtype=float),
        ),
        waypoints_world=(
            np.array([[0.0, 0.0, 0.0]], dtype=float)
            if waypoints_world is None
            else waypoints_world
        ),
        waypoint_index=0,
        phase=phase,
        surface_normal_world=np.array([0.0, 0.0, 1.0], dtype=float),
        query_normal_world=np.array([0.0, 0.0, 1.0], dtype=float),
        contact_error_m=contact_error_m,
        estimated_force_n=estimated_force_n,
        target_force_n=target_force_n,
        normal_force_gain=normal_force_gain,
        force_proxy_stiffness_n_m=600.0,
        contact_tolerance_m=0.002,
        controller_dt_s=0.02,
    )
