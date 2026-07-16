from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose

from continuum_sim.control.tendon_rate_control import (
    BendingRateServoConfig,
    CompatibleBendingRateServo,
    CompatibleTendonRateIntegrator,
    TendonRateLimits,
)
from continuum_sim.control.coordinated_tracking import CoordinatedTrackingConfig
from continuum_sim.control.whole_body_controller import (
    WholeBodyController,
    WholeBodyControllerConfig,
)
from continuum_sim.kinematics.whole_body import (
    SingularityConfig,
    analyze_singularity,
)
from continuum_sim.model.bending_space import BendingSpaceModel
from continuum_sim.model.physical_tendon import load_physical_tendons_from_yaml
from continuum_sim.model.robot_assembly import load_robot_assembly_config
from continuum_sim.model.robot_params import ThreeSegmentRobotParams


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROBOT_CONFIG = PROJECT_ROOT / "configs" / "robot_3seg.yaml"
SINGLE_ASSEMBLY_CONFIG = (
    PROJECT_ROOT / "configs" / "robots" / "assemblies" / "single_spatial.yaml"
)


def _model() -> BendingSpaceModel:
    return BendingSpaceModel.from_arm(
        ThreeSegmentRobotParams.from_yaml(ROBOT_CONFIG),
        load_physical_tendons_from_yaml(ROBOT_CONFIG),
    )


def test_bending_mapping_inserts_zero_axial_strain_and_round_trips() -> None:
    model = _model()
    bending = np.array([2.0, -1.0, 3.0, 4.0, -2.0, 0.5], dtype=float)

    q = model.to_q(bending)
    tendon = model.to_tendon(bending)

    assert_allclose(q[[2, 5, 8]], 0.0)
    assert_allclose(model.estimate(tendon), bending)
    assert_allclose(model.residual(tendon), 0.0, atol=1.0e-14)


def test_projection_removes_incompatible_tendon_component() -> None:
    model = _model()
    raw = np.zeros(model.tendon_count, dtype=float)
    raw[0] = -0.001

    projected = model.project(raw)

    assert model.residual_norm(raw) > model.compatibility_tolerance(raw)
    assert model.is_compatible(projected)


def test_compatible_integrator_uses_one_scale_for_all_tendons() -> None:
    model = _model()
    integrator = CompatibleTendonRateIntegrator(
        model,
        TendonRateLimits(
            displacement_min_m=np.full(model.tendon_count, -0.01),
            displacement_max_m=np.full(model.tendon_count, 0.01),
            max_rate_mps=np.full(model.tendon_count, 0.001),
        ),
    )
    requested = model.to_tendon(np.array([10.0, -5.0, 0.0, 0.0, 0.0, 0.0]))

    step = integrator.step(requested, 0.1)

    nonzero = np.abs(requested) > 1.0e-15
    assert_allclose(
        step.applied_rate_mps[nonzero] / requested[nonzero],
        step.common_scale,
    )
    assert model.is_compatible(step.displacement_m)


def test_raw_debug_mode_allows_incompatible_tendon_rate() -> None:
    model = _model()
    integrator = CompatibleTendonRateIntegrator(
        model,
        TendonRateLimits(
            displacement_min_m=np.full(model.tendon_count, -0.01),
            displacement_max_m=np.full(model.tendon_count, 0.01),
            max_rate_mps=np.full(model.tendon_count, 0.01),
        ),
    )
    requested = np.zeros(model.tendon_count, dtype=float)
    requested[0] = -0.001

    step = integrator.step(requested, 0.1, raw_debug=True)

    assert step.raw_debug is True
    assert np.linalg.norm(step.compatibility_residual_mps) > 0.0


def test_bending_rate_servo_retains_unrealized_rate_error_with_bounded_lead() -> None:
    model = _model()
    lead_limit = 0.0001
    servo = CompatibleBendingRateServo(
        model,
        TendonRateLimits(
            displacement_min_m=np.full(model.tendon_count, -0.01),
            displacement_max_m=np.full(model.tendon_count, 0.01),
            max_rate_mps=np.full(model.tendon_count, 0.005),
            target_lead_m=np.full(model.tendon_count, 0.0005),
        ),
        BendingRateServoConfig(
            rate_filter_time_constant_s=0.0,
            feedforward_lead_time_s=0.0,
            rate_proportional_time_s=0.0,
            rate_integral_gain=1.0,
            max_target_lead_m=lead_limit,
        ),
    )
    actual = np.zeros(model.tendon_count, dtype=float)
    servo.reset(actual)
    requested = model.to_tendon(
        np.array([1.0, -0.5, 0.0, 0.0, 0.0, 0.0], dtype=float)
    )

    first = servo.step(requested, 0.02, actual_displacement_m=actual)
    latest = first
    for _ in range(20):
        latest = servo.step(requested, 0.02, actual_displacement_m=actual)

    assert np.max(np.abs(latest.target_lead_m)) <= lead_limit + 1.0e-10
    assert np.linalg.norm(latest.target_lead_m) >= np.linalg.norm(first.target_lead_m)
    assert latest.guard_feasible is True
    assert model.is_compatible(latest.target_lead_m)


def test_bending_rate_servo_zero_command_hold_preserves_target() -> None:
    model = _model()
    servo = CompatibleBendingRateServo(
        model,
        TendonRateLimits(
            displacement_min_m=np.full(model.tendon_count, -0.01),
            displacement_max_m=np.full(model.tendon_count, 0.01),
            max_rate_mps=np.full(model.tendon_count, 0.005),
        ),
        BendingRateServoConfig(
            rate_filter_time_constant_s=0.0,
            feedforward_lead_time_s=0.02,
            rate_integral_gain=0.0,
            zero_command_mode="hold",
        ),
    )
    actual = np.zeros(model.tendon_count, dtype=float)
    servo.reset(actual)
    requested = model.to_tendon(
        np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
    )

    moving = servo.step(requested, 0.02, actual_displacement_m=actual)
    holding = servo.step(
        np.zeros(model.tendon_count, dtype=float),
        0.02,
        actual_displacement_m=actual,
    )

    assert_allclose(holding.displacement_m, moving.displacement_m, atol=1.0e-10)
    assert_allclose(holding.target_rate_mps, 0.0, atol=1.0e-10)


def test_bending_rate_servo_zero_command_relax_clears_integral() -> None:
    model = _model()
    max_rate = 0.001
    servo = CompatibleBendingRateServo(
        model,
        TendonRateLimits(
            displacement_min_m=np.full(model.tendon_count, -0.01),
            displacement_max_m=np.full(model.tendon_count, 0.01),
            max_rate_mps=np.full(model.tendon_count, max_rate),
        ),
        BendingRateServoConfig(
            rate_filter_time_constant_s=0.0,
            rate_integral_gain=1.0,
            zero_command_mode="relax",
        ),
    )
    actual = np.zeros(model.tendon_count, dtype=float)
    servo.reset(actual)
    requested = model.to_tendon(
        np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
    )

    moving = servo.step(requested, 0.02, actual_displacement_m=actual)
    relaxing = servo.step(
        np.zeros(model.tendon_count, dtype=float),
        0.02,
        actual_displacement_m=actual,
    )

    assert np.linalg.norm(moving.bending_integral) > 0.0
    assert_allclose(relaxing.bending_integral, 0.0, atol=1.0e-12)
    assert np.linalg.norm(relaxing.displacement_m - actual) <= np.linalg.norm(
        moving.displacement_m - actual
    )
    assert np.max(np.abs(relaxing.target_rate_mps)) <= max_rate + 1.0e-12


def test_bending_rate_servo_hard_force_guard_reduces_target_lead() -> None:
    model = _model()
    servo = CompatibleBendingRateServo(
        model,
        TendonRateLimits(
            displacement_min_m=np.full(model.tendon_count, -0.01),
            displacement_max_m=np.full(model.tendon_count, 0.01),
            max_rate_mps=np.full(model.tendon_count, 0.01),
        ),
        BendingRateServoConfig(
            rate_filter_time_constant_s=0.0,
            feedforward_lead_time_s=0.02,
            rate_integral_gain=0.0,
            soft_force_limit_n=2.0,
            hard_force_limit_n=3.0,
        ),
    )
    actual = np.zeros(model.tendon_count, dtype=float)
    servo.reset(actual)
    requested = model.to_tendon(
        np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
    )

    unguarded = servo.step(requested, 0.02, actual_displacement_m=actual)
    guarded = servo.step(
        requested,
        0.02,
        actual_displacement_m=actual,
        actuator_force_n=np.full(model.tendon_count, 6.0),
    )

    assert np.all(guarded.hard_force_saturated)
    assert_allclose(guarded.force_scale, 1.0 / 3.0)
    assert np.linalg.norm(guarded.target_lead_m) < np.linalg.norm(
        unguarded.target_lead_m
    )


def test_structural_zero_singular_value_does_not_throttle_controllable_space() -> None:
    config = SingularityConfig(minimum_singular_value=0.5)

    report = analyze_singularity(np.diag([1.0, 0.0]), config)

    assert report.rank == 1
    assert report.full_rank is False
    assert report.condition_number == float("inf")
    assert_allclose(report.damping, config.nominal_damping)
    assert_allclose(report.velocity_scale, 1.0)


def test_all_zero_jacobian_keeps_maximum_singularity_protection() -> None:
    config = SingularityConfig()

    report = analyze_singularity(np.zeros((2, 3), dtype=float), config)

    assert report.rank == 0
    assert report.damping == config.maximum_damping
    assert report.velocity_scale == config.minimum_velocity_scale


def test_whole_body_regularization_penalizes_mapped_tendon_effort() -> None:
    assembly = load_robot_assembly_config(SINGLE_ASSEMBLY_CONFIG)
    config = WholeBodyControllerConfig(tendon_regularization_weight=0.2)
    controller = WholeBodyController(assembly, config)
    model = controller.layout.bending_models["executor"]

    regularization = controller._regularization_matrix()

    assert_allclose(
        regularization,
        np.sqrt(config.tendon_regularization_weight) * model.coupling_matrix,
    )


def test_dual_arm_minimum_distance_defaults_to_ten_millimetres() -> None:
    assert CoordinatedTrackingConfig().inter_arm_min_distance_m == 0.010


def test_observer_collision_weight_can_override_legacy_collision_weight() -> None:
    assembly = load_robot_assembly_config(SINGLE_ASSEMBLY_CONFIG)
    controller = WholeBodyController(
        assembly,
        WholeBodyControllerConfig(
            executor_collision_avoidance_weight=80.0,
            observer_collision_avoidance_weight=250.0,
        ),
    )

    assert controller.weight_for("observer_collision_avoidance") == 250.0
