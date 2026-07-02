from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose

from continuum_sim.control.tendon_rate_control import (
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
