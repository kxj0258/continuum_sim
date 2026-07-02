from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose

from continuum_sim.control.tendon_rate_control import (
    CompatibleTendonRateIntegrator,
    TendonRateLimits,
)
from continuum_sim.model.bending_space import BendingSpaceModel
from continuum_sim.model.physical_tendon import load_physical_tendons_from_yaml
from continuum_sim.model.robot_params import ThreeSegmentRobotParams


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROBOT_CONFIG = PROJECT_ROOT / "configs" / "robot_3seg.yaml"


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
