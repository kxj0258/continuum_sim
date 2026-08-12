from __future__ import annotations

import numpy as np
from numpy.testing import assert_allclose

from continuum_sim.visualization.mujoco_system_debug_viewer import (
    bounded_compatible_target,
    named_system_target,
    normalize_target_mm,
    target_rates,
)
from continuum_sim.system.types import ArmTendonRateCommand


def test_target_rates_reach_near_target_and_clip_large_error() -> None:
    target = np.array([0.001, 0.010, -0.010], dtype=float)
    current = np.zeros(3, dtype=float)
    max_rate = np.array([0.10, 0.02, 0.03], dtype=float)

    rates = target_rates(target, current, max_rate, dt=0.1)

    assert_allclose(rates, [0.01, 0.02, -0.03])


def test_named_system_target_addresses_arms_by_name() -> None:
    zeros = {
        "executor": np.zeros(9, dtype=float),
        "observer": np.zeros(9, dtype=float),
    }

    single = named_system_target(
        "observer_tendon_1_pull",
        zeros,
        single_pull_m=-0.002,
        triplet_pull_m=-0.001,
    )
    triplet = named_system_target(
        "executor_segment_1_triplet",
        zeros,
        single_pull_m=-0.002,
        triplet_pull_m=-0.001,
    )

    assert single["observer"][0] == -0.002
    assert np.count_nonzero(single["executor"]) == 0
    assert_allclose(triplet["executor"][:3], [-0.001, -0.001, -0.001])
    assert np.count_nonzero(triplet["observer"]) == 0


def test_normalize_target_mm_accepts_and_clips_finite_values() -> None:
    assert normalize_target_mm("12.5", -20.0, 20.0, 0.0) == 12.5
    assert normalize_target_mm("25", -20.0, 20.0, 0.0) == 20.0
    assert normalize_target_mm("-25", -20.0, 20.0, 0.0) == -20.0


def test_normalize_target_mm_restores_fallback_for_invalid_values() -> None:
    assert normalize_target_mm("not-a-number", -20.0, 20.0, 3.25) == 3.25
    assert normalize_target_mm("nan", -20.0, 20.0, 3.25) == 3.25
    assert normalize_target_mm("inf", -20.0, 20.0, 3.25) == 3.25


def test_raw_debug_command_requires_explicit_mode() -> None:
    compatible = ArmTendonRateCommand(np.zeros(9, dtype=float))
    raw = ArmTendonRateCommand(
        np.zeros(9, dtype=float),
        control_space="raw_tendon_debug",
    )

    assert compatible.control_space == "bending_compatible"
    assert raw.control_space == "raw_tendon_debug"


def test_bounded_compatible_target_scales_whole_delta_without_component_clipping() -> None:
    current = np.zeros(3, dtype=float)
    candidate = np.array([2.0, -1.0, 0.5], dtype=float)
    lower = np.array([-0.4, -0.4, -0.4], dtype=float)
    upper = np.array([0.8, 0.8, 0.8], dtype=float)

    result = bounded_compatible_target(current, candidate, lower, upper)

    assert_allclose(result, [0.8, -0.4, 0.2])
