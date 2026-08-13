from __future__ import annotations

from contextlib import nullcontext
from threading import RLock
from types import SimpleNamespace

import numpy as np
from numpy.testing import assert_allclose

from continuum_sim.visualization.mujoco_system_debug_viewer import (
    MujocoSystemDebugViewer,
    bounded_compatible_target,
    named_system_target,
    normalize_curvature_target,
    normalize_target_mm,
    target_rates,
)
from continuum_sim.runtime.concurrency import LatestValueSlot
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


def test_normalize_curvature_target_accepts_and_clips_finite_values() -> None:
    assert normalize_curvature_target("12.5", 0.0) == 12.5
    assert normalize_curvature_target("40", 0.0) == 30.0
    assert normalize_curvature_target("-40", 0.0) == -30.0


def test_normalize_curvature_target_restores_fallback_for_invalid_values() -> None:
    assert normalize_curvature_target("not-a-number", 3.25) == 3.25
    assert normalize_curvature_target("nan", 3.25) == 3.25
    assert normalize_curvature_target("inf", 3.25) == 3.25


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


def test_set_targets_defers_widget_sync_to_ui_refresh() -> None:
    viewer = MujocoSystemDebugViewer.__new__(MujocoSystemDebugViewer)
    viewer.targets = {
        "executor": np.zeros(3, dtype=float),
        "observer": np.zeros(3, dtype=float),
    }
    viewer.sliders = {
        name: [_Slider() for _ in range(3)] for name in viewer.targets
    }
    viewer.target_inputs = {
        name: [_TextBox() for _ in range(3)] for name in viewer.targets
    }
    viewer._updating_controls = False
    viewer._views_dirty = False
    viewer._dirty_target_arms = set()
    viewer._target_lock = RLock()
    viewer._target_slot = LatestValueSlot(_copy_targets(viewer.targets))
    viewer._simulation_lock = _ForbiddenLock()

    viewer.set_targets({"executor": np.array([0.001, 0.0, 0.0])})

    assert_allclose(viewer.targets["executor"], [0.001, 0.0, 0.0])
    assert_allclose(viewer.targets["observer"], np.zeros(3))
    assert [item.set_calls for item in viewer.sliders["executor"]] == [0, 0, 0]
    assert [item.set_calls for item in viewer.target_inputs["executor"]] == [0, 0, 0]
    assert viewer._dirty_target_arms == {"executor"}

    viewer._sync_target_controls()

    assert [item.set_calls for item in viewer.sliders["executor"]] == [1, 0, 0]
    assert [item.set_calls for item in viewer.target_inputs["executor"]] == [1, 0, 0]
    assert [item.set_calls for item in viewer.sliders["observer"]] == [0, 0, 0]
    assert [item.set_calls for item in viewer.target_inputs["observer"]] == [0, 0, 0]
    assert viewer.sliders["executor"][0].set_states == [(False, False)]
    assert viewer.target_inputs["executor"][0].set_states == [(False, False)]
    assert viewer._dirty_target_arms == set()
    assert viewer._views_dirty is True


def test_raw_slider_change_marks_views_dirty() -> None:
    viewer = MujocoSystemDebugViewer.__new__(MujocoSystemDebugViewer)
    viewer.targets = {"executor": np.zeros(1, dtype=float)}
    viewer.target_inputs = {"executor": [_TextBox()]}
    viewer._updating_controls = False
    viewer._views_dirty = False
    viewer._dirty_target_arms = set()
    viewer._target_lock = RLock()
    viewer._target_slot = LatestValueSlot(_copy_targets(viewer.targets))
    viewer._simulation_lock = _ForbiddenLock()
    viewer.control_space = "raw_tendon_debug"
    viewer.runtime_timing = None

    viewer._on_slider("executor", 0, 1.0)

    assert_allclose(viewer.targets["executor"], [0.001])
    assert viewer._views_dirty is True


def test_curvature_slider_defers_widget_sync_and_marks_latency_input() -> None:
    viewer = MujocoSystemDebugViewer.__new__(MujocoSystemDebugViewer)
    viewer.targets = {"executor": np.zeros(6, dtype=float)}
    viewer.sliders = {}
    viewer.target_inputs = {}
    viewer.curvature_sliders = {"executor": [_Slider() for _ in range(6)]}
    viewer.curvature_inputs = {"executor": [_TextBox() for _ in range(6)]}
    viewer._updating_controls = False
    viewer._views_dirty = False
    viewer._dirty_target_arms = set()
    viewer._dirty_curvature_components = {}
    viewer._target_lock = RLock()
    viewer._target_slot = LatestValueSlot(_copy_targets(viewer.targets))
    viewer._target_limits = {
        "executor": (
            np.full(6, -20.0, dtype=float),
            np.full(6, 20.0, dtype=float),
        )
    }
    viewer._simulation_lock = _ForbiddenLock()
    viewer.control_mode = "curvature"
    viewer.runtime_timing = _Timing()
    model = _IdentityBendingModel()
    arm = SimpleNamespace(
        name="executor",
        spatial_arm=SimpleNamespace(
            limits=SimpleNamespace(
                tendon_displacement_min_m=np.full(6, -0.020),
                tendon_displacement_max_m=np.full(6, 0.020),
            )
        ),
    )
    viewer.backend = SimpleNamespace(
        layout=SimpleNamespace(bending_models={"executor": model}),
        assembly=SimpleNamespace(enabled_arms=(arm,)),
    )

    viewer._on_curvature_slider("executor", 0, 12.5)

    assert viewer.runtime_timing.input_labels == ["executor:S1:kx:slider"]
    assert sum(item.set_calls for item in viewer.curvature_sliders["executor"]) == 0
    assert sum(item.set_calls for item in viewer.curvature_inputs["executor"]) == 0
    assert viewer._dirty_target_arms == {"executor"}
    assert viewer._dirty_curvature_components == {"executor": {0}}

    viewer._sync_curvature_controls()

    assert viewer.curvature_sliders["executor"][0].set_calls == 1
    assert viewer.curvature_inputs["executor"][0].set_calls == 1
    assert viewer.curvature_sliders["executor"][0].set_states == [(False, False)]
    assert viewer.curvature_inputs["executor"][0].set_states == [(False, False)]
    assert sum(item.set_calls for item in viewer.curvature_sliders["executor"][1:]) == 0
    assert sum(item.set_calls for item in viewer.curvature_inputs["executor"][1:]) == 0


def test_set_targets_uses_backend_limits_without_tendon_widgets() -> None:
    viewer = MujocoSystemDebugViewer.__new__(MujocoSystemDebugViewer)
    viewer.targets = {"executor": np.zeros(3, dtype=float)}
    viewer.sliders = {}
    viewer._target_limits = {
        "executor": (
            np.full(3, -0.002, dtype=float),
            np.full(3, 0.002, dtype=float),
        )
    }
    viewer._views_dirty = False
    viewer._dirty_target_arms = set()
    viewer._target_lock = RLock()
    viewer._target_slot = LatestValueSlot(_copy_targets(viewer.targets))

    viewer.set_targets({"executor": np.array([0.003, 0.001, -0.003])})

    assert_allclose(viewer.targets["executor"], [0.002, 0.001, -0.002])
    assert viewer._dirty_target_arms == {"executor"}


class _Slider:
    valmin = -20.0
    valmax = 20.0

    def __init__(self) -> None:
        self.val = 0.0
        self.set_calls = 0
        self.drawon = True
        self.eventson = True
        self.set_states: list[tuple[bool, bool]] = []

    def set_val(self, value: float) -> None:
        self.set_states.append((self.drawon, self.eventson))
        self.val = float(value)
        self.set_calls += 1


class _TextBox:
    def __init__(self) -> None:
        self.text = "0.000"
        self.set_calls = 0
        self.drawon = True
        self.eventson = True
        self.set_states: list[tuple[bool, bool]] = []

    def set_val(self, value: str) -> None:
        self.set_states.append((self.drawon, self.eventson))
        self.text = str(value)
        self.set_calls += 1


class _Timing:
    def __init__(self) -> None:
        self.input_labels: list[str] = []

    def mark_input(self, label: str) -> None:
        self.input_labels.append(label)

    def measure(self, stage: str):
        del stage
        return nullcontext()


class _IdentityBendingModel:
    def project(self, values: np.ndarray) -> np.ndarray:
        return np.asarray(values, dtype=float).copy()

    def estimate(self, values: np.ndarray) -> np.ndarray:
        return np.asarray(values, dtype=float).copy()

    def to_tendon(self, values: np.ndarray) -> np.ndarray:
        return np.asarray(values, dtype=float).copy()


class _ForbiddenLock:
    def __enter__(self):
        raise AssertionError("UI target callbacks must not acquire the MuJoCo lock")

    def __exit__(self, exc_type, exc, traceback) -> None:
        del exc_type, exc, traceback


def _copy_targets(targets: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {name: values.copy() for name, values in targets.items()}
