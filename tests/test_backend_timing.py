from types import SimpleNamespace

import pytest

from continuum_sim.application.backend_factory import _validate_runtime_timing


def _scenario_timing(*, controller_dt_s: float, n_substeps: int):
    return SimpleNamespace(
        runtime=SimpleNamespace(
            controller_dt_s=controller_dt_s,
            n_substeps=n_substeps,
        )
    )


def _mujoco_timing(*, timestep: float):
    return SimpleNamespace(solver=SimpleNamespace(timestep=timestep))


def test_runtime_timing_accepts_one_control_period_of_physics() -> None:
    _validate_runtime_timing(
        _scenario_timing(controller_dt_s=0.02, n_substeps=20),
        _mujoco_timing(timestep=0.001),
    )


def test_runtime_timing_rejects_controller_physics_clock_mismatch() -> None:
    with pytest.raises(ValueError, match="runtime timing mismatch"):
        _validate_runtime_timing(
            _scenario_timing(controller_dt_s=0.02, n_substeps=10),
            _mujoco_timing(timestep=0.001),
        )
