"""Analytic PCC backend adapter used by task-level scripts and tests."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from continuum_sim.actuation import load_motor_params_from_yaml
from continuum_sim.actuation.motor_mapping import motor_position_to_tendon_delta
from continuum_sim.backends.base_types import BackendState
from continuum_sim.config import load_yaml
from continuum_sim.kinematics import forward_kinematics
from continuum_sim.model import (
    ThreeSegmentRobotParams,
    load_physical_tendons_from_yaml,
    physical_tendon_delta_to_q,
)


@dataclass(frozen=True)
class AnalyticBackendConfig:
    """Validated YAML configuration for the PCC analytic backend."""

    path: Path
    robot_config_path: Path
    samples_per_segment: int
    timestep: float


class AnalyticBackend:
    """Quasi-static PCC backend that exposes the common backend interface."""

    def __init__(self, config: AnalyticBackendConfig) -> None:
        self.config = config
        self.params = ThreeSegmentRobotParams.from_yaml(config.robot_config_path)
        self.physical_tendons = tuple(load_physical_tendons_from_yaml(config.robot_config_path))
        self.motor_params = tuple(load_motor_params_from_yaml(config.robot_config_path))
        self._time = 0.0
        self._motor_position = np.zeros((len(self.motor_params),), dtype=float)
        self._state = self._compute_state()

    @classmethod
    def from_config(
        cls,
        config: str | Path | AnalyticBackendConfig,
    ) -> "AnalyticBackend":
        analytic_config = (
            config if isinstance(config, AnalyticBackendConfig) else load_analytic_backend_config(config)
        )
        return cls(analytic_config)

    def reset(self) -> BackendState:
        self._time = 0.0
        self._motor_position = np.zeros((len(self.motor_params),), dtype=float)
        self._state = self._compute_state()
        return self._state

    def step(
        self,
        control: np.ndarray | None = None,
        n_substeps: int = 1,
    ) -> BackendState:
        if n_substeps <= 0:
            raise ValueError(f"n_substeps must be positive, got {n_substeps}.")
        if control is not None:
            control_array = np.asarray(control, dtype=float)
            if control_array.shape != self._motor_position.shape:
                raise ValueError(
                    f"Expected control with shape {self._motor_position.shape}, got {control_array.shape}."
                )
            self._motor_position = control_array.copy()
        self._time += float(n_substeps) * self.config.timestep
        self._state = self._compute_state()
        return self._state

    def get_state(self) -> BackendState:
        return self._state

    def _compute_state(self) -> BackendState:
        tendon_delta = motor_position_to_tendon_delta(self._motor_position, self.motor_params)
        q_est = physical_tendon_delta_to_q(tendon_delta, self.params, self.physical_tendons)
        fk = forward_kinematics(
            q_est,
            self.params,
            samples_per_segment=self.config.samples_per_segment,
        )
        return BackendState(
            time=self._time,
            tip_pose=fk.tip_pose.copy(),
            segment_poses=np.asarray(fk.segment_poses, dtype=float).copy(),
            qpos=q_est.copy(),
            qvel=np.zeros_like(q_est),
            tendon_length=tendon_delta.copy(),
            tendon_velocity=np.zeros_like(tendon_delta),
            actuator_force=np.zeros_like(tendon_delta),
        )


def load_analytic_backend_config(path: str | Path) -> AnalyticBackendConfig:
    """Load the PCC analytic-backend YAML."""

    config_path = Path(path).resolve()
    raw = load_yaml(config_path)
    backend = raw.get("backend")
    if backend != "pcc":
        raise ValueError(f"backend must be 'pcc', got {backend!r}.")
    model = _section(raw, "model")
    integration = _section(model, "integration")
    runtime = raw.get("runtime", {})
    if not isinstance(runtime, dict):
        raise ValueError("runtime must be a mapping when provided.")
    robot_config_path = _resolve_path(
        config_path,
        raw.get("robot_config_path", "robot_3seg.yaml"),
    )
    if not robot_config_path.is_file():
        raise FileNotFoundError(f"Robot config file does not exist: {robot_config_path}")
    samples_per_segment = int(integration.get("samples_per_segment", 21))
    if samples_per_segment < 2:
        raise ValueError(
            "model.integration.samples_per_segment must be at least 2, "
            f"got {samples_per_segment}."
        )
    timestep = float(runtime.get("timestep", 1.0))
    if timestep <= 0.0:
        raise ValueError(f"runtime.timestep must be positive, got {timestep}.")
    return AnalyticBackendConfig(
        path=config_path,
        robot_config_path=robot_config_path,
        samples_per_segment=samples_per_segment,
        timestep=timestep,
    )


def _resolve_path(config_path: Path, raw_path: object) -> Path:
    path = Path(str(raw_path))
    if path.is_absolute():
        return path
    parent_candidate = (config_path.parent / path).resolve()
    if parent_candidate.exists():
        return parent_candidate
    cwd_candidate = path.resolve()
    if cwd_candidate.exists():
        return cwd_candidate
    return parent_candidate


def _section(values: dict[str, Any], name: str) -> dict[str, Any]:
    section = values.get(name)
    if not isinstance(section, dict):
        raise ValueError(f"Expected section {name!r} to be a mapping.")
    return section

