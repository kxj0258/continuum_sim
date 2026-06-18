"""Discrete DMP-style trajectory generation for task-space waypoints."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class DMPRollout:
    """Generated DMP samples."""

    time: np.ndarray
    position: np.ndarray
    velocity: np.ndarray


@dataclass
class DiscreteDMP:
    """Small discrete DMP with Gaussian basis forcing terms."""

    basis_count: int = 24
    samples: int = 100
    alpha_z: float = 25.0
    beta_z: float = 6.25
    alpha_x: float = 4.0
    weights: np.ndarray | None = None
    demo_start: np.ndarray | None = None
    demo_goal: np.ndarray | None = None
    demo_tau: float = 1.0

    def imitate(self, time: np.ndarray, trajectory: np.ndarray) -> "DiscreteDMP":
        """Fit forcing weights from a demonstrated Cartesian trajectory."""

        time_array = _as_time(time)
        trajectory_array = _as_trajectory(trajectory)
        if time_array.shape[0] != trajectory_array.shape[0]:
            raise ValueError("time and trajectory must have the same sample count.")
        if self.basis_count <= 0:
            raise ValueError("basis_count must be positive.")
        if self.samples < 2:
            raise ValueError("samples must be at least 2.")

        tau = float(time_array[-1] - time_array[0])
        phase = self._phase((time_array - time_array[0]) / tau)
        features = self._basis_features(phase)
        velocity = np.gradient(trajectory_array, time_array, axis=0)
        acceleration = np.gradient(velocity, time_array, axis=0)
        start = trajectory_array[0].copy()
        goal = trajectory_array[-1].copy()
        scale = _safe_scale(goal - start)
        target_forcing = (
            tau**2 * acceleration
            - self.alpha_z * (self.beta_z * (goal[None, :] - trajectory_array) - tau * velocity)
        ) / scale[None, :]
        self.weights = np.linalg.lstsq(features, target_forcing, rcond=None)[0]
        self.demo_start = start
        self.demo_goal = goal
        self.demo_tau = tau
        return self

    def rollout(self, start_pos: np.ndarray, goal_pos: np.ndarray, tau: float = 1.0) -> DMPRollout:
        """Generate a trajectory between ``start_pos`` and ``goal_pos``."""

        if self.weights is None or self.demo_start is None or self.demo_goal is None:
            raise ValueError("DMP must be fitted with imitate() before rollout().")
        if tau <= 0.0:
            raise ValueError("tau must be positive.")
        start = _as_position(start_pos, "start_pos")
        goal = _as_position(goal_pos, "goal_pos")
        times = np.linspace(0.0, tau, self.samples)
        phase = self._phase(times / tau)
        forcing = self._basis_features(phase) @ self.weights
        forcing *= _safe_scale(goal - start)[None, :]

        position = np.zeros((self.samples, 3), dtype=float)
        velocity = np.zeros((self.samples, 3), dtype=float)
        position[0] = start
        y = start.copy()
        z = np.zeros(3, dtype=float)
        dt = tau / float(self.samples - 1)
        for index in range(1, self.samples):
            z_dot = (
                self.alpha_z * (self.beta_z * (goal - y) - z)
                + forcing[index - 1]
            ) / tau
            z = z + z_dot * dt
            y = y + (z / tau) * dt
            position[index] = y
            velocity[index] = z / tau
        position[0] = start
        position[-1] = goal
        velocity[:] = np.gradient(position, times, axis=0)
        return DMPRollout(time=times, position=position, velocity=velocity)

    def _phase(self, normalized_time: np.ndarray) -> np.ndarray:
        return np.exp(-self.alpha_x * np.asarray(normalized_time, dtype=float))

    def _basis_features(self, phase: np.ndarray) -> np.ndarray:
        centers = np.exp(-self.alpha_x * np.linspace(0.0, 1.0, self.basis_count))
        widths = np.full(self.basis_count, float(self.basis_count**1.5), dtype=float)
        psi = np.exp(-widths[None, :] * (phase[:, None] - centers[None, :]) ** 2)
        denom = np.sum(psi, axis=1, keepdims=True)
        denom = np.maximum(denom, 1.0e-12)
        return phase[:, None] * psi / denom


def load_demonstration(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load a DMP demonstration from CSV/whitespace text or NPZ."""

    demo_path = Path(path)
    if demo_path.suffix.lower() == ".npz":
        data = np.load(demo_path)
        return _as_time(data["time"]), _as_trajectory(data["trajectory"])
    try:
        raw = np.loadtxt(demo_path, delimiter=",")
    except ValueError:
        raw = np.loadtxt(demo_path)
    array = np.asarray(raw, dtype=float)
    if array.ndim != 2 or array.shape[1] not in (3, 4):
        raise ValueError("DMP demo must have shape (N, 3) or (N, 4).")
    if array.shape[1] == 4:
        return _as_time(array[:, 0]), _as_trajectory(array[:, 1:])
    return np.linspace(0.0, 1.0, array.shape[0]), _as_trajectory(array)


def _as_time(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or array.shape[0] < 2:
        raise ValueError("time must be a 1D array with at least two samples.")
    if np.any(np.diff(array) <= 0.0):
        raise ValueError("time must be strictly increasing.")
    return array


def _as_trajectory(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 2 or array.shape[1] != 3 or array.shape[0] < 2:
        raise ValueError(f"trajectory must have shape (N, 3), got {array.shape}.")
    return array


def _as_position(values: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.shape != (3,):
        raise ValueError(f"Expected {name} with shape (3,), got {array.shape}.")
    return array.copy()


def _safe_scale(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    return np.where(np.abs(array) > 1.0e-9, array, 1.0)
