"""Backward-compatible control import path for CBF-QP kinematics utilities."""

from continuum_sim.kinematics.cbf_qp import (
    CBFQPConfig,
    cbf_lower_bound,
    solve_cbf_qp_velocity,
)

__all__ = [
    "CBFQPConfig",
    "cbf_lower_bound",
    "solve_cbf_qp_velocity",
]
