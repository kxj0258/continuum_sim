"""Robot parameter and routing definitions."""

from continuum_sim.model.physical_tendon import (
    PhysicalTendonPath,
    load_physical_tendons_from_yaml,
)
from continuum_sim.model.robot_params import SegmentParams, ThreeSegmentRobotParams
from continuum_sim.model.tendon_routing import TendonRouting
from continuum_sim.model.tendon_coupling import (
    build_coupling_matrix,
    coupling_diagnostics,
    physical_tendon_delta_to_q,
    q_to_physical_tendon_delta,
)

__all__ = [
    "PhysicalTendonPath",
    "SegmentParams",
    "TendonRouting",
    "ThreeSegmentRobotParams",
    "build_coupling_matrix",
    "coupling_diagnostics",
    "load_physical_tendons_from_yaml",
    "physical_tendon_delta_to_q",
    "q_to_physical_tendon_delta",
]
