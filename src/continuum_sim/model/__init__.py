"""Robot parameter and routing definitions."""

from continuum_sim.model.physical_tendon import (
    PhysicalTendonPath,
    load_physical_tendons_from_yaml,
)
from continuum_sim.model.dual_arm_robot import (
    DualArmRobotConfig,
    is_dual_arm_robot_config,
    load_dual_arm_robot_config,
)
from continuum_sim.model.hole_pattern import (
    TendonHole,
    TendonHolePattern,
    TendonHoleSiteGeneration,
    load_tendon_hole_pattern,
)
from continuum_sim.model.mobile_base_context import MobileBaseArmContext
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
    "DualArmRobotConfig",
    "MobileBaseArmContext",
    "SegmentParams",
    "TendonHole",
    "TendonHolePattern",
    "TendonHoleSiteGeneration",
    "TendonRouting",
    "ThreeSegmentRobotParams",
    "build_coupling_matrix",
    "coupling_diagnostics",
    "is_dual_arm_robot_config",
    "load_dual_arm_robot_config",
    "load_physical_tendons_from_yaml",
    "load_tendon_hole_pattern",
    "physical_tendon_delta_to_q",
    "q_to_physical_tendon_delta",
]
