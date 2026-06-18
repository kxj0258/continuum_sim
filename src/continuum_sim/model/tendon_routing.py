"""Tendon routing geometry for one constant-curvature segment."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from continuum_sim.utils.math_utils import deg_to_rad


@dataclass(frozen=True)
class TendonRouting:
    """Cross-section tendon layout for one segment.

    Angles are measured in the segment frame. The project convention places
    the local x axis toward the first tendon, so the default layout is
    [0, 120, 240] degrees.
    """

    angles_deg: tuple[float, float, float] = (0.0, 120.0, 240.0)
    radial_offset: float = 0.005

    @property
    def angles_rad(self) -> np.ndarray:
        return deg_to_rad(self.angles_deg)

    @property
    def xy_offsets(self) -> np.ndarray:
        angles = self.angles_rad
        return self.radial_offset * np.column_stack((np.cos(angles), np.sin(angles)))
