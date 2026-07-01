"""Scenario-native artifact exports.

Legacy CLI artifact helpers remain available from
``continuum_sim.io.run_artifacts`` but are intentionally not imported here.
Keeping this facade narrow prevents the Scenario Application from loading
the retired CLI dependency graph.
"""

from continuum_sim.io.scenario_artifacts import (
    ScenarioArtifactPaths,
    save_scenario_artifacts,
)

__all__ = [
    "ScenarioArtifactPaths",
    "save_scenario_artifacts",
]
