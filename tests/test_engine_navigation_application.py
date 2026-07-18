from continuum_sim.application import SimulationApplication
from continuum_sim.control.staged_engine_navigation import (
    StagedEngineNavigationController,
)


def test_dual_engine_navigation_composes_staged_controller() -> None:
    application = SimulationApplication.from_yaml(
        "configs/scenarios/engine_navigation.yaml"
    )

    assert application.config.task.type == "engine_navigation"
    assert isinstance(
        application.loop.controller,
        StagedEngineNavigationController,
    )
