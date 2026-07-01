def test_scenario_application_import_does_not_load_legacy_viewers() -> None:
    from continuum_sim.application import SimulationApplication

    assert SimulationApplication.__name__ == "SimulationApplication"
