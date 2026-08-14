"""Reusable runtime profile transforms for scenario applications."""

from __future__ import annotations

from dataclasses import replace

from continuum_sim.application.scenario import ScenarioConfig


def with_windowless_batch_profile(config: ScenarioConfig) -> ScenarioConfig:
    """Disable interactive windows while keeping artifact recording enabled."""

    return replace(
        config,
        hooks=replace(
            config.hooks,
            viewer="none",
            keep_viewer_open=False,
            show_live_tendon_panel=False,
            show_live_task_error_panel=False,
            show_live_force_panel=False,
            show_live_diagnostics_panel=False,
            show_observer_camera=False,
        ),
    )
