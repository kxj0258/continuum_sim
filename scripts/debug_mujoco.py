"""Interactively debug a scenario-composed MuJoCo tendon system."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from continuum_sim.application.application import SimulationApplication  # noqa: E402
from continuum_sim.backends.mujoco_system_backend import MujocoSystemBackend  # noqa: E402
from continuum_sim.runtime.hooks import _configure_mujoco_viewer  # noqa: E402
from continuum_sim.visualization.mujoco_system_debug_viewer import (  # noqa: E402
    MujocoSystemDebugViewer,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "scenario",
        type=Path,
        help="Scenario YAML containing a MuJoCo backend.",
    )
    parser.add_argument(
        "--panel-only",
        action="store_true",
        help="Open only the Matplotlib controls without the MuJoCo 3D viewer.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    application = SimulationApplication.from_yaml(args.scenario)
    backend = application.loop.backend
    if not isinstance(backend, MujocoSystemBackend):
        raise ValueError("debug_mujoco.py requires a scenario with backend.type='mujoco'.")

    debug_viewer = None
    if args.panel_only:
        debug_viewer = MujocoSystemDebugViewer(
            backend,
            control_dt_s=application.config.runtime.controller_dt_s,
            n_substeps=application.config.runtime.n_substeps,
        )
        try:
            debug_viewer.show()
        finally:
            debug_viewer.close()
        return 0

    import mujoco.viewer

    with mujoco.viewer.launch_passive(
        backend.physics.model,
        backend.physics.data,
    ) as sim_viewer:
        _configure_mujoco_viewer(sim_viewer, backend.config)

        def sync_sim_viewer(_state) -> None:
            if sim_viewer.is_running():
                sim_viewer.sync()

        debug_viewer = MujocoSystemDebugViewer(
            backend,
            control_dt_s=application.config.runtime.controller_dt_s,
            n_substeps=application.config.runtime.n_substeps,
            state_update_callback=sync_sim_viewer,
        )
        try:
            debug_viewer.show()
        finally:
            debug_viewer.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
