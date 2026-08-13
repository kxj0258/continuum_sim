"""Launch curvature-only dual-arm manual control and status windows."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from continuum_sim.visualization.manual_control_app import (  # noqa: E402
    run_manual_curvature_control,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "scenario",
        nargs="?",
        type=Path,
        default=PROJECT_ROOT / "configs" / "scenarios" / "mujoco_manual_control.yaml",
        help="Scenario used to compose the MuJoCo model.",
    )
    parser.add_argument("--panel-fps", type=float, default=15.0)
    parser.add_argument("--status-fps", type=float, default=5.0)
    parser.add_argument("--viewer-fps", type=float, default=15.0)
    parser.add_argument("--show-tendon-monitor", action="store_true")
    args = parser.parse_args(argv)
    run_manual_curvature_control(
        args.scenario,
        panel_fps=args.panel_fps,
        status_fps=args.status_fps,
        viewer_fps=args.viewer_fps,
        show_tendon_monitor=args.show_tendon_monitor,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
