"""Launch dual-arm manual controls, MuJoCo viewer, and observer camera."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from continuum_sim.visualization.manual_control_app import run_manual_control  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "scenario",
        nargs="?",
        type=Path,
        default=PROJECT_ROOT / "configs" / "scenarios" / "mujoco_manual_control.yaml",
        help="Scenario used to compose the MuJoCo model.",
    )
    parser.add_argument("--camera-fps", type=float, default=20.0)
    parser.add_argument("--curvature-step", type=float, default=0.5, metavar="1/M")
    args = parser.parse_args(argv)
    run_manual_control(
        args.scenario,
        camera_fps=args.camera_fps,
        curvature_step_1_per_m=args.curvature_step,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
