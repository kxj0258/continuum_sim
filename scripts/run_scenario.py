"""Run one reproducible continuum-system scenario."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from continuum_sim.application import SimulationApplication  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scenario", type=Path)
    args = parser.parse_args(argv)
    application = SimulationApplication.from_yaml(args.scenario)
    result = application.run()
    print(f"scenario: {application.config.name}")
    print(f"states: {len(result.states)}")
    print(f"commands: {len(result.commands)}")
    print(f"stopped_early: {result.stopped_early}")
    print(f"stop_reason: {result.metadata.get('stop_reason', '')}")
    if application.last_artifacts is not None:
        print(f"run_dir: {application.last_artifacts.run_dir}")
    recorder = application.hooks_by_name.get("recorder")
    if recorder is not None and recorder.tracking_error_m:
        import numpy as np

        errors = np.asarray(recorder.tracking_error_m, dtype=float)
        errors = errors[np.isfinite(errors)]
        if errors.size:
            print(f"final_error_m: {errors[-1]:.6e}")
            print(f"mean_error_m: {np.mean(errors):.6e}")
            print(f"max_error_m: {np.max(errors):.6e}")
        achieved = np.asarray(
            getattr(recorder, "achieved_waypoint_error_m", ()),
            dtype=float,
        )
        achieved = achieved[np.isfinite(achieved)]
        if achieved.size:
            print(f"final_achieved_error_m: {achieved[-1]:.6e}")
            print(f"mean_achieved_error_m: {np.mean(achieved):.6e}")
            print(f"max_achieved_error_m: {np.max(achieved):.6e}")
    for name, hook in application.hooks_by_name.items():
        samples = getattr(hook, "samples", None)
        if samples is not None:
            print(f"{name}_samples: {len(samples)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
