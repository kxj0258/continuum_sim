"""Run the maintained main scenario YAML files sequentially."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from time import perf_counter

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from continuum_sim.application import SimulationApplication  # noqa: E402
from continuum_sim.application.runtime_profiles import (  # noqa: E402
    with_windowless_batch_profile,
)
from continuum_sim.application.scenario import (  # noqa: E402
    load_scenario_config,
)

MAIN_SCENARIOS = (
    Path("configs/scenarios/mujoco_tracking.yaml"),
    Path("configs/scenarios/mujoco_navigation.yaml"),
    Path("configs/scenarios/engine_navigation.yaml"),
    Path("configs/scenarios/mujoco_wiping.yaml"),
    Path("configs/scenarios/mujoco_point_servo.yaml"),
)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    scenario_paths = _scenario_paths(args.scenarios)
    missing = [path for path in scenario_paths if not path.is_file()]
    if missing:
        for path in missing:
            print(f"missing scenario: {path}", file=sys.stderr)
        return 2

    print(f"scenario_count: {len(scenario_paths)}")
    failures: list[tuple[Path, str]] = []
    started = perf_counter()
    for index, scenario_path in enumerate(scenario_paths, start=1):
        print(f"\n[{index}/{len(scenario_paths)}] {scenario_path}")
        scenario_started = perf_counter()
        try:
            config = load_scenario_config(scenario_path)
            if not args.with_windows:
                config = with_windowless_batch_profile(config)
            application = SimulationApplication.from_config(config)
            result = application.run()
            elapsed = perf_counter() - scenario_started
            _print_result(application, result)
            print(f"elapsed_s: {elapsed:.3f}")
        except Exception as exc:  # noqa: BLE001 - batch runner reports all failures.
            elapsed = perf_counter() - scenario_started
            message = f"{type(exc).__name__}: {exc}"
            failures.append((scenario_path, message))
            print(f"failed: {message}", file=sys.stderr)
            print(f"elapsed_s: {elapsed:.3f}")
            if args.stop_on_error:
                break

    print("\nsummary:")
    print(f"total_scenarios: {len(scenario_paths)}")
    print(f"failures: {len(failures)}")
    print(f"elapsed_s: {perf_counter() - started:.3f}")
    for path, message in failures:
        print(f"- {path}: {message}")
    return 1 if failures else 0


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "scenarios",
        nargs="*",
        type=Path,
        help=(
            "Optional scenario YAML files to run instead of the default main "
            "scenario list."
        ),
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop after the first failed scenario. Defaults to continuing.",
    )
    parser.add_argument(
        "--with-windows",
        action="store_true",
        help=(
            "Keep each scenario's configured viewer, live panels, and observer "
            "camera window. Defaults to a windowless batch profile."
        ),
    )
    return parser.parse_args(argv)


def _scenario_paths(paths: list[Path]) -> list[Path]:
    selected = paths if paths else list(MAIN_SCENARIOS)
    return [
        path if path.is_absolute() else PROJECT_ROOT / path
        for path in selected
    ]


def _print_result(application: SimulationApplication, result: object) -> None:
    print(f"scenario: {application.config.name}")
    print(f"states: {len(result.states)}")
    print(f"commands: {len(result.commands)}")
    print(f"stopped_early: {result.stopped_early}")
    print(f"stop_reason: {result.metadata.get('stop_reason', '')}")
    if application.last_artifacts is not None:
        print(f"run_dir: {application.last_artifacts.run_dir}")
    recorder = application.hooks_by_name.get("recorder")
    if recorder is not None and recorder.tracking_error_m:
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


if __name__ == "__main__":
    raise SystemExit(main())
