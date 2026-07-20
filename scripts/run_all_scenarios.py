"""Run scenario YAML tasks sequentially with an unattended default profile."""

from __future__ import annotations

import argparse
from dataclasses import replace
import fnmatch
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
    ScenarioConfig,
    load_scenario_config,
)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    scenario_paths = _discover_scenarios(
        args.scenarios_dir,
        include_idle=args.include_idle,
        pattern=args.pattern,
    )
    if not scenario_paths:
        print("No scenario YAML files matched.")
        return 1

    print(f"scenario_count: {len(scenario_paths)}")
    failures: list[tuple[Path, str]] = []
    started = perf_counter()
    for index, scenario_path in enumerate(scenario_paths, start=1):
        print(f"\n[{index}/{len(scenario_paths)}] {scenario_path}")
        scenario_started = perf_counter()
        try:
            config = load_scenario_config(scenario_path)
            if args.headless:
                config = with_windowless_batch_profile(config)
            if args.disable_artifacts:
                config = replace(
                    config,
                    artifacts=replace(config.artifacts, enabled=False),
                )
            application = SimulationApplication.from_config(config)
            result = application.run()
            elapsed = perf_counter() - scenario_started
            print(f"scenario: {config.name}")
            print(f"task: {config.task.type}")
            print(f"backend: {config.backend.type}")
            print(f"states: {len(result.states)}")
            print(f"commands: {len(result.commands)}")
            print(f"stopped_early: {result.stopped_early}")
            print(f"stop_reason: {result.metadata.get('stop_reason', '')}")
            if application.last_artifacts is not None:
                print(f"run_dir: {application.last_artifacts.run_dir}")
            _print_tracking_metrics(config, result)
            print(f"elapsed_s: {elapsed:.3f}")
        except Exception as exc:  # noqa: BLE001 - batch runner reports all failures.
            elapsed = perf_counter() - scenario_started
            message = f"{type(exc).__name__}: {exc}"
            failures.append((scenario_path, message))
            print(f"failed: {message}")
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
        "--scenarios-dir",
        type=Path,
        default=PROJECT_ROOT / "configs" / "scenarios",
        help="Directory containing scenario YAML files.",
    )
    parser.add_argument(
        "--pattern",
        default="*.yaml",
        help="Filename glob used after discovery, for example '*mujoco*.yaml'.",
    )
    parser.add_argument(
        "--include-idle",
        action="store_true",
        help="Also run idle/view/smoke scenarios. Defaults to task scenarios only.",
    )
    parser.add_argument(
        "--no-headless",
        dest="headless",
        action="store_false",
        help=(
            "Keep each scenario's configured viewer, live panels, and observer "
            "camera window."
        ),
    )
    parser.add_argument(
        "--disable-artifacts",
        action="store_true",
        help="Run scenarios without saving artifact directories.",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop after the first scenario failure. Defaults to continuing.",
    )
    parser.set_defaults(headless=True)
    return parser.parse_args(argv)


def _discover_scenarios(
    scenarios_dir: Path,
    *,
    include_idle: bool,
    pattern: str,
) -> list[Path]:
    paths = sorted(Path(scenarios_dir).resolve().glob("*.yaml"))
    selected: list[Path] = []
    for path in paths:
        if not fnmatch.fnmatch(path.name, pattern):
            continue
        config = load_scenario_config(path)
        if not include_idle and config.task.type == "idle":
            continue
        selected.append(path)
    return selected


def _print_tracking_metrics(config: ScenarioConfig, result: object) -> None:
    if not result.commands:
        return
    errors = [
        command.metadata.get("tracking_error_m")
        for command in result.commands
        if command.metadata.get("tracking_error_m") is not None
    ]
    if not errors:
        return
    finite = np.asarray(errors, dtype=float)
    finite = finite[np.isfinite(finite)]
    if not finite.size:
        return
    print(f"final_error_m: {finite[-1]:.6e}")
    print(f"mean_error_m: {np.mean(finite):.6e}")
    print(f"max_error_m: {np.max(finite):.6e}")
    if config.task.type != "tracking":
        print("tracking_error_source: command metadata")


if __name__ == "__main__":
    raise SystemExit(main())
