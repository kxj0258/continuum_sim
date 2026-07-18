"""Run the maintained main scenario YAML files sequentially.

Each scenario is executed through ``scripts/run_scenario.py`` in a child
process.  The next scenario starts only after the previous child process exits,
which means scenario hooks and artifact export have finished for that run.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from time import perf_counter


PROJECT_ROOT = Path(__file__).resolve().parents[1]

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
    failures: list[tuple[Path, int]] = []
    started = perf_counter()
    for index, scenario_path in enumerate(scenario_paths, start=1):
        print(f"\n[{index}/{len(scenario_paths)}] {scenario_path}")
        command = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "run_scenario.py"),
            str(scenario_path),
        ]
        scenario_started = perf_counter()
        completed = subprocess.run(command, cwd=PROJECT_ROOT)
        elapsed = perf_counter() - scenario_started
        print(f"elapsed_s: {elapsed:.3f}")
        if completed.returncode != 0:
            failures.append((scenario_path, completed.returncode))
            print(
                f"failed: return_code={completed.returncode}",
                file=sys.stderr,
            )
            if args.stop_on_error:
                break

    print("\nsummary:")
    print(f"total_scenarios: {len(scenario_paths)}")
    print(f"failures: {len(failures)}")
    print(f"elapsed_s: {perf_counter() - started:.3f}")
    for path, return_code in failures:
        print(f"- {path}: return_code={return_code}")
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
    return parser.parse_args(argv)


def _scenario_paths(paths: list[Path]) -> list[Path]:
    selected = paths if paths else list(MAIN_SCENARIOS)
    return [
        path if path.is_absolute() else PROJECT_ROOT / path
        for path in selected
    ]


if __name__ == "__main__":
    raise SystemExit(main())
