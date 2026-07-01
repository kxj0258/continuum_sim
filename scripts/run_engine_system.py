"""Run a configured single- or dual-spatial-arm engine system headlessly."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from continuum_sim.runtime.engine_system_runtime import run_engine_system  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to a single- or dual-spatial-arm engine system YAML.",
    )
    args = parser.parse_args(argv)
    result = run_engine_system(args.config)
    print(f"states: {len(result.states)}")
    print(f"commands: {len(result.commands)}")
    print(f"stopped_early: {result.stopped_early}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

