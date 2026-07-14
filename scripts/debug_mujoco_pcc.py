"""Interactively compare PCC geometry with MuJoCo tip sites under tendon drive."""

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
from continuum_sim.visualization.mujoco_pcc_debug import (  # noqa: E402
    MujocoPccOverlay,
    PccMujocoSampleRecorder,
    compare_pcc_mujoco_state,
    default_pcc_debug_csv_path,
    format_pcc_mujoco_diagnostics,
)
from continuum_sim.visualization.mujoco_system_debug_viewer import (  # noqa: E402
    MujocoSystemDebugViewer,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "scenario",
        type=Path,
        help="Scenario YAML containing a single- or dual-arm MuJoCo backend.",
    )
    parser.add_argument(
        "--samples-per-segment",
        type=int,
        default=21,
        help="PCC centerline samples per segment (default: 21).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help=(
            "CSV path used by Save CSV; defaults to a timestamped file under "
            "output/diagnostics."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.samples_per_segment < 2:
        raise ValueError("--samples-per-segment must be at least 2.")

    application = SimulationApplication.from_yaml(args.scenario)
    backend = application.loop.backend
    if not isinstance(backend, MujocoSystemBackend):
        raise ValueError(
            "debug_mujoco_pcc.py requires a scenario with backend.type='mujoco'."
        )

    import mujoco
    import mujoco.viewer
    from matplotlib.widgets import Button

    recorder = PccMujocoSampleRecorder(
        scenario_name=application.config.name,
    )
    output_path = (
        args.output.resolve()
        if args.output is not None
        else default_pcc_debug_csv_path(PROJECT_ROOT)
    )
    cached_state_identity: int | None = None
    cached_comparisons = None
    status = f"CSV pending: {output_path}"

    def comparisons_for(state):
        nonlocal cached_state_identity, cached_comparisons
        state_identity = id(state)
        if state_identity != cached_state_identity:
            cached_comparisons = compare_pcc_mujoco_state(
                backend.assembly,
                state,
                samples_per_segment=args.samples_per_segment,
            )
            cached_state_identity = state_identity
        return cached_comparisons

    with mujoco.viewer.launch_passive(
        backend.physics.model,
        backend.physics.data,
    ) as sim_viewer:
        _configure_mujoco_viewer(sim_viewer, backend.config)
        overlay = MujocoPccOverlay(sim_viewer, mujoco)

        def diagnostic_text(state, control_space: str) -> str:
            comparisons = comparisons_for(state)
            recorder.record(
                state,
                comparisons,
                control_space=control_space,
            )
            return format_pcc_mujoco_diagnostics(
                state,
                comparisons,
                control_space=control_space,
                sample_count=recorder.sample_count,
                status=status,
            )

        def update_sim_viewer(state) -> None:
            if not sim_viewer.is_running():
                return
            overlay.update(comparisons_for(state))
            sim_viewer.sync()

        debug_viewer = MujocoSystemDebugViewer(
            backend,
            control_dt_s=application.config.runtime.controller_dt_s,
            n_substeps=application.config.runtime.n_substeps,
            state_update_callback=update_sim_viewer,
            diagnostic_text_provider=diagnostic_text,
        )
        debug_viewer.panel.info_ax.set_position((0.68, 0.27, 0.29, 0.25))
        debug_viewer.panel._info_text.set_fontsize(7.0)
        save_button = Button(
            debug_viewer.panel.fig.add_axes((0.54, 0.19, 0.12, 0.04)),
            "Save CSV",
        )

        def save_samples(_event) -> None:
            nonlocal status
            try:
                saved_path = recorder.save_csv(output_path)
            except Exception as exc:  # Display interactive I/O failures in the UI.
                status = f"CSV save failed: {type(exc).__name__}: {exc}"
            else:
                status = f"CSV saved: {saved_path}"
            debug_viewer.refresh()

        save_button.on_clicked(save_samples)
        try:
            debug_viewer.show()
        finally:
            debug_viewer.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
