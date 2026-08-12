from __future__ import annotations

from continuum_sim.utils.runtime_timing import RuntimeTimingReporter


def test_runtime_timing_reports_input_latency_on_next_completed_cycle() -> None:
    clock = _Clock()
    lines: list[str] = []
    timing = RuntimeTimingReporter(
        report_interval_s=0.5,
        clock=clock,
        printer=lines.append,
    )

    timing.mark_input("executor:S1:+kx")
    clock.advance(0.03)
    timing.start_cycle()
    with timing.measure("control.prepare"):
        clock.advance(0.01)
    with timing.measure("mujoco.steps"):
        clock.advance(0.02)
    timing.finish_cycle()

    assert len(lines) == 1
    assert "[manual-timing input]" in lines[0]
    assert "executor:S1:+kx" in lines[0]
    assert "callback->cycle=30.000 ms" in lines[0]
    assert "callback->complete=60.000 ms" in lines[0]


def test_runtime_timing_prints_batched_stage_average_and_max() -> None:
    clock = _Clock()
    lines: list[str] = []
    timing = RuntimeTimingReporter(
        report_interval_s=0.5,
        clock=clock,
        printer=lines.append,
    )

    for _ in range(5):
        timing.start_cycle()
        with timing.measure("control.prepare"):
            clock.advance(0.01)
        with timing.measure("mujoco.steps"):
            clock.advance(0.02)
        clock.advance(0.07)
        timing.finish_cycle()

    assert len(lines) == 1
    assert "[manual-timing]" in lines[0]
    assert "cycles=5" in lines[0]
    assert "frequency=10.00 Hz" in lines[0]
    assert "cycle.total=100.000/100.000 ms" in lines[0]
    assert "control.prepare=10.000/10.000 ms" in lines[0]
    assert "mujoco.steps=20.000/20.000 ms" in lines[0]


def test_runtime_timing_coalesces_multiple_inputs_before_one_cycle() -> None:
    clock = _Clock()
    lines: list[str] = []
    timing = RuntimeTimingReporter(clock=clock, printer=lines.append)

    timing.mark_input("executor:S1:+kx")
    clock.advance(0.01)
    timing.mark_input("executor:S1:+kx")
    clock.advance(0.01)
    timing.start_cycle()
    clock.advance(0.02)
    timing.finish_cycle()

    assert "events=2" in lines[0]
    assert "callback->cycle=20.000 ms" in lines[0]


def test_runtime_timing_does_not_assign_late_input_to_finished_cycle() -> None:
    clock = _Clock()
    lines: list[str] = []
    timing = RuntimeTimingReporter(clock=clock, printer=lines.append)

    timing.mark_input("executor:S1:+kx")
    timing.finish_cycle()

    assert lines == []

    clock.advance(0.02)
    timing.start_cycle()
    clock.advance(0.005)
    timing.finish_cycle()

    assert "callback->cycle=20.000 ms" in lines[0]


def test_runtime_timing_reset_discards_startup_measurements() -> None:
    clock = _Clock()
    lines: list[str] = []
    timing = RuntimeTimingReporter(
        report_interval_s=0.1,
        clock=clock,
        printer=lines.append,
    )

    timing.record("camera.startup", 2.0)
    timing.reset()
    timing.start_cycle()
    with timing.measure("mujoco.steps"):
        clock.advance(0.1)
    timing.finish_cycle()

    assert "camera.startup" not in lines[0]
    assert "mujoco.steps=100.000/100.000 ms" in lines[0]


class _Clock:
    def __init__(self) -> None:
        self.now_s = 0.0

    def __call__(self) -> float:
        return self.now_s

    def advance(self, seconds: float) -> None:
        self.now_s += float(seconds)
