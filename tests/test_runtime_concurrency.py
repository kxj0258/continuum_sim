from __future__ import annotations

from threading import Event, get_ident

import pytest

from continuum_sim.runtime.concurrency import (
    AsyncLinePrinter,
    LatestValueSlot,
    MonotonicRateRunner,
    TimeRateGate,
)


def test_latest_value_slot_overwrites_unconsumed_values() -> None:
    slot = LatestValueSlot("initial")

    first_version = slot.publish("first")
    latest_version = slot.publish("latest")

    assert latest_version == first_version + 1
    assert slot.consume_after(first_version) == ("latest", latest_version)
    assert slot.consume_after(latest_version) is None


def test_time_rate_gate_skips_expired_deadlines_without_drifting() -> None:
    gate = TimeRateGate(0.05, start_s=0.0)

    assert gate.due(0.00) is False
    assert gate.due(0.049) is False
    assert gate.due(0.05) is True
    assert gate.due(0.16) is True
    assert gate.next_deadline_s == pytest.approx(0.20)
    assert gate.due(0.199) is False
    assert gate.due(0.20) is True


def test_monotonic_rate_runner_exposes_callback_failure() -> None:
    called = Event()

    def fail() -> None:
        called.set()
        raise RuntimeError("control failed")

    runner = MonotonicRateRunner(0.01, fail, name="test-rate-runner")
    runner.start()
    assert called.wait(1.0)
    runner.stop()

    with pytest.raises(RuntimeError, match="control failed"):
        runner.raise_if_failed()


def test_async_line_printer_runs_sink_outside_caller_and_drains() -> None:
    caller_thread_id = get_ident()
    received: list[tuple[str, int]] = []

    printer = AsyncLinePrinter(
        lambda line: received.append((line, get_ident())),
        name="test-line-printer",
    )
    printer.write("first")
    printer.write("second")
    printer.close(drain=True)

    assert [line for line, _ in received] == ["first", "second"]
    assert all(thread_id != caller_thread_id for _, thread_id in received)
    printer.close(drain=True)

