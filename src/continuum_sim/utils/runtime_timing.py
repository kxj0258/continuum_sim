"""Low-overhead rolling timing diagnostics for interactive runtimes."""

from __future__ import annotations

from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass
from threading import RLock
from time import perf_counter
from typing import Callable, Iterator


def _print_line(line: str) -> None:
    print(line, flush=True)


@dataclass
class _StageStats:
    total_s: float = 0.0
    maximum_s: float = 0.0
    count: int = 0

    def add(self, duration_s: float) -> None:
        self.total_s += duration_s
        self.maximum_s = max(self.maximum_s, duration_s)
        self.count += 1


class RuntimeTimingReporter:
    """Aggregate runtime stage durations and periodically print one summary."""

    def __init__(
        self,
        *,
        report_interval_s: float = 0.5,
        clock: Callable[[], float] = perf_counter,
        printer: Callable[[str], object] = _print_line,
    ) -> None:
        if report_interval_s <= 0.0:
            raise ValueError("report_interval_s must be positive.")
        self.report_interval_s = float(report_interval_s)
        self._clock = clock
        self._printer = printer
        self._lock = RLock()
        self._window_started_s: float | None = None
        self._stages: OrderedDict[str, _StageStats] = OrderedDict()
        self._cycle_started_s: float | None = None
        self._previous_cycle_started_s: float | None = None
        self._previous_cycle_finished_s: float | None = None
        self._cycle_count = 0
        self._pending_input_started_s: float | None = None
        self._pending_input_label = ""
        self._pending_input_count = 0
        self._pending_input_to_cycle_s: float | None = None

    def reset(self) -> None:
        """Discard startup samples and begin a fresh reporting window."""

        with self._lock:
            self._window_started_s = None
            self._stages.clear()
            self._cycle_started_s = None
            self._previous_cycle_started_s = None
            self._previous_cycle_finished_s = None
            self._cycle_count = 0
            self._pending_input_started_s = None
            self._pending_input_label = ""
            self._pending_input_count = 0
            self._pending_input_to_cycle_s = None

    @contextmanager
    def measure(self, stage: str) -> Iterator[None]:
        """Measure one named stage and add it to the active reporting window."""

        started_s = float(self._clock())
        try:
            yield
        finally:
            self.record(stage, float(self._clock()) - started_s)

    def record(self, stage: str, duration_s: float) -> None:
        """Record a previously measured non-negative duration."""

        duration = max(0.0, float(duration_s))
        with self._lock:
            stats = self._stages.setdefault(str(stage), _StageStats())
            stats.add(duration)

    def mark_input(self, label: str) -> None:
        """Start latency tracking for input consumed by the next control cycle."""

        with self._lock:
            if self._pending_input_started_s is None:
                self._pending_input_started_s = float(self._clock())
            self._pending_input_label = str(label)
            self._pending_input_count += 1

    def start_cycle(self) -> None:
        """Mark the start of one complete control-and-render cycle."""

        now_s = float(self._clock())
        with self._lock:
            if self._window_started_s is None:
                self._window_started_s = now_s
            if self._previous_cycle_started_s is not None:
                self.record("cycle.interval", now_s - self._previous_cycle_started_s)
            if self._previous_cycle_finished_s is not None:
                self.record("control.wait", now_s - self._previous_cycle_finished_s)
            self._previous_cycle_started_s = now_s
            self._cycle_started_s = now_s
            if self._pending_input_started_s is not None:
                self._pending_input_to_cycle_s = now_s - self._pending_input_started_s

    def finish_cycle(self) -> None:
        """Finish a cycle, print pending input latency, and report when due."""

        now_s = float(self._clock())
        with self._lock:
            if self._cycle_started_s is not None:
                self.record("cycle.total", now_s - self._cycle_started_s)
                self._cycle_count += 1
                self._cycle_started_s = None
                self._previous_cycle_finished_s = now_s
            self._print_pending_input(now_s)
            window_started_s = (
                now_s if self._window_started_s is None else self._window_started_s
            )
            elapsed_s = now_s - window_started_s
            if elapsed_s + 1.0e-12 >= self.report_interval_s:
                self._print_summary(now_s, elapsed_s)

    def _print_pending_input(self, now_s: float) -> None:
        if (
            self._pending_input_started_s is None
            or self._pending_input_to_cycle_s is None
        ):
            return
        to_cycle_s = self._pending_input_to_cycle_s
        self._printer(
            "[manual-timing input] "
            f"event={self._pending_input_label} "
            f"events={self._pending_input_count} "
            f"callback->cycle={1000.0 * to_cycle_s:.3f} ms "
            "callback->complete="
            f"{1000.0 * (now_s - self._pending_input_started_s):.3f} ms"
        )
        self._pending_input_started_s = None
        self._pending_input_label = ""
        self._pending_input_count = 0
        self._pending_input_to_cycle_s = None

    def _print_summary(self, now_s: float, elapsed_s: float) -> None:
        frequency_hz = self._cycle_count / elapsed_s if elapsed_s > 0.0 else 0.0
        stage_text = " | ".join(
            f"{name}={1000.0 * stats.total_s / stats.count:.3f}/"
            f"{1000.0 * stats.maximum_s:.3f} ms"
            for name, stats in self._stages.items()
            if stats.count > 0
        )
        line = (
            f"[manual-timing] window={elapsed_s:.3f} s "
            f"cycles={self._cycle_count} frequency={frequency_hz:.2f} Hz"
        )
        if stage_text:
            line = f"{line} | avg/max: {stage_text}"
        self._printer(line)
        self._window_started_s = now_s
        self._cycle_count = 0
        self._stages.clear()


__all__ = ["RuntimeTimingReporter"]
