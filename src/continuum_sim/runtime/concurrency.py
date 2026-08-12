"""Small thread-safe runtime primitives for control and presentation pipelines."""

from __future__ import annotations

from collections.abc import Callable
from queue import Empty, Queue
from threading import Event, Lock, Thread, current_thread
from time import perf_counter
from typing import Generic, TypeVar


T = TypeVar("T")
_STOP = object()


class LatestValueSlot(Generic[T]):
    """Publish versioned values while allowing slow consumers to skip old ones."""

    def __init__(self, initial: T) -> None:
        self._value = initial
        self._version = 0
        self._lock = Lock()

    def publish(self, value: T) -> int:
        with self._lock:
            self._value = value
            self._version += 1
            return self._version

    def snapshot(self) -> tuple[T, int]:
        with self._lock:
            return self._value, self._version

    def consume_after(self, version: int) -> tuple[T, int] | None:
        with self._lock:
            if self._version <= int(version):
                return None
            return self._value, self._version


class TimeRateGate:
    """Gate work against absolute monotonic deadlines without accumulated drift."""

    def __init__(
        self,
        interval_s: float,
        *,
        clock: Callable[[], float] = perf_counter,
        start_s: float | None = None,
        fire_immediately: bool = False,
    ) -> None:
        if interval_s <= 0.0:
            raise ValueError("interval_s must be positive.")
        self.interval_s = float(interval_s)
        self._clock = clock
        self._fire_immediately = bool(fire_immediately)
        self.next_deadline_s = 0.0
        self.reset(start_s)

    def reset(self, now_s: float | None = None) -> None:
        now = self._clock() if now_s is None else float(now_s)
        self.next_deadline_s = (
            now if self._fire_immediately else now + self.interval_s
        )

    def due(self, now_s: float | None = None) -> bool:
        now = self._clock() if now_s is None else float(now_s)
        tolerance_s = 1.0e-12
        if now + tolerance_s < self.next_deadline_s:
            return False
        intervals = max(
            1,
            int((now + tolerance_s - self.next_deadline_s) // self.interval_s) + 1,
        )
        self.next_deadline_s += intervals * self.interval_s
        return True


class MonotonicRateRunner:
    """Run a callback at absolute monotonic deadlines on a stoppable thread."""

    def __init__(
        self,
        interval_s: float,
        callback: Callable[[], object],
        error_callback: Callable[[BaseException], None] | None = None,
        *,
        name: str = "continuum-sim-rate-runner",
        clock: Callable[[], float] = perf_counter,
        precision_window_s: float = 0.001,
    ) -> None:
        if interval_s <= 0.0:
            raise ValueError("interval_s must be positive.")
        self.interval_s = float(interval_s)
        self._callback = callback
        self._error_callback = error_callback
        self._name = str(name)
        self._clock = clock
        self._precision_window_s = max(0.0, float(precision_window_s))
        self._stop_event = Event()
        self._thread: Thread | None = None
        self._failure: BaseException | None = None
        self._lifecycle_lock = Lock()

    @property
    def is_alive(self) -> bool:
        thread = self._thread
        return thread is not None and thread.is_alive()

    def start(self) -> None:
        with self._lifecycle_lock:
            if self.is_alive:
                return
            self._failure = None
            self._stop_event.clear()
            self._thread = Thread(target=self._run, name=self._name, daemon=True)
            self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        thread = self._thread
        if thread is not None and thread is not current_thread():
            thread.join()
        with self._lifecycle_lock:
            if self._thread is None or not self._thread.is_alive():
                self._thread = None

    def raise_if_failed(self) -> None:
        if self._failure is not None:
            raise self._failure

    def _run(self) -> None:
        next_deadline_s = self._clock()
        try:
            while not self._stop_event.is_set():
                self._callback()
                next_deadline_s += self.interval_s
                now_s = self._clock()
                if next_deadline_s <= now_s:
                    missed = int((now_s - next_deadline_s) // self.interval_s) + 1
                    next_deadline_s += missed * self.interval_s
                self._wait_until(next_deadline_s)
        except BaseException as exc:  # noqa: BLE001 - propagated to owner thread.
            self._failure = exc
            if self._error_callback is not None:
                self._error_callback(exc)

    def _wait_until(self, deadline_s: float) -> None:
        while not self._stop_event.is_set():
            remaining_s = deadline_s - self._clock()
            if remaining_s <= 0.0:
                return
            if remaining_s > self._precision_window_s:
                self._stop_event.wait(remaining_s - self._precision_window_s)
                continue
            # A short active tail avoids Windows timer quantisation dominating
            # a 20 ms control period while bounding the CPU cost per cycle.


class AsyncLinePrinter:
    """Serialize line output on a dedicated thread."""

    def __init__(
        self,
        sink: Callable[[str], object] = print,
        *,
        name: str = "continuum-sim-line-printer",
    ) -> None:
        self._sink = sink
        self._queue: Queue[str | object] = Queue()
        self._closed = False
        self._failure: BaseException | None = None
        self._lock = Lock()
        self._thread = Thread(target=self._run, name=name, daemon=True)
        self._thread.start()

    def write(self, line: str) -> None:
        with self._lock:
            if self._closed:
                return
            self._queue.put_nowait(str(line))

    def close(self, *, drain: bool = True) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            if not drain:
                while True:
                    try:
                        self._queue.get_nowait()
                    except Empty:
                        break
                    else:
                        self._queue.task_done()
            self._queue.put_nowait(_STOP)
        if self._thread is not current_thread():
            self._thread.join()

    def raise_if_failed(self) -> None:
        if self._failure is not None:
            raise self._failure

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            try:
                if item is _STOP:
                    return
                self._sink(str(item))
            except BaseException as exc:  # noqa: BLE001 - retained for diagnostics.
                self._failure = exc
            finally:
                self._queue.task_done()


__all__ = [
    "AsyncLinePrinter",
    "LatestValueSlot",
    "MonotonicRateRunner",
    "TimeRateGate",
]
