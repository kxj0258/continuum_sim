"""Best-effort desktop placement for the MuJoCo viewer and live plots."""

from __future__ import annotations

import os
import sys


_MARGIN_PX = 24
_SPI_GETWORKAREA = 0x0030
_SWP_NOZORDER = 0x0004
_SWP_NOACTIVATE = 0x0010


def place_mujoco_viewer_left() -> None:
    """Place the native MuJoCo window in the left half of the work area."""

    if sys.platform != "win32":
        return
    try:
        window = _find_mujoco_window()
        if window is None:
            return
        left, _right = _side_by_side_rectangles(_windows_work_area())
        _move_native_window(window, left)
    except Exception:
        return


def place_matplotlib_figure_right(figure) -> None:
    """Place a Matplotlib figure in the right half of the work area."""

    try:
        manager = getattr(figure.canvas, "manager", None)
        window = None if manager is None else getattr(manager, "window", None)
        if window is None:
            return
        if sys.platform == "win32":
            handle = _native_window_handle(window)
            if handle is not None:
                _left, right = _side_by_side_rectangles(_windows_work_area())
                if _move_native_window(handle, right):
                    return
        _left, right = _side_by_side_rectangles(_screen_geometry(window))
        x, y, width, height = right
        if hasattr(window, "setGeometry"):
            window.setGeometry(x, y, width, height)
        elif hasattr(window, "wm_geometry"):
            window.wm_geometry(f"{width}x{height}+{x}+{y}")
        elif hasattr(window, "SetSize"):
            window.SetSize(x, y, width, height)
    except Exception:
        return


def _side_by_side_rectangles(
    work_area: tuple[int, int, int, int],
) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int]]:
    x, y, width, height = work_area
    margin = min(_MARGIN_PX, max(0, (width - 2) // 3))
    pane_width = max(1, (width - 3 * margin) // 2)
    pane_height = max(1, height - 2 * margin)
    left = (x + margin, y + margin, pane_width, pane_height)
    right = (x + 2 * margin + pane_width, y + margin, pane_width, pane_height)
    return left, right


def _windows_work_area() -> tuple[int, int, int, int]:
    import ctypes
    from ctypes import wintypes

    user32 = ctypes.windll.user32
    user32.SystemParametersInfoW.argtypes = [
        wintypes.UINT,
        wintypes.UINT,
        ctypes.c_void_p,
        wintypes.UINT,
    ]
    user32.SystemParametersInfoW.restype = wintypes.BOOL
    rect = wintypes.RECT()
    if user32.SystemParametersInfoW(
        _SPI_GETWORKAREA,
        0,
        ctypes.byref(rect),
        0,
    ):
        return (
            int(rect.left),
            int(rect.top),
            int(rect.right - rect.left),
            int(rect.bottom - rect.top),
        )
    return (
        0,
        0,
        int(user32.GetSystemMetrics(0)),
        int(user32.GetSystemMetrics(1)),
    )


def _find_mujoco_window() -> int | None:
    import ctypes
    from ctypes import wintypes

    user32 = ctypes.windll.user32
    user32.GetWindowThreadProcessId.argtypes = [
        wintypes.HWND,
        ctypes.POINTER(wintypes.DWORD),
    ]
    user32.GetWindowThreadProcessId.restype = wintypes.DWORD
    user32.IsWindowVisible.argtypes = [wintypes.HWND]
    user32.IsWindowVisible.restype = wintypes.BOOL
    user32.GetWindowTextLengthW.argtypes = [wintypes.HWND]
    user32.GetWindowTextLengthW.restype = ctypes.c_int
    user32.GetWindowTextW.argtypes = [
        wintypes.HWND,
        wintypes.LPWSTR,
        ctypes.c_int,
    ]
    user32.GetWindowTextW.restype = ctypes.c_int
    process_id = os.getpid()
    matches: list[int] = []

    @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    def visit(window, _extra):
        owner = wintypes.DWORD()
        user32.GetWindowThreadProcessId(window, ctypes.byref(owner))
        if owner.value != process_id or not user32.IsWindowVisible(window):
            return True
        length = user32.GetWindowTextLengthW(window)
        if length <= 0:
            return True
        title = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(window, title, length + 1)
        if "mujoco" in title.value.lower():
            matches.append(int(window))
        return True

    user32.EnumWindows(visit, 0)
    return None if not matches else matches[-1]


def _native_window_handle(window) -> int | None:
    for accessor_name in ("winId", "winfo_id", "GetHandle"):
        accessor = getattr(window, accessor_name, None)
        if callable(accessor):
            try:
                return int(accessor())
            except (TypeError, ValueError):
                continue
    return None


def _move_native_window(
    window: int,
    rectangle: tuple[int, int, int, int],
) -> bool:
    import ctypes
    from ctypes import wintypes

    x, y, width, height = rectangle
    user32 = ctypes.windll.user32
    user32.SetWindowPos.argtypes = [
        wintypes.HWND,
        wintypes.HWND,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        wintypes.UINT,
    ]
    user32.SetWindowPos.restype = wintypes.BOOL
    return bool(
        user32.SetWindowPos(
            window,
            0,
            x,
            y,
            width,
            height,
            _SWP_NOZORDER | _SWP_NOACTIVATE,
        )
    )


def _screen_geometry(window) -> tuple[int, int, int, int]:
    if hasattr(window, "screen"):
        screen = window.screen()
        geometry = screen.availableGeometry()
        return (
            int(geometry.x()),
            int(geometry.y()),
            int(geometry.width()),
            int(geometry.height()),
        )
    if hasattr(window, "winfo_screenwidth"):
        return (
            int(getattr(window, "winfo_vrootx", lambda: 0)()),
            int(getattr(window, "winfo_vrooty", lambda: 0)()),
            int(window.winfo_screenwidth()),
            int(window.winfo_screenheight()),
        )
    if sys.platform == "win32":
        return _windows_work_area()
    return 0, 0, 1920, 1080


__all__ = ["place_matplotlib_figure_right", "place_mujoco_viewer_left"]
