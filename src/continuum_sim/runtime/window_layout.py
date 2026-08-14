"""Best-effort desktop placement for the MuJoCo viewer and live plots."""

from __future__ import annotations

import os
import sys


_MARGIN_PX = 24
_PANEL_WIDTH_RATIO = 0.34


def place_mujoco_viewer_left() -> None:
    """Place the native MuJoCo window in the left side of the primary display."""

    if sys.platform != "win32":
        return
    try:
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
                matches.append(window)
            return True

        user32.EnumWindows(visit, 0)
        if not matches:
            return
        screen_width = int(user32.GetSystemMetrics(0))
        screen_height = int(user32.GetSystemMetrics(1))
        panel_width = int(screen_width * _PANEL_WIDTH_RATIO)
        width = max(640, screen_width - panel_width - 3 * _MARGIN_PX)
        height = max(480, screen_height - 2 * _MARGIN_PX)
        user32.SetWindowPos(
            matches[-1],
            0,
            _MARGIN_PX,
            _MARGIN_PX,
            width,
            height,
            0x0004 | 0x0010,
        )
    except Exception:
        return


def place_matplotlib_figure_right(figure) -> None:
    """Place a Matplotlib figure in a narrow right-side diagnostics column."""

    try:
        manager = getattr(figure.canvas, "manager", None)
        window = None if manager is None else getattr(manager, "window", None)
        if window is None:
            return
        screen_width, screen_height = _screen_size(window)
        width = max(420, int(screen_width * _PANEL_WIDTH_RATIO) - 2 * _MARGIN_PX)
        height = max(520, screen_height - 2 * _MARGIN_PX)
        x = screen_width - width - _MARGIN_PX
        y = _MARGIN_PX
        if hasattr(window, "setGeometry"):
            window.setGeometry(x, y, width, height)
        elif hasattr(window, "wm_geometry"):
            window.wm_geometry(f"{width}x{height}+{x}+{y}")
        elif hasattr(window, "SetSize"):
            window.SetSize(x, y, width, height)
    except Exception:
        return


def _screen_size(window) -> tuple[int, int]:
    if hasattr(window, "screen"):
        screen = window.screen()
        geometry = screen.availableGeometry()
        return int(geometry.width()), int(geometry.height())
    if hasattr(window, "winfo_screenwidth"):
        return int(window.winfo_screenwidth()), int(window.winfo_screenheight())
    if sys.platform == "win32":
        import ctypes

        return (
            int(ctypes.windll.user32.GetSystemMetrics(0)),
            int(ctypes.windll.user32.GetSystemMetrics(1)),
        )
    return 1920, 1080


__all__ = ["place_matplotlib_figure_right", "place_mujoco_viewer_left"]
