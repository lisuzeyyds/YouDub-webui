"""Guard against running ffmpeg-heavy stages when the host is out of memory.

Root cause found on 2026-09-26/28: when the Windows commit limit (RAM + page
file) is nearly reached, ffmpeg children fail in confusing ways - exit code 1
with completely empty stderr, or "x264 [error]: malloc of size N failed" -
because they cannot commit their initial buffers. The usual culprits on this
machine are the IndexTTS service (leaked up to ~17 GB), the webui process and
the API worker itself (~9 GB each).

This module turns that silent failure into an explicit, actionable message and
gives the system a short window to recover before the stage runs.
"""
from __future__ import annotations

import os
import time
from typing import Callable

MIN_FREE_COMMIT_GB = float(os.getenv("YOUDUB_MIN_FREE_COMMIT_GB", "4"))
WAIT_SECONDS = int(os.getenv("YOUDUB_MEMORY_WAIT_SECONDS", "120"))
POLL_SECONDS = 10


def available_commit_gb() -> float | None:
    """Committable memory left on this host, or None when unknown/unsupported."""
    if os.name != "nt":
        return None
    import ctypes

    class MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    status = MEMORYSTATUSEX()
    status.dwLength = ctypes.sizeof(status)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        return None
    return status.ullAvailPageFile / (1024**3)


def describe() -> str:
    free = available_commit_gb()
    if free is None:
        return "committable memory: unknown"
    return f"committable memory left: {free:.1f} GB"


def ensure_available_memory(stage: str, log: Callable[[str], None]) -> None:
    """Wait briefly for memory to free up; raise a clear error if it does not."""
    free = available_commit_gb()
    if free is None or free >= MIN_FREE_COMMIT_GB:
        return
    log(
        f"[{stage}] only {free:.1f} GB of committable memory left - waiting up to "
        f"{WAIT_SECONDS}s for it to recover before starting ffmpeg work"
    )
    deadline = time.monotonic() + WAIT_SECONDS
    while time.monotonic() < deadline:
        time.sleep(POLL_SECONDS)
        free = available_commit_gb()
        if free is None or free >= MIN_FREE_COMMIT_GB:
            log(f"[{stage}] memory recovered: {free:.1f} GB available - continuing")
            return
    raise RuntimeError(
        f"System is out of committable memory ({free:.1f} GB left, "
        f"need {MIN_FREE_COMMIT_GB:.0f} GB). ffmpeg cannot start in this state. "
        "Restart the IndexTTS service (port 9005) and the backend (port 8000) to "
        "reclaim several gigabytes, close heavy applications, then resume the task."
    )
