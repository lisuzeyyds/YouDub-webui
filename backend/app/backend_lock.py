"""Single-instance lock for the backend process.

Uvicorn runs the application lifespan *before* it binds the listening socket,
so a duplicate uvicorn that is about to die with "address already in use"
still executes startup code. ``fail_stale_active_tasks()`` (and the self-heal
resume) must therefore only run in the process that actually owns the
runtime: a second supervisor loop left running next to the real one kills
running tasks this way - observed on 2026-10-14, when a duplicate loop
marked a task that was mid-TTS as "Backend restarted before the task
completed." while its worker kept generating clips.

An exclusive OS lock on ``data/tmp/backend.lock`` decides ownership:

- the live backend holds the lock for its whole lifetime,
- a duplicate cannot take it and skips the "previous backend died" cleanup,
- a crashed backend releases it (the OS closes its handles), so a genuine
  restart still cleans up stale tasks.
"""
from __future__ import annotations

import os
from pathlib import Path

LOCK_FILE = Path("data/tmp/backend.lock")

_lock_handle = None


def _lock_region(handle) -> None:  # noqa: ANN001 - file handle
    if os.name == "nt":
        import msvcrt

        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
    else:
        import fcntl

        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)


def _unlock_region(handle) -> None:  # noqa: ANN001 - file handle
    if os.name == "nt":
        import msvcrt

        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
    else:
        import fcntl

        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def claim() -> bool:
    """Try to become the owning backend process.

    Returns True when the lock is ours (including the degenerate case where
    the lock file cannot even be created, which keeps the previous
    behaviour), False when another live process owns the runtime.
    """
    global _lock_handle
    if _lock_handle is not None:
        return True
    try:
        LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
        handle = LOCK_FILE.open("a+")
    except OSError:
        return True
    try:
        _lock_region(handle)
    except OSError:
        handle.close()
        return False
    _lock_handle = handle
    try:
        handle.seek(0)
        handle.truncate()
        handle.write(f"{os.getpid()}\n")
        handle.flush()
    except OSError:
        pass  # bookkeeping only; the lock itself already works
    return True


def release() -> None:
    """Drop the lock explicitly (tests); the OS also drops it on exit."""
    global _lock_handle
    if _lock_handle is None:
        return
    handle, _lock_handle = _lock_handle, None
    try:
        _unlock_region(handle)
    except OSError:
        pass
    finally:
        handle.close()
