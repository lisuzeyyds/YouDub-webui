"""Self-healing support for the local worker process.

On this host the worker can end up in a state where it cannot spawn ffmpeg at
all: every ffmpeg child exits with code 1 and prints nothing (even for
`ffmpeg -version`), while the exact same command succeeds from a freshly
started process. Restarting the service clears the state (verified repeatedly
on 2026-09-25/26).

Instead of leaving the user with a failed task, the worker now:

1. probes ffmpeg (`-version`) when a stage has already failed,
2. if the probe fails, records the task and terminates the process so the
   launcher (``start.bat``) starts a fresh one,
3. on startup, automatically resumes the recorded task.

The number of automatic resumes per task is bounded so a genuinely broken task
cannot loop forever.
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

from . import database
from .config import ffmpeg_binary

STATE_FILE = Path("data/tmp/worker_selfheal.json")
MAX_AUTO_RESUMES = 10
PROBE_TIMEOUT_SECONDS = 30


def _no_window() -> int:
    return subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0


def ffmpeg_healthy() -> bool:
    """True when this process can actually start ffmpeg and get output back."""
    try:
        result = subprocess.run(
            [ffmpeg_binary(), "-nostdin", "-version"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            stdin=subprocess.DEVNULL,
            timeout=PROBE_TIMEOUT_SECONDS,
            creationflags=_no_window(),
        )
    except Exception:  # noqa: BLE001
        return False
    return result.returncode == 0 and bool((result.stdout or "") + (result.stderr or ""))


def _read_state() -> dict[str, Any]:
    try:
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _write_state(state: dict[str, Any]) -> None:
    try:
        STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        STATE_FILE.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError:
        pass


def resume_count(task_id: str) -> int:
    return int(((_read_state().get("resumes") or {}).get(task_id)) or 0)


def request_restart(task_id: str, stage: str, reason: str) -> None:
    """Record the task that must be resumed and flag the process for restart."""
    state = _read_state()
    resumes = dict(state.get("resumes") or {})
    resumes[task_id] = int(resumes.get(task_id, 0)) + 1
    state["resumes"] = resumes
    state["pending"] = {
        "task_id": task_id,
        "stage": stage,
        "reason": reason[:500],
        "at": database.now_iso(),
    }
    _write_state(state)


def take_pending_resume() -> str | None:
    """Return the task recorded by a previous self-restart, if still allowed."""
    state = _read_state()
    pending = state.get("pending") or {}
    task_id = pending.get("task_id")
    if not task_id:
        return None
    state["pending"] = None
    _write_state(state)
    if int((state.get("resumes") or {}).get(task_id) or 0) > MAX_AUTO_RESUMES:
        return None
    task = database.get_task(task_id)
    if not task or task.get("status") != "failed":
        return None
    database.reset_failed_for_resume(task_id)
    return task_id
