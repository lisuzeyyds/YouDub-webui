"""What a backend restart is allowed to touch.

The supervisor restarts the backend on its own when a worker can no longer spawn
ffmpeg, so this cleanup runs more often than a human ever presses "restart".
"""

from __future__ import annotations

import pytest

from backend.app import database


@pytest.fixture
def db(monkeypatch, tmp_path):
    monkeypatch.setattr(database, "DB_PATH", tmp_path / "restart.sqlite")
    database.init_db()
    return database


def _task(video_id: str) -> str:
    task_id, _ = database.create_or_get_video_task(
        f"https://www.youtube.com/watch?v={video_id}", video_id
    )
    return task_id


def test_a_queued_task_survives_a_restart(db):
    """Queued means "not started", so there is nothing stale to clean up.

    Marking them failed emptied the whole queue at every restart, and only a
    failed task can be resumed by hand - so eight queued downloads turned into
    eight tasks nobody was going to run.
    """
    task_id = _task("AAAAAAAAAAA")
    assert database.get_task(task_id)["status"] == "queued"

    database.fail_stale_active_tasks()

    assert database.get_task(task_id)["status"] == "queued"


def test_a_running_task_is_marked_failed_with_its_stage(db):
    task_id = _task("BBBBBBBBBBB")
    database.update_task(task_id, status="running", current_stage="translate")
    database.update_stage(task_id, "translate", status="running")

    database.fail_stale_active_tasks()

    task = database.get_task(task_id)
    assert task["status"] == "failed"
    assert "restarted" in (task["error_message"] or "")
    stage = next(s for s in task["stages"] if s["name"] == "translate")
    assert stage["status"] == "failed"
    # A completed stage is left alone: its output is still on disk.
    assert next(s for s in task["stages"] if s["name"] == "download")["status"] == "pending"
