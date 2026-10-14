"""Tests for the backend single-instance lock.

Uvicorn runs the app lifespan before binding the listening socket, so a
duplicate uvicorn still executes startup code. The lock decides which
process may run fail_stale_active_tasks()/self-heal resume; see
backend/app/backend_lock.py for the incident that motivated it.
"""
from __future__ import annotations

import os

import pytest

from backend.app import backend_lock


def test_lock_region_blocks_a_second_handle(tmp_path):
    path = tmp_path / "backend.lock"
    first = path.open("a+")
    second = path.open("a+")
    try:
        backend_lock._lock_region(first)
        with pytest.raises(OSError):
            backend_lock._lock_region(second)
        backend_lock._unlock_region(first)
        # The byte is free again, so the other handle can take it now.
        backend_lock._lock_region(second)
        backend_lock._unlock_region(second)
    finally:
        first.close()
        second.close()


def test_claim_is_idempotent_within_one_process(tmp_path, monkeypatch):
    monkeypatch.setattr(backend_lock, "LOCK_FILE", tmp_path / "backend.lock")

    assert backend_lock.claim() is True
    assert backend_lock.claim() is True  # same process: still ours
    backend_lock.release()
    assert backend_lock.claim() is True  # reclaim works after release
    backend_lock.release()


def test_claim_records_owner_pid(tmp_path, monkeypatch):
    monkeypatch.setattr(backend_lock, "LOCK_FILE", tmp_path / "backend.lock")

    assert backend_lock.claim() is True
    backend_lock.release()
    # While the lock was held the region was locked for other handles, so
    # read the recorded owner only after releasing.
    content = backend_lock.LOCK_FILE.read_text(encoding="utf-8").strip()
    assert content == str(os.getpid())


def test_release_is_idempotent(tmp_path, monkeypatch):
    monkeypatch.setattr(backend_lock, "LOCK_FILE", tmp_path / "backend.lock")

    backend_lock.release()  # nothing held yet
    assert backend_lock.claim() is True
    backend_lock.release()
    backend_lock.release()
    assert backend_lock.claim() is True
    backend_lock.release()
