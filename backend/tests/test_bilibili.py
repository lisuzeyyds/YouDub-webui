from __future__ import annotations

import json
from pathlib import Path

import pytest

from backend.app.adapters import bilibili
from backend.app.errors import DeterministicStageError


class _Completed:
    """Stand-in for subprocess.CompletedProcess."""

    def __init__(self, stdout: str = "", stderr: str = "", returncode: int = 0) -> None:
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode


def _toolchain(monkeypatch, tmp_path) -> dict[str, str]:
    """Point the adapter at fake browser tooling inside tmp_path."""
    monkeypatch.setattr(bilibili, "find_chrome", lambda *_a, **_k: "chrome")
    monkeypatch.setattr(bilibili, "find_node", lambda *_a, **_k: "node")
    monkeypatch.setattr(bilibili, "find_script", lambda *_a, **_k: "publish.js")
    monkeypatch.setattr(bilibili, "build_cover", lambda *_a, **_k: None)
    return {"profile_dir": str(tmp_path / "profile")}


def _video(tmp_path) -> Path:
    path = tmp_path / "video_final.mp4"
    path.write_bytes(b"mp4")
    return path


def test_parse_result_reads_the_marker_line():
    stdout = 'progress line\nRESULT {"ok": true, "bvid": "BV1iQpi6XEMv"}\n'
    assert bilibili._parse_result(stdout) == {"ok": True, "bvid": "BV1iQpi6XEMv"}


def test_parse_result_ignores_payloads_it_cannot_read():
    assert bilibili._parse_result("") is None
    assert bilibili._parse_result("RESULT not-json") is None
    assert bilibili._parse_result('RESULT ["a"]') is None


def test_publish_requires_the_browser_toolchain(monkeypatch, tmp_path):
    """A missing browser is deterministic - retrying would change nothing."""
    monkeypatch.setattr(bilibili, "find_chrome", lambda *_a, **_k: "")
    monkeypatch.setattr(bilibili, "find_node", lambda *_a, **_k: "node")
    monkeypatch.setattr(bilibili, "find_script", lambda *_a, **_k: "publish.js")

    with pytest.raises(DeterministicStageError) as excinfo:
        bilibili.publish_video(
            video_file=_video(tmp_path),
            session=tmp_path,
            title="标题",
            description="简介",
            tags=[],
            settings={"profile_dir": str(tmp_path / "profile")},
        )
    assert "chrome" in str(excinfo.value)


def test_publish_reports_an_expired_login(monkeypatch, tmp_path):
    settings = _toolchain(monkeypatch, tmp_path)
    monkeypatch.setattr(
        bilibili,
        "_run_browser",
        lambda *_a, **_k: _Completed(
            stdout='RESULT {"ok": false, "stage": "login", "error": "redirected"}',
            returncode=1,
        ),
    )

    with pytest.raises(DeterministicStageError) as excinfo:
        bilibili.publish_video(
            video_file=_video(tmp_path),
            session=tmp_path,
            title="标题",
            description="简介",
            tags=[],
            settings=settings,
        )
    assert "sign in" in str(excinfo.value)


def test_publish_returns_the_video_id_and_sends_the_copy(monkeypatch, tmp_path):
    settings = _toolchain(monkeypatch, tmp_path)
    captured: dict[str, object] = {}

    def fake_run(command, *, cwd):
        captured["command"] = command
        captured["cwd"] = cwd
        captured["payload"] = json.loads(Path(command[2]).read_text(encoding="utf-8"))
        return _Completed(
            stdout='RESULT {"ok": true, "stage": "done", "bvid": "BV1iQpi6XEMv", '
            '"zone": "游戏", "collection": "mc"}'
        )

    monkeypatch.setattr(bilibili, "_run_browser", fake_run)

    outcome = bilibili.publish_video(
        video_file=_video(tmp_path),
        session=tmp_path,
        title="标题",
        description="简介",
        tags=["Minecraft"],
        zone="游戏",
        collection="mc",
        settings=settings,
    )

    assert outcome.ok is True
    assert outcome.bvid == "BV1iQpi6XEMv"
    assert outcome.url == "https://www.bilibili.com/video/BV1iQpi6XEMv"
    assert outcome.zone == "游戏"
    assert outcome.collection == "mc"

    payload = captured["payload"]
    assert payload["title"] == "标题"
    assert payload["tags"] == ["Minecraft"]
    assert payload["zone"] == "游戏"
    assert payload["collection"] == "mc"
    # The creation declaration is not a per-release decision.
    assert payload["declare"] == bilibili.DEFAULT_DECLARE
    assert captured["command"][0] == "node"


def test_publish_without_a_result_line_is_a_retryable_failure(monkeypatch, tmp_path):
    """No verdict at all means the run died early; that is worth one retry."""
    settings = _toolchain(monkeypatch, tmp_path)
    monkeypatch.setattr(
        bilibili,
        "_run_browser",
        lambda *_a, **_k: _Completed(stdout="", stderr="chrome crashed", returncode=1),
    )

    with pytest.raises(RuntimeError) as excinfo:
        bilibili.publish_video(
            video_file=_video(tmp_path),
            session=tmp_path,
            title="标题",
            description="简介",
            tags=[],
            settings=settings,
        )
    assert "chrome crashed" in str(excinfo.value)


def test_publish_result_round_trip(tmp_path):
    path = bilibili.publish_artifact_path(tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"bvid": "BV1iQpi6XEMv", "url": "https://www.bilibili.com/video/BV1iQpi6XEMv"}),
        encoding="utf-8",
    )

    loaded = bilibili.load_publish_result(tmp_path)
    assert loaded is not None
    assert loaded["bvid"] == "BV1iQpi6XEMv"


def test_load_publish_result_is_absent_before_publishing(tmp_path):
    assert bilibili.load_publish_result(tmp_path) is None
