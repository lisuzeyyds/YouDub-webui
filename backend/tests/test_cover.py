"""Cover selection: original thumbnail, localised thumbnail, or a frame.

The real OCR and the real image API are never touched here - the point of these
tests is the decision, not the models.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from backend.app.adapters import bilibili, cover


class _Block:
    """Just enough of a TextBlock for prepare_cover's logging and checks."""

    def __init__(self, text: str) -> None:
        self.text = text


def _session(tmp_path) -> Path:
    session = tmp_path / "some-title__dQw4w9WgXcQ"
    (session / "metadata").mkdir(parents=True)
    return session


def _thumbnail(session: Path, payload: bytes = b"jpg") -> Path:
    path = session / "metadata" / cover.ORIGINAL_NAME
    path.write_bytes(payload)
    return path


def test_no_english_uploads_the_thumbnail_unchanged(monkeypatch, tmp_path):
    session = _session(tmp_path)
    _thumbnail(session, b"original-bytes")
    monkeypatch.setattr(cover.cover_ocr, "english_blocks", lambda *_a, **_k: [])

    chosen = cover.prepare_cover(session=session)

    assert chosen == session / "metadata" / cover.CHOSEN_NAME
    assert chosen.read_bytes() == b"original-bytes"


def test_english_is_redrawn_and_the_result_is_used(monkeypatch, tmp_path):
    session = _session(tmp_path)
    _thumbnail(session)
    monkeypatch.setattr(cover.cover_ocr, "english_blocks", lambda *_a, **_k: [_Block("HELLO WORLD")])

    def fake_localize(image, output, **_kwargs):
        output.write_bytes(b"localised")
        return cover.cover_localize.LocalizeResult(
            output=output, targets=["HELLO WORLD"], chinese="", latin_before=10, latin_after=0, cjk_after=4
        )

    monkeypatch.setattr(cover.cover_localize, "localize_cover", fake_localize)

    chosen = cover.prepare_cover(session=session)

    assert chosen == session / "metadata" / cover.CHOSEN_NAME
    assert chosen.read_bytes() == b"localised"


def test_a_failed_redraw_falls_back_to_a_frame(monkeypatch, tmp_path):
    session = _session(tmp_path)
    _thumbnail(session)
    monkeypatch.setattr(cover.cover_ocr, "english_blocks", lambda *_a, **_k: [_Block("HELLO WORLD")])

    def boom(*_args, **_kwargs):
        raise RuntimeError("image API HTTP 402")

    monkeypatch.setattr(cover.cover_localize, "localize_cover", boom)

    assert cover.prepare_cover(session=session) is None


def test_a_result_without_chinese_falls_back_to_a_frame(monkeypatch, tmp_path):
    session = _session(tmp_path)
    _thumbnail(session)
    monkeypatch.setattr(cover.cover_ocr, "english_blocks", lambda *_a, **_k: [_Block("HELLO WORLD")])

    def fake_localize(_image, output, **_kwargs):
        output.write_bytes(b"still-english")
        return cover.cover_localize.LocalizeResult(
            output=output, targets=["HELLO WORLD"], chinese="", latin_before=10, latin_after=10, cjk_after=0
        )

    monkeypatch.setattr(cover.cover_localize, "localize_cover", fake_localize)

    assert cover.prepare_cover(session=session) is None


def test_a_result_that_kept_the_english_falls_back_to_a_frame(monkeypatch, tmp_path):
    session = _session(tmp_path)
    _thumbnail(session)
    monkeypatch.setattr(cover.cover_ocr, "english_blocks", lambda *_a, **_k: [_Block("HELLO WORLD")])

    def fake_localize(_image, output, **_kwargs):
        output.write_bytes(b"half")
        return cover.cover_localize.LocalizeResult(
            output=output,
            targets=["HELLO WORLD"],
            chinese="",
            latin_before=10,
            latin_after=5,
            cjk_after=3,
            leftover=["HELLO WORLD"],
        )

    monkeypatch.setattr(cover.cover_localize, "localize_cover", fake_localize)

    assert cover.prepare_cover(session=session) is None


def test_a_session_without_a_youtube_id_uses_a_frame(monkeypatch, tmp_path):
    session = tmp_path / "uploaded-file__local-123"
    session.mkdir()

    def no_id(_session):
        raise ValueError("no ytdlp_info.json")

    monkeypatch.setattr(cover.fetch_cover, "video_id_from_session", no_id)

    assert cover.prepare_cover(session=session) is None


def test_the_thumbnail_is_downloaded_when_missing(monkeypatch, tmp_path):
    session = _session(tmp_path)
    calls: list[str] = []

    def fake_fetch(video_id, output, **_kwargs):
        calls.append(video_id)
        output.write_bytes(b"downloaded")
        return {"video_id": video_id, "width": 1280, "height": 720, "path": output, "url": "u", "bytes": 10}

    monkeypatch.setattr(cover.fetch_cover, "fetch_cover", fake_fetch)
    monkeypatch.setattr(cover.cover_ocr, "english_blocks", lambda *_a, **_k: [])
    monkeypatch.setattr(cover, "_proxy", lambda: None)

    chosen = cover.prepare_cover(session=session, video_id="dQw4w9WgXcQ")

    assert calls == ["dQw4w9WgXcQ"]
    assert chosen.read_bytes() == b"downloaded"


def test_a_broken_ocr_still_uploads_the_thumbnail(monkeypatch, tmp_path):
    session = _session(tmp_path)
    _thumbnail(session, b"original")

    def boom(*_args, **_kwargs):
        raise ImportError("rapidocr is not installed")

    monkeypatch.setattr(cover.cover_ocr, "english_blocks", boom)

    chosen = cover.prepare_cover(session=session)

    assert chosen.read_bytes() == b"original"


def test_publish_passes_the_video_id_and_reports_the_cover(monkeypatch, tmp_path):
    """The task id is the YouTube id, so the publish stage can fetch the cover."""
    session = tmp_path / "session"
    (session / "tmp").mkdir(parents=True)
    video = tmp_path / "video_final.mp4"
    video.write_bytes(b"mp4")
    chosen = tmp_path / "cover_upload.jpg"
    chosen.write_bytes(b"jpg")

    monkeypatch.setattr(bilibili, "find_chrome", lambda *_a, **_k: "chrome")
    monkeypatch.setattr(bilibili, "find_node", lambda *_a, **_k: "node")
    monkeypatch.setattr(bilibili, "find_script", lambda *_a, **_k: "publish.js")
    seen: dict[str, str] = {}

    def fake_prepare(**kwargs):
        seen.update(kwargs)
        return chosen

    monkeypatch.setattr(bilibili.cover, "prepare_cover", fake_prepare)
    monkeypatch.setattr(bilibili, "_run_browser", lambda *_a, **_k: SimpleNamespace(
        stdout='RESULT {"ok": true, "bvid": "BV1xx"}', stderr="", returncode=0
    ))

    outcome = bilibili.publish_video(
        video_file=video,
        session=session,
        title="标题",
        description="",
        tags=[],
        settings={"profile_dir": str(tmp_path / "profile")},
        video_id="dQw4w9WgXcQ",
    )

    assert seen["video_id"] == "dQw4w9WgXcQ"
    assert outcome.cover == str(chosen)
    payload = json.loads((session / "tmp" / "bilibili_payload.json").read_text(encoding="utf-8"))
    assert payload["cover"] == str(chosen.resolve())


@pytest.mark.parametrize("usable", [True, False])
def test_usable_needs_chinese_and_no_leftover(usable):
    result = cover.cover_localize.LocalizeResult(
        output=Path("x.jpg"),
        targets=["A"],
        chinese="",
        latin_before=1,
        latin_after=0 if usable else 1,
        cjk_after=4 if usable else 4,
        leftover=[] if usable else ["A"],
    )
    assert result.usable is usable
