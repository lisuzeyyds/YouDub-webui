from __future__ import annotations

import json

import pytest

from backend.app.audio_mode import (
    audio_mode,
    coerce_empty_tts_segments_to_original,
    is_original_audio,
    target_text,
)


def _write_translation(path, items):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"translation": items}, ensure_ascii=False), encoding="utf-8")
    return path


def test_explicit_audio_mode_takes_priority_over_target_text():
    assert audio_mode({"dst": "（笑声）", "audio_mode": "original"}) == "original"
    assert audio_mode({"dst": "", "audio_mode": "tts"}) == "tts"
    assert is_original_audio({"dst": "（呻吟）", "audio_mode": "original"})


def test_legacy_items_infer_audio_mode_from_target_text():
    assert audio_mode({"dst": ""}) == "original"
    assert audio_mode({"dst": "Meaningful."}) == "tts"
    assert audio_mode({"zh": "旧版字幕"}) == "tts"
    assert target_text({"dst": "", "zh": "旧版字幕"}) == "旧版字幕"


def test_invalid_audio_mode_is_rejected():
    with pytest.raises(ValueError, match="audio_mode"):
        audio_mode({"dst": "Hello", "audio_mode": "invalid"})


def test_coerce_rewrites_empty_tts_segments_to_original(tmp_path):
    translation = _write_translation(
        tmp_path / "metadata" / "translation.zh.json",
        [
            {"dst": "有译文。", "audio_mode": "tts"},
            {"dst": "   ", "audio_mode": "tts"},
            {"dst": "", "audio_mode": "tts"},
            {"dst": "（笑声）", "audio_mode": "original"},
        ],
    )

    assert coerce_empty_tts_segments_to_original(translation) == 2

    items = json.loads(translation.read_text(encoding="utf-8"))["translation"]
    assert [item["audio_mode"] for item in items] == [
        "tts",
        "original",
        "original",
        "original",
    ]


def test_coerce_leaves_clean_translation_untouched(tmp_path):
    translation = _write_translation(
        tmp_path / "metadata" / "translation.zh.json",
        [{"dst": "有译文。", "audio_mode": "tts"}],
    )
    before = translation.read_text(encoding="utf-8")

    assert coerce_empty_tts_segments_to_original(translation) == 0
    assert translation.read_text(encoding="utf-8") == before
