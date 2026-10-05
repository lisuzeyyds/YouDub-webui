from __future__ import annotations

import json

import pytest

from backend.app.audio_mode import (
    audio_mode,
    coerce_empty_tts_segments_to_original,
    is_original_audio,
    sanitize_invalid_segment_ranges,
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


def test_sanitize_drops_rows_with_end_before_start(tmp_path):
    """Regression: asr_fix used to emit start=562903 > end=555133."""
    translation = _write_translation(
        tmp_path / "metadata" / "translation.zh.json",
        [
            {"dst": "正常。", "audio_mode": "tts", "start_time": 1000, "end_time": 2000},
            {"dst": "", "audio_mode": "original", "start_time": 562903, "end_time": 555133},
            {"dst": "末尾。", "audio_mode": "tts", "start_time": 563920, "end_time": 555133},
        ],
    )

    assert sanitize_invalid_segment_ranges(translation) == 2

    items = json.loads(translation.read_text(encoding="utf-8"))["translation"]
    assert [item["dst"] for item in items] == ["正常。"]


def test_sanitize_drops_rows_starting_past_the_audio(tmp_path):
    translation = _write_translation(
        tmp_path / "metadata" / "translation.zh.json",
        [
            {"dst": "正常。", "audio_mode": "tts", "start_time": 1000, "end_time": 2000},
            {"dst": "幽灵。", "audio_mode": "tts", "start_time": 559820, "end_time": 562800},
        ],
    )

    assert sanitize_invalid_segment_ranges(translation, duration_ms=555133) == 1

    items = json.loads(translation.read_text(encoding="utf-8"))["translation"]
    assert [item["dst"] for item in items] == ["正常。"]


def test_sanitize_clamps_row_crossing_the_audio_end(tmp_path):
    translation = _write_translation(
        tmp_path / "metadata" / "translation.zh.json",
        [{"dst": "跨界。", "audio_mode": "tts", "start_time": 554000, "end_time": 560000}],
    )

    assert sanitize_invalid_segment_ranges(translation, duration_ms=555133) == 0

    item = json.loads(translation.read_text(encoding="utf-8"))["translation"][0]
    assert item["start_time"] == 554000
    assert item["end_time"] == 555133


def test_sanitize_leaves_clean_translation_untouched(tmp_path):
    translation = _write_translation(
        tmp_path / "metadata" / "translation.zh.json",
        [{"dst": "有译文。", "audio_mode": "tts", "start_time": 100, "end_time": 900}],
    )
    before = translation.read_text(encoding="utf-8")

    assert sanitize_invalid_segment_ranges(translation, duration_ms=555133) == 0
    assert translation.read_text(encoding="utf-8") == before


def test_sanitize_keeps_rows_without_timestamps(tmp_path):
    """Rows that carry no time range must survive.

    Older artifacts and hand-written fixtures store only dst/audio_mode.
    Defaulting a missing range to 0-0 used to make them all look invalid and
    wipe the whole translation, which broke the TTS stage.
    """
    translation = _write_translation(
        tmp_path / "metadata" / "translation.zh.json",
        [
            {"dst": "有译文。", "audio_mode": "tts"},
            {"dst": "", "audio_mode": "tts"},
        ],
    )
    before = translation.read_text(encoding="utf-8")

    assert sanitize_invalid_segment_ranges(translation, duration_ms=555133) == 0
    assert translation.read_text(encoding="utf-8") == before

    items = json.loads(translation.read_text(encoding="utf-8"))["translation"]
    assert [item["audio_mode"] for item in items] == ["tts", "tts"]


def test_sanitize_drops_only_half_timestamped_rows(tmp_path):
    """A row with only one of the two timestamps cannot be placed; drop it."""
    translation = _write_translation(
        tmp_path / "metadata" / "translation.zh.json",
        [
            {"dst": "完整。", "audio_mode": "tts", "start_time": 100, "end_time": 900},
            {"dst": "半个。", "audio_mode": "tts", "start_time": 1000},
        ],
    )

    assert sanitize_invalid_segment_ranges(translation, duration_ms=555133) == 1

    items = json.loads(translation.read_text(encoding="utf-8"))["translation"]
    assert [item["dst"] for item in items] == ["完整。"]
