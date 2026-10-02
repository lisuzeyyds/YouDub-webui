from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import soundfile as sf

from backend.app.adapters import indextts_api as indextts_mod


def _make_wav(path: Path, duration_ms: int = 1500, sample_rate: int = 16000) -> Path:
    samples = int(sample_rate * duration_ms / 1000)
    wav = np.sin(2 * np.pi * 440 * np.linspace(0, duration_ms / 1000, samples)).astype(np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, wav, sample_rate)
    return path


def _write_translation(path: Path, items: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"translation": items}, ensure_ascii=False),
        encoding="utf-8",
    )
    return path


@pytest.fixture(autouse=True)
def _voices_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("INDEXTTS_VOICES_DIR", str(tmp_path / "voices"))


@patch.object(indextts_mod, "_request_json")
def test_original_segments_reuse_source_audio(mock_request, tmp_path):
    session = tmp_path / "session"
    vocals_dir = session / "segments" / "vocals"
    _make_wav(vocals_dir / "0001.wav", duration_ms=1200)
    _make_wav(vocals_dir / "0002.wav", duration_ms=1200)
    _make_wav(vocals_dir / "0003.wav", duration_ms=1200)
    original_vocals = _make_wav(session / "media" / "audio_vocals.wav", duration_ms=5000)
    translation = _write_translation(
        session / "metadata" / "translation.zh.json",
        [
            {"dst": "第一段。", "audio_mode": "tts", "start_time": 0, "end_time": 900},
            {"dst": "", "audio_mode": "original", "start_time": 1000, "end_time": 1500},
            {"dst": "第三段。", "audio_mode": "tts", "start_time": 1600, "end_time": 2500},
        ],
    )

    mock_request.return_value = b"RIFFfake"

    indextts_mod.generate_tts(
        translation,
        vocals_dir,
        session,
        original_vocals_file=original_vocals,
    )

    tts_dir = session / "segments" / "tts"
    assert mock_request.call_count == 2
    assert mock_request.call_args_list[0].args[1]["text"] == "第一段。"
    assert mock_request.call_args_list[1].args[1]["text"] == "第三段。"
    assert (tts_dir / "0001.wav").read_bytes() == b"RIFFfake"
    assert (tts_dir / "0003.wav").read_bytes() == b"RIFFfake"

    copied, sample_rate = sf.read(tts_dir / "0002.wav", dtype="float32")
    source, source_rate = sf.read(original_vocals, dtype="float32")
    assert sample_rate == source_rate
    assert len(copied) == int(0.5 * sample_rate)
    assert np.allclose(
        copied,
        source[int(1.0 * sample_rate) : int(1.5 * sample_rate)],
        atol=2 / 32768,
    )


def test_original_segments_require_original_vocals_file(tmp_path):
    session = tmp_path / "session"
    vocals_dir = session / "segments" / "vocals"
    _make_wav(vocals_dir / "0001.wav", duration_ms=1200)
    translation = _write_translation(
        session / "metadata" / "translation.zh.json",
        [{"dst": "", "audio_mode": "original", "start_time": 0, "end_time": 900}],
    )

    with pytest.raises(indextts_mod.IndexTTSApiError, match="original"):
        indextts_mod.generate_tts(translation, vocals_dir, session)


@patch.object(indextts_mod, "_request_json")
def test_empty_target_text_reports_segment_index(mock_request, tmp_path):
    session = tmp_path / "session"
    vocals_dir = session / "segments" / "vocals"
    _make_wav(vocals_dir / "0001.wav", duration_ms=1200)
    translation = _write_translation(
        session / "metadata" / "translation.zh.json",
        [{"dst": "", "audio_mode": "tts", "start_time": 0, "end_time": 900}],
    )

    with pytest.raises(indextts_mod.IndexTTSApiError, match="第 1 段"):
        indextts_mod.generate_tts(translation, vocals_dir, session)

    mock_request.assert_not_called()


def test_fallback_reference_skips_original_segments(tmp_path):
    vocals_dir = tmp_path / "vocals"
    long_nonverbal = _make_wav(vocals_dir / "0001.wav", duration_ms=3000)
    short_speech = _make_wav(vocals_dir / "0002.wav", duration_ms=400)

    assert indextts_mod._longest_reference(vocals_dir) == long_nonverbal
    assert indextts_mod._longest_reference(vocals_dir, exclude={1}) == short_speech
