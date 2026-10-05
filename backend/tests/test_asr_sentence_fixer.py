from __future__ import annotations

import json

import pytest

from backend.app.adapters import asr_sentence_fixer


def _utt(text: str, start: int, end: int) -> dict:
    return {"text": text, "start_time": start, "end_time": end}


def _write_asr(tmp_path, utterances: list, duration: int = 10000, text: str = "") -> tuple:
    session = tmp_path / "session"
    (session / "metadata").mkdir(parents=True)
    asr_file = session / "metadata" / "asr.json"
    payload = {
        "audio_info": {"duration": duration},
        "result": {"text": text, "utterances": utterances},
    }
    asr_file.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return asr_file, session


def test_fix_asr_sentences_passes_through_utterances(tmp_path):
    utts = [_utt("Hello world.", 100, 1200), _utt("How are you?", 1500, 2800)]
    asr_file, session = _write_asr(tmp_path, utts)

    fixed = asr_sentence_fixer.fix_asr_sentences(asr_file, session, start_pad=50, end_pad=100)
    out = json.loads(fixed.read_text(encoding="utf-8"))["result"]["utterances"]

    assert [u["text"] for u in out] == ["Hello world.", "How are you?"]


def test_fix_asr_sentences_drops_empty_text(tmp_path):
    utts = [_utt("Hello.", 0, 500), _utt("   ", 600, 800), _utt("World.", 900, 1500)]
    asr_file, session = _write_asr(tmp_path, utts)

    fixed = asr_sentence_fixer.fix_asr_sentences(asr_file, session)
    out = json.loads(fixed.read_text(encoding="utf-8"))["result"]["utterances"]

    assert [u["text"] for u in out] == ["Hello.", "World."]


def test_fix_asr_sentences_applies_padding_within_gap(tmp_path):
    utts = [_utt("a", 1000, 2000), _utt("b", 3000, 4000)]
    asr_file, session = _write_asr(tmp_path, utts, duration=5000)

    fixed = asr_sentence_fixer.fix_asr_sentences(asr_file, session, start_pad=100, end_pad=300)
    out = json.loads(fixed.read_text(encoding="utf-8"))["result"]["utterances"]

    assert out[0]["start_time"] == 900
    assert out[0]["end_time"] == 2300
    assert out[1]["start_time"] == 2900
    assert out[1]["end_time"] == 4300


def test_fix_asr_sentences_clamps_to_duration(tmp_path):
    utts = [_utt("only", 100, 4900)]
    asr_file, session = _write_asr(tmp_path, utts, duration=5000)

    fixed = asr_sentence_fixer.fix_asr_sentences(asr_file, session, start_pad=200, end_pad=500)
    out = json.loads(fixed.read_text(encoding="utf-8"))["result"]["utterances"]

    assert out[0]["start_time"] == 0  # 100 - 200 -> clamp 0
    assert out[0]["end_time"] == 5000


def test_fix_asr_sentences_raises_when_empty(tmp_path):
    asr_file, session = _write_asr(tmp_path, [_utt("  ", 0, 100)])

    with pytest.raises(RuntimeError):
        asr_sentence_fixer.fix_asr_sentences(asr_file, session)


def test_fix_asr_sentences_reuses_cache(tmp_path):
    utts = [_utt("hi", 0, 500)]
    asr_file, session = _write_asr(tmp_path, utts)

    first = asr_sentence_fixer.fix_asr_sentences(asr_file, session)
    first.write_text('{"already": true}', encoding="utf-8")
    second = asr_sentence_fixer.fix_asr_sentences(asr_file, session)

    assert json.loads(second.read_text(encoding="utf-8")) == {"already": True}


def test_fix_asr_sentences_drops_trailing_hallucinations(tmp_path):
    """Whisper invents utterances past the end of the audio; those must go.

    Regression: duration 555133 ms but utterances timestamped 559920-567920 ms.
    The old code clamped only end_time, so it emitted start=559820 > end=555133
    and the TTS stage died with "Original audio does not cover target segment".
    """
    utts = [
        _utt("my friends.", 545060, 545580),
        _utt("The last line.", 554360, 555133),
        _utt("One", 559920, 562800),
        _utt("olate", 562920, 563920),
        _utt("I", 563920, 567920),
    ]
    asr_file, session = _write_asr(tmp_path, utts, duration=555133)

    fixed = asr_sentence_fixer.fix_asr_sentences(asr_file, session)
    out = json.loads(fixed.read_text(encoding="utf-8"))["result"]["utterances"]

    assert [u["text"] for u in out] == ["my friends.", "The last line."]
    assert all(u["start_time"] < u["end_time"] for u in out)
    assert all(u["end_time"] <= 555133 for u in out)


def test_fix_asr_sentences_drops_legacy_corrupted_range(tmp_path):
    """An artifact written by the buggy version (start > end) is repaired too."""
    utts = [_utt("keep", 1000, 2000), _utt("olate", 562903, 555133)]
    asr_file, session = _write_asr(tmp_path, utts, duration=555133)

    fixed = asr_sentence_fixer.fix_asr_sentences(asr_file, session)
    out = json.loads(fixed.read_text(encoding="utf-8"))["result"]["utterances"]

    assert [u["text"] for u in out] == ["keep"]
    assert out[0]["start_time"] < out[0]["end_time"]


def test_fix_asr_sentences_clamps_segment_crossing_the_end(tmp_path):
    utts = [_utt("crosses", 554000, 560000)]
    asr_file, session = _write_asr(tmp_path, utts, duration=555133)

    fixed = asr_sentence_fixer.fix_asr_sentences(
        asr_file, session, start_pad=0, end_pad=0
    )
    out = json.loads(fixed.read_text(encoding="utf-8"))["result"]["utterances"]

    assert out[0]["start_time"] == 554000
    assert out[0]["end_time"] == 555133


def test_fix_asr_sentences_raises_when_all_are_outside_the_audio(tmp_path):
    utts = [_utt("One", 559920, 562800), _utt("olate", 562920, 563920)]
    asr_file, session = _write_asr(tmp_path, utts, duration=555133)

    with pytest.raises(RuntimeError):
        asr_sentence_fixer.fix_asr_sentences(asr_file, session)


# --------------------------------------------------------------------------
# Whisper outro-music hallucination ("Thank you." loop)
# --------------------------------------------------------------------------


def test_strip_trailing_repetition_drops_a_music_loop():
    """The real case: 8 identical lines ~29 s apart on the outro music."""
    utts = [
        _utt("This is the real narration, running for a while.", 0, 3000),
        _utt("It continues with a second sentence.", 3400, 6000),
        _utt("And a third one before the music starts.", 6400, 9000),
    ]
    for i in range(8):
        start = 588860 + i * 29000
        utts.append(_utt("Thank you.", start, start + 1400))

    kept, dropped = asr_sentence_fixer._strip_trailing_repetition(utts)

    assert [u["text"] for u in kept] == [
        "This is the real narration, running for a while.",
        "It continues with a second sentence.",
        "And a third one before the music starts.",
    ]
    assert dropped == ["Thank you."] * 8


def test_strip_trailing_repetition_keeps_rapid_real_repeats():
    """Genuine repeated lines sit close together and must survive."""
    utts = [
        _utt("Go!", 0, 500),
        _utt("Go!", 700, 1200),
        _utt("Go!", 1400, 1900),
        _utt("Go!", 2100, 2600),
    ]

    kept, dropped = asr_sentence_fixer._strip_trailing_repetition(utts)

    assert kept == utts
    assert dropped == []


def test_strip_trailing_repetition_needs_at_least_three_lines():
    utts = [
        _utt("Hello.", 0, 1000),
        _utt("Thank you.", 40000, 42000),
        _utt("Thank you.", 80000, 82000),
    ]

    kept, dropped = asr_sentence_fixer._strip_trailing_repetition(utts)

    assert kept == utts
    assert dropped == []


def test_strip_trailing_repetition_ignores_differing_tail():
    utts = [_utt("A.", 0, 1000), _utt("B.", 40000, 41000), _utt("C.", 80000, 81000)]

    kept, dropped = asr_sentence_fixer._strip_trailing_repetition(utts)

    assert kept == utts
    assert dropped == []


def test_strip_trailing_repetition_ignores_a_long_tail_line():
    """A long repeated sentence is real content, not a hallucinated loop."""
    long_line = "This is a genuinely long sentence that someone might repeat verbatim."
    utts = [_utt(long_line, 0, 2000)]
    for i in range(4):
        utts.append(_utt(long_line, 40000 + i * 40000, 42000 + i * 40000))

    kept, dropped = asr_sentence_fixer._strip_trailing_repetition(utts)

    assert kept == utts
    assert dropped == []


def test_fix_asr_sentences_removes_outro_hallucination(tmp_path):
    duration = 100000
    utts = [
        _utt("This is the real narration and it runs for a while.", 0, 3000),
        _utt("A second real sentence follows it closely.", 3400, 6000),
    ]
    for i in range(5):
        start = 40000 + i * 10000
        utts.append(_utt("Thank you.", start, start + 1000))
    asr_file, session = _write_asr(tmp_path, utts, duration=duration)

    fixed = asr_sentence_fixer.fix_asr_sentences(asr_file, session, start_pad=0, end_pad=0)
    out = json.loads(fixed.read_text(encoding="utf-8"))["result"]["utterances"]

    assert [u["text"] for u in out] == [
        "This is the real narration and it runs for a while.",
        "A second real sentence follows it closely.",
    ]


def test_strip_trailing_repetition_keeps_the_closing_line():
    """The real transcript ends with a distinctive closing line.

    The scan reaches it (short, followed by the outro's silence); it must be
    handed back because it does not repeat, unlike the loop after it.
    """
    utts = [
        _utt("The narration explains the mod for a solid minute.", 0, 4000),
        _utt("It keeps going with detailed technical context.", 4400, 9000),
        _utt("Enough talk. Enjoy the fall.", 104280, 106400),
    ]
    for i in range(10):
        start = 121160 + i * 30000
        utts.append(_utt("Thank you.", start, start + 1000))

    kept, dropped = asr_sentence_fixer._strip_trailing_repetition(utts)

    assert kept[-1]["text"] == "Enough talk. Enjoy the fall."
    assert len(dropped) == 10


def test_strip_trailing_repetition_keeps_a_dense_burst_before_real_content():
    """Back-to-back repeats inside the tail do not stop the scan."""
    utts = [
        _utt("Real narration line number one here.", 0, 3000),
        _utt("Real narration line number two here.", 3400, 6000),
    ]
    # A burst of repeats packed together, then the sparse loop.
    for i in range(4):
        start = 12000 + i * 300
        utts.append(_utt("Thank you.", start, start + 250))
    for i in range(8):
        start = 40000 + i * 30000
        utts.append(_utt("Thank you.", start, start + 1000))

    kept, dropped = asr_sentence_fixer._strip_trailing_repetition(utts)

    assert [u["text"] for u in kept] == [
        "Real narration line number one here.",
        "Real narration line number two here.",
    ]
    assert len(dropped) == 12


def test_strip_trailing_repetition_ignores_a_mixed_distinct_tail():
    """Distinct short lines over the outro are not the repeated loop."""
    utts = [
        _utt("Real narration that runs for a while.", 0, 3000),
        _utt("Another real narration sentence.", 3400, 6000),
    ]
    for i, text in enumerate(["Bye.", "See you.", "Take care.", "Good night.", "Later."]):
        start = 40000 + i * 20000
        utts.append(_utt(text, start, start + 800))

    kept, dropped = asr_sentence_fixer._strip_trailing_repetition(utts)

    assert kept == utts
    assert dropped == []
