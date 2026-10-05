from __future__ import annotations

import json
import logging
import re
from pathlib import Path

log = logging.getLogger(__name__)

# Whisper loops on non-speech audio (outro music, room tone) and emits the same
# short phrase over and over until the track ends, e.g. twelve identical
# "Thank you." spanning 340 s with a rock-steady ~29 s spacing - the beat of the
# music, not a human talking. Such a run is safe to drop: real repeated lines
# sit close together, because speech is continuous.
_LOOP_MIN_RUN = 3
_LOOP_MIN_GAP_MS = 8000
_LOOP_MAX_TEXT_CHARS = 40
_LOOP_TEXT_RE = re.compile(r"[^\w\u4e00-\u9fff]+")


def _loop_key(text: str) -> str:
    return _LOOP_TEXT_RE.sub("", text).lower()


def _strip_trailing_repetition(utts: list) -> tuple[list, list]:
    """Drop a trailing run of identical, widely-spaced short utterances.

    Returns ``(kept, dropped_texts)``. Only the very end of the transcript is
    examined, and a run must both repeat (>=3 identical lines) and be spread out
    (average gap >= 8 s) before it is treated as hallucination.
    """
    if len(utts) < _LOOP_MIN_RUN:
        return utts, []
    key = _loop_key(utts[-1].get("text", ""))
    if not key or len(utts[-1].get("text", "")) > _LOOP_MAX_TEXT_CHARS:
        return utts, []
    start = len(utts) - 1
    while start > 0 and _loop_key(utts[start - 1].get("text", "")) == key:
        start -= 1
    run = utts[start:]
    if len(run) < _LOOP_MIN_RUN:
        return utts, []
    gaps = [
        run[i]["start_time"] - run[i - 1]["end_time"] for i in range(1, len(run))
    ]
    average_gap = sum(gaps) / len(gaps)
    if average_gap < _LOOP_MIN_GAP_MS:
        return utts, []
    return utts[:start], [u["text"] for u in run]


def _start_pad(idx: int, utts: list, start_pad: int, end_pad: int, min_gap: int) -> int:
    orig_start = utts[idx]["start_time"]
    if idx == 0:
        return max(0, orig_start - start_pad)

    prev_end = utts[idx - 1]["end_time"]
    gap = orig_start - prev_end
    total = start_pad + end_pad

    if gap >= total + min_gap:
        return orig_start - start_pad
    if gap > min_gap:
        share = int((gap - min_gap) * start_pad / total)
        return orig_start - share
    return prev_end + gap // 2


def _end_pad(idx: int, utts: list, duration: int, start_pad: int, end_pad: int, min_gap: int) -> int:
    orig_end = utts[idx]["end_time"]
    if idx == len(utts) - 1:
        return min(duration, orig_end + end_pad) if duration else orig_end + end_pad

    next_start = utts[idx + 1]["start_time"]
    gap = next_start - orig_end
    total = start_pad + end_pad

    if gap >= total + min_gap:
        return orig_end + end_pad
    if gap > min_gap:
        share = int((gap - min_gap) * end_pad / total)
        return orig_end + share
    return orig_end + gap // 2


def _apply_padding(utts: list, duration: int, start_pad: int, end_pad: int) -> list:
    if not utts:
        return utts

    min_gap = 50
    result = []
    for idx in range(len(utts)):
        new_start = _start_pad(idx, utts, start_pad, end_pad, min_gap)
        new_end = _end_pad(idx, utts, duration, start_pad, end_pad, min_gap)
        start = max(0, new_start)
        end = min(duration, new_end) if duration else new_end
        if end <= start:
            # 兜底：padding 后退化成零长/倒序区间。绝不能把 start >= end 放出去，
            # 下游 TTS 会拿着它去切原声或生成零长片段，进而整条任务失败。
            log.warning("asr_fix dropped degenerate segment after padding: %r", utts[idx])
            continue
        result.append({
            **utts[idx],
            "start_time": start,
            "end_time": end,
        })
    return result


def _normalize(utterances: list) -> list:
    return [
        {"text": u["text"].strip(), "start_time": u["start_time"], "end_time": u["end_time"]}
        for u in utterances if u.get("text", "").strip()
    ]


def _drop_out_of_range(utts: list, duration: int) -> tuple[list, list]:
    """丢弃时间戳落在音频之外的片段，并把跨尾的收进时长内。

    Whisper 在片尾的静音/音乐上会"幻觉"出几个不存在的短句，并给出超过音频总长的
    时间戳（实例：音频 555133 ms，却出现 559920-567920 ms 的 "One"/"olate"/"I"）。
    这些词在录音里根本不存在，保留只会生成错位配音；更糟的是旧逻辑只把 end_time
    截到时长、没截 start_time，于是产出 start > end 的非法区间，直接把 TTS 阶段打断。

    返回 ``(保留的片段, 被丢弃片段的文本)``。
    """
    if not duration:
        return utts, []
    kept, dropped = [], []
    for u in utts:
        if u["start_time"] >= duration:
            dropped.append(u["text"])
            continue
        kept.append({**u, "end_time": min(u["end_time"], duration)})
    return kept, dropped


def fix_asr_sentences(asr_file: Path, session: Path,
                     start_pad: int = 100, end_pad: int = 300,
                     language: str = "en") -> Path:
    output_file = session / "metadata" / "asr_fixed.json"
    if output_file.exists():
        return output_file

    data = json.loads(Path(asr_file).read_text(encoding="utf-8"))
    utterances = data["result"]["utterances"]
    duration = data.get("audio_info", {}).get("duration", 0)

    new_utts = _normalize(utterances)
    if not new_utts:
        raise RuntimeError("ASR result has no utterances.")

    in_range, dropped_texts = _drop_out_of_range(new_utts, duration)
    if dropped_texts:
        log.warning(
            "asr_fix dropped %d utterance(s) timestamped outside the audio "
            "(0-%.3fs): %s",
            len(dropped_texts),
            duration / 1000,
            dropped_texts,
        )
    if not in_range:
        raise RuntimeError("ASR result has no utterances inside the audio range.")

    in_range, looped_texts = _strip_trailing_repetition(in_range)
    if looped_texts:
        log.warning(
            "asr_fix dropped %d hallucinated trailing utterance(s) "
            "(identical text repeated across the outro): %s",
            len(looped_texts),
            looped_texts[0],
        )
    if not in_range:
        raise RuntimeError("ASR result has no utterances inside the audio range.")

    padded = _apply_padding(in_range, duration, start_pad, end_pad)
    if not padded:
        raise RuntimeError("ASR result has no usable utterances after padding.")
    payload = {
        "audio_info": data.get("audio_info", {}),
        "result": {"text": data["result"].get("text", ""), "utterances": padded},
    }
    output_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_file
