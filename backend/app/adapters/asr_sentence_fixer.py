from __future__ import annotations

import json
import logging
import re
from pathlib import Path

log = logging.getLogger(__name__)

# Whisper loops on non-speech audio (outro music, room tone) and emits short
# phrases until the track ends. Observed on an 812 s video: the real narration
# stops at 106 s, then "Thank you." repeats 30+ times across the remaining
# 700 s at a rock-steady ~29 s spacing - the beat of the music, not a human
# talking. A single differing line ("You're welcome.") and a burst of
# back-to-back repeats sit inside that tail, so the run cannot be found by
# looking for one identical string, and walking back line by line stalls on the
# dense burst.
#
# What reliably marks the tail is the long silence that precedes it: when the
# narration stops, the outro music starts with several seconds of quiet, and
# everything after that silence is short, isolated lines. So look for the last
# long gap and drop the tail only when it is entirely made of short lines.
_LOOP_MIN_RUN = 3
_LOOP_MIN_REPEATS = 3
_LOOP_MIN_GAP_MS = 8000
_LOOP_MAX_TEXT_CHARS = 40
_LOOP_MIN_REPEAT_RATIO = 0.6
_LOOP_PUNCT_RE = re.compile(r"[^\w\u4e00-\u9fff]+")


def _normalize_key(text: str) -> str:
    return _LOOP_PUNCT_RE.sub("", text).lower()


def _is_short(text: str) -> bool:
    stripped = text.strip()
    return bool(stripped) and len(stripped) <= _LOOP_MAX_TEXT_CHARS


def _strip_trailing_repetition(utts: list) -> tuple[list, list]:
    """Drop a hallucinated tail of short, isolated lines over the outro.

    Returns ``(kept, dropped_texts)``. Speech is continuous: consecutive real
    utterances are separated by a breath at most. Hallucinations over music are
    the opposite - brief lines scattered across a long stretch of silence. So
    walk backwards from the end and keep extending the tail while the lines are
    short and the running average spacing stays wide. A dense burst inside the
    tail does not end the scan, because the wide spacing either side dominates
    the average.

    The scan can swallow the closing line of the real narration, which is also
    short and followed by the outro's silence. Such a line is distinctive rather
    than repetitive, so a run is only accepted when most of its lines repeat the
    same text.

    A run must be at least 3 lines long, and shorter than the whole transcript,
    so an ordinary exchange is never touched.
    """
    if len(utts) < _LOOP_MIN_RUN:
        return utts, []
    if not _is_short(utts[-1].get("text", "")):
        return utts, []

    start = len(utts) - 1
    gap_total = 0
    gap_count = 0
    while start > 0:
        if not _is_short(utts[start - 1].get("text", "")):
            break
        gap = utts[start]["start_time"] - utts[start - 1]["end_time"]
        gap_total += gap
        gap_count += 1
        if gap_total / gap_count < _LOOP_MIN_GAP_MS:
            # The tail so far is spoken, not scattered: stop here.
            break
        start -= 1

    run = utts[start:]
    if len(run) < _LOOP_MIN_RUN:
        return utts, []
    counts: dict[str, int] = {}
    for item in run:
        key = _normalize_key(item.get("text", ""))
        counts[key] = counts.get(key, 0) + 1
    repeated = max(counts.values(), default=0)
    if repeated < _LOOP_MIN_REPEATS or repeated < len(run) * _LOOP_MIN_REPEAT_RATIO:
        # Distinct lines scattered over the outro are not the Whisper loop, and
        # two repeats are not enough to call it one.
        return utts, []
    if len(counts) == 1 and start == 0:
        # The whole transcript is one repeated line: refuse to delete all of it.
        return utts, []
    # The scan can start on the closing line of the real narration, which is
    # also short and followed by the outro's silence. Hand back the leading
    # lines that do not repeat: a genuine closing line is distinctive, while the
    # loop itself is the repeated text.
    repeated_key = max(counts, key=lambda key: counts[key])
    while start < len(utts) - _LOOP_MIN_RUN:
        if _normalize_key(utts[start].get("text", "")) == repeated_key:
            break
        start += 1
    run = utts[start:]
    if len(run) < _LOOP_MIN_RUN:
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
