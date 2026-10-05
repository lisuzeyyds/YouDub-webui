from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Literal

from . import runtime_security


AudioMode = Literal["tts", "original"]


def target_text(item: dict) -> object:
    return item.get("dst") or item.get("zh", "")


def audio_mode(item: dict) -> AudioMode:
    mode = item.get("audio_mode")
    if mode is not None:
        if mode not in {"tts", "original"}:
            raise ValueError("audio_mode must be one of: tts, original")
        return mode

    text = target_text(item)
    return "original" if isinstance(text, str) and not text.strip() else "tts"


def is_original_audio(item: dict) -> bool:
    return audio_mode(item) == "original"


_WHITESPACE_RE = re.compile(r"\s+")


def coerce_empty_tts_segments_to_original(translation_file: Path) -> int:
    """Keep the source performance for segments that ended up without target text.

    The translator is instructed to blank out pure speech fillers and mark them as
    `original`. Older or hand-edited artifacts may leave such a segment tagged `tts`
    with an empty string, which would make every TTS backend fail the whole task.
    Rewriting the artifact here keeps TTS and the later merges consistent.
    """
    data = json.loads(translation_file.read_text(encoding="utf-8"))
    items = data.get("translation")
    if not isinstance(items, list):
        return 0

    changed = 0
    for item in items:
        if not isinstance(item, dict) or item.get("audio_mode") != "tts":
            continue
        text = target_text(item)
        if isinstance(text, str) and not _WHITESPACE_RE.sub("", text):
            item["audio_mode"] = "original"
            changed += 1

    if changed:
        runtime_security.atomic_write_private_text(
            translation_file,
            json.dumps(data, ensure_ascii=False, indent=2),
        )
    return changed


def sanitize_invalid_segment_ranges(
    translation_file: Path,
    duration_ms: int | None = None,
) -> int:
    """Drop translation rows whose time range cannot exist in the audio.

    Whisper can emit trailing utterances timestamped past the end of the audio;
    an older ``asr_fix`` clamped ``end_time`` to the duration without clamping
    ``start_time``, leaving rows with ``start > end``. The TTS backends then try
    to slice an impossible original-audio range (or a zero-length clip) and fail
    the whole task, even though every other segment was fine.

    Dropping the row here keeps TTS and the later merges consistent, because all
    of them read this same artifact. Returns the number of dropped rows.
    """
    data = json.loads(translation_file.read_text(encoding="utf-8"))
    items = data.get("translation")
    if not isinstance(items, list):
        return 0

    kept: list[dict] = []
    dropped = 0
    changed = False
    for item in items:
        if not isinstance(item, dict):
            dropped += 1
            continue
        # Only validate rows that actually carry a time range. Rows without
        # timestamps (older artifacts, hand-written fixtures) are passed through
        # untouched - defaulting a missing range to 0-0 would drop them all.
        if "start_time" not in item and "end_time" not in item:
            kept.append(item)
            continue
        try:
            start = int(item["start_time"])
            end = int(item["end_time"])
        except (KeyError, TypeError, ValueError):
            dropped += 1
            continue
        if duration_ms:
            if start >= duration_ms:
                dropped += 1
                continue
            end = min(end, duration_ms)
        if end <= start:
            dropped += 1
            continue
        if start != item.get("start_time") or end != item.get("end_time"):
            # 只是收窄到音频范围内，行本身保留
            changed = True
            item = {**item, "start_time": start, "end_time": end}
        kept.append(item)

    if dropped or changed:
        data["translation"] = kept
        runtime_security.atomic_write_private_text(
            translation_file,
            json.dumps(data, ensure_ascii=False, indent=2),
        )
    return dropped
