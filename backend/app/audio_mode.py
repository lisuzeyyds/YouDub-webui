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
