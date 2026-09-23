from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class StageSpec:
    name: str
    label: str


STAGES: tuple[StageSpec, ...] = (
    StageSpec("download", "Download"),
    StageSpec("separate", "Demucs"),
    StageSpec("asr", "Whisper"),
    StageSpec("asr_fix", "Split sentences"),
    StageSpec("translate", "Translate"),
    StageSpec("split_audio", "Split audio"),
    StageSpec("tts", "VoxCPM"),
    StageSpec("merge_audio", "Merge audio"),
    StageSpec("merge_video", "Merge video"),
)


STAGE_NAMES = tuple(stage.name for stage in STAGES)


def tts_label() -> str:
    """根据环境变量动态返回 TTS 阶段的展示名。

    - 配了 INDEXTTS_API_URL → "IndexTTS"（走本地 IndexTTS API）
    - 否则默认 "VoxCPM"
    """
    return "IndexTTS" if (os.getenv("INDEXTTS_API_URL") or "").strip() else "VoxCPM"


def stage_label(stage: StageSpec) -> str:
    """返回阶段的展示名；TTS 阶段会根据环境变量切换。"""
    if stage.name == "tts":
        return tts_label()
    return stage.label

