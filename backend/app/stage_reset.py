from __future__ import annotations

import shutil
from pathlib import Path

from .sources import SourceConfig
from .stages import STAGE_NAMES


STAGE_OWN_ARTIFACTS: dict[str, tuple[str, ...]] = {
    "download": ("media", "metadata", "segments", "tmp"),
    "separate": ("media/audio_vocals.wav", "media/audio_bgm.wav"),
    "asr": ("metadata/asr.json",),
    "asr_fix": ("metadata/asr_fixed.json",),
    "translate": (
        "metadata/translation_preprocess.json",
        "metadata/bilibili_meta.json",
        "metadata/ad_cut.json",
    ),
    "split_audio": ("segments/vocals",),
    "tts": ("segments/tts",),
    "merge_audio": ("tmp/audio_dubbing.wav", "metadata/timings.json", "segments/stretched"),
    "merge_video": ("tmp/audio_mixed.m4a", "media/video_final.mp4"),
}


def _translation_globs(session: Path, target_language: str) -> list[Path]:
    metadata = session / "metadata"
    paths = list(metadata.glob(f"translation.{target_language}.json"))
    paths.extend(metadata.glob("subtitles.*.srt"))
    return paths


def collect_artifact_paths(session: Path, from_stage: str, source: SourceConfig) -> list[Path]:
    if from_stage not in STAGE_NAMES:
        raise ValueError(f"Unknown stage: {from_stage}")

    start = STAGE_NAMES.index(from_stage)
    paths: list[Path] = []
    for stage in STAGE_NAMES[start:]:
        for relative in STAGE_OWN_ARTIFACTS[stage]:
            paths.append(session / relative)
        if stage == "translate":
            paths.extend(_translation_globs(session, source.target_language))
    return paths


def remove_stage_artifacts(session: Path, from_stage: str, source: SourceConfig) -> None:
    for path in collect_artifact_paths(session, from_stage, source):
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
        elif path.exists():
            path.unlink()
