from __future__ import annotations

import json
import os
import shutil
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Callable

from pydub import AudioSegment

from ..audio_mode import is_original_audio, target_text
from .original_audio import write_original_target_audio


class IndexTTSApiError(RuntimeError):
    pass


def _api_url() -> str:
    return (os.getenv("INDEXTTS_API_URL") or "http://127.0.0.1:9005").rstrip("/")


def _voices_dir() -> Path:
    value = (os.getenv("INDEXTTS_VOICES_DIR") or "").strip()
    if not value:
        raise IndexTTSApiError(
            "已启用 IndexTTS API，但未配置 INDEXTTS_VOICES_DIR。"
            "请填写 IndexTTS 服务实际使用的 voices 文件夹路径。"
        )
    path = Path(value).expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def _request_json(path: str, payload: dict) -> bytes:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        f"{_api_url()}/{path.lstrip('/')}",
        data=body,
        headers={"Content-Type": "application/json", "Accept": "audio/wav, application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=float(os.getenv("INDEXTTS_API_TIMEOUT", "600"))) as response:
            content_type = response.headers.get("Content-Type", "")
            result = response.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:1000]
        raise IndexTTSApiError(f"IndexTTS API 返回 HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise IndexTTSApiError(f"无法连接 IndexTTS API {_api_url()}: {exc.reason}") from exc

    if "application/json" in content_type or result[:1] == b"{":
        try:
            data = json.loads(result.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise IndexTTSApiError("IndexTTS API 返回了无法解析的 JSON。") from exc
        if not data.get("ok", True):
            raise IndexTTSApiError(str(data.get("error") or data))
        output_path = data.get("path")
        if not output_path:
            raise IndexTTSApiError("IndexTTS API JSON 没有返回 path。")
        try:
            return Path(output_path).read_bytes()
        except OSError as exc:
            raise IndexTTSApiError(f"IndexTTS API 返回的音频文件无法读取: {output_path}") from exc
    return result


def _longest_reference(vocals_dir: Path, exclude: set[int] | None = None) -> Path:
    excluded = exclude or set()
    files = [
        path
        for path in sorted(vocals_dir.glob("*.wav"))
        if not (path.stem.isdigit() and int(path.stem) in excluded)
    ]
    if not files:
        raise IndexTTSApiError("没有找到可用于 IndexTTS 克隆音色的参考音频。")
    return max(files, key=lambda path: len(AudioSegment.from_file(path)))


def _pick_reference(vocals_dir: Path, index: int, fallback: Path) -> Path:
    segment = vocals_dir / f"{index:04d}.wav"
    if segment.exists() and len(AudioSegment.from_file(segment)) > 0:
        return segment
    return fallback


def _copy_reference(reference: Path, voices_dir: Path) -> str:
    configured_name = (os.getenv("INDEXTTS_REFERENCE_NAME") or "").strip()
    filename = configured_name or f"youdub_reference_{reference.stem}{reference.suffix}"
    target = voices_dir / Path(filename).name
    if target.resolve() != reference.resolve():
        shutil.copy2(reference, target)
    return target.name


def _text(item: dict, index: int) -> str:
    value = target_text(item)
    if not isinstance(value, str) or not value.strip():
        raise IndexTTSApiError(
            f"第 {index} 段译文为空，无法合成语音。"
            "如果该段是非语言人声，请在翻译结果里把 audio_mode 标为 original。"
        )
    return " ".join(value.replace("\n", " ").split())


def _optional_payload() -> dict:
    payload: dict = {
        "speaker": os.getenv("INDEXTTS_SPEAKER", "default"),
        "lang": os.getenv("INDEXTTS_LANG", "ZH"),
        "return_type": "file",
    }
    mappings = {
        "duration_factor": ("INDEXTTS_DURATION_FACTOR", float),
        "diffusion_steps": ("INDEXTTS_DIFFUSION_STEPS", int),
        "segment_pause_ms": ("INDEXTTS_SEGMENT_PAUSE_MS", int),
        "fade_out_ms": ("INDEXTTS_FADE_OUT_MS", int),
        "max_text_tokens_per_segment": ("INDEXTTS_MAX_TEXT_TOKENS", int),
        "emo_control_method": ("INDEXTTS_EMO_CONTROL_METHOD", int),
        "emo_weight": ("INDEXTTS_EMO_WEIGHT", float),
        "do_sample": ("INDEXTTS_DO_SAMPLE", lambda value: value.lower() == "true"),
        "top_p": ("INDEXTTS_TOP_P", float),
        "top_k": ("INDEXTTS_TOP_K", int),
        "temperature": ("INDEXTTS_TEMPERATURE", float),
        "length_penalty": ("INDEXTTS_LENGTH_PENALTY", float),
        "num_beams": ("INDEXTTS_NUM_BEAMS", int),
        "repetition_penalty": ("INDEXTTS_REPETITION_PENALTY", float),
        "max_mel_tokens": ("INDEXTTS_MAX_MEL_TOKENS", int),
    }
    for field, (name, converter) in mappings.items():
        raw = os.getenv(name)
        if raw is not None and raw.strip():
            payload[field] = converter(raw.strip())
    fp16 = os.getenv("INDEXTTS_FP16")
    if fp16 is not None and fp16.strip():
        payload["fp16"] = fp16.strip().lower() == "true"
    for field, name in (("emo_ref_audio", "INDEXTTS_EMO_REF_AUDIO"), ("emo_text", "INDEXTTS_EMO_TEXT")):
        value = os.getenv(name, "").strip()
        if value:
            payload[field] = value
    vector = os.getenv("INDEXTTS_EMO_VECTOR", "").strip()
    if vector:
        payload["emo_vector"] = [float(value.strip()) for value in vector.split(",")]
    return payload


def generate_tts(
    translation_file: Path,
    vocals_dir: Path,
    session: Path,
    progress_callback: Callable[[int, str], None] | None = None,
    *,
    original_vocals_file: Path | None = None,
) -> Path:
    output_dir = session / "segments" / "tts"
    output_dir.mkdir(parents=True, exist_ok=True)
    items = json.loads(translation_file.read_text(encoding="utf-8"))["translation"]
    if not items:
        if progress_callback:
            progress_callback(100, "No TTS clips to generate")
        return output_dir

    if any(is_original_audio(item) for item in items) and original_vocals_file is None:
        raise IndexTTSApiError(
            "翻译结果里存在需要保留原声的片段（audio_mode=original），"
            "但没有提供原始人声音轨，无法生成该片段的音频。"
        )

    original_indices = {
        index for index, item in enumerate(items, start=1) if is_original_audio(item)
    }
    voices_dir = _voices_dir()
    base_payload = _optional_payload()
    fallback_reference: Path | None = None

    for index, item in enumerate(items, start=1):
        output_file = output_dir / f"{index:04d}.wav"
        if is_original_audio(item):
            # Non-verbal / filler segments keep the source performance. merge_tts_audio
            # still expects one wav per item, so always refresh this slot from the
            # original vocals track.
            assert original_vocals_file is not None
            write_original_target_audio(output_file, item, original_vocals_file)
        elif not output_file.exists():
            if fallback_reference is None:
                fallback_reference = _longest_reference(vocals_dir, exclude=original_indices)
            reference = _pick_reference(vocals_dir, index, fallback_reference)
            reference_name = _copy_reference(reference, voices_dir)
            payload = {
                **base_payload,
                "audio": reference_name,
                "text": _text(item, index),
            }
            output_file.write_bytes(_request_json("tts", payload))
        if progress_callback:
            progress_callback(
                round(index / len(items) * 100),
                f"Prepared {index}/{len(items)} IndexTTS clips with per-segment voice reference",
            )
    return output_dir
