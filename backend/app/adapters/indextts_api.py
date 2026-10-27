from __future__ import annotations

import json
import os
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
    """Base URL of the IndexTTS2 v3 service (``http://127.0.0.1:7860``)."""
    return (os.getenv("INDEXTTS_API_URL") or "http://127.0.0.1:7860").rstrip("/")


def _timeout() -> float:
    return float(os.getenv("INDEXTTS_API_TIMEOUT", "600"))


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


def _text(item: dict, index: int) -> str:
    value = target_text(item)
    if not isinstance(value, str) or not value.strip():
        raise IndexTTSApiError(
            f"第 {index} 段译文为空，无法合成语音。"
            "如果该段是非语言人声，请在翻译结果里把 audio_mode 标为 original。"
        )
    return " ".join(value.replace("\n", " ").split())


def _optional_form() -> list[tuple[str, str]]:
    """Extra multipart fields from the environment, mirroring the v3 API defaults."""
    fields: list[tuple[str, str]] = [
        ("emo_mode", os.getenv("INDEXTTS_EMO_MODE", "0")),
        ("use_random", os.getenv("INDEXTTS_USE_RANDOM", "false")),
    ]
    mappings = {
        "emo_weight": ("INDEXTTS_EMO_WEIGHT",),
        "max_text_tokens_per_segment": ("INDEXTTS_MAX_TEXT_TOKENS",),
        "do_sample": ("INDEXTTS_DO_SAMPLE",),
        "top_p": ("INDEXTTS_TOP_P",),
        "top_k": ("INDEXTTS_TOP_K",),
        "temperature": ("INDEXTTS_TEMPERATURE",),
        "length_penalty": ("INDEXTTS_LENGTH_PENALTY",),
        "num_beams": ("INDEXTTS_NUM_BEAMS",),
        "repetition_penalty": ("INDEXTTS_REPETITION_PENALTY",),
        "max_mel_tokens": ("INDEXTTS_MAX_MEL_TOKENS",),
        "interval_silence": ("INDEXTTS_SEGMENT_PAUSE_MS",),
    }
    for field, (name,) in mappings.items():
        raw = os.getenv(name)
        if raw is not None and raw.strip():
            fields.append((field, raw.strip()))
    emo_text = os.getenv("INDEXTTS_EMO_TEXT", "").strip()
    if emo_text:
        fields.extend([("emo_mode", "3"), ("emo_text", emo_text)])
    return fields


def _build_multipart(
    text: str,
    reference: Path,
    fields: list[tuple[str, str]],
) -> tuple[bytes, str]:
    """Encode a multipart/form-data body with ``text``, ``spk_audio`` and extras.

    Built by hand so the reference audio is uploaded as a file part (the v3
    service requires the binary, not a filename) and so tests can stub the
    network without a real socket. Returns ``(body, content_type)``.
    """
    boundary = "----youdubindexttsv3boundary"
    lines: list[bytes] = []
    for name, value in [("text", text), *fields]:
        lines.append(f"--{boundary}\r\n".encode("utf-8"))
        lines.append(f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode("utf-8"))
        lines.append(value.encode("utf-8"))
        lines.append(b"\r\n")
    lines.append(f"--{boundary}\r\n".encode("utf-8"))
    lines.append(
        f'Content-Disposition: form-data; name="spk_audio"; filename="{reference.name}"\r\n'.encode("utf-8")
    )
    lines.append(b"Content-Type: audio/wav\r\n\r\n")
    lines.append(reference.read_bytes())
    lines.append(b"\r\n")
    lines.append(f"--{boundary}--\r\n".encode("utf-8"))
    body = b"".join(lines)
    return body, f"multipart/form-data; boundary={boundary}"


def _read_sse(body: bytes) -> dict:
    """Pull the first terminal event (``done`` / ``error``) out of an SSE stream."""
    for raw_line in body.splitlines():
        line = raw_line.strip()
        if not line.startswith(b"data:"):
            continue
        payload = line[5:].strip()
        if not payload:
            continue
        try:
            event = json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
        if event.get("type") in {"done", "error", "stopped"}:
            return event
    return {}


def _synthesize(text: str, reference: Path, fields: list[tuple[str, str]]) -> bytes:
    """POST /api/v1/tts, follow the SSE stream, and return the wav bytes.

    The v3 service streams progress events and ends with a ``done`` event whose
    ``wav`` field is a relative path (``/api/v1/audio/...``). The audio is then
    fetched in a second request. Any ``error`` event fails with its ``detail``.
    """
    body, content_type = _build_multipart(text, reference, fields)
    request = urllib.request.Request(
        f"{_api_url()}/api/v1/tts",
        data=body,
        headers={"Content-Type": content_type, "Accept": "text/event-stream"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=_timeout()) as response:
            stream = response.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:1000]
        raise IndexTTSApiError(f"IndexTTS API 返回 HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise IndexTTSApiError(f"无法连接 IndexTTS API {_api_url()}: {exc.reason}") from exc

    event = _read_sse(stream)
    if event.get("type") == "error":
        raise IndexTTSApiError(f"IndexTTS 合成失败: {event.get('detail') or event}")
    wav_path = event.get("wav")
    if not wav_path:
        raise IndexTTSApiError("IndexTTS 流没有返回 wav 路径。")

    wav_url = wav_path if wav_path.startswith("http") else f"{_api_url()}{wav_path}"
    # The filename may contain Chinese, which urllib refuses to send un-encoded.
    scheme, netloc, path, query, fragment = urllib.parse.urlsplit(wav_url)
    path = urllib.parse.quote(path)
    wav_url = urllib.parse.urlunsplit((scheme, netloc, path, query, fragment))
    try:
        with urllib.request.urlopen(wav_url, timeout=_timeout()) as response:
            return response.read()
    except urllib.error.HTTPError as exc:
        raise IndexTTSApiError(f"下载 IndexTTS 音频失败 HTTP {exc.code}: {wav_path}") from exc
    except urllib.error.URLError as exc:
        raise IndexTTSApiError(f"下载 IndexTTS 音频失败 {wav_path}: {exc.reason}") from exc


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
    fields = _optional_form()
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
            output_file.write_bytes(_synthesize(_text(item, index), reference, fields))
        if progress_callback:
            progress_callback(
                round(index / len(items) * 100),
                f"Prepared {index}/{len(items)} IndexTTS clips with per-segment voice reference",
            )
    return output_dir
