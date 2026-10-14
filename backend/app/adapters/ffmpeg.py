from __future__ import annotations

import json
import os
import re
import subprocess
import time
import uuid
from pathlib import Path

from ..audio_mode import target_text
from ..config import ffmpeg_binary, ffprobe_binary

# Windows: never allocate a console for ffmpeg. Console allocation is a known
# failure source for children spawned from a console-less service worker (and
# it would otherwise create stray console windows).
_NO_WINDOW = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0


def _run_ffmpeg(command: list[str], cwd: Path | None = None) -> None:
    """Run ffmpeg with piped stdio, a detached stdin, and one retry.

    ffmpeg spawned with inherited stdio handles from a background service
    worker on Windows exits immediately with code 1 and no output. Piping
    the streams makes it run reliably and lets us surface the real error.
    stdin is set to DEVNULL as well: the worker's own stdin can be a stale
    console/pipe handle, and ffmpeg reads stdin for interactive keys.
    A silent exit (rc != 0 with empty stderr) is retried once, as it has
    been observed to clear on its own.
    """
    attempts = 2
    result = None
    for attempt in range(1, attempts + 1):
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=cwd,
            stdin=subprocess.DEVNULL,
            creationflags=_NO_WINDOW,
        )
        if result.returncode == 0:
            return
        if attempt < attempts:
            try:
                debug_path = Path("data/tmp/ffmpeg_debug.log")
                debug_path.parent.mkdir(parents=True, exist_ok=True)
                with open(debug_path, "a", encoding="utf-8") as handle:
                    handle.write(
                        f"=== _run_ffmpeg retry: rc={result.returncode} "
                        f"stderr_len={len(result.stderr or '')} cmd={command[:3]} ===\n"
                    )
                    handle.write((result.stderr or "")[-2000:] + "\n")
            except OSError:
                pass
            time.sleep(5)
    if result is not None and result.returncode != 0:
        raise RuntimeError(
            "ffmpeg failed (exit {}): {}".format(
                result.returncode,
                (result.stderr or "").strip().splitlines()[-1:] or ["no output"],
            )
        )

SUBTITLE_PUNCTUATION = {"，", ",", "；", ";", "：", ":", "。", "?", "？", "!", "！", "、"}
SUBTITLE_PROTECTED_PAIRS = {"《": "》", "（": "）", "【": "】", "「": "」", "『": "』"}
SUBTITLE_CLOSING_QUOTES = {'"', "'", "」", "』", "》", "）", "】", "\u201d", "\u2019", "]"}
SUBTITLE_MIN_FRAGMENT_LEN = 5
SUBTITLE_MIN_DURATION_MS = 200
SUBTITLE_TAIL_BUFFER_MS = 100
SUBTITLE_DURATION_FLOOR_MS = 600


SUBTITLE_FONTS = {
    "zh": "Noto Sans CJK SC",
    "en": "Arial",
}

SUBTITLE_FONT_SIZES = {
    "zh": {"portrait": 12, "landscape": 24},
    "en": {"portrait": 9, "landscape": 18},
}


def _subtitle_style(font: str, size: int, margin_v: int) -> str:
    return (
        f"FontName={font},"
        f"FontSize={size},"
        "PrimaryColour=&H00FFFFFF,"
        "OutlineColour=&H00000000,"
        "BorderStyle=1,"
        "Outline=2,"
        "Alignment=2,"
        f"MarginV={margin_v}"
    )


def _srt_time(ms: int) -> str:
    hours = ms // 3_600_000
    ms -= hours * 3_600_000
    minutes = ms // 60_000
    ms -= minutes * 60_000
    seconds = ms // 1000
    millis = ms - seconds * 1000
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"


def _split_protected(text: str) -> list[str]:
    segments: list[str] = []
    buf: list[str] = []
    inside = None
    for ch in text:
        if inside is None and ch in SUBTITLE_PROTECTED_PAIRS:
            inside = SUBTITLE_PROTECTED_PAIRS[ch]
            buf.append(ch)
            continue
        if inside is not None and ch == inside:
            inside = None
            buf.append(ch)
            continue
        if inside is None and ch in SUBTITLE_PUNCTUATION:
            chunk = "".join(buf).strip()
            if chunk:
                segments.append(chunk)
            buf.clear()
            continue
        buf.append(ch)
    tail = "".join(buf).strip()
    if tail:
        segments.append(tail)
    return segments


def _attach_closing_quotes(segments: list[str]) -> list[str]:
    fixed: list[str] = []
    for seg in segments:
        if seg and seg[0] in SUBTITLE_CLOSING_QUOTES and fixed:
            fixed[-1] = f"{fixed[-1]}{seg}".strip()
            continue
        fixed.append(seg.strip())
    return fixed


def _merge_short_fragments(segments: list[str]) -> list[str]:
    merged: list[str] = []
    i = 0
    while i < len(segments):
        cur = segments[i]
        if len(cur.strip()) < SUBTITLE_MIN_FRAGMENT_LEN and i + 1 < len(segments):
            segments[i + 1] = f"{cur}{segments[i + 1]}".strip()
            i += 1
            continue
        merged.append(cur)
        i += 1
    return merged


def _strip_trailing_punct(segments: list[str]) -> list[str]:
    cleaned: list[str] = []
    for item in segments:
        text = item.strip()
        if not text:
            continue
        if text.endswith(("，", ",", "。")):
            text = text[:-1]
        cleaned.append(re.sub(r"\s+", " ", text).strip())
    return cleaned


def split_subtitle_text(text: str) -> list[str]:
    original = (text or "").strip()
    if not original:
        return []
    segments = _split_protected(original)
    if not segments:
        return [original]
    segments = _attach_closing_quotes(segments)
    segments = _merge_short_fragments(segments)
    cleaned = _strip_trailing_punct(segments)
    return cleaned or [original]


def _allocate_durations(fragments: list[str], total_duration: int) -> list[int]:
    if len(fragments) == 1:
        return [total_duration]
    weights = [max(1, len(f.replace(" ", ""))) for f in fragments]
    total_weight = sum(weights)
    durations: list[int] = []
    allocated = 0
    for i, weight in enumerate(weights[:-1]):
        share = round(total_duration * weight / total_weight)
        if total_duration >= SUBTITLE_DURATION_FLOOR_MS:
            ceiling = total_duration - allocated - SUBTITLE_TAIL_BUFFER_MS
            share = max(SUBTITLE_MIN_DURATION_MS, min(share, ceiling))
        else:
            share = max(int(SUBTITLE_MIN_DURATION_MS / 2), share)
        durations.append(share)
        allocated += share
    durations.append(max(SUBTITLE_TAIL_BUFFER_MS, total_duration - allocated))
    return durations


def _segment_times(item: dict) -> tuple[int, int]:
    # actual_* is written by merge_tts_audio and already lives on the finished
    # timeline; out_* is the same timeline before stretching; start/end is the
    # source video. Fall back in that order.
    start = item.get("actual_start_time", item.get("out_start_time"))
    end = item.get("actual_end_time", item.get("out_end_time"))
    if start is None:
        start = item["start_time"]
    if end is None:
        end = item["end_time"]
    return int(start), int(end)


def _dst_lang(translation: list[dict]) -> str:
    for item in translation:
        lang = item.get("dst_lang")
        if lang:
            return lang
    return "zh"


def write_srt(translation_file: Path, session: Path) -> Path:
    data = json.loads(translation_file.read_text(encoding="utf-8"))
    translation = data["translation"]
    dst_lang = _dst_lang(translation)
    output_file = session / "metadata" / f"subtitles.{dst_lang}.srt"
    lines: list[str] = []
    idx = 1
    for item in translation:
        start, end = _segment_times(item)
        if end <= start:
            continue
        text = target_text(item)
        fragments = split_subtitle_text(text if isinstance(text, str) else "")
        if not fragments:
            continue
        cursor = start
        for fragment, duration in zip(fragments, _allocate_durations(fragments, end - start)):
            lines.extend([str(idx), f"{_srt_time(cursor)} --> {_srt_time(cursor + duration)}", fragment, ""])
            cursor += duration
            idx += 1
    output_file.write_text("\n".join(lines), encoding="utf-8")
    return output_file


def probe_has_audio(video_file: Path) -> bool:
    """Whether the file carries at least one audio stream."""
    result = subprocess.run(
        [
            ffprobe_binary(),
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=index",
            "-of",
            "csv=p=0",
            str(video_file),
        ],
        capture_output=True,
        text=True,
    )
    return result.returncode == 0 and bool(result.stdout.strip())


def build_cut_expression(spans: list[tuple[int, int]]) -> str:
    """A ``select``/``aselect`` expression that keeps everything *but* ``spans``.

    Judging each frame against its own source timestamp (instead of re-timing
    the keeps) means the per-cut frame rounding never accumulates: only the very
    end of the output can drift by one frame.
    """
    clauses = "+".join(
        f"between(t,{start / 1000:.3f},{end / 1000:.3f})" for start, end in spans
    )
    return f"not({clauses})"


def build_cut_chains(
    cut_spans: list[tuple[int, int]],
    *,
    subtitle_filter_str: str | None = None,
    audio_source: str | None = None,
) -> list[str]:
    """``filter_complex`` chains that remove ``cut_spans`` down to ``[vout]``/``[aout]``.

    Video and audio share the same expression so both sides cut at the same
    instants and stay in sync. The subtitle overlay runs *after* the video is
    re-timed, so the burned-in text follows the cut timeline.
    """
    expression = build_cut_expression(cut_spans)
    video = f"[0:v]select='{expression}',setpts=N/FRAME_RATE/TB"
    if subtitle_filter_str:
        video += f",{subtitle_filter_str}"
    chains = [f"{video}[vout]"]
    if audio_source:
        chains.append(f"{audio_source}aselect='{expression}',asetpts=N/SR/TB[aout]")
    return chains


def probe_video_size(video_file: Path) -> tuple[int, int] | None:
    result = subprocess.run(
        [
            ffprobe_binary(),
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height",
            "-of",
            "csv=p=0",
            str(video_file),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None

    lines = result.stdout.strip().splitlines()
    if not lines:
        return None
    parts = lines[0].split(",", maxsplit=1)
    if len(parts) != 2:
        return None
    try:
        return int(parts[0]), int(parts[1])
    except ValueError:
        return None


def get_video_orientation(video_file: Path) -> str:
    size = probe_video_size(video_file)
    if size is None:
        return "landscape"
    width, height = size
    return "portrait" if height > width else "landscape"


def subtitle_style_for_orientation(orientation: str, font: str, lang: str = "zh") -> str:
    sizes = SUBTITLE_FONT_SIZES.get(lang, SUBTITLE_FONT_SIZES["zh"])
    margin_v = 70 if orientation == "portrait" else 5
    return _subtitle_style(font, size=sizes[orientation], margin_v=margin_v)


def _subtitle_filter_path(subtitle_file: Path, session: Path) -> str:
    try:
        return subtitle_file.resolve().relative_to(session.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError("Subtitle file must be inside the session directory.") from exc


def subtitle_filter(video_file: Path, subtitle_file: Path, session: Path) -> str:
    lang = subtitle_file.stem.rsplit(".", 1)[-1]
    font = SUBTITLE_FONTS.get(lang, "Arial")
    style = subtitle_style_for_orientation(get_video_orientation(video_file), font, lang)
    sub_path = _subtitle_filter_path(subtitle_file, session)
    return f"subtitles=filename='{sub_path}':force_style='{style}'"


def merge_video(
    video_file: Path,
    dubbing_file: Path | None,
    bgm_file: Path | None,
    timings_file: Path,
    session: Path,
    *,
    output_mode: str = "both",
    cut_spans: list[tuple[int, int]] | None = None,
) -> Path:
    if output_mode not in {"subtitles", "dubbing", "both"}:
        raise ValueError("output_mode must be one of: subtitles, dubbing, both")

    include_dubbing = output_mode in {"dubbing", "both"}
    include_subtitles = output_mode in {"subtitles", "both"}
    if include_dubbing and (dubbing_file is None or bgm_file is None):
        raise ValueError("Dubbing and background audio are required for this output mode.")

    tmp_dir = session / "tmp"
    media_dir = session / "media"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    media_dir.mkdir(parents=True, exist_ok=True)
    final_video = media_dir / "video_final.mp4"

    session_dir = session.resolve()
    video_input = video_file.resolve()

    # Ad breaks to physically remove (source timestamps, ms).
    spans = sorted(
        (int(start), int(end)) for start, end in (cut_spans or []) if int(end) > int(start)
    )

    mixed_audio_output: Path | None = None
    if include_dubbing:
        assert dubbing_file is not None
        assert bgm_file is not None
        dubbing_input = dubbing_file.resolve()
        bgm_input = bgm_file.resolve()
        mixed_audio_output = (tmp_dir / "audio_mixed.m4a").resolve()
        if spans:
            # The dubbing track already lives on the *finished* timeline:
            # merge_tts_audio lays every segment out on its out_* timestamps,
            # which are the post-cut ones. Only the background track still
            # addresses the source video, so only it may be cut here. Cutting
            # the mix instead re-applied the source-timeline spans to dubbing
            # that had already been re-timed, which shifted everything after
            # the first ad break by its length - measured on a real task: the
            # dubbing ran 58s ahead of the burned-in subtitles (and the
            # picture) from the 107s mark on.
            bgm_input = (tmp_dir / "audio_bgm_cut.m4a").resolve()
            _run_ffmpeg(
                [
                    ffmpeg_binary(),
                    "-y",
                    "-i",
                    str(bgm_file.resolve()),
                    "-filter_complex",
                    f"[0:a]aselect='{build_cut_expression(spans)}',asetpts=N/SR/TB[aout]",
                    "-map",
                    "[aout]",
                    "-c:a",
                    "aac",
                    str(bgm_input),
                ],
            )
        _run_ffmpeg(
            [
                ffmpeg_binary(),
                "-y",
                "-i",
                str(dubbing_input),
                "-i",
                str(bgm_input),
                "-filter_complex",
                "[0:a]volume=1.0[a0];[1:a]volume=0.30[a1];[a0][a1]amix=inputs=2:duration=longest:normalize=0[aout]",
                "-map",
                "[aout]",
                "-c:a",
                "aac",
                str(mixed_audio_output),
            ],
        )

    temporary_video = media_dir / f".video_final.{uuid.uuid4().hex}.mp4"
    try:
        command = [ffmpeg_binary(), "-y", "-i", str(video_input)]
        if mixed_audio_output is not None:
            command.extend(["-i", str(mixed_audio_output)])
        subtitles = write_srt(timings_file, session) if include_subtitles else None

        if spans:
            # Only the picture is re-timed here. With dubbing, both audio inputs
            # are already on the cut timeline (the dubbing track is authored on
            # it, the BGM was cut before mixing) - running the source-timeline
            # selection over them again is what shifted the dubbing ahead of
            # the subtitles by the first break's length. Without dubbing, the
            # video's own audio still addresses the source and is cut with it.
            if mixed_audio_output is not None:
                audio_source = None
            else:
                audio_source = "[0:a]" if probe_has_audio(video_input) else None
            subtitle_str = (
                subtitle_filter(video_input, subtitles, session_dir)
                if subtitles is not None
                else None
            )
            chains = build_cut_chains(
                spans,
                subtitle_filter_str=subtitle_str,
                audio_source=audio_source,
            )
            command.extend(["-filter_complex", ";".join(chains), "-map", "[vout]"])
            if mixed_audio_output is not None:
                command.extend(["-map", "1:a:0"])
            elif audio_source is not None:
                command.extend(["-map", "[aout]"])
            command.extend(["-c:v", "libx264", "-preset", "fast", "-crf", "23"])
            if mixed_audio_output is not None or audio_source is not None:
                command.extend(["-c:a", "aac"])
            command.extend(["-movflags", "+faststart"])
            if mixed_audio_output is not None or audio_source is not None:
                # Frame-accurate cuts leave video and audio a frame apart.
                command.append("-shortest")
        else:
            if subtitles is not None:
                command.extend(["-vf", subtitle_filter(video_input, subtitles, session_dir)])
            command.extend(["-map", "0:v:0"])
            if mixed_audio_output is not None:
                command.extend(["-map", "1:a:0"])
            else:
                command.extend(["-map", "0:a?"])
            command.extend(["-c:v", "libx264", "-preset", "fast", "-crf", "23"])
            command.extend(["-c:a", "aac"])
            command.extend(["-movflags", "+faststart"])
            if mixed_audio_output is not None:
                command.append("-shortest")
        command.append(str(temporary_video.resolve()))
        _run_ffmpeg(command, cwd=session_dir)
        temporary_video.replace(final_video)
        return final_video
    except Exception:
        temporary_video.unlink(missing_ok=True)
        raise
