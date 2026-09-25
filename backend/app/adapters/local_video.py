from __future__ import annotations

import glob
import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from ..config import ffmpeg_binary
from ..sanitize import sanitize_text
from ..sources import SourceConfig
from ..youtube import local_upload_task_id


def upload_dir(workfolder: Path, task_id: str) -> Path:
    return workfolder / "_uploads" / task_id


def uploaded_video_dir(workfolder: Path, task_id: str) -> Path:
    return upload_dir(workfolder, task_id) / "video"


def remove_upload(workfolder: Path, task_id: str) -> None:
    target = upload_dir(workfolder, task_id)
    if target.exists():
        shutil.rmtree(target)


def _single_file(root: Path, task_id: str, label: str) -> Path:
    files = sorted(path for path in root.iterdir() if path.is_file())
    if not files:
        raise FileNotFoundError(f"Local upload {label} is missing for task {task_id}.")
    if len(files) > 1:
        raise RuntimeError(f"Local upload has multiple {label} files for task {task_id}.")
    return files[0]


def _uploaded_video_file(workfolder: Path, task_id: str) -> Path:
    root = upload_dir(workfolder, task_id)
    if not root.exists():
        raise FileNotFoundError(f"Local upload is missing for task {task_id}.")

    video_root = uploaded_video_dir(workfolder, task_id)
    if video_root.exists():
        return _single_file(video_root, task_id, "video")
    return _single_file(root, task_id, "video")


def _title_from_url(url: str, source_file: Path) -> str:
    query = parse_qs(urlparse(url.strip()).query)
    filename = (query.get("filename") or [""])[0].strip()
    if filename:
        return Path(filename).stem or source_file.stem
    return source_file.stem


def _session_path(workfolder: Path, task_id: str, title: str) -> Path:
    safe_title = sanitize_text(title) or "local-video"
    return workfolder / "local" / f"{safe_title}__{task_id}"


def _debug_log(text: str) -> None:
    try:
        debug_path = Path("data/tmp/ffmpeg_debug.log")
        debug_path.parent.mkdir(parents=True, exist_ok=True)
        with open(debug_path, "a", encoding="utf-8") as handle:
            handle.write(text if text.endswith("\n") else text + "\n")
    except OSError:
        pass


def _ffmpeg_environment_report(binary: str) -> str:
    """Collect what the worker process actually sees when it spawns ffmpeg."""
    lines = [
        f"=== probe @{time.strftime('%Y-%m-%dT%H:%M:%S')} ===",
        f"cwd={Path.cwd()}",
        f"argv0={binary!r}",
        f"which={shutil.which(binary)!r}",
        f"PATH={os.environ.get('PATH', '')}",
    ]
    for label, extra in (("plain", []), ("nostdin", ["-nostdin"]), ("devnull-stdin", [])):
        cmd = [binary, *extra, "-version"]
        kwargs = {"stdin": subprocess.DEVNULL} if label == "devnull-stdin" else {}
        try:
            probe = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=30,
                **kwargs,
            )
            first = [ln for ln in (probe.stderr or probe.stdout or "").splitlines() if ln.strip()]
            lines.append(
                f"probe[{label}] rc={probe.returncode} "
                f"first={first[0][:120] if first else '(EMPTY)'} "
                f"stderr_len={len(probe.stderr or '')}"
            )
        except BaseException as exc:  # noqa: BLE001
            lines.append(f"probe[{label}] EXC {type(exc).__name__}: {str(exc)[:160]}")
    lines.append(f"sibling_ffmpegs={glob.glob(str(Path(binary).parent / 'ffmpeg*')) if os.sep in binary else '(bare name)'}")
    return "\n".join(lines)


def _transcode_to_mp4(source_file: Path, video_file: Path) -> None:
    # ffmpeg can fail spuriously on Windows with exit 1 and no stderr output
    # (process dies before it can even print its banner). Retry a few times
    # with a short backoff and keep diagnostics in data/tmp/ffmpeg_debug.log.
    binary = ffmpeg_binary()
    attempts = 3
    result = None
    last_error = ""
    for attempt in range(1, attempts + 1):
        started = time.time()
        result = subprocess.run(
            [
                binary,
                "-nostdin",
                "-y",
                "-i",
                str(source_file),
                "-map",
                "0:v:0",
                "-map",
                "0:a:0?",
                "-c:v",
                "libx264",
                "-preset",
                "fast",
                "-crf",
                "23",
                "-c:a",
                "aac",
                "-movflags",
                "+faststart",
                str(video_file),
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            stdin=subprocess.DEVNULL,
        )
        elapsed = time.time() - started
        if result.returncode == 0:
            return
        last_error = (
            f"attempt {attempt}/{attempts}: rc={result.returncode} elapsed={elapsed:.2f}s "
            f"stderr_len={len(result.stderr or '')}"
        )
        _debug_log(last_error + "\n" + (result.stderr or "")[-4000:])
        if attempt < attempts:
            video_file.unlink(missing_ok=True)
            time.sleep(5)
    if result is not None and result.returncode != 0:
        _debug_log(_ffmpeg_environment_report(binary))
        raise RuntimeError(
            f"ffmpeg failed with exit code {result.returncode}: "
            f"{(result.stderr or '').strip().splitlines()[-1] if (result.stderr or '').strip() else '(no stderr)'}"
            f" [{last_error}]"
        )


def import_local_video(url: str, workfolder: Path, source: SourceConfig) -> tuple[Path, dict]:
    from .local_subtitles import uploaded_subtitle_file

    task_id = local_upload_task_id(url)
    if not task_id:
        raise ValueError("Invalid local upload URL.")

    source_file = _uploaded_video_file(workfolder, task_id)
    subtitle_file = uploaded_subtitle_file(workfolder, task_id)
    title = _title_from_url(url, source_file)
    session = _session_path(workfolder, task_id, title)
    media_dir = session / "media"
    metadata_dir = session / "metadata"
    media_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    video_file = media_dir / "video_source.mp4"
    info = {
        "id": task_id,
        "title": title,
        "source": "local",
        "webpage_url": url,
        "original_path": str(source_file),
        "asr_language": source.asr_language,
        "target_language": source.target_language,
    }
    if subtitle_file:
        info["subtitle_path"] = str(subtitle_file)
    metadata_file = metadata_dir / "local_info.json"
    metadata_file.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")

    if video_file.exists() and video_file.stat().st_size > 0:
        return session, info

    _transcode_to_mp4(source_file, video_file)
    if not video_file.exists() or video_file.stat().st_size == 0:
        raise RuntimeError("ffmpeg finished without producing media/video_source.mp4")
    return session, info
