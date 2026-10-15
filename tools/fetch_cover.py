"""Fetch the original YouTube cover (thumbnail) for a video.

Standalone tool for now - it downloads the highest-resolution thumbnail the
source actually offers (``maxresdefault`` is 1280x720 for most videos) instead
of lifting a frame out of the finished video. Run it, look at the result, and
the publish stage can start preferring it afterwards.

Usage (from the repository root, with the project virtualenv):

    ./.venv/Scripts/python.exe tools/fetch_cover.py https://youtu.be/dQw4w9WgXcQ
    ./.venv/Scripts/python.exe tools/fetch_cover.py dQw4w9WgXcQ -o cover.jpg
    ./.venv/Scripts/python.exe tools/fetch_cover.py --session workfolder/<uploader>/<title>__<task_id>

With ``--session`` the cover of the session's own video is fetched and saved to
``<session>/metadata/cover.jpg``; the thumbnail URLs recorded by the download
stage are tried first, so no extra request is needed to discover them.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import requests

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.app.config import ffmpeg_binary, ffprobe_binary  # noqa: E402

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)

#: A cover below this is a placeholder, not artwork (YouTube answers a missing
#: ``maxresdefault`` with 404, and with a 120x90 grey tile on some videos).
MIN_WIDTH = 640
MIN_HEIGHT = 360

#: Tried in order after the recorded URLs; the first one that exists wins.
#: ``hqdefault`` (480x360) is the floor - every video has it.
PATTERN_NAMES = ("maxresdefault", "sddefault", "hqdefault", "mqdefault", "default")
PATTERN_BASE = "https://i.ytimg.com/vi/{video_id}/{name}.jpg"

_VIDEO_ID_RE = re.compile(r"[A-Za-z0-9_-]{11}")


def video_id_from(value: str) -> str:
    """Accept a watch/short URL or a bare id and return the 11-char id."""
    text = (value or "").strip()
    if not text:
        raise ValueError("no URL or video id given")
    match = re.search(r"(?:v=|youtu\.be/|/shorts/|/embed/|/live/)([A-Za-z0-9_-]{11})", text)
    if match:
        return match.group(1)
    if _VIDEO_ID_RE.fullmatch(text):
        return text
    raise ValueError(f"could not find a YouTube video id in {value!r}")


def video_id_from_session(session: Path) -> str:
    info_file = session / "metadata" / "ytdlp_info.json"
    if not info_file.exists():
        # Sessions are named <title>__<task id>, and the task id is the video id
        # for remote downloads.
        name = session.name.rsplit("__", 1)[-1]
        if _VIDEO_ID_RE.fullmatch(name):
            return name
        raise ValueError(f"{session} has no metadata/ytdlp_info.json to read the id from")
    info = json.loads(info_file.read_text(encoding="utf-8"))
    video_id = str(info.get("id") or "")
    if not _VIDEO_ID_RE.fullmatch(video_id):
        raise ValueError(f"metadata/ytdlp_info.json has no usable id ({video_id!r})")
    return video_id


def recorded_thumbnail_urls(session: Path) -> list[str]:
    """The thumbnail URLs yt-dlp already discovered, best first."""
    info_file = session / "metadata" / "ytdlp_info.json"
    if not info_file.exists():
        return []
    try:
        info = json.loads(info_file.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    entries: list[tuple[int, str]] = []
    for entry in info.get("thumbnails") or []:
        url = str((entry or {}).get("url") or "")
        if not url:
            continue
        area = int((entry.get("width") or 0)) * int((entry.get("height") or 0))
        entries.append((area, url))
    entries.sort(key=lambda item: item[0], reverse=True)
    ordered: list[str] = []
    for _, url in entries:
        if url not in ordered:
            ordered.append(url)
    # The upload page takes jpg/png; a webp would have to be converted, so keep
    # it as the last resort rather than the first choice.
    direct = [url for url in ordered if url.split("?")[0].lower().endswith((".jpg", ".jpeg", ".png"))]
    return direct + [url for url in ordered if url not in direct]


def pattern_urls(video_id: str) -> list[str]:
    return [PATTERN_BASE.format(video_id=video_id, name=name) for name in PATTERN_NAMES]


def resolve_proxy(port: str) -> str:
    """Same rule the download stage uses: explicit port, then the environment."""
    port = (port or "").strip()
    if port:
        return f"http://127.0.0.1:{port}"
    return os.getenv("HTTP_PROXY") or os.getenv("http_proxy") or ""


def _configured_proxy_port() -> str:
    try:
        from backend.app import database

        return str(database.get_ytdlp_settings().get("proxy_port") or "")
    except Exception:  # noqa: BLE001 - the tool must run without the app database
        return ""


def _image_kind(payload: bytes) -> str:
    if payload[:3] == b"\xff\xd8\xff":
        return "jpeg"
    if payload[:8] == b"\x89PNG\r\n\x1a\n":
        return "png"
    if payload[:4] == b"RIFF" and payload[8:12] == b"WEBP":
        return "webp"
    return ""


def image_size(path: Path) -> tuple[int, int] | None:
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
            "csv=p=0:s=x",
            str(path),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    match = re.match(r"(\d+)x(\d+)", result.stdout.strip())
    return (int(match.group(1)), int(match.group(2))) if match else None


def to_jpeg(source: Path, target: Path) -> bool:
    target.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [
            ffmpeg_binary(),
            "-y",
            "-i",
            str(source),
            "-q:v",
            "2",
            "-pix_fmt",
            "yuvj420p",
            str(target),
        ],
        capture_output=True,
        text=True,
    )
    return result.returncode == 0 and target.exists() and target.stat().st_size > 0


def fetch_cover(
    video_id: str,
    output: Path,
    *,
    proxy: str = "",
    session: Path | None = None,
    log=print,
) -> dict | None:
    """Download the best thumbnail for ``video_id`` to ``output`` (a JPEG).

    Returns a summary dict, or ``None`` when nothing usable was found.
    """
    candidates: list[str] = []
    if session is not None:
        candidates.extend(recorded_thumbnail_urls(session))
    candidates.extend(pattern_urls(video_id))
    seen: set[str] = set()
    ordered = [url for url in candidates if not (url in seen or seen.add(url))]

    output.parent.mkdir(parents=True, exist_ok=True)
    scratch = output.with_suffix(".download")
    best: tuple[int, int, Path, str] | None = None
    with requests.Session() as http:
        http.headers.update({"User-Agent": USER_AGENT, "Referer": "https://www.youtube.com/"})
        for url in ordered:
            try:
                response = http.get(url, timeout=30, proxies={"http": proxy, "https": proxy} if proxy else None)
            except requests.RequestException as exc:
                log(f"  {url} -> {type(exc).__name__}: {exc}")
                continue
            if response.status_code != 200 or not _image_kind(response.content):
                log(f"  {url} -> HTTP {response.status_code}, {len(response.content)} bytes (not an image)")
                continue
            scratch.write_bytes(response.content)
            size = image_size(scratch)
            if not size or not size[0] or not size[1]:
                log(f"  {url} -> downloaded {len(response.content)} bytes but could not read its size")
                continue
            width, height = size
            log(f"  {url} -> {width}x{height}, {len(response.content) / 1024:.1f} KB")
            if best is None or width * height > best[0] * best[1]:
                best = (width, height, scratch, url)
            if width >= MIN_WIDTH and height >= MIN_HEIGHT:
                break

    if best is None:
        scratch.unlink(missing_ok=True)
        return None

    width, height, downloaded, url = best
    if _image_kind(downloaded.read_bytes()) == "jpeg":
        downloaded.replace(output)
    elif not to_jpeg(downloaded, output):
        log("converting the thumbnail to JPEG failed")
        downloaded.unlink(missing_ok=True)
        return None
    else:
        downloaded.unlink(missing_ok=True)

    if width < MIN_WIDTH or height < MIN_HEIGHT:
        log(
            f"warning: the largest thumbnail this video offers is {width}x{height}; "
            "the upload page prefers at least 1146x717"
        )
    return {
        "video_id": video_id,
        "url": url,
        "path": output,
        "width": width,
        "height": height,
        "bytes": output.stat().st_size,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download the original YouTube cover of a video.")
    parser.add_argument("url", nargs="?", help="YouTube URL or bare video id")
    parser.add_argument("--session", type=Path, help="session folder (uses its recorded video id)")
    parser.add_argument("-o", "--output", type=Path, help="where to write the JPEG")
    parser.add_argument("--proxy-port", default=None, help="local proxy port (default: the app setting, then the environment)")
    args = parser.parse_args(argv)

    if not args.url and not args.session:
        parser.error("give a URL/video id or --session")

    session = args.session
    if session and not session.is_dir():
        parser.error(f"--session {session} is not a directory")

    try:
        video_id = video_id_from(args.url) if args.url else video_id_from_session(session)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    output = args.output or (session / "metadata" / "cover.jpg" if session else Path("cover.jpg"))
    port = _configured_proxy_port() if args.proxy_port is None else args.proxy_port
    proxy = resolve_proxy(port)

    print(f"video id : {video_id}")
    print(f"proxy    : {proxy or '(direct)'}")
    if session:
        print(f"session  : {session}")
    print("candidates:")
    result = fetch_cover(video_id, output, proxy=proxy, session=session)
    if result is None:
        print("no usable thumbnail found (network down, or the video has none)")
        return 1
    print(
        f"saved    : {result['path']}  {result['width']}x{result['height']}  "
        f"{result['bytes'] / 1024:.1f} KB"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
