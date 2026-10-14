from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Sequence

import time

import requests
import yt_dlp

from .. import runtime_security
from ..errors import DeterministicStageError
from ..sanitize import sanitize_text
from ..sources import SourceConfig
from ..youtube import extract_video_id, validate_video_url

log = logging.getLogger(__name__)


#: The pipeline requires a 1080p master. The selectors below only ever *ask*
#: for one: a source that has no 1080p stream (or a signed-out client that is
#: only shown lower ones) can still satisfy them with something smaller, which
#: is how one task shipped a 360p master without anyone noticing. The file is
#: the only authority, so ``download_video`` probes it and refuses anything
#: below this height instead of carrying it through the pipeline.
MIN_VIDEO_HEIGHT = 1080

FORMAT_CANDIDATES = (
    # The normal case. The cap keeps a 4K stream out (much slower to download
    # and encode); the floor is enforced after the download.
    "bestvideo[height<=1080]+bestaudio",
    # Sources that only offer muxed streams (``bv*`` accepts those).
    "bv*[height<=1080]+ba",
    # Last resort for sources whose formats carry no height at all; the
    # post-download check still rejects anything below 1080p.
    "bestvideo+bestaudio",
)

#: What a download failure usually means, most specific first. A task should
#: stop with the reason a human can act on - cookie? proxy? no 1080p? - instead
#: of a raw yt-dlp message, because only the first kind needs the user.
_FAILURE_RULES: tuple[tuple[str, tuple[str, ...], str, str], ...] = (
    (
        "cookies",
        (
            "sign in",
            "not a bot",
            "login required",
            "login_required",
            "requires login",
            "cookies",
            "unable to log in",
        ),
        "the source asked for a signed-in account (its bot check), so the cookie is missing or expired",
        "refresh the YouTube cookie in Settings, then resume",
    ),
    (
        "unavailable",
        (
            "private video",
            "video unavailable",
            "has been removed",
            "does not exist",
            "members-only",
            "unavailable video",
        ),
        "the source video is not available to this account (private, removed or members-only)",
        "check the URL and the account that is allowed to watch it",
    ),
    (
        "geo",
        ("not available in your country", "geo restricted", "geo-restricted"),
        "the source refuses this region",
        "use a proxy in a supported region (yt-dlp proxy port in Settings)",
    ),
    (
        "network",
        (
            "connection reset",
            "ssl",
            "eof",
            "timed out",
            "timeout",
            "unable to download webpage",
            "temporary failure in name resolution",
            "network is unreachable",
            "proxy",
            "502",
            "503",
            "remote end closed",
            "incomplete read",
        ),
        "the network could not reach the source (the proxy may be off or unstable)",
        "check the proxy (yt-dlp proxy port in Settings), then resume",
    ),
    (
        "format",
        ("requested format is not available",),
        "the source does not offer a stream that can satisfy the 1080p requirement",
        "make sure the video offers 1080p",
    ),
)

DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)


def _bootstrap_bilibili_cookie(cookie_path: Path) -> None:
    response = requests.get(
        "https://www.bilibili.com/",
        headers={"User-Agent": DEFAULT_USER_AGENT, "Referer": "https://www.bilibili.com/"},
        timeout=10,
    )
    response.raise_for_status()
    expires = int(time.time()) + 3600 * 24 * 365
    lines = ["# Netscape HTTP Cookie File", ""]
    cookies = dict(response.cookies)
    cookies.setdefault("SESSDATA", "anonymous_for_webpage_playinfo")
    for name, value in cookies.items():
        lines.append("\t".join([".bilibili.com", "TRUE", "/", "FALSE", str(expires), name, value]))
    runtime_security.atomic_write_private_text(cookie_path, "\n".join(lines) + "\n")


def _proxy_url(proxy_port: str = "") -> str:
    if proxy_port.strip():
        return f"http://127.0.0.1:{proxy_port.strip()}"
    return os.getenv("HTTP_PROXY") or os.getenv("http_proxy") or ""


def _ensure_cookie(source: SourceConfig) -> None:
    cookie_path = source.cookie_path
    if not cookie_path or source.name != "bilibili":
        return
    metadata = runtime_security.private_file_stat(cookie_path)
    if metadata and metadata.st_size > 0:
        return
    _bootstrap_bilibili_cookie(cookie_path)


def _ydl_base(source: SourceConfig, proxy_port: str = "") -> dict[str, Any]:
    opts: dict[str, Any] = {
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "js_runtimes": {"node": {}},
        "http_headers": {"User-Agent": DEFAULT_USER_AGENT},
    }
    cookie_path = source.cookie_path
    if cookie_path:
        metadata = runtime_security.private_file_stat(cookie_path)
        if metadata and metadata.st_size > 0:
            opts["cookiefile"] = str(cookie_path)
    if not source.use_proxy:
        opts["proxy"] = ""
        return opts
    proxy = _proxy_url(proxy_port)
    if proxy:
        opts["proxy"] = proxy
    return opts


def _session_path(
    workfolder: Path,
    info: dict[str, Any],
    task_id: str | None = None,
) -> Path:
    uploader = sanitize_text(str(info.get("uploader") or "unknown"))
    title = sanitize_text(str(info.get("title") or "untitled"))
    video_id = str(info.get("id") or extract_video_id(str(info.get("webpage_url") or "")))
    return workfolder / uploader / f"{title}__{task_id or video_id}"


def _classify_failure(failures: Sequence[str]) -> tuple[str, str, str]:
    """Turn raw download errors into ``(kind, cause, fix)`` for the task log."""
    text = " | ".join(failures).lower()
    for kind, markers, cause, fix in _FAILURE_RULES:
        if any(marker in text for marker in markers):
            return kind, cause, fix
    return (
        "unknown",
        "the download failed for a reason the downloader did not name",
        "see the task log for the raw yt-dlp error",
    )


def _failure_detail(failures: Sequence[str]) -> str:
    """The most recent raw error, trimmed - it is what a human debugs next."""
    for failure in reversed(failures):
        text = failure.strip()
        if text:
            return text[:300]
    return ""


def _download_error(failures: Sequence[str]) -> Exception:
    """An error that names the cause. Network trouble stays retryable, the
    rest cannot be fixed by trying the same download again."""
    kind, cause, fix = _classify_failure(failures)
    message = f"the video download failed: {cause}. {fix}."
    detail = _failure_detail(failures)
    if detail:
        message += f"\nraw error: {detail}"
    if kind in {"cookies", "unavailable", "geo", "format"}:
        return DeterministicStageError(message)
    return RuntimeError(message)


def _offered_heights(info: dict[str, Any]) -> list[int]:
    heights = {
        fmt["height"]
        for fmt in (info.get("formats") or [])
        if isinstance(fmt.get("height"), int) and fmt["height"] > 0
    }
    return sorted(heights)


def _enforce_min_height(
    video_file: Path, offered: Sequence[int], failures: Sequence[str]
) -> None:
    """Refuse a master below 1080p instead of carrying it through the pipeline.

    A source without a 1080p stream - or one that only shows lower qualities
    to a signed-out client - satisfies the selectors with whatever it has.
    One real task reached the upload as a 360p master this way.
    """
    try:
        from .ffmpeg import probe_video_size

        size = probe_video_size(video_file)
    except Exception as exc:  # noqa: BLE001 - a probe failure must not block a good file
        log.warning("could not probe the downloaded master: %s", exc)
        return
    if not size or not size[1]:
        log.warning("could not probe the downloaded master; keeping it unchecked")
        return
    width, height = size
    if height >= MIN_VIDEO_HEIGHT:
        log.info("downloaded master resolution %sx%s", width, height)
        return

    # Reject it and clear the file so a resume re-downloads instead of
    # accepting the bad master through the "already downloaded" shortcut.
    video_file.unlink(missing_ok=True)
    kind, cause, fix = _classify_failure(failures)
    if (
        offered
        and max(offered) < MIN_VIDEO_HEIGHT
        and kind not in {"cookies", "geo", "unavailable"}
    ):
        kind = "format"
        cause = f"the source has no 1080p stream - the best it offers is {max(offered)}p"
        fix = (
            "check the video really offers 1080p (a signed-out client is only shown "
            "lower qualities, so a valid cookie can help), or put a 1080p master at "
            "media/video_source.mp4 and resume"
        )
    message = (
        f"refusing the downloaded master: it is {width}x{height}, below the "
        f"required 1080p. cause: {cause}. {fix}."
    )
    detail = _failure_detail(failures)
    if detail:
        message += f"\nraw error: {detail}"
    if kind == "network":
        raise RuntimeError(message)
    raise DeterministicStageError(message)


def _remove_partial_outputs(video_file: Path) -> None:
    # yt-dlp inserts the format id before the extension when it downloads a
    # video-only stream (``video_source.f399.mp4.part``), so matching on the
    # bare file name missed exactly the big leftovers this is here to clear.
    # Keep the net narrow, though: only download artifacts are ours to delete -
    # a wider pattern once ate a kept backup sitting next to the target.
    for candidate in video_file.parent.glob(f"{video_file.stem}*"):
        if candidate == video_file or not _is_download_artifact(candidate, video_file):
            continue
        if candidate.is_file():
            candidate.unlink(missing_ok=True)


def _is_download_artifact(candidate: Path, video_file: Path) -> bool:
    if candidate.suffix in {".part", ".ytdl"}:
        return True
    # A finished stream waits as ``video_source.f399.mp4`` until it is merged
    # with the audio; the numeric format id is the giveaway. A lookalike backup
    # such as ``video_source.360p.mp4`` is not touched.
    prefix = f"{video_file.stem}.f"
    if candidate.name.startswith(prefix):
        head = candidate.name[len(prefix) :].split(".", 1)[0]
        return head.isdigit()
    return False


def _download_with_format_candidates(
    url: str, video_file: Path, source: SourceConfig, proxy_port: str
) -> list[str]:
    """Run the format candidates in order and report what each one said.

    Returns one note per failed attempt (empty when the first one worked);
    ``download_video`` turns the notes into the error it raises, so the task
    ends with a cause instead of a bare downloader message. No candidate may
    accept a below-1080p master silently - the caller verifies the file.
    """
    failures: list[str] = []
    for format_selector in FORMAT_CANDIDATES:
        download_opts = {
            **_ydl_base(source, proxy_port),
            "format": format_selector,
            "merge_output_format": "mp4",
            "outtmpl": str(video_file),
            "retries": 10,
            "fragment_retries": 10,
        }
        try:
            with yt_dlp.YoutubeDL(download_opts) as ydl:
                ydl.download([url])
            return failures
        except Exception as exc:  # noqa: BLE001 - every failure is a candidate to explain
            failures.append(f"{format_selector}: {str(exc).strip()}")
            log.warning("download format %r failed: %s", format_selector, str(exc)[:200])
            _remove_partial_outputs(video_file)
    return failures


def download_video(
    url: str,
    workfolder: Path,
    source: SourceConfig,
    proxy_port: str = "",
    *,
    task_id: str | None = None,
) -> tuple[Path, dict[str, Any]]:
    validated = validate_video_url(url)
    if validated.source != source.name:
        raise ValueError("The submitted URL does not match the selected video source.")
    canonical_url = validated.url
    video_id = validated.video_id
    _ensure_cookie(source)
    info_opts = _ydl_base(source, proxy_port)
    try:
        with yt_dlp.YoutubeDL(info_opts) as ydl:
            info = ydl.extract_info(canonical_url, download=False)
    except Exception as exc:  # noqa: BLE001 - the reason matters more than the type
        raise _download_error([str(exc)]) from exc

    if str(info.get("id", video_id)) != video_id:
        raise ValueError("The resolved video id does not match the submitted URL.")

    session = _session_path(workfolder, info, task_id)
    media_dir = session / "media"
    metadata_dir = session / "metadata"
    media_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    video_file = media_dir / "video_source.mp4"
    metadata_file = metadata_dir / "ytdlp_info.json"
    metadata_file.write_text(json.dumps(ydl.sanitize_info(info), ensure_ascii=False, indent=2), encoding="utf-8")

    if video_file.exists() and video_file.stat().st_size > 0:
        return session, info

    failures = _download_with_format_candidates(canonical_url, video_file, source, proxy_port)

    if not video_file.exists() or video_file.stat().st_size == 0:
        raise _download_error(failures)

    _enforce_min_height(video_file, _offered_heights(info), failures)

    return session, info
