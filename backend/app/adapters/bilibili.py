"""Publish the finished video to Bilibili by driving the real upload page.

Bilibili exposes no upload API to ordinary accounts, so unattended publishing
means scripting the web page. The browser itself is owned by a small Node
script (``tools/bili-publish/publish.js``); this module owns the decisions and
the failure semantics:

* which asset to send (final video + a frame lifted from it as the cover),
* which copy to use (the title/description/tags the translate stage wrote),
* which partition and collection to pick, matched against the page's own
  option list rather than trusted verbatim,
* how a failure is reported - a missing login is deterministic and must not be
  retried, because every retry would upload the video again.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .. import config
from ..errors import DeterministicStageError
from .ffmpeg import _run_ffmpeg

log = logging.getLogger(__name__)

_NO_WINDOW = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
RESULT_PREFIX = "RESULT "
# Uploading a one-hour video over a home connection takes minutes, and the page
# keeps working while it does, so the ceiling is generous on purpose.
PUBLISH_TIMEOUT_SECONDS = 90 * 60
COVER_SEEK_SECONDS = 5.0
COVER_FALLBACK_SEEK_SECONDS = 1.0
BILIBILI_VIDEO_URL = "https://www.bilibili.com/video/{bvid}"

# The upload page insists on a creation declaration; this is the value the user
# settled on and it applies to every release.
DEFAULT_DECLARE = "内容无需标注"


@dataclass
class PublishOutcome:
    """What the browser reported back."""

    ok: bool
    bvid: str = ""
    zone: str = ""
    collection: str = ""
    stage: str = ""
    error: str = ""

    @property
    def url(self) -> str:
        return BILIBILI_VIDEO_URL.format(bvid=self.bvid) if self.bvid else ""


def default_profile_dir() -> Path:
    """The persistent Chrome profile that carries the Bilibili login."""
    return Path.home() / ".agent-browser" / "profiles" / "bili"


def publish_artifact_path(session: Path) -> Path:
    return session / "metadata" / "bilibili_publish.json"


def load_publish_result(session: Path) -> dict[str, Any] | None:
    path = publish_artifact_path(session)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        log.warning("ignoring unreadable publish result %s: %s", path, exc)
        return None
    return data if isinstance(data, dict) else None


def find_chrome(explicit: str = "") -> str:
    """Locate a Chrome/Chromium binary, newest managed build first."""
    if explicit:
        candidate = Path(explicit).expanduser()
        if candidate.exists():
            return str(candidate)
        log.warning("configured chrome path does not exist: %s", explicit)

    root = Path.home() / ".agent-browser" / "browsers"
    if root.is_dir():
        found = sorted(root.glob("chrome-*/chrome-win64/chrome.exe"))
        found += sorted(root.glob("chrome-*/chrome.exe"))
        found += sorted(root.glob("*/chrome-linux/chrome"))
        for candidate in reversed(found):
            if candidate.exists():
                return str(candidate)

    for base in (
        os.environ.get("PROGRAMFILES", ""),
        os.environ.get("PROGRAMFILES(X86)", ""),
        os.environ.get("LOCALAPPDATA", ""),
    ):
        if not base:
            continue
        candidate = Path(base) / "Google" / "Chrome" / "Application" / "chrome.exe"
        if candidate.exists():
            return str(candidate)
    found = shutil.which("chrome") or shutil.which("google-chrome") or shutil.which("chromium")
    return found or ""


def find_node(explicit: str = "") -> str:
    """Locate the Node runtime used to drive the browser."""
    if explicit:
        candidate = Path(explicit).expanduser()
        if candidate.exists():
            return str(candidate)
        log.warning("configured node path does not exist: %s", explicit)

    versions = Path.home() / ".workbuddy" / "binaries" / "node" / "versions"
    if versions.is_dir():
        for directory in sorted(versions.iterdir(), reverse=True):
            candidate = directory / ("node.exe" if os.name == "nt" else "bin/node")
            if candidate.exists():
                return str(candidate)
    found = shutil.which("node")
    return found or ""


def find_script(explicit: str = "") -> str:
    """Locate the browser-driving script shipped with the repository."""
    if explicit:
        candidate = Path(explicit).expanduser()
        if candidate.exists():
            return str(candidate)
        log.warning("configured publish script does not exist: %s", explicit)

    repo_root = Path(__file__).resolve().parents[3]
    candidate = repo_root / "tools" / "bili-publish" / "publish.js"
    return str(candidate) if candidate.exists() else ""


def build_cover(video_file: Path, out_file: Path) -> Path | None:
    """Lift a frame out of the finished video to use as the cover.

    The cover is derived rather than uploaded by the user, so a failure here is
    not fatal: an upload without a cover is still a valid submission. ffmpeg is
    asked for JPEG directly (``-q:v``) because that is what the page accepts.
    """
    out_file.parent.mkdir(parents=True, exist_ok=True)
    for seek in (COVER_SEEK_SECONDS, COVER_FALLBACK_SEEK_SECONDS):
        command = [
            config.ffmpeg_binary(),
            "-y",
            "-ss",
            f"{seek:.2f}",
            "-i",
            str(video_file),
            "-vframes",
            "1",
            "-q:v",
            "2",
            "-pix_fmt",
            "yuvj420p",
            str(out_file),
        ]
        try:
            _run_ffmpeg(command)
        except Exception as exc:  # noqa: BLE001 - the cover is optional
            log.warning("cover extraction at %.1fs failed: %s", seek, exc)
            continue
        if out_file.exists() and out_file.stat().st_size > 0:
            return out_file
    return None


def _run_browser(command: list[str], *, cwd: str) -> subprocess.CompletedProcess:
    """Run the browser script with piped stdio and no interactive stdin.

    Kept as its own function so tests can replace it without patching
    ``subprocess`` globally (the sandbox's own file-deletion guard runs through
    ``subprocess.run`` and must keep working).
    """
    return subprocess.run(
        command,
        cwd=cwd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdin=subprocess.DEVNULL,
        timeout=PUBLISH_TIMEOUT_SECONDS,
        creationflags=_NO_WINDOW,
    )


def _parse_result(stdout: str) -> dict[str, Any] | None:
    for line in reversed(stdout.splitlines()):
        stripped = line.strip()
        if stripped.startswith(RESULT_PREFIX):
            try:
                parsed = json.loads(stripped[len(RESULT_PREFIX) :])
            except json.JSONDecodeError:
                continue
            return parsed if isinstance(parsed, dict) else None
    return None


def _tail(text: str, limit: int = 400) -> str:
    cleaned = " ".join((text or "").split())
    return cleaned[-limit:]


def publish_video(
    *,
    video_file: Path,
    session: Path,
    title: str,
    description: str,
    tags: list[str],
    zone: str = "",
    collection: str = "",
    settings: dict[str, str] | None = None,
    dry_run: bool = False,
) -> PublishOutcome:
    """Upload ``video_file`` to Bilibili and return what happened.

    Raises :class:`DeterministicStageError` when retrying cannot possibly help
    (no browser/node, expired login), so the pipeline stops instead of spending
    another full upload on the same wall.
    """
    settings = settings or {}
    chrome = find_chrome(settings.get("chrome_path", ""))
    node = find_node(settings.get("node_path", ""))
    script = find_script(settings.get("script_path", ""))
    profile = (settings.get("profile_dir") or "").strip() or str(default_profile_dir())
    declare = (settings.get("declare") or "").strip() or DEFAULT_DECLARE

    missing = [
        name
        for name, value in (("chrome", chrome), ("node", node), ("publish script", script))
        if not value
    ]
    if missing:
        raise DeterministicStageError(
            "Bilibili publishing is not available: could not locate "
            + ", ".join(missing)
            + ". Set BILIBILI_CHROME_PATH / BILIBILI_NODE_PATH / BILIBILI_SCRIPT_PATH "
            "to the right paths and run the stage again."
        )
    if not video_file.exists():
        raise DeterministicStageError(f"Final video is missing: {video_file}")

    Path(profile).mkdir(parents=True, exist_ok=True)

    cover_file = build_cover(video_file, session / "tmp" / "bilibili_cover.jpg")
    if cover_file is None:
        log.warning("publishing without a cover (frame extraction failed)")

    shot_dir = (session / "tmp" / "bilibili_shots").resolve()
    shot_dir.mkdir(parents=True, exist_ok=True)
    # The browser script runs with the script directory as its working
    # directory, so every path handed to it has to be absolute - a relative
    # one would be resolved against tools/bili-publish/ and the payload file
    # itself would not even be found (ENOENT on the first live run).
    payload_file = (session / "tmp" / "bilibili_payload.json").resolve()
    payload_file.write_text(
        json.dumps(
            {
                "video": str(video_file.resolve()),
                "cover": str(cover_file.resolve()) if cover_file else "",
                "title": title,
                "description": description,
                "tags": list(tags),
                "zone": zone,
                "collection": collection,
                "declare": declare,
                "chrome": chrome,
                "node": node,
                "profile": str(Path(profile).resolve()),
                "headless": not dry_run,
                "dryRun": dry_run,
                "shotDir": str(shot_dir),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    command = [node, script, str(payload_file)]
    log.info("publishing to Bilibili: %s (zone=%r, collection=%r)", video_file.name, zone, collection)
    try:
        completed = _run_browser(command, cwd=str(Path(script).parent))
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"Bilibili upload timed out after {PUBLISH_TIMEOUT_SECONDS // 60} minutes"
        ) from exc
    except OSError as exc:
        raise DeterministicStageError(f"Could not start the Bilibili publisher: {exc}") from exc

    parsed = _parse_result(completed.stdout or "")
    if parsed is None:
        raise RuntimeError(
            "Bilibili publisher produced no result (exit code "
            f"{completed.returncode}): {_tail(completed.stderr or completed.stdout)}"
        )

    outcome = PublishOutcome(
        ok=bool(parsed.get("ok")),
        bvid=str(parsed.get("bvid") or ""),
        zone=str(parsed.get("zone") or ""),
        collection=str(parsed.get("collection") or ""),
        stage=str(parsed.get("stage") or ""),
        error=str(parsed.get("error") or ""),
    )
    if outcome.ok:
        return outcome

    detail = outcome.error or "unknown error"
    if outcome.stage == "login":
        raise DeterministicStageError(
            "Bilibili sign-in has expired. Open Chrome with the profile "
            f"{profile} and sign in again, then resume this task. (detail: {detail})"
        )
    raise RuntimeError(f"Bilibili publishing failed at {outcome.stage}: {detail}")
