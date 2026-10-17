"""Choose the cover that goes up with the video.

Publishing used to always lift a frame out of the finished video. The original
YouTube thumbnail is better artwork - it is drawn to be clicked - but its text
is in English, which looks wrong on a Chinese upload. So this module walks three
steps and stops at the first one that works:

1. the thumbnail has no English (local OCR decides) -> upload it unchanged;
2. it has English -> hand the whole picture to the image API
   (``tools/cover_localize.py``), which translates the title and redraws the
   cover; the result is kept only if the OCR can read Chinese on it and the
   English it replaced is gone;
3. no thumbnail, or the redraw failed / did not pass -> ``None``, and the caller
   falls back to a frame from the video.

The tools live in ``tools/`` because they are also run by hand; they are
imported here rather than duplicated.
"""

from __future__ import annotations

import logging
import shutil
import sys
from pathlib import Path

# ``tools`` is a namespace package at the repository root, and the backend may
# be started from anywhere, so the root has to be importable before the tools
# are pulled in.
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools import cover_localize, cover_ocr, fetch_cover  # noqa: E402

from .. import database  # noqa: E402

log = logging.getLogger(__name__)

#: Where the untouched thumbnail and the chosen cover live inside a session.
ORIGINAL_NAME = "cover.jpg"
CHOSEN_NAME = "cover_upload.jpg"


def _proxy() -> str | None:
    """Whatever the download stage uses: the configured port, then the environment."""
    try:
        port = str(database.get_ytdlp_settings().get("proxy_port") or "")
    except Exception:  # noqa: BLE001 - settings may be unavailable in tests
        port = ""
    return fetch_cover.resolve_proxy(port) or None


def _copy(source: Path, target: Path) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    if source.resolve() != target.resolve():
        shutil.copyfile(source, target)
    return target


def prepare_cover(*, session: Path, video_id: str = "", proxy: str | None = None) -> Path | None:
    """Return the cover to upload, or ``None`` when a frame has to be used.

    A returned path is always a JPEG the upload page accepts.
    """
    metadata = session / "metadata"
    original = metadata / ORIGINAL_NAME
    chosen = metadata / CHOSEN_NAME

    if not original.exists():
        try:
            ident = video_id or fetch_cover.video_id_from_session(session)
        except ValueError as exc:
            log.info("no YouTube id for this task (%s) - using a frame from the video", exc)
            return None
        if proxy is None:
            proxy = _proxy()
        try:
            fetched = fetch_cover.fetch_cover(ident, original, proxy=proxy or "", session=session, log=log.info)
        except Exception as exc:  # noqa: BLE001 - any failure just means "use a frame"
            log.warning("could not download the original cover: %s - using a frame", exc)
            return None
        if fetched is None:
            log.warning("no usable thumbnail for %s - using a frame", ident)
            return None
        log.info("original cover: %sx%s", fetched["width"], fetched["height"])

    try:
        blocks = cover_ocr.english_blocks(original)
    except Exception as exc:  # noqa: BLE001 - OCR is an optional dependency
        log.warning("cover OCR failed (%s) - uploading the original cover unchanged", exc)
        return _copy(original, chosen)

    if not blocks:
        log.info("cover has no English - uploading it unchanged")
        return _copy(original, chosen)

    log.info("cover has English: %s", " | ".join(block.text for block in blocks))
    try:
        result = cover_localize.localize_cover(original, chosen, proxy=proxy, log=log.info)
    except Exception as exc:  # noqa: BLE001 - fall back to the known-good path
        log.warning("cover localisation failed: %s - using a frame instead", exc)
        return None

    if result is None:  # OCR saw English a moment ago but not now; keep the original
        return _copy(original, chosen)
    if not result.usable:
        log.warning(
            "localised cover did not pass the check (chinese=%s, leftover=%s) - using a frame instead",
            result.cjk_after,
            result.leftover or "-",
        )
        return None
    log.info("cover localised: %s", result.output)
    return result.output
