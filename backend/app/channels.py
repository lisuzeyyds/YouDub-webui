"""Watch YouTube channels and queue their new uploads into the normal pipeline.

The monitoring itself needs no third-party service: YouTube still publishes an
Atom feed per channel, and it carries the video id and a publish timestamp
accurate to the second::

    https://www.youtube.com/feeds/videos.xml?channel_id=UC...
    https://www.youtube.com/feeds/videos.xml?playlist_id=UULF...   (uploads, no Shorts)

No API key, no quota - unlike the YouTube Data API, which is what the packaged
"YouTube -> Bilibili" projects use and which costs 100 units per search.

The loop is deliberately dumb: every ``YOUDUB_CHANNEL_POLL_MINUTES`` (default
30) it reads each watched channel's feed and hands anything published after the
stored cursor to the ordinary task queue with ``auto_publish`` on. Deduplication
is the task table's own: a video that already has a task is simply not created
again, so a crash between "task created" and "cursor moved" costs nothing.
"""

from __future__ import annotations

import logging
import os
import re
import threading
import xml.etree.ElementTree as ET
from datetime import datetime, timezone

import requests

from . import database, worker

log = logging.getLogger(__name__)

_ATOM = "{http://www.w3.org/2005/Atom}"
_YT = "{http://www.youtube.com/xml/schemas/2015}"
_CHANNEL_FEED = "https://www.youtube.com/feeds/videos.xml?channel_id={channel_id}"
#: ``UC`` -> ``UULF`` is the long-form uploads playlist: no Shorts, no livestreams.
_LONGFORM_FEED = "https://www.youtube.com/feeds/videos.xml?playlist_id=UULF{rest}"
WATCH_URL = "https://www.youtube.com/watch?v={video_id}"

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)
_CHANNEL_ID_RE = re.compile(r"UC[0-9A-Za-z_-]{22}")
#: Ordered by how authoritative they are. ``"channelId"`` is *not* reliable: on
#: a channel page it belongs to whatever video or featured channel the page
#: happens to show - @LinusTechTips yields ShortCircuit's id that way.
_PAGE_ID_PATTERNS = (
    re.compile(r'<link rel="canonical" href="https://www\.youtube\.com/channel/(UC[0-9A-Za-z_-]{22})"'),
    re.compile(r'<meta property="og:url" content="https://www\.youtube\.com/channel/(UC[0-9A-Za-z_-]{22})"'),
    re.compile(r'"externalId":"(UC[0-9A-Za-z_-]{22})"'),
    re.compile(r'"browseId":"(UC[0-9A-Za-z_-]{22})"'),
    re.compile(r'"channelId":"(UC[0-9A-Za-z_-]{22})"'),
)
_TITLE_RE = re.compile(r'<meta[^>]+property="og:title"[^>]+content="([^"]*)"')
_HTML_TITLE_RE = re.compile(r"<title>([^<]*)</title>")

#: A channel that was offline for a month should not dump 15 videos on the
#: machine in one poll; the rest wait for the next tick.
MAX_TASKS_PER_POLL = 5

DEFAULT_INTERVAL_MINUTES = 30
MIN_INTERVAL_MINUTES = 5

_stop = threading.Event()
_thread: threading.Thread | None = None
_last_poll: str = ""
_last_error: str = ""


def interval_minutes() -> int:
    raw = (os.getenv("YOUDUB_CHANNEL_POLL_MINUTES") or "").strip()
    try:
        value = int(raw)
    except ValueError:
        return DEFAULT_INTERVAL_MINUTES
    return max(MIN_INTERVAL_MINUTES, min(value, 24 * 60))


def default_proxy() -> str | None:
    """Same rule the download stage uses: the configured port, then the environment."""
    try:
        port = str(database.get_ytdlp_settings().get("proxy_port") or "").strip()
    except Exception:  # noqa: BLE001 - settings are optional here
        port = ""
    if port:
        return f"http://127.0.0.1:{port}"
    return os.getenv("HTTP_PROXY") or os.getenv("HTTPS_PROXY") or None


def _get(url: str, proxy: str | None, timeout: int = 25) -> requests.Response:
    proxies = {"http": proxy, "https": proxy} if proxy else None
    return requests.get(
        url,
        timeout=timeout,
        proxies=proxies,
        headers={"User-Agent": USER_AGENT, "Accept-Language": "en-US,en;q=0.9"},
    )


def channel_id_from(text: str) -> str:
    """Pick the ``UC...`` id out of a channel URL, a handle or a bare id."""
    value = (text or "").strip()
    if not value:
        raise ValueError("give a channel URL, @handle or UC... id")
    match = _CHANNEL_ID_RE.search(value)
    if match and ("/channel/" in value or value.startswith("UC")):
        return match.group(0)
    raise ValueError(f"no channel id in {text!r} - paste the channel URL instead")


def channel_id_from_page(html: str) -> str:
    """The channel a page *is*, not a channel it merely mentions."""
    for pattern in _PAGE_ID_PATTERNS:
        match = pattern.search(html or "")
        if match:
            return match.group(1)
    return ""


def page_title(html: str) -> str:
    match = _TITLE_RE.search(html or "")
    if match and match.group(1).strip():
        return match.group(1).strip()
    match = _HTML_TITLE_RE.search(html or "")
    return re.sub(r"\s*-\s*YouTube\s*$", "", match.group(1).strip()) if match else ""


def resolve_channel(text: str, proxy: str | None = None) -> dict:
    """Turn a channel URL or @handle into ``{channel_id, title, handle}``."""
    value = (text or "").strip()
    try:
        channel_id = channel_id_from(value)
        handle = ""
    except ValueError:
        url = value if value.startswith("http") else f"https://www.youtube.com/@{value.lstrip('@')}"
        try:
            response = _get(url, proxy)
        except Exception as exc:  # noqa: BLE001 - reported to the caller as-is
            raise ValueError(f"could not open {url}: {exc}") from exc
        if response.status_code != 200:
            raise ValueError(f"channel page returned HTTP {response.status_code}: {url}")
        channel_id = channel_id_from_page(response.text)
        if not channel_id:
            raise ValueError(f"could not find a channel id on {url}")
        handle = value if value.startswith("@") else ""
        return {"channel_id": channel_id, "title": page_title(response.text) or channel_id, "handle": handle}

    title = ""
    try:
        page = _get(f"https://www.youtube.com/channel/{channel_id}", proxy).text
        title = page_title(page)
    except Exception:  # noqa: BLE001 - the name is cosmetic
        pass
    return {"channel_id": channel_id, "title": title or channel_id, "handle": handle}


def _parse_published(value: str) -> datetime:
    try:
        stamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return datetime.min.replace(tzinfo=timezone.utc)
    return stamp if stamp.tzinfo else stamp.replace(tzinfo=timezone.utc)


def _entries(xml: bytes) -> list[dict]:
    root = ET.fromstring(xml)
    items: list[dict] = []
    for entry in root.findall(f"{_ATOM}entry"):
        video_id = (entry.findtext(f"{_YT}videoId") or "").strip()
        published = (entry.findtext(f"{_ATOM}published") or "").strip()
        if not video_id or not published:
            continue
        items.append(
            {
                "video_id": video_id,
                "title": (entry.findtext(f"{_ATOM}title") or "").strip(),
                "published": published,
                "author": (entry.findtext(f"{_ATOM}author/{_ATOM}name") or "").strip(),
            }
        )
    items.sort(key=lambda item: _parse_published(item["published"]), reverse=True)
    return items


def fetch_entries(channel_id: str, proxy: str | None = None, *, long_form_only: bool = True) -> list[dict]:
    """Newest 15 uploads for ``channel_id``, newest first."""
    if not _CHANNEL_ID_RE.fullmatch(channel_id):
        raise ValueError(f"not a channel id: {channel_id!r}")
    urls = []
    if long_form_only:
        urls.append(_LONGFORM_FEED.format(rest=channel_id[2:]))
    urls.append(_CHANNEL_FEED.format(channel_id=channel_id))
    for url in urls:
        response = _get(url, proxy)
        if response.status_code != 200 or not response.content:
            log.debug("feed %s -> HTTP %s", url, response.status_code)
            continue
        try:
            entries = _entries(response.content)
        except ET.ParseError as exc:
            log.debug("feed %s is not valid XML: %s", url, exc)
            continue
        if entries:
            return entries
    return []


def _queue(channel: dict, entry: dict) -> str | None:
    """Create and enqueue a task for one upload; ``None`` when it already exists."""
    url = WATCH_URL.format(video_id=entry["video_id"])
    try:
        task_id, created = database.create_or_get_video_task(
            url,
            entry["video_id"],
            auto_publish=True,
        )
    except Exception as exc:  # noqa: BLE001 - one bad row must not stop the poll
        log.warning("could not queue %s (%s): %s", entry["video_id"], channel["channel_id"], exc)
        return None
    if not created:
        log.info("already have a task for %s (%s)", entry["video_id"], entry["title"][:60])
        return None
    worker.enqueue(task_id)
    log.info(
        "new upload from %s: %s -> task %s",
        channel["title"] or channel["channel_id"],
        entry["title"][:70],
        task_id,
    )
    return task_id


def poll_channel(channel: dict, *, proxy: str | None = None) -> dict:
    """Check one channel and queue what is new. Never raises for network errors."""
    result = {"channel_id": channel["channel_id"], "created": [], "checked": 0, "error": ""}
    try:
        entries = fetch_entries(channel["channel_id"], proxy)
    except Exception as exc:  # noqa: BLE001 - a network hiccup is not a failure
        result["error"] = f"{type(exc).__name__}: {exc}"
        return result
    result["checked"] = len(entries)
    if not entries:
        return result

    cursor = _parse_published(channel["last_seen"]) if channel.get("last_seen") else None
    newest = entries[0]["published"]
    if cursor is None:
        # First sight of this channel: remember where "now" is and queue nothing.
        database.update_channel_seen(channel["channel_id"], newest)
        result["baseline"] = newest
        log.info("%s: watching from %s (nothing backfilled)", channel["channel_id"], newest)
        return result

    fresh = [entry for entry in entries if _parse_published(entry["published"]) > cursor]
    for entry in reversed(fresh):  # oldest first, so the queue runs in order
        if len(result["created"]) >= MAX_TASKS_PER_POLL:
            log.info("%s: %s more uploads left for the next poll", channel["channel_id"], len(fresh))
            break
        task_id = _queue(channel, entry)
        if task_id:
            result["created"].append(task_id)
    # Never move the cursor backwards: a cached feed can briefly serve an older
    # "newest" entry, and that would make the next poll re-queue videos.
    if _parse_published(newest) > cursor:
        database.update_channel_seen(channel["channel_id"], newest)
    return result


def poll_once(*, proxy: str | None = None, channel_ids: list[str] | None = None) -> dict:
    """One pass over every enabled channel (or just ``channel_ids``)."""
    global _last_poll, _last_error
    proxy = proxy if proxy is not None else default_proxy()
    channels = database.list_channels()
    if channel_ids:
        wanted = set(channel_ids)
        channels = [channel for channel in channels if channel["channel_id"] in wanted]
    summary = {"channels": 0, "queued": 0, "errors": []}
    for channel in channels:
        if not channel.get("enabled", True):
            continue
        summary["channels"] += 1
        outcome = poll_channel(channel, proxy=proxy)
        summary["queued"] += len(outcome["created"])
        if outcome.get("error"):
            summary["errors"].append(f"{channel['title'] or channel['channel_id']}: {outcome['error']}")
    _last_poll = database.now_iso()
    _last_error = "; ".join(summary["errors"][:3])
    if summary["queued"]:
        log.info("channel poll queued %s task(s)", summary["queued"])
    return summary


def status() -> dict:
    return {
        "running": bool(_thread and _thread.is_alive()),
        "interval_minutes": interval_minutes(),
        "last_poll": _last_poll,
        "last_error": _last_error,
    }


def _loop() -> None:
    minutes = interval_minutes()
    log.info("channel watcher started (every %s min)", minutes)
    # Let the backend finish starting before the first request goes out.
    if _stop.wait(20):
        return
    while not _stop.is_set():
        try:
            poll_once()
        except Exception:  # noqa: BLE001 - the loop must survive anything
            log.exception("channel poll failed")
        if _stop.wait(minutes * 60):
            return


def start() -> None:
    """Start the background loop once per process."""
    global _thread
    if _thread and _thread.is_alive():
        return
    _stop.clear()
    _thread = threading.Thread(target=_loop, name="channel-watcher", daemon=True)
    _thread.start()


def stop() -> None:
    _stop.set()
    global _thread
    if _thread:
        _thread.join(timeout=5)
    _thread = None
