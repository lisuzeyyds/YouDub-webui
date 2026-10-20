"""Channel watching: resolving a channel, reading its feed, queueing new uploads.

No network here - the feed bytes are fixtures and the fetch is replaced.
"""

from __future__ import annotations

import pytest

from backend.app import channels, database

ATOM = b"""<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom" xmlns:yt="http://www.youtube.com/xml/schemas/2015">
  <title>Channel</title>
  <entry>
    <yt:videoId>AAAAAAAAAAA</yt:videoId>
    <title>Older upload</title>
    <published>2026-10-19T10:00:00+00:00</published>
    <author><name>Chan</name></author>
  </entry>
  <entry>
    <yt:videoId>BBBBBBBBBBB</yt:videoId>
    <title>Newer upload</title>
    <published>2026-10-20T10:00:00+00:00</published>
    <author><name>Chan</name></author>
  </entry>
</feed>
"""

CHANNEL = {
    "channel_id": "UCXuqSBlHAE6Xw-yeJA0Tunw",
    "title": "Linus Tech Tips",
    "handle": "",
    "last_seen": "",
    "enabled": True,
    "created_at": "",
}


@pytest.fixture
def db(monkeypatch, tmp_path):
    monkeypatch.setattr(database, "DB_PATH", tmp_path / "channels.sqlite")
    database.init_db()
    return database


def _entries():
    return channels._entries(ATOM)


def test_channel_id_accepts_a_channel_url_and_a_bare_id():
    assert channels.channel_id_from("https://www.youtube.com/channel/UCXuqSBlHAE6Xw-yeJA0Tunw") == "UCXuqSBlHAE6Xw-yeJA0Tunw"
    assert channels.channel_id_from("UCXuqSBlHAE6Xw-yeJA0Tunw") == "UCXuqSBlHAE6Xw-yeJA0Tunw"


def test_channel_id_rejects_a_handle_so_it_gets_resolved():
    with pytest.raises(ValueError):
        channels.channel_id_from("@LinusTechTips")


def test_page_id_prefers_canonical_over_the_first_channel_id():
    """A channel page mentions other channels; "channelId" is not the page's own.

    Verified against the real @LinusTechTips page: its first "channelId" is
    ShortCircuit, while canonical/og:url/externalId all name Linus Tech Tips.
    """
    html = (
        '<html><head>'
        '<link rel="canonical" href="https://www.youtube.com/channel/UCXuqSBlHAE6Xw-yeJA0Tunw">'
        '<meta property="og:url" content="https://www.youtube.com/channel/UCXuqSBlHAE6Xw-yeJA0Tunw">'
        '<meta property="og:title" content="Linus Tech Tips">'
        '</head><body>'
        '{"channelId":"UCdBK94H6oZT2Q7l0-b0xmMg"}'
        '</body></html>'
    )
    assert channels.channel_id_from_page(html) == "UCXuqSBlHAE6Xw-yeJA0Tunw"
    assert channels.page_title(html) == "Linus Tech Tips"


def test_page_id_falls_back_to_external_id_when_no_canonical():
    html = '<script>{"metadata":{"externalId":"UCdBK94H6oZT2Q7l0-b0xmMg"}}</script>'
    assert channels.channel_id_from_page(html) == "UCdBK94H6oZT2Q7l0-b0xmMg"


def test_page_title_drops_the_youtube_suffix():
    assert channels.page_title("<title>ShortCircuit - YouTube</title>") == "ShortCircuit"


def test_entries_are_newest_first():
    items = _entries()
    assert [item["video_id"] for item in items] == ["BBBBBBBBBBB", "AAAAAAAAAAA"]
    assert items[0]["title"] == "Newer upload"


def test_a_new_channel_only_records_a_baseline(db, monkeypatch):
    """只追新: adding a channel must not dump its back catalogue on the machine."""
    monkeypatch.setattr(channels, "fetch_entries", lambda *_a, **_k: _entries())
    monkeypatch.setattr(channels.worker, "enqueue", lambda _task: pytest.fail("nothing should be queued"))

    outcome = channels.poll_channel({**CHANNEL, "last_seen": ""})

    assert outcome["created"] == []
    assert database.get_channel(CHANNEL["channel_id"]) is None  # poll_channel never writes the row itself
    assert outcome["baseline"] == "2026-10-20T10:00:00+00:00"


def test_a_new_upload_is_queued_with_auto_publish(db, monkeypatch):
    database.upsert_channel(CHANNEL["channel_id"], title=CHANNEL["title"], last_seen="2026-10-19T10:00:00+00:00")
    monkeypatch.setattr(channels, "fetch_entries", lambda *_a, **_k: _entries())
    queued: list[str] = []
    monkeypatch.setattr(channels.worker, "enqueue", queued.append)

    outcome = channels.poll_channel(database.get_channel(CHANNEL["channel_id"]))

    assert outcome["created"] == [queued[0]]
    task = database.get_task(queued[0])
    assert task["url"] == "https://www.youtube.com/watch?v=BBBBBBBBBBB"
    assert task["auto_publish"] == 1
    assert database.get_channel(CHANNEL["channel_id"])["last_seen"] == "2026-10-20T10:00:00+00:00"


def test_an_upload_that_already_has_a_task_is_not_queued_again(db, monkeypatch):
    database.upsert_channel(CHANNEL["channel_id"], last_seen="2026-10-19T10:00:00+00:00")
    database.create_or_get_video_task("https://www.youtube.com/watch?v=BBBBBBBBBBB", "BBBBBBBBBBB")
    monkeypatch.setattr(channels, "fetch_entries", lambda *_a, **_k: _entries())
    monkeypatch.setattr(channels.worker, "enqueue", lambda _task: pytest.fail("nothing should be queued"))

    outcome = channels.poll_channel(database.get_channel(CHANNEL["channel_id"]))

    assert outcome["created"] == []


def test_the_cursor_never_moves_backwards(db, monkeypatch):
    database.upsert_channel(CHANNEL["channel_id"], last_seen="2026-10-21T00:00:00+00:00")
    monkeypatch.setattr(channels, "fetch_entries", lambda *_a, **_k: _entries())
    monkeypatch.setattr(channels.worker, "enqueue", lambda _task: None)

    channels.poll_channel(database.get_channel(CHANNEL["channel_id"]))

    assert database.get_channel(CHANNEL["channel_id"])["last_seen"] == "2026-10-21T00:00:00+00:00"


def test_a_disabled_channel_is_skipped(db, monkeypatch):
    database.upsert_channel(CHANNEL["channel_id"], last_seen="2026-10-19T10:00:00+00:00")
    database.set_channel_enabled(CHANNEL["channel_id"], False)
    monkeypatch.setattr(channels, "fetch_entries", lambda *_a, **_k: pytest.fail("should not be fetched"))

    assert channels.poll_once(proxy=None)["channels"] == 0


def test_a_feed_error_is_reported_and_does_not_raise(db, monkeypatch):
    database.upsert_channel(CHANNEL["channel_id"], last_seen="2026-10-19T10:00:00+00:00")

    def boom(*_a, **_k):
        raise RuntimeError("HTTP 500")

    monkeypatch.setattr(channels, "fetch_entries", boom)

    outcome = channels.poll_channel(database.get_channel(CHANNEL["channel_id"]))
    assert outcome["created"] == []
    assert "HTTP 500" in outcome["error"]


def test_interval_comes_from_the_environment(monkeypatch):
    monkeypatch.delenv("YOUDUB_CHANNEL_POLL_MINUTES", raising=False)
    assert channels.interval_minutes() == channels.DEFAULT_INTERVAL_MINUTES
    monkeypatch.setenv("YOUDUB_CHANNEL_POLL_MINUTES", "7")
    assert channels.interval_minutes() == 7
    monkeypatch.setenv("YOUDUB_CHANNEL_POLL_MINUTES", "1")  # clamped up
    assert channels.interval_minutes() == channels.MIN_INTERVAL_MINUTES
    monkeypatch.setenv("YOUDUB_CHANNEL_POLL_MINUTES", "nonsense")
    assert channels.interval_minutes() == channels.DEFAULT_INTERVAL_MINUTES
