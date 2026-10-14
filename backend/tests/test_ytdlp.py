from pathlib import Path

import pytest

from backend.app.adapters import ffmpeg, ytdlp
from backend.app.errors import DeterministicStageError
from backend.app.sources import SourceConfig


def _make_source(*, use_proxy: bool, cookie_dir: Path) -> SourceConfig:
    cookie_path = cookie_dir / "missing-cookie.txt"

    class _Source(SourceConfig):
        @property
        def cookie_path(self):
            return cookie_path

    return _Source(
        name="test",
        matches=lambda url: True,
        use_proxy=use_proxy,
        cookie_filename="missing-cookie.txt",
        asr_language="en",
        target_language="zh",
    )


def test_ytdlp_proxy_port_takes_priority(monkeypatch, tmp_path):
    monkeypatch.setenv("HTTP_PROXY", "http://env-proxy:8080")
    source = _make_source(use_proxy=True, cookie_dir=tmp_path)

    options = ytdlp._ydl_base(source, "7890")

    assert options["proxy"] == "http://127.0.0.1:7890"


def test_ytdlp_proxy_falls_back_to_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("HTTP_PROXY", "http://env-proxy:8080")
    source = _make_source(use_proxy=True, cookie_dir=tmp_path)

    options = ytdlp._ydl_base(source, "")

    assert options["proxy"] == "http://env-proxy:8080"


def test_ytdlp_disables_proxy_when_source_opts_out(monkeypatch, tmp_path):
    monkeypatch.setenv("HTTP_PROXY", "http://env-proxy:8080")
    source = _make_source(use_proxy=False, cookie_dir=tmp_path)

    options = ytdlp._ydl_base(source, "7890")

    assert options["proxy"] == ""


def test_ytdlp_enables_node_js_runtime(tmp_path):
    source = _make_source(use_proxy=True, cookie_dir=tmp_path)

    options = ytdlp._ydl_base(source, "")

    assert options["js_runtimes"] == {"node": {}}


def test_ytdlp_asks_for_1080p_and_never_settles_for_best():
    assert ytdlp.MIN_VIDEO_HEIGHT == 1080
    assert ytdlp.FORMAT_CANDIDATES[0] == "bestvideo[height<=1080]+bestaudio"
    # Plain ``best`` is what silently produced a 360p master - it must not be
    # one of the candidates.
    assert "best" not in ytdlp.FORMAT_CANDIDATES


def _youtube_source() -> SourceConfig:
    return SourceConfig(
        name="youtube",
        matches=lambda url: True,
        use_proxy=False,
        cookie_filename=None,
        asr_language="en",
        target_language="zh",
    )


def test_download_video_passes_only_the_canonical_url_to_both_ytdlp_sinks(
    monkeypatch, tmp_path
):
    extracted_urls: list[str] = []
    downloaded_urls: list[str] = []

    class FakeYoutubeDL:
        def __init__(self, options):
            self.options = options

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def extract_info(self, url, *, download):
            extracted_urls.append(url)
            assert download is False
            return {
                "id": "abcdefghijk",
                "uploader": "tester",
                "title": "canonical",
                "webpage_url": url,
            }

        def sanitize_info(self, info):
            return info

        def download(self, urls):
            downloaded_urls.extend(urls)
            Path(self.options["outtmpl"]).write_bytes(b"video")

    monkeypatch.setattr(ytdlp.yt_dlp, "YoutubeDL", FakeYoutubeDL)
    # The master this fake writes is not a real video; state its resolution
    # rather than letting the probe fail on it.
    monkeypatch.setattr(ffmpeg, "probe_video_size", lambda path: (1920, 1080))

    session, _ = ytdlp.download_video(
        "HTTPS://WWW.YOUTUBE.COM:443/watch?v=abcdefghijk",
        tmp_path,
        _youtube_source(),
    )

    expected = "https://www.youtube.com/watch?v=abcdefghijk"
    assert extracted_urls == [expected]
    assert downloaded_urls == [expected]
    assert (session / "media" / "video_source.mp4").read_bytes() == b"video"


def test_output_mode_task_ids_use_separate_remote_sessions(tmp_path):
    info = {
        "id": "abcdefghijk",
        "uploader": "tester",
        "title": "same video",
    }

    subtitles = ytdlp._session_path(tmp_path, info, "abcdefghijk-subtitles")
    dubbing = ytdlp._session_path(tmp_path, info, "abcdefghijk-dubbing")
    both = ytdlp._session_path(tmp_path, info, "abcdefghijk")

    assert len({subtitles, dubbing, both}) == 3
    assert subtitles.name.endswith("__abcdefghijk-subtitles")
    assert dubbing.name.endswith("__abcdefghijk-dubbing")
    assert both.name.endswith("__abcdefghijk")


def test_download_video_rejects_deceptive_url_before_cookie_or_ytdlp(
    monkeypatch, tmp_path
):
    calls: list[str] = []
    monkeypatch.setattr(ytdlp, "_ensure_cookie", lambda source: calls.append("cookie"))
    monkeypatch.setattr(
        ytdlp.yt_dlp,
        "YoutubeDL",
        lambda options: calls.append("ytdlp"),
    )

    with pytest.raises(ValueError):
        ytdlp.download_video(
            "https://youtube.com.evil.example/watch?v=abcdefghijk",
            tmp_path,
            _youtube_source(),
        )

    assert calls == []


def test_remove_partial_outputs_clears_download_artifacts_only(tmp_path):
    """yt-dlp names a video-only stream ``video_source.f399.mp4.part``.

    Matching the bare file name missed exactly the large leftovers this helper
    exists for - a real task kept a 233 MB ``video_source.f399.mp4.part``
    behind after the format candidate it belonged to failed. A finished but
    unmerged stream (``video_source.f399.mp4``) is just as big and goes too,
    while a kept backup must survive: an earlier wider pattern deleted one
    during a failed re-download.
    """
    video_file = tmp_path / "video_source.mp4"
    video_file.write_bytes(b"finished")
    partial = tmp_path / "video_source.f399.mp4.part"
    partial.write_bytes(b"partial")
    marker = tmp_path / "video_source.ytdl"
    marker.write_bytes(b"state")
    unmerged = tmp_path / "video_source.f399.mp4"
    unmerged.write_bytes(b"stream without audio")
    backup = tmp_path / "video_source.360p.mp4"
    backup.write_bytes(b"backup")

    ytdlp._remove_partial_outputs(video_file)

    assert not partial.exists()
    assert not marker.exists()
    assert not unmerged.exists()
    assert backup.read_bytes() == b"backup"
    # The finished file itself is never touched.
    assert video_file.read_bytes() == b"finished"


def _youtube_info(*, heights=(360, 720, 1080)) -> dict:
    return {
        "id": "abcdefghijk",
        "uploader": "tester",
        "title": "clip",
        "webpage_url": "https://www.youtube.com/watch?v=abcdefghijk",
        "formats": [{"height": height} for height in heights],
    }


def _install_fake_ytdlp(monkeypatch, behaviour: dict) -> None:
    """Route every yt-dlp call inside ``download_video`` through ``behaviour``."""

    class FakeYoutubeDL:
        def __init__(self, options):
            self.options = options

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def extract_info(self, url, *, download):
            if "info_error" in behaviour:
                raise behaviour["info_error"]
            return behaviour["info"]

        def sanitize_info(self, info):
            return info

        def download(self, urls):
            behaviour["download"](self.options)

    monkeypatch.setattr(ytdlp.yt_dlp, "YoutubeDL", lambda options: FakeYoutubeDL(options))
    monkeypatch.setattr(ytdlp, "_ydl_base", lambda *args, **kwargs: {})


def test_download_names_the_cookie_when_the_source_asks_for_a_sign_in(monkeypatch, tmp_path):
    """A bot check means the cookie is stale - name it instead of dumping yt-dlp."""

    def fail_download(options):
        raise RuntimeError(
            "Sign in to confirm you're not a bot. Use --cookies for the authentication."
        )

    _install_fake_ytdlp(monkeypatch, {"info": _youtube_info(), "download": fail_download})

    with pytest.raises(DeterministicStageError) as info:
        ytdlp.download_video(
            "https://www.youtube.com/watch?v=abcdefghijk", tmp_path, _youtube_source()
        )

    message = str(info.value)
    assert "cookie" in message
    assert "refresh the YouTube cookie" in message


def test_download_keeps_network_trouble_retryable(monkeypatch, tmp_path):
    """An unreachable source is transient: retry it, but name the proxy."""

    def fail_download(options):
        raise RuntimeError(
            "unable to download video data: ConnectionResetError(10054, 'Connection reset by peer')"
        )

    _install_fake_ytdlp(monkeypatch, {"info": _youtube_info(), "download": fail_download})

    with pytest.raises(RuntimeError) as info:
        ytdlp.download_video(
            "https://www.youtube.com/watch?v=abcdefghijk", tmp_path, _youtube_source()
        )

    assert type(info.value) is RuntimeError
    assert "proxy" in str(info.value)


def test_download_refuses_a_below_1080p_master(monkeypatch, tmp_path):
    """The file is the authority: a 360p result is rejected, not shipped.

    The selectors can be satisfied by whatever a source has, so the height is
    checked on the downloaded file. It is deleted as well, otherwise a resume
    would accept it again through the "already downloaded" shortcut.
    """

    def write_low_res(options):
        Path(options["outtmpl"]).write_bytes(b"video")

    _install_fake_ytdlp(
        monkeypatch,
        {"info": _youtube_info(heights=(360, 720)), "download": write_low_res},
    )
    monkeypatch.setattr(ffmpeg, "probe_video_size", lambda path: (640, 360))

    with pytest.raises(DeterministicStageError) as info:
        ytdlp.download_video(
            "https://www.youtube.com/watch?v=abcdefghijk", tmp_path, _youtube_source()
        )

    message = str(info.value)
    assert "below the required 1080p" in message
    assert "no 1080p stream" in message
    assert not (
        tmp_path / "tester" / "clip__abcdefghijk" / "media" / "video_source.mp4"
    ).exists()


def test_download_keeps_a_1080p_master(monkeypatch, tmp_path):
    def write_high_res(options):
        Path(options["outtmpl"]).write_bytes(b"video")

    _install_fake_ytdlp(monkeypatch, {"info": _youtube_info(), "download": write_high_res})
    monkeypatch.setattr(ffmpeg, "probe_video_size", lambda path: (1920, 1080))

    session, _ = ytdlp.download_video(
        "https://www.youtube.com/watch?v=abcdefghijk", tmp_path, _youtube_source()
    )

    assert (session / "media" / "video_source.mp4").read_bytes() == b"video"
