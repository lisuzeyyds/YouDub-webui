from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from backend.app import ad_cut
from backend.app.adapters import ffmpeg


def _write_translation(session: Path, items: list[dict]) -> Path:
    path = session / "metadata" / "translation.zh.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"translation": items}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path


def _read_translation(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))["translation"]


def _line(src: str, dst: str, start: int, end: int, **extra) -> dict:
    return {"src": src, "dst": dst, "start_time": start, "end_time": end, **extra}


# --------------------------------------------------------------------------
# span arithmetic
# --------------------------------------------------------------------------


def test_merge_spans_sorts_without_coalescing_distant_ranges():
    assert ad_cut.merge_spans([(5000, 6000), (1000, 2000), (3500, 4000)]) == [
        (1000, 2000),
        (3500, 4000),
        (5000, 6000),
    ]


def test_merge_spans_joins_ranges_within_the_gap():
    # 2000 -> 2600 is a 600ms gap, below AD_MERGE_GAP_MS (800): one ad break.
    assert ad_cut.merge_spans([(1000, 2000), (2600, 3000)]) == [(1000, 3000)]


def test_merge_spans_drops_empty_or_inverted_ranges():
    assert ad_cut.merge_spans([(1000, 1000), (4000, 3000), (5000, 6000)]) == [(5000, 6000)]


def test_plan_cut_pads_both_ends_and_clamps_to_the_duration():
    assert ad_cut.plan_cut([(10000, 12000)], duration_ms=20000) == [(9880, 12220)]
    # An ad running into the very end cannot pad past the video.
    assert ad_cut.plan_cut([(19000, 20000)], duration_ms=20000) == [(18880, 20000)]
    # ... and one at the very start cannot pad below zero.
    assert ad_cut.plan_cut([(0, 500)], duration_ms=20000) == [(0, 720)]


def test_plan_cut_merges_spans_that_padding_pushes_together():
    # 100ms apart: separate before padding, one cut after.
    assert ad_cut.plan_cut([(1000, 2000), (2100, 3000)]) == [(880, 3220)]


def test_remap_time_before_inside_and_after_a_cut():
    spans = [(10000, 12000)]
    assert ad_cut.remap_time(0, spans) == 0
    assert ad_cut.remap_time(9999, spans) == 9999
    # Inside the cut collapses onto the seam.
    assert ad_cut.remap_time(11000, spans) == 10000
    assert ad_cut.remap_time(12000, spans) == 10000
    assert ad_cut.remap_time(15000, spans) == 13000


def test_remap_time_subtracts_every_preceding_cut():
    spans = [(1000, 2000), (5000, 6000)]
    assert ad_cut.remap_time(10000, spans) == 10000 - 2000


# --------------------------------------------------------------------------
# apply_ad_cut
# --------------------------------------------------------------------------


def test_apply_ad_cut_drops_flagged_rows_and_rewrites_the_timeline(tmp_path):
    session = tmp_path / "session"
    translation = _write_translation(
        session,
        [
            _line("intro", "开头", 0, 1000),
            _line("buy my course", "", 10000, 12000, is_ad=True),
            _line("body", "正文", 20000, 22000),
        ],
    )

    report = ad_cut.apply_ad_cut(translation, session, duration_ms=30000)

    assert report.spans == [(9880, 12220)]
    assert report.cut_ms == 2340
    assert report.output_duration_ms == 27660
    assert report.changed
    assert len(report.removed) == 1

    kept = _read_translation(translation)
    assert [item["src"] for item in kept] == ["intro", "body"]
    assert (kept[0]["out_start_time"], kept[0]["out_end_time"]) == (0, 1000)
    assert (kept[1]["out_start_time"], kept[1]["out_end_time"]) == (17660, 19660)
    # The source timeline is preserved for anything that slices the source audio.
    assert (kept[1]["start_time"], kept[1]["end_time"]) == (20000, 22000)

    plan = json.loads(ad_cut.cut_artifact_path(session).read_text(encoding="utf-8"))
    assert plan["spans"] == [[9880, 12220]]
    assert plan["skipped_reason"] == ""


def test_apply_ad_cut_is_a_no_op_when_nothing_is_flagged(tmp_path):
    session = tmp_path / "session"
    translation = _write_translation(session, [_line("a", "甲", 0, 1000)])
    before = translation.read_text(encoding="utf-8")

    report = ad_cut.apply_ad_cut(translation, session, duration_ms=1000)

    assert report.spans == []
    assert not report.changed
    assert translation.read_text(encoding="utf-8") == before
    assert not ad_cut.cut_artifact_path(session).exists()


def test_apply_ad_cut_keeps_rows_without_timestamps(tmp_path):
    session = tmp_path / "session"
    translation = _write_translation(
        session,
        [
            {"src": "no timing", "dst": "无时间轴"},
            _line("ad", "", 10000, 12000, is_ad=True),
        ],
    )

    ad_cut.apply_ad_cut(translation, session, duration_ms=30000)

    kept = _read_translation(translation)
    assert [item["src"] for item in kept] == ["no timing"]
    assert "out_start_time" not in kept[0]


def test_apply_ad_cut_drops_a_real_row_swallowed_by_an_ad_break(tmp_path):
    session = tmp_path / "session"
    translation = _write_translation(
        session,
        [
            _line("ad", "", 10000, 12000, is_ad=True),
            # Sits fully inside the padded cut -> collapses to zero length.
            _line("bleed", "漏网", 10500, 11000),
        ],
    )

    report = ad_cut.apply_ad_cut(translation, session, duration_ms=20000)

    assert report.spans == [(9880, 12220)]
    assert _read_translation(translation) == []
    assert [item["src"] for item in report.removed] == ["ad", "bleed"]


def test_apply_ad_cut_abandons_the_video_cut_when_ads_cover_too_much(tmp_path):
    session = tmp_path / "session"
    translation = _write_translation(
        session,
        [
            _line("ad", "", 1000, 7000, is_ad=True),
            _line("body", "正文", 8000, 9000),
        ],
    )

    report = ad_cut.apply_ad_cut(translation, session, duration_ms=10000)

    # 6000ms of a 10000ms video trips AD_MAX_TOTAL_RATIO: the rows are still
    # dropped, but the video is left intact.
    assert report.spans == []
    assert "ads cover" in report.skipped_reason
    kept = _read_translation(translation)
    assert [item["src"] for item in kept] == ["body"]
    assert "out_start_time" not in kept[0]


def test_apply_ad_cut_abandons_the_video_cut_when_there_are_too_many_breaks(tmp_path):
    session = tmp_path / "session"
    items = [_line(f"ad{i}", "", i * 10000, i * 10000 + 100, is_ad=True) for i in range(45)]
    translation = _write_translation(session, items)

    report = ad_cut.apply_ad_cut(translation, session)

    assert report.spans == []
    assert "too many ad breaks" in report.skipped_reason
    assert _read_translation(translation) == []


def test_apply_ad_cut_still_records_an_abandoned_plan(tmp_path):
    session = tmp_path / "session"
    translation = _write_translation(
        session, [_line("ad", "", 1000, 7000, is_ad=True)]
    )

    ad_cut.apply_ad_cut(translation, session, duration_ms=10000)

    plan = json.loads(ad_cut.cut_artifact_path(session).read_text(encoding="utf-8"))
    assert plan["spans"] == []
    assert plan["skipped_reason"]
    assert len(plan["removed"]) == 1


def test_load_cut_spans_round_trips_and_tolerates_junk(tmp_path):
    session = tmp_path / "session"
    translation = _write_translation(
        session,
        [_line("ad", "", 10000, 12000, is_ad=True), _line("body", "正文", 20000, 22000)],
    )
    ad_cut.apply_ad_cut(translation, session, duration_ms=30000)
    assert ad_cut.load_cut_spans(session) == [(9880, 12220)]

    path = ad_cut.cut_artifact_path(session)
    path.write_text("not json", encoding="utf-8")
    assert ad_cut.load_cut_spans(session) == []

    path.write_text(json.dumps({"spans": [[1, 2], "bad", [3]]}), encoding="utf-8")
    assert ad_cut.load_cut_spans(session) == [(1, 2)]


def test_load_cut_spans_is_empty_without_a_plan(tmp_path):
    assert ad_cut.load_cut_spans(tmp_path / "session") == []


# --------------------------------------------------------------------------
# ffmpeg filtergraph
# --------------------------------------------------------------------------


def test_build_cut_expression_negates_every_span():
    assert ffmpeg.build_cut_expression([(1000, 2000)]) == "not(between(t,1.000,2.000))"
    assert (
        ffmpeg.build_cut_expression([(1000, 2000), (5000, 6000)])
        == "not(between(t,1.000,2.000)+between(t,5.000,6.000))"
    )


def test_build_cut_chains_burns_subtitles_after_the_cut():
    chains = ffmpeg.build_cut_chains(
        [(1000, 2000)],
        subtitle_filter_str="subtitles=filename='x.srt'",
        audio_source="[1:a]",
    )

    assert len(chains) == 2
    assert chains[0].startswith("[0:v]select='not(between(t,1.000,2.000))',setpts=N/FRAME_RATE/TB,")
    assert chains[0].endswith("[vout]")
    assert "subtitles=filename='x.srt'" in chains[0]
    assert chains[1] == "[1:a]aselect='not(between(t,1.000,2.000))',asetpts=N/SR/TB[aout]"


def test_build_cut_chains_without_audio_has_only_the_video_chain():
    chains = ffmpeg.build_cut_chains([(1000, 2000)])
    assert len(chains) == 1
    assert chains[0].endswith("[vout]")


def test_merge_video_cuts_spans_and_keeps_video_and_audio_in_sync(monkeypatch, tmp_path):
    session = tmp_path / "session"
    (session / "metadata").mkdir(parents=True)
    (session / "media").mkdir(parents=True)
    translation = _write_translation(session, [])
    commands: list[list[str]] = []

    def fake_ffmpeg(cmd, **kwargs):
        commands.append(cmd)
        Path(cmd[-1]).write_bytes(b"mp4")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(ffmpeg, "_run_ffmpeg", fake_ffmpeg)
    monkeypatch.setattr(ffmpeg, "probe_has_audio", lambda _path: True)

    result = ffmpeg.merge_video(
        tmp_path / "video.mp4",
        None,
        None,
        translation,
        session,
        output_mode="subtitles",
        cut_spans=[(1000, 2000)],
    )

    command = commands[-1]
    graph = command[command.index("-filter_complex") + 1]
    assert "select='not(between(t,1.000,2.000))'" in graph
    assert "aselect='not(between(t,1.000,2.000))'" in graph
    assert "subtitles=" in graph
    assert "-vf" not in command
    assert command[command.index("-map") + 1] == "[vout]"
    assert "[aout]" in command
    assert "-shortest" in command
    assert result == session / "media" / "video_final.mp4"


def test_merge_video_ignores_empty_or_inverted_spans(monkeypatch, tmp_path):
    session = tmp_path / "session"
    (session / "metadata").mkdir(parents=True)
    (session / "media").mkdir(parents=True)
    translation = _write_translation(session, [])
    commands: list[list[str]] = []

    def fake_ffmpeg(cmd, **kwargs):
        commands.append(cmd)
        Path(cmd[-1]).write_bytes(b"mp4")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(ffmpeg, "_run_ffmpeg", fake_ffmpeg)

    ffmpeg.merge_video(
        tmp_path / "video.mp4",
        None,
        None,
        translation,
        session,
        output_mode="subtitles",
        cut_spans=[(1000, 1000), (5000, 3000)],
    )

    assert "-filter_complex" not in commands[-1]
