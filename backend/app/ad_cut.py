"""Cut ad/promo segments out of a translated video.

The translator flags promo lines with ``is_ad``. Those lines are useless in a
re-upload (sponsor reads, "smash that subscribe button", affiliate pitches), so
they are dropped from the subtitle/dubbing timeline *and* their time span is cut
out of the finished video.

Cutting video changes the timeline, which is the delicate part:

* ``start_time``/``end_time`` stay the *source* timeline. Everything that reads
  the source audio (vocal reference slicing, source-take playback) keeps using
  them.
* ``out_start_time``/``out_end_time`` are the *output* timeline after the cuts.
  Everything that lays audio out on the finished video uses those.
* The SRT is generated from the merged timings, which already live on the output
  timeline, so the burned-in subtitles line up with the cut video for free.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

from . import runtime_security

log = logging.getLogger(__name__)

# Two flagged lines closer than this are one ad break, not two: an ad read is
# usually split into several ASR sentences with sub-second gaps between them.
AD_MERGE_GAP_MS = 800
# Slightly over-cut each ad break so the handoff does not clip into the ad read
# itself (leading) or leave a syllable of its tail behind (trailing).
AD_LEAD_PAD_MS = 120
AD_TAIL_PAD_MS = 220
# Guard rails: a model that misjudges the whole video as an ad would otherwise
# delete it. Past these limits the cut is abandoned (the ad rows are still
# dropped from the audio, the video is left intact).
AD_MAX_SPANS = 40
AD_MAX_TOTAL_RATIO = 0.4
# A sentence squeezed below this by an overlapping cut is not worth keeping.
AD_MIN_KEPT_MS = 300


def cut_artifact_path(session: Path) -> Path:
    return session / "metadata" / "ad_cut.json"


@dataclass
class AdCutReport:
    spans: list[tuple[int, int]] = field(default_factory=list)
    removed: list[dict] = field(default_factory=list)
    duration_ms: int | None = None
    skipped_reason: str = ""

    @property
    def cut_ms(self) -> int:
        return sum(end - start for start, end in self.spans)

    @property
    def output_duration_ms(self) -> int | None:
        if self.duration_ms is None:
            return None
        return max(0, self.duration_ms - self.cut_ms)

    @property
    def changed(self) -> bool:
        return bool(self.removed)


def merge_spans(spans: list[tuple[int, int]], gap_ms: int = AD_MERGE_GAP_MS) -> list[tuple[int, int]]:
    """Sort and coalesce spans whose gap is at most ``gap_ms``."""
    ordered = sorted((int(start), int(end)) for start, end in spans if int(end) > int(start))
    merged: list[tuple[int, int]] = []
    for start, end in ordered:
        if merged and start - merged[-1][1] <= gap_ms:
            previous_start, previous_end = merged[-1]
            merged[-1] = (previous_start, max(previous_end, end))
        else:
            merged.append((start, end))
    return merged


def plan_cut(
    ad_spans: list[tuple[int, int]],
    *,
    duration_ms: int | None = None,
    lead_ms: int = AD_LEAD_PAD_MS,
    tail_ms: int = AD_TAIL_PAD_MS,
    gap_ms: int = AD_MERGE_GAP_MS,
) -> list[tuple[int, int]]:
    """Turn flagged line ranges into the spans actually removed from the video."""
    padded: list[tuple[int, int]] = []
    for start, end in merge_spans(ad_spans, gap_ms):
        new_start = max(0, start - lead_ms)
        new_end = end + tail_ms
        if duration_ms is not None:
            new_start = min(new_start, duration_ms)
            new_end = min(new_end, duration_ms)
        if new_end > new_start:
            padded.append((new_start, new_end))
    # Padding can make neighbouring spans touch; those are one cut.
    return merge_spans(padded, 0)


def remap_time(t_ms: int, cut_spans: list[tuple[int, int]]) -> int:
    """Map a source timestamp onto the timeline after the cuts.

    A time inside a cut collapses onto that cut's start, i.e. the seam.
    """
    removed = 0
    for start, end in cut_spans:
        if t_ms >= end:
            removed += end - start
        elif t_ms > start:
            removed += t_ms - start
    return t_ms - removed


def _timestamp(item: dict) -> tuple[int, int] | None:
    try:
        start = int(item["start_time"])
        end = int(item["end_time"])
    except (KeyError, TypeError, ValueError):
        return None
    if end <= start:
        return None
    return start, end


def _summary(item: dict) -> dict:
    return {
        "src": str(item.get("src") or "")[:200],
        "start_time": item.get("start_time"),
        "end_time": item.get("end_time"),
    }


def load_cut_plan(session: Path) -> dict | None:
    path = cut_artifact_path(session)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        log.warning("ignoring unreadable ad cut plan %s: %s", path, exc)
        return None
    return data if isinstance(data, dict) else None


def load_cut_spans(session: Path) -> list[tuple[int, int]]:
    data = load_cut_plan(session)
    spans = data.get("spans") if data else None
    if not isinstance(spans, list):
        return []
    result: list[tuple[int, int]] = []
    for entry in spans:
        if isinstance(entry, (list, tuple)) and len(entry) == 2:
            try:
                result.append((int(entry[0]), int(entry[1])))
            except (TypeError, ValueError):
                continue
    return result


def apply_ad_cut(
    translation_file: Path,
    session: Path,
    *,
    duration_ms: int | None = None,
) -> AdCutReport:
    """Drop flagged lines and record the video spans to cut out.

    Returns a report describing what happened; the translation artifact is
    rewritten in place only when something actually changed.
    """
    data = json.loads(translation_file.read_text(encoding="utf-8"))
    items = data.get("translation")
    if not isinstance(items, list) or not items:
        return AdCutReport(duration_ms=duration_ms)

    flagged = [item for item in items if isinstance(item, dict) and item.get("is_ad")]
    if not flagged:
        return AdCutReport(duration_ms=duration_ms)

    # Rows with no usable time range can still be dropped, they just cannot
    # contribute a span to cut.
    spans = plan_cut(
        [span for span in (_timestamp(item) for item in flagged) if span is not None],
        duration_ms=duration_ms,
    )

    reason = ""
    if len(spans) > AD_MAX_SPANS:
        reason = f"too many ad breaks ({len(spans)} > {AD_MAX_SPANS})"
    elif duration_ms and sum(end - start for start, end in spans) > duration_ms * AD_MAX_TOTAL_RATIO:
        ratio = sum(end - start for start, end in spans) / duration_ms
        reason = f"ads cover {ratio:.0%} of the video (> {AD_MAX_TOTAL_RATIO:.0%})"
    if reason:
        log.warning("ad cut abandoned, dropping the ad lines only: %s", reason)
        spans = []

    removed = [_summary(item) for item in flagged]

    if spans:
        kept: list[dict] = []
        for item in items:
            if not isinstance(item, dict) or item.get("is_ad"):
                continue
            span = _timestamp(item)
            if span is None:
                kept.append(item)
                continue
            new_start = remap_time(span[0], spans)
            new_end = remap_time(span[1], spans)
            if new_end - new_start < AD_MIN_KEPT_MS:
                # The sentence sat entirely inside an ad break.
                removed.append(_summary(item))
                continue
            kept.append({**item, "out_start_time": new_start, "out_end_time": new_end})
        data["translation"] = kept
    else:
        data["translation"] = [
            item for item in items if not (isinstance(item, dict) and item.get("is_ad"))
        ]

    runtime_security.atomic_write_private_text(
        translation_file,
        json.dumps(data, ensure_ascii=False, indent=2),
    )

    report = AdCutReport(
        spans=spans,
        removed=removed,
        duration_ms=duration_ms,
        skipped_reason=reason,
    )
    payload = {
        "spans": [[start, end] for start, end in report.spans],
        "removed": removed,
        "duration_ms": duration_ms,
        "cut_ms": report.cut_ms,
        "output_duration_ms": report.output_duration_ms,
        "skipped_reason": reason,
    }
    try:
        runtime_security.atomic_write_private_text(
            cut_artifact_path(session),
            json.dumps(payload, ensure_ascii=False, indent=2),
        )
    except OSError as exc:
        log.warning("could not persist the ad cut plan: %s", exc)
    return report
