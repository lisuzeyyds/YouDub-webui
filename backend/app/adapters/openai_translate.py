from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    OpenAI,
)
from pydantic import BaseModel, Field, ValidationError, model_validator

from ..errors import DeterministicStageError
from ..sources import SourceConfig
from ._translate_prompts import (
    BULK_COPY_PROMPT,
    BULK_TRANSLATION_PROMPT,
    CONTENT_ONLY_TRANSLATION_RULES,
    PREPROCESS_PROMPT,
    TRANSLATE_RULES,
)
from .openai_client import normalize_openai_base_url

log = logging.getLogger(__name__)

API_SETTING_KEYS = ("base_url", "api_key", "model")
PREPROCESS_RETRY = 2
TRANSLATE_RETRY = 2
TRANSLATE_ATTEMPTS = 4
TRANSPORT_BACKOFF = (2, 5, 10, 20)
DESCRIPTION_LIMIT = 500
DEFAULT_CONCURRENCY = 50

# Bulk (whole-transcript) translation.
# Sending every sentence in one request gives the model global context (consistent
# terminology, names, tone) and removes the concurrency that was dropping
# connections mid-batch. The trade-off is long outputs, which can be truncated,
# so the reply protocol is one self-contained line per sentence and only the
# lines that are actually missing get retried individually.
#
# The request is streamed (see ``_call_bulk``), which is what lets a payload this
# large complete at all: the relay's gateway cuts any response whose body has not
# started after ~100s, and a full transcript takes minutes to generate. A
# payload past the cap is split into consecutive chunks rather than falling back
# to one request per sentence, because 1068 sentences at ~27s each (concurrency
# 2) is an hour while the same text in three streamed chunks is a few minutes.
BULK_MAX_CHARS = 40000
BULK_ATTEMPTS = 3
# Share of a chunk a mid-stream drop must have delivered before the partial
# reply is kept instead of retried; below it the chunk is cheap enough to redo.
BULK_PARTIAL_MIN_RATIO = 0.25
# Output budget. This is not a nicety: with no ``max_tokens`` the relay applies
# its own (small) default, and a long reply then *ends cleanly* part-way through
# - ``finish_reason='stop'`` with a third of the lines, no error, nothing to
# retry. Measured on this relay: the same 542-line request stops after 95 lines
# without the parameter and returns 513/513 in 210s with it.
BULK_MAX_TOKENS = 16000
# A long reply can still end early (stream cut, gateway hiccup). Asking again for
# just the uncovered tail is one more request; handing the same leftovers to the
# per-sentence path is 400+ requests and hours.
BULK_TAIL_ROUNDS = 4
# The upload copy (title/description/tags) comes back inside the whole-transcript
# reply, but on a long transcript the model occasionally translates every line
# and drops the header sections. The copy describes the *video*, so it is worth
# one small extra request against an evenly sampled transcript rather than
# shipping no title at all.
COPY_CONDENSE_CHARS = 4000
COPY_ATTEMPTS = 2
_BULK_LINE_RE = re.compile(r"^\s*(\d+)\s*[\t:：.、]\s*(.*)$")

# The whole-transcript request also returns the Bilibili metadata and the list of
# ad lines. Both are emitted *before* the (long, truncatable) translation block,
# so a reply cut short still carries them.
BULK_SECTION_RE = re.compile(r"^\s*#={2,}\s*(.+?)\s*={2,}\s*$")
BULK_AD_MARK = "[AD]"
BULK_SECTION_TITLE = "标题"
BULK_SECTION_DESC = "简介"
BULK_SECTION_TAGS = "标签"
BULK_SECTION_ADS = "广告行号"
BULK_SECTION_LINES = "逐行译文"
_AD_RANGE_RE = re.compile(r"(\d+)\s*[-~—－–至]\s*(\d+)")
_AD_NUM_RE = re.compile(r"\d+")
_AD_NONE_VALUES = {"", "-", "无", "没有", "none", "null", "n/a", "na", "[]", "（）", "()"}
_TAG_SPLIT_RE = re.compile(r"[,，、;；#\s]+")

_TRANSIENT_STATUS = {408, 409, 425, 429}
_QUOTA_MARKERS = (
    "insufficient_user_quota",
    "insufficient_quota",
    "余额不足",
    "quota",
)


class TranslationApiError(DeterministicStageError):
    """The endpoint answered, but the request can never succeed as configured."""


class TransientApiError(RuntimeError):
    """A network-level hiccup that is worth retrying."""


class PartialReplyError(TransientApiError):
    """A streamed reply that died part-way through, carrying what it did send.

    Retrying is the default, but a whole transcript takes minutes to stream and
    a drop near the end would throw away almost all of it. The partial text
    travels with the error so the caller can keep it when it already covers
    enough of the transcript.
    """

    def __init__(self, message: str, partial: str) -> None:
        super().__init__(message)
        self.partial = partial


class HotwordItem(BaseModel):
    src: str
    dst: str


class CorrectionItem(BaseModel):
    wrong: str
    correct: str


class PreprocessResponse(BaseModel):
    summary: str = ""
    hotwords: list[HotwordItem] = Field(default_factory=list)
    corrections: list[CorrectionItem] = Field(default_factory=list)


class TranslationItem(BaseModel):
    dst: str
    audio_mode: Literal["tts", "original"]
    is_ad: bool = False

    @model_validator(mode="after")
    def validate_tts_text(self) -> "TranslationItem":
        if self.audio_mode == "tts" and not self.dst.strip():
            raise ValueError("dst must be non-empty when audio_mode is tts")
        return self


@dataclass
class BulkResult:
    """Everything a single whole-transcript request returns."""

    lines: dict[int, str] = field(default_factory=dict)
    ad_lines: set[int] = field(default_factory=set)
    title: str = ""
    description: str = ""
    tags: list[str] = field(default_factory=list)

    def merged_ad_lines(self, expected: int) -> set[int]:
        """Ad indexes from the explicit list, plus any line marked ``[AD]``.

        The explicit list is emitted before the translation block, so it
        survives truncation; the inline marker is a second witness for replies
        that were cut off before the header. Validating against ``expected``
        keeps a hallucinated index from deleting a real sentence.
        """
        result = {index for index in self.ad_lines if 1 <= index <= expected}
        for index, text in self.lines.items():
            if 1 <= index <= expected and text.strip().upper() == BULK_AD_MARK:
                result.add(index)
        return result


def _error_body(exc: APIStatusError) -> str:
    """Everything the endpoint told us, whichever attribute it landed in."""
    parts: list[str] = [str(exc)]
    body = getattr(exc, "body", None)
    if body is not None:
        if isinstance(body, str):
            parts.append(body)
        else:
            try:
                parts.append(json.dumps(body, ensure_ascii=False))
            except (TypeError, ValueError):
                parts.append(str(body))
    response = getattr(exc, "response", None)
    if response is not None:
        try:
            parts.append(response.text)
        except Exception:  # noqa: BLE001 - response body may be already consumed
            pass
    return " ".join(part for part in parts if part)


def _api_error_message(exc: APIStatusError) -> str:
    body = _error_body(exc)
    status = getattr(exc, "status_code", None)
    lowered = body.lower()
    if any(marker.lower() in lowered for marker in _QUOTA_MARKERS):
        return (
            f"翻译接口账户额度不足（HTTP {status}，服务端返回“余额不足”）。"
            "请到接口服务商控制台充值，或更换 API Key / 模型后重试。"
        )
    if status in (401, 403):
        return (
            f"翻译接口拒绝访问（HTTP {status}）：API Key 无效、无权限或额度不足，"
            f"请在设置里检查接口配置。原文：{body[:200]}"
        )
    if status == 404:
        return (
            f"翻译接口找不到模型（HTTP 404）：请检查设置里的模型名。原文：{body[:200]}"
        )
    return f"翻译接口返回错误（HTTP {status}）：{body[:300]}"


def list_models(*, base_url: str, api_key: str) -> list[str]:
    if not api_key:
        raise ValueError("OpenAI API key is not configured.")
    client = OpenAI(api_key=api_key, base_url=normalize_openai_base_url(base_url))
    response = client.models.list()
    seen: set[str] = set()
    models: list[str] = []
    for item in response.data:
        model_id = getattr(item, "id", "")
        if model_id and model_id not in seen:
            seen.add(model_id)
            models.append(model_id)
    return models


def _client(base_url: str, api_key: str) -> OpenAI:
    if not api_key:
        raise ValueError("OpenAI API key is not configured.")
    return OpenAI(api_key=api_key, base_url=normalize_openai_base_url(base_url))


_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


def _extract_json(raw: str) -> dict[str, Any]:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    match = _JSON_BLOCK_RE.search(raw)
    if not match:
        raise json.JSONDecodeError(f"no JSON object found; raw[:300]={raw[:300]!r}", raw, 0)
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        raise json.JSONDecodeError(
            f"{exc.msg}; len={len(raw)}; raw[:300]={raw[:300]!r}; raw[-200:]={raw[-200:]!r}",
            raw,
            exc.pos,
        ) from None


def _call_json(client: OpenAI, model: str, system: str, user: str) -> dict[str, Any]:
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.2,
        )
    except (APIConnectionError, APITimeoutError) as exc:
        # "Server disconnected without sending a response." lands here; the relay
        # under high concurrency does this occasionally and recovers on its own.
        raise TransientApiError(f"{type(exc).__name__}: {exc}") from exc
    except APIStatusError as exc:
        status = getattr(exc, "status_code", None)
        if status in _TRANSIENT_STATUS or (isinstance(status, int) and status >= 500):
            raise TransientApiError(f"HTTP {status}: {str(exc)[:200]}") from exc
        raise TranslationApiError(_api_error_message(exc)) from exc
    raw = response.choices[0].message.content or "{}"
    return _extract_json(raw)


def _format_terms(items: list, fmt: str, empty: str) -> str:
    if not items:
        return empty
    return "\n".join(fmt.format(**item.model_dump()) for item in items)


def _meta_view(meta: dict[str, Any]) -> dict[str, str]:
    description = (meta.get("description") or "").strip()
    if len(description) > DESCRIPTION_LIMIT:
        description = description[:DESCRIPTION_LIMIT] + "..."
    return {
        "title": str(meta.get("title") or "").strip() or "(unknown)",
        "uploader": str(meta.get("uploader") or "").strip() or "(unknown)",
        "description": description or "(none)",
    }


def preprocess(
    full_text: str,
    meta: dict[str, Any],
    source: SourceConfig,
    *,
    base_url: str,
    api_key: str,
    model: str,
) -> PreprocessResponse:
    user = PREPROCESS_PROMPT.format(
        src_language_name=source.asr_language_name,
        dst_language_name=source.target_language_name,
        full_text=full_text,
        **_meta_view(meta),
    )
    client = _client(base_url, api_key)
    last_error: Exception | None = None
    for attempt in range(PREPROCESS_RETRY + 1):
        try:
            data = _call_json(client, model, "You output strict JSON only.", user)
            return PreprocessResponse.model_validate(data)
        except TranslationApiError:
            raise
        except TransientApiError as exc:
            last_error = exc
            log.warning("preprocess attempt %d failed: %s", attempt + 1, exc)
            if attempt < PREPROCESS_RETRY:
                time.sleep(_backoff(attempt))
        except (json.JSONDecodeError, ValidationError) as exc:
            last_error = exc
            log.warning("preprocess attempt %d failed: %s", attempt + 1, exc)
    log.error("preprocess gave up, returning empty: %s", last_error)
    return PreprocessResponse()


def _translate_system(source: SourceConfig, meta: dict[str, Any], pre: PreprocessResponse) -> str:
    rules = TRANSLATE_RULES[(source.asr_language, source.target_language)]
    formatted = rules.format(
        summary=pre.summary or "(none)",
        hotwords=_format_terms(pre.hotwords, "{src} -> {dst}", "(none)"),
        corrections=_format_terms(pre.corrections, "{wrong} -> {correct}", "(none)"),
        **_meta_view(meta),
    )
    return f"{formatted}\n\n{CONTENT_ONLY_TRANSLATION_RULES}"


def _post_process(text: str, target_language: str) -> str:
    cleaned = text.strip()
    if target_language == "zh":
        cleaned = cleaned.replace("——", "，")
    return cleaned


def _backoff(attempt: int) -> float:
    """Seconds to wait before retry `attempt` (0-based)."""
    return float(TRANSPORT_BACKOFF[min(attempt, len(TRANSPORT_BACKOFF) - 1)])


def _bulk_payload(texts: list[str]) -> str:
    return "\n".join(f"{index}\t{text}" for index, text in enumerate(texts, start=1))


def _bulk_system(
    source: SourceConfig,
    meta: dict[str, Any],
    pre: PreprocessResponse,
    source_url: str = "",
) -> str:
    return BULK_TRANSLATION_PROMPT.format(
        system=_translate_system(source, meta, pre),
        source_title=str(meta.get("title") or "").strip() or "(unknown)",
        source_uploader=str(meta.get("uploader") or "").strip() or "(unknown)",
        source_url=source_url.strip() or "(unknown)",
        payload="",
    ).rstrip()


def _bulk_copy_system(
    meta: dict[str, Any],
    source_url: str = "",
) -> str:
    return BULK_COPY_PROMPT.format(
        source_title=str(meta.get("title") or "").strip() or "(unknown)",
        source_uploader=str(meta.get("uploader") or "").strip() or "(unknown)",
        source_url=source_url.strip() or "(unknown)",
        payload="",
    ).rstrip()


def _condense(texts: list[str], budget: int = COPY_CONDENSE_CHARS) -> list[str]:
    """Evenly sample ``texts`` down to roughly ``budget`` characters.

    Both ends are always kept and the rest is spread evenly, so the sample
    describes the whole video instead of truncating to the opening minutes.
    """
    if not texts:
        return []
    total = sum(len(text) + 1 for text in texts)
    if total <= budget:
        return list(texts)
    keep = max(2, len(texts) * budget // total)
    last = len(texts) - 1
    while keep < len(texts):
        step = last / (keep - 1)
        indices = sorted({round(index * step) for index in range(keep)})
        sampled = [texts[index] for index in indices]
        if len(sampled) <= 2 or sum(len(text) + 1 for text in sampled) <= budget:
            return sampled
        keep = max(2, keep // 2)
    return list(texts)


def _copy_request(
    texts: list[str],
    system: str,
    *,
    client: OpenAI,
    model: str,
) -> BulkResult | None:
    """One small request for the upload copy alone.

    Used when the whole-transcript reply translated every line but omitted the
    header sections; unlike :func:`_bulk_request` a reply with no translation
    block is exactly what is wanted here, so only the copy sections are checked.
    """
    if not texts:
        return None
    payload = _bulk_payload(texts)
    for attempt in range(COPY_ATTEMPTS):
        try:
            raw = _call_bulk(client, model, system, payload)
        except TranslationApiError:
            raise
        except TransientApiError as exc:
            log.warning(
                "copy request transport failure (attempt %d/%d): %s",
                attempt + 1,
                COPY_ATTEMPTS,
                exc,
            )
            if attempt < COPY_ATTEMPTS - 1:
                time.sleep(_backoff(attempt))
            continue
        parsed = _parse_bulk(raw, len(texts))
        if parsed.title or parsed.description:
            log.info(
                "copy request: title=%r, %d tag(s)",
                parsed.title[:40],
                len(parsed.tags),
            )
            return parsed
        log.warning(
            "copy request returned no usable copy (attempt %d/%d); raw[:200]=%r",
            attempt + 1,
            COPY_ATTEMPTS,
            raw[:200],
        )
    return None


def _split_bulk_sections(raw: str) -> dict[str, str]:
    """Split a reply on the ``#=== name ===`` headers into {name: body}.

    Everything before the first header is dropped (model preamble), and the
    header name is whitespace-normalised so a stray space does not lose a
    section.
    """
    sections: dict[str, str] = {}
    name: str | None = None
    buffer: list[str] = []

    def flush() -> None:
        if name is not None and name not in sections:
            sections[name] = "\n".join(buffer).strip()

    for line in raw.splitlines():
        match = BULK_SECTION_RE.match(line)
        if match:
            flush()
            name = re.sub(r"\s+", "", match.group(1))
            buffer = []
            continue
        if name is not None:
            buffer.append(line)
    flush()
    return sections


def _parse_ad_lines(text: str, expected: int) -> set[int]:
    """Parse ``3,7,12-14`` into a set of 1-based line numbers.

    Ranges are expanded first, then the remaining bare numbers are collected.
    Every value is clamped to ``1..expected`` so a hallucinated index cannot
    delete a sentence that does not exist.
    """
    result: set[int] = set()
    if not text or not text.strip():
        return result
    if re.sub(r"\s+", "", text).lower() in _AD_NONE_VALUES:
        return result

    remainder_parts: list[str] = []
    cursor = 0
    for match in _AD_RANGE_RE.finditer(text):
        remainder_parts.append(text[cursor : match.start()])
        cursor = match.end()
        start, end = int(match.group(1)), int(match.group(2))
        if start > end:
            start, end = end, start
        if end - start > expected:
            continue
        for number in range(start, end + 1):
            if 1 <= number <= expected:
                result.add(number)
    remainder_parts.append(text[cursor:])

    for match in _AD_NUM_RE.finditer(" ".join(remainder_parts)):
        number = int(match.group(0))
        if 1 <= number <= expected:
            result.add(number)
    return result


def _parse_tags(text: str) -> list[str]:
    if not text:
        return []
    seen: set[str] = set()
    tags: list[str] = []
    for raw in _TAG_SPLIT_RE.split(text):
        tag = raw.strip()
        if tag and tag not in seen:
            seen.add(tag)
            tags.append(tag)
    return tags


def _parse_bulk_reply(raw: str, expected: int) -> dict[int, str]:
    """Map line number -> translation for a whole-transcript reply.

    Only well-formed ``<n><TAB>text`` lines are kept; anything else (model
    preamble, markdown fences, a truncated tail) is ignored so a partial reply
    still yields every line it did contain.
    """
    result: dict[int, str] = {}
    for line in raw.splitlines():
        match = _BULK_LINE_RE.match(line)
        if not match:
            continue
        number = int(match.group(1))
        if 1 <= number <= expected and number not in result:
            result[number] = match.group(2).strip()
    return result


def _parse_bulk(raw: str, expected: int) -> BulkResult:
    """Parse a whole-transcript reply into translations + metadata + ad lines."""
    sections = _split_bulk_sections(raw)
    return BulkResult(
        lines=_parse_bulk_reply(raw, expected),
        ad_lines=_parse_ad_lines(sections.get(BULK_SECTION_ADS, ""), expected),
        title=sections.get(BULK_SECTION_TITLE, "").strip(),
        description=sections.get(BULK_SECTION_DESC, "").strip(),
        tags=_parse_tags(sections.get(BULK_SECTION_TAGS, "")),
    )


def _coerce_bulk(value: Any) -> BulkResult | None:
    """Accept a BulkResult or a bare {index: text} mapping from tests/stubs."""
    if value is None:
        return None
    if isinstance(value, BulkResult):
        return value
    if isinstance(value, dict):
        return BulkResult(lines=value)
    return None


def _open_bulk_stream(client: OpenAI, model: str, system: str, payload: str):
    """Start the streamed whole-transcript completion with an explicit output cap.

    ``max_tokens`` is set explicitly because leaving it out lets the relay decide,
    and its default is small enough that a long reply simply stops in the middle
    without an error (see ``BULK_MAX_TOKENS``). A relay that does not accept the
    value (some reject anything above the model's own ceiling) is retried without
    the parameter so the request still goes through - just with the old risk.
    """
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": payload},
    ]
    try:
        return client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.2,
            stream=True,
            max_tokens=BULK_MAX_TOKENS,
        )
    except APIStatusError as exc:
        if _rejects_max_tokens(exc):
            log.warning(
                "relay rejected max_tokens=%d (%s); retrying without the cap",
                BULK_MAX_TOKENS,
                str(exc)[:120],
            )
            return client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.2,
                stream=True,
            )
        raise


def _rejects_max_tokens(exc: APIStatusError) -> bool:
    """True when a 4xx is the relay refusing the ``max_tokens`` value itself."""
    status = getattr(exc, "status_code", None)
    return status == 400 and "max_tokens" in str(exc).lower()


def _call_bulk(client: OpenAI, model: str, system: str, payload: str) -> str:
    """One plain-text completion for the whole transcript, read as a stream.

    Streaming is not a nicety here, it is what makes the whole-transcript path
    possible at all. A relay sitting behind Cloudflare (this one does) kills any
    non-streamed request whose *first byte of body* takes longer than ~100s with
    a 524, so a 26k-char transcript that needs ~4 minutes of generation could
    never come back in one piece. Once the response is chunked the gateway sees
    bytes immediately and the transfer is limited only by the model itself.

    Measured on this relay: the same 1068-line transcript is a 524 at ~100s
    without ``stream``, and completes all 1068 lines in 263s with it.
    """
    parts: list[str] = []
    try:
        stream = _open_bulk_stream(client, model, system, payload)
        for chunk in stream:
            choices = getattr(chunk, "choices", None) or []
            if not choices:
                continue
            delta = getattr(choices[0], "delta", None)
            content = getattr(delta, "content", None) if delta is not None else None
            if content:
                parts.append(content)
    except (APIConnectionError, APITimeoutError) as exc:
        partial = "".join(parts)
        message = f"{type(exc).__name__}: {exc}"
        if partial:
            raise PartialReplyError(message, partial) from exc
        raise TransientApiError(message) from exc
    except APIStatusError as exc:
        status = getattr(exc, "status_code", None)
        if status in _TRANSIENT_STATUS or (isinstance(status, int) and status >= 500):
            partial = "".join(parts)
            message = f"HTTP {status}: {str(exc)[:200]}"
            if partial:
                raise PartialReplyError(message, partial) from exc
            raise TransientApiError(message) from exc
        raise TranslationApiError(_api_error_message(exc)) from exc
    return "".join(parts)


def _chunk_by_payload(texts: list[str], max_chars: int) -> list[list[str]]:
    """Greedy split of ``texts`` into consecutive groups that fit ``max_chars``.

    Sizes are measured on the numbered payload the model actually receives
    (``"<n>\\t<text>"``), because that is what the cap protects, and numbering
    restarts inside every chunk. A single sentence longer than the cap still gets
    a chunk of its own rather than being dropped.
    """
    chunks: list[list[str]] = []
    current: list[str] = []
    size = 0
    for text in texts:
        line_len = len(text) + len(str(len(current) + 1)) + 2
        if current and size + line_len > max_chars:
            chunks.append(current)
            current = []
            size = 0
            line_len = len(text) + 2
        current.append(text)
        size += line_len
    if current:
        chunks.append(current)
    return chunks


def _bulk_request(
    texts: list[str],
    system: str,
    *,
    client: OpenAI,
    model: str,
) -> BulkResult | None:
    """Run one streamed whole-transcript request; ``None`` when it is unusable."""
    payload = _bulk_payload(texts)
    last_error: Exception | None = None
    for attempt in range(BULK_ATTEMPTS):
        try:
            raw = _call_bulk(client, model, system, payload)
        except TranslationApiError:
            raise
        except TransientApiError as exc:
            last_error = exc
            partial = getattr(exc, "partial", "")
            log.warning(
                "bulk translate stream failed (attempt %d/%d) after %d char(s): %s",
                attempt + 1,
                BULK_ATTEMPTS,
                len(partial),
                exc,
            )
            if partial:
                parsed = _parse_bulk(partial, len(texts))
                # A whole-transcript reply takes minutes to stream, so a drop
                # near the end must not discard it: keep what arrived once it
                # covers enough of the chunk, and let the caller top up the rest
                # sentence by sentence.
                if len(parsed.lines) >= max(1, int(len(texts) * BULK_PARTIAL_MIN_RATIO)):
                    log.warning(
                        "bulk translate keeping the partial stream: %d/%d lines",
                        len(parsed.lines),
                        len(texts),
                    )
                    return parsed
            if attempt < BULK_ATTEMPTS - 1:
                time.sleep(_backoff(attempt))
            continue
        parsed = _parse_bulk(raw, len(texts))
        if parsed.lines or parsed.ad_lines:
            log.info(
                "bulk translate: %d/%d lines, %d ad line(s), title=%r",
                len(parsed.lines),
                len(texts),
                len(parsed.ad_lines),
                parsed.title[:40],
            )
            return parsed
        log.warning(
            "bulk translate reply had no usable lines (attempt %d/%d); raw[:200]=%r",
            attempt + 1,
            BULK_ATTEMPTS,
            raw[:200],
        )
    log.warning("bulk translate gave up after %d attempts: %s", BULK_ATTEMPTS, last_error)
    return None


def _bulk_request_chunk(
    texts: list[str],
    system: str,
    *,
    client: OpenAI,
    model: str,
) -> BulkResult | None:
    """Serve one chunk, re-asking only for whatever the replies left out.

    A long reply can end before its last line - the relay stops the stream, or the
    model just stops. Everything it did deliver is kept, and the uncovered tail is
    requested again: the tail is a suffix, so line numbers stay comparable and each
    round covers more of a shrinking remainder. Without this the leftovers go to
    the per-sentence path, which is the difference between one more request and a
    few hundred.
    """
    if not texts:
        return BulkResult()
    merged = BulkResult()
    answered: set[int] = set()
    rounds = 0
    while rounds < BULK_TAIL_ROUNDS:
        remaining = [text for number, text in enumerate(texts, start=1) if number not in answered]
        if not remaining:
            break
        reply = _bulk_request(remaining, system, client=client, model=model)
        if reply is None:
            break
        base = len(texts) - len(remaining)
        gained = 0
        # Ads first: an ad line keeps no target text, and a reply that lists it in
        # both the ad block and the translation block must not be recorded as a
        # normal line (the downstream stage would then dub what it is cutting).
        for number in reply.ad_lines:
            position = base + number
            if position not in answered:
                answered.add(position)
                merged.ad_lines.add(position)
                gained += 1
        for number, text in reply.lines.items():
            position = base + number
            if position not in answered:
                answered.add(position)
                merged.lines[position] = text
                gained += 1
        if not merged.title:
            merged.title = reply.title
        if not merged.description:
            merged.description = reply.description
        if not merged.tags:
            merged.tags = list(reply.tags)
        rounds += 1
        if not gained:
            break
        if rounds > 1:
            log.info(
                "bulk translate continued for the missing tail: %d line(s) still open",
                len(texts) - len(answered),
            )
    if not (merged.lines or merged.ad_lines):
        return None
    return merged


def translate_bulk(
    texts: list[str],
    source: SourceConfig,
    meta: dict[str, Any],
    pre: PreprocessResponse,
    *,
    client: OpenAI,
    model: str,
    source_url: str = "",
) -> BulkResult | None:
    """Ask for everything in as few requests as the payload cap allows.

    One request means the model sees every sentence together (consistent
    terminology, names and tone) and there is no concurrency to drop
    connections. It also lets the same call judge which lines are ads and write
    the Bilibili title/description, because those judgements need the same
    whole-video context.

    The cost is a long reply that can be truncated, which the numbered-line
    protocol absorbs: the metadata and ad list are emitted *before* the
    translation block, so a cut-off reply still carries them, and every line it
    did contain survives.

    A transcript past the cap is cut into consecutive chunks that are requested
    one after another and merged, not abandoned to per-sentence translation: the
    fan-out is what turns a five-minute stage into an hour, and the chunks still
    share the video-level ``meta`` that the copy is written from. Only when a
    chunk comes back empty do its own lines fall through to the caller's
    per-sentence path.

    Returns a :class:`BulkResult`, or ``None`` when no request could be served.
    """
    if not texts:
        return BulkResult()
    system = _bulk_system(source, meta, pre, source_url)
    chunks = _chunk_by_payload(texts, BULK_MAX_CHARS)
    if len(chunks) > 1:
        log.info(
            "bulk translate: %d sentences split into %d chunk(s) of <=%d chars",
            len(texts),
            len(chunks),
            BULK_MAX_CHARS,
        )
    merged = BulkResult()
    offset = 0
    served = False
    for chunk in chunks:
        result = _bulk_request_chunk(chunk, system, client=client, model=model)
        if result is not None:
            served = True
            for number, text in result.lines.items():
                merged.lines[offset + number] = text
            for number in result.ad_lines:
                merged.ad_lines.add(offset + number)
            if not merged.title:
                merged.title = result.title
            if not merged.description:
                merged.description = result.description
            if not merged.tags:
                merged.tags = list(result.tags)
        offset += len(chunk)
    return merged if served else None


def _fix_translation_item(dst: str) -> TranslationItem:
    """Build a TranslationItem from a bulk line, matching the single-sentence rules.

    An empty line means "no information worth speaking": keep the source audio
    instead of asking TTS to synthesise silence, which is exactly what
    ``audio_mode=original`` encodes downstream.
    """
    value = dst.strip()
    if not value:
        return TranslationItem(dst="", audio_mode="original")
    return TranslationItem(dst=value, audio_mode="tts")


def _ad_translation_item() -> TranslationItem:
    """A line the model judged to be an ad.

    It carries no target text and is flagged so the translate stage can drop it
    (and cut its time span out of the finished video) instead of dubbing it.
    """
    return TranslationItem(dst="", audio_mode="original", is_ad=True)


def translate_sentence(
    text: str,
    target_language: str,
    client: OpenAI,
    model: str,
    system: str,
    *,
    attempts: int = TRANSLATE_ATTEMPTS,
) -> TranslationItem:
    parse_failures = 0
    attempts = max(1, attempts)
    last_transport: Exception | None = None
    for attempt in range(attempts):
        try:
            data = _call_json(client, model, system, text)
            item = TranslationItem.model_validate(data)
            return item.model_copy(update={"dst": _post_process(item.dst, target_language)})
        except TranslationApiError:
            raise
        except TransientApiError as exc:
            last_transport = exc
            log.warning(
                "translate transport failure (attempt %d/%d) for %r: %s",
                attempt + 1,
                attempts,
                text[:60],
                exc,
            )
            if attempt < attempts - 1:
                time.sleep(_backoff(attempt))
        except (json.JSONDecodeError, ValidationError, ValueError) as exc:
            parse_failures += 1
            log.warning("translate attempt %d failed for %r: %s", parse_failures, text[:60], exc)
            if parse_failures >= TRANSLATE_RETRY:
                raise RuntimeError(
                    f"translate_sentence failed after {parse_failures} attempts: {exc}"
                ) from exc
    raise RuntimeError(f"translate_sentence failed after {attempts} attempts: {last_transport}")


class TranslationCache:
    """Per-sentence translation cache.

    The relay endpoint occasionally drops one connection in the middle of a
    batch. Without a cache that single hiccup throws away every already
    translated (and already paid for) sentence, so a retry restarts from zero
    and can hit the same flake again. Entries are keyed by source text and
    invalidated whenever the model or the system prompt changes.
    """

    def __init__(self, path: Path, fingerprint: str) -> None:
        self.path = path
        self.fingerprint = fingerprint
        self._lock = threading.Lock()
        self._items = self._load()

    def _load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {}
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            log.warning("ignoring unreadable translation cache %s: %s", self.path, exc)
            return {}
        if not isinstance(raw, dict) or raw.get("fingerprint") != self.fingerprint:
            return {}
        items = raw.get("items")
        return items if isinstance(items, dict) else {}

    @staticmethod
    def key(text: str) -> str:
        return hashlib.sha1(text.encode("utf-8")).hexdigest()

    def get(self, text: str) -> TranslationItem | None:
        entry = self._items.get(self.key(text))
        if not isinstance(entry, dict):
            return None
        try:
            return TranslationItem.model_validate(entry)
        except ValidationError:
            return None

    def put(self, text: str, item: TranslationItem) -> None:
        with self._lock:
            self._items[self.key(text)] = item.model_dump()
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                self.path.write_text(
                    json.dumps(
                        {"fingerprint": self.fingerprint, "items": self._items},
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )
            except OSError as exc:
                log.warning("could not persist translation cache %s: %s", self.path, exc)

    @property
    def size(self) -> int:
        return len(self._items)


def translation_cache_path(session: Path, target_language: str) -> Path:
    return session / "metadata" / f"translation_cache.{target_language}.json"


def _cache_fingerprint(system: str, model: str) -> str:
    return hashlib.sha1(f"{model}\n{system}".encode("utf-8")).hexdigest()


def translate_batch(
    texts: list[str],
    source: SourceConfig,
    meta: dict[str, Any],
    pre: PreprocessResponse,
    *,
    base_url: str,
    api_key: str,
    model: str,
    concurrency: int = DEFAULT_CONCURRENCY,
    session: Path | None = None,
    source_url: str = "",
    meta_out: dict[str, Any] | None = None,
) -> list[TranslationItem]:
    if not texts:
        return []
    system = _translate_system(source, meta, pre)
    client = _client(base_url, api_key)
    cache = (
        TranslationCache(
            translation_cache_path(session, source.target_language),
            _cache_fingerprint(system, model),
        )
        if session is not None
        else None
    )
    log.info(
        "translate_batch: %d sentences, concurrency=%d, cached=%d",
        len(texts),
        concurrency,
        0 if cache is None else cache.size,
    )

    counter = {"done": 0}
    counter_lock = threading.Lock()

    def work(text: str) -> TranslationItem:
        if cache is not None:
            cached = cache.get(text)
            if cached is not None:
                with counter_lock:
                    counter["done"] += 1
                return cached
        item = translate_sentence(text, source.target_language, client, model, system)
        if cache is not None:
            cache.put(text, item)
        with counter_lock:
            counter["done"] += 1
        return item

    # --- Preferred path: one request for everything ---
    # Global context gives better terminology/name consistency, a single request
    # cannot drop connections the way a 50-way fan-out can, and the same call is
    # the only place that can judge ads and write the Bilibili copy with full
    # knowledge of the video. Only the sentences the bulk reply actually missed
    # are translated individually, so a truncated reply costs a few calls
    # instead of the whole stage.
    results: list[TranslationItem | None] = [None] * len(texts)
    missing: list[int] = []
    for index, text in enumerate(texts):
        cached = cache.get(text) if cache is not None else None
        if cached is not None:
            results[index] = cached
        else:
            missing.append(index)

    if missing:
        bulk_input = [texts[index] for index in missing]
        bulk = _coerce_bulk(
            translate_bulk(
                bulk_input,
                source,
                meta,
                pre,
                client=client,
                model=model,
                source_url=source_url,
            )
        )
        if bulk is not None:
            if not (bulk.title or bulk.description):
                # A long transcript sometimes makes the model translate every
                # line and silently drop the header sections. The copy is about
                # the whole video, so ask for it directly instead of shipping an
                # upload with no title. This is enrichment on top of a finished
                # translation, so a failure here must never fail the stage.
                try:
                    fallback = _copy_request(
                        _condense(bulk_input),
                        _bulk_copy_system(meta, source_url),
                        client=client,
                        model=model,
                    )
                except TranslationApiError:
                    raise
                except Exception as exc:  # noqa: BLE001 - copy is best-effort
                    log.warning("copy fallback failed, shipping without it: %s", exc)
                    fallback = None
                if fallback is not None:
                    bulk.title = fallback.title
                    bulk.description = fallback.description
                    bulk.tags = list(fallback.tags)
            if meta_out is not None:
                meta_out["title"] = bulk.title
                meta_out["description"] = bulk.description
                meta_out["tags"] = list(bulk.tags)
            ad_positions = bulk.merged_ad_lines(len(bulk_input))
            if ad_positions:
                log.info(
                    "bulk translate flagged %d of %d line(s) as ads",
                    len(ad_positions),
                    len(bulk_input),
                )
            for position, index in enumerate(missing, start=1):
                if position in ad_positions:
                    # Ads are cut from the video, so they keep no target text.
                    # Storing the flag in the cache keeps a re-run from asking
                    # about isolated ad sentences out of context.
                    item = _ad_translation_item()
                elif position in bulk.lines:
                    item = _fix_translation_item(bulk.lines[position])
                else:
                    continue
                results[index] = item
                if cache is not None:
                    cache.put(texts[index], item)

    pending = [index for index, item in enumerate(results) if item is None]
    if pending:
        log.info(
            "translate_batch: %d/%d sentence(s) not covered by the bulk reply; "
            "translating them individually",
            len(pending),
            len(texts),
        )
        try:
            with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
                filled = list(pool.map(work, [texts[index] for index in pending]))
        except TranslationApiError:
            raise
        except Exception as exc:  # noqa: BLE001
            done = sum(1 for item in results if item is not None)
            detail = (
                f"翻译接口网络异常：{exc}"
                if isinstance(exc, TransientApiError)
                else str(exc)
            )
            raise RuntimeError(
                f"{detail}（已完成 {done}/{len(texts)} 句，进度已缓存，重试会接着跑）"
            ) from exc
        for index, item in zip(pending, filled):
            results[index] = item

    return [item for item in results if item is not None]


def _read_meta(session: Path) -> dict[str, Any]:
    info_file = session / "metadata" / "ytdlp_info.json"
    if not info_file.exists():
        return {}
    return json.loads(info_file.read_text(encoding="utf-8"))


def _speaker(utt: dict[str, Any]) -> str:
    additions = utt.get("additions") or {}
    if isinstance(additions, dict):
        return str(additions.get("speaker") or "1")
    return "1"


def _full_text(data: dict[str, Any], texts: list[str]) -> str:
    raw = data.get("result", {}).get("text") or ""
    if raw.strip():
        return raw
    return " ".join(texts)


def preprocess_artifact_path(session: Path) -> Path:
    return session / "metadata" / "translation_preprocess.json"


def write_preprocess_artifact(session: Path, pre: PreprocessResponse) -> Path:
    path = preprocess_artifact_path(session)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(pre.model_dump(), ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_preprocess_artifact(session: Path) -> PreprocessResponse | None:
    path = preprocess_artifact_path(session)
    if not path.exists():
        return None
    return PreprocessResponse.model_validate(json.loads(path.read_text(encoding="utf-8")))


def _concurrency_from(settings: dict[str, str]) -> int:
    raw = str(settings.get("translate_concurrency") or "").strip()
    if not raw or not all("0" <= char <= "9" for char in raw):
        return DEFAULT_CONCURRENCY
    concurrency = int(raw)
    if concurrency < 1 or concurrency > 200:
        return DEFAULT_CONCURRENCY
    return concurrency


def bilibili_meta_path(session: Path) -> Path:
    return session / "metadata" / "bilibili_meta.json"


def load_bilibili_meta(session: Path) -> dict[str, Any] | None:
    path = bilibili_meta_path(session)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        log.warning("ignoring unreadable Bilibili metadata %s: %s", path, exc)
        return None
    return data if isinstance(data, dict) else None


def write_bilibili_meta(
    session: Path,
    *,
    title: str = "",
    description: str = "",
    tags: list[str] | None = None,
    source_url: str = "",
) -> Path | None:
    """Persist the upload copy for the Bilibili release.

    The source link is non-negotiable: the model is asked to include it, but a
    description that forgets it would break the attribution requirement, so it
    is appended when missing.
    """
    title = (title or "").strip()
    description = (description or "").strip()
    if not title and not description:
        return None
    link = (source_url or "").strip()
    if link and link not in description:
        description = f"{description}\n\n原视频：{link}".strip()
    path = bilibili_meta_path(session)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "title": title,
                "description": description,
                "tags": list(tags or []),
                "source_url": link,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return path


def translate_asr(
    asr_file: Path,
    session: Path,
    settings: dict[str, str],
    source: SourceConfig,
    *,
    source_url: str = "",
) -> Path:
    output_file = session / "metadata" / f"translation.{source.target_language}.json"
    if output_file.exists():
        return output_file

    data = json.loads(asr_file.read_text(encoding="utf-8"))
    utterances = data["result"]["utterances"]
    texts = [u["text"].strip() for u in utterances]
    full_text = _full_text(data, texts)
    meta = _read_meta(session)
    link = (source_url or str(meta.get("webpage_url") or "")).strip()

    api = {key: settings[key] for key in API_SETTING_KEYS if key in settings}
    pre = load_preprocess_artifact(session)
    if pre is None:
        pre = preprocess(full_text, meta, source, **api)
        write_preprocess_artifact(session, pre)
        log.info("Wrote translation preprocess artifact to %s", preprocess_artifact_path(session))
    else:
        log.info("Reusing translation preprocess artifact from %s", preprocess_artifact_path(session))
    meta_out: dict[str, Any] = {}
    translated_items = translate_batch(
        texts,
        source,
        meta,
        pre,
        **api,
        concurrency=_concurrency_from(settings),
        session=session,
        source_url=link,
        meta_out=meta_out,
    )
    if meta_out.get("title") or meta_out.get("description"):
        written = write_bilibili_meta(
            session,
            title=meta_out.get("title", ""),
            description=meta_out.get("description", ""),
            tags=meta_out.get("tags") or [],
            source_url=link,
        )
        if written is not None:
            log.info("Wrote Bilibili upload copy to %s", written)

    translation: list[dict[str, Any]] = []
    for text, translated, utt in zip(texts, translated_items, utterances):
        entry: dict[str, Any] = {
            "src": text,
            "dst": translated.dst,
            "audio_mode": translated.audio_mode,
            "src_lang": source.asr_language,
            "dst_lang": source.target_language,
            "start_time": utt["start_time"],
            "end_time": utt["end_time"],
            "speaker": _speaker(utt),
        }
        if translated.is_ad:
            # Flagged lines are removed (and cut out of the video) by ad_cut.
            entry["is_ad"] = True
        translation.append(entry)
    output_file.write_text(
        json.dumps({"translation": translation}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output_file
