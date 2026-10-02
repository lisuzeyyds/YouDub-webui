from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
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

    @model_validator(mode="after")
    def validate_tts_text(self) -> "TranslationItem":
        if self.audio_mode == "tts" and not self.dst.strip():
            raise ValueError("dst must be non-empty when audio_mode is tts")
        return self


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

    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        try:
            return list(pool.map(work, texts))
        except TranslationApiError:
            raise
        except Exception as exc:  # noqa: BLE001
            done = counter["done"]
            detail = (
                f"翻译接口网络异常：{exc}"
                if isinstance(exc, TransientApiError)
                else str(exc)
            )
            raise RuntimeError(
                f"{detail}（已完成 {done}/{len(texts)} 句，进度已缓存，重试会接着跑）"
            ) from exc


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


def translate_asr(
    asr_file: Path,
    session: Path,
    settings: dict[str, str],
    source: SourceConfig,
) -> Path:
    output_file = session / "metadata" / f"translation.{source.target_language}.json"
    if output_file.exists():
        return output_file

    data = json.loads(asr_file.read_text(encoding="utf-8"))
    utterances = data["result"]["utterances"]
    texts = [u["text"].strip() for u in utterances]
    full_text = _full_text(data, texts)
    meta = _read_meta(session)

    api = {key: settings[key] for key in API_SETTING_KEYS if key in settings}
    pre = load_preprocess_artifact(session)
    if pre is None:
        pre = preprocess(full_text, meta, source, **api)
        write_preprocess_artifact(session, pre)
        log.info("Wrote translation preprocess artifact to %s", preprocess_artifact_path(session))
    else:
        log.info("Reusing translation preprocess artifact from %s", preprocess_artifact_path(session))
    translated_items = translate_batch(
        texts,
        source,
        meta,
        pre,
        **api,
        concurrency=_concurrency_from(settings),
        session=session,
    )

    translation = [
        {
            "src": text,
            "dst": translated.dst,
            "audio_mode": translated.audio_mode,
            "src_lang": source.asr_language,
            "dst_lang": source.target_language,
            "start_time": utt["start_time"],
            "end_time": utt["end_time"],
            "speaker": _speaker(utt),
        }
        for text, translated, utt in zip(texts, translated_items, utterances)
    ]
    output_file.write_text(
        json.dumps({"translation": translation}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output_file
