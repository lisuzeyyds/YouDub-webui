from __future__ import annotations

import json

import httpx
import pytest
from openai import (
    APIConnectionError,
    AuthenticationError,
    InternalServerError,
    NotFoundError,
    PermissionDeniedError,
    RateLimitError,
)

from backend.app.adapters import openai_translate
from backend.app.adapters.openai_translate import (
    HotwordItem,
    PreprocessResponse,
    CorrectionItem,
    TransientApiError,
    TranslationApiError,
)
from backend.app.sources import detect_source


YT_SOURCE = detect_source("https://www.youtube.com/watch?v=abcdefghijk")
BB_SOURCE = detect_source("https://www.bilibili.com/video/BV1xx411c7mD")
JA_SOURCE = detect_source("local://upload/japanese-task?direction=ja-zh")


_STATUS_EXCEPTIONS = {
    401: AuthenticationError,
    403: PermissionDeniedError,
    404: NotFoundError,
    429: RateLimitError,
    500: InternalServerError,
}


@pytest.fixture(autouse=True)
def _disable_bulk_translation(monkeypatch):
    """Keep the per-sentence tests on the per-sentence path.

    translate_batch prefers one request for the whole transcript; these tests
    stub _call_json/_client for the single-sentence protocol, so the bulk path
    is disabled here. Tests that exercise bulk restore the real implementation
    with `_enable_bulk(monkeypatch)`.
    """
    _real_translate_bulk = openai_translate.translate_bulk
    monkeypatch.setattr(
        openai_translate, "translate_bulk", lambda *a, **kw: None
    )
    monkeypatch.setattr(
        openai_translate, "_REAL_TRANSLATE_BULK", _real_translate_bulk, raising=False
    )


def _enable_bulk(monkeypatch) -> None:
    """Restore the real bulk implementation (undone by the autouse fixture)."""
    monkeypatch.setattr(
        openai_translate,
        "translate_bulk",
        openai_translate._REAL_TRANSLATE_BULK,
    )


def _status_error(status: int, payload: dict):
    request = httpx.Request("POST", "https://example.com/v1/chat/completions")
    response = httpx.Response(status, request=request, json=payload)
    message = f"Error code: {status} - {json.dumps(payload, ensure_ascii=False)}"
    cls = _STATUS_EXCEPTIONS.get(status, InternalServerError)
    return cls(message, response=response, body=None)


def _connection_error():
    request = httpx.Request("POST", "https://example.com/v1/chat/completions")
    return APIConnectionError(request=request)


class _FakeCompletions:
    def __init__(self, behaviour):
        self._behaviour = behaviour

    def create(self, **kwargs):
        return self._behaviour(kwargs)


class _FakeClient:
    def __init__(self, behaviour):
        self.chat = type("c", (), {"completions": _FakeCompletions(behaviour)})()


def _write_asr(path, n: int, full_text: str | None = None) -> None:
    utterances = [
        {"text": f"S{i}.", "start_time": i * 1000, "end_time": (i + 1) * 1000}
        for i in range(n)
    ]
    payload = {"result": {"utterances": utterances, "text": full_text or " ".join(u["text"] for u in utterances)}}
    path.write_text(json.dumps(payload), encoding="utf-8")


def _settings() -> dict[str, str]:
    return {"base_url": "https://example.com/v1", "api_key": "sk-test", "model": "model-x"}


def _stub_preprocess(monkeypatch, response: PreprocessResponse | None = None):
    seen: list[dict] = []

    def fake(full_text, meta, source, **kw):
        seen.append({"full_text": full_text, "meta": meta, "source": source, **kw})
        return response or PreprocessResponse()

    monkeypatch.setattr(openai_translate, "preprocess", fake)
    return seen


def _stub_translate_batch(monkeypatch, transform):
    seen: list[dict] = []

    def fake(texts, source, meta, pre, **kw):
        seen.append({"texts": list(texts), "source": source, "meta": meta, "pre": pre, **kw})
        return [
            openai_translate.TranslationItem(dst=transform(t), audio_mode="tts")
            for t in texts
        ]

    monkeypatch.setattr(openai_translate, "translate_batch", fake)
    return seen


def test_translate_asr_writes_preprocess_artifact(tmp_path, monkeypatch):
    metadata = tmp_path / "metadata"
    metadata.mkdir()
    asr_file = metadata / "asr_fixed.json"
    _write_asr(asr_file, 1)

    pre = PreprocessResponse(
        summary="Video recap",
        hotwords=[HotwordItem(src="Fable 5", dst="Fable 5")],
        corrections=[CorrectionItem(wrong="java script", correct="JavaScript")],
    )
    monkeypatch.setattr(openai_translate, "preprocess", lambda *a, **kw: pre)
    _stub_translate_batch(monkeypatch, lambda t: f"zh:{t}")

    openai_translate.translate_asr(asr_file, tmp_path, _settings(), YT_SOURCE)
    artifact = metadata / "translation_preprocess.json"
    assert artifact.exists()
    saved = json.loads(artifact.read_text(encoding="utf-8"))
    assert saved["summary"] == "Video recap"
    assert saved["hotwords"][0]["src"] == "Fable 5"
    assert saved["corrections"][0]["correct"] == "JavaScript"


def test_translate_asr_reuses_preprocess_artifact_without_calling_api(tmp_path, monkeypatch):
    metadata = tmp_path / "metadata"
    metadata.mkdir()
    asr_file = metadata / "asr_fixed.json"
    _write_asr(asr_file, 1)
    (metadata / "translation_preprocess.json").write_text(
        json.dumps(
            {
                "summary": "cached",
                "hotwords": [{"src": "GPU", "dst": "GPU"}],
                "corrections": [],
            }
        ),
        encoding="utf-8",
    )

    def fail_preprocess(*args, **kwargs):
        raise AssertionError("preprocess should not run when artifact exists")

    monkeypatch.setattr(openai_translate, "preprocess", fail_preprocess)
    seen = _stub_translate_batch(monkeypatch, lambda t: f"zh:{t}")

    openai_translate.translate_asr(asr_file, tmp_path, _settings(), YT_SOURCE)
    assert len(seen) == 1
    assert seen[0]["pre"].summary == "cached"


def test_translate_asr_writes_schema_with_speaker_and_lang(tmp_path, monkeypatch):
    metadata = tmp_path / "metadata"
    metadata.mkdir()
    asr_file = metadata / "asr.json"
    _write_asr(asr_file, 2)

    _stub_preprocess(monkeypatch)
    _stub_translate_batch(monkeypatch, lambda t: f"zh:{t}")

    out = openai_translate.translate_asr(asr_file, tmp_path, _settings(), YT_SOURCE)
    items = json.loads(out.read_text(encoding="utf-8"))["translation"]
    assert [i["dst"] for i in items] == ["zh:S0.", "zh:S1."]
    assert {i["audio_mode"] for i in items} == {"tts"}
    assert {i["src_lang"] for i in items} == {"en"}
    assert {i["dst_lang"] for i in items} == {"zh"}
    assert {i["speaker"] for i in items} == {"1"}
    assert items[0]["start_time"] == 0


def test_translate_asr_writes_japanese_to_chinese_language_metadata(tmp_path, monkeypatch):
    metadata = tmp_path / "metadata"
    metadata.mkdir()
    asr_file = metadata / "asr.json"
    _write_asr(asr_file, 1, full_text="今日はいい天気です。")

    _stub_preprocess(monkeypatch)
    _stub_translate_batch(monkeypatch, lambda _text: "今天天气很好。")

    out = openai_translate.translate_asr(asr_file, tmp_path, _settings(), JA_SOURCE)
    item = json.loads(out.read_text(encoding="utf-8"))["translation"][0]

    assert out.name == "translation.zh.json"
    assert item["src_lang"] == "ja"
    assert item["dst_lang"] == "zh"
    assert item["dst"] == "今天天气很好。"
    assert item["audio_mode"] == "tts"


def test_translate_asr_output_filename_uses_target_lang(tmp_path, monkeypatch):
    metadata = tmp_path / "metadata"
    metadata.mkdir()
    asr_file = metadata / "asr.json"
    _write_asr(asr_file, 1)

    _stub_preprocess(monkeypatch)
    _stub_translate_batch(monkeypatch, lambda _t: "x")

    out = openai_translate.translate_asr(asr_file, tmp_path, _settings(), BB_SOURCE)
    assert out.name == "translation.en.json"


def test_translate_asr_passes_meta_and_full_text_to_preprocess(tmp_path, monkeypatch):
    metadata = tmp_path / "metadata"
    metadata.mkdir()
    asr_file = metadata / "asr.json"
    _write_asr(asr_file, 1, full_text="hello world")
    (metadata / "ytdlp_info.json").write_text(
        json.dumps({"title": "T", "uploader": "U", "description": "D"}),
        encoding="utf-8",
    )

    seen = _stub_preprocess(monkeypatch)
    _stub_translate_batch(monkeypatch, lambda t: t)

    openai_translate.translate_asr(asr_file, tmp_path, _settings(), YT_SOURCE)
    assert seen[0]["full_text"] == "hello world"
    assert seen[0]["meta"] == {"title": "T", "uploader": "U", "description": "D"}


def test_translate_asr_invokes_translate_batch_with_all_texts_at_once(tmp_path, monkeypatch):
    metadata = tmp_path / "metadata"
    metadata.mkdir()
    asr_file = metadata / "asr.json"
    _write_asr(asr_file, 5)

    _stub_preprocess(monkeypatch, PreprocessResponse(hotwords=[HotwordItem(src="x", dst="y")]))
    seen = _stub_translate_batch(monkeypatch, lambda t: f"zh:{t}")

    openai_translate.translate_asr(asr_file, tmp_path, _settings(), YT_SOURCE)
    assert len(seen) == 1
    assert seen[0]["texts"] == ["S0.", "S1.", "S2.", "S3.", "S4."]
    assert seen[0]["pre"].hotwords[0].src == "x"


def test_translate_batch_replaces_em_dash_for_zh_target(monkeypatch):
    monkeypatch.setattr(
        openai_translate,
        "_call_json",
        lambda *a, **kw: {"dst": "你好——世界", "audio_mode": "tts"},
    )
    monkeypatch.setattr(openai_translate, "_client", lambda *a, **kw: object())

    out = openai_translate.translate_batch(
        ["Hello world."], YT_SOURCE, {}, PreprocessResponse(),
        base_url="u", api_key="k", model="m",
    )
    assert [item.model_dump() for item in out] == [
        {"dst": "你好，世界", "audio_mode": "tts", "is_ad": False}
    ]


def test_translate_batch_does_not_replace_em_dash_for_en_target(monkeypatch):
    monkeypatch.setattr(
        openai_translate,
        "_call_json",
        lambda *a, **kw: {
            "dst": "He said—wait—and left.",
            "audio_mode": "tts",
        },
    )
    monkeypatch.setattr(openai_translate, "_client", lambda *a, **kw: object())

    out = openai_translate.translate_batch(
        ["他说——等等——就走了。"], BB_SOURCE, {}, PreprocessResponse(),
        base_url="u", api_key="k", model="m",
    )
    assert [item.model_dump() for item in out] == [
        {"dst": "He said—wait—and left.", "audio_mode": "tts", "is_ad": False}
    ]


def test_translate_batch_uses_shared_system_prompt(monkeypatch):
    captured: list[str] = []
    lock = __import__("threading").Lock()

    def fake_call_json(client, model, system, user):
        with lock:
            captured.append(system)
        return {"dst": f"dst:{user}", "audio_mode": "tts"}

    monkeypatch.setattr(openai_translate, "_call_json", fake_call_json)
    monkeypatch.setattr(openai_translate, "_client", lambda *a, **kw: object())

    texts = [f"s{i}" for i in range(5)]
    out = openai_translate.translate_batch(
        texts, BB_SOURCE, {}, PreprocessResponse(),
        base_url="u", api_key="k", model="m", concurrency=4,
    )
    assert [item.dst for item in out] == [f"dst:s{i}" for i in range(5)]
    assert {item.audio_mode for item in out} == {"tts"}
    assert len(set(captured)) == 1, "system prompt must be identical across calls for prompt cache"


@pytest.mark.parametrize("value", ["abc", "1.5", "0", "-1", "201", ""])
def test_concurrency_from_bad_saved_values_falls_back_to_default(value):
    assert openai_translate._concurrency_from({"translate_concurrency": value}) == 50


def test_translate_sentence_accepts_empty_dst_for_filler_only_sentence(monkeypatch):
    calls = {"n": 0}

    def fake_call_json(client, model, system, user):
        calls["n"] += 1
        return {"dst": "", "audio_mode": "original"}

    monkeypatch.setattr(openai_translate, "_call_json", fake_call_json)

    out = openai_translate.translate_sentence("um", "en", object(), "m", "sys")
    assert out.dst == ""
    assert out.audio_mode == "original"
    assert calls["n"] == 1


def test_translate_sentence_keeps_mixed_speech_in_tts_mode(monkeypatch):
    monkeypatch.setattr(
        openai_translate,
        "_call_json",
        lambda *args, **kwargs: {"dst": "救命！", "audio_mode": "tts"},
    )

    out = openai_translate.translate_sentence(
        "[laughs] Help!", "zh", object(), "m", "sys"
    )

    assert out.model_dump() == {"dst": "救命！", "audio_mode": "tts", "is_ad": False}


@pytest.mark.parametrize(
    "response",
    [
        {"dst": "Hello"},
        {"dst": "Hello", "audio_mode": "invalid"},
        {"dst": "", "audio_mode": "tts"},
    ],
)
def test_translate_sentence_rejects_invalid_audio_mode_contract(monkeypatch, response):
    monkeypatch.setattr(openai_translate, "_call_json", lambda *args, **kwargs: response)

    with pytest.raises(RuntimeError, match="translate_sentence failed"):
        openai_translate.translate_sentence("x", "en", object(), "m", "sys")


def test_translate_sentence_raises_after_retries(monkeypatch):
    def fake_call_json(client, model, system, user):
        raise ValueError("boom")

    monkeypatch.setattr(openai_translate, "_call_json", fake_call_json)

    with pytest.raises(RuntimeError, match="translate_sentence failed"):
        openai_translate.translate_sentence("x", "en", object(), "m", "sys")


def test_preprocess_returns_empty_when_repeatedly_invalid(monkeypatch):
    def fake_call_json(client, model, system, user):
        return {"summary": 123, "hotwords": "bad"}

    monkeypatch.setattr(openai_translate, "_call_json", fake_call_json)
    monkeypatch.setattr(openai_translate, "_client", lambda *a, **kw: object())

    pre = openai_translate.preprocess(
        "text", {"title": "t"}, YT_SOURCE,
        base_url="u", api_key="k", model="m",
    )
    assert pre.summary == ""
    assert pre.hotwords == []
    assert pre.corrections == []


def _no_sleep(monkeypatch):
    monkeypatch.setattr(openai_translate.time, "sleep", lambda *_: None)


# --- transport failure handling -------------------------------------------------


def test_call_json_maps_quota_error_to_actionable_message():
    exc = _status_error(
        403,
        {"error": {"message": "余额不足", "code": "insufficient_user_quota"}},
    )
    client = _FakeClient(lambda kwargs: (_ for _ in ()).throw(exc))

    with pytest.raises(TranslationApiError) as info:
        openai_translate._call_json(client, "m", "sys", "user")

    message = str(info.value)
    assert "额度不足" in message
    assert "充值" in message


def test_call_json_marks_connection_drop_as_transient():
    client = _FakeClient(lambda kwargs: (_ for _ in ()).throw(_connection_error()))

    with pytest.raises(TransientApiError):
        openai_translate._call_json(client, "m", "sys", "user")


@pytest.mark.parametrize("status", [429, 500, 503])
def test_call_json_marks_rate_limit_and_server_errors_as_transient(status):
    exc = _status_error(status, {"error": {"message": "busy"}})
    client = _FakeClient(lambda kwargs: (_ for _ in ()).throw(exc))

    with pytest.raises(TransientApiError):
        openai_translate._call_json(client, "m", "sys", "user")


def test_translate_sentence_retries_transient_failures(monkeypatch):
    _no_sleep(monkeypatch)
    calls = {"n": 0}

    def fake_call_json(client, model, system, user):
        calls["n"] += 1
        if calls["n"] < 3:
            raise TransientApiError("Server disconnected without sending a response.")
        return {"dst": "你好。", "audio_mode": "tts"}

    monkeypatch.setattr(openai_translate, "_call_json", fake_call_json)

    out = openai_translate.translate_sentence("Hello.", "zh", object(), "m", "sys")

    assert out.model_dump() == {"dst": "你好。", "audio_mode": "tts", "is_ad": False}
    assert calls["n"] == 3


def test_translate_sentence_gives_up_after_transport_attempts(monkeypatch):
    _no_sleep(monkeypatch)
    calls = {"n": 0}

    def fake_call_json(client, model, system, user):
        calls["n"] += 1
        raise TransientApiError("Server disconnected without sending a response.")

    monkeypatch.setattr(openai_translate, "_call_json", fake_call_json)

    with pytest.raises(RuntimeError, match="translate_sentence failed after 4 attempts"):
        openai_translate.translate_sentence("Hello.", "zh", object(), "m", "sys")

    assert calls["n"] == 4


def test_translate_sentence_does_not_retry_quota_errors(monkeypatch):
    _no_sleep(monkeypatch)
    calls = {"n": 0}

    def fake_call_json(client, model, system, user):
        calls["n"] += 1
        raise TranslationApiError("翻译接口账户额度不足")

    monkeypatch.setattr(openai_translate, "_call_json", fake_call_json)

    with pytest.raises(TranslationApiError):
        openai_translate.translate_sentence("Hello.", "zh", object(), "m", "sys")

    assert calls["n"] == 1


# --- per-sentence cache ---------------------------------------------------------


def _batch(session=None, **kwargs):
    return openai_translate.translate_batch(
        ["S0.", "S1.", "S2."],
        YT_SOURCE,
        {},
        PreprocessResponse(),
        base_url="u",
        api_key="k",
        model="m",
        concurrency=1,
        session=session,
        **kwargs,
    )


def test_translate_batch_writes_per_sentence_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(openai_translate, "_client", lambda *a, **kw: object())
    monkeypatch.setattr(
        openai_translate,
        "_call_json",
        lambda client, model, system, user: {"dst": f"zh:{user}", "audio_mode": "tts"},
    )

    _batch(tmp_path)

    cached = json.loads(
        openai_translate.translation_cache_path(tmp_path, "zh").read_text(encoding="utf-8")
    )
    assert len(cached["items"]) == 3
    assert cached["fingerprint"]


def test_translate_batch_reuses_cache_without_calling_api(tmp_path, monkeypatch):
    monkeypatch.setattr(openai_translate, "_client", lambda *a, **kw: object())
    monkeypatch.setattr(
        openai_translate,
        "_call_json",
        lambda client, model, system, user: {"dst": f"zh:{user}", "audio_mode": "tts"},
    )
    _batch(tmp_path)

    def fail_call_json(*args, **kwargs):
        raise AssertionError("cached sentences must not hit the API again")

    monkeypatch.setattr(openai_translate, "_call_json", fail_call_json)
    out = _batch(tmp_path)

    assert [item.dst for item in out] == ["zh:S0.", "zh:S1.", "zh:S2."]


def test_translate_batch_keeps_finished_sentences_after_a_drop(tmp_path, monkeypatch):
    _no_sleep(monkeypatch)
    monkeypatch.setattr(openai_translate, "_client", lambda *a, **kw: object())
    seen: list[str] = []

    def flaky(client, model, system, user):
        seen.append(user)
        if user == "S1.":
            raise TransientApiError("Server disconnected without sending a response.")
        return {"dst": f"zh:{user}", "audio_mode": "tts"}

    monkeypatch.setattr(openai_translate, "_call_json", flaky)

    with pytest.raises(RuntimeError, match="进度已缓存"):
        _batch(tmp_path)

    cached = json.loads(
        openai_translate.translation_cache_path(tmp_path, "zh").read_text(encoding="utf-8")
    )
    # S0 and S2 completed; S1 failed and must be retried on the next run.
    assert len(cached["items"]) == 2

    seen.clear()
    monkeypatch.setattr(
        openai_translate,
        "_call_json",
        lambda client, model, system, user: (
            seen.append(user) or {"dst": f"zh:{user}", "audio_mode": "tts"}
        ),
    )
    out = _batch(tmp_path)

    assert [item.dst for item in out] == ["zh:S0.", "zh:S1.", "zh:S2."]
    assert seen == ["S1."], "only the missing sentence should be translated again"


def test_translation_cache_is_ignored_when_the_prompt_changes(tmp_path, monkeypatch):
    monkeypatch.setattr(openai_translate, "_client", lambda *a, **kw: object())
    monkeypatch.setattr(
        openai_translate,
        "_call_json",
        lambda client, model, system, user: {"dst": "旧译文", "audio_mode": "tts"},
    )
    _batch(tmp_path)

    seen: list[str] = []
    monkeypatch.setattr(
        openai_translate,
        "_call_json",
        lambda client, model, system, user: (
            seen.append(user) or {"dst": "新译文", "audio_mode": "tts"}
        ),
    )
    out = openai_translate.translate_batch(
        ["S0.", "S1.", "S2."],
        YT_SOURCE,
        {},
        PreprocessResponse(summary="a different preprocess summary"),
        base_url="u",
        api_key="k",
        model="m",
        concurrency=1,
        session=tmp_path,
    )

    assert [item.dst for item in out] == ["新译文"] * 3
    assert len(seen) == 3


def test_translate_system_prompt_contains_meta_summary_hotwords(monkeypatch):
    pre = PreprocessResponse(
        summary="Recap of the talk.",
        hotwords=[HotwordItem(src="LEGO", dst="乐高")],
    )
    meta = {"title": "Demo", "uploader": "Alice", "description": "Long description"}
    system = openai_translate._translate_system(YT_SOURCE, meta, pre)
    assert "Demo" in system
    assert "Alice" in system
    assert "Long description" in system
    assert "Recap of the talk." in system
    assert "Content-only translation priority" in system
    assert "speech filler" in system
    assert "screams, laughter, crying, sobbing, moans" in system
    assert "If any translatable speech is present" in system
    assert "呼救、喊话及感叹使用 tts" in system
    assert "动物叫声和用力声" in system
    assert '"audio_mode": "tts 或 original"' in system
    assert "非常短的语气词" not in system
    assert "LEGO -> 乐高" in system


def test_japanese_to_chinese_prompt_and_audio_mode_response_contract(monkeypatch):
    captured: list[str] = []

    def fake_call_json(client, model, system, user):
        captured.append(user)
        return {"summary": "摘要", "hotwords": [], "corrections": []}

    monkeypatch.setattr(openai_translate, "_call_json", fake_call_json)
    monkeypatch.setattr(openai_translate, "_client", lambda *a, **kw: object())

    openai_translate.preprocess(
        "今日はいい天気です。",
        {},
        JA_SOURCE,
        base_url="u",
        api_key="k",
        model="m",
    )
    system = openai_translate._translate_system(JA_SOURCE, {}, PreprocessResponse())

    assert JA_SOURCE.asr_language_name == "Japanese"
    assert JA_SOURCE.target_language_name == "Simplified Chinese"
    assert "转录原始语言：Japanese" in captured[0]
    assert "请将日文逐句翻译" in system
    assert "一句日文原文" in system
    assert "一句英文原文" not in system
    assert '"audio_mode": "tts 或 original"' in system
    assert "只有非语言人声时使用 original" in system
    assert "Content-only translation priority" in system

    captured_systems: list[str] = []

    def valid_translation(client, model, current_system, user):
        captured_systems.append(current_system)
        return {"dst": "今天天气很好。", "audio_mode": "tts"}

    monkeypatch.setattr(openai_translate, "_call_json", valid_translation)
    translated = openai_translate.translate_batch(
        ["今日はいい天気です。"],
        JA_SOURCE,
        {},
        PreprocessResponse(),
        base_url="u",
        api_key="k",
        model="m",
        concurrency=1,
    )

    assert [item.model_dump() for item in translated] == [
        {"dst": "今天天气很好。", "audio_mode": "tts", "is_ad": False}
    ]
    assert "请将日文逐句翻译" in captured_systems[0]
    assert '"audio_mode": "tts 或 original"' in captured_systems[0]
    assert "Content-only translation priority" in captured_systems[0]

    monkeypatch.setattr(
        openai_translate,
        "_call_json",
        lambda *args, **kwargs: {"dst": "今天天气很好。"},
    )
    with pytest.raises(RuntimeError, match="translate_sentence failed"):
        openai_translate.translate_batch(
            ["今日はいい天気です。"],
            JA_SOURCE,
            {},
            PreprocessResponse(),
            base_url="u",
            api_key="k",
            model="m",
            concurrency=1,
        )


# --------------------------------------------------------------------------
# Bulk (whole-transcript) translation
# --------------------------------------------------------------------------


def test_bulk_payload_numbers_every_sentence():
    payload = openai_translate._bulk_payload(["Hello.", "World."])
    assert payload == "1\tHello.\n2\tWorld."


def test_parse_bulk_reply_ignores_preamble_and_bad_lines():
    raw = (
        "Here are the translations:/n"
        "```\n"
        "1\t你好。\n"
        "2\t世界。\n"
        "not a numbered line\n"
        "4\t多余的编号\n"
    )
    parsed = openai_translate._parse_bulk_reply(raw, expected=3)
    assert parsed == {1: "你好。", 2: "世界。"}


def test_parse_bulk_reply_keeps_a_truncated_tail():
    raw = "1\t你好。\n2\t世"
    parsed = openai_translate._parse_bulk_reply(raw, expected=5)
    assert parsed == {1: "你好。", 2: "世"}


def test_parse_bulk_reply_accepts_colon_and_dot_separators():
    parsed = openai_translate._parse_bulk_reply("1: 你好。\n2. 世界。", expected=2)
    assert parsed == {1: "你好。", 2: "世界。"}


def test_bulk_line_without_text_keeps_the_source_audio():
    assert openai_translate._fix_translation_item("   ").model_dump() == {
        "dst": "",
        "audio_mode": "original",
        "is_ad": False,
    }
    assert openai_translate._fix_translation_item("你好。").model_dump() == {
        "dst": "你好。",
        "audio_mode": "tts",
        "is_ad": False,
    }
    assert openai_translate._ad_translation_item().model_dump() == {
        "dst": "",
        "audio_mode": "original",
        "is_ad": True,
    }


def test_translate_bulk_returns_none_when_payload_is_too_long(monkeypatch):
    _enable_bulk(monkeypatch)
    monkeypatch.setattr(openai_translate, "BULK_MAX_CHARS", 10)
    called = {"n": 0}

    def fake_bulk(*args, **kwargs):
        called["n"] += 1
        return "1\tx"

    monkeypatch.setattr(openai_translate, "_call_bulk", fake_bulk)
    out = openai_translate.translate_bulk(
        ["a very long sentence that blows the budget"], YT_SOURCE, {}, PreprocessResponse(),
        client=object(), model="m",
    )
    assert out is None
    assert called["n"] == 0


def test_translate_bulk_parses_a_single_request(monkeypatch):
    _enable_bulk(monkeypatch)
    monkeypatch.setattr(
        openai_translate, "_call_bulk", lambda *a, **kw: "1\t你好。\n2\t世界。"
    )
    out = openai_translate.translate_bulk(
        ["Hello.", "World."], YT_SOURCE, {}, PreprocessResponse(),
        client=object(), model="m",
    )
    assert out is not None
    assert out.lines == {1: "你好。", 2: "世界。"}
    assert out.ad_lines == set()
    assert out.title == ""


def test_translate_bulk_retries_transport_failures(monkeypatch):
    _enable_bulk(monkeypatch)
    monkeypatch.setattr(openai_translate.time, "sleep", lambda *_: None)
    calls = {"n": 0}

    def flaky(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise TransientApiError("boom")
        return "1\t你好。"

    monkeypatch.setattr(openai_translate, "_call_bulk", flaky)
    out = openai_translate.translate_bulk(
        ["Hello."], YT_SOURCE, {}, PreprocessResponse(), client=object(), model="m",
    )
    assert out is not None
    assert out.lines == {1: "你好。"}
    assert calls["n"] == 2


def test_translate_bulk_propagates_deterministic_errors(monkeypatch):
    _enable_bulk(monkeypatch)

    def fatal(*args, **kwargs):
        raise TranslationApiError("余额不足")

    monkeypatch.setattr(openai_translate, "_call_bulk", fatal)
    with pytest.raises(TranslationApiError, match="余额不足"):
        openai_translate.translate_bulk(
            ["Hello."], YT_SOURCE, {}, PreprocessResponse(), client=object(), model="m",
        )


def test_translate_batch_prefers_one_bulk_request(monkeypatch):
    """The whole transcript goes out in a single call, no per-sentence fan-out."""
    bulk_calls: list[list[str]] = []
    single_calls: list[str] = []

    def fake_bulk(texts, source, meta, pre, *, client, model, source_url=""):
        bulk_calls.append(list(texts))
        return {i: f"zh{i}" for i in range(1, len(texts) + 1)}

    def fake_sentence(text, *a, **kw):
        single_calls.append(text)
        return openai_translate.TranslationItem(dst=f"single:{text}", audio_mode="tts")

    monkeypatch.setattr(openai_translate, "translate_bulk", fake_bulk)
    monkeypatch.setattr(openai_translate, "_client", lambda *a, **kw: object())
    monkeypatch.setattr(openai_translate, "_call_json", lambda *a, **kw: {})
    monkeypatch.setattr(openai_translate, "translate_sentence", fake_sentence)

    texts = [f"s{i}" for i in range(6)]
    out = openai_translate.translate_batch(
        texts, BB_SOURCE, {}, PreprocessResponse(),
        base_url="u", api_key="k", model="m", concurrency=4,
    )
    assert [item.dst for item in out] == [f"zh{i}" for i in range(1, 7)]
    assert bulk_calls == [texts]
    assert single_calls == []


def test_translate_batch_tops_up_only_missing_bulk_lines(monkeypatch):
    """A truncated bulk reply only costs the lines it actually missed."""
    def fake_bulk(texts, source, meta, pre, *, client, model, source_url=""):
        return {1: "zh1", 3: "zh3"}

    single_calls: list[str] = []

    def fake_sentence(text, *a, **kw):
        single_calls.append(text)
        return openai_translate.TranslationItem(dst=f"single:{text}", audio_mode="tts")

    monkeypatch.setattr(openai_translate, "translate_bulk", fake_bulk)
    monkeypatch.setattr(openai_translate, "_client", lambda *a, **kw: object())
    monkeypatch.setattr(openai_translate, "translate_sentence", fake_sentence)

    out = openai_translate.translate_batch(
        ["a", "b", "c"], BB_SOURCE, {}, PreprocessResponse(),
        base_url="u", api_key="k", model="m", concurrency=3,
    )
    assert [item.dst for item in out] == ["zh1", "single:b", "zh3"]
    assert single_calls == ["b"]


def test_translate_batch_falls_back_entirely_when_bulk_is_unavailable(monkeypatch):
    def fake_sentence(text, *a, **kw):
        return openai_translate.TranslationItem(dst=f"single:{text}", audio_mode="tts")

    monkeypatch.setattr(openai_translate, "translate_bulk", lambda *a, **kw: None)
    monkeypatch.setattr(openai_translate, "_client", lambda *a, **kw: object())
    monkeypatch.setattr(openai_translate, "translate_sentence", fake_sentence)

    out = openai_translate.translate_batch(
        ["a", "b"], BB_SOURCE, {}, PreprocessResponse(),
        base_url="u", api_key="k", model="m", concurrency=2,
    )
    assert [item.dst for item in out] == ["single:a", "single:b"]


def test_translate_batch_caches_bulk_results(monkeypatch, tmp_path):
    def fake_bulk(texts, source, meta, pre, *, client, model, source_url=""):
        return {i: f"zh{i}" for i in range(1, len(texts) + 1)}

    monkeypatch.setattr(openai_translate, "translate_bulk", fake_bulk)
    monkeypatch.setattr(openai_translate, "_client", lambda *a, **kw: object())
    monkeypatch.setattr(
        openai_translate, "translate_sentence", lambda *a, **kw: pytest.fail("should not run")
    )

    first = openai_translate.translate_batch(
        ["a", "b"], BB_SOURCE, {}, PreprocessResponse(),
        base_url="u", api_key="k", model="m", concurrency=2, session=tmp_path,
    )
    assert [item.dst for item in first] == ["zh1", "zh2"]

    # Second run is served entirely from the cache: no bulk request either.
    monkeypatch.setattr(
        openai_translate, "translate_bulk", lambda *a, **kw: pytest.fail("cache miss")
    )
    second = openai_translate.translate_batch(
        ["a", "b"], BB_SOURCE, {}, PreprocessResponse(),
        base_url="u", api_key="k", model="m", concurrency=2, session=tmp_path,
    )
    assert [item.dst for item in second] == ["zh1", "zh2"]


# --------------------------------------------------------------------------
# One request: translations + ad lines + the Bilibili upload copy
# --------------------------------------------------------------------------

BULK_REPLY = """#=== 标题 ===
【中字】Far Plane 2 全片解析
#=== 简介 ===
原视频：https://www.youtube.com/watch?v=abc123
本期带你逐帧拆解。
#=== 标签 ===
影视解说, 科幻, 预告片解析
#=== 广告行号 ===
3-4, 9
#=== 逐行译文 ===
1\t你好。
2\t世界。
3\t[AD]
4\t[AD]
5\t继续。
"""


def test_split_bulk_sections_ignores_preamble_and_normalises_headers():
    sections = openai_translate._split_bulk_sections("随便说两句\n" + BULK_REPLY)
    assert sections["标题"] == "【中字】Far Plane 2 全片解析"
    assert "youtube.com" in sections["简介"]
    assert sections["广告行号"] == "3-4, 9"
    assert sections["逐行译文"].startswith("1\t你好。")


def test_parse_bulk_reads_every_section_in_one_pass():
    result = openai_translate._parse_bulk(BULK_REPLY, expected=5)
    assert result.title == "【中字】Far Plane 2 全片解析"
    assert result.description.startswith("原视频：")
    assert result.tags == ["影视解说", "科幻", "预告片解析"]
    assert result.lines == {
        1: "你好。", 2: "世界。", 3: "[AD]", 4: "[AD]", 5: "继续。",
    }


def test_parse_ad_lines_accepts_ranges_lists_and_none():
    assert openai_translate._parse_ad_lines("3,7,12-14", 20) == {3, 7, 12, 13, 14}
    assert openai_translate._parse_ad_lines("3、7", 20) == {3, 7}
    assert openai_translate._parse_ad_lines("无", 20) == set()
    assert openai_translate._parse_ad_lines("", 20) == set()
    # Out-of-range values are clamped away instead of deleting a real sentence.
    assert openai_translate._parse_ad_lines("1,99", 5) == {1}


def test_merged_ad_lines_drops_indexes_past_the_end():
    result = openai_translate._parse_bulk(BULK_REPLY, expected=5)
    # The header lists 3-4 and 9; 9 is past the end of a 5-line transcript, so
    # it is dropped before it can delete a sentence that does not exist.
    assert result.ad_lines == {3, 4}
    assert result.merged_ad_lines(5) == {3, 4}


def test_merged_ad_lines_uses_the_inline_marker_when_the_header_is_missing():
    result = openai_translate._parse_bulk("1\t你好。\n2\t[AD]\n3\t世界。", expected=3)
    assert result.ad_lines == set()
    assert result.merged_ad_lines(3) == {2}


def test_bulk_metadata_survives_a_truncated_translation_block():
    raw = (
        "#=== 标题 ===\n好标题\n"
        "#=== 广告行号 ===\n2\n"
        "#=== 逐行译文 ===\n1\t你好。\n2\t"
    )
    result = openai_translate._parse_bulk(raw, expected=3)
    assert result.title == "好标题"
    assert result.merged_ad_lines(3) == {2}
    assert result.lines == {1: "你好。", 2: ""}


def test_translate_batch_flags_ad_lines_and_reports_the_bilibili_copy(monkeypatch):
    def fake_bulk(texts, source, meta, pre, *, client, model, source_url=""):
        return openai_translate.BulkResult(
            lines={1: "你好。", 3: "世界。"},
            ad_lines={2},
            title="好标题",
            description="原视频：https://youtu.be/x",
            tags=["标签a", "标签b"],
        )

    monkeypatch.setattr(openai_translate, "translate_bulk", fake_bulk)
    monkeypatch.setattr(openai_translate, "_client", lambda *a, **kw: object())
    monkeypatch.setattr(
        openai_translate, "translate_sentence", lambda *a, **kw: pytest.fail("should not run")
    )

    meta_out: dict = {}
    out = openai_translate.translate_batch(
        ["a", "b", "c"], BB_SOURCE, {}, PreprocessResponse(),
        base_url="u", api_key="k", model="m", concurrency=3, meta_out=meta_out,
    )

    assert [item.dst for item in out] == ["你好。", "", "世界。"]
    assert [item.is_ad for item in out] == [False, True, False]
    assert out[1].audio_mode == "original"
    assert meta_out == {
        "title": "好标题",
        "description": "原视频：https://youtu.be/x",
        "tags": ["标签a", "标签b"],
    }


def test_translate_asr_records_ad_rows_and_writes_the_bilibili_copy(tmp_path, monkeypatch):
    metadata = tmp_path / "metadata"
    metadata.mkdir()
    asr_file = metadata / "asr_fixed.json"
    _write_asr(asr_file, 2)
    monkeypatch.setattr(openai_translate, "preprocess", lambda *a, **kw: PreprocessResponse())

    def fake_batch(texts, source, meta, pre, **kw):
        kw["meta_out"].update(
            title="好标题",
            description="这期带你拆解全片。",
            tags=["影视解说"],
        )
        return [
            openai_translate.TranslationItem(dst="你好。", audio_mode="tts"),
            openai_translate.TranslationItem(dst="", audio_mode="original", is_ad=True),
        ]

    monkeypatch.setattr(openai_translate, "translate_batch", fake_batch)

    output = openai_translate.translate_asr(
        asr_file, tmp_path, _settings(), YT_SOURCE, source_url="https://youtu.be/x"
    )

    rows = json.loads(output.read_text(encoding="utf-8"))["translation"]
    assert len(rows) == 2
    assert "is_ad" not in rows[0]
    assert rows[1]["is_ad"] is True

    written = openai_translate.load_bilibili_meta(tmp_path)
    assert written["title"] == "好标题"
    assert written["source_url"] == "https://youtu.be/x"
    assert written["tags"] == ["影视解说"]
    # The source link is appended when the model forgot it.
    assert "https://youtu.be/x" in written["description"]


def test_write_bilibili_meta_needs_a_title_or_a_description(tmp_path):
    assert openai_translate.write_bilibili_meta(tmp_path) is None
    assert openai_translate.load_bilibili_meta(tmp_path) is None

    written = openai_translate.write_bilibili_meta(
        tmp_path, title="标题", source_url="https://youtu.be/x"
    )
    assert written is not None
    data = openai_translate.load_bilibili_meta(tmp_path)
    assert data["title"] == "标题"
    assert data["description"] == "原视频：https://youtu.be/x"
