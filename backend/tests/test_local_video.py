from __future__ import annotations

import json
import subprocess
from pathlib import Path

from backend.app.adapters import local_video
from backend.app.sources import detect_source


def test_import_local_video_transcodes_with_configured_ffmpeg(monkeypatch, tmp_path):
    task_id = "local-task"
    upload_dir = local_video.uploaded_video_dir(tmp_path, task_id)
    upload_dir.mkdir(parents=True)
    source_file = upload_dir / "demo.mov"
    source_file.write_bytes(b"video")
    commands: list[list[str]] = []

    def fake_run(cmd, check=False, **kwargs):
        commands.append(cmd)
        output = Path(cmd[-1])
        output.write_bytes(b"mp4")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setenv("FFMPEG_PATH", "/opt/bin/ffmpeg")
    monkeypatch.setenv("YOUDUB_TRANSCODE_ATTEMPTS", "1")
    monkeypatch.setattr(local_video.subprocess, "run", fake_run)

    session, info = local_video.import_local_video(
        f"local://upload/{task_id}?direction=zh-en&filename=demo.mov",
        tmp_path,
        detect_source("local://upload/local-task?direction=zh-en"),
    )

    assert session == tmp_path / "local" / f"demo__{task_id}"
    assert info["title"] == "demo"
    assert info["target_language"] == "en"
    assert commands
    assert commands[0][0] == "/opt/bin/ffmpeg"
    # Transcodes are staged in a .part file and published atomically, so a killed
    # attempt can never leave a truncated video_source.mp4 behind.
    assert commands[0][-1] == str(session / "media" / "video_source.part.mp4")
    assert (session / "media" / "video_source.mp4").read_bytes() == b"mp4"
    assert not (session / "media" / "video_source.part.mp4").exists()
    metadata = json.loads((session / "metadata" / "local_info.json").read_text(encoding="utf-8"))
    assert metadata["original_path"] == str(source_file)


def test_import_local_video_keeps_legacy_root_upload_compatibility(monkeypatch, tmp_path):
    task_id = "legacy-task"
    upload_dir = local_video.upload_dir(tmp_path, task_id)
    upload_dir.mkdir(parents=True)
    source_file = upload_dir / "legacy.mp4"
    source_file.write_bytes(b"video")

    def fake_run(cmd, check=False, **kwargs):
        output = Path(cmd[-1])
        output.write_bytes(b"mp4")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(local_video.subprocess, "run", fake_run)

    session, info = local_video.import_local_video(
        f"local://upload/{task_id}?direction=en-zh&filename=legacy.mp4",
        tmp_path,
        detect_source("local://upload/legacy-task?direction=en-zh"),
    )

    assert session == tmp_path / "local" / f"legacy__{task_id}"
    assert info["original_path"] == str(source_file)
