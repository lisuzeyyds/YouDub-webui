from __future__ import annotations

import io
import shutil
from pathlib import Path

import soundfile as sf

from .. import runtime_security


def write_original_target_audio(
    output_file: Path,
    item: dict,
    original_vocals_file: Path,
) -> None:
    """Copy the original vocal range (start_time..end_time) into a TTS segment slot.

    Segments whose `audio_mode` is `original` keep the source performance instead of
    synthesized speech. Every TTS backend must still emit `segments/tts/NNNN.wav`
    for them, because `merge_tts_audio` consumes one wav per translation item.
    """
    start = max(0, int(item.get("start_time", 0)))
    end = int(item.get("end_time", start))
    if end <= start:
        raise ValueError(f"Original audio does not cover target segment {start}-{end} ms")

    with sf.SoundFile(original_vocals_file) as source:
        start_frame = max(0, int(start * source.samplerate / 1000))
        requested_end_frame = int(end * source.samplerate / 1000)
        end_frame = requested_end_frame
        available_duration_ms = source.frames / source.samplerate * 1000
        range_error = (
            f"Original audio does not cover target segment {start}-{end} ms: "
            f"requested frames {start_frame}-{requested_end_frame}, available audio is "
            f"{source.frames} frames ({available_duration_ms:.3f} ms)"
        )
        if start_frame >= source.frames:
            raise ValueError(range_error)
        overflow_scaled = end * source.samplerate - source.frames * 1000
        if overflow_scaled > 0:
            if overflow_scaled >= source.samplerate:
                raise ValueError(range_error)
            end_frame = source.frames
        if end_frame <= start_frame:
            raise ValueError(range_error)
        source.seek(start_frame)
        frames = source.read(
            end_frame - start_frame,
            dtype="float32",
            always_2d=True,
        )
        if len(frames) <= 0:
            raise ValueError(range_error)

        encoded = io.BytesIO()
        sf.write(
            encoded,
            frames,
            source.samplerate,
            format="WAV",
            subtype="PCM_16",
        )
        encoded.seek(0)

    runtime_security.remove_private_file(output_file, missing_ok=True)
    with runtime_security.open_private_binary_exclusive(output_file) as handle:
        shutil.copyfileobj(encoded, handle)
        handle.flush()
