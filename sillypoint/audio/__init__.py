"""
sillypoint/audio/__init__.py
────────────────────────────
Speech-to-text on the full video clip using faster-whisper.
Extracts audio via ffmpeg, transcribes with word-level timestamps.

Public API
----------
    from sillypoint.audio import transcribe
    segments = transcribe("clip.mp4")
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

from faster_whisper import WhisperModel

from sillypoint.types import TranscriptSegment

_model: WhisperModel | None = None


def _get_model() -> WhisperModel:
    global _model
    if _model is None:
        name = os.getenv("WHISPER_MODEL", "tiny")
        _model = WhisperModel(name, device="cpu", compute_type="int8")
    return _model


def transcribe(video_path: str) -> list[TranscriptSegment]:
    """
    Transcribe umpire speech from a video file.

    Extracts mono 16 kHz audio with ffmpeg, runs faster-whisper with
    word-level timestamps, returns one TranscriptSegment per word.
    Falls back to segment-level timestamps when word data is absent.
    """
    model = _get_model()

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        audio_path = tmp.name

    try:
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", video_path,
                "-ac", "1",
                "-ar", "16000",
                audio_path,
            ],
            check=True,
            capture_output=True,
        )

        segments, _ = model.transcribe(audio_path, word_timestamps=True)

        result: list[TranscriptSegment] = []
        for seg in segments:
            if seg.words:
                for word in seg.words:
                    result.append(TranscriptSegment(
                        start=word.start,
                        end=word.end,
                        text=word.word,
                        word_confidence=word.probability,
                    ))
            else:
                result.append(TranscriptSegment(
                    start=seg.start,
                    end=seg.end,
                    text=seg.text,
                ))

        return result

    finally:
        Path(audio_path).unlink(missing_ok=True)
