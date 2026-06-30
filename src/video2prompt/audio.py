"""Stage: audio transcription (context only, never literal output).

Uses faster-whisper if installed (extra: `video2prompt[transcribe]`). Runs
fully locally/offline. If unavailable, or the video has no audio, or
--no-transcribe is passed, this stage is skipped entirely and every shot's
transcript_segment stays None -- the rest of the pipeline is unaffected.
"""
from __future__ import annotations

from dataclasses import dataclass

from .ingest import VideoInfo

_whisper_model = None


@dataclass
class TranscriptSegment:
    start: float
    end: float
    text: str


def transcribe(video: VideoInfo, model_size: str = "base") -> list[TranscriptSegment]:
    if not video.has_audio:
        return []

    try:
        from faster_whisper import WhisperModel
    except ImportError:
        return []

    global _whisper_model
    if _whisper_model is None or getattr(_whisper_model, "_v2p_size", None) != model_size:
        _whisper_model = WhisperModel(model_size, device="cpu", compute_type="int8")
        _whisper_model._v2p_size = model_size

    segments, _info = _whisper_model.transcribe(str(video.path))
    return [
        TranscriptSegment(start=seg.start, end=seg.end, text=seg.text.strip())
        for seg in segments
    ]


def attach_to_shot(segments: list[TranscriptSegment], shot_start: float, shot_end: float) -> str | None:
    """Concatenate transcript text for any segment overlapping [shot_start, shot_end)."""
    overlapping = [
        seg.text for seg in segments
        if seg.start < shot_end and seg.end > shot_start and seg.text
    ]
    if not overlapping:
        return None
    return " ".join(overlapping)
