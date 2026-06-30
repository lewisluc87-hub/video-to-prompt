"""Stage 1: ingest & probe the input video file."""
from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


class IngestError(RuntimeError):
    pass


@dataclass
class VideoInfo:
    path: Path
    duration: float
    fps: float
    width: int
    height: int
    has_audio: bool

    @property
    def resolution(self) -> str:
        return f"{self.width}x{self.height}"


def _require_ffprobe() -> None:
    if shutil.which("ffprobe") is None:
        raise IngestError(
            "ffprobe not found on PATH. Install ffmpeg "
            "(e.g. `apt install ffmpeg` / `brew install ffmpeg`) and try again."
        )


def probe(path: str | Path) -> VideoInfo:
    """Run ffprobe on the input and return basic video metadata."""
    path = Path(path)
    if not path.exists():
        raise IngestError(f"Input file not found: {path}")

    _require_ffprobe()

    cmd = [
        "ffprobe",
        "-v", "error",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        str(path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as exc:
        raise IngestError(f"ffprobe failed on {path}: {exc.stderr}") from exc

    data = json.loads(result.stdout)
    streams = data.get("streams", [])
    video_stream = next((s for s in streams if s.get("codec_type") == "video"), None)
    audio_stream = next((s for s in streams if s.get("codec_type") == "audio"), None)

    if video_stream is None:
        raise IngestError(f"No video stream found in {path}")

    duration = float(data.get("format", {}).get("duration", video_stream.get("duration", 0.0)))
    fps_raw = video_stream.get("avg_frame_rate", "0/1")
    num, _, den = fps_raw.partition("/")
    fps = float(num) / float(den) if den and float(den) != 0 else 0.0

    return VideoInfo(
        path=path,
        duration=duration,
        fps=fps,
        width=int(video_stream.get("width", 0)),
        height=int(video_stream.get("height", 0)),
        has_audio=audio_stream is not None,
    )
