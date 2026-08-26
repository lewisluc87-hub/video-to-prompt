"""On-screen text capture (OCR), used by breakdown mode for any video that
has burned-in captions, UI text, or other on-screen text -- not limited to
screen recordings or marketing content. Useful for software demos, tutorials,
captioned vlogs, presentations, or any video where what's written on screen
is part of the content.

Unlike audio transcription (spoken words, used as LLM context only), text
captured here is treated as PRIMARY content in breakdown mode -- on-screen
text is quoted directly in the output, since it's literally part of what the
video shows.

Uses pytesseract if installed (extra: `video2prompt[ocr]`). Runs fully
locally/offline (tesseract binary required on PATH). If unavailable, this
stage is skipped and on_screen_text stays empty for all shots -- the rest
of the pipeline is unaffected.
"""
from __future__ import annotations

import os
import re
import shutil
from pathlib import Path

try:
    import wordninja
except ImportError:
    wordninja = None

# Common Windows install locations (the UB-Mannheim installer's default, and
# the per-user variant) -- checked as a fallback when `tesseract` isn't on
# PATH for the current process, which happens often on Windows since PATH
# changes from an installer don't propagate to already-open terminals, and
# pytesseract doesn't search these locations itself.
_WINDOWS_FALLBACK_PATHS = [
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
]

_resolved_cmd: str | None = None
_checked = False


def _resolve_tesseract_cmd() -> str | None:
    """Find a usable tesseract executable: PATH first, then common install
    locations. Caches the result so this only runs once per process.
    """
    global _resolved_cmd, _checked
    if _checked:
        return _resolved_cmd

    _checked = True
    on_path = shutil.which("tesseract")
    if on_path:
        _resolved_cmd = on_path
        return _resolved_cmd

    for candidate in _WINDOWS_FALLBACK_PATHS:
        expanded = os.path.expandvars(candidate)
        if Path(expanded).exists():
            _resolved_cmd = expanded
            return _resolved_cmd

    return None


def _tesseract_available() -> bool:
    try:
        import pytesseract
    except ImportError:
        return False

    cmd = _resolve_tesseract_cmd()
    if cmd is None:
        return False

    pytesseract.pytesseract.tesseract_cmd = cmd
    return True


_warned = False


def extract_text(frame_path: Path) -> str:
    """Run OCR on a single frame, return cleaned text (may be empty)."""
    global _warned
    if not _tesseract_available():
        if not _warned:
            import sys

            try:
                import pytesseract

                print(
                    "[video2prompt] tesseract binary not found (checked PATH and "
                    "common install locations) -- on-screen text extraction will "
                    "be skipped. Install tesseract and ensure it's on PATH, or set "
                    "pytesseract.pytesseract.tesseract_cmd manually. "
                    "Run with --no-ocr to silence this.",
                    file=sys.stderr,
                )
            except ImportError:
                pass  # pytesseract itself not installed; --no-ocr or extras not requested
            _warned = True
        return ""

    import pytesseract
    from PIL import Image

    try:
        img = Image.open(frame_path)
        raw = pytesseract.image_to_string(img)
    except Exception:  # noqa: BLE001 - OCR failures are non-fatal by design; always fall back to empty string
        return ""

    return _clean(raw)


def extract_text_for_shot(frame_paths: list[Path]) -> str:
    """OCR all keyframes in a shot, merge into one deduplicated text blob.

    Burned-in captions are usually static across a shot's keyframes, so
    naive concatenation would repeat the same line 2-3 times; dedupe by
    line while preserving order of first appearance.
    """
    seen: list[str] = []
    for p in frame_paths:
        text = extract_text(p)
        for line in text.splitlines():
            line = line.strip()
            if line and line not in seen:
                seen.append(line)
    return "\n".join(seen)


_WORD_BOUNDARY_RE = re.compile(r"([a-z])([A-Z])")


def _clean(raw: str) -> str:
    lines = []
    for line in raw.splitlines():
        line = line.strip()
        # Drop obvious OCR noise: very short non-word fragments
        if len(line) < 2:
            continue
        if not any(c.isalnum() for c in line):
            continue
        line = _fix_glued_words(line)
        lines.append(line)
    return "\n".join(lines)


_GLUE_LENGTH_THRESHOLD = 12  # only attempt to split words at least this long


def _fix_glued_words(line: str) -> str:
    """Best-effort fix for words tesseract ran together without a space.

    Two passes:
    1. Lowercase->uppercase boundary (e.g. "SelectAl" -> "Select Al") --
       cheap, reliable, no extra dependency.
    2. Long all-lowercase run-on words (e.g. "forselecting") via wordninja,
       a dictionary-based word segmenter (extra: video2prompt[ocr]). Only
       applied to long, all-lowercase tokens to avoid mangling real words,
       brand names, or anything already correctly spaced.
    """
    line = _WORD_BOUNDARY_RE.sub(r"\1 \2", line)

    if wordninja is None:
        return line

    words = line.split(" ")
    fixed_words = []
    for word in words:
        core = word.strip(".,!?:;\"'()")
        if len(core) >= _GLUE_LENGTH_THRESHOLD and core.isalpha() and core.islower():
            split = wordninja.split(core)
            if len(split) > 1:
                prefix = word[: len(word) - len(word.lstrip(".,!?:;\"'()"))]
                suffix = word[len(prefix) + len(core):]
                word = prefix + " ".join(split) + suffix
        fixed_words.append(word)
    return " ".join(fixed_words)