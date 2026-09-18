from __future__ import annotations

import shutil


def find_ffmpeg() -> str | None:
    """Return the ffmpeg executable path when it is available."""
    return shutil.which("ffmpeg")


def has_ffmpeg() -> bool:
    return find_ffmpeg() is not None
