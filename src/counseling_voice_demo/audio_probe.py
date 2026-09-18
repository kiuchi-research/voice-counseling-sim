from __future__ import annotations

from pathlib import Path
from typing import Any


def build_probe_report(audio_path: Path) -> dict[str, Any]:
    return {
        "audio_path": str(audio_path),
        "audio_exists": audio_path.exists(),
        "audio_format": audio_path.suffix.removeprefix(".").lower(),
        "player_strategy": "custom_html_audio_element",
        "standard_streamlit_audio_supports_python_events": False,
        "manual_probe_checks": [
            "play",
            "pause",
            "resume",
            "skip",
            "ended_event_in_browser_log",
            "current_position_in_browser_log",
        ],
    }
