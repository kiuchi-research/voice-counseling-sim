from __future__ import annotations

from pathlib import Path

import yaml

from counseling_voice_demo.audio_probe import build_probe_report


ROOT_DIR = Path(__file__).resolve().parent.parent


def test_audio_probe_report_declares_custom_player_strategy() -> None:
    report = build_probe_report(ROOT_DIR / "app" / "assets" / "dev_probe_tone.mp3")

    assert report["player_strategy"] == "custom_html_audio_element"
    assert report["standard_streamlit_audio_supports_python_events"] is False
    assert "ended_event_in_browser_log" in report["manual_probe_checks"]
    assert "current_position_in_browser_log" in report["manual_probe_checks"]


def test_app_config_declares_audio_probe_and_tts_defaults() -> None:
    config = yaml.safe_load((ROOT_DIR / "config" / "app_config.yaml").read_text())

    assert config["openai"]["tts_endpoint"] == "/v1/audio/speech"
    assert config["openai"]["default_tts_model"] == "${OPENAI_TTS_MODEL:-gpt-4o-mini-tts}"
    assert config["openai"]["fallback_tts_model"] == "${OPENAI_TTS_FALLBACK_MODEL:-tts-1}"
    assert config["openai"]["default_tts_voice"] == "coral"
    assert config["openai"]["fallback_tts_voice"] == "alloy"
    assert "marin" in config["openai"]["allowed_tts_voices"]
    assert "cedar" in config["openai"]["allowed_tts_voices"]
    assert config["openai"]["default_audio_format"] == "mp3"
    assert config["openai"]["low_latency_audio_format"] == "wav"
    assert config["audio_player"]["use_custom_audio_player"] is True
    assert config["audio_player"]["dev_probe_audio_path"] == "app/assets/dev_probe_tone.mp3"
