from __future__ import annotations

import importlib
from pathlib import Path

import yaml


ROOT_DIR = Path(__file__).resolve().parent.parent


def test_application_package_imports() -> None:
    package = importlib.import_module("counseling_voice_demo")

    assert package.__version__


def test_step0_required_paths_exist() -> None:
    # Results directories are runtime outputs and may be absent in clean archives.
    (ROOT_DIR / "results" / "sessions").mkdir(parents=True, exist_ok=True)
    (ROOT_DIR / "results" / "replay_sessions").mkdir(parents=True, exist_ok=True)
    required_paths = [
        ROOT_DIR / ".env.example",
        ROOT_DIR / "app" / "streamlit_app.py",
        ROOT_DIR / "config" / "app_config.yaml",
        ROOT_DIR / "config" / "profiles",
        ROOT_DIR / "config" / "profiles" / "counselors",
        ROOT_DIR / "config" / "profiles" / "clients",
        ROOT_DIR / "config" / "themes",
        ROOT_DIR / "config" / "voice_presets",
        ROOT_DIR / "results" / "sessions",
        ROOT_DIR / "results" / "replay_sessions",
        ROOT_DIR / "utility" / "check_ffmpeg.py",
    ]

    for path in required_paths:
        assert path.exists(), f"Missing required Step 0 path: {path}"


def test_app_config_declares_step0_defaults() -> None:
    config = yaml.safe_load((ROOT_DIR / "config" / "app_config.yaml").read_text())

    assert config["ui"]["show_ai_voice_disclosure_in_ui"] is True
    assert config["ui"]["collapse_unplayed_preview_by_default"] is True
    assert config["session"]["closing_keep_audio_ready_turns"] == 1
    assert config["openai"]["use_store_false_when_supported"] is True
    assert config["paths"]["sessions_dir"] == "results/sessions"


def test_env_example_contains_names_only() -> None:
    env_example = (ROOT_DIR / ".env.example").read_text().splitlines()

    assert "OPENAI_API_KEY=" in env_example
    assert "OPENAI_TEXT_MODEL=gpt-5.1" in env_example
    assert "OPENAI_TEXT_REASONING_EFFORT=low" in env_example
    assert "OPENAI_EVALUATION_MODEL=gpt-5-nano" in env_example
    assert "OPENAI_EVALUATION_REASONING_EFFORT=low" in env_example
    assert "OPENAI_TTS_MODEL=gpt-4o-mini-tts" in env_example
    assert "OPENAI_TTS_FALLBACK_MODEL=tts-1" in env_example
