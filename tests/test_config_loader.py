from __future__ import annotations

from pathlib import Path

import pytest

from counseling_voice_demo.config_loader import ConfigLoadError, load_app_config


ROOT_DIR = Path(__file__).resolve().parent.parent


def test_load_app_config_validates_phase2_defaults(monkeypatch) -> None:
    for env_var in [
        "OPENAI_TEXT_MODEL",
        "OPENAI_TEXT_REASONING_EFFORT",
        "OPENAI_EVALUATION_MODEL",
        "OPENAI_EVALUATION_REASONING_EFFORT",
        "OPENAI_TTS_MODEL",
        "OPENAI_TTS_FALLBACK_MODEL",
    ]:
        monkeypatch.setenv(env_var, "")

    config = load_app_config(ROOT_DIR / "config" / "app_config.yaml")

    assert 2 <= config.app.ahead_generation_turns <= 5
    assert config.app.closing_start_audio_seconds == 120
    assert config.app.closing_start_audio_seconds < config.app.session_duration_audio_seconds
    assert config.session.closing_keep_audio_ready_turns == 1
    assert config.app.enabled_modes["ai_counselor_ai_client"] is True
    assert config.app.enabled_modes["human_counselor_ai_client"] is True
    assert config.app.enabled_modes["ai_counselor_human_client"] is True
    assert config.ui.show_ai_voice_disclosure_in_ui is True
    assert config.app.show_clinical_disclaimer_in_ui is False
    assert config.app.speak_clinical_disclaimer is False
    assert config.openai.use_conversation_state is False
    assert config.openai.use_previous_response_id is False
    assert config.openai.use_store_false_when_supported is True
    assert config.openai.default_text_model == "gpt-5.1"
    assert config.openai.default_text_reasoning_effort == "low"
    assert config.openai.default_evaluation_model == "gpt-5-nano"
    assert config.openai.default_evaluation_reasoning_effort == "low"
    assert config.openai.default_tts_model == "gpt-4o-mini-tts"
    assert config.openai.fallback_tts_model == "tts-1"
    assert config.turn_policy.invalidate_following_unplayed_on_edit is True
    assert config.turn_policy.invalidate_following_unplayed_on_single_regenerate is True
    assert config.replay.forbid_api_calls_in_replay is True
    assert config.audio_merge.create_full_session_mp3 is True
    assert config.audio_merge.allow_cancel_full_mp3_creation is True
    assert config.audio_merge.output_filename == "session_full_public.mp3"
    assert config.audio_merge.include_skipped_partial_in_full_mp3 is False
    assert config.audio_merge.include_skipped_unplayed_in_full_mp3 is False


def test_load_app_config_raises_readable_error_for_invalid_yaml(tmp_path: Path) -> None:
    invalid_config = tmp_path / "app_config.yaml"
    invalid_config.write_text("app:\n  ahead_generation_turns: 1\n", encoding="utf-8")

    with pytest.raises(ConfigLoadError, match="設定ファイルの検証に失敗しました"):
        load_app_config(invalid_config)


def test_load_app_config_accepts_five_ahead_generation_turns(tmp_path: Path) -> None:
    config_path = tmp_path / "app_config.yaml"
    config_text = (ROOT_DIR / "config" / "app_config.yaml").read_text(encoding="utf-8")
    config_path.write_text(
        config_text.replace("ahead_generation_turns: 2", "ahead_generation_turns: 5"),
        encoding="utf-8",
    )

    config = load_app_config(config_path)

    assert config.app.ahead_generation_turns == 5
