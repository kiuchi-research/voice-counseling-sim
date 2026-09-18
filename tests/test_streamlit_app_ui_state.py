from __future__ import annotations

import json
import wave
from pathlib import Path
from types import SimpleNamespace

import pytest

import app.streamlit_app as streamlit_app
from counseling_voice_demo.client_presets import load_client_preset
from counseling_voice_demo.counselor_presets import load_counselor_preset
from app.streamlit_app import (
    AUTO_PLAYBACK_DEFAULT_GAP_SECONDS,
    DEFAULT_RUNTIME_GENERATION_LEAD_LIMIT,
    PUBLIC_TRANSCRIPT_TABLE_COLUMNS,
    RUNTIME_TRANSCRIPT_GATED_PHASES,
    RUNTIME_TRANSCRIPT_REFRESH_INTERVAL_SECONDS,
    PublicTranscriptTurn,
    apply_edit_to_unplayed_turns,
    apply_browser_audio_event_for_ui,
    apply_hold_action_to_unplayed_turns,
    apply_regenerate_from_to_unplayed_turns,
    apply_regenerate_to_unplayed_turns,
    auto_progress_live_session_for_ui,
    build_audio_player_queue_for_ui,
    build_evaluation_internal_context,
    build_conversation_engine_for_ui,
    build_mode_options,
    can_run_evaluation,
    client_preset_session_updates,
    control_disabled_states,
    edit_control_disabled_states,
    format_seconds,
    html_table,
    initial_ui_state,
    is_legacy_seed_unplayed_turns,
    pending_saved_replay_session_id,
    public_transcript_rows_for_display,
    public_transcript_total_audio_seconds,
    public_transcript_latest_visibility_seconds,
    public_transcript_turns_visible_after_audio_end,
    public_transcript_turns_visible_after_previous_audio_end,
    runtime_public_transcript_turns_visible_for_audio,
    prepare_runtime_session_defaults,
    prepare_runtime_text_defaults,
    mark_current_audio_played_for_ui,
    apply_runtime_control_action,
    replay_start_updates,
    refresh_runtime_status,
    refresh_runtime_control_events,
    replay_start_time_options,
    runtime_audio_monitor_document,
    runtime_control_defaults,
    runtime_deferred_interaction_feature_rows,
    runtime_remaining_metric,
    runtime_completed_turn_count_for_display,
    runtime_completed_turn_count_for_metric,
    runtime_completed_turn_count_state_updates,
    runtime_current_turn_metric_value,
    runtime_playback_relative_start_seconds,
    runtime_initial_warmup_state_updates,
    runtime_status_playback_completed_turn_count,
    runtime_status_label_for_timer,
    runtime_status_session_updates,
    runtime_start_options_for_generation_mode,
    runtime_start_options_for_ui,
    runtime_status_rows,
    runtime_generated_turn_count_for_display,
    runtime_generation_throttle_disabled_for_closing,
    runtime_generation_throttle_decision,
    runtime_generation_resume_lead_limit,
    runtime_transcript_visibility_clock_state_updates,
    should_apply_client_preset,
    runtime_transcript_visibility_state_updates,
    runtime_timer_state_updates,
    runtime_monitor_endpoint,
    run_evaluation_for_ui,
    saved_session_replay_ready,
    should_auto_refresh_runtime_transcript,
    skip_current_audio_for_ui,
    start_live_session_for_ui,
    selected_unplayed_turn,
    select_voice_preset_for_profile,
    should_show_evaluation,
    profile_with_session_overrides,
    profile_with_voice_preset,
    TRANSCRIPT_TABLE_COLUMNS,
    transcript_rows_for_display,
    UNPLAYED_TABLE_COLUMNS,
    unplayed_turn_rows_from_session,
    turn_selection_options,
    unplayed_preview_rows,
)
from counseling_voice_demo.content_loader import Profile, Theme, VoicePreset
from counseling_voice_demo.log_writer import create_session_log_dirs
from counseling_voice_demo.openai_evaluation_client import EvaluationRequest, EvaluationResult
from counseling_voice_demo.runtime.control_api import normalize_runtime_start_options
from counseling_voice_demo.models import (
    SessionState,
    SessionStatus,
    SpeakerRole,
    TurnStatus,
    WarningLevel,
)


def test_mode_options_mark_only_enabled_mode_as_startable() -> None:
    options = build_mode_options(
        {
            "ai_counselor_ai_client": True,
            "human_counselor_ai_client": False,
            "ai_counselor_human_client": False,
        }
    )

    assert [option["mode"] for option in options] == [
        "ai_counselor_ai_client",
        "human_counselor_ai_client",
        "ai_counselor_human_client",
    ]
    assert [option["enabled"] for option in options] == [True, False, False]


def test_realtime_model_options_include_2_1_as_default() -> None:
    assert list(streamlit_app.RUNTIME_REALTIME_MODEL_OPTIONS) == [
        "gpt-realtime-2.1",
        "gpt-realtime-2",
        "gpt-realtime-mini",
    ]
    assert streamlit_app.DEFAULT_RUNTIME_REALTIME_MODEL == "gpt-realtime-2.1"


def test_text_model_options_use_5_6_family_for_all_text_tasks() -> None:
    assert list(streamlit_app.RUNTIME_TIMING_LLM_MODEL_OPTIONS) == [
        "gpt-5.6-sol",
        "gpt-5.6-terra",
        "gpt-5.6-luna",
        "gpt-5.4-mini",
    ]
    assert (
        streamlit_app.RUNTIME_PROMPTING_LLM_MODEL_OPTIONS
        is streamlit_app.RUNTIME_TIMING_LLM_MODEL_OPTIONS
    )
    assert (
        streamlit_app.RUNTIME_SUMMARY_LLM_MODEL_OPTIONS
        is streamlit_app.RUNTIME_TIMING_LLM_MODEL_OPTIONS
    )
    assert streamlit_app.DEFAULT_RUNTIME_PROMPTING_LLM_MODEL == "gpt-5.4-mini"
    assert streamlit_app.DEFAULT_RUNTIME_TIMING_LLM_MODEL == "gpt-5.4-mini"
    assert streamlit_app.DEFAULT_RUNTIME_SUMMARY_LLM_MODEL == "gpt-5.4-mini"


def test_control_disabled_states_follow_session_status_and_mode() -> None:
    idle_enabled = control_disabled_states(SessionStatus.IDLE, mode_enabled=True)
    running_enabled = control_disabled_states(SessionStatus.RUNNING, mode_enabled=True)
    paused_enabled = control_disabled_states(SessionStatus.PAUSED, mode_enabled=True)
    idle_disabled_mode = control_disabled_states(SessionStatus.IDLE, mode_enabled=False)

    assert idle_enabled["start"] is False
    assert idle_enabled["pause"] is True
    assert running_enabled["start"] is True
    assert running_enabled["pause"] is False
    assert running_enabled["skip"] is False
    assert paused_enabled["resume"] is False
    assert idle_disabled_mode["start"] is True


def test_initial_ui_state_contains_collapsed_preview_source_and_playback_hold() -> None:
    state = initial_ui_state()

    assert state["session_status"] == SessionStatus.IDLE.value
    assert state["conversation_phase"] == "opening"
    assert state["selected_mode"] == "ai_counselor_ai_client"
    assert state["ui_mode"] == "live"
    assert state["unplayed_turns"] == []
    assert state["selected_unplayed_turn_id"] is None
    assert state["edit_turn_text"] == ""
    assert state["edit_turn_text_source_turn_id"] is None
    assert state["selected_hold_action"] == "このまま再生"
    assert state["selected_theme_id"] is None
    assert state["evaluation_status"] == "not_started"
    assert state["evaluation_public"] == ""
    assert state["evaluation_internal"] == ""
    assert state["auto_playback_gap_seconds"] == AUTO_PLAYBACK_DEFAULT_GAP_SECONDS
    assert state["auto_playback_visible_turn_id"] is None
    assert state["audio_player_queue_version"] == ""
    assert state["audio_player_processed_event_ids"] == []
    assert state["audio_player_last_event"] is None
    assert state["audio_player_continuous_mode"] is False
    assert state["runtime_control_host"] == ""
    assert state["runtime_control_port"] == 0
    assert state["runtime_control_status"] == {}
    assert state["runtime_control_events"] == []
    assert state["runtime_response_instructions"] == []
    assert state["runtime_control_error"] == ""
    assert state["runtime_monitor_connected"] is False
    assert state["runtime_participant_mode"] == "two_clients"
    assert state["runtime_setup_settings_tab"] == "two_clients"
    assert state["counselor_display_name"] == "カウンセラー"
    assert state["client_display_name"] == "妻"
    assert state["client_a_display_name"] == "妻"
    assert state["client_b_display_name"] == "夫"
    assert state["counselor_public_profile_text"] == ""
    assert state["client_public_profile_text"] == ""
    assert state["client_private_profile_text"] == ""
    assert state["client_a_public_profile_text"] == ""
    assert state["client_b_public_profile_text"] == ""
    assert state["counselor_public_profile_source_profile_id"] is None
    assert state["client_public_profile_source_profile_id"] is None
    assert state["client_private_profile_source_profile_id"] is None
    assert state["client_a_public_profile_source_profile_id"] is None
    assert state["client_b_public_profile_source_profile_id"] is None
    assert state["client_common_profile_applied_source_profile_id"] is None
    assert state["selected_counselor_preset_id"] is None
    assert state["applied_counselor_preset_id"] is None
    assert state["selected_client_preset_id"] == "junior_high_school_refusal_couple_v3"
    assert state["applied_client_preset_id"] is None
    assert state["applied_client_preset_participant_mode"] is None
    assert state["client_common_profile_applied_public_profile_text"] is None
    assert state["client_common_profile_applied_prompt_text"] is None
    assert state["runtime_one_client_widget_defaults_version"] == 0
    assert state["runtime_two_client_widget_defaults_version"] == 0
    assert state["counselor_public_profile_file_signature"] is None
    assert state["client_public_profile_file_signature"] is None
    assert state["client_private_profile_file_signature"] is None
    assert state["client_a_public_profile_file_signature"] is None
    assert state["client_b_public_profile_file_signature"] is None
    assert state["runtime_counselor_audio_gain"] is None
    assert state["runtime_client_audio_gain"] == 0.6
    assert state["runtime_one_client_force_stop_after_closing_turns"] == 3
    assert state["runtime_two_client_force_stop_after_closing_turns"] == 3
    assert state["runtime_realtime_model"] == "gpt-realtime-2.1"
    assert state["runtime_prompting_llm_model"] == "gpt-5.4-mini"
    assert state["runtime_prompting_llm_reasoning_effort"] == "low"
    assert state["runtime_summary_llm_model"] == "gpt-5.4-mini"
    assert state["runtime_summary_llm_reasoning_effort"] == "none"
    assert state["runtime_speaker_selection_policy"] == "turn_boundary_timing"
    assert state["runtime_client_tts_voice"] == "marin"
    assert state["runtime_client_a_tts_voice"] == "marin"
    assert state["runtime_client_b_tts_voice"] == "cedar"
    assert state["runtime_counselor_output_speed"] is None
    assert state["runtime_client_output_speed"] == 0.9
    assert state["runtime_client_a_output_speed"] == 0.9
    assert state["runtime_client_b_output_speed"] == 1.0
    assert state["runtime_client_a_audio_gain"] == 0.6
    assert state["runtime_client_b_audio_gain"] == 0.6
    assert state["runtime_closing_start_seconds"] is None
    assert state["runtime_force_stop_after_closing_turns"] is None
    assert (
        state["runtime_generation_lead_limit"]
        == DEFAULT_RUNTIME_GENERATION_LEAD_LIMIT
    )
    assert state["runtime_generation_lead_turns"] == 0
    assert state["runtime_generation_throttle_active"] is False
    assert state["runtime_initial_warmup_session_id"] == ""
    assert state["runtime_initial_warmup_started_at_monotonic"] is None
    assert state["runtime_initial_warmup_released"] is False
    assert state["runtime_initial_warmup_active"] is False
    assert state["runtime_wall_clock_session_id"] == ""
    assert state["runtime_wall_clock_started_at_monotonic"] is None
    assert state["runtime_wall_clock_base_seconds"] == 0.0
    assert state["runtime_transcript_visibility_session_id"] == ""
    assert state["runtime_transcript_visible_turn_count"] == 0
    assert state["runtime_transcript_next_reveal_wall_seconds"] is None
    assert state["runtime_transcript_visibility_clock_seconds"] == 0.0
    assert state["runtime_transcript_visibility_clock_started_at_monotonic"] is None
    assert state["runtime_transcript_visibility_clock_base_seconds"] == 0.0


def test_client_preset_session_updates_uses_wife_for_one_client() -> None:
    preset = load_client_preset(
        Path("config/client_presets/junior_high_school_refusal_couple_v1.yaml")
    )

    updates = client_preset_session_updates(preset, "one_client")

    assert updates["client_display_name"] == "妻"
    assert updates["client_public_profile_text"] == preset.shared.public_profile
    assert updates["client_private_profile_text"] == (
        preset.participants["client_a"].private_profile
    )
    assert updates["client_prompt_text"] == preset.participants["client_a"].prompt
    assert updates["runtime_initial_client_transcript"] == (
        "今日は中2の娘の対応について相談に来ました"
    )
    assert updates["runtime_client_tts_voice"] == "marin"
    assert updates["runtime_client_output_speed"] == 0.9
    assert updates["runtime_client_audio_gain"] == 0.6
    assert updates["client_public_profile_source_profile_id"] == (
        "preset:junior_high_school_refusal_couple_v1"
    )


def test_counselor_preset_session_updates_include_profile_prompt_and_audio() -> None:
    preset = load_counselor_preset(
        Path("config/counselor_presets/counselor_default.yaml")
    )

    updates = streamlit_app.counselor_preset_session_updates(preset)

    assert updates == {
        "counselor_display_name": "カウンセラー",
        "counselor_public_profile_text": preset.counselor.public_profile,
        "counselor_public_profile_source_profile_id": "preset:counselor_default",
        "counselor_prompt_text": preset.counselor.prompt,
        "counselor_prompt_source_profile_id": "preset:counselor_default",
        "runtime_counselor_tts_voice": "shimmer",
        "runtime_counselor_output_speed": 1.05,
        "runtime_counselor_audio_gain": 0.6,
    }


def test_counselor_preset_is_only_reapplied_on_selection_change_or_explicit_reset() -> None:
    applied_state = {"applied_counselor_preset_id": "preset_1"}

    assert not streamlit_app.should_apply_counselor_preset(
        applied_state, "preset_1", force=False
    )
    assert streamlit_app.should_apply_counselor_preset(
        applied_state, "preset_2", force=False
    )
    assert streamlit_app.should_apply_counselor_preset(
        applied_state, "preset_1", force=True
    )


def test_husband_client_preset_uses_husband_as_one_client_and_client_a() -> None:
    preset = load_client_preset(
        Path(
            "config/client_presets/"
            "junior_high_school_refusal_couple_v2_husband.yaml"
        )
    )

    updates = client_preset_session_updates(preset, "one_client")

    assert updates["client_display_name"] == "夫"
    assert updates["client_a_display_name"] == "夫"
    assert updates["client_b_display_name"] == "妻"
    assert updates["runtime_client_tts_voice"] == "cedar"
    assert updates["runtime_client_output_speed"] == 1.0
    assert updates["runtime_client_a_tts_voice"] == "cedar"
    assert updates["runtime_client_b_tts_voice"] == "marin"
    assert updates["runtime_initial_client_a_transcript"] == (
        "今日は中2の娘の対応について相談に来ました"
    )
    assert updates["runtime_initial_client_b_transcript"] == ""
    assert "将来の選択肢が狭くなることが心配です" in updates[
        "client_prompt_text"
    ]


def test_client_preset_session_updates_populates_both_clients_for_two_client_mode() -> None:
    preset = load_client_preset(
        Path("config/client_presets/junior_high_school_refusal_couple_v1.yaml")
    )

    updates = client_preset_session_updates(preset, "two_clients")

    assert updates["client_public_profile_text"] == preset.shared.public_profile
    assert updates["client_prompt_text"] == preset.shared.prompt
    assert updates["client_a_display_name"] == "妻"
    assert updates["client_b_display_name"] == "夫"
    assert preset.shared.prompt in updates["client_a_prompt_text"]
    assert preset.participants["client_a"].prompt in updates["client_a_prompt_text"]
    assert preset.shared.prompt in updates["client_b_prompt_text"]
    assert preset.participants["client_b"].prompt in updates["client_b_prompt_text"]
    assert updates["runtime_client_a_tts_voice"] == "marin"
    assert updates["runtime_client_b_tts_voice"] == "cedar"
    assert updates["runtime_initial_client_a_transcript"] == (
        "今日は中2の娘の対応について相談に来ました"
    )
    assert updates["runtime_initial_client_b_transcript"] == ""


def test_client_preset_is_only_reapplied_on_selection_mode_change_or_explicit_reset() -> None:
    applied_state = {
        "applied_client_preset_id": "preset_1",
        "applied_client_preset_participant_mode": "one_client",
    }

    assert not should_apply_client_preset(
        applied_state, "preset_1", "one_client", force=False
    )
    assert should_apply_client_preset(
        applied_state, "preset_2", "one_client", force=False
    )
    assert should_apply_client_preset(
        applied_state, "preset_1", "two_clients", force=False
    )
    assert should_apply_client_preset(
        applied_state, "preset_1", "one_client", force=True
    )


def test_client_preset_is_active_only_for_applied_participant_mode(monkeypatch) -> None:
    monkeypatch.setattr(
        streamlit_app,
        "st",
        SimpleNamespace(
            session_state={
                "selected_client_preset_id": "preset_1",
                "applied_client_preset_id": "preset_1",
                "applied_client_preset_participant_mode": "one_client",
                "runtime_participant_mode": "one_client",
            }
        ),
    )

    assert streamlit_app.client_preset_is_active("one_client")
    assert not streamlit_app.client_preset_is_active("two_clients")


def test_runtime_control_defaults_falls_back_to_yaml_when_validation_is_stale(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config_path = tmp_path / "runtime_config.yaml"
    config_path.write_text(
        """
runtime:
  control_host: 127.0.0.7
  control_port: 9876
  closing_start_elapsed_seconds: 900
  force_stop_after_closing_turns: 3
latency:
  realtime_api_centered_mode: true
""",
        encoding="utf-8",
    )

    def stale_loader(_config_path):
        raise streamlit_app.RuntimeConfigLoadError("stale runtime model")

    monkeypatch.setattr(streamlit_app, "load_runtime_config", stale_loader)

    assert runtime_control_defaults(config_path) == {
        "host": "127.0.0.7",
        "port": 9876,
        "generation_mode": "realtime_api",
        "warning": "",
    }


def test_apply_runtime_control_action_updates_status_and_session_id() -> None:
    client = _RecordingRuntimeControlClient()

    updates = apply_runtime_control_action(
        "start",
        client=client,
        current_session_id="",
    )

    assert client.actions == ["start"]
    assert updates["runtime_control_error"] == ""
    assert updates["runtime_control_status"] == {
        "session_id": "session-runtime",
        "phase": "running",
    }
    assert updates["runtime_monitor_session_id"] == "session-runtime"
    assert updates["session_status"] == SessionStatus.RUNNING.value
    assert "conversation_phase" not in updates


def test_apply_runtime_control_action_passes_start_options() -> None:
    client = _RecordingRuntimeControlClient()

    apply_runtime_control_action(
        "start",
        client=client,
        start_options={"realtime_api_centered_mode": False},
    )

    assert client.start_options == [{"realtime_api_centered_mode": False}]


def test_apply_runtime_control_action_marks_auto_throttle_pause_reason() -> None:
    client = _RecordingRuntimeControlClient(
        {"session_id": "session-runtime", "phase": "paused"}
    )

    updates = apply_runtime_control_action("pause", client=client)

    assert client.actions == ["pause"]
    assert client.pause_reasons == ["generation_throttle"]
    assert updates["runtime_control_status"]["phase"] == "paused"


def test_runtime_start_options_for_generation_mode_maps_ui_choice_to_api_option() -> None:
    assert runtime_start_options_for_generation_mode("responses_tts") == {
        "realtime_api_centered_mode": False
    }
    assert runtime_start_options_for_generation_mode("realtime_api") == {
        "realtime_api_centered_mode": True
    }


def test_runtime_start_options_for_ui_sends_closing_settings_without_session_time_stop(monkeypatch) -> None:
    monkeypatch.setattr(
        streamlit_app,
        "st",
        SimpleNamespace(
            session_state={
                "runtime_participant_mode": "one_client",
                "counselor_prompt_text": "カウンセラープロンプト",
                "client_prompt_text": "クライアントプロンプト",
                "runtime_initial_client_transcript": "初回相談です。",
                "counselor_display_name": "佐伯",
                "client_display_name": "高橋",
                "runtime_realtime_model": "gpt-realtime-2",
                "runtime_prompting_llm_model": "gpt-5.6-luna",
                "runtime_prompting_llm_reasoning_effort": "medium",
                "runtime_timing_llm_model": "gpt-5.6-terra",
                "runtime_timing_llm_reasoning_effort": "high",
                "runtime_summary_llm_model": "gpt-5.6-luna",
                "runtime_summary_llm_reasoning_effort": "low",
                "runtime_speaker_selection_policy": "distributed_timing",
                "runtime_counselor_tts_voice": "shimmer",
                "runtime_client_tts_voice": "cedar",
                "runtime_counselor_output_speed": 0.85,
                "runtime_client_output_speed": 0.95,
                "runtime_closing_start_seconds": 900,
                "runtime_force_stop_after_closing_turns": 9,
                "runtime_one_client_force_stop_after_closing_turns": 2,
                "runtime_two_client_force_stop_after_closing_turns": 3,
                "runtime_counselor_audio_gain": 0.7,
                "runtime_client_audio_gain": 2.3,
            }
        ),
    )

    options = runtime_start_options_for_ui(
        counselor_profile=_profile(role="counselor", hidden_background=None),
        client_profile=_profile(role="client", hidden_background=None),
    )

    assert options["realtime_api_centered_mode"] is True
    assert options["realtime_model"] == "gpt-realtime-2"
    assert options["prompting_llm_model"] == "gpt-5.6-luna"
    assert options["prompting_llm_reasoning_effort"] == "medium"
    assert options["timing_llm_model"] == "gpt-5.6-terra"
    assert options["timing_llm_reasoning_effort"] == "high"
    assert options["summary_llm_model"] == "gpt-5.6-luna"
    assert options["summary_llm_reasoning_effort"] == "low"
    assert options["speaker_selection_policy"] == "distributed_timing"
    assert options["counselor_tts_voice"] == "shimmer"
    assert options["client_tts_voice"] == "cedar"
    assert options["counselor_realtime_output_speed"] == 0.85
    assert options["client_realtime_output_speed"] == 0.95
    assert options["closing_start_elapsed_seconds"] == 900.0
    assert options["force_stop_after_closing_turns"] == 2
    assert options["speaker_gains"] == {"counselor": 0.7, "client": 2.3}
    assert "stop_condition" not in options
    assert "max_elapsed_seconds" not in options
    assert options["participants"] == {
        "counselor": {
            "role": "counselor",
            "display_name": "佐伯",
            "public_profile_source": "counselor public profile",
        },
        "client": {
            "role": "client",
            "display_name": "高橋",
            "public_profile_source": "client public profile",
        },
    }


def test_runtime_provider_switch_preserves_targets_and_sends_no_credentials(
    monkeypatch,
):
    state = {}
    monkeypatch.setattr(streamlit_app, "st", SimpleNamespace(session_state=state))
    streamlit_app.prepare_runtime_provider_defaults()
    state["runtime_realtime_provider"] = "azure_eastus2"
    state["runtime_realtime_target_azure_eastus2"] = "custom-azure-deployment"
    state["runtime_realtime_target_openai"] = "gpt-realtime-2"
    azure_options = runtime_start_options_for_ui(
        counselor_profile=_profile(role="counselor", hidden_background=None),
        client_profile=_profile(role="client", hidden_background=None),
    )
    assert azure_options["ai_routes"] == {
        "realtime_speech": {
            "provider": "azure_eastus2",
            "model_ref": "custom-azure-deployment",
        },
    }
    assert "realtime_model" not in azure_options
    assert "api_key" not in json.dumps(azure_options)
    assert "endpoint" not in json.dumps(azure_options)
    state["runtime_realtime_provider"] = "openai"
    streamlit_app.prepare_runtime_provider_defaults()
    openai_options = runtime_start_options_for_ui(
        counselor_profile=_profile(role="counselor", hidden_background=None),
        client_profile=_profile(role="client", hidden_background=None),
    )
    assert (
        openai_options["ai_routes"]["realtime_speech"]["model_ref"] == "gpt-realtime-2"
    )
    state["runtime_realtime_provider"] = "azure_eastus2"
    streamlit_app.prepare_runtime_provider_defaults()
    assert state["runtime_realtime_target_azure_eastus2"] == "custom-azure-deployment"


def test_runtime_provider_widgets_keep_edits_when_switched_away():
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_string(
        "from app.streamlit_app import render_runtime_model_session_controls\n"
        "render_runtime_model_session_controls()\n"
    ).run()
    assert not app.exception
    app.selectbox(key="runtime_realtime_provider_widget").select("azure_eastus2").run()
    app.text_input(key="runtime_realtime_target_azure_eastus2_widget").input(
        "edited-deployment"
    ).run()
    app.selectbox(key="runtime_realtime_provider_widget").select("openai").run()
    app.selectbox(key="runtime_realtime_target_openai_widget").select(
        "gpt-realtime-2"
    ).run()
    app.selectbox(key="runtime_realtime_provider_widget").select("azure_eastus2").run()
    assert not app.exception
    assert (
        app.text_input(key="runtime_realtime_target_azure_eastus2_widget").value
        == "edited-deployment"
    )
    app.selectbox(key="runtime_realtime_provider_widget").select("openai").run()
    assert (
        app.selectbox(key="runtime_realtime_target_openai_widget").value
        == "gpt-realtime-2"
    )


def test_runtime_text_provider_defaults_and_payload(monkeypatch):
    state = {}
    monkeypatch.setattr(streamlit_app, "st", SimpleNamespace(session_state=state))
    streamlit_app.prepare_runtime_text_provider_defaults()
    for name in ("prompt_director", "turn_timing", "session_summary"):
        assert state[f"runtime_{name}_provider"] == "azure_eastus2"
        state[f"runtime_{name}_target_azure_eastus2"] = "custom-text-deployment"
    state["runtime_turn_timing_provider"] = "openai"
    state["runtime_turn_timing_target_openai"] = "gpt-5.6-sol"
    options = runtime_start_options_for_ui(
        counselor_profile=_profile(role="counselor", hidden_background=None),
        client_profile=_profile(role="client", hidden_background=None),
    )
    assert options["ai_routes"] == {
        "prompt_director": {
            "provider": "azure_eastus2",
            "model_ref": "custom-text-deployment",
        },
        "session_summary": {
            "provider": "azure_eastus2",
            "model_ref": "custom-text-deployment",
        },
        "turn_timing": {"provider": "openai", "model_ref": "gpt-5.6-sol"},
    }
    for key in ("prompting_llm_model", "timing_llm_model", "summary_llm_model"):
        assert key not in options
    assert options["prompting_llm_reasoning_effort"] == "low"
    assert "api_key" not in json.dumps(options)
    assert "endpoint" not in json.dumps(options)
    assert normalize_runtime_start_options(options)["ai_routes"] == options["ai_routes"]


def test_runtime_stt_provider_defaults_and_payload(monkeypatch):
    state = {}
    monkeypatch.setattr(streamlit_app, "st", SimpleNamespace(session_state=state))
    streamlit_app.prepare_runtime_stt_provider_defaults()
    assert state["runtime_stt_provider"] == "azure_eastus2"
    state["runtime_stt_target_azure_eastus2"] = "custom-stt-deployment"
    options = runtime_start_options_for_ui(
        counselor_profile=_profile(role="counselor", hidden_background=None),
        client_profile=_profile(role="client", hidden_background=None),
    )
    assert options["ai_routes"]["realtime_transcription"] == {
        "provider": "azure_eastus2",
        "model_ref": "custom-stt-deployment",
    }
    assert "api_key" not in json.dumps(options)
    assert "endpoint" not in json.dumps(options)
    assert normalize_runtime_start_options(options)["ai_routes"] == options["ai_routes"]


def test_runtime_stt_provider_widgets_keep_edits_independently():
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_string(
        "from app.streamlit_app import render_runtime_model_session_controls\n"
        "render_runtime_model_session_controls()\n"
    ).run()
    assert not app.exception
    assert app.selectbox(key="runtime_stt_provider_widget").value == "azure_eastus2"
    app.text_input(key="runtime_stt_target_azure_eastus2_widget").input(
        "azure-stt-custom"
    ).run()
    app.selectbox(key="runtime_stt_provider_widget").select("openai").run()
    app.text_input(key="runtime_stt_target_openai_widget").input(
        "gpt-4o-transcribe"
    ).run()
    app.selectbox(key="runtime_stt_provider_widget").select("azure_eastus2").run()
    assert (
        app.text_input(key="runtime_stt_target_azure_eastus2_widget").value
        == "azure-stt-custom"
    )
    assert (
        app.selectbox(key="runtime_realtime_provider_widget").value == "azure_eastus2"
    )
    app.selectbox(key="runtime_stt_provider_widget").select("openai").run()
    assert (
        app.text_input(key="runtime_stt_target_openai_widget").value
        == "gpt-4o-transcribe"
    )
    assert not app.exception


def test_runtime_text_provider_widgets_preserve_independent_edits():
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_string(
        "from app.streamlit_app import render_runtime_model_session_controls\n"
        "render_runtime_model_session_controls()\n"
    ).run()
    assert not app.exception
    for name in ("prompt_director", "turn_timing", "session_summary"):
        assert (
            app.selectbox(key=f"runtime_{name}_provider_widget").value
            == "azure_eastus2"
        )
    app.text_input(key="runtime_session_summary_target_azure_eastus2_widget").input(
        "summary-custom"
    ).run()
    app.selectbox(key="runtime_session_summary_provider_widget").select("openai").run()
    app.selectbox(key="runtime_session_summary_target_openai_widget").select(
        "gpt-5.6-sol"
    ).run()
    app.selectbox(key="runtime_session_summary_provider_widget").select(
        "azure_eastus2"
    ).run()
    assert not app.exception
    assert (
        app.text_input(key="runtime_session_summary_target_azure_eastus2_widget").value
        == "summary-custom"
    )
    app.selectbox(key="runtime_session_summary_provider_widget").select("openai").run()
    assert (
        app.selectbox(key="runtime_session_summary_target_openai_widget").value
        == "gpt-5.6-sol"
    )
    assert (
        app.selectbox(key="runtime_prompt_director_provider_widget").value
        == "azure_eastus2"
    )


def test_runtime_start_options_for_ui_sends_one_client_profile(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        streamlit_app,
        "st",
        SimpleNamespace(
            session_state={
                "runtime_participant_mode": "one_client",
                "counselor_prompt_text": "カウンセラープロンプト",
                "client_prompt_text": "クライアントプロンプト",
                "counselor_public_profile_text": "カウンセラープロフィール",
                "client_public_profile_text": "クライアントプロフィール",
                "client_private_profile_text": "クライアント秘密プロフィール",
                "runtime_initial_client_transcript": "初回相談です。",
                "counselor_display_name": "佐伯",
                "client_display_name": "高橋",
                "runtime_realtime_model": "gpt-realtime-2",
            }
        ),
    )

    options = runtime_start_options_for_ui(
        counselor_profile=_profile(role="counselor", hidden_background=None),
        client_profile=_profile(role="client", hidden_background=None),
    )

    assert options["counselor_prompt"] == "カウンセラープロンプト"
    assert options["client_prompt"] == "クライアントプロンプト"
    assert options["participants"] == {
        "counselor": {
            "role": "counselor",
            "display_name": "佐伯",
            "public_profile_source": "カウンセラープロフィール",
        },
        "client": {
            "role": "client",
            "display_name": "高橋",
            "public_profile_source": "クライアントプロフィール",
            "private_profile_source": "クライアント秘密プロフィール",
        },
    }
    normalized = normalize_runtime_start_options(options)
    assert set(normalized["participants"]) == {"counselor", "client"}


def test_runtime_start_options_for_ui_sends_human_counselor_one_client_mode(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        streamlit_app,
        "st",
        SimpleNamespace(
            session_state={
                "selected_mode": "human_counselor_ai_client",
                "runtime_participant_mode": "one_client",
                "counselor_prompt_text": "送らないカウンセラープロンプト",
                "client_prompt_text": "クライアントプロンプト",
                "client_public_profile_text": "クライアントプロフィール",
                "client_private_profile_text": "クライアント秘密プロフィール",
                "runtime_initial_client_transcript": "クライアント初回発話は送らない",
                "client_display_name": "高橋",
                "runtime_client_tts_voice": "cedar",
                "runtime_client_output_speed": 0.95,
                "runtime_counselor_tts_voice": "shimmer",
                "runtime_counselor_output_speed": 0.85,
                "runtime_counselor_audio_gain": 0.7,
                "runtime_client_audio_gain": 2.3,
            }
        ),
    )

    options = runtime_start_options_for_ui(
        counselor_profile=_profile(role="counselor", hidden_background=None),
        client_profile=_profile(role="client", hidden_background=None),
        selected_mode="human_counselor_ai_client",
    )

    assert options["interaction_mode"] == "human_counselor_ai_client"
    assert options["participant_mode"] == "one_client"
    assert options["human_input_mode"] == "push_to_talk"
    assert options["human_stt_submit_policy"] == "auto_on_final"
    assert options["human_interrupts_enabled"] is True
    assert "counselor_prompt" not in options
    assert "counselor_tts_voice" not in options
    assert "counselor_realtime_output_speed" not in options
    assert "initial_client_transcript" not in options
    assert options["client_prompt"] == "クライアントプロンプト"
    assert options["client_tts_voice"] == "cedar"
    assert options["client_realtime_output_speed"] == 0.95
    assert options["speaker_gains"] == {"client": 2.3}
    assert options["participants"] == {
        "client": {
            "role": "client",
            "display_name": "高橋",
            "public_profile_source": "クライアントプロフィール",
            "private_profile_source": "クライアント秘密プロフィール",
        },
    }
    normalized = normalize_runtime_start_options(options)
    assert normalized["participants"]["counselor"]["actor_kind"] == "human"
    assert set(normalized["participants"]) == {"counselor", "client"}


def test_human_client_start_options_exclude_stale_ai_client_settings(monkeypatch):
    state = {
        "selected_mode": "ai_counselor_human_client",
        "runtime_participant_mode": "two_clients",
        "runtime_speaker_selection_policy": "turn_boundary_timing",
        "client_prompt_text": "以前のAIプロンプト",
        "client_private_profile_text": "以前の秘密背景",
        "runtime_initial_client_transcript": "以前の架空発話",
        "runtime_client_tts_voice": "cedar",
        "runtime_client_audio_gain": 2.0,
        "counselor_prompt_text": "相談を聞いてください",
        "counselor_public_profile_text": "AIカウンセラーの紹介",
        "human_client_display_name": "参加者",
        "human_client_shared_information": "仕事について話したい",
        "runtime_realtime_provider": "azure_openai",
        "runtime_realtime_target_azure_openai": "deployed-realtime",
    }
    monkeypatch.setattr(streamlit_app.st, "session_state", state)
    options = runtime_start_options_for_ui(
        counselor_profile=_profile(role="counselor", hidden_background=None),
        client_profile=_profile(role="client", hidden_background="秘密背景"),
    )
    assert options["interaction_mode"] == "ai_counselor_human_client"
    assert options["participant_mode"] == "one_client"
    assert options["speaker_selection_policy"] == "fixed_round_robin"
    assert options["initial_client_transcript"] == ""
    assert options["shared_case"] == "仕事について話したい"
    assert options["counselor_prompt"] == "相談を聞いてください"
    assert options["participants"]["client"] == {
        "role": "client",
        "actor_kind": "human",
        "display_name": "参加者",
    }
    assert options["participants"]["counselor"]["actor_kind"] == "ai"
    assert options["ai_routes"]["realtime_speech"]["model_ref"] == "deployed-realtime"
    assert "client_prompt" not in options
    assert "client_tts_voice" not in options
    assert "client" not in options.get("speaker_gains", {})
    assert "秘密背景" not in json.dumps(options, ensure_ascii=False)
    assert state["runtime_participant_mode"] == "two_clients"
    normalize_runtime_start_options(options)


def test_human_client_mode_uses_active_status_for_microphone(monkeypatch):
    monkeypatch.setattr(
        streamlit_app.st,
        "session_state",
        {
            "selected_mode": "ai_counselor_ai_client",
            "runtime_control_status": {
                "phase": "running",
                "interaction_mode": "ai_counselor_human_client",
            },
        },
    )
    assert streamlit_app.runtime_human_input_enabled()


def test_runtime_start_options_for_ui_sends_two_client_participants(monkeypatch) -> None:
    monkeypatch.setattr(
        streamlit_app,
        "st",
        SimpleNamespace(
            session_state={
                "runtime_participant_mode": "two_clients",
                "counselor_prompt_text": "カウンセラープロンプト",
                "counselor_public_profile_text": "カウンセラープロフィール",
                "counselor_display_name": "佐伯",
                "client_prompt_text": "旧クライアントプロンプト",
                "client_public_profile_text": "共通プロフィール",
                "client_display_name": "旧クライアント",
                "runtime_initial_client_transcript": "旧初回発話",
                "client_a_display_name": "高橋A",
                "client_b_display_name": "高橋B",
                "client_a_prompt_text": "Aプロンプト",
                "client_b_prompt_text": "Bプロンプト",
                "client_a_public_profile_text": "Aプロフィール",
                "client_b_public_profile_text": "Bプロフィール",
                "runtime_initial_client_a_transcript": "Aの初回発話",
                "runtime_initial_client_b_transcript": "Bの初回発話",
                "runtime_realtime_model": "gpt-realtime-2",
                "runtime_counselor_tts_voice": "shimmer",
                "runtime_client_tts_voice": "legacy-client-voice",
                "runtime_client_a_tts_voice": "cedar",
                "runtime_client_b_tts_voice": "coral",
                "runtime_counselor_output_speed": 1.1,
                "runtime_client_output_speed": 0.5,
                "runtime_client_a_output_speed": 0.9,
                "runtime_client_b_output_speed": 0.85,
                "runtime_counselor_audio_gain": 0.7,
                "runtime_client_audio_gain": 2.3,
                "runtime_client_a_audio_gain": 0.61,
                "runtime_client_b_audio_gain": 0.62,
                "runtime_force_stop_after_closing_turns": 2,
                "runtime_one_client_force_stop_after_closing_turns": 2,
                "runtime_two_client_force_stop_after_closing_turns": 3,
            }
        ),
    )

    options = runtime_start_options_for_ui(
        counselor_profile=_profile(role="counselor", hidden_background=None),
        client_profile=_profile(role="client", hidden_background=None),
    )

    assert options["realtime_api_centered_mode"] is True
    assert options["realtime_model"] == "gpt-realtime-2"
    assert options["fixed_speaker_sequence"] == [
        "counselor",
        "client_a",
        "client_b",
        "counselor",
        "client_b",
        "client_a",
    ]
    assert options["speaker_gains"] == {
        "counselor": 0.7,
        "client_a": 0.61,
        "client_b": 0.62,
    }
    assert options["force_stop_after_closing_turns"] == 3
    assert options["shared_case"] == "共通プロフィール"
    assert "client_prompt" not in options
    assert "initial_client_transcript" not in options
    assert "client_display_name" not in options

    participants = options["participants"]
    assert participants["counselor"]["display_name"] == "佐伯"
    assert participants["counselor"]["prompt_source"] == "カウンセラープロンプト"
    assert participants["counselor"]["public_profile_source"] == "カウンセラープロフィール"
    assert participants["client_a"] == {
        "role": "client",
        "display_name": "高橋A",
        "prompt_source": "Aプロンプト",
        "private_profile_source": "Aプロフィール",
        "initial_transcript": "Aの初回発話",
        "voice": "cedar",
        "realtime_output_speed": 0.9,
    }
    assert participants["client_b"] == {
        "role": "client",
        "display_name": "高橋B",
        "prompt_source": "Bプロンプト",
        "private_profile_source": "Bプロフィール",
        "initial_transcript": "Bの初回発話",
        "voice": "coral",
        "realtime_output_speed": 0.85,
    }
    normalized = normalize_runtime_start_options(options)
    assert set(normalized["participants"]) == {"counselor", "client_a", "client_b"}
    assert normalized["shared_case"] == "共通プロフィール"
    assert normalized["fixed_speaker_sequence"] == (
        "counselor",
        "client_a",
        "client_b",
        "counselor",
        "client_b",
        "client_a",
    )


def test_runtime_start_options_for_ui_sends_human_counselor_two_client_mode(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        streamlit_app,
        "st",
        SimpleNamespace(
            session_state={
                "selected_mode": "human_counselor_ai_client",
                "runtime_participant_mode": "two_clients",
                "counselor_prompt_text": "送らないカウンセラープロンプト",
                "counselor_public_profile_text": "送らないプロフィール",
                "counselor_display_name": "佐伯",
                "client_public_profile_text": "共通プロフィール",
                "client_a_display_name": "高橋A",
                "client_b_display_name": "高橋B",
                "client_a_prompt_text": "Aプロンプト",
                "client_b_prompt_text": "Bプロンプト",
                "client_a_public_profile_text": "Aプロフィール",
                "client_b_public_profile_text": "Bプロフィール",
                "runtime_initial_client_a_transcript": "Aの初回発話",
                "runtime_initial_client_b_transcript": "Bの初回発話",
                "runtime_client_a_tts_voice": "cedar",
                "runtime_client_b_tts_voice": "coral",
                "runtime_client_a_output_speed": 0.9,
                "runtime_client_b_output_speed": 0.85,
                "runtime_counselor_audio_gain": 0.7,
                "runtime_client_a_audio_gain": 0.61,
                "runtime_client_b_audio_gain": 0.62,
                "runtime_two_client_force_stop_after_closing_turns": 3,
            }
        ),
    )

    options = runtime_start_options_for_ui(
        counselor_profile=_profile(role="counselor", hidden_background=None),
        client_profile=_profile(role="client", hidden_background=None),
        selected_mode="human_counselor_ai_client",
    )

    assert options["interaction_mode"] == "human_counselor_ai_client"
    assert options["participant_mode"] == "two_clients"
    assert options["human_interrupts_enabled"] is True
    assert options["fixed_speaker_sequence"] == [
        "counselor",
        "client_a",
        "client_b",
        "counselor",
        "client_b",
        "client_a",
    ]
    assert options["speaker_gains"] == {
        "client_a": 0.61,
        "client_b": 0.62,
    }
    assert options["shared_case"] == "共通プロフィール"
    participants = options["participants"]
    assert set(participants) == {"client_a", "client_b"}
    assert participants["client_a"] == {
        "role": "client",
        "display_name": "高橋A",
        "prompt_source": "Aプロンプト",
        "private_profile_source": "Aプロフィール",
        "initial_transcript": "Aの初回発話",
        "voice": "cedar",
        "realtime_output_speed": 0.9,
    }
    assert participants["client_b"] == {
        "role": "client",
        "display_name": "高橋B",
        "prompt_source": "Bプロンプト",
        "private_profile_source": "Bプロフィール",
        "initial_transcript": "Bの初回発話",
        "voice": "coral",
        "realtime_output_speed": 0.85,
    }
    normalized = normalize_runtime_start_options(options)
    assert normalized["participants"]["counselor"]["actor_kind"] == "human"
    assert set(normalized["participants"]) == {"counselor", "client_a", "client_b"}


def test_prepare_two_client_runtime_widget_defaults_uses_requested_audio_defaults(
    monkeypatch,
) -> None:
    defaults = streamlit_app.two_client_attachment_defaults()
    client_a_common_prompt = streamlit_app._participant_common_prompt(
        "共通クライアントプロンプト",
        streamlit_app.TWO_CLIENT_A_PROMPT_INTRO,
    )
    client_b_common_prompt = streamlit_app._participant_common_prompt(
        "共通クライアントプロンプト",
        streamlit_app.TWO_CLIENT_B_PROMPT_INTRO,
    )
    session_state = {
        "client_prompt_text": "共通クライアントプロンプト",
        "client_public_profile_text": "共通プロフィール",
        "runtime_initial_client_transcript": "共通初回発話",
        "runtime_client_tts_voice": "coral",
        "runtime_client_output_speed": 0.75,
        "runtime_client_audio_gain": 1.2,
        "client_a_display_name": "",
        "client_b_display_name": "",
        "client_a_prompt_source_profile_id": None,
        "client_b_prompt_source_profile_id": None,
        "client_a_prompt_text": "",
        "client_b_prompt_text": "",
        "client_a_public_profile_source_profile_id": None,
        "client_b_public_profile_source_profile_id": None,
        "client_a_public_profile_text": "",
        "client_b_public_profile_text": "",
        "runtime_initial_client_a_transcript": "",
        "runtime_initial_client_b_transcript": "",
        "runtime_client_a_tts_voice": "",
        "runtime_client_b_tts_voice": "",
        "runtime_client_a_output_speed": None,
        "runtime_client_b_output_speed": None,
        "runtime_client_a_audio_gain": None,
        "runtime_client_b_audio_gain": None,
    }
    monkeypatch.setattr(
        streamlit_app,
        "st",
        SimpleNamespace(session_state=session_state),
    )

    streamlit_app.prepare_two_client_runtime_widget_defaults(
        selected_client=_profile(role="client", hidden_background=None),
        runtime_tts_voice_options=["marin", "cedar", "coral"],
    )

    assert session_state["runtime_client_a_tts_voice"] == "marin"
    assert session_state["runtime_client_b_tts_voice"] == "cedar"
    assert session_state["runtime_client_a_output_speed"] == 0.9
    assert session_state["runtime_client_b_output_speed"] == 1.0
    assert session_state["runtime_client_a_audio_gain"] == 0.6
    assert session_state["runtime_client_b_audio_gain"] == 0.6
    assert session_state["client_a_display_name"] == "妻"
    assert session_state["client_b_display_name"] == "夫"
    assert session_state["client_a_prompt_text"] == (
        streamlit_app._join_two_client_default_parts(
            client_a_common_prompt,
            defaults["client_a_prompt"],
        )
    )
    assert session_state["client_b_prompt_text"] == (
        streamlit_app._join_two_client_default_parts(
            client_b_common_prompt,
            defaults["client_b_prompt"],
        )
    )
    assert session_state["client_a_public_profile_text"] == defaults[
        "client_a_profile"
    ]
    assert session_state["client_b_public_profile_text"] == defaults[
        "client_b_profile"
    ]
    assert (
        session_state["client_common_profile_applied_prompt_text"]
        == "共通クライアントプロンプト"
    )
    assert (
        session_state["client_common_profile_applied_public_profile_text"]
        == "共通プロフィール"
    )


def test_prepare_two_client_runtime_widget_defaults_keeps_custom_client_text(
    monkeypatch,
) -> None:
    client = _profile(role="client", hidden_background=None)
    defaults = streamlit_app.two_client_attachment_defaults()
    previous_client_a_common_prompt = streamlit_app._participant_common_prompt(
        "前回の共通プロンプト",
        streamlit_app.TWO_CLIENT_A_PROMPT_INTRO,
    )
    updated_client_a_common_prompt = streamlit_app._participant_common_prompt(
        "更新された共通プロンプト",
        streamlit_app.TWO_CLIENT_A_PROMPT_INTRO,
    )
    session_state = {
        "client_prompt_text": "更新された共通プロンプト",
        "client_public_profile_text": "更新された共通プロフィール",
        "client_common_profile_applied_source_profile_id": client.profile_id,
        "client_common_profile_applied_prompt_text": "前回の共通プロンプト",
        "client_common_profile_applied_public_profile_text": "前回の共通プロフィール",
        "runtime_initial_client_transcript": "",
        "runtime_client_tts_voice": "coral",
        "runtime_client_output_speed": 0.75,
        "runtime_client_audio_gain": 1.2,
        "client_a_display_name": "妻",
        "client_b_display_name": "夫",
        "client_a_prompt_source_profile_id": client.profile_id,
        "client_b_prompt_source_profile_id": client.profile_id,
        "client_a_prompt_text": streamlit_app._join_two_client_default_parts(
            previous_client_a_common_prompt,
            defaults["client_a_prompt"],
        ),
        "client_b_prompt_text": "B専用プロンプト",
        "client_a_public_profile_source_profile_id": client.profile_id,
        "client_b_public_profile_source_profile_id": client.profile_id,
        "client_a_public_profile_text": streamlit_app._join_two_client_default_parts(
            "前回の共通プロフィール",
            defaults["client_a_profile"],
        ),
        "client_b_public_profile_text": "B専用プロフィール",
        "runtime_initial_client_a_transcript": "",
        "runtime_initial_client_b_transcript": "",
        "runtime_client_a_tts_voice": "marin",
        "runtime_client_b_tts_voice": "cedar",
        "runtime_client_a_output_speed": 0.9,
        "runtime_client_b_output_speed": 1.0,
        "runtime_client_a_audio_gain": 0.6,
        "runtime_client_b_audio_gain": 0.6,
    }
    monkeypatch.setattr(streamlit_app, "st", SimpleNamespace(session_state=session_state))

    streamlit_app.prepare_two_client_runtime_widget_defaults(
        selected_client=client,
        runtime_tts_voice_options=["marin", "cedar", "coral"],
    )

    assert session_state["client_a_prompt_text"] == (
        streamlit_app._join_two_client_default_parts(
            updated_client_a_common_prompt,
            defaults["client_a_prompt"],
        )
    )
    assert session_state["client_a_public_profile_text"] == defaults[
        "client_a_profile"
    ]
    assert session_state["client_b_prompt_text"] == "B専用プロンプト"
    assert session_state["client_b_public_profile_text"] == "B専用プロフィール"


def test_prepare_two_client_runtime_widget_defaults_uses_attachment_defaults(
    monkeypatch,
) -> None:
    defaults = streamlit_app.two_client_attachment_defaults()
    client_a_common_prompt = streamlit_app._participant_common_prompt(
        defaults["common_prompt"],
        streamlit_app.TWO_CLIENT_A_PROMPT_INTRO,
    )
    client_b_common_prompt = streamlit_app._participant_common_prompt(
        defaults["common_prompt"],
        streamlit_app.TWO_CLIENT_B_PROMPT_INTRO,
    )
    assert (
        defaults["client_a_initial_transcript"]
        == "今日は中2の娘の対応について相談に来ました"
    )
    assert "夫婦が互いの発言を聞き" in defaults["common_prompt"]
    assert "カウンセラーを介さず" in defaults["common_prompt"]
    assert "夫婦だけの口論を長引かせない" in defaults["common_prompt"]
    assert defaults["client_b_initial_transcript"] == ""
    session_state = {
        "client_prompt_text": "",
        "client_public_profile_text": "",
        "runtime_initial_client_transcript": "",
        "runtime_client_tts_voice": "",
        "runtime_client_output_speed": None,
        "runtime_client_audio_gain": None,
        "client_a_display_name": "",
        "client_b_display_name": "",
        "client_a_prompt_source_profile_id": None,
        "client_b_prompt_source_profile_id": None,
        "client_a_prompt_text": "",
        "client_b_prompt_text": "",
        "client_a_public_profile_source_profile_id": None,
        "client_b_public_profile_source_profile_id": None,
        "client_a_public_profile_text": "",
        "client_b_public_profile_text": "",
        "runtime_initial_client_a_transcript": "",
        "runtime_initial_client_b_transcript": "",
        "runtime_client_a_tts_voice": "",
        "runtime_client_b_tts_voice": "",
        "runtime_client_a_output_speed": None,
        "runtime_client_b_output_speed": None,
        "runtime_client_a_audio_gain": None,
        "runtime_client_b_audio_gain": None,
    }
    monkeypatch.setattr(streamlit_app, "st", SimpleNamespace(session_state=session_state))

    streamlit_app.prepare_two_client_runtime_widget_defaults(
        selected_client=_profile(role="client", hidden_background=None),
        runtime_tts_voice_options=["marin", "cedar", "coral"],
    )

    assert session_state["client_prompt_text"] == defaults["common_prompt"]
    assert session_state["client_public_profile_text"] == defaults["common_profile"]
    assert session_state["client_a_prompt_text"] == (
        streamlit_app._join_two_client_default_parts(
            client_a_common_prompt,
            defaults["client_a_prompt"],
        )
    )
    assert session_state["client_b_prompt_text"] == (
        streamlit_app._join_two_client_default_parts(
            client_b_common_prompt,
            defaults["client_b_prompt"],
        )
    )
    assert session_state["client_a_prompt_text"].startswith(
        streamlit_app.TWO_CLIENT_A_PROMPT_INTRO
    )
    assert session_state["client_b_prompt_text"].startswith(
        streamlit_app.TWO_CLIENT_B_PROMPT_INTRO
    )
    assert "各発言の先頭に「妻：」と付けてください" not in session_state[
        "client_a_prompt_text"
    ]
    assert "各発言の先頭に「夫：」と付けてください" not in session_state[
        "client_b_prompt_text"
    ]
    assert "話者名や役割名を付けないでください" in session_state[
        "client_a_prompt_text"
    ]
    assert "話者名や役割名を付けないでください" in session_state[
        "client_b_prompt_text"
    ]
    assert session_state["client_a_public_profile_text"] == defaults[
        "client_a_profile"
    ]
    assert session_state["client_b_public_profile_text"] == defaults[
        "client_b_profile"
    ]
    assert (
        session_state["runtime_initial_client_a_transcript"]
        == defaults["client_a_initial_transcript"]
    )
    assert (
        session_state["runtime_initial_client_b_transcript"]
        == defaults["client_b_initial_transcript"]
    )


def test_prepare_two_client_runtime_widget_defaults_migrates_old_common_prompt_default(
    monkeypatch,
) -> None:
    defaults = streamlit_app.two_client_attachment_defaults()
    old_common_prompt = (
        streamlit_app.TWO_CLIENT_COMMON_PROMPT_INTRO
        + "\n\n古い共通プロンプトです。"
    )
    old_client_a_common_prompt = streamlit_app._participant_common_prompt(
        old_common_prompt,
        streamlit_app.TWO_CLIENT_A_PROMPT_INTRO,
    )
    old_client_b_common_prompt = streamlit_app._participant_common_prompt(
        old_common_prompt,
        streamlit_app.TWO_CLIENT_B_PROMPT_INTRO,
    )
    session_state = {
        "client_prompt_text": old_common_prompt,
        "client_public_profile_text": defaults["common_profile"],
        "runtime_initial_client_transcript": "",
        "runtime_two_client_widget_defaults_version": 2,
        "client_common_profile_applied_source_profile_id": "client-default",
        "client_common_profile_applied_prompt_text": old_common_prompt,
        "client_common_profile_applied_public_profile_text": defaults[
            "common_profile"
        ],
        "runtime_client_tts_voice": "cedar",
        "runtime_client_output_speed": 1.0,
        "runtime_client_audio_gain": 0.6,
        "client_a_display_name": "妻",
        "client_b_display_name": "夫",
        "client_a_prompt_source_profile_id": None,
        "client_b_prompt_source_profile_id": None,
        "client_a_prompt_text": streamlit_app._join_two_client_default_parts(
            old_client_a_common_prompt,
            defaults["client_a_prompt"],
        ),
        "client_b_prompt_text": streamlit_app._join_two_client_default_parts(
            old_client_b_common_prompt,
            defaults["client_b_prompt"],
        ),
        "client_a_public_profile_source_profile_id": "client-default",
        "client_b_public_profile_source_profile_id": "client-default",
        "client_a_public_profile_text": streamlit_app._join_two_client_default_parts(
            defaults["common_profile"],
            defaults["client_a_profile"],
        ),
        "client_b_public_profile_text": streamlit_app._join_two_client_default_parts(
            defaults["common_profile"],
            defaults["client_b_profile"],
        ),
        "runtime_initial_client_a_transcript": "",
        "runtime_initial_client_b_transcript": defaults[
            "client_b_initial_transcript"
        ],
        "runtime_client_a_tts_voice": "marin",
        "runtime_client_b_tts_voice": "cedar",
        "runtime_client_a_output_speed": 0.9,
        "runtime_client_b_output_speed": 1.0,
        "runtime_client_a_audio_gain": 0.6,
        "runtime_client_b_audio_gain": 0.6,
    }
    monkeypatch.setattr(streamlit_app, "st", SimpleNamespace(session_state=session_state))

    streamlit_app.prepare_two_client_runtime_widget_defaults(
        selected_client=_profile(role="client", hidden_background=None),
        runtime_tts_voice_options=["marin", "cedar", "coral"],
    )

    assert session_state["client_prompt_text"] == defaults["common_prompt"]
    assert "夫婦が互いの発言を聞き" in session_state["client_prompt_text"]
    assert "夫婦が互いの発言を聞き" in session_state["client_a_prompt_text"]
    assert "夫婦が互いの発言を聞き" in session_state["client_b_prompt_text"]
    assert (
        session_state["runtime_two_client_widget_defaults_version"]
        == streamlit_app.TWO_CLIENT_WIDGET_DEFAULTS_VERSION
    )


def test_prepare_two_client_runtime_widget_defaults_replaces_stale_widget_defaults(
    monkeypatch,
) -> None:
    session_state = {
        "client_prompt_text": "",
        "client_public_profile_text": "",
        "runtime_initial_client_transcript": "",
        "runtime_client_tts_voice": "",
        "runtime_client_output_speed": None,
        "runtime_client_audio_gain": None,
        "client_a_display_name": "クライアントA",
        "client_b_display_name": "クライアントB",
        "client_a_prompt_source_profile_id": None,
        "client_b_prompt_source_profile_id": None,
        "client_a_prompt_text": "",
        "client_b_prompt_text": "",
        "client_a_public_profile_source_profile_id": None,
        "client_b_public_profile_source_profile_id": None,
        "client_a_public_profile_text": "",
        "client_b_public_profile_text": "",
        "runtime_initial_client_a_transcript": "",
        "runtime_initial_client_b_transcript": "",
        "runtime_client_a_tts_voice": "alloy",
        "runtime_client_b_tts_voice": "alloy",
        "runtime_client_a_output_speed": 0.25,
        "runtime_client_b_output_speed": 0.25,
        "runtime_client_a_audio_gain": 0.1,
        "runtime_client_b_audio_gain": 0.1,
    }
    monkeypatch.setattr(streamlit_app, "st", SimpleNamespace(session_state=session_state))

    streamlit_app.prepare_two_client_runtime_widget_defaults(
        selected_client=_profile(role="client", hidden_background=None),
        runtime_tts_voice_options=["alloy", "marin", "cedar", "coral"],
    )

    assert session_state["client_a_display_name"] == "妻"
    assert session_state["client_b_display_name"] == "夫"
    assert session_state["runtime_client_a_tts_voice"] == "marin"
    assert session_state["runtime_client_b_tts_voice"] == "cedar"
    assert session_state["runtime_client_a_output_speed"] == 0.9
    assert session_state["runtime_client_b_output_speed"] == 1.0
    assert session_state["runtime_client_a_audio_gain"] == 0.6
    assert session_state["runtime_client_b_audio_gain"] == 0.6
    assert session_state["client_a_prompt_text"]
    assert session_state["client_b_prompt_text"]


def test_prepare_two_client_runtime_widget_defaults_migrates_obsolete_label_prompt(
    monkeypatch,
) -> None:
    session_state = {
        "client_prompt_text": "共通クライアントプロンプト",
        "client_public_profile_text": "共通プロフィール",
        "runtime_initial_client_transcript": "",
        "runtime_client_tts_voice": "",
        "runtime_client_output_speed": None,
        "runtime_client_audio_gain": None,
        "client_a_display_name": "妻",
        "client_b_display_name": "夫",
        "client_a_prompt_source_profile_id": None,
        "client_b_prompt_source_profile_id": None,
        "client_a_prompt_text": "前半\n\n各発言の先頭に「妻：」と付けてください。\n\n後半",
        "client_b_prompt_text": "前半\n\n各発言の先頭に「夫：」と付けてください。\n\n後半",
        "client_a_public_profile_source_profile_id": None,
        "client_b_public_profile_source_profile_id": None,
        "client_a_public_profile_text": "",
        "client_b_public_profile_text": "",
        "runtime_initial_client_a_transcript": "",
        "runtime_initial_client_b_transcript": "",
        "runtime_client_a_tts_voice": "",
        "runtime_client_b_tts_voice": "",
        "runtime_client_a_output_speed": None,
        "runtime_client_b_output_speed": None,
        "runtime_client_a_audio_gain": None,
        "runtime_client_b_audio_gain": None,
    }
    monkeypatch.setattr(streamlit_app, "st", SimpleNamespace(session_state=session_state))

    streamlit_app.prepare_two_client_runtime_widget_defaults(
        selected_client=_profile(role="client", hidden_background=None),
        runtime_tts_voice_options=["alloy", "marin", "cedar", "coral"],
    )

    assert "各発言の先頭に「妻：」と付けてください" not in session_state[
        "client_a_prompt_text"
    ]
    assert "各発言の先頭に「夫：」と付けてください" not in session_state[
        "client_b_prompt_text"
    ]
    assert "妻、」などの話者名や役割名を付けないでください" in session_state[
        "client_a_prompt_text"
    ]
    assert "夫、」などの話者名や役割名を付けないでください" in session_state[
        "client_b_prompt_text"
    ]


def test_prepare_two_client_runtime_widget_defaults_replaces_one_client_defaults(
    monkeypatch,
) -> None:
    defaults = streamlit_app.two_client_attachment_defaults()
    client_a_common_prompt = streamlit_app._participant_common_prompt(
        defaults["common_prompt"],
        streamlit_app.TWO_CLIENT_A_PROMPT_INTRO,
    )
    client_b_common_prompt = streamlit_app._participant_common_prompt(
        defaults["common_prompt"],
        streamlit_app.TWO_CLIENT_B_PROMPT_INTRO,
    )
    counselor = _profile(role="counselor", hidden_background=None)
    client = _profile(role="client", hidden_background=None)
    session_state = streamlit_app.initial_ui_state()
    monkeypatch.setattr(streamlit_app, "st", SimpleNamespace(session_state=session_state))

    prepare_runtime_text_defaults(counselor, client)
    assert (
        session_state["client_a_prompt_text"]
        == streamlit_app.DEFAULT_RUNTIME_CLIENT_PROMPT
    )
    assert (
        session_state["runtime_initial_client_a_transcript"]
        == defaults["client_a_initial_transcript"]
    )
    assert (
        session_state["runtime_initial_client_b_transcript"]
        == defaults["client_b_initial_transcript"]
    )

    streamlit_app.prepare_two_client_runtime_widget_defaults(
        selected_client=client,
        runtime_tts_voice_options=["alloy", "marin", "cedar", "coral"],
    )

    assert session_state["client_a_display_name"] == "妻"
    assert session_state["client_b_display_name"] == "夫"
    assert session_state["client_prompt_text"] == defaults["common_prompt"]
    assert session_state["client_public_profile_text"] == defaults["common_profile"]
    assert session_state["client_a_prompt_text"] == (
        streamlit_app._join_two_client_default_parts(
            client_a_common_prompt,
            defaults["client_a_prompt"],
        )
    )
    assert session_state["client_b_prompt_text"] == (
        streamlit_app._join_two_client_default_parts(
            client_b_common_prompt,
            defaults["client_b_prompt"],
        )
    )
    assert session_state["runtime_initial_client_a_transcript"] == (
        defaults["client_a_initial_transcript"]
    )
    assert session_state["runtime_initial_client_b_transcript"] == (
        defaults["client_b_initial_transcript"]
    )


def test_prepare_two_client_runtime_widget_defaults_migrates_stuck_widget_values(
    monkeypatch,
) -> None:
    session_state = {
        "client_prompt_text": "",
        "client_public_profile_text": "",
        "runtime_initial_client_transcript": (
            streamlit_app.DEFAULT_RUNTIME_INITIAL_CLIENT_TRANSCRIPT
        ),
        "runtime_two_client_widget_defaults_version": (
            streamlit_app.TWO_CLIENT_WIDGET_DEFAULTS_VERSION
        ),
        "runtime_client_tts_voice": "alloy",
        "runtime_client_output_speed": 0.25,
        "runtime_client_audio_gain": 0.1,
        "client_a_display_name": "",
        "client_b_display_name": "",
        "client_a_display_name_user_modified": True,
        "client_b_display_name_user_modified": True,
        "client_a_display_name_default_applied": True,
        "client_b_display_name_default_applied": True,
        "client_a_prompt_source_profile_id": None,
        "client_b_prompt_source_profile_id": None,
        "client_a_prompt_text": streamlit_app.DEFAULT_RUNTIME_CLIENT_PROMPT,
        "client_b_prompt_text": streamlit_app.DEFAULT_RUNTIME_CLIENT_PROMPT,
        "client_a_public_profile_source_profile_id": None,
        "client_b_public_profile_source_profile_id": None,
        "client_a_public_profile_text": "",
        "client_b_public_profile_text": "",
        "runtime_initial_client_a_transcript": (
            streamlit_app.DEFAULT_RUNTIME_INITIAL_CLIENT_TRANSCRIPT
        ),
        "runtime_initial_client_b_transcript": (
            streamlit_app.DEFAULT_RUNTIME_INITIAL_CLIENT_TRANSCRIPT
        ),
        "runtime_initial_client_a_transcript_user_modified": True,
        "runtime_initial_client_b_transcript_user_modified": True,
        "runtime_initial_client_a_transcript_default_applied": True,
        "runtime_initial_client_b_transcript_default_applied": True,
        "runtime_client_a_tts_voice": "alloy",
        "runtime_client_b_tts_voice": "alloy",
        "runtime_client_a_tts_voice_user_modified": True,
        "runtime_client_b_tts_voice_user_modified": True,
        "runtime_client_a_tts_voice_default_applied": True,
        "runtime_client_b_tts_voice_default_applied": True,
        "runtime_client_a_output_speed": 0.25,
        "runtime_client_b_output_speed": 0.25,
        "runtime_client_a_output_speed_user_modified": True,
        "runtime_client_b_output_speed_user_modified": True,
        "runtime_client_a_output_speed_default_applied": True,
        "runtime_client_b_output_speed_default_applied": True,
        "runtime_client_a_audio_gain": 0.1,
        "runtime_client_b_audio_gain": 0.1,
        "runtime_client_a_audio_gain_user_modified": True,
        "runtime_client_b_audio_gain_user_modified": True,
        "runtime_client_a_audio_gain_default_applied": True,
        "runtime_client_b_audio_gain_default_applied": True,
    }
    monkeypatch.setattr(streamlit_app, "st", SimpleNamespace(session_state=session_state))

    streamlit_app.prepare_two_client_runtime_widget_defaults(
        selected_client=_profile(role="client", hidden_background=None),
        runtime_tts_voice_options=["alloy", "marin", "cedar", "coral"],
    )

    assert session_state["client_a_display_name"] == "妻"
    assert session_state["client_b_display_name"] == "夫"
    assert session_state["runtime_client_a_tts_voice"] == "marin"
    assert session_state["runtime_client_b_tts_voice"] == "cedar"
    assert session_state["runtime_client_a_output_speed"] == 0.9
    assert session_state["runtime_client_b_output_speed"] == 1.0
    assert session_state["runtime_client_a_audio_gain"] == 0.6
    assert session_state["runtime_client_b_audio_gain"] == 0.6
    assert (
        session_state["runtime_two_client_widget_defaults_version"]
        == streamlit_app.TWO_CLIENT_WIDGET_DEFAULTS_VERSION
    )


def test_prepare_two_client_runtime_widget_defaults_keeps_user_modified_widget_values(
    monkeypatch,
) -> None:
    session_state = {
        "client_prompt_text": "",
        "client_public_profile_text": "",
        "runtime_initial_client_transcript": (
            streamlit_app.DEFAULT_RUNTIME_INITIAL_CLIENT_TRANSCRIPT
        ),
        "runtime_client_tts_voice": "alloy",
        "runtime_client_output_speed": 0.25,
        "runtime_client_audio_gain": 0.1,
        "client_a_display_name": "",
        "client_b_display_name": "",
        "client_a_display_name_persist_user_modified": True,
        "client_b_display_name_persist_user_modified": True,
        "client_a_prompt_source_profile_id": None,
        "client_b_prompt_source_profile_id": None,
        "client_a_prompt_text": streamlit_app.DEFAULT_RUNTIME_CLIENT_PROMPT,
        "client_b_prompt_text": streamlit_app.DEFAULT_RUNTIME_CLIENT_PROMPT,
        "client_a_public_profile_source_profile_id": None,
        "client_b_public_profile_source_profile_id": None,
        "client_a_public_profile_text": "",
        "client_b_public_profile_text": "",
        "runtime_initial_client_a_transcript": (
            streamlit_app.DEFAULT_RUNTIME_INITIAL_CLIENT_TRANSCRIPT
        ),
        "runtime_initial_client_b_transcript": (
            streamlit_app.DEFAULT_RUNTIME_INITIAL_CLIENT_TRANSCRIPT
        ),
        "runtime_initial_client_a_transcript_persist_user_modified": True,
        "runtime_initial_client_b_transcript_persist_user_modified": True,
        "runtime_client_a_tts_voice": "alloy",
        "runtime_client_b_tts_voice": "alloy",
        "runtime_client_a_tts_voice_persist_user_modified": True,
        "runtime_client_b_tts_voice_persist_user_modified": True,
        "runtime_client_a_output_speed": 0.25,
        "runtime_client_b_output_speed": 0.25,
        "runtime_client_a_output_speed_persist_user_modified": True,
        "runtime_client_b_output_speed_persist_user_modified": True,
        "runtime_client_a_audio_gain": 0.1,
        "runtime_client_b_audio_gain": 0.1,
        "runtime_client_a_audio_gain_persist_user_modified": True,
        "runtime_client_b_audio_gain_persist_user_modified": True,
    }
    monkeypatch.setattr(streamlit_app, "st", SimpleNamespace(session_state=session_state))

    streamlit_app.prepare_two_client_runtime_widget_defaults(
        selected_client=_profile(role="client", hidden_background=None),
        runtime_tts_voice_options=["alloy", "marin", "cedar", "coral"],
    )

    assert session_state["client_a_display_name"] == ""
    assert session_state["client_b_display_name"] == ""
    assert session_state["runtime_client_a_tts_voice"] == "alloy"
    assert session_state["runtime_client_b_tts_voice"] == "alloy"
    assert session_state["runtime_client_a_output_speed"] == 0.25
    assert session_state["runtime_client_b_output_speed"] == 0.25
    assert session_state["runtime_client_a_audio_gain"] == 0.1
    assert session_state["runtime_client_b_audio_gain"] == 0.1


def test_prepare_one_client_runtime_widget_defaults_replaces_stale_defaults(
    monkeypatch,
) -> None:
    session_state = {
        "runtime_one_client_widget_defaults_version": (
            streamlit_app.ONE_CLIENT_WIDGET_DEFAULTS_VERSION
        ),
        "client_display_name": "",
        "client_display_name_persist_user_modified": True,
        "runtime_client_tts_voice": "alloy",
        "runtime_client_tts_voice_persist_user_modified": True,
        "runtime_client_output_speed": 0.25,
        "runtime_client_output_speed_persist_user_modified": True,
        "runtime_client_audio_gain": 0.1,
        "runtime_client_audio_gain_persist_user_modified": True,
        "runtime_force_stop_after_closing_turns": 3,
        "runtime_one_client_force_stop_after_closing_turns": 3,
    }
    monkeypatch.setattr(streamlit_app, "st", SimpleNamespace(session_state=session_state))

    streamlit_app.prepare_one_client_runtime_widget_defaults(
        ["alloy", "marin", "cedar", "coral"]
    )

    assert session_state["client_display_name"] == "妻"
    assert session_state["runtime_client_tts_voice"] == "marin"
    assert session_state["runtime_client_output_speed"] == 0.9
    assert session_state["runtime_client_audio_gain"] == 0.6
    assert session_state["runtime_force_stop_after_closing_turns"] == 3
    assert session_state["runtime_one_client_force_stop_after_closing_turns"] == 3
    assert (
        session_state["runtime_one_client_widget_defaults_version"]
        == streamlit_app.ONE_CLIENT_WIDGET_DEFAULTS_VERSION
    )


def test_prepare_one_client_runtime_widget_defaults_keeps_custom_values(
    monkeypatch,
) -> None:
    session_state = {
        "runtime_one_client_widget_defaults_version": 0,
        "client_display_name": "相談者",
        "runtime_client_tts_voice": "coral",
        "runtime_client_output_speed": 1.2,
        "runtime_client_audio_gain": 0.8,
        "runtime_force_stop_after_closing_turns": 4,
        "runtime_one_client_force_stop_after_closing_turns": 4,
    }
    monkeypatch.setattr(streamlit_app, "st", SimpleNamespace(session_state=session_state))

    streamlit_app.prepare_one_client_runtime_widget_defaults(
        ["alloy", "marin", "cedar", "coral"]
    )

    assert session_state["client_display_name"] == "相談者"
    assert session_state["runtime_client_tts_voice"] == "coral"
    assert session_state["runtime_client_output_speed"] == 1.2
    assert session_state["runtime_client_audio_gain"] == 0.8
    assert session_state["runtime_force_stop_after_closing_turns"] == 4
    assert session_state["runtime_one_client_force_stop_after_closing_turns"] == 4


def test_prepare_one_client_runtime_widget_defaults_keeps_current_widget_values(
    monkeypatch,
) -> None:
    session_state = {
        "runtime_one_client_widget_defaults_version": (
            streamlit_app.ONE_CLIENT_WIDGET_DEFAULTS_VERSION
        ),
        "client_display_name": "相談者",
        "client_display_name_widget": "相談者",
        "client_display_name_persist_user_modified": True,
        "runtime_client_tts_voice": "alloy",
        "runtime_client_tts_voice_widget": "alloy",
        "runtime_client_tts_voice_persist_user_modified": True,
        "runtime_client_output_speed": 0.25,
        "runtime_client_output_speed_widget": 0.25,
        "runtime_client_output_speed_persist_user_modified": True,
        "runtime_client_audio_gain": 0.1,
        "runtime_client_audio_gain_widget": 0.1,
        "runtime_client_audio_gain_persist_user_modified": True,
        "runtime_force_stop_after_closing_turns": 4,
        "runtime_one_client_force_stop_after_closing_turns": 4,
    }
    monkeypatch.setattr(streamlit_app, "st", SimpleNamespace(session_state=session_state))

    streamlit_app.prepare_one_client_runtime_widget_defaults(
        ["alloy", "marin", "cedar", "coral"]
    )

    assert session_state["client_display_name"] == "相談者"
    assert session_state["runtime_client_tts_voice"] == "alloy"
    assert session_state["runtime_client_output_speed"] == 0.25
    assert session_state["runtime_client_audio_gain"] == 0.1
    assert session_state["runtime_force_stop_after_closing_turns"] == 4
    assert session_state["runtime_one_client_force_stop_after_closing_turns"] == 4


def test_prepare_one_client_runtime_widget_defaults_keeps_user_modified_values(
    monkeypatch,
) -> None:
    session_state = {
        "runtime_one_client_widget_defaults_version": 0,
        "client_display_name": "妻",
        "client_display_name_persist_user_modified": True,
        "runtime_client_tts_voice": "coral",
        "runtime_client_tts_voice_persist_user_modified": True,
        "runtime_client_output_speed": 1.2,
        "runtime_client_output_speed_persist_user_modified": True,
        "runtime_client_audio_gain": 0.7,
        "runtime_client_audio_gain_persist_user_modified": True,
        "runtime_force_stop_after_closing_turns": 3,
        "runtime_one_client_force_stop_after_closing_turns": 3,
        "runtime_one_client_force_stop_after_closing_turns_persist_user_modified": True,
    }
    monkeypatch.setattr(streamlit_app, "st", SimpleNamespace(session_state=session_state))

    streamlit_app.prepare_one_client_runtime_widget_defaults(
        ["alloy", "marin", "cedar", "coral"]
    )

    assert session_state["client_display_name"] == "妻"
    assert session_state["runtime_client_tts_voice"] == "coral"
    assert session_state["runtime_client_output_speed"] == 1.2
    assert session_state["runtime_client_audio_gain"] == 0.7
    assert session_state["runtime_force_stop_after_closing_turns"] == 3
    assert session_state["runtime_one_client_force_stop_after_closing_turns"] == 3


class _UploadedPromptFile:
    def __init__(self, name: str, content: bytes) -> None:
        self.name = name
        self.size = len(content)
        self._content = content

    def getvalue(self) -> bytes:
        return self._content


def test_uploaded_prompt_file_updates_prompt_text_once(monkeypatch) -> None:
    session_state = {
        "counselor_prompt_text": "profile prompt",
        "counselor_prompt_source_profile_id": "profile-1",
        "counselor_prompt_file_signature": None,
    }
    monkeypatch.setattr(streamlit_app, "st", SimpleNamespace(session_state=session_state))
    uploaded_file = _UploadedPromptFile(
        "counselor.md",
        "ファイルから読み込んだプロンプト".encode("utf-8"),
    )

    error = streamlit_app.apply_uploaded_prompt_file_to_session(
        uploaded_file,
        prompt_text_key="counselor_prompt_text",
        source_profile_key="counselor_prompt_source_profile_id",
        file_signature_key="counselor_prompt_file_signature",
        source_profile_id="profile-1",
    )
    session_state["counselor_prompt_text"] = "手で編集したプロンプト"
    second_error = streamlit_app.apply_uploaded_prompt_file_to_session(
        uploaded_file,
        prompt_text_key="counselor_prompt_text",
        source_profile_key="counselor_prompt_source_profile_id",
        file_signature_key="counselor_prompt_file_signature",
        source_profile_id="profile-1",
    )

    assert error is None
    assert second_error is None
    assert session_state["counselor_prompt_text"] == "手で編集したプロンプト"
    assert session_state["counselor_prompt_source_profile_id"] == "profile-1"
    assert session_state["counselor_prompt_file_signature"].startswith(
        "profile-1:counselor.md:"
    )


def test_uploaded_profile_file_updates_profile_text_without_touching_preset(
    monkeypatch,
) -> None:
    session_state = {
        "client_public_profile_text": "preset profile",
        "client_public_profile_source_profile_id": "client-profile-1",
        "client_public_profile_file_signature": None,
    }
    monkeypatch.setattr(streamlit_app, "st", SimpleNamespace(session_state=session_state))
    uploaded_file = _UploadedPromptFile(
        "client_profile.md",
        "ファイルから読み込んだプロフィール".encode("utf-8"),
    )

    error = streamlit_app.apply_uploaded_prompt_file_to_session(
        uploaded_file,
        prompt_text_key="client_public_profile_text",
        source_profile_key="client_public_profile_source_profile_id",
        file_signature_key="client_public_profile_file_signature",
        source_profile_id="client-profile-1",
    )

    assert error is None
    assert session_state["client_public_profile_text"] == (
        "ファイルから読み込んだプロフィール"
    )
    assert session_state["client_public_profile_source_profile_id"] == (
        "client-profile-1"
    )
    assert session_state["client_public_profile_file_signature"].startswith(
        "client-profile-1:client_profile.md:"
    )


def test_uploaded_prompt_file_reports_utf8_decode_error(monkeypatch) -> None:
    session_state = {
        "client_prompt_text": "profile prompt",
        "client_prompt_source_profile_id": "profile-1",
        "client_prompt_file_signature": None,
    }
    monkeypatch.setattr(streamlit_app, "st", SimpleNamespace(session_state=session_state))

    error = streamlit_app.apply_uploaded_prompt_file_to_session(
        _UploadedPromptFile("client.txt", b"\xff"),
        prompt_text_key="client_prompt_text",
        source_profile_key="client_prompt_source_profile_id",
        file_signature_key="client_prompt_file_signature",
        source_profile_id="profile-1",
    )

    assert error == "UTF-8 の .md / .txt として読み込めませんでした。"
    assert session_state["client_prompt_text"] == "profile prompt"
    assert session_state["client_prompt_file_signature"] is None


def test_prepare_runtime_session_defaults_migrates_legacy_closing_start_default(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        streamlit_app,
        "runtime_speaker_gain_defaults",
        lambda: {"counselor": 0.6, "client": 0.6},
    )
    monkeypatch.setattr(
        streamlit_app,
        "runtime_voice_defaults",
        lambda: {"counselor": "shimmer", "client": "cedar"},
    )
    monkeypatch.setattr(
        streamlit_app,
        "runtime_output_speed_defaults",
        lambda: {"counselor": 1.05, "client": 0.9},
    )
    session_state = {
        "session_status": SessionStatus.IDLE.value,
        "runtime_participant_mode": "one_client",
        "runtime_realtime_model": "gpt-realtime",
        "runtime_prompting_llm_model": "unsupported-prompting-model",
        "runtime_prompting_llm_reasoning_effort": "minimal",
        "runtime_timing_llm_model": "unsupported-timing-model",
        "runtime_timing_llm_reasoning_effort": "minimal",
        "runtime_summary_llm_model": "unsupported-summary-model",
        "runtime_summary_llm_reasoning_effort": "minimal",
        "runtime_counselor_tts_voice": "",
        "runtime_client_tts_voice": "",
        "runtime_counselor_output_speed": None,
        "runtime_client_output_speed": None,
        "runtime_closing_start_seconds": 900,
        "runtime_force_stop_after_closing_turns": None,
        "runtime_counselor_audio_gain": None,
        "runtime_client_audio_gain": None,
    }
    monkeypatch.setattr(streamlit_app, "st", SimpleNamespace(session_state=session_state))

    prepare_runtime_session_defaults(
        SimpleNamespace(
            app=SimpleNamespace(closing_start_audio_seconds=120, farewell_after_turns=3),
            openai=SimpleNamespace(allowed_tts_voices=["shimmer", "cedar"]),
        )
    )

    assert session_state["runtime_closing_start_seconds"] == 120
    assert session_state["runtime_force_stop_after_closing_turns"] == 3
    assert session_state["runtime_one_client_force_stop_after_closing_turns"] == 3
    assert session_state["runtime_realtime_model"] == "gpt-realtime-2.1"
    assert session_state["runtime_prompting_llm_model"] == "gpt-5.4-mini"
    assert session_state["runtime_prompting_llm_reasoning_effort"] == "low"
    assert session_state["runtime_timing_llm_model"] == "gpt-5.4-mini"
    assert session_state["runtime_timing_llm_reasoning_effort"] == "none"
    assert session_state["runtime_summary_llm_model"] == "gpt-5.4-mini"
    assert session_state["runtime_summary_llm_reasoning_effort"] == "none"
    assert session_state["runtime_counselor_tts_voice"] == "shimmer"
    assert session_state["runtime_client_tts_voice"] == "cedar"
    assert session_state["runtime_counselor_output_speed"] == 1.05
    assert session_state["runtime_client_output_speed"] == 0.9
    assert session_state["runtime_counselor_audio_gain"] == 0.6
    assert session_state["runtime_client_audio_gain"] == 0.6


def test_closing_start_widget_keeps_user_selected_900_seconds_after_rerun() -> None:
    from streamlit.testing.v1 import AppTest

    view = AppTest.from_string(
        "import streamlit as st\n"
        "from types import SimpleNamespace\n"
        "from app.streamlit_app import (initial_ui_state, "
        "prepare_runtime_session_defaults, render_runtime_common_session_controls)\n"
        "for key, value in initial_ui_state().items():\n"
        "    st.session_state.setdefault(key, value)\n"
        "config = SimpleNamespace(\n"
        "    app=SimpleNamespace(closing_start_audio_seconds=120, farewell_after_turns=3),\n"
        "    openai=SimpleNamespace(allowed_tts_voices=['shimmer', 'marin', 'cedar']),\n"
        ")\n"
        "prepare_runtime_session_defaults(config)\n"
        "render_runtime_common_session_controls()\n"
    ).run()
    assert not view.exception
    assert view.number_input(key="runtime_closing_start_seconds").value == 120

    view.number_input(key="runtime_closing_start_seconds").set_value(900).run()
    assert not view.exception
    assert view.number_input(key="runtime_closing_start_seconds").value == 900

    # Editing a different setting and later reruns must preserve the selection.
    view.number_input(key="runtime_force_stop_after_closing_turns").set_value(4).run()
    view.run()
    assert not view.exception
    assert view.number_input(key="runtime_closing_start_seconds").value == 900
    assert normalize_runtime_start_options(
        {
            "closing_start_elapsed_seconds": view.session_state[
                "runtime_closing_start_seconds"
            ]
        }
    )["closing_start_elapsed_seconds"] == 900.0


@pytest.mark.parametrize(
    ("legacy_speed", "user_modified"),
    [(0.25, False), (0.6, False), (0.6, True), (0.9, False)],
)
def test_prepare_runtime_session_defaults_migrates_legacy_counselor_speed(
    monkeypatch,
    legacy_speed: float,
    user_modified: bool,
) -> None:
    monkeypatch.setattr(
        streamlit_app,
        "runtime_speaker_gain_defaults",
        lambda: {"counselor": 0.6, "client": 0.6},
    )
    monkeypatch.setattr(
        streamlit_app,
        "runtime_voice_defaults",
        lambda: {"counselor": "shimmer", "client": "marin"},
    )
    monkeypatch.setattr(
        streamlit_app,
        "runtime_output_speed_defaults",
        lambda: {"counselor": 1.05, "client": 0.9},
    )
    session_state = initial_ui_state()
    session_state["runtime_counselor_output_speed"] = legacy_speed
    if user_modified:
        session_state["runtime_counselor_output_speed_persist_user_modified"] = True
    monkeypatch.setattr(streamlit_app, "st", SimpleNamespace(session_state=session_state))

    prepare_runtime_session_defaults(
        SimpleNamespace(
            app=SimpleNamespace(closing_start_audio_seconds=120, farewell_after_turns=3),
            openai=SimpleNamespace(allowed_tts_voices=["shimmer", "marin"]),
        )
    )

    assert session_state["runtime_counselor_output_speed"] == 1.05
    assert (
        session_state["runtime_output_speed_defaults_version"]
        == streamlit_app.REALTIME_OUTPUT_SPEED_DEFAULTS_VERSION
    )


@pytest.mark.parametrize(
    ("user_modified", "expected_gain"),
    [(False, 0.6), (True, 0.1)],
)
def test_prepare_runtime_session_defaults_migrates_stale_counselor_gain_only_when_unmodified(
    monkeypatch,
    user_modified: bool,
    expected_gain: float,
) -> None:
    monkeypatch.setattr(
        streamlit_app,
        "runtime_speaker_gain_defaults",
        lambda: {"counselor": 0.6, "client": 0.6},
    )
    monkeypatch.setattr(
        streamlit_app,
        "runtime_voice_defaults",
        lambda: {"counselor": "shimmer", "client": "marin"},
    )
    monkeypatch.setattr(
        streamlit_app,
        "runtime_output_speed_defaults",
        lambda: {"counselor": 0.9, "client": 0.9},
    )
    session_state = initial_ui_state()
    session_state["runtime_counselor_audio_gain"] = 0.1
    if user_modified:
        session_state["runtime_counselor_audio_gain_persist_user_modified"] = True
    monkeypatch.setattr(streamlit_app, "st", SimpleNamespace(session_state=session_state))

    prepare_runtime_session_defaults(
        SimpleNamespace(
            app=SimpleNamespace(closing_start_audio_seconds=120, farewell_after_turns=3),
            openai=SimpleNamespace(allowed_tts_voices=["shimmer", "marin"]),
        )
    )

    assert session_state["runtime_counselor_audio_gain"] == expected_gain


@pytest.mark.parametrize("selected_speed", [0.25, 0.6, 0.9])
def test_prepare_runtime_session_defaults_keeps_user_selected_counselor_speed_after_migration(
    monkeypatch,
    selected_speed: float,
) -> None:
    monkeypatch.setattr(
        streamlit_app,
        "runtime_speaker_gain_defaults",
        lambda: {"counselor": 0.6, "client": 0.6},
    )
    monkeypatch.setattr(
        streamlit_app,
        "runtime_voice_defaults",
        lambda: {"counselor": "shimmer", "client": "marin"},
    )
    monkeypatch.setattr(
        streamlit_app,
        "runtime_output_speed_defaults",
        lambda: {"counselor": 1.05, "client": 0.9},
    )
    session_state = initial_ui_state()
    session_state.update(
        {
            "runtime_counselor_output_speed": selected_speed,
            "runtime_counselor_output_speed_persist_user_modified": True,
            "runtime_output_speed_defaults_version": (
                streamlit_app.REALTIME_OUTPUT_SPEED_DEFAULTS_VERSION
            ),
        }
    )
    monkeypatch.setattr(streamlit_app, "st", SimpleNamespace(session_state=session_state))

    prepare_runtime_session_defaults(
        SimpleNamespace(
            app=SimpleNamespace(closing_start_audio_seconds=120, farewell_after_turns=3),
            openai=SimpleNamespace(allowed_tts_voices=["shimmer", "marin"]),
        )
    )

    assert session_state["runtime_counselor_output_speed"] == selected_speed


def test_prepare_runtime_session_defaults_clamps_legacy_realtime_speeds_to_api_max(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        streamlit_app,
        "runtime_speaker_gain_defaults",
        lambda: {"counselor": 0.6, "client": 0.6},
    )
    monkeypatch.setattr(
        streamlit_app,
        "runtime_voice_defaults",
        lambda: {"counselor": "shimmer", "client": "marin"},
    )
    monkeypatch.setattr(
        streamlit_app,
        "runtime_output_speed_defaults",
        lambda: {"counselor": 0.6, "client": 0.9},
    )
    session_state = initial_ui_state()
    session_state.update(
        {
            "runtime_counselor_output_speed": 2.0,
            "runtime_client_output_speed": 4.0,
            "runtime_client_a_output_speed": 3.0,
            "runtime_client_b_output_speed": 1.51,
            "runtime_client_output_speed_persist_user_modified": True,
        }
    )
    monkeypatch.setattr(streamlit_app, "st", SimpleNamespace(session_state=session_state))

    prepare_runtime_session_defaults(
        SimpleNamespace(
            app=SimpleNamespace(closing_start_audio_seconds=120, farewell_after_turns=3),
            openai=SimpleNamespace(allowed_tts_voices=["shimmer", "marin"]),
        )
    )

    assert session_state["runtime_counselor_output_speed"] == 1.5
    assert session_state["runtime_client_output_speed"] == 1.5
    assert session_state["runtime_client_a_output_speed"] == 1.5
    assert session_state["runtime_client_b_output_speed"] == 1.5
    assert session_state["runtime_client_output_speed_persist_user_modified"] is True


def test_prepare_runtime_session_defaults_migrates_legacy_user_modified_closing_turns(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        streamlit_app,
        "runtime_speaker_gain_defaults",
        lambda: {"counselor": 0.6, "client": 0.6},
    )
    monkeypatch.setattr(
        streamlit_app,
        "runtime_voice_defaults",
        lambda: {"counselor": "shimmer", "client": "cedar"},
    )
    monkeypatch.setattr(
        streamlit_app,
        "runtime_output_speed_defaults",
        lambda: {"counselor": 0.9, "client": 0.9},
    )
    session_state = initial_ui_state()
    session_state.update(
        {
            "runtime_participant_mode": "one_client",
            "runtime_force_stop_after_closing_turns": 5,
            "runtime_force_stop_after_closing_turns_persist_user_modified": True,
            "runtime_counselor_tts_voice": "",
            "runtime_client_tts_voice": "",
        }
    )
    monkeypatch.setattr(streamlit_app, "st", SimpleNamespace(session_state=session_state))

    prepare_runtime_session_defaults(
        SimpleNamespace(
            app=SimpleNamespace(closing_start_audio_seconds=120, farewell_after_turns=3),
            openai=SimpleNamespace(allowed_tts_voices=["shimmer", "cedar"]),
        )
    )

    assert session_state["runtime_force_stop_after_closing_turns"] == 5
    assert session_state["runtime_one_client_force_stop_after_closing_turns"] == 5
    assert session_state["runtime_two_client_force_stop_after_closing_turns"] == 3
    assert (
        session_state[
            "runtime_one_client_force_stop_after_closing_turns_persist_user_modified"
        ]
        is True
    )


def test_prepare_runtime_session_defaults_migrates_legacy_two_client_closing_turns(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        streamlit_app,
        "runtime_speaker_gain_defaults",
        lambda: {"counselor": 0.6, "client": 0.6},
    )
    monkeypatch.setattr(
        streamlit_app,
        "runtime_voice_defaults",
        lambda: {"counselor": "shimmer", "client": "cedar"},
    )
    monkeypatch.setattr(
        streamlit_app,
        "runtime_output_speed_defaults",
        lambda: {"counselor": 0.9, "client": 0.9},
    )
    session_state = initial_ui_state()
    session_state.update(
        {
            "runtime_participant_mode": "two_clients",
            "runtime_force_stop_after_closing_turns": 5,
            "runtime_force_stop_after_closing_turns_persist_user_modified": True,
        }
    )
    monkeypatch.setattr(streamlit_app, "st", SimpleNamespace(session_state=session_state))

    prepare_runtime_session_defaults(
        SimpleNamespace(
            app=SimpleNamespace(closing_start_audio_seconds=120, farewell_after_turns=3),
            openai=SimpleNamespace(allowed_tts_voices=["shimmer", "cedar"]),
        )
    )
    streamlit_app.prepare_two_client_runtime_session_defaults(
        SimpleNamespace(app=SimpleNamespace(farewell_after_turns=2))
    )

    assert session_state["runtime_force_stop_after_closing_turns"] == 5
    assert session_state["runtime_one_client_force_stop_after_closing_turns"] == 3
    assert session_state["runtime_two_client_force_stop_after_closing_turns"] == 5
    assert (
        session_state[
            "runtime_two_client_force_stop_after_closing_turns_persist_user_modified"
        ]
        is True
    )


def test_sync_runtime_closing_turns_writes_legacy_fallback_to_mode_key(
    monkeypatch,
) -> None:
    session_state = {
        "runtime_participant_mode": "one_client",
        "runtime_force_stop_after_closing_turns": 5,
        "runtime_one_client_force_stop_after_closing_turns": None,
        "runtime_two_client_force_stop_after_closing_turns": 3,
    }
    monkeypatch.setattr(streamlit_app, "st", SimpleNamespace(session_state=session_state))

    streamlit_app.sync_runtime_force_stop_after_closing_turns_for_mode("one_client")

    assert session_state["runtime_force_stop_after_closing_turns"] == 5
    assert session_state["runtime_one_client_force_stop_after_closing_turns"] == 5
    assert session_state["runtime_two_client_force_stop_after_closing_turns"] == 3


def test_prepare_two_client_runtime_session_defaults_uses_three_closing_turns(
    monkeypatch,
) -> None:
    session_state = {
        "runtime_force_stop_after_closing_turns": 2,
        "runtime_one_client_force_stop_after_closing_turns": 2,
        "runtime_two_client_force_stop_after_closing_turns": None,
    }
    monkeypatch.setattr(streamlit_app, "st", SimpleNamespace(session_state=session_state))

    streamlit_app.prepare_two_client_runtime_session_defaults(
        SimpleNamespace(app=SimpleNamespace(farewell_after_turns=2))
    )

    assert session_state["runtime_force_stop_after_closing_turns"] == 3
    assert session_state["runtime_one_client_force_stop_after_closing_turns"] == 2
    assert session_state["runtime_two_client_force_stop_after_closing_turns"] == 3


def test_prepare_two_client_runtime_session_defaults_keeps_two_client_user_modified_closing_turns(
    monkeypatch,
) -> None:
    session_state = {
        "runtime_force_stop_after_closing_turns": 2,
        "runtime_one_client_force_stop_after_closing_turns": 2,
        "runtime_two_client_force_stop_after_closing_turns": 4,
        "runtime_two_client_force_stop_after_closing_turns_persist_user_modified": True,
    }
    monkeypatch.setattr(streamlit_app, "st", SimpleNamespace(session_state=session_state))

    streamlit_app.prepare_two_client_runtime_session_defaults(
        SimpleNamespace(app=SimpleNamespace(farewell_after_turns=2))
    )

    assert session_state["runtime_force_stop_after_closing_turns"] == 4
    assert session_state["runtime_one_client_force_stop_after_closing_turns"] == 2


def test_prepare_two_client_runtime_session_defaults_keeps_custom_two_client_value(
    monkeypatch,
) -> None:
    session_state = {
        "runtime_force_stop_after_closing_turns": 2,
        "runtime_one_client_force_stop_after_closing_turns": 2,
        "runtime_two_client_force_stop_after_closing_turns": 4,
    }
    monkeypatch.setattr(streamlit_app, "st", SimpleNamespace(session_state=session_state))

    streamlit_app.prepare_two_client_runtime_session_defaults(
        SimpleNamespace(app=SimpleNamespace(farewell_after_turns=2))
    )

    assert session_state["runtime_force_stop_after_closing_turns"] == 4
    assert session_state["runtime_one_client_force_stop_after_closing_turns"] == 2
    assert session_state["runtime_two_client_force_stop_after_closing_turns"] == 4


def test_runtime_closing_turn_defaults_are_independent_between_participant_modes(
    monkeypatch,
) -> None:
    session_state = {
        "runtime_participant_mode": "one_client",
        "runtime_force_stop_after_closing_turns": 2,
        "runtime_one_client_force_stop_after_closing_turns": 2,
        "runtime_two_client_force_stop_after_closing_turns": 3,
    }
    monkeypatch.setattr(streamlit_app, "st", SimpleNamespace(session_state=session_state))

    streamlit_app.sync_runtime_force_stop_after_closing_turns_for_mode("two_clients")

    assert session_state["runtime_force_stop_after_closing_turns"] == 3
    assert session_state["runtime_one_client_force_stop_after_closing_turns"] == 2
    assert session_state["runtime_two_client_force_stop_after_closing_turns"] == 3

    streamlit_app.sync_runtime_force_stop_after_closing_turns_for_mode("one_client")

    assert session_state["runtime_force_stop_after_closing_turns"] == 2
    assert session_state["runtime_one_client_force_stop_after_closing_turns"] == 2
    assert session_state["runtime_two_client_force_stop_after_closing_turns"] == 3


def test_prepare_runtime_text_defaults_keeps_manual_prompts_on_profile_change(
    monkeypatch,
) -> None:
    counselor = _profile(role="counselor", hidden_background=None)
    client = _profile(role="client", hidden_background=None)
    session_state = {
        "counselor_public_profile_source_profile_id": "old-counselor",
        "client_public_profile_source_profile_id": "old-client",
        "client_a_public_profile_source_profile_id": "old-client",
        "client_b_public_profile_source_profile_id": "old-client",
        "counselor_prompt_text": "手動カウンセラープロンプト",
        "client_prompt_text": "手動クライアントプロンプト",
        "client_a_prompt_text": "手動Aプロンプト",
        "client_b_prompt_text": "手動Bプロンプト",
        "runtime_initial_client_transcript": "初回相談です。",
    }
    monkeypatch.setattr(streamlit_app, "st", SimpleNamespace(session_state=session_state))

    prepare_runtime_text_defaults(counselor, client)

    assert session_state["counselor_prompt_text"] == "手動カウンセラープロンプト"
    assert session_state["client_prompt_text"] == "手動クライアントプロンプト"
    assert session_state["client_a_prompt_text"] == "手動Aプロンプト"
    assert session_state["client_b_prompt_text"] == "手動Bプロンプト"
    assert session_state["counselor_public_profile_text"] == counselor.public_profile
    assert session_state["client_public_profile_text"] == client.public_profile
    assert session_state["client_private_profile_text"] == ""
    assert session_state["client_a_public_profile_text"] == client.public_profile
    assert session_state["client_b_public_profile_text"] == client.public_profile


def test_prepare_runtime_text_defaults_does_not_overwrite_active_client_preset(
    monkeypatch,
) -> None:
    counselor = _profile(role="counselor", hidden_background=None)
    client = _profile(role="client", hidden_background="旧プロフィール")
    session_state = {
        "runtime_participant_mode": "one_client",
        "selected_client_preset_id": "preset_1",
        "applied_client_preset_id": "preset_1",
        "applied_client_preset_participant_mode": "one_client",
        "counselor_prompt_text": streamlit_app.DEFAULT_RUNTIME_COUNSELOR_PROMPT,
        "counselor_public_profile_source_profile_id": None,
        "client_display_name": "妻",
        "client_public_profile_text": "YAMLの共通プロフィール",
        "client_private_profile_text": "YAMLの妻プロフィール",
        "client_prompt_text": "YAMLの妻プロンプト",
        "client_public_profile_source_profile_id": "preset:preset_1",
        "client_private_profile_source_profile_id": "preset:preset_1",
    }
    monkeypatch.setattr(streamlit_app, "st", SimpleNamespace(session_state=session_state))

    prepare_runtime_text_defaults(counselor, client)

    assert session_state["client_public_profile_text"] == "YAMLの共通プロフィール"
    assert session_state["client_private_profile_text"] == "YAMLの妻プロフィール"
    assert session_state["client_prompt_text"] == "YAMLの妻プロンプト"
    assert session_state["counselor_public_profile_text"] == counselor.public_profile


def test_prepare_runtime_text_defaults_does_not_overwrite_active_counselor_preset(
    monkeypatch,
) -> None:
    counselor = _profile(role="counselor", hidden_background=None)
    client = _profile(role="client", hidden_background=None)
    session_state = {
        "runtime_participant_mode": "one_client",
        "selected_counselor_preset_id": "counselor_default",
        "applied_counselor_preset_id": "counselor_default",
        "counselor_display_name": "カウンセラー",
        "counselor_public_profile_text": "YAMLの公開プロフィール",
        "counselor_public_profile_source_profile_id": "preset:counselor_default",
        "counselor_prompt_text": "YAMLのカウンセラープロンプト",
        "counselor_prompt_source_profile_id": "preset:counselor_default",
        "client_prompt_text": "手動クライアントプロンプト",
        "client_public_profile_source_profile_id": client.profile_id,
        "client_private_profile_source_profile_id": client.profile_id,
        "client_a_public_profile_source_profile_id": client.profile_id,
        "client_b_public_profile_source_profile_id": client.profile_id,
    }
    monkeypatch.setattr(streamlit_app, "st", SimpleNamespace(session_state=session_state))

    prepare_runtime_text_defaults(counselor, client)

    assert session_state["counselor_public_profile_text"] == "YAMLの公開プロフィール"
    assert session_state["counselor_prompt_text"] == "YAMLのカウンセラープロンプト"


def test_default_runtime_counselor_prompt_uses_family_therapy_phases() -> None:
    preset = load_counselor_preset(
        Path("config/counselor_presets/counselor_default.yaml")
    )
    assert preset.counselor.prompt == streamlit_app.DEFAULT_RUNTIME_COUNSELOR_PROMPT


def test_default_runtime_client_uses_client_a_wife_identity() -> None:
    assert (
        streamlit_app.DEFAULT_RUNTIME_CLIENT_DISPLAY_NAME
        == streamlit_app.DEFAULT_RUNTIME_CLIENT_A_DISPLAY_NAME
    )
    assert (
        streamlit_app.DEFAULT_RUNTIME_CLIENT_PROMPT_INTRO
        == streamlit_app.TWO_CLIENT_A_PROMPT_INTRO
    )
    assert streamlit_app.DEFAULT_RUNTIME_CLIENT_PROMPT.startswith(
        "あなたは、家族カウンセリングのデモに参加するクライアント夫婦の妻を演じます。"
    )


def test_prepare_runtime_text_defaults_replaces_legacy_counselor_prompt(
    monkeypatch,
) -> None:
    counselor = _profile(role="counselor", hidden_background=None)
    client = _profile(role="client", hidden_background=None)
    session_state = {
        "counselor_prompt_text": streamlit_app.LEGACY_RUNTIME_COUNSELOR_SYSTEM_PROMPT,
        "client_prompt_text": "手動クライアントプロンプト",
        "counselor_public_profile_source_profile_id": counselor.profile_id,
        "client_public_profile_source_profile_id": client.profile_id,
        "client_a_public_profile_source_profile_id": client.profile_id,
        "client_b_public_profile_source_profile_id": client.profile_id,
    }
    monkeypatch.setattr(streamlit_app, "st", SimpleNamespace(session_state=session_state))

    prepare_runtime_text_defaults(counselor, client)

    assert (
        session_state["counselor_prompt_text"]
        == streamlit_app.DEFAULT_RUNTIME_COUNSELOR_PROMPT
    )
    assert session_state["client_prompt_text"] == "手動クライアントプロンプト"


def test_prepare_runtime_text_defaults_replaces_legacy_client_prompt_and_initial(
    monkeypatch,
) -> None:
    counselor = _profile(role="counselor", hidden_background=None)
    client = _profile(role="client", hidden_background=None)
    session_state = {
        "counselor_prompt_text": streamlit_app.DEFAULT_RUNTIME_COUNSELOR_PROMPT,
        "client_prompt_text": streamlit_app.LEGACY_RUNTIME_CLIENT_PROMPT,
        "runtime_initial_client_transcript": "今日は家族のことで相談したいです。",
        "counselor_public_profile_source_profile_id": counselor.profile_id,
        "client_public_profile_source_profile_id": client.profile_id,
        "client_a_public_profile_source_profile_id": client.profile_id,
        "client_b_public_profile_source_profile_id": client.profile_id,
    }
    monkeypatch.setattr(streamlit_app, "st", SimpleNamespace(session_state=session_state))

    prepare_runtime_text_defaults(counselor, client)

    assert (
        session_state["client_prompt_text"]
        == streamlit_app.DEFAULT_RUNTIME_CLIENT_PROMPT
    )
    assert (
        session_state["runtime_initial_client_transcript"]
        == streamlit_app.DEFAULT_RUNTIME_INITIAL_CLIENT_TRANSCRIPT
    )


def test_prepare_runtime_text_defaults_replaces_one_client_common_prompt_intro(
    monkeypatch,
) -> None:
    counselor = _profile(role="counselor", hidden_background=None)
    client = _profile(role="client", hidden_background=None)
    session_state = {
        "runtime_participant_mode": "one_client",
        "counselor_prompt_text": streamlit_app.DEFAULT_RUNTIME_COUNSELOR_PROMPT,
        "client_prompt_text": (
            streamlit_app.TWO_CLIENT_COMMON_PROMPT_INTRO
            + "カウンセラーではなく相談者本人です。"
        ),
        "runtime_initial_client_transcript": (
            streamlit_app.DEFAULT_RUNTIME_INITIAL_CLIENT_TRANSCRIPT
        ),
        "counselor_public_profile_source_profile_id": counselor.profile_id,
        "client_public_profile_source_profile_id": client.profile_id,
        "client_a_public_profile_source_profile_id": client.profile_id,
        "client_b_public_profile_source_profile_id": client.profile_id,
    }
    monkeypatch.setattr(streamlit_app, "st", SimpleNamespace(session_state=session_state))

    prepare_runtime_text_defaults(counselor, client)

    assert session_state["client_prompt_text"].startswith(
        streamlit_app.DEFAULT_RUNTIME_CLIENT_PROMPT_INTRO
    )
    assert not session_state["client_prompt_text"].startswith(
        streamlit_app.TWO_CLIENT_COMMON_PROMPT_INTRO
    )


def test_prepare_runtime_text_defaults_replaces_legacy_one_client_husband_intro(
    monkeypatch,
) -> None:
    counselor = _profile(role="counselor", hidden_background=None)
    client = _profile(role="client", hidden_background=None)
    session_state = {
        "runtime_participant_mode": "one_client",
        "counselor_prompt_text": streamlit_app.DEFAULT_RUNTIME_COUNSELOR_PROMPT,
        "client_prompt_text": (
            "あなたは、家族カウンセリングのデモに参加する"
            "クライアント夫婦の夫を演じます。"
            "カウンセラーではなく相談者本人です。"
        ),
        "counselor_public_profile_source_profile_id": counselor.profile_id,
        "client_public_profile_source_profile_id": client.profile_id,
        "client_a_public_profile_source_profile_id": client.profile_id,
        "client_b_public_profile_source_profile_id": client.profile_id,
    }
    monkeypatch.setattr(
        streamlit_app,
        "st",
        SimpleNamespace(session_state=session_state),
    )

    prepare_runtime_text_defaults(counselor, client)

    assert session_state["client_prompt_text"].startswith(
        streamlit_app.TWO_CLIENT_A_PROMPT_INTRO
    )
    assert "クライアント夫婦の夫を演じます" not in session_state[
        "client_prompt_text"
    ]


def test_prepare_runtime_text_defaults_keeps_user_modified_husband_prompt(
    monkeypatch,
) -> None:
    counselor = _profile(role="counselor", hidden_background=None)
    client = _profile(role="client", hidden_background=None)
    husband_prompt = (
        "あなたは、家族カウンセリングのデモに参加する"
        "クライアント夫婦の夫を演じます。手動で編集した内容です。"
    )
    session_state = {
        "runtime_participant_mode": "one_client",
        "counselor_prompt_text": streamlit_app.DEFAULT_RUNTIME_COUNSELOR_PROMPT,
        "client_prompt_text": husband_prompt,
        "client_prompt_text_persist_user_modified": True,
        "counselor_public_profile_source_profile_id": counselor.profile_id,
        "client_public_profile_source_profile_id": client.profile_id,
        "client_a_public_profile_source_profile_id": client.profile_id,
        "client_b_public_profile_source_profile_id": client.profile_id,
    }
    monkeypatch.setattr(
        streamlit_app,
        "st",
        SimpleNamespace(session_state=session_state),
    )

    prepare_runtime_text_defaults(counselor, client)

    assert session_state["client_prompt_text"] == husband_prompt


def test_prepare_runtime_text_defaults_uses_requested_display_name_defaults(
    monkeypatch,
) -> None:
    counselor = _profile(role="counselor", hidden_background=None)
    client = _profile(role="client", hidden_background=None)
    session_state = {
        "counselor_prompt_source_profile_id": None,
        "client_prompt_source_profile_id": None,
        "counselor_public_profile_source_profile_id": None,
        "client_public_profile_source_profile_id": None,
        "client_a_public_profile_source_profile_id": None,
        "client_b_public_profile_source_profile_id": None,
        "client_a_prompt_source_profile_id": None,
        "client_b_prompt_source_profile_id": None,
        "runtime_initial_client_transcript": "初回相談です。",
    }
    monkeypatch.setattr(streamlit_app, "st", SimpleNamespace(session_state=session_state))

    prepare_runtime_text_defaults(counselor, client)

    assert session_state["counselor_display_name"] == "カウンセラー"
    assert session_state["client_display_name"] == "妻"
    assert session_state["client_a_display_name"] == "妻"
    assert session_state["client_b_display_name"] == "夫"
    assert (
        session_state["counselor_prompt_text"]
        == streamlit_app.DEFAULT_RUNTIME_COUNSELOR_PROMPT
    )
    assert session_state["client_prompt_text"] == streamlit_app.DEFAULT_RUNTIME_CLIENT_PROMPT
    assert (
        session_state["client_a_prompt_text"]
        == streamlit_app.DEFAULT_RUNTIME_CLIENT_PROMPT
    )
    assert (
        session_state["client_b_prompt_text"]
        == streamlit_app.DEFAULT_RUNTIME_CLIENT_PROMPT
    )
    assert session_state["counselor_public_profile_text"] == "counselor public profile"
    assert session_state["client_public_profile_text"] == "client public profile"
    assert session_state["client_private_profile_text"] == ""
    assert session_state["client_a_public_profile_text"] == "client public profile"
    assert session_state["client_b_public_profile_text"] == "client public profile"
    assert (
        session_state["runtime_initial_client_a_transcript"]
        == "今日は中2の娘の対応について相談に来ました"
    )
    assert session_state["runtime_initial_client_b_transcript"] == ""


def test_prepare_runtime_text_defaults_loads_client_hidden_background_as_secret_profile(
    monkeypatch,
) -> None:
    counselor = _profile(role="counselor", hidden_background=None)
    client = _profile(role="client", hidden_background="プリセット非公開背景")
    session_state = {
        "counselor_prompt_source_profile_id": None,
        "client_prompt_source_profile_id": None,
        "counselor_public_profile_source_profile_id": None,
        "client_public_profile_source_profile_id": None,
        "client_a_public_profile_source_profile_id": None,
        "client_b_public_profile_source_profile_id": None,
        "client_a_prompt_source_profile_id": None,
        "client_b_prompt_source_profile_id": None,
        "runtime_initial_client_transcript": "初回相談です。",
    }
    monkeypatch.setattr(streamlit_app, "st", SimpleNamespace(session_state=session_state))

    prepare_runtime_text_defaults(counselor, client)

    assert session_state["client_public_profile_text"] == "client public profile"
    assert session_state["client_private_profile_text"] == "プリセット非公開背景"
    assert session_state["client_a_public_profile_text"] == "client public profile"
    assert session_state["client_b_public_profile_text"] == "client public profile"
    assert "client_a_private_profile_text" not in session_state
    assert "client_b_private_profile_text" not in session_state


def test_runtime_status_session_updates_map_runtime_phase_to_ui_status() -> None:
    assert runtime_status_session_updates({"phase": "paused"}) == {
        "session_status": SessionStatus.PAUSED.value,
    }
    assert runtime_status_session_updates({"phase": "unknown"}) == {}


def test_apply_runtime_control_action_preserves_existing_session_id_when_absent() -> None:
    client = _RecordingRuntimeControlClient({"phase": "stopped"})

    updates = apply_runtime_control_action(
        "stop",
        client=client,
        current_session_id="session-existing",
    )

    assert client.actions == ["stop"]
    assert updates["runtime_monitor_session_id"] == "session-existing"


def test_refresh_runtime_status_updates_current_session_id_and_ui_status() -> None:
    client = _RecordingRuntimeControlClient(
        {
            "session_id": "session-current",
            "phase": "running",
            "completed_turns": 2,
        }
    )

    updates = refresh_runtime_status(
        client=client,
        current_session_id="session-old",
    )

    assert client.actions == ["status"]
    assert updates["runtime_monitor_session_id"] == "session-current"
    assert updates["runtime_control_status"]["completed_turns"] == 2
    assert updates["session_status"] == SessionStatus.RUNNING.value
    assert "conversation_phase" not in updates


def test_refresh_runtime_status_preserves_existing_session_id_when_status_has_none() -> None:
    client = _RecordingRuntimeControlClient({"phase": "paused"})

    updates = refresh_runtime_status(
        client=client,
        current_session_id="session-existing",
    )

    assert updates["runtime_monitor_session_id"] == "session-existing"
    assert updates["session_status"] == SessionStatus.PAUSED.value


def test_should_auto_refresh_runtime_transcript_only_for_active_session() -> None:
    assert should_auto_refresh_runtime_transcript(
        {"phase": "running"},
        "session-current",
    )
    assert should_auto_refresh_runtime_transcript(
        {"phase": "paused"},
        "session-current",
    )
    assert not should_auto_refresh_runtime_transcript({"phase": "stopped"}, "session-current")
    assert not should_auto_refresh_runtime_transcript({"phase": "running"}, "")


def test_runtime_transcript_gate_includes_completed_phase() -> None:
    assert "running" in RUNTIME_TRANSCRIPT_GATED_PHASES
    assert "paused" in RUNTIME_TRANSCRIPT_GATED_PHASES
    assert "completed" in RUNTIME_TRANSCRIPT_GATED_PHASES


def test_runtime_generation_throttle_pauses_when_generation_lead_reaches_limit() -> None:
    decision = runtime_generation_throttle_decision(
        status={"phase": "running", "completed_turns": 5},
        playback_completed_turn_count=3,
        lead_limit=2,
        throttle_active=False,
    )

    assert decision["runtime_generation_lead_turns"] == 2
    assert decision["runtime_generation_throttle_action"] == "pause"
    assert decision["runtime_generation_throttle_active"] is True


def test_runtime_generation_throttle_uses_display_generated_turn_count() -> None:
    decision = runtime_generation_throttle_decision(
        status={"phase": "running", "completed_turns": 4, "current_turn_id": 4},
        playback_completed_turn_count=3,
        lead_limit=2,
        throttle_active=False,
    )

    assert decision["runtime_generation_lead_turns"] == 2
    assert decision["runtime_generation_throttle_action"] == "pause"


def test_runtime_generation_throttle_resumes_only_auto_paused_runtime() -> None:
    auto_paused_decision = runtime_generation_throttle_decision(
        status={
            "phase": "paused",
            "pause_reason": "generation_throttle",
            "completed_turns": 5,
        },
        playback_completed_turn_count=4,
        lead_limit=2,
        throttle_active=True,
    )
    manual_paused_decision = runtime_generation_throttle_decision(
        status={"phase": "paused", "pause_reason": "playback", "completed_turns": 5},
        playback_completed_turn_count=4,
        lead_limit=2,
        throttle_active=True,
    )

    assert auto_paused_decision["runtime_generation_lead_turns"] == 1
    assert auto_paused_decision["runtime_generation_throttle_action"] == "resume"
    assert auto_paused_decision["runtime_generation_throttle_active"] is False
    assert manual_paused_decision["runtime_generation_throttle_action"] is None
    assert manual_paused_decision["runtime_generation_throttle_active"] is True


def test_runtime_generation_throttle_uses_high_low_watermarks() -> None:
    still_paused_decision = runtime_generation_throttle_decision(
        status={"phase": "paused", "completed_turns": 13},
        playback_completed_turn_count=9,
        lead_limit=5,
        throttle_active=True,
    )
    resumed_decision = runtime_generation_throttle_decision(
        status={"phase": "paused", "completed_turns": 13},
        playback_completed_turn_count=10,
        lead_limit=5,
        throttle_active=True,
    )

    assert runtime_generation_resume_lead_limit(5) == 3
    assert still_paused_decision["runtime_generation_lead_turns"] == 4
    assert still_paused_decision["runtime_generation_resume_lead_limit"] == 3
    assert still_paused_decision["runtime_generation_throttle_action"] is None
    assert still_paused_decision["runtime_generation_throttle_active"] is True
    assert resumed_decision["runtime_generation_lead_turns"] == 3
    assert resumed_decision["runtime_generation_throttle_action"] == "resume"
    assert resumed_decision["runtime_generation_throttle_active"] is False


def test_runtime_generation_throttle_resumes_when_closing_reaches_start_time() -> None:
    decision = runtime_generation_throttle_decision(
        status={"phase": "paused", "completed_turns": 5},
        playback_completed_turn_count=3,
        lead_limit=2,
        throttle_active=True,
        throttle_disabled=True,
    )

    assert decision["runtime_generation_throttle_action"] == "resume"
    assert decision["runtime_generation_throttle_active"] is False


def test_runtime_generation_throttle_disabled_resumes_stale_auto_pause() -> None:
    decision = runtime_generation_throttle_decision(
        status={
            "phase": "paused",
            "pause_reason": "generation_throttle",
            "completed_turns": 7,
        },
        playback_completed_turn_count=3,
        lead_limit=5,
        throttle_active=False,
        throttle_disabled=True,
    )

    assert decision["runtime_generation_throttle_action"] == "resume"
    assert decision["runtime_generation_throttle_active"] is False


def test_runtime_generation_throttle_disables_after_closing_due_or_started() -> None:
    assert runtime_generation_throttle_disabled_for_closing(
        status={},
        clock_seconds=10.0,
        closing_start_seconds=10.0,
    )
    assert runtime_generation_throttle_disabled_for_closing(
        status={"closing_count_started_turn_id": 3},
        clock_seconds=3.0,
        closing_start_seconds=10.0,
    )
    assert not runtime_generation_throttle_disabled_for_closing(
        status={},
        clock_seconds=9.9,
        closing_start_seconds=10.0,
    )


def test_runtime_generated_turn_count_for_display_includes_initial_read_aloud_turn() -> None:
    assert runtime_generated_turn_count_for_display(
        {"completed_turns": 4, "current_turn_id": 3}
    ) == 4
    assert runtime_generated_turn_count_for_display(
        {"completed_turns": 1, "current_turn_id": 0}
    ) == 1
    assert runtime_generated_turn_count_for_display({"completed_turns": 4}) == 4


def test_runtime_generated_turn_count_for_display_can_exclude_active_turn() -> None:
    assert (
        runtime_generated_turn_count_for_display(
            {"completed_turns": 1, "current_turn_id": 2},
            include_active_turn=False,
        )
        == 1
    )
    assert runtime_current_turn_metric_value({"current_turn_id": 2}) == "2ターン目"
    assert runtime_current_turn_metric_value({}) == "-"


def test_runtime_completed_turn_count_for_display_includes_initial_read_aloud_turn() -> None:
    turns = [
        PublicTranscriptTurn(
            turn_id=0,
            speaker="夫",
            text="初回発話です。",
            created_at=None,
            audio_path=None,
            cumulative_audio_seconds=12.0,
            audio_duration_seconds=1.0,
        ),
        PublicTranscriptTurn(
            turn_id=1,
            speaker="カウンセラー",
            text="2つ目の発話です。",
            created_at=None,
            audio_path=None,
            cumulative_audio_seconds=13.0,
            audio_duration_seconds=1.0,
        ),
        PublicTranscriptTurn(
            turn_id=2,
            speaker="夫",
            text="3つ目の発話です。",
            created_at=None,
            audio_path=None,
            cumulative_audio_seconds=14.0,
            audio_duration_seconds=1.0,
        ),
    ]

    assert (
        runtime_completed_turn_count_for_display(
            turns=turns,
            clock_seconds=0.5,
        )
        == 0
    )
    assert (
        runtime_completed_turn_count_for_display(
            turns=turns,
            clock_seconds=1.5,
        )
        == 1
    )
    assert (
        runtime_completed_turn_count_for_display(
            turns=turns,
            clock_seconds=3.5,
        )
        == 1
    )
    assert (
        runtime_completed_turn_count_for_display(
            turns=turns,
            clock_seconds=3.6,
        )
        == 2
    )
    assert (
        runtime_completed_turn_count_for_display(
            turns=turns,
            clock_seconds=6.2,
        )
        == 3
    )


def test_runtime_completed_turn_count_for_display_ignores_saved_timeline_warmup_gap() -> None:
    turns = [
        PublicTranscriptTurn(
            turn_id=0,
            speaker="夫",
            text="初回発話です。",
            created_at=None,
            audio_path=None,
            cumulative_audio_seconds=18.0,
            audio_duration_seconds=1.0,
        ),
        PublicTranscriptTurn(
            turn_id=1,
            speaker="カウンセラー",
            text="次の発話です。",
            created_at=None,
            audio_path=None,
            cumulative_audio_seconds=23.0,
            audio_duration_seconds=1.0,
        ),
    ]

    assert runtime_playback_relative_start_seconds(turns) == pytest.approx([0.0, 2.6])
    assert (
        runtime_completed_turn_count_for_display(
            turns=turns,
            clock_seconds=1.0,
        )
        == 1
    )
    assert (
        runtime_completed_turn_count_for_display(
            turns=turns,
            clock_seconds=2.5,
        )
        == 1
    )
    assert (
        runtime_completed_turn_count_for_display(
            turns=turns,
            clock_seconds=3.6,
        )
        == 2
    )


def test_runtime_status_playback_completed_turn_count_handles_missing_or_invalid_status() -> None:
    assert runtime_status_playback_completed_turn_count({}) is None
    assert (
        runtime_status_playback_completed_turn_count(
            {"playback_completed_turns": "invalid"}
        )
        is None
    )
    assert (
        runtime_status_playback_completed_turn_count({"playback_completed_turns": -2})
        == 0
    )
    assert (
        runtime_status_playback_completed_turn_count({"playback_completed_turns": "3"})
        == 3
    )


def test_runtime_completed_turn_count_for_metric_prefers_status_playback_completion() -> None:
    turns = [
        PublicTranscriptTurn(
            turn_id=0,
            speaker="夫",
            text="初回発話です。",
            created_at=None,
            audio_path=None,
            cumulative_audio_seconds=18.0,
            audio_duration_seconds=1.0,
        ),
        PublicTranscriptTurn(
            turn_id=1,
            speaker="カウンセラー",
            text="次の発話です。",
            created_at=None,
            audio_path=None,
            cumulative_audio_seconds=23.0,
            audio_duration_seconds=1.0,
        ),
    ]

    playback_completed = runtime_status_playback_completed_turn_count(
        {"playback_completed_turns": 1}
    )

    assert playback_completed == 1
    assert (
        runtime_completed_turn_count_for_metric(
            status={"phase": "running", "current_turn_id": 1},
            turns=turns,
            clock_seconds=999.0,
            session_id="session-a",
            previous_session_id="session-a",
            previous_completed_turns=2,
            browser_completed_turn_count=playback_completed,
        )
        == 1
    )


def test_runtime_completed_turn_count_for_display_uses_next_start_boundary() -> None:
    turns = [
        PublicTranscriptTurn(
            turn_id=1,
            speaker="夫",
            text="初回発話です。",
            created_at=None,
            audio_path=None,
            cumulative_audio_seconds=0.0,
            audio_duration_seconds=None,
        ),
        PublicTranscriptTurn(
            turn_id=2,
            speaker="カウンセラー",
            text="2つ目の発話です。",
            created_at=None,
            audio_path=None,
            cumulative_audio_seconds=4.0,
            audio_duration_seconds=None,
        ),
        PublicTranscriptTurn(
            turn_id=3,
            speaker="夫",
            text="3つ目の発話です。",
            created_at=None,
            audio_path=None,
            cumulative_audio_seconds=7.0,
            audio_duration_seconds=None,
        ),
    ]

    assert (
        runtime_completed_turn_count_for_display(
            turns=turns,
            clock_seconds=6.9,
        )
        == 1
    )
    assert (
        runtime_completed_turn_count_for_display(
            turns=turns,
            clock_seconds=7.0,
        )
        == 2
    )


def test_runtime_completed_turn_count_for_display_does_not_drop_first_nonzero_turn() -> None:
    turns = [
        PublicTranscriptTurn(
            turn_id=1,
            speaker="カウンセラー",
            text="最初の生成発話です。",
            created_at=None,
            audio_path=None,
            cumulative_audio_seconds=0.0,
            audio_duration_seconds=1.0,
        ),
        PublicTranscriptTurn(
            turn_id=2,
            speaker="夫",
            text="次の生成発話です。",
            created_at=None,
            audio_path=None,
            cumulative_audio_seconds=1.0,
            audio_duration_seconds=1.0,
        ),
    ]

    assert (
        runtime_completed_turn_count_for_display(
            turns=turns,
            clock_seconds=1.0,
        )
        == 1
    )


def test_runtime_completed_turn_count_for_metric_keeps_same_session_monotonic() -> None:
    turns = [
        PublicTranscriptTurn(
            turn_id=0,
            speaker="夫",
            text="初回発話です。",
            created_at=None,
            audio_path=None,
            cumulative_audio_seconds=0.0,
            audio_duration_seconds=1.0,
        ),
        PublicTranscriptTurn(
            turn_id=1,
            speaker="カウンセラー",
            text="1つ目の生成発話です。",
            created_at=None,
            audio_path=None,
            cumulative_audio_seconds=1.0,
            audio_duration_seconds=1.0,
        ),
        PublicTranscriptTurn(
            turn_id=2,
            speaker="夫",
            text="2つ目の生成発話です。",
            created_at=None,
            audio_path=None,
            cumulative_audio_seconds=2.0,
            audio_duration_seconds=1.0,
        ),
        PublicTranscriptTurn(
            turn_id=3,
            speaker="カウンセラー",
            text="3つ目の生成発話です。",
            created_at=None,
            audio_path=None,
            cumulative_audio_seconds=3.0,
            audio_duration_seconds=1.0,
        ),
        PublicTranscriptTurn(
            turn_id=4,
            speaker="夫",
            text="最後の生成発話です。",
            created_at=None,
            audio_path=None,
            cumulative_audio_seconds=4.0,
            audio_duration_seconds=None,
        ),
    ]

    assert (
        runtime_completed_turn_count_for_display(
            turns=turns,
            clock_seconds=20.0,
        )
        == 4
    )
    assert (
        runtime_completed_turn_count_for_metric(
            status={"current_turn_id": 4},
            turns=turns,
            clock_seconds=20.0,
            session_id="session-live",
            previous_session_id="session-live",
            previous_completed_turns=5,
        )
        == 5
    )


def test_runtime_completed_turn_count_for_metric_resets_for_new_session() -> None:
    turns = [
        PublicTranscriptTurn(
            turn_id=1,
            speaker="カウンセラー",
            text="最初の生成発話です。",
            created_at=None,
            audio_path=None,
            cumulative_audio_seconds=0.0,
            audio_duration_seconds=1.0,
        ),
        PublicTranscriptTurn(
            turn_id=2,
            speaker="夫",
            text="次の生成発話です。",
            created_at=None,
            audio_path=None,
            cumulative_audio_seconds=1.0,
            audio_duration_seconds=None,
        ),
    ]

    assert (
        runtime_completed_turn_count_for_metric(
            status={"current_turn_id": 2},
            turns=turns,
            clock_seconds=10.0,
            session_id="session-new",
            previous_session_id="session-old",
            previous_completed_turns=2,
        )
        == 1
    )


def test_runtime_completed_turn_count_for_metric_counts_final_turn_at_audio_end() -> None:
    turns = [
        PublicTranscriptTurn(
            turn_id=0,
            speaker="夫",
            text="初回発話です。",
            created_at=None,
            audio_path=None,
            cumulative_audio_seconds=0.0,
            audio_duration_seconds=1.0,
        ),
        PublicTranscriptTurn(
            turn_id=1,
            speaker="カウンセラー",
            text="1つ目の生成発話です。",
            created_at=None,
            audio_path=None,
            cumulative_audio_seconds=1.0,
            audio_duration_seconds=1.0,
        ),
        PublicTranscriptTurn(
            turn_id=2,
            speaker="夫",
            text="2つ目の生成発話です。",
            created_at=None,
            audio_path=None,
            cumulative_audio_seconds=2.0,
            audio_duration_seconds=1.0,
        ),
        PublicTranscriptTurn(
            turn_id=3,
            speaker="カウンセラー",
            text="3つ目の生成発話です。",
            created_at=None,
            audio_path=None,
            cumulative_audio_seconds=3.0,
            audio_duration_seconds=1.0,
        ),
        PublicTranscriptTurn(
            turn_id=4,
            speaker="夫",
            text="最後の生成発話です。",
            created_at=None,
            audio_path=None,
            cumulative_audio_seconds=4.0,
            audio_duration_seconds=1.0,
        ),
    ]

    assert (
        runtime_completed_turn_count_for_metric(
            status={"current_turn_id": 4},
            turns=turns,
            clock_seconds=4.0,
            session_id="session-live",
            previous_session_id="session-live",
            previous_completed_turns=4,
        )
        == 4
    )
    assert (
        runtime_completed_turn_count_for_metric(
            status={"current_turn_id": 4},
            turns=turns,
            clock_seconds=5.0,
            session_id="session-live",
            previous_session_id="session-live",
            previous_completed_turns=4,
        )
        == 4
    )
    assert (
        runtime_completed_turn_count_for_metric(
            status={"current_turn_id": 4},
            turns=turns,
            clock_seconds=11.4,
            session_id="session-live",
            previous_session_id="session-live",
            previous_completed_turns=4,
        )
        == 5
    )


def test_runtime_completed_turn_count_for_metric_catches_up_terminal_final_turn() -> None:
    turns = [
        PublicTranscriptTurn(
            turn_id=0,
            speaker="夫",
            text="初回発話です。",
            created_at=None,
            audio_path=None,
            cumulative_audio_seconds=0.0,
            audio_duration_seconds=1.0,
        ),
        PublicTranscriptTurn(
            turn_id=1,
            speaker="カウンセラー",
            text="1つ目の生成発話です。",
            created_at=None,
            audio_path=None,
            cumulative_audio_seconds=1.0,
            audio_duration_seconds=1.0,
        ),
        PublicTranscriptTurn(
            turn_id=2,
            speaker="夫",
            text="最後の生成発話です。",
            created_at=None,
            audio_path=None,
            cumulative_audio_seconds=2.0,
            audio_duration_seconds=1.0,
        ),
    ]

    assert (
        runtime_completed_turn_count_for_metric(
            status={"phase": "completed", "current_turn_id": 2},
            turns=turns,
            clock_seconds=2.0,
            session_id="session-live",
            previous_session_id="session-live",
            previous_completed_turns=2,
        )
        == 2
    )
    assert (
        runtime_completed_turn_count_for_metric(
            status={"phase": "completed", "current_turn_id": 2},
            turns=turns,
            clock_seconds=3.0,
            session_id="session-live",
            previous_session_id="session-live",
            previous_completed_turns=2,
        )
        == 2
    )
    assert (
        runtime_completed_turn_count_for_metric(
            status={"phase": "completed", "current_turn_id": 2},
            turns=turns,
            clock_seconds=6.2,
            session_id="session-live",
            previous_session_id="session-live",
            previous_completed_turns=2,
        )
        == 3
    )


def test_runtime_completed_turn_count_state_updates_uses_playback_progress() -> None:
    turns = [
        PublicTranscriptTurn(
            turn_id=turn_id,
            speaker="話者",
            text=f"{turn_id}番目の発話です。",
            created_at=None,
            audio_path=None,
            cumulative_audio_seconds=float(turn_id),
            audio_duration_seconds=1.0,
        )
        for turn_id in range(5)
    ]

    updates = runtime_completed_turn_count_state_updates(
        status={"phase": "running", "current_turn_id": 4},
        turns=turns,
        clock_seconds=3.6,
        session_id="session-live",
        previous_session_id="session-live",
        previous_completed_turns=0,
    )

    assert updates["runtime_completed_turn_count_session_id"] == "session-live"
    assert updates["runtime_completed_turn_count_for_display"] == 2


def test_runtime_status_label_for_timer_waits_for_completed_turn_catchup() -> None:
    assert (
        runtime_status_label_for_timer(
            SessionStatus.COMPLETED.value,
            generated_turns=3,
            completed_turns=2,
        )
        == SessionStatus.RUNNING.value
    )
    assert (
        runtime_status_label_for_timer(
            SessionStatus.COMPLETED.value,
            generated_turns=3,
            completed_turns=3,
        )
        == SessionStatus.COMPLETED.value
    )
    assert (
        runtime_status_label_for_timer(
            SessionStatus.PAUSED.value,
            generated_turns=3,
            completed_turns=2,
        )
        == SessionStatus.PAUSED.value
    )


def test_runtime_timer_state_updates_running_wall_clock_and_audio_elapsed() -> None:
    updates = runtime_timer_state_updates(
        status={"phase": "running"},
        session_id="session-current",
        transcript_audio_seconds=7.5,
        current_wall_clock_seconds=2.0,
        wall_clock_session_id="session-current",
        wall_clock_started_at_monotonic=100.0,
        wall_clock_base_seconds=2.0,
        now_monotonic=103.0,
    )

    assert updates["cumulative_audio_seconds"] == 7.5
    assert updates["wall_clock_seconds"] == 5.0
    assert updates["runtime_wall_clock_started_at_monotonic"] == 100.0
    assert updates["runtime_wall_clock_base_seconds"] == 2.0


def test_runtime_timer_state_updates_freezes_wall_clock_during_initial_warmup() -> None:
    updates = runtime_timer_state_updates(
        status={"phase": "running"},
        session_id="session-current",
        transcript_audio_seconds=7.5,
        current_wall_clock_seconds=2.0,
        wall_clock_session_id="session-current",
        wall_clock_started_at_monotonic=100.0,
        wall_clock_base_seconds=2.0,
        now_monotonic=105.0,
        playback_warmup_active=True,
    )

    assert updates["cumulative_audio_seconds"] == 7.5
    assert updates["wall_clock_seconds"] == 2.0
    assert updates["runtime_wall_clock_started_at_monotonic"] is None
    assert updates["runtime_wall_clock_base_seconds"] == 2.0


def test_runtime_timer_state_updates_keeps_new_session_clock_zero_during_initial_warmup() -> None:
    updates = runtime_timer_state_updates(
        status={"phase": "running"},
        session_id="session-new",
        transcript_audio_seconds=0.0,
        current_wall_clock_seconds=45.0,
        wall_clock_session_id="session-old",
        wall_clock_started_at_monotonic=None,
        wall_clock_base_seconds=45.0,
        now_monotonic=200.0,
        playback_warmup_active=True,
    )

    assert updates["runtime_wall_clock_session_id"] == "session-new"
    assert updates["wall_clock_seconds"] == 0.0
    assert updates["runtime_wall_clock_started_at_monotonic"] is None
    assert updates["runtime_wall_clock_base_seconds"] == 0.0


def test_runtime_timer_state_updates_resets_for_new_runtime_session() -> None:
    updates = runtime_timer_state_updates(
        status={"phase": "running"},
        session_id="session-new",
        transcript_audio_seconds=0.0,
        current_wall_clock_seconds=45.0,
        wall_clock_session_id="session-old",
        wall_clock_started_at_monotonic=None,
        wall_clock_base_seconds=45.0,
        now_monotonic=200.0,
    )

    assert updates["runtime_wall_clock_session_id"] == "session-new"
    assert updates["runtime_wall_clock_started_at_monotonic"] == 200.0
    assert updates["runtime_wall_clock_base_seconds"] == 0.0
    assert updates["wall_clock_seconds"] == 0.0


def test_runtime_timer_state_updates_freezes_wall_clock_when_paused() -> None:
    updates = runtime_timer_state_updates(
        status={"phase": "paused"},
        session_id="session-current",
        transcript_audio_seconds=8.0,
        current_wall_clock_seconds=2.0,
        wall_clock_session_id="session-current",
        wall_clock_started_at_monotonic=100.0,
        wall_clock_base_seconds=2.0,
        now_monotonic=105.0,
    )

    assert updates["cumulative_audio_seconds"] == 8.0
    assert updates["wall_clock_seconds"] == 7.0
    assert updates["runtime_wall_clock_started_at_monotonic"] is None
    assert updates["runtime_wall_clock_base_seconds"] == 7.0


def test_runtime_timer_state_updates_freezes_wall_clock_when_audio_playback_paused() -> None:
    updates = runtime_timer_state_updates(
        status={"phase": "running"},
        session_id="session-current",
        transcript_audio_seconds=8.0,
        current_wall_clock_seconds=2.0,
        wall_clock_session_id="session-current",
        wall_clock_started_at_monotonic=100.0,
        wall_clock_base_seconds=2.0,
        now_monotonic=105.0,
        playback_paused=True,
    )

    assert updates["cumulative_audio_seconds"] == 8.0
    assert updates["wall_clock_seconds"] == 7.0
    assert updates["runtime_wall_clock_started_at_monotonic"] is None
    assert updates["runtime_wall_clock_base_seconds"] == 7.0


def test_runtime_timer_state_updates_continues_during_auto_throttle_pause() -> None:
    updates = runtime_timer_state_updates(
        status={"phase": "paused"},
        session_id="session-current",
        transcript_audio_seconds=8.0,
        current_wall_clock_seconds=2.0,
        wall_clock_session_id="session-current",
        wall_clock_started_at_monotonic=100.0,
        wall_clock_base_seconds=2.0,
        now_monotonic=105.0,
        continue_when_paused=True,
    )

    assert updates["cumulative_audio_seconds"] == 8.0
    assert updates["wall_clock_seconds"] == 7.0
    assert updates["runtime_wall_clock_started_at_monotonic"] == 100.0
    assert updates["runtime_wall_clock_base_seconds"] == 2.0


def test_runtime_timer_continues_until_all_generated_turns_finish_playback() -> None:
    updates = runtime_timer_state_updates(
        status={
            "phase": "completed",
            "current_turn_id": 4,
            "completed_turns": 5,
            "playback_completed_turns": 3,
        },
        session_id="session-current",
        transcript_audio_seconds=8.0,
        current_wall_clock_seconds=2.0,
        wall_clock_session_id="session-current",
        wall_clock_started_at_monotonic=100.0,
        wall_clock_base_seconds=2.0,
        now_monotonic=105.0,
    )

    assert updates["wall_clock_seconds"] == 7.0
    assert updates["runtime_wall_clock_started_at_monotonic"] == 100.0
    assert updates["runtime_wall_clock_base_seconds"] == 2.0


def test_runtime_timer_stops_after_all_generated_turns_finish_playback() -> None:
    updates = runtime_timer_state_updates(
        status={
            "phase": "completed",
            "current_turn_id": 4,
            "completed_turns": 5,
            "playback_completed_turns": 5,
        },
        session_id="session-current",
        transcript_audio_seconds=8.0,
        current_wall_clock_seconds=2.0,
        wall_clock_session_id="session-current",
        wall_clock_started_at_monotonic=100.0,
        wall_clock_base_seconds=2.0,
        now_monotonic=105.0,
    )

    assert updates["wall_clock_seconds"] == 7.0
    assert updates["runtime_wall_clock_started_at_monotonic"] is None
    assert updates["runtime_wall_clock_base_seconds"] == 7.0


def test_runtime_timer_state_updates_audio_pause_overrides_auto_throttle_pause() -> None:
    updates = runtime_timer_state_updates(
        status={"phase": "paused"},
        session_id="session-current",
        transcript_audio_seconds=8.0,
        current_wall_clock_seconds=2.0,
        wall_clock_session_id="session-current",
        wall_clock_started_at_monotonic=100.0,
        wall_clock_base_seconds=2.0,
        now_monotonic=105.0,
        continue_when_paused=True,
        playback_paused=True,
    )

    assert updates["wall_clock_seconds"] == 7.0
    assert updates["runtime_wall_clock_started_at_monotonic"] is None
    assert updates["runtime_wall_clock_base_seconds"] == 7.0


def test_runtime_initial_warmup_state_releases_on_target_turn_count() -> None:
    initial_updates = runtime_initial_warmup_state_updates(
        status={"phase": "running", "completed_turns": 1},
        session_id="session-current",
        warmup_target_turns=4,
        warmup_max_wait_seconds=24.0,
        warmup_session_id="",
        warmup_started_at_monotonic=None,
        warmup_released=False,
        now_monotonic=100.0,
    )

    assert initial_updates["runtime_initial_warmup_active"] is True
    assert initial_updates["runtime_initial_warmup_session_id"] == "session-current"
    assert initial_updates["runtime_initial_warmup_started_at_monotonic"] == 100.0
    assert initial_updates["runtime_initial_warmup_released"] is False

    released_updates = runtime_initial_warmup_state_updates(
        status={"phase": "running", "completed_turns": 4},
        session_id="session-current",
        warmup_target_turns=4,
        warmup_max_wait_seconds=24.0,
        warmup_session_id="session-current",
        warmup_started_at_monotonic=100.0,
        warmup_released=False,
        now_monotonic=105.0,
    )

    assert released_updates["runtime_initial_warmup_active"] is False
    assert released_updates["runtime_initial_warmup_released"] is True
    assert released_updates["runtime_initial_warmup_started_at_monotonic"] is None


def test_runtime_initial_warmup_state_releases_on_timeout() -> None:
    updates = runtime_initial_warmup_state_updates(
        status={"phase": "running", "completed_turns": 1},
        session_id="session-current",
        warmup_target_turns=4,
        warmup_max_wait_seconds=24.0,
        warmup_session_id="session-current",
        warmup_started_at_monotonic=100.0,
        warmup_released=False,
        now_monotonic=125.0,
    )

    assert updates["runtime_initial_warmup_active"] is False
    assert updates["runtime_initial_warmup_released"] is True
    assert updates["runtime_initial_warmup_started_at_monotonic"] is None


def test_runtime_remaining_metric_uses_closing_time_then_closing_turn_count() -> None:
    assert runtime_remaining_metric(
        status={"completed_turns": 2},
        wall_clock_seconds=125.0,
        closing_start_seconds=300.0,
        force_stop_after_closing_turns=3,
    ) == ("クロージングまで", "02:55")
    assert runtime_remaining_metric(
        status={"completed_turns": 2, "closing_started_turn_id": 2},
        wall_clock_seconds=125.0,
        closing_start_seconds=300.0,
        force_stop_after_closing_turns=4,
    ) == ("終了まで", "4ターン")
    assert runtime_remaining_metric(
        status={
            "completed_turns": 3,
            "current_turn_id": 3,
            "closing_started_turn_id": 2,
            "closing_count_started_turn_id": 3,
        },
        wall_clock_seconds=125.0,
        closing_start_seconds=300.0,
        force_stop_after_closing_turns=4,
    ) == ("終了まで", "3ターン")


def test_refresh_runtime_control_events_requires_session_id() -> None:
    client = _RecordingRuntimeControlClient()

    updates = refresh_runtime_control_events(client=client, session_id="")

    assert "session_id が必要" in updates["runtime_control_error"]
    assert client.event_session_ids == []


def test_refresh_runtime_control_events_reads_client_events() -> None:
    client = _RecordingRuntimeControlClient()

    updates = refresh_runtime_control_events(
        client=client,
        session_id="session-runtime",
    )

    assert client.event_session_ids == ["session-runtime"]
    assert client.response_instruction_session_ids == ["session-runtime"]
    assert updates["runtime_control_events"] == [{"event_type": "tts_stream_done"}]
    assert updates["runtime_response_instructions"] == [
        {
            "turn_id": 1,
            "speaker_id": "counselor",
            "resolved_instructions": "送信済みinstructions",
        }
    ]
    assert updates["runtime_control_error"] == ""


def test_runtime_status_rows_keep_expected_order() -> None:
    rows = runtime_status_rows(
        {
            "session_id": "session-runtime",
            "phase": "running",
            "pause_reason": None,
            "current_turn_id": 2,
            "current_speaker": "client",
            "completed_turns": 1,
            "last_event_type": "generated_final",
            "error_message": "",
        }
    )

    assert [row["項目"] for row in rows] == [
        "session_id",
        "phase",
        "pause_reason",
        "current_turn_id",
        "current_speaker",
        "completed_turns",
        "last_event_type",
        "error_message",
    ]
    assert rows[1]["値"] == "running"
    assert rows[2]["値"] == ""
    assert rows[3]["値"] == "2"
    assert rows[5]["値"] == "1"


def test_runtime_status_rows_convert_none_to_empty_string() -> None:
    rows = runtime_status_rows(
        {
            "session_id": "runtime_control",
            "phase": "idle",
            "pause_reason": None,
            "current_turn_id": None,
            "current_speaker": None,
            "completed_turns": 0,
            "last_event_type": None,
            "error_message": None,
        }
    )

    assert rows[3]["値"] == ""
    assert rows[4]["値"] == ""
    assert rows[5]["値"] == "0"
    assert rows[6]["値"] == ""


def test_runtime_deferred_interaction_feature_rows_mark_step22_as_backend_ready() -> None:
    rows = runtime_deferred_interaction_feature_rows()

    assert rows == [
        {
            "機能": "動的ターンテイク",
            "状態": "発話後の各Agent判断",
            "次段階": "応答遅延・発話量チューニング",
        },
        {
            "機能": "譲り合い",
            "状態": "YIELD/CONTINUE調停",
            "次段階": "重なり時の話者選択チューニング",
        },
        {
            "機能": "割り込み",
            "状態": "停止/truncate接続実装",
            "次段階": "端末別レイテンシ検証",
        },
        {
            "機能": "重なり",
            "状態": "grace + 片側停止",
            "次段階": "譲る話者の選択精度調整",
        },
    ]


def test_runtime_monitor_endpoint_uses_stable_monitor_url_without_session_id() -> None:
    endpoint = runtime_monitor_endpoint(host="127.0.0.1", port=8765)

    assert endpoint.monitor_ws_url == "ws://127.0.0.1:8765/runtime/monitor-audio"


def test_runtime_panel_renders_audio_monitor_with_component_controls() -> None:
    source = Path("app/streamlit_app.py").read_text(encoding="utf-8")

    panel_source = source[
        source.index("def render_runtime_session_panel") : source.index(
            "def render_runtime_debug_panel"
        )
    ]
    assert '("Start", "start")' not in panel_source
    assert '("Stop", "stop")' not in panel_source
    assert '("Refresh", "status")' not in panel_source
    assert "st.checkbox(\"Mute\"" not in panel_source
    assert "runtime_audio_monitor(" in panel_source
    assert 'key="runtime_audio_monitor_v12"' in panel_source
    assert "start_url=f\"{endpoint.base_url}/runtime/start\"" in panel_source
    assert "runtime_pause_url=f\"{endpoint.base_url}/runtime/pause\"" in panel_source
    assert "runtime_resume_url=f\"{endpoint.base_url}/runtime/resume\"" in panel_source
    assert "runtime_stop_url=f\"{endpoint.base_url}/runtime/stop\"" in panel_source
    assert "runtime_interrupt_url=f\"{endpoint.base_url}/runtime/interrupt\"" in panel_source
    assert "runtime_playback_completed_url=(" in panel_source
    assert "f\"{endpoint.base_url}/runtime/playback-completed\"" in panel_source
    assert "runtime_status_url=f\"{endpoint.base_url}/runtime/status\"" in panel_source
    assert "Runtime error:" in panel_source
    assert "runtime_error_message" in panel_source
    assert "start_options=runtime_start_options" in panel_source
    assert "warmup_target_turns=int(" in panel_source
    assert "warmup_max_wait_ms=int(" in panel_source
    assert "RUNTIME_AUDIO_MONITOR_WARMUP_MAX_WAIT_SECONDS * 1000" in panel_source
    assert "runtime_generation_lead_limit" in panel_source
    assert "volume=1.0" in panel_source
    assert "muted=False" in panel_source
    assert "ライブ音声を聞く場合は" not in panel_source
    assert "Connect & Start audio から開始してください" not in panel_source
    assert '"Monitor volume"' not in panel_source
    assert "runtime_monitor_volume" not in panel_source
    assert "render_runtime_debug_panel(" not in panel_source
    assert "runtime_status_rows(status)" not in panel_source
    assert 'column.metric(str(row["項目"]), str(row["値"]))' not in panel_source


def test_runtime_debug_panel_shows_deferred_step22_feature_state() -> None:
    source = Path("app/streamlit_app.py").read_text(encoding="utf-8")

    debug_source = source[
        source.index("def render_runtime_debug_panel") : source.index(
            "def render_runtime_observer_panel"
        )
    ]

    assert "runtime_deferred_interaction_feature_rows()" in debug_source
    assert '"Speaker selection policy"' in debug_source
    assert "RUNTIME_SPEAKER_SELECTION_POLICY_OPTIONS" in debug_source
    generation_mode_index = debug_source.index('"Generation mode: Realtime API centered"')
    deferred_state_index = debug_source.index(
        "runtime_deferred_interaction_feature_rows()"
    )
    runtime_status_index = debug_source.index("runtime_status_rows(status)")
    assert generation_mode_index < deferred_state_index < runtime_status_index


def test_runtime_debug_panel_shows_fixed_system_prompts_read_only() -> None:
    source = Path("app/streamlit_app.py").read_text(encoding="utf-8")

    debug_source = source[
        source.index("def render_runtime_debug_panel") : source.index(
            "def render_runtime_observer_panel"
        )
    ]

    assert 'st.markdown("##### 固定システムプロンプト")' in debug_source
    assert '"固定カウンセラー・システムプロンプト"' in debug_source
    assert '"固定クライアント・システムプロンプト"' in debug_source
    assert "COUNSELOR_SYSTEM_PROMPT" in debug_source
    assert "CLIENT_SYSTEM_PROMPT" in debug_source
    assert debug_source.count("disabled=True") >= 2


def test_runtime_debug_panel_shows_sent_response_instructions_read_only() -> None:
    source = Path("app/streamlit_app.py").read_text(encoding="utf-8")
    debug_source = source[
        source.index("def render_runtime_debug_panel") : source.index(
            "def render_runtime_observer_panel"
        )
    ]

    assert 'st.markdown("##### 送信済み response.create.instructions")' in debug_source
    assert "runtime_response_instructions" in debug_source
    assert '"実際に送信した instructions"' in debug_source
    assert '"固定セッション指示"' in debug_source
    assert '"プリセット／Session setup 指示"' in debug_source
    assert '"ターン固有指示"' in debug_source
    assert debug_source.count("disabled=True") >= 6


def test_counselor_gain_controls_use_persistent_widget_state() -> None:
    source = Path("app/streamlit_app.py").read_text(encoding="utf-8")
    settings_source = source[
        source.index("def render_one_client_setup_tab") : source.index(
            "def render_runtime_common_session_controls"
        )
    ]

    control_starts = []
    search_from = 0
    while True:
        start = settings_source.find('"カウンセラー音量"', search_from)
        if start < 0:
            break
        control_starts.append(start)
        search_from = start + 1

    assert len(control_starts) == 2
    for start in control_starts:
        control_source = settings_source[start : start + 650]
        assert "key=sync_runtime_widget_from_state(" in control_source
        assert '"runtime_counselor_audio_gain"' in control_source
        assert "on_change=mark_runtime_widget_user_modified" in control_source
        assert 'runtime_widget_key("runtime_counselor_audio_gain")' in control_source


def test_counselor_prompt_controls_use_persistent_widget_state() -> None:
    source = Path("app/streamlit_app.py").read_text(encoding="utf-8")
    settings_source = source[
        source.index("def render_one_client_setup_tab") : source.index(
            "def render_runtime_common_session_controls"
        )
    ]

    assert settings_source.count(
        'key=sync_runtime_widget_from_state("counselor_prompt_text")'
    ) == 2
    assert settings_source.count('runtime_widget_key("counselor_prompt_text")') == 2


def test_counselor_prompt_widget_restores_preset_text_after_hidden_mode(
    monkeypatch,
) -> None:
    session_state = {
        "counselor_prompt_text": "YAMLのカウンセラープロンプト",
    }
    monkeypatch.setattr(streamlit_app, "st", SimpleNamespace(session_state=session_state))

    widget_key = streamlit_app.sync_runtime_widget_from_state(
        "counselor_prompt_text"
    )

    assert widget_key == "counselor_prompt_text_widget"
    assert session_state[widget_key] == "YAMLのカウンセラープロンプト"

    session_state[widget_key] = "UIで編集したカウンセラープロンプト"
    streamlit_app.mark_runtime_widget_user_modified(
        widget_key,
        "counselor_prompt_text",
    )

    assert session_state["counselor_prompt_text"] == (
        "UIで編集したカウンセラープロンプト"
    )


def test_runtime_audio_settings_rows_support_one_and_two_client_start_options() -> None:
    one_client_rows = streamlit_app.runtime_audio_settings_rows_from_start_options(
        {
            "participants": {
                "counselor": {"display_name": "佐伯"},
                "client": {"display_name": "高橋"},
            },
            "counselor_tts_voice": "shimmer",
            "client_tts_voice": "cedar",
            "counselor_realtime_output_speed": 1.0,
            "client_realtime_output_speed": 0.9,
            "speaker_gains": {"counselor": 0.6, "client": 0.7},
        }
    )
    two_client_rows = streamlit_app.runtime_audio_settings_rows_from_start_options(
        {
            "participants": {
                "counselor": {
                    "display_name": "佐伯",
                    "voice": "shimmer",
                    "realtime_output_speed": 1.0,
                },
                "client_a": {
                    "display_name": "妻",
                    "voice": "marin",
                    "realtime_output_speed": 0.9,
                },
                "client_b": {
                    "display_name": "夫",
                    "voice": "cedar",
                    "realtime_output_speed": 1.1,
                },
            },
            "speaker_gains": {
                "counselor": 0.6,
                "client_a": 0.7,
                "client_b": 0.8,
            },
        }
    )

    assert one_client_rows == [
        {
            "話者ID": "counselor",
            "表示名": "佐伯",
            "音声": "shimmer",
            "話速": 1.0,
            "音量": 0.6,
        },
        {
            "話者ID": "client",
            "表示名": "高橋",
            "音声": "cedar",
            "話速": 0.9,
            "音量": 0.7,
        },
    ]
    assert [row["話者ID"] for row in two_client_rows] == [
        "counselor",
        "client_a",
        "client_b",
    ]
    assert [row["話速"] for row in two_client_rows] == [1.0, 0.9, 1.1]
    assert [row["音量"] for row in two_client_rows] == [0.6, 0.7, 0.8]


def test_runtime_resolved_audio_settings_rows_read_latest_start_event() -> None:
    rows = streamlit_app.runtime_resolved_audio_settings_rows(
        [
            {
                "event_type": "runtime_audio_settings_resolved",
                "details": {
                    "speakers": [
                        {
                            "speaker_id": "counselor",
                            "display_name": "佐伯",
                            "voice": "shimmer",
                            "realtime_output_speed": 0.9,
                            "audio_gain": 0.6,
                        }
                    ]
                },
            },
            {
                "event_type": "runtime_audio_settings_resolved",
                "details": {
                    "speakers": [
                        {
                            "speaker_id": "counselor",
                            "display_name": "佐伯",
                            "voice": "shimmer",
                            "realtime_output_speed": 1.0,
                            "audio_gain": 0.7,
                        },
                        {
                            "speaker_id": "client_a",
                            "display_name": "妻",
                            "voice": "marin",
                            "realtime_output_speed": 0.9,
                            "audio_gain": 0.6,
                        },
                    ]
                },
            },
        ]
    )

    assert rows == [
        {
            "話者ID": "counselor",
            "表示名": "佐伯",
            "音声": "shimmer",
            "話速": 1.0,
            "音量": 0.7,
        },
        {
            "話者ID": "client_a",
            "表示名": "妻",
            "音声": "marin",
            "話速": 0.9,
            "音量": 0.6,
        },
    ]


def test_runtime_debug_panel_shows_pending_and_resolved_audio_settings() -> None:
    source = Path("app/streamlit_app.py").read_text(encoding="utf-8")
    runtime_panel_source = source[
        source.index("def render_runtime_session_panel") : source.index(
            "def render_runtime_debug_panel"
        )
    ]
    debug_source = source[
        source.index("def render_runtime_debug_panel") : source.index(
            "def render_runtime_observer_panel"
        )
    ]

    assert "runtime_audio_settings_rows_from_start_options(" in runtime_panel_source
    assert 'st.markdown("##### 音声設定")' in debug_source
    assert '"次回セッション開始時"' in debug_source
    assert "runtime_pending_audio_settings_rows" in debug_source
    assert "runtime_resolved_audio_settings_rows(events)" in debug_source
    assert '"取得済みセッションの実効値"' in debug_source


def test_main_uses_runtime_ui_instead_of_v02_queue_controls() -> None:
    source = Path("app/streamlit_app.py").read_text(encoding="utf-8")
    main_source = source[
        source.index("def main()") : source.index("def render_mode_selector")
    ]
    selector_source = source[
        source.index("def render_mode_selector") : source.index("def render_timer_band")
    ]

    assert "render_mode_selector(config.app.enabled_modes)" in main_source
    assert "render_runtime_session_panel(" in main_source
    assert "render_live_transcript_panel(config)" in main_source
    assert "render_saved_session_replay(config)" in main_source
    assert "render_runtime_observer_panel()" in main_source
    assert main_source.index("render_runtime_session_panel(") < main_source.index(
        "render_live_transcript_panel(config)"
    )
    assert main_source.index("render_live_transcript_panel(config)") < main_source.index(
        "render_saved_session_replay(config)"
    )
    assert main_source.index("render_saved_session_replay(config)") < main_source.index(
        "render_runtime_observer_panel()"
    )
    assert "render_controls(" not in main_source
    assert "render_current_audio_player(" not in main_source
    assert "render_unplayed_preview(" not in main_source
    assert "render_mode_tabs" not in source
    assert "st.radio(" in selector_source
    assert "runtime_mode_selector" in selector_source
    assert "st.tabs(" not in selector_source
    assert "このモードを選択" not in source


def test_live_transcript_panel_keeps_downloads_without_duplicate_transcript_table() -> None:
    source = Path("app/streamlit_app.py").read_text(encoding="utf-8")
    live_source = source[
        source.index("def render_live_transcript_panel") : source.index(
            "def render_transcript_download_gap"
        )
    ]
    main_source = source[
        source.index("def main()") : source.index("def render_mode_selector")
    ]

    assert "render_transcripts(display_turns)" not in live_source
    assert (
        live_source.index("render_timer_band(config, turns=turns)")
        < live_source.index("render_transcript_download_gap()")
    )
    assert live_source.index("render_transcript_download_gap()") < live_source.index(
        "render_script_downloads()"
    )
    assert main_source.index("render_live_transcript_panel(config)") < main_source.index(
        "render_saved_session_replay(config)"
    )


def test_live_transcript_panel_throttles_against_playback_completed_turns() -> None:
    source = Path("app/streamlit_app.py").read_text(encoding="utf-8")
    live_source = source[
        source.index("def render_live_transcript_panel") : source.index(
            "def render_transcript_download_gap"
        )
    ]
    throttle_call = live_source[
        live_source.index("runtime_generation_throttle_decision(") : live_source.index(
            "    st.session_state[\"runtime_generation_lead_turns\"]"
        )
    ]

    assert "runtime_completed_turn_count_state_updates(" in live_source
    assert "playback_completed_turn_count = int(" in live_source
    assert (
        "playback_completed_turn_count=playback_completed_turn_count"
        in throttle_call
    )
    assert "runtime_human_input_enabled()" in live_source
    assert "or human_counselor_mode" in throttle_call
    assert "runtime_transcript_visible_turn_count" not in throttle_call


def test_timer_band_labels_elapsed_time_instead_of_wall_clock() -> None:
    source = Path("app/streamlit_app.py").read_text(encoding="utf-8")
    timer_source = source[
        source.index("def render_timer_band") : source.index("def render_controls")
    ]

    assert '"経過時間"' in timer_source
    assert '"生成ターン数"' in timer_source
    assert '"現在ターン"' in timer_source
    assert '"完了ターン数"' in timer_source
    assert '"経過ターン数"' not in timer_source
    human_timer_source = timer_source[
        timer_source.index("if human_counselor_mode:") : timer_source.index(
            "else:", timer_source.index("if human_counselor_mode:")
        )
    ]
    ai_timer_source = timer_source[
        timer_source.index("else:", timer_source.index("if human_counselor_mode:")) :
    ]
    assert '"完了ターン数"' not in human_timer_source
    assert "remaining_label" not in human_timer_source
    assert "remaining_value" not in human_timer_source
    assert 'st.columns(3)' in human_timer_source
    assert "include_active_turn=not human_counselor_mode" in timer_source
    assert "runtime_current_turn_metric_value(status)" in timer_source
    assert "runtime_completed_turn_count_state_updates(" in timer_source
    assert "runtime_status_label_for_timer(" in timer_source
    assert "render_timer_band(config, turns=turns)" in source
    assert '"累積音声"' not in timer_source
    assert "runtime_remaining_metric(" in timer_source
    assert ai_timer_source.index('columns[0].metric("経過時間"') < ai_timer_source.index(
        "columns[1].metric(remaining_label"
    )
    assert ai_timer_source.index("columns[1].metric(remaining_label") < ai_timer_source.index(
        "columns[2].metric("
    )
    assert ai_timer_source.index("columns[2].metric(") < ai_timer_source.index(
        'columns[3].metric("完了ターン数"'
    )
    assert ai_timer_source.index('columns[3].metric("完了ターン数"') < ai_timer_source.index(
        'columns[4].metric("状態"'
    )
    assert '"壁時計"' not in timer_source
    assert '"フェーズ"' not in timer_source


def test_transcript_download_gap_matches_section_spacing() -> None:
    source = Path("app/streamlit_app.py").read_text(encoding="utf-8")
    gap_source = source[
        source.index("def render_transcript_download_gap") : source.index(
            "def render_script_downloads"
        )
    ]

    assert 'height: 24px;' in gap_source
    assert "unsafe_allow_html=True" in gap_source


def test_current_transcript_table_shows_six_rows_before_scrolling() -> None:
    source = Path("app/streamlit_app.py").read_text(encoding="utf-8")
    transcript_source = source[
        source.index("def render_transcripts") : source.index(
            "@st.fragment(run_every=RUNTIME_TRANSCRIPT_REFRESH_INTERVAL_SECONDS)"
        )
    ]

    assert "large_body_text=True" in transcript_source
    assert "max_visible_body_rows=6" in transcript_source


def test_runtime_transcript_refresh_interval_keeps_reveal_latency_low() -> None:
    assert RUNTIME_TRANSCRIPT_REFRESH_INTERVAL_SECONDS == 0.1


def test_session_setup_is_collapsed_like_saved_sessions_and_debug() -> None:
    source = Path("app/streamlit_app.py").read_text(encoding="utf-8")
    settings_source = source[
        source.index("def render_settings_panel") : source.index(
            "def prepare_runtime_text_defaults"
        )
    ]

    assert 'with st.expander("Session setup", expanded=False):' in settings_source
    assert 'st.subheader("Session setup")' not in settings_source
    assert '"停止条件"' not in settings_source
    assert '"セッション時間（秒）"' not in settings_source
    assert '"相談テーマ"' not in settings_source
    assert "runtime_stop_condition" not in settings_source
    assert "runtime_session_duration_seconds" not in settings_source
    assert "runtime_max_turns" not in settings_source
    assert settings_source.index('"初回クライアント発話"') < settings_source.index(
        'with columns[1]:'
    )
    assert "st.radio(" in settings_source
    assert '"参加クライアント数"' in settings_source
    assert "st.segmented_control(" not in settings_source
    assert '"設定タブ"' not in settings_source
    assert "sync_runtime_setup_tab_to_participant_mode" not in settings_source
    assert "sync_runtime_participant_mode_to_setup_tab" not in settings_source
    assert 'if participant_mode == "two_clients":' in settings_source
    assert "render_one_client_setup_tab" in settings_source
    assert "render_two_client_setup_tab" in settings_source
    assert "runtime_closing_start_seconds" in settings_source
    assert '"クロージング開始（秒）"' in settings_source
    assert '"Realtime model"' in settings_source
    assert "runtime_realtime_model" in settings_source
    assert "RUNTIME_REALTIME_MODEL_OPTIONS" in settings_source
    # The text fields are rendered per route; their behavior is covered by AppTest.
    assert "prepare_runtime_text_provider_defaults" in settings_source
    assert "RUNTIME_TEXT_ROUTE_CONTROLS.items()" in settings_source
    assert '"カウンセラー音声"' in settings_source
    assert '"クライアント音声"' in settings_source
    assert '"カウンセラー話速"' in settings_source
    assert '"クライアント話速"' in settings_source
    assert "runtime_counselor_tts_voice" in settings_source
    assert "runtime_client_tts_voice" in settings_source
    assert "runtime_counselor_output_speed" in settings_source
    assert "runtime_client_output_speed" in settings_source
    assert "runtime_tts_voice_options" in settings_source
    left_column_source = settings_source[
        settings_source.index("with columns[0]:") : settings_source.index(
            "with columns[1]:"
        )
    ]
    right_column_source = settings_source[
        settings_source.index("with columns[1]:") : settings_source.index(
            '    st.markdown("##### プロフィール")'
        )
    ]
    assert "render_runtime_model_session_controls()" in left_column_source
    assert "render_runtime_model_session_controls()" not in right_column_source
    assert "render_runtime_common_session_controls()" in right_column_source
    assert '"カウンセラー音声"' in left_column_source
    assert '"クライアント音声"' in left_column_source
    assert '"カウンセラー話速"' in left_column_source
    assert '"クライアント話速"' in left_column_source
    assert '"カウンセラー音量"' in left_column_source
    assert '"クライアント音量"' in left_column_source
    assert left_column_source.index('"カウンセラー音量"') < left_column_source.index(
        "render_runtime_model_session_controls()"
    )
    assert "min_value=1" in settings_source
    assert "runtime_force_stop_after_closing_turns" in settings_source
    assert '"クロージング最大ターン数（カウンセラー開始）"' in settings_source
    assert "runtime_generation_lead_limit" in settings_source
    assert '"先行生成上限（表示との差）"' in settings_source
    assert '"カウンセラー音量"' in settings_source
    assert '"クライアント音量"' in settings_source
    assert "runtime_counselor_audio_gain" in settings_source
    assert "runtime_client_audio_gain" in settings_source
    assert "render_prompt_file_loader" in settings_source
    assert '"##### プロフィール"' in settings_source
    assert '"##### プロンプト本文"' in settings_source
    assert '"カウンセラー公開プロフィール"' in settings_source
    assert '"カウンセラープロフィール"' not in settings_source
    assert '"クライアント公開プロフィール"' in settings_source
    assert '"クライアントプロフィール"' not in settings_source
    assert '"クライアント共通プロフィール"' not in settings_source
    assert "秘密プロフィール" in settings_source
    assert '"クライアント秘密プロフィール"' in settings_source
    assert "固有プロフィール" not in settings_source
    assert "非公開プロフィール" not in settings_source
    assert '"カウンセラー役プロフィール"' not in settings_source
    assert '"クライアント役プロフィール"' not in settings_source
    assert '"カウンセラープリセット"' in settings_source
    assert '"クライアントプリセット"' in settings_source
    assert settings_source.index('"カウンセラープリセット"') < settings_source.index(
        '"クライアントプリセット"'
    )
    assert "preset_columns = st.columns(2)" in settings_source
    assert '"カウンセラープリセットを再適用"' in settings_source
    assert '"クライアントプリセットを再適用"' in settings_source
    assert "selected_counselor_preset_id" in settings_source
    assert "selected_client_preset_id" in settings_source
    assert "apply_counselor_preset_to_session_state" in settings_source
    assert "apply_client_preset_to_session_state" in settings_source
    assert '"カウンセラー公開プロフィール読込（.md/.txt）"' in settings_source
    assert '"クライアント公開プロフィール読込（.md/.txt）"' in settings_source
    assert '"クライアント秘密プロフィール読込（.md/.txt）"' in settings_source
    assert '"クライアント共通プロフィール読込（.md/.txt）"' not in settings_source
    assert '"カウンセラー役プロンプト読込（.md/.txt）"' in settings_source
    assert '"クライアント役プロンプト読込（.md/.txt）"' in settings_source
    assert '"クライアント共通プロンプト読込（.md/.txt）"' in settings_source
    assert "counselor_public_profile_file_upload" in settings_source
    assert "client_public_profile_file_upload" in settings_source
    assert "client_private_profile_file_upload" in settings_source
    assert "counselor_public_profile_file_signature" in settings_source
    assert "client_public_profile_file_signature" in settings_source
    assert "client_private_profile_file_signature" in settings_source
    assert "counselor_prompt_file_upload" in settings_source
    assert "client_prompt_file_upload" in settings_source
    assert "counselor_prompt_file_signature" in settings_source
    assert "client_prompt_file_signature" in settings_source
    assert "counselor_public_profile_text" in settings_source
    assert "client_public_profile_text" in settings_source
    assert "client_private_profile_text" in settings_source
    assert "runtime_participant_mode" in settings_source
    assert "RUNTIME_PARTICIPANT_MODE_LABELS" in settings_source
    assert '"1クライアント"' in source
    assert '"2クライアント"' in source
    assert '"クライアント公開プロフィール"' in settings_source
    assert '"クライアントA"' in settings_source
    assert '"クライアントB"' in settings_source
    assert "render_two_client_settings_panel" in settings_source
    assert 'st.tabs(["クライアントA", "クライアントB"])' not in settings_source
    one_client_setup_source = settings_source[
        settings_source.index("def render_one_client_setup_tab")
        : settings_source.index("def render_two_client_setup_tab")
    ]
    two_client_setup_source = settings_source[
        settings_source.index("def render_two_client_setup_tab")
        : settings_source.index("def render_runtime_common_session_controls")
    ]
    assert one_client_setup_source.index(
        '"カウンセラー音量"'
    ) < one_client_setup_source.index("render_runtime_model_session_controls()")
    assert two_client_setup_source.index(
        '"カウンセラー音量"'
    ) < two_client_setup_source.index("render_runtime_model_session_controls()")
    assert two_client_setup_source.index(
        "prepare_two_client_runtime_widget_defaults("
    ) < two_client_setup_source.index(
        'st.text_area(\n            "クライアント公開プロフィール"'
    )
    assert two_client_setup_source.index(
        "prepare_two_client_runtime_widget_defaults("
    ) < two_client_setup_source.index(
        'st.text_area(\n            "クライアント共通プロンプト"'
    )
    assert '"カウンセラー役プロフィール"' not in one_client_setup_source
    assert one_client_setup_source.index(
        '"カウンセラー公開プロフィール読込（.md/.txt）"'
    ) < one_client_setup_source.index('"カウンセラー公開プロフィール"')
    assert one_client_setup_source.index(
        '"クライアント公開プロフィール読込（.md/.txt）"'
    ) < one_client_setup_source.index('"クライアント公開プロフィール"')
    assert one_client_setup_source.index('"クライアント公開プロフィール"') < (
        one_client_setup_source.index('"クライアント秘密プロフィール読込（.md/.txt）"')
    )
    assert one_client_setup_source.index(
        '"クライアント秘密プロフィール読込（.md/.txt）"'
    ) < one_client_setup_source.index('"クライアント秘密プロフィール"')
    assert '"カウンセラー役プロフィール"' not in two_client_setup_source
    assert two_client_setup_source.index(
        '"カウンセラー公開プロフィール読込（.md/.txt）"'
    ) < two_client_setup_source.index('"カウンセラー公開プロフィール"')
    assert two_client_setup_source.index(
        '"クライアント公開プロフィール読込（.md/.txt）"'
    ) < two_client_setup_source.index(
        'st.text_area(\n            "クライアント公開プロフィール"'
    )
    assert (
        'st.text_area(\n            "カウンセラー役プロンプト",\n'
        '            key=sync_runtime_widget_from_state("counselor_prompt_text"),\n'
        '            height=220,\n'
        in two_client_setup_source
    )
    assert (
        'st.text_area(\n            "クライアント共通プロンプト",\n'
        '            key="client_prompt_text",\n            height=220,\n'
        in two_client_setup_source
    )
    assert one_client_setup_source.index('"初回クライアント発話"') < (
        one_client_setup_source.index('"##### プロフィール"')
    )
    assert one_client_setup_source.index('"##### プロフィール"') < (
        one_client_setup_source.index('"##### プロンプト本文"')
    )
    assert two_client_setup_source.index('"##### カウンセラー設定"') < (
        two_client_setup_source.index('"##### プロフィール"')
    )
    assert two_client_setup_source.index('"##### プロフィール"') < (
        two_client_setup_source.index('"##### プロンプト本文"')
    )
    for client_key in ("client_a", "client_b"):
        assert f"{client_key}_display_name" in settings_source
        assert f"{client_key}_public_profile_text" in settings_source
        assert f"{client_key}_public_profile_file_upload" in settings_source
        assert f"{client_key}_public_profile_file_signature" in settings_source
        assert f"{client_key}_prompt_text" in settings_source
        assert f"{client_key}_prompt_file_upload" in settings_source
        assert f"{client_key}_prompt_source_profile_id" in settings_source
        assert f"{client_key}_prompt_file_signature" in settings_source
        assert f"runtime_initial_{client_key}_transcript" in settings_source
        assert f"runtime_{client_key}_tts_voice" in settings_source
        assert f"runtime_{client_key}_output_speed" in settings_source
        assert f"runtime_{client_key}_audio_gain" in settings_source
    assert "mark_runtime_widget_user_modified" in settings_source
    assert "sync_runtime_widget_from_state" in settings_source
    assert "runtime_widget_key" in settings_source
    assert 'settings["display_name_key"],' in settings_source
    assert 'settings["voice_key"],' in settings_source
    assert 'settings["speed_key"],' in settings_source
    assert 'settings["gain_key"],' in settings_source
    assert 'settings["public_text_key"],' in settings_source
    assert 'settings["prompt_text_key"],' in settings_source


def test_runtime_voice_options_exclude_noisy_sage_voice(monkeypatch) -> None:
    monkeypatch.setattr(
        streamlit_app,
        "runtime_voice_defaults",
        lambda: {"counselor": "shimmer", "client": "cedar"},
    )

    options = streamlit_app.runtime_voice_options(SimpleNamespace())

    assert options[:10] == [
        "alloy",
        "marin",
        "cedar",
        "coral",
        "ash",
        "ballad",
        "echo",
        "shimmer",
        "verse",
    ]
    assert "sage" not in options


def test_saved_sessions_replay_remains_separate_from_current_transcript() -> None:
    source = Path("app/streamlit_app.py").read_text(encoding="utf-8")
    current_source = source[
        source.index("def current_public_transcript_turns") : source.index(
            "def display_public_transcript_turns"
        )
    ]
    replay_source = source[
        source.index("def render_saved_session_replay") : source.index(
            "def render_unplayed_preview"
        )
    ]

    assert "runtime_monitor_session_id" in current_source
    assert "selected_replay_session_dir" not in current_source
    assert "pending_saved_replay_session_id(" in replay_source
    assert "を保存中です" in replay_source
    assert 'with st.expander("Saved sessions", expanded=False):' in replay_source
    assert "selected_replay_session_dir" in replay_source
    assert "session_path = Path(selected_dir)" in replay_source
    assert "load_public_transcript_turns(session_path)" in replay_source
    assert "session_realtime_audio_path(session_path)" in replay_source
    assert "session_has_human_audio_turns(artifact_turns)" in replay_source
    assert "build_compact_session_audio(session_path, artifact_turns)" in replay_source
    assert "Start from start time" in replay_source
    assert "selected_replay_start_time_key" in replay_source
    assert "Start from turn" not in replay_source
    assert "selected_replay_start_turn" not in replay_source
    assert "Download replay script (.md)" in replay_source
    assert "Download replay script (.txt)" in replay_source
    assert "Download session audio (.wav)" in replay_source
    assert "Download compact session audio (.wav)" in replay_source
    assert 'mime="audio/wav"' in replay_source
    assert "session_audio_path.read_bytes()" in replay_source
    assert "session_compact_audio" in replay_source
    assert 'format_public_script(replay_turns, format="txt")' in replay_source
    assert 'mime="text/plain"' in replay_source
    assert "start_time=start_seconds" in replay_source
    assert "for turn in replay_turns:" not in replay_source


def test_pending_saved_replay_session_id_reports_until_replay_artifacts_exist(
    tmp_path: Path,
) -> None:
    status = {"session_id": "session-new", "phase": "completed"}

    assert (
        pending_saved_replay_session_id(
            sessions_root=tmp_path,
            sessions=[],
            current_session_id="session-new",
            runtime_status=status,
        )
        == "session-new"
    )

    session_dir = tmp_path / "session-new"
    transcript_dir = session_dir / "internal" / "transcripts"
    audio_dir = session_dir / "internal" / "audio"
    transcript_dir.mkdir(parents=True)
    audio_dir.mkdir(parents=True)
    sessions = [session_dir]

    assert not saved_session_replay_ready(session_dir)
    assert (
        pending_saved_replay_session_id(
            sessions_root=tmp_path,
            sessions=sessions,
            current_session_id="session-new",
            runtime_status=status,
        )
        == "session-new"
    )

    (transcript_dir / "session-new.jsonl").write_text(
        json.dumps(
            {
                "created_at": "2026-06-29T00:00:00+00:00",
                "session_id": "session-new",
                "turn_id": 0,
                "speaker": "counselor",
                "speaker_id": "counselor",
                "text": "保存済みセッションです。",
                "transcript_type": "generated_final",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    (audio_dir / "session_realtime.wav").touch()

    assert saved_session_replay_ready(session_dir)
    assert (
        pending_saved_replay_session_id(
            sessions_root=tmp_path,
            sessions=sessions,
            current_session_id="session-new",
            runtime_status=status,
        )
        == ""
    )


def test_saved_session_replay_ready_waits_for_human_turn_audio(
    tmp_path: Path,
) -> None:
    session_dir = tmp_path / "session-human-audio-pending"
    transcript_dir = session_dir / "internal" / "transcripts"
    audio_dir = session_dir / "internal" / "audio"
    transcript_dir.mkdir(parents=True)
    audio_dir.mkdir(parents=True)
    (transcript_dir / "session-human-audio-pending.jsonl").write_text(
        json.dumps(
            {
                "created_at": "2026-06-29T00:00:00+00:00",
                "session_id": "session-human-audio-pending",
                "turn_id": 1,
                "speaker": "counselor",
                "speaker_id": "counselor",
                "text": "ここで確認します。",
                "transcript_type": "human_final",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    (audio_dir / "session_realtime.wav").touch()

    assert not saved_session_replay_ready(session_dir)

    (audio_dir / "turn_0001_counselor.wav").touch()

    assert saved_session_replay_ready(session_dir)


def test_pending_saved_replay_session_id_ignores_idle_or_different_status_session(
    tmp_path: Path,
) -> None:
    assert (
        pending_saved_replay_session_id(
            sessions_root=tmp_path,
            sessions=[],
            current_session_id="session-new",
            runtime_status={"session_id": "session-new", "phase": "idle"},
        )
        == ""
    )
    assert (
        pending_saved_replay_session_id(
            sessions_root=tmp_path,
            sessions=[],
            current_session_id="session-new",
            runtime_status={"session_id": "session-other", "phase": "completed"},
        )
        == ""
    )


def test_runtime_audio_monitor_document_embeds_initial_args() -> None:
    document = runtime_audio_monitor_document(
        ws_url="ws://127.0.0.1:8765/runtime/monitor-audio",
        volume=0.45,
        muted=True,
        start_url="http://127.0.0.1:8765/runtime/start",
        runtime_pause_url="http://127.0.0.1:8765/runtime/pause",
        runtime_resume_url="http://127.0.0.1:8765/runtime/resume",
        runtime_stop_url="http://127.0.0.1:8765/runtime/stop",
        runtime_interrupt_url="http://127.0.0.1:8765/runtime/interrupt",
        runtime_playback_completed_url=(
            "http://127.0.0.1:8765/runtime/playback-completed"
        ),
        runtime_status_url="http://127.0.0.1:8765/runtime/status",
        human_audio_url="http://127.0.0.1:8765/runtime/human-audio",
        human_audio_stream_url="ws://127.0.0.1:8765/runtime/human-audio-stream",
        human_turn_url="http://127.0.0.1:8765/runtime/human-turn",
        human_input_enabled=True,
        warmup_target_turns=5,
        warmup_max_wait_ms=24000,
        start_options={
            "closing_start_elapsed_seconds": 120,
            "fixed_speaker_sequence": [
                "counselor",
                "client_b",
                "client_a",
                "counselor",
                "client_a",
                "client_b",
            ],
        },
    )

    assert "window.RUNTIME_AUDIO_MONITOR_ARGS" in document
    assert '"ws_url": "ws://127.0.0.1:8765/runtime/monitor-audio"' in document
    assert '"volume": 0.45' in document
    assert '"muted": true' in document
    assert '"start_url": "http://127.0.0.1:8765/runtime/start"' in document
    assert '"runtime_pause_url": "http://127.0.0.1:8765/runtime/pause"' in document
    assert '"runtime_resume_url": "http://127.0.0.1:8765/runtime/resume"' in document
    assert '"runtime_stop_url": "http://127.0.0.1:8765/runtime/stop"' in document
    assert (
        '"runtime_interrupt_url": "http://127.0.0.1:8765/runtime/interrupt"'
        in document
    )
    assert (
        '"runtime_playback_completed_url": '
        '"http://127.0.0.1:8765/runtime/playback-completed"'
        in document
    )
    assert '"runtime_status_url": "http://127.0.0.1:8765/runtime/status"' in document
    assert '"human_audio_url": "http://127.0.0.1:8765/runtime/human-audio"' in document
    assert (
        '"human_audio_stream_url": '
        '"ws://127.0.0.1:8765/runtime/human-audio-stream"'
        in document
    )
    assert '"human_turn_url": "http://127.0.0.1:8765/runtime/human-turn"' in document
    assert '"human_input_enabled": true' in document
    assert '"warmup_target_turns": 5' in document
    assert '"warmup_max_wait_ms": 24000' in document
    assert '"pause_url"' not in document
    assert '"closing_start_elapsed_seconds": 120' in document
    assert (
        '"fixed_speaker_sequence": ["counselor", "client_b", "client_a", '
        '"counselor", "client_a", "client_b"]'
        in document
    )
    assert "Connect & Resume audio" not in document
    assert '<button id="connect"' not in document
    assert '<button id="disconnect"' not in document
    assert "Connect & Start audio" in document
    assert '<button id="pause-runtime" type="button">Pause</button>' in document
    assert '<button id="resume-runtime" type="button">Resume</button>' in document
    assert (
        '<button id="recover-runtime" type="button">Refresh & Reconnect</button>'
        in document
    )


def test_runtime_audio_monitor_has_human_microphone_input_gain_control() -> None:
    component_source = Path(
        "app/components/runtime_audio_monitor/index.html"
    ).read_text(encoding="utf-8")

    assert 'id="human-input-gain"' in component_source
    assert "Mic gain" in component_source
    assert "HUMAN_INPUT_GAIN_STORAGE_KEY" in component_source
    assert "humanInputGainValue()" in component_source
    assert "preferredHumanAudioBufferChannelIndex" in component_source
    assert "HUMAN_CHANNEL_MIX_MIN_AVERAGE_POWER_RATIO" in component_source
    assert "mixed * inputGain" in component_source
    assert "(merged[sourceIndex] || 0) * inputGain" in component_source


def test_runtime_audio_monitor_has_always_sensitivity_control() -> None:
    component_source = Path(
        "app/components/runtime_audio_monitor/index.html"
    ).read_text(encoding="utf-8")

    assert 'id="human-always-sensitivity"' in component_source
    assert "Always sens" in component_source
    assert "HUMAN_ALWAYS_SENSITIVITY_STORAGE_KEY" in component_source
    assert "HUMAN_ALWAYS_SENSITIVITY_PROFILES" in component_source
    assert "humanAlwaysSensitivitySettings()" in component_source
    assert "settings.minActiveMs" in component_source


def test_runtime_audio_monitor_interrupt_payload_uses_turn_level_playback_progress() -> None:
    component_source = Path(
        "app/components/runtime_audio_monitor/index.html"
    ).read_text(encoding="utf-8")
    turn_end_body = component_source.split(
        "function handleTurnEndMessage(message) {", maxsplit=1
    )[1].split("function handlePlaybackMessage(message) {", maxsplit=1)[0]
    completion_body = component_source.split(
        "function emitPlaybackCompletedIfReady(key) {", maxsplit=1
    )[1].split("function interruptPayloadForSegment(segment", maxsplit=1)[0]

    assert "const mainPlaybackProgressByKey = new Map();" in component_source
    assert "updateMainPlaybackProgress(key, playback);" in component_source
    assert "function playedSecondsForInterruptedSegment(segment)" in component_source
    assert (
        "Math.round(playedSecondsForInterruptedSegment(segment) * 1000)"
        in component_source
    )
    assert "mainPlaybackProgressByKey.delete(playbackKey)" not in turn_end_body
    assert "mainPlaybackProgressByKey.delete(key);" in completion_body


def test_unplayed_preview_rows_include_hold_reason_and_warning() -> None:
    rows = unplayed_preview_rows(_sample_unplayed_turns())

    assert rows[1]["再生"] == TurnStatus.PLAYBACK_HOLD.value
    assert rows[1]["警告"] == WarningLevel.HIGH.value
    assert rows[1]["hold理由"] == "moderation_flagged"


def test_unplayed_table_layout_prioritizes_utterance_and_wraps_text() -> None:
    html = html_table(
        unplayed_preview_rows(_sample_unplayed_turns()),
        columns=UNPLAYED_TABLE_COLUMNS,
    )

    assert 'width: 38%;' in html
    assert 'width: 5%;' in html
    assert 'class="utterance wrap"' in html
    assert 'overflow-wrap: anywhere;' in html
    assert 'word-break: break-word;' in html


def test_transcript_rows_and_table_escape_wrapped_text() -> None:
    rows = transcript_rows_for_display(
        [
            {
                "turn_id": 7,
                "speaker_name": "佐伯 <script>",
                "speaker_role": "counselor",
                "text": "長い発話 <unsafe>",
                "spoken_at": "-",
                "cumulative_audio_seconds": 0,
            }
        ]
    )
    html = html_table(rows, columns=TRANSCRIPT_TABLE_COLUMNS)

    assert "7" in html
    assert "佐伯 &lt;script&gt;" in html
    assert "長い発話 &lt;unsafe&gt;" in html
    assert 'class="speaker wrap"' in html
    assert 'class="utterance wrap"' in html
    assert "累積音声" not in html


def test_public_transcript_rows_use_time_speaker_and_utterance_only() -> None:
    rows = public_transcript_rows_for_display(
        [
            PublicTranscriptTurn(
                turn_id=1,
                speaker="佐伯 <unsafe>",
                text="発話 <script>",
                created_at="2026-06-23T08:07:06+09:00",
                audio_path=None,
            )
        ]
    )
    html = html_table(rows, columns=PUBLIC_TRANSCRIPT_TABLE_COLUMNS)

    assert rows == [
        {
            "開始時刻": "00:00",
            "話者": "佐伯 <unsafe>",
            "発話": "発話 <script>",
        }
    ]
    assert "turn_id" not in rows[0]
    assert "役割" not in rows[0]
    assert "佐伯 &lt;unsafe&gt;" in html
    assert "発話 &lt;script&gt;" in html


def test_public_transcript_rows_prefer_cumulative_audio_time() -> None:
    rows = public_transcript_rows_for_display(
        [
            PublicTranscriptTurn(
                turn_id=2,
                speaker="佐伯",
                text="累積時間で表示します。",
                created_at="2026-06-23T08:07:06+09:00",
                audio_path=None,
                cumulative_audio_seconds=125.9,
            )
        ]
    )

    assert rows[0]["開始時刻"] == "02:05"


def test_public_transcript_total_audio_seconds_uses_cumulative_start_time() -> None:
    total_seconds = public_transcript_total_audio_seconds(
        [
            PublicTranscriptTurn(
                turn_id=1,
                speaker="佐伯",
                text="無音込みの開始秒を使います。",
                created_at=None,
                audio_path=None,
                cumulative_audio_seconds=5.0,
            )
        ]
    )

    assert total_seconds == 5.0


def test_public_transcript_turns_visible_after_audio_end(tmp_path: Path) -> None:
    first_audio = tmp_path / "turn_0001_counselor.wav"
    second_audio = tmp_path / "turn_0002_client.wav"
    _write_silent_wav(first_audio, duration_seconds=1.0)
    _write_silent_wav(second_audio, duration_seconds=1.0)
    turns = [
        PublicTranscriptTurn(
            turn_id=1,
            speaker="佐伯",
            text="最初の発話です。",
            created_at=None,
            audio_path=first_audio,
            cumulative_audio_seconds=0.0,
        ),
        PublicTranscriptTurn(
            turn_id=2,
            speaker="高橋",
            text="次の発話です。",
            created_at=None,
            audio_path=second_audio,
            cumulative_audio_seconds=1.0,
        ),
    ]

    visible_turns = public_transcript_turns_visible_after_audio_end(
        turns,
        playback_seconds=1.5,
    )

    assert [turn.turn_id for turn in visible_turns] == [1]
    assert [
        turn.turn_id
        for turn in public_transcript_turns_visible_after_audio_end(
            turns,
            playback_seconds=2.0,
        )
    ] == [1, 2]


def test_public_transcript_turns_visible_after_previous_audio_end(
    tmp_path: Path,
) -> None:
    first_audio = tmp_path / "turn_0001_client_a.wav"
    second_audio = tmp_path / "turn_0002_client_b.wav"
    _write_silent_wav(first_audio, duration_seconds=1.0)
    _write_silent_wav(second_audio, duration_seconds=1.0)
    turns = [
        PublicTranscriptTurn(
            turn_id=1,
            speaker="妻",
            text="最初の発話です。",
            created_at=None,
            audio_path=first_audio,
            cumulative_audio_seconds=0.0,
        ),
        PublicTranscriptTurn(
            turn_id=2,
            speaker="夫",
            text="次の発話です。",
            created_at=None,
            audio_path=second_audio,
            cumulative_audio_seconds=1.0,
        ),
    ]

    assert [
        turn.turn_id
        for turn in public_transcript_turns_visible_after_previous_audio_end(
            turns,
            playback_seconds=0.5,
        )
    ] == [1]
    assert [
        turn.turn_id
        for turn in public_transcript_turns_visible_after_previous_audio_end(
            turns,
            playback_seconds=1.0,
        )
    ] == [1, 2]
    assert [
        turn.turn_id
        for turn in runtime_public_transcript_turns_visible_for_audio(
            turns,
            playback_seconds=1.0,
        )
    ] == [1, 2]


def test_public_transcript_turns_visible_after_previous_audio_end_respects_gap(
    tmp_path: Path,
) -> None:
    first_audio = tmp_path / "turn_0001_client_a.wav"
    second_audio = tmp_path / "turn_0002_client_b.wav"
    _write_silent_wav(first_audio, duration_seconds=1.0)
    _write_silent_wav(second_audio, duration_seconds=1.0)
    turns = [
        PublicTranscriptTurn(
            turn_id=1,
            speaker="妻",
            text="最初の発話です。",
            created_at=None,
            audio_path=first_audio,
            cumulative_audio_seconds=0.0,
        ),
        PublicTranscriptTurn(
            turn_id=2,
            speaker="夫",
            text="少し間を置いた発話です。",
            created_at=None,
            audio_path=second_audio,
            cumulative_audio_seconds=3.0,
        ),
    ]

    assert [
        turn.turn_id
        for turn in public_transcript_turns_visible_after_previous_audio_end(
            turns,
            playback_seconds=1.0,
        )
    ] == [1]
    assert [
        turn.turn_id
        for turn in public_transcript_turns_visible_after_previous_audio_end(
            turns,
            playback_seconds=3.0,
        )
    ] == [1, 2]


def test_runtime_transcript_visibility_state_updates_reveals_one_turn_per_audio_duration(
    tmp_path: Path,
) -> None:
    first_audio = tmp_path / "turn_0001_client_a.wav"
    second_audio = tmp_path / "turn_0002_client_b.wav"
    third_audio = tmp_path / "turn_0003_counselor.wav"
    _write_silent_wav(first_audio, duration_seconds=1.0)
    _write_silent_wav(second_audio, duration_seconds=1.0)
    _write_silent_wav(third_audio, duration_seconds=1.0)
    first_turn = PublicTranscriptTurn(
        turn_id=1,
        speaker="妻",
        text="最初の発話です。",
        created_at=None,
        audio_path=first_audio,
        cumulative_audio_seconds=0.0,
    )
    second_turn = PublicTranscriptTurn(
        turn_id=2,
        speaker="夫",
        text="次の発話です。",
        created_at=None,
        audio_path=second_audio,
        cumulative_audio_seconds=1.0,
    )
    third_turn = PublicTranscriptTurn(
        turn_id=3,
        speaker="佐伯",
        text="まだ表示しない発話です。",
        created_at=None,
        audio_path=third_audio,
        cumulative_audio_seconds=2.0,
    )

    first_updates = runtime_transcript_visibility_state_updates(
        session_id="session-live",
        turns=[first_turn],
        wall_clock_seconds=12.0,
        visibility_session_id="",
        visible_turn_count=0,
        next_reveal_wall_seconds=None,
    )
    early_second_updates = runtime_transcript_visibility_state_updates(
        session_id="session-live",
        turns=[first_turn, second_turn, third_turn],
        wall_clock_seconds=12.1,
        visibility_session_id=first_updates["runtime_transcript_visibility_session_id"],
        visible_turn_count=first_updates["runtime_transcript_visible_turn_count"],
        next_reveal_wall_seconds=first_updates[
            "runtime_transcript_next_reveal_wall_seconds"
        ],
    )
    still_waiting_second_updates = runtime_transcript_visibility_state_updates(
        session_id="session-live",
        turns=[first_turn, second_turn, third_turn],
        wall_clock_seconds=13.0,
        visibility_session_id=first_updates["runtime_transcript_visibility_session_id"],
        visible_turn_count=first_updates["runtime_transcript_visible_turn_count"],
        next_reveal_wall_seconds=first_updates[
            "runtime_transcript_next_reveal_wall_seconds"
        ],
    )
    second_updates = runtime_transcript_visibility_state_updates(
        session_id="session-live",
        turns=[first_turn, second_turn, third_turn],
        wall_clock_seconds=14.6,
        visibility_session_id=first_updates["runtime_transcript_visibility_session_id"],
        visible_turn_count=first_updates["runtime_transcript_visible_turn_count"],
        next_reveal_wall_seconds=first_updates[
            "runtime_transcript_next_reveal_wall_seconds"
        ],
    )
    third_too_early_updates = runtime_transcript_visibility_state_updates(
        session_id="session-live",
        turns=[first_turn, second_turn, third_turn],
        wall_clock_seconds=14.7,
        visibility_session_id=first_updates["runtime_transcript_visibility_session_id"],
        visible_turn_count=second_updates["runtime_transcript_visible_turn_count"],
        next_reveal_wall_seconds=second_updates[
            "runtime_transcript_next_reveal_wall_seconds"
        ],
    )

    assert first_updates["runtime_transcript_visible_turn_count"] == 1
    assert first_updates["runtime_transcript_next_reveal_wall_seconds"] == pytest.approx(14.6)
    assert early_second_updates["runtime_transcript_visible_turn_count"] == 1
    assert still_waiting_second_updates["runtime_transcript_visible_turn_count"] == 1
    assert second_updates["runtime_transcript_visible_turn_count"] == 2
    assert second_updates["runtime_transcript_next_reveal_wall_seconds"] == pytest.approx(17.2)
    assert third_too_early_updates["runtime_transcript_visible_turn_count"] == 2


def test_runtime_transcript_visibility_schedule_does_not_accumulate_late_refresh(
    tmp_path: Path,
) -> None:
    first_audio = tmp_path / "turn_0001_client_a.wav"
    second_audio = tmp_path / "turn_0002_client_b.wav"
    third_audio = tmp_path / "turn_0003_counselor.wav"
    _write_silent_wav(first_audio, duration_seconds=1.0)
    _write_silent_wav(second_audio, duration_seconds=1.0)
    _write_silent_wav(third_audio, duration_seconds=1.0)
    turns = [
        PublicTranscriptTurn(
            turn_id=1,
            speaker="妻",
            text="最初の発話です。",
            created_at=None,
            audio_path=first_audio,
            cumulative_audio_seconds=0.0,
        ),
        PublicTranscriptTurn(
            turn_id=2,
            speaker="夫",
            text="次の発話です。",
            created_at=None,
            audio_path=second_audio,
            cumulative_audio_seconds=1.0,
        ),
        PublicTranscriptTurn(
            turn_id=3,
            speaker="佐伯",
            text="さらに次の発話です。",
            created_at=None,
            audio_path=third_audio,
            cumulative_audio_seconds=2.0,
        ),
    ]

    first_updates = runtime_transcript_visibility_state_updates(
        session_id="session-live",
        turns=turns[:1],
        wall_clock_seconds=12.0,
        visibility_session_id="",
        visible_turn_count=0,
        next_reveal_wall_seconds=None,
    )
    late_second_updates = runtime_transcript_visibility_state_updates(
        session_id="session-live",
        turns=turns,
        wall_clock_seconds=14.8,
        visibility_session_id=first_updates["runtime_transcript_visibility_session_id"],
        visible_turn_count=first_updates["runtime_transcript_visible_turn_count"],
        next_reveal_wall_seconds=first_updates[
            "runtime_transcript_next_reveal_wall_seconds"
        ],
    )
    third_updates = runtime_transcript_visibility_state_updates(
        session_id="session-live",
        turns=turns,
        wall_clock_seconds=17.2,
        visibility_session_id=first_updates["runtime_transcript_visibility_session_id"],
        visible_turn_count=late_second_updates[
            "runtime_transcript_visible_turn_count"
        ],
        next_reveal_wall_seconds=late_second_updates[
            "runtime_transcript_next_reveal_wall_seconds"
        ],
    )

    assert first_updates["runtime_transcript_next_reveal_wall_seconds"] == pytest.approx(14.6)
    assert late_second_updates["runtime_transcript_visible_turn_count"] == 2
    assert late_second_updates["runtime_transcript_next_reveal_wall_seconds"] == pytest.approx(17.2)
    assert third_updates["runtime_transcript_visible_turn_count"] == 3


def test_runtime_transcript_visibility_waits_when_audio_duration_is_unknown(
    tmp_path: Path,
) -> None:
    second_audio = tmp_path / "turn_0002_client_b.wav"
    _write_silent_wav(second_audio, duration_seconds=1.0)
    first_turn_without_audio = PublicTranscriptTurn(
        turn_id=1,
        speaker="妻",
        text="音声ファイルがまだ見えていない発話です。",
        created_at=None,
        audio_path=None,
        cumulative_audio_seconds=0.0,
    )
    second_turn = PublicTranscriptTurn(
        turn_id=2,
        speaker="夫",
        text="まだ表示しない発話です。",
        created_at=None,
        audio_path=second_audio,
        cumulative_audio_seconds=1.0,
    )

    first_updates = runtime_transcript_visibility_state_updates(
        session_id="session-live",
        turns=[first_turn_without_audio],
        wall_clock_seconds=12.0,
        visibility_session_id="",
        visible_turn_count=0,
        next_reveal_wall_seconds=None,
    )
    second_updates = runtime_transcript_visibility_state_updates(
        session_id="session-live",
        turns=[first_turn_without_audio, second_turn],
        wall_clock_seconds=30.0,
        visibility_session_id=first_updates["runtime_transcript_visibility_session_id"],
        visible_turn_count=first_updates["runtime_transcript_visible_turn_count"],
        next_reveal_wall_seconds=first_updates[
            "runtime_transcript_next_reveal_wall_seconds"
        ],
    )

    assert first_updates["runtime_transcript_visible_turn_count"] == 1
    assert first_updates["runtime_transcript_next_reveal_wall_seconds"] is None
    assert second_updates["runtime_transcript_visible_turn_count"] == 1


def test_runtime_transcript_visibility_uses_timeline_audio_duration_without_turn_wav() -> None:
    first_turn = PublicTranscriptTurn(
        turn_id=1,
        speaker="妻",
        text="最初の発話です。",
        created_at=None,
        audio_path=None,
        cumulative_audio_seconds=0.0,
        audio_duration_seconds=1.0,
    )
    second_turn = PublicTranscriptTurn(
        turn_id=2,
        speaker="夫",
        text="次の発話です。",
        created_at=None,
        audio_path=None,
        cumulative_audio_seconds=1.0,
        audio_duration_seconds=1.0,
    )

    first_updates = runtime_transcript_visibility_state_updates(
        session_id="session-live",
        turns=[first_turn, second_turn],
        wall_clock_seconds=12.0,
        visibility_session_id="",
        visible_turn_count=0,
        next_reveal_wall_seconds=None,
    )
    early_second_updates = runtime_transcript_visibility_state_updates(
        session_id="session-live",
        turns=[first_turn, second_turn],
        wall_clock_seconds=13.0,
        visibility_session_id=first_updates["runtime_transcript_visibility_session_id"],
        visible_turn_count=first_updates["runtime_transcript_visible_turn_count"],
        next_reveal_wall_seconds=first_updates[
            "runtime_transcript_next_reveal_wall_seconds"
        ],
    )
    second_updates = runtime_transcript_visibility_state_updates(
        session_id="session-live",
        turns=[first_turn, second_turn],
        wall_clock_seconds=14.6,
        visibility_session_id=first_updates["runtime_transcript_visibility_session_id"],
        visible_turn_count=first_updates["runtime_transcript_visible_turn_count"],
        next_reveal_wall_seconds=first_updates[
            "runtime_transcript_next_reveal_wall_seconds"
        ],
    )

    assert first_updates["runtime_transcript_next_reveal_wall_seconds"] == pytest.approx(14.6)
    assert early_second_updates["runtime_transcript_visible_turn_count"] == 1
    assert second_updates["runtime_transcript_visible_turn_count"] == 2
    assert second_updates["runtime_transcript_next_reveal_wall_seconds"] == pytest.approx(17.2)


def test_runtime_transcript_visibility_clock_continues_after_completed() -> None:
    first_updates = runtime_transcript_visibility_clock_state_updates(
        session_id="session-live",
        phase="completed",
        wall_clock_seconds=20.0,
        visibility_session_id="session-live",
        visible_turn_count=1,
        total_turn_count=3,
        clock_started_at_monotonic=None,
        clock_base_seconds=0.0,
        now_monotonic=100.0,
    )
    second_updates = runtime_transcript_visibility_clock_state_updates(
        session_id="session-live",
        phase="completed",
        wall_clock_seconds=20.0,
        visibility_session_id="session-live",
        visible_turn_count=1,
        total_turn_count=3,
        clock_started_at_monotonic=first_updates[
            "runtime_transcript_visibility_clock_started_at_monotonic"
        ],
        clock_base_seconds=first_updates[
            "runtime_transcript_visibility_clock_base_seconds"
        ],
        now_monotonic=101.25,
    )

    assert first_updates["runtime_transcript_visibility_clock_seconds"] == 20.0
    assert (
        first_updates["runtime_transcript_visibility_clock_started_at_monotonic"]
        == 100.0
    )
    assert first_updates["runtime_transcript_visibility_clock_base_seconds"] == 20.0
    assert second_updates["runtime_transcript_visibility_clock_seconds"] == 21.25


def test_runtime_transcript_visibility_clock_continues_during_auto_throttle_pause() -> None:
    auto_throttle_updates = runtime_transcript_visibility_clock_state_updates(
        session_id="session-live",
        phase="paused",
        wall_clock_seconds=20.0,
        visibility_session_id="session-live",
        visible_turn_count=1,
        total_turn_count=3,
        clock_started_at_monotonic=None,
        clock_base_seconds=0.0,
        now_monotonic=100.0,
        continue_when_paused=True,
    )
    manual_pause_updates = runtime_transcript_visibility_clock_state_updates(
        session_id="session-live",
        phase="paused",
        wall_clock_seconds=20.0,
        visibility_session_id="session-live",
        visible_turn_count=1,
        total_turn_count=3,
        clock_started_at_monotonic=None,
        clock_base_seconds=0.0,
        now_monotonic=100.0,
        continue_when_paused=False,
    )

    assert (
        auto_throttle_updates["runtime_transcript_visibility_clock_started_at_monotonic"]
        == 100.0
    )
    assert (
        manual_pause_updates["runtime_transcript_visibility_clock_started_at_monotonic"]
        is None
    )


def test_runtime_transcript_visibility_clock_stops_when_audio_playback_paused() -> None:
    updates = runtime_transcript_visibility_clock_state_updates(
        session_id="session-live",
        phase="paused",
        wall_clock_seconds=20.0,
        visibility_session_id="session-live",
        visible_turn_count=1,
        total_turn_count=3,
        clock_started_at_monotonic=None,
        clock_base_seconds=0.0,
        now_monotonic=100.0,
        continue_when_paused=True,
        playback_paused=True,
    )

    assert updates["runtime_transcript_visibility_clock_seconds"] == 20.0
    assert updates["runtime_transcript_visibility_clock_started_at_monotonic"] is None
    assert updates["runtime_transcript_visibility_clock_base_seconds"] == 20.0


def test_runtime_transcript_visibility_clock_stops_after_catchup() -> None:
    updates = runtime_transcript_visibility_clock_state_updates(
        session_id="session-live",
        phase="completed",
        wall_clock_seconds=20.0,
        visibility_session_id="session-live",
        visible_turn_count=3,
        total_turn_count=3,
        clock_started_at_monotonic=100.0,
        clock_base_seconds=20.0,
        now_monotonic=105.0,
    )

    assert updates["runtime_transcript_visibility_clock_seconds"] == 20.0
    assert updates["runtime_transcript_visibility_clock_started_at_monotonic"] is None
    assert updates["runtime_transcript_visibility_clock_base_seconds"] == 20.0


def test_runtime_transcript_visibility_clock_continues_to_final_audio_end() -> None:
    first_updates = runtime_transcript_visibility_clock_state_updates(
        session_id="session-live",
        phase="completed",
        wall_clock_seconds=20.0,
        visibility_session_id="session-live",
        visible_turn_count=3,
        total_turn_count=3,
        clock_started_at_monotonic=None,
        clock_base_seconds=0.0,
        now_monotonic=100.0,
        completion_clock_seconds=22.0,
    )
    second_updates = runtime_transcript_visibility_clock_state_updates(
        session_id="session-live",
        phase="completed",
        wall_clock_seconds=20.0,
        visibility_session_id="session-live",
        visible_turn_count=3,
        total_turn_count=3,
        clock_started_at_monotonic=first_updates[
            "runtime_transcript_visibility_clock_started_at_monotonic"
        ],
        clock_base_seconds=first_updates[
            "runtime_transcript_visibility_clock_base_seconds"
        ],
        now_monotonic=101.5,
        completion_clock_seconds=22.0,
    )
    final_updates = runtime_transcript_visibility_clock_state_updates(
        session_id="session-live",
        phase="completed",
        wall_clock_seconds=20.0,
        visibility_session_id="session-live",
        visible_turn_count=3,
        total_turn_count=3,
        clock_started_at_monotonic=first_updates[
            "runtime_transcript_visibility_clock_started_at_monotonic"
        ],
        clock_base_seconds=first_updates[
            "runtime_transcript_visibility_clock_base_seconds"
        ],
        now_monotonic=103.0,
        completion_clock_seconds=22.0,
    )
    stable_updates = runtime_transcript_visibility_clock_state_updates(
        session_id="session-live",
        phase="completed",
        wall_clock_seconds=20.0,
        visibility_session_id="session-live",
        visible_turn_count=3,
        total_turn_count=3,
        clock_started_at_monotonic=final_updates[
            "runtime_transcript_visibility_clock_started_at_monotonic"
        ],
        clock_base_seconds=final_updates[
            "runtime_transcript_visibility_clock_base_seconds"
        ],
        now_monotonic=104.0,
        completion_clock_seconds=22.0,
    )

    assert first_updates["runtime_transcript_visibility_clock_seconds"] == 20.0
    assert (
        first_updates["runtime_transcript_visibility_clock_started_at_monotonic"]
        == 100.0
    )
    assert second_updates["runtime_transcript_visibility_clock_seconds"] == 21.5
    assert final_updates["runtime_transcript_visibility_clock_seconds"] == 22.0
    assert (
        final_updates["runtime_transcript_visibility_clock_started_at_monotonic"]
        is None
    )
    assert final_updates["runtime_transcript_visibility_clock_base_seconds"] == 22.0
    assert stable_updates["runtime_transcript_visibility_clock_seconds"] == 22.0


def test_runtime_transcript_visibility_reveals_after_completed_with_display_clock(
    tmp_path: Path,
) -> None:
    first_audio = tmp_path / "turn_0001_client_a.wav"
    second_audio = tmp_path / "turn_0002_client_b.wav"
    _write_silent_wav(first_audio, duration_seconds=1.0)
    _write_silent_wav(second_audio, duration_seconds=1.0)
    first_turn = PublicTranscriptTurn(
        turn_id=1,
        speaker="妻",
        text="クロージング前の発話です。",
        created_at=None,
        audio_path=first_audio,
        cumulative_audio_seconds=0.0,
    )
    second_turn = PublicTranscriptTurn(
        turn_id=2,
        speaker="佐伯",
        text="クロージングの次発話です。",
        created_at=None,
        audio_path=second_audio,
        cumulative_audio_seconds=1.0,
    )

    first_visibility_updates = runtime_transcript_visibility_state_updates(
        session_id="session-live",
        turns=[first_turn, second_turn],
        wall_clock_seconds=20.0,
        visibility_session_id="",
        visible_turn_count=0,
        next_reveal_wall_seconds=None,
    )
    early_clock_updates = runtime_transcript_visibility_clock_state_updates(
        session_id="session-live",
        phase="completed",
        wall_clock_seconds=20.0,
        visibility_session_id=first_visibility_updates[
            "runtime_transcript_visibility_session_id"
        ],
        visible_turn_count=first_visibility_updates[
            "runtime_transcript_visible_turn_count"
        ],
        total_turn_count=2,
        clock_started_at_monotonic=None,
        clock_base_seconds=0.0,
        now_monotonic=100.5,
    )
    early_visibility_updates = runtime_transcript_visibility_state_updates(
        session_id="session-live",
        turns=[first_turn, second_turn],
        wall_clock_seconds=early_clock_updates[
            "runtime_transcript_visibility_clock_seconds"
        ],
        visibility_session_id=first_visibility_updates[
            "runtime_transcript_visibility_session_id"
        ],
        visible_turn_count=first_visibility_updates[
            "runtime_transcript_visible_turn_count"
        ],
        next_reveal_wall_seconds=first_visibility_updates[
            "runtime_transcript_next_reveal_wall_seconds"
        ],
    )
    reveal_clock_updates = runtime_transcript_visibility_clock_state_updates(
        session_id="session-live",
        phase="completed",
        wall_clock_seconds=20.0,
        visibility_session_id=first_visibility_updates[
            "runtime_transcript_visibility_session_id"
        ],
        visible_turn_count=first_visibility_updates[
            "runtime_transcript_visible_turn_count"
        ],
        total_turn_count=2,
        clock_started_at_monotonic=early_clock_updates[
            "runtime_transcript_visibility_clock_started_at_monotonic"
        ],
        clock_base_seconds=early_clock_updates[
            "runtime_transcript_visibility_clock_base_seconds"
        ],
        now_monotonic=103.2,
    )
    reveal_visibility_updates = runtime_transcript_visibility_state_updates(
        session_id="session-live",
        turns=[first_turn, second_turn],
        wall_clock_seconds=reveal_clock_updates[
            "runtime_transcript_visibility_clock_seconds"
        ],
        visibility_session_id=first_visibility_updates[
            "runtime_transcript_visibility_session_id"
        ],
        visible_turn_count=first_visibility_updates[
            "runtime_transcript_visible_turn_count"
        ],
        next_reveal_wall_seconds=first_visibility_updates[
            "runtime_transcript_next_reveal_wall_seconds"
        ],
    )

    assert first_visibility_updates["runtime_transcript_visible_turn_count"] == 1
    assert (
        first_visibility_updates["runtime_transcript_next_reveal_wall_seconds"]
        == pytest.approx(22.6)
    )
    assert early_visibility_updates["runtime_transcript_visible_turn_count"] == 1
    assert reveal_visibility_updates["runtime_transcript_visible_turn_count"] == 2


def test_public_transcript_latest_visibility_seconds_uses_previous_audio_end(
    tmp_path: Path,
) -> None:
    first_audio = tmp_path / "turn_0001_client_a.wav"
    second_audio = tmp_path / "turn_0002_client_b.wav"
    _write_silent_wav(first_audio, duration_seconds=1.0)
    _write_silent_wav(second_audio, duration_seconds=1.0)

    assert public_transcript_latest_visibility_seconds(
        [
            PublicTranscriptTurn(
                turn_id=1,
                speaker="妻",
                text="最初の発話です。",
                created_at=None,
                audio_path=first_audio,
                cumulative_audio_seconds=0.0,
            ),
            PublicTranscriptTurn(
                turn_id=2,
                speaker="夫",
                text="次の発話です。",
                created_at=None,
                audio_path=second_audio,
                cumulative_audio_seconds=3.0,
            ),
        ]
    ) == 3.0


def test_replay_start_time_options_use_start_times_instead_of_turn_numbers() -> None:
    options = replay_start_time_options(
        [
            PublicTranscriptTurn(
                turn_id=3,
                speaker="佐伯",
                text="開始時刻で選びます。",
                created_at=None,
                audio_path=None,
                cumulative_audio_seconds=2.0,
            ),
            PublicTranscriptTurn(
                turn_id=4,
                speaker="高橋",
                text="続きです。",
                created_at=None,
                audio_path=None,
                cumulative_audio_seconds=8.0,
            ),
        ]
    )

    assert [option["label"] for option in options] == ["00:02", "00:08"]
    assert [option["start_seconds"] for option in options] == [2.0, 8.0]
    assert [option["start_index"] for option in options] == [0, 1]
    assert all("turn" not in option["label"].lower() for option in options)


def test_public_transcript_rows_tolerate_legacy_turn_without_cumulative_audio() -> None:
    rows = public_transcript_rows_for_display(
        [
            SimpleNamespace(
                turn_id=1,
                speaker="佐伯",
                text="古いキャッシュでも落としません。",
                created_at="2026-06-23T08:07:06+09:00",
                audio_path=None,
            )
        ]
    )

    assert rows[0]["開始時刻"] == "00:00"
    assert rows[0]["発話"] == "古いキャッシュでも落としません。"


def test_public_transcript_columns_keep_time_narrower_than_speaker() -> None:
    assert PUBLIC_TRANSCRIPT_TABLE_COLUMNS == [
        ("開始時刻", "time", "11%"),
        ("話者", "speaker wrap", "13%"),
        ("発話", "utterance wrap", "76%"),
    ]


def test_display_public_transcript_turns_uses_two_client_display_names(
    monkeypatch,
) -> None:
    session_state = {
        "counselor_display_name": "佐伯",
        "client_a_display_name": "妻",
        "client_b_display_name": "夫",
    }
    monkeypatch.setattr(streamlit_app, "st", SimpleNamespace(session_state=session_state))

    displayed_turns = streamlit_app.display_public_transcript_turns(
        [
            PublicTranscriptTurn(
                turn_id=1,
                speaker="client_a",
                text="妻の発話です。",
                created_at=None,
                audio_path=None,
                speaker_id="client_a",
            ),
            PublicTranscriptTurn(
                turn_id=2,
                speaker="client_b",
                text="夫の発話です。",
                created_at=None,
                audio_path=None,
                speaker_id="client_b",
            ),
            PublicTranscriptTurn(
                turn_id=3,
                speaker="counselor",
                text="カウンセラーの発話です。",
                created_at=None,
                audio_path=None,
                speaker_id="counselor",
            ),
        ]
    )

    assert [turn.speaker for turn in displayed_turns] == ["妻", "夫", "佐伯"]


def test_display_public_transcript_turns_prefers_logged_speaker_display_name(
    monkeypatch,
) -> None:
    session_state = {"client_a_display_name": "妻"}
    monkeypatch.setattr(streamlit_app, "st", SimpleNamespace(session_state=session_state))

    displayed_turns = streamlit_app.display_public_transcript_turns(
        [
            PublicTranscriptTurn(
                turn_id=1,
                speaker="client_a",
                text="ログに記録された表示名を使います。",
                created_at=None,
                audio_path=None,
                speaker_id="client_a",
                speaker_display_name="母",
            )
        ]
    )

    assert displayed_turns[0].speaker == "母"
    assert displayed_turns[0].speaker_display_name == "母"


def test_public_transcript_table_can_enlarge_body_text_without_changing_headers() -> None:
    rows = public_transcript_rows_for_display(
        [
            PublicTranscriptTurn(
                turn_id=1,
                speaker="佐伯",
                text="本文だけ大きく表示します。",
                created_at="2026-06-23T08:07:06+09:00",
                audio_path=None,
            )
        ]
    )
    html = html_table(
        rows,
        columns=PUBLIC_TRANSCRIPT_TABLE_COLUMNS,
        large_body_text=True,
    )

    assert 'class="cvd-table cvd-table--large-body"' in html
    assert ".cvd-table--large-body tbody td" in html
    assert "font-size: 2rem;" in html
    assert "<th>開始時刻</th><th>話者</th><th>発話</th>" in html


def test_public_transcript_table_can_limit_visible_body_rows_with_scroll() -> None:
    rows = public_transcript_rows_for_display(
        [
            PublicTranscriptTurn(
                turn_id=turn_id,
                speaker="佐伯",
                text=f"{turn_id}行目です。",
                created_at="2026-06-23T08:07:06+09:00",
                audio_path=None,
            )
            for turn_id in range(1, 8)
        ]
    )
    html = html_table(
        rows,
        columns=PUBLIC_TRANSCRIPT_TABLE_COLUMNS,
        large_body_text=True,
        max_visible_body_rows=6,
    )

    assert 'class="cvd-table-wrap cvd-table-wrap--scroll"' in html
    assert "--cvd-visible-body-rows: 6;" in html
    assert "overflow-y: auto;" in html
    assert "flex-direction: column-reverse;" in html
    assert "overflow-anchor: auto;" in html
    assert "position: sticky;" in html
    assert "border-collapse: separate;" in html
    assert "background: var(--secondary-background-color, #111827);" in html


def test_legacy_seed_unplayed_turns_can_be_detected_for_state_migration() -> None:
    assert is_legacy_seed_unplayed_turns(_sample_unplayed_turns()) is True
    assert is_legacy_seed_unplayed_turns([]) is False
    assert is_legacy_seed_unplayed_turns([{"turn_id": 1, "text": "実生成"}]) is False


def test_format_seconds_uses_audio_elapsed_not_wall_clock() -> None:
    assert format_seconds(0) == "00:00"
    assert format_seconds(125.9) == "02:05"
    assert format_seconds(-1) == "00:00"


def test_replay_start_updates_do_not_write_replay_into_conversation_phase() -> None:
    updates = replay_start_updates()

    assert updates == {
        "session_status": SessionStatus.RUNNING.value,
        "ui_mode": "replay",
    }
    assert "conversation_phase" not in updates


def test_should_show_evaluation_only_after_session_end_when_enabled() -> None:
    assert should_show_evaluation(SessionStatus.COMPLETED, show_evaluation=True) is True
    assert should_show_evaluation(SessionStatus.STOPPED, show_evaluation=True) is True
    assert should_show_evaluation(SessionStatus.RUNNING, show_evaluation=True) is False
    assert should_show_evaluation(SessionStatus.COMPLETED, show_evaluation=False) is False


def test_can_run_evaluation_only_after_session_end_and_not_in_replay() -> None:
    assert can_run_evaluation(SessionStatus.COMPLETED, ui_mode="live") is True
    assert can_run_evaluation(SessionStatus.STOPPED, ui_mode="live") is True
    assert can_run_evaluation(SessionStatus.RUNNING, ui_mode="live") is False
    assert can_run_evaluation(SessionStatus.COMPLETED, ui_mode="replay") is False


def test_build_evaluation_internal_context_keeps_hidden_background_internal_only() -> None:
    counselor = _profile(role="counselor", hidden_background=None)
    client = _profile(role="client", hidden_background="公開しない背景")
    theme = _theme()

    context = build_evaluation_internal_context(
        selected_mode="ai_counselor_ai_client",
        counselor_profile=counselor,
        client_profile=client,
        theme=theme,
    )

    assert context["selected_mode"] == "ai_counselor_ai_client"
    assert context["counselor_profile"]["public_profile"] == "counselor public profile"
    assert context["counselor_profile"]["prompt"] == "counselor prompt"
    assert context["client_profile"]["public_profile"] == "client public profile"
    assert context["client_profile"]["prompt"] == "client prompt"
    assert context["client_profile"]["hidden_background"] == "公開しない背景"
    assert context["theme"]["body"] == "テーマ本文"


def test_run_evaluation_for_ui_uses_conversation_log_when_session_model_is_absent(tmp_path) -> None:
    client = _RecordingEvaluationClient()

    updates = run_evaluation_for_ui(
        session_model=None,
        conversation_log=[
            {
                "speaker_name": "佐伯",
                "speaker_role": "counselor",
                "text": "今日はどんなことを話したいですか。",
            }
        ],
        status=SessionStatus.STOPPED,
        evaluation_client=client,
        model="gpt-eval-test",
        reasoning_effort="low",
        log_paths=None,
        sessions_dir=tmp_path,
        internal_context={"hidden_background": "内部背景"},
    )

    assert updates["evaluation_status"] == "completed"
    assert "public evaluation" in updates["evaluation_public"]
    assert "internal evaluation" in updates["evaluation_internal"]
    assert [request.public for request in client.requests] == [True, False]
    assert client.requests[0].session.status == SessionStatus.STOPPED
    assert client.requests[0].session.turns[0].active_revision().canonical_text == (
        "今日はどんなことを話したいですか。"
    )
    assert client.requests[0].internal_context is None
    assert client.requests[1].internal_context == {"hidden_background": "内部背景"}
    assert updates["log_paths"].evaluation_public_json.exists()
    assert updates["log_paths"].evaluation_internal_json.exists()
    assert "内部背景" not in updates["log_paths"].evaluation_public_json.read_text(
        encoding="utf-8"
    )
    assert "内部背景" in updates["log_paths"].evaluation_internal_json.read_text(
        encoding="utf-8"
    )


def test_turn_selection_options_use_turn_ids_and_speaker_labels() -> None:
    options = turn_selection_options(_sample_unplayed_turns())

    assert options == [
        {"turn_id": 1, "label": "1: 佐伯 / counselor"},
        {"turn_id": 2, "label": "2: 高橋 / client"},
    ]


def test_selected_unplayed_turn_falls_back_to_first_available_turn() -> None:
    turns = _sample_unplayed_turns()

    assert selected_unplayed_turn(turns, 2)["speaker_name"] == "高橋"
    assert selected_unplayed_turn(turns, 99)["turn_id"] == 1
    assert selected_unplayed_turn([], 1) is None


def test_edit_control_disabled_states_require_running_or_paused_session_and_selection() -> (
    None
):
    turns = _sample_unplayed_turns()

    running = edit_control_disabled_states(
        SessionStatus.RUNNING,
        selected_turn_id=2,
        unplayed_turns=turns,
        ui_mode="live",
    )
    paused = edit_control_disabled_states(
        SessionStatus.PAUSED,
        selected_turn_id=2,
        unplayed_turns=turns,
        ui_mode="live",
    )
    idle = edit_control_disabled_states(
        SessionStatus.IDLE,
        selected_turn_id=2,
        unplayed_turns=turns,
        ui_mode="live",
    )
    replay = edit_control_disabled_states(
        SessionStatus.RUNNING,
        selected_turn_id=2,
        unplayed_turns=turns,
        ui_mode="replay",
    )
    missing_selection = edit_control_disabled_states(
        SessionStatus.RUNNING,
        selected_turn_id=99,
        unplayed_turns=turns,
        ui_mode="live",
    )

    assert running == {"edit": False, "regenerate": False, "regenerate_from": False}
    assert paused == {"edit": False, "regenerate": False, "regenerate_from": False}
    assert idle == {"edit": True, "regenerate": True, "regenerate_from": True}
    assert replay == {"edit": True, "regenerate": True, "regenerate_from": True}
    assert missing_selection == {
        "edit": True,
        "regenerate": True,
        "regenerate_from": True,
    }


def test_apply_edit_to_unplayed_turns_updates_selected_turn_and_invalidates_following_turns() -> (
    None
):
    turns = _sample_unplayed_turns()

    result = apply_edit_to_unplayed_turns(
        turns,
        selected_turn_id=1,
        edited_text="今日は家族のことから話してみたいです。",
    )

    edited_turn = result["unplayed_turns"][0]
    following_turn = result["unplayed_turns"][1]
    assert edited_turn["revision_id"] == 2
    assert edited_turn["text"] == "今日は家族のことから話してみたいです。"
    assert edited_turn["audio_status"] == TurnStatus.TEXT_READY.value
    assert edited_turn["playback_status"] == TurnStatus.TEXT_READY.value
    assert edited_turn["edited"] is True
    assert edited_turn["warning_level"] == WarningLevel.NONE.value
    assert edited_turn["hold_reason"] == ""
    assert following_turn["audio_status"] == TurnStatus.INVALIDATED.value
    assert following_turn["playback_status"] == TurnStatus.INVALIDATED.value
    assert result["warnings"] == [
        "Turn 1 を編集し、後続未再生ターン 1 件を無効化しました。"
    ]
    assert turns[0]["revision_id"] == 1


def test_apply_edit_to_unplayed_turns_reports_missing_selection_without_changes() -> (
    None
):
    turns = _sample_unplayed_turns()

    result = apply_edit_to_unplayed_turns(
        turns,
        selected_turn_id=99,
        edited_text="存在しないターンです。",
    )

    assert result["unplayed_turns"] == turns
    assert result["warnings"] == ["編集対象ターンが見つかりません。"]


def test_apply_regenerate_to_unplayed_turns_marks_selected_turn_ready_and_invalidates_following() -> (
    None
):
    turns = _sample_unplayed_turns()

    result = apply_regenerate_to_unplayed_turns(turns, selected_turn_id=1)

    regenerated_turn = result["unplayed_turns"][0]
    following_turn = result["unplayed_turns"][1]
    assert regenerated_turn["revision_id"] == 2
    assert regenerated_turn["text"] == "今日はどんなことを話したいですか。"
    assert regenerated_turn["audio_status"] == TurnStatus.TEXT_READY.value
    assert regenerated_turn["playback_status"] == TurnStatus.TEXT_READY.value
    assert regenerated_turn["edited"] is False
    assert regenerated_turn["warning_level"] == WarningLevel.NONE.value
    assert regenerated_turn["hold_reason"] == ""
    assert following_turn["audio_status"] == TurnStatus.INVALIDATED.value
    assert following_turn["playback_status"] == TurnStatus.INVALIDATED.value
    assert result["warnings"] == [
        "Turn 1 を再生成待ちにし、後続未再生ターン 1 件を無効化しました。"
    ]
    assert turns[0]["revision_id"] == 1


def test_apply_regenerate_to_unplayed_turns_without_following_turns_keeps_order_and_text() -> (
    None
):
    turns = _sample_unplayed_turns()

    result = apply_regenerate_to_unplayed_turns(turns, selected_turn_id=2)

    assert [turn["turn_id"] for turn in result["unplayed_turns"]] == [1, 2]
    assert result["unplayed_turns"][0] == turns[0]
    assert result["unplayed_turns"][1]["revision_id"] == 2
    assert result["unplayed_turns"][1]["text"] == turns[1]["text"]
    assert result["warnings"] == ["Turn 2 を再生成待ちにしました。"]


def test_apply_regenerate_to_unplayed_turns_reports_missing_selection_without_changes() -> (
    None
):
    turns = _sample_unplayed_turns()

    result = apply_regenerate_to_unplayed_turns(turns, selected_turn_id=99)

    assert result["unplayed_turns"] == turns
    assert result["warnings"] == ["再生成対象ターンが見つかりません。"]


def test_apply_regenerate_from_to_unplayed_turns_invalidates_selected_and_following() -> (
    None
):
    turns = _sample_unplayed_turns()

    result = apply_regenerate_from_to_unplayed_turns(turns, selected_turn_id=1)

    assert [turn["turn_id"] for turn in result["unplayed_turns"]] == [1, 2]
    assert [turn["audio_status"] for turn in result["unplayed_turns"]] == [
        TurnStatus.INVALIDATED.value,
        TurnStatus.INVALIDATED.value,
    ]
    assert [turn["playback_status"] for turn in result["unplayed_turns"]] == [
        TurnStatus.INVALIDATED.value,
        TurnStatus.INVALIDATED.value,
    ]
    assert result["warnings"] == [
        "Turn 1 以降の未再生ターン 2 件を無効化し、再生成待ちにしました。"
    ]


def test_apply_regenerate_from_to_unplayed_turns_reports_missing_selection_without_changes() -> None:
    turns = _sample_unplayed_turns()

    result = apply_regenerate_from_to_unplayed_turns(turns, selected_turn_id=99)

    assert result["unplayed_turns"] == turns
    assert result["warnings"] == ["選択ターン以降の再生成対象が見つかりません。"]


def test_unplayed_turn_rows_from_session_uses_real_session_state() -> None:
    session = SessionState(session_id="session_ui_adapter_test")
    session.add_turn(
        speaker_role=SpeakerRole.COUNSELOR,
        speaker_name="佐伯",
        profile_id="counselor_default",
        text="再生済みです。",
        status=TurnStatus.PLAYED,
    )
    unplayed = session.add_turn(
        speaker_role=SpeakerRole.CLIENT,
        speaker_name="高橋",
        profile_id="client_default",
        text="未再生です。",
        status=TurnStatus.AUDIO_READY,
    )
    unplayed.active_revision().edited = True
    unplayed.warning_level = WarningLevel.MEDIUM
    unplayed.active_revision().warning_level = WarningLevel.MEDIUM
    unplayed.active_revision().hold_reason = "clinical_ng_medium"

    rows = unplayed_turn_rows_from_session(session)

    assert rows == [
        {
            "turn_id": 2,
            "revision_id": 1,
            "speaker_name": "高橋",
            "speaker_role": "client",
            "text": "未再生です。",
            "audio_status": TurnStatus.AUDIO_READY.value,
            "playback_status": TurnStatus.AUDIO_READY.value,
            "edited": True,
            "warning_level": WarningLevel.MEDIUM.value,
            "hold_reason": "clinical_ng_medium",
        }
    ]


def test_apply_hold_action_to_unplayed_turns_can_release_or_skip_hold() -> None:
    turns = _sample_unplayed_turns()

    released = apply_hold_action_to_unplayed_turns(turns, selected_turn_id=2, hold_action="このまま再生")
    assert released["unplayed_turns"][1]["playback_status"] == TurnStatus.AUDIO_READY.value
    assert released["warnings"] == ["Turn 2 の playback_hold を解除しました。"]

    skipped = apply_hold_action_to_unplayed_turns(turns, selected_turn_id=2, hold_action="スキップ")
    assert skipped["unplayed_turns"][1]["playback_status"] == TurnStatus.SKIPPED_UNPLAYED.value
    assert skipped["warnings"] == ["Turn 2 を未再生スキップにしました。"]


def test_start_live_session_for_ui_replaces_seed_state_with_generated_turns(tmp_path) -> None:
    text_client = _RecordingTextClient()

    updates = start_live_session_for_ui(
        config=_app_config(),
        counselor_profile=_profile(role="counselor", hidden_background=None),
        client_profile=_profile(role="client", hidden_background="内部背景"),
        theme=_theme(),
        text_client=text_client,
        sessions_dir=tmp_path,
    )

    assert updates["session_status"] == SessionStatus.RUNNING.value
    assert updates["conversation_phase"] == "main"
    assert updates["selected_unplayed_turn_id"] == 1
    assert [turn["turn_id"] for turn in updates["unplayed_turns"]] == [1, 2]
    assert [turn["speaker_role"] for turn in updates["unplayed_turns"]] == [
        "counselor",
        "client",
    ]
    assert [turn["text"] for turn in updates["unplayed_turns"]] == [
        "counselorの生成発話1",
        "clientの生成発話2",
    ]
    assert updates["conversation_log"] == []
    assert updates["log_paths"].session_dir.exists()
    assert [request.speaker_profile.role for request in text_client.requests] == [
        "counselor",
        "client",
    ]


def test_build_conversation_engine_for_ui_accepts_five_ahead_turn_override() -> None:
    engine = build_conversation_engine_for_ui(
        config=_app_config(),
        text_client=_RecordingTextClient(),
        counselor_profile=_profile(role="counselor", hidden_background=None),
        client_profile=_profile(role="client", hidden_background="内部背景"),
        theme=_theme(),
        moderation_client=None,
        clinical_ng_checker=None,
        log_paths=None,
        ahead_generation_turns=5,
    )

    assert engine.config.ahead_generation_turns == 5


def test_start_live_session_for_ui_uses_five_ahead_turn_override(tmp_path) -> None:
    text_client = _RecordingTextClient()

    updates = start_live_session_for_ui(
        config=_app_config(),
        counselor_profile=_profile(role="counselor", hidden_background=None),
        client_profile=_profile(role="client", hidden_background="内部背景"),
        theme=_theme(),
        text_client=text_client,
        sessions_dir=tmp_path,
        ahead_generation_turns=5,
    )

    assert [turn["turn_id"] for turn in updates["unplayed_turns"]] == [1, 2, 3, 4, 5]
    assert [request.speaker_profile.role for request in text_client.requests] == [
        "counselor",
        "client",
        "counselor",
        "client",
        "counselor",
    ]


def test_auto_progress_live_session_does_not_mark_due_audio_played_by_timer() -> None:
    session = SessionState(status=SessionStatus.RUNNING)
    first = session.add_turn(
        speaker_role=SpeakerRole.COUNSELOR,
        speaker_name="佐伯",
        profile_id="counselor_default",
        text="最初の発話です。",
        status=TurnStatus.AUDIO_READY,
    )
    first.active_revision().audio_path = "turn1.mp3"
    second = session.add_turn(
        speaker_role=SpeakerRole.CLIENT,
        speaker_name="高橋",
        profile_id="client_default",
        text="次の発話です。",
        status=TurnStatus.AUDIO_READY,
    )
    second.active_revision().audio_path = "turn2.mp3"
    engine = build_test_engine(_RecordingTextClient())

    updates = auto_progress_live_session_for_ui(
        session,
        conversation_engine=engine,
        cumulative_audio_seconds=0.0,
        tts_client=None,
        log_paths=None,
        response_format="mp3",
        selected_turn_id=1,
        playback_turn_id=1,
        playback_started_at=100.0,
        playback_duration_seconds=2.5,
        now_monotonic=103.0,
        playback_start_monotonic=103.0,
    )

    assert session.turns[0].status == TurnStatus.AUDIO_READY
    assert updates["conversation_log"] == []
    assert updates["cumulative_audio_seconds"] == 0.0
    assert updates["auto_playback_turn_id"] == 1
    assert updates["auto_playback_started_at"] == 100.0
    assert updates["unplayed_turns"][0]["turn_id"] == 1


def test_auto_progress_live_session_never_completes_audio_by_server_timer() -> None:
    session = SessionState(status=SessionStatus.RUNNING)
    first = session.add_turn(
        speaker_role=SpeakerRole.COUNSELOR,
        speaker_name="佐伯",
        profile_id="counselor_default",
        text="最初の発話です。",
        status=TurnStatus.AUDIO_READY,
    )
    first.active_revision().audio_path = "turn1.mp3"
    second = session.add_turn(
        speaker_role=SpeakerRole.CLIENT,
        speaker_name="高橋",
        profile_id="client_default",
        text="次の発話です。",
        status=TurnStatus.AUDIO_READY,
    )
    second.active_revision().audio_path = "turn2.mp3"
    engine = build_test_engine(_RecordingTextClient())

    too_early = auto_progress_live_session_for_ui(
        session,
        conversation_engine=engine,
        cumulative_audio_seconds=0.0,
        tts_client=None,
        log_paths=None,
        response_format="mp3",
        selected_turn_id=1,
        playback_turn_id=1,
        playback_started_at=100.0,
        playback_duration_seconds=2.5,
        now_monotonic=201.0,
        playback_start_monotonic=201.0,
        allow_server_side_playback_completion=True,
        playback_rendered_at=200.0,
    )

    assert session.turns[0].status == TurnStatus.AUDIO_READY
    assert too_early["conversation_log"] == []
    assert too_early["auto_playback_turn_id"] == 1

    due_by_timer = auto_progress_live_session_for_ui(
        session,
        conversation_engine=engine,
        cumulative_audio_seconds=0.0,
        tts_client=None,
        log_paths=None,
        response_format="mp3",
        selected_turn_id=1,
        playback_turn_id=1,
        playback_started_at=100.0,
        playback_duration_seconds=2.5,
        now_monotonic=203.0,
        playback_start_monotonic=203.0,
        playback_gap_seconds=1.2,
        allow_server_side_playback_completion=True,
        playback_rendered_at=200.0,
    )

    assert session.turns[0].status == TurnStatus.AUDIO_READY
    assert due_by_timer["conversation_log"] == []
    assert due_by_timer["cumulative_audio_seconds"] == 0.0
    assert due_by_timer["auto_playback_turn_id"] == 1
    assert due_by_timer["auto_playback_started_at"] == 100.0


def test_select_voice_preset_for_profile_defaults_to_profile_voice_preset() -> None:
    client_preset = _voice_preset("client_anxious_soft", "alloy", "client instructions")
    counselor_preset = _voice_preset(
        "counselor_calm_neutral",
        "coral",
        "counselor instructions",
    )
    counselor = _profile(role="counselor", hidden_background=None).model_copy(
        update={"voice_preset": "counselor_calm_neutral"}
    )

    selected = select_voice_preset_for_profile(
        [client_preset, counselor_preset],
        selected_preset_id=None,
        profile=counselor,
    )
    overridden = profile_with_voice_preset(counselor, selected)

    assert selected.preset_id == "counselor_calm_neutral"
    assert overridden.voice_preset == "counselor_calm_neutral"
    assert overridden.tts_voice == "coral"
    assert overridden.tts_instructions == "counselor instructions"


def test_profile_with_session_overrides_does_not_mutate_preset_profile(
    monkeypatch,
) -> None:
    client = _profile(role="client", hidden_background="プリセット背景")
    session_state = {
        "client_display_name": "妻",
        "client_public_profile_text": "セッション用公開プロフィール",
        "client_prompt_text": "セッション用プロンプト本文",
        "client_private_profile_text": "セッション用非公開背景",
    }
    monkeypatch.setattr(streamlit_app, "st", SimpleNamespace(session_state=session_state))

    overridden = profile_with_session_overrides(
        client,
        display_name_key="client_display_name",
        public_profile_text_key="client_public_profile_text",
        prompt_text_key="client_prompt_text",
        hidden_background_key="client_private_profile_text",
    )

    assert overridden.display_name == "妻"
    assert overridden.public_profile == "セッション用公開プロフィール"
    assert overridden.prompt == "セッション用プロンプト本文"
    assert overridden.hidden_background == "セッション用非公開背景"
    assert client.display_name == "client"
    assert client.public_profile == "client public profile"
    assert client.prompt == "client prompt"
    assert client.hidden_background == "プリセット背景"


def test_skip_current_audio_for_ui_refills_generation_queue() -> None:
    session = SessionState(status=SessionStatus.RUNNING)
    first = session.add_turn(
        speaker_role=SpeakerRole.COUNSELOR,
        speaker_name="佐伯",
        profile_id="counselor_default",
        text="最初の発話です。",
        status=TurnStatus.AUDIO_READY,
    )
    first.active_revision().audio_path = "turn1.mp3"
    second = session.add_turn(
        speaker_role=SpeakerRole.CLIENT,
        speaker_name="高橋",
        profile_id="client_default",
        text="次の発話です。",
        status=TurnStatus.AUDIO_READY,
    )
    second.active_revision().audio_path = "turn2.mp3"
    engine = build_test_engine(_RecordingTextClient())

    updates = skip_current_audio_for_ui(
        session,
        conversation_engine=engine,
        cumulative_audio_seconds=0.0,
        tts_client=None,
        log_paths=None,
        response_format="mp3",
        selected_turn_id=1,
    )

    assert session.turns[0].status == TurnStatus.SKIPPED_UNPLAYED
    assert [turn["turn_id"] for turn in updates["unplayed_turns"]] == [2, 3]
    assert updates["unplayed_turns"][1]["speaker_role"] == "counselor"
    assert updates["selected_unplayed_turn_id"] == 2


def test_mark_current_audio_played_for_ui_appends_log_and_refills_queue() -> None:
    session = SessionState(status=SessionStatus.RUNNING)
    first = session.add_turn(
        speaker_role=SpeakerRole.COUNSELOR,
        speaker_name="佐伯",
        profile_id="counselor_default",
        text="最初の発話です。",
        status=TurnStatus.AUDIO_READY,
    )
    first.active_revision().audio_path = "turn1.mp3"
    second = session.add_turn(
        speaker_role=SpeakerRole.CLIENT,
        speaker_name="高橋",
        profile_id="client_default",
        text="次の発話です。",
        status=TurnStatus.AUDIO_READY,
    )
    second.active_revision().audio_path = "turn2.mp3"
    engine = build_test_engine(_RecordingTextClient())

    updates = mark_current_audio_played_for_ui(
        session,
        conversation_engine=engine,
        cumulative_audio_seconds=0.0,
        tts_client=None,
        log_paths=None,
        response_format="mp3",
        selected_turn_id=1,
        duration_seconds=2.5,
    )

    assert session.turns[0].status == TurnStatus.PLAYED
    assert updates["cumulative_audio_seconds"] == 2.5
    assert updates["conversation_log"] == [
        {
            "turn_id": 1,
            "speaker_name": "佐伯",
            "speaker_role": "counselor",
            "text": "最初の発話です。",
            "spoken_at": session.turns[0].active_revision().spoken_at.isoformat(),
            "cumulative_audio_seconds": 2.5,
        }
    ]
    assert [turn["turn_id"] for turn in updates["unplayed_turns"]] == [2, 3]
    assert updates["unplayed_turns"][1]["speaker_role"] == "counselor"
    assert updates["selected_unplayed_turn_id"] == 2


def test_build_audio_player_queue_for_ui_reads_active_audio_file(tmp_path) -> None:
    session = SessionState(status=SessionStatus.RUNNING)
    first = session.add_turn(
        speaker_role=SpeakerRole.COUNSELOR,
        speaker_name="佐伯",
        profile_id="counselor_default",
        text="最初の発話です。",
        status=TurnStatus.AUDIO_READY,
    )
    audio_path = tmp_path / "turn1.mp3"
    audio_path.write_bytes(b"fake mp3")
    first.active_revision().audio_path = str(audio_path)
    first.active_revision().audio_format = "mp3"

    queue_version, queue = build_audio_player_queue_for_ui(session)

    assert queue_version
    assert queue[0]["turn_id"] == 1
    assert queue[0]["revision_id"] == 1
    assert queue[0]["audio_src"].startswith("data:audio/mpeg;base64,")


def test_apply_browser_audio_ended_event_marks_turn_played_once(tmp_path) -> None:
    session = SessionState(status=SessionStatus.RUNNING)
    first = session.add_turn(
        speaker_role=SpeakerRole.COUNSELOR,
        speaker_name="佐伯",
        profile_id="counselor_default",
        text="最初の発話です。",
        status=TurnStatus.AUDIO_READY,
    )
    audio_path = tmp_path / "turn1.mp3"
    audio_path.write_bytes(b"fake mp3")
    first.active_revision().audio_path = str(audio_path)
    first.active_revision().audio_format = "mp3"
    engine = build_test_engine(_RecordingTextClient())
    queue_version, _ = build_audio_player_queue_for_ui(session)
    event = {
        "event_id": "ended-1",
        "event_type": "playback_ended",
        "turn_id": 1,
        "revision_id": 1,
        "queue_version": queue_version,
        "current_seconds": 2.5,
        "played_seconds": 2.5,
        "duration_seconds": 2.5,
    }

    updates = apply_browser_audio_event_for_ui(
        session,
        event,
        queue_version=queue_version,
        processed_event_ids=[],
        conversation_engine=engine,
        cumulative_audio_seconds=0.0,
        tts_client=None,
        log_paths=None,
        response_format="mp3",
        selected_turn_id=1,
    )
    duplicate = apply_browser_audio_event_for_ui(
        session,
        event,
        queue_version=queue_version,
        processed_event_ids=updates["audio_player_processed_event_ids"],
        conversation_engine=engine,
        cumulative_audio_seconds=updates["cumulative_audio_seconds"],
        tts_client=None,
        log_paths=None,
        response_format="mp3",
        selected_turn_id=1,
    )

    assert session.turns[0].status == TurnStatus.PLAYED
    assert updates["cumulative_audio_seconds"] == 2.5
    assert updates["conversation_log"][0]["text"] == "最初の発話です。"
    assert updates["audio_player_processed_event_ids"] == ["ended-1"]
    assert updates["audio_player_continuous_mode"] is True
    assert "_audio_player_event_requires_rerun" not in updates
    assert duplicate == {}


def test_apply_browser_audio_event_accepts_queue_version_drift_for_current_turn(
    tmp_path,
) -> None:
    session = SessionState(status=SessionStatus.RUNNING)
    first = session.add_turn(
        speaker_role=SpeakerRole.COUNSELOR,
        speaker_name="佐伯",
        profile_id="counselor_default",
        text="最初の発話です。",
        status=TurnStatus.AUDIO_READY,
    )
    audio_path = tmp_path / "turn1.mp3"
    audio_path.write_bytes(b"fake mp3")
    first.active_revision().audio_path = str(audio_path)
    first.active_revision().audio_format = "mp3"

    updates = apply_browser_audio_event_for_ui(
        session,
        {
            "event_id": "drifted-ended",
            "event_type": "playback_ended",
            "turn_id": 1,
            "revision_id": 1,
            "queue_version": "old-version",
            "current_seconds": 2.5,
            "played_seconds": 2.5,
            "duration_seconds": 2.5,
        },
        queue_version="current-version",
        processed_event_ids=[],
        conversation_engine=build_test_engine(_RecordingTextClient()),
        cumulative_audio_seconds=0.0,
        tts_client=None,
        log_paths=None,
        response_format="mp3",
        selected_turn_id=1,
    )

    assert session.turns[0].status == TurnStatus.PLAYED
    assert updates["audio_player_processed_event_ids"] == ["drifted-ended"]
    assert updates["audio_player_continuous_mode"] is True
    assert updates["cumulative_audio_seconds"] == 2.5
    assert updates["conversation_log"][0]["text"] == "最初の発話です。"
    assert "_audio_player_event_requires_rerun" not in updates


def test_apply_browser_audio_event_schedules_next_turn_after_terminal_event(
    tmp_path,
) -> None:
    session = SessionState(status=SessionStatus.RUNNING)
    first = session.add_turn(
        speaker_role=SpeakerRole.COUNSELOR,
        speaker_name="佐伯",
        profile_id="counselor_default",
        text="最初の発話です。",
        status=TurnStatus.AUDIO_READY,
    )
    first_audio_path = tmp_path / "turn1.mp3"
    first_audio_path.write_bytes(b"fake mp3")
    first.active_revision().audio_path = str(first_audio_path)
    first.active_revision().audio_format = "mp3"
    second = session.add_turn(
        speaker_role=SpeakerRole.CLIENT,
        speaker_name="高橋",
        profile_id="client_default",
        text="次の発話です。",
        status=TurnStatus.AUDIO_READY,
    )
    second_audio_path = tmp_path / "turn2.mp3"
    second_audio_path.write_bytes(b"fake mp3")
    second.active_revision().audio_path = str(second_audio_path)
    second.active_revision().audio_format = "mp3"

    updates = apply_browser_audio_event_for_ui(
        session,
        {
            "event_id": "ended-schedules-next",
            "event_type": "playback_ended",
            "turn_id": 1,
            "revision_id": 1,
            "queue_version": "queue-version",
            "current_seconds": 2.5,
            "played_seconds": 2.5,
            "duration_seconds": 2.5,
        },
        queue_version="queue-version",
        processed_event_ids=[],
        conversation_engine=build_test_engine(_RecordingTextClient()),
        cumulative_audio_seconds=0.0,
        tts_client=None,
        log_paths=None,
        response_format="mp3",
        selected_turn_id=1,
        playback_gap_seconds=1.2,
    )

    assert session.turns[0].status == TurnStatus.PLAYED
    assert updates["auto_playback_turn_id"] == 2
    assert updates["auto_playback_started_at"] is not None
    assert updates["auto_playback_duration_seconds"] == 1.0
    assert "_audio_player_event_requires_rerun" not in updates


def test_apply_browser_audio_event_ignores_stale_revision(tmp_path) -> None:
    session = SessionState(status=SessionStatus.RUNNING)
    first = session.add_turn(
        speaker_role=SpeakerRole.COUNSELOR,
        speaker_name="佐伯",
        profile_id="counselor_default",
        text="最初の発話です。",
        status=TurnStatus.AUDIO_READY,
    )
    audio_path = tmp_path / "turn1.mp3"
    audio_path.write_bytes(b"fake mp3")
    first.active_revision().audio_path = str(audio_path)
    first.active_revision().audio_format = "mp3"

    updates = apply_browser_audio_event_for_ui(
        session,
        {
            "event_id": "stale-revision-ended",
            "event_type": "playback_ended",
            "turn_id": 1,
            "revision_id": 99,
            "queue_version": "old-version",
            "current_seconds": 2.5,
            "played_seconds": 2.5,
            "duration_seconds": 2.5,
        },
        queue_version="current-version",
        processed_event_ids=[],
        conversation_engine=build_test_engine(_RecordingTextClient()),
        cumulative_audio_seconds=0.0,
        tts_client=None,
        log_paths=None,
        response_format="mp3",
        selected_turn_id=1,
    )

    assert session.turns[0].status == TurnStatus.AUDIO_READY
    assert updates["audio_player_processed_event_ids"] == ["stale-revision-ended"]
    assert "audio_player_continuous_mode" not in updates
    assert "cumulative_audio_seconds" not in updates
    assert updates["warnings"] == ["古い音声イベントを無視しました: turn_id=1"]


def test_auto_progress_live_session_starts_playback_timer_without_manual_click() -> None:
    session = SessionState(status=SessionStatus.RUNNING)
    first = session.add_turn(
        speaker_role=SpeakerRole.COUNSELOR,
        speaker_name="佐伯",
        profile_id="counselor_default",
        text="最初の発話です。",
        status=TurnStatus.AUDIO_READY,
    )
    first.active_revision().audio_path = "turn1.mp3"
    second = session.add_turn(
        speaker_role=SpeakerRole.CLIENT,
        speaker_name="高橋",
        profile_id="client_default",
        text="次の発話です。",
        status=TurnStatus.AUDIO_READY,
    )
    second.active_revision().audio_path = "turn2.mp3"
    engine = build_test_engine(_RecordingTextClient())

    updates = auto_progress_live_session_for_ui(
        session,
        conversation_engine=engine,
        cumulative_audio_seconds=0.0,
        tts_client=None,
        log_paths=None,
        response_format="mp3",
        selected_turn_id=1,
        playback_turn_id=None,
        playback_started_at=None,
        playback_duration_seconds=None,
        now_monotonic=100.0,
        playback_start_monotonic=100.0,
        existing_warnings=["既存の警告"],
    )

    assert session.turns[0].status == TurnStatus.AUDIO_READY
    assert updates["conversation_log"] == []
    assert updates["auto_playback_turn_id"] == 1
    assert updates["auto_playback_started_at"] == 100.0
    assert updates["auto_playback_duration_seconds"] == 1.0
    assert updates["warnings"] == ["既存の警告"]


def test_auto_progress_live_session_does_not_schedule_next_when_browser_manages_playback() -> None:
    session = SessionState(status=SessionStatus.RUNNING)
    first = session.add_turn(
        speaker_role=SpeakerRole.COUNSELOR,
        speaker_name="佐伯",
        profile_id="counselor_default",
        text="最初の発話です。",
        status=TurnStatus.AUDIO_READY,
    )
    first.active_revision().audio_path = "turn1.mp3"
    engine = build_test_engine(_RecordingTextClient())

    updates = auto_progress_live_session_for_ui(
        session,
        conversation_engine=engine,
        cumulative_audio_seconds=0.0,
        tts_client=None,
        log_paths=None,
        response_format="mp3",
        selected_turn_id=1,
        playback_turn_id=None,
        playback_started_at=None,
        playback_duration_seconds=None,
        now_monotonic=100.0,
        playback_start_monotonic=100.0,
        browser_audio_player_manages_playback=True,
    )

    assert session.turns[0].status == TurnStatus.AUDIO_READY
    assert updates["auto_playback_turn_id"] is None
    assert updates["auto_playback_started_at"] is None
    assert updates["auto_playback_duration_seconds"] is None


def test_auto_progress_live_session_schedules_after_waiting_audio_becomes_ready(
    tmp_path,
) -> None:
    session = SessionState(status=SessionStatus.RUNNING, session_id="session_wait_tts")
    first = session.add_turn(
        speaker_role=SpeakerRole.COUNSELOR,
        speaker_name="佐伯",
        profile_id="counselor_default",
        text="音声パスが欠けた発話です。",
        status=TurnStatus.AUDIO_READY,
    )
    first.active_revision().audio_path = None
    text_client = _RecordingTextClient()
    tts_client = _RecordingTtsClient()
    engine = build_test_engine(text_client)
    paths = create_session_log_dirs(tmp_path, session.session_id)

    updates = auto_progress_live_session_for_ui(
        session,
        conversation_engine=engine,
        cumulative_audio_seconds=0.0,
        tts_client=tts_client,
        log_paths=paths,
        response_format="mp3",
        selected_turn_id=1,
        playback_turn_id=None,
        playback_started_at=None,
        playback_duration_seconds=None,
        now_monotonic=100.0,
        playback_start_monotonic=100.0,
        browser_audio_player_manages_playback=True,
    )

    assert session.turns[0].status == TurnStatus.AUDIO_READY
    assert Path(session.turns[0].active_revision().audio_path).exists()
    assert tts_client.turn_ids[0] == 1
    assert updates["auto_playback_turn_id"] == 1
    assert updates["auto_playback_started_at"] == 100.0
    assert updates["auto_playback_duration_seconds"] == 1.0
    assert updates["_auto_progress_changed"] is True


def test_auto_progress_live_session_refills_future_audio_buffer_while_current_audio_is_playing(
    tmp_path,
) -> None:
    session = SessionState(status=SessionStatus.RUNNING, session_id="session_prebuffer")
    first = session.add_turn(
        speaker_role=SpeakerRole.COUNSELOR,
        speaker_name="佐伯",
        profile_id="counselor_default",
        text="最初の発話です。",
        status=TurnStatus.AUDIO_READY,
    )
    first.active_revision().audio_path = "turn1.mp3"
    second = session.add_turn(
        speaker_role=SpeakerRole.CLIENT,
        speaker_name="高橋",
        profile_id="client_default",
        text="次の発話です。",
        status=TurnStatus.AUDIO_READY,
    )
    second.active_revision().audio_path = "turn2.mp3"
    text_client = _RecordingTextClient()
    tts_client = _RecordingTtsClient()
    engine = build_test_engine(text_client)
    paths = create_session_log_dirs(tmp_path, session.session_id)

    updates = auto_progress_live_session_for_ui(
        session,
        conversation_engine=engine,
        cumulative_audio_seconds=0.0,
        tts_client=tts_client,
        log_paths=paths,
        response_format="mp3",
        selected_turn_id=1,
        playback_turn_id=1,
        playback_started_at=100.0,
        playback_duration_seconds=2.5,
        now_monotonic=101.0,
        playback_start_monotonic=101.0,
        existing_warnings=["既存の警告"],
    )

    assert session.turns[0].status == TurnStatus.AUDIO_READY
    assert [turn.speaker_role for turn in session.turns] == [
        SpeakerRole.COUNSELOR,
        SpeakerRole.CLIENT,
        SpeakerRole.COUNSELOR,
    ]
    assert session.turns[2].status == TurnStatus.AUDIO_READY
    assert Path(session.turns[2].active_revision().audio_path).exists()
    assert [request.speaker_profile.role for request in text_client.requests] == [
        "counselor"
    ]
    assert tts_client.turn_ids == [3]
    assert updates["auto_playback_turn_id"] == 1
    assert [turn["turn_id"] for turn in updates["unplayed_turns"]] == [1, 2, 3]
    assert updates["warnings"] == ["既存の警告"]
    assert updates["_auto_progress_changed"] is False


def test_auto_progress_live_session_does_not_finish_due_audio_without_browser_event() -> None:
    session = SessionState(status=SessionStatus.RUNNING)
    first = session.add_turn(
        speaker_role=SpeakerRole.COUNSELOR,
        speaker_name="佐伯",
        profile_id="counselor_default",
        text="最初の発話です。",
        status=TurnStatus.AUDIO_READY,
    )
    first.active_revision().audio_path = "turn1.mp3"
    second = session.add_turn(
        speaker_role=SpeakerRole.CLIENT,
        speaker_name="高橋",
        profile_id="client_default",
        text="次の発話です。",
        status=TurnStatus.AUDIO_READY,
    )
    second.active_revision().audio_path = "turn2.mp3"
    text_client = _RecordingTextClient()
    engine = build_test_engine(text_client)

    updates = auto_progress_live_session_for_ui(
        session,
        conversation_engine=engine,
        cumulative_audio_seconds=0.0,
        tts_client=None,
        log_paths=None,
        response_format="mp3",
        selected_turn_id=1,
        playback_turn_id=1,
        playback_started_at=100.0,
        playback_duration_seconds=2.5,
        now_monotonic=103.0,
        playback_start_monotonic=103.0,
        playback_gap_seconds=1.2,
        allow_server_side_playback_completion=True,
    )

    assert session.turns[0].status == TurnStatus.AUDIO_READY
    assert updates["cumulative_audio_seconds"] == 0.0
    assert [turn["turn_id"] for turn in updates["unplayed_turns"]] == [1, 2, 3]
    assert updates["auto_playback_turn_id"] == 1
    assert updates["auto_playback_started_at"] == 100.0
    assert updates["auto_playback_duration_seconds"] == 2.5
    assert updates["conversation_log"] == []
    assert [request.speaker_profile.role for request in text_client.requests] == [
        "counselor"
    ]


def test_auto_progress_live_session_refills_future_buffer_without_completion() -> None:
    session = SessionState(status=SessionStatus.RUNNING)
    first = session.add_turn(
        speaker_role=SpeakerRole.COUNSELOR,
        speaker_name="佐伯",
        profile_id="counselor_default",
        text="最初の発話です。",
        status=TurnStatus.AUDIO_READY,
    )
    first.active_revision().audio_path = "turn1.mp3"
    text_client = _RecordingTextClient()
    engine = build_test_engine(text_client)

    updates = auto_progress_live_session_for_ui(
        session,
        conversation_engine=engine,
        cumulative_audio_seconds=0.0,
        tts_client=None,
        log_paths=None,
        response_format="mp3",
        selected_turn_id=1,
        playback_turn_id=1,
        playback_started_at=100.0,
        playback_duration_seconds=2.5,
        now_monotonic=103.0,
        playback_start_monotonic=120.0,
        allow_server_side_playback_completion=True,
    )

    assert session.turns[0].status == TurnStatus.AUDIO_READY
    assert [turn["turn_id"] for turn in updates["unplayed_turns"]] == [1, 2, 3]
    assert updates["auto_playback_turn_id"] == 1
    assert [request.speaker_profile.role for request in text_client.requests] == [
        "client",
        "counselor",
    ]


def test_auto_progress_live_session_repairs_duplicate_unplayed_speaker_before_playback() -> None:
    session = SessionState(status=SessionStatus.RUNNING)
    session.add_turn(
        speaker_role=SpeakerRole.COUNSELOR,
        speaker_name="佐伯",
        profile_id="counselor_default",
        text="再生済みのカウンセラー発話です。",
        status=TurnStatus.PLAYED,
    )
    session.add_turn(
        speaker_role=SpeakerRole.CLIENT,
        speaker_name="高橋",
        profile_id="client_default",
        text="再生済みのクライアント発話です。",
        status=TurnStatus.PLAYED,
    )
    duplicate = session.add_turn(
        speaker_role=SpeakerRole.CLIENT,
        speaker_name="高橋",
        profile_id="client_default",
        text="連続してしまったクライアント発話です。",
        status=TurnStatus.AUDIO_READY,
    )
    duplicate.active_revision().audio_path = "turn3.mp3"
    text_client = _RecordingTextClient()
    engine = build_test_engine(text_client)

    updates = auto_progress_live_session_for_ui(
        session,
        conversation_engine=engine,
        cumulative_audio_seconds=18.0,
        tts_client=None,
        log_paths=None,
        response_format="mp3",
        selected_turn_id=3,
        playback_turn_id=None,
        playback_started_at=None,
        playback_duration_seconds=None,
        now_monotonic=100.0,
        playback_start_monotonic=100.0,
    )

    assert duplicate.status == TurnStatus.INVALIDATED
    assert updates["auto_playback_turn_id"] is None
    assert [turn["speaker_role"] for turn in updates["unplayed_turns"]] == [
        "counselor",
        "client",
    ]
    assert [request.speaker_profile.role for request in text_client.requests] == [
        "counselor",
        "client",
    ]
    assert updates["warnings"][0] == "話者順が連続した未再生ターン 3 を無効化しました。"


class _RecordingEvaluationClient:
    def __init__(self) -> None:
        self.requests: list[EvaluationRequest] = []

    def evaluate(self, request: EvaluationRequest) -> EvaluationResult:
        self.requests.append(request)
        text = "public evaluation" if request.public else "internal evaluation"
        return EvaluationResult(text=text, raw_response={})


class _RecordingTextClient:
    def __init__(self) -> None:
        self.requests = []

    def generate_turn(self, request):
        self.requests.append(request)
        return type(
            "TextGenerationResult",
            (),
            {
                "text": f"{request.speaker_profile.display_name}の生成発話{len(self.requests)}",
                "raw_response": {"id": f"resp_{len(self.requests)}"},
            },
        )()


class _RecordingTtsClient:
    def __init__(self) -> None:
        self.turn_ids: list[int] = []

    def synthesize_turn(self, request) -> None:
        self.turn_ids.append(request.turn.turn_id)
        audio_path = (
            request.log_paths.audio_all_revisions_dir
            / f"turn_{request.turn.turn_id:04d}_rev{request.turn.active_revision_id:02d}_{request.turn.speaker_role.value}.{request.response_format}"
        )
        audio_path.write_bytes(b"fake mp3")
        revision = request.turn.active_revision()
        revision.audio_path = str(audio_path)
        revision.audio_format = request.response_format


class _RecordingRuntimeControlClient:
    def __init__(self, status: dict | None = None) -> None:
        self.status_payload = status or {
            "session_id": "session-runtime",
            "phase": "running",
        }
        self.actions: list[str] = []
        self.start_options: list[dict | None] = []
        self.pause_reasons: list[str | None] = []
        self.event_session_ids: list[str] = []
        self.response_instruction_session_ids: list[str] = []

    def start(self, options: dict | None = None) -> dict:
        self.actions.append("start")
        self.start_options.append(options)
        return self.status_payload

    def pause(self, reason: str | None = None) -> dict:
        self.actions.append("pause")
        self.pause_reasons.append(reason)
        return self.status_payload

    def resume(self) -> dict:
        self.actions.append("resume")
        return self.status_payload

    def stop(self) -> dict:
        self.actions.append("stop")
        return self.status_payload

    def status(self) -> dict:
        self.actions.append("status")
        return self.status_payload

    def events(self, session_id: str) -> list[dict]:
        self.event_session_ids.append(session_id)
        return [{"event_type": "tts_stream_done"}]

    def response_instructions(self, session_id: str) -> list[dict]:
        self.response_instruction_session_ids.append(session_id)
        return [
            {
                "turn_id": 1,
                "speaker_id": "counselor",
                "resolved_instructions": "送信済みinstructions",
            }
        ]


def _sample_unplayed_turns() -> list[dict]:
    return [
        {
            "turn_id": 1,
            "revision_id": 1,
            "speaker_name": "佐伯",
            "speaker_role": "counselor",
            "text": "今日はどんなことを話したいですか。",
            "audio_status": TurnStatus.AUDIO_READY.value,
            "playback_status": TurnStatus.QUEUED_FOR_PLAYBACK.value,
            "edited": False,
            "warning_level": WarningLevel.NONE.value,
            "hold_reason": "",
        },
        {
            "turn_id": 2,
            "revision_id": 1,
            "speaker_name": "高橋",
            "speaker_role": "client",
            "text": "家族との距離感に悩んでいます。",
            "audio_status": TurnStatus.AUDIO_READY.value,
            "playback_status": TurnStatus.PLAYBACK_HOLD.value,
            "edited": False,
            "warning_level": WarningLevel.HIGH.value,
            "hold_reason": "moderation_flagged",
        },
    ]


def _write_silent_wav(path: Path, *, duration_seconds: float) -> None:
    sample_rate = 10
    frame_count = int(sample_rate * duration_seconds)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(b"\0\0" * frame_count)


def _app_config():
    return SimpleNamespace(
        app=SimpleNamespace(
            ahead_generation_turns=2,
            closing_start_audio_seconds=900,
            farewell_after_turns=2,
        ),
        session=SimpleNamespace(closing_keep_audio_ready_turns=1),
        openai=SimpleNamespace(
            default_text_reasoning_effort="low",
            default_audio_format="mp3",
            moderation_model="omni-moderation-test",
            default_text_model="gpt-clinical-test",
        ),
    )


def build_test_engine(text_client):
    return build_conversation_engine_for_ui(
        config=_app_config(),
        text_client=text_client,
        counselor_profile=_profile(role="counselor", hidden_background=None),
        client_profile=_profile(role="client", hidden_background="内部背景"),
        theme=_theme(),
        moderation_client=None,
        clinical_ng_checker=None,
        log_paths=None,
    )


def _profile(*, role: str, hidden_background: str | None) -> Profile:
    return Profile(
        profile_id=f"{role}_default",
        role=role,
        display_name=role,
        text_model="gpt-test",
        temperature=0.4,
        voice_preset="voice_default",
        tts_model="tts-test",
        tts_voice="alloy",
        tts_instructions="",
        public_profile=f"{role} public profile",
        prompt=f"{role} prompt",
        hidden_background=hidden_background,
    )


def _theme() -> Theme:
    return Theme(
        theme_id="family_conflict",
        display_name="家族葛藤",
        severity="medium",
        default_client_profile="client_default",
        body="テーマ本文",
    )


def _voice_preset(preset_id: str, voice: str, instructions: str) -> VoicePreset:
    return VoicePreset(
        preset_id=preset_id,
        display_name=preset_id,
        tts_model="tts-test",
        fallback_tts_model="tts-fallback-test",
        voice=voice,
        fallback_voice="fallback-voice",
        instructions=instructions,
        speed_hint="standard",
        response_format="mp3",
    )


def test_human_client_evaluation_does_not_use_simulated_client_or_call_models():
    mode = "ai_counselor_human_client"
    assert not should_show_evaluation(
        SessionStatus.COMPLETED, show_evaluation=True, selected_mode=mode
    )
    assert not can_run_evaluation(
        SessionStatus.COMPLETED, ui_mode="live", selected_mode=mode
    )
    context = build_evaluation_internal_context(
        selected_mode=mode,
        counselor_profile=_profile(role="counselor", hidden_background=None),
        client_profile=_profile(role="client", hidden_background="架空の秘密背景"),
        theme=None,
    )
    assert context == {"selected_mode": mode, "evaluation_supported": False}
    client = _RecordingEvaluationClient()
    with pytest.raises(ValueError, match="未対応"):
        run_evaluation_for_ui(
            session_model=None,
            conversation_log=[],
            status=SessionStatus.COMPLETED,
            evaluation_client=client,
            model="unused",
            reasoning_effort=None,
            log_paths=None,
            internal_context=context,
        )
    assert client.requests == []


def test_three_mode_settings_render_without_exposing_ai_client_fields():
    from streamlit.testing.v1 import AppTest

    view = AppTest.from_file("app/streamlit_app.py", default_timeout=20).run()
    assert not view.exception
    assert not view.error
    for mode in [
        "ai_counselor_human_client",
        "human_counselor_ai_client",
        "ai_counselor_ai_client",
        "ai_counselor_human_client",
    ]:
        view.radio(key="runtime_mode_selector").set_value(streamlit_app.MODE_LABELS[mode]).run()
        assert not view.exception
        assert not view.error
        if mode == "ai_counselor_human_client":
            inputs = [item.label for item in view.text_input]
            areas = [item.label for item in view.text_area]
            selectors = [item.label for item in view.selectbox]
            assert "人間クライアント表示名" in inputs
            assert "AIに事前共有する情報（任意）" in areas
            assert "初回クライアント発話" not in inputs
            assert "クライアント秘密プロフィール" not in areas
            assert "クライアントプリセット" not in selectors
            assert "クライアント音声" not in selectors
            assert not any(item.label == "参加クライアント数" for item in view.radio)
