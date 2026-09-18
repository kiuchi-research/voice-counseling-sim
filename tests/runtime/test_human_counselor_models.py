from __future__ import annotations

import pytest

from counseling_voice_demo.runtime.models import (
    ActorKind,
    HumanAudioInput,
    HumanTurnInput,
    InteractionMode,
    ParticipantConfig,
    ParticipantMode,
    RuntimeConfig,
)


def test_participant_config_defaults_to_ai_actor_kind() -> None:
    participant = ParticipantConfig(
        speaker_id="client",
        role="client",
        display_name="クライアント",
    )

    assert participant.actor_kind == ActorKind.AI


def test_participant_config_accepts_human_counselor_actor_kind() -> None:
    participant = ParticipantConfig(
        speaker_id="counselor",
        role="counselor",
        display_name="カウンセラー",
        actor_kind=ActorKind.HUMAN,
    )

    assert participant.actor_kind == ActorKind.HUMAN


def test_human_audio_input_validates_push_to_talk_audio() -> None:
    request = HumanAudioInput(
        session_id="session-human",
        audio_bytes=b"\0\1",
        sample_rate=24000,
        channels=1,
        recipient_ids=("client",),
        client_message_id="message-1",
    )

    assert request.recording_mode == "push_to_talk"
    assert request.recipient_ids == ("client",)


def test_human_audio_input_rejects_empty_audio() -> None:
    with pytest.raises(ValueError, match="audio_bytes"):
        HumanAudioInput(
            session_id="session-human",
            audio_bytes=b"",
            sample_rate=24000,
            channels=1,
            recipient_ids=("client",),
        )


def test_human_turn_input_defaults_to_mic_stt_push_to_talk() -> None:
    request = HumanTurnInput(
        session_id="session-human",
        text="今日はどんなことを相談したいですか。",
        recipient_ids=("client",),
        source_audio_ref="human_audio/session-human/turn_0001.wav",
    )

    assert request.input_mode == "human_mic_stt"
    assert request.recording_mode == "push_to_talk"


def test_human_turn_input_rejects_empty_text() -> None:
    with pytest.raises(ValueError, match="text"):
        HumanTurnInput(
            session_id="session-human",
            text=" ",
            recipient_ids=("client",),
        )


def test_human_turn_input_rejects_invalid_input_mode() -> None:
    with pytest.raises(ValueError, match="input_mode"):
        HumanTurnInput(
            session_id="session-human",
            text="今日はどんなことを相談したいですか。",
            recipient_ids=("client",),
            input_mode="browser_text",
        )


def test_human_turn_input_rejects_invalid_recording_mode() -> None:
    with pytest.raises(ValueError, match="recording_mode"):
        HumanTurnInput(
            session_id="session-human",
            text="今日はどんなことを相談したいですか。",
            recipient_ids=("client",),
            recording_mode="always_on",
        )


def test_human_turn_input_rejects_empty_source_audio_ref() -> None:
    with pytest.raises(ValueError, match="source_audio_ref"):
        HumanTurnInput(
            session_id="session-human",
            text="今日はどんなことを相談したいですか。",
            recipient_ids=("client",),
            source_audio_ref=" ",
        )


def test_human_turn_input_rejects_empty_recipient_id() -> None:
    with pytest.raises(ValueError, match="recipient_ids"):
        HumanTurnInput(
            session_id="session-human",
            text="今日はどんなことを相談したいですか。",
            recipient_ids=("client", " "),
        )


def test_runtime_config_accepts_human_counselor_interaction_mode() -> None:
    config = RuntimeConfig(
        interaction_mode=InteractionMode.HUMAN_COUNSELOR_AI_CLIENT,
        participant_mode=ParticipantMode.TWO_CLIENTS,
        human_input_mode="push_to_talk",
        human_stt_submit_policy="auto_on_final",
        human_interrupts_enabled=False,
        participants={
            "counselor": ParticipantConfig(
                speaker_id="counselor",
                role="counselor",
                display_name="カウンセラー",
                actor_kind=ActorKind.HUMAN,
            ),
            "client_a": ParticipantConfig(
                speaker_id="client_a",
                role="client",
                display_name="妻",
            ),
            "client_b": ParticipantConfig(
                speaker_id="client_b",
                role="client",
                display_name="夫",
            ),
        },
        fixed_speaker_sequence=("counselor", "client_a", "client_b"),
    )

    assert config.interaction_mode is InteractionMode.HUMAN_COUNSELOR_AI_CLIENT
    assert config.participants["counselor"].actor_kind is ActorKind.HUMAN
