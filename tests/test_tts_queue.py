from __future__ import annotations

from counseling_voice_demo.models import SessionState, SpeakerRole, TurnStatus, WarningLevel
from counseling_voice_demo.tts_queue import mark_tts_ready, next_tts_turn


def test_next_tts_turn_uses_smallest_text_ready_turn_id() -> None:
    session = SessionState()
    later = session.add_turn(
        speaker_role=SpeakerRole.CLIENT,
        speaker_name="高橋",
        profile_id="client",
        text="少し困っています。",
        status=TurnStatus.TEXT_READY,
    )
    earlier = session.add_turn(
        speaker_role=SpeakerRole.COUNSELOR,
        speaker_name="佐伯",
        profile_id="counselor",
        text="もう少し聞かせてください。",
        status=TurnStatus.TEXT_READY,
    )
    later.turn_id = 2
    earlier.turn_id = 1

    assert next_tts_turn(session) is earlier


def test_next_tts_turn_includes_playback_hold_without_audio() -> None:
    session = SessionState()
    held = session.add_turn(
        speaker_role=SpeakerRole.CLIENT,
        speaker_name="高橋",
        profile_id="client",
        text="確認が必要な発話です。",
        status=TurnStatus.PLAYBACK_HOLD,
    )
    held.warning_level = WarningLevel.HIGH
    held.active_revision().warning_level = WarningLevel.HIGH
    held.active_revision().hold_reason = "moderation_flagged"

    assert next_tts_turn(session) is held


def test_mark_tts_ready_keeps_medium_warning_ready_for_playback() -> None:
    session = SessionState()
    turn = session.add_turn(
        speaker_role=SpeakerRole.CLIENT,
        speaker_name="高橋",
        profile_id="client",
        text="診断を断定する発話です。",
        status=TurnStatus.TEXT_READY,
    )
    turn.warning_level = WarningLevel.MEDIUM
    turn.active_revision().warning_level = WarningLevel.MEDIUM
    turn.active_revision().hold_reason = "clinical_ng_medium"

    mark_tts_ready(turn)

    assert turn.status == TurnStatus.AUDIO_READY
    assert turn.playback_status == TurnStatus.AUDIO_READY


def test_mark_tts_ready_moves_high_warning_to_playback_hold() -> None:
    session = SessionState()
    turn = session.add_turn(
        speaker_role=SpeakerRole.CLIENT,
        speaker_name="高橋",
        profile_id="client",
        text="高リスク警告の発話です。",
        status=TurnStatus.TEXT_READY,
    )
    turn.warning_level = WarningLevel.HIGH
    turn.active_revision().warning_level = WarningLevel.HIGH
    turn.active_revision().hold_reason = "moderation_flagged"

    mark_tts_ready(turn)

    assert turn.status == TurnStatus.PLAYBACK_HOLD
    assert turn.playback_status == TurnStatus.PLAYBACK_HOLD
