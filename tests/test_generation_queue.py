from __future__ import annotations

from counseling_voice_demo.generation_queue import (
    count_unplayed_turns,
    generation_slots_available,
    has_playback_hold,
)
from counseling_voice_demo.models import SessionState, SessionStatus, SpeakerRole, TurnStatus


def test_generation_slots_fill_lookahead_only_while_running() -> None:
    session = SessionState(status=SessionStatus.RUNNING)
    session.add_turn(
        speaker_role=SpeakerRole.COUNSELOR,
        speaker_name="佐伯",
        profile_id="counselor",
        text="今日はどんなことを話したいですか。",
        status=TurnStatus.AUDIO_READY,
    )

    assert count_unplayed_turns(session) == 1
    assert generation_slots_available(session, ahead_generation_turns=2) == 1

    session.status = SessionStatus.PAUSED
    assert generation_slots_available(session, ahead_generation_turns=2) == 0


def test_generation_slots_can_exclude_currently_playing_turn_from_buffer_count() -> None:
    session = SessionState(status=SessionStatus.RUNNING)
    first = session.add_turn(
        speaker_role=SpeakerRole.COUNSELOR,
        speaker_name="佐伯",
        profile_id="counselor",
        text="再生中です。",
        status=TurnStatus.AUDIO_READY,
    )
    session.add_turn(
        speaker_role=SpeakerRole.CLIENT,
        speaker_name="高橋",
        profile_id="client",
        text="次に再生します。",
        status=TurnStatus.AUDIO_READY,
    )

    assert generation_slots_available(session, ahead_generation_turns=2) == 0
    assert (
        generation_slots_available(
            session,
            ahead_generation_turns=2,
            exclude_turn_ids={first.turn_id},
        )
        == 1
    )


def test_playback_hold_blocks_additional_lookahead_generation() -> None:
    session = SessionState(status=SessionStatus.RUNNING)
    session.add_turn(
        speaker_role=SpeakerRole.CLIENT,
        speaker_name="高橋",
        profile_id="client",
        text="不安が強いです。",
        status=TurnStatus.PLAYBACK_HOLD,
    )

    assert has_playback_hold(session) is True
    assert generation_slots_available(session, ahead_generation_turns=3) == 0
