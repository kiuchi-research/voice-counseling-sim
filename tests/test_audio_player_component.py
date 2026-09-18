from __future__ import annotations

from counseling_voice_demo.audio_player_component import (
    build_browser_audio_queue,
    event_was_processed,
    parse_browser_audio_event,
    playable_turns_for_browser_queue,
    remember_processed_event_id,
)
from counseling_voice_demo.models import SessionState, SpeakerRole, TurnStatus


def test_browser_audio_queue_uses_audio_ready_turns_in_turn_id_order() -> None:
    session = SessionState()
    second = session.add_turn(
        speaker_role=SpeakerRole.CLIENT,
        speaker_name="高橋",
        profile_id="client",
        text="2番目です。",
        status=TurnStatus.AUDIO_READY,
    )
    first = session.add_turn(
        speaker_role=SpeakerRole.COUNSELOR,
        speaker_name="佐伯",
        profile_id="counselor",
        text="1番目です。",
        status=TurnStatus.AUDIO_READY,
    )
    second.turn_id = 2
    first.turn_id = 1
    second.active_revision().turn_id = 2
    first.active_revision().turn_id = 1
    second.active_revision().audio_path = "turn2.mp3"
    first.active_revision().audio_path = "turn1.mp3"

    queue_version, queue = build_browser_audio_queue(
        session,
        audio_src_for_turn=lambda turn: f"data:audio/mpeg;base64,{turn.turn_id}",
        duration_hint_for_turn=lambda turn: float(turn.turn_id),
    )

    assert queue_version
    assert [item["turn_id"] for item in queue] == [1, 2]
    assert [item["audio_src"] for item in queue] == [
        "data:audio/mpeg;base64,1",
        "data:audio/mpeg;base64,2",
    ]
    assert all(item["queue_version"] == queue_version for item in queue)


def test_browser_audio_queue_stops_before_playback_hold() -> None:
    session = SessionState()
    held = session.add_turn(
        speaker_role=SpeakerRole.CLIENT,
        speaker_name="高橋",
        profile_id="client",
        text="確認待ちです。",
        status=TurnStatus.PLAYBACK_HOLD,
    )
    following = session.add_turn(
        speaker_role=SpeakerRole.COUNSELOR,
        speaker_name="佐伯",
        profile_id="counselor",
        text="後続です。",
        status=TurnStatus.AUDIO_READY,
    )
    held.active_revision().audio_path = "turn1.mp3"
    following.active_revision().audio_path = "turn2.mp3"

    assert playable_turns_for_browser_queue(session) == []


def test_browser_audio_queue_deduplicates_same_turn_revision() -> None:
    session = SessionState()
    first = session.add_turn(
        speaker_role=SpeakerRole.COUNSELOR,
        speaker_name="佐伯",
        profile_id="counselor",
        text="1回だけ再生します。",
        status=TurnStatus.AUDIO_READY,
    )
    duplicate = session.add_turn(
        speaker_role=SpeakerRole.COUNSELOR,
        speaker_name="佐伯",
        profile_id="counselor",
        text="重複した同一ターンです。",
        status=TurnStatus.PLAYBACK_HOLD,
    )
    second = session.add_turn(
        speaker_role=SpeakerRole.CLIENT,
        speaker_name="高橋",
        profile_id="client",
        text="次のターンです。",
        status=TurnStatus.AUDIO_READY,
    )
    duplicate.turn_id = first.turn_id
    duplicate.active_revision().turn_id = first.turn_id
    second.turn_id = 2
    second.active_revision().turn_id = 2
    first.active_revision().audio_path = "turn1.mp3"
    second.active_revision().audio_path = "turn2.mp3"

    _, queue = build_browser_audio_queue(
        session,
        audio_src_for_turn=lambda turn: (
            f"data:audio/mpeg;base64,{turn.active_revision().canonical_text}"
        ),
        duration_hint_for_turn=lambda _turn: 1.0,
    )

    assert [item["turn_id"] for item in queue] == [1, 2]
    assert [item["text"] for item in queue] == [
        "1回だけ再生します。",
        "次のターンです。",
    ]


def test_browser_audio_queue_version_changes_when_revision_changes() -> None:
    session = SessionState()
    turn = session.add_turn(
        speaker_role=SpeakerRole.COUNSELOR,
        speaker_name="佐伯",
        profile_id="counselor",
        text="最初です。",
        status=TurnStatus.AUDIO_READY,
    )
    turn.active_revision().audio_path = "turn1_rev1.mp3"
    version_before, _ = build_browser_audio_queue(
        session,
        audio_src_for_turn=lambda _turn: "data:audio/mpeg;base64,1",
        duration_hint_for_turn=lambda _turn: 1.0,
    )

    turn.add_revision(text="編集後です。", edited=True)
    turn.status = TurnStatus.AUDIO_READY
    turn.playback_status = TurnStatus.AUDIO_READY
    turn.active_revision().audio_path = "turn1_rev2.mp3"
    version_after, _ = build_browser_audio_queue(
        session,
        audio_src_for_turn=lambda _turn: "data:audio/mpeg;base64,2",
        duration_hint_for_turn=lambda _turn: 1.0,
    )

    assert version_after != version_before


def test_parse_browser_audio_event_and_duplicate_tracking() -> None:
    event = parse_browser_audio_event(
        {
            "event_id": "event-1",
            "event_type": "playback_ended",
            "turn_id": 3,
            "revision_id": 2,
            "queue_version": "queue-a",
            "current_seconds": "1.2",
            "played_seconds": "1.2",
            "duration_seconds": "1.4",
        }
    )

    assert event is not None
    assert event.event_id == "event-1"
    assert event.turn_id == 3
    assert event.revision_id == 2
    assert event.played_seconds == 1.2
    assert event_was_processed(event, []) is False
    processed = remember_processed_event_id([], event.event_id)
    assert event_was_processed(event, processed) is True


def test_parse_browser_audio_event_rejects_invalid_payload() -> None:
    assert parse_browser_audio_event(None) is None
    assert parse_browser_audio_event({"event_type": "unknown"}) is None
    assert (
        parse_browser_audio_event(
            {
                "event_id": "event-1",
                "event_type": "playback_ended",
                "turn_id": "bad",
                "revision_id": 1,
                "queue_version": "queue-a",
            }
        )
        is None
    )
