from __future__ import annotations

import json

import pytest

from counseling_voice_demo.content_loader import Profile, Theme
from counseling_voice_demo.conversation_engine import (
    ConversationEngine,
    ConversationEngineConfig,
    ConversationParticipants,
)
from counseling_voice_demo.log_writer import create_session_log_dirs
from counseling_voice_demo.models import SessionState, SessionStatus, SpeakerRole, TurnStatus
from counseling_voice_demo.tts_queue import next_tts_turn
from counseling_voice_demo.turn_editing import (
    edit_unplayed_turn,
    invalidate_from_turn,
    regenerate_from_turn,
    replace_with_regenerated_turn,
)


def _session_with_unplayed_turns() -> SessionState:
    session = SessionState(session_id="session_edit_test")
    session.add_turn(
        speaker_role=SpeakerRole.COUNSELOR,
        speaker_name="佐伯",
        profile_id="counselor_default",
        text="今日はどんなことを話したいですか。",
        status=TurnStatus.PLAYED,
    )
    session.add_turn(
        speaker_role=SpeakerRole.CLIENT,
        speaker_name="高橋",
        profile_id="client_default",
        text="家族のことで困っています。",
        status=TurnStatus.AUDIO_READY,
    )
    session.add_turn(
        speaker_role=SpeakerRole.COUNSELOR,
        speaker_name="佐伯",
        profile_id="counselor_default",
        text="もう少し詳しく教えてください。",
        status=TurnStatus.AUDIO_READY,
    )
    return session


def test_edit_unplayed_turn_adds_revision_and_invalidates_following_unplayed_turns() -> None:
    session = _session_with_unplayed_turns()

    result = edit_unplayed_turn(
        session,
        turn_id=2,
        edited_text="家族との距離感で困っています。",
    )

    edited_turn = session.turns[1]
    assert result.turn is edited_turn
    assert result.revision.revision_id == 2
    assert edited_turn.active_revision_id == 2
    assert edited_turn.active_revision().canonical_text == "家族との距離感で困っています。"
    assert edited_turn.active_revision().edited is True
    assert edited_turn.revisions[0].active is False
    assert edited_turn.revisions[0].invalidated_reason == "edited_by_operator"
    assert edited_turn.status == TurnStatus.TEXT_READY
    assert edited_turn.playback_status == TurnStatus.TEXT_READY
    assert [turn.turn_id for turn in result.invalidated_following_turns] == [3]
    assert session.turns[2].status == TurnStatus.INVALIDATED
    assert session.turns[2].revisions[0].active is False


def test_edit_played_turn_is_rejected_by_default() -> None:
    session = _session_with_unplayed_turns()

    with pytest.raises(ValueError, match="未再生ターンだけを編集できます"):
        edit_unplayed_turn(session, turn_id=1, edited_text="再生済みを変えたい。")


def test_replace_with_regenerated_turn_marks_revision_and_invalidates_following_unplayed() -> None:
    session = _session_with_unplayed_turns()

    result = replace_with_regenerated_turn(
        session,
        turn_id=2,
        regenerated_text="家族との話し合いがうまくいかず、困っています。",
    )

    regenerated_turn = session.turns[1]
    assert result.revision.revision_id == 2
    assert regenerated_turn.active_revision().regenerated is True
    assert regenerated_turn.active_revision().canonical_text == "家族との話し合いがうまくいかず、困っています。"
    assert regenerated_turn.revisions[0].invalidated_reason == "regenerated_by_operator"
    assert [turn.turn_id for turn in result.invalidated_following_turns] == [3]
    assert session.turns[2].status == TurnStatus.INVALIDATED


def test_edited_turn_is_requeued_for_tts_with_new_revision_text() -> None:
    session = _session_with_unplayed_turns()
    turn = session.turns[1]
    turn.active_revision().audio_path = "old_turn_0002_rev01.mp3"
    turn.active_revision().audio_format = "mp3"

    result = edit_unplayed_turn(
        session,
        turn_id=2,
        edited_text="家族との距離感で困っています。",
    )

    assert next_tts_turn(session) is turn
    assert result.revision.audio_path is None
    assert result.revision.audio_format is None
    assert result.revision.canonical_text == "家族との距離感で困っています。"
    assert turn.revisions[0].audio_path == "old_turn_0002_rev01.mp3"
    assert turn.revisions[0].active is False


def test_regenerated_turn_is_requeued_for_tts_with_new_revision_text() -> None:
    session = _session_with_unplayed_turns()
    turn = session.turns[1]
    turn.active_revision().audio_path = "old_turn_0002_rev01.mp3"
    turn.active_revision().audio_format = "mp3"

    result = replace_with_regenerated_turn(
        session,
        turn_id=2,
        regenerated_text="家族との話し合いがうまくいかず、困っています。",
    )

    assert next_tts_turn(session) is turn
    assert result.revision.audio_path is None
    assert result.revision.audio_format is None
    assert result.revision.canonical_text == "家族との話し合いがうまくいかず、困っています。"
    assert turn.revisions[0].audio_path == "old_turn_0002_rev01.mp3"
    assert turn.revisions[0].active is False


def test_invalidate_from_turn_invalidates_selected_and_following_unplayed_turns_only() -> None:
    session = _session_with_unplayed_turns()

    invalidated = invalidate_from_turn(session, turn_id=2, reason="regenerate_from_selected_turn")

    assert [turn.turn_id for turn in invalidated] == [2, 3]
    assert session.turns[0].status == TurnStatus.PLAYED
    assert [turn.status for turn in session.turns[1:]] == [TurnStatus.INVALIDATED, TurnStatus.INVALIDATED]
    assert all(len(turn.revisions) == 1 for turn in session.turns[1:])


def test_regenerate_from_turn_invalidates_selected_turn_and_refills_generation_queue() -> None:
    session = _session_with_unplayed_turns()
    session.status = SessionStatus.RUNNING
    text_client = _FakeTextClient()
    engine = ConversationEngine(
        text_client=text_client,
        participants=_participants(),
        config=ConversationEngineConfig(ahead_generation_turns=2),
    )

    result = regenerate_from_turn(
        session,
        turn_id=2,
        conversation_engine=engine,
        cumulative_audio_seconds=120.0,
    )

    assert [turn.turn_id for turn in result.invalidated_turns] == [2, 3]
    assert [turn.turn_id for turn in result.generated_turns] == [4, 5]
    assert [turn.speaker_role for turn in result.generated_turns] == [
        SpeakerRole.CLIENT,
        SpeakerRole.COUNSELOR,
    ]
    assert [turn.status for turn in session.turns[1:3]] == [TurnStatus.INVALIDATED, TurnStatus.INVALIDATED]
    assert [request.speaker_profile.profile_id for request in text_client.calls] == [
        "client_test",
        "counselor_test",
    ]
    assert next_tts_turn(session) is result.generated_turns[0]


def test_unknown_turn_id_is_reported() -> None:
    session = _session_with_unplayed_turns()

    with pytest.raises(ValueError, match="turn_id=99"):
        edit_unplayed_turn(session, turn_id=99, edited_text="存在しないターンです。")


def test_edit_unplayed_turn_records_intervention_and_turn_events(tmp_path) -> None:
    session = _session_with_unplayed_turns()
    paths = create_session_log_dirs(tmp_path, session.session_id)

    edit_unplayed_turn(
        session,
        turn_id=2,
        edited_text="家族との距離感で困っています。",
        log_paths=paths,
    )

    interventions = _read_jsonl(paths.interventions_jsonl)
    events = _read_jsonl(paths.turn_events_jsonl)

    assert interventions == [
        {
            "action": "edit_unplayed_turn",
            "details": {
                "invalidated_following_turn_ids": [3],
                "new_revision_id": 2,
                "previous_revision_id": 1,
                "turn_id": 2,
            },
            "recorded_at": interventions[0]["recorded_at"],
        }
    ]
    assert [event["event_type"] for event in events] == [
        "turn_edited",
        "turn_invalidated_after_edit",
    ]
    assert events[0]["turn_id"] == 2
    assert events[0]["details"]["new_revision_id"] == 2
    assert events[1]["turn_id"] == 3
    assert events[1]["status"] == TurnStatus.INVALIDATED.value


def test_regenerate_from_turn_records_each_invalidated_turn(tmp_path) -> None:
    session = _session_with_unplayed_turns()
    paths = create_session_log_dirs(tmp_path, session.session_id)

    invalidate_from_turn(
        session,
        turn_id=2,
        reason="regenerate_from_selected_turn",
        log_paths=paths,
    )

    interventions = _read_jsonl(paths.interventions_jsonl)
    events = _read_jsonl(paths.turn_events_jsonl)

    assert interventions[0]["action"] == "regenerate_from_turn"
    assert interventions[0]["details"]["invalidated_turn_ids"] == [2, 3]
    assert [event["event_type"] for event in events] == [
        "turn_invalidated_for_regenerate_from_turn",
        "turn_invalidated_for_regenerate_from_turn",
    ]
    assert [event["turn_id"] for event in events] == [2, 3]


def test_replace_with_regenerated_turn_records_intervention_and_turn_events(tmp_path) -> None:
    session = _session_with_unplayed_turns()
    paths = create_session_log_dirs(tmp_path, session.session_id)

    replace_with_regenerated_turn(
        session,
        turn_id=2,
        regenerated_text="家族との話し合いがうまくいかず、困っています。",
        log_paths=paths,
    )

    interventions = _read_jsonl(paths.interventions_jsonl)
    events = _read_jsonl(paths.turn_events_jsonl)

    assert interventions[0]["action"] == "regenerate_unplayed_turn"
    assert interventions[0]["details"]["previous_revision_id"] == 1
    assert interventions[0]["details"]["new_revision_id"] == 2
    assert interventions[0]["details"]["invalidated_following_turn_ids"] == [3]
    assert [event["event_type"] for event in events] == [
        "turn_regenerated",
        "turn_invalidated_after_regenerate",
    ]
    assert events[0]["turn_id"] == 2
    assert events[1]["turn_id"] == 3


def _read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


class _FakeTextClient:
    def __init__(self) -> None:
        self.calls = []

    def generate_turn(self, request):
        self.calls.append(request)
        text = f"{request.speaker_profile.display_name}の再生成{len(self.calls)}"
        return type("Result", (), {"text": text, "raw_response": {"id": f"resp_{len(self.calls)}"}})()


def _participants() -> ConversationParticipants:
    return ConversationParticipants(
        counselor_profile=Profile(
            profile_id="counselor_test",
            role="counselor",
            display_name="佐伯",
            text_model="gpt-test",
            temperature=0.4,
            voice_preset="counselor",
            tts_model="gpt-4o-mini-tts",
            tts_voice="coral",
            tts_instructions="落ち着いて話す。",
            prompt="あなたはカウンセラー役です。",
        ),
        client_profile=Profile(
            profile_id="client_test",
            role="client",
            display_name="高橋",
            text_model="gpt-test",
            temperature=0.6,
            voice_preset="client",
            tts_model="gpt-4o-mini-tts",
            tts_voice="alloy",
            tts_instructions="少し不安そうに話す。",
            prompt="あなたはクライアント役です。",
            hidden_background="家族に言えていない背景",
        ),
        theme=Theme(
            theme_id="family_conflict",
            display_name="家族関係の葛藤",
            severity="standard",
            default_client_profile="client_family_default",
            body="家族との距離感に悩む場面。",
        ),
    )
