from __future__ import annotations

import pytest

from counseling_voice_demo.models import (
    ConversationPhase,
    SessionState,
    SessionStatus,
    SpeakerRole,
    TurnStatus,
    Turn,
    TurnRevision,
    WarningLevel,
)


def test_session_state_has_status_and_conversation_phase() -> None:
    session = SessionState()

    assert session.status == SessionStatus.IDLE
    assert session.conversation_phase == ConversationPhase.OPENING


def test_turn_and_revision_are_created_with_revision_one() -> None:
    session = SessionState(session_id="session_test")
    turn = session.add_turn(
        speaker_role=SpeakerRole.COUNSELOR,
        speaker_name="佐伯",
        profile_id="counselor_brief_default",
        text="こんにちは。",
    )

    assert turn.turn_id == 1
    assert turn.active_revision_id == 1
    assert turn.active_revision().revision_id == 1
    assert turn.active_revision().canonical_text == "こんにちは。"


def test_revision_id_increments_and_old_revision_is_kept_inactive() -> None:
    session = SessionState(session_id="session_test")
    turn = session.add_turn(
        speaker_role=SpeakerRole.CLIENT,
        speaker_name="高橋",
        profile_id="client_family_default",
        text="少し困っています。",
    )

    new_revision = turn.add_revision(text="家族のことで少し困っています。", edited=True)

    assert new_revision.revision_id == 2
    assert turn.active_revision_id == 2
    assert turn.revisions[0].active is False
    assert turn.revisions[0].invalidated_reason == "superseded"
    assert turn.revisions[1].active is True
    assert turn.revisions[1].edited is True


def test_multiple_revision_ids_increment_monotonically_and_regenerated_flag_is_kept() -> None:
    session = SessionState(session_id="session_test")
    turn = session.add_turn(
        speaker_role=SpeakerRole.CLIENT,
        speaker_name="高橋",
        profile_id="client_family_default",
        text="最初の発話です。",
    )

    turn.add_revision(text="編集後の発話です。", edited=True)
    third_revision = turn.add_revision(text="再生成後の発話です。", regenerated=True)

    assert [revision.revision_id for revision in turn.revisions] == [1, 2, 3]
    assert third_revision.revision_id == 3
    assert third_revision.regenerated is True
    assert turn.active_revision().canonical_text == "再生成後の発話です。"


def test_turn_statuses_distinguish_playback_hold_skip_and_after_stop() -> None:
    assert TurnStatus.PLAYBACK_HOLD.value == "playback_hold"
    assert TurnStatus.SKIPPED_PARTIAL.value == "skipped_partial"
    assert TurnStatus.SKIPPED_UNPLAYED.value == "skipped_unplayed"
    assert TurnStatus.IGNORED_AFTER_STOP.value == "ignored_after_stop"
    assert WarningLevel.MEDIUM.value == "medium"


def test_mark_ignored_after_stop_deactivates_active_revision() -> None:
    session = SessionState(session_id="session_test")
    turn = session.add_turn(
        speaker_role=SpeakerRole.COUNSELOR,
        speaker_name="佐伯",
        profile_id="counselor_brief_default",
        text="後から返ってきた応答です。",
    )

    turn.mark_ignored_after_stop()

    assert turn.status == TurnStatus.IGNORED_AFTER_STOP
    assert turn.revisions[0].active is False
    assert turn.revisions[0].invalidated_reason == "completed_after_stop"


def test_invalidate_deactivates_active_revision_without_deleting_it() -> None:
    session = SessionState(session_id="session_test")
    turn = session.add_turn(
        speaker_role=SpeakerRole.CLIENT,
        speaker_name="高橋",
        profile_id="client_family_default",
        text="無効化される発話です。",
    )

    turn.invalidate("edit_invalidated_following_turn")

    assert turn.status == TurnStatus.INVALIDATED
    assert turn.playback_status == TurnStatus.INVALIDATED
    assert len(turn.revisions) == 1
    assert turn.revisions[0].active is False
    assert turn.revisions[0].invalidated_reason == "edit_invalidated_following_turn"


def test_turn_requires_matching_active_revision_at_creation() -> None:
    with pytest.raises(ValueError, match="must match active_revision_id"):
        Turn(
            session_id="session_test",
            turn_id=1,
            speaker_role=SpeakerRole.CLIENT,
            speaker_name="高橋",
            profile_id="client_family_default",
            active_revision_id=2,
            revisions=[TurnRevision(turn_id=1, revision_id=1, text_original="本文")],
        )


def test_turn_rejects_multiple_active_revisions_at_creation() -> None:
    with pytest.raises(ValueError, match="exactly one active revision"):
        Turn(
            session_id="session_test",
            turn_id=1,
            speaker_role=SpeakerRole.CLIENT,
            speaker_name="高橋",
            profile_id="client_family_default",
            active_revision_id=2,
            revisions=[
                TurnRevision(turn_id=1, revision_id=1, text_original="旧本文"),
                TurnRevision(turn_id=1, revision_id=2, text_original="新本文"),
            ],
        )
