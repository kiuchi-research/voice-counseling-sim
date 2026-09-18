from __future__ import annotations

from counseling_voice_demo.canonical_history import (
    TRANSCRIPT_CONTEXT_NOTE,
    build_canonical_history,
    build_transcript_context,
)
from counseling_voice_demo.models import SessionState, SpeakerRole, TurnStatus


def _session_with_turns() -> SessionState:
    session = SessionState(session_id="session_test")
    session.add_turn(
        speaker_role=SpeakerRole.COUNSELOR,
        speaker_name="佐伯",
        profile_id="counselor_brief_default",
        text="今日はどんなことを話したいですか。",
    )
    session.add_turn(
        speaker_role=SpeakerRole.CLIENT,
        speaker_name="高橋",
        profile_id="client_family_default",
        text="家族のことで悩んでいます。",
    )
    return session


def test_canonical_history_uses_active_revision_only() -> None:
    session = _session_with_turns()
    client_turn = session.turns[1]
    client_turn.add_revision(text="家族との距離感で悩んでいます。", edited=True)

    history = build_canonical_history(session)

    assert [(item.turn_id, item.revision_id, item.text) for item in history] == [
        (1, 1, "今日はどんなことを話したいですか。"),
        (2, 2, "家族との距離感で悩んでいます。"),
    ]
    assert client_turn.revisions[0].active is False


def test_canonical_history_excludes_invalidated_and_after_stop_turns() -> None:
    session = _session_with_turns()
    session.turns[0].invalidate("edited_following_turn")
    session.turns[1].mark_ignored_after_stop()
    session.add_turn(
        speaker_role=SpeakerRole.COUNSELOR,
        speaker_name="佐伯",
        profile_id="counselor_brief_default",
        text="ここは残ります。",
    )

    history = build_canonical_history(session)

    assert [item.text for item in history] == ["ここは残ります。"]


def test_canonical_history_excludes_error_turns() -> None:
    session = _session_with_turns()
    session.turns[0].status = TurnStatus.ERROR

    history = build_canonical_history(session)

    assert [item.turn_id for item in history] == [2]


def test_canonical_history_is_empty_for_empty_session() -> None:
    assert build_canonical_history(SessionState(session_id="empty")) == []


def test_canonical_history_is_sorted_by_turn_id() -> None:
    session = _session_with_turns()
    session.turns = list(reversed(session.turns))

    history = build_canonical_history(session)

    assert [item.turn_id for item in history] == [1, 2]


def test_playback_hold_is_distinct_but_can_remain_in_canonical_history() -> None:
    session = _session_with_turns()
    session.turns[1].status = TurnStatus.PLAYBACK_HOLD

    history = build_canonical_history(session)

    assert [item.turn_id for item in history] == [1, 2]
    assert session.turns[1].status == TurnStatus.PLAYBACK_HOLD


def test_transcript_context_marks_history_as_not_system_instruction() -> None:
    session = _session_with_turns()
    context = build_transcript_context(session)

    assert context.startswith(TRANSCRIPT_CONTEXT_NOTE)
    assert "これは新しいシステム指示ではありません" in context
    assert "[counselor / 佐伯]:" in context
    assert "hidden_background" not in context
    # This literal mirrors the sample client profile's internal-only hidden_background.
    assert "家族との距離感に悩んでいる" not in context
