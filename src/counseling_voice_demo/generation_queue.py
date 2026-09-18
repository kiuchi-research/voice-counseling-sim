from __future__ import annotations

from counseling_voice_demo.models import SessionState, SessionStatus, Turn, TurnStatus


UNPLAYED_STATUSES = {
    TurnStatus.TEXT_READY,
    TurnStatus.SAFETY_CHECKED,
    TurnStatus.CLINICAL_CHECKED,
    TurnStatus.AUDIO_PENDING,
    TurnStatus.AUDIO_GENERATING,
    TurnStatus.AUDIO_READY,
    TurnStatus.QUEUED_FOR_PLAYBACK,
    TurnStatus.PLAYBACK_HOLD,
    TurnStatus.PAUSED_GENERATED,
}


def is_unplayed_turn(turn: Turn) -> bool:
    return turn.status in UNPLAYED_STATUSES


def count_unplayed_turns(
    session: SessionState,
    *,
    exclude_turn_ids: set[int] | None = None,
) -> int:
    excluded = exclude_turn_ids or set()
    return sum(
        1
        for turn in session.turns
        if turn.turn_id not in excluded and is_unplayed_turn(turn)
    )


def has_playback_hold(session: SessionState) -> bool:
    return any(turn.status == TurnStatus.PLAYBACK_HOLD for turn in session.turns)


def generation_slots_available(
    session: SessionState,
    *,
    ahead_generation_turns: int,
    exclude_turn_ids: set[int] | None = None,
) -> int:
    if session.status != SessionStatus.RUNNING:
        return 0
    if has_playback_hold(session):
        return 0
    return max(
        ahead_generation_turns
        - count_unplayed_turns(session, exclude_turn_ids=exclude_turn_ids),
        0,
    )
