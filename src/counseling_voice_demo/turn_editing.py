from __future__ import annotations

from dataclasses import dataclass

from counseling_voice_demo.generation_queue import is_unplayed_turn
from counseling_voice_demo.log_writer import SessionLogPaths, append_intervention, append_turn_event
from counseling_voice_demo.models import SessionState, Turn, TurnRevision, TurnStatus, WarningLevel, utc_now


@dataclass(frozen=True)
class TurnEditResult:
    turn: Turn
    revision: TurnRevision
    invalidated_following_turns: list[Turn]


@dataclass(frozen=True)
class RegenerateFromTurnResult:
    invalidated_turns: list[Turn]
    generated_turns: list[Turn]


def edit_unplayed_turn(
    session: SessionState,
    *,
    turn_id: int,
    edited_text: str,
    invalidate_following: bool = True,
    allow_played_turns: bool = False,
    log_paths: SessionLogPaths | None = None,
) -> TurnEditResult:
    turn = _editable_turn(session, turn_id=turn_id, allow_played_turns=allow_played_turns)
    previous_revision_id = turn.active_revision_id
    revision = turn.add_revision(
        text=edited_text,
        edited=True,
        invalidates_previous_reason="edited_by_operator",
    )
    _reset_turn_for_new_text(turn)
    invalidated = (
        invalidate_following_unplayed_turns(
            session,
            turn_id=turn_id,
            reason="edit_invalidated_following_turn",
        )
        if invalidate_following
        else []
    )
    session.updated_at = utc_now()
    if log_paths is not None:
        _record_revision_intervention(
            log_paths,
            action="edit_unplayed_turn",
            event_type="turn_edited",
            followup_invalidated_event_type="turn_invalidated_after_edit",
            turn=turn,
            previous_revision_id=previous_revision_id,
            new_revision_id=revision.revision_id,
            invalidated_following_turns=invalidated,
        )
    return TurnEditResult(turn=turn, revision=revision, invalidated_following_turns=invalidated)


def replace_with_regenerated_turn(
    session: SessionState,
    *,
    turn_id: int,
    regenerated_text: str,
    invalidate_following: bool = True,
    allow_played_turns: bool = False,
    log_paths: SessionLogPaths | None = None,
) -> TurnEditResult:
    turn = _editable_turn(session, turn_id=turn_id, allow_played_turns=allow_played_turns)
    previous_revision_id = turn.active_revision_id
    revision = turn.add_revision(
        text=regenerated_text,
        regenerated=True,
        invalidates_previous_reason="regenerated_by_operator",
    )
    _reset_turn_for_new_text(turn)
    invalidated = (
        invalidate_following_unplayed_turns(
            session,
            turn_id=turn_id,
            reason="regenerate_invalidated_following_turn",
        )
        if invalidate_following
        else []
    )
    session.updated_at = utc_now()
    if log_paths is not None:
        _record_revision_intervention(
            log_paths,
            action="regenerate_unplayed_turn",
            event_type="turn_regenerated",
            followup_invalidated_event_type="turn_invalidated_after_regenerate",
            turn=turn,
            previous_revision_id=previous_revision_id,
            new_revision_id=revision.revision_id,
            invalidated_following_turns=invalidated,
        )
    return TurnEditResult(turn=turn, revision=revision, invalidated_following_turns=invalidated)


def invalidate_following_unplayed_turns(session: SessionState, *, turn_id: int, reason: str) -> list[Turn]:
    invalidated: list[Turn] = []
    for turn in sorted(session.turns, key=lambda item: item.turn_id):
        if turn.turn_id <= turn_id:
            continue
        if is_unplayed_turn(turn):
            turn.invalidate(reason)
            invalidated.append(turn)
    if invalidated:
        session.updated_at = utc_now()
    return invalidated


def invalidate_from_turn(
    session: SessionState,
    *,
    turn_id: int,
    reason: str,
    log_paths: SessionLogPaths | None = None,
) -> list[Turn]:
    _find_turn(session, turn_id=turn_id)
    invalidated: list[Turn] = []
    for turn in sorted(session.turns, key=lambda item: item.turn_id):
        if turn.turn_id < turn_id:
            continue
        if is_unplayed_turn(turn):
            turn.invalidate(reason)
            invalidated.append(turn)
    if invalidated:
        session.updated_at = utc_now()
    if log_paths is not None:
        append_intervention(
            log_paths,
            action="regenerate_from_turn",
            details={
                "turn_id": turn_id,
                "invalidated_turn_ids": [turn.turn_id for turn in invalidated],
                "reason": reason,
            },
        )
        for invalidated_turn in invalidated:
            append_turn_event(
                log_paths,
                event_type="turn_invalidated_for_regenerate_from_turn",
                turn=invalidated_turn,
                details={"reason": reason, "selected_turn_id": turn_id},
            )
    return invalidated


def regenerate_from_turn(
    session: SessionState,
    *,
    turn_id: int,
    conversation_engine: object,
    cumulative_audio_seconds: float,
    reason: str = "regenerate_from_selected_turn",
    log_paths: SessionLogPaths | None = None,
) -> RegenerateFromTurnResult:
    invalidated = invalidate_from_turn(
        session,
        turn_id=turn_id,
        reason=reason,
        log_paths=log_paths,
    )
    generated = conversation_engine.fill_generation_queue(
        session,
        cumulative_audio_seconds=cumulative_audio_seconds,
    )
    return RegenerateFromTurnResult(invalidated_turns=invalidated, generated_turns=generated)


def _editable_turn(session: SessionState, *, turn_id: int, allow_played_turns: bool) -> Turn:
    turn = _find_turn(session, turn_id=turn_id)
    if allow_played_turns:
        return turn
    if not is_unplayed_turn(turn):
        raise ValueError(f"未再生ターンだけを編集できます: turn_id={turn_id} status={turn.status.value}")
    return turn


def _find_turn(session: SessionState, *, turn_id: int) -> Turn:
    for turn in session.turns:
        if turn.turn_id == turn_id:
            return turn
    raise ValueError(f"turn_id={turn_id} のターンが見つかりません")


def _reset_turn_for_new_text(turn: Turn) -> None:
    turn.status = TurnStatus.TEXT_READY
    turn.playback_status = TurnStatus.TEXT_READY
    turn.warning_level = WarningLevel.NONE
    revision = turn.active_revision()
    revision.warning_level = WarningLevel.NONE
    revision.hold_reason = None
    revision.audio_path = None
    revision.audio_format = None
    revision.tts_generated_at = None
    revision.spoken_at = None
    revision.played_audio_elapsed_seconds_start = None
    revision.played_audio_elapsed_seconds_end = None
    revision.actual_played_seconds = 0.0
    turn.updated_at = utc_now()


def _record_revision_intervention(
    log_paths: SessionLogPaths,
    *,
    action: str,
    event_type: str,
    followup_invalidated_event_type: str,
    turn: Turn,
    previous_revision_id: int,
    new_revision_id: int,
    invalidated_following_turns: list[Turn],
) -> None:
    append_intervention(
        log_paths,
        action=action,
        details={
            "turn_id": turn.turn_id,
            "previous_revision_id": previous_revision_id,
            "new_revision_id": new_revision_id,
            "invalidated_following_turn_ids": [item.turn_id for item in invalidated_following_turns],
        },
    )
    append_turn_event(
        log_paths,
        event_type=event_type,
        turn=turn,
        details={
            "previous_revision_id": previous_revision_id,
            "new_revision_id": new_revision_id,
        },
    )
    for invalidated_turn in invalidated_following_turns:
        append_turn_event(
            log_paths,
            event_type=followup_invalidated_event_type,
            turn=invalidated_turn,
            details={"source_turn_id": turn.turn_id},
        )
