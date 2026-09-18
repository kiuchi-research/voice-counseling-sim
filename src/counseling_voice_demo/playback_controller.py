from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from counseling_voice_demo.log_writer import SessionLogPaths, append_turn_event
from counseling_voice_demo.models import SkipStatus, SessionState, Turn, TurnStatus, utc_now


class PlaybackOutcome(StrEnum):
    PLAYED = "played"
    SKIPPED_PARTIAL = "skipped_partial"
    SKIPPED_UNPLAYED = "skipped_unplayed"


@dataclass(frozen=True)
class PlaybackEvent:
    event_name: str
    current_seconds: float
    duration_seconds: float


@dataclass(frozen=True)
class PlaybackSummary:
    outcome: PlaybackOutcome
    elapsed_audio_seconds: float


@dataclass(frozen=True)
class PlaybackQueueDecision:
    action: str
    turn: Turn | None
    reason: str


def classify_skip(current_seconds: float) -> PlaybackOutcome:
    if current_seconds <= 0:
        return PlaybackOutcome.SKIPPED_UNPLAYED
    return PlaybackOutcome.SKIPPED_PARTIAL


def summarize_playback_event(event: PlaybackEvent) -> PlaybackSummary:
    current_seconds = max(0.0, event.current_seconds)
    duration_seconds = max(0.0, event.duration_seconds)

    if event.event_name == "ended":
        return PlaybackSummary(
            outcome=PlaybackOutcome.PLAYED,
            elapsed_audio_seconds=duration_seconds,
        )

    if event.event_name == "skip":
        return PlaybackSummary(
            outcome=classify_skip(current_seconds),
            elapsed_audio_seconds=min(current_seconds, duration_seconds),
        )

    raise ValueError(f"Unsupported playback terminal event: {event.event_name}")


def add_elapsed_once(
    previous_total_seconds: float,
    summary: PlaybackSummary,
    *,
    already_counted: bool,
) -> float:
    if already_counted:
        return previous_total_seconds
    return previous_total_seconds + summary.elapsed_audio_seconds


def next_playback_decision(session: SessionState) -> PlaybackQueueDecision:
    for turn in sorted(session.turns, key=lambda item: item.turn_id):
        if turn.status in {TurnStatus.INVALIDATED, TurnStatus.IGNORED_AFTER_STOP, TurnStatus.ERROR}:
            continue
        if turn.status in {TurnStatus.PLAYED, TurnStatus.SKIPPED_PARTIAL, TurnStatus.SKIPPED_UNPLAYED}:
            continue
        if turn.status == TurnStatus.PLAYBACK_HOLD:
            return PlaybackQueueDecision(action="hold", turn=turn, reason="playback_hold")
        if turn.status == TurnStatus.AUDIO_READY:
            if turn.active_revision().audio_path is None:
                return PlaybackQueueDecision(action="wait", turn=turn, reason="audio_path_missing")
            return PlaybackQueueDecision(action="play", turn=turn, reason="audio_ready")
        return PlaybackQueueDecision(action="wait", turn=turn, reason=turn.status.value)
    return PlaybackQueueDecision(action="complete", turn=None, reason="no_playable_turns")


def record_playback_terminal_event(
    turn: Turn,
    event: PlaybackEvent,
    *,
    previous_total_seconds: float,
    already_counted: bool,
    log_paths: SessionLogPaths | None = None,
) -> float:
    summary = summarize_playback_event(event)
    revision = turn.active_revision()
    revision.played_audio_elapsed_seconds_end = max(0.0, event.current_seconds)
    revision.actual_played_seconds = summary.elapsed_audio_seconds

    if summary.outcome == PlaybackOutcome.PLAYED:
        turn.status = TurnStatus.PLAYED
        turn.playback_status = TurnStatus.PLAYED
        revision.spoken_at = utc_now()
    elif summary.outcome == PlaybackOutcome.SKIPPED_UNPLAYED:
        turn.status = TurnStatus.SKIPPED_UNPLAYED
        turn.playback_status = TurnStatus.SKIPPED_UNPLAYED
        revision.skip_status = SkipStatus.SKIPPED_UNPLAYED
    else:
        turn.status = TurnStatus.SKIPPED_PARTIAL
        turn.playback_status = TurnStatus.SKIPPED_PARTIAL
        revision.skip_status = SkipStatus.SKIPPED_PARTIAL

    if log_paths is not None:
        append_turn_event(
            log_paths,
            event_type=f"playback_{event.event_name}",
            turn=turn,
            details={
                "current_seconds": max(0.0, event.current_seconds),
                "duration_seconds": max(0.0, event.duration_seconds),
                "outcome": summary.outcome.value,
                "elapsed_audio_seconds": summary.elapsed_audio_seconds,
            },
        )

    return add_elapsed_once(previous_total_seconds, summary, already_counted=already_counted)


def release_playback_hold(turn: Turn, *, log_paths: SessionLogPaths | None = None) -> None:
    if turn.status != TurnStatus.PLAYBACK_HOLD:
        raise ValueError(f"playback_hold のターンだけを解除できます: turn_id={turn.turn_id} status={turn.status.value}")
    turn.status = TurnStatus.AUDIO_READY
    turn.playback_status = TurnStatus.AUDIO_READY
    turn.updated_at = utc_now()
    if log_paths is not None:
        append_turn_event(
            log_paths,
            event_type="playback_hold_released",
            turn=turn,
            details={"hold_reason": turn.active_revision().hold_reason},
        )


def skip_unplayed_hold(turn: Turn, *, log_paths: SessionLogPaths | None = None) -> None:
    if turn.status != TurnStatus.PLAYBACK_HOLD:
        raise ValueError(f"playback_hold のターンだけをスキップできます: turn_id={turn.turn_id} status={turn.status.value}")
    turn.status = TurnStatus.SKIPPED_UNPLAYED
    turn.playback_status = TurnStatus.SKIPPED_UNPLAYED
    revision = turn.active_revision()
    revision.skip_status = SkipStatus.SKIPPED_UNPLAYED
    turn.updated_at = utc_now()
    if log_paths is not None:
        append_turn_event(
            log_paths,
            event_type="playback_hold_skipped",
            turn=turn,
            details={"hold_reason": revision.hold_reason},
        )


def record_playback_control_event(
    log_paths: SessionLogPaths,
    *,
    turn: Turn,
    event_name: str,
    current_seconds: float,
) -> None:
    if event_name == "started":
        turn.status = TurnStatus.PLAYING
        turn.playback_status = TurnStatus.PLAYING
        turn.active_revision().played_audio_elapsed_seconds_start = max(0.0, current_seconds)
    elif event_name not in {"pause", "resume"}:
        raise ValueError(f"Unsupported playback control event: {event_name}")

    append_turn_event(
        log_paths,
        event_type=f"playback_{event_name}",
        turn=turn,
        details={"current_seconds": max(0.0, current_seconds)},
    )
