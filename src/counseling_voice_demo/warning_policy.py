from __future__ import annotations

from counseling_voice_demo.models import Turn, TurnStatus, WarningLevel


HOLD_WARNING_LEVELS = {WarningLevel.HIGH}


def warning_rank(warning_level: WarningLevel) -> int:
    return {
        WarningLevel.NONE: 0,
        WarningLevel.LOW: 1,
        WarningLevel.MEDIUM: 2,
        WarningLevel.HIGH: 3,
    }[warning_level]


def apply_warning(turn: Turn, warning_level: WarningLevel, *, hold_reason: str | None) -> None:
    if warning_rank(warning_level) > warning_rank(turn.warning_level):
        turn.warning_level = warning_level
    revision = turn.active_revision()
    if warning_rank(warning_level) > warning_rank(revision.warning_level):
        revision.warning_level = warning_level
        revision.hold_reason = hold_reason
    elif revision.hold_reason is None and hold_reason is not None:
        revision.hold_reason = hold_reason


def should_hold_unplayed(turn: Turn) -> bool:
    revision = turn.active_revision()
    return turn.warning_level in HOLD_WARNING_LEVELS or revision.warning_level in HOLD_WARNING_LEVELS


def apply_playback_hold_if_needed(turn: Turn) -> bool:
    if not should_hold_unplayed(turn):
        return False
    turn.status = TurnStatus.PLAYBACK_HOLD
    turn.playback_status = TurnStatus.PLAYBACK_HOLD
    revision = turn.active_revision()
    if revision.hold_reason is None:
        revision.hold_reason = "warning_requires_review"
    return True
