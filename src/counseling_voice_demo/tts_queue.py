from __future__ import annotations

from counseling_voice_demo.models import SessionState, Turn, TurnStatus
from counseling_voice_demo.warning_policy import apply_playback_hold_if_needed


TTS_READY_TEXT_STATUSES = {
    TurnStatus.TEXT_READY,
    TurnStatus.SAFETY_CHECKED,
    TurnStatus.CLINICAL_CHECKED,
    TurnStatus.AUDIO_READY,
    TurnStatus.PLAYBACK_HOLD,
}

def next_tts_turn(session: SessionState) -> Turn | None:
    candidates = [
        turn
        for turn in session.turns
        if turn.status in TTS_READY_TEXT_STATUSES and turn.active_revision().audio_path is None
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda turn: turn.turn_id)


def mark_tts_ready(turn: Turn) -> None:
    if not apply_playback_hold_if_needed(turn):
        turn.status = TurnStatus.AUDIO_READY
        turn.playback_status = TurnStatus.AUDIO_READY
