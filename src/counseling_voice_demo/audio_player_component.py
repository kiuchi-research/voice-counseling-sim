from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha1
from typing import Any, Callable

from counseling_voice_demo.models import SessionState, Turn, TurnStatus


TERMINAL_EVENT_TYPES = {"playback_ended", "playback_skipped"}
CONTROL_EVENT_TYPES = {
    "playback_started",
    "playback_paused",
    "playback_resumed",
    "position_update",
}
ERROR_EVENT_TYPES = {"playback_error"}
SUPPORTED_EVENT_TYPES = TERMINAL_EVENT_TYPES | CONTROL_EVENT_TYPES | ERROR_EVENT_TYPES


@dataclass(frozen=True)
class AudioQueueItem:
    turn_id: int
    revision_id: int
    speaker_role: str
    speaker_name: str
    text: str
    audio_src: str
    duration_hint_seconds: float
    queue_version: str

    def to_component_value(self) -> dict[str, Any]:
        return {
            "turn_id": self.turn_id,
            "revision_id": self.revision_id,
            "speaker_role": self.speaker_role,
            "speaker_name": self.speaker_name,
            "text": self.text,
            "audio_src": self.audio_src,
            "duration_hint_seconds": self.duration_hint_seconds,
            "queue_version": self.queue_version,
        }


@dataclass(frozen=True)
class BrowserAudioEvent:
    event_id: str
    event_type: str
    turn_id: int
    revision_id: int
    queue_version: str
    current_seconds: float = 0.0
    played_seconds: float = 0.0
    duration_seconds: float = 0.0
    error_message: str = ""


def playable_turns_for_browser_queue(
    session: SessionState,
    *,
    max_items: int = 5,
) -> list[Turn]:
    playable: list[Turn] = []
    seen_turn_revisions: set[tuple[int, int]] = set()
    for turn in sorted(session.turns, key=lambda item: item.turn_id):
        if turn.status in {
            TurnStatus.INVALIDATED,
            TurnStatus.IGNORED_AFTER_STOP,
            TurnStatus.ERROR,
            TurnStatus.PLAYED,
            TurnStatus.SKIPPED_PARTIAL,
            TurnStatus.SKIPPED_UNPLAYED,
        }:
            continue
        turn_revision = (turn.turn_id, turn.active_revision_id)
        if turn_revision in seen_turn_revisions:
            continue
        if turn.status == TurnStatus.PLAYBACK_HOLD:
            break
        if turn.status != TurnStatus.AUDIO_READY:
            break
        if turn.active_revision().audio_path is None:
            break
        seen_turn_revisions.add(turn_revision)
        playable.append(turn)
        if len(playable) >= max_items:
            break
    return playable


def compute_audio_queue_version(turns: list[Turn]) -> str:
    payload = "|".join(
        ":".join(
            [
                str(turn.turn_id),
                str(turn.active_revision_id),
                turn.status.value,
                turn.playback_status.value,
                str(turn.active_revision().audio_path or ""),
            ]
        )
        for turn in turns
    )
    return sha1(payload.encode("utf-8")).hexdigest()[:16]


def build_browser_audio_queue(
    session: SessionState,
    *,
    audio_src_for_turn: Callable[[Turn], str | None],
    duration_hint_for_turn: Callable[[Turn], float],
    max_items: int = 5,
) -> tuple[str, list[dict[str, Any]]]:
    turns = playable_turns_for_browser_queue(session, max_items=max_items)
    queue_version = compute_audio_queue_version(turns)
    items: list[dict[str, Any]] = []
    for turn in turns:
        audio_src = audio_src_for_turn(turn)
        if not audio_src:
            break
        revision = turn.active_revision()
        item = AudioQueueItem(
            turn_id=turn.turn_id,
            revision_id=revision.revision_id,
            speaker_role=turn.speaker_role.value,
            speaker_name=turn.speaker_name,
            text=revision.canonical_text,
            audio_src=audio_src,
            duration_hint_seconds=max(0.0, float(duration_hint_for_turn(turn))),
            queue_version=queue_version,
        )
        items.append(item.to_component_value())
    return queue_version, items


def parse_browser_audio_event(value: Any) -> BrowserAudioEvent | None:
    if not isinstance(value, dict):
        return None
    event_type = str(value.get("event_type") or "")
    if event_type not in SUPPORTED_EVENT_TYPES:
        return None
    event_id = str(value.get("event_id") or "")
    queue_version = str(value.get("queue_version") or "")
    if not event_id or not queue_version:
        return None
    try:
        turn_id = int(value.get("turn_id"))
        revision_id = int(value.get("revision_id"))
    except (TypeError, ValueError):
        return None
    return BrowserAudioEvent(
        event_id=event_id,
        event_type=event_type,
        turn_id=turn_id,
        revision_id=revision_id,
        queue_version=queue_version,
        current_seconds=_float_or_zero(value.get("current_seconds")),
        played_seconds=_float_or_zero(value.get("played_seconds")),
        duration_seconds=_float_or_zero(value.get("duration_seconds")),
        error_message=str(value.get("error_message") or ""),
    )


def event_was_processed(
    event: BrowserAudioEvent | None,
    processed_event_ids: list[str] | set[str] | tuple[str, ...],
) -> bool:
    if event is None:
        return True
    return event.event_id in set(processed_event_ids)


def remember_processed_event_id(
    processed_event_ids: list[str] | set[str] | tuple[str, ...],
    event_id: str,
    *,
    max_event_ids: int = 200,
) -> list[str]:
    ids = list(processed_event_ids)
    if event_id not in ids:
        ids.append(event_id)
    return ids[-max_event_ids:]


def _float_or_zero(value: Any) -> float:
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        return 0.0
