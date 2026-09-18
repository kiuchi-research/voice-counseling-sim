from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any
from uuid import uuid4


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class SessionStatus(StrEnum):
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    STOPPED = "stopped"
    ERROR = "error"


class ConversationPhase(StrEnum):
    OPENING = "opening"
    MAIN = "main"
    CLOSING = "closing"
    FAREWELL = "farewell"


class SpeakerRole(StrEnum):
    COUNSELOR = "counselor"
    CLIENT = "client"


class WarningLevel(StrEnum):
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class TurnStatus(StrEnum):
    PLANNED = "planned"
    TEXT_PENDING = "text_pending"
    TEXT_GENERATING = "text_generating"
    TEXT_READY = "text_ready"
    SAFETY_CHECKED = "safety_checked"
    CLINICAL_CHECKED = "clinical_checked"
    AUDIO_PENDING = "audio_pending"
    AUDIO_GENERATING = "audio_generating"
    AUDIO_READY = "audio_ready"
    QUEUED_FOR_PLAYBACK = "queued_for_playback"
    PLAYBACK_HOLD = "playback_hold"
    PLAYING = "playing"
    PLAYED = "played"
    PAUSED_GENERATED = "paused_generated"
    SKIPPED_PARTIAL = "skipped_partial"
    SKIPPED_UNPLAYED = "skipped_unplayed"
    INVALIDATED = "invalidated"
    IGNORED_AFTER_STOP = "ignored_after_stop"
    ERROR = "error"


class SkipStatus(StrEnum):
    NONE = "none"
    SKIPPED_PARTIAL = "skipped_partial"
    SKIPPED_UNPLAYED = "skipped_unplayed"


@dataclass
class TurnRevision:
    turn_id: int
    revision_id: int
    text_original: str
    text_final: str | None = None
    edited: bool = False
    regenerated: bool = False
    active: bool = True
    invalidated_reason: str | None = None
    audio_path: str | None = None
    audio_format: str | None = None
    moderation_result: dict[str, Any] | None = None
    clinical_ng_result: dict[str, Any] | None = None
    warning_level: WarningLevel = WarningLevel.NONE
    hold_reason: str | None = None
    generated_at: datetime = field(default_factory=utc_now)
    tts_generated_at: datetime | None = None
    spoken_at: datetime | None = None
    played_audio_elapsed_seconds_start: float | None = None
    played_audio_elapsed_seconds_end: float | None = None
    actual_played_seconds: float = 0.0
    skip_status: SkipStatus = SkipStatus.NONE

    @property
    def canonical_text(self) -> str:
        return self.text_final if self.text_final is not None else self.text_original

    def deactivate(self, reason: str) -> None:
        self.active = False
        self.invalidated_reason = reason


@dataclass
class Turn:
    session_id: str
    turn_id: int
    speaker_role: SpeakerRole
    speaker_name: str
    profile_id: str
    active_revision_id: int
    # status is the generation/safety lifecycle. playback_status is the audio playback lifecycle.
    # They start aligned, then later controllers may update them independently.
    status: TurnStatus = TurnStatus.PLANNED
    warning_level: WarningLevel = WarningLevel.NONE
    playback_status: TurnStatus = TurnStatus.PLANNED
    revisions: list[TurnRevision] = field(default_factory=list)
    created_at: datetime = field(default_factory=utc_now)
    updated_at: datetime = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        active_revisions = [revision for revision in self.revisions if revision.active]
        active_matches = [
            revision
            for revision in active_revisions
            if revision.revision_id == self.active_revision_id and revision.active
        ]
        if len(active_revisions) != 1 or len(active_matches) != 1:
            raise ValueError(
                f"Turn requires exactly one active revision and it must match active_revision_id: "
                f"turn_id={self.turn_id} active_revision_id={self.active_revision_id}"
            )

    @classmethod
    def create(
        cls,
        *,
        session_id: str,
        turn_id: int,
        speaker_role: SpeakerRole,
        speaker_name: str,
        profile_id: str,
        text: str,
        status: TurnStatus = TurnStatus.TEXT_READY,
    ) -> Turn:
        revision = TurnRevision(turn_id=turn_id, revision_id=1, text_original=text)
        return cls(
            session_id=session_id,
            turn_id=turn_id,
            speaker_role=speaker_role,
            speaker_name=speaker_name,
            profile_id=profile_id,
            active_revision_id=1,
            status=status,
            playback_status=status,
            revisions=[revision],
        )

    def active_revision(self) -> TurnRevision:
        for revision in self.revisions:
            if revision.revision_id == self.active_revision_id and revision.active:
                return revision
        raise ValueError(f"active revision is missing: turn_id={self.turn_id} revision_id={self.active_revision_id}")

    def next_revision_id(self) -> int:
        if not self.revisions:
            return 1
        return max(revision.revision_id for revision in self.revisions) + 1

    def add_revision(
        self,
        *,
        text: str,
        edited: bool = False,
        regenerated: bool = False,
        invalidates_previous_reason: str = "superseded",
    ) -> TurnRevision:
        for revision in self.revisions:
            if revision.active:
                revision.deactivate(invalidates_previous_reason)

        revision = TurnRevision(
            turn_id=self.turn_id,
            revision_id=self.next_revision_id(),
            text_original=text,
            text_final=text if edited else None,
            edited=edited,
            regenerated=regenerated,
        )
        self.revisions.append(revision)
        self.active_revision_id = revision.revision_id
        self.updated_at = utc_now()
        return revision

    def invalidate(self, reason: str) -> None:
        self.status = TurnStatus.INVALIDATED
        self.playback_status = TurnStatus.INVALIDATED
        for revision in self.revisions:
            if revision.active:
                revision.deactivate(reason)
        self.updated_at = utc_now()

    def mark_ignored_after_stop(self, reason: str = "completed_after_stop") -> None:
        self.status = TurnStatus.IGNORED_AFTER_STOP
        self.playback_status = TurnStatus.IGNORED_AFTER_STOP
        for revision in self.revisions:
            if revision.active:
                revision.deactivate(reason)
        self.updated_at = utc_now()


@dataclass
class SessionState:
    session_id: str = field(default_factory=lambda: f"session_{uuid4().hex[:12]}")
    status: SessionStatus = SessionStatus.IDLE
    conversation_phase: ConversationPhase = ConversationPhase.OPENING
    turns: list[Turn] = field(default_factory=list)
    created_at: datetime = field(default_factory=utc_now)
    updated_at: datetime = field(default_factory=utc_now)

    def next_turn_id(self) -> int:
        if not self.turns:
            return 1
        return max(turn.turn_id for turn in self.turns) + 1

    def add_turn(
        self,
        *,
        speaker_role: SpeakerRole,
        speaker_name: str,
        profile_id: str,
        text: str,
        status: TurnStatus = TurnStatus.TEXT_READY,
    ) -> Turn:
        turn = Turn.create(
            session_id=self.session_id,
            turn_id=self.next_turn_id(),
            speaker_role=speaker_role,
            speaker_name=speaker_name,
            profile_id=profile_id,
            text=text,
            status=status,
        )
        self.turns.append(turn)
        self.updated_at = utc_now()
        return turn
