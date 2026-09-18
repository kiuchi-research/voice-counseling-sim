from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any

from counseling_voice_demo.content_loader import Profile, Theme
from counseling_voice_demo.generation_queue import (
    generation_slots_available,
    is_unplayed_turn,
)
from counseling_voice_demo.models import (
    ConversationPhase,
    SessionState,
    SessionStatus,
    SpeakerRole,
    Turn,
    TurnStatus,
    WarningLevel,
)
from counseling_voice_demo.openai_text_client import TextGenerationRequest


@dataclass(frozen=True)
class ConversationParticipants:
    counselor_profile: Profile
    client_profile: Profile
    theme: Theme


@dataclass(frozen=True)
class ConversationEngineConfig:
    ahead_generation_turns: int
    closing_start_audio_seconds: int = 900
    farewell_after_turns: int = 2
    closing_keep_audio_ready_turns: int = 1
    max_output_tokens: int = 800
    reasoning_effort: str | None = None


@dataclass
class ConversationEngine:
    text_client: object
    participants: ConversationParticipants
    config: ConversationEngineConfig
    safety_check_runner: object | None = None
    _closing_started_turn_count_by_session: dict[str, int] = field(default_factory=dict)
    _generation_lock: Any = field(default_factory=threading.RLock, repr=False)

    def fill_generation_queue(
        self,
        session: SessionState,
        *,
        cumulative_audio_seconds: float,
        exclude_turn_ids_from_buffer: set[int] | None = None,
    ) -> list[Turn]:
        invalidate_unplayed_speaker_sequence_violations(session)
        self.update_conversation_phase(
            session, cumulative_audio_seconds=cumulative_audio_seconds
        )
        generated: list[Turn] = []
        slots = generation_slots_available(
            session,
            ahead_generation_turns=self.config.ahead_generation_turns,
            exclude_turn_ids=exclude_turn_ids_from_buffer,
        )
        for _ in range(slots):
            turn = self.generate_next_turn(
                session, cumulative_audio_seconds=cumulative_audio_seconds
            )
            if turn is None:
                break
            generated.append(turn)
            if session.status != SessionStatus.RUNNING:
                break
        return generated

    def generate_next_turn(
        self, session: SessionState, *, cumulative_audio_seconds: float
    ) -> Turn | None:
        with self._generation_lock:
            return self._generate_next_turn_locked(
                session, cumulative_audio_seconds=cumulative_audio_seconds
            )

    def _generate_next_turn_locked(
        self, session: SessionState, *, cumulative_audio_seconds: float
    ) -> Turn | None:
        if session.status != SessionStatus.RUNNING:
            return None

        speaker_profile = self._next_speaker_profile(session)
        result = None
        for _ in range(3):
            request = self.build_generation_request(
                session,
                speaker_profile=speaker_profile,
                cumulative_audio_seconds=cumulative_audio_seconds,
            )
            result = self.text_client.generate_turn(request)
            if session.status != SessionStatus.RUNNING:
                break
            latest_expected_profile = self._next_speaker_profile(session)
            if latest_expected_profile.profile_id == speaker_profile.profile_id:
                break
            speaker_profile = latest_expected_profile
            result = None
        if result is None:
            return None

        turn = session.add_turn(
            speaker_role=SpeakerRole(speaker_profile.role),
            speaker_name=speaker_profile.display_name,
            profile_id=speaker_profile.profile_id,
            text=result.text,
            status=_turn_status_for_completed_generation(session.status),
        )
        if session.status in {SessionStatus.STOPPED, SessionStatus.COMPLETED}:
            turn.mark_ignored_after_stop()
        elif (
            session.conversation_phase == ConversationPhase.FAREWELL
            and turn.speaker_role == SpeakerRole.CLIENT
        ):
            session.status = SessionStatus.COMPLETED
        elif self.safety_check_runner is not None:
            self.safety_check_runner.check_turn(turn)
        return turn

    def build_generation_request(
        self,
        session: SessionState,
        *,
        speaker_profile: Profile,
        cumulative_audio_seconds: float,
    ) -> TextGenerationRequest:
        return TextGenerationRequest(
            speaker_profile=speaker_profile,
            theme=self.participants.theme,
            session=session,
            conversation_phase=session.conversation_phase,
            cumulative_audio_seconds=cumulative_audio_seconds,
            max_output_tokens=self.config.max_output_tokens,
            reasoning_effort=self.config.reasoning_effort,
            safety_instruction=_safety_instruction_for_next_turn(
                session, speaker_profile
            ),
            closing_instruction=_closing_instruction(session.conversation_phase),
            farewell_instruction=_farewell_instruction(
                session.conversation_phase, speaker_profile
            ),
        )

    def speaker_profile_for_role(self, speaker_role: SpeakerRole) -> Profile:
        if speaker_role == SpeakerRole.COUNSELOR:
            return self.participants.counselor_profile
        if speaker_role == SpeakerRole.CLIENT:
            return self.participants.client_profile
        raise ValueError(f"未対応の話者ロールです: {speaker_role}")

    def update_conversation_phase(
        self, session: SessionState, *, cumulative_audio_seconds: float
    ) -> None:
        if session.conversation_phase == ConversationPhase.OPENING and session.turns:
            session.conversation_phase = ConversationPhase.MAIN

        if (
            session.conversation_phase
            in {ConversationPhase.OPENING, ConversationPhase.MAIN}
            and cumulative_audio_seconds >= self.config.closing_start_audio_seconds
        ):
            session.conversation_phase = ConversationPhase.CLOSING
            invalidate_unplayed_after_keep_limit(
                session,
                keep_turns=self.config.closing_keep_audio_ready_turns,
                reason="closing_started_regenerate_following_turns",
            )
            self._closing_started_turn_count_by_session[session.session_id] = (
                _count_non_invalidated_turns(session)
            )

        if session.conversation_phase == ConversationPhase.CLOSING:
            closing_started_count = (
                self._closing_started_turn_count_by_session.setdefault(
                    session.session_id,
                    _count_non_invalidated_turns(session),
                )
            )
            if (
                _count_non_invalidated_turns(session) - closing_started_count
                >= self.config.farewell_after_turns
            ):
                session.conversation_phase = ConversationPhase.FAREWELL

    def _next_speaker_profile(self, session: SessionState) -> Profile:
        active_turns = [
            turn
            for turn in sorted(session.turns, key=lambda item: item.turn_id)
            if turn.status
            not in {
                TurnStatus.INVALIDATED,
                TurnStatus.IGNORED_AFTER_STOP,
                TurnStatus.ERROR,
            }
        ]
        if not active_turns:
            return self.participants.counselor_profile
        if active_turns[-1].speaker_role == SpeakerRole.COUNSELOR:
            return self.participants.client_profile
        return self.participants.counselor_profile


def invalidate_unplayed_after_keep_limit(
    session: SessionState, *, keep_turns: int, reason: str
) -> list[Turn]:
    invalidated: list[Turn] = []
    unplayed_turns = [
        turn
        for turn in sorted(session.turns, key=lambda item: item.turn_id)
        if is_unplayed_turn(turn)
    ]
    for turn in unplayed_turns[keep_turns:]:
        turn.invalidate(reason)
        invalidated.append(turn)
    return invalidated


def invalidate_unplayed_speaker_sequence_violations(
    session: SessionState,
    *,
    reason: str = "speaker_sequence_violation",
) -> list[Turn]:
    invalidated: list[Turn] = []
    previous_role: SpeakerRole | None = None
    for turn in sorted(session.turns, key=lambda item: item.turn_id):
        if turn.status in {
            TurnStatus.INVALIDATED,
            TurnStatus.IGNORED_AFTER_STOP,
            TurnStatus.ERROR,
        }:
            continue
        if previous_role == turn.speaker_role and is_unplayed_turn(turn):
            turn.invalidate(reason)
            invalidated.append(turn)
            continue
        previous_role = turn.speaker_role
    return invalidated


def _turn_status_for_completed_generation(session_status: SessionStatus) -> TurnStatus:
    if session_status == SessionStatus.PAUSED:
        return TurnStatus.PAUSED_GENERATED
    if session_status in {SessionStatus.STOPPED, SessionStatus.COMPLETED}:
        return TurnStatus.IGNORED_AFTER_STOP
    return TurnStatus.TEXT_READY


def _closing_instruction(conversation_phase: ConversationPhase) -> str | None:
    if conversation_phase != ConversationPhase.CLOSING:
        return None
    return "終結に向けて、急がず自然に今日の話をまとめる方向へ進めてください。"


def _farewell_instruction(
    conversation_phase: ConversationPhase, speaker_profile: Profile
) -> str | None:
    if (
        conversation_phase != ConversationPhase.FAREWELL
        or speaker_profile.role != SpeakerRole.COUNSELOR.value
    ):
        return None
    return "お別れの発話として、短く温かく締めてください。"


def _safety_instruction_for_next_turn(
    session: SessionState, speaker_profile: Profile
) -> str | None:
    if speaker_profile.role != SpeakerRole.COUNSELOR.value:
        return None
    latest_client_turn = _latest_client_turn(session)
    if latest_client_turn is None:
        return None

    revision = latest_client_turn.active_revision()
    moderation_flagged = bool((revision.moderation_result or {}).get("flagged", False))
    if (
        latest_client_turn.warning_level == WarningLevel.HIGH
        or revision.warning_level == WarningLevel.HIGH
        or moderation_flagged
    ):
        return "直前の発話に危機の可能性があります。診断や断定は避け、落ち着いて安全確認を行ってください。"
    return None


def _latest_client_turn(session: SessionState) -> Turn | None:
    for turn in sorted(session.turns, key=lambda item: item.turn_id, reverse=True):
        if turn.speaker_role == SpeakerRole.CLIENT and turn.status not in {
            TurnStatus.INVALIDATED,
            TurnStatus.IGNORED_AFTER_STOP,
            TurnStatus.ERROR,
        }:
            return turn
    return None


def _count_non_invalidated_turns(session: SessionState) -> int:
    return sum(
        1
        for turn in session.turns
        if turn.status
        not in {TurnStatus.INVALIDATED, TurnStatus.IGNORED_AFTER_STOP, TurnStatus.ERROR}
    )
