from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from counseling_voice_demo.log_writer import SessionLogPaths
from counseling_voice_demo.models import SessionState, Turn, TurnRevision, utc_now
from counseling_voice_demo.openai_tts_client import TtsRequest
from counseling_voice_demo.safety_checks import SafetyCheckConfig, run_safety_checks_for_turn
from counseling_voice_demo.tts_queue import mark_tts_ready
from counseling_voice_demo.turn_editing import (
    invalidate_from_turn,
    replace_with_regenerated_turn,
)


@dataclass(frozen=True)
class SingleTurnRegenerationResult:
    turn: Turn
    revision: TurnRevision
    invalidated_following_turns: list[Turn]
    generated_following_turns: list[Turn]


def regenerate_single_turn(
    session: SessionState,
    *,
    turn_id: int,
    conversation_engine: object,
    cumulative_audio_seconds: float,
    moderation_client: object | None = None,
    moderation_model: str | None = None,
    clinical_ng_checker: object | None = None,
    clinical_model: str | None = None,
    tts_client: object | None = None,
    log_paths: SessionLogPaths | None = None,
    tts_response_format: str = "mp3",
) -> SingleTurnRegenerationResult:
    target_turn = _find_turn(session, turn_id=turn_id)
    speaker_profile = conversation_engine.speaker_profile_for_role(
        target_turn.speaker_role
    )
    context_session = deepcopy(session)
    invalidate_from_turn(
        context_session,
        turn_id=turn_id,
        reason="single_regenerate_context_exclusion",
    )
    request = conversation_engine.build_generation_request(
        context_session,
        speaker_profile=speaker_profile,
        cumulative_audio_seconds=cumulative_audio_seconds,
    )
    generation_result = conversation_engine.text_client.generate_turn(request)
    edit_result = replace_with_regenerated_turn(
        session,
        turn_id=turn_id,
        regenerated_text=generation_result.text,
        log_paths=log_paths,
    )

    _run_post_generation_checks_and_tts(
        edit_result.turn,
        speaker_profile=speaker_profile,
        moderation_client=moderation_client,
        moderation_model=moderation_model,
        clinical_ng_checker=clinical_ng_checker,
        clinical_model=clinical_model,
        tts_client=tts_client,
        log_paths=log_paths,
        tts_response_format=tts_response_format,
    )
    generated_following_turns = conversation_engine.fill_generation_queue(
        session,
        cumulative_audio_seconds=cumulative_audio_seconds,
    )
    return SingleTurnRegenerationResult(
        turn=edit_result.turn,
        revision=edit_result.revision,
        invalidated_following_turns=edit_result.invalidated_following_turns,
        generated_following_turns=generated_following_turns,
    )


def _run_post_generation_checks_and_tts(
    turn: Turn,
    *,
    speaker_profile: Any,
    moderation_client: object | None,
    moderation_model: str | None,
    clinical_ng_checker: object | None,
    clinical_model: str | None,
    tts_client: object | None,
    log_paths: SessionLogPaths | None,
    tts_response_format: str,
) -> None:
    revision = turn.active_revision()

    if moderation_model is not None and clinical_model is not None:
        run_safety_checks_for_turn(
            turn,
            moderation_client=moderation_client,
            clinical_ng_checker=clinical_ng_checker,
            config=SafetyCheckConfig(
                moderation_model=moderation_model,
                clinical_model=clinical_model,
            ),
            log_paths=log_paths,
        )

    if tts_client is not None and log_paths is not None:
        tts_client.synthesize_turn(
            TtsRequest(
                turn=turn,
                log_paths=log_paths,
                model=speaker_profile.tts_model,
                voice=speaker_profile.tts_voice,
                instructions=speaker_profile.tts_instructions,
                response_format=tts_response_format,
            )
        )
        revision.tts_generated_at = utc_now()
        mark_tts_ready(turn)


def _find_turn(session: SessionState, *, turn_id: int) -> Turn:
    for turn in session.turns:
        if turn.turn_id == turn_id:
            return turn
    raise ValueError(f"turn_id={turn_id} のターンが見つかりません")
