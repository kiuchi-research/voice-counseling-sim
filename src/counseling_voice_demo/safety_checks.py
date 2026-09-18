from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from counseling_voice_demo.clinical_ng_checker import ClinicalNgCheckRequest
from counseling_voice_demo.log_writer import SessionLogPaths, append_clinical_ng_log, append_safety_log
from counseling_voice_demo.models import Turn, TurnStatus, WarningLevel, utc_now
from counseling_voice_demo.warning_policy import apply_playback_hold_if_needed, apply_warning


@dataclass(frozen=True)
class SafetyCheckConfig:
    moderation_model: str
    clinical_model: str


@dataclass(frozen=True)
class SafetyCheckResult:
    moderation_checked: bool
    clinical_checked: bool
    moderation_flagged: bool
    warning_level: WarningLevel
    playback_hold: bool


@dataclass
class SafetyCheckRunner:
    moderation_client: object
    clinical_ng_checker: object
    config: SafetyCheckConfig
    log_paths: SessionLogPaths | None = None

    def check_turn(self, turn: Turn) -> SafetyCheckResult:
        return run_safety_checks_for_turn(
            turn,
            moderation_client=self.moderation_client,
            clinical_ng_checker=self.clinical_ng_checker,
            config=self.config,
            log_paths=self.log_paths,
        )


def run_safety_checks_for_turn(
    turn: Turn,
    *,
    moderation_client: object | None,
    clinical_ng_checker: object | None,
    config: SafetyCheckConfig,
    log_paths: SessionLogPaths | None = None,
) -> SafetyCheckResult:
    revision = turn.active_revision()
    text = revision.canonical_text
    moderation_flagged = False
    moderation_checked = False
    clinical_checked = False

    if moderation_client is not None:
        moderation_checked = True
        moderation_result = moderation_client.moderate(text=text, model=config.moderation_model)
        moderation_flagged = bool(moderation_result.flagged)
        revision.moderation_result = {
            "flagged": moderation_flagged,
            "raw": moderation_result.raw,
        }
        if moderation_flagged:
            apply_warning(turn, WarningLevel.HIGH, hold_reason="moderation_flagged")
        turn.status = TurnStatus.SAFETY_CHECKED
        if log_paths is not None:
            append_safety_log(log_paths, _moderation_log_record(turn, moderation_result))

    if clinical_ng_checker is not None:
        clinical_checked = True
        clinical_result = clinical_ng_checker.check(
            ClinicalNgCheckRequest(text=text, model=config.clinical_model)
        )
        revision.clinical_ng_result = {
            "warning_level": clinical_result.warning_level.value,
            "categories": clinical_result.categories,
            "reason": clinical_result.reason,
            "hold_reason": clinical_result.hold_reason,
            "raw_response": clinical_result.raw_response,
        }
        apply_warning(
            turn,
            clinical_result.warning_level,
            hold_reason=clinical_result.hold_reason,
        )
        turn.status = TurnStatus.CLINICAL_CHECKED
        if log_paths is not None:
            append_clinical_ng_log(log_paths, _clinical_log_record(turn, clinical_result))

    playback_hold = apply_playback_hold_if_needed(turn)
    return SafetyCheckResult(
        moderation_checked=moderation_checked,
        clinical_checked=clinical_checked,
        moderation_flagged=moderation_flagged,
        warning_level=turn.warning_level,
        playback_hold=playback_hold,
    )


def _moderation_log_record(turn: Turn, moderation_result: Any) -> dict[str, Any]:
    revision = turn.active_revision()
    return {
        "recorded_at": utc_now(),
        "session_id": turn.session_id,
        "turn_id": turn.turn_id,
        "revision_id": revision.revision_id,
        "speaker_role": turn.speaker_role.value,
        "flagged": bool(moderation_result.flagged),
        "raw": moderation_result.raw,
    }


def _clinical_log_record(turn: Turn, clinical_result: Any) -> dict[str, Any]:
    revision = turn.active_revision()
    return {
        "recorded_at": utc_now(),
        "session_id": turn.session_id,
        "turn_id": turn.turn_id,
        "revision_id": revision.revision_id,
        "speaker_role": turn.speaker_role.value,
        "warning_level": clinical_result.warning_level.value,
        "categories": clinical_result.categories,
        "reason": clinical_result.reason,
        "hold_reason": clinical_result.hold_reason,
        "raw_response": clinical_result.raw_response,
    }
