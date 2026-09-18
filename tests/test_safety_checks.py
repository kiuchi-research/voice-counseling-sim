from __future__ import annotations

import json

from counseling_voice_demo.clinical_ng_checker import ClinicalNgCheckRequest
from counseling_voice_demo.log_writer import create_session_log_dirs
from counseling_voice_demo.models import SessionState, SpeakerRole, TurnStatus, WarningLevel
from counseling_voice_demo.safety_checks import SafetyCheckConfig, run_safety_checks_for_turn


def test_safety_checks_log_moderation_and_clinical_results_without_holding_medium_clinical_ng(tmp_path) -> None:
    paths = create_session_log_dirs(tmp_path, "session_safety_test")
    session = SessionState(session_id="session_safety_test")
    turn = session.add_turn(
        speaker_role=SpeakerRole.COUNSELOR,
        speaker_name="佐伯",
        profile_id="counselor",
        text="診断を断定する発話です。",
        status=TurnStatus.TEXT_READY,
    )
    moderation_client = _FakeModerationClient(flagged=False)
    clinical_checker = _FakeClinicalChecker(
        warning_level=WarningLevel.MEDIUM,
        hold_reason="clinical_ng_medium",
    )

    result = run_safety_checks_for_turn(
        turn,
        moderation_client=moderation_client,
        clinical_ng_checker=clinical_checker,
        config=SafetyCheckConfig(
            moderation_model="omni-moderation-test",
            clinical_model="gpt-clinical-test",
        ),
        log_paths=paths,
    )

    assert result.moderation_checked is True
    assert result.clinical_checked is True
    assert moderation_client.calls == [("診断を断定する発話です。", "omni-moderation-test")]
    assert clinical_checker.calls == [
        ClinicalNgCheckRequest(text="診断を断定する発話です。", model="gpt-clinical-test")
    ]
    assert turn.warning_level == WarningLevel.MEDIUM
    assert result.playback_hold is False
    assert turn.status == TurnStatus.CLINICAL_CHECKED
    assert turn.playback_status == TurnStatus.TEXT_READY
    assert turn.active_revision().hold_reason == "clinical_ng_medium"

    safety_logs = _read_jsonl(paths.safety_log_jsonl)
    clinical_logs = _read_jsonl(paths.clinical_ng_log_jsonl)
    assert safety_logs[0]["turn_id"] == turn.turn_id
    assert safety_logs[0]["flagged"] is False
    assert clinical_logs[0]["turn_id"] == turn.turn_id
    assert clinical_logs[0]["warning_level"] == WarningLevel.MEDIUM.value


def test_moderation_flag_sets_high_warning_without_stopping_session(tmp_path) -> None:
    paths = create_session_log_dirs(tmp_path, "session_moderation_test")
    session = SessionState(session_id="session_moderation_test")
    turn = session.add_turn(
        speaker_role=SpeakerRole.CLIENT,
        speaker_name="高橋",
        profile_id="client",
        text="もう消えてしまいたいです。",
        status=TurnStatus.TEXT_READY,
    )

    run_safety_checks_for_turn(
        turn,
        moderation_client=_FakeModerationClient(flagged=True),
        clinical_ng_checker=_FakeClinicalChecker(
            warning_level=WarningLevel.LOW,
            hold_reason=None,
        ),
        config=SafetyCheckConfig(
            moderation_model="omni-moderation-test",
            clinical_model="gpt-clinical-test",
        ),
        log_paths=paths,
    )

    assert session.status.value == "idle"
    assert turn.warning_level == WarningLevel.HIGH
    assert turn.status == TurnStatus.PLAYBACK_HOLD
    assert turn.active_revision().moderation_result["flagged"] is True
    assert turn.active_revision().hold_reason == "moderation_flagged"


def test_low_warning_keeps_turn_ready_for_tts_or_playback() -> None:
    session = SessionState(session_id="session_low_warning_test")
    turn = session.add_turn(
        speaker_role=SpeakerRole.CLIENT,
        speaker_name="高橋",
        profile_id="client",
        text="軽い警告です。",
        status=TurnStatus.TEXT_READY,
    )

    run_safety_checks_for_turn(
        turn,
        moderation_client=_FakeModerationClient(flagged=False),
        clinical_ng_checker=_FakeClinicalChecker(
            warning_level=WarningLevel.LOW,
            hold_reason=None,
        ),
        config=SafetyCheckConfig(
            moderation_model="omni-moderation-test",
            clinical_model="gpt-clinical-test",
        ),
    )

    assert turn.warning_level == WarningLevel.LOW
    assert turn.status == TurnStatus.CLINICAL_CHECKED
    assert turn.playback_status == TurnStatus.TEXT_READY


class _FakeModerationClient:
    def __init__(self, *, flagged: bool) -> None:
        self._flagged = flagged
        self.calls = []

    def moderate(self, *, text: str, model: str):
        self.calls.append((text, model))
        return type(
            "ModerationResult",
            (),
            {
                "flagged": self._flagged,
                "raw": {"model": model, "flagged": self._flagged},
            },
        )()


class _FakeClinicalChecker:
    def __init__(self, *, warning_level: WarningLevel, hold_reason: str | None) -> None:
        self._warning_level = warning_level
        self._hold_reason = hold_reason
        self.calls = []

    def check(self, request):
        self.calls.append(request)
        return type(
            "ClinicalNgCheckResult",
            (),
            {
                "warning_level": self._warning_level,
                "categories": ["diagnosis"] if self._warning_level != WarningLevel.NONE else [],
                "reason": "確認が必要です。",
                "hold_reason": self._hold_reason,
                "raw_response": {"ok": True},
            },
        )()


def _read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
