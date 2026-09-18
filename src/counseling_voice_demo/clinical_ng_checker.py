from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from counseling_voice_demo.models import WarningLevel
from counseling_voice_demo.openai_text_client import (
    ReplayModeApiCallError,
    _call_with_retries,
    _extract_output_text,
    _model_dump,
    supports_temperature_parameter,
)


CLINICAL_CHECK_PARSE_FAILED_HOLD_REASON = "clinical_check_parse_failed"
_PARSE_FAILURE_REASON_PREFIX = "臨床NGチェックのJSON解析に失敗しました"


@dataclass(frozen=True)
class ClinicalNgCheckRequest:
    text: str
    model: str


@dataclass(frozen=True)
class ClinicalNgCheckResult:
    warning_level: WarningLevel
    categories: list[str]
    reason: str
    hold_reason: str | None
    raw_response: dict[str, Any]
    should_stop_session: bool = False


class ClinicalNgChecker:
    def __init__(self, *, client: Any, max_retries: int = 3, replay_mode: bool = False) -> None:
        self._client = client
        self._max_retries = max_retries
        self._replay_mode = replay_mode

    def check(self, request: ClinicalNgCheckRequest) -> ClinicalNgCheckResult:
        if self._replay_mode:
            raise ReplayModeApiCallError("Replay modeではOpenAI APIを呼び出せません")

        payload = {
            "model": request.model,
            "input": _build_clinical_ng_input(request.text),
            "store": False,
        }
        if supports_temperature_parameter(request.model):
            payload["temperature"] = 0
        response = _call_with_retries(
            lambda: self._client.responses.create(**payload),
            max_retries=self._max_retries,
        )
        parsed = _parse_json_object(_extract_output_text(response))
        warning_level_value = parsed.get("warning_level")
        warning_level = _parse_warning_level(parsed.get("warning_level"))
        hold_reason = (
            parsed.get("hold_reason")
            if isinstance(parsed.get("hold_reason"), str)
            else None
        )
        if not _is_valid_warning_level(warning_level_value) and hold_reason is None:
            hold_reason = CLINICAL_CHECK_PARSE_FAILED_HOLD_REASON
        return ClinicalNgCheckResult(
            warning_level=warning_level,
            categories=[str(item) for item in parsed.get("categories", []) if isinstance(item, str)],
            reason=str(parsed.get("reason", "")),
            hold_reason=hold_reason,
            raw_response=_model_dump(response),
        )


def _build_clinical_ng_input(text: str) -> list[dict[str, Any]]:
    instruction = (
        "カウンセリングデモの発話を臨床NG観点で評価し、JSONだけを返してください。"
        "対象カテゴリは診断断定、治療方針断定、服薬指示、就業判定、法的判断、緊急時放置、"
        "自律性を損なう指示、説教的助言です。"
        "schema: {\"warning_level\":\"none|low|medium|high\",\"categories\":[],\"reason\":\"\",\"hold_reason\":null}"
    )
    return [
        {"role": "system", "content": [{"type": "input_text", "text": instruction}]},
        {"role": "user", "content": [{"type": "input_text", "text": text}]},
    ]


def _parse_json_object(text: str) -> dict[str, Any]:
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return _parse_failure_result(text)
    if not isinstance(parsed, dict):
        return _parse_failure_result(text)
    return parsed


def _parse_warning_level(value: Any) -> WarningLevel:
    try:
        return WarningLevel(str(value))
    except ValueError:
        return WarningLevel.MEDIUM


def _is_valid_warning_level(value: Any) -> bool:
    try:
        WarningLevel(str(value))
    except ValueError:
        return False
    return True


def _parse_failure_result(text: str) -> dict[str, Any]:
    snippet = str(text).strip().replace("\n", " ")[:160]
    reason = _PARSE_FAILURE_REASON_PREFIX
    if snippet:
        reason = f"{reason}: {snippet}"
    return {
        "warning_level": WarningLevel.MEDIUM.value,
        "categories": ["parse_error"],
        "reason": reason,
        "hold_reason": CLINICAL_CHECK_PARSE_FAILED_HOLD_REASON,
    }
