from __future__ import annotations

import pytest

from counseling_voice_demo.clinical_ng_checker import ClinicalNgCheckRequest, ClinicalNgChecker
from counseling_voice_demo.models import WarningLevel
from counseling_voice_demo.openai_text_client import ReplayModeApiCallError


class _FakeResponses:
    def __init__(self, output_text: str) -> None:
        self.output_text = output_text
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return type("Response", (), {"output_text": self.output_text, "model_dump": lambda self: {"id": "resp_eval"}})()


class _FakeOpenAIClient:
    def __init__(self, output_text: str) -> None:
        self.responses = _FakeResponses(output_text)


def test_clinical_ng_checker_uses_llm_json_with_store_false_and_no_stateful_keys() -> None:
    fake_client = _FakeOpenAIClient(
        '{"warning_level":"medium","categories":["diagnosis"],"reason":"診断を断定している","hold_reason":"clinical_ng_medium"}'
    )

    result = ClinicalNgChecker(client=fake_client).check(
        ClinicalNgCheckRequest(
            text="あなたはうつ病です。",
            model="gpt-test",
        )
    )

    kwargs = fake_client.responses.calls[0]
    assert kwargs["store"] is False
    assert "conversation" not in kwargs
    assert "previous_response_id" not in kwargs
    assert result.warning_level == WarningLevel.MEDIUM
    assert result.categories == ["diagnosis"]
    assert result.hold_reason == "clinical_ng_medium"
    assert result.should_stop_session is False


def test_clinical_ng_checker_omits_temperature_for_gpt5_models() -> None:
    fake_client = _FakeOpenAIClient('{"warning_level":"none","categories":[],"reason":"","hold_reason":null}')

    ClinicalNgChecker(client=fake_client).check(
        ClinicalNgCheckRequest(
            text="確認したい発話です。",
            model="gpt-5-nano",
        )
    )

    kwargs = fake_client.responses.calls[0]
    assert "temperature" not in kwargs
    assert kwargs["store"] is False


def test_clinical_ng_checker_parse_failure_is_fail_safe_medium() -> None:
    fake_client = _FakeOpenAIClient("臨床NGはありません")

    result = ClinicalNgChecker(client=fake_client).check(
        ClinicalNgCheckRequest(
            text="確認したい発話です。",
            model="gpt-test",
        )
    )

    assert result.warning_level == WarningLevel.MEDIUM
    assert result.categories == ["parse_error"]
    assert result.hold_reason == "clinical_check_parse_failed"
    assert "JSON解析に失敗" in result.reason


def test_clinical_ng_checker_invalid_warning_level_is_fail_safe_medium() -> None:
    fake_client = _FakeOpenAIClient(
        '{"warning_level":"unknown","categories":[],"reason":"invalid","hold_reason":null}'
    )

    result = ClinicalNgChecker(client=fake_client).check(
        ClinicalNgCheckRequest(
            text="確認したい発話です。",
            model="gpt-test",
        )
    )

    assert result.warning_level == WarningLevel.MEDIUM
    assert result.hold_reason == "clinical_check_parse_failed"


def test_clinical_ng_checker_forbids_api_calls_in_replay_mode() -> None:
    fake_client = _FakeOpenAIClient("{}")

    with pytest.raises(ReplayModeApiCallError):
        ClinicalNgChecker(client=fake_client, replay_mode=True).check(
            ClinicalNgCheckRequest(text="確認したい発話です。", model="gpt-test")
        )

    assert fake_client.responses.calls == []
