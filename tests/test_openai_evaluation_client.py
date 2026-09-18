from __future__ import annotations

import pytest

from counseling_voice_demo.models import SessionState, SpeakerRole
from counseling_voice_demo.openai_evaluation_client import OpenAIEvaluationClient, EvaluationRequest
from counseling_voice_demo.openai_text_client import ReplayModeApiCallError


class _FakeResponses:
    def __init__(self) -> None:
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return type("Response", (), {"output_text": "研究デモ用の参考評価です。", "model_dump": lambda self: {}})()


class _FakeOpenAIClient:
    def __init__(self) -> None:
        self.responses = _FakeResponses()


def test_public_evaluation_uses_observable_transcript_only_and_store_false() -> None:
    fake_client = _FakeOpenAIClient()
    session = SessionState(session_id="session_eval_test")
    session.add_turn(
        speaker_role=SpeakerRole.CLIENT,
        speaker_name="高橋",
        profile_id="client_family_default",
        text="家族のことで悩んでいます。",
    )

    result = OpenAIEvaluationClient(client=fake_client).evaluate(
        EvaluationRequest(
            session=session,
            model="gpt-eval-test",
            public=True,
            reasoning_effort="low",
            internal_context={
                "hidden_background": "家族との距離感に悩んでいる",
                "prompt": "内部プロンプト",
            },
        )
    )

    kwargs = fake_client.responses.calls[0]
    assert result.text == "研究デモ用の参考評価です。"
    assert kwargs["store"] is False
    assert kwargs["reasoning"] == {"effort": "low"}
    assert "conversation" not in kwargs
    assert "previous_response_id" not in kwargs
    assert "家族のことで悩んでいます。" in str(kwargs["input"])
    assert "家族との距離感に悩んでいる" not in str(kwargs["input"])
    assert "内部プロンプト" not in str(kwargs["input"])


def test_internal_evaluation_can_include_internal_context() -> None:
    fake_client = _FakeOpenAIClient()
    session = SessionState(session_id="session_eval_internal_test")

    OpenAIEvaluationClient(client=fake_client).evaluate(
        EvaluationRequest(
            session=session,
            model="gpt-eval-test",
            public=False,
            internal_context={"hidden_background": "内部背景"},
        )
    )

    assert "内部背景" in str(fake_client.responses.calls[0]["input"])


def test_evaluation_forbids_api_calls_in_replay_mode() -> None:
    fake_client = _FakeOpenAIClient()

    with pytest.raises(ReplayModeApiCallError):
        OpenAIEvaluationClient(client=fake_client, replay_mode=True).evaluate(
            EvaluationRequest(session=SessionState(session_id="session_eval_replay"), model="gpt-eval-test")
        )

    assert fake_client.responses.calls == []
