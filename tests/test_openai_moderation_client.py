from __future__ import annotations

import pytest

from counseling_voice_demo.openai_moderation_client import OpenAIModerationClient
from counseling_voice_demo.openai_text_client import ReplayModeApiCallError


class _FakeModerations:
    def __init__(self) -> None:
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return _FakeModerationResponse(flagged=True)


class _FakeOpenAIClient:
    def __init__(self) -> None:
        self.moderations = _FakeModerations()


class _FakeModerationResponse:
    def __init__(self, flagged: bool) -> None:
        self.results = [type("Result", (), {"flagged": flagged})()]

    def model_dump(self):
        return {"id": "modr_test", "results": [{"flagged": True}]}


def test_moderation_client_returns_loggable_result_without_auto_stop() -> None:
    fake_client = _FakeOpenAIClient()

    result = OpenAIModerationClient(client=fake_client).moderate(
        text="確認したい発話です。",
        model="omni-moderation-latest",
    )

    assert fake_client.moderations.calls == [
        {"input": "確認したい発話です。", "model": "omni-moderation-latest"}
    ]
    assert result.flagged is True
    assert result.should_stop_session is False
    assert result.raw["id"] == "modr_test"


def test_moderation_client_forbids_api_calls_in_replay_mode() -> None:
    fake_client = _FakeOpenAIClient()

    with pytest.raises(ReplayModeApiCallError):
        OpenAIModerationClient(client=fake_client, replay_mode=True).moderate(
            text="確認したい発話です。",
            model="omni-moderation-latest",
        )

    assert fake_client.moderations.calls == []
