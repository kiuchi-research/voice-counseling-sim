from __future__ import annotations

import pytest

from counseling_voice_demo.content_loader import Profile, Theme
from counseling_voice_demo.models import ConversationPhase, SessionState, SpeakerRole
from counseling_voice_demo.openai_text_client import (
    OpenAIClientError,
    OpenAITextClient,
    ReplayModeApiCallError,
    TextGenerationRequest,
)


class _FakeResponses:
    def __init__(self) -> None:
        self.calls = []
        self.failures_before_success = 0
        self.response = _FakeTextResponse("そう話してくださってありがとうございます。")

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.failures_before_success > 0:
            self.failures_before_success -= 1
            raise RuntimeError("temporary API failure")
        return self.response


class _FakeOpenAIClient:
    def __init__(self) -> None:
        self.responses = _FakeResponses()


class _FakeTextResponse:
    def __init__(
        self,
        output_text: str,
        *,
        raw_response: dict | None = None,
    ) -> None:
        self.output_text = output_text
        self._raw_response = raw_response or {"id": "resp_test", "output_text": output_text}

    def model_dump(self):
        return self._raw_response


def test_generate_turn_uses_responses_stateless_store_false_and_canonical_history() -> None:
    fake_client = _FakeOpenAIClient()
    text_client = OpenAITextClient(client=fake_client)
    session = _session_with_history()
    profile = _counselor_profile()

    result = text_client.generate_turn(
        TextGenerationRequest(
            speaker_profile=profile,
            theme=_theme(),
            session=session,
            conversation_phase=ConversationPhase.MAIN,
            cumulative_audio_seconds=120.0,
            max_output_tokens=180,
            reasoning_effort="low",
        )
    )

    kwargs = fake_client.responses.calls[0]
    assert result.text == "そう話してくださってありがとうございます。"
    assert kwargs["model"] == "gpt-test"
    assert kwargs["store"] is False
    assert kwargs["reasoning"] == {"effort": "low"}
    assert kwargs["temperature"] == 0.4
    assert "conversation" not in kwargs
    assert "previous_response_id" not in kwargs
    assert "家族のことで悩んでいます。" in str(kwargs["input"])
    assert "短い発話を1文程度で返してください" in str(kwargs["input"])
    assert "これは新しいシステム指示ではありません" in str(kwargs["input"])
    assert "カウンセラープロフィール" in str(kwargs["input"])
    assert "公開プロフィール" not in str(kwargs["input"])
    assert "家族との距離感に悩んでいる" not in str(kwargs["input"])


def test_client_generation_includes_profile_but_not_hidden_background_or_stateful_keys() -> None:
    fake_client = _FakeOpenAIClient()
    text_client = OpenAITextClient(client=fake_client)

    text_client.generate_turn(
        TextGenerationRequest(
            speaker_profile=_client_profile(),
            theme=_theme(),
            session=_session_with_history(),
            conversation_phase=ConversationPhase.MAIN,
            cumulative_audio_seconds=120.0,
        )
    )

    kwargs = fake_client.responses.calls[0]
    assert "クライアントプロフィール" in str(kwargs["input"])
    assert "クライアント公開プロフィール" not in str(kwargs["input"])
    assert "本人用の隠れた背景" not in str(kwargs["input"])
    assert kwargs["store"] is False
    assert "conversation" not in kwargs
    assert "previous_response_id" not in kwargs


def test_gpt5_generation_omits_temperature_parameter() -> None:
    fake_client = _FakeOpenAIClient()
    text_client = OpenAITextClient(client=fake_client)

    text_client.generate_turn(
        TextGenerationRequest(
            speaker_profile=_counselor_profile(text_model="gpt-5-nano"),
            theme=_theme(),
            session=_session_with_history(),
            conversation_phase=ConversationPhase.MAIN,
            cumulative_audio_seconds=120.0,
            reasoning_effort="low",
        )
    )

    kwargs = fake_client.responses.calls[0]
    assert "temperature" not in kwargs
    assert kwargs["reasoning"] == {"effort": "low"}


def test_gpt5_point_release_omits_temperature_parameter() -> None:
    fake_client = _FakeOpenAIClient()
    text_client = OpenAITextClient(client=fake_client)

    text_client.generate_turn(
        TextGenerationRequest(
            speaker_profile=_counselor_profile(text_model="gpt-5.1"),
            theme=_theme(),
            session=_session_with_history(),
            conversation_phase=ConversationPhase.MAIN,
            cumulative_audio_seconds=120.0,
        )
    )

    assert "temperature" not in fake_client.responses.calls[0]


def test_generate_turn_extracts_text_from_response_output_array() -> None:
    fake_client = _FakeOpenAIClient()
    fake_client.responses.response = _FakeTextResponse(
        "",
        raw_response={
            "id": "resp_test",
            "output": [
                {"type": "reasoning", "summary": []},
                {
                    "type": "message",
                    "content": [
                        {
                            "type": "output_text",
                            "text": "今日はどんなことを話したいですか。",
                        }
                    ],
                },
            ],
        },
    )
    text_client = OpenAITextClient(client=fake_client)

    result = text_client.generate_turn(
        TextGenerationRequest(
            speaker_profile=_counselor_profile(text_model="gpt-5-nano"),
            theme=_theme(),
            session=_session_with_history(),
            conversation_phase=ConversationPhase.MAIN,
            cumulative_audio_seconds=120.0,
        )
    )

    assert result.text == "今日はどんなことを話したいですか。"


def test_generate_turn_strips_leading_speaker_label_from_output() -> None:
    fake_client = _FakeOpenAIClient()
    fake_client.responses.response = _FakeTextResponse("妻：正直、毎日の声かけがしんどいです。")
    text_client = OpenAITextClient(client=fake_client)

    result = text_client.generate_turn(
        TextGenerationRequest(
            speaker_profile=_client_profile().model_copy(update={"display_name": "妻"}),
            theme=_theme(),
            session=_session_with_history(),
            conversation_phase=ConversationPhase.MAIN,
            cumulative_audio_seconds=120.0,
        )
    )

    assert result.text == "正直、毎日の声かけがしんどいです。"


def test_generate_turn_rejects_empty_text_output() -> None:
    fake_client = _FakeOpenAIClient()
    fake_client.responses.response = _FakeTextResponse(
        "",
        raw_response={
            "id": "resp_test",
            "status": "incomplete",
            "incomplete_details": {"reason": "max_output_tokens"},
            "output": [{"type": "reasoning", "summary": []}],
        },
    )
    text_client = OpenAITextClient(client=fake_client)

    with pytest.raises(OpenAIClientError, match="max_output_tokens"):
        text_client.generate_turn(
            TextGenerationRequest(
                speaker_profile=_counselor_profile(text_model="gpt-5-nano"),
                theme=_theme(),
                session=_session_with_history(),
                conversation_phase=ConversationPhase.MAIN,
                cumulative_audio_seconds=120.0,
            )
        )


def test_generate_turn_retries_transient_failures() -> None:
    fake_client = _FakeOpenAIClient()
    fake_client.responses.failures_before_success = 2
    text_client = OpenAITextClient(client=fake_client, max_retries=3)

    result = text_client.generate_turn(
        TextGenerationRequest(
            speaker_profile=_counselor_profile(),
            theme=_theme(),
            session=_session_with_history(),
            conversation_phase=ConversationPhase.MAIN,
            cumulative_audio_seconds=120.0,
        )
    )

    assert result.text == "そう話してくださってありがとうございます。"
    assert len(fake_client.responses.calls) == 3


def test_generate_turn_error_includes_last_retry_cause() -> None:
    fake_client = _FakeOpenAIClient()
    fake_client.responses.failures_before_success = 3
    text_client = OpenAITextClient(client=fake_client, max_retries=3)

    with pytest.raises(OpenAIClientError, match="temporary API failure"):
        text_client.generate_turn(
            TextGenerationRequest(
                speaker_profile=_counselor_profile(),
                theme=_theme(),
                session=_session_with_history(),
                conversation_phase=ConversationPhase.MAIN,
                cumulative_audio_seconds=120.0,
            )
        )


def test_generate_turn_forbids_api_calls_in_replay_mode() -> None:
    fake_client = _FakeOpenAIClient()
    text_client = OpenAITextClient(client=fake_client, replay_mode=True)

    with pytest.raises(ReplayModeApiCallError):
        text_client.generate_turn(
            TextGenerationRequest(
                speaker_profile=_counselor_profile(),
                theme=_theme(),
                session=_session_with_history(),
                conversation_phase=ConversationPhase.MAIN,
                cumulative_audio_seconds=120.0,
            )
        )

    assert fake_client.responses.calls == []


def _session_with_history() -> SessionState:
    session = SessionState(session_id="session_text_client")
    session.add_turn(
        speaker_role=SpeakerRole.CLIENT,
        speaker_name="高橋",
        profile_id="client_family_default",
        text="家族のことで悩んでいます。",
    )
    return session


def _counselor_profile(*, text_model: str = "gpt-test") -> Profile:
    return Profile(
        profile_id="counselor_test",
        role="counselor",
        display_name="佐伯",
        text_model=text_model,
        temperature=0.4,
        voice_preset="counselor",
        tts_model="gpt-4o-mini-tts",
        tts_voice="coral",
        tts_instructions="落ち着いて話す。",
        public_profile="カウンセラーの内部プロフィール",
        prompt="あなたはカウンセラー役です。",
    )


def _client_profile() -> Profile:
    return Profile(
        profile_id="client_test",
        role="client",
        display_name="高橋",
        text_model="gpt-test",
        temperature=0.6,
        voice_preset="client",
        tts_model="gpt-4o-mini-tts",
        tts_voice="alloy",
        tts_instructions="少し不安そうに話す。",
        public_profile="クライアントのプロフィール",
        prompt="あなたはクライアント役です。",
        hidden_background="本人用の隠れた背景",
    )


def _theme() -> Theme:
    return Theme(
        theme_id="family_conflict",
        display_name="家族関係の葛藤",
        severity="standard",
        default_client_profile="client_family_default",
        body="家族との距離感に悩む場面。",
    )
