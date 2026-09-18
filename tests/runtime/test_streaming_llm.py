from __future__ import annotations

import asyncio

import pytest

from counseling_voice_demo.runtime.streaming_llm import (
    FakeStreamingLLM,
    OpenAIStreamingLLM,
    StreamingLLMError,
    build_responses_input,
    extract_text_delta,
)


class _TypedEvent:
    def __init__(self, *, type: str, delta: str | None = None) -> None:
        self.type = type
        self.delta = delta


class _TypedError:
    def __init__(self, message: str) -> None:
        self.message = message


class _TypedErrorEvent:
    def __init__(self) -> None:
        self.type = "error"
        self.error = _TypedError("stream failed")


class _FakeResponses:
    def __init__(self, events: list[object]) -> None:
        self.events = events
        self.calls: list[dict] = []

    def stream(self, **kwargs):
        self.calls.append(kwargs)
        return _FakeStream(self.events)


class _FakeOpenAIClient:
    def __init__(self, events: list[object]) -> None:
        self.responses = _FakeResponses(events)


class _FakeStream:
    def __init__(self, events: list[object]) -> None:
        self.events = events

    def __enter__(self):
        return iter(self.events)

    def __exit__(self, exc_type, exc, traceback) -> None:
        return None


def test_extract_text_delta_from_dict_event() -> None:
    event = {"type": "response.output_text.delta", "delta": "こんにちは"}

    assert extract_text_delta(event) == "こんにちは"


def test_extract_text_delta_from_typed_like_event() -> None:
    event = _TypedEvent(type="response.output_text.delta", delta="少しずつ")

    assert extract_text_delta(event) == "少しずつ"


def test_extract_text_delta_raises_for_error_event() -> None:
    with pytest.raises(StreamingLLMError, match="stream failed"):
        extract_text_delta(_TypedErrorEvent())


def test_extract_text_delta_raises_for_incomplete_response() -> None:
    event = {
        "type": "response.incomplete",
        "response": {
            "incomplete_details": {"reason": "max_output_tokens"},
        },
    }

    with pytest.raises(
        StreamingLLMError,
        match="response incomplete.*max_output_tokens",
    ):
        extract_text_delta(event)


def test_build_responses_input_keeps_history_and_latest_input() -> None:
    items = build_responses_input(
        "短く返してください。",
        [
            {"role": "user", "content": "前の相談です。"},
            {"role": "assistant", "content": "受け止めました。"},
        ],
        "続きです。",
    )

    assert items == [
        {"role": "system", "content": "短く返してください。"},
        {"role": "user", "content": "前の相談です。"},
        {"role": "assistant", "content": "受け止めました。"},
        {"role": "user", "content": "続きです。"},
    ]


def test_openai_streaming_llm_payload_uses_store_false_and_yields_deltas() -> None:
    async def scenario() -> None:
        client = _FakeOpenAIClient(
            [
                {"type": "response.created"},
                {"type": "response.output_text.delta", "delta": "今日は"},
                _TypedEvent(type="response.output_text.delta", delta="よろしく"),
                {"type": "response.completed"},
                {"type": "response.output_text.delta", "delta": "ignored"},
            ]
        )
        llm = OpenAIStreamingLLM(
            client=client,
            model="gpt-test",
            system_prompt="日本語で短く返してください。",
            reasoning_effort="low",
        )

        parts = []
        async for delta in llm.iter_text(
            latest_input="相談したいです。",
            history=[{"role": "assistant", "content": "どうぞ。"}],
        ):
            parts.append(delta)

        assert parts == ["今日は", "よろしく"]
        payload = client.responses.calls[0]
        assert payload["model"] == "gpt-test"
        assert payload["store"] is False
        assert payload["instructions"] == "日本語で短く返してください。"
        assert payload["reasoning"] == {"effort": "low"}
        assert payload["input"] == [
            {"role": "assistant", "content": "どうぞ。"},
            {"role": "user", "content": "相談したいです。"},
        ]
        assert "conversation" not in payload
        assert "previous_response_id" not in payload

    asyncio.run(scenario())


def test_openai_streaming_llm_payload_supports_structured_text_format() -> None:
    text_format = {
        "type": "json_schema",
        "name": "turn_directive",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {"next": {"type": "string"}},
            "required": ["next"],
            "additionalProperties": False,
        },
    }
    llm = OpenAIStreamingLLM(
        client=_FakeOpenAIClient([]),
        model="gpt-test",
        text_format=text_format,
    )

    payload = llm.build_payload(latest_input="次の方針を決めてください。")

    assert payload["text"] == {"format": text_format}


def test_request_text_format_is_passed_to_api_without_changing_shared_default() -> None:
    async def scenario():
        client = _FakeOpenAIClient(
            [
                {"type": "response.output_text.delta", "delta": "完了"},
                {"type": "response.completed"},
            ]
        )
        default_format = {"type": "json_schema", "name": "default"}
        short_format = {"type": "json_schema", "name": "one_line"}
        long_format = {"type": "json_schema", "name": "three_lines"}
        llm = OpenAIStreamingLLM(
            client=client, model="gpt-test", text_format=default_format
        )

        async def collect(text_format):
            return [
                part
                async for part in llm.stream_text(
                    latest_input="確認", text_format=text_format
                )
            ]

        assert await asyncio.gather(collect(short_format), collect(long_format)) == [
            ["完了"],
            ["完了"],
        ]
        assert [call["text"]["format"] for call in client.responses.calls] == [
            short_format,
            long_format,
        ]
        assert (
            llm.build_payload(latest_input="通常")["text"]["format"] == default_format
        )

    asyncio.run(scenario())


def test_fake_streaming_llm_accepts_latest_input_like_openai_streaming_llm() -> None:
    async def scenario() -> None:
        llm = FakeStreamingLLM(
            response_template="{speaker}:{turn_id}:{input_transcript}:{latest_input}"
        )

        parts = [
            part
            async for part in llm.stream_text(
                latest_input="相談したいです。",
                turn_id=7,
                speaker="client",
            )
        ]

        assert "".join(parts) == "client:7:相談したいです。:相談したいです。"

    asyncio.run(scenario())
