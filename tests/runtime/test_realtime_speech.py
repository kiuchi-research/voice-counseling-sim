from __future__ import annotations

import asyncio
import base64
import json
from contextlib import asynccontextmanager
from typing import Any

import pytest

from counseling_voice_demo.runtime.models import AudioDeliveryMode
from counseling_voice_demo.runtime.prompt_director import (
    PromptDirectorResult,
    render_prompt_director_instruction,
)
from counseling_voice_demo.runtime.realtime_speech import (
    OpenAIRealtimeSpeechAgent,
    RealtimePlaybackStopResult,
    RealtimeSpeechConfig,
    RealtimeSpeechError,
    RealtimeSpeechSession,
    build_realtime_conversation_item_truncate,
    build_realtime_input_audio_buffer_append,
    build_realtime_input_audio_buffer_commit,
    build_realtime_response_create,
    build_realtime_response_cancel,
    build_realtime_speech_session_update,
    build_realtime_text_conversation_item,
)


class _FakeTransport:
    def __init__(self, events: list[dict[str, Any]]) -> None:
        self.events = list(events)
        self.sent: list[dict[str, Any]] = []

    def send_json(self, event: dict[str, Any]) -> None:
        self.sent.append(event)

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self.events:
            raise StopAsyncIteration
        return self.events.pop(0)


class _SessionDurationClosedTransport(_FakeTransport):
    async def __anext__(self):
        raise RuntimeError(
            "received 1001 (going away) Your session hit the maximum duration "
            "of 60 minutes.; then sent 1001 (going away) Your session hit the "
            "maximum duration of 60 minutes."
        )


class _AbruptlyClosedTransport(_FakeTransport):
    async def __anext__(self):
        raise RuntimeError("no close frame received or sent")


async def _collect_realtime_events(stream):
    return [event async for event in stream]


def test_reused_connection_sends_only_current_snapshot_and_logs_exact_input() -> None:
    async def scenario():
        transport = _FakeTransport([{"type": "response.done"}] * 2)

        @asynccontextmanager
        async def transport_context():
            yield transport

        snapshots = {
            1: [
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "先行生成時の仮の入力"}],
                }
            ],
            2: [
                {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": "全員の確定発話から作った現在の入力",
                        }
                    ],
                }
            ],
        }
        agent = OpenAIRealtimeSpeechAgent(
            speaker="client_b",
            transport_context_factory=transport_context,
            response_input_formatter=lambda **kwargs: snapshots[kwargs["turn_id"]],
        )
        records = []
        for turn_id in (1, 2):
            await _collect_realtime_events(
                agent.stream_audio_response(
                    session_id="snapshot_test",
                    turn_id=turn_id,
                    speaker="client_b",
                    input_transcript="直前発話",
                    sample_rate=24000,
                    sample_width_bits=16,
                    channels=1,
                    delivery_mode=AudioDeliveryMode.REAL_TIME,
                    response_instructions_observer=records.append,
                )
            )
        assert [event["type"] for event in transport.sent] == [
            "session.update",
            "response.create",
            "response.create",
        ]
        for turn_id, event, record in zip(
            (1, 2), transport.sent[1:], records, strict=True
        ):
            assert event["response"]["input"] == snapshots[turn_id]
            assert record.response_input == event["response"]["input"]
            # Outputs remain in the default conversation so interruption/truncate works.
            assert event["response"].get("conversation", "auto") == "auto"
        await agent.close()

    asyncio.run(scenario())


def test_build_realtime_speech_session_update_uses_ga_realtime_audio_shape() -> None:
    event = build_realtime_speech_session_update(
        model="gpt-realtime-mini",
        instructions="短く返してください。",
        voice="coral",
        output_speed=0.9,
    )

    assert event["type"] == "session.update"
    session = event["session"]
    assert session["type"] == "realtime"
    assert session["model"] == "gpt-realtime-mini"
    assert session["output_modalities"] == ["audio"]
    assert session["tools"] == []
    assert session["max_output_tokens"] == "inf"
    assert session["instructions"] == "短く返してください。"
    assert session["audio"]["output"] == {
        "format": {"type": "audio/pcm", "rate": 24000},
        "voice": "coral",
        "speed": 0.9,
    }
    assert session["audio"]["input"]["format"] == {"type": "audio/pcm", "rate": 24000}
    assert session["audio"]["input"]["transcription"] == {
        "model": "gpt-realtime-whisper",
    }
    assert session["audio"]["input"]["noise_reduction"] == {"type": "far_field"}
    assert session["audio"]["input"]["turn_detection"] == {
        "type": "server_vad",
        "threshold": 0.5,
        "prefix_padding_ms": 300,
        "silence_duration_ms": 500,
        "idle_timeout_ms": None,
    }


def test_build_realtime_text_conversation_item_uses_input_text_content() -> None:
    event = build_realtime_text_conversation_item("前の発話です。")

    assert event == {
        "type": "conversation.item.create",
        "item": {
            "type": "message",
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": "前の発話です。",
                }
            ],
        },
    }


def test_build_realtime_input_audio_buffer_events_encode_pcm() -> None:
    pcm = b"\x01\x00\x02\x00"

    assert build_realtime_input_audio_buffer_append(pcm) == {
        "type": "input_audio_buffer.append",
        "audio": base64.b64encode(pcm).decode("ascii"),
    }
    assert build_realtime_input_audio_buffer_commit() == {
        "type": "input_audio_buffer.commit"
    }


def test_build_realtime_response_create_can_request_audio_output() -> None:
    event = build_realtime_response_create(
        instructions="一文で。",
        voice="cedar",
        output_format={"type": "audio/pcm", "rate": 24000},
        output_modalities=("audio",),
    )

    assert event == {
        "type": "response.create",
        "response": {
            "output_modalities": ["audio"],
            "instructions": "一文で。",
            "audio": {
                "output": {
                    "format": {"type": "audio/pcm", "rate": 24000},
                    "voice": "cedar",
                }
            },
        },
    }


def test_build_realtime_response_cancel_can_target_response() -> None:
    assert build_realtime_response_cancel() == {"type": "response.cancel"}
    assert build_realtime_response_cancel(response_id="resp_123") == {
        "type": "response.cancel",
        "response_id": "resp_123",
    }


def test_build_realtime_conversation_item_truncate_uses_played_ms() -> None:
    event = build_realtime_conversation_item_truncate(
        item_id="item_123",
        played_ms=1500,
        content_index=0,
    )

    assert event == {
        "type": "conversation.item.truncate",
        "item_id": "item_123",
        "content_index": 0,
        "audio_end_ms": 1500,
    }


def test_realtime_speech_session_sends_cancel_and_truncate_for_played_audio() -> None:
    async def scenario() -> None:
        transport = _FakeTransport([])
        session = RealtimeSpeechSession(transport)

        result = await session.stop_current_response_playback(
            played_ms=1250,
            item_id="item_123",
            response_id="resp_123",
        )

        assert transport.sent == [
            {
                "type": "response.cancel",
                "response_id": "resp_123",
            },
            {
                "type": "conversation.item.truncate",
                "item_id": "item_123",
                "content_index": 0,
                "audio_end_ms": 1250,
            },
        ]
        assert result == RealtimePlaybackStopResult(
            played_ms=1250,
            audio_end_ms=1250,
            cancel_sent=True,
            truncate_sent=True,
            item_id="item_123",
            response_id="resp_123",
            content_index=0,
            sent_event_types=("response.cancel", "conversation.item.truncate"),
        )

    asyncio.run(scenario())


def test_realtime_speech_session_reports_missing_truncate_item() -> None:
    async def scenario() -> None:
        transport = _FakeTransport([])
        session = RealtimeSpeechSession(transport)

        result = await session.stop_current_response_playback(played_ms=600)

        assert transport.sent == [{"type": "response.cancel"}]
        assert result.played_ms == 600
        assert result.cancel_sent is True
        assert result.truncate_sent is False
        assert result.item_id is None
        assert result.sent_event_types == ("response.cancel",)

    asyncio.run(scenario())


@pytest.mark.parametrize("input_kind", ["text", "audio"])
@pytest.mark.parametrize(
    ("output_speed", "audio_end_ms", "repeated_audio_end_ms"),
    [
        (None, 16650, 8000),
        (0.9, 14985, 7200),
        (1.0, 16650, 8000),
        (1.05, 17482, 8400),
        (0.25, 4162, 2000),
        (1.5, 24975, 12000),
    ],
)
def test_playback_stop_caps_position_to_received_pcm_duration(
    input_kind, output_speed, audio_end_ms, repeated_audio_end_ms
):
    async def scenario():
        # Reproduce the observed failure: 17,780 ms requested for 16,650 ms audio.
        pcm = b"\x01\x00" * (24 * 16650)
        transport = _FakeTransport(
            [
                {"type": "response.created", "response": {"id": "resp_cap"}},
                {
                    "type": "response.output_audio.delta",
                    "response_id": "resp_cap",
                    "item_id": "item_cap",
                    "content_index": 0,
                    "delta": base64.b64encode(pcm).decode("ascii"),
                },
                {"type": "response.output_audio.done"},
                {
                    "type": "conversation.item.input_audio_transcription.completed",
                    "transcript": "テスト",
                },
                {"type": "response.done"},
            ]
        )
        session = RealtimeSpeechSession(
            transport, config=RealtimeSpeechConfig(output_speed=output_speed)
        )
        kwargs = dict(
            session_id="cap",
            turn_id=1,
            speaker="client",
            sample_rate=24000,
            sample_width_bits=16,
            channels=1,
            delivery_mode=AudioDeliveryMode.REAL_TIME,
        )
        if input_kind == "text":
            stream = session.stream_audio_response(latest_input="test", **kwargs)
        else:
            stream = session.stream_audio_input_response(
                input_audio=b"\x00" * 4800, **kwargs
            )
        _ = [event async for event in stream]
        result = await session.stop_current_response_playback(
            played_ms=17780, cancel_response=False
        )
        assert transport.sent[-1]["audio_end_ms"] == audio_end_ms
        assert result.played_ms == 16650
        assert result.audio_end_ms == audio_end_ms
        # Once truncated, a repeated stop cannot extend the remaining server audio.
        await session.stop_current_response_playback(
            played_ms=8000, cancel_response=False
        )
        await session.stop_current_response_playback(
            played_ms=16000, cancel_response=False
        )
        assert transport.sent[-1]["audio_end_ms"] == repeated_audio_end_ms

    asyncio.run(scenario())


def test_playback_stop_counts_submillisecond_deltas_and_keeps_item_lengths_separate():
    async def scenario():
        # Three 0.5 ms deltas must be accumulated before flooring to 1 ms.
        pcm_delta = base64.b64encode(b"\x00" * 24).decode("ascii")
        events = [{"type": "response.created", "response": {"id": "resp_small"}}]
        events += [
            {
                "type": "response.output_audio.delta",
                "response_id": "resp_small",
                "item_id": "item_small",
                "content_index": 0,
                "delta": pcm_delta,
            }
        ] * 3
        events += [
            {"type": "response.done"},
            {"type": "response.created", "response": {"id": "resp_other"}},
            {
                "type": "response.output_audio.delta",
                "response_id": "resp_other",
                "item_id": "item_other",
                "content_index": 0,
                "delta": base64.b64encode(b"\x00" * 4800).decode("ascii"),
            },
            {"type": "response.done"},
        ]
        transport = _FakeTransport(events)
        session = RealtimeSpeechSession(transport)
        for turn_id in (1, 2):
            _ = [
                event
                async for event in session.stream_audio_response(
                    session_id="cap",
                    turn_id=turn_id,
                    speaker="client",
                    latest_input="test",
                    sample_rate=24000,
                    sample_width_bits=16,
                    channels=1,
                    delivery_mode=AudioDeliveryMode.REAL_TIME,
                )
            ]
        await session.stop_current_response_playback(
            played_ms=50, item_id="item_other", cancel_response=False
        )
        assert transport.sent[-1]["audio_end_ms"] == 50
        await session.stop_current_response_playback(
            played_ms=50, item_id="item_small", cancel_response=False
        )
        assert transport.sent[-1]["audio_end_ms"] == 1

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("output_speed", "audio_end_ms"), [(None, 25), (0.9, 22), (1.05, 26)]
)
def test_realtime_speech_session_can_truncate_last_streamed_audio_item(
    output_speed, audio_end_ms
) -> None:
    async def scenario() -> None:
        pcm = b"\x01\x00" * 960
        transport = _FakeTransport(
            [
                {
                    "type": "response.created",
                    "response": {"id": "resp_456", "status": "in_progress"},
                },
                {
                    "type": "response.output_audio.delta",
                    "response_id": "resp_456",
                    "item_id": "item_456",
                    "content_index": 0,
                    "delta": base64.b64encode(pcm).decode("ascii"),
                },
                {"type": "response.output_audio.done"},
                {"type": "response.done"},
            ]
        )
        session = RealtimeSpeechSession(
            transport,
            config=RealtimeSpeechConfig(
                target_chunk_duration_ms=40, output_speed=output_speed
            ),
        )

        stream = session.stream_audio_response(
            session_id="session_realtime_speech_truncate_test",
            turn_id=1,
            speaker="counselor",
            latest_input="前の発話です。",
            sample_rate=24_000,
            sample_width_bits=16,
            channels=1,
            delivery_mode=AudioDeliveryMode.REAL_TIME,
        )
        try:
            first_event = await stream.__anext__()
            assert first_event.audio_chunk is not None
            result = await session.stop_current_response_playback(played_ms=25)
        finally:
            await stream.aclose()

        assert transport.sent[-2:] == [
            {
                "type": "response.cancel",
                "response_id": "resp_456",
            },
            {
                "type": "conversation.item.truncate",
                "item_id": "item_456",
                "content_index": 0,
                "audio_end_ms": audio_end_ms,
            },
        ]
        assert result.played_ms == 25
        assert result.cancel_sent is True
        assert result.truncate_sent is True
        assert result.item_id == "item_456"
        assert result.response_id == "resp_456"

    asyncio.run(scenario())


def test_realtime_speech_session_streams_text_and_audio_chunks() -> None:
    async def scenario() -> None:
        pcm = b"\x01\x00" * 120
        transport = _FakeTransport(
            [
                {
                    "type": "response.output_audio_transcript.delta",
                    "delta": "はい、",
                },
                {
                    "type": "response.output_audio.delta",
                    "delta": base64.b64encode(pcm).decode("ascii"),
                },
                {
                    "type": "response.output_audio_transcript.delta",
                    "delta": "短く返します。",
                },
                {"type": "response.output_audio.done"},
                {"type": "response.done"},
            ]
        )
        session = RealtimeSpeechSession(
            transport,
            config=RealtimeSpeechConfig(
                output_speed=0.9,
                target_chunk_duration_ms=40,
            ),
        )

        await session.configure()
        events = [
            event
            async for event in session.stream_audio_response(
                session_id="session_realtime_speech_test",
                turn_id=1,
                speaker="counselor",
                latest_input="前の発話です。",
                sample_rate=24_000,
                sample_width_bits=16,
                channels=1,
                delivery_mode=AudioDeliveryMode.REAL_TIME,
            )
        ]

        assert [event["type"] for event in transport.sent] == [
            "session.update",
            "conversation.item.create",
            "response.create",
        ]
        assert transport.sent[0]["session"]["audio"]["output"]["speed"] == 0.9
        assert transport.sent[1]["item"]["content"][0]["text"] == "前の発話です。"
        assert [event.text_delta for event in events if event.text_delta] == [
            "はい、",
            "短く返します。",
        ]
        audio_chunks = [
            event.audio_chunk for event in events if event.audio_chunk is not None
        ]
        assert len(audio_chunks) == 1
        assert audio_chunks[0].pcm == pcm
        assert audio_chunks[0].duration_ms == 5
        assert audio_chunks[0].delivery_mode is AudioDeliveryMode.REAL_TIME

    asyncio.run(scenario())


def test_realtime_speech_session_streams_audio_input_response_and_transcription() -> (
    None
):
    async def scenario() -> None:
        input_pcm = b"\x01\x00" * 240
        output_pcm = b"\x02\x00" * 120
        transport = _FakeTransport(
            [
                {
                    "type": "conversation.item.input_audio_transcription.delta",
                    "delta": "今日は",
                },
                {
                    "type": "response.output_audio_transcript.delta",
                    "delta": "はい、",
                },
                {
                    "type": "response.output_audio.delta",
                    "delta": base64.b64encode(output_pcm).decode("ascii"),
                },
                {
                    "type": "conversation.item.input_audio_transcription.completed",
                    "transcript": "今日は相談に来ました。",
                },
                {"type": "response.output_audio.done"},
                {"type": "response.done"},
            ]
        )
        session = RealtimeSpeechSession(
            transport,
            config=RealtimeSpeechConfig(target_chunk_duration_ms=40),
        )

        await session.configure()
        events = [
            event
            async for event in session.stream_audio_input_response(
                session_id="session_realtime_audio_input_test",
                turn_id=2,
                speaker="client",
                input_audio=input_pcm,
                sample_rate=24_000,
                sample_width_bits=16,
                channels=1,
                delivery_mode=AudioDeliveryMode.REAL_TIME,
            )
        ]

        assert [event["type"] for event in transport.sent] == [
            "session.update",
            "input_audio_buffer.append",
            "input_audio_buffer.commit",
            "response.create",
        ]
        assert transport.sent[1]["audio"] == base64.b64encode(input_pcm).decode("ascii")
        assert [
            event.input_transcript_delta
            for event in events
            if event.input_transcript_delta
        ] == [
            "今日は",
        ]
        assert [
            event.input_transcript_completed
            for event in events
            if event.input_transcript_completed is not None
        ] == ["今日は相談に来ました。"]
        assert [event.text_delta for event in events if event.text_delta] == ["はい、"]
        audio_chunks = [
            event.audio_chunk for event in events if event.audio_chunk is not None
        ]
        assert len(audio_chunks) == 1
        assert audio_chunks[0].pcm == output_pcm

    asyncio.run(scenario())


@pytest.mark.parametrize("output_speed", [0.2, 1.51])
def test_realtime_speech_config_rejects_invalid_output_speed(
    output_speed: float,
) -> None:
    with pytest.raises(ValueError, match="output_speed"):
        RealtimeSpeechConfig(output_speed=output_speed)


def test_realtime_speech_session_raises_on_error_event() -> None:
    async def scenario() -> None:
        session = RealtimeSpeechSession(
            _FakeTransport(
                [
                    {
                        "type": "error",
                        "error": {
                            "code": "invalid_request_error",
                            "param": "response.audio.output.format",
                            "message": "Bad realtime speech request.",
                        },
                    }
                ]
            )
        )

        with pytest.raises(
            RealtimeSpeechError,
            match="response.audio.output.format.*Bad realtime speech request",
        ):
            async for _ in session.stream_audio_response(
                session_id="session_realtime_speech_error_test",
                turn_id=1,
                speaker="client",
                latest_input="前の発話です。",
                sample_rate=24_000,
                sample_width_bits=16,
                channels=1,
                delivery_mode=AudioDeliveryMode.ACCELERATED,
            ):
                raise AssertionError("error event should not yield")

    asyncio.run(scenario())


def test_realtime_speech_session_ignores_inactive_cancel_error() -> None:
    async def scenario() -> None:
        session = RealtimeSpeechSession(
            _FakeTransport(
                [
                    {
                        "type": "error",
                        "error": {
                            "code": "response_cancel_not_active",
                            "message": "Cancellation failed: no active response found",
                        },
                    },
                    {"type": "response.done"},
                ]
            )
        )

        events = await _collect_realtime_events(
            session.stream_audio_response(
                session_id="session_realtime_cancel_not_active_test",
                turn_id=1,
                speaker="client",
                latest_input="前の発話です。",
                sample_rate=24_000,
                sample_width_bits=16,
                channels=1,
                delivery_mode=AudioDeliveryMode.REAL_TIME,
            )
        )

        assert events == []

    asyncio.run(scenario())


def test_realtime_speech_session_ignores_inactive_cancel_error_for_audio_input() -> (
    None
):
    async def scenario() -> None:
        session = RealtimeSpeechSession(
            _FakeTransport(
                [
                    {
                        "type": "error",
                        "error": {
                            "code": "response_cancel_not_active",
                            "message": "Cancellation failed: no active response found",
                        },
                    },
                    {"type": "response.done"},
                ]
            )
        )

        events = await _collect_realtime_events(
            session.stream_audio_input_response(
                session_id="session_realtime_input_cancel_not_active_test",
                turn_id=1,
                speaker="client",
                input_audio=b"\x00\x00" * 240,
                sample_rate=24_000,
                sample_width_bits=16,
                channels=1,
                delivery_mode=AudioDeliveryMode.REAL_TIME,
            )
        )

        assert events == []

    asyncio.run(scenario())


def test_realtime_speech_session_ignores_stale_response_events_before_current_response_created() -> (
    None
):
    async def scenario() -> None:
        stale_pcm = b"\x01\x00" * 960
        current_pcm = b"\x02\x00" * 960
        session = RealtimeSpeechSession(
            _FakeTransport(
                [
                    {
                        "type": "response.output_audio.delta",
                        "response_id": "resp_stale",
                        "delta": base64.b64encode(stale_pcm).decode("ascii"),
                    },
                    {
                        "type": "response.output_audio.done",
                        "response_id": "resp_stale",
                    },
                    {
                        "type": "response.done",
                        "response": {"id": "resp_stale", "status": "cancelled"},
                    },
                    {
                        "type": "response.created",
                        "response": {"id": "resp_current", "status": "in_progress"},
                    },
                    {
                        "type": "response.output_audio_transcript.delta",
                        "response_id": "resp_current",
                        "delta": "現在の応答です。",
                    },
                    {
                        "type": "response.output_audio.delta",
                        "response_id": "resp_current",
                        "item_id": "item_current",
                        "content_index": 0,
                        "delta": base64.b64encode(current_pcm).decode("ascii"),
                    },
                    {
                        "type": "response.output_audio.done",
                        "response_id": "resp_current",
                        "item_id": "item_current",
                        "content_index": 0,
                    },
                    {
                        "type": "response.done",
                        "response": {"id": "resp_current", "status": "completed"},
                    },
                ]
            )
        )

        events = await _collect_realtime_events(
            session.stream_audio_response(
                session_id="session_realtime_stale_response_test",
                turn_id=4,
                speaker="client",
                latest_input="前の発話です。",
                sample_rate=24_000,
                sample_width_bits=16,
                channels=1,
                delivery_mode=AudioDeliveryMode.REAL_TIME,
            )
        )

        assert [event.text_delta for event in events if event.text_delta] == [
            "現在の応答です。"
        ]
        audio_chunks = [event.audio_chunk for event in events if event.audio_chunk]
        assert len(audio_chunks) == 1
        assert audio_chunks[0].pcm == current_pcm
        assert [event["type"] for event in session.transport.sent] == [
            "conversation.item.create",
            "response.create",
        ]

    asyncio.run(scenario())


def test_openai_realtime_speech_agent_omits_response_instruction_by_default() -> None:
    async def scenario() -> None:
        transport = _FakeTransport([{"type": "response.done"}])

        @asynccontextmanager
        async def transport_context():
            yield transport

        agent = OpenAIRealtimeSpeechAgent(
            speaker="counselor",
            api_key="test-api-key",
            config=RealtimeSpeechConfig(
                instructions="あなたはカウンセラー役です。1文程度で返してください。"
            ),
            transport_context_factory=transport_context,
            reuse_transport=True,
        )

        events = [
            event
            async for event in agent.stream_audio_response(
                session_id="session_realtime_agent_instruction_test",
                turn_id=1,
                speaker="counselor",
                input_transcript="前の発話です。",
                sample_rate=24_000,
                sample_width_bits=16,
                channels=1,
                delivery_mode=AudioDeliveryMode.REAL_TIME,
            )
        ]

        assert events == []
        assert [event["type"] for event in transport.sent] == [
            "session.update",
            "conversation.item.create",
            "response.create",
        ]
        assert "instructions" not in transport.sent[2]["response"]

        await agent.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("with_director", [False, True])
def test_openai_realtime_speech_agent_repeats_counselor_instructions_per_response(
    with_director,
) -> None:
    async def scenario() -> None:
        transport = _FakeTransport(
            [{"type": "response.done"}, {"type": "response.done"}]
        )

        @asynccontextmanager
        async def transport_context():
            yield transport

        agent = OpenAIRealtimeSpeechAgent(
            speaker="counselor",
            api_key="test-api-key",
            config=RealtimeSpeechConfig(
                instructions="固定カウンセラーシステムプロンプト",
                response_instructions="Session setupカウンセラープロンプト",
                repeat_instructions_per_response=True,
            ),
            transport_context_factory=transport_context,
            reuse_transport=True,
        )
        observed_instruction_records = []
        additional_instructions_by_turn = {}

        async def observe_response_instructions(record) -> None:
            observed_instruction_records.append(record)

        for turn_id in (1, 3):
            additional_instruction = (
                render_prompt_director_instruction(
                    PromptDirectorResult(
                        instruction_checks=[
                            {
                                "source": "configured_response_prompt",
                                "quote": "Session setupカウンセラープロンプト",
                                "applicability": "applies",
                                "evidence": f"turn {turn_id}でカウンセラーが応答する。",
                                "force": "required",
                                "priority": "指定なし",
                                "response_excerpt": "どのような状態になればよいと思いますか？",
                                "response_assessment": "望む状態を質問している。",
                            }
                        ],
                        context_basis="望む状態はまだ未確認。",
                        response_intent=f"turn {turn_id}で話者の望む状態を確認する。",
                        response_example="どのような状態になればよいと思いますか？",
                    ),
                    include_audit=False,
                )
                if with_director
                else ""
            )
            additional_instructions_by_turn[turn_id] = additional_instruction
            events = [
                event
                async for event in agent.stream_audio_response(
                    session_id="session_realtime_agent_repeated_instruction_test",
                    turn_id=turn_id,
                    speaker="counselor",
                    input_transcript="前の発話です。",
                    additional_instruction=additional_instruction,
                    sample_rate=24_000,
                    sample_width_bits=16,
                    channels=1,
                    delivery_mode=AudioDeliveryMode.REAL_TIME,
                    response_instructions_observer=observe_response_instructions,
                )
            ]
            assert events == []

        response_events = [
            event for event in transport.sent if event["type"] == "response.create"
        ]
        assert len(response_events) == 2
        for event in response_events:
            response_instructions = event["response"]["instructions"]
            assert "固定カウンセラーシステムプロンプト" in response_instructions
            assert "Session setupカウンセラープロンプト" in response_instructions
        assert [record.turn_id for record in observed_instruction_records] == [1, 3]
        for record, event in zip(
            observed_instruction_records, response_events, strict=True
        ):
            assert record.session_instructions == "固定カウンセラーシステムプロンプト"
            assert (
                record.participant_instructions == "Session setupカウンセラープロンプト"
            )
            expected_additional = additional_instructions_by_turn[record.turn_id]
            assert record.additional_instructions == expected_additional
            assert record.repeat_session_instructions is True
            assert record.resolved_instructions == event["response"]["instructions"]
            if with_director:
                assert expected_additional in record.resolved_instructions
                assert (
                    "＜原文の条件と今回の適用判断＞" not in record.resolved_instructions
                )
                assert "望む状態はまだ未確認。" not in record.resolved_instructions
                encoded = record.resolved_instructions.split(
                    "＜今回発話する本文＞\n", 1
                )[1].split("\n\n", 1)[0]
                assert json.loads(encoded) == {
                    "response_text": "どのような状態になればよいと思いますか？",
                    "require_repeat_verbatim": True,
                }

        await agent.close()

    asyncio.run(scenario())


def test_openai_realtime_speech_agent_decide_timing_uses_timing_llm() -> None:
    class RecordingTimingLLM:
        def __init__(self) -> None:
            self.latest_inputs: list[str] = []
            self.system_prompts: list[str | None] = []

        async def stream_text(self, *, latest_input, history=None, system_prompt=None):
            _ = history
            self.latest_inputs.append(latest_input)
            self.system_prompts.append(system_prompt)
            yield '{"agent_id":"client_b","action":"REQUEST_MAIN_FLOOR"}'

    async def scenario() -> None:
        timing_llm = RecordingTimingLLM()
        agent = OpenAIRealtimeSpeechAgent(
            speaker="client_b",
            api_key="test-api-key",
            input_formatter=lambda *, speaker, turn_id, input_transcript: (
                f"context:{speaker}:{turn_id}:{input_transcript}"
            ),
            timing_llm=timing_llm,
        )

        decision = await agent.decide_timing(
            input_transcript="大変でした。",
            turn_id=8,
            previous_speaker="client_a",
        )

        assert decision.agent_id == "client_b"
        assert decision.action == "REQUEST_MAIN_FLOOR"
        assert decision.prepared_intent is None
        assert "REQUEST_BACKCHANNEL" not in timing_llm.system_prompts[0]
        assert "context:client_b:8:大変でした。" in timing_llm.latest_inputs[0]

    asyncio.run(scenario())


def test_openai_realtime_speech_agent_combines_base_and_additional_instruction() -> (
    None
):
    async def scenario() -> None:
        transport = _FakeTransport([{"type": "response.done"}])

        @asynccontextmanager
        async def transport_context():
            yield transport

        agent = OpenAIRealtimeSpeechAgent(
            speaker="counselor",
            api_key="test-api-key",
            config=RealtimeSpeechConfig(
                instructions="あなたはカウンセラー役です。1文程度で返してください。"
            ),
            transport_context_factory=transport_context,
            reuse_transport=True,
        )

        events = [
            event
            async for event in agent.stream_audio_response(
                session_id="session_realtime_agent_instruction_test",
                turn_id=3,
                speaker="counselor",
                input_transcript="前の発話です。",
                additional_instruction="【クロージング指示】終結に入ってください。",
                sample_rate=24_000,
                sample_width_bits=16,
                channels=1,
                delivery_mode=AudioDeliveryMode.REAL_TIME,
            )
        ]

        assert events == []
        assert [event["type"] for event in transport.sent] == [
            "session.update",
            "conversation.item.create",
            "response.create",
        ]
        conversation_text = transport.sent[1]["item"]["content"][0]["text"]
        assert conversation_text == "前の発話です。"
        assert "クロージング指示" not in conversation_text
        response_instructions = transport.sent[2]["response"]["instructions"]
        assert (
            "あなたはカウンセラー役です。1文程度で返してください。"
            in response_instructions
        )
        assert "追加の内部指示:" in response_instructions
        assert "【クロージング指示】終結に入ってください。" in response_instructions
        assert agent.received_inputs == ["前の発話です。"]

        await agent.close()

    asyncio.run(scenario())


def test_openai_realtime_speech_agent_can_bypass_input_formatter_for_scripted_audio() -> (
    None
):
    async def scenario() -> None:
        transport = _FakeTransport([{"type": "response.done"}])

        @asynccontextmanager
        async def transport_context():
            yield transport

        agent = OpenAIRealtimeSpeechAgent(
            speaker="client",
            api_key="test-api-key",
            config=RealtimeSpeechConfig(instructions="あなたはクライアント役です。"),
            input_formatter=lambda **_: "formatter should not run",
            transport_context_factory=transport_context,
            reuse_transport=True,
        )

        events = [
            event
            async for event in agent.stream_audio_response(
                session_id="session_realtime_agent_scripted_test",
                turn_id=0,
                speaker="client",
                input_transcript="今日は家族のことで相談したいです。",
                additional_instruction="入力された本文をそのまま読み上げてください。",
                format_input=False,
                sample_rate=24_000,
                sample_width_bits=16,
                channels=1,
                delivery_mode=AudioDeliveryMode.REAL_TIME,
            )
        ]

        assert events == []
        assert [event["type"] for event in transport.sent] == [
            "session.update",
            "response.create",
        ]
        response = transport.sent[1]["response"]
        assert response["conversation"] == "none"
        assert response["instructions"]
        assert "今日は家族のことで相談したいです。" in response["instructions"]
        assert "<target>" in response["instructions"]
        assert "承知しました" in response["instructions"]
        assert "読み上げます" in response["instructions"]
        assert (
            "入力された本文をそのまま読み上げてください。"
            not in response["instructions"]
        )
        assert "あなたはクライアント役です。" not in response["instructions"]
        assert agent.received_inputs == ["今日は家族のことで相談したいです。"]

        await agent.close()

    asyncio.run(scenario())


def test_openai_realtime_speech_agent_streams_audio_input_with_context_instruction() -> (
    None
):
    async def scenario() -> None:
        transport = _FakeTransport([{"type": "response.done"}])

        @asynccontextmanager
        async def transport_context():
            yield transport

        agent = OpenAIRealtimeSpeechAgent(
            speaker="client",
            api_key="test-api-key",
            config=RealtimeSpeechConfig(instructions="あなたはクライアント役です。"),
            input_formatter=lambda *, speaker, turn_id, input_transcript, purpose: (
                f"context:{speaker}:{turn_id}:{purpose}:{input_transcript}"
            ),
            transport_context_factory=transport_context,
            reuse_transport=True,
        )

        events = [
            event
            async for event in agent.stream_audio_response_from_audio(
                session_id="session_realtime_agent_audio_input_test",
                turn_id=2,
                speaker="client",
                input_audio=b"\x01\x00" * 240,
                sample_rate=24_000,
                sample_width_bits=16,
                channels=1,
                delivery_mode=AudioDeliveryMode.REAL_TIME,
            )
        ]

        assert events == []
        assert [event["type"] for event in transport.sent] == [
            "session.update",
            "input_audio_buffer.append",
            "input_audio_buffer.commit",
            "response.create",
        ]
        response_instructions = transport.sent[3]["response"]["instructions"]
        assert "あなたはクライアント役です。" in response_instructions
        assert "input audio" in response_instructions
        assert "context:client:2:generation:[audio]" in response_instructions
        assert agent.received_inputs == ["context:client:2:generation:[audio]"]

        await agent.close()

    asyncio.run(scenario())


def test_openai_realtime_speech_agent_scripted_audio_does_not_wait_for_reused_operation() -> (
    None
):
    class _BlockingTransport(_FakeTransport):
        def __init__(self) -> None:
            super().__init__([])
            self.waiting = asyncio.Event()
            self.release = asyncio.Event()
            self._done = False

        async def __anext__(self):
            if self._done:
                raise StopAsyncIteration
            self.waiting.set()
            await self.release.wait()
            self._done = True
            return {"type": "response.done"}

    async def scenario() -> None:
        main_transport = _BlockingTransport()
        scripted_transport = _FakeTransport([{"type": "response.done"}])
        transports = [main_transport, scripted_transport]
        open_count = 0
        close_count = 0

        @asynccontextmanager
        async def transport_context():
            nonlocal open_count, close_count
            transport = transports[open_count]
            open_count += 1
            try:
                yield transport
            finally:
                close_count += 1

        agent = OpenAIRealtimeSpeechAgent(
            speaker="client",
            api_key="test-api-key",
            transport_context_factory=transport_context,
            reuse_transport=True,
        )

        async def run_main_response() -> None:
            _ = [
                event
                async for event in agent.stream_audio_response(
                    session_id="session_realtime_agent_no_wait_test",
                    turn_id=1,
                    speaker="client",
                    input_transcript="通常発話の入力",
                    sample_rate=24_000,
                    sample_width_bits=16,
                    channels=1,
                    delivery_mode=AudioDeliveryMode.ACCELERATED,
                )
            ]

        main_task = asyncio.create_task(run_main_response())
        await asyncio.wait_for(main_transport.waiting.wait(), timeout=1.0)

        scripted_events = await asyncio.wait_for(
            _collect_realtime_events(
                agent.stream_audio_response(
                    session_id="session_realtime_agent_no_wait_test",
                    turn_id=1,
                    speaker="client",
                    input_transcript="うん",
                    format_input=False,
                    sample_rate=24_000,
                    sample_width_bits=16,
                    channels=1,
                    delivery_mode=AudioDeliveryMode.ACCELERATED,
                )
            ),
            timeout=1.0,
        )

        assert scripted_events == []
        assert open_count == 2
        assert close_count == 1
        assert [event["type"] for event in scripted_transport.sent] == [
            "session.update",
            "response.create",
        ]

        main_transport.release.set()
        await main_task
        await agent.close()

    asyncio.run(scenario())


def test_openai_realtime_speech_agent_reuses_transport_across_turns() -> None:
    async def scenario() -> None:
        first_pcm = b"\x01\x00" * 120
        second_pcm = b"\x02\x00" * 120
        transport = _FakeTransport(
            [
                {
                    "type": "response.output_audio_transcript.delta",
                    "delta": "1ターン目。",
                },
                {
                    "type": "response.output_audio.delta",
                    "delta": base64.b64encode(first_pcm).decode("ascii"),
                },
                {"type": "response.output_audio.done"},
                {"type": "response.done"},
                {"type": "rate_limits.updated"},
                {
                    "type": "response.output_audio_transcript.delta",
                    "delta": "2ターン目。",
                },
                {
                    "type": "response.output_audio.delta",
                    "delta": base64.b64encode(second_pcm).decode("ascii"),
                },
                {"type": "response.output_audio.done"},
                {"type": "response.done"},
            ]
        )
        open_count = 0
        close_count = 0

        @asynccontextmanager
        async def transport_context():
            nonlocal open_count, close_count
            open_count += 1
            try:
                yield transport
            finally:
                close_count += 1

        agent = OpenAIRealtimeSpeechAgent(
            speaker="counselor",
            api_key="test-api-key",
            transport_context_factory=transport_context,
            reuse_transport=True,
        )

        first_events = [
            event
            async for event in agent.stream_audio_response(
                session_id="session_realtime_agent_test",
                turn_id=1,
                speaker="counselor",
                input_transcript="入力1",
                sample_rate=24_000,
                sample_width_bits=16,
                channels=1,
                delivery_mode=AudioDeliveryMode.ACCELERATED,
            )
        ]
        second_events = [
            event
            async for event in agent.stream_audio_response(
                session_id="session_realtime_agent_test",
                turn_id=2,
                speaker="counselor",
                input_transcript="入力2",
                sample_rate=24_000,
                sample_width_bits=16,
                channels=1,
                delivery_mode=AudioDeliveryMode.ACCELERATED,
            )
        ]

        assert open_count == 1
        assert close_count == 0
        assert [event["type"] for event in transport.sent] == [
            "session.update",
            "conversation.item.create",
            "response.create",
            "conversation.item.create",
            "response.create",
        ]
        assert [event.text_delta for event in first_events if event.text_delta] == [
            "1ターン目。"
        ]
        assert [event.text_delta for event in second_events if event.text_delta] == [
            "2ターン目。"
        ]
        assert agent.received_inputs == ["入力1", "入力2"]

        await agent.close()
        assert close_count == 1

    asyncio.run(scenario())


def test_openai_realtime_speech_agent_reopens_expired_reused_session_before_output() -> (
    None
):
    async def scenario() -> None:
        expired_transport = _SessionDurationClosedTransport([])
        retry_transport = _FakeTransport([{"type": "response.done"}])
        transports = [expired_transport, retry_transport]
        open_count = 0
        close_count = 0

        @asynccontextmanager
        async def transport_context():
            nonlocal open_count, close_count
            transport = transports[open_count]
            open_count += 1
            try:
                yield transport
            finally:
                close_count += 1

        agent = OpenAIRealtimeSpeechAgent(
            speaker="client",
            api_key="test-api-key",
            transport_context_factory=transport_context,
            reuse_transport=True,
        )

        events = [
            event
            async for event in agent.stream_audio_response(
                session_id="session_realtime_agent_expired_retry_test",
                turn_id=8,
                speaker="client",
                input_transcript="こんにちは。",
                sample_rate=24_000,
                sample_width_bits=16,
                channels=1,
                delivery_mode=AudioDeliveryMode.REAL_TIME,
            )
        ]

        assert events == []
        assert open_count == 2
        assert close_count == 1
        assert [event["type"] for event in expired_transport.sent] == [
            "session.update",
            "conversation.item.create",
            "response.create",
        ]
        assert [event["type"] for event in retry_transport.sent] == [
            "session.update",
            "conversation.item.create",
            "response.create",
        ]

        await agent.close()
        assert close_count == 2

    asyncio.run(scenario())


def test_openai_realtime_speech_agent_reopens_abruptly_closed_reused_session_before_output() -> (
    None
):
    async def scenario() -> None:
        closed_transport = _AbruptlyClosedTransport([])
        retry_transport = _FakeTransport([{"type": "response.done"}])
        transports = [closed_transport, retry_transport]
        open_count = 0
        close_count = 0

        @asynccontextmanager
        async def transport_context():
            nonlocal open_count, close_count
            transport = transports[open_count]
            open_count += 1
            try:
                yield transport
            finally:
                close_count += 1

        agent = OpenAIRealtimeSpeechAgent(
            speaker="client_a",
            api_key="test-api-key",
            transport_context_factory=transport_context,
            reuse_transport=True,
        )

        events = [
            event
            async for event in agent.stream_audio_response(
                session_id="session_realtime_agent_abrupt_retry_test",
                turn_id=9,
                speaker="client_a",
                input_transcript="こんばんは。",
                sample_rate=24_000,
                sample_width_bits=16,
                channels=1,
                delivery_mode=AudioDeliveryMode.REAL_TIME,
            )
        ]

        assert events == []
        assert open_count == 2
        assert close_count == 1
        assert [event["type"] for event in closed_transport.sent] == [
            "session.update",
            "conversation.item.create",
            "response.create",
        ]
        assert [event["type"] for event in retry_transport.sent] == [
            "session.update",
            "conversation.item.create",
            "response.create",
        ]

        await agent.close()
        assert close_count == 2

    asyncio.run(scenario())


def test_openai_realtime_speech_agent_reopens_active_response_conflict_before_output() -> (
    None
):
    async def scenario() -> None:
        conflict_transport = _FakeTransport(
            [
                {
                    "type": "error",
                    "error": {
                        "code": "conversation_already_has_active_response",
                        "message": (
                            "Conversation already has an active response in progress."
                        ),
                    },
                }
            ]
        )
        retry_transport = _FakeTransport([{"type": "response.done"}])
        transports = [conflict_transport, retry_transport]
        open_count = 0
        close_count = 0

        @asynccontextmanager
        async def transport_context():
            nonlocal open_count, close_count
            transport = transports[open_count]
            open_count += 1
            try:
                yield transport
            finally:
                close_count += 1

        agent = OpenAIRealtimeSpeechAgent(
            speaker="client",
            api_key="test-api-key",
            transport_context_factory=transport_context,
            reuse_transport=True,
        )

        events = [
            event
            async for event in agent.stream_audio_response(
                session_id="session_realtime_agent_active_conflict_retry_test",
                turn_id=8,
                speaker="client",
                input_transcript="こんにちは。",
                sample_rate=24_000,
                sample_width_bits=16,
                channels=1,
                delivery_mode=AudioDeliveryMode.REAL_TIME,
            )
        ]

        assert events == []
        assert open_count == 2
        assert close_count == 1
        assert [event["type"] for event in conflict_transport.sent] == [
            "session.update",
            "conversation.item.create",
            "response.create",
        ]
        assert [event["type"] for event in retry_transport.sent] == [
            "session.update",
            "conversation.item.create",
            "response.create",
        ]

        await agent.close()
        assert close_count == 2

    asyncio.run(scenario())


def test_openai_realtime_speech_agent_reopens_expired_audio_input_session_before_output() -> (
    None
):
    async def scenario() -> None:
        expired_transport = _SessionDurationClosedTransport([])
        retry_transport = _FakeTransport([{"type": "response.done"}])
        transports = [expired_transport, retry_transport]
        open_count = 0

        @asynccontextmanager
        async def transport_context():
            nonlocal open_count
            transport = transports[open_count]
            open_count += 1
            yield transport

        agent = OpenAIRealtimeSpeechAgent(
            speaker="client",
            api_key="test-api-key",
            transport_context_factory=transport_context,
            reuse_transport=True,
        )
        input_audio = b"\x01\x00" * 240

        events = [
            event
            async for event in agent.stream_audio_response_from_audio(
                session_id="session_realtime_agent_audio_expired_retry_test",
                turn_id=8,
                speaker="client",
                input_audio=input_audio,
                sample_rate=24_000,
                sample_width_bits=16,
                channels=1,
                delivery_mode=AudioDeliveryMode.REAL_TIME,
            )
        ]

        assert events == []
        assert open_count == 2
        assert [event["type"] for event in expired_transport.sent] == [
            "session.update",
            "input_audio_buffer.append",
            "input_audio_buffer.commit",
            "response.create",
        ]
        assert [event["type"] for event in retry_transport.sent] == [
            "session.update",
            "input_audio_buffer.append",
            "input_audio_buffer.commit",
            "response.create",
        ]
        assert retry_transport.sent[1]["audio"] == base64.b64encode(input_audio).decode(
            "ascii"
        )

        await agent.close()

    asyncio.run(scenario())


def test_openai_realtime_speech_agent_can_stop_active_reused_response() -> None:
    async def scenario() -> None:
        pcm = b"\x01\x00" * 960
        transport = _FakeTransport(
            [
                {
                    "type": "response.created",
                    "response": {"id": "resp_agent", "status": "in_progress"},
                },
                {
                    "type": "response.output_audio.delta",
                    "response_id": "resp_agent",
                    "item_id": "item_agent",
                    "content_index": 0,
                    "delta": base64.b64encode(pcm).decode("ascii"),
                },
                {"type": "response.output_audio.done"},
                {"type": "response.done"},
            ]
        )

        @asynccontextmanager
        async def transport_context():
            yield transport

        agent = OpenAIRealtimeSpeechAgent(
            speaker="counselor",
            api_key="test-api-key",
            transport_context_factory=transport_context,
            reuse_transport=True,
        )

        stream = agent.stream_audio_response(
            session_id="session_realtime_agent_stop_test",
            turn_id=1,
            speaker="counselor",
            input_transcript="入力",
            sample_rate=24_000,
            sample_width_bits=16,
            channels=1,
            delivery_mode=AudioDeliveryMode.REAL_TIME,
        )
        try:
            first_event = await stream.__anext__()
            assert first_event.audio_chunk is not None
            result = await agent.stop_current_response_playback(played_ms=25)
        finally:
            await stream.aclose()
            await agent.close()

        assert result == RealtimePlaybackStopResult(
            played_ms=25,
            audio_end_ms=25,
            cancel_sent=True,
            truncate_sent=True,
            item_id="item_agent",
            response_id="resp_agent",
            content_index=0,
            sent_event_types=("response.cancel", "conversation.item.truncate"),
        )
        assert transport.sent[-2:] == [
            {
                "type": "response.cancel",
                "response_id": "resp_agent",
            },
            {
                "type": "conversation.item.truncate",
                "item_id": "item_agent",
                "content_index": 0,
                "audio_end_ms": 25,
            },
        ]

    asyncio.run(scenario())
