from __future__ import annotations

import asyncio

import pytest

from counseling_voice_demo.runtime.models import AudioDeliveryMode
from counseling_voice_demo.runtime.streaming_tts import (
    OpenAIStreamingTTS,
    OpenAIStreamingTTSConfig,
    StreamingTtsError,
    pcm_duration_ms,
)


class _AsyncByteResponse:
    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = chunks

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback) -> None:
        return None

    async def aiter_bytes(self):
        for chunk in self._chunks:
            yield chunk


def test_openai_streaming_tts_yields_pcm_audio_chunks_with_continuous_indexes() -> None:
    async def scenario() -> None:
        payloads = []

        def response_factory(payload):
            payloads.append(payload)
            return _AsyncByteResponse([b"\x01\x00" * 240, b"\x02\x00" * 120])

        tts = OpenAIStreamingTTS(
            client=object(),
            config=OpenAIStreamingTTSConfig(
                model="gpt-4o-mini-tts",
                voice="coral",
                instructions="落ち着いて話す。",
            ),
            response_factory=response_factory,
        )

        chunks = [
            chunk
            async for chunk in tts.synthesize(
                session_id="session_tts_stream",
                turn_id=3,
                speaker="counselor",
                text_chunks=["最初の文です。", "次の文です。"],
                sample_rate=24000,
                sample_width_bits=16,
                channels=1,
                delivery_mode=AudioDeliveryMode.ACCELERATED,
            )
        ]

        assert payloads == [
            {
                "model": "gpt-4o-mini-tts",
                "voice": "coral",
                "input": "最初の文です。",
                "response_format": "pcm",
                "instructions": "落ち着いて話す。",
            },
            {
                "model": "gpt-4o-mini-tts",
                "voice": "coral",
                "input": "次の文です。",
                "response_format": "pcm",
                "instructions": "落ち着いて話す。",
            },
        ]
        assert [chunk.chunk_index for chunk in chunks] == [0, 1]
        assert [chunk.duration_ms for chunk in chunks] == [15, 15]
        assert all(chunk.pcm for chunk in chunks)

    asyncio.run(scenario())


def test_openai_streaming_tts_rejects_non_pcm_hot_path_format() -> None:
    with pytest.raises(ValueError, match="response_format='pcm'"):
        OpenAIStreamingTTSConfig(response_format="mp3")


def test_openai_streaming_tts_reframes_arbitrary_byte_boundaries() -> None:
    async def scenario() -> None:
        def response_factory(payload):
            return _AsyncByteResponse([b"\x01", b"\x00\x02", b"\x00"])

        tts = OpenAIStreamingTTS(client=object(), response_factory=response_factory)

        chunks = [
            chunk
            async for chunk in tts.synthesize(
                session_id="session_tts_frame",
                turn_id=1,
                speaker="counselor",
                text_chunks=["短文です。"],
                sample_rate=24000,
                sample_width_bits=16,
                channels=1,
                delivery_mode=AudioDeliveryMode.ACCELERATED,
            )
        ]

        assert [chunk.pcm for chunk in chunks] == [b"\x01\x00\x02\x00"]
        assert all(len(chunk.pcm) % 2 == 0 for chunk in chunks)

    asyncio.run(scenario())


def test_openai_streaming_tts_uses_speaker_specific_voice_when_configured() -> None:
    async def scenario() -> None:
        payloads = []

        def response_factory(payload):
            payloads.append(payload)
            return _AsyncByteResponse([b"\x00\x00"])

        tts = OpenAIStreamingTTS(
            client=object(),
            config=OpenAIStreamingTTSConfig(
                voice="coral",
                voice_by_speaker={"client": "cedar"},
            ),
            response_factory=response_factory,
        )

        [
            chunk
            async for chunk in tts.synthesize(
                session_id="session_tts_voice",
                turn_id=2,
                speaker="client",
                text_chunks=["クライアント側の発話です。"],
                sample_rate=24000,
                sample_width_bits=16,
                channels=1,
                delivery_mode=AudioDeliveryMode.ACCELERATED,
            )
        ]

        assert payloads[0]["voice"] == "cedar"

    asyncio.run(scenario())


def test_openai_streaming_tts_coalesces_small_pcm_frames() -> None:
    async def scenario() -> None:
        def response_factory(payload):
            return _AsyncByteResponse([b"\x01\x00" * 240, b"\x02\x00" * 720])

        tts = OpenAIStreamingTTS(
            client=object(),
            config=OpenAIStreamingTTSConfig(target_chunk_duration_ms=40),
            response_factory=response_factory,
        )

        chunks = [
            chunk
            async for chunk in tts.synthesize(
                session_id="session_tts_coalesce",
                turn_id=1,
                speaker="counselor",
                text_chunks=["短文です。"],
                sample_rate=24000,
                sample_width_bits=16,
                channels=1,
                delivery_mode=AudioDeliveryMode.ACCELERATED,
            )
        ]

        assert len(chunks) == 1
        assert chunks[0].duration_ms == 40

    asyncio.run(scenario())


def test_openai_streaming_tts_uses_configured_target_chunk_duration() -> None:
    async def scenario() -> None:
        def response_factory(payload):
            return _AsyncByteResponse([b"\x01\x00" * 960])

        tts = OpenAIStreamingTTS(
            client=object(),
            config=OpenAIStreamingTTSConfig(target_chunk_duration_ms=20),
            response_factory=response_factory,
        )

        chunks = [
            chunk
            async for chunk in tts.synthesize(
                session_id="session_tts_target_duration",
                turn_id=1,
                speaker="counselor",
                text_chunks=["短文です。"],
                sample_rate=24000,
                sample_width_bits=16,
                channels=1,
                delivery_mode=AudioDeliveryMode.ACCELERATED,
            )
        ]

        assert [chunk.duration_ms for chunk in chunks] == [20, 20]

    asyncio.run(scenario())


def test_openai_streaming_tts_raises_on_incomplete_final_pcm_frame() -> None:
    async def scenario() -> None:
        def response_factory(payload):
            return _AsyncByteResponse([b"\x01\x00", b"\x02"])

        tts = OpenAIStreamingTTS(client=object(), response_factory=response_factory)

        with pytest.raises(StreamingTtsError, match="incomplete audio frame"):
            [
                chunk
                async for chunk in tts.synthesize(
                    session_id="session_tts_frame",
                    turn_id=1,
                    speaker="counselor",
                    text_chunks=["短文です。"],
                    sample_rate=24000,
                    sample_width_bits=16,
                    channels=1,
                    delivery_mode=AudioDeliveryMode.ACCELERATED,
                )
            ]

    asyncio.run(scenario())


def test_pcm_duration_ms_uses_24khz_16bit_mono_math() -> None:
    assert pcm_duration_ms(b"\0" * 4800, sample_rate=24000, sample_width_bits=16, channels=1) == 100
