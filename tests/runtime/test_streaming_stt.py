from __future__ import annotations

import asyncio
import base64
import inspect
from contextlib import asynccontextmanager

import pytest

from counseling_voice_demo.runtime.models import AudioChunk, AudioDeliveryMode
from counseling_voice_demo.runtime.streaming_stt import (
    FakeStreamingSTT,
    OpenAIRealtimeTranscriptionSTT,
    RealtimeTranscriptionError,
    build_input_audio_append_event,
    build_input_audio_commit_event,
    build_realtime_transcription_session_update,
    iter_turn_audio_chunks,
    parse_realtime_transcription_event,
)
from counseling_voice_demo.runtime.streaming_tts import FakeStreamingTTS


def test_build_input_audio_append_event_base64_encodes_pcm() -> None:
    chunk = AudioChunk(
        session_id="session_stt_test",
        turn_id=1,
        speaker="counselor",
        chunk_index=0,
        pcm=b"\x00\x01\x02\xff",
    )

    event = build_input_audio_append_event(chunk)

    assert event == {
        "type": "input_audio_buffer.append",
        "audio": base64.b64encode(chunk.pcm).decode("ascii"),
    }


def test_build_realtime_transcription_session_update_keeps_null_turn_detection() -> None:
    event = build_realtime_transcription_session_update(turn_detection=None)

    assert event["type"] == "session.update"
    session = event["session"]
    assert session["type"] == "transcription"
    audio_input = session["audio"]["input"]
    assert audio_input["format"] == {
        "type": "audio/pcm",
        "rate": 24000,
    }
    assert audio_input["transcription"] == {
        "model": "gpt-4o-transcribe",
        "language": "ja",
    }
    assert "turn_detection" in audio_input
    assert audio_input["turn_detection"] is None


def test_openai_realtime_transcription_stt_vad_auto_uses_server_vad_and_returns_on_final() -> None:
    class FakeTransport:
        def __init__(self) -> None:
            self.sent: list[dict] = []
            self.release_final = asyncio.Event()

        def send_json(self, event: dict) -> None:
            self.sent.append(event)

        def __aiter__(self):
            return self._iter_events()

        async def _iter_events(self):
            await self.release_final.wait()
            yield {
                "type": "conversation.item.input_audio_transcription.completed",
                "transcript": "自動区切りで聞き取れました。",
            }

    async def scenario() -> None:
        transport = FakeTransport()
        opened = False

        @asynccontextmanager
        async def transport_context():
            nonlocal opened
            opened = True
            yield transport

        queue: asyncio.Queue[object] = asyncio.Queue()
        task = asyncio.create_task(
            OpenAIRealtimeTranscriptionSTT(
                api_key="test-api-key",
                transport_context_factory=transport_context,
            ).transcribe_from_queue(
                session_id="session_openai_stt_vad_auto_test",
                turn_id=11,
                speaker="counselor",
                queue=queue,
                recording_mode="vad_auto",
            )
        )
        await queue.put(
            AudioChunk(
                session_id="session_openai_stt_vad_auto_test",
                turn_id=11,
                speaker="counselor",
                chunk_index=0,
                pcm=b"\x00\x01",
            )
        )
        await asyncio.sleep(0)
        transport.release_final.set()

        partials, final = await asyncio.wait_for(task, timeout=1.0)

        assert opened is True
        assert transport.sent[0]["type"] == "session.update"
        assert transport.sent[0]["session"]["audio"]["input"]["turn_detection"] == {
            "type": "server_vad",
            "threshold": 0.5,
            "prefix_padding_ms": 300,
            "silence_duration_ms": 1400,
        }
        assert [event["type"] for event in transport.sent[1:]] == [
            "input_audio_buffer.append",
        ]
        assert partials == []
        assert final.text == "自動区切りで聞き取れました。"

    asyncio.run(scenario())


def test_parse_realtime_transcription_delta_returns_partial_transcript_event() -> None:
    parsed = parse_realtime_transcription_event(
        {
            "event_id": "event_delta",
            "type": "conversation.item.input_audio_transcription.delta",
            "item_id": "item_1",
            "content_index": 0,
            "delta": "こんにちは",
        },
        session_id="session_stt_test",
        turn_id=2,
        speaker="client",
    )

    assert parsed is not None
    assert parsed.session_id == "session_stt_test"
    assert parsed.turn_id == 2
    assert parsed.speaker == "client"
    assert parsed.transcript_type == "partial"
    assert parsed.text == "こんにちは"
    assert parsed.metadata == {
        "event_id": "event_delta",
        "item_id": "item_1",
        "content_index": 0,
    }


def test_parse_realtime_transcription_completed_returns_final_transcript_event() -> None:
    parsed = parse_realtime_transcription_event(
        {
            "event_id": "event_completed",
            "type": "conversation.item.input_audio_transcription.completed",
            "item_id": "item_2",
            "content_index": 1,
            "transcript": "相談したいです。",
        },
        session_id="session_stt_test",
        turn_id=3,
        speaker="counselor",
    )

    assert parsed is not None
    assert parsed.transcript_type == "final"
    assert parsed.text == "相談したいです。"
    assert parsed.metadata == {
        "event_id": "event_completed",
        "item_id": "item_2",
        "content_index": 1,
    }


def test_parse_realtime_transcription_event_ignores_unrelated_events() -> None:
    parsed = parse_realtime_transcription_event(
        {"type": "rate_limits.updated"},
        session_id="session_stt_test",
        turn_id=4,
        speaker="client",
    )

    assert parsed is None


def test_parse_realtime_transcription_event_raises_on_error_event() -> None:
    with pytest.raises(
        RealtimeTranscriptionError,
        match="code=unknown_parameter: param=session.type",
    ):
        parse_realtime_transcription_event(
            {
                "type": "error",
                "error": {
                    "code": "unknown_parameter",
                    "param": "session.type",
                    "message": "Unknown parameter: 'session.type'.",
                },
            },
            session_id="session_stt_test",
            turn_id=4,
            speaker="client",
        )


def test_build_input_audio_commit_event() -> None:
    assert build_input_audio_commit_event() == {"type": "input_audio_buffer.commit"}


def test_fake_streaming_stt_uses_realtime_compatible_signature() -> None:
    signature = inspect.signature(FakeStreamingSTT.transcribe)

    assert "expected_text" not in signature.parameters
    assert set(signature.parameters) == {
        "self",
        "session_id",
        "turn_id",
        "speaker",
        "chunks",
    }


def test_fake_streaming_stt_reconstructs_text_from_fake_tts_chunks() -> None:
    async def scenario() -> None:
        tts = FakeStreamingTTS()
        chunks = [
            chunk
            async for chunk in tts.synthesize(
                session_id="session_stt_test",
                turn_id=5,
                speaker="client",
                text_chunks=["最初の文。", "次の文。"],
                sample_rate=24_000,
                sample_width_bits=16,
                channels=1,
                delivery_mode=AudioDeliveryMode.ACCELERATED,
            )
        ]

        partials, final = await FakeStreamingSTT().transcribe(
            session_id="session_stt_test",
            turn_id=5,
            speaker="client",
            chunks=reversed(chunks),
        )

        assert [partial.text for partial in partials] == ["partial:client:5:2"]
        assert final.transcript_type == "final"
        assert final.text == "stt_final:最初の文。次の文。"

    asyncio.run(scenario())


def test_iter_turn_audio_chunks_reads_until_matching_end_of_audio() -> None:
    async def scenario() -> None:
        queue: asyncio.Queue[object] = asyncio.Queue()
        chunk_1 = AudioChunk(
            session_id="session_stt_test",
            turn_id=7,
            speaker="counselor",
            chunk_index=0,
            pcm=b"one",
        )
        chunk_2 = AudioChunk(
            session_id="session_stt_test",
            turn_id=7,
            speaker="counselor",
            chunk_index=1,
            pcm=b"two",
        )
        await queue.put(chunk_1)
        await queue.put(chunk_2)
        from counseling_voice_demo.runtime.models import EndOfAudio

        await queue.put(EndOfAudio(session_id="session_stt_test", turn_id=7, speaker="counselor"))

        chunks = [
            chunk
            async for chunk in iter_turn_audio_chunks(
                queue,
                session_id="session_stt_test",
                turn_id=7,
                speaker="counselor",
            )
        ]

        assert chunks == [chunk_1, chunk_2]

    asyncio.run(scenario())


def test_realtime_streaming_stt_transcribes_from_queue_with_commit_after_turn_end() -> None:
    class FakeTransport:
        def __init__(self) -> None:
            self.sent: list[dict] = []
            self.events = [
                {
                    "type": "conversation.item.input_audio_transcription.delta",
                    "delta": "相談",
                },
                {
                    "type": "conversation.item.input_audio_transcription.completed",
                    "transcript": "相談内容です。",
                },
            ]

        def send_json(self, event: dict) -> None:
            self.sent.append(event)

        def __aiter__(self):
            return self._iter_events()

        async def _iter_events(self):
            for event in self.events:
                yield event

    async def scenario() -> None:
        from counseling_voice_demo.runtime.models import EndOfAudio
        from counseling_voice_demo.runtime.streaming_stt import RealtimeStreamingSTT

        queue: asyncio.Queue[object] = asyncio.Queue()
        await queue.put(
            AudioChunk(
                session_id="session_stt_test",
                turn_id=8,
                speaker="client",
                chunk_index=0,
                pcm=b"\x00\x01",
            )
        )
        await queue.put(EndOfAudio(session_id="session_stt_test", turn_id=8, speaker="client"))
        transport = FakeTransport()

        partials, final = await RealtimeStreamingSTT(transport).transcribe_from_queue(
            session_id="session_stt_test",
            turn_id=8,
            speaker="client",
            queue=queue,
        )

        assert [event["type"] for event in transport.sent] == [
            "input_audio_buffer.append",
            "input_audio_buffer.commit",
        ]
        assert [partial.text for partial in partials] == ["相談"]
        assert final.text == "相談内容です。"

    asyncio.run(scenario())


def test_realtime_streaming_stt_observed_callback_receives_transcripts() -> None:
    class FakeTransport:
        def __init__(self) -> None:
            self.sent: list[dict] = []
            self.events = [
                {
                    "type": "conversation.item.input_audio_transcription.delta",
                    "delta": "相談",
                },
                {
                    "type": "conversation.item.input_audio_transcription.completed",
                    "transcript": "相談内容です。",
                },
            ]

        def send_json(self, event: dict) -> None:
            self.sent.append(event)

        def __aiter__(self):
            return self._iter_events()

        async def _iter_events(self):
            for event in self.events:
                yield event

    async def scenario() -> None:
        from counseling_voice_demo.runtime.models import EndOfAudio
        from counseling_voice_demo.runtime.streaming_stt import RealtimeStreamingSTT

        queue: asyncio.Queue[object] = asyncio.Queue()
        await queue.put(
            AudioChunk(
                session_id="session_stt_observed_test",
                turn_id=8,
                speaker="client",
                chunk_index=0,
                pcm=b"\x00\x01",
            )
        )
        await queue.put(
            EndOfAudio(
                session_id="session_stt_observed_test",
                turn_id=8,
                speaker="client",
            )
        )
        observed: list[tuple[str, str]] = []

        async def on_transcript(transcript) -> None:
            observed.append((transcript.transcript_type, transcript.text))

        partials, final = await RealtimeStreamingSTT(FakeTransport()).transcribe_from_queue_observed(
            session_id="session_stt_observed_test",
            turn_id=8,
            speaker="client",
            queue=queue,
            on_transcript=on_transcript,
        )

        assert observed == [("partial", "相談"), ("final", "相談内容です。")]
        assert [partial.text for partial in partials] == ["相談"]
        assert final.text == "相談内容です。"

    asyncio.run(scenario())


def test_realtime_streaming_stt_raises_api_error_instead_of_waiting_for_final() -> None:
    class FakeTransport:
        def send_json(self, event: dict) -> None:
            pass

        def __aiter__(self):
            return self._iter_events()

        async def _iter_events(self):
            yield {
                "type": "error",
                "error": {
                    "code": "invalid_request_error",
                    "message": "Bad realtime transcription request.",
                },
            }

    async def scenario() -> None:
        from counseling_voice_demo.runtime.streaming_stt import RealtimeStreamingSTT

        with pytest.raises(RealtimeTranscriptionError, match="Bad realtime transcription request"):
            async for _ in RealtimeStreamingSTT(FakeTransport()).stream_transcripts(
                session_id="session_stt_test",
                turn_id=8,
                speaker="client",
            ):
                raise AssertionError("error event should not yield transcripts")

    asyncio.run(scenario())


@pytest.mark.parametrize("end_order", ["after_vad", "before_vad_ack", "trailing_silence"])
def test_vad_client_end_preserves_the_committed_speech_transcript(end_order) -> None:
    from counseling_voice_demo.runtime.models import EndOfAudio
    from counseling_voice_demo.runtime.streaming_stt import RealtimeStreamingSTT

    async def scenario() -> None:
        queue = asyncio.Queue()
        events = asyncio.Queue()
        end = EndOfAudio(session_id="vad_race", turn_id=1, speaker="client")
        committed = {"type": "input_audio_buffer.committed", "item_id": "speech"}
        delta = {
            "type": "conversation.item.input_audio_transcription.delta",
            "item_id": "speech",
            "delta": "相談",
        }
        final = {
            "type": "conversation.item.input_audio_transcription.completed",
            "item_id": "speech",
            "transcript": "相談内容です。",
        }

        class Transport:
            def __init__(self):
                self.sent = []

            async def send_json(self, event):
                self.sent.append(event)
                if event["type"] == "input_audio_buffer.append":
                    if end_order == "after_vad":
                        events.put_nowait(committed)
                        events.put_nowait(delta)
                    else:
                        queue.put_nowait(end)
                elif event["type"] == "input_audio_buffer.commit":
                    if end_order != "after_vad":
                        events.put_nowait(committed)
                        events.put_nowait(delta)
                    if end_order == "trailing_silence":
                        events.put_nowait(
                            {
                                "type": "input_audio_buffer.committed",
                                "item_id": "silence",
                            }
                        )
                        events.put_nowait(
                            {**final, "item_id": "silence", "transcript": ""}
                        )
                    else:
                        events.put_nowait(
                            {
                                "type": "error",
                                "error": {
                                    "code": "input_audio_buffer_commit_empty",
                                    "event_id": event.get("event_id"),
                                    "message": "buffer too small",
                                },
                            }
                        )
                    events.put_nowait(final)

            async def __aiter__(self):
                while True:
                    yield await events.get()

        async def observe(transcript):
            if end_order == "after_vad" and transcript.transcript_type == "partial":
                queue.put_nowait(end)
                # The append task gets a chance to process end before the final.
                await asyncio.sleep(0)
                events.put_nowait(final)

        transport = Transport()
        queue.put_nowait(
            AudioChunk(
                session_id="vad_race",
                turn_id=1,
                speaker="client",
                chunk_index=0,
                pcm=b"\x00\x01" * 2400,
            )
        )
        partials, result = await asyncio.wait_for(
            RealtimeStreamingSTT(
                transport, turn_detection={"type": "server_vad"}
            ).transcribe_from_queue_observed(
                session_id="vad_race",
                turn_id=1,
                speaker="client",
                queue=queue,
                on_transcript=observe,
            ),
            timeout=1,
        )
        assert result.text == "相談内容です。"
        assert result.metadata["item_id"] == "speech"
        assert [partial.text for partial in partials] == ["相談"]
        if end_order == "after_vad":
            assert not any(
                e["type"] == "input_audio_buffer.commit" for e in transport.sent
            )

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "case", ["no_commit", "wrong_event", "other_error", "ptt", "failed"]
)
def test_transcription_errors_without_a_proven_vad_commit_race_still_raise(
    case,
) -> None:
    from counseling_voice_demo.runtime.models import EndOfAudio
    from counseling_voice_demo.runtime.streaming_stt import RealtimeStreamingSTT

    async def scenario() -> None:
        class Transport:
            def __init__(self):
                self.commit_sent = asyncio.Event()
                self.commit_event_id = None

            def send_json(self, event):
                if event["type"] == "input_audio_buffer.commit":
                    self.commit_event_id = event.get("event_id")
                    self.commit_sent.set()

            async def __aiter__(self):
                await self.commit_sent.wait()
                if case != "no_commit":
                    yield {"type": "input_audio_buffer.committed", "item_id": "speech"}
                yield {
                    "type": (
                        "conversation.item.input_audio_transcription.failed"
                        if case == "failed"
                        else "error"
                    ),
                    "item_id": "speech",
                    "error": {
                        "code": (
                            "transcription_failed"
                            if case in ("other_error", "failed")
                            else "input_audio_buffer_commit_empty"
                        ),
                        "event_id": (
                            "unrelated"
                            if case == "wrong_event"
                            else self.commit_event_id
                        ),
                        "message": "expected test failure",
                    },
                }
                yield {
                    "type": "conversation.item.input_audio_transcription.completed",
                    "item_id": "speech",
                    "transcript": "must not succeed",
                }

        queue = asyncio.Queue()
        queue.put_nowait(
            AudioChunk(
                session_id="error",
                turn_id=1,
                speaker="client",
                chunk_index=0,
                pcm=b"\x00\x01" * 2400,
            )
        )
        queue.put_nowait(EndOfAudio(session_id="error", turn_id=1, speaker="client"))
        stt = RealtimeStreamingSTT(
            Transport(),
            turn_detection=None if case == "ptt" else {"type": "server_vad"},
        )
        with pytest.raises(RealtimeTranscriptionError, match="expected test failure"):
            await asyncio.wait_for(
                stt.transcribe_from_queue(
                    session_id="error",
                    turn_id=1,
                    speaker="client",
                    queue=queue,
                ),
                timeout=1,
            )

    asyncio.run(scenario())


@pytest.mark.parametrize("recording_mode", ["push_to_talk", "browser_vad"])
def test_openai_realtime_transcription_stt_opens_transport_and_configures_session(recording_mode) -> None:
    class FakeTransport:
        def __init__(self) -> None:
            self.sent: list[dict] = []
            self.events = [
                {
                    "type": "conversation.item.input_audio_transcription.completed",
                    "transcript": "聞き取れました。",
                },
            ]

        def send_json(self, event: dict) -> None:
            self.sent.append(event)

        def __aiter__(self):
            return self._iter_events()

        async def _iter_events(self):
            for event in self.events:
                yield event

    async def scenario() -> None:
        from counseling_voice_demo.runtime.models import EndOfAudio

        transport = FakeTransport()
        opened = False
        closed = False

        @asynccontextmanager
        async def transport_context():
            nonlocal opened, closed
            opened = True
            try:
                yield transport
            finally:
                closed = True

        queue: asyncio.Queue[object] = asyncio.Queue()
        await queue.put(
            AudioChunk(
                session_id="session_openai_stt_test",
                turn_id=9,
                speaker="counselor",
                chunk_index=0,
                pcm=b"\x00\x01",
            )
        )
        await queue.put(EndOfAudio(session_id="session_openai_stt_test", turn_id=9, speaker="counselor"))

        partials, final = await OpenAIRealtimeTranscriptionSTT(
            api_key="test-api-key",
            model="gpt-test-transcribe",
            turn_detection={"type": "server_vad"} if recording_mode == "browser_vad" else None,
            transport_context_factory=transport_context,
        ).transcribe_from_queue(
            session_id="session_openai_stt_test",
            turn_id=9,
            speaker="counselor",
            queue=queue,
            recording_mode=recording_mode,
        )

        assert opened is True
        assert closed is True
        assert transport.sent[0]["session"]["audio"]["input"]["turn_detection"] is None
        assert [event["type"] for event in transport.sent] == [
            "session.update",
            "input_audio_buffer.append",
            "input_audio_buffer.commit",
        ]
        assert (
            transport.sent[0]["session"]["audio"]["input"]["transcription"]["model"]
            == "gpt-test-transcribe"
        )
        assert partials == []
        assert final.text == "聞き取れました。"

    asyncio.run(scenario())


def test_openai_realtime_transcription_stt_waits_for_first_audio_before_opening_transport() -> None:
    class FakeTransport:
        def __init__(self) -> None:
            self.sent: list[dict] = []
            self.events = [
                {
                    "type": "conversation.item.input_audio_transcription.completed",
                    "transcript": "遅延なく接続しました。",
                },
            ]

        def send_json(self, event: dict) -> None:
            self.sent.append(event)

        def __aiter__(self):
            return self._iter_events()

        async def _iter_events(self):
            for event in self.events:
                yield event

    async def scenario() -> None:
        from counseling_voice_demo.runtime.models import EndOfAudio

        transport = FakeTransport()
        opened = False

        @asynccontextmanager
        async def transport_context():
            nonlocal opened
            opened = True
            yield transport

        queue: asyncio.Queue[object] = asyncio.Queue()
        task = asyncio.create_task(
            OpenAIRealtimeTranscriptionSTT(
                api_key="test-api-key",
                transport_context_factory=transport_context,
            ).transcribe_from_queue(
                session_id="session_openai_stt_lazy_test",
                turn_id=10,
                speaker="client",
                queue=queue,
            )
        )
        await asyncio.sleep(0)
        assert opened is False

        await queue.put(
            AudioChunk(
                session_id="session_openai_stt_lazy_test",
                turn_id=10,
                speaker="client",
                chunk_index=0,
                pcm=b"\x00\x01",
            )
        )
        await queue.put(EndOfAudio(session_id="session_openai_stt_lazy_test", turn_id=10, speaker="client"))

        partials, final = await task

        assert opened is True
        assert [event["type"] for event in transport.sent] == [
            "session.update",
            "input_audio_buffer.append",
            "input_audio_buffer.commit",
        ]
        assert partials == []
        assert final.text == "遅延なく接続しました。"

    asyncio.run(scenario())


def test_openai_realtime_transcription_stt_can_reuse_transport_across_turns() -> None:
    class FakeTransport:
        def __init__(self) -> None:
            self.sent: list[dict] = []
            self.events = [
                {
                    "type": "conversation.item.input_audio_transcription.completed",
                    "transcript": "1ターン目。",
                },
                {
                    "type": "conversation.item.input_audio_transcription.completed",
                    "transcript": "2ターン目。",
                },
            ]

        def send_json(self, event: dict) -> None:
            self.sent.append(event)

        def __aiter__(self):
            return self

        async def __anext__(self):
            if not self.events:
                raise StopAsyncIteration
            return self.events.pop(0)

    async def scenario() -> None:
        transport = FakeTransport()
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

        stt = OpenAIRealtimeTranscriptionSTT(
            api_key="test-api-key",
            model="gpt-test-transcribe",
            transport_context_factory=transport_context,
            reuse_transport=True,
        )

        _, first = await stt.transcribe(
            session_id="session_reuse_stt_test",
            turn_id=1,
            speaker="counselor",
            chunks=[
                AudioChunk(
                    session_id="session_reuse_stt_test",
                    turn_id=1,
                    speaker="counselor",
                    chunk_index=0,
                    pcm=b"\x00\x01",
                )
            ],
        )
        _, second = await stt.transcribe(
            session_id="session_reuse_stt_test",
            turn_id=2,
            speaker="client",
            chunks=[
                AudioChunk(
                    session_id="session_reuse_stt_test",
                    turn_id=2,
                    speaker="client",
                    chunk_index=0,
                    pcm=b"\x02\x03",
                )
            ],
        )

        assert first.text == "1ターン目。"
        assert second.text == "2ターン目。"
        assert open_count == 1
        assert close_count == 0
        assert [event["type"] for event in transport.sent] == [
            "session.update",
            "input_audio_buffer.append",
            "input_audio_buffer.commit",
            "input_audio_buffer.append",
            "input_audio_buffer.commit",
        ]

        await stt.close()
        assert close_count == 1

    asyncio.run(scenario())
