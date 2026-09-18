from __future__ import annotations

import asyncio
import hashlib
import json

import pytest

from counseling_voice_demo.runtime.agents import FakeAgent
from counseling_voice_demo.runtime.audio_bus import AudioBus
from counseling_voice_demo.runtime.controller import (
    GenerationInterruptedForHumanInput,
    TurnController,
    _apply_speaker_audio_gain,
)
from counseling_voice_demo.runtime.logger import AsyncLogger
from counseling_voice_demo.runtime.models import (
    AudioChunk,
    AudioDeliveryMode,
    EndOfAudio,
    ParticipantConfig,
    RuntimeConfig,
    TranscriptEvent,
)
from counseling_voice_demo.runtime.realtime_speech import ResponseCreateInstructions
from counseling_voice_demo.runtime.speaker_selection import SpeakerSelectionError


async def first_monitor_audio_chunk(audio_bus: AudioBus) -> AudioChunk:
    while True:
        item = await asyncio.wait_for(audio_bus.monitor_queue.get(), timeout=1.0)
        if isinstance(item, AudioChunk):
            return item


def test_turn_controller_completes_fake_turn_from_runtime_signals() -> None:
    async def scenario() -> None:
        config = RuntimeConfig(session_id="session_turn_test", max_turns=1)
        agents = {
            "counselor": FakeAgent("counselor", "受け止めました。{input_transcript}"),
            "client": FakeAgent("client", "話してみます。{input_transcript}"),
        }
        controller = TurnController(config=config, agents=agents)
        audio_bus = AudioBus()

        state = await controller.run_turn(
            turn_id=1,
            input_transcript="家族のことで困っています。",
            audio_bus=audio_bus,
        )
        await audio_bus.close()

        assert state.tts_done is True
        assert state.audio_delivered_to_stt is True
        assert state.stt_final_transcript == state.generated_text
        assert state.can_advance is True
        assert state.completed_at is not None

    asyncio.run(scenario())


def test_turn_controller_logs_generated_final_transcript(tmp_path) -> None:
    async def scenario() -> None:
        config = RuntimeConfig(session_id="session_generated_log_test", max_turns=1)
        controller = TurnController(
            config=config,
            agents={
                "counselor": FakeAgent("counselor", "生成元テキストです。"),
                "client": FakeAgent("client", "話してみます。"),
            },
        )
        logger = AsyncLogger(
            sessions_dir=tmp_path,
            session_id="session_generated_log_test",
        )
        await logger.start()
        audio_bus = AudioBus()

        state = await controller.run_turn(
            turn_id=1,
            input_transcript="相談したいです。",
            audio_bus=audio_bus,
            logger=logger,
        )
        await audio_bus.close()
        await logger.close()

        transcript_path = (
            tmp_path
            / "session_generated_log_test"
            / "internal"
            / "transcripts"
            / "session_generated_log_test.jsonl"
        )
        transcript_events = [
            json.loads(line)
            for line in transcript_path.read_text(encoding="utf-8").splitlines()
        ]

        assert state.generated_text == "生成元テキストです。"
        assert not any(event["transcript_type"] == "final" for event in transcript_events)
        assert any(
            event["transcript_type"] == "generated_final"
            and event["text"] == "生成元テキストです。"
            for event in transcript_events
        )

    asyncio.run(scenario())


def test_turn_controller_logs_generated_interrupted_transcript(tmp_path) -> None:
    class InterruptingStreamingAgent:
        speaker = "client"

        def __init__(self) -> None:
            self.yielded_parts = 0

        async def stream_generate(self, **_kwargs):
            self.yielded_parts += 1
            yield "途中まで話していた内容です。"
            self.yielded_parts += 1
            yield "続きです。"

    async def scenario() -> None:
        agent = InterruptingStreamingAgent()
        config = RuntimeConfig(
            session_id="session_generated_interrupted_log_test",
            max_turns=1,
            fixed_speaker_sequence=("client",),
            participants={
                "counselor": ParticipantConfig("counselor", "counselor", "カウンセラー"),
                "client": ParticipantConfig("client", "client", "夫"),
            },
        )
        controller = TurnController(
            config=config,
            agents={
                "counselor": FakeAgent("counselor", "確認します。"),
                "client": agent,
            },
        )
        logger = AsyncLogger(
            sessions_dir=tmp_path,
            session_id="session_generated_interrupted_log_test",
        )
        await logger.start()
        audio_bus = AudioBus()

        with pytest.raises(GenerationInterruptedForHumanInput):
            await controller.run_turn(
                turn_id=1,
                input_transcript="確認します。",
                audio_bus=audio_bus,
                speaker="client",
                logger=logger,
                generation_interrupt_requested=lambda: agent.yielded_parts > 1,
            )
        await audio_bus.close()
        await logger.close()

        transcript_path = (
            tmp_path
            / "session_generated_interrupted_log_test"
            / "internal"
            / "transcripts"
            / "session_generated_interrupted_log_test.jsonl"
        )
        transcript_events = [
            json.loads(line)
            for line in transcript_path.read_text(encoding="utf-8").splitlines()
        ]

        assert len(transcript_events) == 1
        assert transcript_events[0]["transcript_type"] == "generated_interrupted"
        assert transcript_events[0]["text"] == "途中まで話していた内容です。"
        assert transcript_events[0]["metadata"] == {
            "audio_log_path": "internal/audio/turn_0001_client.wav",
            "interrupted": True,
            "role": "client",
            "speaker_display_name": "夫",
            "text_source": "generated_text",
        }

    asyncio.run(scenario())


def test_turn_controller_publishes_monitor_transcript_events() -> None:
    async def scenario() -> None:
        config = RuntimeConfig(session_id="session_monitor_transcript_test", max_turns=1)
        controller = TurnController(
            config=config,
            agents={
                "counselor": FakeAgent("counselor", "生成中です。完了です。"),
                "client": FakeAgent("client", "unused"),
            },
        )
        audio_bus = AudioBus()

        state = await controller.run_turn(
            turn_id=1,
            input_transcript="相談したいです。",
            audio_bus=audio_bus,
        )
        await audio_bus.close()

        monitor_items: list[object] = []
        while not audio_bus.monitor_queue.empty():
            monitor_items.append(audio_bus.monitor_queue.get_nowait())

        transcript_events = [
            item for item in monitor_items if isinstance(item, TranscriptEvent)
        ]
        assert [event.transcript_type for event in transcript_events] == [
            "delta",
            "delta",
            "generated_final",
        ]
        assert "".join(event.text for event in transcript_events[:2]) == state.generated_text
        assert transcript_events[-1].text == state.generated_text
        assert transcript_events[-1].metadata == {
            "speaker_display_name": "カウンセラー",
            "role": "counselor",
            "text_source": "generated_text",
        }

    asyncio.run(scenario())


def test_turn_controller_strips_leading_speaker_label_before_tts() -> None:
    class SplitLabelAgent:
        async def generate(self, *, input_transcript: str, turn_id: int):
            _ = input_transcript, turn_id
            return ["妻", "：", "正直、毎日の声かけがしんどいです。"]

    async def scenario() -> None:
        config = RuntimeConfig(
            session_id="session_strip_speaker_label_test",
            participants={
                "counselor": ParticipantConfig(
                    speaker_id="counselor",
                    role="counselor",
                    display_name="カウンセラー",
                ),
                "client_a": ParticipantConfig(
                    speaker_id="client_a",
                    role="client",
                    display_name="妻",
                ),
            },
            fixed_speaker_sequence=("counselor", "client_a"),
            max_turns=1,
        )
        controller = TurnController(
            config=config,
            agents={
                "counselor": FakeAgent("counselor", "確認します。"),
                "client_a": SplitLabelAgent(),
            },
        )
        audio_bus = AudioBus()

        state = await controller.run_turn(
            turn_id=2,
            input_transcript="今日はどうされましたか。",
            audio_bus=audio_bus,
        )
        await audio_bus.close()

        assert state.generated_text == "正直、毎日の声かけがしんどいです。"
        assert state.stt_final_transcript == "正直、毎日の声かけがしんどいです。"

    asyncio.run(scenario())


def test_turn_controller_alternates_speakers() -> None:
    controller = TurnController(
        config=RuntimeConfig(),
        agents={
            "counselor": FakeAgent("counselor", "a"),
            "client": FakeAgent("client", "b"),
        },
    )

    assert controller.speaker_for_turn(1) == "counselor"
    assert controller.speaker_for_turn(2) == "client"
    assert controller.speaker_for_turn(3) == "counselor"
    assert controller.speaker_for_turn(4) == "client"


def test_turn_controller_uses_configured_fixed_speaker_sequence_for_two_clients() -> None:
    config = RuntimeConfig(
        participants={
            "counselor": ParticipantConfig(
                speaker_id="counselor",
                role="counselor",
                display_name="カウンセラー",
            ),
            "client_a": ParticipantConfig(
                speaker_id="client_a",
                role="client",
                display_name="クライアントA",
            ),
            "client_b": ParticipantConfig(
                speaker_id="client_b",
                role="client",
                display_name="クライアントB",
            ),
        },
        fixed_speaker_sequence=("counselor", "client_a", "counselor", "client_b"),
    )
    controller = TurnController(
        config=config,
        agents={
            "counselor": FakeAgent("counselor", "c"),
            "client_a": FakeAgent("client_a", "a"),
            "client_b": FakeAgent("client_b", "b"),
        },
    )

    assert [controller.speaker_for_turn(turn_id) for turn_id in range(1, 7)] == [
        "counselor",
        "client_a",
        "counselor",
        "client_b",
        "counselor",
        "client_a",
    ]


def test_turn_controller_does_not_transcribe_listener_stt_queues_for_generated_text() -> None:
    class FailingQueueStt:
        def __init__(self) -> None:
            self.calls = 0

        async def transcribe_from_queue(self, *, session_id, turn_id, speaker, queue):
            _ = session_id, turn_id, speaker, queue
            self.calls += 1
            raise AssertionError("STT should not run during generated-text turns")

    async def scenario() -> None:
        config = RuntimeConfig(
            session_id="session_three_listener_stt_test",
            participants={
                "counselor": ParticipantConfig(
                    speaker_id="counselor",
                    role="counselor",
                    display_name="カウンセラー",
                ),
                "client_a": ParticipantConfig(
                    speaker_id="client_a",
                    role="client",
                    display_name="クライアントA",
                ),
                "client_b": ParticipantConfig(
                    speaker_id="client_b",
                    role="client",
                    display_name="クライアントB",
                ),
            },
            fixed_speaker_sequence=("counselor", "client_b", "counselor", "client_a"),
        )
        stt = FailingQueueStt()
        controller = TurnController(
            config=config,
            agents={
                "counselor": FakeAgent("counselor", "カウンセラー応答"),
                "client_a": FakeAgent("client_a", "A応答"),
                "client_b": FakeAgent("client_b", "B応答"),
            },
            stt=stt,
        )
        audio_bus = AudioBus()
        audio_bus.subscribe_listener_stt("counselor")

        turn0 = await asyncio.wait_for(
            controller.run_scripted_audio_turn(
                turn_id=0,
                speaker="client_a",
                text="Aの初回発話です。",
                audio_bus=audio_bus,
            ),
            timeout=1.0,
        )
        audio_bus.subscribe_listener_stt("client_b")
        await asyncio.wait_for(
            controller.run_turn(
                turn_id=1,
                input_transcript=turn0.stt_final_transcript or "",
                audio_bus=audio_bus,
            ),
            timeout=1.0,
        )
        await audio_bus.close()

        assert turn0.stt_final_transcript == "Aの初回発話です。"
        assert stt.calls == 0

    asyncio.run(scenario())


def test_turn_controller_reports_invalid_fixed_speaker_sequence() -> None:
    controller = TurnController(
        config=RuntimeConfig(fixed_speaker_sequence=("counselor", "client_b")),
        agents={
            "counselor": FakeAgent("counselor", "a"),
            "client": FakeAgent("client", "b"),
        },
    )

    with pytest.raises(SpeakerSelectionError, match="unknown speaker"):
        controller.speaker_for_turn(1)


def test_turn_controller_applies_configured_speaker_audio_gain_before_delivery() -> None:
    pcm = b"".join(
        int(sample).to_bytes(2, byteorder="little", signed=True)
        for sample in [10000, -10000, 32767, -32768]
    )

    class FixedPcmTTS:
        async def synthesize(
            self,
            *,
            session_id: str,
            turn_id: int,
            speaker: str,
            text_chunks,
            sample_rate: int,
            sample_width_bits: int,
            channels: int,
            delivery_mode: AudioDeliveryMode,
        ):
            _ = list(text_chunks)
            yield AudioChunk(
                session_id=session_id,
                turn_id=turn_id,
                speaker=speaker,
                chunk_index=0,
                pcm=pcm,
                sample_rate=sample_rate,
                sample_width_bits=sample_width_bits,
                channels=channels,
                duration_ms=1,
                delivery_mode=delivery_mode,
            )

    class CapturingQueueStt:
        def __init__(self) -> None:
            self.received_pcm = b""

        async def transcribe_from_queue(self, *, session_id, turn_id, speaker, queue):
            from counseling_voice_demo.runtime.streaming_stt import iter_turn_audio_chunks

            async for chunk in iter_turn_audio_chunks(
                queue,
                session_id=session_id,
                turn_id=turn_id,
                speaker=speaker,
            ):
                self.received_pcm += chunk.pcm
            return [], TranscriptEvent(
                session_id=session_id,
                turn_id=turn_id,
                speaker=speaker,
                transcript_type="final",
                text="captured",
            )

    async def scenario() -> None:
        stt = CapturingQueueStt()
        controller = TurnController(
            config=RuntimeConfig(
                session_id="session_speaker_gain_test",
                max_turns=1,
                speaker_audio_gains={"counselor": 0.5, "client": 1.0},
            ),
            agents={
                "counselor": FakeAgent("counselor", "受け止めました。"),
                "client": FakeAgent("client", "unused"),
            },
            tts=FixedPcmTTS(),
            stt=stt,
        )
        audio_bus = AudioBus()

        await controller.run_turn(
            turn_id=1,
            input_transcript="家族のことで困っています。",
            audio_bus=audio_bus,
        )
        monitor_chunk = await first_monitor_audio_chunk(audio_bus)
        await audio_bus.close()

        expected_pcm = b"".join(
            int(sample).to_bytes(2, byteorder="little", signed=True)
            for sample in [5000, -5000, 16384, -16384]
        )
        assert stt.received_pcm == b""
        assert monitor_chunk.pcm == expected_pcm

    asyncio.run(scenario())


def test_turn_controller_limits_speaker_audio_gain_to_avoid_pcm_clipping() -> None:
    pcm = b"".join(
        int(sample).to_bytes(2, byteorder="little", signed=True)
        for sample in [12000, -12000, 30000, -30000]
    )

    class FixedPcmTTS:
        async def synthesize(
            self,
            *,
            session_id: str,
            turn_id: int,
            speaker: str,
            text_chunks,
            sample_rate: int,
            sample_width_bits: int,
            channels: int,
            delivery_mode: AudioDeliveryMode,
        ):
            _ = list(text_chunks)
            yield AudioChunk(
                session_id=session_id,
                turn_id=turn_id,
                speaker=speaker,
                chunk_index=0,
                pcm=pcm,
                sample_rate=sample_rate,
                sample_width_bits=sample_width_bits,
                channels=channels,
                duration_ms=1,
                delivery_mode=delivery_mode,
            )

    class CapturingQueueStt:
        async def transcribe_from_queue(self, *, session_id, turn_id, speaker, queue):
            from counseling_voice_demo.runtime.streaming_stt import iter_turn_audio_chunks

            chunks = [
                chunk
                async for chunk in iter_turn_audio_chunks(
                    queue,
                    session_id=session_id,
                    turn_id=turn_id,
                    speaker=speaker,
                )
            ]
            return [], TranscriptEvent(
                session_id=session_id,
                turn_id=turn_id,
                speaker=speaker,
                transcript_type="final",
                text="captured",
            )

    async def scenario() -> None:
        controller = TurnController(
            config=RuntimeConfig(
                session_id="session_speaker_limiter_test",
                max_turns=1,
                speaker_audio_gains={"counselor": 2.4, "client": 1.0},
            ),
            agents={
                "counselor": FakeAgent("counselor", "受け止めました。"),
                "client": FakeAgent("client", "unused"),
            },
            tts=FixedPcmTTS(),
            stt=CapturingQueueStt(),
        )
        audio_bus = AudioBus()

        await controller.run_turn(
            turn_id=1,
            input_transcript="家族のことで困っています。",
            audio_bus=audio_bus,
        )
        monitor_chunk = await first_monitor_audio_chunk(audio_bus)
        await audio_bus.close()
        samples = [
            int.from_bytes(
                monitor_chunk.pcm[offset : offset + 2],
                byteorder="little",
                signed=True,
            )
            for offset in range(0, len(monitor_chunk.pcm), 2)
        ]

        assert max(samples) == 32767
        assert min(samples) == -32767
        assert 32767 not in samples[:2]
        assert -32768 not in samples

    asyncio.run(scenario())


def test_speaker_audio_processing_declicks_isolated_pcm_spikes_after_gain() -> None:
    def pcm_from_samples(samples: list[int]) -> bytes:
        return b"".join(
            sample.to_bytes(2, byteorder="little", signed=True)
            for sample in samples
        )

    def samples_from_pcm(pcm: bytes) -> list[int]:
        return [
            int.from_bytes(pcm[offset : offset + 2], byteorder="little", signed=True)
            for offset in range(0, len(pcm), 2)
        ]

    samples = ([0] * 20) + [100, 4200, -5664, 3070, 200] + ([0] * 15)
    chunk = AudioChunk(
        session_id="session_declick_test",
        turn_id=0,
        speaker="client",
        chunk_index=0,
        pcm=pcm_from_samples(samples),
        sample_rate=24000,
        sample_width_bits=16,
        channels=1,
        duration_ms=1,
        delivery_mode=AudioDeliveryMode.REAL_TIME,
    )

    cleaned = _apply_speaker_audio_gain(chunk, gain=2.0)

    cleaned_samples = samples_from_pcm(cleaned.pcm)
    assert cleaned_samples[20:25] == [2200, 1418, -2029, 338, 400]
    assert cleaned_samples[:20] == [0] * 20
    assert cleaned_samples[25:] == [0] * 15


def test_turn_controller_accepts_async_stt_without_expected_text_argument() -> None:
    class AsyncStt:
        def __init__(self) -> None:
            self.calls = 0

        async def transcribe(self, *, session_id, turn_id, speaker, chunks):
            self.calls += 1
            chunk_count = len(chunks)
            final = TranscriptEvent(
                session_id=session_id,
                turn_id=turn_id,
                speaker=speaker,
                transcript_type="final",
                text=f"async final from {chunk_count} chunks",
            )
            return [], final

    async def scenario() -> None:
        config = RuntimeConfig(session_id="session_async_stt_test", max_turns=1)
        agents = {
            "counselor": FakeAgent("counselor", "受け止めました。{input_transcript}"),
            "client": FakeAgent("client", "話してみます。{input_transcript}"),
        }
        stt = AsyncStt()
        controller = TurnController(config=config, agents=agents, stt=stt)
        audio_bus = AudioBus()

        state = await controller.run_turn(
            turn_id=1,
            input_transcript="家族のことで困っています。",
            audio_bus=audio_bus,
        )
        await audio_bus.close()

        assert stt.calls == 0
        assert state.stt_final_transcript == state.generated_text
        assert state.can_advance is True

    asyncio.run(scenario())


def test_turn_controller_starts_tts_before_streaming_agent_finishes() -> None:
    class StreamingAgent:
        def __init__(self) -> None:
            self.finished = False

        async def stream_generate(self, *, input_transcript: str, turn_id: int):
            yield "最初の文。"
            await asyncio.sleep(0)
            self.finished = True
            yield "次の文。"

    class RecordingTTS:
        def __init__(self, agent: StreamingAgent) -> None:
            self.agent = agent
            self.agent_finished_at_first_tts_request: bool | None = None
            self.text_chunks: list[str] = []

        async def synthesize(
            self,
            *,
            session_id: str,
            turn_id: int,
            speaker: str,
            text_chunks,
            sample_rate: int,
            sample_width_bits: int,
            channels: int,
            delivery_mode: AudioDeliveryMode,
        ):
            if self.agent_finished_at_first_tts_request is None:
                self.agent_finished_at_first_tts_request = self.agent.finished
            for text_chunk in text_chunks:
                self.text_chunks.append(text_chunk)
                yield AudioChunk(
                    session_id=session_id,
                    turn_id=turn_id,
                    speaker=speaker,
                    chunk_index=0,
                    pcm=f"{speaker}:{turn_id}:0:{text_chunk}".encode("utf-8"),
                    sample_rate=sample_rate,
                    sample_width_bits=sample_width_bits,
                    channels=channels,
                    duration_ms=1,
                    delivery_mode=delivery_mode,
                )

    async def scenario() -> None:
        config = RuntimeConfig(session_id="session_streaming_agent_test", max_turns=1)
        agent = StreamingAgent()
        tts = RecordingTTS(agent)
        controller = TurnController(
            config=config,
            agents={
                "counselor": agent,
                "client": FakeAgent("client", "unused"),
            },
            tts=tts,
        )
        audio_bus = AudioBus()

        state = await controller.run_turn(
            turn_id=1,
            input_transcript="家族のことで困っています。",
            audio_bus=audio_bus,
        )
        await audio_bus.close()

        assert tts.agent_finished_at_first_tts_request is False
        assert tts.text_chunks == ["最初の文。", "次の文。"]
        assert state.generated_text == "最初の文。次の文。"
        assert state.stt_final_transcript == "最初の文。次の文。"

    asyncio.run(scenario())


def test_turn_controller_keeps_reading_llm_while_tts_is_in_flight() -> None:
    class StreamingAgent:
        def __init__(self) -> None:
            self.finished_reading = asyncio.Event()

        async def stream_generate(self, *, input_transcript: str, turn_id: int):
            yield "最初の文。"
            await asyncio.sleep(0)
            self.finished_reading.set()
            yield "次の文。"

    class BlockingTTS:
        def __init__(self) -> None:
            self.started = asyncio.Event()
            self.release = asyncio.Event()
            self.text_chunks: list[str] = []

        async def synthesize(
            self,
            *,
            session_id: str,
            turn_id: int,
            speaker: str,
            text_chunks,
            sample_rate: int,
            sample_width_bits: int,
            channels: int,
            delivery_mode: AudioDeliveryMode,
        ):
            for text_chunk in text_chunks:
                self.text_chunks.append(text_chunk)
                self.started.set()
                await self.release.wait()
                yield AudioChunk(
                    session_id=session_id,
                    turn_id=turn_id,
                    speaker=speaker,
                    chunk_index=0,
                    pcm=f"{speaker}:{turn_id}:0:{text_chunk}".encode("utf-8"),
                    sample_rate=sample_rate,
                    sample_width_bits=sample_width_bits,
                    channels=channels,
                    duration_ms=1,
                    delivery_mode=delivery_mode,
                )

    async def scenario() -> None:
        config = RuntimeConfig(session_id="session_parallel_tts_test", max_turns=1)
        agent = StreamingAgent()
        tts = BlockingTTS()
        controller = TurnController(
            config=config,
            agents={
                "counselor": agent,
                "client": FakeAgent("client", "unused"),
            },
            tts=tts,
        )
        audio_bus = AudioBus()

        task = asyncio.create_task(
            controller.run_turn(
                turn_id=1,
                input_transcript="家族のことで困っています。",
                audio_bus=audio_bus,
            )
        )
        await asyncio.wait_for(tts.started.wait(), timeout=1.0)
        await asyncio.wait_for(agent.finished_reading.wait(), timeout=1.0)

        assert tts.text_chunks == ["最初の文。"]

        tts.release.set()
        state = await asyncio.wait_for(task, timeout=1.0)
        await audio_bus.close()

        assert tts.text_chunks == ["最初の文。", "次の文。"]
        assert state.generated_text == "最初の文。次の文。"
        assert state.stt_final_transcript == "最初の文。次の文。"

    asyncio.run(scenario())


def test_turn_controller_uses_configured_tts_text_chunk_thresholds() -> None:
    class StreamingAgent:
        async def stream_generate(self, *, input_transcript: str, turn_id: int):
            yield "今日は家族との距離が近すぎて、少し息苦しいです。"

    class RecordingTTS:
        def __init__(self) -> None:
            self.text_chunks: list[str] = []

        async def synthesize(
            self,
            *,
            session_id: str,
            turn_id: int,
            speaker: str,
            text_chunks,
            sample_rate: int,
            sample_width_bits: int,
            channels: int,
            delivery_mode: AudioDeliveryMode,
        ):
            for text_chunk in text_chunks:
                self.text_chunks.append(text_chunk)
                yield AudioChunk(
                    session_id=session_id,
                    turn_id=turn_id,
                    speaker=speaker,
                    chunk_index=0,
                    pcm=f"{speaker}:{turn_id}:0:{text_chunk}".encode("utf-8"),
                    sample_rate=sample_rate,
                    sample_width_bits=sample_width_bits,
                    channels=channels,
                    duration_ms=1,
                    delivery_mode=delivery_mode,
                )

    async def scenario() -> None:
        tts = RecordingTTS()
        controller = TurnController(
            config=RuntimeConfig(
                session_id="session_tts_chunk_threshold_test",
                max_turns=1,
                tts_chunk_comma_min_chars=12,
                tts_chunk_soft_max_chars=24,
            ),
            agents={
                "counselor": StreamingAgent(),
                "client": FakeAgent("client", "unused"),
            },
            tts=tts,
        )
        audio_bus = AudioBus()

        await controller.run_turn(
            turn_id=1,
            input_transcript="家族のことで困っています。",
            audio_bus=audio_bus,
        )
        await audio_bus.close()

        assert tts.text_chunks == ["今日は家族との距離が近すぎて、", "少し息苦しいです。"]

    asyncio.run(scenario())


def test_turn_controller_can_use_realtime_speech_agent_without_rest_tts() -> None:
    class RealtimeSpeechAgent:
        def __init__(self) -> None:
            self.calls: list[dict] = []

        async def stream_audio_response(
            self,
            *,
            session_id: str,
            turn_id: int,
            speaker: str,
            input_transcript: str,
            sample_rate: int,
            sample_width_bits: int,
            channels: int,
            delivery_mode: AudioDeliveryMode,
        ):
            self.calls.append(
                {
                    "session_id": session_id,
                    "turn_id": turn_id,
                    "speaker": speaker,
                    "input_transcript": input_transcript,
                    "delivery_mode": delivery_mode,
                }
            )
            yield {"text_delta": "はい、"}
            yield {
                "audio_chunk": AudioChunk(
                    session_id=session_id,
                    turn_id=turn_id,
                    speaker=speaker,
                    chunk_index=7,
                    pcm=b"first",
                    sample_rate=sample_rate,
                    sample_width_bits=sample_width_bits,
                    channels=channels,
                    duration_ms=1,
                    delivery_mode=delivery_mode,
                )
            }
            yield {
                "text_delta": "短く返します。",
                "audio_chunk": AudioChunk(
                    session_id=session_id,
                    turn_id=turn_id,
                    speaker=speaker,
                    chunk_index=8,
                    pcm=b"second",
                    sample_rate=sample_rate,
                    sample_width_bits=sample_width_bits,
                    channels=channels,
                    duration_ms=1,
                    delivery_mode=delivery_mode,
                ),
            }

    class FailingTTS:
        async def synthesize(self, **kwargs):
            raise AssertionError("REST TTS path should not be used")

    class QueueStt:
        def __init__(self) -> None:
            self.received_chunk_indexes: list[int] = []
            self.queue = None

        async def transcribe_from_queue(self, *, session_id, turn_id, speaker, queue):
            from counseling_voice_demo.runtime.streaming_stt import iter_turn_audio_chunks

            self.queue = queue
            async for chunk in iter_turn_audio_chunks(
                queue,
                session_id=session_id,
                turn_id=turn_id,
                speaker=speaker,
            ):
                self.received_chunk_indexes.append(chunk.chunk_index)
            final = TranscriptEvent(
                session_id=session_id,
                turn_id=turn_id,
                speaker=speaker,
                transcript_type="final",
                text="realtime speech final transcript",
            )
            return [], final

    async def scenario() -> None:
        agent = RealtimeSpeechAgent()
        stt = QueueStt()
        controller = TurnController(
            config=RuntimeConfig(session_id="session_realtime_speech_turn_test", max_turns=1),
            agents={
                "counselor": agent,
                "client": FakeAgent("client", "unused"),
            },
            tts=FailingTTS(),
            stt=stt,
        )
        audio_bus = AudioBus()
        audio_bus.subscribe_listener_stt("client")

        state = await controller.run_turn(
            turn_id=1,
            input_transcript="家族のことで困っています。",
            audio_bus=audio_bus,
        )
        await audio_bus.close()
        monitor_items: list[object] = []
        while not audio_bus.monitor_queue.empty():
            monitor_items.append(audio_bus.monitor_queue.get_nowait())
        monitor_transcripts = [
            item for item in monitor_items if isinstance(item, TranscriptEvent)
        ]

        assert agent.calls == [
            {
                "session_id": "session_realtime_speech_turn_test",
                "turn_id": 1,
                "speaker": "counselor",
                "input_transcript": "家族のことで困っています。",
                "delivery_mode": AudioDeliveryMode.ACCELERATED,
            }
        ]
        assert [event.transcript_type for event in monitor_transcripts] == [
            "delta",
            "delta",
            "generated_final",
        ]
        assert [event.text for event in monitor_transcripts] == [
            "はい、",
            "短く返します。",
            "はい、短く返します。",
        ]
        assert state.generated_text == "はい、短く返します。"
        assert state.tts_done is True
        assert state.audio_delivered_to_stt is True
        assert stt.queue is None
        assert stt.received_chunk_indexes == []
        assert state.stt_final_transcript == "はい、短く返します。"
        assert state.can_advance is True

    asyncio.run(scenario())


def test_turn_controller_logs_sent_response_instructions(tmp_path) -> None:
    class RealtimeSpeechAgent:
        async def stream_audio_response(
            self,
            *,
            session_id: str,
            turn_id: int,
            speaker: str,
            input_transcript: str,
            sample_rate: int,
            sample_width_bits: int,
            channels: int,
            delivery_mode: AudioDeliveryMode,
            response_instructions_observer=None,
        ):
            _ = (
                input_transcript,
                sample_rate,
                sample_width_bits,
                channels,
                delivery_mode,
            )
            if response_instructions_observer is not None:
                await response_instructions_observer(
                    ResponseCreateInstructions(
                        session_id=session_id,
                        turn_id=turn_id,
                        speaker=speaker,
                        session_instructions="固定システムプロンプト",
                        participant_instructions="カウンセラープリセット",
                        additional_instructions="ターン固有指示",
                        resolved_instructions="送信済みinstructions",
                        repeat_session_instructions=True,
                        response_input=[
                            {
                                "type": "message",
                                "role": "assistant",
                                "content": [
                                    {
                                        "type": "output_text",
                                        "text": "本人が実際に話した内容",
                                    }
                                ],
                            }
                        ],
                    )
                )
            yield {"text_delta": "短く返答します。"}

    class FailingTTS:
        async def synthesize(self, **kwargs):
            raise AssertionError("REST TTS path should not be used")

    async def scenario() -> None:
        logger = AsyncLogger(
            sessions_dir=tmp_path,
            session_id="session_instruction_controller_test",
        )
        await logger.start()
        controller = TurnController(
            config=RuntimeConfig(
                session_id="session_instruction_controller_test",
                max_turns=1,
            ),
            agents={
                "counselor": RealtimeSpeechAgent(),
                "client": FakeAgent("client", "unused"),
            },
            tts=FailingTTS(),
        )
        audio_bus = AudioBus()

        await controller.run_turn(
            turn_id=1,
            input_transcript="前の発話です。",
            audio_bus=audio_bus,
            logger=logger,
        )
        await audio_bus.close()
        await logger.close()

        prompt_path = (
            tmp_path
            / "session_instruction_controller_test"
            / "internal"
            / "prompts"
            / "response_instructions.jsonl"
        )
        record = json.loads(prompt_path.read_text(encoding="utf-8").splitlines()[0])
        assert record["resolved_instructions"] == "送信済みinstructions"
        assert record["response_input"] == [
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "本人が実際に話した内容"}],
            }
        ]
        assert (
            record["instructions_sha256"]
            == hashlib.sha256("送信済みinstructions".encode("utf-8")).hexdigest()
        )

    asyncio.run(scenario())


def test_turn_controller_buffers_realtime_audio_until_text_delta() -> None:
    class RealtimeSpeechAgent:
        async def stream_audio_response(
            self,
            *,
            session_id: str,
            turn_id: int,
            speaker: str,
            input_transcript: str,
            sample_rate: int,
            sample_width_bits: int,
            channels: int,
            delivery_mode: AudioDeliveryMode,
        ):
            _ = input_transcript
            yield {
                "audio_chunk": AudioChunk(
                    session_id=session_id,
                    turn_id=turn_id,
                    speaker=speaker,
                    chunk_index=5,
                    pcm=b"first",
                    sample_rate=sample_rate,
                    sample_width_bits=sample_width_bits,
                    channels=channels,
                    duration_ms=10,
                    delivery_mode=delivery_mode,
                )
            }
            yield {"text_delta": "はい、"}
            yield {
                "audio_chunk": AudioChunk(
                    session_id=session_id,
                    turn_id=turn_id,
                    speaker=speaker,
                    chunk_index=6,
                    pcm=b"second",
                    sample_rate=sample_rate,
                    sample_width_bits=sample_width_bits,
                    channels=channels,
                    duration_ms=10,
                    delivery_mode=delivery_mode,
                )
            }
            yield {"text_delta": "続けます。"}

    class FailingTTS:
        async def synthesize(self, **kwargs):
            raise AssertionError("REST TTS path should not be used")

    async def scenario() -> None:
        controller = TurnController(
            config=RuntimeConfig(
                session_id="session_realtime_audio_buffer_test",
                max_turns=1,
            ),
            agents={
                "counselor": RealtimeSpeechAgent(),
                "client": FakeAgent("client", "unused"),
            },
            tts=FailingTTS(),
        )
        audio_bus = AudioBus()

        state = await controller.run_turn(
            turn_id=1,
            input_transcript="相談したいです。",
            audio_bus=audio_bus,
        )
        monitor_items: list[object] = []
        while not audio_bus.monitor_queue.empty():
            monitor_items.append(audio_bus.monitor_queue.get_nowait())
        await audio_bus.close()

        first_delta_index = next(
            index
            for index, item in enumerate(monitor_items)
            if isinstance(item, TranscriptEvent)
            and item.transcript_type == "delta"
            and item.text == "はい、"
        )
        first_audio_index = next(
            index
            for index, item in enumerate(monitor_items)
            if isinstance(item, AudioChunk)
        )
        audio_chunks = [item for item in monitor_items if isinstance(item, AudioChunk)]

        assert first_delta_index < first_audio_index
        assert [chunk.chunk_index for chunk in audio_chunks] == [0, 1]
        assert [chunk.pcm for chunk in audio_chunks] == [b"first", b"second"]
        assert state.generated_text == "はい、続けます。"
        assert state.can_advance is True

    asyncio.run(scenario())


def test_turn_controller_plays_realtime_audio_without_transcript() -> None:
    class RealtimeSpeechAgent:
        async def stream_audio_response(
            self,
            *,
            session_id: str,
            turn_id: int,
            speaker: str,
            input_transcript: str,
            sample_rate: int,
            sample_width_bits: int,
            channels: int,
            delivery_mode: AudioDeliveryMode,
        ):
            _ = input_transcript
            yield {
                "audio_chunk": AudioChunk(
                    session_id=session_id,
                    turn_id=turn_id,
                    speaker=speaker,
                    chunk_index=0,
                    pcm=b"orphan",
                    sample_rate=sample_rate,
                    sample_width_bits=sample_width_bits,
                    channels=channels,
                    duration_ms=800,
                    delivery_mode=delivery_mode,
                )
            }

    class FailingTTS:
        async def synthesize(self, **kwargs):
            raise AssertionError("REST TTS path should not be used")

    async def scenario() -> None:
        controller = TurnController(
            config=RuntimeConfig(
                session_id="session_realtime_orphan_audio_test",
                max_turns=1,
            ),
            agents={
                "counselor": RealtimeSpeechAgent(),
                "client": FakeAgent("client", "unused"),
            },
            tts=FailingTTS(),
        )
        audio_bus = AudioBus()

        state = await controller.run_turn(
            turn_id=1,
            input_transcript="相談したいです。",
            audio_bus=audio_bus,
        )
        monitor_items: list[object] = []
        while not audio_bus.monitor_queue.empty():
            monitor_items.append(audio_bus.monitor_queue.get_nowait())
        await audio_bus.close()

        audio_chunks = [item for item in monitor_items if isinstance(item, AudioChunk)]

        assert [chunk.pcm for chunk in audio_chunks] == [b"orphan"]
        assert any(isinstance(item, EndOfAudio) for item in monitor_items)
        assert not any(isinstance(item, TranscriptEvent) for item in monitor_items)
        assert state.generated_text == ""
        assert state.tts_done is True
        assert state.audio_delivered_to_stt is True
        assert state.stt_final_transcript == ""
        assert state.audio_log_path is None
        assert state.can_advance is True

    asyncio.run(scenario())


def test_turn_controller_uses_realtime_speech_agent_for_scripted_audio_turn() -> None:
    class RealtimeSpeechAgent:
        def __init__(self) -> None:
            self.calls: list[dict] = []

        async def stream_audio_response(
            self,
            *,
            session_id: str,
            turn_id: int,
            speaker: str,
            input_transcript: str,
            additional_instruction: str | None = None,
            format_input: bool = True,
            sample_rate: int,
            sample_width_bits: int,
            channels: int,
            delivery_mode: AudioDeliveryMode,
        ):
            self.calls.append(
                {
                    "session_id": session_id,
                    "turn_id": turn_id,
                    "speaker": speaker,
                    "input_transcript": input_transcript,
                    "additional_instruction": additional_instruction,
                    "format_input": format_input,
                    "delivery_mode": delivery_mode,
                }
            )
            yield {"text_delta": "初回相談です。"}
            yield {
                "audio_chunk": AudioChunk(
                    session_id=session_id,
                    turn_id=turn_id,
                    speaker=speaker,
                    chunk_index=3,
                    pcm=b"\x01\x00" * 120,
                    sample_rate=sample_rate,
                    sample_width_bits=sample_width_bits,
                    channels=channels,
                    duration_ms=5,
                    delivery_mode=delivery_mode,
                )
            }

    class FailingTTS:
        async def synthesize(self, **kwargs):
            raise AssertionError("REST TTS path should not be used for realtime scripted audio")

    class QueueStt:
        def __init__(self) -> None:
            self.received_chunk_indexes: list[int] = []

        async def transcribe_from_queue(self, *, session_id, turn_id, speaker, queue):
            from counseling_voice_demo.runtime.streaming_stt import iter_turn_audio_chunks

            async for chunk in iter_turn_audio_chunks(
                queue,
                session_id=session_id,
                turn_id=turn_id,
                speaker=speaker,
            ):
                self.received_chunk_indexes.append(chunk.chunk_index)
            return [], TranscriptEvent(
                session_id=session_id,
                turn_id=turn_id,
                speaker=speaker,
                transcript_type="final",
                text="初回相談です。",
            )

    async def scenario() -> None:
        agent = RealtimeSpeechAgent()
        stt = QueueStt()
        controller = TurnController(
            config=RuntimeConfig(session_id="session_realtime_scripted_test", max_turns=1),
            agents={
                "counselor": FakeAgent("counselor", "unused"),
                "client": agent,
            },
            tts=FailingTTS(),
            stt=stt,
        )
        audio_bus = AudioBus()

        state = await controller.run_scripted_audio_turn(
            turn_id=0,
            speaker="client",
            text="初回相談です。",
            audio_bus=audio_bus,
        )
        monitor_chunk = await first_monitor_audio_chunk(audio_bus)
        await audio_bus.close()

        assert agent.calls == [
            {
                "session_id": "session_realtime_scripted_test",
                "turn_id": 0,
                "speaker": "client",
                "input_transcript": "初回相談です。",
                "additional_instruction": (
                    "【読み上げ指示】入力された本文を、一字一句そのまま音声として読み上げてください。"
                    "本文以外の挨拶、相づち、説明、言い換え、追加の一文は入れないでください。"
                ),
                "format_input": False,
                "delivery_mode": AudioDeliveryMode.ACCELERATED,
            }
        ]
        assert stt.received_chunk_indexes == []
        assert monitor_chunk.pace_after_delivery is False
        assert state.generated_text == "初回相談です。"
        assert state.stt_final_transcript == "初回相談です。"
        assert state.can_advance is True

    asyncio.run(scenario())


def test_turn_controller_does_not_use_stt_queue_consumer_for_generated_final() -> None:
    class QueueStt:
        def __init__(self) -> None:
            self.received_chunk_indexes: list[int] = []
            self.used_chunk_iterable_fallback = False
            self.queue = None

        async def transcribe_from_queue(self, *, session_id, turn_id, speaker, queue):
            from counseling_voice_demo.runtime.streaming_stt import iter_turn_audio_chunks

            self.queue = queue
            async for chunk in iter_turn_audio_chunks(
                queue,
                session_id=session_id,
                turn_id=turn_id,
                speaker=speaker,
            ):
                self.received_chunk_indexes.append(chunk.chunk_index)
            final = TranscriptEvent(
                session_id=session_id,
                turn_id=turn_id,
                speaker=speaker,
                transcript_type="final",
                text="queue final transcript",
            )
            return [], final

        async def transcribe(self, *, session_id, turn_id, speaker, chunks):
            self.used_chunk_iterable_fallback = True
            raise AssertionError("chunk iterable STT fallback should not be used")

    async def scenario() -> None:
        config = RuntimeConfig(session_id="session_queue_stt_test", max_turns=1)
        agents = {
            "counselor": FakeAgent("counselor", "受け止めました。{input_transcript}"),
            "client": FakeAgent("client", "unused"),
        }
        stt = QueueStt()
        controller = TurnController(config=config, agents=agents, stt=stt)
        audio_bus = AudioBus()

        state = await controller.run_turn(
            turn_id=1,
            input_transcript="家族のことで困っています。",
            audio_bus=audio_bus,
        )
        await audio_bus.close()

        assert stt.used_chunk_iterable_fallback is False
        assert stt.queue is None
        assert stt.received_chunk_indexes == []
        assert state.stt_final_transcript == state.generated_text
        assert state.can_advance is True

    asyncio.run(scenario())
