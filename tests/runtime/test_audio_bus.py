from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone

from counseling_voice_demo.runtime.audio_bus import AudioBus
from counseling_voice_demo.runtime.models import (
    AudioChunk,
    AudioDeliveryMode,
    EndOfAudio,
    TranscriptEvent,
)


@dataclass(frozen=True)
class FakeAudioChunk:
    session_id: str
    turn_id: int
    speaker: str
    chunk_index: int
    pcm: bytes
    sample_rate: int = 24000
    sample_width_bits: int = 16
    channels: int = 1
    duration_ms: int = 20
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


def make_chunk(chunk_index: int, *, turn_id: int = 7) -> FakeAudioChunk:
    return FakeAudioChunk(
        session_id="session-1",
        turn_id=turn_id,
        speaker="counselor",
        chunk_index=chunk_index,
        pcm=f"pcm-{chunk_index}".encode("ascii"),
    )


async def drain(queue: asyncio.Queue[object], count: int) -> list[object]:
    return [await asyncio.wait_for(queue.get(), timeout=1.0) for _ in range(count)]


def test_one_chunk_reaches_all_three_queues() -> None:
    async def scenario() -> None:
        monitor_queue: asyncio.Queue[object] = asyncio.Queue()
        opponent_stt_queue: asyncio.Queue[object] = asyncio.Queue()
        logger_queue: asyncio.Queue[object] = asyncio.Queue()
        bus = AudioBus(
            monitor_queue=monitor_queue,
            opponent_stt_queue=opponent_stt_queue,
            logger_queue=logger_queue,
            end_event=None,
        )
        chunk = make_chunk(0)

        await bus.publish(chunk)
        await bus.flush()

        assert await monitor_queue.get() is chunk
        assert await opponent_stt_queue.get() is chunk
        assert await logger_queue.get() is chunk

        await bus.close()

    asyncio.run(scenario())


def test_real_time_mode_paces_after_stt_delivery() -> None:
    async def scenario() -> None:
        sleeps = []

        async def sleeper(seconds: float) -> None:
            sleeps.append(seconds)

        opponent_stt_queue: asyncio.Queue[object] = asyncio.Queue()
        bus = AudioBus(
            opponent_stt_queue=opponent_stt_queue,
            audio_delivery_mode=AudioDeliveryMode.REAL_TIME,
            sleeper=sleeper,
            end_event=None,
        )
        chunk = make_chunk(0)

        await bus.publish(chunk)

        assert await opponent_stt_queue.get() is chunk
        assert sleeps == [0.02]
        await bus.close()

    asyncio.run(scenario())


def test_real_time_mode_can_skip_additional_pacing_per_chunk() -> None:
    async def scenario() -> None:
        sleeps = []

        async def sleeper(seconds: float) -> None:
            sleeps.append(seconds)

        opponent_stt_queue: asyncio.Queue[object] = asyncio.Queue()
        bus = AudioBus(
            opponent_stt_queue=opponent_stt_queue,
            audio_delivery_mode=AudioDeliveryMode.REAL_TIME,
            sleeper=sleeper,
            end_event=None,
        )
        chunk = make_chunk(0)
        chunk = AudioChunk(
            session_id=chunk.session_id,
            turn_id=chunk.turn_id,
            speaker=chunk.speaker,
            chunk_index=chunk.chunk_index,
            pcm=chunk.pcm,
            sample_rate=chunk.sample_rate,
            sample_width_bits=chunk.sample_width_bits,
            channels=chunk.channels,
            duration_ms=chunk.duration_ms,
            delivery_mode=AudioDeliveryMode.REAL_TIME,
            pace_after_delivery=False,
        )

        await bus.publish(chunk)

        assert await opponent_stt_queue.get() is chunk
        assert sleeps == []
        await bus.close()

    asyncio.run(scenario())


def test_multiple_chunks_keep_order_and_metadata() -> None:
    async def scenario() -> None:
        monitor_queue: asyncio.Queue[object] = asyncio.Queue()
        opponent_stt_queue: asyncio.Queue[object] = asyncio.Queue()
        logger_queue: asyncio.Queue[object] = asyncio.Queue()
        bus = AudioBus(
            monitor_queue=monitor_queue,
            opponent_stt_queue=opponent_stt_queue,
            logger_queue=logger_queue,
            end_event=None,
        )
        chunks = [make_chunk(index, turn_id=31) for index in range(4)]

        for chunk in chunks:
            await bus.publish(chunk)
        await bus.flush()

        assert await drain(monitor_queue, len(chunks)) == chunks
        stt_chunks = await drain(opponent_stt_queue, len(chunks))
        assert stt_chunks == chunks
        assert [(chunk.turn_id, chunk.chunk_index) for chunk in stt_chunks] == [
            (31, 0),
            (31, 1),
            (31, 2),
            (31, 3),
        ]
        assert await drain(logger_queue, len(chunks)) == chunks

        await bus.close()

    asyncio.run(scenario())


def test_monitor_subscribers_each_receive_all_chunks() -> None:
    async def scenario() -> None:
        bus = AudioBus(end_event=None)
        first_monitor = bus.subscribe_monitor()
        second_monitor = bus.subscribe_monitor()
        counselor_chunk = make_chunk(0, turn_id=1)
        client_chunk = FakeAudioChunk(
            session_id="session-1",
            turn_id=2,
            speaker="client",
            chunk_index=0,
            pcm=b"client-pcm",
        )

        await bus.publish(counselor_chunk)
        await bus.end_turn("session-1", 1, "counselor")
        await bus.publish(client_chunk)

        assert await first_monitor.get() is counselor_chunk
        assert isinstance(await first_monitor.get(), EndOfAudio)
        assert await first_monitor.get() is client_chunk
        assert await second_monitor.get() is counselor_chunk
        assert isinstance(await second_monitor.get(), EndOfAudio)
        assert await second_monitor.get() is client_chunk

        bus.unsubscribe_monitor(first_monitor)
        bus.unsubscribe_monitor(second_monitor)
        await bus.close()

    asyncio.run(scenario())


def test_monitor_only_event_reaches_monitor_queues_only() -> None:
    async def scenario() -> None:
        opponent_stt_queue: asyncio.Queue[object] = asyncio.Queue()
        logger_queue: asyncio.Queue[object] = asyncio.Queue()
        bus = AudioBus(
            opponent_stt_queue=opponent_stt_queue,
            logger_queue=logger_queue,
            end_event=None,
        )
        second_monitor = bus.subscribe_monitor()
        event = TranscriptEvent(
            session_id="session-1",
            turn_id=7,
            speaker="client_a",
            transcript_type="delta",
            text="途中までの表示テキスト",
        )

        await bus.publish_monitor_event(event)
        await bus.flush()

        assert await bus.monitor_queue.get() is event
        assert await second_monitor.get() is event
        assert opponent_stt_queue.empty()
        assert logger_queue.empty()

        await bus.close()

    asyncio.run(scenario())


def test_logger_only_audio_does_not_reach_monitor_or_stt_queues() -> None:
    async def scenario() -> None:
        monitor_queue: asyncio.Queue[object] = asyncio.Queue()
        opponent_stt_queue: asyncio.Queue[object] = asyncio.Queue()
        logger_queue: asyncio.Queue[object] = asyncio.Queue()
        bus = AudioBus(
            monitor_queue=monitor_queue,
            opponent_stt_queue=opponent_stt_queue,
            logger_queue=logger_queue,
            end_event=None,
        )
        chunk = FakeAudioChunk(
            session_id="session-1",
            turn_id=1,
            speaker="counselor",
            chunk_index=0,
            pcm=b"human-pcm",
        )

        await bus.publish_logger_audio(chunk)
        await bus.end_logger_turn("session-1", 1, "counselor")
        await bus.flush()

        assert await logger_queue.get() is chunk
        assert isinstance(await logger_queue.get(), EndOfAudio)
        assert monitor_queue.empty()
        assert opponent_stt_queue.empty()

        await bus.close()

    asyncio.run(scenario())


def test_listener_stt_queues_receive_audio_from_other_speakers_only() -> None:
    async def scenario() -> None:
        bus = AudioBus(end_event=None)
        counselor_stt = bus.subscribe_listener_stt("counselor")
        client_a_stt = bus.subscribe_listener_stt("client_a")
        client_b_stt = bus.subscribe_listener_stt("client_b")
        chunk = FakeAudioChunk(
            session_id="session-1",
            turn_id=3,
            speaker="client_a",
            chunk_index=0,
            pcm=b"client-a-pcm",
        )

        await bus.publish(chunk)

        assert await counselor_stt.get() is chunk
        assert await client_b_stt.get() is chunk
        assert client_a_stt.empty()

        await bus.close()

    asyncio.run(scenario())


def test_listener_stt_queues_receive_end_of_audio_from_other_speakers_only() -> None:
    async def scenario() -> None:
        bus = AudioBus()
        counselor_stt = bus.subscribe_listener_stt("counselor")
        client_a_stt = bus.subscribe_listener_stt("client_a")
        client_b_stt = bus.subscribe_listener_stt("client_b")

        await bus.end_turn("session-1", 3, "client_a")

        counselor_event = await counselor_stt.get()
        client_b_event = await client_b_stt.get()
        assert isinstance(counselor_event, EndOfAudio)
        assert isinstance(client_b_event, EndOfAudio)
        assert counselor_event.speaker == "client_a"
        assert client_b_event.speaker == "client_a"
        assert client_a_stt.empty()

        await bus.close()

    asyncio.run(scenario())


def test_close_sends_end_event_to_all_listener_stt_queues() -> None:
    async def scenario() -> None:
        end_event = object()
        bus = AudioBus(end_event=end_event)
        counselor_stt = bus.subscribe_listener_stt("counselor")
        client_a_stt = bus.subscribe_listener_stt("client_a")

        await bus.close()

        assert await counselor_stt.get() is end_event
        assert await client_a_stt.get() is end_event

    asyncio.run(scenario())


def test_unsubscribed_listener_stt_queue_no_longer_receives_audio() -> None:
    async def scenario() -> None:
        bus = AudioBus(end_event=None)
        client_b_stt = bus.subscribe_listener_stt("client_b")
        bus.unsubscribe_listener_stt("client_b")
        chunk = FakeAudioChunk(
            session_id="session-1",
            turn_id=4,
            speaker="client_a",
            chunk_index=0,
            pcm=b"client-a-pcm",
        )

        await bus.publish(chunk)

        assert client_b_stt.empty()
        await bus.close()

    asyncio.run(scenario())


def test_logger_backpressure_does_not_block_opponent_stt_delivery() -> None:
    async def scenario() -> None:
        monitor_queue: asyncio.Queue[object] = asyncio.Queue()
        opponent_stt_queue: asyncio.Queue[object] = asyncio.Queue()
        logger_queue: asyncio.Queue[object] = asyncio.Queue(maxsize=1)
        logger_queue.put_nowait("occupied")
        bus = AudioBus(
            monitor_queue=monitor_queue,
            opponent_stt_queue=opponent_stt_queue,
            logger_queue=logger_queue,
            end_event=None,
        )
        chunk = make_chunk(0)

        await asyncio.wait_for(bus.publish(chunk), timeout=0.1)

        assert await asyncio.wait_for(opponent_stt_queue.get(), timeout=0.1) is chunk
        assert logger_queue.get_nowait() == "occupied"
        await bus.flush()
        assert await logger_queue.get() is chunk

        await bus.close()

    asyncio.run(scenario())


def test_stop_sends_end_event_to_all_queues() -> None:
    async def scenario() -> None:
        monitor_queue: asyncio.Queue[object] = asyncio.Queue()
        opponent_stt_queue: asyncio.Queue[object] = asyncio.Queue()
        logger_queue: asyncio.Queue[object] = asyncio.Queue()
        bus = AudioBus(
            monitor_queue=monitor_queue,
            opponent_stt_queue=opponent_stt_queue,
            logger_queue=logger_queue,
            end_event=None,
        )

        await bus.stop()

        assert await monitor_queue.get() is None
        assert await opponent_stt_queue.get() is None
        assert await logger_queue.get() is None

    asyncio.run(scenario())


def test_end_turn_sends_turn_specific_end_event() -> None:
    async def scenario() -> None:
        monitor_queue: asyncio.Queue[object] = asyncio.Queue()
        opponent_stt_queue: asyncio.Queue[object] = asyncio.Queue()
        logger_queue: asyncio.Queue[object] = asyncio.Queue()
        bus = AudioBus(
            monitor_queue=monitor_queue,
            opponent_stt_queue=opponent_stt_queue,
            logger_queue=logger_queue,
        )

        await bus.end_turn("session-1", 7, "counselor")

        monitor_event = await monitor_queue.get()
        stt_event = await opponent_stt_queue.get()
        await bus.flush()
        logger_event = await logger_queue.get()
        assert isinstance(monitor_event, EndOfAudio)
        assert isinstance(stt_event, EndOfAudio)
        assert isinstance(logger_event, EndOfAudio)
        assert (logger_event.session_id, logger_event.turn_id, logger_event.speaker) == (
            "session-1",
            7,
            "counselor",
        )

        await bus.close()

    asyncio.run(scenario())
