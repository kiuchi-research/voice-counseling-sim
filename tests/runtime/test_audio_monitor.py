from __future__ import annotations

import asyncio

import pytest

from counseling_voice_demo.runtime.audio_monitor import (
    audio_chunk_to_browser_message,
    audio_monitor_messages,
    end_of_audio_to_browser_message,
    transcript_event_to_browser_message,
)
from counseling_voice_demo.runtime.models import (
    AudioChunk,
    AudioDeliveryMode,
    EndOfAudio,
    TranscriptEvent,
)


def test_audio_chunk_to_browser_message_encodes_pcm_and_metadata() -> None:
    chunk = AudioChunk(
        session_id="session-monitor",
        turn_id=3,
        speaker="counselor",
        chunk_index=2,
        pcm=b"\x00\x01\xfe\xff",
        sample_rate=24000,
        sample_width_bits=16,
        channels=1,
        duration_ms=20,
        delivery_mode=AudioDeliveryMode.REAL_TIME,
    )

    message = audio_chunk_to_browser_message(chunk)

    assert message == {
        "type": "audio_chunk",
        "pcm_base64": "AAH+/w==",
        "metadata": {
            "session_id": "session-monitor",
            "turn_id": 3,
            "speaker": "counselor",
            "chunk_index": 2,
            "audio_role": "main",
            "sample_rate": 24000,
            "sample_width_bits": 16,
            "channels": 1,
            "duration_ms": 20,
            "delivery_mode": "real_time",
        },
    }


def test_end_of_audio_to_browser_message_is_turn_end_only() -> None:
    message = end_of_audio_to_browser_message(
        EndOfAudio(session_id="session-monitor", turn_id=3, speaker="counselor")
    )

    assert message == {
        "type": "turn_end",
        "metadata": {
            "session_id": "session-monitor",
            "turn_id": 3,
            "speaker": "counselor",
            "audio_role": "main",
        },
    }


def test_transcript_event_to_browser_message_uses_display_safe_metadata() -> None:
    message = transcript_event_to_browser_message(
        TranscriptEvent(
            session_id="session-monitor",
            turn_id=3,
            speaker="client_a",
            speaker_id="client_a",
            transcript_type="generated_final",
            text="今日は娘のことで相談したいです。",
            recipient_ids=("counselor", "client_b"),
            metadata={
                "speaker_display_name": "妻",
                "role": "client",
                "text_source": "generated_text",
                "private_profile": "ブラウザへ送ってはいけない情報",
            },
        )
    )

    assert message == {
        "type": "transcript_final",
        "text": "今日は娘のことで相談したいです。",
        "metadata": {
            "session_id": "session-monitor",
            "turn_id": 3,
            "speaker": "client_a",
            "speaker_id": "client_a",
            "display_name": "妻",
            "role": "client",
            "recipient_ids": ["counselor", "client_b"],
            "transcript_type": "generated_final",
            "segment_id": "session-monitor:3:client_a",
            "text_source": "generated_text",
        },
    }


def test_transcript_event_to_browser_message_supports_interrupted_timeline_item() -> None:
    message = transcript_event_to_browser_message(
        TranscriptEvent(
            session_id="session-monitor",
            turn_id=9,
            speaker="client_b",
            speaker_id="client_b",
            transcript_type="interrupted",
            text="",
            recipient_ids=("counselor", "client_a"),
            metadata={
                "speaker_display_name": "夫",
                "role": "client",
                "text_source": "generated_text",
                "reason": "human_barge_in",
            },
        )
    )

    assert message == {
        "type": "transcript_interrupted",
        "text": "",
        "metadata": {
            "session_id": "session-monitor",
            "turn_id": 9,
            "speaker": "client_b",
            "speaker_id": "client_b",
            "display_name": "夫",
            "role": "client",
            "recipient_ids": ["counselor", "client_a"],
            "transcript_type": "interrupted",
            "segment_id": "session-monitor:9:client_b",
            "text_source": "generated_text",
        },
    }


def test_transcript_event_to_browser_message_supports_invalidated_timeline_item() -> None:
    message = transcript_event_to_browser_message(
        TranscriptEvent(
            session_id="session-monitor",
            turn_id=10,
            speaker="client_a",
            speaker_id="client_a",
            transcript_type="invalidated",
            text="",
            recipient_ids=("counselor", "client_b"),
            metadata={
                "speaker_display_name": "妻",
                "role": "client",
                "text_source": "generated_text",
                "reason": "human_barge_in",
            },
        )
    )

    assert message == {
        "type": "transcript_invalidated",
        "text": "",
        "metadata": {
            "session_id": "session-monitor",
            "turn_id": 10,
            "speaker": "client_a",
            "speaker_id": "client_a",
            "display_name": "妻",
            "role": "client",
            "recipient_ids": ["counselor", "client_b"],
            "transcript_type": "invalidated",
            "segment_id": "session-monitor:10:client_a",
            "text_source": "generated_text",
        },
    }


def test_audio_monitor_messages_keeps_stream_open_after_turn_end() -> None:
    async def scenario() -> None:
        queue: asyncio.Queue[object] = asyncio.Queue()
        await queue.put(
            AudioChunk(
                session_id="session-monitor",
                turn_id=3,
                speaker="counselor",
                chunk_index=0,
                pcm=b"abc",
                duration_ms=10,
            )
        )
        await queue.put(EndOfAudio(session_id="session-monitor", turn_id=3, speaker="counselor"))
        await queue.put(
            TranscriptEvent(
                session_id="session-monitor",
                turn_id=4,
                speaker="client",
                transcript_type="delta",
                text="次の発話",
            )
        )
        await queue.put(
            AudioChunk(
                session_id="session-monitor",
                turn_id=4,
                speaker="client",
                chunk_index=0,
                pcm=b"def",
                duration_ms=10,
            )
        )
        await queue.put(EndOfAudio())

        messages = [message async for message in audio_monitor_messages(queue)]

        assert [message["type"] for message in messages] == [
            "audio_chunk",
            "turn_end",
            "transcript_delta",
            "audio_chunk",
            "stream_end",
        ]
        assert messages[0]["pcm_base64"] == "YWJj"
        assert messages[1]["metadata"] == {
            "session_id": "session-monitor",
            "turn_id": 3,
            "speaker": "counselor",
            "audio_role": "main",
        }
        assert messages[2]["text"] == "次の発話"
        assert messages[3]["pcm_base64"] == "ZGVm"
        assert messages[4]["metadata"] is None

    asyncio.run(scenario())


def test_audio_monitor_messages_treats_none_as_stream_end_and_stops() -> None:
    async def scenario() -> None:
        queue: asyncio.Queue[object] = asyncio.Queue()
        await queue.put(None)
        await queue.put(
            AudioChunk(
                session_id="session-monitor",
                turn_id=99,
                speaker="client",
                chunk_index=0,
                pcm=b"not-read",
            )
        )

        messages = [message async for message in audio_monitor_messages(queue)]

        assert messages == [{"type": "stream_end", "metadata": None}]

    asyncio.run(scenario())


def test_audio_monitor_messages_rejects_unknown_items() -> None:
    async def scenario() -> None:
        queue: asyncio.Queue[object] = asyncio.Queue()
        await queue.put(object())

        with pytest.raises(TypeError, match="unsupported audio monitor item"):
            [message async for message in audio_monitor_messages(queue)]

    asyncio.run(scenario())
