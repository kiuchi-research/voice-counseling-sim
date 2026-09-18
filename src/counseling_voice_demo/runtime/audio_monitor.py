from __future__ import annotations

import asyncio
import base64
from collections.abc import AsyncIterator
from typing import Any

from counseling_voice_demo.runtime.models import AudioChunk, EndOfAudio, TranscriptEvent

BrowserAudioMessage = dict[str, Any]


def audio_chunk_to_browser_message(chunk: AudioChunk) -> BrowserAudioMessage:
    """Format one PCM chunk for browser-side observation over WebSocket JSON."""
    return {
        "type": "audio_chunk",
        "pcm_base64": base64.b64encode(chunk.pcm).decode("ascii"),
        "metadata": {
            "session_id": chunk.session_id,
            "turn_id": chunk.turn_id,
            "speaker": chunk.speaker,
            "chunk_index": chunk.chunk_index,
            "audio_role": chunk.audio_role,
            "sample_rate": chunk.sample_rate,
            "sample_width_bits": chunk.sample_width_bits,
            "channels": chunk.channels,
            "duration_ms": chunk.duration_ms,
            "delivery_mode": chunk.delivery_mode.value,
        },
    }


def end_of_audio_to_browser_message(end_of_audio: EndOfAudio) -> BrowserAudioMessage:
    if end_of_audio.turn_id is None and end_of_audio.speaker is None:
        return {"type": "stream_end", "metadata": None}
    return {
        "type": "turn_end",
        "metadata": {
            "session_id": end_of_audio.session_id,
            "turn_id": end_of_audio.turn_id,
            "speaker": end_of_audio.speaker,
            "audio_role": end_of_audio.audio_role,
        },
    }


def transcript_event_to_browser_message(event: TranscriptEvent) -> BrowserAudioMessage:
    """Format a display-safe transcript event for the browser timeline."""
    if event.transcript_type in {"delta", "partial", "transcript_delta"}:
        message_type = "transcript_delta"
    elif event.transcript_type in {"interrupted", "transcript_interrupted"}:
        message_type = "transcript_interrupted"
    elif event.transcript_type in {"invalidated", "transcript_invalidated"}:
        message_type = "transcript_invalidated"
    else:
        message_type = "transcript_final"
    display_name = str(
        event.metadata.get("speaker_display_name")
        or event.metadata.get("display_name")
        or event.speaker_id
        or event.speaker
    )
    segment_id = str(
        event.metadata.get("segment_id")
        or f"{event.session_id}:{event.turn_id}:{event.speaker_id or event.speaker}"
    )
    metadata: dict[str, Any] = {
        "session_id": event.session_id,
        "turn_id": event.turn_id,
        "speaker": event.speaker,
        "speaker_id": event.speaker_id or event.speaker,
        "display_name": display_name,
        "role": str(event.metadata.get("role") or event.speaker),
        "recipient_ids": list(event.recipient_ids),
        "transcript_type": event.transcript_type,
        "segment_id": segment_id,
    }
    text_source = event.metadata.get("text_source")
    if text_source is not None:
        metadata["text_source"] = str(text_source)
    return {
        "type": message_type,
        "text": event.text,
        "metadata": metadata,
    }


async def audio_monitor_messages(
    queue: asyncio.Queue[object],
) -> AsyncIterator[BrowserAudioMessage]:
    while True:
        item = await queue.get()
        if isinstance(item, AudioChunk):
            yield audio_chunk_to_browser_message(item)
            continue
        if isinstance(item, EndOfAudio):
            yield end_of_audio_to_browser_message(item)
            if item.turn_id is None and item.speaker is None:
                break
            continue
        if isinstance(item, TranscriptEvent):
            yield transcript_event_to_browser_message(item)
            continue
        if item is None:
            yield {"type": "stream_end", "metadata": None}
            break
        raise TypeError(f"unsupported audio monitor item: {type(item).__name__}")
