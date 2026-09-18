from __future__ import annotations

import asyncio
import base64
import inspect
from collections.abc import AsyncIterator, Callable, Iterable, Mapping
from typing import Any, AsyncContextManager
from uuid import uuid4

from counseling_voice_demo.runtime.models import AudioChunk, EndOfAudio, TranscriptEvent
from counseling_voice_demo.runtime.protocols import TranscriptObserver


class RealtimeTranscriptionError(RuntimeError):
    def __init__(self, message: str, *, code: str | None = None) -> None:
        super().__init__(message)
        self.code = code


DEFAULT_REALTIME_TRANSCRIPTION_SERVER_VAD: dict[str, Any] = {
    "type": "server_vad",
    "threshold": 0.5,
    "prefix_padding_ms": 300,
    # Match the browser's Always silence window so brief pauses stay in one turn.
    "silence_duration_ms": 1400,
}
_USE_DEFAULT_TURN_DETECTION = object()


def build_realtime_transcription_session_update(
    model: str = "gpt-4o-transcribe",
    language: str = "ja",
    turn_detection: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "type": "session.update",
        "session": {
            "type": "transcription",
            "audio": {
                "input": {
                    "format": {
                        "type": "audio/pcm",
                        "rate": 24000,
                    },
                    "transcription": {
                        "model": model,
                        "language": language,
                    },
                    "turn_detection": turn_detection,
                },
            },
        },
    }


def build_input_audio_append_event(chunk: AudioChunk) -> dict[str, str]:
    return {
        "type": "input_audio_buffer.append",
        "audio": base64.b64encode(chunk.pcm).decode("ascii"),
    }


def build_input_audio_commit_event() -> dict[str, str]:
    return {"type": "input_audio_buffer.commit"}


async def iter_turn_audio_chunks(
    queue: Any,
    *,
    session_id: str,
    turn_id: int,
    speaker: str,
) -> AsyncIterator[AudioChunk]:
    while True:
        item = await queue.get()
        if isinstance(item, AudioChunk):
            _validate_turn_audio_item(
                item,
                session_id=session_id,
                turn_id=turn_id,
                speaker=speaker,
            )
            yield item
            continue
        if isinstance(item, EndOfAudio):
            if item.turn_id is None and item.speaker is None:
                raise RuntimeError("audio stream ended before target turn audio was committed")
            _validate_turn_end_item(
                item,
                session_id=session_id,
                turn_id=turn_id,
                speaker=speaker,
            )
            return
        if item is None:
            raise RuntimeError("audio stream ended before target turn audio was committed")
        raise TypeError(f"unsupported STT queue item: {type(item).__name__}")


def parse_realtime_transcription_event(
    event: Mapping[str, Any],
    session_id: str,
    turn_id: int,
    speaker: str,
) -> TranscriptEvent | None:
    event_type = event.get("type")
    if event_type in ("error", "conversation.item.input_audio_transcription.failed"):
        error = event.get("error")
        raise RealtimeTranscriptionError(
            _format_realtime_error_event(event),
            code=error.get("code") if isinstance(error, Mapping) else None,
        )
    if event_type == "conversation.item.input_audio_transcription.delta":
        transcript_type = "partial"
        text = event.get("delta", "")
    elif event_type == "conversation.item.input_audio_transcription.completed":
        transcript_type = "final"
        text = event.get("transcript", "")
    else:
        return None

    metadata = {
        key: event[key]
        for key in ("event_id", "item_id", "content_index")
        if key in event
    }
    return TranscriptEvent(
        session_id=session_id,
        turn_id=turn_id,
        speaker=speaker,
        transcript_type=transcript_type,
        text="" if text is None else str(text),
        metadata=metadata,
    )


class RealtimeStreamingSTT:
    def __init__(
        self,
        transport: Any,
        *,
        model: str = "gpt-4o-transcribe",
        language: str = "ja",
        turn_detection: dict[str, Any] | None = None,
    ) -> None:
        self.transport = transport
        self.model = model
        self.language = language
        self.turn_detection = turn_detection
        self._committed_item_id: str | None = None
        self._manual_commit_event_id: str | None = None

    async def configure(self) -> None:
        await self._send_json(
            build_realtime_transcription_session_update(
                model=self.model,
                language=self.language,
                turn_detection=self.turn_detection,
            )
        )

    async def append_audio(self, chunk: AudioChunk) -> None:
        await self._send_json(build_input_audio_append_event(chunk))

    async def commit_audio(self) -> None:
        if self.turn_detection is not None and self._committed_item_id is not None:
            return
        event = build_input_audio_commit_event()
        self._manual_commit_event_id = f"stt_commit_{uuid4().hex}"
        event["event_id"] = self._manual_commit_event_id
        await self._send_json(event)

    async def transcribe(
        self,
        *,
        session_id: str,
        turn_id: int,
        speaker: str,
        chunks: Iterable[AudioChunk],
    ) -> tuple[list[TranscriptEvent], TranscriptEvent]:
        return await self._transcribe_chunks(
            session_id=session_id,
            turn_id=turn_id,
            speaker=speaker,
            chunks=chunks,
            on_transcript=None,
        )

    async def transcribe_observed(
        self,
        *,
        session_id: str,
        turn_id: int,
        speaker: str,
        chunks: Iterable[AudioChunk],
        on_transcript: TranscriptObserver,
    ) -> tuple[list[TranscriptEvent], TranscriptEvent]:
        return await self._transcribe_chunks(
            session_id=session_id,
            turn_id=turn_id,
            speaker=speaker,
            chunks=chunks,
            on_transcript=on_transcript,
        )

    async def _transcribe_chunks(
        self,
        *,
        session_id: str,
        turn_id: int,
        speaker: str,
        chunks: Iterable[AudioChunk],
        on_transcript: TranscriptObserver | None,
    ) -> tuple[list[TranscriptEvent], TranscriptEvent]:
        self._committed_item_id = None
        self._manual_commit_event_id = None
        for chunk in chunks:
            await self.append_audio(chunk)
        await self.commit_audio()

        partials: list[TranscriptEvent] = []
        async for transcript in self.stream_transcripts(
            session_id=session_id,
            turn_id=turn_id,
            speaker=speaker,
            on_transcript=on_transcript,
        ):
            if transcript.transcript_type == "final":
                return partials, transcript
            partials.append(transcript)

        raise RuntimeError("realtime transcription stream ended before final transcript")

    async def transcribe_from_queue(
        self,
        *,
        session_id: str,
        turn_id: int,
        speaker: str,
        queue: Any,
        recording_mode: str = "push_to_talk",
    ) -> tuple[list[TranscriptEvent], TranscriptEvent]:
        _ = recording_mode
        return await self.transcribe_audio_stream(
            session_id=session_id,
            turn_id=turn_id,
            speaker=speaker,
            chunks=iter_turn_audio_chunks(
                queue,
                session_id=session_id,
                turn_id=turn_id,
                speaker=speaker,
            ),
        )

    async def transcribe_from_queue_observed(
        self,
        *,
        session_id: str,
        turn_id: int,
        speaker: str,
        queue: Any,
        on_transcript: TranscriptObserver,
        recording_mode: str = "push_to_talk",
    ) -> tuple[list[TranscriptEvent], TranscriptEvent]:
        _ = recording_mode
        return await self.transcribe_audio_stream(
            session_id=session_id,
            turn_id=turn_id,
            speaker=speaker,
            chunks=iter_turn_audio_chunks(
                queue,
                session_id=session_id,
                turn_id=turn_id,
                speaker=speaker,
            ),
            on_transcript=on_transcript,
        )

    async def transcribe_audio_stream(
        self,
        *,
        session_id: str,
        turn_id: int,
        speaker: str,
        chunks: AsyncIterator[AudioChunk],
        on_transcript: TranscriptObserver | None = None,
    ) -> tuple[list[TranscriptEvent], TranscriptEvent]:
        self._committed_item_id = None
        self._manual_commit_event_id = None
        append_task = asyncio.create_task(self._append_and_commit_audio(chunks))
        transcript_task = asyncio.create_task(
            self._collect_streamed_transcripts(
                session_id=session_id,
                turn_id=turn_id,
                speaker=speaker,
                on_transcript=on_transcript,
            )
        )
        try:
            done, _pending = await asyncio.wait(
                {append_task, transcript_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
            if append_task in done:
                await append_task
            result = await transcript_task
            if self.turn_detection is None:
                await append_task
            return result
        finally:
            for task in (append_task, transcript_task):
                if not task.done():
                    task.cancel()
            await asyncio.gather(append_task, transcript_task, return_exceptions=True)

    async def _append_and_commit_audio(
        self,
        chunks: AsyncIterator[AudioChunk],
    ) -> None:
        async for chunk in chunks:
            await self.append_audio(chunk)
        await self.commit_audio()

    async def _collect_streamed_transcripts(
        self,
        *,
        session_id: str,
        turn_id: int,
        speaker: str,
        on_transcript: TranscriptObserver | None,
    ) -> tuple[list[TranscriptEvent], TranscriptEvent]:
        partials: list[TranscriptEvent] = []
        async for transcript in self.stream_transcripts(
            session_id=session_id,
            turn_id=turn_id,
            speaker=speaker,
            on_transcript=on_transcript,
        ):
            if transcript.transcript_type == "final":
                return partials, transcript
            partials.append(transcript)

        raise RuntimeError(
            "realtime transcription stream ended before final transcript"
        )

    async def stream_transcripts(
        self,
        *,
        session_id: str,
        turn_id: int,
        speaker: str,
        on_transcript: TranscriptObserver | None = None,
    ) -> AsyncIterator[TranscriptEvent]:
        async for event in self.transport:
            event_type = event.get("type")
            if event_type == "input_audio_buffer.committed":
                if self._committed_item_id is None:
                    self._committed_item_id = event.get("item_id")
                continue
            if event_type == "error" and self._is_redundant_vad_commit_error(event):
                # VAD committed this speech while our explicit end was in flight.
                # Only that request failed; the committed item's final is pending.
                continue
            if (
                self._committed_item_id is not None
                and event_type
                in (
                    "conversation.item.input_audio_transcription.delta",
                    "conversation.item.input_audio_transcription.completed",
                    "conversation.item.input_audio_transcription.failed",
                )
                and event.get("item_id") != self._committed_item_id
            ):
                # An explicit end may also commit trailing silence. Its transcript
                # can finish before the actual speech; correlate by committed item.
                continue
            transcript = parse_realtime_transcription_event(
                event,
                session_id=session_id,
                turn_id=turn_id,
                speaker=speaker,
            )
            if transcript is None:
                continue
            await _notify_transcript_observer(on_transcript, transcript)
            yield transcript
            if transcript.transcript_type == "final":
                return

    def _is_redundant_vad_commit_error(self, event: Mapping[str, Any]) -> bool:
        error = event.get("error")
        return bool(
            self.turn_detection is not None
            and self._committed_item_id is not None
            and self._manual_commit_event_id is not None
            and isinstance(error, Mapping)
            and error.get("code") == "input_audio_buffer_commit_empty"
            and error.get("event_id") == self._manual_commit_event_id
        )

    async def _send_json(self, event: dict[str, Any]) -> None:
        result = self.transport.send_json(event)
        if inspect.isawaitable(result):
            await result


class OpenAIRealtimeTranscriptionSTT:

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str = "gpt-4o-transcribe",
        language: str = "ja",
        turn_detection: dict[str, Any] | None = None,
        vad_auto_turn_detection: dict[str, Any] | None = None,
        transport_context_factory: Callable[[], AsyncContextManager[Any]] | None = None,
        connect: Any | None = None,
        reuse_transport: bool = False,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.language = language
        self.turn_detection = turn_detection
        self.vad_auto_turn_detection = vad_auto_turn_detection
        self.transport_context_factory = transport_context_factory
        self.connect = connect
        self.reuse_transport = reuse_transport
        self._connection_lock = asyncio.Lock()
        self._operation_lock = asyncio.Lock()
        self._active_transport_context: AsyncContextManager[Any] | None = None
        self._active_session: RealtimeStreamingSTT | None = None

    async def transcribe(
        self,
        *,
        session_id: str,
        turn_id: int,
        speaker: str,
        chunks: Iterable[AudioChunk],
    ) -> tuple[list[TranscriptEvent], TranscriptEvent]:
        return await self._transcribe(
            session_id=session_id,
            turn_id=turn_id,
            speaker=speaker,
            chunks=chunks,
            on_transcript=None,
        )

    async def transcribe_observed(
        self,
        *,
        session_id: str,
        turn_id: int,
        speaker: str,
        chunks: Iterable[AudioChunk],
        on_transcript: TranscriptObserver,
    ) -> tuple[list[TranscriptEvent], TranscriptEvent]:
        return await self._transcribe(
            session_id=session_id,
            turn_id=turn_id,
            speaker=speaker,
            chunks=chunks,
            on_transcript=on_transcript,
        )

    async def _transcribe(
        self,
        *,
        session_id: str,
        turn_id: int,
        speaker: str,
        chunks: Iterable[AudioChunk],
        on_transcript: TranscriptObserver | None,
    ) -> tuple[list[TranscriptEvent], TranscriptEvent]:
        if self.reuse_transport:
            async with self._operation_lock:
                session = await self._get_active_session()
                try:
                    return await session._transcribe_chunks(
                        session_id=session_id,
                        turn_id=turn_id,
                        speaker=speaker,
                        chunks=chunks,
                        on_transcript=on_transcript,
                    )
                except Exception:
                    await self.close()
                    raise

        async with self._open_transport() as transport:
            stt = self._build_session(transport)
            await stt.configure()
            return await stt._transcribe_chunks(
                session_id=session_id,
                turn_id=turn_id,
                speaker=speaker,
                chunks=chunks,
                on_transcript=on_transcript,
            )

    async def transcribe_from_queue(
        self,
        *,
        session_id: str,
        turn_id: int,
        speaker: str,
        queue: Any,
        recording_mode: str = "push_to_talk",
    ) -> tuple[list[TranscriptEvent], TranscriptEvent]:
        return await self._transcribe_from_queue(
            session_id=session_id,
            turn_id=turn_id,
            speaker=speaker,
            queue=queue,
            on_transcript=None,
            recording_mode=recording_mode,
        )

    async def transcribe_from_queue_observed(
        self,
        *,
        session_id: str,
        turn_id: int,
        speaker: str,
        queue: Any,
        on_transcript: TranscriptObserver,
        recording_mode: str = "push_to_talk",
    ) -> tuple[list[TranscriptEvent], TranscriptEvent]:
        return await self._transcribe_from_queue(
            session_id=session_id,
            turn_id=turn_id,
            speaker=speaker,
            queue=queue,
            on_transcript=on_transcript,
            recording_mode=recording_mode,
        )

    async def _transcribe_from_queue(
        self,
        *,
        session_id: str,
        turn_id: int,
        speaker: str,
        queue: Any,
        on_transcript: TranscriptObserver | None,
        recording_mode: str,
    ) -> tuple[list[TranscriptEvent], TranscriptEvent]:
        chunk_iterator = iter_turn_audio_chunks(
            queue,
            session_id=session_id,
            turn_id=turn_id,
            speaker=speaker,
        )
        try:
            first_chunk = await anext(chunk_iterator)
        except StopAsyncIteration as exc:
            raise RuntimeError("audio stream ended before target turn audio was committed") from exc

        turn_detection = self._turn_detection_for_recording_mode(recording_mode)
        if self.reuse_transport:
            async with self._operation_lock:
                session = await self._get_active_session(turn_detection=turn_detection)
                try:
                    return await session.transcribe_audio_stream(
                        session_id=session_id,
                        turn_id=turn_id,
                        speaker=speaker,
                        chunks=_prepend_async(first_chunk, chunk_iterator),
                        on_transcript=on_transcript,
                    )
                except Exception:
                    await self.close()
                    raise

        async with self._open_transport() as transport:
            stt = self._build_session(transport, turn_detection=turn_detection)
            await stt.configure()
            return await stt.transcribe_audio_stream(
                session_id=session_id,
                turn_id=turn_id,
                speaker=speaker,
                chunks=_prepend_async(first_chunk, chunk_iterator),
                on_transcript=on_transcript,
            )

    async def close(self) -> None:
        async with self._connection_lock:
            active_transport_context = self._active_transport_context
            self._active_transport_context = None
            self._active_session = None
        if active_transport_context is not None:
            await active_transport_context.__aexit__(None, None, None)

    async def _get_active_session(
        self,
        *,
        turn_detection: dict[str, Any] | None | object = _USE_DEFAULT_TURN_DETECTION,
    ) -> RealtimeStreamingSTT:
        async with self._connection_lock:
            if self._active_session is not None:
                return self._active_session

            transport_context = self._open_transport()
            transport: Any | None = None
            try:
                transport = await transport_context.__aenter__()
                session = self._build_session(transport, turn_detection=turn_detection)
                await session.configure()
            except Exception:
                if transport is not None:
                    await transport_context.__aexit__(None, None, None)
                raise

            self._active_transport_context = transport_context
            self._active_session = session
            return session

    def _open_transport(self) -> AsyncContextManager[Any]:
        if self.transport_context_factory is not None:
            return self.transport_context_factory()
        if not self.api_key or not self.api_key.strip():
            raise ValueError(
                "api_key is required without an injected transcription transport"
            )

        from counseling_voice_demo.runtime.realtime_transport import connect_realtime_transcription

        return connect_realtime_transcription(api_key=self.api_key, connect=self.connect)

    def _turn_detection_for_recording_mode(
        self,
        recording_mode: str,
    ) -> dict[str, Any] | None:
        if recording_mode == "browser_vad":
            # The browser sends end after silence. Keep all audio in one item
            # until that explicit commit, including short speech and barge-in.
            return None
        if recording_mode == "vad_auto":
            return (
                self.vad_auto_turn_detection
                or self.turn_detection
                or DEFAULT_REALTIME_TRANSCRIPTION_SERVER_VAD
            )
        return self.turn_detection

    def _build_session(
        self,
        transport: Any,
        *,
        turn_detection: dict[str, Any] | None | object = _USE_DEFAULT_TURN_DETECTION,
    ) -> RealtimeStreamingSTT:
        resolved_turn_detection = (
            self.turn_detection
            if turn_detection is _USE_DEFAULT_TURN_DETECTION
            else turn_detection
        )
        return RealtimeStreamingSTT(
            transport,
            model=self.model,
            language=self.language,
            turn_detection=resolved_turn_detection,
        )


class FakeStreamingSTT:
    async def transcribe(
        self,
        *,
        session_id: str,
        turn_id: int,
        speaker: str,
        chunks: Iterable[AudioChunk],
    ) -> tuple[list[TranscriptEvent], TranscriptEvent]:
        return await self._transcribe(
            session_id=session_id,
            turn_id=turn_id,
            speaker=speaker,
            chunks=chunks,
            on_transcript=None,
        )

    async def transcribe_observed(
        self,
        *,
        session_id: str,
        turn_id: int,
        speaker: str,
        chunks: Iterable[AudioChunk],
        on_transcript: TranscriptObserver,
    ) -> tuple[list[TranscriptEvent], TranscriptEvent]:
        return await self._transcribe(
            session_id=session_id,
            turn_id=turn_id,
            speaker=speaker,
            chunks=chunks,
            on_transcript=on_transcript,
        )

    async def _transcribe(
        self,
        *,
        session_id: str,
        turn_id: int,
        speaker: str,
        chunks: Iterable[AudioChunk],
        on_transcript: TranscriptObserver | None,
    ) -> tuple[list[TranscriptEvent], TranscriptEvent]:
        audio_chunks = list(chunks)
        chunk_count = len(audio_chunks)
        partial_text = f"partial:{speaker}:{turn_id}:{chunk_count}"
        final_text = f"stt_final:{_text_from_fake_tts_chunks(audio_chunks)}"
        partial = TranscriptEvent(
            session_id=session_id,
            turn_id=turn_id,
            speaker=speaker,
            transcript_type="partial",
            text=partial_text,
        )
        final = TranscriptEvent(
            session_id=session_id,
            turn_id=turn_id,
            speaker=speaker,
            transcript_type="final",
            text=final_text,
        )
        await _notify_transcript_observer(on_transcript, partial)
        await _notify_transcript_observer(on_transcript, final)
        return [partial], final

    async def transcribe_from_queue(
        self,
        *,
        session_id: str,
        turn_id: int,
        speaker: str,
        queue: Any,
        recording_mode: str = "push_to_talk",
    ) -> tuple[list[TranscriptEvent], TranscriptEvent]:
        _ = recording_mode
        return await self._transcribe_from_queue(
            session_id=session_id,
            turn_id=turn_id,
            speaker=speaker,
            queue=queue,
            on_transcript=None,
        )

    async def transcribe_from_queue_observed(
        self,
        *,
        session_id: str,
        turn_id: int,
        speaker: str,
        queue: Any,
        on_transcript: TranscriptObserver,
        recording_mode: str = "push_to_talk",
    ) -> tuple[list[TranscriptEvent], TranscriptEvent]:
        _ = recording_mode
        return await self._transcribe_from_queue(
            session_id=session_id,
            turn_id=turn_id,
            speaker=speaker,
            queue=queue,
            on_transcript=on_transcript,
        )

    async def _transcribe_from_queue(
        self,
        *,
        session_id: str,
        turn_id: int,
        speaker: str,
        queue: Any,
        on_transcript: TranscriptObserver | None,
    ) -> tuple[list[TranscriptEvent], TranscriptEvent]:
        chunks = [
            chunk
            async for chunk in iter_turn_audio_chunks(
                queue,
                session_id=session_id,
                turn_id=turn_id,
                speaker=speaker,
            )
        ]
        return await self._transcribe(
            session_id=session_id,
            turn_id=turn_id,
            speaker=speaker,
            chunks=chunks,
            on_transcript=on_transcript,
        )


def _text_from_fake_tts_chunks(chunks: Iterable[AudioChunk]) -> str:
    parts: list[str] = []
    for chunk in sorted(chunks, key=lambda item: item.chunk_index):
        marker = chunk.pcm.split(b"\0", 1)[0].decode("utf-8", errors="ignore")
        fields = marker.split(":", 3)
        if len(fields) == 4:
            parts.append(fields[3])
    return "".join(parts)


async def _prepend_async(
    first: AudioChunk,
    rest: AsyncIterator[AudioChunk],
) -> AsyncIterator[AudioChunk]:
    yield first
    async for item in rest:
        yield item


async def _notify_transcript_observer(
    on_transcript: TranscriptObserver | None,
    transcript: TranscriptEvent,
) -> None:
    if on_transcript is None:
        return
    result = on_transcript(transcript)
    if inspect.isawaitable(result):
        await result


def _validate_turn_audio_item(
    item: AudioChunk,
    *,
    session_id: str,
    turn_id: int,
    speaker: str,
) -> None:
    if (item.session_id, item.turn_id, item.speaker) != (session_id, turn_id, speaker):
        raise RuntimeError(
            "STT queue received audio for an unexpected turn: "
            f"{item.session_id}/{item.turn_id}/{item.speaker}"
        )


def _validate_turn_end_item(
    item: EndOfAudio,
    *,
    session_id: str,
    turn_id: int,
    speaker: str,
) -> None:
    if (item.session_id, item.turn_id, item.speaker) != (session_id, turn_id, speaker):
        raise RuntimeError(
            "STT queue received end-of-audio for an unexpected turn: "
            f"{item.session_id}/{item.turn_id}/{item.speaker}"
        )


def _format_realtime_error_event(event: Mapping[str, Any]) -> str:
    error = event.get("error")
    if not isinstance(error, Mapping):
        return "Realtime transcription error"

    parts: list[str] = ["Realtime transcription error"]
    code = error.get("code")
    param = error.get("param")
    message = error.get("message")
    if code:
        parts.append(f"code={code}")
    if param:
        parts.append(f"param={param}")
    if message:
        parts.append(str(message))
    return ": ".join(parts)
