from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any, Final

if TYPE_CHECKING:
    from .models import AudioChunk, AudioDeliveryMode
else:
    AudioChunk = Any
    AudioDeliveryMode = Any

try:
    from .models import AudioDeliveryMode as _AudioDeliveryMode
    from .models import EndOfAudio as _EndOfAudio
except (AttributeError, ImportError):
    _AudioDeliveryMode = None
    _EndOfAudio = None


_LOGGER_STOP: Final = object()
_MISSING: Final = object()


def _default_end_event() -> object | None:
    if _EndOfAudio is None:
        return None
    if isinstance(_EndOfAudio, type):
        try:
            return _EndOfAudio()
        except TypeError:
            return _EndOfAudio
    return _EndOfAudio


def _turn_end_event(
    session_id: str,
    turn_id: int,
    speaker: str,
    *,
    audio_role: str = "main",
) -> object | None:
    if _EndOfAudio is None:
        return None
    if isinstance(_EndOfAudio, type):
        try:
            return _EndOfAudio(
                session_id=session_id,
                turn_id=turn_id,
                speaker=speaker,
                audio_role=audio_role,
            )
        except TypeError:
            try:
                return _EndOfAudio(
                    session_id=session_id,
                    turn_id=turn_id,
                    speaker=speaker,
                )
            except TypeError:
                return _default_end_event()
    return _EndOfAudio


class AudioBus:
    """Fan out TTS audio chunks to monitor, STT, and logger queues."""

    def __init__(
        self,
        *,
        monitor_queue: asyncio.Queue[AudioChunk | object | None] | None = None,
        opponent_stt_queue: asyncio.Queue[AudioChunk | object | None] | None = None,
        logger_queue: asyncio.Queue[AudioChunk | object | None] | None = None,
        end_event: object | None = _MISSING,
        audio_delivery_mode: AudioDeliveryMode | None = None,
        stt_delivery_enabled: bool = True,
        sleeper: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self.monitor_queue = monitor_queue if monitor_queue is not None else asyncio.Queue()
        self.opponent_stt_queue = opponent_stt_queue if opponent_stt_queue is not None else asyncio.Queue()
        self.logger_queue = logger_queue if logger_queue is not None else asyncio.Queue()
        self._monitor_queues: list[asyncio.Queue[AudioChunk | object | None]] = [
            self.monitor_queue
        ]
        self.listener_stt_queues: dict[
            str, asyncio.Queue[AudioChunk | object | None]
        ] = {}
        self.end_event = _default_end_event() if end_event is _MISSING else end_event
        self.audio_delivery_mode = audio_delivery_mode
        self.stt_delivery_enabled = stt_delivery_enabled
        self._sleeper = sleeper

        self._logger_pending: asyncio.Queue[AudioChunk | object | None] = asyncio.Queue()
        self._logger_worker: asyncio.Task[None] | None = None
        self._closed = False

    async def publish(self, chunk: AudioChunk) -> None:
        if self._closed:
            raise RuntimeError("AudioBus is already closed")

        self._ensure_logger_worker()
        self._logger_pending.put_nowait(chunk)
        if self.stt_delivery_enabled:
            await self.opponent_stt_queue.put(chunk)
            await self._publish_to_listener_stt(
                chunk,
                exclude_listener_id=getattr(chunk, "speaker", _MISSING),
            )
        await self._publish_to_monitors(chunk)
        await self._pace_after_stt_delivery(chunk)

    async def put(self, chunk: AudioChunk) -> None:
        await self.publish(chunk)

    async def publish_logger_audio(self, chunk: AudioChunk) -> None:
        """Send audio to persistent logs without live monitor/STT fan-out."""
        if self._closed:
            raise RuntimeError("AudioBus is already closed")

        self._ensure_logger_worker()
        self._logger_pending.put_nowait(chunk)

    async def end_logger_turn(
        self,
        session_id: str,
        turn_id: int,
        speaker: str,
        *,
        audio_role: str = "main",
    ) -> None:
        if self._closed:
            raise RuntimeError("AudioBus is already closed")

        self._ensure_logger_worker()
        self._logger_pending.put_nowait(
            _turn_end_event(
                session_id,
                turn_id,
                speaker,
                audio_role=audio_role,
            )
        )

    async def end_turn(
        self,
        session_id: str,
        turn_id: int,
        speaker: str,
        *,
        audio_role: str = "main",
    ) -> None:
        if self._closed:
            raise RuntimeError("AudioBus is already closed")

        self._ensure_logger_worker()
        end_event = _turn_end_event(
            session_id,
            turn_id,
            speaker,
            audio_role=audio_role,
        )
        self._logger_pending.put_nowait(end_event)
        if self.stt_delivery_enabled:
            await self.opponent_stt_queue.put(end_event)
            await self._publish_to_listener_stt(end_event, exclude_listener_id=speaker)
        await self._publish_to_monitors(end_event)

    async def publish_monitor_event(self, item: object) -> None:
        if self._closed:
            raise RuntimeError("AudioBus is already closed")

        await self._publish_to_monitors(item)

    async def flush(self) -> None:
        await self._logger_pending.join()

    async def flush_logger(self) -> None:
        await self.flush()

    async def close(self) -> None:
        if self._closed:
            await self.flush()
            if self._logger_worker is not None:
                await self._logger_worker
            return

        self._closed = True
        self._ensure_logger_worker()

        if self.stt_delivery_enabled:
            await self.opponent_stt_queue.put(self.end_event)
            await self._publish_to_listener_stt(self.end_event)
        await self._publish_to_monitors(self.end_event)
        self._logger_pending.put_nowait(self.end_event)
        self._logger_pending.put_nowait(_LOGGER_STOP)

        await self.flush()
        if self._logger_worker is not None:
            await self._logger_worker

    async def stop(self) -> None:
        await self.close()

    def subscribe_monitor(self) -> asyncio.Queue[AudioChunk | object | None]:
        if self._closed:
            raise RuntimeError("AudioBus is already closed")
        queue: asyncio.Queue[AudioChunk | object | None] = asyncio.Queue()
        self._monitor_queues.append(queue)
        return queue

    def unsubscribe_monitor(
        self,
        queue: asyncio.Queue[AudioChunk | object | None],
    ) -> None:
        if queue is self.monitor_queue:
            return
        try:
            self._monitor_queues.remove(queue)
        except ValueError:
            return

    def subscribe_listener_stt(
        self,
        listener_id: str,
    ) -> asyncio.Queue[AudioChunk | object | None]:
        if self._closed:
            raise RuntimeError("AudioBus is already closed")
        queue = self.listener_stt_queues.get(listener_id)
        if queue is None:
            queue = asyncio.Queue()
            self.listener_stt_queues[listener_id] = queue
        return queue

    def unsubscribe_listener_stt(self, listener_id: str) -> None:
        self.listener_stt_queues.pop(listener_id, None)

    def _ensure_logger_worker(self) -> None:
        if self._logger_worker is None or self._logger_worker.done():
            self._logger_worker = asyncio.create_task(self._run_logger_worker())

    async def _publish_to_monitors(self, item: AudioChunk | object | None) -> None:
        for queue in list(self._monitor_queues):
            await queue.put(item)

    async def _publish_to_listener_stt(
        self,
        item: AudioChunk | object | None,
        *,
        exclude_listener_id: object = _MISSING,
    ) -> None:
        for listener_id, queue in list(self.listener_stt_queues.items()):
            if (
                exclude_listener_id is not _MISSING
                and listener_id == exclude_listener_id
            ):
                continue
            await queue.put(item)

    async def _pace_after_stt_delivery(self, chunk: AudioChunk) -> None:
        if not getattr(chunk, "pace_after_delivery", True):
            return
        mode = self.audio_delivery_mode if self.audio_delivery_mode is not None else getattr(chunk, "delivery_mode", None)
        if _AudioDeliveryMode is None or mode != _AudioDeliveryMode.REAL_TIME:
            return
        duration_ms = getattr(chunk, "duration_ms", 0)
        if duration_ms > 0:
            await self._sleeper(duration_ms / 1000)

    async def _run_logger_worker(self) -> None:
        while True:
            item = await self._logger_pending.get()
            try:
                if item is _LOGGER_STOP:
                    return
                await self.logger_queue.put(item)
            finally:
                self._logger_pending.task_done()
