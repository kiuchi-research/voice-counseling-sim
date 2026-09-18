from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable, Iterable, Iterator
from dataclasses import dataclass, field
from typing import Any

from counseling_voice_demo.runtime.models import AudioChunk, AudioDeliveryMode


class StreamingTtsError(RuntimeError):
    pass


@dataclass(frozen=True)
class OpenAIStreamingTTSConfig:
    model: str = "gpt-4o-mini-tts"
    voice: str = "coral"
    voice_by_speaker: dict[str, str] = field(default_factory=dict)
    instructions: str = ""
    response_format: str = "pcm"
    sdk_chunk_size: int | None = None
    target_chunk_duration_ms: int = 40

    def __post_init__(self) -> None:
        if self.response_format != "pcm":
            raise ValueError("OpenAIStreamingTTS only supports response_format='pcm' in the v0.3 hot path")
        if self.sdk_chunk_size is not None and self.sdk_chunk_size <= 0:
            raise ValueError("sdk_chunk_size must be positive when provided")
        if self.target_chunk_duration_ms <= 0:
            raise ValueError("target_chunk_duration_ms must be positive")


class OpenAIStreamingTTS:
    def __init__(
        self,
        *,
        client: Any,
        config: OpenAIStreamingTTSConfig | None = None,
        response_factory: Callable[[dict[str, Any]], Any] | None = None,
    ) -> None:
        self._client = client
        self._config = config or OpenAIStreamingTTSConfig()
        self._response_factory = response_factory

    async def synthesize(
        self,
        *,
        session_id: str,
        turn_id: int,
        speaker: str,
        text_chunks: Iterable[str],
        sample_rate: int,
        sample_width_bits: int,
        channels: int,
        delivery_mode: AudioDeliveryMode,
    ) -> AsyncIterator[AudioChunk]:
        chunk_index = 0
        bytes_per_frame = _bytes_per_frame(
            sample_width_bits=sample_width_bits,
            channels=channels,
        )
        target_chunk_bytes = _target_chunk_bytes(
            sample_rate=sample_rate,
            bytes_per_frame=bytes_per_frame,
            target_chunk_duration_ms=self._config.target_chunk_duration_ms,
        )
        for text_chunk in text_chunks:
            text = text_chunk.strip()
            if not text:
                continue
            payload = self.build_payload(text, speaker=speaker)
            pending_pcm = b""
            ready_pcm = b""
            async with await _ensure_async(self._create_response(payload)) as response:
                async for pcm in _iter_response_bytes(response, chunk_size=self._config.sdk_chunk_size):
                    if not pcm:
                        continue
                    pcm = pending_pcm + pcm
                    complete_byte_length = len(pcm) - (len(pcm) % bytes_per_frame)
                    if complete_byte_length <= 0:
                        pending_pcm = pcm
                        continue
                    pending_pcm = pcm[complete_byte_length:]
                    pcm = pcm[:complete_byte_length]
                    ready_pcm += pcm
                    while len(ready_pcm) >= target_chunk_bytes:
                        output_pcm = ready_pcm[:target_chunk_bytes]
                        ready_pcm = ready_pcm[target_chunk_bytes:]
                        yield _audio_chunk(
                            session_id=session_id,
                            turn_id=turn_id,
                            speaker=speaker,
                            chunk_index=chunk_index,
                            pcm=output_pcm,
                            sample_rate=sample_rate,
                            sample_width_bits=sample_width_bits,
                            channels=channels,
                            delivery_mode=delivery_mode,
                        )
                        chunk_index += 1
            if pending_pcm:
                raise StreamingTtsError(
                    "streaming TTS PCM response ended with an incomplete audio frame"
                )
            if ready_pcm:
                yield _audio_chunk(
                    session_id=session_id,
                    turn_id=turn_id,
                    speaker=speaker,
                    chunk_index=chunk_index,
                    pcm=ready_pcm,
                    sample_rate=sample_rate,
                    sample_width_bits=sample_width_bits,
                    channels=channels,
                    delivery_mode=delivery_mode,
                )
                chunk_index += 1

    def build_payload(self, text: str, *, speaker: str | None = None) -> dict[str, Any]:
        voice = self._config.voice_by_speaker.get(speaker or "", self._config.voice)
        payload = {
            "model": self._config.model,
            "voice": voice,
            "input": text,
            "response_format": self._config.response_format,
        }
        if self._config.instructions:
            payload["instructions"] = self._config.instructions
        return payload

    def _create_response(self, payload: dict[str, Any]) -> Any:
        if self._response_factory is not None:
            return self._response_factory(payload)
        return self._client.audio.speech.with_streaming_response.create(**payload)


class FakeStreamingTTS:
    def __init__(self, *, chunk_duration_ms: int = 40) -> None:
        if chunk_duration_ms <= 0:
            raise ValueError("chunk_duration_ms must be positive")
        self._chunk_duration_ms = chunk_duration_ms

    async def synthesize(
        self,
        *,
        session_id: str,
        turn_id: int,
        speaker: str,
        text_chunks: Iterable[str],
        sample_rate: int,
        sample_width_bits: int,
        channels: int,
        delivery_mode: AudioDeliveryMode,
    ) -> AsyncIterator[AudioChunk]:
        bytes_per_sample = sample_width_bits // 8
        frame_count = max(1, sample_rate * self._chunk_duration_ms // 1000)
        pcm_size = frame_count * bytes_per_sample * channels

        for chunk_index, text_chunk in enumerate(text_chunks):
            marker = f"{speaker}:{turn_id}:{chunk_index}:{text_chunk}".encode("utf-8")
            if len(marker) >= pcm_size:
                pcm = marker[:pcm_size]
            else:
                pcm = marker + (b"\0" * (pcm_size - len(marker)))
            yield AudioChunk(
                session_id=session_id,
                turn_id=turn_id,
                speaker=speaker,
                chunk_index=chunk_index,
                pcm=pcm,
                sample_rate=sample_rate,
                sample_width_bits=sample_width_bits,
                channels=channels,
                duration_ms=self._chunk_duration_ms,
                delivery_mode=delivery_mode,
            )


def pcm_duration_ms(
    pcm: bytes,
    *,
    sample_rate: int,
    sample_width_bits: int,
    channels: int,
) -> int:
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if sample_width_bits <= 0 or sample_width_bits % 8 != 0:
        raise ValueError("sample_width_bits must be a positive multiple of 8")
    if channels <= 0:
        raise ValueError("channels must be positive")

    bytes_per_frame = _bytes_per_frame(sample_width_bits=sample_width_bits, channels=channels)
    frames = len(pcm) / bytes_per_frame
    return round((frames / sample_rate) * 1000)


def _bytes_per_frame(*, sample_width_bits: int, channels: int) -> int:
    if sample_width_bits <= 0 or sample_width_bits % 8 != 0:
        raise ValueError("sample_width_bits must be a positive multiple of 8")
    if channels <= 0:
        raise ValueError("channels must be positive")
    bytes_per_frame = (sample_width_bits // 8) * channels
    if bytes_per_frame <= 0:
        raise ValueError("bytes_per_frame must be positive")
    return bytes_per_frame


def _target_chunk_bytes(
    *,
    sample_rate: int,
    bytes_per_frame: int,
    target_chunk_duration_ms: int,
) -> int:
    frames = max(1, sample_rate * target_chunk_duration_ms // 1000)
    return frames * bytes_per_frame


def _audio_chunk(
    *,
    session_id: str,
    turn_id: int,
    speaker: str,
    chunk_index: int,
    pcm: bytes,
    sample_rate: int,
    sample_width_bits: int,
    channels: int,
    delivery_mode: AudioDeliveryMode,
) -> AudioChunk:
    return AudioChunk(
        session_id=session_id,
        turn_id=turn_id,
        speaker=speaker,
        chunk_index=chunk_index,
        pcm=pcm,
        sample_rate=sample_rate,
        sample_width_bits=sample_width_bits,
        channels=channels,
        duration_ms=pcm_duration_ms(
            pcm,
            sample_rate=sample_rate,
            sample_width_bits=sample_width_bits,
            channels=channels,
        ),
        delivery_mode=delivery_mode,
    )


async def _ensure_async(value: Any) -> Any:
    if hasattr(value, "__await__"):
        return await value
    return value


async def _iter_response_bytes(response: Any, *, chunk_size: int | None = None) -> AsyncIterator[bytes]:
    iterator = _call_byte_iterator(response, "aiter_bytes", chunk_size)
    if iterator is not None:
        async for chunk in iterator:
            yield chunk
        return

    iterator = _call_byte_iterator(response, "iter_bytes", chunk_size)
    if iterator is not None:
        if hasattr(iterator, "__aiter__"):
            async for chunk in iterator:
                yield chunk
        else:
            for chunk in iterator:
                yield chunk
        return

    if hasattr(response, "read"):
        content = response.read()
        if hasattr(content, "__await__"):
            content = await content
        yield content
        return

    content = getattr(response, "content", None)
    if content is not None:
        yield content
        return

    raise StreamingTtsError("streaming TTS response does not expose bytes")


def _call_byte_iterator(response: Any, method_name: str, chunk_size: int | None) -> AsyncIterator[bytes] | Iterator[bytes] | None:
    method = getattr(response, method_name, None)
    if method is None:
        return None
    if chunk_size is None:
        return method()
    try:
        return method(chunk_size=chunk_size)
    except TypeError:
        return method()
