from __future__ import annotations

import asyncio
import base64
import inspect
import json
from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass, field
from typing import Any

from counseling_voice_demo.runtime.floor_mediator import TimingDecisionInput
from counseling_voice_demo.runtime.models import (
    AudioChunk,
    AudioDeliveryMode,
    MAX_REALTIME_OUTPUT_SPEED,
    MIN_REALTIME_OUTPUT_SPEED,
)
from counseling_voice_demo.runtime.protocols import StreamingLLMLike
from counseling_voice_demo.runtime.streaming_tts import pcm_duration_ms
from counseling_voice_demo.runtime.timing_decision_prompt import (
    TIMING_DECISION_SYSTEM_PROMPT,
    build_timing_decision_input,
    parse_timing_decision_response,
)

REALTIME_OUTPUT_AUDIO_DELTA_EVENT = "response.output_audio.delta"
REALTIME_OUTPUT_AUDIO_DONE_EVENT = "response.output_audio.done"
REALTIME_OUTPUT_AUDIO_TRANSCRIPT_DELTA_EVENT = "response.output_audio_transcript.delta"
REALTIME_OUTPUT_TEXT_DELTA_EVENT = "response.output_text.delta"
REALTIME_RESPONSE_CREATED_EVENT = "response.created"
REALTIME_RESPONSE_DONE_EVENT = "response.done"
REALTIME_ERROR_EVENT = "error"
REALTIME_RESPONSE_CANCEL_EVENT = "response.cancel"
REALTIME_CONVERSATION_ITEM_TRUNCATE_EVENT = "conversation.item.truncate"
REALTIME_INPUT_AUDIO_BUFFER_APPEND_EVENT = "input_audio_buffer.append"
REALTIME_INPUT_AUDIO_BUFFER_COMMIT_EVENT = "input_audio_buffer.commit"
REALTIME_INPUT_AUDIO_TRANSCRIPTION_DELTA_EVENT = (
    "conversation.item.input_audio_transcription.delta"
)
REALTIME_INPUT_AUDIO_TRANSCRIPTION_COMPLETED_EVENT = (
    "conversation.item.input_audio_transcription.completed"
)
_REALTIME_INPUT_TRANSCRIPTION_GRACE_SECONDS = 0.5

AgentInputFormatter = Callable[..., str]
ResponseInstructionsObserver = Callable[["ResponseCreateInstructions"], Any]


class RealtimeSpeechError(RuntimeError):
    pass


@dataclass(frozen=True)
class ResponseCreateInstructions:
    session_id: str
    turn_id: int
    speaker: str
    session_instructions: str
    participant_instructions: str
    additional_instructions: str
    resolved_instructions: str
    repeat_session_instructions: bool
    input_text: str = ""
    response_input: list[dict[str, Any]] | None = None
    response_conversation: str | None = None


@dataclass(frozen=True)
class RealtimeSpeechConfig:
    model: str = "gpt-realtime-mini"
    voice: str = "coral"
    output_speed: float | None = None
    instructions: str = ""
    response_instructions: str = ""
    repeat_instructions_per_response: bool = False
    input_format: dict[str, Any] = field(
        default_factory=lambda: {"type": "audio/pcm", "rate": 24000}
    )
    input_transcription: dict[str, Any] = field(
        default_factory=lambda: {"model": "gpt-realtime-whisper"}
    )
    noise_reduction: dict[str, Any] = field(
        default_factory=lambda: {"type": "far_field"}
    )
    turn_detection: dict[str, Any] = field(
        default_factory=lambda: {
            "type": "server_vad",
            "threshold": 0.5,
            "prefix_padding_ms": 300,
            "silence_duration_ms": 500,
            "idle_timeout_ms": None,
        }
    )
    output_format: dict[str, Any] = field(
        default_factory=lambda: {"type": "audio/pcm", "rate": 24000}
    )
    output_modalities: tuple[str, ...] = ("audio",)
    tools: tuple[dict[str, Any], ...] = ()
    max_output_tokens: str | int = "inf"
    target_chunk_duration_ms: int = 40

    def __post_init__(self) -> None:
        if self.input_format.get("type") != "audio/pcm":
            raise ValueError("RealtimeSpeechConfig only supports audio/pcm input")
        if int(self.input_format.get("rate", 0)) <= 0:
            raise ValueError("RealtimeSpeechConfig input rate must be positive")
        if self.output_format.get("type") != "audio/pcm":
            raise ValueError("RealtimeSpeechConfig only supports audio/pcm output")
        if int(self.output_format.get("rate", 0)) <= 0:
            raise ValueError("RealtimeSpeechConfig output rate must be positive")
        if self.target_chunk_duration_ms <= 0:
            raise ValueError("target_chunk_duration_ms must be positive")
        if (
            self.output_speed is not None
            and not MIN_REALTIME_OUTPUT_SPEED
            <= self.output_speed
            <= MAX_REALTIME_OUTPUT_SPEED
        ):
            raise ValueError(
                "output_speed must be between "
                f"{MIN_REALTIME_OUTPUT_SPEED} and {MAX_REALTIME_OUTPUT_SPEED}"
            )
        if not self.output_modalities:
            raise ValueError("output_modalities must not be empty")


@dataclass(frozen=True)
class RealtimeSpeechEvent:
    text_delta: str | None = None
    audio_chunk: AudioChunk | None = None
    input_transcript_delta: str | None = None
    input_transcript_completed: str | None = None


@dataclass(frozen=True)
class RealtimePlaybackStopResult:
    played_ms: int
    cancel_sent: bool
    truncate_sent: bool
    item_id: str | None = None
    response_id: str | None = None
    content_index: int = 0
    sent_event_types: tuple[str, ...] = ()
    audio_end_ms: int | None = None


@dataclass
class _RealtimeOutputAudioProgress:
    sample_rate: int
    bytes_per_frame: int
    output_speed: float = 1.0
    received_bytes: int = 0
    truncated_at_ms: int | None = None

    @property
    def available_ms(self) -> int:
        frames = self.received_bytes // self.bytes_per_frame
        duration_ms = frames * 1000 // self.sample_rate
        if self.truncated_at_ms is not None:
            return min(duration_ms, self.truncated_at_ms)
        return duration_ms


def build_response_instructions(
    base_instructions: str,
    additional_instruction: str | None,
    *,
    response_instructions: str = "",
    repeat_base_instructions: bool = False,
) -> str | None:
    normalized_additional = (
        additional_instruction.strip()
        if additional_instruction is not None and additional_instruction.strip()
        else ""
    )
    normalized_response = response_instructions.strip()
    if not (normalized_additional or normalized_response or repeat_base_instructions):
        return None
    normalized_base = base_instructions.strip()
    sections: list[str] = []
    if normalized_base:
        sections.append(normalized_base)
    if normalized_response:
        sections.extend(
            [
                "Session setupで設定されたカウンセラープロンプト:",
                normalized_response,
            ]
        )
    if normalized_additional:
        sections.extend(["追加の内部指示:", normalized_additional])
    return "\n\n".join(sections) or None


def build_scripted_audio_response_instructions(
    text: str,
    additional_instruction: str | None = None,
) -> str:
    _ = additional_instruction
    return "\n\n".join(
        [
            "これは会話応答ではなく、指定本文だけを音声出力するタスクです。",
            "あなたの出力本文は <target> 内の1文と完全一致しなければなりません。",
            "出力本文の前後に、承知しました、では、読み上げます、などの語句を一切追加しないでください。",
            "target 外の挨拶、相づち、説明、言い換え、要約、質問、追加文は禁止です。",
            "<target>",
            text,
            "</target>",
        ]
    )


def build_realtime_speech_session_update(
    *,
    model: str = "gpt-realtime-mini",
    instructions: str = "",
    voice: str = "coral",
    output_speed: float | None = None,
    output_format: dict[str, Any] | None = None,
    input_format: dict[str, Any] | None = None,
    input_transcription: dict[str, Any] | None = None,
    noise_reduction: dict[str, Any] | None = None,
    turn_detection: dict[str, Any] | None = None,
    output_modalities: tuple[str, ...] | list[str] = ("audio",),
    tools: tuple[dict[str, Any], ...] | list[dict[str, Any]] | None = None,
    max_output_tokens: str | int | None = "inf",
) -> dict[str, Any]:
    output_format = output_format or {"type": "audio/pcm", "rate": 24000}
    input_format = input_format or {"type": "audio/pcm", "rate": 24000}
    input_transcription = input_transcription or {"model": "gpt-realtime-whisper"}
    noise_reduction = noise_reduction or {"type": "far_field"}
    turn_detection = turn_detection or {
        "type": "server_vad",
        "threshold": 0.5,
        "prefix_padding_ms": 300,
        "silence_duration_ms": 500,
        "idle_timeout_ms": None,
    }
    session: dict[str, Any] = {
        "type": "realtime",
        "model": model,
        "output_modalities": list(output_modalities),
        "tools": list(tools or []),
        "audio": {
            "input": {
                "format": input_format,
                "transcription": input_transcription,
                "noise_reduction": noise_reduction,
                "turn_detection": turn_detection,
            },
            "output": {
                "format": output_format,
                "voice": voice,
            },
        },
    }
    if output_speed is not None:
        session["audio"]["output"]["speed"] = output_speed
    if max_output_tokens is not None:
        session["max_output_tokens"] = max_output_tokens
    if instructions:
        session["instructions"] = instructions
    return {"type": "session.update", "session": session}


def build_realtime_text_conversation_item(text: str) -> dict[str, Any]:
    return {
        "type": "conversation.item.create",
        "item": {
            "type": "message",
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": text,
                }
            ],
        },
    }


def build_realtime_input_audio_buffer_append(audio: bytes) -> dict[str, Any]:
    if not isinstance(audio, bytes) or not audio:
        raise ValueError("audio must be non-empty bytes")
    return {
        "type": REALTIME_INPUT_AUDIO_BUFFER_APPEND_EVENT,
        "audio": base64.b64encode(audio).decode("ascii"),
    }


def build_realtime_input_audio_buffer_commit() -> dict[str, Any]:
    return {"type": REALTIME_INPUT_AUDIO_BUFFER_COMMIT_EVENT}


def build_realtime_response_create(
    *,
    instructions: str | None = None,
    voice: str | None = None,
    output_format: dict[str, Any] | None = None,
    output_modalities: tuple[str, ...] | list[str] = ("audio",),
    conversation: str | None = None,
    response_input: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    response: dict[str, Any] = {"output_modalities": list(output_modalities)}
    if instructions:
        response["instructions"] = instructions
    if voice is not None or output_format is not None:
        audio_output: dict[str, Any] = {}
        if output_format is not None:
            audio_output["format"] = output_format
        if voice is not None:
            audio_output["voice"] = voice
        response["audio"] = {"output": audio_output}
    if conversation is not None:
        response["conversation"] = conversation
    if response_input is not None:
        response["input"] = response_input
    return {"type": "response.create", "response": response}


def build_realtime_response_cancel(
    *,
    response_id: str | None = None,
) -> dict[str, Any]:
    event: dict[str, Any] = {"type": REALTIME_RESPONSE_CANCEL_EVENT}
    normalized_response_id = _normalize_optional_id("response_id", response_id)
    if normalized_response_id is not None:
        event["response_id"] = normalized_response_id
    return event


def build_realtime_conversation_item_truncate(
    *,
    item_id: str,
    played_ms: int,
    content_index: int = 0,
) -> dict[str, Any]:
    return {
        "type": REALTIME_CONVERSATION_ITEM_TRUNCATE_EVENT,
        "item_id": _normalize_required_id("item_id", item_id),
        "content_index": _validate_non_negative_int("content_index", content_index),
        "audio_end_ms": _validate_non_negative_int("played_ms", played_ms),
    }


class RealtimeSpeechSession:
    def __init__(
        self,
        transport: Any,
        *,
        config: RealtimeSpeechConfig | None = None,
    ) -> None:
        self.transport = transport
        self.config = config or RealtimeSpeechConfig()
        self._current_response_id: str | None = None
        self._current_item_id: str | None = None
        self._current_content_index = 0
        self._output_audio_progress: dict[
            tuple[str, int], _RealtimeOutputAudioProgress
        ] = {}

    async def configure(self) -> None:
        await self._send_json(
            build_realtime_speech_session_update(
                model=self.config.model,
                instructions=self.config.instructions,
                voice=self.config.voice,
                output_speed=self.config.output_speed,
                input_format=self.config.input_format,
                input_transcription=self.config.input_transcription,
                noise_reduction=self.config.noise_reduction,
                turn_detection=self.config.turn_detection,
                output_format=self.config.output_format,
                output_modalities=self.config.output_modalities,
                tools=self.config.tools,
                max_output_tokens=self.config.max_output_tokens,
            )
        )

    async def stream_audio_response(
        self,
        *,
        session_id: str,
        turn_id: int,
        speaker: str,
        latest_input: str,
        sample_rate: int,
        sample_width_bits: int,
        channels: int,
        delivery_mode: AudioDeliveryMode,
        response_instructions: str | None = None,
        response_instructions_record: ResponseCreateInstructions | None = None,
        response_instructions_observer: ResponseInstructionsObserver | None = None,
        create_conversation_item: bool = True,
        response_conversation: str | None = None,
        response_input: list[dict[str, Any]] | None = None,
    ) -> AsyncIterator[RealtimeSpeechEvent]:
        self._reset_current_output_reference()
        if create_conversation_item and response_input is None:
            await self._send_json(build_realtime_text_conversation_item(latest_input))
        await self._send_json(
            build_realtime_response_create(
                instructions=response_instructions,
                voice=self.config.voice,
                output_format=self.config.output_format,
                output_modalities=self.config.output_modalities,
                conversation=response_conversation,
                response_input=response_input,
            )
        )
        await _notify_response_instructions_observer(
            response_instructions_observer,
            response_instructions_record,
        )

        chunk_index = 0
        pending_pcm = b""
        ready_pcm = b""
        bytes_per_frame = _bytes_per_frame(
            sample_width_bits=sample_width_bits,
            channels=channels,
        )
        target_chunk_bytes = _target_chunk_bytes(
            sample_rate=sample_rate,
            bytes_per_frame=bytes_per_frame,
            target_chunk_duration_ms=self.config.target_chunk_duration_ms,
        )
        active_response_id: str | None = None

        async for event in self.transport:
            event_type = _event_value(event, "type")
            if event_type == REALTIME_ERROR_EVENT:
                if _is_nonfatal_realtime_error_event(event):
                    continue
                raise RealtimeSpeechError(_format_realtime_error_event(event))

            event_response_id = _extract_response_id(event)
            if event_type == REALTIME_RESPONSE_CREATED_EVENT:
                if event_response_id is not None:
                    active_response_id = event_response_id
                    self._current_response_id = event_response_id
                continue
            if _is_stale_response_event(
                event_response_id=event_response_id,
                active_response_id=active_response_id,
            ):
                continue

            text_delta = _extract_text_delta(event)
            if text_delta:
                yield RealtimeSpeechEvent(text_delta=text_delta)

            if event_type == REALTIME_OUTPUT_AUDIO_DELTA_EVENT:
                self._remember_current_output_reference(event)
                pending_pcm, ready_pcm, emitted, received_bytes = _buffer_audio_delta(
                    event,
                    pending_pcm=pending_pcm,
                    ready_pcm=ready_pcm,
                    bytes_per_frame=bytes_per_frame,
                    target_chunk_bytes=target_chunk_bytes,
                )
                self._remember_current_output_audio_size(
                    received_bytes,
                    sample_rate=sample_rate,
                    bytes_per_frame=bytes_per_frame,
                )
                for pcm in emitted:
                    yield RealtimeSpeechEvent(
                        audio_chunk=_build_audio_chunk(
                            session_id=session_id,
                            turn_id=turn_id,
                            speaker=speaker,
                            chunk_index=chunk_index,
                            pcm=pcm,
                            sample_rate=sample_rate,
                            sample_width_bits=sample_width_bits,
                            channels=channels,
                            delivery_mode=delivery_mode,
                        )
                    )
                    chunk_index += 1
                continue

            if event_type == REALTIME_OUTPUT_AUDIO_DONE_EVENT:
                self._remember_current_output_reference(event)
                if pending_pcm:
                    raise RealtimeSpeechError(
                        "Realtime output audio ended with an incomplete PCM frame"
                    )
                if ready_pcm:
                    yield RealtimeSpeechEvent(
                        audio_chunk=_build_audio_chunk(
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
                    )
                    ready_pcm = b""
                    chunk_index += 1
                continue

            if event_type == REALTIME_RESPONSE_DONE_EVENT:
                if pending_pcm:
                    raise RealtimeSpeechError(
                        "Realtime response ended with an incomplete PCM frame"
                    )
                if ready_pcm:
                    yield RealtimeSpeechEvent(
                        audio_chunk=_build_audio_chunk(
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
                    )
                return

        raise RealtimeSpeechError("Realtime speech stream ended before response.done")

    async def stream_audio_input_response(
        self,
        *,
        session_id: str,
        turn_id: int,
        speaker: str,
        input_audio: bytes,
        sample_rate: int,
        sample_width_bits: int,
        channels: int,
        delivery_mode: AudioDeliveryMode,
        response_instructions: str | None = None,
        response_instructions_record: ResponseCreateInstructions | None = None,
        response_instructions_observer: ResponseInstructionsObserver | None = None,
        response_conversation: str | None = None,
    ) -> AsyncIterator[RealtimeSpeechEvent]:
        self._reset_current_output_reference()
        await self._send_json(build_realtime_input_audio_buffer_append(input_audio))
        await self._send_json(build_realtime_input_audio_buffer_commit())
        await self._send_json(
            build_realtime_response_create(
                instructions=response_instructions,
                voice=self.config.voice,
                output_format=self.config.output_format,
                output_modalities=self.config.output_modalities,
                conversation=response_conversation,
            )
        )
        await _notify_response_instructions_observer(
            response_instructions_observer,
            response_instructions_record,
        )

        chunk_index = 0
        pending_pcm = b""
        ready_pcm = b""
        bytes_per_frame = _bytes_per_frame(
            sample_width_bits=sample_width_bits,
            channels=channels,
        )
        target_chunk_bytes = _target_chunk_bytes(
            sample_rate=sample_rate,
            bytes_per_frame=bytes_per_frame,
            target_chunk_duration_ms=self.config.target_chunk_duration_ms,
        )
        active_response_id: str | None = None
        response_done = False
        input_transcription_completed = False
        iterator = self.transport.__aiter__()

        while True:
            timeout_seconds = (
                _REALTIME_INPUT_TRANSCRIPTION_GRACE_SECONDS
                if response_done and not input_transcription_completed
                else None
            )
            try:
                event = await _next_async_event(
                    iterator,
                    timeout_seconds=timeout_seconds,
                )
            except asyncio.TimeoutError:
                return
            except StopAsyncIteration:
                if response_done:
                    return
                raise RealtimeSpeechError(
                    "Realtime speech stream ended before response.done"
                )

            event_type = _event_value(event, "type")
            if event_type == REALTIME_ERROR_EVENT:
                if _is_nonfatal_realtime_error_event(event):
                    continue
                raise RealtimeSpeechError(_format_realtime_error_event(event))

            input_transcript_delta = _extract_input_transcript_delta(event)
            if input_transcript_delta:
                yield RealtimeSpeechEvent(
                    input_transcript_delta=input_transcript_delta,
                )

            input_transcript_completed = _extract_input_transcript_completed(event)
            if input_transcript_completed is not None:
                yield RealtimeSpeechEvent(
                    input_transcript_completed=input_transcript_completed,
                )
                input_transcription_completed = True
                if response_done:
                    return

            event_response_id = _extract_response_id(event)
            if event_type == REALTIME_RESPONSE_CREATED_EVENT:
                if event_response_id is not None:
                    active_response_id = event_response_id
                    self._current_response_id = event_response_id
                continue
            if _is_stale_response_event(
                event_response_id=event_response_id,
                active_response_id=active_response_id,
            ):
                continue

            text_delta = _extract_text_delta(event)
            if text_delta:
                yield RealtimeSpeechEvent(text_delta=text_delta)

            if event_type == REALTIME_OUTPUT_AUDIO_DELTA_EVENT:
                self._remember_current_output_reference(event)
                pending_pcm, ready_pcm, emitted, received_bytes = _buffer_audio_delta(
                    event,
                    pending_pcm=pending_pcm,
                    ready_pcm=ready_pcm,
                    bytes_per_frame=bytes_per_frame,
                    target_chunk_bytes=target_chunk_bytes,
                )
                self._remember_current_output_audio_size(
                    received_bytes,
                    sample_rate=sample_rate,
                    bytes_per_frame=bytes_per_frame,
                )
                for pcm in emitted:
                    yield RealtimeSpeechEvent(
                        audio_chunk=_build_audio_chunk(
                            session_id=session_id,
                            turn_id=turn_id,
                            speaker=speaker,
                            chunk_index=chunk_index,
                            pcm=pcm,
                            sample_rate=sample_rate,
                            sample_width_bits=sample_width_bits,
                            channels=channels,
                            delivery_mode=delivery_mode,
                        )
                    )
                    chunk_index += 1
                continue

            if event_type == REALTIME_OUTPUT_AUDIO_DONE_EVENT:
                self._remember_current_output_reference(event)
                if pending_pcm:
                    raise RealtimeSpeechError(
                        "Realtime output audio ended with an incomplete PCM frame"
                    )
                if ready_pcm:
                    yield RealtimeSpeechEvent(
                        audio_chunk=_build_audio_chunk(
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
                    )
                    ready_pcm = b""
                    chunk_index += 1
                continue

            if event_type == REALTIME_RESPONSE_DONE_EVENT:
                if pending_pcm:
                    raise RealtimeSpeechError(
                        "Realtime response ended with an incomplete PCM frame"
                    )
                if ready_pcm:
                    yield RealtimeSpeechEvent(
                        audio_chunk=_build_audio_chunk(
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
                    )
                    ready_pcm = b""
                response_done = True
                if input_transcription_completed:
                    return

    async def stop_current_response_playback(
        self,
        *,
        played_ms: int,
        item_id: str | None = None,
        response_id: str | None = None,
        content_index: int | None = None,
        cancel_response: bool = True,
        truncate_item: bool = True,
    ) -> RealtimePlaybackStopResult:
        played_ms = _validate_non_negative_int("played_ms", played_ms)
        resolved_response_id = (
            _normalize_required_id("response_id", response_id)
            if response_id is not None
            else self._current_response_id
        )
        resolved_item_id = (
            _normalize_required_id("item_id", item_id)
            if item_id is not None
            else self._current_item_id
        )
        resolved_content_index = _validate_non_negative_int(
            "content_index",
            self._current_content_index if content_index is None else content_index,
        )

        sent_event_types: list[str] = []
        if cancel_response:
            await self._send_json(
                build_realtime_response_cancel(response_id=resolved_response_id)
            )
            sent_event_types.append(REALTIME_RESPONSE_CANCEL_EVENT)

        truncate_sent = False
        audio_end_ms = None
        if truncate_item and resolved_item_id is not None:
            progress = self._output_audio_progress.get(
                (resolved_item_id, resolved_content_index)
            )
            output_speed = self.config.output_speed or 1.0
            if progress is not None:
                # Browser playback time can include pauses between audio batches.
                # The API rejects truncation beyond the item's actual audio length.
                played_ms = min(played_ms, progress.available_ms)
                output_speed = progress.output_speed
            # Output speed is applied after generation. The server truncates the
            # original audio timeline, while played_ms measures delivered PCM.
            audio_end_ms = int(played_ms * output_speed)
            await self._send_json(
                build_realtime_conversation_item_truncate(
                    item_id=resolved_item_id,
                    played_ms=audio_end_ms,
                    content_index=resolved_content_index,
                )
            )
            if progress is not None:
                progress.truncated_at_ms = played_ms
            sent_event_types.append(REALTIME_CONVERSATION_ITEM_TRUNCATE_EVENT)
            truncate_sent = True

        return RealtimePlaybackStopResult(
            played_ms=played_ms,
            cancel_sent=cancel_response,
            truncate_sent=truncate_sent,
            item_id=resolved_item_id,
            response_id=resolved_response_id,
            content_index=resolved_content_index,
            sent_event_types=tuple(sent_event_types),
            audio_end_ms=audio_end_ms,
        )

    async def _send_json(self, event: dict[str, Any]) -> None:
        result = self.transport.send_json(event)
        if inspect.isawaitable(result):
            await result

    def _reset_current_output_reference(self) -> None:
        self._current_response_id = None
        self._current_item_id = None
        self._current_content_index = 0

    def _remember_current_output_reference(self, event: Any) -> None:
        response_id = _event_value(event, "response_id")
        if response_id is not None:
            self._current_response_id = str(response_id)
        item_id = _event_value(event, "item_id")
        if item_id is not None:
            self._current_item_id = str(item_id)
        content_index = _event_value(event, "content_index")
        if (
            isinstance(content_index, int)
            and not isinstance(content_index, bool)
            and content_index >= 0
        ):
            self._current_content_index = content_index

    def _remember_current_output_audio_size(
        self,
        received_bytes: int,
        *,
        sample_rate: int,
        bytes_per_frame: int,
    ) -> None:
        if self._current_item_id is None:
            return
        key = (self._current_item_id, self._current_content_index)
        progress = self._output_audio_progress.get(key)
        if progress is None:
            progress = _RealtimeOutputAudioProgress(
                sample_rate=sample_rate,
                bytes_per_frame=bytes_per_frame,
                output_speed=self.config.output_speed or 1.0,
            )
            self._output_audio_progress[key] = progress
        progress.received_bytes += received_bytes


class OpenAIRealtimeSpeechAgent:

    def __init__(
        self,
        *,
        speaker: str,
        api_key: str | None = None,
        config: RealtimeSpeechConfig | None = None,
        input_formatter: AgentInputFormatter | None = None,
        response_input_formatter: Callable[..., list[dict[str, Any]]] | None = None,
        transport_context_factory: (
            Callable[[], AbstractAsyncContextManager[Any]] | None
        ) = None,
        connect: Any | None = None,
        reuse_transport: bool = True,
        timing_llm: StreamingLLMLike | None = None,
        timing_system_prompt: str = TIMING_DECISION_SYSTEM_PROMPT,
    ) -> None:
        self.speaker = speaker
        self.api_key = api_key
        self.config = config or RealtimeSpeechConfig()
        self.input_formatter = input_formatter
        self.response_input_formatter = response_input_formatter
        self.transport_context_factory = transport_context_factory
        self.connect = connect
        self.reuse_transport = reuse_transport
        self.timing_llm = timing_llm
        self.timing_system_prompt = timing_system_prompt
        self.received_inputs: list[str] = []
        self.received_timing_inputs: list[str] = []
        self._connection_lock = asyncio.Lock()
        self._operation_lock = asyncio.Lock()
        self._active_transport_context: AbstractAsyncContextManager[Any] | None = None
        self._active_session: RealtimeSpeechSession | None = None

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
        response_instructions_observer: ResponseInstructionsObserver | None = None,
    ) -> AsyncIterator[RealtimeSpeechEvent]:
        response_input = None
        latest_input = input_transcript
        if format_input and self.response_input_formatter is not None:
            response_input = self.response_input_formatter(
                speaker=self.speaker,
                turn_id=turn_id,
                input_transcript=input_transcript,
            )
            latest_input = json.dumps(response_input, ensure_ascii=False)
        elif format_input and self.input_formatter is not None:
            latest_input = _format_agent_input(
                self.input_formatter,
                speaker=self.speaker,
                turn_id=turn_id,
                input_transcript=input_transcript,
                purpose="generation",
            )
        self.received_inputs.append(latest_input)
        if format_input:
            response_instructions = build_response_instructions(
                self.config.instructions,
                additional_instruction,
                response_instructions=self.config.response_instructions,
                repeat_base_instructions=(self.config.repeat_instructions_per_response),
            )
            create_conversation_item = True
            response_conversation = None
        else:
            response_instructions = build_scripted_audio_response_instructions(
                latest_input,
                additional_instruction,
            )
            create_conversation_item = False
            response_conversation = "none"
        response_instructions_record = ResponseCreateInstructions(
            session_id=session_id,
            turn_id=turn_id,
            speaker=speaker,
            session_instructions=self.config.instructions,
            participant_instructions=(
                self.config.response_instructions if format_input else ""
            ),
            additional_instructions=(
                (additional_instruction or "").strip() if format_input else ""
            ),
            resolved_instructions=response_instructions or "",
            repeat_session_instructions=(
                bool(self.config.repeat_instructions_per_response)
                if format_input
                else False
            ),
            input_text=latest_input if response_input is None else "",
            response_input=response_input,
            response_conversation=response_conversation,
        )

        if self.reuse_transport and format_input:
            async with self._operation_lock:
                for attempt in range(2):
                    session = await self._get_active_session()
                    emitted_event = False
                    try:
                        async for event in session.stream_audio_response(
                            session_id=session_id,
                            turn_id=turn_id,
                            speaker=speaker,
                            latest_input=latest_input,
                            sample_rate=sample_rate,
                            sample_width_bits=sample_width_bits,
                            channels=channels,
                            delivery_mode=delivery_mode,
                            response_instructions=response_instructions,
                            response_instructions_record=response_instructions_record,
                            response_instructions_observer=(
                                response_instructions_observer
                            ),
                            create_conversation_item=create_conversation_item,
                            response_conversation=response_conversation,
                            response_input=response_input,
                        ):
                            emitted_event = True
                            yield event
                        return
                    except Exception as exc:
                        await self.close()
                        if (
                            attempt == 0
                            and not emitted_event
                            and (
                                _is_realtime_session_recoverable_close(exc)
                                or _is_realtime_active_response_conflict(exc)
                            )
                        ):
                            continue
                        raise
            return

        async with self._open_transport() as transport:
            session = self._build_session(transport)
            await session.configure()
            async for event in session.stream_audio_response(
                session_id=session_id,
                turn_id=turn_id,
                speaker=speaker,
                latest_input=latest_input,
                sample_rate=sample_rate,
                sample_width_bits=sample_width_bits,
                channels=channels,
                delivery_mode=delivery_mode,
                response_instructions=response_instructions,
                response_instructions_record=response_instructions_record,
                response_instructions_observer=response_instructions_observer,
                create_conversation_item=create_conversation_item,
                response_conversation=response_conversation,
                response_input=response_input,
            ):
                yield event

    async def close(self) -> None:
        async with self._connection_lock:
            active_transport_context = self._active_transport_context
            self._active_transport_context = None
            self._active_session = None
        if active_transport_context is not None:
            await active_transport_context.__aexit__(None, None, None)

    async def stream_audio_response_from_audio(
        self,
        *,
        session_id: str,
        turn_id: int,
        speaker: str,
        input_audio: bytes,
        additional_instruction: str | None = None,
        sample_rate: int,
        sample_width_bits: int,
        channels: int,
        delivery_mode: AudioDeliveryMode,
        response_instructions_observer: ResponseInstructionsObserver | None = None,
    ) -> AsyncIterator[RealtimeSpeechEvent]:
        latest_input = "[audio]"
        if self.input_formatter is not None:
            latest_input = _format_agent_input(
                self.input_formatter,
                speaker=self.speaker,
                turn_id=turn_id,
                input_transcript="[audio]",
                purpose="generation",
            )
        self.received_inputs.append(latest_input)
        audio_context_instruction = (
            "直前のユーザー発話は、このRealtime sessionのinput audioとして"
            "すでに渡されています。以下の会話文脈と参加者設定を踏まえ、"
            "音声内容に直接応答してください。\n\n"
            f"{latest_input}"
            if latest_input != "[audio]"
            else ""
        )
        combined_additional_instruction = "\n\n".join(
            part for part in (audio_context_instruction, additional_instruction) if part
        )
        response_instructions = build_response_instructions(
            self.config.instructions,
            combined_additional_instruction,
            response_instructions=self.config.response_instructions,
            repeat_base_instructions=(self.config.repeat_instructions_per_response),
        )
        response_instructions_record = ResponseCreateInstructions(
            session_id=session_id,
            turn_id=turn_id,
            speaker=speaker,
            session_instructions=self.config.instructions,
            participant_instructions=self.config.response_instructions,
            additional_instructions=combined_additional_instruction.strip(),
            resolved_instructions=response_instructions or "",
            repeat_session_instructions=bool(
                self.config.repeat_instructions_per_response
            ),
            input_text=latest_input,
        )

        if self.reuse_transport:
            async with self._operation_lock:
                for attempt in range(2):
                    session = await self._get_active_session()
                    emitted_event = False
                    try:
                        async for event in session.stream_audio_input_response(
                            session_id=session_id,
                            turn_id=turn_id,
                            speaker=speaker,
                            input_audio=input_audio,
                            sample_rate=sample_rate,
                            sample_width_bits=sample_width_bits,
                            channels=channels,
                            delivery_mode=delivery_mode,
                            response_instructions=response_instructions,
                            response_instructions_record=response_instructions_record,
                            response_instructions_observer=(
                                response_instructions_observer
                            ),
                        ):
                            emitted_event = True
                            yield event
                        return
                    except Exception as exc:
                        await self.close()
                        if (
                            attempt == 0
                            and not emitted_event
                            and (
                                _is_realtime_session_recoverable_close(exc)
                                or _is_realtime_active_response_conflict(exc)
                            )
                        ):
                            continue
                        raise
            return

        async with self._open_transport() as transport:
            session = self._build_session(transport)
            await session.configure()
            async for event in session.stream_audio_input_response(
                session_id=session_id,
                turn_id=turn_id,
                speaker=speaker,
                input_audio=input_audio,
                sample_rate=sample_rate,
                sample_width_bits=sample_width_bits,
                channels=channels,
                delivery_mode=delivery_mode,
                response_instructions=response_instructions,
                response_instructions_record=response_instructions_record,
                response_instructions_observer=response_instructions_observer,
            ):
                yield event

    async def stop_current_response_playback(
        self,
        *,
        played_ms: int,
        item_id: str | None = None,
        response_id: str | None = None,
        content_index: int | None = None,
        cancel_response: bool = True,
        truncate_item: bool = True,
    ) -> RealtimePlaybackStopResult:
        async with self._connection_lock:
            session = self._active_session
        if session is None:
            raise RealtimeSpeechError("no active realtime speech session")
        return await session.stop_current_response_playback(
            played_ms=played_ms,
            item_id=item_id,
            response_id=response_id,
            content_index=content_index,
            cancel_response=cancel_response,
            truncate_item=truncate_item,
        )

    async def decide_timing(
        self,
        *,
        input_transcript: str,
        turn_id: int,
        previous_speaker: str | None = None,
    ) -> TimingDecisionInput:
        if self.timing_llm is None:
            raise RealtimeSpeechError("realtime speech agent has no timing LLM")
        latest_input = input_transcript
        if self.input_formatter is not None:
            latest_input = _format_agent_input(
                self.input_formatter,
                speaker=self.speaker,
                turn_id=turn_id,
                input_transcript=input_transcript,
                purpose="timing",
            )
        timing_input = build_timing_decision_input(
            speaker=self.speaker,
            turn_id=turn_id,
            input_transcript=latest_input,
            previous_speaker=previous_speaker,
        )
        self.received_timing_inputs.append(timing_input)
        parts = [
            part
            async for part in self.timing_llm.stream_text(
                latest_input=timing_input,
                system_prompt=self.timing_system_prompt,
            )
        ]
        return parse_timing_decision_response("".join(parts))

    async def _get_active_session(self) -> RealtimeSpeechSession:
        async with self._connection_lock:
            if self._active_session is not None:
                return self._active_session

            transport_context = self._open_transport()
            transport: Any | None = None
            try:
                transport = await transport_context.__aenter__()
                session = self._build_session(transport)
                await session.configure()
            except Exception:
                if transport is not None:
                    await transport_context.__aexit__(None, None, None)
                raise

            self._active_transport_context = transport_context
            self._active_session = session
            return session

    def _open_transport(self) -> AbstractAsyncContextManager[Any]:
        if self.transport_context_factory is not None:
            return self.transport_context_factory()
        if not self.api_key or not self.api_key.strip():
            raise ValueError("api_key is required without transport_context_factory")

        from counseling_voice_demo.runtime.realtime_transport import (
            connect_realtime_session,
        )

        return connect_realtime_session(
            api_key=self.api_key,
            model=self.config.model,
            connect=self.connect,
        )

    def _build_session(self, transport: Any) -> RealtimeSpeechSession:
        return RealtimeSpeechSession(transport, config=self.config)


def _extract_text_delta(event: Any) -> str | None:
    event_type = _event_value(event, "type")
    if event_type == REALTIME_OUTPUT_AUDIO_TRANSCRIPT_DELTA_EVENT:
        return _string_delta(event)
    if event_type == REALTIME_OUTPUT_TEXT_DELTA_EVENT:
        return _string_delta(event)
    return None


def _extract_input_transcript_delta(event: Any) -> str | None:
    event_type = _event_value(event, "type")
    if event_type != REALTIME_INPUT_AUDIO_TRANSCRIPTION_DELTA_EVENT:
        return None
    return _string_delta(event)


def _extract_input_transcript_completed(event: Any) -> str | None:
    event_type = _event_value(event, "type")
    if event_type != REALTIME_INPUT_AUDIO_TRANSCRIPTION_COMPLETED_EVENT:
        return None
    transcript = _event_value(event, "transcript")
    return "" if transcript is None else str(transcript)


def _extract_response_id(event: Any) -> str | None:
    response_id = _event_value(event, "response_id")
    if response_id is not None:
        return str(response_id)
    response = _event_value(event, "response")
    if isinstance(response, dict):
        nested_response_id = response.get("id")
    else:
        nested_response_id = getattr(response, "id", None)
    if nested_response_id is None:
        return None
    return str(nested_response_id)


def _is_stale_response_event(
    *,
    event_response_id: str | None,
    active_response_id: str | None,
) -> bool:
    if event_response_id is None:
        return False
    if active_response_id is None:
        return True
    return event_response_id != active_response_id


async def _next_async_event(
    iterator: Any,
    *,
    timeout_seconds: float | None,
) -> Any:
    next_event = iterator.__anext__()
    if timeout_seconds is None:
        return await next_event
    return await asyncio.wait_for(next_event, timeout=timeout_seconds)


def _format_agent_input(
    formatter: AgentInputFormatter,
    *,
    speaker: str,
    turn_id: int,
    input_transcript: str,
    purpose: str,
    response_target: str | None = None,
) -> str:
    kwargs: dict[str, Any] = {
        "speaker": speaker,
        "turn_id": turn_id,
        "input_transcript": input_transcript,
    }
    if _callable_accepts_keyword(formatter, "purpose"):
        kwargs["purpose"] = purpose
    if response_target is not None and _callable_accepts_keyword(
        formatter,
        "response_target",
    ):
        kwargs["response_target"] = response_target
    return formatter(**kwargs)


def _callable_accepts_keyword(callable_object: Callable[..., Any], key: str) -> bool:
    try:
        signature = inspect.signature(callable_object)
    except (TypeError, ValueError):
        return False
    return key in signature.parameters or any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )


def _string_delta(event: Any) -> str | None:
    delta = _event_value(event, "delta")
    if delta is None:
        return None
    return str(delta)


def _buffer_audio_delta(
    event: Any,
    *,
    pending_pcm: bytes,
    ready_pcm: bytes,
    bytes_per_frame: int,
    target_chunk_bytes: int,
) -> tuple[bytes, bytes, list[bytes], int]:
    encoded_audio = _event_value(event, "delta")
    if not isinstance(encoded_audio, str):
        raise RealtimeSpeechError("Realtime output audio delta must be a base64 string")
    try:
        decoded_pcm = base64.b64decode(encoded_audio, validate=True)
    except ValueError as exc:
        raise RealtimeSpeechError(
            "Realtime output audio delta is not valid base64"
        ) from exc

    pcm = pending_pcm + decoded_pcm
    complete_byte_length = len(pcm) - (len(pcm) % bytes_per_frame)
    pending_pcm = pcm[complete_byte_length:]
    ready_pcm += pcm[:complete_byte_length]

    emitted: list[bytes] = []
    while len(ready_pcm) >= target_chunk_bytes:
        emitted.append(ready_pcm[:target_chunk_bytes])
        ready_pcm = ready_pcm[target_chunk_bytes:]
    return pending_pcm, ready_pcm, emitted, len(decoded_pcm)


def _build_audio_chunk(
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


def _bytes_per_frame(*, sample_width_bits: int, channels: int) -> int:
    if sample_width_bits <= 0 or sample_width_bits % 8 != 0:
        raise ValueError("sample_width_bits must be a positive multiple of 8")
    if channels <= 0:
        raise ValueError("channels must be positive")
    return (sample_width_bits // 8) * channels


def _target_chunk_bytes(
    *,
    sample_rate: int,
    bytes_per_frame: int,
    target_chunk_duration_ms: int,
) -> int:
    frames = max(1, sample_rate * target_chunk_duration_ms // 1000)
    return frames * bytes_per_frame


def _normalize_optional_id(name: str, value: str | None) -> str | None:
    if value is None:
        return None
    return _normalize_required_id(name, value)


def _normalize_required_id(name: str, value: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{name} must not be empty")
    return normalized


def _validate_non_negative_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


def _event_value(event: Any, key: str) -> Any:
    if isinstance(event, dict):
        return event.get(key)
    return getattr(event, key, None)


def _format_realtime_error_event(event: Any) -> str:
    error = _event_value(event, "error")
    if not isinstance(error, dict):
        return "Realtime speech error"

    parts: list[str] = ["Realtime speech error"]
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


def _is_nonfatal_realtime_error_event(event: Any) -> bool:
    error = _event_value(event, "error")
    if not isinstance(error, dict):
        return False
    return error.get("code") == "response_cancel_not_active"


def _is_realtime_session_recoverable_close(exc: BaseException) -> bool:
    message = str(exc).lower()
    return (
        "session hit the maximum duration" in message
        or "maximum duration of 60 minutes" in message
        or "no close frame received or sent" in message
    )


async def _notify_response_instructions_observer(
    observer: ResponseInstructionsObserver | None,
    record: ResponseCreateInstructions | None,
) -> None:
    if observer is None or record is None:
        return
    result = observer(record)
    if inspect.isawaitable(result):
        await result


def _is_realtime_active_response_conflict(exc: BaseException) -> bool:
    return "conversation_already_has_active_response" in str(exc).lower()
