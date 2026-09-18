from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable, Iterable, Mapping
from typing import Any, Protocol, runtime_checkable

from counseling_voice_demo.runtime.floor_mediator import TimingDecisionInput
from counseling_voice_demo.runtime.models import AudioChunk, AudioDeliveryMode, TranscriptEvent


TranscriptObserver = Callable[[TranscriptEvent], Awaitable[None] | None]


class AgentLike(Protocol):
    async def generate(
        self,
        *,
        input_transcript: str,
        turn_id: int,
        additional_instruction: str | None = None,
    ) -> Iterable[str]:
        ...


@runtime_checkable
class StreamingAgentLike(Protocol):
    def stream_generate(
        self,
        *,
        input_transcript: str,
        turn_id: int,
        additional_instruction: str | None = None,
    ) -> AsyncIterator[str]:
        ...


@runtime_checkable
class TimingDecisionAgentLike(Protocol):
    async def decide_timing(
        self,
        *,
        input_transcript: str,
        turn_id: int,
        previous_speaker: str | None = None,
    ) -> TimingDecisionInput:
        ...


@runtime_checkable
class OverlapDecisionAgentLike(Protocol):
    async def decide_overlap(
        self,
        *,
        input_transcript: str,
        turn_id: int,
        conflict_agent_ids: tuple[str, ...],
        active_speaker_id: str | None = None,
    ) -> TimingDecisionInput:
        ...


class StreamingLLMLike(Protocol):
    def stream_text(
        self,
        *,
        latest_input: str,
        history: Iterable[Mapping[str, Any]] | None = None,
        system_prompt: str | None = None,
        text_format: Mapping[str, Any] | None = None,
    ) -> AsyncIterator[str]:
        ...


class StreamingTTSLike(Protocol):
    def synthesize(
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
        ...


class StreamingSTTLike(Protocol):
    async def transcribe(
        self,
        *,
        session_id: str,
        turn_id: int,
        speaker: str,
        chunks: Iterable[AudioChunk],
    ) -> tuple[list[TranscriptEvent], TranscriptEvent]:
        ...


@runtime_checkable
class ObservableStreamingSTTLike(Protocol):
    async def transcribe_observed(
        self,
        *,
        session_id: str,
        turn_id: int,
        speaker: str,
        chunks: Iterable[AudioChunk],
        on_transcript: TranscriptObserver,
    ) -> tuple[list[TranscriptEvent], TranscriptEvent]:
        ...


@runtime_checkable
class QueueStreamingSTTLike(Protocol):
    async def transcribe_from_queue(
        self,
        *,
        session_id: str,
        turn_id: int,
        speaker: str,
        queue: Any,
        recording_mode: str = "push_to_talk",
    ) -> tuple[list[TranscriptEvent], TranscriptEvent]:
        ...


@runtime_checkable
class ObservableQueueStreamingSTTLike(Protocol):
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
        ...
