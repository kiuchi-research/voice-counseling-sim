from __future__ import annotations

import asyncio
import hashlib
import inspect
import random
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Iterable, Mapping
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Final
from uuid import uuid4

from counseling_voice_demo.runtime.agents import default_fake_agents
from counseling_voice_demo.runtime.audio_bus import AudioBus
from counseling_voice_demo.runtime.floor_mediator import (
    FloorMediator,
    FloorMediatorResult,
    FloorMediatorResultType,
    TimingDecision,
    TimingDecisionAction,
    TimingDecisionValidationError,
    timing_decision_conflict_priority,
)
from counseling_voice_demo.runtime.logger import AsyncLogger
from counseling_voice_demo.runtime.models import (
    ActorKind,
    AudioChunk,
    AudioDeliveryMode,
    EndOfAudio,
    HumanAudioInput,
    HumanAudioStreamStart,
    HumanTurnInput,
    InteractionMode,
    ParticipantConfig,
    PromptDirectorAttemptRecord,
    ResponseInstructionsRecord,
    RuntimeConfig,
    RuntimeEvent,
    RuntimeMetric,
    RuntimePhase,
    RuntimeStatus,
    RuntimeStopCondition,
    TranscriptEvent,
    TurnRuntimeState,
)
from counseling_voice_demo.runtime.protocols import (
    AgentLike,
    QueueStreamingSTTLike,
    StreamingLLMLike,
    StreamingAgentLike,
    StreamingSTTLike,
    StreamingTTSLike,
    TranscriptObserver,
)
from counseling_voice_demo.runtime.prompt_context import (
    PublicHistoryMessage,
    RuntimePromptContextStore,
)
from counseling_voice_demo.runtime.prompt_director import (
    ClosingAssessmentRequest,
    PromptDirector,
    PromptDirectorLike,
    PromptDirectorRequest,
    PromptDirectorResult,
    PromptDirectorRetriesExhausted,
    SessionEndContext,
    render_prompt_director_instruction,
)
from counseling_voice_demo.runtime.session_memory import (
    SessionSummaryRequest,
    SessionSummaryState,
    SessionSummaryUpdate,
    build_runtime_prompt_context,
    build_session_summary_request,
    build_session_summary_update,
    recent_public_history,
    should_update_session_summary,
    summarize_session_context,
)
from counseling_voice_demo.runtime.speaker_selection import (
    CLIENT_ROLE,
    COUNSELOR_ROLE,
    DISTRIBUTED_TIMING_POLICY,
    FixedSequenceFloorPredictor,
    FixedSpeakerSequence,
    SpeakerSelectionError,
    TURN_BOUNDARY_TIMING_POLICY,
)
from counseling_voice_demo.runtime.streaming_llm import StreamingLLMError, TtsTextChunker
from counseling_voice_demo.runtime.streaming_stt import FakeStreamingSTT
from counseling_voice_demo.runtime.streaming_tts import FakeStreamingTTS
from counseling_voice_demo.runtime.system_prompts import (
    COUNSELOR_SYSTEM_PROMPT,
)
from counseling_voice_demo.speaker_label_sanitizer import (
    strip_leading_speaker_label_from_text_parts,
)


_TTS_TEXT_QUEUE_DONE: Final = object()
_PROVISIONAL_TEXT_DONE: Final = object()
CLOSING_COUNSELOR_INSTRUCTION: Final = (
    "【クロージング指示】ここからセッションの終結に入ってください。"
    "進め方は設定されたカウンセラープロンプトに従ってください。"
    "終結に入った後もこの指示は毎ターン付くため、終了手順を最初からやり直す意味ではありません。"
    "履歴で既に行った終了確認と各人の回答を踏まえ、原文の次の行為へ進んでください。"
    "『まだ分からない』『まだ迷う』も回答です。回答済みの人へ同じ確認をせず、"
    "未回答の人がいればその人だけに確認し、全員回答済みなら回答を受け取って終結を進めてください。"
)
EARLY_CLOSING_PROGRESS_RATIO: Final[float] = 0.9
SESSION_TIME_INSTRUCTION: Final[str] = (
    "【時間枠と継続】原則として設定されたクロージング開始時刻まで相談を続けてください。"
    "具体的な進め方は設定プロンプトに従い、既に得た回答や本人には分からないことを"
    "同じ質問で求め続けず、相談者が実際に求めていたことと照合してください。"
    "一つの話題への追加情報がないことを、相談全体の終了希望や目的達成へ置き換えないでください。"
    "ただしクライアントが面接全体の終了を明確に希望した場合は、それを優先して終結してください。"
    "自分からの早期終了提案は時間枠の終盤だけにし、停滞だけを終了理由にしないでください。"
    "提案した場合も、全クライアントの終了への明確な同意と回答待ちがないことを確認してください。"
    "感謝や単なる相づちを終了への同意と決めつけないでください。"
)
CLOSING_CLIENT_INSTRUCTION: Final = (
    "【クロージング指示】ここからセッションの終結に協力してください。"
    "新しい話題を広げず、設定されたクライアントのプロンプト、人物像、直前の発話に沿って応答してください。"
    "直前に質問や確認があれば、その内容に自然に反応してください。"
    "まだ分からない、迷っている、答えたくない場合も、その人物として表してかまいません。"
    "終了の指示だけを理由に質問への反応を感謝や同意へ置き換えたり、"
    "納得・理解・改善が得られたことにしたりしないでください。"
)
SCRIPTED_REALTIME_SPEECH_INSTRUCTION: Final = (
    "【読み上げ指示】入力された本文を、一字一句そのまま音声として読み上げてください。"
    "本文以外の挨拶、相づち、説明、言い換え、追加の一文は入れないでください。"
)
_TURN_START_TIMING_DELAY_RANGES_MS: Final[dict[str, tuple[int, int]]] = {
    "immediate": (100, 200),
    "natural_pause": (400, 1200),
    "short_hold": (1500, 1800),
    "long_hold": (1900, 3000),
}
_TURN_START_HIGH_URGENCY_IMMEDIATE_THRESHOLD: Final = 0.85
_FALLBACK_RECENT_TURN_LIMIT: Final = 6
_FALLBACK_FIXED_SEQUENCE_SCORE: Final = 20
_FALLBACK_CLIENT_REPLY_CONTINUITY_SCORE: Final = 6
_FALLBACK_CLIENT_AFTER_COUNSELOR_SCORE: Final = 8
_FALLBACK_COUNSELOR_AFTER_CLIENT_SCORE: Final = 6
_FALLBACK_RECENTLY_SILENT_SCORE: Final = 8
_FALLBACK_RECENT_BALANCE_SCORE: Final = 3
_FALLBACK_EXPLICIT_TARGET_PREVIOUS_SCORE: Final = 6
_FALLBACK_PREVIOUS_TURN_TARGET_SCORE: Final = 10
_FALLBACK_EXPLICITLY_ADDRESSED_SCORE: Final = 18
_FALLBACK_TIMEOUT_PENALTY: Final = -6
_FALLBACK_PREVIOUS_SPEAKER_PENALTY: Final = -28
_FALLBACK_REPEATED_SPEAKER_PATTERN_PENALTY: Final = -6
_FALLBACK_REPEATED_SPEAKER_PATTERN_MAX_LENGTH: Final = 4
_CLIENT_REPLY_REPEATED_PATTERN_COUNSELOR_URGENCY_MARGIN: Final = 0.05
_CLIENT_REPLY_OPPORTUNITY_URGENCY_FLOOR: Final = 0.35
_CLIENT_REPLY_AUTO_PRIORITY_MAX_CONSECUTIVE_CLIENT_TURNS: Final = 1
_CLIENT_REPLY_EXTENDED_IMMEDIATE_TIMING_RANK: Final = 0
_CLIENT_REPLY_EXTENDED_URGENCY_FLOOR: Final = 0.75
_CLIENT_REPLY_EXTENDED_URGENCY_STEP: Final = 0.1
_CLIENT_REPLY_EXTENDED_IMMEDIATE_MARGIN: Final = 0.15
_CLIENT_REPLY_EXTENDED_IMMEDIATE_MARGIN_STEP: Final = 0.1
_CLIENT_REPLY_EXTENDED_DIRECT_REPLY_URGENCY_FLOOR: Final = 0.6
_CLIENT_REPLY_EXTENDED_DIRECT_REPLY_MARGIN: Final = 0.03
_CLIENT_REPLY_EXTENDED_COUNSELOR_TARGET_URGENCY_FLOOR: Final = 0.62
_CLIENT_REPLY_EXTENDED_COUNSELOR_TARGET_MARGIN: Final = 0.08
_CLIENT_CONTEXTUAL_REPLY_URGENCY_FLOOR: Final = 0.5
_CLIENT_CONTEXTUAL_REPLY_COUNSELOR_URGENCY_MARGIN: Final = 0.1
_EXPLICITLY_ADDRESSED_OVERRIDE_URGENCY_FLOOR: Final = 0.95
_HUMAN_TURN_TAKING_SIGNAL_MIN_CHARS: Final = 8


@dataclass
class ProvisionalGeneration:
    session_id: str
    turn_id: int
    speaker: str
    input_transcript: str
    source_turn_id: int
    queue: asyncio.Queue[Any | object] = field(default_factory=asyncio.Queue)
    generated_parts: list[str] = field(default_factory=list)
    task: asyncio.Task[None] | None = None
    started_monotonic: float = field(default_factory=time.monotonic)
    first_token_monotonic: float | None = None
    completed_monotonic: float | None = None
    error: BaseException | None = None
    status: str = "running"
    source: str = "stt_partial"
    prediction_reason: str | None = None
    candidate_speaker_ids: tuple[str, ...] = ()
    generation_modality: str = "text"
    response_target: str | None = None


@dataclass(frozen=True)
class SpeakerSelection:
    speaker: str
    timing_decision: TimingDecision | None = None
    response_target: str | None = None


@dataclass(frozen=True)
class HumanRealtimeAudioTurnInput:
    request: HumanAudioInput
    turn_id: int
    speaker: str
    target_speaker: str
    audio_chunk: AudioChunk
    source_audio_ref: str


@dataclass(frozen=True)
class HumanStreamingAudioTurnInput:
    request: HumanAudioStreamStart
    stream_id: str
    turn_id: int
    speaker: str
    audio_queue: asyncio.Queue[AudioChunk | EndOfAudio]
    source_audio_ref: str
    completion: asyncio.Future[dict[str, Any]]
    aborted: asyncio.Event = field(default_factory=asyncio.Event)
    committed: asyncio.Event = field(default_factory=asyncio.Event)
    abort_logged: asyncio.Event = field(default_factory=asyncio.Event)
    chunk_digests: dict[int, bytes] = field(default_factory=dict)


@dataclass(frozen=True)
class FallbackSpeakerCandidate:
    speaker_id: str
    score: int
    sequence_distance: int
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class FallbackSpeakerSelection:
    speaker_id: str
    base_speaker_id: str
    strategy: str
    candidates: tuple[FallbackSpeakerCandidate, ...]


class ProvisionalGenerationFailedError(RuntimeError):
    def __init__(self, provisional: ProvisionalGeneration) -> None:
        super().__init__("provisional generation failed")
        self.provisional = provisional


class GenerationInterruptedForHumanInput(RuntimeError):
    pass


@dataclass
class TurnTakingSignalTracker:
    turn_id: int
    previous_speaker: str | None
    min_chars: int
    collect: Callable[[str], Awaitable[list[TimingDecision]]]
    on_collected: Callable[[str, list[TimingDecision]], Awaitable[None]]
    _latest_text: str = ""
    _collected_text: str = ""
    _task: asyncio.Task[list[TimingDecision]] | None = None

    async def observe(self, transcript: TranscriptEvent) -> None:
        is_partial = transcript.transcript_type == "partial"
        is_final = transcript.transcript_type in {"final", "human_final"}
        if not is_partial and not is_final:
            return
        text = transcript.text.strip()
        if not text:
            return
        self._latest_text = text
        if self._task is not None:
            return
        if is_partial and len(text) < self.min_chars:
            return
        self._task = asyncio.create_task(self._collect_and_log(text))

    async def decisions(self) -> tuple[list[TimingDecision], str]:
        if self._task is None:
            return [], ""
        return await self._task, self._collected_text

    async def _collect_and_log(self, text: str) -> list[TimingDecision]:
        self._collected_text = text
        decisions = await self.collect(text)
        await self.on_collected(text, decisions)
        return decisions


@dataclass
class TurnController:
    config: RuntimeConfig
    agents: dict[str, AgentLike | StreamingAgentLike]
    tts: StreamingTTSLike = field(default_factory=FakeStreamingTTS)
    stt: StreamingSTTLike | QueueStreamingSTTLike = field(default_factory=FakeStreamingSTT)
    _speaker_selector: FixedSpeakerSequence | None = field(
        init=False,
        default=None,
        repr=False,
    )
    _speaker_overrides: dict[int, str] = field(
        init=False,
        default_factory=dict,
        repr=False,
    )

    def __post_init__(self) -> None:
        try:
            speaker_selector = FixedSpeakerSequence(
                participant_roles=self.config.participants,
                sequence=self.config.fixed_speaker_sequence,
            )
        except SpeakerSelectionError:
            speaker_selector = None
        self._speaker_selector = speaker_selector
        missing = set()
        if speaker_selector:
            ai_speakers = {
                speaker_id
                for speaker_id, participant in self.config.participants.items()
                if participant.actor_kind is ActorKind.AI
            }
            missing = (set(speaker_selector.sequence) & ai_speakers) - set(self.agents)
        if missing:
            raise ValueError(f"missing agents: {', '.join(sorted(missing))}")

    def speaker_for_turn(self, turn_id: int) -> str:
        if turn_id in self._speaker_overrides:
            return self._speaker_overrides[turn_id]
        selector = self._speaker_selector
        if selector is None:
            selector = FixedSpeakerSequence(
                participant_roles=self.config.participants,
                sequence=self.config.fixed_speaker_sequence,
            )
            self._speaker_selector = selector
        return selector.speaker_for_turn(turn_id)

    def set_speaker_for_turn(self, turn_id: int, speaker: str) -> None:
        if turn_id <= 0:
            raise SpeakerSelectionError("turn_id must be positive")
        if speaker not in self.config.participants:
            raise SpeakerSelectionError(f"unknown speaker: {speaker}")
        participant = self.config.participants[speaker]
        if participant.actor_kind is ActorKind.AI and speaker not in self.agents:
            raise ValueError(f"missing agent: {speaker}")
        self._speaker_overrides[turn_id] = speaker

    def _stt_listener_id_for_turn(self, turn_id: int) -> str:
        return self.speaker_for_turn(turn_id + 1)

    def recipient_ids_for_speaker(self, speaker: str) -> tuple[str, ...]:
        return tuple(
            participant_id
            for participant_id in self.config.participants
            if participant_id != speaker
        )

    async def run_turn(
        self,
        *,
        turn_id: int,
        input_transcript: str,
        audio_bus: AudioBus,
        speaker: str | None = None,
        logger: AsyncLogger | None = None,
        provisional_generation: ProvisionalGeneration | None = None,
        stt_transcript_observer: TranscriptObserver | None = None,
        generated_transcript_observer: TranscriptObserver | None = None,
        additional_instruction: str | None = None,
        generation_interrupt_requested: Callable[[], bool] | None = None,
        publish_audio_without_transcript: bool = True,
        suppress_audio_without_transcript: Callable[[], bool] | None = None,
    ) -> TurnRuntimeState:
        if speaker is None:
            speaker = self.speaker_for_turn(turn_id)
        elif speaker not in self.config.participants:
            raise SpeakerSelectionError(f"unknown speaker: {speaker}")
        if provisional_generation is not None and (
            provisional_generation.turn_id != turn_id
            or provisional_generation.speaker != speaker
            or provisional_generation.session_id != self.config.session_id
        ):
            raise ValueError("provisional generation does not match the requested turn")
        state = TurnRuntimeState(
            session_id=self.config.session_id,
            turn_id=turn_id,
            speaker=speaker,
            input_transcript=input_transcript,
            recipient_ids=self.recipient_ids_for_speaker(speaker),
        )
        speaker_config = self.config.participants[speaker]
        generation_source = "provisional" if provisional_generation is not None else "final"
        if provisional_generation is not None:
            state.provisional_input_transcript = provisional_generation.input_transcript
            state.used_provisional_generation = True
            llm_request_started_at = provisional_generation.started_monotonic
        else:
            llm_request_started_at = time.monotonic()
        state.llm_request_started_monotonic = llm_request_started_at
        await _log_event(
            logger,
            state,
            "llm_request_started",
            {"generation_source": generation_source},
        )
        if provisional_generation is not None:
            await _log_event(
                logger,
                state,
                "provisional_llm_adopted",
                {
                    "source_turn_id": provisional_generation.source_turn_id,
                    "source": provisional_generation.source,
                    "prediction_reason": provisional_generation.prediction_reason,
                    "candidate_speaker_ids": list(
                        provisional_generation.candidate_speaker_ids
                    ),
                    "generation_modality": provisional_generation.generation_modality,
                    "provisional_input_char_count": len(provisional_generation.input_transcript),
                    "final_input_char_count": len(input_transcript),
                },
            )
        if _is_realtime_speech_agent(self.agents[speaker]):
            try:
                return await self._run_realtime_speech_turn(
                    turn_id=turn_id,
                    speaker=speaker,
                    input_transcript=input_transcript,
                    additional_instruction=additional_instruction,
                    audio_bus=audio_bus,
                    logger=logger,
                    state=state,
                    llm_request_started_at=llm_request_started_at,
                    stt_transcript_observer=stt_transcript_observer,
                    generated_transcript_observer=generated_transcript_observer,
                    provisional_generation=provisional_generation,
                    generation_interrupt_requested=generation_interrupt_requested,
                    publish_audio_without_transcript=publish_audio_without_transcript,
                    suppress_audio_without_transcript=suppress_audio_without_transcript,
                )
            except ProvisionalGenerationFailedError:
                if provisional_generation is None:
                    raise
                llm_request_started_at = await _prepare_final_realtime_fallback(
                    logger,
                    state,
                    provisional_generation,
                )
                return await self._run_realtime_speech_turn(
                    turn_id=turn_id,
                    speaker=speaker,
                    input_transcript=input_transcript,
                    additional_instruction=additional_instruction,
                    audio_bus=audio_bus,
                    logger=logger,
                    state=state,
                    llm_request_started_at=llm_request_started_at,
                    stt_transcript_observer=stt_transcript_observer,
                    generated_transcript_observer=generated_transcript_observer,
                    provisional_generation=None,
                    generation_interrupt_requested=generation_interrupt_requested,
                    publish_audio_without_transcript=publish_audio_without_transcript,
                    suppress_audio_without_transcript=suppress_audio_without_transcript,
                )
        text_chunker = TtsTextChunker(
            comma_min_chars=self.config.tts_chunk_comma_min_chars,
            soft_max_chars=self.config.tts_chunk_soft_max_chars,
        )
        generated_parts: list[str] = []
        audio_chunks: list[AudioChunk] = []
        first_token_at: float | None = None
        first_tts_request_at: float | None = None
        next_audio_chunk_index = 0
        _ = stt_transcript_observer

        async def synthesize_text_chunk(text_chunk_index: int, text_chunk: str) -> None:
            nonlocal first_tts_request_at
            nonlocal next_audio_chunk_index

            _raise_if_generation_interrupted(generation_interrupt_requested)
            if not state.tts_started:
                state.tts_started = True
            tts_request_started_at = time.monotonic()
            if first_tts_request_at is None:
                first_tts_request_at = tts_request_started_at
                if first_token_at is not None:
                    await _log_metric(
                        logger,
                        state,
                        "first_token_to_first_tts_request_ms",
                        _elapsed_ms(first_token_at, first_tts_request_at),
                        {"audio_delivery_mode": self.config.audio_delivery_mode.value},
                    )
            await _log_event(
                logger,
                state,
                "tts_request_started",
                {
                    "text_chunk_index": text_chunk_index,
                    "audio_delivery_mode": self.config.audio_delivery_mode.value,
                },
            )
            async for audio_chunk in self.tts.synthesize(
                session_id=self.config.session_id,
                turn_id=turn_id,
                speaker=speaker,
                text_chunks=[text_chunk],
                sample_rate=self.config.sample_rate,
                sample_width_bits=self.config.sample_width_bits,
                channels=self.config.channels,
                delivery_mode=self.config.audio_delivery_mode,
            ):
                _raise_if_generation_interrupted(generation_interrupt_requested)
                if audio_chunk.chunk_index != next_audio_chunk_index:
                    audio_chunk = replace(audio_chunk, chunk_index=next_audio_chunk_index)
                audio_chunk = _apply_speaker_audio_gain(
                    audio_chunk,
                    gain=self.config.speaker_audio_gains.get(speaker, 1.0),
                )
                if state.first_audio_chunk_at is None:
                    state.first_audio_chunk_at = audio_chunk.created_at
                    state.first_audio_chunk_monotonic = time.monotonic()
                    latency_ms = _elapsed_ms(
                        first_tts_request_at,
                        state.first_audio_chunk_monotonic,
                    )
                    await _log_metric(
                        logger,
                        state,
                        "tts_first_audio_chunk_latency_ms",
                        latency_ms,
                        {"audio_delivery_mode": self.config.audio_delivery_mode.value},
                    )
                    await _log_event(
                        logger,
                        state,
                        "tts_first_audio_chunk",
                        {
                            "latency_ms": latency_ms,
                            "audio_delivery_mode": self.config.audio_delivery_mode.value,
                        },
                    )
                audio_chunks.append(audio_chunk)
                await audio_bus.publish(audio_chunk)
                next_audio_chunk_index += 1

        tts_text_queue: asyncio.Queue[tuple[int, str] | object] = asyncio.Queue()

        async def enqueue_tts_text_chunk(
            text_chunk_index: int,
            text_chunk: str,
        ) -> None:
            await _log_event(
                logger,
                state,
                "llm_chunk_ready_for_tts",
                {"text_chunk_index": text_chunk_index, "char_count": len(text_chunk)},
            )
            await tts_text_queue.put((text_chunk_index, text_chunk))

        async def read_llm_text_chunks() -> None:
            nonlocal first_token_at

            text_chunk_index = 0
            try:
                text_iterator = _iter_generation_text(
                    self.agents[speaker],
                    input_transcript=input_transcript,
                    turn_id=turn_id,
                    provisional_generation=provisional_generation,
                    additional_instruction=additional_instruction,
                )
                text_iterator = strip_leading_speaker_label_from_text_parts(
                    text_iterator,
                    _speaker_label_candidates(self.config, speaker),
                )
                async for text_part in text_iterator:
                    _raise_if_generation_interrupted(generation_interrupt_requested)
                    if not text_part:
                        continue
                    if first_token_at is None:
                        first_token_at = (
                            provisional_generation.first_token_monotonic
                            if provisional_generation is not None
                            and provisional_generation.first_token_monotonic is not None
                            else time.monotonic()
                        )
                        latency_ms = _elapsed_ms(llm_request_started_at, first_token_at)
                        await _log_metric(
                            logger,
                            state,
                            "llm_first_token_latency_ms",
                            latency_ms,
                            {
                                "audio_delivery_mode": self.config.audio_delivery_mode.value,
                                "generation_source": generation_source,
                            },
                        )
                        await _log_event(
                            logger,
                            state,
                            "llm_first_token",
                            {
                                "latency_ms": latency_ms,
                                "generation_source": generation_source,
                            },
                        )
                    generated_parts.append(text_part)
                    state.generated_text = "".join(generated_parts)
                    await _publish_monitor_transcript(
                        audio_bus,
                        state,
                        transcript_type="delta",
                        text=text_part,
                        speaker_display_name=speaker_config.display_name,
                        speaker_role=speaker_config.role,
                    )
                    await _notify_transcript_observer(
                        generated_transcript_observer,
                        TranscriptEvent(
                            session_id=state.session_id,
                            turn_id=state.turn_id,
                            speaker=state.speaker,
                            speaker_id=state.speaker,
                            transcript_type="partial",
                            text=state.generated_text,
                            recipient_ids=state.recipient_ids,
                            metadata={
                                "speaker_display_name": speaker_config.display_name,
                                "role": speaker_config.role,
                                "text_source": "generated_text",
                            },
                        ),
                    )
                    for text_chunk in text_chunker.push(text_part):
                        await enqueue_tts_text_chunk(text_chunk_index, text_chunk)
                        text_chunk_index += 1

                for text_chunk in text_chunker.flush():
                    await enqueue_tts_text_chunk(text_chunk_index, text_chunk)
                    text_chunk_index += 1

                state.generated_text = "".join(generated_parts)
            finally:
                await tts_text_queue.put(_TTS_TEXT_QUEUE_DONE)

        async def synthesize_queued_text_chunks() -> None:
            while True:
                item = await tts_text_queue.get()
                try:
                    if item is _TTS_TEXT_QUEUE_DONE:
                        return
                    _raise_if_generation_interrupted(generation_interrupt_requested)
                    text_chunk_index, text_chunk = item
                    await synthesize_text_chunk(text_chunk_index, text_chunk)
                finally:
                    tts_text_queue.task_done()

        try:
            await _run_concurrently(
                read_llm_text_chunks,
                synthesize_queued_text_chunks,
            )

            state.tts_done = True
            tts_stream_done_at = time.monotonic()
            await _log_event(
                logger,
                state,
                "tts_stream_done",
                {"audio_delivery_mode": self.config.audio_delivery_mode.value},
            )
            await audio_bus.end_turn(self.config.session_id, turn_id, speaker)
            if logger is not None:
                state.audio_log_path = str(logger.paths.audio_dir / f"turn_{turn_id:04d}_{speaker}.wav")
        except GenerationInterruptedForHumanInput:
            await _finalize_interrupted_generated_turn(
                logger,
                state,
                audio_bus=audio_bus,
                speaker_display_name=speaker_config.display_name,
                speaker_role=speaker_config.role,
            )
            await _log_event(
                logger,
                state,
                "ai_generation_interrupted_for_human_barge_in",
                {"generated_char_count": len(state.generated_text)},
            )
            await audio_bus.end_turn(self.config.session_id, turn_id, speaker)
            raise
        except Exception:
            if (
                provisional_generation is not None
                and provisional_generation.task is not None
                and not provisional_generation.task.done()
            ):
                provisional_generation.task.cancel()
                await asyncio.gather(provisional_generation.task, return_exceptions=True)
            raise
        await _finalize_generated_turn(
            logger,
            state,
            audio_bus=audio_bus,
            speaker_display_name=self.config.participants[speaker].display_name,
            speaker_role=self.config.participants[speaker].role,
        )
        return state

    async def _run_realtime_speech_turn(
        self,
        *,
        turn_id: int,
        speaker: str,
        input_transcript: str,
        additional_instruction: str | None,
        audio_bus: AudioBus,
        logger: AsyncLogger | None,
        state: TurnRuntimeState,
        llm_request_started_at: float,
        stt_transcript_observer: TranscriptObserver | None,
        generated_transcript_observer: TranscriptObserver | None,
        format_input: bool = True,
        provisional_generation: ProvisionalGeneration | None = None,
        generation_interrupt_requested: Callable[[], bool] | None = None,
        publish_audio_without_transcript: bool = True,
        suppress_audio_without_transcript: Callable[[], bool] | None = None,
    ) -> TurnRuntimeState:
        realtime_agent = self.agents[speaker]
        speaker_config = self.config.participants[speaker]
        generated_parts: list[str] = []
        audio_chunks: list[AudioChunk] = []
        pending_audio_chunks: list[AudioChunk] = []
        first_token_at: float | None = None
        first_tts_request_at = llm_request_started_at
        next_audio_chunk_index = 0
        transcript_started = False
        speech_source = (
            "realtime_provisional"
            if provisional_generation is not None
            else "realtime_api"
        )
        generation_source = (
            "provisional"
            if provisional_generation is not None
            else "realtime_api"
        )
        state.tts_started = True
        _ = stt_transcript_observer
        await _log_event(
            logger,
            state,
            "tts_request_started",
            {
                "text_chunk_index": None,
                "audio_delivery_mode": self.config.audio_delivery_mode.value,
                "speech_source": speech_source,
            },
        )

        def prepare_realtime_audio_chunk(audio_chunk: AudioChunk) -> AudioChunk:
            nonlocal next_audio_chunk_index
            if audio_chunk.chunk_index != next_audio_chunk_index:
                audio_chunk = replace(audio_chunk, chunk_index=next_audio_chunk_index)
            audio_chunk = replace(audio_chunk, pace_after_delivery=False)
            audio_chunk = _apply_speaker_audio_gain(
                audio_chunk,
                gain=self.config.speaker_audio_gains.get(speaker, 1.0),
            )
            next_audio_chunk_index += 1
            return audio_chunk

        async def publish_realtime_audio_chunk(audio_chunk: AudioChunk) -> None:
            if state.first_audio_chunk_at is None:
                state.first_audio_chunk_at = audio_chunk.created_at
                state.first_audio_chunk_monotonic = time.monotonic()
                latency_ms = _elapsed_ms(
                    first_tts_request_at,
                    state.first_audio_chunk_monotonic,
                )
                await _log_metric(
                    logger,
                    state,
                    "tts_first_audio_chunk_latency_ms",
                    latency_ms,
                    {
                        "audio_delivery_mode": self.config.audio_delivery_mode.value,
                        "speech_source": speech_source,
                    },
                )
                await _log_event(
                    logger,
                    state,
                    "tts_first_audio_chunk",
                    {
                        "latency_ms": latency_ms,
                        "audio_delivery_mode": self.config.audio_delivery_mode.value,
                        "speech_source": speech_source,
                    },
                )
            audio_chunks.append(audio_chunk)
            await audio_bus.publish(audio_chunk)

        async def flush_pending_realtime_audio() -> None:
            if not pending_audio_chunks:
                return
            await _log_event(
                logger,
                state,
                "realtime_audio_buffer_flushed_after_text",
                {
                    "chunk_count": len(pending_audio_chunks),
                    "duration_ms": sum(
                        chunk.duration_ms for chunk in pending_audio_chunks
                    ),
                    "speech_source": speech_source,
                },
            )
            while pending_audio_chunks:
                await publish_realtime_audio_chunk(pending_audio_chunks.pop(0))

        try:
            if provisional_generation is None:
                realtime_events = _iter_realtime_speech_events(
                    realtime_agent,
                    session_id=self.config.session_id,
                    turn_id=turn_id,
                    speaker=speaker,
                    input_transcript=input_transcript,
                    additional_instruction=additional_instruction,
                    format_input=format_input,
                    sample_rate=self.config.sample_rate,
                    sample_width_bits=self.config.sample_width_bits,
                    channels=self.config.channels,
                    delivery_mode=self.config.audio_delivery_mode,
                    response_instructions_observer=(
                        _response_instructions_logger(logger)
                    ),
                )
            else:
                realtime_events = _iter_provisional_realtime_events(
                    provisional_generation
                )
            async for event in realtime_events:
                _raise_if_generation_interrupted(generation_interrupt_requested)
                text_delta = _realtime_speech_text_delta(event)
                if text_delta:
                    if first_token_at is None:
                        first_token_at = (
                            provisional_generation.first_token_monotonic
                            if provisional_generation is not None
                            and provisional_generation.first_token_monotonic is not None
                            else time.monotonic()
                        )
                        latency_ms = _elapsed_ms(llm_request_started_at, first_token_at)
                        await _log_metric(
                            logger,
                            state,
                            "llm_first_token_latency_ms",
                            latency_ms,
                            {
                                "audio_delivery_mode": self.config.audio_delivery_mode.value,
                                "generation_source": generation_source,
                            },
                        )
                        await _log_event(
                            logger,
                            state,
                            "llm_first_token",
                            {
                                "latency_ms": latency_ms,
                                "generation_source": generation_source,
                            },
                        )
                    generated_parts.append(text_delta)
                    state.generated_text = "".join(generated_parts)
                    await _publish_monitor_transcript(
                        audio_bus,
                        state,
                        transcript_type="delta",
                        text=text_delta,
                        speaker_display_name=speaker_config.display_name,
                        speaker_role=speaker_config.role,
                    )
                    await _notify_transcript_observer(
                        generated_transcript_observer,
                        TranscriptEvent(
                            session_id=state.session_id,
                            turn_id=state.turn_id,
                            speaker=state.speaker,
                            speaker_id=state.speaker,
                            transcript_type="partial",
                            text=state.generated_text,
                            recipient_ids=state.recipient_ids,
                            metadata={
                                "speaker_display_name": speaker_config.display_name,
                                "role": speaker_config.role,
                                "text_source": "generated_text",
                            },
                        ),
                    )
                    if not transcript_started and state.generated_text.strip():
                        transcript_started = True
                        await flush_pending_realtime_audio()

                audio_chunk = _realtime_speech_audio_chunk(event)
                if audio_chunk is None:
                    continue
                _raise_if_generation_interrupted(generation_interrupt_requested)
                audio_chunk = prepare_realtime_audio_chunk(audio_chunk)
                if transcript_started:
                    await publish_realtime_audio_chunk(audio_chunk)
                    continue
                pending_audio_chunks.append(audio_chunk)

            # A cancelled response can end before yielding any text/audio event.
            # Check exhaustion too, so early barge-in reaches the human input slot.
            _raise_if_generation_interrupted(generation_interrupt_requested)
            state.generated_text = "".join(generated_parts)
            state.tts_done = True
            tts_stream_done_at = time.monotonic()
            await _log_event(
                logger,
                state,
                "tts_stream_done",
                {
                    "audio_delivery_mode": self.config.audio_delivery_mode.value,
                    "speech_source": speech_source,
                },
            )
            if not state.generated_text.strip():
                suppress_audio_only = (
                    not publish_audio_without_transcript
                    or (
                        suppress_audio_without_transcript is not None
                        and suppress_audio_without_transcript()
                    )
                )
                if suppress_audio_only:
                    discarded_chunks = tuple(pending_audio_chunks)
                    pending_audio_chunks.clear()
                    await _log_event(
                        logger,
                        state,
                        "realtime_audio_discarded_without_transcript",
                        {
                            "chunk_count": len(discarded_chunks),
                            "duration_ms": sum(
                                chunk.duration_ms for chunk in discarded_chunks
                            ),
                            "speech_source": speech_source,
                            "reason": "post_barge_in_recovery",
                        },
                    )
                    state.audio_delivered_to_stt = True
                    state.stt_final_transcript = ""
                    state.stt_final_monotonic = time.monotonic()
                    state.mark_completed()
                    state.completed_monotonic = time.monotonic()
                    return state
                await _log_event(
                    logger,
                    state,
                    "realtime_audio_published_without_transcript",
                    {
                        "chunk_count": len(pending_audio_chunks),
                        "duration_ms": sum(
                            chunk.duration_ms for chunk in pending_audio_chunks
                        ),
                        "speech_source": speech_source,
                    },
                )
                while pending_audio_chunks:
                    await publish_realtime_audio_chunk(pending_audio_chunks.pop(0))
                await audio_bus.end_turn(self.config.session_id, turn_id, speaker)
                if logger is not None and audio_chunks:
                    state.audio_log_path = str(
                        logger.paths.audio_dir / f"turn_{turn_id:04d}_{speaker}.wav"
                    )
                state.audio_delivered_to_stt = True
                state.stt_final_transcript = ""
                state.stt_final_monotonic = time.monotonic()
                state.mark_completed()
                state.completed_monotonic = time.monotonic()
                return state
            await audio_bus.end_turn(self.config.session_id, turn_id, speaker)
            if logger is not None:
                state.audio_log_path = str(logger.paths.audio_dir / f"turn_{turn_id:04d}_{speaker}.wav")
        except GenerationInterruptedForHumanInput:
            await _finalize_interrupted_generated_turn(
                logger,
                state,
                audio_bus=audio_bus,
                speaker_display_name=speaker_config.display_name,
                speaker_role=speaker_config.role,
            )
            await _log_event(
                logger,
                state,
                "ai_generation_interrupted_for_human_barge_in",
                {"generated_char_count": len(state.generated_text)},
            )
            await audio_bus.end_turn(self.config.session_id, turn_id, speaker)
            raise
        except Exception:
            raise

        await _finalize_generated_turn(
            logger,
            state,
            audio_bus=audio_bus,
            speaker_display_name=self.config.participants[speaker].display_name,
            speaker_role=self.config.participants[speaker].role,
        )
        return state

    async def run_scripted_audio_turn(
        self,
        *,
        turn_id: int,
        speaker: str,
        text: str,
        audio_bus: AudioBus,
        logger: AsyncLogger | None = None,
        stt_transcript_observer: TranscriptObserver | None = None,
    ) -> TurnRuntimeState:
        state = TurnRuntimeState(
            session_id=self.config.session_id,
            turn_id=turn_id,
            speaker=speaker,
            input_transcript="",
            recipient_ids=self.recipient_ids_for_speaker(speaker),
            generated_text=text,
        )
        speaker_config = self.config.participants[speaker]
        request_started_at = time.monotonic()
        state.llm_request_started_monotonic = request_started_at
        await _log_event(
            logger,
            state,
            "scripted_audio_turn_started",
            {"char_count": len(text)},
        )
        if _is_realtime_speech_agent(self.agents[speaker]):
            return await self._run_realtime_speech_turn(
                turn_id=turn_id,
                speaker=speaker,
                input_transcript=text,
                additional_instruction=SCRIPTED_REALTIME_SPEECH_INSTRUCTION,
                audio_bus=audio_bus,
                logger=logger,
                state=state,
                llm_request_started_at=request_started_at,
                stt_transcript_observer=stt_transcript_observer,
                generated_transcript_observer=None,
                format_input=False,
            )
        state.tts_started = True
        await _publish_monitor_transcript(
            audio_bus,
            state,
            transcript_type="generated_final",
            text=text,
            speaker_display_name=speaker_config.display_name,
            speaker_role=speaker_config.role,
        )
        await _log_event(
            logger,
            state,
            "tts_request_started",
            {
                "text_chunk_index": 0,
                "audio_delivery_mode": self.config.audio_delivery_mode.value,
                "speech_source": "scripted_text",
            },
        )
        _ = stt_transcript_observer
        audio_chunks: list[AudioChunk] = []
        next_audio_chunk_index = 0
        tts_stream_done_at = request_started_at
        try:
            async for audio_chunk in self.tts.synthesize(
                session_id=self.config.session_id,
                turn_id=turn_id,
                speaker=speaker,
                text_chunks=[text],
                sample_rate=self.config.sample_rate,
                sample_width_bits=self.config.sample_width_bits,
                channels=self.config.channels,
                delivery_mode=self.config.audio_delivery_mode,
            ):
                if audio_chunk.chunk_index != next_audio_chunk_index:
                    audio_chunk = replace(
                        audio_chunk,
                        chunk_index=next_audio_chunk_index,
                    )
                audio_chunk = _apply_speaker_audio_gain(
                    audio_chunk,
                    gain=self.config.speaker_audio_gains.get(speaker, 1.0),
                )
                if state.first_audio_chunk_at is None:
                    state.first_audio_chunk_at = audio_chunk.created_at
                    state.first_audio_chunk_monotonic = time.monotonic()
                    latency_ms = _elapsed_ms(
                        request_started_at,
                        state.first_audio_chunk_monotonic,
                    )
                    await _log_metric(
                        logger,
                        state,
                        "tts_first_audio_chunk_latency_ms",
                        latency_ms,
                        {
                            "audio_delivery_mode": self.config.audio_delivery_mode.value,
                            "speech_source": "scripted_text",
                        },
                    )
                    await _log_event(
                        logger,
                        state,
                        "tts_first_audio_chunk",
                        {
                            "latency_ms": latency_ms,
                            "audio_delivery_mode": self.config.audio_delivery_mode.value,
                            "speech_source": "scripted_text",
                        },
                    )
                audio_chunks.append(audio_chunk)
                await audio_bus.publish(audio_chunk)
                next_audio_chunk_index += 1

            state.tts_done = True
            tts_stream_done_at = time.monotonic()
            await _log_event(
                logger,
                state,
                "tts_stream_done",
                {
                    "audio_delivery_mode": self.config.audio_delivery_mode.value,
                    "speech_source": "scripted_text",
                },
            )
            await audio_bus.end_turn(self.config.session_id, turn_id, speaker)
            if logger is not None:
                state.audio_log_path = str(
                    logger.paths.audio_dir / f"turn_{turn_id:04d}_{speaker}.wav"
                )
        except Exception:
            raise

        await _finalize_generated_turn(
            logger,
            state,
            audio_bus=audio_bus,
            speaker_display_name=self.config.participants[speaker].display_name,
            speaker_role=self.config.participants[speaker].role,
            publish_monitor_final=False,
        )
        return state

    async def run_realtime_audio_input_turn(
        self,
        *,
        turn_id: int,
        speaker: str,
        input_audio: bytes,
        source_human_turn_id: int,
        source_human_speaker: str,
        human_recipient_ids: tuple[str, ...],
        source_audio_ref: str | None,
        recording_mode: str,
        client_message_id: str | None,
        audio_bus: AudioBus,
        logger: AsyncLogger | None = None,
        generated_transcript_observer: TranscriptObserver | None = None,
        human_input_transcript_observer: TranscriptObserver | None = None,
        additional_instruction: str | None = None,
    ) -> TurnRuntimeState:
        agent = self.agents[speaker]
        if not _is_realtime_audio_input_agent(agent):
            raise ValueError(f"speaker does not support realtime audio input: {speaker}")
        speaker_config = self.config.participants[speaker]
        human_participant = self.config.participants[source_human_speaker]
        state = TurnRuntimeState(
            session_id=self.config.session_id,
            turn_id=turn_id,
            speaker=speaker,
            input_transcript="",
            recipient_ids=self.recipient_ids_for_speaker(speaker),
        )
        request_started_at = time.monotonic()
        state.llm_request_started_monotonic = request_started_at
        state.tts_started = True
        await _log_event(
            logger,
            state,
            "llm_request_started",
            {"generation_source": "realtime_human_audio"},
        )
        await _log_event(
            logger,
            state,
            "tts_request_started",
            {
                "text_chunk_index": None,
                "audio_delivery_mode": self.config.audio_delivery_mode.value,
                "speech_source": "realtime_human_audio",
            },
        )

        generated_parts: list[str] = []
        human_input_parts: list[str] = []
        first_token_at: float | None = None
        next_audio_chunk_index = 0

        async for event in _iter_realtime_audio_input_speech_events(
            agent,
            session_id=self.config.session_id,
            turn_id=turn_id,
            speaker=speaker,
            input_audio=input_audio,
            additional_instruction=additional_instruction,
            sample_rate=self.config.sample_rate,
            sample_width_bits=self.config.sample_width_bits,
            channels=self.config.channels,
            delivery_mode=self.config.audio_delivery_mode,
            response_instructions_observer=(
                _response_instructions_logger(logger)
            ),
        ):
            input_transcript_delta = _realtime_speech_input_transcript_delta(event)
            if input_transcript_delta:
                human_input_parts.append(input_transcript_delta)
                await _publish_monitor_transcript_event(
                    audio_bus,
                    logger,
                    TranscriptEvent(
                        session_id=self.config.session_id,
                        turn_id=source_human_turn_id,
                        speaker=source_human_speaker,
                        speaker_id=source_human_speaker,
                        transcript_type="partial",
                        text="".join(human_input_parts),
                        recipient_ids=human_recipient_ids,
                        metadata={
                            "speaker_display_name": human_participant.display_name,
                            "role": human_participant.role,
                            "text_source": "realtime_input_audio_transcription",
                            "recording_mode": recording_mode,
                            "source_audio_ref": source_audio_ref,
                            "client_message_id": client_message_id,
                        },
                    ),
                )
            input_transcript_completed = (
                _realtime_speech_input_transcript_completed(event)
            )
            if input_transcript_completed is not None:
                human_input_parts = [input_transcript_completed.strip()]
                transcript = TranscriptEvent(
                    session_id=self.config.session_id,
                    turn_id=source_human_turn_id,
                    speaker=source_human_speaker,
                    speaker_id=source_human_speaker,
                    transcript_type="human_final",
                    text=input_transcript_completed.strip(),
                    recipient_ids=human_recipient_ids,
                    metadata={
                        "speaker_display_name": human_participant.display_name,
                        "role": human_participant.role,
                        "text_source": "realtime_input_audio_transcription",
                        "recording_mode": recording_mode,
                        "source_audio_ref": source_audio_ref,
                        "client_message_id": client_message_id,
                    },
                )
                await _publish_monitor_transcript_event(audio_bus, logger, transcript)
                await _notify_transcript_observer(
                    human_input_transcript_observer,
                    transcript,
                )

            text_delta = _realtime_speech_text_delta(event)
            if text_delta:
                if first_token_at is None:
                    first_token_at = time.monotonic()
                    latency_ms = _elapsed_ms(request_started_at, first_token_at)
                    await _log_metric(
                        logger,
                        state,
                        "llm_first_token_latency_ms",
                        latency_ms,
                        {
                            "audio_delivery_mode": self.config.audio_delivery_mode.value,
                            "generation_source": "realtime_human_audio",
                        },
                    )
                    await _log_event(
                        logger,
                        state,
                        "llm_first_token",
                        {
                            "latency_ms": latency_ms,
                            "generation_source": "realtime_human_audio",
                        },
                    )
                generated_parts.append(text_delta)
                state.generated_text = "".join(generated_parts)
                await _publish_monitor_transcript(
                    audio_bus,
                    state,
                    transcript_type="delta",
                    text=text_delta,
                    speaker_display_name=speaker_config.display_name,
                    speaker_role=speaker_config.role,
                )
                await _notify_transcript_observer(
                    generated_transcript_observer,
                    TranscriptEvent(
                        session_id=state.session_id,
                        turn_id=state.turn_id,
                        speaker=state.speaker,
                        speaker_id=state.speaker,
                        transcript_type="partial",
                        text=state.generated_text,
                        recipient_ids=state.recipient_ids,
                        metadata={
                            "speaker_display_name": speaker_config.display_name,
                            "role": speaker_config.role,
                            "text_source": "generated_text",
                        },
                    ),
                )

            audio_chunk = _realtime_speech_audio_chunk(event)
            if audio_chunk is None:
                continue
            if audio_chunk.chunk_index != next_audio_chunk_index:
                audio_chunk = replace(audio_chunk, chunk_index=next_audio_chunk_index)
            audio_chunk = replace(audio_chunk, pace_after_delivery=False)
            audio_chunk = _apply_speaker_audio_gain(
                audio_chunk,
                gain=self.config.speaker_audio_gains.get(speaker, 1.0),
            )
            if state.first_audio_chunk_at is None:
                state.first_audio_chunk_at = audio_chunk.created_at
                state.first_audio_chunk_monotonic = time.monotonic()
                latency_ms = _elapsed_ms(
                    request_started_at,
                    state.first_audio_chunk_monotonic,
                )
                await _log_metric(
                    logger,
                    state,
                    "tts_first_audio_chunk_latency_ms",
                    latency_ms,
                    {
                        "audio_delivery_mode": self.config.audio_delivery_mode.value,
                        "speech_source": "realtime_human_audio",
                    },
                )
                await _log_event(
                    logger,
                    state,
                    "tts_first_audio_chunk",
                    {
                        "latency_ms": latency_ms,
                        "audio_delivery_mode": self.config.audio_delivery_mode.value,
                        "speech_source": "realtime_human_audio",
                    },
                )
            await audio_bus.publish(audio_chunk)
            next_audio_chunk_index += 1

        state.generated_text = "".join(generated_parts)
        state.tts_done = True
        await _log_event(
            logger,
            state,
            "tts_stream_done",
            {
                "audio_delivery_mode": self.config.audio_delivery_mode.value,
                "speech_source": "realtime_human_audio",
            },
        )
        await audio_bus.end_turn(self.config.session_id, turn_id, speaker)
        if logger is not None:
            state.audio_log_path = str(
                logger.paths.audio_dir / f"turn_{turn_id:04d}_{speaker}.wav"
            )
        await _finalize_generated_turn(
            logger,
            state,
            audio_bus=audio_bus,
            speaker_display_name=speaker_config.display_name,
            speaker_role=speaker_config.role,
        )
        return state

def _initial_client_turn(config: RuntimeConfig) -> tuple[str, str] | None:
    initial_participant = _initial_client_participant(config)
    if initial_participant is None:
        return None
    initial_transcript = (
        initial_participant.initial_transcript.strip()
        or config.initial_client_transcript.strip()
    )
    if not initial_transcript:
        return None
    return initial_participant.speaker_id, initial_transcript


def _initial_client_participant(config: RuntimeConfig) -> ParticipantConfig | None:
    client_participants: list[ParticipantConfig] = []
    seen_speaker_ids: set[str] = set()
    for speaker_id in config.fixed_speaker_sequence:
        participant = config.participants.get(speaker_id)
        if participant is not None and participant.role == CLIENT_ROLE:
            client_participants.append(participant)
            seen_speaker_ids.add(participant.speaker_id)

    legacy_client = config.participants.get("client")
    if (
        legacy_client is not None
        and legacy_client.role == CLIENT_ROLE
        and legacy_client.speaker_id not in seen_speaker_ids
    ):
        client_participants.append(legacy_client)
        seen_speaker_ids.add(legacy_client.speaker_id)

    for participant in config.participants.values():
        if (
            participant.role == CLIENT_ROLE
            and participant.speaker_id not in seen_speaker_ids
        ):
            client_participants.append(participant)
            seen_speaker_ids.add(participant.speaker_id)

    participants_with_initial_transcript = [
        participant
        for participant in client_participants
        if participant.initial_transcript.strip()
    ]
    if len(participants_with_initial_transcript) == 1:
        return participants_with_initial_transcript[0]
    if participants_with_initial_transcript:
        return random.choice(participants_with_initial_transcript)
    return client_participants[0] if client_participants else None


AI_COUNSELOR_INITIAL_CLIENT_OPENING_INSTRUCTION: Final[str] = (
    "これはAIカウンセラー x AIクライアントセッションの冒頭で、"
    "AIクライアントが最初に話し始めるターンです。"
    "初回相談内容はあなたの内部メモであり、発話本文ではありません。"
    "短く挨拶してから初回相談内容に触れてください。"
    "初回相談内容の本文を固定文として丸読みせず、"
    "原文と同じ一文をそのまま出力せず、"
    "あなたの立場に合わせて自然に言い換えてください。"
    "まだカウンセラーから詳しく尋ねられていない段階なので、"
    "詳細説明に入りすぎず、相談テーマの入口を短く伝えてください。"
    "\n\n初回相談内容:\n{initial_transcript}"
)


HUMAN_COUNSELOR_INITIAL_CLIENT_RESPONSE_INSTRUCTION: Final[str] = (
    "これは人間カウンセラーの最初の発話に対する、"
    "AIクライアントの最初の応答です。"
    "初回相談内容はあなたの内部メモであり、発話本文ではありません。"
    "カウンセラーの挨拶や問いかけに自然に応答し、"
    "短く挨拶してから初回相談内容に触れてください。"
    "初回相談内容の本文を固定文として丸読みせず、"
    "原文と同じ一文をそのまま出力せず、"
    "あなたの立場と直前のカウンセラー発話に合わせて自然に言い換えてください。"
    "まだ詳細を尋ねられていない場合は、詳しい説明に入りすぎず、"
    "相談テーマの入口を短く伝えてください。"
    "\n\n初回相談内容:\n{initial_transcript}"
)


AI_COUNSELOR_HUMAN_CLIENT_OPENING_INSTRUCTION: Final[str] = (
    "対話相手は人間クライアントです。人間クライアントはまだ発話していません。"
    "設定されたカウンセラーとして短く挨拶し、今話したいことや相談したいことを"
    "一つの開かれた問いかけで尋ね、返答を待ってください。"
    "このターンは相談の入り口です。カウンセラープロンプトにキー質問があっても、このターンでは使わず、"
    "面接の成果や理想の未来像、例外、点数は尋ねないでください。"
    "人間の相談内容や返答を作らず、事前共有情報は実際の発話と区別してください。"
)


EMPTY_BARGE_IN_RESUME_DELAY_SECONDS: Final[float] = 3.0
EMPTY_BARGE_IN_CONTINUATION_INSTRUCTION: Final[str] = (
    "一時的な音で前の発話が中断されましたが、人間の発話は確認されませんでした。"
    "新しい人間の発話を想定せず、直前に伝えようとしていた内容を短く続けてください。"
    "中断理由やシステム操作には言及せず、すでに伝えた長い説明は繰り返さないでください。"
)


class ResumeAfterEmptyBargeIn(Exception):
    """Resume the interrupted AI without committing a synthetic human turn."""

    def __init__(self, turn_id: int) -> None:
        self.turn_id = turn_id
        super().__init__(turn_id)


class HumanInputWaitExpired(Exception):
    """The configured session deadline expired while waiting for human input."""


class ConversationRuntime:

    def __init__(
        self,
        *,
        config: RuntimeConfig | None = None,
        agents: dict[str, AgentLike | StreamingAgentLike] | None = None,
        tts: StreamingTTSLike | None = None,
        stt: StreamingSTTLike | QueueStreamingSTTLike | None = None,
        session_summary_llm: StreamingLLMLike | None = None,
        prompt_director: PromptDirectorLike | None = None,
        prompt_context_store: RuntimePromptContextStore | None = None,
        sessions_dir: Path | str | None = None,
        ai_route_manifest: dict[str, dict[str, str]] | None = None,
    ) -> None:
        self.config = config or RuntimeConfig()
        self.controller = TurnController(
            self.config,
            agents or default_fake_agents(self.config.participants),
            tts=tts or FakeStreamingTTS(),
            stt=stt or FakeStreamingSTT(),
        )
        self.audio_bus = AudioBus(stt_delivery_enabled=False)
        self.logger = AsyncLogger(
            sessions_dir=sessions_dir or Path("results") / "runtime_sessions",
            session_id=self.config.session_id,
        )
        self.session_summary_llm = session_summary_llm
        self.prompt_director = prompt_director
        self.prompt_context_store = prompt_context_store
        self.ai_route_manifest = ai_route_manifest or {}
        self.turns: list[TurnRuntimeState] = []
        self._phase = RuntimePhase.IDLE
        self._last_event_type: str | None = None
        self._closing_started_turn_id: int | None = None
        self._closing_count_started_turn_id: int | None = None
        self._closing_assessed_turn_id: int | None = None
        self._closing_reply_speakers: tuple[str, ...] = ()
        self._closing_reply_counselor: str | None = None
        self._session_started_monotonic: float | None = None
        self._early_closing_reason: str | None = None
        self._session_summary_state = SessionSummaryState()
        self._session_summary_task: asyncio.Task[SessionSummaryUpdate] | None = None
        self._session_summary_task_request: SessionSummaryRequest | None = None
        self._session_summary_retry_request: SessionSummaryRequest | None = None
        self._previous_turn_target_hint = ""
        self._previous_turn_target_speaker: str | None = None
        self._generation_resume_event = asyncio.Event()
        self._generation_resume_event.set()
        self._human_turn_queue: asyncio.Queue[
            HumanTurnInput | HumanRealtimeAudioTurnInput | HumanStreamingAudioTurnInput
        ] = asyncio.Queue()
        self._pending_realtime_human_audio: dict[int, HumanRealtimeAudioTurnInput] = {}
        self._active_human_audio_streams: dict[str, HumanStreamingAudioTurnInput] = {}
        self._human_audio_streams: dict[str, HumanStreamingAudioTurnInput] = {}
        self._human_request_results: dict[str, tuple[Any, dict[str, Any]]] = {}
        self._human_input_deadline: float | None = None
        self._human_audio_stream_completion_futures: dict[
            str, asyncio.Future[dict[str, Any]]
        ] = {}
        self._human_input_turn_id: int | None = None
        self._human_input_speaker: str | None = None
        self._human_input_state: str | None = None
        self._active_turn_id: int | None = None
        self._active_speaker_id: str | None = None
        self._barge_in_human_turn_id: int | None = None
        self._barge_in_human_speaker: str | None = None
        self._interrupted_ai_turn_id: int | None = None
        self._empty_barge_in_resume_deadline: float | None = None
        self._resume_ai_turn_id: int | None = None
        self._running_ai_turn_id: int | None = None
        self._running_ai_speaker: str | None = None
        self._post_barge_in_recovery_turns_remaining = 0
        self._prompt_director_tasks: dict[
            PromptDirectorRequest,
            asyncio.Task[PromptDirectorResult],
        ] = {}
        self._prompt_director_prefetch_tasks: set[asyncio.Task[str | None]] = set()
        self._prompt_director_pause_reason: str | None = None
        self._selected_prompt_director_texts: dict[tuple[int, str], str] = {}

    @property
    def status(self) -> RuntimeStatus:
        current = self.turns[-1] if self.turns else None
        awaiting_human_input = (
            self._phase is RuntimePhase.RUNNING
            and self._human_input_turn_id is not None
            and self._human_input_speaker is not None
        )
        return RuntimeStatus(
            session_id=self.config.session_id,
            phase=(
                RuntimePhase.PAUSED
                if self._phase is RuntimePhase.RUNNING
                and self._prompt_director_pause_reason
                else self._phase
            ),
            pause_reason=(
                self._prompt_director_pause_reason
                if self._phase is RuntimePhase.RUNNING
                else None
            ),
            current_turn_id=(
                self._active_turn_id
                if self._active_turn_id is not None
                else current.turn_id if current else None
            ),
            current_speaker=(
                self._active_speaker_id
                if self._active_speaker_id is not None
                else current.speaker if current else None
            ),
            completed_turns=len(self.turns),
            last_event_type=self._last_event_type,
            closing_started=self._closing_started_turn_id is not None,
            closing_started_turn_id=self._closing_started_turn_id,
            closing_count_started_turn_id=self._closing_count_started_turn_id,
            awaiting_human_input=awaiting_human_input,
            active_speaker_id=self._active_speaker_id,
            pending_human_turn_allowed=awaiting_human_input,
            human_input_state=self._human_input_state,
            interaction_mode=self.config.interaction_mode.value,
            participants=self.config.public_participants(),
            human_speaker_id=self.config.human_speaker_id,
            human_recipient_ids=(
                self.controller.recipient_ids_for_speaker(self.config.human_speaker_id)
                if self.config.human_speaker_id
                else ()
            ),
        )

    def pause_generation(self) -> None:
        self._generation_resume_event.clear()
        self._empty_barge_in_resume_deadline = None
        self._interrupted_ai_turn_id = None
        self._resume_ai_turn_id = None
        if self.config.interaction_mode is InteractionMode.AI_COUNSELOR_HUMAN_CLIENT:
            for stream in self._human_audio_streams.values():
                if not stream.committed.is_set():
                    stream.aborted.set()

    def resume_generation(self) -> None:
        self._prompt_director_pause_reason = None
        self._generation_resume_event.set()

    def _barge_in_generation_interrupt_requested(self) -> bool:
        return (
            self._barge_in_human_turn_id is not None
            and self._barge_in_human_speaker is not None
            and self._running_ai_turn_id is not None
        )

    async def submit_human_turn(self, request: HumanTurnInput) -> dict[str, Any]:
        previous = self.human_input_result(request)
        if previous is not None:
            return previous
        self._validate_human_recipients(request.recipient_ids)
        self._ensure_can_accept_human_input(request.session_id)
        assert self._human_input_turn_id is not None
        assert self._human_input_speaker is not None
        turn_id = self._human_input_turn_id
        speaker = self._human_input_speaker
        self._human_input_state = "human_turn_received"
        self._empty_barge_in_resume_deadline = None
        self._human_input_turn_id = None
        self._human_input_speaker = None
        await self._human_turn_queue.put(request)
        return self._remember_human_input(
            request,
            {
                "accepted": True,
                "turn_id": turn_id,
                "speaker": speaker,
                "awaiting_human_input": False,
            },
        )

    async def submit_human_audio(self, request: HumanAudioInput) -> dict[str, Any]:
        if self.config.interaction_mode is InteractionMode.AI_COUNSELOR_HUMAN_CLIENT:
            previous = self.human_input_result(request)
            if previous is not None:
                return previous
            result = await self.start_human_audio_stream(
                HumanAudioStreamStart(
                    session_id=request.session_id,
                    sample_rate=request.sample_rate,
                    channels=request.channels,
                    recording_mode=request.recording_mode,
                    recipient_ids=request.recipient_ids,
                    client_message_id=request.client_message_id,
                )
            )
            await self.append_human_audio_stream_chunk(
                stream_id=result["stream_id"],
                audio_bytes=request.audio_bytes,
                chunk_index=0,
            )
            await self.end_human_audio_stream(stream_id=result["stream_id"])
            return self._remember_human_input(
                request,
                {
                    **result,
                    "transcript": "",
                    "pending_transcription": True,
                },
            )
        self._validate_human_recipients(request.recipient_ids)
        self._ensure_can_accept_human_input(request.session_id)
        assert self._human_input_turn_id is not None
        assert self._human_input_speaker is not None
        turn_id = self._human_input_turn_id
        speaker = self._human_input_speaker
        chunk = _human_audio_input_to_chunk(
            request,
            turn_id=turn_id,
            speaker=speaker,
            sample_width_bits=self.config.sample_width_bits,
            delivery_mode=self.config.audio_delivery_mode,
        )
        await self.audio_bus.publish_logger_audio(chunk)
        await self.audio_bus.end_logger_turn(self.config.session_id, turn_id, speaker)
        direct_target_speaker = self._direct_realtime_human_audio_target(request)
        if direct_target_speaker is not None:
            source_audio_ref = _turn_audio_log_ref(turn_id, speaker)
            self._human_input_state = "human_realtime_audio_received"
            self._human_input_turn_id = None
            self._human_input_speaker = None
            await self._human_turn_queue.put(
                HumanRealtimeAudioTurnInput(
                    request=request,
                    turn_id=turn_id,
                    speaker=speaker,
                    target_speaker=direct_target_speaker,
                    audio_chunk=chunk,
                    source_audio_ref=source_audio_ref,
                )
            )
            return {
                "accepted": True,
                "turn_id": turn_id,
                "speaker": speaker,
                "transcript": "",
                "source_audio_ref": source_audio_ref,
                "input_mode": "realtime_audio",
                "target_speaker": direct_target_speaker,
                "awaiting_human_input": False,
            }

        self._human_input_state = "transcribing_human_audio"
        partials, final = await self.controller.stt.transcribe(
            session_id=self.config.session_id,
            turn_id=turn_id,
            speaker=speaker,
            chunks=[chunk],
        )
        final_text = _normalized_human_stt_final_text(final.text)
        speaker_config = self.config.participants[speaker]
        if not final_text:
            self._human_input_state = "waiting_for_human_speech"
            await _log_event(
                self.logger,
                TurnRuntimeState(
                    session_id=self.config.session_id,
                    turn_id=turn_id,
                    speaker=speaker,
                    input_transcript="",
                    recipient_ids=request.recipient_ids,
                ),
                "human_stt_blank",
                {
                    "input_mode": "human_mic_stt",
                    "recording_mode": request.recording_mode,
                    "source_audio_ref": _turn_audio_log_ref(turn_id, speaker),
                    "client_message_id": request.client_message_id,
                },
            )
            return {
                "accepted": False,
                "turn_id": turn_id,
                "speaker": speaker,
                "transcript": "",
                "source_audio_ref": _turn_audio_log_ref(turn_id, speaker),
                "awaiting_human_input": True,
                "reason": "blank_transcript",
            }
        for transcript in partials:
            await _publish_monitor_transcript_event(
                self.audio_bus,
                self.logger,
                _with_transcript_display_metadata(
                    transcript,
                    participant=speaker_config,
                    recipient_ids=request.recipient_ids,
                    text_source="human_stt",
                    source_audio_ref=_turn_audio_log_ref(turn_id, speaker),
                ),
            )
        turn_request = HumanTurnInput(
            session_id=request.session_id,
            text=final_text,
            recipient_ids=request.recipient_ids,
            input_mode="human_mic_stt",
            recording_mode=request.recording_mode,
            source_audio_ref=_turn_audio_log_ref(turn_id, speaker),
            client_message_id=request.client_message_id,
        )
        self._human_input_state = "human_turn_received"
        await self._human_turn_queue.put(turn_request)
        return {
            "accepted": True,
            "turn_id": turn_id,
            "speaker": speaker,
            "transcript": final_text,
            "source_audio_ref": turn_request.source_audio_ref,
            "awaiting_human_input": False,
        }

    async def start_human_audio_stream(
        self,
        request: HumanAudioStreamStart,
    ) -> dict[str, Any]:
        previous = self.human_input_result(request)
        if previous is not None:
            return previous
        self._validate_human_recipients(request.recipient_ids)
        self._ensure_can_accept_human_input(request.session_id)
        assert self._human_input_turn_id is not None
        assert self._human_input_speaker is not None
        turn_id = self._human_input_turn_id
        speaker = self._human_input_speaker
        stream_id = (
            request.client_message_id
            or f"human-audio-stream:{self.config.session_id}:{uuid4().hex}"
        )
        if stream_id in self._active_human_audio_streams:
            raise RuntimeError("human audio stream is already active")
        completion: asyncio.Future[dict[str, Any]] = (
            asyncio.get_running_loop().create_future()
        )
        stream_input = HumanStreamingAudioTurnInput(
            request=request,
            stream_id=stream_id,
            turn_id=turn_id,
            speaker=speaker,
            audio_queue=asyncio.Queue(),
            source_audio_ref=_turn_audio_log_ref(turn_id, speaker),
            completion=completion,
        )
        self._active_human_audio_streams[stream_id] = stream_input
        self._human_audio_streams[stream_id] = stream_input
        self._human_audio_stream_completion_futures[stream_id] = completion
        self._human_input_state = "streaming_human_audio"
        self._empty_barge_in_resume_deadline = None
        self._human_input_turn_id = None
        self._human_input_speaker = None
        await self._human_turn_queue.put(stream_input)
        return self._remember_human_input(
            request,
            {
                "accepted": True,
                "stream_id": stream_id,
                "turn_id": turn_id,
                "speaker": speaker,
                "source_audio_ref": stream_input.source_audio_ref,
                "awaiting_human_input": False,
            },
        )

    async def append_human_audio_stream_chunk(
        self,
        *,
        stream_id: str,
        audio_bytes: bytes,
        chunk_index: int,
    ) -> dict[str, Any]:
        stream_input = self._active_human_audio_streams.get(stream_id)
        if stream_input is None or stream_input.aborted.is_set():
            raise RuntimeError("human audio stream is not active")
        chunk = _human_audio_stream_chunk_to_audio_chunk(
            stream_input.request,
            audio_bytes=audio_bytes,
            chunk_index=chunk_index,
            turn_id=stream_input.turn_id,
            speaker=stream_input.speaker,
            sample_width_bits=self.config.sample_width_bits,
            delivery_mode=self.config.audio_delivery_mode,
        )
        digest = hashlib.sha256(audio_bytes).digest()
        previous_digest = stream_input.chunk_digests.get(chunk_index)
        if previous_digest is not None and previous_digest != digest:
            raise RuntimeError("chunk_index was already used for different audio")
        if previous_digest is None:
            stream_input.chunk_digests[chunk_index] = digest
            await self.audio_bus.publish_logger_audio(chunk)
            await stream_input.audio_queue.put(chunk)
        return {
            "accepted": True,
            "stream_id": stream_id,
            "turn_id": stream_input.turn_id,
            "speaker": stream_input.speaker,
            "chunk_index": chunk_index,
            "duration_ms": chunk.duration_ms,
        }

    async def end_human_audio_stream(self, *, stream_id: str) -> dict[str, Any]:
        stream_input = self._active_human_audio_streams.get(stream_id)
        if (
            stream_input is None
            and self.config.interaction_mode
            is InteractionMode.AI_COUNSELOR_HUMAN_CLIENT
        ):
            stream_input = self._human_audio_streams.get(stream_id)
            if stream_input is not None:
                if stream_input.completion.done():
                    return dict(stream_input.completion.result())
                return await self._complete_human_audio_stream(
                    stream_input,
                    completion_reason="client_end",
                    require_active=False,
                )
        if stream_input is None:
            raise RuntimeError("human audio stream is not active")
        result = await self._complete_human_audio_stream(
            stream_input,
            completion_reason="client_end",
            require_active=True,
        )
        self._human_input_state = "human_audio_stream_committing"
        return result

    async def wait_human_audio_stream_completion(
        self,
        *,
        stream_id: str,
    ) -> dict[str, Any]:
        completion = self._human_audio_stream_completion_futures.get(stream_id)
        if completion is None and stream_id in self._human_audio_streams:
            completion = self._human_audio_streams[stream_id].completion
        if completion is None:
            raise RuntimeError("human audio stream is not active")
        result = await asyncio.shield(completion)
        self._human_audio_stream_completion_futures.pop(stream_id, None)
        return result

    async def abort_human_audio_stream(self, *, stream_id: str) -> dict[str, Any]:
        stream = self._human_audio_streams.get(stream_id)
        if stream is None:
            raise RuntimeError("human audio stream is not active")
        if stream.committed.is_set():
            return {
                "accepted": True,
                "stream_id": stream_id,
                "turn_id": stream.turn_id,
                "speaker": stream.speaker,
                "completion_reason": "already_committed",
            }
        stream.aborted.set()
        result = await self._complete_human_audio_stream(
            stream,
            completion_reason="aborted",
            require_active=False,
        )
        if not stream.abort_logged.is_set():
            stream.abort_logged.set()
            await self.logger.log_event(
                RuntimeEvent(
                    session_id=self.config.session_id,
                    event_type="human_audio_stream_aborted",
                    turn_id=stream.turn_id,
                    speaker=stream.speaker,
                    details={
                        "stream_id": stream_id,
                        "client_message_id": stream.request.client_message_id,
                    },
                )
            )
        return result

    async def abort_pending_human_audio_streams(self) -> None:
        pending = [
            stream
            for stream in self._human_audio_streams.values()
            if not stream.committed.is_set()
        ]
        for stream in pending:
            stream.aborted.set()
        for stream in pending:
            await self.abort_human_audio_stream(stream_id=stream.stream_id)

    def human_input_result(self, request: Any) -> dict[str, Any] | None:
        if (
            not request.client_message_id
            or request.session_id != self.config.session_id
        ):
            return None
        previous = self._human_request_results.get(request.client_message_id)
        if previous is None:
            return None
        previous_request, result = previous
        if previous_request != request:
            raise RuntimeError(
                "client_message_id was already used for different human input"
            )
        return dict(result)

    def _remember_human_input(
        self, request: Any, result: dict[str, Any]
    ) -> dict[str, Any]:
        if request.client_message_id:
            self._human_request_results[request.client_message_id] = (
                request,
                dict(result),
            )
        return result

    async def _next_human_request(self, state: TurnRuntimeState) -> Any:
        while True:
            deadline = self._empty_barge_in_resume_deadline
            operation = self._human_turn_queue.get()
            if deadline is None:
                return await self._wait_for_human_operation(operation)
            try:
                return await self._wait_for_human_operation(
                    asyncio.wait_for(
                        operation, timeout=max(0.0, deadline - time.monotonic()),
                    )
                )
            except TimeoutError:
                # A new input, Pause, or Stop invalidates this recovery window.
                if (
                    self._empty_barge_in_resume_deadline != deadline
                    or not self._generation_resume_event.is_set()
                ):
                    continue
                interrupted_turn_id = self._interrupted_ai_turn_id
                self._empty_barge_in_resume_deadline = None
                self._interrupted_ai_turn_id = None
                self._barge_in_human_turn_id = None
                self._barge_in_human_speaker = None
                self._resume_ai_turn_id = state.turn_id
                self._human_input_state = "resuming_after_empty_barge_in"
                await _log_event(
                    self.logger, state, "empty_barge_in_ai_resuming",
                    {
                        "interrupted_turn_id": interrupted_turn_id,
                        "resume_turn_id": state.turn_id,
                    },
                )
                raise ResumeAfterEmptyBargeIn(state.turn_id)

    async def _wait_for_human_operation(self, operation: Awaitable[Any]) -> Any:
        if self._human_input_deadline is None:
            return await operation
        remaining = max(0.0, self._human_input_deadline - time.monotonic())
        try:
            return await asyncio.wait_for(operation, timeout=remaining)
        except TimeoutError as exc:
            if time.monotonic() < self._human_input_deadline:
                raise
            raise HumanInputWaitExpired() from exc

    async def _transcribe_human_stream(
        self,
        stream: HumanStreamingAudioTurnInput,
        transcribe: Any,
        kwargs: dict[str, Any],
    ) -> Any:
        task = asyncio.create_task(transcribe(**kwargs))
        aborted = asyncio.create_task(stream.aborted.wait())
        try:
            await self._wait_for_human_operation(
                asyncio.wait({task, aborted}, return_when=asyncio.FIRST_COMPLETED)
            )
            if stream.aborted.is_set():
                await self.abort_human_audio_stream(stream_id=stream.stream_id)
                return None
            return task.result()
        finally:
            for pending in (task, aborted):
                if not pending.done():
                    pending.cancel()
            await asyncio.gather(task, aborted, return_exceptions=True)

    async def _complete_human_audio_stream(
        self,
        stream_input: HumanStreamingAudioTurnInput,
        *,
        completion_reason: str,
        require_active: bool,
        text: str | None = None,
    ) -> dict[str, Any]:
        active = self._active_human_audio_streams.pop(stream_input.stream_id, None)
        if active is None:
            if require_active:
                raise RuntimeError("human audio stream is not active")
        elif active is not stream_input:
            if require_active:
                raise RuntimeError("human audio stream is not active")
        else:
            end = EndOfAudio(
                session_id=self.config.session_id,
                turn_id=stream_input.turn_id,
                speaker=stream_input.speaker,
            )
            await stream_input.audio_queue.put(end)
            await self.audio_bus.end_logger_turn(
                self.config.session_id,
                stream_input.turn_id,
                stream_input.speaker,
            )
        result = {
            "accepted": True,
            "stream_id": stream_input.stream_id,
            "turn_id": stream_input.turn_id,
            "speaker": stream_input.speaker,
            "source_audio_ref": stream_input.source_audio_ref,
            "completion_reason": (
                "aborted" if stream_input.aborted.is_set() else completion_reason
            ),
        }
        if text is not None:
            result.update(accepted=bool(text), text=text)
        pending_transcription = (
            self.config.interaction_mode is InteractionMode.AI_COUNSELOR_HUMAN_CLIENT
            and not stream_input.aborted.is_set()
            and completion_reason in {"client_end", "server_vad"}
        )
        if pending_transcription:
            result["pending_transcription"] = True
        if (
            completion_reason == "blank_transcript"
            and self._empty_barge_in_resume_deadline is not None
        ):
            result["resume_pending"] = True
            result["resume_after_ms"] = round(EMPTY_BARGE_IN_RESUME_DELAY_SECONDS * 1000)
        if not pending_transcription and not stream_input.completion.done():
            stream_input.completion.set_result(result)
        return result

    def _direct_realtime_human_audio_target(
        self,
        request: HumanAudioInput,
    ) -> str | None:
        if self.config.interaction_mode is InteractionMode.AI_COUNSELOR_HUMAN_CLIENT:
            return None
        if request.sample_rate != self.config.sample_rate:
            return None
        if request.channels != self.config.channels:
            return None
        client_recipients = tuple(
            recipient_id
            for recipient_id in request.recipient_ids
            if (
                recipient_id in self.config.participants
                and self.config.participants[recipient_id].role == CLIENT_ROLE
            )
        )
        if len(client_recipients) != 1:
            return None
        target_speaker = client_recipients[0]
        agent = self.controller.agents.get(target_speaker)
        if agent is None or not _is_realtime_audio_input_agent(agent):
            return None
        return target_speaker

    def _validate_human_recipients(self, recipient_ids: tuple[str, ...]) -> None:
        if (
            self.config.interaction_mode is InteractionMode.AI_COUNSELOR_HUMAN_CLIENT
            and recipient_ids != ("counselor",)
        ):
            raise RuntimeError("human client recipient_ids must be [counselor]")

    def _ensure_can_accept_human_input(self, session_id: str) -> None:
        if session_id != self.config.session_id:
            raise RuntimeError("session_id does not match active runtime")
        if self._phase is not RuntimePhase.RUNNING:
            raise RuntimeError("runtime is not running")
        if (
            self.config.interaction_mode is InteractionMode.AI_COUNSELOR_HUMAN_CLIENT
            and not self._generation_resume_event.is_set()
        ):
            raise RuntimeError("runtime is paused")
        if self._human_input_turn_id is None or self._human_input_speaker is None:
            raise RuntimeError("runtime is not awaiting human input")

    def _speaker_is_human(self, speaker: str) -> bool:
        participant = self.config.participants.get(speaker)
        return participant is not None and participant.actor_kind is ActorKind.HUMAN

    def _speaker_is_ai(self, speaker: str) -> bool:
        participant = self.config.participants.get(speaker)
        return participant is not None and participant.actor_kind is ActorKind.AI

    def _ai_turn_is_empty(self, turn: TurnRuntimeState) -> bool:
        participant = self.config.participants.get(turn.speaker)
        return (
            participant is not None
            and participant.actor_kind is not ActorKind.HUMAN
            and not turn.generated_text.strip()
            and not (turn.stt_final_transcript or "").strip()
        )

    async def _log_empty_ai_turn_skipped_after_interrupt(
        self,
        *,
        turn: TurnRuntimeState,
        recovery_turns_remaining: int,
    ) -> None:
        await self.logger.log_event(
            RuntimeEvent(
                session_id=self.config.session_id,
                event_type="empty_ai_turn_skipped_after_interrupt",
                turn_id=turn.turn_id,
                speaker=turn.speaker,
                speaker_id=turn.speaker,
                recipient_ids=turn.recipient_ids,
                monotonic_time=time.monotonic(),
                details={
                    "reason": "post_barge_in_recovery",
                    "audio_log_path": turn.audio_log_path,
                    "recovery_turns_remaining": recovery_turns_remaining,
                },
            )
        )

    async def _log_barge_in_reserved_turn_advanced(
        self,
        *,
        from_turn_id: int,
        reserved_turn_id: int,
        speaker: str,
    ) -> None:
        await self.logger.log_event(
            RuntimeEvent(
                session_id=self.config.session_id,
                event_type="barge_in_reserved_turn_advanced",
                turn_id=reserved_turn_id,
                speaker=speaker,
                speaker_id=speaker,
                recipient_ids=self.controller.recipient_ids_for_speaker(speaker),
                monotonic_time=time.monotonic(),
                details={
                    "from_turn_id": from_turn_id,
                    "reserved_turn_id": reserved_turn_id,
                    "reason": "human_barge_in_reserved_future_turn",
                },
            )
        )

    def _open_human_input_slot_after_interrupt(
        self,
        *,
        interrupted_turn_id: int | None,
    ) -> dict[str, Any]:
        if not self.config.human_interrupts_enabled:
            return {}
        human_counselor = self.config.human_speaker_id
        if human_counselor is None:
            return {}
        active_human_stream = max(
            (
                stream
                for stream in self._active_human_audio_streams.values()
                if stream.speaker == human_counselor
            ),
            key=lambda stream: stream.turn_id,
            default=None,
        )
        if active_human_stream is not None:
            return {
                "awaiting_human_input": False,
                "human_audio_stream_active": True,
                "human_turn_id": active_human_stream.turn_id,
                "human_speaker": active_human_stream.speaker,
            }
        if (
            self._human_input_turn_id is not None
            and self._human_input_speaker is not None
        ):
            return {
                "awaiting_human_input": True,
                "human_turn_id": self._human_input_turn_id,
                "human_speaker": self._human_input_speaker,
            }
        base_turn_candidates = [0]
        if interrupted_turn_id is not None:
            base_turn_candidates.append(interrupted_turn_id)
        if self._active_turn_id is not None:
            base_turn_candidates.append(self._active_turn_id)
        if self.turns:
            base_turn_candidates.append(self.turns[-1].turn_id)
        base_turn_id = max(base_turn_candidates)
        human_turn_id = max(1, base_turn_id + 1)
        self.controller.set_speaker_for_turn(human_turn_id, human_counselor)
        self._active_turn_id = human_turn_id
        self._active_speaker_id = human_counselor
        self._human_input_turn_id = human_turn_id
        self._human_input_speaker = human_counselor
        self._barge_in_human_turn_id = human_turn_id
        self._barge_in_human_speaker = human_counselor
        self._human_input_state = "waiting_for_human_speech"
        self._last_event_type = "human_barge_in_wait_started"
        return {
            "awaiting_human_input": True,
            "human_turn_id": human_turn_id,
            "human_speaker": human_counselor,
        }

    async def _await_human_turn(
        self,
        *,
        turn_id: int,
        speaker: str,
        streaming_transcript_observer: TranscriptObserver | None = None,
    ) -> TurnRuntimeState:
        if self.config.interaction_mode is InteractionMode.AI_COUNSELOR_HUMAN_CLIENT:
            await self._wait_for_human_operation(self._generation_resume_event.wait())
        if (
            self._barge_in_human_turn_id is not None
            and self._barge_in_human_speaker is not None
            and turn_id < self._barge_in_human_turn_id
        ):
            await self._log_barge_in_reserved_turn_advanced(
                from_turn_id=turn_id,
                reserved_turn_id=self._barge_in_human_turn_id,
                speaker=self._barge_in_human_speaker,
            )
            turn_id = self._barge_in_human_turn_id
            speaker = self._barge_in_human_speaker
            self._active_turn_id = turn_id
            self._active_speaker_id = speaker
        participant = self.config.participants[speaker]

        def mark_barge_in_human_turn_consumed() -> None:
            self._interrupted_ai_turn_id = None
            self._empty_barge_in_resume_deadline = None
            if (
                self._barge_in_human_turn_id == turn_id
                and self._barge_in_human_speaker == speaker
            ):
                self._barge_in_human_turn_id = None
                self._barge_in_human_speaker = None
                self._post_barge_in_recovery_turns_remaining = 2

        state = TurnRuntimeState(
            session_id=self.config.session_id,
            turn_id=turn_id,
            speaker=speaker,
            input_transcript="",
            recipient_ids=self.controller.recipient_ids_for_speaker(speaker),
        )
        self._human_input_turn_id = turn_id
        self._human_input_speaker = speaker
        self._human_input_state = (
            "waiting_to_resume_interrupted_ai"
            if self._empty_barge_in_resume_deadline is not None
            else "waiting_for_human_speech"
        )
        self._last_event_type = "human_input_wait_started"
        await _log_event(
            self.logger,
            state,
            "human_input_wait_started",
            {
                "input_mode": self.config.human_input_mode,
                "stt_submit_policy": self.config.human_stt_submit_policy,
                "interrupts_enabled": self.config.human_interrupts_enabled,
            },
        )
        try:
            request = await self._next_human_request(state)
        finally:
            self._human_input_turn_id = None
            self._human_input_speaker = None
        self._human_input_state = "human_turn_committing"
        if isinstance(request, HumanStreamingAudioTurnInput):
            state.recipient_ids = request.request.recipient_ids
            state.audio_log_path = request.source_audio_ref
            state.audio_delivered_to_stt = True
            state.tts_done = True
            transcribe_from_queue_observed = getattr(
                self.controller.stt,
                "transcribe_from_queue_observed",
                None,
            )
            if not callable(transcribe_from_queue_observed):
                raise RuntimeError("streaming STT does not support queued audio input")

            partial_event_count = 0
            partial_text_observed = False
            stt_empty_error: RuntimeError | None = None

            async def observe_human_streaming_transcript(
                transcript: TranscriptEvent,
            ) -> None:
                nonlocal partial_event_count, partial_text_observed
                if request.aborted.is_set():
                    return
                if transcript.transcript_type == "partial":
                    partial_event_count += 1
                    partial_text_observed = partial_text_observed or bool(
                        transcript.text.strip()
                    )
                # A final transcript is public only after this input is committed.
                if (
                    transcript.transcript_type == "final"
                    and not request.committed.is_set()
                ):
                    return
                display_transcript = transcript
                if transcript.transcript_type == "final":
                    display_transcript = replace(
                        transcript,
                        transcript_type="human_final",
                    )
                display_transcript = _with_transcript_display_metadata(
                    display_transcript,
                    participant=participant,
                    recipient_ids=request.request.recipient_ids,
                    text_source="human_streaming_stt",
                    source_audio_ref=request.source_audio_ref,
                )
                await _publish_monitor_transcript_event(
                    self.audio_bus,
                    self.logger,
                    display_transcript,
                )
                await _notify_transcript_observer(
                    streaming_transcript_observer,
                    transcript,
                )

            self._human_input_state = "streaming_human_audio_transcribing"
            stt_kwargs: dict[str, Any] = {
                "session_id": self.config.session_id,
                "turn_id": turn_id,
                "speaker": speaker,
                "queue": request.audio_queue,
                "on_transcript": observe_human_streaming_transcript,
            }
            if _callable_accepts_keyword(
                transcribe_from_queue_observed,
                "recording_mode",
            ):
                stt_kwargs["recording_mode"] = request.request.recording_mode
            try:
                transcription = await self._transcribe_human_stream(
                    request,
                    transcribe_from_queue_observed,
                    stt_kwargs,
                )
                if transcription is None or request.aborted.is_set():
                    return await self._await_human_turn(
                        turn_id=turn_id,
                        speaker=speaker,
                        streaming_transcript_observer=streaming_transcript_observer,
                    )
                _partials, final = transcription
            except RuntimeError as exc:
                if not _is_empty_human_audio_stream_error(exc):
                    raise
                stt_empty_error = exc
                final = TranscriptEvent(
                    session_id=self.config.session_id,
                    turn_id=turn_id,
                    speaker=speaker,
                    transcript_type="final",
                    text="",
                )
            if request.request.recording_mode == "vad_auto":
                await self._complete_human_audio_stream(
                    request,
                    completion_reason="server_vad",
                    require_active=False,
                )
            final_text = _normalized_human_stt_final_text(final.text)
            if request.aborted.is_set():
                return await self._await_human_turn(
                    turn_id=turn_id,
                    speaker=speaker,
                    streaming_transcript_observer=streaming_transcript_observer,
                )
            if final_text:
                request.committed.set()
                await observe_human_streaming_transcript(final)
            elif (
                self.config.interaction_mode is InteractionMode.AI_COUNSELOR_HUMAN_CLIENT
                and request.request.recording_mode == "browser_vad"
                and self._interrupted_ai_turn_id is not None
                and stt_empty_error is None
                and not partial_text_observed
                and self._generation_resume_event.is_set()
            ):
                self._empty_barge_in_resume_deadline = (
                    time.monotonic() + EMPTY_BARGE_IN_RESUME_DELAY_SECONDS
                )
                await _log_event(
                    self.logger, state, "empty_barge_in_resume_scheduled",
                    {
                        "interrupted_turn_id": self._interrupted_ai_turn_id,
                        "delay_seconds": EMPTY_BARGE_IN_RESUME_DELAY_SECONDS,
                        "stream_id": request.stream_id,
                    },
                )
            if (
                self.config.interaction_mode
                is InteractionMode.AI_COUNSELOR_HUMAN_CLIENT
            ):
                await self._complete_human_audio_stream(
                    request,
                    completion_reason=(
                        "transcribed" if final_text else "blank_transcript"
                    ),
                    require_active=False,
                    text=final_text,
                )
            state.input_transcript = final_text
            state.stt_final_transcript = final_text
            state.stt_final_monotonic = time.monotonic()
            if not final_text:
                self._human_input_state = "waiting_for_human_speech"
                self._last_event_type = "human_stt_blank"
                await _log_event(
                    self.logger,
                    state,
                    "human_stt_blank",
                    {
                        "input_mode": "human_streaming_stt",
                        "stt_outcome": (
                            "empty_audio_error" if stt_empty_error else "empty_transcript"
                        ),
                        "stt_error_code": getattr(stt_empty_error, "code", None),
                        "partial_event_count": partial_event_count,
                        "recording_mode": request.request.recording_mode,
                        "source_audio_ref": request.source_audio_ref,
                        "client_message_id": request.request.client_message_id,
                        "stream_id": request.stream_id,
                    },
                )
                return await self._await_human_turn(
                    turn_id=turn_id,
                    speaker=speaker,
                    streaming_transcript_observer=streaming_transcript_observer,
                )
            await _log_event(
                self.logger,
                state,
                "human_streaming_audio_committed",
                {
                    "input_mode": "human_streaming_stt",
                    "recording_mode": request.request.recording_mode,
                    "source_audio_ref": request.source_audio_ref,
                    "client_message_id": request.request.client_message_id,
                    "char_count": len(final_text),
                    "stream_id": request.stream_id,
                },
            )
            state.mark_completed()
            state.completed_monotonic = time.monotonic()
            self._human_input_state = None
            mark_barge_in_human_turn_consumed()
            return state

        if isinstance(request, HumanRealtimeAudioTurnInput):
            state.recipient_ids = request.request.recipient_ids
            state.stt_final_transcript = ""
            state.stt_final_monotonic = time.monotonic()
            state.audio_delivered_to_stt = True
            state.tts_done = True
            state.audio_log_path = request.source_audio_ref
            self._pending_realtime_human_audio[turn_id] = request
            await _log_event(
                self.logger,
                state,
                "human_realtime_audio_committed",
                {
                    "input_mode": "realtime_audio",
                    "recording_mode": request.request.recording_mode,
                    "source_audio_ref": request.source_audio_ref,
                    "client_message_id": request.request.client_message_id,
                    "target_speaker": request.target_speaker,
                    "audio_duration_ms": request.audio_chunk.duration_ms,
                },
            )
            state.mark_completed()
            state.completed_monotonic = time.monotonic()
            self._human_input_state = None
            mark_barge_in_human_turn_consumed()
            return state

        state.stt_final_transcript = request.text
        state.stt_final_monotonic = time.monotonic()
        state.audio_delivered_to_stt = True
        state.tts_done = True
        state.audio_log_path = request.source_audio_ref
        await _publish_monitor_transcript_event(
            self.audio_bus,
            self.logger,
            TranscriptEvent(
                session_id=self.config.session_id,
                turn_id=turn_id,
                speaker=speaker,
                speaker_id=speaker,
                transcript_type="human_final",
                text=request.text,
                recipient_ids=request.recipient_ids,
                metadata={
                    "speaker_display_name": participant.display_name,
                    "role": participant.role,
                    "actor_kind": participant.actor_kind.value,
                    "text_source": request.input_mode,
                    "recording_mode": request.recording_mode,
                    "source_audio_ref": request.source_audio_ref,
                    "client_message_id": request.client_message_id,
                },
            ),
        )
        await _log_event(
            self.logger,
            state,
            "human_turn_committed",
            {
                "input_mode": request.input_mode,
                "recording_mode": request.recording_mode,
                "source_audio_ref": request.source_audio_ref,
                "client_message_id": request.client_message_id,
                "char_count": len(request.text),
            },
        )
        state.mark_completed()
        state.completed_monotonic = time.monotonic()
        self._human_input_state = None
        mark_barge_in_human_turn_consumed()
        return state

    def _set_generation_prompt_context_for_turn(
        self,
        *,
        turn_id: int,
        current_objective: str,
        response_target: str | None = None,
    ) -> None:
        if self.prompt_context_store is None:
            return
        self.prompt_context_store.set_context(
            turn_id,
            build_runtime_prompt_context(
                turns=self.turns,
                session_summary=self._session_summary_state.text,
                current_objective=current_objective,
                response_target=_response_target_context_text(
                    self.config,
                    response_target,
                ),
                previous_turn_target_hint=self._previous_turn_target_hint,
                recent_turn_limit=self.config.conversation_context_recent_turns,
                last_summarized_turn_id=self._session_summary_state.last_summarized_turn_id,
            ),
        )

    def _current_objective_for_turn(
        self,
        *,
        turn_id: int,
        closing_instruction: str | None,
    ) -> str:
        _ = turn_id
        if closing_instruction or self._closing_started_turn_id is not None:
            return "クロージング。プロンプトの指示に従ってクロージングを行う。"
        return (
            "カウンセリングの継続。直近の発話と共有履歴を踏まえて、"
            "プロンプトの指示に従って、カウンセリングを継続する。"
        )

    async def _add_prompt_director_instruction(
        self,
        *,
        turn_id: int,
        speaker: str,
        input_transcript: str,
        current_objective: str,
        response_target: str | None,
        existing_instruction: str | None,
        speculative: bool = False,
    ) -> str | None:
        participant = self.config.participants[speaker]
        session_end_context = self._session_end_context(speaker)
        if session_end_context is not None:
            existing_instruction = _combine_additional_instructions(
                existing_instruction,
                SESSION_TIME_INSTRUCTION,
                (
                    "現在は時間枠の終盤です。"
                    if session_end_context.allow_agreed_end
                    else "現在はまだ時間枠の終盤ではありません。"
                ),
            )
        if (
            self.prompt_director is None
            or participant.actor_kind is not ActorKind.AI
            or participant.role != COUNSELOR_ROLE
        ):
            return existing_instruction
        request = PromptDirectorRequest(
            turn_id=turn_id,
            speaker_id=speaker,
            fixed_system_prompt=COUNSELOR_SYSTEM_PROMPT,
            configured_prompt=participant.prompt_source or "",
            shared_context=self.config.shared_case,
            speaker_profile=_prompt_director_speaker_profile(participant),
            public_history=self._prompt_director_public_history(
                input_transcript=input_transcript,
            ),
            session_summary=self._session_summary_state.text,
            current_objective=current_objective,
            response_target=_response_target_context_text(
                self.config,
                response_target,
            ),
            turn_specific_instructions=(existing_instruction or "").strip(),
            session_end_context=session_end_context,
        )
        while True:
            task = self._prompt_director_tasks.get(request)
            if task is None:
                task = asyncio.create_task(self._run_prompt_director(request))
                self._prompt_director_tasks[request] = task
            try:
                result = await asyncio.shield(task)
                break
            except PromptDirectorRetriesExhausted as exc:
                if speculative:
                    # A speculative speaker may never be selected. Only the
                    # actual matching turn is allowed to pause the conversation.
                    raise
                self._generation_resume_event.clear()
                self._prompt_director_pause_reason = exc.pause_reason
                self._last_event_type = "prompt_director_waiting_for_resume"
                await self.logger.log_event(
                    RuntimeEvent(
                        session_id=self.config.session_id,
                        event_type=self._last_event_type,
                        turn_id=turn_id,
                        speaker=speaker,
                        details={
                            "pause_reason": exc.pause_reason,
                            "message": (
                                "通信の再試行後も応答を取得できませんでした。履歴を保持しています。再開すると同じターンを再生成します。"
                                if exc.pause_reason == "prompt_director_transport"
                                else "応答の再生成後も発話内容または引用の検証を通過できませんでした。履歴を保持しています。再開すると同じターンを再生成します。"
                            ),
                        },
                    )
                )
                await self._generation_resume_event.wait()
                if self._prompt_director_tasks.get(request) is task:
                    del self._prompt_director_tasks[request]
        if not speculative:
            await self._apply_early_closing_assessment(request, result)
            if self._early_closing_reason is not None:
                existing_instruction = _combine_additional_instructions(
                    existing_instruction,
                    CLOSING_COUNSELOR_INSTRUCTION,
                )
            self._selected_prompt_director_texts[(turn_id, speaker)] = (
                result.response_example
            )
        return _combine_additional_instructions(
            existing_instruction,
            render_prompt_director_instruction(result, include_audit=False),
        )

    def _session_end_context(self, speaker: str) -> SessionEndContext | None:
        scheduled = self.config.closing_start_elapsed_seconds
        if (
            self._session_started_monotonic is None
            or self._closing_started_turn_id is not None
            or scheduled is None
            or scheduled <= 0
            or self.config.participants[speaker].role != COUNSELOR_ROLE
        ):
            return None
        return SessionEndContext(
            client_ids=tuple(
                participant.speaker_id
                for participant in self.config.participants.values()
                if participant.role == CLIENT_ROLE
            ),
            allow_agreed_end=(time.monotonic() - self._session_started_monotonic)
            >= scheduled * EARLY_CLOSING_PROGRESS_RATIO,
        )

    async def _apply_early_closing_assessment(
        self, request: PromptDirectorRequest, result: PromptDirectorResult
    ) -> None:
        assessment = result.session_end_assessment
        context = request.session_end_context
        if (
            context is None
            or assessment is None
            or self._closing_started_turn_id is not None
        ):
            return
        reason = None
        if assessment.explicit_end_request_client_ids:
            reason = "client_requested_end"
        elif (
            context.allow_agreed_end
            and assessment.counselor_proposed_end
            and set(assessment.consenting_client_ids) == set(context.client_ids)
            and not assessment.pending_question
        ):
            reason = "agreed_end"
        await self.logger.log_event(
            RuntimeEvent(
                session_id=self.config.session_id,
                event_type="session_end_assessed",
                turn_id=request.turn_id,
                speaker=request.speaker_id,
                details={
                    **assessment.model_dump(),
                    "allow_agreed_end": context.allow_agreed_end,
                    "accepted_reason": reason,
                },
            )
        )
        if reason is None:
            return
        self._early_closing_reason = reason
        self._closing_started_turn_id = request.turn_id
        self._closing_count_started_turn_id = request.turn_id
        assert self._session_started_monotonic is not None
        await self._log_closing_started(
            request.turn_id, self._session_started_monotonic
        )

    async def _audit_prompt_director_output(self, turn: TurnRuntimeState) -> None:
        expected = self._selected_prompt_director_texts.pop(
            (turn.turn_id, turn.speaker), None
        )
        if expected is None:
            return
        actual = turn.generated_text
        await self.logger.log_event(
            RuntimeEvent(
                session_id=self.config.session_id,
                event_type=(
                    "prompt_director_output_matched"
                    if actual == expected
                    else "prompt_director_output_mismatch"
                ),
                turn_id=turn.turn_id,
                speaker=turn.speaker,
                speaker_id=turn.speaker,
                details={
                    "expected_char_count": len(expected),
                    "actual_char_count": len(actual),
                    "expected_sha256": hashlib.sha256(
                        expected.encode("utf-8")
                    ).hexdigest(),
                    "actual_sha256": hashlib.sha256(actual.encode("utf-8")).hexdigest(),
                },
            )
        )

    def _prompt_director_public_history(
        self,
        *,
        input_transcript: str,
    ) -> tuple[PublicHistoryMessage, ...]:
        history = list(
            recent_public_history(
                self.turns,
                limit=self.config.conversation_context_recent_turns,
                retain_after_turn_id=(
                    self._session_summary_state.last_summarized_turn_id
                    if self._session_summary_state.text.strip()
                    else -1
                ),
            )
        )
        normalized_input = input_transcript.strip()
        if normalized_input and (
            not history or history[-1].text.strip() != normalized_input
        ):
            source_speaker = self._active_speaker_id or (
                self.turns[-1].speaker if self.turns else ""
            )
            history.append(
                PublicHistoryMessage(
                    speaker_id=source_speaker,
                    text=normalized_input,
                )
            )
        return tuple(history)

    async def _run_prompt_director(
        self,
        request: PromptDirectorRequest,
    ) -> PromptDirectorResult:
        director = self.prompt_director
        if director is None:
            raise RuntimeError("Prompt Director is not configured")
        started_at = time.monotonic()
        participant = self.config.participants[request.speaker_id]
        attempt_group_id = uuid4().hex

        async def record_attempt(details: dict[str, Any]) -> None:
            llm = director.llm
            await self.logger.log_prompt_director_attempt(
                PromptDirectorAttemptRecord(
                    session_id=self.config.session_id,
                    turn_id=request.turn_id,
                    speaker_id=request.speaker_id,
                    details={
                        **details,
                        "attempt_group_id": attempt_group_id,
                        "request": asdict(request),
                        "route": self.ai_route_manifest.get("prompt_director", {}),
                        "model": getattr(llm, "_model", None),
                        "reasoning_effort": getattr(llm, "_reasoning_effort", None),
                        "max_output_tokens": getattr(llm, "_max_output_tokens", None),
                    },
                )
            )
            if details.get("transport_error"):
                await self.logger.log_event(
                    RuntimeEvent(
                        session_id=self.config.session_id,
                        event_type=(
                            "prompt_director_transport_retry"
                            if details["will_retry"]
                            else "prompt_director_transport_exhausted"
                        ),
                        turn_id=request.turn_id,
                        speaker=request.speaker_id,
                        details={
                            "attempt_group_id": attempt_group_id,
                            "stage": details["stage"],
                            "transport_attempt": details["transport_attempt"],
                            "will_retry": details["will_retry"],
                            "retry_delay_seconds": details["retry_delay_seconds"],
                            **details["transport_error"],
                        },
                    )
                )
            elif details["validation_error"] or details["response_issues"]:
                event_type = (
                    "prompt_director_regeneration_started"
                    if details["will_retry"]
                    else "prompt_director_validation_failed"
                )
                await self.logger.log_event(
                    RuntimeEvent(
                        session_id=self.config.session_id,
                        event_type=event_type,
                        turn_id=request.turn_id,
                        speaker=request.speaker_id,
                        details={
                            "attempt_group_id": attempt_group_id,
                            "stage": details["stage"],
                            "attempt": details["attempt"],
                            "regeneration_round": details["regeneration_round"],
                            "response_issue_count": len(details["response_issues"]),
                            "will_retry": details["will_retry"],
                        },
                    )
                )

        await self.logger.log_event(
            RuntimeEvent(
                session_id=self.config.session_id,
                event_type="prompt_director_request_started",
                turn_id=request.turn_id,
                speaker=request.speaker_id,
                speaker_id=request.speaker_id,
                recipient_ids=self.controller.recipient_ids_for_speaker(
                    request.speaker_id
                ),
                monotonic_time=started_at,
                details={
                    "configured_prompt_sha256": hashlib.sha256(
                        request.configured_prompt.encode("utf-8")
                    ).hexdigest(),
                    "history_turn_count": len(request.public_history),
                    "display_name": participant.display_name,
                },
            )
        )
        try:
            result = (
                await director.create_directive(request, on_attempt=record_attempt)
                if isinstance(director, PromptDirector)
                else await director.create_directive(request)
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            await self.logger.log_event(
                RuntimeEvent(
                    session_id=self.config.session_id,
                    event_type="prompt_director_error",
                    turn_id=request.turn_id,
                    speaker=request.speaker_id,
                    speaker_id=request.speaker_id,
                    recipient_ids=self.controller.recipient_ids_for_speaker(
                        request.speaker_id
                    ),
                    monotonic_time=time.monotonic(),
                    details={
                        "error_type": type(exc).__name__,
                        "message": str(exc),
                    },
                )
            )
            raise
        completed_at = time.monotonic()
        latency_ms = _elapsed_ms(started_at, completed_at)
        await self.logger.log_metric(
            RuntimeMetric(
                session_id=self.config.session_id,
                metric_name="prompt_director_latency_ms",
                value=latency_ms,
                turn_id=request.turn_id,
                speaker=request.speaker_id,
                speaker_id=request.speaker_id,
                recipient_ids=self.controller.recipient_ids_for_speaker(
                    request.speaker_id
                ),
            )
        )
        await self.logger.log_event(
            RuntimeEvent(
                session_id=self.config.session_id,
                event_type="prompt_director_completed",
                turn_id=request.turn_id,
                speaker=request.speaker_id,
                speaker_id=request.speaker_id,
                recipient_ids=self.controller.recipient_ids_for_speaker(
                    request.speaker_id
                ),
                monotonic_time=completed_at,
                details={
                    "latency_ms": latency_ms,
                    "instruction_check_count": len(result.instruction_checks),
                    "response_intent_char_count": len(result.response_intent),
                    "response_example_char_count": len(result.response_example),
                },
            )
        )
        return result

    async def _cancel_prompt_director_tasks(self) -> None:
        prefetch_tasks = tuple(self._prompt_director_prefetch_tasks)
        self._prompt_director_prefetch_tasks.clear()
        tasks = tuple(self._prompt_director_tasks.values())
        self._prompt_director_tasks.clear()
        for task in (*prefetch_tasks, *tasks):
            if not task.done():
                task.cancel()
        if prefetch_tasks or tasks:
            await asyncio.gather(
                *prefetch_tasks,
                *tasks,
                return_exceptions=True,
            )
        self._selected_prompt_director_texts.clear()

    def _prefetch_prompt_director_instruction(
        self,
        *,
        turn_id: int,
        speaker: str,
        input_transcript: str,
        current_objective: str,
        response_target: str | None,
        existing_instruction: str | None,
    ) -> None:
        if self.config.interaction_mode is InteractionMode.AI_COUNSELOR_HUMAN_CLIENT:
            return
        participant = self.config.participants[speaker]
        if (
            self.prompt_director is None
            or participant.actor_kind is not ActorKind.AI
            or participant.role != COUNSELOR_ROLE
        ):
            return
        task = asyncio.create_task(
            self._add_prompt_director_instruction(
                turn_id=turn_id,
                speaker=speaker,
                input_transcript=input_transcript,
                current_objective=current_objective,
                response_target=response_target,
                existing_instruction=existing_instruction,
                speculative=True,
            )
        )
        self._prompt_director_prefetch_tasks.add(task)

        def consume_prefetch_result(completed: asyncio.Task[str | None]) -> None:
            self._prompt_director_prefetch_tasks.discard(completed)
            try:
                completed.result()
            except (asyncio.CancelledError, Exception):
                # The cached director task preserves the error for an actual
                # matching AI participant turn. A speculative mismatch must
                # not fail the currently selected turn.
                pass

        task.add_done_callback(consume_prefetch_result)

    async def _stop_response_playback_for_speaker(
        self,
        *,
        speaker: str,
        played_ms: int,
        item_id: str | None = None,
        response_id: str | None = None,
        content_index: int | None = None,
        cancel_response: bool = True,
        truncate_item: bool = True,
    ) -> dict[str, Any]:
        agent = self.controller.agents.get(speaker)
        if agent is None:
            raise RuntimeError(f"unknown speaker: {speaker}")
        hook = getattr(agent, "stop_current_response_playback", None)
        if hook is None:
            raise RuntimeError(f"speaker does not support response playback stop: {speaker}")
        result = hook(
            played_ms=played_ms,
            item_id=item_id,
            response_id=response_id,
            content_index=content_index,
            cancel_response=cancel_response,
            truncate_item=truncate_item,
        )
        if inspect.isawaitable(result):
            result = await result
        if hasattr(result, "__dataclass_fields__"):
            return asdict(result)
        if isinstance(result, Mapping):
            return dict(result)
        return {"result": result}

    async def stop_current_response_playback(
        self,
        *,
        played_ms: int,
        speaker: str | None = None,
        turn_id: int | None = None,
        item_id: str | None = None,
        response_id: str | None = None,
        content_index: int | None = None,
        cancel_response: bool = True,
        truncate_item: bool = True,
        reason: str | None = None,
    ) -> dict[str, Any]:
        target_speaker = speaker or self.status.current_speaker
        if not target_speaker:
            raise RuntimeError("speaker is required to stop current response playback")
        if (
            reason == "human_barge_in" and target_speaker == "counselor"
            and self.config.interaction_mode is InteractionMode.AI_COUNSELOR_HUMAN_CLIENT
        ):
            self._interrupted_ai_turn_id = turn_id or self._running_ai_turn_id
        payload = await self._stop_response_playback_for_speaker(
            speaker=target_speaker,
            played_ms=played_ms,
            item_id=item_id,
            response_id=response_id,
            content_index=content_index,
            cancel_response=cancel_response,
            truncate_item=truncate_item,
        )
        active_response_stop: dict[str, Any] | None = None
        running_speaker = self._running_ai_speaker
        running_turn_id = self._running_ai_turn_id
        running_response_is_distinct = (
            running_speaker != target_speaker
            or (
                running_turn_id is not None
                and turn_id is not None
                and running_turn_id != turn_id
            )
        )
        if (
            reason == "human_barge_in"
            and running_speaker is not None
            and running_response_is_distinct
        ):
            active_response_stop = await self._stop_response_playback_for_speaker(
                speaker=running_speaker,
                played_ms=0,
                cancel_response=True,
                truncate_item=False,
            )
            if running_turn_id is not None:
                active_response_stop.setdefault("turn_id", running_turn_id)
            active_response_stop.setdefault("speaker", running_speaker)
            active_response_stop.setdefault("speaker_id", running_speaker)
        if reason == "human_barge_in":
            await self._publish_interrupted_monitor_turn(
                turn_id=turn_id,
                speaker=target_speaker,
                reason=reason,
            )
            if (
                active_response_stop is not None
                and running_speaker is not None
                and running_turn_id is not None
            ):
                await self._publish_interrupted_monitor_turn(
                    turn_id=running_turn_id,
                    speaker=running_speaker,
                    reason=reason,
                )
            invalidated_turns = await self._invalidate_future_ai_turns_after_interrupt(
                interrupted_turn_id=turn_id,
                reason=reason,
            )
        else:
            invalidated_turns = []
        human_input_slot = self._open_human_input_slot_after_interrupt(
            interrupted_turn_id=turn_id,
        )
        payload.update(human_input_slot)
        if invalidated_turns:
            payload["invalidated_turn_ids"] = [
                invalidated.turn_id for invalidated in invalidated_turns
            ]
            payload["invalidated_turns"] = [
                {
                    "turn_id": invalidated.turn_id,
                    "speaker": invalidated.speaker,
                    "speaker_id": invalidated.speaker,
                }
                for invalidated in invalidated_turns
            ]
        if active_response_stop is not None:
            payload["active_response_stop"] = active_response_stop
        payload.setdefault("played_ms", played_ms)
        payload.setdefault("speaker", target_speaker)
        payload.setdefault("turn_id", turn_id)
        payload.setdefault("reason", reason)
        await self.logger.log_event(
            RuntimeEvent(
                session_id=self.config.session_id,
                event_type="realtime_playback_interrupted",
                turn_id=turn_id,
                speaker=target_speaker,
                speaker_id=target_speaker,
                recipient_ids=self.controller.recipient_ids_for_speaker(target_speaker),
                monotonic_time=time.monotonic(),
                details={
                    "played_ms": payload["played_ms"],
                    "requested_played_ms": played_ms,
                    "audio_end_ms": payload.get("audio_end_ms"),
                    "reason": reason,
                    "item_id": item_id,
                    "response_id": response_id,
                    "content_index": content_index,
                    "cancel_response": cancel_response,
                    "truncate_item": truncate_item,
                    "human_input_slot": human_input_slot,
                    "active_response_stop": active_response_stop,
                    "invalidated_turn_ids": [
                        invalidated.turn_id for invalidated in invalidated_turns
                    ],
                    "invalidated_turns": [
                        {
                            "turn_id": invalidated.turn_id,
                            "speaker": invalidated.speaker,
                            "speaker_id": invalidated.speaker,
                        }
                        for invalidated in invalidated_turns
                    ],
                    "result": payload,
                },
            )
        )
        return payload

    async def _publish_interrupted_monitor_turn(
        self,
        *,
        turn_id: int | None,
        speaker: str | None,
        reason: str | None,
    ) -> None:
        if turn_id is None or speaker is None:
            return
        participant = self.config.participants.get(speaker)
        if participant is None:
            return
        state = TurnRuntimeState(
            session_id=self.config.session_id,
            turn_id=turn_id,
            speaker=speaker,
            input_transcript="",
            recipient_ids=self.controller.recipient_ids_for_speaker(speaker),
        )
        await _publish_monitor_transcript_interrupted(
            self.audio_bus,
            state,
            speaker_display_name=participant.display_name,
            speaker_role=participant.role,
            reason=reason or "interrupted",
        )

    async def _publish_invalidated_monitor_turn(
        self,
        *,
        turn: TurnRuntimeState,
        reason: str,
        interrupted_turn_id: int | None,
    ) -> None:
        participant = self.config.participants.get(turn.speaker)
        if participant is None:
            return
        await self.audio_bus.publish_monitor_event(
            TranscriptEvent(
                session_id=self.config.session_id,
                turn_id=turn.turn_id,
                speaker=turn.speaker,
                speaker_id=turn.speaker,
                transcript_type="invalidated",
                text="",
                recipient_ids=turn.recipient_ids,
                metadata={
                    "speaker_display_name": participant.display_name,
                    "role": participant.role,
                    "text_source": "generated_text",
                    "reason": reason,
                    "interrupted_turn_id": interrupted_turn_id,
                },
            )
        )

    async def _invalidate_future_ai_turns_after_interrupt(
        self,
        *,
        interrupted_turn_id: int | None,
        reason: str | None,
    ) -> list[TurnRuntimeState]:
        if interrupted_turn_id is None:
            return []
        invalidated: list[TurnRuntimeState] = []
        retained: list[TurnRuntimeState] = []
        for turn in self.turns:
            if turn.turn_id > interrupted_turn_id and self._speaker_is_ai(turn.speaker):
                invalidated.append(turn)
                continue
            retained.append(turn)
        if not invalidated:
            return []

        self.turns = retained
        invalidation_reason = reason or "human_barge_in"
        for turn in invalidated:
            await self._publish_invalidated_monitor_turn(
                turn=turn,
                reason=invalidation_reason,
                interrupted_turn_id=interrupted_turn_id,
            )
        await self.logger.log_event(
            RuntimeEvent(
                session_id=self.config.session_id,
                event_type="barge_in_future_turns_invalidated",
                turn_id=interrupted_turn_id,
                speaker=None,
                monotonic_time=time.monotonic(),
                details={
                    "reason": invalidation_reason,
                    "interrupted_turn_id": interrupted_turn_id,
                    "invalidated_turn_ids": [turn.turn_id for turn in invalidated],
                    "invalidated_turns": [
                        {
                            "turn_id": turn.turn_id,
                            "speaker": turn.speaker,
                            "speaker_id": turn.speaker,
                            "text_source": "generated_text",
                        }
                        for turn in invalidated
                    ],
                },
            )
        )
        return invalidated

    async def run(self) -> list[TurnRuntimeState]:
        self._phase = RuntimePhase.RUNNING
        await self.logger.start()
        await self._log_resolved_audio_settings()
        logger_consumer = asyncio.create_task(
            self.logger.consume_audio(self.audio_bus.logger_queue)
        )
        started_monotonic = time.monotonic()
        self._session_started_monotonic = started_monotonic
        if (
            self.config.interaction_mode is InteractionMode.AI_COUNSELOR_HUMAN_CLIENT
            and self.config.stop_condition is RuntimeStopCondition.ELAPSED_TIME
        ):
            self._human_input_deadline = (
                started_monotonic + self.config.max_elapsed_seconds
            )
        next_input = self.config.initial_client_transcript
        pending_provisional: ProvisionalGeneration | None = None
        active_provisional: ProvisionalGeneration | None = None
        floor_provisionals: list[ProvisionalGeneration] = []
        turn_taking_trackers: dict[int, TurnTakingSignalTracker] = {}
        current_turn_id: int | None = None
        current_speaker: str | None = None
        try:
            initial_client_turn = (
                _initial_client_turn(self.config)
                if self.config.interaction_mode
                is InteractionMode.AI_COUNSELOR_AI_CLIENT
                else None
            )
            if initial_client_turn is not None:
                initial_speaker, initial_transcript = initial_client_turn
                current_turn_id = 0
                current_speaker = initial_speaker
                self._active_turn_id = current_turn_id
                self._active_speaker_id = current_speaker
                initial_instruction = _ai_counselor_initial_client_opening_instruction(
                    self.config,
                    speaker=initial_speaker,
                    initial_transcript=initial_transcript,
                )
                initial_response_target = _fallback_response_target(
                    config=self.config,
                    speaker=initial_speaker,
                    previous_speaker=None,
                )
                initial_objective = self._current_objective_for_turn(
                    turn_id=0,
                    closing_instruction=None,
                )
                self._set_generation_prompt_context_for_turn(
                    turn_id=0,
                    current_objective=initial_objective,
                    response_target=initial_response_target,
                )
                initial_instruction = await self._add_prompt_director_instruction(
                    turn_id=0,
                    speaker=initial_speaker,
                    input_transcript="",
                    current_objective=initial_objective,
                    response_target=initial_response_target,
                    existing_instruction=initial_instruction,
                )
                initial_turn = await self.controller.run_turn(
                    turn_id=0,
                    speaker=initial_speaker,
                    input_transcript="",
                    audio_bus=self.audio_bus,
                    logger=self.logger,
                    additional_instruction=initial_instruction,
                )
                self.turns.append(initial_turn)
                await self._audit_prompt_director_output(initial_turn)
                next_input = (
                    initial_turn.generated_text
                    or initial_turn.stt_final_transcript
                    or initial_transcript
                )
                await self._maybe_schedule_session_summary_update(
                    current_objective=self._current_objective_for_turn(
                        turn_id=1,
                        closing_instruction=None,
                    )
                )

            turn_id = 1
            while True:
                await self._generation_resume_event.wait()
                await self._finish_session_summary_task_if_ready()
                if (
                    self._barge_in_human_turn_id is not None
                    and self._barge_in_human_speaker is not None
                ):
                    if turn_id < self._barge_in_human_turn_id:
                        await self._log_barge_in_reserved_turn_advanced(
                            from_turn_id=turn_id,
                            reserved_turn_id=self._barge_in_human_turn_id,
                            speaker=self._barge_in_human_speaker,
                        )
                        turn_id = self._barge_in_human_turn_id
                    current_turn_id = turn_id
                    current_speaker = self._barge_in_human_speaker
                    self._active_turn_id = turn_id
                    self._active_speaker_id = current_speaker
                    try:
                        turn = await self._await_human_turn(
                            turn_id=turn_id,
                            speaker=current_speaker,
                        )
                    except ResumeAfterEmptyBargeIn as resume:
                        turn_id = resume.turn_id
                        continue
                    self._barge_in_human_turn_id = None
                    self._barge_in_human_speaker = None
                    self._post_barge_in_recovery_turns_remaining = 2
                    if self.turns:
                        await self._log_inter_turn_latency(self.turns[-1], turn)
                    self.turns.append(turn)
                    next_input = turn.stt_final_transcript or ""
                    await self._maybe_schedule_session_summary_update(
                        current_objective=self._current_objective_for_turn(
                            turn_id=turn.turn_id + 1,
                            closing_instruction=None,
                        )
                    )
                    turn_id = turn.turn_id + 1
                    continue
                await self._assess_closing_before_stop(turn_id)
                if not self._should_start_turn(
                    turn_id,
                    started_monotonic=started_monotonic,
                ):
                    pending_closing_replies = self._closing_pending_reply_speakers()
                    if pending_closing_replies:
                        await self.logger.log_event(
                            RuntimeEvent(
                                session_id=self.config.session_id,
                                event_type="closing_reply_limit_reached",
                                turn_id=self.turns[-1].turn_id,
                                speaker=self.turns[-1].speaker,
                                details={
                                    "reply_speaker_ids": list(pending_closing_replies)
                                },
                            )
                        )
                    break
                current_turn_id = turn_id
                self._active_turn_id = turn_id
                closing_next_speaker = self._closing_next_speaker()
                if closing_next_speaker is not None:
                    self.controller.set_speaker_for_turn(turn_id, closing_next_speaker)
                if (
                    self.config.interaction_mode
                    is InteractionMode.AI_COUNSELOR_HUMAN_CLIENT
                ):
                    self.controller.set_speaker_for_turn(
                        turn_id,
                        (
                            "client"
                            if self.turns and self.turns[-1].speaker == "counselor"
                            and self._resume_ai_turn_id != turn_id
                            else "counselor"
                        ),
                    )
                prospective_closing_instruction = self._closing_instruction_for_turn(
                    turn_id,
                    started_monotonic=started_monotonic,
                    mark_started=False,
                )
                self._set_generation_prompt_context_for_turn(
                    turn_id=turn_id,
                    current_objective=self._current_objective_for_turn(
                        turn_id=turn_id,
                        closing_instruction=prospective_closing_instruction,
                    ),
                )
                predicted_speaker = self.controller.speaker_for_turn(turn_id)
                predicted_response_target = _fallback_response_target(
                    config=self.config,
                    speaker=predicted_speaker,
                    previous_speaker=self.turns[-1].speaker if self.turns else None,
                )
                predicted_instruction = _combine_additional_instructions(
                    prospective_closing_instruction,
                    _same_speaker_continuation_instruction(
                        self.config,
                        speaker=predicted_speaker,
                        previous_speaker=(
                            self.turns[-1].speaker if self.turns else None
                        ),
                    ),
                    _human_counselor_initial_client_response_instruction(
                        self.config,
                        speaker=predicted_speaker,
                        turns=self.turns,
                    ),
                    _response_target_instruction(
                        self.config,
                        speaker=predicted_speaker,
                        target=predicted_response_target,
                    ),
                )
                self._prefetch_prompt_director_instruction(
                    turn_id=turn_id,
                    speaker=predicted_speaker,
                    input_transcript=next_input,
                    current_objective=self._current_objective_for_turn(
                        turn_id=turn_id,
                        closing_instruction=prospective_closing_instruction,
                    ),
                    response_target=predicted_response_target,
                    existing_instruction=predicted_instruction,
                )
                timing_signal_tracker = turn_taking_trackers.pop(turn_id, None)
                floor_provisionals = await self._start_floor_prediction_provisional_generations(
                    turn_id=turn_id,
                    input_transcript=next_input,
                    previous_speaker=self.turns[-1].speaker if self.turns else None,
                    started_monotonic=started_monotonic,
                )
                speaker_selection = await self._select_speaker_for_turn(
                    turn_id,
                    input_transcript=next_input,
                    previous_speaker=self.turns[-1].speaker if self.turns else None,
                    timing_signal_tracker=timing_signal_tracker,
                )
                current_speaker = speaker_selection.speaker
                self._active_speaker_id = current_speaker
                response_target = _response_target_from_selection(
                    speaker_selection,
                    config=self.config,
                    speaker=current_speaker,
                    previous_speaker=self.turns[-1].speaker if self.turns else None,
                )
                next_turn_target_hint = _previous_turn_target_hint_from_selection(
                    speaker_selection,
                    config=self.config,
                    speaker=current_speaker,
                )
                next_turn_target_speaker = _previous_turn_target_speaker_from_selection(
                    speaker_selection,
                    config=self.config,
                    speaker=current_speaker,
                )
                # Make the hint visible to any next-turn timing signal collected
                # while the current turn is still streaming.
                self._previous_turn_target_hint = next_turn_target_hint
                self._previous_turn_target_speaker = next_turn_target_speaker
                if self._speaker_is_human(current_speaker):
                    for floor_provisional in floor_provisionals:
                        await self._discard_provisional_generation(
                            floor_provisional,
                            reason="human_speaker_selected",
                            final_transcript=next_input,
                        )
                    floor_provisionals = []
                    human_turn_taking_tracker = self._turn_taking_signal_tracker_for_next_turn(
                        target_turn_id=turn_id + 1,
                        previous_speaker=current_speaker,
                        started_monotonic=started_monotonic,
                    )
                    try:
                        turn = await self._await_human_turn(
                            turn_id=turn_id,
                            speaker=current_speaker,
                            streaming_transcript_observer=(
                                human_turn_taking_tracker.observe
                                if human_turn_taking_tracker is not None
                                else None
                            ),
                        )
                    except ResumeAfterEmptyBargeIn as resume:
                        turn_id = resume.turn_id
                        continue
                    if human_turn_taking_tracker is not None:
                        turn_taking_trackers[turn_id + 1] = human_turn_taking_tracker
                    if self.turns:
                        await self._log_inter_turn_latency(self.turns[-1], turn)
                    self.turns.append(turn)
                    next_input = turn.stt_final_transcript or ""
                    await self._maybe_schedule_session_summary_update(
                        current_objective=self._current_objective_for_turn(
                            turn_id=turn.turn_id + 1,
                            closing_instruction=None,
                        )
                    )
                    turn_id = turn.turn_id + 1
                    continue
                provisional_for_turn = None
                if floor_provisionals:
                    for floor_provisional in floor_provisionals:
                        if (
                            floor_provisional.speaker == current_speaker
                            and provisional_for_turn is None
                        ):
                            skip_reason = _provisional_adoption_skip_reason(
                                floor_provisional,
                                response_target=response_target,
                            )
                            if skip_reason is None:
                                provisional_for_turn = floor_provisional
                                continue
                            await self._discard_provisional_generation(
                                floor_provisional,
                                reason=skip_reason,
                                final_transcript=next_input,
                            )
                            continue
                        await self._discard_provisional_generation(
                            floor_provisional,
                            reason="floor_prediction_mismatch",
                            final_transcript=next_input,
                        )
                    floor_provisionals = []
                if pending_provisional is not None and pending_provisional.turn_id == turn_id:
                    provisional_for_turn = pending_provisional
                    pending_provisional = None
                partial_transcript_parts: list[str] = []

                async def observe_stt_transcript(transcript: TranscriptEvent) -> None:
                    nonlocal active_provisional
                    if not self.config.stt_partial_prefetch:
                        return
                    if not self._generation_resume_event.is_set():
                        return
                    if not self._should_allow_next_turn_prefetch(
                        turn_id,
                        started_monotonic=started_monotonic,
                    ):
                        return
                    if transcript.transcript_type != "partial":
                        return
                    partial_transcript_parts.append(transcript.text)
                    if active_provisional is not None:
                        return
                    provisional_input = "".join(partial_transcript_parts).strip()
                    if (
                        len(_normalize_transcript_for_prefetch(provisional_input))
                        < self.config.stt_partial_prefetch_min_chars
                    ):
                        return
                    next_turn_id = turn_id + 1
                    next_speaker = self.controller.speaker_for_turn(next_turn_id)
                    if self._speaker_is_human(next_speaker):
                        return
                    active_provisional = await self._start_provisional_generation(
                        turn_id=next_turn_id,
                        speaker=next_speaker,
                        input_transcript=provisional_input,
                        source_turn_id=turn_id,
                        same_speaker_instruction=_same_speaker_continuation_instruction(
                            self.config,
                            speaker=next_speaker,
                            previous_speaker=current_speaker,
                        ),
                    )

                observer = (
                    observe_stt_transcript
                    if self.config.stt_partial_prefetch
                    and self._should_allow_next_turn_prefetch(
                        turn_id,
                        started_monotonic=started_monotonic,
                    )
                    else None
                )
                turn_taking_tracker = self._turn_taking_signal_tracker_for_next_turn(
                    target_turn_id=turn_id + 1,
                    previous_speaker=current_speaker,
                    started_monotonic=started_monotonic,
                )
                previous_closing_turn_id = self._closing_started_turn_id
                closing_instruction = self._closing_instruction_for_turn(
                    turn_id,
                    started_monotonic=started_monotonic,
                    mark_started=True,
                )
                if (
                    previous_closing_turn_id is None
                    and self._closing_started_turn_id is not None
                ):
                    await self._log_closing_started(turn_id, started_monotonic)
                self._set_generation_prompt_context_for_turn(
                    turn_id=turn_id,
                    current_objective=self._current_objective_for_turn(
                        turn_id=turn_id,
                        closing_instruction=closing_instruction,
                    ),
                    response_target=response_target,
                )
                response_target_instruction = _response_target_instruction(
                    self.config,
                    speaker=current_speaker,
                    target=response_target,
                )
                same_speaker_instruction = _same_speaker_continuation_instruction(
                    self.config,
                    speaker=current_speaker,
                    previous_speaker=self.turns[-1].speaker if self.turns else None,
                )
                initial_client_response_instruction = (
                    _human_counselor_initial_client_response_instruction(
                        self.config,
                        speaker=current_speaker,
                        turns=self.turns,
                    )
                )
                turn_additional_instruction = _combine_additional_instructions(
                    closing_instruction,
                    same_speaker_instruction,
                    initial_client_response_instruction,
                    response_target_instruction,
                    (
                        EMPTY_BARGE_IN_CONTINUATION_INSTRUCTION
                        if self._resume_ai_turn_id == turn_id else None
                    ),
                    (
                        AI_COUNSELOR_HUMAN_CLIENT_OPENING_INSTRUCTION
                        if self.config.interaction_mode
                        is InteractionMode.AI_COUNSELOR_HUMAN_CLIENT
                        and not self.turns
                        and current_speaker == "counselor"
                        else None
                    ),
                )
                turn_additional_instruction = (
                    await self._add_prompt_director_instruction(
                        turn_id=turn_id,
                        speaker=current_speaker,
                        input_transcript=next_input,
                        current_objective=self._current_objective_for_turn(
                            turn_id=turn_id,
                            closing_instruction=closing_instruction,
                        ),
                        response_target=response_target,
                        existing_instruction=turn_additional_instruction,
                    )
                )
                await self._apply_turn_start_delay(
                    turn_id=turn_id,
                    selection=speaker_selection,
                )
                realtime_human_audio = self._pending_realtime_human_audio.get(
                    turn_id - 1
                )
                post_barge_in_recovery_turn = (
                    self._post_barge_in_recovery_turns_remaining > 0
                    or self._barge_in_human_turn_id is not None
                )
                running_turn_id = turn_id
                self._running_ai_turn_id = running_turn_id
                self._running_ai_speaker = current_speaker
                try:
                    if (
                        realtime_human_audio is not None
                        and realtime_human_audio.target_speaker == current_speaker
                    ):
                        self._pending_realtime_human_audio.pop(turn_id - 1, None)
                        source_human_turn = (
                            self.turns[-1]
                            if self.turns
                            and self.turns[-1].turn_id == realtime_human_audio.turn_id
                            else None
                        )

                        async def observe_realtime_human_transcript(
                            transcript: TranscriptEvent,
                        ) -> None:
                            if (
                                source_human_turn is not None
                                and transcript.transcript_type == "human_final"
                            ):
                                source_human_turn.input_transcript = transcript.text
                                source_human_turn.stt_final_transcript = transcript.text
                                source_human_turn.stt_final_monotonic = time.monotonic()

                        turn = await self.controller.run_realtime_audio_input_turn(
                            turn_id=turn_id,
                            speaker=current_speaker,
                            input_audio=realtime_human_audio.request.audio_bytes,
                            source_human_turn_id=realtime_human_audio.turn_id,
                            source_human_speaker=realtime_human_audio.speaker,
                            human_recipient_ids=realtime_human_audio.request.recipient_ids,
                            source_audio_ref=realtime_human_audio.source_audio_ref,
                            recording_mode=realtime_human_audio.request.recording_mode,
                            client_message_id=(
                                realtime_human_audio.request.client_message_id
                            ),
                            audio_bus=self.audio_bus,
                            logger=self.logger,
                            generated_transcript_observer=(
                                turn_taking_tracker.observe
                                if turn_taking_tracker is not None
                                else None
                            ),
                            human_input_transcript_observer=(
                                observe_realtime_human_transcript
                            ),
                            additional_instruction=turn_additional_instruction,
                        )
                    else:
                        turn = await self.controller.run_turn(
                            turn_id=turn_id,
                            input_transcript=next_input,
                            audio_bus=self.audio_bus,
                            logger=self.logger,
                            provisional_generation=provisional_for_turn,
                            stt_transcript_observer=observer,
                            generated_transcript_observer=(
                                turn_taking_tracker.observe
                                if turn_taking_tracker is not None
                                else None
                            ),
                            additional_instruction=turn_additional_instruction,
                            generation_interrupt_requested=(
                                self._barge_in_generation_interrupt_requested
                            ),
                            publish_audio_without_transcript=(
                                not post_barge_in_recovery_turn
                            ),
                            suppress_audio_without_transcript=(
                                lambda: self._barge_in_human_turn_id is not None
                            ),
                        )
                except GenerationInterruptedForHumanInput:
                    if self._barge_in_human_turn_id is None:
                        raise
                    if active_provisional is not None:
                        await self._discard_provisional_generation(
                            active_provisional,
                            reason="human_barge_in_interrupted_generation",
                            final_transcript=None,
                        )
                        active_provisional = None
                    if pending_provisional is not None:
                        await self._discard_provisional_generation(
                            pending_provisional,
                            reason="human_barge_in_interrupted_generation",
                            final_transcript=None,
                        )
                        pending_provisional = None
                    turn_id = self._barge_in_human_turn_id
                    continue
                finally:
                    if self._running_ai_turn_id == running_turn_id:
                        self._running_ai_turn_id = None
                        self._running_ai_speaker = None
                if (
                    self.config.interaction_mode
                    is InteractionMode.AI_COUNSELOR_HUMAN_CLIENT
                    and self._ai_turn_is_empty(turn)
                ):
                    raise RuntimeError("AI counselor returned an empty response")
                if (
                    post_barge_in_recovery_turn
                    and self._ai_turn_is_empty(turn)
                ):
                    await self._log_empty_ai_turn_skipped_after_interrupt(
                        turn=turn,
                        recovery_turns_remaining=(
                            self._post_barge_in_recovery_turns_remaining
                        ),
                    )
                    if self._post_barge_in_recovery_turns_remaining > 0:
                        self._post_barge_in_recovery_turns_remaining -= 1
                    if self._barge_in_human_turn_id is not None:
                        turn_id = self._barge_in_human_turn_id
                    else:
                        turn_id += 1
                    continue
                if self._post_barge_in_recovery_turns_remaining > 0:
                    self._post_barge_in_recovery_turns_remaining -= 1
                if turn_taking_tracker is not None:
                    turn_taking_trackers[turn_id + 1] = turn_taking_tracker
                if self.turns:
                    await self._log_inter_turn_latency(self.turns[-1], turn)
                self.turns.append(turn)
                if self._resume_ai_turn_id == turn.turn_id:
                    self._resume_ai_turn_id = None
                await self._audit_prompt_director_output(turn)
                self._previous_turn_target_hint = next_turn_target_hint
                self._previous_turn_target_speaker = next_turn_target_speaker
                next_input = turn.generated_text or turn.stt_final_transcript or ""
                await self._maybe_schedule_session_summary_update(
                    current_objective=self._current_objective_for_turn(
                        turn_id=turn_id + 1,
                        closing_instruction=None,
                    )
                )
                if active_provisional is not None:
                    if _provisional_input_matches_final(
                        active_provisional.input_transcript,
                        next_input,
                    ):
                        pending_provisional = active_provisional
                    else:
                        await self._discard_provisional_generation(
                            active_provisional,
                            reason="final_transcript_mismatch",
                            final_transcript=next_input,
                        )
                    active_provisional = None
                turn_id += 1
            for provisional in (
                pending_provisional,
                active_provisional,
                *floor_provisionals,
            ):
                if provisional is not None:
                    await self._discard_provisional_generation(
                        provisional,
                        reason="runtime_completed",
                        final_transcript=next_input,
                    )
            await self._cancel_turn_taking_trackers(turn_taking_trackers.values())
            await self._finish_session_summary_task_if_ready()
            self._phase = RuntimePhase.COMPLETED
            return self.turns
        except HumanInputWaitExpired:
            self._phase = RuntimePhase.COMPLETED
            self._human_input_state = None
            return self.turns
        except asyncio.CancelledError:
            self._phase = RuntimePhase.STOPPED
            raise
        except Exception as exc:
            self._phase = RuntimePhase.ERROR
            await self._log_runtime_error(
                exc,
                turn_id=current_turn_id,
                speaker=current_speaker,
            )
            for provisional in (
                pending_provisional,
                active_provisional,
                *floor_provisionals,
            ):
                if provisional is not None:
                    await self._discard_provisional_generation(
                        provisional,
                        reason="runtime_error",
                        final_transcript=None,
                    )
            await self._cancel_turn_taking_trackers(turn_taking_trackers.values())
            raise
        finally:
            await self.abort_pending_human_audio_streams()
            await self._cancel_session_summary_task()
            await self._cancel_prompt_director_tasks()
            await self.audio_bus.close()
            await logger_consumer
            close_stt = getattr(self.controller.stt, "close", None)
            if close_stt is not None:
                result = close_stt()
                if inspect.isawaitable(result):
                    await result
            await _close_runtime_agents(self.controller.agents)
            await self.logger.close()

    async def _log_resolved_audio_settings(self) -> None:
        if self.ai_route_manifest:
            await self.logger.log_event(
                RuntimeEvent(
                    session_id=self.config.session_id,
                    event_type="ai_routes_resolved",
                    monotonic_time=time.monotonic(),
                    details={"routes": self.ai_route_manifest},
                )
            )
        speakers: list[dict[str, object]] = []
        for speaker_id, participant in self.config.participants.items():
            agent = self.controller.agents.get(speaker_id)
            agent_config = getattr(agent, "config", None)
            voice = getattr(agent_config, "voice", None)
            if not isinstance(voice, str) or not voice.strip():
                voice = participant.voice
            output_speed = getattr(agent_config, "output_speed", None)
            if not isinstance(output_speed, (int, float)) or isinstance(
                output_speed, bool
            ):
                output_speed = participant.realtime_output_speed
            speakers.append(
                {
                    "speaker_id": speaker_id,
                    "display_name": participant.display_name,
                    "role": participant.role,
                    "actor_kind": participant.actor_kind.value,
                    "voice": voice,
                    "realtime_output_speed": output_speed,
                    "audio_gain": float(
                        self.config.speaker_audio_gains.get(speaker_id, 1.0)
                    ),
                }
            )
        await self.logger.log_event(
            RuntimeEvent(
                session_id=self.config.session_id,
                event_type="runtime_audio_settings_resolved",
                monotonic_time=time.monotonic(),
                details={
                    "sample_rate": self.config.sample_rate,
                    "interaction_mode": self.config.interaction_mode.value,
                    "participant_mode": self.config.participant_mode.value,
                    "sample_width_bits": self.config.sample_width_bits,
                    "channels": self.config.channels,
                    "speakers": speakers,
                },
            )
        )

    def _closing_pending_reply_speakers(self) -> tuple[str, ...]:
        if self._closing_assessed_turn_id is None:
            return ()
        replied = {
            turn.speaker
            for turn in self.turns
            if turn.turn_id > self._closing_assessed_turn_id
            and (turn.generated_text or turn.stt_final_transcript)
        }
        return tuple(
            speaker
            for speaker in self._closing_reply_speakers
            if speaker not in replied
        )

    def _closing_next_speaker(self) -> str | None:
        pending = self._closing_pending_reply_speakers()
        if pending:
            return pending[0]
        if (
            self._closing_reply_speakers
            and self.turns
            and _participant_role(self.config, self.turns[-1].speaker) != COUNSELOR_ROLE
        ):
            return self._closing_reply_counselor
        return None

    async def _assess_closing_before_stop(self, turn_id: int) -> None:
        if (
            self.prompt_director is None
            or self._closing_count_started_turn_id is None
            or (
                self._early_closing_reason is None
                and turn_id
                < self._closing_count_started_turn_id
                + self.config.force_stop_after_closing_turns
            )
            or not self.turns
        ):
            return
        last_turn = self.turns[-1]
        if (
            _participant_role(self.config, last_turn.speaker) != COUNSELOR_ROLE
            or self._closing_assessed_turn_id == last_turn.turn_id
        ):
            return
        request = ClosingAssessmentRequest(
            counselor_speaker_id=last_turn.speaker,
            client_display_names={
                participant.speaker_id: participant.display_name
                for participant in self.config.participants.values()
                if participant.role == CLIENT_ROLE
            },
            public_history=self._prompt_director_public_history(input_transcript=""),
            session_summary=self._session_summary_state.text,
        )
        started_at = time.monotonic()
        await self.logger.log_event(
            RuntimeEvent(
                session_id=self.config.session_id,
                event_type="closing_response_assessment_started",
                turn_id=last_turn.turn_id,
                speaker=last_turn.speaker,
            )
        )
        result = await self.prompt_director.assess_closing(request)
        self._closing_assessed_turn_id = last_turn.turn_id
        self._closing_reply_speakers = tuple(result.reply_speaker_ids)
        self._closing_reply_counselor = last_turn.speaker
        await self.logger.log_event(
            RuntimeEvent(
                session_id=self.config.session_id,
                event_type="closing_response_assessed",
                turn_id=last_turn.turn_id,
                speaker=last_turn.speaker,
                details={
                    "reply_speaker_ids": result.reply_speaker_ids,
                    "reason": result.reason,
                    "latency_ms": _elapsed_ms(started_at, time.monotonic()),
                },
            )
        )

    def _closing_hard_stop_turn_id(self) -> int:
        assert self._closing_count_started_turn_id is not None
        return (
            self._closing_count_started_turn_id
            + self.config.force_stop_after_closing_turns
            + max(
                len(self.config.fixed_speaker_sequence),
                len(self.config.participants),
                1,
            )
        )

    def _should_start_turn(
        self,
        turn_id: int,
        *,
        started_monotonic: float,
    ) -> bool:
        if self._closing_count_started_turn_id is not None:
            forced_stop_turn_id = (
                self._closing_count_started_turn_id
                + self.config.force_stop_after_closing_turns
            )
            if (
                self._early_closing_reason is not None
                and self.turns
                and self._closing_assessed_turn_id == self.turns[-1].turn_id
                and not self._closing_pending_reply_speakers()
            ):
                return False
            if turn_id < forced_stop_turn_id:
                return True
            if turn_id >= self._closing_hard_stop_turn_id():
                return False
            last_turn = self.turns[-1] if self.turns else None
            if self.prompt_director is not None and last_turn is not None:
                if self._closing_pending_reply_speakers():
                    return True
                if _participant_role(self.config, last_turn.speaker) == COUNSELOR_ROLE:
                    # Speculation can start before the actual speech has been
                    # assessed. The main loop always awaits that assessment.
                    return self._closing_assessed_turn_id != last_turn.turn_id
            return (
                last_turn is not None
                and _participant_role(self.config, last_turn.speaker) != COUNSELOR_ROLE
            )
        if self._closing_started_turn_id is not None:
            # Closing normally sets the count anchor on the same counselor turn.
            # Keep this guard for defensive continuity if that invariant changes.
            return turn_id <= self.config.max_turns
        if self.config.closing_start_elapsed_seconds is not None:
            elapsed_seconds = time.monotonic() - started_monotonic
            if elapsed_seconds >= self.config.closing_start_elapsed_seconds:
                closing_start_grace_turns = max(
                    len(self.config.fixed_speaker_sequence),
                    len(self.config.participants),
                    1,
                )
                return turn_id <= self.config.max_turns + closing_start_grace_turns
            if self.config.stop_condition is RuntimeStopCondition.TURNS:
                return turn_id <= self.config.max_turns
        if self.config.stop_condition is RuntimeStopCondition.TURNS:
            return turn_id <= self.config.max_turns
        max_elapsed_seconds = self.config.max_elapsed_seconds
        if max_elapsed_seconds is None:
            return True
        return (time.monotonic() - started_monotonic) < max_elapsed_seconds

    async def _select_speaker_for_turn(
        self,
        turn_id: int,
        *,
        input_transcript: str,
        previous_speaker: str | None,
        timing_signal_tracker: TurnTakingSignalTracker | None = None,
    ) -> SpeakerSelection:
        if self.config.interaction_mode is InteractionMode.AI_COUNSELOR_HUMAN_CLIENT:
            speaker = (
                "client" if previous_speaker == "counselor"
                and self._resume_ai_turn_id != turn_id else "counselor"
            )
            self.controller.set_speaker_for_turn(turn_id, speaker)
            return SpeakerSelection(speaker)
        realtime_audio_target = self._realtime_human_audio_target_for_turn(turn_id)
        if realtime_audio_target is not None:
            self.controller.set_speaker_for_turn(turn_id, realtime_audio_target)
            return SpeakerSelection(realtime_audio_target)
        closing_next_speaker = self._closing_next_speaker()
        if closing_next_speaker is not None:
            self.controller.set_speaker_for_turn(turn_id, closing_next_speaker)
            return SpeakerSelection(closing_next_speaker)
        initial_human_mode_client = (
            self._human_counselor_initial_client_speaker(previous_speaker)
        )
        if initial_human_mode_client is not None:
            self.controller.set_speaker_for_turn(turn_id, initial_human_mode_client)
            return SpeakerSelection(initial_human_mode_client)
        scheduled_human_speaker = self._scheduled_human_counselor_speaker_for_turn(
            turn_id,
            previous_speaker=previous_speaker,
        )
        if scheduled_human_speaker is not None:
            return SpeakerSelection(scheduled_human_speaker)
        if self.config.speaker_selection_policy == TURN_BOUNDARY_TIMING_POLICY:
            return await self._select_turn_boundary_timing_speaker(
                turn_id,
                input_transcript=input_transcript,
                previous_speaker=previous_speaker,
                timing_signal_tracker=timing_signal_tracker,
            )
        if self.config.speaker_selection_policy != DISTRIBUTED_TIMING_POLICY:
            fallback_speaker = self.controller.speaker_for_turn(turn_id)
            fallback_selection = (
                self._human_counselor_followup_client_fallback_selection(
                    turn_id=turn_id,
                    previous_speaker=previous_speaker,
                    base_speaker=fallback_speaker,
                    decisions=(),
                )
            )
            if fallback_selection is not None:
                fallback_speaker = fallback_selection.speaker_id
                self.controller.set_speaker_for_turn(turn_id, fallback_speaker)
            return SpeakerSelection(fallback_speaker)

        decisions = await self._collect_timing_decisions(
            turn_id=turn_id,
            input_transcript=input_transcript,
            previous_speaker=previous_speaker,
            timeout_ms=self.config.turn_taking_decision_timeout_ms,
            wait_for_pending=True,
        )
        explicitly_addressed_priority = _explicitly_addressed_priority_decision(
            decisions,
            previous_speaker=previous_speaker,
            config=self.config,
        )
        if explicitly_addressed_priority is not None:
            result = FloorMediatorResult(
                result_type=FloorMediatorResultType.GRANT,
                agent_id=explicitly_addressed_priority.agent_id,
                reason="explicitly_addressed_participant",
                conflict_agent_ids=_priority_conflict_agent_ids(
                    decisions,
                    explicitly_addressed_priority.agent_id,
                ),
            )
            self.controller.set_speaker_for_turn(
                turn_id,
                explicitly_addressed_priority.agent_id,
            )
            await self._log_floor_result(
                turn_id,
                result=result,
                decisions=decisions,
                fallback_speaker=None,
            )
            return SpeakerSelection(
                explicitly_addressed_priority.agent_id,
                explicitly_addressed_priority,
            )
        mediator = FloorMediator()
        result = mediator.decide(
            decisions,
            detect_overlap=self._can_resolve_overlap(decisions),
        )
        if result.result_type is not FloorMediatorResultType.OVERLAP:
            client_reply_opportunity = _client_reply_opportunity_decision(
                decisions,
                turns=self.turns,
                previous_speaker=previous_speaker,
                config=self.config,
            )
            if client_reply_opportunity is not None:
                result = FloorMediatorResult(
                    result_type=FloorMediatorResultType.GRANT,
                    agent_id=client_reply_opportunity.agent_id,
                    reason="client_reply_opportunity_after_client",
                    conflict_agent_ids=_requesting_agent_ids(decisions),
                )
                self.controller.set_speaker_for_turn(
                    turn_id,
                    client_reply_opportunity.agent_id,
                )
                await self._log_floor_result(
                    turn_id,
                    result=result,
                    decisions=decisions,
                    fallback_speaker=None,
                )
                return SpeakerSelection(
                    client_reply_opportunity.agent_id,
                    client_reply_opportunity,
                    _fallback_response_target(
                        config=self.config,
                        speaker=client_reply_opportunity.agent_id,
                        previous_speaker=previous_speaker,
                    ),
                )
        if result.result_type is FloorMediatorResultType.OVERLAP:
            await self._log_floor_result(
                turn_id,
                result=result,
                decisions=decisions,
                fallback_speaker=None,
            )
            if self.config.overlap_grace_ms > 0:
                await asyncio.sleep(self.config.overlap_grace_ms / 1000)
            overlap_resolution_timed_out = False
            try:
                overlap_decisions = await asyncio.wait_for(
                    self._collect_overlap_decisions(
                        turn_id=turn_id,
                        input_transcript=input_transcript,
                        previous_speaker=previous_speaker,
                        conflict_agent_ids=result.conflict_agent_ids,
                    ),
                    timeout=self.config.unresolved_overlap_limit_ms / 1000,
                )
            except TimeoutError:
                overlap_resolution_timed_out = True
                overlap_decisions = []
            resolved_result = mediator.resolve_overlap(
                overlap_decisions,
                active_speaker_id=previous_speaker,
                conflict_agent_ids=result.conflict_agent_ids,
            )
            if (
                resolved_result.result_type is FloorMediatorResultType.GRANT
                and resolved_result.agent_id
            ):
                self.controller.set_speaker_for_turn(turn_id, resolved_result.agent_id)
                await self._log_floor_result(
                    turn_id,
                    result=resolved_result,
                    decisions=decisions,
                    fallback_speaker=None,
                    overlap_decisions=overlap_decisions,
                    overlap_resolution_timed_out=overlap_resolution_timed_out,
                )
                return SpeakerSelection(
                    resolved_result.agent_id,
                    _timing_decision_for_agent(decisions, resolved_result.agent_id),
                )
            await self._log_floor_result(
                turn_id,
                result=resolved_result,
                decisions=decisions,
                fallback_speaker=None,
                overlap_decisions=overlap_decisions,
                overlap_resolution_timed_out=overlap_resolution_timed_out,
            )

            fallback_speaker = self.controller.speaker_for_turn(turn_id)
            fallback_selection = (
                self._human_counselor_followup_client_fallback_selection(
                    turn_id=turn_id,
                    previous_speaker=previous_speaker,
                    base_speaker=fallback_speaker,
                    decisions=decisions,
                )
            )
            if fallback_selection is not None:
                fallback_speaker = fallback_selection.speaker_id
            self.controller.set_speaker_for_turn(turn_id, fallback_speaker)
            await self._log_floor_result(
                turn_id,
                result=FloorMediatorResult(
                    result_type=FloorMediatorResultType.WAIT,
                    agent_id=None,
                    reason="overlap_unresolved_fallback",
                    conflict_agent_ids=result.conflict_agent_ids,
                    yielded_agent_ids=resolved_result.yielded_agent_ids,
                ),
                decisions=decisions,
                fallback_speaker=fallback_speaker,
                fallback_selection=fallback_selection,
                overlap_decisions=overlap_decisions,
                overlap_resolution_timed_out=overlap_resolution_timed_out,
            )
            return SpeakerSelection(
                fallback_speaker,
                _timing_decision_for_agent(decisions, fallback_speaker),
            )

        if result.result_type is FloorMediatorResultType.GRANT and result.agent_id:
            self.controller.set_speaker_for_turn(turn_id, result.agent_id)
            await self._log_floor_result(
                turn_id,
                result=result,
                decisions=decisions,
                fallback_speaker=None,
            )
            return SpeakerSelection(
                result.agent_id,
                _timing_decision_for_agent(decisions, result.agent_id),
            )

        human_wait_selection = (
            await self._human_counselor_wait_selection_for_no_request(
                turn_id=turn_id,
                result=result,
                decisions=decisions,
            )
        )
        if human_wait_selection is not None:
            return human_wait_selection

        fallback_selection = self._fallback_speaker_for_timing_result(
            turn_id=turn_id,
            previous_speaker=previous_speaker,
            result=result,
            decisions=decisions,
        )
        fallback_speaker = fallback_selection.speaker_id
        self.controller.set_speaker_for_turn(turn_id, fallback_speaker)
        await self._log_floor_result(
            turn_id,
            result=result,
            decisions=decisions,
            fallback_speaker=fallback_speaker,
            fallback_selection=fallback_selection,
        )
        return SpeakerSelection(
            fallback_speaker,
            _timing_decision_for_agent(decisions, fallback_speaker),
            _fallback_response_target(
                config=self.config,
                speaker=fallback_speaker,
                previous_speaker=previous_speaker,
            ),
        )

    async def _select_turn_boundary_timing_speaker(
        self,
        turn_id: int,
        *,
        input_transcript: str,
        previous_speaker: str | None,
        timing_signal_tracker: TurnTakingSignalTracker | None = None,
    ) -> SpeakerSelection:
        decisions: list[TimingDecision] = []
        signal_text = ""
        if timing_signal_tracker is not None:
            decisions, signal_text = await timing_signal_tracker.decisions()
        if decisions and (
            _should_refresh_timing_signal_decisions(
                signal_text=signal_text,
                final_transcript=input_transcript,
            )
            or any(
                decision.reason_code == "timing_decision_timeout"
                for decision in decisions
            )
        ):
            decisions = await self._collect_timing_decisions(
                turn_id=turn_id,
                input_transcript=input_transcript,
                previous_speaker=previous_speaker,
                timeout_ms=self.config.turn_taking_decision_timeout_ms,
                wait_for_pending=True,
            )
        if not decisions:
            decisions = await self._collect_timing_decisions(
                turn_id=turn_id,
                input_transcript=input_transcript,
                previous_speaker=previous_speaker,
                timeout_ms=self.config.turn_taking_decision_timeout_ms,
                wait_for_pending=True,
            )
        allow_client_reply_auto_priority = _allow_client_reply_auto_priority(
            turns=self.turns,
            previous_speaker=previous_speaker,
            config=self.config,
        )
        if allow_client_reply_auto_priority:
            priority_decision = _client_reply_priority_decision(
                decisions,
                turns=self.turns,
                previous_speaker=previous_speaker,
                config=self.config,
            )
            if priority_decision is not None:
                result = FloorMediatorResult(
                    result_type=FloorMediatorResultType.GRANT,
                    agent_id=priority_decision.agent_id,
                    reason="client_reply_to_previous_client",
                    conflict_agent_ids=_requesting_agent_ids(decisions),
                )
                self.controller.set_speaker_for_turn(
                    turn_id,
                    priority_decision.agent_id,
                )
                await self._log_floor_result(
                    turn_id,
                    result=result,
                    decisions=decisions,
                    fallback_speaker=None,
                )
                return SpeakerSelection(priority_decision.agent_id, priority_decision)
            client_reply_opportunity = _client_reply_opportunity_decision(
                decisions,
                turns=self.turns,
                previous_speaker=previous_speaker,
                config=self.config,
            )
            if client_reply_opportunity is not None:
                result = FloorMediatorResult(
                    result_type=FloorMediatorResultType.GRANT,
                    agent_id=client_reply_opportunity.agent_id,
                    reason="client_reply_opportunity_after_client",
                    conflict_agent_ids=_requesting_agent_ids(decisions),
                )
                self.controller.set_speaker_for_turn(
                    turn_id,
                    client_reply_opportunity.agent_id,
                )
                await self._log_floor_result(
                    turn_id,
                    result=result,
                    decisions=decisions,
                    fallback_speaker=None,
                )
                return SpeakerSelection(
                    client_reply_opportunity.agent_id,
                    client_reply_opportunity,
                    _fallback_response_target(
                        config=self.config,
                        speaker=client_reply_opportunity.agent_id,
                        previous_speaker=previous_speaker,
                    ),
                )
            contextual_reply = _client_contextual_reply_opportunity_decision(
                decisions,
                previous_speaker=previous_speaker,
                config=self.config,
            )
            if contextual_reply is not None:
                result = FloorMediatorResult(
                    result_type=FloorMediatorResultType.GRANT,
                    agent_id=contextual_reply.agent_id,
                    reason="client_contextual_reply_after_client",
                    conflict_agent_ids=_requesting_agent_ids(decisions),
                )
                self.controller.set_speaker_for_turn(
                    turn_id,
                    contextual_reply.agent_id,
                )
                await self._log_floor_result(
                    turn_id,
                    result=result,
                    decisions=decisions,
                    fallback_speaker=None,
                )
                return SpeakerSelection(
                    contextual_reply.agent_id,
                    contextual_reply,
                )
        extended_client_reply = _extended_client_reply_after_client_run_decision(
            decisions,
            turns=self.turns,
            previous_speaker=previous_speaker,
            config=self.config,
        )
        if extended_client_reply is not None:
            result = FloorMediatorResult(
                result_type=FloorMediatorResultType.GRANT,
                agent_id=extended_client_reply.agent_id,
                reason="extended_client_reply_after_client_run",
                conflict_agent_ids=_requesting_agent_ids(decisions),
            )
            self.controller.set_speaker_for_turn(
                turn_id,
                extended_client_reply.agent_id,
            )
            await self._log_floor_result(
                turn_id,
                result=result,
                decisions=decisions,
                fallback_speaker=None,
            )
            return SpeakerSelection(
                extended_client_reply.agent_id,
                extended_client_reply,
            )
        explicitly_addressed_priority = _explicitly_addressed_priority_decision(
            decisions,
            previous_speaker=previous_speaker,
            config=self.config,
        )
        if explicitly_addressed_priority is not None:
            result = FloorMediatorResult(
                result_type=FloorMediatorResultType.GRANT,
                agent_id=explicitly_addressed_priority.agent_id,
                reason="explicitly_addressed_participant",
                conflict_agent_ids=_priority_conflict_agent_ids(
                    decisions,
                    explicitly_addressed_priority.agent_id,
                ),
            )
            self.controller.set_speaker_for_turn(
                turn_id,
                explicitly_addressed_priority.agent_id,
            )
            await self._log_floor_result(
                turn_id,
                result=result,
                decisions=decisions,
                fallback_speaker=None,
            )
            return SpeakerSelection(
                explicitly_addressed_priority.agent_id,
                explicitly_addressed_priority,
            )
        decisions = await self._reconsider_turn_boundary_conflicts(
            turn_id=turn_id,
            input_transcript=input_transcript,
            previous_speaker=previous_speaker,
            decisions=decisions,
        )
        counselor_reentry_decision = _counselor_reentry_after_client_run_decision(
            decisions,
            turns=self.turns,
            previous_speaker=previous_speaker,
            config=self.config,
        )
        if counselor_reentry_decision is not None:
            result = FloorMediatorResult(
                result_type=FloorMediatorResultType.GRANT,
                agent_id=counselor_reentry_decision.agent_id,
                reason="counselor_reentry_after_client_run",
                conflict_agent_ids=_requesting_agent_ids(decisions),
            )
            self.controller.set_speaker_for_turn(
                turn_id,
                counselor_reentry_decision.agent_id,
            )
            await self._log_floor_result(
                turn_id,
                result=result,
                decisions=decisions,
                fallback_speaker=None,
            )
            return SpeakerSelection(
                counselor_reentry_decision.agent_id,
                counselor_reentry_decision,
            )
        result = FloorMediator().decide(decisions, detect_overlap=False)
        if result.result_type is FloorMediatorResultType.GRANT and result.agent_id:
            self.controller.set_speaker_for_turn(turn_id, result.agent_id)
            await self._log_floor_result(
                turn_id,
                result=result,
                decisions=decisions,
                fallback_speaker=None,
            )
            return SpeakerSelection(
                result.agent_id,
                _timing_decision_for_agent(decisions, result.agent_id),
            )

        human_wait_selection = (
            await self._human_counselor_wait_selection_for_no_request(
                turn_id=turn_id,
                result=result,
                decisions=decisions,
            )
        )
        if human_wait_selection is not None:
            return human_wait_selection

        fallback_selection = self._fallback_speaker_for_timing_result(
            turn_id=turn_id,
            previous_speaker=previous_speaker,
            result=result,
            decisions=decisions,
        )
        fallback_speaker = fallback_selection.speaker_id
        self.controller.set_speaker_for_turn(turn_id, fallback_speaker)
        await self._log_floor_result(
            turn_id,
            result=result,
            decisions=decisions,
            fallback_speaker=fallback_speaker,
            fallback_selection=fallback_selection,
        )
        return SpeakerSelection(
            fallback_speaker,
            _timing_decision_for_agent(decisions, fallback_speaker),
            _fallback_response_target(
                config=self.config,
                speaker=fallback_speaker,
                previous_speaker=previous_speaker,
            ),
        )

    async def _collect_timing_decisions(
        self,
        *,
        turn_id: int,
        input_transcript: str,
        previous_speaker: str | None,
        agent_ids: tuple[str, ...] | None = None,
        timeout_ms: int | None = None,
        wait_for_pending: bool = False,
    ) -> list[TimingDecision]:
        async def collect_for_agent(
            speaker_id: str,
            agent: AgentLike | StreamingAgentLike,
        ) -> TimingDecision:
            if speaker_id == previous_speaker:
                return _complete_timing_decision_metadata(
                    TimingDecision(
                        agent_id=speaker_id,
                        action=TimingDecisionAction.WAIT,
                        reason_code="previous_speaker",
                    ),
                    previous_speaker=previous_speaker,
                    config=self.config,
                )
            decide_timing = getattr(agent, "decide_timing", None)
            if decide_timing is None:
                return _complete_timing_decision_metadata(
                    TimingDecision(
                        agent_id=speaker_id,
                        action=TimingDecisionAction.REQUEST_MAIN_FLOOR,
                        reason_code="timing_decision_not_supported",
                    ),
                    previous_speaker=previous_speaker,
                    config=self.config,
                )
            try:
                raw_decision = decide_timing(
                    input_transcript=input_transcript,
                    turn_id=turn_id,
                    previous_speaker=previous_speaker,
                )
                if inspect.isawaitable(raw_decision):
                    raw_decision = await raw_decision
                decision = TimingDecision.validate(raw_decision)
                if decision.agent_id != speaker_id:
                    raise TimingDecisionValidationError(
                        "timing decision agent_id must match the participant speaker_id"
                    )
            except (TimingDecisionValidationError, StreamingLLMError) as exc:
                is_validation_error = isinstance(
                    exc,
                    TimingDecisionValidationError,
                )
                event_type = (
                    "timing_decision_invalid"
                    if is_validation_error
                    else "timing_decision_error"
                )
                await self.logger.log_event(
                    RuntimeEvent(
                        session_id=self.config.session_id,
                        event_type=event_type,
                        turn_id=turn_id,
                        speaker=speaker_id,
                        speaker_id=speaker_id,
                        recipient_ids=self.controller.recipient_ids_for_speaker(
                            speaker_id
                        ),
                        monotonic_time=time.monotonic(),
                        details={
                            "error_type": type(exc).__name__,
                            "message": str(exc),
                            "reason_code": event_type,
                        },
                    )
                )
                return _complete_timing_decision_metadata(
                    TimingDecision(
                        agent_id=speaker_id,
                        action=TimingDecisionAction.WAIT,
                        reason_code=event_type,
                    ),
                    previous_speaker=previous_speaker,
                    config=self.config,
                )
            return _complete_timing_decision_metadata(
                decision,
                previous_speaker=previous_speaker,
                config=self.config,
            )

        if agent_ids is None:
            selected_agents = tuple(self.controller.agents.items())
        else:
            missing_agent_ids = tuple(
                speaker_id
                for speaker_id in agent_ids
                if speaker_id not in self.controller.agents
            )
            if missing_agent_ids:
                raise ValueError(
                    "unknown timing decision agent(s): "
                    + ", ".join(missing_agent_ids)
                )
            selected_agents = tuple(
                (speaker_id, self.controller.agents[speaker_id])
                for speaker_id in agent_ids
            )
        tasks_by_agent_id = {
            speaker_id: asyncio.create_task(collect_for_agent(speaker_id, agent))
            for speaker_id, agent in selected_agents
        }
        timeout_seconds = timeout_ms / 1000 if timeout_ms is not None else None
        try:
            done, pending = await asyncio.wait(
                tasks_by_agent_id.values(),
                timeout=timeout_seconds,
            )
            if pending and wait_for_pending:
                # At a real turn boundary, latency is not a decision to yield.
                # Keep the original requests alive until their decisions arrive.
                await self.logger.log_event(
                    RuntimeEvent(
                        session_id=self.config.session_id,
                        event_type="timing_decision_waiting",
                        turn_id=turn_id,
                        monotonic_time=time.monotonic(),
                        details={
                            "pending_agent_ids": [
                                speaker_id
                                for speaker_id, task in tasks_by_agent_id.items()
                                if task in pending
                            ],
                            "timeout_ms": timeout_ms,
                        },
                    )
                )
                await asyncio.gather(*pending)
                done.update(pending)
        finally:
            # Cancellation of a turn/session must also stop all owned requests.
            for task in tasks_by_agent_id.values():
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks_by_agent_id.values(), return_exceptions=True)

        task_agent_ids = {
            task: speaker_id for speaker_id, task in tasks_by_agent_id.items()
        }
        decision_by_agent_id: dict[str, TimingDecision] = {}
        for task in done:
            speaker_id = task_agent_ids[task]
            decision_by_agent_id[speaker_id] = task.result()

        return [
            decision_by_agent_id.get(
                speaker_id,
                _complete_timing_decision_metadata(
                    TimingDecision(
                        agent_id=speaker_id,
                        action=TimingDecisionAction.WAIT,
                        reason_code="timing_decision_timeout",
                    ),
                    previous_speaker=previous_speaker,
                    config=self.config,
                ),
            )
            for speaker_id, _agent in selected_agents
        ]

    def _realtime_human_audio_target_for_turn(self, turn_id: int) -> str | None:
        pending = self._pending_realtime_human_audio.get(turn_id - 1)
        if pending is None:
            return None
        return pending.target_speaker

    def _fallback_speaker_for_timing_result(
        self,
        *,
        turn_id: int,
        previous_speaker: str | None,
        result: FloorMediatorResult,
        decisions: list[TimingDecision],
    ) -> FallbackSpeakerSelection:
        base_speaker = self.controller.speaker_for_turn(turn_id)
        if result.reason != "no_request":
            return FallbackSpeakerSelection(
                speaker_id=base_speaker,
                base_speaker_id=base_speaker,
                strategy="fixed_sequence",
                candidates=(),
            )
        return _select_contextual_fallback_speaker(
            config=self.config,
            turns=self.turns,
            turn_id=turn_id,
            base_speaker=base_speaker,
            previous_speaker=previous_speaker,
            previous_turn_target_speaker=self._previous_turn_target_speaker,
            decisions=decisions,
        )

    def _human_counselor_speaker_id(self) -> str | None:
        for speaker_id, participant in self.config.participants.items():
            if (
                participant.actor_kind is ActorKind.HUMAN
                and participant.role == COUNSELOR_ROLE
            ):
                return speaker_id
        return None

    def _human_counselor_initial_client_speaker(
        self,
        previous_speaker: str | None,
    ) -> str | None:
        if (
            self.config.interaction_mode
            is not InteractionMode.HUMAN_COUNSELOR_AI_CLIENT
        ):
            return None
        human_counselor = self._human_counselor_speaker_id()
        if human_counselor is None or previous_speaker != human_counselor:
            return None
        if any(
            _participant_role(self.config, turn.speaker) == CLIENT_ROLE
            for turn in self.turns
        ):
            return None
        participant = _initial_client_participant(self.config)
        if participant is None or not participant.initial_transcript.strip():
            return None
        return participant.speaker_id

    def _scheduled_human_counselor_speaker_for_turn(
        self,
        turn_id: int,
        *,
        previous_speaker: str | None,
    ) -> str | None:
        if (
            self.config.interaction_mode
            is not InteractionMode.HUMAN_COUNSELOR_AI_CLIENT
        ):
            return None
        scheduled_speaker = self.controller.speaker_for_turn(turn_id)
        human_counselor = self._human_counselor_speaker_id()
        if (
            scheduled_speaker == human_counselor
            and previous_speaker == human_counselor
        ):
            return None
        if scheduled_speaker == human_counselor:
            return scheduled_speaker
        return None

    def _human_counselor_followup_client_fallback_selection(
        self,
        *,
        turn_id: int,
        previous_speaker: str | None,
        base_speaker: str,
        decisions: Iterable[TimingDecision],
    ) -> FallbackSpeakerSelection | None:
        if (
            self.config.interaction_mode
            is not InteractionMode.HUMAN_COUNSELOR_AI_CLIENT
        ):
            return None
        human_counselor = self._human_counselor_speaker_id()
        if human_counselor is None:
            return None
        if previous_speaker != human_counselor or base_speaker != human_counselor:
            return None
        decisions_tuple = tuple(decisions)
        recent_speakers = tuple(
            turn.speaker for turn in self.turns[-_FALLBACK_RECENT_TURN_LIMIT:]
        )
        candidate_scores = tuple(
            _score_fallback_candidate(
                config=self.config,
                candidate_speaker_id=speaker_id,
                turn_id=turn_id,
                base_speaker=base_speaker,
                previous_speaker=previous_speaker,
                previous_turn_target_speaker=self._previous_turn_target_speaker,
                recent_speakers=recent_speakers,
                decisions=decisions_tuple,
            )
            for speaker_id, participant in self.config.participants.items()
            if participant.role == CLIENT_ROLE
        )
        if not candidate_scores:
            return None
        winner = min(
            candidate_scores,
            key=lambda candidate: (
                -candidate.score,
                candidate.sequence_distance,
                candidate.speaker_id,
            ),
        )
        return FallbackSpeakerSelection(
            speaker_id=winner.speaker_id,
            base_speaker_id=base_speaker,
            strategy="human_counselor_no_consecutive_turns",
            candidates=tuple(
                sorted(
                    candidate_scores,
                    key=lambda candidate: (
                        -candidate.score,
                        candidate.sequence_distance,
                        candidate.speaker_id,
                    ),
                )
            ),
        )

    async def _human_counselor_wait_selection_for_no_request(
        self,
        *,
        turn_id: int,
        result: FloorMediatorResult,
        decisions: list[TimingDecision],
    ) -> SpeakerSelection | None:
        if (
            self.config.interaction_mode
            is not InteractionMode.HUMAN_COUNSELOR_AI_CLIENT
        ):
            return None
        if result.reason != "no_request":
            return None
        if not _human_counselor_clients_deliberately_waited(decisions):
            return None
        if not self._has_client_turn_since_last_human_counselor_turn():
            return None
        human_counselor = self._human_counselor_speaker_id()
        if human_counselor is None:
            return None
        base_speaker = self.controller.speaker_for_turn(turn_id)
        fallback_selection = FallbackSpeakerSelection(
            speaker_id=human_counselor,
            base_speaker_id=base_speaker,
            strategy="human_counselor_after_clients_wait",
            candidates=(),
        )
        self.controller.set_speaker_for_turn(turn_id, human_counselor)
        await self._log_floor_result(
            turn_id,
            result=result,
            decisions=decisions,
            fallback_speaker=human_counselor,
            fallback_selection=fallback_selection,
        )
        return SpeakerSelection(human_counselor)

    def _has_client_turn_since_last_human_counselor_turn(self) -> bool:
        for turn in reversed(self.turns):
            if self._speaker_is_human(turn.speaker):
                return False
            if _participant_role(self.config, turn.speaker) == CLIENT_ROLE:
                return True
        return False

    async def _reconsider_turn_boundary_conflicts(
        self,
        *,
        turn_id: int,
        input_transcript: str,
        previous_speaker: str | None,
        decisions: list[TimingDecision],
    ) -> list[TimingDecision]:
        current_decisions = list(decisions)
        for round_index in range(self.config.turn_taking_max_reconsider_rounds):
            conflict_agent_ids = _requesting_agent_ids(current_decisions)
            if len(conflict_agent_ids) <= 1:
                return current_decisions
            await self._log_floor_negotiation_conflict(
                turn_id=turn_id,
                round_index=round_index + 1,
                conflict_agent_ids=conflict_agent_ids,
                decisions=current_decisions,
            )
            reconsider_input = _turn_taking_conflict_prompt(
                input_transcript=input_transcript,
                conflict_agent_ids=conflict_agent_ids,
                round_index=round_index + 1,
                decisions=current_decisions,
            )
            reconsidered = await self._collect_timing_decisions(
                turn_id=turn_id,
                input_transcript=reconsider_input,
                previous_speaker=previous_speaker,
                agent_ids=conflict_agent_ids,
                timeout_ms=self.config.turn_taking_decision_timeout_ms,
                wait_for_pending=True,
            )
            current_decisions = _replace_timing_decisions(
                current_decisions,
                reconsidered,
            )
        return current_decisions

    def _turn_taking_signal_tracker_for_next_turn(
        self,
        *,
        target_turn_id: int,
        previous_speaker: str | None,
        started_monotonic: float,
    ) -> TurnTakingSignalTracker | None:
        if self.config.speaker_selection_policy != TURN_BOUNDARY_TIMING_POLICY:
            return None
        if not self._should_start_turn(
            target_turn_id,
            started_monotonic=started_monotonic,
        ):
            return None

        source_turn_id = target_turn_id - 1

        async def collect(input_transcript: str) -> list[TimingDecision]:
            return await self._collect_timing_decisions(
                turn_id=target_turn_id,
                input_transcript=input_transcript,
                previous_speaker=previous_speaker,
                timeout_ms=self.config.turn_taking_decision_timeout_ms,
            )

        async def on_collected(
            input_transcript: str,
            decisions: list[TimingDecision],
        ) -> None:
            await self._log_turn_taking_signal_collected(
                turn_id=target_turn_id,
                source_turn_id=source_turn_id,
                previous_speaker=previous_speaker,
                input_transcript=input_transcript,
                decisions=decisions,
            )

        min_chars = self.config.turn_taking_signal_min_chars
        if previous_speaker is not None and self._speaker_is_human(previous_speaker):
            min_chars = min(min_chars, _HUMAN_TURN_TAKING_SIGNAL_MIN_CHARS)

        return TurnTakingSignalTracker(
            turn_id=target_turn_id,
            previous_speaker=previous_speaker,
            min_chars=min_chars,
            collect=collect,
            on_collected=on_collected,
        )

    async def _cancel_turn_taking_trackers(
        self,
        trackers: Iterable[TurnTakingSignalTracker],
    ) -> None:
        tasks = [
            tracker._task
            for tracker in trackers
            if tracker._task is not None and not tracker._task.done()
        ]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _maybe_schedule_session_summary_update(
        self,
        *,
        current_objective: str,
    ) -> None:
        await self._finish_session_summary_task_if_ready()
        if self.session_summary_llm is None or self._session_summary_task is not None:
            return
        request = self._session_summary_retry_request
        is_retry = request is not None
        if request is None:
            if not should_update_session_summary(
                turns=self.turns,
                summary_state=self._session_summary_state,
                recent_turn_limit=self.config.conversation_context_recent_turns,
                trigger_completed_turns=(
                    self.config.session_summary_trigger_completed_turns
                ),
                update_interval_turns=self.config.session_summary_update_interval_turns,
            ):
                return
            request = build_session_summary_request(
                turns=self.turns,
                summary_state=self._session_summary_state,
                current_objective=current_objective,
                recent_turn_limit=self.config.conversation_context_recent_turns,
                participants=self.config.participants,
            )
            if request is None:
                return
        self._last_event_type = "session_summary_update_started"
        await self.logger.log_event(
            RuntimeEvent(
                session_id=self.config.session_id,
                event_type="session_summary_update_started",
                monotonic_time=time.monotonic(),
                details={
                    "turns_to_summarize": len(request.turns_to_summarize),
                    "last_turn_id_to_summarize": request.last_turn_id_to_summarize,
                    "recent_turn_limit": self.config.conversation_context_recent_turns,
                    "is_retry": is_retry,
                },
            )
        )
        self._session_summary_task_request = request
        self._session_summary_task = asyncio.create_task(
            self._run_session_summary_update(request)
        )

    async def _run_session_summary_update(
        self,
        request: SessionSummaryRequest,
    ) -> SessionSummaryUpdate:
        if self.session_summary_llm is None:
            raise RuntimeError("session summary LLM is not configured")
        summary_text = await summarize_session_context(
            llm=self.session_summary_llm,
            request=request,
        )
        return build_session_summary_update(
            request=request,
            summary_text=summary_text,
        )

    async def _finish_session_summary_task_if_ready(self) -> None:
        task = self._session_summary_task
        if task is None or not task.done():
            return
        request = self._session_summary_task_request
        self._session_summary_task = None
        self._session_summary_task_request = None
        try:
            update = task.result()
        except asyncio.CancelledError:
            raise
        except BaseException as exc:
            if request is not None:
                self._session_summary_retry_request = request
            self._last_event_type = "session_summary_update_error"
            await self.logger.log_event(
                RuntimeEvent(
                    session_id=self.config.session_id,
                    event_type="session_summary_update_error",
                    monotonic_time=time.monotonic(),
                    details={
                        "error_type": type(exc).__name__,
                        "message": str(exc),
                    },
                )
            )
            return
        self._session_summary_retry_request = None
        self._session_summary_state = SessionSummaryState(
            text=update.text,
            last_summarized_turn_id=update.last_summarized_turn_id,
        )
        self._last_event_type = "session_summary_updated"
        await self.logger.log_event(
            RuntimeEvent(
                session_id=self.config.session_id,
                event_type="session_summary_updated",
                monotonic_time=time.monotonic(),
                details={
                    "summary_char_count": len(update.text),
                    "last_summarized_turn_id": update.last_summarized_turn_id,
                    "summarized_turn_count": update.summarized_turn_count,
                },
            )
        )

    async def _cancel_session_summary_task(self) -> None:
        task = self._session_summary_task
        if task is None:
            return
        self._session_summary_task = None
        self._session_summary_task_request = None
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    async def _start_floor_prediction_provisional_generations(
        self,
        *,
        turn_id: int,
        input_transcript: str,
        previous_speaker: str | None,
        started_monotonic: float,
    ) -> list[ProvisionalGeneration]:
        if self.config.human_speaker_id is not None:
            return []
        if self.config.speaker_selection_policy not in {
            DISTRIBUTED_TIMING_POLICY,
            TURN_BOUNDARY_TIMING_POLICY,
        }:
            return []
        if _has_multiple_client_participants(self.config):
            return []
        if (
            self._closing_instruction_for_turn(
                turn_id,
                started_monotonic=started_monotonic,
                mark_started=False,
            )
            is not None
        ):
            return []
        prediction = FixedSequenceFloorPredictor(
            participant_roles=self.config.participants,
            sequence=self.config.fixed_speaker_sequence,
            max_candidates=2,
        ).predict(
            turn_id=turn_id,
            previous_speaker=previous_speaker,
            input_transcript=input_transcript,
        )
        provisionals: list[ProvisionalGeneration] = []
        for candidate_speaker in prediction.candidate_speaker_ids:
            agent = self.controller.agents.get(candidate_speaker)
            if agent is None:
                continue
            if not (
                _is_realtime_speech_agent(agent)
                or callable(getattr(agent, "stream_generate", None))
                or callable(getattr(agent, "generate", None))
            ):
                continue
            response_target = _fallback_response_target(
                config=self.config,
                speaker=candidate_speaker,
                previous_speaker=previous_speaker,
            )
            same_speaker_instruction = _same_speaker_continuation_instruction(
                self.config,
                speaker=candidate_speaker,
                previous_speaker=previous_speaker,
            )
            provisionals.append(
                await self._start_provisional_generation(
                    turn_id=turn_id,
                    speaker=candidate_speaker,
                    input_transcript=input_transcript,
                    source_turn_id=self.turns[-1].turn_id if self.turns else turn_id - 1,
                    source="floor_prediction",
                    prediction_reason=prediction.reason,
                    candidate_speaker_ids=prediction.candidate_speaker_ids,
                    response_target=response_target,
                    same_speaker_instruction=same_speaker_instruction,
                )
            )
        return provisionals

    def _can_resolve_overlap(self, decisions: list[TimingDecision]) -> bool:
        request_agent_ids = tuple(
            decision.agent_id
            for decision in decisions
            if decision.action in {
                TimingDecisionAction.REQUEST_MAIN_FLOOR,
                TimingDecisionAction.INTERRUPT,
            }
        )
        return len(set(request_agent_ids)) > 1 and all(
            getattr(self.controller.agents[agent_id], "decide_overlap", None)
            is not None
            for agent_id in request_agent_ids
        )

    async def _collect_overlap_decisions(
        self,
        *,
        turn_id: int,
        input_transcript: str,
        previous_speaker: str | None,
        conflict_agent_ids: tuple[str, ...],
    ) -> list[TimingDecision]:
        decisions: list[TimingDecision] = []
        for agent_id in conflict_agent_ids:
            decide_overlap = getattr(
                self.controller.agents[agent_id],
                "decide_overlap",
                None,
            )
            if decide_overlap is None:
                raise ValueError(f"agent cannot resolve overlap: {agent_id}")
            raw_decision = decide_overlap(
                input_transcript=input_transcript,
                turn_id=turn_id,
                conflict_agent_ids=conflict_agent_ids,
                active_speaker_id=previous_speaker,
            )
            if inspect.isawaitable(raw_decision):
                raw_decision = await raw_decision
            decision = TimingDecision.validate(raw_decision)
            if decision.agent_id != agent_id:
                raise ValueError(
                    "overlap decision agent_id must match the participant speaker_id"
                )
            if decision.action not in {
                TimingDecisionAction.CONTINUE,
                TimingDecisionAction.YIELD,
            }:
                raise ValueError("overlap decision action must be CONTINUE or YIELD")
            decisions.append(decision)
        return decisions

    async def _log_floor_result(
        self,
        turn_id: int,
        *,
        result: FloorMediatorResult,
        decisions: list[TimingDecision],
        fallback_speaker: str | None,
        fallback_selection: FallbackSpeakerSelection | None = None,
        overlap_decisions: list[TimingDecision] | None = None,
        overlap_resolution_timed_out: bool = False,
    ) -> None:
        if result.result_type is FloorMediatorResultType.WAIT:
            event_type = "floor_wait"
            speaker = fallback_speaker
        elif result.result_type is FloorMediatorResultType.OVERLAP:
            event_type = "overlap_detected"
            speaker = result.agent_id
        elif result.result_type is FloorMediatorResultType.YIELD:
            event_type = "floor_yield"
            speaker = fallback_speaker
        elif result.conflict_agent_ids:
            event_type = "floor_conflict_resolved"
            speaker = result.agent_id
        else:
            event_type = "floor_granted"
            speaker = result.agent_id
        self._last_event_type = event_type
        await self.logger.log_event(
            RuntimeEvent(
                session_id=self.config.session_id,
                event_type=event_type,
                turn_id=turn_id,
                speaker=speaker,
                speaker_id=speaker,
                recipient_ids=(
                    self.controller.recipient_ids_for_speaker(speaker)
                    if speaker is not None
                    else ()
                ),
                monotonic_time=time.monotonic(),
                details={
                    "speaker_selection_policy": self.config.speaker_selection_policy,
                    "result_type": result.result_type.value,
                    "reason": result.reason,
                    "granted_agent_id": result.agent_id,
                    "fallback_speaker": fallback_speaker,
                    "fallback_selection": (
                        _fallback_selection_log_details(fallback_selection)
                        if fallback_selection is not None
                        else None
                    ),
                    "conflict_agent_ids": list(result.conflict_agent_ids),
                    "yielded_agent_ids": list(result.yielded_agent_ids),
                    "overlap_grace_ms": self.config.overlap_grace_ms,
                    "unresolved_overlap_limit_ms": self.config.unresolved_overlap_limit_ms,
                    "overlap_resolution_timed_out": overlap_resolution_timed_out,
                    "decisions": [_timing_decision_log_details(item) for item in decisions],
                    "overlap_decisions": (
                        [_timing_decision_log_details(item) for item in overlap_decisions]
                        if overlap_decisions is not None
                        else []
                    ),
                },
            )
        )

    async def _log_turn_taking_signal_collected(
        self,
        *,
        turn_id: int,
        source_turn_id: int,
        previous_speaker: str | None,
        input_transcript: str,
        decisions: list[TimingDecision],
    ) -> None:
        self._last_event_type = "turn_taking_signal_collected"
        await self.logger.log_event(
            RuntimeEvent(
                session_id=self.config.session_id,
                event_type="turn_taking_signal_collected",
                turn_id=turn_id,
                speaker=None,
                speaker_id=None,
                recipient_ids=(),
                monotonic_time=time.monotonic(),
                details={
                    "speaker_selection_policy": self.config.speaker_selection_policy,
                    "source_turn_id": source_turn_id,
                    "previous_speaker": previous_speaker,
                    "signal_min_chars": self.config.turn_taking_signal_min_chars,
                    "input_char_count": len(input_transcript),
                    "decisions": [_timing_decision_log_details(item) for item in decisions],
                },
            )
        )

    async def _log_floor_negotiation_conflict(
        self,
        *,
        turn_id: int,
        round_index: int,
        conflict_agent_ids: tuple[str, ...],
        decisions: list[TimingDecision],
    ) -> None:
        self._last_event_type = "floor_negotiation_conflict"
        await self.logger.log_event(
            RuntimeEvent(
                session_id=self.config.session_id,
                event_type="floor_negotiation_conflict",
                turn_id=turn_id,
                speaker=None,
                speaker_id=None,
                recipient_ids=(),
                monotonic_time=time.monotonic(),
                details={
                    "speaker_selection_policy": self.config.speaker_selection_policy,
                    "round_index": round_index,
                    "max_reconsider_rounds": self.config.turn_taking_max_reconsider_rounds,
                    "conflict_agent_ids": list(conflict_agent_ids),
                    "decisions": [_timing_decision_log_details(item) for item in decisions],
                },
            )
        )

    async def _apply_turn_start_delay(
        self,
        *,
        turn_id: int,
        selection: SpeakerSelection,
    ) -> None:
        delay_ms = _turn_start_delay_ms(selection.timing_decision)
        if delay_ms <= 0:
            return
        decision = selection.timing_decision
        self._last_event_type = "turn_start_delay_applied"
        await self.logger.log_event(
            RuntimeEvent(
                session_id=self.config.session_id,
                event_type="turn_start_delay_applied",
                turn_id=turn_id,
                speaker=selection.speaker,
                speaker_id=selection.speaker,
                recipient_ids=self.controller.recipient_ids_for_speaker(
                    selection.speaker
                ),
                monotonic_time=time.monotonic(),
                details={
                    "delay_ms": delay_ms,
                    "preferred_timing": (
                        decision.preferred_timing if decision is not None else None
                    ),
                    "effective_preferred_timing": (
                        _effective_preferred_timing(decision)
                        if decision is not None
                        else None
                    ),
                    "urgency": decision.urgency if decision is not None else None,
                    "action": (
                        decision.action.value if decision is not None else None
                    ),
                },
            )
        )
        delay_end = time.monotonic() + delay_ms / 1000
        while True:
            await self._generation_resume_event.wait()
            remaining_seconds = delay_end - time.monotonic()
            if remaining_seconds <= 0:
                return
            await asyncio.sleep(min(0.05, remaining_seconds))

    def _should_allow_next_turn_prefetch(
        self,
        turn_id: int,
        *,
        started_monotonic: float,
    ) -> bool:
        if self.config.human_speaker_id is not None:
            return False
        if self.config.speaker_selection_policy in {
            DISTRIBUTED_TIMING_POLICY,
            TURN_BOUNDARY_TIMING_POLICY,
        }:
            return False
        if (
            self._closing_instruction_for_turn(
                turn_id + 1,
                started_monotonic=started_monotonic,
                mark_started=False,
            )
            is not None
        ):
            return False
        if self.config.stop_condition is RuntimeStopCondition.TURNS:
            return turn_id < self.config.max_turns
        return self._should_start_turn(
            turn_id + 1,
            started_monotonic=started_monotonic,
        )

    def _closing_instruction_for_turn(
        self,
        turn_id: int,
        *,
        started_monotonic: float,
        mark_started: bool,
    ) -> str | None:
        closing_start_seconds = self.config.closing_start_elapsed_seconds
        speaker = self.controller.speaker_for_turn(turn_id)
        if self._closing_started_turn_id is None:
            if closing_start_seconds is None:
                return None
            elapsed_seconds = time.monotonic() - started_monotonic
            if elapsed_seconds < closing_start_seconds:
                return None
            if speaker != "counselor":
                return None
            if mark_started:
                self._closing_started_turn_id = turn_id

        if speaker == "counselor":
            if mark_started and self._closing_count_started_turn_id is None:
                self._closing_count_started_turn_id = turn_id
            return CLOSING_COUNSELOR_INSTRUCTION
        participant = self.config.participants.get(speaker)
        if participant is not None and participant.role == CLIENT_ROLE:
            return CLOSING_CLIENT_INSTRUCTION
        return None

    async def _log_closing_started(
        self,
        turn_id: int,
        started_monotonic: float,
    ) -> None:
        speaker = self.controller.speaker_for_turn(turn_id)
        self._last_event_type = "closing_started"
        await self.logger.log_event(
            RuntimeEvent(
                session_id=self.config.session_id,
                event_type="closing_started",
                turn_id=turn_id,
                speaker=speaker,
                speaker_id=speaker,
                recipient_ids=self.controller.recipient_ids_for_speaker(speaker),
                monotonic_time=time.monotonic(),
                details={
                    "elapsed_seconds": max(0.0, time.monotonic() - started_monotonic),
                    "closing_start_elapsed_seconds": self.config.closing_start_elapsed_seconds,
                    "force_stop_after_closing_turns": self.config.force_stop_after_closing_turns,
                    "reason": self._early_closing_reason or "scheduled_time",
                },
            )
        )

    async def _log_runtime_error(
        self,
        exc: BaseException,
        *,
        turn_id: int | None,
        speaker: str | None,
    ) -> None:
        self._last_event_type = "runtime_error"
        recipient_ids = (
            self.controller.recipient_ids_for_speaker(speaker)
            if speaker is not None
            else ()
        )
        await self.logger.log_event(
            RuntimeEvent(
                session_id=self.config.session_id,
                event_type="runtime_error",
                turn_id=turn_id,
                speaker=speaker,
                speaker_id=speaker,
                recipient_ids=recipient_ids,
                monotonic_time=time.monotonic(),
                details={
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                },
            )
        )

    async def _start_provisional_generation(
        self,
        *,
        turn_id: int,
        speaker: str,
        input_transcript: str,
        source_turn_id: int,
        source: str = "stt_partial",
        prediction_reason: str | None = None,
        candidate_speaker_ids: tuple[str, ...] = (),
        response_target: str | None = None,
        same_speaker_instruction: str | None = None,
    ) -> ProvisionalGeneration:
        agent = self.controller.agents[speaker]
        generation_modality = (
            "realtime_audio" if _is_realtime_speech_agent(agent) else "text"
        )
        provisional = ProvisionalGeneration(
            session_id=self.config.session_id,
            turn_id=turn_id,
            speaker=speaker,
            input_transcript=input_transcript,
            source_turn_id=source_turn_id,
            source=source,
            prediction_reason=prediction_reason,
            candidate_speaker_ids=candidate_speaker_ids,
            generation_modality=generation_modality,
            response_target=response_target,
        )
        additional_instruction = _combine_additional_instructions(
            same_speaker_instruction,
            _response_target_instruction(
                self.config,
                speaker=speaker,
                target=response_target,
            ),
        )
        await self.logger.log_event(
            RuntimeEvent(
                session_id=self.config.session_id,
                event_type="provisional_llm_request_started",
                turn_id=turn_id,
                speaker=speaker,
                speaker_id=speaker,
                recipient_ids=self.controller.recipient_ids_for_speaker(speaker),
                monotonic_time=provisional.started_monotonic,
                details={
                    "source_turn_id": source_turn_id,
                    "source": source,
                    "prediction_reason": prediction_reason,
                    "candidate_speaker_ids": list(candidate_speaker_ids),
                    "generation_modality": generation_modality,
                    "provisional_input_char_count": len(input_transcript),
                    "response_target": response_target,
                },
            )
        )

        async def generate() -> None:
            try:
                resolved_additional_instruction = (
                    await self._add_prompt_director_instruction(
                        turn_id=turn_id,
                        speaker=speaker,
                        input_transcript=input_transcript,
                        current_objective=self._current_objective_for_turn(
                            turn_id=turn_id,
                            closing_instruction=None,
                        ),
                        response_target=response_target,
                        existing_instruction=additional_instruction,
                        speculative=True,
                    )
                )
                if _is_realtime_speech_agent(agent):
                    async for event in _iter_realtime_speech_events(
                        agent,
                        session_id=self.config.session_id,
                        turn_id=turn_id,
                        speaker=speaker,
                        input_transcript=input_transcript,
                        additional_instruction=resolved_additional_instruction,
                        format_input=True,
                        sample_rate=self.config.sample_rate,
                        sample_width_bits=self.config.sample_width_bits,
                        channels=self.config.channels,
                        delivery_mode=self.config.audio_delivery_mode,
                        response_instructions_observer=(
                            _response_instructions_logger(self.logger)
                        ),
                    ):
                        text_part = _realtime_speech_text_delta(event) or ""
                        audio_chunk = _realtime_speech_audio_chunk(event)
                        if (
                            provisional.first_token_monotonic is None
                            and (text_part or audio_chunk is not None)
                        ):
                            provisional.first_token_monotonic = time.monotonic()
                            await self.logger.log_metric(
                                RuntimeMetric(
                                    session_id=self.config.session_id,
                                    metric_name="provisional_llm_first_token_latency_ms",
                                    value=_elapsed_ms(
                                        provisional.started_monotonic,
                                        provisional.first_token_monotonic,
                                    ),
                                    turn_id=turn_id,
                                    speaker=speaker,
                                    speaker_id=speaker,
                                    recipient_ids=self.controller.recipient_ids_for_speaker(
                                        speaker
                                    ),
                                    details={
                                        "source_turn_id": source_turn_id,
                                        "source": source,
                                        "generation_modality": generation_modality,
                                        "audio_delivery_mode": self.config.audio_delivery_mode.value,
                                    },
                                )
                            )
                        if text_part:
                            provisional.generated_parts.append(text_part)
                        await provisional.queue.put(event)
                else:
                    async for text_part in _iter_agent_text(
                        agent,
                        input_transcript=input_transcript,
                        turn_id=turn_id,
                        additional_instruction=resolved_additional_instruction,
                    ):
                        if not text_part:
                            continue
                        if provisional.first_token_monotonic is None:
                            provisional.first_token_monotonic = time.monotonic()
                            await self.logger.log_metric(
                                RuntimeMetric(
                                    session_id=self.config.session_id,
                                    metric_name="provisional_llm_first_token_latency_ms",
                                    value=_elapsed_ms(
                                        provisional.started_monotonic,
                                        provisional.first_token_monotonic,
                                    ),
                                    turn_id=turn_id,
                                    speaker=speaker,
                                    speaker_id=speaker,
                                    recipient_ids=self.controller.recipient_ids_for_speaker(
                                        speaker
                                    ),
                                    details={
                                        "source_turn_id": source_turn_id,
                                        "source": source,
                                        "generation_modality": generation_modality,
                                        "audio_delivery_mode": self.config.audio_delivery_mode.value,
                                    },
                                )
                            )
                        provisional.generated_parts.append(text_part)
                        await provisional.queue.put(text_part)
                provisional.completed_monotonic = time.monotonic()
                provisional.status = "completed"
                await self.logger.log_event(
                    RuntimeEvent(
                        session_id=self.config.session_id,
                        event_type="provisional_llm_completed",
                        turn_id=turn_id,
                        speaker=speaker,
                        speaker_id=speaker,
                        recipient_ids=self.controller.recipient_ids_for_speaker(speaker),
                        monotonic_time=provisional.completed_monotonic,
                        details={
                            "source_turn_id": source_turn_id,
                            "source": source,
                            "prediction_reason": prediction_reason,
                            "candidate_speaker_ids": list(candidate_speaker_ids),
                            "generation_modality": generation_modality,
                            "response_target": response_target,
                            "generated_char_count": len("".join(provisional.generated_parts)),
                        },
                    )
                )
            except asyncio.CancelledError as exc:
                provisional.status = "cancelled"
                provisional.error = exc
                raise
            except BaseException as exc:
                provisional.status = "error"
                provisional.error = exc
                await self.logger.log_event(
                    RuntimeEvent(
                        session_id=self.config.session_id,
                        event_type="provisional_llm_error",
                        turn_id=turn_id,
                        speaker=speaker,
                        speaker_id=speaker,
                        recipient_ids=self.controller.recipient_ids_for_speaker(speaker),
                        monotonic_time=time.monotonic(),
                        details={
                            "source_turn_id": source_turn_id,
                            "source": source,
                            "prediction_reason": prediction_reason,
                            "candidate_speaker_ids": list(candidate_speaker_ids),
                            "generation_modality": generation_modality,
                            "response_target": response_target,
                            "error_type": type(exc).__name__,
                            "error_message": str(exc),
                        },
                    )
                )
            finally:
                await provisional.queue.put(_PROVISIONAL_TEXT_DONE)

        provisional.task = asyncio.create_task(generate())
        return provisional

    async def _discard_provisional_generation(
        self,
        provisional: ProvisionalGeneration,
        *,
        reason: str,
        final_transcript: str | None,
    ) -> None:
        if provisional.task is not None and not provisional.task.done():
            provisional.task.cancel()
            await asyncio.gather(provisional.task, return_exceptions=True)
        if provisional.generation_modality == "realtime_audio":
            await _close_runtime_agent(
                self.controller.agents.get(provisional.speaker)
            )
        provisional.status = "discarded"
        await self.logger.log_event(
            RuntimeEvent(
                session_id=self.config.session_id,
                event_type="provisional_llm_discarded",
                turn_id=provisional.turn_id,
                speaker=provisional.speaker,
                speaker_id=provisional.speaker,
                recipient_ids=self.controller.recipient_ids_for_speaker(
                    provisional.speaker
                ),
                monotonic_time=time.monotonic(),
                details={
                    "source_turn_id": provisional.source_turn_id,
                    "source": provisional.source,
                    "prediction_reason": provisional.prediction_reason,
                    "candidate_speaker_ids": list(
                        provisional.candidate_speaker_ids
                    ),
                    "response_target": provisional.response_target,
                    "reason": reason,
                    "provisional_input_char_count": len(provisional.input_transcript),
                    "final_input_char_count": len(final_transcript or ""),
                },
            )
        )

    async def _log_inter_turn_latency(
        self,
        previous_turn: TurnRuntimeState,
        current_turn: TurnRuntimeState,
    ) -> None:
        if (
            previous_turn.stt_final_monotonic is not None
            and current_turn.llm_request_started_monotonic is not None
        ):
            await _log_metric(
                self.logger,
                current_turn,
                "previous_stt_final_to_next_llm_request_ms",
                _elapsed_ms(
                    previous_turn.stt_final_monotonic,
                    current_turn.llm_request_started_monotonic,
                ),
                {
                    "previous_turn_id": previous_turn.turn_id,
                    "audio_delivery_mode": self.config.audio_delivery_mode.value,
                    "generation_source": (
                        "provisional" if current_turn.used_provisional_generation else "final"
                    ),
                },
            )
        if (
            previous_turn.stt_final_monotonic is not None
            and current_turn.first_audio_chunk_monotonic is not None
        ):
            await _log_metric(
                self.logger,
                current_turn,
                "previous_stt_final_to_next_first_audio_ms",
                _elapsed_ms(
                    previous_turn.stt_final_monotonic,
                    current_turn.first_audio_chunk_monotonic,
                ),
                {
                    "previous_turn_id": previous_turn.turn_id,
                    "audio_delivery_mode": self.config.audio_delivery_mode.value,
                    "generation_source": (
                        "provisional" if current_turn.used_provisional_generation else "final"
                    ),
                },
            )
        if (
            previous_turn.completed_monotonic is not None
            and current_turn.first_audio_chunk_monotonic is not None
        ):
            await _log_metric(
                self.logger,
                current_turn,
                "previous_turn_completed_to_next_first_audio_ms",
                _elapsed_ms(
                    previous_turn.completed_monotonic,
                    current_turn.first_audio_chunk_monotonic,
                ),
                {
                    "previous_turn_id": previous_turn.turn_id,
                    "audio_delivery_mode": self.config.audio_delivery_mode.value,
                    "generation_source": (
                        "provisional" if current_turn.used_provisional_generation else "final"
                    ),
                },
            )


def _select_contextual_fallback_speaker(
    *,
    config: RuntimeConfig,
    turns: Iterable[TurnRuntimeState],
    turn_id: int,
    base_speaker: str,
    previous_speaker: str | None,
    previous_turn_target_speaker: str | None,
    decisions: Iterable[TimingDecision],
) -> FallbackSpeakerSelection:
    candidate_speaker_ids = _fallback_candidate_speaker_ids(config)
    if not candidate_speaker_ids:
        return FallbackSpeakerSelection(
            speaker_id=base_speaker,
            base_speaker_id=base_speaker,
            strategy="fixed_sequence",
            candidates=(),
        )

    turn_history = tuple(turns)
    recent_speakers = tuple(
        turn.speaker for turn in turn_history[-_FALLBACK_RECENT_TURN_LIMIT:]
    )
    candidate_scores = tuple(
        _score_fallback_candidate(
            config=config,
            candidate_speaker_id=speaker_id,
            turn_id=turn_id,
            base_speaker=base_speaker,
            previous_speaker=previous_speaker,
            previous_turn_target_speaker=previous_turn_target_speaker,
            recent_speakers=recent_speakers,
            decisions=decisions,
        )
        for speaker_id in candidate_speaker_ids
    )
    winner = min(
        candidate_scores,
        key=lambda candidate: (
            -candidate.score,
            candidate.sequence_distance,
            candidate.speaker_id,
        ),
    )
    return FallbackSpeakerSelection(
        speaker_id=winner.speaker_id,
        base_speaker_id=base_speaker,
        strategy="contextual_score",
        candidates=tuple(
            sorted(
                candidate_scores,
                key=lambda candidate: (
                    -candidate.score,
                    candidate.sequence_distance,
                    candidate.speaker_id,
                ),
            )
        ),
    )


def _fallback_candidate_speaker_ids(config: RuntimeConfig) -> tuple[str, ...]:
    candidate_speaker_ids: list[str] = []
    seen: set[str] = set()
    for speaker_id in (*config.fixed_speaker_sequence, *config.participants):
        if speaker_id in seen:
            continue
        if speaker_id not in config.participants:
            continue
        candidate_speaker_ids.append(speaker_id)
        seen.add(speaker_id)
    return tuple(candidate_speaker_ids)


def _score_fallback_candidate(
    *,
    config: RuntimeConfig,
    candidate_speaker_id: str,
    turn_id: int,
    base_speaker: str,
    previous_speaker: str | None,
    previous_turn_target_speaker: str | None,
    recent_speakers: tuple[str, ...],
    decisions: Iterable[TimingDecision],
) -> FallbackSpeakerCandidate:
    score = 0
    reasons: list[str] = []
    if candidate_speaker_id == base_speaker:
        score += _FALLBACK_FIXED_SEQUENCE_SCORE
        reasons.append("fixed_sequence")

    candidate_role = _participant_role(config, candidate_speaker_id)
    previous_role = _participant_role(config, previous_speaker)
    if previous_speaker is not None and candidate_speaker_id == previous_speaker:
        score += _FALLBACK_PREVIOUS_SPEAKER_PENALTY
        reasons.append("previous_speaker")
    elif previous_role == CLIENT_ROLE and candidate_role == CLIENT_ROLE:
        score += _FALLBACK_CLIENT_REPLY_CONTINUITY_SCORE
        reasons.append("client_reply_continuity")
    elif previous_role == COUNSELOR_ROLE and candidate_role == CLIENT_ROLE:
        score += _FALLBACK_CLIENT_AFTER_COUNSELOR_SCORE
        reasons.append("client_after_counselor")
    elif previous_role == CLIENT_ROLE and candidate_role == COUNSELOR_ROLE:
        score += _FALLBACK_COUNSELOR_AFTER_CLIENT_SCORE
        reasons.append("counselor_after_client")

    if (
        previous_turn_target_speaker is not None
        and candidate_speaker_id == previous_turn_target_speaker
        and candidate_speaker_id != previous_speaker
        and _participant_role(config, candidate_speaker_id) == CLIENT_ROLE
    ):
        score += _FALLBACK_PREVIOUS_TURN_TARGET_SCORE
        reasons.append("previous_turn_target")

    recent_count = recent_speakers.count(candidate_speaker_id)
    if recent_speakers and recent_count == 0:
        score += _FALLBACK_RECENTLY_SILENT_SCORE
        reasons.append("recently_silent")
    if recent_speakers:
        max_recent_count = max(
            recent_speakers.count(speaker_id) for speaker_id in set(recent_speakers)
        )
        if recent_count < max_recent_count:
            score += (
                max_recent_count - recent_count
            ) * _FALLBACK_RECENT_BALANCE_SCORE
            reasons.append("recent_turn_balance")

    if _candidate_continues_repeated_speaker_pattern(
        config=config,
        recent_speakers=recent_speakers,
        candidate_speaker_id=candidate_speaker_id,
    ):
        score += _FALLBACK_REPEATED_SPEAKER_PATTERN_PENALTY
        reasons.append("repeated_speaker_pattern")

    decision = _timing_decision_for_agent(decisions, candidate_speaker_id)
    if decision is not None:
        if (
            decision.explicitly_addressed is True
            and candidate_speaker_id != previous_speaker
            and decision.action is not TimingDecisionAction.CANCEL
        ):
            score += _FALLBACK_EXPLICITLY_ADDRESSED_SCORE
            reasons.append("explicitly_addressed")
        if (
            previous_speaker is not None
            and decision.target == previous_speaker
            and not decision.target_defaulted
        ):
            score += _FALLBACK_EXPLICIT_TARGET_PREVIOUS_SCORE
            reasons.append("explicit_target_previous")
        if decision.reason_code == "timing_decision_timeout":
            score += _FALLBACK_TIMEOUT_PENALTY
            reasons.append("timing_timeout")

    if not reasons:
        reasons.append("available")
    return FallbackSpeakerCandidate(
        speaker_id=candidate_speaker_id,
        score=score,
        sequence_distance=_fallback_sequence_distance(
            config=config,
            turn_id=turn_id,
            speaker_id=candidate_speaker_id,
        ),
        reasons=tuple(reasons),
    )


def _candidate_continues_repeated_speaker_pattern(
    *,
    config: RuntimeConfig,
    recent_speakers: tuple[str, ...],
    candidate_speaker_id: str,
) -> bool:
    if candidate_speaker_id not in config.participants:
        return False
    normalized_recent_speakers = tuple(
        speaker_id
        for speaker_id in recent_speakers
        if speaker_id in config.participants
    )
    max_pattern_length = min(
        _FALLBACK_REPEATED_SPEAKER_PATTERN_MAX_LENGTH,
        len(normalized_recent_speakers),
    )
    for pattern_length in range(2, max_pattern_length + 1):
        if _candidate_continues_pattern_prefix(
            recent_speakers=normalized_recent_speakers,
            candidate_speaker_id=candidate_speaker_id,
            pattern_length=pattern_length,
        ):
            return True
    return False


def _candidate_continues_pattern_prefix(
    *,
    recent_speakers: tuple[str, ...],
    candidate_speaker_id: str,
    pattern_length: int,
) -> bool:
    for prefix_length in range(0, pattern_length + 1):
        repeated_window_length = pattern_length + prefix_length
        if len(recent_speakers) < repeated_window_length:
            continue
        window = recent_speakers[-repeated_window_length:]
        pattern = window[:pattern_length]
        if len(set(pattern)) < 2:
            continue
        if window[pattern_length:] != pattern[:prefix_length]:
            continue
        next_pattern_index = 0 if prefix_length == pattern_length else prefix_length
        if candidate_speaker_id == pattern[next_pattern_index]:
            return True
    return False


def _fallback_sequence_distance(
    *,
    config: RuntimeConfig,
    turn_id: int,
    speaker_id: str,
) -> int:
    sequence = tuple(config.fixed_speaker_sequence)
    if not sequence:
        return 0
    start_index = (turn_id - 1) % len(sequence)
    for offset in range(len(sequence)):
        if sequence[(start_index + offset) % len(sequence)] == speaker_id:
            return offset
    return len(sequence)


def _fallback_selection_log_details(
    selection: FallbackSpeakerSelection,
) -> dict[str, Any]:
    return {
        "speaker_id": selection.speaker_id,
        "base_speaker_id": selection.base_speaker_id,
        "strategy": selection.strategy,
        "candidates": [
            {
                "speaker_id": candidate.speaker_id,
                "score": candidate.score,
                "sequence_distance": candidate.sequence_distance,
                "reasons": list(candidate.reasons),
            }
            for candidate in selection.candidates
        ],
    }


def _complete_timing_decision_metadata(
    decision: TimingDecision,
    *,
    previous_speaker: str | None,
    config: RuntimeConfig,
) -> TimingDecision:
    target = _valid_timing_target(decision, config)
    target_defaulted = decision.target_defaulted
    if target is None:
        target = _default_timing_target(
            decision,
            previous_speaker=previous_speaker,
            config=config,
        )
        target_defaulted = True
    preferred_timing = _normalize_preferred_timing(decision.preferred_timing)
    preferred_timing_defaulted = decision.preferred_timing_defaulted
    if preferred_timing is None:
        preferred_timing = _default_preferred_timing(
            decision,
            previous_speaker=previous_speaker,
            config=config,
        )
        preferred_timing_defaulted = True
    urgency_defaulted = decision.urgency_defaulted
    urgency = (
        decision.urgency
        if decision.urgency is not None
        else _default_timing_urgency(
            decision,
            previous_speaker=previous_speaker,
            config=config,
        )
    )
    if decision.urgency is None:
        urgency_defaulted = True
    if (
        target == decision.target
        and preferred_timing == decision.preferred_timing
        and urgency == decision.urgency
        and target_defaulted == decision.target_defaulted
        and preferred_timing_defaulted == decision.preferred_timing_defaulted
        and urgency_defaulted == decision.urgency_defaulted
    ):
        return decision
    return replace(
        decision,
        target=target,
        preferred_timing=preferred_timing,
        urgency=urgency,
        target_defaulted=target_defaulted,
        preferred_timing_defaulted=preferred_timing_defaulted,
        urgency_defaulted=urgency_defaulted,
    )


def _valid_timing_target(
    decision: TimingDecision,
    config: RuntimeConfig,
) -> str | None:
    target = (decision.target or "").strip()
    if not target:
        return None
    if target == decision.agent_id:
        return None
    if target not in config.participants:
        return None
    return target


def _default_timing_target(
    decision: TimingDecision,
    *,
    previous_speaker: str | None,
    config: RuntimeConfig,
) -> str:
    if (
        previous_speaker is not None
        and previous_speaker != decision.agent_id
        and previous_speaker in config.participants
    ):
        return previous_speaker
    if _participant_role(config, decision.agent_id) == CLIENT_ROLE:
        if "counselor" in config.participants and decision.agent_id != "counselor":
            return "counselor"
    for speaker_id in config.participants:
        if speaker_id != decision.agent_id:
            return speaker_id
    return decision.agent_id


def _default_preferred_timing(
    decision: TimingDecision,
    *,
    previous_speaker: str | None,
    config: RuntimeConfig,
) -> str:
    if decision.action is TimingDecisionAction.INTERRUPT:
        return "immediate"
    if (
        decision.action is TimingDecisionAction.REQUEST_MAIN_FLOOR
        and _is_client_to_client_transition(
            speaker=decision.agent_id,
            previous_speaker=previous_speaker,
            config=config,
        )
    ):
        return "immediate"
    return "natural_pause"


def _default_timing_urgency(
    decision: TimingDecision,
    *,
    previous_speaker: str | None,
    config: RuntimeConfig,
) -> float:
    if decision.action is TimingDecisionAction.INTERRUPT:
        return 1.0
    if (
        decision.action is TimingDecisionAction.REQUEST_MAIN_FLOOR
        and _is_client_to_client_transition(
            speaker=decision.agent_id,
            previous_speaker=previous_speaker,
            config=config,
        )
    ):
        return 0.7
    if decision.action is TimingDecisionAction.REQUEST_MAIN_FLOOR:
        return 0.5
    if decision.action is TimingDecisionAction.CANCEL:
        return 0.1
    return 0.2


def _allow_client_reply_auto_priority(
    *,
    turns: Iterable[TurnRuntimeState],
    previous_speaker: str | None,
    config: RuntimeConfig,
) -> bool:
    if previous_speaker is None:
        return True
    if _participant_role(config, previous_speaker) != CLIENT_ROLE:
        return True
    return (
        _trailing_role_turn_count(turns, role=CLIENT_ROLE, config=config)
        <= _CLIENT_REPLY_AUTO_PRIORITY_MAX_CONSECUTIVE_CLIENT_TURNS
    )


def _counselor_reentry_after_client_run_decision(
    decisions: Iterable[TimingDecision],
    *,
    turns: Iterable[TurnRuntimeState],
    previous_speaker: str | None,
    config: RuntimeConfig,
) -> TimingDecision | None:
    if previous_speaker is None:
        return None
    if _participant_role(config, previous_speaker) != CLIENT_ROLE:
        return None
    consecutive_client_turns = _trailing_role_turn_count(
        turns,
        role=CLIENT_ROLE,
        config=config,
    )
    if (
        consecutive_client_turns
        <= _CLIENT_REPLY_AUTO_PRIORITY_MAX_CONSECUTIVE_CLIENT_TURNS
    ):
        return None
    if not _has_multiple_client_participants(config):
        return None

    all_decisions = tuple(decisions)
    counselor_decisions = [
        decision
        for decision in all_decisions
        if _participant_role(config, decision.agent_id) == COUNSELOR_ROLE
    ]
    if not counselor_decisions:
        return None
    counselor_decision = min(
        counselor_decisions,
        key=_timing_decision_controller_priority_key,
    )

    requests = [
        decision
        for decision in all_decisions
        if decision.action
        in {
            TimingDecisionAction.REQUEST_MAIN_FLOOR,
            TimingDecisionAction.INTERRUPT,
        }
    ]
    if any(decision.safety_intervention is True for decision in requests):
        return None

    counselor_requests = [
        decision
        for decision in requests
        if _participant_role(config, decision.agent_id) == COUNSELOR_ROLE
    ]
    client_continuation_requests = [
        decision
        for decision in requests
        if decision.action is TimingDecisionAction.REQUEST_MAIN_FLOOR
        and decision.agent_id != previous_speaker
        and _participant_role(config, decision.agent_id) == CLIENT_ROLE
    ]
    if not client_continuation_requests:
        return counselor_decision

    if not counselor_requests:
        return None

    counselor_decision = min(
        counselor_requests,
        key=_timing_decision_controller_priority_key,
    )
    client_decision = min(
        client_continuation_requests,
        key=_timing_decision_controller_priority_key,
    )
    if _client_reply_can_extend_after_client_run(
        client_decision,
        counselor_decision=counselor_decision,
        turns=turns,
        previous_speaker=previous_speaker,
        config=config,
        consecutive_client_turns=consecutive_client_turns,
    ):
        return None
    return counselor_decision


def _extended_client_reply_after_client_run_decision(
    decisions: Iterable[TimingDecision],
    *,
    turns: Iterable[TurnRuntimeState],
    previous_speaker: str | None,
    config: RuntimeConfig,
) -> TimingDecision | None:
    if previous_speaker is None:
        return None
    if _participant_role(config, previous_speaker) != CLIENT_ROLE:
        return None
    consecutive_client_turns = _trailing_role_turn_count(
        turns,
        role=CLIENT_ROLE,
        config=config,
    )
    if (
        consecutive_client_turns
        <= _CLIENT_REPLY_AUTO_PRIORITY_MAX_CONSECUTIVE_CLIENT_TURNS
    ):
        return None
    if not _has_multiple_client_participants(config):
        return None

    all_decisions = tuple(decisions)
    requests = [
        decision
        for decision in all_decisions
        if decision.action
        in {
            TimingDecisionAction.REQUEST_MAIN_FLOOR,
            TimingDecisionAction.INTERRUPT,
        }
    ]
    if any(decision.safety_intervention is True for decision in requests):
        return None
    counselor_decision = _strongest_counselor_request(requests, config=config)
    if counselor_decision is None:
        return None
    client_continuation_requests = [
        decision
        for decision in requests
        if decision.action is TimingDecisionAction.REQUEST_MAIN_FLOOR
        and decision.agent_id != previous_speaker
        and _participant_role(config, decision.agent_id) == CLIENT_ROLE
    ]
    if not client_continuation_requests:
        return None

    client_decision = min(
        client_continuation_requests,
        key=_timing_decision_controller_priority_key,
    )
    if not _client_reply_can_extend_after_client_run(
        client_decision,
        counselor_decision=counselor_decision,
        turns=turns,
        previous_speaker=previous_speaker,
        config=config,
        consecutive_client_turns=consecutive_client_turns,
    ):
        return None
    return client_decision


def _client_reply_can_extend_after_client_run(
    decision: TimingDecision,
    *,
    counselor_decision: TimingDecision,
    turns: Iterable[TurnRuntimeState],
    previous_speaker: str | None,
    config: RuntimeConfig,
    consecutive_client_turns: int,
) -> bool:
    if _extended_client_reply_is_strong_enough(
        decision,
        counselor_decision=counselor_decision,
        consecutive_client_turns=consecutive_client_turns,
    ):
        return True
    return _direct_client_reply_can_complete_counselor_started_exchange(
        decision,
        counselor_decision=counselor_decision,
        turns=turns,
        previous_speaker=previous_speaker,
        config=config,
        consecutive_client_turns=consecutive_client_turns,
    )


def _extended_client_reply_is_strong_enough(
    decision: TimingDecision,
    *,
    counselor_decision: TimingDecision,
    consecutive_client_turns: int,
) -> bool:
    timing_rank, urgency_rank = timing_decision_conflict_priority(decision)
    if timing_rank != _CLIENT_REPLY_EXTENDED_IMMEDIATE_TIMING_RANK:
        return False
    urgency = -urgency_rank
    extra_client_turns = max(
        0,
        consecutive_client_turns
        - (_CLIENT_REPLY_AUTO_PRIORITY_MAX_CONSECUTIVE_CLIENT_TURNS + 1),
    )
    required_urgency = min(
        0.95,
        _CLIENT_REPLY_EXTENDED_URGENCY_FLOOR
        + extra_client_turns * _CLIENT_REPLY_EXTENDED_URGENCY_STEP,
    )
    if urgency < required_urgency:
        return False

    counselor_timing_rank, counselor_urgency_rank = (
        timing_decision_conflict_priority(counselor_decision)
    )
    if counselor_timing_rank != _CLIENT_REPLY_EXTENDED_IMMEDIATE_TIMING_RANK:
        return True
    counselor_urgency = -counselor_urgency_rank
    required_margin = (
        _CLIENT_REPLY_EXTENDED_IMMEDIATE_MARGIN
        + extra_client_turns * _CLIENT_REPLY_EXTENDED_IMMEDIATE_MARGIN_STEP
    )
    return urgency >= counselor_urgency + required_margin


def _direct_client_reply_can_complete_counselor_started_exchange(
    decision: TimingDecision,
    *,
    counselor_decision: TimingDecision,
    turns: Iterable[TurnRuntimeState],
    previous_speaker: str | None,
    config: RuntimeConfig,
    consecutive_client_turns: int,
) -> bool:
    if (
        consecutive_client_turns
        != _CLIENT_REPLY_AUTO_PRIORITY_MAX_CONSECUTIVE_CLIENT_TURNS + 1
    ):
        return False
    if not _trailing_client_run_started_after_counselor(
        turns,
        consecutive_client_turns=consecutive_client_turns,
        config=config,
    ):
        return False
    if previous_speaker is None:
        return False
    if decision.target_defaulted:
        return False
    target = (decision.target or "").strip()
    target_is_previous_client = target == previous_speaker
    target_is_counselor = _participant_role(config, target) == COUNSELOR_ROLE
    if not target_is_previous_client and not target_is_counselor:
        return False
    if (
        counselor_decision.explicitly_addressed is True
        and decision.explicitly_addressed is not True
    ):
        return False

    timing_rank, urgency_rank = timing_decision_conflict_priority(decision)
    counselor_timing_rank, counselor_urgency_rank = (
        timing_decision_conflict_priority(counselor_decision)
    )
    if counselor_timing_rank < timing_rank:
        return False
    urgency = -urgency_rank
    counselor_urgency = -counselor_urgency_rank
    if target_is_previous_client:
        required_urgency = _CLIENT_REPLY_EXTENDED_DIRECT_REPLY_URGENCY_FLOOR
        required_margin = _CLIENT_REPLY_EXTENDED_DIRECT_REPLY_MARGIN
    else:
        required_urgency = _CLIENT_REPLY_EXTENDED_COUNSELOR_TARGET_URGENCY_FLOOR
        required_margin = _CLIENT_REPLY_EXTENDED_COUNSELOR_TARGET_MARGIN
    if urgency < required_urgency:
        return False
    return urgency >= counselor_urgency + required_margin


def _trailing_client_run_started_after_counselor(
    turns: Iterable[TurnRuntimeState],
    *,
    consecutive_client_turns: int,
    config: RuntimeConfig,
) -> bool:
    turn_history = tuple(turns)
    run_start_index = len(turn_history) - consecutive_client_turns
    if run_start_index <= 0:
        return False
    preceding_turn = turn_history[run_start_index - 1]
    return _participant_role(config, preceding_turn.speaker) == COUNSELOR_ROLE


def _timing_decision_controller_priority_key(
    decision: TimingDecision,
) -> tuple[int, int, float, str]:
    timing_rank, urgency_rank = timing_decision_conflict_priority(decision)
    return (
        0 if decision.explicitly_addressed is True else 1,
        timing_rank,
        urgency_rank,
        decision.agent_id,
    )


def _trailing_role_turn_count(
    turns: Iterable[TurnRuntimeState],
    *,
    role: str,
    config: RuntimeConfig,
) -> int:
    count = 0
    for turn in reversed(tuple(turns)):
        if _participant_role(config, turn.speaker) != role:
            break
        count += 1
    return count


def _client_reply_priority_decision(
    decisions: Iterable[TimingDecision],
    *,
    turns: Iterable[TurnRuntimeState],
    previous_speaker: str | None,
    config: RuntimeConfig,
) -> TimingDecision | None:
    if previous_speaker is None:
        return None
    if _participant_role(config, previous_speaker) != CLIENT_ROLE:
        return None
    requests = [
        decision
        for decision in decisions
        if decision.action
        in {
            TimingDecisionAction.REQUEST_MAIN_FLOOR,
            TimingDecisionAction.INTERRUPT,
        }
    ]
    if len(requests) <= 1:
        return None
    if any(decision.safety_intervention is True for decision in requests):
        return None
    candidates = [
        decision
        for decision in requests
        if decision.action is TimingDecisionAction.REQUEST_MAIN_FLOOR
        and decision.agent_id != previous_speaker
        and _participant_role(config, decision.agent_id) == CLIENT_ROLE
        and (decision.target or "").strip() == previous_speaker
        and not decision.target_defaulted
    ]
    if not candidates:
        return None
    candidate = min(
        candidates,
        key=lambda decision: (
            0 if decision.explicitly_addressed is True else 1,
            _missing_last_float(decision.request_timestamp_ms),
            decision.agent_id,
        ),
    )
    counselor_decision = _strongest_counselor_request(requests, config=config)
    if (
        counselor_decision is not None
        and _repeated_pattern_client_reply_should_yield_to_counselor(
            candidate,
            counselor_decision=counselor_decision,
            turns=turns,
            config=config,
        )
    ):
        return None
    return candidate


def _client_reply_opportunity_decision(
    decisions: Iterable[TimingDecision],
    *,
    turns: Iterable[TurnRuntimeState],
    previous_speaker: str | None,
    config: RuntimeConfig,
) -> TimingDecision | None:
    if previous_speaker is None:
        return None
    if _participant_role(config, previous_speaker) != CLIENT_ROLE:
        return None
    requests = [
        decision
        for decision in decisions
        if decision.action
        in {
            TimingDecisionAction.REQUEST_MAIN_FLOOR,
            TimingDecisionAction.INTERRUPT,
        }
    ]
    if len(requests) <= 1:
        return None
    if any(decision.safety_intervention is True for decision in requests):
        return None
    if not any(
        _participant_role(config, decision.agent_id) == COUNSELOR_ROLE
        for decision in requests
    ):
        return None
    candidates = [
        decision
        for decision in requests
        if decision.action is TimingDecisionAction.REQUEST_MAIN_FLOOR
        and decision.agent_id != previous_speaker
        and _participant_role(config, decision.agent_id) == CLIENT_ROLE
        and (decision.target or "").strip() == previous_speaker
        and not decision.target_defaulted
        and (
            decision.urgency is None
            or decision.urgency >= _CLIENT_REPLY_OPPORTUNITY_URGENCY_FLOOR
        )
    ]
    if not candidates:
        return None
    candidate = min(
        candidates,
        key=lambda decision: (
            0
            if (decision.target or "").strip() == previous_speaker
            and not decision.target_defaulted
            else 1,
            0 if decision.explicitly_addressed is True else 1,
            -float(decision.urgency or 0.0),
            _missing_last_float(decision.request_timestamp_ms),
            decision.agent_id,
        ),
    )
    counselor_decision = _strongest_counselor_request(requests, config=config)
    if (
        counselor_decision is not None
        and _repeated_pattern_client_reply_should_yield_to_counselor(
            candidate,
            counselor_decision=counselor_decision,
            turns=turns,
            config=config,
        )
    ):
        return None
    return candidate


def _explicitly_addressed_priority_decision(
    decisions: Iterable[TimingDecision],
    *,
    previous_speaker: str | None,
    config: RuntimeConfig,
) -> TimingDecision | None:
    all_decisions = tuple(decisions)
    addressed_candidates = [
        decision
        for decision in all_decisions
        if decision.explicitly_addressed is True
        and decision.agent_id != previous_speaker
        and decision.agent_id in config.participants
        and decision.action is not TimingDecisionAction.CANCEL
    ]
    if not addressed_candidates:
        return None

    addressed_agent_ids = {decision.agent_id for decision in addressed_candidates}
    competing_requests = [
        decision
        for decision in all_decisions
        if decision.agent_id not in addressed_agent_ids
        and decision.action
        in {
            TimingDecisionAction.REQUEST_MAIN_FLOOR,
            TimingDecisionAction.INTERRUPT,
        }
    ]
    if any(
        _competing_request_can_override_explicit_addressing(decision)
        for decision in competing_requests
    ):
        return None

    return min(
        addressed_candidates,
        key=_explicitly_addressed_priority_key,
    )


def _explicitly_addressed_priority_key(
    decision: TimingDecision,
) -> tuple[int, int, float, str]:
    timing_rank, urgency_rank = timing_decision_conflict_priority(decision)
    return (
        0
        if decision.action
        in {
            TimingDecisionAction.REQUEST_MAIN_FLOOR,
            TimingDecisionAction.INTERRUPT,
        }
        else 1,
        timing_rank,
        urgency_rank,
        decision.agent_id,
    )


def _competing_request_can_override_explicit_addressing(
    decision: TimingDecision,
) -> bool:
    if decision.safety_intervention is True:
        return True
    if decision.action is TimingDecisionAction.INTERRUPT:
        return True
    timing_rank, urgency_rank = timing_decision_conflict_priority(decision)
    urgency = -urgency_rank
    return (
        timing_rank == _CLIENT_REPLY_EXTENDED_IMMEDIATE_TIMING_RANK
        and urgency >= _EXPLICITLY_ADDRESSED_OVERRIDE_URGENCY_FLOOR
    )


def _strongest_counselor_request(
    decisions: Iterable[TimingDecision],
    *,
    config: RuntimeConfig,
) -> TimingDecision | None:
    counselor_requests = [
        decision
        for decision in decisions
        if decision.action
        in {
            TimingDecisionAction.REQUEST_MAIN_FLOOR,
            TimingDecisionAction.INTERRUPT,
        }
        and _participant_role(config, decision.agent_id) == COUNSELOR_ROLE
    ]
    if not counselor_requests:
        return None
    return min(counselor_requests, key=_timing_decision_controller_priority_key)


def _repeated_pattern_client_reply_should_yield_to_counselor(
    decision: TimingDecision,
    *,
    counselor_decision: TimingDecision,
    turns: Iterable[TurnRuntimeState],
    config: RuntimeConfig,
) -> bool:
    if decision.safety_intervention is True:
        return False
    if (
        decision.explicitly_addressed is True
        and counselor_decision.explicitly_addressed is not True
    ):
        return False
    recent_speakers = tuple(
        turn.speaker for turn in tuple(turns)[-_FALLBACK_RECENT_TURN_LIMIT:]
    )
    if not _candidate_continues_repeated_speaker_pattern(
        config=config,
        recent_speakers=recent_speakers,
        candidate_speaker_id=decision.agent_id,
    ):
        return False

    timing_rank, urgency_rank = timing_decision_conflict_priority(decision)
    counselor_timing_rank, counselor_urgency_rank = (
        timing_decision_conflict_priority(counselor_decision)
    )
    if counselor_timing_rank > timing_rank:
        return False
    urgency = -urgency_rank
    counselor_urgency = -counselor_urgency_rank
    return (
        counselor_urgency
        + _CLIENT_REPLY_REPEATED_PATTERN_COUNSELOR_URGENCY_MARGIN
    ) >= urgency


def _client_contextual_reply_opportunity_decision(
    decisions: Iterable[TimingDecision],
    *,
    previous_speaker: str | None,
    config: RuntimeConfig,
) -> TimingDecision | None:
    if previous_speaker is None:
        return None
    if _participant_role(config, previous_speaker) != CLIENT_ROLE:
        return None
    requests = [
        decision
        for decision in decisions
        if decision.action
        in {
            TimingDecisionAction.REQUEST_MAIN_FLOOR,
            TimingDecisionAction.INTERRUPT,
        }
    ]
    if len(requests) <= 1:
        return None
    if any(decision.safety_intervention is True for decision in requests):
        return None

    counselor_requests = [
        decision
        for decision in requests
        if _participant_role(config, decision.agent_id) == COUNSELOR_ROLE
    ]
    candidates = [
        decision
        for decision in requests
        if decision.action is TimingDecisionAction.REQUEST_MAIN_FLOOR
        and decision.agent_id != previous_speaker
        and _participant_role(config, decision.agent_id) == CLIENT_ROLE
        and (decision.target or "").strip() == "counselor"
        and not decision.target_defaulted
        and (decision.urgency or 0.0) >= _CLIENT_CONTEXTUAL_REPLY_URGENCY_FLOOR
    ]
    if not counselor_requests or not candidates:
        return None

    counselor_decision = min(
        counselor_requests,
        key=_timing_decision_controller_priority_key,
    )
    candidate = min(
        candidates,
        key=_timing_decision_controller_priority_key,
    )
    if _timing_decision_controller_priority_key(candidate) <= (
        _timing_decision_controller_priority_key(counselor_decision)
    ):
        return None
    if not _contextual_client_reply_can_precede_counselor(
        candidate,
        counselor_decision=counselor_decision,
    ):
        return None
    return candidate


def _contextual_client_reply_can_precede_counselor(
    decision: TimingDecision,
    *,
    counselor_decision: TimingDecision,
) -> bool:
    if (
        counselor_decision.explicitly_addressed is True
        and decision.explicitly_addressed is not True
    ):
        return False
    timing_rank, urgency_rank = timing_decision_conflict_priority(decision)
    counselor_timing_rank, counselor_urgency_rank = (
        timing_decision_conflict_priority(counselor_decision)
    )
    if counselor_timing_rank < timing_rank:
        return False
    urgency = -urgency_rank
    counselor_urgency = -counselor_urgency_rank
    return (
        counselor_urgency - urgency
    ) <= _CLIENT_CONTEXTUAL_REPLY_COUNSELOR_URGENCY_MARGIN


def _is_client_to_client_transition(
    *,
    speaker: str,
    previous_speaker: str | None,
    config: RuntimeConfig,
) -> bool:
    return (
        previous_speaker is not None
        and previous_speaker != speaker
        and _participant_role(config, speaker) == CLIENT_ROLE
        and _participant_role(config, previous_speaker) == CLIENT_ROLE
    )


def _participant_role(config: RuntimeConfig, speaker_id: str | None) -> str | None:
    if speaker_id is None:
        return None
    participant = config.participants.get(speaker_id)
    return participant.role if participant is not None else None


def _prompt_director_speaker_profile(participant: ParticipantConfig) -> str:
    sections = [f"表示名: {participant.display_name.strip() or participant.speaker_id}"]
    profile_sources = (
        ("公開プロフィール", participant.public_profile_source),
        ("個別プロフィール", participant.private_profile_source),
    )
    seen_profiles: set[str] = set()
    for label, source in profile_sources:
        normalized = (source or "").strip()
        if not normalized or normalized in seen_profiles:
            continue
        sections.append(f"{label}:\n{normalized}")
        seen_profiles.add(normalized)
    return "\n\n".join(sections)


def _ai_counselor_initial_client_opening_instruction(
    config: RuntimeConfig,
    *,
    speaker: str,
    initial_transcript: str,
) -> str | None:
    if config.interaction_mode is not InteractionMode.AI_COUNSELOR_AI_CLIENT:
        return None
    participant = config.participants.get(speaker)
    if participant is None or participant.role != CLIENT_ROLE:
        return None
    initial_transcript = initial_transcript.strip()
    if not initial_transcript:
        return None
    return AI_COUNSELOR_INITIAL_CLIENT_OPENING_INSTRUCTION.format(
        initial_transcript=initial_transcript,
    )


def _human_counselor_initial_client_response_instruction(
    config: RuntimeConfig,
    *,
    speaker: str,
    turns: list[TurnRuntimeState],
) -> str | None:
    if config.interaction_mode is not InteractionMode.HUMAN_COUNSELOR_AI_CLIENT:
        return None
    participant = config.participants.get(speaker)
    if participant is None or participant.role != CLIENT_ROLE:
        return None
    initial_transcript = participant.initial_transcript.strip()
    if not initial_transcript:
        return None
    if not turns:
        return None
    last_turn = turns[-1]
    last_participant = config.participants.get(last_turn.speaker)
    if (
        last_participant is None
        or last_participant.role != COUNSELOR_ROLE
        or last_participant.actor_kind is not ActorKind.HUMAN
    ):
        return None
    if any(_participant_role(config, turn.speaker) == CLIENT_ROLE for turn in turns):
        return None
    return HUMAN_COUNSELOR_INITIAL_CLIENT_RESPONSE_INSTRUCTION.format(
        initial_transcript=initial_transcript,
    )


def _human_counselor_clients_deliberately_waited(
    decisions: Iterable[TimingDecision],
) -> bool:
    found_wait = False
    for decision in decisions:
        if decision.reason_code == "timing_decision_timeout":
            return False
        if decision.action is not TimingDecisionAction.WAIT:
            return False
        found_wait = True
    return found_wait


def _has_multiple_client_participants(config: RuntimeConfig) -> bool:
    return (
        sum(
            1
            for participant in config.participants.values()
            if participant.role == CLIENT_ROLE
        )
        >= 2
    )


def _missing_last_float(value: int | None) -> float:
    if value is None:
        return float("inf")
    return float(value)


def _priority_conflict_agent_ids(
    decisions: Iterable[TimingDecision],
    selected_agent_id: str,
) -> tuple[str, ...]:
    agent_ids = {selected_agent_id}
    agent_ids.update(_requesting_agent_ids(decisions))
    return tuple(sorted(agent_ids))


def _requesting_agent_ids(decisions: Iterable[TimingDecision]) -> tuple[str, ...]:
    seen: set[str] = set()
    requesting_agent_ids: list[str] = []
    for decision in decisions:
        if decision.action not in {
            TimingDecisionAction.REQUEST_MAIN_FLOOR,
            TimingDecisionAction.INTERRUPT,
        }:
            continue
        if decision.agent_id in seen:
            continue
        requesting_agent_ids.append(decision.agent_id)
        seen.add(decision.agent_id)
    return tuple(requesting_agent_ids)


def _timing_decision_for_agent(
    decisions: Iterable[TimingDecision],
    agent_id: str,
) -> TimingDecision | None:
    for decision in decisions:
        if decision.agent_id == agent_id:
            return decision
    return None


def _turn_start_delay_ms(
    decision: TimingDecision | None,
    *,
    rng: random.Random | None = None,
) -> int:
    if decision is None:
        return 0
    preferred_timing = _effective_preferred_timing(decision)
    min_ms, max_ms = _TURN_START_TIMING_DELAY_RANGES_MS[preferred_timing]
    if rng is None:
        return random.randint(min_ms, max_ms)
    return rng.randint(min_ms, max_ms)


def _effective_preferred_timing(decision: TimingDecision) -> str:
    preferred_timing = _normalize_preferred_timing(decision.preferred_timing)
    if preferred_timing is None:
        return "natural_pause"
    urgency = decision.urgency
    if (
        urgency is not None
        and urgency >= _TURN_START_HIGH_URGENCY_IMMEDIATE_THRESHOLD
    ):
        return "immediate"
    return preferred_timing


def _normalize_preferred_timing(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
    if normalized in _TURN_START_TIMING_DELAY_RANGES_MS:
        return normalized
    aliases = {
        "now": "immediate",
        "right_now": "immediate",
        "asap": "immediate",
        "natural": "natural_pause",
        "normal": "natural_pause",
        "short": "short_hold",
        "hold": "short_hold",
        "long": "long_hold",
    }
    return aliases.get(normalized)


def _replace_timing_decisions(
    decisions: Iterable[TimingDecision],
    replacements: Iterable[TimingDecision],
) -> list[TimingDecision]:
    replacement_by_agent_id = {decision.agent_id: decision for decision in replacements}
    updated_decisions: list[TimingDecision] = []
    for decision in decisions:
        updated_decisions.append(
            replacement_by_agent_id.pop(decision.agent_id, decision)
        )
    updated_decisions.extend(replacement_by_agent_id.values())
    return updated_decisions


def _turn_taking_conflict_prompt(
    *,
    input_transcript: str,
    conflict_agent_ids: tuple[str, ...],
    round_index: int,
    decisions: list[TimingDecision],
) -> str:
    conflict_ids = ", ".join(conflict_agent_ids)
    decision_lines = "\n".join(
        f"- {decision.agent_id}: {decision.action.value}"
        + (f" / {decision.reason_code}" if decision.reason_code else "")
        for decision in decisions
    )
    return "\n\n".join(
        [
            input_transcript.strip() or "（直前発話は空です。）",
            (
                "【ターンテイク競合】\n"
                f"同じ次ターンで複数のエージェントが主発話を希望しました: {conflict_ids}\n"
                f"再判断ラウンド: {round_index}\n"
                "あなたが今すぐ主発話を取る必要が高い場合だけ REQUEST_MAIN_FLOOR を返し、"
                "相手へ譲る場合は WAIT を返してください。"
                "発話途中の割り込みは使わず、次ターンの取得意思だけを判断してください。"
                "action に関わらず preferred_timing を immediate, natural_pause, "
                "short_hold, long_hold から選び、urgency も 0.0 から 1.0 で返してください。"
                "action に関わらず target も counselor, client_a, client_b から選んでください。"
                "action に関わらず explicitly_addressed も true/false で返してください。"
                "直前発話が現在の agent を名前、表示名、役割名、家族内呼称、視点指定で明示的に指名している場合だけ true です。"
                "explicitly_addressed は誰かが指名されたという意味ではなく、現在の agent_id 本人が指名された場合だけ true です。"
                "例: 妻役なら「奥さんは」「お母さんは」「妻としては」、夫役なら「旦那さんは」「お父さんは」「夫としては」などです。"
                "他参加者が明示指名されているだけなら false です。"
                "カウンセラーが特定の参加者へ質問した場合は、指名された当人の回答を最優先し、指名されていない配偶者は安全介入が必要な場合を除いて WAIT にしてください。"
                "自分が明示指名された質問なら、安全上の理由や明確に譲る理由がない限り REQUEST_MAIN_FLOOR を選びやすくしてください。"
                "自分以外が明示指名されている場合、安全介入、重大な誤解訂正、必要な短い補足以外では WAIT を選びやすくしてください。"
                "自分以外への明示質問に割り込んで REQUEST_MAIN_FLOOR を選ぶのは、その場で入らないと会話が不自然または危険になるほど必要性が高い場合に限ってください。"
                "カウンセラーへ返す、別クライアントへ話す、今は譲る、のいずれもあり得ます。"
                "共通プロンプト、プロフィール、直近履歴に基づいて、次に自分が話す必要があるかと自然な target を選んでください。"
                "target は会話の進行役ではなく、発話内容を主に誰へ向けるかです。"
                "クライアント役は、迷ったときに機械的に counselor を選ばないでください。"
                "counselor は、カウンセラーへの回答、質問、助言依頼、面談進行への反応が主な内容の場合に選んでください。"
                "カウンセラーが同席していても、配偶者へ確認、同意、相談、応答をする内容なら target はその別クライアントです。"
                "別クライアントへ直接話す強さは、シナリオの共通プロンプトとプロフィールに従ってください。"
                "クライアント役は、すでに答えていて新たな質問もなく、言えることが同じ説明や相手の同意への相づちだけなら WAIT を選んでください。"
                "自分への未回答の質問・確認には短い同意や「分からない」も回答になります。"
                "カウンセラー役は、直前のクライアント発話が別クライアントへの"
                "直接の質問・確認で、まだ相手からの回答を待っている場合、"
                "安全介入や明確な進行整理が不要なら WAIT を検討し、"
                "相手クライアントが応答する余地を残してください。"
                "カウンセラー役もクライアント役も、特定の相手を常に優先せず、現在の面談の流れに合う場合に REQUEST_MAIN_FLOOR を選んでください。"
                "直近履歴で同じ発話者順序の短いパターンが反復している場合、その順序を次ターンの既定として模倣しないでください。"
                "内容上必要な場合を除き、同じ二者往復や同じ三者サイクルが続くより、現在の発話内容に基づく自然な譲り先を優先してください。"
                "WAIT の場合は、もし fallback で自分が次に話すことになった場合の間合いを返してください。"
                "urgency が非常に高い場合は natural_pause ではなく immediate を選んでください。"
                "各 preferred_timing の具体的な待ち時間のばらつきはシステム側で付けます。"
                "質問を受けた直後は short_hold を選びやすく、難しい質問や言いにくい内容では long_hold を検討してください。"
                "質問ではない短い受け継ぎ、明確な補足や言い換え、安全上すぐに入る必要がある場合は immediate を選びやすくしてください。"
            ),
            "直前の判断:\n" + decision_lines,
        ]
    )


def _timing_decision_log_details(decision: TimingDecision) -> dict[str, Any]:
    return {
        "agent_id": decision.agent_id,
        "action": decision.action.value,
        "preferred_timing": decision.preferred_timing,
        "urgency": decision.urgency,
        "target": decision.target,
        "reason_code": decision.reason_code,
        "prepared_intent": decision.prepared_intent,
        "expires_after_ms": decision.expires_after_ms,
        "explicitly_addressed": decision.explicitly_addressed,
        "speech_ms_last_window": decision.speech_ms_last_window,
        "request_timestamp_ms": decision.request_timestamp_ms,
        "safety_intervention": decision.safety_intervention,
    }


def _speaker_label_candidates(config: RuntimeConfig, speaker: str) -> tuple[str, ...]:
    participant = config.participants.get(speaker)
    labels = [speaker]
    if participant is not None:
        labels.append(participant.display_name)
        if participant.role == "counselor":
            labels.append("カウンセラー")
        elif participant.role == CLIENT_ROLE:
            labels.append("クライアント")
    return tuple(label for label in labels if label)


async def _log_event(
    logger: AsyncLogger | None,
    state: TurnRuntimeState,
    event_type: str,
    details: dict | None = None,
) -> None:
    if logger is None:
        return
    await logger.log_event(
        RuntimeEvent(
            session_id=state.session_id,
            event_type=event_type,
            turn_id=state.turn_id,
            speaker=state.speaker,
            speaker_id=state.speaker,
            recipient_ids=state.recipient_ids,
            monotonic_time=time.monotonic(),
            details=details or {},
        )
    )


async def _log_generated_transcript(
    logger: AsyncLogger | None,
    state: TurnRuntimeState,
    *,
    speaker_display_name: str,
) -> None:
    if logger is None or not state.generated_text.strip():
        return
    await logger.log_transcript(
        TranscriptEvent(
            session_id=state.session_id,
            turn_id=state.turn_id,
            speaker=state.speaker,
            speaker_id=state.speaker,
            transcript_type="generated_final",
            text=state.generated_text,
            recipient_ids=state.recipient_ids,
            metadata={
                "speaker_display_name": speaker_display_name,
                "text_source": "generated_text",
            },
        )
    )


async def _publish_monitor_transcript(
    audio_bus: AudioBus | None,
    state: TurnRuntimeState,
    *,
    transcript_type: str,
    text: str,
    speaker_display_name: str,
    speaker_role: str,
    text_source: str = "generated_text",
) -> None:
    if audio_bus is None or not text:
        return
    await audio_bus.publish_monitor_event(
        TranscriptEvent(
            session_id=state.session_id,
            turn_id=state.turn_id,
            speaker=state.speaker,
            speaker_id=state.speaker,
            transcript_type=transcript_type,
            text=text,
            recipient_ids=state.recipient_ids,
            metadata={
                "speaker_display_name": speaker_display_name,
                "role": speaker_role,
                "text_source": text_source,
            },
        )
    )


async def _publish_monitor_transcript_interrupted(
    audio_bus: AudioBus | None,
    state: TurnRuntimeState,
    *,
    speaker_display_name: str,
    speaker_role: str,
    reason: str,
) -> None:
    if audio_bus is None:
        return
    await audio_bus.publish_monitor_event(
        TranscriptEvent(
            session_id=state.session_id,
            turn_id=state.turn_id,
            speaker=state.speaker,
            speaker_id=state.speaker,
            transcript_type="interrupted",
            text="",
            recipient_ids=state.recipient_ids,
            metadata={
                "speaker_display_name": speaker_display_name,
                "role": speaker_role,
                "text_source": "generated_text",
                "reason": reason,
            },
        )
    )


async def _notify_transcript_observer(
    observer: TranscriptObserver | None,
    transcript: TranscriptEvent,
) -> None:
    if observer is None:
        return
    result = observer(transcript)
    if inspect.isawaitable(result):
        await result


def _is_empty_human_audio_stream_error(error: RuntimeError) -> bool:
    message = str(error)
    return any(
        marker in message
        for marker in (
            "input_audio_buffer_commit_empty",
            "buffer too small",
            "buffer only has 0.00ms of audio",
            "audio stream ended before target turn audio was committed",
        )
    )


def _callable_accepts_keyword(candidate: Callable[..., Any], keyword: str) -> bool:
    try:
        signature = inspect.signature(candidate)
    except (TypeError, ValueError):
        return False
    return any(
        name == keyword or parameter.kind is inspect.Parameter.VAR_KEYWORD
        for name, parameter in signature.parameters.items()
    )


def _human_audio_input_to_chunk(
    request: HumanAudioInput,
    *,
    turn_id: int,
    speaker: str,
    sample_width_bits: int,
    delivery_mode: AudioDeliveryMode,
) -> AudioChunk:
    bytes_per_frame = max(1, (sample_width_bits // 8) * request.channels)
    frame_count = len(request.audio_bytes) // bytes_per_frame
    duration_ms = round(frame_count / request.sample_rate * 1000)
    return AudioChunk(
        session_id=request.session_id,
        turn_id=turn_id,
        speaker=speaker,
        chunk_index=0,
        pcm=request.audio_bytes,
        sample_rate=request.sample_rate,
        sample_width_bits=sample_width_bits,
        channels=request.channels,
        duration_ms=max(0, duration_ms),
        delivery_mode=delivery_mode,
        pace_after_delivery=False,
    )


def _human_audio_stream_chunk_to_audio_chunk(
    request: HumanAudioStreamStart,
    *,
    audio_bytes: bytes,
    chunk_index: int,
    turn_id: int,
    speaker: str,
    sample_width_bits: int,
    delivery_mode: AudioDeliveryMode,
) -> AudioChunk:
    bytes_per_frame = max(1, (sample_width_bits // 8) * request.channels)
    frame_count = len(audio_bytes) // bytes_per_frame
    duration_ms = round(frame_count / request.sample_rate * 1000)
    return AudioChunk(
        session_id=request.session_id,
        turn_id=turn_id,
        speaker=speaker,
        chunk_index=chunk_index,
        pcm=audio_bytes,
        sample_rate=request.sample_rate,
        sample_width_bits=sample_width_bits,
        channels=request.channels,
        duration_ms=max(0, duration_ms),
        delivery_mode=delivery_mode,
        pace_after_delivery=False,
    )


def _turn_audio_log_ref(turn_id: int, speaker: str) -> str:
    return f"internal/audio/turn_{turn_id:04d}_{speaker}.wav"


def _normalized_human_stt_final_text(text: str) -> str:
    normalized = text.strip()
    if normalized == "stt_final:":
        return ""
    return normalized


def _with_transcript_display_metadata(
    transcript: TranscriptEvent,
    *,
    participant: ParticipantConfig,
    recipient_ids: tuple[str, ...],
    text_source: str,
    source_audio_ref: str | None = None,
) -> TranscriptEvent:
    metadata = {
        **transcript.metadata,
        "speaker_display_name": participant.display_name,
        "role": participant.role,
        "actor_kind": participant.actor_kind.value,
        "text_source": text_source,
    }
    if source_audio_ref is not None:
        metadata["source_audio_ref"] = source_audio_ref
    return replace(
        transcript,
        speaker_id=participant.speaker_id,
        recipient_ids=recipient_ids,
        metadata=metadata,
    )


async def _publish_monitor_transcript_event(
    audio_bus: AudioBus | None,
    logger: AsyncLogger | None,
    transcript: TranscriptEvent,
) -> None:
    if not transcript.text:
        return
    if audio_bus is not None:
        await audio_bus.publish_monitor_event(transcript)
    if logger is not None:
        await logger.log_transcript(transcript)


async def _finalize_generated_turn(
    logger: AsyncLogger | None,
    state: TurnRuntimeState,
    *,
    audio_bus: AudioBus | None = None,
    speaker_display_name: str,
    speaker_role: str,
    publish_monitor_final: bool = True,
) -> None:
    final_text = state.generated_text
    state.audio_delivered_to_stt = True
    state.stt_final_transcript = final_text
    state.stt_final_monotonic = time.monotonic()
    if publish_monitor_final:
        await _publish_monitor_transcript(
            audio_bus,
            state,
            transcript_type="generated_final",
            text=final_text,
            speaker_display_name=speaker_display_name,
            speaker_role=speaker_role,
        )
    await _log_generated_transcript(
        logger,
        state,
        speaker_display_name=speaker_display_name,
    )
    await _log_event(
        logger,
        state,
        "generated_final",
        {"text": final_text, "text_source": "generated_text"},
    )
    state.mark_completed()
    state.completed_monotonic = time.monotonic()


async def _finalize_interrupted_generated_turn(
    logger: AsyncLogger | None,
    state: TurnRuntimeState,
    *,
    audio_bus: AudioBus | None = None,
    speaker_display_name: str,
    speaker_role: str,
) -> None:
    final_text = state.generated_text.strip()
    audio_log_path = state.audio_log_path or _turn_audio_log_ref(
        state.turn_id,
        state.speaker,
    )
    state.audio_log_path = audio_log_path
    if not final_text:
        await _publish_monitor_transcript_interrupted(
            audio_bus,
            state,
            speaker_display_name=speaker_display_name,
            speaker_role=speaker_role,
            reason="human_barge_in",
        )
        return
    state.generated_text = final_text
    state.audio_delivered_to_stt = True
    state.stt_final_transcript = final_text
    state.stt_final_monotonic = time.monotonic()
    await _publish_monitor_transcript(
        audio_bus,
        state,
        transcript_type="generated_interrupted",
        text=final_text,
        speaker_display_name=speaker_display_name,
        speaker_role=speaker_role,
    )
    if logger is not None:
        await logger.log_transcript(
            TranscriptEvent(
                session_id=state.session_id,
                turn_id=state.turn_id,
                speaker=state.speaker,
                speaker_id=state.speaker,
                transcript_type="generated_interrupted",
                text=final_text,
                recipient_ids=state.recipient_ids,
                metadata={
                    "speaker_display_name": speaker_display_name,
                    "role": speaker_role,
                    "text_source": "generated_text",
                    "interrupted": True,
                    "audio_log_path": audio_log_path,
                },
            )
        )
    await _log_event(
        logger,
        state,
        "generated_interrupted",
        {
            "text": final_text,
            "text_source": "generated_text",
            "interrupted": True,
            "audio_log_path": audio_log_path,
        },
    )


async def _prepare_final_realtime_fallback(
    logger: AsyncLogger | None,
    state: TurnRuntimeState,
    provisional: ProvisionalGeneration,
) -> float:
    await _log_event(
        logger,
        state,
        "provisional_llm_fallback",
        {
            "source_turn_id": provisional.source_turn_id,
            "source": provisional.source,
            "prediction_reason": provisional.prediction_reason,
            "candidate_speaker_ids": list(provisional.candidate_speaker_ids),
            "generation_modality": provisional.generation_modality,
            "reason": "provisional_generation_failed",
            "error_type": (
                type(provisional.error).__name__
                if provisional.error is not None
                else None
            ),
            "error_message": (
                str(provisional.error) if provisional.error is not None else ""
            ),
        },
    )
    state.generated_text = ""
    state.provisional_input_transcript = None
    state.used_provisional_generation = False
    state.provisional_generation_discard_reason = "provisional_generation_failed"
    state.tts_started = False
    state.tts_done = False
    state.audio_delivered_to_stt = False
    state.stt_final_transcript = None
    state.first_audio_chunk_at = None
    state.first_audio_chunk_monotonic = None
    llm_request_started_at = time.monotonic()
    state.llm_request_started_monotonic = llm_request_started_at
    await _log_event(
        logger,
        state,
        "llm_request_started",
        {"generation_source": "realtime_api_fallback"},
    )
    return llm_request_started_at


async def _log_metric(
    logger: AsyncLogger | None,
    state: TurnRuntimeState,
    metric_name: str,
    value: float,
    details: dict[str, Any] | None = None,
) -> None:
    if logger is None:
        return
    await logger.log_metric(
        RuntimeMetric(
            session_id=state.session_id,
            metric_name=metric_name,
            value=value,
            turn_id=state.turn_id,
            speaker=state.speaker,
            speaker_id=state.speaker,
            recipient_ids=state.recipient_ids,
            details=details or {},
        )
    )


def _elapsed_ms(start: float, end: float) -> float:
    return max(0.0, (end - start) * 1000)


async def _run_concurrently(
    *task_factories: Callable[[], Awaitable[None]],
) -> None:
    try:
        async with asyncio.TaskGroup() as task_group:
            for task_factory in task_factories:
                task_group.create_task(task_factory())
    except* GenerationInterruptedForHumanInput:
        raise GenerationInterruptedForHumanInput()


def _raise_if_generation_interrupted(
    generation_interrupt_requested: Callable[[], bool] | None,
) -> None:
    if generation_interrupt_requested is not None and generation_interrupt_requested():
        raise GenerationInterruptedForHumanInput()


async def _iter_agent_text(
    agent: AgentLike | StreamingAgentLike,
    *,
    input_transcript: str,
    turn_id: int,
    additional_instruction: str | None = None,
):
    if isinstance(agent, StreamingAgentLike):
        if additional_instruction is None:
            result = agent.stream_generate(
                input_transcript=input_transcript,
                turn_id=turn_id,
            )
        else:
            result = agent.stream_generate(
                input_transcript=input_transcript,
                turn_id=turn_id,
                additional_instruction=additional_instruction,
            )
        async for part in _aiter_text_result(result):
            yield part
        return

    if additional_instruction is None:
        result = agent.generate(input_transcript=input_transcript, turn_id=turn_id)
    else:
        result = agent.generate(
            input_transcript=input_transcript,
            turn_id=turn_id,
            additional_instruction=additional_instruction,
        )
    if inspect.isawaitable(result):
        result = await result
    async for part in _aiter_text_result(result):
        yield part


async def _iter_generation_text(
    agent: AgentLike | StreamingAgentLike,
    *,
    input_transcript: str,
    turn_id: int,
    provisional_generation: ProvisionalGeneration | None,
    additional_instruction: str | None = None,
) -> AsyncIterator[str]:
    if provisional_generation is not None:
        async for part in _iter_provisional_text(provisional_generation):
            yield part
        return

    async for part in _iter_agent_text(
        agent,
        input_transcript=input_transcript,
        turn_id=turn_id,
        additional_instruction=additional_instruction,
    ):
        yield part


async def _iter_provisional_text(provisional: ProvisionalGeneration) -> AsyncIterator[str]:
    while True:
        item = await provisional.queue.get()
        if item is _PROVISIONAL_TEXT_DONE:
            if provisional.error is not None:
                raise ProvisionalGenerationFailedError(
                    provisional
                ) from provisional.error
            return
        yield str(item)


async def _iter_provisional_realtime_events(
    provisional: ProvisionalGeneration,
) -> AsyncIterator[Any]:
    while True:
        item = await provisional.queue.get()
        if item is _PROVISIONAL_TEXT_DONE:
            if provisional.error is not None:
                raise ProvisionalGenerationFailedError(
                    provisional
                ) from provisional.error
            return
        yield item


async def _aiter_text_result(result: Any):
    if inspect.isawaitable(result):
        result = await result
    if hasattr(result, "__aiter__"):
        async for part in result:
            yield str(part)
        return
    for part in result:
        yield str(part)


def _is_realtime_speech_agent(agent: Any) -> bool:
    return callable(getattr(agent, "stream_audio_response", None))


def _is_realtime_audio_input_agent(agent: Any) -> bool:
    return callable(getattr(agent, "stream_audio_response_from_audio", None))


async def _iter_realtime_speech_events(
    agent: Any,
    *,
    session_id: str,
    turn_id: int,
    speaker: str,
    input_transcript: str,
    additional_instruction: str | None,
    format_input: bool = True,
    sample_rate: int,
    sample_width_bits: int,
    channels: int,
    delivery_mode: AudioDeliveryMode,
    response_instructions_observer: Callable[[Any], Any] | None = None,
) -> AsyncIterator[Any]:
    kwargs = {
        "session_id": session_id,
        "turn_id": turn_id,
        "speaker": speaker,
        "input_transcript": input_transcript,
        "sample_rate": sample_rate,
        "sample_width_bits": sample_width_bits,
        "channels": channels,
        "delivery_mode": delivery_mode,
    }
    if additional_instruction is not None:
        kwargs["additional_instruction"] = additional_instruction
    if not format_input:
        kwargs["format_input"] = False
    if (
        response_instructions_observer is not None
        and _callable_accepts_keyword(
            agent.stream_audio_response,
            "response_instructions_observer",
        )
    ):
        kwargs["response_instructions_observer"] = (
            response_instructions_observer
        )
    result = agent.stream_audio_response(**kwargs)
    if inspect.isawaitable(result):
        result = await result
    async for event in result:
        yield event


async def _iter_realtime_audio_input_speech_events(
    agent: Any,
    *,
    session_id: str,
    turn_id: int,
    speaker: str,
    input_audio: bytes,
    additional_instruction: str | None,
    sample_rate: int,
    sample_width_bits: int,
    channels: int,
    delivery_mode: AudioDeliveryMode,
    response_instructions_observer: Callable[[Any], Any] | None = None,
) -> AsyncIterator[Any]:
    kwargs = {
        "session_id": session_id,
        "turn_id": turn_id,
        "speaker": speaker,
        "input_audio": input_audio,
        "sample_rate": sample_rate,
        "sample_width_bits": sample_width_bits,
        "channels": channels,
        "delivery_mode": delivery_mode,
    }
    if additional_instruction is not None:
        kwargs["additional_instruction"] = additional_instruction
    if (
        response_instructions_observer is not None
        and _callable_accepts_keyword(
            agent.stream_audio_response_from_audio,
            "response_instructions_observer",
        )
    ):
        kwargs["response_instructions_observer"] = (
            response_instructions_observer
        )
    result = agent.stream_audio_response_from_audio(**kwargs)
    if inspect.isawaitable(result):
        result = await result
    async for event in result:
        yield event


def _response_instructions_logger(
    logger: AsyncLogger | None,
) -> Callable[[Any], Awaitable[None]] | None:
    if logger is None:
        return None

    async def log_record(record: Any) -> None:
        resolved_instructions = str(
            getattr(record, "resolved_instructions", "") or ""
        )
        await logger.log_response_instructions(
            ResponseInstructionsRecord(
                session_id=str(getattr(record, "session_id", "")),
                turn_id=int(getattr(record, "turn_id")),
                speaker=str(getattr(record, "speaker", "")),
                session_instructions=str(
                    getattr(record, "session_instructions", "") or ""
                ),
                participant_instructions=str(
                    getattr(record, "participant_instructions", "") or ""
                ),
                additional_instructions=str(
                    getattr(record, "additional_instructions", "") or ""
                ),
                resolved_instructions=resolved_instructions,
                instructions_sha256=hashlib.sha256(
                    resolved_instructions.encode("utf-8")
                ).hexdigest(),
                repeat_session_instructions=bool(
                    getattr(record, "repeat_session_instructions", False)
                ),
                input_text=str(getattr(record, "input_text", "") or ""),
                response_input=getattr(record, "response_input", None),
                response_conversation=getattr(record, "response_conversation", None),
            )
        )

    return log_record


def _realtime_speech_audio_chunk(event: Any) -> AudioChunk | None:
    if isinstance(event, AudioChunk):
        return event
    if isinstance(event, dict):
        audio_chunk = event.get("audio_chunk")
    else:
        audio_chunk = getattr(event, "audio_chunk", None)
    if audio_chunk is None:
        return None
    if not isinstance(audio_chunk, AudioChunk):
        raise TypeError("realtime speech event audio_chunk must be an AudioChunk")
    return audio_chunk


def _apply_speaker_audio_gain(chunk: AudioChunk, *, gain: float) -> AudioChunk:
    if gain == 1.0 or not chunk.pcm:
        return chunk
    if chunk.sample_width_bits != 16:
        raise ValueError("speaker audio gain currently supports 16-bit PCM only")
    pcm = _apply_pcm16le_gain(chunk.pcm, gain=gain)
    pcm = _declick_pcm16le_spikes(pcm)
    if pcm == chunk.pcm:
        return chunk
    return replace(chunk, pcm=pcm)


def _apply_pcm16le_gain(pcm: bytes, *, gain: float) -> bytes:
    if gain <= 0:
        raise ValueError("gain must be positive")
    sample_width_bytes = 2
    complete_byte_length = len(pcm) - (len(pcm) % sample_width_bytes)
    peak = 0
    for offset in range(0, complete_byte_length, sample_width_bytes):
        sample = int.from_bytes(pcm[offset : offset + 2], byteorder="little", signed=True)
        peak = max(peak, abs(sample))
    effective_gain = gain
    if peak > 0:
        effective_gain = min(gain, 32767 / peak)
    output = bytearray(pcm)
    for offset in range(0, complete_byte_length, sample_width_bytes):
        sample = int.from_bytes(pcm[offset : offset + 2], byteorder="little", signed=True)
        scaled = round(sample * effective_gain)
        output[offset : offset + 2] = int(scaled).to_bytes(
            2,
            byteorder="little",
            signed=True,
        )
    return bytes(output)


def _declick_pcm16le_spikes(
    pcm: bytes,
    *,
    jump_threshold: int = 8000,
) -> bytes:
    sample_width_bytes = 2
    complete_byte_length = len(pcm) - (len(pcm) % sample_width_bytes)
    sample_count = complete_byte_length // sample_width_bytes
    if sample_count < 32:
        return pcm

    samples = [
        int.from_bytes(pcm[offset : offset + 2], byteorder="little", signed=True)
        for offset in range(0, complete_byte_length, sample_width_bytes)
    ]
    output: list[int] | None = None
    for index in range(1, sample_count - 1):
        previous_sample = samples[index - 1]
        current_sample = samples[index]
        next_sample = samples[index + 1]
        # Smooth only abrupt sample-to-sample jumps; broad filtering audibly dulls voices.
        if (
            abs(current_sample - previous_sample) >= jump_threshold
            or abs(next_sample - current_sample) >= jump_threshold
        ):
            if output is None:
                output = samples.copy()
            output[index] = round(
                (0.25 * previous_sample)
                + (0.5 * current_sample)
                + (0.25 * next_sample)
            )

    if output is None:
        return pcm

    cleaned = bytearray(pcm)
    for index, sample in enumerate(output):
        offset = index * sample_width_bytes
        cleaned[offset : offset + 2] = int(sample).to_bytes(
            2,
            byteorder="little",
            signed=True,
        )
    return bytes(cleaned)


def _realtime_speech_text_delta(event: Any) -> str | None:
    if isinstance(event, AudioChunk):
        return None
    if isinstance(event, dict):
        value = (
            event.get("text_delta")
            or event.get("transcript_delta")
            or event.get("output_text_delta")
        )
    else:
        value = (
            getattr(event, "text_delta", None)
            or getattr(event, "transcript_delta", None)
            or getattr(event, "output_text_delta", None)
        )
    if value is None:
        return None
    return str(value)


def _realtime_speech_input_transcript_delta(event: Any) -> str | None:
    if isinstance(event, AudioChunk):
        return None
    if isinstance(event, dict):
        value = event.get("input_transcript_delta")
    else:
        value = getattr(event, "input_transcript_delta", None)
    if value is None:
        return None
    return str(value)


def _realtime_speech_input_transcript_completed(event: Any) -> str | None:
    if isinstance(event, AudioChunk):
        return None
    if isinstance(event, dict):
        value = (
            event.get("input_transcript_completed")
            if "input_transcript_completed" in event
            else event.get("input_transcript")
        )
    else:
        value = getattr(event, "input_transcript_completed", None)
        if value is None:
            value = getattr(event, "input_transcript", None)
    if value is None:
        return None
    return str(value)


async def _close_runtime_agents(agents: dict[str, AgentLike | StreamingAgentLike]) -> None:
    seen: set[int] = set()
    for agent in agents.values():
        agent_id = id(agent)
        if agent_id in seen:
            continue
        seen.add(agent_id)
        await _close_runtime_agent(agent)


async def _close_runtime_agent(agent: Any | None) -> None:
    if agent is None:
        return
    close = getattr(agent, "close", None)
    if close is None:
        return
    result = close()
    if inspect.isawaitable(result):
        await result


def _provisional_adoption_skip_reason(
    provisional: ProvisionalGeneration,
    *,
    response_target: str | None = None,
) -> str | None:
    if (response_target or "") != (provisional.response_target or ""):
        return "response_target_mismatch"
    if provisional.generation_modality != "realtime_audio":
        return None
    if provisional.status == "running":
        return "realtime_provisional_incomplete"
    return None


def _response_target_from_selection(
    selection: SpeakerSelection,
    *,
    config: RuntimeConfig | None = None,
    speaker: str | None = None,
    previous_speaker: str | None = None,
) -> str | None:
    fallback_target = _valid_response_target_for_generation(
        selection.response_target,
        config=config,
        speaker=speaker,
    )
    if fallback_target is not None:
        return fallback_target
    decision = selection.timing_decision
    decision_target = None
    if (
        decision is not None
        and decision.target is not None
        and not decision.target_defaulted
    ):
        decision_target = _valid_response_target_for_generation(
            decision.target,
            config=config,
            speaker=speaker,
        )
    if decision_target is not None:
        return decision_target
    _ = previous_speaker
    return None


def _previous_turn_target_hint_from_selection(
    selection: SpeakerSelection,
    *,
    config: RuntimeConfig,
    speaker: str,
) -> str:
    target = _previous_turn_target_speaker_from_selection(
        selection,
        config=config,
        speaker=speaker,
    )
    if target is None:
        return ""
    speaker_participant = config.participants.get(speaker)
    target_participant = config.participants.get(target)
    if speaker_participant is None or target_participant is None:
        return ""
    speaker_label = _participant_target_hint_label(speaker, speaker_participant)
    target_label = _participant_target_hint_label(target, target_participant)
    return "\n".join(
        [
            f"直前ターンでは、{speaker_label} が {target_label} を主な宛先として選んでいました。",
            "これは強制ではなく、次に発話を取る必要があるかを判断するための弱い参考情報です。",
            "宛先に選ばれた参加者は、直近履歴に照らして短く応答するか、今は待つかを判断してください。",
            "カウンセラー役は、安全介入や明確な進行整理が不要なら、宛先参加者が応答する余地を検討してください。",
        ]
    )


def _previous_turn_target_speaker_from_selection(
    selection: SpeakerSelection,
    *,
    config: RuntimeConfig,
    speaker: str,
) -> str | None:
    decision = selection.timing_decision
    if decision is None or decision.target_defaulted:
        return None
    target = (decision.target or "").strip()
    if not target or target == speaker:
        return None
    if target not in config.participants:
        return None
    return target


def _participant_target_hint_label(
    speaker_id: str,
    participant: ParticipantConfig,
) -> str:
    display_name = participant.display_name.strip() or speaker_id
    return f"{display_name}（{speaker_id}）"


def _fallback_response_target(
    *,
    config: RuntimeConfig,
    speaker: str,
    previous_speaker: str | None,
) -> str | None:
    _ = config, speaker, previous_speaker
    return None


def _valid_response_target_for_generation(
    target: str | None,
    *,
    config: RuntimeConfig | None = None,
    speaker: str | None = None,
) -> str | None:
    if target is None:
        return None
    target = target.strip()
    if not target:
        return None
    if speaker is not None and target == speaker:
        return None
    if config is not None:
        target_participant = config.participants.get(target)
        if target_participant is None:
            return None
        if speaker is not None:
            speaker_participant = config.participants.get(speaker)
            if speaker_participant is None:
                return None
    return target


def _response_target_context_text(
    config: RuntimeConfig,
    target: str | None,
) -> str | None:
    if not target:
        return None
    participant = config.participants.get(target)
    if participant is None:
        return None
    display_name = participant.display_name.strip() or target
    return f"今回の主な宛先: {display_name} ({target})"


def _response_target_instruction(
    config: RuntimeConfig,
    *,
    speaker: str,
    target: str | None,
) -> str | None:
    if not target or target == speaker:
        if (
            _participant_role(config, speaker) == CLIENT_ROLE
            and _has_multiple_client_participants(config)
        ):
            return "\n".join(
                [
                    "今回の主な宛先は固定されていません。",
                    "カウンセラーに返すことも、別クライアントへ直接話すこともできます。",
                    "共通プロンプト、プロフィール、直近履歴に基づいて自然な宛先と話し方を選んでください。",
                    "カウンセラーに向ける場合は敬語・丁寧語、別クライアントに向ける場合は関係性に合う口調を使ってください。",
                    "特定の参加者を常に優先せず、現在の面談の流れに合う応答にしてください。",
                ]
            )
        return None
    participant = config.participants.get(target)
    if participant is None:
        return None
    display_name = participant.display_name.strip() or target
    speaker_role = _participant_role(config, speaker)
    lines = [
        f"今回の主な宛先は「{display_name}（{target}）」です。",
        "直前文脈で自然なら、その相手へ直接反応してください。",
    ]
    if speaker_role == CLIENT_ROLE and participant.role == CLIENT_ROLE:
        lines.extend(
            [
                "関係性、口調、どの程度相手に話すかは、共通プロンプトとプロフィールに従ってください。",
            ]
        )
    elif speaker_role == CLIENT_ROLE and participant.role == COUNSELOR_ROLE:
        lines.extend(
            [
                "カウンセラーに向けて敬語・丁寧語で話してください。",
            ]
        )
    elif speaker_role == COUNSELOR_ROLE and participant.role == CLIENT_ROLE:
        lines.append(
            "そのクライアントに向けて、カウンセラーとして応答してください。"
        )
    if speaker_role == COUNSELOR_ROLE:
        lines.append(
            "応答の内容・長さ・話し方は、設定されたカウンセラープロンプトに従ってください。"
        )
    lines.append("発話の先頭に宛名や話者名を付けないでください。")
    return "\n".join(lines)


def _same_speaker_continuation_instruction(
    config: RuntimeConfig,
    *,
    speaker: str,
    previous_speaker: str | None,
) -> str | None:
    if previous_speaker != speaker:
        return None
    participant = config.participants.get(speaker)
    display_name = (
        participant.display_name.strip()
        if participant is not None and participant.display_name.strip()
        else speaker
    )
    return "\n".join(
        [
            f"直前発話者も今回の話者も「{display_name}（{speaker}）」です。",
            "直前発話を他者からの発話として受けないでください。",
            "同じ内容の言い換えや同意表現から始めず、続ける必要がある場合だけ新しい情報を短く足してください。",
        ]
    )


def _combine_additional_instructions(*instructions: str | None) -> str | None:
    parts = [
        instruction.strip()
        for instruction in instructions
        if instruction and instruction.strip()
    ]
    if not parts:
        return None
    return "\n\n".join(parts)


def _provisional_input_matches_final(provisional_input: str, final_transcript: str) -> bool:
    normalized_provisional = _normalize_transcript_for_prefetch(provisional_input)
    normalized_final = _normalize_transcript_for_prefetch(final_transcript)
    if not normalized_provisional or not normalized_final:
        return False
    return normalized_final.startswith(normalized_provisional)


def _should_refresh_timing_signal_decisions(
    *,
    signal_text: str,
    final_transcript: str,
) -> bool:
    normalized_signal = _normalize_transcript_for_prefetch(signal_text)
    normalized_final = _normalize_transcript_for_prefetch(final_transcript)
    return normalized_signal != normalized_final


def _normalize_transcript_for_prefetch(text: str) -> str:
    return "".join(str(text).split())
