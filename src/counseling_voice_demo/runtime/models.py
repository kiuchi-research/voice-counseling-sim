from __future__ import annotations

from dataclasses import dataclass, field
from collections.abc import Mapping, Sequence
from datetime import datetime
from enum import StrEnum
from typing import Any
from uuid import uuid4

from counseling_voice_demo.models import utc_now
from counseling_voice_demo.runtime.speaker_selection import (
    LEGACY_ALTERNATING_SEQUENCE,
    FIXED_ROUND_ROBIN_POLICY,
    SUPPORTED_SPEAKER_SELECTION_POLICIES,
    TURN_BOUNDARY_TIMING_POLICY,
)


class AudioDeliveryMode(StrEnum):
    REAL_TIME = "real_time"
    ACCELERATED = "accelerated"


class RuntimePhase(StrEnum):
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    STOPPED = "stopped"
    ERROR = "error"


class RuntimeStopCondition(StrEnum):
    TURNS = "turns"
    ELAPSED_TIME = "elapsed_time"


class InteractionMode(StrEnum):
    AI_COUNSELOR_AI_CLIENT = "ai_counselor_ai_client"
    HUMAN_COUNSELOR_AI_CLIENT = "human_counselor_ai_client"
    AI_COUNSELOR_HUMAN_CLIENT = "ai_counselor_human_client"


class ParticipantMode(StrEnum):
    ONE_CLIENT = "one_client"
    TWO_CLIENTS = "two_clients"


class ActorKind(StrEnum):
    AI = "ai"
    HUMAN = "human"


SUPPORTED_HUMAN_INPUT_MODES = {"human_mic_stt", "human_text"}
SUPPORTED_HUMAN_RECORDING_MODES = {"push_to_talk", "vad_auto", "browser_vad"}
SUPPORTED_HUMAN_STT_SUBMIT_POLICIES = {"auto_on_final"}
MIN_REALTIME_OUTPUT_SPEED = 0.25
MAX_REALTIME_OUTPUT_SPEED = 1.5


def validate_human_client_configuration(
    *,
    interaction_mode: str,
    participant_mode: str,
    participants: Mapping[str, Any],
    speaker_selection_policy: str,
    fixed_speaker_sequence: Sequence[str],
    initial_client_transcript: str,
) -> None:
    if interaction_mode != InteractionMode.AI_COUNSELOR_HUMAN_CLIENT:
        return
    if participant_mode != ParticipantMode.ONE_CLIENT:
        raise ValueError(
            "ai_counselor_human_client requires participant_mode=one_client"
        )
    if set(participants) != {"counselor", "client"}:
        raise ValueError("ai_counselor_human_client requires counselor and client only")
    for speaker_id, actor_kind in (
        ("counselor", ActorKind.AI),
        ("client", ActorKind.HUMAN),
    ):
        participant = participants[speaker_id]
        if participant.role != speaker_id or participant.actor_kind != actor_kind:
            raise ValueError(
                f"participants.{speaker_id} requires role={speaker_id}, actor_kind={actor_kind}"
            )
    if speaker_selection_policy != FIXED_ROUND_ROBIN_POLICY:
        raise ValueError(
            "ai_counselor_human_client requires speaker_selection_policy=fixed_round_robin"
        )
    if tuple(fixed_speaker_sequence) != LEGACY_ALTERNATING_SEQUENCE:
        raise ValueError(
            "ai_counselor_human_client requires fixed_speaker_sequence=[counselor, client]"
        )
    if initial_client_transcript.strip():
        raise ValueError("human client initial_client_transcript must be empty")
    client = participants["client"]
    for field_name in (
        "prompt_path",
        "prompt_source",
        "public_profile_path",
        "public_profile_source",
        "private_profile_path",
        "private_profile_source",
        "voice",
        "realtime_output_speed",
        "initial_transcript",
    ):
        if getattr(client, field_name, None):
            raise ValueError(f"human client must not have {field_name}")


@dataclass(frozen=True)
class ParticipantConfig:
    speaker_id: str
    role: str
    display_name: str
    actor_kind: ActorKind | str = ActorKind.AI
    prompt_path: str | None = None
    public_profile_path: str | None = None
    private_profile_path: str | None = None
    prompt_source: str | None = None
    public_profile_source: str | None = None
    private_profile_source: str | None = None
    initial_transcript: str = ""
    voice: str | None = None
    realtime_output_speed: float | None = None

    def __post_init__(self) -> None:
        if not self.speaker_id:
            raise ValueError("participant speaker_id must not be empty")
        if self.role not in {"counselor", "client"}:
            raise ValueError("participant role must be counselor or client")
        if not self.display_name:
            raise ValueError("participant display_name must not be empty")
        if not isinstance(self.actor_kind, ActorKind):
            try:
                object.__setattr__(self, "actor_kind", ActorKind(str(self.actor_kind)))
            except ValueError as exc:
                raise ValueError("participant actor_kind must be ai or human") from exc
        if (
            self.realtime_output_speed is not None
            and not MIN_REALTIME_OUTPUT_SPEED
            <= self.realtime_output_speed
            <= MAX_REALTIME_OUTPUT_SPEED
        ):
            raise ValueError(
                "participant realtime_output_speed must be between "
                f"{MIN_REALTIME_OUTPUT_SPEED} and {MAX_REALTIME_OUTPUT_SPEED}"
            )


@dataclass(frozen=True)
class HumanAudioInput:
    session_id: str
    audio_bytes: bytes
    sample_rate: int
    channels: int
    recording_mode: str = "push_to_talk"
    recipient_ids: tuple[str, ...] = field(default_factory=tuple)
    client_message_id: str | None = None

    def __post_init__(self) -> None:
        if not self.session_id.strip():
            raise ValueError("session_id must not be empty")
        if not isinstance(self.audio_bytes, bytes) or not self.audio_bytes:
            raise ValueError("audio_bytes must not be empty")
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if self.channels <= 0:
            raise ValueError("channels must be positive")
        if self.recording_mode not in SUPPORTED_HUMAN_RECORDING_MODES:
            raise ValueError("recording_mode must be push_to_talk, vad_auto, or browser_vad")
        object.__setattr__(
            self,
            "recipient_ids",
            _normalize_non_empty_string_tuple(
                self.recipient_ids,
                field_name="recipient_ids",
            ),
        )
        if self.client_message_id is not None and not self.client_message_id.strip():
            raise ValueError("client_message_id must not be empty")


@dataclass(frozen=True)
class HumanTurnInput:
    session_id: str
    text: str
    recipient_ids: tuple[str, ...]
    input_mode: str = "human_mic_stt"
    recording_mode: str = "push_to_talk"
    source_audio_ref: str | None = None
    client_message_id: str | None = None

    def __post_init__(self) -> None:
        if not self.session_id.strip():
            raise ValueError("session_id must not be empty")
        text = self.text.strip()
        if not text:
            raise ValueError("text must not be empty")
        object.__setattr__(self, "text", text)
        object.__setattr__(
            self,
            "recipient_ids",
            _normalize_non_empty_string_tuple(
                self.recipient_ids,
                field_name="recipient_ids",
            ),
        )
        if self.input_mode not in SUPPORTED_HUMAN_INPUT_MODES:
            raise ValueError("input_mode must be human_mic_stt or human_text")
        if self.recording_mode not in SUPPORTED_HUMAN_RECORDING_MODES:
            raise ValueError("recording_mode must be push_to_talk, vad_auto, or browser_vad")
        if self.source_audio_ref is not None and not self.source_audio_ref.strip():
            raise ValueError("source_audio_ref must not be empty")
        if self.client_message_id is not None and not self.client_message_id.strip():
            raise ValueError("client_message_id must not be empty")


@dataclass(frozen=True)
class HumanAudioStreamStart:
    session_id: str
    sample_rate: int
    channels: int
    recording_mode: str = "push_to_talk"
    recipient_ids: tuple[str, ...] = field(default_factory=tuple)
    client_message_id: str | None = None

    def __post_init__(self) -> None:
        if not self.session_id.strip():
            raise ValueError("session_id must not be empty")
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if self.channels <= 0:
            raise ValueError("channels must be positive")
        if self.recording_mode not in SUPPORTED_HUMAN_RECORDING_MODES:
            raise ValueError("recording_mode must be push_to_talk, vad_auto, or browser_vad")
        object.__setattr__(
            self,
            "recipient_ids",
            tuple(recipient_id.strip() for recipient_id in self.recipient_ids),
        )
        if not self.recipient_ids or any(not item for item in self.recipient_ids):
            raise ValueError("recipient_ids must not be empty")
        if self.client_message_id is not None and not self.client_message_id.strip():
            raise ValueError("client_message_id must not be empty")


@dataclass(frozen=True)
class AudioChunk:
    session_id: str
    turn_id: int
    speaker: str
    chunk_index: int
    pcm: bytes
    sample_rate: int = 24000
    sample_width_bits: int = 16
    channels: int = 1
    duration_ms: int = 0
    created_at: datetime = field(default_factory=utc_now)
    delivery_mode: AudioDeliveryMode = AudioDeliveryMode.ACCELERATED
    pace_after_delivery: bool = True
    audio_role: str = "main"

    def __post_init__(self) -> None:
        if not isinstance(self.delivery_mode, AudioDeliveryMode):
            object.__setattr__(
                self,
                "delivery_mode",
                AudioDeliveryMode(str(self.delivery_mode)),
            )


@dataclass(frozen=True)
class EndOfAudio:
    session_id: str | None = None
    turn_id: int | None = None
    speaker: str | None = None
    created_at: datetime = field(default_factory=utc_now)
    audio_role: str = "main"


AudioBusItem = AudioChunk | EndOfAudio


@dataclass
class TurnRuntimeState:
    session_id: str
    turn_id: int
    speaker: str
    input_transcript: str
    recipient_ids: tuple[str, ...] = ()
    generated_text: str = ""
    tts_started: bool = False
    first_audio_chunk_at: datetime | None = None
    tts_done: bool = False
    audio_delivered_to_stt: bool = False
    stt_partial_transcripts: list[str] = field(default_factory=list)
    stt_final_transcript: str | None = None
    listener_stt_partial_transcripts: dict[str, list[str]] = field(default_factory=dict)
    listener_stt_final_transcripts: dict[str, str] = field(default_factory=dict)
    provisional_input_transcript: str | None = None
    used_provisional_generation: bool = False
    provisional_generation_discard_reason: str | None = None
    audio_log_path: str | None = None
    started_at: datetime = field(default_factory=utc_now)
    completed_at: datetime | None = None
    llm_request_started_monotonic: float | None = None
    first_audio_chunk_monotonic: float | None = None
    stt_commit_sent_monotonic: float | None = None
    stt_final_monotonic: float | None = None
    completed_monotonic: float | None = None

    @property
    def can_advance(self) -> bool:
        return (
            self.tts_done
            and self.audio_delivered_to_stt
            and self.stt_final_transcript is not None
        )

    def mark_completed(self) -> None:
        if not self.can_advance:
            raise RuntimeError(
                "turn cannot complete before tts_done, audio_delivered_to_stt, and stt_final_transcript"
            )
        self.completed_at = utc_now()


@dataclass(frozen=True)
class TranscriptEvent:
    session_id: str
    turn_id: int
    speaker: str
    transcript_type: str
    text: str
    created_at: datetime = field(default_factory=utc_now)
    speaker_id: str | None = None
    recipient_ids: tuple[str, ...] = field(default_factory=tuple)
    listener_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.speaker_id is None:
            object.__setattr__(self, "speaker_id", self.speaker)
        object.__setattr__(self, "recipient_ids", tuple(self.recipient_ids))


@dataclass(frozen=True)
class RuntimeEvent:
    session_id: str
    event_type: str
    turn_id: int | None = None
    speaker: str | None = None
    created_at: datetime = field(default_factory=utc_now)
    monotonic_time: float | None = None
    speaker_id: str | None = None
    recipient_ids: tuple[str, ...] = field(default_factory=tuple)
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.speaker_id is None:
            object.__setattr__(self, "speaker_id", self.speaker)
        object.__setattr__(self, "recipient_ids", tuple(self.recipient_ids))


@dataclass(frozen=True)
class PromptDirectorAttemptRecord:
    session_id: str
    turn_id: int
    speaker_id: str
    details: dict[str, Any]
    created_at: datetime = field(default_factory=utc_now)


@dataclass(frozen=True)
class ResponseInstructionsRecord:
    session_id: str
    turn_id: int
    speaker: str
    session_instructions: str
    participant_instructions: str
    additional_instructions: str
    resolved_instructions: str
    instructions_sha256: str
    repeat_session_instructions: bool
    request_type: str = "response.create"
    created_at: datetime = field(default_factory=utc_now)
    speaker_id: str | None = None
    input_text: str = ""
    response_input: list[dict[str, Any]] | None = None
    response_conversation: str | None = None

    def __post_init__(self) -> None:
        if self.speaker_id is None:
            object.__setattr__(self, "speaker_id", self.speaker)


@dataclass(frozen=True)
class RuntimeMetric:
    session_id: str
    metric_name: str
    value: float
    unit: str = "ms"
    turn_id: int | None = None
    speaker: str | None = None
    created_at: datetime = field(default_factory=utc_now)
    speaker_id: str | None = None
    recipient_ids: tuple[str, ...] = field(default_factory=tuple)
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.speaker_id is None:
            object.__setattr__(self, "speaker_id", self.speaker)
        object.__setattr__(self, "recipient_ids", tuple(self.recipient_ids))


@dataclass(frozen=True)
class RuntimeConfig:
    session_id: str = field(default_factory=lambda: f"session_{uuid4().hex[:12]}")
    interaction_mode: InteractionMode = InteractionMode.AI_COUNSELOR_AI_CLIENT
    participant_mode: ParticipantMode = ParticipantMode.ONE_CLIENT
    human_input_mode: str = "push_to_talk"
    human_stt_submit_policy: str = "auto_on_final"
    human_interrupts_enabled: bool = False
    stop_condition: RuntimeStopCondition = RuntimeStopCondition.TURNS
    max_turns: int = 2
    max_elapsed_seconds: float | None = None
    closing_start_elapsed_seconds: float | None = None
    force_stop_after_closing_turns: int = 5
    initial_client_transcript: str = "娘のことで相談があります"
    audio_delivery_mode: AudioDeliveryMode = AudioDeliveryMode.ACCELERATED
    sample_rate: int = 24000
    sample_width_bits: int = 16
    channels: int = 1
    speaker_audio_gains: dict[str, float] = field(
        default_factory=lambda: {"counselor": 1.0, "client": 1.0}
    )
    participants: dict[str, ParticipantConfig] = field(
        default_factory=lambda: {
            "counselor": ParticipantConfig(
                speaker_id="counselor",
                role="counselor",
                display_name="カウンセラー",
            ),
            "client": ParticipantConfig(
                speaker_id="client",
                role="client",
                display_name="クライアント",
            ),
        }
    )
    shared_case: str = ""
    speaker_selection_policy: str = TURN_BOUNDARY_TIMING_POLICY
    fixed_speaker_sequence: tuple[str, ...] = LEGACY_ALTERNATING_SEQUENCE
    tts_chunk_comma_min_chars: int = 20
    tts_chunk_soft_max_chars: int = 40
    stt_partial_prefetch: bool = False
    stt_partial_prefetch_min_chars: int = 8
    overlap_grace_ms: int = 200
    unresolved_overlap_limit_ms: int = 800
    turn_taking_signal_min_chars: int = 40
    turn_taking_decision_timeout_ms: int = 2000
    turn_taking_max_reconsider_rounds: int = 1
    conversation_context_recent_turns: int = 8
    session_summary_trigger_completed_turns: int = 12
    session_summary_update_interval_turns: int = 6

    def __post_init__(self) -> None:
        if not isinstance(self.interaction_mode, InteractionMode):
            object.__setattr__(
                self,
                "interaction_mode",
                InteractionMode(str(self.interaction_mode)),
            )
        if not isinstance(self.participant_mode, ParticipantMode):
            object.__setattr__(
                self,
                "participant_mode",
                ParticipantMode(str(self.participant_mode)),
            )
        if self.human_input_mode not in {"push_to_talk", "vad_auto"}:
            raise ValueError("human_input_mode must be push_to_talk or vad_auto")
        if self.human_stt_submit_policy not in SUPPORTED_HUMAN_STT_SUBMIT_POLICIES:
            raise ValueError("human_stt_submit_policy must be auto_on_final")
        if not isinstance(self.human_interrupts_enabled, bool):
            raise ValueError("human_interrupts_enabled must be a boolean")
        if not isinstance(self.stop_condition, RuntimeStopCondition):
            object.__setattr__(
                self,
                "stop_condition",
                RuntimeStopCondition(str(self.stop_condition)),
            )
        if not isinstance(self.audio_delivery_mode, AudioDeliveryMode):
            object.__setattr__(
                self,
                "audio_delivery_mode",
                AudioDeliveryMode(str(self.audio_delivery_mode)),
            )
        if self.speaker_selection_policy not in SUPPORTED_SPEAKER_SELECTION_POLICIES:
            supported = ", ".join(sorted(SUPPORTED_SPEAKER_SELECTION_POLICIES))
            raise ValueError(
                f"speaker_selection_policy must be one of: {supported}"
            )
        if isinstance(self.fixed_speaker_sequence, str):
            raise ValueError("fixed_speaker_sequence must be a sequence of speaker ids")
        try:
            normalized_fixed_speaker_sequence = tuple(self.fixed_speaker_sequence)
        except TypeError as exc:
            raise ValueError(
                "fixed_speaker_sequence must be a sequence of speaker ids"
            ) from exc
        if not normalized_fixed_speaker_sequence:
            raise ValueError("fixed_speaker_sequence must not be empty")
        if any(
            not isinstance(speaker_id, str) or not speaker_id
            for speaker_id in normalized_fixed_speaker_sequence
        ):
            raise ValueError(
                "fixed_speaker_sequence must contain non-empty speaker ids"
            )
        object.__setattr__(
            self,
            "fixed_speaker_sequence",
            normalized_fixed_speaker_sequence,
        )
        if self.max_turns <= 0:
            raise ValueError("max_turns must be positive")
        if self.max_elapsed_seconds is not None and self.max_elapsed_seconds <= 0:
            raise ValueError("max_elapsed_seconds must be positive when provided")
        if (
            self.closing_start_elapsed_seconds is not None
            and self.closing_start_elapsed_seconds < 0
        ):
            raise ValueError(
                "closing_start_elapsed_seconds must be non-negative when provided"
            )
        if self.force_stop_after_closing_turns <= 0:
            raise ValueError("force_stop_after_closing_turns must be positive")
        if (
            self.stop_condition is RuntimeStopCondition.ELAPSED_TIME
            and self.max_elapsed_seconds is None
        ):
            raise ValueError(
                "max_elapsed_seconds is required when stop_condition is elapsed_time"
            )
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if self.sample_width_bits not in {8, 16, 24, 32}:
            raise ValueError("sample_width_bits must be a supported PCM width")
        if self.channels <= 0:
            raise ValueError("channels must be positive")
        for speaker, gain in self.speaker_audio_gains.items():
            if gain <= 0:
                raise ValueError(f"speaker audio gain must be positive: {speaker}")
        normalized_participants: dict[str, ParticipantConfig] = {}
        for participant_id, participant in self.participants.items():
            if participant.speaker_id != participant_id:
                raise ValueError(
                    "participant speaker_id must match its RuntimeConfig participants key"
                )
            normalized_participants[participant_id] = participant
        if "counselor" not in normalized_participants:
            raise ValueError("participants must include counselor")
        if normalized_participants["counselor"].role != "counselor":
            raise ValueError("participants.counselor.role must be counselor")
        if not any(
            participant.role == "client"
            for participant_id, participant in normalized_participants.items()
            if participant_id != "counselor"
        ):
            raise ValueError("participants must include at least one client")
        if self.tts_chunk_comma_min_chars <= 0:
            raise ValueError("tts_chunk_comma_min_chars must be positive")
        if self.tts_chunk_soft_max_chars <= 0:
            raise ValueError("tts_chunk_soft_max_chars must be positive")
        if self.stt_partial_prefetch_min_chars <= 0:
            raise ValueError("stt_partial_prefetch_min_chars must be positive")
        if self.overlap_grace_ms < 0:
            raise ValueError("overlap_grace_ms must be non-negative")
        if self.unresolved_overlap_limit_ms <= 0:
            raise ValueError("unresolved_overlap_limit_ms must be positive")
        if self.turn_taking_signal_min_chars <= 0:
            raise ValueError("turn_taking_signal_min_chars must be positive")
        if self.turn_taking_decision_timeout_ms <= 0:
            raise ValueError("turn_taking_decision_timeout_ms must be positive")
        if self.turn_taking_max_reconsider_rounds < 0:
            raise ValueError("turn_taking_max_reconsider_rounds must be non-negative")
        if self.conversation_context_recent_turns <= 0:
            raise ValueError("conversation_context_recent_turns must be positive")
        if self.session_summary_trigger_completed_turns <= 0:
            raise ValueError("session_summary_trigger_completed_turns must be positive")
        if self.session_summary_update_interval_turns <= 0:
            raise ValueError("session_summary_update_interval_turns must be positive")
        validate_human_client_configuration(
            interaction_mode=self.interaction_mode,
            participant_mode=self.participant_mode,
            participants=self.participants,
            speaker_selection_policy=self.speaker_selection_policy,
            fixed_speaker_sequence=self.fixed_speaker_sequence,
            initial_client_transcript=self.initial_client_transcript,
        )

    @property
    def human_speaker_id(self) -> str | None:
        return next(
            (
                speaker
                for speaker, participant in self.participants.items()
                if participant.actor_kind is ActorKind.HUMAN
            ),
            None,
        )

    def public_participants(self) -> dict[str, dict[str, str]]:
        return {
            speaker: {
                "role": participant.role,
                "actor_kind": participant.actor_kind.value,
                "display_name": participant.display_name,
            }
            for speaker, participant in self.participants.items()
        }


@dataclass(frozen=True)
class RuntimeStatus:
    session_id: str
    phase: RuntimePhase
    pause_reason: str | None = None
    current_turn_id: int | None = None
    current_speaker: str | None = None
    completed_turns: int = 0
    playback_completed_turns: int = 0
    last_event_type: str | None = None
    closing_started: bool = False
    closing_started_turn_id: int | None = None
    closing_count_started_turn_id: int | None = None
    awaiting_human_input: bool = False
    active_speaker_id: str | None = None
    pending_human_turn_allowed: bool = False
    human_input_state: str | None = None
    error_message: str | None = None
    interaction_mode: str | None = None
    participants: dict[str, dict[str, str]] = field(default_factory=dict)
    human_speaker_id: str | None = None
    human_recipient_ids: tuple[str, ...] = ()


def _normalize_non_empty_string_tuple(
    values: tuple[str, ...],
    *,
    field_name: str,
) -> tuple[str, ...]:
    if isinstance(values, str):
        raise ValueError(f"{field_name} must be a sequence of strings")
    try:
        normalized = tuple(str(value).strip() for value in values)
    except TypeError as exc:
        raise ValueError(f"{field_name} must be a sequence of strings") from exc
    if not normalized or any(not value for value in normalized):
        raise ValueError(f"{field_name} must contain non-empty strings")
    return normalized
