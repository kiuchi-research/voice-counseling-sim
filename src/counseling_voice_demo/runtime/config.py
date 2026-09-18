from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Literal

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from counseling_voice_demo.runtime.speaker_selection import (
    SUPPORTED_SPEAKER_SELECTION_POLICIES,
    TURN_BOUNDARY_TIMING_POLICY,
)
from counseling_voice_demo.runtime.models import (
    MAX_REALTIME_OUTPUT_SPEED,
    MIN_REALTIME_OUTPUT_SPEED,
    validate_human_client_configuration,
)


ROOT_DIR = Path(__file__).resolve().parents[3]
DEFAULT_RUNTIME_CONFIG_PATH = ROOT_DIR / "config" / "runtime_config.yaml"
ENV_DEFAULT_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*):-([^}]*)\}")
SUPPORTED_AUDIO_DELIVERY_MODES = {"real_time", "accelerated"}
SUPPORTED_RUNTIME_BACKENDS = {"fake", "openai"}
SUPPORTED_STOP_CONDITIONS = {"turns", "elapsed_time"}
SUPPORTED_PARTICIPANT_ROLES = {"counselor", "client"}
SUPPORTED_INTERACTION_MODES = {
    "ai_counselor_ai_client",
    "human_counselor_ai_client",
    "ai_counselor_human_client",
}
SUPPORTED_PARTICIPANT_MODES = {"one_client", "two_clients"}
SUPPORTED_ACTOR_KINDS = {"ai", "human"}
SUPPORTED_HUMAN_INPUT_MODES = {"push_to_talk", "vad_auto"}
SUPPORTED_HUMAN_STT_SUBMIT_POLICIES = {"auto_on_final"}


class RuntimeConfigLoadError(RuntimeError):
    pass


class RuntimeSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    engine: str
    backend: str = "fake"
    control_api: str
    control_host: str
    control_port: int
    monitor_transport: str
    audio_delivery_mode: str
    turn_boundary_policy: str
    interaction_mode: str = "ai_counselor_ai_client"
    participant_mode: str = "one_client"
    human_input_mode: str = "push_to_talk"
    human_stt_submit_policy: str = "auto_on_final"
    human_interrupts_enabled: bool = False
    stop_condition: str = "turns"
    max_turns: int = 2
    max_elapsed_seconds: float | None = None
    closing_start_elapsed_seconds: float | None = None
    force_stop_after_closing_turns: int = 5
    initial_client_transcript: str = "娘のことで相談があります"
    speaker_selection_policy: str = TURN_BOUNDARY_TIMING_POLICY
    fixed_speaker_sequence: list[str] | None = None
    overlap_grace_ms: int = 200
    unresolved_overlap_limit_ms: int = 800
    turn_taking_signal_min_chars: int = 40
    turn_taking_decision_timeout_ms: int = 2000
    turn_taking_max_reconsider_rounds: int = 1
    conversation_context_recent_turns: int = 8
    session_summary_trigger_completed_turns: int = 12
    session_summary_update_interval_turns: int = 6
    prompt_director_enabled: bool = True
    prompt_director_max_retries: int = Field(default=2, ge=0, le=5, strict=True)

    @field_validator("audio_delivery_mode")
    @classmethod
    def audio_delivery_mode_is_supported(cls, value: str) -> str:
        if value not in SUPPORTED_AUDIO_DELIVERY_MODES:
            raise ValueError("audio_delivery_mode must be one of real_time, accelerated")
        return value

    @field_validator("backend")
    @classmethod
    def backend_is_supported(cls, value: str) -> str:
        if value not in SUPPORTED_RUNTIME_BACKENDS:
            raise ValueError("backend must be one of fake, openai")
        return value

    @field_validator("stop_condition")
    @classmethod
    def stop_condition_is_supported(cls, value: str) -> str:
        if value not in SUPPORTED_STOP_CONDITIONS:
            raise ValueError("stop_condition must be one of turns, elapsed_time")
        return value

    @field_validator("speaker_selection_policy")
    @classmethod
    def speaker_selection_policy_is_supported(cls, value: str) -> str:
        if value not in SUPPORTED_SPEAKER_SELECTION_POLICIES:
            supported = ", ".join(sorted(SUPPORTED_SPEAKER_SELECTION_POLICIES))
            raise ValueError(f"speaker_selection_policy must be one of: {supported}")
        return value

    @field_validator("interaction_mode")
    @classmethod
    def interaction_mode_is_supported(cls, value: str) -> str:
        if value not in SUPPORTED_INTERACTION_MODES:
            supported = ", ".join(sorted(SUPPORTED_INTERACTION_MODES))
            raise ValueError(f"interaction_mode must be one of: {supported}")
        return value

    @field_validator("participant_mode")
    @classmethod
    def participant_mode_is_supported(cls, value: str) -> str:
        if value not in SUPPORTED_PARTICIPANT_MODES:
            supported = ", ".join(sorted(SUPPORTED_PARTICIPANT_MODES))
            raise ValueError(f"participant_mode must be one of: {supported}")
        return value

    @field_validator("human_input_mode")
    @classmethod
    def human_input_mode_is_supported(cls, value: str) -> str:
        if value not in SUPPORTED_HUMAN_INPUT_MODES:
            supported = ", ".join(sorted(SUPPORTED_HUMAN_INPUT_MODES))
            raise ValueError(f"human_input_mode must be one of: {supported}")
        return value

    @field_validator("human_stt_submit_policy")
    @classmethod
    def human_stt_submit_policy_is_supported(cls, value: str) -> str:
        if value not in SUPPORTED_HUMAN_STT_SUBMIT_POLICIES:
            supported = ", ".join(sorted(SUPPORTED_HUMAN_STT_SUBMIT_POLICIES))
            raise ValueError(f"human_stt_submit_policy must be one of: {supported}")
        return value

    @field_validator("control_port")
    @classmethod
    def control_port_is_valid(cls, value: int) -> int:
        if value <= 0 or value > 65535:
            raise ValueError("control_port must be between 1 and 65535")
        return value

    @field_validator("max_turns")
    @classmethod
    def max_turns_is_positive(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("max_turns must be positive")
        return value

    @field_validator("max_elapsed_seconds")
    @classmethod
    def max_elapsed_seconds_is_positive(cls, value: float | None) -> float | None:
        if value is not None and value <= 0:
            raise ValueError("max_elapsed_seconds must be positive when provided")
        return value

    @field_validator("closing_start_elapsed_seconds")
    @classmethod
    def closing_start_elapsed_seconds_is_non_negative(
        cls,
        value: float | None,
    ) -> float | None:
        if value is not None and value < 0:
            raise ValueError(
                "closing_start_elapsed_seconds must be non-negative when provided"
            )
        return value

    @field_validator("force_stop_after_closing_turns")
    @classmethod
    def force_stop_after_closing_turns_is_positive(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("force_stop_after_closing_turns must be positive")
        return value

    @field_validator("fixed_speaker_sequence")
    @classmethod
    def fixed_speaker_sequence_is_valid(
        cls,
        value: list[str] | None,
    ) -> list[str] | None:
        if value is None:
            return value
        if not value:
            raise ValueError("fixed_speaker_sequence must not be empty")
        for speaker_id in value:
            if not speaker_id.strip():
                raise ValueError(
                    "fixed_speaker_sequence must contain non-empty speaker ids"
                )
        return [speaker_id.strip() for speaker_id in value]

    @field_validator("overlap_grace_ms")
    @classmethod
    def overlap_grace_ms_is_non_negative(cls, value: int) -> int:
        if value < 0:
            raise ValueError("overlap_grace_ms must be non-negative")
        return value

    @field_validator("unresolved_overlap_limit_ms")
    @classmethod
    def unresolved_overlap_limit_ms_is_positive(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("unresolved_overlap_limit_ms must be positive")
        return value

    @field_validator("turn_taking_signal_min_chars")
    @classmethod
    def turn_taking_signal_min_chars_is_positive(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("turn_taking_signal_min_chars must be positive")
        return value

    @field_validator("turn_taking_decision_timeout_ms")
    @classmethod
    def turn_taking_decision_timeout_ms_is_positive(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("turn_taking_decision_timeout_ms must be positive")
        return value

    @field_validator("turn_taking_max_reconsider_rounds")
    @classmethod
    def turn_taking_max_reconsider_rounds_is_non_negative(cls, value: int) -> int:
        if value < 0:
            raise ValueError(
                "turn_taking_max_reconsider_rounds must be non-negative"
            )
        return value

    @field_validator(
        "conversation_context_recent_turns",
        "session_summary_trigger_completed_turns",
        "session_summary_update_interval_turns",
    )
    @classmethod
    def conversation_memory_values_are_positive(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("conversation memory values must be positive")
        return value

    @model_validator(mode="after")
    def validate_stop_condition(self) -> RuntimeSection:
        if self.stop_condition == "elapsed_time" and self.max_elapsed_seconds is None:
            raise ValueError(
                "runtime.max_elapsed_seconds is required when stop_condition is elapsed_time"
            )
        return self


class AudioSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    hot_path_format: str
    sample_rate: int
    sample_width_bits: int
    channels: int
    speaker_gains: dict[str, float] = Field(
        default_factory=lambda: {"counselor": 1.0, "client": 1.0}
    )
    turn_log_format: str
    archive_format: str
    public_export_format: str
    create_public_mp3: bool

    @field_validator("hot_path_format")
    @classmethod
    def hot_path_format_is_pcm(cls, value: str) -> str:
        if value != "pcm":
            raise ValueError("audio.hot_path_format must initially be pcm")
        return value

    @field_validator("sample_rate")
    @classmethod
    def sample_rate_is_positive(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("sample_rate must be positive")
        return value

    @field_validator("sample_width_bits")
    @classmethod
    def sample_width_bits_is_supported(cls, value: int) -> int:
        if value not in {8, 16, 24, 32}:
            raise ValueError("sample_width_bits must be a supported PCM width")
        return value

    @field_validator("channels")
    @classmethod
    def channels_is_positive(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("channels must be positive")
        return value

    @field_validator("speaker_gains")
    @classmethod
    def speaker_gains_are_positive(cls, value: dict[str, float]) -> dict[str, float]:
        for speaker, gain in value.items():
            if gain <= 0:
                raise ValueError(f"audio.speaker_gains.{speaker} must be positive")
        return value


class RealtimeTranscriptionFormat(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: str
    rate: int

    @field_validator("type")
    @classmethod
    def type_is_pcm_audio(cls, value: str) -> str:
        if value != "audio/pcm":
            raise ValueError("realtime_transcription_format.type must initially be audio/pcm")
        return value

    @field_validator("rate")
    @classmethod
    def rate_is_positive(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("realtime_transcription_format.rate must be positive")
        return value


class OpenAISection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    llm_model: str
    timing_llm_model: str = "gpt-5.4-mini"
    timing_llm_reasoning_effort: str = "none"
    summary_llm_model: str = "gpt-5.4-mini"
    summary_llm_reasoning_effort: str = "none"
    prompting_llm_model: str = "gpt-5.4-mini"
    prompting_llm_reasoning_effort: str = "low"
    tts_model: str
    realtime_model: str = "gpt-realtime-2.1"
    tts_voice: str
    counselor_tts_voice: str | None = None
    client_tts_voice: str | None = None
    realtime_output_speed: float = Field(
        default=1.0,
        ge=MIN_REALTIME_OUTPUT_SPEED,
        le=MAX_REALTIME_OUTPUT_SPEED,
    )
    counselor_realtime_output_speed: float | None = Field(
        default=None,
        ge=MIN_REALTIME_OUTPUT_SPEED,
        le=MAX_REALTIME_OUTPUT_SPEED,
    )
    client_realtime_output_speed: float | None = Field(
        default=None,
        ge=MIN_REALTIME_OUTPUT_SPEED,
        le=MAX_REALTIME_OUTPUT_SPEED,
    )
    tts_instructions: str = ""
    tts_response_format: str
    stt_model: str
    realtime_transcription_format: RealtimeTranscriptionFormat
    realtime_output_format: RealtimeTranscriptionFormat = Field(
        default_factory=lambda: RealtimeTranscriptionFormat(type="audio/pcm", rate=24000)
    )

    @field_validator("tts_response_format")
    @classmethod
    def tts_response_format_is_pcm(cls, value: str) -> str:
        if value != "pcm":
            raise ValueError("tts_response_format must initially be pcm")
        return value

    @field_validator(
        "timing_llm_reasoning_effort",
        "summary_llm_reasoning_effort",
        "prompting_llm_reasoning_effort",
    )
    @classmethod
    def llm_reasoning_effort_is_supported(cls, value: str) -> str:
        if value not in {"none", "low", "medium", "high", "xhigh"}:
            raise ValueError(
                "llm reasoning effort must be one of none, low, medium, high, xhigh"
            )
        return value


class ParticipantSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    speaker_id: str | None = None
    role: str
    display_name: str
    actor_kind: str = "ai"
    prompt_path: str | None = None
    public_profile_path: str | None = None
    private_profile_path: str | None = None
    prompt_source: str | None = None
    public_profile_source: str | None = None
    private_profile_source: str | None = None
    initial_transcript: str = ""
    voice: str | None = None
    realtime_output_speed: float | None = Field(
        default=None,
        ge=MIN_REALTIME_OUTPUT_SPEED,
        le=MAX_REALTIME_OUTPUT_SPEED,
    )

    @field_validator("role")
    @classmethod
    def role_is_supported(cls, value: str) -> str:
        if value not in SUPPORTED_PARTICIPANT_ROLES:
            supported = ", ".join(sorted(SUPPORTED_PARTICIPANT_ROLES))
            raise ValueError(f"participant role must be one of: {supported}")
        return value

    @field_validator("actor_kind")
    @classmethod
    def actor_kind_is_supported(cls, value: str) -> str:
        if value not in SUPPORTED_ACTOR_KINDS:
            supported = ", ".join(sorted(SUPPORTED_ACTOR_KINDS))
            raise ValueError(f"participant actor_kind must be one of: {supported}")
        return value


class SharedCaseSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt_path: str | None = None
    prompt_source: str | None = None


class PathsSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sessions_dir: str
    runtime_sessions_dir: str
    replay_sessions_dir: str


class LatencySection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    llm_max_output_tokens: int | None = None
    timing_llm_max_output_tokens: int = 128
    summary_llm_max_output_tokens: int = 2048
    prompting_llm_max_output_tokens: int = 8192
    tts_chunk_comma_min_chars: int = 20
    tts_chunk_soft_max_chars: int = 40
    tts_target_chunk_duration_ms: int = 40
    tts_sdk_chunk_size: int | None = None
    stt_partial_prefetch: bool = False
    stt_partial_prefetch_min_chars: int = 8
    realtime_api_centered_mode: bool = False

    @field_validator(
        "tts_chunk_comma_min_chars",
        "tts_chunk_soft_max_chars",
        "tts_target_chunk_duration_ms",
        "stt_partial_prefetch_min_chars",
    )
    @classmethod
    def positive_int_latency_value(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("latency values must be positive")
        return value

    @field_validator("llm_max_output_tokens")
    @classmethod
    def optional_llm_max_output_tokens_is_positive(cls, value: int | None) -> int | None:
        if value is not None and value <= 0:
            raise ValueError("llm_max_output_tokens must be positive when provided")
        return value

    @field_validator(
        "timing_llm_max_output_tokens",
        "summary_llm_max_output_tokens",
        "prompting_llm_max_output_tokens",
    )
    @classmethod
    def control_llm_max_output_tokens_is_positive(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("control LLM max_output_tokens must be positive")
        return value

    @field_validator("tts_sdk_chunk_size")
    @classmethod
    def optional_tts_sdk_chunk_size_is_positive(cls, value: int | None) -> int | None:
        if value is not None and value <= 0:
            raise ValueError("tts_sdk_chunk_size must be positive when provided")
        return value


AI_ROUTE_MODEL_FIELDS = {
    "realtime_speech": "realtime_model",
    "realtime_transcription": "stt_model",
    "conversation_text": "llm_model",
    "turn_timing": "timing_llm_model",
    "session_summary": "summary_llm_model",
    "prompt_director": "prompting_llm_model",
}


class AIProviderSection(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    kind: Literal["openai", "azure_openai"]
    auth: Literal["api_key"] = "api_key"
    api_key_env: str = Field(pattern=r"^[A-Za-z_][A-Za-z0-9_]*$")
    endpoint_env: str | None = Field(default=None, pattern=r"^[A-Za-z_][A-Za-z0-9_]*$")

    @model_validator(mode="after")
    def require_azure_endpoint(self) -> AIProviderSection:
        if self.kind == "azure_openai" and self.endpoint_env is None:
            raise ValueError("Azure provider requires endpoint_env")
        return self


class AIRouteTargetSection(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    model: str | None = None
    deployment: str | None = None
    # Transcription inside the speech session uses the speech provider.
    input_transcription_model: str | None = None


class AIRouteSection(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    provider: str | None = None
    targets: dict[str, AIRouteTargetSection]


class AISection(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    default_provider: str = "openai"
    providers: dict[str, AIProviderSection]
    # Reject routes whose transports have not yet been implemented.
    routes: dict[
        Literal[
            "realtime_speech",
            "realtime_transcription",
            "conversation_text",
            "turn_timing",
            "session_summary",
            "prompt_director",
        ],
        AIRouteSection,
    ]

    @model_validator(mode="after")
    def validate_routes(self) -> AISection:
        if self.default_provider not in self.providers:
            raise ValueError("default_provider must exist in ai.providers")
        if "realtime_speech" not in self.routes:
            raise ValueError("ai.routes requires realtime_speech")
        for name, route in self.routes.items():
            provider_id = route.provider or self.default_provider
            if provider_id not in self.providers:
                raise ValueError(f"route={name}: unknown provider")
            if provider_id not in route.targets:
                raise ValueError(f"route={name}: selected provider target is missing")
            for target_provider, target in route.targets.items():
                if target_provider not in self.providers:
                    raise ValueError(f"route={name}: unknown target provider")
                kind = self.providers[target_provider].kind
                if kind == "openai" and (
                    target.model is None or target.deployment is not None
                ):
                    raise ValueError(f"route={name}: OpenAI target requires model only")
                if kind == "azure_openai" and (
                    target.deployment is None or target.model is not None
                ):
                    raise ValueError(
                        f"route={name}: Azure target requires deployment only"
                    )
        # Empty deployment values are checked only when the route is used.
        return self


class RuntimeSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    runtime: RuntimeSection
    audio: AudioSection
    openai: OpenAISection
    ai: AISection | None = None
    paths: PathsSection
    participants: dict[str, ParticipantSection] = Field(
        default_factory=lambda: {
            "counselor": ParticipantSection(
                speaker_id="counselor",
                role="counselor",
                display_name="カウンセラー",
            ),
            "client": ParticipantSection(
                speaker_id="client",
                role="client",
                display_name="クライアント",
            ),
        }
    )
    shared_case: SharedCaseSection = Field(default_factory=SharedCaseSection)
    latency: LatencySection = Field(default_factory=LatencySection)

    @model_validator(mode="after")
    def validate_cross_section_policy(self) -> RuntimeSettings:
        if self.openai.realtime_transcription_format.rate != self.audio.sample_rate:
            raise ValueError("realtime_transcription_format.rate must match audio.sample_rate")
        if (
            self.latency.realtime_api_centered_mode
            and self.openai.realtime_output_format.rate != self.audio.sample_rate
        ):
            raise ValueError("realtime_output_format.rate must match audio.sample_rate")
        normalized_participants: dict[str, ParticipantSection] = {}
        for participant_id, participant in self.participants.items():
            speaker_id = participant.speaker_id or participant_id
            if speaker_id != participant_id:
                raise ValueError(
                    "participant speaker_id must match its participants mapping key"
                )
            normalized_participants[participant_id] = participant.model_copy(
                update={"speaker_id": speaker_id}
            )
        if "counselor" not in normalized_participants:
            raise ValueError("participants must include counselor")
        if normalized_participants["counselor"].role != "counselor":
            raise ValueError("participants.counselor.role must be counselor")
        if not any(
            participant.role == "client"
            for speaker_id, participant in normalized_participants.items()
            if speaker_id != "counselor"
        ):
            raise ValueError("participants must include at least one client")
        self.participants = normalized_participants
        validate_human_client_configuration(
            interaction_mode=self.runtime.interaction_mode,
            participant_mode=self.runtime.participant_mode,
            participants=self.participants,
            speaker_selection_policy=self.runtime.speaker_selection_policy,
            fixed_speaker_sequence=self.runtime.fixed_speaker_sequence
            or ("counselor", "client"),
            initial_client_transcript=self.runtime.initial_client_transcript,
        )
        if (
            self.runtime.interaction_mode == "ai_counselor_human_client"
            and not self.latency.realtime_api_centered_mode
        ):
            raise ValueError(
                "ai_counselor_human_client requires realtime_api_centered_mode=true"
            )
        return self


RuntimeConfigDocument = RuntimeSettings


def _expand_env_vars(value: Any) -> Any:
    if isinstance(value, str):
        expanded = ENV_DEFAULT_PATTERN.sub(
            lambda match: os.environ.get(match.group(1)) or match.group(2),
            value,
        )
        return os.path.expandvars(expanded)
    if isinstance(value, list):
        return [_expand_env_vars(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand_env_vars(item) for key, item in value.items()}
    return value


def load_runtime_config(path: Path | str = DEFAULT_RUNTIME_CONFIG_PATH) -> RuntimeSettings:
    load_dotenv(ROOT_DIR / ".env", override=False)
    config_path = Path(path)
    try:
        raw_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RuntimeConfigLoadError(f"設定ファイルが見つかりません: {config_path}") from exc
    except yaml.YAMLError as exc:
        raise RuntimeConfigLoadError(f"設定ファイルのYAML解析に失敗しました: {config_path}: {exc}") from exc

    try:
        return RuntimeSettings.model_validate(_expand_env_vars(raw_config))
    except ValidationError as exc:
        raise RuntimeConfigLoadError(f"設定ファイルの検証に失敗しました: {config_path}: {exc}") from exc


def main() -> None:
    config = load_runtime_config()
    print(f"loaded runtime config: {config.runtime.engine}")
    print(f"audio_delivery_mode: {config.runtime.audio_delivery_mode}")


if __name__ == "__main__":
    main()
