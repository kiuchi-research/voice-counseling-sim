from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

from counseling_voice_demo.runtime.agents import StreamingAgent
from counseling_voice_demo.runtime.config import RuntimeSettings, load_runtime_config
from counseling_voice_demo.runtime.controller import ConversationRuntime
from counseling_voice_demo.runtime.models import (
    ActorKind,
    AudioDeliveryMode,
    ParticipantConfig,
    RuntimeConfig,
)
from counseling_voice_demo.runtime.prompt_context import (
    RuntimePromptContextStore,
    build_prompt_context_input_formatter,
    build_realtime_prompt_input_formatter,
)
from counseling_voice_demo.runtime.prompt_director import (
    PromptDirector,
    prompt_director_text_format,
)
from counseling_voice_demo.runtime.prompt_sources import (
    PromptSourceReadError,
    resolve_optional_text_source,
    resolve_participant_prompt_sources,
)
from counseling_voice_demo.runtime.provider_registry import (
    ProviderConfigurationError,
    ProviderRegistry,
)
from counseling_voice_demo.runtime.realtime_speech import (
    OpenAIRealtimeSpeechAgent,
    RealtimeSpeechConfig,
)
from counseling_voice_demo.runtime.speaker_selection import (
    DISTRIBUTED_TIMING_POLICY,
    TURN_BOUNDARY_TIMING_POLICY,
)
from counseling_voice_demo.runtime.streaming_llm import OpenAIStreamingLLM
from counseling_voice_demo.runtime.streaming_stt import OpenAIRealtimeTranscriptionSTT
from counseling_voice_demo.runtime.streaming_tts import (
    FakeStreamingTTS,
    OpenAIStreamingTTS,
    OpenAIStreamingTTSConfig,
)
from counseling_voice_demo.runtime.system_prompts import (
    CLIENT_SYSTEM_PROMPT,
    COUNSELOR_SYSTEM_PROMPT,
)


RuntimeConfigInput = RuntimeSettings | Mapping[str, Any] | Path | str | None
ClientFactory = Any


class RuntimeFactoryError(RuntimeError):
    pass


def build_runtime_config_model(config: RuntimeConfigInput = None) -> RuntimeConfig:
    settings = _coerce_runtime_settings(config)
    participants = _resolve_participant_prompt_sources(settings)
    runtime_config_kwargs: dict[str, Any] = {
        "interaction_mode": settings.runtime.interaction_mode,
        "participant_mode": settings.runtime.participant_mode,
        "human_input_mode": settings.runtime.human_input_mode,
        "human_stt_submit_policy": settings.runtime.human_stt_submit_policy,
        "human_interrupts_enabled": settings.runtime.human_interrupts_enabled,
        "stop_condition": settings.runtime.stop_condition,
        "max_turns": settings.runtime.max_turns,
        "max_elapsed_seconds": settings.runtime.max_elapsed_seconds,
        "closing_start_elapsed_seconds": settings.runtime.closing_start_elapsed_seconds,
        "force_stop_after_closing_turns": settings.runtime.force_stop_after_closing_turns,
        "initial_client_transcript": settings.runtime.initial_client_transcript,
        "audio_delivery_mode": AudioDeliveryMode(settings.runtime.audio_delivery_mode),
        "sample_rate": settings.audio.sample_rate,
        "sample_width_bits": settings.audio.sample_width_bits,
        "channels": settings.audio.channels,
        "speaker_audio_gains": dict(settings.audio.speaker_gains),
        "participants": participants,
        "shared_case": _resolve_shared_case_source(settings),
        "speaker_selection_policy": settings.runtime.speaker_selection_policy,
        "overlap_grace_ms": settings.runtime.overlap_grace_ms,
        "unresolved_overlap_limit_ms": settings.runtime.unresolved_overlap_limit_ms,
        "turn_taking_signal_min_chars": settings.runtime.turn_taking_signal_min_chars,
        "turn_taking_decision_timeout_ms": (
            settings.runtime.turn_taking_decision_timeout_ms
        ),
        "turn_taking_max_reconsider_rounds": (
            settings.runtime.turn_taking_max_reconsider_rounds
        ),
        "conversation_context_recent_turns": (
            settings.runtime.conversation_context_recent_turns
        ),
        "session_summary_trigger_completed_turns": (
            settings.runtime.session_summary_trigger_completed_turns
        ),
        "session_summary_update_interval_turns": (
            settings.runtime.session_summary_update_interval_turns
        ),
        "tts_chunk_comma_min_chars": settings.latency.tts_chunk_comma_min_chars,
        "tts_chunk_soft_max_chars": settings.latency.tts_chunk_soft_max_chars,
        "stt_partial_prefetch": settings.latency.stt_partial_prefetch,
        "stt_partial_prefetch_min_chars": settings.latency.stt_partial_prefetch_min_chars,
    }
    if settings.runtime.fixed_speaker_sequence is not None:
        runtime_config_kwargs["fixed_speaker_sequence"] = tuple(
            settings.runtime.fixed_speaker_sequence
        )
    return RuntimeConfig(**runtime_config_kwargs)


def build_fake_runtime(config: RuntimeConfigInput = None) -> ConversationRuntime:
    settings = _coerce_runtime_settings(config)
    return ConversationRuntime(
        config=build_runtime_config_model(settings),
        sessions_dir=Path(settings.paths.runtime_sessions_dir),
    )


def build_openai_runtime(
    config: RuntimeConfigInput = None,
    *,
    openai_client: Any | None = None,
    client_factory: ClientFactory | None = None,
    api_key: str | None = None,
    stt: Any | None = None,
    stt_transport_context_factory: Any | None = None,
    stt_connect: Any | None = None,
    realtime_connect: Any | None = None,
) -> ConversationRuntime:
    settings = _coerce_runtime_settings(config)
    use_realtime_speech = settings.latency.realtime_api_centered_mode
    runtime_config = build_runtime_config_model(settings)
    prompt_context_store = RuntimePromptContextStore()
    agent_input_formatter = _build_agent_input_formatter(
        runtime_config,
        runtime_context_provider=prompt_context_store,
    )
    realtime_route = None
    realtime_transport_factory = None
    realtime_input_transcription_model = None
    registry = ProviderRegistry(settings, openai_api_key=api_key)
    text_route_names = ["session_summary"]
    if settings.runtime.prompt_director_enabled and any(
        participant.actor_kind is ActorKind.AI and participant.role == "counselor"
        for participant in runtime_config.participants.values()
    ):
        text_route_names.append("prompt_director")
    if runtime_config.speaker_selection_policy in {
        DISTRIBUTED_TIMING_POLICY,
        TURN_BOUNDARY_TIMING_POLICY,
    }:
        text_route_names.append("turn_timing")
    if not use_realtime_speech:
        text_route_names.append("conversation_text")
    try:
        text_routes = {name: registry.resolve_route(name) for name in text_route_names}
        # The legacy injected OpenAI client owns its authentication. All other
        # selected providers must pass validation before constructing any clients.
        registry.preflight_validate(
            name
            for name, route in text_routes.items()
            if not (
                route.provider_id == "openai"
                and route.provider_kind == "openai"
                and registry.ai.providers[route.provider_id].api_key_env
                == "OPENAI_API_KEY"
                and (openai_client is not None or client_factory is not None)
            )
        )
    except ProviderConfigurationError as exc:
        raise RuntimeFactoryError(str(exc)) from exc
    if use_realtime_speech:
        try:
            realtime_route = registry.resolve_route("realtime_speech")
            realtime_transport_factory = registry.build_realtime_transport_factory(
                "realtime_speech",
                connect=realtime_connect,
            )
        except ProviderConfigurationError as exc:
            raise RuntimeFactoryError(str(exc)) from exc
        target = registry.ai.routes["realtime_speech"].targets[
            realtime_route.provider_id
        ]
        realtime_input_transcription_model = target.input_transcription_model
    try:
        stt_route = registry.resolve_route("realtime_transcription")
        resolved_stt_transport_factory = stt_transport_context_factory
        if stt is None and resolved_stt_transport_factory is None:
            resolved_stt_transport_factory = registry.build_realtime_transport_factory(
                "realtime_transcription",
                connect=stt_connect,
            )
    except ProviderConfigurationError as exc:
        raise RuntimeFactoryError(str(exc)) from exc
    control_llm_client: Any | None = None

    def get_control_llm_client() -> Any:
        nonlocal control_llm_client
        if control_llm_client is None:
            control_llm_client = (
                openai_client
                if openai_client is not None
                else _build_openai_client(
                    _resolve_openai_api_key(api_key, required=True),
                    client_factory=client_factory,
                )
            )
        return control_llm_client

    def get_text_client(name: str) -> Any:
        route = text_routes[name]
        if (
            route.provider_id == "openai"
            and route.provider_kind == "openai"
            and registry.ai.providers[route.provider_id].api_key_env == "OPENAI_API_KEY"
        ):
            return get_control_llm_client()
        return registry.build_async_client(name)

    session_summary_llm = OpenAIStreamingLLM(
        client=get_text_client("session_summary"),
        model=text_routes["session_summary"].model_ref,
        max_output_tokens=settings.latency.summary_llm_max_output_tokens,
        reasoning_effort=settings.openai.summary_llm_reasoning_effort,
    )
    prompt_director = None
    if "prompt_director" in text_routes:
        prompt_director = PromptDirector(
            max_retries=settings.runtime.prompt_director_max_retries,
            llm=OpenAIStreamingLLM(
                client=get_text_client("prompt_director"),
                model=text_routes["prompt_director"].model_ref,
                max_output_tokens=(settings.latency.prompting_llm_max_output_tokens),
                reasoning_effort=(settings.openai.prompting_llm_reasoning_effort),
                text_format=prompt_director_text_format(),
            ),
        )
    timing_llm = None
    if "turn_timing" in text_routes:
        timing_llm = OpenAIStreamingLLM(
            client=get_text_client("turn_timing"),
            model=text_routes["turn_timing"].model_ref,
            max_output_tokens=settings.latency.timing_llm_max_output_tokens,
            reasoning_effort=settings.openai.timing_llm_reasoning_effort,
        )
    if use_realtime_speech:
        agents = {
            speaker_id: OpenAIRealtimeSpeechAgent(
                speaker=speaker_id,
                transport_context_factory=realtime_transport_factory,
                config=_build_realtime_speech_config(
                    settings=settings,
                    runtime_config=runtime_config,
                    speaker=speaker_id,
                    model=realtime_route.model_ref,
                    input_transcription_model=realtime_input_transcription_model,
                    instructions=_system_prompt_for_role(participant.role),
                    response_instructions=participant.prompt_source or "",
                    repeat_instructions_per_response=True,
                ),
                input_formatter=agent_input_formatter,
                response_input_formatter=build_realtime_prompt_input_formatter(
                    participants=runtime_config.participants,
                    shared_case=runtime_config.shared_case,
                    runtime_context_provider=prompt_context_store,
                ),
                connect=realtime_connect,
                timing_llm=timing_llm,
            )
            for speaker_id, participant in runtime_config.participants.items()
            if participant.actor_kind is ActorKind.AI
        }
        tts = FakeStreamingTTS()
    else:
        client = get_control_llm_client()
        agents = {
            speaker_id: StreamingAgent(
                speaker=speaker_id,
                llm=OpenAIStreamingLLM(
                    client=get_text_client("conversation_text"),
                    model=text_routes["conversation_text"].model_ref,
                    system_prompt=_system_prompt_for_role(participant.role),
                    max_output_tokens=settings.latency.llm_max_output_tokens,
                ),
                timing_llm=timing_llm,
                input_formatter=agent_input_formatter,
            )
            for speaker_id, participant in runtime_config.participants.items()
            if participant.actor_kind is ActorKind.AI
        }
        tts = OpenAIStreamingTTS(
            client=client,
            config=OpenAIStreamingTTSConfig(
                model=settings.openai.tts_model,
                voice=settings.openai.tts_voice,
                voice_by_speaker=_resolve_voice_by_speaker(settings, runtime_config),
                instructions=settings.openai.tts_instructions,
                response_format=settings.openai.tts_response_format,
                sdk_chunk_size=settings.latency.tts_sdk_chunk_size,
                target_chunk_duration_ms=settings.latency.tts_target_chunk_duration_ms,
            ),
        )
    resolved_stt = stt or OpenAIRealtimeTranscriptionSTT(
        model=stt_route.model_ref,
        transport_context_factory=resolved_stt_transport_factory,
        connect=stt_connect,
    )
    ai_route_manifest = {"realtime_transcription": stt_route.manifest()}
    ai_route_manifest.update(
        {name: route.manifest() for name, route in text_routes.items()}
    )
    if realtime_route is not None:
        ai_route_manifest["realtime_speech"] = realtime_route.manifest()
    else:
        for name, model in (("tts", settings.openai.tts_model),):
            ai_route_manifest[name] = {
                "provider": "openai",
                "kind": "openai",
                "model_ref": model,
                "auth": "api_key",
            }
    return ConversationRuntime(
        config=runtime_config,
        agents=agents,
        tts=tts,
        stt=resolved_stt,
        session_summary_llm=session_summary_llm,
        prompt_director=prompt_director,
        prompt_context_store=prompt_context_store,
        sessions_dir=Path(settings.paths.runtime_sessions_dir),
        ai_route_manifest=ai_route_manifest,
    )


def _system_prompt_for_role(role: str) -> str:
    if role == "counselor":
        return COUNSELOR_SYSTEM_PROMPT
    if role == "client":
        return CLIENT_SYSTEM_PROMPT
    raise RuntimeFactoryError(f"unsupported participant role: {role}")


def _resolve_participant_prompt_sources(
    settings: RuntimeSettings,
) -> dict[str, ParticipantConfig]:
    participants: dict[str, ParticipantConfig] = {}
    try:
        for speaker_id, participant in settings.participants.items():
            participants[speaker_id] = resolve_participant_prompt_sources(
                ParticipantConfig(
                    speaker_id=participant.speaker_id or speaker_id,
                    role=participant.role,
                    display_name=participant.display_name,
                    actor_kind=participant.actor_kind,
                    prompt_path=participant.prompt_path,
                    public_profile_path=participant.public_profile_path,
                    private_profile_path=participant.private_profile_path,
                    prompt_source=participant.prompt_source,
                    public_profile_source=participant.public_profile_source,
                    private_profile_source=participant.private_profile_source,
                    initial_transcript=participant.initial_transcript,
                    voice=participant.voice,
                    realtime_output_speed=participant.realtime_output_speed,
                ),
            )
    except PromptSourceReadError as exc:
        raise RuntimeFactoryError(str(exc)) from exc
    return participants


def _resolve_shared_case_source(settings: RuntimeSettings) -> str:
    try:
        return (
            resolve_optional_text_source(
                source=settings.shared_case.prompt_source,
                path=settings.shared_case.prompt_path,
                label="shared case",
            )
            or ""
        )
    except PromptSourceReadError as exc:
        raise RuntimeFactoryError(str(exc)) from exc


def _build_agent_input_formatter(
    runtime_config: RuntimeConfig,
    *,
    runtime_context_provider: Any | None = None,
):
    if runtime_context_provider is None and not _has_prompt_context_sources(runtime_config):
        return format_agent_latest_input
    return build_prompt_context_input_formatter(
        participants=runtime_config.participants,
        shared_case=runtime_config.shared_case,
        runtime_context_provider=runtime_context_provider,
    )


def _has_prompt_context_sources(runtime_config: RuntimeConfig) -> bool:
    if runtime_config.shared_case.strip():
        return True
    return any(
        (participant.prompt_source or "").strip()
        or (participant.public_profile_source or "").strip()
        or (participant.private_profile_source or "").strip()
        for participant in runtime_config.participants.values()
    )


def _build_realtime_speech_config(
    *,
    settings: RuntimeSettings,
    runtime_config: RuntimeConfig,
    speaker: str,
    instructions: str,
    model: str | None = None,
    input_transcription_model: str | None = None,
    response_instructions: str = "",
    repeat_instructions_per_response: bool = False,
) -> RealtimeSpeechConfig:
    config = RealtimeSpeechConfig(
        model=model if model is not None else settings.openai.realtime_model,
        voice=_resolve_voice_for_speaker(
            settings,
            runtime_config.participants,
            speaker,
        ),
        output_speed=_resolve_realtime_output_speed_for_speaker(
            settings,
            runtime_config.participants,
            speaker,
        ),
        instructions=instructions,
        response_instructions=response_instructions,
        repeat_instructions_per_response=repeat_instructions_per_response,
        output_format=settings.openai.realtime_output_format.model_dump(),
        target_chunk_duration_ms=settings.latency.tts_target_chunk_duration_ms,
    )
    if input_transcription_model is not None:
        config = replace(
            config, input_transcription={"model": input_transcription_model}
        )
    return config


def _resolve_voice_by_speaker(
    settings: RuntimeSettings,
    runtime_config: RuntimeConfig,
) -> dict[str, str]:
    voice_by_speaker = {
        speaker_id: _resolve_voice_for_speaker(
            settings,
            runtime_config.participants,
            speaker_id,
        )
        for speaker_id in runtime_config.participants
    }
    for legacy_speaker in ("counselor", "client"):
        voice_by_speaker.setdefault(
            legacy_speaker,
            _resolve_voice_for_speaker(
                settings,
                runtime_config.participants,
                legacy_speaker,
            ),
        )
    return voice_by_speaker


def _resolve_voice_for_speaker(
    settings: RuntimeSettings,
    participants: Mapping[str, ParticipantConfig],
    speaker: str,
) -> str:
    participant = participants.get(speaker)
    if participant is not None:
        return participant.voice or _legacy_voice_for_role(settings, participant.role)
    return _legacy_voice_for_speaker(settings, speaker)


def _resolve_realtime_output_speed_for_speaker(
    settings: RuntimeSettings,
    participants: Mapping[str, ParticipantConfig],
    speaker: str,
) -> float:
    participant = participants.get(speaker)
    if participant is not None:
        if participant.realtime_output_speed is not None:
            return participant.realtime_output_speed
        return _legacy_realtime_output_speed_for_role(settings, participant.role)
    return _legacy_realtime_output_speed_for_speaker(settings, speaker)


def _legacy_voice_for_speaker(settings: RuntimeSettings, speaker: str) -> str:
    if speaker == "counselor":
        return _legacy_voice_for_role(settings, "counselor")
    if speaker == "client":
        return _legacy_voice_for_role(settings, "client")
    return settings.openai.tts_voice


def _legacy_voice_for_role(settings: RuntimeSettings, role: str) -> str:
    if role == "counselor":
        return settings.openai.counselor_tts_voice or settings.openai.tts_voice
    if role == "client":
        return settings.openai.client_tts_voice or settings.openai.tts_voice
    return settings.openai.tts_voice


def _legacy_realtime_output_speed_for_speaker(
    settings: RuntimeSettings,
    speaker: str,
) -> float:
    if speaker == "counselor":
        return _legacy_realtime_output_speed_for_role(settings, "counselor")
    if speaker == "client":
        return _legacy_realtime_output_speed_for_role(settings, "client")
    return settings.openai.realtime_output_speed


def _legacy_realtime_output_speed_for_role(
    settings: RuntimeSettings,
    role: str,
) -> float:
    if (
        role == "counselor"
        and settings.openai.counselor_realtime_output_speed is not None
    ):
        return settings.openai.counselor_realtime_output_speed
    if role == "client" and settings.openai.client_realtime_output_speed is not None:
        return settings.openai.client_realtime_output_speed
    return settings.openai.realtime_output_speed


def format_agent_latest_input(
    *,
    speaker: str,
    turn_id: int,
    input_transcript: str,
    purpose: str | None = None,
) -> str:
    _ = purpose
    normalized_input = input_transcript.strip() or "（直前発話は空です。）"
    if speaker == "counselor":
        return "\n".join(
            [
                f"生成対象: turn={turn_id} speaker=counselor",
                "直前のクライアント発話:",
                normalized_input,
            ]
        )
    if speaker == "client":
        return "\n".join(
            [
                f"生成対象: turn={turn_id} speaker=client",
                "直前のカウンセラー発話:",
                normalized_input,
            ]
        )
    return "\n".join(
        [
            f"生成対象: turn={turn_id} speaker={speaker}",
            "直前の相手発話:",
            normalized_input,
        ]
    )


def _coerce_runtime_settings(config: RuntimeConfigInput) -> RuntimeSettings:
    if config is None:
        return load_runtime_config()
    if isinstance(config, RuntimeSettings):
        return config
    if isinstance(config, Path | str):
        return load_runtime_config(config)
    if isinstance(config, Mapping):
        return RuntimeSettings.model_validate(config)
    raise TypeError("config must be a RuntimeSettings, mapping, path, or None")


def _resolve_openai_api_key(api_key: str | None, *, required: bool) -> str:
    resolved_api_key = api_key if api_key is not None else os.getenv("OPENAI_API_KEY")
    if resolved_api_key and resolved_api_key.strip():
        return resolved_api_key.strip()
    if required:
        raise RuntimeFactoryError(
            "OPENAI_API_KEY is not set. Provide api_key, pass openai_client, "
            "or set OPENAI_API_KEY in the environment."
        )
    return ""


def _build_openai_client(api_key: str, *, client_factory: ClientFactory | None) -> Any:
    if client_factory is not None:
        return client_factory(api_key)

    try:
        from openai import AsyncOpenAI
    except ModuleNotFoundError as exc:
        raise RuntimeFactoryError("openai is not installed; cannot build OpenAI runtime") from exc

    return AsyncOpenAI(api_key=api_key)
