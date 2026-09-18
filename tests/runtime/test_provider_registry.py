from __future__ import annotations

import asyncio
import json
from copy import deepcopy
from urllib.parse import parse_qs, urlsplit

import pytest
from pydantic import ValidationError

from counseling_voice_demo.runtime.config import AISection, load_runtime_config
from counseling_voice_demo.runtime.control_api import (
    RuntimeControlError,
    apply_runtime_start_options,
    normalize_runtime_start_options,
)
from counseling_voice_demo.runtime.factory import build_openai_runtime
from counseling_voice_demo.runtime.provider_registry import (
    ProviderConfigurationError,
    ProviderRegistry,
)
from counseling_voice_demo.runtime.realtime_speech import OpenAIRealtimeSpeechAgent
from counseling_voice_demo.runtime.realtime_transport import (
    RealtimeAuthHeaders,
    build_azure_realtime_session_url,
)


def ai_config(provider="azure_eastus2", deployment="custom-realtime-prod"):
    return {
        "default_provider": "openai",
        "providers": {
            "openai": {"kind": "openai", "api_key_env": "OPENAI_API_KEY"},
            "azure_eastus2": {
                "kind": "azure_openai",
                "endpoint_env": "TEST_AZURE_ENDPOINT",
                "api_key_env": "TEST_AZURE_KEY",
            },
        },
        "routes": {
            "realtime_speech": {
                "provider": provider,
                "targets": {
                    "openai": {"model": "gpt-realtime-2.1"},
                    "azure_eastus2": {"deployment": deployment},
                },
            },
        },
    }


def settings_with_ai(**kwargs):
    return load_runtime_config().model_copy(
        update={"ai": AISection.model_validate(ai_config(**kwargs))}
    )


@pytest.fixture
def azure_env(monkeypatch):
    monkeypatch.setenv("TEST_AZURE_ENDPOINT", "https://example.openai.azure.com/")
    monkeypatch.setenv("TEST_AZURE_KEY", "test-only-azure-key")


@pytest.mark.parametrize("suffix", ["", "/", "/openai/v1", "/openai/v1/"])
def test_azure_url_normalizes_resource_and_v1_endpoints(suffix):
    url = build_azure_realtime_session_url(
        "https://example.openai.azure.com" + suffix, "custom deployment &?/#"
    )
    parsed = urlsplit(url)
    assert parsed.scheme == "wss"
    assert parsed.path == "/openai/v1/realtime"
    assert parse_qs(parsed.query) == {"model": ["custom deployment &?/#"]}


@pytest.mark.parametrize(
    "endpoint",
    [
        "http://example.openai.azure.com",
        "wss://example.openai.azure.com",
        "https://example.openai.azure.com/projects/project",
        "https://user:private-value@example.openai.azure.com",
        "https://example.openai.azure.com/?api-key=private-value",
        "https://example.openai.azure.com/#private-value",
        "https://",
    ],
)
def test_invalid_endpoint_fails_without_echoing_value(endpoint):
    with pytest.raises(ValueError) as error:
        build_azure_realtime_session_url(endpoint, "deployment")
    assert "private-value" not in str(error.value)


def test_headers_redact_both_auth_schemes_case_insensitively():
    headers = RealtimeAuthHeaders(
        {
            "Authorization": "Bearer private-one",
            "API-Key": "private-two",
            "api-key": "private-three",
            "OpenAI-Beta": "realtime=v1",
        }
    )
    assert "private" not in repr(headers)
    assert "private" not in str(headers)
    assert "realtime=v1" in repr(headers)


def test_legacy_settings_resolve_openai_model(monkeypatch):
    settings = load_runtime_config().model_copy(update={"ai": None})
    settings.openai.realtime_model = "legacy-custom-model"
    registry = ProviderRegistry(settings, openai_api_key="test-only-key")
    route = registry.resolve_route("realtime_speech")
    assert route.provider_id == "openai"
    assert route.model_ref == "legacy-custom-model"
    spec = registry.build_realtime_connection_spec("realtime_speech")
    assert spec.url == "wss://api.openai.com/v1/realtime?model=legacy-custom-model"
    assert set(spec.headers) == {"Authorization"}
    assert "test-only-key" not in repr(spec)


def test_unselected_azure_needs_no_endpoint_key_or_deployment(monkeypatch):
    monkeypatch.delenv("TEST_AZURE_ENDPOINT", raising=False)
    monkeypatch.delenv("TEST_AZURE_KEY", raising=False)
    registry = ProviderRegistry(
        settings_with_ai(provider="openai", deployment=""),
        openai_api_key="test-only-key",
    )
    registry.preflight_validate(["realtime_speech"])


@pytest.mark.parametrize("missing", ["TEST_AZURE_KEY", "TEST_AZURE_ENDPOINT"])
def test_selected_azure_fails_preflight_for_missing_env(
    azure_env, monkeypatch, missing
):
    monkeypatch.delenv(missing)
    registry = ProviderRegistry(settings_with_ai())
    with pytest.raises(ProviderConfigurationError, match=missing) as error:
        registry.preflight_validate(["realtime_speech"])
    assert "realtime_speech" in str(error.value)
    assert "azure_eastus2" in str(error.value)
    assert "test-only-azure-key" not in str(error.value)


def test_selected_azure_rejects_empty_deployment(azure_env):
    registry = ProviderRegistry(settings_with_ai(deployment=""))
    with pytest.raises(ProviderConfigurationError, match="deployment"):
        registry.preflight_validate(["realtime_speech"])


@pytest.mark.parametrize(
    "change",
    [
        lambda data: data["routes"]["realtime_speech"].update(provider="unknown"),
        lambda data: data["routes"]["realtime_speech"]["targets"].pop("azure_eastus2"),
        lambda data: data["providers"]["azure_eastus2"].pop("endpoint_env"),
        lambda data: data["routes"].update(
            evaluation=deepcopy(data["routes"]["realtime_speech"])
        ),
    ],
)
def test_invalid_or_unimplemented_route_config_is_rejected(change):
    data = ai_config()
    change(data)
    with pytest.raises(ValidationError):
        AISection.model_validate(data)


def test_control_options_switch_provider_without_overwriting_other_targets():
    settings = settings_with_ai(provider="openai")
    original = settings.model_dump()
    options = {
        "ai_routes": {
            "realtime_speech": {
                "provider": "azure_eastus2",
                "model_ref": "my-custom-deployment",
            }
        }
    }
    active = apply_runtime_start_options(settings, options)
    route = ProviderRegistry(active).resolve_route("realtime_speech")
    assert route.provider_id == "azure_eastus2"
    assert route.model_ref == "my-custom-deployment"
    assert (
        active.ai.routes["realtime_speech"].targets["openai"].model
        == "gpt-realtime-2.1"
    )
    assert active.openai == settings.openai
    assert settings.model_dump() == original
    switched_back = apply_runtime_start_options(
        active,
        {
            "ai_routes": {"realtime_speech": {"provider": "openai"}},
            "realtime_model": "gpt-realtime-2",
        },
    )
    assert (
        ProviderRegistry(switched_back).resolve_route("realtime_speech").model_ref
        == "gpt-realtime-2"
    )
    assert (
        switched_back.ai.routes["realtime_speech"].targets["azure_eastus2"].deployment
        == "my-custom-deployment"
    )


@pytest.mark.parametrize(
    "override",
    [
        {"unknown_route": {"provider": "openai"}},
        {"realtime_speech": {"provider": "azure_eastus2", "api_key": "test-secret"}},
        {
            "realtime_speech": {
                "provider": "azure_eastus2",
                "endpoint": "https://example.com",
            }
        },
        {"realtime_speech": {"provider": "azure_eastus2", "model_ref": " "}},
    ],
)
def test_control_api_rejects_unknown_routes_or_secret_options(override):
    with pytest.raises(RuntimeControlError):
        normalize_runtime_start_options({"ai_routes": override})


def test_control_api_rejects_unknown_provider():
    with pytest.raises(RuntimeControlError, match="provider"):
        apply_runtime_start_options(
            settings_with_ai(),
            {
                "ai_routes": {"realtime_speech": {"provider": "unknown"}},
            },
        )


def test_azure_runtime_injects_transport_and_deployment_without_openai_speech_key(
    azure_env,
    monkeypatch,
    tmp_path,
):
    settings = settings_with_ai()
    settings.paths.runtime_sessions_dir = str(tmp_path)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    fake_client = object()
    runtime = build_openai_runtime(settings, openai_client=fake_client, stt=object())
    for agent in runtime.controller.agents.values():
        assert agent.api_key is None
        assert agent.config.model == "custom-realtime-prod"
        assert agent.transport_context_factory is not None
        assert agent.timing_llm._client is fake_client
    assert runtime.session_summary_llm._client is fake_client
    manifest = runtime.ai_route_manifest
    assert manifest["realtime_speech"]["provider"] == "azure_eastus2"
    assert manifest["realtime_transcription"]["provider"] == "openai"
    assert "test-only-azure-key" not in json.dumps(manifest)


def test_speech_session_transcription_override_and_manifest_are_recorded(
    azure_env, tmp_path
):
    settings = settings_with_ai()
    settings.paths.runtime_sessions_dir = str(tmp_path)
    settings.ai.routes["realtime_speech"].targets[
        "azure_eastus2"
    ].input_transcription_model = "whisper-1"
    runtime = build_openai_runtime(settings, openai_client=object(), stt=object())
    assert runtime.controller.agents["client"].config.input_transcription == {
        "model": "whisper-1"
    }

    async def scenario():
        await runtime.logger.start()
        await runtime._log_resolved_audio_settings()
        await runtime.logger.close()

    asyncio.run(scenario())
    content = runtime.logger.paths.events_jsonl.read_text()
    assert "test-only-azure-key" not in content
    events = [json.loads(line) for line in content.splitlines()]
    assert events[0]["event_type"] == "ai_routes_resolved"
    assert (
        events[0]["details"]["routes"]["realtime_speech"]["model_ref"]
        == "custom-realtime-prod"
    )


def test_injected_transport_opens_without_agent_api_key():
    sentinel = object()
    agent = OpenAIRealtimeSpeechAgent(
        speaker="client",
        transport_context_factory=lambda: sentinel,
    )
    assert agent._open_transport() is sentinel
    with pytest.raises(ValueError, match="api_key"):
        OpenAIRealtimeSpeechAgent(speaker="client")._open_transport()


@pytest.mark.parametrize("route_name", ["realtime_speech", "realtime_transcription"])
def test_azure_transport_failure_does_not_retry_with_openai(azure_env, route_name):
    calls = []

    def fail_connect(url, **kwargs):
        calls.append((url, kwargs))
        raise ConnectionError("test connection failure")

    async def scenario():
        factory = ProviderRegistry(
            settings_with_stt_route()
        ).build_realtime_transport_factory(
            route_name,
            connect=fail_connect,
        )
        with pytest.raises(ConnectionError):
            async with factory():
                pytest.fail("connection must fail")

    asyncio.run(scenario())
    assert len(calls) == 1
    assert urlsplit(calls[0][0]).hostname == "example.openai.azure.com"
    headers = calls[0][1]["additional_headers"]
    assert set(headers) == {"api-key"}
    assert headers["api-key"] == "test-only-azure-key"


TEXT_MODELS = {
    "conversation_text": "llm_model",
    "turn_timing": "timing_llm_model",
    "session_summary": "summary_llm_model",
    "prompt_director": "prompting_llm_model",
}


def settings_with_text_routes():
    data = ai_config()
    for name in TEXT_MODELS:
        data["routes"][name] = {
            "provider": "azure_eastus2",
            "targets": {
                "openai": {"model": "gpt-5.4-mini"},
                "azure_eastus2": {"deployment": "custom-text-prod"},
            },
        }
    return load_runtime_config().model_copy(
        update={"ai": AISection.model_validate(data)}
    )


def settings_with_stt_route(provider="azure_eastus2", deployment="custom-stt-prod"):
    settings = settings_with_text_routes()
    data = settings.ai.model_dump()
    data["routes"]["realtime_transcription"] = {
        "provider": provider,
        "targets": {
            "openai": {"model": "gpt-4o-mini-transcribe"},
            "azure_eastus2": {"deployment": deployment},
        },
    }
    settings.ai = AISection.model_validate(data)
    return settings


@pytest.mark.parametrize("provider", ["openai", "azure_eastus2"])
def test_transcription_route_has_transcription_intent_and_selected_auth(
    azure_env, provider
):
    registry = ProviderRegistry(
        settings_with_stt_route(provider), openai_api_key="test-openai-key"
    )
    spec = registry.build_realtime_connection_spec("realtime_transcription")
    parsed = urlsplit(spec.url)
    assert parse_qs(parsed.query) == {"intent": ["transcription"]}
    if provider == "azure_eastus2":
        assert parsed.hostname == "example.openai.azure.com"
        assert parsed.path == "/openai/v1/realtime"
        assert dict(spec.headers) == {"api-key": "test-only-azure-key"}
    else:
        assert parsed.hostname == "api.openai.com"
        assert parsed.path == "/v1/realtime"
        assert dict(spec.headers) == {"Authorization": "Bearer test-openai-key"}
    assert "test-only-azure-key" not in repr(spec)


def test_omitted_stt_route_retains_legacy_openai_model():
    settings = settings_with_text_routes()
    route = ProviderRegistry(settings).resolve_route("realtime_transcription")
    assert route.provider_id == "openai"
    assert route.model_ref == settings.openai.stt_model


def test_all_azure_runtime_builds_stt_without_openai_key(
    azure_env, monkeypatch, tmp_path
):
    settings = settings_with_stt_route()
    settings.paths.runtime_sessions_dir = str(tmp_path)
    # Remove the key after loading settings, since load_dotenv can populate it.
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    def create_client(**kwargs):
        assert kwargs["api_key"] == "test-only-azure-key"
        return object()

    monkeypatch.setattr("openai.AsyncOpenAI", create_client)
    runtime = build_openai_runtime(settings)
    assert runtime.controller.stt.api_key is None
    assert runtime.controller.stt.model == "custom-stt-prod"
    assert runtime.controller.stt.transport_context_factory is not None
    assert (
        runtime.ai_route_manifest["realtime_transcription"]["provider"]
        == "azure_eastus2"
    )


def test_selecting_openai_stt_requires_its_key(azure_env, monkeypatch):
    from counseling_voice_demo.runtime.factory import RuntimeFactoryError

    settings = settings_with_stt_route(provider="openai")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(
        RuntimeFactoryError, match="realtime_transcription.*OPENAI_API_KEY"
    ):
        build_openai_runtime(settings)


@pytest.mark.parametrize("missing", ["TEST_AZURE_KEY", "TEST_AZURE_ENDPOINT"])
def test_transcription_preflight_validates_selected_provider(
    azure_env, monkeypatch, missing
):
    monkeypatch.delenv(missing)
    with pytest.raises(ProviderConfigurationError, match=missing):
        ProviderRegistry(settings_with_stt_route()).preflight_validate(
            ["realtime_transcription"]
        )


def test_unselected_stt_deployment_may_be_empty():
    registry = ProviderRegistry(
        settings_with_stt_route("openai", ""), openai_api_key="test-only-key"
    )
    registry.preflight_validate(["realtime_transcription"])


def test_transcription_control_override_is_independent_of_speech_and_text():
    settings = settings_with_stt_route()
    active = apply_runtime_start_options(
        settings,
        {
            "ai_routes": {
                "realtime_transcription": {
                    "provider": "openai",
                    "model_ref": "gpt-4o-transcribe",
                },
            }
        },
    )
    registry = ProviderRegistry(active)
    assert registry.resolve_route("realtime_transcription").provider_id == "openai"
    assert (
        registry.resolve_route("realtime_transcription").model_ref
        == "gpt-4o-transcribe"
    )
    assert registry.resolve_route("realtime_speech").provider_id == "azure_eastus2"
    assert registry.resolve_route("session_summary").provider_id == "azure_eastus2"
    assert settings.ai.routes["realtime_transcription"].provider == "azure_eastus2"


def test_injected_transcription_transport_needs_no_api_key():
    from counseling_voice_demo.runtime.streaming_stt import (
        OpenAIRealtimeTranscriptionSTT,
    )

    sentinel = object()
    stt = OpenAIRealtimeTranscriptionSTT(transport_context_factory=lambda: sentinel)
    assert stt._open_transport() is sentinel
    with pytest.raises(ValueError, match="api_key"):
        OpenAIRealtimeTranscriptionSTT()._open_transport()


@pytest.mark.parametrize("recording_mode", ["push_to_talk", "vad_auto"])
def test_azure_stt_streams_with_selected_deployment_and_recording_mode(
    azure_env,
    monkeypatch,
    tmp_path,
    recording_mode,
):
    from contextlib import asynccontextmanager
    from counseling_voice_demo.runtime.models import AudioChunk, EndOfAudio

    sent = []
    connections = []
    monkeypatch.setattr("openai.AsyncOpenAI", lambda **kwargs: object())

    async def scenario():
        ready = asyncio.Event()

        class WebSocket:
            async def send(self, message):
                event = json.loads(message)
                sent.append(event)
                if event["type"] == "input_audio_buffer.commit" or (
                    recording_mode == "vad_auto"
                    and event["type"] == "input_audio_buffer.append"
                ):
                    ready.set()

            def __aiter__(self):
                return self.events()

            async def events(self):
                await ready.wait()
                yield json.dumps(
                    {
                        "type": "conversation.item.input_audio_transcription.delta",
                        "delta": "相談です",
                    }
                )
                yield json.dumps(
                    {
                        "type": "conversation.item.input_audio_transcription.completed",
                        "transcript": "相談です。",
                    }
                )

        @asynccontextmanager
        async def connect(url, **kwargs):
            connections.append((url, kwargs))
            yield WebSocket()

        settings = settings_with_stt_route()
        settings.paths.runtime_sessions_dir = str(tmp_path)
        settings.openai.stt_model = "legacy-model-must-not-be-sent"
        runtime = build_openai_runtime(settings, stt_connect=connect)
        queue = asyncio.Queue()
        queue.put_nowait(
            AudioChunk(
                session_id="test",
                turn_id=1,
                speaker="counselor",
                chunk_index=0,
                pcm=b"\x00" * 4800,
            )
        )
        if recording_mode == "push_to_talk":
            queue.put_nowait(
                EndOfAudio(session_id="test", turn_id=1, speaker="counselor")
            )
        observed = []
        partials, final = await asyncio.wait_for(
            runtime.controller.stt.transcribe_from_queue_observed(
                session_id="test",
                turn_id=1,
                speaker="counselor",
                queue=queue,
                on_transcript=observed.append,
                recording_mode=recording_mode,
            ),
            timeout=2,
        )
        assert len(partials) == 1
        assert final.text == "相談です。"
        assert [event.transcript_type for event in observed] == ["partial", "final"]
        await runtime.controller.stt.close()

    asyncio.run(scenario())
    assert len(connections) == 1
    assert (
        connections[0][0]
        == "wss://example.openai.azure.com/openai/v1/realtime?intent=transcription"
    )
    assert set(connections[0][1]["additional_headers"]) == {"api-key"}
    audio_input = sent[0]["session"]["audio"]["input"]
    assert audio_input["transcription"] == {
        "model": "custom-stt-prod",
        "language": "ja",
    }
    assert audio_input["format"] == {"type": "audio/pcm", "rate": 24000}
    if recording_mode == "vad_auto":
        assert audio_input["turn_detection"]["type"] == "server_vad"
    else:
        assert audio_input["turn_detection"] is None


def test_missing_selected_stt_deployment_fails_before_creating_clients(
    azure_env, monkeypatch
):
    from counseling_voice_demo.runtime.factory import RuntimeFactoryError

    monkeypatch.setattr(
        "openai.AsyncOpenAI", lambda **kwargs: pytest.fail("client must not be created")
    )
    with pytest.raises(RuntimeFactoryError, match="realtime_transcription.*deployment"):
        build_openai_runtime(settings_with_stt_route(deployment=""))


@pytest.mark.parametrize("legacy_ai", [True, False])
def test_legacy_and_speech_only_configs_preserve_openai_text_models(legacy_ai):
    settings = settings_with_ai()
    if not legacy_ai:
        settings.ai = None
    registry = ProviderRegistry(settings)
    for name, field in TEXT_MODELS.items():
        route = registry.resolve_route(name)
        assert route.provider_id == "openai"
        assert route.model_ref == getattr(settings.openai, field)


@pytest.mark.parametrize("suffix", ["", "/", "/openai/v1", "/openai/v1/"])
def test_text_clients_use_normalized_azure_url_and_are_shared_per_provider(
    azure_env, monkeypatch, suffix
):
    from types import SimpleNamespace

    calls = []
    monkeypatch.setenv(
        "TEST_AZURE_ENDPOINT", "https://example.openai.azure.com" + suffix
    )
    monkeypatch.setattr(
        "openai.AsyncOpenAI",
        lambda **kwargs: calls.append(kwargs) or SimpleNamespace(),
    )
    registry = ProviderRegistry(settings_with_text_routes())
    clients = [registry.build_async_client(name) for name in TEXT_MODELS]
    assert all(client is clients[0] for client in clients)
    assert calls == [
        {
            "api_key": "test-only-azure-key",
            "base_url": "https://example.openai.azure.com/openai/v1/",
        }
    ]


def test_runtime_routes_text_clients_and_models_independently(
    azure_env, monkeypatch, tmp_path
):
    azure_client = object()
    openai_client = object()
    monkeypatch.setattr("openai.AsyncOpenAI", lambda **kwargs: azure_client)
    settings = settings_with_text_routes()
    settings.paths.runtime_sessions_dir = str(tmp_path)
    settings.ai.routes["turn_timing"].provider = "openai"
    runtime = build_openai_runtime(settings, openai_client=openai_client, stt=object())
    assert runtime.session_summary_llm._client is azure_client
    assert runtime.session_summary_llm._model == "custom-text-prod"
    assert runtime.prompt_director.llm._client is azure_client
    for agent in runtime.controller.agents.values():
        assert agent.timing_llm._client is openai_client
        assert agent.timing_llm._model == "gpt-5.4-mini"
    assert runtime.ai_route_manifest["session_summary"]["provider"] == "azure_eastus2"
    assert runtime.ai_route_manifest["turn_timing"]["provider"] == "openai"


def test_azure_text_runtime_does_not_require_openai_client_with_injected_stt(
    azure_env, monkeypatch, tmp_path
):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr("openai.AsyncOpenAI", lambda **kwargs: object())
    settings = settings_with_text_routes()
    settings.paths.runtime_sessions_dir = str(tmp_path)
    runtime = build_openai_runtime(settings, stt=object())
    assert (
        runtime.ai_route_manifest["prompt_director"]["model_ref"] == "custom-text-prod"
    )


def test_nonrealtime_conversation_uses_azure_but_tts_keeps_openai(
    azure_env, monkeypatch, tmp_path
):
    azure_client, openai_client = object(), object()
    monkeypatch.setattr("openai.AsyncOpenAI", lambda **kwargs: azure_client)
    settings = settings_with_text_routes()
    settings.paths.runtime_sessions_dir = str(tmp_path)
    settings.latency.realtime_api_centered_mode = False
    runtime = build_openai_runtime(settings, openai_client=openai_client, stt=object())
    for agent in runtime.controller.agents.values():
        assert agent.llm._client is azure_client
        assert agent.llm._model == "custom-text-prod"
    assert runtime.controller.tts._client is openai_client
    assert runtime.ai_route_manifest["conversation_text"]["provider"] == "azure_eastus2"
    assert runtime.ai_route_manifest["tts"]["provider"] == "openai"


@pytest.mark.parametrize("name,legacy_field", TEXT_MODELS.items())
def test_text_route_override_accepts_deployment_and_preserves_legacy_target(
    name, legacy_field
):
    settings = settings_with_text_routes()
    original = settings.model_dump()
    options = {
        "ai_routes": {name: {"provider": "azure_eastus2", "model_ref": "my-text-v2"}}
    }
    if name != "conversation_text":
        options[legacy_field] = "gpt-5.6-sol"
    active = apply_runtime_start_options(settings, options)
    assert ProviderRegistry(active).resolve_route(name).model_ref == "my-text-v2"
    assert active.ai.routes[name].targets["openai"].model == "gpt-5.4-mini"
    assert settings.model_dump() == original


@pytest.mark.parametrize("name,legacy_field", list(TEXT_MODELS.items())[1:])
def test_legacy_text_selector_updates_selected_openai_target(name, legacy_field):
    settings = settings_with_text_routes()
    active = apply_runtime_start_options(
        settings,
        {
            "ai_routes": {name: {"provider": "openai"}},
            legacy_field: "gpt-5.6-sol",
        },
    )
    assert ProviderRegistry(active).resolve_route(name).model_ref == "gpt-5.6-sol"
    assert (
        active.ai.routes[name].targets["azure_eastus2"].deployment == "custom-text-prod"
    )


@pytest.mark.parametrize("missing", ["TEST_AZURE_KEY", "TEST_AZURE_ENDPOINT"])
def test_text_preflight_requires_selected_azure_environment(
    azure_env, monkeypatch, missing
):
    monkeypatch.delenv(missing)
    with pytest.raises(ProviderConfigurationError, match=missing):
        ProviderRegistry(settings_with_text_routes()).preflight_validate(
            ["session_summary"]
        )


@pytest.mark.parametrize("failed", [False, True])
def test_azure_responses_stream_uses_deployment_and_does_not_fallback(
    azure_env, monkeypatch, failed
):
    import httpx
    from openai import AsyncOpenAI, BadRequestError
    from counseling_voice_demo.runtime.streaming_llm import OpenAIStreamingLLM

    requests = []

    def handle(request):
        requests.append(request)
        if failed:
            return httpx.Response(
                400,
                json={
                    "error": {
                        "message": "test failure",
                        "type": "invalid_request_error",
                    }
                },
            )
        message = {
            "id": "msg_test",
            "type": "message",
            "role": "assistant",
            "status": "in_progress",
            "content": [],
        }
        response = {"id": "resp_test", "object": "response", "output": []}
        events = [
            {"type": "response.created", "response": response, "sequence_number": 0},
            {
                "type": "response.output_item.added",
                "item": message,
                "output_index": 0,
                "sequence_number": 1,
            },
            {
                "type": "response.content_part.added",
                "item_id": "msg_test",
                "output_index": 0,
                "content_index": 0,
                "part": {"type": "output_text", "text": "", "annotations": []},
                "sequence_number": 2,
            },
            {
                "type": "response.output_text.delta",
                "delta": "接続成功",
                "item_id": "msg_test",
                "output_index": 0,
                "content_index": 0,
                "sequence_number": 3,
                "logprobs": [],
            },
            {"type": "response.completed", "response": response, "sequence_number": 4},
        ]
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content="".join(f"data: {json.dumps(event)}\n\n" for event in events),
        )

    monkeypatch.setattr(
        "openai.AsyncOpenAI",
        lambda **kwargs: AsyncOpenAI(
            **kwargs,
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(handle)),
        ),
    )

    async def scenario():
        registry = ProviderRegistry(settings_with_text_routes())
        client = registry.build_async_client("session_summary")
        llm = OpenAIStreamingLLM(
            client=client,
            model=registry.resolve_route("session_summary").model_ref,
            reasoning_effort="none",
            max_output_tokens=128,
        )
        try:
            if failed:
                with pytest.raises(BadRequestError):
                    _ = [
                        part
                        async for part in llm.stream_text(latest_input="接続テスト")
                    ]
            else:
                assert (
                    "".join(
                        [
                            part
                            async for part in llm.stream_text(latest_input="接続テスト")
                        ]
                    )
                    == "接続成功"
                )
        finally:
            await client.close()

    asyncio.run(scenario())
    assert len(requests) == 1
    assert (
        str(requests[0].url) == "https://example.openai.azure.com/openai/v1/responses"
    )
    payload = json.loads(requests[0].content)
    assert payload["model"] == "custom-text-prod"
    assert payload["reasoning"] == {"effort": "none"}
    assert payload["store"] is False


def test_missing_text_deployment_fails_before_client_creation(azure_env, monkeypatch):
    from counseling_voice_demo.runtime.factory import RuntimeFactoryError

    monkeypatch.setattr(
        "openai.AsyncOpenAI", lambda **kwargs: pytest.fail("client must not be created")
    )
    settings = settings_with_text_routes()
    settings.ai.routes["prompt_director"].targets["azure_eastus2"].deployment = ""
    with pytest.raises(RuntimeFactoryError, match="prompt_director.*deployment"):
        build_openai_runtime(settings, stt=object())
