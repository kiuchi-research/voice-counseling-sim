from __future__ import annotations

import asyncio
import builtins
import json
import sys
from types import SimpleNamespace

import pytest

import counseling_voice_demo.runtime.control_api as control_api
from counseling_voice_demo.runtime.audio_bus import AudioBus
from counseling_voice_demo.runtime.control_api import (
    RuntimeControlError,
    RuntimeControlService,
    apply_runtime_start_options,
    build_default_control_app,
    build_default_control_service,
    normalize_runtime_start_options,
    read_runtime_events,
    read_runtime_response_instructions,
    run_control_api,
    stream_audio_monitor_to_websocket,
)
from counseling_voice_demo.runtime.models import AudioChunk, EndOfAudio, RuntimePhase, RuntimeStatus
from counseling_voice_demo.runtime.models import (
    HumanAudioInput,
    HumanAudioStreamStart,
    HumanTurnInput,
)


class BlockingRuntime:
    def __init__(self, session_id: str = "session_control_test") -> None:
        self.session_id = session_id
        self.started = asyncio.Event()
        self.cancelled = False
        self._status = RuntimeStatus(session_id=session_id, phase=RuntimePhase.IDLE)

    @property
    def status(self) -> RuntimeStatus:
        return self._status

    async def run(self) -> None:
        self._status = RuntimeStatus(session_id=self.session_id, phase=RuntimePhase.RUNNING)
        self.started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.cancelled = True
            self._status = RuntimeStatus(session_id=self.session_id, phase=RuntimePhase.STOPPED)
            raise


class PausableBlockingRuntime(BlockingRuntime):
    def __init__(self, session_id: str = "session_pausable_control_test") -> None:
        super().__init__(session_id=session_id)
        self.pause_generation_calls = 0
        self.resume_generation_calls = 0

    async def pause_generation(self) -> None:
        self.pause_generation_calls += 1

    async def resume_generation(self) -> None:
        self.resume_generation_calls += 1


def test_director_recovery_pause_overrides_playback_throttle_until_explicit_resume():
    async def scenario():
        runtime = PausableBlockingRuntime()
        service = RuntimeControlService(lambda: runtime)
        await service.start()
        await runtime.started.wait()
        try:
            await service.pause(reason="generation_throttle")
            runtime._status = RuntimeStatus(
                session_id=runtime.session_id,
                phase=RuntimePhase.PAUSED,
                pause_reason="prompt_director_validation",
            )
            assert (await service.status()).pause_reason == "prompt_director_validation"
            assert (
                await service.pause(reason="playback")
            ).pause_reason == "prompt_director_validation"
            await service._resume_after_human_input_unlocked()
            assert runtime.resume_generation_calls == 0
            assert (await service.resume()).phase is RuntimePhase.RUNNING
            assert runtime.resume_generation_calls == 1
        finally:
            await service.stop()

    asyncio.run(scenario())


class InterruptibleBlockingRuntime(BlockingRuntime):
    def __init__(self, session_id: str = "session_interruptible_control_test") -> None:
        super().__init__(session_id=session_id)
        self.stop_current_response_playback_calls: list[dict[str, object]] = []

    async def stop_current_response_playback(self, **kwargs) -> dict[str, object]:
        self.stop_current_response_playback_calls.append(dict(kwargs))
        return {
            "played_ms": kwargs["played_ms"],
            "speaker": kwargs.get("speaker"),
            "cancel_sent": kwargs.get("cancel_response", True),
            "truncate_sent": kwargs.get("truncate_item", True),
            "sent_event_types": ["response.cancel", "conversation.item.truncate"],
        }


class FailingInterruptBlockingRuntime(BlockingRuntime):
    async def stop_current_response_playback(self, **kwargs) -> dict[str, object]:
        _ = kwargs
        raise RuntimeError("no active realtime speech session")


class SlowCancellationRuntime(BlockingRuntime):
    def __init__(self, session_id: str = "session_slow_cancel_test") -> None:
        super().__init__(session_id=session_id)
        self.cancel_requested = asyncio.Event()
        self.finish_cleanup = asyncio.Event()
        self.cleanup_finished = asyncio.Event()

    async def run(self) -> None:
        self._status = RuntimeStatus(session_id=self.session_id, phase=RuntimePhase.RUNNING)
        self.started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.cancel_requested.set()
            self._status = RuntimeStatus(session_id=self.session_id, phase=RuntimePhase.STOPPED)
            await self.finish_cleanup.wait()
            self.cancelled = True
            self.cleanup_finished.set()
            raise


class CompletingRuntime:
    def __init__(self, session_id: str = "session_completed_test") -> None:
        self.session_id = session_id
        self.started = asyncio.Event()
        self.allow_complete = asyncio.Event()
        self._status = RuntimeStatus(session_id=session_id, phase=RuntimePhase.IDLE)

    @property
    def status(self) -> RuntimeStatus:
        return self._status

    async def run(self) -> None:
        self._status = RuntimeStatus(session_id=self.session_id, phase=RuntimePhase.RUNNING)
        self.started.set()
        await self.allow_complete.wait()
        self._status = RuntimeStatus(session_id=self.session_id, phase=RuntimePhase.COMPLETED)


class FailingRuntime(BlockingRuntime):
    async def run(self) -> None:
        self._status = RuntimeStatus(session_id=self.session_id, phase=RuntimePhase.RUNNING)
        self.started.set()
        raise RuntimeError("Your session hit the maximum duration of 60 minutes.")


class RuntimeWithAudioBus(BlockingRuntime):
    def __init__(self, session_id: str = "session_monitor_queue_test") -> None:
        super().__init__(session_id=session_id)
        self.audio_bus = AudioBus()


class HumanInputRuntime(BlockingRuntime):
    def __init__(self, session_id: str = "session_human_input_test") -> None:
        super().__init__(session_id=session_id)
        self.human_audio_requests: list[HumanAudioInput] = []
        self.human_audio_stream_starts: list[HumanAudioStreamStart] = []
        self.human_audio_stream_chunks: list[tuple[str, bytes, int]] = []
        self.human_audio_stream_ends: list[str] = []
        self.human_turn_requests: list[HumanTurnInput] = []

    @property
    def status(self) -> RuntimeStatus:
        return RuntimeStatus(
            session_id=self.session_id,
            phase=RuntimePhase.RUNNING,
            awaiting_human_input=True,
            pending_human_turn_allowed=True,
        )

    async def submit_human_audio(self, request: HumanAudioInput) -> dict[str, object]:
        self.human_audio_requests.append(request)
        return {
            "accepted": True,
            "session_id": request.session_id,
            "source_audio_ref": "human_audio/session_human_input_test/turn_0001.wav",
        }

    async def start_human_audio_stream(
        self,
        request: HumanAudioStreamStart,
    ) -> dict[str, object]:
        self.human_audio_stream_starts.append(request)
        return {
            "accepted": True,
            "session_id": request.session_id,
            "stream_id": request.client_message_id or "stream-1",
            "turn_id": 1,
            "speaker": "counselor",
        }

    async def append_human_audio_stream_chunk(
        self,
        *,
        stream_id: str,
        audio_bytes: bytes,
        chunk_index: int,
    ) -> dict[str, object]:
        self.human_audio_stream_chunks.append((stream_id, audio_bytes, chunk_index))
        return {
            "accepted": True,
            "stream_id": stream_id,
            "chunk_index": chunk_index,
        }

    async def end_human_audio_stream(self, *, stream_id: str) -> dict[str, object]:
        self.human_audio_stream_ends.append(stream_id)
        return {"accepted": True, "stream_id": stream_id}

    async def submit_human_turn(self, request: HumanTurnInput) -> dict[str, object]:
        self.human_turn_requests.append(request)
        return {
            "accepted": True,
            "session_id": request.session_id,
            "turn_id": 1,
            "awaiting_human_input": False,
        }


class PausableHumanInputRuntime(HumanInputRuntime):
    def __init__(self, session_id: str = "session_human_input_test") -> None:
        super().__init__(session_id=session_id)
        self.pause_generation_calls = 0
        self.resume_generation_calls = 0

    async def pause_generation(self) -> None:
        self.pause_generation_calls += 1

    async def resume_generation(self) -> None:
        self.resume_generation_calls += 1


class RejectingPausableHumanAudioRuntime(PausableHumanInputRuntime):
    async def submit_human_audio(self, request: HumanAudioInput) -> dict[str, object]:
        self.human_audio_requests.append(request)
        return {
            "accepted": False,
            "session_id": request.session_id,
            "awaiting_human_input": True,
            "reason": "blank_transcript",
        }


class SlowCancellationRuntimeWithAudioBus(SlowCancellationRuntime):
    def __init__(self, session_id: str = "session_slow_cancel_monitor_test") -> None:
        super().__init__(session_id=session_id)
        self.audio_bus = AudioBus()


class RecordingWebSocket:
    def __init__(self) -> None:
        self.messages: list[dict[str, object]] = []

    async def send_json(self, data: dict[str, object]) -> None:
        self.messages.append(data)


async def wait_for_phase(service: RuntimeControlService, phase: RuntimePhase) -> RuntimeStatus:
    for _ in range(100):
        status = await service.status()
        if status.phase is phase:
            return status
        await asyncio.sleep(0.01)
    raise AssertionError(f"runtime phase did not become {phase}")


async def wait_for_runtime_cancelled(runtime: BlockingRuntime) -> None:
    for _ in range(100):
        if runtime.cancelled:
            return
        await asyncio.sleep(0.01)
    raise AssertionError("runtime task was not cancelled")


def make_runtime_settings(*, backend: str = "fake") -> SimpleNamespace:
    return SimpleNamespace(
        runtime=SimpleNamespace(
            backend=backend,
            control_host="127.0.0.9",
            control_port=9876,
            speaker_selection_policy="fixed_round_robin",
        ),
        latency=SimpleNamespace(realtime_api_centered_mode=True),
        audio=SimpleNamespace(speaker_gains={"counselor": 0.8, "client": 0.8}),
        openai=SimpleNamespace(realtime_model="gpt-realtime-mini"),
        paths=SimpleNamespace(runtime_sessions_dir="results/runtime-test-sessions"),
        participants={
            "counselor": SimpleNamespace(
                speaker_id="counselor",
                role="counselor",
                display_name="カウンセラー",
            ),
            "client": SimpleNamespace(
                speaker_id="client",
                role="client",
                display_name="クライアント",
            ),
        },
    )


def test_start_runs_runtime_in_background() -> None:
    async def scenario() -> None:
        runtimes: list[BlockingRuntime] = []
        service = RuntimeControlService(lambda: runtimes.append(BlockingRuntime()) or runtimes[-1])

        status = await service.start()
        await asyncio.wait_for(runtimes[0].started.wait(), timeout=1.0)

        assert status.phase is RuntimePhase.RUNNING
        assert (await service.status()).phase is RuntimePhase.RUNNING
        await service.stop()

    asyncio.run(scenario())


def test_run_completion_sets_completed_status() -> None:
    async def scenario() -> None:
        runtimes: list[CompletingRuntime] = []
        service = RuntimeControlService(lambda: runtimes.append(CompletingRuntime()) or runtimes[-1])

        await service.start()
        await asyncio.wait_for(runtimes[0].started.wait(), timeout=1.0)
        runtimes[0].allow_complete.set()

        status = await wait_for_phase(service, RuntimePhase.COMPLETED)
        assert status.phase is RuntimePhase.COMPLETED

    asyncio.run(scenario())


def test_run_error_status_includes_error_message() -> None:
    async def scenario() -> None:
        runtimes: list[FailingRuntime] = []
        service = RuntimeControlService(
            lambda: runtimes.append(FailingRuntime()) or runtimes[-1]
        )

        await service.start()
        await asyncio.wait_for(runtimes[0].started.wait(), timeout=1.0)

        status = await wait_for_phase(service, RuntimePhase.ERROR)
        assert status.phase is RuntimePhase.ERROR
        assert status.error_message == (
            "Your session hit the maximum duration of 60 minutes."
        )

    asyncio.run(scenario())


def test_service_exposes_active_runtime_monitor_queue() -> None:
    async def scenario() -> None:
        runtimes: list[RuntimeWithAudioBus] = []
        service = RuntimeControlService(
            lambda: runtimes.append(RuntimeWithAudioBus()) or runtimes[-1]
        )

        assert service.audio_monitor_queue() is None
        await service.start()
        await asyncio.wait_for(runtimes[0].started.wait(), timeout=1.0)

        assert service.audio_monitor_queue() is runtimes[0].audio_bus.monitor_queue
        await service.stop()

    asyncio.run(scenario())


def test_service_waits_for_audio_monitor_queue_until_runtime_starts() -> None:
    async def scenario() -> None:
        runtimes: list[RuntimeWithAudioBus] = []
        service = RuntimeControlService(
            lambda: runtimes.append(RuntimeWithAudioBus()) or runtimes[-1]
        )

        wait_task = asyncio.create_task(
            service.wait_for_audio_monitor_queue(timeout_seconds=1.0)
        )
        await asyncio.sleep(0)
        assert not wait_task.done()

        await service.start()
        await asyncio.wait_for(runtimes[0].started.wait(), timeout=1.0)

        assert await wait_task is runtimes[0].audio_bus.monitor_queue
        await service.stop()

    asyncio.run(scenario())


def test_service_audio_monitor_queue_wait_times_out() -> None:
    async def scenario() -> None:
        service = RuntimeControlService(lambda: RuntimeWithAudioBus())

        queue = await service.wait_for_audio_monitor_queue(
            timeout_seconds=0.01,
            poll_interval_seconds=0.01,
        )

        assert queue is None

    asyncio.run(scenario())


def test_service_creates_dedicated_audio_monitor_subscriptions() -> None:
    async def scenario() -> None:
        runtimes: list[RuntimeWithAudioBus] = []
        service = RuntimeControlService(
            lambda: runtimes.append(RuntimeWithAudioBus()) or runtimes[-1]
        )

        await service.start()
        await asyncio.wait_for(runtimes[0].started.wait(), timeout=1.0)
        first_queue, first_cleanup = await service.subscribe_audio_monitor_queue()
        second_queue, second_cleanup = await service.subscribe_audio_monitor_queue()
        chunk = AudioChunk(
            session_id="session-monitor-subscription",
            turn_id=2,
            speaker="client",
            chunk_index=0,
            pcm=b"client",
            duration_ms=1,
        )

        await runtimes[0].audio_bus.publish(chunk)

        assert first_queue is not second_queue
        assert await first_queue.get() is chunk
        assert await second_queue.get() is chunk

        first_cleanup()
        second_cleanup()
        await service.stop()

    asyncio.run(scenario())


def test_start_passes_options_to_option_aware_runtime_factory() -> None:
    async def scenario() -> None:
        captured_options: list[dict[str, object]] = []
        runtimes: list[CompletingRuntime] = []

        def runtime_factory(options) -> CompletingRuntime:
            captured_options.append(dict(options))
            runtime = CompletingRuntime(session_id="session-start-options")
            runtimes.append(runtime)
            return runtime

        service = RuntimeControlService(
            runtime_factory,
            runtime_factory_accepts_start_options=True,
        )

        await service.start({"realtime_api_centered_mode": False})
        await asyncio.wait_for(runtimes[0].started.wait(), timeout=1.0)
        runtimes[0].allow_complete.set()
        await wait_for_phase(service, RuntimePhase.COMPLETED)

        assert captured_options == [{"realtime_api_centered_mode": False}]

    asyncio.run(scenario())


def test_start_rejects_options_for_option_unaware_runtime_factory() -> None:
    async def scenario() -> None:
        service = RuntimeControlService(lambda: CompletingRuntime())

        with pytest.raises(RuntimeControlError, match="start options are not supported"):
            await service.start({"realtime_api_centered_mode": False})

    asyncio.run(scenario())


def test_build_default_control_service_uses_fake_runtime(monkeypatch) -> None:
    async def scenario() -> None:
        settings = make_runtime_settings()
        runtimes: list[CompletingRuntime] = []
        captured_configs: list[object] = []

        monkeypatch.setattr(
            control_api,
            "_load_default_control_config",
            lambda config=None: settings,
        )

        def fake_build_runtime(config) -> CompletingRuntime:
            captured_configs.append(config)
            runtime = CompletingRuntime(session_id="default-fake-runtime")
            runtimes.append(runtime)
            return runtime

        import counseling_voice_demo.runtime.factory as factory

        monkeypatch.setattr(factory, "build_fake_runtime", fake_build_runtime)

        service = build_default_control_service()
        await service.start()
        await asyncio.wait_for(runtimes[0].started.wait(), timeout=1.0)
        runtimes[0].allow_complete.set()
        await wait_for_phase(service, RuntimePhase.COMPLETED)

        assert captured_configs == [settings]

    asyncio.run(scenario())


def test_build_default_control_service_uses_openai_runtime_when_configured(monkeypatch) -> None:
    async def scenario() -> None:
        settings = make_runtime_settings(backend="openai")
        runtimes: list[CompletingRuntime] = []
        captured_configs: list[object] = []

        monkeypatch.setattr(control_api, "_load_default_control_config", lambda config=None: settings)

        def fake_build_runtime(config, **kwargs) -> CompletingRuntime:
            captured_configs.append(config)
            runtime = CompletingRuntime(session_id="default-openai-runtime")
            runtimes.append(runtime)
            return runtime

        import counseling_voice_demo.runtime.factory as factory

        monkeypatch.setattr(factory, "build_openai_runtime", fake_build_runtime)

        service = build_default_control_service()
        await service.start()
        await asyncio.wait_for(runtimes[0].started.wait(), timeout=1.0)
        runtimes[0].allow_complete.set()
        await wait_for_phase(service, RuntimePhase.COMPLETED)

        assert captured_configs == [settings]

    asyncio.run(scenario())


def test_build_default_control_service_applies_realtime_mode_start_option(monkeypatch) -> None:
    async def scenario() -> None:
        settings = make_runtime_settings(backend="openai")
        runtimes: list[CompletingRuntime] = []
        captured_realtime_modes: list[bool] = []
        captured_realtime_models: list[str] = []

        monkeypatch.setattr(control_api, "_load_default_control_config", lambda config=None: settings)

        def fake_build_runtime(config, **kwargs) -> CompletingRuntime:
            captured_realtime_modes.append(config.latency.realtime_api_centered_mode)
            captured_realtime_models.append(config.openai.realtime_model)
            runtime = CompletingRuntime(session_id="default-openai-runtime")
            runtimes.append(runtime)
            return runtime

        import counseling_voice_demo.runtime.factory as factory

        monkeypatch.setattr(factory, "build_openai_runtime", fake_build_runtime)

        service = build_default_control_service()
        await service.start(
            {
                "realtime_api_centered_mode": False,
                "realtime_model": "gpt-realtime-2",
            }
        )
        await asyncio.wait_for(runtimes[0].started.wait(), timeout=1.0)
        runtimes[0].allow_complete.set()
        await wait_for_phase(service, RuntimePhase.COMPLETED)

        assert captured_realtime_modes == [False]
        assert captured_realtime_models == ["gpt-realtime-2"]
        assert settings.latency.realtime_api_centered_mode is True
        assert settings.openai.realtime_model == "gpt-realtime-mini"

    asyncio.run(scenario())


def test_build_default_control_service_keeps_prompts_out_of_system_prompt_kwargs(
    monkeypatch,
) -> None:
    async def scenario() -> None:
        settings = make_runtime_settings(backend="openai")
        runtimes: list[CompletingRuntime] = []
        captured_kwargs: list[dict[str, object]] = []
        captured_participant_prompts: list[dict[str, object]] = []

        monkeypatch.setattr(control_api, "_load_default_control_config", lambda config=None: settings)

        def fake_build_runtime(config, **kwargs) -> CompletingRuntime:
            captured_kwargs.append(kwargs)
            captured_participant_prompts.append(
                {
                    participant_id: getattr(participant, "prompt_source", None)
                    for participant_id, participant in config.participants.items()
                }
            )
            runtime = CompletingRuntime(session_id="default-openai-runtime")
            runtimes.append(runtime)
            return runtime

        import counseling_voice_demo.runtime.factory as factory

        monkeypatch.setattr(factory, "build_openai_runtime", fake_build_runtime)

        service = build_default_control_service()
        await service.start(
            {
                "counselor_prompt": "counselor prompt",
                "client_prompt": "client prompt",
            }
        )
        await asyncio.wait_for(runtimes[0].started.wait(), timeout=1.0)
        runtimes[0].allow_complete.set()
        await wait_for_phase(service, RuntimePhase.COMPLETED)

        assert captured_kwargs == [{}]
        assert captured_participant_prompts == [
            {
                "counselor": "counselor prompt",
                "client": "client prompt",
            }
        ]

    asyncio.run(scenario())


def test_build_default_control_app_uses_configured_host_port_and_sessions_dir(monkeypatch) -> None:
    settings = make_runtime_settings()
    captured: dict[str, object] = {}
    app = object()

    monkeypatch.setattr(control_api, "_load_default_control_config", lambda config=None: settings)

    def fake_create_app(service, *, session_dir=None, sessions_dir=None, audio_queue=None):
        captured["service"] = service
        captured["session_dir"] = session_dir
        captured["sessions_dir"] = sessions_dir
        captured["audio_queue"] = audio_queue
        return app

    monkeypatch.setattr(control_api, "create_app", fake_create_app)

    built_app, host, port = build_default_control_app()

    assert built_app is app
    assert host == "127.0.0.9"
    assert port == 9876
    assert isinstance(captured["service"], RuntimeControlService)
    assert captured["session_dir"] is None
    assert captured["sessions_dir"] == "results/runtime-test-sessions"
    assert captured["audio_queue"] is None


def test_create_app_registers_session_monitor_audio_route(monkeypatch) -> None:
    class FakeWebSocket:
        pass

    class FakeWebSocketDisconnect(Exception):
        pass

    class FakeFastAPI:
        def __init__(self) -> None:
            self.routes: dict[tuple[str, str], object] = {}

        def post(self, path):
            return self._record("POST", path)

        def get(self, path):
            return self._record("GET", path)

        def websocket(self, path):
            return self._record("WS", path)

        def _record(self, method, path):
            def decorator(func):
                self.routes[(method, path)] = func
                return func

            return decorator

    fake_fastapi_module = SimpleNamespace(
        Body=lambda default=None: default,
        FastAPI=FakeFastAPI,
        HTTPException=Exception,
        WebSocket=FakeWebSocket,
        WebSocketDisconnect=FakeWebSocketDisconnect,
    )
    monkeypatch.setitem(sys.modules, "fastapi", fake_fastapi_module)

    service = RuntimeControlService(lambda: CompletingRuntime())
    app = control_api.create_app(service, audio_queue=asyncio.Queue())

    assert ("POST", "/runtime/start") in app.routes
    assert ("POST", "/runtime/human-audio") in app.routes
    assert ("POST", "/runtime/human-turn") in app.routes
    assert ("POST", "/runtime/playback-completed") in app.routes
    assert ("GET", "/runtime/status") in app.routes
    assert ("WS", "/runtime/monitor-audio") in app.routes
    assert ("WS", "/runtime/human-audio-stream") in app.routes
    assert ("WS", "/runtime/sessions/{session_id}/monitor-audio") in app.routes


def test_create_app_allows_local_browser_origins_for_component_start() -> None:
    service = RuntimeControlService(lambda: CompletingRuntime())
    app = control_api.create_app(service)

    cors_middleware = [
        middleware
        for middleware in app.user_middleware
        if getattr(middleware.cls, "__name__", "") == "CORSMiddleware"
    ]

    assert cors_middleware
    assert cors_middleware[0].kwargs["allow_origin_regex"] == (
        r"(null|https?://(localhost|127\.0\.0\.1|\[::1\])(:\d+)?)"
    )
    assert cors_middleware[0].kwargs["allow_methods"] == ["*"]
    assert cors_middleware[0].kwargs["allow_headers"] == ["*"]


def test_start_endpoint_accepts_json_start_options() -> None:
    class StartOnlyService:
        def __init__(self) -> None:
            self.captured_options: list[dict[str, object]] = []

        async def start(self, options=None) -> RuntimeStatus:
            self.captured_options.append(dict(options or {}))
            return RuntimeStatus(
                session_id="session_start_endpoint_test",
                phase=RuntimePhase.RUNNING,
            )

        async def stop(self) -> RuntimeStatus:
            return RuntimeStatus(
                session_id="session_start_endpoint_test",
                phase=RuntimePhase.STOPPED,
            )

        async def pause(self, reason=None) -> RuntimeStatus:
            return RuntimeStatus(
                session_id="session_start_endpoint_test",
                phase=RuntimePhase.PAUSED,
                pause_reason=reason,
            )

        async def resume(self) -> RuntimeStatus:
            return RuntimeStatus(
                session_id="session_start_endpoint_test",
                phase=RuntimePhase.RUNNING,
            )

        async def status(self) -> RuntimeStatus:
            return RuntimeStatus(
                session_id="session_start_endpoint_test",
                phase=RuntimePhase.RUNNING,
            )

    service = StartOnlyService()
    app = control_api.create_app(service)
    operation = app.openapi()["paths"]["/runtime/start"]["post"]

    assert operation.get("parameters") in (None, [])
    assert operation.get("requestBody", {}).get("required") is not True


def test_normalize_runtime_start_options_accepts_realtime_mode_override() -> None:
    assert normalize_runtime_start_options({"realtime_api_centered_mode": False}) == {
        "realtime_api_centered_mode": False
    }


def test_service_delegates_human_audio_and_turn_when_awaiting_input() -> None:
    async def scenario() -> None:
        runtimes: list[HumanInputRuntime] = []
        service = RuntimeControlService(
            lambda: runtimes.append(HumanInputRuntime()) or runtimes[-1]
        )
        await service.start()
        await asyncio.wait_for(runtimes[0].started.wait(), timeout=1.0)

        audio_result = await service.submit_human_audio(
            {
                "session_id": "session_human_input_test",
                "audio_bytes": b"\0\1",
                "sample_rate": 24000,
                "channels": 1,
                "recording_mode": "push_to_talk",
                "recipient_ids": ["client"],
                "client_message_id": "message-1",
            }
        )
        turn_result = await service.submit_human_turn(
            {
                "session_id": "session_human_input_test",
                "text": "今日はどんなことを相談したいですか。",
                "recipient_ids": ["client"],
                "input_mode": "human_mic_stt",
                "recording_mode": "push_to_talk",
                "source_audio_ref": "human_audio/session_human_input_test/turn_0001.wav",
                "client_message_id": "message-1",
            }
        )

        assert audio_result["accepted"] is True
        assert turn_result["turn_id"] == 1
        assert runtimes[0].human_audio_requests[0].recipient_ids == ("client",)
        assert runtimes[0].human_turn_requests[0].text == (
            "今日はどんなことを相談したいですか。"
        )
        await service.stop()

    asyncio.run(scenario())


def test_service_accepts_base64_human_audio_payload() -> None:
    async def scenario() -> None:
        runtimes: list[HumanInputRuntime] = []
        service = RuntimeControlService(
            lambda: runtimes.append(HumanInputRuntime()) or runtimes[-1]
        )
        await service.start()
        await asyncio.wait_for(runtimes[0].started.wait(), timeout=1.0)

        result = await service.submit_human_audio(
            {
                "session_id": "session_human_input_test",
                "audio_base64": "AAE=",
                "sample_rate": 24000,
                "channels": 1,
                "recipient_ids": ["client"],
            }
        )

        assert result["accepted"] is True
        assert runtimes[0].human_audio_requests[0].audio_bytes == b"\0\1"
        await service.stop()

    asyncio.run(scenario())


def test_service_streams_base64_human_audio_chunks() -> None:
    async def scenario() -> None:
        runtimes: list[HumanInputRuntime] = []
        service = RuntimeControlService(
            lambda: runtimes.append(HumanInputRuntime()) or runtimes[-1]
        )
        await service.start()
        await asyncio.wait_for(runtimes[0].started.wait(), timeout=1.0)

        start = await service.start_human_audio_stream(
            {
                "type": "start",
                "session_id": "session_human_input_test",
                "sample_rate": 24000,
                "channels": 1,
                "recipient_ids": ["client"],
                "client_message_id": "stream-message-1",
            }
        )
        chunk = await service.append_human_audio_stream_chunk(
            "stream-message-1",
            {"type": "chunk", "audio_base64": "AAE=", "chunk_index": 3},
        )
        end = await service.end_human_audio_stream("stream-message-1")

        assert start["accepted"] is True
        assert chunk["chunk_index"] == 3
        assert end["stream_id"] == "stream-message-1"
        assert runtimes[0].human_audio_stream_starts[0].recipient_ids == ("client",)
        assert runtimes[0].human_audio_stream_chunks == [
            ("stream-message-1", b"\0\1", 3)
        ]
        assert runtimes[0].human_audio_stream_ends == ["stream-message-1"]
        await service.stop()

    asyncio.run(scenario())


def test_service_accepts_human_audio_while_playback_paused_and_resumes_runtime() -> None:
    async def scenario() -> None:
        runtimes: list[PausableHumanInputRuntime] = []
        service = RuntimeControlService(
            lambda: runtimes.append(PausableHumanInputRuntime()) or runtimes[-1]
        )
        await service.start()
        await asyncio.wait_for(runtimes[0].started.wait(), timeout=1.0)
        paused = await service.pause(reason="playback")

        result = await service.submit_human_audio(
            {
                "session_id": "session_human_input_test",
                "audio_base64": "AAE=",
                "sample_rate": 24000,
                "channels": 1,
                "recipient_ids": ["client"],
            }
        )

        assert paused.phase is RuntimePhase.PAUSED
        assert result["accepted"] is True
        assert runtimes[0].pause_generation_calls == 1
        assert runtimes[0].resume_generation_calls == 1
        assert (await service.status()).phase is RuntimePhase.RUNNING
        await service.stop()

    asyncio.run(scenario())


def test_service_keeps_playback_pause_when_human_audio_is_not_accepted() -> None:
    async def scenario() -> None:
        runtimes: list[RejectingPausableHumanAudioRuntime] = []
        service = RuntimeControlService(
            lambda: runtimes.append(RejectingPausableHumanAudioRuntime())
            or runtimes[-1]
        )
        await service.start()
        await asyncio.wait_for(runtimes[0].started.wait(), timeout=1.0)
        paused = await service.pause(reason="playback")

        result = await service.submit_human_audio(
            {
                "session_id": "session_human_input_test",
                "audio_base64": "AAE=",
                "sample_rate": 24000,
                "channels": 1,
                "recipient_ids": ["client"],
            }
        )
        status = await service.status()

        assert paused.phase is RuntimePhase.PAUSED
        assert result["accepted"] is False
        assert result["reason"] == "blank_transcript"
        assert runtimes[0].pause_generation_calls == 1
        assert runtimes[0].resume_generation_calls == 0
        assert status.phase is RuntimePhase.PAUSED
        assert status.pause_reason == "playback"
        await service.stop()

    asyncio.run(scenario())


def test_service_rejects_human_turn_when_not_awaiting_input() -> None:
    async def scenario() -> None:
        runtimes: list[BlockingRuntime] = []
        service = RuntimeControlService(
            lambda: runtimes.append(BlockingRuntime()) or runtimes[-1]
        )
        await service.start()
        await asyncio.wait_for(runtimes[0].started.wait(), timeout=1.0)

        with pytest.raises(RuntimeControlError, match="awaiting human input"):
            await service.submit_human_turn(
                {
                    "session_id": "session_control_test",
                    "text": "質問です。",
                    "recipient_ids": ["client"],
                }
            )
        await service.stop()

    asyncio.run(scenario())


def test_normalize_runtime_start_options_accepts_prompt_and_runtime_options() -> None:
    assert normalize_runtime_start_options(
        {
            "initial_client_transcript": " 初回相談です。 ",
            "closing_start_elapsed_seconds": 180,
            "force_stop_after_closing_turns": 4,
            "speaker_selection_policy": " distributed_timing ",
            "counselor_prompt": " counselor ",
            "client_prompt": " client ",
            "counselor_display_name": " 佐伯 ",
            "client_display_name": " 高橋 ",
            "shared_case": " 共通プロフィール ",
            "realtime_model": " gpt-realtime-2.1 ",
            "prompting_llm_model": " gpt-5.6-luna ",
            "prompting_llm_reasoning_effort": " medium ",
            "timing_llm_model": " gpt-5.6-terra ",
            "timing_llm_reasoning_effort": " high ",
            "summary_llm_model": " gpt-5.6-luna ",
            "summary_llm_reasoning_effort": " low ",
            "counselor_tts_voice": " marin ",
            "client_tts_voice": " alloy ",
            "counselor_realtime_output_speed": 0.85,
            "client_realtime_output_speed": 0.95,
            "speaker_gains": {"counselor": 0.7, "client": 2.3},
        }
    ) == {
        "initial_client_transcript": "初回相談です。",
        "closing_start_elapsed_seconds": 180.0,
        "force_stop_after_closing_turns": 4,
        "speaker_selection_policy": "distributed_timing",
        "counselor_prompt": "counselor",
        "client_prompt": "client",
        "counselor_display_name": "佐伯",
        "client_display_name": "高橋",
        "shared_case": "共通プロフィール",
        "realtime_model": "gpt-realtime-2.1",
        "prompting_llm_model": "gpt-5.6-luna",
        "prompting_llm_reasoning_effort": "medium",
        "timing_llm_model": "gpt-5.6-terra",
        "timing_llm_reasoning_effort": "high",
        "summary_llm_model": "gpt-5.6-luna",
        "summary_llm_reasoning_effort": "low",
        "counselor_tts_voice": "marin",
        "client_tts_voice": "alloy",
        "counselor_realtime_output_speed": 0.85,
        "client_realtime_output_speed": 0.95,
        "speaker_gains": {"counselor": 0.7, "client": 2.3},
    }


def test_normalize_runtime_start_options_accepts_human_counselor_mode() -> None:
    assert normalize_runtime_start_options(
        {
            "interaction_mode": "human_counselor_ai_client",
            "participant_mode": "two_clients",
            "human_input_mode": "push_to_talk",
            "human_stt_submit_policy": "auto_on_final",
            "human_interrupts_enabled": False,
            "participants": {
                "client_a": {
                    "role": "client",
                    "display_name": "妻",
                    "voice": "marin",
                },
                "client_b": {
                    "role": "client",
                    "display_name": "夫",
                    "voice": "cedar",
                },
            },
        }
    ) == {
        "interaction_mode": "human_counselor_ai_client",
        "participant_mode": "two_clients",
        "human_input_mode": "push_to_talk",
        "human_stt_submit_policy": "auto_on_final",
        "human_interrupts_enabled": False,
        "participants": {
            "counselor": {
                "speaker_id": "counselor",
                "role": "counselor",
                "display_name": "カウンセラー",
                "actor_kind": "human",
            },
            "client_a": {
                "speaker_id": "client_a",
                "role": "client",
                "display_name": "妻",
                "voice": "marin",
                "actor_kind": "ai",
            },
            "client_b": {
                "speaker_id": "client_b",
                "role": "client",
                "display_name": "夫",
                "voice": "cedar",
                "actor_kind": "ai",
            },
        },
    }


def test_normalize_runtime_start_options_accepts_two_client_participants() -> None:
    assert normalize_runtime_start_options(
        {
            "participants": {
                "counselor": {
                    "role": "counselor",
                    "display_name": " 佐伯 ",
                    "public_profile_source": " カウンセラー公開 ",
                    "voice": " shimmer ",
                    "realtime_output_speed": 1.0,
                },
                "client_a": {
                    "role": "client",
                    "display_name": " 高橋A ",
                    "public_profile_source": " A公開 ",
                    "initial_transcript": " Aの初回発話 ",
                    "voice": " cedar ",
                    "realtime_output_speed": 0.9,
                },
                "client_b": {
                    "role": "client",
                    "display_name": " 高橋B ",
                    "public_profile_source": " B公開 ",
                    "initial_transcript": " Bの初回発話 ",
                    "voice": " coral ",
                    "realtime_output_speed": 0.85,
                },
            },
            "speaker_gains": {"counselor": 0.7, "client_a": 0.8, "client_b": 0.9},
        }
    ) == {
        "participants": {
            "counselor": {
                "speaker_id": "counselor",
                "role": "counselor",
                "display_name": "佐伯",
                "public_profile_source": "カウンセラー公開",
                "voice": "shimmer",
                "realtime_output_speed": 1.0,
            },
            "client_a": {
                "speaker_id": "client_a",
                "role": "client",
                "display_name": "高橋A",
                "public_profile_source": "A公開",
                "initial_transcript": "Aの初回発話",
                "voice": "cedar",
                "realtime_output_speed": 0.9,
            },
            "client_b": {
                "speaker_id": "client_b",
                "role": "client",
                "display_name": "高橋B",
                "public_profile_source": "B公開",
                "initial_transcript": "Bの初回発話",
                "voice": "coral",
                "realtime_output_speed": 0.85,
            },
        },
        "speaker_gains": {"counselor": 0.7, "client_a": 0.8, "client_b": 0.9},
    }


def test_normalize_runtime_start_options_accepts_fixed_speaker_sequence() -> None:
    assert normalize_runtime_start_options(
        {
            "fixed_speaker_sequence": [
                " counselor ",
                "client_a",
                "counselor",
                "client_b",
            ],
        }
    ) == {
        "fixed_speaker_sequence": (
            "counselor",
            "client_a",
            "counselor",
            "client_b",
        )
    }


def test_normalize_runtime_start_options_rejects_unsupported_timing_options() -> None:
    with pytest.raises(RuntimeControlError, match="prompting_llm_model"):
        normalize_runtime_start_options({"prompting_llm_model": "gpt-5.1"})
    with pytest.raises(RuntimeControlError, match="prompting_llm_reasoning_effort"):
        normalize_runtime_start_options(
            {"prompting_llm_reasoning_effort": "minimal"}
        )
    with pytest.raises(RuntimeControlError, match="timing_llm_model"):
        normalize_runtime_start_options({"timing_llm_model": "gpt-5.1"})
    with pytest.raises(RuntimeControlError, match="timing_llm_reasoning_effort"):
        normalize_runtime_start_options({"timing_llm_reasoning_effort": "minimal"})
    with pytest.raises(RuntimeControlError, match="summary_llm_model"):
        normalize_runtime_start_options({"summary_llm_model": "gpt-5.1"})
    with pytest.raises(RuntimeControlError, match="summary_llm_reasoning_effort"):
        normalize_runtime_start_options({"summary_llm_reasoning_effort": "minimal"})

    for option_name in (
        "prompting_llm_model",
        "timing_llm_model",
        "summary_llm_model",
    ):
        for removed_model in ("gpt-5.5", "gpt-5.4"):
            with pytest.raises(RuntimeControlError, match=option_name):
                normalize_runtime_start_options({option_name: removed_model})


def test_normalize_runtime_start_options_rejects_invalid_human_options() -> None:
    with pytest.raises(RuntimeControlError, match="interaction_mode"):
        normalize_runtime_start_options({"interaction_mode": "human_client"})
    with pytest.raises(RuntimeControlError, match="participant_mode"):
        normalize_runtime_start_options({"participant_mode": "three_clients"})
    with pytest.raises(RuntimeControlError, match="human_input_mode"):
        normalize_runtime_start_options({"human_input_mode": "keyboard"})
    with pytest.raises(RuntimeControlError, match="human_stt_submit_policy"):
        normalize_runtime_start_options({"human_stt_submit_policy": "manual_review"})
    with pytest.raises(RuntimeControlError, match="human_interrupts_enabled"):
        normalize_runtime_start_options({"human_interrupts_enabled": "yes"})


@pytest.mark.parametrize(
    "value",
    [
        ("counselor", "client_a", "counselor", "client_b"),
        '["counselor", "client_a", "counselor", "client_b"]',
        "counselor,client_a,counselor,client_b",
        "counselor\nclient_a\ncounselor\nclient_b",
        {"0": "counselor", "1": "client_a", "2": "counselor", "3": "client_b"},
        {"items": ["counselor", "client_a", "counselor", "client_b"]},
        {"value": '["counselor", "client_a", "counselor", "client_b"]'},
    ],
)
def test_normalize_runtime_start_options_accepts_serialized_fixed_speaker_sequence(
    value: object,
) -> None:
    assert normalize_runtime_start_options({"fixed_speaker_sequence": value}) == {
        "fixed_speaker_sequence": (
            "counselor",
            "client_a",
            "counselor",
            "client_b",
        )
    }


def test_normalize_runtime_start_options_rejects_unknown_options() -> None:
    with pytest.raises(RuntimeControlError, match="unsupported runtime start option"):
        normalize_runtime_start_options({"unknown": True})


def test_normalize_runtime_start_options_rejects_invalid_stop_options() -> None:
    with pytest.raises(RuntimeControlError, match="stop_condition"):
        normalize_runtime_start_options({"stop_condition": "audio"})
    with pytest.raises(RuntimeControlError, match="max_elapsed_seconds"):
        normalize_runtime_start_options({"max_elapsed_seconds": 0})
    with pytest.raises(RuntimeControlError, match="closing_start_elapsed_seconds"):
        normalize_runtime_start_options({"closing_start_elapsed_seconds": -1})
    with pytest.raises(RuntimeControlError, match="force_stop_after_closing_turns"):
        normalize_runtime_start_options({"force_stop_after_closing_turns": 0})
    with pytest.raises(RuntimeControlError, match="realtime_model"):
        normalize_runtime_start_options({"realtime_model": "gpt-realtime"})
    with pytest.raises(RuntimeControlError, match="client_tts_voice"):
        normalize_runtime_start_options({"client_tts_voice": ""})
    with pytest.raises(RuntimeControlError, match="realtime_output_speed"):
        normalize_runtime_start_options({"realtime_output_speed": 0.2})
    with pytest.raises(RuntimeControlError, match="client_realtime_output_speed"):
        normalize_runtime_start_options({"client_realtime_output_speed": 1.51})
    with pytest.raises(RuntimeControlError, match="speaker_gains"):
        normalize_runtime_start_options({"speaker_gains": {"counselor": 0}})
    with pytest.raises(RuntimeControlError, match="speaker_gains"):
        normalize_runtime_start_options({"speaker_gains": {"observer": 1.0}})
    with pytest.raises(RuntimeControlError, match="participants"):
        normalize_runtime_start_options({"participants": []})
    with pytest.raises(RuntimeControlError, match="fixed_speaker_sequence"):
        normalize_runtime_start_options({"fixed_speaker_sequence": "counselor"})
    with pytest.raises(RuntimeControlError, match="fixed_speaker_sequence"):
        normalize_runtime_start_options({"fixed_speaker_sequence": []})
    with pytest.raises(RuntimeControlError, match="fixed_speaker_sequence"):
        normalize_runtime_start_options({"fixed_speaker_sequence": ["counselor", ""]})
    with pytest.raises(RuntimeControlError, match="speaker_id"):
        normalize_runtime_start_options(
            {
                "participants": {
                    "counselor": {"role": "counselor", "display_name": "佐伯"},
                    "client_a": {
                        "speaker_id": "client_b",
                        "role": "client",
                        "display_name": "高橋",
                    },
                }
            }
        )
    with pytest.raises(RuntimeControlError, match="counselor"):
        normalize_runtime_start_options(
            {
                "participants": {
                    "client_a": {"role": "client", "display_name": "高橋"},
                }
            }
        )


def test_apply_runtime_start_options_overrides_realtime_mode_without_mutating_source() -> None:
    settings = make_runtime_settings(backend="openai")

    updated = apply_runtime_start_options(
        settings,
        {
            "realtime_api_centered_mode": False,
            "initial_client_transcript": "初回相談です。",
            "closing_start_elapsed_seconds": 300,
            "force_stop_after_closing_turns": 5,
            "speaker_selection_policy": "distributed_timing",
            "shared_case": "共通プロフィール",
            "realtime_model": "gpt-realtime-2",
            "prompting_llm_model": "gpt-5.6-luna",
            "prompting_llm_reasoning_effort": "medium",
            "timing_llm_model": "gpt-5.6-terra",
            "timing_llm_reasoning_effort": "high",
            "summary_llm_model": "gpt-5.6-luna",
            "summary_llm_reasoning_effort": "low",
            "counselor_tts_voice": "marin",
            "client_tts_voice": "alloy",
            "counselor_realtime_output_speed": 0.85,
            "client_realtime_output_speed": 0.95,
            "speaker_gains": {"counselor": 0.6, "client": 2.1},
        },
    )

    assert updated.latency.realtime_api_centered_mode is False
    assert updated.runtime.initial_client_transcript == "初回相談です。"
    assert updated.runtime.closing_start_elapsed_seconds == 300.0
    assert updated.runtime.force_stop_after_closing_turns == 5
    assert updated.runtime.speaker_selection_policy == "distributed_timing"
    assert updated.shared_case.prompt_source == "共通プロフィール"
    assert updated.shared_case.prompt_path is None
    assert updated.openai.realtime_model == "gpt-realtime-2"
    assert updated.openai.prompting_llm_model == "gpt-5.6-luna"
    assert updated.openai.prompting_llm_reasoning_effort == "medium"
    assert updated.openai.timing_llm_model == "gpt-5.6-terra"
    assert updated.openai.timing_llm_reasoning_effort == "high"
    assert updated.openai.summary_llm_model == "gpt-5.6-luna"
    assert updated.openai.summary_llm_reasoning_effort == "low"
    assert updated.openai.counselor_tts_voice == "marin"
    assert updated.openai.client_tts_voice == "alloy"
    assert updated.openai.counselor_realtime_output_speed == 0.85
    assert updated.openai.client_realtime_output_speed == 0.95
    assert updated.audio.speaker_gains == {"counselor": 0.6, "client": 2.1}
    assert settings.latency.realtime_api_centered_mode is True
    assert not hasattr(settings.runtime, "initial_client_transcript")
    assert settings.runtime.speaker_selection_policy == "fixed_round_robin"
    assert not hasattr(settings, "shared_case")
    assert settings.openai.realtime_model == "gpt-realtime-mini"
    assert not hasattr(settings.openai, "prompting_llm_model")
    assert not hasattr(settings.openai, "prompting_llm_reasoning_effort")
    assert not hasattr(settings.openai, "timing_llm_model")
    assert not hasattr(settings.openai, "timing_llm_reasoning_effort")
    assert not hasattr(settings.openai, "summary_llm_model")
    assert not hasattr(settings.openai, "summary_llm_reasoning_effort")
    assert settings.audio.speaker_gains == {"counselor": 0.8, "client": 0.8}


def test_apply_runtime_start_options_routes_role_prompts_to_participant_context() -> None:
    settings = make_runtime_settings(backend="openai")

    updated = apply_runtime_start_options(
        settings,
        {
            "counselor_prompt": "セッション用カウンセラー指示",
            "client_prompt": "セッション用クライアント指示",
        },
    )

    assert updated.participants["counselor"].prompt_source == (
        "セッション用カウンセラー指示"
    )
    assert updated.participants["client"].prompt_source == (
        "セッション用クライアント指示"
    )
    assert not hasattr(settings.participants["counselor"], "prompt_source")
    assert not hasattr(settings.participants["client"], "prompt_source")


def test_apply_runtime_start_options_applies_common_client_prompt_to_each_client() -> None:
    settings = make_runtime_settings(backend="openai")

    updated = apply_runtime_start_options(
        settings,
        {
            "client_prompt": "全クライアント共通のセッション指示",
            "participants": {
                "counselor": {
                    "role": "counselor",
                    "display_name": "佐伯",
                },
                "client_a": {
                    "role": "client",
                    "display_name": "妻",
                },
                "client_b": {
                    "role": "client",
                    "display_name": "夫",
                },
            },
        },
    )

    assert updated.participants["client_a"].prompt_source == (
        "全クライアント共通のセッション指示"
    )
    assert updated.participants["client_b"].prompt_source == (
        "全クライアント共通のセッション指示"
    )


def test_apply_runtime_start_options_updates_human_counselor_mode() -> None:
    settings = make_runtime_settings(backend="openai")

    updated = apply_runtime_start_options(
        settings,
        {
            "interaction_mode": "human_counselor_ai_client",
            "participant_mode": "two_clients",
            "human_input_mode": "push_to_talk",
            "human_stt_submit_policy": "auto_on_final",
            "human_interrupts_enabled": False,
        },
    )

    assert updated.runtime.interaction_mode == "human_counselor_ai_client"
    assert updated.runtime.participant_mode == "two_clients"
    assert updated.runtime.human_input_mode == "push_to_talk"
    assert updated.runtime.human_stt_submit_policy == "auto_on_final"
    assert updated.runtime.human_interrupts_enabled is False
    assert updated.participants["counselor"].actor_kind == "human"
    assert not hasattr(settings.runtime, "interaction_mode")
    assert not hasattr(settings.participants["counselor"], "actor_kind")


def test_apply_runtime_start_options_updates_participants_without_mutating_source() -> None:
    settings = make_runtime_settings(backend="openai")

    updated = apply_runtime_start_options(
        settings,
        {
            "participants": {
                "counselor": {
                    "role": "counselor",
                    "display_name": "佐伯",
                    "voice": "shimmer",
                },
                "client_a": {
                    "role": "client",
                    "display_name": "高橋A",
                    "public_profile_source": "A公開",
                    "initial_transcript": "Aの初回発話",
                    "voice": "cedar",
                },
                "client_b": {
                    "role": "client",
                    "display_name": "高橋B",
                    "initial_transcript": "Bの初回発話",
                    "voice": "coral",
                },
            },
            "speaker_gains": {"client_a": 0.7, "client_b": 0.8},
        },
    )

    assert set(updated.participants) == {"counselor", "client_a", "client_b"}
    assert updated.participants["client_a"].display_name == "高橋A"
    assert updated.participants["client_a"].public_profile_source == "A公開"
    assert updated.participants["client_a"].initial_transcript == "Aの初回発話"
    assert updated.participants["client_b"].voice == "coral"
    assert updated.audio.speaker_gains == {
        "counselor": 0.8,
        "client": 0.8,
        "client_a": 0.7,
        "client_b": 0.8,
    }
    assert set(settings.participants) == {"counselor", "client"}
    assert settings.audio.speaker_gains == {"counselor": 0.8, "client": 0.8}


def test_apply_runtime_start_options_updates_fixed_speaker_sequence() -> None:
    settings = make_runtime_settings(backend="openai")

    updated = apply_runtime_start_options(
        settings,
        {
            "fixed_speaker_sequence": [
                "counselor",
                "client_a",
                "counselor",
                "client_b",
            ],
        },
    )

    assert updated.runtime.fixed_speaker_sequence == (
        "counselor",
        "client_a",
        "counselor",
        "client_b",
    )
    assert not hasattr(settings.runtime, "fixed_speaker_sequence")


def test_monitor_audio_websocket_accepts_connection_and_streams_json(monkeypatch) -> None:
    class FakeWebSocket(RecordingWebSocket):
        def __init__(self) -> None:
            super().__init__()
            self.accepted = False
            self.closed = False

        async def accept(self) -> None:
            self.accepted = True

        async def close(self) -> None:
            self.closed = True

    class FakeWebSocketDisconnect(Exception):
        pass

    class FakeFastAPI:
        def __init__(self) -> None:
            self.routes: dict[tuple[str, str], object] = {}

        def post(self, path):
            return self._record("POST", path)

        def get(self, path):
            return self._record("GET", path)

        def websocket(self, path):
            return self._record("WS", path)

        def _record(self, method, path):
            def decorator(func):
                self.routes[(method, path)] = func
                return func

            return decorator

    fake_fastapi_module = SimpleNamespace(
        Body=lambda default=None: default,
        FastAPI=FakeFastAPI,
        HTTPException=Exception,
        WebSocket=FakeWebSocket,
        WebSocketDisconnect=FakeWebSocketDisconnect,
    )
    monkeypatch.setitem(sys.modules, "fastapi", fake_fastapi_module)

    async def scenario() -> None:
        queue: asyncio.Queue[object] = asyncio.Queue()
        await queue.put(
            AudioChunk(
                session_id="session-monitor-websocket-test",
                turn_id=1,
                speaker="counselor",
                chunk_index=0,
                pcm=b"\x00\x00\x01\x00",
                duration_ms=1,
            )
        )
        await queue.put(EndOfAudio())

        service = RuntimeControlService(lambda: CompletingRuntime())
        app = control_api.create_app(service, audio_queue=queue)
        websocket = FakeWebSocket()

        handler = app.routes[("WS", "/runtime/monitor-audio")]
        await handler(websocket)

        assert websocket.accepted is True
        assert websocket.closed is True
        assert websocket.messages[0]["type"] == "audio_chunk"
        assert (
            websocket.messages[0]["metadata"]["session_id"]
            == "session-monitor-websocket-test"
        )
        assert websocket.messages[1]["type"] == "stream_end"

    asyncio.run(scenario())


def test_monitor_audio_websocket_waits_long_enough_for_runtime_start(monkeypatch) -> None:
    class FakeWebSocket(RecordingWebSocket):
        def __init__(self) -> None:
            super().__init__()
            self.accepted = False
            self.closed = False

        async def accept(self) -> None:
            self.accepted = True

        async def close(self) -> None:
            self.closed = True

    class FakeWebSocketDisconnect(Exception):
        pass

    class FakeFastAPI:
        def __init__(self) -> None:
            self.routes: dict[tuple[str, str], object] = {}

        def post(self, path):
            return self._record("POST", path)

        def get(self, path):
            return self._record("GET", path)

        def websocket(self, path):
            return self._record("WS", path)

        def _record(self, method, path):
            def decorator(func):
                self.routes[(method, path)] = func
                return func

            return decorator

    class CapturingService:
        def __init__(self) -> None:
            self.timeout_seconds: list[float] = []

        async def subscribe_audio_monitor_queue(self, *, timeout_seconds: float):
            self.timeout_seconds.append(timeout_seconds)
            queue: asyncio.Queue[object] = asyncio.Queue()
            await queue.put(EndOfAudio())
            return queue, lambda: None

        async def start(self, options=None) -> RuntimeStatus:
            return RuntimeStatus(
                session_id="session-monitor-wait-test",
                phase=RuntimePhase.RUNNING,
            )

        async def stop(self) -> RuntimeStatus:
            return RuntimeStatus(
                session_id="session-monitor-wait-test",
                phase=RuntimePhase.STOPPED,
            )

        async def pause(self, reason=None) -> RuntimeStatus:
            return RuntimeStatus(
                session_id="session-monitor-wait-test",
                phase=RuntimePhase.PAUSED,
                pause_reason=reason,
            )

        async def resume(self) -> RuntimeStatus:
            return RuntimeStatus(
                session_id="session-monitor-wait-test",
                phase=RuntimePhase.RUNNING,
            )

        async def status(self) -> RuntimeStatus:
            return RuntimeStatus(
                session_id="session-monitor-wait-test",
                phase=RuntimePhase.RUNNING,
            )

    fake_fastapi_module = SimpleNamespace(
        Body=lambda default=None: default,
        FastAPI=FakeFastAPI,
        HTTPException=Exception,
        WebSocket=FakeWebSocket,
        WebSocketDisconnect=FakeWebSocketDisconnect,
    )
    monkeypatch.setitem(sys.modules, "fastapi", fake_fastapi_module)

    async def scenario() -> None:
        service = CapturingService()
        app = control_api.create_app(service)
        websocket = FakeWebSocket()

        handler = app.routes[("WS", "/runtime/monitor-audio")]
        await handler(websocket)

        assert websocket.accepted is True
        assert websocket.closed is True
        assert service.timeout_seconds == [
            control_api.MONITOR_AUDIO_RUNTIME_WAIT_SECONDS
        ]
        assert service.timeout_seconds[0] >= 60.0 * 60.0

    asyncio.run(scenario())


def test_run_control_api_calls_uvicorn_with_app_host_and_port(monkeypatch) -> None:
    calls: list[dict[str, object]] = []
    app = object()
    fake_uvicorn = SimpleNamespace(
        run=lambda passed_app, *, host, port: calls.append(
            {"app": passed_app, "host": host, "port": port}
        )
    )

    monkeypatch.setitem(sys.modules, "uvicorn", fake_uvicorn)

    run_control_api(app, host="127.0.0.9", port=9876)

    assert sys.modules["uvicorn"] is fake_uvicorn
    assert calls == [{"app": app, "host": "127.0.0.9", "port": 9876}]


def test_run_control_api_raises_clear_error_when_uvicorn_is_missing(monkeypatch) -> None:
    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "uvicorn":
            raise ModuleNotFoundError("No module named 'uvicorn'")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(RuntimeControlError, match="uvicorn is not installed"):
        run_control_api(object(), host="127.0.0.1", port=8765)


def test_main_builds_default_app_and_runs_control_api(monkeypatch) -> None:
    app = object()
    calls: list[dict[str, object]] = []

    monkeypatch.setattr(
        control_api,
        "build_default_control_app",
        lambda config_path=None: (app, "127.0.0.9", 9876),
    )
    monkeypatch.setattr(
        control_api,
        "run_control_api",
        lambda passed_app, *, host, port: calls.append(
            {"app": passed_app, "host": host, "port": port}
        ),
    )

    control_api.main(["config/runtime_config.yaml"])

    assert calls == [{"app": app, "host": "127.0.0.9", "port": 9876}]


def test_start_while_running_returns_current_runtime_status() -> None:
    async def scenario() -> None:
        runtimes: list[BlockingRuntime] = []
        service = RuntimeControlService(lambda: runtimes.append(BlockingRuntime()) or runtimes[-1])

        first_status = await service.start()
        await asyncio.wait_for(runtimes[0].started.wait(), timeout=1.0)
        second_status = await service.start()

        assert first_status.phase is RuntimePhase.RUNNING
        assert second_status.phase is RuntimePhase.RUNNING
        assert second_status.session_id == first_status.session_id
        assert len(runtimes) == 1
        await service.stop()

    asyncio.run(scenario())


def test_start_while_running_restarts_when_start_options_change() -> None:
    async def scenario() -> None:
        runtimes: list[BlockingRuntime] = []

        def factory(options: dict[str, object]) -> BlockingRuntime:
            runtime = BlockingRuntime(session_id=f"session_mode_{len(runtimes)}")
            runtime.start_options = dict(options)
            runtimes.append(runtime)
            return runtime

        service = RuntimeControlService(
            factory,
            runtime_factory_accepts_start_options=True,
        )

        first_status = await service.start(
            {"interaction_mode": "ai_counselor_ai_client"}
        )
        await asyncio.wait_for(runtimes[0].started.wait(), timeout=1.0)
        second_status = await service.start(
            {"interaction_mode": "human_counselor_ai_client"}
        )
        await asyncio.wait_for(runtimes[1].started.wait(), timeout=1.0)

        assert first_status.session_id == "session_mode_0"
        assert second_status.session_id == "session_mode_1"
        assert len(runtimes) == 2
        assert runtimes[1].start_options == {
            "interaction_mode": "human_counselor_ai_client"
        }
        await wait_for_runtime_cancelled(runtimes[0])
        await service.stop()

    asyncio.run(scenario())


def test_stop_cancels_running_task_and_marks_stopped() -> None:
    async def scenario() -> None:
        runtimes: list[BlockingRuntime] = []
        service = RuntimeControlService(lambda: runtimes.append(BlockingRuntime()) or runtimes[-1])

        await service.start()
        await asyncio.wait_for(runtimes[0].started.wait(), timeout=1.0)
        status = await service.stop()

        assert status.phase is RuntimePhase.STOPPED
        await wait_for_runtime_cancelled(runtimes[0])
        assert runtimes[0].cancelled is True
        assert (await service.status()).phase is RuntimePhase.STOPPED

    asyncio.run(scenario())


def test_stop_returns_before_runtime_cancellation_cleanup_finishes() -> None:
    async def scenario() -> None:
        runtimes: list[SlowCancellationRuntime] = []
        service = RuntimeControlService(
            lambda: runtimes.append(SlowCancellationRuntime()) or runtimes[-1]
        )

        await service.start()
        await asyncio.wait_for(runtimes[0].started.wait(), timeout=1.0)
        status = await asyncio.wait_for(service.stop(), timeout=0.1)

        assert status.phase is RuntimePhase.STOPPED
        assert (await service.status()).phase is RuntimePhase.STOPPED
        assert runtimes[0].cleanup_finished.is_set() is False

        await asyncio.wait_for(runtimes[0].cancel_requested.wait(), timeout=1.0)
        assert runtimes[0].cancelled is False

        runtimes[0].finish_cleanup.set()
        await asyncio.wait_for(runtimes[0].cleanup_finished.wait(), timeout=1.0)

    asyncio.run(scenario())


def test_start_after_stop_can_switch_options_before_previous_cleanup_finishes() -> None:
    async def scenario() -> None:
        captured_options: list[dict[str, object]] = []
        runtimes: list[BlockingRuntime] = []

        def runtime_factory(options) -> BlockingRuntime:
            captured_options.append(dict(options))
            if not runtimes:
                runtime = SlowCancellationRuntime(session_id="session_responses_cleanup")
            else:
                runtime = BlockingRuntime(session_id="session_realtime_restart")
            runtimes.append(runtime)
            return runtime

        service = RuntimeControlService(
            runtime_factory,
            runtime_factory_accepts_start_options=True,
        )

        await service.start({"realtime_api_centered_mode": False})
        await asyncio.wait_for(runtimes[0].started.wait(), timeout=1.0)
        stop_status = await service.stop()
        await asyncio.wait_for(runtimes[0].cancel_requested.wait(), timeout=1.0)

        assert stop_status.phase is RuntimePhase.STOPPED
        assert runtimes[0].cleanup_finished.is_set() is False

        restart_status = await service.start({"realtime_api_centered_mode": True})
        await asyncio.wait_for(runtimes[1].started.wait(), timeout=1.0)

        assert restart_status.phase is RuntimePhase.RUNNING
        assert restart_status.session_id == "session_realtime_restart"
        assert captured_options == [
            {"realtime_api_centered_mode": False},
            {"realtime_api_centered_mode": True},
        ]

        runtimes[0].finish_cleanup.set()
        await asyncio.wait_for(runtimes[0].cleanup_finished.wait(), timeout=1.0)

        status = await service.status()
        assert status.phase is RuntimePhase.RUNNING
        assert status.session_id == "session_realtime_restart"

        await service.stop()
        await wait_for_runtime_cancelled(runtimes[1])

    asyncio.run(scenario())


def test_audio_monitor_subscription_waits_for_new_runtime_after_stop() -> None:
    async def scenario() -> None:
        runtimes: list[BlockingRuntime] = []

        def runtime_factory() -> BlockingRuntime:
            if not runtimes:
                runtime = SlowCancellationRuntimeWithAudioBus(
                    session_id="session_stopped_monitor"
                )
            else:
                runtime = RuntimeWithAudioBus(session_id="session_restart_monitor")
            runtimes.append(runtime)
            return runtime

        service = RuntimeControlService(runtime_factory)

        await service.start()
        await asyncio.wait_for(runtimes[0].started.wait(), timeout=1.0)
        assert service.audio_bus() is runtimes[0].audio_bus

        await service.stop()
        await asyncio.wait_for(runtimes[0].cancel_requested.wait(), timeout=1.0)
        assert service.audio_bus() is None
        assert service.audio_monitor_queue() is None

        subscribe_task = asyncio.create_task(
            service.subscribe_audio_monitor_queue(timeout_seconds=1.0)
        )
        await asyncio.sleep(0)
        assert subscribe_task.done() is False

        await service.start()
        await asyncio.wait_for(runtimes[1].started.wait(), timeout=1.0)
        queue, cleanup = await asyncio.wait_for(subscribe_task, timeout=1.0)
        chunk = AudioChunk(
            session_id="session_restart_monitor",
            turn_id=1,
            speaker="counselor",
            chunk_index=0,
            pcm=b"new-runtime",
            duration_ms=1,
        )

        await runtimes[1].audio_bus.publish(chunk)

        assert await asyncio.wait_for(queue.get(), timeout=1.0) is chunk

        cleanup()
        runtimes[0].finish_cleanup.set()
        await asyncio.wait_for(runtimes[0].cleanup_finished.wait(), timeout=1.0)
        await service.stop()
        await wait_for_runtime_cancelled(runtimes[1])

    asyncio.run(scenario())


def test_pause_and_resume_update_service_phase() -> None:
    async def scenario() -> None:
        runtimes: list[PausableBlockingRuntime] = []
        service = RuntimeControlService(
            lambda: runtimes.append(PausableBlockingRuntime()) or runtimes[-1]
        )

        await service.start()
        await asyncio.wait_for(runtimes[0].started.wait(), timeout=1.0)

        paused = await service.pause()
        resumed = await service.resume()

        assert paused.phase is RuntimePhase.PAUSED
        assert paused.pause_reason == "manual"
        assert resumed.phase is RuntimePhase.RUNNING
        assert resumed.pause_reason is None
        assert runtimes[0].pause_generation_calls == 1
        assert runtimes[0].resume_generation_calls == 1
        await service.stop()

    asyncio.run(scenario())


def test_pause_updates_reason_while_already_paused() -> None:
    async def scenario() -> None:
        runtimes: list[PausableBlockingRuntime] = []
        service = RuntimeControlService(
            lambda: runtimes.append(PausableBlockingRuntime()) or runtimes[-1]
        )

        await service.start()
        await asyncio.wait_for(runtimes[0].started.wait(), timeout=1.0)

        auto_paused = await service.pause(reason="generation_throttle")
        playback_paused = await service.pause(reason="playback")

        assert auto_paused.phase is RuntimePhase.PAUSED
        assert auto_paused.pause_reason == "generation_throttle"
        assert playback_paused.phase is RuntimePhase.PAUSED
        assert playback_paused.pause_reason == "playback"
        assert runtimes[0].pause_generation_calls == 1
        await service.stop()

    asyncio.run(scenario())


def test_interrupt_playback_forwards_played_ms_to_runtime() -> None:
    async def scenario() -> None:
        runtimes: list[InterruptibleBlockingRuntime] = []
        service = RuntimeControlService(
            lambda: runtimes.append(InterruptibleBlockingRuntime()) or runtimes[-1]
        )

        await service.start()
        await asyncio.wait_for(runtimes[0].started.wait(), timeout=1.0)

        result = await service.interrupt_playback(
            {
                "played_ms": 1250,
                "speaker": "counselor",
                "turn_id": 3,
                "reason": "browser_overlap",
            }
        )

        assert runtimes[0].stop_current_response_playback_calls == [
            {
                "played_ms": 1250,
                "speaker": "counselor",
                "turn_id": 3,
                "item_id": None,
                "response_id": None,
                "content_index": None,
                "cancel_response": True,
                "truncate_item": True,
                "reason": "browser_overlap",
            }
        ]
        assert result["played_ms"] == 1250
        assert result["speaker"] == "counselor"
        await service.stop()

    asyncio.run(scenario())


def test_interrupt_playback_rejects_invalid_payload() -> None:
    async def scenario() -> None:
        service = RuntimeControlService(lambda: InterruptibleBlockingRuntime())

        with pytest.raises(RuntimeControlError, match="played_ms"):
            await service.interrupt_playback({"played_ms": -1})

    asyncio.run(scenario())


def test_interrupt_playback_wraps_runtime_hook_errors() -> None:
    async def scenario() -> None:
        runtimes: list[FailingInterruptBlockingRuntime] = []
        service = RuntimeControlService(
            lambda: runtimes.append(FailingInterruptBlockingRuntime())
            or runtimes[-1]
        )

        await service.start()
        await asyncio.wait_for(runtimes[0].started.wait(), timeout=1.0)

        with pytest.raises(RuntimeControlError, match="no active realtime speech session"):
            await service.interrupt_playback({"played_ms": 250, "speaker": "counselor"})

        await service.stop()

    asyncio.run(scenario())


def test_record_playback_completed_deduplicates_playback_keys() -> None:
    async def scenario() -> None:
        runtimes: list[BlockingRuntime] = []
        service = RuntimeControlService(
            lambda: runtimes.append(BlockingRuntime()) or runtimes[-1]
        )

        await service.start()
        await asyncio.wait_for(runtimes[0].started.wait(), timeout=1.0)

        first_status = await service.record_playback_completed(
            {
                "playback_key": "session_control_test:0:client",
                "session_id": "session_control_test",
            }
        )
        duplicate_status = await service.record_playback_completed(
            {
                "playback_key": "session_control_test:0:client",
                "session_id": "session_control_test",
            }
        )
        next_status = await service.record_playback_completed(
            {
                "playback_key": "session_control_test:1:counselor",
                "session_id": "session_control_test",
            }
        )
        stale_status = await service.record_playback_completed(
            {
                "playback_key": "session_stale:2:client",
                "session_id": "session_stale",
            }
        )

        assert first_status.playback_completed_turns == 1
        assert duplicate_status.playback_completed_turns == 1
        assert next_status.playback_completed_turns == 2
        assert stale_status.playback_completed_turns == 2
        await service.stop()

    asyncio.run(scenario())


def test_playback_completed_endpoint_records_completion(monkeypatch) -> None:
    class FakeWebSocket:
        pass

    class FakeWebSocketDisconnect(Exception):
        pass

    class FakeFastAPI:
        def __init__(self) -> None:
            self.routes: dict[tuple[str, str], object] = {}

        def post(self, path):
            return self._record("POST", path)

        def get(self, path):
            return self._record("GET", path)

        def websocket(self, path):
            return self._record("WS", path)

        def _record(self, method, path):
            def decorator(func):
                self.routes[(method, path)] = func
                return func

            return decorator

    fake_fastapi_module = SimpleNamespace(
        Body=lambda default=None: default,
        FastAPI=FakeFastAPI,
        HTTPException=Exception,
        WebSocket=FakeWebSocket,
        WebSocketDisconnect=FakeWebSocketDisconnect,
    )
    monkeypatch.setitem(sys.modules, "fastapi", fake_fastapi_module)

    service = RuntimeControlService(
        lambda: BlockingRuntime(),
        session_id="session_control_test",
    )
    app = control_api.create_app(service)
    handler = app.routes[("POST", "/runtime/playback-completed")]

    async def scenario() -> None:
        response = await handler(
            {
                "event_type": "runtime_playback_ended",
                "playback_key": "session_control_test:0:client",
                "session_id": "session_control_test",
            }
        )
        duplicate_response = await handler(
            {
                "event_type": "runtime_playback_ended",
                "playback_key": "session_control_test:0:client",
                "session_id": "session_control_test",
            }
        )
        assert response["playback_completed_turns"] == 1
        assert duplicate_response["playback_completed_turns"] == 1

    asyncio.run(scenario())


def test_playback_completed_endpoint_rejects_invalid_payload(monkeypatch) -> None:
    class FakeWebSocket:
        pass

    class FakeWebSocketDisconnect(Exception):
        pass

    class FakeHTTPException(Exception):
        def __init__(self, *, status_code, detail):
            super().__init__(detail)
            self.status_code = status_code
            self.detail = detail

    class FakeFastAPI:
        def __init__(self) -> None:
            self.routes: dict[tuple[str, str], object] = {}

        def post(self, path):
            return self._record("POST", path)

        def get(self, path):
            return self._record("GET", path)

        def websocket(self, path):
            return self._record("WS", path)

        def _record(self, method, path):
            def decorator(func):
                self.routes[(method, path)] = func
                return func

            return decorator

    fake_fastapi_module = SimpleNamespace(
        Body=lambda default=None: default,
        FastAPI=FakeFastAPI,
        HTTPException=FakeHTTPException,
        WebSocket=FakeWebSocket,
        WebSocketDisconnect=FakeWebSocketDisconnect,
    )
    monkeypatch.setitem(sys.modules, "fastapi", fake_fastapi_module)

    service = RuntimeControlService(
        lambda: BlockingRuntime(),
        session_id="session_control_test",
    )
    app = control_api.create_app(service)
    handler = app.routes[("POST", "/runtime/playback-completed")]

    async def scenario() -> None:
        with pytest.raises(FakeHTTPException) as exc_info:
            await handler(
                {
                    "event_type": "runtime_playback_ended",
                    "session_id": "session_control_test",
                }
            )
        assert exc_info.value.status_code == 400
        assert "playback_key" in exc_info.value.detail

    asyncio.run(scenario())


def test_read_runtime_events_from_session_dir(tmp_path) -> None:
    session_id = "session_events_test"
    event_path = tmp_path / "internal" / "events" / f"{session_id}.jsonl"
    event_path.parent.mkdir(parents=True)
    records = [
        {"session_id": session_id, "event_type": "runtime_started"},
        {"session_id": session_id, "event_type": "tts_stream_done", "turn_id": 1},
    ]
    event_path.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n",
        encoding="utf-8",
    )

    assert read_runtime_events(session_id, session_dir=tmp_path) == records


def test_read_runtime_events_from_sessions_dir(tmp_path) -> None:
    session_id = "session_events_test"
    event_path = tmp_path / session_id / "internal" / "events" / f"{session_id}.jsonl"
    event_path.parent.mkdir(parents=True)
    event_path.write_text(
        '{"session_id": "session_events_test", "event_type": "runtime_started"}\n',
        encoding="utf-8",
    )

    assert read_runtime_events(session_id, sessions_dir=tmp_path) == [
        {"session_id": "session_events_test", "event_type": "runtime_started"}
    ]


def test_read_runtime_events_raises_for_broken_jsonl(tmp_path) -> None:
    session_id = "session_events_test"
    event_path = tmp_path / "internal" / "events" / f"{session_id}.jsonl"
    event_path.parent.mkdir(parents=True)
    event_path.write_text('{"event_type": "ok"}\n{"event_type": \n', encoding="utf-8")

    with pytest.raises(json.JSONDecodeError):
        read_runtime_events(session_id, session_dir=tmp_path)


def test_read_runtime_events_requires_root_directory() -> None:
    with pytest.raises(RuntimeControlError, match="session_dir or sessions_dir"):
        read_runtime_events("session_events_test")


def test_read_runtime_events_rejects_unsafe_session_id(tmp_path) -> None:
    with pytest.raises(RuntimeControlError, match="session_id"):
        read_runtime_events("../session_events_test", sessions_dir=tmp_path)


def test_read_runtime_response_instructions_from_internal_prompts(tmp_path) -> None:
    session_id = "session_instructions_test"
    prompt_path = (
        tmp_path / "internal" / "prompts" / "response_instructions.jsonl"
    )
    prompt_path.parent.mkdir(parents=True)
    records = [
        {
            "session_id": session_id,
            "turn_id": 1,
            "speaker_id": "counselor",
            "resolved_instructions": "送信済みinstructions",
        }
    ]
    prompt_path.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n",
        encoding="utf-8",
    )

    assert read_runtime_response_instructions(
        session_id,
        session_dir=tmp_path,
    ) == records


def test_read_runtime_response_instructions_returns_empty_for_legacy_session(
    tmp_path,
) -> None:
    assert read_runtime_response_instructions(
        "session_legacy_test",
        session_dir=tmp_path,
    ) == []


def test_stream_audio_monitor_to_websocket_sends_browser_messages() -> None:
    async def scenario() -> None:
        queue: asyncio.Queue[object] = asyncio.Queue()
        await queue.put(
            AudioChunk(
                session_id="session-monitor",
                turn_id=1,
                speaker="counselor",
                chunk_index=0,
                pcm=b"abc",
                duration_ms=10,
            )
        )
        await queue.put(EndOfAudio(session_id="session-monitor", turn_id=1, speaker="counselor"))
        await queue.put(EndOfAudio())
        websocket = RecordingWebSocket()

        await stream_audio_monitor_to_websocket(queue, websocket)

        assert [message["type"] for message in websocket.messages] == [
            "audio_chunk",
            "turn_end",
            "stream_end",
        ]
        assert websocket.messages[0]["pcm_base64"] == "YWJj"

    asyncio.run(scenario())
