from __future__ import annotations

import asyncio
import base64
import inspect
import json
import re
import sys
from collections.abc import Callable, Mapping
from copy import copy
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Protocol, Sequence

from counseling_voice_demo.runtime.audio_monitor import audio_monitor_messages
from counseling_voice_demo.runtime.config import (
    AI_ROUTE_MODEL_FIELDS,
    ParticipantSection,
    RuntimeSettings,
)
from counseling_voice_demo.runtime.provider_registry import (
    ProviderConfigurationError,
    ProviderRegistry,
    effective_ai_settings,
)
from counseling_voice_demo.runtime.speaker_selection import (
    SUPPORTED_SPEAKER_SELECTION_POLICIES,
)
from counseling_voice_demo.runtime.models import (
    HumanAudioInput,
    HumanAudioStreamStart,
    HumanTurnInput,
    MAX_REALTIME_OUTPUT_SPEED,
    MIN_REALTIME_OUTPUT_SPEED,
    RuntimePhase,
    RuntimeStatus,
)


_RUNTIME_EVENT_SESSION_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")


class RuntimeLike(Protocol):
    @property
    def status(self) -> RuntimeStatus | dict[str, Any]:
        ...

    def run(self) -> Any:
        ...


class WebSocketJsonLike(Protocol):
    def send_json(self, data: Any) -> Any:
        ...


RuntimeFactory = Callable[..., RuntimeLike]
MONITOR_AUDIO_RUNTIME_WAIT_SECONDS = 60.0 * 60.0
LOCAL_BROWSER_ORIGIN_REGEX = (
    r"(null|https?://(localhost|127\.0\.0\.1|\[::1\])(:\d+)?)"
)
SUPPORTED_REALTIME_MODELS = {
    "gpt-realtime-mini",
    "gpt-realtime-2",
    "gpt-realtime-2.1",
}
SUPPORTED_TIMING_LLM_MODELS = {
    "gpt-5.6-sol",
    "gpt-5.6-terra",
    "gpt-5.6-luna",
    "gpt-5.4-mini",
}
SUPPORTED_TIMING_LLM_REASONING_EFFORTS = {"none", "low", "medium", "high", "xhigh"}
SUPPORTED_PROMPTING_LLM_MODELS = SUPPORTED_TIMING_LLM_MODELS
SUPPORTED_PROMPTING_LLM_REASONING_EFFORTS = SUPPORTED_TIMING_LLM_REASONING_EFFORTS
SUPPORTED_SUMMARY_LLM_MODELS = SUPPORTED_TIMING_LLM_MODELS
SUPPORTED_SUMMARY_LLM_REASONING_EFFORTS = SUPPORTED_TIMING_LLM_REASONING_EFFORTS
SUPPORTED_SPEAKER_GAIN_KEYS = {"counselor", "client", "client_a", "client_b"}
SUPPORTED_INTERACTION_MODES = {
    "ai_counselor_ai_client",
    "human_counselor_ai_client",
    "ai_counselor_human_client",
}
SUPPORTED_PARTICIPANT_MODES = {"one_client", "two_clients"}
SUPPORTED_HUMAN_INPUT_MODES = {"push_to_talk", "vad_auto"}
SUPPORTED_HUMAN_STT_SUBMIT_POLICIES = {"auto_on_final"}
SUPPORTED_ACTOR_KINDS = {"ai", "human"}
SUPPORTED_PARTICIPANT_OPTION_KEYS = {
    "speaker_id",
    "role",
    "actor_kind",
    "display_name",
    "prompt_path",
    "public_profile_path",
    "private_profile_path",
    "prompt_source",
    "public_profile_source",
    "private_profile_source",
    "initial_transcript",
    "voice",
    "realtime_output_speed",
}
SUPPORTED_PARTICIPANT_ROLES = {"counselor", "client"}


class RuntimeControlError(RuntimeError):
    """Base error for runtime control operations."""


class RuntimeAlreadyStartedError(RuntimeControlError):
    """Raised when start is called while a runtime task is active."""


class RuntimeControlService:
    def __init__(
        self,
        runtime_factory: RuntimeFactory,
        *,
        session_id: str = "runtime_control",
        runtime_factory_accepts_start_options: bool = False,
    ) -> None:
        self._runtime_factory = runtime_factory
        self._runtime_factory_accepts_start_options = runtime_factory_accepts_start_options
        self._session_id = session_id
        self._runtime: RuntimeLike | None = None
        self._task: asyncio.Task[Any] | None = None
        self._phase = RuntimePhase.IDLE
        self._pause_reason: str | None = None
        self._error_message: str | None = None
        self._playback_completed_keys: set[str] = set()
        self._start_options_fingerprint: str | None = None
        self._lock = asyncio.Lock()

    async def start(self, options: Mapping[str, Any] | None = None) -> RuntimeStatus:
        async with self._lock:
            self._sync_completed_task_unlocked()
            start_options = dict(options or {})
            next_fingerprint = _runtime_start_options_fingerprint(start_options)
            if self._task is not None and not self._task.done():
                if (
                    not self._runtime_factory_accepts_start_options
                    or next_fingerprint == self._start_options_fingerprint
                ):
                    return self._current_status_unlocked()
                task_to_cancel = self._task
                task_to_cancel.cancel()
                task_to_cancel.add_done_callback(self._consume_stopped_task_result)

            self._runtime = self._create_runtime_unlocked(start_options)
            self._session_id = self._read_runtime_status_unlocked().session_id
            self._phase = RuntimePhase.RUNNING
            self._pause_reason = None
            self._error_message = None
            self._playback_completed_keys.clear()
            self._start_options_fingerprint = next_fingerprint
            self._task = asyncio.create_task(self._run_runtime(self._runtime))
            return self._current_status_unlocked()

    async def stop(self) -> RuntimeStatus:
        async with self._lock:
            self._sync_completed_task_unlocked()
            if self._task is None:
                self._phase = RuntimePhase.STOPPED
                self._pause_reason = None
                self._error_message = None
                self._start_options_fingerprint = None
                return self._current_status_unlocked()
            if self._task.done():
                return self._current_status_unlocked()

            self._phase = RuntimePhase.STOPPED
            self._pause_reason = None
            self._error_message = None
            self._start_options_fingerprint = None
            task_to_cancel = self._task
            await self._call_runtime_control_hook_unlocked(
                "abort_pending_human_audio_streams"
            )
            task_to_cancel.cancel()
            task_to_cancel.add_done_callback(self._consume_stopped_task_result)
            self._task = None
            return self._current_status_unlocked()

    async def pause(self, reason: str | None = None) -> RuntimeStatus:
        async with self._lock:
            self._sync_completed_task_unlocked()
            if self._phase is RuntimePhase.RUNNING:
                await self._call_runtime_control_hook_unlocked("pause_generation")
                if (
                    self._read_runtime_status_unlocked().interaction_mode
                    == "ai_counselor_human_client"
                ):
                    await self._call_runtime_control_hook_unlocked(
                        "abort_pending_human_audio_streams"
                    )
                self._phase = RuntimePhase.PAUSED
                self._pause_reason = _normalize_pause_reason(reason)
            elif self._phase is RuntimePhase.PAUSED and self._pause_reason not in {
                "prompt_director_validation",
                "prompt_director_transport",
            }:
                self._pause_reason = _normalize_pause_reason(reason)
            return self._current_status_unlocked()

    async def resume(self) -> RuntimeStatus:
        async with self._lock:
            self._sync_completed_task_unlocked()
            if self._phase is RuntimePhase.PAUSED:
                await self._call_runtime_control_hook_unlocked("resume_generation")
                self._phase = RuntimePhase.RUNNING
                self._pause_reason = None
            return self._current_status_unlocked()

    async def interrupt_playback(
        self,
        payload: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        options = _normalize_interrupt_playback_payload(payload or {})
        async with self._lock:
            self._sync_completed_task_unlocked()
            runtime = self._runtime
            if runtime is None or self._task is None or self._task.done():
                raise RuntimeControlError("runtime is not running")
            hook = getattr(runtime, "stop_current_response_playback", None)
            if hook is None:
                raise RuntimeControlError(
                    "runtime does not support playback interruption"
                )
            try:
                result = hook(**options)
                if inspect.isawaitable(result):
                    result = await result
            except RuntimeControlError:
                raise
            except Exception as exc:
                raise RuntimeControlError(str(exc)) from exc
            if hasattr(result, "__dataclass_fields__"):
                return asdict(result)
            if isinstance(result, Mapping):
                return dict(result)
            return {"result": result}

    async def submit_human_audio(
        self,
        payload: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        request = _normalize_human_audio_payload(payload or {})
        async with self._lock:
            self._sync_completed_task_unlocked()
            previous = self._human_input_result_unlocked(request)
            if previous is not None:
                return previous
            self._ensure_human_input_allowed_unlocked()
            runtime = self._runtime
            assert runtime is not None
            hook = getattr(runtime, "submit_human_audio", None)
            if hook is None:
                raise RuntimeControlError(
                    "runtime does not support human audio input"
                )
            result = await _call_runtime_mapping_hook(hook, request)
            if result.get("accepted") is not False:
                await self._resume_after_human_input_unlocked()
            return result

    async def start_human_audio_stream(
        self,
        payload: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        request = _normalize_human_audio_stream_start_payload(payload or {})
        async with self._lock:
            self._sync_completed_task_unlocked()
            previous = self._human_input_result_unlocked(request)
            if previous is not None:
                return previous
            self._ensure_human_input_allowed_unlocked()
            runtime = self._runtime
            assert runtime is not None
            hook = getattr(runtime, "start_human_audio_stream", None)
            if hook is None:
                raise RuntimeControlError(
                    "runtime does not support streaming human audio input"
                )
            result = await _call_runtime_mapping_hook(hook, request)
            if result.get("accepted") is not False:
                await self._resume_after_human_input_unlocked()
            return result

    async def append_human_audio_stream_chunk(
        self,
        stream_id: str,
        payload: Mapping[str, Any] | None = None,
        *,
        expected_session_id: str | None = None,
    ) -> dict[str, Any]:
        audio_bytes, chunk_index = _normalize_human_audio_stream_chunk_payload(
            payload or {}
        )
        async with self._lock:
            if (
                expected_session_id is not None
                and expected_session_id != self._session_id
            ):
                raise RuntimeControlError(
                    "human audio stream belongs to a different session"
                )
            self._sync_completed_task_unlocked()
            self._ensure_runtime_running_unlocked()
            runtime = self._runtime
            assert runtime is not None
            hook = getattr(runtime, "append_human_audio_stream_chunk", None)
            if hook is None:
                raise RuntimeControlError(
                    "runtime does not support streaming human audio input"
                )
            try:
                result = hook(
                    stream_id=stream_id,
                    audio_bytes=audio_bytes,
                    chunk_index=chunk_index,
                )
                if inspect.isawaitable(result):
                    result = await result
            except RuntimeControlError:
                raise
            except Exception as exc:
                raise RuntimeControlError(str(exc)) from exc
            if isinstance(result, Mapping):
                return dict(result)
            return {"result": result}

    async def end_human_audio_stream(
        self, stream_id: str, *, expected_session_id: str | None = None
    ) -> dict[str, Any]:
        async with self._lock:
            if (
                expected_session_id is not None
                and expected_session_id != self._session_id
            ):
                raise RuntimeControlError(
                    "human audio stream belongs to a different session"
                )
            self._sync_completed_task_unlocked()
            self._ensure_runtime_running_unlocked()
            runtime = self._runtime
            assert runtime is not None
            hook = getattr(runtime, "end_human_audio_stream", None)
            if hook is None:
                raise RuntimeControlError(
                    "runtime does not support streaming human audio input"
                )
            try:
                result = hook(stream_id=stream_id)
                if inspect.isawaitable(result):
                    result = await result
            except RuntimeControlError:
                raise
            except Exception as exc:
                raise RuntimeControlError(str(exc)) from exc
            if isinstance(result, Mapping):
                return dict(result)
            return {"result": result}

    async def abort_human_audio_stream(
        self, stream_id: str, *, expected_session_id: str | None = None
    ) -> dict[str, Any]:
        async with self._lock:
            if (
                expected_session_id is not None
                and expected_session_id != self._session_id
            ):
                raise RuntimeControlError(
                    "human audio stream belongs to a different session"
                )
            self._sync_completed_task_unlocked()
            runtime = self._runtime
            hook = getattr(runtime, "abort_human_audio_stream", None)
            if hook is None:
                raise RuntimeControlError("runtime does not support human audio abort")
            try:
                return dict(await hook(stream_id=stream_id))
            except RuntimeError as exc:
                raise RuntimeControlError(str(exc)) from exc

    async def wait_human_audio_stream_completion(
        self, stream_id: str, *, expected_session_id: str | None = None
    ) -> dict[str, Any]:
        async with self._lock:
            if (
                expected_session_id is not None
                and expected_session_id != self._session_id
            ):
                raise RuntimeControlError(
                    "human audio stream belongs to a different session"
                )
            self._sync_completed_task_unlocked()
            self._ensure_runtime_running_unlocked()
            runtime = self._runtime
            assert runtime is not None
            hook = getattr(runtime, "wait_human_audio_stream_completion", None)
            if hook is None:
                raise RuntimeControlError(
                    "runtime does not support streaming human audio completion"
                )
            try:
                result = hook(stream_id=stream_id)
            except RuntimeControlError:
                raise
            except Exception as exc:
                raise RuntimeControlError(str(exc)) from exc
        try:
            if inspect.isawaitable(result):
                result = await result
        except RuntimeControlError:
            raise
        except Exception as exc:
            raise RuntimeControlError(str(exc)) from exc
        if isinstance(result, Mapping):
            return dict(result)
        return {"result": result}

    async def submit_human_turn(
        self,
        payload: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        request = _normalize_human_turn_payload(payload or {})
        async with self._lock:
            self._sync_completed_task_unlocked()
            previous = self._human_input_result_unlocked(request)
            if previous is not None:
                return previous
            self._ensure_human_input_allowed_unlocked()
            runtime = self._runtime
            assert runtime is not None
            hook = getattr(runtime, "submit_human_turn", None)
            if hook is None:
                raise RuntimeControlError(
                    "runtime does not support human turn input"
                )
            result = await _call_runtime_mapping_hook(hook, request)
            if result.get("accepted") is not False:
                await self._resume_after_human_input_unlocked()
            return result

    async def status(self) -> RuntimeStatus:
        async with self._lock:
            self._sync_completed_task_unlocked()
            return self._current_status_unlocked()

    async def record_playback_completed(
        self,
        payload: Mapping[str, Any] | None = None,
    ) -> RuntimeStatus:
        playback_key, event_session_id = _normalize_playback_completed_payload(
            payload or {},
        )
        async with self._lock:
            self._sync_completed_task_unlocked()
            if event_session_id in {"", self._session_id}:
                self._playback_completed_keys.add(playback_key)
            return self._current_status_unlocked()

    def audio_monitor_queue(self) -> asyncio.Queue[object] | None:
        audio_bus = self.audio_bus()
        queue = getattr(audio_bus, "monitor_queue", None)
        return queue

    def audio_bus(self) -> Any | None:
        if self._task is None or self._task.done():
            return None
        runtime = self._runtime
        return getattr(runtime, "audio_bus", None)

    async def _call_runtime_control_hook_unlocked(self, method_name: str) -> None:
        runtime = self._runtime
        if runtime is None:
            return
        hook = getattr(runtime, method_name, None)
        if hook is None:
            return
        result = hook()
        if inspect.isawaitable(result):
            await result

    def _human_input_result_unlocked(self, request: Any) -> dict[str, Any] | None:
        hook = getattr(self._runtime, "human_input_result", None)
        if hook is None:
            return None
        try:
            return hook(request)
        except RuntimeError as exc:
            raise RuntimeControlError(str(exc)) from exc

    def _ensure_human_input_allowed_unlocked(self) -> None:
        self._ensure_runtime_running_unlocked()
        status = self._read_runtime_status_unlocked()
        if (
            self._phase is RuntimePhase.PAUSED
            and status.interaction_mode == "ai_counselor_human_client"
        ):
            raise RuntimeControlError("runtime is paused; Resume before recording")
        if not (
            status.awaiting_human_input
            or status.pending_human_turn_allowed
        ):
            raise RuntimeControlError("runtime is not awaiting human input")

    def _ensure_runtime_running_unlocked(self) -> None:
        if self._phase not in {RuntimePhase.RUNNING, RuntimePhase.PAUSED}:
            raise RuntimeControlError("runtime is not running")
        if self._runtime is None or self._task is None or self._task.done():
            raise RuntimeControlError("runtime is not running")

    async def _resume_after_human_input_unlocked(self) -> None:
        if self._phase is not RuntimePhase.PAUSED or self._pause_reason in {
            "prompt_director_validation",
            "prompt_director_transport",
        }:
            return
        await self._call_runtime_control_hook_unlocked("resume_generation")
        self._phase = RuntimePhase.RUNNING
        self._pause_reason = None

    async def wait_for_audio_monitor_queue(
        self,
        *,
        timeout_seconds: float = 30.0,
        poll_interval_seconds: float = 0.05,
    ) -> asyncio.Queue[object] | None:
        deadline = asyncio.get_running_loop().time() + max(0.0, timeout_seconds)
        while True:
            queue = self.audio_monitor_queue()
            if queue is not None:
                return queue
            if asyncio.get_running_loop().time() >= deadline:
                return None
            await asyncio.sleep(max(0.01, poll_interval_seconds))

    async def wait_for_audio_bus(
        self,
        *,
        timeout_seconds: float = 30.0,
        poll_interval_seconds: float = 0.05,
    ) -> Any | None:
        deadline = asyncio.get_running_loop().time() + max(0.0, timeout_seconds)
        while True:
            audio_bus = self.audio_bus()
            if audio_bus is not None:
                return audio_bus
            if asyncio.get_running_loop().time() >= deadline:
                return None
            await asyncio.sleep(max(0.01, poll_interval_seconds))

    async def subscribe_audio_monitor_queue(
        self,
        *,
        timeout_seconds: float = 30.0,
    ) -> tuple[asyncio.Queue[object], Callable[[], None]]:
        audio_bus = await self.wait_for_audio_bus(timeout_seconds=timeout_seconds)
        if audio_bus is None:
            raise RuntimeControlError("runtime monitor audio queue is not available")
        subscribe_monitor = getattr(audio_bus, "subscribe_monitor", None)
        unsubscribe_monitor = getattr(audio_bus, "unsubscribe_monitor", None)
        if callable(subscribe_monitor) and callable(unsubscribe_monitor):
            queue = subscribe_monitor()

            def cleanup() -> None:
                unsubscribe_monitor(queue)

            return queue, cleanup
        queue = getattr(audio_bus, "monitor_queue", None)
        if queue is None:
            raise RuntimeControlError("runtime monitor audio queue is not available")
        return queue, lambda: None

    async def _run_runtime(self, runtime: RuntimeLike) -> None:
        try:
            result = runtime.run()
            if inspect.isawaitable(result):
                await result
        except asyncio.CancelledError:
            if self._runtime is runtime:
                self._phase = RuntimePhase.STOPPED
                self._pause_reason = None
            raise
        except Exception as exc:
            if self._runtime is runtime:
                self._phase = RuntimePhase.ERROR
                self._pause_reason = None
                self._error_message = str(exc)
            raise
        else:
            if self._runtime is runtime and self._phase is not RuntimePhase.STOPPED:
                self._phase = RuntimePhase.COMPLETED
                self._pause_reason = None
                self._error_message = None

    def _consume_stopped_task_result(self, task: asyncio.Task[Any]) -> None:
        try:
            task.result()
        except asyncio.CancelledError:
            return
        except Exception:
            return

    def _sync_completed_task_unlocked(self) -> None:
        if self._task is None or not self._task.done():
            if self._task is not None and self._phase in {
                RuntimePhase.RUNNING,
                RuntimePhase.PAUSED,
            }:
                status = self._read_runtime_status_unlocked()
                if status.phase is RuntimePhase.PAUSED and status.pause_reason in {
                    "prompt_director_validation",
                    "prompt_director_transport",
                }:
                    self._phase = RuntimePhase.PAUSED
                    self._pause_reason = status.pause_reason
            return
        if self._task.cancelled():
            self._phase = RuntimePhase.STOPPED
            self._pause_reason = None
            self._error_message = None
            return
        exc = self._task.exception()
        if exc is not None:
            self._phase = RuntimePhase.ERROR
            self._pause_reason = None
            self._error_message = str(exc)
            return
        if self._phase is not RuntimePhase.STOPPED:
            self._phase = RuntimePhase.COMPLETED
            self._pause_reason = None
            self._error_message = None

    def _current_status_unlocked(self) -> RuntimeStatus:
        status = replace(
            self._read_runtime_status_unlocked(),
            phase=self._phase,
            playback_completed_turns=len(self._playback_completed_keys),
            error_message=(
                self._error_message
                if self._phase is RuntimePhase.ERROR
                else None
            ),
        )
        if self._phase is RuntimePhase.PAUSED:
            return replace(status, pause_reason=self._pause_reason)
        return replace(status, pause_reason=None)

    def _create_runtime_unlocked(self, options: Mapping[str, Any] | None) -> RuntimeLike:
        normalized_options = dict(options or {})
        if self._runtime_factory_accepts_start_options:
            return self._runtime_factory(normalized_options)
        if normalized_options:
            raise RuntimeControlError("runtime start options are not supported by this service")
        return self._runtime_factory()

    def _read_runtime_status_unlocked(self) -> RuntimeStatus:
        if self._runtime is None:
            return RuntimeStatus(session_id=self._session_id, phase=self._phase)

        raw_status = self._runtime.status
        if isinstance(raw_status, RuntimeStatus):
            return raw_status
        if isinstance(raw_status, dict):
            return self._status_from_dict(raw_status)
        return RuntimeStatus(session_id=self._session_id, phase=self._phase)

    def _status_from_dict(self, raw_status: dict[str, Any]) -> RuntimeStatus:
        return RuntimeStatus(
            session_id=str(raw_status.get("session_id", self._session_id)),
            phase=self._coerce_phase(raw_status.get("phase", self._phase)),
            pause_reason=raw_status.get("pause_reason"),
            current_turn_id=raw_status.get("current_turn_id"),
            current_speaker=raw_status.get("current_speaker"),
            completed_turns=int(raw_status.get("completed_turns") or 0),
            playback_completed_turns=int(
                raw_status.get("playback_completed_turns") or 0
            ),
            last_event_type=raw_status.get("last_event_type"),
            closing_started=bool(
                raw_status.get("closing_started")
                or raw_status.get("closing_started_turn_id") is not None
                or raw_status.get("closing_count_started_turn_id") is not None
            ),
            closing_started_turn_id=(
                int(raw_status["closing_started_turn_id"])
                if raw_status.get("closing_started_turn_id") is not None
                else None
            ),
            closing_count_started_turn_id=(
                int(raw_status["closing_count_started_turn_id"])
                if raw_status.get("closing_count_started_turn_id") is not None
                else None
            ),
            awaiting_human_input=bool(raw_status.get("awaiting_human_input")),
            active_speaker_id=raw_status.get("active_speaker_id"),
            pending_human_turn_allowed=bool(
                raw_status.get("pending_human_turn_allowed")
            ),
            human_input_state=raw_status.get("human_input_state"),
            error_message=raw_status.get("error_message"),
            interaction_mode=raw_status.get("interaction_mode"),
            participants=dict(raw_status.get("participants") or {}),
            human_speaker_id=raw_status.get("human_speaker_id"),
            human_recipient_ids=tuple(raw_status.get("human_recipient_ids") or ()),
        )

    def _coerce_phase(self, value: Any) -> RuntimePhase:
        if isinstance(value, RuntimePhase):
            return value
        try:
            return RuntimePhase(str(value))
        except ValueError as exc:
            raise RuntimeControlError(f"unknown runtime phase: {value!r}") from exc


def _normalize_pause_reason(reason: str | None) -> str:
    value = str(reason or "manual").strip() or "manual"
    if value not in {"manual", "playback", "generation_throttle"}:
        return "manual"
    return value


def _normalize_interrupt_playback_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise RuntimeControlError("interrupt payload must be a JSON object")
    allowed_keys = {
        "played_ms",
        "speaker",
        "turn_id",
        "item_id",
        "response_id",
        "content_index",
        "cancel_response",
        "truncate_item",
        "reason",
    }
    unknown_keys = set(payload) - allowed_keys
    if unknown_keys:
        joined = ", ".join(sorted(str(key) for key in unknown_keys))
        raise RuntimeControlError(f"unsupported interrupt option(s): {joined}")
    if "played_ms" not in payload:
        raise RuntimeControlError("played_ms is required")
    played_ms = payload["played_ms"]
    if not isinstance(played_ms, int) or isinstance(played_ms, bool) or played_ms < 0:
        raise RuntimeControlError("played_ms must be a non-negative integer")

    options: dict[str, Any] = {
        "played_ms": played_ms,
        "speaker": _optional_non_empty_string(payload, "speaker"),
        "turn_id": _optional_non_negative_int(payload, "turn_id"),
        "item_id": _optional_non_empty_string(payload, "item_id"),
        "response_id": _optional_non_empty_string(payload, "response_id"),
        "content_index": _optional_non_negative_int(payload, "content_index"),
        "cancel_response": _optional_boolean(payload, "cancel_response", default=True),
        "truncate_item": _optional_boolean(payload, "truncate_item", default=True),
        "reason": _optional_non_empty_string(payload, "reason") or "browser_overlap",
    }
    return options


def _normalize_playback_completed_payload(payload: Mapping[str, Any]) -> tuple[str, str]:
    if not isinstance(payload, Mapping):
        raise RuntimeControlError("playback completion payload must be a JSON object")
    raw_playback_key = payload.get("playback_key")
    if not isinstance(raw_playback_key, str) or not raw_playback_key.strip():
        raise RuntimeControlError("playback_key is required")
    raw_session_id = payload.get("session_id")
    if raw_session_id is None:
        return raw_playback_key.strip(), ""
    if not isinstance(raw_session_id, str):
        raise RuntimeControlError("session_id must be a string")
    return raw_playback_key.strip(), raw_session_id.strip()


def _optional_non_empty_string(
    payload: Mapping[str, Any],
    key: str,
) -> str | None:
    if key not in payload or payload[key] is None:
        return None
    value = payload[key]
    if not isinstance(value, str) or not value.strip():
        raise RuntimeControlError(f"{key} must be a non-empty string")
    return value.strip()


def _optional_non_negative_int(
    payload: Mapping[str, Any],
    key: str,
) -> int | None:
    if key not in payload or payload[key] is None:
        return None
    value = payload[key]
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise RuntimeControlError(f"{key} must be a non-negative integer")
    return value


def _optional_boolean(
    payload: Mapping[str, Any],
    key: str,
    *,
    default: bool,
) -> bool:
    if key not in payload:
        return default
    value = payload[key]
    if not isinstance(value, bool):
        raise RuntimeControlError(f"{key} must be a boolean")
    return value


async def _call_runtime_mapping_hook(hook: Callable[[Any], Any], request: Any) -> dict[str, Any]:
    try:
        result = hook(request)
        if inspect.isawaitable(result):
            result = await result
    except RuntimeControlError:
        raise
    except Exception as exc:
        raise RuntimeControlError(str(exc)) from exc
    if hasattr(result, "__dataclass_fields__"):
        return asdict(result)
    if isinstance(result, Mapping):
        return dict(result)
    return {"result": result}


def _normalize_human_audio_payload(payload: Mapping[str, Any]) -> HumanAudioInput:
    if not isinstance(payload, Mapping):
        raise RuntimeControlError("human audio payload must be a JSON object")
    allowed_keys = {
        "session_id",
        "audio_bytes",
        "audio_base64",
        "sample_rate",
        "channels",
        "recording_mode",
        "recipient_ids",
        "client_message_id",
    }
    unknown_keys = set(payload) - allowed_keys
    if unknown_keys:
        joined = ", ".join(sorted(str(key) for key in unknown_keys))
        raise RuntimeControlError(f"unsupported human audio option(s): {joined}")
    audio_bytes = payload.get("audio_bytes")
    audio_base64 = payload.get("audio_base64")
    if audio_bytes is not None and audio_base64 is not None:
        raise RuntimeControlError("audio_bytes and audio_base64 cannot both be provided")
    if audio_base64 is not None:
        if not isinstance(audio_base64, str) or not audio_base64.strip():
            raise RuntimeControlError("audio_base64 must be a non-empty string")
        try:
            audio_bytes = base64.b64decode(audio_base64, validate=True)
        except ValueError as exc:
            raise RuntimeControlError("audio_base64 must be valid base64") from exc
    if isinstance(audio_bytes, str):
        audio_bytes = audio_bytes.encode("latin1")
    try:
        return HumanAudioInput(
            session_id=_required_non_empty_string(payload, "session_id"),
            audio_bytes=audio_bytes,  # type: ignore[arg-type]
            sample_rate=_required_positive_int(payload, "sample_rate"),
            channels=_required_positive_int(payload, "channels"),
            recording_mode=_optional_non_empty_string(payload, "recording_mode")
            or "push_to_talk",
            recipient_ids=_normalize_recipient_ids(payload.get("recipient_ids")),
            client_message_id=_optional_non_empty_string(
                payload,
                "client_message_id",
            ),
        )
    except ValueError as exc:
        raise RuntimeControlError(str(exc)) from exc


def _normalize_human_audio_stream_start_payload(
    payload: Mapping[str, Any],
) -> HumanAudioStreamStart:
    if not isinstance(payload, Mapping):
        raise RuntimeControlError(
            "human audio stream start payload must be a JSON object"
        )
    allowed_keys = {
        "type",
        "session_id",
        "sample_rate",
        "channels",
        "recording_mode",
        "recipient_ids",
        "client_message_id",
    }
    unknown_keys = set(payload) - allowed_keys
    if unknown_keys:
        joined = ", ".join(sorted(str(key) for key in unknown_keys))
        raise RuntimeControlError(
            f"unsupported human audio stream option(s): {joined}"
        )
    message_type = payload.get("type")
    if message_type is not None and message_type != "start":
        raise RuntimeControlError("human audio stream start type must be start")
    try:
        return HumanAudioStreamStart(
            session_id=_required_non_empty_string(payload, "session_id"),
            sample_rate=_required_positive_int(payload, "sample_rate"),
            channels=_required_positive_int(payload, "channels"),
            recording_mode=_optional_non_empty_string(payload, "recording_mode")
            or "push_to_talk",
            recipient_ids=_normalize_recipient_ids(payload.get("recipient_ids")),
            client_message_id=_optional_non_empty_string(
                payload,
                "client_message_id",
            ),
        )
    except ValueError as exc:
        raise RuntimeControlError(str(exc)) from exc


def _normalize_human_audio_stream_chunk_payload(
    payload: Mapping[str, Any],
) -> tuple[bytes, int]:
    if not isinstance(payload, Mapping):
        raise RuntimeControlError(
            "human audio stream chunk payload must be a JSON object"
        )
    allowed_keys = {"type", "audio_bytes", "audio_base64", "chunk_index"}
    unknown_keys = set(payload) - allowed_keys
    if unknown_keys:
        joined = ", ".join(sorted(str(key) for key in unknown_keys))
        raise RuntimeControlError(
            f"unsupported human audio stream chunk option(s): {joined}"
        )
    message_type = payload.get("type")
    if message_type is not None and message_type != "chunk":
        raise RuntimeControlError("human audio stream chunk type must be chunk")
    audio_bytes = payload.get("audio_bytes")
    audio_base64 = payload.get("audio_base64")
    if audio_bytes is not None and audio_base64 is not None:
        raise RuntimeControlError("audio_bytes and audio_base64 cannot both be provided")
    if audio_base64 is not None:
        if not isinstance(audio_base64, str) or not audio_base64.strip():
            raise RuntimeControlError("audio_base64 must be a non-empty string")
        try:
            audio_bytes = base64.b64decode(audio_base64, validate=True)
        except ValueError as exc:
            raise RuntimeControlError("audio_base64 must be valid base64") from exc
    if isinstance(audio_bytes, str):
        audio_bytes = audio_bytes.encode("latin1")
    if not isinstance(audio_bytes, bytes) or not audio_bytes:
        raise RuntimeControlError("audio_bytes must not be empty")
    chunk_index = _optional_non_negative_int(payload, "chunk_index")
    if chunk_index is None:
        raise RuntimeControlError("chunk_index is required")
    return audio_bytes, chunk_index


def _normalize_human_turn_payload(payload: Mapping[str, Any]) -> HumanTurnInput:
    if not isinstance(payload, Mapping):
        raise RuntimeControlError("human turn payload must be a JSON object")
    allowed_keys = {
        "session_id",
        "text",
        "recipient_ids",
        "input_mode",
        "recording_mode",
        "source_audio_ref",
        "client_message_id",
    }
    unknown_keys = set(payload) - allowed_keys
    if unknown_keys:
        joined = ", ".join(sorted(str(key) for key in unknown_keys))
        raise RuntimeControlError(f"unsupported human turn option(s): {joined}")
    try:
        return HumanTurnInput(
            session_id=_required_non_empty_string(payload, "session_id"),
            text=_required_non_empty_string(payload, "text"),
            recipient_ids=_normalize_recipient_ids(payload.get("recipient_ids")),
            input_mode=_optional_non_empty_string(payload, "input_mode")
            or "human_mic_stt",
            recording_mode=_optional_non_empty_string(payload, "recording_mode")
            or "push_to_talk",
            source_audio_ref=_optional_non_empty_string(payload, "source_audio_ref"),
            client_message_id=_optional_non_empty_string(
                payload,
                "client_message_id",
            ),
        )
    except ValueError as exc:
        raise RuntimeControlError(str(exc)) from exc


def _required_non_empty_string(payload: Mapping[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise RuntimeControlError(f"{key} must be a non-empty string")
    return value.strip()


def _required_positive_int(payload: Mapping[str, Any], key: str) -> int:
    value = payload.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise RuntimeControlError(f"{key} must be a positive integer")
    return value


def _normalize_recipient_ids(value: Any) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise RuntimeControlError("recipient_ids must be a non-empty array")
    recipient_ids: list[str] = []
    for index, recipient_id in enumerate(value):
        if not isinstance(recipient_id, str) or not recipient_id.strip():
            raise RuntimeControlError(
                f"recipient_ids[{index}] must be a non-empty string"
            )
        recipient_ids.append(recipient_id.strip())
    if not recipient_ids:
        raise RuntimeControlError("recipient_ids must not be empty")
    return tuple(recipient_ids)


def read_runtime_events(
    session_id: str,
    *,
    session_dir: Path | str | None = None,
    sessions_dir: Path | str | None = None,
) -> list[dict[str, Any]]:
    _validate_runtime_event_session_id(session_id)
    event_path = _runtime_events_path(
        session_id,
        session_dir=session_dir,
        sessions_dir=sessions_dir,
    )
    events: list[dict[str, Any]] = []
    for line in event_path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if not isinstance(record, dict):
            raise RuntimeControlError(f"runtime event line is not a JSON object: {event_path}")
        events.append(record)
    return events


def read_runtime_response_instructions(
    session_id: str,
    *,
    session_dir: Path | str | None = None,
    sessions_dir: Path | str | None = None,
) -> list[dict[str, Any]]:
    _validate_runtime_event_session_id(session_id)
    prompt_path = _runtime_response_instructions_path(
        session_id,
        session_dir=session_dir,
        sessions_dir=sessions_dir,
    )
    if not prompt_path.exists():
        return []
    records: list[dict[str, Any]] = []
    for line in prompt_path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if not isinstance(record, dict):
            raise RuntimeControlError(
                "runtime response instructions line is not a JSON object: "
                f"{prompt_path}"
            )
        records.append(record)
    return records


def _validate_runtime_event_session_id(session_id: str) -> None:
    if not isinstance(session_id, str) or not _RUNTIME_EVENT_SESSION_ID_PATTERN.fullmatch(
        session_id
    ):
        raise RuntimeControlError(
            "session_id must contain only ASCII letters, digits, underscores, or hyphens"
        )


async def stream_audio_monitor_to_websocket(
    queue: asyncio.Queue[object],
    websocket: WebSocketJsonLike,
) -> None:
    async for message in audio_monitor_messages(queue):
        result = websocket.send_json(message)
        if inspect.isawaitable(result):
            await result


def create_app(
    service: RuntimeControlService,
    *,
    session_dir: Path | str | None = None,
    sessions_dir: Path | str | None = None,
    audio_queue: asyncio.Queue[object] | None = None,
) -> Any:
    try:
        from fastapi import Body, FastAPI, HTTPException, WebSocket, WebSocketDisconnect
    except ModuleNotFoundError as exc:
        raise RuntimeControlError(
            "FastAPI is not installed; cannot create runtime control app"
        ) from exc
    try:
        from fastapi.middleware.cors import CORSMiddleware
    except ModuleNotFoundError:
        CORSMiddleware = None
    globals()["WebSocket"] = WebSocket

    app = FastAPI()
    if CORSMiddleware is not None and hasattr(app, "add_middleware"):
        app.add_middleware(
            CORSMiddleware,
            allow_origin_regex=LOCAL_BROWSER_ORIGIN_REGEX,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    @app.post("/runtime/start")
    async def start_runtime(payload: dict[str, Any] | None = Body(default=None)) -> dict[str, Any]:
        try:
            options = normalize_runtime_start_options(payload or {})
            return _status_to_dict(await service.start(options))
        except RuntimeControlError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/runtime/stop")
    async def stop_runtime() -> dict[str, Any]:
        return _status_to_dict(await service.stop())

    @app.post("/runtime/interrupt")
    async def interrupt_runtime(
        payload: dict[str, Any] | None = Body(default=None),
    ) -> dict[str, Any]:
        try:
            return await service.interrupt_playback(payload or {})
        except RuntimeControlError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/runtime/human-audio")
    async def human_audio_runtime(
        payload: dict[str, Any] | None = Body(default=None),
    ) -> dict[str, Any]:
        try:
            return await service.submit_human_audio(payload or {})
        except RuntimeControlError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/runtime/human-turn")
    async def human_turn_runtime(
        payload: dict[str, Any] | None = Body(default=None),
    ) -> dict[str, Any]:
        try:
            return await service.submit_human_turn(payload or {})
        except RuntimeControlError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/runtime/playback-completed")
    async def playback_completed_runtime(
        payload: dict[str, Any] | None = Body(default=None),
    ) -> dict[str, Any]:
        try:
            return _status_to_dict(await service.record_playback_completed(payload or {}))
        except RuntimeControlError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/runtime/pause")
    async def pause_runtime(
        payload: dict[str, Any] | None = Body(default=None),
    ) -> dict[str, Any]:
        reason = str((payload or {}).get("reason") or "manual")
        return _status_to_dict(await service.pause(reason=reason))

    @app.post("/runtime/resume")
    async def resume_runtime() -> dict[str, Any]:
        return _status_to_dict(await service.resume())

    @app.get("/runtime/status")
    async def runtime_status() -> dict[str, Any]:
        return _status_to_dict(await service.status())

    @app.get("/runtime/sessions/{session_id}/events")
    async def runtime_events(session_id: str) -> list[dict[str, Any]]:
        return read_runtime_events(
            session_id,
            session_dir=session_dir,
            sessions_dir=sessions_dir or Path("results") / "runtime_sessions",
        )

    @app.get("/runtime/sessions/{session_id}/response-instructions")
    async def runtime_response_instructions(
        session_id: str,
    ) -> list[dict[str, Any]]:
        return read_runtime_response_instructions(
            session_id,
            session_dir=session_dir,
            sessions_dir=sessions_dir or Path("results") / "runtime_sessions",
        )

    async def _human_audio_stream(websocket: WebSocket) -> None:
        await websocket.accept()
        stream_id = ""
        stream_ended = False
        stream_session_id = ""
        human_client_mode = False
        completion_task: asyncio.Task[dict[str, Any]] | None = None
        receive_task: asyncio.Task[Any] | None = None
        try:
            start_payload = await websocket.receive_json()
            stream_session_id = str((start_payload or {}).get("session_id") or "")
            start_result = await service.start_human_audio_stream(start_payload or {})
            human_client_mode = (
                await service.status()
            ).interaction_mode == "ai_counselor_human_client"
            stream_id = str(start_result.get("stream_id") or "")
            if not stream_id:
                raise RuntimeControlError("human audio stream id is missing")
            await websocket.send_json({"type": "accepted", **start_result})
            completion_task = asyncio.create_task(
                service.wait_human_audio_stream_completion(
                    stream_id, expected_session_id=stream_session_id
                )
            )
            receive_task = asyncio.create_task(websocket.receive_json())
            while True:
                done, _pending = await asyncio.wait(
                    {completion_task, receive_task},
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if completion_task in done:
                    completion_result = await completion_task
                    stream_ended = True
                    await websocket.send_json({"type": "completed", **completion_result})
                    await websocket.close()
                    return
                payload = await receive_task
                receive_task = asyncio.create_task(websocket.receive_json())
                if not isinstance(payload, Mapping):
                    raise RuntimeControlError(
                        "human audio stream message must be a JSON object"
                    )
                message_type = payload.get("type")
                if message_type == "chunk":
                    if completion_task.done():
                        completion_result = await completion_task
                        stream_ended = True
                        await websocket.send_json({"type": "completed", **completion_result})
                        await websocket.close()
                        return
                    try:
                        chunk_result = await service.append_human_audio_stream_chunk(
                            stream_id,
                            payload,
                            expected_session_id=stream_session_id,
                        )
                    except RuntimeControlError:
                        if completion_task.done():
                            completion_result = await completion_task
                            stream_ended = True
                            await websocket.send_json(
                                {"type": "completed", **completion_result}
                            )
                            await websocket.close()
                            return
                        raise
                    await websocket.send_json(
                        {
                            "type": "chunk_received",
                            "stream_id": stream_id,
                            "chunk_index": chunk_result.get("chunk_index"),
                        }
                    )
                    continue
                if message_type == "abort":
                    result = await service.abort_human_audio_stream(
                        stream_id, expected_session_id=stream_session_id
                    )
                    stream_ended = True
                    await websocket.send_json({"type": "completed", **result})
                    await websocket.close()
                    return
                if message_type == "end":
                    end_result = await service.end_human_audio_stream(
                        stream_id, expected_session_id=stream_session_id
                    )
                    if end_result.get("pending_transcription"):
                        await websocket.send_json(
                            {"type": "end_received", **end_result}
                        )
                        continue
                    stream_ended = True
                    await websocket.send_json({"type": "completed", **end_result})
                    await websocket.close()
                    return
                raise RuntimeControlError(
                    f"unsupported human audio stream message type: {message_type!r}"
                )
        except WebSocketDisconnect:
            return
        except RuntimeControlError as exc:
            await websocket.send_json({"type": "error", "message": str(exc)})
            await websocket.close(code=1008)
        finally:
            for task in (completion_task, receive_task):
                if task is not None and not task.done():
                    task.cancel()
            if stream_id and not stream_ended:
                try:
                    if human_client_mode:
                        await service.abort_human_audio_stream(
                            stream_id, expected_session_id=stream_session_id
                        )
                    else:
                        await service.end_human_audio_stream(
                            stream_id, expected_session_id=stream_session_id
                        )
                except RuntimeControlError:
                    pass

    async def _monitor_audio(websocket: WebSocket) -> None:
        await websocket.accept()
        cleanup: Callable[[], None] = lambda: None
        if audio_queue is None:
            try:
                active_queue, cleanup = await service.subscribe_audio_monitor_queue(
                    timeout_seconds=MONITOR_AUDIO_RUNTIME_WAIT_SECONDS,
                )
            except RuntimeControlError:
                await websocket.send_json(
                    {
                        "type": "error",
                        "message": "runtime monitor audio queue is not available",
                    }
                )
                await websocket.close()
                return
        else:
            active_queue = audio_queue
        try:
            await stream_audio_monitor_to_websocket(active_queue, websocket)
            await websocket.close()
        except WebSocketDisconnect:
            return
        finally:
            cleanup()

    @app.websocket("/runtime/monitor-audio")
    async def monitor_audio(websocket: WebSocket) -> None:
        await _monitor_audio(websocket)

    @app.websocket("/runtime/human-audio-stream")
    async def human_audio_stream(websocket: WebSocket) -> None:
        await _human_audio_stream(websocket)

    @app.websocket("/runtime/sessions/{session_id}/monitor-audio")
    async def monitor_session_audio(websocket: WebSocket, session_id: str) -> None:
        _ = session_id
        await _monitor_audio(websocket)

    return app


def build_default_control_service(config: Any = None) -> RuntimeControlService:
    from counseling_voice_demo.runtime import factory

    settings = _load_default_control_config(config)

    def runtime_factory(start_options: Mapping[str, Any] | None = None) -> RuntimeLike:
        normalized_options = normalize_runtime_start_options(start_options or {})
        active_settings = apply_runtime_start_options(settings, normalized_options)
        backend = getattr(active_settings.runtime, "backend", "fake")
        if backend == "fake":
            return factory.build_fake_runtime(active_settings)
        if backend == "openai":
            try:
                return factory.build_openai_runtime(active_settings)
            except factory.RuntimeFactoryError as exc:
                raise RuntimeControlError(str(exc)) from exc
        raise RuntimeControlError(f"unsupported runtime backend: {backend}")

    return RuntimeControlService(
        runtime_factory,
        runtime_factory_accepts_start_options=True,
    )


def build_default_control_app(config: Any = None) -> tuple[Any, str, int]:
    settings = _load_default_control_config(config)
    service = build_default_control_service(settings)
    app = create_app(service, sessions_dir=settings.paths.runtime_sessions_dir)
    return app, settings.runtime.control_host, settings.runtime.control_port


def run_control_api(app: Any, *, host: str, port: int) -> None:
    try:
        import uvicorn
    except ModuleNotFoundError as exc:
        raise RuntimeControlError(
            "uvicorn is not installed; cannot run runtime control API"
        ) from exc

    uvicorn.run(app, host=host, port=port)


async def read_runtime_start_options(request: Any) -> dict[str, Any]:
    try:
        body = await request.body()
    except AttributeError:
        body = b""
    if not body:
        return {}
    try:
        payload = await request.json()
    except json.JSONDecodeError as exc:
        raise RuntimeControlError("runtime start payload must be valid JSON") from exc
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise RuntimeControlError("runtime start payload must be a JSON object")
    return normalize_runtime_start_options(payload)


def _runtime_start_options_fingerprint(options: Mapping[str, Any]) -> str:
    return json.dumps(
        dict(options),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def normalize_runtime_start_options(payload: Mapping[str, Any]) -> dict[str, Any]:
    allowed_keys = {
        "ai_routes",
        "realtime_api_centered_mode",
        "interaction_mode",
        "participant_mode",
        "human_input_mode",
        "human_stt_submit_policy",
        "human_interrupts_enabled",
        "initial_client_transcript",
        "stop_condition",
        "max_turns",
        "max_elapsed_seconds",
        "closing_start_elapsed_seconds",
        "force_stop_after_closing_turns",
        "speaker_selection_policy",
        "fixed_speaker_sequence",
        "counselor_prompt",
        "client_prompt",
        "counselor_display_name",
        "client_display_name",
        "realtime_model",
        "prompting_llm_model",
        "prompting_llm_reasoning_effort",
        "timing_llm_model",
        "timing_llm_reasoning_effort",
        "summary_llm_model",
        "summary_llm_reasoning_effort",
        "shared_case",
        "counselor_tts_voice",
        "client_tts_voice",
        "realtime_output_speed",
        "counselor_realtime_output_speed",
        "client_realtime_output_speed",
        "speaker_gains",
        "participants",
    }
    unknown_keys = set(payload) - allowed_keys
    if unknown_keys:
        joined = ", ".join(sorted(unknown_keys))
        raise RuntimeControlError(f"unsupported runtime start option(s): {joined}")
    options: dict[str, Any] = {}
    if "ai_routes" in payload:
        routes = payload["ai_routes"]
        if not isinstance(routes, Mapping) or set(routes) - set(AI_ROUTE_MODEL_FIELDS):
            raise RuntimeControlError("ai_routes contains an unsupported route")
        normalized_routes = {}
        for name, route in routes.items():
            if not isinstance(route, Mapping) or set(route) - {"provider", "model_ref"}:
                raise RuntimeControlError(
                    "AI route options accept provider and model_ref only"
                )
            if "provider" not in route:
                raise RuntimeControlError("AI route requires provider")
            normalized_route = {}
            for field, value in route.items():
                if (
                    not isinstance(value, str)
                    or not value.strip()
                    or any(ord(char) < 32 for char in value)
                ):
                    raise RuntimeControlError(
                        f"AI route {field} must be a nonempty string without control characters"
                    )
                normalized_route[field] = value.strip()
            normalized_routes[name] = normalized_route
        options["ai_routes"] = normalized_routes
    if "realtime_api_centered_mode" in payload:
        value = payload["realtime_api_centered_mode"]
        if not isinstance(value, bool):
            raise RuntimeControlError("realtime_api_centered_mode must be a boolean")
        options["realtime_api_centered_mode"] = value
    if "interaction_mode" in payload:
        value = payload["interaction_mode"]
        if not isinstance(value, str) or not value.strip():
            raise RuntimeControlError("interaction_mode must be a string")
        normalized_value = value.strip()
        if normalized_value not in SUPPORTED_INTERACTION_MODES:
            supported = ", ".join(sorted(SUPPORTED_INTERACTION_MODES))
            raise RuntimeControlError(
                f"interaction_mode must be one of: {supported}"
            )
        options["interaction_mode"] = normalized_value
    if "participant_mode" in payload:
        value = payload["participant_mode"]
        if not isinstance(value, str) or not value.strip():
            raise RuntimeControlError("participant_mode must be a string")
        normalized_value = value.strip()
        if normalized_value not in SUPPORTED_PARTICIPANT_MODES:
            supported = ", ".join(sorted(SUPPORTED_PARTICIPANT_MODES))
            raise RuntimeControlError(
                f"participant_mode must be one of: {supported}"
            )
        options["participant_mode"] = normalized_value
    if "human_input_mode" in payload:
        value = payload["human_input_mode"]
        if not isinstance(value, str) or not value.strip():
            raise RuntimeControlError("human_input_mode must be a string")
        normalized_value = value.strip()
        if normalized_value not in SUPPORTED_HUMAN_INPUT_MODES:
            supported = ", ".join(sorted(SUPPORTED_HUMAN_INPUT_MODES))
            raise RuntimeControlError(
                f"human_input_mode must be one of: {supported}"
            )
        options["human_input_mode"] = normalized_value
    if "human_stt_submit_policy" in payload:
        value = payload["human_stt_submit_policy"]
        if not isinstance(value, str) or not value.strip():
            raise RuntimeControlError("human_stt_submit_policy must be a string")
        normalized_value = value.strip()
        if normalized_value not in SUPPORTED_HUMAN_STT_SUBMIT_POLICIES:
            supported = ", ".join(sorted(SUPPORTED_HUMAN_STT_SUBMIT_POLICIES))
            raise RuntimeControlError(
                f"human_stt_submit_policy must be one of: {supported}"
            )
        options["human_stt_submit_policy"] = normalized_value
    if "human_interrupts_enabled" in payload:
        value = payload["human_interrupts_enabled"]
        if not isinstance(value, bool):
            raise RuntimeControlError("human_interrupts_enabled must be a boolean")
        options["human_interrupts_enabled"] = value
    if "max_turns" in payload:
        value = payload["max_turns"]
        if not isinstance(value, int) or value <= 0:
            raise RuntimeControlError("max_turns must be a positive integer")
        options["max_turns"] = value
    if "stop_condition" in payload:
        value = payload["stop_condition"]
        if value not in {"turns", "elapsed_time"}:
            raise RuntimeControlError(
                "stop_condition must be 'turns' or 'elapsed_time'"
            )
        options["stop_condition"] = value
    if "max_elapsed_seconds" in payload:
        value = payload["max_elapsed_seconds"]
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or value <= 0
        ):
            raise RuntimeControlError("max_elapsed_seconds must be a positive number")
        options["max_elapsed_seconds"] = float(value)
    if "closing_start_elapsed_seconds" in payload:
        value = payload["closing_start_elapsed_seconds"]
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or value < 0
        ):
            raise RuntimeControlError(
                "closing_start_elapsed_seconds must be a non-negative number"
            )
        options["closing_start_elapsed_seconds"] = float(value)
    if "force_stop_after_closing_turns" in payload:
        value = payload["force_stop_after_closing_turns"]
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise RuntimeControlError(
                "force_stop_after_closing_turns must be a positive integer"
            )
        options["force_stop_after_closing_turns"] = value
    if "speaker_selection_policy" in payload:
        value = payload["speaker_selection_policy"]
        if not isinstance(value, str) or not value.strip():
            raise RuntimeControlError("speaker_selection_policy must be a string")
        normalized_policy = value.strip()
        if normalized_policy not in SUPPORTED_SPEAKER_SELECTION_POLICIES:
            supported = ", ".join(sorted(SUPPORTED_SPEAKER_SELECTION_POLICIES))
            raise RuntimeControlError(
                f"speaker_selection_policy must be one of: {supported}"
            )
        options["speaker_selection_policy"] = normalized_policy
    if "fixed_speaker_sequence" in payload:
        options["fixed_speaker_sequence"] = _normalize_fixed_speaker_sequence(
            payload["fixed_speaker_sequence"]
        )
    if (
        options.get("stop_condition") == "elapsed_time"
        and "max_elapsed_seconds" not in options
    ):
        raise RuntimeControlError(
            "max_elapsed_seconds is required when stop_condition is elapsed_time"
        )
    if (
        options.get("stop_condition") == "turns"
        and "max_elapsed_seconds" in options
    ):
        raise RuntimeControlError(
            "max_elapsed_seconds is only supported when stop_condition is elapsed_time"
        )
    for key in [
        "initial_client_transcript",
        "counselor_prompt",
        "client_prompt",
        "counselor_display_name",
        "client_display_name",
        "realtime_model",
        "prompting_llm_model",
        "prompting_llm_reasoning_effort",
        "timing_llm_model",
        "timing_llm_reasoning_effort",
        "summary_llm_model",
        "summary_llm_reasoning_effort",
        "shared_case",
        "counselor_tts_voice",
        "client_tts_voice",
    ]:
        if key not in payload:
            continue
        value = payload[key]
        if not isinstance(value, str):
            raise RuntimeControlError(f"{key} must be a string")
        normalized_value = value.strip()
        allow_empty = options.get(
            "interaction_mode"
        ) == "ai_counselor_human_client" and key in {
            "initial_client_transcript",
            "shared_case",
        }
        if not normalized_value and not allow_empty:
            raise RuntimeControlError(f"{key} must not be empty")
        if (
            key == "realtime_model"
            and normalized_value not in SUPPORTED_REALTIME_MODELS
        ):
            supported = ", ".join(sorted(SUPPORTED_REALTIME_MODELS))
            raise RuntimeControlError(
                f"realtime_model must be one of: {supported}"
            )
        if (
            key == "prompting_llm_model"
            and normalized_value not in SUPPORTED_PROMPTING_LLM_MODELS
        ):
            supported = ", ".join(sorted(SUPPORTED_PROMPTING_LLM_MODELS))
            raise RuntimeControlError(
                f"prompting_llm_model must be one of: {supported}"
            )
        if (
            key == "timing_llm_model"
            and normalized_value not in SUPPORTED_TIMING_LLM_MODELS
        ):
            supported = ", ".join(sorted(SUPPORTED_TIMING_LLM_MODELS))
            raise RuntimeControlError(
                f"timing_llm_model must be one of: {supported}"
            )
        if (
            key == "summary_llm_model"
            and normalized_value not in SUPPORTED_SUMMARY_LLM_MODELS
        ):
            supported = ", ".join(sorted(SUPPORTED_SUMMARY_LLM_MODELS))
            raise RuntimeControlError(
                f"summary_llm_model must be one of: {supported}"
            )
        if (
            key == "prompting_llm_reasoning_effort"
            and normalized_value not in SUPPORTED_PROMPTING_LLM_REASONING_EFFORTS
        ):
            supported = ", ".join(
                sorted(SUPPORTED_PROMPTING_LLM_REASONING_EFFORTS)
            )
            raise RuntimeControlError(
                "prompting_llm_reasoning_effort must be one of: "
                f"{supported}"
            )
        if (
            key == "timing_llm_reasoning_effort"
            and normalized_value not in SUPPORTED_TIMING_LLM_REASONING_EFFORTS
        ):
            supported = ", ".join(sorted(SUPPORTED_TIMING_LLM_REASONING_EFFORTS))
            raise RuntimeControlError(
                f"timing_llm_reasoning_effort must be one of: {supported}"
            )
        if (
            key == "summary_llm_reasoning_effort"
            and normalized_value not in SUPPORTED_SUMMARY_LLM_REASONING_EFFORTS
        ):
            supported = ", ".join(sorted(SUPPORTED_SUMMARY_LLM_REASONING_EFFORTS))
            raise RuntimeControlError(
                f"summary_llm_reasoning_effort must be one of: {supported}"
            )
        options[key] = normalized_value
    for key in [
        "realtime_output_speed",
        "counselor_realtime_output_speed",
        "client_realtime_output_speed",
    ]:
        if key not in payload:
            continue
        value = payload[key]
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not MIN_REALTIME_OUTPUT_SPEED <= value <= MAX_REALTIME_OUTPUT_SPEED
        ):
            raise RuntimeControlError(
                f"{key} must be a number between "
                f"{MIN_REALTIME_OUTPUT_SPEED} and {MAX_REALTIME_OUTPUT_SPEED}"
            )
        options[key] = float(value)
    if "speaker_gains" in payload:
        value = payload["speaker_gains"]
        if not isinstance(value, Mapping):
            raise RuntimeControlError("speaker_gains must be a JSON object")
        speaker_gains: dict[str, float] = {}
        for speaker, gain in value.items():
            if speaker not in SUPPORTED_SPEAKER_GAIN_KEYS:
                supported = ", ".join(sorted(SUPPORTED_SPEAKER_GAIN_KEYS))
                raise RuntimeControlError(
                    f"speaker_gains keys must be one of: {supported}"
                )
            if (
                not isinstance(gain, (int, float))
                or isinstance(gain, bool)
                or gain <= 0
            ):
                raise RuntimeControlError(
                    f"speaker_gains.{speaker} must be a positive number"
                )
            speaker_gains[str(speaker)] = float(gain)
        if not speaker_gains:
            raise RuntimeControlError("speaker_gains must not be empty")
        options["speaker_gains"] = speaker_gains
    if "participants" in payload:
        options["participants"] = _normalize_participants_start_option(
            payload["participants"],
            interaction_mode=options.get("interaction_mode"),
        )
    if options.get("interaction_mode") == "ai_counselor_human_client":
        _normalize_human_client_start_options(options)
    return options


def _normalize_human_client_start_options(options: dict[str, Any]) -> None:
    required = {
        "participant_mode": "one_client",
        "speaker_selection_policy": "fixed_round_robin",
        "fixed_speaker_sequence": ("counselor", "client"),
        "initial_client_transcript": "",
        "realtime_api_centered_mode": True,
    }
    for key, expected in required.items():
        value = options.get(key, expected)
        if value != expected:
            raise RuntimeControlError(
                f"ai_counselor_human_client requires {key}={expected}"
            )
        options[key] = expected
    for key in ("client_prompt", "client_tts_voice", "client_realtime_output_speed"):
        if key in options:
            raise RuntimeControlError(f"human client must not have {key}")
    options.setdefault("shared_case", "")
    options.setdefault("human_interrupts_enabled", True)


def _normalize_participants_start_option(
    value: Any,
    *,
    interaction_mode: str | None = None,
) -> dict[str, dict[str, Any]]:
    if not isinstance(value, Mapping):
        raise RuntimeControlError("participants must be a JSON object")
    if not value:
        raise RuntimeControlError("participants must not be empty")
    participants: dict[str, dict[str, Any]] = {}
    for participant_id, raw_participant in value.items():
        if not isinstance(participant_id, str) or not participant_id.strip():
            raise RuntimeControlError("participants keys must be non-empty strings")
        normalized_id = participant_id.strip()
        if not isinstance(raw_participant, Mapping):
            raise RuntimeControlError(
                f"participants.{normalized_id} must be a JSON object"
            )
        unknown_keys = set(raw_participant) - SUPPORTED_PARTICIPANT_OPTION_KEYS
        if unknown_keys:
            joined = ", ".join(sorted(str(key) for key in unknown_keys))
            raise RuntimeControlError(
                f"unsupported participants.{normalized_id} option(s): {joined}"
            )
        participant: dict[str, Any] = {}
        speaker_id = raw_participant.get("speaker_id", normalized_id)
        if not isinstance(speaker_id, str) or not speaker_id.strip():
            raise RuntimeControlError(
                f"participants.{normalized_id}.speaker_id must be a non-empty string"
            )
        speaker_id = speaker_id.strip()
        if speaker_id != normalized_id:
            raise RuntimeControlError(
                "participant speaker_id must match its participants mapping key"
            )
        participant["speaker_id"] = speaker_id

        role = raw_participant.get("role")
        if not isinstance(role, str) or not role.strip():
            raise RuntimeControlError(
                f"participants.{normalized_id}.role must be a non-empty string"
            )
        role = role.strip()
        if role not in SUPPORTED_PARTICIPANT_ROLES:
            supported = ", ".join(sorted(SUPPORTED_PARTICIPANT_ROLES))
            raise RuntimeControlError(
                f"participants.{normalized_id}.role must be one of: {supported}"
            )
        participant["role"] = role

        if (
            "actor_kind" in raw_participant
            or interaction_mode == "human_counselor_ai_client"
        ):
            actor_kind = str(raw_participant.get("actor_kind", "ai")).strip()
            if actor_kind not in SUPPORTED_ACTOR_KINDS:
                supported = ", ".join(sorted(SUPPORTED_ACTOR_KINDS))
                raise RuntimeControlError(
                    f"participants.{normalized_id}.actor_kind must be one of: {supported}"
                )
            participant["actor_kind"] = actor_kind

        display_name = raw_participant.get("display_name")
        if not isinstance(display_name, str) or not display_name.strip():
            raise RuntimeControlError(
                f"participants.{normalized_id}.display_name must be a non-empty string"
            )
        participant["display_name"] = display_name.strip()

        for key in [
            "prompt_path",
            "public_profile_path",
            "private_profile_path",
            "prompt_source",
            "public_profile_source",
            "private_profile_source",
            "voice",
        ]:
            if key not in raw_participant:
                continue
            field_value = raw_participant[key]
            if not isinstance(field_value, str) or not field_value.strip():
                raise RuntimeControlError(
                    f"participants.{normalized_id}.{key} must be a non-empty string"
                )
            participant[key] = field_value.strip()

        if "initial_transcript" in raw_participant:
            initial_transcript = raw_participant["initial_transcript"]
            if not isinstance(initial_transcript, str):
                raise RuntimeControlError(
                    f"participants.{normalized_id}.initial_transcript must be a string"
                )
            participant["initial_transcript"] = initial_transcript.strip()

        if "realtime_output_speed" in raw_participant:
            output_speed = raw_participant["realtime_output_speed"]
            if (
                not isinstance(output_speed, (int, float))
                or isinstance(output_speed, bool)
                or not MIN_REALTIME_OUTPUT_SPEED
                <= output_speed
                <= MAX_REALTIME_OUTPUT_SPEED
            ):
                raise RuntimeControlError(
                    f"participants.{normalized_id}.realtime_output_speed must be a "
                    f"number between {MIN_REALTIME_OUTPUT_SPEED} and "
                    f"{MAX_REALTIME_OUTPUT_SPEED}"
                )
            participant["realtime_output_speed"] = float(output_speed)
        participants[normalized_id] = participant

    if (
        "counselor" not in participants
        and interaction_mode == "human_counselor_ai_client"
    ):
        participants = {
            "counselor": {
                "speaker_id": "counselor",
                "role": "counselor",
                "display_name": "カウンセラー",
                "actor_kind": "human",
            },
            **participants,
        }
    if "counselor" not in participants:
        raise RuntimeControlError("participants must include counselor")
    if participants["counselor"]["role"] != "counselor":
        raise RuntimeControlError("participants.counselor.role must be counselor")
    if not any(
        participant["role"] == "client"
        for participant_id, participant in participants.items()
        if participant_id != "counselor"
    ):
        raise RuntimeControlError("participants must include at least one client")
    return participants


def apply_runtime_start_options(
    settings: Any,
    options: Mapping[str, Any] | None = None,
) -> Any:
    normalized_options = normalize_runtime_start_options(options or {})
    if not {
        "ai_routes",
        "realtime_api_centered_mode",
        "interaction_mode",
        "participant_mode",
        "human_input_mode",
        "human_stt_submit_policy",
        "human_interrupts_enabled",
        "initial_client_transcript",
        "stop_condition",
        "max_turns",
        "max_elapsed_seconds",
        "closing_start_elapsed_seconds",
        "force_stop_after_closing_turns",
        "speaker_selection_policy",
        "fixed_speaker_sequence",
        "counselor_prompt",
        "client_prompt",
        "shared_case",
        "realtime_model",
        "prompting_llm_model",
        "prompting_llm_reasoning_effort",
        "timing_llm_model",
        "timing_llm_reasoning_effort",
        "summary_llm_model",
        "summary_llm_reasoning_effort",
        "counselor_tts_voice",
        "client_tts_voice",
        "realtime_output_speed",
        "counselor_realtime_output_speed",
        "client_realtime_output_speed",
        "speaker_gains",
        "participants",
    }.intersection(normalized_options):
        return settings
    latency = getattr(settings, "latency", None)
    runtime = getattr(settings, "runtime", None)
    openai = getattr(settings, "openai", None)
    audio = getattr(settings, "audio", None)
    participants = getattr(settings, "participants", None)
    shared_case = getattr(settings, "shared_case", None)
    ai_updates = _apply_ai_route_options(settings, normalized_options)
    latency_updates: dict[str, Any] = {}
    runtime_updates: dict[str, Any] = {}
    openai_updates: dict[str, Any] = {}
    audio_updates: dict[str, Any] = {}
    participant_updates: dict[str, Any] = {}
    shared_case_updates: dict[str, Any] = {}
    if "realtime_api_centered_mode" in normalized_options:
        latency_updates["realtime_api_centered_mode"] = normalized_options[
            "realtime_api_centered_mode"
        ]
    if "initial_client_transcript" in normalized_options:
        runtime_updates["initial_client_transcript"] = normalized_options[
            "initial_client_transcript"
        ]
    for key in [
        "interaction_mode",
        "participant_mode",
        "human_input_mode",
        "human_stt_submit_policy",
        "human_interrupts_enabled",
    ]:
        if key in normalized_options:
            runtime_updates[key] = normalized_options[key]
    if "stop_condition" in normalized_options:
        runtime_updates["stop_condition"] = normalized_options["stop_condition"]
    if "max_turns" in normalized_options:
        runtime_updates["max_turns"] = normalized_options["max_turns"]
    if "max_elapsed_seconds" in normalized_options:
        runtime_updates["max_elapsed_seconds"] = normalized_options[
            "max_elapsed_seconds"
        ]
    if "closing_start_elapsed_seconds" in normalized_options:
        runtime_updates["closing_start_elapsed_seconds"] = normalized_options[
            "closing_start_elapsed_seconds"
        ]
    if "force_stop_after_closing_turns" in normalized_options:
        runtime_updates["force_stop_after_closing_turns"] = normalized_options[
            "force_stop_after_closing_turns"
        ]
    if "speaker_selection_policy" in normalized_options:
        runtime_updates["speaker_selection_policy"] = normalized_options[
            "speaker_selection_policy"
        ]
    if "fixed_speaker_sequence" in normalized_options:
        runtime_updates["fixed_speaker_sequence"] = normalized_options[
            "fixed_speaker_sequence"
        ]
        if normalized_options.get("interaction_mode") == "ai_counselor_human_client":
            runtime_updates["fixed_speaker_sequence"] = list(
                runtime_updates["fixed_speaker_sequence"]
            )
    if "shared_case" in normalized_options:
        shared_case_updates["prompt_source"] = normalized_options["shared_case"]
        shared_case_updates["prompt_path"] = None
    if "realtime_model" in normalized_options:
        openai_updates["realtime_model"] = normalized_options["realtime_model"]
    if "prompting_llm_model" in normalized_options:
        openai_updates["prompting_llm_model"] = normalized_options[
            "prompting_llm_model"
        ]
    if "prompting_llm_reasoning_effort" in normalized_options:
        openai_updates["prompting_llm_reasoning_effort"] = normalized_options[
            "prompting_llm_reasoning_effort"
        ]
    if "timing_llm_model" in normalized_options:
        openai_updates["timing_llm_model"] = normalized_options["timing_llm_model"]
    if "timing_llm_reasoning_effort" in normalized_options:
        openai_updates["timing_llm_reasoning_effort"] = normalized_options[
            "timing_llm_reasoning_effort"
        ]
    if "summary_llm_model" in normalized_options:
        openai_updates["summary_llm_model"] = normalized_options["summary_llm_model"]
    if "summary_llm_reasoning_effort" in normalized_options:
        openai_updates["summary_llm_reasoning_effort"] = normalized_options[
            "summary_llm_reasoning_effort"
        ]
    if "counselor_tts_voice" in normalized_options:
        openai_updates["counselor_tts_voice"] = normalized_options[
            "counselor_tts_voice"
        ]
    if "client_tts_voice" in normalized_options:
        openai_updates["client_tts_voice"] = normalized_options["client_tts_voice"]
    if "realtime_output_speed" in normalized_options:
        openai_updates["realtime_output_speed"] = normalized_options[
            "realtime_output_speed"
        ]
    if "counselor_realtime_output_speed" in normalized_options:
        openai_updates["counselor_realtime_output_speed"] = normalized_options[
            "counselor_realtime_output_speed"
        ]
    if "client_realtime_output_speed" in normalized_options:
        openai_updates["client_realtime_output_speed"] = normalized_options[
            "client_realtime_output_speed"
        ]
    if "speaker_gains" in normalized_options:
        current_gains = dict(getattr(audio, "speaker_gains", {}) or {})
        current_gains.update(normalized_options["speaker_gains"])
        audio_updates["speaker_gains"] = current_gains
    if "participants" in normalized_options:
        participant_updates = normalized_options["participants"]
    elif normalized_options.get("interaction_mode") == "ai_counselor_human_client":
        previous = _participant_updates_from_settings(participants)
        counselor = previous.get(
            "counselor", {"role": "counselor", "display_name": "AIカウンセラー"}
        )
        counselor["actor_kind"] = "ai"
        participant_updates = {
            "counselor": counselor,
            "client": {
                "speaker_id": "client",
                "role": "client",
                "actor_kind": "human",
                "display_name": normalized_options.get(
                    "client_display_name", "クライアント"
                ),
            },
        }
    elif (
        normalized_options.get("interaction_mode")
        == "human_counselor_ai_client"
    ):
        participant_updates = _participant_updates_with_human_counselor(
            participants
        )
    elif (
        "counselor_prompt" in normalized_options
        or "client_prompt" in normalized_options
    ):
        participant_updates = _participant_updates_from_settings(participants)
    if "counselor_prompt" in normalized_options:
        _apply_prompt_to_participant_role(
            participant_updates,
            role="counselor",
            prompt=normalized_options["counselor_prompt"],
        )
    if "client_prompt" in normalized_options:
        _apply_prompt_to_participant_role(
            participant_updates,
            role="client",
            prompt=normalized_options["client_prompt"],
        )
    if (
        latency is not None
        and hasattr(settings, "model_copy")
        and hasattr(latency, "model_copy")
    ):
        update: dict[str, Any] = {}
        if ai_updates is not None:
            update["ai"] = ai_updates
        if latency_updates:
            update["latency"] = latency.model_copy(update=latency_updates)
        if runtime_updates and runtime is not None and hasattr(runtime, "model_copy"):
            update["runtime"] = runtime.model_copy(update=runtime_updates)
        if openai_updates and openai is not None and hasattr(openai, "model_copy"):
            update["openai"] = openai.model_copy(update=openai_updates)
        if audio_updates and audio is not None and hasattr(audio, "model_copy"):
            update["audio"] = audio.model_copy(update=audio_updates)
        if participant_updates:
            update["participants"] = {
                participant_id: ParticipantSection.model_validate(participant)
                for participant_id, participant in participant_updates.items()
            }
        if (
            shared_case_updates
            and shared_case is not None
            and hasattr(shared_case, "model_copy")
        ):
            update["shared_case"] = shared_case.model_copy(
                update=shared_case_updates
            )
        result = settings.model_copy(update=update)
        if (
            isinstance(result, RuntimeSettings)
            and result.runtime.interaction_mode == "ai_counselor_human_client"
        ):
            # model_copy does not run cross-section validators.
            result = RuntimeSettings.model_validate(result.model_dump())
        return result

    next_settings = copy(settings)
    if ai_updates is not None:
        next_settings.ai = ai_updates
    next_latency = copy(latency) if latency is not None else SimpleNamespace()
    for key, value in latency_updates.items():
        setattr(next_latency, key, value)
    if latency_updates:
        setattr(next_settings, "latency", next_latency)
    next_runtime = copy(runtime) if runtime is not None else SimpleNamespace()
    for key, value in runtime_updates.items():
        setattr(next_runtime, key, value)
    if runtime_updates:
        setattr(next_settings, "runtime", next_runtime)
    next_openai = copy(openai) if openai is not None else SimpleNamespace()
    for key, value in openai_updates.items():
        setattr(next_openai, key, value)
    if openai_updates:
        setattr(next_settings, "openai", next_openai)
    next_audio = copy(audio) if audio is not None else SimpleNamespace()
    for key, value in audio_updates.items():
        setattr(next_audio, key, value)
    if audio_updates:
        setattr(next_settings, "audio", next_audio)
    if participant_updates:
        setattr(
            next_settings,
            "participants",
            {
                participant_id: SimpleNamespace(**participant)
                for participant_id, participant in participant_updates.items()
            },
        )
    next_shared_case = (
        copy(shared_case) if shared_case is not None else SimpleNamespace()
    )
    for key, value in shared_case_updates.items():
        setattr(next_shared_case, key, value)
    if shared_case_updates:
        setattr(next_settings, "shared_case", next_shared_case)
    return next_settings


def _apply_ai_route_options(settings: Any, options: Mapping[str, Any]) -> Any:
    route_options = options.get("ai_routes", {})
    if not route_options and (
        not set(AI_ROUTE_MODEL_FIELDS.values()).intersection(options)
        or getattr(settings, "ai", None) is None
    ):
        return None
    ai = effective_ai_settings(settings)
    for name, override in route_options.items():
        provider_id = override["provider"]
        if provider_id not in ai.providers:
            raise RuntimeControlError(f"route={name}: unknown provider")
        route = ai.routes[name]
        if provider_id not in route.targets:
            raise RuntimeControlError(f"route={name}: provider target is missing")
        route.provider = provider_id
        if "model_ref" in override:
            target = route.targets[provider_id]
            field = (
                "model" if ai.providers[provider_id].kind == "openai" else "deployment"
            )
            setattr(target, field, override["model_ref"])
    # Legacy model selectors update only the selected OpenAI target.
    for name, field in AI_ROUTE_MODEL_FIELDS.items():
        route = ai.routes[name]
        provider_id = route.provider or ai.default_provider
        if (
            field in options
            and ai.providers[provider_id].kind == "openai"
            and "model_ref" not in route_options.get(name, {})
        ):
            route.targets[provider_id].model = options[field]
    active = settings.model_copy(update={"ai": ai})
    try:
        # Resolve nonsecret settings here; credentials are checked at runtime creation.
        registry = ProviderRegistry(active)
        for name in route_options:
            registry.resolve_route(name)
    except ProviderConfigurationError as exc:
        raise RuntimeControlError(str(exc)) from exc
    return ai


def _participant_updates_with_human_counselor(
    participants: Any,
) -> dict[str, dict[str, Any]]:
    updates = _participant_updates_from_settings(participants)
    counselor = updates.get("counselor")
    if counselor is None:
        return {}
    counselor["actor_kind"] = "human"
    return updates


def _participant_updates_from_settings(
    participants: Any,
) -> dict[str, dict[str, Any]]:
    if not isinstance(participants, Mapping):
        return {}
    updates: dict[str, dict[str, Any]] = {}
    for participant_id, participant in participants.items():
        if hasattr(participant, "model_dump"):
            data = dict(participant.model_dump())
        elif isinstance(participant, Mapping):
            data = dict(participant)
        else:
            data = dict(vars(participant))
        updates[str(participant_id)] = data
    return updates


def _apply_prompt_to_participant_role(
    participants: Mapping[str, dict[str, Any]],
    *,
    role: str,
    prompt: str,
) -> None:
    for participant in participants.values():
        if participant.get("role") != role:
            continue
        participant["prompt_source"] = prompt
        participant["prompt_path"] = None


def _normalize_fixed_speaker_sequence(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        value = _parse_fixed_speaker_sequence_string(value)
    if isinstance(value, Mapping):
        value = _parse_fixed_speaker_sequence_mapping(value)
    if not isinstance(value, (list, tuple)):
        raise RuntimeControlError(
            "fixed_speaker_sequence must be a JSON array "
            f"(received {type(value).__name__})"
        )
    if not value:
        raise RuntimeControlError("fixed_speaker_sequence must not be empty")
    sequence: list[str] = []
    for index, speaker_id in enumerate(value):
        if not isinstance(speaker_id, str) or not speaker_id.strip():
            raise RuntimeControlError(
                f"fixed_speaker_sequence[{index}] must be a non-empty string"
            )
        sequence.append(speaker_id.strip())
    return tuple(sequence)


def _parse_fixed_speaker_sequence_string(value: str) -> Any:
    normalized = value.strip()
    if not normalized:
        raise RuntimeControlError("fixed_speaker_sequence must not be empty")
    if normalized.startswith("["):
        try:
            return json.loads(normalized)
        except json.JSONDecodeError as exc:
            raise RuntimeControlError(
                "fixed_speaker_sequence string must be a JSON array"
            ) from exc
    if "," not in normalized and "\n" not in normalized:
        return value
    parts = [
        part.strip()
        for part in normalized.replace("\n", ",").split(",")
        if part.strip()
    ]
    return parts


def _parse_fixed_speaker_sequence_mapping(value: Mapping[str, Any]) -> Any:
    for key in ("items", "value", "sequence", "fixed_speaker_sequence"):
        nested = value.get(key)
        if nested is not None:
            if isinstance(nested, str):
                return _parse_fixed_speaker_sequence_string(nested)
            return nested
    indexed_items: list[tuple[int, Any]] = []
    for key, item in value.items():
        if not isinstance(key, str) or not key.isdigit():
            return value
        indexed_items.append((int(key), item))
    return [item for _, item in sorted(indexed_items)]


def main(argv: Sequence[str] | None = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) > 1:
        raise RuntimeControlError(
            "usage: python -m counseling_voice_demo.runtime.control_api [config_path]"
        )

    config_path = args[0] if args else None
    app, host, port = build_default_control_app(config_path)
    run_control_api(app, host=host, port=port)


def _load_default_control_config(config: Any = None) -> Any:
    from counseling_voice_demo.runtime.config import RuntimeSettings, load_runtime_config

    if config is None:
        return load_runtime_config()
    if isinstance(config, RuntimeSettings):
        return config
    if isinstance(config, Path | str):
        return load_runtime_config(config)
    return RuntimeSettings.model_validate(config)


def _runtime_events_path(
    session_id: str,
    *,
    session_dir: Path | str | None,
    sessions_dir: Path | str | None,
) -> Path:
    if session_dir is not None:
        return Path(session_dir) / "internal" / "events" / f"{session_id}.jsonl"
    if sessions_dir is not None:
        return Path(sessions_dir) / session_id / "internal" / "events" / f"{session_id}.jsonl"
    raise RuntimeControlError("session_dir or sessions_dir is required")


def _runtime_response_instructions_path(
    session_id: str,
    *,
    session_dir: Path | str | None,
    sessions_dir: Path | str | None,
) -> Path:
    if session_dir is not None:
        return (
            Path(session_dir)
            / "internal"
            / "prompts"
            / "response_instructions.jsonl"
        )
    if sessions_dir is not None:
        return (
            Path(sessions_dir)
            / session_id
            / "internal"
            / "prompts"
            / "response_instructions.jsonl"
        )
    raise RuntimeControlError("session_dir or sessions_dir is required")


def _status_to_dict(status: RuntimeStatus) -> dict[str, Any]:
    return asdict(status)


__all__ = [
    "RuntimeAlreadyStartedError",
    "RuntimeControlError",
    "RuntimeControlService",
    "RuntimeFactory",
    "MONITOR_AUDIO_RUNTIME_WAIT_SECONDS",
    "WebSocketJsonLike",
    "build_default_control_app",
    "build_default_control_service",
    "create_app",
    "main",
    "read_runtime_events",
    "read_runtime_response_instructions",
    "run_control_api",
    "stream_audio_monitor_to_websocket",
]


if __name__ == "__main__":
    main()
