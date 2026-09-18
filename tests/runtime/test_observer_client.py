from __future__ import annotations

import sys

import pytest

from counseling_voice_demo.runtime.observer_client import (
    RuntimeControlClient,
    RuntimeObserverClientError,
    build_runtime_control_endpoint,
)


class FakeResponse:
    def __init__(self, payload, *, error: Exception | None = None) -> None:
        self._payload = payload
        self._error = error

    def raise_for_status(self) -> None:
        if self._error is not None:
            raise self._error

    def json(self):
        return self._payload


class RecordingSession:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, float, dict | None]] = []
        self.responses: dict[tuple[str, str], FakeResponse] = {}

    def post(self, url: str, *, timeout: float, json: dict | None = None) -> FakeResponse:
        self.calls.append(("post", url, timeout, json))
        return self.responses[("post", url)]

    def get(self, url: str, *, timeout: float) -> FakeResponse:
        self.calls.append(("get", url, timeout, None))
        return self.responses[("get", url)]


def test_build_runtime_control_endpoint_uses_session_monitor_path() -> None:
    endpoint = build_runtime_control_endpoint(
        host="127.0.0.1",
        port=8765,
        session_id="session 1/2",
    )

    assert endpoint.base_url == "http://127.0.0.1:8765"
    assert endpoint.monitor_ws_url == (
        "ws://127.0.0.1:8765/runtime/sessions/session%201%2F2/monitor-audio"
    )


def test_build_runtime_control_endpoint_brackets_ipv6_hosts() -> None:
    endpoint = build_runtime_control_endpoint(host="::1", port=8765)

    assert endpoint.base_url == "http://[::1]:8765"
    assert endpoint.monitor_ws_url == "ws://[::1]:8765/runtime/monitor-audio"


def test_runtime_control_client_posts_actions_and_gets_status() -> None:
    session = RecordingSession()
    session.responses[("post", "http://runtime.test/runtime/start")] = FakeResponse(
        {"session_id": "session-runtime", "phase": "running"}
    )
    session.responses[("get", "http://runtime.test/runtime/status")] = FakeResponse(
        {"session_id": "session-runtime", "phase": "running", "completed_turns": 1}
    )
    client = RuntimeControlClient(
        "http://runtime.test/",
        timeout_seconds=3.0,
        session=session,
    )

    assert client.start()["phase"] == "running"
    assert client.status()["completed_turns"] == 1
    assert session.calls == [
        ("post", "http://runtime.test/runtime/start", 3.0, None),
        ("get", "http://runtime.test/runtime/status", 3.0, None),
    ]


def test_runtime_control_client_posts_start_options_as_json() -> None:
    session = RecordingSession()
    session.responses[("post", "http://runtime.test/runtime/start")] = FakeResponse(
        {"session_id": "session-runtime", "phase": "running"}
    )
    client = RuntimeControlClient("http://runtime.test", session=session)

    client.start({"realtime_api_centered_mode": False})

    assert session.calls == [
        (
            "post",
            "http://runtime.test/runtime/start",
            2.0,
            {"realtime_api_centered_mode": False},
        )
    ]


def test_runtime_control_client_posts_pause_reason_as_json() -> None:
    session = RecordingSession()
    session.responses[("post", "http://runtime.test/runtime/pause")] = FakeResponse(
        {"session_id": "session-runtime", "phase": "paused", "pause_reason": "playback"}
    )
    client = RuntimeControlClient("http://runtime.test", session=session)

    response = client.pause(reason="playback")

    assert response["pause_reason"] == "playback"
    assert session.calls == [
        (
            "post",
            "http://runtime.test/runtime/pause",
            2.0,
            {"reason": "playback"},
        )
    ]


def test_runtime_control_client_posts_interrupt_payload() -> None:
    session = RecordingSession()
    session.responses[("post", "http://runtime.test/runtime/interrupt")] = FakeResponse(
        {"played_ms": 1250, "speaker": "counselor"}
    )
    client = RuntimeControlClient("http://runtime.test", session=session)

    response = client.interrupt(
        played_ms=1250,
        speaker="counselor",
        turn_id=3,
        reason="browser_overlap",
    )

    assert response["played_ms"] == 1250
    assert session.calls == [
        (
            "post",
            "http://runtime.test/runtime/interrupt",
            2.0,
            {
                "played_ms": 1250,
                "speaker": "counselor",
                "turn_id": 3,
                "reason": "browser_overlap",
            },
        )
    ]


def test_runtime_control_client_reads_session_events() -> None:
    session = RecordingSession()
    session.responses[
        ("get", "http://runtime.test/runtime/sessions/session%201/events")
    ] = FakeResponse(
        [
            {"event_type": "tts_stream_done"},
            "ignored-non-object",
        ]
    )
    client = RuntimeControlClient("http://runtime.test", session=session)

    assert client.events("session 1") == [{"event_type": "tts_stream_done"}]


def test_runtime_control_client_reads_session_response_instructions() -> None:
    session = RecordingSession()
    session.responses[
        (
            "get",
            "http://runtime.test/runtime/sessions/session%201/response-instructions",
        )
    ] = FakeResponse(
        [
            {"turn_id": 1, "resolved_instructions": "送信済みinstructions"},
            "ignored-non-object",
        ]
    )
    client = RuntimeControlClient("http://runtime.test", session=session)

    assert client.response_instructions("session 1") == [
        {"turn_id": 1, "resolved_instructions": "送信済みinstructions"}
    ]


def test_runtime_control_client_wraps_request_errors() -> None:
    session = RecordingSession()
    session.responses[("post", "http://runtime.test/runtime/stop")] = FakeResponse(
        {},
        error=RuntimeError("boom"),
    )
    client = RuntimeControlClient("http://runtime.test", session=session)

    with pytest.raises(RuntimeObserverClientError, match="POST /runtime/stop"):
        client.stop()


def test_runtime_control_client_reports_missing_requests(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "requests", None)
    client = RuntimeControlClient("http://runtime.test")

    with pytest.raises(RuntimeObserverClientError, match="requests is not installed"):
        client.status()
