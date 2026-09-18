from __future__ import annotations

import asyncio
import importlib
import json
from typing import Any

import pytest

from counseling_voice_demo.runtime.realtime_transport import (
    DEFAULT_REALTIME_TRANSCRIPTION_URL,
    RealtimeWebSocketDependencyError,
    WebSocketJsonTransport,
    build_realtime_session_url,
    build_realtime_transcription_headers,
    connect_realtime_session,
    connect_realtime_transcription,
)


class _FakeWebSocket:
    def __init__(self, incoming: list[str] | None = None) -> None:
        self.incoming = list(incoming or [])
        self.sent: list[str] = []
        self.closed = False

    async def send(self, message: str) -> None:
        self.sent.append(message)

    def __aiter__(self) -> _FakeWebSocket:
        return self

    async def __anext__(self) -> str:
        if not self.incoming:
            raise StopAsyncIteration
        return self.incoming.pop(0)

    async def close(self) -> None:
        self.closed = True


class _FakeConnection:
    def __init__(self, websocket: _FakeWebSocket) -> None:
        self.websocket = websocket
        self.exited = False

    async def __aenter__(self) -> _FakeWebSocket:
        return self.websocket

    async def __aexit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.exited = True


class _FakeConnect:
    def __init__(self, websocket: _FakeWebSocket) -> None:
        self.websocket = websocket
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def __call__(self, url: str, **kwargs: Any) -> _FakeConnection:
        self.calls.append((url, kwargs))
        return _FakeConnection(self.websocket)


def test_websocket_json_transport_send_json_sends_json_string() -> None:
    async def scenario() -> None:
        websocket = _FakeWebSocket()
        transport = WebSocketJsonTransport(websocket)

        await transport.send_json({"type": "input_audio_buffer.commit", "count": 1})

        assert len(websocket.sent) == 1
        assert isinstance(websocket.sent[0], str)
        assert json.loads(websocket.sent[0]) == {
            "type": "input_audio_buffer.commit",
            "count": 1,
        }

    asyncio.run(scenario())


def test_websocket_json_transport_async_iteration_parses_json_objects() -> None:
    async def scenario() -> None:
        transport = WebSocketJsonTransport(
            _FakeWebSocket(
                [
                    '{"type": "transcription_session.created", "id": "sess_1"}',
                    '{"type": "conversation.item.input_audio_transcription.completed", "transcript": "完了"}',
                ]
            )
        )

        events: list[dict[str, Any]] = []
        async for event in transport:
            events.append(event)

        assert events == [
            {"type": "transcription_session.created", "id": "sess_1"},
            {
                "type": "conversation.item.input_audio_transcription.completed",
                "transcript": "完了",
            },
        ]

    asyncio.run(scenario())


def test_connect_realtime_transcription_requires_websockets_without_connect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_import_module = importlib.import_module

    def fake_import_module(name: str, package: str | None = None) -> Any:
        if name.startswith("websockets"):
            raise ModuleNotFoundError("No module named 'websockets'")
        return real_import_module(name, package)

    monkeypatch.setattr(importlib, "import_module", fake_import_module)

    async def scenario() -> None:
        with pytest.raises(RealtimeWebSocketDependencyError, match="optional 'websockets' package"):
            async with connect_realtime_transcription(api_key="test-api-key"):
                raise AssertionError("connection should not be opened")

    asyncio.run(scenario())


def test_connect_realtime_transcription_passes_headers_without_repr_leaking_key() -> None:
    async def scenario() -> None:
        websocket = _FakeWebSocket()
        fake_connect = _FakeConnect(websocket)
        api_key = "api-key-that-must-not-be-in-repr"

        async with connect_realtime_transcription(api_key=api_key, connect=fake_connect) as transport:
            await transport.send_json({"type": "ping"})

        assert fake_connect.calls
        url, kwargs = fake_connect.calls[0]
        assert url == DEFAULT_REALTIME_TRANSCRIPTION_URL
        headers = kwargs.get("additional_headers") or kwargs.get("extra_headers")
        assert headers is not None
        assert set(headers) == {"Authorization"}
        assert headers["Authorization"].startswith("Bearer ")
        assert kwargs["ping_interval"] is None
        assert kwargs["ping_timeout"] is None
        header_repr_leaked = api_key in repr(headers)
        kwargs_repr_leaked = api_key in repr(kwargs)
        assert not header_repr_leaked
        assert not kwargs_repr_leaked
        assert json.loads(websocket.sent[0]) == {"type": "ping"}

    asyncio.run(scenario())


def test_build_realtime_transcription_headers_can_request_beta_header() -> None:
    headers = build_realtime_transcription_headers("test-api-key", use_beta_header=True)

    assert headers["OpenAI-Beta"] == "realtime=v1"
    assert "test-api-key" not in repr(headers)


def test_build_realtime_session_url_encodes_model() -> None:
    assert (
        build_realtime_session_url("gpt-realtime-mini")
        == "wss://api.openai.com/v1/realtime?model=gpt-realtime-mini"
    )


def test_connect_realtime_session_uses_model_url_without_beta_header() -> None:
    async def scenario() -> None:
        websocket = _FakeWebSocket()
        fake_connect = _FakeConnect(websocket)

        async with connect_realtime_session(
            api_key="test-api-key",
            model="gpt-realtime-mini",
            connect=fake_connect,
        ) as transport:
            await transport.send_json({"type": "session.update"})

        url, kwargs = fake_connect.calls[0]
        assert url == "wss://api.openai.com/v1/realtime?model=gpt-realtime-mini"
        headers = kwargs.get("additional_headers") or kwargs.get("extra_headers")
        assert headers is not None
        assert set(headers) == {"Authorization"}
        assert json.loads(websocket.sent[0]) == {"type": "session.update"}

    asyncio.run(scenario())
