from __future__ import annotations

import importlib
import inspect
import json
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlencode, urlsplit, urlunsplit


DEFAULT_REALTIME_MODEL = "gpt-realtime-mini"
DEFAULT_REALTIME_URL = "wss://api.openai.com/v1/realtime"
DEFAULT_REALTIME_TRANSCRIPTION_URL = "wss://api.openai.com/v1/realtime?intent=transcription"
REALTIME_BETA_HEADER_VALUE = "realtime=v1"


class RealtimeWebSocketDependencyError(RuntimeError):
    pass


class RealtimeAuthHeaders(dict[str, str]):
    def __repr__(self) -> str:
        redacted = dict(self)
        for key in redacted:
            if key.lower() in {"authorization", "api-key"}:
                redacted[key] = "<redacted>"
        return repr(redacted)

    __str__ = __repr__


@dataclass(frozen=True)
class RealtimeConnectionSpec:
    url: str
    headers: RealtimeAuthHeaders


def build_azure_openai_base_url(endpoint: str) -> str:
    error = "Azure endpoint must be an HTTPS resource or /openai/v1 URL without credentials, query, or fragment"
    try:
        parsed = urlsplit(endpoint.strip())
        if (
            parsed.scheme != "https"
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
            or parsed.path not in {"", "/", "/openai/v1", "/openai/v1/"}
            or any(char.isspace() for char in endpoint.strip())
        ):
            raise ValueError(error)
        _ = parsed.port
    except ValueError:
        raise ValueError(error) from None
    return urlunsplit(("https", parsed.netloc, "/openai/v1/", "", ""))


def build_azure_realtime_session_url(endpoint: str, deployment: str) -> str:
    parsed = urlsplit(build_azure_openai_base_url(endpoint))
    return urlunsplit(
        (
            "wss",
            parsed.netloc,
            "/openai/v1/realtime",
            urlencode({"model": deployment}),
            "",
        )
    )


def build_azure_realtime_transcription_url(endpoint: str) -> str:
    parsed = urlsplit(build_azure_openai_base_url(endpoint))
    return urlunsplit(
        ("wss", parsed.netloc, "/openai/v1/realtime", "intent=transcription", "")
    )


class WebSocketJsonTransport:
    def __init__(self, websocket: Any) -> None:
        self._websocket = websocket
        self._iterator: Any | None = None

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(websocket={self._websocket.__class__.__name__})"

    async def send_json(self, event: Mapping[str, Any]) -> None:
        await self._websocket.send(json.dumps(dict(event)))

    def __aiter__(self) -> WebSocketJsonTransport:
        self._iterator = self._websocket.__aiter__()
        return self

    async def __anext__(self) -> dict[str, Any]:
        if self._iterator is None:
            self._iterator = self._websocket.__aiter__()
        message = await self._iterator.__anext__()
        event = json.loads(message)
        if not isinstance(event, dict):
            raise ValueError("Realtime WebSocket message must be a JSON object")
        return event


@asynccontextmanager
async def connect_realtime_transcription(
    api_key: str,
    url: str = DEFAULT_REALTIME_TRANSCRIPTION_URL,
    connect: Any | None = None,
    use_beta_header: bool = False,
    ping_interval: float | None = None,
    ping_timeout: float | None = None,
) -> AsyncIterator[WebSocketJsonTransport]:
    headers = build_realtime_transcription_headers(api_key, use_beta_header=use_beta_header)
    async with connect_realtime(
        RealtimeConnectionSpec(url=url, headers=headers),
        connect=connect,
        ping_interval=ping_interval,
        ping_timeout=ping_timeout,
    ) as transport:
        yield transport


@asynccontextmanager
async def connect_realtime(
    spec: RealtimeConnectionSpec,
    *,
    connect: Any | None = None,
    ping_interval: float | None = None,
    ping_timeout: float | None = None,
) -> AsyncIterator[WebSocketJsonTransport]:
    connect_fn = connect if connect is not None else _load_websockets_connect()
    connection = connect_fn(
        spec.url,
        **_connect_kwargs(
            connect_fn,
            spec.headers,
            ping_interval=ping_interval,
            ping_timeout=ping_timeout,
        ),
    )

    if hasattr(connection, "__aenter__") and hasattr(connection, "__aexit__"):
        async with connection as websocket:
            yield WebSocketJsonTransport(websocket)
        return

    websocket = await connection if inspect.isawaitable(connection) else connection
    try:
        yield WebSocketJsonTransport(websocket)
    finally:
        close = getattr(websocket, "close", None)
        if close is not None:
            result = close()
            if inspect.isawaitable(result):
                await result


@asynccontextmanager
async def connect_realtime_session(
    api_key: str,
    *,
    model: str = DEFAULT_REALTIME_MODEL,
    url: str | None = None,
    connect: Any | None = None,
    use_beta_header: bool = False,
    ping_interval: float | None = None,
    ping_timeout: float | None = None,
) -> AsyncIterator[WebSocketJsonTransport]:
    realtime_url = url or build_realtime_session_url(model)
    async with connect_realtime_transcription(
        api_key=api_key,
        url=realtime_url,
        connect=connect,
        use_beta_header=use_beta_header,
        ping_interval=ping_interval,
        ping_timeout=ping_timeout,
    ) as transport:
        yield transport


def build_realtime_session_url(model: str = DEFAULT_REALTIME_MODEL) -> str:
    return f"{DEFAULT_REALTIME_URL}?{urlencode({'model': model})}"


def build_realtime_transcription_headers(
    api_key: str,
    *,
    use_beta_header: bool = False,
) -> RealtimeAuthHeaders:
    headers = RealtimeAuthHeaders({"Authorization": f"Bearer {api_key}"})
    if use_beta_header:
        headers["OpenAI-Beta"] = REALTIME_BETA_HEADER_VALUE
    return headers


def _load_websockets_connect() -> Any:
    try:
        return importlib.import_module("websockets.asyncio.client").connect
    except (ImportError, AttributeError):
        try:
            return importlib.import_module("websockets").connect
        except (ImportError, AttributeError) as exc:
            raise RealtimeWebSocketDependencyError(
                "The optional 'websockets' package is required when no connect "
                "callable is provided. Install websockets or pass a compatible "
                "connect function."
            ) from exc


def _connect_header_kwargs(connect: Any, headers: RealtimeAuthHeaders) -> dict[str, RealtimeAuthHeaders]:
    parameter_name = _headers_parameter_name(connect)
    return {parameter_name: headers}


def _connect_kwargs(
    connect: Any,
    headers: RealtimeAuthHeaders,
    *,
    ping_interval: float | None,
    ping_timeout: float | None,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = dict(_connect_header_kwargs(connect, headers))
    kwargs.update(
        _supported_optional_kwargs(
            connect,
            {
                "ping_interval": ping_interval,
                "ping_timeout": ping_timeout,
            },
        )
    )
    return kwargs


def _supported_optional_kwargs(connect: Any, options: dict[str, Any]) -> dict[str, Any]:
    try:
        parameters = inspect.signature(connect).parameters
    except (TypeError, ValueError):
        return options

    if any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()):
        return options
    return {name: value for name, value in options.items() if name in parameters}


def _headers_parameter_name(connect: Any) -> str:
    try:
        parameters = inspect.signature(connect).parameters
    except (TypeError, ValueError):
        return "additional_headers"

    if "additional_headers" in parameters:
        return "additional_headers"
    if "extra_headers" in parameters:
        return "extra_headers"
    if any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()):
        return "additional_headers"
    return "additional_headers"
