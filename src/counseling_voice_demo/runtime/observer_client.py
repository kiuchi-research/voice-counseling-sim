from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from urllib.parse import quote


class RuntimeObserverClientError(RuntimeError):
    """Raised when the observer UI cannot reach the runtime control API."""


@dataclass(frozen=True)
class RuntimeControlEndpoint:
    host: str
    port: int
    base_url: str
    monitor_ws_url: str


def build_runtime_control_endpoint(
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
    session_id: str | None = None,
) -> RuntimeControlEndpoint:
    clean_host = host.strip() or "127.0.0.1"
    clean_port = int(port)
    http_host = _host_for_url(clean_host)
    base_url = f"http://{http_host}:{clean_port}"
    if session_id:
        encoded_session_id = quote(session_id, safe="")
        monitor_path = f"/runtime/sessions/{encoded_session_id}/monitor-audio"
    else:
        monitor_path = "/runtime/monitor-audio"
    return RuntimeControlEndpoint(
        host=clean_host,
        port=clean_port,
        base_url=base_url,
        monitor_ws_url=f"ws://{http_host}:{clean_port}{monitor_path}",
    )


class RuntimeControlClient:
    def __init__(
        self,
        base_url: str,
        *,
        timeout_seconds: float = 2.0,
        session: Any | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = max(0.1, float(timeout_seconds))
        self._session = session

    def start(self, options: dict[str, Any] | None = None) -> dict[str, Any]:
        return self._post("/runtime/start", payload=options)

    def pause(self, reason: str | None = None) -> dict[str, Any]:
        payload = {"reason": reason} if reason else None
        return self._post("/runtime/pause", payload=payload)

    def resume(self) -> dict[str, Any]:
        return self._post("/runtime/resume")

    def stop(self) -> dict[str, Any]:
        return self._post("/runtime/stop")

    def interrupt(
        self,
        *,
        played_ms: int,
        speaker: str | None = None,
        turn_id: int | None = None,
        reason: str | None = None,
        item_id: str | None = None,
        response_id: str | None = None,
        content_index: int | None = None,
        cancel_response: bool = True,
        truncate_item: bool = True,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {"played_ms": played_ms}
        for key, value in {
            "speaker": speaker,
            "turn_id": turn_id,
            "reason": reason,
            "item_id": item_id,
            "response_id": response_id,
            "content_index": content_index,
        }.items():
            if value is not None:
                payload[key] = value
        if not cancel_response:
            payload["cancel_response"] = False
        if not truncate_item:
            payload["truncate_item"] = False
        return self._post("/runtime/interrupt", payload=payload)

    def status(self) -> dict[str, Any]:
        return self._get("/runtime/status")

    def events(self, session_id: str) -> list[dict[str, Any]]:
        encoded_session_id = quote(session_id, safe="")
        payload = self._get(f"/runtime/sessions/{encoded_session_id}/events")
        if not isinstance(payload, list):
            raise RuntimeObserverClientError("runtime events response is not a list")
        return [dict(item) for item in payload if isinstance(item, dict)]

    def response_instructions(self, session_id: str) -> list[dict[str, Any]]:
        encoded_session_id = quote(session_id, safe="")
        payload = self._get(
            f"/runtime/sessions/{encoded_session_id}/response-instructions"
        )
        if not isinstance(payload, list):
            raise RuntimeObserverClientError(
                "runtime response instructions response is not a list"
            )
        return [dict(item) for item in payload if isinstance(item, dict)]

    def _post(self, path: str, *, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        return self._request_json("post", path, payload=payload)

    def _get(self, path: str) -> Any:
        return self._request_json("get", path)

    def _request_json(
        self,
        method_name: str,
        path: str,
        *,
        payload: dict[str, Any] | None = None,
    ) -> Any:
        session = self._session or _requests_module()
        method = getattr(session, method_name)
        try:
            request_kwargs = {"timeout": self.timeout_seconds}
            if payload is not None:
                request_kwargs["json"] = payload
            response = method(f"{self.base_url}{path}", **request_kwargs)
            response.raise_for_status()
            return response.json()
        except Exception as exc:
            raise RuntimeObserverClientError(
                f"runtime control API request failed: {method_name.upper()} {path}: {exc}"
            ) from exc


def _requests_module() -> Any:
    try:
        import requests
    except ModuleNotFoundError as exc:
        raise RuntimeObserverClientError(
            "requests is not installed; cannot call runtime control API"
        ) from exc
    return requests


def _host_for_url(host: str) -> str:
    if ":" in host and not host.startswith("["):
        return f"[{host}]"
    return host
