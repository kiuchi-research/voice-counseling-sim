"""Resolve configured AI destinations without exposing credentials to the UI or logs."""

from __future__ import annotations

import os
from collections.abc import Iterable
from dataclasses import dataclass
from functools import partial
from typing import Any

from counseling_voice_demo.runtime.config import (
    AI_ROUTE_MODEL_FIELDS,
    AISection,
    RuntimeSettings,
)
from counseling_voice_demo.runtime.realtime_transport import (
    DEFAULT_REALTIME_TRANSCRIPTION_URL,
    RealtimeAuthHeaders,
    RealtimeConnectionSpec,
    build_azure_openai_base_url,
    build_azure_realtime_session_url,
    build_azure_realtime_transcription_url,
    build_realtime_session_url,
    build_realtime_transcription_headers,
    connect_realtime,
)


REALTIME_ROUTES = {"realtime_speech", "realtime_transcription"}


class ProviderConfigurationError(ValueError):
    pass


@dataclass(frozen=True)
class ResolvedRoute:
    route_name: str
    provider_id: str
    provider_kind: str
    model_ref: str
    auth: str

    def manifest(self) -> dict[str, str]:
        return {
            "provider": self.provider_id,
            "kind": self.provider_kind,
            "model_ref": self.model_ref,
            "auth": self.auth,
        }


def effective_ai_settings(settings: RuntimeSettings) -> AISection:
    """Translate omitted routes using their existing OpenAI configuration."""
    data = (
        settings.ai.model_dump()
        if settings.ai is not None
        else {
            "providers": {
                "openai": {"kind": "openai", "api_key_env": "OPENAI_API_KEY"},
            },
            "routes": {},
        }
    )
    for name, field in AI_ROUTE_MODEL_FIELDS.items():
        if name not in data["routes"]:
            data["providers"].setdefault(
                "openai", {"kind": "openai", "api_key_env": "OPENAI_API_KEY"}
            )
            data["routes"][name] = {
                "provider": "openai",
                "targets": {"openai": {"model": getattr(settings.openai, field)}},
            }
    return AISection.model_validate(data)


class ProviderRegistry:
    def __init__(
        self,
        settings: RuntimeSettings,
        *,
        openai_api_key: str | None = None,
    ) -> None:
        self.ai = effective_ai_settings(settings)
        self._openai_api_key = openai_api_key
        self._clients: dict[str, Any] = {}

    def resolve_route(self, name: str) -> ResolvedRoute:
        if name not in self.ai.routes:
            raise ProviderConfigurationError(f"Unknown AI route: {name}")
        route = self.ai.routes[name]
        provider_id = route.provider or self.ai.default_provider
        provider = self.ai.providers[provider_id]
        target = route.targets[provider_id]
        field = "model" if provider.kind == "openai" else "deployment"
        model_ref = (getattr(target, field) or "").strip()
        if not model_ref or any(ord(char) < 32 for char in model_ref):
            raise ProviderConfigurationError(
                f"route={name} provider={provider_id}: {field} must not be empty or contain control characters"
            )
        return ResolvedRoute(name, provider_id, provider.kind, model_ref, provider.auth)

    def _api_key_for_route(self, route: ResolvedRoute) -> str:
        provider = self.ai.providers[route.provider_id]
        api_key = (
            self._openai_api_key
            if provider.kind == "openai"
            and provider.api_key_env == "OPENAI_API_KEY"
            and self._openai_api_key is not None
            else os.getenv(provider.api_key_env)
        )
        prefix = f"route={route.route_name} provider={route.provider_id}: "
        if not api_key or not api_key.strip():
            raise ProviderConfigurationError(
                prefix + f"{provider.api_key_env} is empty"
            )
        if any(char in api_key for char in ("\r", "\n")):
            raise ProviderConfigurationError(
                prefix + f"{provider.api_key_env} has invalid header characters"
            )
        return api_key.strip()

    def _azure_base_url(self, route: ResolvedRoute) -> str:
        provider = self.ai.providers[route.provider_id]
        prefix = f"route={route.route_name} provider={route.provider_id}: "
        endpoint = os.getenv(provider.endpoint_env or "")
        if not endpoint or not endpoint.strip():
            raise ProviderConfigurationError(
                prefix + f"{provider.endpoint_env} is empty"
            )
        try:
            return build_azure_openai_base_url(endpoint)
        except ValueError:
            raise ProviderConfigurationError(
                prefix
                + f"{provider.endpoint_env} must be an HTTPS resource or /openai/v1 URL without credentials, query, or fragment"
            ) from None

    def build_realtime_connection_spec(self, name: str) -> RealtimeConnectionSpec:
        if name not in REALTIME_ROUTES:
            raise ProviderConfigurationError(f"Not a Realtime route: {name}")
        route = self.resolve_route(name)
        api_key = self._api_key_for_route(route)
        if route.provider_kind == "openai":
            return RealtimeConnectionSpec(
                url=(
                    DEFAULT_REALTIME_TRANSCRIPTION_URL
                    if name == "realtime_transcription"
                    else build_realtime_session_url(route.model_ref)
                ),
                headers=build_realtime_transcription_headers(api_key),
            )
        return RealtimeConnectionSpec(
            url=(
                build_azure_realtime_transcription_url(self._azure_base_url(route))
                if name == "realtime_transcription"
                else build_azure_realtime_session_url(
                    self._azure_base_url(route), route.model_ref
                )
            ),
            headers=RealtimeAuthHeaders({"api-key": api_key}),
        )

    def _text_client_options(self, name: str) -> dict[str, str]:
        if name in REALTIME_ROUTES:
            raise ProviderConfigurationError(f"Not a text route: {name}")
        route = self.resolve_route(name)
        return {
            "api_key": self._api_key_for_route(route),
            "base_url": (
                self._azure_base_url(route)
                if route.provider_kind == "azure_openai"
                else "https://api.openai.com/v1/"
            ),
        }

    def build_async_client(self, name: str) -> Any:
        if name in REALTIME_ROUTES:
            raise ProviderConfigurationError(f"Not a text route: {name}")
        route = self.resolve_route(name)
        if route.provider_id not in self._clients:
            from openai import AsyncOpenAI

            self._clients[route.provider_id] = AsyncOpenAI(
                **self._text_client_options(name)
            )
        return self._clients[route.provider_id]

    def preflight_validate(self, route_names: Iterable[str]) -> None:
        for name in route_names:
            if name in REALTIME_ROUTES:
                self.build_realtime_connection_spec(name)
            else:
                self._text_client_options(name)

    def build_realtime_transport_factory(self, name: str, *, connect: Any = None):
        spec = self.build_realtime_connection_spec(name)
        return partial(connect_realtime, spec, connect=connect)
