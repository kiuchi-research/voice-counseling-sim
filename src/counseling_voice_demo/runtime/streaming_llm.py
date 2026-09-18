from __future__ import annotations

from collections.abc import AsyncIterator, Iterable, Iterator, Mapping
from typing import Any


TERMINAL_BOUNDARIES = {"。", "？", "！", "?", "!"}
RESPONSES_TEXT_DELTA_EVENT = "response.output_text.delta"
RESPONSES_COMPLETED_EVENT = "response.completed"
RESPONSES_INCOMPLETE_EVENT = "response.incomplete"
RESPONSES_FAILED_EVENT = "response.failed"
RESPONSES_ERROR_EVENT = "error"
RESPONSES_INPUT_ROLES = {"system", "developer", "user", "assistant"}


class StreamingLLMError(RuntimeError):
    pass


class TtsTextChunker:
    """Incrementally split streamed text into TTS-sized Japanese phrase chunks."""

    def __init__(self, *, comma_min_chars: int = 20, soft_max_chars: int = 40) -> None:
        self.comma_min_chars = comma_min_chars
        self.soft_max_chars = soft_max_chars
        self._buffer = ""

    def push(self, part: str) -> list[str]:
        chunks: list[str] = []
        for char in part:
            self._buffer += char
            stripped = self._buffer.strip()
            if not stripped:
                continue
            if char in TERMINAL_BOUNDARIES:
                chunks.append(stripped)
                self._buffer = ""
            elif char == "、" and len(stripped) >= self.comma_min_chars:
                chunks.append(stripped)
                self._buffer = ""
            elif len(stripped) >= self.soft_max_chars and char in {"、", " ", "\n"}:
                chunks.append(stripped)
                self._buffer = ""
            elif len(stripped) >= self.soft_max_chars:
                chunks.append(stripped)
                self._buffer = ""
        return chunks

    def flush(self) -> list[str]:
        if not self._buffer.strip():
            self._buffer = ""
            return []
        chunk = self._buffer.strip()
        self._buffer = ""
        return [chunk]


def build_responses_input(
    system_prompt: str | None,
    history: Iterable[Mapping[str, Any]] | None,
    latest_input: str,
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    if system_prompt:
        items.append(_build_input_message("system", system_prompt))
    for item in history or []:
        items.append(_build_input_message(item.get("role"), item.get("content", "")))
    items.append(_build_input_message("user", latest_input))
    return items


def extract_text_delta(event: Any) -> str | None:
    event_type = _event_value(event, "type")
    if event_type == RESPONSES_ERROR_EVENT:
        raise StreamingLLMError(_stream_error_message(event))
    if event_type == RESPONSES_INCOMPLETE_EVENT:
        raise StreamingLLMError(_response_incomplete_message(event))
    if event_type == RESPONSES_FAILED_EVENT:
        response = _event_value(event, "response")
        raise StreamingLLMError(
            f"OpenAI response failed: {_stream_error_message(response)}"
        )
    if event_type != RESPONSES_TEXT_DELTA_EVENT:
        return None

    delta = _event_value(event, "delta")
    if isinstance(delta, str):
        return delta
    return None


def iter_text_deltas(events: Iterable[Any]) -> Iterator[str]:
    for event in events:
        delta = extract_text_delta(event)
        if delta is not None:
            yield delta
        if _event_value(event, "type") == RESPONSES_COMPLETED_EVENT:
            break


class OpenAIStreamingLLM:
    def __init__(
        self,
        *,
        client: Any,
        model: str,
        system_prompt: str | None = None,
        max_output_tokens: int | None = None,
        reasoning_effort: str | None = None,
        text_format: Mapping[str, Any] | None = None,
    ) -> None:
        self._client = client
        self._model = model
        self._system_prompt = system_prompt
        self._max_output_tokens = max_output_tokens
        self._reasoning_effort = reasoning_effort
        self._text_format = dict(text_format) if text_format is not None else None

    def build_payload(
        self,
        *,
        latest_input: str,
        history: Iterable[Mapping[str, Any]] | None = None,
        system_prompt: str | None = None,
        text_format: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        active_system_prompt = (
            self._system_prompt if system_prompt is None else system_prompt
        )
        payload: dict[str, Any] = {
            "model": self._model,
            "input": build_responses_input(None, history, latest_input),
            "store": False,
        }
        if active_system_prompt:
            payload["instructions"] = active_system_prompt
        if self._max_output_tokens is not None:
            payload["max_output_tokens"] = self._max_output_tokens
        if self._reasoning_effort:
            payload["reasoning"] = {"effort": self._reasoning_effort}
        active_text_format = self._text_format if text_format is None else text_format
        if active_text_format is not None:
            payload["text"] = {"format": dict(active_text_format)}
        return payload

    async def iter_text(
        self,
        *,
        latest_input: str,
        history: Iterable[Mapping[str, Any]] | None = None,
        system_prompt: str | None = None,
        text_format: Mapping[str, Any] | None = None,
    ) -> AsyncIterator[str]:
        payload = self.build_payload(
            latest_input=latest_input,
            history=history,
            system_prompt=system_prompt,
            text_format=text_format,
        )
        stream = self._client.responses.stream(**payload)
        async for event in _aiter_stream_events(stream):
            delta = extract_text_delta(event)
            if delta is not None:
                yield delta
            if _event_value(event, "type") == RESPONSES_COMPLETED_EVENT:
                break

    def stream_text(
        self,
        *,
        latest_input: str,
        history: Iterable[Mapping[str, Any]] | None = None,
        system_prompt: str | None = None,
        text_format: Mapping[str, Any] | None = None,
    ) -> AsyncIterator[str]:
        return self.iter_text(
            latest_input=latest_input,
            history=history,
            system_prompt=system_prompt,
            text_format=text_format,
        )


def chunk_text_for_tts(
    parts: Iterable[str],
    *,
    comma_min_chars: int = 20,
    soft_max_chars: int = 40,
) -> list[str]:
    """Split streamed Japanese text into TTS-sized phrase chunks."""
    chunks: list[str] = []
    chunker = TtsTextChunker(
        comma_min_chars=comma_min_chars,
        soft_max_chars=soft_max_chars,
    )
    for part in parts:
        chunks.extend(chunker.push(part))
    chunks.extend(chunker.flush())

    return chunks


class FakeStreamingLLM:
    def __init__(self, *, response_template: str) -> None:
        self._response_template = response_template

    async def stream_text(
        self,
        *,
        latest_input: str,
        history: Iterable[Mapping[str, Any]] | None = None,
        system_prompt: str | None = None,
        text_format: Mapping[str, Any] | None = None,
        turn_id: int | None = None,
        speaker: str | None = None,
    ) -> AsyncIterator[str]:
        text = self._response_template.format(
            input_transcript=latest_input,
            latest_input=latest_input,
            turn_id="" if turn_id is None else turn_id,
            speaker="" if speaker is None else speaker,
            system_prompt="" if system_prompt is None else system_prompt,
        )
        midpoint = max(1, len(text) // 2)
        yield text[:midpoint]
        if text[midpoint:]:
            yield text[midpoint:]


def _build_input_message(role: Any, content: Any) -> dict[str, Any]:
    role_text = str(role).strip()
    if role_text not in RESPONSES_INPUT_ROLES:
        raise ValueError(f"unsupported Responses input role: {role_text}")
    return {"role": role_text, "content": content}


def _event_value(event: Any, key: str) -> Any:
    if isinstance(event, Mapping):
        return event.get(key)
    return getattr(event, key, None)


def _stream_error_message(event: Any) -> str:
    error = _event_value(event, "error")
    message = _error_value(error, "message") or _event_value(event, "message")
    code = _error_value(error, "code") or _event_value(event, "code")
    if message and code:
        return f"{message} ({code})"
    if message:
        return str(message)
    if code:
        return str(code)
    return "OpenAI streaming error"


def _response_incomplete_message(event: Any) -> str:
    response = _event_value(event, "response")
    details = _event_value(response, "incomplete_details")
    reason = _event_value(details, "reason")
    if reason:
        return f"OpenAI response incomplete: {reason}"
    return "OpenAI response incomplete"


def _error_value(error: Any, key: str) -> Any:
    if isinstance(error, Mapping):
        return error.get(key)
    return getattr(error, key, None)


async def _aiter_stream_events(stream: Any) -> AsyncIterator[Any]:
    if hasattr(stream, "__aenter__"):
        async with stream as active_stream:
            async for event in _aiter_any(active_stream):
                yield event
        return

    if hasattr(stream, "__enter__"):
        with stream as active_stream:
            async for event in _aiter_any(active_stream):
                yield event
        return

    async for event in _aiter_any(stream):
        yield event


async def _aiter_any(events: Any) -> AsyncIterator[Any]:
    if hasattr(events, "__aiter__"):
        async for event in events:
            yield event
        return

    for event in events:
        yield event
