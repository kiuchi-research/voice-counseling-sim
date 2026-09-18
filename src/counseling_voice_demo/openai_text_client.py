from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from counseling_voice_demo.canonical_history import build_transcript_context
from counseling_voice_demo.content_loader import Profile, Theme
from counseling_voice_demo.models import ConversationPhase, SessionState
from counseling_voice_demo.speaker_label_sanitizer import strip_leading_speaker_label


class OpenAIClientError(RuntimeError):
    pass


class ReplayModeApiCallError(OpenAIClientError):
    pass


@dataclass(frozen=True)
class TextGenerationRequest:
    speaker_profile: Profile
    theme: Theme
    session: SessionState
    conversation_phase: ConversationPhase
    cumulative_audio_seconds: float
    max_output_tokens: int = 800
    reasoning_effort: str | None = None
    safety_instruction: str | None = None
    closing_instruction: str | None = None
    farewell_instruction: str | None = None


@dataclass(frozen=True)
class TextGenerationResult:
    text: str
    raw_response: dict[str, Any]


class OpenAITextClient:
    def __init__(self, *, client: Any, max_retries: int = 3, replay_mode: bool = False) -> None:
        self._client = client
        self._max_retries = max_retries
        self._replay_mode = replay_mode

    def generate_turn(self, request: TextGenerationRequest) -> TextGenerationResult:
        if self._replay_mode:
            raise ReplayModeApiCallError("Replay modeではOpenAI APIを呼び出せません")

        payload = build_text_generation_payload(request)
        response = _call_with_retries(
            lambda: self._client.responses.create(**payload),
            max_retries=self._max_retries,
        )
        text = strip_leading_speaker_label(
            _extract_output_text(response),
            _speaker_label_candidates(request.speaker_profile),
        )
        if not text:
            raise OpenAIClientError(
                f"OpenAI APIレスポンスに生成テキストが含まれていません: {_empty_output_reason(response)}"
            )
        return TextGenerationResult(text=text, raw_response=_model_dump(response))


def build_text_generation_payload(request: TextGenerationRequest) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": request.speaker_profile.text_model,
        "input": _build_text_input(request),
        "max_output_tokens": request.max_output_tokens,
        "store": False,
    }
    if supports_temperature_parameter(request.speaker_profile.text_model):
        payload["temperature"] = request.speaker_profile.temperature
    if request.reasoning_effort:
        payload["reasoning"] = {"effort": request.reasoning_effort}
    return payload


def _build_text_input(request: TextGenerationRequest) -> list[dict[str, Any]]:
    system_text = "\n".join(
        item
        for item in [
            request.speaker_profile.prompt,
            "日本語のみで、音声で聞き取りやすい短い発話を1文程度で返してください。",
            "診断、服薬、法的判断、就業判定を断定しないでください。",
        ]
        if item
    )

    user_parts = [
        f"話者: {request.speaker_profile.display_name} ({request.speaker_profile.role})",
        f"テーマ: {request.theme.display_name}",
        f"テーマ概要: {request.theme.body}",
        f"セッション状態: {request.session.status.value}",
        f"会話フェーズ: {request.conversation_phase.value}",
        f"累積音声再生秒数: {request.cumulative_audio_seconds}",
        build_transcript_context(request.session),
    ]
    if request.speaker_profile.public_profile:
        profile_label = (
            "クライアントプロフィール"
            if request.speaker_profile.role == "client"
            else "カウンセラープロフィール"
        )
        user_parts.append(f"{profile_label}: {request.speaker_profile.public_profile}")
    for optional_instruction in [
        request.safety_instruction,
        request.closing_instruction,
        request.farewell_instruction,
    ]:
        if optional_instruction:
            user_parts.append(optional_instruction)

    return [
        {
            "role": "system",
            "content": [{"type": "input_text", "text": system_text}],
        },
        {
            "role": "user",
            "content": [{"type": "input_text", "text": "\n".join(user_parts)}],
        },
    ]


def _speaker_label_candidates(profile: Profile) -> tuple[str, ...]:
    labels = [profile.display_name, profile.profile_id]
    if profile.role == "counselor":
        labels.append("カウンセラー")
    elif profile.role == "client":
        labels.append("クライアント")
    return tuple(label for label in labels if label)


def _call_with_retries(call, *, max_retries: int):
    last_error: Exception | None = None
    for _ in range(max_retries):
        try:
            return call()
        except Exception as exc:  # pragma: no cover - exact SDK exception classes vary by version.
            last_error = exc
    reason = summarize_exception(last_error)
    raise OpenAIClientError(
        f"OpenAI API呼び出しに{max_retries}回失敗しました: {reason}"
    ) from last_error


def supports_temperature_parameter(model: str) -> bool:
    normalized = model.lower()
    return not normalized.startswith("gpt-5")


def summarize_exception(exc: Exception | None) -> str:
    if exc is None:
        return "原因不明"
    message = str(exc).strip()
    if not message:
        message = exc.__class__.__name__
    return message


def _extract_output_text(response: Any) -> str:
    output_text = getattr(response, "output_text", None)
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()

    raw = _model_dump(response)
    raw_output_text = raw.get("output_text")
    if isinstance(raw_output_text, str) and raw_output_text.strip():
        return raw_output_text.strip()
    return "\n".join(_extract_output_text_parts(raw)).strip()


def _extract_output_text_parts(value: Any) -> list[str]:
    if isinstance(value, dict):
        if isinstance(value.get("text"), str) and value.get("type") in {
            "output_text",
            "text",
        }:
            return [value["text"]]
        parts: list[str] = []
        for item in value.values():
            parts.extend(_extract_output_text_parts(item))
        return parts
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            parts.extend(_extract_output_text_parts(item))
        return parts
    return []


def _empty_output_reason(response: Any) -> str:
    raw = _model_dump(response)
    details = raw.get("incomplete_details")
    if details:
        return f"incomplete_details={details}"
    status = raw.get("status")
    if status:
        return f"status={status}"
    return "output_textが空です"


def _model_dump(response: Any) -> dict[str, Any]:
    if hasattr(response, "model_dump"):
        dumped = response.model_dump()
        if isinstance(dumped, dict):
            return dumped
    return {}
