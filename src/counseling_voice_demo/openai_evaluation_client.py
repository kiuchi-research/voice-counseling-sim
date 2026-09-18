from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from counseling_voice_demo.canonical_history import build_transcript_context
from counseling_voice_demo.models import SessionState
from counseling_voice_demo.openai_text_client import ReplayModeApiCallError, _call_with_retries, _extract_output_text, _model_dump


@dataclass(frozen=True)
class EvaluationRequest:
    session: SessionState
    model: str
    public: bool = True
    internal_context: dict[str, Any] | None = None
    max_output_tokens: int = 700
    reasoning_effort: str | None = None


@dataclass(frozen=True)
class EvaluationResult:
    text: str
    raw_response: dict[str, Any]


class OpenAIEvaluationClient:
    def __init__(self, *, client: Any, max_retries: int = 3, replay_mode: bool = False) -> None:
        self._client = client
        self._max_retries = max_retries
        self._replay_mode = replay_mode

    def evaluate(self, request: EvaluationRequest) -> EvaluationResult:
        if self._replay_mode:
            raise ReplayModeApiCallError("Replay modeではOpenAI APIを呼び出せません")

        payload = {
            "model": request.model,
            "input": _build_evaluation_input(request),
            "max_output_tokens": request.max_output_tokens,
            "store": False,
        }
        if request.reasoning_effort:
            payload["reasoning"] = {"effort": request.reasoning_effort}
        response = _call_with_retries(
            lambda: self._client.responses.create(**payload),
            max_retries=self._max_retries,
        )
        return EvaluationResult(text=_extract_output_text(response), raw_response=_model_dump(response))


def _build_evaluation_input(request: EvaluationRequest) -> list[dict[str, Any]]:
    instruction = (
        "以下の逐語録を、妥当性未検証の研究デモ用参考評価として日本語で短く評価してください。"
        "診断や治療効果の断定は避けてください。"
    )
    user_parts = [build_transcript_context(request.session)]
    if not request.public and request.internal_context:
        user_parts.append("内部改善用コンテキスト:")
        user_parts.append(json.dumps(request.internal_context, ensure_ascii=False, sort_keys=True))

    return [
        {"role": "system", "content": [{"type": "input_text", "text": instruction}]},
        {"role": "user", "content": [{"type": "input_text", "text": "\n".join(user_parts)}]},
    ]
