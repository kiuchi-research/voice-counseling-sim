from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from counseling_voice_demo.openai_text_client import ReplayModeApiCallError, _call_with_retries, _model_dump


@dataclass(frozen=True)
class ModerationResult:
    flagged: bool
    raw: dict[str, Any]
    should_stop_session: bool = False


class OpenAIModerationClient:
    def __init__(self, *, client: Any, max_retries: int = 3, replay_mode: bool = False) -> None:
        self._client = client
        self._max_retries = max_retries
        self._replay_mode = replay_mode

    def moderate(self, *, text: str, model: str) -> ModerationResult:
        if self._replay_mode:
            raise ReplayModeApiCallError("Replay modeではOpenAI APIを呼び出せません")

        response = _call_with_retries(
            lambda: self._client.moderations.create(input=text, model=model),
            max_retries=self._max_retries,
        )
        return ModerationResult(flagged=_is_flagged(response), raw=_model_dump(response))


def _is_flagged(response: Any) -> bool:
    results = getattr(response, "results", None)
    if results and hasattr(results[0], "flagged"):
        return bool(results[0].flagged)
    raw = _model_dump(response)
    raw_results = raw.get("results")
    if isinstance(raw_results, list) and raw_results:
        return bool(raw_results[0].get("flagged", False))
    return False
