from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from counseling_voice_demo.log_writer import SessionLogPaths
from counseling_voice_demo.models import Turn
from counseling_voice_demo.openai_text_client import (
    OpenAIClientError,
    ReplayModeApiCallError,
    _call_with_retries,
)


@dataclass(frozen=True)
class TtsRequest:
    turn: Turn
    log_paths: SessionLogPaths
    model: str
    voice: str
    instructions: str
    response_format: str = "mp3"


@dataclass(frozen=True)
class TtsResult:
    audio_path: Path


class OpenAITtsClient:
    def __init__(self, *, client: Any, max_retries: int = 3, replay_mode: bool = False) -> None:
        self._client = client
        self._max_retries = max_retries
        self._replay_mode = replay_mode

    def synthesize_turn(self, request: TtsRequest) -> TtsResult:
        if self._replay_mode:
            raise ReplayModeApiCallError("Replay modeではOpenAI APIを呼び出せません")

        revision = request.turn.active_revision()
        text = revision.canonical_text.strip()
        if not text:
            raise OpenAIClientError(
                f"TTS入力テキストが空です: turn_id={request.turn.turn_id} revision_id={revision.revision_id}"
            )
        audio_path = request.log_paths.audio_all_revisions_dir / _audio_filename(request.turn, request.response_format)
        payload = {
            "model": request.model,
            "voice": request.voice,
            "input": text,
            "instructions": request.instructions,
            "response_format": request.response_format,
        }

        response = _call_with_retries(
            lambda: self._client.audio.speech.with_streaming_response.create(**payload),
            max_retries=self._max_retries,
        )
        with response as streaming_response:
            streaming_response.stream_to_file(audio_path)

        revision.audio_path = str(audio_path)
        revision.audio_format = request.response_format
        return TtsResult(audio_path=audio_path)


def _audio_filename(turn: Turn, response_format: str) -> str:
    role = turn.speaker_role.value
    return f"turn_{turn.turn_id:04d}_rev{turn.active_revision_id:02d}_{role}.{response_format}"
