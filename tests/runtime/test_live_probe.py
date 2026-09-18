from __future__ import annotations

import asyncio
import wave
from contextlib import asynccontextmanager
from pathlib import Path

import pytest

from counseling_voice_demo.runtime.live_probe import (
    MissingOpenAIAPIKeyError,
    run_probe,
)


class _FakeResponses:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def stream(self, **kwargs):
        self.calls.append(kwargs)
        return _FakeLLMStream(
            [
                {"type": "response.output_text.delta", "delta": "はい、"},
                {"type": "response.output_text.delta", "delta": "短く返します。"},
                {"type": "response.completed"},
            ]
        )


class _FakeLLMStream:
    def __init__(self, events: list[dict]) -> None:
        self._events = events

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback) -> None:
        return None

    def __aiter__(self):
        return self._iter_events()

    async def _iter_events(self):
        for event in self._events:
            yield event


class _FakeTTSResponse:
    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = chunks

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback) -> None:
        return None

    async def aiter_bytes(self):
        for chunk in self._chunks:
            yield chunk


class _FakeStreamingResponse:
    def __init__(self, chunks: list[bytes]) -> None:
        self.calls: list[dict] = []
        self._chunks = chunks

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        return _FakeTTSResponse(self._chunks)


class _FakeSpeech:
    def __init__(self, chunks: list[bytes]) -> None:
        self.with_streaming_response = _FakeStreamingResponse(chunks)


class _FakeAudio:
    def __init__(self, chunks: list[bytes]) -> None:
        self.speech = _FakeSpeech(chunks)


class _FakeOpenAIClient:
    def __init__(self, *, tts_chunks: list[bytes] | None = None) -> None:
        self.responses = _FakeResponses()
        self.audio = _FakeAudio(tts_chunks or [b"\x01\x00" * 120, b"\x02\x00" * 60])


class _FakeRealtimeTransport:
    def __init__(self, events: list[dict]) -> None:
        self.events = events
        self.sent: list[dict] = []

    def send_json(self, event: dict) -> None:
        self.sent.append(event)

    def __aiter__(self):
        return self._iter_events()

    async def _iter_events(self):
        for event in self.events:
            yield event


class _StepClock:
    def __init__(self, values: list[float]) -> None:
        self._values = values
        self._index = 0

    def __call__(self) -> float:
        value = self._values[self._index]
        self._index += 1
        return value


def test_run_probe_returns_fake_llm_and_tts_results(tmp_path: Path) -> None:
    client = _FakeOpenAIClient()
    clock = _StepClock([10.0, 10.25])
    messages: list[str] = []

    result = asyncio.run(
        run_probe(
            run_llm=True,
            llm_model="gpt-5.1",
            tts_model="gpt-4o-mini-tts",
            run_tts=True,
            output_dir=tmp_path,
            client=client,
            clock=clock,
            printer=messages.append,
        )
    )

    assert result.llm is not None
    assert result.llm.model == "gpt-5.1"
    assert result.llm.deltas == ["はい、", "短く返します。"]
    assert result.llm.text == "はい、短く返します。"
    assert client.responses.calls[0]["model"] == "gpt-5.1"

    assert result.tts is not None
    assert result.tts.model == "gpt-4o-mini-tts"
    assert result.tts.chunk_count == 1
    assert result.tts.total_bytes == 360
    assert result.tts.first_chunk_seconds == pytest.approx(0.25)
    assert result.tts.wav_path == tmp_path / "live_probe_tts.wav"
    assert result.tts.wav_path.exists()
    assert client.audio.speech.with_streaming_response.calls[0]["model"] == "gpt-4o-mini-tts"
    assert any(message.startswith("llm_delta=") for message in messages)
    assert any(message.startswith("tts_chunks=") for message in messages)


def test_run_probe_can_transcribe_existing_tts_wav(tmp_path: Path) -> None:
    wav_path = tmp_path / "live_probe_tts.wav"
    with wave.open(str(wav_path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(24000)
        wav_file.writeframes(b"\x00\x00" * 2400)

    transport = _FakeRealtimeTransport(
        [
            {
                "type": "conversation.item.input_audio_transcription.delta",
                "delta": "接続",
            },
            {
                "type": "conversation.item.input_audio_transcription.completed",
                "transcript": "接続確認です。",
            },
        ]
    )

    @asynccontextmanager
    async def transport_context():
        yield transport

    messages: list[str] = []
    result = asyncio.run(
        run_probe(
            run_llm=False,
            run_tts=False,
            run_stt=True,
            stt_model="gpt-4o-transcribe",
            output_dir=tmp_path,
            api_key="test-api-key",
            stt_transport_context_factory=transport_context,
            printer=messages.append,
        )
    )

    assert result.stt is not None
    assert result.stt.model == "gpt-4o-transcribe"
    assert result.stt.partials == ["接続"]
    assert result.stt.final_text == "接続確認です。"
    assert [event["type"] for event in transport.sent] == [
        "session.update",
        "input_audio_buffer.append",
        "input_audio_buffer.commit",
    ]
    assert any(message == "stt_text=接続確認です。" for message in messages)


def test_run_probe_raises_clear_error_when_api_key_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(MissingOpenAIAPIKeyError, match="OPENAI_API_KEY"):
        asyncio.run(run_probe(run_llm=True, run_tts=False, output_dir=tmp_path))


def test_run_probe_writes_tts_pcm_as_wav(tmp_path: Path) -> None:
    pcm = b"\x00\x00\x01\x00" * 100
    result = asyncio.run(
        run_probe(
            run_llm=False,
            run_tts=True,
            output_dir=tmp_path,
            client=_FakeOpenAIClient(tts_chunks=[pcm]),
            printer=None,
        )
    )

    assert result.tts is not None
    with wave.open(str(result.tts.wav_path), "rb") as wav_file:
        assert wav_file.getframerate() == 24000
        assert wav_file.getsampwidth() == 2
        assert wav_file.getnchannels() == 1
        assert wav_file.readframes(wav_file.getnframes()) == pcm
