from __future__ import annotations

import argparse
import asyncio
import os
import time
import wave
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncContextManager

from dotenv import load_dotenv

from counseling_voice_demo.runtime.models import AudioChunk, AudioDeliveryMode
from counseling_voice_demo.runtime.streaming_llm import OpenAIStreamingLLM
from counseling_voice_demo.runtime.streaming_stt import OpenAIRealtimeTranscriptionSTT
from counseling_voice_demo.runtime.streaming_tts import (
    OpenAIStreamingTTS,
    OpenAIStreamingTTSConfig,
)


DEFAULT_TEXT_MODEL = "gpt-5.1"
DEFAULT_TTS_MODEL = "gpt-4o-mini-tts"
DEFAULT_STT_MODEL = "gpt-4o-transcribe"
DEFAULT_TTS_VOICE = "coral"
DEFAULT_TEXT_REASONING_EFFORT = "low"
SAMPLE_RATE = 24000
SAMPLE_WIDTH_BITS = 16
CHANNELS = 1
STT_CHUNK_SECONDS = 0.2
LLM_PROMPT = "音声カウンセリングの接続確認です。日本語で一文だけ短く返してください。"
LLM_SYSTEM_PROMPT = "日本語で、聞き取りやすい短い一文だけ返してください。"
TTS_TEXT = "接続確認です。短い音声を生成します。"


class MissingOpenAIAPIKeyError(RuntimeError):
    pass


@dataclass(frozen=True)
class LLMProbeResult:
    model: str
    deltas: list[str]
    text: str


@dataclass(frozen=True)
class TTSProbeResult:
    model: str
    chunk_count: int
    total_bytes: int
    first_chunk_seconds: float | None
    wav_path: Path


@dataclass(frozen=True)
class STTProbeResult:
    model: str
    wav_path: Path
    partials: list[str]
    final_text: str


@dataclass(frozen=True)
class LiveProbeResult:
    llm: LLMProbeResult | None
    tts: TTSProbeResult | None
    stt: STTProbeResult | None


async def run_probe(
    *,
    run_llm: bool,
    run_tts: bool,
    run_stt: bool = False,
    output_dir: str | Path,
    client: Any | None = None,
    client_factory: Callable[[str], Any] | None = None,
    api_key: str | None = None,
    llm_model: str | None = None,
    llm_reasoning_effort: str | None = None,
    tts_model: str | None = None,
    tts_voice: str = DEFAULT_TTS_VOICE,
    stt_model: str | None = None,
    stt_wav_path: str | Path | None = None,
    stt_transport_context_factory: Callable[[], AsyncContextManager[Any]] | None = None,
    printer: Callable[[str], None] | None = print,
    clock: Callable[[], float] = time.perf_counter,
) -> LiveProbeResult:
    if not run_llm and not run_tts and not run_stt:
        raise ValueError("at least one of run_llm, run_tts, or run_stt must be enabled")

    resolved_client = None
    if run_llm or run_tts:
        resolved_client = _resolve_client(
            client=client,
            client_factory=client_factory,
            api_key=api_key,
        )
    resolved_llm_model = llm_model or _env_or_default("OPENAI_TEXT_MODEL", DEFAULT_TEXT_MODEL)
    resolved_llm_reasoning_effort = llm_reasoning_effort or _env_or_default(
        "OPENAI_TEXT_REASONING_EFFORT",
        DEFAULT_TEXT_REASONING_EFFORT,
    )
    resolved_tts_model = tts_model or _env_or_default("OPENAI_TTS_MODEL", DEFAULT_TTS_MODEL)
    resolved_stt_model = stt_model or _env_or_default("OPENAI_STT_MODEL", DEFAULT_STT_MODEL)
    output_path = Path(output_dir)

    llm_result = None
    if run_llm:
        assert resolved_client is not None
        llm_result = await _run_llm_probe(
            client=resolved_client,
            model=resolved_llm_model,
            reasoning_effort=resolved_llm_reasoning_effort,
            printer=printer,
        )

    tts_result = None
    if run_tts:
        assert resolved_client is not None
        output_path.mkdir(parents=True, exist_ok=True)
        tts_result = await _run_tts_probe(
            client=resolved_client,
            model=resolved_tts_model,
            voice=tts_voice,
            output_dir=output_path,
            printer=printer,
            clock=clock,
        )

    stt_result = None
    if run_stt:
        resolved_wav_path = (
            Path(stt_wav_path)
            if stt_wav_path is not None
            else tts_result.wav_path
            if tts_result is not None
            else output_path / "live_probe_tts.wav"
        )
        stt_result = await _run_stt_probe(
            api_key=_resolve_api_key(api_key),
            model=resolved_stt_model,
            wav_path=resolved_wav_path,
            transport_context_factory=stt_transport_context_factory,
            printer=printer,
        )

    return LiveProbeResult(llm=llm_result, tts=tts_result, stt=stt_result)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Probe real OpenAI LLM, TTS, and Realtime STT streaming for the v0.3 runtime.",
    )
    parser.add_argument("--llm", action="store_true", help="Run a short Responses API text stream probe.")
    parser.add_argument("--tts", action="store_true", help="Run a short TTS PCM stream probe.")
    parser.add_argument("--stt", action="store_true", help="Run a Realtime transcription probe from a WAV file.")
    parser.add_argument("--output-dir", required=True, help="Directory for probe artifacts.")
    parser.add_argument(
        "--llm-model",
        default=None,
        help=f"Text model override. Defaults to OPENAI_TEXT_MODEL or {DEFAULT_TEXT_MODEL}.",
    )
    parser.add_argument(
        "--llm-reasoning-effort",
        default=None,
        help=(
            "Text reasoning effort override. Defaults to OPENAI_TEXT_REASONING_EFFORT "
            f"or {DEFAULT_TEXT_REASONING_EFFORT}."
        ),
    )
    parser.add_argument(
        "--tts-model",
        default=None,
        help=f"TTS model override. Defaults to OPENAI_TTS_MODEL or {DEFAULT_TTS_MODEL}.",
    )
    parser.add_argument("--tts-voice", default=DEFAULT_TTS_VOICE, help="TTS voice name.")
    parser.add_argument(
        "--stt-model",
        default=None,
        help=f"STT model override. Defaults to OPENAI_STT_MODEL or {DEFAULT_STT_MODEL}.",
    )
    parser.add_argument(
        "--stt-wav-path",
        default=None,
        help="WAV file to transcribe. Defaults to generated --tts WAV or <output-dir>/live_probe_tts.wav.",
    )
    args = parser.parse_args(argv)
    if not args.llm and not args.tts and not args.stt:
        parser.error("at least one of --llm, --tts, or --stt is required")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    load_dotenv()
    asyncio.run(
        run_probe(
            run_llm=args.llm,
            run_tts=args.tts,
            run_stt=args.stt,
            output_dir=args.output_dir,
            llm_model=args.llm_model,
            llm_reasoning_effort=args.llm_reasoning_effort,
            tts_model=args.tts_model,
            tts_voice=args.tts_voice,
            stt_model=args.stt_model,
            stt_wav_path=args.stt_wav_path,
        )
    )


async def _run_llm_probe(
    *,
    client: Any,
    model: str,
    reasoning_effort: str | None,
    printer: Callable[[str], None] | None,
) -> LLMProbeResult:
    llm = OpenAIStreamingLLM(
        client=client,
        model=model,
        system_prompt=LLM_SYSTEM_PROMPT,
        max_output_tokens=300,
        reasoning_effort=reasoning_effort,
    )
    deltas: list[str] = []
    _print(printer, f"llm_model={model}")
    async for delta in llm.iter_text(latest_input=LLM_PROMPT):
        deltas.append(delta)
        _print(printer, f"llm_delta={delta}")

    text = "".join(deltas)
    _print(printer, f"llm_text={text}")
    return LLMProbeResult(model=model, deltas=deltas, text=text)


async def _run_tts_probe(
    *,
    client: Any,
    model: str,
    voice: str,
    output_dir: Path,
    printer: Callable[[str], None] | None,
    clock: Callable[[], float],
) -> TTSProbeResult:
    tts = OpenAIStreamingTTS(
        client=client,
        config=OpenAIStreamingTTSConfig(
            model=model,
            voice=voice,
            instructions="落ち着いた自然な声で、短く読み上げてください。",
            response_format="pcm",
        ),
    )

    chunks: list[bytes] = []
    started_at = clock()
    first_chunk_seconds: float | None = None
    async for chunk in tts.synthesize(
        session_id="live_probe",
        turn_id=1,
        speaker="counselor",
        text_chunks=[TTS_TEXT],
        sample_rate=SAMPLE_RATE,
        sample_width_bits=SAMPLE_WIDTH_BITS,
        channels=CHANNELS,
        delivery_mode=AudioDeliveryMode.ACCELERATED,
    ):
        if first_chunk_seconds is None:
            first_chunk_seconds = clock() - started_at
        chunks.append(chunk.pcm)

    wav_path = output_dir / "live_probe_tts.wav"
    _write_wav(
        wav_path,
        chunks,
        sample_rate=SAMPLE_RATE,
        sample_width_bits=SAMPLE_WIDTH_BITS,
        channels=CHANNELS,
    )

    total_bytes = sum(len(chunk) for chunk in chunks)
    first_chunk_text = "n/a" if first_chunk_seconds is None else f"{first_chunk_seconds:.3f}"
    _print(printer, f"tts_model={model}")
    _print(
        printer,
        f"tts_chunks={len(chunks)} tts_total_bytes={total_bytes} "
        f"tts_first_chunk_seconds={first_chunk_text}",
    )
    _print(printer, f"tts_wav_path={wav_path}")
    return TTSProbeResult(
        model=model,
        chunk_count=len(chunks),
        total_bytes=total_bytes,
        first_chunk_seconds=first_chunk_seconds,
        wav_path=wav_path,
    )


async def _run_stt_probe(
    *,
    api_key: str,
    model: str,
    wav_path: Path,
    transport_context_factory: Callable[[], AsyncContextManager[Any]] | None,
    printer: Callable[[str], None] | None,
) -> STTProbeResult:
    chunks = _read_wav_audio_chunks(wav_path)
    stt = OpenAIRealtimeTranscriptionSTT(
        api_key=api_key,
        model=model,
        language="ja",
        turn_detection=None,
        transport_context_factory=transport_context_factory,
    )

    _print(printer, f"stt_model={model}")
    _print(printer, f"stt_wav_path={wav_path}")
    partials, final = await stt.transcribe(
        session_id="live_probe",
        turn_id=2,
        speaker="client",
        chunks=chunks,
    )
    partial_texts = [partial.text for partial in partials]
    for partial_text in partial_texts:
        _print(printer, f"stt_partial={partial_text}")
    _print(printer, f"stt_text={final.text}")
    return STTProbeResult(
        model=model,
        wav_path=wav_path,
        partials=partial_texts,
        final_text=final.text,
    )


def _resolve_client(
    *,
    client: Any | None,
    client_factory: Callable[[str], Any] | None,
    api_key: str | None,
) -> Any:
    if client is not None:
        return client

    resolved_api_key = api_key if api_key is not None else os.getenv("OPENAI_API_KEY")
    if not resolved_api_key or not resolved_api_key.strip():
        raise MissingOpenAIAPIKeyError(
            "OPENAI_API_KEY is not set. Add it to .env or export it before running live_probe."
        )

    factory = client_factory or _build_openai_client
    return factory(resolved_api_key)


def _resolve_api_key(api_key: str | None) -> str:
    resolved_api_key = api_key if api_key is not None else os.getenv("OPENAI_API_KEY")
    if not resolved_api_key or not resolved_api_key.strip():
        raise MissingOpenAIAPIKeyError(
            "OPENAI_API_KEY is not set. Add it to .env or export it before running live_probe."
        )
    return resolved_api_key


def _env_or_default(name: str, default: str) -> str:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    return value.strip()


def _build_openai_client(api_key: str) -> Any:
    from openai import AsyncOpenAI

    return AsyncOpenAI(api_key=api_key)


def _write_wav(
    wav_path: Path,
    chunks: list[bytes],
    *,
    sample_rate: int,
    sample_width_bits: int,
    channels: int,
) -> None:
    sample_width_bytes = sample_width_bits // 8
    with wave.open(str(wav_path), "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(sample_width_bytes)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(b"".join(chunks))


def _read_wav_audio_chunks(wav_path: Path) -> list[AudioChunk]:
    with wave.open(str(wav_path), "rb") as wav_file:
        sample_rate = wav_file.getframerate()
        sample_width_bits = wav_file.getsampwidth() * 8
        channels = wav_file.getnchannels()
        pcm = wav_file.readframes(wav_file.getnframes())

    if (sample_rate, sample_width_bits, channels) != (SAMPLE_RATE, SAMPLE_WIDTH_BITS, CHANNELS):
        raise ValueError(
            "Realtime STT probe WAV must be 24kHz mono 16-bit PCM; "
            f"got {sample_rate}Hz, {channels} channel(s), {sample_width_bits}-bit"
        )

    sample_width_bytes = sample_width_bits // 8
    chunk_bytes = max(1, int(sample_rate * channels * sample_width_bytes * STT_CHUNK_SECONDS))
    return [
        AudioChunk(
            session_id="live_probe",
            turn_id=2,
            speaker="client",
            chunk_index=index,
            pcm=pcm[offset : offset + chunk_bytes],
            sample_rate=sample_rate,
            sample_width_bits=sample_width_bits,
            channels=channels,
            delivery_mode=AudioDeliveryMode.ACCELERATED,
        )
        for index, offset in enumerate(range(0, len(pcm), chunk_bytes))
        if pcm[offset : offset + chunk_bytes]
    ]


def _print(printer: Callable[[str], None] | None, message: str) -> None:
    if printer is not None:
        printer(message)


if __name__ == "__main__":
    main()
