"""Stream a test WAV through the application's independent transcription route."""

from __future__ import annotations

import argparse
import asyncio
import json
import wave
from pathlib import Path
from tempfile import TemporaryDirectory

from counseling_voice_demo.runtime.config import load_runtime_config
from counseling_voice_demo.runtime.control_api import apply_runtime_start_options
from counseling_voice_demo.runtime.factory import build_openai_runtime
from counseling_voice_demo.runtime.models import AudioChunk, EndOfAudio


async def check_provider(
    provider: str | None,
    audio_path: Path,
    output: Path,
    recording_mode: str,
    timeout: float,
) -> None:
    with wave.open(str(audio_path), "rb") as wav:
        if (wav.getframerate(), wav.getnchannels(), wav.getsampwidth()) != (
            24000,
            1,
            2,
        ):
            raise ValueError(
                "テスト音声は 24kHz mono 16-bit PCM WAV を指定してください"
            )
        pcm = wav.readframes(wav.getnframes())
    if not pcm:
        raise ValueError("テスト音声が空です")
    settings = load_runtime_config()
    if provider:
        settings = apply_runtime_start_options(
            settings,
            {
                "ai_routes": {
                    "realtime_transcription": {"provider": provider},
                }
            },
        )
    # Only STT sends data. Other routes use injected stubs and legacy settings.
    settings.latency.realtime_api_centered_mode = False
    settings.ai.routes = {
        name: route
        for name, route in settings.ai.routes.items()
        if name in {"realtime_speech", "realtime_transcription"}
    }
    with TemporaryDirectory(prefix="stt-provider-check-") as temporary:
        settings.paths.runtime_sessions_dir = temporary
        runtime = build_openai_runtime(settings, openai_client=object())
        stt = runtime.controller.stt
        route = runtime.ai_route_manifest["realtime_transcription"]
        print(
            f"接続中: provider={route['provider']} model/deployment={route['model_ref']} mode={recording_mode}",
            flush=True,
        )
        queue = asyncio.Queue()
        observed = []

        async def observe(event):
            observed.append(event)
            if len(observed) == 1:
                print("最初の文字起こしイベントを受信しました。", flush=True)

        async def send_audio():
            # Match the microphone's paced input, including trailing silence for VAD.
            input_pcm = pcm + b"\x00" * 48000
            for index, offset in enumerate(range(0, len(input_pcm), 4800)):
                await queue.put(
                    AudioChunk(
                        session_id="stt_check",
                        turn_id=1,
                        speaker="counselor",
                        chunk_index=index,
                        pcm=input_pcm[offset : offset + 4800],
                    )
                )
                await asyncio.sleep(0.1)
            if recording_mode == "push_to_talk":
                await queue.put(
                    EndOfAudio(session_id="stt_check", turn_id=1, speaker="counselor")
                )

        sender = asyncio.create_task(send_audio())
        try:
            async with asyncio.timeout(timeout):
                partials, final = await stt.transcribe_from_queue_observed(
                    session_id="stt_check",
                    turn_id=1,
                    speaker="counselor",
                    queue=queue,
                    on_transcript=observe,
                    recording_mode=recording_mode,
                )
            if not final.text.strip():
                raise RuntimeError("確定テキストが空です")
            result = {
                **route,
                "recording_mode": recording_mode,
                "partial_events": len(partials),
                "transcript": final.text,
                "success": True,
            }
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(
                json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            print(f"文字起こし成功: {final.text}\n確認結果: {output}", flush=True)
        finally:
            sender.cancel()
            await asyncio.gather(sender, return_exceptions=True)
            await stt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", help="省略時は設定済みの接続先を使います")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument(
        "--output", type=Path, default=Path("workbench/provider_checks/stt.json")
    )
    parser.add_argument(
        "--recording-mode", choices=("push_to_talk", "vad_auto"), default="push_to_talk"
    )
    parser.add_argument("--timeout", type=float, default=45)
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    try:
        asyncio.run(
            check_provider(
                args.provider,
                args.input,
                args.output,
                args.recording_mode,
                args.timeout,
            )
        )
    except Exception as exc:
        print(f"接続確認に失敗しました: {type(exc).__name__}", flush=True)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
