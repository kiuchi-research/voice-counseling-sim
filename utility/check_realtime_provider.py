"""Send a short synthetic prompt through the configured Realtime speech agent."""

from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
import wave

from counseling_voice_demo.runtime.config import load_runtime_config
from counseling_voice_demo.runtime.control_api import apply_runtime_start_options
from counseling_voice_demo.runtime.factory import build_openai_runtime
from counseling_voice_demo.runtime.models import AudioDeliveryMode, ParticipantConfig
from counseling_voice_demo.runtime.prompt_context import (
    PublicHistoryMessage,
    RuntimePromptContextData,
    RuntimePromptContextStore,
    build_realtime_prompt_input_formatter,
)
from counseling_voice_demo.runtime.provider_registry import ProviderRegistry


async def check_provider(
    provider: str, output: Path, timeout: float, *, history_check: bool = False
) -> None:
    settings = apply_runtime_start_options(
        load_runtime_config(),
        {
            "ai_routes": {"realtime_speech": {"provider": provider}},
            "realtime_api_centered_mode": True,
        },
    )
    route = ProviderRegistry(settings).resolve_route("realtime_speech")
    # The application factory creates the actual speech agent. Auxiliary clients
    # are injected because this check must call only the selected speech route.
    # Omitted text routes retain the legacy OpenAI configuration for those stubs.
    settings.ai.routes = {"realtime_speech": settings.ai.routes["realtime_speech"]}
    with TemporaryDirectory(prefix="realtime-provider-check-") as temporary:
        settings.paths.runtime_sessions_dir = temporary
        runtime = build_openai_runtime(settings, openai_client=object(), stt=object())
        agent = next(iter(runtime.controller.agents.values()))
        agent.config = replace(
            agent.config,
            instructions="日本語で短い接続テストの音声を返してください。",
            response_instructions="",
        )
        agent.input_formatter = None
        agent.response_input_formatter = None
        agent.reuse_transport = history_check
        snapshots = (("青", "緑", "赤"), ("白", "黒", "黄")) if history_check else ((),)
        context_store = RuntimePromptContextStore()
        if history_check:
            agent.config = replace(
                agent.config,
                instructions=(
                    "あなたは架空の色の記憶テストの進行役です。"
                    "今回渡された会話履歴からA、B、進行役の選んだ色を参照し、"
                    "『Aは何色、Bは何色、進行役は何色』の形式で実際の色を短く答えてください。"
                    "挨拶や接続テストの定型文は不要です。"
                ),
            )
            # Build the same structured input as the application, using only
            # invented participants and facts. Never send configured presets.
            participants = {
                "check_a": ParticipantConfig(
                    speaker_id="check_a", role="client", display_name="A"
                ),
                "check_b": ParticipantConfig(
                    speaker_id="check_b", role="client", display_name="B"
                ),
                agent.speaker: ParticipantConfig(
                    speaker_id=agent.speaker, role="counselor", display_name="進行役"
                ),
            }
            agent.response_input_formatter = build_realtime_prompt_input_formatter(
                participants=participants,
                shared_case="架空の色の記憶テスト。",
                runtime_context_provider=context_store,
            )
        print(
            f"接続中: provider={route.provider_id} model/deployment={route.model_ref}",
            flush=True,
        )
        pcm = bytearray()
        results = []
        try:
            async with asyncio.timeout(timeout):
                for turn_id, colors in enumerate(snapshots, 1):
                    print(f"[{turn_id}/{len(snapshots)}] 応答を確認中", flush=True)
                    prompt = "『こんにちは。接続テストに成功しました。』と一度だけ言ってください。"
                    if history_check:
                        context_store.set_context(
                            turn_id,
                            RuntimePromptContextData(
                                public_history=tuple(
                                    PublicHistoryMessage(
                                        speaker, f"私が選んだ色は{color}です。"
                                    )
                                    for speaker, color in zip(
                                        participants, colors, strict=True
                                    )
                                )
                            ),
                        )
                        prompt = "履歴にあるA、B、進行役それぞれの選んだ色を、この順で短く答えてください。"
                    records = []
                    transcript = []
                    async for event in agent.stream_audio_response(
                        session_id="provider_check",
                        turn_id=turn_id,
                        speaker=agent.speaker,
                        input_transcript=prompt,
                        sample_rate=settings.audio.sample_rate,
                        sample_width_bits=settings.audio.sample_width_bits,
                        channels=settings.audio.channels,
                        delivery_mode=AudioDeliveryMode.ACCELERATED,
                        response_instructions_observer=records.append,
                    ):
                        if event.text_delta:
                            transcript.append(event.text_delta)
                        if event.audio_chunk is not None:
                            if not pcm:
                                print("最初の音声チャンクを受信しました。", flush=True)
                            pcm.extend(event.audio_chunk.pcm)
                    results.append(
                        {
                            "turn_id": turn_id,
                            "expected_colors": colors,
                            "transcript": "".join(transcript),
                            "response_input": (
                                records[0].response_input if records else None
                            ),
                        }
                    )
        finally:
            await agent.close()
        if not pcm:
            raise RuntimeError("音声データを受信できませんでした")
        output.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(output), "wb") as wav:
            wav.setnchannels(settings.audio.channels)
            wav.setsampwidth(settings.audio.sample_width_bits // 8)
            wav.setframerate(settings.audio.sample_rate)
            wav.writeframes(pcm)
        print(f"音声受信成功: {len(pcm)} bytes; 保存先={output}", flush=True)
        if history_check:
            report_path = output.with_suffix(".json")
            report_path.write_text(
                json.dumps(results, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            print(f"送信履歴と実際の応答: {report_path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--timeout", type=float, default=45)
    parser.add_argument(
        "--history-check",
        action="store_true",
        help="架空の3人の履歴を同一接続で2回渡し、送信内容と応答を保存する",
    )
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    try:
        asyncio.run(
            check_provider(
                args.provider,
                args.output,
                args.timeout,
                history_check=args.history_check,
            )
        )
    except Exception as exc:
        # Do not print SDK/WebSocket exception representations or request headers.
        print(f"接続確認失敗: {type(exc).__name__}", flush=True)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
