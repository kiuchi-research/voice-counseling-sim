from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from counseling_voice_demo.runtime.controller import ConversationRuntime
from counseling_voice_demo.runtime.models import AudioDeliveryMode, RuntimeConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the v0.3 fake conversation runtime.")
    parser.add_argument("--fake", action="store_true", help="Run with fake LLM/TTS/STT clients.")
    parser.add_argument("--turns", type=int, default=2, help="Number of fake turns to run.")
    parser.add_argument(
        "--audio-delivery-mode",
        choices=[mode.value for mode in AudioDeliveryMode],
        default=AudioDeliveryMode.ACCELERATED.value,
    )
    parser.add_argument("--sessions-dir", default="results/runtime_sessions")
    return parser.parse_args()


async def _main() -> None:
    args = parse_args()
    if not args.fake:
        raise SystemExit("Only --fake is implemented in the initial runtime skeleton.")
    config = RuntimeConfig(
        max_turns=args.turns,
        audio_delivery_mode=AudioDeliveryMode(args.audio_delivery_mode),
    )
    runtime = ConversationRuntime(config=config, sessions_dir=Path(args.sessions_dir))
    turns = await runtime.run()
    print(f"session_id={config.session_id}")
    print(f"phase={runtime.status.phase.value}")
    for turn in turns:
        print(
            f"turn={turn.turn_id} speaker={turn.speaker} "
            f"generated={turn.generated_text} stt_final={turn.stt_final_transcript}"
        )


def main() -> None:
    asyncio.run(_main())


if __name__ == "__main__":
    main()
