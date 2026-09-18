"""Check runtime text routes with short synthetic prompts and safe result metadata."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

from counseling_voice_demo.runtime.config import load_runtime_config
from counseling_voice_demo.runtime.control_api import apply_runtime_start_options
from counseling_voice_demo.runtime.factory import build_openai_runtime
from counseling_voice_demo.runtime.prompt_director import (
    PromptDirectorRequest,
    parse_prompt_director_response,
)


async def check_provider(provider: str | None, output: Path, timeout: float) -> None:
    route_names = (
        "prompt_director",
        "turn_timing",
        "session_summary",
        "conversation_text",
    )
    options = {"realtime_api_centered_mode": False}
    if provider:
        options["ai_routes"] = {name: {"provider": provider} for name in route_names}
    settings = apply_runtime_start_options(load_runtime_config(), options)
    # Build the real application text clients; audio clients make no API calls.
    with TemporaryDirectory(prefix="text-provider-check-") as temporary:
        settings.paths.runtime_sessions_dir = temporary
        runtime = build_openai_runtime(settings, openai_client=object(), stt=object())
        agent = next(iter(runtime.controller.agents.values()))
        llms = {
            "prompt_director": runtime.prompt_director.llm,
            "turn_timing": agent.timing_llm,
            "session_summary": runtime.session_summary_llm,
            "conversation_text": agent.llm,
        }
        results = {}
        try:
            for name, llm in llms.items():
                route = runtime.ai_route_manifest[name]
                print(
                    f"接続中: {name} provider={route['provider']} model/deployment={route['model_ref']}",
                    flush=True,
                )
                started = perf_counter()
                instruction = "接続テストです。『接続成功』とだけ返してください。"
                if name == "prompt_director":
                    instruction = (
                        "接続テストです。instruction_checksは1件で、"
                        "sourceはfixed_role_constraints、start_lineとend_lineは1、"
                        "applicabilityはapplies、evidenceは『接続確認中』とし、"
                        "forceはrequired、priorityは『指定なし』、"
                        "response_excerptは『接続成功』、response_assessmentは『指定本文を出力』、"
                        "context_basisは『接続確認のみ』としてください。"
                        "response_intent、response_exampleはそれぞれ『接続成功』の"
                        "JSONを返してください。"
                    )
                async with asyncio.timeout(timeout):
                    chunks = [
                        part
                        async for part in llm.stream_text(
                            latest_input="接続を確認してください。",
                            system_prompt=instruction,
                        )
                    ]
                response = "".join(chunks)
                if not response.strip():
                    raise RuntimeError("テキストを受信できませんでした")
                if name == "prompt_director":
                    parse_prompt_director_response(
                        response,
                        request=PromptDirectorRequest(
                            turn_id=0,
                            speaker_id="connectivity_check",
                            fixed_system_prompt=instruction,
                            configured_prompt="",
                        ),
                    )
                results[name] = {
                    **route,
                    "response_chars": len(response),
                    "text_chunks": len(chunks),
                    "elapsed_seconds": round(perf_counter() - started, 3),
                    "success": True,
                }
                print(f"成功: {name} ({len(chunks)} chunks)", flush=True)
        finally:
            for client in {
                id(llm._client): llm._client for llm in llms.values()
            }.values():
                close = getattr(client, "close", None)
                if close is not None:
                    await close()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"確認結果: {output}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", help="省略すると設定済みの接続先を使います")
    parser.add_argument(
        "--output", type=Path, default=Path("workbench/provider_checks/text.json")
    )
    parser.add_argument("--timeout", type=float, default=45.0)
    args = parser.parse_args()
    try:
        asyncio.run(check_provider(args.provider, args.output, args.timeout))
    except Exception as exc:
        print(f"接続確認に失敗しました: {type(exc).__name__}", flush=True)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
