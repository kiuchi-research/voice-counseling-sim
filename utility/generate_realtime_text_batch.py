#!/usr/bin/env python3
"""Generate text-only Realtime runtime sessions for turn-taking review."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


logging.getLogger("streamlit").setLevel(logging.ERROR)

from app import streamlit_app as ui_defaults  # noqa: E402
from counseling_voice_demo.content_loader import list_profiles  # noqa: E402
from counseling_voice_demo.runtime.config import load_runtime_config  # noqa: E402
from counseling_voice_demo.runtime.control_api import apply_runtime_start_options  # noqa: E402
from counseling_voice_demo.runtime.factory import build_openai_runtime  # noqa: E402
from counseling_voice_demo.runtime.models import TurnRuntimeState  # noqa: E402
from dotenv import load_dotenv  # noqa: E402


DEFAULT_BATCH_ROOT = ROOT_DIR / "results" / "runtime_text_batches"
FIXED_TWO_CLIENT_SEQUENCE = list(
    ui_defaults.DEFAULT_RUNTIME_TWO_CLIENT_SPEAKER_SEQUENCE
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run multiple two-client Realtime API sessions and save transcripts only."
        ),
    )
    parser.add_argument("--sessions", type=int, default=3)
    parser.add_argument(
        "--turns",
        type=int,
        default=30,
        help="Total transcript rows per session, including the scripted initial turn.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT_DIR / "config" / "runtime_config.yaml",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_BATCH_ROOT)
    parser.add_argument("--batch-id", default="")
    parser.add_argument("--progress-interval-seconds", type=float, default=5.0)
    return parser.parse_args(argv)


def build_two_client_start_options(total_turns: int) -> dict[str, Any]:
    if total_turns < 2:
        raise ValueError("--turns must be at least 2")

    two_client_defaults = ui_defaults.two_client_attachment_defaults()
    counselor_profile = list_profiles("counselor")[0]
    common_prompt = two_client_defaults["common_prompt"]
    client_a_prompt = ui_defaults._join_two_client_default_parts(
        ui_defaults._participant_common_prompt(
            common_prompt,
            ui_defaults.TWO_CLIENT_A_PROMPT_INTRO,
        ),
        two_client_defaults["client_a_prompt"],
    )
    client_b_prompt = ui_defaults._join_two_client_default_parts(
        ui_defaults._participant_common_prompt(
            common_prompt,
            ui_defaults.TWO_CLIENT_B_PROMPT_INTRO,
        ),
        two_client_defaults["client_b_prompt"],
    )

    return {
        "realtime_api_centered_mode": True,
        "stop_condition": "turns",
        "max_turns": total_turns - 1,
        "closing_start_elapsed_seconds": 9999.0,
        "force_stop_after_closing_turns": (
            ui_defaults.DEFAULT_RUNTIME_TWO_CLIENT_FORCE_STOP_AFTER_CLOSING_TURNS
        ),
        "realtime_model": ui_defaults.DEFAULT_RUNTIME_REALTIME_MODEL,
        "timing_llm_model": ui_defaults.DEFAULT_RUNTIME_TIMING_LLM_MODEL,
        "timing_llm_reasoning_effort": (
            ui_defaults.DEFAULT_RUNTIME_TIMING_REASONING_EFFORT
        ),
        "summary_llm_model": ui_defaults.DEFAULT_RUNTIME_SUMMARY_LLM_MODEL,
        "summary_llm_reasoning_effort": (
            ui_defaults.DEFAULT_RUNTIME_SUMMARY_REASONING_EFFORT
        ),
        "speaker_selection_policy": "turn_boundary_timing",
        "fixed_speaker_sequence": FIXED_TWO_CLIENT_SEQUENCE,
        "shared_case": two_client_defaults["common_profile"],
        "participants": {
            "counselor": {
                "role": "counselor",
                "display_name": ui_defaults.DEFAULT_RUNTIME_COUNSELOR_DISPLAY_NAME,
                "prompt_source": ui_defaults.DEFAULT_RUNTIME_COUNSELOR_PROMPT,
                "public_profile_source": counselor_profile.public_profile,
            },
            "client_a": {
                "role": "client",
                "display_name": ui_defaults.DEFAULT_RUNTIME_CLIENT_A_DISPLAY_NAME,
                "prompt_source": client_a_prompt,
                "private_profile_source": two_client_defaults["client_a_profile"],
                "initial_transcript": two_client_defaults[
                    "client_a_initial_transcript"
                ],
                "voice": ui_defaults.DEFAULT_RUNTIME_CLIENT_A_TTS_VOICE,
                "realtime_output_speed": (
                    ui_defaults.DEFAULT_RUNTIME_CLIENT_A_OUTPUT_SPEED
                ),
            },
            "client_b": {
                "role": "client",
                "display_name": ui_defaults.DEFAULT_RUNTIME_CLIENT_B_DISPLAY_NAME,
                "prompt_source": client_b_prompt,
                "private_profile_source": two_client_defaults["client_b_profile"],
                "initial_transcript": two_client_defaults[
                    "client_b_initial_transcript"
                ],
                "voice": ui_defaults.DEFAULT_RUNTIME_CLIENT_B_TTS_VOICE,
                "realtime_output_speed": (
                    ui_defaults.DEFAULT_RUNTIME_CLIENT_B_OUTPUT_SPEED
                ),
            },
        },
        "speaker_gains": {
            "counselor": ui_defaults.DEFAULT_RUNTIME_COUNSELOR_AUDIO_GAIN,
            "client_a": ui_defaults.DEFAULT_RUNTIME_CLIENT_AUDIO_GAIN,
            "client_b": ui_defaults.DEFAULT_RUNTIME_CLIENT_AUDIO_GAIN,
        },
    }


async def run_one_session(
    *,
    run_index: int,
    args: argparse.Namespace,
    output_dir: Path,
    start_options: dict[str, Any],
) -> dict[str, Any]:
    settings = load_runtime_config(args.config)
    active_settings = apply_runtime_start_options(settings, start_options)
    runtime = build_openai_runtime(active_settings)
    session_id = runtime.config.session_id

    print(
        f"[{run_index}/{args.sessions}] start {session_id} "
        f"(target transcript rows={args.turns})",
        flush=True,
    )
    started = time.monotonic()
    task = asyncio.create_task(runtime.run())
    last_completed = -1
    while not task.done():
        status = runtime.status
        if status.completed_turns != last_completed:
            print(
                f"[{run_index}/{args.sessions}] {session_id}: "
                f"completed={status.completed_turns} "
                f"phase={status.phase.value} "
                f"current={status.current_turn_id}:{status.current_speaker}",
                flush=True,
            )
            last_completed = status.completed_turns
        await asyncio.sleep(max(args.progress_interval_seconds, 0.5))

    try:
        turns = await task
    except Exception:
        turns = list(runtime.turns)
        save_session_outputs(
            output_dir=output_dir,
            session_id=session_id,
            turns=turns,
            target_turns=args.turns,
            elapsed_seconds=time.monotonic() - started,
            error=True,
        )
        raise

    elapsed_seconds = time.monotonic() - started
    print(
        f"[{run_index}/{args.sessions}] done {session_id}: "
        f"{len(turns)} rows in {elapsed_seconds:.1f}s",
        flush=True,
    )
    return save_session_outputs(
        output_dir=output_dir,
        session_id=session_id,
        turns=turns,
        target_turns=args.turns,
        elapsed_seconds=elapsed_seconds,
        error=False,
    )


def turn_text(turn: TurnRuntimeState) -> str:
    return (
        turn.generated_text
        or turn.stt_final_transcript
        or turn.provisional_input_transcript
        or turn.input_transcript
        or ""
    ).strip()


def display_name_for_turn(turn: TurnRuntimeState) -> str:
    if turn.speaker == "counselor":
        return ui_defaults.DEFAULT_RUNTIME_COUNSELOR_DISPLAY_NAME
    if turn.speaker == "client_a":
        return ui_defaults.DEFAULT_RUNTIME_CLIENT_A_DISPLAY_NAME
    if turn.speaker == "client_b":
        return ui_defaults.DEFAULT_RUNTIME_CLIENT_B_DISPLAY_NAME
    return turn.speaker


def save_session_outputs(
    *,
    output_dir: Path,
    session_id: str,
    turns: list[TurnRuntimeState],
    target_turns: int,
    elapsed_seconds: float,
    error: bool,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [turn_to_record(turn) for turn in turns[:target_turns]]
    sequence = [row["speaker"] for row in rows]
    stats = analyze_sequence(sequence)
    summary = {
        "session_id": session_id,
        "rows": len(rows),
        "raw_rows": len(turns),
        "target_rows": target_turns,
        "elapsed_seconds": round(elapsed_seconds, 3),
        "error": error,
        "speaker_counts": dict(Counter(sequence)),
        **stats,
    }

    jsonl_path = output_dir / f"{session_id}.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    md_path = output_dir / f"{session_id}.md"
    md_path.write_text(render_session_markdown(summary, rows), encoding="utf-8")
    summary["jsonl_path"] = str(jsonl_path)
    summary["markdown_path"] = str(md_path)
    return summary


def turn_to_record(turn: TurnRuntimeState) -> dict[str, Any]:
    return {
        "turn_id": turn.turn_id,
        "speaker": turn.speaker,
        "display_name": display_name_for_turn(turn),
        "recipient_ids": list(turn.recipient_ids),
        "text": turn_text(turn),
    }


def analyze_sequence(sequence: list[str]) -> dict[str, Any]:
    max_same_speaker_run = 0
    max_client_run = 0
    current_same = 0
    current_client = 0
    previous = None
    repeated_windows: dict[str, int] = {}

    for speaker in sequence:
        current_same = current_same + 1 if speaker == previous else 1
        max_same_speaker_run = max(max_same_speaker_run, current_same)
        if speaker.startswith("client_"):
            current_client += 1
        else:
            current_client = 0
        max_client_run = max(max_client_run, current_client)
        previous = speaker

    for window_size in (3, 4):
        windows = Counter(
            "->".join(sequence[index : index + window_size])
            for index in range(0, max(len(sequence) - window_size + 1, 0))
        )
        for pattern, count in windows.items():
            if count >= 2:
                repeated_windows[pattern] = count

    return {
        "sequence": " -> ".join(sequence),
        "max_same_speaker_run": max_same_speaker_run,
        "max_client_run": max_client_run,
        "repeated_windows": repeated_windows,
    }


def render_session_markdown(
    summary: dict[str, Any],
    rows: list[dict[str, Any]],
) -> str:
    lines = [
        f"# {summary['session_id']}",
        "",
        f"- rows: {summary['rows']} / target {summary['target_rows']}",
        f"- elapsed_seconds: {summary['elapsed_seconds']}",
        f"- speaker_counts: {json.dumps(summary['speaker_counts'], ensure_ascii=False)}",
        f"- max_client_run: {summary['max_client_run']}",
        f"- max_same_speaker_run: {summary['max_same_speaker_run']}",
        f"- repeated_windows: {json.dumps(summary['repeated_windows'], ensure_ascii=False)}",
        "",
        "## Sequence",
        "",
        summary["sequence"],
        "",
        "## Transcript",
        "",
    ]
    for row in rows:
        text = row["text"].replace("\n", " ").strip()
        lines.append(
            f"{row['turn_id']:02d}. {row['display_name']} "
            f"({row['speaker']}): {text}"
        )
    lines.append("")
    return "\n".join(lines)


def write_batch_summary(
    *,
    output_dir: Path,
    batch_id: str,
    args: argparse.Namespace,
    summaries: list[dict[str, Any]],
) -> None:
    manifest = {
        "batch_id": batch_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "sessions": args.sessions,
        "turns": args.turns,
        "config": str(args.config),
        "summaries": summaries,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    lines = [
        f"# {batch_id}",
        "",
        f"- sessions: {args.sessions}",
        f"- turns: {args.turns} including the scripted initial turn",
        f"- config: {args.config}",
        "",
    ]
    for summary in summaries:
        lines.extend(
            [
                f"## {summary['session_id']}",
                "",
                f"- rows: {summary['rows']}",
                f"- elapsed_seconds: {summary['elapsed_seconds']}",
                f"- speaker_counts: {json.dumps(summary['speaker_counts'], ensure_ascii=False)}",
                f"- max_client_run: {summary['max_client_run']}",
                f"- max_same_speaker_run: {summary['max_same_speaker_run']}",
                f"- repeated_windows: {json.dumps(summary['repeated_windows'], ensure_ascii=False)}",
                "",
                summary["sequence"],
                "",
                f"- transcript: {summary['markdown_path']}",
                "",
            ]
        )
    (output_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


async def async_main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.sessions <= 0:
        raise ValueError("--sessions must be positive")
    load_dotenv(ROOT_DIR / ".env", override=False)
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set")

    batch_id = args.batch_id.strip() or datetime.now().strftime(
        "realtime_text_%Y%m%d_%H%M%S"
    )
    output_dir = (args.output_dir / batch_id).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    start_options = build_two_client_start_options(args.turns)

    summaries: list[dict[str, Any]] = []
    try:
        for index in range(1, args.sessions + 1):
            summary = await run_one_session(
                run_index=index,
                args=args,
                output_dir=output_dir,
                start_options=start_options,
            )
            summaries.append(summary)
    finally:
        write_batch_summary(
            output_dir=output_dir,
            batch_id=batch_id,
            args=args,
            summaries=summaries,
        )
        print(f"output_dir={output_dir}", flush=True)
    return 0


def main() -> None:
    try:
        raise SystemExit(asyncio.run(async_main()))
    except KeyboardInterrupt:
        raise SystemExit(130)


if __name__ == "__main__":
    main()
