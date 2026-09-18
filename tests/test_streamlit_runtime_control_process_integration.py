from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

from app.streamlit_app import apply_runtime_control_action
from counseling_voice_demo.runtime.observer_client import (
    RuntimeControlClient,
    RuntimeObserverClientError,
)


def test_streamlit_runtime_controls_separate_control_api_process(tmp_path: Path) -> None:
    port = _unused_local_port()
    config_path = _write_fake_runtime_config(tmp_path, port=port)
    repo_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = (
        str(repo_root)
        if not env.get("PYTHONPATH")
        else f"{repo_root}{os.pathsep}{env['PYTHONPATH']}"
    )
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "counseling_voice_demo.runtime.control_api",
            str(config_path),
        ],
        cwd=repo_root,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    client = RuntimeControlClient(
        f"http://127.0.0.1:{port}",
        timeout_seconds=1.0,
    )
    try:
        _wait_until_control_api_is_ready(client, process)

        start_updates = apply_runtime_control_action(
            "start",
            client=client,
            start_options={
                "max_turns": 100000,
                "initial_client_transcript": "別プロセス結合確認です。",
            },
        )
        assert start_updates["runtime_control_error"] == ""
        assert start_updates["runtime_control_status"]["phase"] == "running"
        assert start_updates["session_status"] == "running"
        assert start_updates["runtime_monitor_session_id"]

        pause_updates = apply_runtime_control_action("pause", client=client)
        assert pause_updates["runtime_control_status"]["phase"] == "paused"
        assert (
            pause_updates["runtime_control_status"]["pause_reason"]
            == "generation_throttle"
        )
        assert pause_updates["session_status"] == "paused"

        resume_updates = apply_runtime_control_action("resume", client=client)
        assert resume_updates["runtime_control_status"]["phase"] == "running"
        assert resume_updates["session_status"] == "running"

        stop_updates = apply_runtime_control_action(
            "stop",
            client=client,
            current_session_id=start_updates["runtime_monitor_session_id"],
        )
        assert stop_updates["runtime_control_status"]["phase"] == "stopped"
        assert stop_updates["session_status"] == "stopped"
        assert (
            stop_updates["runtime_monitor_session_id"]
            == start_updates["runtime_monitor_session_id"]
        )
    finally:
        _terminate_process(process)


def _unused_local_port() -> int:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            return int(sock.getsockname()[1])
    except PermissionError as exc:
        pytest.skip(f"local TCP sockets are unavailable in this sandbox: {exc}")


def _wait_until_control_api_is_ready(
    client: RuntimeControlClient,
    process: subprocess.Popen[str],
    *,
    timeout_seconds: float = 10.0,
) -> None:
    deadline = time.monotonic() + timeout_seconds
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        if process.poll() is not None:
            stdout, stderr = process.communicate(timeout=1)
            pytest.fail(
                "RuntimeControlAPI process exited before readiness check.\n"
                f"stdout:\n{stdout}\nstderr:\n{stderr}"
            )
        try:
            assert client.status()["phase"] in {"idle", "stopped"}
            return
        except (RuntimeObserverClientError, AssertionError) as exc:
            last_error = exc
            time.sleep(0.1)
    pytest.fail(f"RuntimeControlAPI did not become ready: {last_error}")


def _terminate_process(process: subprocess.Popen[str]) -> None:
    if process.poll() is None:
        process.terminate()
        try:
            process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.communicate(timeout=5)


def _write_fake_runtime_config(tmp_path: Path, *, port: int) -> Path:
    config_path = tmp_path / "runtime_config.yaml"
    config_path.write_text(
        f"""
runtime:
  engine: asyncio
  backend: fake
  control_api: fastapi
  control_host: 127.0.0.1
  control_port: {port}
  monitor_transport: websocket
  audio_delivery_mode: real_time
  turn_boundary_policy: explicit_commit
  max_turns: 100000
  initial_client_transcript: 別プロセス結合確認です。

audio:
  hot_path_format: pcm
  sample_rate: 24000
  sample_width_bits: 16
  channels: 1
  speaker_gains:
    counselor: 0.6
    client: 0.6
  turn_log_format: wav
  archive_format: flac
  public_export_format: mp3
  create_public_mp3: false

openai:
  llm_model: default-text-model
  tts_model: default-tts-model
  realtime_model: gpt-realtime-mini
  tts_voice: coral
  counselor_tts_voice: shimmer
  client_tts_voice: cedar
  tts_instructions: ""
  tts_response_format: pcm
  stt_model: default-stt-model
  realtime_transcription_format:
    type: audio/pcm
    rate: 24000
  realtime_output_format:
    type: audio/pcm
    rate: 24000

paths:
  sessions_dir: {tmp_path / "sessions"}
  runtime_sessions_dir: {tmp_path / "runtime_sessions"}
  replay_sessions_dir: {tmp_path / "replay_sessions"}

latency:
  llm_max_output_tokens: null
  tts_chunk_comma_min_chars: 18
  tts_chunk_soft_max_chars: 36
  tts_target_chunk_duration_ms: 40
  tts_sdk_chunk_size: null
  stt_partial_prefetch: false
  stt_partial_prefetch_min_chars: 8
  realtime_api_centered_mode: false
""".lstrip(),
        encoding="utf-8",
    )
    return config_path
