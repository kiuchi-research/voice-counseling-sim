from __future__ import annotations

import json
import wave
from pathlib import Path

from counseling_voice_demo.replay_engine import ReplaySession, load_replay_session


def test_load_replay_session_from_public_transcript_and_audio_files(tmp_path) -> None:
    session_dir = tmp_path / "session_public"
    public_dir = session_dir / "public"
    audio_dir = public_dir / "audio"
    audio_dir.mkdir(parents=True)
    (audio_dir / "turn_0002_rev01_client.mp3").write_bytes(b"audio2")
    (audio_dir / "turn_0001_rev01_counselor.mp3").write_bytes(b"audio1")
    _write_jsonl(
        public_dir / "transcript_public.jsonl",
        [
            {"turn_id": 2, "revision_id": 1, "speaker_role": "client", "speaker_name": "高橋", "text": "はい。"},
            {"turn_id": 1, "revision_id": 1, "speaker_role": "counselor", "speaker_name": "佐伯", "text": "こんにちは。"},
        ],
    )

    replay = load_replay_session(session_dir)

    assert replay == ReplaySession(
        session_dir=session_dir,
        mode_label="Replay mode",
        turns=(
            replay.turns[0],
            replay.turns[1],
        ),
    )
    assert [turn.turn_id for turn in replay.turns] == [1, 2]
    assert [turn.audio_path.name for turn in replay.turns] == [
        "turn_0001_rev01_counselor.mp3",
        "turn_0002_rev01_client.mp3",
    ]
    assert replay.api_calls_forbidden is True


def test_load_replay_session_from_internal_transcript_audio_path_without_api_clients(tmp_path) -> None:
    session_dir = tmp_path / "session_internal"
    internal_dir = session_dir / "internal"
    audio_dir = internal_dir / "audio_all_revisions"
    audio_dir.mkdir(parents=True)
    audio_path = audio_dir / "turn_0001_rev01_counselor.mp3"
    audio_path.write_bytes(b"audio")
    _write_jsonl(
        internal_dir / "transcript_full.jsonl",
        [
            {
                "turn_id": 1,
                "speaker_role": "counselor",
                "speaker_name": "佐伯",
                "status": "played",
                "active_revision_id": 1,
                "revisions": [
                    {
                        "revision_id": 1,
                        "active": True,
                        "canonical_text": "こんにちは。",
                        "audio_path": str(audio_path),
                    }
                ],
            }
        ],
    )

    replay = load_replay_session(session_dir)

    assert len(replay.turns) == 1
    assert replay.turns[0].text == "こんにちは。"
    assert replay.turns[0].audio_path == audio_path
    assert replay.ui_mode_text == "Replay mode: 既存セッション再生"


def test_load_replay_session_from_runtime_transcripts_with_human_audio_and_session_audio(
    tmp_path: Path,
) -> None:
    session_dir = tmp_path / "session_runtime_replay"
    transcripts_dir = session_dir / "internal" / "transcripts"
    audio_dir = session_dir / "internal" / "audio"
    transcripts_dir.mkdir(parents=True)
    audio_dir.mkdir(parents=True)
    human_audio = audio_dir / "human_counselor_turn_0001.wav"
    client_audio = audio_dir / "turn_0002_client.wav"
    session_audio = audio_dir / "session_realtime.wav"
    _write_silent_wav(human_audio, duration_seconds=1.0)
    _write_silent_wav(client_audio, duration_seconds=1.0)
    _write_silent_wav(session_audio, duration_seconds=2.5)
    _write_jsonl(
        transcripts_dir / "session_runtime_replay.jsonl",
        [
            {
                "session_id": "session_runtime_replay",
                "turn_id": 1,
                "speaker": "counselor",
                "speaker_id": "counselor",
                "transcript_type": "human_final",
                "text": "人間カウンセラーの音声発話です。",
                "audio_log_path": "internal/audio/human_counselor_turn_0001.wav",
                "metadata": {
                    "speaker_display_name": "人間カウンセラー",
                    "role": "counselor",
                    "source_audio_ref": "internal/audio/human_counselor_turn_0001.wav",
                },
            },
            {
                "session_id": "session_runtime_replay",
                "turn_id": 2,
                "speaker": "client",
                "speaker_id": "client",
                "transcript_type": "generated_final",
                "text": "AIクライアントの返答です。",
                "metadata": {"speaker_display_name": "クライアント"},
            },
        ],
    )
    _write_jsonl(
        audio_dir / "session_timeline.jsonl",
        [
            {
                "session_id": "session_runtime_replay",
                "turn_id": 1,
                "speaker": "counselor",
                "start_seconds": 0.25,
                "end_seconds": 1.25,
                "audio_path": "internal/audio/session_realtime.wav",
            },
            {
                "session_id": "session_runtime_replay",
                "turn_id": 2,
                "speaker": "client",
                "start_seconds": 1.5,
                "end_seconds": 2.5,
                "audio_path": "internal/audio/session_realtime.wav",
            },
        ],
    )

    replay = load_replay_session(session_dir)

    assert replay.session_audio_path == session_audio
    assert replay.api_calls_forbidden is True
    assert [turn.turn_id for turn in replay.turns] == [1, 2]
    assert [turn.revision_id for turn in replay.turns] == [1, 1]
    assert [turn.transcript_type for turn in replay.turns] == [
        "human_final",
        "generated_final",
    ]
    assert [turn.speaker_role for turn in replay.turns] == ["counselor", "client"]
    assert [turn.speaker_name for turn in replay.turns] == [
        "人間カウンセラー",
        "クライアント",
    ]
    assert [turn.text for turn in replay.turns] == [
        "人間カウンセラーの音声発話です。",
        "AIクライアントの返答です。",
    ]
    assert [turn.audio_path for turn in replay.turns] == [human_audio, client_audio]
    assert replay.turns[0].audio_log_path == human_audio
    assert [turn.cumulative_audio_seconds for turn in replay.turns] == [0.25, 1.5]
    assert [turn.audio_duration_seconds for turn in replay.turns] == [1.0, 1.0]


def _write_jsonl(path, records) -> None:
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )


def _write_silent_wav(path: Path, *, duration_seconds: float) -> None:
    sample_rate = 8000
    frame_count = int(sample_rate * duration_seconds)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(b"\x00\x00" * frame_count)
