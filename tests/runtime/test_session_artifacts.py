from __future__ import annotations

import json
import os
import wave
from pathlib import Path

import pytest

from counseling_voice_demo.runtime.session_artifacts import (
    CompactSessionAudioConfig,
    PublicTranscriptTurn,
    build_compact_session_audio,
    format_public_script,
    list_runtime_sessions,
    load_public_transcript_turns,
    session_has_human_audio_turns,
    session_realtime_audio_path,
)


def test_list_runtime_sessions_returns_directories_newest_first(tmp_path: Path) -> None:
    older = tmp_path / "session_older"
    newer = tmp_path / "session_newer"
    older.mkdir()
    newer.mkdir()
    (tmp_path / "not_a_session.txt").write_text("ignore me", encoding="utf-8")
    os.utime(older, (100, 100))
    os.utime(newer, (200, 200))

    assert list_runtime_sessions(tmp_path) == [newer, older]


def test_load_public_transcript_turns_uses_final_only_and_last_duplicate(tmp_path: Path) -> None:
    session_dir = tmp_path / "session_artifacts"
    transcripts_dir = session_dir / "internal" / "transcripts"
    audio_dir = session_dir / "internal" / "audio"
    transcripts_dir.mkdir(parents=True)
    audio_dir.mkdir(parents=True)
    counselor_audio = audio_dir / "turn_0001_counselor.wav"
    _write_silent_wav(counselor_audio, duration_seconds=1.0)
    _write_jsonl(
        transcripts_dir / "session_artifacts.jsonl",
        [
            {
                "session_id": "session_artifacts",
                "turn_id": 1,
                "speaker": "counselor",
                "transcript_type": "partial",
                "text": "partial text",
                "created_at": "2026-06-23T09:00:00+09:00",
            },
            {
                "session_id": "session_artifacts",
                "turn_id": 2,
                "speaker": "client",
                "transcript_type": "final",
                "text": "second turn",
                "created_at": "2026-06-23T09:01:00+09:00",
            },
            {
                "session_id": "session_artifacts",
                "turn_id": 1,
                "speaker": "counselor",
                "transcript_type": "final",
                "text": "old final",
                "created_at": "2026-06-23T09:00:01+09:00",
            },
            "not json",
            {
                "session_id": "session_artifacts",
                "turn_id": 1,
                "speaker": "counselor",
                "transcript_type": "final",
                "text": "last final",
                "created_at": "2026-06-23T09:00:02+09:00",
            },
        ],
    )

    turns = load_public_transcript_turns(session_dir)

    assert [turn.turn_id for turn in turns] == [1, 2]
    assert turns[0].speaker == "counselor"
    assert turns[0].text == "last final"
    assert turns[0].audio_path == counselor_audio
    assert turns[0].cumulative_audio_seconds == pytest.approx(0.0)
    assert turns[1].speaker == "client"
    assert turns[1].text == "second turn"
    assert turns[1].audio_path is None
    assert turns[1].cumulative_audio_seconds == pytest.approx(1.0)


def test_load_public_transcript_turns_prefers_generated_final_over_stt_final(
    tmp_path: Path,
) -> None:
    session_dir = tmp_path / "session_generated_public"
    transcripts_dir = session_dir / "internal" / "transcripts"
    audio_dir = session_dir / "internal" / "audio"
    transcripts_dir.mkdir(parents=True)
    audio_dir.mkdir(parents=True)
    _write_silent_wav(audio_dir / "turn_0001_client_a.wav", duration_seconds=1.0)
    _write_jsonl(
        transcripts_dir / "session_generated_public.jsonl",
        [
            {
                "session_id": "session_generated_public",
                "turn_id": 1,
                "speaker": "client_a",
                "speaker_id": "client_a",
                "transcript_type": "final",
                "text": "STTの誤認識です。",
                "metadata": {"speaker_display_name": "妻"},
            },
            {
                "session_id": "session_generated_public",
                "turn_id": 1,
                "speaker": "client_a",
                "speaker_id": "client_a",
                "transcript_type": "generated_final",
                "text": "生成元テキストです。",
                "metadata": {"speaker_display_name": "妻"},
            },
        ],
    )

    turns = load_public_transcript_turns(session_dir)

    assert len(turns) == 1
    assert turns[0].speaker == "client_a"
    assert turns[0].speaker_display_name == "妻"
    assert turns[0].text == "生成元テキストです。"
    assert format_public_script(turns, format="txt") == (
        "00:00 妻: 生成元テキストです。"
    )
    assert not session_has_human_audio_turns(turns)


def test_load_public_transcript_turns_includes_generated_interrupted(
    tmp_path: Path,
) -> None:
    session_dir = tmp_path / "session_generated_interrupted_public"
    transcripts_dir = session_dir / "internal" / "transcripts"
    audio_dir = session_dir / "internal" / "audio"
    transcripts_dir.mkdir(parents=True)
    audio_dir.mkdir(parents=True)
    _write_silent_wav(audio_dir / "turn_0001_client_b.wav", duration_seconds=1.0)
    _write_jsonl(
        transcripts_dir / "session_generated_interrupted_public.jsonl",
        [
            {
                "session_id": "session_generated_interrupted_public",
                "turn_id": 1,
                "speaker": "client_b",
                "speaker_id": "client_b",
                "transcript_type": "generated_interrupted",
                "text": "途中まで話していた内容です。",
                "metadata": {
                    "speaker_display_name": "夫",
                    "interrupted": True,
                },
            },
            {
                "session_id": "session_generated_interrupted_public",
                "turn_id": 2,
                "speaker": "counselor",
                "speaker_id": "counselor",
                "transcript_type": "human_final",
                "text": "ここで確認させてください。",
                "metadata": {"speaker_display_name": "カウンセラー"},
            },
        ],
    )

    turns = load_public_transcript_turns(session_dir)

    assert [turn.turn_id for turn in turns] == [1, 2]
    assert [turn.transcript_type for turn in turns] == [
        "generated_interrupted",
        "human_final",
    ]
    assert [turn.text for turn in turns] == [
        "途中まで話していた内容です。",
        "ここで確認させてください。",
    ]
    assert turns[0].speaker_display_name == "夫"


def test_load_public_transcript_turns_filters_invalidated_future_ai_turns(
    tmp_path: Path,
) -> None:
    session_dir = tmp_path / "session_invalidated_future_turn"
    transcripts_dir = session_dir / "internal" / "transcripts"
    events_dir = session_dir / "internal" / "events"
    transcripts_dir.mkdir(parents=True)
    events_dir.mkdir(parents=True)
    _write_jsonl(
        transcripts_dir / "session_invalidated_future_turn.jsonl",
        [
            {
                "session_id": "session_invalidated_future_turn",
                "turn_id": 2,
                "speaker": "client_b",
                "speaker_id": "client_b",
                "transcript_type": "generated_interrupted",
                "text": "割り込まれた夫の発話です。",
                "metadata": {"speaker_display_name": "夫"},
            },
            {
                "session_id": "session_invalidated_future_turn",
                "turn_id": 3,
                "speaker": "client_a",
                "speaker_id": "client_a",
                "transcript_type": "generated_final",
                "text": "割り込み前に先行生成された妻の発話です。",
                "metadata": {"speaker_display_name": "妻"},
            },
            {
                "session_id": "session_invalidated_future_turn",
                "turn_id": 3,
                "speaker": "counselor",
                "speaker_id": "counselor",
                "transcript_type": "human_final",
                "text": "奥さんにも確認させてください。",
                "metadata": {"speaker_display_name": "カウンセラー"},
            },
            {
                "session_id": "session_invalidated_future_turn",
                "turn_id": 4,
                "speaker": "client_a",
                "speaker_id": "client_a",
                "transcript_type": "generated_final",
                "text": "割り込み後の妻の応答です。",
                "metadata": {"speaker_display_name": "妻"},
            },
        ],
    )
    _write_jsonl(
        events_dir / "session_invalidated_future_turn.jsonl",
        [
            {
                "session_id": "session_invalidated_future_turn",
                "event_type": "barge_in_future_turns_invalidated",
                "turn_id": 2,
                "details": {
                    "interrupted_turn_id": 2,
                    "invalidated_turn_ids": [3],
                    "invalidated_turns": [
                        {
                            "turn_id": 3,
                            "speaker": "client_a",
                            "speaker_id": "client_a",
                        }
                    ],
                },
            }
        ],
    )

    turns = load_public_transcript_turns(session_dir)

    assert [(turn.turn_id, turn.speaker, turn.text) for turn in turns] == [
        (2, "client_b", "割り込まれた夫の発話です。"),
        (3, "counselor", "奥さんにも確認させてください。"),
        (4, "client_a", "割り込み後の妻の応答です。"),
    ]


def test_load_public_transcript_turns_includes_human_and_generated_final_on_same_timeline(
    tmp_path: Path,
) -> None:
    session_dir = tmp_path / "session_human_counselor_replay"
    transcripts_dir = session_dir / "internal" / "transcripts"
    audio_dir = session_dir / "internal" / "audio"
    transcripts_dir.mkdir(parents=True)
    audio_dir.mkdir(parents=True)
    human_audio = audio_dir / "human_counselor_turn_0001.wav"
    client_audio = audio_dir / "turn_0002_client.wav"
    session_audio = audio_dir / "session_realtime.wav"
    _write_silent_wav(human_audio, duration_seconds=1.0)
    _write_silent_wav(client_audio, duration_seconds=1.5)
    _write_silent_wav(session_audio, duration_seconds=3.0)
    _write_jsonl(
        transcripts_dir / "session_human_counselor_replay.jsonl",
        [
            {
                "session_id": "session_human_counselor_replay",
                "turn_id": 1,
                "speaker": "counselor",
                "speaker_id": "counselor",
                "transcript_type": "partial",
                "text": "まだ公開しない途中テキスト",
            },
            {
                "session_id": "session_human_counselor_replay",
                "turn_id": 1,
                "speaker": "counselor",
                "speaker_id": "counselor",
                "transcript_type": "human_final",
                "text": "今日はどのような相談でしょうか。",
                "audio_log_path": "internal/audio/human_counselor_turn_0001.wav",
                "metadata": {
                    "speaker_display_name": "人間カウンセラー",
                    "role": "counselor",
                    "source_audio_ref": "internal/audio/human_counselor_turn_0001.wav",
                    "TimingDecision": "DO_NOT_EXPORT_TIMING_DECISION",
                    "system_prompt": "SECRET_PROMPT",
                },
            },
            {
                "session_id": "session_human_counselor_replay",
                "turn_id": 2,
                "speaker": "client",
                "speaker_id": "client",
                "transcript_type": "generated_final",
                "text": "娘との関係で悩んでいます。",
                "metadata": {
                    "speaker_display_name": "クライアント",
                    "system_prompt": "SECRET_CLIENT_PROMPT",
                },
            },
        ],
    )
    _write_jsonl(
        audio_dir / "session_timeline.jsonl",
        [
            {
                "session_id": "session_human_counselor_replay",
                "turn_id": 1,
                "speaker": "counselor",
                "start_seconds": 0.0,
                "end_seconds": 1.0,
                "audio_path": "internal/audio/session_realtime.wav",
            },
            {
                "session_id": "session_human_counselor_replay",
                "turn_id": 2,
                "speaker": "client",
                "start_seconds": 1.25,
                "end_seconds": 2.75,
                "audio_path": "internal/audio/session_realtime.wav",
            },
        ],
    )

    turns = load_public_transcript_turns(session_dir)
    script = format_public_script(turns, format="txt")

    assert [turn.turn_id for turn in turns] == [1, 2]
    assert [turn.transcript_type for turn in turns] == [
        "human_final",
        "generated_final",
    ]
    assert [turn.text for turn in turns] == [
        "今日はどのような相談でしょうか。",
        "娘との関係で悩んでいます。",
    ]
    assert turns[0].audio_path == human_audio
    assert turns[0].audio_log_path == human_audio
    assert turns[1].audio_path == client_audio
    assert [turn.cumulative_audio_seconds for turn in turns] == [0.0, 1.25]
    assert [turn.audio_duration_seconds for turn in turns] == [1.0, 1.5]
    assert script == (
        "00:00 人間カウンセラー: 今日はどのような相談でしょうか。\n"
        "00:01 クライアント: 娘との関係で悩んでいます。"
    )
    assert "DO_NOT_EXPORT_TIMING_DECISION" not in script
    assert "SECRET_PROMPT" not in script
    assert "SECRET_CLIENT_PROMPT" not in script
    assert session_realtime_audio_path(session_dir) == session_audio


def test_build_compact_session_audio_compresses_human_wait_without_trimming_turns(
    tmp_path: Path,
) -> None:
    session_dir = tmp_path / "session_compact_boundary_audio"
    transcripts_dir = session_dir / "internal" / "transcripts"
    audio_dir = session_dir / "internal" / "audio"
    transcripts_dir.mkdir(parents=True)
    audio_dir.mkdir(parents=True)
    first_human_audio = audio_dir / "human_counselor_turn_0001.wav"
    client_a_audio = audio_dir / "turn_0002_client_a.wav"
    client_b_audio = audio_dir / "turn_0003_client_b.wav"
    second_human_audio = audio_dir / "human_counselor_turn_0004.wav"
    followup_client_audio = audio_dir / "turn_0005_client_a.wav"
    _write_silent_wav(first_human_audio, duration_seconds=1.2)
    _write_silent_wav(client_a_audio, duration_seconds=0.5)
    _write_silent_wav(client_b_audio, duration_seconds=0.4)
    _write_constant_wav(second_human_audio, duration_seconds=0.7, amplitude=1000)
    _write_silent_wav(followup_client_audio, duration_seconds=0.3)
    _write_jsonl(
        transcripts_dir / "session_compact_boundary_audio.jsonl",
        [
            {
                "session_id": "session_compact_boundary_audio",
                "turn_id": 1,
                "speaker": "counselor",
                "speaker_id": "counselor",
                "transcript_type": "human_final",
                "text": "今日はどのような相談でしょうか。",
                "audio_log_path": "internal/audio/human_counselor_turn_0001.wav",
                "metadata": {"speaker_display_name": "カウンセラー"},
            },
            {
                "session_id": "session_compact_boundary_audio",
                "turn_id": 2,
                "speaker": "client_a",
                "speaker_id": "client_a",
                "transcript_type": "generated_final",
                "text": "娘のことで相談に来ました。",
                "metadata": {"speaker_display_name": "妻"},
            },
            {
                "session_id": "session_compact_boundary_audio",
                "turn_id": 3,
                "speaker": "client_b",
                "speaker_id": "client_b",
                "transcript_type": "generated_final",
                "text": "私も心配しています。",
                "metadata": {"speaker_display_name": "夫"},
            },
            {
                "session_id": "session_compact_boundary_audio",
                "turn_id": 4,
                "speaker": "counselor",
                "speaker_id": "counselor",
                "transcript_type": "human_final",
                "text": "その間、ご本人はどう過ごしていますか。",
                "audio_log_path": "internal/audio/human_counselor_turn_0004.wav",
                "metadata": {"speaker_display_name": "カウンセラー"},
            },
            {
                "session_id": "session_compact_boundary_audio",
                "turn_id": 5,
                "speaker": "client_a",
                "speaker_id": "client_a",
                "transcript_type": "generated_final",
                "text": "部屋で動画を見ていることが多いです。",
                "metadata": {"speaker_display_name": "妻"},
            },
        ],
    )
    _write_jsonl(
        audio_dir / "session_timeline.jsonl",
        [
            {
                "session_id": "session_compact_boundary_audio",
                "turn_id": 1,
                "speaker_id": "counselor",
                "start_seconds": 0.0,
                "end_seconds": 1.2,
            },
            {
                "session_id": "session_compact_boundary_audio",
                "turn_id": 2,
                "speaker_id": "client_a",
                "start_seconds": 2.1,
                "end_seconds": 2.6,
            },
            {
                "session_id": "session_compact_boundary_audio",
                "turn_id": 3,
                "speaker_id": "client_b",
                "start_seconds": 3.7,
                "end_seconds": 4.1,
            },
            {
                "session_id": "session_compact_boundary_audio",
                "turn_id": 4,
                "speaker_id": "counselor",
                "start_seconds": 8.0,
                "end_seconds": 8.7,
            },
            {
                "session_id": "session_compact_boundary_audio",
                "turn_id": 5,
                "speaker_id": "client_a",
                "start_seconds": 11.7,
                "end_seconds": 12.0,
            },
        ],
    )

    turns = load_public_transcript_turns(session_dir)
    result = build_compact_session_audio(session_dir, turns)

    assert session_has_human_audio_turns(turns)
    assert CompactSessionAudioConfig().human_boundary_silence_seconds == pytest.approx(
        0.8
    )
    assert (
        CompactSessionAudioConfig().human_trailing_boundary_silence_seconds
        == pytest.approx(0.4)
    )
    assert result is not None
    assert result.audio_path == audio_dir / "session_compact.wav"
    assert result.audio_path.exists()
    assert _read_wav_duration_seconds(result.audio_path) == pytest.approx(5.4)
    assert [turn.cumulative_audio_seconds for turn in result.turns] == pytest.approx(
        [0.0, 1.2, 2.8, 3.2, 5.1]
    )
    assert [turn.audio_duration_seconds for turn in result.turns] == pytest.approx(
        [1.2, 0.5, 0.4, 1.9, 0.3]
    )
    assert result.turns[2].cumulative_audio_seconds == pytest.approx(
        result.turns[1].cumulative_audio_seconds
        + result.turns[1].audio_duration_seconds
        + 1.1
    )
    assert result.turns[3].cumulative_audio_seconds == pytest.approx(
        result.turns[2].cumulative_audio_seconds
        + result.turns[2].audio_duration_seconds
    )
    assert result.turns[4].cumulative_audio_seconds == pytest.approx(
        result.turns[3].cumulative_audio_seconds
        + result.turns[3].audio_duration_seconds
    )
    assert format_public_script(result.turns, format="txt") == (
        "00:00 カウンセラー: 今日はどのような相談でしょうか。\n"
        "00:01 妻: 娘のことで相談に来ました。\n"
        "00:02 夫: 私も心配しています。\n"
        "00:03 カウンセラー: その間、ご本人はどう過ごしていますか。\n"
        "00:05 妻: 部屋で動画を見ていることが多いです。"
    )


def test_build_compact_session_audio_converts_human_recording_to_ai_audio_format(
    tmp_path: Path,
) -> None:
    session_dir = tmp_path / "session_compact_mixed_format_audio"
    transcripts_dir = session_dir / "internal" / "transcripts"
    audio_dir = session_dir / "internal" / "audio"
    transcripts_dir.mkdir(parents=True)
    audio_dir.mkdir(parents=True)
    human_audio = audio_dir / "turn_0001_counselor.wav"
    client_audio = audio_dir / "turn_0002_client.wav"
    _write_constant_wav(
        human_audio,
        duration_seconds=0.5,
        amplitude=1000,
        sample_rate=48_000,
        channels=2,
    )
    _write_silent_wav(
        client_audio,
        duration_seconds=0.5,
        sample_rate=24_000,
        channels=1,
    )
    _write_jsonl(
        transcripts_dir / "session_compact_mixed_format_audio.jsonl",
        [
            {
                "session_id": "session_compact_mixed_format_audio",
                "turn_id": 1,
                "speaker": "counselor",
                "speaker_id": "counselor",
                "transcript_type": "human_final",
                "text": "少し詳しく教えてください。",
                "metadata": {
                    "speaker_display_name": "カウンセラー",
                    "source_audio_ref": "internal/audio/turn_0001_counselor.wav",
                },
            },
            {
                "session_id": "session_compact_mixed_format_audio",
                "turn_id": 2,
                "speaker": "client",
                "speaker_id": "client",
                "transcript_type": "generated_final",
                "text": "朝が一番大変です。",
                "metadata": {"speaker_display_name": "クライアント"},
            },
        ],
    )

    turns = load_public_transcript_turns(session_dir)
    result = build_compact_session_audio(
        session_dir,
        turns,
        config=CompactSessionAudioConfig(
            human_boundary_silence_seconds=0.0,
            human_trailing_boundary_silence_seconds=0.0,
        ),
    )

    assert result is not None
    assert result.audio_path == audio_dir / "session_compact.wav"
    with wave.open(str(result.audio_path), "rb") as wav_file:
        assert wav_file.getframerate() == 24_000
        assert wav_file.getnchannels() == 1
        assert wav_file.getsampwidth() == 2
        assert wav_file.getnframes() == 24_000
        pcm = wav_file.readframes(wav_file.getnframes())
    first_human_sample = int.from_bytes(pcm[:2], "little", signed=True)
    assert first_human_sample == pytest.approx(1000, abs=1)
    assert [turn.cumulative_audio_seconds for turn in result.turns] == pytest.approx(
        [0.0, 0.5]
    )
    assert [turn.audio_duration_seconds for turn in result.turns] == pytest.approx(
        [0.5, 0.5]
    )


def test_build_compact_session_audio_trims_interrupted_ai_turn_at_played_ms(
    tmp_path: Path,
) -> None:
    session_dir = tmp_path / "session_compact_interrupted_audio"
    transcripts_dir = session_dir / "internal" / "transcripts"
    events_dir = session_dir / "internal" / "events"
    audio_dir = session_dir / "internal" / "audio"
    transcripts_dir.mkdir(parents=True)
    events_dir.mkdir(parents=True)
    audio_dir.mkdir(parents=True)
    _write_constant_wav(
        audio_dir / "turn_0001_client_b.wav",
        duration_seconds=1.0,
        amplitude=1200,
    )
    _write_silent_wav(audio_dir / "turn_0002_counselor.wav", duration_seconds=0.2)
    _write_jsonl(
        transcripts_dir / "session_compact_interrupted_audio.jsonl",
        [
            {
                "session_id": "session_compact_interrupted_audio",
                "turn_id": 1,
                "speaker": "client_b",
                "speaker_id": "client_b",
                "transcript_type": "generated_interrupted",
                "text": "途中まで話していた内容です。",
                "metadata": {
                    "speaker_display_name": "夫",
                    "interrupted": True,
                },
            },
            {
                "session_id": "session_compact_interrupted_audio",
                "turn_id": 2,
                "speaker": "counselor",
                "speaker_id": "counselor",
                "transcript_type": "human_final",
                "text": "ここで確認させてください。",
                "metadata": {"speaker_display_name": "カウンセラー"},
            },
        ],
    )
    _write_jsonl(
        events_dir / "session_compact_interrupted_audio.jsonl",
        [
            {
                "session_id": "session_compact_interrupted_audio",
                "event_type": "realtime_playback_interrupted",
                "turn_id": 1,
                "speaker": "client_b",
                "speaker_id": "client_b",
                "details": {
                    "played_ms": 250,
                    "reason": "human_barge_in",
                },
            }
        ],
    )

    turns = load_public_transcript_turns(session_dir)
    result = build_compact_session_audio(
        session_dir,
        turns,
        config=CompactSessionAudioConfig(
            human_boundary_silence_seconds=0.0,
            human_trailing_boundary_silence_seconds=0.0,
        ),
    )

    assert result is not None
    assert _read_wav_duration_seconds(result.audio_path) == pytest.approx(0.45)
    assert [turn.audio_duration_seconds for turn in result.turns] == pytest.approx(
        [0.25, 0.2]
    )
    with wave.open(str(result.audio_path), "rb") as wav_file:
        assert wav_file.getnframes() == round(0.45 * wav_file.getframerate())
        pcm = wav_file.readframes(wav_file.getnframes())
    assert int.from_bytes(pcm[:2], "little", signed=True) == pytest.approx(
        1200,
        abs=1,
    )


def test_build_compact_session_audio_trims_interrupted_generated_final_turn(
    tmp_path: Path,
) -> None:
    session_dir = tmp_path / "session_compact_interrupted_final_audio"
    transcripts_dir = session_dir / "internal" / "transcripts"
    events_dir = session_dir / "internal" / "events"
    audio_dir = session_dir / "internal" / "audio"
    transcripts_dir.mkdir(parents=True)
    events_dir.mkdir(parents=True)
    audio_dir.mkdir(parents=True)
    _write_constant_wav(
        audio_dir / "turn_0001_client_a.wav",
        duration_seconds=1.0,
        amplitude=1200,
    )
    _write_silent_wav(audio_dir / "turn_0002_counselor.wav", duration_seconds=0.2)
    _write_jsonl(
        transcripts_dir / "session_compact_interrupted_final_audio.jsonl",
        [
            {
                "session_id": "session_compact_interrupted_final_audio",
                "turn_id": 1,
                "speaker": "client_a",
                "speaker_id": "client_a",
                "transcript_type": "generated_final",
                "text": "生成は完了したが再生中に割り込まれた内容です。",
                "metadata": {"speaker_display_name": "妻"},
            },
            {
                "session_id": "session_compact_interrupted_final_audio",
                "turn_id": 2,
                "speaker": "counselor",
                "speaker_id": "counselor",
                "transcript_type": "human_final",
                "text": "ここで確認させてください。",
                "metadata": {"speaker_display_name": "カウンセラー"},
            },
        ],
    )
    _write_jsonl(
        events_dir / "session_compact_interrupted_final_audio.jsonl",
        [
            {
                "session_id": "session_compact_interrupted_final_audio",
                "event_type": "realtime_playback_interrupted",
                "turn_id": 1,
                "speaker": "client_a",
                "speaker_id": "client_a",
                "details": {
                    "played_ms": 125,
                    "reason": "human_barge_in",
                },
            }
        ],
    )

    turns = load_public_transcript_turns(session_dir)
    result = build_compact_session_audio(
        session_dir,
        turns,
        config=CompactSessionAudioConfig(
            human_boundary_silence_seconds=0.0,
            human_trailing_boundary_silence_seconds=0.0,
        ),
    )

    assert result is not None
    assert _read_wav_duration_seconds(result.audio_path) == pytest.approx(0.325)
    assert [turn.audio_duration_seconds for turn in result.turns] == pytest.approx(
        [0.125, 0.2]
    )


def test_build_compact_session_audio_trims_structured_active_interrupted_turn(
    tmp_path: Path,
) -> None:
    session_dir = tmp_path / "session_compact_structured_active_interrupted_audio"
    transcripts_dir = session_dir / "internal" / "transcripts"
    events_dir = session_dir / "internal" / "events"
    audio_dir = session_dir / "internal" / "audio"
    transcripts_dir.mkdir(parents=True)
    events_dir.mkdir(parents=True)
    audio_dir.mkdir(parents=True)
    _write_constant_wav(
        audio_dir / "turn_0006_client_b.wav",
        duration_seconds=1.0,
        amplitude=1200,
    )
    _write_silent_wav(audio_dir / "turn_0007_counselor.wav", duration_seconds=0.2)
    _write_jsonl(
        transcripts_dir / "session_compact_structured_active_interrupted_audio.jsonl",
        [
            {
                "session_id": "session_compact_structured_active_interrupted_audio",
                "turn_id": 6,
                "speaker": "client_b",
                "speaker_id": "client_b",
                "transcript_type": "generated_interrupted",
                "text": "実際に割り込まれた発話です。",
                "metadata": {"speaker_display_name": "夫", "interrupted": True},
            },
            {
                "session_id": "session_compact_structured_active_interrupted_audio",
                "turn_id": 7,
                "speaker": "counselor",
                "speaker_id": "counselor",
                "transcript_type": "human_final",
                "text": "ここで確認させてください。",
                "metadata": {"speaker_display_name": "カウンセラー"},
            },
        ],
    )
    _write_jsonl(
        events_dir / "session_compact_structured_active_interrupted_audio.jsonl",
        [
            {
                "session_id": "session_compact_structured_active_interrupted_audio",
                "event_type": "realtime_playback_interrupted",
                "turn_id": 5,
                "speaker": "client_a",
                "speaker_id": "client_a",
                "details": {
                    "played_ms": 144,
                    "reason": "human_barge_in",
                    "active_response_stop": {
                        "turn_id": 6,
                        "speaker": "client_b",
                        "speaker_id": "client_b",
                        "played_ms": 125,
                        "cancel_sent": True,
                    },
                },
            }
        ],
    )

    turns = load_public_transcript_turns(session_dir)
    result = build_compact_session_audio(
        session_dir,
        turns,
        config=CompactSessionAudioConfig(
            human_boundary_silence_seconds=0.0,
            human_trailing_boundary_silence_seconds=0.0,
        ),
    )

    assert result is not None
    assert _read_wav_duration_seconds(result.audio_path) == pytest.approx(0.325)
    assert [turn.audio_duration_seconds for turn in result.turns] == pytest.approx(
        [0.125, 0.2]
    )


def test_build_compact_session_audio_trims_legacy_active_interrupted_turn(
    tmp_path: Path,
) -> None:
    session_dir = tmp_path / "session_compact_legacy_active_interrupted_audio"
    transcripts_dir = session_dir / "internal" / "transcripts"
    events_dir = session_dir / "internal" / "events"
    audio_dir = session_dir / "internal" / "audio"
    transcripts_dir.mkdir(parents=True)
    events_dir.mkdir(parents=True)
    audio_dir.mkdir(parents=True)
    _write_constant_wav(
        audio_dir / "turn_0006_client_b.wav",
        duration_seconds=1.0,
        amplitude=1200,
    )
    _write_silent_wav(audio_dir / "turn_0007_counselor.wav", duration_seconds=0.2)
    _write_jsonl(
        transcripts_dir / "session_compact_legacy_active_interrupted_audio.jsonl",
        [
            {
                "session_id": "session_compact_legacy_active_interrupted_audio",
                "turn_id": 6,
                "speaker": "client_b",
                "speaker_id": "client_b",
                "transcript_type": "generated_interrupted",
                "text": "実際に割り込まれた発話です。",
                "metadata": {"speaker_display_name": "夫", "interrupted": True},
            },
            {
                "session_id": "session_compact_legacy_active_interrupted_audio",
                "turn_id": 7,
                "speaker": "counselor",
                "speaker_id": "counselor",
                "transcript_type": "human_final",
                "text": "ここで確認させてください。",
                "metadata": {"speaker_display_name": "カウンセラー"},
            },
        ],
    )
    _write_jsonl(
        events_dir / "session_compact_legacy_active_interrupted_audio.jsonl",
        [
            {
                "session_id": "session_compact_legacy_active_interrupted_audio",
                "event_type": "realtime_playback_interrupted",
                "turn_id": 5,
                "speaker": "client_a",
                "speaker_id": "client_a",
                "monotonic_time": 100.0,
                "details": {
                    "played_ms": 144,
                    "reason": "human_barge_in",
                    "active_response_stop": {
                        "played_ms": 0,
                        "cancel_sent": True,
                        "sent_event_types": ["response.cancel"],
                    },
                },
            },
            {
                "session_id": "session_compact_legacy_active_interrupted_audio",
                "event_type": "generated_interrupted",
                "turn_id": 6,
                "speaker": "client_b",
                "speaker_id": "client_b",
                "monotonic_time": 100.05,
                "details": {"interrupted": True},
            },
        ],
    )

    turns = load_public_transcript_turns(session_dir)
    result = build_compact_session_audio(
        session_dir,
        turns,
        config=CompactSessionAudioConfig(
            human_boundary_silence_seconds=0.0,
            human_trailing_boundary_silence_seconds=0.0,
        ),
    )

    assert result is not None
    assert _read_wav_duration_seconds(result.audio_path) == pytest.approx(0.2)
    assert [turn.audio_duration_seconds for turn in result.turns] == pytest.approx(
        [0.0, 0.2]
    )


def test_load_public_transcript_turns_returns_empty_when_transcript_file_is_missing(tmp_path: Path) -> None:
    assert load_public_transcript_turns(tmp_path / "missing_session") == []


def test_load_public_transcript_turns_prefers_session_audio_timeline(tmp_path: Path) -> None:
    session_dir = tmp_path / "session_timeline"
    transcripts_dir = session_dir / "internal" / "transcripts"
    audio_dir = session_dir / "internal" / "audio"
    transcripts_dir.mkdir(parents=True)
    audio_dir.mkdir(parents=True)
    session_audio = audio_dir / "session_realtime.wav"
    _write_silent_wav(audio_dir / "turn_0001_counselor.wav", duration_seconds=1.0)
    _write_silent_wav(audio_dir / "turn_0002_client.wav", duration_seconds=1.0)
    _write_silent_wav(session_audio, duration_seconds=6.0)
    _write_jsonl(
        transcripts_dir / "session_timeline.jsonl",
        [
            {
                "session_id": "session_timeline",
                "turn_id": 1,
                "speaker": "counselor",
                "transcript_type": "final",
                "text": "first turn",
            },
            {
                "session_id": "session_timeline",
                "turn_id": 2,
                "speaker": "client",
                "transcript_type": "final",
                "text": "second turn",
            },
        ],
    )
    _write_jsonl(
        audio_dir / "session_timeline.jsonl",
        [
            {
                "session_id": "session_timeline",
                "turn_id": 1,
                "speaker": "counselor",
                "start_seconds": 2.0,
                "end_seconds": 3.0,
            },
            {
                "session_id": "session_timeline",
                "turn_id": 2,
                "speaker": "client",
                "start_seconds": 5.0,
                "end_seconds": 6.0,
            },
        ],
    )

    turns = load_public_transcript_turns(session_dir)

    assert [turn.cumulative_audio_seconds for turn in turns] == [2.0, 5.0]
    assert [turn.audio_duration_seconds for turn in turns] == [1.0, 1.0]
    assert format_public_script(turns, format="txt") == (
        "00:02 counselor: first turn\n00:05 client: second turn"
    )
    assert session_realtime_audio_path(session_dir) == session_audio


def test_load_public_transcript_turns_supports_speaker_id_display_name_and_legacy_timeline(
    tmp_path: Path,
) -> None:
    session_dir = tmp_path / "session_speaker_id"
    transcripts_dir = session_dir / "internal" / "transcripts"
    audio_dir = session_dir / "internal" / "audio"
    transcripts_dir.mkdir(parents=True)
    audio_dir.mkdir(parents=True)
    client_audio = audio_dir / "turn_0001_client_alpha.wav"
    observer_audio = audio_dir / "turn_0002_observer_alpha.wav"
    _write_silent_wav(client_audio, duration_seconds=1.0)
    _write_silent_wav(observer_audio, duration_seconds=1.0)
    _write_jsonl(
        transcripts_dir / "session_speaker_id.jsonl",
        [
            {
                "session_id": "session_speaker_id",
                "turn_id": 1,
                "speaker_id": "client_alpha",
                "transcript_type": "final",
                "text": "new speaker id only",
                "metadata": {
                    "speaker_display_name": "Client A",
                    "hidden_client_background": "DO_NOT_EXPORT",
                },
            },
            {
                "session_id": "session_speaker_id",
                "turn_id": 2,
                "speaker_id": "observer_alpha",
                "speaker": "observer",
                "transcript_type": "final",
                "text": "legacy timeline speaker",
                "metadata": {"speaker_display_name": "Observer"},
            },
        ],
    )
    _write_jsonl(
        audio_dir / "session_timeline.jsonl",
        [
            {
                "session_id": "session_speaker_id",
                "turn_id": 1,
                "speaker_id": "client_alpha",
                "start_seconds": 12.0,
                "end_seconds": 13.0,
            },
            {
                "session_id": "session_speaker_id",
                "turn_id": 2,
                "speaker": "observer",
                "start_seconds": 15.0,
                "end_seconds": 16.0,
            },
        ],
    )

    turns = load_public_transcript_turns(session_dir)
    script = format_public_script(turns)

    assert [turn.speaker_id for turn in turns] == ["client_alpha", "observer_alpha"]
    assert [turn.speaker for turn in turns] == ["client_alpha", "observer"]
    assert [turn.speaker_display_name for turn in turns] == ["Client A", "Observer"]
    assert [turn.audio_path for turn in turns] == [client_audio, observer_audio]
    assert [turn.cumulative_audio_seconds for turn in turns] == [12.0, 15.0]
    assert script == (
        "- 00:12 Client A: new speaker id only\n"
        "- 00:15 Observer: legacy timeline speaker"
    )
    assert "DO_NOT_EXPORT" not in script
    assert "metadata" not in script


def test_load_public_transcript_turns_uses_wav_duration_after_partial_timeline(
    tmp_path: Path,
) -> None:
    session_dir = tmp_path / "session_partial_timeline"
    transcripts_dir = session_dir / "internal" / "transcripts"
    audio_dir = session_dir / "internal" / "audio"
    transcripts_dir.mkdir(parents=True)
    audio_dir.mkdir(parents=True)
    _write_silent_wav(audio_dir / "turn_0001_counselor.wav", duration_seconds=1.0)
    _write_silent_wav(audio_dir / "turn_0002_client.wav", duration_seconds=2.0)
    _write_silent_wav(audio_dir / "turn_0003_counselor.wav", duration_seconds=3.0)
    _write_jsonl(
        transcripts_dir / "session_partial_timeline.jsonl",
        [
            {
                "session_id": "session_partial_timeline",
                "turn_id": 1,
                "speaker": "counselor",
                "transcript_type": "final",
                "text": "first turn",
            },
            {
                "session_id": "session_partial_timeline",
                "turn_id": 2,
                "speaker": "client",
                "transcript_type": "final",
                "text": "second turn",
            },
            {
                "session_id": "session_partial_timeline",
                "turn_id": 3,
                "speaker": "counselor",
                "transcript_type": "final",
                "text": "third turn",
            },
        ],
    )
    _write_jsonl(
        audio_dir / "session_timeline.jsonl",
        [
            {
                "session_id": "session_partial_timeline",
                "turn_id": 1,
                "speaker": "counselor",
                "start_seconds": 5.0,
                "end_seconds": 6.0,
            },
        ],
    )

    turns = load_public_transcript_turns(session_dir)

    assert [turn.cumulative_audio_seconds for turn in turns] == [5.0, 6.0, 8.0]
    assert [turn.audio_duration_seconds for turn in turns] == [1.0, 2.0, 3.0]
    assert format_public_script(turns, format="txt") == (
        "00:05 counselor: first turn\n"
        "00:06 client: second turn\n"
        "00:08 counselor: third turn"
    )


def test_format_public_script_excludes_metadata_and_hidden_info(tmp_path: Path) -> None:
    session_dir = tmp_path / "session_metadata"
    transcripts_dir = session_dir / "internal" / "transcripts"
    transcripts_dir.mkdir(parents=True)
    _write_jsonl(
        transcripts_dir / "session_metadata.jsonl",
        [
            {
                "session_id": "session_metadata",
                "turn_id": 1,
                "speaker": "client",
                "transcript_type": "final",
                "text": "visible transcript",
                "created_at": "2026-06-23T08:07:06Z",
                "metadata": {
                    "system_prompt": "SECRET_METADATA",
                    "hidden_client_background": "DO_NOT_EXPORT",
                },
            },
            {
                "session_id": "session_metadata",
                "turn_id": 2,
                "speaker": "counselor",
                "transcript_type": "final",
                "text": "   ",
                "created_at": "2026-06-23T08:08:00Z",
                "metadata": {"system_prompt": "EMPTY_TEXT_SECRET"},
            },
        ],
    )

    script = format_public_script(load_public_transcript_turns(session_dir))

    assert script == "- 00:00 client: visible transcript"
    assert "SECRET_METADATA" not in script
    assert "DO_NOT_EXPORT" not in script
    assert "EMPTY_TEXT_SECRET" not in script
    assert "metadata" not in script


def test_format_public_script_supports_txt_and_rejects_invalid_format() -> None:
    turns = [
        PublicTranscriptTurn(
            turn_id=1,
            speaker="counselor",
            text="hello",
            created_at=None,
            audio_path=None,
        )
    ]

    assert format_public_script(turns, format="txt") == "--:-- counselor: hello"
    assert (
        format_public_script(
            [
                PublicTranscriptTurn(
                    turn_id=1,
                    speaker="counselor",
                    text="hello",
                    created_at=None,
                    audio_path=None,
                    cumulative_audio_seconds=65.2,
                )
            ],
            format="txt",
        )
        == "01:05 counselor: hello"
    )
    with pytest.raises(ValueError, match="format"):
        format_public_script(turns, format="html")


def _write_silent_wav(
    path: Path,
    *,
    duration_seconds: float,
    sample_rate: int = 8000,
    channels: int = 1,
) -> None:
    frame_count = int(sample_rate * duration_seconds)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(b"\x00\x00" * frame_count * channels)


def _write_constant_wav(
    path: Path,
    *,
    duration_seconds: float,
    amplitude: int,
    sample_rate: int = 8000,
    channels: int = 1,
) -> None:
    frame_count = int(sample_rate * duration_seconds)
    sample = int(amplitude).to_bytes(2, "little", signed=True)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(sample * frame_count * channels)


def _read_wav_duration_seconds(path: Path) -> float:
    with wave.open(str(path), "rb") as wav_file:
        return wav_file.getnframes() / wav_file.getframerate()


def _write_jsonl(path: Path, records: list[dict[str, object] | str]) -> None:
    lines = [record if isinstance(record, str) else json.dumps(record) for record in records]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
