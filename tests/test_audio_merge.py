from __future__ import annotations

import json
from pathlib import Path

import pytest

from counseling_voice_demo.audio_merge import (
    AudioMergeConfig,
    AudioMergeInterrupted,
    MissingAudioError,
    MissingFfmpegError,
    collect_merge_inputs,
    merge_session_audio,
)
from counseling_voice_demo.log_writer import create_session_log_dirs
from counseling_voice_demo.models import SessionState, SpeakerRole, TurnStatus


def test_collect_merge_inputs_uses_active_revisions_in_turn_id_order_and_skips_default_exclusions(tmp_path) -> None:
    session = SessionState(session_id="session_merge_inputs")
    older_path = _write_audio(tmp_path / "old_turn_0001_rev01.mp3")
    active_late = _write_audio(tmp_path / "turn_0002_rev02.mp3")
    active_early = _write_audio(tmp_path / "turn_0001_rev02.mp3")
    skipped_path = _write_audio(tmp_path / "turn_0003_rev01.mp3")
    invalidated_path = _write_audio(tmp_path / "turn_0004_rev01.mp3")

    late = session.add_turn(
        speaker_role=SpeakerRole.CLIENT,
        speaker_name="高橋",
        profile_id="client",
        text="後の発話です。",
        status=TurnStatus.PLAYED,
    )
    early = session.add_turn(
        speaker_role=SpeakerRole.COUNSELOR,
        speaker_name="佐伯",
        profile_id="counselor",
        text="先の発話です。",
        status=TurnStatus.PLAYED,
    )
    skipped = session.add_turn(
        speaker_role=SpeakerRole.CLIENT,
        speaker_name="高橋",
        profile_id="client",
        text="スキップされます。",
        status=TurnStatus.SKIPPED_UNPLAYED,
    )
    invalidated = session.add_turn(
        speaker_role=SpeakerRole.COUNSELOR,
        speaker_name="佐伯",
        profile_id="counselor",
        text="無効化されます。",
        status=TurnStatus.PLAYED,
    )

    late.turn_id = 2
    early.turn_id = 1
    early.active_revision().audio_path = str(older_path)
    early.add_revision(text="先の発話の新版です。").audio_path = str(active_early)
    late.active_revision().audio_path = str(active_late)
    skipped.active_revision().audio_path = str(skipped_path)
    invalidated.active_revision().audio_path = str(invalidated_path)
    invalidated.invalidate("test_invalidated")

    inputs = collect_merge_inputs(session, config=AudioMergeConfig())

    assert [item.turn_id for item in inputs] == [1, 2]
    assert [item.audio_path for item in inputs] == [active_early, active_late]


def test_collect_merge_inputs_can_include_skipped_partial_by_config(tmp_path) -> None:
    session = SessionState(session_id="session_merge_skipped_partial")
    played_path = _write_audio(tmp_path / "turn_0001.mp3")
    skipped_path = _write_audio(tmp_path / "turn_0002.mp3")
    played = session.add_turn(
        speaker_role=SpeakerRole.COUNSELOR,
        speaker_name="佐伯",
        profile_id="counselor",
        text="再生済みです。",
        status=TurnStatus.PLAYED,
    )
    skipped = session.add_turn(
        speaker_role=SpeakerRole.CLIENT,
        speaker_name="高橋",
        profile_id="client",
        text="途中スキップです。",
        status=TurnStatus.SKIPPED_PARTIAL,
    )
    played.active_revision().audio_path = str(played_path)
    skipped.active_revision().audio_path = str(skipped_path)

    default_inputs = collect_merge_inputs(session, config=AudioMergeConfig())
    configured_inputs = collect_merge_inputs(
        session,
        config=AudioMergeConfig(include_skipped_partial=True),
    )

    assert [item.turn_id for item in default_inputs] == [1]
    assert [item.turn_id for item in configured_inputs] == [1, 2]


def test_collect_merge_inputs_excludes_unpublished_and_error_statuses(tmp_path) -> None:
    session = SessionState(session_id="session_merge_unpublished_statuses")
    included_path = _write_audio(tmp_path / "turn_0001_played.mp3")
    playback_hold_path = _write_audio(tmp_path / "turn_0002_playback_hold.mp3")
    ignored_after_stop_path = _write_audio(tmp_path / "turn_0003_ignored_after_stop.mp3")
    error_path = _write_audio(tmp_path / "turn_0004_error.mp3")
    for status, audio_path in [
        (TurnStatus.PLAYED, included_path),
        (TurnStatus.PLAYBACK_HOLD, playback_hold_path),
        (TurnStatus.IGNORED_AFTER_STOP, ignored_after_stop_path),
        (TurnStatus.ERROR, error_path),
    ]:
        turn = session.add_turn(
            speaker_role=SpeakerRole.COUNSELOR,
            speaker_name="佐伯",
            profile_id="counselor",
            text=f"{status.value} の発話です。",
            status=status,
        )
        turn.active_revision().audio_path = str(audio_path)

    inputs = collect_merge_inputs(session, config=AudioMergeConfig())

    assert [item.turn_id for item in inputs] == [1]
    assert [item.audio_path for item in inputs] == [included_path]


def test_merge_session_audio_writes_concat_file_and_runs_ffmpeg(tmp_path) -> None:
    paths = create_session_log_dirs(tmp_path, "session_merge_run")
    session = SessionState(session_id="session_merge_run")
    first_path = _write_audio(tmp_path / "turn_0001.mp3")
    second_path = _write_audio(tmp_path / "turn_0002.mp3")
    for path, role in [(first_path, SpeakerRole.COUNSELOR), (second_path, SpeakerRole.CLIENT)]:
        turn = session.add_turn(
            speaker_role=role,
            speaker_name=role.value,
            profile_id=role.value,
            text=f"{role.value} text",
            status=TurnStatus.PLAYED,
        )
        turn.active_revision().audio_path = str(path)
    runner = _FakeRunner()

    result = merge_session_audio(
        session,
        paths,
        ffmpeg_path="/usr/bin/ffmpeg",
        runner=runner,
    )

    assert result.output_path == paths.public_merged_dir / "session_full_public.mp3"
    assert result.input_paths == [first_path, second_path]
    assert result.output_path.read_bytes() == b"merged"
    assert runner.commands[0][:5] == ["/usr/bin/ffmpeg", "-y", "-f", "concat", "-safe"]
    concat_file = Path(runner.commands[0][runner.commands[0].index("-i") + 1])
    assert concat_file.read_text(encoding="utf-8").splitlines() == [
        f"file '{first_path}'",
        f"file '{second_path}'",
    ]


def test_merge_session_audio_can_be_interrupted_before_next_turn(tmp_path) -> None:
    paths = create_session_log_dirs(tmp_path, "session_merge_interrupt")
    session = _session_with_two_played_audio(tmp_path)
    should_cancel = _CancelOnSecondCheck()

    with pytest.raises(AudioMergeInterrupted, match="中断"):
        merge_session_audio(
            session,
            paths,
            ffmpeg_path="/usr/bin/ffmpeg",
            should_cancel=should_cancel,
            runner=_FakeRunner(),
        )

    assert not (paths.public_merged_dir / "session_full_public.mp3").exists()
    error_logs = _read_jsonl(paths.errors_jsonl)
    assert error_logs[0]["error_type"] == "audio_merge_interrupted"


def test_missing_ffmpeg_reports_clear_error_and_logs_it(tmp_path) -> None:
    paths = create_session_log_dirs(tmp_path, "session_merge_no_ffmpeg")
    session = _session_with_two_played_audio(tmp_path)

    with pytest.raises(MissingFfmpegError, match="ffmpeg が見つかりません"):
        merge_session_audio(
            session,
            paths,
            ffmpeg_finder=lambda: None,
            runner=_FakeRunner(),
        )

    error_logs = _read_jsonl(paths.errors_jsonl)
    assert error_logs[0]["error_type"] == "ffmpeg_missing"


def test_missing_audio_file_reports_error_without_deleting_existing_logs(tmp_path) -> None:
    paths = create_session_log_dirs(tmp_path, "session_merge_missing_audio")
    session = SessionState(session_id="session_merge_missing_audio")
    turn = session.add_turn(
        speaker_role=SpeakerRole.COUNSELOR,
        speaker_name="佐伯",
        profile_id="counselor",
        text="音声がありません。",
        status=TurnStatus.PLAYED,
    )
    turn.active_revision().audio_path = str(tmp_path / "missing.mp3")

    with pytest.raises(MissingAudioError, match="音声ファイルが見つかりません"):
        merge_session_audio(
            session,
            paths,
            ffmpeg_path="/usr/bin/ffmpeg",
            runner=_FakeRunner(),
        )

    assert paths.public_transcript_jsonl.exists()
    error_logs = _read_jsonl(paths.errors_jsonl)
    assert error_logs[0]["error_type"] == "audio_missing"


class _FakeRunner:
    def __init__(self) -> None:
        self.commands = []

    def __call__(self, command, *, check, capture_output, text):
        self.commands.append(command)
        Path(command[-1]).write_bytes(b"merged")
        return type("CompletedProcess", (), {"returncode": 0, "stderr": "", "stdout": ""})()


class _CancelOnSecondCheck:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self) -> bool:
        self.calls += 1
        return self.calls >= 2


def _session_with_two_played_audio(tmp_path) -> SessionState:
    session = SessionState(session_id="session_merge")
    for index in [1, 2]:
        audio_path = _write_audio(tmp_path / f"turn_{index:04d}.mp3")
        turn = session.add_turn(
            speaker_role=SpeakerRole.COUNSELOR,
            speaker_name="佐伯",
            profile_id="counselor",
            text=f"発話{index}",
            status=TurnStatus.PLAYED,
        )
        turn.active_revision().audio_path = str(audio_path)
    return session


def _write_audio(path: Path) -> Path:
    path.write_bytes(b"mp3")
    return path


def _read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
