from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

from counseling_voice_demo.log_writer import SessionLogPaths, append_error_log
from counseling_voice_demo.models import SessionState, Turn, TurnStatus, utc_now
from counseling_voice_demo.system_checks import find_ffmpeg


EXCLUDED_STATUSES = {
    TurnStatus.INVALIDATED,
    TurnStatus.PLAYBACK_HOLD,
    TurnStatus.IGNORED_AFTER_STOP,
    TurnStatus.ERROR,
}


class AudioMergeError(RuntimeError):
    error_type = "audio_merge_error"


class MissingFfmpegError(AudioMergeError):
    error_type = "ffmpeg_missing"


class MissingAudioError(AudioMergeError):
    error_type = "audio_missing"


class AudioMergeInterrupted(AudioMergeError):
    error_type = "audio_merge_interrupted"


@dataclass(frozen=True)
class AudioMergeConfig:
    output_filename: str = "session_full_public.mp3"
    include_skipped_partial: bool = False
    include_skipped_unplayed: bool = False


@dataclass(frozen=True)
class AudioMergeInput:
    turn_id: int
    revision_id: int
    audio_path: Path


@dataclass(frozen=True)
class AudioMergeResult:
    output_path: Path
    input_paths: list[Path]
    command: list[str]


def collect_merge_inputs(
    session: SessionState,
    *,
    config: AudioMergeConfig,
) -> list[AudioMergeInput]:
    inputs: list[AudioMergeInput] = []
    for turn in sorted(session.turns, key=lambda item: item.turn_id):
        if not _is_merge_candidate(turn, config=config):
            continue
        revision = turn.active_revision()
        if revision.audio_path is None:
            raise MissingAudioError(f"音声ファイルパスがありません: turn_id={turn.turn_id}")
        audio_path = Path(revision.audio_path)
        if not audio_path.exists():
            raise MissingAudioError(f"音声ファイルが見つかりません: turn_id={turn.turn_id} path={audio_path}")
        inputs.append(
            AudioMergeInput(
                turn_id=turn.turn_id,
                revision_id=revision.revision_id,
                audio_path=audio_path,
            )
        )
    return inputs


def merge_session_audio(
    session: SessionState,
    log_paths: SessionLogPaths,
    *,
    config: AudioMergeConfig | None = None,
    ffmpeg_path: str | None = None,
    ffmpeg_finder: Callable[[], str | None] = find_ffmpeg,
    should_cancel: Callable[[], bool] | None = None,
    runner: Callable[..., object] = subprocess.run,
) -> AudioMergeResult:
    merge_config = config or AudioMergeConfig()
    should_cancel = should_cancel or (lambda: False)
    output_path = log_paths.public_merged_dir / merge_config.output_filename

    try:
        resolved_ffmpeg = ffmpeg_path or ffmpeg_finder()
        if resolved_ffmpeg is None:
            raise MissingFfmpegError("ffmpeg が見つかりません。MP3結合を使う前に ffmpeg をインストールしてください。")

        inputs = _collect_inputs_with_cancel_checks(
            session,
            config=merge_config,
            should_cancel=should_cancel,
        )
        if not inputs:
            raise MissingAudioError("結合対象の音声ファイルがありません。")

        concat_list_path = log_paths.internal_dir / "session_full_public_concat.txt"
        _write_concat_list(concat_list_path, [item.audio_path for item in inputs])
        command = [
            resolved_ffmpeg,
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_list_path),
            "-c",
            "copy",
            str(output_path),
        ]
        runner(command, check=True, capture_output=True, text=True)
        return AudioMergeResult(
            output_path=output_path,
            input_paths=[item.audio_path for item in inputs],
            command=command,
        )
    except AudioMergeError as exc:
        _log_audio_merge_error(log_paths, exc)
        raise
    except subprocess.CalledProcessError as exc:
        error = AudioMergeError(f"ffmpeg によるMP3結合に失敗しました: {exc.stderr or exc}")
        _log_audio_merge_error(log_paths, error)
        raise error from exc


def _collect_inputs_with_cancel_checks(
    session: SessionState,
    *,
    config: AudioMergeConfig,
    should_cancel: Callable[[], bool],
) -> list[AudioMergeInput]:
    inputs: list[AudioMergeInput] = []
    for turn in sorted(session.turns, key=lambda item: item.turn_id):
        if not _is_merge_candidate(turn, config=config):
            continue
        if should_cancel():
            raise AudioMergeInterrupted("セッション全体MP3作成を中断しました。")
        revision = turn.active_revision()
        if revision.audio_path is None:
            raise MissingAudioError(f"音声ファイルパスがありません: turn_id={turn.turn_id}")
        audio_path = Path(revision.audio_path)
        if not audio_path.exists():
            raise MissingAudioError(f"音声ファイルが見つかりません: turn_id={turn.turn_id} path={audio_path}")
        inputs.append(
            AudioMergeInput(
                turn_id=turn.turn_id,
                revision_id=revision.revision_id,
                audio_path=audio_path,
            )
        )
    return inputs


def _is_merge_candidate(turn: Turn, *, config: AudioMergeConfig) -> bool:
    if turn.status in EXCLUDED_STATUSES:
        return False
    if turn.status == TurnStatus.SKIPPED_PARTIAL and not config.include_skipped_partial:
        return False
    if turn.status == TurnStatus.SKIPPED_UNPLAYED and not config.include_skipped_unplayed:
        return False
    try:
        turn.active_revision()
    except ValueError:
        return False
    return True


def _write_concat_list(path: Path, audio_paths: Sequence[Path]) -> None:
    lines = [f"file '{_escape_concat_path(audio_path)}'" for audio_path in audio_paths]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _escape_concat_path(path: Path) -> str:
    return str(path).replace("'", "'\\''")


def _log_audio_merge_error(log_paths: SessionLogPaths, error: AudioMergeError) -> None:
    append_error_log(
        log_paths,
        {
            "recorded_at": utc_now(),
            "error_type": error.error_type,
            "message": str(error),
        },
    )
