from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class ReplayLoadError(RuntimeError):
    pass


@dataclass(frozen=True)
class ReplayTurn:
    turn_id: int
    revision_id: int
    speaker_role: str
    speaker_name: str
    text: str
    audio_path: Path
    transcript_type: str | None = None
    audio_log_path: Path | None = None
    cumulative_audio_seconds: float | None = None
    audio_duration_seconds: float | None = None
    speaker_id: str | None = None
    speaker_display_name: str | None = None


@dataclass(frozen=True)
class ReplaySession:
    session_dir: Path
    mode_label: str
    turns: tuple[ReplayTurn, ...]
    api_calls_forbidden: bool = True
    session_audio_path: Path | None = None

    @property
    def ui_mode_text(self) -> str:
        return f"{self.mode_label}: 既存セッション再生"


def load_replay_session(session_dir: Path | str) -> ReplaySession:
    root = Path(session_dir)
    public_transcript = root / "public" / "transcript_public.jsonl"
    internal_transcript = root / "internal" / "transcript_full.jsonl"
    runtime_transcripts_dir = root / "internal" / "transcripts"
    session_audio_path: Path | None = None

    if public_transcript.exists():
        turns = _load_public_turns(public_transcript, root)
    elif internal_transcript.exists():
        turns = _load_internal_turns(internal_transcript, root)
    elif runtime_transcripts_dir.exists():
        turns = _load_runtime_turns(root)
        session_audio_path = _runtime_session_audio_path(root)
    else:
        raise ReplayLoadError(f"Replay用の逐語録が見つかりません: {root}")

    if not turns:
        raise ReplayLoadError(f"Replay可能なターンがありません: {root}")
    return ReplaySession(
        session_dir=root,
        mode_label="Replay mode",
        turns=tuple(sorted(turns, key=lambda item: item.turn_id)),
        session_audio_path=session_audio_path,
    )


def _load_public_turns(transcript_path: Path, session_dir: Path) -> list[ReplayTurn]:
    turns: list[ReplayTurn] = []
    for record in _read_jsonl(transcript_path):
        turn_id = _require_int(record, "turn_id", transcript_path)
        revision_id = _require_int(record, "revision_id", transcript_path)
        audio_path = _resolve_audio_path(record, session_dir, turn_id=turn_id, revision_id=revision_id)
        if not audio_path.exists():
            raise ReplayLoadError(f"Replay音声ファイルが見つかりません: {audio_path}")
        turns.append(
            ReplayTurn(
                turn_id=turn_id,
                revision_id=revision_id,
                speaker_role=str(record.get("speaker_role", "")),
                speaker_name=str(record.get("speaker_name", "")),
                text=str(record.get("text", "")),
                audio_path=audio_path,
            )
        )
    return turns


def _load_internal_turns(transcript_path: Path, session_dir: Path) -> list[ReplayTurn]:
    turns: list[ReplayTurn] = []
    for record in _read_jsonl(transcript_path):
        if record.get("status") in {"invalidated", "ignored_after_stop", "error"}:
            continue
        active_revision = _active_revision(record)
        if active_revision is None:
            continue
        turn_id = _require_int(record, "turn_id", transcript_path)
        revision_id = _require_int(active_revision, "revision_id", transcript_path)
        audio_path = _resolve_audio_path(active_revision, session_dir, turn_id=turn_id, revision_id=revision_id)
        if not audio_path.exists():
            raise ReplayLoadError(f"Replay音声ファイルが見つかりません: {audio_path}")
        turns.append(
            ReplayTurn(
                turn_id=turn_id,
                revision_id=revision_id,
                speaker_role=str(record.get("speaker_role", "")),
                speaker_name=str(record.get("speaker_name", "")),
                text=str(active_revision.get("canonical_text") or active_revision.get("text_final") or active_revision.get("text_original") or ""),
                audio_path=audio_path,
            )
        )
    return turns


def _load_runtime_turns(session_dir: Path) -> list[ReplayTurn]:
    from counseling_voice_demo.runtime.session_artifacts import (
        load_public_transcript_turns,
    )

    turns: list[ReplayTurn] = []
    for public_turn in load_public_transcript_turns(session_dir):
        audio_path = public_turn.audio_path
        if audio_path is None or not audio_path.exists():
            raise ReplayLoadError(
                "Replay音声ファイルが見つかりません: "
                f"turn_id={public_turn.turn_id} session={session_dir}"
            )
        speaker_role = (
            public_turn.speaker_role
            or public_turn.speaker_id
            or public_turn.speaker
        )
        speaker_name = public_turn.speaker_display_name or public_turn.speaker
        turns.append(
            ReplayTurn(
                turn_id=public_turn.turn_id,
                revision_id=1,
                speaker_role=speaker_role,
                speaker_name=speaker_name,
                text=public_turn.text,
                audio_path=audio_path,
                transcript_type=public_turn.transcript_type,
                audio_log_path=public_turn.audio_log_path,
                cumulative_audio_seconds=public_turn.cumulative_audio_seconds,
                audio_duration_seconds=public_turn.audio_duration_seconds,
                speaker_id=public_turn.speaker_id,
                speaker_display_name=public_turn.speaker_display_name,
            )
        )
    return turns


def _runtime_session_audio_path(session_dir: Path) -> Path | None:
    from counseling_voice_demo.runtime.session_artifacts import (
        session_realtime_audio_path,
    )

    return session_realtime_audio_path(session_dir)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ReplayLoadError(f"Replay逐語録のJSONL解析に失敗しました: {path}:{line_number}") from exc
        if not isinstance(record, dict):
            raise ReplayLoadError(f"Replay逐語録の行がJSON objectではありません: {path}:{line_number}")
        records.append(record)
    return records


def _active_revision(record: dict[str, Any]) -> dict[str, Any] | None:
    active_revision_id = record.get("active_revision_id")
    revisions = record.get("revisions", [])
    if not isinstance(revisions, list):
        return None
    for revision in revisions:
        if not isinstance(revision, dict):
            continue
        if revision.get("active") is True and revision.get("revision_id") == active_revision_id:
            return revision
    return None


def _resolve_audio_path(record: dict[str, Any], session_dir: Path, *, turn_id: int, revision_id: int) -> Path:
    raw_audio_path = record.get("audio_path")
    if isinstance(raw_audio_path, str) and raw_audio_path:
        audio_path = Path(raw_audio_path)
        if audio_path.is_absolute():
            return audio_path
        return session_dir / audio_path

    pattern = f"turn_{turn_id:04d}_rev{revision_id:02d}_*.*"
    candidates = sorted((session_dir / "public" / "audio").glob(pattern))
    candidates.extend(sorted((session_dir / "internal" / "audio_all_revisions").glob(pattern)))
    if candidates:
        return candidates[0]
    return session_dir / "public" / "audio" / f"turn_{turn_id:04d}_rev{revision_id:02d}.mp3"


def _require_int(record: dict[str, Any], key: str, source_path: Path) -> int:
    value = record.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ReplayLoadError(f"Replay逐語録に整数の{key}がありません: {source_path}")
    return value
