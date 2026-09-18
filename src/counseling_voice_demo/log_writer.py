from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable

from counseling_voice_demo.canonical_history import build_canonical_history, is_turn_eligible_for_canonical_history
from counseling_voice_demo.models import SessionState, Turn, TurnRevision, utc_now


@dataclass(frozen=True)
class SessionLogPaths:
    session_dir: Path
    public_dir: Path
    internal_dir: Path
    public_transcript_jsonl: Path
    public_transcript_md: Path
    evaluation_public_json: Path
    public_audio_dir: Path
    public_merged_dir: Path
    session_config_json: Path
    internal_transcript_jsonl: Path
    internal_transcript_md: Path
    turn_events_jsonl: Path
    interventions_jsonl: Path
    evaluation_internal_json: Path
    safety_log_jsonl: Path
    clinical_ng_log_jsonl: Path
    errors_jsonl: Path
    prompts_dir: Path
    audio_all_revisions_dir: Path


@dataclass(frozen=True)
class LogWriteResult:
    success: bool
    warning: str | None = None
    error: Exception | None = None


def create_session_log_dirs(
    sessions_dir: Path | str,
    session_id: str,
    *,
    now: datetime | None = None,
) -> SessionLogPaths:
    timestamp = (now or utc_now()).astimezone(timezone.utc).strftime("%Y%m%d_%H%M%S")
    session_dir = Path(sessions_dir) / f"{timestamp}_{session_id}"
    public_dir = session_dir / "public"
    internal_dir = session_dir / "internal"

    paths = SessionLogPaths(
        session_dir=session_dir,
        public_dir=public_dir,
        internal_dir=internal_dir,
        public_transcript_jsonl=public_dir / "transcript_public.jsonl",
        public_transcript_md=public_dir / "transcript_public.md",
        evaluation_public_json=public_dir / "evaluation_public.json",
        public_audio_dir=public_dir / "audio",
        public_merged_dir=public_dir / "merged",
        session_config_json=internal_dir / "session_config.json",
        internal_transcript_jsonl=internal_dir / "transcript_full.jsonl",
        internal_transcript_md=internal_dir / "transcript_full.md",
        turn_events_jsonl=internal_dir / "turn_events.jsonl",
        interventions_jsonl=internal_dir / "interventions.jsonl",
        evaluation_internal_json=internal_dir / "evaluation_internal.json",
        safety_log_jsonl=internal_dir / "safety_log.jsonl",
        clinical_ng_log_jsonl=internal_dir / "clinical_ng_log.jsonl",
        errors_jsonl=internal_dir / "errors.jsonl",
        prompts_dir=internal_dir / "prompts",
        audio_all_revisions_dir=internal_dir / "audio_all_revisions",
    )

    for directory in [
        paths.public_dir,
        paths.internal_dir,
        paths.public_audio_dir,
        paths.public_merged_dir,
        paths.prompts_dir,
        paths.audio_all_revisions_dir,
    ]:
        directory.mkdir(parents=True, exist_ok=True)

    for file_path in [
        paths.public_transcript_jsonl,
        paths.public_transcript_md,
        paths.internal_transcript_jsonl,
        paths.internal_transcript_md,
        paths.turn_events_jsonl,
        paths.interventions_jsonl,
        paths.safety_log_jsonl,
        paths.clinical_ng_log_jsonl,
        paths.errors_jsonl,
    ]:
        file_path.touch(exist_ok=True)

    return paths


def write_session_config(paths: SessionLogPaths, config: dict[str, Any]) -> None:
    _write_json(paths.session_config_json, config)


def write_evaluation_result(
    paths: SessionLogPaths,
    *,
    public: bool,
    record: dict[str, Any],
) -> None:
    target_path = (
        paths.evaluation_public_json if public else paths.evaluation_internal_json
    )
    _write_json(target_path, record)


def append_turn_transcripts(paths: SessionLogPaths, turn: Turn) -> None:
    _append_jsonl(paths.internal_transcript_jsonl, _turn_internal_record(turn))
    if is_turn_eligible_for_canonical_history(turn):
        _append_jsonl(paths.public_transcript_jsonl, _turn_public_record(turn))


def update_transcript_markdown(paths: SessionLogPaths, session: SessionState) -> None:
    public_lines = [f"# Transcript: {session.session_id}", ""]
    for item in build_canonical_history(session):
        public_lines.append(f"- {item.turn_id}. {item.speaker_name} ({item.speaker_role.value}): {item.text}")

    internal_lines = [f"# Full Transcript: {session.session_id}", ""]
    for turn in sorted(session.turns, key=lambda item: item.turn_id):
        internal_lines.append(
            f"## Turn {turn.turn_id}: {turn.speaker_name} ({turn.speaker_role.value}) "
            f"[{turn.status.value}]"
        )
        for revision in sorted(turn.revisions, key=lambda item: item.revision_id):
            active_marker = "active" if revision.active else "inactive"
            internal_lines.append(f"- r{revision.revision_id} {active_marker}: {revision.canonical_text}")
            if revision.invalidated_reason:
                internal_lines.append(f"  - invalidated_reason: {revision.invalidated_reason}")

    paths.public_transcript_md.write_text("\n".join(public_lines).rstrip() + "\n", encoding="utf-8")
    paths.internal_transcript_md.write_text("\n".join(internal_lines).rstrip() + "\n", encoding="utf-8")


def append_turn_event(
    paths: SessionLogPaths,
    *,
    event_type: str,
    turn: Turn,
    details: dict[str, Any] | None = None,
    at: datetime | None = None,
) -> None:
    _append_jsonl(
        paths.turn_events_jsonl,
        {
            "event_type": event_type,
            "recorded_at": at or utc_now(),
            "session_id": turn.session_id,
            "turn_id": turn.turn_id,
            "active_revision_id": turn.active_revision_id,
            "status": turn.status,
            "playback_status": turn.playback_status,
            "details": details or {},
        },
    )


def append_intervention(
    paths: SessionLogPaths,
    *,
    action: str,
    details: dict[str, Any] | None = None,
    at: datetime | None = None,
) -> None:
    _append_jsonl(
        paths.interventions_jsonl,
        {
            "action": action,
            "recorded_at": at or utc_now(),
            "details": details or {},
        },
    )


def append_safety_log(paths: SessionLogPaths, record: dict[str, Any]) -> None:
    _append_jsonl(paths.safety_log_jsonl, record)


def append_clinical_ng_log(paths: SessionLogPaths, record: dict[str, Any]) -> None:
    _append_jsonl(paths.clinical_ng_log_jsonl, record)


def append_error_log(paths: SessionLogPaths, record: dict[str, Any]) -> None:
    _append_jsonl(paths.errors_jsonl, record)


def safe_log_write(write_func: Callable[[], None]) -> LogWriteResult:
    try:
        write_func()
    except OSError as exc:
        return LogWriteResult(
            success=False,
            warning=f"ログ保存に失敗しました: {exc}",
            error=exc,
        )
    return LogWriteResult(success=True)


def _turn_public_record(turn: Turn) -> dict[str, Any]:
    revision = turn.active_revision()
    return {
        "session_id": turn.session_id,
        "turn_id": turn.turn_id,
        "revision_id": revision.revision_id,
        "speaker_role": turn.speaker_role,
        "speaker_name": turn.speaker_name,
        "text": revision.canonical_text,
        "status": turn.status,
        "playback_status": turn.playback_status,
        "warning_level": turn.warning_level,
        "spoken_at": revision.spoken_at,
        "actual_played_seconds": revision.actual_played_seconds,
        "skip_status": revision.skip_status,
    }


def _turn_internal_record(turn: Turn) -> dict[str, Any]:
    return {
        "session_id": turn.session_id,
        "turn_id": turn.turn_id,
        "speaker_role": turn.speaker_role,
        "speaker_name": turn.speaker_name,
        "profile_id": turn.profile_id,
        "active_revision_id": turn.active_revision_id,
        "status": turn.status,
        "warning_level": turn.warning_level,
        "playback_status": turn.playback_status,
        "created_at": turn.created_at,
        "updated_at": turn.updated_at,
        "revisions": [_revision_internal_record(revision) for revision in turn.revisions],
    }


def _revision_internal_record(revision: TurnRevision) -> dict[str, Any]:
    return {
        "turn_id": revision.turn_id,
        "revision_id": revision.revision_id,
        "text_original": revision.text_original,
        "text_final": revision.text_final,
        "canonical_text": revision.canonical_text,
        "edited": revision.edited,
        "regenerated": revision.regenerated,
        "active": revision.active,
        "invalidated_reason": revision.invalidated_reason,
        "audio_path": revision.audio_path,
        "audio_format": revision.audio_format,
        "moderation_result": revision.moderation_result,
        "clinical_ng_result": revision.clinical_ng_result,
        "warning_level": revision.warning_level,
        "hold_reason": revision.hold_reason,
        "generated_at": revision.generated_at,
        "tts_generated_at": revision.tts_generated_at,
        "spoken_at": revision.spoken_at,
        "played_audio_elapsed_seconds_start": revision.played_audio_elapsed_seconds_start,
        "played_audio_elapsed_seconds_end": revision.played_audio_elapsed_seconds_end,
        "actual_played_seconds": revision.actual_played_seconds,
        "skip_status": revision.skip_status,
    }


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(_jsonable(record), ensure_ascii=False, sort_keys=True) + "\n")


def _write_json(path: Path, record: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(_jsonable(record), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _jsonable(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    return value
