from __future__ import annotations

import json
from datetime import datetime, timezone

from counseling_voice_demo.log_writer import (
    append_intervention,
    append_turn_event,
    append_turn_transcripts,
    create_session_log_dirs,
    safe_log_write,
    update_transcript_markdown,
    write_evaluation_result,
    write_session_config,
)
from counseling_voice_demo.models import SessionState, SpeakerRole, TurnStatus, WarningLevel


def _read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _tree_text(path) -> str:
    return "\n".join(
        file_path.read_text(encoding="utf-8")
        for file_path in sorted(path.rglob("*"))
        if file_path.is_file()
    )


def test_create_session_log_dirs_prepares_public_and_internal_files(tmp_path) -> None:
    paths = create_session_log_dirs(
        sessions_dir=tmp_path,
        session_id="session_abcd1234",
        now=datetime(2026, 4, 24, 15, 30, tzinfo=timezone.utc),
    )

    assert paths.session_dir == tmp_path / "20260424_153000_session_abcd1234"
    assert paths.public_dir.is_dir()
    assert paths.internal_dir.is_dir()
    assert (paths.public_dir / "audio").is_dir()
    assert (paths.public_dir / "merged").is_dir()
    assert (paths.internal_dir / "prompts").is_dir()
    assert (paths.internal_dir / "audio_all_revisions").is_dir()
    assert paths.public_transcript_jsonl.exists()
    assert paths.internal_transcript_jsonl.exists()
    assert paths.turn_events_jsonl.exists()
    assert paths.interventions_jsonl.exists()
    assert paths.errors_jsonl.exists()


def test_write_session_config_is_internal_only(tmp_path) -> None:
    paths = create_session_log_dirs(tmp_path, "session_config_test")

    write_session_config(
        paths,
        {
            "default_text_model": "gpt-test",
            "client_hidden_background": "家族との距離感に悩んでいる",
        },
    )

    assert "家族との距離感に悩んでいる" in paths.session_config_json.read_text(encoding="utf-8")
    assert "家族との距離感に悩んでいる" not in _tree_text(paths.public_dir)


def test_write_evaluation_result_splits_public_and_internal_files(tmp_path) -> None:
    paths = create_session_log_dirs(tmp_path, "session_evaluation_test")
    internal_only_text = "公開してはいけない内部評価メモ"

    write_evaluation_result(
        paths,
        public=True,
        record={
            "summary": "聴衆向け評価",
            "generated_at": datetime(2026, 4, 24, 16, 45, tzinfo=timezone.utc),
            "max_warning_level": WarningLevel.LOW,
            "artifact_path": paths.public_transcript_jsonl,
        },
    )
    write_evaluation_result(
        paths,
        public=False,
        record={
            "summary": "内部改善用評価",
            "internal_note": internal_only_text,
        },
    )

    public_record = json.loads(paths.evaluation_public_json.read_text(encoding="utf-8"))
    internal_record = json.loads(paths.evaluation_internal_json.read_text(encoding="utf-8"))

    assert public_record == {
        "artifact_path": str(paths.public_transcript_jsonl),
        "generated_at": "2026-04-24T16:45:00+00:00",
        "max_warning_level": "low",
        "summary": "聴衆向け評価",
    }
    assert internal_record["internal_note"] == internal_only_text
    assert internal_only_text not in _tree_text(paths.public_dir)


def test_append_turn_transcripts_splits_public_and_internal_fields(tmp_path) -> None:
    paths = create_session_log_dirs(tmp_path, "session_transcript_test")
    session = SessionState(session_id="session_transcript_test")
    turn = session.add_turn(
        speaker_role=SpeakerRole.CLIENT,
        speaker_name="高橋",
        profile_id="client_family_default",
        text="家族のことで悩んでいます。",
    )
    revision = turn.active_revision()
    revision.moderation_result = {"flagged": False, "hidden_background": "公開してはいけない内部情報"}
    revision.clinical_ng_result = {"warning": "diagnosis_like"}
    revision.warning_level = WarningLevel.MEDIUM
    revision.hold_reason = "clinical_review"
    turn.status = TurnStatus.PLAYBACK_HOLD
    turn.warning_level = WarningLevel.MEDIUM

    append_turn_transcripts(paths, turn)

    public_records = _read_jsonl(paths.public_transcript_jsonl)
    internal_records = _read_jsonl(paths.internal_transcript_jsonl)
    public_text = paths.public_transcript_jsonl.read_text(encoding="utf-8")
    internal_text = paths.internal_transcript_jsonl.read_text(encoding="utf-8")

    assert public_records == [
        {
            "session_id": "session_transcript_test",
            "turn_id": 1,
            "revision_id": 1,
            "speaker_role": "client",
            "speaker_name": "高橋",
            "text": "家族のことで悩んでいます。",
            "status": "playback_hold",
            "playback_status": "text_ready",
            "warning_level": "medium",
            "spoken_at": None,
            "actual_played_seconds": 0.0,
            "skip_status": "none",
        }
    ]
    assert internal_records[0]["profile_id"] == "client_family_default"
    assert internal_records[0]["revisions"][0]["moderation_result"]["hidden_background"] == "公開してはいけない内部情報"
    assert "公開してはいけない内部情報" not in public_text
    assert "公開してはいけない内部情報" in internal_text


def test_invalidated_turn_is_kept_internal_but_not_public(tmp_path) -> None:
    paths = create_session_log_dirs(tmp_path, "session_invalidated_test")
    session = SessionState(session_id="session_invalidated_test")
    turn = session.add_turn(
        speaker_role=SpeakerRole.COUNSELOR,
        speaker_name="佐伯",
        profile_id="counselor_default",
        text="無効化される発話です。",
    )
    turn.invalidate("edit_invalidated_following_turn")

    append_turn_transcripts(paths, turn)

    assert _read_jsonl(paths.public_transcript_jsonl) == []
    internal_records = _read_jsonl(paths.internal_transcript_jsonl)
    assert internal_records[0]["status"] == "invalidated"
    assert internal_records[0]["revisions"][0]["invalidated_reason"] == "edit_invalidated_following_turn"


def test_turn_events_and_interventions_are_jsonl(tmp_path) -> None:
    paths = create_session_log_dirs(tmp_path, "session_event_test")
    session = SessionState(session_id="session_event_test")
    turn = session.add_turn(
        speaker_role=SpeakerRole.COUNSELOR,
        speaker_name="佐伯",
        profile_id="counselor_default",
        text="開始します。",
    )

    append_turn_event(paths, event_type="status_changed", turn=turn, details={"to": "text_ready"})
    append_intervention(paths, action="pause", details={"reason": "operator_pause"})

    assert _read_jsonl(paths.turn_events_jsonl)[0]["event_type"] == "status_changed"
    assert _read_jsonl(paths.interventions_jsonl)[0]["action"] == "pause"


def test_update_transcript_markdown_writes_public_and_full_versions(tmp_path) -> None:
    paths = create_session_log_dirs(tmp_path, "session_markdown_test")
    session = SessionState(session_id="session_markdown_test")
    session.add_turn(
        speaker_role=SpeakerRole.COUNSELOR,
        speaker_name="佐伯",
        profile_id="counselor_default",
        text="今日はどんなことを話したいですか。",
    )
    invalidated_turn = session.add_turn(
        speaker_role=SpeakerRole.CLIENT,
        speaker_name="高橋",
        profile_id="client_default",
        text="これは公開しない発話です。",
    )
    invalidated_turn.invalidate("regenerated")

    update_transcript_markdown(paths, session)

    public_md = paths.public_transcript_md.read_text(encoding="utf-8")
    full_md = paths.internal_transcript_md.read_text(encoding="utf-8")
    assert "今日はどんなことを話したいですか。" in public_md
    assert "これは公開しない発話です。" not in public_md
    assert "これは公開しない発話です。" in full_md


def test_safe_log_write_returns_warning_instead_of_raising() -> None:
    def failing_write() -> None:
        raise OSError("disk is full")

    result = safe_log_write(failing_write)

    assert result.success is False
    assert "ログ保存に失敗しました" in result.warning
