from __future__ import annotations

import json

import pytest

from counseling_voice_demo.log_writer import create_session_log_dirs
from counseling_voice_demo.playback_controller import (
    PlaybackEvent,
    PlaybackOutcome,
    PlaybackQueueDecision,
    add_elapsed_once,
    release_playback_hold,
    next_playback_decision,
    record_playback_control_event,
    record_playback_terminal_event,
    skip_unplayed_hold,
    summarize_playback_event,
)
from counseling_voice_demo.models import SessionState, SpeakerRole, TurnStatus, WarningLevel


def test_ended_counts_full_duration_as_played() -> None:
    summary = summarize_playback_event(
        PlaybackEvent(event_name="ended", current_seconds=1.18, duration_seconds=1.2)
    )

    assert summary.outcome == PlaybackOutcome.PLAYED
    assert summary.elapsed_audio_seconds == 1.2


def test_skip_before_playback_is_skipped_unplayed() -> None:
    summary = summarize_playback_event(
        PlaybackEvent(event_name="skip", current_seconds=0.0, duration_seconds=1.2)
    )

    assert summary.outcome == PlaybackOutcome.SKIPPED_UNPLAYED
    assert summary.elapsed_audio_seconds == 0.0


def test_skip_after_playback_started_counts_actual_seconds_only() -> None:
    summary = summarize_playback_event(
        PlaybackEvent(event_name="skip", current_seconds=0.4, duration_seconds=1.2)
    )

    assert summary.outcome == PlaybackOutcome.SKIPPED_PARTIAL
    assert summary.elapsed_audio_seconds == 0.4


def test_elapsed_is_not_added_twice_after_pause_resume() -> None:
    summary = summarize_playback_event(
        PlaybackEvent(event_name="ended", current_seconds=1.2, duration_seconds=1.2)
    )

    total = add_elapsed_once(10.0, summary, already_counted=False)
    total = add_elapsed_once(total, summary, already_counted=True)

    assert total == 11.2


def test_unknown_terminal_event_fails_explicitly() -> None:
    with pytest.raises(ValueError, match="Unsupported playback terminal event"):
        summarize_playback_event(
            PlaybackEvent(event_name="pause", current_seconds=0.2, duration_seconds=1.2)
        )


def test_next_playback_decision_uses_lowest_turn_id_with_audio_ready() -> None:
    session = SessionState()
    later = session.add_turn(
        speaker_role=SpeakerRole.CLIENT,
        speaker_name="高橋",
        profile_id="client",
        text="少し困っています。",
        status=TurnStatus.AUDIO_READY,
    )
    earlier = session.add_turn(
        speaker_role=SpeakerRole.COUNSELOR,
        speaker_name="佐伯",
        profile_id="counselor",
        text="もう少し聞かせてください。",
        status=TurnStatus.AUDIO_READY,
    )
    later.turn_id = 2
    earlier.turn_id = 1
    later.active_revision().audio_path = "turn_0002.mp3"
    earlier.active_revision().audio_path = "turn_0001.mp3"

    decision = next_playback_decision(session)

    assert decision == PlaybackQueueDecision(action="play", turn=earlier, reason="audio_ready")


def test_next_playback_decision_stops_on_playback_hold() -> None:
    session = SessionState()
    held = session.add_turn(
        speaker_role=SpeakerRole.CLIENT,
        speaker_name="高橋",
        profile_id="client",
        text="確認が必要です。",
        status=TurnStatus.PLAYBACK_HOLD,
    )
    following = session.add_turn(
        speaker_role=SpeakerRole.COUNSELOR,
        speaker_name="佐伯",
        profile_id="counselor",
        text="後続発話です。",
        status=TurnStatus.AUDIO_READY,
    )
    held.active_revision().audio_path = "turn_0001.mp3"
    following.active_revision().audio_path = "turn_0002.mp3"

    decision = next_playback_decision(session)

    assert decision.action == "hold"
    assert decision.turn is held
    assert decision.reason == "playback_hold"


def test_release_playback_hold_allows_audio_ready_turn_to_play(tmp_path) -> None:
    paths = create_session_log_dirs(tmp_path, "session_hold_release")
    session = SessionState(session_id="session_hold_release")
    held = session.add_turn(
        speaker_role=SpeakerRole.CLIENT,
        speaker_name="高橋",
        profile_id="client",
        text="確認済みとして再生します。",
        status=TurnStatus.PLAYBACK_HOLD,
    )
    held.warning_level = WarningLevel.MEDIUM
    held.active_revision().warning_level = WarningLevel.MEDIUM
    held.active_revision().hold_reason = "clinical_ng_medium"
    held.active_revision().audio_path = "turn_0001.mp3"

    release_playback_hold(held, log_paths=paths)

    assert held.status == TurnStatus.AUDIO_READY
    assert held.playback_status == TurnStatus.AUDIO_READY
    assert next_playback_decision(session).action == "play"
    records = _read_jsonl(paths.turn_events_jsonl)
    assert records[0]["event_type"] == "playback_hold_released"


def test_skip_unplayed_hold_marks_turn_skipped_unplayed(tmp_path) -> None:
    paths = create_session_log_dirs(tmp_path, "session_hold_skip")
    session = SessionState(session_id="session_hold_skip")
    held = session.add_turn(
        speaker_role=SpeakerRole.CLIENT,
        speaker_name="高橋",
        profile_id="client",
        text="再生しません。",
        status=TurnStatus.PLAYBACK_HOLD,
    )

    skip_unplayed_hold(held, log_paths=paths)

    assert held.status == TurnStatus.SKIPPED_UNPLAYED
    assert held.playback_status == TurnStatus.SKIPPED_UNPLAYED
    assert next_playback_decision(session).action == "complete"
    records = _read_jsonl(paths.turn_events_jsonl)
    assert records[0]["event_type"] == "playback_hold_skipped"


def test_record_playback_terminal_event_updates_turn_and_elapsed_seconds() -> None:
    session = SessionState()
    turn = session.add_turn(
        speaker_role=SpeakerRole.COUNSELOR,
        speaker_name="佐伯",
        profile_id="counselor",
        text="開始します。",
        status=TurnStatus.AUDIO_READY,
    )

    total = record_playback_terminal_event(
        turn,
        PlaybackEvent(event_name="skip", current_seconds=0.4, duration_seconds=1.2),
        previous_total_seconds=10.0,
        already_counted=False,
    )

    assert total == 10.4
    assert turn.status == TurnStatus.SKIPPED_PARTIAL
    assert turn.playback_status == TurnStatus.SKIPPED_PARTIAL
    assert turn.active_revision().actual_played_seconds == 0.4


def test_playback_events_are_recorded_to_turn_events_log(tmp_path) -> None:
    paths = create_session_log_dirs(tmp_path, "session_playback_events")
    session = SessionState(session_id="session_playback_events")
    turn = session.add_turn(
        speaker_role=SpeakerRole.COUNSELOR,
        speaker_name="佐伯",
        profile_id="counselor",
        text="開始します。",
        status=TurnStatus.AUDIO_READY,
    )

    record_playback_control_event(paths, turn=turn, event_name="started", current_seconds=0.0)
    record_playback_control_event(paths, turn=turn, event_name="pause", current_seconds=0.3)
    record_playback_control_event(paths, turn=turn, event_name="resume", current_seconds=0.3)
    record_playback_terminal_event(
        turn,
        PlaybackEvent(event_name="ended", current_seconds=1.2, duration_seconds=1.2),
        previous_total_seconds=0.0,
        already_counted=False,
        log_paths=paths,
    )

    records = [
        json.loads(line)
        for line in paths.turn_events_jsonl.read_text(encoding="utf-8").splitlines()
    ]
    assert [record["event_type"] for record in records] == [
        "playback_started",
        "playback_pause",
        "playback_resume",
        "playback_ended",
    ]
    assert turn.status == TurnStatus.PLAYED
    assert turn.active_revision().spoken_at is not None


def _read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
