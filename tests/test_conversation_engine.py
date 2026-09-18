from __future__ import annotations

from counseling_voice_demo.content_loader import Profile, Theme
from counseling_voice_demo.conversation_engine import (
    ConversationEngine,
    ConversationEngineConfig,
    ConversationParticipants,
    invalidate_unplayed_speaker_sequence_violations,
)
from counseling_voice_demo.models import (
    ConversationPhase,
    SessionState,
    SessionStatus,
    SpeakerRole,
    TurnStatus,
    WarningLevel,
)


class _FakeTextClient:
    def __init__(self) -> None:
        self.calls = []
        self.after_call = None

    def generate_turn(self, request):
        self.calls.append(request)
        if self.after_call is not None:
            self.after_call()
        text = f"{request.speaker_profile.display_name}の発話{len(self.calls)}"
        return type("Result", (), {"text": text, "raw_response": {"id": f"resp_{len(self.calls)}"}})()


class _FakeSafetyRunner:
    def __init__(self) -> None:
        self.calls = []

    def check_turn(self, turn):
        self.calls.append((turn.turn_id, turn.speaker_role))


def test_fill_generation_queue_creates_two_lookahead_turns_and_tts_candidates() -> None:
    session = SessionState(status=SessionStatus.RUNNING)
    text_client = _FakeTextClient()
    engine = ConversationEngine(
        text_client=text_client,
        participants=_participants(),
        config=ConversationEngineConfig(ahead_generation_turns=2, reasoning_effort="low"),
    )

    generated = engine.fill_generation_queue(session, cumulative_audio_seconds=0)

    assert [turn.speaker_role for turn in generated] == [SpeakerRole.COUNSELOR, SpeakerRole.CLIENT]
    assert [turn.status for turn in generated] == [TurnStatus.TEXT_READY, TurnStatus.TEXT_READY]
    assert [turn.turn_id for turn in generated] == [1, 2]
    assert [request.speaker_profile.profile_id for request in text_client.calls] == [
        "counselor_test",
        "client_test",
    ]
    assert [request.reasoning_effort for request in text_client.calls] == ["low", "low"]
    assert [request.max_output_tokens for request in text_client.calls] == [800, 800]


def test_fill_generation_queue_runs_safety_checks_for_both_speakers() -> None:
    session = SessionState(status=SessionStatus.RUNNING)
    safety_runner = _FakeSafetyRunner()
    engine = ConversationEngine(
        text_client=_FakeTextClient(),
        participants=_participants(),
        config=ConversationEngineConfig(ahead_generation_turns=2),
        safety_check_runner=safety_runner,
    )

    generated = engine.fill_generation_queue(session, cumulative_audio_seconds=0)

    assert [turn.speaker_role for turn in generated] == [SpeakerRole.COUNSELOR, SpeakerRole.CLIENT]
    assert safety_runner.calls == [
        (generated[0].turn_id, SpeakerRole.COUNSELOR),
        (generated[1].turn_id, SpeakerRole.CLIENT),
    ]


def test_generation_result_completed_after_pause_is_marked_paused_generated() -> None:
    session = SessionState(status=SessionStatus.RUNNING)
    text_client = _FakeTextClient()
    text_client.after_call = lambda: setattr(session, "status", SessionStatus.PAUSED)
    engine = ConversationEngine(
        text_client=text_client,
        participants=_participants(),
        config=ConversationEngineConfig(ahead_generation_turns=2),
    )

    turn = engine.generate_next_turn(session, cumulative_audio_seconds=0)

    assert turn is not None
    assert turn.status == TurnStatus.PAUSED_GENERATED
    assert turn.playback_status == TurnStatus.PAUSED_GENERATED


def test_generation_result_completed_after_stop_is_ignored_after_stop() -> None:
    session = SessionState(status=SessionStatus.RUNNING)
    text_client = _FakeTextClient()
    text_client.after_call = lambda: setattr(session, "status", SessionStatus.STOPPED)
    engine = ConversationEngine(
        text_client=text_client,
        participants=_participants(),
        config=ConversationEngineConfig(ahead_generation_turns=2),
    )

    turn = engine.generate_next_turn(session, cumulative_audio_seconds=0)

    assert turn is not None
    assert turn.status == TurnStatus.IGNORED_AFTER_STOP
    assert turn.playback_status == TurnStatus.IGNORED_AFTER_STOP
    assert turn.active_revision_id == 1
    assert turn.revisions[0].active is False


def test_generate_next_turn_rechecks_next_speaker_after_concurrent_turn_added() -> None:
    session = SessionState(status=SessionStatus.RUNNING)
    text_client = _FakeTextClient()

    def add_competing_turn_once() -> None:
        text_client.after_call = None
        session.add_turn(
            speaker_role=SpeakerRole.COUNSELOR,
            speaker_name="佐伯",
            profile_id="counselor_test",
            text="別の進行で追加済みのカウンセラー発話です。",
            status=TurnStatus.TEXT_READY,
        )

    text_client.after_call = add_competing_turn_once
    engine = ConversationEngine(
        text_client=text_client,
        participants=_participants(),
        config=ConversationEngineConfig(ahead_generation_turns=2),
    )

    turn = engine.generate_next_turn(session, cumulative_audio_seconds=0)

    assert turn is not None
    assert turn.speaker_role == SpeakerRole.CLIENT
    assert [request.speaker_profile.role for request in text_client.calls] == [
        "counselor",
        "client",
    ]
    assert [item.speaker_role for item in session.turns] == [
        SpeakerRole.COUNSELOR,
        SpeakerRole.CLIENT,
    ]


def test_generate_next_turn_does_not_add_stale_result_after_retry_limit() -> None:
    session = SessionState(status=SessionStatus.RUNNING)
    text_client = _FakeTextClient()

    def add_matching_turn_after_each_request() -> None:
        request = text_client.calls[-1]
        session.add_turn(
            speaker_role=SpeakerRole(request.speaker_profile.role),
            speaker_name=request.speaker_profile.display_name,
            profile_id=request.speaker_profile.profile_id,
            text="別の進行で追加済みの発話です。",
            status=TurnStatus.TEXT_READY,
        )

    text_client.after_call = add_matching_turn_after_each_request
    engine = ConversationEngine(
        text_client=text_client,
        participants=_participants(),
        config=ConversationEngineConfig(ahead_generation_turns=2),
    )

    turn = engine.generate_next_turn(session, cumulative_audio_seconds=0)

    assert turn is None
    assert [request.speaker_profile.role for request in text_client.calls] == [
        "counselor",
        "client",
        "counselor",
    ]
    assert [item.speaker_role for item in session.turns] == [
        SpeakerRole.COUNSELOR,
        SpeakerRole.CLIENT,
        SpeakerRole.COUNSELOR,
    ]


def test_invalidate_unplayed_speaker_sequence_violations_keeps_played_history() -> None:
    session = SessionState(status=SessionStatus.RUNNING)
    session.add_turn(
        speaker_role=SpeakerRole.COUNSELOR,
        speaker_name="佐伯",
        profile_id="counselor_test",
        text="再生済みのカウンセラー発話です。",
        status=TurnStatus.PLAYED,
    )
    session.add_turn(
        speaker_role=SpeakerRole.CLIENT,
        speaker_name="高橋",
        profile_id="client_test",
        text="再生済みのクライアント発話です。",
        status=TurnStatus.PLAYED,
    )
    duplicate = session.add_turn(
        speaker_role=SpeakerRole.CLIENT,
        speaker_name="高橋",
        profile_id="client_test",
        text="連続してしまった未再生クライアント発話です。",
        status=TurnStatus.AUDIO_READY,
    )
    following = session.add_turn(
        speaker_role=SpeakerRole.COUNSELOR,
        speaker_name="佐伯",
        profile_id="counselor_test",
        text="次のカウンセラー発話です。",
        status=TurnStatus.AUDIO_READY,
    )

    invalidated = invalidate_unplayed_speaker_sequence_violations(session)

    assert invalidated == [duplicate]
    assert duplicate.status == TurnStatus.INVALIDATED
    assert duplicate.revisions[0].active is False
    assert following.status == TurnStatus.AUDIO_READY


def test_fill_generation_queue_repairs_duplicate_unplayed_speaker_before_generation() -> None:
    session = SessionState(status=SessionStatus.RUNNING)
    session.add_turn(
        speaker_role=SpeakerRole.COUNSELOR,
        speaker_name="佐伯",
        profile_id="counselor_test",
        text="再生済みのカウンセラー発話です。",
        status=TurnStatus.PLAYED,
    )
    session.add_turn(
        speaker_role=SpeakerRole.CLIENT,
        speaker_name="高橋",
        profile_id="client_test",
        text="再生済みのクライアント発話です。",
        status=TurnStatus.PLAYED,
    )
    duplicate = session.add_turn(
        speaker_role=SpeakerRole.CLIENT,
        speaker_name="高橋",
        profile_id="client_test",
        text="連続してしまった未再生クライアント発話です。",
        status=TurnStatus.AUDIO_READY,
    )
    engine = ConversationEngine(
        text_client=_FakeTextClient(),
        participants=_participants(),
        config=ConversationEngineConfig(ahead_generation_turns=2),
    )

    generated = engine.fill_generation_queue(session, cumulative_audio_seconds=20)

    assert duplicate.status == TurnStatus.INVALIDATED
    assert [turn.speaker_role for turn in generated] == [
        SpeakerRole.COUNSELOR,
        SpeakerRole.CLIENT,
    ]


def test_closing_invalidates_unplayed_turns_after_keep_limit_and_farewell_after_two_turns() -> None:
    session = SessionState(status=SessionStatus.RUNNING)
    for index in range(3):
        session.add_turn(
            speaker_role=SpeakerRole.COUNSELOR if index % 2 == 0 else SpeakerRole.CLIENT,
            speaker_name="話者",
            profile_id="profile",
            text=f"未再生{index}",
            status=TurnStatus.AUDIO_READY,
        )
    engine = ConversationEngine(
        text_client=_FakeTextClient(),
        participants=_participants(),
        config=ConversationEngineConfig(
            ahead_generation_turns=2,
            closing_start_audio_seconds=900,
            farewell_after_turns=2,
            closing_keep_audio_ready_turns=1,
        ),
    )

    engine.update_conversation_phase(session, cumulative_audio_seconds=900)

    assert session.conversation_phase == ConversationPhase.CLOSING
    assert session.turns[0].status == TurnStatus.AUDIO_READY
    assert [turn.status for turn in session.turns[1:]] == [TurnStatus.INVALIDATED, TurnStatus.INVALIDATED]

    session.add_turn(
        speaker_role=SpeakerRole.COUNSELOR,
        speaker_name="佐伯",
        profile_id="counselor",
        text="まとめに入ります。",
    )
    session.add_turn(
        speaker_role=SpeakerRole.CLIENT,
        speaker_name="高橋",
        profile_id="client",
        text="わかりました。",
    )
    engine.update_conversation_phase(session, cumulative_audio_seconds=920)

    assert session.conversation_phase == ConversationPhase.FAREWELL


def test_closing_and_farewell_instructions_are_added_to_requests() -> None:
    session = SessionState(status=SessionStatus.RUNNING, conversation_phase=ConversationPhase.CLOSING)
    text_client = _FakeTextClient()
    engine = ConversationEngine(
        text_client=text_client,
        participants=_participants(),
        config=ConversationEngineConfig(ahead_generation_turns=2),
    )

    engine.generate_next_turn(session, cumulative_audio_seconds=901)

    assert "終結" in (text_client.calls[0].closing_instruction or "")

    session.add_turn(
        speaker_role=SpeakerRole.CLIENT,
        speaker_name="高橋",
        profile_id="client_test",
        text="わかりました。",
    )
    session.conversation_phase = ConversationPhase.FAREWELL
    engine.generate_next_turn(session, cumulative_audio_seconds=930)

    assert "お別れ" in (text_client.calls[1].farewell_instruction or "")


def test_high_warning_client_turn_adds_safety_instruction_to_next_counselor_without_stopping() -> None:
    session = SessionState(status=SessionStatus.RUNNING)
    client_turn = session.add_turn(
        speaker_role=SpeakerRole.CLIENT,
        speaker_name="高橋",
        profile_id="client_test",
        text="もう消えてしまいたいです。",
    )
    client_turn.warning_level = WarningLevel.HIGH
    client_turn.active_revision().warning_level = WarningLevel.HIGH
    text_client = _FakeTextClient()
    engine = ConversationEngine(
        text_client=text_client,
        participants=_participants(),
        config=ConversationEngineConfig(ahead_generation_turns=2),
    )

    engine.generate_next_turn(session, cumulative_audio_seconds=100)

    assert session.status == SessionStatus.RUNNING
    assert "安全確認" in (text_client.calls[0].safety_instruction or "")


def test_farewell_client_response_can_complete_session() -> None:
    session = SessionState(status=SessionStatus.RUNNING, conversation_phase=ConversationPhase.FAREWELL)
    session.add_turn(
        speaker_role=SpeakerRole.COUNSELOR,
        speaker_name="佐伯",
        profile_id="counselor_test",
        text="今日はここまでにしましょう。",
    )
    engine = ConversationEngine(
        text_client=_FakeTextClient(),
        participants=_participants(),
        config=ConversationEngineConfig(ahead_generation_turns=2),
    )

    turn = engine.generate_next_turn(session, cumulative_audio_seconds=930)

    assert turn is not None
    assert turn.speaker_role == SpeakerRole.CLIENT
    assert session.status == SessionStatus.COMPLETED


def _participants() -> ConversationParticipants:
    return ConversationParticipants(
        counselor_profile=Profile(
            profile_id="counselor_test",
            role="counselor",
            display_name="佐伯",
            text_model="gpt-test",
            temperature=0.4,
            voice_preset="counselor",
            tts_model="gpt-4o-mini-tts",
            tts_voice="coral",
            tts_instructions="落ち着いて話す。",
            prompt="あなたはカウンセラー役です。",
        ),
        client_profile=Profile(
            profile_id="client_test",
            role="client",
            display_name="高橋",
            text_model="gpt-test",
            temperature=0.6,
            voice_preset="client",
            tts_model="gpt-4o-mini-tts",
            tts_voice="alloy",
            tts_instructions="少し不安そうに話す。",
            prompt="あなたはクライアント役です。",
            hidden_background="家族に言えていない背景",
        ),
        theme=Theme(
            theme_id="family_conflict",
            display_name="家族関係の葛藤",
            severity="standard",
            default_client_profile="client_family_default",
            body="家族との距離感に悩む場面。",
        ),
    )
