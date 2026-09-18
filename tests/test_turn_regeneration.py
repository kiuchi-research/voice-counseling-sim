from __future__ import annotations

import json

from counseling_voice_demo.content_loader import Profile, Theme
from counseling_voice_demo.conversation_engine import (
    ConversationEngine,
    ConversationEngineConfig,
    ConversationParticipants,
)
from counseling_voice_demo.log_writer import create_session_log_dirs
from counseling_voice_demo.models import (
    SessionState,
    SessionStatus,
    SpeakerRole,
    TurnStatus,
    WarningLevel,
)
from counseling_voice_demo.turn_regeneration import regenerate_single_turn


def test_regenerate_single_turn_uses_same_speaker_without_old_target_context(
    tmp_path,
) -> None:
    session = _session_with_unplayed_turns()
    session.status = SessionStatus.RUNNING
    text_client = _FakeTextClient(
        ["同じクライアントの新しい発話です。", "後続のカウンセラー発話です。"]
    )
    engine = _engine(text_client)
    paths = create_session_log_dirs(tmp_path, session.session_id)

    result = regenerate_single_turn(
        session,
        turn_id=2,
        conversation_engine=engine,
        cumulative_audio_seconds=120.0,
        log_paths=paths,
    )

    assert result.turn.turn_id == 2
    assert result.revision.revision_id == 2
    assert result.revision.regenerated is True
    assert result.revision.canonical_text == "同じクライアントの新しい発話です。"
    assert session.turns[2].status == TurnStatus.INVALIDATED
    assert [turn.turn_id for turn in result.generated_following_turns] == [4]
    assert text_client.calls[0].speaker_profile.profile_id == "client_test"
    assert text_client.active_turn_ids_by_call[0] == [1]
    assert text_client.active_turn_ids_by_call[1] == [1, 2]


def test_regenerate_single_turn_runs_moderation_clinical_ng_and_tts(tmp_path) -> None:
    session = _session_with_unplayed_turns()
    text_client = _FakeTextClient(["診断を断定する再生成です。"])
    engine = _engine(text_client)
    moderation_client = _FakeModerationClient(flagged=True)
    clinical_checker = _FakeClinicalChecker(
        warning_level=WarningLevel.MEDIUM, hold_reason="clinical_ng_medium"
    )
    tts_client = _FakeTtsClient()
    paths = create_session_log_dirs(tmp_path, session.session_id)

    result = regenerate_single_turn(
        session,
        turn_id=2,
        conversation_engine=engine,
        cumulative_audio_seconds=120.0,
        moderation_client=moderation_client,
        moderation_model="omni-moderation-test",
        clinical_ng_checker=clinical_checker,
        clinical_model="gpt-clinical-test",
        tts_client=tts_client,
        log_paths=paths,
    )

    revision = result.turn.active_revision()
    assert moderation_client.calls == [
        ("診断を断定する再生成です。", "omni-moderation-test")
    ]
    assert clinical_checker.calls == [
        ("診断を断定する再生成です。", "gpt-clinical-test")
    ]
    assert tts_client.calls[0].model == "gpt-4o-mini-tts"
    assert tts_client.calls[0].voice == "alloy"
    assert revision.moderation_result["flagged"] is True
    assert revision.clinical_ng_result["warning_level"] == WarningLevel.MEDIUM.value
    assert result.turn.warning_level == WarningLevel.HIGH
    assert result.turn.playback_status == TurnStatus.PLAYBACK_HOLD
    assert revision.hold_reason == "moderation_flagged"
    assert revision.tts_generated_at is not None

    safety_logs = _read_jsonl(paths.safety_log_jsonl)
    clinical_logs = _read_jsonl(paths.clinical_ng_log_jsonl)
    assert safety_logs[0]["turn_id"] == 2
    assert safety_logs[0]["flagged"] is True
    assert clinical_logs[0]["turn_id"] == 2
    assert clinical_logs[0]["warning_level"] == WarningLevel.MEDIUM.value


def _session_with_unplayed_turns() -> SessionState:
    session = SessionState(session_id="session_regenerate_test")
    session.add_turn(
        speaker_role=SpeakerRole.COUNSELOR,
        speaker_name="佐伯",
        profile_id="counselor_test",
        text="今日はどんなことを話したいですか。",
        status=TurnStatus.PLAYED,
    )
    session.add_turn(
        speaker_role=SpeakerRole.CLIENT,
        speaker_name="高橋",
        profile_id="client_test",
        text="家族のことで困っています。",
        status=TurnStatus.AUDIO_READY,
    )
    session.add_turn(
        speaker_role=SpeakerRole.COUNSELOR,
        speaker_name="佐伯",
        profile_id="counselor_test",
        text="もう少し詳しく教えてください。",
        status=TurnStatus.AUDIO_READY,
    )
    return session


def _engine(text_client: _FakeTextClient) -> ConversationEngine:
    return ConversationEngine(
        text_client=text_client,
        participants=ConversationParticipants(
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
        ),
        config=ConversationEngineConfig(ahead_generation_turns=2),
    )


class _FakeTextClient:
    def __init__(self, texts: list[str]) -> None:
        self._texts = texts
        self.calls = []
        self.active_turn_ids_by_call = []

    def generate_turn(self, request):
        self.calls.append(request)
        self.active_turn_ids_by_call.append(
            [
                turn.turn_id
                for turn in request.session.turns
                if turn.status != TurnStatus.INVALIDATED
            ]
        )
        text = self._texts[len(self.calls) - 1]
        return type(
            "Result",
            (),
            {"text": text, "raw_response": {"id": f"resp_{len(self.calls)}"}},
        )()


class _FakeModerationClient:
    def __init__(self, *, flagged: bool) -> None:
        self._flagged = flagged
        self.calls = []

    def moderate(self, *, text: str, model: str):
        self.calls.append((text, model))
        return type(
            "ModerationResult", (), {"flagged": self._flagged, "raw": {"ok": True}}
        )()


class _FakeClinicalChecker:
    def __init__(self, *, warning_level: WarningLevel, hold_reason: str | None) -> None:
        self._warning_level = warning_level
        self._hold_reason = hold_reason
        self.calls = []

    def check(self, request):
        self.calls.append((request.text, request.model))
        return type(
            "ClinicalResult",
            (),
            {
                "warning_level": self._warning_level,
                "categories": ["diagnosis"],
                "reason": "診断を断定している",
                "hold_reason": self._hold_reason,
                "raw_response": {"ok": True},
            },
        )()


class _FakeTtsClient:
    def __init__(self) -> None:
        self.calls = []

    def synthesize_turn(self, request):
        self.calls.append(request)
        request.turn.active_revision().audio_path = str(
            request.log_paths.audio_all_revisions_dir / "fake.mp3"
        )
        request.turn.active_revision().audio_format = request.response_format
        return type(
            "TtsResult", (), {"audio_path": request.turn.active_revision().audio_path}
        )()


def _read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
