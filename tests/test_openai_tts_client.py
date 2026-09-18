from __future__ import annotations

import pytest

from counseling_voice_demo.log_writer import create_session_log_dirs
from counseling_voice_demo.models import SessionState, SpeakerRole
from counseling_voice_demo.openai_text_client import OpenAIClientError, ReplayModeApiCallError
from counseling_voice_demo.openai_tts_client import OpenAITtsClient, TtsRequest


class _FakeStreamingResponse:
    def __init__(self, calls) -> None:
        self.calls = calls

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return _FakeAudioResponse()


class _FakeAudioResponse:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        return None

    def stream_to_file(self, path) -> None:
        path.write_bytes(b"fake mp3")


class _FakeSpeech:
    def __init__(self) -> None:
        self.calls = []
        self.with_streaming_response = _FakeStreamingResponse(self.calls)


class _FakeOpenAIClient:
    def __init__(self) -> None:
        self.audio = type("Audio", (), {"speech": _FakeSpeech()})()


def test_synthesize_turn_writes_internal_revision_audio(tmp_path) -> None:
    fake_client = _FakeOpenAIClient()
    paths = create_session_log_dirs(tmp_path, "session_tts_test")
    session = SessionState(session_id="session_tts_test")
    turn = session.add_turn(
        speaker_role=SpeakerRole.COUNSELOR,
        speaker_name="佐伯",
        profile_id="counselor_default",
        text="今日はどんなことを話したいですか。",
    )

    result = OpenAITtsClient(client=fake_client).synthesize_turn(
        TtsRequest(
            turn=turn,
            log_paths=paths,
            model="gpt-4o-mini-tts",
            voice="coral",
            instructions="落ち着いて話す。",
            response_format="mp3",
        )
    )

    kwargs = fake_client.audio.speech.calls[0]
    # TTS uses the Audio Speech API, not Responses API, so passing store=False would be invalid here.
    assert kwargs == {
        "model": "gpt-4o-mini-tts",
        "voice": "coral",
        "input": "今日はどんなことを話したいですか。",
        "instructions": "落ち着いて話す。",
        "response_format": "mp3",
    }
    assert result.audio_path == paths.audio_all_revisions_dir / "turn_0001_rev01_counselor.mp3"
    assert result.audio_path.read_bytes() == b"fake mp3"
    assert turn.active_revision().audio_path == str(result.audio_path)
    assert turn.active_revision().audio_format == "mp3"


def test_synthesize_turn_forbids_api_calls_in_replay_mode(tmp_path) -> None:
    fake_client = _FakeOpenAIClient()
    paths = create_session_log_dirs(tmp_path, "session_tts_replay_test")
    session = SessionState(session_id="session_tts_replay_test")
    turn = session.add_turn(
        speaker_role=SpeakerRole.CLIENT,
        speaker_name="高橋",
        profile_id="client_default",
        text="少し困っています。",
    )

    with pytest.raises(ReplayModeApiCallError):
        OpenAITtsClient(client=fake_client, replay_mode=True).synthesize_turn(
            TtsRequest(
                turn=turn,
                log_paths=paths,
                model="gpt-4o-mini-tts",
                voice="alloy",
                instructions="少し不安そうに話す。",
                response_format="mp3",
            )
        )

    assert fake_client.audio.speech.calls == []


def test_synthesize_turn_rejects_empty_text_before_api_call(tmp_path) -> None:
    fake_client = _FakeOpenAIClient()
    paths = create_session_log_dirs(tmp_path, "session_tts_empty_test")
    session = SessionState(session_id="session_tts_empty_test")
    turn = session.add_turn(
        speaker_role=SpeakerRole.COUNSELOR,
        speaker_name="佐伯",
        profile_id="counselor_default",
        text="",
    )

    with pytest.raises(OpenAIClientError, match="TTS入力テキストが空です"):
        OpenAITtsClient(client=fake_client).synthesize_turn(
            TtsRequest(
                turn=turn,
                log_paths=paths,
                model="gpt-4o-mini-tts",
                voice="coral",
                instructions="落ち着いて話す。",
                response_format="mp3",
            )
        )

    assert fake_client.audio.speech.calls == []
