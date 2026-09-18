from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import pytest

from counseling_voice_demo.evaluation_runner import (
    EVALUATION_VALIDITY_NOTICE,
    EvaluationRunConfig,
    run_session_evaluations,
)
from counseling_voice_demo.log_writer import create_session_log_dirs
from counseling_voice_demo.models import SessionState, SessionStatus, SpeakerRole
from counseling_voice_demo.openai_evaluation_client import EvaluationRequest, EvaluationResult


@dataclass
class _RecordingEvaluationClient:
    responses: list[str]
    requests: list[EvaluationRequest] | None = None

    def __post_init__(self) -> None:
        self.requests = []

    def evaluate(self, request: EvaluationRequest) -> EvaluationResult:
        assert self.requests is not None
        self.requests.append(request)
        return EvaluationResult(
            text=self.responses.pop(0),
            raw_response={"id": f"response_{len(self.requests)}"},
        )


def _completed_session() -> SessionState:
    session = SessionState(session_id="session_eval_runner")
    session.status = SessionStatus.COMPLETED
    session.add_turn(
        speaker_role=SpeakerRole.COUNSELOR,
        speaker_name="佐伯",
        profile_id="counselor_default",
        text="今日はどんなことを話したいですか。",
    )
    return session


def _read_json(path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_run_session_evaluations_runs_public_and_internal_after_completion_and_saves(tmp_path) -> None:
    paths = create_session_log_dirs(tmp_path, "session_eval_runner")
    client = _RecordingEvaluationClient(
        responses=[
            "観察可能な逐語録に基づく評価です。",
            "内部背景を踏まえた改善用評価です。",
        ]
    )

    result = run_session_evaluations(
        session=_completed_session(),
        client=client,
        config=EvaluationRunConfig(model="gpt-eval-test", reasoning_effort="low"),
        log_paths=paths,
        internal_context={"hidden_background": "公開してはいけない背景"},
    )

    assert result.public.text.startswith(EVALUATION_VALIDITY_NOTICE)
    assert "観察可能な逐語録に基づく評価です。" in result.public.text
    assert result.internal.text.startswith(EVALUATION_VALIDITY_NOTICE)
    assert "内部背景を踏まえた改善用評価です。" in result.internal.text
    assert [request.public for request in client.requests] == [True, False]
    assert [request.reasoning_effort for request in client.requests] == ["low", "low"]
    assert client.requests[0].internal_context is None
    assert client.requests[1].internal_context == {"hidden_background": "公開してはいけない背景"}

    public_record = _read_json(paths.evaluation_public_json)
    internal_record = _read_json(paths.evaluation_internal_json)
    assert public_record["evaluation_type"] == "public"
    assert internal_record["evaluation_type"] == "internal"
    assert public_record["text"].startswith(EVALUATION_VALIDITY_NOTICE)
    assert internal_record["text"].startswith(EVALUATION_VALIDITY_NOTICE)
    assert "公開してはいけない背景" not in paths.evaluation_public_json.read_text(encoding="utf-8")
    assert "公開してはいけない背景" in paths.evaluation_internal_json.read_text(encoding="utf-8")


def test_run_session_evaluations_does_not_run_before_session_completion(tmp_path) -> None:
    session = _completed_session()
    session.status = SessionStatus.RUNNING
    client = _RecordingEvaluationClient(responses=["should not be used"])

    with pytest.raises(ValueError, match="セッション終了後"):
        run_session_evaluations(
            session=session,
            client=client,
            config=EvaluationRunConfig(model="gpt-eval-test"),
            log_paths=create_session_log_dirs(tmp_path, "session_eval_not_completed"),
        )

    assert client.requests == []
