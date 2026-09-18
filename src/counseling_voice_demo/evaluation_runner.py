from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from counseling_voice_demo.log_writer import SessionLogPaths, write_evaluation_result
from counseling_voice_demo.models import SessionState, SessionStatus, utc_now
from counseling_voice_demo.openai_evaluation_client import (
    EvaluationRequest,
    EvaluationResult,
)


EVALUATION_VALIDITY_NOTICE = (
    "※これは妥当性未検証の研究デモ用参考評価であり、"
    "臨床技能評価や診断ではありません。"
)


@dataclass(frozen=True)
class EvaluationRunConfig:
    model: str
    max_output_tokens: int = 700
    reasoning_effort: str | None = None
    save_public: bool = True
    save_internal: bool = True


@dataclass(frozen=True)
class EvaluationRunResult:
    public: EvaluationResult
    internal: EvaluationResult


class EvaluationClient(Protocol):
    def evaluate(self, request: EvaluationRequest) -> EvaluationResult:
        ...


def run_session_evaluations(
    *,
    session: SessionState,
    client: EvaluationClient,
    config: EvaluationRunConfig,
    log_paths: SessionLogPaths | None = None,
    internal_context: dict[str, Any] | None = None,
) -> EvaluationRunResult:
    if session.status not in {SessionStatus.COMPLETED, SessionStatus.STOPPED}:
        raise ValueError("自動評価はセッション終了後にだけ実行できます")

    public = _evaluate_with_notice(
        client,
        EvaluationRequest(
            session=session,
            model=config.model,
            public=True,
            internal_context=None,
            max_output_tokens=config.max_output_tokens,
            reasoning_effort=config.reasoning_effort,
        ),
    )
    internal = _evaluate_with_notice(
        client,
        EvaluationRequest(
            session=session,
            model=config.model,
            public=False,
            internal_context=internal_context,
            max_output_tokens=config.max_output_tokens,
            reasoning_effort=config.reasoning_effort,
        ),
    )

    if log_paths is not None:
        if config.save_public:
            write_evaluation_result(
                log_paths,
                public=True,
                record=_evaluation_record(
                    session=session,
                    evaluation_type="public",
                    result=public,
                ),
            )
        if config.save_internal:
            write_evaluation_result(
                log_paths,
                public=False,
                record=_evaluation_record(
                    session=session,
                    evaluation_type="internal",
                    result=internal,
                    internal_context=internal_context,
                ),
            )

    return EvaluationRunResult(public=public, internal=internal)


def _evaluate_with_notice(
    client: EvaluationClient,
    request: EvaluationRequest,
) -> EvaluationResult:
    result = client.evaluate(request)
    text = _ensure_validity_notice(result.text)
    return EvaluationResult(text=text, raw_response=result.raw_response)


def _ensure_validity_notice(text: str) -> str:
    if EVALUATION_VALIDITY_NOTICE in text:
        return text
    return f"{EVALUATION_VALIDITY_NOTICE}\n\n{text}"


def _evaluation_record(
    *,
    session: SessionState,
    evaluation_type: str,
    result: EvaluationResult,
    internal_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "session_id": session.session_id,
        "evaluation_type": evaluation_type,
        "generated_at": utc_now(),
        "validity_notice": EVALUATION_VALIDITY_NOTICE,
        "text": result.text,
        "raw_response": result.raw_response,
    }
    if internal_context is not None:
        record["internal_context"] = internal_context
    return record
