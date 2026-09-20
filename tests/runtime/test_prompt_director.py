from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime

import pytest
import httpx
from openai import APIError, APIConnectionError, APIStatusError, APITimeoutError

from counseling_voice_demo.runtime.prompt_context import PublicHistoryMessage
from counseling_voice_demo.runtime.prompt_director import (
    ClosingAssessmentRequest,
    SessionEndContext,
    PROMPT_DIRECTOR_REVIEW_SYSTEM_PROMPT,
    PROMPT_DIRECTOR_SYSTEM_PROMPT,
    PromptDirector,
    PromptDirectorError,
    PromptDirectorRequest,
    PromptDirectorResult,
    format_prompt_director_input,
    parse_prompt_director_response,
    prompt_director_text_format,
    render_prompt_director_instruction,
)


def test_closing_assessment_uses_actual_history_and_constrained_client_ids() -> None:
    request = ClosingAssessmentRequest(
        counselor_speaker_id="counselor",
        client_display_names={"client_a": "夫", "client_b": "妻"},
        public_history=(
            PublicHistoryMessage("client_a", "はい、ありがとうございます。"),
            PublicHistoryMessage("counselor", "ご主人は、どうなりそうですか。"),
        ),
        session_summary="これまでの対処について話している。",
    )
    llm = _RecordingLLM(
        json.dumps({"reply_speaker_ids": ["client_a"], "reason": "夫への確認がある。"})
    )

    result = asyncio.run(PromptDirector(llm).assess_closing(request))

    assert result.reply_speaker_ids == ["client_a"]
    assert len(llm.calls) == 1
    payload = json.loads(llm.calls[0]["latest_input"])
    assert payload["public_history"][-1]["text"] == "ご主人は、どうなりそうですか。"
    schema = llm.calls[0]["text_format"]["schema"]
    assert schema["properties"]["reply_speaker_ids"]["items"]["enum"] == [
        "client_a",
        "client_b",
    ]


@pytest.mark.parametrize(
    "response",
    [
        "not json",
        '{"reply_speaker_ids": ["unknown"], "reason": "確認"}',
        '{"reply_speaker_ids": ["counselor"], "reason": "確認"}',
        '{"reply_speaker_ids": ["client_a", "client_a"], "reason": "確認"}',
        '{"reply_speaker_ids": [], "reason": " "}',
    ],
)
def test_closing_assessment_rejects_invalid_decisions(response) -> None:
    request = ClosingAssessmentRequest(
        counselor_speaker_id="counselor",
        client_display_names={"client_a": "夫"},
        public_history=(PublicHistoryMessage("counselor", "今日はここまでです。"),),
    )
    with pytest.raises(PromptDirectorError):
        asyncio.run(PromptDirector(_RecordingLLM(response)).assess_closing(request))


class _RecordingLLM:
    def __init__(self, response: str | list[str]) -> None:
        self.response = response
        self.calls: list[dict[str, object]] = []

    async def stream_text(
        self, *, latest_input, history=None, system_prompt=None, text_format=None
    ):
        self.calls.append(
            {
                "latest_input": latest_input,
                "history": history,
                "system_prompt": system_prompt,
                "text_format": text_format,
            }
        )
        response = (
            self.response[len(self.calls) - 1]
            if isinstance(self.response, list)
            else self.response
        )
        midpoint = len(response) // 2
        yield response[:midpoint]
        yield response[midpoint:]


def _request() -> PromptDirectorRequest:
    return PromptDirectorRequest(
        turn_id=7,
        speaker_id="counselor",
        fixed_system_prompt="カウンセラーの役割を維持してください。",
        configured_prompt=("任意の面接方針です。相手の言葉を短く言い換えてください。"),
        shared_context="家族で相談に来ている。",
        speaker_profile="家族面接を担当するカウンセラー。",
        public_history=(
            PublicHistoryMessage("client_a", "少し話し合えるようになりたいです。"),
            PublicHistoryMessage("counselor", "話し合えることが大切なのですね。"),
            PublicHistoryMessage("client_b", "昨日は少しだけ話せました。"),
        ),
        session_summary="家族内の会話を扱っている。",
        current_objective="設定されたプロンプトに従って対話を継続する。",
        response_target="主な宛先はclient_b。",
        turn_specific_instructions="応答は短くする。",
    )


def _upstream_error():
    return APIError(
        "upstream connect error or disconnect/reset before headers. "
        "reset reason: connection termination",
        request=httpx.Request("POST", "https://example.invalid/responses"),
        body=None,
    )


@pytest.mark.parametrize("fail_review", [False, True])
def test_director_retries_transient_stream_without_reusing_partial_text(
    monkeypatch, fail_review
):
    async def scenario():
        delays = []
        attempts = []

        async def wait(delay):
            delays.append(delay)

        monkeypatch.setattr(
            "counseling_voice_demo.runtime.prompt_director.sleep", wait, raising=False
        )

        class InterruptedLLM(_RecordingLLM):
            async def stream_text(self, **kwargs):
                call = len(self.calls)
                if call == int(fail_review):
                    self.calls.append(kwargs)
                    yield "partial JSON that must be discarded"
                    raise _upstream_error()
                async for part in super().stream_text(**kwargs):
                    yield part

        async def record(details):
            attempts.append(details)

        llm = InterruptedLLM(json.dumps(_payload(), ensure_ascii=False))
        result = await PromptDirector(llm).create_directive(
            _request(), on_attempt=record
        )
        assert result.response_example == _result().response_example
        assert len(llm.calls) == 3
        failed_index = int(fail_review)
        assert llm.calls[failed_index] == llm.calls[failed_index + 1]
        assert len(delays) == 1 and delays[0] > 0
        failed = [item for item in attempts if item.get("transport_error")]
        assert len(failed) == 1
        assert failed[0]["stage"] == ("確認" if fail_review else "生成")
        assert failed[0]["will_retry"] is True
        assert failed[0]["response_json"] == ""

    asyncio.run(scenario())


@pytest.mark.parametrize("retry_limit", [0, 2])
def test_director_transient_exhaustion_is_resumable(monkeypatch, retry_limit):
    async def scenario():
        calls = []
        delays = []

        async def wait(delay):
            delays.append(delay)

        monkeypatch.setattr(
            "counseling_voice_demo.runtime.prompt_director.sleep", wait, raising=False
        )

        class UnavailableLLM:
            async def stream_text(self, **kwargs):
                calls.append(kwargs)
                yield "discard"
                raise _upstream_error()

        with pytest.raises(PromptDirectorError) as caught:
            await PromptDirector(
                UnavailableLLM(), max_retries=retry_limit
            ).create_directive(_request())
        assert caught.value.pause_reason == "prompt_director_transport"
        assert isinstance(caught.value.__cause__, APIError)
        assert len(calls) == retry_limit + 1
        assert len(delays) == retry_limit
        assert all(call == calls[0] for call in calls)

    asyncio.run(scenario())


@pytest.mark.parametrize("status", [400, 401, 403, 404, 429])
def test_director_does_not_retry_permanent_api_failures(status):
    async def scenario():
        calls = []
        request = httpx.Request("POST", "https://example.invalid/responses")
        error = APIStatusError(
            "permanent failure",
            response=httpx.Response(status, request=request),
            body={"code": "insufficient_quota"} if status == 429 else None,
        )

        class FailingLLM:
            async def stream_text(self, **kwargs):
                calls.append(kwargs)
                raise error
                yield

        with pytest.raises(APIStatusError) as caught:
            await PromptDirector(FailingLLM()).create_directive(_request())
        assert caught.value is error
        assert len(calls) == 1

    asyncio.run(scenario())


def test_director_transport_backoff_can_be_cancelled(monkeypatch):
    async def scenario():
        waiting = asyncio.Event()
        calls = []

        async def wait(delay):
            waiting.set()
            await asyncio.Event().wait()

        monkeypatch.setattr(
            "counseling_voice_demo.runtime.prompt_director.sleep", wait, raising=False
        )

        class FailingLLM:
            async def stream_text(self, **kwargs):
                calls.append(kwargs)
                raise APIConnectionError(
                    request=httpx.Request("POST", "https://example.invalid/responses")
                )
                yield

        task = asyncio.create_task(
            PromptDirector(FailingLLM()).create_directive(_request())
        )
        try:
            await asyncio.wait_for(waiting.wait(), timeout=1)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert len(calls) == 1
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "failure",
    ["timeout", "stream_disconnect", "server_event", "http_503", "rate_limit"],
)
def test_director_recovers_transient_error_types(monkeypatch, failure):
    from counseling_voice_demo.runtime.streaming_llm import StreamingLLMError

    async def scenario():
        request = httpx.Request("POST", "https://example.invalid/responses")
        errors = {
            "timeout": APITimeoutError(request=request),
            "stream_disconnect": httpx.RemoteProtocolError("stream disconnected"),
            "server_event": StreamingLLMError("server failed", code="server_error"),
            "http_503": APIStatusError(
                "unavailable", response=httpx.Response(503, request=request), body=None
            ),
            "rate_limit": APIStatusError(
                "slow down",
                response=httpx.Response(429, request=request),
                body={"code": "rate_limit_exceeded"},
            ),
        }
        delays = []

        async def wait(delay):
            delays.append(delay)

        monkeypatch.setattr("counseling_voice_demo.runtime.prompt_director.sleep", wait)

        class InterruptedLLM(_RecordingLLM):
            async def stream_text(self, **kwargs):
                if not self.calls:
                    self.calls.append(kwargs)
                    raise errors[failure]
                async for part in super().stream_text(**kwargs):
                    yield part

        llm = InterruptedLLM(json.dumps(_payload(), ensure_ascii=False))
        await PromptDirector(llm).create_directive(_request())
        assert len(llm.calls) == 3
        assert len(delays) == 1 and delays[0] > 0

    asyncio.run(scenario())


@pytest.mark.parametrize("header", ["retry-after", "retry-after-ms", "http-date"])
def test_director_honors_server_retry_delay(monkeypatch, header):
    async def scenario():
        request = httpx.Request("POST", "https://example.invalid/responses")
        name = "retry-after" if header == "http-date" else header
        value = (
            format_datetime(
                datetime.now(timezone.utc) + timedelta(seconds=120), usegmt=True
            )
            if header == "http-date"
            else "120000" if header == "retry-after-ms" else "120"
        )
        error = APIStatusError(
            "unavailable",
            response=httpx.Response(503, request=request, headers={name: value}),
            body=None,
        )
        delays = []

        async def wait(delay):
            delays.append(delay)

        monkeypatch.setattr("counseling_voice_demo.runtime.prompt_director.sleep", wait)

        class InterruptedLLM(_RecordingLLM):
            async def stream_text(self, **kwargs):
                if not self.calls:
                    self.calls.append(kwargs)
                    raise error
                async for part in super().stream_text(**kwargs):
                    yield part

        await PromptDirector(InterruptedLLM(json.dumps(_payload()))).create_directive(
            _request()
        )
        assert len(delays) == 1
        assert 118 <= delays[0] <= 120

    asyncio.run(scenario())


def _result() -> PromptDirectorResult:
    return PromptDirectorResult(
        instruction_checks=[
            {
                "source": "configured_response_prompt",
                "quote": _request().configured_prompt,
                "applicability": "applies",
                "evidence": "client_bが昨日の短い会話を報告している。",
                "force": "required",
                "priority": "指定なし",
                "response_excerpt": None,
                "response_assessment": "昨日の短い会話を言い換えている。",
            }
        ],
        context_basis="client_bの報告: 昨日は少し話せた。両者の合意や継続性は未確認。",
        response_intent="昨日の短い会話を伝え返す。新しい問いは加えない。",
        response_example=(
            "昨日は少し話せたという、これまでとは違う時間があったのですね。"
        ),
    )


def _payload():
    payload = _result().model_dump()
    payload["response_issues"] = []
    check = payload["instruction_checks"][0]
    del check["quote"]
    check.update(start_line=1, end_line=1)
    return payload


def _review_issue(reason, *, must_fix=True, category=None):
    return {
        "severity": "must_fix" if must_fix else "advisory",
        "category": category
        or ("history_contradiction" if must_fix else "optional_improvement"),
        "reason": reason,
        "evidence": "原文の適用条件と話者別の実際の履歴を照合した根拠。",
    }


@pytest.mark.parametrize(
    "draft_text,note",
    [
        ("昨日は少し話せたのですね。", "伝え返しを別の表現にする余地がある。"),
        ("それぞれに迷いが残っているのですね。", "終了時に別の問いを加える案もある。"),
    ],
)
def test_advisory_review_keeps_original_utterance_without_regeneration(
    draft_text, note
) -> None:
    draft = {**_payload(), "response_example": draft_text}
    advisory = _review_issue(note, must_fix=False)
    reviewed = {**draft, "response_issues": [advisory]}
    llm = _RecordingLLM([json.dumps(p) for p in (draft, reviewed)])
    attempts = []

    async def scenario():
        async def observe(attempt):
            attempts.append(attempt)

        return await PromptDirector(llm, max_retries=0).create_directive(
            _request(), on_attempt=observe
        )

    result = asyncio.run(scenario())
    assert result.response_example == draft_text
    assert len(llm.calls) == 2
    assert attempts[-1]["response_issues"] == [advisory]
    assert attempts[-1]["must_fix_issue_count"] == 0
    assert attempts[-1]["advisory_issue_count"] == 1
    assert attempts[-1]["will_retry"] is False
    assert note not in render_prompt_director_instruction(result, include_audit=False)


@pytest.mark.parametrize(
    "category",
    [
        "history_contradiction",
        "speaker_confusion",
        "required_instruction_violation",
        "prohibited_instruction_violation",
    ],
)
def test_must_fix_issues_regenerate_without_promoting_advisory_notes(category) -> None:
    draft = _payload()
    required = _review_issue(
        "本人が言っていないことを既発話としている。", category=category
    )
    advisory = _review_issue("語尾を少し柔らかくする任意の案。", must_fix=False)
    rejected = {**draft, "response_issues": [advisory, required]}
    repaired = {**draft, "response_example": "昨日は少し話せたのですね。"}
    llm = _RecordingLLM([json.dumps(p) for p in (draft, rejected, repaired, repaired)])

    result = asyncio.run(PromptDirector(llm).create_directive(_request()))

    assert result.response_example == repaired["response_example"]
    assert len(llm.calls) == 4
    feedback = (
        llm.calls[2]["latest_input"]
        .split("<rejected_draft_to_regenerate>\n", 1)[1]
        .split("\n</rejected_draft_to_regenerate>", 1)[0]
    )
    assert json.loads(feedback)["response_issues"] == [required]
    assert advisory["reason"] not in llm.calls[2]["latest_input"]


@pytest.mark.parametrize(
    "issue",
    [
        "重要度のない旧形式の指摘。",
        {
            "severity": "unknown",
            "category": "style",
            "reason": "文体",
            "evidence": "根拠",
        },
        {
            "severity": "must_fix",
            "category": "style",
            "reason": "文体",
            "evidence": "根拠",
        },
        {
            "severity": "must_fix",
            "category": "history_contradiction",
            "reason": "矛盾",
            "evidence": " ",
        },
    ],
)
def test_review_rejects_unclassified_or_unsupported_mandatory_issue(issue) -> None:
    payload = {**_payload(), "response_issues": [issue]}
    with pytest.raises(PromptDirectorError, match="JSON"):
        parse_prompt_director_response(json.dumps(payload), request=_request())


def test_advisory_does_not_bypass_reviewed_utterance_lock() -> None:
    draft = _payload()
    advisory = _review_issue("別の言い回しも可能。", must_fix=False, category="style")
    rewritten = {
        **draft,
        "response_example": "候補と違う本文を審査担当が書きました。",
        "response_issues": [advisory],
    }
    accepted = {**draft, "response_issues": [advisory]}
    llm = _RecordingLLM([json.dumps(p) for p in (draft, rewritten, accepted)])

    result = asyncio.run(PromptDirector(llm).create_directive(_request()))

    assert result.response_example == draft["response_example"]
    assert len(llm.calls) == 3
    assert llm.calls[-1]["system_prompt"] == PROMPT_DIRECTOR_REVIEW_SYSTEM_PROMPT


def _end_assessment():
    return {
        "explicit_end_request_client_ids": [],
        "counselor_proposed_end": True,
        "consenting_client_ids": ["client_a", "client_b"],
        "pending_question": False,
        "reason": "両者が面接全体の終了に同意した。",
    }


def test_director_reviews_end_consent_in_existing_two_calls() -> None:
    request = replace(
        _request(),
        session_end_context=SessionEndContext(
            client_ids=("client_a", "client_b"), allow_agreed_end=True
        ),
    )
    payload = _payload()
    payload["session_end_assessment"] = _end_assessment()
    llm = _RecordingLLM(json.dumps(payload, ensure_ascii=False))
    result = asyncio.run(PromptDirector(llm).create_directive(request))
    assert len(llm.calls) == 2
    assert result.session_end_assessment.consenting_client_ids == [
        "client_a",
        "client_b",
    ]
    for call in llm.calls:
        schema = call["text_format"]["schema"]
        assert "session_end_assessment" in schema["required"]
        end_schema = schema["$defs"]["SessionEndAssessment"]
        assert end_schema["properties"]["consenting_client_ids"]["items"]["enum"] == [
            "client_a",
            "client_b",
        ]
        assert "<session_end_context>" in call["latest_input"]
    # Closure facts are internal control data, never speech instructions.
    assert "consenting_client_ids" not in render_prompt_director_instruction(
        result, include_audit=False
    )


@pytest.mark.parametrize("invalid", [None, ["other"], ["client_a", "client_a"]])
def test_director_rejects_missing_or_invalid_session_end_assessment(invalid) -> None:
    request = replace(
        _request(),
        session_end_context=SessionEndContext(
            client_ids=("client_a", "client_b"), allow_agreed_end=False
        ),
    )
    payload = _payload()
    if invalid is not None:
        payload["session_end_assessment"] = {
            **_end_assessment(),
            "consenting_client_ids": invalid,
        }
    with pytest.raises(PromptDirectorError):
        parse_prompt_director_response(json.dumps(payload), request=request)


@pytest.mark.parametrize(
    "prompt", [PROMPT_DIRECTOR_SYSTEM_PROMPT, PROMPT_DIRECTOR_REVIEW_SYSTEM_PROMPT]
)
def test_prompt_director_system_prompt_is_method_agnostic(prompt) -> None:
    assert "設定された応答プロンプト" in prompt
    assert "フェーズ" not in prompt
    assert "スケーリング" not in prompt
    assert "例外探索" not in prompt


def test_format_prompt_director_input_keeps_prompt_but_does_not_duplicate_history() -> (
    None
):
    prompt = format_prompt_director_input(_request())

    assert "任意の面接方針です" in prompt
    assert "少し話し合えるようになりたいです。" not in prompt
    assert "昨日は少しだけ話せました。" not in prompt
    assert "家族内の会話を扱っている。" in prompt
    assert "家族で相談に来ている。" in prompt
    assert "家族面接を担当するカウンセラー。" in prompt
    assert "応答は短くする。" in prompt
    assert "JSON" in prompt


def test_format_prompt_director_input_allows_missing_response_target() -> None:
    request = PromptDirectorRequest(
        turn_id=1,
        speaker_id="counselor",
        fixed_system_prompt="役割制約",
        configured_prompt="任意の応答プロンプト",
        response_target=None,
    )

    prompt = format_prompt_director_input(request)

    assert "<response_target>\n（指定なし）\n</response_target>" in prompt


def test_director_keeps_target_reply_visible_after_another_person_speaks() -> None:
    history = (
        PublicHistoryMessage("client_a", "以前は土曜日を考えていました。"),
        PublicHistoryMessage("counselor", "今はどの日がよさそうですか？"),
        PublicHistoryMessage(
            "client_a", "日曜日なら参加できます。\n午前を希望します。"
        ),
        PublicHistoryMessage("client_b", "私はまだ決められません。"),
    )
    request = replace(_request(), public_history=history, response_target_id="client_a")
    llm = _RecordingLLM(json.dumps(_payload(), ensure_ascii=False))
    asyncio.run(PromptDirector(llm).create_directive(request))

    assert len(llm.calls) == 2
    for call in llm.calls:
        # This is a small view of actual utterances, not an inferred answer ledger.
        focused = json.loads(
            call["latest_input"]
            .split("<response_target_recent_statement>\n", 1)[1]
            .split("\n</response_target_recent_statement>", 1)[0]
        )
        assert focused == {
            "preceding_utterance": {"speaker_id": "counselor", "text": history[1].text},
            "target_utterance": {"speaker_id": "client_a", "text": history[2].text},
        }
        assert [json.loads(m["content"])["text"] for m in call["history"]] == [
            m.text for m in history
        ]


@pytest.mark.parametrize("target_id", [None, "not_in_history", "client_b"])
def test_target_statement_view_does_not_invent_missing_or_other_person_answers(
    target_id,
) -> None:
    request = replace(
        _request(),
        public_history=(PublicHistoryMessage("client_a", "私は未定です。"),),
        response_target_id=target_id,
    )
    assert "<response_target_recent_statement>" not in format_prompt_director_input(
        request
    )


def test_target_statement_view_allows_no_preceding_utterance() -> None:
    request = replace(
        _request(),
        public_history=(PublicHistoryMessage("client_a", "日曜日を希望します。"),),
        response_target_id="client_a",
    )
    prompt = format_prompt_director_input(request)
    focused = json.loads(
        prompt.split("<response_target_recent_statement>\n", 1)[1].split(
            "\n</response_target_recent_statement>", 1
        )[0]
    )
    assert focused["preceding_utterance"] is None
    assert focused["target_utterance"]["text"] == "日曜日を希望します。"


@pytest.mark.parametrize("history_length", [0, 1, 3])
def test_director_passes_ordered_history_with_roles_to_generation_and_review(
    history_length,
) -> None:
    history = (
        PublicHistoryMessage("person_b", "費用を抑えたいです。"),
        PublicHistoryMessage(
            "assistant", "まず「楽さ」を整理しませんか。\nどうでしょう。"
        ),
        PublicHistoryMessage("person_a", "私はそれでいいです。"),
    )[:history_length]
    request = replace(_request(), public_history=history, speaker_id="person_b")
    llm = _RecordingLLM(json.dumps(_payload(), ensure_ascii=False))
    attempts = []

    async def scenario():
        async def observe(record):
            attempts.append(record)

        await PromptDirector(llm).create_directive(request, on_attempt=observe)

    asyncio.run(scenario())
    for call, attempt in zip(llm.calls, attempts, strict=True):
        assert attempt["conversation_messages"] == call["history"]
        assert len(call["history"]) == history_length
        for actual, message in zip(call["history"], history, strict=True):
            assert actual["role"] == (
                "assistant"
                if attempt["stage"] == "生成" and message.speaker_id == "person_b"
                else "user"
            )
            assert json.loads(actual["content"]) == {
                "speaker_id": message.speaker_id,
                "text": message.text,
            }
            assert message.text not in call["latest_input"]


def test_prompt_director_text_format_uses_strict_json_schema() -> None:
    text_format = prompt_director_text_format()

    assert text_format["type"] == "json_schema"
    assert text_format["strict"] is True
    schema = text_format["schema"]
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == {
        "response_issues",
        "instruction_checks",
        "response_intent",
        "response_example",
        "context_basis",
    }
    check_schema = schema["$defs"]["_InstructionSelection"]
    assert check_schema["additionalProperties"] is False
    assert set(check_schema["required"]) == {
        "source",
        "start_line",
        "end_line",
        "applicability",
        "evidence",
        "force",
        "priority",
        "response_excerpt",
        "response_assessment",
    }


def test_request_schema_bounds_line_numbers_separately_for_each_source() -> None:
    request = replace(
        _request(),
        fixed_system_prompt="固定制約は一行です。",
        configured_prompt="一行目\n\n三行目\n",
        turn_specific_instructions="今回の指示\nその条件",
    )
    text_format = prompt_director_text_format(request)
    branches = text_format["schema"]["$defs"]["_InstructionSelection"]["anyOf"]
    maxima = {}
    for branch in branches:
        properties = branch["properties"]
        source = properties["source"]["enum"]
        assert len(source) == 1
        assert branch["additionalProperties"] is False
        assert set(branch["required"]) == set(properties)
        for field in ("start_line", "end_line"):
            assert properties[field]["type"] == "integer"
            assert properties[field]["minimum"] == 1
        assert properties["start_line"]["maximum"] == properties["end_line"]["maximum"]
        maxima[source[0]] = properties["end_line"]["maximum"]
    assert maxima == {
        "fixed_role_constraints": 1,
        "configured_response_prompt": 3,
        "turn_specific_instructions": 2,
    }
    # Building another request must not mutate an earlier or the static schema.
    prompt_director_text_format(replace(request, fixed_system_prompt="一\n二\n三"))
    assert branches[0]["properties"]["start_line"]["maximum"] == 1
    assert (
        "anyOf"
        not in prompt_director_text_format()["schema"]["$defs"]["_InstructionSelection"]
    )


@pytest.mark.parametrize("empty_source", ["", "  \n\t"])
def test_request_schema_does_not_offer_empty_instruction_sources(empty_source) -> None:
    request = replace(
        _request(),
        configured_prompt=empty_source,
        turn_specific_instructions=empty_source,
    )
    branches = prompt_director_text_format(request)["schema"]["$defs"][
        "_InstructionSelection"
    ]["anyOf"]
    assert len(branches) == 1
    assert branches[0]["properties"]["source"]["enum"] == ["fixed_role_constraints"]


@pytest.mark.parametrize(
    ("source", "request_field"),
    [
        ("fixed_role_constraints", "fixed_system_prompt"),
        ("configured_response_prompt", "configured_prompt"),
        ("turn_specific_instructions", "turn_specific_instructions"),
    ],
)
@pytest.mark.parametrize("newline", ["\n", "\r\n"])
def test_request_schema_excludes_blank_quote_endpoints(source, request_field, newline):
    original = newline.join(["", "  条件", "本文", " \t", "続き", "　"])
    request = replace(_request(), **{request_field: original})
    branches = prompt_director_text_format(request)["schema"]["$defs"][
        "_InstructionSelection"
    ]["anyOf"]
    properties = next(
        branch["properties"]
        for branch in branches
        if branch["properties"]["source"]["enum"] == [source]
    )
    for field in ("start_line", "end_line"):
        assert properties[field]["anyOf"] == [
            {"type": "integer", "minimum": 2, "maximum": 3},
            {"type": "integer", "minimum": 5, "maximum": 5},
        ]
    payload = _payload()
    payload["instruction_checks"][0].update(source=source, start_line=2, end_line=5)
    result = parse_prompt_director_response(json.dumps(payload), request=request)
    # Blank lines inside a real quotation remain part of the unchanged source.
    assert result.instruction_checks[0].quote == "".join(
        original.splitlines(keepends=True)[1:5]
    )


def test_director_sends_bounded_schema_and_original_lines_to_both_stages() -> None:
    async def scenario():
        request = replace(_request(), configured_prompt="\n本文\n別の指示\n \n続き")
        payload = _payload()
        payload["instruction_checks"][0].update(start_line=3, end_line=5)
        llm = _RecordingLLM(json.dumps(payload, ensure_ascii=False))
        await PromptDirector(llm).create_directive(request)
        assert len(llm.calls) == 2
        assert llm.calls[0]["text_format"] == prompt_director_text_format(request)
        assert llm.calls[1]["text_format"] == prompt_director_text_format(
            request, locked_response=_payload()["response_example"]
        )
        review_input = llm.calls[1]["latest_input"]
        draft_json = review_input.split("<draft_to_review>\n", 1)[1].split(
            "\n</draft_to_review>", 1
        )[0]
        assert json.loads(draft_json) == {
            "response_example": payload["response_example"]
        }
        assert "[L3] 別の指示" in review_input
        assert "[L5] 続き" in review_input

    asyncio.run(scenario())


@pytest.mark.parametrize("director_request", [None, _request()])
def test_review_schema_only_allows_locked_text_or_null_for_response_evidence(
    director_request,
):
    draft = "「日曜日の午前」がよいのですね。\nその予定を確認します。"
    schema = prompt_director_text_format(director_request, locked_response=draft)[
        "schema"
    ]
    locked = schema["properties"]["response_example"]
    assert locked == {"$ref": "#/$defs/_LockedResponseText"}
    assert schema["$defs"]["_LockedResponseText"]["enum"] == [draft]
    selections = schema["$defs"]["_InstructionSelection"]
    choices = selections["anyOf"] if director_request is not None else [selections]
    for selection in choices:
        assert selection["properties"]["response_excerpt"]["anyOf"] == [
            locked,
            {"type": "null"},
        ]
    # One shared enum avoids multiplying long utterances across source variants.
    assert (
        json.dumps(schema, ensure_ascii=False).count(
            json.dumps(draft, ensure_ascii=False)
        )
        == 1
    )


def test_director_rejects_missing_instruction_sources_before_calling_llm() -> None:
    async def scenario():
        request = replace(
            _request(),
            fixed_system_prompt="",
            configured_prompt="  \n",
            turn_specific_instructions="",
        )
        llm = _RecordingLLM(json.dumps(_payload(), ensure_ascii=False))
        with pytest.raises(
            PromptDirectorError, match="引用可能な原文の指示がありません"
        ):
            await PromptDirector(llm).create_directive(request)
        assert llm.calls == []

    asyncio.run(scenario())


@pytest.mark.parametrize("stage,valid_drafts", [("生成", 0), ("確認", 1)])
def test_invalid_line_range_error_identifies_stage_and_source_length(
    stage, valid_drafts
) -> None:
    async def scenario():
        invalid_payload = _payload()
        invalid_payload["instruction_checks"][0].update(
            source="fixed_role_constraints", start_line=2, end_line=3
        )
        llm = _RecordingLLM(
            [json.dumps(_payload(), ensure_ascii=False)] * valid_drafts
            + [json.dumps(invalid_payload, ensure_ascii=False)]
        )
        with pytest.raises(PromptDirectorError) as caught:
            await PromptDirector(llm).create_directive(_request())
        message = str(caught.value)
        assert f"{stage}段階" in message
        assert "fixed_role_constraints" in message
        assert "指定: 2〜3、原文: 1行" in message
        assert len(llm.calls) == valid_drafts + 1

    asyncio.run(scenario())


def test_parse_and_render_prompt_director_result() -> None:
    result = _result()

    parsed = parse_prompt_director_response(
        json.dumps(_payload(), ensure_ascii=False), request=_request()
    )
    rendered = render_prompt_director_instruction(parsed)

    assert parsed == result
    assert rendered.startswith("このように応答してください。")
    assert "＜原文の条件と今回の適用判断＞" in rendered
    assert "＜プロンプトの要約＞" not in rendered
    assert "＜この応答の目的＞" in rendered
    assert "＜今回発話する本文＞" in rendered
    assert "本文をそのまま発話" in rendered
    assert "自然な言い換えはできます" not in rendered
    assert result.instruction_checks[0].quote in rendered
    assert result.instruction_checks[0].evidence in rendered
    assert result.response_intent in rendered
    assert result.response_example in rendered
    assert "＜現在の対話状況＞" not in rendered
    assert "＜次に行うべき応答＞" not in rendered


def test_parse_prompt_director_response_rejects_invalid_json() -> None:
    with pytest.raises(PromptDirectorError, match="JSON"):
        parse_prompt_director_response("次は質問してください。", request=_request())


def test_director_render_preserves_exact_text_in_verbatim_payload() -> None:
    text = '「a\\b」と "c" ですね。\nその内容を承りました。'
    result = _result().model_copy(update={"response_example": text})
    rendered = render_prompt_director_instruction(result)
    encoded = rendered.split("＜今回発話する本文＞\n", 1)[1].split("\n\n", 1)[0]
    assert json.loads(encoded) == {
        "response_text": text,
        "require_repeat_verbatim": True,
    }


def test_speech_instruction_keeps_text_without_director_interpretations() -> None:
    result = _result().model_copy(
        update={"response_example": '「a\\b」と "c" ですね。\n続けてください。'}
    )
    rendered = render_prompt_director_instruction(result, include_audit=False)
    encoded = rendered.split("＜今回発話する本文＞\n", 1)[1].split("\n\n", 1)[0]

    assert json.loads(encoded)["response_text"] == result.response_example
    for internal_text in (
        result.context_basis,
        result.response_intent,
        result.instruction_checks[0].evidence,
        result.instruction_checks[0].response_assessment,
    ):
        assert internal_text not in rendered
    assert "原文の制約を優先" in rendered
    assert "＜原文の条件と今回の適用判断＞" not in rendered


@pytest.mark.parametrize(
    "field_name", ["response_intent", "response_example", "context_basis"]
)
def test_prompt_director_result_rejects_blank_required_text(field_name) -> None:
    payload = _payload()
    payload[field_name] = "   "

    with pytest.raises(PromptDirectorError, match="JSON"):
        parse_prompt_director_response(
            json.dumps(payload, ensure_ascii=False), request=_request()
        )


@pytest.mark.parametrize(
    "field_name",
    ["instruction_checks", "response_intent", "context_basis", "response_issues"],
)
def test_prompt_director_result_requires_explicit_decisions(field_name) -> None:
    payload = _payload()
    del payload[field_name]

    with pytest.raises(PromptDirectorError, match="JSON"):
        parse_prompt_director_response(
            json.dumps(payload, ensure_ascii=False), request=_request()
        )


def test_prompt_director_streams_and_validates_result() -> None:
    async def scenario() -> None:
        expected = _result()
        llm = _RecordingLLM(json.dumps(_payload(), ensure_ascii=False))
        director = PromptDirector(llm=llm)

        result = await director.create_directive(_request())

        assert result == expected
        assert len(llm.calls) == 2
        assert llm.calls[0]["system_prompt"] == PROMPT_DIRECTOR_SYSTEM_PROMPT
        assert "任意の面接方針です" in str(llm.calls[0]["latest_input"])
        assert llm.calls[1]["system_prompt"] != llm.calls[0]["system_prompt"]

    asyncio.run(scenario())


def test_review_cannot_replace_persons_draft_with_peers_utterance() -> None:
    draft = _payload()
    draft["response_example"] = (
        "連絡は続けたいですが、どのくらいの頻度がいいかはまだ迷っています。"
    )
    changed = {
        **draft,
        "response_example": "まだ見えていませんが、急かさないことは大事にしたいです。",
    }
    llm = _RecordingLLM(
        [json.dumps(p, ensure_ascii=False) for p in (draft, changed, draft)]
    )
    result = asyncio.run(PromptDirector(llm).create_directive(_request()))
    assert result.response_example == draft["response_example"]
    assert len(llm.calls) == 3
    assert llm.calls[1]["text_format"]["schema"]["$defs"]["_LockedResponseText"][
        "enum"
    ] == [draft["response_example"]]
    assert "審査で本文を変更" in llm.calls[2]["latest_input"]


@pytest.mark.parametrize(
    "issue",
    [
        "他者の直前発話を丸ごと繰り返し、本人の継続したいことを落としている。",
        "終了確認には両者が分からないと回答済みで、同じ質問をする根拠がない。",
    ],
)
def test_semantic_issue_regenerates_with_original_author_then_reviews_again(
    issue,
) -> None:
    issue = _review_issue(issue)
    draft = _payload()
    rejected = {**draft, "response_issues": [issue]}
    repaired = {
        **draft,
        "response_example": "まだ迷いが残っているのですね。今日はここまでにしましょう。",
    }
    llm = _RecordingLLM(
        [
            json.dumps(p, ensure_ascii=False)
            for p in (draft, rejected, repaired, repaired)
        ]
    )
    attempts = []

    async def scenario():
        async def observe(attempt):
            attempts.append(attempt)

        return await PromptDirector(llm).create_directive(
            _request(), on_attempt=observe
        )

    result = asyncio.run(scenario())
    assert result.response_example == repaired["response_example"]
    assert [c["system_prompt"] for c in llm.calls] == [
        PROMPT_DIRECTOR_SYSTEM_PROMPT,
        PROMPT_DIRECTOR_REVIEW_SYSTEM_PROMPT,
        PROMPT_DIRECTOR_SYSTEM_PROMPT,
        PROMPT_DIRECTOR_REVIEW_SYSTEM_PROMPT,
    ]
    assert issue["reason"] in llm.calls[2]["latest_input"]
    assert llm.calls[2]["history"] == llm.calls[0]["history"]
    assert llm.calls[3]["text_format"]["schema"]["$defs"]["_LockedResponseText"][
        "enum"
    ] == [repaired["response_example"]]
    assert attempts[1]["response_issues"] == [issue]
    assert attempts[1]["will_retry"] is True
    assert attempts[-1]["regeneration_round"] == 1


@pytest.mark.parametrize("max_retries", [0, 2])
def test_unresolved_semantic_issues_are_bounded_and_never_adopted(max_retries) -> None:
    from counseling_voice_demo.runtime.prompt_director import (
        PromptDirectorRetriesExhausted,
    )

    draft = _payload()
    rejected = {
        **draft,
        "response_issues": [_review_issue("回答済みの問いを繰り返している。")],
    }
    expected_calls = 2 * (max_retries + 1)
    llm = _RecordingLLM([json.dumps(p) for p in (draft, rejected) * (max_retries + 1)])
    with pytest.raises(PromptDirectorRetriesExhausted, match="意味検証"):
        asyncio.run(
            PromptDirector(llm, max_retries=max_retries).create_directive(_request())
        )
    assert len(llm.calls) == expected_calls


def test_director_repairs_rejected_response_and_keeps_original_context() -> None:
    async def scenario():
        request = replace(
            _request(),
            configured_prompt="注文確定後は受領を伝える。\n商品名を尋ね直さない。",
            public_history=(
                PublicHistoryMessage("customer", "赤いノートで確定です。"),
            ),
        )
        draft = _payload()
        draft["response_example"] = "何をご注文ですか？"
        reviewed = _payload()
        reviewed["response_intent"] = "確定した注文の受領を伝える。"
        reviewed["response_example"] = "赤いノートのご注文を承りました。"
        reviewed["instruction_checks"][0].update(
            start_line=1,
            end_line=2,
            evidence="注文は確定済み。",
        )
        llm = _RecordingLLM(
            [
                json.dumps(draft, ensure_ascii=False),
                json.dumps(
                    {
                        **draft,
                        "response_issues": [
                            _review_issue(
                                "注文は確定済みで、商品名の再質問は禁止されている。",
                                category="prohibited_instruction_violation",
                            )
                        ],
                    },
                    ensure_ascii=False,
                ),
                json.dumps(reviewed, ensure_ascii=False),
                json.dumps(reviewed, ensure_ascii=False),
            ]
        )

        result = await PromptDirector(llm).create_directive(request)

        assert result.response_example == reviewed["response_example"]
        assert len(llm.calls) == 4
        review_input = str(llm.calls[1]["latest_input"])
        assert format_prompt_director_input(request) in review_input
        assert draft["response_example"] in review_input
        assert "[L2] 商品名を尋ね直さない。" in review_input
        assert (
            json.loads(llm.calls[1]["history"][0]["content"])["text"]
            == "赤いノートで確定です。"
        )
        rendered = render_prompt_director_instruction(result)
        assert reviewed["response_example"] in rendered
        assert draft["response_example"] not in rendered

    asyncio.run(scenario())


def test_client_review_keeps_own_and_peer_history_separate_from_unspoken_draft() -> (
    None
):
    request = replace(
        _request(),
        speaker_id="client_b",
        client_peer_ids=("client_a",),
        public_history=(
            PublicHistoryMessage("client_a", "駅は朝に混み、夕方は空きます。"),
            PublicHistoryMessage("client_b", "遅刻しないか気になります。"),
            PublicHistoryMessage("counselor", "そう感じているのですね。"),
        ),
    )
    draft = _payload()
    draft["response_example"] = "駅は朝に混み、夕方は空くんですね。遅刻が気になります。"
    llm = _RecordingLLM(json.dumps(draft, ensure_ascii=False))

    asyncio.run(PromptDirector(llm).create_directive(request))

    review_input = llm.calls[1]["latest_input"]
    assert [item["role"] for item in llm.calls[1]["history"]] == [
        "user",
        "user",
        "user",
    ]
    assert (
        json.loads(llm.calls[1]["history"][1]["content"])["text"]
        == "遅刻しないか気になります。"
    )
    assert draft["response_example"] not in str(llm.calls[1]["history"])
    assert draft["response_example"] in review_input
    assert '"other_client_ids": ["client_a"]' in review_input
    assert format_prompt_director_input(request) in review_input


def test_counselor_review_does_not_add_client_repetition_comparison() -> None:
    llm = _RecordingLLM(json.dumps(_payload(), ensure_ascii=False))
    asyncio.run(PromptDirector(llm).create_directive(_request()))
    assert "<client_repetition_comparison>" not in llm.calls[1]["latest_input"]


def test_review_does_not_inherit_drafts_false_claim_of_fulfilling_a_question() -> None:
    async def scenario():
        request = replace(
            _request(),
            configured_prompt=(
                "終了時は今回分かったことを利用者に一つ尋ねる。\n"
                "利用者がすでに答えていれば、繰り返し尋ねない。"
            ),
            public_history=(PublicHistoryMessage("user", "二つの案を比較しました。"),),
            turn_specific_instructions="話した内容を短くまとめて終了に向かう。",
        )
        draft = _payload()
        draft["context_basis"] = "誤った推測: 利用者はすべて理解済み。"
        draft["response_intent"] = "誤った目的: 必須質問は省いて終える。"
        draft["response_example"] = "二つの案を比べられましたね。"
        draft["instruction_checks"][0].update(
            evidence="誤った根拠: 終了なので質問は不要。",
            response_excerpt="二つの案を比べられましたね。",
            response_assessment="誤った自己評価: 要約で必須質問を満たしている。",
        )
        reviewed = _payload()
        reviewed["context_basis"] = "利用者は案を比較したが、分かったことは未確認。"
        reviewed["response_intent"] = "今回分かったことを本人に尋ねる。"
        reviewed["response_example"] = "二つの案を比べて、どんなことが分かりましたか？"
        reviewed["instruction_checks"][0].update(
            evidence="終了時の確認に対する回答は履歴にない。",
            response_excerpt=reviewed["response_example"],
            response_assessment="今回分かったことを本人に尋ねている。",
        )
        llm = _RecordingLLM(
            [
                json.dumps(draft, ensure_ascii=False),
                json.dumps(
                    {
                        **draft,
                        "response_issues": [
                            _review_issue(
                                "本人が分かったことは未確認で、要約だけでは必須質問を満たさない。",
                                category="required_instruction_violation",
                            )
                        ],
                    },
                    ensure_ascii=False,
                ),
                json.dumps(reviewed, ensure_ascii=False),
                json.dumps(reviewed, ensure_ascii=False),
            ]
        )
        result = await PromptDirector(llm).create_directive(request)
        assert result.response_example == reviewed["response_example"]
        assert len(llm.calls) == 4
        review_input = llm.calls[1]["latest_input"]
        assert format_prompt_director_input(request) in review_input
        assert draft["response_example"] in review_input
        for value in (
            draft["context_basis"],
            draft["response_intent"],
            draft["instruction_checks"][0]["evidence"],
            draft["instruction_checks"][0]["response_assessment"],
        ):
            assert value not in review_input

    asyncio.run(scenario())


def test_draft_excerpt_mismatch_does_not_block_independent_review() -> None:
    async def scenario():
        draft = _payload()
        draft["instruction_checks"][0]["response_excerpt"] = "本文にない旧案の言葉"
        reviewed = _payload()
        reviewed["instruction_checks"][0]["response_excerpt"] = "昨日は少し話せた"
        llm = _RecordingLLM(
            [
                json.dumps(draft, ensure_ascii=False),
                json.dumps(reviewed, ensure_ascii=False),
            ]
        )
        result = await PromptDirector(llm).create_directive(_request())
        assert result.instruction_checks[0].response_excerpt == "昨日は少し話せた"
        assert len(llm.calls) == 2
        assert "本文にない旧案の言葉" not in llm.calls[1]["latest_input"]

    asyncio.run(scenario())


def test_final_excerpt_mismatch_is_regenerated_with_feedback_and_logged() -> None:
    async def scenario():
        invalid = _payload()
        invalid["instruction_checks"][0]["response_excerpt"] = "本文にない言葉"
        corrected = _payload()
        corrected["instruction_checks"][0]["response_excerpt"] = "昨日は少し話せた"
        llm = _RecordingLLM(
            [
                json.dumps(_payload(), ensure_ascii=False),
                json.dumps(invalid, ensure_ascii=False),
                json.dumps(corrected, ensure_ascii=False),
            ]
        )
        attempts = []

        async def record(attempt):
            attempts.append(attempt)

        result = await PromptDirector(llm).create_directive(
            _request(), on_attempt=record
        )
        assert result.instruction_checks[0].response_excerpt == "昨日は少し話せた"
        assert len(llm.calls) == 3
        retry = llm.calls[2]
        assert retry["system_prompt"] == PROMPT_DIRECTOR_REVIEW_SYSTEM_PROMPT
        assert format_prompt_director_input(_request()) in retry["latest_input"]
        assert "本文にない言葉" in retry["latest_input"]
        assert "発話本文に存在しません" in retry["latest_input"]
        assert retry["text_format"] == llm.calls[1]["text_format"]
        assert [(a["stage"], a["attempt"]) for a in attempts] == [
            ("生成", 1),
            ("確認", 1),
            ("確認", 2),
        ]
        failed = attempts[1]
        assert json.loads(failed["response_json"]) == invalid
        assert (
            failed["validation_details"]["response_example"]
            == invalid["response_example"]
        )
        assert (
            failed["validation_details"]["mismatches"][0]["response_excerpt"]
            == "本文にない言葉"
        )
        assert failed["will_retry"] is True
        assert attempts[2]["validation_error"] is None

    asyncio.run(scenario())


def test_repeated_excerpt_mismatches_exhaust_a_bounded_retry_budget() -> None:
    from counseling_voice_demo.runtime.prompt_director import (
        PromptDirectorRetriesExhausted,
    )

    async def scenario():
        invalid = _payload()
        invalid["instruction_checks"][0]["response_excerpt"] = "未修正の引用"
        llm = _RecordingLLM([json.dumps(_payload())] + [json.dumps(invalid)] * 3)
        attempts = []

        async def record(attempt):
            attempts.append(attempt)

        with pytest.raises(PromptDirectorRetriesExhausted, match="確認段階"):
            await PromptDirector(llm, max_retries=2).create_directive(
                _request(), on_attempt=record
            )
        assert len(llm.calls) == 4
        assert [a["will_retry"] for a in attempts[1:]] == [True, True, False]
        assert all(json.loads(a["response_json"]) == invalid for a in attempts[1:])

    asyncio.run(scenario())


def test_stop_cancels_prompt_director_regeneration() -> None:
    async def scenario():
        invalid = _payload()
        invalid["instruction_checks"][0]["response_excerpt"] = "本文にない引用"
        retry_started = asyncio.Event()

        class BlockingRepairLLM(_RecordingLLM):
            async def stream_text(self, **kwargs):
                if len(self.calls) == 2:
                    retry_started.set()
                    await asyncio.Event().wait()
                async for part in super().stream_text(**kwargs):
                    yield part

        llm = BlockingRepairLLM([json.dumps(_payload()), json.dumps(invalid)])
        task = asyncio.create_task(PromptDirector(llm).create_directive(_request()))
        try:
            await asyncio.wait_for(retry_started.wait(), timeout=1)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "failure", ["invalid_json", "invalid_source_range", "provider_error"]
)
def test_director_does_not_use_unreviewed_draft_when_review_fails(failure) -> None:
    async def scenario():
        response = _payload()
        response["instruction_checks"][0]["end_line"] = 999
        invalid_review = (
            "not json"
            if failure == "invalid_json"
            else json.dumps(response, ensure_ascii=False)
        )

        class ReviewFailureLLM(_RecordingLLM):
            async def stream_text(self, **kwargs):
                if self.calls and failure == "provider_error":
                    raise RuntimeError("review provider failed")
                async for chunk in super().stream_text(**kwargs):
                    yield chunk

        llm = ReviewFailureLLM(
            [json.dumps(_payload(), ensure_ascii=False), invalid_review]
        )
        expected_error = (
            RuntimeError if failure == "provider_error" else PromptDirectorError
        )
        with pytest.raises(expected_error):
            await PromptDirector(llm).create_directive(_request())

    asyncio.run(scenario())


@pytest.mark.parametrize("field_name", ["evidence", "priority", "response_assessment"])
def test_instruction_check_rejects_blank_text(field_name) -> None:
    payload = _payload()
    payload["instruction_checks"][0][field_name] = "  "
    with pytest.raises(PromptDirectorError, match="JSON"):
        parse_prompt_director_response(
            json.dumps(payload, ensure_ascii=False), request=_request()
        )


def test_instruction_checks_cannot_be_empty() -> None:
    payload = _payload()
    payload["instruction_checks"] = []
    with pytest.raises(PromptDirectorError, match="JSON"):
        parse_prompt_director_response(
            json.dumps(payload, ensure_ascii=False), request=_request()
        )


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("source", "public_conversation_history"),
        ("applicability", "maybe"),
        ("force", "always_ask"),
    ],
)
def test_instruction_check_rejects_unknown_source_or_decision(
    field_name, value
) -> None:
    payload = _payload()
    payload["instruction_checks"][0][field_name] = value
    with pytest.raises(PromptDirectorError, match="JSON"):
        parse_prompt_director_response(
            json.dumps(payload, ensure_ascii=False), request=_request()
        )


@pytest.mark.parametrize(
    ("source", "quote"),
    [
        ("configured_response_prompt", _request().configured_prompt),
        ("fixed_role_constraints", "カウンセラーの役割を維持してください。"),
        ("turn_specific_instructions", "応答は短くする。"),
    ],
)
def test_director_validates_quotes_against_their_declared_source(source, quote) -> None:
    async def scenario():
        payload = _payload()
        payload["instruction_checks"][0].update(source=source)
        llm = _RecordingLLM(json.dumps(payload, ensure_ascii=False))
        result = await PromptDirector(llm).create_directive(_request())
        assert result.instruction_checks[0].quote == quote
        assert len(llm.calls) == 2

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("start_line", "end_line"),
    [(0, 1), (2, 1), (1, 100), ("1", 1)],
)
def test_director_rejects_invalid_source_ranges(start_line, end_line) -> None:
    async def scenario():
        payload = _payload()
        payload["instruction_checks"][0].update(
            start_line=start_line, end_line=end_line
        )
        llm = _RecordingLLM(json.dumps(payload, ensure_ascii=False))
        with pytest.raises(PromptDirectorError, match="JSON|原文"):
            await PromptDirector(llm).create_directive(_request())
        assert len(llm.calls) == 1

    asyncio.run(scenario())


def test_model_cannot_supply_a_rewritten_quote() -> None:
    payload = _payload()
    payload["instruction_checks"][0]["quote"] = "必要に応じて言い換える。"
    with pytest.raises(PromptDirectorError, match="JSON"):
        parse_prompt_director_response(
            json.dumps(payload, ensure_ascii=False), request=_request()
        )


def test_source_range_preserves_whitespace_and_multiline_conditions() -> None:
    source = "\r\n  条件が成立する場合は、\r\n次の指示に従ってください。\r\n"
    request = replace(_request(), configured_prompt=source)
    payload = _payload()
    payload["instruction_checks"][0].update(start_line=2, end_line=3)
    result = parse_prompt_director_response(
        json.dumps(payload, ensure_ascii=False), request=request
    )
    assert result.instruction_checks[0].quote == source[2:]
    formatted = format_prompt_director_input(request)
    assert (
        "[L1] \n[L2]   条件が成立する場合は、\n[L3] 次の指示に従ってください。"
        in formatted
    )


@pytest.mark.parametrize("source", ["", "\n", "   \n"])
def test_source_range_cannot_resolve_an_empty_source(source) -> None:
    request = replace(_request(), configured_prompt=source)
    with pytest.raises(PromptDirectorError, match="JSON|原文"):
        parse_prompt_director_response(
            json.dumps(_payload(), ensure_ascii=False), request=request
        )


@pytest.mark.parametrize(
    ("start_line", "end_line"), [(1, 1), (1, 2), (2, 3), (3, 3), (3, 4), (2, 5)]
)
def test_blank_quote_endpoint_error_identifies_stage_source_and_range(
    start_line, end_line
) -> None:
    async def scenario():
        request = replace(_request(), configured_prompt="\n条件\n \t\n続き\n　\n")
        payload = _payload()
        payload["instruction_checks"][0].update(
            start_line=start_line, end_line=end_line
        )
        llm = _RecordingLLM(json.dumps(payload, ensure_ascii=False))
        with pytest.raises(
            PromptDirectorError,
            match=(
                "生成段階.*instruction_checks\\[0\\].*configured_response_prompt.*空行"
                f".*{start_line}〜{end_line}.*5行"
            ),
        ):
            await PromptDirector(llm).create_directive(request)
        assert len(llm.calls) == 1

    asyncio.run(scenario())


def test_conditional_rule_keeps_its_premise_and_inactive_status() -> None:
    async def scenario():
        rule = "相談内容がまだ語られていない冒頭では、まず何を話したいかを尋ね、返答を待ってください。"
        request = replace(_request(), configured_prompt=rule)
        payload = _payload()
        payload["instruction_checks"][0].update(
            applicability="not_applicable",
            evidence="公開履歴で両者が会話についての相談内容を既に語っている。",
        )
        result = await PromptDirector(
            _RecordingLLM(json.dumps(payload, ensure_ascii=False))
        ).create_directive(request)
        rendered = render_prompt_director_instruction(result)
        assert rule in rendered
        assert "今回は適用しない" in rendered
        assert result.instruction_checks[0].evidence in rendered

    asyncio.run(scenario())


def test_unconfirmed_condition_is_rendered_as_unconfirmed() -> None:
    payload = _payload()
    payload["instruction_checks"][0].update(
        applicability="uncertain", evidence="必要な履歴が入力に含まれていない。"
    )
    rendered = render_prompt_director_instruction(
        parse_prompt_director_response(
            json.dumps(payload, ensure_ascii=False), request=_request()
        )
    )
    assert "適用条件が未確認" in rendered
    assert "必要な履歴が入力に含まれていない。" in rendered


@pytest.mark.parametrize("force", ["required", "prohibited", "preferred", "permitted"])
def test_director_preserves_instruction_force_priority_and_response_evidence(
    force,
) -> None:
    payload = _payload()
    excerpt = "昨日は少し話せた"
    payload["instruction_checks"][0].update(
        force=force,
        priority="この指示を優先する",
        response_excerpt=excerpt,
    )
    result = parse_prompt_director_response(
        json.dumps(payload, ensure_ascii=False), request=_request()
    )
    assert result.instruction_checks[0].force == force
    assert result.instruction_checks[0].response_excerpt == excerpt
    rendered = render_prompt_director_instruction(result)
    assert "この指示を優先する" in rendered
    assert result.context_basis in rendered
    assert result.instruction_checks[0].response_assessment in rendered


@pytest.mark.parametrize("excerpt", ["実際には書いていない言葉", "   ", ""])
def test_response_evidence_must_be_a_nonblank_exact_excerpt(excerpt) -> None:
    payload = _payload()
    payload["instruction_checks"][0]["response_excerpt"] = excerpt
    with pytest.raises(PromptDirectorError, match="JSON|発話本文"):
        parse_prompt_director_response(
            json.dumps(payload, ensure_ascii=False), request=_request()
        )


@pytest.mark.parametrize(
    "field",
    [
        "force",
        "priority",
        "response_excerpt",
        "response_assessment",
    ],
)
def test_instruction_interpretation_and_response_audit_are_required(field) -> None:
    payload = _payload()
    del payload["instruction_checks"][0][field]
    with pytest.raises(PromptDirectorError, match="JSON"):
        parse_prompt_director_response(
            json.dumps(payload, ensure_ascii=False), request=_request()
        )
