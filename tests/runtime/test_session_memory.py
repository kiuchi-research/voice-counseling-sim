from __future__ import annotations

import pytest

from counseling_voice_demo.runtime.models import ParticipantConfig, TurnRuntimeState
from counseling_voice_demo.runtime.session_memory import (
    SessionSummaryState,
    build_runtime_prompt_context,
    build_session_summary_request,
    build_session_summary_update,
    format_session_summary_input,
    recent_public_history,
    should_update_session_summary,
)


def _turn(turn_id: int, speaker: str, text: str) -> TurnRuntimeState:
    state = TurnRuntimeState(
        session_id="session-test",
        turn_id=turn_id,
        speaker=speaker,
        input_transcript="",
    )
    state.generated_text = text
    return state


def test_recent_public_history_keeps_last_eight_visible_turns() -> None:
    turns = [
        _turn(index, f"speaker_{index % 3}", f"発話{index}") for index in range(12)
    ]

    history = recent_public_history(turns, limit=8)

    assert [message.text for message in history] == [
        f"発話{index}" for index in range(4, 12)
    ]
    assert history[0].speaker_id == "speaker_1"


def test_summary_request_targets_old_turns_excluding_recent_history() -> None:
    turns = [_turn(index, "client_a", f"発話{index}") for index in range(12)]

    request = build_session_summary_request(
        turns=turns,
        summary_state=SessionSummaryState(),
        current_objective="継続中",
        recent_turn_limit=8,
    )

    assert request is not None
    assert [message.text for message in request.turns_to_summarize] == [
        "発話0",
        "発話1",
        "発話2",
        "発話3",
    ]
    assert [message.text for message in request.recent_turns] == [
        f"発話{index}" for index in range(4, 12)
    ]
    assert request.last_turn_id_to_summarize == 3
    assert should_update_session_summary(
        turns=turns,
        summary_state=SessionSummaryState(),
        recent_turn_limit=8,
        trigger_completed_turns=12,
        update_interval_turns=6,
    )


def test_summary_checkpoint_advances_to_last_summarized_turn_id() -> None:
    turns = [_turn(index, "client_a", f"発話{index}") for index in range(12)]
    request = build_session_summary_request(
        turns=turns,
        summary_state=SessionSummaryState(),
        current_objective="継続中",
        recent_turn_limit=8,
    )

    update = build_session_summary_update(
        request=request,
        summary_text="更新済み要約",
    )

    assert update.text == "更新済み要約"
    assert update.last_summarized_turn_id == 3
    assert update.summarized_turn_count == 4


def test_existing_summary_waits_until_update_interval_is_reached() -> None:
    turns = [_turn(index, "client_a", f"発話{index}") for index in range(12)]

    assert not should_update_session_summary(
        turns=turns,
        summary_state=SessionSummaryState(text="既存要約"),
        recent_turn_limit=8,
        trigger_completed_turns=12,
        update_interval_turns=6,
    )


def test_runtime_prompt_context_includes_summary_objective_and_recent_history() -> None:
    turns = [_turn(index, "client_b", f"発話{index}") for index in range(10)]

    context = build_runtime_prompt_context(
        turns=turns,
        session_summary="長期要約",
        current_objective="目標設定",
        response_target="今回の主な宛先: 妻 (client_a)",
        previous_turn_target_hint="直前ターンでは、夫が妻を主な宛先として選んでいました。",
        recent_turn_limit=3,
        last_summarized_turn_id=6,
    )

    assert context.session_summary == "長期要約"
    assert context.current_objective == "目標設定"
    assert context.response_target == "今回の主な宛先: 妻 (client_a)"
    assert (
        context.previous_turn_target_hint
        == "直前ターンでは、夫が妻を主な宛先として選んでいました。"
    )
    assert [message.text for message in context.public_history] == [
        "発話7",
        "発話8",
        "発話9",
    ]


def test_format_session_summary_input_contains_structured_fields() -> None:
    turns = [_turn(index, "client_a", f"発話{index}") for index in range(12)]
    request = build_session_summary_request(
        turns=turns,
        summary_state=SessionSummaryState(text="既存要約"),
        current_objective="ギャップ整理",
        recent_turn_limit=8,
        participants={
            "counselor": ParticipantConfig(
                speaker_id="counselor",
                role="counselor",
                display_name="相談員",
            ),
            "client_x": ParticipantConfig(
                speaker_id="client_x",
                role="client",
                display_name="母",
            ),
            "client_y": ParticipantConfig(
                speaker_id="client_y",
                role="client",
                display_name="祖父",
            ),
        },
    )

    prompt = format_session_summary_input(request)

    assert "既存要約" in prompt
    assert "発話0" in prompt
    assert "発話11" in prompt
    assert "- 現在のフェーズ:" in prompt
    assert "- 相談員（counselor, counselor）の状態:" in prompt
    assert "- 母（client_x, client）の状態:" in prompt
    assert "- 祖父（client_y, client）の状態:" in prompt
    assert "クライアントA/妻" not in prompt
    assert "クライアントB/夫" not in prompt
    assert "各項目は1〜2文" in prompt
    assert "新たな事実や詳細を加えない" in prompt


@pytest.mark.parametrize("checkpoint,first_turn", [(-1, 0), (3, 4), (13, 7)])
def test_history_keeps_unsummarized_turns_and_at_least_recent_window(
    checkpoint, first_turn
) -> None:
    turns = [
        _turn(index, f"speaker_{index % 3}", f"発話{index}") for index in range(15)
    ]
    history = recent_public_history(turns, limit=8, retain_after_turn_id=checkpoint)
    assert [message.text for message in history] == [
        f"発話{index}" for index in range(first_turn, 15)
    ]
    assert [message.speaker_id for message in history] == [
        f"speaker_{index % 3}" for index in range(first_turn, 15)
    ]


def test_history_retention_uses_turn_ids_and_filters_empty_turns() -> None:
    turns = [
        _turn(2, "client_a", "最初の希望"),
        _turn(5, "client_b", ""),
        _turn(9, "client_b", "相手の希望"),
        _turn(15, "client_a", "同意していない"),
    ]
    history = recent_public_history(iter(turns), limit=1, retain_after_turn_id=2)
    assert [message.text for message in history] == ["相手の希望", "同意していない"]


def test_runtime_context_retains_history_until_summary_is_available() -> None:
    turns = [_turn(index, "client_a", f"発話{index}") for index in range(15)]
    context = build_runtime_prompt_context(
        turns=turns,
        session_summary="",
        current_objective="継続",
        recent_turn_limit=8,
        last_summarized_turn_id=6,
    )
    assert len(context.public_history) == 15


def test_empty_summary_cannot_advance_checkpoint() -> None:
    request = build_session_summary_request(
        turns=[_turn(index, "client_a", f"発話{index}") for index in range(12)],
        summary_state=SessionSummaryState(),
        current_objective="継続",
    )
    with pytest.raises(ValueError, match="empty|blank"):
        build_session_summary_update(request=request, summary_text="  \n")
