from __future__ import annotations

import json

import pytest

from counseling_voice_demo.runtime.models import ParticipantConfig
from counseling_voice_demo.runtime.prompt_context import (
    PublicHistoryMessage,
    RuntimePromptContextData,
    RuntimePromptContextStore,
    build_prompt_context_input_formatter,
    build_prompt_context,
    format_prompt_context_input,
    build_realtime_prompt_input_formatter,
)

SHARED_CASE = "SHARED_CASE_TOKEN: public case formulation"
PUBLIC_HISTORY_TOKEN = "PUBLIC_HISTORY_TOKEN: already spoken in the room"
COUNSELOR_PUBLIC = "COUNSELOR_PUBLIC_TOKEN: visible counselor profile"
CLIENT_A_PUBLIC = "CLIENT_A_PUBLIC_TOKEN: visible client A profile"
CLIENT_B_PUBLIC = "CLIENT_B_PUBLIC_TOKEN: visible client B profile"
CLIENT_A_PRIVATE = "CLIENT_A_PRIVATE_TOKEN: only client A knows this"
CLIENT_B_PRIVATE = "CLIENT_B_PRIVATE_TOKEN: only client B knows this"


@pytest.mark.parametrize("speaker", ["client_a", "client_b", "counselor"])
def test_realtime_input_uses_each_speakers_actual_history_once(speaker) -> None:
    history = (
        PublicHistoryMessage("client_a", "私は短い連絡ならできそうです。"),
        PublicHistoryMessage("client_b", "私はまだ決められません。"),
        PublicHistoryMessage("counselor", "それぞれのお考えを聞かせてください。"),
    )
    store = RuntimePromptContextStore()
    store.set_context(
        3,
        RuntimePromptContextData(
            public_history=history, session_summary="以前は会う頻度を相談した。"
        ),
    )
    formatter = build_realtime_prompt_input_formatter(
        participants=_participants(),
        shared_case=SHARED_CASE,
        runtime_context_provider=store,
    )
    items = formatter(speaker=speaker, turn_id=3, input_transcript=history[-1].text)
    encoded = json.dumps(items, ensure_ascii=False)
    assert "以前は会う頻度を相談した。" in encoded
    for item, message in zip(items[1:-1], history, strict=True):
        assert item["role"] == (
            "assistant" if message.speaker_id == speaker else "user"
        )
        assert item["content"][0]["type"] == (
            "output_text" if message.speaker_id == speaker else "input_text"
        )
        assert json.loads(item["content"][0]["text"]) == {
            "speaker_id": message.speaker_id,
            "text": message.text,
        }
        assert encoded.count(message.text) == 1
    if speaker == "client_a":
        assert CLIENT_A_PRIVATE in encoded
        assert CLIENT_B_PRIVATE not in encoded
    elif speaker == "client_b":
        assert CLIENT_B_PRIVATE in encoded
        assert CLIENT_A_PRIVATE not in encoded
    else:
        assert CLIENT_A_PRIVATE not in encoded and CLIENT_B_PRIVATE not in encoded


def test_realtime_input_preserves_new_partial_input_without_inventing_speaker() -> None:
    store = RuntimePromptContextStore()
    store.set_context(
        4,
        RuntimePromptContextData(
            public_history=(PublicHistoryMessage("client_a", "前の確定発話。"),)
        ),
    )
    formatter = build_realtime_prompt_input_formatter(
        participants=_participants(),
        shared_case=SHARED_CASE,
        runtime_context_provider=store,
    )
    items = formatter(
        speaker="client_b", turn_id=4, input_transcript="新しい途中発話。"
    )
    assert json.loads(items[-2]["content"][0]["text"]) == {
        "speaker_id": "",
        "text": "新しい途中発話。",
    }
    assert items[-2]["role"] == "user"


def _participants() -> dict[str, ParticipantConfig]:
    return {
        "counselor": ParticipantConfig(
            speaker_id="counselor",
            role="counselor",
            display_name="Counselor",
            prompt_source="COUNSELOR_PROMPT_TOKEN",
            public_profile_source=COUNSELOR_PUBLIC,
        ),
        "client_a": ParticipantConfig(
            speaker_id="client_a",
            role="client",
            display_name="Client A",
            prompt_source="CLIENT_A_PROMPT_TOKEN",
            public_profile_source=CLIENT_A_PUBLIC,
            private_profile_source=CLIENT_A_PRIVATE,
        ),
        "client_b": ParticipantConfig(
            speaker_id="client_b",
            role="client",
            display_name="Client B",
            prompt_source="CLIENT_B_PROMPT_TOKEN",
            public_profile_source=CLIENT_B_PUBLIC,
            private_profile_source=CLIENT_B_PRIVATE,
        ),
    }


def _public_history() -> list[PublicHistoryMessage]:
    return [
        PublicHistoryMessage(speaker_id="counselor", text="Welcome."),
        PublicHistoryMessage(speaker_id="client_a", text=PUBLIC_HISTORY_TOKEN),
    ]


def test_client_a_context_contains_own_and_counselor_profiles() -> None:
    context = build_prompt_context(
        speaker_id="client_a",
        participants=_participants(),
        shared_case=SHARED_CASE,
        public_history=_public_history(),
    )

    prompt = context.as_text()

    assert SHARED_CASE in prompt
    assert PUBLIC_HISTORY_TOKEN in prompt
    assert "Client A（client_a / client）: " + PUBLIC_HISTORY_TOKEN in prompt
    assert COUNSELOR_PUBLIC in prompt
    assert CLIENT_A_PUBLIC in prompt
    assert CLIENT_B_PUBLIC not in prompt
    assert CLIENT_A_PRIVATE in prompt
    assert CLIENT_B_PRIVATE not in prompt


def test_client_b_context_contains_own_and_counselor_profiles() -> None:
    context = build_prompt_context(
        speaker_id="client_b",
        participants=_participants(),
        shared_case=SHARED_CASE,
        public_history=_public_history(),
    )

    prompt = context.as_text()

    assert SHARED_CASE in prompt
    assert PUBLIC_HISTORY_TOKEN in prompt
    assert COUNSELOR_PUBLIC in prompt
    assert CLIENT_B_PUBLIC in prompt
    assert CLIENT_A_PUBLIC not in prompt
    assert CLIENT_A_PRIVATE not in prompt
    assert CLIENT_B_PRIVATE in prompt


def test_counselor_context_excludes_client_profiles() -> None:
    context = build_prompt_context(
        speaker_id="counselor",
        participants=_participants(),
        shared_case=SHARED_CASE,
        public_history=_public_history(),
    )

    prompt = context.as_text()

    assert SHARED_CASE in prompt
    assert PUBLIC_HISTORY_TOKEN in prompt
    assert COUNSELOR_PUBLIC in prompt
    assert CLIENT_A_PUBLIC not in prompt
    assert CLIENT_B_PUBLIC not in prompt
    assert CLIENT_A_PRIVATE not in prompt
    assert CLIENT_B_PRIVATE not in prompt


def test_public_history_terms_are_visible_to_all_roles() -> None:
    prompts = [
        build_prompt_context(
            speaker_id=speaker_id,
            participants=_participants(),
            shared_case=SHARED_CASE,
            public_history=_public_history(),
        ).as_text()
        for speaker_id in ("counselor", "client_a", "client_b")
    ]

    assert all(PUBLIC_HISTORY_TOKEN in prompt for prompt in prompts)


def test_unknown_speaker_id_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown speaker_id"):
        build_prompt_context(
            speaker_id="client_c",
            participants=_participants(),
            shared_case=SHARED_CASE,
            public_history=_public_history(),
        )


def test_prompt_context_input_formatter_keeps_client_profiles_separate() -> None:
    formatter = build_prompt_context_input_formatter(
        participants=_participants(),
        shared_case=SHARED_CASE,
    )

    client_a_input = formatter(
        speaker="client_a",
        turn_id=2,
        input_transcript="直前発話です。",
    )
    client_b_input = formatter(
        speaker="client_b",
        turn_id=4,
        input_transcript="直前発話です。",
    )
    counselor_input = formatter(
        speaker="counselor",
        turn_id=3,
        input_transcript="直前発話です。",
    )

    assert CLIENT_A_PUBLIC in client_a_input
    assert COUNSELOR_PUBLIC in client_a_input
    assert CLIENT_B_PUBLIC not in client_a_input
    assert CLIENT_A_PRIVATE in client_a_input
    assert CLIENT_B_PRIVATE not in client_a_input
    assert CLIENT_B_PUBLIC in client_b_input
    assert COUNSELOR_PUBLIC in client_b_input
    assert CLIENT_A_PUBLIC not in client_b_input
    assert CLIENT_B_PRIVATE in client_b_input
    assert CLIENT_A_PRIVATE not in client_b_input
    assert CLIENT_A_PUBLIC not in counselor_input
    assert CLIENT_B_PUBLIC not in counselor_input
    assert CLIENT_A_PRIVATE not in counselor_input
    assert CLIENT_B_PRIVATE not in counselor_input


def test_client_b_generation_context_excludes_client_a_private_profile() -> None:
    formatter = build_prompt_context_input_formatter(
        participants=_participants(),
        shared_case=SHARED_CASE,
        runtime_context_provider=lambda **_kwargs: RuntimePromptContextData(
            public_history=(
                PublicHistoryMessage("counselor", "カウンセラー発話です。"),
                PublicHistoryMessage("client_a", "Aの公開発話です。"),
            ),
            current_objective="クライアントBが自分の視点で返答する。",
        ),
    )

    client_b_input = formatter(
        speaker="client_b",
        turn_id=4,
        input_transcript="Aの公開発話です。",
        purpose="generation",
    )

    assert CLIENT_B_PUBLIC in client_b_input
    assert CLIENT_B_PRIVATE in client_b_input
    assert CLIENT_A_PUBLIC not in client_b_input
    assert CLIENT_A_PRIVATE not in client_b_input
    assert "Aの公開発話です。" in client_b_input


def test_generation_prompt_identifies_self_and_other_client() -> None:
    formatter = build_prompt_context_input_formatter(
        participants=_participants(),
        shared_case=SHARED_CASE,
    )

    client_a_input = formatter(
        speaker="client_a",
        turn_id=2,
        input_transcript="直前発話です。",
        purpose="generation",
    )
    client_b_input = formatter(
        speaker="client_b",
        turn_id=4,
        input_transcript="直前発話です。",
        purpose="generation",
    )

    assert "Participant context:" in client_a_input
    assert "カウンセラー、自分、相手クライアントを含む3者対話" in client_a_input
    assert "あなた: client_a: Client A (client)" in client_a_input
    assert "このターンでは必ず「Client A」として" in client_a_input
    assert "相手クライアント:" in client_a_input
    assert "- client_b: Client B (client)" in client_a_input
    assert "カウンセラー:" in client_a_input
    assert "- counselor: Counselor (counselor)" in client_a_input
    assert "相手クライアントやカウンセラーの発話を代筆しない" in client_a_input
    assert CLIENT_B_PUBLIC not in client_a_input
    assert CLIENT_B_PRIVATE not in client_a_input

    assert "あなた: client_b: Client B (client)" in client_b_input
    assert "このターンでは必ず「Client B」として" in client_b_input
    assert "- client_a: Client A (client)" in client_b_input
    assert CLIENT_A_PUBLIC not in client_b_input
    assert CLIENT_A_PRIVATE not in client_b_input


def test_format_prompt_context_input_includes_runtime_target_and_latest_input() -> None:
    prompt = format_prompt_context_input(
        speaker="client_a",
        turn_id=2,
        input_transcript="家族との距離感に悩んでいます。",
        participants=_participants(),
        shared_case=SHARED_CASE,
        response_target="今回の主な宛先: Client B (client_b)",
    )

    assert "生成対象: turn=2 speaker=Client A（client_a / client）" in prompt
    assert "直前発話者: 不明" in prompt
    assert "Response target:" in prompt
    assert "今回の主な宛先: Client B (client_b)" in prompt
    assert "直前発話:" in prompt
    assert "家族との距離感に悩んでいます。" in prompt
    assert SHARED_CASE in prompt


def test_prompt_context_input_formatter_uses_runtime_context_for_generation_and_timing() -> (
    None
):
    store = RuntimePromptContextStore()
    store.set_context(
        3,
        RuntimePromptContextData(
            public_history=(
                PublicHistoryMessage("client_a", "直近Aです。"),
                PublicHistoryMessage("counselor", "直近カウンセラーです。"),
            ),
            session_summary="要約済みの長期文脈です。",
            current_objective="理想と現実のギャップを整理する。",
            response_target="今回の主な宛先: Client A (client_a)",
            previous_turn_target_hint=(
                "直前ターンでは、Client A（client_a）が "
                "Client B（client_b）を主な宛先として選んでいました。"
            ),
        ),
    )
    formatter = build_prompt_context_input_formatter(
        participants=_participants(),
        shared_case=SHARED_CASE,
        runtime_context_provider=store,
    )

    generation_prompt = formatter(
        speaker="client_b",
        turn_id=3,
        input_transcript="これは直前発話です。",
        purpose="generation",
    )
    timing_prompt = formatter(
        speaker="client_b",
        turn_id=3,
        input_transcript="これは直前発話です。",
        purpose="timing",
    )

    assert "Session summary:" in generation_prompt
    assert "要約済みの長期文脈です。" in generation_prompt
    assert "Current objective / phase:" in generation_prompt
    assert "Participant context:" in generation_prompt
    assert "カウンセラー、自分、相手クライアントを含む3者対話" in generation_prompt
    assert "あなた: client_b: Client B (client)" in generation_prompt
    assert "- client_a: Client A (client)" in generation_prompt
    assert "- counselor: Counselor (counselor)" in generation_prompt
    assert "Response target:" in generation_prompt
    assert "今回の主な宛先: Client A (client_a)" in generation_prompt
    assert "Previous turn target hint:" not in generation_prompt
    assert "Client A（client_a / client）: 直近Aです。" in generation_prompt
    assert (
        "Counselor（counselor / counselor）: 直近カウンセラーです。"
        in generation_prompt
    )
    assert "直前発話者: Counselor（counselor / counselor）" in generation_prompt
    assert "これは直前発話です。" in generation_prompt
    assert "Session summary:" in timing_prompt
    assert "要約済みの長期文脈です。" in timing_prompt
    assert "Current objective / phase:" in timing_prompt
    assert "理想と現実のギャップを整理する。" in timing_prompt
    assert "Previous turn target hint:" in timing_prompt
    assert "Client B（client_b）を主な宛先" in timing_prompt
    assert "Participant context:" in timing_prompt
    assert "Visible other profiles:" in timing_prompt
    assert "counselor: Counselor (counselor)" in timing_prompt
    assert "client_a: Client A (client)" in timing_prompt
    assert "client_b: Client B (client)" in timing_prompt
    assert SHARED_CASE in timing_prompt
    assert CLIENT_B_PUBLIC in timing_prompt
    assert CLIENT_B_PRIVATE in timing_prompt
    assert CLIENT_A_PUBLIC not in timing_prompt
    assert CLIENT_A_PRIVATE not in timing_prompt
    assert COUNSELOR_PUBLIC in timing_prompt
    assert "Response target:" not in timing_prompt
    assert "Client A（client_a / client）: 直近Aです。" in timing_prompt
    assert "Counselor（counselor / counselor）: 直近カウンセラーです。" in timing_prompt
    assert "直前発話者: Counselor（counselor / counselor）" in timing_prompt
    assert "これは直前発話です。" in timing_prompt
