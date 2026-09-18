from __future__ import annotations

import asyncio

from counseling_voice_demo.runtime.agents import StreamingAgent
from counseling_voice_demo.runtime.floor_mediator import (
    TimingDecisionAction,
    TimingDecisionValidationError,
)


def test_streaming_agent_passes_formatted_latest_input_to_llm() -> None:
    class RecordingLLM:
        def __init__(self) -> None:
            self.latest_inputs: list[str] = []

        async def stream_text(self, *, latest_input, history=None, system_prompt=None):
            self.latest_inputs.append(latest_input)
            yield "応答"

    async def scenario() -> None:
        llm = RecordingLLM()
        agent = StreamingAgent(
            speaker="client",
            llm=llm,
            input_formatter=lambda *, speaker, turn_id, input_transcript: (
                f"{speaker}:{turn_id}:{input_transcript}"
            ),
        )

        parts = [
            part
            async for part in agent.stream_generate(
                input_transcript="直前発話",
                turn_id=2,
            )
        ]

        assert parts == ["応答"]
        assert llm.latest_inputs == ["client:2:直前発話"]
        assert agent.received_inputs == ["直前発話"]

    asyncio.run(scenario())


def test_streaming_agent_decide_timing_uses_main_floor_prompt() -> None:
    class RecordingLLM:
        def __init__(self) -> None:
            self.latest_inputs: list[str] = []
            self.system_prompts: list[str | None] = []

        async def stream_text(self, *, latest_input, history=None, system_prompt=None):
            _ = history
            self.latest_inputs.append(latest_input)
            self.system_prompts.append(system_prompt)
            yield '{"agent_id":"client_b","action":"REQUEST_MAIN_FLOOR"}'

    async def scenario() -> None:
        llm = RecordingLLM()
        agent = StreamingAgent(
            speaker="client_b",
            llm=llm,
            input_formatter=lambda *, speaker, turn_id, input_transcript: (
                f"context:{speaker}:{turn_id}:{input_transcript}"
            ),
        )

        decision = await agent.decide_timing(
            input_transcript="つらかったです。",
            turn_id=4,
            previous_speaker="client_a",
        )

        assert decision.agent_id == "client_b"
        assert decision.action is TimingDecisionAction.REQUEST_MAIN_FLOOR
        assert decision.prepared_intent is None
        assert "REQUEST_BACKCHANNEL" not in llm.system_prompts[0]
        assert "prepared_intent" in llm.system_prompts[0]
        assert "REQUEST_MAIN_FLOOR" in llm.system_prompts[0]
        assert "action に関わらず preferred_timing" in llm.system_prompts[0]
        assert "必須キー: agent_id, action, target" in llm.system_prompts[0]
        assert "action に関わらず target" in llm.system_prompts[0]
        assert "fallback で自分が次に話すことになった場合の間合い" in llm.system_prompts[0]
        assert "共有ケース、参加者一覧、自分のプロフィール" in llm.system_prompts[0]
        assert "他者の理解は、共有ケース・共通プロフィール" in llm.system_prompts[0]
        assert "他者の秘密プロフィール" in llm.system_prompts[0]
        assert "相手の同意への相づちだけなら WAIT" in llm.system_prompts[0]
        assert "自分への未回答の質問・確認" in llm.system_prompts[0]
        assert "同じ発話者順序の短いパターン" in llm.system_prompts[0]
        assert "同じ二者往復や同じ三者サイクル" in llm.system_prompts[0]
        assert "agent_id: client_b" in llm.latest_inputs[0]
        assert "previous_speaker: client_a" in llm.latest_inputs[0]
        assert '"target":"counselor"' in llm.latest_inputs[0]
        assert "context:client_b:4:つらかったです。" in llm.latest_inputs[0]

    asyncio.run(scenario())


def test_streaming_agent_decide_timing_rejects_non_json_response() -> None:
    class NonJsonLLM:
        async def stream_text(self, *, latest_input, history=None, system_prompt=None):
            _ = latest_input, history, system_prompt
            yield "相づちします。"

    async def scenario() -> None:
        agent = StreamingAgent(speaker="client_b", llm=NonJsonLLM())

        try:
            await agent.decide_timing(
                input_transcript="つらかったです。",
                turn_id=4,
                previous_speaker="client_a",
            )
        except TimingDecisionValidationError as exc:
            assert "JSON" in str(exc)
        else:
            raise AssertionError("non-JSON timing response should be rejected")

    asyncio.run(scenario())


def test_streaming_agent_decide_timing_rejects_backchannel_response() -> None:
    class BackchannelLLM:
        async def stream_text(self, *, latest_input, history=None, system_prompt=None):
            _ = latest_input, history, system_prompt
            yield '{"agent_id":"client_b","action":"REQUEST_BACKCHANNEL",'
            yield '"prepared_intent":"うん"}'

    async def scenario() -> None:
        agent = StreamingAgent(speaker="client_b", llm=BackchannelLLM())

        try:
            await agent.decide_timing(
                input_transcript="つらかったです。",
                turn_id=4,
                previous_speaker="client_a",
            )
        except TimingDecisionValidationError as exc:
            assert "REQUEST_BACKCHANNEL" in str(exc)
        else:
            raise AssertionError("backchannel timing response should be rejected")

    asyncio.run(scenario())
