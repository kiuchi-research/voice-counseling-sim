from __future__ import annotations

import inspect
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from counseling_voice_demo.runtime.floor_mediator import TimingDecisionInput
from counseling_voice_demo.runtime.models import ParticipantConfig
from counseling_voice_demo.runtime.models import ActorKind
from counseling_voice_demo.runtime.protocols import StreamingLLMLike
from counseling_voice_demo.runtime.streaming_llm import FakeStreamingLLM
from counseling_voice_demo.runtime.timing_decision_prompt import (
    TIMING_DECISION_SYSTEM_PROMPT,
    build_timing_decision_input,
    parse_timing_decision_response,
)

AgentInputFormatter = Callable[..., str]
COUNSELOR_FAKE_RESPONSE_TEMPLATE = (
    "相談内容を受け止めました。{input_transcript} について少し整理しましょう。"
)
CLIENT_FAKE_RESPONSE_TEMPLATE = (
    "そうですね。{input_transcript} と聞いて、少し話しやすくなりました。"
)


@dataclass
class FakeAgent:
    speaker: str
    response_template: str
    received_inputs: list[str] = field(default_factory=list)
    timing_decisions: list[TimingDecisionInput] = field(default_factory=list)
    received_timing_inputs: list[str] = field(default_factory=list)
    overlap_decisions: list[TimingDecisionInput] = field(default_factory=list)
    received_overlap_inputs: list[str] = field(default_factory=list)

    async def generate(
        self,
        *,
        input_transcript: str,
        turn_id: int,
        additional_instruction: str | None = None,
    ) -> list[str]:
        self.received_inputs.append(
            prompt_with_additional_instruction(
                input_transcript,
                additional_instruction,
            )
        )
        llm = FakeStreamingLLM(response_template=self.response_template)
        parts = [
            part
            async for part in llm.stream_text(
                latest_input=input_transcript,
                turn_id=turn_id,
                speaker=self.speaker,
            )
        ]
        return parts

    async def decide_timing(
        self,
        *,
        input_transcript: str,
        turn_id: int,
        previous_speaker: str | None = None,
    ) -> TimingDecisionInput:
        self.received_timing_inputs.append(input_transcript)
        if self.timing_decisions:
            return self.timing_decisions.pop(0)
        _ = turn_id, previous_speaker
        return {
            "agent_id": self.speaker,
            "action": "REQUEST_MAIN_FLOOR",
        }

    async def decide_overlap(
        self,
        *,
        input_transcript: str,
        turn_id: int,
        conflict_agent_ids: tuple[str, ...],
        active_speaker_id: str | None = None,
    ) -> TimingDecisionInput:
        _ = turn_id, conflict_agent_ids, active_speaker_id
        self.received_overlap_inputs.append(input_transcript)
        if self.overlap_decisions:
            return self.overlap_decisions.pop(0)
        return {
            "agent_id": self.speaker,
            "action": "CONTINUE",
        }


@dataclass
class StreamingAgent:
    speaker: str
    llm: StreamingLLMLike
    system_prompt: str | None = None
    history: Iterable[Mapping[str, Any]] | None = None
    input_formatter: AgentInputFormatter | None = None
    timing_llm: StreamingLLMLike | None = None
    timing_system_prompt: str = TIMING_DECISION_SYSTEM_PROMPT
    received_inputs: list[str] = field(default_factory=list)
    received_timing_inputs: list[str] = field(default_factory=list)

    async def stream_generate(
        self,
        *,
        input_transcript: str,
        turn_id: int,
        additional_instruction: str | None = None,
    ):
        latest_input = input_transcript
        if self.input_formatter is not None:
            latest_input = _format_agent_input(
                self.input_formatter,
                speaker=self.speaker,
                turn_id=turn_id,
                input_transcript=input_transcript,
                purpose="generation",
            )
        self.received_inputs.append(
            prompt_with_additional_instruction(
                input_transcript,
                additional_instruction,
            )
        )
        latest_input = prompt_with_additional_instruction(
            latest_input,
            additional_instruction,
        )
        async for part in self.llm.stream_text(
            latest_input=latest_input,
            history=self.history,
            system_prompt=self.system_prompt,
        ):
            yield part

    async def generate(
        self,
        *,
        input_transcript: str,
        turn_id: int,
        additional_instruction: str | None = None,
    ) -> list[str]:
        return [
            part
            async for part in self.stream_generate(
                input_transcript=input_transcript,
                turn_id=turn_id,
                additional_instruction=additional_instruction,
            )
        ]

    async def decide_timing(
        self,
        *,
        input_transcript: str,
        turn_id: int,
        previous_speaker: str | None = None,
    ) -> TimingDecisionInput:
        latest_input = input_transcript
        if self.input_formatter is not None:
            latest_input = _format_agent_input(
                self.input_formatter,
                speaker=self.speaker,
                turn_id=turn_id,
                input_transcript=input_transcript,
                purpose="timing",
            )
        timing_input = build_timing_decision_input(
            speaker=self.speaker,
            turn_id=turn_id,
            input_transcript=latest_input,
            previous_speaker=previous_speaker,
        )
        self.received_timing_inputs.append(timing_input)
        llm = self.timing_llm or self.llm
        parts = [
            part
            async for part in llm.stream_text(
                latest_input=timing_input,
                system_prompt=self.timing_system_prompt,
            )
        ]
        return parse_timing_decision_response("".join(parts))


def prompt_with_additional_instruction(
    base_prompt: str,
    additional_instruction: str | None,
) -> str:
    if not additional_instruction:
        return base_prompt
    normalized_prompt = base_prompt.strip() or "（直前発話は空です。）"
    return "\n\n".join(
        [
            normalized_prompt,
            "追加の内部指示:",
            additional_instruction.strip(),
        ]
    )


def _format_agent_input(
    formatter: AgentInputFormatter,
    *,
    speaker: str,
    turn_id: int,
    input_transcript: str,
    purpose: str,
    response_target: str | None = None,
) -> str:
    kwargs: dict[str, Any] = {
        "speaker": speaker,
        "turn_id": turn_id,
        "input_transcript": input_transcript,
    }
    if _callable_accepts_keyword(formatter, "purpose"):
        kwargs["purpose"] = purpose
    if response_target is not None and _callable_accepts_keyword(
        formatter,
        "response_target",
    ):
        kwargs["response_target"] = response_target
    return formatter(**kwargs)


def _callable_accepts_keyword(callable_object: Callable[..., Any], key: str) -> bool:
    try:
        signature = inspect.signature(callable_object)
    except (TypeError, ValueError):
        return False
    return key in signature.parameters or any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )


def default_fake_agents(
    participants: Mapping[str, ParticipantConfig] | None = None,
) -> dict[str, FakeAgent]:
    if participants is not None:
        return {
            speaker_id: FakeAgent(
                speaker=speaker_id,
                response_template=_fake_response_template_for_role(participant.role),
            )
            for speaker_id, participant in participants.items()
            if participant.actor_kind is ActorKind.AI
        }
    return {
        "counselor": FakeAgent(
            speaker="counselor",
            response_template=COUNSELOR_FAKE_RESPONSE_TEMPLATE,
        ),
        "client": FakeAgent(
            speaker="client",
            response_template=CLIENT_FAKE_RESPONSE_TEMPLATE,
        ),
    }


def _fake_response_template_for_role(role: str) -> str:
    if role == "counselor":
        return COUNSELOR_FAKE_RESPONSE_TEMPLATE
    if role == "client":
        return CLIENT_FAKE_RESPONSE_TEMPLATE
    raise ValueError(f"unsupported participant role: {role}")
