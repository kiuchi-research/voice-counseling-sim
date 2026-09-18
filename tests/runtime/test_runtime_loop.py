from __future__ import annotations

import asyncio
import json
import random
import time

import pytest

from counseling_voice_demo.runtime.agents import FakeAgent, StreamingAgent
from counseling_voice_demo.runtime.controller import (
    ConversationRuntime,
    SpeakerSelection,
    TurnTakingSignalTracker,
    _initial_client_turn,
    _client_reply_opportunity_decision,
    _client_reply_priority_decision,
    _extended_client_reply_after_client_run_decision,
    _previous_turn_target_hint_from_selection,
    _select_contextual_fallback_speaker,
    _same_speaker_continuation_instruction,
    _should_refresh_timing_signal_decisions,
    _turn_start_delay_ms,
)
from counseling_voice_demo.runtime.floor_mediator import (
    TimingDecision,
    TimingDecisionAction,
)
from counseling_voice_demo.runtime.models import (
    ActorKind,
    AudioChunk,
    AudioDeliveryMode,
    EndOfAudio,
    HumanAudioInput,
    HumanAudioStreamStart,
    HumanTurnInput,
    ParticipantConfig,
    RuntimeConfig,
    RuntimePhase,
    TranscriptEvent,
    TurnRuntimeState,
)
from counseling_voice_demo.runtime.prompt_context import (
    RuntimePromptContextStore,
    build_prompt_context_input_formatter,
)
from counseling_voice_demo.runtime.prompt_director import (
    ClosingAssessment,
    SessionEndAssessment,
    PROMPT_DIRECTOR_REVIEW_SYSTEM_PROMPT,
    PromptDirector,
    PromptDirectorResult,
)
from counseling_voice_demo.runtime.realtime_speech import RealtimeSpeechEvent
from counseling_voice_demo.runtime.session_memory import SessionSummaryState
from counseling_voice_demo.runtime.streaming_llm import StreamingLLMError
from counseling_voice_demo.runtime.streaming_stt import (
    FakeStreamingSTT,
    iter_turn_audio_chunks,
)
from counseling_voice_demo.runtime.system_prompts import (
    COUNSELOR_SYSTEM_PROMPT,
)


class SlowFakeAgent(FakeAgent):
    delay_seconds = 0.02

    async def generate(
        self,
        *,
        input_transcript: str,
        turn_id: int,
        additional_instruction: str | None = None,
    ) -> list[str]:
        await asyncio.sleep(self.delay_seconds)
        return await super().generate(
            input_transcript=input_transcript,
            turn_id=turn_id,
            additional_instruction=additional_instruction,
        )


class VerySlowFakeAgent(SlowFakeAgent):
    delay_seconds = 0.08


class TimingErrorFakeAgent(FakeAgent):
    async def decide_timing(
        self,
        *,
        input_transcript: str,
        turn_id: int,
        previous_speaker: str | None = None,
    ):
        _ = input_transcript, turn_id, previous_speaker
        raise StreamingLLMError("OpenAI response incomplete: max_output_tokens")


class InterruptiblePlaybackFakeAgent(FakeAgent):
    def __init__(self, speaker: str, response_template: str) -> None:
        super().__init__(speaker, response_template)
        self.stop_current_response_playback_calls: list[dict[str, object]] = []

    async def stop_current_response_playback(self, **kwargs):
        self.stop_current_response_playback_calls.append(dict(kwargs))
        return {
            "played_ms": kwargs["played_ms"],
            "cancel_sent": kwargs.get("cancel_response", True),
            "truncate_sent": kwargs.get("truncate_item", True),
        }


class SlowInterruptiblePlaybackFakeAgent(InterruptiblePlaybackFakeAgent):
    delay_seconds = 0.08

    def __init__(self, speaker: str, response_template: str) -> None:
        super().__init__(speaker, response_template)
        self.generated_turns: list[int] = []

    async def generate(
        self,
        *,
        input_transcript: str,
        turn_id: int,
        additional_instruction: str | None = None,
    ) -> list[str]:
        await asyncio.sleep(self.delay_seconds)
        self.generated_turns.append(turn_id)
        return await super().generate(
            input_transcript=input_transcript,
            turn_id=turn_id,
            additional_instruction=additional_instruction,
        )


class SlowOverlapFakeAgent(FakeAgent):
    async def decide_overlap(
        self,
        *,
        input_transcript: str,
        turn_id: int,
        conflict_agent_ids: tuple[str, ...],
        active_speaker_id: str | None = None,
    ):
        await asyncio.sleep(0.05)
        return await super().decide_overlap(
            input_transcript=input_transcript,
            turn_id=turn_id,
            conflict_agent_ids=conflict_agent_ids,
            active_speaker_id=active_speaker_id,
        )


class SlowTimingFakeAgent(FakeAgent):
    async def decide_timing(
        self,
        *,
        input_transcript: str,
        turn_id: int,
        previous_speaker: str | None = None,
    ):
        await asyncio.sleep(0.02)
        return await super().decide_timing(
            input_transcript=input_transcript,
            turn_id=turn_id,
            previous_speaker=previous_speaker,
        )


class CoordinatedTimingFakeAgent(FakeAgent):
    def __init__(
        self,
        speaker: str,
        response_template: str,
        *,
        started: asyncio.Event,
        release: asyncio.Event,
        action: str,
    ) -> None:
        super().__init__(speaker, response_template)
        self.started = started
        self.release = release
        self.action = action

    async def decide_timing(
        self,
        *,
        input_transcript: str,
        turn_id: int,
        previous_speaker: str | None = None,
    ):
        self.received_timing_inputs.append(input_transcript)
        self.started.set()
        await self.release.wait()
        _ = turn_id, previous_speaker
        return {"agent_id": self.speaker, "action": self.action}


class ScriptedTextLLM:
    def __init__(self, *parts: str) -> None:
        self.parts = parts
        self.latest_inputs: list[str] = []
        self.system_prompts: list[str | None] = []

    async def stream_text(self, *, latest_input, history=None, system_prompt=None):
        _ = history
        self.latest_inputs.append(latest_input)
        self.system_prompts.append(system_prompt)
        for part in self.parts:
            yield part


class FailingThenRecordingSummaryLLM:
    def __init__(self) -> None:
        self.latest_inputs: list[str] = []

    async def stream_text(self, *, latest_input, history=None, system_prompt=None):
        _ = history, system_prompt
        self.latest_inputs.append(latest_input)
        if len(self.latest_inputs) == 1:
            raise StreamingLLMError("OpenAI response incomplete: max_output_tokens")
        yield "再試行で更新された要約"


class RecordingPromptDirector:
    def __init__(self) -> None:
        self.requests = []

    async def create_directive(self, request):
        self.requests.append(request)
        return PromptDirectorResult(
            instruction_checks=[
                {
                    "source": "fixed_role_constraints",
                    "quote": request.fixed_system_prompt,
                    "applicability": "applies",
                    "evidence": "対象の話者として応答する。",
                    "force": "required",
                    "priority": "指定なし",
                    "response_excerpt": None,
                    "response_assessment": "対象の話者として発話する。",
                }
            ],
            context_basis="共有履歴にある発言のみ参照。合意は未確認。",
            response_intent=f"{request.speaker_id}のturn {request.turn_id}の応答目的。",
            response_example="相談の入口に立てたことが、まず大切なのですね。",
        )


class FailingRealtimeAgent:
    async def stream_audio_response(self, **kwargs):
        _ = kwargs
        if False:
            yield None
        raise RuntimeError("realtime voice rejected")


class RealtimeProvisionalAgent:
    def __init__(self, speaker: str, *, action: str = "REQUEST_MAIN_FLOOR") -> None:
        self.speaker = speaker
        self.action = action
        self.stream_calls: list[dict] = []
        self.received_timing_inputs: list[str] = []

    async def stream_audio_response(
        self,
        *,
        session_id,
        turn_id,
        speaker,
        input_transcript,
        additional_instruction=None,
        format_input=True,
        sample_rate,
        sample_width_bits,
        channels,
        delivery_mode,
    ):
        self.stream_calls.append(
            {
                "turn_id": turn_id,
                "speaker": speaker,
                "input_transcript": input_transcript,
                "format_input": format_input,
                "additional_instruction": additional_instruction,
            }
        )
        await asyncio.sleep(0)
        yield {"text_delta": f"{self.speaker} realtime "}
        yield {
            "audio_chunk": AudioChunk(
                session_id=session_id,
                turn_id=turn_id,
                speaker=speaker,
                chunk_index=0,
                pcm=b"\x01\x00" * 120,
                sample_rate=sample_rate,
                sample_width_bits=sample_width_bits,
                channels=channels,
                duration_ms=5,
                delivery_mode=delivery_mode,
            )
        }
        yield {"text_delta": "done"}

    async def decide_timing(
        self,
        *,
        input_transcript: str,
        turn_id: int,
        previous_speaker: str | None = None,
    ):
        _ = turn_id, previous_speaker
        self.received_timing_inputs.append(input_transcript)
        await asyncio.sleep(0.01)
        return {"agent_id": self.speaker, "action": self.action}


class FailingAfterAudioRealtimeProvisionalAgent(RealtimeProvisionalAgent):
    def __init__(self, speaker: str, *, action: str = "REQUEST_MAIN_FLOOR") -> None:
        super().__init__(speaker, action=action)
        self.close_count = 0

    async def stream_audio_response(self, **kwargs):
        self.stream_calls.append(
            {
                "turn_id": kwargs["turn_id"],
                "speaker": kwargs["speaker"],
                "input_transcript": kwargs["input_transcript"],
                "format_input": kwargs.get("format_input", True),
                "additional_instruction": kwargs.get("additional_instruction"),
            }
        )
        if len(self.stream_calls) == 1:
            await asyncio.sleep(0)
            yield {"text_delta": f"{self.speaker} stale "}
            yield {
                "audio_chunk": AudioChunk(
                    session_id=kwargs["session_id"],
                    turn_id=kwargs["turn_id"],
                    speaker=kwargs["speaker"],
                    chunk_index=0,
                    pcm=b"\x01\x00" * 120,
                    sample_rate=kwargs["sample_rate"],
                    sample_width_bits=kwargs["sample_width_bits"],
                    channels=kwargs["channels"],
                    duration_ms=5,
                    delivery_mode=kwargs["delivery_mode"],
                )
            }
            await asyncio.sleep(0.05)
            raise RuntimeError("stale provisional failed after audio")
        await asyncio.sleep(0)
        yield {"text_delta": f"{self.speaker} final "}
        yield {
            "audio_chunk": AudioChunk(
                session_id=kwargs["session_id"],
                turn_id=kwargs["turn_id"],
                speaker=kwargs["speaker"],
                chunk_index=0,
                pcm=b"\x02\x00" * 120,
                sample_rate=kwargs["sample_rate"],
                sample_width_bits=kwargs["sample_width_bits"],
                channels=kwargs["channels"],
                duration_ms=5,
                delivery_mode=kwargs["delivery_mode"],
            )
        }
        yield {"text_delta": "done"}

    async def close(self) -> None:
        self.close_count += 1


class FailingFirstRealtimeProvisionalAgent(RealtimeProvisionalAgent):
    async def stream_audio_response(self, **kwargs):
        self.stream_calls.append(
            {
                "turn_id": kwargs["turn_id"],
                "speaker": kwargs["speaker"],
                "input_transcript": kwargs["input_transcript"],
                "format_input": kwargs.get("format_input", True),
                "additional_instruction": kwargs.get("additional_instruction"),
            }
        )
        if len(self.stream_calls) == 1:
            if False:
                yield None
            raise RuntimeError("provisional realtime rejected")
        await asyncio.sleep(0)
        yield {"text_delta": f"{self.speaker} fallback "}
        yield {
            "audio_chunk": AudioChunk(
                session_id=kwargs["session_id"],
                turn_id=kwargs["turn_id"],
                speaker=kwargs["speaker"],
                chunk_index=0,
                pcm=b"\x01\x00" * 120,
                sample_rate=kwargs["sample_rate"],
                sample_width_bits=kwargs["sample_width_bits"],
                channels=kwargs["channels"],
                duration_ms=5,
                delivery_mode=kwargs["delivery_mode"],
            )
        }
        yield {"text_delta": "done"}


@pytest.mark.parametrize(
    ("preferred_timing", "urgency", "expected_min_ms", "expected_max_ms"),
    [
        ("immediate", None, 100, 200),
        ("immediate", 1.0, 100, 200),
        ("natural_pause", 0.5, 400, 1200),
        ("natural_pause", 1.0, 100, 200),
        ("short_hold", 0.5, 1500, 1800),
        ("long_hold", 0.0, 1900, 3000),
    ],
)
def test_turn_start_delay_uses_random_range_for_preferred_timing(
    preferred_timing: str,
    urgency: float | None,
    expected_min_ms: int,
    expected_max_ms: int,
) -> None:
    decision = TimingDecision.validate(
        {
            "agent_id": "counselor",
            "action": "REQUEST_MAIN_FLOOR",
            "preferred_timing": preferred_timing,
            "urgency": urgency,
        }
    )

    delays = {
        _turn_start_delay_ms(decision, rng=random.Random(seed)) for seed in range(32)
    }

    assert len(delays) > 1
    assert all(expected_min_ms <= delay <= expected_max_ms for delay in delays)


def test_turn_start_delay_defaults_unknown_timing_to_natural_pause() -> None:
    decision = TimingDecision.validate(
        {
            "agent_id": "counselor",
            "action": "REQUEST_MAIN_FLOOR",
            "preferred_timing": "unknown",
            "urgency": 0.5,
        }
    )

    delays = {
        _turn_start_delay_ms(decision, rng=random.Random(seed)) for seed in range(32)
    }

    assert len(delays) > 1
    assert all(400 <= delay <= 1200 for delay in delays)


def test_turn_start_delay_uses_wait_decision_timing_for_fallback() -> None:
    decision = TimingDecision.validate(
        {
            "agent_id": "counselor",
            "action": "WAIT",
            "preferred_timing": "immediate",
            "urgency": 0.2,
        }
    )

    delays = {
        _turn_start_delay_ms(decision, rng=random.Random(seed)) for seed in range(32)
    }

    assert len(delays) > 1
    assert all(100 <= delay <= 200 for delay in delays)


def test_turn_start_delay_defaults_missing_timing_to_natural_pause() -> None:
    decision = TimingDecision.validate(
        {
            "agent_id": "counselor",
            "action": "WAIT",
        }
    )

    delays = {
        _turn_start_delay_ms(decision, rng=random.Random(seed)) for seed in range(32)
    }

    assert len(delays) > 1
    assert all(400 <= delay <= 1200 for delay in delays)


def test_previous_turn_target_hint_uses_explicit_timing_target() -> None:
    config = RuntimeConfig(
        session_id="session_previous_target_hint_test",
        participants={
            "counselor": ParticipantConfig(
                speaker_id="counselor",
                role="counselor",
                display_name="カウンセラー",
            ),
            "client_a": ParticipantConfig(
                speaker_id="client_a",
                role="client",
                display_name="妻",
            ),
            "client_b": ParticipantConfig(
                speaker_id="client_b",
                role="client",
                display_name="夫",
            ),
        },
    )

    hint = _previous_turn_target_hint_from_selection(
        SpeakerSelection(
            speaker="client_b",
            timing_decision=TimingDecision(
                agent_id="client_b",
                action=TimingDecisionAction.REQUEST_MAIN_FLOOR,
                target="client_a",
            ),
        ),
        config=config,
        speaker="client_b",
    )

    assert "直前ターンでは、夫（client_b） が 妻（client_a） を主な宛先" in hint
    assert "弱い参考情報" in hint
    assert "宛先参加者が応答する余地" in hint


def test_previous_turn_target_hint_ignores_defaulted_timing_target() -> None:
    config = RuntimeConfig(
        session_id="session_previous_defaulted_target_hint_test",
        participants={
            "counselor": ParticipantConfig(
                speaker_id="counselor",
                role="counselor",
                display_name="カウンセラー",
            ),
            "client_a": ParticipantConfig(
                speaker_id="client_a",
                role="client",
                display_name="妻",
            ),
            "client_b": ParticipantConfig(
                speaker_id="client_b",
                role="client",
                display_name="夫",
            ),
        },
    )

    hint = _previous_turn_target_hint_from_selection(
        SpeakerSelection(
            speaker="client_b",
            timing_decision=TimingDecision(
                agent_id="client_b",
                action=TimingDecisionAction.WAIT,
                target="client_a",
                target_defaulted=True,
            ),
        ),
        config=config,
        speaker="client_b",
    )

    assert hint == ""


def test_timing_signal_refresh_detects_early_partial_signal() -> None:
    assert _should_refresh_timing_signal_decisions(
        signal_text="少しずつ任せるのは試してみたい",
        final_transcript=(
            "少しずつ任せるのは試してみたいけど、"
            "今、夫はどう思ってるのか確認していい？"
        ),
    )
    assert _should_refresh_timing_signal_decisions(
        signal_text="少しずつ任せるのは試してみたいけど、今、夫はどう思ってるのか",
        final_transcript=(
            "少しずつ任せるのは試してみたいけど、"
            "今、夫はどう思ってるのか確認していい？"
        ),
    )


def test_timing_signal_refresh_keeps_only_complete_unchanged_text() -> None:
    assert not _should_refresh_timing_signal_decisions(
        signal_text="  Aさんはどうですか？\n", final_transcript="Aさんはどうですか？"
    )
    # A short suffix can change who is asked, even after most of the utterance.
    question = "そのように考えておられるのですね。これからの関わり方について、"
    assert _should_refresh_timing_signal_decisions(
        signal_text=question, final_transcript=question + "Bさんは？"
    )


@pytest.mark.parametrize(
    ("policy", "use_prefetch"),
    [
        ("turn_boundary_timing", False),
        ("turn_boundary_timing", True),
        ("distributed_timing", False),
    ],
)
@pytest.mark.parametrize("addressed", ["client_a", "client_b"])
def test_speaker_selection_waits_for_slow_addressed_participant(
    tmp_path, policy, use_prefetch, addressed
) -> None:
    async def scenario():
        other = "client_b" if addressed == "client_a" else "client_a"
        runtime = ConversationRuntime(
            config=RuntimeConfig(
                session_id="slow-addressed-participant",
                max_turns=10,
                speaker_selection_policy=policy,
                turn_taking_decision_timeout_ms=1,
                participants={
                    "counselor": ParticipantConfig("counselor", "counselor", "進行役"),
                    "client_a": ParticipantConfig("client_a", "client", "参加者A"),
                    "client_b": ParticipantConfig("client_b", "client", "参加者B"),
                },
                fixed_speaker_sequence=("counselor", "client_a", "client_b"),
            ),
            agents={
                "counselor": FakeAgent("counselor", "進行役"),
                addressed: SlowTimingFakeAgent(
                    addressed,
                    "回答",
                    timing_decisions=[
                        {
                            "agent_id": addressed,
                            "action": "REQUEST_MAIN_FLOOR",
                            "explicitly_addressed": True,
                            "target": "counselor",
                            "urgency": 0.3,
                        }
                    ],
                ),
                other: FakeAgent(
                    other,
                    "補足",
                    timing_decisions=[
                        {
                            "agent_id": other,
                            "action": "REQUEST_MAIN_FLOOR",
                            "explicitly_addressed": False,
                            "target": "counselor",
                            "urgency": 0.72,
                        }
                    ]
                    * 2,
                ),
            },
            sessions_dir=tmp_path,
        )
        await runtime.logger.start()
        try:
            tracker = None
            question = "指名された方はどうですか？"
            if use_prefetch:
                tracker = runtime._turn_taking_signal_tracker_for_next_turn(
                    target_turn_id=8,
                    previous_speaker="counselor",
                    started_monotonic=asyncio.get_running_loop().time(),
                )
                assert tracker is not None
                await tracker.observe(
                    TranscriptEvent(
                        session_id=runtime.config.session_id,
                        turn_id=7,
                        speaker="counselor",
                        transcript_type="final",
                        text=question,
                    )
                )
                preliminary, _ = await tracker.decisions()
                assert any(
                    d.agent_id == addressed
                    and d.reason_code == "timing_decision_timeout"
                    for d in preliminary
                )
            selection = await runtime._select_speaker_for_turn(
                8,
                input_transcript=question,
                previous_speaker="counselor",
                timing_signal_tracker=tracker,
            )
            assert selection.speaker == addressed
            assert selection.timing_decision.explicitly_addressed is True
        finally:
            await runtime.logger.close()

    asyncio.run(scenario())


def test_cancelling_timing_selection_cancels_pending_agent_request(tmp_path) -> None:
    async def scenario():
        started = asyncio.Event()
        cancelled = asyncio.Event()

        class WaitingTimingAgent(FakeAgent):
            async def decide_timing(self, **kwargs):
                started.set()
                try:
                    await asyncio.Event().wait()
                finally:
                    cancelled.set()

        runtime = ConversationRuntime(
            config=RuntimeConfig(
                session_id="cancel-timing-selection",
                speaker_selection_policy="turn_boundary_timing",
                turn_taking_decision_timeout_ms=1,
            ),
            agents={
                "counselor": FakeAgent("counselor", "質問"),
                "client": WaitingTimingAgent("client", "回答"),
            },
            sessions_dir=tmp_path,
        )
        await runtime.logger.start()
        task = asyncio.create_task(
            runtime._select_speaker_for_turn(
                2, input_transcript="どう思いますか？", previous_speaker="counselor"
            )
        )
        try:
            await asyncio.wait_for(started.wait(), timeout=1.0)
            await asyncio.sleep(0.01)
            assert not task.done()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert cancelled.is_set()
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            await runtime.logger.close()

    asyncio.run(scenario())


def test_turn_taking_signal_tracker_collects_short_final_below_min_chars() -> None:
    async def scenario() -> None:
        collected_inputs: list[str] = []
        logged_inputs: list[str] = []

        async def collect(input_transcript: str) -> list[TimingDecision]:
            collected_inputs.append(input_transcript)
            return [
                TimingDecision(
                    agent_id="client",
                    action=TimingDecisionAction.REQUEST_MAIN_FLOOR,
                )
            ]

        async def on_collected(
            input_transcript: str,
            decisions: list[TimingDecision],
        ) -> None:
            _ = decisions
            logged_inputs.append(input_transcript)

        tracker = TurnTakingSignalTracker(
            turn_id=2,
            previous_speaker="counselor",
            min_chars=40,
            collect=collect,
            on_collected=on_collected,
        )

        await tracker.observe(
            TranscriptEvent(
                session_id="session-short-final",
                turn_id=1,
                speaker="counselor",
                transcript_type="final",
                text="短い質問です",
            )
        )
        decisions, signal_text = await tracker.decisions()

        assert signal_text == "短い質問です"
        assert collected_inputs == ["短い質問です"]
        assert logged_inputs == ["短い質問です"]
        assert [decision.agent_id for decision in decisions] == ["client"]

    asyncio.run(scenario())


def test_human_counselor_signal_tracker_uses_lower_streaming_threshold(
    tmp_path,
) -> None:
    runtime = ConversationRuntime(
        config=RuntimeConfig(
            session_id="session-human-signal-threshold",
            interaction_mode="human_counselor_ai_client",
            max_turns=2,
            speaker_selection_policy="turn_boundary_timing",
            turn_taking_signal_min_chars=40,
            participants={
                "counselor": ParticipantConfig(
                    "counselor",
                    "counselor",
                    "カウンセラー",
                    actor_kind=ActorKind.HUMAN,
                ),
                "client": ParticipantConfig("client", "client", "クライアント"),
            },
        ),
        agents={"client": FakeAgent("client", "クライアント:{input_transcript}")},
        sessions_dir=tmp_path,
    )

    human_tracker = runtime._turn_taking_signal_tracker_for_next_turn(
        target_turn_id=2,
        previous_speaker="counselor",
        started_monotonic=0,
    )
    client_tracker = runtime._turn_taking_signal_tracker_for_next_turn(
        target_turn_id=2,
        previous_speaker="client",
        started_monotonic=0,
    )

    assert human_tracker is not None
    assert client_tracker is not None
    assert human_tracker.min_chars == 8
    assert client_tracker.min_chars == 40


def test_contextual_fallback_scores_previous_turn_target() -> None:
    config = RuntimeConfig(
        session_id="session_previous_turn_target_score_test",
        participants={
            "counselor": ParticipantConfig(
                speaker_id="counselor",
                role="counselor",
                display_name="カウンセラー",
            ),
            "client_a": ParticipantConfig(
                speaker_id="client_a",
                role="client",
                display_name="妻",
            ),
            "client_b": ParticipantConfig(
                speaker_id="client_b",
                role="client",
                display_name="夫",
            ),
        },
        fixed_speaker_sequence=("counselor", "client_a", "client_b"),
    )

    selection = _select_contextual_fallback_speaker(
        config=config,
        turns=(),
        turn_id=2,
        base_speaker="client_a",
        previous_speaker="client_a",
        previous_turn_target_speaker="client_b",
        decisions=[
            TimingDecision(
                agent_id="counselor",
                action=TimingDecisionAction.WAIT,
                target="client_a",
            ),
            TimingDecision(
                agent_id="client_a",
                action=TimingDecisionAction.WAIT,
                target="counselor",
            ),
            TimingDecision(
                agent_id="client_b",
                action=TimingDecisionAction.WAIT,
                target="client_a",
            ),
        ],
    )
    client_b_candidate = next(
        item for item in selection.candidates if item.speaker_id == "client_b"
    )

    assert "previous_turn_target" in client_b_candidate.reasons


def test_contextual_fallback_penalizes_previous_speaker() -> None:
    config = RuntimeConfig(
        session_id="session_previous_speaker_penalty_test",
        participants={
            "counselor": ParticipantConfig(
                speaker_id="counselor",
                role="counselor",
                display_name="カウンセラー",
            ),
            "client_a": ParticipantConfig(
                speaker_id="client_a",
                role="client",
                display_name="妻",
            ),
            "client_b": ParticipantConfig(
                speaker_id="client_b",
                role="client",
                display_name="夫",
            ),
        },
        fixed_speaker_sequence=("counselor", "client_a", "client_b"),
    )

    selection = _select_contextual_fallback_speaker(
        config=config,
        turns=(),
        turn_id=2,
        base_speaker="client_a",
        previous_speaker="client_a",
        previous_turn_target_speaker=None,
        decisions=[
            TimingDecision(agent_id="counselor", action=TimingDecisionAction.WAIT),
            TimingDecision(agent_id="client_a", action=TimingDecisionAction.WAIT),
            TimingDecision(agent_id="client_b", action=TimingDecisionAction.WAIT),
        ],
    )

    assert selection.speaker_id != "client_a"
    previous_candidate = next(
        item for item in selection.candidates if item.speaker_id == "client_a"
    )
    assert "previous_speaker" in previous_candidate.reasons


def test_contextual_fallback_strongly_penalizes_same_client_speaker() -> None:
    config = RuntimeConfig(
        session_id="session_same_client_fallback_penalty_test",
        participants={
            "counselor": ParticipantConfig(
                speaker_id="counselor",
                role="counselor",
                display_name="カウンセラー",
            ),
            "client_a": ParticipantConfig(
                speaker_id="client_a",
                role="client",
                display_name="妻",
            ),
            "client_b": ParticipantConfig(
                speaker_id="client_b",
                role="client",
                display_name="夫",
            ),
        },
        fixed_speaker_sequence=(
            "counselor",
            "client_b",
            "client_a",
            "counselor",
            "client_a",
            "client_b",
        ),
    )
    turns = [
        TurnRuntimeState(
            session_id=config.session_id,
            turn_id=turn_id,
            speaker=speaker,
            input_transcript=f"{speaker} の発話",
        )
        for turn_id, speaker in enumerate(["client_b", "counselor", "client_a"])
    ]

    selection = _select_contextual_fallback_speaker(
        config=config,
        turns=turns,
        turn_id=3,
        base_speaker="client_a",
        previous_speaker="client_a",
        previous_turn_target_speaker=None,
        decisions=[
            TimingDecision(agent_id="counselor", action=TimingDecisionAction.WAIT),
            TimingDecision(agent_id="client_a", action=TimingDecisionAction.WAIT),
            TimingDecision(agent_id="client_b", action=TimingDecisionAction.WAIT),
        ],
    )

    same_speaker_candidate = next(
        item for item in selection.candidates if item.speaker_id == "client_a"
    )
    assert selection.speaker_id != "client_a"
    assert "previous_speaker" in same_speaker_candidate.reasons
    assert same_speaker_candidate.score < 0


def test_contextual_fallback_lightly_penalizes_repeated_speaker_pattern() -> None:
    config = RuntimeConfig(
        session_id="session_repeated_speaker_pattern_penalty_test",
        participants={
            "counselor": ParticipantConfig(
                speaker_id="counselor",
                role="counselor",
                display_name="カウンセラー",
            ),
            "client_a": ParticipantConfig(
                speaker_id="client_a",
                role="client",
                display_name="妻",
            ),
            "client_b": ParticipantConfig(
                speaker_id="client_b",
                role="client",
                display_name="夫",
            ),
        },
        fixed_speaker_sequence=(
            "counselor",
            "client_a",
            "counselor",
            "client_b",
        ),
    )
    turns = [
        TurnRuntimeState(
            session_id=config.session_id,
            turn_id=turn_id,
            speaker=speaker,
            input_transcript=f"{speaker} の発話",
        )
        for turn_id, speaker in enumerate(
            ["counselor", "client_a", "counselor", "client_b", "counselor"]
        )
    ]

    selection = _select_contextual_fallback_speaker(
        config=config,
        turns=turns,
        turn_id=4,
        base_speaker="client_a",
        previous_speaker="counselor",
        previous_turn_target_speaker=None,
        decisions=[
            TimingDecision(agent_id="counselor", action=TimingDecisionAction.WAIT),
            TimingDecision(agent_id="client_a", action=TimingDecisionAction.WAIT),
            TimingDecision(agent_id="client_b", action=TimingDecisionAction.WAIT),
        ],
    )
    client_a_candidate = next(
        item for item in selection.candidates if item.speaker_id == "client_a"
    )

    assert selection.speaker_id == "client_a"
    assert "repeated_speaker_pattern" in client_a_candidate.reasons
    assert "fixed_sequence" in client_a_candidate.reasons


def test_same_speaker_continuation_instruction_marks_own_previous_turn() -> None:
    config = RuntimeConfig(
        session_id="session_same_speaker_instruction_test",
        participants={
            "counselor": ParticipantConfig(
                speaker_id="counselor",
                role="counselor",
                display_name="カウンセラー",
            ),
            "client_a": ParticipantConfig(
                speaker_id="client_a",
                role="client",
                display_name="妻",
            ),
        },
    )

    instruction = _same_speaker_continuation_instruction(
        config,
        speaker="client_a",
        previous_speaker="client_a",
    )

    assert instruction is not None
    assert "直前発話者も今回の話者も「妻（client_a）」" in instruction
    assert "直前発話を他者からの発話として受けない" in instruction
    assert "新しい情報を短く足してください" in instruction
    assert (
        _same_speaker_continuation_instruction(
            config,
            speaker="client_a",
            previous_speaker="counselor",
        )
        is None
    )


def test_runtime_completes_missing_turn_taking_metadata(tmp_path) -> None:
    async def scenario() -> None:
        counselor = FakeAgent(
            "counselor",
            "カウンセラー生成:{input_transcript}",
            timing_decisions=[{"agent_id": "counselor", "action": "WAIT"}],
        )
        client_a = FakeAgent("client_a", "A生成:{input_transcript}")
        client_b = FakeAgent(
            "client_b",
            "B生成:{input_transcript}",
            timing_decisions=[{"agent_id": "client_b", "action": "REQUEST_MAIN_FLOOR"}],
        )
        runtime = ConversationRuntime(
            config=RuntimeConfig(
                session_id="session_complete_timing_metadata_test",
                max_turns=1,
                speaker_selection_policy="turn_boundary_timing",
                participants={
                    "counselor": ParticipantConfig(
                        speaker_id="counselor",
                        role="counselor",
                        display_name="カウンセラー",
                    ),
                    "client_a": ParticipantConfig(
                        speaker_id="client_a",
                        role="client",
                        display_name="妻",
                        initial_transcript="Aの初回発話です。",
                    ),
                    "client_b": ParticipantConfig(
                        speaker_id="client_b",
                        role="client",
                        display_name="夫",
                    ),
                },
                fixed_speaker_sequence=(
                    "counselor",
                    "client_b",
                    "client_a",
                    "counselor",
                ),
            ),
            agents={
                "counselor": counselor,
                "client_a": client_a,
                "client_b": client_b,
            },
            sessions_dir=tmp_path,
        )

        turns = await runtime.run()

        assert [turn.speaker for turn in turns] == ["client_a", "client_b"]

        event_path = (
            tmp_path
            / "session_complete_timing_metadata_test"
            / "internal"
            / "events"
            / "session_complete_timing_metadata_test.jsonl"
        )
        events = [
            json.loads(line)
            for line in event_path.read_text(encoding="utf-8").splitlines()
        ]
        floor_event = next(
            event for event in events if event["event_type"] == "floor_granted"
        )
        decisions = {
            item["agent_id"]: item for item in floor_event["details"]["decisions"]
        }

        assert decisions["client_b"]["target"] == "client_a"
        assert decisions["client_b"]["preferred_timing"] == "immediate"
        assert decisions["client_b"]["urgency"] == 0.7

    asyncio.run(scenario())


def test_runtime_treats_invalid_timing_decision_as_wait(tmp_path) -> None:
    async def scenario() -> None:
        runtime = ConversationRuntime(
            config=RuntimeConfig(
                session_id="session_invalid_timing_decision_test",
                max_turns=1,
                speaker_selection_policy="turn_boundary_timing",
            ),
            agents={
                "counselor": FakeAgent(
                    "counselor",
                    "カウンセラー生成:{input_transcript}",
                    timing_decisions=["not json"],
                ),
                "client": FakeAgent(
                    "client",
                    "クライアント生成:{input_transcript}",
                    timing_decisions=[
                        {"agent_id": "client", "action": "REQUEST_MAIN_FLOOR"}
                    ],
                ),
            },
            sessions_dir=tmp_path,
        )
        await runtime.logger.start()
        try:
            decisions = await runtime._collect_timing_decisions(
                turn_id=1,
                input_transcript="相談したいです。",
                previous_speaker=None,
            )
        finally:
            await runtime.logger.close()

        decisions_by_agent = {decision.agent_id: decision for decision in decisions}

        assert decisions_by_agent["counselor"].action is TimingDecisionAction.WAIT
        assert decisions_by_agent["counselor"].reason_code == "timing_decision_invalid"
        assert (
            decisions_by_agent["client"].action
            is TimingDecisionAction.REQUEST_MAIN_FLOOR
        )

        event_path = (
            tmp_path
            / "session_invalid_timing_decision_test"
            / "internal"
            / "events"
            / "session_invalid_timing_decision_test.jsonl"
        )
        events = [
            json.loads(line)
            for line in event_path.read_text(encoding="utf-8").splitlines()
        ]

        assert any(
            event["event_type"] == "timing_decision_invalid"
            and event["speaker"] == "counselor"
            for event in events
        )
        assert not any(event["event_type"] == "runtime_error" for event in events)

    asyncio.run(scenario())


def test_runtime_treats_timing_llm_stream_error_as_wait(tmp_path) -> None:
    async def scenario() -> None:
        runtime = ConversationRuntime(
            config=RuntimeConfig(
                session_id="session_timing_llm_error_test",
                max_turns=1,
                speaker_selection_policy="turn_boundary_timing",
            ),
            agents={
                "counselor": TimingErrorFakeAgent(
                    "counselor",
                    "カウンセラー生成:{input_transcript}",
                ),
                "client": FakeAgent(
                    "client",
                    "クライアント生成:{input_transcript}",
                    timing_decisions=[
                        {"agent_id": "client", "action": "REQUEST_MAIN_FLOOR"}
                    ],
                ),
            },
            sessions_dir=tmp_path,
        )
        await runtime.logger.start()
        try:
            decisions = await runtime._collect_timing_decisions(
                turn_id=1,
                input_transcript="相談したいです。",
                previous_speaker=None,
            )
        finally:
            await runtime.logger.close()

        decisions_by_agent = {decision.agent_id: decision for decision in decisions}
        assert decisions_by_agent["counselor"].action is TimingDecisionAction.WAIT
        assert decisions_by_agent["counselor"].reason_code == "timing_decision_error"

        event_path = (
            tmp_path
            / "session_timing_llm_error_test"
            / "internal"
            / "events"
            / "session_timing_llm_error_test.jsonl"
        )
        events = [
            json.loads(line)
            for line in event_path.read_text(encoding="utf-8").splitlines()
        ]
        assert any(
            event["event_type"] == "timing_decision_error"
            and event["speaker"] == "counselor"
            and event["details"]["message"]
            == "OpenAI response incomplete: max_output_tokens"
            for event in events
        )
        assert not any(event["event_type"] == "runtime_error" for event in events)

    asyncio.run(scenario())


def test_runtime_retries_same_session_summary_range_after_failure(tmp_path) -> None:
    async def scenario() -> None:
        summary_llm = FailingThenRecordingSummaryLLM()
        runtime = ConversationRuntime(
            config=RuntimeConfig(
                session_id="session-summary-retry-range-test",
                conversation_context_recent_turns=2,
                session_summary_trigger_completed_turns=3,
                session_summary_update_interval_turns=2,
            ),
            session_summary_llm=summary_llm,
            sessions_dir=tmp_path,
        )

        def completed_turn(turn_id: int) -> TurnRuntimeState:
            turn = TurnRuntimeState(
                session_id=runtime.config.session_id,
                turn_id=turn_id,
                speaker="client",
                input_transcript="",
            )
            turn.generated_text = f"発話{turn_id}"
            return turn

        await runtime.logger.start()
        try:
            runtime.turns = [completed_turn(turn_id) for turn_id in range(3)]
            await runtime._maybe_schedule_session_summary_update(
                current_objective="継続中",
            )
            first_task = runtime._session_summary_task
            assert first_task is not None
            assert (
                len(runtime._prompt_director_public_history(input_transcript="発話2"))
                == 3
            )
            await asyncio.gather(first_task, return_exceptions=True)
            await runtime._finish_session_summary_task_if_ready()
            assert (
                len(runtime._prompt_director_public_history(input_transcript="発話2"))
                == 3
            )

            runtime.turns.append(completed_turn(3))
            await runtime._maybe_schedule_session_summary_update(
                current_objective="継続中",
            )
            retry_task = runtime._session_summary_task
            assert retry_task is not None
            await asyncio.gather(retry_task, return_exceptions=True)
            await runtime._finish_session_summary_task_if_ready()

            assert len(summary_llm.latest_inputs) == 2
            assert summary_llm.latest_inputs[1] == summary_llm.latest_inputs[0]
            assert runtime._session_summary_state.text == "再試行で更新された要約"
            assert runtime._session_summary_state.last_summarized_turn_id == 0
            assert [
                message.text
                for message in runtime._prompt_director_public_history(
                    input_transcript="発話3"
                )
            ] == ["発話1", "発話2", "発話3"]
        finally:
            await runtime._cancel_session_summary_task()
            await runtime.logger.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("client_wants_reply", [True, False])
def test_runtime_prioritizes_client_reply_only_when_client_requests_it(
    tmp_path,
    client_wants_reply,
) -> None:
    async def scenario() -> None:
        counselor = FakeAgent(
            "counselor",
            "カウンセラー生成:{input_transcript}",
            timing_decisions=[
                {
                    "agent_id": "counselor",
                    "action": "REQUEST_MAIN_FLOOR",
                    "target": "client_a",
                    "preferred_timing": "natural_pause",
                    "urgency": 0.6,
                    "explicitly_addressed": True,
                }
            ],
        )
        client_a = FakeAgent("client_a", "A生成:{input_transcript}")
        client_b = FakeAgent(
            "client_b",
            "B生成:{input_transcript}",
            timing_decisions=[
                {
                    "agent_id": "client_b",
                    "action": "REQUEST_MAIN_FLOOR" if client_wants_reply else "WAIT",
                    "target": "client_a",
                    "preferred_timing": "immediate",
                    "urgency": 0.7,
                }
            ],
        )
        runtime = ConversationRuntime(
            config=RuntimeConfig(
                session_id="session_client_reply_priority_test",
                max_turns=1,
                speaker_selection_policy="turn_boundary_timing",
                participants={
                    "counselor": ParticipantConfig(
                        speaker_id="counselor",
                        role="counselor",
                        display_name="カウンセラー",
                    ),
                    "client_a": ParticipantConfig(
                        speaker_id="client_a",
                        role="client",
                        display_name="妻",
                        initial_transcript="Aの初回発話です。",
                    ),
                    "client_b": ParticipantConfig(
                        speaker_id="client_b",
                        role="client",
                        display_name="夫",
                    ),
                },
                fixed_speaker_sequence=(
                    "counselor",
                    "client_b",
                    "client_a",
                    "counselor",
                ),
            ),
            agents={
                "counselor": counselor,
                "client_a": client_a,
                "client_b": client_b,
            },
            sessions_dir=tmp_path,
        )

        turns = await runtime.run()

        assert [turn.speaker for turn in turns] == [
            "client_a",
            "client_b" if client_wants_reply else "counselor",
        ]
        if not client_wants_reply:
            assert not client_b.received_inputs
            return

        event_path = (
            tmp_path
            / "session_client_reply_priority_test"
            / "internal"
            / "events"
            / "session_client_reply_priority_test.jsonl"
        )
        events = [
            json.loads(line)
            for line in event_path.read_text(encoding="utf-8").splitlines()
        ]
        floor_event = next(
            event
            for event in events
            if event["details"].get("reason") == "client_reply_to_previous_client"
        )

        assert floor_event["speaker_id"] == "client_b"
        assert floor_event["event_type"] == "floor_conflict_resolved"
        assert floor_event["details"]["reason"] == "client_reply_to_previous_client"
        assert floor_event["details"]["conflict_agent_ids"] == [
            "counselor",
            "client_b",
        ]

    asyncio.run(scenario())


def test_client_reply_pattern_yields_to_comparable_counselor_request() -> None:
    config = RuntimeConfig(
        session_id="session_client_reply_pattern_yield_test",
        participants={
            "counselor": ParticipantConfig(
                speaker_id="counselor",
                role="counselor",
                display_name="カウンセラー",
            ),
            "client_a": ParticipantConfig(
                speaker_id="client_a",
                role="client",
                display_name="妻",
            ),
            "client_b": ParticipantConfig(
                speaker_id="client_b",
                role="client",
                display_name="夫",
            ),
        },
    )
    turns = [
        TurnRuntimeState(
            session_id=config.session_id,
            turn_id=turn_id,
            speaker=speaker,
            input_transcript=f"{speaker} の発話",
        )
        for turn_id, speaker in enumerate(
            ["client_b", "counselor", "client_a", "client_b", "counselor", "client_a"]
        )
    ]
    decisions = [
        TimingDecision(
            agent_id="counselor",
            action=TimingDecisionAction.REQUEST_MAIN_FLOOR,
            target="client_a",
            preferred_timing="immediate",
            urgency=0.82,
        ),
        TimingDecision(
            agent_id="client_a",
            action=TimingDecisionAction.WAIT,
            target="counselor",
            reason_code="previous_speaker",
        ),
        TimingDecision(
            agent_id="client_b",
            action=TimingDecisionAction.REQUEST_MAIN_FLOOR,
            target="client_a",
            preferred_timing="immediate",
            urgency=0.78,
        ),
    ]

    assert (
        _client_reply_priority_decision(
            decisions,
            turns=turns,
            previous_speaker="client_a",
            config=config,
        )
        is None
    )
    assert (
        _client_reply_opportunity_decision(
            decisions,
            turns=turns,
            previous_speaker="client_a",
            config=config,
        )
        is None
    )


def test_client_reply_pattern_still_allows_stronger_immediate_reply() -> None:
    config = RuntimeConfig(
        session_id="session_strong_client_reply_pattern_test",
        participants={
            "counselor": ParticipantConfig(
                speaker_id="counselor",
                role="counselor",
                display_name="カウンセラー",
            ),
            "client_a": ParticipantConfig(
                speaker_id="client_a",
                role="client",
                display_name="妻",
            ),
            "client_b": ParticipantConfig(
                speaker_id="client_b",
                role="client",
                display_name="夫",
            ),
        },
    )
    turns = [
        TurnRuntimeState(
            session_id=config.session_id,
            turn_id=turn_id,
            speaker=speaker,
            input_transcript=f"{speaker} の発話",
        )
        for turn_id, speaker in enumerate(
            ["client_b", "counselor", "client_a", "client_b", "counselor", "client_a"]
        )
    ]
    decisions = [
        TimingDecision(
            agent_id="counselor",
            action=TimingDecisionAction.REQUEST_MAIN_FLOOR,
            target="client_a",
            preferred_timing="natural_pause",
            urgency=0.72,
        ),
        TimingDecision(
            agent_id="client_a",
            action=TimingDecisionAction.WAIT,
            target="counselor",
            reason_code="previous_speaker",
        ),
        TimingDecision(
            agent_id="client_b",
            action=TimingDecisionAction.REQUEST_MAIN_FLOOR,
            target="client_a",
            preferred_timing="immediate",
            urgency=0.9,
        ),
    ]

    decision = _client_reply_priority_decision(
        decisions,
        turns=turns,
        previous_speaker="client_a",
        config=config,
    )

    assert decision is not None
    assert decision.agent_id == "client_b"


def test_extended_client_reply_allows_direct_reply_after_counselor_started_exchange() -> (
    None
):
    config = RuntimeConfig(
        session_id="session_direct_client_reply_extension_test",
        participants={
            "counselor": ParticipantConfig(
                speaker_id="counselor",
                role="counselor",
                display_name="カウンセラー",
            ),
            "client_a": ParticipantConfig(
                speaker_id="client_a",
                role="client",
                display_name="妻",
            ),
            "client_b": ParticipantConfig(
                speaker_id="client_b",
                role="client",
                display_name="夫",
            ),
        },
    )
    turns = [
        TurnRuntimeState(
            session_id=config.session_id,
            turn_id=turn_id,
            speaker=speaker,
            input_transcript=f"{speaker} の発話",
        )
        for turn_id, speaker in enumerate(["counselor", "client_a", "client_b"])
    ]
    decisions = [
        TimingDecision(
            agent_id="counselor",
            action=TimingDecisionAction.REQUEST_MAIN_FLOOR,
            target="client_b",
            preferred_timing="natural_pause",
            urgency=0.62,
        ),
        TimingDecision(
            agent_id="client_a",
            action=TimingDecisionAction.REQUEST_MAIN_FLOOR,
            target="client_b",
            preferred_timing="natural_pause",
            urgency=0.74,
        ),
        TimingDecision(
            agent_id="client_b",
            action=TimingDecisionAction.WAIT,
            target="counselor",
            reason_code="previous_speaker",
        ),
    ]

    decision = _extended_client_reply_after_client_run_decision(
        decisions,
        turns=turns,
        previous_speaker="client_b",
        config=config,
    )

    assert decision is not None
    assert decision.agent_id == "client_a"


def test_extended_client_reply_rejects_soft_direct_reply_after_two_client_turns() -> (
    None
):
    config = RuntimeConfig(
        session_id="session_soft_direct_client_reply_extension_test",
        participants={
            "counselor": ParticipantConfig(
                speaker_id="counselor",
                role="counselor",
                display_name="カウンセラー",
            ),
            "client_a": ParticipantConfig(
                speaker_id="client_a",
                role="client",
                display_name="妻",
            ),
            "client_b": ParticipantConfig(
                speaker_id="client_b",
                role="client",
                display_name="夫",
            ),
        },
    )
    turns = [
        TurnRuntimeState(
            session_id=config.session_id,
            turn_id=turn_id,
            speaker=speaker,
            input_transcript=f"{speaker} の発話",
        )
        for turn_id, speaker in enumerate(["counselor", "client_a", "client_b"])
    ]
    decisions = [
        TimingDecision(
            agent_id="counselor",
            action=TimingDecisionAction.REQUEST_MAIN_FLOOR,
            target="client_b",
            preferred_timing="natural_pause",
            urgency=0.62,
        ),
        TimingDecision(
            agent_id="client_a",
            action=TimingDecisionAction.REQUEST_MAIN_FLOOR,
            target="client_b",
            preferred_timing="natural_pause",
            urgency=0.64,
        ),
        TimingDecision(
            agent_id="client_b",
            action=TimingDecisionAction.WAIT,
            target="counselor",
            reason_code="previous_speaker",
        ),
    ]

    assert (
        _extended_client_reply_after_client_run_decision(
            decisions,
            turns=turns,
            previous_speaker="client_b",
            config=config,
        )
        is None
    )


def test_extended_client_reply_allows_mid_strength_direct_reply_after_two_client_turns() -> (
    None
):
    config = RuntimeConfig(
        session_id="session_mid_direct_client_reply_extension_test",
        participants={
            "counselor": ParticipantConfig(
                speaker_id="counselor",
                role="counselor",
                display_name="カウンセラー",
            ),
            "client_a": ParticipantConfig(
                speaker_id="client_a",
                role="client",
                display_name="妻",
            ),
            "client_b": ParticipantConfig(
                speaker_id="client_b",
                role="client",
                display_name="夫",
            ),
        },
    )
    turns = [
        TurnRuntimeState(
            session_id=config.session_id,
            turn_id=turn_id,
            speaker=speaker,
            input_transcript=f"{speaker} の発話",
        )
        for turn_id, speaker in enumerate(["counselor", "client_a", "client_b"])
    ]
    decisions = [
        TimingDecision(
            agent_id="counselor",
            action=TimingDecisionAction.REQUEST_MAIN_FLOOR,
            target="client_b",
            preferred_timing="natural_pause",
            urgency=0.58,
        ),
        TimingDecision(
            agent_id="client_a",
            action=TimingDecisionAction.REQUEST_MAIN_FLOOR,
            target="client_b",
            preferred_timing="natural_pause",
            urgency=0.63,
        ),
        TimingDecision(
            agent_id="client_b",
            action=TimingDecisionAction.WAIT,
            target="counselor",
            reason_code="previous_speaker",
        ),
    ]

    decision = _extended_client_reply_after_client_run_decision(
        decisions,
        turns=turns,
        previous_speaker="client_b",
        config=config,
    )

    assert decision is not None
    assert decision.agent_id == "client_a"


def test_extended_client_reply_allows_strong_counselor_target_after_two_client_turns() -> (
    None
):
    config = RuntimeConfig(
        session_id="session_counselor_target_client_reply_extension_test",
        participants={
            "counselor": ParticipantConfig(
                speaker_id="counselor",
                role="counselor",
                display_name="カウンセラー",
            ),
            "client_a": ParticipantConfig(
                speaker_id="client_a",
                role="client",
                display_name="妻",
            ),
            "client_b": ParticipantConfig(
                speaker_id="client_b",
                role="client",
                display_name="夫",
            ),
        },
    )
    turns = [
        TurnRuntimeState(
            session_id=config.session_id,
            turn_id=turn_id,
            speaker=speaker,
            input_transcript=f"{speaker} の発話",
        )
        for turn_id, speaker in enumerate(["counselor", "client_a", "client_b"])
    ]
    decisions = [
        TimingDecision(
            agent_id="counselor",
            action=TimingDecisionAction.REQUEST_MAIN_FLOOR,
            target="client_b",
            preferred_timing="natural_pause",
            urgency=0.52,
        ),
        TimingDecision(
            agent_id="client_a",
            action=TimingDecisionAction.REQUEST_MAIN_FLOOR,
            target="counselor",
            preferred_timing="natural_pause",
            urgency=0.62,
        ),
        TimingDecision(
            agent_id="client_b",
            action=TimingDecisionAction.WAIT,
            target="counselor",
            reason_code="previous_speaker",
        ),
    ]

    decision = _extended_client_reply_after_client_run_decision(
        decisions,
        turns=turns,
        previous_speaker="client_b",
        config=config,
    )

    assert decision is not None
    assert decision.agent_id == "client_a"


def test_extended_client_reply_rejects_soft_counselor_target_after_two_client_turns() -> (
    None
):
    config = RuntimeConfig(
        session_id="session_soft_counselor_target_client_reply_extension_test",
        participants={
            "counselor": ParticipantConfig(
                speaker_id="counselor",
                role="counselor",
                display_name="カウンセラー",
            ),
            "client_a": ParticipantConfig(
                speaker_id="client_a",
                role="client",
                display_name="妻",
            ),
            "client_b": ParticipantConfig(
                speaker_id="client_b",
                role="client",
                display_name="夫",
            ),
        },
    )
    turns = [
        TurnRuntimeState(
            session_id=config.session_id,
            turn_id=turn_id,
            speaker=speaker,
            input_transcript=f"{speaker} の発話",
        )
        for turn_id, speaker in enumerate(["counselor", "client_a", "client_b"])
    ]
    decisions = [
        TimingDecision(
            agent_id="counselor",
            action=TimingDecisionAction.REQUEST_MAIN_FLOOR,
            target="client_b",
            preferred_timing="natural_pause",
            urgency=0.56,
        ),
        TimingDecision(
            agent_id="client_a",
            action=TimingDecisionAction.REQUEST_MAIN_FLOOR,
            target="counselor",
            preferred_timing="natural_pause",
            urgency=0.61,
        ),
        TimingDecision(
            agent_id="client_b",
            action=TimingDecisionAction.WAIT,
            target="counselor",
            reason_code="previous_speaker",
        ),
    ]

    assert (
        _extended_client_reply_after_client_run_decision(
            decisions,
            turns=turns,
            previous_speaker="client_b",
            config=config,
        )
        is None
    )


def test_extended_client_reply_keeps_counselor_when_client_run_was_not_counselor_started() -> (
    None
):
    config = RuntimeConfig(
        session_id="session_no_direct_client_reply_extension_test",
        participants={
            "counselor": ParticipantConfig(
                speaker_id="counselor",
                role="counselor",
                display_name="カウンセラー",
            ),
            "client_a": ParticipantConfig(
                speaker_id="client_a",
                role="client",
                display_name="妻",
            ),
            "client_b": ParticipantConfig(
                speaker_id="client_b",
                role="client",
                display_name="夫",
            ),
        },
    )
    turns = [
        TurnRuntimeState(
            session_id=config.session_id,
            turn_id=turn_id,
            speaker=speaker,
            input_transcript=f"{speaker} の発話",
        )
        for turn_id, speaker in enumerate(["client_a", "client_b"])
    ]
    decisions = [
        TimingDecision(
            agent_id="counselor",
            action=TimingDecisionAction.REQUEST_MAIN_FLOOR,
            target="client_b",
            preferred_timing="natural_pause",
            urgency=0.55,
        ),
        TimingDecision(
            agent_id="client_a",
            action=TimingDecisionAction.REQUEST_MAIN_FLOOR,
            target="client_b",
            preferred_timing="natural_pause",
            urgency=0.7,
        ),
        TimingDecision(
            agent_id="client_b",
            action=TimingDecisionAction.WAIT,
            target="counselor",
            reason_code="previous_speaker",
        ),
    ]

    assert (
        _extended_client_reply_after_client_run_decision(
            decisions,
            turns=turns,
            previous_speaker="client_b",
            config=config,
        )
        is None
    )


def test_runtime_allows_contextual_client_reply_even_when_targeting_counselor(
    tmp_path,
) -> None:
    async def scenario() -> None:
        counselor = FakeAgent(
            "counselor",
            "カウンセラー生成:{input_transcript}",
            timing_decisions=[
                {
                    "agent_id": "counselor",
                    "action": "REQUEST_MAIN_FLOOR",
                    "target": "client_a",
                    "preferred_timing": "natural_pause",
                    "urgency": 0.62,
                }
            ],
        )
        client_a = FakeAgent(
            "client_a",
            "A生成:{input_transcript}",
        )
        client_b = FakeAgent(
            "client_b",
            "B生成:{input_transcript}",
            timing_decisions=[
                {
                    "agent_id": "client_b",
                    "action": "REQUEST_MAIN_FLOOR",
                    "target": "counselor",
                    "preferred_timing": "natural_pause",
                    "urgency": 0.55,
                }
            ],
        )
        runtime = ConversationRuntime(
            config=RuntimeConfig(
                session_id="session_contextual_client_reply_counselor_target_test",
                max_turns=1,
                speaker_selection_policy="turn_boundary_timing",
                turn_taking_max_reconsider_rounds=0,
                participants={
                    "counselor": ParticipantConfig(
                        speaker_id="counselor",
                        role="counselor",
                        display_name="カウンセラー",
                    ),
                    "client_a": ParticipantConfig(
                        speaker_id="client_a",
                        role="client",
                        display_name="妻",
                        initial_transcript=(
                            "毎日の声かけをほとんど私がしていて、"
                            "夫にも少し分かってほしいです。"
                        ),
                    ),
                    "client_b": ParticipantConfig(
                        speaker_id="client_b",
                        role="client",
                        display_name="夫",
                    ),
                },
                fixed_speaker_sequence=(
                    "counselor",
                    "client_a",
                    "counselor",
                    "client_b",
                ),
            ),
            agents={
                "counselor": counselor,
                "client_a": client_a,
                "client_b": client_b,
            },
            sessions_dir=tmp_path,
        )

        turns = await runtime.run()

        assert [turn.speaker for turn in turns] == ["client_a", "client_b"]
        assert any(
            "今回の主な宛先は「カウンセラー（counselor）」です。" in item
            for item in client_b.received_inputs
        )

        event_path = (
            tmp_path
            / "session_contextual_client_reply_counselor_target_test"
            / "internal"
            / "events"
            / "session_contextual_client_reply_counselor_target_test.jsonl"
        )
        events = [
            json.loads(line)
            for line in event_path.read_text(encoding="utf-8").splitlines()
        ]
        floor_event = next(
            event
            for event in events
            if event["details"].get("reason") == "client_contextual_reply_after_client"
        )

        assert floor_event["speaker_id"] == "client_b"
        assert floor_event["event_type"] == "floor_conflict_resolved"

    asyncio.run(scenario())


def test_runtime_weakens_client_reply_priority_after_two_client_turns(
    tmp_path,
) -> None:
    async def scenario() -> None:
        counselor = FakeAgent(
            "counselor",
            "カウンセラー生成:{input_transcript}",
            timing_decisions=[
                {
                    "agent_id": "counselor",
                    "action": "REQUEST_MAIN_FLOOR",
                    "target": "client_a",
                    "preferred_timing": "natural_pause",
                    "urgency": 0.5,
                },
                {
                    "agent_id": "counselor",
                    "action": "REQUEST_MAIN_FLOOR",
                    "target": "client_b",
                    "preferred_timing": "natural_pause",
                    "urgency": 0.55,
                },
            ],
        )
        client_a = FakeAgent(
            "client_a",
            "A生成:{input_transcript}",
            timing_decisions=[
                {
                    "agent_id": "client_a",
                    "action": "REQUEST_MAIN_FLOOR",
                    "target": "client_b",
                    "preferred_timing": "natural_pause",
                    "urgency": 0.7,
                }
            ],
        )
        client_b = FakeAgent(
            "client_b",
            "B生成:{input_transcript}",
            timing_decisions=[
                {
                    "agent_id": "client_b",
                    "action": "REQUEST_MAIN_FLOOR",
                    "target": "client_a",
                    "preferred_timing": "immediate",
                    "urgency": 0.7,
                }
            ],
        )
        runtime = ConversationRuntime(
            config=RuntimeConfig(
                session_id="session_client_reply_priority_decay_test",
                max_turns=2,
                speaker_selection_policy="turn_boundary_timing",
                participants={
                    "counselor": ParticipantConfig(
                        speaker_id="counselor",
                        role="counselor",
                        display_name="カウンセラー",
                    ),
                    "client_a": ParticipantConfig(
                        speaker_id="client_a",
                        role="client",
                        display_name="妻",
                        initial_transcript="Aの初回発話です。",
                    ),
                    "client_b": ParticipantConfig(
                        speaker_id="client_b",
                        role="client",
                        display_name="夫",
                    ),
                },
                fixed_speaker_sequence=(
                    "counselor",
                    "client_b",
                    "client_a",
                    "counselor",
                ),
            ),
            agents={
                "counselor": counselor,
                "client_a": client_a,
                "client_b": client_b,
            },
            sessions_dir=tmp_path,
        )

        turns = await runtime.run()

        assert [turn.speaker for turn in turns] == [
            "client_a",
            "client_b",
            "counselor",
        ]

        event_path = (
            tmp_path
            / "session_client_reply_priority_decay_test"
            / "internal"
            / "events"
            / "session_client_reply_priority_decay_test.jsonl"
        )
        events = [
            json.loads(line)
            for line in event_path.read_text(encoding="utf-8").splitlines()
        ]
        floor_event = next(
            event
            for event in events
            if event["details"].get("reason") == "counselor_reentry_after_client_run"
        )

        assert floor_event["speaker_id"] == "counselor"
        assert floor_event["event_type"] == "floor_conflict_resolved"
        assert floor_event["details"]["conflict_agent_ids"] == [
            "counselor",
            "client_a",
        ]

    asyncio.run(scenario())


def test_runtime_allows_client_request_after_two_client_turns_when_counselor_waits(
    tmp_path,
) -> None:
    async def scenario() -> None:
        counselor = FakeAgent(
            "counselor",
            "カウンセラー生成:{input_transcript}",
            timing_decisions=[
                {
                    "agent_id": "counselor",
                    "action": "WAIT",
                    "target": "client_a",
                    "preferred_timing": "natural_pause",
                    "urgency": 0.2,
                    "reason_code": "timing_decision_timeout",
                },
                {
                    "agent_id": "counselor",
                    "action": "WAIT",
                    "target": "client_b",
                    "preferred_timing": "natural_pause",
                    "urgency": 0.2,
                    "reason_code": "timing_decision_timeout",
                },
            ],
        )
        client_a = FakeAgent(
            "client_a",
            "A生成:{input_transcript}",
            timing_decisions=[
                {
                    "agent_id": "client_a",
                    "action": "REQUEST_MAIN_FLOOR",
                    "target": "client_b",
                    "preferred_timing": "natural_pause",
                    "urgency": 0.58,
                }
            ],
        )
        client_b = FakeAgent(
            "client_b",
            "B生成:{input_transcript}",
            timing_decisions=[
                {
                    "agent_id": "client_b",
                    "action": "REQUEST_MAIN_FLOOR",
                    "target": "client_a",
                    "preferred_timing": "natural_pause",
                    "urgency": 0.62,
                }
            ],
        )
        runtime = ConversationRuntime(
            config=RuntimeConfig(
                session_id="session_client_run_single_request_reentry_test",
                max_turns=2,
                speaker_selection_policy="turn_boundary_timing",
                turn_taking_max_reconsider_rounds=0,
                participants={
                    "counselor": ParticipantConfig(
                        speaker_id="counselor",
                        role="counselor",
                        display_name="カウンセラー",
                    ),
                    "client_a": ParticipantConfig(
                        speaker_id="client_a",
                        role="client",
                        display_name="妻",
                        initial_transcript="Aの初回発話です。",
                    ),
                    "client_b": ParticipantConfig(
                        speaker_id="client_b",
                        role="client",
                        display_name="夫",
                    ),
                },
                fixed_speaker_sequence=(
                    "counselor",
                    "client_b",
                    "client_a",
                    "counselor",
                ),
            ),
            agents={
                "counselor": counselor,
                "client_a": client_a,
                "client_b": client_b,
            },
            sessions_dir=tmp_path,
        )

        turns = await runtime.run()

        assert [turn.speaker for turn in turns] == [
            "client_a",
            "client_b",
            "client_a",
        ]

        event_path = (
            tmp_path
            / "session_client_run_single_request_reentry_test"
            / "internal"
            / "events"
            / "session_client_run_single_request_reentry_test.jsonl"
        )
        events = [
            json.loads(line)
            for line in event_path.read_text(encoding="utf-8").splitlines()
        ]
        floor_events = [
            event
            for event in events
            if event.get("event_type", "").startswith("floor_")
        ]

        assert not any(
            event["details"].get("reason") == "counselor_reentry_after_client_run"
            for event in floor_events
        )
        assert floor_events[-1]["speaker_id"] == "client_a"

    asyncio.run(scenario())


def test_runtime_prefers_counselor_fallback_after_two_client_turns(
    tmp_path,
) -> None:
    async def scenario() -> None:
        counselor = FakeAgent(
            "counselor",
            "カウンセラー生成:{input_transcript}",
            timing_decisions=[
                {
                    "agent_id": "counselor",
                    "action": "WAIT",
                    "target": "client_a",
                    "preferred_timing": "natural_pause",
                    "urgency": 0.2,
                },
                {
                    "agent_id": "counselor",
                    "action": "WAIT",
                    "target": "client_b",
                    "preferred_timing": "natural_pause",
                    "urgency": 0.2,
                },
            ],
        )
        client_a = FakeAgent(
            "client_a",
            "A生成:{input_transcript}",
            timing_decisions=[
                {
                    "agent_id": "client_a",
                    "action": "WAIT",
                    "target": "client_b",
                    "preferred_timing": "natural_pause",
                    "urgency": 0.28,
                }
            ],
        )
        client_b = FakeAgent(
            "client_b",
            "B生成:{input_transcript}",
            timing_decisions=[
                {
                    "agent_id": "client_b",
                    "action": "REQUEST_MAIN_FLOOR",
                    "target": "client_a",
                    "preferred_timing": "natural_pause",
                    "urgency": 0.62,
                }
            ],
        )
        runtime = ConversationRuntime(
            config=RuntimeConfig(
                session_id="session_client_run_fallback_reentry_test",
                max_turns=2,
                speaker_selection_policy="turn_boundary_timing",
                turn_taking_max_reconsider_rounds=0,
                participants={
                    "counselor": ParticipantConfig(
                        speaker_id="counselor",
                        role="counselor",
                        display_name="カウンセラー",
                    ),
                    "client_a": ParticipantConfig(
                        speaker_id="client_a",
                        role="client",
                        display_name="妻",
                        initial_transcript="Aの初回発話です。",
                    ),
                    "client_b": ParticipantConfig(
                        speaker_id="client_b",
                        role="client",
                        display_name="夫",
                    ),
                },
                fixed_speaker_sequence=(
                    "counselor",
                    "client_b",
                    "client_a",
                    "counselor",
                ),
            ),
            agents={
                "counselor": counselor,
                "client_a": client_a,
                "client_b": client_b,
            },
            sessions_dir=tmp_path,
        )

        turns = await runtime.run()

        assert [turn.speaker for turn in turns] == [
            "client_a",
            "client_b",
            "counselor",
        ]

        event_path = (
            tmp_path
            / "session_client_run_fallback_reentry_test"
            / "internal"
            / "events"
            / "session_client_run_fallback_reentry_test.jsonl"
        )
        events = [
            json.loads(line)
            for line in event_path.read_text(encoding="utf-8").splitlines()
        ]
        floor_event = next(
            event
            for event in events
            if event["details"].get("reason") == "counselor_reentry_after_client_run"
        )

        assert floor_event["speaker_id"] == "counselor"
        assert floor_event["details"]["conflict_agent_ids"] == []

    asyncio.run(scenario())


def test_runtime_still_allows_extended_client_run_for_strong_immediate_reply(
    tmp_path,
) -> None:
    async def scenario() -> None:
        counselor = FakeAgent(
            "counselor",
            "カウンセラー生成:{input_transcript}",
            timing_decisions=[
                {
                    "agent_id": "counselor",
                    "action": "REQUEST_MAIN_FLOOR",
                    "target": "client_a",
                    "preferred_timing": "natural_pause",
                    "urgency": 0.4,
                },
                {
                    "agent_id": "counselor",
                    "action": "REQUEST_MAIN_FLOOR",
                    "target": "client_b",
                    "preferred_timing": "natural_pause",
                    "urgency": 0.4,
                },
                {
                    "agent_id": "counselor",
                    "action": "REQUEST_MAIN_FLOOR",
                    "target": "client_a",
                    "preferred_timing": "natural_pause",
                    "urgency": 0.4,
                },
                {
                    "agent_id": "counselor",
                    "action": "REQUEST_MAIN_FLOOR",
                    "target": "client_b",
                    "preferred_timing": "natural_pause",
                    "urgency": 0.4,
                },
                {
                    "agent_id": "counselor",
                    "action": "REQUEST_MAIN_FLOOR",
                    "target": "client_a",
                    "preferred_timing": "natural_pause",
                    "urgency": 0.4,
                },
            ],
        )
        client_a = FakeAgent(
            "client_a",
            "A生成:{input_transcript}",
            timing_decisions=[
                {
                    "agent_id": "client_a",
                    "action": "REQUEST_MAIN_FLOOR",
                    "target": "client_b",
                    "preferred_timing": "immediate",
                    "urgency": 0.9,
                },
                {
                    "agent_id": "client_a",
                    "action": "REQUEST_MAIN_FLOOR",
                    "target": "client_b",
                    "preferred_timing": "immediate",
                    "urgency": 0.9,
                },
            ],
        )
        client_b = FakeAgent(
            "client_b",
            "B生成:{input_transcript}",
            timing_decisions=[
                {
                    "agent_id": "client_b",
                    "action": "REQUEST_MAIN_FLOOR",
                    "target": "client_a",
                    "preferred_timing": "immediate",
                    "urgency": 0.75,
                },
                {
                    "agent_id": "client_b",
                    "action": "REQUEST_MAIN_FLOOR",
                    "target": "client_a",
                    "preferred_timing": "immediate",
                    "urgency": 0.95,
                },
                {
                    "agent_id": "client_b",
                    "action": "REQUEST_MAIN_FLOOR",
                    "target": "client_a",
                    "preferred_timing": "immediate",
                    "urgency": 0.95,
                },
            ],
        )
        runtime = ConversationRuntime(
            config=RuntimeConfig(
                session_id="session_client_reply_strong_extension_test",
                max_turns=3,
                speaker_selection_policy="turn_boundary_timing",
                participants={
                    "counselor": ParticipantConfig(
                        speaker_id="counselor",
                        role="counselor",
                        display_name="カウンセラー",
                    ),
                    "client_a": ParticipantConfig(
                        speaker_id="client_a",
                        role="client",
                        display_name="妻",
                        initial_transcript="Aの初回発話です。",
                    ),
                    "client_b": ParticipantConfig(
                        speaker_id="client_b",
                        role="client",
                        display_name="夫",
                    ),
                },
                fixed_speaker_sequence=(
                    "counselor",
                    "client_b",
                    "client_a",
                    "counselor",
                ),
            ),
            agents={
                "counselor": counselor,
                "client_a": client_a,
                "client_b": client_b,
            },
            sessions_dir=tmp_path,
        )

        turns = await runtime.run()

        assert [turn.speaker for turn in turns] == [
            "client_a",
            "client_b",
            "client_a",
            "client_b",
        ]

    asyncio.run(scenario())


def test_fake_runtime_uses_generated_text_as_next_agent_input(tmp_path) -> None:
    async def scenario() -> None:
        counselor = FakeAgent("counselor", "カウンセラー生成:{input_transcript}")
        client = FakeAgent("client", "クライアント生成:{input_transcript}")
        runtime = ConversationRuntime(
            config=RuntimeConfig(
                session_id="session_runtime_test",
                max_turns=2,
                initial_client_transcript="初回相談です。",
            ),
            agents={"counselor": counselor, "client": client},
            sessions_dir=tmp_path,
        )

        turns = await runtime.run()

        assert runtime.status.phase is RuntimePhase.COMPLETED
        assert [turn.turn_id for turn in turns] == [0, 1, 2]
        assert [turn.speaker for turn in turns] == ["client", "counselor", "client"]
        assert turns[0].generated_text == "クライアント生成:"
        assert turns[0].stt_final_transcript == turns[0].generated_text
        assert counselor.received_inputs == [turns[0].generated_text]
        assert len(client.received_inputs) == 2
        assert "初回相談内容" in client.received_inputs[0]
        assert "初回相談です。" in client.received_inputs[0]
        assert "発話本文ではありません" in client.received_inputs[0]
        assert "原文と同じ一文をそのまま出力せず" in client.received_inputs[0]
        assert client.received_inputs[1] == turns[1].generated_text
        assert turns[2].input_transcript == turns[1].generated_text

        event_path = (
            tmp_path
            / "session_runtime_test"
            / "internal"
            / "events"
            / "session_runtime_test.jsonl"
        )
        event_types = [
            json.loads(line)["event_type"]
            for line in event_path.read_text(encoding="utf-8").splitlines()
        ]
        assert "scripted_audio_turn_started" not in event_types
        assert "llm_request_started" in event_types
        assert "tts_stream_done" in event_types
        assert "generated_final" in event_types
        assert "stt_final" not in event_types

    asyncio.run(scenario())


@pytest.mark.parametrize("checkpoint,first_turn", [(-1, 0), (3, 4), (13, 7)])
def test_director_and_speech_keep_same_unsummarized_history(
    tmp_path, checkpoint, first_turn
) -> None:
    async def scenario():
        store = RuntimePromptContextStore()
        director = RecordingPromptDirector()
        runtime = ConversationRuntime(
            config=RuntimeConfig(session_id="memory-continuity"),
            agents={
                "counselor": FakeAgent("counselor", "応答"),
                "client": FakeAgent("client", "相談"),
            },
            prompt_context_store=store,
            prompt_director=director,
            sessions_dir=tmp_path,
        )
        for index in range(15):
            turn = TurnRuntimeState(
                session_id="memory-continuity",
                turn_id=index,
                speaker="client",
                input_transcript="",
            )
            turn.generated_text = f"発話{index}"
            runtime.turns.append(turn)
        runtime._session_summary_state = SessionSummaryState(
            text="完了した要約" if checkpoint >= 0 else "",
            last_summarized_turn_id=checkpoint,
        )
        runtime._set_generation_prompt_context_for_turn(
            turn_id=15, current_objective="継続"
        )
        await runtime.logger.start()
        try:
            await runtime._add_prompt_director_instruction(
                turn_id=15,
                speaker="counselor",
                input_transcript="発話14",
                current_objective="継続",
                response_target=None,
                existing_instruction=None,
            )
            history = director.requests[0].public_history
            assert [message.text for message in history] == [
                f"発話{index}" for index in range(first_turn, 15)
            ]
            assert (
                store(
                    speaker="counselor",
                    turn_id=15,
                    input_transcript="発話14",
                    purpose="generation",
                ).public_history
                == history
            )
            # An in-flight input must not remove an older, unsummarized message.
            extended = runtime._prompt_director_public_history(
                input_transcript="最新の入力"
            )
            assert extended[:-1] == history
            assert extended[-1].text == "最新の入力"
        finally:
            await runtime._cancel_prompt_director_tasks()
            await runtime.logger.close()

    asyncio.run(scenario())


def test_runtime_directs_only_counselor_and_lets_clients_generate_freely(
    tmp_path,
) -> None:
    async def scenario() -> None:
        counselor = FakeAgent("counselor", "カウンセラー生成:{input_transcript}")
        client = FakeAgent("client", "クライアント生成:{input_transcript}")
        director = RecordingPromptDirector()
        runtime = ConversationRuntime(
            config=RuntimeConfig(
                session_id="session_prompt_director_test",
                max_turns=2,
                initial_client_transcript="初回相談です。",
                speaker_selection_policy="fixed_round_robin",
                participants={
                    "counselor": ParticipantConfig(
                        speaker_id="counselor",
                        role="counselor",
                        display_name="カウンセラー",
                        prompt_source="どのような内容でもよい任意の応答プロンプト。",
                    ),
                    "client": ParticipantConfig(
                        speaker_id="client",
                        role="client",
                        display_name="クライアント",
                        prompt_source="相談者本人として短く話す。",
                        public_profile_source="学校生活について相談している本人。",
                    ),
                },
                shared_case="学校生活をめぐるカウンセリング。",
            ),
            agents={"counselor": counselor, "client": client},
            prompt_director=director,
            sessions_dir=tmp_path,
        )

        turns = await runtime.run()

        assert [request.speaker_id for request in director.requests] == [
            "counselor",
        ]
        counselor_request = next(
            request
            for request in director.requests
            if request.speaker_id == "counselor"
        )
        assert counselor_request.configured_prompt == (
            "どのような内容でもよい任意の応答プロンプト。"
        )
        assert counselor_request.fixed_system_prompt == COUNSELOR_SYSTEM_PROMPT
        assert counselor_request.client_peer_ids == ()
        assert counselor_request.public_history[-1].text == turns[0].generated_text
        assert "＜原文の条件と今回の適用判断＞" not in counselor.received_inputs[0]
        assert "＜今回発話する本文＞" in counselor.received_inputs[0]
        assert "＜この応答の目的＞\ncounselorのturn 1の応答目的。" not in (
            counselor.received_inputs[0]
        )
        assert "相談の入口に立てたことが、まず大切なのですね。" in (
            counselor.received_inputs[0]
        )
        assert all(
            "＜原文の条件と今回の適用判断＞" not in received
            and "＜今回発話する本文＞" not in received
            and "require_repeat_verbatim" not in received
            for received in client.received_inputs
        )
        assert len(client.received_inputs) == 2

        event_path = (
            tmp_path
            / "session_prompt_director_test"
            / "internal"
            / "events"
            / "session_prompt_director_test.jsonl"
        )
        event_types = [
            json.loads(line)["event_type"]
            for line in event_path.read_text(encoding="utf-8").splitlines()
        ]
        assert "prompt_director_request_started" in event_types
        assert "prompt_director_completed" in event_types

    asyncio.run(scenario())


def test_client_director_bypass_includes_prefetch_and_preserves_turn_instruction(
    tmp_path,
) -> None:
    async def scenario():
        director = RecordingPromptDirector()
        runtime = ConversationRuntime(prompt_director=director, sessions_dir=tmp_path)
        instruction = "自分に残された質問へ応答してください。"
        kwargs = dict(
            turn_id=3,
            speaker="client",
            input_transcript="今の確認でよいですか？",
            current_objective="クロージング",
            response_target="counselor",
            existing_instruction=instruction,
        )
        try:
            runtime._prefetch_prompt_director_instruction(**kwargs)
            assert not runtime._prompt_director_prefetch_tasks
            assert (
                await runtime._add_prompt_director_instruction(**kwargs) == instruction
            )
            assert not director.requests
            assert not runtime._prompt_director_tasks
            assert not runtime._selected_prompt_director_texts
        finally:
            await runtime._cancel_prompt_director_tasks()

    asyncio.run(scenario())


class ExcerptRepairLLM:
    def __init__(
        self,
        failed_reviews: int,
        successful_reviews_first: int = 0,
        *,
        semantic: bool = False,
    ) -> None:
        self.failed_reviews = failed_reviews
        self.successful_reviews_first = successful_reviews_first
        self.semantic = semantic
        self.calls = []

    async def stream_text(self, **kwargs):
        self.calls.append(kwargs)
        review = kwargs["system_prompt"] == PROMPT_DIRECTOR_REVIEW_SYSTEM_PROMPT
        invalid = (
            review and self.successful_reviews_first == 0 and self.failed_reviews > 0
        )
        if review and self.successful_reviews_first:
            self.successful_reviews_first -= 1
        if invalid:
            self.failed_reviews -= 1
        yield json.dumps(
            {
                "context_basis": "共有履歴のみ参照。",
                "response_issues": (
                    ["他者の説明を不要に復唱している。"]
                    if invalid and self.semantic
                    else []
                ),
                "response_intent": "相談者の話を受け止める。",
                "response_example": "確認済みの発話です。",
                "instruction_checks": [
                    {
                        "source": "fixed_role_constraints",
                        "start_line": 1,
                        "end_line": 1,
                        "applicability": "applies",
                        "evidence": "対象話者として話す。",
                        "force": "required",
                        "priority": "指定なし",
                        "response_excerpt": (
                            "本文にない失敗引用"
                            if invalid and not self.semantic
                            else "確認済みの発話です。"
                        ),
                        "response_assessment": "対象話者としての発話。",
                    }
                ],
            },
            ensure_ascii=False,
        )


@pytest.mark.parametrize("semantic", [False, True])
def test_runtime_recovers_review_issues_and_saves_attempts_internally(
    tmp_path, semantic
):
    async def scenario():
        llm = ExcerptRepairLLM(failed_reviews=1, semantic=semantic)
        runtime = ConversationRuntime(
            config=RuntimeConfig(
                max_turns=1, speaker_selection_policy="fixed_round_robin"
            ),
            prompt_director=PromptDirector(llm),
            sessions_dir=tmp_path,
        )
        await asyncio.wait_for(runtime.run(), timeout=3)
        assert runtime.status.phase is RuntimePhase.COMPLETED
        assert len(runtime.turns) == 2
        path = runtime.logger.paths.prompt_director_attempts_jsonl
        attempts = [json.loads(line) for line in path.read_text().splitlines()]
        failed = [
            r
            for r in attempts
            if r["details"]["validation_error"] or r["details"]["response_issues"]
        ]
        assert len(failed) == 1
        issue = "他者の説明を不要に復唱している。" if semantic else "本文にない失敗引用"
        assert issue in failed[0]["details"]["response_json"]
        assert failed[0]["details"]["request"]["speaker_id"] == "counselor"
        events = [
            json.loads(line)
            for line in runtime.logger.paths.events_jsonl.read_text().splitlines()
        ]
        types = [r["event_type"] for r in events]
        assert "prompt_director_regeneration_started" in types
        assert "runtime_error" not in types
        assert issue not in runtime.logger.paths.transcripts_jsonl.read_text()
        assert issue not in runtime.logger.paths.events_jsonl.read_text()

    asyncio.run(scenario())


@pytest.mark.parametrize("stop_instead_of_resume", [False, True])
@pytest.mark.parametrize("completed_before_failure", [0, 1])
@pytest.mark.parametrize("semantic", [False, True])
def test_director_exhaustion_keeps_session_resumable_or_stoppable(
    tmp_path, stop_instead_of_resume, completed_before_failure, semantic
):
    from counseling_voice_demo.runtime.control_api import RuntimeControlService

    async def scenario():
        llm = ExcerptRepairLLM(
            failed_reviews=3,
            successful_reviews_first=completed_before_failure,
            semantic=semantic,
        )
        runtime = ConversationRuntime(
            config=RuntimeConfig(
                max_turns=3, speaker_selection_policy="fixed_round_robin"
            ),
            prompt_director=PromptDirector(llm),
            sessions_dir=tmp_path,
        )
        service = RuntimeControlService(lambda: runtime)
        await service.start()
        try:
            async with asyncio.timeout(3):
                while runtime.status.phase is not RuntimePhase.PAUSED:
                    await asyncio.sleep(0.001)
            status = await service.status()
            assert status.phase is RuntimePhase.PAUSED
            assert status.pause_reason == "prompt_director_validation"
            assert status.error_message is None
            completed_turns = 1 + 2 * completed_before_failure
            assert status.current_turn_id == completed_turns
            expected_calls = (6 if semantic else 4) + completed_before_failure * 2
            assert len(llm.calls) == expected_calls
            assert len(runtime.turns) == completed_turns
            previous_turns = list(runtime.turns)
            session_id = status.session_id
            if stop_instead_of_resume:
                assert (await service.stop()).phase is RuntimePhase.STOPPED
                async with asyncio.timeout(3):
                    while runtime._phase is not RuntimePhase.STOPPED:
                        await asyncio.sleep(0.001)
                assert len(llm.calls) == expected_calls
                assert runtime.status.pause_reason is None
            else:
                resumed = await service.resume()
                assert resumed.phase is RuntimePhase.RUNNING
                assert resumed.session_id == session_id
                async with asyncio.timeout(3):
                    while (await service.status()).phase is not RuntimePhase.COMPLETED:
                        await asyncio.sleep(0.001)
                assert [turn.turn_id for turn in runtime.turns] == [0, 1, 2, 3]
                assert runtime.turns[:completed_turns] == previous_turns
                assert (
                    "runtime_error" not in runtime.logger.paths.events_jsonl.read_text()
                )
        finally:
            await service.stop()
            await runtime._cancel_prompt_director_tasks()

    asyncio.run(scenario())


def test_failed_speculative_director_does_not_pause_current_session(tmp_path):
    async def scenario():
        runtime = ConversationRuntime(
            prompt_director=PromptDirector(ExcerptRepairLLM(failed_reviews=3)),
            sessions_dir=tmp_path,
        )
        runtime._phase = RuntimePhase.RUNNING
        await runtime.logger.start()
        try:
            runtime._prefetch_prompt_director_instruction(
                turn_id=1,
                speaker="counselor",
                input_transcript="相談内容",
                current_objective="継続",
                response_target="client",
                existing_instruction=None,
            )
            await asyncio.wait_for(
                asyncio.gather(
                    *runtime._prompt_director_prefetch_tasks, return_exceptions=True
                ),
                timeout=3,
            )
            assert runtime.status.phase is RuntimePhase.RUNNING
            assert runtime._generation_resume_event.is_set()
            assert len(runtime.prompt_director.llm.calls) == 4
            actual = asyncio.create_task(
                runtime._add_prompt_director_instruction(
                    turn_id=1,
                    speaker="counselor",
                    input_transcript="相談内容",
                    current_objective="継続",
                    response_target="client",
                    existing_instruction=None,
                )
            )
            try:
                async with asyncio.timeout(1):
                    while runtime.status.phase is not RuntimePhase.PAUSED:
                        await asyncio.sleep(0.001)
                assert len(runtime.prompt_director.llm.calls) == 4
                runtime.resume_generation()
                instruction = await asyncio.wait_for(actual, timeout=1)
                assert "確認済みの発話です。" in instruction
                assert "本文にない失敗引用" not in instruction
            finally:
                actual.cancel()
                await asyncio.gather(actual, return_exceptions=True)
        finally:
            await runtime._cancel_prompt_director_tasks()
            await runtime.logger.close()

    asyncio.run(scenario())


def test_runtime_skips_prompt_director_for_both_clients(
    tmp_path,
) -> None:
    async def scenario() -> None:
        director = RecordingPromptDirector()
        runtime = ConversationRuntime(
            config=RuntimeConfig(
                session_id="session_two_client_prompt_director_test",
                max_turns=3,
                initial_client_transcript="娘のことで相談があります。",
                participant_mode="two_clients",
                speaker_selection_policy="fixed_round_robin",
                fixed_speaker_sequence=("counselor", "client_a", "client_b"),
                participants={
                    "counselor": ParticipantConfig(
                        speaker_id="counselor",
                        role="counselor",
                        display_name="カウンセラー",
                        prompt_source="家族面接を進める。",
                    ),
                    "client_a": ParticipantConfig(
                        speaker_id="client_a",
                        role="client",
                        display_name="妻",
                        prompt_source="夫婦の一人として自然に話す。",
                        public_profile_source="妻固有のプロフィール。",
                    ),
                    "client_b": ParticipantConfig(
                        speaker_id="client_b",
                        role="client",
                        display_name="夫",
                        prompt_source="夫婦の一人として自然に話す。",
                        public_profile_source="夫固有のプロフィール。",
                        private_profile_source="夫だけの個別設定。",
                    ),
                },
                shared_case="夫婦共通の相談事例。",
            ),
            agents={
                "counselor": FakeAgent("counselor", "カウンセラー応答"),
                "client_a": FakeAgent("client_a", "妻の応答"),
                "client_b": FakeAgent("client_b", "夫の応答"),
            },
            prompt_director=director,
            sessions_dir=tmp_path,
        )

        turns = await runtime.run()

        assert {turn.speaker for turn in turns} == {"counselor", "client_a", "client_b"}
        assert director.requests
        assert {request.speaker_id for request in director.requests} == {"counselor"}
        for speaker in ("client_a", "client_b"):
            assert runtime.controller.agents[speaker].received_inputs
            assert all(
                "require_repeat_verbatim" not in text
                for text in runtime.controller.agents[speaker].received_inputs
            )

    asyncio.run(scenario())


def test_runtime_logs_resolved_audio_settings_at_session_start(tmp_path) -> None:
    async def scenario() -> None:
        runtime = ConversationRuntime(
            config=RuntimeConfig(
                session_id="session_audio_settings_log_test",
                max_turns=1,
                initial_client_transcript="初回相談です。",
                sample_rate=24000,
                sample_width_bits=16,
                channels=1,
                speaker_audio_gains={"counselor": 0.6, "client": 0.7},
                participants={
                    "counselor": ParticipantConfig(
                        speaker_id="counselor",
                        role="counselor",
                        display_name="佐伯",
                        voice="shimmer",
                        realtime_output_speed=1.0,
                    ),
                    "client": ParticipantConfig(
                        speaker_id="client",
                        role="client",
                        display_name="高橋",
                        voice="cedar",
                        realtime_output_speed=0.9,
                    ),
                },
            ),
            agents={
                "counselor": FakeAgent("counselor", "相談内容を確認します。"),
                "client": FakeAgent("client", "よろしくお願いします。"),
            },
            sessions_dir=tmp_path,
        )

        await runtime.run()

        event_path = (
            tmp_path
            / "session_audio_settings_log_test"
            / "internal"
            / "events"
            / "session_audio_settings_log_test.jsonl"
        )
        events = [
            json.loads(line)
            for line in event_path.read_text(encoding="utf-8").splitlines()
        ]
        settings_event = next(
            event
            for event in events
            if event["event_type"] == "runtime_audio_settings_resolved"
        )

        assert events[0]["event_type"] == "runtime_audio_settings_resolved"
        assert settings_event["turn_id"] is None
        assert settings_event["speaker"] is None
        assert settings_event["details"] == {
            "interaction_mode": "ai_counselor_ai_client",
            "participant_mode": "one_client",
            "sample_rate": 24000,
            "sample_width_bits": 16,
            "channels": 1,
            "speakers": [
                {
                    "speaker_id": "counselor",
                    "display_name": "佐伯",
                    "role": "counselor",
                    "actor_kind": "ai",
                    "voice": "shimmer",
                    "realtime_output_speed": 1.0,
                    "audio_gain": 0.6,
                },
                {
                    "speaker_id": "client",
                    "display_name": "高橋",
                    "role": "client",
                    "actor_kind": "ai",
                    "voice": "cedar",
                    "realtime_output_speed": 0.9,
                    "audio_gain": 0.7,
                },
            ],
        }

    asyncio.run(scenario())


def test_runtime_prompt_context_store_feeds_recent_history_to_generation(
    tmp_path,
) -> None:
    async def scenario() -> None:
        config = RuntimeConfig(
            session_id="session_prompt_context_store_test",
            max_turns=2,
            initial_client_transcript="初回相談です。",
            speaker_selection_policy="fixed_round_robin",
            conversation_context_recent_turns=8,
        )
        context_store = RuntimePromptContextStore()
        input_formatter = build_prompt_context_input_formatter(
            participants=config.participants,
            shared_case="",
            runtime_context_provider=context_store,
        )
        counselor_llm = ScriptedTextLLM("カウンセラー応答です。")
        client_llm = ScriptedTextLLM("クライアント応答です。")
        runtime = ConversationRuntime(
            config=config,
            agents={
                "counselor": StreamingAgent(
                    speaker="counselor",
                    llm=counselor_llm,
                    input_formatter=input_formatter,
                ),
                "client": StreamingAgent(
                    speaker="client",
                    llm=client_llm,
                    input_formatter=input_formatter,
                ),
            },
            prompt_context_store=context_store,
            sessions_dir=tmp_path,
        )

        await runtime.run()

        assert "初回相談内容" in client_llm.latest_inputs[0]
        assert "初回相談です。" in client_llm.latest_inputs[0]
        assert (
            "Public history:\nクライアント（client / client）: クライアント応答です。"
            in counselor_llm.latest_inputs[0]
        )
        assert (
            "直前発話者: クライアント（client / client）"
            in counselor_llm.latest_inputs[0]
        )
        assert "Current objective / phase:" in counselor_llm.latest_inputs[0]
        assert (
            "クライアント（client / client）: クライアント応答です。"
            in client_llm.latest_inputs[1]
        )
        assert (
            "カウンセラー（counselor / counselor）: カウンセラー応答です。"
            in client_llm.latest_inputs[1]
        )
        assert (
            "直前発話者: カウンセラー（counselor / counselor）"
            in client_llm.latest_inputs[1]
        )

    asyncio.run(scenario())


def test_runtime_current_objective_uses_generic_prompt_following_language(
    tmp_path,
) -> None:
    runtime = ConversationRuntime(
        config=RuntimeConfig(session_id="session_generic_objective_test"),
        sessions_dir=tmp_path,
    )

    assert runtime._current_objective_for_turn(
        turn_id=1,
        closing_instruction=None,
    ) == ("カウンセリングの継続。直近の発話と共有履歴を踏まえて、" "プロンプトの指示に従って、カウンセリングを継続する。")
    assert (
        runtime._current_objective_for_turn(
            turn_id=2,
            closing_instruction="クロージング指示",
        )
        == "クロージング。プロンプトの指示に従ってクロージングを行う。"
    )


def test_runtime_pause_generation_waits_before_next_generated_turn(tmp_path) -> None:
    async def scenario() -> None:
        counselor = FakeAgent("counselor", "カウンセラー生成:{input_transcript}")
        client = FakeAgent("client", "クライアント生成:{input_transcript}")
        runtime = ConversationRuntime(
            config=RuntimeConfig(
                session_id="session_runtime_pause_generation_test",
                max_turns=1,
                initial_client_transcript="初回相談です。",
            ),
            agents={"counselor": counselor, "client": client},
            sessions_dir=tmp_path,
        )

        runtime.pause_generation()
        task = asyncio.create_task(runtime.run())
        for _ in range(20):
            if runtime.status.completed_turns >= 1:
                break
            await asyncio.sleep(0.01)

        assert runtime.status.completed_turns == 1
        assert counselor.received_inputs == []

        runtime.resume_generation()
        turns = await asyncio.wait_for(task, timeout=2.0)

        assert [turn.turn_id for turn in turns] == [0, 1]
        assert counselor.received_inputs == [turns[0].generated_text]

    asyncio.run(scenario())


def test_fake_runtime_uses_randomly_selected_client_initial_transcript(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(random, "choice", lambda candidates: candidates[0])

    async def scenario() -> None:
        counselor = FakeAgent("counselor", "カウンセラー生成:{input_transcript}")
        client_a = FakeAgent("client_a", "A生成:{input_transcript}")
        client_b = FakeAgent("client_b", "B生成:{input_transcript}")
        runtime = ConversationRuntime(
            config=RuntimeConfig(
                session_id="session_runtime_two_client_initial_test",
                max_turns=4,
                initial_client_transcript="後方互換の初回です。",
                speaker_selection_policy="fixed_round_robin",
                participants={
                    "counselor": ParticipantConfig(
                        speaker_id="counselor",
                        role="counselor",
                        display_name="カウンセラー",
                    ),
                    "client_a": ParticipantConfig(
                        speaker_id="client_a",
                        role="client",
                        display_name="クライアントA",
                        initial_transcript="Aの初回発話です。",
                    ),
                    "client_b": ParticipantConfig(
                        speaker_id="client_b",
                        role="client",
                        display_name="クライアントB",
                        initial_transcript="Bの初回発話です。",
                    ),
                },
                fixed_speaker_sequence=(
                    "counselor",
                    "client_a",
                    "counselor",
                    "client_b",
                ),
            ),
            agents={
                "counselor": counselor,
                "client_a": client_a,
                "client_b": client_b,
            },
            sessions_dir=tmp_path,
        )

        turns = await runtime.run()

        assert [turn.speaker for turn in turns] == [
            "client_a",
            "counselor",
            "client_a",
            "counselor",
            "client_b",
        ]
        assert turns[0].generated_text == "A生成:"
        assert turns[0].stt_final_transcript == turns[0].generated_text
        assert counselor.received_inputs[0] == turns[0].generated_text
        assert len(client_a.received_inputs) == 2
        assert "初回相談内容" in client_a.received_inputs[0]
        assert "Aの初回発話です。" in client_a.received_inputs[0]
        assert "原文と同じ一文をそのまま出力せず" in (client_a.received_inputs[0])
        assert "短く挨拶してから初回相談内容に触れてください" in (
            client_a.received_inputs[0]
        )
        assert turns[1].generated_text in client_a.received_inputs[1]
        assert "今回の主な宛先は固定されていません。" in client_a.received_inputs[1]
        assert "自然な宛先と話し方" in client_a.received_inputs[1]
        assert "特定の参加者を常に優先せず" in client_a.received_inputs[1]
        assert len(client_b.received_inputs) == 1
        assert turns[3].generated_text in client_b.received_inputs[0]
        assert "今回の主な宛先は固定されていません。" in client_b.received_inputs[0]
        assert "自然な宛先と話し方" in client_b.received_inputs[0]
        assert "特定の参加者を常に優先せず" in client_b.received_inputs[0]

    asyncio.run(scenario())


def test_initial_client_turn_randomly_selects_when_both_clients_have_topics(
    monkeypatch,
) -> None:
    config = RuntimeConfig(
        participants={
            "counselor": ParticipantConfig(
                speaker_id="counselor",
                role="counselor",
                display_name="カウンセラー",
            ),
            "client_a": ParticipantConfig(
                speaker_id="client_a",
                role="client",
                display_name="クライアントA",
                initial_transcript="Aの初回発話です。",
            ),
            "client_b": ParticipantConfig(
                speaker_id="client_b",
                role="client",
                display_name="クライアントB",
                initial_transcript="Bの初回発話です。",
            ),
        },
        fixed_speaker_sequence=("counselor", "client_a", "client_b"),
    )
    monkeypatch.setattr(random, "choice", lambda candidates: candidates[-1])

    assert _initial_client_turn(config) == ("client_b", "Bの初回発話です。")


def test_runtime_distributed_timing_grants_floor_to_requesting_agent(
    tmp_path,
) -> None:
    async def scenario() -> None:
        counselor = FakeAgent(
            "counselor",
            "カウンセラー生成:{input_transcript}",
            timing_decisions=[{"agent_id": "counselor", "action": "WAIT"}],
        )
        client_a = FakeAgent("client_a", "A生成:{input_transcript}")
        client_b = FakeAgent(
            "client_b",
            "B生成:{input_transcript}",
            timing_decisions=[
                {
                    "agent_id": "client_b",
                    "action": "REQUEST_MAIN_FLOOR",
                    "speech_ms_last_window": 0,
                    "request_timestamp_ms": 1,
                }
            ],
        )
        runtime = ConversationRuntime(
            config=RuntimeConfig(
                session_id="session_distributed_timing_test",
                max_turns=1,
                initial_client_transcript="後方互換の初回です。",
                speaker_selection_policy="distributed_timing",
                participants={
                    "counselor": ParticipantConfig(
                        speaker_id="counselor",
                        role="counselor",
                        display_name="カウンセラー",
                    ),
                    "client_a": ParticipantConfig(
                        speaker_id="client_a",
                        role="client",
                        display_name="クライアントA",
                        initial_transcript="Aの初回発話です。",
                    ),
                    "client_b": ParticipantConfig(
                        speaker_id="client_b",
                        role="client",
                        display_name="クライアントB",
                    ),
                },
                fixed_speaker_sequence=(
                    "counselor",
                    "client_a",
                    "counselor",
                    "client_b",
                ),
            ),
            agents={
                "counselor": counselor,
                "client_a": client_a,
                "client_b": client_b,
            },
            sessions_dir=tmp_path,
        )

        turns = await runtime.run()

        assert [turn.speaker for turn in turns] == ["client_a", "client_b"]
        assert counselor.received_inputs == []
        assert "初回相談内容" in client_a.received_inputs[0]
        assert "Aの初回発話です。" in client_a.received_inputs[0]
        assert len(client_b.received_inputs) == 1
        assert turns[0].generated_text in client_b.received_inputs[0]
        assert "今回の主な宛先は固定されていません。" in client_b.received_inputs[0]
        assert "自然な宛先と話し方" in client_b.received_inputs[0]
        assert counselor.received_timing_inputs == [turns[0].generated_text]
        assert client_b.received_timing_inputs == [turns[0].generated_text]

        event_path = (
            tmp_path
            / "session_distributed_timing_test"
            / "internal"
            / "events"
            / "session_distributed_timing_test.jsonl"
        )
        events = [
            json.loads(line)
            for line in event_path.read_text(encoding="utf-8").splitlines()
        ]
        floor_events = [
            event for event in events if event["event_type"] == "floor_granted"
        ]
        assert floor_events[0]["speaker_id"] == "client_b"
        assert floor_events[0]["details"]["speaker_selection_policy"] == (
            "distributed_timing"
        )
        assert floor_events[0]["details"]["granted_agent_id"] == "client_b"
        event_types = [event["event_type"] for event in events]
        discarded = [
            event
            for event in events
            if event["event_type"] == "provisional_llm_discarded"
        ]
        assert "provisional_llm_request_started" not in event_types
        assert discarded == []

    asyncio.run(scenario())


def test_runtime_turn_boundary_timing_uses_agent_requests_without_overlap(
    tmp_path,
) -> None:
    async def scenario() -> None:
        counselor = FakeAgent(
            "counselor",
            "カウンセラー生成:{input_transcript}",
            timing_decisions=[
                {
                    "agent_id": "counselor",
                    "action": "REQUEST_MAIN_FLOOR",
                    "speech_ms_last_window": 200,
                    "request_timestamp_ms": 2,
                }
            ],
        )
        client_a = FakeAgent(
            "client_a",
            "A生成:{input_transcript}",
            timing_decisions=[{"agent_id": "client_a", "action": "WAIT"}],
        )
        client_b = FakeAgent(
            "client_b",
            "B生成:{input_transcript}",
            timing_decisions=[
                {
                    "agent_id": "client_b",
                    "action": "REQUEST_MAIN_FLOOR",
                    "speech_ms_last_window": 0,
                    "request_timestamp_ms": 1,
                }
            ],
        )
        runtime = ConversationRuntime(
            config=RuntimeConfig(
                session_id="session_turn_boundary_timing_test",
                max_turns=1,
                speaker_selection_policy="turn_boundary_timing",
                turn_taking_max_reconsider_rounds=0,
                participants={
                    "counselor": ParticipantConfig(
                        speaker_id="counselor",
                        role="counselor",
                        display_name="カウンセラー",
                    ),
                    "client_a": ParticipantConfig(
                        speaker_id="client_a",
                        role="client",
                        display_name="クライアントA",
                        initial_transcript="Aの初回発話です。",
                    ),
                    "client_b": ParticipantConfig(
                        speaker_id="client_b",
                        role="client",
                        display_name="クライアントB",
                    ),
                },
                fixed_speaker_sequence=(
                    "counselor",
                    "client_a",
                    "counselor",
                    "client_b",
                ),
            ),
            agents={
                "counselor": counselor,
                "client_a": client_a,
                "client_b": client_b,
            },
            sessions_dir=tmp_path,
        )

        turns = await runtime.run()

        assert [turn.speaker for turn in turns] == ["client_a", "client_b"]
        assert client_b.received_timing_inputs == [turns[0].generated_text]

        event_path = (
            tmp_path
            / "session_turn_boundary_timing_test"
            / "internal"
            / "events"
            / "session_turn_boundary_timing_test.jsonl"
        )
        events = [
            json.loads(line)
            for line in event_path.read_text(encoding="utf-8").splitlines()
        ]
        assert not any(event["event_type"] == "overlap_detected" for event in events)
        floor_events = [
            event
            for event in events
            if event["event_type"] == "floor_conflict_resolved"
        ]
        assert floor_events[0]["speaker_id"] == "client_b"
        assert floor_events[0]["details"]["speaker_selection_policy"] == (
            "turn_boundary_timing"
        )
        assert floor_events[0]["details"]["granted_agent_id"] == "client_b"
        assert set(floor_events[0]["details"]["conflict_agent_ids"]) == {
            "counselor",
            "client_b",
        }

    asyncio.run(scenario())


def test_runtime_turn_boundary_timing_reconsiders_conflicting_requests(
    tmp_path,
) -> None:
    async def scenario() -> None:
        counselor = FakeAgent(
            "counselor",
            "カウンセラー生成:{input_transcript}",
            timing_decisions=[
                {"agent_id": "counselor", "action": "REQUEST_MAIN_FLOOR"},
                {"agent_id": "counselor", "action": "WAIT"},
            ],
        )
        client_a = FakeAgent("client_a", "A生成:{input_transcript}")
        client_b = FakeAgent(
            "client_b",
            "B生成:{input_transcript}",
            timing_decisions=[
                {
                    "agent_id": "client_b",
                    "action": "REQUEST_MAIN_FLOOR",
                    "target": "counselor",
                    "urgency": 0.2,
                },
                {"agent_id": "client_b", "action": "REQUEST_MAIN_FLOOR"},
            ],
        )
        runtime = ConversationRuntime(
            config=RuntimeConfig(
                session_id="session_turn_boundary_conflict_reconsider_test",
                max_turns=1,
                speaker_selection_policy="turn_boundary_timing",
                participants={
                    "counselor": ParticipantConfig(
                        speaker_id="counselor",
                        role="counselor",
                        display_name="カウンセラー",
                    ),
                    "client_a": ParticipantConfig(
                        speaker_id="client_a",
                        role="client",
                        display_name="クライアントA",
                        initial_transcript="Aの初回発話です。",
                    ),
                    "client_b": ParticipantConfig(
                        speaker_id="client_b",
                        role="client",
                        display_name="クライアントB",
                    ),
                },
                fixed_speaker_sequence=(
                    "counselor",
                    "client_a",
                    "counselor",
                    "client_b",
                ),
            ),
            agents={
                "counselor": counselor,
                "client_a": client_a,
                "client_b": client_b,
            },
            sessions_dir=tmp_path,
        )

        turns = await runtime.run()

        assert [turn.speaker for turn in turns] == ["client_a", "client_b"]
        assert len(counselor.received_timing_inputs) == 2
        assert len(client_b.received_timing_inputs) == 2
        assert counselor.received_timing_inputs[0] == turns[0].generated_text
        assert client_b.received_timing_inputs[0] == turns[0].generated_text
        assert "ターンテイク競合" in counselor.received_timing_inputs[1]
        assert "ターンテイク競合" in client_b.received_timing_inputs[1]
        assert "同じ発話者順序の短いパターン" in counselor.received_timing_inputs[1]
        assert "同じ二者往復や同じ三者サイクル" in client_b.received_timing_inputs[1]
        assert client_a.received_timing_inputs == []

        event_path = (
            tmp_path
            / "session_turn_boundary_conflict_reconsider_test"
            / "internal"
            / "events"
            / "session_turn_boundary_conflict_reconsider_test.jsonl"
        )
        events = [
            json.loads(line)
            for line in event_path.read_text(encoding="utf-8").splitlines()
        ]
        conflict_event = next(
            event
            for event in events
            if event["event_type"] == "floor_negotiation_conflict"
        )
        floor_event = next(
            event for event in events if event["event_type"] == "floor_granted"
        )

        assert conflict_event["speaker_id"] is None
        assert conflict_event["details"]["conflict_agent_ids"] == [
            "counselor",
            "client_b",
        ]
        assert conflict_event["details"]["round_index"] == 1
        assert floor_event["speaker_id"] == "client_b"
        assert floor_event["details"]["granted_agent_id"] == "client_b"
        assert not any(event["event_type"] == "overlap_detected" for event in events)

    asyncio.run(scenario())


def test_runtime_turn_boundary_timing_collects_next_turn_signal_while_streaming(
    tmp_path,
) -> None:
    async def scenario() -> None:
        counselor = FakeAgent(
            "counselor",
            "カウンセラー生成:{input_transcript}。もう少し聞かせてください。",
        )
        client = FakeAgent("client", "クライアント生成:{input_transcript}")
        runtime = ConversationRuntime(
            config=RuntimeConfig(
                session_id="session_turn_taking_streaming_signal_test",
                max_turns=2,
                initial_client_transcript="初回相談です。",
                speaker_selection_policy="turn_boundary_timing",
                turn_taking_signal_min_chars=1,
            ),
            agents={"counselor": counselor, "client": client},
            sessions_dir=tmp_path,
        )

        turns = await runtime.run()

        assert [turn.speaker for turn in turns] == ["client", "counselor", "client"]
        assert counselor.received_timing_inputs == [turns[0].generated_text]
        assert len(client.received_timing_inputs) == 2
        assert client.received_timing_inputs[0] != turns[1].generated_text
        assert client.received_timing_inputs[0] in turns[1].generated_text
        assert client.received_timing_inputs[1] == turns[1].generated_text

        event_path = (
            tmp_path
            / "session_turn_taking_streaming_signal_test"
            / "internal"
            / "events"
            / "session_turn_taking_streaming_signal_test.jsonl"
        )
        events = [
            json.loads(line)
            for line in event_path.read_text(encoding="utf-8").splitlines()
        ]
        signal_event = next(
            event
            for event in events
            if event["event_type"] == "turn_taking_signal_collected"
        )
        floor_events = [
            event for event in events if event["event_type"] == "floor_granted"
        ]

        assert signal_event["turn_id"] == 2
        assert signal_event["details"]["source_turn_id"] == 1
        assert signal_event["details"]["previous_speaker"] == "counselor"
        assert signal_event["details"]["signal_min_chars"] == 1
        assert [item["action"] for item in signal_event["details"]["decisions"]] == [
            "WAIT",
            "REQUEST_MAIN_FLOOR",
        ]
        assert floor_events[-1]["turn_id"] == 2
        assert floor_events[-1]["speaker_id"] == "client"

    asyncio.run(scenario())


def test_runtime_applies_turn_start_delay_from_timing_decision(tmp_path) -> None:
    async def scenario() -> None:
        counselor = FakeAgent(
            "counselor",
            "カウンセラー生成:{input_transcript}",
            timing_decisions=[
                {
                    "agent_id": "counselor",
                    "action": "REQUEST_MAIN_FLOOR",
                    "preferred_timing": "natural_pause",
                    "urgency": 1.0,
                }
            ],
        )
        client = FakeAgent("client", "クライアント生成:{input_transcript}")
        runtime = ConversationRuntime(
            config=RuntimeConfig(
                session_id="session_turn_start_delay_test",
                max_turns=1,
                initial_client_transcript="初回相談です。",
                speaker_selection_policy="turn_boundary_timing",
            ),
            agents={"counselor": counselor, "client": client},
            sessions_dir=tmp_path,
        )

        turns = await runtime.run()

        assert [turn.speaker for turn in turns] == ["client", "counselor"]

        event_path = (
            tmp_path
            / "session_turn_start_delay_test"
            / "internal"
            / "events"
            / "session_turn_start_delay_test.jsonl"
        )
        events = [
            json.loads(line)
            for line in event_path.read_text(encoding="utf-8").splitlines()
        ]
        delay_event = next(
            event
            for event in events
            if event["event_type"] == "turn_start_delay_applied"
        )
        llm_started_event = next(
            event
            for event in events
            if event["event_type"] == "llm_request_started" and event["turn_id"] == 1
        )

        assert delay_event["turn_id"] == 1
        assert delay_event["speaker_id"] == "counselor"
        assert 100 <= delay_event["details"]["delay_ms"] <= 200
        assert delay_event["details"]["preferred_timing"] == "natural_pause"
        assert delay_event["details"]["effective_preferred_timing"] == "immediate"
        assert delay_event["details"]["urgency"] == 1.0
        assert delay_event["monotonic_time"] <= llm_started_event["monotonic_time"]

    asyncio.run(scenario())


def test_runtime_applies_turn_start_delay_from_fallback_wait_timing(tmp_path) -> None:
    async def scenario() -> None:
        counselor = FakeAgent(
            "counselor",
            "カウンセラー生成:{input_transcript}",
            timing_decisions=[
                {
                    "agent_id": "counselor",
                    "action": "WAIT",
                    "preferred_timing": "immediate",
                    "urgency": 0.2,
                }
            ],
        )
        client = FakeAgent("client", "クライアント生成:{input_transcript}")
        runtime = ConversationRuntime(
            config=RuntimeConfig(
                session_id="session_fallback_wait_timing_delay_test",
                max_turns=1,
                initial_client_transcript="初回相談です。",
                speaker_selection_policy="turn_boundary_timing",
            ),
            agents={"counselor": counselor, "client": client},
            sessions_dir=tmp_path,
        )

        turns = await runtime.run()

        assert [turn.speaker for turn in turns] == ["client", "counselor"]

        event_path = (
            tmp_path
            / "session_fallback_wait_timing_delay_test"
            / "internal"
            / "events"
            / "session_fallback_wait_timing_delay_test.jsonl"
        )
        events = [
            json.loads(line)
            for line in event_path.read_text(encoding="utf-8").splitlines()
        ]
        floor_event = next(
            event for event in events if event["event_type"] == "floor_wait"
        )
        delay_event = next(
            event
            for event in events
            if event["event_type"] == "turn_start_delay_applied"
        )

        assert floor_event["details"]["reason"] == "no_request"
        assert floor_event["details"]["fallback_speaker"] == "counselor"
        assert delay_event["turn_id"] == 1
        assert delay_event["speaker_id"] == "counselor"
        assert 100 <= delay_event["details"]["delay_ms"] <= 200
        assert delay_event["details"]["preferred_timing"] == "immediate"
        assert delay_event["details"]["effective_preferred_timing"] == "immediate"
        assert delay_event["details"]["action"] == "WAIT"

    asyncio.run(scenario())


def test_runtime_contextual_fallback_uses_fixed_sequence_without_client_request(
    tmp_path,
) -> None:
    async def scenario() -> None:
        counselor = FakeAgent(
            "counselor",
            "カウンセラー生成:{input_transcript}",
            timing_decisions=[{"agent_id": "counselor", "action": "WAIT"}],
        )
        client_a = FakeAgent("client_a", "A生成:{input_transcript}")
        client_b = FakeAgent(
            "client_b",
            "B生成:{input_transcript}",
            timing_decisions=[{"agent_id": "client_b", "action": "WAIT"}],
        )
        runtime = ConversationRuntime(
            config=RuntimeConfig(
                session_id="session_contextual_fallback_fixed_sequence_test",
                max_turns=1,
                speaker_selection_policy="turn_boundary_timing",
                participants={
                    "counselor": ParticipantConfig(
                        speaker_id="counselor",
                        role="counselor",
                        display_name="カウンセラー",
                    ),
                    "client_a": ParticipantConfig(
                        speaker_id="client_a",
                        role="client",
                        display_name="妻",
                        initial_transcript="Aの初回発話です。",
                    ),
                    "client_b": ParticipantConfig(
                        speaker_id="client_b",
                        role="client",
                        display_name="夫",
                    ),
                },
                fixed_speaker_sequence=(
                    "counselor",
                    "client_b",
                    "client_a",
                    "counselor",
                ),
            ),
            agents={
                "counselor": counselor,
                "client_a": client_a,
                "client_b": client_b,
            },
            sessions_dir=tmp_path,
        )

        turns = await runtime.run()

        assert [turn.speaker for turn in turns] == ["client_a", "counselor"]

        event_path = (
            tmp_path
            / "session_contextual_fallback_fixed_sequence_test"
            / "internal"
            / "events"
            / "session_contextual_fallback_fixed_sequence_test.jsonl"
        )
        events = [
            json.loads(line)
            for line in event_path.read_text(encoding="utf-8").splitlines()
        ]
        floor_event = next(
            event for event in events if event["event_type"] == "floor_wait"
        )
        selection = floor_event["details"]["fallback_selection"]
        client_b_candidate = next(
            item for item in selection["candidates"] if item["speaker_id"] == "client_b"
        )

        assert floor_event["details"]["fallback_speaker"] == "counselor"
        assert selection["base_speaker_id"] == "counselor"
        assert selection["speaker_id"] == "counselor"
        assert selection["strategy"] == "contextual_score"
        assert "client_reply_continuity" in client_b_candidate["reasons"]
        assert "fixed_sequence" in selection["candidates"][0]["reasons"]

    asyncio.run(scenario())


def test_runtime_turn_boundary_timing_passes_counselor_target_to_generation(
    tmp_path,
) -> None:
    async def scenario() -> None:
        counselor = FakeAgent(
            "counselor",
            "カウンセラー生成:{input_transcript}",
            timing_decisions=[
                {
                    "agent_id": "counselor",
                    "action": "REQUEST_MAIN_FLOOR",
                    "target": "client_a",
                    "preferred_timing": "natural_pause",
                    "urgency": 0.45,
                },
            ],
        )
        client_a = FakeAgent("client_a", "A生成:{input_transcript}")
        client_b = FakeAgent(
            "client_b",
            "B生成:{input_transcript}",
            timing_decisions=[
                {
                    "agent_id": "client_b",
                    "action": "REQUEST_MAIN_FLOOR",
                    "target": "counselor",
                    "preferred_timing": "natural_pause",
                    "urgency": 0.55,
                }
            ],
        )
        runtime = ConversationRuntime(
            config=RuntimeConfig(
                session_id="session_client_reply_counselor_target_test",
                max_turns=1,
                speaker_selection_policy="turn_boundary_timing",
                turn_taking_max_reconsider_rounds=0,
                participants={
                    "counselor": ParticipantConfig(
                        speaker_id="counselor",
                        role="counselor",
                        display_name="カウンセラー",
                    ),
                    "client_a": ParticipantConfig(
                        speaker_id="client_a",
                        role="client",
                        display_name="妻",
                        initial_transcript="妻の直前発話です。",
                    ),
                    "client_b": ParticipantConfig(
                        speaker_id="client_b",
                        role="client",
                        display_name="夫",
                    ),
                },
                fixed_speaker_sequence=(
                    "counselor",
                    "client_a",
                    "counselor",
                    "client_b",
                ),
            ),
            agents={
                "counselor": counselor,
                "client_a": client_a,
                "client_b": client_b,
            },
            sessions_dir=tmp_path,
        )

        turns = await runtime.run()

        assert [turn.speaker for turn in turns] == ["client_a", "client_b"]
        assert any(
            "今回の主な宛先は「カウンセラー（counselor）」です。" in item
            for item in client_b.received_inputs
        )
        assert any("敬語・丁寧語" in item for item in client_b.received_inputs)
        assert not any(
            "関係性、口調、どの程度相手に話すか" in item
            for item in client_b.received_inputs
        )

        event_path = (
            tmp_path
            / "session_client_reply_counselor_target_test"
            / "internal"
            / "events"
            / "session_client_reply_counselor_target_test.jsonl"
        )
        events = [
            json.loads(line)
            for line in event_path.read_text(encoding="utf-8").splitlines()
        ]
        floor_event = next(
            event
            for event in events
            if event["event_type"] == "floor_conflict_resolved"
        )

        assert floor_event["speaker_id"] == "client_b"
        assert floor_event["details"]["reason"] == "neutral_priority"

    asyncio.run(scenario())


def test_runtime_turn_boundary_timing_waits_for_slow_decisions(
    tmp_path,
) -> None:
    async def scenario() -> None:
        counselor = SlowTimingFakeAgent(
            "counselor",
            "カウンセラー生成:{input_transcript}",
            timing_decisions=[
                {
                    "agent_id": "counselor",
                    "action": "REQUEST_MAIN_FLOOR",
                    "explicitly_addressed": True,
                }
            ],
        )
        client_a = FakeAgent("client_a", "A生成:{input_transcript}")
        client_b = FakeAgent(
            "client_b",
            "B生成:{input_transcript}",
            timing_decisions=[{"agent_id": "client_b", "action": "REQUEST_MAIN_FLOOR"}],
        )
        runtime = ConversationRuntime(
            config=RuntimeConfig(
                session_id="session_turn_boundary_timing_timeout_test",
                max_turns=1,
                speaker_selection_policy="turn_boundary_timing",
                turn_taking_decision_timeout_ms=1,
                turn_taking_max_reconsider_rounds=0,
                participants={
                    "counselor": ParticipantConfig(
                        speaker_id="counselor",
                        role="counselor",
                        display_name="カウンセラー",
                    ),
                    "client_a": ParticipantConfig(
                        speaker_id="client_a",
                        role="client",
                        display_name="クライアントA",
                        initial_transcript="Aの初回発話です。",
                    ),
                    "client_b": ParticipantConfig(
                        speaker_id="client_b",
                        role="client",
                        display_name="クライアントB",
                    ),
                },
                fixed_speaker_sequence=(
                    "counselor",
                    "client_a",
                    "counselor",
                    "client_b",
                ),
            ),
            agents={
                "counselor": counselor,
                "client_a": client_a,
                "client_b": client_b,
            },
            sessions_dir=tmp_path,
        )

        turns = await runtime.run()

        assert [turn.speaker for turn in turns] == ["client_a", "counselor"]
        assert client_b.received_timing_inputs == [turns[0].generated_text]

        event_path = (
            tmp_path
            / "session_turn_boundary_timing_timeout_test"
            / "internal"
            / "events"
            / "session_turn_boundary_timing_timeout_test.jsonl"
        )
        events = [
            json.loads(line)
            for line in event_path.read_text(encoding="utf-8").splitlines()
        ]
        floor_event = next(
            event
            for event in events
            if event["details"].get("reason") == "explicitly_addressed_participant"
        )
        decisions = {
            item["agent_id"]: item for item in floor_event["details"]["decisions"]
        }

        assert floor_event["speaker_id"] == "counselor"
        assert decisions["counselor"]["action"] == "REQUEST_MAIN_FLOOR"
        assert decisions["counselor"]["explicitly_addressed"] is True
        assert decisions["client_b"]["action"] == "REQUEST_MAIN_FLOOR"
        assert any(event["event_type"] == "timing_decision_waiting" for event in events)

    asyncio.run(scenario())


def test_runtime_turn_boundary_timing_prioritizes_explicitly_addressed_client(
    tmp_path,
) -> None:
    async def scenario() -> None:
        counselor = FakeAgent(
            "counselor",
            "奥さんはどう思われますか？",
            timing_decisions=[
                {"agent_id": "counselor", "action": "REQUEST_MAIN_FLOOR"}
            ],
        )
        client_a = FakeAgent(
            "client_a",
            "妻応答:{input_transcript}",
            timing_decisions=[
                {
                    "agent_id": "client_a",
                    "action": "WAIT",
                    "target": "counselor",
                    "explicitly_addressed": True,
                    "preferred_timing": "short_hold",
                    "urgency": 0.25,
                }
            ],
        )
        client_b = FakeAgent(
            "client_b",
            "夫応答:{input_transcript}",
            timing_decisions=[
                {"agent_id": "client_b", "action": "WAIT"},
                {
                    "agent_id": "client_b",
                    "action": "REQUEST_MAIN_FLOOR",
                    "target": "counselor",
                    "explicitly_addressed": False,
                    "preferred_timing": "natural_pause",
                    "urgency": 0.5,
                },
            ],
        )
        runtime = ConversationRuntime(
            config=RuntimeConfig(
                session_id="session_explicit_address_ai_counselor_test",
                max_turns=2,
                speaker_selection_policy="turn_boundary_timing",
                turn_taking_max_reconsider_rounds=0,
                participants={
                    "counselor": ParticipantConfig(
                        speaker_id="counselor",
                        role="counselor",
                        display_name="カウンセラー",
                    ),
                    "client_a": ParticipantConfig(
                        speaker_id="client_a",
                        role="client",
                        display_name="妻",
                        initial_transcript="今日は娘のことで相談に来ました。",
                    ),
                    "client_b": ParticipantConfig(
                        speaker_id="client_b",
                        role="client",
                        display_name="夫",
                    ),
                },
                fixed_speaker_sequence=(
                    "counselor",
                    "client_a",
                    "counselor",
                    "client_b",
                ),
            ),
            agents={
                "counselor": counselor,
                "client_a": client_a,
                "client_b": client_b,
            },
            sessions_dir=tmp_path,
        )

        turns = await runtime.run()

        assert [turn.speaker for turn in turns] == [
            "client_a",
            "counselor",
            "client_a",
        ]

        event_path = (
            tmp_path
            / "session_explicit_address_ai_counselor_test"
            / "internal"
            / "events"
            / "session_explicit_address_ai_counselor_test.jsonl"
        )
        events = [
            json.loads(line)
            for line in event_path.read_text(encoding="utf-8").splitlines()
        ]
        floor_event = next(
            event
            for event in events
            if event["details"].get("reason") == "explicitly_addressed_participant"
        )

        assert floor_event["speaker_id"] == "client_a"
        assert floor_event["details"]["reason"] == "explicitly_addressed_participant"
        assert floor_event["details"]["granted_agent_id"] == "client_a"
        assert set(floor_event["details"]["conflict_agent_ids"]) == {
            "client_a",
            "client_b",
        }

    asyncio.run(scenario())


def test_runtime_distributed_timing_collects_timing_decisions_concurrently(
    tmp_path,
) -> None:
    async def scenario() -> None:
        counselor_started = asyncio.Event()
        client_b_started = asyncio.Event()
        release = asyncio.Event()
        counselor = CoordinatedTimingFakeAgent(
            "counselor",
            "カウンセラー生成:{input_transcript}",
            started=counselor_started,
            release=release,
            action="REQUEST_MAIN_FLOOR",
        )
        client_a = FakeAgent("client_a", "A生成:{input_transcript}")
        client_b = CoordinatedTimingFakeAgent(
            "client_b",
            "B生成:{input_transcript}",
            started=client_b_started,
            release=release,
            action="WAIT",
        )
        runtime = ConversationRuntime(
            config=RuntimeConfig(
                session_id="session_distributed_timing_parallel_test",
                max_turns=1,
                speaker_selection_policy="distributed_timing",
                participants={
                    "counselor": ParticipantConfig(
                        speaker_id="counselor",
                        role="counselor",
                        display_name="カウンセラー",
                    ),
                    "client_a": ParticipantConfig(
                        speaker_id="client_a",
                        role="client",
                        display_name="クライアントA",
                        initial_transcript="Aの初回発話です。",
                    ),
                    "client_b": ParticipantConfig(
                        speaker_id="client_b",
                        role="client",
                        display_name="クライアントB",
                    ),
                },
                fixed_speaker_sequence=(
                    "counselor",
                    "client_a",
                    "counselor",
                    "client_b",
                ),
            ),
            agents={
                "counselor": counselor,
                "client_a": client_a,
                "client_b": client_b,
            },
            sessions_dir=tmp_path,
        )

        task = asyncio.create_task(runtime.run())
        await asyncio.wait_for(counselor_started.wait(), timeout=1.0)
        timed_out_waiting_for_second_decision = False
        try:
            await asyncio.wait_for(client_b_started.wait(), timeout=0.05)
        except TimeoutError:
            timed_out_waiting_for_second_decision = True
        release.set()
        turns = await asyncio.wait_for(task, timeout=2.0)

        assert timed_out_waiting_for_second_decision is False
        assert [turn.speaker for turn in turns] == ["client_a", "counselor"]

    asyncio.run(scenario())


def test_runtime_distributed_timing_adopts_floor_prediction_provisional_generation(
    tmp_path,
) -> None:
    async def scenario() -> None:
        counselor = SlowTimingFakeAgent(
            "counselor",
            "カウンセラー生成:{input_transcript}",
            timing_decisions=[
                {"agent_id": "counselor", "action": "REQUEST_MAIN_FLOOR"}
            ],
        )
        client_a = FakeAgent("client_a", "A生成:{input_transcript}")
        runtime = ConversationRuntime(
            config=RuntimeConfig(
                session_id="session_floor_prediction_provisional_adopt_test",
                max_turns=1,
                speaker_selection_policy="distributed_timing",
                participants={
                    "counselor": ParticipantConfig(
                        speaker_id="counselor",
                        role="counselor",
                        display_name="カウンセラー",
                    ),
                    "client_a": ParticipantConfig(
                        speaker_id="client_a",
                        role="client",
                        display_name="クライアントA",
                        initial_transcript="Aの初回発話です。",
                    ),
                },
                fixed_speaker_sequence=(
                    "counselor",
                    "client_a",
                ),
            ),
            agents={
                "counselor": counselor,
                "client_a": client_a,
            },
            sessions_dir=tmp_path,
        )

        turns = await runtime.run()

        assert [turn.speaker for turn in turns] == ["client_a", "counselor"]
        assert turns[1].used_provisional_generation is True
        assert turns[1].provisional_input_transcript == turns[0].generated_text
        assert counselor.received_inputs == [turns[0].generated_text]

        event_path = (
            tmp_path
            / "session_floor_prediction_provisional_adopt_test"
            / "internal"
            / "events"
            / "session_floor_prediction_provisional_adopt_test.jsonl"
        )
        events = [
            json.loads(line)
            for line in event_path.read_text(encoding="utf-8").splitlines()
        ]
        request_event = next(
            event
            for event in events
            if event["event_type"] == "provisional_llm_request_started"
        )
        adopted_event = next(
            event
            for event in events
            if event["event_type"] == "provisional_llm_adopted"
        )
        floor_event = next(
            event for event in events if event["event_type"] == "floor_granted"
        )

        assert request_event["details"]["source"] == "floor_prediction"
        assert request_event["details"]["prediction_reason"] == "fixed_sequence_window"
        assert adopted_event["details"]["source"] == "floor_prediction"
        assert floor_event["speaker_id"] == "counselor"

    asyncio.run(scenario())


def test_runtime_distributed_timing_adopts_realtime_floor_prediction_audio(
    tmp_path,
) -> None:
    async def scenario() -> None:
        counselor = RealtimeProvisionalAgent("counselor")
        client_a = FakeAgent("client_a", "A生成:{input_transcript}")
        runtime = ConversationRuntime(
            config=RuntimeConfig(
                session_id="session_realtime_floor_prediction_adopt_test",
                max_turns=1,
                speaker_selection_policy="distributed_timing",
                participants={
                    "counselor": ParticipantConfig(
                        speaker_id="counselor",
                        role="counselor",
                        display_name="カウンセラー",
                    ),
                    "client_a": ParticipantConfig(
                        speaker_id="client_a",
                        role="client",
                        display_name="クライアントA",
                        initial_transcript="Aの初回発話です。",
                    ),
                },
                fixed_speaker_sequence=(
                    "counselor",
                    "client_a",
                ),
            ),
            agents={
                "counselor": counselor,
                "client_a": client_a,
            },
            sessions_dir=tmp_path,
        )

        turns = await runtime.run()

        assert [turn.speaker for turn in turns] == ["client_a", "counselor"]
        assert turns[1].used_provisional_generation is True
        assert turns[1].generated_text == "counselor realtime done"
        assert counselor.stream_calls == [
            {
                "turn_id": 1,
                "speaker": "counselor",
                "input_transcript": turns[0].generated_text,
                "format_input": True,
                "additional_instruction": None,
            }
        ]

        event_path = (
            tmp_path
            / "session_realtime_floor_prediction_adopt_test"
            / "internal"
            / "events"
            / "session_realtime_floor_prediction_adopt_test.jsonl"
        )
        events = [
            json.loads(line)
            for line in event_path.read_text(encoding="utf-8").splitlines()
        ]
        adopted_event = next(
            event
            for event in events
            if event["event_type"] == "provisional_llm_adopted"
        )
        assert adopted_event["details"]["source"] == "floor_prediction"
        assert adopted_event["details"]["generation_modality"] == "realtime_audio"

    asyncio.run(scenario())


def test_runtime_skips_running_realtime_floor_prediction_audio(
    tmp_path,
) -> None:
    async def scenario() -> None:
        counselor = FailingAfterAudioRealtimeProvisionalAgent("counselor")
        client_a = FakeAgent("client_a", "A生成:{input_transcript}")
        runtime = ConversationRuntime(
            config=RuntimeConfig(
                session_id="session_realtime_floor_prediction_running_skip_test",
                max_turns=1,
                speaker_selection_policy="distributed_timing",
                participants={
                    "counselor": ParticipantConfig(
                        speaker_id="counselor",
                        role="counselor",
                        display_name="カウンセラー",
                    ),
                    "client_a": ParticipantConfig(
                        speaker_id="client_a",
                        role="client",
                        display_name="クライアントA",
                        initial_transcript="Aの初回発話です。",
                    ),
                },
                fixed_speaker_sequence=(
                    "counselor",
                    "client_a",
                ),
            ),
            agents={
                "counselor": counselor,
                "client_a": client_a,
            },
            sessions_dir=tmp_path,
        )

        turns = await runtime.run()

        assert [turn.speaker for turn in turns] == ["client_a", "counselor"]
        assert turns[1].used_provisional_generation is False
        assert turns[1].generated_text == "counselor final done"
        assert len(counselor.stream_calls) == 2
        assert counselor.close_count >= 1

        event_path = (
            tmp_path
            / "session_realtime_floor_prediction_running_skip_test"
            / "internal"
            / "events"
            / "session_realtime_floor_prediction_running_skip_test.jsonl"
        )
        events = [
            json.loads(line)
            for line in event_path.read_text(encoding="utf-8").splitlines()
        ]
        event_types = [event["event_type"] for event in events]
        discarded_event = next(
            event
            for event in events
            if event["event_type"] == "provisional_llm_discarded"
        )
        tts_first_audio_events = [
            event
            for event in events
            if event["event_type"] == "tts_first_audio_chunk" and event["turn_id"] == 1
        ]

        assert "provisional_llm_adopted" not in event_types
        assert discarded_event["speaker_id"] == "counselor"
        assert discarded_event["details"]["reason"] == "realtime_provisional_incomplete"
        assert len(tts_first_audio_events) == 1
        assert tts_first_audio_events[0]["details"]["speech_source"] == "realtime_api"

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "speaker_selection_policy",
    ["distributed_timing", "turn_boundary_timing"],
)
def test_runtime_retries_final_realtime_when_provisional_audio_fails(
    tmp_path,
    speaker_selection_policy: str,
) -> None:
    async def scenario() -> None:
        counselor = FailingFirstRealtimeProvisionalAgent("counselor")
        client_a = FakeAgent("client_a", "A生成:{input_transcript}")
        runtime = ConversationRuntime(
            config=RuntimeConfig(
                session_id="session_realtime_provisional_fallback_test",
                max_turns=1,
                speaker_selection_policy=speaker_selection_policy,
                participants={
                    "counselor": ParticipantConfig(
                        speaker_id="counselor",
                        role="counselor",
                        display_name="カウンセラー",
                    ),
                    "client_a": ParticipantConfig(
                        speaker_id="client_a",
                        role="client",
                        display_name="クライアントA",
                        initial_transcript="Aの初回発話です。",
                    ),
                },
                fixed_speaker_sequence=(
                    "counselor",
                    "client_a",
                ),
            ),
            agents={
                "counselor": counselor,
                "client_a": client_a,
            },
            sessions_dir=tmp_path,
        )

        turns = await runtime.run()

        assert runtime.status.phase is RuntimePhase.COMPLETED
        assert [turn.speaker for turn in turns] == ["client_a", "counselor"]
        assert turns[1].used_provisional_generation is False
        assert turns[1].provisional_input_transcript is None
        assert turns[1].generated_text == "counselor fallback done"
        assert len(counselor.stream_calls) == 2
        assert counselor.stream_calls[0]["additional_instruction"] is None
        assert counselor.stream_calls[1]["additional_instruction"] is None

        event_path = (
            tmp_path
            / "session_realtime_provisional_fallback_test"
            / "internal"
            / "events"
            / "session_realtime_provisional_fallback_test.jsonl"
        )
        events = [
            json.loads(line)
            for line in event_path.read_text(encoding="utf-8").splitlines()
        ]
        event_types = [event["event_type"] for event in events]
        assert "provisional_llm_error" in event_types
        assert "provisional_llm_fallback" in event_types
        assert "runtime_error" not in event_types
        error_event = next(
            event for event in events if event["event_type"] == "provisional_llm_error"
        )
        assert error_event["details"]["error_message"] == (
            "provisional realtime rejected"
        )

    asyncio.run(scenario())


def test_runtime_distributed_timing_skips_floor_prediction_for_two_clients(
    tmp_path,
) -> None:
    async def scenario() -> None:
        counselor = SlowTimingFakeAgent(
            "counselor",
            "カウンセラー生成:{input_transcript}",
            timing_decisions=[{"agent_id": "counselor", "action": "WAIT"}],
        )
        client_a = FakeAgent("client_a", "A生成:{input_transcript}")
        client_b = FakeAgent(
            "client_b",
            "B生成:{input_transcript}",
            timing_decisions=[
                {
                    "agent_id": "client_b",
                    "action": "REQUEST_MAIN_FLOOR",
                    "speech_ms_last_window": 0,
                    "request_timestamp_ms": 1,
                }
            ],
        )
        runtime = ConversationRuntime(
            config=RuntimeConfig(
                session_id="session_two_client_floor_prediction_skip_test",
                max_turns=1,
                speaker_selection_policy="distributed_timing",
                participants={
                    "counselor": ParticipantConfig(
                        speaker_id="counselor",
                        role="counselor",
                        display_name="カウンセラー",
                    ),
                    "client_a": ParticipantConfig(
                        speaker_id="client_a",
                        role="client",
                        display_name="クライアントA",
                        initial_transcript="Aの初回発話です。",
                    ),
                    "client_b": ParticipantConfig(
                        speaker_id="client_b",
                        role="client",
                        display_name="クライアントB",
                    ),
                },
                fixed_speaker_sequence=(
                    "counselor",
                    "client_a",
                    "counselor",
                    "client_b",
                ),
            ),
            agents={
                "counselor": counselor,
                "client_a": client_a,
                "client_b": client_b,
            },
            sessions_dir=tmp_path,
        )

        turns = await runtime.run()

        assert [turn.speaker for turn in turns] == ["client_a", "client_b"]
        assert counselor.received_inputs == []
        assert len(client_b.received_inputs) == 1
        assert turns[0].generated_text in client_b.received_inputs[0]
        assert "今回の主な宛先は固定されていません。" in client_b.received_inputs[0]
        assert "自然な宛先と話し方" in client_b.received_inputs[0]
        assert turns[1].used_provisional_generation is False

        event_path = (
            tmp_path
            / "session_two_client_floor_prediction_skip_test"
            / "internal"
            / "events"
            / "session_two_client_floor_prediction_skip_test.jsonl"
        )
        event_types = [
            json.loads(line)["event_type"]
            for line in event_path.read_text(encoding="utf-8").splitlines()
        ]

        assert "provisional_llm_request_started" not in event_types
        assert "provisional_llm_discarded" not in event_types

    asyncio.run(scenario())


def test_runtime_passes_turn_taking_target_to_generation_instruction(
    tmp_path,
) -> None:
    async def scenario() -> None:
        counselor = FakeAgent(
            "counselor",
            "カウンセラー生成:{input_transcript}",
            timing_decisions=[{"agent_id": "counselor", "action": "WAIT"}],
        )
        client_a = FakeAgent("client_a", "A生成:{input_transcript}")
        client_b = FakeAgent(
            "client_b",
            "B生成:{input_transcript}",
            timing_decisions=[
                {
                    "agent_id": "client_b",
                    "action": "REQUEST_MAIN_FLOOR",
                    "target": "client_a",
                    "preferred_timing": "immediate",
                    "urgency": 0.7,
                }
            ],
        )
        runtime = ConversationRuntime(
            config=RuntimeConfig(
                session_id="session_turn_target_generation_test",
                max_turns=1,
                speaker_selection_policy="distributed_timing",
                participants={
                    "counselor": ParticipantConfig(
                        speaker_id="counselor",
                        role="counselor",
                        display_name="カウンセラー",
                    ),
                    "client_a": ParticipantConfig(
                        speaker_id="client_a",
                        role="client",
                        display_name="妻",
                        initial_transcript="Aの初回発話です。",
                    ),
                    "client_b": ParticipantConfig(
                        speaker_id="client_b",
                        role="client",
                        display_name="夫",
                    ),
                },
                fixed_speaker_sequence=(
                    "counselor",
                    "client_a",
                    "counselor",
                    "client_b",
                ),
            ),
            agents={
                "counselor": counselor,
                "client_a": client_a,
                "client_b": client_b,
            },
            sessions_dir=tmp_path,
        )

        turns = await runtime.run()

        assert [turn.speaker for turn in turns] == ["client_a", "client_b"]
        assert turns[1].used_provisional_generation is False
        assert any(
            "今回の主な宛先は「妻（client_a）」です。" in item
            for item in client_b.received_inputs
        )
        assert any(
            "直前文脈で自然なら、その相手へ直接反応" in item
            for item in client_b.received_inputs
        )
        assert all(
            "自分の感じ方や具体的な経験を続けて" not in item
            for item in client_b.received_inputs
        )
        assert any(
            "関係性、口調、どの程度相手に話すか" in item
            for item in client_b.received_inputs
        )
        event_path = (
            tmp_path
            / "session_turn_target_generation_test"
            / "internal"
            / "events"
            / "session_turn_target_generation_test.jsonl"
        )
        event_types = [
            json.loads(line)["event_type"]
            for line in event_path.read_text(encoding="utf-8").splitlines()
        ]
        assert "provisional_llm_request_started" not in event_types
        assert "provisional_llm_discarded" not in event_types

    asyncio.run(scenario())


def test_runtime_distributed_timing_logs_overlap_and_yield_resolution(
    tmp_path,
) -> None:
    async def scenario() -> None:
        counselor = FakeAgent(
            "counselor",
            "カウンセラー生成:{input_transcript}",
            overlap_decisions=[{"agent_id": "counselor", "action": "YIELD"}],
        )
        client_a = FakeAgent(
            "client_a",
            "A生成:{input_transcript}",
        )
        client_b = FakeAgent(
            "client_b",
            "B生成:{input_transcript}",
            overlap_decisions=[{"agent_id": "client_b", "action": "CONTINUE"}],
        )
        runtime = ConversationRuntime(
            config=RuntimeConfig(
                session_id="session_distributed_timing_overlap_test",
                max_turns=1,
                speaker_selection_policy="distributed_timing",
                overlap_grace_ms=1,
                unresolved_overlap_limit_ms=9,
                participants={
                    "counselor": ParticipantConfig(
                        speaker_id="counselor",
                        role="counselor",
                        display_name="カウンセラー",
                    ),
                    "client_a": ParticipantConfig(
                        speaker_id="client_a",
                        role="client",
                        display_name="クライアントA",
                        initial_transcript="Aの初回発話です。",
                    ),
                    "client_b": ParticipantConfig(
                        speaker_id="client_b",
                        role="client",
                        display_name="クライアントB",
                    ),
                },
                fixed_speaker_sequence=(
                    "counselor",
                    "client_a",
                    "counselor",
                    "client_b",
                ),
            ),
            agents={
                "counselor": counselor,
                "client_a": client_a,
                "client_b": client_b,
            },
            sessions_dir=tmp_path,
        )

        turns = await runtime.run()

        assert [turn.speaker for turn in turns] == ["client_a", "client_b"]
        assert len(client_b.received_inputs) == 1
        assert turns[0].generated_text in client_b.received_inputs[0]
        assert "今回の主な宛先は固定されていません。" in client_b.received_inputs[0]
        assert "自然な宛先と話し方" in client_b.received_inputs[0]
        assert counselor.received_overlap_inputs == [turns[0].generated_text]
        assert client_b.received_overlap_inputs == [turns[0].generated_text]

        event_path = (
            tmp_path
            / "session_distributed_timing_overlap_test"
            / "internal"
            / "events"
            / "session_distributed_timing_overlap_test.jsonl"
        )
        events = [
            json.loads(line)
            for line in event_path.read_text(encoding="utf-8").splitlines()
        ]
        overlap_event = next(
            event for event in events if event["event_type"] == "overlap_detected"
        )
        resolved_event = next(
            event
            for event in events
            if event["event_type"] == "floor_conflict_resolved"
        )

        assert overlap_event["speaker_id"] is None
        assert overlap_event["details"]["result_type"] == "OVERLAP"
        assert overlap_event["details"]["overlap_grace_ms"] == 1
        assert overlap_event["details"]["unresolved_overlap_limit_ms"] == 9
        assert overlap_event["details"]["conflict_agent_ids"] == [
            "client_b",
            "counselor",
        ]
        assert resolved_event["speaker_id"] == "client_b"
        assert resolved_event["details"]["reason"] == "overlap_continue"
        assert resolved_event["details"]["yielded_agent_ids"] == ["counselor"]
        assert resolved_event["details"]["overlap_decisions"] == [
            {
                "agent_id": "client_b",
                "action": "CONTINUE",
                "preferred_timing": None,
                "urgency": None,
                "target": None,
                "reason_code": None,
                "prepared_intent": None,
                "expires_after_ms": None,
                "explicitly_addressed": None,
                "speech_ms_last_window": None,
                "request_timestamp_ms": None,
                "safety_intervention": None,
            },
            {
                "agent_id": "counselor",
                "action": "YIELD",
                "preferred_timing": None,
                "urgency": None,
                "target": None,
                "reason_code": None,
                "prepared_intent": None,
                "expires_after_ms": None,
                "explicitly_addressed": None,
                "speech_ms_last_window": None,
                "request_timestamp_ms": None,
                "safety_intervention": None,
            },
        ]

    asyncio.run(scenario())


def test_runtime_distributed_timing_times_out_unresolved_overlap(
    tmp_path,
) -> None:
    async def scenario() -> None:
        counselor = SlowOverlapFakeAgent(
            "counselor",
            "カウンセラー生成:{input_transcript}",
            overlap_decisions=[{"agent_id": "counselor", "action": "YIELD"}],
        )
        client_a = FakeAgent(
            "client_a",
            "A生成:{input_transcript}",
        )
        client_b = SlowOverlapFakeAgent(
            "client_b",
            "B生成:{input_transcript}",
            overlap_decisions=[{"agent_id": "client_b", "action": "CONTINUE"}],
        )
        runtime = ConversationRuntime(
            config=RuntimeConfig(
                session_id="session_distributed_timing_overlap_timeout_test",
                max_turns=1,
                speaker_selection_policy="distributed_timing",
                overlap_grace_ms=0,
                unresolved_overlap_limit_ms=1,
                participants={
                    "counselor": ParticipantConfig(
                        speaker_id="counselor",
                        role="counselor",
                        display_name="カウンセラー",
                    ),
                    "client_a": ParticipantConfig(
                        speaker_id="client_a",
                        role="client",
                        display_name="クライアントA",
                        initial_transcript="Aの初回発話です。",
                    ),
                    "client_b": ParticipantConfig(
                        speaker_id="client_b",
                        role="client",
                        display_name="クライアントB",
                    ),
                },
                fixed_speaker_sequence=(
                    "counselor",
                    "client_a",
                    "counselor",
                    "client_b",
                ),
            ),
            agents={
                "counselor": counselor,
                "client_a": client_a,
                "client_b": client_b,
            },
            sessions_dir=tmp_path,
        )

        turns = await runtime.run()

        assert [turn.speaker for turn in turns] == ["client_a", "client_a"]

        event_path = (
            tmp_path
            / "session_distributed_timing_overlap_timeout_test"
            / "internal"
            / "events"
            / "session_distributed_timing_overlap_timeout_test.jsonl"
        )
        events = [
            json.loads(line)
            for line in event_path.read_text(encoding="utf-8").splitlines()
        ]
        resolved_event = next(
            event
            for event in events
            if event["details"].get("reason") == "overlap_no_resolution_keep_active"
        )

        assert resolved_event["event_type"] == "floor_conflict_resolved"
        assert resolved_event["speaker_id"] == "client_a"
        assert resolved_event["details"]["overlap_resolution_timed_out"] is True
        assert resolved_event["details"]["unresolved_overlap_limit_ms"] == 1
        assert resolved_event["details"]["overlap_decisions"] == []

    asyncio.run(scenario())


def test_runtime_distributed_timing_prompt_requests_main_floor(
    tmp_path,
) -> None:
    async def scenario() -> None:
        counselor = FakeAgent(
            "counselor",
            "カウンセラー生成:{input_transcript}",
            timing_decisions=[
                {"agent_id": "counselor", "action": "REQUEST_MAIN_FLOOR"}
            ],
        )
        client_a = FakeAgent("client_a", "A生成:{input_transcript}")
        timing_llm = ScriptedTextLLM(
            '{"agent_id":"client_b","action":"WAIT"}',
        )
        client_b = StreamingAgent(
            speaker="client_b",
            llm=ScriptedTextLLM("unused"),
            timing_llm=timing_llm,
        )
        runtime = ConversationRuntime(
            config=RuntimeConfig(
                session_id="session_streaming_agent_main_floor_prompt_test",
                max_turns=1,
                speaker_selection_policy="distributed_timing",
                participants={
                    "counselor": ParticipantConfig(
                        speaker_id="counselor",
                        role="counselor",
                        display_name="カウンセラー",
                    ),
                    "client_a": ParticipantConfig(
                        speaker_id="client_a",
                        role="client",
                        display_name="クライアントA",
                        initial_transcript="Aの初回発話です。",
                    ),
                    "client_b": ParticipantConfig(
                        speaker_id="client_b",
                        role="client",
                        display_name="クライアントB",
                    ),
                },
                fixed_speaker_sequence=("counselor", "client_a", "client_b"),
            ),
            agents={
                "counselor": counselor,
                "client_a": client_a,
                "client_b": client_b,
            },
            sessions_dir=tmp_path,
        )

        turns = await runtime.run()

        assert [turn.speaker for turn in turns] == ["client_a", "counselor"]
        assert "REQUEST_BACKCHANNEL" not in timing_llm.system_prompts[0]
        assert "REQUEST_MAIN_FLOOR" in timing_llm.system_prompts[0]
        assert "具体的な待ち時間のばらつき" in timing_llm.system_prompts[0]
        assert "preferred_timing も immediate" in timing_llm.system_prompts[0]
        assert "agent_id: client_b" in timing_llm.latest_inputs[0]

    asyncio.run(scenario())


def test_fake_runtime_falls_back_to_legacy_initial_client_transcript(
    tmp_path,
) -> None:
    async def scenario() -> None:
        runtime = ConversationRuntime(
            config=RuntimeConfig(
                session_id="session_runtime_two_client_initial_fallback_test",
                max_turns=1,
                initial_client_transcript="共通の初回発話です。",
                speaker_selection_policy="fixed_round_robin",
                participants={
                    "counselor": ParticipantConfig(
                        speaker_id="counselor",
                        role="counselor",
                        display_name="カウンセラー",
                    ),
                    "client_a": ParticipantConfig(
                        speaker_id="client_a",
                        role="client",
                        display_name="クライアントA",
                        initial_transcript="",
                    ),
                    "client_b": ParticipantConfig(
                        speaker_id="client_b",
                        role="client",
                        display_name="クライアントB",
                        initial_transcript="",
                    ),
                },
                fixed_speaker_sequence=(
                    "counselor",
                    "client_b",
                    "counselor",
                    "client_a",
                ),
            ),
            agents={
                "counselor": FakeAgent(
                    "counselor",
                    "カウンセラー生成:{input_transcript}",
                ),
                "client_a": FakeAgent("client_a", "A生成:{input_transcript}"),
                "client_b": FakeAgent("client_b", "B生成:{input_transcript}"),
            },
            sessions_dir=tmp_path,
        )

        turns = await runtime.run()

        assert [turn.speaker for turn in turns] == ["client_b", "counselor"]
        assert turns[0].generated_text == "B生成:"
        assert turns[0].stt_final_transcript == turns[0].generated_text
        assert turns[1].input_transcript == turns[0].generated_text
        assert (
            "初回相談内容" in runtime.controller.agents["client_b"].received_inputs[0]
        )
        assert "共通の初回発話です。" in (
            runtime.controller.agents["client_b"].received_inputs[0]
        )

    asyncio.run(scenario())


def test_fake_runtime_can_run_two_client_fixed_sequence_from_client_a(
    tmp_path,
) -> None:
    async def scenario() -> None:
        counselor = FakeAgent("counselor", "カウンセラー生成:{input_transcript}")
        client_a = FakeAgent("client_a", "A生成:{input_transcript}")
        client_b = FakeAgent("client_b", "B生成:{input_transcript}")
        runtime = ConversationRuntime(
            config=RuntimeConfig(
                session_id="session_runtime_two_client_fixed_mvp_test",
                max_turns=3,
                initial_client_transcript="共通の初回発話です。",
                speaker_selection_policy="fixed_round_robin",
                participants={
                    "counselor": ParticipantConfig(
                        speaker_id="counselor",
                        role="counselor",
                        display_name="カウンセラー",
                    ),
                    "client_a": ParticipantConfig(
                        speaker_id="client_a",
                        role="client",
                        display_name="クライアントA",
                        initial_transcript="Aの初回発話です。",
                    ),
                    "client_b": ParticipantConfig(
                        speaker_id="client_b",
                        role="client",
                        display_name="クライアントB",
                        initial_transcript="",
                    ),
                },
                fixed_speaker_sequence=(
                    "counselor",
                    "client_b",
                    "counselor",
                    "client_a",
                ),
            ),
            agents={
                "counselor": counselor,
                "client_a": client_a,
                "client_b": client_b,
            },
            sessions_dir=tmp_path,
        )

        turns = await runtime.run()

        assert [turn.speaker for turn in turns] == [
            "client_a",
            "counselor",
            "client_b",
            "counselor",
        ]
        assert turns[0].generated_text == "A生成:"
        assert counselor.received_inputs == [
            turns[0].generated_text,
            turns[2].generated_text,
        ]
        assert len(client_b.received_inputs) == 1
        assert turns[1].generated_text in client_b.received_inputs[0]
        assert "今回の主な宛先は固定されていません。" in client_b.received_inputs[0]
        assert "自然な宛先と話し方" in client_b.received_inputs[0]
        assert "特定の参加者を常に優先せず" in client_b.received_inputs[0]
        assert len(client_a.received_inputs) == 1
        assert "初回相談内容" in client_a.received_inputs[0]
        assert "Aの初回発話です。" in client_a.received_inputs[0]

    asyncio.run(scenario())


def test_runtime_does_not_register_participant_listener_stt_queues_for_generated_text(
    tmp_path,
) -> None:
    class FailingQueueStt:
        def __init__(self) -> None:
            self.calls = 0

        async def transcribe_from_queue(self, *, session_id, turn_id, speaker, queue):
            _ = session_id, turn_id, speaker, queue
            self.calls += 1
            raise AssertionError("STT should not run during generated-text runtime")

    async def scenario() -> None:
        stt = FailingQueueStt()
        runtime = ConversationRuntime(
            config=RuntimeConfig(
                session_id="session_runtime_two_client_listener_stt_test",
                max_turns=2,
                speaker_selection_policy="fixed_round_robin",
                participants={
                    "counselor": ParticipantConfig(
                        speaker_id="counselor",
                        role="counselor",
                        display_name="カウンセラー",
                    ),
                    "client_a": ParticipantConfig(
                        speaker_id="client_a",
                        role="client",
                        display_name="クライアントA",
                        initial_transcript="Aの初回発話です。",
                    ),
                    "client_b": ParticipantConfig(
                        speaker_id="client_b",
                        role="client",
                        display_name="クライアントB",
                    ),
                },
                fixed_speaker_sequence=(
                    "counselor",
                    "client_b",
                    "counselor",
                    "client_a",
                ),
            ),
            agents={
                "counselor": FakeAgent(
                    "counselor", "カウンセラー生成:{input_transcript}"
                ),
                "client_a": FakeAgent("client_a", "A生成:{input_transcript}"),
                "client_b": FakeAgent("client_b", "B生成:{input_transcript}"),
            },
            stt=stt,
            sessions_dir=tmp_path,
        )

        turns = await runtime.run()

        assert [turn.speaker for turn in turns] == [
            "client_a",
            "counselor",
            "client_b",
        ]
        assert runtime.audio_bus.listener_stt_queues == {}
        assert stt.calls == 0
        assert turns[0].recipient_ids == ("counselor", "client_b")
        assert turns[0].listener_stt_final_transcripts == {}
        assert turns[1].recipient_ids == ("client_a", "client_b")
        assert turns[1].listener_stt_final_transcripts == {}
        transcript_path = (
            tmp_path
            / "session_runtime_two_client_listener_stt_test"
            / "internal"
            / "transcripts"
            / "session_runtime_two_client_listener_stt_test.jsonl"
        )
        transcripts = [
            json.loads(line)
            for line in transcript_path.read_text(encoding="utf-8").splitlines()
            if json.loads(line)["transcript_type"] == "generated_final"
        ]
        first_final = transcripts[0]
        assert first_final["speaker_id"] == "client_a"
        assert first_final["recipient_ids"] == ["counselor", "client_b"]
        assert first_final["metadata"]["speaker_display_name"] == "クライアントA"
        assert first_final["metadata"]["text_source"] == "generated_text"

    asyncio.run(scenario())


def test_runtime_logs_error_event_when_initial_realtime_audio_fails(tmp_path) -> None:
    async def scenario() -> None:
        runtime = ConversationRuntime(
            config=RuntimeConfig(
                session_id="session_runtime_error_test",
                max_turns=2,
                initial_client_transcript="初回相談です。",
            ),
            agents={
                "counselor": FakeAgent(
                    "counselor", "カウンセラー生成:{input_transcript}"
                ),
                "client": FailingRealtimeAgent(),
            },
            sessions_dir=tmp_path,
        )

        try:
            await runtime.run()
        except RuntimeError as exc:
            assert str(exc) == "realtime voice rejected"
        else:
            raise AssertionError("runtime.run() should fail")

        assert runtime.status.phase is RuntimePhase.ERROR
        assert runtime.status.last_event_type == "runtime_error"
        event_path = (
            tmp_path
            / "session_runtime_error_test"
            / "internal"
            / "events"
            / "session_runtime_error_test.jsonl"
        )
        events = [
            json.loads(line)
            for line in event_path.read_text(encoding="utf-8").splitlines()
        ]
        runtime_error = events[-1]
        assert runtime_error["event_type"] == "runtime_error"
        assert runtime_error["turn_id"] == 0
        assert runtime_error["speaker"] == "client"
        assert runtime_error["details"] == {
            "error_type": "RuntimeError",
            "message": "realtime voice rejected",
        }

    asyncio.run(scenario())


def test_runtime_elapsed_time_stop_condition_stops_before_next_turn(tmp_path) -> None:
    async def scenario() -> None:
        runtime = ConversationRuntime(
            config=RuntimeConfig(
                session_id="session_elapsed_stop_test",
                stop_condition="elapsed_time",
                max_elapsed_seconds=0.000001,
                max_turns=10,
                initial_client_transcript="初回相談です。",
            ),
            agents={
                "counselor": SlowFakeAgent(
                    "counselor",
                    "カウンセラー生成:{input_transcript}",
                ),
                "client": SlowFakeAgent(
                    "client", "クライアント生成:{input_transcript}"
                ),
            },
            sessions_dir=tmp_path,
        )

        turns = await runtime.run()

        assert runtime.status.phase is RuntimePhase.COMPLETED
        assert [turn.turn_id for turn in turns] == [0]

    asyncio.run(scenario())


def test_counselor_target_instruction_leaves_style_to_configured_prompt() -> None:
    from counseling_voice_demo.runtime.controller import _response_target_instruction

    instruction = _response_target_instruction(
        RuntimeConfig(session_id="target-style"),
        speaker="counselor",
        target="client",
    )

    assert "client" in instruction
    assert "1発話は短く" not in instruction
    assert "口論を長引かせず" not in instruction
    assert "設定されたカウンセラープロンプト" in instruction


@pytest.mark.parametrize(
    "actual_text",
    ["相談の入口に立てたことが、まず大切なのですね。", "勝手に追加された説明です。"],
)
def test_runtime_audits_selected_director_text_without_rewriting_output(
    tmp_path, actual_text
) -> None:
    async def scenario():
        runtime = ConversationRuntime(
            config=RuntimeConfig(session_id="director-output-audit", max_turns=1),
            agents={
                "counselor": FakeAgent("counselor", actual_text),
                "client": FakeAgent("client", "相談です。"),
            },
            prompt_director=RecordingPromptDirector(),
            sessions_dir=tmp_path,
        )
        turns = await runtime.run()
        assert turns[-1].generated_text == actual_text
        events_path = (
            tmp_path
            / runtime.config.session_id
            / "internal/events"
            / f"{runtime.config.session_id}.jsonl"
        )
        events = [json.loads(line) for line in events_path.read_text().splitlines()]
        checks = [
            event
            for event in events
            if event["event_type"]
            in {"prompt_director_output_matched", "prompt_director_output_mismatch"}
        ]
        assert [check["speaker_id"] for check in checks] == ["counselor"]
        check = checks[0]
        expected = "相談の入口に立てたことが、まず大切なのですね。"
        assert check["event_type"] == (
            "prompt_director_output_matched"
            if expected == actual_text
            else "prompt_director_output_mismatch"
        )
        assert check["details"]["expected_char_count"] == len(expected)
        assert check["details"]["actual_char_count"] == len(actual_text)
        assert actual_text not in json.dumps(check, ensure_ascii=False)
        assert not runtime._selected_prompt_director_texts

    asyncio.run(scenario())


def test_speculative_director_result_cannot_replace_selected_output_audit(tmp_path):
    class DistinctDirector(RecordingPromptDirector):
        async def create_directive(self, request):
            result = await super().create_directive(request)
            return result.model_copy(
                update={"response_example": request.current_objective}
            )

    async def scenario():
        runtime = ConversationRuntime(
            config=RuntimeConfig(session_id="director-speculative-audit"),
            prompt_director=DistinctDirector(),
            sessions_dir=tmp_path,
        )
        await runtime.logger.start()
        try:
            for objective, speculative in [
                ("採用された本文", False),
                ("未採用の別案", True),
            ]:
                await runtime._add_prompt_director_instruction(
                    turn_id=1,
                    speaker="counselor",
                    input_transcript="相談",
                    current_objective=objective,
                    response_target="client",
                    existing_instruction=None,
                    speculative=speculative,
                )
            assert (
                runtime._selected_prompt_director_texts[(1, "counselor")]
                == "採用された本文"
            )
        finally:
            await runtime._cancel_prompt_director_tasks()
            await runtime.logger.close()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    (
        "elapsed",
        "requested",
        "consents",
        "proposed",
        "pending",
        "speculative",
        "expected",
    ),
    [
        (899, [], ["client_a", "client_b"], True, False, False, None),
        (900, [], ["client_a", "client_b"], True, False, False, "agreed_end"),
        (950, [], ["client_a"], True, False, False, None),
        (950, [], ["client_a", "client_b"], False, False, False, None),
        (950, [], ["client_a", "client_b"], True, True, False, None),
        (100, ["client_a"], [], False, True, False, "client_requested_end"),
        (950, [], ["client_a", "client_b"], True, False, True, None),
    ],
)
def test_early_closing_requires_time_consent_and_actual_selection(
    tmp_path, elapsed, requested, consents, proposed, pending, speculative, expected
) -> None:
    class EndDirector(RecordingPromptDirector):
        async def create_directive(self, request):
            result = await super().create_directive(request)
            return result.model_copy(
                update={
                    "session_end_assessment": SessionEndAssessment(
                        explicit_end_request_client_ids=requested,
                        counselor_proposed_end=proposed,
                        consenting_client_ids=consents,
                        pending_question=pending,
                        reason="公開履歴の終了意思を確認した。",
                    )
                }
            )

    async def scenario():
        director = EndDirector()
        runtime = ConversationRuntime(
            config=RuntimeConfig(
                session_id="early_end_gate",
                closing_start_elapsed_seconds=1000,
                fixed_speaker_sequence=("counselor", "client_a", "client_b"),
                participants={
                    "counselor": ParticipantConfig("counselor", "counselor", "相談員"),
                    "client_a": ParticipantConfig("client_a", "client", "妻"),
                    "client_b": ParticipantConfig("client_b", "client", "夫"),
                },
            ),
            prompt_director=director,
            sessions_dir=tmp_path,
        )
        runtime._session_started_monotonic = time.monotonic() - elapsed
        await runtime.logger.start()
        try:
            await runtime._add_prompt_director_instruction(
                turn_id=3,
                speaker="counselor",
                input_transcript="今日はここまででよいです。",
                current_objective="継続",
                response_target=None,
                existing_instruction=None,
                speculative=speculative,
            )
            assert runtime._early_closing_reason == expected
            assert runtime.status.closing_started == (expected is not None)
            assert director.requests[0].session_end_context.allow_agreed_end == (
                elapsed >= 900
            )
        finally:
            await runtime._cancel_prompt_director_tasks()
            await runtime.logger.close()

    asyncio.run(scenario())


def test_explicit_end_request_gets_one_counselor_closing_and_actual_speech_check(
    tmp_path,
) -> None:
    class EndDirector(RecordingPromptDirector):
        def __init__(self):
            super().__init__()
            self.closing_requests = []

        async def create_directive(self, request):
            result = await super().create_directive(request)
            if request.session_end_context is not None:
                result = result.model_copy(
                    update={
                        "session_end_assessment": SessionEndAssessment(
                            explicit_end_request_client_ids=["client"],
                            counselor_proposed_end=False,
                            consenting_client_ids=[],
                            pending_question=False,
                            reason="本人が面接全体の終了を希望した。",
                        )
                    }
                )
            return result

        async def assess_closing(self, request):
            self.closing_requests.append(request)
            return ClosingAssessment(
                reply_speaker_ids=[], reason="実発話も終結している。"
            )

    async def scenario():
        director = EndDirector()
        runtime = ConversationRuntime(
            config=RuntimeConfig(
                session_id="early_closing_run",
                closing_start_elapsed_seconds=1000,
                force_stop_after_closing_turns=3,
                max_turns=10,
                speaker_selection_policy="fixed_round_robin",
            ),
            agents={
                "client": FakeAgent("client", "疲れたので今日はここで終わりたいです。"),
                "counselor": FakeAgent(
                    "counselor", "分かりました。今日はここまでにしましょう。"
                ),
            },
            prompt_director=director,
            sessions_dir=tmp_path,
        )
        turns = await asyncio.wait_for(runtime.run(), timeout=5)
        assert [turn.speaker for turn in turns] == ["client", "counselor"]
        assert runtime.status.phase is RuntimePhase.COMPLETED
        assert (
            director.closing_requests[0].public_history[-1].text
            == turns[-1].generated_text
        )
        events = [
            json.loads(line)
            for line in (
                tmp_path / "early_closing_run/internal/events/early_closing_run.jsonl"
            )
            .read_text()
            .splitlines()
        ]
        closing = [
            event for event in events if event["event_type"] == "closing_started"
        ]
        assert closing[0]["details"]["reason"] == "client_requested_end"
        assert closing[0]["details"]["elapsed_seconds"] < 900

    asyncio.run(scenario())


def test_runtime_closing_start_instructs_counselor_and_stops_after_closing_turns(
    tmp_path,
) -> None:
    async def scenario() -> None:
        counselor = FakeAgent("counselor", "カウンセラー生成:{input_transcript}")
        client = FakeAgent("client", "クライアント生成:{input_transcript}")
        runtime = ConversationRuntime(
            config=RuntimeConfig(
                session_id="session_closing_turn_stop_test",
                max_turns=10,
                closing_start_elapsed_seconds=0,
                force_stop_after_closing_turns=3,
                initial_client_transcript="初回相談です。",
            ),
            agents={"counselor": counselor, "client": client},
            sessions_dir=tmp_path,
        )

        turns = await runtime.run()

        assert runtime.status.phase is RuntimePhase.COMPLETED
        assert runtime.status.closing_started is True
        assert runtime.status.closing_started_turn_id == 1
        assert runtime.status.closing_count_started_turn_id == 1
        assert [turn.speaker for turn in turns] == [
            "client",
            "counselor",
            "client",
            "counselor",
        ]
        assert [turn.turn_id for turn in turns if turn.turn_id > 0] == [1, 2, 3]
        assert turns[0].generated_text == "クライアント生成:"
        assert turns[1].input_transcript == turns[0].generated_text
        assert "クロージング" in counselor.received_inputs[0]
        assert "クロージング" in counselor.received_inputs[1]
        assert "初回相談内容" in client.received_inputs[0]
        assert "クロージング" in client.received_inputs[1]
        assert "新しい話題を広げず" in client.received_inputs[1]

        event_path = (
            tmp_path
            / "session_closing_turn_stop_test"
            / "internal"
            / "events"
            / "session_closing_turn_stop_test.jsonl"
        )
        event_types = [
            json.loads(line)["event_type"]
            for line in event_path.read_text(encoding="utf-8").splitlines()
        ]
        assert "closing_started" in event_types

    asyncio.run(scenario())


@pytest.mark.parametrize("pending", [True, False])
def test_closing_checks_actual_speech_before_stopping(tmp_path, pending) -> None:
    class ClosingDirector(RecordingPromptDirector):
        def __init__(self):
            super().__init__()
            self.closing_requests = []

        async def assess_closing(self, request):
            self.closing_requests.append(request)
            return ClosingAssessment(
                reply_speaker_ids=(
                    ["client"] if pending and len(self.closing_requests) == 1 else []
                ),
                reason="実際の発話の確認。",
            )

    async def scenario():
        director = ClosingDirector()
        runtime = ConversationRuntime(
            config=RuntimeConfig(
                session_id="closing_actual_speech",
                closing_start_elapsed_seconds=0,
                force_stop_after_closing_turns=3,
                max_turns=10,
            ),
            agents={
                "counselor": FakeAgent(
                    "counselor", "対処をしなかったらどうなりそうですか。"
                ),
                "client": FakeAgent("client", "まだ分かりません。"),
            },
            prompt_director=director,
            sessions_dir=tmp_path,
        )
        turns = await runtime.run()
        assert runtime.status.phase is RuntimePhase.COMPLETED
        assert [turn.speaker for turn in turns] == (
            ["client", "counselor", "client", "counselor", "client", "counselor"]
            if pending
            else ["client", "counselor", "client", "counselor"]
        )
        # The actual utterance differs from the director's planned statement.
        assert director.closing_requests[0].public_history[-1].text == (
            "対処をしなかったらどうなりそうですか。"
        )
        if pending:
            assert (
                director.closing_requests[-1].public_history[-2].text
                == "まだ分かりません。"
            )
        events = [
            json.loads(line)
            for line in (
                tmp_path
                / "closing_actual_speech/internal/events/closing_actual_speech.jsonl"
            )
            .read_text()
            .splitlines()
        ]
        checks = [
            event
            for event in events
            if event["event_type"] == "closing_response_assessed"
        ]
        assert len(checks) == (2 if pending else 1)
        assert checks[0]["details"]["reply_speaker_ids"] == (
            ["client"] if pending else []
        )

    asyncio.run(scenario())


def test_closing_gives_named_clients_turns_and_keeps_extension_bounded(
    tmp_path,
) -> None:
    class ClosingDirector(RecordingPromptDirector):
        async def assess_closing(self, request):
            return ClosingAssessment(
                reply_speaker_ids=["client_b", "client_a"], reason="二人への質問。"
            )

    async def scenario():
        runtime = ConversationRuntime(
            config=RuntimeConfig(
                session_id="closing_bounded_replies",
                speaker_selection_policy="fixed_round_robin",
                closing_start_elapsed_seconds=0,
                force_stop_after_closing_turns=1,
                max_turns=10,
                # The ordinary sequence would choose client_a first and never
                # client_b. The assessed reply order must take precedence.
                fixed_speaker_sequence=("counselor", "client_a"),
                participants={
                    "counselor": ParticipantConfig("counselor", "counselor", "相談員"),
                    "client_a": ParticipantConfig(
                        "client_a", "client", "妻", initial_transcript="相談です。"
                    ),
                    "client_b": ParticipantConfig("client_b", "client", "夫"),
                },
            ),
            agents={
                "counselor": FakeAgent("counselor", "お二人はどう思いますか。"),
                "client_a": FakeAgent("client_a", "まだ分かりません。"),
                "client_b": FakeAgent("client_b", "今は答えたくありません。"),
            },
            prompt_director=ClosingDirector(),
            sessions_dir=tmp_path,
        )
        turns = await asyncio.wait_for(runtime.run(), timeout=5)
        assert [turn.speaker for turn in turns] == [
            "client_a",
            "counselor",
            "client_b",
            "client_a",
            "counselor",
        ]
        events = [
            json.loads(line)
            for line in (
                tmp_path
                / "closing_bounded_replies/internal/events/closing_bounded_replies.jsonl"
            )
            .read_text()
            .splitlines()
        ]
        assert any(
            event["event_type"] == "closing_reply_limit_reached" for event in events
        )

    asyncio.run(scenario())


def test_closing_assessment_can_be_cancelled_without_speaking_again(tmp_path) -> None:
    class WaitingDirector(RecordingPromptDirector):
        def __init__(self):
            super().__init__()
            self.started = asyncio.Event()
            self.cancelled = False

        async def assess_closing(self, request):
            self.started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                self.cancelled = True
                raise

    async def scenario():
        director = WaitingDirector()
        runtime = ConversationRuntime(
            config=RuntimeConfig(
                session_id="closing_cancel",
                closing_start_elapsed_seconds=0,
                force_stop_after_closing_turns=1,
            ),
            agents={
                "counselor": FakeAgent("counselor", "どう思いますか。"),
                "client": FakeAgent("client", "相談です。"),
            },
            prompt_director=director,
            sessions_dir=tmp_path,
        )
        task = asyncio.create_task(runtime.run())
        await asyncio.wait_for(director.started.wait(), timeout=5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=2)
        assert director.cancelled
        assert runtime.status.phase is RuntimePhase.STOPPED
        assert len(runtime.turns) == 2

    asyncio.run(scenario())


def test_closing_pending_question_waits_for_human_client(tmp_path) -> None:
    class ClosingDirector(RecordingPromptDirector):
        async def assess_closing(self, request):
            return ClosingAssessment(
                reply_speaker_ids=(
                    ["client"] if len(request.public_history) == 1 else []
                ),
                reason="最後の発話を確認した。",
            )

    async def scenario():
        runtime = ConversationRuntime(
            config=RuntimeConfig(
                session_id="closing_human_reply",
                interaction_mode="ai_counselor_human_client",
                initial_client_transcript="",
                speaker_selection_policy="fixed_round_robin",
                closing_start_elapsed_seconds=0,
                force_stop_after_closing_turns=1,
                participants={
                    "counselor": ParticipantConfig("counselor", "counselor", "相談員"),
                    "client": ParticipantConfig(
                        "client", "client", "相談者", actor_kind=ActorKind.HUMAN
                    ),
                },
            ),
            agents={"counselor": FakeAgent("counselor", "どう思いますか。")},
            prompt_director=ClosingDirector(),
            sessions_dir=tmp_path,
        )
        task = asyncio.create_task(runtime.run())
        await _wait_for_human_input(runtime)
        assert not task.done()
        await runtime.submit_human_turn(
            HumanTurnInput(
                session_id=runtime.config.session_id,
                text="今は分かりません。",
                recipient_ids=("counselor",),
            )
        )
        turns = await asyncio.wait_for(task, timeout=5)
        assert [turn.speaker for turn in turns] == ["counselor", "client", "counselor"]
        assert turns[1].stt_final_transcript == "今は分かりません。"

    asyncio.run(scenario())


def test_runtime_defers_closing_start_to_next_counselor_turn(tmp_path) -> None:
    async def scenario() -> None:
        counselor = VerySlowFakeAgent(
            "counselor",
            "カウンセラー生成:{input_transcript}",
        )
        client = FakeAgent("client", "クライアント生成:{input_transcript}")
        runtime = ConversationRuntime(
            config=RuntimeConfig(
                session_id="session_closing_defer_to_counselor_test",
                max_turns=10,
                closing_start_elapsed_seconds=0.05,
                force_stop_after_closing_turns=2,
                initial_client_transcript="初回相談です。",
            ),
            agents={"counselor": counselor, "client": client},
            sessions_dir=tmp_path,
        )

        turns = await runtime.run()

        assert runtime.status.phase is RuntimePhase.COMPLETED
        assert runtime.status.closing_started is True
        assert runtime.status.closing_started_turn_id == 3
        assert runtime.status.closing_count_started_turn_id == 3
        assert [turn.speaker for turn in turns] == [
            "client",
            "counselor",
            "client",
            "counselor",
            "client",
            "counselor",
        ]
        assert "【クロージング指示】" not in counselor.received_inputs[0]
        assert "【クロージング指示】" in counselor.received_inputs[1]
        assert "【クロージング指示】" in counselor.received_inputs[2]
        assert "クロージング" not in client.received_inputs[0]
        assert "クロージング" not in client.received_inputs[1]
        assert "クロージング" in client.received_inputs[2]
        assert "新しい話題を広げず" in client.received_inputs[2]

    asyncio.run(scenario())


def test_runtime_closing_due_can_continue_past_pre_closing_max_turns(tmp_path) -> None:
    async def scenario() -> None:
        counselor = SlowFakeAgent("counselor", "カウンセラー生成:{input_transcript}")
        client = FakeAgent("client", "クライアント生成:{input_transcript}")
        runtime = ConversationRuntime(
            config=RuntimeConfig(
                session_id="session_closing_after_turn_cap_test",
                max_turns=1,
                closing_start_elapsed_seconds=0.03,
                force_stop_after_closing_turns=2,
                initial_client_transcript="初回相談です。",
            ),
            agents={"counselor": counselor, "client": client},
            sessions_dir=tmp_path,
        )

        turns = await runtime.run()

        assert runtime.status.phase is RuntimePhase.COMPLETED
        assert runtime.status.closing_started is True
        assert runtime.status.closing_started_turn_id == 3
        assert runtime.status.closing_count_started_turn_id == 3
        assert [turn.speaker for turn in turns] == [
            "client",
            "counselor",
            "client",
            "counselor",
            "client",
            "counselor",
        ]
        assert "クロージング" in counselor.received_inputs[1]
        assert "クロージング" in counselor.received_inputs[2]
        assert "クロージング" in client.received_inputs[2]

    asyncio.run(scenario())


def test_fake_runtime_logs_latency_metrics_and_audio_delivery_mode(tmp_path) -> None:
    async def scenario() -> None:
        runtime = ConversationRuntime(
            config=RuntimeConfig(
                session_id="session_runtime_metrics_test",
                max_turns=1,
                audio_delivery_mode=AudioDeliveryMode.REAL_TIME,
            ),
            sessions_dir=tmp_path,
        )

        await runtime.run()

        session_dir = tmp_path / "session_runtime_metrics_test" / "internal"
        metric_path = session_dir / "metrics" / "session_runtime_metrics_test.jsonl"
        metrics = [
            json.loads(line)
            for line in metric_path.read_text(encoding="utf-8").splitlines()
        ]
        metric_names = {metric["metric_name"] for metric in metrics}
        assert "llm_first_token_latency_ms" in metric_names
        assert "tts_first_audio_chunk_latency_ms" in metric_names
        assert "tts_stream_done_to_stt_commit_sent_ms" not in metric_names
        assert "stt_commit_to_final_ms" not in metric_names
        assert all(metric["unit"] == "ms" for metric in metrics)
        assert all(metric["value"] >= 0 for metric in metrics)
        assert {metric["details"]["audio_delivery_mode"] for metric in metrics} == {
            "real_time"
        }

        event_path = session_dir / "events" / "session_runtime_metrics_test.jsonl"
        events = [
            json.loads(line)
            for line in event_path.read_text(encoding="utf-8").splitlines()
        ]
        tts_first_audio = next(
            event for event in events if event["event_type"] == "tts_first_audio_chunk"
        )
        assert tts_first_audio["details"]["audio_delivery_mode"] == "real_time"
        assert tts_first_audio["details"]["latency_ms"] >= 0
        event_types = {event["event_type"] for event in events}
        assert "generated_final" in event_types
        assert "stt_commit_sent" not in event_types

    asyncio.run(scenario())


def test_fake_runtime_logs_previous_turn_to_next_first_audio_metric(tmp_path) -> None:
    async def scenario() -> None:
        runtime = ConversationRuntime(
            config=RuntimeConfig(
                session_id="session_turn_interval_metrics_test",
                max_turns=2,
            ),
            sessions_dir=tmp_path,
        )

        await runtime.run()

        metric_path = (
            tmp_path
            / "session_turn_interval_metrics_test"
            / "internal"
            / "metrics"
            / "session_turn_interval_metrics_test.jsonl"
        )
        metrics = [
            json.loads(line)
            for line in metric_path.read_text(encoding="utf-8").splitlines()
        ]
        interval_metric = next(
            metric
            for metric in metrics
            if metric["metric_name"] == "previous_stt_final_to_next_first_audio_ms"
        )
        llm_request_metric = next(
            metric
            for metric in metrics
            if metric["metric_name"] == "previous_stt_final_to_next_llm_request_ms"
        )

        assert interval_metric["turn_id"] == 1
        assert interval_metric["details"]["previous_turn_id"] == 0
        assert interval_metric["value"] >= 0
        assert llm_request_metric["turn_id"] == 1
        assert llm_request_metric["details"]["previous_turn_id"] == 0
        assert llm_request_metric["value"] >= 0

    asyncio.run(scenario())


def test_runtime_closes_reusable_stt_after_completion(tmp_path) -> None:
    class ClosableQueueStt:
        def __init__(self) -> None:
            self.close_calls = 0

        async def transcribe_from_queue(self, *, session_id, turn_id, speaker, queue):
            chunks = [
                chunk
                async for chunk in iter_turn_audio_chunks(
                    queue,
                    session_id=session_id,
                    turn_id=turn_id,
                    speaker=speaker,
                )
            ]
            return await FakeStreamingSTT().transcribe(
                session_id=session_id,
                turn_id=turn_id,
                speaker=speaker,
                chunks=chunks,
            )

        async def close(self) -> None:
            self.close_calls += 1

    async def scenario() -> None:
        stt = ClosableQueueStt()
        runtime = ConversationRuntime(
            config=RuntimeConfig(
                session_id="session_runtime_close_stt_test",
                max_turns=2,
            ),
            stt=stt,
            sessions_dir=tmp_path,
        )

        await runtime.run()

        assert stt.close_calls == 1

    asyncio.run(scenario())


def test_runtime_closes_realtime_speech_agents_after_completion(tmp_path) -> None:
    class ClosableAgent(FakeAgent):
        def __init__(self, speaker: str, response_template: str) -> None:
            super().__init__(speaker, response_template)
            self.close_calls = 0

        async def close(self) -> None:
            self.close_calls += 1

    async def scenario() -> None:
        counselor = ClosableAgent("counselor", "カウンセラー生成:{input_transcript}")
        client = ClosableAgent("client", "クライアント生成:{input_transcript}")
        runtime = ConversationRuntime(
            config=RuntimeConfig(
                session_id="session_runtime_close_agents_test",
                max_turns=2,
            ),
            agents={"counselor": counselor, "client": client},
            sessions_dir=tmp_path,
        )

        await runtime.run()

        assert counselor.close_calls == 1
        assert client.close_calls == 1

    asyncio.run(scenario())


def test_runtime_ignores_stt_partial_prefetch_with_generated_transcripts(
    tmp_path,
) -> None:
    async def scenario() -> None:
        counselor = _RecordingStreamingAgent("counselor")
        client = _RecordingStreamingAgent("client")
        runtime = ConversationRuntime(
            config=RuntimeConfig(
                session_id="session_runtime_prefetch_adopt_test",
                max_turns=2,
                initial_client_transcript="初回相談です。",
                speaker_selection_policy="fixed_round_robin",
                stt_partial_prefetch=True,
                stt_partial_prefetch_min_chars=4,
            ),
            agents={"counselor": counselor, "client": client},
            stt=_ScriptedObservableStt(
                first_partial="相談したい",
                first_final="相談したいです。",
            ),
            sessions_dir=tmp_path,
        )

        turns = await runtime.run()

        assert counselor.received_inputs == [turns[0].generated_text]
        assert client.received_inputs == [
            "",
            f"counselor generated from {turns[0].generated_text}",
        ]
        assert "初回相談内容" in client.received_additional_instructions[0]
        assert "初回相談です。" in client.received_additional_instructions[0]
        assert turns[2].input_transcript == (
            f"counselor generated from {turns[0].generated_text}"
        )
        assert turns[2].provisional_input_transcript is None
        assert turns[2].used_provisional_generation is False
        assert (
            turns[2].generated_text
            == f"client generated from counselor generated from {turns[0].generated_text}"
        )

        event_path = (
            tmp_path
            / "session_runtime_prefetch_adopt_test"
            / "internal"
            / "events"
            / "session_runtime_prefetch_adopt_test.jsonl"
        )
        event_types = [
            json.loads(line)["event_type"]
            for line in event_path.read_text(encoding="utf-8").splitlines()
        ]
        assert "provisional_llm_request_started" not in event_types
        assert "provisional_llm_adopted" not in event_types

    asyncio.run(scenario())


def test_runtime_does_not_discard_provisional_generation_without_stt_partials(
    tmp_path,
) -> None:
    async def scenario() -> None:
        counselor = _RecordingStreamingAgent("counselor")
        client = _RecordingStreamingAgent("client")
        runtime = ConversationRuntime(
            config=RuntimeConfig(
                session_id="session_runtime_prefetch_discard_test",
                max_turns=2,
                initial_client_transcript="初回相談です。",
                speaker_selection_policy="fixed_round_robin",
                stt_partial_prefetch=True,
                stt_partial_prefetch_min_chars=4,
            ),
            agents={"counselor": counselor, "client": client},
            stt=_ScriptedObservableStt(
                first_partial="相談したい",
                first_final="別の話です。",
            ),
            sessions_dir=tmp_path,
        )

        turns = await runtime.run()

        assert client.received_inputs == [
            "",
            f"counselor generated from {turns[0].generated_text}",
        ]
        assert turns[2].input_transcript == (
            f"counselor generated from {turns[0].generated_text}"
        )
        assert turns[2].used_provisional_generation is False
        assert turns[2].provisional_input_transcript is None
        assert (
            turns[2].generated_text
            == f"client generated from counselor generated from {turns[0].generated_text}"
        )

        event_path = (
            tmp_path
            / "session_runtime_prefetch_discard_test"
            / "internal"
            / "events"
            / "session_runtime_prefetch_discard_test.jsonl"
        )
        discarded = [
            json.loads(line)
            for line in event_path.read_text(encoding="utf-8").splitlines()
            if json.loads(line)["event_type"] == "provisional_llm_discarded"
        ]
        assert discarded == []

    asyncio.run(scenario())


class _RecordingStreamingAgent:
    def __init__(self, speaker: str) -> None:
        self.speaker = speaker
        self.received_inputs: list[str] = []
        self.received_additional_instructions: list[str | None] = []

    async def stream_generate(
        self,
        *,
        input_transcript: str,
        turn_id: int,
        additional_instruction: str | None = None,
    ):
        _ = turn_id
        self.received_inputs.append(input_transcript)
        self.received_additional_instructions.append(additional_instruction)
        await asyncio.sleep(0)
        yield f"{self.speaker} generated from {input_transcript}"


class _ScriptedObservableStt:
    def __init__(self, *, first_partial: str, first_final: str) -> None:
        self.first_partial = first_partial
        self.first_final = first_final

    async def transcribe_from_queue_observed(
        self,
        *,
        session_id,
        turn_id,
        speaker,
        queue,
        on_transcript,
    ):
        _ = [
            chunk
            async for chunk in iter_turn_audio_chunks(
                queue,
                session_id=session_id,
                turn_id=turn_id,
                speaker=speaker,
            )
        ]
        if turn_id == 1:
            partial = TranscriptEvent(
                session_id=session_id,
                turn_id=turn_id,
                speaker=speaker,
                transcript_type="partial",
                text=self.first_partial,
            )
            await on_transcript(partial)
            await asyncio.sleep(0)
            final = TranscriptEvent(
                session_id=session_id,
                turn_id=turn_id,
                speaker=speaker,
                transcript_type="final",
                text=self.first_final,
            )
            await on_transcript(final)
            return [partial], final

        final = TranscriptEvent(
            session_id=session_id,
            turn_id=turn_id,
            speaker=speaker,
            transcript_type="final",
            text=f"turn {turn_id} final",
        )
        await on_transcript(final)
        return [], final

    async def transcribe_from_queue(self, *, session_id, turn_id, speaker, queue):
        async def ignore_transcript(_event):
            return None

        return await self.transcribe_from_queue_observed(
            session_id=session_id,
            turn_id=turn_id,
            speaker=speaker,
            queue=queue,
            on_transcript=ignore_transcript,
        )


class _FinalTextStt:
    def __init__(
        self,
        final_text: str,
        *,
        partial_texts: tuple[str, ...] = (),
    ) -> None:
        self.final_text = final_text
        self.partial_texts = partial_texts
        self.received_chunks: list[AudioChunk] = []

    async def transcribe(
        self,
        *,
        session_id,
        turn_id,
        speaker,
        chunks,
    ):
        self.received_chunks.extend(list(chunks))
        partials = [
            TranscriptEvent(
                session_id=session_id,
                turn_id=turn_id,
                speaker=speaker,
                transcript_type="partial",
                text=partial_text,
            )
            for partial_text in self.partial_texts
        ]
        final = TranscriptEvent(
            session_id=session_id,
            turn_id=turn_id,
            speaker=speaker,
            transcript_type="final",
            text=self.final_text,
        )
        return partials, final

    async def transcribe_from_queue_observed(
        self,
        *,
        session_id,
        turn_id,
        speaker,
        queue,
        on_transcript,
    ):
        chunks: list[AudioChunk] = []
        while True:
            item = await queue.get()
            if isinstance(item, AudioChunk):
                chunks.append(item)
                if chunks and self.partial_texts:
                    partial = TranscriptEvent(
                        session_id=session_id,
                        turn_id=turn_id,
                        speaker=speaker,
                        transcript_type="partial",
                        text=self.partial_texts[
                            min(len(chunks), len(self.partial_texts)) - 1
                        ],
                    )
                    await on_transcript(partial)
                continue
            if isinstance(item, EndOfAudio):
                break
            raise AssertionError(f"unexpected STT queue item: {item!r}")
        self.received_chunks.extend(chunks)
        final = TranscriptEvent(
            session_id=session_id,
            turn_id=turn_id,
            speaker=speaker,
            transcript_type="final",
            text=self.final_text,
        )
        await on_transcript(final)
        return [], final


class _FailingStt:
    async def transcribe(self, **_kwargs):
        raise AssertionError("STT should not be called for direct Realtime audio input")


class _RealtimeAudioInputAgent(FakeAgent):
    def __init__(
        self,
        speaker: str,
        *,
        input_transcript: str,
        response_text: str,
    ) -> None:
        super().__init__(speaker, response_text)
        self.input_transcript = input_transcript
        self.response_text = response_text
        self.received_audio_inputs: list[bytes] = []

    async def stream_audio_response(self, **_kwargs):
        raise AssertionError("text Realtime path should not be used")
        yield

    async def stream_audio_response_from_audio(
        self,
        *,
        session_id,
        turn_id,
        speaker,
        input_audio,
        sample_rate,
        sample_width_bits,
        channels,
        delivery_mode,
        additional_instruction=None,
    ):
        _ = additional_instruction
        self.received_audio_inputs.append(input_audio)
        yield RealtimeSpeechEvent(input_transcript_completed=self.input_transcript)
        yield RealtimeSpeechEvent(text_delta=self.response_text)
        yield RealtimeSpeechEvent(
            audio_chunk=AudioChunk(
                session_id=session_id,
                turn_id=turn_id,
                speaker=speaker,
                chunk_index=0,
                pcm=b"\x02\x00" * 120,
                sample_rate=sample_rate,
                sample_width_bits=sample_width_bits,
                channels=channels,
                duration_ms=5,
                delivery_mode=delivery_mode,
            )
        )


async def _wait_for_human_input(runtime: ConversationRuntime) -> None:
    for _ in range(500):
        status = runtime.status
        if status.awaiting_human_input:
            return
        if status.phase in {RuntimePhase.COMPLETED, RuntimePhase.ERROR}:
            raise AssertionError(
                "runtime stopped before human input wait: "
                f"phase={status.phase.value}, "
                f"turns={[turn.speaker for turn in runtime.turns]}"
            )
        await asyncio.sleep(0.01)
    raise AssertionError(
        "runtime did not enter human input wait state: "
        f"turns={[turn.speaker for turn in runtime.turns]}"
    )


def test_human_counselor_runtime_waits_for_user_first_turn(tmp_path) -> None:
    async def scenario() -> None:
        client_agent = FakeAgent("client", "クライアント応答:{input_transcript}")
        director = RecordingPromptDirector()
        runtime = ConversationRuntime(
            config=RuntimeConfig(
                session_id="session-human-text",
                interaction_mode="human_counselor_ai_client",
                participant_mode="one_client",
                max_turns=2,
                fixed_speaker_sequence=("counselor", "client"),
                speaker_selection_policy="fixed_round_robin",
                initial_client_transcript="これは先に話さない",
                participants={
                    "counselor": ParticipantConfig(
                        "counselor",
                        "counselor",
                        "カウンセラー",
                        actor_kind=ActorKind.HUMAN,
                    ),
                    "client": ParticipantConfig(
                        "client",
                        "client",
                        "クライアント",
                        initial_transcript="これも先に話さない",
                        prompt_source="相談者として自分の経験を話す。",
                        public_profile_source="人間カウンセラーに相談する本人。",
                    ),
                },
            ),
            agents={"client": client_agent},
            prompt_director=director,
            sessions_dir=tmp_path,
        )
        run_task = asyncio.create_task(runtime.run())
        await _wait_for_human_input(runtime)

        result = await runtime.submit_human_turn(
            HumanTurnInput(
                session_id="session-human-text",
                text="今日はどんなことを相談したいですか。",
                recipient_ids=("client",),
            )
        )
        assert result["accepted"] is True
        assert result["turn_id"] == 1

        turns = await asyncio.wait_for(run_task, timeout=2.0)

        assert [turn.speaker for turn in turns] == ["counselor", "client"]
        assert turns[0].stt_final_transcript == "今日はどんなことを相談したいですか。"
        assert turns[0].generated_text == ""
        assert turns[1].input_transcript == "今日はどんなことを相談したいですか。"
        assert client_agent.received_inputs[0].startswith(
            "今日はどんなことを相談したいですか。"
        )
        assert "初回相談内容" in client_agent.received_inputs[0]
        assert "これも先に話さない" in client_agent.received_inputs[0]
        assert "＜原文の条件と今回の適用判断＞" not in client_agent.received_inputs[0]
        assert "＜今回発話する本文＞" not in client_agent.received_inputs[0]
        assert not director.requests
        assert runtime.status.phase is RuntimePhase.COMPLETED

    asyncio.run(scenario())


def test_human_counselor_first_ai_client_response_gets_initial_topic_once(
    tmp_path,
) -> None:
    async def scenario() -> None:
        initial_topic = "今日は中2の娘の対応について相談に来ました"
        client_agent = FakeAgent("client", "クライアント応答:{input_transcript}")
        runtime = ConversationRuntime(
            config=RuntimeConfig(
                session_id="session-human-initial-topic",
                interaction_mode="human_counselor_ai_client",
                participant_mode="one_client",
                max_turns=4,
                fixed_speaker_sequence=("counselor", "client"),
                speaker_selection_policy="fixed_round_robin",
                participants={
                    "counselor": ParticipantConfig(
                        "counselor",
                        "counselor",
                        "カウンセラー",
                        actor_kind=ActorKind.HUMAN,
                    ),
                    "client": ParticipantConfig(
                        "client",
                        "client",
                        "クライアント",
                        initial_transcript=initial_topic,
                    ),
                },
            ),
            agents={"client": client_agent},
            sessions_dir=tmp_path,
        )
        run_task = asyncio.create_task(runtime.run())
        await _wait_for_human_input(runtime)

        await runtime.submit_human_turn(
            HumanTurnInput(
                session_id="session-human-initial-topic",
                text="こんにちは。今日はよろしくお願いします。",
                recipient_ids=("client",),
            )
        )
        await _wait_for_human_input(runtime)

        await runtime.submit_human_turn(
            HumanTurnInput(
                session_id="session-human-initial-topic",
                text="もう少し今の状況を聞かせてください。",
                recipient_ids=("client",),
            )
        )
        turns = await asyncio.wait_for(run_task, timeout=2.0)

        assert [turn.speaker for turn in turns] == [
            "counselor",
            "client",
            "counselor",
            "client",
        ]
        assert len(client_agent.received_inputs) == 2
        assert "初回相談内容" in client_agent.received_inputs[0]
        assert initial_topic in client_agent.received_inputs[0]
        assert "カウンセラーの挨拶や問いかけに自然に応答" in (
            client_agent.received_inputs[0]
        )
        assert "原文と同じ一文をそのまま出力せず" in (client_agent.received_inputs[0])
        assert "初回相談内容" not in client_agent.received_inputs[1]
        assert initial_topic not in client_agent.received_inputs[1]

    asyncio.run(scenario())


def test_human_counselor_first_ai_response_uses_client_with_initial_topic(
    tmp_path,
) -> None:
    async def scenario() -> None:
        initial_topic = "Aの初回相談内容です。"
        client_a = FakeAgent(
            "client_a",
            "A応答:{input_transcript}",
            timing_decisions=[
                {"agent_id": "client_a", "action": "WAIT"},
            ],
        )
        client_b = FakeAgent(
            "client_b",
            "B応答:{input_transcript}",
            timing_decisions=[
                {"agent_id": "client_b", "action": "REQUEST_MAIN_FLOOR"},
            ],
        )
        runtime = ConversationRuntime(
            config=RuntimeConfig(
                session_id="session-human-initial-topic-holder",
                interaction_mode="human_counselor_ai_client",
                participant_mode="two_clients",
                max_turns=2,
                fixed_speaker_sequence=(
                    "counselor",
                    "client_b",
                    "client_a",
                ),
                speaker_selection_policy="turn_boundary_timing",
                participants={
                    "counselor": ParticipantConfig(
                        "counselor",
                        "counselor",
                        "カウンセラー",
                        actor_kind=ActorKind.HUMAN,
                    ),
                    "client_a": ParticipantConfig(
                        "client_a",
                        "client",
                        "クライアントA",
                        initial_transcript=initial_topic,
                    ),
                    "client_b": ParticipantConfig(
                        "client_b",
                        "client",
                        "クライアントB",
                        initial_transcript="",
                    ),
                },
            ),
            agents={"client_a": client_a, "client_b": client_b},
            sessions_dir=tmp_path,
        )
        run_task = asyncio.create_task(runtime.run())
        await _wait_for_human_input(runtime)

        await runtime.submit_human_turn(
            HumanTurnInput(
                session_id="session-human-initial-topic-holder",
                text="まず、お話を聞かせてください。",
                recipient_ids=("client_a", "client_b"),
            )
        )
        turns = await asyncio.wait_for(run_task, timeout=2.0)

        assert [turn.speaker for turn in turns] == ["counselor", "client_a"]
        assert "初回相談内容" in client_a.received_inputs[0]
        assert initial_topic in client_a.received_inputs[0]
        assert client_b.received_inputs == []

    asyncio.run(scenario())


def test_human_counselor_two_client_mode_returns_to_user_when_clients_wait(
    tmp_path,
) -> None:
    async def scenario() -> None:
        client_a = FakeAgent(
            "client_a",
            "A応答:{input_transcript}",
            timing_decisions=[
                {"agent_id": "client_a", "action": "WAIT"},
                {"agent_id": "client_a", "action": "WAIT"},
            ],
        )
        client_b = FakeAgent(
            "client_b",
            "B応答:{input_transcript}",
            timing_decisions=[{"agent_id": "client_b", "action": "REQUEST_MAIN_FLOOR"}],
        )
        runtime = ConversationRuntime(
            config=RuntimeConfig(
                session_id="session-human-two-client-wait",
                interaction_mode="human_counselor_ai_client",
                participant_mode="two_clients",
                max_turns=3,
                fixed_speaker_sequence=(
                    "counselor",
                    "client_b",
                    "client_a",
                    "counselor",
                ),
                speaker_selection_policy="turn_boundary_timing",
                turn_taking_max_reconsider_rounds=0,
                participants={
                    "counselor": ParticipantConfig(
                        "counselor",
                        "counselor",
                        "カウンセラー",
                        actor_kind=ActorKind.HUMAN,
                    ),
                    "client_a": ParticipantConfig(
                        "client_a",
                        "client",
                        "クライアントA",
                    ),
                    "client_b": ParticipantConfig(
                        "client_b",
                        "client",
                        "クライアントB",
                    ),
                },
            ),
            agents={"client_a": client_a, "client_b": client_b},
            sessions_dir=tmp_path,
        )
        run_task = asyncio.create_task(runtime.run())
        await _wait_for_human_input(runtime)

        await runtime.submit_human_turn(
            HumanTurnInput(
                session_id="session-human-two-client-wait",
                text="今の状況を教えてください。",
                recipient_ids=("client_a", "client_b"),
            )
        )
        await _wait_for_human_input(runtime)
        assert [turn.speaker for turn in runtime.turns] == ["counselor", "client_b"]
        assert runtime.status.current_speaker == "counselor"

        await runtime.submit_human_turn(
            HumanTurnInput(
                session_id="session-human-two-client-wait",
                text="少し整理しましょう。",
                recipient_ids=("client_a", "client_b"),
            )
        )
        turns = await asyncio.wait_for(run_task, timeout=2.0)

        assert [turn.speaker for turn in turns] == [
            "counselor",
            "client_b",
            "counselor",
        ]

    asyncio.run(scenario())


def test_human_counselor_two_client_mode_waits_for_addressed_client(
    tmp_path,
) -> None:
    async def scenario() -> None:
        client_a = SlowTimingFakeAgent(
            "client_a",
            "A応答:{input_transcript}",
            timing_decisions=[
                {
                    "agent_id": "client_a",
                    "action": "REQUEST_MAIN_FLOOR",
                    "explicitly_addressed": True,
                }
            ],
        )
        client_b = SlowTimingFakeAgent("client_b", "B応答:{input_transcript}")
        runtime = ConversationRuntime(
            config=RuntimeConfig(
                session_id="session-human-two-client-slow-addressed",
                interaction_mode="human_counselor_ai_client",
                participant_mode="two_clients",
                max_turns=2,
                fixed_speaker_sequence=(
                    "counselor",
                    "client_b",
                    "client_a",
                    "counselor",
                ),
                speaker_selection_policy="turn_boundary_timing",
                turn_taking_decision_timeout_ms=1,
                turn_taking_max_reconsider_rounds=0,
                participants={
                    "counselor": ParticipantConfig(
                        "counselor",
                        "counselor",
                        "カウンセラー",
                        actor_kind=ActorKind.HUMAN,
                    ),
                    "client_a": ParticipantConfig(
                        "client_a",
                        "client",
                        "クライアントA",
                    ),
                    "client_b": ParticipantConfig(
                        "client_b",
                        "client",
                        "クライアントB",
                    ),
                },
            ),
            agents={"client_a": client_a, "client_b": client_b},
            sessions_dir=tmp_path,
        )
        run_task = asyncio.create_task(runtime.run())
        await _wait_for_human_input(runtime)

        await runtime.submit_human_turn(
            HumanTurnInput(
                session_id="session-human-two-client-slow-addressed",
                text="クライアントAさん、今の気持ちを聞かせてください。",
                recipient_ids=("client_a", "client_b"),
            )
        )
        turns = await asyncio.wait_for(run_task, timeout=2.0)

        assert [turn.speaker for turn in turns] == ["counselor", "client_a"]

    asyncio.run(scenario())


def test_human_counselor_two_client_mode_prioritizes_explicitly_addressed_client(
    tmp_path,
) -> None:
    async def scenario() -> None:
        client_a = FakeAgent(
            "client_a",
            "妻応答:{input_transcript}",
            timing_decisions=[
                {
                    "agent_id": "client_a",
                    "action": "WAIT",
                    "target": "counselor",
                    "explicitly_addressed": True,
                    "preferred_timing": "short_hold",
                    "urgency": 0.25,
                }
            ],
        )
        client_b = FakeAgent(
            "client_b",
            "夫応答:{input_transcript}",
            timing_decisions=[
                {
                    "agent_id": "client_b",
                    "action": "REQUEST_MAIN_FLOOR",
                    "target": "counselor",
                    "explicitly_addressed": False,
                    "preferred_timing": "natural_pause",
                    "urgency": 0.5,
                }
            ],
        )
        runtime = ConversationRuntime(
            config=RuntimeConfig(
                session_id="session-human-explicit-address-client",
                interaction_mode="human_counselor_ai_client",
                participant_mode="two_clients",
                max_turns=2,
                fixed_speaker_sequence=(
                    "counselor",
                    "client_b",
                    "client_a",
                    "counselor",
                ),
                speaker_selection_policy="turn_boundary_timing",
                turn_taking_max_reconsider_rounds=0,
                participants={
                    "counselor": ParticipantConfig(
                        "counselor",
                        "counselor",
                        "カウンセラー",
                        actor_kind=ActorKind.HUMAN,
                    ),
                    "client_a": ParticipantConfig(
                        "client_a",
                        "client",
                        "妻",
                    ),
                    "client_b": ParticipantConfig(
                        "client_b",
                        "client",
                        "夫",
                    ),
                },
            ),
            agents={"client_a": client_a, "client_b": client_b},
            sessions_dir=tmp_path,
        )
        run_task = asyncio.create_task(runtime.run())
        await _wait_for_human_input(runtime)

        await runtime.submit_human_turn(
            HumanTurnInput(
                session_id="session-human-explicit-address-client",
                text="奥さんはどう思われますか？",
                recipient_ids=("client_a", "client_b"),
            )
        )
        turns = await asyncio.wait_for(run_task, timeout=2.0)

        assert [turn.speaker for turn in turns] == ["counselor", "client_a"]

        event_path = (
            tmp_path
            / "session-human-explicit-address-client"
            / "internal"
            / "events"
            / "session-human-explicit-address-client.jsonl"
        )
        events = [
            json.loads(line)
            for line in event_path.read_text(encoding="utf-8").splitlines()
        ]
        floor_event = next(
            event
            for event in events
            if event["details"].get("reason") == "explicitly_addressed_participant"
        )

        assert floor_event["speaker_id"] == "client_a"
        assert floor_event["details"]["reason"] == "explicitly_addressed_participant"
        assert floor_event["details"]["granted_agent_id"] == "client_a"

    asyncio.run(scenario())


def test_human_counselor_two_client_mode_allows_urgent_reply_to_override_addressing(
    tmp_path,
) -> None:
    async def scenario() -> None:
        client_a = FakeAgent(
            "client_a",
            "妻応答:{input_transcript}",
            timing_decisions=[
                {
                    "agent_id": "client_a",
                    "action": "WAIT",
                    "target": "counselor",
                    "explicitly_addressed": True,
                    "preferred_timing": "short_hold",
                    "urgency": 0.25,
                }
            ],
        )
        client_b = FakeAgent(
            "client_b",
            "夫応答:{input_transcript}",
            timing_decisions=[
                {
                    "agent_id": "client_b",
                    "action": "REQUEST_MAIN_FLOOR",
                    "target": "counselor",
                    "explicitly_addressed": False,
                    "preferred_timing": "immediate",
                    "urgency": 0.95,
                }
            ],
        )
        runtime = ConversationRuntime(
            config=RuntimeConfig(
                session_id="session-human-urgent-reply-overrides-addressing",
                interaction_mode="human_counselor_ai_client",
                participant_mode="two_clients",
                max_turns=2,
                fixed_speaker_sequence=(
                    "counselor",
                    "client_a",
                    "client_b",
                    "counselor",
                ),
                speaker_selection_policy="turn_boundary_timing",
                turn_taking_max_reconsider_rounds=0,
                participants={
                    "counselor": ParticipantConfig(
                        "counselor",
                        "counselor",
                        "カウンセラー",
                        actor_kind=ActorKind.HUMAN,
                    ),
                    "client_a": ParticipantConfig(
                        "client_a",
                        "client",
                        "妻",
                    ),
                    "client_b": ParticipantConfig(
                        "client_b",
                        "client",
                        "夫",
                    ),
                },
            ),
            agents={"client_a": client_a, "client_b": client_b},
            sessions_dir=tmp_path,
        )
        run_task = asyncio.create_task(runtime.run())
        await _wait_for_human_input(runtime)

        await runtime.submit_human_turn(
            HumanTurnInput(
                session_id="session-human-urgent-reply-overrides-addressing",
                text="奥さんはどう思われますか？",
                recipient_ids=("client_a", "client_b"),
            )
        )
        turns = await asyncio.wait_for(run_task, timeout=2.0)

        assert [turn.speaker for turn in turns] == ["counselor", "client_b"]

    asyncio.run(scenario())


def test_human_counselor_two_client_mode_dampens_non_addressed_reply(
    tmp_path,
) -> None:
    async def scenario() -> None:
        client_a = FakeAgent(
            "client_a",
            "妻応答:{input_transcript}",
            timing_decisions=[
                {
                    "agent_id": "client_a",
                    "action": "WAIT",
                    "target": "counselor",
                    "explicitly_addressed": True,
                    "preferred_timing": "short_hold",
                    "urgency": 0.25,
                }
            ],
        )
        client_b = FakeAgent(
            "client_b",
            "夫応答:{input_transcript}",
            timing_decisions=[
                {
                    "agent_id": "client_b",
                    "action": "REQUEST_MAIN_FLOOR",
                    "target": "counselor",
                    "explicitly_addressed": False,
                    "preferred_timing": "immediate",
                    "urgency": 0.90,
                }
            ],
        )
        runtime = ConversationRuntime(
            config=RuntimeConfig(
                session_id="session-human-dampen-non-addressed-reply",
                interaction_mode="human_counselor_ai_client",
                participant_mode="two_clients",
                max_turns=2,
                fixed_speaker_sequence=(
                    "counselor",
                    "client_b",
                    "client_a",
                    "counselor",
                ),
                speaker_selection_policy="turn_boundary_timing",
                turn_taking_max_reconsider_rounds=0,
                participants={
                    "counselor": ParticipantConfig(
                        "counselor",
                        "counselor",
                        "カウンセラー",
                        actor_kind=ActorKind.HUMAN,
                    ),
                    "client_a": ParticipantConfig(
                        "client_a",
                        "client",
                        "妻",
                    ),
                    "client_b": ParticipantConfig(
                        "client_b",
                        "client",
                        "夫",
                    ),
                },
            ),
            agents={"client_a": client_a, "client_b": client_b},
            sessions_dir=tmp_path,
        )
        run_task = asyncio.create_task(runtime.run())
        await _wait_for_human_input(runtime)

        await runtime.submit_human_turn(
            HumanTurnInput(
                session_id="session-human-dampen-non-addressed-reply",
                text="奥さんはどう思われますか？",
                recipient_ids=("client_a", "client_b"),
            )
        )
        turns = await asyncio.wait_for(run_task, timeout=2.0)

        assert [turn.speaker for turn in turns] == ["counselor", "client_a"]

    asyncio.run(scenario())


def test_human_counselor_two_client_mode_prevents_consecutive_user_turns(
    tmp_path,
) -> None:
    async def scenario() -> None:
        client_a = FakeAgent(
            "client_a",
            "A応答:{input_transcript}",
            timing_decisions=[{"agent_id": "client_a", "action": "REQUEST_MAIN_FLOOR"}],
        )
        client_b = FakeAgent(
            "client_b",
            "B応答:{input_transcript}",
            timing_decisions=[{"agent_id": "client_b", "action": "WAIT"}],
        )
        runtime = ConversationRuntime(
            config=RuntimeConfig(
                session_id="session-human-two-client-no-consecutive-user",
                interaction_mode="human_counselor_ai_client",
                participant_mode="two_clients",
                max_turns=2,
                fixed_speaker_sequence=(
                    "counselor",
                    "counselor",
                    "client_a",
                    "client_b",
                ),
                speaker_selection_policy="turn_boundary_timing",
                turn_taking_max_reconsider_rounds=0,
                participants={
                    "counselor": ParticipantConfig(
                        "counselor",
                        "counselor",
                        "カウンセラー",
                        actor_kind=ActorKind.HUMAN,
                    ),
                    "client_a": ParticipantConfig(
                        "client_a",
                        "client",
                        "クライアントA",
                    ),
                    "client_b": ParticipantConfig(
                        "client_b",
                        "client",
                        "クライアントB",
                    ),
                },
            ),
            agents={"client_a": client_a, "client_b": client_b},
            sessions_dir=tmp_path,
        )
        run_task = asyncio.create_task(runtime.run())
        await _wait_for_human_input(runtime)

        await runtime.submit_human_turn(
            HumanTurnInput(
                session_id="session-human-two-client-no-consecutive-user",
                text="まず二人の反応を聞かせてください。",
                recipient_ids=("client_a", "client_b"),
            )
        )
        turns = await asyncio.wait_for(run_task, timeout=2.0)

        assert [turn.speaker for turn in turns] == ["counselor", "client_a"]

    asyncio.run(scenario())


def test_human_counselor_two_client_mode_uses_client_fallback_before_first_client_reply(
    tmp_path,
) -> None:
    async def scenario() -> None:
        client_a = FakeAgent(
            "client_a",
            "A応答:{input_transcript}",
            timing_decisions=[
                {"agent_id": "client_a", "action": "REQUEST_MAIN_FLOOR"},
                {"agent_id": "client_a", "action": "WAIT"},
            ],
        )
        client_b = FakeAgent(
            "client_b",
            "B応答:{input_transcript}",
            timing_decisions=[
                {"agent_id": "client_b", "action": "REQUEST_MAIN_FLOOR"},
                {"agent_id": "client_b", "action": "WAIT"},
            ],
        )
        runtime = ConversationRuntime(
            config=RuntimeConfig(
                session_id="session-human-two-client-initial-wait-fallback",
                interaction_mode="human_counselor_ai_client",
                participant_mode="two_clients",
                max_turns=2,
                fixed_speaker_sequence=(
                    "counselor",
                    "client_b",
                    "client_a",
                    "counselor",
                ),
                speaker_selection_policy="turn_boundary_timing",
                turn_taking_max_reconsider_rounds=1,
                participants={
                    "counselor": ParticipantConfig(
                        "counselor",
                        "counselor",
                        "カウンセラー",
                        actor_kind=ActorKind.HUMAN,
                    ),
                    "client_a": ParticipantConfig(
                        "client_a",
                        "client",
                        "クライアントA",
                    ),
                    "client_b": ParticipantConfig(
                        "client_b",
                        "client",
                        "クライアントB",
                    ),
                },
            ),
            agents={"client_a": client_a, "client_b": client_b},
            sessions_dir=tmp_path,
        )
        run_task = asyncio.create_task(runtime.run())
        await _wait_for_human_input(runtime)

        await runtime.submit_human_turn(
            HumanTurnInput(
                session_id="session-human-two-client-initial-wait-fallback",
                text="まず二人の反応を聞かせてください。",
                recipient_ids=("client_a", "client_b"),
            )
        )
        turns = await asyncio.wait_for(run_task, timeout=2.0)

        assert [turn.speaker for turn in turns] == ["counselor", "client_b"]

    asyncio.run(scenario())


def test_human_counselor_two_client_mode_keeps_scheduled_user_turn(
    tmp_path,
) -> None:
    async def scenario() -> None:
        client_a = FakeAgent(
            "client_a",
            "A応答:{input_transcript}",
            timing_decisions=[
                {"agent_id": "client_a", "action": "WAIT"},
                {"agent_id": "client_a", "action": "REQUEST_MAIN_FLOOR"},
            ],
        )
        client_b = FakeAgent(
            "client_b",
            "B応答:{input_transcript}",
            timing_decisions=[{"agent_id": "client_b", "action": "REQUEST_MAIN_FLOOR"}],
        )
        runtime = ConversationRuntime(
            config=RuntimeConfig(
                session_id="session-human-two-client-handoff",
                interaction_mode="human_counselor_ai_client",
                participant_mode="two_clients",
                max_turns=4,
                fixed_speaker_sequence=(
                    "counselor",
                    "client_b",
                    "client_a",
                    "counselor",
                ),
                speaker_selection_policy="turn_boundary_timing",
                turn_taking_max_reconsider_rounds=0,
                participants={
                    "counselor": ParticipantConfig(
                        "counselor",
                        "counselor",
                        "カウンセラー",
                        actor_kind=ActorKind.HUMAN,
                    ),
                    "client_a": ParticipantConfig(
                        "client_a",
                        "client",
                        "クライアントA",
                    ),
                    "client_b": ParticipantConfig(
                        "client_b",
                        "client",
                        "クライアントB",
                    ),
                },
            ),
            agents={"client_a": client_a, "client_b": client_b},
            sessions_dir=tmp_path,
        )
        run_task = asyncio.create_task(runtime.run())
        await _wait_for_human_input(runtime)

        await runtime.submit_human_turn(
            HumanTurnInput(
                session_id="session-human-two-client-handoff",
                text="二人の受け止めを聞かせてください。",
                recipient_ids=("client_a", "client_b"),
            )
        )
        await _wait_for_human_input(runtime)
        assert [turn.speaker for turn in runtime.turns] == [
            "counselor",
            "client_b",
            "client_a",
        ]
        assert runtime.status.current_speaker == "counselor"

        await runtime.submit_human_turn(
            HumanTurnInput(
                session_id="session-human-two-client-handoff",
                text="ありがとうございます。ここで一度止めます。",
                recipient_ids=("client_a", "client_b"),
            )
        )
        turns = await asyncio.wait_for(run_task, timeout=2.0)

        assert [turn.speaker for turn in turns] == [
            "counselor",
            "client_b",
            "client_a",
            "counselor",
        ]

    asyncio.run(scenario())


def test_human_audio_submission_transcribes_saves_and_auto_submits_turn(
    tmp_path,
) -> None:
    async def scenario() -> None:
        stt = _FinalTextStt("音声から確定した相談導入です。")
        runtime = ConversationRuntime(
            config=RuntimeConfig(
                session_id="session-human-audio",
                interaction_mode="human_counselor_ai_client",
                participant_mode="one_client",
                max_turns=2,
                fixed_speaker_sequence=("counselor", "client"),
                speaker_selection_policy="fixed_round_robin",
                participants={
                    "counselor": ParticipantConfig(
                        "counselor",
                        "counselor",
                        "カウンセラー",
                        actor_kind="human",
                    ),
                    "client": ParticipantConfig("client", "client", "クライアント"),
                },
            ),
            agents={"client": FakeAgent("client", "返答:{input_transcript}")},
            stt=stt,
            sessions_dir=tmp_path,
        )
        monitor_queue = runtime.audio_bus.subscribe_monitor()
        run_task = asyncio.create_task(runtime.run())
        await _wait_for_human_input(runtime)

        result = await runtime.submit_human_audio(
            HumanAudioInput(
                session_id="session-human-audio",
                audio_bytes=b"\x00\x00" * 240,
                sample_rate=24000,
                channels=1,
                recording_mode="push_to_talk",
                recipient_ids=("client",),
                client_message_id="message-1",
            )
        )
        turns = await asyncio.wait_for(run_task, timeout=2.0)

        first_monitor_item = await asyncio.wait_for(monitor_queue.get(), timeout=1.0)
        assert isinstance(first_monitor_item, TranscriptEvent)
        assert first_monitor_item.speaker == "counselor"
        assert first_monitor_item.text == "音声から確定した相談導入です。"
        assert result["accepted"] is True
        assert result["transcript"] == "音声から確定した相談導入です。"
        assert result["source_audio_ref"] == "internal/audio/turn_0001_counselor.wav"
        assert [turn.speaker for turn in turns] == ["counselor", "client"]
        assert turns[0].audio_log_path == "internal/audio/turn_0001_counselor.wav"
        assert stt.received_chunks[0].speaker == "counselor"
        assert (
            tmp_path
            / "session-human-audio"
            / "internal"
            / "audio"
            / "turn_0001_counselor.wav"
        ).exists()
        assert (
            tmp_path
            / "session-human-audio"
            / "internal"
            / "audio"
            / "session_realtime.wav"
        ).exists()
        assert (
            tmp_path
            / "session-human-audio"
            / "internal"
            / "audio"
            / "session_realtime.wav"
        ).stat().st_size > 0

    asyncio.run(scenario())


def test_human_counselor_one_client_returns_to_user_after_client_reply(
    tmp_path,
) -> None:
    async def scenario() -> None:
        client_agent = FakeAgent("client", "クライアント応答:{input_transcript}")
        runtime = ConversationRuntime(
            config=RuntimeConfig(
                session_id="session-human-one-client-return",
                interaction_mode="human_counselor_ai_client",
                participant_mode="one_client",
                max_turns=4,
                fixed_speaker_sequence=("counselor", "client"),
                speaker_selection_policy="fixed_round_robin",
                participants={
                    "counselor": ParticipantConfig(
                        "counselor",
                        "counselor",
                        "カウンセラー",
                        actor_kind=ActorKind.HUMAN,
                    ),
                    "client": ParticipantConfig("client", "client", "クライアント"),
                },
            ),
            agents={"client": client_agent},
            sessions_dir=tmp_path,
        )
        run_task = asyncio.create_task(runtime.run())
        await _wait_for_human_input(runtime)

        result = await runtime.submit_human_turn(
            HumanTurnInput(
                session_id="session-human-one-client-return",
                text="まず今の気持ちを聞かせてください。",
                recipient_ids=("client",),
            )
        )
        await _wait_for_human_input(runtime)

        assert result["accepted"] is True
        assert runtime.status.awaiting_human_input is True
        assert runtime.status.current_speaker == "counselor"
        assert [turn.speaker for turn in runtime.turns] == ["counselor", "client"]
        assert runtime.turns[1].generated_text == (
            "クライアント応答:まず今の気持ちを聞かせてください。"
        )

        run_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await run_task

    asyncio.run(scenario())


def test_human_audio_stream_publishes_partial_and_auto_submits_turn(tmp_path) -> None:
    async def scenario() -> None:
        stt = _FinalTextStt(
            "ストリーミング音声から確定した相談導入です。",
            partial_texts=("ストリーミング音声の途中です",),
        )
        runtime = ConversationRuntime(
            config=RuntimeConfig(
                session_id="session-human-audio-stream",
                interaction_mode="human_counselor_ai_client",
                participant_mode="one_client",
                max_turns=2,
                fixed_speaker_sequence=("counselor", "client"),
                speaker_selection_policy="fixed_round_robin",
                participants={
                    "counselor": ParticipantConfig(
                        "counselor",
                        "counselor",
                        "カウンセラー",
                        actor_kind="human",
                    ),
                    "client": ParticipantConfig("client", "client", "クライアント"),
                },
            ),
            agents={"client": FakeAgent("client", "返答:{input_transcript}")},
            stt=stt,
            sessions_dir=tmp_path,
        )
        monitor_queue = runtime.audio_bus.subscribe_monitor()
        run_task = asyncio.create_task(runtime.run())
        await _wait_for_human_input(runtime)

        start = await runtime.start_human_audio_stream(
            HumanAudioStreamStart(
                session_id="session-human-audio-stream",
                sample_rate=24000,
                channels=1,
                recording_mode="push_to_talk",
                recipient_ids=("client",),
                client_message_id="stream-message-1",
            )
        )
        chunk = await runtime.append_human_audio_stream_chunk(
            stream_id="stream-message-1",
            audio_bytes=b"\x01\x00" * 240,
            chunk_index=0,
        )
        partial = await asyncio.wait_for(monitor_queue.get(), timeout=1.0)
        end = await runtime.end_human_audio_stream(stream_id="stream-message-1")
        turns = await asyncio.wait_for(run_task, timeout=2.0)

        monitor_items = [partial]
        for _ in range(4):
            try:
                monitor_items.append(
                    await asyncio.wait_for(monitor_queue.get(), timeout=0.2)
                )
            except TimeoutError:
                break
        final_items = [
            item
            for item in monitor_items
            if isinstance(item, TranscriptEvent)
            and item.transcript_type == "human_final"
        ]

        assert start["stream_id"] == "stream-message-1"
        assert chunk["chunk_index"] == 0
        assert end["source_audio_ref"] == "internal/audio/turn_0001_counselor.wav"
        assert isinstance(partial, TranscriptEvent)
        assert partial.transcript_type == "partial"
        assert partial.text == "ストリーミング音声の途中です"
        assert partial.metadata["text_source"] == "human_streaming_stt"
        assert final_items
        assert final_items[0].text == "ストリーミング音声から確定した相談導入です。"
        assert [turn.speaker for turn in turns] == ["counselor", "client"]
        assert turns[0].stt_final_transcript == (
            "ストリーミング音声から確定した相談導入です。"
        )
        assert stt.received_chunks[0].chunk_index == 0
        assert (
            tmp_path
            / "session-human-audio-stream"
            / "internal"
            / "audio"
            / "turn_0001_counselor.wav"
        ).exists()

    asyncio.run(scenario())


def test_vad_auto_human_audio_stream_completes_on_stt_final_without_client_end(
    tmp_path,
) -> None:
    class VadAutoStt:
        def __init__(self) -> None:
            self.recording_modes: list[str] = []
            self.received_chunks: list[AudioChunk] = []

        async def transcribe_from_queue_observed(
            self,
            *,
            session_id,
            turn_id,
            speaker,
            queue,
            on_transcript,
            recording_mode="push_to_talk",
        ):
            self.recording_modes.append(recording_mode)
            item = await queue.get()
            assert isinstance(item, AudioChunk)
            self.received_chunks.append(item)
            final = TranscriptEvent(
                session_id=session_id,
                turn_id=turn_id,
                speaker=speaker,
                transcript_type="final",
                text="VADで区切られた相談内容です。",
            )
            await on_transcript(final)
            return [], final

    async def scenario() -> None:
        stt = VadAutoStt()
        runtime = ConversationRuntime(
            config=RuntimeConfig(
                session_id="session-vad-auto-audio-stream",
                interaction_mode="human_counselor_ai_client",
                participant_mode="one_client",
                max_turns=2,
                fixed_speaker_sequence=("counselor", "client"),
                speaker_selection_policy="fixed_round_robin",
                participants={
                    "counselor": ParticipantConfig(
                        "counselor",
                        "counselor",
                        "カウンセラー",
                        actor_kind="human",
                    ),
                    "client": ParticipantConfig("client", "client", "クライアント"),
                },
            ),
            agents={"client": FakeAgent("client", "返答:{input_transcript}")},
            stt=stt,
            sessions_dir=tmp_path,
        )
        run_task = asyncio.create_task(runtime.run())
        await _wait_for_human_input(runtime)

        start = await runtime.start_human_audio_stream(
            HumanAudioStreamStart(
                session_id="session-vad-auto-audio-stream",
                sample_rate=24000,
                channels=1,
                recording_mode="vad_auto",
                recipient_ids=("client",),
                client_message_id="stream-message-vad-auto",
            )
        )
        completion_task = asyncio.create_task(
            runtime.wait_human_audio_stream_completion(
                stream_id="stream-message-vad-auto"
            )
        )
        await runtime.append_human_audio_stream_chunk(
            stream_id="stream-message-vad-auto",
            audio_bytes=b"\x01\x00" * 240,
            chunk_index=0,
        )

        completion = await asyncio.wait_for(completion_task, timeout=1.0)
        turns = await asyncio.wait_for(run_task, timeout=2.0)

        assert start["stream_id"] == "stream-message-vad-auto"
        assert completion["completion_reason"] == "server_vad"
        assert (
            completion["source_audio_ref"] == "internal/audio/turn_0001_counselor.wav"
        )
        assert stt.recording_modes == ["vad_auto"]
        assert stt.received_chunks[0].chunk_index == 0
        assert turns[0].stt_final_transcript == "VADで区切られた相談内容です。"
        assert [turn.speaker for turn in turns] == ["counselor", "client"]

    asyncio.run(scenario())


def test_blank_human_audio_stream_keeps_waiting_for_human_turn(tmp_path) -> None:
    async def scenario() -> None:
        stt = _FinalTextStt("")
        runtime = ConversationRuntime(
            config=RuntimeConfig(
                session_id="session-human-audio-stream-blank",
                interaction_mode="human_counselor_ai_client",
                participant_mode="one_client",
                max_turns=2,
                fixed_speaker_sequence=("counselor", "client"),
                speaker_selection_policy="fixed_round_robin",
                participants={
                    "counselor": ParticipantConfig(
                        "counselor",
                        "counselor",
                        "カウンセラー",
                        actor_kind="human",
                    ),
                    "client": ParticipantConfig("client", "client", "クライアント"),
                },
            ),
            agents={"client": FakeAgent("client", "返答:{input_transcript}")},
            stt=stt,
            sessions_dir=tmp_path,
        )
        run_task = asyncio.create_task(runtime.run())
        await _wait_for_human_input(runtime)

        await runtime.start_human_audio_stream(
            HumanAudioStreamStart(
                session_id="session-human-audio-stream-blank",
                sample_rate=24000,
                channels=1,
                recording_mode="push_to_talk",
                recipient_ids=("client",),
                client_message_id="stream-message-blank",
            )
        )
        await runtime.append_human_audio_stream_chunk(
            stream_id="stream-message-blank",
            audio_bytes=b"\x00\x00" * 240,
            chunk_index=0,
        )
        await runtime.end_human_audio_stream(stream_id="stream-message-blank")
        await _wait_for_human_input(runtime)

        status = runtime.status
        assert status.awaiting_human_input is True
        assert status.current_turn_id == 1
        assert status.completed_turns == 0
        assert status.human_input_state == "waiting_for_human_speech"

        stt.final_text = "認識できた相談導入です。"
        await runtime.start_human_audio_stream(
            HumanAudioStreamStart(
                session_id="session-human-audio-stream-blank",
                sample_rate=24000,
                channels=1,
                recording_mode="push_to_talk",
                recipient_ids=("client",),
                client_message_id="stream-message-accepted",
            )
        )
        await runtime.append_human_audio_stream_chunk(
            stream_id="stream-message-accepted",
            audio_bytes=b"\x01\x00" * 240,
            chunk_index=0,
        )
        await runtime.end_human_audio_stream(stream_id="stream-message-accepted")
        turns = await asyncio.wait_for(run_task, timeout=2.0)

        assert [turn.speaker for turn in turns] == ["counselor", "client"]
        assert turns[0].turn_id == 1
        assert turns[0].stt_final_transcript == "認識できた相談導入です。"

        event_path = (
            tmp_path
            / "session-human-audio-stream-blank"
            / "internal"
            / "events"
            / "session-human-audio-stream-blank.jsonl"
        )
        events = [
            json.loads(line)
            for line in event_path.read_text(encoding="utf-8").splitlines()
        ]
        blank_events = [
            event for event in events if event["event_type"] == "human_stt_blank"
        ]
        committed_events = [
            event
            for event in events
            if event["event_type"] == "human_streaming_audio_committed"
        ]

        assert len(blank_events) == 1
        assert blank_events[0]["turn_id"] == 1
        assert blank_events[0]["details"]["input_mode"] == "human_streaming_stt"
        assert blank_events[0]["details"]["stt_outcome"] == "empty_transcript"
        assert blank_events[0]["details"]["stt_error_code"] is None
        assert len(committed_events) == 1
        assert committed_events[0]["details"]["char_count"] > 0

    asyncio.run(scenario())


def test_empty_commit_human_audio_stream_error_keeps_waiting_for_human_turn(
    tmp_path,
) -> None:
    from counseling_voice_demo.runtime.streaming_stt import RealtimeTranscriptionError

    class EmptyCommitStt:
        def __init__(self) -> None:
            self.calls = 0

        async def transcribe_from_queue_observed(
            self,
            *,
            session_id,
            turn_id,
            speaker,
            queue,
            on_transcript,
            recording_mode="push_to_talk",
        ):
            _ = session_id, turn_id, speaker, queue, on_transcript, recording_mode
            self.calls += 1
            raise RealtimeTranscriptionError(
                "Realtime transcription error: "
                "code=input_audio_buffer_commit_empty: "
                "Error committing input audio buffer: buffer too small. "
                "Expected at least 100ms of audio, "
                "but buffer only has 0.00ms of audio.",
                code="input_audio_buffer_commit_empty",
            )

    async def scenario() -> None:
        runtime = ConversationRuntime(
            config=RuntimeConfig(
                session_id="session-human-audio-stream-empty-commit",
                interaction_mode="human_counselor_ai_client",
                participant_mode="one_client",
                max_turns=2,
                fixed_speaker_sequence=("counselor", "client"),
                speaker_selection_policy="fixed_round_robin",
                participants={
                    "counselor": ParticipantConfig(
                        "counselor",
                        "counselor",
                        "カウンセラー",
                        actor_kind="human",
                    ),
                    "client": ParticipantConfig("client", "client", "クライアント"),
                },
            ),
            agents={"client": FakeAgent("client", "返答:{input_transcript}")},
            stt=EmptyCommitStt(),
            sessions_dir=tmp_path,
        )
        run_task = asyncio.create_task(runtime.run())
        await _wait_for_human_input(runtime)

        await runtime.start_human_audio_stream(
            HumanAudioStreamStart(
                session_id="session-human-audio-stream-empty-commit",
                sample_rate=24000,
                channels=1,
                recording_mode="vad_auto",
                recipient_ids=("client",),
                client_message_id="stream-message-empty-commit",
            )
        )
        await runtime.end_human_audio_stream(stream_id="stream-message-empty-commit")
        await _wait_for_human_input(runtime)

        status = runtime.status
        assert status.phase is RuntimePhase.RUNNING
        assert status.awaiting_human_input is True
        assert status.current_turn_id == 1
        assert status.completed_turns == 0
        assert status.human_input_state == "waiting_for_human_speech"

        run_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await run_task

        events = [
            json.loads(line)
            for line in runtime.logger.paths.events_jsonl.read_text().splitlines()
        ]
        blank = next(
            event for event in events if event["event_type"] == "human_stt_blank"
        )
        assert blank["details"]["stt_outcome"] == "empty_audio_error"
        assert blank["details"]["stt_error_code"] == "input_audio_buffer_commit_empty"
        assert blank["details"]["partial_event_count"] == 0

    asyncio.run(scenario())


def test_playback_interrupt_log_uses_applied_audio_position(tmp_path) -> None:
    class CappedPlaybackAgent(InterruptiblePlaybackFakeAgent):
        async def stop_current_response_playback(self, **kwargs):
            result = await super().stop_current_response_playback(**kwargs)
            result["played_ms"] = 16650
            result["audio_end_ms"] = 14985
            return result

    async def scenario() -> None:
        session_id = "session-playback-applied-position"
        runtime = ConversationRuntime(
            config=RuntimeConfig(session_id=session_id),
            agents={
                "counselor": CappedPlaybackAgent("counselor", "test"),
                "client": FakeAgent("client", "test"),
            },
            sessions_dir=tmp_path,
        )
        await runtime.logger.start()
        try:
            result = await runtime.stop_current_response_playback(
                played_ms=17780,
                speaker="counselor",
                turn_id=1,
            )
        finally:
            await runtime.logger.close()
        events_path = (
            tmp_path / session_id / "internal" / "events" / f"{session_id}.jsonl"
        )
        records = [json.loads(line) for line in events_path.read_text().splitlines()]
        interrupt = next(
            record
            for record in records
            if record["event_type"] == "realtime_playback_interrupted"
        )
        assert result["played_ms"] == 16650
        assert interrupt["details"]["played_ms"] == 16650
        assert interrupt["details"]["requested_played_ms"] == 17780
        assert interrupt["details"]["audio_end_ms"] == 14985

    asyncio.run(scenario())


def test_human_barge_in_interrupt_opens_human_audio_stream_slot(tmp_path) -> None:
    async def scenario() -> None:
        client_agent = InterruptiblePlaybackFakeAgent(
            "client",
            "返答:{input_transcript}",
        )
        runtime = ConversationRuntime(
            config=RuntimeConfig(
                session_id="session-human-barge-in-slot",
                interaction_mode="human_counselor_ai_client",
                participant_mode="one_client",
                human_interrupts_enabled=True,
                max_turns=4,
                fixed_speaker_sequence=("counselor", "client"),
                speaker_selection_policy="fixed_round_robin",
                participants={
                    "counselor": ParticipantConfig(
                        "counselor",
                        "counselor",
                        "カウンセラー",
                        actor_kind="human",
                    ),
                    "client": ParticipantConfig("client", "client", "クライアント"),
                },
            ),
            agents={"client": client_agent},
            sessions_dir=tmp_path,
        )
        runtime._phase = RuntimePhase.RUNNING
        runtime._active_turn_id = 3
        runtime._active_speaker_id = "client"

        interrupt_result = await runtime.stop_current_response_playback(
            played_ms=850,
            speaker="client",
            turn_id=3,
            item_id="item_3",
            response_id="response_3",
            content_index=0,
            reason="human_barge_in",
        )

        status = runtime.status
        assert interrupt_result["awaiting_human_input"] is True
        assert interrupt_result["human_turn_id"] == 4
        assert interrupt_result["human_speaker"] == "counselor"
        assert status.awaiting_human_input is True
        assert status.current_turn_id == 4
        assert status.current_speaker == "counselor"
        assert status.active_speaker_id == "counselor"
        assert status.human_input_state == "waiting_for_human_speech"

        stream_start = await runtime.start_human_audio_stream(
            HumanAudioStreamStart(
                session_id="session-human-barge-in-slot",
                sample_rate=24000,
                channels=1,
                recording_mode="push_to_talk",
                recipient_ids=("client",),
                client_message_id="barge-stream",
            )
        )

        assert stream_start["accepted"] is True
        assert stream_start["turn_id"] == 4
        assert stream_start["speaker"] == "counselor"
        assert client_agent.stop_current_response_playback_calls == [
            {
                "played_ms": 850,
                "item_id": "item_3",
                "response_id": "response_3",
                "content_index": 0,
                "cancel_response": True,
                "truncate_item": True,
            }
        ]

    asyncio.run(scenario())


def test_human_barge_in_reuses_active_human_audio_stream_without_reserving_extra_turn(
    tmp_path,
) -> None:
    async def scenario() -> None:
        client_agent = InterruptiblePlaybackFakeAgent(
            "client",
            "返答:{input_transcript}",
        )
        runtime = ConversationRuntime(
            config=RuntimeConfig(
                session_id="session-human-barge-in-active-stream",
                interaction_mode="human_counselor_ai_client",
                participant_mode="one_client",
                human_interrupts_enabled=True,
                max_turns=6,
                fixed_speaker_sequence=("counselor", "client"),
                speaker_selection_policy="fixed_round_robin",
                participants={
                    "counselor": ParticipantConfig(
                        "counselor",
                        "counselor",
                        "カウンセラー",
                        actor_kind="human",
                    ),
                    "client": ParticipantConfig("client", "client", "クライアント"),
                },
            ),
            agents={"client": client_agent},
            sessions_dir=tmp_path,
        )
        runtime._phase = RuntimePhase.RUNNING
        runtime._active_turn_id = 5
        runtime._active_speaker_id = "counselor"
        runtime._human_input_turn_id = 5
        runtime._human_input_speaker = "counselor"
        runtime._human_input_state = "waiting_for_human_speech"

        stream_start = await runtime.start_human_audio_stream(
            HumanAudioStreamStart(
                session_id="session-human-barge-in-active-stream",
                sample_rate=24000,
                channels=1,
                recording_mode="vad_auto",
                recipient_ids=("client",),
                client_message_id="active-counselor-stream",
            )
        )
        assert stream_start["turn_id"] == 5

        interrupt_result = await runtime.stop_current_response_playback(
            played_ms=1200,
            speaker="client",
            turn_id=4,
            reason="human_barge_in",
        )

        assert interrupt_result["awaiting_human_input"] is False
        assert interrupt_result["human_audio_stream_active"] is True
        assert interrupt_result["human_turn_id"] == 5
        assert interrupt_result["human_speaker"] == "counselor"
        assert runtime._barge_in_human_turn_id is None
        assert runtime._barge_in_human_speaker is None
        assert runtime._active_turn_id == 5
        assert runtime._active_speaker_id == "counselor"

    asyncio.run(scenario())


def test_post_barge_in_recovery_skips_empty_realtime_ai_turn(tmp_path) -> None:
    class AudioOnlyThenTextRealtimeAgent:
        def __init__(self) -> None:
            self.calls: list[dict] = []

        async def stream_audio_response(
            self,
            *,
            session_id: str,
            turn_id: int,
            speaker: str,
            input_transcript: str,
            sample_rate: int,
            sample_width_bits: int,
            channels: int,
            delivery_mode: AudioDeliveryMode,
            **_kwargs,
        ):
            self.calls.append(
                {
                    "turn_id": turn_id,
                    "speaker": speaker,
                    "input_transcript": input_transcript,
                }
            )
            if len(self.calls) == 1:
                yield {
                    "audio_chunk": AudioChunk(
                        session_id=session_id,
                        turn_id=turn_id,
                        speaker=speaker,
                        chunk_index=0,
                        pcm=b"orphan",
                        sample_rate=sample_rate,
                        sample_width_bits=sample_width_bits,
                        channels=channels,
                        duration_ms=800,
                        delivery_mode=delivery_mode,
                    )
                }
                return
            yield {"text_delta": "正常応答です。"}
            yield {
                "audio_chunk": AudioChunk(
                    session_id=session_id,
                    turn_id=turn_id,
                    speaker=speaker,
                    chunk_index=0,
                    pcm=b"\x01\x00" * 120,
                    sample_rate=sample_rate,
                    sample_width_bits=sample_width_bits,
                    channels=channels,
                    duration_ms=5,
                    delivery_mode=delivery_mode,
                )
            }

    async def scenario() -> None:
        client_agent = AudioOnlyThenTextRealtimeAgent()
        runtime = ConversationRuntime(
            config=RuntimeConfig(
                session_id="session-post-barge-empty-realtime",
                interaction_mode="human_counselor_ai_client",
                participant_mode="one_client",
                human_interrupts_enabled=True,
                max_turns=3,
                fixed_speaker_sequence=("counselor", "client", "client"),
                speaker_selection_policy="fixed_round_robin",
                participants={
                    "counselor": ParticipantConfig(
                        "counselor",
                        "counselor",
                        "カウンセラー",
                        actor_kind=ActorKind.HUMAN,
                    ),
                    "client": ParticipantConfig(
                        "client",
                        "client",
                        "クライアント",
                    ),
                },
            ),
            agents={"client": client_agent},
            sessions_dir=tmp_path,
        )
        runtime._open_human_input_slot_after_interrupt(interrupted_turn_id=0)
        run_task = asyncio.create_task(runtime.run())
        await _wait_for_human_input(runtime)

        result = await runtime.submit_human_turn(
            HumanTurnInput(
                session_id="session-post-barge-empty-realtime",
                text="具体的にはどういう分担ができるでしょうか。",
                recipient_ids=("client",),
            )
        )
        turns = await asyncio.wait_for(run_task, timeout=2.0)

        assert result["accepted"] is True
        assert [turn.speaker for turn in turns] == ["counselor", "client"]
        assert [call["turn_id"] for call in client_agent.calls] == [2, 3]
        assert turns[1].turn_id == 3
        assert turns[1].generated_text == "正常応答です。"
        assert client_agent.calls[1]["input_transcript"] == (
            "具体的にはどういう分担ができるでしょうか。"
        )

        session_dir = tmp_path / "session-post-barge-empty-realtime"
        assert not (
            session_dir / "internal" / "audio" / "turn_0002_client.wav"
        ).exists()
        assert (session_dir / "internal" / "audio" / "turn_0003_client.wav").exists()

        events = [
            json.loads(line)
            for line in (
                session_dir
                / "internal"
                / "events"
                / "session-post-barge-empty-realtime.jsonl"
            )
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        assert any(
            event["event_type"] == "realtime_audio_discarded_without_transcript"
            and event["turn_id"] == 2
            for event in events
        )
        assert any(
            event["event_type"] == "empty_ai_turn_skipped_after_interrupt"
            and event["turn_id"] == 2
            for event in events
        )

    asyncio.run(scenario())


def test_human_barge_in_interrupt_uses_future_turn_when_payload_is_stale(
    tmp_path,
) -> None:
    async def scenario() -> None:
        client_a = InterruptiblePlaybackFakeAgent(
            "client_a",
            "夫の返答:{input_transcript}",
        )
        client_b = InterruptiblePlaybackFakeAgent(
            "client_b",
            "妻の返答:{input_transcript}",
        )
        runtime = ConversationRuntime(
            config=RuntimeConfig(
                session_id="session-human-barge-in-stale-payload",
                interaction_mode="human_counselor_ai_client",
                participant_mode="two_clients",
                human_interrupts_enabled=True,
                max_turns=5,
                fixed_speaker_sequence=("counselor", "client_a", "client_b"),
                speaker_selection_policy="fixed_round_robin",
                participants={
                    "counselor": ParticipantConfig(
                        "counselor",
                        "counselor",
                        "カウンセラー",
                        actor_kind="human",
                    ),
                    "client_a": ParticipantConfig("client_a", "client", "夫"),
                    "client_b": ParticipantConfig("client_b", "client", "妻"),
                },
            ),
            agents={"client_a": client_a, "client_b": client_b},
            sessions_dir=tmp_path,
        )
        runtime._phase = RuntimePhase.RUNNING
        runtime._active_turn_id = 3
        runtime._active_speaker_id = "client_b"
        runtime.turns.append(
            TurnRuntimeState(
                session_id="session-human-barge-in-stale-payload",
                turn_id=1,
                speaker="counselor",
                input_transcript="相談導入です。",
                stt_final_transcript="相談導入です。",
            )
        )
        runtime.turns.append(
            TurnRuntimeState(
                session_id="session-human-barge-in-stale-payload",
                turn_id=2,
                speaker="client_a",
                input_transcript="相談導入です。",
                generated_text="夫の返答です。",
            )
        )

        interrupt_result = await runtime.stop_current_response_playback(
            played_ms=500,
            speaker="client_a",
            turn_id=2,
            reason="human_barge_in",
        )

        status = runtime.status
        assert interrupt_result["human_turn_id"] == 4
        assert interrupt_result["human_speaker"] == "counselor"
        assert status.awaiting_human_input is True
        assert status.current_turn_id == 4
        assert status.current_speaker == "counselor"
        assert status.active_speaker_id == "counselor"

    asyncio.run(scenario())


def test_runtime_advances_to_reserved_barge_in_human_turn(tmp_path) -> None:
    async def scenario() -> None:
        runtime = ConversationRuntime(
            config=RuntimeConfig(
                session_id="session-barge-in-reserved-turn-advance",
                interaction_mode="human_counselor_ai_client",
                participant_mode="one_client",
                human_interrupts_enabled=True,
                max_turns=4,
                fixed_speaker_sequence=(
                    "counselor",
                    "client",
                    "counselor",
                    "client",
                ),
                speaker_selection_policy="fixed_round_robin",
                participants={
                    "counselor": ParticipantConfig(
                        "counselor",
                        "counselor",
                        "カウンセラー",
                        actor_kind=ActorKind.HUMAN,
                    ),
                    "client": ParticipantConfig("client", "client", "クライアント"),
                },
            ),
            agents={"client": FakeAgent("client", "応答:{input_transcript}")},
            sessions_dir=tmp_path,
        )
        runtime._active_turn_id = 3
        runtime._active_speaker_id = "client"
        runtime._open_human_input_slot_after_interrupt(interrupted_turn_id=2)

        run_task = asyncio.create_task(runtime.run())
        await _wait_for_human_input(runtime)

        result = await runtime.submit_human_turn(
            HumanTurnInput(
                session_id="session-barge-in-reserved-turn-advance",
                text="割り込み後の発話です。",
                recipient_ids=("client",),
            )
        )
        turns = await asyncio.wait_for(run_task, timeout=2.0)

        assert result["accepted"] is True
        assert result["turn_id"] == 4
        assert [turn.turn_id for turn in turns] == [4]
        assert turns[0].speaker == "counselor"
        assert turns[0].stt_final_transcript == "割り込み後の発話です。"

        events = [
            json.loads(line)
            for line in (
                tmp_path
                / "session-barge-in-reserved-turn-advance"
                / "internal"
                / "events"
                / "session-barge-in-reserved-turn-advance.jsonl"
            )
            .read_text(encoding="utf-8")
            .splitlines()
        ]

        assert any(
            event["event_type"] == "barge_in_reserved_turn_advanced"
            and event["details"]["from_turn_id"] == 1
            and event["details"]["reserved_turn_id"] == 4
            for event in events
        )
        assert not any(event["event_type"] == "runtime_error" for event in events)

    asyncio.run(scenario())


def test_runtime_adopts_reserved_barge_in_stream_when_human_wait_races(
    tmp_path,
) -> None:
    async def scenario() -> None:
        runtime = ConversationRuntime(
            config=RuntimeConfig(
                session_id="session-barge-in-human-wait-race",
                interaction_mode="human_counselor_ai_client",
                participant_mode="one_client",
                human_interrupts_enabled=True,
                max_turns=5,
                fixed_speaker_sequence=(
                    "counselor",
                    "client",
                    "counselor",
                    "client",
                ),
                speaker_selection_policy="fixed_round_robin",
                participants={
                    "counselor": ParticipantConfig(
                        "counselor",
                        "counselor",
                        "カウンセラー",
                        actor_kind=ActorKind.HUMAN,
                    ),
                    "client": ParticipantConfig("client", "client", "クライアント"),
                },
            ),
            agents={"client": FakeAgent("client", "応答:{input_transcript}")},
            stt=FakeStreamingSTT(),
            sessions_dir=tmp_path,
        )
        await runtime.logger.start()
        try:
            runtime._phase = RuntimePhase.RUNNING
            runtime._active_turn_id = 3
            runtime._active_speaker_id = "counselor"
            slot = runtime._open_human_input_slot_after_interrupt(
                interrupted_turn_id=2,
            )
            assert slot["human_turn_id"] == 4
            start = await runtime.start_human_audio_stream(
                HumanAudioStreamStart(
                    session_id="session-barge-in-human-wait-race",
                    sample_rate=24000,
                    channels=1,
                    recording_mode="push_to_talk",
                    recipient_ids=("client",),
                    client_message_id="barge-race-stream-turn-4",
                )
            )
            await runtime.append_human_audio_stream_chunk(
                stream_id="barge-race-stream-turn-4",
                audio_bytes=b"fake:a:b:barge human\0",
                chunk_index=0,
            )
            await runtime.end_human_audio_stream(
                stream_id="barge-race-stream-turn-4",
            )

            turn = await runtime._await_human_turn(
                turn_id=3,
                speaker="counselor",
            )
        finally:
            await runtime.logger.close()

        assert start["turn_id"] == 4
        assert turn.turn_id == 4
        assert turn.speaker == "counselor"
        assert turn.stt_final_transcript == "stt_final:barge human"
        assert runtime._barge_in_human_turn_id is None
        assert runtime._barge_in_human_speaker is None
        assert runtime._post_barge_in_recovery_turns_remaining == 2

        events = [
            json.loads(line)
            for line in (
                tmp_path
                / "session-barge-in-human-wait-race"
                / "internal"
                / "events"
                / "session-barge-in-human-wait-race.jsonl"
            )
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        assert any(
            event["event_type"] == "barge_in_reserved_turn_advanced"
            and event["details"]["from_turn_id"] == 3
            and event["details"]["reserved_turn_id"] == 4
            for event in events
        )
        assert not any(event["event_type"] == "runtime_error" for event in events)

    asyncio.run(scenario())


def test_human_barge_in_interrupt_stops_same_speaker_running_turn_when_payload_is_stale(
    tmp_path,
) -> None:
    async def scenario() -> None:
        client_a = InterruptiblePlaybackFakeAgent(
            "client_a",
            "妻の返答:{input_transcript}",
        )
        client_b = InterruptiblePlaybackFakeAgent(
            "client_b",
            "夫の返答:{input_transcript}",
        )
        runtime = ConversationRuntime(
            config=RuntimeConfig(
                session_id="session-human-barge-in-stale-same-speaker",
                interaction_mode="human_counselor_ai_client",
                participant_mode="two_clients",
                human_interrupts_enabled=True,
                max_turns=12,
                fixed_speaker_sequence=("counselor", "client_a", "client_b"),
                speaker_selection_policy="fixed_round_robin",
                participants={
                    "counselor": ParticipantConfig(
                        "counselor",
                        "counselor",
                        "カウンセラー",
                        actor_kind="human",
                    ),
                    "client_a": ParticipantConfig("client_a", "client", "妻"),
                    "client_b": ParticipantConfig("client_b", "client", "夫"),
                },
            ),
            agents={"client_a": client_a, "client_b": client_b},
            sessions_dir=tmp_path,
        )
        monitor_queue = runtime.audio_bus.subscribe_monitor()
        runtime._phase = RuntimePhase.RUNNING
        runtime._active_turn_id = 9
        runtime._active_speaker_id = "client_b"
        runtime._running_ai_turn_id = 9
        runtime._running_ai_speaker = "client_b"

        interrupt_result = await runtime.stop_current_response_playback(
            played_ms=190,
            speaker="client_b",
            turn_id=7,
            item_id="stale_item_7",
            response_id="stale_response_7",
            content_index=0,
            reason="human_barge_in",
        )

        assert interrupt_result["human_turn_id"] == 10
        assert interrupt_result["human_speaker"] == "counselor"
        assert interrupt_result["active_response_stop"]["cancel_sent"] is True
        assert interrupt_result["active_response_stop"]["turn_id"] == 9
        assert interrupt_result["active_response_stop"]["speaker"] == "client_b"
        interrupted_events = [
            await asyncio.wait_for(monitor_queue.get(), timeout=1.0) for _ in range(2)
        ]
        assert [
            (event.transcript_type, event.turn_id, event.speaker)
            for event in interrupted_events
        ] == [
            ("interrupted", 7, "client_b"),
            ("interrupted", 9, "client_b"),
        ]
        assert client_b.stop_current_response_playback_calls == [
            {
                "played_ms": 190,
                "item_id": "stale_item_7",
                "response_id": "stale_response_7",
                "content_index": 0,
                "cancel_response": True,
                "truncate_item": True,
            },
            {
                "played_ms": 0,
                "item_id": None,
                "response_id": None,
                "content_index": None,
                "cancel_response": True,
                "truncate_item": False,
            },
        ]

    asyncio.run(scenario())


def test_human_barge_in_invalidates_already_generated_future_ai_turns(
    tmp_path,
) -> None:
    async def scenario() -> None:
        client_a = InterruptiblePlaybackFakeAgent(
            "client_a",
            "妻の返答:{input_transcript}",
        )
        client_b = InterruptiblePlaybackFakeAgent(
            "client_b",
            "夫の返答:{input_transcript}",
        )
        runtime = ConversationRuntime(
            config=RuntimeConfig(
                session_id="session-human-barge-in-invalidates-future",
                interaction_mode="human_counselor_ai_client",
                participant_mode="two_clients",
                human_interrupts_enabled=True,
                max_turns=6,
                fixed_speaker_sequence=("counselor", "client_b", "client_a"),
                speaker_selection_policy="fixed_round_robin",
                participants={
                    "counselor": ParticipantConfig(
                        "counselor",
                        "counselor",
                        "カウンセラー",
                        actor_kind="human",
                    ),
                    "client_a": ParticipantConfig("client_a", "client", "妻"),
                    "client_b": ParticipantConfig("client_b", "client", "夫"),
                },
            ),
            agents={"client_a": client_a, "client_b": client_b},
            sessions_dir=tmp_path,
        )
        runtime._phase = RuntimePhase.RUNNING
        runtime._active_turn_id = 2
        runtime._active_speaker_id = "client_b"
        runtime.turns = [
            TurnRuntimeState(
                session_id="session-human-barge-in-invalidates-future",
                turn_id=1,
                speaker="counselor",
                input_transcript="まず状況を教えてください。",
                recipient_ids=("client_a", "client_b"),
                stt_final_transcript="まず状況を教えてください。",
            ),
            TurnRuntimeState(
                session_id="session-human-barge-in-invalidates-future",
                turn_id=2,
                speaker="client_b",
                input_transcript="まず状況を教えてください。",
                recipient_ids=("counselor", "client_a"),
                generated_text="夫の途中発話です。",
                stt_final_transcript="夫の途中発話です。",
            ),
            TurnRuntimeState(
                session_id="session-human-barge-in-invalidates-future",
                turn_id=3,
                speaker="client_a",
                input_transcript="夫の途中発話です。",
                recipient_ids=("counselor", "client_b"),
                generated_text="割り込み前に先行生成された妻の発話です。",
                stt_final_transcript="割り込み前に先行生成された妻の発話です。",
            ),
        ]
        monitor_queue = runtime.audio_bus.subscribe_monitor()
        await runtime.logger.start()

        interrupt_result = await runtime.stop_current_response_playback(
            played_ms=420,
            speaker="client_b",
            turn_id=2,
            reason="human_barge_in",
        )
        await runtime.logger.close()

        monitor_events = [
            await asyncio.wait_for(monitor_queue.get(), timeout=1.0) for _ in range(2)
        ]
        events_path = (
            tmp_path
            / "session-human-barge-in-invalidates-future"
            / "internal"
            / "events"
            / "session-human-barge-in-invalidates-future.jsonl"
        )
        event_records = [
            json.loads(line)
            for line in events_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

        assert [turn.turn_id for turn in runtime.turns] == [1, 2]
        assert interrupt_result["human_turn_id"] == 3
        assert interrupt_result["human_speaker"] == "counselor"
        assert interrupt_result["invalidated_turns"] == [
            {"turn_id": 3, "speaker": "client_a", "speaker_id": "client_a"}
        ]
        assert [
            (event.transcript_type, event.turn_id, event.speaker)
            for event in monitor_events
        ] == [
            ("interrupted", 2, "client_b"),
            ("invalidated", 3, "client_a"),
        ]
        invalidation_events = [
            event
            for event in event_records
            if event["event_type"] == "barge_in_future_turns_invalidated"
        ]
        assert invalidation_events
        assert invalidation_events[0]["details"]["invalidated_turns"] == [
            {
                "turn_id": 3,
                "speaker": "client_a",
                "speaker_id": "client_a",
                "text_source": "generated_text",
            }
        ]

    asyncio.run(scenario())


def test_human_barge_in_reserved_stream_turn_preempts_next_client_turn(
    tmp_path,
) -> None:
    async def scenario() -> None:
        stt = _FinalTextStt("割り込み後のカウンセラー発話です。")
        client_a = SlowInterruptiblePlaybackFakeAgent(
            "client_a",
            "夫の返答:{input_transcript}",
        )
        client_b = SlowInterruptiblePlaybackFakeAgent(
            "client_b",
            "妻の返答:{input_transcript}",
        )
        runtime = ConversationRuntime(
            config=RuntimeConfig(
                session_id="session-human-barge-in-stream-slot",
                interaction_mode="human_counselor_ai_client",
                participant_mode="two_clients",
                human_interrupts_enabled=True,
                max_turns=4,
                fixed_speaker_sequence=("counselor", "client_a", "client_b"),
                speaker_selection_policy="fixed_round_robin",
                participants={
                    "counselor": ParticipantConfig(
                        "counselor",
                        "counselor",
                        "カウンセラー",
                        actor_kind="human",
                    ),
                    "client_a": ParticipantConfig("client_a", "client", "夫"),
                    "client_b": ParticipantConfig("client_b", "client", "妻"),
                },
            ),
            agents={"client_a": client_a, "client_b": client_b},
            stt=stt,
            sessions_dir=tmp_path,
        )
        run_task = asyncio.create_task(runtime.run())
        await _wait_for_human_input(runtime)
        await runtime.submit_human_turn(
            HumanTurnInput(
                session_id="session-human-barge-in-stream-slot",
                text="まず夫に状況を聞かせてください。",
                recipient_ids=("client_a", "client_b"),
            )
        )

        for _ in range(500):
            status = runtime.status
            if status.current_turn_id == 3 and status.current_speaker == "client_b":
                break
            await asyncio.sleep(0.01)
        else:
            raise AssertionError(f"client_b turn did not start: {runtime.status}")

        interrupt = await runtime.stop_current_response_playback(
            played_ms=400,
            speaker="client_a",
            turn_id=2,
            reason="human_barge_in",
        )
        assert interrupt["human_turn_id"] == 4
        assert interrupt["active_response_stop"]["cancel_sent"] is True
        assert interrupt["active_response_stop"]["turn_id"] == 3
        assert interrupt["active_response_stop"]["speaker"] == "client_b"
        start = await runtime.start_human_audio_stream(
            HumanAudioStreamStart(
                session_id="session-human-barge-in-stream-slot",
                sample_rate=24000,
                channels=1,
                recording_mode="push_to_talk",
                recipient_ids=("client_a", "client_b"),
                client_message_id="barge-stream-turn-4",
            )
        )
        await runtime.append_human_audio_stream_chunk(
            stream_id="barge-stream-turn-4",
            audio_bytes=b"\x01\x00" * 240,
            chunk_index=0,
        )
        await runtime.end_human_audio_stream(stream_id="barge-stream-turn-4")
        turns = await asyncio.wait_for(run_task, timeout=3.0)

        assert start["turn_id"] == 4
        assert start["speaker"] == "counselor"
        assert [turn.speaker for turn in turns[:3]] == [
            "counselor",
            "client_a",
            "counselor",
        ]
        assert all(turn.speaker != "client_b" for turn in turns)
        assert turns[2].turn_id == 4
        assert turns[2].stt_final_transcript == "割り込み後のカウンセラー発話です。"
        assert stt.received_chunks[0].turn_id == 4
        assert stt.received_chunks[0].speaker == "counselor"
        assert client_b.generated_turns == [3]
        assert client_b.stop_current_response_playback_calls == [
            {
                "played_ms": 0,
                "item_id": None,
                "response_id": None,
                "content_index": None,
                "cancel_response": True,
                "truncate_item": False,
            }
        ]

    asyncio.run(scenario())


def test_human_audio_submission_uses_realtime_audio_input_when_client_supports_it(
    tmp_path,
) -> None:
    async def scenario() -> None:
        client_agent = _RealtimeAudioInputAgent(
            "client",
            input_transcript="音声から直接認識した相談導入です。",
            response_text="はい、聞かせてください。",
        )
        runtime = ConversationRuntime(
            config=RuntimeConfig(
                session_id="session-human-audio-realtime-direct",
                interaction_mode="human_counselor_ai_client",
                participant_mode="one_client",
                max_turns=2,
                fixed_speaker_sequence=("counselor", "client"),
                speaker_selection_policy="fixed_round_robin",
                participants={
                    "counselor": ParticipantConfig(
                        "counselor",
                        "counselor",
                        "カウンセラー",
                        actor_kind="human",
                    ),
                    "client": ParticipantConfig("client", "client", "クライアント"),
                },
            ),
            agents={"client": client_agent},
            stt=_FailingStt(),
            sessions_dir=tmp_path,
        )
        monitor_queue = runtime.audio_bus.subscribe_monitor()
        run_task = asyncio.create_task(runtime.run())
        await _wait_for_human_input(runtime)
        pcm = b"\x01\x00" * 240

        result = await runtime.submit_human_audio(
            HumanAudioInput(
                session_id="session-human-audio-realtime-direct",
                audio_bytes=pcm,
                sample_rate=24000,
                channels=1,
                recording_mode="push_to_talk",
                recipient_ids=("client",),
                client_message_id="message-direct",
            )
        )
        turns = await asyncio.wait_for(run_task, timeout=2.0)

        monitor_items = []
        for _ in range(6):
            try:
                monitor_items.append(
                    await asyncio.wait_for(monitor_queue.get(), timeout=0.2)
                )
            except TimeoutError:
                break
        human_final_items = [
            item
            for item in monitor_items
            if isinstance(item, TranscriptEvent)
            and item.transcript_type == "human_final"
        ]

        assert result["accepted"] is True
        assert result["input_mode"] == "realtime_audio"
        assert result["transcript"] == ""
        assert result["target_speaker"] == "client"
        assert client_agent.received_audio_inputs == [pcm]
        assert [turn.speaker for turn in turns] == ["counselor", "client"]
        assert turns[0].stt_final_transcript == "音声から直接認識した相談導入です。"
        assert turns[0].audio_log_path == "internal/audio/turn_0001_counselor.wav"
        assert turns[1].input_transcript == ""
        assert turns[1].generated_text == "はい、聞かせてください。"
        assert human_final_items
        assert human_final_items[0].text == "音声から直接認識した相談導入です。"
        assert human_final_items[0].metadata["text_source"] == (
            "realtime_input_audio_transcription"
        )
        assert (
            tmp_path
            / "session-human-audio-realtime-direct"
            / "internal"
            / "audio"
            / "turn_0001_counselor.wav"
        ).exists()

    asyncio.run(scenario())


def test_human_audio_submission_keeps_waiting_when_stt_is_blank(tmp_path) -> None:
    async def scenario() -> None:
        runtime = ConversationRuntime(
            config=RuntimeConfig(
                session_id="session-human-audio-blank",
                interaction_mode="human_counselor_ai_client",
                participant_mode="one_client",
                max_turns=2,
                fixed_speaker_sequence=("counselor", "client"),
                speaker_selection_policy="fixed_round_robin",
                participants={
                    "counselor": ParticipantConfig(
                        "counselor",
                        "counselor",
                        "カウンセラー",
                        actor_kind="human",
                    ),
                    "client": ParticipantConfig("client", "client", "クライアント"),
                },
            ),
            agents={"client": FakeAgent("client", "返答:{input_transcript}")},
            stt=_FinalTextStt("", partial_texts=("partial:counselor:1:1",)),
            sessions_dir=tmp_path,
        )
        monitor_queue = runtime.audio_bus.subscribe_monitor()
        run_task = asyncio.create_task(runtime.run())
        await _wait_for_human_input(runtime)

        result = await runtime.submit_human_audio(
            HumanAudioInput(
                session_id="session-human-audio-blank",
                audio_bytes=b"\x00\x00" * 240,
                sample_rate=24000,
                channels=1,
                recording_mode="push_to_talk",
                recipient_ids=("client",),
                client_message_id="message-blank",
            )
        )

        assert result["accepted"] is False
        assert result["transcript"] == ""
        assert result["awaiting_human_input"] is True
        assert runtime.status.awaiting_human_input is True
        assert runtime.turns == []
        assert monitor_queue.empty()

        run_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await run_task

    asyncio.run(scenario())
