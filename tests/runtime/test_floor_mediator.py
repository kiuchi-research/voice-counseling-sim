from __future__ import annotations

import pytest

from counseling_voice_demo.runtime.floor_mediator import (
    FloorMediator,
    FloorMediatorResultType,
    TimingDecision,
    TimingDecisionAction,
    TimingDecisionValidationError,
)


def test_timing_decision_validates_dict_and_json() -> None:
    decision = TimingDecision.validate(
        {
            "agent_id": "observer_a",
            "action": "REQUEST_MAIN_FLOOR",
            "preferred_timing": "after_current_clause",
            "urgency": 0.75,
            "target": "client",
            "reason_code": "clarify",
            "prepared_intent": "ask a brief follow-up",
            "expires_after_ms": 1500,
            "explicitly_addressed": True,
            "speech_ms_last_window": 250,
            "request_timestamp_ms": 100,
            "safety_intervention": False,
        }
    )

    assert decision.agent_id == "observer_a"
    assert decision.action is TimingDecisionAction.REQUEST_MAIN_FLOOR
    assert decision.urgency == 0.75
    assert decision.explicitly_addressed is True

    parsed = TimingDecision.validate(
        '{"agent_id": "observer_b", "action": "WAIT", "urgency": 0}'
    )

    assert parsed.agent_id == "observer_b"
    assert parsed.action is TimingDecisionAction.WAIT
    assert parsed.urgency == 0.0


@pytest.mark.parametrize(
    "action",
    [
        "INTERRUPT",
        "CONTINUE",
        "YIELD",
    ],
)
def test_timing_decision_accepts_step22_actions(action: str) -> None:
    decision = TimingDecision.validate({"agent_id": "observer_a", "action": action})

    assert decision.action is TimingDecisionAction(action)


@pytest.mark.parametrize(
    "payload",
    [
        "not-json",
        '["not", "an", "object"]',
        {"agent_id": "observer_a", "action": "SPEAK_NOW"},
        {"agent_id": "observer_a", "action": "REQUEST_BACKCHANNEL"},
        {"agent_id": "observer_a", "action": "WAIT", "urgency": 1.01},
        {"agent_id": "observer_a", "action": "WAIT", "expires_after_ms": -1},
    ],
)
def test_timing_decision_rejects_invalid_payloads(payload: object) -> None:
    with pytest.raises(TimingDecisionValidationError):
        TimingDecision.validate(payload)


def test_floor_mediator_excludes_wait_and_cancel_from_request_candidates() -> None:
    result = FloorMediator().decide(
        [
            TimingDecision.validate({"agent_id": "observer_a", "action": "WAIT"}),
            TimingDecision.validate({"agent_id": "observer_b", "action": "CANCEL"}),
        ]
    )

    assert result.result_type is FloorMediatorResultType.WAIT
    assert result.agent_id is None
    assert result.conflict_agent_ids == ()

    grant = FloorMediator().decide(
        [
            {"agent_id": "observer_a", "action": "WAIT"},
            {"agent_id": "observer_b", "action": "REQUEST_MAIN_FLOOR"},
            {"agent_id": "observer_c", "action": "CANCEL"},
        ]
    )

    assert grant.result_type is FloorMediatorResultType.GRANT
    assert grant.agent_id == "observer_b"
    assert grant.conflict_agent_ids == ()


def test_floor_mediator_grants_single_request() -> None:
    result = FloorMediator().decide(
        [{"agent_id": "observer_a", "action": "REQUEST_MAIN_FLOOR"}]
    )

    assert result.result_type is FloorMediatorResultType.GRANT
    assert result.agent_id == "observer_a"
    assert result.reason == "single_request"
    assert result.conflict_agent_ids == ()


def test_floor_mediator_reports_overlap_for_simultaneous_main_requests() -> None:
    result = FloorMediator().decide(
        [
            {"agent_id": "observer_a", "action": "REQUEST_MAIN_FLOOR"},
            {"agent_id": "observer_b", "action": "REQUEST_MAIN_FLOOR"},
        ],
        detect_overlap=True,
    )

    assert result.result_type is FloorMediatorResultType.OVERLAP
    assert result.agent_id is None
    assert result.reason == "simultaneous_requests"
    assert result.conflict_agent_ids == ("observer_a", "observer_b")


def test_floor_mediator_reports_overlap_for_interrupting_active_speaker() -> None:
    result = FloorMediator().decide(
        [{"agent_id": "observer_b", "action": "INTERRUPT"}],
        active_speaker_id="observer_a",
    )

    assert result.result_type is FloorMediatorResultType.OVERLAP
    assert result.agent_id == "observer_a"
    assert result.reason == "interrupt_requested"
    assert result.conflict_agent_ids == ("observer_a", "observer_b")


def test_floor_mediator_resolves_overlap_to_continue_decision() -> None:
    result = FloorMediator().resolve_overlap(
        [
            {"agent_id": "observer_a", "action": "YIELD"},
            {"agent_id": "observer_b", "action": "CONTINUE"},
        ],
        conflict_agent_ids=("observer_a", "observer_b"),
    )

    assert result.result_type is FloorMediatorResultType.GRANT
    assert result.agent_id == "observer_b"
    assert result.reason == "overlap_continue"
    assert result.conflict_agent_ids == ("observer_a", "observer_b")
    assert result.yielded_agent_ids == ("observer_a",)


def test_floor_mediator_resolves_all_yield_as_no_floor_holder() -> None:
    result = FloorMediator().resolve_overlap(
        [
            {"agent_id": "observer_a", "action": "YIELD"},
            {"agent_id": "observer_b", "action": "YIELD"},
        ],
        conflict_agent_ids=("observer_a", "observer_b"),
    )

    assert result.result_type is FloorMediatorResultType.YIELD
    assert result.agent_id is None
    assert result.reason == "all_yield"
    assert result.yielded_agent_ids == ("observer_a", "observer_b")


@pytest.mark.parametrize(
    ("winner", "decisions"),
    [
        (
            "observer_safety",
            [
                {
                    "agent_id": "observer_addressed",
                    "action": "REQUEST_MAIN_FLOOR",
                    "explicitly_addressed": True,
                    "speech_ms_last_window": 0,
                    "request_timestamp_ms": 1,
                },
                {
                    "agent_id": "observer_safety",
                    "action": "REQUEST_MAIN_FLOOR",
                    "safety_intervention": True,
                    "speech_ms_last_window": 900,
                    "request_timestamp_ms": 99,
                },
            ],
        ),
        (
            "observer_addressed",
            [
                {
                    "agent_id": "observer_low_speech",
                    "action": "REQUEST_MAIN_FLOOR",
                    "speech_ms_last_window": 0,
                    "request_timestamp_ms": 1,
                },
                {
                    "agent_id": "observer_addressed",
                    "action": "REQUEST_MAIN_FLOOR",
                    "explicitly_addressed": True,
                    "speech_ms_last_window": 900,
                    "request_timestamp_ms": 99,
                },
            ],
        ),
        (
            "observer_less_speech",
            [
                {
                    "agent_id": "observer_more_speech",
                    "action": "REQUEST_MAIN_FLOOR",
                    "speech_ms_last_window": 400,
                    "request_timestamp_ms": 1,
                },
                {
                    "agent_id": "observer_less_speech",
                    "action": "REQUEST_MAIN_FLOOR",
                    "speech_ms_last_window": 100,
                    "request_timestamp_ms": 99,
                },
            ],
        ),
        (
            "observer_earlier",
            [
                {
                    "agent_id": "observer_later",
                    "action": "REQUEST_MAIN_FLOOR",
                    "speech_ms_last_window": 100,
                    "request_timestamp_ms": 200,
                },
                {
                    "agent_id": "observer_earlier",
                    "action": "REQUEST_MAIN_FLOOR",
                    "speech_ms_last_window": 100,
                    "request_timestamp_ms": 100,
                },
            ],
        ),
    ],
)
def test_floor_mediator_uses_neutral_conflict_priority(
    winner: str,
    decisions: list[dict[str, object]],
) -> None:
    result = FloorMediator().decide(decisions)

    assert result.result_type is FloorMediatorResultType.GRANT
    assert result.agent_id == winner
    assert result.reason == "neutral_priority"
    assert result.conflict_agent_ids == tuple(
        sorted(decision["agent_id"] for decision in decisions)
    )


def test_floor_mediator_prioritizes_immediate_timing_in_conflict() -> None:
    result = FloorMediator().decide(
        [
            {
                "agent_id": "observer_a",
                "action": "REQUEST_MAIN_FLOOR",
                "preferred_timing": "natural_pause",
                "urgency": 0.8,
            },
            {
                "agent_id": "observer_z",
                "action": "REQUEST_MAIN_FLOOR",
                "preferred_timing": "immediate",
                "urgency": 0.3,
            },
        ]
    )

    assert result.result_type is FloorMediatorResultType.GRANT
    assert result.agent_id == "observer_z"
    assert result.reason == "neutral_priority"


def test_floor_mediator_prioritizes_higher_urgency_with_same_timing() -> None:
    result = FloorMediator().decide(
        [
            {
                "agent_id": "observer_a",
                "action": "REQUEST_MAIN_FLOOR",
                "preferred_timing": "natural_pause",
                "urgency": 0.2,
            },
            {
                "agent_id": "observer_z",
                "action": "REQUEST_MAIN_FLOOR",
                "preferred_timing": "natural_pause",
                "urgency": 0.75,
            },
        ]
    )

    assert result.result_type is FloorMediatorResultType.GRANT
    assert result.agent_id == "observer_z"
    assert result.reason == "neutral_priority"


def test_reason_code_does_not_affect_selection_priority() -> None:
    result = FloorMediator().decide(
        [
            {
                "agent_id": "observer_b",
                "action": "REQUEST_MAIN_FLOOR",
                "reason_code": "high_priority_clinical_content",
            },
            {
                "agent_id": "observer_a",
                "action": "REQUEST_MAIN_FLOOR",
                "reason_code": "low_priority_housekeeping",
            },
        ]
    )

    assert result.agent_id == "observer_a"


def test_floor_mediator_uses_deterministic_agent_id_tie_break() -> None:
    mediator = FloorMediator()
    decisions = [
        {"agent_id": "observer_z", "action": "REQUEST_MAIN_FLOOR"},
        {"agent_id": "observer_a", "action": "REQUEST_MAIN_FLOOR"},
    ]

    assert mediator.decide(decisions).agent_id == "observer_a"
    assert mediator.decide(reversed(decisions)).agent_id == "observer_a"
