from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Final


class TimingDecisionValidationError(ValueError):
    pass


class TimingDecisionAction(StrEnum):
    WAIT = "WAIT"
    REQUEST_MAIN_FLOOR = "REQUEST_MAIN_FLOOR"
    INTERRUPT = "INTERRUPT"
    CONTINUE = "CONTINUE"
    YIELD = "YIELD"
    CANCEL = "CANCEL"


class FloorMediatorResultType(StrEnum):
    WAIT = "WAIT"
    GRANT = "GRANT"
    OVERLAP = "OVERLAP"
    YIELD = "YIELD"


_TIMING_DECISION_FIELDS: Final = {
    "agent_id",
    "action",
    "preferred_timing",
    "urgency",
    "target",
    "reason_code",
    "prepared_intent",
    "expires_after_ms",
    "explicitly_addressed",
    "speech_ms_last_window",
    "request_timestamp_ms",
    "safety_intervention",
}
_PREFERRED_TIMING_PRIORITY: Final = {
    "immediate": 0,
    "natural_pause": 1,
    "short_hold": 2,
    "long_hold": 3,
}
_HIGH_URGENCY_IMMEDIATE_THRESHOLD: Final = 0.85


@dataclass(frozen=True)
class TimingDecision:
    agent_id: str
    action: TimingDecisionAction
    preferred_timing: str | None = None
    urgency: float | None = None
    target: str | None = None
    reason_code: str | None = None
    prepared_intent: Any | None = None
    expires_after_ms: int | None = None
    explicitly_addressed: bool | None = None
    speech_ms_last_window: int | None = None
    request_timestamp_ms: int | None = None
    safety_intervention: bool | None = None
    target_defaulted: bool = False
    preferred_timing_defaulted: bool = False
    urgency_defaulted: bool = False

    @classmethod
    def validate(cls, value: TimingDecisionInput) -> TimingDecision:
        if isinstance(value, TimingDecision):
            return value
        if isinstance(value, str):
            return cls.from_json(value)
        if isinstance(value, Mapping):
            return cls.from_dict(value)
        raise TimingDecisionValidationError(
            "timing decision must be a JSON object string or mapping"
        )

    @classmethod
    def from_json(cls, value: str) -> TimingDecision:
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError as exc:
            raise TimingDecisionValidationError(
                "timing decision JSON must be valid"
            ) from exc
        if not isinstance(parsed, Mapping):
            raise TimingDecisionValidationError(
                "timing decision JSON must decode to an object"
            )
        return cls.from_dict(parsed)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> TimingDecision:
        unknown_fields = sorted(set(value) - _TIMING_DECISION_FIELDS)
        if unknown_fields:
            fields = ", ".join(unknown_fields)
            raise TimingDecisionValidationError(
                f"timing decision contains unknown field(s): {fields}"
            )

        return cls(
            agent_id=_required_str(value, "agent_id"),
            action=_parse_action(value.get("action")),
            preferred_timing=_optional_str(value, "preferred_timing"),
            urgency=_optional_urgency(value, "urgency"),
            target=_optional_str(value, "target"),
            reason_code=_optional_str(value, "reason_code"),
            prepared_intent=value.get("prepared_intent"),
            expires_after_ms=_optional_non_negative_int(value, "expires_after_ms"),
            explicitly_addressed=_optional_bool(value, "explicitly_addressed"),
            speech_ms_last_window=_optional_non_negative_int(
                value, "speech_ms_last_window"
            ),
            request_timestamp_ms=_optional_non_negative_int(
                value, "request_timestamp_ms"
            ),
            safety_intervention=_optional_bool(value, "safety_intervention"),
        )


TimingDecisionInput = TimingDecision | Mapping[str, Any] | str


@dataclass(frozen=True)
class FloorMediatorResult:
    result_type: FloorMediatorResultType
    agent_id: str | None
    reason: str
    conflict_agent_ids: tuple[str, ...] = ()
    yielded_agent_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class FloorMediator:
    def decide(
        self,
        decisions: Iterable[TimingDecisionInput],
        *,
        active_speaker_id: str | None = None,
        detect_overlap: bool = False,
    ) -> FloorMediatorResult:
        parsed_decisions = tuple(TimingDecision.validate(item) for item in decisions)
        interrupts = tuple(
            decision
            for decision in parsed_decisions
            if decision.action is TimingDecisionAction.INTERRUPT
        )
        if active_speaker_id is not None and interrupts:
            conflict_agent_ids = _sorted_agent_ids(
                (active_speaker_id, *(decision.agent_id for decision in interrupts))
            )
            return FloorMediatorResult(
                result_type=FloorMediatorResultType.OVERLAP,
                agent_id=active_speaker_id,
                reason="interrupt_requested",
                conflict_agent_ids=conflict_agent_ids,
            )

        main_requests = tuple(
            decision
            for decision in parsed_decisions
            if decision.action in {
                TimingDecisionAction.REQUEST_MAIN_FLOOR,
                TimingDecisionAction.INTERRUPT,
            }
        )
        if main_requests:
            return self._decide_main_floor(
                main_requests,
                detect_overlap=detect_overlap,
            )

        return FloorMediatorResult(
            result_type=FloorMediatorResultType.WAIT,
            agent_id=None,
            reason="no_request",
        )

    def _decide_main_floor(
        self,
        requests: tuple[TimingDecision, ...],
        *,
        detect_overlap: bool,
    ) -> FloorMediatorResult:
        if not requests:
            return FloorMediatorResult(
                result_type=FloorMediatorResultType.WAIT,
                agent_id=None,
                reason="no_request",
            )
        if len(requests) == 1:
            return FloorMediatorResult(
                result_type=FloorMediatorResultType.GRANT,
                agent_id=requests[0].agent_id,
                reason="single_request",
            )
        if detect_overlap:
            return FloorMediatorResult(
                result_type=FloorMediatorResultType.OVERLAP,
                agent_id=None,
                reason="simultaneous_requests",
                conflict_agent_ids=_sorted_agent_ids(
                    decision.agent_id for decision in requests
                ),
            )

        winner = min(requests, key=_neutral_priority_key)
        conflict_agent_ids = _sorted_agent_ids(decision.agent_id for decision in requests)
        return FloorMediatorResult(
            result_type=FloorMediatorResultType.GRANT,
            agent_id=winner.agent_id,
            reason="neutral_priority",
            conflict_agent_ids=conflict_agent_ids,
        )

    def resolve_overlap(
        self,
        decisions: Iterable[TimingDecisionInput],
        *,
        active_speaker_id: str | None = None,
        conflict_agent_ids: Iterable[str] = (),
    ) -> FloorMediatorResult:
        parsed_decisions = tuple(TimingDecision.validate(item) for item in decisions)
        continue_decisions = tuple(
            decision
            for decision in parsed_decisions
            if decision.action is TimingDecisionAction.CONTINUE
        )
        yield_decisions = tuple(
            decision
            for decision in parsed_decisions
            if decision.action is TimingDecisionAction.YIELD
        )
        participant_ids = _sorted_agent_ids(
            [decision.agent_id for decision in parsed_decisions]
            + list(conflict_agent_ids)
        )
        yielded_agent_ids = _sorted_agent_ids(
            decision.agent_id for decision in yield_decisions
        )

        if len(continue_decisions) == 1:
            return FloorMediatorResult(
                result_type=FloorMediatorResultType.GRANT,
                agent_id=continue_decisions[0].agent_id,
                reason="overlap_continue",
                conflict_agent_ids=participant_ids,
                yielded_agent_ids=yielded_agent_ids,
            )
        if len(continue_decisions) > 1:
            winner = min(
                continue_decisions,
                key=lambda decision: _overlap_priority_key(
                    decision,
                    active_speaker_id=active_speaker_id,
                ),
            )
            return FloorMediatorResult(
                result_type=FloorMediatorResultType.GRANT,
                agent_id=winner.agent_id,
                reason="overlap_final_mediation",
                conflict_agent_ids=participant_ids,
                yielded_agent_ids=yielded_agent_ids,
            )
        if yield_decisions:
            return FloorMediatorResult(
                result_type=FloorMediatorResultType.YIELD,
                agent_id=None,
                reason="all_yield",
                conflict_agent_ids=participant_ids,
                yielded_agent_ids=yielded_agent_ids,
            )
        if active_speaker_id is not None:
            return FloorMediatorResult(
                result_type=FloorMediatorResultType.GRANT,
                agent_id=active_speaker_id,
                reason="overlap_no_resolution_keep_active",
                conflict_agent_ids=participant_ids,
            )
        return FloorMediatorResult(
            result_type=FloorMediatorResultType.WAIT,
            agent_id=None,
            reason="overlap_no_resolution",
            conflict_agent_ids=participant_ids,
        )


def _neutral_priority_key(
    decision: TimingDecision,
) -> tuple[int, int, float, float, int, float, str]:
    timing_rank, urgency_rank = timing_decision_conflict_priority(decision)
    return (
        0 if decision.safety_intervention is True else 1,
        0 if decision.explicitly_addressed is True else 1,
        _missing_last_number(decision.speech_ms_last_window),
        _missing_last_number(decision.request_timestamp_ms),
        timing_rank,
        urgency_rank,
        decision.agent_id,
    )


def _overlap_priority_key(
    decision: TimingDecision,
    *,
    active_speaker_id: str | None,
) -> tuple[int, int, int, float, float, str]:
    return (
        0 if decision.safety_intervention is True else 1,
        0 if decision.agent_id == active_speaker_id else 1,
        0 if decision.explicitly_addressed is True else 1,
        _missing_last_number(decision.request_timestamp_ms),
        _missing_last_number(decision.speech_ms_last_window),
        decision.agent_id,
    )


def timing_decision_conflict_priority(decision: TimingDecision) -> tuple[int, float]:
    urgency = _normalized_urgency(decision.urgency)
    return (
        _effective_preferred_timing_priority(
            decision.preferred_timing,
            urgency=urgency,
        ),
        -urgency,
    )


def _effective_preferred_timing_priority(
    preferred_timing: str | None,
    *,
    urgency: float,
) -> int:
    if urgency >= _HIGH_URGENCY_IMMEDIATE_THRESHOLD:
        return _PREFERRED_TIMING_PRIORITY["immediate"]
    normalized = (preferred_timing or "").strip().lower().replace("-", "_")
    return _PREFERRED_TIMING_PRIORITY.get(
        normalized,
        len(_PREFERRED_TIMING_PRIORITY),
    )


def _normalized_urgency(value: float | None) -> float:
    if value is None or math.isnan(value):
        return 0.0
    return max(0.0, min(1.0, float(value)))


def _sorted_agent_ids(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(sorted(set(values)))


def _parse_action(value: Any) -> TimingDecisionAction:
    if isinstance(value, TimingDecisionAction):
        return value
    if isinstance(value, str):
        try:
            return TimingDecisionAction(value)
        except ValueError as exc:
            raise TimingDecisionValidationError(
                f"unknown timing decision action: {value}"
            ) from exc
    raise TimingDecisionValidationError("timing decision action is required")


def _required_str(value: Mapping[str, Any], field_name: str) -> str:
    field_value = value.get(field_name)
    if not isinstance(field_value, str) or not field_value.strip():
        raise TimingDecisionValidationError(
            f"timing decision {field_name} must be a non-empty string"
        )
    return field_value.strip()


def _optional_str(value: Mapping[str, Any], field_name: str) -> str | None:
    if field_name not in value or value[field_name] is None:
        return None
    field_value = value[field_name]
    if not isinstance(field_value, str):
        raise TimingDecisionValidationError(
            f"timing decision {field_name} must be a string"
        )
    return field_value


def _optional_bool(value: Mapping[str, Any], field_name: str) -> bool | None:
    if field_name not in value or value[field_name] is None:
        return None
    field_value = value[field_name]
    if not isinstance(field_value, bool):
        raise TimingDecisionValidationError(
            f"timing decision {field_name} must be a boolean"
        )
    return field_value


def _optional_urgency(value: Mapping[str, Any], field_name: str) -> float | None:
    if field_name not in value or value[field_name] is None:
        return None
    field_value = value[field_name]
    if (
        isinstance(field_value, bool)
        or not isinstance(field_value, int | float)
        or not math.isfinite(field_value)
        or not 0.0 <= field_value <= 1.0
    ):
        raise TimingDecisionValidationError(
            f"timing decision {field_name} must be a number between 0.0 and 1.0"
        )
    return float(field_value)


def _optional_non_negative_int(
    value: Mapping[str, Any],
    field_name: str,
) -> int | None:
    if field_name not in value or value[field_name] is None:
        return None
    field_value = value[field_name]
    if (
        isinstance(field_value, bool)
        or not isinstance(field_value, int)
        or field_value < 0
    ):
        raise TimingDecisionValidationError(
            f"timing decision {field_name} must be a non-negative integer"
        )
    return field_value


def _missing_last_number(value: int | None) -> float:
    if value is None:
        return math.inf
    return float(value)
