from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Final, Protocol


COUNSELOR_ROLE: Final = "counselor"
CLIENT_ROLE: Final = "client"
LEGACY_ALTERNATING_SEQUENCE: Final[tuple[str, str]] = ("counselor", "client")
_SUPPORTED_PARTICIPANT_ROLES: Final = {COUNSELOR_ROLE, CLIENT_ROLE}
FIXED_ROUND_ROBIN_POLICY: Final = "fixed_round_robin"
TURN_BOUNDARY_TIMING_POLICY: Final = "turn_boundary_timing"
DISTRIBUTED_TIMING_POLICY: Final = "distributed_timing"
SUPPORTED_SPEAKER_SELECTION_POLICIES: Final = {
    FIXED_ROUND_ROBIN_POLICY,
    TURN_BOUNDARY_TIMING_POLICY,
    DISTRIBUTED_TIMING_POLICY,
}


class ParticipantRoleLike(Protocol):
    role: str


ParticipantRoleInput = str | ParticipantRoleLike


class SpeakerSelectionError(ValueError):
    pass


@dataclass(frozen=True)
class FixedSpeakerSequence:
    participant_roles: Mapping[str, ParticipantRoleInput]
    sequence: Sequence[str]

    def __post_init__(self) -> None:
        normalized_participant_roles = _normalize_participant_roles(
            self.participant_roles
        )
        normalized_sequence = tuple(self.sequence)
        validate_fixed_speaker_sequence(
            participant_roles=normalized_participant_roles,
            sequence=normalized_sequence,
        )
        object.__setattr__(self, "participant_roles", normalized_participant_roles)
        object.__setattr__(self, "sequence", normalized_sequence)

    def speaker_for_turn(self, turn_id: int) -> str:
        if turn_id <= 0:
            raise SpeakerSelectionError("turn_id must be positive")
        return self.sequence[(turn_id - 1) % len(self.sequence)]


@dataclass(frozen=True)
class FloorPrediction:
    candidate_speaker_ids: tuple[str, ...]
    reason: str
    source: str = "fixed_sequence"


@dataclass(frozen=True)
class FixedSequenceFloorPredictor:
    participant_roles: Mapping[str, ParticipantRoleInput]
    sequence: Sequence[str]
    max_candidates: int = 1
    _selector: FixedSpeakerSequence = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.max_candidates <= 0:
            raise SpeakerSelectionError("max_candidates must be positive")
        object.__setattr__(
            self,
            "_selector",
            FixedSpeakerSequence(
                participant_roles=self.participant_roles,
                sequence=self.sequence,
            ),
        )

    def predict(
        self,
        *,
        turn_id: int,
        previous_speaker: str | None = None,
        input_transcript: str | None = None,
    ) -> FloorPrediction:
        _ = input_transcript
        primary_speaker = self._selector.speaker_for_turn(turn_id)
        if self.max_candidates == 1:
            return FloorPrediction(
                candidate_speaker_ids=(primary_speaker,),
                reason="fixed_sequence",
                source="fixed_sequence",
            )

        candidates: list[str] = []
        start_index = (turn_id - 1) % len(self._selector.sequence)
        for offset in range(len(self._selector.sequence)):
            speaker_id = self._selector.sequence[
                (start_index + offset) % len(self._selector.sequence)
            ]
            if speaker_id == previous_speaker:
                continue
            if speaker_id in candidates:
                continue
            candidates.append(speaker_id)
            if len(candidates) >= self.max_candidates:
                break
        if not candidates:
            candidates.append(primary_speaker)
        return FloorPrediction(
            candidate_speaker_ids=tuple(candidates),
            reason="fixed_sequence_window",
            source="fixed_sequence",
        )


def speaker_for_fixed_sequence(
    turn_id: int,
    *,
    participant_roles: Mapping[str, ParticipantRoleInput],
    sequence: Sequence[str],
) -> str:
    selector = FixedSpeakerSequence(
        participant_roles=participant_roles,
        sequence=sequence,
    )
    return selector.speaker_for_turn(turn_id)


def validate_fixed_speaker_sequence(
    *,
    participant_roles: Mapping[str, ParticipantRoleInput],
    sequence: Sequence[str],
) -> None:
    normalized_participant_roles = _normalize_participant_roles(participant_roles)
    normalized_sequence = tuple(sequence)
    if not normalized_sequence:
        raise SpeakerSelectionError("speaker sequence must not be empty")

    unknown_speakers = sorted(
        set(normalized_sequence) - set(normalized_participant_roles)
    )
    if unknown_speakers:
        speakers = ", ".join(unknown_speakers)
        raise SpeakerSelectionError(
            f"speaker sequence contains unknown speaker(s): {speakers}"
        )

    roles_in_sequence = {
        normalized_participant_roles[speaker_id] for speaker_id in normalized_sequence
    }
    if COUNSELOR_ROLE not in roles_in_sequence:
        raise SpeakerSelectionError(
            "speaker sequence must include at least one counselor"
        )
    if CLIENT_ROLE not in roles_in_sequence:
        raise SpeakerSelectionError("speaker sequence must include at least one client")


def _normalize_participant_roles(
    participant_roles: Mapping[str, ParticipantRoleInput],
) -> dict[str, str]:
    normalized: dict[str, str] = {}
    for speaker_id, role_input in participant_roles.items():
        if not speaker_id:
            raise SpeakerSelectionError("participant speaker_id must not be empty")
        role = (
            role_input
            if isinstance(role_input, str)
            else getattr(role_input, "role", None)
        )
        if role not in _SUPPORTED_PARTICIPANT_ROLES:
            supported = ", ".join(sorted(_SUPPORTED_PARTICIPANT_ROLES))
            raise SpeakerSelectionError(
                f"participant {speaker_id} role must be one of: {supported}"
            )
        normalized[speaker_id] = role
    return normalized
