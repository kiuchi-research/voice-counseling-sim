from __future__ import annotations

import pytest

from counseling_voice_demo.runtime.models import ParticipantConfig
from counseling_voice_demo.runtime.speaker_selection import (
    FixedSequenceFloorPredictor,
    FixedSpeakerSequence,
    LEGACY_ALTERNATING_SEQUENCE,
    SpeakerSelectionError,
    speaker_for_fixed_sequence,
)


def test_fixed_speaker_sequence_maps_turns_across_two_clients() -> None:
    selector = FixedSpeakerSequence(
        participant_roles={
            "counselor": "counselor",
            "client_a": "client",
            "client_b": "client",
        },
        sequence=("counselor", "client_a", "counselor", "client_b"),
    )

    assert [selector.speaker_for_turn(turn_id) for turn_id in range(1, 7)] == [
        "counselor",
        "client_a",
        "counselor",
        "client_b",
        "counselor",
        "client_a",
    ]


def test_fixed_speaker_sequence_supports_legacy_alternation() -> None:
    selector = FixedSpeakerSequence(
        participant_roles={"counselor": "counselor", "client": "client"},
        sequence=LEGACY_ALTERNATING_SEQUENCE,
    )

    assert [selector.speaker_for_turn(turn_id) for turn_id in range(1, 5)] == [
        "counselor",
        "client",
        "counselor",
        "client",
    ]


def test_speaker_for_fixed_sequence_accepts_participant_config_values() -> None:
    participants = {
        "counselor": ParticipantConfig(
            speaker_id="counselor",
            role="counselor",
            display_name="カウンセラー",
        ),
        "client_a": ParticipantConfig(
            speaker_id="client_a",
            role="client",
            display_name="クライアントA",
        ),
        "client_b": ParticipantConfig(
            speaker_id="client_b",
            role="client",
            display_name="クライアントB",
        ),
    }

    assert (
        speaker_for_fixed_sequence(
            4,
            participant_roles=participants,
            sequence=("counselor", "client_a", "counselor", "client_b"),
        )
        == "client_b"
    )


@pytest.mark.parametrize(
    ("participant_roles", "sequence", "message"),
    [
        (
            {"counselor": "counselor", "client": "client"},
            (),
            "must not be empty",
        ),
        (
            {"counselor": "counselor", "client": "client"},
            ("counselor", "client_b"),
            "unknown speaker",
        ),
        (
            {"counselor": "counselor", "client": "client"},
            ("client",),
            "at least one counselor",
        ),
        (
            {"counselor": "counselor", "client": "client"},
            ("counselor",),
            "at least one client",
        ),
    ],
)
def test_fixed_speaker_sequence_rejects_invalid_sequences(
    participant_roles: dict[str, str],
    sequence: tuple[str, ...],
    message: str,
) -> None:
    with pytest.raises(SpeakerSelectionError, match=message):
        FixedSpeakerSequence(participant_roles=participant_roles, sequence=sequence)


def test_fixed_speaker_sequence_rejects_non_positive_turn_id() -> None:
    selector = FixedSpeakerSequence(
        participant_roles={"counselor": "counselor", "client": "client"},
        sequence=LEGACY_ALTERNATING_SEQUENCE,
    )

    with pytest.raises(SpeakerSelectionError, match="turn_id must be positive"):
        selector.speaker_for_turn(0)


def test_fixed_sequence_floor_predictor_returns_single_next_candidate() -> None:
    predictor = FixedSequenceFloorPredictor(
        participant_roles={
            "counselor": "counselor",
            "client_a": "client",
            "client_b": "client",
        },
        sequence=("counselor", "client_a", "counselor", "client_b"),
    )

    prediction = predictor.predict(turn_id=4, previous_speaker="counselor")

    assert prediction.candidate_speaker_ids == ("client_b",)
    assert prediction.reason == "fixed_sequence"
    assert prediction.source == "fixed_sequence"


def test_fixed_sequence_floor_predictor_can_return_multiple_unique_candidates() -> None:
    predictor = FixedSequenceFloorPredictor(
        participant_roles={
            "counselor": "counselor",
            "client_a": "client",
            "client_b": "client",
        },
        sequence=("counselor", "client_a", "counselor", "client_b"),
        max_candidates=2,
    )

    prediction = predictor.predict(turn_id=1, previous_speaker="client_a")

    assert prediction.candidate_speaker_ids == ("counselor", "client_b")
    assert prediction.reason == "fixed_sequence_window"
