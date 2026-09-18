from __future__ import annotations

from pathlib import Path

import pytest

from counseling_voice_demo.runtime.models import ParticipantConfig
from counseling_voice_demo.runtime.prompt_sources import (
    PromptSourceReadError,
    resolve_optional_text_source,
    resolve_participant_prompt_sources,
)


def test_resolve_optional_text_source_prefers_source_without_reading_path(
    tmp_path: Path,
) -> None:
    missing_path = tmp_path / "missing.md"

    assert (
        resolve_optional_text_source(
            source="  INLINE_PROMPT_TOKEN  ",
            path=missing_path,
            base_dir=tmp_path,
            label="test prompt",
        )
        == "INLINE_PROMPT_TOKEN"
    )


def test_resolve_optional_text_source_reads_relative_path_from_base_dir(
    tmp_path: Path,
) -> None:
    profile_path = tmp_path / "profiles" / "client.md"
    profile_path.parent.mkdir()
    profile_path.write_text("CLIENT_PROFILE_TOKEN\nsecond line", encoding="utf-8")

    assert (
        resolve_optional_text_source(
            source=None,
            path="profiles/client.md",
            base_dir=tmp_path,
            label="client profile",
        )
        == "CLIENT_PROFILE_TOKEN\nsecond line"
    )


def test_resolve_optional_text_source_normalizes_blank_values() -> None:
    assert resolve_optional_text_source(source="  ", path="  ") is None


def test_resolve_optional_text_source_uses_path_when_source_is_blank(
    tmp_path: Path,
) -> None:
    prompt_path = tmp_path / "prompt.md"
    prompt_path.write_text("PATH_PROMPT_TOKEN", encoding="utf-8")

    assert (
        resolve_optional_text_source(
            source="\n\t",
            path="prompt.md",
            base_dir=tmp_path,
            label="prompt",
        )
        == "PATH_PROMPT_TOKEN"
    )


def test_resolve_optional_text_source_missing_file_mentions_target_path(
    tmp_path: Path,
) -> None:
    with pytest.raises(PromptSourceReadError, match="missing.md"):
        resolve_optional_text_source(
            source=None,
            path="profiles/missing.md",
            base_dir=tmp_path,
            label="client prompt",
        )


def test_resolve_participant_prompt_sources_returns_resolved_participant(
    tmp_path: Path,
) -> None:
    prompt_path = tmp_path / "prompt.md"
    public_path = tmp_path / "public.md"
    private_path = tmp_path / "private.md"
    prompt_path.write_text("PROMPT_FROM_PATH", encoding="utf-8")
    public_path.write_text("PUBLIC_FROM_PATH", encoding="utf-8")
    private_path.write_text("PRIVATE_FROM_PATH", encoding="utf-8")
    participant = ParticipantConfig(
        speaker_id="client_a",
        role="client",
        display_name="Client A",
        prompt_path="prompt.md",
        public_profile_path="public.md",
        private_profile_path="private.md",
    )

    resolved = resolve_participant_prompt_sources(participant, base_dir=tmp_path)

    assert resolved is not participant
    assert resolved.prompt_source == "PROMPT_FROM_PATH"
    assert resolved.public_profile_source == "PUBLIC_FROM_PATH"
    assert resolved.private_profile_source == "PRIVATE_FROM_PATH"
    assert resolved.prompt_path == "prompt.md"
    assert resolved.public_profile_path == "public.md"
    assert resolved.private_profile_path == "private.md"
