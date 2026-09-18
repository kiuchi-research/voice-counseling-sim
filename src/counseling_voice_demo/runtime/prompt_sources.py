from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import TypeAlias

from counseling_voice_demo.runtime.models import ParticipantConfig


PathLike: TypeAlias = str | Path
REPO_ROOT = Path(__file__).resolve().parents[3]


class PromptSourceReadError(ValueError):
    """Raised when a configured prompt/profile path cannot be read."""


def resolve_optional_text_source(
    *,
    source: str | None,
    path: PathLike | None,
    base_dir: PathLike | None = None,
    label: str = "text source",
) -> str | None:
    clean_source = _clean_optional(source)
    if clean_source is not None:
        return clean_source

    source_path = _clean_optional(str(path) if path is not None else None)
    if source_path is None:
        return None

    resolved_path = _resolve_path(source_path, base_dir=base_dir)
    try:
        return resolved_path.read_text(encoding="utf-8")
    except OSError as error:
        raise PromptSourceReadError(
            f"failed to read {label} path: {resolved_path}"
        ) from error
    except UnicodeDecodeError as error:
        raise PromptSourceReadError(
            f"failed to decode {label} path as UTF-8: {resolved_path}"
        ) from error


def resolve_participant_prompt_sources(
    participant: ParticipantConfig,
    *,
    base_dir: PathLike | None = None,
) -> ParticipantConfig:
    return replace(
        participant,
        prompt_source=resolve_optional_text_source(
            source=participant.prompt_source,
            path=participant.prompt_path,
            base_dir=base_dir,
            label=f"{participant.speaker_id} prompt",
        ),
        public_profile_source=resolve_optional_text_source(
            source=participant.public_profile_source,
            path=participant.public_profile_path,
            base_dir=base_dir,
            label=f"{participant.speaker_id} public profile",
        ),
        private_profile_source=resolve_optional_text_source(
            source=participant.private_profile_source,
            path=participant.private_profile_path,
            base_dir=base_dir,
            label=f"{participant.speaker_id} secret profile",
        ),
    )


def _resolve_path(path: str, *, base_dir: PathLike | None) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    root = Path(base_dir) if base_dir is not None else REPO_ROOT
    return root / candidate


def _clean_optional(value: str | None) -> str | None:
    if value is None:
        return None
    clean_value = value.strip()
    return clean_value or None


__all__ = [
    "PromptSourceReadError",
    "resolve_optional_text_source",
    "resolve_participant_prompt_sources",
]
