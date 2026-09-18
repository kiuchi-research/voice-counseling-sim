from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from counseling_voice_demo.config_loader import _expand_env_vars
from counseling_voice_demo.runtime.models import (
    MAX_REALTIME_OUTPUT_SPEED,
    MIN_REALTIME_OUTPUT_SPEED,
)


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_COUNSELOR_PRESETS_DIR = ROOT_DIR / "config" / "counselor_presets"


class CounselorPresetLoadError(RuntimeError):
    pass


class CounselorAudioDefaults(BaseModel):
    model_config = ConfigDict(extra="forbid")

    voice: str
    output_speed: float = Field(
        ge=MIN_REALTIME_OUTPUT_SPEED,
        le=MAX_REALTIME_OUTPUT_SPEED,
    )
    gain: float = Field(ge=0.1, le=4.0)

    @field_validator("voice")
    @classmethod
    def voice_is_not_blank(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("voice must not be blank")
        return normalized


class CounselorPresetParticipant(BaseModel):
    model_config = ConfigDict(extra="forbid")

    display_name: str
    public_profile: str
    prompt: str
    audio: CounselorAudioDefaults

    @field_validator("display_name", "public_profile", "prompt")
    @classmethod
    def text_is_not_blank(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("counselor preset text must not be blank")
        return normalized


class CounselorPreset(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int
    preset_id: str
    display_name: str
    description: str
    read_only: bool = True
    counselor: CounselorPresetParticipant

    @field_validator("schema_version")
    @classmethod
    def schema_version_is_supported(cls, value: int) -> int:
        if value != 1:
            raise ValueError("schema_version must be 1")
        return value

    @field_validator("preset_id", "display_name", "description")
    @classmethod
    def metadata_is_not_blank(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("preset metadata must not be blank")
        return normalized


def load_counselor_preset(preset_path: Path) -> CounselorPreset:
    try:
        raw_preset = yaml.safe_load(preset_path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise CounselorPresetLoadError(
            f"カウンセラープリセットのYAML解析に失敗しました: {preset_path}: {exc}"
        ) from exc
    if not isinstance(raw_preset, dict):
        raise CounselorPresetLoadError(
            f"カウンセラープリセットのルートはmappingである必要があります: {preset_path}"
        )
    try:
        return CounselorPreset.model_validate(_expand_env_vars(raw_preset))
    except ValidationError as exc:
        raise CounselorPresetLoadError(
            f"カウンセラープリセットの検証に失敗しました: {preset_path}: {exc}"
        ) from exc


def list_counselor_presets(
    counselor_presets_dir: Path | None = None,
) -> list[CounselorPreset]:
    if counselor_presets_dir is None:
        counselor_presets_dir = DEFAULT_COUNSELOR_PRESETS_DIR

    preset_paths = sorted(counselor_presets_dir.glob("*.yaml"))
    if not preset_paths:
        raise CounselorPresetLoadError(
            f"カウンセラープリセットが見つかりません: {counselor_presets_dir}"
        )
    presets = [load_counselor_preset(preset_path) for preset_path in preset_paths]
    seen_ids: set[str] = set()
    duplicate_ids: set[str] = set()
    for preset in presets:
        if preset.preset_id in seen_ids:
            duplicate_ids.add(preset.preset_id)
        seen_ids.add(preset.preset_id)
    if duplicate_ids:
        raise CounselorPresetLoadError(
            "カウンセラープリセットのpreset_idが重複しています: "
            + ", ".join(sorted(duplicate_ids))
        )
    return sorted(presets, key=lambda preset: preset.preset_id)
