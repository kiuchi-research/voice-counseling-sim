from __future__ import annotations

from pathlib import Path
from typing import Self

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from counseling_voice_demo.config_loader import _expand_env_vars
from counseling_voice_demo.runtime.models import (
    MAX_REALTIME_OUTPUT_SPEED,
    MIN_REALTIME_OUTPUT_SPEED,
)


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_CLIENT_PRESETS_DIR = ROOT_DIR / "config" / "client_presets"


class ClientPresetLoadError(RuntimeError):
    pass


class ClientAudioDefaults(BaseModel):
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


class ClientPresetSharedContent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    public_profile: str
    prompt: str

    @field_validator("public_profile", "prompt")
    @classmethod
    def text_is_not_blank(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("preset text must not be blank")
        return normalized


class ClientPresetParticipant(BaseModel):
    model_config = ConfigDict(extra="forbid")

    profile_label: str
    display_name: str
    private_profile: str
    prompt: str
    initial_transcript: str
    audio: ClientAudioDefaults

    @field_validator(
        "profile_label",
        "display_name",
        "private_profile",
        "prompt",
    )
    @classmethod
    def text_is_not_blank(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("participant text must not be blank")
        return normalized

    @field_validator("initial_transcript")
    @classmethod
    def initial_transcript_is_normalized(cls, value: str) -> str:
        return value.strip()


class ClientPresetModes(BaseModel):
    model_config = ConfigDict(extra="forbid")

    one_client: list[str]
    two_clients: list[str]


class ClientPreset(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int
    preset_id: str
    display_name: str
    description: str
    read_only: bool = True
    shared: ClientPresetSharedContent
    participants: dict[str, ClientPresetParticipant]
    modes: ClientPresetModes

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

    @model_validator(mode="after")
    def participant_modes_are_consistent(self) -> Self:
        expected_counts = {"one_client": 1, "two_clients": 2}
        for mode_name, expected_count in expected_counts.items():
            participant_ids = getattr(self.modes, mode_name)
            if len(participant_ids) != expected_count:
                raise ValueError(
                    f"{mode_name}には{expected_count}人の参加者が必要です"
                )
            if len(set(participant_ids)) != len(participant_ids):
                raise ValueError(f"{mode_name}の参加者は重複できません")
            missing = [
                participant_id
                for participant_id in participant_ids
                if participant_id not in self.participants
            ]
            if missing:
                raise ValueError(
                    f"{mode_name}が存在しない参加者を参照しています: "
                    + ", ".join(missing)
                )
        return self

    def participant_ids_for_mode(self, participant_mode: str) -> list[str]:
        if participant_mode == "one_client":
            return list(self.modes.one_client)
        if participant_mode == "two_clients":
            return list(self.modes.two_clients)
        raise ValueError(f"未対応の参加クライアント数です: {participant_mode}")


class DerivedClientPreset(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int
    preset_id: str
    display_name: str
    description: str
    read_only: bool = True
    extends: str
    participant_mapping: dict[str, str]
    initial_transcripts: dict[str, str] = Field(default_factory=dict)

    @field_validator("schema_version")
    @classmethod
    def schema_version_is_supported(cls, value: int) -> int:
        if value != 1:
            raise ValueError("schema_version must be 1")
        return value

    @field_validator(
        "preset_id",
        "display_name",
        "description",
        "extends",
    )
    @classmethod
    def metadata_is_not_blank(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("derived preset metadata must not be blank")
        return normalized


def _read_client_preset_yaml(preset_path: Path) -> dict[str, object]:
    try:
        raw_preset = yaml.safe_load(preset_path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise ClientPresetLoadError(
            f"クライアントプリセットのYAML解析に失敗しました: {preset_path}: {exc}"
        ) from exc
    if not isinstance(raw_preset, dict):
        raise ClientPresetLoadError(
            f"クライアントプリセットのルートはmappingである必要があります: {preset_path}"
        )
    return _expand_env_vars(raw_preset)


def _find_client_preset_path_by_id(
    presets_dir: Path,
    preset_id: str,
) -> Path:
    matches: list[Path] = []
    for candidate_path in sorted(presets_dir.glob("*.yaml")):
        candidate = _read_client_preset_yaml(candidate_path)
        if candidate.get("preset_id") == preset_id:
            matches.append(candidate_path)
    if not matches:
        raise ClientPresetLoadError(
            f"継承元のクライアントプリセットが見つかりません: {preset_id}"
        )
    if len(matches) > 1:
        raise ClientPresetLoadError(
            f"継承元のクライアントプリセットIDが重複しています: {preset_id}"
        )
    return matches[0]


def _load_client_preset(
    preset_path: Path,
    *,
    loading_paths: tuple[Path, ...],
) -> ClientPreset:
    resolved_path = preset_path.resolve()
    if resolved_path in loading_paths:
        inheritance_chain = [*loading_paths, resolved_path]
        raise ClientPresetLoadError(
            "クライアントプリセットの継承が循環しています: "
            + " -> ".join(path.name for path in inheritance_chain)
        )
    raw_preset = _read_client_preset_yaml(preset_path)
    if "extends" not in raw_preset:
        try:
            return ClientPreset.model_validate(raw_preset)
        except ValidationError as exc:
            raise ClientPresetLoadError(
                f"クライアントプリセットの検証に失敗しました: {preset_path}: {exc}"
            ) from exc

    try:
        derived = DerivedClientPreset.model_validate(raw_preset)
    except ValidationError as exc:
        raise ClientPresetLoadError(
            f"クライアントプリセットの検証に失敗しました: {preset_path}: {exc}"
        ) from exc
    base_path = _find_client_preset_path_by_id(preset_path.parent, derived.extends)
    base = _load_client_preset(
        base_path,
        loading_paths=(*loading_paths, resolved_path),
    )
    participant_ids = set(base.participants)
    mapping_targets = set(derived.participant_mapping)
    mapping_sources = set(derived.participant_mapping.values())
    if mapping_targets != participant_ids or mapping_sources != participant_ids:
        raise ClientPresetLoadError(
            "派生クライアントプリセットのparticipant_mappingは、"
            "継承元の参加者を重複なくすべて割り当てる必要があります: "
            f"{preset_path}"
        )
    derived_data = base.model_dump()
    remapped_participants = {
        target_id: base.participants[source_id].model_dump()
        for target_id, source_id in derived.participant_mapping.items()
    }
    unknown_initial_transcript_ids = (
        set(derived.initial_transcripts) - participant_ids
    )
    if unknown_initial_transcript_ids:
        raise ClientPresetLoadError(
            "派生クライアントプリセットのinitial_transcriptsが"
            "存在しない参加者を参照しています: "
            + ", ".join(sorted(unknown_initial_transcript_ids))
        )
    for participant_id, initial_transcript in derived.initial_transcripts.items():
        remapped_participants[participant_id]["initial_transcript"] = (
            initial_transcript.strip()
        )
    derived_data.update(
        {
            "schema_version": derived.schema_version,
            "preset_id": derived.preset_id,
            "display_name": derived.display_name,
            "description": derived.description,
            "read_only": derived.read_only,
            "participants": remapped_participants,
        }
    )
    try:
        return ClientPreset.model_validate(derived_data)
    except ValidationError as exc:
        raise ClientPresetLoadError(
            f"派生クライアントプリセットの検証に失敗しました: {preset_path}: {exc}"
        ) from exc


def load_client_preset(preset_path: Path) -> ClientPreset:
    return _load_client_preset(preset_path, loading_paths=())


def list_client_presets(client_presets_dir: Path | None = None) -> list[ClientPreset]:
    if client_presets_dir is None:
        client_presets_dir = DEFAULT_CLIENT_PRESETS_DIR

    preset_paths = sorted(client_presets_dir.glob("*.yaml"))
    if not preset_paths:
        raise ClientPresetLoadError(
            f"クライアントプリセットが見つかりません: {client_presets_dir}"
        )
    presets = [load_client_preset(preset_path) for preset_path in preset_paths]
    seen_ids: set[str] = set()
    duplicate_ids: set[str] = set()
    for preset in presets:
        if preset.preset_id in seen_ids:
            duplicate_ids.add(preset.preset_id)
        seen_ids.add(preset.preset_id)
    if duplicate_ids:
        raise ClientPresetLoadError(
            "クライアントプリセットのpreset_idが重複しています: "
            + ", ".join(sorted(duplicate_ids))
        )
    return sorted(presets, key=lambda preset: preset.preset_id)
