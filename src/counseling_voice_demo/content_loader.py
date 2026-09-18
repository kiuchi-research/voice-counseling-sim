from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, ValidationError, field_validator

from counseling_voice_demo.config_loader import ConfigLoadError, _expand_env_vars, load_app_config


ROOT_DIR = Path(__file__).resolve().parents[2]
ROLE_DIR_MAP = {
    "counselor": "counselors",
    "client": "clients",
}


class ContentLoadError(RuntimeError):
    pass


class Profile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    profile_id: str
    role: str
    display_name: str
    image_path: str | None = None
    text_model: str
    temperature: float
    voice_preset: str
    tts_model: str
    tts_voice: str
    tts_instructions: str
    public_profile: str = ""
    prompt: str = ""
    hidden_background: str | None = None

    @field_validator("role")
    @classmethod
    def role_is_supported(cls, value: str) -> str:
        if value not in ROLE_DIR_MAP:
            raise ValueError("role must be counselor or client")
        return value


class Theme(BaseModel):
    model_config = ConfigDict(extra="forbid")

    theme_id: str
    display_name: str
    severity: str
    default_client_profile: str
    body: str


class VoicePreset(BaseModel):
    model_config = ConfigDict(extra="forbid")

    preset_id: str
    display_name: str
    tts_model: str
    fallback_tts_model: str
    voice: str
    fallback_voice: str
    instructions: str
    speed_hint: str
    response_format: str


def parse_front_matter(markdown: str, source_path: Path) -> tuple[dict[str, Any], str]:
    if not markdown.startswith("---\n"):
        raise ContentLoadError(f"YAMLフロントマターが見つかりません: {source_path}")

    try:
        _, front_matter, body = markdown.split("---", 2)
    except ValueError as exc:
        raise ContentLoadError(f"YAMLフロントマターの終端が見つかりません: {source_path}") from exc

    try:
        metadata = yaml.safe_load(front_matter) or {}
    except yaml.YAMLError as exc:
        raise ContentLoadError(f"YAMLフロントマターの解析に失敗しました: {source_path}: {exc}") from exc

    return metadata, body.strip()


def load_profile(profile_path: Path, expected_role: str) -> Profile:
    metadata, body = parse_front_matter(profile_path.read_text(encoding="utf-8"), profile_path)
    metadata = _expand_env_vars(metadata)
    metadata["public_profile"] = body

    try:
        profile = Profile.model_validate(metadata)
    except ValidationError as exc:
        raise ContentLoadError(f"プロフィールの検証に失敗しました: {profile_path}: {exc}") from exc

    if profile.role != expected_role:
        raise ContentLoadError(
            f"プロフィールのroleが配置ディレクトリと一致しません: {profile_path}: expected={expected_role} actual={profile.role}"
        )
    if profile.role == "counselor" and profile.hidden_background is not None:
        raise ContentLoadError(f"カウンセラー役プロフィールにhidden_backgroundは設定できません: {profile_path}")
    return profile


def list_profiles(role: str, profiles_dir: Path | None = None) -> list[Profile]:
    if role not in ROLE_DIR_MAP:
        raise ContentLoadError(f"未対応のプロフィールroleです: {role}")

    if profiles_dir is None:
        config = load_app_config()
        profiles_dir = ROOT_DIR / config.paths.profiles_dir

    role_dir = profiles_dir / ROLE_DIR_MAP[role]
    return sorted(
        (
            load_profile(profile_path, role)
            for profile_path in role_dir.glob("*/profile.md")
        ),
        key=lambda profile: profile.profile_id,
    )


def load_theme(theme_path: Path) -> Theme:
    metadata, body = parse_front_matter(theme_path.read_text(encoding="utf-8"), theme_path)
    metadata["body"] = body

    try:
        return Theme.model_validate(metadata)
    except ValidationError as exc:
        raise ContentLoadError(f"テーマの検証に失敗しました: {theme_path}: {exc}") from exc


def list_themes(themes_dir: Path | None = None) -> list[Theme]:
    if themes_dir is None:
        config = load_app_config()
        themes_dir = ROOT_DIR / config.paths.themes_dir

    return sorted(
        (load_theme(theme_path) for theme_path in themes_dir.glob("*.md")),
        key=lambda theme: theme.theme_id,
    )


def load_voice_preset(preset_path: Path) -> VoicePreset:
    try:
        raw_preset = yaml.safe_load(preset_path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise ContentLoadError(f"音声プリセットのYAML解析に失敗しました: {preset_path}: {exc}") from exc

    try:
        return VoicePreset.model_validate(_expand_env_vars(raw_preset))
    except ValidationError as exc:
        raise ContentLoadError(f"音声プリセットの検証に失敗しました: {preset_path}: {exc}") from exc


def list_voice_presets(voice_presets_dir: Path | None = None) -> list[VoicePreset]:
    if voice_presets_dir is None:
        config = load_app_config()
        voice_presets_dir = ROOT_DIR / config.paths.voice_presets_dir

    return sorted(
        (load_voice_preset(preset_path) for preset_path in voice_presets_dir.glob("*.yaml")),
        key=lambda preset: preset.preset_id,
    )


def main() -> None:
    print(f"counselor_profiles: {len(list_profiles('counselor'))}")
    print(f"client_profiles: {len(list_profiles('client'))}")
    print(f"themes: {len(list_themes())}")
    print(f"voice_presets: {len(list_voice_presets())}")


if __name__ == "__main__":
    main()
