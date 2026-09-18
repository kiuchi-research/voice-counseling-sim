from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = ROOT_DIR / "config" / "app_config.yaml"
ENV_DEFAULT_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*):-([^}]*)\}")


class ConfigLoadError(RuntimeError):
    pass


class AppSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    version: str
    language: str = "ja"
    default_mode: str
    enabled_modes: dict[str, bool]
    session_duration_audio_seconds: int
    closing_start_audio_seconds: int
    farewell_after_turns: int
    optional_wall_clock_hard_stop_seconds: int | None = None
    ahead_generation_turns: int
    show_clinical_disclaimer_in_ui: bool
    speak_clinical_disclaimer: bool

    @field_validator("session_duration_audio_seconds", "closing_start_audio_seconds")
    @classmethod
    def positive_seconds(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("must be a positive integer")
        return value

    @field_validator("ahead_generation_turns")
    @classmethod
    def ahead_generation_turns_is_supported(cls, value: int) -> int:
        if value < 2 or value > 5:
            raise ValueError("must be between 2 and 5")
        return value

    @model_validator(mode="after")
    def validate_app_policy(self) -> AppSection:
        if self.closing_start_audio_seconds >= self.session_duration_audio_seconds:
            raise ValueError("closing_start_audio_seconds must be shorter than session_duration_audio_seconds")
        if self.enabled_modes.get("ai_counselor_ai_client") is not True:
            raise ValueError("ai_counselor_ai_client must be enabled")
        unsupported_enabled = [
            mode
            for mode, enabled in self.enabled_modes.items()
            if enabled
            and mode
            not in {
                "ai_counselor_ai_client",
                "human_counselor_ai_client",
                "ai_counselor_human_client",
            }
        ]
        if unsupported_enabled:
            raise ValueError(f"unsupported modes are enabled: {', '.join(unsupported_enabled)}")
        return self


class UiSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    framework: str
    entrypoint: str
    single_projection_ui: bool
    separate_admin_screen: bool
    show_ai_voice_disclosure_in_ui: bool
    ai_voice_disclosure_text: str
    collapse_unplayed_preview_by_default: bool
    show_future_generated_text_default: str
    settings_popup: bool


class SessionSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    target_audio_seconds: int
    closing_keep_audio_ready_turns: int
    lookahead_turns: int

    @field_validator("closing_keep_audio_ready_turns")
    @classmethod
    def keep_turns_is_non_negative(cls, value: int) -> int:
        if value < 0:
            raise ValueError("must be greater than or equal to 0")
        return value


class OpenAISection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text_generation_endpoint: str
    tts_endpoint: str
    default_text_model: str
    default_text_reasoning_effort: str | None = None
    default_temperature: float
    default_evaluation_model: str
    default_evaluation_reasoning_effort: str | None = None
    default_tts_model: str
    fallback_tts_model: str
    disallowed_tts_models: list[str]
    default_tts_voice: str
    fallback_tts_voice: str
    allowed_tts_voices: list[str]
    default_audio_format: str
    low_latency_audio_format: str
    moderation_endpoint: str
    moderation_model: str
    use_conversation_state: bool
    use_previous_response_id: bool
    use_store_false_when_supported: bool

    @field_validator("default_text_reasoning_effort", "default_evaluation_reasoning_effort")
    @classmethod
    def reasoning_effort_is_supported(cls, value: str | None) -> str | None:
        if value in {None, ""}:
            return None
        if value not in {"low", "medium", "high"}:
            raise ValueError("reasoning_effort must be one of low, medium, high")
        return value


class AudioPlayerSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    framework: str
    standard_audio_supports_python_events: bool
    use_custom_audio_player: bool
    dev_probe_audio_path: str


class AudioMergeSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    create_full_session_mp3: bool
    allow_cancel_full_mp3_creation: bool
    output_filename: str
    include_skipped_partial_in_full_mp3: bool
    include_skipped_unplayed_in_full_mp3: bool


class TurnPolicySection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    invalidate_following_unplayed_on_edit: bool
    invalidate_following_unplayed_on_single_regenerate: bool
    allow_edit_played_turns: bool
    keep_invalidated_revisions: bool


class ReplaySection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enable_replay_mode: bool
    forbid_api_calls_in_replay: bool


class PathsSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    profiles_dir: str
    themes_dir: str
    voice_presets_dir: str
    assets_dir: str
    sessions_dir: str
    replay_sessions_dir: str


class AppConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    app: AppSection
    ui: UiSection
    session: SessionSection
    openai: OpenAISection
    audio_player: AudioPlayerSection
    audio_merge: AudioMergeSection
    turn_policy: TurnPolicySection
    replay: ReplaySection
    paths: PathsSection

    @model_validator(mode="after")
    def validate_cross_section_policy(self) -> AppConfig:
        if self.session.closing_keep_audio_ready_turns != 1:
            raise ValueError("closing_keep_audio_ready_turns must initially be 1")
        if not self.ui.show_ai_voice_disclosure_in_ui:
            raise ValueError("show_ai_voice_disclosure_in_ui must initially be true")
        if self.app.show_clinical_disclaimer_in_ui:
            raise ValueError("show_clinical_disclaimer_in_ui must initially be false")
        if self.app.speak_clinical_disclaimer:
            raise ValueError("speak_clinical_disclaimer must initially be false")
        if self.openai.use_conversation_state:
            raise ValueError("use_conversation_state must initially be false")
        if self.openai.use_previous_response_id:
            raise ValueError("use_previous_response_id must initially be false")
        if not self.openai.use_store_false_when_supported:
            raise ValueError("use_store_false_when_supported must initially be true")
        if not self.turn_policy.invalidate_following_unplayed_on_edit:
            raise ValueError("invalidate_following_unplayed_on_edit must initially be true")
        if not self.turn_policy.invalidate_following_unplayed_on_single_regenerate:
            raise ValueError("invalidate_following_unplayed_on_single_regenerate must initially be true")
        if not self.replay.forbid_api_calls_in_replay:
            raise ValueError("forbid_api_calls_in_replay must initially be true")
        if not self.audio_merge.create_full_session_mp3:
            raise ValueError("create_full_session_mp3 must initially be true")
        if not self.audio_merge.allow_cancel_full_mp3_creation:
            raise ValueError("allow_cancel_full_mp3_creation must initially be true")
        if self.audio_merge.output_filename != "session_full_public.mp3":
            raise ValueError("output_filename must initially be session_full_public.mp3")
        if self.audio_merge.include_skipped_partial_in_full_mp3:
            raise ValueError("include_skipped_partial_in_full_mp3 must initially be false")
        if self.audio_merge.include_skipped_unplayed_in_full_mp3:
            raise ValueError("include_skipped_unplayed_in_full_mp3 must initially be false")
        return self


def _expand_env_vars(value: Any) -> Any:
    if isinstance(value, str):
        expanded = ENV_DEFAULT_PATTERN.sub(
            lambda match: os.environ.get(match.group(1)) or match.group(2),
            value,
        )
        return os.path.expandvars(expanded)
    if isinstance(value, list):
        return [_expand_env_vars(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand_env_vars(item) for key, item in value.items()}
    return value


def load_app_config(config_path: Path = DEFAULT_CONFIG_PATH) -> AppConfig:
    load_dotenv(ROOT_DIR / ".env", override=False)
    try:
        raw_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigLoadError(f"設定ファイルが見つかりません: {config_path}") from exc
    except yaml.YAMLError as exc:
        raise ConfigLoadError(f"設定ファイルのYAML解析に失敗しました: {config_path}: {exc}") from exc

    try:
        return AppConfig.model_validate(_expand_env_vars(raw_config))
    except ValidationError as exc:
        raise ConfigLoadError(f"設定ファイルの検証に失敗しました: {config_path}: {exc}") from exc


def main() -> None:
    config = load_app_config()
    print(f"loaded config: {config.app.name} {config.app.version}")
    print(f"profiles_dir: {config.paths.profiles_dir}")


if __name__ == "__main__":
    main()
