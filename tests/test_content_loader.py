from __future__ import annotations

from pathlib import Path

import pytest

from counseling_voice_demo.content_loader import (
    ContentLoadError,
    list_profiles,
    list_themes,
    list_voice_presets,
    load_profile,
    parse_front_matter,
)


ROOT_DIR = Path(__file__).resolve().parent.parent


def test_list_counselor_and_client_profiles(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_TEXT_MODEL", "")
    monkeypatch.setenv("OPENAI_TTS_MODEL", "")
    profiles_dir = ROOT_DIR / "config" / "profiles"

    counselors = list_profiles("counselor", profiles_dir)
    clients = list_profiles("client", profiles_dir)

    assert [profile.profile_id for profile in counselors] == ["counselor_brief_default"]
    assert [profile.profile_id for profile in clients] == ["client_family_default"]
    assert counselors[0].prompt == ""
    assert "心理支援デモに登場するカウンセラー" in counselors[0].public_profile
    assert "理想の未来像" in counselors[0].public_profile
    assert "家族との距離感に悩む相談者" in clients[0].public_profile
    assert counselors[0].text_model == "gpt-5.1"
    assert counselors[0].tts_model == "gpt-4o-mini-tts"
    assert clients[0].hidden_background is None


def test_profile_role_must_match_directory() -> None:
    profile_path = ROOT_DIR / "config" / "profiles" / "clients" / "client_family_default" / "profile.md"

    with pytest.raises(ContentLoadError, match="roleが配置ディレクトリと一致しません"):
        load_profile(profile_path, "counselor")


def test_profile_rejects_unknown_front_matter_keys(tmp_path: Path) -> None:
    profile_path = tmp_path / "profile.md"
    profile_path.write_text(
        """---
profile_id: counselor_typo
role: counselor
display_name: 佐伯
text_model: ${OPENAI_TEXT_MODEL}
temperature: 0.4
voice_preset: counselor_calm_neutral
tts_model: gpt-4o-mini-tts-2025-12-15
tts_voice: coral
tts_instructions: 落ち着いて話してください。
hiddn_background: typo should fail
---
本文
""",
        encoding="utf-8",
    )

    with pytest.raises(ContentLoadError, match="プロフィールの検証に失敗しました"):
        load_profile(profile_path, "counselor")


def test_counselor_profile_rejects_hidden_background(tmp_path: Path) -> None:
    profile_path = tmp_path / "profile.md"
    profile_path.write_text(
        """---
profile_id: counselor_with_hidden_background
role: counselor
display_name: 佐伯
text_model: ${OPENAI_TEXT_MODEL}
temperature: 0.4
voice_preset: counselor_calm_neutral
tts_model: gpt-4o-mini-tts-2025-12-15
tts_voice: coral
tts_instructions: 落ち着いて話してください。
hidden_background: counselor should not have this
---
本文
""",
        encoding="utf-8",
    )

    with pytest.raises(ContentLoadError, match="hidden_background"):
        load_profile(profile_path, "counselor")


def test_parse_front_matter_requires_front_matter(tmp_path: Path) -> None:
    profile_path = tmp_path / "profile.md"

    with pytest.raises(ContentLoadError, match="YAMLフロントマターが見つかりません"):
        parse_front_matter("本文だけ", profile_path)


def test_list_themes() -> None:
    themes = list_themes(ROOT_DIR / "config" / "themes")

    assert [theme.theme_id for theme in themes] == ["family_conflict"]
    assert themes[0].default_client_profile == "client_family_default"
    assert themes[0].body


def test_list_voice_presets(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_TTS_MODEL", "")
    monkeypatch.setenv("OPENAI_TTS_FALLBACK_MODEL", "")
    presets = list_voice_presets(ROOT_DIR / "config" / "voice_presets")

    assert [preset.preset_id for preset in presets] == [
        "client_anxious_soft",
        "counselor_calm_neutral",
    ]
    assert all(preset.response_format == "mp3" for preset in presets)
    assert all(preset.tts_model == "gpt-4o-mini-tts" for preset in presets)
    assert all(preset.fallback_tts_model == "tts-1" for preset in presets)
