from __future__ import annotations

from pathlib import Path

import pytest

from counseling_voice_demo.client_presets import (
    ClientPresetLoadError,
    list_client_presets,
    load_client_preset,
)


ROOT_DIR = Path(__file__).resolve().parent.parent


def test_baseline_client_preset_contains_both_modes_and_wife_is_client_a() -> None:
    preset = load_client_preset(
        ROOT_DIR / "config" / "client_presets" / "junior_high_school_refusal_couple_v1.yaml"
    )

    assert preset.schema_version == 1
    assert preset.preset_id == "junior_high_school_refusal_couple_v1"
    assert preset.display_name == "中2不登校夫婦（妻）"
    assert preset.read_only is True
    assert preset.modes.one_client == ["client_a"]
    assert preset.modes.two_clients == ["client_a", "client_b"]

    wife = preset.participants["client_a"]
    husband = preset.participants["client_b"]
    assert wife.display_name == "妻"
    assert wife.profile_label == "中2不登校夫婦（妻）"
    assert wife.audio.voice == "marin"
    assert wife.audio.output_speed == 0.9
    assert wife.audio.gain == 0.6
    assert wife.initial_transcript == "今日は中2の娘の対応について相談に来ました"
    assert "毎日のことなので、正直しんどいです" in wife.prompt
    assert husband.display_name == "夫"
    assert husband.audio.voice == "cedar"
    assert husband.audio.output_speed == 1.0
    assert husband.audio.gain == 0.6
    assert husband.initial_transcript == ""
    assert "将来の選択肢が狭くなることが心配です" in husband.prompt
    assert "夫婦が互いの発言を聞き" in preset.shared.prompt
    assert "中学2年生の娘" in preset.shared.public_profile


def test_default_client_preset_directory_is_independent_from_app_config_schema() -> None:
    presets = list_client_presets()

    assert [preset.preset_id for preset in presets] == [
        "junior_high_school_refusal_couple_v1",
        "junior_high_school_refusal_couple_v2_husband",
        "junior_high_school_refusal_couple_v3",
        "junior_high_school_refusal_couple_v4_husband",
    ]


def test_third_client_preset_models_first_interview_for_sons_school_refusal() -> None:
    preset = load_client_preset(
        ROOT_DIR
        / "config"
        / "client_presets"
        / "junior_high_school_refusal_couple_v3.yaml"
    )

    assert preset.preset_id == "junior_high_school_refusal_couple_v3"
    assert preset.display_name == "中2男子不登校夫婦（妻）"
    assert preset.modes.one_client == ["client_a"]
    assert preset.modes.two_clients == ["client_a", "client_b"]
    assert "約4か月前から登校が難しく" in preset.shared.public_profile
    assert "朝になると腹痛や頭痛" in preset.shared.public_profile
    assert "自分のせいで家の雰囲気が悪い" in preset.shared.public_profile
    assert "固定設定と矛盾する事実を作らない" in preset.shared.prompt
    assert "少しずつ変化" in preset.shared.prompt

    wife = preset.participants["client_a"]
    husband = preset.participants["client_b"]
    assert wife.display_name == "妻"
    assert wife.audio.voice == "marin"
    assert wife.audio.output_speed == 0.9
    assert wife.audio.gain == 0.6
    assert wife.initial_transcript.startswith("中学2年生の息子が")
    assert "まず心を休ませてあげたい" in wife.private_profile
    assert "夫婦で関わり方を整理したい" in wife.prompt
    assert husband.display_name == "夫"
    assert husband.audio.voice == "cedar"
    assert husband.audio.output_speed == 1.0
    assert husband.audio.gain == 0.6
    assert husband.initial_transcript == ""
    assert "厳格な家庭" in husband.private_profile
    assert "学校へ戻す方法を知りたい" in husband.prompt


def test_sons_husband_preset_swaps_client_a_and_b_while_reusing_case_content() -> None:
    preset_dir = ROOT_DIR / "config" / "client_presets"
    wife_preset = load_client_preset(
        preset_dir / "junior_high_school_refusal_couple_v3.yaml"
    )
    husband_preset = load_client_preset(
        preset_dir / "junior_high_school_refusal_couple_v4_husband.yaml"
    )

    assert husband_preset.preset_id == "junior_high_school_refusal_couple_v4_husband"
    assert husband_preset.display_name == "中2男子不登校夫婦（夫）"
    assert husband_preset.shared == wife_preset.shared
    assert husband_preset.modes.one_client == ["client_a"]
    assert husband_preset.modes.two_clients == ["client_a", "client_b"]

    husband = husband_preset.participants["client_a"]
    source_husband = wife_preset.participants["client_b"]
    assert husband.display_name == "夫"
    assert husband.profile_label == "中2男子不登校夫婦（夫）"
    assert husband.private_profile == source_husband.private_profile
    assert husband.prompt == source_husband.prompt
    assert husband.audio == source_husband.audio
    assert husband.initial_transcript.startswith("中学2年生の息子が")

    wife = husband_preset.participants["client_b"]
    source_wife = wife_preset.participants["client_a"]
    assert wife.display_name == "妻"
    assert wife.profile_label == "中2男子不登校夫婦（妻）"
    assert wife.private_profile == source_wife.private_profile
    assert wife.prompt == source_wife.prompt
    assert wife.audio == source_wife.audio
    assert wife.initial_transcript == ""


def test_husband_preset_swaps_client_a_and_b_while_reusing_case_content() -> None:
    preset_dir = ROOT_DIR / "config" / "client_presets"
    wife_preset = load_client_preset(
        preset_dir / "junior_high_school_refusal_couple_v1.yaml"
    )
    husband_preset = load_client_preset(
        preset_dir / "junior_high_school_refusal_couple_v2_husband.yaml"
    )

    assert husband_preset.preset_id == "junior_high_school_refusal_couple_v2_husband"
    assert husband_preset.display_name == "中2不登校夫婦（夫）"
    assert husband_preset.shared == wife_preset.shared
    assert husband_preset.modes.one_client == ["client_a"]
    assert husband_preset.modes.two_clients == ["client_a", "client_b"]
    assert husband_preset.participants["client_a"].display_name == "夫"
    assert husband_preset.participants["client_a"].audio.voice == "cedar"
    assert husband_preset.participants["client_a"].initial_transcript == (
        "今日は中2の娘の対応について相談に来ました"
    )
    assert husband_preset.participants["client_b"].display_name == "妻"
    assert husband_preset.participants["client_b"].audio.voice == "marin"
    assert husband_preset.participants["client_b"].initial_transcript == ""


def test_list_client_presets_rejects_duplicate_preset_ids(tmp_path: Path) -> None:
    valid_yaml = """\
schema_version: 1
preset_id: duplicate
display_name: テスト
description: テスト用
read_only: true
shared:
  public_profile: 共通プロフィール
  prompt: 共通プロンプト
participants:
  client_a:
    profile_label: 妻
    display_name: 妻
    private_profile: 妻プロフィール
    prompt: 妻プロンプト
    initial_transcript: 妻の初回発話
    audio:
      voice: marin
      output_speed: 0.9
      gain: 0.6
  client_b:
    profile_label: 夫
    display_name: 夫
    private_profile: 夫プロフィール
    prompt: 夫プロンプト
    initial_transcript: 夫の初回発話
    audio:
      voice: cedar
      output_speed: 1.0
      gain: 0.6
modes:
  one_client: [client_a]
  two_clients: [client_a, client_b]
"""
    (tmp_path / "a.yaml").write_text(valid_yaml, encoding="utf-8")
    (tmp_path / "b.yaml").write_text(valid_yaml, encoding="utf-8")

    with pytest.raises(ClientPresetLoadError, match="preset_idが重複"):
        list_client_presets(tmp_path)


@pytest.mark.parametrize(
    ("replacement", "error_match"),
    [
        ("output_speed: 0.9", "output_speed"),
        ("client_b", "存在しない参加者"),
    ],
)
def test_client_preset_rejects_invalid_values(
    tmp_path: Path,
    replacement: str,
    error_match: str,
) -> None:
    yaml_text = """\
schema_version: 1
preset_id: invalid
display_name: テスト
description: テスト用
read_only: true
shared:
  public_profile: 共通プロフィール
  prompt: 共通プロンプト
participants:
  client_a:
    profile_label: 妻
    display_name: 妻
    private_profile: 妻プロフィール
    prompt: 妻プロンプト
    initial_transcript: 妻の初回発話
    audio:
      voice: marin
      output_speed: 0.9
      gain: 0.6
modes:
  one_client: [client_a]
  two_clients: [client_a, client_a]
"""
    if replacement == "output_speed: 0.9":
        yaml_text = yaml_text.replace(replacement, "output_speed: 1.51")
    else:
        yaml_text = yaml_text.replace(
            "two_clients: [client_a, client_a]",
            "two_clients: [client_a, client_b]",
        )
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml_text, encoding="utf-8")

    with pytest.raises(ClientPresetLoadError, match=error_match):
        load_client_preset(path)


def test_client_preset_rejects_unknown_keys(tmp_path: Path) -> None:
    path = tmp_path / "unknown.yaml"
    path.write_text(
        """\
schema_version: 1
preset_id: invalid
display_name: テスト
description: テスト用
read_only: true
unexpected: true
shared:
  public_profile: 共通プロフィール
  prompt: 共通プロンプト
participants: {}
modes:
  one_client: [client_a]
  two_clients: [client_a, client_b]
""",
        encoding="utf-8",
    )

    with pytest.raises(ClientPresetLoadError, match="検証に失敗"):
        load_client_preset(path)
