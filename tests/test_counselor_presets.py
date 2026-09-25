from __future__ import annotations

from pathlib import Path

import pytest

from counseling_voice_demo.counselor_presets import (
    CounselorPresetLoadError,
    list_counselor_presets,
    load_counselor_preset,
)


ROOT_DIR = Path(__file__).resolve().parents[1]


def test_default_counselor_preset_loads_profile_prompt_and_audio() -> None:
    preset = load_counselor_preset(
        ROOT_DIR / "config" / "counselor_presets" / "counselor_default.yaml"
    )

    assert preset.preset_id == "counselor_default"
    assert preset.display_name == "カウンセラー（デフォルト）"
    assert preset.counselor.display_name == "カウンセラー"
    assert "2人のクライアントとのファミリーセラピー" in preset.counselor.prompt
    assert "1. あいさつ・ねぎらい（1ターン）" in preset.counselor.prompt
    assert "3. 例外（今できていること、問題が少しはましな場面）の探索" in (
        preset.counselor.prompt
    )
    assert "5. 振り返り" in preset.counselor.prompt
    assert "6. クロージング" in preset.counselor.prompt
    assert "2. 理想の未来像の確認（このフェーズで2ターン以上）" in preset.counselor.prompt
    assert "安易にアドバイスはしない" in preset.counselor.prompt
    assert "気持ち、思い、考え、状況、見通し" in preset.counselor.prompt
    assert "スケーリングクエスチョンの回答が得られたら" in preset.counselor.prompt
    assert "適宜コンプリメント" in preset.counselor.prompt
    assert "質問の末尾には「？」を付ける" in preset.counselor.prompt
    assert "聞き返しの末尾には「？」を付けない" in preset.counselor.prompt
    assert "聞き返しと質問をそれぞれ独立した一文にし、計2文で応答する" in (
        preset.counselor.prompt
    )
    assert "## キー質問" in preset.counselor.prompt
    assert "エピソードについては、表面的に理解せず" in preset.counselor.prompt
    assert preset.counselor.audio.voice == "shimmer"
    assert preset.counselor.audio.output_speed == 1.05
    assert preset.counselor.audio.gain == 0.6


def test_list_counselor_presets_uses_default_directory() -> None:
    presets = list_counselor_presets()

    preset_ids = [preset.preset_id for preset in presets]
    assert {"counselor_default", "counselor_role"}.issubset(preset_ids)
    assert preset_ids == sorted(
        path.stem for path in (ROOT_DIR / "config" / "counselor_presets").glob("*.yaml")
    )


def test_counselor_preset_rejects_unknown_keys(tmp_path: Path) -> None:
    path = tmp_path / "unknown.yaml"
    path.write_text(
        """\
schema_version: 1
preset_id: invalid
display_name: テスト
description: テスト用
read_only: true
unexpected: true
counselor:
  display_name: カウンセラー
  public_profile: 公開プロフィール
  prompt: プロンプト
  audio:
    voice: shimmer
    output_speed: 1.05
    gain: 0.6
""",
        encoding="utf-8",
    )

    with pytest.raises(CounselorPresetLoadError, match="検証に失敗"):
        load_counselor_preset(path)
