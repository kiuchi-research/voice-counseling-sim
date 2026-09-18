from __future__ import annotations

from counseling_voice_demo.runtime.streaming_llm import chunk_text_for_tts


def test_chunk_text_for_tts_splits_at_japanese_sentence_boundaries() -> None:
    chunks = chunk_text_for_tts(["今日はよろしくお願いします。", "家族の話をしたいです。"])

    assert chunks == ["今日はよろしくお願いします。", "家族の話をしたいです。"]


def test_chunk_text_for_tts_keeps_short_comma_phrase_until_sentence_end() -> None:
    chunks = chunk_text_for_tts(["そうですね、", "少しずつ整理していきましょう。"])

    assert chunks == ["そうですね、少しずつ整理していきましょう。"]


def test_chunk_text_for_tts_can_split_long_comma_phrase() -> None:
    chunks = chunk_text_for_tts(["家族との距離感についてもう少し詳しく伺いながら、次に進みます。"])

    assert chunks == ["家族との距離感についてもう少し詳しく伺いながら、", "次に進みます。"]


def test_chunk_text_for_tts_splits_long_text_without_punctuation_at_soft_limit() -> None:
    chunks = chunk_text_for_tts(["あ" * 45], soft_max_chars=20)

    assert chunks == ["あ" * 20, "あ" * 20, "あ" * 5]
