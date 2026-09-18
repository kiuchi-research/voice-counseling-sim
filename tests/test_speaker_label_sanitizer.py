from __future__ import annotations

import asyncio

from counseling_voice_demo.speaker_label_sanitizer import (
    strip_leading_speaker_label,
    strip_leading_speaker_label_from_text_parts,
)


def test_strip_leading_speaker_label_removes_role_label_only_at_start() -> None:
    assert strip_leading_speaker_label("妻：そうですね。", ["妻"]) == "そうですね。"
    assert strip_leading_speaker_label("  夫、少し心配です。", ["夫"]) == "少し心配です。"
    assert strip_leading_speaker_label("【妻】正直しんどいです。", ["妻"]) == "正直しんどいです。"
    assert strip_leading_speaker_label("夫婦で相談したいです。", ["夫"]) == "夫婦で相談したいです。"
    assert strip_leading_speaker_label("妻は少し疲れています。", ["妻"]) == "妻は少し疲れています。"


def test_strip_leading_speaker_label_from_streamed_parts_waits_for_separator() -> None:
    async def parts():
        for part in ["妻", "：", "そう", "ですね。"]:
            yield part

    async def collect() -> list[str]:
        return [
            part
            async for part in strip_leading_speaker_label_from_text_parts(
                parts(),
                ["妻"],
            )
        ]

    assert asyncio.run(collect()) == ["そう", "ですね。"]


def test_strip_leading_speaker_label_from_streamed_parts_keeps_regular_text() -> None:
    async def parts():
        for part in ["妻", "は", "少し", "疲れています。"]:
            yield part

    async def collect() -> list[str]:
        return [
            part
            async for part in strip_leading_speaker_label_from_text_parts(
                parts(),
                ["妻"],
            )
        ]

    assert asyncio.run(collect()) == ["妻は", "少し", "疲れています。"]
