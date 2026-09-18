from __future__ import annotations

import re
from collections.abc import AsyncIterator, Iterable


_OPENING_LABEL_WRAPPERS = "「『（(【["
_LABEL_SEPARATORS = ":：,，、。．."
_CLOSING_LABEL_WRAPPERS = "」』）)】]"


def strip_leading_speaker_label(text: str, labels: Iterable[str]) -> str:
    normalized_labels = _normalized_labels(labels)
    if not normalized_labels:
        return text

    stripped = text
    for label in normalized_labels:
        stripped = re.sub(
            rf"^\s*(?:[{re.escape(_OPENING_LABEL_WRAPPERS)}]\s*)?"
            rf"{re.escape(label)}\s*"
            rf"(?:[{re.escape(_LABEL_SEPARATORS)}]|"
            rf"[{re.escape(_CLOSING_LABEL_WRAPPERS)}])\s*",
            "",
            stripped,
            count=1,
        )
    return stripped


async def strip_leading_speaker_label_from_text_parts(
    parts: AsyncIterator[str],
    labels: Iterable[str],
) -> AsyncIterator[str]:
    normalized_labels = _normalized_labels(labels)
    if not normalized_labels:
        async for part in parts:
            yield part
        return

    buffer = ""
    decided = False
    async for part in parts:
        if decided:
            yield part
            continue
        buffer += part
        stripped = strip_leading_speaker_label(buffer, normalized_labels)
        if stripped != buffer:
            decided = True
            if stripped:
                yield stripped
            continue
        if not _could_still_be_leading_label(buffer, normalized_labels):
            decided = True
            if buffer:
                yield buffer

    if not decided and buffer:
        stripped = strip_leading_speaker_label(buffer, normalized_labels)
        if stripped:
            yield stripped


def _normalized_labels(labels: Iterable[str]) -> tuple[str, ...]:
    unique = {
        str(label).strip()
        for label in labels
        if isinstance(label, str) and str(label).strip()
    }
    return tuple(sorted(unique, key=len, reverse=True))


def _could_still_be_leading_label(text: str, labels: tuple[str, ...]) -> bool:
    candidate = text.lstrip()
    if not candidate:
        return True
    if candidate[0] in _OPENING_LABEL_WRAPPERS:
        candidate = candidate[1:].lstrip()
        if not candidate:
            return True

    for label in labels:
        if label.startswith(candidate):
            return True
        if candidate.startswith(label):
            tail = candidate[len(label) :].lstrip()
            if not tail:
                return True
            if tail[0] in f"{_LABEL_SEPARATORS}{_CLOSING_LABEL_WRAPPERS}":
                return True
    return False
