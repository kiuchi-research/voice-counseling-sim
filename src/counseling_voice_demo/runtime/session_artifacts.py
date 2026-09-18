from __future__ import annotations

import json
import wave
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class PublicTranscriptTurn:
    turn_id: int
    speaker: str
    text: str
    created_at: str | None
    audio_path: Path | None
    cumulative_audio_seconds: float | None = None
    audio_duration_seconds: float | None = None
    speaker_id: str | None = None
    speaker_display_name: str | None = None
    speaker_role: str | None = None
    transcript_type: str | None = None
    audio_log_path: Path | None = None
    actor_kind: str | None = None


@dataclass(frozen=True)
class CompactSessionAudioConfig:
    human_boundary_silence_seconds: float = 0.8
    human_trailing_boundary_silence_seconds: float = 0.4
    silence_threshold_ratio: float = 0.003
    output_filename: str = "session_compact.wav"


@dataclass(frozen=True)
class CompactSessionAudioResult:
    audio_path: Path
    turns: list[PublicTranscriptTurn]


def list_runtime_sessions(root: Path | str) -> list[Path]:
    sessions_root = Path(root)
    if not sessions_root.exists():
        return []
    return sorted(
        [path for path in sessions_root.iterdir() if path.is_dir()],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )


def load_session_identity(session_dir: Path | str) -> dict[str, Any]:
    events_dir = Path(session_dir) / "internal" / "events"
    for event_path in sorted(events_dir.glob("*.jsonl")):
        with event_path.open(encoding="utf-8") as file:
            for line in file:
                event = _load_jsonl_event(line)
                if (
                    event
                    and event.get("event_type") == "runtime_audio_settings_resolved"
                ):
                    details = event.get("details") or {}
                    return {
                        "interaction_mode": details.get("interaction_mode"),
                        "participants": {
                            item["speaker_id"]: {
                                key: item.get(key)
                                for key in ("role", "actor_kind", "display_name")
                            }
                            for item in details.get("speakers", [])
                            if isinstance(item, dict) and item.get("speaker_id")
                        },
                    }
    return {}


def load_public_transcript_turns(session_dir: Path | str) -> list[PublicTranscriptTurn]:
    session_path = Path(session_dir)
    transcripts_dir = session_path / "internal" / "transcripts"
    if not transcripts_dir.exists():
        return []

    invalidated_transcript_keys = _load_invalidated_transcript_keys(session_path)
    turns_by_id: dict[int, PublicTranscriptTurn] = {}
    priorities_by_id: dict[int, int] = {}
    for transcript_path in sorted(transcripts_dir.glob("*.jsonl")):
        with transcript_path.open(encoding="utf-8") as file:
            for line in file:
                event = _load_jsonl_event(line)
                if event is None:
                    continue
                if _transcript_event_is_invalidated(event, invalidated_transcript_keys):
                    continue
                priority = _public_transcript_priority(event)
                if priority is None:
                    continue
                turn = _public_turn_from_event(event, session_path)
                if turn is not None and priority >= priorities_by_id.get(
                    turn.turn_id,
                    -1,
                ):
                    turns_by_id[turn.turn_id] = turn
                    priorities_by_id[turn.turn_id] = priority

    participants = load_session_identity(session_path).get("participants", {})
    ordered_turns = []
    for turn_id in sorted(turns_by_id):
        turn = turns_by_id[turn_id]
        participant = participants.get(turn.speaker_id or turn.speaker, {})
        ordered_turns.append(
            replace(
                turn,
                speaker_display_name=turn.speaker_display_name
                or participant.get("display_name"),
                speaker_role=turn.speaker_role or participant.get("role"),
                actor_kind=turn.actor_kind or participant.get("actor_kind"),
            )
        )
    return _with_cumulative_audio_seconds(
        ordered_turns,
        session_path=session_path,
    )


def session_realtime_audio_path(session_dir: Path | str) -> Path | None:
    audio_path = Path(session_dir) / "internal" / "audio" / "session_realtime.wav"
    return audio_path if audio_path.exists() else None


def session_has_human_audio_turns(turns: Iterable[PublicTranscriptTurn]) -> bool:
    return any(turn.transcript_type == "human_final" for turn in turns)


def build_compact_session_audio(
    session_dir: Path | str,
    turns: Iterable[PublicTranscriptTurn],
    *,
    config: CompactSessionAudioConfig | None = None,
) -> CompactSessionAudioResult | None:
    session_path = Path(session_dir)
    audio_config = config or CompactSessionAudioConfig()
    ordered_turns = list(turns)
    audio_turns = [turn for turn in ordered_turns if turn.audio_path is not None]
    if not audio_turns:
        return None

    timeline = _load_session_audio_timeline(session_path)
    interrupted_played_seconds = _load_interrupted_played_seconds(session_path)
    raw_segments: list[_CompactAudioSegment] = []
    for turn in audio_turns:
        segment = _compact_audio_segment_for_turn(turn, config=audio_config)
        if segment is None:
            return None
        raw_segments.append(segment)
    if not raw_segments:
        return None
    wav_format = _compact_audio_target_format(raw_segments)
    segments: list[_CompactAudioSegment] = []
    for segment in raw_segments:
        converted_segment = _compact_audio_segment_in_format(
            segment,
            wav_format=wav_format,
            config=audio_config,
        )
        if converted_segment is None:
            return None
        interrupt_seconds = _interrupted_played_seconds_for_turn(
            interrupted_played_seconds,
            converted_segment.turn,
        )
        if (
            _turn_audio_should_trim_for_interrupt(converted_segment.turn)
            and interrupt_seconds is not None
        ):
            converted_segment = _trim_compact_audio_segment(
                converted_segment,
                duration_seconds=interrupt_seconds,
                config=audio_config,
            )
        segments.append(converted_segment)

    sample_rate, sample_width_bytes, channels = wav_format
    output_path = session_path / "internal" / "audio" / audio_config.output_filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    timeline_entries = [
        _timeline_entry_for_turn(timeline, segment.turn) for segment in segments
    ]

    updated_by_turn_id: dict[int, PublicTranscriptTurn] = {}
    output_frame_count = 0
    previous_timeline_end: float | None = None
    with wave.open(str(output_path), "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(sample_width_bytes)
        wav_file.setframerate(sample_rate)
        for index, segment in enumerate(segments):
            timeline_entry = timeline_entries[index]
            original_start = timeline_entry[0] if timeline_entry is not None else None
            original_end = timeline_entry[1] if timeline_entry is not None else None
            original_gap_seconds = 0.0
            if (
                previous_timeline_end is not None
                and original_start is not None
                and original_start > previous_timeline_end
            ):
                original_gap_seconds = original_start - previous_timeline_end
            gap_seconds = _compact_inter_turn_gap_seconds(
                original_gap_seconds,
                previous_turn=segments[index - 1].turn if index > 0 else None,
                current_turn=segment.turn,
            )
            if gap_seconds > 0:
                gap_frames = round(gap_seconds * sample_rate)
                if gap_frames > 0:
                    wav_file.writeframes(
                        _silent_pcm_frames(
                            gap_frames,
                            sample_width_bytes=sample_width_bytes,
                            channels=channels,
                        )
                    )
                    output_frame_count += gap_frames

            leading_pad_seconds = _compact_human_leading_pad_seconds(
                segment,
                preceding_gap_seconds=gap_seconds,
                config=audio_config,
            )
            trailing_pad_seconds = _compact_human_trailing_pad_seconds(
                segment,
                following_gap_seconds=_compact_following_gap_seconds(
                    index,
                    segments=segments,
                    timeline_entries=timeline_entries,
                ),
                config=audio_config,
            )
            compact_start = output_frame_count / sample_rate
            leading_pad_frames = round(leading_pad_seconds * sample_rate)
            if leading_pad_frames > 0:
                wav_file.writeframes(
                    _silent_pcm_frames(
                        leading_pad_frames,
                        sample_width_bytes=sample_width_bytes,
                        channels=channels,
                    )
                )
                output_frame_count += leading_pad_frames
            wav_file.writeframes(segment.pcm)
            output_frame_count += segment.frame_count
            trailing_pad_frames = round(trailing_pad_seconds * sample_rate)
            if trailing_pad_frames > 0:
                wav_file.writeframes(
                    _silent_pcm_frames(
                        trailing_pad_frames,
                        sample_width_bytes=sample_width_bytes,
                        channels=channels,
                    )
                )
                output_frame_count += trailing_pad_frames
            compact_end = output_frame_count / sample_rate
            updated_by_turn_id[segment.turn.turn_id] = replace(
                segment.turn,
                cumulative_audio_seconds=compact_start,
                audio_duration_seconds=max(0.0, compact_end - compact_start),
            )
            if (
                original_start is not None
                and _turn_audio_should_trim_for_interrupt(segment.turn)
                and _interrupted_played_seconds_for_turn(
                    interrupted_played_seconds,
                    segment.turn,
                )
                is not None
            ):
                previous_timeline_end = original_start + segment.duration_seconds
            elif original_end is not None:
                previous_timeline_end = original_end
            elif original_start is not None:
                previous_timeline_end = original_start + segment.duration_seconds
            else:
                previous_timeline_end = None

    updated_turns: list[PublicTranscriptTurn] = []
    compact_cursor_seconds = 0.0
    for turn in ordered_turns:
        updated_turn = updated_by_turn_id.get(turn.turn_id)
        if updated_turn is not None:
            updated_turns.append(updated_turn)
            compact_cursor_seconds = (
                float(updated_turn.cumulative_audio_seconds or 0.0)
                + float(updated_turn.audio_duration_seconds or 0.0)
            )
            continue
        updated_turns.append(
            replace(
                turn,
                cumulative_audio_seconds=compact_cursor_seconds,
                audio_duration_seconds=None,
            )
        )
    return CompactSessionAudioResult(audio_path=output_path, turns=updated_turns)


def format_public_script(
    turns: Iterable[PublicTranscriptTurn],
    format: str = "markdown",
) -> str:
    if format not in {"markdown", "txt"}:
        raise ValueError("format must be 'markdown' or 'txt'")

    lines: list[str] = []
    for turn in turns:
        text = turn.text.strip()
        if not text:
            continue
        speaker_label = turn.speaker_display_name or turn.speaker
        prefix = "- " if format == "markdown" else ""
        lines.append(f"{prefix}{_format_turn_time(turn)} {speaker_label}: {text}")
    return "\n".join(lines)


def _load_jsonl_event(line: str) -> dict[str, Any] | None:
    if not line.strip():
        return None
    try:
        event = json.loads(line)
    except json.JSONDecodeError:
        return None
    return event if isinstance(event, dict) else None


def _load_invalidated_transcript_keys(
    session_path: Path,
) -> set[tuple[int, str | None]]:
    events_dir = session_path / "internal" / "events"
    if not events_dir.exists():
        return set()
    invalidated: set[tuple[int, str | None]] = set()
    for events_path in sorted(events_dir.glob("*.jsonl")):
        with events_path.open(encoding="utf-8") as file:
            for line in file:
                event = _load_jsonl_event(line)
                if event is None:
                    continue
                if event.get("event_type") != "barge_in_future_turns_invalidated":
                    continue
                details = event.get("details")
                if not isinstance(details, dict):
                    continue
                invalidated_turns = details.get("invalidated_turns")
                structured_keys_added = False
                if isinstance(invalidated_turns, list):
                    for item in invalidated_turns:
                        if not isinstance(item, dict):
                            continue
                        key = _turn_speaker_key_from_mapping(item)
                        if key is not None:
                            invalidated.add(key)
                            structured_keys_added = True
                if structured_keys_added:
                    continue
                invalidated_turn_ids = details.get("invalidated_turn_ids")
                if isinstance(invalidated_turn_ids, list):
                    for value in invalidated_turn_ids:
                        try:
                            invalidated.add((int(value), None))
                        except (TypeError, ValueError):
                            continue
    return invalidated


def _turn_speaker_key_from_mapping(
    event: dict[str, Any],
) -> tuple[int, str | None] | None:
    try:
        turn_id = int(event["turn_id"])
    except (KeyError, TypeError, ValueError):
        return None
    speaker = event.get("speaker_id")
    if not isinstance(speaker, str):
        speaker = event.get("speaker")
    if not isinstance(speaker, str):
        speaker = None
    return (turn_id, speaker)


def _transcript_event_is_invalidated(
    event: dict[str, Any],
    invalidated_transcript_keys: set[tuple[int, str | None]],
) -> bool:
    if not invalidated_transcript_keys:
        return False
    key = _turn_speaker_key_from_mapping(event)
    if key is None:
        return False
    return key in invalidated_transcript_keys or (key[0], None) in invalidated_transcript_keys


def _public_transcript_priority(event: dict[str, Any]) -> int | None:
    transcript_type = event.get("transcript_type")
    if transcript_type in {"generated_final", "generated_interrupted", "human_final"}:
        return 2
    if transcript_type == "final":
        return 1
    return None


def _public_turn_from_event(
    event: dict[str, Any],
    session_path: Path,
) -> PublicTranscriptTurn | None:
    try:
        turn_id = int(event["turn_id"])
    except (KeyError, TypeError, ValueError):
        return None
    speaker_id = event.get("speaker_id")
    if not isinstance(speaker_id, str):
        speaker_id = None
    speaker = event.get("speaker")
    if not isinstance(speaker, str):
        speaker = None
    text = event.get("text")
    speaker_key = speaker if speaker is not None else speaker_id
    if speaker_key is None or not isinstance(text, str):
        return None
    created_at = event.get("created_at")
    if created_at is not None and not isinstance(created_at, str):
        created_at = str(created_at)
    audio_log_path = _event_audio_log_path(event, session_path)
    audio_path = audio_log_path or _turn_audio_path(
        session_path,
        turn_id=turn_id,
        speaker_keys=_unique_strings(speaker_id, speaker),
    )
    return PublicTranscriptTurn(
        turn_id=turn_id,
        speaker=speaker_key,
        text=text,
        created_at=created_at,
        audio_path=audio_path,
        speaker_id=speaker_id,
        speaker_display_name=_speaker_display_name_from_event(event),
        speaker_role=_speaker_role_from_event(event),
        transcript_type=str(event.get("transcript_type")),
        audio_log_path=audio_log_path,
        actor_kind=(event.get("metadata") or {}).get("actor_kind")
        or ("human" if event.get("transcript_type") == "human_final" else None),
    )


def _event_audio_log_path(event: dict[str, Any], session_path: Path) -> Path | None:
    for raw_audio_path in _event_audio_path_values(event):
        audio_path = _resolve_existing_audio_path(raw_audio_path, session_path)
        if audio_path is not None:
            return audio_path
    return None


def _event_audio_path_values(event: dict[str, Any]) -> tuple[Any, ...]:
    values: list[Any] = [
        event.get("audio_log_path"),
        event.get("source_audio_ref"),
    ]
    metadata = event.get("metadata")
    if isinstance(metadata, dict):
        values.extend(
            [
                metadata.get("audio_log_path"),
                metadata.get("source_audio_ref"),
            ]
        )
    return tuple(values)


def _resolve_existing_audio_path(raw_audio_path: Any, session_path: Path) -> Path | None:
    if not isinstance(raw_audio_path, str) or not raw_audio_path.strip():
        return None
    audio_path = Path(raw_audio_path.strip())
    if not audio_path.is_absolute():
        audio_path = session_path / audio_path
    return audio_path if audio_path.exists() else None


def _turn_audio_path(
    session_path: Path,
    *,
    turn_id: int,
    speaker_keys: Iterable[str],
) -> Path | None:
    for speaker_key in speaker_keys:
        audio_path = (
            session_path
            / "internal"
            / "audio"
            / f"turn_{turn_id:04d}_{speaker_key}.wav"
        )
        if audio_path.exists():
            return audio_path
    return None


def _speaker_display_name_from_event(event: dict[str, Any]) -> str | None:
    metadata = event.get("metadata")
    if not isinstance(metadata, dict):
        return None
    display_name = metadata.get("speaker_display_name")
    if not isinstance(display_name, str):
        return None
    display_name = display_name.strip()
    return display_name or None


def _speaker_role_from_event(event: dict[str, Any]) -> str | None:
    metadata = event.get("metadata")
    if not isinstance(metadata, dict):
        return None
    role = metadata.get("role")
    if not isinstance(role, str):
        return None
    role = role.strip()
    return role or None


def _with_cumulative_audio_seconds(
    turns: list[PublicTranscriptTurn],
    *,
    session_path: Path,
) -> list[PublicTranscriptTurn]:
    timeline = _load_session_audio_timeline(session_path)
    cumulative_seconds = 0.0
    updated_turns: list[PublicTranscriptTurn] = []
    for turn in turns:
        timeline_entry = _timeline_entry_for_turn(timeline, turn)
        if timeline_entry is not None:
            start_seconds, end_seconds = timeline_entry
            updated_turns.append(
                replace(
                    turn,
                    cumulative_audio_seconds=start_seconds,
                    audio_duration_seconds=max(0.0, end_seconds - start_seconds),
                )
            )
            cumulative_seconds = end_seconds
            continue
        # If a timeline is partial, continue from the last known session-audio end.
        duration_seconds = _wav_duration_seconds(turn.audio_path)
        updated_turns.append(
            replace(
                turn,
                cumulative_audio_seconds=cumulative_seconds,
                audio_duration_seconds=duration_seconds if duration_seconds > 0 else None,
            )
        )
        cumulative_seconds += duration_seconds
    return updated_turns


def _load_session_audio_timeline(
    session_path: Path,
) -> dict[tuple[int, str], tuple[float, float]]:
    timeline_path = session_path / "internal" / "audio" / "session_timeline.jsonl"
    if not timeline_path.exists():
        return {}
    timeline: dict[tuple[int, str], tuple[float, float]] = {}
    with timeline_path.open(encoding="utf-8") as file:
        for line in file:
            event = _load_jsonl_event(line)
            if event is None:
                continue
            try:
                turn_id = int(event["turn_id"])
            except (KeyError, TypeError, ValueError):
                continue
            speakers = _unique_strings(event.get("speaker_id"), event.get("speaker"))
            start_seconds = _nonnegative_float(event.get("start_seconds"))
            end_seconds = _nonnegative_float(event.get("end_seconds"))
            if not speakers or start_seconds is None:
                continue
            if end_seconds is None or end_seconds < start_seconds:
                end_seconds = start_seconds
            for speaker in speakers:
                timeline[(turn_id, speaker)] = (start_seconds, end_seconds)
    return timeline


def _load_interrupted_played_seconds(
    session_path: Path,
) -> dict[tuple[int, str], float]:
    events_dir = session_path / "internal" / "events"
    if not events_dir.exists():
        return {}
    played_seconds_by_turn: dict[tuple[int, str], float] = {}
    pending_active_response_stops: list[tuple[float | None, float]] = []
    for events_path in sorted(events_dir.glob("*.jsonl")):
        with events_path.open(encoding="utf-8") as file:
            for line in file:
                event = _load_jsonl_event(line)
                if event is None:
                    continue
                event_type = event.get("event_type")
                if event_type in {
                    "generated_interrupted",
                    "ai_generation_interrupted_for_human_barge_in",
                }:
                    _apply_pending_active_response_stop(
                        pending_active_response_stops,
                        played_seconds_by_turn,
                        event,
                    )
                    continue
                if event_type != "realtime_playback_interrupted":
                    continue
                details = event.get("details")
                if not isinstance(details, dict):
                    details = {}
                played_ms = _nonnegative_float(details.get("played_ms"))
                if played_ms is None:
                    played_ms = _nonnegative_float(event.get("played_ms"))
                if played_ms is not None:
                    _add_interrupted_played_seconds(
                        played_seconds_by_turn,
                        event,
                        played_ms=played_ms,
                    )
                    result = details.get("result")
                    if isinstance(result, dict):
                        _add_interrupted_played_seconds(
                            played_seconds_by_turn,
                            result,
                            played_ms=played_ms,
                        )
                active_response_stop = details.get("active_response_stop")
                if isinstance(active_response_stop, dict):
                    active_played_ms = _nonnegative_float(
                        active_response_stop.get("played_ms")
                    )
                    if active_played_ms is None:
                        active_played_ms = 0.0
                    added = _add_interrupted_played_seconds(
                        played_seconds_by_turn,
                        active_response_stop,
                        played_ms=active_played_ms,
                    )
                    if not added:
                        pending_active_response_stops.append(
                            (
                                _nonnegative_float(event.get("monotonic_time")),
                                max(0.0, active_played_ms / 1000.0),
                            )
                        )
    return played_seconds_by_turn


def _add_interrupted_played_seconds(
    played_seconds_by_turn: dict[tuple[int, str], float],
    event: dict[str, Any],
    *,
    played_ms: float,
) -> bool:
    key = _turn_speaker_key_from_mapping(event)
    if key is None or key[1] is None:
        return False
    seconds = max(0.0, played_ms / 1000.0)
    existing = played_seconds_by_turn.get((key[0], key[1]))
    if existing is None or seconds < existing:
        played_seconds_by_turn[(key[0], key[1])] = seconds
    return True


def _apply_pending_active_response_stop(
    pending_active_response_stops: list[tuple[float | None, float]],
    played_seconds_by_turn: dict[tuple[int, str], float],
    event: dict[str, Any],
) -> None:
    key = _turn_speaker_key_from_mapping(event)
    if key is None or key[1] is None:
        return
    if (key[0], key[1]) in played_seconds_by_turn:
        return
    event_monotonic = _nonnegative_float(event.get("monotonic_time"))
    match_index: int | None = None
    for index in range(len(pending_active_response_stops) - 1, -1, -1):
        stop_monotonic, _played_seconds = pending_active_response_stops[index]
        if (
            event_monotonic is not None
            and stop_monotonic is not None
            and event_monotonic - stop_monotonic > 5.0
        ):
            continue
        match_index = index
        break
    if match_index is None:
        return
    _stop_monotonic, played_seconds = pending_active_response_stops.pop(match_index)
    played_seconds_by_turn[(key[0], key[1])] = played_seconds


def _interrupted_played_seconds_for_turn(
    played_seconds_by_turn: dict[tuple[int, str], float],
    turn: PublicTranscriptTurn,
) -> float | None:
    for speaker in _unique_strings(turn.speaker_id, turn.speaker):
        played_seconds = played_seconds_by_turn.get((turn.turn_id, speaker))
        if played_seconds is not None:
            return played_seconds
    return None


def _turn_audio_should_trim_for_interrupt(turn: PublicTranscriptTurn) -> bool:
    return turn.transcript_type in {"generated_final", "generated_interrupted"}


def _timeline_entry_for_turn(
    timeline: dict[tuple[int, str], tuple[float, float]],
    turn: PublicTranscriptTurn,
) -> tuple[float, float] | None:
    for speaker in _unique_strings(turn.speaker_id, turn.speaker):
        timeline_entry = timeline.get((turn.turn_id, speaker))
        if timeline_entry is not None:
            return timeline_entry
    return None


def _unique_strings(*values: Any) -> tuple[str, ...]:
    strings: list[str] = []
    for value in values:
        if isinstance(value, str) and value not in strings:
            strings.append(value)
    return tuple(strings)


def _nonnegative_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return max(0.0, number)


def _wav_duration_seconds(audio_path: Path | None) -> float:
    if audio_path is None:
        return 0.0
    try:
        with wave.open(str(audio_path), "rb") as wav_file:
            frame_rate = wav_file.getframerate()
            if frame_rate <= 0:
                return 0.0
            return wav_file.getnframes() / frame_rate
    except (EOFError, OSError, wave.Error):
        return 0.0


def _format_turn_time(turn: PublicTranscriptTurn) -> str:
    if turn.cumulative_audio_seconds is not None:
        return _format_cumulative_audio_time(turn.cumulative_audio_seconds)
    return _format_time(turn.created_at)


def _format_cumulative_audio_time(seconds: float) -> str:
    total_seconds = max(0, int(seconds))
    minutes, remaining_seconds = divmod(total_seconds, 60)
    return f"{minutes:02d}:{remaining_seconds:02d}"


def _format_time(created_at: str | None) -> str:
    if not created_at:
        return "--:--"
    try:
        parsed = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
    except ValueError:
        return "--:--"
    return parsed.strftime("%H:%M:%S")


@dataclass(frozen=True)
class _CompactAudioSegment:
    turn: PublicTranscriptTurn
    pcm: bytes
    wav_format: tuple[int, int, int]
    frame_count: int
    duration_seconds: float
    leading_silence_seconds: float
    trailing_silence_seconds: float


def _compact_audio_segment_for_turn(
    turn: PublicTranscriptTurn,
    *,
    config: CompactSessionAudioConfig,
) -> _CompactAudioSegment | None:
    if turn.audio_path is None:
        return None
    try:
        with wave.open(str(turn.audio_path), "rb") as wav_file:
            channels = wav_file.getnchannels()
            sample_width_bytes = wav_file.getsampwidth()
            sample_rate = wav_file.getframerate()
            frame_count = wav_file.getnframes()
            pcm = wav_file.readframes(frame_count)
    except (EOFError, OSError, wave.Error):
        return None
    if channels <= 0 or sample_width_bytes <= 0 or sample_rate <= 0:
        return None
    duration_seconds = frame_count / sample_rate
    leading_silence, trailing_silence = _pcm_edge_silence_seconds(
        pcm,
        frame_count=frame_count,
        sample_rate=sample_rate,
        sample_width_bytes=sample_width_bytes,
        channels=channels,
        silence_threshold_ratio=config.silence_threshold_ratio,
    )
    return _CompactAudioSegment(
        turn=turn,
        pcm=pcm,
        wav_format=(sample_rate, sample_width_bytes, channels),
        frame_count=frame_count,
        duration_seconds=duration_seconds,
        leading_silence_seconds=leading_silence,
        trailing_silence_seconds=trailing_silence,
    )


def _compact_audio_target_format(
    segments: list[_CompactAudioSegment],
) -> tuple[int, int, int]:
    for segment in segments:
        if segment.turn.transcript_type != "human_final":
            return segment.wav_format
    return segments[0].wav_format


def _compact_audio_segment_in_format(
    segment: _CompactAudioSegment,
    *,
    wav_format: tuple[int, int, int],
    config: CompactSessionAudioConfig,
) -> _CompactAudioSegment | None:
    if segment.wav_format == wav_format:
        return segment
    converted = _convert_pcm_format(
        segment.pcm,
        source_format=segment.wav_format,
        target_format=wav_format,
    )
    if converted is None:
        return None
    pcm, frame_count = converted
    sample_rate, sample_width_bytes, channels = wav_format
    leading_silence, trailing_silence = _pcm_edge_silence_seconds(
        pcm,
        frame_count=frame_count,
        sample_rate=sample_rate,
        sample_width_bytes=sample_width_bytes,
        channels=channels,
        silence_threshold_ratio=config.silence_threshold_ratio,
    )
    return replace(
        segment,
        pcm=pcm,
        wav_format=wav_format,
        frame_count=frame_count,
        duration_seconds=frame_count / sample_rate if sample_rate > 0 else 0.0,
        leading_silence_seconds=leading_silence,
        trailing_silence_seconds=trailing_silence,
    )


def _trim_compact_audio_segment(
    segment: _CompactAudioSegment,
    *,
    duration_seconds: float,
    config: CompactSessionAudioConfig,
) -> _CompactAudioSegment:
    sample_rate, sample_width_bytes, channels = segment.wav_format
    if sample_rate <= 0 or sample_width_bytes <= 0 or channels <= 0:
        return segment
    target_frame_count = max(0, round(max(0.0, duration_seconds) * sample_rate))
    if target_frame_count >= segment.frame_count:
        return segment
    bytes_per_frame = sample_width_bytes * channels
    pcm = segment.pcm[: target_frame_count * bytes_per_frame]
    leading_silence, trailing_silence = _pcm_edge_silence_seconds(
        pcm,
        frame_count=target_frame_count,
        sample_rate=sample_rate,
        sample_width_bytes=sample_width_bytes,
        channels=channels,
        silence_threshold_ratio=config.silence_threshold_ratio,
    )
    return replace(
        segment,
        pcm=pcm,
        frame_count=target_frame_count,
        duration_seconds=target_frame_count / sample_rate,
        leading_silence_seconds=leading_silence,
        trailing_silence_seconds=trailing_silence,
    )


def _convert_pcm_format(
    pcm: bytes,
    *,
    source_format: tuple[int, int, int],
    target_format: tuple[int, int, int],
) -> tuple[bytes, int] | None:
    source_rate, source_width_bytes, source_channels = source_format
    target_rate, target_width_bytes, target_channels = target_format
    if (
        source_rate <= 0
        or target_rate <= 0
        or source_channels <= 0
        or target_channels <= 0
    ):
        return None
    if source_width_bytes != 2 or target_width_bytes != 2:
        return None
    source_frame_bytes = source_width_bytes * source_channels
    if source_frame_bytes <= 0:
        return None
    source_frame_count = len(pcm) // source_frame_bytes
    if source_frame_count <= 0:
        return b"", 0
    target_frame_count = max(
        1,
        round(source_frame_count * target_rate / source_rate),
    )
    output = bytearray(target_frame_count * target_channels * target_width_bytes)
    output_offset = 0
    for target_frame_index in range(target_frame_count):
        source_position = target_frame_index * source_rate / target_rate
        left_frame = min(source_frame_count - 1, int(source_position))
        right_frame = min(source_frame_count - 1, left_frame + 1)
        fraction = source_position - left_frame
        for target_channel in range(target_channels):
            left_sample = _source_sample_for_target_channel(
                pcm,
                frame_index=left_frame,
                target_channel=target_channel,
                source_channels=source_channels,
                target_channels=target_channels,
            )
            right_sample = _source_sample_for_target_channel(
                pcm,
                frame_index=right_frame,
                target_channel=target_channel,
                source_channels=source_channels,
                target_channels=target_channels,
            )
            sample = round(left_sample + (right_sample - left_sample) * fraction)
            output[output_offset : output_offset + 2] = _int16_bytes(sample)
            output_offset += 2
    return bytes(output), target_frame_count


def _source_sample_for_target_channel(
    pcm: bytes,
    *,
    frame_index: int,
    target_channel: int,
    source_channels: int,
    target_channels: int,
) -> int:
    if source_channels == 1:
        return _int16_sample_at(
            pcm,
            frame_index=frame_index,
            channel_index=0,
            channels=source_channels,
        )
    if target_channels == source_channels and target_channel < source_channels:
        return _int16_sample_at(
            pcm,
            frame_index=frame_index,
            channel_index=target_channel,
            channels=source_channels,
        )
    total = 0
    for source_channel in range(source_channels):
        total += _int16_sample_at(
            pcm,
            frame_index=frame_index,
            channel_index=source_channel,
            channels=source_channels,
        )
    return round(total / source_channels)


def _int16_sample_at(
    pcm: bytes,
    *,
    frame_index: int,
    channel_index: int,
    channels: int,
) -> int:
    offset = (frame_index * channels + channel_index) * 2
    return int.from_bytes(pcm[offset : offset + 2], "little", signed=True)


def _int16_bytes(value: int) -> bytes:
    clamped = max(-32768, min(32767, int(value)))
    return clamped.to_bytes(2, "little", signed=True)


def _compact_inter_turn_gap_seconds(
    original_gap_seconds: float,
    *,
    previous_turn: PublicTranscriptTurn | None = None,
    current_turn: PublicTranscriptTurn | None = None,
) -> float:
    original_gap_seconds = max(0.0, original_gap_seconds)
    if (
        previous_turn is not None
        and previous_turn.transcript_type == "human_final"
    ) or (
        current_turn is not None
        and current_turn.transcript_type == "human_final"
    ):
        return 0.0
    return original_gap_seconds


def _compact_human_leading_pad_seconds(
    segment: _CompactAudioSegment,
    *,
    preceding_gap_seconds: float,
    config: CompactSessionAudioConfig,
) -> float:
    if segment.turn.transcript_type != "human_final":
        return 0.0
    minimum_silence = max(0.0, config.human_boundary_silence_seconds)
    available_silence = max(0.0, preceding_gap_seconds) + max(
        0.0,
        segment.leading_silence_seconds,
    )
    return max(0.0, minimum_silence - available_silence)


def _compact_human_trailing_pad_seconds(
    segment: _CompactAudioSegment,
    *,
    following_gap_seconds: float,
    config: CompactSessionAudioConfig,
) -> float:
    if segment.turn.transcript_type != "human_final":
        return 0.0
    minimum_silence = max(0.0, config.human_trailing_boundary_silence_seconds)
    available_silence = max(0.0, segment.trailing_silence_seconds) + max(
        0.0,
        following_gap_seconds,
    )
    return max(0.0, minimum_silence - available_silence)


def _compact_following_gap_seconds(
    index: int,
    *,
    segments: list[_CompactAudioSegment],
    timeline_entries: list[tuple[float, float] | None],
) -> float:
    if index + 1 >= len(segments):
        return 0.0
    current_entry = timeline_entries[index]
    next_entry = timeline_entries[index + 1]
    if current_entry is None or next_entry is None:
        return 0.0
    original_gap_seconds = max(0.0, next_entry[0] - current_entry[1])
    return _compact_inter_turn_gap_seconds(
        original_gap_seconds,
        previous_turn=segments[index].turn,
        current_turn=segments[index + 1].turn,
    )


def _pcm_edge_silence_seconds(
    pcm: bytes,
    *,
    frame_count: int,
    sample_rate: int,
    sample_width_bytes: int,
    channels: int,
    silence_threshold_ratio: float,
) -> tuple[float, float]:
    if frame_count <= 0 or sample_rate <= 0:
        return 0.0, 0.0
    bytes_per_frame = sample_width_bytes * channels
    if bytes_per_frame <= 0:
        return 0.0, 0.0
    threshold = _pcm_silence_threshold(
        sample_width_bytes,
        silence_threshold_ratio=silence_threshold_ratio,
    )
    leading_frames = 0
    for frame_index in range(frame_count):
        offset = frame_index * bytes_per_frame
        if not _pcm_frame_is_silent(
            pcm,
            offset,
            sample_width_bytes=sample_width_bytes,
            channels=channels,
            silence_threshold=threshold,
        ):
            break
        leading_frames += 1
    if leading_frames >= frame_count:
        duration_seconds = frame_count / sample_rate
        return duration_seconds, duration_seconds

    trailing_frames = 0
    for frame_index in range(frame_count - 1, -1, -1):
        offset = frame_index * bytes_per_frame
        if not _pcm_frame_is_silent(
            pcm,
            offset,
            sample_width_bytes=sample_width_bytes,
            channels=channels,
            silence_threshold=threshold,
        ):
            break
        trailing_frames += 1
    return leading_frames / sample_rate, trailing_frames / sample_rate


def _pcm_silence_threshold(
    sample_width_bytes: int,
    *,
    silence_threshold_ratio: float,
) -> int:
    if sample_width_bytes <= 1:
        full_scale = 128
    else:
        full_scale = (1 << (sample_width_bytes * 8 - 1)) - 1
    return max(1, round(full_scale * max(0.0, silence_threshold_ratio)))


def _pcm_frame_is_silent(
    pcm: bytes,
    offset: int,
    *,
    sample_width_bytes: int,
    channels: int,
    silence_threshold: int,
) -> bool:
    for channel_index in range(channels):
        sample_offset = offset + channel_index * sample_width_bytes
        sample = pcm[sample_offset : sample_offset + sample_width_bytes]
        if len(sample) < sample_width_bytes:
            return True
        if abs(_pcm_sample_value(sample, sample_width_bytes)) > silence_threshold:
            return False
    return True


def _pcm_sample_value(sample: bytes, sample_width_bytes: int) -> int:
    if sample_width_bytes == 1:
        return sample[0] - 128
    return int.from_bytes(sample, "little", signed=True)


def _silent_pcm_frames(
    frame_count: int,
    *,
    sample_width_bytes: int,
    channels: int,
) -> bytes:
    if frame_count <= 0:
        return b""
    if sample_width_bytes == 1:
        return b"\x80" * frame_count * channels
    return b"\0" * frame_count * sample_width_bytes * channels


__all__ = [
    "CompactSessionAudioConfig",
    "CompactSessionAudioResult",
    "PublicTranscriptTurn",
    "build_compact_session_audio",
    "format_public_script",
    "list_runtime_sessions",
    "load_public_transcript_turns",
    "session_has_human_audio_turns",
    "session_realtime_audio_path",
]
