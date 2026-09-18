from __future__ import annotations

import asyncio
import json
import time
import wave
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable

from counseling_voice_demo.runtime.models import (
    AudioBusItem,
    AudioChunk,
    EndOfAudio,
    PromptDirectorAttemptRecord,
    ResponseInstructionsRecord,
    RuntimeEvent,
    RuntimeMetric,
    TranscriptEvent,
)

@dataclass(frozen=True)
class RuntimeLogPaths:
    session_dir: Path
    public_dir: Path
    internal_dir: Path
    events_jsonl: Path
    response_instructions_jsonl: Path
    prompt_director_attempts_jsonl: Path
    transcripts_jsonl: Path
    audio_dir: Path
    audio_chunks_dir: Path
    session_audio_wav: Path
    session_audio_timeline_jsonl: Path
    metrics_jsonl: Path


def create_runtime_log_dirs(
    sessions_dir: Path | str,
    session_id: str,
) -> RuntimeLogPaths:
    session_dir = Path(sessions_dir) / session_id
    public_dir = session_dir / "public"
    internal_dir = session_dir / "internal"
    paths = RuntimeLogPaths(
        session_dir=session_dir,
        public_dir=public_dir,
        internal_dir=internal_dir,
        events_jsonl=internal_dir / "events" / f"{session_id}.jsonl",
        response_instructions_jsonl=(
            internal_dir / "prompts" / "response_instructions.jsonl"
        ),
        prompt_director_attempts_jsonl=(
            internal_dir / "prompts" / "prompt_director_attempts.jsonl"
        ),
        transcripts_jsonl=internal_dir / "transcripts" / f"{session_id}.jsonl",
        audio_dir=internal_dir / "audio",
        audio_chunks_dir=internal_dir / "audio_chunks",
        session_audio_wav=internal_dir / "audio" / "session_realtime.wav",
        session_audio_timeline_jsonl=internal_dir / "audio" / "session_timeline.jsonl",
        metrics_jsonl=internal_dir / "metrics" / f"{session_id}.jsonl",
    )
    for directory in [
        paths.public_dir,
        paths.internal_dir,
        paths.events_jsonl.parent,
        paths.response_instructions_jsonl.parent,
        paths.transcripts_jsonl.parent,
        paths.audio_dir,
        paths.audio_chunks_dir,
        paths.metrics_jsonl.parent,
    ]:
        directory.mkdir(parents=True, exist_ok=True)
    for file_path in [
        paths.events_jsonl,
        paths.response_instructions_jsonl,
        paths.prompt_director_attempts_jsonl,
        paths.transcripts_jsonl,
        paths.session_audio_timeline_jsonl,
        paths.metrics_jsonl,
    ]:
        file_path.touch(exist_ok=True)
    return paths


class AsyncLogger:
    def __init__(
        self,
        *,
        sessions_dir: Path | str,
        session_id: str,
        monotonic: Callable[[], float] | None = None,
    ) -> None:
        self.paths = create_runtime_log_dirs(sessions_dir, session_id)
        self._record_queue: asyncio.Queue[
            RuntimeEvent
            | RuntimeMetric
            | TranscriptEvent
            | ResponseInstructionsRecord
            | PromptDirectorAttemptRecord
            | None
        ] = asyncio.Queue()
        self._record_task: asyncio.Task[None] | None = None
        self._monotonic = monotonic or time.monotonic
        self._started_monotonic: float | None = None
        self._session_audio_writer: wave.Wave_write | None = None
        self._session_audio_format: tuple[int, int, int] | None = None
        self._session_audio_frame_count = 0
        self._session_audio_turn_starts: dict[tuple[int, str], float] = {}

    async def start(self) -> None:
        if self._started_monotonic is None:
            self._started_monotonic = self._monotonic()
        if self._record_task is None:
            self._record_task = asyncio.create_task(self._record_worker())

    async def log_event(self, event: RuntimeEvent) -> None:
        await self._record_queue.put(event)

    async def log_transcript(self, transcript: TranscriptEvent) -> None:
        await self._record_queue.put(transcript)

    async def log_response_instructions(
        self,
        record: ResponseInstructionsRecord,
    ) -> None:
        await self._record_queue.put(record)

    async def log_prompt_director_attempt(
        self, record: PromptDirectorAttemptRecord
    ) -> None:
        await self._record_queue.put(record)

    async def log_metric(self, metric: RuntimeMetric) -> None:
        await self._record_queue.put(metric)

    async def consume_audio(self, queue: asyncio.Queue[AudioBusItem | None]) -> None:
        buffers: dict[tuple[int, str], list[AudioChunk]] = {}
        try:
            while True:
                item = await queue.get()
                if item is None:
                    self._flush_all_audio(buffers)
                    return
                if isinstance(item, EndOfAudio):
                    if item.turn_id is None or item.speaker is None:
                        self._flush_all_audio(buffers)
                        return
                    self._write_turn_audio(
                        item.turn_id,
                        item.speaker,
                        buffers.pop((item.turn_id, item.speaker), []),
                    )
                    self._write_session_audio_timeline(item.turn_id, item.speaker)
                    continue
                self._write_session_audio_chunk(
                    item,
                    received_monotonic=self._monotonic(),
                )
                buffers.setdefault((item.turn_id, item.speaker), []).append(item)
                self._append_jsonl(
                    self.paths.audio_chunks_dir
                    / f"turn_{item.turn_id:04d}_{item.speaker}.jsonl",
                    item,
                )
        finally:
            self._close_session_audio()

    async def close(self) -> None:
        await self._record_queue.put(None)
        if self._record_task is not None:
            await self._record_task
            self._record_task = None

    async def _record_worker(self) -> None:
        while True:
            record = await self._record_queue.get()
            if record is None:
                return
            if isinstance(record, RuntimeEvent):
                self._append_jsonl(self.paths.events_jsonl, record)
            elif isinstance(record, ResponseInstructionsRecord):
                self._append_jsonl(self.paths.response_instructions_jsonl, record)
            elif isinstance(record, PromptDirectorAttemptRecord):
                self._append_jsonl(self.paths.prompt_director_attempts_jsonl, record)
            elif isinstance(record, RuntimeMetric):
                self._append_jsonl(self.paths.metrics_jsonl, record)
            elif isinstance(record, TranscriptEvent):
                self._append_jsonl(self.paths.transcripts_jsonl, record)

    def _flush_all_audio(self, buffers: dict[tuple[int, str], list[AudioChunk]]) -> None:
        for (turn_id, speaker), chunks in list(buffers.items()):
            self._write_turn_audio(turn_id, speaker, chunks)
            self._write_session_audio_timeline(turn_id, speaker)
        buffers.clear()

    def _write_turn_audio(
        self,
        turn_id: int,
        speaker: str,
        chunks: list[AudioChunk],
    ) -> None:
        if not chunks:
            return
        first = chunks[0]
        wav_path = self.paths.audio_dir / f"turn_{turn_id:04d}_{speaker}.wav"
        with wave.open(str(wav_path), "wb") as wav_file:
            wav_file.setnchannels(first.channels)
            wav_file.setsampwidth(first.sample_width_bits // 8)
            wav_file.setframerate(first.sample_rate)
            wav_file.writeframes(b"".join(chunk.pcm for chunk in chunks))

    def _write_session_audio_chunk(
        self,
        chunk: AudioChunk,
        *,
        received_monotonic: float,
    ) -> None:
        if self._started_monotonic is None:
            self._started_monotonic = received_monotonic
        self._ensure_session_audio_writer(chunk)
        assert self._session_audio_writer is not None
        assert self._session_audio_format is not None
        sample_rate, sample_width_bits, channels = self._session_audio_format
        bytes_per_frame = _bytes_per_frame(
            sample_width_bits=sample_width_bits,
            channels=channels,
        )
        turn_key = (chunk.turn_id, chunk.speaker)
        if turn_key not in self._session_audio_turn_starts:
            target_start_seconds = max(
                0.0,
                received_monotonic - self._started_monotonic,
            )
            current_seconds = self._session_audio_seconds()
            if target_start_seconds > current_seconds:
                silence_frames = round(
                    (target_start_seconds - current_seconds) * sample_rate
                )
                if silence_frames > 0:
                    self._session_audio_writer.writeframes(
                        b"\0" * silence_frames * bytes_per_frame
                    )
                    self._session_audio_frame_count += silence_frames

            self._session_audio_turn_starts[turn_key] = self._session_audio_seconds()
        self._session_audio_writer.writeframes(chunk.pcm)
        self._session_audio_frame_count += len(chunk.pcm) // bytes_per_frame

    def _ensure_session_audio_writer(self, chunk: AudioChunk) -> None:
        chunk_format = (chunk.sample_rate, chunk.sample_width_bits, chunk.channels)
        if self._session_audio_format is not None:
            if chunk_format != self._session_audio_format:
                raise ValueError(
                    "session audio chunk format changed from "
                    f"{self._session_audio_format} to {chunk_format}"
                )
            return
        self._session_audio_format = chunk_format
        self._session_audio_writer = wave.open(str(self.paths.session_audio_wav), "wb")
        self._session_audio_writer.setnchannels(chunk.channels)
        self._session_audio_writer.setsampwidth(chunk.sample_width_bits // 8)
        self._session_audio_writer.setframerate(chunk.sample_rate)

    def _write_session_audio_timeline(self, turn_id: int, speaker: str) -> None:
        start_seconds = self._session_audio_turn_starts.pop((turn_id, speaker), None)
        if start_seconds is None or self._session_audio_format is None:
            return
        self._append_jsonl(
            self.paths.session_audio_timeline_jsonl,
            {
                "session_id": self.paths.session_dir.name,
                "turn_id": turn_id,
                "speaker": speaker,
                "start_seconds": round(start_seconds, 6),
                "end_seconds": round(self._session_audio_seconds(), 6),
                "audio_path": str(
                    self.paths.session_audio_wav.relative_to(self.paths.session_dir)
                ),
            },
        )

    def _session_audio_seconds(self) -> float:
        if self._session_audio_format is None:
            return 0.0
        sample_rate, _, _ = self._session_audio_format
        return self._session_audio_frame_count / sample_rate

    def _close_session_audio(self) -> None:
        if self._session_audio_writer is None:
            return
        self._session_audio_writer.close()
        self._session_audio_writer = None

    def _append_jsonl(self, path: Path, record: Any) -> None:
        with path.open("a", encoding="utf-8") as file:
            file.write(
                json.dumps(_jsonable(record), ensure_ascii=False, sort_keys=True)
                + "\n"
            )


def _bytes_per_frame(*, sample_width_bits: int, channels: int) -> int:
    if sample_width_bits <= 0 or sample_width_bits % 8 != 0:
        raise ValueError("sample_width_bits must be a positive multiple of 8")
    if channels <= 0:
        raise ValueError("channels must be positive")
    return (sample_width_bits // 8) * channels


def _jsonable(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "__dataclass_fields__"):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, bytes):
        return {"byte_length": len(value)}
    return value
