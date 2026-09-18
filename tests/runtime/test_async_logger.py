from __future__ import annotations

import asyncio
import json
import wave

from counseling_voice_demo.runtime.logger import AsyncLogger
from counseling_voice_demo.runtime.models import (
    AudioChunk,
    EndOfAudio,
    ResponseInstructionsRecord,
    RuntimeEvent,
    TranscriptEvent,
)


def test_async_logger_writes_events_transcripts_and_wav(tmp_path) -> None:
    async def scenario() -> None:
        logger = AsyncLogger(
            sessions_dir=tmp_path,
            session_id="session_logger_test",
            monotonic=lambda: 0.0,
        )
        await logger.start()
        await logger.log_event(
            RuntimeEvent(
                session_id="session_logger_test",
                event_type="tts_stream_done",
            )
        )
        await logger.log_transcript(
            TranscriptEvent(
                session_id="session_logger_test",
                turn_id=1,
                speaker="counselor",
                transcript_type="final",
                text="聞こえた内容です。",
            )
        )

        audio_queue = asyncio.Queue()
        consumer = asyncio.create_task(logger.consume_audio(audio_queue))
        await audio_queue.put(
            AudioChunk(
                session_id="session_logger_test",
                turn_id=1,
                speaker="counselor",
                chunk_index=0,
                pcm=b"\x01\x00\x02\x00",
                duration_ms=1,
            )
        )
        await audio_queue.put(
            EndOfAudio(
                session_id="session_logger_test",
                turn_id=1,
                speaker="counselor",
            )
        )
        await audio_queue.put(None)
        await consumer
        await logger.close()

        events_path = (
            tmp_path
            / "session_logger_test"
            / "internal"
            / "events"
            / "session_logger_test.jsonl"
        )
        transcripts_path = (
            tmp_path
            / "session_logger_test"
            / "internal"
            / "transcripts"
            / "session_logger_test.jsonl"
        )
        wav_path = (
            tmp_path
            / "session_logger_test"
            / "internal"
            / "audio"
            / "turn_0001_counselor.wav"
        )
        session_wav_path = (
            tmp_path
            / "session_logger_test"
            / "internal"
            / "audio"
            / "session_realtime.wav"
        )
        timeline_path = (
            tmp_path
            / "session_logger_test"
            / "internal"
            / "audio"
            / "session_timeline.jsonl"
        )

        assert (
            json.loads(events_path.read_text(encoding="utf-8").splitlines()[0])[
                "event_type"
            ]
            == "tts_stream_done"
        )
        assert (
            json.loads(transcripts_path.read_text(encoding="utf-8").splitlines()[0])[
                "text"
            ]
            == "聞こえた内容です。"
        )
        with wave.open(str(wav_path), "rb") as wav_file:
            assert wav_file.getframerate() == 24000
            assert wav_file.getnchannels() == 1
            assert wav_file.getsampwidth() == 2
            assert wav_file.readframes(2) == b"\x01\x00\x02\x00"
        with wave.open(str(session_wav_path), "rb") as wav_file:
            assert wav_file.getframerate() == 24000
            assert wav_file.getnchannels() == 1
            assert wav_file.getsampwidth() == 2
            assert wav_file.readframes(2) == b"\x01\x00\x02\x00"
        assert json.loads(timeline_path.read_text(encoding="utf-8").splitlines()[0]) == {
            "audio_path": "internal/audio/session_realtime.wav",
            "end_seconds": 8.3e-05,
            "session_id": "session_logger_test",
            "speaker": "counselor",
            "start_seconds": 0.0,
            "turn_id": 1,
        }

    asyncio.run(scenario())


def test_async_logger_writes_response_instructions_only_to_internal_prompts(
    tmp_path,
) -> None:
    async def scenario() -> None:
        logger = AsyncLogger(
            sessions_dir=tmp_path,
            session_id="session_instruction_log_test",
        )
        await logger.start()
        await logger.log_response_instructions(
            ResponseInstructionsRecord(
                session_id="session_instruction_log_test",
                turn_id=3,
                speaker="counselor",
                session_instructions="固定システムプロンプト",
                participant_instructions="カウンセラープリセット",
                additional_instructions="クロージング指示",
                resolved_instructions="実際に送信したinstructions",
                instructions_sha256="sha256-value",
                repeat_session_instructions=True,
            )
        )
        await logger.close()

        session_dir = tmp_path / "session_instruction_log_test"
        prompt_path = (
            session_dir
            / "internal"
            / "prompts"
            / "response_instructions.jsonl"
        )
        records = [
            json.loads(line)
            for line in prompt_path.read_text(encoding="utf-8").splitlines()
        ]

        assert records[0]["turn_id"] == 3
        assert records[0]["speaker_id"] == "counselor"
        assert records[0]["session_instructions"] == "固定システムプロンプト"
        assert records[0]["participant_instructions"] == "カウンセラープリセット"
        assert records[0]["additional_instructions"] == "クロージング指示"
        assert records[0]["resolved_instructions"] == "実際に送信したinstructions"
        assert records[0]["instructions_sha256"] == "sha256-value"
        assert records[0]["repeat_session_instructions"] is True
        assert not any(
            "実際に送信したinstructions" in path.read_text(
                encoding="utf-8", errors="ignore"
            )
            for path in (session_dir / "public").rglob("*")
            if path.is_file()
        )

    asyncio.run(scenario())


def test_async_logger_does_not_insert_silence_between_chunks_in_same_turn(
    tmp_path,
) -> None:
    class FakeClock:
        value = 0.0

        def __call__(self) -> float:
            return self.value

    async def scenario() -> None:
        clock = FakeClock()
        logger = AsyncLogger(
            sessions_dir=tmp_path,
            session_id="session_same_turn_chunks_test",
            monotonic=clock,
        )
        await logger.start()

        audio_queue = asyncio.Queue()
        consumer = asyncio.create_task(logger.consume_audio(audio_queue))
        clock.value = 1.0
        await audio_queue.put(
            AudioChunk(
                session_id="session_same_turn_chunks_test",
                turn_id=1,
                speaker="counselor",
                chunk_index=0,
                pcm=b"\x01\x00" * 4,
                sample_rate=4,
                sample_width_bits=16,
                channels=1,
                duration_ms=1000,
            )
        )
        await asyncio.sleep(0)
        clock.value = 3.0
        await audio_queue.put(
            AudioChunk(
                session_id="session_same_turn_chunks_test",
                turn_id=1,
                speaker="counselor",
                chunk_index=1,
                pcm=b"\x02\x00" * 4,
                sample_rate=4,
                sample_width_bits=16,
                channels=1,
                duration_ms=1000,
            )
        )
        await audio_queue.put(
            EndOfAudio(
                session_id="session_same_turn_chunks_test",
                turn_id=1,
                speaker="counselor",
            )
        )
        await audio_queue.put(None)
        await consumer
        await logger.close()

        session_dir = tmp_path / "session_same_turn_chunks_test"
        session_wav_path = session_dir / "internal" / "audio" / "session_realtime.wav"
        timeline_path = session_dir / "internal" / "audio" / "session_timeline.jsonl"

        with wave.open(str(session_wav_path), "rb") as wav_file:
            assert wav_file.getframerate() == 4
            assert wav_file.getnframes() == 12
            assert wav_file.readframes(12) == (
                (b"\x00\x00" * 4)
                + (b"\x01\x00" * 4)
                + (b"\x02\x00" * 4)
            )
        timeline = [
            json.loads(line)
            for line in timeline_path.read_text(encoding="utf-8").splitlines()
        ]
        assert timeline[0]["start_seconds"] == 1.0
        assert timeline[0]["end_seconds"] == 3.0

    asyncio.run(scenario())


def test_async_logger_writes_continuous_session_wav_with_silence(tmp_path) -> None:
    class FakeClock:
        value = 0.0

        def __call__(self) -> float:
            return self.value

    async def scenario() -> None:
        clock = FakeClock()
        logger = AsyncLogger(
            sessions_dir=tmp_path,
            session_id="session_continuous_audio_test",
            monotonic=clock,
        )
        await logger.start()

        audio_queue = asyncio.Queue()
        consumer = asyncio.create_task(logger.consume_audio(audio_queue))
        clock.value = 1.0
        await audio_queue.put(
            AudioChunk(
                session_id="session_continuous_audio_test",
                turn_id=1,
                speaker="counselor",
                chunk_index=0,
                pcm=b"\x01\x00" * 4,
                sample_rate=4,
                sample_width_bits=16,
                channels=1,
                duration_ms=1000,
            )
        )
        await asyncio.sleep(0)
        await audio_queue.put(
            EndOfAudio(
                session_id="session_continuous_audio_test",
                turn_id=1,
                speaker="counselor",
            )
        )
        await asyncio.sleep(0)
        clock.value = 3.0
        await audio_queue.put(
            AudioChunk(
                session_id="session_continuous_audio_test",
                turn_id=2,
                speaker="client",
                chunk_index=0,
                pcm=b"\x02\x00" * 4,
                sample_rate=4,
                sample_width_bits=16,
                channels=1,
                duration_ms=1000,
            )
        )
        await asyncio.sleep(0)
        await audio_queue.put(
            EndOfAudio(
                session_id="session_continuous_audio_test",
                turn_id=2,
                speaker="client",
            )
        )
        await audio_queue.put(None)
        await consumer
        await logger.close()

        session_dir = tmp_path / "session_continuous_audio_test"
        session_wav_path = session_dir / "internal" / "audio" / "session_realtime.wav"
        timeline_path = session_dir / "internal" / "audio" / "session_timeline.jsonl"

        with wave.open(str(session_wav_path), "rb") as wav_file:
            assert wav_file.getframerate() == 4
            assert wav_file.getnframes() == 16
            assert wav_file.readframes(16) == (
                (b"\x00\x00" * 4)
                + (b"\x01\x00" * 4)
                + (b"\x00\x00" * 4)
                + (b"\x02\x00" * 4)
            )
        timeline = [
            json.loads(line)
            for line in timeline_path.read_text(encoding="utf-8").splitlines()
        ]
        assert timeline == [
            {
                "audio_path": "internal/audio/session_realtime.wav",
                "end_seconds": 2.0,
                "session_id": "session_continuous_audio_test",
                "speaker": "counselor",
                "start_seconds": 1.0,
                "turn_id": 1,
            },
            {
                "audio_path": "internal/audio/session_realtime.wav",
                "end_seconds": 4.0,
                "session_id": "session_continuous_audio_test",
                "speaker": "client",
                "start_seconds": 3.0,
                "turn_id": 2,
            },
        ]

    asyncio.run(scenario())
