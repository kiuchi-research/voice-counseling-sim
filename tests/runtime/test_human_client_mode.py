from __future__ import annotations

from dataclasses import replace
import asyncio
import json

import pytest

from counseling_voice_demo.runtime.config import RuntimeSettings, load_runtime_config
from counseling_voice_demo.runtime.control_api import (
    RuntimeControlError,
    apply_runtime_start_options,
    RuntimeControlService,
)
from counseling_voice_demo.runtime.factory import (
    build_fake_runtime,
    build_runtime_config_model,
)
from counseling_voice_demo.runtime.models import (
    ActorKind,
    ParticipantConfig,
)
from counseling_voice_demo.runtime.models import (
    HumanTurnInput,
    HumanAudioStreamStart,
    RuntimePhase,
    TranscriptEvent,
)
from counseling_voice_demo.runtime.agents import FakeAgent
from counseling_voice_demo.runtime.controller import ConversationRuntime


HUMAN_CLIENT_MODE = "ai_counselor_human_client"


def human_client_settings(**options):
    return apply_runtime_start_options(
        load_runtime_config(),
        {"interaction_mode": HUMAN_CLIENT_MODE, **options},
    )


def test_human_client_start_removes_previous_ai_client_settings():
    settings = human_client_settings()
    config = build_runtime_config_model(settings)

    assert config.interaction_mode == HUMAN_CLIENT_MODE
    assert config.participant_mode == "one_client"
    assert config.speaker_selection_policy == "fixed_round_robin"
    assert config.fixed_speaker_sequence == ("counselor", "client")
    assert config.human_interrupts_enabled
    assert config.initial_client_transcript == ""
    assert config.shared_case == ""
    assert set(config.participants) == {"counselor", "client"}
    assert config.participants["counselor"].actor_kind is ActorKind.AI
    assert config.participants["client"] == ParticipantConfig(
        speaker_id="client",
        role="client",
        actor_kind="human",
        display_name="クライアント",
    )


def test_human_client_start_accepts_empty_shared_information_and_initial_text():
    settings = human_client_settings(initial_client_transcript="", shared_case="")
    assert settings.shared_case.prompt_source == ""
    assert settings.runtime.initial_client_transcript == ""


@pytest.mark.parametrize(
    "options, message",
    [
        ({"participant_mode": "two_clients"}, "one_client"),
        ({"speaker_selection_policy": "turn_boundary_timing"}, "fixed_round_robin"),
        ({"fixed_speaker_sequence": ["client", "counselor"]}, "fixed_speaker_sequence"),
        ({"initial_client_transcript": "架空の相談"}, "initial_client_transcript"),
        ({"client_prompt": "クライアントを演じて"}, "client_prompt"),
        ({"client_tts_voice": "marin"}, "client_tts_voice"),
        ({"client_realtime_output_speed": 0.9}, "client_realtime_output_speed"),
        ({"realtime_api_centered_mode": False}, "realtime_api_centered_mode"),
    ],
)
def test_human_client_start_rejects_incompatible_options(options, message):
    with pytest.raises((RuntimeControlError, ValueError), match=message):
        human_client_settings(**options)


def test_human_client_model_and_settings_validate_actor_kind():
    settings = human_client_settings()
    config = build_runtime_config_model(settings)
    participants = dict(config.participants)
    participants["client"] = replace(participants["client"], actor_kind=ActorKind.AI)
    with pytest.raises(ValueError, match="human"):
        replace(config, participants=participants)

    document = settings.model_dump()
    document["participants"]["client"]["actor_kind"] = "ai"
    with pytest.raises(ValueError, match="human"):
        RuntimeSettings.model_validate(document)


def test_human_client_model_rejects_human_ai_profile():
    settings = human_client_settings()
    config = build_runtime_config_model(settings)
    participants = dict(config.participants)
    participants["client"] = replace(
        participants["client"], private_profile_source="以前のAI用隠し背景"
    )
    with pytest.raises(ValueError, match="private_profile_source"):
        replace(config, participants=participants)


def test_human_client_fake_runtime_builds_only_counselor(tmp_path):
    settings = human_client_settings()
    settings = settings.model_copy(
        update={
            "paths": settings.paths.model_copy(
                update={"runtime_sessions_dir": str(tmp_path)}
            )
        }
    )
    runtime = build_fake_runtime(settings)
    assert set(runtime.controller.agents) == {"counselor"}


def test_human_client_start_preserves_counselor_and_routes_without_mutating_base():
    original = load_runtime_config()
    before = original.model_dump()
    settings = apply_runtime_start_options(
        original,
        {
            "interaction_mode": HUMAN_CLIENT_MODE,
            "counselor_prompt": "短く問いかける",
            "shared_case": "AIへ明示的に共有した情報",
        },
    )
    assert settings.participants["counselor"].prompt_source == "短く問いかける"
    assert settings.shared_case.prompt_source == "AIへ明示的に共有した情報"
    assert settings.ai == original.ai
    assert original.model_dump() == before


def make_runtime(tmp_path, *, agent=None, stt=None, **overrides):
    config = build_runtime_config_model(human_client_settings())
    config = replace(
        config, **{"max_turns": 5, "closing_start_elapsed_seconds": None, **overrides}
    )
    return ConversationRuntime(
        config=config,
        agents={
            "counselor": agent or FakeAgent("counselor", "応答:{input_transcript}")
        },
        stt=stt,
        sessions_dir=tmp_path,
    )


async def wait_for_input(runtime, turn_id):
    async with asyncio.timeout(2):
        while not (
            runtime.status.awaiting_human_input
            and runtime.status.current_turn_id == turn_id
        ):
            assert runtime.status.phase not in {
                RuntimePhase.ERROR,
                RuntimePhase.COMPLETED,
            }
            await asyncio.sleep(0.005)


async def cancel_runtime(task):
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)


def test_human_client_greeting_then_two_human_turns_without_timing_or_prefetch(
    tmp_path,
):
    async def scenario():
        runtime = make_runtime(tmp_path)
        task = asyncio.create_task(runtime.run())
        try:
            await wait_for_input(runtime, 2)
            counselor = runtime.controller.agents["counselor"]
            assert len(counselor.received_inputs) == 1
            assert (
                "人間クライアントはまだ発話していません" in counselor.received_inputs[0]
            )
            for turn_id in (2, 4):
                await wait_for_input(runtime, turn_id)
                await runtime.submit_human_turn(
                    HumanTurnInput(
                        session_id=runtime.config.session_id,
                        text=f"実際の発話{turn_id}",
                        recipient_ids=("counselor",),
                        client_message_id=f"message-{turn_id}",
                    )
                )
            turns = await asyncio.wait_for(task, 2)
            assert [(turn.turn_id, turn.speaker) for turn in turns] == [
                (1, "counselor"),
                (2, "client"),
                (3, "counselor"),
                (4, "client"),
                (5, "counselor"),
            ]
            assert counselor.received_timing_inputs == []
            assert len(counselor.received_inputs) == 3
            assert turns[2].input_transcript == "実際の発話2"
            assert (
                "人間クライアントはまだ発話していません"
                not in counselor.received_inputs[-1]
            )
            events = [
                json.loads(line)
                for line in runtime.logger.paths.events_jsonl.read_text().splitlines()
            ]
            settings = next(
                e
                for e in events
                if e["event_type"] == "runtime_audio_settings_resolved"
            )
            assert settings["details"]["interaction_mode"] == HUMAN_CLIENT_MODE
        finally:
            await cancel_runtime(task)

    asyncio.run(scenario())


def test_human_client_status_exposes_only_public_participant_metadata(tmp_path):
    runtime = make_runtime(tmp_path)
    status = runtime.status
    assert status.interaction_mode == HUMAN_CLIENT_MODE
    assert status.human_speaker_id == "client"
    assert status.human_recipient_ids == ("counselor",)
    assert status.participants["client"]["actor_kind"] == "human"
    assert set(status.participants["counselor"]) == {
        "role",
        "actor_kind",
        "display_name",
    }


def test_human_client_rejects_wrong_recipient_without_consuming_slot(tmp_path):
    async def scenario():
        runtime = make_runtime(tmp_path)
        task = asyncio.create_task(runtime.run())
        try:
            await wait_for_input(runtime, 2)
            with pytest.raises(RuntimeError, match="recipient"):
                await runtime.submit_human_turn(
                    HumanTurnInput(
                        session_id=runtime.config.session_id,
                        text="発話",
                        recipient_ids=("client",),
                    )
                )
            assert runtime.status.awaiting_human_input
        finally:
            await cancel_runtime(task)

    asyncio.run(scenario())


def test_human_client_interrupt_reserves_client_and_next_speaker_is_ai(tmp_path):
    async def scenario():
        runtime = make_runtime(tmp_path)
        runtime._active_turn_id = 2
        runtime._active_speaker_id = "counselor"
        slot = runtime._open_human_input_slot_after_interrupt(interrupted_turn_id=2)
        assert slot["human_speaker"] == "client"
        assert slot["human_turn_id"] == 3
        assert (
            runtime._open_human_input_slot_after_interrupt(interrupted_turn_id=2)
            == slot
        )
        selected = await runtime._select_speaker_for_turn(
            4,
            input_transcript="割り込んだ発話",
            previous_speaker="client",
        )
        assert selected.speaker == "counselor"

    asyncio.run(scenario())


class ControlledHumanStt:
    def __init__(self):
        self.started = asyncio.Event()
        self.finish = asyncio.Event()
        self.cancelled = False
        self.text = "取り消す前の発話"

    async def transcribe_from_queue_observed(
        self, *, session_id, turn_id, speaker, queue, on_transcript, recording_mode
    ):
        self.started.set()
        try:
            await self.finish.wait()
        except asyncio.CancelledError:
            self.cancelled = True
            raise
        final = TranscriptEvent(session_id, turn_id, speaker, "final", self.text)
        await on_transcript(final)
        return [], final


def stream_request(runtime, message_id, mode="push_to_talk"):
    return HumanAudioStreamStart(
        session_id=runtime.config.session_id,
        sample_rate=24000,
        channels=1,
        recipient_ids=("counselor",),
        recording_mode=mode,
        client_message_id=message_id,
    )


@pytest.mark.parametrize("mode", ["push_to_talk", "vad_auto", "browser_vad"])
def test_human_client_abort_discards_pending_stt_and_accepts_next_recording(
    tmp_path, mode
):
    async def scenario():
        stt = ControlledHumanStt()
        runtime = make_runtime(tmp_path, stt=stt)
        task = asyncio.create_task(runtime.run())
        try:
            await wait_for_input(runtime, 2)
            await runtime.start_human_audio_stream(
                stream_request(runtime, "aborted", mode)
            )
            await asyncio.wait_for(stt.started.wait(), 1)
            if mode != "vad_auto":
                await runtime.end_human_audio_stream(stream_id="aborted")
            result = await runtime.abort_human_audio_stream(stream_id="aborted")
            assert result["completion_reason"] == "aborted"
            await wait_for_input(runtime, 2)
            assert stt.cancelled
            assert len(runtime.turns) == 1
            stt.text = "新しい実際の発話"
            stt.finish.set()
            await runtime.start_human_audio_stream(
                stream_request(runtime, "next", mode)
            )
            if mode != "vad_auto":
                await runtime.end_human_audio_stream(stream_id="next")
            await wait_for_input(runtime, 4)
            assert runtime.turns[1].stt_final_transcript == "新しい実際の発話"
        finally:
            await cancel_runtime(task)
        transcripts = [
            json.loads(line)
            for line in runtime.logger.paths.transcripts_jsonl.read_text().splitlines()
        ]
        human = [row for row in transcripts if row["transcript_type"] == "human_final"]
        assert [row["text"] for row in human] == ["新しい実際の発話"]

    asyncio.run(scenario())


def test_human_client_pause_aborts_stream_and_does_not_resume_from_audio(tmp_path):
    async def scenario():
        stt = ControlledHumanStt()
        runtime = make_runtime(tmp_path, stt=stt)
        service = RuntimeControlService(lambda: runtime)
        await service.start()
        task = service._task
        try:
            await wait_for_input(runtime, 2)
            await runtime.start_human_audio_stream(
                stream_request(runtime, "before-pause")
            )
            await asyncio.wait_for(stt.started.wait(), 1)
            status = await service.pause("playback")
            assert status.phase is RuntimePhase.PAUSED
            with pytest.raises(RuntimeControlError, match="paused"):
                await service.start_human_audio_stream(
                    {
                        "session_id": runtime.config.session_id,
                        "sample_rate": 24000,
                        "channels": 1,
                        "recipient_ids": ["counselor"],
                        "client_message_id": "during-pause",
                    }
                )
            assert (await service.status()).phase is RuntimePhase.PAUSED
            await service.resume()
            await wait_for_input(runtime, 2)
            assert len(runtime.turns) == 1
        finally:
            await service.stop()
            await asyncio.gather(task, return_exceptions=True)

    asyncio.run(scenario())


def test_human_client_repeated_text_submission_does_not_consume_next_turn(tmp_path):
    async def scenario():
        runtime = make_runtime(tmp_path)
        task = asyncio.create_task(runtime.run())
        request = HumanTurnInput(
            session_id=runtime.config.session_id,
            text="一度だけ送る発話",
            recipient_ids=("counselor",),
            client_message_id="same-message",
        )
        try:
            await wait_for_input(runtime, 2)
            first = await runtime.submit_human_turn(request)
            second = await runtime.submit_human_turn(request)
            assert second["turn_id"] == first["turn_id"] == 2
            await wait_for_input(runtime, 4)
            third = await runtime.submit_human_turn(request)
            assert third["turn_id"] == 2
            assert runtime.status.awaiting_human_input
            assert len(runtime.turns) == 3
        finally:
            await cancel_runtime(task)

    asyncio.run(scenario())


def test_human_client_elapsed_limit_finishes_while_waiting_for_input(tmp_path):
    async def scenario():
        runtime = make_runtime(
            tmp_path, stop_condition="elapsed_time", max_elapsed_seconds=0.1
        )
        task = asyncio.create_task(runtime.run())
        try:
            await wait_for_input(runtime, 2)
            await asyncio.wait_for(task, 1)
            assert runtime.status.phase is RuntimePhase.COMPLETED
            assert len(runtime.turns) == 1
        finally:
            await cancel_runtime(task)

    asyncio.run(scenario())


def test_bulk_audio_is_queued_and_cancellable_without_blocking_control(tmp_path):
    from counseling_voice_demo.runtime.models import HumanAudioInput

    async def scenario():
        stt = ControlledHumanStt()
        runtime = make_runtime(tmp_path, stt=stt)
        task = asyncio.create_task(runtime.run())
        try:
            await wait_for_input(runtime, 2)
            request = HumanAudioInput(
                session_id=runtime.config.session_id,
                audio_bytes=b"\x01\x00" * 2400,
                sample_rate=24000,
                channels=1,
                recipient_ids=("counselor",),
                client_message_id="bulk-one",
            )
            result = await asyncio.wait_for(
                runtime.submit_human_audio(request), timeout=1
            )
            assert result["accepted"]
            assert await runtime.submit_human_audio(request) == result
            await stt.started.wait()
            await runtime.abort_pending_human_audio_streams()
            await wait_for_input(runtime, 2)
            assert len(runtime.turns) == 1
            assert stt.cancelled
        finally:
            await cancel_runtime(task)

    asyncio.run(scenario())


@pytest.mark.parametrize("mode", ["push_to_talk", "browser_vad"])
def test_human_client_end_waits_for_stt_and_blank_is_not_sent_success(tmp_path, mode):
    async def scenario():
        stt = ControlledHumanStt()
        stt.text = ""
        runtime = make_runtime(tmp_path, stt=stt)
        task = asyncio.create_task(runtime.run())
        try:
            await wait_for_input(runtime, 2)
            await runtime.start_human_audio_stream(stream_request(runtime, "empty-one", mode))
            await stt.started.wait()
            ended = await runtime.end_human_audio_stream(stream_id="empty-one")
            assert ended["pending_transcription"]
            assert await runtime.end_human_audio_stream(stream_id="empty-one") == ended
            assert not runtime._human_audio_streams["empty-one"].completion.done()
            stt.finish.set()
            result = await asyncio.wait_for(
                runtime.wait_human_audio_stream_completion(stream_id="empty-one"), 1
            )
            assert result["accepted"] is False
            assert result["completion_reason"] == "blank_transcript"
            await wait_for_input(runtime, 2)
            assert len(runtime.turns) == 1
        finally:
            await cancel_runtime(task)

    asyncio.run(scenario())


def test_saved_human_client_session_restores_roles_and_audio(tmp_path):
    from counseling_voice_demo.runtime.session_artifacts import (
        load_public_transcript_turns,
        load_session_identity,
        build_compact_session_audio,
    )

    async def scenario():
        stt = ControlledHumanStt()
        stt.text = "最近の仕事について相談したいです。"
        runtime = make_runtime(tmp_path, stt=stt, max_turns=3)
        task = asyncio.create_task(runtime.run())
        try:
            await wait_for_input(runtime, 2)
            await runtime.start_human_audio_stream(
                stream_request(runtime, "saved-input")
            )
            await runtime.append_human_audio_stream_chunk(
                stream_id="saved-input",
                audio_bytes=b"\x01\x00" * 2400,
                chunk_index=0,
            )
            await runtime.end_human_audio_stream(stream_id="saved-input")
            stt.finish.set()
            await asyncio.wait_for(task, 2)
            path = runtime.logger.paths.events_jsonl.parent.parent.parent
            identity = load_session_identity(path)
            assert identity["interaction_mode"] == HUMAN_CLIENT_MODE
            assert identity["participants"]["client"]["actor_kind"] == "human"
            turns = load_public_transcript_turns(path)
            assert [(turn.speaker, turn.actor_kind) for turn in turns] == [
                ("counselor", "ai"),
                ("client", "human"),
                ("counselor", "ai"),
            ]
            assert turns[1].speaker_role == "client"
            assert turns[1].text == stt.text
            assert turns[1].audio_path.is_file()
            compact = build_compact_session_audio(path, turns)
            assert compact is not None
            assert compact.audio_path.is_file()
            assert len(compact.turns) == 3
        finally:
            await cancel_runtime(task)

    asyncio.run(scenario())


@pytest.mark.parametrize("during_generation", [True, False])
def test_interrupting_first_ai_counselor_turn_allows_reply_and_next_input(
    tmp_path, during_generation
):
    class Counselor(FakeAgent):
        def __init__(self):
            super().__init__("counselor", "応答:{input_transcript}")
            self.started = asyncio.Event()
            self.interrupts = []
            self.released = asyncio.Event()

        async def generate(self, **kwargs):
            self.started.set()
            if during_generation and kwargs["turn_id"] == 1:
                await self.released.wait()
            return await super().generate(**kwargs)

        async def stop_current_response_playback(self, **kwargs):
            self.interrupts.append(kwargs)
            self.released.set()
            return {
                "cancel_sent": True,
                "truncate_sent": True,
                "played_ms": kwargs["played_ms"],
            }

    async def scenario():
        counselor = Counselor()
        runtime = make_runtime(tmp_path, agent=counselor)
        task = asyncio.create_task(runtime.run())
        try:
            await asyncio.wait_for(counselor.started.wait(), 2)
            if not during_generation:
                await wait_for_input(runtime, 2)
            result = await runtime.stop_current_response_playback(
                speaker="counselor",
                turn_id=1,
                played_ms=100,
                reason="human_barge_in",
            )
            assert result["human_speaker"] == "client"
            assert result["human_turn_id"] == 2
            await wait_for_input(runtime, 2)
            await runtime.submit_human_turn(
                HumanTurnInput(
                    session_id=runtime.config.session_id,
                    text="途中ですが、仕事の相談です。",
                    recipient_ids=("counselor",),
                    client_message_id="interrupt-input",
                )
            )
            await wait_for_input(runtime, 4)
            assert runtime.turns[-1].speaker == "counselor"
            assert runtime.turns[-1].turn_id == 3
            assert counselor.interrupts
            assert runtime.status.phase is RuntimePhase.RUNNING
        finally:
            await cancel_runtime(task)

    asyncio.run(scenario())


@pytest.mark.parametrize("target_turn", [1, 3])
@pytest.mark.parametrize("interrupt", [True, False])
def test_realtime_ending_before_first_output_preserves_only_interrupted_input(
    tmp_path, target_turn, interrupt
):
    from counseling_voice_demo.runtime.models import AudioChunk
    from counseling_voice_demo.runtime.realtime_speech import RealtimeSpeechSession

    class Counselor:
        def __init__(self):
            self.started = asyncio.Event()
            self.finish = asyncio.Event()
            self.cancel_sent = False
            self.session = RealtimeSpeechSession(self)

        async def send_json(self, event):
            if event["type"] == "response.cancel":
                self.cancel_sent = True

        async def __aiter__(self):
            yield {"type": "response.created", "response": {"id": "early-response"}}
            self.started.set()
            await self.finish.wait()
            # The real session turns response.done into iterator exhaustion, with
            # no text/audio event for the controller to check for interruption.
            yield {
                "type": "response.done",
                "response": {
                    "id": "early-response",
                    "status": "cancelled" if self.cancel_sent else "completed",
                },
            }

        async def stream_audio_response(self, *, input_transcript, **kwargs):
            if kwargs["turn_id"] == target_turn:
                kwargs.pop("additional_instruction", None)
                async for event in self.session.stream_audio_response(
                    latest_input=input_transcript, **kwargs
                ):
                    yield event
                return
            yield {"text_delta": "お話を聞いています。"}
            yield {
                "audio_chunk": AudioChunk(
                    session_id=kwargs["session_id"],
                    turn_id=kwargs["turn_id"],
                    speaker="counselor",
                    chunk_index=0,
                    pcm=b"\x01\x00" * 120,
                    duration_ms=5,
                )
            }

        async def stop_current_response_playback(self, **kwargs):
            return await self.session.stop_current_response_playback(**kwargs)

    async def scenario():
        counselor = Counselor()
        stt = ControlledHumanStt()
        stt.text = "続けて相談したいです。"
        runtime = make_runtime(tmp_path, agent=counselor, stt=stt, max_turns=7)
        task = asyncio.create_task(runtime.run())
        try:
            if target_turn == 3:
                await wait_for_input(runtime, 2)
                await runtime.submit_human_turn(
                    HumanTurnInput(
                        session_id=runtime.config.session_id,
                        text="仕事の相談です。",
                        recipient_ids=("counselor",),
                    )
                )
            await asyncio.wait_for(counselor.started.wait(), 1)
            if not interrupt:
                counselor.finish.set()
                with pytest.raises(
                    RuntimeError, match="AI counselor returned an empty response"
                ):
                    await asyncio.wait_for(task, 1)
                return
            result = await runtime.stop_current_response_playback(
                speaker="counselor",
                turn_id=target_turn,
                played_ms=0,
                reason="human_barge_in",
            )
            assert result["human_turn_id"] == target_turn + 1
            await runtime.start_human_audio_stream(
                stream_request(runtime, "early-input", "vad_auto")
            )
            await runtime.append_human_audio_stream_chunk(
                stream_id="early-input",
                audio_bytes=b"\x01\x00" * 2400,
                chunk_index=0,
            )
            counselor.finish.set()
            waiter = asyncio.create_task(stt.started.wait())
            try:
                done, _ = await asyncio.wait(
                    {task, waiter}, timeout=1, return_when=asyncio.FIRST_COMPLETED
                )
                if task in done:
                    await task
                assert waiter in done, "interrupted input never reached STT"
            finally:
                waiter.cancel()
                await asyncio.gather(waiter, return_exceptions=True)
            assert not runtime._human_audio_streams["early-input"].aborted.is_set()
            stt.finish.set()
            completed = await asyncio.wait_for(
                runtime.wait_human_audio_stream_completion(stream_id="early-input"), 1
            )
            assert completed["accepted"] and completed["text"] == stt.text
            await wait_for_input(runtime, target_turn + 3)
            assert runtime.turns[-1].speaker == "counselor"
            assert runtime.turns[-1].input_transcript == stt.text
            assert all(turn.turn_id != target_turn for turn in runtime.turns)
            assert runtime.status.phase is RuntimePhase.RUNNING
        finally:
            await cancel_runtime(task)

    asyncio.run(scenario())


@pytest.mark.parametrize("mode", ["push_to_talk", "browser_vad"])
def test_human_client_websocket_abort_then_retry_and_commit(tmp_path, mode):
    import base64
    import time
    from fastapi.testclient import TestClient
    from counseling_voice_demo.runtime.control_api import create_app
    from counseling_voice_demo.runtime.models import EndOfAudio

    class QueueStt:
        async def transcribe_from_queue_observed(self, **kwargs):
            assert kwargs["recording_mode"] == mode
            while not isinstance(await kwargs["queue"].get(), EndOfAudio):
                pass
            final = TranscriptEvent(
                session_id=kwargs["session_id"],
                turn_id=kwargs["turn_id"],
                speaker=kwargs["speaker"],
                transcript_type="final",
                text="相談します。",
            )
            await kwargs["on_transcript"](final)
            return [], final

    service = RuntimeControlService(lambda: make_runtime(tmp_path, stt=QueueStt()))

    def await_human(client):
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            status = client.get("/runtime/status").json()
            if status["awaiting_human_input"]:
                return status
            time.sleep(0.005)
        raise AssertionError(status)

    with TestClient(create_app(service)) as client:
        started = client.post("/runtime/start", json={}).json()
        session_id = started["session_id"]
        await_human(client)
        for number in (1, 2):
            with client.websocket_connect("/runtime/human-audio-stream") as ws:
                ws.send_json(
                    {
                        "type": "start",
                        "session_id": session_id,
                        "sample_rate": 24000,
                        "channels": 1,
                        "recording_mode": mode,
                        "recipient_ids": ["counselor"],
                        "client_message_id": f"ws-{number}",
                    }
                )
                assert ws.receive_json()["type"] == "accepted"
                ws.send_json(
                    {
                        "type": "chunk",
                        "chunk_index": 0,
                        "audio_base64": base64.b64encode(b"\x01\x00" * 2400).decode(),
                    }
                )
                assert ws.receive_json()["type"] == "chunk_received"
                ws.send_json({"type": "abort" if number == 1 else "end"})
                result = ws.receive_json()
                if result["type"] == "end_received":
                    result = ws.receive_json()
                assert result["type"] == "completed"
                assert result["completion_reason"] == (
                    "aborted" if number == 1 else "transcribed"
                )
                assert result["turn_id"] == 2
            await_human(client)
        client.post("/runtime/stop")


class ResumableCounselor(FakeAgent):
    def __init__(self, *, block_first=False):
        super().__init__("counselor", "続き:{input_transcript}")
        self.block_first = block_first
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def generate(self, **kwargs):
        self.started.set()
        if self.block_first and kwargs["turn_id"] == 1:
            await self.release.wait()
        return await super().generate(**kwargs)

    async def stop_current_response_playback(self, **kwargs):
        self.release.set()
        return {"cancel_sent": True, "truncate_sent": True}


@pytest.mark.parametrize("during_generation", [False, True])
def test_empty_always_barge_in_resumes_ai_without_inventing_a_human_turn(
    tmp_path, monkeypatch, during_generation
):
    monkeypatch.setattr(
        "counseling_voice_demo.runtime.controller.EMPTY_BARGE_IN_RESUME_DELAY_SECONDS",
        0.04, raising=False,
    )

    async def scenario():
        counselor = ResumableCounselor(block_first=during_generation)
        stt = ControlledHumanStt()
        stt.text = ""
        runtime = make_runtime(tmp_path, agent=counselor, stt=stt)
        task = asyncio.create_task(runtime.run())
        try:
            await counselor.started.wait()
            if not during_generation:
                await wait_for_input(runtime, 2)
            await runtime.stop_current_response_playback(
                speaker="counselor", turn_id=1, played_ms=100, reason="human_barge_in"
            )
            await wait_for_input(runtime, 2)
            await runtime.start_human_audio_stream(stream_request(runtime, "noise", "browser_vad"))
            await stt.started.wait()
            ended = await runtime.end_human_audio_stream(stream_id="noise")
            assert ended["pending_transcription"]
            # A delayed STT result must never be treated as silence.
            await asyncio.sleep(0.06)
            assert not any(t.turn_id == 2 for t in runtime.turns)
            stt.finish.set()
            completed = await runtime.wait_human_audio_stream_completion(stream_id="noise")
            assert completed["completion_reason"] == "blank_transcript"
            assert completed["resume_pending"] is True
            await wait_for_input(runtime, 3)
            assert all(t.speaker == "counselor" for t in runtime.turns)
            assert runtime.turns[-1].turn_id == 2
            assert "人間の発話は確認されませんでした" in counselor.received_inputs[-1]
        finally:
            await cancel_runtime(task)

    asyncio.run(scenario())


@pytest.mark.parametrize("action", ["new_audio", "pause", "stop", "text", "pause_at_resume"])
def test_empty_barge_in_recovery_is_cancelled_by_user_activity(tmp_path, monkeypatch, action):
    monkeypatch.setattr(
        "counseling_voice_demo.runtime.controller.EMPTY_BARGE_IN_RESUME_DELAY_SECONDS",
        0.06, raising=False,
    )

    async def scenario():
        stt = ControlledHumanStt()
        stt.text = ""
        runtime = make_runtime(tmp_path, agent=ResumableCounselor(), stt=stt)
        paused_at_resume = asyncio.Event()
        original_log_event = runtime.logger.log_event

        async def log_event(event):
            if action == "pause_at_resume" and event.event_type == "empty_barge_in_ai_resuming":
                runtime.pause_generation()
                paused_at_resume.set()
            await original_log_event(event)

        runtime.logger.log_event = log_event
        task = asyncio.create_task(runtime.run())
        try:
            await wait_for_input(runtime, 2)
            await runtime.stop_current_response_playback(
                speaker="counselor", turn_id=1, played_ms=100, reason="human_barge_in"
            )
            await runtime.start_human_audio_stream(stream_request(runtime, "noise", "browser_vad"))
            await runtime.end_human_audio_stream(stream_id="noise")
            stt.finish.set()
            completed = await runtime.wait_human_audio_stream_completion(stream_id="noise")
            assert completed["resume_pending"]
            await wait_for_input(runtime, 2)
            if action == "new_audio":
                stt.finish.clear()
                await runtime.start_human_audio_stream(stream_request(runtime, "real", "browser_vad"))
            elif action == "text":
                await runtime.submit_human_turn(HumanTurnInput(
                    session_id=runtime.config.session_id, text="今から話します。",
                    recipient_ids=("counselor",), client_message_id="real",
                ))
                await wait_for_input(runtime, 4)
            elif action == "pause":
                runtime.pause_generation()
            elif action == "pause_at_resume":
                await asyncio.wait_for(paused_at_resume.wait(), 1)
                runtime.resume_generation()
                await wait_for_input(runtime, 2)
            else:
                await cancel_runtime(task)
            await asyncio.sleep(0.09)
            if action == "text":
                assert [(t.turn_id, t.speaker) for t in runtime.turns] == [
                    (1, "counselor"), (2, "client"), (3, "counselor")
                ]
            else:
                assert [(t.turn_id, t.speaker) for t in runtime.turns] == [(1, "counselor")]
        finally:
            await cancel_runtime(task)

    asyncio.run(scenario())


@pytest.mark.parametrize("case", ["ordinary_silence", "push_to_talk", "stt_error", "partial_text"])
def test_empty_barge_in_recovery_requires_confirmed_empty_always_input(tmp_path, monkeypatch, case):
    monkeypatch.setattr(
        "counseling_voice_demo.runtime.controller.EMPTY_BARGE_IN_RESUME_DELAY_SECONDS", 0.02,
    )

    class Stt(ControlledHumanStt):
        async def transcribe_from_queue_observed(self, **kwargs):
            if case == "stt_error":
                from counseling_voice_demo.runtime.streaming_stt import RealtimeTranscriptionError
                raise RealtimeTranscriptionError("input_audio_buffer_commit_empty", code="input_audio_buffer_commit_empty")
            if case == "partial_text":
                await kwargs["on_transcript"](TranscriptEvent(
                    kwargs["session_id"], kwargs["turn_id"], kwargs["speaker"], "partial", "話している途中"
                ))
            return await super().transcribe_from_queue_observed(**kwargs)

    async def scenario():
        stt = Stt()
        stt.text = ""
        stt.finish.set()
        runtime = make_runtime(tmp_path, agent=ResumableCounselor(), stt=stt)
        task = asyncio.create_task(runtime.run())
        try:
            await wait_for_input(runtime, 2)
            if case != "ordinary_silence":
                await runtime.stop_current_response_playback(
                    speaker="counselor", turn_id=1, played_ms=100, reason="human_barge_in"
                )
            mode = "push_to_talk" if case == "push_to_talk" else "browser_vad"
            await runtime.start_human_audio_stream(stream_request(runtime, "blank", mode))
            await runtime.end_human_audio_stream(stream_id="blank")
            result = await runtime.wait_human_audio_stream_completion(stream_id="blank")
            assert not result.get("resume_pending")
            await asyncio.sleep(0.04)
            assert runtime.status.awaiting_human_input
            assert len(runtime.turns) == 1
        finally:
            await cancel_runtime(task)

    asyncio.run(scenario())
