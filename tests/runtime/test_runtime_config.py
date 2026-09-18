from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from pathlib import Path

import pytest

from counseling_voice_demo.client_presets import load_client_preset

from counseling_voice_demo.runtime import config as runtime_config_module
from counseling_voice_demo.runtime.config import (
    DEFAULT_RUNTIME_CONFIG_PATH,
    RuntimeConfigLoadError,
    load_runtime_config,
)
from counseling_voice_demo.runtime.factory import (
    CLIENT_SYSTEM_PROMPT,
    COUNSELOR_SYSTEM_PROMPT,
    RuntimeFactoryError,
    build_fake_runtime,
    build_openai_runtime,
    build_runtime_config_model,
    format_agent_latest_input,
)
from counseling_voice_demo.runtime.models import (
    ActorKind,
    AudioDeliveryMode,
    InteractionMode,
    ParticipantMode,
    RuntimeConfig,
)
from counseling_voice_demo.runtime.prompt_context import (
    PublicHistoryMessage,
    RuntimePromptContextData,
)
from counseling_voice_demo.runtime.speaker_selection import (
    LEGACY_ALTERNATING_SEQUENCE,
    TURN_BOUNDARY_TIMING_POLICY,
)
from counseling_voice_demo.runtime.timing_decision_prompt import (
    TIMING_DECISION_SYSTEM_PROMPT,
)

ROOT_DIR = Path(__file__).resolve().parents[2]


def test_runtime_config_defaults_to_legacy_fixed_speaker_sequence() -> None:
    runtime_config = RuntimeConfig()

    assert runtime_config.fixed_speaker_sequence == LEGACY_ALTERNATING_SEQUENCE
    assert runtime_config.speaker_selection_policy == TURN_BOUNDARY_TIMING_POLICY
    assert runtime_config.overlap_grace_ms == 200
    assert runtime_config.unresolved_overlap_limit_ms == 800
    assert runtime_config.turn_taking_signal_min_chars == 40
    assert runtime_config.turn_taking_decision_timeout_ms == 2000
    assert runtime_config.turn_taking_max_reconsider_rounds == 1


def test_timing_decision_prompt_keeps_client_target_open_after_counselor() -> None:
    assert "カウンセラーへ返す、別クライアントへ話す、今は譲る" in (
        TIMING_DECISION_SYSTEM_PROMPT
    )
    assert "別クライアントへ直接話す強さ" in TIMING_DECISION_SYSTEM_PROMPT
    assert "target は会話の進行役ではなく" in TIMING_DECISION_SYSTEM_PROMPT
    assert "機械的に counselor を選ばない" in TIMING_DECISION_SYSTEM_PROMPT
    assert "カウンセラーへの回答、質問、助言依頼" in TIMING_DECISION_SYSTEM_PROMPT
    assert "explicitly_addressed" in TIMING_DECISION_SYSTEM_PROMPT
    assert "奥さんは" in TIMING_DECISION_SYSTEM_PROMPT
    assert "他参加者が明示指名されているだけなら" in TIMING_DECISION_SYSTEM_PROMPT
    assert "自分が明示指名された質問" in TIMING_DECISION_SYSTEM_PROMPT
    assert "自分以外が明示指名されている場合" in TIMING_DECISION_SYSTEM_PROMPT
    assert "現在の agent_id 本人" in TIMING_DECISION_SYSTEM_PROMPT
    assert "指名された当人の回答を最優先" in TIMING_DECISION_SYSTEM_PROMPT
    assert "相手の同意への相づちだけなら WAIT" in TIMING_DECISION_SYSTEM_PROMPT
    assert "自分への未回答の質問・確認" in TIMING_DECISION_SYSTEM_PROMPT
    assert "カウンセラー役は、直前のクライアント発話が別クライアントへの直接" in (
        TIMING_DECISION_SYSTEM_PROMPT
    )
    assert "相手クライアントが応答する余地を残してください" in (
        TIMING_DECISION_SYSTEM_PROMPT
    )
    assert "特定の相手を常に優先せず" in TIMING_DECISION_SYSTEM_PROMPT
    assert "直前話者に関わらず" not in TIMING_DECISION_SYSTEM_PROMPT
    assert "別クライアントへ譲る WAIT を選びやすく" not in (
        TIMING_DECISION_SYSTEM_PROMPT
    )
    assert "質問への回答なら target は counselor" not in TIMING_DECISION_SYSTEM_PROMPT


def test_two_client_default_prompt_controls_couple_dialogue_balance() -> None:
    prompt = load_client_preset(
        ROOT_DIR
        / "config"
        / "client_presets"
        / "junior_high_school_refusal_couple_v1.yaml"
    ).shared.prompt

    assert "配偶者本人に向けた夫婦間の会話" in prompt
    assert "配偶者へ直接話す頻度や強さ" in prompt
    assert "自分のプロフィール、配偶者との関係、直近の発言内容" in prompt
    assert "カウンセラーに返す前に配偶者へ短く一言返す" in prompt
    assert "配偶者がタメ語・常体で確認、相談、同意、提案" in prompt
    assert "夫婦間の会話ばかりを続けず" in prompt
    assert "カウンセラーの存在も意識して対話を進めてください" in prompt
    assert "基本的に短い1文" in prompt
    assert "長くても3文まで" in prompt
    assert "2～6文程度" not in prompt
    assert "必ず毎回返す必要はありません" in prompt
    assert "終始二人だけで話し続けない" not in prompt


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("overlap_grace_ms", -1),
        ("unresolved_overlap_limit_ms", 0),
        ("turn_taking_signal_min_chars", 0),
        ("turn_taking_decision_timeout_ms", 0),
        ("turn_taking_max_reconsider_rounds", -1),
    ],
)
def test_runtime_config_rejects_invalid_overlap_timing(
    field_name: str,
    value: int,
) -> None:
    with pytest.raises(ValueError, match=field_name):
        RuntimeConfig(**{field_name: value})


def test_runtime_config_normalizes_fixed_speaker_sequence_to_tuple() -> None:
    runtime_config = RuntimeConfig(
        fixed_speaker_sequence=["counselor", "client"],  # type: ignore[arg-type]
    )

    assert runtime_config.fixed_speaker_sequence == ("counselor", "client")


@pytest.mark.parametrize(
    "fixed_speaker_sequence",
    [
        "counselor",
        (),
        ("counselor", ""),
        ("counselor", 1),
    ],
)
def test_runtime_config_rejects_invalid_fixed_speaker_sequence_shape(
    fixed_speaker_sequence: object,
) -> None:
    with pytest.raises(ValueError, match="fixed_speaker_sequence"):
        RuntimeConfig(
            fixed_speaker_sequence=fixed_speaker_sequence,  # type: ignore[arg-type]
        )


def test_runtime_config_rejects_invalid_speaker_selection_policy() -> None:
    with pytest.raises(ValueError, match="speaker_selection_policy"):
        RuntimeConfig(speaker_selection_policy="clinical_priority")


def test_default_runtime_config_uses_120_second_closing_start() -> None:
    config = load_runtime_config(DEFAULT_RUNTIME_CONFIG_PATH)

    assert config.runtime.closing_start_elapsed_seconds == 120
    assert config.runtime.force_stop_after_closing_turns == 3
    assert config.runtime.speaker_selection_policy == TURN_BOUNDARY_TIMING_POLICY
    assert config.openai.counselor_tts_voice == "shimmer"
    assert config.openai.client_tts_voice == "marin"
    assert config.openai.timing_llm_model == "gpt-5.4-mini"
    assert config.openai.timing_llm_reasoning_effort == "none"
    assert config.openai.summary_llm_model == "gpt-5.4-mini"
    assert config.openai.summary_llm_reasoning_effort == "none"
    assert config.openai.prompting_llm_model == "gpt-5.4-mini"
    assert config.openai.prompting_llm_reasoning_effort == "low"
    assert config.openai.counselor_realtime_output_speed == 1.05
    assert config.openai.client_realtime_output_speed == 0.9
    assert config.runtime.initial_client_transcript == "娘のことで相談があります"
    assert config.runtime.overlap_grace_ms == 200
    assert config.runtime.unresolved_overlap_limit_ms == 800
    assert config.runtime.turn_taking_signal_min_chars == 40
    assert config.runtime.turn_taking_decision_timeout_ms == 2000
    assert config.runtime.turn_taking_max_reconsider_rounds == 0
    assert config.runtime.conversation_context_recent_turns == 8
    assert config.runtime.session_summary_trigger_completed_turns == 12
    assert config.runtime.session_summary_update_interval_turns == 6
    assert config.latency.timing_llm_max_output_tokens == 128
    assert config.latency.summary_llm_max_output_tokens == 2048
    assert config.latency.prompting_llm_max_output_tokens == 8192
    assert config.runtime.prompt_director_enabled is True
    assert config.runtime.prompt_director_max_retries == 2
    assert config.audio.speaker_gains == {"counselor": 0.6, "client": 0.6}


@pytest.mark.parametrize("retry_limit", [-1, 6, True, "2", 1.5])
def test_runtime_config_rejects_invalid_director_retry_limits(retry_limit):
    from pydantic import ValidationError

    config = load_runtime_config()
    raw = config.runtime.model_dump()
    raw["prompt_director_max_retries"] = retry_limit
    with pytest.raises(ValidationError, match="prompt_director_max_retries"):
        type(config.runtime).model_validate(raw)


def test_load_runtime_config_reads_dotenv_and_expands_env_defaults(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.delenv("OPENAI_TEXT_MODEL", raising=False)
    monkeypatch.delenv("OPENAI_TIMING_MODEL", raising=False)
    monkeypatch.delenv("OPENAI_SUMMARY_MODEL", raising=False)
    monkeypatch.delenv("OPENAI_TTS_MODEL", raising=False)
    monkeypatch.delenv("OPENAI_STT_MODEL", raising=False)
    monkeypatch.setattr(runtime_config_module, "ROOT_DIR", tmp_path)
    (tmp_path / ".env").write_text(
        "OPENAI_TEXT_MODEL=dotenv-text-model\n", encoding="utf-8"
    )
    config_path = _write_runtime_config(tmp_path)

    config = load_runtime_config(config_path)

    assert config.openai.llm_model == "dotenv-text-model"
    assert config.openai.timing_llm_model == "default-timing-model"
    assert config.openai.timing_llm_reasoning_effort == "none"
    assert config.openai.summary_llm_model == "default-summary-model"
    assert config.openai.summary_llm_reasoning_effort == "none"
    assert config.openai.prompting_llm_model == "default-prompting-model"
    assert config.openai.prompting_llm_reasoning_effort == "none"
    assert config.openai.tts_model == "default-tts-model"
    assert config.openai.stt_model == "default-stt-model"


def test_load_runtime_config_rejects_non_pcm_tts_response_format(
    tmp_path: Path,
) -> None:
    config_path = _write_runtime_config(tmp_path, tts_response_format="mp3")

    with pytest.raises(RuntimeConfigLoadError, match="tts_response_format"):
        load_runtime_config(config_path)


def test_build_runtime_config_model_reflects_audio_settings(tmp_path: Path) -> None:
    config_path = _write_runtime_config(
        tmp_path,
        audio_delivery_mode="accelerated",
        sample_rate=48000,
        channels=2,
        tts_chunk_comma_min_chars=12,
        tts_chunk_soft_max_chars=24,
        tts_target_chunk_duration_ms=20,
        tts_sdk_chunk_size=1024,
        llm_max_output_tokens=120,
        stt_partial_prefetch=True,
        stt_partial_prefetch_min_chars=6,
        runtime_extra_block="""
  turn_taking_signal_min_chars: 3
  turn_taking_decision_timeout_ms: 5
  turn_taking_max_reconsider_rounds: 1
""",
    )
    config = load_runtime_config(config_path)

    runtime_config = build_runtime_config_model(config)

    assert runtime_config.sample_rate == 48000
    assert runtime_config.channels == 2
    assert runtime_config.speaker_audio_gains == {"counselor": 0.8, "client": 0.8}
    assert runtime_config.audio_delivery_mode is AudioDeliveryMode.ACCELERATED
    assert runtime_config.speaker_selection_policy == TURN_BOUNDARY_TIMING_POLICY
    assert runtime_config.tts_chunk_comma_min_chars == 12
    assert runtime_config.tts_chunk_soft_max_chars == 24
    assert config.latency.tts_target_chunk_duration_ms == 20
    assert config.latency.tts_sdk_chunk_size == 1024
    assert config.latency.llm_max_output_tokens == 120
    assert config.latency.timing_llm_max_output_tokens == 128
    assert runtime_config.stt_partial_prefetch is True
    assert runtime_config.stt_partial_prefetch_min_chars == 6
    assert runtime_config.overlap_grace_ms == 200
    assert runtime_config.unresolved_overlap_limit_ms == 800
    assert runtime_config.turn_taking_signal_min_chars == 3
    assert runtime_config.turn_taking_decision_timeout_ms == 5
    assert runtime_config.turn_taking_max_reconsider_rounds == 1


def test_build_runtime_config_model_reflects_speaker_selection_policy(
    tmp_path: Path,
) -> None:
    config = load_runtime_config(
        _write_runtime_config(
            tmp_path,
            runtime_extra_block="  speaker_selection_policy: distributed_timing\n",
        )
    )

    runtime_config = build_runtime_config_model(config)

    assert config.runtime.speaker_selection_policy == "distributed_timing"
    assert runtime_config.speaker_selection_policy == "distributed_timing"


def test_build_runtime_config_model_preserves_two_client_participants(
    tmp_path: Path,
) -> None:
    config = load_runtime_config(
        _write_runtime_config(
            tmp_path,
            runtime_extra_block="""
  fixed_speaker_sequence:
    - counselor
    - client_a
    - counselor
    - client_b
""",
            participants_block="""
participants:
  counselor:
    role: counselor
    display_name: カウンセラー
    voice: shimmer
    realtime_output_speed: 1.0
  client_a:
    role: client
    display_name: クライアントA
    initial_transcript: Aの初回発話です。
    voice: cedar
    realtime_output_speed: 0.9
  client_b:
    role: client
    display_name: クライアントB
    initial_transcript: Bの初回発話です。
    voice: coral
    realtime_output_speed: 0.85
""",
        )
    )

    runtime_config = build_runtime_config_model(config)

    assert set(runtime_config.participants) == {"counselor", "client_a", "client_b"}
    assert runtime_config.participants["counselor"].role == "counselor"
    assert runtime_config.participants["client_a"].speaker_id == "client_a"
    assert runtime_config.participants["client_a"].display_name == "クライアントA"
    assert runtime_config.participants["client_a"].initial_transcript == (
        "Aの初回発話です。"
    )
    assert runtime_config.participants["client_b"].voice == "coral"
    assert runtime_config.participants["client_b"].realtime_output_speed == 0.85


def test_build_runtime_config_model_preserves_human_counselor_mode(
    tmp_path: Path,
) -> None:
    config = load_runtime_config(
        _write_runtime_config(
            tmp_path,
            runtime_extra_block="""
  interaction_mode: human_counselor_ai_client
  participant_mode: two_clients
  human_input_mode: push_to_talk
  human_stt_submit_policy: auto_on_final
  human_interrupts_enabled: false
  fixed_speaker_sequence:
    - counselor
    - client_a
    - client_b
""",
            participants_block="""
participants:
  counselor:
    role: counselor
    display_name: カウンセラー
    actor_kind: human
  client_a:
    role: client
    display_name: 妻
  client_b:
    role: client
    display_name: 夫
""",
        )
    )

    runtime_config = build_runtime_config_model(config)

    assert runtime_config.interaction_mode is InteractionMode.HUMAN_COUNSELOR_AI_CLIENT
    assert runtime_config.participant_mode is ParticipantMode.TWO_CLIENTS
    assert runtime_config.human_input_mode == "push_to_talk"
    assert runtime_config.human_stt_submit_policy == "auto_on_final"
    assert runtime_config.human_interrupts_enabled is False
    assert runtime_config.participants["counselor"].actor_kind is ActorKind.HUMAN


def test_build_runtime_config_model_reads_fixed_speaker_sequence(
    tmp_path: Path,
) -> None:
    config = load_runtime_config(
        _write_runtime_config(
            tmp_path,
            runtime_extra_block="""
  fixed_speaker_sequence:
    - counselor
    - client_a
    - counselor
    - client_b
""",
            participants_block="""
participants:
  counselor:
    role: counselor
    display_name: カウンセラー
  client_a:
    role: client
    display_name: クライアントA
  client_b:
    role: client
    display_name: クライアントB
""",
        )
    )

    runtime_config = build_runtime_config_model(config)

    assert runtime_config.fixed_speaker_sequence == (
        "counselor",
        "client_a",
        "counselor",
        "client_b",
    )


def test_build_fake_runtime_uses_runtime_config_without_api_clients(
    tmp_path: Path,
) -> None:
    config_path = _write_runtime_config(tmp_path, sample_rate=16000, channels=1)
    config = load_runtime_config(config_path)

    runtime = build_fake_runtime(config)

    assert runtime.config.sample_rate == 16000
    assert runtime.config.channels == 1
    assert runtime.controller.tts.__class__.__name__ == "FakeStreamingTTS"
    assert runtime.controller.stt.__class__.__name__ == "FakeStreamingSTT"


def test_build_fake_runtime_creates_agents_for_two_client_participants(
    tmp_path: Path,
) -> None:
    config = load_runtime_config(
        _write_runtime_config(
            tmp_path,
            runtime_extra_block="""
  fixed_speaker_sequence:
    - counselor
    - client_a
    - counselor
    - client_b
""",
            participants_block="""
participants:
  counselor:
    role: counselor
    display_name: カウンセラー
  client_a:
    role: client
    display_name: クライアントA
  client_b:
    role: client
    display_name: クライアントB
""",
        )
    )

    runtime = build_fake_runtime(config)

    assert set(runtime.controller.agents) == {"counselor", "client_a", "client_b"}
    assert runtime.controller.agents["counselor"].speaker == "counselor"
    assert runtime.controller.agents["client_a"].speaker == "client_a"
    assert runtime.controller.agents["client_b"].speaker == "client_b"
    speakers = [runtime.controller.speaker_for_turn(turn_id) for turn_id in range(1, 5)]
    assert speakers == [
        "counselor",
        "client_a",
        "counselor",
        "client_b",
    ]


def test_build_fake_runtime_skips_human_counselor_agent(
    tmp_path: Path,
) -> None:
    config = load_runtime_config(
        _write_runtime_config(
            tmp_path,
            runtime_extra_block="""
  interaction_mode: human_counselor_ai_client
  participant_mode: one_client
  fixed_speaker_sequence:
    - counselor
    - client
""",
            participants_block="""
participants:
  counselor:
    role: counselor
    display_name: カウンセラー
    actor_kind: human
  client:
    role: client
    display_name: クライアント
""",
        )
    )

    runtime = build_fake_runtime(config)

    assert set(runtime.controller.agents) == {"client"}


@pytest.mark.parametrize(
    ("prompting_llm_max_output_tokens", "expected_prompting_limit"),
    [(None, 8192), (4096, 4096)],
)
def test_build_openai_runtime_wires_openai_clients_without_network(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    prompting_llm_max_output_tokens: int | None,
    expected_prompting_limit: int,
) -> None:
    monkeypatch.setenv("OPENAI_TTS_MODEL", "default-tts-model")
    config_path = _write_runtime_config(
        tmp_path,
        audio_delivery_mode="accelerated",
        tts_target_chunk_duration_ms=20,
        tts_sdk_chunk_size=1024,
        llm_max_output_tokens=120,
        prompting_llm_max_output_tokens=prompting_llm_max_output_tokens,
    )
    config = load_runtime_config(config_path)
    fake_client = object()
    fake_stt = object()

    runtime = build_openai_runtime(config, openai_client=fake_client, stt=fake_stt)

    assert runtime.config.audio_delivery_mode is AudioDeliveryMode.ACCELERATED
    assert runtime.controller.tts.__class__.__name__ == "OpenAIStreamingTTS"
    assert runtime.controller.tts._client is fake_client
    assert runtime.controller.tts._config.model == "default-tts-model"
    assert runtime.controller.tts._config.voice == "coral"
    assert runtime.controller.tts._config.target_chunk_duration_ms == 20
    assert runtime.controller.tts._config.sdk_chunk_size == 1024
    assert runtime.controller.tts._config.voice_by_speaker == {
        "counselor": "coral",
        "client": "cedar",
    }
    assert runtime.controller.stt is fake_stt
    assert set(runtime.controller.agents) == {"counselor", "client"}
    assert all(
        agent.__class__.__name__ == "StreamingAgent"
        for agent in runtime.controller.agents.values()
    )
    counselor_prompt = runtime.controller.agents["counselor"].llm._system_prompt
    assert counselor_prompt == COUNSELOR_SYSTEM_PROMPT
    client_prompt = runtime.controller.agents["client"].llm._system_prompt
    assert client_prompt == CLIENT_SYSTEM_PROMPT
    assert runtime.controller.agents["counselor"].llm._max_output_tokens == 120
    assert runtime.controller.agents["client"].llm._max_output_tokens == 120
    assert runtime.controller.agents["counselor"].timing_llm._model == (
        "default-timing-model"
    )
    assert runtime.controller.agents["client"].timing_llm._model == (
        "default-timing-model"
    )
    assert runtime.controller.agents["counselor"].timing_llm._max_output_tokens == 128
    assert runtime.controller.agents["client"].timing_llm._max_output_tokens == 128
    assert runtime.controller.agents["counselor"].timing_llm._reasoning_effort == (
        "none"
    )
    assert runtime.controller.agents["client"].timing_llm._reasoning_effort == "none"
    assert runtime.session_summary_llm._client is fake_client
    assert runtime.session_summary_llm._model == "default-summary-model"
    assert runtime.session_summary_llm._max_output_tokens == 2048
    assert runtime.session_summary_llm._reasoning_effort == "none"
    assert runtime.prompt_director is not None
    assert runtime.prompt_director.llm._client is fake_client
    assert runtime.prompt_director.max_retries == 2
    assert runtime.prompt_director.llm._model == "default-prompting-model"
    director_payload = runtime.prompt_director.llm.build_payload(latest_input="確認")
    assert director_payload["max_output_tokens"] == expected_prompting_limit
    assert runtime.prompt_director.llm._reasoning_effort == "none"
    assert runtime.prompt_director.llm._text_format["type"] == "json_schema"
    assert runtime.prompt_context_store is not None
    assert callable(runtime.controller.agents["counselor"].input_formatter)
    assert callable(runtime.controller.agents["client"].input_formatter)


def test_build_openai_runtime_uses_participant_voices_for_streaming_tts(
    tmp_path: Path,
) -> None:
    config = load_runtime_config(
        _write_runtime_config(
            tmp_path,
            participants_block="""
participants:
  counselor:
    role: counselor
    display_name: カウンセラー
    voice: shimmer
  client:
    role: client
    display_name: クライアント
    voice: nova
""",
        )
    )

    runtime = build_openai_runtime(config, openai_client=object(), stt=object())

    assert runtime.controller.tts._config.voice_by_speaker == {
        "counselor": "shimmer",
        "client": "nova",
    }


def test_build_openai_runtime_creates_agents_for_two_client_participants(
    tmp_path: Path,
) -> None:
    config = load_runtime_config(
        _write_runtime_config(
            tmp_path,
            participants_block="""
participants:
  counselor:
    role: counselor
    display_name: カウンセラー
    voice: shimmer
  client_a:
    role: client
    display_name: クライアントA
    voice: echo
  client_b:
    role: client
    display_name: クライアントB
""",
        )
    )

    runtime = build_openai_runtime(config, openai_client=object(), stt=object())

    assert runtime.controller.tts._config.voice_by_speaker == {
        "counselor": "shimmer",
        "client": "cedar",
        "client_a": "echo",
        "client_b": "cedar",
    }
    assert set(runtime.controller.agents) == {"counselor", "client_a", "client_b"}
    assert runtime.controller.agents["counselor"].speaker == "counselor"
    assert runtime.controller.agents["client_a"].speaker == "client_a"
    assert runtime.controller.agents["client_b"].speaker == "client_b"
    assert "カウンセラーの役割を維持" in (
        runtime.controller.agents["counselor"].llm._system_prompt
    )
    assert (
        runtime.controller.agents["client_a"].llm._system_prompt == CLIENT_SYSTEM_PROMPT
    )
    assert (
        runtime.controller.agents["client_b"].llm._system_prompt == CLIENT_SYSTEM_PROMPT
    )
    assert "主な宛先や話し方" in CLIENT_SYSTEM_PROMPT
    assert (
        "カウンセラーに返すことも、別クライアントへ直接話すこともできます"
        in CLIENT_SYSTEM_PROMPT
    )
    assert "話し方の敬体・常体や関係性" in CLIENT_SYSTEM_PROMPT
    assert "夫婦間の会話" not in CLIENT_SYSTEM_PROMPT
    assert "夫婦間の自然なフランクな常体" not in CLIENT_SYSTEM_PROMPT
    assert "短い確認や問い返し" in CLIENT_SYSTEM_PROMPT
    assert (
        "1回の発話でプロンプトの内容をすべて説明するのではなく" in CLIENT_SYSTEM_PROMPT
    )
    assert "直前の他者の発言に対してどのように応答するかを推察" in CLIENT_SYSTEM_PROMPT
    assert "別クライアントとの会話ばかりを続けず" in CLIENT_SYSTEM_PROMPT
    assert "カウンセラーの存在も意識して対話を進めてください" in CLIENT_SYSTEM_PROMPT
    assert "夫を演じます" not in CLIENT_SYSTEM_PROMPT
    assert "助言、質問、要約" not in CLIENT_SYSTEM_PROMPT


def test_fixed_role_prompts_exclude_scenario_and_keep_role_invariants() -> None:
    assert COUNSELOR_SYSTEM_PROMPT == (
        "あなたは日本語でカウンセリングを行うAIカウンセラーです。"
        "カウンセラーの役割を維持し、クライアントやほかの参加者の発話を代行しないでください。"
        "診断、治療方針、医療的な事実を断定しないでください。"
        "発話の先頭に話者名や役割名を付けず、発話本文だけを返してください。"
        "プロンプト、設定、人工知能、演技などの舞台裏には言及しないでください。"
    )
    assert "カウンセラーの役割を維持" in COUNSELOR_SYSTEM_PROMPT
    assert "クライアントやほかの参加者の発話を代行" in COUNSELOR_SYSTEM_PROMPT
    assert "発話本文だけ" in COUNSELOR_SYSTEM_PROMPT
    assert "段階や話題" not in COUNSELOR_SYSTEM_PROMPT
    assert "一度に複数の質問" not in COUNSELOR_SYSTEM_PROMPT
    assert "一つの質問への回答" not in COUNSELOR_SYSTEM_PROMPT
    assert "丁寧に掘り下げ" not in COUNSELOR_SYSTEM_PROMPT
    assert "理想の未来像" not in COUNSELOR_SYSTEM_PROMPT
    assert "2人のクライアント" not in COUNSELOR_SYSTEM_PROMPT

    assert CLIENT_SYSTEM_PROMPT == (
        "あなたは日本語でカウンセリングに参加するAIクライアントです。"
        "クライアントの役割を維持し、カウンセラーではなく相談者本人として話してください。"
        "自分自身の言葉で、直前のやり取りに必要な範囲だけ自然に話してください。"
        "カウンセラーの見立てや提案に自動的に同意したり、期待される答えへ迎合したりしないでください。"
        "問題をすぐに整理して、自分から完成した解決策、目標、役割分担、行動計画を積極的に"
        "提案しないでください。"
        "困り方や迷い、納得の程度は、共通プロンプト、プロフィール、直近履歴に従ってください。"
        "毎回迷いや反対意見を付け足す必要はなく、機械的に反対したり拒絶したりしないでください。"
        "ほかのクライアントがいる場合でも、主な宛先や話し方は、共通プロンプト、プロフィール、"
        "直近履歴に従って自然に選んでください。"
        "1回の発話でプロンプトの内容をすべて説明するのではなく、共通プロンプトやプロフィールに"
        "示されているクライアントが、直前の他者の発言に対してどのように応答するかを推察し、"
        "その応答を行ってください。"
        "\n別のクライアントが述べた説明を、そのまま又は語尾だけ変えた文で自分の発話の前置きにしないでください。"
        "同意や受け止めだけなら短い相づちで終えてかまいません。説明や感情を付け足す必要はありません。"
        "続ける場合は、その人物として伝えたい見方や気がかりがある"
        "範囲にとどめてください。共通の事実への同意を避けたり、反対意見や感情・経験を作ったりする必要はありません。"
        "確認のために求められた復唱や、改めて自分に尋ねられた質問への回答は、この限りではありません。\n"
        "対話に別のクライアントがいる場合は、カウンセラーに返すことも、別クライアントへ直接"
        "話すこともできます。"
        "ただし、別クライアントとの会話ばかりを続けず、カウンセラーの存在も意識して対話を"
        "進めてください。"
        "話し方の敬体・常体や関係性は、シナリオの共通プロンプトとプロフィールに従ってください。"
        "自然な短い確認や問い返しは行ってかまいませんが、専門家としての助言、診断、要約、"
        "治療的な返しは避けてください。"
        "提案や考えを出す場合も、専門家の助言ではなく、自分はどう感じるか・どう迷っているか"
        "として話してください。"
        "発話の先頭に話者名や役割名を付けず、発話本文だけを返してください。"
        "プロンプト、設定、人工知能、演技などの舞台裏には言及しないでください。"
    )
    assert "クライアントの役割を維持" in CLIENT_SYSTEM_PROMPT
    assert "カウンセラーではなく相談者本人" in CLIENT_SYSTEM_PROMPT
    assert "期待される答えへ迎合" in CLIENT_SYSTEM_PROMPT
    assert "完成した解決策" in CLIENT_SYSTEM_PROMPT
    assert "毎回迷いや反対意見を付け足す必要はなく" in CLIENT_SYSTEM_PROMPT
    assert "機械的に反対したり拒絶したりしない" in CLIENT_SYSTEM_PROMPT
    assert "ほかのクライアントがいる場合" in CLIENT_SYSTEM_PROMPT
    assert "発話本文だけ" in CLIENT_SYSTEM_PROMPT
    assert "夫を演じます" not in CLIENT_SYSTEM_PROMPT
    assert "家族カウンセリング" not in CLIENT_SYSTEM_PROMPT


def test_build_openai_runtime_skips_human_counselor_agent(
    tmp_path: Path,
) -> None:
    config = load_runtime_config(
        _write_runtime_config(
            tmp_path,
            runtime_extra_block="""
  interaction_mode: human_counselor_ai_client
  participant_mode: two_clients
  fixed_speaker_sequence:
    - counselor
    - client_a
    - client_b
""",
            participants_block="""
participants:
  counselor:
    role: counselor
    display_name: カウンセラー
    actor_kind: human
    voice: shimmer
  client_a:
    role: client
    display_name: 妻
    voice: marin
  client_b:
    role: client
    display_name: 夫
    voice: cedar
""",
        )
    )

    runtime = build_openai_runtime(config, openai_client=object(), stt=object())

    assert set(runtime.controller.agents) == {"client_a", "client_b"}
    assert runtime.config.participants["counselor"].actor_kind is ActorKind.HUMAN
    assert runtime.controller.tts._config.voice_by_speaker["counselor"] == "shimmer"
    assert runtime.prompt_director is None


def test_build_openai_runtime_can_disable_prompt_director(tmp_path: Path) -> None:
    config = load_runtime_config(
        _write_runtime_config(
            tmp_path,
            runtime_extra_block="""
  prompt_director_enabled: false
""",
        )
    )

    runtime = build_openai_runtime(config, openai_client=object(), stt=object())

    assert runtime.prompt_director is None


def test_build_runtime_config_model_resolves_prompt_paths_and_shared_case(
    tmp_path: Path,
) -> None:
    shared_case_path = tmp_path / "shared_case.md"
    counselor_prompt_path = tmp_path / "counselor.md"
    client_a_public_path = tmp_path / "client_a_public.md"
    client_a_private_path = tmp_path / "client_a_private.md"
    shared_case_path.write_text("SHARED_CASE_FROM_PATH", encoding="utf-8")
    counselor_prompt_path.write_text("COUNSELOR_PROMPT_FROM_PATH", encoding="utf-8")
    client_a_public_path.write_text("CLIENT_A_PUBLIC_FROM_PATH", encoding="utf-8")
    client_a_private_path.write_text("CLIENT_A_PRIVATE_FROM_PATH", encoding="utf-8")
    config = load_runtime_config(
        _write_runtime_config(
            tmp_path,
            shared_case_block=f"""
shared_case:
  prompt_path: {shared_case_path}
""",
            participants_block=f"""
participants:
  counselor:
    role: counselor
    display_name: カウンセラー
    prompt_path: {counselor_prompt_path}
  client_a:
    role: client
    display_name: クライアントA
    public_profile_path: {client_a_public_path}
    private_profile_path: {client_a_private_path}
  client_b:
    role: client
    display_name: クライアントB
    prompt_source: CLIENT_B_INLINE_PROMPT
""",
        )
    )

    runtime_config = build_runtime_config_model(config)

    assert runtime_config.shared_case == "SHARED_CASE_FROM_PATH"
    assert runtime_config.participants["counselor"].prompt_source == (
        "COUNSELOR_PROMPT_FROM_PATH"
    )
    assert runtime_config.participants["client_a"].private_profile_source == (
        "CLIENT_A_PRIVATE_FROM_PATH"
    )
    assert runtime_config.participants["client_a"].public_profile_source == (
        "CLIENT_A_PUBLIC_FROM_PATH"
    )
    assert runtime_config.participants["client_b"].prompt_source == (
        "CLIENT_B_INLINE_PROMPT"
    )


def test_build_runtime_config_model_wraps_prompt_source_read_errors(
    tmp_path: Path,
) -> None:
    config = load_runtime_config(
        _write_runtime_config(
            tmp_path,
            shared_case_block=f"""
shared_case:
  prompt_path: {tmp_path / "missing_shared_case.md"}
""",
        )
    )

    with pytest.raises(RuntimeFactoryError, match="missing_shared_case.md"):
        build_runtime_config_model(config)


def test_build_openai_runtime_uses_prompt_context_input_formatter_for_sources(
    tmp_path: Path,
) -> None:
    config = load_runtime_config(
        _write_runtime_config(
            tmp_path,
            shared_case_block="""
shared_case:
  prompt_source: SHARED_CASE_INLINE_TOKEN
""",
            participants_block="""
participants:
  counselor:
    role: counselor
    display_name: カウンセラー
    prompt_source: COUNSELOR_PROMPT_INLINE_TOKEN
    public_profile_source: COUNSELOR_PUBLIC_INLINE_TOKEN
  client_a:
    role: client
    display_name: クライアントA
    public_profile_source: CLIENT_A_PUBLIC_INLINE_TOKEN
    private_profile_source: CLIENT_A_PRIVATE_INLINE_TOKEN
  client_b:
    role: client
    display_name: クライアントB
    public_profile_source: CLIENT_B_PUBLIC_INLINE_TOKEN
    private_profile_source: CLIENT_B_PRIVATE_INLINE_TOKEN
""",
        )
    )

    runtime = build_openai_runtime(config, openai_client=object(), stt=object())
    counselor_prompt = runtime.controller.agents["counselor"].input_formatter(
        speaker="counselor",
        turn_id=1,
        input_transcript="Aさんが話しました。",
    )
    client_a_prompt = runtime.controller.agents["client_a"].input_formatter(
        speaker="client_a",
        turn_id=2,
        input_transcript="カウンセラーが話しました。",
    )
    client_b_prompt = runtime.controller.agents["client_b"].input_formatter(
        speaker="client_b",
        turn_id=4,
        input_transcript="カウンセラーが話しました。",
    )

    assert "SHARED_CASE_INLINE_TOKEN" in counselor_prompt
    assert "COUNSELOR_PROMPT_INLINE_TOKEN" in counselor_prompt
    assert "COUNSELOR_PUBLIC_INLINE_TOKEN" in counselor_prompt
    assert "CLIENT_A_PUBLIC_INLINE_TOKEN" not in counselor_prompt
    assert "CLIENT_A_PRIVATE_INLINE_TOKEN" not in counselor_prompt
    assert "CLIENT_B_PRIVATE_INLINE_TOKEN" not in counselor_prompt
    assert "COUNSELOR_PUBLIC_INLINE_TOKEN" in client_a_prompt
    assert "CLIENT_A_PUBLIC_INLINE_TOKEN" in client_a_prompt
    assert "CLIENT_B_PUBLIC_INLINE_TOKEN" not in client_a_prompt
    assert "CLIENT_A_PRIVATE_INLINE_TOKEN" in client_a_prompt
    assert "CLIENT_B_PRIVATE_INLINE_TOKEN" not in client_a_prompt
    assert "COUNSELOR_PUBLIC_INLINE_TOKEN" in client_b_prompt
    assert "CLIENT_B_PUBLIC_INLINE_TOKEN" in client_b_prompt
    assert "CLIENT_A_PUBLIC_INLINE_TOKEN" not in client_b_prompt
    assert "CLIENT_B_PRIVATE_INLINE_TOKEN" in client_b_prompt
    assert "CLIENT_A_PRIVATE_INLINE_TOKEN" not in client_b_prompt


def test_format_agent_latest_input_keeps_client_from_answering_as_counselor() -> None:
    formatted = format_agent_latest_input(
        speaker="client",
        turn_id=2,
        input_transcript="今日は家族のことで辛い点を教えてください。",
    )

    assert "生成対象: turn=2 speaker=client" in formatted
    assert "直前のカウンセラー発話" in formatted
    assert "今日は家族のことで辛い点を教えてください。" in formatted
    assert "あなたは" not in formatted


def test_format_agent_latest_input_labels_counselor_input_as_client_utterance() -> None:
    formatted = format_agent_latest_input(
        speaker="counselor",
        turn_id=1,
        input_transcript="家族との距離感に悩んでいます。",
    )

    assert "生成対象: turn=1 speaker=counselor" in formatted
    assert "直前のクライアント発話" in formatted
    assert "家族との距離感に悩んでいます。" in formatted
    assert "あなたは" not in formatted


def test_build_openai_runtime_uses_client_factory_and_api_key(tmp_path: Path) -> None:
    config = load_runtime_config(_write_runtime_config(tmp_path))
    created_with_api_keys: list[str] = []
    fake_client = object()

    def client_factory(api_key: str) -> object:
        created_with_api_keys.append(api_key)
        return fake_client

    runtime = build_openai_runtime(
        config,
        api_key="test-api-key",
        client_factory=client_factory,
        stt=object(),
    )

    assert created_with_api_keys == ["test-api-key"]
    assert runtime.controller.tts._client is fake_client


def test_build_openai_runtime_creates_realtime_stt_by_default(tmp_path: Path) -> None:
    config = load_runtime_config(_write_runtime_config(tmp_path))

    runtime = build_openai_runtime(
        config, api_key="test-api-key", openai_client=object()
    )

    assert runtime.controller.stt.__class__.__name__ == "OpenAIRealtimeTranscriptionSTT"
    assert runtime.controller.stt.model == config.openai.stt_model


def test_build_openai_runtime_can_use_realtime_api_centered_mode(
    tmp_path: Path,
) -> None:
    config = load_runtime_config(
        _write_runtime_config(
            tmp_path,
            realtime_api_centered_mode=True,
            tts_target_chunk_duration_ms=20,
        )
    )

    runtime = build_openai_runtime(
        config,
        api_key="test-api-key",
        stt=object(),
    )

    assert runtime.controller.tts.__class__.__name__ == "FakeStreamingTTS"
    assert (
        runtime.controller.agents["counselor"].__class__.__name__
        == "OpenAIRealtimeSpeechAgent"
    )
    assert (
        runtime.controller.agents["client"].__class__.__name__
        == "OpenAIRealtimeSpeechAgent"
    )
    assert runtime.controller.agents["counselor"].config.model == "gpt-realtime-2.1"
    assert runtime.controller.agents["counselor"].config.voice == "coral"
    assert runtime.controller.agents["counselor"].config.output_speed == 0.85
    assert runtime.controller.agents["client"].config.voice == "cedar"
    assert runtime.controller.agents["client"].config.output_speed == 0.95
    assert runtime.controller.agents["counselor"].config.target_chunk_duration_ms == 20
    assert runtime.prompt_context_store is not None
    assert callable(runtime.controller.agents["counselor"].input_formatter)
    assert callable(runtime.controller.agents["client"].input_formatter)
    assert runtime.controller.agents["counselor"].timing_llm._model == (
        "default-timing-model"
    )
    assert runtime.controller.agents["counselor"].timing_llm._reasoning_effort == (
        "none"
    )


def test_realtime_runtime_repeats_fixed_and_session_counselor_prompts_per_turn(
    tmp_path: Path,
) -> None:
    config = load_runtime_config(
        _write_runtime_config(
            tmp_path,
            realtime_api_centered_mode=True,
            participants_block="""
participants:
  counselor:
    role: counselor
    display_name: カウンセラー
    prompt_source: Session setupカウンセラープロンプト
  client:
    role: client
    display_name: クライアント
    prompt_source: クライアントプロンプト
""",
        )
    )

    runtime = build_openai_runtime(
        config,
        api_key="test-api-key",
        stt=object(),
    )

    counselor_config = runtime.controller.agents["counselor"].config
    client_config = runtime.controller.agents["client"].config
    assert counselor_config.instructions == COUNSELOR_SYSTEM_PROMPT
    assert counselor_config.response_instructions == (
        "Session setupカウンセラープロンプト"
    )
    assert counselor_config.repeat_instructions_per_response is True
    assert client_config.instructions == CLIENT_SYSTEM_PROMPT
    assert client_config.response_instructions == "クライアントプロンプト"
    assert client_config.repeat_instructions_per_response is True
    assert runtime.controller.agents["client"].response_input_formatter is not None


@pytest.mark.parametrize(
    "speaker_selection_policy",
    ["distributed_timing", TURN_BOUNDARY_TIMING_POLICY],
)
def test_build_openai_runtime_wires_timing_llm_in_realtime_timing_mode(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    speaker_selection_policy: str,
) -> None:
    monkeypatch.setenv("OPENAI_TEXT_MODEL", "default-text-model")
    monkeypatch.setenv("OPENAI_TIMING_MODEL", "fast-timing-model")
    fake_client = object()
    config = load_runtime_config(
        _write_runtime_config(
            tmp_path,
            realtime_api_centered_mode=True,
            runtime_extra_block=(
                f"  speaker_selection_policy: {speaker_selection_policy}\n"
            ),
        )
    )

    runtime = build_openai_runtime(
        config,
        api_key="test-api-key",
        openai_client=fake_client,
        stt=object(),
    )

    assert runtime.controller.agents["counselor"].timing_llm._client is fake_client
    assert runtime.controller.agents["client"].timing_llm._client is fake_client
    assert runtime.controller.agents["counselor"].timing_llm._model == (
        "fast-timing-model"
    )
    assert runtime.controller.agents["client"].timing_llm._model == (
        "fast-timing-model"
    )
    assert runtime.controller.agents["counselor"].timing_llm._reasoning_effort == (
        "none"
    )


def test_build_openai_runtime_uses_participant_voice_and_speed_in_realtime_mode(
    tmp_path: Path,
) -> None:
    config = load_runtime_config(
        _write_runtime_config(
            tmp_path,
            realtime_api_centered_mode=True,
            participants_block="""
participants:
  counselor:
    role: counselor
    display_name: カウンセラー
    voice: shimmer
    realtime_output_speed: 1.2
  client:
    role: client
    display_name: クライアント
    voice: nova
    realtime_output_speed: 0.75
""",
        )
    )

    runtime = build_openai_runtime(
        config,
        api_key="test-api-key",
        stt=object(),
    )

    assert runtime.controller.agents["counselor"].config.voice == "shimmer"
    assert runtime.controller.agents["counselor"].config.output_speed == 1.2
    assert runtime.controller.agents["client"].config.voice == "nova"
    assert runtime.controller.agents["client"].config.output_speed == 0.75


def test_build_openai_runtime_creates_realtime_agents_for_two_client_participants(
    tmp_path: Path,
) -> None:
    config = load_runtime_config(
        _write_runtime_config(
            tmp_path,
            realtime_api_centered_mode=True,
            runtime_extra_block="""
  fixed_speaker_sequence:
    - counselor
    - client_a
    - counselor
    - client_b
""",
            participants_block="""
participants:
  counselor:
    role: counselor
    display_name: カウンセラー
    voice: shimmer
    realtime_output_speed: 1.1
  client_a:
    role: client
    display_name: クライアントA
    voice: echo
    realtime_output_speed: 0.9
  client_b:
    role: client
    display_name: クライアントB
    voice: coral
    realtime_output_speed: 0.8
""",
        )
    )

    runtime = build_openai_runtime(
        config,
        api_key="test-api-key",
        stt=object(),
    )

    assert set(runtime.controller.agents) == {"counselor", "client_a", "client_b"}
    assert runtime.controller.agents["counselor"].speaker == "counselor"
    assert runtime.controller.agents["client_a"].speaker == "client_a"
    assert runtime.controller.agents["client_b"].speaker == "client_b"
    assert runtime.controller.agents["counselor"].config.voice == "shimmer"
    assert runtime.controller.agents["client_a"].config.voice == "echo"
    assert runtime.controller.agents["client_b"].config.voice == "coral"
    assert runtime.controller.agents["counselor"].config.output_speed == 1.1
    assert runtime.controller.agents["client_a"].config.output_speed == 0.9
    assert runtime.controller.agents["client_b"].config.output_speed == 0.8
    assert "カウンセラーの役割を維持" in (
        runtime.controller.agents["counselor"].config.instructions
    )
    assert (
        runtime.controller.agents["client_a"].config.instructions
        == CLIENT_SYSTEM_PROMPT
    )
    assert (
        runtime.controller.agents["client_b"].config.instructions
        == CLIENT_SYSTEM_PROMPT
    )
    assert "主な宛先や話し方" in CLIENT_SYSTEM_PROMPT
    assert (
        "カウンセラーに返すことも、別クライアントへ直接話すこともできます"
        in CLIENT_SYSTEM_PROMPT
    )
    assert "話し方の敬体・常体や関係性" in CLIENT_SYSTEM_PROMPT
    assert "夫婦間の会話" not in CLIENT_SYSTEM_PROMPT
    assert "夫婦間の自然なフランクな常体" not in CLIENT_SYSTEM_PROMPT
    assert (
        "1回の発話でプロンプトの内容をすべて説明するのではなく" in CLIENT_SYSTEM_PROMPT
    )
    assert "直前の他者の発言に対してどのように応答するかを推察" in CLIENT_SYSTEM_PROMPT
    assert "別クライアントとの会話ばかりを続けず" in CLIENT_SYSTEM_PROMPT
    assert "カウンセラーの存在も意識して対話を進めてください" in CLIENT_SYSTEM_PROMPT


def test_build_openai_runtime_uses_prompt_context_formatter_in_realtime_mode(
    tmp_path: Path,
) -> None:
    config = load_runtime_config(
        _write_runtime_config(
            tmp_path,
            realtime_api_centered_mode=True,
            shared_case_block="""
shared_case:
  prompt_source: REALTIME_SHARED_CASE_TOKEN
""",
            participants_block="""
participants:
  counselor:
    role: counselor
    display_name: カウンセラー
  client_a:
    role: client
    display_name: クライアントA
    private_profile_source: REALTIME_CLIENT_A_PRIVATE_TOKEN
  client_b:
    role: client
    display_name: クライアントB
    private_profile_source: REALTIME_CLIENT_B_PRIVATE_TOKEN
""",
        )
    )

    runtime = build_openai_runtime(
        config,
        api_key="test-api-key",
        stt=object(),
    )

    client_a_prompt = runtime.controller.agents["client_a"].input_formatter(
        speaker="client_a",
        turn_id=2,
        input_transcript="カウンセラーが話しました。",
    )

    assert "REALTIME_SHARED_CASE_TOKEN" in client_a_prompt
    assert "REALTIME_CLIENT_A_PRIVATE_TOKEN" in client_a_prompt
    assert "REALTIME_CLIENT_B_PRIVATE_TOKEN" not in client_a_prompt


@pytest.mark.parametrize("speaker", ["client_a", "client_b"])
def test_realtime_clients_keep_presets_profiles_and_everyones_history_without_script(
    tmp_path: Path,
    speaker: str,
) -> None:
    async def scenario():
        config = load_runtime_config(
            _write_runtime_config(
                tmp_path,
                realtime_api_centered_mode=True,
                participants_block="""
participants:
  counselor:
    role: counselor
    display_name: 相談員
  client_a:
    role: client
    display_name: 妻
    prompt_source: Aのプリセット
    private_profile_source: Aだけの設定
  client_b:
    role: client
    display_name: 夫
    prompt_source: Bのプリセット
    private_profile_source: Bだけの設定
""",
            )
        )
        runtime = build_openai_runtime(
            config,
            openai_client=object(),
            api_key="test-api-key",
            stt=object(),
        )
        assert runtime.prompt_director is not None
        # Any accidental director call would fail against the inert text client.
        agent = runtime.controller.agents[speaker]
        sent = []

        class Transport:
            def send_json(self, event):
                sent.append(event)

            async def __aiter__(self):
                yield {"type": "response.done"}

        @asynccontextmanager
        async def transport_context():
            yield Transport()

        agent.transport_context_factory = transport_context
        history = (
            PublicHistoryMessage("client_a", "私は休ませたいです。"),
            PublicHistoryMessage("client_b", "私は遅れが心配です。"),
            PublicHistoryMessage("counselor", "それぞれ気がかりがあるんですね。"),
        )
        runtime.prompt_context_store.set_context(
            3, RuntimePromptContextData(public_history=history)
        )
        try:
            instruction = await runtime._add_prompt_director_instruction(
                turn_id=3,
                speaker=speaker,
                input_transcript=history[-1].text,
                current_objective="継続",
                response_target="counselor",
                existing_instruction=None,
            )
            async for _ in agent.stream_audio_response(
                session_id="client-direct-check",
                turn_id=3,
                speaker=speaker,
                input_transcript=history[-1].text,
                additional_instruction=instruction,
                sample_rate=24000,
                sample_width_bits=16,
                channels=1,
                delivery_mode=AudioDeliveryMode.ACCELERATED,
            ):
                pass
            response = next(
                e["response"] for e in sent if e["type"] == "response.create"
            )
            assert CLIENT_SYSTEM_PROMPT in response["instructions"]
            own, peer = ("A", "B") if speaker == "client_a" else ("B", "A")
            assert f"{own}のプリセット" in response["instructions"]
            encoded = json.dumps(response, ensure_ascii=False)
            assert f"{own}だけの設定" in encoded
            assert f"{peer}だけの設定" not in encoded
            assert "require_repeat_verbatim" not in encoded
            for item in history:
                message = next(
                    m for m in response["input"] if item.text in m["content"][0]["text"]
                )
                content = json.loads(message["content"][0]["text"])
                assert content == {"speaker_id": item.speaker_id, "text": item.text}
                assert message["role"] == (
                    "assistant" if item.speaker_id == speaker else "user"
                )
        finally:
            await agent.close()

    asyncio.run(scenario())


def test_build_openai_runtime_keeps_fixed_prompts_in_realtime_mode(
    tmp_path: Path,
) -> None:
    config = load_runtime_config(
        _write_runtime_config(
            tmp_path,
            realtime_api_centered_mode=True,
            participants_block="""
participants:
  counselor:
    role: counselor
    display_name: カウンセラー
    prompt_source: custom counselor prompt
  client:
    role: client
    display_name: クライアント
    prompt_source: custom client prompt
""",
        )
    )

    runtime = build_openai_runtime(
        config,
        api_key="test-api-key",
        stt=object(),
    )

    assert runtime.controller.agents["counselor"].config.instructions == (
        COUNSELOR_SYSTEM_PROMPT
    )
    assert runtime.controller.agents["client"].config.instructions == (
        CLIENT_SYSTEM_PROMPT
    )
    counselor_input = runtime.controller.agents["counselor"].input_formatter(
        speaker="counselor",
        turn_id=1,
        input_transcript="相談があります。",
    )
    client_input = runtime.controller.agents["client"].input_formatter(
        speaker="client",
        turn_id=2,
        input_transcript="どうされましたか。",
    )
    assert "custom counselor prompt" in counselor_input
    assert "custom client prompt" in client_input


def test_build_openai_runtime_uses_profile_aligned_counselor_prompt_fallback(
    tmp_path: Path,
) -> None:
    config = load_runtime_config(
        _write_runtime_config(tmp_path, realtime_api_centered_mode=True)
    )

    runtime = build_openai_runtime(
        config,
        api_key="test-api-key",
        stt=object(),
    )

    assert runtime.controller.agents["counselor"].config.instructions == (
        COUNSELOR_SYSTEM_PROMPT
    )
    assert "カウンセラーの役割を維持" in (
        runtime.controller.agents["counselor"].config.instructions
    )


def test_build_openai_runtime_raises_clear_error_when_api_key_missing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    config = load_runtime_config(_write_runtime_config(tmp_path))

    with pytest.raises(RuntimeFactoryError, match="OPENAI_API_KEY"):
        build_openai_runtime(config, api_key="")


def test_default_runtime_config_is_independent_from_v02_app_config() -> None:
    config = load_runtime_config()

    assert DEFAULT_RUNTIME_CONFIG_PATH == ROOT_DIR / "config" / "runtime_config.yaml"
    assert DEFAULT_RUNTIME_CONFIG_PATH.name != "app_config.yaml"
    assert not hasattr(config, "app")
    assert config.runtime.engine == "asyncio"
    assert config.runtime.backend == "openai"


def test_load_runtime_config_rejects_unsupported_audio_delivery_mode(
    tmp_path: Path,
) -> None:
    config_path = _write_runtime_config(
        tmp_path, audio_delivery_mode="browser_playback"
    )

    with pytest.raises(RuntimeConfigLoadError, match="audio_delivery_mode"):
        load_runtime_config(config_path)


def test_load_runtime_config_rejects_unsupported_runtime_backend(
    tmp_path: Path,
) -> None:
    config_path = _write_runtime_config(tmp_path, backend="classic")

    with pytest.raises(RuntimeConfigLoadError, match="backend"):
        load_runtime_config(config_path)


def test_load_runtime_config_defaults_to_legacy_participants(tmp_path: Path) -> None:
    config = load_runtime_config(_write_runtime_config(tmp_path))

    assert set(config.participants) == {"counselor", "client"}
    assert config.participants["counselor"].role == "counselor"
    assert config.participants["client"].role == "client"
    assert config.participants["client"].display_name == "クライアント"


def test_load_runtime_config_accepts_two_client_participants(tmp_path: Path) -> None:
    config = load_runtime_config(
        _write_runtime_config(
            tmp_path,
            participants_block="""
participants:
  counselor:
    role: counselor
    display_name: カウンセラー
    prompt_path: config/profiles/counselors/counselor_brief_default/profile.md
    voice: shimmer
    realtime_output_speed: 1.0
  client_a:
    role: client
    display_name: クライアントA
    prompt_path: config/profiles/clients/client_a_default/profile.md
    private_profile_path: config/profiles/clients/client_a_default/private.md
    initial_transcript: 今日は、相手に言いたいことがあって来ました。
    voice: cedar
    realtime_output_speed: 0.9
  client_b:
    role: client
    display_name: クライアントB
    prompt_path: config/profiles/clients/client_b_default/profile.md
    private_profile_path: config/profiles/clients/client_b_default/private.md
    initial_transcript: 正直、何を話せばいいかまだ分かりません。
    voice: coral
    realtime_output_speed: 0.85
""",
        )
    )

    assert set(config.participants) == {"counselor", "client_a", "client_b"}
    assert config.participants["client_a"].role == "client"
    assert config.participants["client_a"].private_profile_path == (
        "config/profiles/clients/client_a_default/private.md"
    )
    assert config.participants["client_b"].initial_transcript == (
        "正直、何を話せばいいかまだ分かりません。"
    )
    assert config.participants["client_b"].voice == "coral"
    assert config.participants["client_b"].realtime_output_speed == 0.85


def test_load_runtime_config_accepts_shared_case_prompt_path(tmp_path: Path) -> None:
    shared_case_path = tmp_path / "shared_case.md"
    config = load_runtime_config(
        _write_runtime_config(
            tmp_path,
            shared_case_block=f"""
shared_case:
  prompt_path: {shared_case_path}
""",
        )
    )

    assert config.shared_case.prompt_path == str(shared_case_path)


def test_load_runtime_config_rejects_participant_speaker_id_mismatch(
    tmp_path: Path,
) -> None:
    config_path = _write_runtime_config(
        tmp_path,
        participants_block="""
participants:
  counselor:
    role: counselor
    display_name: カウンセラー
  client_a:
    speaker_id: client_b
    role: client
    display_name: クライアントA
""",
    )

    with pytest.raises(RuntimeConfigLoadError, match="speaker_id"):
        load_runtime_config(config_path)


def test_load_runtime_config_rejects_participants_without_counselor(
    tmp_path: Path,
) -> None:
    config_path = _write_runtime_config(
        tmp_path,
        participants_block="""
participants:
  client_a:
    role: client
    display_name: クライアントA
  client_b:
    role: client
    display_name: クライアントB
""",
    )

    with pytest.raises(RuntimeConfigLoadError, match="counselor"):
        load_runtime_config(config_path)


def test_load_runtime_config_rejects_empty_fixed_speaker_sequence(
    tmp_path: Path,
) -> None:
    config_path = _write_runtime_config(
        tmp_path,
        runtime_extra_block="""
  fixed_speaker_sequence: []
""",
    )

    with pytest.raises(RuntimeConfigLoadError, match="fixed_speaker_sequence"):
        load_runtime_config(config_path)


def _write_runtime_config(
    tmp_path: Path,
    *,
    backend: str = "fake",
    audio_delivery_mode: str = "real_time",
    sample_rate: int = 24000,
    channels: int = 1,
    tts_response_format: str = "pcm",
    tts_chunk_comma_min_chars: int = 20,
    tts_chunk_soft_max_chars: int = 40,
    tts_target_chunk_duration_ms: int = 40,
    tts_sdk_chunk_size: int | None = None,
    llm_max_output_tokens: int | None = None,
    timing_llm_max_output_tokens: int = 128,
    prompting_llm_max_output_tokens: int | None = None,
    stt_partial_prefetch: bool = False,
    stt_partial_prefetch_min_chars: int = 8,
    realtime_api_centered_mode: bool = False,
    runtime_extra_block: str = "",
    participants_block: str = "",
    shared_case_block: str = "",
) -> Path:
    tts_sdk_chunk_size_value = (
        "null" if tts_sdk_chunk_size is None else str(tts_sdk_chunk_size)
    )
    llm_max_output_tokens_value = (
        "null" if llm_max_output_tokens is None else str(llm_max_output_tokens)
    )
    prompting_limit_line = (
        ""
        if prompting_llm_max_output_tokens is None
        else f"  prompting_llm_max_output_tokens: {prompting_llm_max_output_tokens}"
    )
    config_path = tmp_path / "runtime_config.yaml"
    config_path.write_text(
        f"""
runtime:
  engine: asyncio
  backend: {backend}
  control_api: fastapi
  control_host: 127.0.0.1
  control_port: 8765
  monitor_transport: websocket
  audio_delivery_mode: {audio_delivery_mode}
  turn_boundary_policy: explicit_commit
  max_turns: 2
  initial_client_transcript: 初回相談です。
{runtime_extra_block}

audio:
  hot_path_format: pcm
  sample_rate: {sample_rate}
  sample_width_bits: 16
  channels: {channels}
  speaker_gains:
    counselor: 0.8
    client: 0.8
  turn_log_format: wav
  archive_format: flac
  public_export_format: mp3
  create_public_mp3: false

openai:
  llm_model: "${{OPENAI_TEXT_MODEL:-default-text-model}}"
  timing_llm_model: "${{OPENAI_TIMING_MODEL:-default-timing-model}}"
  timing_llm_reasoning_effort: none
  summary_llm_model: "${{OPENAI_SUMMARY_MODEL:-default-summary-model}}"
  summary_llm_reasoning_effort: none
  prompting_llm_model: "${{OPENAI_PROMPTING_MODEL:-default-prompting-model}}"
  prompting_llm_reasoning_effort: none
  tts_model: "${{OPENAI_TTS_MODEL:-default-tts-model}}"
  tts_voice: coral
  counselor_tts_voice: coral
  client_tts_voice: cedar
  counselor_realtime_output_speed: 0.85
  client_realtime_output_speed: 0.95
  tts_instructions: ""
  tts_response_format: {tts_response_format}
  stt_model: "${{OPENAI_STT_MODEL:-default-stt-model}}"
  realtime_transcription_format:
    type: audio/pcm
    rate: {sample_rate}

paths:
  sessions_dir: {tmp_path / "sessions"}
  runtime_sessions_dir: {tmp_path / "runtime_sessions"}
  replay_sessions_dir: {tmp_path / "replay_sessions"}

{shared_case_block}

latency:
  llm_max_output_tokens: {llm_max_output_tokens_value}
  timing_llm_max_output_tokens: {timing_llm_max_output_tokens}
{prompting_limit_line}
  tts_chunk_comma_min_chars: {tts_chunk_comma_min_chars}
  tts_chunk_soft_max_chars: {tts_chunk_soft_max_chars}
  tts_target_chunk_duration_ms: {tts_target_chunk_duration_ms}
  tts_sdk_chunk_size: {tts_sdk_chunk_size_value}
  stt_partial_prefetch: {str(stt_partial_prefetch).lower()}
  stt_partial_prefetch_min_chars: {stt_partial_prefetch_min_chars}
  realtime_api_centered_mode: {str(realtime_api_centered_mode).lower()}
{participants_block}
""".lstrip(),
        encoding="utf-8",
    )
    return config_path
