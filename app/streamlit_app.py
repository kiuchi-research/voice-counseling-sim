from __future__ import annotations

import base64
import hashlib
import html
import json
import sys
import time
import wave
from pathlib import Path
from typing import Any

import streamlit as st
import streamlit.components.v1 as components
import yaml


ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from counseling_voice_demo import __version__
from counseling_voice_demo.client_presets import (
    ClientPreset,
    ClientPresetLoadError,
    list_client_presets,
    load_client_preset,
)
from counseling_voice_demo.counselor_presets import (
    CounselorPreset,
    CounselorPresetLoadError,
    list_counselor_presets,
)
from counseling_voice_demo.clinical_ng_checker import ClinicalNgChecker
from counseling_voice_demo.config_loader import ConfigLoadError, load_app_config
from counseling_voice_demo.content_loader import (
    ContentLoadError,
    Profile,
    Theme,
    VoicePreset,
    list_profiles,
    list_themes,
    list_voice_presets,
)
from counseling_voice_demo.conversation_engine import (
    ConversationEngine,
    ConversationEngineConfig,
    ConversationParticipants,
    invalidate_unplayed_speaker_sequence_violations,
)
from counseling_voice_demo.evaluation_runner import (
    EvaluationRunConfig,
    run_session_evaluations,
)
from counseling_voice_demo.audio_player_component import (
    BrowserAudioEvent,
    build_browser_audio_queue,
    event_was_processed,
    parse_browser_audio_event,
    remember_processed_event_id,
)
from counseling_voice_demo.generation_queue import is_unplayed_turn
from counseling_voice_demo.log_writer import (
    SessionLogPaths,
    append_turn_transcripts,
    create_session_log_dirs,
    update_transcript_markdown,
)
from counseling_voice_demo.openai_moderation_client import OpenAIModerationClient
from counseling_voice_demo.models import (
    ConversationPhase,
    SessionState,
    SessionStatus,
    SpeakerRole,
    Turn,
    TurnStatus,
    WarningLevel,
)
from counseling_voice_demo.openai_evaluation_client import OpenAIEvaluationClient
from counseling_voice_demo.openai_text_client import OpenAITextClient
from counseling_voice_demo.openai_tts_client import OpenAITtsClient, TtsRequest
from counseling_voice_demo.playback_controller import (
    PlaybackEvent,
    next_playback_decision,
    record_playback_terminal_event,
    release_playback_hold,
    skip_unplayed_hold,
)
from counseling_voice_demo.safety_checks import SafetyCheckConfig, SafetyCheckRunner
from counseling_voice_demo.tts_queue import mark_tts_ready, next_tts_turn
from counseling_voice_demo.turn_editing import edit_unplayed_turn, regenerate_from_turn
from counseling_voice_demo.turn_regeneration import regenerate_single_turn
from counseling_voice_demo.warning_policy import warning_rank
from counseling_voice_demo.runtime.config import RuntimeConfigLoadError, load_runtime_config
from counseling_voice_demo.runtime.provider_registry import effective_ai_settings
from counseling_voice_demo.runtime.models import (
    MAX_REALTIME_OUTPUT_SPEED,
    MIN_REALTIME_OUTPUT_SPEED,
)
from counseling_voice_demo.runtime.observer_client import (
    RuntimeControlClient,
    RuntimeObserverClientError,
    build_runtime_control_endpoint,
)
from counseling_voice_demo.runtime.session_artifacts import (
    PublicTranscriptTurn,
    build_compact_session_audio,
    format_public_script,
    list_runtime_sessions,
    load_public_transcript_turns,
    load_session_identity,
    session_has_human_audio_turns,
    session_realtime_audio_path,
)
from counseling_voice_demo.runtime.system_prompts import (
    CLIENT_SYSTEM_PROMPT,
    COUNSELOR_SYSTEM_PROMPT,
)


MODE_LABELS = {
    "ai_counselor_ai_client": "AIカウンセラー x AIクライアント",
    "human_counselor_ai_client": "人間カウンセラー x AIクライアント",
    "ai_counselor_human_client": "AIカウンセラー x 人間クライアント",
}

TRANSCRIPT_TABLE_COLUMNS = [
    ("turn_id", "id", "6%"),
    ("話者", "speaker wrap", "12%"),
    ("役割", "role", "10%"),
    ("発話", "utterance wrap", "60%"),
    ("再生時刻", "time", "12%"),
]
PUBLIC_TRANSCRIPT_TABLE_COLUMNS = [
    ("開始時刻", "time", "11%"),
    ("話者", "speaker wrap", "13%"),
    ("発話", "utterance wrap", "76%"),
]
RUNTIME_GENERATION_MODE_OPTIONS = {
    "realtime_api": "Realtime API centered",
    "responses_tts": "Responses + REST TTS",
}
RUNTIME_REALTIME_MODEL_OPTIONS = {
    "gpt-realtime-2.1": "realtime-2.1 (gpt-realtime-2.1)",
    "gpt-realtime-2": "realtime-2 (gpt-realtime-2)",
    "gpt-realtime-mini": "mini (gpt-realtime-mini)",
}
RUNTIME_TIMING_LLM_MODEL_OPTIONS = {
    "gpt-5.6-sol": "gpt-5.6-sol",
    "gpt-5.6-terra": "gpt-5.6-terra",
    "gpt-5.6-luna": "gpt-5.6-luna",
    "gpt-5.4-mini": "gpt-5.4-mini",
}
RUNTIME_TIMING_REASONING_EFFORT_OPTIONS = {
    "none": "none",
    "low": "low",
    "medium": "medium",
    "high": "high",
    "xhigh": "xhigh",
}
RUNTIME_PROMPTING_LLM_MODEL_OPTIONS = RUNTIME_TIMING_LLM_MODEL_OPTIONS
RUNTIME_PROMPTING_REASONING_EFFORT_OPTIONS = RUNTIME_TIMING_REASONING_EFFORT_OPTIONS
RUNTIME_SUMMARY_LLM_MODEL_OPTIONS = RUNTIME_TIMING_LLM_MODEL_OPTIONS
RUNTIME_SUMMARY_REASONING_EFFORT_OPTIONS = RUNTIME_TIMING_REASONING_EFFORT_OPTIONS
RUNTIME_SPEAKER_SELECTION_POLICY_OPTIONS = {
    "fixed_round_robin": "固定順序",
    "turn_boundary_timing": "動的ターンテイク（発話後のみ）",
    "distributed_timing": "分散タイミング（実験: 重なり・譲り合い）",
}
RUNTIME_REALTIME_VOICE_OPTIONS = (
    "alloy",
    "marin",
    "cedar",
    "coral",
    "ash",
    "ballad",
    "echo",
    "shimmer",
    "verse",
)
DEFAULT_RUNTIME_REALTIME_MODEL = "gpt-realtime-2.1"
DEFAULT_RUNTIME_TIMING_LLM_MODEL = "gpt-5.4-mini"
DEFAULT_RUNTIME_TIMING_REASONING_EFFORT = "none"
DEFAULT_RUNTIME_PROMPTING_LLM_MODEL = "gpt-5.4-mini"
DEFAULT_RUNTIME_PROMPTING_REASONING_EFFORT = "low"
DEFAULT_RUNTIME_SUMMARY_LLM_MODEL = "gpt-5.4-mini"
DEFAULT_RUNTIME_SUMMARY_REASONING_EFFORT = "none"
RUNTIME_TEXT_ROUTE_CONTROLS = {
    "prompt_director": ("prompting", "プロンプティング"),
    "turn_timing": ("timing", "ターンテイク"),
    "session_summary": ("summary", "セッション要約"),
}
DEFAULT_RUNTIME_COUNSELOR_AUDIO_GAIN = 0.6
DEFAULT_RUNTIME_CLIENT_AUDIO_GAIN = 0.6
DEFAULT_RUNTIME_COUNSELOR_OUTPUT_SPEED = 1.05
DEFAULT_RUNTIME_CLIENT_A_OUTPUT_SPEED = 0.9
DEFAULT_RUNTIME_CLIENT_B_OUTPUT_SPEED = 1.0
DEFAULT_RUNTIME_TWO_CLIENT_SPEAKER_SEQUENCE = (
    "counselor",
    "client_a",
    "client_b",
    "counselor",
    "client_b",
    "client_a",
)
DEFAULT_RUNTIME_CLIENT_OUTPUT_SPEED = DEFAULT_RUNTIME_CLIENT_A_OUTPUT_SPEED
REALTIME_OUTPUT_SPEED_DEFAULTS_VERSION = 2
DEFAULT_RUNTIME_CLIENT_A_TTS_VOICE = "marin"
DEFAULT_RUNTIME_CLIENT_B_TTS_VOICE = "cedar"
LEGACY_RUNTIME_ONE_CLIENT_FORCE_STOP_AFTER_CLOSING_TURNS = 2
LEGACY_RUNTIME_TWO_CLIENT_FORCE_STOP_AFTER_CLOSING_TURNS = 3
DEFAULT_RUNTIME_ONE_CLIENT_FORCE_STOP_AFTER_CLOSING_TURNS = 3
DEFAULT_RUNTIME_TWO_CLIENT_FORCE_STOP_AFTER_CLOSING_TURNS = 3
DEFAULT_RUNTIME_COUNSELOR_DISPLAY_NAME = "カウンセラー"
LEGACY_RUNTIME_CLIENT_DISPLAY_NAME = "クライアント"
DEFAULT_RUNTIME_CLIENT_A_DISPLAY_NAME = "妻"
DEFAULT_RUNTIME_CLIENT_B_DISPLAY_NAME = "夫"
DEFAULT_RUNTIME_CLIENT_DISPLAY_NAME = DEFAULT_RUNTIME_CLIENT_A_DISPLAY_NAME
ONE_CLIENT_WIDGET_DEFAULTS_VERSION = 3
TWO_CLIENT_WIDGET_DEFAULTS_VERSION = 4
PROMPT_FILE_UPLOAD_TYPES = ("md", "txt")
DEFAULT_CLIENT_PRESET_PATH = (
    ROOT_DIR
    / "config"
    / "client_presets"
    / "junior_high_school_refusal_couple_v1.yaml"
)

UNPLAYED_TABLE_COLUMNS = [
    ("turn_id", "id", "5%"),
    ("revision_id", "id", "6%"),
    ("話者", "speaker wrap", "10%"),
    ("役割", "role", "7%"),
    ("発話", "utterance wrap", "38%"),
    ("音声化", "status", "8%"),
    ("再生", "status", "8%"),
    ("編集済み", "edited", "6%"),
    ("警告", "warning", "5%"),
    ("hold理由", "hold", "7%"),
]

AUTO_PLAYBACK_FALLBACK_DURATION_SECONDS = 1.0
AUTO_PLAYBACK_END_GRACE_SECONDS = 0.35
AUTO_PLAYBACK_DEFAULT_GAP_SECONDS = 1.2
RUNTIME_AUDIO_MONITOR_PLAYBACK_GAP_MAX_SECONDS = 1.6
AUTO_PROGRESS_FRAGMENT_INTERVAL_SECONDS = 0.5
RUNTIME_TRANSCRIPT_REFRESH_INTERVAL_SECONDS = 0.1
RUNTIME_TRANSCRIPT_GATED_PHASES = {"running", "paused", "completed"}
DEFAULT_RUNTIME_GENERATION_LEAD_LIMIT = 5
# Wait for the selected number of complete turns before initial playback.
RUNTIME_AUDIO_MONITOR_WARMUP_MAX_WAIT_SECONDS = 0.0
LEGACY_RUNTIME_CLOSING_START_DEFAULT_SECONDS = 900
RUNTIME_PARTICIPANT_MODE_LABELS = {
    "one_client": "1クライアント",
    "two_clients": "2クライアント",
}
DEFAULT_RUNTIME_PARTICIPANT_MODE = "two_clients"
LEGACY_RUNTIME_COUNSELOR_SYSTEM_PROMPT = (
    "あなたは研究デモ用の日本語カウンセラー役AIです。"
    "診断や治療方針の断定を避け、1文程度で短く落ち着いて応答してください。"
    "相談者本人の発話として返答してはいけません。"
)
DEFAULT_RUNTIME_COUNSELOR_PROMPT = (
    "あなたは日本語カウンセラー役AIとして、2人のクライアントとのファミリーセラピーを行います。\n"
    "クライアントの語りと会話の流れを優先し、以下のフェーズと応答のルールに沿って進めてください。\n"
    "クライアントの発話として返答してはいけません。\n"
    "\n"
    "## 用語\n"
    "- このプロンプトの「1ターン」は、カウンセラーの1回の発話です。クライアントの発話は、この回数に含めません。\n"
    "- 「聞き返し」とは、相手の発言を言い換えて伝え返すことです。聞き直す質問のことではありません。\n"
    "- 聞き返しでは、ただのオウム返しではなく、クライアントの気持ち、思い、考え、状況、見通しを推察し、解釈を交えて伝え返します。\n"
    "\n"
    "## フェーズ\n"
    "1. あいさつ・ねぎらい（1ターン）\n"
    "2. 理想の未来像の確認（このフェーズで2ターン以上）\n"
    "3. 例外（今できていること、問題が少しはましな場面）の探索（このフェーズで2ターン以上）\n"
    "4. スモールステップの目標設定（このフェーズで2ターン以上）\n"
    "5. 振り返り（このフェーズで2ターン以上。今日の面接での学びや気づきを振り返り、クライアント自身が自分たちをコンプリメントする）\n"
    "6. クロージング（このフェーズで2ターン以上）\n"
    "\n"
    "## フェーズを進める条件\n"
    "- 通常は1から6の順に進みます。各フェーズの最低回数を満たし、その内容をクライアントと確認してから、次へ進んでください。\n"
    "- 各フェーズの回数は独立して数えます。別のフェーズでの発話を、現在のフェーズの回数に含めないでください。\n"
    "- 最低回数に達しても、内容の理解が不十分なら同じフェーズで聞き返しや具体的な確認を続けてください。\n"
    "- 面接を終結する指示が届いた場合は、その指示に従ってクロージングを行ってください。まだ扱っていないフェーズを、完了したことにしないでください。\n"
    "\n"
    "## 冒頭の応答\n"
    "- 最初のカウンセラー発話では、短くあいさつ・ねぎらいをしてください。\n"
    "- 相談内容がまだ語られていない場合は、何を話したいかを尋ね、返答を待ってください。\n"
    "- 相談内容が既に語られている場合は、その内容を聞き返してください。改めて何を話したいかを尋ねる必要はありません。\n"
    "- 相談内容を聞いて受け止めてから、理想の未来像の確認に進んでください。\n"
    "\n"
    "## 応答のルール\n"
    "- 安易にアドバイスはしない\n"
    "- 聞き返しを中心にします。聞き返しだけで応答を終え、相手が語り続けるのを待つことを基本にしてください。\n"
    "- 質問は、今扱っている話題で確かめたい点があるときに加えてください。聞き返しの後に毎回質問を付ける必要はありません。\n"
    "- 聞き返しは、ネガティブな内容よりもどうなりたいか、どう変わりたいか、何が起こったらいいか、何があるといいかなど、建設的な内容にできるだけ着目する\n"
    "- 聞き返しでは、「受け止めています」とは言わない\n"
    "- エピソードについては、表面的に理解せずに、聞き返しや質問により、何ターンかかけて具体的に掘り下げる\n"
    "- 適宜コンプリメント（承認、是認、賞賛、励まし、ねぎらい）を行う\n"
    "- 相手の語りに合う軽い比喩やユーモアを交えた聞き返しやねぎらい、リフレーミングを行う\n"
    "\n"
    "## 発話の形式\n"
    "- 普段は1文程度で短く落ち着いて応答してください。\n"
    "- 聞き返しは「～ということ」「～だと」など、言いきりの形にしてください。\n"
    "- 質問の末尾には「？」を付ける\n"
    "- 聞き返しの末尾には「？」を付けない\n"
    "- 聞き返しと質問を同じ発話内で行う場合は、1文程度という目安よりも2文に分ける指示を優先し、聞き返しと質問をそれぞれ独立した一文にし、計2文で応答する。「～が、～？」のように一文へつなげないでください。\n"
    "\n"
    "## キー質問\n"
    "以下は、対応するフェーズで会話の流れに合う場合に使う質問の候補です。優先して出すべき質問や、順番に消化する必須項目ではありません。\n"
    "クライアントの直前の発言と今扱っている話題を優先し、キー質問を使うために話題を切り替えたり、理解が不十分なまま次のフェーズへ進めたりしないでください。\n"
    "タイミングが合わない場合はキー質問を使わず、聞き返しや具体的な確認を続けてください。\n"
    "\n"
    "- 面接を終えた後、どのようになっていたら今日来たかいがあったと言えそうですか？\n"
    "- 例外探し：今できていることはなんですか？あるいは、問題が少しはましな時はどんな時ですか？\n"
    "- スケーリングクエスチョン：理想が大方達成された状態を10点として、今は何点ですか？\n"
    "- 状況がいい方向に進んでいることを示す最初の兆しは何ですか？\n"
    "- これまでに生きる助けになった音楽や物語はありますか？\n"
    "- ちょっと変わった方法でこの問題に対処するとしたら、どんな方法が考えられますか？\n"
    "\n"
    "## 点数の回答を受けた後\n"
    "- スケーリングクエスチョンの回答が得られたら、出された点数よりも少ない点数でなくて、その点数にしたのは、何があるからですか？と質問する。\n"
    "- この追加質問は点数の回答を受けた後に行います。点数を尋ねるキー質問を使うかどうかは、会話の流れに合わせて別に判断してください。"
)
LEGACY_RUNTIME_CLIENT_PROMPT = (
    "あなたは研究デモ用の日本語クライアント役AIです。"
    "カウンセラーではなく相談者本人です。"
    "助言、質問、要約、相手への治療的な返しは避け、"
    "一人称で不安や迷いを1文程度で短く話してください。"
    "医療的な事実の断定は避けてください。"
    "発話の先頭に話者名や役割名を付けず、発話本文だけを返してください。"
)
LEGACY_RUNTIME_ONE_CLIENT_PROMPT_INTRO = (
    "あなたは、家族カウンセリングのデモに参加するクライアント夫婦の夫を演じます。"
)
TWO_CLIENT_A_PROMPT_INTRO = (
    "あなたは、家族カウンセリングのデモに参加するクライアント夫婦の妻を演じます。"
)
TWO_CLIENT_B_PROMPT_INTRO = LEGACY_RUNTIME_ONE_CLIENT_PROMPT_INTRO
DEFAULT_RUNTIME_CLIENT_PROMPT_INTRO = TWO_CLIENT_A_PROMPT_INTRO
DEFAULT_RUNTIME_CLIENT_PROMPT = (
    DEFAULT_RUNTIME_CLIENT_PROMPT_INTRO
    + "カウンセラーではなく相談者本人です。"
    "助言、質問、要約、相手への治療的な返しは避け、"
    "一人称で不安や迷いを1文程度で短く話してください。"
    "医療的な事実の断定は避けてください。"
    "発話の先頭に話者名や役割名を付けず、発話本文だけを返してください。"
)
DEFAULT_RUNTIME_INITIAL_CLIENT_TRANSCRIPT = "娘のことで相談があります"
LEGACY_RUNTIME_INITIAL_CLIENT_TRANSCRIPTS = {
    "今日は家族のことで相談したいです。",
}
TWO_CLIENT_COMMON_PROMPT_INTRO = (
    "あなたは、家族カウンセリングのデモに参加するクライアント夫婦の一人を演じます。"
)
AUDIO_PLAYER_COMPONENT_DIR = ROOT_DIR / "app" / "components" / "audio_player"
browser_audio_player = components.declare_component(
    "browser_audio_player",
    path=str(AUDIO_PLAYER_COMPONENT_DIR),
)
RUNTIME_AUDIO_MONITOR_COMPONENT_DIR = (
    ROOT_DIR / "app" / "components" / "runtime_audio_monitor"
)
runtime_audio_monitor = components.declare_component(
    "runtime_audio_monitor",
    path=str(RUNTIME_AUDIO_MONITOR_COMPONENT_DIR),
)
RUNTIME_AUDIO_MONITOR_HTML = (
    RUNTIME_AUDIO_MONITOR_COMPONENT_DIR / "index.html"
).read_text(encoding="utf-8")


def build_mode_options(enabled_modes: dict[str, bool]) -> list[dict[str, Any]]:
    return [
        {
            "mode": mode,
            "label": MODE_LABELS.get(mode, mode),
            "enabled": bool(enabled_modes.get(mode, False)),
        }
        for mode in MODE_LABELS
    ]


def _join_client_preset_prompt_parts(*parts: str) -> str:
    return "\n\n".join(part.strip() for part in parts if part.strip())


def counselor_preset_session_updates(
    preset: CounselorPreset,
) -> dict[str, Any]:
    preset_source = f"preset:{preset.preset_id}"
    return {
        "counselor_display_name": preset.counselor.display_name,
        "counselor_public_profile_text": preset.counselor.public_profile,
        "counselor_public_profile_source_profile_id": preset_source,
        "counselor_prompt_text": preset.counselor.prompt,
        "counselor_prompt_source_profile_id": preset_source,
        "runtime_counselor_tts_voice": preset.counselor.audio.voice,
        "runtime_counselor_output_speed": preset.counselor.audio.output_speed,
        "runtime_counselor_audio_gain": preset.counselor.audio.gain,
    }


def should_apply_counselor_preset(
    state: dict[str, Any],
    preset_id: str,
    *,
    force: bool,
) -> bool:
    return bool(
        force or state.get("applied_counselor_preset_id") != preset_id
    )


def counselor_preset_is_active() -> bool:
    selected_id = st.session_state.get("selected_counselor_preset_id")
    return bool(
        selected_id
        and st.session_state.get("applied_counselor_preset_id") == selected_id
    )


def active_counselor_content_source_id(fallback_profile_id: str) -> str:
    preset_id = st.session_state.get("selected_counselor_preset_id")
    if counselor_preset_is_active() and preset_id:
        return f"preset:{preset_id}"
    return fallback_profile_id


def apply_counselor_preset_to_session_state(
    preset: CounselorPreset,
) -> None:
    updates = counselor_preset_session_updates(preset)
    for state_key, value in updates.items():
        st.session_state[state_key] = value
        st.session_state.pop(runtime_widget_key(state_key), None)
        st.session_state.pop(f"{state_key}_persist_user_modified", None)
        st.session_state.pop(f"{state_key}_default_applied", None)
    for state_key in (
        "counselor_public_profile_file_signature",
        "counselor_prompt_file_signature",
    ):
        st.session_state[state_key] = None
    for widget_key in (
        "counselor_public_profile_file_upload",
        "counselor_prompt_file_upload",
    ):
        st.session_state.pop(widget_key, None)
    st.session_state["applied_counselor_preset_id"] = preset.preset_id


def client_preset_session_updates(
    preset: ClientPreset,
    participant_mode: str,
) -> dict[str, Any]:
    participant_ids = preset.participant_ids_for_mode(participant_mode)
    preset_source = f"preset:{preset.preset_id}"
    client_a_id = preset.modes.one_client[0]
    client_a = preset.participants[client_a_id]
    updates: dict[str, Any] = {
        "client_public_profile_text": preset.shared.public_profile,
        "client_public_profile_source_profile_id": preset_source,
        "client_private_profile_text": client_a.private_profile,
        "client_private_profile_source_profile_id": preset_source,
        "client_common_profile_applied_source_profile_id": preset_source,
        "client_common_profile_applied_public_profile_text": (
            preset.shared.public_profile
        ),
        "client_common_profile_applied_prompt_text": preset.shared.prompt,
    }
    for slot_name, participant_id in zip(
        ("client_a", "client_b"),
        preset.modes.two_clients,
        strict=True,
    ):
        participant = preset.participants[participant_id]
        updates.update(
            {
                f"{slot_name}_display_name": participant.display_name,
                f"{slot_name}_public_profile_text": participant.private_profile,
                f"{slot_name}_public_profile_source_profile_id": preset_source,
                f"{slot_name}_prompt_text": _join_client_preset_prompt_parts(
                    preset.shared.prompt,
                    participant.prompt,
                ),
                f"{slot_name}_prompt_source_profile_id": preset_source,
                f"runtime_initial_{slot_name}_transcript": (
                    participant.initial_transcript
                ),
                f"runtime_{slot_name}_tts_voice": participant.audio.voice,
                f"runtime_{slot_name}_output_speed": participant.audio.output_speed,
                f"runtime_{slot_name}_audio_gain": participant.audio.gain,
            }
        )
    if participant_mode == "one_client":
        active_participant = preset.participants[participant_ids[0]]
        updates.update(
            {
                "client_display_name": active_participant.display_name,
                "client_private_profile_text": active_participant.private_profile,
                "client_prompt_text": active_participant.prompt,
                "client_prompt_source_profile_id": preset_source,
                "runtime_initial_client_transcript": (
                    active_participant.initial_transcript
                ),
                "runtime_client_tts_voice": active_participant.audio.voice,
                "runtime_client_output_speed": active_participant.audio.output_speed,
                "runtime_client_audio_gain": active_participant.audio.gain,
            }
        )
    else:
        updates.update(
            {
                "client_prompt_text": preset.shared.prompt,
                "client_prompt_source_profile_id": preset_source,
            }
        )
    return updates


def should_apply_client_preset(
    state: dict[str, Any],
    preset_id: str,
    participant_mode: str,
    *,
    force: bool,
) -> bool:
    return bool(
        force
        or state.get("applied_client_preset_id") != preset_id
        or state.get("applied_client_preset_participant_mode") != participant_mode
    )


def client_preset_is_active(participant_mode: str | None = None) -> bool:
    active_mode = str(
        participant_mode
        or st.session_state.get("runtime_participant_mode")
        or DEFAULT_RUNTIME_PARTICIPANT_MODE
    )
    selected_id = st.session_state.get("selected_client_preset_id")
    return bool(
        selected_id
        and st.session_state.get("applied_client_preset_id") == selected_id
        and st.session_state.get("applied_client_preset_participant_mode")
        == active_mode
    )


def active_client_content_source_id(fallback_profile_id: str) -> str:
    preset_id = st.session_state.get("selected_client_preset_id")
    if client_preset_is_active() and preset_id:
        return f"preset:{preset_id}"
    return fallback_profile_id


def apply_client_preset_to_session_state(
    preset: ClientPreset,
    participant_mode: str,
) -> None:
    updates = client_preset_session_updates(preset, participant_mode)
    for state_key, value in updates.items():
        st.session_state[state_key] = value
        st.session_state.pop(runtime_widget_key(state_key), None)
        st.session_state.pop(f"{state_key}_persist_user_modified", None)
        st.session_state.pop(f"{state_key}_default_applied", None)
    for state_key in (
        "client_public_profile_file_signature",
        "client_private_profile_file_signature",
        "client_prompt_file_signature",
        "client_a_public_profile_file_signature",
        "client_b_public_profile_file_signature",
        "client_a_prompt_file_signature",
        "client_b_prompt_file_signature",
    ):
        st.session_state[state_key] = None
    for widget_key in (
        "client_public_profile_file_upload",
        "client_private_profile_file_upload",
        "client_prompt_file_upload",
        "client_a_public_profile_file_upload",
        "client_b_public_profile_file_upload",
        "client_a_prompt_file_upload",
        "client_b_prompt_file_upload",
    ):
        st.session_state.pop(widget_key, None)
    st.session_state["applied_client_preset_id"] = preset.preset_id
    st.session_state["applied_client_preset_participant_mode"] = participant_mode


def initial_ui_state() -> dict[str, Any]:
    return {
        "session_status": SessionStatus.IDLE.value,
        "conversation_phase": ConversationPhase.OPENING.value,
        "cumulative_audio_seconds": 0.0,
        "wall_clock_seconds": 0.0,
        "selected_mode": "ai_counselor_ai_client",
        "ui_mode": "live",
        "selected_counselor_profile_id": None,
        "selected_client_profile_id": None,
        "selected_counselor_preset_id": None,
        "applied_counselor_preset_id": None,
        "selected_client_preset_id": "junior_high_school_refusal_couple_v3",
        "applied_client_preset_id": None,
        "applied_client_preset_participant_mode": None,
        "selected_theme_id": None,
        "counselor_display_name": DEFAULT_RUNTIME_COUNSELOR_DISPLAY_NAME,
        "client_display_name": DEFAULT_RUNTIME_CLIENT_DISPLAY_NAME,
        "counselor_public_profile_text": "",
        "client_public_profile_text": "",
        "client_private_profile_text": "",
        "counselor_prompt_text": "",
        "client_prompt_text": "",
        "runtime_participant_mode": DEFAULT_RUNTIME_PARTICIPANT_MODE,
        "runtime_setup_settings_tab": DEFAULT_RUNTIME_PARTICIPANT_MODE,
        "client_a_display_name": DEFAULT_RUNTIME_CLIENT_A_DISPLAY_NAME,
        "client_b_display_name": DEFAULT_RUNTIME_CLIENT_B_DISPLAY_NAME,
        "client_a_public_profile_text": "",
        "client_b_public_profile_text": "",
        "client_a_prompt_text": "",
        "client_b_prompt_text": "",
        "counselor_public_profile_source_profile_id": None,
        "client_public_profile_source_profile_id": None,
        "client_private_profile_source_profile_id": None,
        "counselor_prompt_source_profile_id": None,
        "client_prompt_source_profile_id": None,
        "client_common_profile_applied_source_profile_id": None,
        "client_common_profile_applied_public_profile_text": None,
        "client_common_profile_applied_prompt_text": None,
        "runtime_one_client_widget_defaults_version": 0,
        "runtime_two_client_widget_defaults_version": 0,
        "client_a_public_profile_source_profile_id": None,
        "client_b_public_profile_source_profile_id": None,
        "client_a_prompt_source_profile_id": None,
        "client_b_prompt_source_profile_id": None,
        "counselor_public_profile_file_signature": None,
        "client_public_profile_file_signature": None,
        "client_private_profile_file_signature": None,
        "counselor_prompt_file_signature": None,
        "client_prompt_file_signature": None,
        "client_a_public_profile_file_signature": None,
        "client_b_public_profile_file_signature": None,
        "client_a_prompt_file_signature": None,
        "client_b_prompt_file_signature": None,
        "runtime_initial_client_transcript": "",
        "runtime_initial_client_a_transcript": "",
        "runtime_initial_client_b_transcript": "",
        "runtime_realtime_model": DEFAULT_RUNTIME_REALTIME_MODEL,
        "runtime_prompting_llm_model": DEFAULT_RUNTIME_PROMPTING_LLM_MODEL,
        "runtime_prompting_llm_reasoning_effort": (
            DEFAULT_RUNTIME_PROMPTING_REASONING_EFFORT
        ),
        "runtime_timing_llm_model": DEFAULT_RUNTIME_TIMING_LLM_MODEL,
        "runtime_timing_llm_reasoning_effort": (
            DEFAULT_RUNTIME_TIMING_REASONING_EFFORT
        ),
        "runtime_summary_llm_model": DEFAULT_RUNTIME_SUMMARY_LLM_MODEL,
        "runtime_summary_llm_reasoning_effort": (
            DEFAULT_RUNTIME_SUMMARY_REASONING_EFFORT
        ),
        "runtime_speaker_selection_policy": "turn_boundary_timing",
        "runtime_counselor_tts_voice": "",
        "runtime_client_tts_voice": DEFAULT_RUNTIME_CLIENT_A_TTS_VOICE,
        "runtime_client_a_tts_voice": DEFAULT_RUNTIME_CLIENT_A_TTS_VOICE,
        "runtime_client_b_tts_voice": DEFAULT_RUNTIME_CLIENT_B_TTS_VOICE,
        "runtime_counselor_output_speed": None,
        "runtime_client_output_speed": DEFAULT_RUNTIME_CLIENT_OUTPUT_SPEED,
        "runtime_client_a_output_speed": DEFAULT_RUNTIME_CLIENT_A_OUTPUT_SPEED,
        "runtime_client_b_output_speed": DEFAULT_RUNTIME_CLIENT_B_OUTPUT_SPEED,
        "runtime_output_speed_defaults_version": 0,
        "runtime_closing_start_seconds": None,
        "runtime_force_stop_after_closing_turns": None,
        "runtime_one_client_force_stop_after_closing_turns": (
            DEFAULT_RUNTIME_ONE_CLIENT_FORCE_STOP_AFTER_CLOSING_TURNS
        ),
        "runtime_two_client_force_stop_after_closing_turns": (
            DEFAULT_RUNTIME_TWO_CLIENT_FORCE_STOP_AFTER_CLOSING_TURNS
        ),
        "selected_counselor_voice_preset_id": None,
        "selected_client_voice_preset_id": None,
        "ahead_generation_turns": None,
        "show_evaluation": False,
        "show_prompts": False,
        "evaluation_public": "",
        "evaluation_internal": "",
        "evaluation_status": "not_started",
        "evaluation_error": "",
        "conversation_log": [],
        "selected_unplayed_turn_id": None,
        "edit_turn_text": "",
        "edit_turn_text_source_turn_id": None,
        "unplayed_turns": [],
        "warnings": [],
        "selected_hold_action": "このまま再生",
        "auto_progress_enabled": True,
        "auto_progress_busy": False,
        "auto_playback_gap_seconds": AUTO_PLAYBACK_DEFAULT_GAP_SECONDS,
        "auto_playback_turn_id": None,
        "auto_playback_started_at": None,
        "auto_playback_duration_seconds": None,
        "auto_playback_visible_turn_id": None,
        "auto_playback_visible_started_at": None,
        "audio_player_queue_version": "",
        "audio_player_processed_event_ids": [],
        "audio_player_last_event": None,
        "audio_player_continuous_mode": False,
        "runtime_control_host": "",
        "runtime_control_port": 0,
        "runtime_control_status": {},
        "runtime_control_events": [],
        "runtime_response_instructions": [],
        "runtime_control_error": "",
        "runtime_monitor_session_id": "",
        "runtime_monitor_connected": False,
        "runtime_generation_lead_limit": DEFAULT_RUNTIME_GENERATION_LEAD_LIMIT,
        "runtime_generation_lead_turns": 0,
        "runtime_generation_throttle_active": False,
        "runtime_generation_mode": "",
        "runtime_initial_warmup_session_id": "",
        "runtime_initial_warmup_started_at_monotonic": None,
        "runtime_initial_warmup_released": False,
        "runtime_initial_warmup_active": False,
        "runtime_counselor_audio_gain": None,
        "runtime_client_audio_gain": DEFAULT_RUNTIME_CLIENT_AUDIO_GAIN,
        "runtime_client_a_audio_gain": DEFAULT_RUNTIME_CLIENT_AUDIO_GAIN,
        "runtime_client_b_audio_gain": DEFAULT_RUNTIME_CLIENT_AUDIO_GAIN,
        "runtime_wall_clock_session_id": "",
        "runtime_wall_clock_started_at_monotonic": None,
        "runtime_wall_clock_base_seconds": 0.0,
        "runtime_transcript_visibility_session_id": "",
        "runtime_transcript_visible_turn_count": 0,
        "runtime_transcript_next_reveal_wall_seconds": None,
        "runtime_transcript_visibility_clock_seconds": 0.0,
        "runtime_transcript_visibility_clock_started_at_monotonic": None,
        "runtime_transcript_visibility_clock_base_seconds": 0.0,
        "runtime_completed_turn_count_session_id": "",
        "runtime_completed_turn_count_for_display": 0,
        "selected_replay_session_dir": "",
        "selected_replay_start_time_key": "",
    }


def ensure_ui_state() -> None:
    for key, value in initial_ui_state().items():
        st.session_state.setdefault(key, value)
    if (
        not isinstance(st.session_state.get("session_model"), SessionState)
        and is_legacy_seed_unplayed_turns(st.session_state.get("unplayed_turns"))
    ):
        st.session_state["unplayed_turns"] = []
        st.session_state["selected_unplayed_turn_id"] = None
        st.session_state["edit_turn_text"] = ""
        st.session_state["edit_turn_text_source_turn_id"] = None
        st.session_state["warnings"] = []


def is_legacy_seed_unplayed_turns(value: Any) -> bool:
    if not isinstance(value, list) or len(value) != 2:
        return False
    first, second = value
    if not isinstance(first, dict) or not isinstance(second, dict):
        return False
    return (
        first.get("turn_id") == 1
        and first.get("speaker_name") == "佐伯"
        and first.get("text") == "今日はどんなことを話したいですか。"
        and second.get("turn_id") == 2
        and second.get("speaker_name") == "高橋"
        and second.get("text") == "家族との距離感に悩んでいます。"
    )


def start_live_session_for_ui(
    *,
    config: Any,
    counselor_profile: Profile,
    client_profile: Profile,
    theme: Theme,
    text_client: object | None = None,
    moderation_client: object | None = None,
    clinical_ng_checker: object | None = None,
    tts_client: object | None = None,
    log_paths: SessionLogPaths | None = None,
    sessions_dir: Path | str | None = None,
    ahead_generation_turns: int | None = None,
) -> dict[str, Any]:
    session = SessionState(status=SessionStatus.RUNNING)
    if log_paths is None and sessions_dir is not None:
        log_paths = create_session_log_dirs(sessions_dir, session.session_id)

    if text_client is None:
        live_clients = create_openai_live_clients()
        text_client = live_clients["text_client"]
        moderation_client = moderation_client or live_clients["moderation_client"]
        clinical_ng_checker = clinical_ng_checker or live_clients["clinical_ng_checker"]
        tts_client = tts_client or live_clients["tts_client"]

    engine = build_conversation_engine_for_ui(
        config=config,
        text_client=text_client,
        counselor_profile=counselor_profile,
        client_profile=client_profile,
        theme=theme,
        moderation_client=moderation_client,
        clinical_ng_checker=clinical_ng_checker,
        log_paths=log_paths,
        ahead_generation_turns=ahead_generation_turns,
    )
    generation_warnings = refill_generation_queue_for_ui(
        session,
        conversation_engine=engine,
        cumulative_audio_seconds=0.0,
        tts_client=tts_client,
        log_paths=log_paths,
        response_format=config.openai.default_audio_format,
    )
    engine.update_conversation_phase(session, cumulative_audio_seconds=0.0)

    warnings = generation_warnings or [
        f"先行生成ターン {len(unplayed_turn_rows_from_session(session))} 件を生成しました。"
    ]
    return live_session_ui_updates(
        session,
        selected_turn_id=None,
        warnings=warnings,
        conversation_engine=engine,
        log_paths=log_paths,
        text_client=text_client,
        moderation_client=moderation_client,
        clinical_ng_checker=clinical_ng_checker,
        tts_client=tts_client,
        moderation_model=config.openai.moderation_model,
        clinical_model=config.openai.default_text_model,
    )


def skip_current_audio_for_ui(
    session: SessionState,
    *,
    conversation_engine: ConversationEngine,
    cumulative_audio_seconds: float,
    tts_client: object | None,
    log_paths: SessionLogPaths | None,
    response_format: str,
    selected_turn_id: int | None,
) -> dict[str, Any]:
    decision = next_playback_decision(session)
    if decision.action == "hold":
        return live_session_ui_updates(
            session,
            selected_turn_id=selected_turn_id,
            warnings=["playback_hold のターンがあるため、先に確認してください。"],
        )
    if decision.turn is None:
        return live_session_ui_updates(
            session,
            selected_turn_id=selected_turn_id,
            warnings=["スキップ対象の未再生ターンがありません。"],
        )

    updated_total_seconds = record_playback_terminal_event(
        decision.turn,
        PlaybackEvent(
            event_name="skip",
            current_seconds=0.0,
            duration_seconds=0.0,
        ),
        previous_total_seconds=cumulative_audio_seconds,
        already_counted=False,
        log_paths=log_paths,
    )
    warnings = [f"Turn {decision.turn.turn_id} をスキップしました。"]
    warnings.extend(
        refill_generation_queue_for_ui(
            session,
            conversation_engine=conversation_engine,
            cumulative_audio_seconds=updated_total_seconds,
            tts_client=tts_client,
            log_paths=log_paths,
            response_format=response_format,
        )
    )
    return live_session_ui_updates(
        session,
        selected_turn_id=selected_turn_id,
        warnings=warnings,
    ) | {"cumulative_audio_seconds": updated_total_seconds}


def mark_current_audio_played_for_ui(
    session: SessionState,
    *,
    conversation_engine: ConversationEngine,
    cumulative_audio_seconds: float,
    tts_client: object | None,
    log_paths: SessionLogPaths | None,
    response_format: str,
    selected_turn_id: int | None,
    duration_seconds: float | None = None,
    refill_after_playback: bool = True,
) -> dict[str, Any]:
    decision = next_playback_decision(session)
    if decision.action == "hold":
        return live_session_ui_updates(
            session,
            selected_turn_id=selected_turn_id,
            warnings=["playback_hold のターンがあるため、先に確認してください。"],
        )
    if decision.action != "play" or decision.turn is None:
        return live_session_ui_updates(
            session,
            selected_turn_id=selected_turn_id,
            warnings=[f"再生完了にできる音声がありません: {decision.reason}"],
        )

    played_turn = decision.turn
    played_seconds = (
        max(0.0, duration_seconds)
        if duration_seconds is not None
        else audio_duration_seconds(played_turn.active_revision().audio_path)
    )
    updated_total_seconds = record_playback_terminal_event(
        played_turn,
        PlaybackEvent(
            event_name="ended",
            current_seconds=played_seconds,
            duration_seconds=played_seconds,
        ),
        previous_total_seconds=cumulative_audio_seconds,
        already_counted=False,
        log_paths=log_paths,
    )
    if log_paths is not None:
        append_turn_transcripts(log_paths, played_turn)
        update_transcript_markdown(log_paths, session)

    warnings = [f"Turn {played_turn.turn_id} を再生済みにしました。"]
    if refill_after_playback:
        warnings.extend(
            refill_generation_queue_for_ui(
                session,
                conversation_engine=conversation_engine,
                cumulative_audio_seconds=updated_total_seconds,
                tts_client=tts_client,
                log_paths=log_paths,
                response_format=response_format,
            )
        )
    return live_session_ui_updates(
        session,
        selected_turn_id=selected_turn_id,
        warnings=warnings,
    ) | {"cumulative_audio_seconds": updated_total_seconds}


def auto_progress_live_session_for_ui(
    session: SessionState,
    *,
    conversation_engine: ConversationEngine,
    cumulative_audio_seconds: float,
    tts_client: object | None,
    log_paths: SessionLogPaths | None,
    response_format: str,
    selected_turn_id: int | None,
    playback_turn_id: int | None,
    playback_started_at: float | None,
    playback_duration_seconds: float | None,
    now_monotonic: float,
    playback_start_monotonic: float | None = None,
    playback_gap_seconds: float = AUTO_PLAYBACK_DEFAULT_GAP_SECONDS,
    existing_warnings: list[str] | None = None,
    allow_server_side_playback_completion: bool = False,
    playback_rendered_at: float | None = None,
    browser_audio_player_manages_playback: bool = False,
) -> dict[str, Any]:
    if session.status != SessionStatus.RUNNING:
        return clear_auto_playback_updates()

    warnings: list[str] = []
    updated_total_seconds = cumulative_audio_seconds
    current_playback_turn_id = playback_turn_id
    current_playback_started_at = playback_started_at
    current_playback_duration_seconds = playback_duration_seconds
    original_playback_state = (
        playback_turn_id,
        playback_started_at,
        playback_duration_seconds,
    )
    last_playback_finished_at: float | None = None
    sequence_warnings, invalidated_turn_ids = repair_speaker_sequence_for_ui(session)
    warnings.extend(sequence_warnings)
    if current_playback_turn_id in invalidated_turn_ids:
        current_playback_turn_id = None
        current_playback_started_at = None
        current_playback_duration_seconds = None

    # Browser playback events are the only source of truth for completion.
    # The server-side timer remains a scheduling hint and must not mark audio played.

    if current_playback_turn_id is not None and session.status == SessionStatus.RUNNING:
        warnings.extend(
            refill_generation_queue_for_ui(
                session,
                conversation_engine=conversation_engine,
                cumulative_audio_seconds=updated_total_seconds,
                tts_client=tts_client,
                log_paths=log_paths,
                response_format=response_format,
                active_playback_turn_id=current_playback_turn_id,
                suppress_success_warnings=True,
            )
        )

    next_playback_started_at = (
        time.monotonic()
        if playback_start_monotonic is None
        else playback_start_monotonic
    )
    decision = next_playback_decision(session)
    recovered_from_wait = False
    if (
        current_playback_turn_id is None
        and session.status == SessionStatus.RUNNING
        and decision.action not in {"play", "hold"}
    ):
        recovered_from_wait = True
        warnings.extend(
            refill_generation_queue_for_ui(
                session,
                conversation_engine=conversation_engine,
                cumulative_audio_seconds=updated_total_seconds,
                tts_client=tts_client,
                log_paths=log_paths,
                response_format=response_format,
            )
        )
        decision = next_playback_decision(session)

    if decision.action == "play" and decision.turn is not None:
        if (
            current_playback_turn_id != decision.turn.turn_id
            and (recovered_from_wait or not browser_audio_player_manages_playback)
        ):
            current_playback_turn_id = decision.turn.turn_id
            current_playback_started_at = next_playback_started_at
            if last_playback_finished_at is not None:
                current_playback_started_at = max(
                    current_playback_started_at,
                    last_playback_finished_at + max(0.0, playback_gap_seconds),
                )
            current_playback_duration_seconds = playback_duration_for_turn(
                decision.turn
            )
    elif decision.action in {"complete", "hold"}:
        current_playback_turn_id = None
        current_playback_started_at = None
        current_playback_duration_seconds = None

    if any("失敗しました" in warning for warning in warnings):
        session.status = SessionStatus.ERROR
        current_playback_turn_id = None
        current_playback_started_at = None
        current_playback_duration_seconds = None

    ui_warnings = warnings if warnings else list(existing_warnings or [])
    progress_changed = (
        bool(warnings)
        or updated_total_seconds != cumulative_audio_seconds
        or original_playback_state
        != (
            current_playback_turn_id,
            current_playback_started_at,
            current_playback_duration_seconds,
        )
        or session.status != SessionStatus.RUNNING
    )
    return live_session_ui_updates(
        session,
        selected_turn_id=selected_turn_id,
        warnings=ui_warnings,
    ) | {
        "cumulative_audio_seconds": updated_total_seconds,
        "auto_playback_turn_id": current_playback_turn_id,
        "auto_playback_started_at": current_playback_started_at,
        "auto_playback_duration_seconds": current_playback_duration_seconds,
        "_auto_progress_changed": progress_changed,
    }


def repair_speaker_sequence_for_ui(session: SessionState) -> tuple[list[str], set[int]]:
    invalidated = invalidate_unplayed_speaker_sequence_violations(session)
    if not invalidated:
        return [], set()
    turn_ids = {turn.turn_id for turn in invalidated}
    joined_turn_ids = ", ".join(str(turn_id) for turn_id in sorted(turn_ids))
    return (
        [f"話者順が連続した未再生ターン {joined_turn_ids} を無効化しました。"],
        turn_ids,
    )


def playback_duration_for_turn(turn: Turn) -> float:
    duration_seconds = audio_duration_seconds(turn.active_revision().audio_path)
    if duration_seconds > 0:
        return duration_seconds
    return AUTO_PLAYBACK_FALLBACK_DURATION_SECONDS


def clear_auto_playback_updates() -> dict[str, Any]:
    return {
        "auto_playback_turn_id": None,
        "auto_playback_started_at": None,
        "auto_playback_duration_seconds": None,
        "auto_playback_visible_turn_id": None,
        "auto_playback_visible_started_at": None,
    }


def schedule_next_audio_playback_updates(
    session: SessionState,
    *,
    start_delay_seconds: float = 0.0,
) -> dict[str, Any]:
    decision = next_playback_decision(session)
    if decision.action != "play" or decision.turn is None:
        return clear_auto_playback_updates()
    return {
        "auto_playback_turn_id": decision.turn.turn_id,
        "auto_playback_started_at": time.monotonic() + max(0.0, start_delay_seconds),
        "auto_playback_duration_seconds": playback_duration_for_turn(decision.turn),
        "auto_playback_visible_turn_id": None,
        "auto_playback_visible_started_at": None,
    }


def audio_duration_seconds(audio_path: str | Path | None) -> float:
    if audio_path is None:
        return 0.0
    path = Path(audio_path)
    if not path.exists():
        return 0.0
    try:
        from mutagen import File as mutagen_file

        audio = mutagen_file(path)
    except Exception:
        return 0.0
    info = getattr(audio, "info", None)
    length = getattr(info, "length", 0.0)
    try:
        return max(0.0, float(length))
    except (TypeError, ValueError):
        return 0.0


def audio_mime_type(audio_format: str | None, audio_path: str | Path | None) -> str:
    normalized = (audio_format or Path(str(audio_path or "")).suffix.lstrip(".")).lower()
    return {
        "mp3": "audio/mpeg",
        "mpeg": "audio/mpeg",
        "wav": "audio/wav",
        "ogg": "audio/ogg",
        "opus": "audio/ogg",
        "flac": "audio/flac",
    }.get(normalized, "audio/mpeg")


def audio_data_uri_for_turn(turn: Turn) -> str | None:
    revision = turn.active_revision()
    audio_path = Path(revision.audio_path or "")
    if not audio_path.exists():
        return None
    encoded = base64.b64encode(audio_path.read_bytes()).decode("ascii")
    mime_type = audio_mime_type(revision.audio_format, audio_path)
    return f"data:{mime_type};base64,{encoded}"


def build_audio_player_queue_for_ui(
    session: SessionState,
    *,
    max_items: int = 5,
) -> tuple[str, list[dict[str, Any]]]:
    return build_browser_audio_queue(
        session,
        audio_src_for_turn=audio_data_uri_for_turn,
        duration_hint_for_turn=playback_duration_for_turn,
        max_items=max_items,
    )


def runtime_control_defaults(
    config_path: Path | str | None = None,
) -> dict[str, Any]:
    resolved_config_path = Path(config_path or ROOT_DIR / "config" / "runtime_config.yaml")
    try:
        settings = load_runtime_config(resolved_config_path)
    except RuntimeConfigLoadError as exc:
        fallback = _runtime_control_defaults_from_yaml(resolved_config_path)
        if fallback is not None:
            return fallback
        return {
            "host": "127.0.0.1",
            "port": 8765,
            "generation_mode": "realtime_api",
            "warning": str(exc),
        }
    return {
        "host": settings.runtime.control_host,
        "port": settings.runtime.control_port,
        "generation_mode": (
            "realtime_api"
            if settings.latency.realtime_api_centered_mode
            else "responses_tts"
        ),
        "warning": "",
    }


def _runtime_control_defaults_from_yaml(config_path: Path) -> dict[str, Any] | None:
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(raw, dict):
        return None
    runtime = raw.get("runtime")
    latency = raw.get("latency")
    if not isinstance(runtime, dict):
        return None
    generation_mode = "realtime_api"
    if isinstance(latency, dict) and latency.get("realtime_api_centered_mode") is False:
        generation_mode = "responses_tts"
    try:
        port = int(runtime.get("control_port") or 8765)
    except (TypeError, ValueError):
        port = 8765
    return {
        "host": str(runtime.get("control_host") or "127.0.0.1"),
        "port": port,
        "generation_mode": generation_mode,
        "warning": "",
    }


def runtime_monitor_endpoint(*, host: str, port: int):
    return build_runtime_control_endpoint(host=host, port=port)


def runtime_audio_monitor_document(
    *,
    ws_url: str,
    volume: float,
    muted: bool,
    start_url: str = "",
    runtime_pause_url: str = "",
    runtime_resume_url: str = "",
    runtime_stop_url: str = "",
    runtime_interrupt_url: str = "",
    runtime_playback_completed_url: str = "",
    runtime_status_url: str = "",
    human_audio_url: str = "",
    human_audio_stream_url: str = "",
    human_turn_url: str = "",
    human_input_enabled: bool = False,
    start_options: dict[str, Any] | None = None,
    warmup_target_turns: int | None = None,
    warmup_max_wait_ms: int | None = None,
) -> str:
    args = {
        "ws_url": ws_url,
        "connect": False,
        "volume": volume,
        "muted": muted,
        "start_url": start_url,
        "runtime_pause_url": runtime_pause_url,
        "runtime_resume_url": runtime_resume_url,
        "runtime_stop_url": runtime_stop_url,
        "runtime_interrupt_url": runtime_interrupt_url,
        "runtime_playback_completed_url": runtime_playback_completed_url,
        "runtime_status_url": runtime_status_url,
        "human_audio_url": human_audio_url,
        "human_audio_stream_url": human_audio_stream_url,
        "human_turn_url": human_turn_url,
        "human_input_enabled": bool(human_input_enabled),
        "start_options": start_options or {},
    }
    if warmup_target_turns is not None:
        args["warmup_target_turns"] = max(0, int(warmup_target_turns))
    if warmup_max_wait_ms is not None:
        args["warmup_max_wait_ms"] = max(0, int(warmup_max_wait_ms))
    args_json = json.dumps(
        args,
        ensure_ascii=False,
    )
    initial_args = f"<script>window.RUNTIME_AUDIO_MONITOR_ARGS = {args_json};</script>"
    return RUNTIME_AUDIO_MONITOR_HTML.replace("</head>", f"{initial_args}\n  </head>")


def apply_runtime_control_action(
    action: str,
    *,
    client: RuntimeControlClient,
    current_session_id: str = "",
    start_options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if action not in {"start", "pause", "resume", "stop", "status"}:
        return {"runtime_control_error": f"未対応の runtime 操作です: {action}"}
    try:
        if action == "start":
            status = client.start(start_options)
        elif action == "pause":
            status = client.pause(reason="generation_throttle")
        else:
            status = getattr(client, action)()
    except RuntimeObserverClientError as exc:
        return {"runtime_control_error": str(exc)}
    session_id = str(status.get("session_id") or current_session_id or "")
    return {
        "runtime_control_status": status,
        "runtime_control_error": "",
        "runtime_monitor_session_id": session_id,
    } | runtime_status_session_updates(status)


def refresh_runtime_status(
    *,
    client: RuntimeControlClient,
    current_session_id: str = "",
) -> dict[str, Any]:
    try:
        status = client.status()
    except RuntimeObserverClientError as exc:
        return {"runtime_control_error": str(exc)}
    session_id = str(status.get("session_id") or current_session_id or "")
    return {
        "runtime_control_status": status,
        "runtime_control_error": "",
        "runtime_monitor_session_id": session_id,
    } | runtime_status_session_updates(status)


def should_auto_refresh_runtime_transcript(status: dict[str, Any], session_id: str) -> bool:
    if not session_id.strip():
        return False
    return str(status.get("phase") or "") in {"running", "paused"}


def runtime_generation_throttle_decision(
    *,
    status: dict[str, Any],
    playback_completed_turn_count: int,
    lead_limit: int,
    throttle_active: bool,
    throttle_disabled: bool = False,
) -> dict[str, Any]:
    phase = str(status.get("phase") or "")
    pause_reason = str(status.get("pause_reason") or "")
    generated_turns = runtime_generated_turn_count_for_display(
        status, include_active_turn=False
    )
    playback_completed_count = max(0, int(playback_completed_turn_count))
    normalized_limit = max(1, int(lead_limit))
    resume_limit = runtime_generation_resume_lead_limit(normalized_limit)
    lead_turns = max(0, generated_turns - playback_completed_count)
    action: str | None = None
    next_throttle_active = bool(throttle_active)

    auto_throttle_pause = (
        phase == "paused"
        and (
            (throttle_active and pause_reason in {"", "generation_throttle"})
            or pause_reason == "generation_throttle"
        )
    )

    if throttle_disabled:
        if auto_throttle_pause:
            action = "resume"
        next_throttle_active = False
    elif phase == "running" and lead_turns >= normalized_limit:
        action = "pause"
        next_throttle_active = True
    elif auto_throttle_pause and lead_turns <= resume_limit:
        action = "resume"
        next_throttle_active = False
    elif phase not in {"running", "paused"}:
        next_throttle_active = False

    return {
        "runtime_generation_lead_turns": lead_turns,
        "runtime_generation_resume_lead_limit": resume_limit,
        "runtime_generation_throttle_active": next_throttle_active,
        "runtime_generation_throttle_action": action,
    }


def runtime_generation_resume_lead_limit(lead_limit: int) -> int:
    normalized_limit = max(1, int(lead_limit))
    if normalized_limit <= 1:
        return 0
    return max(1, normalized_limit - 2)


def runtime_generated_turn_count_for_display(
    status: dict[str, Any],
    *,
    include_active_turn: bool = True,
) -> int:
    current_turn_id = status.get("current_turn_id")
    if include_active_turn and current_turn_id is not None:
        return max(0, int(current_turn_id) + 1)
    completed_turns = max(0, int(status.get("completed_turns") or 0))
    return max(0, completed_turns)


def runtime_current_turn_metric_value(status: dict[str, Any]) -> str:
    current_turn_id = status.get("current_turn_id")
    if current_turn_id is None:
        return "-"
    return f"{max(0, int(current_turn_id))}ターン目"


def runtime_status_playback_completed_turn_count(
    status: dict[str, Any],
) -> int | None:
    if "playback_completed_turns" not in status:
        return None
    try:
        return max(0, int(status.get("playback_completed_turns") or 0))
    except (TypeError, ValueError):
        return None


def runtime_completed_turn_count_for_display(
    *,
    turns: list[PublicTranscriptTurn],
    clock_seconds: float,
) -> int:
    if not turns:
        return 0
    current_seconds = max(0.0, float(clock_seconds))
    completed_visible_count = 0
    start_seconds_by_index = runtime_playback_relative_start_seconds(turns)

    for index, turn in enumerate(turns):
        completion_boundary_seconds: float | None
        duration_seconds = public_turn_known_audio_duration_seconds(turn)
        if duration_seconds is not None:
            completion_boundary_seconds = (
                start_seconds_by_index[index] + duration_seconds
            )
        elif index + 1 < len(start_seconds_by_index):
            completion_boundary_seconds = start_seconds_by_index[index + 1]
        else:
            completion_boundary_seconds = None
        if (
            completion_boundary_seconds is not None
            and current_seconds >= completion_boundary_seconds
        ):
            completed_visible_count += 1
    return max(0, completed_visible_count)


def runtime_playback_relative_start_seconds(
    turns: list[PublicTranscriptTurn],
    *,
    inter_turn_gap_seconds: float = RUNTIME_AUDIO_MONITOR_PLAYBACK_GAP_MAX_SECONDS,
) -> list[float]:
    """Approximate browser playback starts, excluding only leading generation silence."""
    if not turns:
        return []
    raw_start_seconds: list[float] = []
    fallback_start_seconds = 0.0
    for turn in turns:
        start_seconds = public_turn_start_seconds(
            turn,
            fallback_cumulative_seconds=fallback_start_seconds,
        )
        raw_start_seconds.append(start_seconds)
        fallback_start_seconds = start_seconds + public_turn_audio_duration_seconds(turn)

    relative_starts = [0.0]
    first_raw_start = raw_start_seconds[0]
    gap_seconds = max(0.0, float(inter_turn_gap_seconds))
    for index in range(1, len(turns)):
        previous_duration = public_turn_known_audio_duration_seconds(turns[index - 1])
        if previous_duration is not None:
            relative_starts.append(
                relative_starts[-1] + previous_duration + gap_seconds
            )
            continue
        relative_starts.append(
            max(
                relative_starts[-1],
                raw_start_seconds[index] - first_raw_start,
            )
        )
    return relative_starts


def runtime_playback_estimated_total_seconds(
    turns: list[PublicTranscriptTurn],
) -> float:
    if not turns:
        return 0.0
    start_seconds_by_index = runtime_playback_relative_start_seconds(turns)
    last_turn = turns[-1]
    last_duration = public_turn_known_audio_duration_seconds(last_turn)
    if last_duration is None:
        return start_seconds_by_index[-1]
    return start_seconds_by_index[-1] + last_duration


def runtime_completed_turn_count_for_metric(
    *,
    status: dict[str, Any],
    turns: list[PublicTranscriptTurn],
    clock_seconds: float,
    session_id: str,
    previous_session_id: str,
    previous_completed_turns: int,
    browser_completed_turn_count: int | None = None,
) -> int:
    generated_turns = runtime_generated_turn_count_for_display(status)
    if browser_completed_turn_count is not None:
        completed_turns = max(0, int(browser_completed_turn_count))
        if generated_turns > 0:
            completed_turns = min(completed_turns, generated_turns)
        return completed_turns

    computed_turns = runtime_completed_turn_count_for_display(
        turns=turns,
        clock_seconds=clock_seconds,
    )
    previous_turns = (
        max(0, int(previous_completed_turns))
        if session_id.strip() and session_id.strip() == previous_session_id.strip()
        else 0
    )
    completed_turns = max(computed_turns, previous_turns)
    terminal_phase = str(status.get("phase") or "") in {"completed", "stopped"}
    final_turn_index = next(
        (
            index
            for index, turn in reversed(list(enumerate(turns)))
            if int(turn.turn_id) == generated_turns - 1
        ),
        None,
    )
    final_turn = turns[final_turn_index] if final_turn_index is not None else None
    final_turn_duration_seconds = (
        public_turn_known_audio_duration_seconds(final_turn)
        if final_turn is not None
        else None
    )
    playback_start_seconds = (
        runtime_playback_relative_start_seconds(turns)[final_turn_index]
        if final_turn_index is not None
        else None
    )
    final_turn_end_seconds = (
        playback_start_seconds + final_turn_duration_seconds
        if playback_start_seconds is not None
        and final_turn_duration_seconds is not None
        else None
    )
    if (
        terminal_phase
        and generated_turns > 0
        and completed_turns >= generated_turns - 1
        and final_turn_end_seconds is not None
        and max(0.0, float(clock_seconds)) >= final_turn_end_seconds
    ):
        completed_turns = generated_turns
    if generated_turns > 0:
        completed_turns = min(completed_turns, generated_turns)
    return max(0, completed_turns)


def runtime_completed_turn_count_state_updates(
    *,
    status: dict[str, Any],
    turns: list[PublicTranscriptTurn],
    clock_seconds: float,
    session_id: str,
    previous_session_id: str,
    previous_completed_turns: int,
    browser_completed_turn_count: int | None = None,
) -> dict[str, Any]:
    completed_turns = runtime_completed_turn_count_for_metric(
        status=status,
        turns=turns,
        clock_seconds=clock_seconds,
        session_id=session_id,
        previous_session_id=previous_session_id,
        previous_completed_turns=previous_completed_turns,
        browser_completed_turn_count=browser_completed_turn_count,
    )
    return {
        "runtime_completed_turn_count_session_id": session_id,
        "runtime_completed_turn_count_for_display": completed_turns,
    }


def runtime_status_label_for_timer(
    status_label: str, *, generated_turns: int, completed_turns: int
) -> str:
    if (
        status_label == SessionStatus.COMPLETED.value
        and int(completed_turns) < int(generated_turns)
    ):
        return SessionStatus.RUNNING.value
    return status_label


def runtime_generation_throttle_disabled_for_closing(
    *,
    status: dict[str, Any],
    clock_seconds: float,
    closing_start_seconds: float,
) -> bool:
    if (
        status.get("closing_started_turn_id") is not None
        or status.get("closing_count_started_turn_id") is not None
    ):
        return True
    return max(0.0, float(clock_seconds)) >= max(0.0, float(closing_start_seconds))


def public_transcript_total_audio_seconds(turns: list[PublicTranscriptTurn]) -> float:
    total_seconds = 0.0
    fallback_seconds = 0.0
    for turn in turns:
        duration_seconds = public_turn_audio_duration_seconds(turn)
        cumulative_audio_seconds = getattr(turn, "cumulative_audio_seconds", None)
        if cumulative_audio_seconds is not None:
            total_seconds = max(
                total_seconds,
                float(cumulative_audio_seconds) + duration_seconds,
            )
        else:
            fallback_seconds += duration_seconds
            total_seconds = max(total_seconds, fallback_seconds)
    return total_seconds


def public_transcript_turns_visible_after_audio_end(
    turns: list[PublicTranscriptTurn],
    *,
    playback_seconds: float,
) -> list[PublicTranscriptTurn]:
    visible_turns: list[PublicTranscriptTurn] = []
    visible_until_seconds = max(0.0, float(playback_seconds))
    fallback_start_seconds = 0.0
    for turn in turns:
        start_seconds = public_turn_start_seconds(
            turn,
            fallback_cumulative_seconds=fallback_start_seconds,
        )
        duration_seconds = public_turn_audio_duration_seconds(turn)
        end_seconds = start_seconds + duration_seconds
        if end_seconds <= visible_until_seconds:
            visible_turns.append(turn)
        fallback_start_seconds = end_seconds
    return visible_turns


def public_transcript_turns_visible_after_previous_audio_end(
    turns: list[PublicTranscriptTurn],
    *,
    playback_seconds: float,
) -> list[PublicTranscriptTurn]:
    visible_turns: list[PublicTranscriptTurn] = []
    visible_until_seconds = max(0.0, float(playback_seconds))
    previous_end_seconds = 0.0
    for turn in turns:
        start_seconds = public_turn_start_seconds(
            turn,
            fallback_cumulative_seconds=previous_end_seconds,
        )
        visible_from_seconds = max(previous_end_seconds, start_seconds)
        if visible_from_seconds <= visible_until_seconds:
            visible_turns.append(turn)
        duration_seconds = public_turn_audio_duration_seconds(turn)
        previous_end_seconds = start_seconds + duration_seconds
    return visible_turns


def runtime_public_transcript_turns_visible_for_audio(
    turns: list[PublicTranscriptTurn],
    *,
    playback_seconds: float,
) -> list[PublicTranscriptTurn]:
    return public_transcript_turns_visible_after_previous_audio_end(
        turns,
        playback_seconds=playback_seconds,
    )


def public_transcript_latest_visibility_seconds(
    turns: list[PublicTranscriptTurn],
) -> float:
    latest_visibility_seconds = 0.0
    previous_end_seconds = 0.0
    for turn in turns:
        start_seconds = public_turn_start_seconds(
            turn,
            fallback_cumulative_seconds=previous_end_seconds,
        )
        latest_visibility_seconds = max(previous_end_seconds, start_seconds)
        duration_seconds = public_turn_audio_duration_seconds(turn)
        previous_end_seconds = start_seconds + duration_seconds
    return latest_visibility_seconds


def runtime_transcript_visibility_state_updates(
    *,
    session_id: str,
    turns: list[PublicTranscriptTurn],
    wall_clock_seconds: float,
    visibility_session_id: str,
    visible_turn_count: int,
    next_reveal_wall_seconds: float | None,
) -> dict[str, Any]:
    active_session_id = session_id.strip()
    current_wall_seconds = max(0.0, float(wall_clock_seconds))
    updates: dict[str, Any] = {}
    if not active_session_id:
        return {
            "runtime_transcript_visibility_session_id": "",
            "runtime_transcript_visible_turn_count": 0,
            "runtime_transcript_next_reveal_wall_seconds": None,
        }
    if active_session_id != visibility_session_id:
        updates["runtime_transcript_visibility_session_id"] = active_session_id
        visible_turn_count = 0
        next_reveal_wall_seconds = None

    if not turns:
        return updates | {
            "runtime_transcript_visible_turn_count": 0,
            "runtime_transcript_next_reveal_wall_seconds": None,
        }

    visible_turn_count = max(0, min(int(visible_turn_count), len(turns)))
    if visible_turn_count == 0:
        visible_turn_count = 1
        visible_duration = public_turn_known_audio_duration_seconds(turns[0])
        next_reveal_wall_seconds = (
            current_wall_seconds
            + visible_duration
            + RUNTIME_AUDIO_MONITOR_PLAYBACK_GAP_MAX_SECONDS
            if visible_duration is not None
            else None
        )
    elif (
        visible_turn_count < len(turns)
        and next_reveal_wall_seconds is not None
        and current_wall_seconds >= float(next_reveal_wall_seconds)
    ):
        scheduled_reveal_wall_seconds = float(next_reveal_wall_seconds)
        visible_turn_count += 1
        visible_duration = public_turn_known_audio_duration_seconds(
            turns[visible_turn_count - 1]
        )
        next_reveal_wall_seconds = (
            scheduled_reveal_wall_seconds
            + visible_duration
            + RUNTIME_AUDIO_MONITOR_PLAYBACK_GAP_MAX_SECONDS
            if visible_duration is not None
            else None
        )
    elif next_reveal_wall_seconds is None and visible_turn_count > 0:
        visible_duration = public_turn_known_audio_duration_seconds(
            turns[visible_turn_count - 1]
        )
        next_reveal_wall_seconds = (
            current_wall_seconds
            + visible_duration
            + RUNTIME_AUDIO_MONITOR_PLAYBACK_GAP_MAX_SECONDS
            if visible_duration is not None
            else None
        )

    return updates | {
        "runtime_transcript_visible_turn_count": visible_turn_count,
        "runtime_transcript_next_reveal_wall_seconds": next_reveal_wall_seconds,
    }


def runtime_transcript_visibility_clock_state_updates(
    *,
    session_id: str,
    phase: str,
    wall_clock_seconds: float,
    visibility_session_id: str,
    visible_turn_count: int,
    total_turn_count: int,
    clock_started_at_monotonic: float | None,
    clock_base_seconds: float,
    now_monotonic: float,
    continue_when_paused: bool = False,
    playback_paused: bool = False,
    completion_clock_seconds: float | None = None,
) -> dict[str, Any]:
    active_session_id = session_id.strip()
    current_wall_seconds = max(0.0, float(wall_clock_seconds))
    if not active_session_id:
        return {
            "runtime_transcript_visibility_clock_seconds": 0.0,
            "runtime_transcript_visibility_clock_started_at_monotonic": None,
            "runtime_transcript_visibility_clock_base_seconds": 0.0,
        }

    started_at = clock_started_at_monotonic
    base_seconds = max(0.0, float(clock_base_seconds))
    if active_session_id != visibility_session_id:
        started_at = None
        base_seconds = current_wall_seconds

    visible_count = max(0, int(visible_turn_count))
    total_count = max(0, int(total_turn_count))
    has_pending_turns = visible_count < total_count
    completion_seconds = (
        max(0.0, float(completion_clock_seconds))
        if completion_clock_seconds is not None
        else None
    )
    current_visibility_seconds = (
        base_seconds + max(0.0, now_monotonic - started_at)
        if started_at is not None
        else max(current_wall_seconds, base_seconds)
    )
    has_pending_completion = (
        completion_seconds is not None
        and total_count > 0
        and current_visibility_seconds < completion_seconds
    )
    should_continue_clock = not playback_paused and (
        phase == "completed" or (phase == "paused" and continue_when_paused)
    )
    if (
        should_continue_clock
        and not has_pending_turns
        and completion_seconds is not None
        and current_visibility_seconds >= completion_seconds
        and current_wall_seconds < completion_seconds
    ):
        return {
            "runtime_transcript_visibility_clock_seconds": completion_seconds,
            "runtime_transcript_visibility_clock_started_at_monotonic": None,
            "runtime_transcript_visibility_clock_base_seconds": completion_seconds,
        }
    if should_continue_clock and (has_pending_turns or has_pending_completion):
        if started_at is None:
            started_at = now_monotonic
            base_seconds = current_wall_seconds
        updated_seconds = base_seconds + max(0.0, now_monotonic - started_at)
        if (
            not has_pending_turns
            and completion_seconds is not None
            and updated_seconds >= completion_seconds
        ):
            return {
                "runtime_transcript_visibility_clock_seconds": completion_seconds,
                "runtime_transcript_visibility_clock_started_at_monotonic": None,
                "runtime_transcript_visibility_clock_base_seconds": completion_seconds,
            }
        return {
            "runtime_transcript_visibility_clock_seconds": updated_seconds,
            "runtime_transcript_visibility_clock_started_at_monotonic": started_at,
            "runtime_transcript_visibility_clock_base_seconds": base_seconds,
        }

    return {
        "runtime_transcript_visibility_clock_seconds": current_wall_seconds,
        "runtime_transcript_visibility_clock_started_at_monotonic": None,
        "runtime_transcript_visibility_clock_base_seconds": current_wall_seconds,
    }


def runtime_initial_warmup_state_updates(
    *,
    status: dict[str, Any],
    session_id: str,
    warmup_target_turns: int,
    warmup_max_wait_seconds: float,
    warmup_session_id: str,
    warmup_started_at_monotonic: float | None,
    warmup_released: bool,
    now_monotonic: float,
) -> dict[str, Any]:
    active_session_id = session_id.strip()
    phase = str(status.get("phase") or "")
    if not active_session_id or phase in {"idle", "stopped", "completed", "error"}:
        return {
            "runtime_initial_warmup_session_id": "",
            "runtime_initial_warmup_started_at_monotonic": None,
            "runtime_initial_warmup_released": False,
            "runtime_initial_warmup_active": False,
        }

    target_turns = max(0, int(warmup_target_turns))
    max_wait_seconds = max(0.0, float(warmup_max_wait_seconds))
    if active_session_id != warmup_session_id.strip():
        started_at = None
        released = False
    else:
        started_at = warmup_started_at_monotonic
        released = bool(warmup_released)

    if target_turns <= 0 or phase not in {"running", "paused"}:
        return {
            "runtime_initial_warmup_session_id": active_session_id,
            "runtime_initial_warmup_started_at_monotonic": None,
            "runtime_initial_warmup_released": True,
            "runtime_initial_warmup_active": False,
        }

    completed_turns = max(0, int(status.get("completed_turns") or 0))
    if completed_turns >= target_turns:
        released = True

    if released:
        return {
            "runtime_initial_warmup_session_id": active_session_id,
            "runtime_initial_warmup_started_at_monotonic": None,
            "runtime_initial_warmup_released": True,
            "runtime_initial_warmup_active": False,
        }

    if started_at is None:
        started_at = float(now_monotonic)

    if (
        max_wait_seconds > 0
        and max(0.0, float(now_monotonic) - float(started_at)) >= max_wait_seconds
    ):
        return {
            "runtime_initial_warmup_session_id": active_session_id,
            "runtime_initial_warmup_started_at_monotonic": None,
            "runtime_initial_warmup_released": True,
            "runtime_initial_warmup_active": False,
        }

    return {
        "runtime_initial_warmup_session_id": active_session_id,
        "runtime_initial_warmup_started_at_monotonic": float(started_at),
        "runtime_initial_warmup_released": False,
        "runtime_initial_warmup_active": True,
    }


def runtime_timer_state_updates(
    *,
    status: dict[str, Any],
    session_id: str,
    transcript_audio_seconds: float,
    current_wall_clock_seconds: float,
    wall_clock_session_id: str,
    wall_clock_started_at_monotonic: float | None,
    wall_clock_base_seconds: float,
    now_monotonic: float,
    continue_when_paused: bool = False,
    playback_paused: bool = False,
    playback_warmup_active: bool = False,
) -> dict[str, Any]:
    phase = str(status.get("phase") or "")
    active_session_id = session_id.strip()
    updates: dict[str, Any] = {
        "cumulative_audio_seconds": max(0.0, float(transcript_audio_seconds)),
    }

    if not active_session_id:
        return updates

    started_at = wall_clock_started_at_monotonic
    base_seconds = max(0.0, float(wall_clock_base_seconds))
    current_seconds = max(0.0, float(current_wall_clock_seconds))
    if active_session_id != wall_clock_session_id:
        updates["runtime_wall_clock_session_id"] = active_session_id
        base_seconds = 0.0
        current_seconds = 0.0
        started_at = None

    if playback_warmup_active:
        updates["wall_clock_seconds"] = current_seconds
        updates["runtime_wall_clock_started_at_monotonic"] = None
        updates["runtime_wall_clock_base_seconds"] = current_seconds
        return updates

    generated_turns = runtime_generated_turn_count_for_display(status)
    playback_completed_turns = runtime_status_playback_completed_turn_count(status)
    has_pending_playback = (
        phase == "completed"
        and playback_completed_turns is not None
        and playback_completed_turns < generated_turns
    )
    should_continue_clock = not playback_paused and (
        phase == "running"
        or (phase == "paused" and continue_when_paused)
        or has_pending_playback
    )
    if should_continue_clock:
        if started_at is None:
            started_at = now_monotonic
            base_seconds = current_seconds
        updates["runtime_wall_clock_started_at_monotonic"] = started_at
        updates["runtime_wall_clock_base_seconds"] = base_seconds
        updates["wall_clock_seconds"] = base_seconds + max(0.0, now_monotonic - started_at)
        return updates

    if started_at is not None:
        current_seconds = base_seconds + max(0.0, now_monotonic - started_at)
    updates["wall_clock_seconds"] = current_seconds
    updates["runtime_wall_clock_started_at_monotonic"] = None
    updates["runtime_wall_clock_base_seconds"] = current_seconds
    return updates


def runtime_status_session_updates(status: dict[str, Any]) -> dict[str, Any]:
    phase = str(status.get("phase") or "")
    status_by_phase = {
        "idle": SessionStatus.IDLE.value,
        "running": SessionStatus.RUNNING.value,
        "paused": SessionStatus.PAUSED.value,
        "completed": SessionStatus.COMPLETED.value,
        "stopped": SessionStatus.STOPPED.value,
        "error": SessionStatus.ERROR.value,
    }
    updates: dict[str, Any] = {}
    if phase in status_by_phase:
        updates["session_status"] = status_by_phase[phase]
    return updates


def runtime_start_options_for_generation_mode(mode: str) -> dict[str, Any]:
    if mode == "responses_tts":
        return {"realtime_api_centered_mode": False}
    return {"realtime_api_centered_mode": True}


def uploaded_prompt_file_text_and_signature(uploaded_file: Any) -> tuple[str, str]:
    raw_value = uploaded_file.getvalue()
    if isinstance(raw_value, bytes):
        raw_bytes = raw_value
    elif isinstance(raw_value, bytearray):
        raw_bytes = bytes(raw_value)
    else:
        raise TypeError("uploaded prompt file must provide bytes")
    text = raw_bytes.decode("utf-8-sig")
    filename = str(getattr(uploaded_file, "name", "uploaded_prompt"))
    size = getattr(uploaded_file, "size", len(raw_bytes))
    digest = hashlib.sha256(raw_bytes).hexdigest()
    return text, f"{filename}:{size}:{digest}"


def apply_uploaded_prompt_file_to_session(
    uploaded_file: Any | None,
    *,
    prompt_text_key: str,
    source_profile_key: str,
    file_signature_key: str,
    source_profile_id: str,
) -> str | None:
    if uploaded_file is None:
        st.session_state[file_signature_key] = None
        return None
    try:
        text, file_signature = uploaded_prompt_file_text_and_signature(uploaded_file)
    except UnicodeDecodeError:
        return "UTF-8 の .md / .txt として読み込めませんでした。"
    signature = f"{source_profile_id}:{file_signature}"
    if st.session_state.get(file_signature_key) != signature:
        st.session_state[prompt_text_key] = text
        st.session_state[source_profile_key] = source_profile_id
        st.session_state[file_signature_key] = signature
    return None


def render_prompt_file_loader(
    *,
    label: str,
    upload_key: str,
    prompt_text_key: str,
    source_profile_key: str,
    file_signature_key: str,
    source_profile_id: str,
) -> None:
    uploaded_file = st.file_uploader(
        label,
        type=list(PROMPT_FILE_UPLOAD_TYPES),
        key=upload_key,
    )
    error = apply_uploaded_prompt_file_to_session(
        uploaded_file,
        prompt_text_key=prompt_text_key,
        source_profile_key=source_profile_key,
        file_signature_key=file_signature_key,
        source_profile_id=source_profile_id,
    )
    if error:
        st.error(error)


def two_client_attachment_defaults() -> dict[str, str]:
    preset = load_client_preset(DEFAULT_CLIENT_PRESET_PATH)
    wife_id, husband_id = preset.modes.two_clients
    wife = preset.participants[wife_id]
    husband = preset.participants[husband_id]
    return {
        "common_profile": preset.shared.public_profile,
        "common_prompt": preset.shared.prompt,
        "client_a_profile": wife.private_profile,
        "client_a_prompt": wife.prompt,
        "client_a_initial_transcript": wife.initial_transcript,
        "client_b_profile": husband.private_profile,
        "client_b_prompt": husband.prompt,
        "client_b_initial_transcript": husband.initial_transcript,
    }


def _opening_transcript_from_prompt(prompt: str) -> str:
    marker = "面談の冒頭で自由に話すよう促された場合は"
    if marker not in prompt:
        return ""
    after_marker = prompt.split(marker, 1)[1]
    start = after_marker.find("「")
    end = after_marker.find("」", start + 1)
    if start == -1 or end == -1:
        return ""
    return after_marker[start + 1 : end].strip()


def _join_two_client_default_parts(common_text: str, role_text: str) -> str:
    return "\n\n".join(
        part.strip() for part in (common_text, role_text) if part.strip()
    )


def _participant_common_prompt(common_prompt: str, intro: str) -> str:
    return common_prompt.replace(TWO_CLIENT_COMMON_PROMPT_INTRO, intro, 1)


def runtime_counselor_prompt_for_ui(prompt: Any) -> str:
    prompt_text = prompt.strip() if isinstance(prompt, str) else ""
    if not prompt_text or prompt_text == LEGACY_RUNTIME_COUNSELOR_SYSTEM_PROMPT:
        return DEFAULT_RUNTIME_COUNSELOR_PROMPT
    return prompt_text


def runtime_human_input_enabled() -> bool:
    status = st.session_state.get("runtime_control_status") or {}
    mode = st.session_state.get("selected_mode")
    if status.get("phase") in {"running", "paused"}:
        mode = status.get("interaction_mode") or mode
    return mode in {"human_counselor_ai_client", "ai_counselor_human_client"}


def runtime_start_options_for_ui(
    *,
    counselor_profile: Profile,
    client_profile: Profile,
    selected_mode: str | None = None,
) -> dict[str, Any]:
    options: dict[str, Any] = {"realtime_api_centered_mode": True}
    mode = str(
        selected_mode
        or st.session_state.get("selected_mode")
        or "ai_counselor_ai_client"
    ).strip()
    human_counselor_mode = mode == "human_counselor_ai_client"
    human_client_mode = mode == "ai_counselor_human_client"
    participant_mode = str(
        st.session_state.get("runtime_participant_mode")
        or DEFAULT_RUNTIME_PARTICIPANT_MODE
    ).strip()
    if human_client_mode:
        participant_mode = "one_client"
    if human_counselor_mode:
        options.update(
            {
                "interaction_mode": "human_counselor_ai_client",
                "participant_mode": participant_mode,
                "human_input_mode": "push_to_talk",
                "human_stt_submit_policy": "auto_on_final",
                "human_interrupts_enabled": True,
            }
        )
    counselor_prompt = runtime_counselor_prompt_for_ui(
        st.session_state.get("counselor_prompt_text")
    )
    client_prompt = str(
        st.session_state.get("client_prompt_text")
        or DEFAULT_RUNTIME_CLIENT_PROMPT
    ).strip()
    counselor_public_profile = str(
        st.session_state.get("counselor_public_profile_text")
        or counselor_profile.public_profile
        or ""
    ).strip()
    client_public_profile = str(
        st.session_state.get("client_public_profile_text")
        or client_profile.public_profile
        or ""
    ).strip()
    client_private_profile = str(
        st.session_state.get("client_private_profile_text")
        or client_profile.hidden_background
        or ""
    ).strip()
    initial_client_transcript = str(
        st.session_state.get("runtime_initial_client_transcript") or ""
    ).strip()
    counselor_display_name = str(
        st.session_state.get("counselor_display_name")
        or DEFAULT_RUNTIME_COUNSELOR_DISPLAY_NAME
    ).strip()
    client_display_name = str(
        st.session_state.get("client_display_name")
        or DEFAULT_RUNTIME_CLIENT_DISPLAY_NAME
    ).strip()
    realtime_model = str(
        st.session_state.get("runtime_realtime_model")
        or DEFAULT_RUNTIME_REALTIME_MODEL
    ).strip()
    prompting_llm_model = str(
        st.session_state.get("runtime_prompting_llm_model")
        or DEFAULT_RUNTIME_PROMPTING_LLM_MODEL
    ).strip()
    prompting_llm_reasoning_effort = str(
        st.session_state.get("runtime_prompting_llm_reasoning_effort")
        or DEFAULT_RUNTIME_PROMPTING_REASONING_EFFORT
    ).strip()
    timing_llm_model = str(
        st.session_state.get("runtime_timing_llm_model")
        or DEFAULT_RUNTIME_TIMING_LLM_MODEL
    ).strip()
    timing_llm_reasoning_effort = str(
        st.session_state.get("runtime_timing_llm_reasoning_effort")
        or DEFAULT_RUNTIME_TIMING_REASONING_EFFORT
    ).strip()
    summary_llm_model = str(
        st.session_state.get("runtime_summary_llm_model")
        or DEFAULT_RUNTIME_SUMMARY_LLM_MODEL
    ).strip()
    summary_llm_reasoning_effort = str(
        st.session_state.get("runtime_summary_llm_reasoning_effort")
        or DEFAULT_RUNTIME_SUMMARY_REASONING_EFFORT
    ).strip()
    speaker_selection_policy = str(
        st.session_state.get("runtime_speaker_selection_policy")
        or "turn_boundary_timing"
    ).strip()
    counselor_tts_voice = str(
        st.session_state.get("runtime_counselor_tts_voice") or ""
    ).strip()
    client_tts_voice = str(
        st.session_state.get("runtime_client_tts_voice") or ""
    ).strip()
    closing_start_seconds = st.session_state.get("runtime_closing_start_seconds")
    force_stop_after_closing_turns = runtime_force_stop_after_closing_turns_for_mode(
        participant_mode
    )
    output_speeds: dict[str, float] = {}
    for speaker, key in [
        ("counselor", "runtime_counselor_output_speed"),
        ("client", "runtime_client_output_speed"),
    ]:
        value = st.session_state.get(key)
        if (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and MIN_REALTIME_OUTPUT_SPEED <= value <= MAX_REALTIME_OUTPUT_SPEED
        ):
            output_speeds[speaker] = float(value)
    speaker_gains: dict[str, float] = {}
    for speaker, key in [
        ("counselor", "runtime_counselor_audio_gain"),
        ("client", "runtime_client_audio_gain"),
    ]:
        value = st.session_state.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0:
            speaker_gains[speaker] = float(value)
    realtime_provider = st.session_state.get("runtime_realtime_provider")
    if realtime_provider:
        options["ai_routes"] = {
            "realtime_speech": {
                "provider": realtime_provider,
                "model_ref": str(
                    st.session_state.get(
                        f"runtime_realtime_target_{realtime_provider}", ""
                    )
                ).strip(),
            },
        }
    elif realtime_model:
        options["realtime_model"] = realtime_model
    stt_provider = st.session_state.get("runtime_stt_provider")
    if stt_provider:
        options.setdefault("ai_routes", {})["realtime_transcription"] = {
            "provider": stt_provider,
            "model_ref": str(
                st.session_state.get(f"runtime_stt_target_{stt_provider}", "")
            ).strip(),
        }
    if prompting_llm_model in RUNTIME_PROMPTING_LLM_MODEL_OPTIONS:
        options["prompting_llm_model"] = prompting_llm_model
    if (
        prompting_llm_reasoning_effort
        in RUNTIME_PROMPTING_REASONING_EFFORT_OPTIONS
    ):
        options["prompting_llm_reasoning_effort"] = (
            prompting_llm_reasoning_effort
        )
    if timing_llm_model in RUNTIME_TIMING_LLM_MODEL_OPTIONS:
        options["timing_llm_model"] = timing_llm_model
    if timing_llm_reasoning_effort in RUNTIME_TIMING_REASONING_EFFORT_OPTIONS:
        options["timing_llm_reasoning_effort"] = timing_llm_reasoning_effort
    if summary_llm_model in RUNTIME_SUMMARY_LLM_MODEL_OPTIONS:
        options["summary_llm_model"] = summary_llm_model
    if summary_llm_reasoning_effort in RUNTIME_SUMMARY_REASONING_EFFORT_OPTIONS:
        options["summary_llm_reasoning_effort"] = summary_llm_reasoning_effort
    for name, (legacy_prefix, _) in RUNTIME_TEXT_ROUTE_CONTROLS.items():
        provider_id = st.session_state.get(f"runtime_{name}_provider")
        if provider_id:
            options.setdefault("ai_routes", {})[name] = {
                "provider": provider_id,
                "model_ref": str(
                    st.session_state.get(f"runtime_{name}_target_{provider_id}", "")
                ).strip(),
            }
            options.pop(f"{legacy_prefix}_llm_model", None)
    if speaker_selection_policy in RUNTIME_SPEAKER_SELECTION_POLICY_OPTIONS:
        options["speaker_selection_policy"] = speaker_selection_policy
    if human_client_mode:
        options.update(
            interaction_mode=mode,
            participant_mode="one_client",
            speaker_selection_policy="fixed_round_robin",
            fixed_speaker_sequence=["counselor", "client"],
            initial_client_transcript="",
            human_input_mode="push_to_talk",
            human_stt_submit_policy="auto_on_final",
            human_interrupts_enabled=True,
            shared_case=str(
                st.session_state.get("human_client_shared_information") or ""
            ).strip(),
            counselor_prompt=counselor_prompt,
            participants={
                "counselor": {
                    "role": "counselor",
                    "actor_kind": "ai",
                    "display_name": counselor_display_name,
                    "public_profile_source": counselor_public_profile,
                },
                "client": {
                    "role": "client",
                    "actor_kind": "human",
                    "display_name": str(
                        st.session_state.get("human_client_display_name")
                        or "クライアント"
                    ).strip(),
                },
            },
        )
        speaker_gains.pop("client", None)
        options.pop("timing_llm_model", None)
        options.pop("timing_llm_reasoning_effort", None)
        options.get("ai_routes", {}).pop("turn_timing", None)
    elif participant_mode == "two_clients":
        speaker_gains.pop("client", None)
        participants = _runtime_two_client_participants_for_ui(
            counselor_profile=counselor_profile,
            client_profile=client_profile,
            counselor_prompt=counselor_prompt,
            counselor_public_profile=counselor_public_profile,
            counselor_display_name=counselor_display_name,
        )
        if human_counselor_mode:
            participants.pop("counselor", None)
        if participants:
            options["participants"] = participants
            if client_public_profile:
                options["shared_case"] = client_public_profile
            options["fixed_speaker_sequence"] = list(
                DEFAULT_RUNTIME_TWO_CLIENT_SPEAKER_SEQUENCE
            )
        speaker_gains.update(_runtime_two_client_speaker_gains_for_ui())
    else:
        if counselor_prompt and not human_counselor_mode:
            options["counselor_prompt"] = counselor_prompt
        if client_prompt:
            options["client_prompt"] = client_prompt
        if initial_client_transcript and not human_counselor_mode:
            options["initial_client_transcript"] = initial_client_transcript
        if counselor_display_name and not human_counselor_mode:
            options["counselor_display_name"] = counselor_display_name
        if client_display_name:
            options["client_display_name"] = client_display_name
        if client_tts_voice:
            options["client_tts_voice"] = client_tts_voice
        if "client" in output_speeds:
            options["client_realtime_output_speed"] = output_speeds["client"]
        if (
            (counselor_public_profile and not human_counselor_mode)
            or client_public_profile
            or client_private_profile
            or human_counselor_mode
        ):
            participants = _runtime_one_client_participants_for_ui(
                counselor_display_name=counselor_display_name,
                client_display_name=client_display_name,
                counselor_public_profile=counselor_public_profile,
                client_public_profile=client_public_profile,
                client_private_profile=client_private_profile,
            )
            if human_counselor_mode:
                participants.pop("counselor", None)
            options["participants"] = participants
    if counselor_tts_voice and not human_counselor_mode:
        options["counselor_tts_voice"] = counselor_tts_voice
    if "counselor" in output_speeds and not human_counselor_mode:
        options["counselor_realtime_output_speed"] = output_speeds["counselor"]
    if isinstance(closing_start_seconds, (int, float)):
        options["closing_start_elapsed_seconds"] = float(closing_start_seconds)
    if (
        isinstance(force_stop_after_closing_turns, int)
        and force_stop_after_closing_turns > 0
    ):
        options["force_stop_after_closing_turns"] = force_stop_after_closing_turns
    if human_counselor_mode:
        speaker_gains.pop("counselor", None)
    if speaker_gains:
        options["speaker_gains"] = speaker_gains
    return options


def _runtime_one_client_participants_for_ui(
    *,
    counselor_display_name: str,
    client_display_name: str,
    counselor_public_profile: str,
    client_public_profile: str,
    client_private_profile: str,
) -> dict[str, dict[str, Any]]:
    participants: dict[str, dict[str, Any]] = {
        "counselor": {
            "role": "counselor",
            "display_name": counselor_display_name
            or DEFAULT_RUNTIME_COUNSELOR_DISPLAY_NAME,
        },
        "client": {
            "role": "client",
            "display_name": client_display_name or DEFAULT_RUNTIME_CLIENT_DISPLAY_NAME,
        },
    }
    _set_optional_text(
        participants["counselor"],
        "public_profile_source",
        counselor_public_profile,
    )
    _set_optional_text(
        participants["client"],
        "public_profile_source",
        client_public_profile,
    )
    _set_optional_text(
        participants["client"],
        "private_profile_source",
        client_private_profile,
    )
    return participants


def _runtime_two_client_participants_for_ui(
    *,
    counselor_profile: Profile,
    client_profile: Profile,
    counselor_prompt: str,
    counselor_public_profile: str,
    counselor_display_name: str,
) -> dict[str, dict[str, Any]]:
    participants: dict[str, dict[str, Any]] = {
        "counselor": {
            "role": "counselor",
            "display_name": counselor_display_name
            or DEFAULT_RUNTIME_COUNSELOR_DISPLAY_NAME,
        },
        "client_a": {
            "role": "client",
            "display_name": _runtime_state_text(
                "client_a_display_name",
                DEFAULT_RUNTIME_CLIENT_A_DISPLAY_NAME,
            ),
        },
        "client_b": {
            "role": "client",
            "display_name": _runtime_state_text(
                "client_b_display_name",
                DEFAULT_RUNTIME_CLIENT_B_DISPLAY_NAME,
            ),
        },
    }
    _set_optional_text(participants["counselor"], "prompt_source", counselor_prompt)
    _set_optional_text(
        participants["counselor"],
        "public_profile_source",
        counselor_public_profile,
    )
    _set_optional_text(
        participants["client_a"],
        "prompt_source",
        _runtime_state_text("client_a_prompt_text", DEFAULT_RUNTIME_CLIENT_PROMPT),
    )
    _set_optional_text(
        participants["client_b"],
        "prompt_source",
        _runtime_state_text("client_b_prompt_text", DEFAULT_RUNTIME_CLIENT_PROMPT),
    )
    _set_optional_text(
        participants["client_a"],
        "private_profile_source",
        _runtime_state_text(
            "client_a_public_profile_text",
            "",
        ),
    )
    _set_optional_text(
        participants["client_b"],
        "private_profile_source",
        _runtime_state_text(
            "client_b_public_profile_text",
            "",
        ),
    )
    _set_optional_text(
        participants["client_a"],
        "initial_transcript",
        _runtime_state_text("runtime_initial_client_a_transcript", ""),
    )
    _set_optional_text(
        participants["client_b"],
        "initial_transcript",
        _runtime_state_text("runtime_initial_client_b_transcript", ""),
    )
    for participant_id, key in [
        ("counselor", "runtime_counselor_tts_voice"),
        ("client_a", "runtime_client_a_tts_voice"),
        ("client_b", "runtime_client_b_tts_voice"),
    ]:
        _set_optional_text(
            participants[participant_id],
            "voice",
            _runtime_state_text(key, ""),
        )
    for participant_id, key in [
        ("counselor", "runtime_counselor_output_speed"),
        ("client_a", "runtime_client_a_output_speed"),
        ("client_b", "runtime_client_b_output_speed"),
    ]:
        value = st.session_state.get(key)
        if (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and MIN_REALTIME_OUTPUT_SPEED <= value <= MAX_REALTIME_OUTPUT_SPEED
        ):
            participants[participant_id]["realtime_output_speed"] = float(value)
    return participants


def _runtime_two_client_speaker_gains_for_ui() -> dict[str, float]:
    gains: dict[str, float] = {}
    for speaker, key in [
        ("client_a", "runtime_client_a_audio_gain"),
        ("client_b", "runtime_client_b_audio_gain"),
    ]:
        value = st.session_state.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0:
            gains[speaker] = float(value)
    return gains


def runtime_audio_settings_rows_from_start_options(
    options: dict[str, Any],
) -> list[dict[str, Any]]:
    raw_participants = options.get("participants")
    participants = raw_participants if isinstance(raw_participants, dict) else {}
    raw_gains = options.get("speaker_gains")
    gains = raw_gains if isinstance(raw_gains, dict) else {}
    speaker_ids = list(participants)
    if not speaker_ids:
        speaker_ids = [
            speaker_id
            for speaker_id in ("counselor", "client", "client_a", "client_b")
            if speaker_id in gains
        ]

    rows: list[dict[str, Any]] = []
    for speaker_id in speaker_ids:
        raw_participant = participants.get(speaker_id)
        participant = raw_participant if isinstance(raw_participant, dict) else {}
        voice = participant.get("voice")
        output_speed = participant.get("realtime_output_speed")
        if speaker_id == "counselor":
            voice = voice or options.get("counselor_tts_voice")
            output_speed = (
                output_speed
                if output_speed is not None
                else options.get("counselor_realtime_output_speed")
            )
        elif speaker_id == "client":
            voice = voice or options.get("client_tts_voice")
            output_speed = (
                output_speed
                if output_speed is not None
                else options.get("client_realtime_output_speed")
            )
        rows.append(
            {
                "話者ID": speaker_id,
                "表示名": str(participant.get("display_name") or speaker_id),
                "音声": voice,
                "話速": output_speed,
                "音量": gains.get(speaker_id),
            }
        )
    return rows


def runtime_resolved_audio_settings_rows(
    events: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    for event in reversed(events):
        if not isinstance(event, dict):
            continue
        if event.get("event_type") != "runtime_audio_settings_resolved":
            continue
        details = event.get("details")
        if not isinstance(details, dict):
            return []
        speakers = details.get("speakers")
        if not isinstance(speakers, list):
            return []
        return [
            {
                "話者ID": str(speaker.get("speaker_id") or ""),
                "表示名": str(
                    speaker.get("display_name")
                    or speaker.get("speaker_id")
                    or ""
                ),
                "音声": speaker.get("voice"),
                "話速": speaker.get("realtime_output_speed"),
                "音量": speaker.get("audio_gain"),
            }
            for speaker in speakers
            if isinstance(speaker, dict)
        ]
    return []


def _runtime_state_text(key: str, fallback: Any) -> str:
    return str(st.session_state.get(key) or fallback or "").strip()


def _set_optional_text(target: dict[str, Any], key: str, value: str) -> None:
    clean_value = str(value or "").strip()
    if clean_value:
        target[key] = clean_value


def refresh_runtime_control_events(
    *,
    client: RuntimeControlClient,
    session_id: str,
) -> dict[str, Any]:
    if not session_id:
        return {"runtime_control_error": "イベント取得には session_id が必要です。"}
    try:
        events = client.events(session_id)
        response_instructions = client.response_instructions(session_id)
    except RuntimeObserverClientError as exc:
        return {"runtime_control_error": str(exc)}
    return {
        "runtime_control_events": events,
        "runtime_response_instructions": response_instructions,
        "runtime_control_error": "",
    }


def runtime_status_rows(status: dict[str, Any]) -> list[dict[str, Any]]:
    fields = [
        "session_id",
        "phase",
        "pause_reason",
        "current_turn_id",
        "current_speaker",
        "completed_turns",
        "last_event_type",
        "error_message",
    ]
    return [
        {"項目": field, "値": "" if status.get(field) is None else str(status.get(field))}
        for field in fields
    ]


def runtime_deferred_interaction_feature_rows() -> list[dict[str, str]]:
    return [
        {
            "機能": "動的ターンテイク",
            "状態": "発話後の各Agent判断",
            "次段階": "応答遅延・発話量チューニング",
        },
        {
            "機能": "譲り合い",
            "状態": "YIELD/CONTINUE調停",
            "次段階": "重なり時の話者選択チューニング",
        },
        {
            "機能": "割り込み",
            "状態": "停止/truncate接続実装",
            "次段階": "端末別レイテンシ検証",
        },
        {
            "機能": "重なり",
            "状態": "grace + 片側停止",
            "次段階": "譲る話者の選択精度調整",
        },
    ]


def apply_browser_audio_event_for_ui(
    session: SessionState,
    event_value: Any,
    *,
    queue_version: str,
    processed_event_ids: list[str] | set[str] | tuple[str, ...],
    conversation_engine: ConversationEngine | None,
    cumulative_audio_seconds: float,
    tts_client: object | None,
    log_paths: SessionLogPaths | None,
    response_format: str,
    selected_turn_id: int | None,
    playback_gap_seconds: float = AUTO_PLAYBACK_DEFAULT_GAP_SECONDS,
) -> dict[str, Any]:
    event = parse_browser_audio_event(event_value)
    if event is None or event_was_processed(event, processed_event_ids):
        return {}

    updates: dict[str, Any] = {
        "audio_player_processed_event_ids": remember_processed_event_id(
            processed_event_ids,
            event.event_id,
        ),
        "audio_player_last_event": {
            "event_id": event.event_id,
            "event_type": event.event_type,
            "turn_id": event.turn_id,
            "revision_id": event.revision_id,
            "queue_version": event.queue_version,
        },
    }
    turn = active_turn_for_browser_event(session, event)
    if turn is None:
        updates["warnings"] = [
            f"古い音声イベントを無視しました: turn_id={event.turn_id}"
        ]
        return updates
    decision = next_playback_decision(session)
    if decision.turn is None or decision.turn.turn_id != event.turn_id:
        updates["warnings"] = [
            f"再生順序外の音声イベントを無視しました: turn_id={event.turn_id}"
        ]
        return updates
    updates["audio_player_continuous_mode"] = event.event_type in {
        "playback_ended",
        "playback_skipped",
    }

    if event.event_type == "playback_ended":
        if conversation_engine is None:
            updates["warnings"] = ["再生完了を反映するライブセッションがありません。"]
            return updates
        played_seconds = event.played_seconds or event.duration_seconds or event.current_seconds
        updates.update(
            mark_current_audio_played_for_ui(
                session,
                conversation_engine=conversation_engine,
                cumulative_audio_seconds=cumulative_audio_seconds,
                tts_client=tts_client,
                log_paths=log_paths,
                response_format=response_format,
                selected_turn_id=selected_turn_id,
                duration_seconds=played_seconds,
            )
        )
        updates.update(
            schedule_next_audio_playback_updates(
                session,
                start_delay_seconds=playback_gap_seconds,
            )
        )
        return updates

    if event.event_type == "playback_skipped":
        if conversation_engine is None:
            updates["warnings"] = ["音声スキップを反映するライブセッションがありません。"]
            return updates
        played_seconds = event.played_seconds or event.current_seconds
        duration_seconds = event.duration_seconds or played_seconds
        updated_total_seconds = record_playback_terminal_event(
            turn,
            PlaybackEvent(
                event_name="skip",
                current_seconds=played_seconds,
                duration_seconds=duration_seconds,
            ),
            previous_total_seconds=cumulative_audio_seconds,
            already_counted=False,
            log_paths=log_paths,
        )
        warnings = [f"Turn {turn.turn_id} をスキップしました。"]
        warnings.extend(
            refill_generation_queue_for_ui(
                session,
                conversation_engine=conversation_engine,
                cumulative_audio_seconds=updated_total_seconds,
                tts_client=tts_client,
                log_paths=log_paths,
                response_format=response_format,
            )
        )
        updates.update(
            live_session_ui_updates(
                session,
                selected_turn_id=selected_turn_id,
                warnings=warnings,
            )
        )
        updates["cumulative_audio_seconds"] = updated_total_seconds
        updates.update(
            schedule_next_audio_playback_updates(
                session,
                start_delay_seconds=playback_gap_seconds,
            )
        )
        return updates

    if event.event_type == "playback_error":
        updates["warnings"] = [
            f"音声再生エラー: Turn {turn.turn_id} {event.error_message}".strip()
        ]
        return updates

    return updates


def active_turn_for_browser_event(
    session: SessionState,
    event: BrowserAudioEvent,
) -> Turn | None:
    for turn in session.turns:
        if turn.turn_id != event.turn_id:
            continue
        if turn.active_revision_id != event.revision_id:
            return None
        if turn.status in {
            TurnStatus.INVALIDATED,
            TurnStatus.IGNORED_AFTER_STOP,
            TurnStatus.ERROR,
            TurnStatus.PLAYED,
            TurnStatus.SKIPPED_PARTIAL,
            TurnStatus.SKIPPED_UNPLAYED,
        }:
            return None
        return turn
    return None


def create_openai_live_clients() -> dict[str, object]:
    from openai import OpenAI

    client = OpenAI()
    return {
        "text_client": OpenAITextClient(client=client),
        "moderation_client": OpenAIModerationClient(client=client),
        "clinical_ng_checker": ClinicalNgChecker(client=client),
        "tts_client": OpenAITtsClient(client=client),
    }


def build_conversation_engine_for_ui(
    *,
    config: Any,
    text_client: object,
    counselor_profile: Profile,
    client_profile: Profile,
    theme: Theme,
    moderation_client: object | None,
    clinical_ng_checker: object | None,
    log_paths: SessionLogPaths | None,
    ahead_generation_turns: int | None = None,
) -> ConversationEngine:
    safety_runner = None
    if moderation_client is not None or clinical_ng_checker is not None:
        safety_runner = SafetyCheckRunner(
            moderation_client=moderation_client,
            clinical_ng_checker=clinical_ng_checker,
            config=SafetyCheckConfig(
                moderation_model=config.openai.moderation_model,
                clinical_model=config.openai.default_text_model,
            ),
            log_paths=log_paths,
        )

    return ConversationEngine(
        text_client=text_client,
        participants=ConversationParticipants(
            counselor_profile=counselor_profile,
            client_profile=client_profile,
            theme=theme,
        ),
        config=ConversationEngineConfig(
            ahead_generation_turns=(
                config.app.ahead_generation_turns
                if ahead_generation_turns is None
                else int(ahead_generation_turns)
            ),
            closing_start_audio_seconds=config.app.closing_start_audio_seconds,
            farewell_after_turns=config.app.farewell_after_turns,
            closing_keep_audio_ready_turns=config.session.closing_keep_audio_ready_turns,
            reasoning_effort=config.openai.default_text_reasoning_effort,
        ),
        safety_check_runner=safety_runner,
    )


def refill_generation_queue_for_ui(
    session: SessionState,
    *,
    conversation_engine: ConversationEngine,
    cumulative_audio_seconds: float,
    tts_client: object | None,
    log_paths: SessionLogPaths | None,
    response_format: str,
    active_playback_turn_id: int | None = None,
    suppress_success_warnings: bool = False,
) -> list[str]:
    generated = conversation_engine.fill_generation_queue(
        session,
        cumulative_audio_seconds=cumulative_audio_seconds,
        exclude_turn_ids_from_buffer=(
            {active_playback_turn_id} if active_playback_turn_id is not None else None
        ),
    )
    warnings: list[str] = []
    if generated:
        warnings.append(f"先行生成ターン {len(generated)} 件を生成しました。")
    warnings.extend(
        synthesize_pending_audio_for_ui(
            session,
            conversation_engine=conversation_engine,
            tts_client=tts_client,
            log_paths=log_paths,
            response_format=response_format,
        )
    )
    if suppress_success_warnings:
        return [warning for warning in warnings if "失敗しました" in warning]
    return warnings


def synthesize_pending_audio_for_ui(
    session: SessionState,
    *,
    conversation_engine: ConversationEngine,
    tts_client: object | None,
    log_paths: SessionLogPaths | None,
    response_format: str,
) -> list[str]:
    if tts_client is None or log_paths is None:
        return []

    warnings: list[str] = []
    while True:
        turn = next_tts_turn(session)
        if turn is None:
            return warnings
        profile = conversation_engine.speaker_profile_for_role(turn.speaker_role)
        try:
            tts_client.synthesize_turn(
                TtsRequest(
                    turn=turn,
                    log_paths=log_paths,
                    model=profile.tts_model,
                    voice=profile.tts_voice,
                    instructions=profile.tts_instructions,
                    response_format=response_format,
                )
            )
        except Exception as exc:
            warnings.append(f"Turn {turn.turn_id} の音声生成に失敗しました: {exc}")
            return warnings
        mark_tts_ready(turn)
        warnings.append(f"Turn {turn.turn_id} の音声を生成しました。")


def live_session_ui_updates(
    session: SessionState,
    *,
    selected_turn_id: int | None,
    warnings: list[str],
    conversation_engine: ConversationEngine | None = None,
    log_paths: SessionLogPaths | None = None,
    text_client: object | None = None,
    moderation_client: object | None = None,
    clinical_ng_checker: object | None = None,
    tts_client: object | None = None,
    moderation_model: str | None = None,
    clinical_model: str | None = None,
) -> dict[str, Any]:
    unplayed_turns = unplayed_turn_rows_from_session(session)
    selected_turn = selected_unplayed_turn(unplayed_turns, selected_turn_id)
    updates: dict[str, Any] = {
        "session_status": session.status.value,
        "conversation_phase": session.conversation_phase.value,
        "session_model": session,
        "conversation_log": conversation_log_rows_from_session(session),
        "unplayed_turns": unplayed_turns,
        "selected_unplayed_turn_id": (
            selected_turn["turn_id"] if selected_turn is not None else None
        ),
        "edit_turn_text": selected_turn["text"] if selected_turn is not None else "",
        "edit_turn_text_source_turn_id": (
            selected_turn["turn_id"] if selected_turn is not None else None
        ),
        "warnings": warnings,
    }
    optional_updates = {
        "conversation_engine": conversation_engine,
        "log_paths": log_paths,
        "text_client": text_client,
        "moderation_client": moderation_client,
        "clinical_ng_checker": clinical_ng_checker,
        "tts_client": tts_client,
        "moderation_model": moderation_model,
        "clinical_model": clinical_model,
    }
    updates.update(
        {key: value for key, value in optional_updates.items() if value is not None}
    )
    return updates


def conversation_log_rows_from_session(session: SessionState) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    cumulative_seconds = 0.0
    for turn in sorted(session.turns, key=lambda item: item.turn_id):
        if turn.status != TurnStatus.PLAYED:
            continue
        revision = turn.active_revision()
        cumulative_seconds += revision.actual_played_seconds
        rows.append(
            {
                "turn_id": turn.turn_id,
                "speaker_name": turn.speaker_name,
                "speaker_role": turn.speaker_role.value,
                "text": revision.canonical_text,
                "spoken_at": revision.spoken_at.isoformat()
                if revision.spoken_at is not None
                else "-",
                "cumulative_audio_seconds": cumulative_seconds,
            }
        )
    return rows


def set_session_status(status: SessionStatus) -> None:
    st.session_state["session_status"] = status.value
    if isinstance(st.session_state.get("session_model"), SessionState):
        st.session_state["session_model"].status = status
    if status != SessionStatus.RUNNING:
        clear_auto_playback_session_state()


def format_seconds(seconds: float) -> str:
    total_seconds = max(0, int(seconds))
    minutes, remaining_seconds = divmod(total_seconds, 60)
    return f"{minutes:02d}:{remaining_seconds:02d}"


def runtime_remaining_metric(
    *,
    status: dict[str, Any],
    wall_clock_seconds: float,
    closing_start_seconds: float,
    force_stop_after_closing_turns: int,
) -> tuple[str, str]:
    closing_started_turn_id = status.get("closing_started_turn_id")
    closing_count_started_turn_id = status.get("closing_count_started_turn_id")
    if (
        closing_started_turn_id is not None
        or closing_count_started_turn_id is not None
    ):
        completed_closing_turns = 0
        if closing_count_started_turn_id is not None:
            current_turn_id = int(status.get("current_turn_id") or 0)
            completed_closing_turns = max(
                current_turn_id - int(closing_count_started_turn_id) + 1,
                0,
            )
        remaining_turns = max(
            int(force_stop_after_closing_turns) - completed_closing_turns,
            0,
        )
        return "終了まで", f"{remaining_turns}ターン"
    remaining_seconds = max(float(closing_start_seconds) - wall_clock_seconds, 0.0)
    return "クロージングまで", format_seconds(remaining_seconds)


def control_disabled_states(
    status: SessionStatus, *, mode_enabled: bool
) -> dict[str, bool]:
    return {
        "start": not mode_enabled or status == SessionStatus.RUNNING,
        "pause": status != SessionStatus.RUNNING,
        "resume": status != SessionStatus.PAUSED,
        "stop": status in {SessionStatus.IDLE, SessionStatus.STOPPED},
        "skip": status != SessionStatus.RUNNING,
    }


def unplayed_preview_rows(unplayed_turns: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "turn_id": turn["turn_id"],
            "revision_id": turn["revision_id"],
            "話者": turn["speaker_name"],
            "役割": turn["speaker_role"],
            "発話": turn["text"],
            "音声化": turn["audio_status"],
            "再生": turn["playback_status"],
            "編集済み": turn["edited"],
            "警告": turn["warning_level"],
            "hold理由": turn["hold_reason"],
        }
        for turn in unplayed_turns
    ]


def transcript_rows_for_display(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not records:
        return [
            {
                "turn_id": "",
                "話者": "",
                "役割": "",
                "発話": "",
                "再生時刻": "",
            }
        ]
    return [
        {
            "turn_id": record.get("turn_id", ""),
            "話者": record["speaker_name"],
            "役割": record["speaker_role"],
            "発話": record["text"],
            "再生時刻": record["spoken_at"],
        }
        for record in records
    ]


def runtime_sessions_root() -> Path:
    try:
        settings = load_runtime_config(ROOT_DIR / "config" / "runtime_config.yaml")
    except RuntimeConfigLoadError:
        return ROOT_DIR / "results" / "runtime_sessions"
    return ROOT_DIR / settings.paths.runtime_sessions_dir


def saved_session_replay_ready(session_dir: Path | str) -> bool:
    session_path = Path(session_dir)
    turns = load_public_transcript_turns(session_path)
    if not turns or session_realtime_audio_path(session_path) is None:
        return False
    human_audio_turns = [
        turn for turn in turns if turn.transcript_type == "human_final"
    ]
    return all(turn.audio_path is not None for turn in human_audio_turns)


def pending_saved_replay_session_id(
    *,
    sessions_root: Path | str,
    sessions: list[Path],
    current_session_id: str,
    runtime_status: dict[str, Any],
) -> str:
    session_id = current_session_id.strip()
    if not session_id:
        return ""
    phase = str(runtime_status.get("phase") or "").strip().lower()
    if not phase or phase == "idle":
        return ""
    status_session_id = str(runtime_status.get("session_id") or "").strip()
    if status_session_id and status_session_id != session_id:
        return ""

    session_dir = Path(sessions_root) / session_id
    session_paths = {Path(path) for path in sessions}
    if session_dir not in session_paths:
        return session_id
    if not saved_session_replay_ready(session_dir):
        return session_id
    return ""


def current_public_transcript_turns() -> list[PublicTranscriptTurn]:
    session_id = str(st.session_state.get("runtime_monitor_session_id") or "").strip()
    if session_id:
        turns = load_public_transcript_turns(runtime_sessions_root() / session_id)
        return display_public_transcript_turns(turns)
    return display_public_transcript_turns(
        [
            make_public_transcript_turn(
                turn_id=int(record.get("turn_id") or index + 1),
                speaker=str(
                    record.get("speaker_role") or record.get("speaker_name") or ""
                ),
                text=str(record.get("text") or ""),
                created_at=(
                    None
                    if record.get("spoken_at") in {None, "-"}
                    else str(record.get("spoken_at"))
                ),
                audio_path=None,
                audio_duration_seconds=None,
                cumulative_audio_seconds=(
                    float(record["cumulative_audio_seconds"])
                    if "cumulative_audio_seconds" in record
                    else None
                ),
            )
            for index, record in enumerate(st.session_state.get("conversation_log", []))
        ]
    )


def display_public_transcript_turns(
    turns: list[PublicTranscriptTurn],
) -> list[PublicTranscriptTurn]:
    return [
        make_public_transcript_turn(
            turn_id=turn.turn_id,
            speaker=display_name_for_speaker(
                turn.speaker,
                speaker_id=getattr(turn, "speaker_id", None),
                speaker_display_name=getattr(turn, "speaker_display_name", None),
            ),
            text=turn.text,
            created_at=turn.created_at,
            audio_path=turn.audio_path,
            audio_duration_seconds=getattr(turn, "audio_duration_seconds", None),
            cumulative_audio_seconds=getattr(turn, "cumulative_audio_seconds", None),
            speaker_id=getattr(turn, "speaker_id", None),
            speaker_display_name=getattr(turn, "speaker_display_name", None),
            speaker_role=getattr(turn, "speaker_role", None),
            transcript_type=getattr(turn, "transcript_type", None),
            audio_log_path=getattr(turn, "audio_log_path", None),
            actor_kind=getattr(turn, "actor_kind", None),
        )
        for turn in turns
    ]


def make_public_transcript_turn(
    *,
    turn_id: int,
    speaker: str,
    text: str,
    created_at: str | None,
    audio_path: Path | None,
    audio_duration_seconds: float | None = None,
    cumulative_audio_seconds: float | None = None,
    speaker_id: str | None = None,
    speaker_display_name: str | None = None,
    speaker_role: str | None = None,
    transcript_type: str | None = None,
    audio_log_path: Path | None = None,
    actor_kind: str | None = None,
) -> PublicTranscriptTurn:
    try:
        return PublicTranscriptTurn(
            turn_id=turn_id,
            speaker=speaker,
            text=text,
            created_at=created_at,
            audio_path=audio_path,
            audio_duration_seconds=audio_duration_seconds,
            cumulative_audio_seconds=cumulative_audio_seconds,
            speaker_id=speaker_id,
            speaker_display_name=speaker_display_name,
            speaker_role=speaker_role,
            transcript_type=transcript_type,
            audio_log_path=audio_log_path,
            actor_kind=actor_kind,
        )
    except TypeError:
        return PublicTranscriptTurn(
            turn_id=turn_id,
            speaker=speaker,
            text=text,
            created_at=created_at,
            audio_path=audio_path,
        )


def display_name_for_speaker(
    speaker: str,
    *,
    speaker_id: str | None = None,
    speaker_display_name: str | None = None,
) -> str:
    if isinstance(speaker_display_name, str) and speaker_display_name.strip():
        return speaker_display_name.strip()
    speaker_key = speaker_id or speaker
    if speaker_key == "counselor":
        return str(
            st.session_state.get("counselor_display_name")
            or DEFAULT_RUNTIME_COUNSELOR_DISPLAY_NAME
        )
    if speaker_key == "client":
        return str(
            st.session_state.get("client_display_name")
            or DEFAULT_RUNTIME_CLIENT_DISPLAY_NAME
        )
    if speaker_key == "client_a":
        return str(
            st.session_state.get("client_a_display_name")
            or DEFAULT_RUNTIME_CLIENT_A_DISPLAY_NAME
        )
    if speaker_key == "client_b":
        return str(
            st.session_state.get("client_b_display_name")
            or DEFAULT_RUNTIME_CLIENT_B_DISPLAY_NAME
        )
    return speaker


def public_transcript_rows_for_display(
    turns: list[PublicTranscriptTurn],
) -> list[dict[str, Any]]:
    if not turns:
        return [{"開始時刻": "", "話者": "", "発話": ""}]
    rows: list[dict[str, Any]] = []
    cumulative_seconds = 0.0
    for turn in turns:
        start_seconds = public_turn_start_seconds(
            turn,
            fallback_cumulative_seconds=cumulative_seconds,
        )
        rows.append(
            {
                "開始時刻": format_seconds(start_seconds),
                "話者": turn.speaker,
                "発話": turn.text,
            }
        )
        cumulative_seconds += public_turn_audio_duration_seconds(turn)
    return rows


def public_turn_start_seconds(
    turn: PublicTranscriptTurn,
    *,
    fallback_cumulative_seconds: float | None = None,
) -> float:
    cumulative_audio_seconds = getattr(turn, "cumulative_audio_seconds", None)
    if cumulative_audio_seconds is not None:
        return max(0.0, float(cumulative_audio_seconds))
    if fallback_cumulative_seconds is not None:
        return max(0.0, float(fallback_cumulative_seconds))
    return 0.0


def public_turn_time_label(
    turn: PublicTranscriptTurn,
    *,
    fallback_cumulative_seconds: float | None = None,
) -> str:
    cumulative_audio_seconds = getattr(turn, "cumulative_audio_seconds", None)
    if cumulative_audio_seconds is not None:
        return format_seconds(float(cumulative_audio_seconds))
    if fallback_cumulative_seconds is not None:
        return format_seconds(fallback_cumulative_seconds)
    created_at = getattr(turn, "created_at", None)
    if not created_at:
        return "--:--"
    try:
        from datetime import datetime

        parsed = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
    except ValueError:
        return "--:--"
    return parsed.strftime("%H:%M:%S")


def public_turn_audio_duration_seconds(turn: PublicTranscriptTurn) -> float:
    explicit_duration = getattr(turn, "audio_duration_seconds", None)
    if explicit_duration is not None:
        return max(0.0, float(explicit_duration))
    audio_path = getattr(turn, "audio_path", None)
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


def public_turn_known_audio_duration_seconds(
    turn: PublicTranscriptTurn,
) -> float | None:
    duration_seconds = public_turn_audio_duration_seconds(turn)
    return duration_seconds if duration_seconds > 0 else None


def replay_start_time_options(
    turns: list[PublicTranscriptTurn],
) -> list[dict[str, Any]]:
    options: list[dict[str, Any]] = []
    cumulative_seconds = 0.0
    for index, turn in enumerate(turns):
        start_seconds = public_turn_start_seconds(
            turn,
            fallback_cumulative_seconds=cumulative_seconds,
        )
        label = format_seconds(start_seconds)
        options.append(
            {
                "key": f"{index}:{start_seconds:.3f}",
                "label": label,
                "start_seconds": start_seconds,
                "start_index": index,
            }
        )
        cumulative_seconds += public_turn_audio_duration_seconds(turn)
    return options


def html_table(
    rows: list[dict[str, Any]],
    *,
    columns: list[tuple[str, str, str]],
    large_body_text: bool = False,
    max_visible_body_rows: int | None = None,
) -> str:
    table_class = "cvd-table cvd-table--large-body" if large_body_text else "cvd-table"
    wrap_class = (
        "cvd-table-wrap cvd-table-wrap--scroll"
        if max_visible_body_rows is not None
        else "cvd-table-wrap"
    )
    max_rows_style = (
        f' style="--cvd-visible-body-rows: {max(1, int(max_visible_body_rows))};"'
        if max_visible_body_rows is not None
        else ""
    )
    colgroup = "\n".join(
        f'<col style="width: {html.escape(width)};" />' for _, _, width in columns
    )
    header = "".join(
        f"<th>{html.escape(label)}</th>" for label, _, _ in columns
    )
    body_rows = []
    for row in rows:
        cells = "".join(
            f'<td class="{html.escape(css_class)}">{html.escape(str(row.get(label, "")))}</td>'
            for label, css_class, _ in columns
        )
        body_rows.append(f"<tr>{cells}</tr>")
    body = "\n".join(body_rows)
    return f"""
<style>
.cvd-table-wrap {{
  overflow-x: auto;
  width: 100%;
}}
.cvd-table-wrap--scroll {{
  display: flex;
  flex-direction: column-reverse;
  max-height: calc(2.7rem + (2.9rem * var(--cvd-visible-body-rows)));
  overflow-y: auto;
  border: 1px solid rgba(148, 163, 184, 0.28);
  overflow-anchor: auto;
}}
.cvd-table-wrap--scroll .cvd-table {{
  border-collapse: separate;
  border-spacing: 0;
  flex: 0 0 auto;
}}
.cvd-table-wrap--scroll .cvd-table th {{
  position: sticky;
  top: 0;
  z-index: 2;
  background: var(--secondary-background-color, #111827);
  box-shadow: 0 1px 0 rgba(148, 163, 184, 0.28);
}}
.cvd-table {{
  border-collapse: collapse;
  table-layout: fixed;
  width: 100%;
  font-size: 0.875rem;
  line-height: 1.4;
}}
.cvd-table th,
.cvd-table td {{
  border: 1px solid rgba(148, 163, 184, 0.28);
  padding: 6px 8px;
  vertical-align: top;
  white-space: normal;
  overflow-wrap: anywhere;
  word-break: break-word;
}}
.cvd-table th {{
  background: rgba(148, 163, 184, 0.12);
  font-weight: 650;
  text-align: left;
}}
.cvd-table--large-body tbody td {{
  font-size: 2rem;
  line-height: 1.25;
}}
.cvd-table .id,
.cvd-table .edited,
.cvd-table .warning,
.cvd-table .time {{
  text-align: center;
}}
.cvd-table .utterance {{
  min-width: 280px;
}}
.cvd-table .speaker {{
  min-width: 72px;
}}
</style>
<div class="{wrap_class}"{max_rows_style}>
  <table class="{table_class}">
    <colgroup>
      {colgroup}
    </colgroup>
    <thead><tr>{header}</tr></thead>
    <tbody>
      {body}
    </tbody>
  </table>
</div>
"""


def turn_selection_options(
    unplayed_turns: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        {
            "turn_id": turn["turn_id"],
            "label": f"{turn['turn_id']}: {turn['speaker_name']} / {turn['speaker_role']}",
        }
        for turn in unplayed_turns
    ]


def selected_unplayed_turn(
    unplayed_turns: list[dict[str, Any]],
    selected_turn_id: int | None,
) -> dict[str, Any] | None:
    if not unplayed_turns:
        return None
    for turn in unplayed_turns:
        if turn["turn_id"] == selected_turn_id:
            return turn
    return unplayed_turns[0]


def edit_control_disabled_states(
    status: SessionStatus,
    *,
    selected_turn_id: int | None,
    unplayed_turns: list[dict[str, Any]],
    ui_mode: str,
) -> dict[str, bool]:
    selected_turn = selected_unplayed_turn(unplayed_turns, selected_turn_id)
    disabled = (
        ui_mode != "live"
        or status not in {SessionStatus.RUNNING, SessionStatus.PAUSED}
        or selected_turn is None
        or selected_turn["turn_id"] != selected_turn_id
    )
    return {
        "edit": disabled,
        "regenerate": disabled,
        "regenerate_from": disabled,
    }


def apply_edit_to_unplayed_turns(
    unplayed_turns: list[dict[str, Any]],
    *,
    selected_turn_id: int | None,
    edited_text: str,
) -> dict[str, Any]:
    selected_turn = next(
        (turn for turn in unplayed_turns if turn["turn_id"] == selected_turn_id), None
    )
    if selected_turn is None:
        return {
            "unplayed_turns": [dict(turn) for turn in unplayed_turns],
            "warnings": ["編集対象ターンが見つかりません。"],
        }

    invalidated_count = 0
    updated_turns: list[dict[str, Any]] = []
    for turn in unplayed_turns:
        updated = dict(turn)
        if turn["turn_id"] == selected_turn_id:
            updated.update(
                {
                    "revision_id": int(turn["revision_id"]) + 1,
                    "text": edited_text,
                    "audio_status": TurnStatus.TEXT_READY.value,
                    "playback_status": TurnStatus.TEXT_READY.value,
                    "edited": True,
                    "warning_level": WarningLevel.NONE.value,
                    "hold_reason": "",
                }
            )
        elif turn["turn_id"] > selected_turn_id:
            updated["audio_status"] = TurnStatus.INVALIDATED.value
            updated["playback_status"] = TurnStatus.INVALIDATED.value
            invalidated_count += 1
        updated_turns.append(updated)

    if invalidated_count:
        warning = f"Turn {selected_turn_id} を編集し、後続未再生ターン {invalidated_count} 件を無効化しました。"
    else:
        warning = f"Turn {selected_turn_id} を編集しました。"
    return {"unplayed_turns": updated_turns, "warnings": [warning]}


def apply_regenerate_to_unplayed_turns(
    unplayed_turns: list[dict[str, Any]],
    *,
    selected_turn_id: int | None,
) -> dict[str, Any]:
    selected_turn = next(
        (turn for turn in unplayed_turns if turn["turn_id"] == selected_turn_id), None
    )
    if selected_turn is None:
        return {
            "unplayed_turns": [dict(turn) for turn in unplayed_turns],
            "warnings": ["再生成対象ターンが見つかりません。"],
        }

    invalidated_count = 0
    updated_turns: list[dict[str, Any]] = []
    for turn in unplayed_turns:
        updated = dict(turn)
        if turn["turn_id"] == selected_turn_id:
            updated.update(
                {
                    "revision_id": int(turn["revision_id"]) + 1,
                    "audio_status": TurnStatus.TEXT_READY.value,
                    "playback_status": TurnStatus.TEXT_READY.value,
                    "edited": False,
                    "warning_level": WarningLevel.NONE.value,
                    "hold_reason": "",
                }
            )
        elif turn["turn_id"] > selected_turn_id:
            updated["audio_status"] = TurnStatus.INVALIDATED.value
            updated["playback_status"] = TurnStatus.INVALIDATED.value
            invalidated_count += 1
        updated_turns.append(updated)

    if invalidated_count:
        warning = f"Turn {selected_turn_id} を再生成待ちにし、後続未再生ターン {invalidated_count} 件を無効化しました。"
    else:
        warning = f"Turn {selected_turn_id} を再生成待ちにしました。"
    return {"unplayed_turns": updated_turns, "warnings": [warning]}


def apply_regenerate_from_to_unplayed_turns(
    unplayed_turns: list[dict[str, Any]],
    *,
    selected_turn_id: int | None,
) -> dict[str, Any]:
    selected_turn = next(
        (turn for turn in unplayed_turns if turn["turn_id"] == selected_turn_id), None
    )
    if selected_turn is None:
        return {
            "unplayed_turns": [dict(turn) for turn in unplayed_turns],
            "warnings": ["選択ターン以降の再生成対象が見つかりません。"],
        }

    invalidated_count = 0
    updated_turns: list[dict[str, Any]] = []
    for turn in unplayed_turns:
        updated = dict(turn)
        if turn["turn_id"] >= selected_turn_id:
            updated["audio_status"] = TurnStatus.INVALIDATED.value
            updated["playback_status"] = TurnStatus.INVALIDATED.value
            invalidated_count += 1
        updated_turns.append(updated)

    return {
        "unplayed_turns": updated_turns,
        "warnings": [
            f"Turn {selected_turn_id} 以降の未再生ターン {invalidated_count} 件を無効化し、再生成待ちにしました。"
        ],
    }


def apply_hold_action_to_unplayed_turns(
    unplayed_turns: list[dict[str, Any]],
    *,
    selected_turn_id: int | None,
    hold_action: str,
) -> dict[str, Any]:
    selected_turn = next(
        (turn for turn in unplayed_turns if turn["turn_id"] == selected_turn_id), None
    )
    if selected_turn is None:
        return {
            "unplayed_turns": [dict(turn) for turn in unplayed_turns],
            "warnings": ["playback_hold 操作対象が見つかりません。"],
        }
    if selected_turn["playback_status"] != TurnStatus.PLAYBACK_HOLD.value:
        return {
            "unplayed_turns": [dict(turn) for turn in unplayed_turns],
            "warnings": ["選択ターンは playback_hold ではありません。"],
        }

    if hold_action not in {"このまま再生", "スキップ"}:
        return {
            "unplayed_turns": [dict(turn) for turn in unplayed_turns],
            "warnings": [f"未対応の playback_hold 操作です: {hold_action}"],
        }

    updated_turns: list[dict[str, Any]] = []
    for turn in unplayed_turns:
        updated = dict(turn)
        if turn["turn_id"] == selected_turn_id:
            if hold_action == "このまま再生":
                updated["audio_status"] = TurnStatus.AUDIO_READY.value
                updated["playback_status"] = TurnStatus.AUDIO_READY.value
            else:
                updated["audio_status"] = TurnStatus.SKIPPED_UNPLAYED.value
                updated["playback_status"] = TurnStatus.SKIPPED_UNPLAYED.value
        updated_turns.append(updated)

    if hold_action == "このまま再生":
        warning = f"Turn {selected_turn_id} の playback_hold を解除しました。"
    else:
        warning = f"Turn {selected_turn_id} を未再生スキップにしました。"
    return {"unplayed_turns": updated_turns, "warnings": [warning]}


def unplayed_turn_rows_from_session(session: SessionState) -> list[dict[str, Any]]:
    return [
        _turn_to_unplayed_row(turn)
        for turn in sorted(session.turns, key=lambda item: item.turn_id)
        if is_unplayed_turn(turn)
    ]


def apply_edit_to_session(
    session: SessionState,
    *,
    selected_turn_id: int | None,
    edited_text: str,
    log_paths: SessionLogPaths | None = None,
) -> dict[str, Any]:
    if selected_turn_id is None:
        return {
            "unplayed_turns": unplayed_turn_rows_from_session(session),
            "warnings": ["編集対象ターンが見つかりません。"],
        }
    try:
        result = edit_unplayed_turn(
            session,
            turn_id=selected_turn_id,
            edited_text=edited_text,
            log_paths=log_paths,
        )
    except ValueError as exc:
        return {
            "unplayed_turns": unplayed_turn_rows_from_session(session),
            "warnings": [str(exc)],
        }

    invalidated_count = len(result.invalidated_following_turns)
    if invalidated_count:
        warning = f"Turn {selected_turn_id} を編集し、後続未再生ターン {invalidated_count} 件を無効化しました。"
    else:
        warning = f"Turn {selected_turn_id} を編集しました。"
    return {
        "unplayed_turns": unplayed_turn_rows_from_session(session),
        "warnings": [warning],
    }


def apply_regenerate_to_session(
    session: SessionState,
    *,
    selected_turn_id: int | None,
    conversation_engine: object | None,
    cumulative_audio_seconds: float,
    moderation_client: object | None = None,
    moderation_model: str | None = None,
    clinical_ng_checker: object | None = None,
    clinical_model: str | None = None,
    tts_client: object | None = None,
    log_paths: SessionLogPaths | None = None,
) -> dict[str, Any]:
    if selected_turn_id is None or conversation_engine is None:
        return {
            "unplayed_turns": unplayed_turn_rows_from_session(session),
            "warnings": ["再生成対象ターンが見つかりません。"],
        }
    try:
        result = regenerate_single_turn(
            session,
            turn_id=selected_turn_id,
            conversation_engine=conversation_engine,
            cumulative_audio_seconds=cumulative_audio_seconds,
            moderation_client=moderation_client,
            moderation_model=moderation_model,
            clinical_ng_checker=clinical_ng_checker,
            clinical_model=clinical_model,
            tts_client=tts_client,
            log_paths=log_paths,
        )
    except ValueError as exc:
        return {
            "unplayed_turns": unplayed_turn_rows_from_session(session),
            "warnings": [str(exc)],
        }

    invalidated_count = len(result.invalidated_following_turns)
    generated_count = len(result.generated_following_turns)
    warning = f"Turn {selected_turn_id} を再生成し、後続未再生ターン {invalidated_count} 件を無効化しました。"
    if generated_count:
        warning = f"{warning} 後続ターン {generated_count} 件を再生成しました。"
    return {
        "unplayed_turns": unplayed_turn_rows_from_session(session),
        "warnings": [warning],
    }


def apply_regenerate_from_to_session(
    session: SessionState,
    *,
    selected_turn_id: int | None,
    conversation_engine: object | None,
    cumulative_audio_seconds: float,
    log_paths: SessionLogPaths | None = None,
) -> dict[str, Any]:
    if selected_turn_id is None or conversation_engine is None:
        return {
            "unplayed_turns": unplayed_turn_rows_from_session(session),
            "warnings": ["選択ターン以降の再生成対象が見つかりません。"],
        }
    try:
        result = regenerate_from_turn(
            session,
            turn_id=selected_turn_id,
            conversation_engine=conversation_engine,
            cumulative_audio_seconds=cumulative_audio_seconds,
            log_paths=log_paths,
        )
    except ValueError as exc:
        return {
            "unplayed_turns": unplayed_turn_rows_from_session(session),
            "warnings": [str(exc)],
        }

    return {
        "unplayed_turns": unplayed_turn_rows_from_session(session),
        "warnings": [
            f"Turn {selected_turn_id} 以降の未再生ターン {len(result.invalidated_turns)} 件を無効化し、"
            f"後続ターン {len(result.generated_turns)} 件を再生成しました。"
        ],
    }


def apply_hold_action_to_session(
    session: SessionState,
    *,
    selected_turn_id: int | None,
    hold_action: str,
    log_paths: SessionLogPaths | None = None,
) -> dict[str, Any]:
    if selected_turn_id is None:
        return {
            "unplayed_turns": unplayed_turn_rows_from_session(session),
            "warnings": ["playback_hold 操作対象が見つかりません。"],
        }
    turn = next((item for item in session.turns if item.turn_id == selected_turn_id), None)
    if turn is None:
        return {
            "unplayed_turns": unplayed_turn_rows_from_session(session),
            "warnings": ["playback_hold 操作対象が見つかりません。"],
        }
    try:
        if hold_action == "このまま再生":
            release_playback_hold(turn, log_paths=log_paths)
            warning = f"Turn {selected_turn_id} の playback_hold を解除しました。"
        elif hold_action == "スキップ":
            skip_unplayed_hold(turn, log_paths=log_paths)
            warning = f"Turn {selected_turn_id} を未再生スキップにしました。"
        else:
            return {
                "unplayed_turns": unplayed_turn_rows_from_session(session),
                "warnings": [f"未対応の playback_hold 操作です: {hold_action}"],
            }
    except ValueError as exc:
        return {
            "unplayed_turns": unplayed_turn_rows_from_session(session),
            "warnings": [str(exc)],
        }
    return {"unplayed_turns": unplayed_turn_rows_from_session(session), "warnings": [warning]}


def replay_start_updates() -> dict[str, str]:
    return {
        "session_status": SessionStatus.RUNNING.value,
        "ui_mode": "replay",
    }


def should_show_evaluation(
    status: SessionStatus,
    *,
    show_evaluation: bool,
    selected_mode: str | None = None,
) -> bool:
    if selected_mode == "ai_counselor_human_client":
        return False
    return show_evaluation and status in {SessionStatus.COMPLETED, SessionStatus.STOPPED}


def can_run_evaluation(
    status: SessionStatus,
    *,
    ui_mode: str,
    selected_mode: str | None = None,
) -> bool:
    if selected_mode == "ai_counselor_human_client":
        return False
    return ui_mode != "replay" and status in {SessionStatus.COMPLETED, SessionStatus.STOPPED}


def build_evaluation_internal_context(
    *,
    selected_mode: str,
    counselor_profile: Profile,
    client_profile: Profile,
    theme: Theme,
) -> dict[str, Any]:
    if selected_mode == "ai_counselor_human_client":
        return {"selected_mode": selected_mode, "evaluation_supported": False}
    context: dict[str, Any] = {
        "selected_mode": selected_mode,
        "counselor_profile": {
            "profile_id": counselor_profile.profile_id,
            "display_name": counselor_profile.display_name,
            "public_profile": counselor_profile.public_profile,
            "prompt": counselor_profile.prompt,
        },
        "client_profile": {
            "profile_id": client_profile.profile_id,
            "display_name": client_profile.display_name,
            "public_profile": client_profile.public_profile,
            "prompt": client_profile.prompt,
        },
        "theme": {
            "theme_id": theme.theme_id,
            "display_name": theme.display_name,
            "body": theme.body,
        },
    }
    if client_profile.hidden_background:
        context["client_profile"]["hidden_background"] = client_profile.hidden_background
    return context


def session_for_evaluation(
    *,
    session_model: SessionState | None,
    conversation_log: list[dict[str, Any]],
    status: SessionStatus,
) -> SessionState:
    if session_model is not None:
        session_model.status = status
        return session_model

    session = SessionState()
    session.status = status
    for record in conversation_log:
        speaker_role = _speaker_role_from_record(record)
        session.add_turn(
            speaker_role=speaker_role,
            speaker_name=str(record.get("speaker_name") or speaker_role.value),
            profile_id=str(record.get("profile_id") or f"{speaker_role.value}_ui"),
            text=str(record.get("text") or ""),
        )
    return session


def run_evaluation_for_ui(
    *,
    session_model: SessionState | None,
    conversation_log: list[dict[str, Any]],
    status: SessionStatus,
    evaluation_client: Any,
    model: str,
    reasoning_effort: str | None,
    log_paths: SessionLogPaths | None,
    sessions_dir: Path | str | None = None,
    internal_context: dict[str, Any],
) -> dict[str, Any]:
    if internal_context.get("selected_mode") == "ai_counselor_human_client":
        raise ValueError("人間クライアントモードの自動評価には未対応です。")
    session = session_for_evaluation(
        session_model=session_model,
        conversation_log=conversation_log,
        status=status,
    )
    if log_paths is None and sessions_dir is not None:
        log_paths = create_session_log_dirs(sessions_dir, session.session_id)
    saved = log_paths is not None
    result = run_session_evaluations(
        session=session,
        client=evaluation_client,
        config=EvaluationRunConfig(model=model, reasoning_effort=reasoning_effort),
        log_paths=log_paths,
        internal_context=internal_context,
    )
    return {
        "evaluation_public": result.public.text,
        "evaluation_internal": result.internal.text,
        "evaluation_status": "completed",
        "evaluation_error": "",
        "warnings": ["自動評価を保存しました。" if saved else "自動評価を実行しました。"],
        "log_paths": log_paths,
    }


def create_openai_evaluation_client() -> OpenAIEvaluationClient:
    from openai import OpenAI

    return OpenAIEvaluationClient(client=OpenAI())


def _speaker_role_from_record(record: dict[str, Any]) -> SpeakerRole:
    try:
        return SpeakerRole(str(record.get("speaker_role")))
    except ValueError:
        return SpeakerRole.COUNSELOR


def select_profile_by_id(profiles: list[Profile], profile_id: str | None) -> Profile:
    for profile in profiles:
        if profile.profile_id == profile_id:
            return profile
    return profiles[0]


def select_theme_by_id(themes: list[Theme], theme_id: str | None) -> Theme:
    for theme in themes:
        if theme.theme_id == theme_id:
            return theme
    return themes[0]


def select_voice_preset_by_id(
    presets: list[VoicePreset], preset_id: str | None
) -> VoicePreset:
    for preset in presets:
        if preset.preset_id == preset_id:
            return preset
    return presets[0]


def select_voice_preset_for_profile(
    presets: list[VoicePreset],
    *,
    selected_preset_id: str | None,
    profile: Profile,
) -> VoicePreset:
    return select_voice_preset_by_id(
        presets,
        selected_preset_id or profile.voice_preset,
    )


def profile_with_voice_preset(profile: Profile, preset: VoicePreset) -> Profile:
    return profile.model_copy(
        update={
            "voice_preset": preset.preset_id,
            "tts_model": preset.tts_model,
            "tts_voice": preset.voice,
            "tts_instructions": preset.instructions,
        }
    )


def profile_with_session_overrides(
    profile: Profile,
    *,
    display_name_key: str | None = None,
    public_profile_text_key: str | None = None,
    prompt_text_key: str | None = None,
    hidden_background_key: str | None = None,
) -> Profile:
    updates: dict[str, Any] = {}
    if display_name_key:
        display_name = st.session_state.get(display_name_key)
        if isinstance(display_name, str) and display_name.strip():
            updates["display_name"] = display_name.strip()
    if public_profile_text_key:
        public_profile_text = st.session_state.get(public_profile_text_key)
        if isinstance(public_profile_text, str):
            updates["public_profile"] = public_profile_text.strip()
    if prompt_text_key:
        prompt_text = st.session_state.get(prompt_text_key)
        if isinstance(prompt_text, str) and prompt_text.strip():
            updates["prompt"] = prompt_text
    if hidden_background_key:
        hidden_background = st.session_state.get(hidden_background_key)
        if isinstance(hidden_background, str):
            updates["hidden_background"] = hidden_background.strip() or None
    if not updates:
        return profile
    return profile.model_copy(update=updates)


def apply_auto_progress_tick_to_session_state(config: Any) -> bool:
    if not st.session_state.get("auto_progress_enabled", True):
        return False
    if st.session_state.get("auto_progress_busy", False):
        return False
    session = st.session_state.get("session_model")
    engine = st.session_state.get("conversation_engine")
    if not isinstance(session, SessionState) or not isinstance(
        engine, ConversationEngine
    ):
        return False
    st.session_state["auto_progress_busy"] = True
    try:
        old_playback_turn_id = st.session_state.get("auto_playback_turn_id")
        updates = auto_progress_live_session_for_ui(
            session,
            conversation_engine=engine,
            cumulative_audio_seconds=float(st.session_state["cumulative_audio_seconds"]),
            tts_client=st.session_state.get("tts_client"),
            log_paths=st.session_state.get("log_paths"),
            response_format=config.openai.default_audio_format,
            selected_turn_id=st.session_state.get("selected_unplayed_turn_id"),
            playback_turn_id=old_playback_turn_id,
            playback_started_at=st.session_state.get("auto_playback_started_at"),
            playback_duration_seconds=st.session_state.get(
                "auto_playback_duration_seconds"
            ),
            now_monotonic=time.monotonic(),
            playback_gap_seconds=float(
                st.session_state.get(
                    "auto_playback_gap_seconds",
                    AUTO_PLAYBACK_DEFAULT_GAP_SECONDS,
                )
            ),
            existing_warnings=st.session_state.get("warnings", []),
            allow_server_side_playback_completion=False,
            playback_rendered_at=st.session_state.get("auto_playback_visible_started_at"),
            browser_audio_player_manages_playback=bool(
                st.session_state.get("audio_player_continuous_mode", False)
            ),
        )
        changed = bool(updates.pop("_auto_progress_changed", False))
        if updates.get("auto_playback_turn_id") != old_playback_turn_id:
            updates["auto_playback_visible_turn_id"] = None
            updates["auto_playback_visible_started_at"] = None
        for key, value in updates.items():
            st.session_state[key] = value
        return changed
    finally:
        st.session_state["auto_progress_busy"] = False


def clear_auto_playback_session_state() -> None:
    for key, value in clear_auto_playback_updates().items():
        st.session_state[key] = value


def scheduled_playback_is_due_to_render(now_monotonic: float | None = None) -> bool:
    turn_id = st.session_state.get("auto_playback_turn_id")
    started_at = st.session_state.get("auto_playback_started_at")
    visible_turn_id = st.session_state.get("auto_playback_visible_turn_id")
    if turn_id is None or started_at is None or visible_turn_id == turn_id:
        return False
    now = time.monotonic() if now_monotonic is None else now_monotonic
    return now >= float(started_at)


@st.fragment(run_every=AUTO_PROGRESS_FRAGMENT_INTERVAL_SECONDS)
def render_auto_progress_controller(config: Any) -> None:
    if st.session_state.get("session_status") != SessionStatus.RUNNING.value:
        return
    if apply_auto_progress_tick_to_session_state(config):
        st.rerun(scope="app")


def main() -> None:
    st.set_page_config(
        page_title="Counseling Voice Demo",
        page_icon=None,
        layout="wide",
    )
    ensure_ui_state()

    try:
        config = load_app_config()
        counselor_profiles = list_profiles("counselor")
        client_profiles = list_profiles("client")
        counselor_presets = list_counselor_presets(
            ROOT_DIR / "config" / "counselor_presets"
        )
        client_presets = list_client_presets(ROOT_DIR / "config" / "client_presets")
        themes = list_themes()
        voice_presets = list_voice_presets()
    except (
        ConfigLoadError,
        ContentLoadError,
        CounselorPresetLoadError,
        ClientPresetLoadError,
    ) as exc:
        st.error(str(exc))
        return

    st.title("Counseling Voice Demo")
    st.caption(config.ui.ai_voice_disclosure_text)

    selected_mode = render_mode_selector(config.app.enabled_modes)
    render_settings_panel(
        config,
        counselor_profiles,
        client_profiles,
        counselor_presets,
        client_presets,
        themes,
        voice_presets,
    )
    render_role_area(counselor_profiles, client_profiles, voice_presets)
    render_runtime_session_panel(
        selected_mode,
        config.app.enabled_modes.get(selected_mode, False),
        counselor_profiles,
        client_profiles,
        themes,
    )
    render_live_transcript_panel(config)
    render_saved_session_replay(config)
    render_runtime_observer_panel()
    render_warnings()

    st.caption(f"Application package version: {__version__}")


def render_mode_selector(enabled_modes: dict[str, bool]) -> str:
    mode_options = build_mode_options(enabled_modes)
    enabled_options = [option for option in mode_options if option["enabled"]]
    current_mode = str(
        st.session_state.get("selected_mode") or "ai_counselor_ai_client"
    )
    if not enabled_options:
        st.session_state["selected_mode"] = current_mode
        return current_mode

    current_enabled_index = next(
        (
            index
            for index, option in enumerate(enabled_options)
            if option["mode"] == current_mode
        ),
        0,
    )
    selected_label = st.radio(
        "実行モード",
        [option["label"] for option in enabled_options],
        index=current_enabled_index,
        horizontal=True,
        key="runtime_mode_selector",
    )
    mode_by_label = {option["label"]: option["mode"] for option in enabled_options}
    selected_mode = str(mode_by_label[str(selected_label)])
    st.session_state["selected_mode"] = selected_mode
    return selected_mode


def render_timer_band(
    config, *, turns: list[PublicTranscriptTurn] | None = None
) -> None:
    wall_clock_seconds = float(st.session_state["wall_clock_seconds"])
    visibility_clock_seconds = float(
        st.session_state.get("runtime_transcript_visibility_clock_seconds")
        or wall_clock_seconds
    )
    status = st.session_state.get("runtime_control_status") or {}
    human_counselor_mode = runtime_human_input_enabled()
    generated_turns = runtime_generated_turn_count_for_display(
        status,
        include_active_turn=not human_counselor_mode,
    )
    timer_turns = current_public_transcript_turns() if turns is None else turns
    session_id = str(st.session_state.get("runtime_monitor_session_id") or "")
    completed_updates = runtime_completed_turn_count_state_updates(
        status=status,
        turns=timer_turns,
        clock_seconds=visibility_clock_seconds,
        session_id=session_id,
        previous_session_id=str(
            st.session_state.get("runtime_completed_turn_count_session_id") or ""
        ),
        previous_completed_turns=int(
            st.session_state.get("runtime_completed_turn_count_for_display") or 0
        ),
        browser_completed_turn_count=runtime_status_playback_completed_turn_count(
            status
        ),
    )
    for key, value in completed_updates.items():
        st.session_state[key] = value
    completed_turns = int(completed_updates["runtime_completed_turn_count_for_display"])
    closing_start_seconds = float(
        st.session_state.get("runtime_closing_start_seconds")
        or config.app.closing_start_audio_seconds
    )
    force_stop_after_closing_turns = int(
        st.session_state.get("runtime_force_stop_after_closing_turns")
        or config.app.farewell_after_turns
    )
    status_label = runtime_status_label_for_timer(
        st.session_state["session_status"],
        generated_turns=generated_turns,
        completed_turns=completed_turns,
    )
    if st.session_state.get("ui_mode") == "replay":
        status_label = f"{status_label} / replay"
    if human_counselor_mode:
        columns = st.columns(3)
        columns[0].metric("経過時間", format_seconds(wall_clock_seconds))
        columns[1].metric("現在ターン", runtime_current_turn_metric_value(status))
        columns[2].metric("状態", status_label)
    else:
        remaining_label, remaining_value = runtime_remaining_metric(
            status=status,
            wall_clock_seconds=wall_clock_seconds,
            closing_start_seconds=closing_start_seconds,
            force_stop_after_closing_turns=force_stop_after_closing_turns,
        )
        columns = st.columns(5)
        columns[0].metric("経過時間", format_seconds(wall_clock_seconds))
        columns[1].metric(remaining_label, remaining_value)
        columns[2].metric("生成ターン数", f"{generated_turns}ターン")
        columns[3].metric("完了ターン数", f"{completed_turns}ターン")
        columns[4].metric("状態", status_label)


def render_controls(
    selected_mode: str,
    mode_enabled: bool,
    config: Any,
    counselor_profiles: list[Profile],
    client_profiles: list[Profile],
    themes: list[Theme],
    voice_presets: list[VoicePreset],
) -> None:
    status = SessionStatus(st.session_state["session_status"])
    disabled = control_disabled_states(status, mode_enabled=mode_enabled)
    edit_disabled = edit_control_disabled_states(
        status,
        selected_turn_id=st.session_state.get("selected_unplayed_turn_id"),
        unplayed_turns=st.session_state["unplayed_turns"],
        ui_mode=st.session_state.get("ui_mode", "live"),
    )
    controls = st.columns(6)
    if controls[0].button(
        "Start", disabled=disabled["start"], width="stretch"
    ):
        clear_auto_playback_session_state()
        counselor = select_profile_by_id(
            counselor_profiles, st.session_state["selected_counselor_profile_id"]
        )
        client = select_profile_by_id(
            client_profiles, st.session_state["selected_client_profile_id"]
        )
        counselor_voice = select_voice_preset_for_profile(
            voice_presets,
            selected_preset_id=st.session_state["selected_counselor_voice_preset_id"],
            profile=counselor,
        )
        client_voice = select_voice_preset_for_profile(
            voice_presets,
            selected_preset_id=st.session_state["selected_client_voice_preset_id"],
            profile=client,
        )
        counselor = profile_with_voice_preset(counselor, counselor_voice)
        client = profile_with_voice_preset(client, client_voice)
        counselor = profile_with_session_overrides(
            counselor,
            display_name_key="counselor_display_name",
            public_profile_text_key="counselor_public_profile_text",
            prompt_text_key="counselor_prompt_text",
        )
        client = profile_with_session_overrides(
            client,
            display_name_key="client_display_name",
            public_profile_text_key="client_public_profile_text",
            prompt_text_key="client_prompt_text",
            hidden_background_key="client_private_profile_text",
        )
        theme = select_theme_by_id(themes, st.session_state["selected_theme_id"])
        try:
            with st.spinner("先行生成と音声化を実行中です..."):
                updates = start_live_session_for_ui(
                    config=config,
                    counselor_profile=counselor,
                    client_profile=client,
                    theme=theme,
                    sessions_dir=ROOT_DIR / config.paths.sessions_dir,
                    ahead_generation_turns=st.session_state.get(
                        "ahead_generation_turns"
                    ),
                )
        except Exception as exc:
            updates = {
                "session_status": SessionStatus.ERROR.value,
                "warnings": [f"セッション開始に失敗しました: {exc}"],
        }
        for key, value in updates.items():
            st.session_state[key] = value
        clear_auto_playback_session_state()
    if controls[1].button(
        "Pause", disabled=disabled["pause"], width="stretch"
    ):
        set_session_status(SessionStatus.PAUSED)
    if controls[2].button(
        "Resume", disabled=disabled["resume"], width="stretch"
    ):
        set_session_status(SessionStatus.RUNNING)
    if controls[3].button("Stop", disabled=disabled["stop"], width="stretch"):
        set_session_status(SessionStatus.STOPPED)
    if controls[4].button(
        "Skip current audio", disabled=disabled["skip"], width="stretch"
    ):
        clear_auto_playback_session_state()
        if isinstance(st.session_state.get("session_model"), SessionState) and isinstance(
            st.session_state.get("conversation_engine"), ConversationEngine
        ):
            updates = skip_current_audio_for_ui(
                st.session_state["session_model"],
                conversation_engine=st.session_state["conversation_engine"],
                cumulative_audio_seconds=float(
                    st.session_state["cumulative_audio_seconds"]
                ),
                tts_client=st.session_state.get("tts_client"),
                log_paths=st.session_state.get("log_paths"),
                response_format=config.openai.default_audio_format,
                selected_turn_id=st.session_state.get("selected_unplayed_turn_id"),
            )
            for key, value in updates.items():
                st.session_state[key] = value
        else:
            st.session_state["warnings"] = ["スキップ対象のライブセッションがありません。"]
    if controls[5].button(
        "Start replay",
        disabled=selected_mode != "ai_counselor_ai_client",
        width="stretch",
    ):
        for key, value in replay_start_updates().items():
            st.session_state[key] = value

    secondary = st.columns(6)
    if secondary[0].button(
        "Edit selected turn", disabled=edit_disabled["edit"], width="stretch"
    ):
        clear_auto_playback_session_state()
        if isinstance(st.session_state.get("session_model"), SessionState):
            result = apply_edit_to_session(
                st.session_state["session_model"],
                selected_turn_id=st.session_state.get("selected_unplayed_turn_id"),
                edited_text=st.session_state.get("edit_turn_text", ""),
                log_paths=st.session_state.get("log_paths"),
            )
        else:
            result = apply_edit_to_unplayed_turns(
                st.session_state["unplayed_turns"],
                selected_turn_id=st.session_state.get("selected_unplayed_turn_id"),
                edited_text=st.session_state.get("edit_turn_text", ""),
            )
        st.session_state["unplayed_turns"] = result["unplayed_turns"]
        st.session_state["warnings"] = result["warnings"]
        st.session_state["edit_turn_text_source_turn_id"] = st.session_state.get(
            "selected_unplayed_turn_id"
        )
    if secondary[1].button(
        "Regenerate selected turn",
        disabled=edit_disabled["regenerate"],
        width="stretch",
    ):
        clear_auto_playback_session_state()
        if isinstance(st.session_state.get("session_model"), SessionState):
            result = apply_regenerate_to_session(
                st.session_state["session_model"],
                selected_turn_id=st.session_state.get("selected_unplayed_turn_id"),
                conversation_engine=st.session_state.get("conversation_engine"),
                cumulative_audio_seconds=float(
                    st.session_state["cumulative_audio_seconds"]
                ),
                moderation_client=st.session_state.get("moderation_client"),
                moderation_model=st.session_state.get("moderation_model"),
                clinical_ng_checker=st.session_state.get("clinical_ng_checker"),
                clinical_model=st.session_state.get("clinical_model"),
                tts_client=st.session_state.get("tts_client"),
                log_paths=st.session_state.get("log_paths"),
            )
        else:
            result = apply_regenerate_to_unplayed_turns(
                st.session_state["unplayed_turns"],
                selected_turn_id=st.session_state.get("selected_unplayed_turn_id"),
            )
        st.session_state["unplayed_turns"] = result["unplayed_turns"]
        st.session_state["warnings"] = result["warnings"]
        st.session_state["edit_turn_text_source_turn_id"] = st.session_state.get(
            "selected_unplayed_turn_id"
        )
    if secondary[2].button(
        "Regenerate from selected turn",
        disabled=edit_disabled["regenerate_from"],
        width="stretch",
    ):
        clear_auto_playback_session_state()
        if isinstance(st.session_state.get("session_model"), SessionState):
            result = apply_regenerate_from_to_session(
                st.session_state["session_model"],
                selected_turn_id=st.session_state.get("selected_unplayed_turn_id"),
                conversation_engine=st.session_state.get("conversation_engine"),
                cumulative_audio_seconds=float(
                    st.session_state["cumulative_audio_seconds"]
                ),
                log_paths=st.session_state.get("log_paths"),
            )
        else:
            result = apply_regenerate_from_to_unplayed_turns(
                st.session_state["unplayed_turns"],
                selected_turn_id=st.session_state.get("selected_unplayed_turn_id"),
            )
        st.session_state["unplayed_turns"] = result["unplayed_turns"]
        st.session_state["warnings"] = result["warnings"]
        st.session_state["edit_turn_text_source_turn_id"] = st.session_state.get(
            "selected_unplayed_turn_id"
        )
    if secondary[3].button("Save session", width="stretch"):
        st.session_state["warnings"] = ["現在のUI状態を保持しています。"]
    if secondary[4].button(
        (
            "Show evaluation"
            if not st.session_state["show_evaluation"]
            else "Hide evaluation"
        ),
        width="stretch",
    ):
        st.session_state["show_evaluation"] = not st.session_state["show_evaluation"]
    if secondary[5].button("Show prompts", width="stretch"):
        st.session_state["show_prompts"] = not st.session_state["show_prompts"]


def render_runtime_session_panel(
    selected_mode: str,
    mode_enabled: bool,
    counselor_profiles: list[Profile],
    client_profiles: list[Profile],
    themes: list[Theme],
) -> None:
    defaults = runtime_control_defaults()
    if not st.session_state.get("runtime_control_host"):
        st.session_state["runtime_control_host"] = defaults["host"]
    if not st.session_state.get("runtime_control_port"):
        st.session_state["runtime_control_port"] = int(defaults["port"])
    if not st.session_state.get("runtime_generation_mode"):
        st.session_state["runtime_generation_mode"] = "realtime_api"

    st.subheader("Realtime session")
    if defaults["warning"]:
        st.warning(defaults["warning"])
    if not mode_enabled:
        st.info("このモードは将来拡張として準備中です。")

    endpoint = build_runtime_control_endpoint(
        host=str(st.session_state["runtime_control_host"]),
        port=int(st.session_state["runtime_control_port"]),
        session_id=st.session_state.get("runtime_monitor_session_id") or None,
    )
    client = RuntimeControlClient(endpoint.base_url)
    counselor = select_profile_by_id(
        counselor_profiles, st.session_state["selected_counselor_profile_id"]
    )
    runtime_client = select_profile_by_id(
        client_profiles, st.session_state["selected_client_profile_id"]
    )
    _ = select_theme_by_id(themes, st.session_state["selected_theme_id"])
    runtime_start_options = runtime_start_options_for_ui(
        counselor_profile=counselor,
        client_profile=runtime_client,
        selected_mode=selected_mode,
    )
    st.session_state["runtime_pending_audio_settings_rows"] = (
        runtime_audio_settings_rows_from_start_options(runtime_start_options)
    )
    human_input_enabled = runtime_human_input_enabled()
    warmup_target_turns=int(
        st.session_state.get("runtime_generation_lead_limit")
        or DEFAULT_RUNTIME_GENERATION_LEAD_LIMIT
    )
    warmup_max_wait_ms=int(RUNTIME_AUDIO_MONITOR_WARMUP_MAX_WAIT_SECONDS * 1000)
    runtime_audio_monitor(
        ws_url=endpoint.monitor_ws_url,
        volume=1.0,
        muted=False,
        start_url=f"{endpoint.base_url}/runtime/start",
        runtime_pause_url=f"{endpoint.base_url}/runtime/pause",
        runtime_resume_url=f"{endpoint.base_url}/runtime/resume",
        runtime_stop_url=f"{endpoint.base_url}/runtime/stop",
        runtime_interrupt_url=f"{endpoint.base_url}/runtime/interrupt",
        runtime_playback_completed_url=(
            f"{endpoint.base_url}/runtime/playback-completed"
        ),
        runtime_status_url=f"{endpoint.base_url}/runtime/status",
        human_audio_url=(
            f"{endpoint.base_url}/runtime/human-audio"
            if human_input_enabled
            else ""
        ),
        human_audio_stream_url=(
            f"{endpoint.base_url.replace('http://', 'ws://', 1)}"
            "/runtime/human-audio-stream"
            if human_input_enabled
            else ""
        ),
        human_turn_url=(
            f"{endpoint.base_url}/runtime/human-turn" if human_input_enabled else ""
        ),
        human_input_enabled=human_input_enabled,
        start_options=runtime_start_options,
        warmup_target_turns=0 if human_input_enabled else warmup_target_turns,
        warmup_max_wait_ms=0 if human_input_enabled else warmup_max_wait_ms,
        default=None,
        key="runtime_audio_monitor_v12",
    )

    if st.session_state.get("runtime_control_error"):
        st.warning(st.session_state["runtime_control_error"])
    runtime_status = st.session_state.get("runtime_control_status") or {}
    runtime_error_message = str(runtime_status.get("error_message") or "").strip()
    if str(runtime_status.get("phase") or "") == "error" and runtime_error_message:
        st.warning(f"Runtime error: {runtime_error_message}")
    if (
        runtime_status.get("phase") == "paused"
        and runtime_status.get("pause_reason") == "prompt_director_validation"
    ):
        st.warning(
            "応答の再生成を繰り返しましたが、確認を完了できなかったため一時停止しています。"
            "会話の履歴は保持されています。Resume を押すと同じターンから再生成します。"
        )
    if (
        runtime_status.get("phase") == "paused"
        and runtime_status.get("pause_reason") == "prompt_director_transport"
    ):
        st.warning(
            "通信の再試行後も応答を取得できなかったため、一時停止しています。"
            "会話の履歴は保持されています。接続の回復後に Resume を押すと、"
            "同じターンから再生成します。"
        )


def render_runtime_debug_panel(client: RuntimeControlClient) -> None:
    with st.expander("Debug", expanded=False):
        columns = st.columns([2, 1])
        with columns[0]:
            st.text_input("Control host", key="runtime_control_host")
        with columns[1]:
            st.number_input(
                "Control port",
                min_value=1,
                max_value=65535,
                step=1,
                key="runtime_control_port",
            )
        st.caption("Generation mode: Realtime API centered")
        st.markdown("##### 音声設定")
        st.caption("次回セッション開始時")
        pending_audio_settings_rows = st.session_state.get(
            "runtime_pending_audio_settings_rows"
        ) or []
        if pending_audio_settings_rows:
            st.dataframe(
                pending_audio_settings_rows,
                hide_index=True,
                width="stretch",
            )
        else:
            st.info("次回セッションの音声設定はまだ準備されていません。")
        st.markdown("##### 固定システムプロンプト")
        st.caption(
            "プリセットや Session setup の変更では置き換わらない、"
            "AI話者の共通指示です。"
        )
        prompt_columns = st.columns(2)
        with prompt_columns[0]:
            st.text_area(
                "固定カウンセラー・システムプロンプト",
                value=COUNSELOR_SYSTEM_PROMPT,
                height=260,
                disabled=True,
            )
        with prompt_columns[1]:
            st.text_area(
                "固定クライアント・システムプロンプト",
                value=CLIENT_SYSTEM_PROMPT,
                height=260,
                disabled=True,
            )
        st.selectbox(
            "Speaker selection policy",
            list(RUNTIME_SPEAKER_SELECTION_POLICY_OPTIONS),
            key="runtime_speaker_selection_policy",
            format_func=RUNTIME_SPEAKER_SELECTION_POLICY_OPTIONS.get,
        )
        st.dataframe(
            runtime_deferred_interaction_feature_rows(),
            hide_index=True,
            width="stretch",
        )
        endpoint = build_runtime_control_endpoint(
            host=str(st.session_state["runtime_control_host"]),
            port=int(st.session_state["runtime_control_port"]),
            session_id=st.session_state.get("runtime_monitor_session_id") or None,
        )

        session_id = st.text_input(
            "Runtime session_id",
            key="runtime_monitor_session_id",
        )
        if st.button(
            "Fetch events",
            key="runtime_control_fetch_events",
            width="stretch",
        ):
            updates = refresh_runtime_control_events(client=client, session_id=session_id)
            for key, value in updates.items():
                st.session_state[key] = value
            st.session_state.pop(
                "runtime_selected_response_instruction_index",
                None,
            )

        if st.session_state.get("runtime_control_error"):
            st.warning(st.session_state["runtime_control_error"])

        events = st.session_state.get("runtime_control_events") or []
        for event in reversed(events):
            if event.get("event_type") == "ai_routes_resolved":
                routes = event.get("details", {}).get("routes", {})
                st.caption("取得済みセッションの API 送信先")
                st.dataframe(
                    [
                        {
                            "用途": name,
                            "接続先": route.get("provider"),
                            "モデル／デプロイ": route.get("model_ref"),
                        }
                        for name, route in routes.items()
                    ],
                    hide_index=True,
                    width="stretch",
                )
                break
        response_instruction_records = (
            st.session_state.get("runtime_response_instructions") or []
        )
        resolved_audio_settings_rows = runtime_resolved_audio_settings_rows(events)
        if resolved_audio_settings_rows:
            st.caption("取得済みセッションの実効値")
            st.dataframe(
                resolved_audio_settings_rows,
                hide_index=True,
                width="stretch",
            )

        status = st.session_state.get("runtime_control_status") or {}
        if status:
            st.dataframe(
                runtime_status_rows(status),
                hide_index=True,
                width="stretch",
            )
        else:
            st.info("RuntimeControlAPI から status をまだ取得していません。")

        st.markdown("##### 送信済み response.create.instructions")
        st.caption(
            "各 response.create で実際に送信した内部指示です。"
            "公開スクリプトには含まれません。"
        )
        if response_instruction_records:
            st.dataframe(
                [
                    {
                        "turn_id": record.get("turn_id"),
                        "speaker_id": record.get("speaker_id"),
                        "created_at": record.get("created_at"),
                        "sha256": record.get("instructions_sha256"),
                    }
                    for record in response_instruction_records
                    if isinstance(record, dict)
                ],
                hide_index=True,
                width="stretch",
            )
            instruction_indexes = list(
                range(len(response_instruction_records) - 1, -1, -1)
            )
            selected_instruction_index = st.selectbox(
                "表示する送信記録",
                instruction_indexes,
                format_func=lambda index: (
                    f"turn {response_instruction_records[index].get('turn_id')} / "
                    f"{response_instruction_records[index].get('speaker_id')}"
                ),
                key="runtime_selected_response_instruction_index",
            )
            selected_instruction = response_instruction_records[
                selected_instruction_index
            ]
            st.caption(
                "SHA-256: "
                + str(selected_instruction.get("instructions_sha256") or "")
            )
            st.text_area(
                "実際に送信した instructions",
                value=str(
                    selected_instruction.get("resolved_instructions") or ""
                ),
                height=320,
                disabled=True,
            )
            st.text_area(
                "固定セッション指示",
                value=str(
                    selected_instruction.get("session_instructions") or ""
                ),
                height=180,
                disabled=True,
            )
            st.text_area(
                "プリセット／Session setup 指示",
                value=str(
                    selected_instruction.get("participant_instructions") or ""
                ),
                height=180,
                disabled=True,
            )
            st.text_area(
                "ターン固有指示",
                value=str(
                    selected_instruction.get("additional_instructions") or ""
                ),
                height=180,
                disabled=True,
            )
        else:
            st.info("送信済みinstructionsログはありません。")

        if events:
            st.dataframe(events[-100:], hide_index=True, width="stretch")


def render_runtime_observer_panel() -> None:
    defaults = runtime_control_defaults()
    if not st.session_state.get("runtime_control_host"):
        st.session_state["runtime_control_host"] = defaults["host"]
    if not st.session_state.get("runtime_control_port"):
        st.session_state["runtime_control_port"] = int(defaults["port"])
    endpoint = build_runtime_control_endpoint(
        host=str(st.session_state["runtime_control_host"]),
        port=int(st.session_state["runtime_control_port"]),
    )
    client = RuntimeControlClient(endpoint.base_url)
    render_runtime_debug_panel(client)


def render_settings_panel(
    config,
    counselor_profiles,
    client_profiles,
    counselor_presets,
    client_presets,
    themes,
    voice_presets,
) -> None:
    human_client_mode = (
        st.session_state.get("selected_mode") == "ai_counselor_human_client"
    )
    _ = themes, voice_presets
    prepare_runtime_session_defaults(config)
    runtime_tts_voice_options = runtime_voice_options(config)
    if (
        st.session_state.get("runtime_participant_mode")
        not in RUNTIME_PARTICIPANT_MODE_LABELS
    ):
        st.session_state["runtime_participant_mode"] = DEFAULT_RUNTIME_PARTICIPANT_MODE
    with st.expander("Session setup", expanded=False):
        counselor_preset_by_id = {
            preset.preset_id: preset for preset in counselor_presets
        }
        client_preset_by_id = {
            preset.preset_id: preset for preset in client_presets
        }
        counselor_preset_ids = list(counselor_preset_by_id)
        selected_counselor_preset_id = st.session_state.get(
            "selected_counselor_preset_id"
        )
        if selected_counselor_preset_id not in counselor_preset_by_id:
            selected_counselor_preset_id = counselor_preset_ids[0]
            st.session_state["selected_counselor_preset_id"] = (
                selected_counselor_preset_id
            )
        client_preset_ids = list(client_preset_by_id)
        selected_client_preset_id = st.session_state.get("selected_client_preset_id")
        if selected_client_preset_id not in client_preset_by_id:
            selected_client_preset_id = client_preset_ids[0]
            st.session_state["selected_client_preset_id"] = selected_client_preset_id
        force_client_preset_reapply = False
        preset_columns = st.columns(2)
        with preset_columns[0]:
            selected_counselor_preset_id = st.selectbox(
                "カウンセラープリセット",
                counselor_preset_ids,
                format_func=lambda preset_id: counselor_preset_by_id[
                    preset_id
                ].display_name,
                key="selected_counselor_preset_id",
            )
            force_counselor_preset_reapply = st.button(
                "カウンセラープリセットを再適用",
                help="画面上で変更したカウンセラー設定を、選択中のYAMLの初期値へ戻します。",
                key="reapply_counselor_preset",
            )
        if human_client_mode:
            participant_mode = "one_client"
            st.caption("AIカウンセラーと人間クライアント1人で対話します。")
        else:
            with preset_columns[1]:
                selected_client_preset_id = st.selectbox(
                    "クライアントプリセット",
                    client_preset_ids,
                    format_func=lambda preset_id: client_preset_by_id[
                        preset_id
                    ].display_name,
                    key="selected_client_preset_id",
                )
                force_client_preset_reapply = st.button(
                    "クライアントプリセットを再適用",
                    help="画面上で変更したクライアント設定を、選択中のYAMLの初期値へ戻します。",
                    key="reapply_client_preset",
                )
            participant_mode = st.radio(
                "参加クライアント数",
                list(RUNTIME_PARTICIPANT_MODE_LABELS),
                format_func=RUNTIME_PARTICIPANT_MODE_LABELS.get,
                horizontal=True,
                key="runtime_participant_mode",
            )
        if should_apply_counselor_preset(
            st.session_state,
            selected_counselor_preset_id,
            force=force_counselor_preset_reapply,
        ):
            apply_counselor_preset_to_session_state(
                counselor_preset_by_id[selected_counselor_preset_id]
            )
        if not human_client_mode and should_apply_client_preset(
            st.session_state,
            selected_client_preset_id,
            participant_mode,
            force=force_client_preset_reapply,
        ):
            apply_client_preset_to_session_state(
                client_preset_by_id[selected_client_preset_id],
                participant_mode,
            )
        if participant_mode == "two_clients":
            prepare_two_client_runtime_session_defaults(config)
            render_two_client_setup_tab(
                counselor_profiles=counselor_profiles,
                client_profiles=client_profiles,
                runtime_tts_voice_options=runtime_tts_voice_options,
            )
        else:
            render_one_client_setup_tab(
                counselor_profiles=counselor_profiles,
                client_profiles=client_profiles,
                runtime_tts_voice_options=runtime_tts_voice_options,
            )


def mark_runtime_widget_user_modified(widget_key: str, state_key: str) -> None:
    st.session_state[state_key] = st.session_state.get(widget_key)
    st.session_state[f"{state_key}_persist_user_modified"] = True


def mark_runtime_state_user_modified(state_key: str) -> None:
    if state_key == "runtime_force_stop_after_closing_turns":
        active_key = runtime_force_stop_after_closing_turns_state_key()
        st.session_state[active_key] = st.session_state.get(state_key)
        st.session_state[f"{active_key}_persist_user_modified"] = True
    st.session_state[f"{state_key}_persist_user_modified"] = True


def runtime_state_was_user_modified(state_key: str) -> bool:
    return bool(st.session_state.get(f"{state_key}_persist_user_modified"))


def runtime_widget_key(state_key: str) -> str:
    return f"{state_key}_widget"


def sync_runtime_widget_from_state(state_key: str) -> str:
    widget_key = runtime_widget_key(state_key)
    st.session_state[widget_key] = st.session_state.get(state_key)
    return widget_key


def runtime_force_stop_after_closing_turns_state_key(
    participant_mode: str | None = None,
) -> str:
    mode = (
        participant_mode
        or st.session_state.get("runtime_participant_mode")
        or DEFAULT_RUNTIME_PARTICIPANT_MODE
    )
    if str(mode) == "two_clients":
        return "runtime_two_client_force_stop_after_closing_turns"
    return "runtime_one_client_force_stop_after_closing_turns"


def runtime_force_stop_after_closing_turns_default(
    participant_mode: str | None = None,
) -> int:
    mode = (
        participant_mode
        or st.session_state.get("runtime_participant_mode")
        or DEFAULT_RUNTIME_PARTICIPANT_MODE
    )
    if str(mode) == "two_clients":
        return DEFAULT_RUNTIME_TWO_CLIENT_FORCE_STOP_AFTER_CLOSING_TURNS
    return DEFAULT_RUNTIME_ONE_CLIENT_FORCE_STOP_AFTER_CLOSING_TURNS


def runtime_force_stop_after_closing_turns_legacy_defaults(
    participant_mode: str | None = None,
) -> set[int]:
    mode = (
        participant_mode
        or st.session_state.get("runtime_participant_mode")
        or DEFAULT_RUNTIME_PARTICIPANT_MODE
    )
    if str(mode) == "two_clients":
        return {
            LEGACY_RUNTIME_TWO_CLIENT_FORCE_STOP_AFTER_CLOSING_TURNS,
            LEGACY_RUNTIME_ONE_CLIENT_FORCE_STOP_AFTER_CLOSING_TURNS,
        }
    return {
        LEGACY_RUNTIME_ONE_CLIENT_FORCE_STOP_AFTER_CLOSING_TURNS,
        LEGACY_RUNTIME_TWO_CLIENT_FORCE_STOP_AFTER_CLOSING_TURNS,
    }


def runtime_force_stop_after_closing_turns_for_mode(
    participant_mode: str | None = None,
) -> int | None:
    key = runtime_force_stop_after_closing_turns_state_key(participant_mode)
    value = st.session_state.get(key)
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return value
    legacy_value = st.session_state.get("runtime_force_stop_after_closing_turns")
    if (
        isinstance(legacy_value, int)
        and not isinstance(legacy_value, bool)
        and legacy_value > 0
    ):
        return legacy_value
    return None


def sync_runtime_force_stop_after_closing_turns_for_mode(
    participant_mode: str | None = None,
) -> None:
    key = runtime_force_stop_after_closing_turns_state_key(participant_mode)
    value = runtime_force_stop_after_closing_turns_for_mode(participant_mode)
    if value is None:
        value = runtime_force_stop_after_closing_turns_default(participant_mode)
    st.session_state[key] = value
    st.session_state["runtime_force_stop_after_closing_turns"] = value


def render_one_client_setup_tab(
    *,
    counselor_profiles: list[Profile],
    client_profiles: list[Profile],
    runtime_tts_voice_options: list[str],
) -> None:
    human_counselor_mode = (
        st.session_state.get("selected_mode") == "human_counselor_ai_client"
    )
    human_client_mode = (
        st.session_state.get("selected_mode") == "ai_counselor_human_client"
    )
    st.session_state.setdefault("human_client_display_name", "クライアント")
    st.session_state.setdefault("human_client_shared_information", "")
    current_counselor = select_profile_by_id(
        counselor_profiles,
        st.session_state["selected_counselor_profile_id"],
    )
    current_client = select_profile_by_id(
        client_profiles,
        st.session_state["selected_client_profile_id"],
    )
    st.session_state["selected_client_profile_id"] = current_client.profile_id
    if not human_client_mode:
        prepare_one_client_runtime_widget_defaults(runtime_tts_voice_options)
    sync_runtime_force_stop_after_closing_turns_for_mode("one_client")
    prepare_runtime_text_defaults(current_counselor, current_client)

    columns = st.columns(2)
    with columns[0]:
        if not human_counselor_mode and not human_client_mode:
            st.text_input(
                "初回クライアント発話",
                key=sync_runtime_widget_from_state("runtime_initial_client_transcript"),
                on_change=mark_runtime_widget_user_modified,
                args=(
                    runtime_widget_key("runtime_initial_client_transcript"),
                    "runtime_initial_client_transcript",
                ),
            )
        voice_columns = st.columns(2)
        if not human_counselor_mode:
            with voice_columns[0]:
                st.selectbox(
                    "カウンセラー音声",
                    runtime_tts_voice_options,
                    key="runtime_counselor_tts_voice",
                )
        if not human_client_mode:
            with voice_columns[1]:
                st.selectbox(
                    "クライアント音声",
                    runtime_tts_voice_options,
                    key=sync_runtime_widget_from_state("runtime_client_tts_voice"),
                    on_change=mark_runtime_widget_user_modified,
                    args=(
                        runtime_widget_key("runtime_client_tts_voice"),
                        "runtime_client_tts_voice",
                    ),
                )
        speed_columns = st.columns(2)
        if not human_counselor_mode:
            with speed_columns[0]:
                st.slider(
                    "カウンセラー話速",
                    min_value=MIN_REALTIME_OUTPUT_SPEED,
                    max_value=MAX_REALTIME_OUTPUT_SPEED,
                    step=0.05,
                    key=sync_runtime_widget_from_state(
                        "runtime_counselor_output_speed"
                    ),
                    on_change=mark_runtime_widget_user_modified,
                    args=(
                        runtime_widget_key("runtime_counselor_output_speed"),
                        "runtime_counselor_output_speed",
                    ),
                )
        if not human_client_mode:
            with speed_columns[1]:
                st.slider(
                    "クライアント話速",
                    min_value=MIN_REALTIME_OUTPUT_SPEED,
                    max_value=MAX_REALTIME_OUTPUT_SPEED,
                    step=0.05,
                    key=sync_runtime_widget_from_state("runtime_client_output_speed"),
                    on_change=mark_runtime_widget_user_modified,
                    args=(
                        runtime_widget_key("runtime_client_output_speed"),
                        "runtime_client_output_speed",
                    ),
                )
        gain_columns = st.columns(2)
        if not human_counselor_mode:
            with gain_columns[0]:
                st.slider(
                    "カウンセラー音量",
                    min_value=0.1,
                    max_value=4.0,
                    step=0.1,
                    key=sync_runtime_widget_from_state("runtime_counselor_audio_gain"),
                    on_change=mark_runtime_widget_user_modified,
                    args=(
                        runtime_widget_key("runtime_counselor_audio_gain"),
                        "runtime_counselor_audio_gain",
                    ),
                )
        if not human_client_mode:
            with gain_columns[1]:
                st.slider(
                    "クライアント音量",
                    min_value=0.1,
                    max_value=4.0,
                    step=0.1,
                    key=sync_runtime_widget_from_state("runtime_client_audio_gain"),
                    on_change=mark_runtime_widget_user_modified,
                    args=(
                        runtime_widget_key("runtime_client_audio_gain"),
                        "runtime_client_audio_gain",
                    ),
                )
        render_runtime_model_session_controls()
    with columns[1]:
        if not human_counselor_mode:
            st.text_input("カウンセラー表示名", key="counselor_display_name")
        if not human_client_mode:
            st.text_input(
                "クライアント表示名",
                key=sync_runtime_widget_from_state("client_display_name"),
                on_change=mark_runtime_widget_user_modified,
                args=(runtime_widget_key("client_display_name"), "client_display_name"),
            )
        if human_client_mode:
            st.text_input("人間クライアント表示名", key="human_client_display_name")
            st.text_area(
                "AIに事前共有する情報（任意）",
                key="human_client_shared_information",
                height=120,
                help="空欄でも開始できます。ここに入力した内容だけを事前情報としてAIに共有します。",
            )
        render_runtime_common_session_controls()

    st.markdown("##### プロフィール")
    profile_columns = st.columns(2)
    selected_counselor = current_counselor
    selected_client = current_client
    counselor_content_source_id = active_counselor_content_source_id(
        selected_counselor.profile_id
    )
    client_content_source_id = active_client_content_source_id(
        selected_client.profile_id
    )
    prepare_runtime_text_defaults(selected_counselor, selected_client)
    if not human_counselor_mode:
        with profile_columns[0]:
            render_prompt_file_loader(
                label="カウンセラー公開プロフィール読込（.md/.txt）",
                upload_key="counselor_public_profile_file_upload",
                prompt_text_key="counselor_public_profile_text",
                source_profile_key="counselor_public_profile_source_profile_id",
                file_signature_key="counselor_public_profile_file_signature",
                source_profile_id=counselor_content_source_id,
            )
            st.text_area(
                "カウンセラー公開プロフィール",
                key="counselor_public_profile_text",
                height=120,
            )
    if not human_client_mode:
        with profile_columns[1]:
            render_prompt_file_loader(
                label="クライアント公開プロフィール読込（.md/.txt）",
                upload_key="client_public_profile_file_upload",
                prompt_text_key="client_public_profile_text",
                source_profile_key="client_public_profile_source_profile_id",
                file_signature_key="client_public_profile_file_signature",
                source_profile_id=client_content_source_id,
            )
            st.text_area(
                "クライアント公開プロフィール",
                key="client_public_profile_text",
                height=120,
            )
            render_prompt_file_loader(
                label="クライアント秘密プロフィール読込（.md/.txt）",
                upload_key="client_private_profile_file_upload",
                prompt_text_key="client_private_profile_text",
                source_profile_key="client_private_profile_source_profile_id",
                file_signature_key="client_private_profile_file_signature",
                source_profile_id=client_content_source_id,
            )
            st.text_area(
                "クライアント秘密プロフィール",
                key="client_private_profile_text",
                height=120,
            )

    st.markdown("##### プロンプト本文")
    prompt_columns = st.columns(2)
    with prompt_columns[0]:
        if not human_counselor_mode:
            render_prompt_file_loader(
                label="カウンセラー役プロンプト読込（.md/.txt）",
                upload_key="counselor_prompt_file_upload",
                prompt_text_key="counselor_prompt_text",
                source_profile_key="counselor_prompt_source_profile_id",
                file_signature_key="counselor_prompt_file_signature",
                source_profile_id=counselor_content_source_id,
            )
    if not human_counselor_mode:
        st.text_area(
            "カウンセラー役プロンプト",
            key=sync_runtime_widget_from_state("counselor_prompt_text"),
            height=220,
            on_change=mark_runtime_widget_user_modified,
            args=(
                runtime_widget_key("counselor_prompt_text"),
                "counselor_prompt_text",
            ),
        )
    if not human_client_mode:
        with prompt_columns[1]:
            render_prompt_file_loader(
                label="クライアント役プロンプト読込（.md/.txt）",
                upload_key="client_prompt_file_upload",
                prompt_text_key="client_prompt_text",
                source_profile_key="client_prompt_source_profile_id",
                file_signature_key="client_prompt_file_signature",
                source_profile_id=client_content_source_id,
            )
            st.text_area(
                "クライアント役プロンプト",
                key=sync_runtime_widget_from_state("client_prompt_text"),
                height=220,
                on_change=mark_runtime_widget_user_modified,
                args=(runtime_widget_key("client_prompt_text"), "client_prompt_text"),
            )


def render_two_client_setup_tab(
    *,
    counselor_profiles: list[Profile],
    client_profiles: list[Profile],
    runtime_tts_voice_options: list[str],
) -> None:
    human_counselor_mode = (
        st.session_state.get("selected_mode") == "human_counselor_ai_client"
    )
    current_counselor = select_profile_by_id(
        counselor_profiles,
        st.session_state["selected_counselor_profile_id"],
    )
    current_client = select_profile_by_id(
        client_profiles,
        st.session_state["selected_client_profile_id"],
    )
    st.session_state["selected_client_profile_id"] = current_client.profile_id
    prepare_runtime_text_defaults(current_counselor, current_client)

    if not human_counselor_mode:
        st.markdown("##### カウンセラー設定")
        columns = st.columns(2)
        with columns[0]:
            st.selectbox(
                "カウンセラー音声",
                runtime_tts_voice_options,
                key="runtime_counselor_tts_voice",
            )
            counselor_audio_columns = st.columns(2)
            with counselor_audio_columns[0]:
                st.slider(
                    "カウンセラー話速",
                    min_value=MIN_REALTIME_OUTPUT_SPEED,
                    max_value=MAX_REALTIME_OUTPUT_SPEED,
                    step=0.05,
                    key=sync_runtime_widget_from_state(
                        "runtime_counselor_output_speed"
                    ),
                    on_change=mark_runtime_widget_user_modified,
                    args=(
                        runtime_widget_key("runtime_counselor_output_speed"),
                        "runtime_counselor_output_speed",
                    ),
                )
            with counselor_audio_columns[1]:
                st.slider(
                    "カウンセラー音量",
                    min_value=0.1,
                    max_value=4.0,
                    step=0.1,
                    key=sync_runtime_widget_from_state("runtime_counselor_audio_gain"),
                    on_change=mark_runtime_widget_user_modified,
                    args=(
                        runtime_widget_key("runtime_counselor_audio_gain"),
                        "runtime_counselor_audio_gain",
                    ),
                )
            render_runtime_model_session_controls()
        with columns[1]:
            st.text_input("カウンセラー表示名", key="counselor_display_name")
            render_runtime_common_session_controls()
    else:
        st.markdown("##### セッション設定")
        columns = st.columns(2)
        with columns[0]:
            render_runtime_model_session_controls()
        with columns[1]:
            render_runtime_common_session_controls()

    st.markdown("##### プロフィール")
    profile_columns = st.columns(2)
    selected_counselor = current_counselor
    selected_client = current_client
    counselor_content_source_id = active_counselor_content_source_id(
        selected_counselor.profile_id
    )
    client_content_source_id = active_client_content_source_id(
        selected_client.profile_id
    )
    prepare_runtime_text_defaults(selected_counselor, selected_client)
    prepare_two_client_runtime_widget_defaults(
        selected_client=selected_client,
        runtime_tts_voice_options=runtime_tts_voice_options,
    )
    if not human_counselor_mode:
        with profile_columns[0]:
            render_prompt_file_loader(
                label="カウンセラー公開プロフィール読込（.md/.txt）",
                upload_key="counselor_public_profile_file_upload",
                prompt_text_key="counselor_public_profile_text",
                source_profile_key="counselor_public_profile_source_profile_id",
                file_signature_key="counselor_public_profile_file_signature",
                source_profile_id=counselor_content_source_id,
            )
            st.text_area(
                "カウンセラー公開プロフィール",
                key="counselor_public_profile_text",
                height=120,
            )
    with profile_columns[1]:
        render_prompt_file_loader(
            label="クライアント公開プロフィール読込（.md/.txt）",
            upload_key="client_public_profile_file_upload",
            prompt_text_key="client_public_profile_text",
            source_profile_key="client_public_profile_source_profile_id",
            file_signature_key="client_public_profile_file_signature",
            source_profile_id=client_content_source_id,
        )
        st.text_area(
            "クライアント公開プロフィール",
            key="client_public_profile_text",
            height=120,
        )

    st.markdown("##### プロンプト本文")
    prompt_columns = st.columns(2)
    with prompt_columns[0]:
        if not human_counselor_mode:
            render_prompt_file_loader(
                label="カウンセラー役プロンプト読込（.md/.txt）",
                upload_key="counselor_prompt_file_upload",
                prompt_text_key="counselor_prompt_text",
                source_profile_key="counselor_prompt_source_profile_id",
                file_signature_key="counselor_prompt_file_signature",
                source_profile_id=counselor_content_source_id,
            )
    if not human_counselor_mode:
        st.text_area(
            "カウンセラー役プロンプト",
            key=sync_runtime_widget_from_state("counselor_prompt_text"),
            height=220,
            on_change=mark_runtime_widget_user_modified,
            args=(
                runtime_widget_key("counselor_prompt_text"),
                "counselor_prompt_text",
            ),
        )
    with prompt_columns[1]:
        render_prompt_file_loader(
            label="クライアント共通プロンプト読込（.md/.txt）",
            upload_key="client_prompt_file_upload",
            prompt_text_key="client_prompt_text",
            source_profile_key="client_prompt_source_profile_id",
            file_signature_key="client_prompt_file_signature",
            source_profile_id=client_content_source_id,
        )
        st.text_area(
            "クライアント共通プロンプト",
            key="client_prompt_text",
            height=220,
        )
    st.markdown("##### クライアント設定")
    render_two_client_settings_panel(
        selected_client=selected_client,
        runtime_tts_voice_options=runtime_tts_voice_options,
    )


def prepare_runtime_provider_defaults() -> dict[str, str]:
    if "runtime_realtime_provider_kinds" not in st.session_state:
        ai = effective_ai_settings(load_runtime_config())
        route = ai.routes["realtime_speech"]
        st.session_state["runtime_realtime_provider_kinds"] = {
            provider_id: ai.providers[provider_id].kind for provider_id in route.targets
        }
        st.session_state.setdefault(
            "runtime_realtime_provider",
            route.provider or ai.default_provider,
        )
        for provider_id, target in route.targets.items():
            value = (
                target.model
                if ai.providers[provider_id].kind == "openai"
                else target.deployment
            )
            st.session_state.setdefault(
                f"runtime_realtime_target_{provider_id}", value or ""
            )
    return st.session_state["runtime_realtime_provider_kinds"]


def prepare_runtime_text_provider_defaults() -> dict[str, dict[str, str]]:
    if "runtime_text_provider_kinds" not in st.session_state:
        ai = effective_ai_settings(load_runtime_config())
        kinds = {}
        for name, (legacy_prefix, _) in RUNTIME_TEXT_ROUTE_CONTROLS.items():
            route = ai.routes[name]
            kinds[name] = {
                provider_id: ai.providers[provider_id].kind
                for provider_id in route.targets
            }
            st.session_state.setdefault(
                f"runtime_{name}_provider", route.provider or ai.default_provider
            )
            for provider_id, target in route.targets.items():
                value = target.deployment
                if ai.providers[provider_id].kind == "openai":
                    value = (
                        st.session_state.get(f"runtime_{legacy_prefix}_llm_model")
                        or target.model
                    )
                st.session_state.setdefault(
                    f"runtime_{name}_target_{provider_id}", value or ""
                )
        st.session_state["runtime_text_provider_kinds"] = kinds
    return st.session_state["runtime_text_provider_kinds"]


def prepare_runtime_stt_provider_defaults() -> dict[str, str]:
    if "runtime_stt_provider_kinds" not in st.session_state:
        ai = effective_ai_settings(load_runtime_config())
        route = ai.routes["realtime_transcription"]
        st.session_state["runtime_stt_provider_kinds"] = {
            provider_id: ai.providers[provider_id].kind for provider_id in route.targets
        }
        st.session_state.setdefault(
            "runtime_stt_provider", route.provider or ai.default_provider
        )
        for provider_id, target in route.targets.items():
            value = (
                target.model
                if ai.providers[provider_id].kind == "openai"
                else target.deployment
            )
            st.session_state.setdefault(
                f"runtime_stt_target_{provider_id}", value or ""
            )
    return st.session_state["runtime_stt_provider_kinds"]


def render_runtime_stt_controls() -> str:
    kinds = prepare_runtime_stt_provider_defaults()
    provider_key = "runtime_stt_provider"
    st.selectbox(
        "人間のマイク文字起こしの接続先",
        list(kinds),
        key=sync_runtime_widget_from_state(provider_key),
        format_func=lambda provider_id: (
            f"Azure OpenAI ({provider_id})"
            if kinds[provider_id] == "azure_openai"
            else f"OpenAI ({provider_id})"
        ),
        on_change=mark_runtime_widget_user_modified,
        args=(runtime_widget_key(provider_key), provider_key),
        help="人間のマイク入力を独立して文字起こしする場合の送信先です。変更は次のセッションから反映されます。",
    )
    provider_id = st.session_state[provider_key]
    target_key = f"runtime_stt_target_{provider_id}"
    st.text_input(
        (
            "Azure 文字起こしデプロイ名"
            if kinds[provider_id] == "azure_openai"
            else "文字起こし model"
        ),
        key=sync_runtime_widget_from_state(target_key),
        on_change=mark_runtime_widget_user_modified,
        args=(runtime_widget_key(target_key), target_key),
    )
    return provider_id


def render_runtime_model_session_controls() -> None:
    provider_kinds = prepare_runtime_provider_defaults()
    provider_key = "runtime_realtime_provider"
    st.selectbox(
        "Realtime 音声の接続先",
        list(provider_kinds),
        key=sync_runtime_widget_from_state(provider_key),
        format_func=lambda provider_id: (
            f"Azure OpenAI ({provider_id})"
            if provider_kinds[provider_id] == "azure_openai"
            else f"OpenAI ({provider_id})"
        ),
        on_change=mark_runtime_widget_user_modified,
        args=(runtime_widget_key(provider_key), provider_key),
        help="変更は次のセッション開始時に反映されます。",
    )
    provider_id = st.session_state[provider_key]
    target_key = f"runtime_realtime_target_{provider_id}"
    if provider_kinds[provider_id] == "azure_openai":
        st.text_input(
            "Azure Realtime デプロイ名",
            key=sync_runtime_widget_from_state(target_key),
            on_change=mark_runtime_widget_user_modified,
            args=(runtime_widget_key(target_key), target_key),
            help="Azure で作成したデプロイの名前を指定してください。",
        )
    else:
        model_options = list(RUNTIME_REALTIME_MODEL_OPTIONS)
        current_model = st.session_state[target_key]
        if current_model and current_model not in model_options:
            model_options.append(current_model)
        st.selectbox(
            "Realtime model",
            model_options,
            key=sync_runtime_widget_from_state(target_key),
            on_change=mark_runtime_widget_user_modified,
            args=(runtime_widget_key(target_key), target_key),
        )
    stt_provider_id = render_runtime_stt_controls()
    st.caption(
        f"次セッションの送信先：Realtime 音声（セッション内の文字起こしを含む）は {provider_id}。"
        f"人間のマイク入力の独立した文字起こしは {stt_provider_id} です。"
    )
    text_provider_kinds = prepare_runtime_text_provider_defaults()
    for name, (legacy_prefix, label) in RUNTIME_TEXT_ROUTE_CONTROLS.items():
        if (
            name == "turn_timing"
            and st.session_state.get("selected_mode") == "ai_counselor_human_client"
        ):
            continue
        kinds = text_provider_kinds[name]
        provider_key = f"runtime_{name}_provider"
        st.selectbox(
            f"{label}の接続先",
            list(kinds),
            key=sync_runtime_widget_from_state(provider_key),
            format_func=lambda provider_id, kinds=kinds: (
                f"Azure OpenAI ({provider_id})"
                if kinds[provider_id] == "azure_openai"
                else f"OpenAI ({provider_id})"
            ),
            on_change=mark_runtime_widget_user_modified,
            args=(runtime_widget_key(provider_key), provider_key),
            help="変更は次のセッション開始時に反映されます。",
        )
        provider_id = st.session_state[provider_key]
        target_key = f"runtime_{name}_target_{provider_id}"
        model_column, reasoning_column = st.columns(2)
        with model_column:
            widget_args = {
                "key": sync_runtime_widget_from_state(target_key),
                "on_change": mark_runtime_widget_user_modified,
                "args": (runtime_widget_key(target_key), target_key),
            }
            if kinds[provider_id] == "azure_openai":
                st.text_input(f"{label} Azure デプロイ名", **widget_args)
            else:
                model_options = list(RUNTIME_TIMING_LLM_MODEL_OPTIONS)
                current_model = st.session_state[target_key]
                if current_model and current_model not in model_options:
                    model_options.append(current_model)
                st.selectbox(f"{label} model", model_options, **widget_args)
        with reasoning_column:
            st.selectbox(
                f"{label} reasoning",
                list(RUNTIME_TIMING_REASONING_EFFORT_OPTIONS),
                key=f"runtime_{legacy_prefix}_llm_reasoning_effort",
                format_func=RUNTIME_TIMING_REASONING_EFFORT_OPTIONS.get,
            )


def render_runtime_common_session_controls() -> None:
    st.number_input(
        "クロージング開始（秒）",
        min_value=1,
        step=1,
        key="runtime_closing_start_seconds",
        on_change=mark_runtime_state_user_modified,
        args=("runtime_closing_start_seconds",),
    )
    st.number_input(
        "クロージング最大ターン数（カウンセラー開始）",
        min_value=1,
        max_value=20,
        step=1,
        key="runtime_force_stop_after_closing_turns",
        on_change=mark_runtime_state_user_modified,
        args=("runtime_force_stop_after_closing_turns",),
    )
    if st.session_state.get("selected_mode") == "ai_counselor_human_client":
        st.caption(
            "AIが挨拶した後、マイクで話すと応答します。AIの発話中も割り込めます。"
        )
        return
    st.number_input(
        "先行生成上限（表示との差）",
        min_value=1,
        max_value=10,
        step=1,
        help=(
            "開始時は指定数の発話の音声生成が完了してから再生します。"
            "1ターンは1人の発話で、指定数が多いほど開始まで待ちます。"
            "再生中は未再生の生成完了数が上限に達すると生成を自動Pauseし、"
            "上限-2以下でResumeします（上限7なら5以下）。"
            "生成中の発話は数えず、再生中に常に指定数を保つ設定ではありません。"
        ),
        key="runtime_generation_lead_limit",
    )


def prepare_one_client_runtime_widget_defaults(
    runtime_tts_voice_options: list[str],
) -> None:
    if client_preset_is_active("one_client"):
        return
    default_voice = (
        DEFAULT_RUNTIME_CLIENT_A_TTS_VOICE
        if DEFAULT_RUNTIME_CLIENT_A_TTS_VOICE in runtime_tts_voice_options
        else runtime_tts_voice_options[0]
        if runtime_tts_voice_options
        else DEFAULT_RUNTIME_CLIENT_A_TTS_VOICE
    )

    display_name_value = st.session_state.get("client_display_name")
    stale_display_names = {
        "",
        DEFAULT_RUNTIME_CLIENT_A_DISPLAY_NAME,
        DEFAULT_RUNTIME_CLIENT_B_DISPLAY_NAME,
        LEGACY_RUNTIME_CLIENT_DISPLAY_NAME,
        "クライアントA",
        "クライアントB",
    }
    if not display_name_value or (
        not runtime_state_was_user_modified("client_display_name")
        and display_name_value in stale_display_names
    ):
        st.session_state["client_display_name"] = DEFAULT_RUNTIME_CLIENT_DISPLAY_NAME

    voice_value = st.session_state.get("runtime_client_tts_voice")
    stale_voice_values = {"", DEFAULT_RUNTIME_CLIENT_B_TTS_VOICE}
    if runtime_tts_voice_options:
        stale_voice_values.add(runtime_tts_voice_options[0])
    voice_widget_exists = (
        runtime_widget_key("runtime_client_tts_voice") in st.session_state
    )
    voice_is_valid = voice_value in runtime_tts_voice_options
    if not voice_is_valid or (
        (
            not runtime_state_was_user_modified("runtime_client_tts_voice")
            or not voice_widget_exists
        )
        and voice_value in stale_voice_values
    ):
        st.session_state["runtime_client_tts_voice"] = default_voice

    speed_value = st.session_state.get("runtime_client_output_speed")
    speed_widget_exists = (
        runtime_widget_key("runtime_client_output_speed") in st.session_state
    )
    speed_is_valid = (
        isinstance(speed_value, (int, float))
        and not isinstance(speed_value, bool)
        and MIN_REALTIME_OUTPUT_SPEED
        <= float(speed_value)
        <= MAX_REALTIME_OUTPUT_SPEED
    )
    if not speed_is_valid or (
        (
            not runtime_state_was_user_modified("runtime_client_output_speed")
            or not speed_widget_exists
        )
        and float(speed_value)
        in {0.25, DEFAULT_RUNTIME_CLIENT_B_OUTPUT_SPEED}
    ):
        st.session_state["runtime_client_output_speed"] = (
            DEFAULT_RUNTIME_CLIENT_OUTPUT_SPEED
        )

    gain_value = st.session_state.get("runtime_client_audio_gain")
    gain_widget_exists = (
        runtime_widget_key("runtime_client_audio_gain") in st.session_state
    )
    gain_is_valid = (
        isinstance(gain_value, (int, float))
        and not isinstance(gain_value, bool)
        and float(gain_value) > 0
    )
    if not gain_is_valid or (
        (
            not runtime_state_was_user_modified("runtime_client_audio_gain")
            or not gain_widget_exists
        )
        and float(gain_value) == 0.1
    ):
        st.session_state["runtime_client_audio_gain"] = DEFAULT_RUNTIME_CLIENT_AUDIO_GAIN

    closing_value = st.session_state.get(
        "runtime_one_client_force_stop_after_closing_turns"
    )
    closing_is_valid = (
        isinstance(closing_value, int)
        and not isinstance(closing_value, bool)
        and closing_value > 0
    )
    closing_is_legacy_default = (
        closing_is_valid
        and closing_value
        in runtime_force_stop_after_closing_turns_legacy_defaults("one_client")
    )
    if not runtime_state_was_user_modified(
        "runtime_one_client_force_stop_after_closing_turns"
    ) and (
        not closing_is_valid
        or closing_is_legacy_default
    ):
        st.session_state["runtime_one_client_force_stop_after_closing_turns"] = (
            DEFAULT_RUNTIME_ONE_CLIENT_FORCE_STOP_AFTER_CLOSING_TURNS
        )
    elif not closing_is_valid:
        st.session_state["runtime_one_client_force_stop_after_closing_turns"] = (
            DEFAULT_RUNTIME_ONE_CLIENT_FORCE_STOP_AFTER_CLOSING_TURNS
        )
    if (
        st.session_state.get("runtime_force_stop_after_closing_turns")
        in runtime_force_stop_after_closing_turns_legacy_defaults("one_client")
        and not runtime_state_was_user_modified(
            "runtime_one_client_force_stop_after_closing_turns"
        )
    ):
        st.session_state["runtime_force_stop_after_closing_turns"] = (
            DEFAULT_RUNTIME_ONE_CLIENT_FORCE_STOP_AFTER_CLOSING_TURNS
        )
    st.session_state["runtime_force_stop_after_closing_turns"] = (
        st.session_state["runtime_one_client_force_stop_after_closing_turns"]
    )
    st.session_state["runtime_one_client_widget_defaults_version"] = (
        ONE_CLIENT_WIDGET_DEFAULTS_VERSION
    )


def prepare_two_client_runtime_widget_defaults(
    *,
    selected_client: Profile,
    runtime_tts_voice_options: list[str],
) -> None:
    if client_preset_is_active("two_clients"):
        return
    two_client_defaults = two_client_attachment_defaults()
    common_prompt_value = st.session_state.get("client_prompt_text")
    last_common_prompt_for_migration = st.session_state.get(
        "client_common_profile_applied_prompt_text"
    )
    try:
        previous_defaults_version = int(
            st.session_state.get("runtime_two_client_widget_defaults_version") or 0
        )
    except (TypeError, ValueError):
        previous_defaults_version = 0
    if (
        previous_defaults_version < TWO_CLIENT_WIDGET_DEFAULTS_VERSION
        and isinstance(common_prompt_value, str)
        and isinstance(last_common_prompt_for_migration, str)
        and common_prompt_value == last_common_prompt_for_migration
        and "夫婦が互いの発言を聞き" not in common_prompt_value
    ):
        common_prompt_value = two_client_defaults["common_prompt"]
        st.session_state["client_prompt_text"] = common_prompt_value
    if (
        not isinstance(common_prompt_value, str)
        or not common_prompt_value.strip()
        or common_prompt_value == DEFAULT_RUNTIME_CLIENT_PROMPT
    ):
        st.session_state["client_prompt_text"] = two_client_defaults["common_prompt"]
    common_profile_value = st.session_state.get("client_public_profile_text")
    if (
        not isinstance(common_profile_value, str)
        or not common_profile_value.strip()
        or common_profile_value == selected_client.public_profile
    ):
        st.session_state["client_public_profile_text"] = two_client_defaults[
            "common_profile"
        ]
    prompt_text = str(
        st.session_state.get("client_prompt_text")
        or two_client_defaults["common_prompt"]
    )
    client_public_profile_value = st.session_state.get("client_public_profile_text")
    if isinstance(client_public_profile_value, str):
        public_profile_text = client_public_profile_value
    else:
        public_profile_text = two_client_defaults["common_profile"]
    initial_transcript = str(
        st.session_state.get("runtime_initial_client_transcript") or ""
    )
    legacy_initial_transcript_defaults = {
        value
        for value in (
            DEFAULT_RUNTIME_INITIAL_CLIENT_TRANSCRIPT,
            _runtime_initial_client_transcript_default(),
        )
        if value
    }
    client_voice = str(st.session_state.get("runtime_client_tts_voice") or "")
    if client_voice not in runtime_tts_voice_options:
        client_voice = runtime_tts_voice_options[0]
    client_speed = st.session_state.get("runtime_client_output_speed")
    if (
        not isinstance(client_speed, (int, float))
        or isinstance(client_speed, bool)
        or not MIN_REALTIME_OUTPUT_SPEED
        <= client_speed
        <= MAX_REALTIME_OUTPUT_SPEED
    ):
        client_speed = DEFAULT_RUNTIME_CLIENT_OUTPUT_SPEED
    client_gain = st.session_state.get("runtime_client_audio_gain")
    if (
        not isinstance(client_gain, (int, float))
        or isinstance(client_gain, bool)
        or client_gain <= 0
    ):
        client_gain = DEFAULT_RUNTIME_CLIENT_AUDIO_GAIN
    last_common_source = st.session_state.get(
        "client_common_profile_applied_source_profile_id"
    )
    last_common_prompt = st.session_state.get(
        "client_common_profile_applied_prompt_text"
    )
    last_common_public = st.session_state.get(
        "client_common_profile_applied_public_profile_text"
    )
    common_source_changed = last_common_source != selected_client.profile_id
    common_prompt_changed = last_common_prompt != prompt_text
    common_public_changed = last_common_public != public_profile_text

    for settings in [
        {
            "display_name_key": "client_a_display_name",
            "display_name": DEFAULT_RUNTIME_CLIENT_A_DISPLAY_NAME,
            "stale_display_names": ("クライアントA",),
            "public_text_key": "client_a_public_profile_text",
            "public_source_key": "client_a_public_profile_source_profile_id",
            "default_profile": two_client_defaults["client_a_profile"],
            "prompt_text_key": "client_a_prompt_text",
            "prompt_source_key": "client_a_prompt_source_profile_id",
            "default_prompt": two_client_defaults["client_a_prompt"],
            "common_prompt_intro": TWO_CLIENT_A_PROMPT_INTRO,
            "obsolete_label_instruction": "各発言の先頭に「妻：」と付けてください。",
            "label_instruction": "各発言の先頭に「妻：」「妻、」などの話者名や役割名を付けないでください。",
            "initial_transcript_key": "runtime_initial_client_a_transcript",
            "default_initial_transcript": two_client_defaults[
                "client_a_initial_transcript"
            ],
            "voice_key": "runtime_client_a_tts_voice",
            "default_voice": DEFAULT_RUNTIME_CLIENT_A_TTS_VOICE,
            "speed_key": "runtime_client_a_output_speed",
            "default_speed": DEFAULT_RUNTIME_CLIENT_A_OUTPUT_SPEED,
            "gain_key": "runtime_client_a_audio_gain",
            "default_gain": DEFAULT_RUNTIME_CLIENT_AUDIO_GAIN,
        },
        {
            "display_name_key": "client_b_display_name",
            "display_name": DEFAULT_RUNTIME_CLIENT_B_DISPLAY_NAME,
            "stale_display_names": ("クライアントB",),
            "public_text_key": "client_b_public_profile_text",
            "public_source_key": "client_b_public_profile_source_profile_id",
            "default_profile": two_client_defaults["client_b_profile"],
            "prompt_text_key": "client_b_prompt_text",
            "prompt_source_key": "client_b_prompt_source_profile_id",
            "default_prompt": two_client_defaults["client_b_prompt"],
            "common_prompt_intro": TWO_CLIENT_B_PROMPT_INTRO,
            "obsolete_label_instruction": "各発言の先頭に「夫：」と付けてください。",
            "label_instruction": "各発言の先頭に「夫：」「夫、」などの話者名や役割名を付けないでください。",
            "initial_transcript_key": "runtime_initial_client_b_transcript",
            "default_initial_transcript": two_client_defaults[
                "client_b_initial_transcript"
            ],
            "voice_key": "runtime_client_b_tts_voice",
            "default_voice": DEFAULT_RUNTIME_CLIENT_B_TTS_VOICE,
            "speed_key": "runtime_client_b_output_speed",
            "default_speed": DEFAULT_RUNTIME_CLIENT_B_OUTPUT_SPEED,
            "gain_key": "runtime_client_b_audio_gain",
            "default_gain": DEFAULT_RUNTIME_CLIENT_AUDIO_GAIN,
        },
    ]:
        display_name_value = st.session_state.get(settings["display_name_key"])
        display_name_user_modified = runtime_state_was_user_modified(
            settings["display_name_key"]
        )
        if not display_name_user_modified and (
            not display_name_value
            or display_name_value in settings["stale_display_names"]
        ):
            st.session_state[settings["display_name_key"]] = settings["display_name"]
            st.session_state[f"{settings['display_name_key']}_default_applied"] = True
        elif display_name_value == settings["display_name"]:
            st.session_state[f"{settings['display_name_key']}_default_applied"] = True
        participant_common_prompt = _participant_common_prompt(
            prompt_text,
            str(settings["common_prompt_intro"]),
        )
        participant_prompt_default = _join_two_client_default_parts(
            participant_common_prompt,
            str(settings["default_prompt"]),
        )
        last_participant_prompt_default = (
            _join_two_client_default_parts(
                _participant_common_prompt(
                    str(last_common_prompt),
                    str(settings["common_prompt_intro"]),
                ),
                str(settings["default_prompt"]),
            )
            if isinstance(last_common_prompt, str)
            else None
        )
        participant_prompt_text = st.session_state.get(settings["prompt_text_key"])
        if isinstance(participant_prompt_text, str):
            migrated_prompt_text = participant_prompt_text.replace(
                str(settings["obsolete_label_instruction"]),
                str(settings["label_instruction"]),
            )
            if migrated_prompt_text != participant_prompt_text:
                participant_prompt_text = migrated_prompt_text
                st.session_state[settings["prompt_text_key"]] = migrated_prompt_text
        stale_prompt_defaults = {
            DEFAULT_RUNTIME_CLIENT_PROMPT,
            str(selected_client.prompt or ""),
        }
        if (
            not isinstance(participant_prompt_text, str)
            or not participant_prompt_text
            or participant_prompt_text in stale_prompt_defaults
        ):
            st.session_state[settings["prompt_text_key"]] = participant_prompt_default
        elif (
            common_prompt_changed
            and participant_prompt_text == last_participant_prompt_default
        ):
            st.session_state[settings["prompt_text_key"]] = participant_prompt_default
        participant_public_default = str(settings["default_profile"]).strip()
        last_participant_public_default = (
            _join_two_client_default_parts(
                str(last_common_public),
                str(settings["default_profile"]),
            )
            if isinstance(last_common_public, str)
            else None
        )
        legacy_participant_public_default = _join_two_client_default_parts(
            public_profile_text,
            str(settings["default_profile"]),
        )
        current_participant_public_text = st.session_state.get(
            settings["public_text_key"]
        )
        if (
            st.session_state.get(settings["public_source_key"])
            != selected_client.profile_id
            or common_source_changed
            or current_participant_public_text == legacy_participant_public_default
            or (
                common_public_changed
                and current_participant_public_text == last_participant_public_default
            )
        ):
            st.session_state[settings["public_text_key"]] = participant_public_default
            st.session_state[settings["public_source_key"]] = selected_client.profile_id
        initial_transcript_value = st.session_state.get(
            settings["initial_transcript_key"]
        )
        if not isinstance(initial_transcript_value, str):
            initial_transcript_value = ""
        initial_transcript_value = initial_transcript_value.strip()
        initial_transcript_user_modified = runtime_state_was_user_modified(
            settings["initial_transcript_key"]
        )
        if not initial_transcript_user_modified and (
            not initial_transcript_value
            or initial_transcript_value in legacy_initial_transcript_defaults
        ):
            st.session_state[settings["initial_transcript_key"]] = (
                settings["default_initial_transcript"]
            )
            st.session_state[
                f"{settings['initial_transcript_key']}_default_applied"
            ] = True
        elif initial_transcript_value == settings["default_initial_transcript"]:
            st.session_state[
                f"{settings['initial_transcript_key']}_default_applied"
            ] = True
        voice_user_modified = runtime_state_was_user_modified(settings["voice_key"])
        if st.session_state.get(settings["voice_key"]) not in runtime_tts_voice_options:
            default_voice = str(settings["default_voice"])
            st.session_state[settings["voice_key"]] = (
                default_voice
                if default_voice in runtime_tts_voice_options
                else client_voice
            )
            st.session_state[f"{settings['voice_key']}_default_applied"] = True
        elif (
            not voice_user_modified
            and st.session_state.get(settings["voice_key"])
            == runtime_tts_voice_options[0]
        ):
            default_voice = str(settings["default_voice"])
            st.session_state[settings["voice_key"]] = (
                default_voice
                if default_voice in runtime_tts_voice_options
                else client_voice
            )
            st.session_state[f"{settings['voice_key']}_default_applied"] = True
        elif st.session_state.get(settings["voice_key"]) == settings["default_voice"]:
            st.session_state[f"{settings['voice_key']}_default_applied"] = True
        speed_value = st.session_state.get(settings["speed_key"])
        speed_user_modified = runtime_state_was_user_modified(settings["speed_key"])
        if (
            not isinstance(speed_value, (int, float))
            or isinstance(speed_value, bool)
            or not MIN_REALTIME_OUTPUT_SPEED
            <= speed_value
            <= MAX_REALTIME_OUTPUT_SPEED
        ):
            st.session_state[settings["speed_key"]] = float(
                settings.get("default_speed", client_speed)
            )
            st.session_state[f"{settings['speed_key']}_default_applied"] = True
        elif (
            not speed_user_modified
            and float(speed_value) == 0.25
        ):
            st.session_state[settings["speed_key"]] = float(
                settings.get("default_speed", client_speed)
            )
            st.session_state[f"{settings['speed_key']}_default_applied"] = True
        elif float(speed_value) == float(settings["default_speed"]):
            st.session_state[f"{settings['speed_key']}_default_applied"] = True
        gain_value = st.session_state.get(settings["gain_key"])
        gain_user_modified = runtime_state_was_user_modified(settings["gain_key"])
        if (
            not isinstance(gain_value, (int, float))
            or isinstance(gain_value, bool)
            or gain_value <= 0
        ):
            st.session_state[settings["gain_key"]] = float(
                settings.get("default_gain", client_gain)
            )
            st.session_state[f"{settings['gain_key']}_default_applied"] = True
        elif (
            not gain_user_modified
            and float(gain_value) == 0.1
        ):
            st.session_state[settings["gain_key"]] = float(
                settings.get("default_gain", client_gain)
            )
            st.session_state[f"{settings['gain_key']}_default_applied"] = True
        elif float(gain_value) == float(settings["default_gain"]):
            st.session_state[f"{settings['gain_key']}_default_applied"] = True
    st.session_state["client_common_profile_applied_source_profile_id"] = (
        selected_client.profile_id
    )
    st.session_state["client_common_profile_applied_prompt_text"] = prompt_text
    st.session_state["client_common_profile_applied_public_profile_text"] = (
        public_profile_text
    )
    st.session_state["runtime_two_client_widget_defaults_version"] = (
        TWO_CLIENT_WIDGET_DEFAULTS_VERSION
    )


def _runtime_initial_client_transcript_default() -> str:
    try:
        runtime_settings = load_runtime_config(
            ROOT_DIR / "config" / "runtime_config.yaml"
        )
    except RuntimeConfigLoadError:
        return DEFAULT_RUNTIME_INITIAL_CLIENT_TRANSCRIPT
    return str(runtime_settings.runtime.initial_client_transcript or "").strip()


def render_two_client_settings_panel(
    *,
    selected_client: Profile,
    runtime_tts_voice_options: list[str],
) -> None:
    client_content_source_id = active_client_content_source_id(
        selected_client.profile_id
    )
    client_columns = st.columns(2)
    for column, settings in zip(
        client_columns,
        two_client_settings_specs(),
        strict=False,
    ):
        with column:
            st.markdown(f"###### {settings['label']}")
            st.text_input(
                f"{settings['label']}表示名",
                key=sync_runtime_widget_from_state(settings["display_name_key"]),
                on_change=mark_runtime_widget_user_modified,
                args=(
                    runtime_widget_key(settings["display_name_key"]),
                    settings["display_name_key"],
                ),
            )
            st.text_input(
                f"{settings['label']}初回発話",
                key=sync_runtime_widget_from_state(settings["initial_transcript_key"]),
                on_change=mark_runtime_widget_user_modified,
                args=(
                    runtime_widget_key(settings["initial_transcript_key"]),
                    settings["initial_transcript_key"],
                ),
            )
            st.selectbox(
                f"{settings['label']}音声",
                runtime_tts_voice_options,
                key=sync_runtime_widget_from_state(settings["voice_key"]),
                on_change=mark_runtime_widget_user_modified,
                args=(
                    runtime_widget_key(settings["voice_key"]),
                    settings["voice_key"],
                ),
            )
            audio_columns = st.columns(2)
            with audio_columns[0]:
                st.slider(
                    f"{settings['label']}話速",
                    min_value=MIN_REALTIME_OUTPUT_SPEED,
                    max_value=MAX_REALTIME_OUTPUT_SPEED,
                    step=0.05,
                    key=sync_runtime_widget_from_state(settings["speed_key"]),
                    on_change=mark_runtime_widget_user_modified,
                    args=(
                        runtime_widget_key(settings["speed_key"]),
                        settings["speed_key"],
                    ),
                )
            with audio_columns[1]:
                st.slider(
                    f"{settings['label']}音量",
                    min_value=0.1,
                    max_value=4.0,
                    step=0.1,
                    key=sync_runtime_widget_from_state(settings["gain_key"]),
                    on_change=mark_runtime_widget_user_modified,
                    args=(
                        runtime_widget_key(settings["gain_key"]),
                        settings["gain_key"],
                    ),
                )
            render_prompt_file_loader(
                label=f"{settings['label']}秘密プロフィール読込（.md/.txt）",
                upload_key=settings["public_upload_key"],
                prompt_text_key=settings["public_text_key"],
                source_profile_key=settings["public_source_key"],
                file_signature_key=settings["public_file_signature_key"],
                source_profile_id=client_content_source_id,
            )
            st.text_area(
                f"{settings['label']}秘密プロフィール",
                key=sync_runtime_widget_from_state(settings["public_text_key"]),
                height=120,
                on_change=mark_runtime_widget_user_modified,
                args=(
                    runtime_widget_key(settings["public_text_key"]),
                    settings["public_text_key"],
                ),
            )
            render_prompt_file_loader(
                label=f"{settings['label']}プロンプト読込（.md/.txt）",
                upload_key=settings["prompt_upload_key"],
                prompt_text_key=settings["prompt_text_key"],
                source_profile_key=settings["prompt_source_key"],
                file_signature_key=settings["prompt_file_signature_key"],
                source_profile_id=client_content_source_id,
            )
            st.text_area(
                f"{settings['label']}プロンプト",
                key=sync_runtime_widget_from_state(settings["prompt_text_key"]),
                height=180,
                on_change=mark_runtime_widget_user_modified,
                args=(
                    runtime_widget_key(settings["prompt_text_key"]),
                    settings["prompt_text_key"],
                ),
            )


def two_client_settings_specs() -> list[dict[str, str]]:
    return [
        {
            "label": "クライアントA",
            "display_name_key": "client_a_display_name",
            "public_text_key": "client_a_public_profile_text",
            "public_source_key": "client_a_public_profile_source_profile_id",
            "public_upload_key": "client_a_public_profile_file_upload",
            "public_file_signature_key": "client_a_public_profile_file_signature",
            "prompt_upload_key": "client_a_prompt_file_upload",
            "prompt_text_key": "client_a_prompt_text",
            "prompt_source_key": "client_a_prompt_source_profile_id",
            "prompt_file_signature_key": "client_a_prompt_file_signature",
            "initial_transcript_key": "runtime_initial_client_a_transcript",
            "voice_key": "runtime_client_a_tts_voice",
            "speed_key": "runtime_client_a_output_speed",
            "gain_key": "runtime_client_a_audio_gain",
        },
        {
            "label": "クライアントB",
            "display_name_key": "client_b_display_name",
            "public_text_key": "client_b_public_profile_text",
            "public_source_key": "client_b_public_profile_source_profile_id",
            "public_upload_key": "client_b_public_profile_file_upload",
            "public_file_signature_key": "client_b_public_profile_file_signature",
            "prompt_upload_key": "client_b_prompt_file_upload",
            "prompt_text_key": "client_b_prompt_text",
            "prompt_source_key": "client_b_prompt_source_profile_id",
            "prompt_file_signature_key": "client_b_prompt_file_signature",
            "initial_transcript_key": "runtime_initial_client_b_transcript",
            "voice_key": "runtime_client_b_tts_voice",
            "speed_key": "runtime_client_b_output_speed",
            "gain_key": "runtime_client_b_audio_gain",
        },
    ]


def prepare_runtime_session_defaults(config: Any) -> None:
    if (
        st.session_state.get("runtime_realtime_model")
        not in RUNTIME_REALTIME_MODEL_OPTIONS
    ):
        st.session_state["runtime_realtime_model"] = DEFAULT_RUNTIME_REALTIME_MODEL
    if (
        st.session_state.get("runtime_prompting_llm_model")
        not in RUNTIME_PROMPTING_LLM_MODEL_OPTIONS
    ):
        st.session_state["runtime_prompting_llm_model"] = (
            DEFAULT_RUNTIME_PROMPTING_LLM_MODEL
        )
    if (
        st.session_state.get("runtime_prompting_llm_reasoning_effort")
        not in RUNTIME_PROMPTING_REASONING_EFFORT_OPTIONS
    ):
        st.session_state["runtime_prompting_llm_reasoning_effort"] = (
            DEFAULT_RUNTIME_PROMPTING_REASONING_EFFORT
        )
    if (
        st.session_state.get("runtime_timing_llm_model")
        not in RUNTIME_TIMING_LLM_MODEL_OPTIONS
    ):
        st.session_state["runtime_timing_llm_model"] = (
            DEFAULT_RUNTIME_TIMING_LLM_MODEL
        )
    if (
        st.session_state.get("runtime_timing_llm_reasoning_effort")
        not in RUNTIME_TIMING_REASONING_EFFORT_OPTIONS
    ):
        st.session_state["runtime_timing_llm_reasoning_effort"] = (
            DEFAULT_RUNTIME_TIMING_REASONING_EFFORT
        )
    if (
        st.session_state.get("runtime_summary_llm_model")
        not in RUNTIME_SUMMARY_LLM_MODEL_OPTIONS
    ):
        st.session_state["runtime_summary_llm_model"] = (
            DEFAULT_RUNTIME_SUMMARY_LLM_MODEL
        )
    if (
        st.session_state.get("runtime_summary_llm_reasoning_effort")
        not in RUNTIME_SUMMARY_REASONING_EFFORT_OPTIONS
    ):
        st.session_state["runtime_summary_llm_reasoning_effort"] = (
            DEFAULT_RUNTIME_SUMMARY_REASONING_EFFORT
        )
    default_closing_start_seconds = int(config.app.closing_start_audio_seconds)
    current_closing_start_seconds = st.session_state.get("runtime_closing_start_seconds")
    should_migrate_legacy_default = (
        current_closing_start_seconds == LEGACY_RUNTIME_CLOSING_START_DEFAULT_SECONDS
        and not runtime_state_was_user_modified("runtime_closing_start_seconds")
        and default_closing_start_seconds
        != LEGACY_RUNTIME_CLOSING_START_DEFAULT_SECONDS
        and st.session_state.get("session_status") == SessionStatus.IDLE.value
    )
    if current_closing_start_seconds is None or should_migrate_legacy_default:
        st.session_state["runtime_closing_start_seconds"] = int(
            default_closing_start_seconds
        )
    legacy_closing_turns = st.session_state.get("runtime_force_stop_after_closing_turns")
    legacy_closing_turns_is_valid = (
        isinstance(legacy_closing_turns, int)
        and not isinstance(legacy_closing_turns, bool)
        and legacy_closing_turns > 0
    )
    legacy_closing_turns_was_user_modified = runtime_state_was_user_modified(
        "runtime_force_stop_after_closing_turns"
    )
    active_closing_turns_key = runtime_force_stop_after_closing_turns_state_key()
    active_closing_turns = st.session_state.get(active_closing_turns_key)
    active_closing_turns_is_valid = (
        isinstance(active_closing_turns, int)
        and not isinstance(active_closing_turns, bool)
        and active_closing_turns > 0
    )
    active_closing_turns_is_unmodified_default = (
        active_closing_turns_is_valid
        and active_closing_turns == runtime_force_stop_after_closing_turns_default()
    )
    active_closing_turns_is_legacy_default = (
        active_closing_turns_is_valid
        and active_closing_turns
        in runtime_force_stop_after_closing_turns_legacy_defaults()
    )
    if (
        legacy_closing_turns_was_user_modified
        and legacy_closing_turns_is_valid
        and not runtime_state_was_user_modified(active_closing_turns_key)
        and (
            not active_closing_turns_is_valid
            or active_closing_turns_is_unmodified_default
            or active_closing_turns_is_legacy_default
        )
    ):
        st.session_state[active_closing_turns_key] = legacy_closing_turns
        st.session_state[f"{active_closing_turns_key}_persist_user_modified"] = True
    if st.session_state.get("runtime_one_client_force_stop_after_closing_turns") is None:
        st.session_state["runtime_one_client_force_stop_after_closing_turns"] = (
            legacy_closing_turns
            if (
                legacy_closing_turns_was_user_modified
                and legacy_closing_turns_is_valid
                and active_closing_turns_key
                == "runtime_one_client_force_stop_after_closing_turns"
            )
            else DEFAULT_RUNTIME_ONE_CLIENT_FORCE_STOP_AFTER_CLOSING_TURNS
        )
    elif (
        active_closing_turns_key == "runtime_one_client_force_stop_after_closing_turns"
        and active_closing_turns_is_legacy_default
        and not runtime_state_was_user_modified(
            "runtime_one_client_force_stop_after_closing_turns"
        )
    ):
        st.session_state["runtime_one_client_force_stop_after_closing_turns"] = (
            DEFAULT_RUNTIME_ONE_CLIENT_FORCE_STOP_AFTER_CLOSING_TURNS
        )
    if st.session_state.get("runtime_two_client_force_stop_after_closing_turns") is None:
        st.session_state["runtime_two_client_force_stop_after_closing_turns"] = (
            legacy_closing_turns
            if (
                legacy_closing_turns_was_user_modified
                and legacy_closing_turns_is_valid
                and active_closing_turns_key
                == "runtime_two_client_force_stop_after_closing_turns"
            )
            else DEFAULT_RUNTIME_TWO_CLIENT_FORCE_STOP_AFTER_CLOSING_TURNS
        )
    elif (
        active_closing_turns_key == "runtime_two_client_force_stop_after_closing_turns"
        and active_closing_turns_is_legacy_default
        and not runtime_state_was_user_modified(
            "runtime_two_client_force_stop_after_closing_turns"
        )
    ):
        st.session_state["runtime_two_client_force_stop_after_closing_turns"] = (
            DEFAULT_RUNTIME_TWO_CLIENT_FORCE_STOP_AFTER_CLOSING_TURNS
        )
    sync_runtime_force_stop_after_closing_turns_for_mode()
    counselor_preset_active = counselor_preset_is_active()
    voice_options = runtime_voice_options(config)
    voice_defaults = runtime_voice_defaults()
    for key, speaker in [
        ("runtime_counselor_tts_voice", "counselor"),
        ("runtime_client_tts_voice", "client"),
        ("runtime_client_a_tts_voice", "client_a"),
        ("runtime_client_b_tts_voice", "client_b"),
    ]:
        if speaker == "counselor" and counselor_preset_active:
            continue
        current = st.session_state.get(key)
        if not isinstance(current, str) or current.strip() not in voice_options:
            default_voice = voice_defaults.get(speaker, voice_defaults["client"])
            st.session_state[key] = (
                default_voice if default_voice in voice_options else voice_options[0]
            )
    gain_defaults = runtime_speaker_gain_defaults()
    for key, speaker in [
        ("runtime_counselor_audio_gain", "counselor"),
        ("runtime_client_audio_gain", "client"),
        ("runtime_client_a_audio_gain", "client_a"),
        ("runtime_client_b_audio_gain", "client_b"),
    ]:
        if speaker == "counselor" and counselor_preset_active:
            continue
        current = st.session_state.get(key)
        if (
            not isinstance(current, (int, float))
            or isinstance(current, bool)
            or current <= 0
            or (
                key == "runtime_counselor_audio_gain"
                and float(current) == 0.1
                and not runtime_state_was_user_modified(key)
            )
        ):
            st.session_state[key] = float(
                gain_defaults.get(speaker, gain_defaults["client"])
            )
    output_speed_defaults = runtime_output_speed_defaults()
    output_speed_defaults_version = st.session_state.get(
        "runtime_output_speed_defaults_version"
    )
    should_migrate_legacy_counselor_default = (
        not isinstance(output_speed_defaults_version, int)
        or isinstance(output_speed_defaults_version, bool)
        or output_speed_defaults_version < REALTIME_OUTPUT_SPEED_DEFAULTS_VERSION
    )
    legacy_output_speed = st.session_state.get("runtime_output_speed")
    legacy_output_speed_is_valid = (
        isinstance(legacy_output_speed, (int, float))
        and not isinstance(legacy_output_speed, bool)
        and MIN_REALTIME_OUTPUT_SPEED
        <= legacy_output_speed
        <= MAX_REALTIME_OUTPUT_SPEED
    )
    for key, speaker in [
        ("runtime_counselor_output_speed", "counselor"),
        ("runtime_client_output_speed", "client"),
        ("runtime_client_a_output_speed", "client_a"),
        ("runtime_client_b_output_speed", "client_b"),
    ]:
        if speaker == "counselor" and counselor_preset_active:
            continue
        current = st.session_state.get(key)
        if (
            isinstance(current, (int, float))
            and not isinstance(current, bool)
            and current > MAX_REALTIME_OUTPUT_SPEED
        ):
            st.session_state[key] = MAX_REALTIME_OUTPUT_SPEED
            continue
        legacy_counselor_speed = (
            key == "runtime_counselor_output_speed"
            and isinstance(current, (int, float))
            and not isinstance(current, bool)
            and (
                (
                    float(current) == 0.25
                    and not runtime_state_was_user_modified(key)
                )
                or (
                    float(current) == 0.6
                    and (
                        should_migrate_legacy_counselor_default
                        or not runtime_state_was_user_modified(key)
                    )
                )
                or (
                    float(current) == 0.9
                    and should_migrate_legacy_counselor_default
                    and not runtime_state_was_user_modified(key)
                )
            )
        )
        if (
            not isinstance(current, (int, float))
            or isinstance(current, bool)
            or not MIN_REALTIME_OUTPUT_SPEED
            <= current
            <= MAX_REALTIME_OUTPUT_SPEED
            or legacy_counselor_speed
        ):
            st.session_state[key] = float(
                legacy_output_speed
                if legacy_output_speed_is_valid
                else output_speed_defaults.get(speaker, output_speed_defaults["client"])
            )
    st.session_state["runtime_output_speed_defaults_version"] = (
        REALTIME_OUTPUT_SPEED_DEFAULTS_VERSION
    )


def prepare_two_client_runtime_session_defaults(config: Any) -> None:
    key = "runtime_two_client_force_stop_after_closing_turns"
    if runtime_state_was_user_modified(key):
        sync_runtime_force_stop_after_closing_turns_for_mode("two_clients")
        return

    current_value = st.session_state.get(key)
    one_client_default = DEFAULT_RUNTIME_ONE_CLIENT_FORCE_STOP_AFTER_CLOSING_TURNS
    current_value_is_valid = (
        isinstance(current_value, int)
        and not isinstance(current_value, bool)
        and current_value > 0
    )
    current_value_is_legacy_default = (
        current_value_is_valid
        and current_value
        in runtime_force_stop_after_closing_turns_legacy_defaults("two_clients")
    )
    if (
        not current_value_is_valid
        or current_value == one_client_default
        or current_value_is_legacy_default
    ):
        st.session_state[key] = (
            DEFAULT_RUNTIME_TWO_CLIENT_FORCE_STOP_AFTER_CLOSING_TURNS
        )
    sync_runtime_force_stop_after_closing_turns_for_mode("two_clients")


def runtime_speaker_gain_defaults() -> dict[str, float]:
    defaults = {
        "counselor": DEFAULT_RUNTIME_COUNSELOR_AUDIO_GAIN,
        "client": DEFAULT_RUNTIME_CLIENT_AUDIO_GAIN,
        "client_a": DEFAULT_RUNTIME_CLIENT_AUDIO_GAIN,
        "client_b": DEFAULT_RUNTIME_CLIENT_AUDIO_GAIN,
    }
    try:
        runtime_settings = load_runtime_config(ROOT_DIR / "config" / "runtime_config.yaml")
    except RuntimeConfigLoadError:
        return defaults
    speaker_gains = getattr(runtime_settings.audio, "speaker_gains", {}) or {}
    return {
        "counselor": float(
            speaker_gains.get("counselor", DEFAULT_RUNTIME_COUNSELOR_AUDIO_GAIN)
        ),
        "client": float(
            speaker_gains.get("client", DEFAULT_RUNTIME_CLIENT_AUDIO_GAIN)
        ),
        "client_a": float(
            speaker_gains.get(
                "client_a",
                speaker_gains.get("client", DEFAULT_RUNTIME_CLIENT_AUDIO_GAIN),
            )
        ),
        "client_b": float(
            speaker_gains.get(
                "client_b",
                speaker_gains.get("client", DEFAULT_RUNTIME_CLIENT_AUDIO_GAIN),
            )
        ),
    }


def runtime_output_speed_defaults() -> dict[str, float]:
    defaults = {
        "counselor": DEFAULT_RUNTIME_COUNSELOR_OUTPUT_SPEED,
        "client": DEFAULT_RUNTIME_CLIENT_OUTPUT_SPEED,
        "client_a": DEFAULT_RUNTIME_CLIENT_A_OUTPUT_SPEED,
        "client_b": DEFAULT_RUNTIME_CLIENT_B_OUTPUT_SPEED,
    }
    try:
        runtime_settings = load_runtime_config(ROOT_DIR / "config" / "runtime_config.yaml")
    except RuntimeConfigLoadError:
        return defaults
    openai_settings = runtime_settings.openai
    fallback_speed = float(getattr(openai_settings, "realtime_output_speed", 1.0))
    client_speed = float(
        getattr(
            openai_settings,
            "client_realtime_output_speed",
            None,
        )
        or fallback_speed
        or DEFAULT_RUNTIME_CLIENT_OUTPUT_SPEED
    )
    participants = getattr(runtime_settings, "participants", {}) or {}
    client_a = participants.get("client_a") if isinstance(participants, dict) else None
    client_b = participants.get("client_b") if isinstance(participants, dict) else None
    return {
        "counselor": float(
            getattr(
                openai_settings,
                "counselor_realtime_output_speed",
                None,
            )
            or fallback_speed
            or DEFAULT_RUNTIME_COUNSELOR_OUTPUT_SPEED
        ),
        "client": client_speed,
        "client_a": float(
            getattr(client_a, "realtime_output_speed", None) or defaults["client_a"]
        ),
        "client_b": float(
            getattr(client_b, "realtime_output_speed", None) or defaults["client_b"]
        ),
    }


def runtime_voice_defaults() -> dict[str, str]:
    defaults = {
        "counselor": "shimmer",
        "client": DEFAULT_RUNTIME_CLIENT_A_TTS_VOICE,
        "client_a": DEFAULT_RUNTIME_CLIENT_A_TTS_VOICE,
        "client_b": DEFAULT_RUNTIME_CLIENT_B_TTS_VOICE,
    }
    try:
        runtime_settings = load_runtime_config(ROOT_DIR / "config" / "runtime_config.yaml")
    except RuntimeConfigLoadError:
        return defaults
    openai_settings = runtime_settings.openai
    fallback_voice = str(openai_settings.tts_voice or "").strip()
    client_voice = str(
        openai_settings.client_tts_voice or fallback_voice or defaults["client"]
    ).strip()
    participants = getattr(runtime_settings, "participants", {}) or {}
    client_a = participants.get("client_a") if isinstance(participants, dict) else None
    client_b = participants.get("client_b") if isinstance(participants, dict) else None
    return {
        "counselor": str(
            openai_settings.counselor_tts_voice
            or fallback_voice
            or defaults["counselor"]
        ).strip(),
        "client": client_voice,
        "client_a": str(
            getattr(client_a, "voice", None) or defaults["client_a"]
        ).strip(),
        "client_b": str(
            getattr(client_b, "voice", None) or defaults["client_b"]
        ).strip(),
    }


def runtime_voice_options(config: Any) -> list[str]:
    voice_defaults = runtime_voice_defaults()
    options: list[str] = []
    for voice in [*RUNTIME_REALTIME_VOICE_OPTIONS, *voice_defaults.values()]:
        if not isinstance(voice, str):
            continue
        normalized_voice = voice.strip()
        if normalized_voice and normalized_voice not in options:
            options.append(normalized_voice)
    return options or ["marin", "coral"]


def prepare_runtime_text_defaults(
    selected_counselor: Profile,
    selected_client: Profile,
) -> None:
    if not st.session_state.get("counselor_display_name"):
        st.session_state["counselor_display_name"] = (
            DEFAULT_RUNTIME_COUNSELOR_DISPLAY_NAME
        )
    counselor_prompt_text = st.session_state.get("counselor_prompt_text")
    normalized_counselor_prompt = runtime_counselor_prompt_for_ui(
        counselor_prompt_text
    )
    if counselor_prompt_text != normalized_counselor_prompt:
        st.session_state["counselor_prompt_text"] = normalized_counselor_prompt
    participant_mode = str(
        st.session_state.get("runtime_participant_mode")
        or DEFAULT_RUNTIME_PARTICIPANT_MODE
    )
    if (
        not counselor_preset_is_active()
        and st.session_state.get("counselor_public_profile_source_profile_id")
        != selected_counselor.profile_id
    ):
        st.session_state["counselor_public_profile_text"] = (
            selected_counselor.public_profile or ""
        )
        st.session_state["counselor_public_profile_source_profile_id"] = (
            selected_counselor.profile_id
        )
    if st.session_state.get("selected_mode") == "ai_counselor_human_client":
        return
    if client_preset_is_active(participant_mode):
        return
    if not st.session_state.get("client_display_name"):
        st.session_state["client_display_name"] = DEFAULT_RUNTIME_CLIENT_DISPLAY_NAME
    client_prompt_text = st.session_state.get("client_prompt_text")
    if participant_mode == "one_client" and isinstance(client_prompt_text, str):
        stale_intros = [TWO_CLIENT_COMMON_PROMPT_INTRO]
        if not runtime_state_was_user_modified("client_prompt_text"):
            stale_intros.append(LEGACY_RUNTIME_ONE_CLIENT_PROMPT_INTRO)
        for stale_intro in stale_intros:
            if not client_prompt_text.startswith(stale_intro):
                continue
            client_prompt_text = client_prompt_text.replace(
                stale_intro,
                DEFAULT_RUNTIME_CLIENT_PROMPT_INTRO,
                1,
            )
            st.session_state["client_prompt_text"] = client_prompt_text
            break
    if (
        not client_prompt_text
        or client_prompt_text == LEGACY_RUNTIME_CLIENT_PROMPT
    ):
        st.session_state["client_prompt_text"] = DEFAULT_RUNTIME_CLIENT_PROMPT
    if (
        st.session_state.get("client_public_profile_source_profile_id")
        != selected_client.profile_id
    ):
        st.session_state["client_public_profile_text"] = (
            selected_client.public_profile or ""
        )
        st.session_state["client_public_profile_source_profile_id"] = (
            selected_client.profile_id
        )
    if (
        st.session_state.get("client_private_profile_source_profile_id")
        != selected_client.profile_id
    ):
        st.session_state["client_private_profile_text"] = (
            selected_client.hidden_background or ""
        )
        st.session_state["client_private_profile_source_profile_id"] = (
            selected_client.profile_id
        )
    for participant_id, display_name in [
        ("client_a", DEFAULT_RUNTIME_CLIENT_A_DISPLAY_NAME),
        ("client_b", DEFAULT_RUNTIME_CLIENT_B_DISPLAY_NAME),
    ]:
        if not st.session_state.get(f"{participant_id}_display_name"):
            st.session_state[f"{participant_id}_display_name"] = display_name
        participant_prompt_text = st.session_state.get(f"{participant_id}_prompt_text")
        if (
            not participant_prompt_text
            or participant_prompt_text == LEGACY_RUNTIME_CLIENT_PROMPT
        ):
            st.session_state[f"{participant_id}_prompt_text"] = (
                DEFAULT_RUNTIME_CLIENT_PROMPT
            )
        public_source_key = f"{participant_id}_public_profile_source_profile_id"
        if st.session_state.get(public_source_key) != selected_client.profile_id:
            st.session_state[f"{participant_id}_public_profile_text"] = (
                selected_client.public_profile or ""
            )
            st.session_state[public_source_key] = selected_client.profile_id
    initial_transcript = st.session_state.get("runtime_initial_client_transcript")
    initial_transcript_widget_exists = (
        runtime_widget_key("runtime_initial_client_transcript") in st.session_state
    )
    initial_transcript_is_legacy_default = (
        isinstance(initial_transcript, str)
        and initial_transcript.strip() in LEGACY_RUNTIME_INITIAL_CLIENT_TRANSCRIPTS
    )
    if (
        not initial_transcript
        or (
            initial_transcript_is_legacy_default
            and (
                not runtime_state_was_user_modified(
                    "runtime_initial_client_transcript"
                )
                or not initial_transcript_widget_exists
            )
        )
    ):
        try:
            runtime_settings = load_runtime_config(ROOT_DIR / "config" / "runtime_config.yaml")
        except RuntimeConfigLoadError:
            st.session_state["runtime_initial_client_transcript"] = (
                DEFAULT_RUNTIME_INITIAL_CLIENT_TRANSCRIPT
            )
        else:
            configured_initial_transcript = str(
                runtime_settings.runtime.initial_client_transcript or ""
            ).strip()
            if (
                not configured_initial_transcript
                or configured_initial_transcript
                in LEGACY_RUNTIME_INITIAL_CLIENT_TRANSCRIPTS
            ):
                configured_initial_transcript = (
                    DEFAULT_RUNTIME_INITIAL_CLIENT_TRANSCRIPT
                )
            st.session_state["runtime_initial_client_transcript"] = (
                configured_initial_transcript
            )
            if st.session_state.get("runtime_closing_start_seconds") is None:
                st.session_state["runtime_closing_start_seconds"] = int(
                    runtime_settings.runtime.closing_start_elapsed_seconds
                    or 0
                )
            if st.session_state.get("runtime_force_stop_after_closing_turns") is None:
                st.session_state["runtime_force_stop_after_closing_turns"] = int(
                    runtime_settings.runtime.force_stop_after_closing_turns
                )
    _prepare_runtime_two_client_initial_transcript_defaults()


def _prepare_runtime_two_client_initial_transcript_defaults() -> None:
    try:
        runtime_settings = load_runtime_config(ROOT_DIR / "config" / "runtime_config.yaml")
    except RuntimeConfigLoadError:
        runtime_settings = None
    participants = getattr(runtime_settings, "participants", {}) if runtime_settings else {}
    two_client_defaults = two_client_attachment_defaults()
    for participant_id in ["client_a", "client_b"]:
        key = f"runtime_initial_{participant_id}_transcript"
        if st.session_state.get(key):
            continue
        participant = participants.get(participant_id) if isinstance(participants, dict) else None
        initial_transcript = str(
            getattr(participant, "initial_transcript", "") or ""
        ).strip()
        if not initial_transcript:
            initial_transcript = str(
                two_client_defaults[f"{participant_id}_initial_transcript"]
            )
        st.session_state[key] = initial_transcript


def render_role_area(
    counselor_profiles: list[Profile],
    client_profiles: list[Profile],
    voice_presets: list[VoicePreset],
) -> None:
    _ = voice_presets
    counselor = select_profile_by_id(
        counselor_profiles, st.session_state["selected_counselor_profile_id"]
    )
    client = select_profile_by_id(
        client_profiles, st.session_state["selected_client_profile_id"]
    )
    columns = st.columns(2)
    with columns[0]:
        st.subheader("カウンセラー")
        counselor_display_name = (
            st.session_state.get("counselor_display_name")
            or DEFAULT_RUNTIME_COUNSELOR_DISPLAY_NAME
        )
        st.markdown(f"**{counselor_display_name}**")
        st.write(f"状態: {st.session_state['session_status']}")
    with columns[1]:
        st.subheader("クライアント")
        status = st.session_state.get("runtime_control_status") or {}
        mode = (
            status.get("interaction_mode")
            if status.get("phase") in {"running", "paused"}
            else st.session_state.get("selected_mode")
        )
        if mode == "ai_counselor_human_client":
            client_display_name = (
                (
                    status.get("participants", {}).get("client", {}).get("display_name")
                    if status.get("phase") in {"running", "paused"}
                    else None
                )
                or st.session_state.get("human_client_display_name")
                or "クライアント"
            )
        else:
            client_display_name = (
                st.session_state.get("client_display_name")
                or DEFAULT_RUNTIME_CLIENT_DISPLAY_NAME
            )
        st.markdown(f"**{client_display_name}**")
        st.write(f"状態: {st.session_state['session_status']}")


def render_current_audio_player(config: Any) -> None:
    session = st.session_state.get("session_model")
    if not isinstance(session, SessionState):
        return

    decision = next_playback_decision(session)
    if decision.action == "complete":
        return

    st.subheader("現在の音声")
    if decision.action == "hold":
        st.warning(
            "playback_hold のターンがあります。未再生先行生成テキストで確認してください。"
        )
        return
    if decision.action != "play" or decision.turn is None:
        st.info(f"音声の準備待ちです: {decision.reason}")
        return

    turn = decision.turn
    queue_version, audio_queue = build_audio_player_queue_for_ui(session)
    st.session_state["audio_player_queue_version"] = queue_version
    if not audio_queue:
        st.warning(f"音声ファイルが見つかりません: Turn {turn.turn_id}")
        return

    scheduled_start_at = st.session_state.get("auto_playback_started_at")
    start_delay_seconds = 0.0
    if (
        st.session_state.get("auto_playback_turn_id") == turn.turn_id
        and scheduled_start_at is not None
    ):
        remaining_seconds = float(scheduled_start_at) - time.monotonic()
        if remaining_seconds > 0:
            st.caption(f"次の発話まで {remaining_seconds:.1f} 秒")
            start_delay_seconds = remaining_seconds
    if st.session_state.get("auto_playback_visible_turn_id") != turn.turn_id:
        st.session_state["auto_playback_visible_started_at"] = time.monotonic()
    st.session_state["auto_playback_visible_turn_id"] = turn.turn_id

    event_value = browser_audio_player(
        queue=audio_queue,
        queue_version=queue_version,
        current_turn_id=turn.turn_id,
        autoplay=(
            SessionStatus(st.session_state["session_status"]) == SessionStatus.RUNNING
            and st.session_state.get("auto_playback_turn_id") == turn.turn_id
        ),
        start_delay_seconds=max(0.0, start_delay_seconds),
        gap_seconds=float(
            st.session_state.get(
                "auto_playback_gap_seconds",
                AUTO_PLAYBACK_DEFAULT_GAP_SECONDS,
            )
        ),
        default=None,
        key="browser_audio_player_v3",
    )
    playback_gap_seconds = float(
        st.session_state.get(
            "auto_playback_gap_seconds",
            AUTO_PLAYBACK_DEFAULT_GAP_SECONDS,
        )
    )
    updates = apply_browser_audio_event_for_ui(
        session,
        event_value,
        queue_version=queue_version,
        processed_event_ids=st.session_state.get("audio_player_processed_event_ids", []),
        conversation_engine=st.session_state.get("conversation_engine"),
        cumulative_audio_seconds=float(st.session_state["cumulative_audio_seconds"]),
        tts_client=st.session_state.get("tts_client"),
        log_paths=st.session_state.get("log_paths"),
        response_format=config.openai.default_audio_format,
        selected_turn_id=st.session_state.get("selected_unplayed_turn_id"),
        playback_gap_seconds=playback_gap_seconds,
    )
    if updates:
        for key, value in updates.items():
            st.session_state[key] = value


def render_transcripts(turns: list[PublicTranscriptTurn] | None = None) -> None:
    st.subheader("対話履歴")
    if turns is None:
        turns = current_public_transcript_turns()
    st.markdown(
        html_table(
            public_transcript_rows_for_display(turns),
            columns=PUBLIC_TRANSCRIPT_TABLE_COLUMNS,
            large_body_text=True,
            max_visible_body_rows=6,
        ),
        unsafe_allow_html=True,
    )


@st.fragment(run_every=RUNTIME_TRANSCRIPT_REFRESH_INTERVAL_SECONDS)
def render_live_transcript_panel(config: Any) -> None:
    defaults = runtime_control_defaults()
    if not st.session_state.get("runtime_control_host"):
        st.session_state["runtime_control_host"] = defaults["host"]
    if not st.session_state.get("runtime_control_port"):
        st.session_state["runtime_control_port"] = int(defaults["port"])

    endpoint = build_runtime_control_endpoint(
        host=str(st.session_state["runtime_control_host"]),
        port=int(st.session_state["runtime_control_port"]),
        session_id=st.session_state.get("runtime_monitor_session_id") or None,
    )
    client = RuntimeControlClient(endpoint.base_url)
    updates = refresh_runtime_status(
        client=client,
        current_session_id=st.session_state.get("runtime_monitor_session_id", ""),
    )
    for key, value in updates.items():
        st.session_state[key] = value

    turns = current_public_transcript_turns()
    transcript_audio_seconds = public_transcript_total_audio_seconds(turns)
    transcript_playback_seconds = runtime_playback_estimated_total_seconds(turns)
    status = st.session_state.get("runtime_control_status") or {}
    phase = str(status.get("phase") or "")
    pause_reason = str(status.get("pause_reason") or "")
    human_counselor_mode = runtime_human_input_enabled()
    continue_auto_throttle_pause = (
        phase == "paused"
        and pause_reason == "generation_throttle"
        and bool(st.session_state.get("runtime_generation_throttle_active", False))
    )
    now_monotonic = time.monotonic()
    warmup_updates = runtime_initial_warmup_state_updates(
        status=status,
        session_id=str(st.session_state.get("runtime_monitor_session_id") or ""),
        warmup_target_turns=int(
            st.session_state.get("runtime_generation_lead_limit")
            or DEFAULT_RUNTIME_GENERATION_LEAD_LIMIT
        ),
        warmup_max_wait_seconds=RUNTIME_AUDIO_MONITOR_WARMUP_MAX_WAIT_SECONDS,
        warmup_session_id=str(
            st.session_state.get("runtime_initial_warmup_session_id") or ""
        ),
        warmup_started_at_monotonic=st.session_state.get(
            "runtime_initial_warmup_started_at_monotonic"
        ),
        warmup_released=bool(
            st.session_state.get("runtime_initial_warmup_released", False)
        ),
        now_monotonic=now_monotonic,
    )
    for key, value in warmup_updates.items():
        st.session_state[key] = value
    timer_updates = runtime_timer_state_updates(
        status=status,
        session_id=str(st.session_state.get("runtime_monitor_session_id") or ""),
        transcript_audio_seconds=transcript_audio_seconds,
        current_wall_clock_seconds=float(st.session_state["wall_clock_seconds"]),
        wall_clock_session_id=str(
            st.session_state.get("runtime_wall_clock_session_id") or ""
        ),
        wall_clock_started_at_monotonic=st.session_state.get(
            "runtime_wall_clock_started_at_monotonic"
        ),
        wall_clock_base_seconds=float(
            st.session_state.get("runtime_wall_clock_base_seconds") or 0.0
        ),
        now_monotonic=now_monotonic,
        continue_when_paused=continue_auto_throttle_pause,
        playback_warmup_active=bool(
            st.session_state.get("runtime_initial_warmup_active", False)
        ),
    )
    for key, value in timer_updates.items():
        st.session_state[key] = value
    visibility_clock_updates = runtime_transcript_visibility_clock_state_updates(
        session_id=str(st.session_state.get("runtime_monitor_session_id") or ""),
        phase=phase,
        wall_clock_seconds=float(st.session_state.get("wall_clock_seconds") or 0.0),
        visibility_session_id=str(
            st.session_state.get("runtime_transcript_visibility_session_id") or ""
        ),
        visible_turn_count=int(
            st.session_state.get("runtime_transcript_visible_turn_count") or 0
        ),
        total_turn_count=len(turns),
        clock_started_at_monotonic=st.session_state.get(
            "runtime_transcript_visibility_clock_started_at_monotonic"
        ),
        clock_base_seconds=float(
            st.session_state.get("runtime_transcript_visibility_clock_base_seconds")
            or 0.0
        ),
        now_monotonic=now_monotonic,
        continue_when_paused=continue_auto_throttle_pause,
        completion_clock_seconds=transcript_playback_seconds,
    )
    for key, value in visibility_clock_updates.items():
        st.session_state[key] = value
    visibility_updates = runtime_transcript_visibility_state_updates(
        session_id=str(st.session_state.get("runtime_monitor_session_id") or ""),
        turns=turns,
        wall_clock_seconds=float(
            st.session_state.get("runtime_transcript_visibility_clock_seconds") or 0.0
        ),
        visibility_session_id=str(
            st.session_state.get("runtime_transcript_visibility_session_id") or ""
        ),
        visible_turn_count=int(
            st.session_state.get("runtime_transcript_visible_turn_count") or 0
        ),
        next_reveal_wall_seconds=st.session_state.get(
            "runtime_transcript_next_reveal_wall_seconds"
        ),
    )
    for key, value in visibility_updates.items():
        st.session_state[key] = value

    completed_updates = runtime_completed_turn_count_state_updates(
        status=st.session_state.get("runtime_control_status") or {},
        turns=turns,
        clock_seconds=float(
            st.session_state.get("runtime_transcript_visibility_clock_seconds") or 0.0
        ),
        session_id=str(st.session_state.get("runtime_monitor_session_id") or ""),
        previous_session_id=str(
            st.session_state.get("runtime_completed_turn_count_session_id") or ""
        ),
        previous_completed_turns=int(
            st.session_state.get("runtime_completed_turn_count_for_display") or 0
        ),
        browser_completed_turn_count=runtime_status_playback_completed_turn_count(
            st.session_state.get("runtime_control_status") or {}
        ),
    )
    for key, value in completed_updates.items():
        st.session_state[key] = value
    playback_completed_turn_count = int(
        completed_updates["runtime_completed_turn_count_for_display"]
    )

    throttle_decision = runtime_generation_throttle_decision(
        status=st.session_state.get("runtime_control_status") or {},
        playback_completed_turn_count=playback_completed_turn_count,
        lead_limit=int(
            st.session_state.get("runtime_generation_lead_limit")
            or DEFAULT_RUNTIME_GENERATION_LEAD_LIMIT
        ),
        throttle_active=bool(
            st.session_state.get("runtime_generation_throttle_active", False)
        ),
        throttle_disabled=runtime_generation_throttle_disabled_for_closing(
            status=st.session_state.get("runtime_control_status") or {},
            clock_seconds=float(
                st.session_state.get("wall_clock_seconds")
                or st.session_state.get("runtime_transcript_visibility_clock_seconds")
                or 0.0
            ),
            closing_start_seconds=float(
                st.session_state.get("runtime_closing_start_seconds")
                or config.app.closing_start_audio_seconds
            ),
        )
        or human_counselor_mode,
    )
    st.session_state["runtime_generation_lead_turns"] = throttle_decision[
        "runtime_generation_lead_turns"
    ]
    throttle_action = throttle_decision["runtime_generation_throttle_action"]
    if throttle_action is None:
        st.session_state["runtime_generation_throttle_active"] = throttle_decision[
            "runtime_generation_throttle_active"
        ]
    else:
        throttle_updates = apply_runtime_control_action(
            str(throttle_action),
            client=client,
            current_session_id=st.session_state.get("runtime_monitor_session_id", ""),
        )
        for key, value in throttle_updates.items():
            st.session_state[key] = value
        if not throttle_updates.get("runtime_control_error"):
            st.session_state["runtime_generation_throttle_active"] = throttle_decision[
                "runtime_generation_throttle_active"
            ]

    render_timer_band(config, turns=turns)
    render_transcript_download_gap()
    render_script_downloads()


def render_transcript_download_gap() -> None:
    st.markdown(
        '<div style="height: 24px;"></div>',
        unsafe_allow_html=True,
    )


def render_script_downloads() -> None:
    turns = current_public_transcript_turns()
    if not any(turn.text.strip() for turn in turns):
        return
    session_id = str(st.session_state.get("runtime_monitor_session_id") or "session")
    columns = st.columns(2)
    columns[0].download_button(
        "Download script (.md)",
        data=format_public_script(turns, format="markdown"),
        file_name=f"{session_id}_script.md",
        mime="text/markdown",
        width="stretch",
    )
    columns[1].download_button(
        "Download script (.txt)",
        data=format_public_script(turns, format="txt"),
        file_name=f"{session_id}_script.txt",
        mime="text/plain",
        width="stretch",
    )


@st.fragment(run_every=RUNTIME_TRANSCRIPT_REFRESH_INTERVAL_SECONDS)
def render_saved_session_replay(config: Any) -> None:
    _ = config
    sessions_root = runtime_sessions_root()
    sessions = list_runtime_sessions(sessions_root)
    pending_session_id = pending_saved_replay_session_id(
        sessions_root=sessions_root,
        sessions=sessions,
        current_session_id=str(st.session_state.get("runtime_monitor_session_id") or ""),
        runtime_status=st.session_state.get("runtime_control_status") or {},
    )
    with st.expander("Saved sessions", expanded=False):
        if pending_session_id:
            st.info(
                f"`{pending_session_id}` を保存中です。"
                "Replay session に反映されるまでしばらく待ってください。"
            )
        if not sessions:
            st.info("保存済み runtime session はまだありません。")
            return
        labels = {str(path): path.name for path in sessions}
        session_options = [str(path) for path in sessions]
        if st.session_state.get("selected_replay_session_dir") not in session_options:
            st.session_state["selected_replay_session_dir"] = session_options[0]
        selected_dir = st.selectbox(
            "Replay session",
            session_options,
            format_func=lambda value: labels.get(value, value),
            key="selected_replay_session_dir",
        )
        session_path = Path(selected_dir)
        identity = load_session_identity(session_path)
        saved_mode = identity.get("interaction_mode")
        if saved_mode in MODE_LABELS:
            st.caption(MODE_LABELS[saved_mode])
        artifact_turns = load_public_transcript_turns(session_path)
        session_audio_path = session_realtime_audio_path(session_path)
        session_audio_label = "Session audio"
        session_audio_download_label = "Download session audio (.wav)"
        session_audio_filename_suffix = "session_audio"
        if session_has_human_audio_turns(artifact_turns):
            compact_audio = build_compact_session_audio(session_path, artifact_turns)
            if compact_audio is not None:
                artifact_turns = compact_audio.turns
                session_audio_path = compact_audio.audio_path
                session_audio_label = "Compact session audio"
                session_audio_download_label = "Download compact session audio (.wav)"
                session_audio_filename_suffix = "session_compact_audio"
        turns = display_public_transcript_turns(artifact_turns)
        if not turns:
            st.info("このセッションには公開用 transcript がありません。")
            return
        start_options = replay_start_time_options(turns)
        start_option_keys = [option["key"] for option in start_options]
        start_option_by_key = {option["key"]: option for option in start_options}
        if (
            st.session_state.get("selected_replay_start_time_key")
            not in start_option_keys
        ):
            st.session_state["selected_replay_start_time_key"] = start_option_keys[0]
        start_time_key = st.selectbox(
            "Start from start time",
            start_option_keys,
            format_func=lambda key: start_option_by_key[key]["label"],
            key="selected_replay_start_time_key",
        )
        selected_start_option = start_option_by_key[start_time_key]
        start_seconds = float(selected_start_option["start_seconds"])
        replay_turns = turns[int(selected_start_option["start_index"]) :]
        st.markdown(
            html_table(
                public_transcript_rows_for_display(replay_turns),
                columns=PUBLIC_TRANSCRIPT_TABLE_COLUMNS,
            ),
            unsafe_allow_html=True,
        )
        replay_script_name = (
            f"{Path(selected_dir).name}_from_"
            f"{format_seconds(start_seconds).replace(':', '_')}"
        )
        download_columns = st.columns(2)
        download_columns[0].download_button(
            "Download replay script (.md)",
            data=format_public_script(replay_turns, format="markdown"),
            file_name=f"{replay_script_name}.md",
            mime="text/markdown",
            width="stretch",
        )
        download_columns[1].download_button(
            "Download replay script (.txt)",
            data=format_public_script(replay_turns, format="txt"),
            file_name=f"{replay_script_name}.txt",
            mime="text/plain",
            width="stretch",
        )
        if session_audio_path is None:
            st.info("このセッションには連結音声がありません。")
            return
        st.caption(f"{session_audio_label} from {format_seconds(start_seconds)}")
        st.audio(
            str(session_audio_path),
            format="audio/wav",
            start_time=start_seconds,
        )
        st.download_button(
            session_audio_download_label,
            data=session_audio_path.read_bytes(),
            file_name=f"{session_path.name}_{session_audio_filename_suffix}.wav",
            mime="audio/wav",
            width="stretch",
        )


def render_unplayed_preview(config: Any) -> None:
    with st.expander("未再生先行生成テキスト", expanded=False):
        options = turn_selection_options(st.session_state["unplayed_turns"])
        if options:
            option_ids = [option["turn_id"] for option in options]
            labels = {option["turn_id"]: option["label"] for option in options}
            current_id = st.session_state.get("selected_unplayed_turn_id")
            current_index = (
                option_ids.index(current_id) if current_id in option_ids else 0
            )
            selected_turn_id = st.selectbox(
                "編集対象ターン",
                option_ids,
                index=current_index,
                format_func=lambda turn_id: labels[turn_id],
            )
            st.session_state["selected_unplayed_turn_id"] = selected_turn_id
            selected_turn = selected_unplayed_turn(
                st.session_state["unplayed_turns"], selected_turn_id
            )
            if (
                selected_turn is not None
                and st.session_state.get("edit_turn_text_source_turn_id")
                != selected_turn_id
            ):
                st.session_state["edit_turn_text"] = selected_turn["text"]
                st.session_state["edit_turn_text_source_turn_id"] = selected_turn_id
            st.text_area("編集テキスト", key="edit_turn_text")
        st.markdown(
            html_table(
                unplayed_preview_rows(st.session_state["unplayed_turns"]),
                columns=UNPLAYED_TABLE_COLUMNS,
            ),
            unsafe_allow_html=True,
        )
        held_turns = [
            turn
            for turn in st.session_state["unplayed_turns"]
            if turn["playback_status"] == TurnStatus.PLAYBACK_HOLD.value
        ]
        if held_turns:
            st.radio(
                "playback_hold 操作",
                ["このまま再生", "編集", "再生成", "スキップ"],
                key="selected_hold_action",
                horizontal=True,
            )
            if st.button("Apply playback_hold action", width="stretch"):
                selected_turn_id = st.session_state.get("selected_unplayed_turn_id")
                hold_action = st.session_state.get("selected_hold_action", "このまま再生")
                if isinstance(st.session_state.get("session_model"), SessionState):
                    result = apply_hold_action_to_session(
                        st.session_state["session_model"],
                        selected_turn_id=selected_turn_id,
                        hold_action=hold_action,
                        log_paths=st.session_state.get("log_paths"),
                    )
                    warnings = result["warnings"]
                    if isinstance(
                        st.session_state.get("conversation_engine"), ConversationEngine
                    ):
                        warnings.extend(
                            refill_generation_queue_for_ui(
                                st.session_state["session_model"],
                                conversation_engine=st.session_state[
                                    "conversation_engine"
                                ],
                                cumulative_audio_seconds=float(
                                    st.session_state["cumulative_audio_seconds"]
                                ),
                                tts_client=st.session_state.get("tts_client"),
                                log_paths=st.session_state.get("log_paths"),
                                response_format=config.openai.default_audio_format,
                            )
                        )
                        result = live_session_ui_updates(
                            st.session_state["session_model"],
                            selected_turn_id=selected_turn_id,
                            warnings=warnings,
                        )
                else:
                    result = apply_hold_action_to_unplayed_turns(
                        st.session_state["unplayed_turns"],
                        selected_turn_id=selected_turn_id,
                        hold_action=hold_action,
                    )
                st.session_state["unplayed_turns"] = result["unplayed_turns"]
                st.session_state["warnings"] = result["warnings"]


def render_warnings() -> None:
    st.subheader("警告")
    warnings = st.session_state.get("warnings", [])
    if not warnings:
        st.info("現在の警告はありません。")
    else:
        for warning in warnings:
            st.warning(warning)


def _profile_label(profile: Profile) -> str:
    return f"{profile.display_name} ({profile.profile_id})"


def _profile_index(profiles: list[Profile], profile_id: str | None) -> int:
    for index, profile in enumerate(profiles):
        if profile.profile_id == profile_id:
            return index
    return 0


def _theme_label(theme: Theme) -> str:
    return f"{theme.display_name} ({theme.theme_id})"


def _theme_index(themes: list[Theme], theme_id: str | None) -> int:
    for index, theme in enumerate(themes):
        if theme.theme_id == theme_id:
            return index
    return 0


def _voice_preset_label(preset: VoicePreset) -> str:
    return f"{preset.display_name} / {preset.voice}"


def _voice_preset_index(presets: list[VoicePreset], preset_id: str | None) -> int:
    for index, preset in enumerate(presets):
        if preset.preset_id == preset_id:
            return index
    return 0


def _turn_to_unplayed_row(turn: Turn) -> dict[str, Any]:
    revision = turn.active_revision()
    return {
        "turn_id": turn.turn_id,
        "revision_id": revision.revision_id,
        "speaker_name": turn.speaker_name,
        "speaker_role": turn.speaker_role.value,
        "text": revision.canonical_text,
        "audio_status": turn.status.value,
        "playback_status": turn.playback_status.value,
        "edited": revision.edited,
        "warning_level": max(
            turn.warning_level,
            revision.warning_level,
            key=lambda warning_level: warning_rank(WarningLevel(warning_level)),
        ).value,
        "hold_reason": revision.hold_reason or "",
    }


if __name__ == "__main__":
    main()
