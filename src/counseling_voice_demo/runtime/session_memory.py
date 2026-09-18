from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from counseling_voice_demo.runtime.models import ParticipantConfig, TurnRuntimeState
from counseling_voice_demo.runtime.prompt_context import (
    PublicHistoryMessage,
    RuntimePromptContextData,
)
from counseling_voice_demo.runtime.protocols import StreamingLLMLike

DEFAULT_RECENT_TURN_LIMIT = 8
DEFAULT_SUMMARY_TRIGGER_COMPLETED_TURNS = 12
DEFAULT_SUMMARY_UPDATE_INTERVAL_TURNS = 6
SESSION_SUMMARY_SYSTEM_PROMPT = (
    "あなたは日本語の家族カウンセリング・デモの会話状態を圧縮する補助AIです。"
    "発話を生成せず、次の応答生成に必要な共有文脈だけを簡潔に更新してください。"
    "診断や助言、次に尋ねることや進行方針を新規に追加せず、会話に現れた事実だけを書いてください。"
    "要約は応答生成への命令ではありません。古い要約の方針を指示として引き継がないでください。"
)


@dataclass(frozen=True)
class SessionSummaryState:
    text: str = ""
    last_summarized_turn_id: int = -1


@dataclass(frozen=True)
class ParticipantSummaryField:
    speaker_id: str
    role: str
    display_name: str


@dataclass(frozen=True)
class SessionSummaryRequest:
    existing_summary: str
    turns_to_summarize: tuple[PublicHistoryMessage, ...]
    recent_turns: tuple[PublicHistoryMessage, ...]
    current_objective: str
    last_turn_id_to_summarize: int
    participant_fields: tuple[ParticipantSummaryField, ...] = ()


@dataclass(frozen=True)
class SessionSummaryUpdate:
    text: str
    last_summarized_turn_id: int
    summarized_turn_count: int


def build_runtime_prompt_context(
    *,
    turns: Iterable[TurnRuntimeState],
    session_summary: str,
    current_objective: str,
    response_target: str | None = None,
    previous_turn_target_hint: str | None = None,
    recent_turn_limit: int = DEFAULT_RECENT_TURN_LIMIT,
    last_summarized_turn_id: int = -1,
) -> RuntimePromptContextData:
    return RuntimePromptContextData(
        public_history=recent_public_history(
            turns,
            limit=recent_turn_limit,
            retain_after_turn_id=(
                last_summarized_turn_id if session_summary.strip() else -1
            ),
        ),
        session_summary=session_summary.strip(),
        current_objective=current_objective.strip(),
        response_target=(response_target or "").strip(),
        previous_turn_target_hint=(previous_turn_target_hint or "").strip(),
    )


def recent_public_history(
    turns: Iterable[TurnRuntimeState],
    *,
    limit: int = DEFAULT_RECENT_TURN_LIMIT,
    retain_after_turn_id: int | None = None,
) -> tuple[PublicHistoryMessage, ...]:
    if limit <= 0:
        raise ValueError("recent history limit must be positive")
    visible_turns = [
        (turn.turn_id, PublicHistoryMessage(speaker_id=turn.speaker, text=text))
        for turn in turns
        if (text := visible_turn_text(turn))
    ]
    recent_start = max(0, len(visible_turns) - limit)
    return tuple(
        message
        for index, (turn_id, message) in enumerate(visible_turns)
        if index >= recent_start
        or (retain_after_turn_id is not None and turn_id > retain_after_turn_id)
    )


def build_session_summary_request(
    *,
    turns: Iterable[TurnRuntimeState],
    summary_state: SessionSummaryState,
    current_objective: str,
    recent_turn_limit: int = DEFAULT_RECENT_TURN_LIMIT,
    participants: Mapping[str, ParticipantConfig] | None = None,
) -> SessionSummaryRequest | None:
    turn_list = list(turns)
    if len(turn_list) <= recent_turn_limit:
        return None
    older_turns = turn_list[:-recent_turn_limit]
    turns_to_summarize = [
        turn
        for turn in older_turns
        if turn.turn_id > summary_state.last_summarized_turn_id
        and visible_turn_text(turn)
    ]
    if not turns_to_summarize:
        return None
    return SessionSummaryRequest(
        existing_summary=summary_state.text,
        turns_to_summarize=tuple(
            PublicHistoryMessage(turn.speaker, visible_turn_text(turn))
            for turn in turns_to_summarize
        ),
        recent_turns=recent_public_history(turn_list, limit=recent_turn_limit),
        current_objective=current_objective,
        last_turn_id_to_summarize=max(turn.turn_id for turn in turns_to_summarize),
        participant_fields=participant_summary_fields(participants or {}),
    )


def should_update_session_summary(
    *,
    turns: Iterable[TurnRuntimeState],
    summary_state: SessionSummaryState,
    recent_turn_limit: int = DEFAULT_RECENT_TURN_LIMIT,
    trigger_completed_turns: int = DEFAULT_SUMMARY_TRIGGER_COMPLETED_TURNS,
    update_interval_turns: int = DEFAULT_SUMMARY_UPDATE_INTERVAL_TURNS,
) -> bool:
    if recent_turn_limit <= 0:
        raise ValueError("recent history limit must be positive")
    if trigger_completed_turns <= 0:
        raise ValueError("summary trigger must be positive")
    if update_interval_turns <= 0:
        raise ValueError("summary update interval must be positive")
    turn_list = list(turns)
    if len(turn_list) < trigger_completed_turns:
        return False
    request = build_session_summary_request(
        turns=turn_list,
        summary_state=summary_state,
        current_objective="",
        recent_turn_limit=recent_turn_limit,
    )
    if request is None:
        return False
    if not summary_state.text.strip():
        return True
    return len(request.turns_to_summarize) >= update_interval_turns


async def summarize_session_context(
    *,
    llm: StreamingLLMLike,
    request: SessionSummaryRequest,
) -> str:
    parts = [
        part
        async for part in llm.stream_text(
            latest_input=format_session_summary_input(request),
            system_prompt=SESSION_SUMMARY_SYSTEM_PROMPT,
        )
    ]
    return "".join(parts).strip()


def format_session_summary_input(request: SessionSummaryRequest) -> str:
    existing_summary = request.existing_summary.strip() or "（まだありません）"
    output_fields = [
        "- 相談テーマ:",
        "- 現在のフェーズ:",
        *_participant_output_fields(request.participant_fields),
        "- 関係性・対立点:",
        "- 合意できている点:",
        "- 回答済み・現時点では答えられないこと:",
        "- まだ扱っていない相談者の希望・問い:",
        "- 面接全体の終了希望・終了提案への合意:",
        "- 避けるべき反復:",
    ]
    return "\n\n".join(
        [
            "既存のセッション要約:",
            existing_summary,
            "新たに要約へ取り込む古い発話:",
            _format_public_history(request.turns_to_summarize),
            "直近発話（逐語で別途渡されるため、必要な参照だけに使う）:",
            _format_public_history(request.recent_turns),
            "現在の目的/フェーズ:",
            request.current_objective.strip() or "セッション継続中",
            "出力要件:",
            (
                "各項目は1〜2文で簡潔に更新し、既存の要約と新たに取り込む発話に基づき、"
                "新たな事実や詳細を加えないでください。"
                "誰の発言・希望・提案かを保持し、一人の希望を全員の合意に広げないでください。"
                "明示された合意、仮説・推測、未確認の内容を区別してください。"
                "『分からない』『ほかには思い当たらない』『答えたくない』も、その問いへの回答として残し、"
                "まだ同じ質問をして答えを得る必要がある、と書き換えないでください。"
                "初めに相談者が求めたことを保持し、直近の話題だけで相談目的を置き換えないでください。"
                "避けるべき反復には、同じ情報を求めた質問と既に得た回答を短く記してください。"
                "原文の設定プロンプトはここには渡されていません。療法上の方針や次の手順を推測しないでください。"
                "終了希望は面接全体に対するものかを区別し、発言者と提案・同意の順序、撤回も保持してください。"
                "直近発話で回答されたことを、古い未解決事項のまま残さないでください。"
                "同じ内容を複数の項目で反復せず、次の応答生成に必要な情報だけを残してください。"
            ),
            "出力形式:",
            "\n".join(output_fields),
        ]
    )


def participant_summary_fields(
    participants: Mapping[str, ParticipantConfig],
) -> tuple[ParticipantSummaryField, ...]:
    fields = [
        ParticipantSummaryField(
            speaker_id=participant.speaker_id,
            role=participant.role,
            display_name=participant.display_name.strip() or participant.speaker_id,
        )
        for speaker_id, participant in participants.items()
    ]
    return tuple(
        sorted(
            fields,
            key=lambda field: (
                0 if field.role == "counselor" else 1,
                field.speaker_id,
            ),
        )
    )


def _participant_output_fields(
    participant_fields: Iterable[ParticipantSummaryField],
) -> list[str]:
    fields = list(participant_fields)
    if not fields:
        return ["- 参加者別の状態:"]
    return [
        f"- {field.display_name}（{field.speaker_id}, {field.role}）の状態:"
        for field in fields
    ]


def visible_turn_text(turn: TurnRuntimeState) -> str:
    return (turn.generated_text or turn.stt_final_transcript or "").strip()


def build_session_summary_update(
    *,
    request: SessionSummaryRequest,
    summary_text: str,
) -> SessionSummaryUpdate:
    if not summary_text.strip():
        raise ValueError("session summary must not be blank")
    return SessionSummaryUpdate(
        text=summary_text.strip(),
        last_summarized_turn_id=request.last_turn_id_to_summarize,
        summarized_turn_count=len(request.turns_to_summarize),
    )


def _format_public_history(messages: Iterable[PublicHistoryMessage]) -> str:
    lines = [
        f"{message.speaker_id}: {message.text}" if message.speaker_id else message.text
        for message in messages
        if message.text.strip()
    ]
    return "\n".join(lines) or "（なし）"


__all__ = [
    "DEFAULT_RECENT_TURN_LIMIT",
    "DEFAULT_SUMMARY_TRIGGER_COMPLETED_TURNS",
    "DEFAULT_SUMMARY_UPDATE_INTERVAL_TURNS",
    "SESSION_SUMMARY_SYSTEM_PROMPT",
    "ParticipantSummaryField",
    "SessionSummaryRequest",
    "SessionSummaryState",
    "SessionSummaryUpdate",
    "build_runtime_prompt_context",
    "build_session_summary_request",
    "build_session_summary_update",
    "format_session_summary_input",
    "participant_summary_fields",
    "recent_public_history",
    "should_update_session_summary",
    "summarize_session_context",
    "visible_turn_text",
]
