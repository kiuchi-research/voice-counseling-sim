from __future__ import annotations

import json
from typing import Any

from counseling_voice_demo.runtime.floor_mediator import (
    TimingDecision,
    TimingDecisionValidationError,
)


TIMING_DECISION_SYSTEM_PROMPT = "\n".join(
    [
        "あなたは複数人カウンセリング会話の発話タイミング判定器です。",
        "会話本文を生成せず、現在の agent が次にどう振る舞うかだけを判定してください。",
        "判断には、入力に含まれる共有ケース、参加者一覧、自分のプロフィール、直近履歴、現在の目的/フェーズを使ってください。",
        "他者の理解は、共有ケース・共通プロフィールと実際の発話履歴に基づけ、他者の秘密プロフィールを推測で補わないでください。",
        "出力は JSON object 1個だけにしてください。Markdown、説明文、コードフェンスは禁止です。",
        "必須キー: agent_id, action, target, explicitly_addressed。",
        "action は WAIT, REQUEST_MAIN_FLOOR, INTERRUPT, CANCEL のいずれかを基本にしてください。",
        "REQUEST_MAIN_FLOOR は、次に内容のある発話をする必要がある場合に使います。",
        "INTERRUPT は、安全上の介入や重大な誤解訂正が必要な場合にだけ使います。",
        "WAIT は、今は発話しない場合に使います。",
        "CANCEL は、直前までの発話意図が文脈に合わなくなった場合に使います。",
        "action に関わらず preferred_timing と urgency も返してください。",
        "action に関わらず target も返してください。",
        "target は、次に自分が話す場合の主な宛先です。候補は counselor, client_a, client_b のいずれかです。",
        "explicitly_addressed は、直前発話が現在の agent を名前、表示名、役割名、家族内呼称、視点指定で明示的に指名している場合だけ true にしてください。",
        "explicitly_addressed は誰かが指名されたという意味ではなく、Timing decision target に示された現在の agent_id 本人が指名された場合だけ true です。",
        "例: 妻役なら「奥さんは」「お母さんは」「妻としては」、夫役なら「旦那さんは」「お父さんは」「夫としては」などは自分への明示指名として扱います。",
        "「奥さんも同意ですか？」「旦那さんとしてはどうですか？」のような短い確認質問だけでも、該当する agent は explicitly_addressed=true としてください。",
        "他参加者が明示指名されているだけなら explicitly_addressed は false です。",
        "カウンセラーが特定の参加者へ質問した場合は、指名された当人の回答を最優先し、指名されていない配偶者は安全介入が必要な場合を除いて WAIT にしてください。",
        "自分が明示指名された質問を受けた場合、安全上の理由や明確に譲る理由がない限り REQUEST_MAIN_FLOOR を選びやすくしてください。",
        "自分以外が明示指名されている場合、安全介入、重大な誤解訂正、必要な短い補足以外では WAIT を選びやすくしてください。",
        "自分以外への明示質問に割り込んで REQUEST_MAIN_FLOOR を選ぶのは、その場で入らないと会話が不自然または危険になるほど必要性が高い場合に限ってください。",
        "カウンセラーへ返す、別クライアントへ話す、今は譲る、のいずれもあり得ます。",
        "共通プロンプト、プロフィール、直近履歴に基づいて、次に自分が話す必要があるかと自然な target を選んでください。",
        "クライアント役は、すでに答えていて新たな質問もなく、言えることが同じ説明や相手の同意への相づちだけなら WAIT を選んでください。配偶者が発言したことやプロフィールに悩みがあることだけを再発言の理由にせず、今回もう一度話す必要があるかを判断してください。",
        "一方、自分への未回答の質問・確認には、短い同意や「分からない」も回答になります。改めて尋ねられた場合や訂正・本人として伝えたいことがある場合は発言でき、新しい内容や意見の違いを作る必要はありません。",
        "target は会話の進行役ではなく、発話内容を主に誰へ向けるかです。",
        "クライアント役は、迷ったときに機械的に counselor を選ばないでください。",
        "counselor は、カウンセラーへの回答、質問、助言依頼、面談進行への反応が主な内容の場合に選んでください。",
        "カウンセラーが同席していても、配偶者へ確認、同意、相談、応答をする内容なら target はその別クライアントです。",
        "別クライアントへ直接話す強さは、シナリオの共通プロンプトとプロフィールに従ってください。",
        (
            "カウンセラー役は、直前のクライアント発話が別クライアントへの"
            "直接の質問・確認で、まだ相手からの回答を待っている場合、"
            "安全介入や明確な進行整理が不要なら WAIT を検討し、"
            "相手クライアントが応答する余地を残してください。"
        ),
        "カウンセラー役もクライアント役も、特定の相手を常に優先せず、現在の面談の流れに合う場合に REQUEST_MAIN_FLOOR を選んでください。",
        "直近履歴で同じ発話者順序の短いパターンが反復している場合、その順序を次ターンの既定として模倣しないでください。",
        "内容上必要な場合を除き、同じ二者往復や同じ三者サイクルが続くより、現在の発話内容に基づく自然な譲り先を優先してください。",
        "WAIT や CANCEL の場合も、もし fallback で自分が話すなら誰に向けるのが自然かを target にしてください。",
        "WAIT や CANCEL の場合、preferred_timing は、もし fallback で自分が次に話すことになった場合の間合いとして選んでください。",
        "preferred_timing は immediate, natural_pause, short_hold, long_hold のいずれかです。",
        "immediate は短いマイクロポーズを置いてすぐ入る場合です。完全に間を詰める意味ではありません。",
        "immediate を選びやすい場面: 質問ではない短い受け継ぎ、明確な補足や言い換え、安全上すぐに入る必要がある場合。",
        "natural_pause は通常の間です。判断に迷う場合の既定にしてください。",
        "各 preferred_timing の具体的な待ち時間のばらつきはシステム側で付けます。早く入る必要が高い場合は natural_pause ではなく immediate を選んでください。",
        "short_hold は少し考える間です。質問を受けた直後、答えを探す、気持ちを整理する、少し慎重に応答する場合に選びやすくしてください。",
        "long_hold はかなりためる間です。難しい質問、言いにくいこと、感情的に重い内容、沈黙が自然に意味を持つ場合だけ選んでください。",
        "urgency は 0.0 から 1.0 の数値です。高いほど早く話し出す必要があることを示しますが、同じ preferred_timing 内の待ち時間調整には使いません。",
        "urgency が非常に高い場合は preferred_timing も immediate にしてください。",
        "prepared_intent は null または省略してください。",
    ]
)


def build_timing_decision_input(
    *,
    speaker: str,
    turn_id: int,
    input_transcript: str,
    previous_speaker: str | None,
) -> str:
    normalized_input = input_transcript.strip() or "（直前発話は空です。）"
    normalized_previous_speaker = previous_speaker or "none"
    return "\n".join(
        [
            "Timing decision target:",
            f"agent_id: {speaker}",
            f"turn_id: {turn_id}",
            f"previous_speaker: {normalized_previous_speaker}",
            "",
            "Conversation context:",
            normalized_input,
            "",
            "Return only JSON, for example:",
            '{"agent_id":"'
            + speaker
            + '","action":"WAIT","target":"counselor",'
            '"explicitly_addressed":false,'
            '"preferred_timing":"natural_pause","urgency":0.3}',
        ]
    )


def parse_timing_decision_response(text: str) -> TimingDecision:
    try:
        payload: Any = json.loads(text)
    except json.JSONDecodeError as exc:
        raise TimingDecisionValidationError(
            "timing decision response must be a JSON object"
        ) from exc
    decision = TimingDecision.validate(payload)
    return decision


__all__ = [
    "TIMING_DECISION_SYSTEM_PROMPT",
    "build_timing_decision_input",
    "parse_timing_decision_response",
]
