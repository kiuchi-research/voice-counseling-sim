"""Generate inspectable Prompt Director responses for synthetic, generic cases.

Uses the configured text route. No saved sessions or counselor presets are sent.
Schema validity is checked by the runtime; adherence is assessed from the report.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

from counseling_voice_demo.runtime.config import load_runtime_config
from counseling_voice_demo.runtime.control_api import apply_runtime_start_options
from counseling_voice_demo.runtime.controller import (
    CLOSING_CLIENT_INSTRUCTION,
    CLOSING_COUNSELOR_INSTRUCTION,
    SESSION_TIME_INSTRUCTION,
)
from counseling_voice_demo.runtime.factory import build_openai_runtime
from counseling_voice_demo.runtime.prompt_context import PublicHistoryMessage
from counseling_voice_demo.runtime.prompt_director import (
    ClosingAssessmentRequest,
    PromptDirector,
    PromptDirectorError,
    PromptDirectorRequest,
    SessionEndContext,
)

from counseling_voice_demo.runtime.session_memory import (
    SessionSummaryRequest,
    summarize_session_context,
)
from counseling_voice_demo.runtime.system_prompts import CLIENT_SYSTEM_PROMPT

CASES = (
    {
        "id": "proposal_is_not_completion",
        "prompt": (
            "利用者が比較したい点を具体的に一つ尋ねる。\n"
            "整理への同意だけを、具体的な内容の回答や理解の完了として扱わない。\n"
            "すでに得られた回答を繰り返し尋ねず、原文にない助言は加えない。"
        ),
        "history": (
            ("assistant", "二つの案を一緒に整理しましょうか。"),
            ("user", "はい、整理したいです。"),
        ),
        "criteria": "比較したい点の中身を尋ねる。整理の提案を繰り返したり、整理できたと扱ったりしない。",
    },
    {
        "id": "purpose_and_partial_agreement",
        "prompt": (
            "参加者ごとの希望と合意を、発言者を区別して短く確認する。\n"
            "目的と手段を入れ替えず、全員の合意が確認された場合だけ共通方針として扱う。\n"
            "質問や新しい提案はしない。"
        ),
        "history": (
            ("person_b", "費用を抑えるために経路を比較したいです。"),
            ("assistant", "まず移動の楽さを整理しませんか。"),
            ("person_a", "私はそれでいいです。"),
        ),
        "criteria": "Bの目的は費用削減、Aは楽さの整理に同意。Bの同意は未確認と区別し、『費用より比較』『二人とも合意』に変えない。",
    },
    {
        "id": "other_speaker_action",
        "prompt": (
            "あなたは相談者B。仕事を止める不安があり、自動的には同意しない。\n"
            "Aが休む方法を話したら、その気持ちを否定せず自分の不安も表す。\n"
            "Aの行動を自分自身が行ったことにしない。専門家として助言しない。"
        ),
        "speaker_id": "person_b",
        "history": (
            ("person_b", "休むと仕事が遅れるのが気になります。"),
            ("person_a", "私は昼に一時間休んでいます。"),
        ),
        "criteria": "直前のAの発言を認識し、自分も一時間休んでいるとは言わず、Bの不安を保って反応する。",
    },
    {
        "id": "closing_client_pending_question",
        "prompt": (
            "あなたは迷いを抱える相談者。選択を急がず、自分の不安を自然に話す。\n"
            "提案に自動的に同意したり、話していない改善を報告したりしない。\n"
            "分からなければ分からないと答えてよい。"
        ),
        "speaker_id": "client",
        "history": (
            ("client", "今の方法をやめると困るかもしれないので、迷っています。"),
            ("counselor", "その方法をやめたら、ご自身にはどんな変化が起きそうですか？"),
        ),
        "turn_instruction": CLOSING_CLIENT_INSTRUCTION,
        "criteria": "質問内容に、自分に起こりそうな変化、不安、分からなさのいずれかで反応する。感謝・同意だけで終わらず、改善を捏造しない。",
    },
    {
        "id": "specific_question",
        "prompt": "最優先は、利用者が試した操作の結果を確認すること。\n操作の内容が分かっている場合は、操作後に画面に出たメッセージを具体的に一つ尋ねる。\nあいさつは必要なら添えてよい。原因の推測や操作の提案はまだしない。",
        "history": (("user", "保存ボタンを押したのですが、うまくいきませんでした。"),),
        "criteria": "保存後の画面メッセージを具体的に尋ねる。操作を尋ね直さず、原因や対処を追加しない。",
    },
    {
        "id": "optional_question",
        "prompt": "最優先は利用者が言葉を続ける余地を残すこと。\n今回の返答は短い受領だけにする。質問は加えない。\n普段は必要に応じて確認質問をしてもよい。",
        "history": (
            (
                "user",
                "新しい案を考えています。まだ途中なので、もう少し自分で説明を続けたいです。",
            ),
        ),
        "criteria": "短い受領のみ。任意の確認質問を必須と誤解せず、質問や提案を追加しない。",
    },
    {
        "id": "separate_preferences",
        "prompt": "参加者それぞれの希望を尊重する。\n全員の合意が確認された場合だけ、共通の予定として扱う。\n予定が一致していなければ、相違を短く確認する。予定を勝手に決めない。",
        "history": (
            ("person_a", "私は金曜日の開催を希望します。"),
            ("person_b", "私は日曜日なら参加できます。"),
        ),
        "criteria": "金曜と日曜を別の希望として扱う。共通の合意や確定した予定にしない。",
    },
    {
        "id": "closing_without_advice",
        "prompt": "終了時は確認できた内容だけを簡潔にまとめる。\n新しい助言や、合意していない次の行動を追加しない。\n未決の内容は未決として扱う。",
        "history": (
            (
                "person_a",
                "選択肢が二つあることは分かりました。どちらにするかは、まだ決めていません。",
            ),
        ),
        "turn_instruction": "ここで対話を終了する。",
        "criteria": "二つの選択肢があり未決であることを保つ。次回の課題や様子見などを追加しない。",
    },
    {
        "id": "closing_required_question",
        "prompt": (
            "利用者の言葉を尊重し、短く自然に応答する。\n"
            "終了の指示が来たら、新しい話題や助言は始めず、終了時の確認へ移る。\n"
            "終了時の確認: 今日の比較で選ぶ基準が見えたかどうかを、利用者本人に必ず尋ねる。\n"
            "本人に代わって理解度や結論を決めない。\n"
            "すでに基準が見えたかどうか答えている場合は、繰り返し尋ねない。\n"
            "一度に質問は一つ。"
        ),
        "history": (
            ("user", "今日は、二つの通勤経路を選ぶ基準が分かると助かります。"),
            ("assistant", "それぞれの経路で気になっていることは何ですか？"),
            (
                "user",
                "電車は安いですが乗り換えが多く、バスは楽ですが時間がかかります。",
            ),
        ),
        "turn_instruction": "ここから終結に入る。話した内容を簡潔にまとめ、終了に向けた一言を伝える。",
        "criteria": "選ぶ基準が見えたかを本人に尋ねる。要約や『整理できましたね』で必須質問を代用しない。",
    },
    {
        "id": "closing_question_already_answered",
        "prompt": (
            "終了時は、今日の比較で選ぶ基準が見えたかどうかを本人に必ず尋ねる。\n"
            "ただし、その答えがすでに語られていれば繰り返し尋ねず、その言葉を短く受け止める。\n"
            "新しい質問、助言、課題は追加しない。"
        ),
        "history": (
            (
                "user",
                "通勤経路を比べたら、費用より乗り換えの少なさを大事にしたいとはっきりしました。",
            ),
        ),
        "turn_instruction": "話を簡潔にまとめ、終了に向けた一言を伝える。",
        "criteria": "乗り換えの少なさという本人の基準を受け止める。確認済みの内容を尋ね直さない。",
    },
    {
        "id": "closing_question_forbidden",
        "prompt": (
            "普段は必要に応じて疑問点を確認してもよい。\n"
            "終了時は質問をせず、本人が語った内容を一文で受け止める。\n"
            "助言や次の行動は追加しない。"
        ),
        "history": (
            ("user", "比較の続きは自分で考えたいので、今日はここで終わりたいです。"),
        ),
        "turn_instruction": "ここから終了に入る。",
        "criteria": "短い受領で終える。終了時だからという理由で質問を作らない。",
    },
    {
        "id": "closing_flexible_sentence_guideline",
        "prompt": (
            "原則1〜2文で、短く伝え返す1文＋質問1文を基本とする。\n"
            "一度に質問は一つだけにする。\n"
            "終了時は、二人それぞれの希望、比較で分かった違い、未決の点を具体的に整理する。\n"
            "必須質問: 今日の比較で選ぶ基準が見えたかを本人たちに尋ねる。\n"
            "二人の希望を混同せず、原文や履歴にない結論・助言は追加しない。"
        ),
        "history": (
            ("person_a", "私は毎日の交通費を抑えたいので電車が気になります。"),
            ("person_b", "私は乗り換えが少なくて疲れにくい方がいいです。"),
            ("person_a", "比べると電車は安いけれど乗り換えが二回あります。"),
            (
                "person_b",
                "バスは一本で行けるけれど、電車より時間も費用もかかります。まだどちらにするかは相談中です。",
            ),
        ),
        "turn_instruction": CLOSING_COUNSELOR_INSTRUCTION
        + "\n1発話は短くしてください。",
        "criteria": (
            "文数・形式はpreferred、質問数と必須質問はrequiredとして分ける。"
            "双方の希望・経路の違い・未決事項・本人への質問を残し、"
            "文数の目安だけを理由に省略しない。二文を超えること自体は必須ではない。"
        ),
    },
    {
        "id": "mixed_strength_same_line",
        "prompt": (
            "原則1〜2文で短く伝え返す1文＋質問1文を基本とする。一度に質問は一つだけにする。\n"
            "利用者の希望を受け止め、まだ分かっていない希望日を必ず尋ねる。"
        ),
        "history": (("user", "屋外ではなく、室内の会場を予約したいです。"),),
        "criteria": (
            "同じ1行目でも文数・形式はpreferred、質問数はrequiredの別項目にする。"
            "室内という希望を受け止め、希望日を一つ尋ねる。必要なく長くしない。"
        ),
    },
    {
        "id": "strict_sentence_limit",
        "prompt": (
            "どの場面でも必ず2文以内で応答する。\n"
            "終了時は本人の報告を受け止め、解決していない点があるかを一つ尋ねる。"
        ),
        "history": (("user", "通知設定を変えたら、通知が届くようになりました。"),),
        "turn_instruction": CLOSING_COUNSELOR_INSTRUCTION,
        "criteria": "文数上限はrequiredを維持し、二文以内で報告の受領と未解決事項の確認を行う。",
    },
)


CLIENT_CASES = (
    {
        "id": "client_initial_position_survives_peer_reply",
        "speaker_id": "person_b",
        "prompt": "夫婦で引っ越し後の片づけを相談に来た夫として、プロフィールに沿って自然に話してください。",
        "profile": "夫。いつまでも荷物を置いたままにするのが嫌で、早く片づけたい。妻の負担は気になるが、初めから急がなくてよいとは思っていない。",
        "history": (
            ("person_a", "片づけについて夫婦でどう話せばいいか相談に来ました。"),
            ("counselor", "どういうところが気になっていますか？"),
            (
                "person_a",
                "夫は片づけを急いでいますが、私は今は無理をしないほうがいいと思っています。",
            ),
        ),
        "draft": "私も、無理に片づけようとして疲れてしまわないか気になっています。",
        "criteria": "妻の見方を夫自身の立場にしない。夫が早く片づけたい気持ちを保って応じる。新しい出来事や対立を作らない。",
    },
    {
        "id": "client_review_preserves_own_answer_after_repeated_questions",
        "speaker_id": "person_a",
        "prompt": "家族への連絡を相談に来た妻として、本人の考えと会話の経過に沿って自然に話してください。",
        "profile": "妻。連絡の負担を減らしたい。夫がまだ決められない気持ちは理解するが、自分の答えは履歴のとおり。",
        "history": (
            ("counselor", "どのような連絡なら負担が少なそうですか？"),
            ("person_a", "電話より、短いメッセージなら送れそうです。"),
            ("person_b", "私はまだ、どう連絡するか決められません。"),
            ("counselor", "どのような連絡なら負担が少なそうですか？"),
            ("person_a", "短いメッセージのほうが、今は気が楽です。"),
            ("person_b", "私はまだ決められないです。"),
            ("counselor", "どのような連絡なら負担が少なそうですか？"),
            ("person_a", "やっぱり、短いメッセージなら送れそうです。"),
            ("person_b", "まだどうするか決まらないんです。"),
            ("counselor", "どのような連絡なら負担が少なそうですか？"),
        ),
        "draft": "短いメッセージで、まず様子を聞くくらいならできそうです。",
        "criteria": "妻の『短いメッセージなら送れる』という答えを消さない。夫の『決められない』に置換しない。同じ質問が続いた経過を踏まえ、架空の経験や計画を足さない。",
    },
    {
        "id": "client_repeated_unknown_is_not_a_repeated_speech",
        "speaker_id": "person_b",
        "prompt": "家族への連絡を相談に来た夫として、本人の考えと会話の経過に沿って自然に話してください。短い返事でもかまいません。",
        "profile": "夫。何から話せばいいかまだ思い浮かばない。妻も迷っている。解決策を作る必要はない。",
        "history": (
            ("counselor", "どんな言葉なら伝えられそうですか？"),
            (
                "person_a",
                "負担はかけたくないんですが、具体的な言葉はまだ決められません。",
            ),
            (
                "person_b",
                "私も負担はかけたくないんですが、具体的な言葉はまだ決められません。",
            ),
            ("counselor", "どんな言葉なら伝えられそうですか？"),
            (
                "person_a",
                "負担はかけたくないんですが、具体的な言葉はまだ決められません。",
            ),
        ),
        "criteria": "迷いは保ち、夫婦が同じ一文を三度言い直すループを続けない。短い返事や繰り返し聞かれたことへの反応でよい。解決策や経験を捏造しない。",
    },
    {
        "id": "client_echo_is_not_supplement",
        "speaker_id": "person_b",
        "prompt": "あなたは夫婦で通勤の負担を相談に来た夫です。本人として自然に話してください。",
        "profile": "夫。時間に正確でありたい。混雑で遅刻しないか気になる。妻の観察には異論はない。",
        "shared_context": "夫婦は同じ駅を使う。朝は混雑し、夕方は比較的空く。",
        "history": (
            ("counselor", "駅の混み方について教えてください。"),
            ("person_a", "朝の駅はとても混んでいて、夕方には少し空くことが多いです。"),
        ),
        "draft": "朝の駅はとても混んでいて、夕方には少し空くことが多いです。遅刻しないか気になります。",
        "criteria": "妻の一文を丸ごと繰り返す前置きを修正する。共有の観察への同意と、夫の遅刻への懸念は保ち、新しい出来事や対立を作らない。",
    },
    {
        "id": "client_shared_unknown",
        "speaker_id": "person_a",
        "prompt": "夫婦で住まいの相談に来た妻として話してください。短い返事でもかまいません。",
        "profile": "妻。建物について夫と同じ情報しか知らない。追加の経験や気がかりはない。",
        "history": (
            ("person_a", "建物の詳しいことは私にも分かりません。"),
            ("counselor", "お二人とも、ほかに分かることはありますか？"),
            (
                "person_b",
                "今のところ、ほかには分かりません。分かったことがあれば、またお伝えします。",
            ),
        ),
        "criteria": "妻も不明と短く答えてよい。夫の二文全体を言い直すだけにせず、情報・反対意見・感情を捏造しない。",
    },
    {
        "id": "client_short_agreement_is_valid",
        "speaker_id": "person_b",
        "prompt": "利用者B本人として、Aの確認に自然に返事をしてください。",
        "profile": "B。明日の集合は午後3時に駅前と合意済み。その予定に異論も迷いもない。",
        "history": (("person_a", "明日は午後3時に駅前でいいよね？"),),
        "draft": "うん、それで大丈夫。",
        "criteria": "自然な短い同意を保持する。差を出すための不安・反論・追加情報を要求しない。",
    },
    {
        "id": "client_requested_repetition_is_valid",
        "speaker_id": "person_b",
        "prompt": "利用者B本人として話してください。確認のため頼まれた予定は、そのまま復唱してください。",
        "profile": "B。Aと同じ集合予定を把握しており、内容に同意している。",
        "history": (
            ("person_a", "明日の午後3時に駅前で集合です。"),
            ("counselor", "Bさんも、今の集合予定をそのまま復唱していただけますか？"),
        ),
        "draft": "明日の午後3時に駅前で集合です。",
        "criteria": "他者と同文でも、依頼された確認の復唱をそのまま保持する。重複だけを理由に削らない。",
    },
    {
        "id": "client_keeps_own_perspective",
        "speaker_id": "person_b",
        "prompt": "旅行の相談に来た夫として、自分の希望と相手への反応を自然に話してください。",
        "profile": "夫。旅行費用を抑えたい。妻が静かな宿を好むことは理解しているが、値段を確認せず高い宿には決められない。",
        "history": (("person_a", "私は少し高くても、静かな宿でゆっくりしたいです。"),),
        "criteria": "妻の希望を本人の希望としてコピーせず、予算への懸念を保つ。妻を非難せず、金額・過去の揉め事を作らない。",
    },
    {
        "id": "client_does_not_borrow_experience",
        "speaker_id": "person_b",
        "prompt": "会場選びを相談している利用者B本人として話してください。",
        "profile": "B。会場へ行ったことがない。Aの下見の報告を初めて聞いた。車いすで入れるかが気がかり。",
        "history": (("person_a", "昨日見に行ったら、入口に階段がありました。"),),
        "criteria": "自分も下見したとは言わない。聞いた報告への本人の懸念や確認として応じ、スロープの有無を勝手に断定しない。",
    },
    {
        "id": "counselor_prescribed_reflection_is_valid",
        "speaker_id": "counselor",
        "prompt": "あなたはカウンセラーです。今回は、利用者の最後の一文を一字一句そのまま返してください。質問や解説は加えません。",
        "history": (("person_a", "まだ、どちらにするか決められません。"),),
        "draft": "まだ、どちらにするか決められません。",
        "criteria": "カウンセラーの原文が指定した反復はそのまま保持する。クライアント間の復唱対策を別の役割へ適用しない。",
    },
)


_LATE_CLIENT_HISTORY = (
    ("person_a", "旅行は、予定を詰めすぎずに過ごしたいです。"),
    ("person_b", "私は移動の時間を決めておきたいです。遅れないか気になります。"),
    ("counselor", "お二人はそれぞれ、どこがまだ気がかりでしょうか？"),
    ("person_a", "予定を決めすぎると、休みにくそうで。"),
    ("person_b", "何も決めないと、乗り遅れないか心配です。"),
    ("counselor", "ここまで話してみて、何か見えてきたことはありますか？"),
    (
        "person_a",
        "まだはっきり見えていませんが、予定を詰めすぎないことは大事にしたいです。",
    ),
)
_LATE_CLIENT_DRAFT = (
    "私は移動の時間は決めておきたいです。ただ、どこまで決めるかは、まだ迷っています。"
)
_LATE_COUNSELOR_PROMPT = (
    "あなたはカウンセラーです。終了時は、ここまで話して見えてきたことを二人に尋ねます。\n"
    "回答を受け取ってから終了します。分からないという回答も尊重してください。\n"
    "回答済みの質問は繰り返さず、原文にない助言や改善の断定はしません。"
)
LATE_CASES = (
    {
        "id": "late_review_preserves_own_draft",
        "client": True,
        "speaker_id": "person_b",
        "prompt": "夫婦で旅行の相談に来た夫として、本人の考えと会話の経過に沿って自然に話してください。",
        "profile": "夫。移動の時間を決めておきたい。どこまで計画するか迷っている。妻は予定を詰めすぎたくない。",
        "history": _LATE_CLIENT_HISTORY,
        "turn_instruction": CLOSING_CLIENT_INSTRUCTION,
        "draft": _LATE_CLIENT_DRAFT,
        "criteria": "本人の希望と迷いを述べた案をそのまま保持する。妻の回答で上書きせず、意味上の再生成は不要。",
    },
    {
        "id": "late_review_rejects_spouse_copy",
        "client": True,
        "speaker_id": "person_b",
        "prompt": "夫婦で旅行の相談に来た夫として、本人の考えと会話の経過に沿って自然に話してください。",
        "profile": "夫。移動の時間を決めておきたい。どこまで計画するか迷っている。妻は予定を詰めすぎたくない。",
        "history": _LATE_CLIENT_HISTORY,
        "turn_instruction": CLOSING_CLIENT_INSTRUCTION,
        "draft": _LATE_CLIENT_HISTORY[-1][1],
        "criteria": "妻の文をそのまま本人の考えにした案をresponse_issuesで差し戻す。生成担当が夫自身の希望・迷いを保って再生成し、再審査する。",
    },
    {
        "id": "late_closing_does_not_reask_answered_unknown",
        "speaker_id": "counselor",
        "shared_context": "person_aは妻、person_bは夫。二人で旅行の計画について相談している。",
        "prompt": _LATE_COUNSELOR_PROMPT,
        "history": (*_LATE_CLIENT_HISTORY, ("person_b", _LATE_CLIENT_DRAFT)),
        "turn_instruction": CLOSING_COUNSELOR_INSTRUCTION,
        "draft": "ここまで話してみて、何か見えてきたことはありますか？",
        "criteria": "二人とも回答済みであり、迷いを未回答へ戻さない。再質問案を差し戻し、回答を受け取って終える本文に再生成する。",
    },
    {
        "id": "late_closing_keeps_question_for_unanswered_person",
        "speaker_id": "counselor",
        "shared_context": "person_aは妻、person_bは夫。二人で旅行の計画について相談している。",
        "prompt": _LATE_COUNSELOR_PROMPT,
        "history": _LATE_CLIENT_HISTORY,
        "turn_instruction": CLOSING_COUNSELOR_INSTRUCTION,
        "draft": "ご主人は、ここまで話してみて何か見えてきたことはありますか？",
        "criteria": "夫はまだ答えていない。夫にだけ向けた必要な確認を保持し、妻の回答で全員が回答済みと扱わない。意味上の再生成は不要。",
    },
)


class _SeededDraftLLM:
    """Supply a deliberately chosen draft, then use the real LLM for review."""

    def __init__(self, llm, draft: str) -> None:
        self.llm = llm
        self.draft = draft
        self.draft_sent = False

    async def stream_text(self, **kwargs):
        if not self.draft_sent:
            self.draft_sent = True
            yield json.dumps(
                {
                    "context_basis": "評価用の未確認案。",
                    "response_issues": [],
                    "response_intent": "評価用の未確認案を発話する。",
                    "response_example": self.draft,
                    "instruction_checks": [
                        {
                            "source": "configured_response_prompt",
                            "start_line": 1,
                            "end_line": 1,
                            "applicability": "applies",
                            "evidence": "評価用の未確認案。",
                            "force": "required",
                            "priority": "指定なし",
                            "response_excerpt": None,
                            "response_assessment": "確認段階で独立に判定する。",
                        }
                    ],
                },
                ensure_ascii=False,
            )
            return
        async for part in self.llm.stream_text(**kwargs):
            yield part


CLOSING_CASES = (
    {
        "id": "pending_after_thanks",
        "history": (
            ("assistant", "通知を止めたら、どうなりそうですか。"),
            ("person_b", "はい、ありがとうございます。"),
            ("assistant", "利用者Bさん、通知を止めたらどんな変化がありそうでしょう。"),
        ),
        "expected_reply_speaker_ids": ["person_b"],
    },
    {
        "id": "uncertainty_accepted",
        "history": (
            ("assistant", "通知を止めたらどうなりそうですか。"),
            ("person_b", "まだ分かりません。迷っています。"),
            ("assistant", "今はまだ分からないのですね。今日はここまでにしましょう。"),
        ),
        "expected_reply_speaker_ids": [],
    },
    {
        "id": "declined_answer_accepted",
        "history": (
            ("assistant", "通知を止めたらどうなりそうですか。"),
            ("person_b", "今は答えたくありません。"),
            ("assistant", "承知しました。今日はここで終わりにしましょう。"),
        ),
        "expected_reply_speaker_ids": [],
    },
    {
        "id": "quoted_question_is_not_pending",
        "history": (
            ("person_a", "通知が来ない理由が分かりました。"),
            (
                "assistant",
                "最初の『なぜ通知が来ないのか？』は設定が原因でしたね。今日はこれで終了です。",
            ),
        ),
        "expected_reply_speaker_ids": [],
    },
    {
        "id": "both_clients_pending",
        "history": (
            ("person_a", "設定の説明は分かりました。"),
            (
                "assistant",
                "終わる前に、お二人それぞれ、まだ確認しておきたい点を聞かせてください。",
            ),
        ),
        "expected_reply_speaker_ids": ["person_a", "person_b"],
    },
)


PROGRESS_CASES = (
    {
        "id": "stalled_unknown_with_stale_summary",
        "prompt": "対話の進行役として振る舞ってください。",
        "history": (
            (
                "person_a",
                "二人で旅行の予算を決めるとき、どう相談するか整理したいです。",
            ),
            ("assistant", "行き先の詳しい様子で、気になる点はありますか。"),
            ("person_b", "現地の様子はよく分かりません。"),
            ("assistant", "ほかに現地の様子で気になることはありますか。"),
            ("person_a", "私も分かりません。ほかには思い当たりません。"),
        ),
        "summary": "相談は予算についての話し合い方。次に意識する方針: 現地の様子を引き続き丁寧に確認する。未解決の問い: 現地の詳しい様子。",
        "near_end": False,
        "criteria": "現地の様子を再質問せず、終了せず、本人が求めた予算の相談の仕方へ戻る。",
    },
    {
        "id": "end_agreed_near_time",
        "prompt": "対話の進行役として振る舞ってください。",
        "history": (
            ("person_a", "予算を相談する手順はこれで分かりました。"),
            ("assistant", "では、今日はここで終えてよいでしょうか。"),
            ("person_a", "はい、今日はここまでで大丈夫です。"),
            ("person_b", "私も、今日はここまでで大丈夫です。"),
        ),
        "near_end": True,
        "criteria": "自発的終了要求は空、終了提案true、同意者A/B、回答待ちfalse。短く終結する。",
    },
    {
        "id": "end_only_one_client_agreed",
        "prompt": "対話の進行役として振る舞ってください。",
        "history": (
            ("assistant", "では、今日はここで終えてよいでしょうか。"),
            ("person_a", "はい、今日はここまでで大丈夫です。"),
        ),
        "near_end": True,
        "criteria": "同意者はAだけ。Bの終了意思を確認し、全員合意として終了しない。",
    },
    {
        "id": "end_explicit_request_before_time",
        "prompt": "対話の進行役として振る舞ってください。",
        "history": (
            ("assistant", "予算について今いちばん気になっていることは何ですか。"),
            ("person_b", "疲れたので、今日はここで面談を終わりにしたいです。"),
        ),
        "near_end": False,
        "criteria": "Bの面接全体の終了要求を認識し、時間前でも終結する。追加回答を強要しない。",
    },
    {
        "id": "end_thanks_is_not_consent",
        "prompt": "対話の進行役として振る舞ってください。",
        "history": (
            ("person_b", "どう予算を相談すればいいか、まだ迷っています。"),
            ("assistant", "予算についての相談を続けましょう。"),
            ("person_a", "はい、ありがとうございます。"),
        ),
        "near_end": True,
        "criteria": "感謝を面接終了の希望・同意にせず、予算についての相談を継続する。",
    },
    {
        "id": "summary_preserves_answered_unknown",
        "kind": "summary",
        "history": (
            (
                "person_a",
                "二人で旅行の予算を決めるとき、どう相談するか整理したいです。",
            ),
            ("assistant", "行き先の詳しい様子で、気になる点はありますか。"),
            ("person_b", "現地の様子は分かりません。ほかには思い当たりません。"),
        ),
        "summary": "未解決の問い: 現地の詳しい様子。次の方針: 現地の様子を繰り返し確認する。",
        "criteria": "現地の様子は本人にも分からないという回答を保持し、元の相談は予算の相談方法とする。再質問を指示しない。",
    },
)


async def evaluate(
    output: Path,
    timeout: float,
    case_id: str | None = None,
    *,
    closing: bool = False,
    progress: bool = False,
    clients: bool = False,
    late: bool = False,
    reasoning_effort: str | None = None,
) -> None:
    settings = apply_runtime_start_options(
        load_runtime_config(), {"realtime_api_centered_mode": False}
    )
    if reasoning_effort is not None:
        settings.openai.prompting_llm_reasoning_effort = reasoning_effort
    with TemporaryDirectory(prefix="prompt-director-evaluation-") as temporary:
        settings.paths.runtime_sessions_dir = temporary
        runtime = build_openai_runtime(settings, openai_client=object(), stt=object())
        director = runtime.prompt_director
        if director is None:
            raise RuntimeError("Prompt Director is disabled")
        report = {"route": runtime.ai_route_manifest["prompt_director"], "cases": []}
        output.parent.mkdir(parents=True, exist_ok=True)
        try:
            if case_id:
                selected_cases = (*CASES, *CLIENT_CASES, *LATE_CASES)
            elif late:
                selected_cases = LATE_CASES
            elif clients:
                selected_cases = CLIENT_CASES
            elif progress:
                selected_cases = PROGRESS_CASES
            elif closing:
                selected_cases = CLOSING_CASES
            else:
                selected_cases = CASES
            cases = [
                case
                for case in selected_cases
                if case_id is None or case["id"] == case_id
            ]
            for index, case in enumerate(cases, 1):
                print(f"[{index}/{len(cases)}] {case['id']} を確認中", flush=True)
                is_client = case.get("client", False) or (
                    case in CLIENT_CASES and case["speaker_id"] != "counselor"
                )
                attempts = []

                async def record_attempt(attempt):
                    attempts.append(attempt)

                history = tuple(
                    PublicHistoryMessage(*entry) for entry in case["history"]
                )
                request = (
                    ClosingAssessmentRequest(
                        counselor_speaker_id="assistant",
                        client_display_names={
                            "person_a": "利用者A",
                            "person_b": "利用者B",
                        },
                        public_history=history,
                    )
                    if closing
                    else PromptDirectorRequest(
                        turn_id=len(case["history"]),
                        speaker_id=case.get("speaker_id", "assistant"),
                        fixed_system_prompt=(
                            CLIENT_SYSTEM_PROMPT
                            if is_client
                            else "あなたは日本語の対話役です。自分の発話本文だけを返してください。"
                        ),
                        configured_prompt=case.get("prompt", ""),
                        speaker_profile=case.get("profile", ""),
                        shared_context=case.get("shared_context", ""),
                        client_peer_ids=(
                            tuple(
                                peer_id
                                for peer_id in ("person_a", "person_b")
                                if peer_id != case["speaker_id"]
                            )
                            if is_client
                            else ()
                        ),
                        public_history=history,
                        session_summary=case.get("summary", ""),
                        turn_specific_instructions=(
                            SESSION_TIME_INSTRUCTION
                            if progress
                            else case.get("turn_instruction", "")
                        ),
                        session_end_context=(
                            SessionEndContext(
                                client_ids=("person_a", "person_b"),
                                allow_agreed_end=case.get("near_end", False),
                            )
                            if progress
                            else None
                        ),
                    )
                )
                started = perf_counter()
                try:
                    async with asyncio.timeout(timeout):
                        if case.get("kind") == "summary":
                            summary = await summarize_session_context(
                                llm=runtime.session_summary_llm,
                                request=SessionSummaryRequest(
                                    existing_summary=case["summary"],
                                    turns_to_summarize=history,
                                    recent_turns=(),
                                    current_objective="設定プロンプトに沿って相談を続ける。",
                                    last_turn_id_to_summarize=len(history) - 1,
                                ),
                            )
                            result_data = {"summary": summary}
                        else:
                            case_director = (
                                PromptDirector(
                                    llm=_SeededDraftLLM(director.llm, case["draft"]),
                                    max_retries=director.max_retries,
                                )
                                if "draft" in case
                                else director
                            )
                            result = (
                                await director.assess_closing(request)
                                if closing
                                else await case_director.create_directive(
                                    request, on_attempt=record_attempt
                                )
                            )
                            result_data = result.model_dump()
                except (PromptDirectorError, TimeoutError) as exc:
                    report["cases"].append(
                        {"case": case, "error": str(exc), "attempts": attempts}
                    )
                    output.write_text(
                        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8",
                    )
                    print(str(exc), flush=True)
                    raise
                report["cases"].append(
                    {
                        "case": case,
                        "elapsed_seconds": round(perf_counter() - started, 3),
                        "result": result_data,
                        "attempts": attempts,
                    }
                )
                # Save each completed case so an interrupted run remains inspectable.
                output.write_text(
                    json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
                print(f"[{index}/{len(cases)}] 生成・形式検証完了", flush=True)
        finally:
            llms = [director.llm, runtime.session_summary_llm]
            for agent in runtime.controller.agents.values():
                llms.extend([agent.llm, agent.timing_llm])
            for client in {
                id(llm._client): llm._client for llm in llms if llm is not None
            }.values():
                await client.close()
    print(f"内容の評価用レポート: {output}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("workbench/prompt_director_checks/generic_contract.json"),
    )
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--reasoning-effort", choices=("none", "low", "medium", "high"))
    parser.add_argument(
        "--case", choices=[case["id"] for case in (*CASES, *CLIENT_CASES, *LATE_CASES)]
    )
    parser.add_argument(
        "--closing", action="store_true", help="終了判定の架空会話5件を確認する"
    )
    parser.add_argument(
        "--progress",
        action="store_true",
        help="停滞・早期終了・要約の架空会話を確認する",
    )
    parser.add_argument(
        "--late",
        action="store_true",
        help="終盤の重複・回答済み確認の架空会話4件を確認する",
    )
    parser.add_argument(
        "--clients",
        action="store_true",
        help="クライアントの復唱・同意・人物の区別を架空会話で確認する",
    )
    args = parser.parse_args()
    if sum((args.closing, args.progress, args.clients, args.late, bool(args.case))) > 1:
        parser.error(
            "--closing、--progress、--clients、--late、--case は併用できません"
        )
    try:
        asyncio.run(
            evaluate(
                args.output,
                args.timeout,
                args.case,
                closing=args.closing,
                progress=args.progress,
                clients=args.clients,
                late=args.late,
                reasoning_effort=args.reasoning_effort,
            )
        )
    except Exception as exc:
        print(f"検証を完了できませんでした: {type(exc).__name__}", flush=True)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
