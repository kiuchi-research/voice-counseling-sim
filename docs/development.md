# 開発と動作確認

## 構成

```text
app/                         Streamlit UI、音声モニター
src/counseling_voice_demo/    アプリケーション本体
src/counseling_voice_demo/runtime/
                             音声対話ランタイム、Control API、接続アダプター
config/                      設定、プロフィール、プリセット
tests/                       Python / JavaScript テスト
utility/                     環境確認・起動・接続確認などの補助ツール
results/                     実行時に作成するログと音声（Git 管理対象外）
```

依存関係の正本は `environment.yml` です。`requirements.txt` は CI 用に同じ依存を列挙します。パッケージ名は `counseling-voice-demo`、Python の import 名は `counseling_voice_demo` です。

## テスト

README のセットアップを完了し、リポジトリのルートで実行します。

```bash
python utility/check_env.py
python -m pytest -q
node --test tests/runtime_audio_monitor_playback.test.cjs tests/runtime_audio_monitor_input.test.cjs
```

JavaScript テストには Node.js が必要です。CI では Node.js 24 を使います。npm パッケージのインストールは不要です。

通常の自動テストは fake / mock を使い、実 API を呼び出しません。一部の統合テストはローカルポートで Control API を起動するため、ローカル通信が許可された環境で実行してください。テストの通過だけでは、各 API アカウントでのモデル利用可否や音声対話の品質までは確認できません。

CI は push / pull request / 手動実行で Python と JavaScript のテストを実行します。API キーの登録は不要です。使用する Actions の公式説明は [checkout](https://github.com/actions/checkout)、[setup-python](https://github.com/actions/setup-python)、[setup-node](https://github.com/actions/setup-node) を参照してください。

## API キーなしのランタイム確認

```bash
python -m counseling_voice_demo.runtime.cli --fake --turns 2 --sessions-dir workbench/fake-smoke
```

`--fake` を明示することで、外部 API を呼ばずに会話進行とログ保存を確認できます。音声モデルの品質評価にはなりません。

## 実 API の接続確認

次のツールは実際に API を呼び、利用料金が発生します。`.env` と `config/runtime_config.yaml` を設定したうえで、まず `--help` で対象・オプションを確認してください。

```bash
python utility/check_realtime_provider.py --help
python utility/check_text_provider.py --help
python utility/check_stt_provider.py --help
```

通常の pytest や CI がこれらを自動実行することはありません。

## 設定やプリセットを変更するとき

審査指摘は `response_issues` の各項目に `severity`、`category`、`reason`、`evidence` を持ちます。`must_fix` は明確な履歴矛盾・話者混同・今回適用される必須指示の不足・禁止指示の違反に限り、`advisory` は言い回し、任意の改善、不確かな疑義として記録だけにします。参考意見だけなら本文を保って通常の2回で完了し、再生成や一時停止を起こしません。再生成へ渡すフィードバックにも必須修正だけを含めます。短い伝え返しや、回答済みの終了確認を含まないことだけで差し戻さないよう指示します。分類はLLMが行い、コードで発話の語句を判定しません。分類・根拠の欠落や不正な形式を合格扱いする処理はありません。 内部ログには分類別の件数も残し、参考意見のみの場合は `prompt_director_advisory_recorded` を記録します。

審査時の `response_excerpt` は、JSON Schemaで固定された候補本文の全文または `null` に制約します。本文に対応する指示では全文を選び、対応する発話がない場合だけ `null` にします。対象箇所と適合・不適合は `response_assessment` に記します。本文と引用の文字列は共有の `_LockedResponseText` 定義を参照し、出典ごとのスキーマへ長い候補を複製しません。指示文や履歴を誤って引用する出力を生成前に防ぎ、受信後の本文一致・引用存在・原文行範囲の検証も継続します。生成段階は従来どおり正確な抜粋を使えます。通常のAPI呼び出し数は変わりませんが、全文引用により出力文字数が増える場合があります。

質問を作る・審査する際は、`context_basis` に質問先本人のID、求める情報についての既回答、今回確かめる未確認の具体点または再確認の理由を短く記します。直前に別の人が発言しても本人の既回答を未回答へ戻さず、自発的に語った内容も照合します。質問の言い回しではなく求める情報で重複を判断し、原文に反する再質問は必須修正として扱います。同じ話題の未確認の具体点、新しい事情による再確認、短い伝え返しは区別します。既存の生成・審査内で行い、追加のLLM呼び出しや文字列類似度による判定は入れません。

主な宛先IDがある場合は、その本人の最後の公開発話と直前の発話を `response_target_recent_statement` として入力末尾に再掲します。話者IDによる抽出だけを行い、前の発話が質問か、本人が回答済みかはコードで推定しません。全員の履歴は保持し、後続の訂正・事情変更も照合します。宛先未指定や本人の発話がまだない場合は再掲せず、別人の発話で埋めません。通常のLLM呼び出し数は変わりませんが、再掲分の入力は増えます。

`python utility/evaluate_prompt_director.py --repetition --timeout 120 --output workbench/prompt_director_checks/repetition.json` で、回答済みの再質問、言い換えによる再質問、未回答者への確認、未確認の具体点、事情変更後の再確認、伝え返し、初稿生成と別の話題での言い換え再質問、回答済み／未回答の優先順位、以前の回答の保持を架空会話11件で比較できます。うち1件は解釈が分かれる境界ケースとして `evaluation_kind=boundary` を付け、明確な反復の検出率に含めません。設定済みの実APIを使います。各ケースの `criteria` と実際の初稿・審査・最終本文を照合してください。形式検証の成功だけでは内容の合格を意味しません。

カウンセラー応答の生成・審査は、通信切断・タイムアウト・一時的なサーバー障害・レート制限に対して、失敗した段階を `runtime.prompt_director_max_retries`（既定2）回まで再試行します。SDK側の通信再試行とは別の上限です。`Retry-After` があれば指定時間を守り、なければ指数バックオフとジッターを使います。途中の応答は破棄し、同じプロンプト・履歴・審査対象の本文を保持します。上限到達時は `pause_reason=prompt_director_transport` で一時停止し、Resume で同じ未完了ターンをやり直します。認証・権限・利用枠不足・未知のAPIエラーは再試行しません。内部ログには失敗した段階、エラー種別、回数、待ち時間を記録します。

会話制御の実装とプリセットの内容は、別々に確認します。プロンプトを変える場合は目的を明確にし、該当テストに加えて架空の事例による音声対話で確認してください。発言が自然か、履歴を踏まえているか、終了が適切かは、自動テストだけでは判断できません。

会話ログや実アカウントの設定を、テスト用の fixture としてそのまま追加しないでください。`.gitignore` は未追跡ファイルの誤追加を防ぐ補助であり、既に追跡されたファイルや手動で強制追加したファイルの秘密情報を除去するものではありません。
