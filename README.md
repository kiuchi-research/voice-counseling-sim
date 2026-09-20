# voice-counseling-sim

AI カウンセラーと AI クライアントによる、日本語の音声カウンセリング・シミュレーションです。Realtime API による音声対話をブラウザで聞きながら、発言履歴とセッションの状態を確認できます。研究・教育用のロールプレイを想定しています。

これは診断・治療を提供するシステムではありません。終了後の自動評価も、妥当性を検証した臨床評価ではなく参考情報です。

## できること

- AI カウンセラー × AI クライアントの自動対話。クライアントは 1 人または夫婦 2 人。
- 人間カウンセラー × AI クライアント、AI カウンセラー × 人間クライアントの音声対話。
- カウンセラー・クライアントのプリセット、声、接続先、クロージング開始時間の設定。
- セッションの開始・一時停止・再開・終了、発言履歴・音声・イベントのローカル保存。
- 音声・文字起こし・補助テキスト処理ごとの OpenAI / Azure OpenAI の選択。

会話の進行は Python の非同期ランタイムが担当し、Streamlit UI が Control API を介して操作します。既定の Realtime モードでは、カウンセラーの発言文を Prompt Director がテキストモデルで生成・確認して音声化します。クライアントは、プリセットやプロフィール、共有された会話履歴を踏まえて Realtime API が応答します。最近の発言と過去の要約を使って文脈を維持しますが、応答の重複や不自然な展開を完全に防ぐものではありません。

## セットアップ

Python 3.11 と conda を使用します。以下は Linux / WSL の bash 向けです。ネイティブ Windows / macOS での動作は、この公開版の準備時には確認していません。

リポジトリを clone または ZIP から展開し、そのディレクトリで実行してください。依存関係の正本は [`environment.yml`](environment.yml) です。`requirements.txt` は CI・互換用途です。

OpenAI Python SDK は、現在の実装・テストに合わせて 3 未満に制限しています。SDK のメジャーバージョンを変更する場合は、HTTP クライアントを含む互換性の確認が必要です。

```bash
cd voice-counseling-sim
conda env create -f environment.yml
conda activate py-voice-counseling
python utility/check_env.py
python -m pip install -e . --no-build-isolation
cp .env.example .env
```

同名の環境が既にある場合は、作成の代わりに `conda env update -n py-voice-counseling -f environment.yml` で更新します。環境名を変更する場合は `environment.yml` と有効化コマンドをそろえてください。

MP3 の結合・書き出しには別途 `ffmpeg` が必要です。Ubuntu / WSL では次のように導入できます。

```bash
sudo apt update
sudo apt install ffmpeg
python utility/check_ffmpeg.py
```

## API の設定

ご自身の API アカウントが必要です。実際の対話や接続確認では API 利用料金が発生します。モデル名の既定値は設定例であり、アカウント・リージョンで利用できるモデルやデプロイに合わせて変更してください。

**同梱設定の既定の接続先は Azure OpenAI です。** `.env` に OpenAI のキーを記入しただけでは接続先は変わりません。キーやエンドポイントは `.env` に設定し、コミットしないでください。

### Azure OpenAI を使う場合

[`.env.example`](.env.example) をコピーした `.env` に、利用する Azure リソースの値を記入します。

| キー | 設定する値 |
| --- | --- |
| `AZURE_OPENAI_ENDPOINT` | Azure OpenAI リソースのエンドポイント |
| `AZURE_OPENAI_API_KEY` | そのリソースの API キー |
| `AZURE_OPENAI_REALTIME_DEPLOYMENT` | Realtime 音声モデルのデプロイ名 |
| `AZURE_OPENAI_TEXT_DEPLOYMENT` | 補助テキスト処理に使う Responses API 対応モデルのデプロイ名 |
| `AZURE_OPENAI_STT_DEPLOYMENT` | 人間のマイク入力の文字起こしに使うデプロイ名 |

デプロイ名はモデル名と一致している必要はありません。Endpoint はリソース URL と `/openai/v1` 付き URL に対応します。`azure_eastus2` は設定内の接続プロファイル名で、実際の接続先は `.env` で決まります。

AI 同士の Realtime 対話は独立したマイク文字起こしを呼び出しません。すべての使用ルートが Azure であれば、このモードに `OPENAI_API_KEY` は不要です。

### OpenAI を使う場合

`.env` に `OPENAI_API_KEY` を記入し、必要に応じて `OPENAI_REALTIME_MODEL`、`OPENAI_STT_MODEL`、`OPENAI_PROMPTING_MODEL`、`OPENAI_TIMING_MODEL`、`OPENAI_SUMMARY_MODEL` を変更します。

UI の **Session setup** で、次の接続先を OpenAI に変更してからセッションを開始してください。

- Realtime 音声
- 人間のマイク文字起こし（人間が参加する場合）
- Prompting、ターンテイク、セッション要約

常に OpenAI を既定にする場合は、[`config/runtime_config.yaml`](config/runtime_config.yaml) の `ai.routes` 内の各 `provider` を `openai` に変更できます。非 Realtime モードで使用する `conversation_text` もここで設定します。`ai.default_provider` だけを変更しても、用途別に指定済みの接続先は変わりません。

旧テキスト生成・独立 TTS・モデレーション・評価には OpenAI 専用の経路も残っています。それらを利用する場合は、Realtime の接続先とは別に `OPENAI_API_KEY` が必要です。

`.env` を編集した後は、Control API と Streamlit の両方を再起動します。UI のモデル・接続先の変更は、次のセッションから反映されます。

## 起動と使い方

2 つのターミナルで同じ環境を有効化し、リポジトリのルートから起動します。

ターミナル 1 — Control API:

```bash
conda activate py-voice-counseling
python -m counseling_voice_demo.runtime.control_api config/runtime_config.yaml
```

ターミナル 2 — UI:

```bash
conda activate py-voice-counseling
python -m streamlit run app/streamlit_app.py --server.address 127.0.0.1 --server.port 8501
```

ブラウザで `http://127.0.0.1:8501` を開きます。

1. Realtime セッションの画面を開き、Session setup で参加モード、プリセット、接続先とモデルを確認します。
2. クロージング開始時間やターン数の上限を設定します。クロージング開始時間は、終了処理を始める目安です。厳密な終了時刻を保証するものではありません。
3. **Connect & Start audio** から開始します。人間が参加する場合はブラウザのマイク使用を許可します。
4. 必要に応じて Pause / Resume / Stop を使います。

AI同士の対話では、「先行生成上限（表示との差）」で指定した数の発話の音声生成が完了してから初回の再生を始めます。1ターンは1人の発話です。7なら7発話を待ち、24秒で待機を打ち切りません。指定数に達する前にセッションが終了した場合は、そこまでの音声を再生します。再生中は未再生の生成完了数が上限に達すると生成を一時停止し、上限-2以下で再開します（7なら5以下）。生成中の発話は数えず、生成速度によっては先行数がさらに少なくなるため、常に7発話を保つ設定ではありません。

カウンセラー応答の審査では、明確な履歴矛盾・話者混同・適用される必須/禁止指示の違反だけを必須修正として再生成します。言い回しなどの参考意見は内部に記録し、発話を作り直す理由にはしません。短い伝え返しや回答済みの終了確認を繰り返し要求しないようにしています。質問先本人の最後の発話と直前の文脈も明示し、別の参加者の発言で既回答を見落とさないようにします。分類はLLMによるため、誤判定の可能性は残ります。

カウンセラー応答の生成・審査中に一時的な通信エラーが起きた場合は、待ち時間を設けて既定で最大2回再試行します。回復しなければ会話履歴を保持して一時停止し、画面に案内を表示します。接続の回復後に **Resume** を押すと同じ未完了ターンから再生成できます。再試行中・一時停止中も **Stop** が使えます。認証失敗・権限不足・利用枠不足は自動再試行の対象外です。

この手順は同じ PC 上でのローカル利用向けです。ソースコードの GitHub 公開だけで、Web アプリがインターネット上に公開されることはありません。

## 保存データと取り扱い

音声・発言・プロフィールなどは、使用する機能に応じて選択した API の提供元に送信されます。まず架空の事例で動作を確認してください。

実行結果は `results/runtime_sessions/`、`results/sessions/` などに作成されます。`.env`、`results/`、`workbench/` は `.gitignore` の対象です。ログ内の `public` という名前は匿名化の保証ではありません。音声や会話ログを別途共有する場合は内容を確認してください。

このソース配布には過去の会話ログ・録音・開発用 ZIP・Git 履歴を含めていません。`app/assets/dev_probe_tone.mp3` は動作確認用に生成した合成音です。

## 開発と確認

API キーなしでの動作確認、テスト、構成の説明は [`docs/development.md`](docs/development.md) を参照してください。UI のデザイン規約は [`DESIGN.md`](DESIGN.md)、エージェント向けの作業指示は [`AGENTS.md`](AGENTS.md) にあります。

## ライセンス

[GNU General Public License v3.0](LICENSE)。
