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

会話制御の実装とプリセットの内容は、別々に確認します。プロンプトを変える場合は目的を明確にし、該当テストに加えて架空の事例による音声対話で確認してください。発言が自然か、履歴を踏まえているか、終了が適切かは、自動テストだけでは判断できません。

会話ログや実アカウントの設定を、テスト用の fixture としてそのまま追加しないでください。`.gitignore` は未追跡ファイルの誤追加を防ぐ補助であり、既に追跡されたファイルや手動で強制追加したファイルの秘密情報を除去するものではありません。
