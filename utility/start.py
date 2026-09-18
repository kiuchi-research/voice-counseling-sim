#!/usr/bin/env python3
"""
プロジェクトセットアップスクリプト

environment.yml の `name:` を読み取り、ワークスペース内の
`.conda/envs/<name>` に conda 環境を作成または更新する。
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
ENV_FILE = ROOT_DIR / "environment.yml"
ENV_ROOT = ROOT_DIR / ".conda" / "envs"
CHECK_SCRIPT = ROOT_DIR / "utility" / "check_env.py"


def run_command(command: list[str], description: str, check: bool = True) -> bool:
    """コマンドを実行し、結果を表示する。"""
    print(f"🔄 {description}...")
    print(f"   実行コマンド: {' '.join(command)}")

    try:
        result = subprocess.run(
            command,
            cwd=ROOT_DIR,
            check=check,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        print(f"   ❌ {description}でエラーが発生: {exc}")
        if exc.stdout:
            print(f"   出力: {exc.stdout.strip()}")
        if exc.stderr:
            print(f"   エラー: {exc.stderr.strip()}")
        return False
    except Exception as exc:  # pragma: no cover - 予期しない保険
        print(f"   ❌ {description}で予期しないエラー: {exc}")
        return False

    if result.stdout:
        print(f"   出力: {result.stdout.strip()}")
    if result.stderr:
        label = "警告" if result.returncode == 0 else "エラー"
        print(f"   {label}: {result.stderr.strip()}")

    if result.returncode != 0:
        print(f"   ❌ {description}が終了コード {result.returncode} で失敗しました")
        return False

    print(f"   ✅ {description}完了")
    return True


def check_conda_installed() -> bool:
    """conda が利用可能かを確認する。"""
    return shutil.which("conda") is not None


def parse_env_name(env_file: Path) -> str:
    """environment.yml から環境名を取得する。"""
    for line in env_file.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("name:"):
            env_name = stripped.split(":", 1)[1].strip()
            if env_name:
                return env_name
            break
    raise RuntimeError("environment.yml から環境名を取得できませんでした")


def run_environment_check(env_path: Path) -> bool:
    """作成した conda 環境で環境チェックスクリプトを実行する。"""
    if not CHECK_SCRIPT.exists():
        print("⚠️ utility/check_env.py が見つかりません。スキップします。")
        return True

    return run_command(
        ["conda", "run", "-p", str(env_path), "python", str(CHECK_SCRIPT)],
        "環境チェック",
        check=False,
    )


def main() -> None:
    """メイン処理。"""
    print("🚀 プロジェクト起動スクリプトを開始します...")
    print("=" * 60)

    print("📋 ステップ 1: conda の確認")
    if not check_conda_installed():
        print("❌ エラー: conda がインストールされていません")
        print("   Anaconda または Miniconda をインストールしてください")
        print("   https://docs.conda.io/en/latest/miniconda.html")
        sys.exit(1)
    print("✅ conda がインストールされています")
    print()

    print("📋 ステップ 2: environment.yml の確認")
    if not ENV_FILE.exists():
        print(f"❌ エラー: {ENV_FILE.name} が見つかりません")
        sys.exit(1)

    try:
        env_name = parse_env_name(ENV_FILE)
    except RuntimeError as exc:
        print(f"❌ エラー: {exc}")
        sys.exit(1)

    ENV_ROOT.mkdir(parents=True, exist_ok=True)
    env_path = (ENV_ROOT / env_name).resolve()

    print(f"✅ {ENV_FILE.name} が存在します")
    print(f"✅ 環境名: {env_name}")
    print(f"✅ 作成先: {env_path}")
    print()

    print("📋 ステップ 3: conda 環境の確認・作成・更新")
    if not env_path.exists():
        print(f"環境 '{env_name}' が存在しません。ワークスペース内に作成します...")
        if not run_command(
            ["conda", "env", "create", "-p", str(env_path), "-f", str(ENV_FILE)],
            "environment.yml からの環境作成",
        ):
            sys.exit(1)
    else:
        print(f"✅ 環境 '{env_name}' が存在します")
        if not run_command(
            ["conda", "env", "update", "-p", str(env_path), "-f", str(ENV_FILE)],
            "environment.yml での環境更新",
        ):
            sys.exit(1)
    print()

    print("📋 ステップ 4: 環境のアクティベート")
    print(f"   実行してください: conda activate \"{env_path}\"")
    print()

    print("📋 ステップ 5: 環境チェック")
    if not run_environment_check(env_path):
        print("⚠️ 環境チェックで問題が検出されました")
        print(f"   手動確認: conda run -p \"{env_path}\" python utility/check_env.py")
    print()

    print("=" * 60)
    print("🎉 セットアップ完了！")
    print()
    print("📋 次のステップ:")
    print(f"1. conda activate \"{env_path}\"")
    print("2. python utility/check_env.py  # 環境チェック")
    print("3. プロジェクト固有のコマンドを実行")
    print()
    print("💡 ヒント:")
    print("   - 環境を非アクティブにする: conda deactivate")
    print(f"   - 環境を削除する: conda env remove -p \"{env_path}\"")
    print(f"   - 環境を更新する: conda env update -p \"{env_path}\" -f environment.yml")
    print("   - 利用可能な環境を確認: conda env list")


if __name__ == "__main__":
    main()
